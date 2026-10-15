// Monocular ORB-SLAM3 whose trajectory is scaled to metric units by comparing the
// SLAM distance to features on ArUco tags against the distance from the tags' known sizes.

#include <chrono>
#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <ORB_SLAM3/System.h>
#include <ament_index_cpp/get_package_share_path.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

namespace
{

// ORB_SLAM3::System keeps its tracker and atlas private, but the full per-frame trajectory
// only lives there. Explicit template instantiations are exempt from access checks
// ([temp.explicit]), so instantiating PrivateMember with a private member's address defines
// member_pointer(Tag), which hands that address out.
template<typename Tag, typename Tag::type Member>
struct PrivateMember
{
  friend typename Tag::type member_pointer(Tag) {return Member;}
};

struct SystemTracker
{
  using type = ORB_SLAM3::Tracking * ORB_SLAM3::System::*;
  friend type member_pointer(SystemTracker);
};
template struct PrivateMember<SystemTracker, &ORB_SLAM3::System::mpTracker>;

struct SystemAtlas
{
  using type = ORB_SLAM3::Atlas * ORB_SLAM3::System::*;
  friend type member_pointer(SystemAtlas);
};
template struct PrivateMember<SystemAtlas, &ORB_SLAM3::System::mpAtlas>;

const Eigen::Matrix3d CV_TO_ROS = (Eigen::Matrix3d() <<
  0.0, 0.0, 1.0,
  -1.0, 0.0, 0.0,
  0.0, -1.0, 0.0).finished();

const cv::Scalar GREEN(0, 255, 0);

struct TrajectoryPose
{
  double timestamp;
  Eigen::Isometry3d camera_to_world;
};

struct TrackedFeature
{
  cv::Point2f pixel;  // undistorted
  Eigen::Vector3d point;  // in the map's world frame
};

// Full per-frame camera trajectory of the active map, rebuilt the way ORB_SLAM3's
// System::SaveTrajectoryTUM does: each frame's pose is stored relative to its reference
// keyframe, which bundle adjustment and loop closure keep refining. Frames that were not
// localized, or that are anchored in another map, are skipped.
std::vector<TrajectoryPose> get_trajectory(ORB_SLAM3::System & slam)
{
  ORB_SLAM3::Tracking * tracker = slam.*member_pointer(SystemTracker());
  ORB_SLAM3::Atlas * atlas = slam.*member_pointer(SystemAtlas());
  ORB_SLAM3::Map * active_map = atlas->GetCurrentMap();

  std::vector<TrajectoryPose> trajectory;
  auto reference = tracker->mlpReferences.begin();
  auto frame_time = tracker->mlFrameTimes.begin();
  auto lost = tracker->mlbLost.begin();
  for (const Sophus::SE3f & frame_to_reference : tracker->mlRelativeFramePoses) {
    ORB_SLAM3::KeyFrame * keyframe = *reference++;
    const double timestamp = *frame_time++;
    if (*lost++) {
      continue;
    }

    // if the reference keyframe was culled, walk up the spanning tree to a good one
    Sophus::SE3f reference_to_world;
    while (keyframe && keyframe->isBad()) {
      reference_to_world = reference_to_world * keyframe->mTcp;
      keyframe = keyframe->GetParent();
    }
    if (!keyframe || keyframe->GetMap() != active_map) {
      continue;
    }
    reference_to_world = reference_to_world * keyframe->GetPose();

    const Sophus::SE3f world_to_camera = frame_to_reference * reference_to_world;
    trajectory.push_back(
      {timestamp, Eigen::Isometry3d(world_to_camera.inverse().matrix().cast<double>())});
  }
  return trajectory;
}

// Keypoints of the current frame that have a triangulated map point. ORB-SLAM3 returns the
// keypoints and map points as index-aligned vectors, with null or bad map points for
// keypoints that have no landmark yet.
std::vector<TrackedFeature> get_tracked_features(ORB_SLAM3::System & slam)
{
  const std::vector<cv::KeyPoint> keypoints = slam.GetTrackedKeyPointsUn();
  const std::vector<ORB_SLAM3::MapPoint *> map_points = slam.GetTrackedMapPoints();

  std::vector<TrackedFeature> features;
  for (std::size_t i = 0; i < keypoints.size() && i < map_points.size(); ++i) {
    if (map_points[i] && !map_points[i]->isBad()) {
      features.push_back({keypoints[i].pt, map_points[i]->GetWorldPos().cast<double>()});
    }
  }
  return features;
}

cv::aruco::PredefinedDictionaryType dictionary_from_name(const std::string & name)
{
  static const std::unordered_map<std::string, cv::aruco::PredefinedDictionaryType> dictionaries{
    {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
    {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
    {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
    {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
    {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
    {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
    {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
    {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
    {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
    {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
    {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
    {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
    {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
    {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
    {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
    {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
    {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
    {"DICT_APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
    {"DICT_APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
    {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
    {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11},
    {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12},
  };
  return dictionaries.at(name);
}

// like numpy, the mean of no values is nan (0.0 / 0)
template<typename Container>
double mean(const Container & values)
{
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

}  // namespace

class ScaledOrbNode : public rclcpp::Node
{
public:
  ScaledOrbNode()
  : Node("scaled_orb_node")
  {
    // load node params from the same settings file orb-slam3 uses
    const std::string settings_file =
      (ament_index_cpp::get_package_share_path("aruco_orbslam") / "config" / "uffda.yaml").string();
    cv::FileStorage fs(settings_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      throw std::runtime_error("could not open settings file " + settings_file);
    }
    camera_matrix_ = (cv::Mat_<double>(3, 3) <<
      fs["Camera.fx"].real(), 0.0, fs["Camera.cx"].real(),
      0.0, fs["Camera.fy"].real(), fs["Camera.cy"].real(),
      0.0, 0.0, 1.0);
    dist_coeffs_ = (cv::Mat_<double>(5, 1) <<
      fs["Camera.k1"].real(), fs["Camera.k2"].real(), fs["Camera.p1"].real(),
      fs["Camera.p2"].real(), fs["Camera.k3"].real());
    const double deg = M_PI / 180.0;
    camera_correction_ = Eigen::Isometry3d::Identity();
    camera_correction_.linear() =
      (Eigen::AngleAxisd(fs["Correction.yaw"].real() * deg, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(fs["Correction.pitch"].real() * deg, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(fs["Correction.roll"].real() * deg, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
    scale_window_size_ = static_cast<std::size_t>(fs["Scale.windowSize"].real());
    frame_id_ = fs["Topics.frameId"].string();

    // load orb-slam3
    slam_ = std::make_unique<ORB_SLAM3::System>(
      fs["ORB.vocabulary"].string(), settings_file, ORB_SLAM3::System::MONOCULAR,
      fs["Viewer.UseViewer"].string() == "True");

    // aruco params
    detector_.setDictionary(
      cv::aruco::getPredefinedDictionary(dictionary_from_name(fs["Aruco.dictionary"].string())));
    for (const cv::FileNode tag : fs["Aruco.tags"]) {
      const float half = static_cast<float>(tag["size"].real() / 2.0);
      object_points_[static_cast<int>(tag["id"].real())] = {
        {-half, half, 0.0f},
        {half, half, 0.0f},
        {half, -half, 0.0f},
        {-half, -half, 0.0f},
      };
    }

    // pub, sub, blub blub
    cam_sub_ = create_subscription<sensor_msgs::msg::Image>(
      fs["Topics.video"].string(), 1,
      [this](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {camera_cb(msg);});
    annotated_publisher_ = create_publisher<sensor_msgs::msg::Image>(
      fs["Topics.annotated"].string(), 1);
    trajectory_publisher_ = create_publisher<nav_msgs::msg::Path>(
      fs["Topics.trajectory"].string(), 1);
  }

private:
  void camera_cb(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    const cv_bridge::CvImageConstPtr cv_image = cv_bridge::toCvShare(msg, "bgr8");
    const cv::Mat & bgr = cv_image->image;

    // ARUCO PROCESSING
    // pixels inside each marker hold the metric distance to it, all others are 0
    cv::Mat annotated_aruco = bgr.clone();
    cv::Mat marker_distances = cv::Mat::zeros(bgr.size(), CV_32F);
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detector_.detectMarkers(bgr, corners, ids);
    cv::aruco::drawDetectedMarkers(annotated_aruco, corners, ids);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      const auto object_points = object_points_.find(ids[i]);
      if (object_points == object_points_.end()) {
        continue;  // the settings give no size for this tag
      }
      cv::Mat rvec, tvec;
      if (!cv::solvePnP(
          object_points->second, corners[i], camera_matrix_, dist_coeffs_, rvec, tvec))
      {
        continue;
      }
      const double distance = cv::norm(tvec);

      std::vector<cv::Point> polygon;
      for (const cv::Point2f & corner : corners[i]) {
        polygon.emplace_back(static_cast<int>(corner.x), static_cast<int>(corner.y));
      }
      cv::fillConvexPoly(marker_distances, polygon, cv::Scalar(distance));

      // annotate image to publish
      cv::drawFrameAxes(annotated_aruco, camera_matrix_, dist_coeffs_, rvec, tvec, 0.1 / 2.0);
      // corners are ordered top-left, top-right, bottom-right, bottom-left
      const cv::Point2f & top_right = corners[i][1];
      cv::putText(
        annotated_aruco, cv::format("%.2fm", distance),
        cv::Point(static_cast<int>(top_right.x) + 5, static_cast<int>(top_right.y) - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv::LINE_AA);
    }

    // ORB PROCESSING
    const double now =
      std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    slam_->TrackMonocular(bgr, now);
    const std::vector<TrackedFeature> features = get_tracked_features(*slam_);
    const std::vector<TrajectoryPose> trajectory = get_trajectory(*slam_);

    // orb-slam3 starts a new map (with its own scale) after losing track and later merges maps,
    // and the trajectory only covers the active map, so a new first pose means the old scale is stale
    std::optional<double> map_start;
    if (!trajectory.empty()) {
      map_start = trajectory.front().timestamp;
    }
    if (map_start != map_start_) {
      rolling_scale_window_.clear();
      map_start_ = map_start;
    }

    // SCALE ESTIMATION
    // for tracked features on a marker, compare the metric distance to the marker with the
    // slam distance from the current camera position (the latest trajectory pose)
    std::vector<double> ratios;
    for (const TrackedFeature & feature : features) {
      const cv::Point pixel(cvRound(feature.pixel.x), cvRound(feature.pixel.y));
      if (!cv::Rect(cv::Point(), marker_distances.size()).contains(pixel)) {
        continue;
      }
      const float metric_distance = marker_distances.at<float>(pixel);
      if (metric_distance == 0) {
        continue;
      }
      cv::circle(annotated_aruco, pixel, 3, GREEN, -1);
      if (!trajectory.empty()) {
        const Eigen::Vector3d camera_position = trajectory.back().camera_to_world.translation();
        ratios.push_back(metric_distance / (feature.point - camera_position).norm());
      }
    }
    if (!ratios.empty()) {
      rolling_scale_window_.push_back(mean(ratios));
      if (rolling_scale_window_.size() > scale_window_size_) {
        rolling_scale_window_.pop_front();
      }
    }
    const double scale = mean(rolling_scale_window_);

    // PUBLISH TRAJECTORY
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = msg->header.stamp;
    path_msg.header.frame_id = frame_id_;
    Eigen::Vector3d origin_offset;
    for (const TrajectoryPose & trajectory_pose : trajectory) {
      // both the world and camera axes are in CV convention, so the same
      // change-of-basis rotation is applied on both sides of the pose
      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      pose.linear() = CV_TO_ROS * trajectory_pose.camera_to_world.linear() * CV_TO_ROS.transpose();
      pose.translation() = CV_TO_ROS * trajectory_pose.camera_to_world.translation() * scale;

      // correct for the camera's known mounting offset, in ROS frame
      pose = camera_correction_ * pose;

      // shift the whole path so the first pose sits at the origin
      if (path_msg.poses.empty()) {
        origin_offset = pose.translation();
      }
      pose.translation() -= origin_offset;

      geometry_msgs::msg::PoseStamped pose_msg;
      pose_msg.header = path_msg.header;
      pose_msg.pose = tf2::toMsg(pose);
      path_msg.poses.push_back(pose_msg);
    }
    trajectory_publisher_->publish(path_msg);

    // DEBUG IMAGE
    cv::putText(
      annotated_aruco, cv::format("%.2f", scale), cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv::LINE_AA);
    annotated_publisher_->publish(
      *cv_bridge::CvImage(msg->header, "bgr8", annotated_aruco).toImageMsg());
  }

  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  Eigen::Isometry3d camera_correction_;
  std::size_t scale_window_size_;
  std::string frame_id_;

  std::unique_ptr<ORB_SLAM3::System> slam_;
  std::deque<double> rolling_scale_window_;
  std::optional<double> map_start_;

  cv::aruco::ArucoDetector detector_;
  std::map<int, std::vector<cv::Point3f>> object_points_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectory_publisher_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ScaledOrbNode>());
  rclcpp::shutdown();
  return 0;
}
