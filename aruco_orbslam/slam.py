import os
import cv2
import time
import rclpy
import orbslam3
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Path
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R  # pyrefly: ignore[untyped-import]

# change of basis from the OpenCV camera convention (x right, y down, z forward) to ROS (x forward, y left, z up)
CV_TO_ROS = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


class ScaledOrbNode(Node):
    def __init__(self):
        super().__init__('scaled_orb_node')
        self.bridge = CvBridge()

        # node params live in the same settings file orb-slam3 uses
        settings_file = os.path.join(get_package_share_directory('aruco_orbslam'), 'config', 'uffda.yaml')
        fs = cv2.FileStorage(settings_file, cv2.FILE_STORAGE_READ) # pyrefly: ignore[bad-argument-count]

        def real(key):
            return fs.getNode(key).real()

        def string(key):
            return fs.getNode(key).string()

        # camera calibration (full resolution)
        self.camera_matrix = np.array([
            [real('Camera.fx'), 0.0, real('Camera.cx')],
            [0.0, real('Camera.fy'), real('Camera.cy')],
            [0.0, 0.0, 1.0],
        ])
        self.dist_coeffs = np.array([real(k) for k in ('Camera.k1', 'Camera.k2', 'Camera.p1', 'Camera.p2', 'Camera.k3')])
        self.image_scale = real('Camera.imageScale') if not fs.getNode('Camera.imageScale').empty() else 1.0

        # slam world (CV convention) -> ROS world, including the camera's mounting rotation
        correction = R.from_euler('xyz', [real('Correction.roll'), real('Correction.pitch'), real('Correction.yaw')], degrees=True)
        self.world_to_ros = correction.as_matrix() @ CV_TO_ROS

        # point cloud
        self.pointcloud_min_observations = int(real('PointCloud.minObservations'))
        self.pointcloud_period = 1.0 / real('PointCloud.rate')
        self.last_pointcloud_time = 0.0

        # aruco detector and each tag's corners in its own frame, keyed by tag id
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, string('Aruco.dictionary'))) # pyrefly: ignore[missing-attribute]
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters()) # pyrefly: ignore[missing-attribute]
        self.tag_corners = {}
        tags = fs.getNode('Aruco.tags')
        for i in range(tags.size()):
            half = tags.at(i).getNode('size').real() / 2.0
            self.tag_corners[int(tags.at(i).getNode('id').real())] = np.array([
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ], dtype=np.float32)

        # orb-slam3
        self.slam = orbslam3.System(string('ORB.vocabulary'), settings_file, orbslam3.Sensor.MONOCULAR)
        self.slam.set_use_viewer(string('Viewer.UseViewer') == 'True')
        self.slam.initialize()

        # running mean and spread (welford) of metric sclae
        self.scale_min_samples = int(real('Scale.minSamples'))
        self.scale_tolerance = real('Scale.tolerance')
        self.map_start = None
        self.reset_scale()

        # pub sub blub blub
        self.frame_id = string('Topics.frameId')
        self.cam_sub = self.create_subscription(Image, string('Topics.video'), self.camera_cb, 1)
        self.trajectory_publisher = self.create_publisher(Path, string('Topics.trajectory'), 1)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, string('Topics.pointcloud'), 1)
        self.scale_publisher = self.create_publisher(Float32, string('Topics.scale'), 1)
        fs.release()

    def camera_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # update orbslam using downscaled image
        slam_frame = frame
        if self.image_scale != 1.0:
            slam_frame = cv2.resize(frame, None, fx=self.image_scale, fy=self.image_scale, interpolation=cv2.INTER_AREA) # pyrefly: ignore[bad-argument-type]
        self.slam.process_image_mono(slam_frame, time.monotonic())

        # get SLAM trajectory, cols = timestamp, 4x4 pose matrix
        trajectory = self.slam.get_trajectory_points()

        # update scale estimate
        self.update_scale(frame, trajectory)

        # publish SLAM pointcloud and trajectory
        header = Header(stamp=msg.header.stamp, frame_id=self.frame_id)
        origin_offset = self.publish_trajectory(header, trajectory)
        if origin_offset is not None:
            self.publish_pointcloud(header, origin_offset)

        # publish current scale estimate
        self.publish_scale()

    def reset_scale(self):
        self.scale = 1.0
        self.scale_samples = 0
        self.scale_m2 = 0.0  # sum of squared differences from the mean
        self.scale_converged = False

    def scale_relative_error(self):
        """Standard error of the scale mean as a fraction of the mean (inf until there are 2 samples)."""
        if self.scale_samples < 2:
            return float('inf')
        variance = self.scale_m2 / (self.scale_samples - 1)
        return np.sqrt(variance / self.scale_samples) / self.scale

    def update_scale(self, frame, trajectory):
        """update scale estimate using known aruco tag sizes"""

        # check if first keyframe has changed to identify orbslam restarts
        map_start = trajectory[0][0] if trajectory else None
        if map_start != self.map_start:
            if self.scale_samples:
                self.get_logger().info('slam map reset, restarting scale estimation')
            self.map_start = map_start
            self.reset_scale()
        if not trajectory or self.scale_converged:
            return

        # detect known markers
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return

        # calculate image mask where detected tags hold their estimated distance using known sizes
        distance_mask = np.zeros(frame.shape[:2], dtype=np.float32)
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            object_points = self.tag_corners.get(int(marker_id))
            if object_points is None:
                continue
            ok, _, tvec = cv2.solvePnP(object_points, marker_corners[0], self.camera_matrix, self.dist_coeffs)
            if ok:
                cv2.fillConvexPoly(distance_mask, marker_corners[0].astype(np.int32), float(np.linalg.norm(tvec))) # pyrefly: ignore[bad-argument-type]

        # get tracked features in the 2D input image
        features = np.array(self.slam.get_tracked_features(), dtype=np.float64)
        if features.size == 0:
            return
        height, width = distance_mask.shape
        xs = np.clip(np.round(features[:, 0] / self.image_scale).astype(int), 0, width - 1)
        ys = np.clip(np.round(features[:, 1] / self.image_scale).astype(int), 0, height - 1)

        # calculate ratio of aruco and orbslam distances for all features within aruco masks
        metric_distances = distance_mask[ys, xs]
        on_tag = metric_distances > 0
        if not on_tag.any():
            return
            
        camera_position = np.array(trajectory[-1][4::4])
        slam_distances = np.linalg.norm(features[on_tag, 2:5] - camera_position, axis=1)
        ratio = np.mean(metric_distances[on_tag] / slam_distances)
        if not np.isfinite(ratio):
            return

        self.scale_samples += 1
        delta = ratio - self.scale
        self.scale += delta / self.scale_samples
        self.scale_m2 += delta * (ratio - self.scale)

        error = self.scale_relative_error()
        if self.scale_samples >= self.scale_min_samples and error < self.scale_tolerance:
            self.scale_converged = True
            self.get_logger().info(f'scale converged to {self.scale:.4f} (±{100 * error:.2f}%) after {self.scale_samples} samples')

    def publish_trajectory(self, header, trajectory):
        """Publish orbslam map using current origin"""
        path_msg = Path(header=header)
        if not trajectory:
            self.trajectory_publisher.publish(path_msg)
            return None

        poses = np.asarray(trajectory, dtype=np.float64)[:, 1:].reshape(-1, 3, 4)
        # the world and camera axes are both in CV convention, so the change of basis applies on both sides
        rotations = self.world_to_ros @ poses[:, :, :3] @ CV_TO_ROS.T
        translations = self.scale * poses[:, :, 3] @ self.world_to_ros.T
        origin_offset = translations[0].copy()
        translations -= origin_offset
        quaternions = R.from_matrix(rotations).as_quat()  # x, y, z, w

        for (x, y, z), (qx, qy, qz, qw) in zip(translations, quaternions):
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
            pose.pose.orientation.x, pose.pose.orientation.y = qx, qy
            pose.pose.orientation.z, pose.pose.orientation.w = qz, qw
            path_msg.poses.append(pose)

        self.trajectory_publisher.publish(path_msg)
        return origin_offset

    def publish_pointcloud(self, header, origin_offset):
        """Publish the slam map aligned with current orbslam origin"""
        now = time.monotonic()
        if now - self.last_pointcloud_time < self.pointcloud_period:
            return
        self.last_pointcloud_time = now

        points = np.array(self.slam.get_map_points(self.pointcloud_min_observations), dtype=np.float64).reshape(-1, 3)
        points = self.scale * points @ self.world_to_ros.T - origin_offset
        self.pointcloud_publisher.publish(point_cloud2.create_cloud_xyz32(header, points.astype(np.float32)))

    def publish_scale(self):
        self.scale_publisher.publish(Float32(data=float(self.scale)))

def main(args=None):
    rclpy.init(args=args)
    node = ScaledOrbNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
