from ament_index_python.packages import get_package_share_directory
from enum import Enum, auto
from rclpy.executors import ExternalShutdownException
from scipy.spatial.transform import Rotation as R # pyrefly: ignore[untyped-import]
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from nav_msgs.msg import Path
from rclpy.node import Node
import open3d as o3d
import numpy as np
import orbslam3
import rclpy
import time
import cv2
import os

CV_TO_ROS = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


class State(Enum):
    SINED = 0 # scale not recovered
    SEELED = 1
    DELIVERED = 2 # scale recovered


class ScaledOrbNode(Node):
    def __init__(self):
        super().__init__('scaled_orb_node')
        self.bridge = CvBridge()

        # load node params from the same settings file orb-slam3 uses
        settings_file = os.path.join(get_package_share_directory('aruco_orbslam'), 'config', 'uffda.yaml')
        fs = cv2.FileStorage(settings_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = np.array([
            [fs.getNode('Camera.fx').real(), 0.0, fs.getNode('Camera.cx').real()],
            [0.0, fs.getNode('Camera.fy').real(), fs.getNode('Camera.cy').real()],
            [0.0, 0.0, 1.0],
        ])
        self.dist_coeffs = np.array([fs.getNode(k).real() for k in ('Camera.k1', 'Camera.k2', 'Camera.p1', 'Camera.p2', 'Camera.k3')])
        tags = fs.getNode('Aruco.tags')
        tag_sizes = {str(int(tags.at(i).getNode('id').real())): tags.at(i).getNode('size').real() for i in range(tags.size())}
        self.camera_correction = np.eye(4)
        self.camera_correction[:3, :3] = R.from_euler('xyz', [
            fs.getNode('Correction.roll').real(),
            fs.getNode('Correction.pitch').real(),
            fs.getNode('Correction.yaw').real(),
        ], degrees=True).as_matrix()
        image_scale = fs.getNode('Camera.imageScale')
        self.image_scale = image_scale.real() if not image_scale.empty() else 1.0
        self.pointcloud_min_observations = int(fs.getNode('PointCloud.minObservations').real())
        self.pointcloud_period = 1.0 / fs.getNode('PointCloud.rate').real()
        self.last_pointcloud_time = 0.0
        self.frame_id = fs.getNode('Topics.frameId').string()

        # load orb-slam3
        self.slam = orbslam3.System(fs.getNode('ORB.vocabulary').string(), settings_file, orbslam3.Sensor.MONOCULAR)
        self.slam.set_use_viewer(fs.getNode('Viewer.UseViewer').string() == "True")
        self.slam.initialize()

        self.avg_estimate = 1
        self.avg_estimate_count = 0

        self.map_start = None

        # aruco params
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, fs.getNode('Aruco.dictionary').string()))
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.object_points = {}
        for tag_id, tag_len in tag_sizes.items():
            half = tag_len / 2.0
            self.object_points[tag_id] = np.array([
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ], dtype=np.float32)

        # scale recovery state
        self.scale_recovery_state: State = State.SINED

        # pub, sub, blub blub
        self.cam_sub = self.create_subscription(Image, fs.getNode('Topics.video').string(), self.camera_cb, 1)
        self.trajectory_publisher = self.create_publisher(Path, fs.getNode('Topics.trajectory').string(), 1)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, fs.getNode('Topics.pointcloud').string(), 1)
        fs.release()


    def camera_cb(self, msg: Image):
        orginal_resolution_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # ORB PROCESSING
        # orb-slam3 only scales its calibration by Camera.imageScale, it does not resize the frames
        downsampled_frame = orginal_resolution_frame
        if self.image_scale != 1.0:
            downsampled_frame = cv2.resize(orginal_resolution_frame, None, fx=self.image_scale, fy=self.image_scale, interpolation=cv2.INTER_AREA)
        self.slam.process_image_mono(downsampled_frame, time.monotonic())

        # SCALE RECOVERY ------------------------------------------------------
        # estimate monocular scale if not recovered yet
        if self.scale_recovery_state != State.DELIVERED:
            # IDENTIFY ARUCO TAGS
            corners, ids, _ = self.detector.detectMarkers(orginal_resolution_frame)
            markers = {}
            if ids is not None:
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    ok, rvec, tvec = cv2.solvePnP(self.object_points[str(marker_id)], marker_corners[0], self.camera_matrix, self.dist_coeffs)
                    if not ok:
                        continue
                    
                    # get distance to marker center
                    distance = float(np.linalg.norm(tvec))

                    # create marker mask, valid points have distance instead of 0
                    mask = np.zeros(orginal_resolution_frame.shape[:2], dtype=np.float32)
                    cv2.fillConvexPoly(mask, marker_corners[0].astype(np.int32), distance)

                    # save it all for later
                    markers[str(marker_id)] = mask

            # IDENTIFY TRACKED FEATURES OVERLAPPING TAGS
            # get tracked orb features in 2D and 3D
            features = np.array(self.slam.get_tracked_features(), dtype=np.float64)
            if features.size:
                # feature pixels are in the resized frame, scale them back up to index the full resolution marker masks
                tracked_pixels_2d = features[:, 0:2] / self.image_scale
                tracked_points_3d = features[:, 2:5]
            else:
                tracked_pixels_2d = np.empty((0, 2))
                tracked_points_3d = np.empty((0, 3))

            # current camera position is the translation of the latest trajectory pose
            trajectory = self.slam.get_trajectory_points()
            current_position = np.array(trajectory[-1][4::4]) if trajectory else None

            # reset starting pose and scale estimate if SLAM restarts
            map_start = trajectory[0][0] if trajectory else None
            if map_start != self.map_start:
                self.avg_estimate_count = 0
                self.avg_estimate = 1
                self.map_start = map_start

            combined_mask = np.zeros(orginal_resolution_frame.shape[:2], dtype=np.float32)
            for marker_mask in markers.values():
                combined_mask = np.maximum(combined_mask, marker_mask)

            xs = np.round(tracked_pixels_2d[:, 0]).astype(int)
            ys = np.round(tracked_pixels_2d[:, 1]).astype(int)
            metric_distances = combined_mask[ys, xs]
            marker_bool_mask = metric_distances != 0

            # COMPARE FEATURE DISTANCES
            slam_distances = np.zeros(len(tracked_pixels_2d))
            if current_position is not None:
                slam_distances[marker_bool_mask] = np.linalg.norm(tracked_points_3d[marker_bool_mask] - current_position, axis=1)

            # get the average difference between metric distances and slam distances and print it
            if current_position is not None and marker_bool_mask.any():
                avg_ratio = np.mean(metric_distances[marker_bool_mask] / slam_distances[marker_bool_mask])
                
                self.avg_estimate = (self.avg_estimate * self.avg_estimate_count + avg_ratio) / (self.avg_estimate_count + 1)
                self.avg_estimate_count += 1

                print(f"{avg_ratio} - {self.avg_estimate}")

        # PUBLISH TRAJECTORY --------------------------------------------------
        # get the current trajectory from orbslam and scale it by self.rolling_scale_estimate
        path_msg = Path()
        path_msg.header.stamp = msg.header.stamp
        path_msg.header.frame_id = self.frame_id
        origin_offset = None
        for _, r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz in trajectory:
            rotation_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
            translation = np.array([tx, ty, tz]) * self.avg_estimate

            # both the world and camera axes are in CV convention, so the same
            # change-of-basis rotation is applied on both sides of the pose
            translation = CV_TO_ROS @ translation
            rotation_matrix = CV_TO_ROS @ rotation_matrix @ CV_TO_ROS.T

            # correct for the camera's known mounting offset, in ROS frame
            pose_ros = np.eye(4)
            pose_ros[:3, :3] = rotation_matrix
            pose_ros[:3, 3] = translation
            pose_ros = self.camera_correction @ pose_ros
            rotation_matrix = pose_ros[:3, :3]
            translation = pose_ros[:3, 3]

            # shift the whole path so the first pose sits at the origin
            if origin_offset is None:
                origin_offset = translation.copy()
            translation = translation - origin_offset

            qx, qy, qz, qw = R.from_matrix(rotation_matrix).as_quat()

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = translation[0]
            pose.pose.position.y = translation[1]
            pose.pose.position.z = translation[2]
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path_msg.poses.append(pose)

        # publish the scaled trajectory as a ros2 path topic at /reference_trajectory
        self.trajectory_publisher.publish(path_msg)

        # PUBLISH POINT CLOUD
        # the map points are in the same slam world frame as the trajectory, so they share its scale and
        # origin shift. without a scale estimate yet (no marker seen) the points have no metric size.
        # this is the whole map rather than just the points tracked in this frame, so it is republished
        # at a lower rate than the camera
        now = time.monotonic()
        if origin_offset is not None and np.isfinite(self.avg_estimate) and now - self.last_pointcloud_time >= self.pointcloud_period:
            self.last_pointcloud_time = now
            map_points = np.array(self.slam.get_map_points(self.pointcloud_min_observations), dtype=np.float64).reshape(-1, 3)
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(map_points)
            
            # cloud outlier removal
            # cl, ind = cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
            # cl, ind = cloud.remove_radius_outlier(nb_points=16, radius=0.05)
            # cloud = cloud.select_by_index(ind)

            # the path scales translations about the slam world origin, so scale the points about it too
            cloud.scale(self.avg_estimate, center=np.zeros(3))

            # same change of basis, camera correction and origin shift that the path poses go through
            cv_to_ros = np.eye(4)
            cv_to_ros[:3, :3] = CV_TO_ROS
            world_to_ros = self.camera_correction @ cv_to_ros
            world_to_ros[:3, 3] -= origin_offset
            cloud.transform(world_to_ros)

            self.pointcloud_publisher.publish(point_cloud2.create_cloud_xyz32(path_msg.header, np.asarray(cloud.points, dtype=np.float32)))


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
