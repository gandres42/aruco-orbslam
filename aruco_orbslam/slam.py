import os
import time

import cv2
import numpy as np
import orbslam3
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R  # pyrefly: ignore[untyped-import]
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

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
        fs = cv2.FileStorage(settings_file, cv2.FILE_STORAGE_READ)

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
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, string('Aruco.dictionary')))
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
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

        # running mean of the metric / slam scale ratio, reset whenever slam starts a new map
        self.scale = 1.0
        self.scale_samples = 0
        self.map_start = None

        # ros interfaces
        self.frame_id = string('Topics.frameId')
        self.cam_sub = self.create_subscription(Image, string('Topics.video'), self.camera_cb, 1)
        self.trajectory_publisher = self.create_publisher(Path, string('Topics.trajectory'), 1)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, string('Topics.pointcloud'), 1)
        fs.release()

    def camera_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # orb-slam3 only scales its calibration by Camera.imageScale, it does not resize the frames itself
        slam_frame = frame
        if self.image_scale != 1.0:
            slam_frame = cv2.resize(frame, None, fx=self.image_scale, fy=self.image_scale, interpolation=cv2.INTER_AREA)
        self.slam.process_image_mono(slam_frame, time.monotonic())

        # each row: timestamp, then a row-major 3x4 [R | t] camera pose in the slam world
        trajectory = self.slam.get_trajectory_points()

        self.update_scale(frame, trajectory)
        header = Header(stamp=msg.header.stamp, frame_id=self.frame_id)
        origin_offset = self.publish_trajectory(header, trajectory)
        if origin_offset is not None:
            self.publish_pointcloud(header, origin_offset)

    def update_scale(self, frame, trajectory):
        """Refine the scale estimate by comparing aruco (metric) and slam distances to features on the tags."""
        # the first keyframe's timestamp identifies the map, so a change means slam restarted
        map_start = trajectory[0][0] if trajectory else None
        if map_start != self.map_start:
            self.map_start = map_start
            self.scale = 1.0
            self.scale_samples = 0
        if not trajectory:
            return

        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return

        # image mask holding each tag's metric distance from the camera over its pixels, 0 elsewhere
        distance_mask = np.zeros(frame.shape[:2], dtype=np.float32)
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            object_points = self.tag_corners.get(int(marker_id))
            if object_points is None:
                continue
            ok, _, tvec = cv2.solvePnP(object_points, marker_corners[0], self.camera_matrix, self.dist_coeffs)
            if ok:
                cv2.fillConvexPoly(distance_mask, marker_corners[0].astype(np.int32), float(np.linalg.norm(tvec)))

        # tracked features: [u, v, x, y, z], with pixels in the downsampled slam frame
        features = np.array(self.slam.get_tracked_features(), dtype=np.float64)
        if features.size == 0:
            return
        height, width = distance_mask.shape
        xs = np.clip(np.round(features[:, 0] / self.image_scale).astype(int), 0, width - 1)
        ys = np.clip(np.round(features[:, 1] / self.image_scale).astype(int), 0, height - 1)
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
        self.scale += (ratio - self.scale) / self.scale_samples

    def publish_trajectory(self, header, trajectory):
        """Publish the scaled trajectory in the ROS frame, shifted so it starts at the origin.

        Returns the shift applied (None for an empty trajectory) so the point cloud can match it.
        """
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
        """Publish the whole slam map, transformed exactly like the trajectory.

        The map is large, so it is republished at a lower rate than the camera. Until a tag
        has been seen the scale is 1 and the points have no metric size.
        """
        now = time.monotonic()
        if now - self.last_pointcloud_time < self.pointcloud_period:
            return
        self.last_pointcloud_time = now

        points = np.array(self.slam.get_map_points(self.pointcloud_min_observations), dtype=np.float64).reshape(-1, 3)
        points = self.scale * points @ self.world_to_ros.T - origin_offset
        self.pointcloud_publisher.publish(point_cloud2.create_cloud_xyz32(header, points.astype(np.float32)))


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
