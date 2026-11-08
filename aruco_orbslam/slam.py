import os
import cv2
import time
import rclpy
import orbslam3
import numpy as np
from collections import deque
from rclpy.node import Node
from nav_msgs.msg import Path
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R  # pyrefly: ignore[untyped-import]

# change of basis from the OpenCV camera body convention (x right, y down, z forward) to ROS (x forward, y left, z up)
CV_TO_ROS = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])

# change of basis from the aruco tag convention (x right, y up the marker, z out of its face) to ROS
# (x out of the face, y to the tag's left, z up the marker)
ARUCO_TO_ROS = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

class ScaledOrbNode(Node):
    def __init__(self):
        super().__init__('scaled_orb_node')
        self.bridge = CvBridge()

        # node params live in the same settings file orb-slam3 uses
        settings_file = os.path.join(get_package_share_directory('aruco_orbslam'), 'config', 'uffda.yaml')
        fs = cv2.FileStorage(settings_file, cv2.FILE_STORAGE_READ) # pyrefly: ignore[bad-argument-count]

        def real(key, default=0.0) -> float:
            node = fs.getNode(key)
            return default if node.empty() else node.real()

        def string(key) -> str:
            return fs.getNode(key).string()

        # camera calibration (full resolution)
        self.camera_matrix = np.array([
            [real('Camera.fx'), 0.0, real('Camera.cx')],
            [0.0, real('Camera.fy'), real('Camera.cy')],
            [0.0, 0.0, 1.0],
        ])
        self.dist_coeffs = np.array([real(k) for k in ('Camera.k1', 'Camera.k2', 'Camera.p1', 'Camera.p2', 'Camera.k3')])
        self.image_scale = real('Camera.imageScale', 1.0)

        # point cloud
        self.pointcloud_min_observations = int(real('PointCloud.minObservations'))
        self.pointcloud_period = 1.0 / real('PointCloud.rate')
        self.last_pointcloud_time = 0.0

        # aruco detector and each tag's corners in its own frame, keyed by tag id
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, string('Aruco.dictionary'))) # pyrefly: ignore[missing-attribute]
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters()) # pyrefly: ignore[missing-attribute]
        self.tag_corners = {}
        tags = fs.getNode('Aruco.tags')
        for tag in (tags.at(i) for i in range(tags.size())):
            half = tag.getNode('size').real() / 2.0
            self.tag_corners[int(tag.getNode('id').real())] = half * np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=np.float32)

        # anchor tag: published poses and points are moved so it sits at its configured pose, averaged over a
        # rolling history of sightings and published once there are enough of them
        self.anchor_id = int(real('Anchor.id', -1))
        if self.anchor_id not in self.tag_corners:
            if self.anchor_id >= 0:
                self.get_logger().warning(f'anchor tag {self.anchor_id} has no size in Aruco.tags, anchoring disabled')
            self.anchor_id = None
        self.anchor_position = np.array([real('Anchor.px'), real('Anchor.py'), real('Anchor.pz')])
        self.anchor_rotation = R.from_euler('xyz', [real('Anchor.rx'), real('Anchor.ry'), real('Anchor.rz')], degrees=True).as_matrix()
        self.anchor_min_observations = max(1, int(real('Anchor.observations')))
        self.anchor_samples = deque(maxlen=max(self.anchor_min_observations, int(real('Anchor.history'))))

        # metric scale is the median of a rolling history of samples
        self.scale = 1.0
        self.scale_samples = deque(maxlen=int(real('Scale.history')))
        self.map_start = None

        # orb-slam3
        self.slam = orbslam3.System(string('ORB.vocabulary'), settings_file, orbslam3.Sensor.MONOCULAR)
        self.slam.set_use_viewer(string('Viewer.UseViewer') == 'True')
        self.slam.initialize()

        # pub sub blub blub
        self.frame_id = string('Topics.frameId')
        self.cam_sub = self.create_subscription(Image, string('Topics.video'), self.camera_cb, 1)
        self.trajectory_publisher = self.create_publisher(Path, string('Topics.trajectory'), 1)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, string('Topics.pointcloud'), 1)
        self.scale_publisher = self.create_publisher(Float32, string('Topics.scale'), 1)
        self.reanchor_service = self.create_service(Trigger, string('Services.reanchor'), self.reanchor_cb)
        self.rescale_service = self.create_service(Trigger, string('Services.rescale'), self.rescale_cb)
        fs.release()

    def camera_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # update orbslam using downscaled image
        slam_frame = frame
        if self.image_scale != 1.0:
            slam_frame = cv2.resize(frame, None, fx=self.image_scale, fy=self.image_scale, interpolation=cv2.INTER_AREA) # pyrefly: ignore[bad-argument-type]
        timestamp = time.monotonic()
        tracked = self.slam.process_image_mono(slam_frame, timestamp)

        # get SLAM trajectory, cols = timestamp, 3x4 camera-to-world pose matrix
        trajectory = self.slam.get_trajectory_points()
        self.check_map_reset(trajectory)

        # sample scale and anchor from frames that were localized (so the latest trajectory entry is their pose)
        if tracked and trajectory and trajectory[-1][0] == timestamp:
            detections = self.detect_tags(frame)
            if detections:
                self.update_scale(frame, detections, trajectory[-1])
                self.update_anchor(detections, trajectory[-1])

        # publish SLAM trajectory and pointcloud, once anchored if anchoring is enabled
        if trajectory and (self.anchor_id is None or self.anchored()):
            trajectory = np.asarray(trajectory, dtype=np.float64)
            timestamps, poses = trajectory[:, 0], trajectory[:, 1:].reshape(-1, 3, 4)
            rotation, translation = self.alignment(timestamps, poses)
            header = Header(stamp=msg.header.stamp, frame_id=self.frame_id)
            self.publish_trajectory(header, poses, rotation, translation)
            self.publish_pointcloud(header, rotation, translation)

        self.scale_publisher.publish(Float32(data=self.scale))

    def check_map_reset(self, trajectory):
        """detect orbslam restarts by a change of first frame, scale and anchor belong to the old map"""
        map_start = trajectory[0][0] if trajectory else None
        if map_start == self.map_start:
            return
        if self.scale_samples or self.anchor_samples:
            self.get_logger().info('slam map reset, restarting scale estimation and anchor search')
        self.map_start = map_start
        self.scale = 1.0
        self.scale_samples.clear()
        self.anchor_samples.clear()

    def reanchor_cb(self, request, response):
        """drop the anchor history, path and map are not published until there are enough new sightings"""
        if self.anchor_id is None:
            response.success = False
            response.message = 'anchoring is disabled'
            return response
        self.anchor_samples.clear()
        response.success = True
        response.message = f'publishing paused, collecting {self.anchor_min_observations} new sightings of anchor tag {self.anchor_id}'
        self.get_logger().info(response.message)
        return response

    def rescale_cb(self, request, response):
        """drop the scale history, the current scale stays in use until the first new sample"""
        self.scale_samples.clear()
        response.success = True
        response.message = f'restarting scale estimation from {self.scale:.4f}'
        self.get_logger().info(response.message)
        return response

    def detect_tags(self, frame):
        """pose of every visible known tag in the camera frame (CV convention, metres), keyed by tag id"""
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return {}
        detections = {}
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            object_points = self.tag_corners.get(int(marker_id))
            if object_points is None:
                continue
            ok, rvec, tvec = cv2.solvePnP(object_points, marker_corners[0], self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                detections[int(marker_id)] = (marker_corners[0], rvec.reshape(3), tvec.reshape(3))
        return detections

    def update_scale(self, frame, detections, current_pose):
        """add a scale sample from known aruco tag sizes, scale is the median of the recent samples"""
        # calculate image mask where detected tags hold their estimated distance using known sizes
        distance_mask = np.zeros(frame.shape[:2], dtype=np.float32)
        for marker_corners, _, tvec in detections.values():
            cv2.fillConvexPoly(distance_mask, marker_corners.astype(np.int32), float(np.linalg.norm(tvec))) # pyrefly: ignore[bad-argument-type]

        # get tracked features (u, v, x, y, z) in the 2D input image
        features = np.array(self.slam.get_tracked_features(), dtype=np.float64).reshape(-1, 5)
        height, width = distance_mask.shape
        xs = np.clip(np.round(features[:, 0] / self.image_scale).astype(int), 0, width - 1)
        ys = np.clip(np.round(features[:, 1] / self.image_scale).astype(int), 0, height - 1)

        # calculate ratio of aruco and orbslam distances for all features within aruco masks
        metric_distances = distance_mask[ys, xs]
        on_tag = metric_distances > 0
        if not on_tag.any():
            return
        camera_position = np.array(current_pose[4::4])
        slam_distances = np.linalg.norm(features[on_tag, 2:5] - camera_position, axis=1)
        ratio = np.mean(metric_distances[on_tag] / slam_distances)
        if np.isfinite(ratio):
            self.scale_samples.append(ratio)
            self.scale = float(np.median(self.scale_samples))

    def anchored(self):
        return self.anchor_id is not None and len(self.anchor_samples) >= self.anchor_min_observations

    def update_anchor(self, detections, current_pose):
        """add a sighting of the anchor tag, the oldest one drops out once the history is full"""
        if self.anchor_id not in detections:
            return
        was_anchored = self.anchored()
        _, rvec, tvec = detections[self.anchor_id]
        # keep the frame's timestamp so its pose can be looked up again after bundle adjustment / loop closure,
        # and the tag offset in metres so the anchor follows the scale estimate
        self.anchor_samples.append((
            current_pose[0],
            np.reshape(current_pose[1:], (3, 4)),
            cv2.Rodrigues(rvec)[0] @ ARUCO_TO_ROS.T,
            tvec,
        ))
        if not was_anchored and self.anchored():
            self.get_logger().info(f'anchor tag {self.anchor_id} seen {self.anchor_min_observations} times, aligning output to it')

    def alignment(self, timestamps, poses):
        """rotation and translation taking scaled slam world points to the published frame"""
        if not self.anchored():
            # slam world axes, origin at the first frame
            return np.eye(3), -self.scale * poses[0, :, 3]

        # camera pose at each sighting, refined by orbslam where that frame is still in the (time ordered) trajectory
        sample_times, camera_poses, tag_rotations, tag_translations = (np.array(column) for column in zip(*self.anchor_samples))
        index = np.minimum(np.searchsorted(timestamps, sample_times), len(timestamps) - 1)
        found = timestamps[index] == sample_times
        camera_poses[found] = poses[index[found]]
        camera_rotations, camera_positions = camera_poses[:, :, :3], camera_poses[:, :, 3]

        # average anchor tag pose in the scaled slam world
        tag_rotation = R.from_matrix(camera_rotations @ tag_rotations).mean().as_matrix()
        tag_position = np.mean(self.scale * camera_positions + np.einsum('nij,nj->ni', camera_rotations, tag_translations), axis=0)

        # move the tag onto its configured pose: T_out = T_anchor * inv(T_tag)
        rotation = self.anchor_rotation @ tag_rotation.T
        return rotation, self.anchor_position - rotation @ tag_position

    def publish_trajectory(self, header, poses, rotation, translation):
        """Publish the camera trajectory in the published frame, with camera body axes in ROS convention"""
        positions = self.scale * poses[:, :, 3] @ rotation.T + translation
        quaternions = R.from_matrix(rotation @ poses[:, :, :3] @ CV_TO_ROS.T).as_quat()  # x, y, z, w
        self.trajectory_publisher.publish(Path(header=header, poses=[
            PoseStamped(header=header, pose=Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(x=qx, y=qy, z=qz, w=qw)))
            for (x, y, z), (qx, qy, qz, qw) in zip(positions.tolist(), quaternions.tolist())
        ]))

    def publish_pointcloud(self, header, rotation, translation):
        """Publish the slam map in the same frame as the trajectory"""
        now = time.monotonic()
        if now - self.last_pointcloud_time < self.pointcloud_period:
            return
        self.last_pointcloud_time = now

        points = np.array(self.slam.get_map_points(self.pointcloud_min_observations), dtype=np.float64).reshape(-1, 3)
        points = self.scale * points @ rotation.T + translation
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
