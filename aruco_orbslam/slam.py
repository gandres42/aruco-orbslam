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
from std_srvs.srv import Trigger
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
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

        # anchor tag: once seen, published poses and points are moved so this tag sits at its configured pose
        anchor_node = fs.getNode('Anchor.id')
        self.anchor_id = None if anchor_node.empty() else int(anchor_node.real())
        if self.anchor_id is not None and self.anchor_id < 0:
            self.anchor_id = None
        if self.anchor_id is not None and self.anchor_id not in self.tag_corners:
            self.get_logger().warn(f'anchor tag {self.anchor_id} has no size in Aruco.tags, anchoring disabled')
            self.anchor_id = None
        self.anchor_position = np.array([real('Anchor.px'), real('Anchor.py'), real('Anchor.pz')])
        self.anchor_rotation = R.from_euler('xyz', [real('Anchor.rx'), real('Anchor.ry'), real('Anchor.rz')], degrees=True).as_matrix()
        observations_node = fs.getNode('Anchor.observations')
        self.anchor_min_observations = 1 if observations_node.empty() else max(1, int(observations_node.real()))
        self.anchor_observations = None  # sightings the output is aligned to
        self.anchor_samples = []  # sightings being collected
        self.anchor_searching = self.anchor_id is not None

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
        def service_name(key, default):
            return default if fs.getNode(key).empty() else string(key)
        self.reanchor_service = self.create_service(Trigger, service_name('Services.reanchor', '/aruco/reanchor'), self.reanchor_cb)
        self.rescale_service = self.create_service(Trigger, service_name('Services.rescale', '/aruco/rescale'), self.rescale_cb)
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

        # the latest trajectory entry is only this frame's pose if this frame was localized
        localized = tracked and bool(trajectory) and trajectory[-1][0] == timestamp

        # detect known tags only while something still needs them
        detections = {}
        if trajectory and (not self.scale_converged or self.anchor_pending()):
            detections = self.detect_tags(frame)

        # update scale estimate and look for the anchor tag
        self.update_scale(frame, detections, trajectory)
        if localized:
            self.update_anchor(detections, trajectory[-1])

        # publish SLAM pointcloud and trajectory, once anchored if anchoring is enabled
        if self.anchor_id is None or self.anchor_observations is not None:
            header = Header(stamp=msg.header.stamp, frame_id=self.frame_id)
            alignment = self.publish_trajectory(header, trajectory)
            if alignment is not None:
                self.publish_pointcloud(header, *alignment)

        # publish current scale estimate
        self.publish_scale()

    def check_map_reset(self, trajectory):
        """detect orbslam restarts by a change of first frame, scale and anchor belong to the old map"""
        map_start = trajectory[0][0] if trajectory else None
        if map_start == self.map_start:
            return
        if self.scale_samples or self.anchor_observations or self.anchor_samples:
            self.get_logger().info('slam map reset, restarting scale estimation and anchor search')
        self.map_start = map_start
        self.reset_scale()
        self.anchor_observations = None
        self.anchor_samples = []
        self.anchor_searching = self.anchor_id is not None

    def reanchor_cb(self, request, response):
        """drop the current anchor and collect fresh sightings, path and map are not published until they are complete"""
        if self.anchor_id is None:
            response.success = False
            response.message = 'anchoring is disabled'
            return response
        self.anchor_observations = None
        self.anchor_samples = []
        self.anchor_searching = True
        response.success = True
        response.message = f'publishing paused, collecting {self.anchor_min_observations} new sightings of anchor tag {self.anchor_id}'
        self.get_logger().info(response.message)
        return response

    def rescale_cb(self, request, response):
        """restart scale estimation, the current scale stays in use until the first new sample"""
        self.reset_scale(keep_value=True)
        response.success = True
        response.message = f'restarting scale estimation from {self.scale:.4f}'
        self.get_logger().info(response.message)
        return response

    def detect_tags(self, frame):
        """pose of every visible known tag in the camera frame (CV convention, metres), keyed by tag id"""
        corners, ids, _ = self.detector.detectMarkers(frame)
        detections = {}
        if ids is None:
            return detections
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            object_points = self.tag_corners.get(int(marker_id))
            if object_points is None:
                continue
            ok, rvec, tvec = cv2.solvePnP(object_points, marker_corners[0], self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                detections[int(marker_id)] = (marker_corners[0], rvec.reshape(3), tvec.reshape(3))
        return detections

    def reset_scale(self, keep_value=False):
        # the first new sample replaces the mean, so keeping the old value only matters until then
        if not keep_value:
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

    def update_scale(self, frame, detections, trajectory):
        """update scale estimate using known aruco tag sizes"""
        if not trajectory or self.scale_converged or not detections:
            return

        # calculate image mask where detected tags hold their estimated distance using known sizes
        distance_mask = np.zeros(frame.shape[:2], dtype=np.float32)
        for marker_corners, _, tvec in detections.values():
            cv2.fillConvexPoly(distance_mask, marker_corners.astype(np.int32), float(np.linalg.norm(tvec))) # pyrefly: ignore[bad-argument-type]

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

    def anchor_pending(self):
        return self.anchor_searching

    def update_anchor(self, detections, current_pose):
        """collect the first Anchor.observations sightings of the anchor tag from localized frames"""
        if not self.anchor_pending() or self.anchor_id not in detections:
            return
        _, rvec, tvec = detections[self.anchor_id]
        # keep the frame's timestamp so its pose can be looked up again after bundle adjustment / loop closure,
        # and the tag offset in metres so the anchor follows the scale estimate as it converges
        self.anchor_samples.append({
            'timestamp': current_pose[0],
            'camera_pose': np.asarray(current_pose[1:], dtype=np.float64).reshape(3, 4),
            'tag_rotation': cv2.Rodrigues(rvec)[0] @ ARUCO_TO_ROS.T,
            'tag_translation': tvec.astype(np.float64),
        })
        if len(self.anchor_samples) >= self.anchor_min_observations:
            self.anchor_observations, self.anchor_samples = self.anchor_samples, []
            self.anchor_searching = False
            self.get_logger().info(f'anchor tag {self.anchor_id} averaged over {len(self.anchor_observations)} observations, aligning output to it')

    def anchor_tag_pose(self, timestamps, poses):
        """averaged anchor tag pose (rotation, position) in the scaled slam world"""
        rotations, positions = [], []
        for observation in self.anchor_observations:
            # camera pose when the anchor was seen, refined by orbslam if that frame is still in the trajectory
            match = np.flatnonzero(timestamps == observation['timestamp'])
            camera_pose = poses[match[-1]] if match.size else observation['camera_pose']
            camera_rotation, camera_position = camera_pose[:, :3], camera_pose[:, 3]
            rotations.append(camera_rotation @ observation['tag_rotation'])
            positions.append(self.scale * camera_position + camera_rotation @ observation['tag_translation'])
        return R.from_matrix(np.array(rotations)).mean().as_matrix(), np.mean(positions, axis=0)

    def alignment(self, timestamps, poses):
        """rotation and translation taking scaled slam world points to the published frame"""
        if self.anchor_observations is None:
            # no anchor yet: slam world axes, origin at the first frame
            return np.eye(3), -self.scale * poses[0, :, 3]

        # move the tag onto its configured pose: T_out = T_anchor * inv(T_tag)
        tag_rotation, tag_position = self.anchor_tag_pose(timestamps, poses)
        rotation = self.anchor_rotation @ tag_rotation.T
        return rotation, self.anchor_position - rotation @ tag_position

    def publish_trajectory(self, header, trajectory):
        """Publish orbslam trajectory in the published frame, returns that frame's alignment"""
        path_msg = Path(header=header)
        if not trajectory:
            self.trajectory_publisher.publish(path_msg)
            return None

        trajectory = np.asarray(trajectory, dtype=np.float64)
        timestamps = trajectory[:, 0]
        poses = trajectory[:, 1:].reshape(-1, 3, 4)
        world_rotation, world_translation = self.alignment(timestamps, poses)
        # align the world side, and switch the camera body axes from CV to ROS convention
        rotations = world_rotation @ poses[:, :, :3] @ CV_TO_ROS.T
        translations = self.scale * poses[:, :, 3] @ world_rotation.T + world_translation
        quaternions = R.from_matrix(rotations).as_quat()  # x, y, z, w

        for (x, y, z), (qx, qy, qz, qw) in zip(translations, quaternions):
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
            pose.pose.orientation.x, pose.pose.orientation.y = qx, qy
            pose.pose.orientation.z, pose.pose.orientation.w = qz, qw
            path_msg.poses.append(pose)

        self.trajectory_publisher.publish(path_msg)
        return world_rotation, world_translation

    def publish_pointcloud(self, header, world_rotation, world_translation):
        """Publish the slam map in the same frame as the trajectory"""
        now = time.monotonic()
        if now - self.last_pointcloud_time < self.pointcloud_period:
            return
        self.last_pointcloud_time = now

        points = np.array(self.slam.get_map_points(self.pointcloud_min_observations), dtype=np.float64).reshape(-1, 3)
        points = self.scale * points @ world_rotation.T + world_translation
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
