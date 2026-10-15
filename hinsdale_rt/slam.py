from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from scipy.spatial.transform import Rotation as R
import cv2 # pyrefly: ignore [missing-import]
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nav_msgs.msg import Path
from rclpy.node import Node
import numpy as np
import orbslam3
import rclpy
import time
import os

CV_TO_ROS = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])

class ScaledOrbNode(Node):
    def __init__(self):
        super().__init__('scaled_orb_node')
        self.bridge = CvBridge()

        # load node params from the same settings file orb-slam3 uses
        settings_file = os.path.join(get_package_share_directory('hinsdale_rt'), 'config', 'uffda.yaml')
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
        self.scale_window_size = int(fs.getNode('Scale.windowSize').real())
        self.frame_id = fs.getNode('Topics.frameId').string()

        # load orb-slam3
        self.slam = orbslam3.System(fs.getNode('ORB.vocabulary').string(), settings_file, orbslam3.Sensor.MONOCULAR)
        self.slam.set_use_viewer(fs.getNode('Viewer.UseViewer').string() == "True")
        self.slam.initialize()
        self.rolling_scale_window = []
        self.rolling_scale_estimate = 0
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

        # pub, sub, blub blub
        self.cam_sub = self.create_subscription(Image, fs.getNode('Topics.video').string(), self.camera_cb, 1)
        self.annotated_publisher = self.create_publisher(Image, fs.getNode('Topics.annotated').string(), 1)
        self.trajectory_publisher = self.create_publisher(Path, fs.getNode('Topics.trajectory').string(), 1)
        fs.release()


    def camera_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # ARUCO PROCESSING
        annotated_aruco = bgr.copy()
        corners, ids, _ = self.detector.detectMarkers(bgr)
        markers = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated_aruco, corners, ids)
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                ok, rvec, tvec = cv2.solvePnP(self.object_points[str(marker_id)], marker_corners[0], self.camera_matrix, self.dist_coeffs)
                if not ok:
                    continue
                
                # get distance to marker center
                distance = float(np.linalg.norm(tvec))

                # create marker mask, valid points have distance instead of 0
                mask = np.zeros(bgr.shape[:2], dtype=np.float32)
                cv2.fillConvexPoly(mask, marker_corners[0].astype(np.int32), distance)

                # annotate image to publish
                cv2.drawFrameAxes(annotated_aruco, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1 / 2.0)
                # corners are ordered top-left, top-right, bottom-right, bottom-left
                top_right = marker_corners[0][1]
                text_pos = (int(top_right[0]) + 5, int(top_right[1]) - 5)
                cv2.putText(annotated_aruco, f'{distance:.2f}m', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # save it all for later
                markers[str(marker_id)] = mask

        # ORB PROCESSING 
        self.slam.process_image_mono(bgr, time.monotonic())

        # get tracked orb features in 2D and 3D
        features = np.array(self.slam.get_tracked_features(), dtype=np.float64)
        if features.size:
            tracked_pixels_2d = features[:, 0:2]
            tracked_points_3d = features[:, 2:5]
        else:
            tracked_pixels_2d = np.empty((0, 2))
            tracked_points_3d = np.empty((0, 3))

        # current camera position is the translation of the latest trajectory pose
        trajectory = self.slam.get_trajectory_points()
        current_position = np.array(trajectory[-1][4::4]) if trajectory else None

        # orb-slam3 starts a new map (with its own scale) after losing track and later merges maps,
        # and the trajectory only covers the active map, so a new first pose means the old scale is stale
        map_start = trajectory[0][0] if trajectory else None
        if map_start != self.map_start:
            self.rolling_scale_window.clear()
            self.map_start = map_start

        # marker masks don't overlap, so taking the max keeps each marker's own distance value intact
        combined_mask = np.zeros(bgr.shape[:2], dtype=np.float32)
        for marker_mask in markers.values():
            combined_mask = np.maximum(combined_mask, marker_mask)

        xs = np.round(tracked_pixels_2d[:, 0]).astype(int)
        ys = np.round(tracked_pixels_2d[:, 1]).astype(int)
        metric_distances = combined_mask[ys, xs]
        marker_bool_mask = metric_distances != 0

        slam_distances = np.zeros(len(tracked_pixels_2d))
        if current_position is not None:
            slam_distances[marker_bool_mask] = np.linalg.norm(tracked_points_3d[marker_bool_mask] - current_position, axis=1)

        # get the average difference between metric distances and slam distances and print it
        if current_position is not None and marker_bool_mask.any():
            avg_ratio = np.mean(metric_distances[marker_bool_mask] / slam_distances[marker_bool_mask])
            self.rolling_scale_window.append(avg_ratio)
            if len(self.rolling_scale_window) > self.scale_window_size:
                self.rolling_scale_window.pop(0)
        self.rolling_scale_estimate = np.mean(self.rolling_scale_window)

        for x, y in tracked_pixels_2d[marker_bool_mask]:
            cv2.circle(annotated_aruco, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)

        # PUBLISH TRAJECTORY
        # get the current trajectory from orbslam and scale it by self.rolling_scale_estimate
        path_msg = Path()
        path_msg.header.stamp = msg.header.stamp
        path_msg.header.frame_id = self.frame_id
        origin_offset = None
        for _, r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz in trajectory:
            rotation_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
            translation = np.array([tx, ty, tz]) * self.rolling_scale_estimate

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

            qx, qy, qz, qw = Rotation.from_matrix(rotation_matrix).as_quat()

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

        # DEBUG IMAGE
        cv2.putText(annotated_aruco, f'{self.rolling_scale_estimate:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        annotated_out_msg = self.bridge.cv2_to_imgmsg(annotated_aruco, 'bgr8')
        annotated_out_msg.header = msg.header
        self.annotated_publisher.publish(annotated_out_msg)


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
