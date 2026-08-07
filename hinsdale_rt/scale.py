import cv2  # ty: ignore[unresolved-import]
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
import os
import orbslam2
import time

VIDEO_TOPIC = '/BlueROV2/video'
ANNOTATED_TOPIC = '/aruco/annotated_image'
CAMERA_MATRIX = np.array([
    [1078.17559, 0.0, 1010.57086],
    [0.0, 1076.46176, 463.06243],
    [0.0, 0.0, 1.0],
])
DIST_COEFFS = np.zeros(5)
# CV camera convention (x-right, y-down, z-forward) -> ROS convention (x-forward, y-left, z-up)
CV_TO_ROS = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])
TAG_SIZES = {
    "14": 0.15,
    "12": 0.20,
    "7":  0.10,
    "15": 0.15,
    "16": 0.075,
    "17": 0.075,
    "18": 0.05
}

class ScaledOrbNode(Node):
    def __init__(self):
        super().__init__('scaled_orb_node')
        self.bridge = CvBridge()

        # load orb-slam2
        self.slam = orbslam2.System("/tmp/ORB_SLAM2/Vocabulary/ORBvoc.txt", "uffda.yaml", orbslam2.Sensor.MONOCULAR)
        self.slam.set_use_viewer(True)
        self.slam.initialize()
        self.rolling_scale_window = []
        self.rolling_scale_estimate = 0

        # aruco params
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, 'DICT_6X6_50'))
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.object_points = {}
        for tag_id, tag_len in TAG_SIZES.items():
            half = tag_len / 2.0
            self.object_points[tag_id] = np.array([
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ], dtype=np.float32)

        # pub, sub, blub blub
        self.cam_sub = self.create_subscription(Image, "/BlueROV2/video", self.camera_cb, 1)
        self.annotated_publisher = self.create_publisher(Image, "/aruco/annotated", 1)
        self.trajectory_publisher = self.create_publisher(Path, "/reference_trajectory", 1)


    def camera_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # ARUCO PROCESSING
        annotated_aruco = bgr.copy()
        corners, ids, _ = self.detector.detectMarkers(bgr)
        markers = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated_aruco, corners, ids)
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                ok, rvec, tvec = cv2.solvePnP(self.object_points[str(marker_id)], marker_corners[0], CAMERA_MATRIX, DIST_COEFFS)
                if not ok:
                    continue
                
                # get distance to marker center
                distance = float(np.linalg.norm(tvec))

                # create marker mask, valid points have distance instead of 0
                mask = np.zeros(bgr.shape[:2], dtype=np.float32)
                cv2.fillConvexPoly(mask, marker_corners[0].astype(np.int32), distance)

                # annotate image to publish
                cv2.drawFrameAxes(annotated_aruco, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.1 / 2.0)
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
            if len(self.rolling_scale_window) > 200:
                self.rolling_scale_window.pop(0)
        self.rolling_scale_estimate = np.mean(self.rolling_scale_window)
        print(f'avg metric/slam distance diff: {self.rolling_scale_estimate:.2f}')


        for x, y in tracked_pixels_2d[marker_bool_mask]:
            cv2.circle(annotated_aruco, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)

        # PUBLISH TRAJECTORY
        # get the current trajectory from orbslam and scale it by self.rolling_scale_estimate
        path_msg = Path()
        path_msg.header.stamp = msg.header.stamp
        path_msg.header.frame_id = 'map'
        for _, r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz in trajectory:
            rotation_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
            translation = np.array([tx, ty, tz]) * self.rolling_scale_estimate

            # both the world and camera axes are in CV convention, so the same
            # change-of-basis rotation is applied on both sides of the pose
            translation = CV_TO_ROS @ translation
            rotation_matrix = CV_TO_ROS @ rotation_matrix @ CV_TO_ROS.T
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
