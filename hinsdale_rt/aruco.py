import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
import os

VIDEO_TOPIC = '/BlueROV2/video'
ANNOTATED_TOPIC = '/aruco/annotated_image'

CAMERA_MATRIX = np.array([
    [1078.17559, 0.0, 1010.57086],
    [0.0, 1076.46176, 463.06243],
    [0.0, 0.0, 1.0],
])
DIST_COEFFS = np.zeros(5)
TAG_SIZES = {
    "14": 0.15,
    "12": 0.20,
    "7":  0.10,
    "15": 0.15,
    "16": 0.075,
    "17": 0.075,
    "18": 0.05
}

class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_node')
        self.bridge = CvBridge()

        # ARUCO TAGS
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
        self.video_subscription = self.create_subscription(Image, VIDEO_TOPIC, self.camera_cb, 10)
        self.annotated_publisher = self.create_publisher(Image, ANNOTATED_TOPIC, 10)

    def camera_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        corners, ids, _ = self.detector.detectMarkers(bgr)

        markers = {}

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                ok, rvec, tvec = cv2.solvePnP(self.object_points[str(marker_id)], marker_corners[0], CAMERA_MATRIX, DIST_COEFFS)
                if not ok:
                    continue

                distance = float(np.linalg.norm(tvec))
                cv2.drawFrameAxes(bgr, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.1 / 2.0)
                center = marker_corners[0].mean(axis=0)
                cv2.putText(bgr, f'{distance:.2f}m', (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, marker_corners[0].astype(np.int32), 255)

                markers[str(marker_id)] = (distance, mask)

        out = self.bridge.cv2_to_imgmsg(bgr, 'bgr8')
        out.header = msg.header
        self.annotated_publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
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
