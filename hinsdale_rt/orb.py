import cv2  # ty: ignore[unresolved-import]
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
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

class OrbNode(Node):
    def __init__(self):
        super().__init__('orb_node')
        self.bridge = CvBridge()

        self.slam = orbslam2.System("/tmp/ORB_SLAM2/Vocabulary/ORBvoc.txt", "uffda.yaml", orbslam2.Sensor.MONOCULAR)
        self.slam.set_use_viewer(True)
        self.slam.initialize()

        self.cam_sub = self.create_subscription(Image, "/BlueROV2/video", self.camera_cb, 1)
        self.cam_pub = self.create_publisher(Image, "/orbslam/tracked_feats", 1)

        self.tracked_pixels_2d = np.empty((0, 2))
        self.tracked_points_3d = np.empty((0, 3))


    def camera_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.slam.process_image_mono(bgr, time.monotonic())

        features = np.array(self.slam.get_tracked_features(), dtype=np.float64)
        if features.size:
            self.tracked_pixels_2d = features[:, 0:2]
            self.tracked_points_3d = features[:, 2:5]
        else:
            self.tracked_pixels_2d = np.empty((0, 2))
            self.tracked_points_3d = np.empty((0, 3))

        annotated = bgr.copy()
        for x, y in self.tracked_pixels_2d:
            cv2.circle(annotated, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)

        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = msg.header
        self.cam_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OrbNode()
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
