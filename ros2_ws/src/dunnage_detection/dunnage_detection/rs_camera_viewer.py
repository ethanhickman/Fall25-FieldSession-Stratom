import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class RealSenseViewer(Node):
    def __init__(self):
        super().__init__('rs_camera_viewer')

        self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        self.bridge = CvBridge()


    def color_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("RealSense RGB image", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            print(e)
            self.get_logger().error("Error converting image")

    def depth_callback(self, msg):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
            depth_normalized = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

            self.depth_image = depth_colored
            cv2.imshow('Depth Image', self.depth_image)
            cv2.waitKey(1)
        except Exception as e:
            print(e)
            self.get_logger().error("Error converting image")


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseViewer()

    rclpy.spin(node)

    node.destroy_node() 
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
