# Basic boilerplate tutorial code to get camera reading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class RealSenseSubscriber(Node):
    def __init__(self):
        super().__init__('rs_sub')
        self.subscription = self.create_subscription(Image, '/camera/color/image_raw', self.listener_callback, 10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("RealSense RGB image", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            print(e)
            self.get_logger().error("Error converting image")


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseSubscriber()

    rclpy.spin(node)

    node.destroy_node() 
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
