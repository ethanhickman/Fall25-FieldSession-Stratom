import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import numpy as np
import cv2
import os
import tempfile

from . import DarkHelp


class DarknetDetection(Node):
    def __init__(self):
        super().__init__('darknet_detection')

        self.pkg_share_path = get_package_share_directory('dunnage_detection')
        self.cfg_path = os.path.join(self.pkg_share_path, 'dunnage.cfg')
        self.weights_path = os.path.join(self.pkg_share_path, 'dunnage.weights')
        self.names_path = os.path.join(self.pkg_share_path, 'dunnage.names')
        print("Paths:")
        print(self.cfg_path)
        print(self.weights_path)
        print(self.names_path)

        print("DarkHelp v" + DarkHelp.DarkHelpVersion().decode())
        print("Darknet v" + DarkHelp.DarknetVersion().decode())

        self.dh = DarkHelp.CreateDarkHelpNN(self.cfg_path.encode(), self.weights_path.encode(), self.names_path.encode())
        DarkHelp.EnableAnnotationAutoHideLabels(self.dh, False)
        DarkHelp.EnableNamesIncludePercentage(self.dh, False)
        DarkHelp.EnableSnapping(self.dh, False)
        DarkHelp.EnableTiles(self.dh, False)
        DarkHelp.SetThreshold(self.dh, 0.45)
        DarkHelp.SetAnnotationLineThickness(self.dh, 1)

        self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, 10)

        self.bridge = CvBridge()


    def color_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            with tempfile.NamedTemporaryFile(suffix=".jpg") as prediction_img:
                cv2.imwrite(prediction_img.name, cv_image)
                DarkHelp.PredictFN(self.dh, prediction_img.name.encode())

                with tempfile.NamedTemporaryFile(suffix=".jpg") as annotated_img:
                    DarkHelp.Annotate(self.dh, annotated_img.name.encode())
                    annotated_cv_image = cv2.imread(annotated_img.name)

                    cv2.imshow("RealSense RGB image", annotated_cv_image)
                    cv2.waitKey(1)

        except Exception as e:
            print(e)
            self.get_logger().error("Error converting image")

def main(args=None):
    rclpy.init(args=args)
    node = DarknetDetection()

    rclpy.spin(node)

    node.destroy_node() 
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
