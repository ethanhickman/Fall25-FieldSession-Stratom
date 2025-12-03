import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from dunnage_detection_interfaces.msg import DetectionOutput

import numpy as np
import cv2
import os
import tempfile
import json

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

        self.image_publisher = self.create_publisher(Image, 'annotated_image', 10)
        self.output_publisher = self.create_publisher(DetectionOutput, 'detection_output', 10)
        self.bridge = CvBridge()

        # self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.color_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.info_sub = Subscriber(self, CameraInfo, '/camera/camera/color/camera_info')

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=10,
            slop=0.05
        )
        self.sync.registerCallback(self.sync_callback)

    def sync_callback(self, color, depth, info):
        try:
            
            # Darknet requires reading from files so write color image to file and then predict
            cv_image = self.bridge.imgmsg_to_cv2(color, desired_encoding='bgr8')
            with tempfile.NamedTemporaryFile(suffix=".jpg") as prediction_img:
                cv2.imwrite(prediction_img.name, cv_image)
                DarkHelp.PredictFN(self.dh, prediction_img.name.encode())

                # Write annotated image to another temporary file
                with tempfile.NamedTemporaryFile(suffix=".jpg") as annotated_img:
                    DarkHelp.Annotate(self.dh, annotated_img.name.encode())
                    annotated_cv_image = cv2.imread(annotated_img.name)

                    # Publish annotated message to be view in rviz
                    annotated_img_msg = self.bridge.cv2_to_imgmsg(annotated_cv_image, encoding="bgr8")
                    self.image_publisher.publish(annotated_img_msg)

                # store camera intrinsics as dictionary
                camera_data = {
                    "K": np.array(info.k).reshape(3, 3).tolist(),
                    "resolution": [info.height, info.width]
                }

                # load prediction results and publish output message for each object in image
                j = json.loads(DarkHelp.GetPredictionResults(self.dh))
                for prediction in j['file'][0]['prediction']:
                    data = [{
                        "label": prediction['name'],
                        "bbox_modal": [prediction['rect']['x'], prediction['rect']['y'], prediction['rect']['x'] + prediction['rect']['width'], prediction['rect']['y'] + prediction['rect']['height']]
                    }]
                    output_msg = DetectionOutput()
                    output_msg.color_image = color
                    output_msg.aligned_depth_image = depth
                    output_msg.camera_info_json = json.dumps(camera_data)
                    output_msg.bounding_box_json = json.dumps(data)

                    self.output_publisher.publish(output_msg)

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
