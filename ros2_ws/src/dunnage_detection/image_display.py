import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import json

from dunnage_detection_interfaces.msg import DetectionOutput


class DetectionViewer(Node):
    def __init__(self):
        super().__init__('detection_viewer')

        # Subscribe to the DetectionOutput topic
        self.subscription = self.create_subscription(
            DetectionOutput,
            '/detection_output',          # <-- adjust if your topic name is different
            self.callback,
            10
        )

        self.bridge = CvBridge()
        self.get_logger().info("Detection Viewer started. Waiting for messages...")

    def callback(self, msg: DetectionOutput):
        try:
            # Convert ROS Image → OpenCV image
            cv_img = self.bridge.imgmsg_to_cv2(msg.color_image, desired_encoding='bgr8')

            # Parse bounding boxes from JSON
            bbox_list = json.loads(msg.bounding_box_json)

            # Draw bounding boxes
            for det in bbox_list:
                label = det["label"]
                x1, y1, x2, y2 = det["bbox_modal"]

                # Draw rectangle on the image
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put label above the box
                cv2.putText(
                    cv_img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Display the annotated image
            cv2.imshow("Detection Viewer", cv_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing DetectionOutput: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DetectionViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()