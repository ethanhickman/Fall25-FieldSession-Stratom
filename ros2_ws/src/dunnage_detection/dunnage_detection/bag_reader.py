import os
import cv2
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String


DATA_PATH = '/app/data/'
TOPIC = "/camera/camera/color/image_raw"
IMAGE_NAME = "image.png"
BAG_FOLDER_NAME = "bag"


class SimpleBagReader(Node):

    def __init__(self):
        super().__init__('bag_reader')
        self.read_messages()


    def read_messages(self):
        bridge = CvBridge()
        reader = SequentialReader()

        for entry in os.listdir(DATA_PATH):
            bag_path = os.path.join(DATA_PATH, entry, BAG_FOLDER_NAME)
            if os.path.isdir(bag_path):

                storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
                converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
                reader.open(storage_options, converter_options)

                topic_types = reader.get_all_topics_and_types()
                type_dict = {t.name: t.type for t in topic_types}
                msg_type_str = type_dict.get(TOPIC)
                if not msg_type_str:
                    print(f"Topic {TOPIC} not found in bag: {bag_path}")
                    continue
                msg_type = get_message(msg_type_str)

                while reader.has_next():
                    topic, data, timestamp = reader.read_next()
                    if topic == TOPIC:
                        msg = rclpy.serialization.deserialize_message(data, msg_type)

                        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                        cv2.imwrite(os.path.join(bag_path, "../", IMAGE_NAME), cv_img)
                        print("Saved image")
                        break

                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    sbr = SimpleBagReader()
    rclpy.spin(sbr)

if __name__ == '__main__':
    main()
