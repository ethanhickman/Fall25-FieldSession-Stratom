import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message

from sensor_msgs.msg import Image
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

import time
import threading


class BagRecorder(Node):
    def __init__(self):
        super().__init__("rs_bag_snapshot")

        # create writer and flags
        self.writer = SequentialWriter()
        self.writer_is_open = False

        # create topic info
        self.color_topic_info = TopicMetadata(
            name="/camera/camera/color/image_raw",
            type="sensor_msgs/msg/Image",
            serialization_format="cdr")
        self.depth_topic_info = TopicMetadata(
            name="/camera/camera/aligned_depth_to_color/image_raw",
            type="sensor_msgs/msg/Image",
            serialization_format="cdr")

        self.input_thread = threading.Thread(target=self.capture_input, daemon=True)
        self.input_thread.start()

    def capture_input(self):
        while rclpy.ok():
            user_in = input("Input: ")
            if user_in == "h":
                self.display_help()
            elif user_in == "c":
                self.capture_frame()
                time.sleep(3)  # Just sleep for 3 seconds (change this)
            elif user_in == "q":
                rclpy.shutdown()

    def display_help(self):
        print("Help msg")

    def create_color_sub(self):
        self.color_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.color_callback,
            10)

    def create_depth_sub(self):
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            10)

    def capture_frame(self):
        fn = "rs_frame_" + str(time.time())
        storage_options = StorageOptions(uri=fn, storage_id="sqlite3")
        converter_options = ConverterOptions("", "")

        self.writer.open(storage_options, converter_options)
        self.writer_is_open = True
        self.color_write_complete = False
        self.depth_write_complete = False

        self.writer.create_topic(self.color_topic_info)
        self.writer.create_topic(self.depth_topic_info)

        self.get_logger().info("creating subs callback started")
        self.create_color_sub()
        self.create_depth_sub()

    def color_callback(self, msg):
        self.get_logger().info("color callback started")
        if self.color_sub is not None:
            self.destroy_subscription(self.color_sub)
            self.color_sub = None

            self.writer.write(
                "/camera/camera/color/image_raw",
                serialize_message(msg),
                self.get_clock().now().nanoseconds)
            self.color_write_complete = True
            self.try_close_writer()

    def depth_callback(self, msg):
        self.get_logger().info("depth callback started")
        if self.depth_sub is not None:
            self.destroy_subscription(self.depth_sub)
            self.depth_sub = None

            self.writer.write(
                "/camera/camera/aligned_depth_to_color/image_raw",
                serialize_message(msg),
                self.get_clock().now().nanoseconds)
            self.depth_write_complete = True
            self.try_close_writer()

    def try_close_writer(self):
        if self.color_write_complete and self.depth_write_complete:
            self.get_logger().info("closing writer")
            self.writer.close()
            self.writer_is_open = False


def main(args=None):
    rclpy.init(args=args)
    br = BagRecorder()
    rclpy.spin(br)
    rclpy.shutdown()
