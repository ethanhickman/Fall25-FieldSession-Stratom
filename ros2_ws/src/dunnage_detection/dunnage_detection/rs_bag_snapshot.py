import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message

from sensor_msgs.msg import Image
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

import time
import threading


class BagRecorder(Node):
    class TopicData:
        def __init__(self, reader, topic_name, topic_type, sub_type):
            self.reader = reader
            self.topic_name = topic_name
            self.topic_type = topic_type
            self.sub_type = sub_type

            self.write_complete = True
            self.topic_info = TopicMetadata(
                name=topic_name,
                type=topic_type,
                serialization_format="cdr")

        def create_sub(self):
            self.sub = self.reader.create_subscription(
                self.sub_type,
                self.topic_name,
                self.generic_callback,
                10)

        def generic_callback(self, msg):
            self.reader.get_logger().info(self.topic_name + " callback started")
            if self.sub is not None:
                # destroy sub after first callback to only get one message
                self.reader.destroy_subscription(self.sub)
                self.sub = None

                # write message
                self.reader.writer.write(
                    self.topic_name,
                    serialize_message(msg),
                    self.reader.get_clock().now().nanoseconds)
                # signal write complete and attempt close of writer
                self.write_complete = True
                self.reader.try_close_writer()

    def __init__(self):
        super().__init__("rs_bag_snapshot")

        # create writer and flags
        self.writer = SequentialWriter()

        # create topic info
        self.topics = []
        self.topics.append(BagRecorder.TopicData(
            self,
            "/camera/camera/color/image_raw",
            "sensor_msgs/msg/Image",
            Image
        ))
        self.topics.append(BagRecorder.TopicData(
            self,
            "/camera/camera/aligned_depth_to_color/image_raw",
            "sensor_msgs/msg/Image",
            Image
        ))

        # threading and thread communication
        self.capturing_flag = False
        self.lock = threading.Lock()
        self.input_thread = threading.Thread(target=self.capture_input, daemon=True)
        self.input_thread.start()

    def capture_input(self):
        while rclpy.ok():
            user_in = input("Input: ")
            if user_in == "h":
                self.display_help()
            elif user_in == "c":
                self.set_capture_status(True)
                self.capture_frame()
                while self.get_capture_status():
                    time.sleep(0.1)
            elif user_in == "q":
                rclpy.shutdown()
                break

    def display_help(self):
        print("h: help message")
        print("c: capture")
        print("q: quit")

    def capture_frame(self):
        fn = "data/rs_frame_" + str(time.time())
        storage_options = StorageOptions(uri=fn, storage_id="sqlite3")
        converter_options = ConverterOptions("", "")

        # open writer and set flags
        self.writer.open(storage_options, converter_options)
        for t in self.topics:
            t.write_complete = False

        for t in self.topics:
            self.writer.create_topic(t.topic_info)

        self.get_logger().info("creating subs callback started")
        for t in self.topics:
            t.create_sub()

    def set_capture_status(self, status):
        with self.lock:
            self.capturing_flag = status

    def get_capture_status(self):
        with self.lock:
            return self.capturing_flag

    def try_close_writer(self):
        write_status = False
        for t in self.topics:
            if not t.write_complete:
                write_status = True
        if not write_status:
            self.get_logger().info("closing writer")
            self.writer.close()
            self.set_capture_status(False)


def main(args=None):
    rclpy.init(args=args)
    br = BagRecorder()
    rclpy.spin(br)
    if rclpy.ok():
        rclpy.shutdown()
