import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message

from sensor_msgs.msg import Image, CameraInfo
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

import time
from datetime import datetime
import threading

import yaml


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
        self.out_folder = ""

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
        self.topics.append(BagRecorder.TopicData(
            self,
            "/camera/camera/color/camera_info",
            "sensor_msgs/msg/CameraInfo",
            CameraInfo
        ))
        self.topics.append(BagRecorder.TopicData(
            self,
            "/camera/camera/depth/image_rect_raw",
            "sensor_msgs/msg/Image",
            Image
        ))
        self.topics.append(BagRecorder.TopicData(
            self,
            "/camera/camera/depth/camera_info",
            "sensor_msgs/msg/CameraInfo",
            CameraInfo
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
                measurements = {}
                measurements["dunnage_voffset"] = self.record_measurement("Dunnage Height Relative to Camera [below camera is negative] (meters)")
                measurements["dunnage_hoffset"] = self.record_measurement("Dunnage Horizontal Position Relative to Camera [left of camera is negative] (meters)")
                measurements["dunnage_doffset"] = self.record_measurement("Dunnage Distance Relative to Camera (meters)")
                measurements["dunnage_angle"] = self.record_measurement("Dunnage Angle Relative to Camera (degrees)")
                with open(self.out_folder + "measurements.yaml", "w") as f:
                    yaml.dump(measurements, f)
            elif user_in == "q":
                rclpy.shutdown()
                break

    def record_measurement(self, name):
        while True:
            user_in = input("Enter the " + name + ": ")
            try:
                return float(user_in)
            except ValueError:
                print("Invalid measurment!")
        

    def display_help(self):
        print("h: help message")
        print("c: capture")
        print("q: quit")

    def capture_frame(self):
        #self.out_folder = "data/rs_frame_" + str(time.time()) + "/"
        self.out_folder = "data/rs_frame_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/"
        fn = self.out_folder + "bag"
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
