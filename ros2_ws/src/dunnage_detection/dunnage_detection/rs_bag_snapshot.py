import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions

import message_filters
import os
import time


class BagRecorder(Node):
    def __init__(self):
        super().__init__("bag_recorder")

        self.writer = SequentialWriter()
        self.storage_options = StorageOptions(uri="test_bag", storage_id="sqlite3")
        self.converter_options = ConverterOptions("", "")
        self.writer.open(self.storage_options, self.converter_options)

        self.color_flag = False
        self.depth_flag = False

        # General Idea:
        # Have loop in another thread that waits for input key
        # When key is pressed set flags to True
        # when callbacks occur set flags to False and write
        # will need locks for writer and flags (potentially read_write_lock for flags)
    
    def color_callback(self, msg):
        # lock color_flag
        if self.color_flag:
            self.color_flag = False
            # perform write
        # unlock color_flag

    def depth_callback(self, msg):
        # lock depth_flag
        if self.depth_flag:
            self.depth_flag = False
            # perform write
        # unlock depth_flag


def main(args=None):
    rclpy.init(args=args)
    br = BagRecorder()
    rclpy.spin(br)
    rclpy.shutdown()
