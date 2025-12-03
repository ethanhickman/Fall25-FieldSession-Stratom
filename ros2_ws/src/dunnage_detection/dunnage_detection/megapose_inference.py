# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from ament_index_python.packages import get_package_share_directory

from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
from dunnage_detection_interfaces.msg import DetectionOutput

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.plotting import gridplot
#from PIL import Image

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)


class MegaposeInference(Node):
    def __init__(self):
        super().__init__('megapose_inference')

        self.pkg_share_path = get_package_share_directory('dunnage_detection')
        self.ply_path = os.path.join(self.pkg_share_path, 'dunnage.ply')

        self.bridge = CvBridge()

        self.object_dataset = self.make_object_dataset(self.ply_path)
        self.pose_estimator = load_named_model("megapose-1.0-RGB-multi-hypothesis", self.object_dataset).cuda()

        self.pose_pub = self.create_publisher(PoseStamped, 'pose', 10)
        self.marker_pub = self.create_publisher(Marker, "model_marker", 10)
        self.reference_pub = self.create_publisher(Image, "reference_image", 10)

        self.sub = self.create_subscription(DetectionOutput, '/detection_output', self.detection_output_callback, 10)
        self.sub

    def detection_output_callback(self, msg):
        try:

            self.run_inference(msg.camera_info_json, msg.bounding_box_json, msg.color_image, msg.aligned_depth_image, self.ply_path, "megapose-1.0-RGB-multi-hypothesis")
            self.reference_pub.publish(msg.color_image)
            

        except Exception as e:
            print(e)
            self.get_logger().error("Error converting image")

# MEGAPOSE ---------------

    def load_observation(self, camera_data_str: str, color_msg: Image, depth_msg: Image, load_depth: bool = False) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
        camera_data = CameraData.from_json(camera_data_str)

        rgb_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        rgb = np.array(rgb_img, dtype=np.uint8)
        assert rgb.shape[:2] == camera_data.resolution

        depth = None
        if load_depth:
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            depth = np.array(depth_img, dtype=np.float32) / 1000
            assert depth.shape[:2] == camera_data.resolution

        return rgb, depth, camera_data


    def load_observation_tensor(self, camera_data_str: str, color_msg: Image, depth_msg: Image, load_depth: bool = False) -> ObservationTensor:
        rgb, depth, camera_data = self.load_observation(camera_data_str, color_msg, depth_msg, load_depth)
        observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
        return observation


    def load_object_data(self, obj_data_str: str) -> List[ObjectData]:
        object_data = json.loads(obj_data_str)
        object_data = [ObjectData.from_json(d) for d in object_data]
        return object_data


    def load_detections(self, obj_data_str: str) -> DetectionsType:
        input_object_data = self.load_object_data(obj_data_str)
        detections = make_detections_from_object_data(input_object_data).cuda()
        return detections


    def make_object_dataset(self, mesh_path: str) -> RigidObjectDataset:
        rigid_objects = []
        rigid_objects.append(RigidObject(label="dunnage", mesh_path=mesh_path, mesh_units="mm"))
        # TODO: fix mesh units
        rigid_object_dataset = RigidObjectDataset(rigid_objects)
        return rigid_object_dataset


    def save_predictions(self, pose_estimates: PoseEstimatesType) -> None:
        labels = pose_estimates.infos["label"]
        poses = pose_estimates.poses.cpu().numpy()
        object_data = [
            ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
        ]
        #object_data_json = json.dumps([x.to_json() for x in object_data])
        #print(object_data_json)
        for x in object_data:

            # pose data as pose stamped message
            pose_json = x.to_json()
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"

            pose_msg.pose.position.x = pose_json["TWO"][1][0]
            pose_msg.pose.position.y = pose_json["TWO"][1][1]
            pose_msg.pose.position.z = pose_json["TWO"][1][2]

            pose_msg.pose.orientation.x = pose_json["TWO"][0][0]
            pose_msg.pose.orientation.y = pose_json["TWO"][0][1]
            pose_msg.pose.orientation.z = pose_json["TWO"][0][2]
            pose_msg.pose.orientation.w = pose_json["TWO"][0][3]

            self.pose_pub.publish(pose_msg)

            # Marker so model is viewable in rviz
            marker = Marker()
            marker.header = pose_msg.header
            marker.ns = "model"
            marker.id = 0
            marker.type = Marker.MESH_RESOURCE
            marker.action = Marker.ADD

            marker.mesh_resource = "file://" + self.ply_path
            marker.mesh_use_embedded_materials = False
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            # Use pose from PoseStamped
            marker.pose = pose_msg.pose

            marker.scale.x = 0.005
            marker.scale.y = 0.005
            marker.scale.z = 0.005

            self.marker_pub.publish(marker)


        return


    def run_inference(self, camera_data_str: str, object_data_str: str, color_msg: Image, depth_msg: Image, mesh_path: str, model_name: str) -> None:

        model_info = NAMED_MODELS[model_name]

        observation = self.load_observation_tensor(camera_data_str, color_msg, depth_msg, load_depth=model_info["requires_depth"]).cuda()
        detections = self.load_detections(object_data_str).cuda()
        #object_dataset = self.make_object_dataset(mesh_path)

        #print(f"Loading model {model_name}")
        #pose_estimator = load_named_model(model_name, object_dataset).cuda()

        print(f"Running inference.")
        output, _ = self.pose_estimator.run_inference_pipeline(
            observation, detections=detections, **model_info["inference_parameters"]
        )

        self.save_predictions(output)
        return


# MEGAPOSE ---------------
def main(args=None):
    rclpy.init(args=args)
    node = MegaposeInference()

    rclpy.spin(node)

    node.destroy_node() 
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
