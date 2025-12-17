"""
Lightweight ROS2 node that wraps the CPU RandLA-Net model.
Launch only if ROS2 + sensor_msgs_py are installed.

Usage:
    ros2 run <pkg> segmentation_node --ros-args -p model_path:=outputs/model.pt -p tile:=L001.ply
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError:
    rclpy = None

import torch

from src.data_io import normalize_features
from src.model import RandLANetSmall


def cloud_to_numpy(msg: PointCloud2):
    pts = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z", "intensity", "r", "g", "b"), skip_nans=True):
        pts.append([p[0], p[1], p[2], p[3], p[4], p[5], p[6]])
    if len(pts) == 0:
        return None, None
    arr = np.array(pts, dtype=np.float32)
    coords = arr[:, :3]
    colors = arr[:, 4:7]
    intensity = arr[:, 3]
    feats = normalize_features(colors, intensity)
    return coords, feats


def numpy_to_cloud(points: np.ndarray, labels: np.ndarray, header):
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="label", offset=12, datatype=PointField.UINT32, count=1),
    ]
    data = [(*p, int(l)) for p, l in zip(points, labels)]
    return pc2.create_cloud(header, fields, data)


class SegmentationNode(Node):
    def __init__(self, model_path: Path):
        super().__init__("randlanet_seg")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RandLANetSmall(num_classes=9, feat_channels=4).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device
        self.sub = self.create_subscription(PointCloud2, "input_cloud", self.cb, 1)
        self.pub = self.create_publisher(PointCloud2, "segmented_cloud", 1)
        self.get_logger().info(f"Segmentation node loaded model {model_path}")

    def cb(self, msg: PointCloud2):
        pts, feats = cloud_to_numpy(msg)
        if pts is None:
            return
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(pts[None]).float().to(self.device),
                torch.from_numpy(feats[None]).float().to(self.device),
            )
            pred = logits.argmax(dim=-1).cpu().numpy().squeeze()
        out = numpy_to_cloud(pts, pred, msg.header)
        self.pub.publish(out)


def main():
    if rclpy is None:
        print("rclpy/sensor_msgs_py not installed. Skipping node.")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/model.pt")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = SegmentationNode(Path(args.model_path))
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

