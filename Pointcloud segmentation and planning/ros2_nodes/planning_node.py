"""
ROS2 node that subscribes to a segmented point cloud, builds a 2D occupancy
grid, plans with A*, and publishes nav_msgs/Path. Optional CPU-only example.
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import OccupancyGrid, Path as PathMsg
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError:
    rclpy = None

from src.planning import GridSpec, astar, build_grid


def cloud_to_numpy(msg: PointCloud2):
    pts = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z", "label"), skip_nans=True):
        pts.append([p[0], p[1], p[2], p[3]])
    if not pts:
        return None, None
    arr = np.array(pts, dtype=np.float32)
    return arr[:, :2], arr[:, 3].astype(np.int64)


class PlanningNode(Node):
    def __init__(self, spec: GridSpec):
        super().__init__("grid_planner")
        self.spec = spec
        self.sub = self.create_subscription(PointCloud2, "segmented_cloud", self.cb, 1)
        self.grid_pub = self.create_publisher(OccupancyGrid, "grid", 1)
        self.path_pub = self.create_publisher(PathMsg, "a_star_path", 1)
        self.get_logger().info("Planning node ready")

    def cb(self, msg: PointCloud2):
        xy, labels = cloud_to_numpy(msg)
        if xy is None:
            return
        grid, origin, res = build_grid(xy, labels, self.spec)
        start = (grid.shape[0] // 2, 1)
        goal = (grid.shape[0] // 2, grid.shape[1] - 2)
        path = astar(grid, start, goal)
        self.publish_grid(grid, origin, res, msg.header.frame_id)
        self.publish_path(path, origin, res, msg.header.frame_id)

    def publish_grid(self, grid, origin, res, frame):
        msg = OccupancyGrid()
        msg.header.frame_id = frame
        msg.info.resolution = res
        msg.info.width = grid.shape[1]
        msg.info.height = grid.shape[0]
        msg.info.origin.position.x = origin[0]
        msg.info.origin.position.y = origin[1]
        msg.data = (grid.flatten() * 100).tolist()
        self.grid_pub.publish(msg)

    def publish_path(self, path, origin, res, frame):
        msg = PathMsg()
        msg.header.frame_id = frame
        for (i, j) in path:
            pose = PoseStamped()
            pose.header.frame_id = frame
            pose.pose.position.x = origin[0] + i * res
            pose.pose.position.y = origin[1] + j * res
            msg.poses.append(pose)
        self.path_pub.publish(msg)


def main():
    if rclpy is None:
        print("rclpy/sensor_msgs_py/nav_msgs not installed. Skipping node.")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--inflate", type=int, default=1)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    spec = GridSpec(resolution=args.resolution, inflate=args.inflate)
    node = PlanningNode(spec)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

