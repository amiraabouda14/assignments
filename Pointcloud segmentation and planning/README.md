# Pointcloud segmentation and planning

Overview

The goal of this assignment is to extract semantic information from a 3D point cloud and exploit it for object-level reasoning and navigation planning in an urban environment.

The pipeline consists of three main stages:

*Semantic segmentation of the 3D point cloud

*Instance counting of vehicles

*Grid-based path planning on the segmented environment

1. Semantic Segmentation

A learning-based semantic segmentation approach is applied to a selected Toronto-3D point cloud.

*Input: Raw LiDAR point cloud (.ply)

*Output: Per-point semantic labels


Preprocessing steps such as filtering and normalization may be applied to improve segmentation quality.

2. Instance Counting (Car Class)

Using the semantic segmentation output:

*Points labeled as car are isolated.

*A clustering-based method is applied to group points belonging to the same physical object.

*The total number of distinct car instances is then computed.

*This step bridges semantic and instance-level perception.

3. 2D Grid-Based Path Planning

To enable navigation:

*The 3D point cloud is projected onto a 2D occupancy grid.

*Traversable and non-traversable regions are derived from semantic labels.

*A grid-based planner (e.g., A* algorithm) is implemented to find a path between a selected start and goal location while avoiding obstacles.

Implementation Notes

*The project was implemented on a CPU-only machine (Intel i5, no GPU).

*Model and algorithm choices prioritize computational efficiency and clarity of methodology.

*The focus is on demonstrating a complete perception-to-planning pipeline rather than achieving maximum performance.


