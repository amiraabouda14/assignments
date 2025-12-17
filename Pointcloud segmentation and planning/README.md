# Pointcloud segmentation and planning

## Toronto-3D RandLA-Net

This repo contains a minimal, CPU-friendly RandLA-Net pipeline for one Toronto-3D tile (default: `L001.ply`) with:

- Semantic segmentation (Task 1)
- Car instance counting (Task 2)
- 2D grid-based path planning with A* (Task 3)
- Lightweight ROS2 nodes (optional) to wrap inference and planning

> Designed for a personal Windows machine with an i5 CPU and no GPU. Training is kept tiny; inference runs on a single tile.

## Quickstart
1) Install Python 3.8+ and run:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Run a tiny end-to-end pass on `L001.ply`:
```
python -m src.pipeline --tile L001.ply --epochs 2 --voxel 0.3
```
Outputs go to `outputs/`:
- `segmented_L001.ply` (semantic labels + colors)
- `car_instances.json` (instance count + centroids/boxes)
- `occupancy_grid.png` and `planned_path.json`

3) (Optional) Export a ROS2-ready inference node package in `ros2_nodes/`. Launch only if ROS2 is installed:
```
python ros2_nodes/segmentation_node.py --help
```

## Files
- `src/pipeline.py` – orchestrates preprocessing, RandLA-Net training/inference, instance clustering, planning.
- `src/model.py` – compact RandLA-Net implementation tuned for CPU.
- `src/data_io.py` – PLY loader and feature utilities.
- `src/planning.py` – occupancy grid and A*.
- `ros2_nodes/` – ROS2 wrappers (rclpy-based) for segmentation + planning.

## Notes for CPU use
- Default model width (16) and small batch sizes keep memory in check.
- Training epochs default to 2; increase slowly if you have time.
- kNN is computed with scikit-learn on subsampled clouds to avoid O(N²) blowup.

## Dataset assumptions
- Toronto-3D tiles contain properties: `x y z intensity red green blue label`.
- Classes (from `Mavericks_classes_9.txt`): Unclassified, Ground, Road_markings, Natural, Building, Utility_line, Pole, Car, Fence.


