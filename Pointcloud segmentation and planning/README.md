````markdown
# Point Cloud Segmentation and Planning Pipeline

This project implements a complete **perception → reasoning → planning** pipeline on urban LiDAR data using a tile from the **Toronto-3D dataset** (`L001.ply`).  
The pipeline performs **semantic segmentation**, **vehicle instance counting**, and **2D path planning**.

---

## Pipeline Overview

The workflow is composed of three sequential stages:

- **Perception – Semantic Segmentation**  
  A RandLA-Net–style deep neural network predicts a semantic class for each point.

- **Reasoning – Car Instance Counting**  
  Points classified as *car* are clustered to estimate individual vehicle instances.

- **Planning – 2D Grid-Based Path Planning**  
  A bird’s-eye-view occupancy grid is constructed from the segmented map and an **A\*** planner computes a collision-free path.

The entire pipeline can be executed using a single command:

```bash
python -m src.pipeline --tile L001.ply
````

**Main orchestration file:**

```text
src/pipeline.py
```

---

## Dataset and Class Taxonomy

### Input Data

* **File:** `L001.ply` (Toronto-3D tile)

**Contains:**

* 3D coordinates `(x, y, z)`
* RGB color
* Intensity
* Semantic labels (if available)

Both naming conventions used in Toronto-3D are supported:

* `label` / `scalar_Label`
* `intensity` / `scalar_Intensity`

---

### Semantic Classes

* **File:** `Mavericks_classes_9.txt`
* **Class mapping defined in:** `src/data_io.py`

| ID | Class         |
| -- | ------------- |
| 0  | Unclassified  |
| 1  | Ground        |
| 2  | Road markings |
| 3  | Natural       |
| 4  | Building      |
| 5  | Utility line  |
| 6  | Pole          |
| 7  | Car           |
| 8  | Fence         |

---

## 1. Semantic Segmentation (Perception)

### Data Loading

* **File:** `src/data_io.py`

`load_tile(ply_path)` loads the point cloud using `plyfile.PlyData`.

**Outputs:**

* `points` (N × 3)
* `colors` (N × 3)
* `intensity` (N)
* `labels` (N)

---

### Preprocessing

Implemented in `src/data_io.py`:

* Coordinate offset removal (UTM stabilization)
* NaN / Inf filtering
* Z-axis clipping (outlier removal)
* Feature normalization:

  * RGB → `[0, 1]`
  * Intensity → percentile-based normalization

These steps improve numerical stability and training robustness.

---

### Label-Preserving Downsampling

* **Implemented in:** `src/pipeline.py`

Voxel-based downsampling:

* Mean XYZ and features per voxel
* Majority-vote semantic label

This reduces computation while preserving small objects such as cars and poles.

---

### Model Architecture

* **File:** `src/model.py`

`RandLANetSmall`: lightweight RandLA-Net–inspired architecture.

**Characteristics:**

* k-NN local feature aggregation
* Hierarchical encoder–decoder structure
* Optimized for CPU execution

---

### Class Imbalance Handling

* **File:** `src/data_io.py`

`compute_class_weights(labels)` assigns inverse-frequency weights.
Used in **weighted cross-entropy loss** (`src/model.py`).

---

### Training

* **File:** `src/pipeline.py`

* `TileDataset`: samples fixed-size point sets

* `train_model(...)`: runs training epochs

**Outputs:**

* `outputs/model.pt`
* `outputs/training_loss.png`

---

### Inference and Export

* Chunked inference for memory efficiency
* Segmented point cloud saved as:

```text
outputs/segmented_L001.ply
```

---

## 2. Segmentation Evaluation

Implemented in `src/pipeline.py`.

### Metrics

* Overall accuracy
* Per-class IoU
* Mean IoU (mIoU)

### Visualizations

* Confusion matrix
* Per-class IoU bar chart

**Saved outputs:**

```text
outputs/metrics.json
outputs/confusion_matrix.png
outputs/per_class_iou.png
```

---

## 3. Car Instance Counting (Reasoning)

### Semantic Filtering

Points with **class ID = 7 (Car)** are extracted.

---

### Clustering

* **Algorithm:** DBSCAN
* **File:** `src/pipeline.py`

**Parameters exposed via CLI:**

* `--dbscan_eps`
* `--dbscan_min_samples`

---

### Instance Representation

For each detected vehicle:

* Centroid (XYZ)
* Bounding box (min/max XYZ)

Saved to:

```text
outputs/car_instances.json
```

---

### Visualization

Cluster visualization saved as:

```text
outputs/car_clusters.png
```

---

## 4. 2D Grid-Based Path Planning (Planning)

### Bird’s-Eye-View Projection

* Point cloud projected onto the XY plane
* Semantic labels determine obstacle occupancy

---

### Occupancy Grid Construction

* **File:** `src/planning.py`

**Free space:**

* Ground
* Road markings

**Obstacles:**

* Buildings
* Cars
* Poles
* Fences
* Natural

Grid inflation is applied for safety margins.

---

### A* Path Planning

* **File:** `src/planning.py`

Computes the shortest collision-free path between selected start and goal points.

---

### Outputs

**Planned path:**

```text
outputs/planned_path.json
```

**Visualization:**

```text
outputs/occupancy_grid.png
```

---

## ROS2 Integration

Training and evaluation are performed **offline**.
ROS2 is used for **runtime modularity**.

### ROS2 Nodes

* `ros2_nodes/segmentation_node.py`

  * Input: `PointCloud2`
  * Output: segmented `PointCloud2`

* `ros2_nodes/planning_node.py`

  * Input: segmented cloud
  * Output: occupancy grid and planned path

Training and evaluation were executed offline in Python, while ROS2 was used as the integration framework for runtime perception-to-planning deployment.

---

## Output Artifacts

All results are stored in the `outputs/` directory:

* `metrics.json` – accuracy and IoU scores
* `training_loss.png` – training curve
* `segmented_L001.ply` – predicted semantic segmentation
* `car_instances.json` – detected vehicle instances
* `car_clusters.png` – clustering visualization
* `planned_path.json` – path waypoints
* `occupancy_grid.png` – grid and A* path
* `preprocessing_*.png` – preprocessing visualizations

---

## Implementation Notes

* Executed on a **CPU-only machine** (Intel i5, no GPU)
* Design prioritizes:

  * Computational efficiency
  * Methodological clarity
  * End-to-end pipeline completeness

```
```
