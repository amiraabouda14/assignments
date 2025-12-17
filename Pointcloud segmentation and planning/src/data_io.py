import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from plyfile import PlyData


CLASS_MAP = {
    0: "Unclassified",
    1: "Ground",
    2: "Road_markings",
    3: "Natural",
    4: "Building",
    5: "Utility_line",
    6: "Pole",
    7: "Car",
    8: "Fence",
}


def load_tile(ply_path: Path) -> Dict[str, np.ndarray]:
    """Read a Toronto-3D PLY tile and return arrays.

    Handles both legacy names (label/intensity) and scalar_* variants.
    """
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names

    # Required geometry + color
    required = ["x", "y", "z", "red", "green", "blue"]
    for name in required:
        if name not in names:
            raise ValueError(f"Missing property {name} in {ply_path}")

    # Intensity field variants
    intensity_key = None
    for cand in ("intensity", "scalar_Intensity"):
        if cand in names:
            intensity_key = cand
            break

    # Label field variants
    label_key = None
    for cand in ("label", "scalar_Label"):
        if cand in names:
            label_key = cand
            break

    if label_key is None:
        raise ValueError(f"Missing property label or scalar_Label in {ply_path}")

    pts = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
    colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(
        np.float32
    )
    if intensity_key:
        intensity = vertex[intensity_key].astype(np.float32)
    else:
        intensity = np.zeros(len(pts), dtype=np.float32)
    labels = vertex[label_key].astype(np.int64)
    return {"points": pts, "colors": colors, "intensity": intensity, "labels": labels}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def normalize_features(colors: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Normalize RGB to 0-1 and intensity with percentile scaling."""
    colors = colors / 255.0
    p1, p99 = np.percentile(intensity, [1, 99])
    denom = max(p99 - p1, 1e-3)
    intensity_norm = np.clip((intensity - p1) / denom, 0.0, 1.0)[:, None]
    return np.hstack([colors, intensity_norm]).astype(np.float32)


def apply_offset(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Translate points to reduce magnitude (UTM offset)."""
    offset = points.mean(axis=0, keepdims=True)
    return points - offset, offset.squeeze()


def filter_invalid(points: np.ndarray, features: np.ndarray, labels: np.ndarray):
    mask = np.isfinite(points).all(axis=1)
    mask &= np.isfinite(features).all(axis=1)
    return points[mask], features[mask], labels[mask]


def clip_z(
    points: np.ndarray, features: np.ndarray, labels: np.ndarray, z_min=-5.0, z_max=80.0
):
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask], features[mask], labels[mask]


def compute_class_weights(labels: np.ndarray, num_classes: int = 9) -> np.ndarray:
    hist = np.bincount(labels, minlength=num_classes).astype(np.float32)
    hist = np.maximum(hist, 1.0)
    inv = 1.0 / np.log(1.2 + hist / hist.sum())
    inv = inv / inv.sum() * num_classes
    return inv.astype(np.float32)


