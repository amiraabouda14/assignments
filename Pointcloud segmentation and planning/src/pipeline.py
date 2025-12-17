import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import DBSCAN

from src.data_io import (
    CLASS_MAP,
    apply_offset,
    clip_z,
    compute_class_weights,
    filter_invalid,
    load_tile,
    normalize_features,
    save_json,
)
from src.model import RandLANetSmall, cross_entropy_loss
from src.planning import GridSpec, astar, build_grid, save_path


def voxel_downsample(points, feats, labels, voxel: float):
    coords = np.floor(points / voxel).astype(np.int64)
    key = coords[:, 0] * 73856093 ^ coords[:, 1] * 19349663 ^ coords[:, 2] * 83492791
    order = np.argsort(key)
    key_sorted = key[order]
    pts_sorted, feats_sorted, lbl_sorted = points[order], feats[order], labels[order]

    unique_mask = np.ones(len(key_sorted), dtype=bool)
    unique_mask[1:] = key_sorted[1:] != key_sorted[:-1]
    unique_indices = np.nonzero(unique_mask)[0]
    splits = np.split(np.arange(len(key_sorted)), unique_indices[1:])

    pts_out, feats_out, labels_out = [], [], []
    for idxs in splits:
        pts_out.append(pts_sorted[idxs].mean(axis=0))
        feats_out.append(feats_sorted[idxs].mean(axis=0))
        labels_out.append(np.bincount(lbl_sorted[idxs]).argmax())
    return np.vstack(pts_out), np.vstack(feats_out), np.array(labels_out)


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, points, feats, labels, num_points=4096):
        self.points = points
        self.feats = feats
        self.labels = labels
        self.num_points = num_points

    def __len__(self):
        return max(len(self.points) // self.num_points, 1)

    def __getitem__(self, idx):
        N = len(self.points)
        choice = np.random.choice(N, self.num_points, replace=N < self.num_points)
        pts = torch.from_numpy(self.points[choice]).float()
        feats = torch.from_numpy(self.feats[choice]).float()
        labels = torch.from_numpy(self.labels[choice]).long()
        return pts, feats, labels


def train_model(model, loader, class_weights, device, epochs=2, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        total = 0.0
        for pts, feats, labels in loader:
            pts, feats, labels = pts.to(device), feats.to(device), labels.to(device)
            logits = model(pts, feats)
            loss = cross_entropy_loss(logits, labels, class_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(len(loader), 1)
        epoch_losses.append(avg)
        print(f"Epoch {epoch+1}/{epochs} loss {avg:.4f}")
    return epoch_losses


def inference(model, points, feats, device, chunk=8192):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(points), chunk):
            pts_t = torch.from_numpy(points[i : i + chunk]).float().unsqueeze(0).to(device)
            feats_t = torch.from_numpy(feats[i : i + chunk]).float().unsqueeze(0).to(device)
            logits = model(pts_t, feats_t)  # [1, M, C]
            preds.append(logits.argmax(dim=-1).cpu().numpy().squeeze(0))
    return np.concatenate(preds, axis=0)


def car_instances(points, labels, class_id=7, eps=0.8, min_samples=20):
    mask = labels == class_id
    car_pts = points[mask]
    if len(car_pts) == 0:
        return {"count": 0, "instances": []}
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(car_pts[:, :2])
    clusters = clustering.labels_
    instances = []
    for cid in sorted(set(clusters)):
        if cid == -1:
            continue
        sel = car_pts[clusters == cid]
        bbox_min = sel.min(axis=0).tolist()
        bbox_max = sel.max(axis=0).tolist()
        centroid = sel.mean(axis=0).tolist()
        instances.append({"id": int(cid), "centroid": centroid, "bbox_min": bbox_min, "bbox_max": bbox_max})
    return {"count": len(instances), "instances": instances}


def metrics(labels: np.ndarray, preds: np.ndarray, num_classes: int = 9):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    ious = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 0.0)
    acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    miou = float(np.mean(ious))
    return cm, ious, acc, miou


def plot_confusion(cm: np.ndarray, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_iou(ious, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(ious)), ious)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("IoU")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_grid_with_path(grid: np.ndarray, path, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid.T, origin="lower", cmap="gray_r")
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="red", linewidth=1.5)
        ax.scatter(xs[0], ys[0], c="green", s=20, label="start")
        ax.scatter(xs[-1], ys[-1], c="blue", s=20, label="goal")
        ax.legend(loc="upper right")
    ax.set_title("Occupancy grid with A* path")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_car_clusters(
    points: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    eps: float,
    min_samples: int,
    out_path: Path,
):
    """Visualize car instances in 2D (XY) with DBSCAN clusters."""
    mask = labels == class_id
    car_pts = points[mask]
    if len(car_pts) == 0:
        return
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(car_pts[:, :2])
    cluster_ids = clustering.labels_

    fig, ax = plt.subplots(figsize=(6, 6))
    # noise first (label = -1)
    noise = cluster_ids == -1
    if noise.any():
        ax.scatter(
            car_pts[noise, 0],
            car_pts[noise, 1],
            c="lightgray",
            s=1,
            label="noise",
        )

    valid_ids = sorted([cid for cid in set(cluster_ids) if cid != -1])
    for cid in valid_ids:
        sel = cluster_ids == cid
        pts_c = car_pts[sel]
        ax.scatter(
            pts_c[:, 0],
            pts_c[:, 1],
            s=2,
            label=f"car {cid}",
        )
        # draw simple bbox in XY
        xmin, ymin = pts_c[:, 0].min(), pts_c[:, 1].min()
        xmax, ymax = pts_c[:, 0].max(), pts_c[:, 1].max()
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="red",
                linewidth=0.8,
            )
        )

    ax.set_aspect("equal", "box")
    ax.set_title("Car instances (DBSCAN clusters)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if len(valid_ids) <= 15:
        ax.legend(markerscale=4, fontsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _subsample(points: np.ndarray, max_n: int = 80000) -> np.ndarray:
    """Randomly subsample points for plotting."""
    n = len(points)
    if n <= max_n:
        return points
    idx = np.random.choice(n, max_n, replace=False)
    return points[idx]


def plot_preprocessing(points_before: np.ndarray, points_clipped: np.ndarray, points_ds: np.ndarray, out_dir: Path):
    """Visualize XY distribution before/after clipping and voxel downsampling, plus Z histograms."""
    # XY scatter comparison
    pb = _subsample(points_before)
    pc = _subsample(points_clipped)
    pd = _subsample(points_ds)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, pts, title in zip(
        axes,
        (pb, pc, pd),
        ("Raw (offset only)", "After Z-clipping", "After voxel downsampling"),
    ):
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.1, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Z height")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "preprocessing_xy.png", dpi=200)
    plt.close(fig)

    # Z histograms
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.hist(points_before[:, 2], bins=80, alpha=0.4, label="raw", density=True)
    ax2.hist(points_clipped[:, 2], bins=80, alpha=0.4, label="z-clipped", density=True)
    ax2.hist(points_ds[:, 2], bins=80, alpha=0.4, label="voxelized", density=True)
    ax2.set_xlabel("Z height")
    ax2.set_ylabel("Density")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "preprocessing_z_hist.png", dpi=200)
    plt.close(fig2)


def plot_loss_curve(losses, out_path: Path):
    """Plot training loss vs epoch."""
    if not losses:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("RandLA-Net training loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default="L001.ply")
    parser.add_argument("--voxel", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--zmin", type=float, default=-5.0)
    parser.add_argument("--zmax", type=float, default=80.0)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--dbscan_eps", type=float, default=0.8)
    parser.add_argument("--dbscan_min_samples", type=int, default=20)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    tile_path = root / args.tile
    out_dir = root / args.out_dir
    out_dir.mkdir(exist_ok=True)

    print(f"Loading {tile_path}")
    data = load_tile(tile_path)
    points_raw_offset, offset = apply_offset(data["points"])
    points = points_raw_offset.copy()
    feats = normalize_features(data["colors"], data["intensity"])
    points, feats, labels = filter_invalid(points, feats, data["labels"])
    points_clipped, feats, labels = clip_z(points, feats, labels, z_min=args.zmin, z_max=args.zmax)

    print("Voxel downsampling ...")
    pts_ds, feats_ds, lbl_ds = voxel_downsample(points_clipped, feats, labels, voxel=args.voxel)
    print(f"Downsampled to {len(pts_ds)} points")

    # Preprocessing visualization
    plot_preprocessing(points_raw_offset, points_clipped, pts_ds, out_dir)

    class_weights = compute_class_weights(lbl_ds)
    dataset = TileDataset(pts_ds, feats_ds, lbl_ds, num_points=args.num_points)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RandLANetSmall(num_classes=9, feat_channels=feats_ds.shape[1]).to(device)
    epoch_losses = train_model(model, loader, class_weights, device, epochs=args.epochs)
    torch.save(model.state_dict(), out_dir / "model.pt")

    print("Running inference on downsampled cloud ...")
    preds = inference(model, pts_ds, feats_ds, device)
    car_info = car_instances(pts_ds, preds, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    save_json(car_info, out_dir / "car_instances.json")

    print(f"Detected {car_info['count']} cars")
    grid, origin, res = build_grid(pts_ds[:, :2], preds, GridSpec())
    start = (grid.shape[0] // 2, 1)
    goal = (grid.shape[0] // 2, grid.shape[1] - 2)
    path = astar(grid, start, goal)
    save_path(path, origin, res, out_dir / "planned_path.json")

    # Metrics vs ground truth
    cm, ious, acc, miou = metrics(lbl_ds, preds, num_classes=9)
    metrics_obj = {
        "overall_accuracy": acc,
        "miou": miou,
        "per_class_iou": {CLASS_MAP[i]: float(v) for i, v in enumerate(ious)},
    }
    save_json(metrics_obj, out_dir / "metrics.json")
    plot_confusion(cm, [CLASS_MAP[i] for i in range(9)], out_dir / "confusion_matrix.png")
    plot_iou(ious, [CLASS_MAP[i] for i in range(9)], out_dir / "per_class_iou.png")
    plot_grid_with_path(grid, path, out_dir / "occupancy_grid.png")
    plot_car_clusters(pts_ds, preds, class_id=7, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, out_path=out_dir / "car_clusters.png")

    # save colored point cloud
    colors_pred = np.array([plt_color(CLASS_MAP[c]) for c in preds], dtype=np.uint8)
    save_ply(out_dir / f"segmented_{tile_path.stem}.ply", pts_ds + offset, colors_pred, preds)

    # plot training loss curve
    plot_loss_curve(epoch_losses, out_dir / "training_loss.png")
    print("Done. Outputs in", out_dir)


def plt_color(name: str) -> Tuple[int, int, int]:
    palette = {
        "Unclassified": (128, 0, 128),
        "Ground": (128, 128, 128),
        "Road_markings": (32, 255, 255),
        "Natural": (16, 128, 1),
        "Building": (0, 0, 255),
        "Utility_line": (33, 255, 6),
        "Pole": (252, 2, 255),
        "Car": (253, 128, 8),
        "Fence": (255, 255, 10),
    }
    return palette.get(name, (255, 0, 0))


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray, labels: np.ndarray):
    import plyfile

    vertex = np.array(
        [
            (p[0], p[1], p[2], c[0], c[1], c[2], int(l))
            for p, c, l in zip(points, colors, labels)
        ],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "u1"),
        ],
    )
    el = plyfile.PlyElement.describe(vertex, "vertex")
    ply = plyfile.PlyData([el], text=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(path))


if __name__ == "__main__":
    main()


