from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of k nearest neighbors within the same set."""
    dist = torch.cdist(x, x)  # [B, N, N]
    idx = dist.topk(k=k, largest=False).indices  # [B, N, k]
    return idx


def knn_cross(src: torch.Tensor, dst: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Nearest neighbors from src set to dst set."""
    dist = torch.cdist(src, dst)  # [B, Ns, Nd]
    idx = dist.topk(k=k, largest=False).indices  # [B, Ns, k] (indices in dst)
    return idx


class SharedMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        y = self.linear(x)
        y = self.bn(y.transpose(1, 2)).transpose(1, 2)
        return self.dropout(F.relu(y))


class LocalFeatureAggregation(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int):
        super().__init__()
        self.k = k
        mid = out_ch // 2
        self.mlp1 = SharedMLP(in_ch, mid)
        self.attn_mlp = SharedMLP(mid, mid)
        self.mlp2 = SharedMLP(mid, out_ch)

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        idx = knn(coords, self.k)  # [B, N, k]
        B, N, _ = coords.shape
        batch_idx = torch.arange(B, device=coords.device)[:, None, None]
        neighbors = feats[batch_idx, idx]  # [B, N, k, C]

        agg = self.mlp1(neighbors.mean(dim=2))
        attn = torch.sigmoid(self.attn_mlp(agg))
        fused = agg * attn + agg
        return self.mlp2(fused)


class RandLANetSmall(nn.Module):
    """Compact RandLA-Net-style encoder-decoder tuned for CPU use."""

    def __init__(self, num_classes: int = 9, k: int = 8, feat_channels: int = 4):
        super().__init__()
        d0 = 16
        self.fc_start = SharedMLP(3 + feat_channels, d0)  # xyz + features

        self.enc1 = LocalFeatureAggregation(d0, d0 * 2, k)
        self.enc2 = LocalFeatureAggregation(d0 * 2, d0 * 4, k)
        self.enc3 = LocalFeatureAggregation(d0 * 4, d0 * 4, k)

        self.dec2 = SharedMLP(d0 * 4 + d0 * 4, d0 * 2)
        self.dec1 = SharedMLP(d0 * 2 + d0 * 2, d0)
        self.classifier = nn.Sequential(
            SharedMLP(d0, d0),
            nn.Linear(d0, num_classes),
        )

    def downsample(self, x: torch.Tensor, feats: torch.Tensor, ratio: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        idx = torch.randperm(N, device=x.device)[: max(1, N // ratio)]
        return x[:, idx], feats[:, idx], idx

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # coords: [B, N, 3]; feats: [B, N, 1] (intensity)
        x0 = torch.cat([coords, feats], dim=-1)
        f0 = self.fc_start(x0)

        f1 = self.enc1(coords, f0)
        c2, f1_ds, idx1 = self.downsample(coords, f1, ratio=4)
        f2 = self.enc2(c2, f1_ds)

        c3, f2_ds, idx2 = self.downsample(c2, f2, ratio=4)
        f3 = self.enc3(c3, f2_ds)

        # decode
        idx_up2 = knn_cross(c2, c3, k=1).squeeze(-1)  # [B, N2]
        batch = torch.arange(coords.shape[0], device=coords.device)[:, None]
        up2 = f3[batch, idx_up2]
        f_dec2 = self.dec2(torch.cat([f2, up2], dim=-1))

        idx_up1 = knn_cross(coords, c2, k=1).squeeze(-1)
        up1 = f_dec2[batch, idx_up1]
        f_dec1 = self.dec1(torch.cat([f1, up1], dim=-1))

        logits = self.classifier(f_dec1)
        return logits


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: List[float]):
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), weight=torch.tensor(class_weights, device=logits.device))
    return loss


