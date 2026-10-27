from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/mnt/d/comsol/AZ91D_solver_surrogate_20260830/38_formal_v1_1_108_0_24h_surrogate_dataset_20260905")
OUT = Path("/mnt/d/comsol/AZ91D_solver_surrogate_20260830/39_formal_v1_1_surrogate_stage1_20260906")
TRAIN_HOURS = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24]
HELDOUT_HOURS = [3, 5, 7, 9, 14, 18, 22]
FIELD_INDEX = 3
FIELD_ORDER = ["aqt.phil", "q_alpha", "q_beta", "pf.Vf1", "aqt.c_Mg", "aqt.pH"]
SCALAR_NAMES = ["Faraday_mass_kg_per_m", "I_alpha_A_per_m", "I_beta_A_per_m"]
DESCRIPTOR_NAMES = [
    "f_beta_actual",
    "connected_component_count",
    "largest_component_fraction",
    "alpha_beta_interface_length_density_per_m",
    "beta_surface_contact_fraction",
    "beta_equivalent_diameter_mean_um",
    "beta_equivalent_diameter_median_um",
    "beta_adjacency_per_pixel",
    "beta_min_feature_diameter_um",
]
TOPOLOGIES = ["fragmented", "moderate", "network"]
SEED = 20260906


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


@dataclass
class BinaryHeader:
    nx: int
    ny: int
    nfields: int
    ntimes: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    times: list[float]
    offset: int


def read_header(path: Path) -> BinaryHeader:
    with path.open("rb") as f:
        magic, version, nx, ny, nf, nt = struct.unpack(">6i", f.read(24))
        if magic != 0x415A5632 or version != 3:
            raise ValueError(f"Unsupported field binary: {path}")
        xmin, xmax, ymin, ymax = struct.unpack(">4d", f.read(32))
        times = list(struct.unpack(f">{nt}d", f.read(8 * nt)))
        offset = 56 + 8 * nt
    return BinaryHeader(nx, ny, nf, nt, xmin, xmax, ymin, ymax, times, offset)


class DataStore:
    def __init__(self, root: Path):
        self.root = root
        self.split = pd.read_csv(root / "FORMAL_V11_108_SPLIT.csv")
        self.desc = pd.read_csv(root / "FORMAL_V11_108_GEOMETRY_DESCRIPTORS.csv").set_index("run_id")
        first = root / "fields" / f"{self.split.iloc[0].run_id}_fields.bin"
        self.header = read_header(first)
        self.times = np.asarray(self.header.times, dtype=np.float64)
        self.hours = self.times / 3600.0
        self._maps: dict[str, np.memmap] = {}
        self._micro: dict[str, np.ndarray] = {}
        self._metrics: dict[str, pd.DataFrame] = {}
        self._pf0: dict[str, np.ndarray] = {}

    def ids(self, split: str) -> list[str]:
        return self.split.loc[self.split.split == split, "run_id"].tolist()

    def mmap(self, run_id: str) -> np.memmap:
        if run_id not in self._maps:
            path = self.root / "fields" / f"{run_id}_fields.bin"
            h = read_header(path)
            if asdict(h) != asdict(self.header):
                raise ValueError(f"Header mismatch: {run_id}")
            self._maps[run_id] = np.memmap(
                path,
                dtype=">f8",
                mode="r",
                offset=h.offset,
                shape=(h.nfields, h.ntimes, h.ny, h.nx),
            )
        return self._maps[run_id]

    def phase(self, run_id: str, time_index: int) -> np.ndarray:
        return np.asarray(self.mmap(run_id)[FIELD_INDEX, time_index], dtype=np.float32)

    def phase0(self, run_id: str) -> np.ndarray:
        if run_id not in self._pf0:
            self._pf0[run_id] = self.phase(run_id, 0)
        return self._pf0[run_id]

    def micro(self, run_id: str) -> np.ndarray:
        if run_id in self._micro:
            return self._micro[run_id]
        source = np.load(self.root / "microstructures" / f"{run_id}_smooth.npy").astype(np.float32)
        h = self.header
        ys = np.linspace(h.ymin, h.ymax, h.ny)
        source_y = np.linspace(-25e-6, 0.0, source.shape[0])
        full = np.full((h.ny, h.nx), -1.0, dtype=np.float32)
        metal = ys <= 0.0
        target_y = ys[metal]
        for ix in range(h.nx):
            full[metal, ix] = np.interp(target_y, source_y, source[:, ix], left=source[0, ix], right=source[-1, ix])
        self._micro[run_id] = full
        return full

    def metrics(self, run_id: str) -> pd.DataFrame:
        if run_id not in self._metrics:
            frame = pd.read_csv(self.root / "results" / f"metrics_{run_id}.csv")
            if len(frame) != self.header.ntimes or not np.allclose(frame.time_s, self.times):
                raise ValueError(f"Metrics time mismatch: {run_id}")
            self._metrics[run_id] = frame
        return self._metrics[run_id]

    def scalars(self, run_id: str, time_index: int) -> np.ndarray:
        row = self.metrics(run_id).iloc[time_index]
        return np.asarray(
            [row.Faraday_mass_kg_per_m, row.I_alpha_A_per_m, row.I_beta_A_per_m], dtype=np.float32
        )

    def descriptors(self, run_id: str, tau: float) -> np.ndarray:
        row = self.desc.loc[run_id]
        values = [float(row[name]) for name in DESCRIPTOR_NAMES]
        values += [1.0 if row.topology == name else 0.0 for name in TOPOLOGIES]
        values.append(float(tau))
        return np.asarray(values, dtype=np.float64)


class PhaseDataset(Dataset):
    def __init__(
        self,
        store: DataStore,
        run_ids: list[str],
        time_indices: list[int],
        scalar_mean: np.ndarray,
        scalar_std: np.ndarray,
    ):
        self.store = store
        self.items = [(r, t) for r in run_ids for t in time_indices]
        self.scalar_mean = scalar_mean.astype(np.float32)
        self.scalar_std = scalar_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        run_id, ti = self.items[index]
        pf0 = self.store.phase0(run_id)
        pf = self.store.phase(run_id, ti)
        delta = pf - pf0
        scalar = (self.store.scalars(run_id, ti) - self.scalar_mean) / self.scalar_std
        tau = np.float32(self.store.times[ti] / 86400.0)
        current_mask = np.float32(self.store.times[ti] > 0.0)
        return {
            "micro": torch.from_numpy(self.store.micro(run_id)[None]),
            "tau": torch.tensor([tau]),
            "delta": torch.from_numpy(delta[None]),
            "scalar": torch.from_numpy(scalar),
            "current_mask": torch.tensor(current_mask),
            "run_id": run_id,
            "time_index": ti,
        }


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        groups = min(8, cout)
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlainUNet(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.e1 = ConvBlock(2, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(base * 8, base * 16)
        self.d4 = ConvBlock(base * 24, base * 8)
        self.d3 = ConvBlock(base * 12, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.field = nn.Conv2d(base, 1, 1)
        self.scalar = nn.Sequential(nn.Linear(base * 16, base * 8), nn.SiLU(), nn.Linear(base * 8, 3))

    def forward(self, micro: torch.Tensor, tau: torch.Tensor):
        pad = (0, 0, 5, 5)
        x = F.pad(micro, pad, value=-1.0)
        tmap = tau[:, :, None, None].expand(-1, 1, x.shape[-2], x.shape[-1])
        e1 = self.e1(torch.cat([x, tmap], 1))
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        mid = self.mid(self.pool(e4))
        d4 = self.d4(torch.cat([F.interpolate(mid, size=e4.shape[-2:], mode="bilinear", align_corners=False), e4], 1))
        d3 = self.d3(torch.cat([F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        field = self.field(d1)[..., 5:-5, :]
        scalar = self.scalar(F.adaptive_avg_pool2d(mid, 1).flatten(1))
        return field, scalar


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64, frequencies: int = 8):
        super().__init__()
        self.frequencies = frequencies
        self.net = nn.Sequential(nn.Linear(1 + 2 * frequencies, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU())

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        freq = (2.0 ** torch.arange(self.frequencies, device=tau.device, dtype=tau.dtype))[None]
        angle = 2.0 * math.pi * tau * freq
        return self.net(torch.cat([tau, torch.sin(angle), torch.cos(angle)], dim=1))


class FilmBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.block = ConvBlock(cin, cout)
        self.affine = nn.Linear(tdim, 2 * cout)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        gamma, beta = self.affine(temb).chunk(2, dim=1)
        return h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class FilmUNet(nn.Module):
    def __init__(self, base: int = 16, tdim: int = 64):
        super().__init__()
        self.time = TimeEmbedding(tdim)
        self.e1 = FilmBlock(1, base, tdim)
        self.e2 = FilmBlock(base, base * 2, tdim)
        self.e3 = FilmBlock(base * 2, base * 4, tdim)
        self.e4 = FilmBlock(base * 4, base * 8, tdim)
        self.pool = nn.MaxPool2d(2)
        self.mid = FilmBlock(base * 8, base * 16, tdim)
        self.d4 = FilmBlock(base * 24, base * 8, tdim)
        self.d3 = FilmBlock(base * 12, base * 4, tdim)
        self.d2 = FilmBlock(base * 6, base * 2, tdim)
        self.d1 = FilmBlock(base * 3, base, tdim)
        self.field = nn.Conv2d(base, 1, 1)
        self.scalar = nn.Sequential(nn.Linear(base * 16 + tdim, base * 8), nn.SiLU(), nn.Linear(base * 8, 3))

    def forward(self, micro: torch.Tensor, tau: torch.Tensor):
        t = self.time(tau)
        x = F.pad(micro, (0, 0, 5, 5), value=-1.0)
        e1 = self.e1(x, t)
        e2 = self.e2(self.pool(e1), t)
        e3 = self.e3(self.pool(e2), t)
        e4 = self.e4(self.pool(e3), t)
        mid = self.mid(self.pool(e4), t)
        d4 = self.d4(torch.cat([F.interpolate(mid, size=e4.shape[-2:], mode="bilinear", align_corners=False), e4], 1), t)
        d3 = self.d3(torch.cat([F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1), t)
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1), t)
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1), t)
        field = self.field(d1)[..., 5:-5, :]
        pooled = F.adaptive_avg_pool2d(mid, 1).flatten(1)
        scalar = self.scalar(torch.cat([pooled, t], dim=1))
        return field, scalar


def time_indices(store: DataStore, hours: list[float], include_zero: bool = False) -> list[int]:
    wanted = ([0.0] if include_zero else []) + hours
    result = []
    for hour in wanted:
        match = np.flatnonzero(np.isclose(store.hours, hour))
        if len(match) != 1:
            raise ValueError(f"Missing time {hour} h")
        result.append(int(match[0]))
    return result


def scalar_stats(store: DataStore, train_ids: list[str], train_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for run_id in train_ids:
        for ti in train_indices:
            value = store.scalars(run_id, ti).astype(np.float64)
            if store.times[ti] == 0:
                value[1:] = np.nan
            rows.append(value)
    values = np.asarray(rows)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[std == 0] = 1.0
    return mean, std


def field_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    l1 = (pred - true).abs().mean()
    diff = (pred - true).flatten(1).norm(dim=1)
    denom = true.flatten(1).norm(dim=1).clamp_min(1e-5)
    relative = (diff / denom).clamp(max=20.0).mean()
    return l1 / 0.01 + 0.05 * relative


def scalar_loss(pred: torch.Tensor, true: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
    mass = F.smooth_l1_loss(pred[:, 0], true[:, 0])
    mask = current_mask > 0.5
    currents = F.smooth_l1_loss(pred[mask, 1:], true[mask, 1:]) if mask.any() else pred[:, 1:].sum() * 0.0
    return mass + currents


def train_model(
    kind: str,
    store: DataStore,
    train_ids: list[str],
    val_ids: list[str],
    train_indices: list[int],
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
    out_dir: Path,
    max_epochs: int,
    patience: int,
    save_checkpoint: bool,
) -> tuple[nn.Module, dict]:
    device = torch.device("cuda")
    model_seed = SEED + (1 if kind == "plain_unet" else 2)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    model = PlainUNet() if kind == "plain_unet" else FilmUNet()
    model.to(device)
    train_ds = PhaseDataset(store, train_ids, train_indices, scalar_mean, scalar_std)
    val_ds = PhaseDataset(store, val_ids, train_indices, scalar_mean, scalar_std)
    loader_generator = torch.Generator().manual_seed(model_seed)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best = float("inf")
    best_state = None
    stale = 0
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sum = 0.0
        for batch in train_loader:
            micro = batch["micro"].to(device, non_blocking=True)
            tau = batch["tau"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            target_scalar = batch["scalar"].to(device, non_blocking=True)
            mask = batch["current_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred_delta, pred_scalar = model(micro, tau)
                loss = field_loss(pred_delta, delta) + 0.25 * scalar_loss(pred_scalar, target_scalar, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.detach()) * len(micro)
        scheduler.step()
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                micro = batch["micro"].to(device, non_blocking=True)
                tau = batch["tau"].to(device, non_blocking=True)
                delta = batch["delta"].to(device, non_blocking=True)
                target_scalar = batch["scalar"].to(device, non_blocking=True)
                mask = batch["current_mask"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    pred_delta, pred_scalar = model(micro, tau)
                    loss = field_loss(pred_delta, delta) + 0.25 * scalar_loss(pred_scalar, target_scalar, mask)
                val_sum += float(loss) * len(micro)
        row = {
            "epoch": epoch,
            "train_loss": train_sum / len(train_ds),
            "val_loss": val_sum / len(val_ds),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if row["val_loss"] < best - 1e-5:
            best = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(json.dumps({"event": "epoch", "kind": kind, "ntrain": len(train_ids), **row}), flush=True)
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No model checkpoint selected")
    model.load_state_dict(best_state)
    model.to(device).eval()
    if save_checkpoint:
        torch.save(
            {
                "kind": kind,
                "state_dict": best_state,
                "scalar_mean": scalar_mean,
                "scalar_std": scalar_std,
                "train_ids": train_ids,
                "train_time_hours": [float(store.hours[i]) for i in train_indices],
            },
            out_dir / f"{kind}_best.pt",
        )
    pd.DataFrame(history).to_csv(out_dir / f"{kind}_n{len(train_ids)}_history.csv", index=False)
    return model, {
        "best_val_loss": best,
        "epochs": len(history),
        "wall_s": time.time() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def crossings(field: np.ndarray, y: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    ny, nx = field.shape
    out = np.full(nx, np.nan)
    for ix in range(nx):
        v = field[:, ix]
        idx = np.flatnonzero((v[:-1] - 0.5) * (v[1:] - 0.5) <= 0)
        idx = idx[v[idx] != v[idx + 1]]
        if not len(idx):
            continue
        yc = y[idx] + (0.5 - v[idx]) * (y[idx + 1] - y[idx]) / (v[idx + 1] - v[idx])
        target = 0.0 if reference is None else reference[ix]
        out[ix] = yc[np.argmin(np.abs(yc - target))]
    return out


def phase_metrics(pf_pred: np.ndarray, pf_true: np.ndarray, pf0: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = pf_pred - pf_true
    true_norm = max(float(np.linalg.norm(pf_true)), 1e-12)
    delta_norm = max(float(np.linalg.norm(pf_true - pf0)), 1e-12)
    pred_binary = pf_pred >= 0.5
    true_binary = pf_true >= 0.5
    intersection = float(np.logical_and(pred_binary, true_binary).sum())
    union = float(np.logical_or(pred_binary, true_binary).sum())
    denom = float(pred_binary.sum() + true_binary.sum())
    ref = crossings(pf0, y)
    true_interface = crossings(pf_true, y, ref)
    pred_interface = crossings(pf_pred, y, ref)
    valid = np.isfinite(ref) & np.isfinite(true_interface) & np.isfinite(pred_interface)
    result = {
        "phase_mae": float(np.mean(np.abs(err))),
        "phase_rmse": float(np.sqrt(np.mean(err**2))),
        "phase_relative_l2": float(np.linalg.norm(err) / true_norm),
        "delta_phase_relative_l2": float(np.linalg.norm(err) / delta_norm),
        "dice": 2.0 * intersection / max(denom, 1.0),
        "iou": intersection / max(union, 1.0),
    }
    if valid.any():
        true_depth = ref[valid] - true_interface[valid]
        pred_depth = ref[valid] - pred_interface[valid]
        result.update(
            {
                "mean_depth_error_m": float(abs(pred_depth.mean() - true_depth.mean())),
                "p95_depth_error_m": float(abs(np.quantile(pred_depth, 0.95) - np.quantile(true_depth, 0.95))),
                "max_depth_error_m": float(abs(pred_depth.max() - true_depth.max())),
                "true_mean_depth_m": float(true_depth.mean()),
                "pred_mean_depth_m": float(pred_depth.mean()),
                "true_p95_depth_m": float(np.quantile(true_depth, 0.95)),
                "pred_p95_depth_m": float(np.quantile(pred_depth, 0.95)),
                "true_max_depth_m": float(true_depth.max()),
                "pred_max_depth_m": float(pred_depth.max()),
            }
        )
    else:
        result.update({k: float("nan") for k in ["mean_depth_error_m", "p95_depth_error_m", "max_depth_error_m"]})
    return result


def predict_deep(
    model: nn.Module,
    store: DataStore,
    run_ids: list[str],
    indices: list[int],
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
    model_name: str,
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    ds = PhaseDataset(store, run_ids, indices, scalar_mean, scalar_std)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda")
    y = np.linspace(store.header.ymin, store.header.ymax, store.header.ny)
    rows = []
    saved: dict[tuple[str, int], np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            micro = batch["micro"].to(device, non_blocking=True)
            tau = batch["tau"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred_delta, pred_scalar_z = model(micro, tau)
            pred_delta = pred_delta.float().cpu().numpy()[:, 0]
            pred_scalar = pred_scalar_z.float().cpu().numpy() * scalar_std + scalar_mean
            for bi, run_id in enumerate(batch["run_id"]):
                ti = int(batch["time_index"][bi])
                pf0 = store.phase0(run_id)
                true = store.phase(run_id, ti)
                pred = np.clip(pf0 + pred_delta[bi], 0.0, 1.0)
                row = {
                    "model": model_name,
                    "run_id": run_id,
                    "time_index": ti,
                    "time_h": float(store.hours[ti]),
                    "f_beta": float(store.desc.loc[run_id, "f_beta_actual"]),
                    "topology": str(store.desc.loc[run_id, "topology"]),
                    **phase_metrics(pred, true, pf0, y),
                }
                true_scalar = store.scalars(run_id, ti)
                for j, name in enumerate(SCALAR_NAMES):
                    row[f"true_{name}"] = float(true_scalar[j])
                    row[f"pred_{name}"] = float(pred_scalar[bi, j])
                denom = abs(pred_scalar[bi, 1]) + abs(pred_scalar[bi, 2]) + 1e-30
                row["closure_pred"] = float(abs(pred_scalar[bi, 1] + pred_scalar[bi, 2]) / denom)
                row["closure_teacher"] = float(store.metrics(run_id).iloc[ti].current_closure_error)
                rows.append(row)
                if run_id == run_ids[0] and float(store.hours[ti]) in {1.0, 12.0, 24.0}:
                    saved[(run_id, ti)] = pred
    return pd.DataFrame(rows), saved


def zero_baseline(store: DataStore, run_ids: list[str], indices: list[int]) -> pd.DataFrame:
    y = np.linspace(store.header.ymin, store.header.ymax, store.header.ny)
    rows = []
    for run_id in run_ids:
        pf0 = store.phase0(run_id)
        for ti in indices:
            row = {
                "model": "zero_change",
                "run_id": run_id,
                "time_index": ti,
                "time_h": float(store.hours[ti]),
                "f_beta": float(store.desc.loc[run_id, "f_beta_actual"]),
                "topology": str(store.desc.loc[run_id, "topology"]),
                **phase_metrics(pf0, store.phase(run_id, ti), pf0, y),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def fit_scalar_baselines(store: DataStore, train_ids: list[str], val_ids: list[str], train_indices: list[int]):
    models = {"ridge": [], "extra_trees": []}
    metadata = defaultdict(list)
    for target_index, target in enumerate(SCALAR_NAMES):
        def matrix(ids: list[str]):
            x, y = [], []
            for run_id in ids:
                for ti in train_indices:
                    if target_index > 0 and store.times[ti] == 0:
                        continue
                    x.append(store.descriptors(run_id, store.times[ti] / 86400.0))
                    y.append(store.scalars(run_id, ti)[target_index])
            return np.asarray(x), np.asarray(y)

        xtr, ytr = matrix(train_ids)
        xva, yva = matrix(val_ids)
        xscaler = StandardScaler().fit(xtr)
        ymean, ystd = float(ytr.mean()), float(ytr.std() or 1.0)
        best_ridge = None
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            candidate = Ridge(alpha=alpha).fit(xscaler.transform(xtr), (ytr - ymean) / ystd)
            pred = candidate.predict(xscaler.transform(xva)) * ystd + ymean
            score = float(np.sqrt(mean_squared_error(yva, pred)))
            if best_ridge is None or score < best_ridge[0]:
                best_ridge = (score, alpha, candidate)
        models["ridge"].append((xscaler, ymean, ystd, best_ridge[2]))
        metadata["ridge"].append({"target": target, "alpha": best_ridge[1], "val_rmse": best_ridge[0]})
        best_tree = None
        for leaf in [1, 2, 4]:
            candidate = ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=leaf,
                max_features=1.0,
                random_state=SEED,
                n_jobs=-1,
            ).fit(xtr, ytr)
            pred = candidate.predict(xva)
            score = float(np.sqrt(mean_squared_error(yva, pred)))
            if best_tree is None or score < best_tree[0]:
                best_tree = (score, leaf, candidate)
        models["extra_trees"].append((None, 0.0, 1.0, best_tree[2]))
        metadata["extra_trees"].append({"target": target, "min_samples_leaf": best_tree[1], "val_rmse": best_tree[0]})
    return models, dict(metadata)


def predict_scalar_baseline(name: str, fitted, store: DataStore, run_ids: list[str], indices: list[int]) -> pd.DataFrame:
    rows = []
    for run_id in run_ids:
        for ti in indices:
            x = store.descriptors(run_id, store.times[ti] / 86400.0)[None]
            pred = []
            for scaler, mean, std, model in fitted[name]:
                xx = scaler.transform(x) if scaler is not None else x
                pred.append(float(model.predict(xx)[0] * std + mean))
            true = store.scalars(run_id, ti)
            row = {"model": name, "run_id": run_id, "time_h": float(store.hours[ti])}
            for j, target in enumerate(SCALAR_NAMES):
                row[f"true_{target}"] = float(true[j])
                row[f"pred_{target}"] = pred[j]
            row["closure_pred"] = abs(pred[1] + pred[2]) / (abs(pred[1]) + abs(pred[2]) + 1e-30)
            row["closure_teacher"] = float(store.metrics(run_id).iloc[ti].current_closure_error)
            rows.append(row)
    return pd.DataFrame(rows)


def group_name(hour: float) -> str:
    if any(abs(hour - x) < 1e-8 for x in TRAIN_HOURS):
        return "training_grid"
    if any(abs(hour - x) < 1e-8 for x in HELDOUT_HOURS):
        return "heldout_interpolation"
    return "auxiliary"


def regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[mask], pred[mask]
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred)) if len(true) > 1 else float("nan"),
        "relative_mae": float(np.mean(np.abs(pred - true)) / max(np.mean(np.abs(true)), 1e-30)),
    }


def aggregate_outputs(phase: pd.DataFrame, scalar: pd.DataFrame):
    comparison = []
    for (model, group), frame in phase.groupby(["model", phase.time_h.map(group_name)]):
        row = {"task": "phase", "model": model, "time_group": group, "n": len(frame)}
        for metric in ["phase_mae", "phase_rmse", "phase_relative_l2", "delta_phase_relative_l2", "dice", "iou", "mean_depth_error_m", "p95_depth_error_m", "max_depth_error_m"]:
            row[metric] = float(frame[metric].mean())
        comparison.append(row)
    for (model, group), frame in scalar.groupby(["model", scalar.time_h.map(group_name)]):
        for target in SCALAR_NAMES:
            use = frame if target == SCALAR_NAMES[0] else frame[frame.time_h > 0]
            comparison.append({"task": target, "model": model, "time_group": group, "n": len(use), **regression_metrics(use[f"true_{target}"].to_numpy(), use[f"pred_{target}"].to_numpy())})
        closure = frame.loc[frame.time_h > 0, "closure_pred"]
        comparison.append({
            "task": "current_closure",
            "model": model,
            "time_group": group,
            "n": len(closure),
            "mean": float(closure.mean()),
            "median": float(closure.median()),
            "p95": float(closure.quantile(0.95)),
            "max": float(closure.max()),
            "teacher_mean": float(frame.loc[frame.time_h > 0, "closure_teacher"].mean()),
        })
    comparison_df = pd.DataFrame(comparison)

    time_rows = []
    for (model, hour), frame in phase.groupby(["model", "time_h"]):
        time_rows.append({"task": "phase", "model": model, "time_h": hour, "time_group": group_name(hour), "n": len(frame), "mae": frame.phase_mae.mean(), "rmse": frame.phase_rmse.mean(), "relative_l2": frame.phase_relative_l2.mean(), "delta_relative_l2": frame.delta_phase_relative_l2.mean(), "dice": frame.dice.mean(), "iou": frame.iou.mean()})
    for (model, hour), frame in scalar.groupby(["model", "time_h"]):
        for target in SCALAR_NAMES:
            if target != SCALAR_NAMES[0] and hour == 0:
                continue
            met = regression_metrics(frame[f"true_{target}"].to_numpy(), frame[f"pred_{target}"].to_numpy())
            time_rows.append({"task": target, "model": model, "time_h": hour, "time_group": group_name(hour), "n": len(frame), **met})
    time_df = pd.DataFrame(time_rows)

    stratified = []
    for model, model_frame in phase.groupby("model"):
        for dim in ["f_beta", "topology", "time_h"]:
            for value, frame in model_frame.groupby(dim):
                stratified.append({"model": model, "stratifier": dim, "stratum": value, "n": len(frame), "phase_mae": frame.phase_mae.mean(), "phase_rmse": frame.phase_rmse.mean(), "phase_relative_l2": frame.phase_relative_l2.mean(), "delta_phase_relative_l2": frame.delta_phase_relative_l2.mean(), "dice": frame.dice.mean(), "iou": frame.iou.mean(), "mean_depth_error_m": frame.mean_depth_error_m.mean(), "p95_depth_error_m": frame.p95_depth_error_m.mean(), "max_depth_error_m": frame.max_depth_error_m.mean()})
    return comparison_df, time_df, pd.DataFrame(stratified)


def select_balanced(store: DataStore, n: int) -> list[str]:
    frame = store.split[store.split.split == "train"].sort_values(["f_beta_target", "topology", "replicate"])
    per_cell = n // 9
    selected = frame.groupby(["f_beta_target", "topology"], sort=True).head(per_cell)
    if len(selected) != n:
        raise ValueError(f"Could not select balanced n={n}")
    return selected.run_id.tolist()


def plot_examples(store: DataStore, film_saved: dict, test_id: str, out: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    for row, hour in enumerate([1.0, 12.0, 24.0]):
        ti = int(np.flatnonzero(np.isclose(store.hours, hour))[0])
        true = store.phase(test_id, ti)
        pred = film_saved[(test_id, ti)]
        for col, (title, field, cmap) in enumerate([("COMSOL", true, "viridis"), ("FiLM U-Net", pred, "viridis"), ("Absolute error", np.abs(pred - true), "magma")]):
            im = axes[row, col].imshow(field, origin="lower", aspect="equal", cmap=cmap, extent=[-100, 100, -40, 100])
            axes[row, col].set_title(f"{hour:g} h | {title}")
            fig.colorbar(im, ax=axes[row, col], fraction=0.035)
    fig.savefig(out / "01_phase_evolution_examples.png", dpi=180)
    plt.close(fig)


def make_plots(store: DataStore, phase: pd.DataFrame, scalar: pd.DataFrame, learning: pd.DataFrame, film_saved: dict, out: Path) -> None:
    test_id = store.ids("test")[0]
    plot_examples(store, film_saved, test_id, out)
    rep_phase = phase[phase.run_id == test_id]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for model, frame in rep_phase.groupby("model"):
        ax.plot(frame.time_h, frame.pred_max_depth_m * 1e9, label=f"{model} predicted")
    teacher = rep_phase[rep_phase.model == "time_conditioned_unet"]
    ax.plot(teacher.time_h, teacher.true_max_depth_m * 1e9, "k--", label="COMSOL")
    ax.set(xlabel="Time (h)", ylabel="Maximum depth (nm)", title=f"Interface evolution: {test_id}")
    ax.legend(fontsize=8)
    fig.savefig(out / "02_interface_depth_evolution.png", dpi=180)
    plt.close(fig)

    rep_scalar = scalar[scalar.run_id == test_id]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for model, frame in rep_scalar.groupby("model"):
        ax.plot(frame.time_h, frame[f"pred_{SCALAR_NAMES[0]}"] * 1e9, label=model)
    teacher = rep_scalar[rep_scalar.model == "time_conditioned_unet"]
    ax.plot(teacher.time_h, teacher[f"true_{SCALAR_NAMES[0]}"] * 1e9, "k--", label="COMSOL")
    ax.set(xlabel="Time (h)", ylabel="Faraday mass (1e-9 kg/m)", title=f"Mass trajectory: {test_id}")
    ax.legend(fontsize=8)
    fig.savefig(out / "03_faraday_mass_trajectories.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for target, ax in zip(SCALAR_NAMES[1:], axes):
        for model, frame in rep_scalar.groupby("model"):
            ax.plot(frame.time_h, frame[f"pred_{target}"] * 1e6, label=model)
        ax.plot(teacher.time_h, teacher[f"true_{target}"] * 1e6, "k--", label="COMSOL")
        ax.set(xlabel="Time (h)", ylabel="Current (uA/m)", title=target)
    axes[1].legend(fontsize=8)
    fig.savefig(out / "04_current_trajectories.png", dpi=180)
    plt.close(fig)

    held = phase[phase.time_h.isin(HELDOUT_HOURS)].groupby(["model", "time_h"]).phase_relative_l2.mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for model, frame in held.groupby("model"):
        ax.plot(frame.time_h, frame.phase_relative_l2, marker="o", label=model)
    ax.set(xlabel="Held-out time (h)", ylabel="Phase relative L2", title="Unseen-time generalization")
    ax.legend()
    fig.savefig(out / "05_unseen_time_error.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    labels = [("phase_relative_l2", "Phase relative L2"), ("mass_relative_mae", "Mass relative MAE"), ("current_relative_mae", "Current relative MAE")]
    for ax, (column, title) in zip(axes, labels):
        for group, frame in learning.groupby("time_group"):
            ax.plot(frame.n_train_trajectories, frame[column], marker="o", label=group)
        ax.set(xlabel="Training trajectories", ylabel=title)
    axes[-1].legend(fontsize=8)
    fig.savefig(out / "06_learning_curve.png", dpi=180)
    plt.close(fig)


def evaluate_learning_model(model, store, test_ids, indices, mean, std, ntrain):
    frame, _ = predict_deep(model, store, test_ids, indices, mean, std, f"film_n{ntrain}")
    rows = []
    for group, part in frame.groupby(frame.time_h.map(group_name)):
        mass = regression_metrics(part[f"true_{SCALAR_NAMES[0]}"].to_numpy(), part[f"pred_{SCALAR_NAMES[0]}"].to_numpy())
        currents = []
        for target in SCALAR_NAMES[1:]:
            use = part[part.time_h > 0]
            currents.append(regression_metrics(use[f"true_{target}"].to_numpy(), use[f"pred_{target}"].to_numpy())["relative_mae"])
        rows.append({"n_train_trajectories": ntrain, "time_group": group, "phase_relative_l2": float(part.phase_relative_l2.mean()), "phase_mae": float(part.phase_mae.mean()), "mass_relative_mae": mass["relative_mae"], "mass_rmse": mass["rmse"], "current_relative_mae": float(np.mean(currents))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    set_seed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Stage 1 training")
    OUT.mkdir(parents=True, exist_ok=True)
    pass_dir = OUT / "PASS_TO_CHAT"
    pass_dir.mkdir(exist_ok=True)
    checkpoint_dir = OUT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    store = DataStore(ROOT)
    train_ids, val_ids, test_ids = store.ids("train"), store.ids("validation"), store.ids("test")
    train_idx = time_indices(store, TRAIN_HOURS, include_zero=True)
    all_idx = list(range(store.header.ntimes))
    scalar_mean, scalar_std = scalar_stats(store, train_ids, train_idx)
    write_json(OUT / "normalization.json", {"source": "train_only", "scalar_names": SCALAR_NAMES, "mean": scalar_mean.tolist(), "std": scalar_std.tolist()})

    # Data and split integrity checks happen before any fit.
    if (len(train_ids), len(val_ids), len(test_ids)) != (72, 18, 18):
        raise RuntimeError("Frozen trajectory split changed")
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise RuntimeError("Trajectory leakage in frozen split")
    sample = store.phase(train_ids[0], 1)
    if sample.shape != (358, 512) or not np.isfinite(sample).all():
        raise RuntimeError("Invalid phase export")

    scalar_models, scalar_meta = fit_scalar_baselines(store, train_ids, val_ids, train_idx)
    scalar_frames = [predict_scalar_baseline(name, scalar_models, store, test_ids, all_idx) for name in ["ridge", "extra_trees"]]
    max_epochs = 2 if args.smoke else 30
    patience = 2 if args.smoke else 6
    plain, plain_meta = train_model("plain_unet", store, train_ids, val_ids, train_idx, scalar_mean, scalar_std, checkpoint_dir, max_epochs, patience, not args.smoke)
    film, film_meta = train_model("time_conditioned_unet", store, train_ids, val_ids, train_idx, scalar_mean, scalar_std, checkpoint_dir, max_epochs, patience, not args.smoke)
    plain_eval, _ = predict_deep(plain, store, test_ids, all_idx, scalar_mean, scalar_std, "plain_unet")
    film_eval, film_saved = predict_deep(film, store, test_ids, all_idx, scalar_mean, scalar_std, "time_conditioned_unet")
    scalar_frames += [plain_eval, film_eval]
    scalar_eval = pd.concat(scalar_frames, ignore_index=True)
    zero_eval = zero_baseline(store, test_ids, all_idx)
    phase_eval = pd.concat([zero_eval, plain_eval, film_eval], ignore_index=True)

    learning_rows = []
    full_learning = evaluate_learning_model(film, store, test_ids, all_idx, scalar_mean, scalar_std, 72)
    for ntrain in [18, 36, 54]:
        subset = select_balanced(store, ntrain)
        lc_model, lc_meta = train_model("time_conditioned_unet", store, subset, val_ids, train_idx, scalar_mean, scalar_std, checkpoint_dir, max_epochs, patience, False)
        learning_rows.extend(evaluate_learning_model(lc_model, store, test_ids, all_idx, scalar_mean, scalar_std, ntrain))
        write_json(OUT / f"learning_n{ntrain}_meta.json", lc_meta)
        del lc_model
        torch.cuda.empty_cache()
    learning_rows.extend(full_learning)
    learning = pd.DataFrame(learning_rows).sort_values(["time_group", "n_train_trajectories"])

    comparison, time_generalization, stratified = aggregate_outputs(phase_eval, scalar_eval)
    comparison.to_csv(pass_dir / "SURROGATE_STAGE1_MODEL_COMPARISON.csv", index=False)
    time_generalization.to_csv(pass_dir / "SURROGATE_STAGE1_TIME_GENERALIZATION.csv", index=False)
    stratified.to_csv(pass_dir / "SURROGATE_STAGE1_STRATIFIED_METRICS.csv", index=False)
    learning.to_csv(pass_dir / "SURROGATE_STAGE1_LEARNING_CURVE.csv", index=False)
    phase_eval.to_csv(OUT / "test_phase_predictions_metrics.csv", index=False)
    scalar_eval.to_csv(OUT / "test_scalar_predictions.csv", index=False)
    if not args.smoke:
        make_plots(store, phase_eval, scalar_eval, learning, film_saved, pass_dir)

    def best_row(task: str, group: str):
        frame = comparison[(comparison.task == task) & (comparison.time_group == group)]
        return frame.replace({np.nan: None}).to_dict(orient="records")

    closure = {}
    for model, frame in scalar_eval[scalar_eval.time_h > 0].groupby("model"):
        closure[model] = {
            "mean": float(frame.closure_pred.mean()),
            "median": float(frame.closure_pred.median()),
            "p95": float(frame.closure_pred.quantile(0.95)),
            "max": float(frame.closure_pred.max()),
            "teacher_mean": float(frame.closure_teacher.mean()),
            "teacher_p95": float(frame.closure_teacher.quantile(0.95)),
            "teacher_max": float(frame.closure_teacher.max()),
        }
    film_phase = comparison[(comparison.task == "phase") & (comparison.model == "time_conditioned_unet")]
    zero_phase = comparison[(comparison.task == "phase") & (comparison.model == "zero_change")]
    decision = {
        "film_beats_zero_change_training_grid": bool(float(film_phase[film_phase.time_group == "training_grid"].phase_relative_l2.iloc[0]) < float(zero_phase[zero_phase.time_group == "training_grid"].phase_relative_l2.iloc[0])),
        "film_beats_zero_change_heldout": bool(float(film_phase[film_phase.time_group == "heldout_interpolation"].phase_relative_l2.iloc[0]) < float(zero_phase[zero_phase.time_group == "heldout_interpolation"].phase_relative_l2.iloc[0])),
        "dataset_expansion": "decide_from_learning_curve; no expansion launched",
        "physics_loss": "not used in Stage 1; closure evaluated only",
    }
    report = {
        "title": "Formal V1.1 108-trajectory surrogate Stage 1",
        "status": "SMOKE" if args.smoke else "COMPLETE",
        "data": {
            "root": str(ROOT),
            "split": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids), "unit": "complete microstructure trajectory"},
            "times_h": store.hours.tolist(),
            "training_times_h": [0.0] + TRAIN_HOURS,
            "heldout_interpolation_times_h": HELDOUT_HOURS,
            "normalization": "train trajectories at training times only",
            "field_shape": [store.header.ny, store.header.nx],
            "physical_bounds_m": [store.header.xmin, store.header.xmax, store.header.ymin, store.header.ymax],
            "micro_encoding": "smooth beta indicator in metal; -1 deterministic electrolyte sentinel; no physical aspect-ratio distortion",
        },
        "targets": {"field": "delta_pf = pf.Vf1(t)-pf.Vf1(0)", "reconstruction": "clip(pf0+delta_pf_pred,0,1)", "scalars": SCALAR_NAMES, "t0_current_loss_masked": True},
        "models": {"ridge": scalar_meta["ridge"], "extra_trees": scalar_meta["extra_trees"], "plain_unet": plain_meta, "time_conditioned_unet": film_meta},
        "loss": {"field": "L1/0.01 + 0.05*clipped sample relative L2", "scalar": "train-standardized SmoothL1; currents masked at t=0", "multitask": "field + 0.25*scalar"},
        "evaluation": {
            "phase_on": "absolute reconstructed pf.Vf1",
            "depth_definition": "pf.Vf1=0.5 crossing nearest initial interface, linearly interpolated on common 512x358 export grid",
            "strict_sets": ["unseen test microstructures at training-grid times", "unseen test microstructures at held-out times"],
            "closure": closure,
            "key_rows": {"phase_training_grid": best_row("phase", "training_grid"), "phase_heldout": best_row("phase", "heldout_interpolation")},
        },
        "learning_curve": learning.replace({np.nan: None}).to_dict(orient="records"),
        "decision": decision,
        "prohibitions_honored": {"new_comsol_runs": False, "split_changed": False, "test_used_for_selection": False, "physics_loss": False, "q_fields_trained": False, "raw_phi_trained": False, "transformer_fno_ensemble": False},
        "files": [p.name for p in sorted(pass_dir.glob("*"))],
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
    }
    write_json(pass_dir / "SURROGATE_STAGE1_REPORT.json", report)
    print(json.dumps({"event": "complete", "pass_to_chat": str(pass_dir), "decision": decision}), flush=True)


if __name__ == "__main__":
    main()
