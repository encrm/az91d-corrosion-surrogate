from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from surrogate_stage1 import (
    FIELD_INDEX,
    HELDOUT_HOURS,
    OUT as STAGE1_ROOT,
    ROOT as DATA_ROOT,
    TRAIN_HOURS,
    ConvBlock,
    DataStore,
    FilmUNet,
    PlainUNet,
    group_name,
    phase_metrics,
    regression_metrics,
    time_indices,
)


OUT = Path("/mnt/d/comsol/AZ91D_solver_surrogate_20260830/40_formal_v1_1_surrogate_stage2_20260906")
PASS = OUT / "PASS_TO_CHAT"
CHECKPOINTS = OUT / "checkpoints"
SELECTION_DIR = OUT / "selection"
FINAL_SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=float), encoding="utf-8")


@dataclass(frozen=True)
class Stage2Config:
    time_mode: str
    interface_lambda: float
    gradient_weight: float
    scalar_transform: str
    dynamic_weight_cap: float = 4.0
    base_channels: int = 16


class Stage2Dataset(Dataset):
    def __init__(self, store: DataStore, run_ids: list[str], indices: list[int]):
        self.store = store
        self.items = [(run_id, ti) for run_id in run_ids for ti in indices]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        run_id, ti = self.items[index]
        pf0 = self.store.phase0(run_id)
        pf = self.store.phase(run_id, ti)
        raw = self.store.scalars(run_id, ti)
        current_magnitude = 0.5 * (raw[1] - raw[2])
        return {
            "micro": torch.from_numpy(self.store.micro(run_id)[None]),
            "pf0": torch.from_numpy(pf0[None]),
            "delta": torch.from_numpy((pf - pf0)[None]),
            "tau": torch.tensor([self.store.times[ti] / 86400.0], dtype=torch.float32),
            "mass": torch.tensor(raw[0], dtype=torch.float32),
            "current": torch.tensor(current_magnitude, dtype=torch.float32),
            "current_mask": torch.tensor(float(self.store.times[ti] > 0.0), dtype=torch.float32),
            "run_id": run_id,
            "time_index": ti,
        }


class MLPTimeBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.block = ConvBlock(cin, cout)
        self.affine = nn.Linear(tdim, 2 * cout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        gamma, beta = self.affine(t).chunk(2, dim=1)
        return h * (1.0 + 0.1 * torch.tanh(gamma[:, :, None, None])) + 0.1 * beta[:, :, None, None]


class ConstrainedUNet(nn.Module):
    def __init__(self, time_mode: str, mass_scale: float, current_scale: float, base: int = 16):
        super().__init__()
        if time_mode not in {"T1_tau", "T2_tau_tau2", "T3_mlp"}:
            raise ValueError(time_mode)
        self.time_mode = time_mode
        self.register_buffer("mass_scale", torch.tensor(float(mass_scale)))
        self.register_buffer("current_scale", torch.tensor(float(current_scale)))
        self.pool = nn.MaxPool2d(2)
        if time_mode == "T3_mlp":
            tdim = 16
            self.time_mlp = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, tdim), nn.SiLU())
            self.e1 = MLPTimeBlock(1, base, tdim)
            self.e2 = MLPTimeBlock(base, base * 2, tdim)
            self.e3 = MLPTimeBlock(base * 2, base * 4, tdim)
            self.e4 = MLPTimeBlock(base * 4, base * 8, tdim)
            self.mid = MLPTimeBlock(base * 8, base * 16, tdim)
            self.d4 = MLPTimeBlock(base * 24, base * 8, tdim)
            self.d3 = MLPTimeBlock(base * 12, base * 4, tdim)
            self.d2 = MLPTimeBlock(base * 6, base * 2, tdim)
            self.d1 = MLPTimeBlock(base * 3, base, tdim)
            scalar_time_dim = tdim
        else:
            channels = 2 if time_mode == "T1_tau" else 3
            self.e1 = ConvBlock(channels, base)
            self.e2 = ConvBlock(base, base * 2)
            self.e3 = ConvBlock(base * 2, base * 4)
            self.e4 = ConvBlock(base * 4, base * 8)
            self.mid = ConvBlock(base * 8, base * 16)
            self.d4 = ConvBlock(base * 24, base * 8)
            self.d3 = ConvBlock(base * 12, base * 4)
            self.d2 = ConvBlock(base * 6, base * 2)
            self.d1 = ConvBlock(base * 3, base)
            scalar_time_dim = 1 if time_mode == "T1_tau" else 2
        self.field = nn.Conv2d(base, 1, 1)
        self.scalar = nn.Sequential(
            nn.Linear(base * 16 + scalar_time_dim, base * 8),
            nn.SiLU(),
            nn.Linear(base * 8, 2),
        )

    def _block(self, block, x: torch.Tensor, tfeature: torch.Tensor | None) -> torch.Tensor:
        return block(x, tfeature) if self.time_mode == "T3_mlp" else block(x)

    def forward(self, micro: torch.Tensor, tau: torch.Tensor):
        x = F.pad(micro, (0, 0, 5, 5), value=-1.0)
        if self.time_mode == "T1_tau":
            tfeature = tau
            maps = tau[:, :, None, None].expand(-1, 1, x.shape[-2], x.shape[-1])
            x = torch.cat([x, maps], dim=1)
        elif self.time_mode == "T2_tau_tau2":
            tfeature = torch.cat([tau, tau.square()], dim=1)
            maps = tfeature[:, :, None, None].expand(-1, 2, x.shape[-2], x.shape[-1])
            x = torch.cat([x, maps], dim=1)
        else:
            tfeature = self.time_mlp(tau)
        e1 = self._block(self.e1, x, tfeature)
        e2 = self._block(self.e2, self.pool(e1), tfeature)
        e3 = self._block(self.e3, self.pool(e2), tfeature)
        e4 = self._block(self.e4, self.pool(e3), tfeature)
        mid = self._block(self.mid, self.pool(e4), tfeature)
        d4 = self._block(self.d4, torch.cat([F.interpolate(mid, size=e4.shape[-2:], mode="bilinear", align_corners=False), e4], dim=1), tfeature)
        d3 = self._block(self.d3, torch.cat([F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1), tfeature)
        d2 = self._block(self.d2, torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1), tfeature)
        d1 = self._block(self.d1, torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1), tfeature)
        raw_delta = self.field(d1)[..., 5:-5, :]
        delta = tau[:, :, None, None] * raw_delta
        latent = F.adaptive_avg_pool2d(mid, 1).flatten(1)
        raw_scalar = self.scalar(torch.cat([latent, tfeature], dim=1))
        mass = tau[:, 0] * F.softplus(raw_scalar[:, 0]) * self.mass_scale
        current = F.softplus(raw_scalar[:, 1]) * self.current_scale
        return delta, mass, current


def train_scales(store: DataStore, train_ids: list[str], indices: list[int]) -> dict[str, float]:
    mass, current = [], []
    for run_id in train_ids:
        for ti in indices:
            values = store.scalars(run_id, ti)
            mass.append(float(values[0]))
            if store.times[ti] > 0:
                current.append(float(0.5 * (values[1] - values[2])))
    mass_a, current_a = np.asarray(mass), np.asarray(current)
    return {
        "mass_scale": float(np.max(mass_a)),
        "current_scale": float(np.median(current_a)),
        "mass_mean": float(np.mean(mass_a)),
        "mass_std": float(np.std(mass_a)),
        "current_mean": float(np.mean(current_a)),
        "current_std": float(np.std(current_a)),
    }


def dynamic_reference(store: DataStore, train_ids: list[str], indices: list[int]) -> float:
    sampled = []
    for run_id in train_ids:
        pf0 = store.phase0(run_id)
        for ti in indices:
            if store.times[ti] == 0:
                continue
            sampled.append(np.abs(store.phase(run_id, ti)[::4, ::4] - pf0[::4, ::4]).ravel())
    reference = float(np.quantile(np.concatenate(sampled), 0.99))
    return max(reference, 1e-4)


def scalar_training_loss(pred_mass, pred_current, true_mass, true_current, mask, scales, transform: str):
    if transform == "raw_standardized":
        mass_loss = F.smooth_l1_loss((pred_mass - scales["mass_mean"]) / max(scales["mass_std"], 1e-30), (true_mass - scales["mass_mean"]) / max(scales["mass_std"], 1e-30))
        use = mask > 0.5
        current_loss = F.smooth_l1_loss((pred_current[use] - scales["current_mean"]) / max(scales["current_std"], 1e-30), (true_current[use] - scales["current_mean"]) / max(scales["current_std"], 1e-30)) if use.any() else pred_current.sum() * 0.0
    elif transform == "asinh_scaled":
        mass_loss = F.smooth_l1_loss(torch.asinh(pred_mass / scales["mass_scale"]), torch.asinh(true_mass / scales["mass_scale"]))
        use = mask > 0.5
        current_loss = F.smooth_l1_loss(torch.asinh(pred_current[use] / scales["current_scale"]), torch.asinh(true_current[use] / scales["current_scale"])) if use.any() else pred_current.sum() * 0.0
    else:
        raise ValueError(transform)
    return mass_loss + current_loss


def field_training_loss(pred_delta, true_delta, pf0, reference: float, config: Stage2Config):
    error = pred_delta - true_delta
    global_l1 = error.abs().mean() / 0.01
    relative = (error.flatten(1).norm(dim=1) / true_delta.flatten(1).norm(dim=1).clamp_min(1e-5)).clamp(max=20).mean()
    weight = 1.0 + config.interface_lambda * torch.clamp(true_delta.abs() / reference, 0.0, config.dynamic_weight_cap)
    weighted_l1 = (weight * error.abs()).mean() / 0.01
    pred_pf = pf0 + pred_delta
    true_pf = pf0 + true_delta
    grad = (pred_pf[..., 1:, :] - pred_pf[..., :-1, :] - true_pf[..., 1:, :] + true_pf[..., :-1, :]).abs().mean()
    grad += (pred_pf[..., :, 1:] - pred_pf[..., :, :-1] - true_pf[..., :, 1:] + true_pf[..., :, :-1]).abs().mean()
    return global_l1 + 0.05 * relative + 0.35 * weighted_l1 + config.gradient_weight * grad / 0.01


def evaluate_loss(model, loader, device, reference, config, scales):
    total = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            micro = batch["micro"].to(device, non_blocking=True)
            tau = batch["tau"].to(device, non_blocking=True)
            pf0 = batch["pf0"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            mass = batch["mass"].to(device, non_blocking=True)
            current = batch["current"].to(device, non_blocking=True)
            mask = batch["current_mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred_delta, pm, pc = model(micro, tau)
                loss = field_training_loss(pred_delta, delta, pf0, reference, config) + 0.25 * scalar_training_loss(pm, pc, mass, current, mask, scales, config.scalar_transform)
            total += float(loss) * len(micro)
    return total / len(loader.dataset)


def train_model(store, train_ids, val_ids, train_idx, held_idx, config, scales, reference, seed, max_epochs, patience, checkpoint: Path | None):
    set_seed(seed)
    device = torch.device("cuda")
    model = ConstrainedUNet(config.time_mode, scales["mass_scale"], scales["current_scale"], config.base_channels).to(device)
    train_ds = Stage2Dataset(store, train_ids, train_idx)
    val_train_ds = Stage2Dataset(store, val_ids, train_idx)
    val_held_ds = Stage2Dataset(store, val_ids, held_idx)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    val_train_loader = DataLoader(val_train_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    val_held_loader = DataLoader(val_held_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    amp = torch.amp.GradScaler("cuda")
    best_score, best_state, stale = float("inf"), None, 0
    history = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            micro = batch["micro"].to(device, non_blocking=True)
            tau = batch["tau"].to(device, non_blocking=True)
            pf0 = batch["pf0"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            mass = batch["mass"].to(device, non_blocking=True)
            current = batch["current"].to(device, non_blocking=True)
            mask = batch["current_mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred_delta, pm, pc = model(micro, tau)
                loss = field_training_loss(pred_delta, delta, pf0, reference, config) + 0.25 * scalar_training_loss(pm, pc, mass, current, mask, scales, config.scalar_transform)
            amp.scale(loss).backward()
            amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            amp.step(opt)
            amp.update()
            total += float(loss.detach()) * len(micro)
        sched.step()
        val_train = evaluate_loss(model, val_train_loader, device, reference, config, scales)
        val_held = evaluate_loss(model, val_held_loader, device, reference, config, scales)
        score = 0.35 * val_train + 0.65 * val_held
        history.append({"epoch": epoch, "train_loss": total / len(train_ds), "val_training_grid_loss": val_train, "val_heldout_loss": val_held, "checkpoint_score": score})
        print(json.dumps({"event": "stage2_epoch", "seed": seed, "config": config.__dict__, **history[-1]}), flush=True)
        if score < best_score - 1e-5:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.to(device).eval()
    meta = {"seed": seed, "epochs": len(history), "best_checkpoint_score": best_score, "wall_s": time.time() - started, "parameters": sum(p.numel() for p in model.parameters())}
    if checkpoint is not None:
        torch.save({"state_dict": best_state, "config": config.__dict__, "scales": scales, "reference": reference, "seed": seed}, checkpoint)
        pd.DataFrame(history).to_csv(checkpoint.with_suffix(".history.csv"), index=False)
    return model, meta


def evaluate_model(model, store, run_ids, indices, model_name, seed, save_fields_for: str | None = None):
    ds = Stage2Dataset(store, run_ids, indices)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda")
    y = np.linspace(store.header.ymin, store.header.ymax, store.header.ny)
    rows, saved = [], {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            micro = batch["micro"].to(device, non_blocking=True)
            tau = batch["tau"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred_delta_tensor, pm, pc = model(micro, tau)
            pred_delta_array = pred_delta_tensor.float().cpu().numpy()[:, 0]
            pm = pm.float().cpu().numpy()
            pc = pc.float().cpu().numpy()
            for bi, run_id in enumerate(batch["run_id"]):
                ti = int(batch["time_index"][bi])
                pf0 = store.phase0(run_id)
                true_pf = store.phase(run_id, ti)
                pred_pf = np.clip(pf0 + pred_delta_array[bi], 0.0, 1.0)
                metrics = phase_metrics(pred_pf, true_pf, pf0, y)
                if store.times[ti] == 0:
                    metrics["delta_phase_relative_l2"] = float("nan")
                teacher = store.scalars(run_id, ti)
                ia, ib = float(pc[bi]), float(-pc[bi])
                row = {
                    "model": model_name,
                    "seed": seed,
                    "run_id": run_id,
                    "time_index": ti,
                    "time_h": float(store.hours[ti]),
                    "time_group": group_name(float(store.hours[ti])),
                    "f_beta": float(store.desc.loc[run_id, "f_beta_actual"]),
                    "topology": str(store.desc.loc[run_id, "topology"]),
                    **metrics,
                    "true_mass": float(teacher[0]),
                    "pred_mass": float(pm[bi]),
                    "true_I_alpha": float(teacher[1]),
                    "pred_I_alpha": ia,
                    "true_I_beta": float(teacher[2]),
                    "pred_I_beta": ib,
                    "closure_pred": float(abs(ia + ib) / (abs(ia) + abs(ib) + 1e-30)),
                    "closure_teacher": float(store.metrics(run_id).iloc[ti].current_closure_error),
                }
                rows.append(row)
                if save_fields_for == run_id:
                    saved[(run_id, ti)] = pred_pf
    return pd.DataFrame(rows), saved


def phase_selection_score(frame: pd.DataFrame) -> dict[str, float]:
    result = {}
    scores = []
    for group, weight in [("training_grid", 0.35), ("heldout_interpolation", 0.65)]:
        f = frame[frame.time_group == group]
        delta = float(f.delta_phase_relative_l2.mean())
        depth = float(f.max_depth_error_m.mean() / max(f.true_max_depth_m.abs().mean(), 1e-12))
        composite = delta + 0.25 * depth
        result[f"{group}_delta_relative_l2"] = delta
        result[f"{group}_max_depth_normalized_error"] = depth
        result[f"{group}_composite"] = composite
        scores.append(weight * composite)
    result["selection_score"] = float(sum(scores))
    return result


def scalar_selection_score(frame: pd.DataFrame) -> dict[str, float]:
    result, total = {}, 0.0
    for group, weight in [("training_grid", 0.35), ("heldout_interpolation", 0.65)]:
        f = frame[(frame.time_group == group) & (frame.time_h > 0)]
        mass = regression_metrics(f.true_mass.to_numpy(), f.pred_mass.to_numpy())["relative_mae"]
        current = regression_metrics((0.5 * (f.true_I_alpha - f.true_I_beta)).to_numpy(), f.pred_I_alpha.to_numpy())["relative_mae"]
        score = 0.5 * (mass + current)
        result[f"{group}_mass_relative_mae"] = mass
        result[f"{group}_current_relative_mae"] = current
        result[f"{group}_composite"] = score
        total += weight * score
    result["selection_score"] = float(total)
    return result


def run_candidate(store, train_ids, val_ids, train_idx, held_idx, config, scales, reference, name, epochs, patience):
    model, meta = train_model(store, train_ids, val_ids, train_idx, held_idx, config, scales, reference, 8801, epochs, patience, None)
    val_indices = sorted(set(train_idx + held_idx))
    frame, _ = evaluate_model(model, store, val_ids, val_indices, name, 8801)
    result = {"name": name, "config": config.__dict__, "training": meta, "phase": phase_selection_score(frame), "scalar": scalar_selection_score(frame)}
    del model
    torch.cuda.empty_cache()
    return result


def metric_rows(frame: pd.DataFrame, seed: int):
    rows = []
    for group, f in frame.groupby("time_group"):
        phase = {
            "phase_mae": float(f.phase_mae.mean()),
            "phase_rmse": float(f.phase_rmse.mean()),
            "phase_relative_l2": float(f.phase_relative_l2.mean()),
            "delta_phase_relative_l2": float(f.delta_phase_relative_l2.mean()),
            "dice": float(f.dice.mean()),
            "iou": float(f.iou.mean()),
            "mean_depth_error_m": float(f.mean_depth_error_m.mean()),
            "p95_depth_error_m": float(f.p95_depth_error_m.mean()),
            "max_depth_error_m": float(f.max_depth_error_m.mean()),
        }
        rows.append({"seed": seed, "time_group": group, **phase})
        for target, true_col, pred_col in [
            ("mass", "true_mass", "pred_mass"),
            ("I_alpha", "true_I_alpha", "pred_I_alpha"),
            ("I_beta", "true_I_beta", "pred_I_beta"),
        ]:
            use = f if target == "mass" else f[f.time_h > 0]
            rows[-1].update({f"{target}_{k}": v for k, v in regression_metrics(use[true_col].to_numpy(), use[pred_col].to_numpy()).items()})
        nonzero = f[f.time_h > 0]
        rows[-1].update({
            "current_relative_mae_mean": 0.5 * (rows[-1]["I_alpha_relative_mae"] + rows[-1]["I_beta_relative_mae"]),
            "closure_mean": float(nonzero.closure_pred.mean()),
            "closure_p95": float(nonzero.closure_pred.quantile(0.95)),
            "closure_max": float(nonzero.closure_pred.max()),
            "teacher_closure_mean": float(nonzero.closure_teacher.mean()),
        })
    return rows


def aggregate_stage2(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["time_group"]
    value_cols = [c for c in seed_metrics.columns if c not in {"seed", *id_cols}]
    rows = []
    for group, frame in seed_metrics.groupby("time_group"):
        for metric in value_cols:
            rows.append({"source": "Stage2", "task": "phase" if metric in {"phase_mae", "phase_rmse", "phase_relative_l2", "delta_phase_relative_l2", "dice", "iou", "mean_depth_error_m", "p95_depth_error_m", "max_depth_error_m"} else "scalar_or_closure", "model": "constrained_smooth_time_unet", "time_group": group, "metric": metric, "value_mean": float(frame[metric].mean()), "value_sd": float(frame[metric].std(ddof=1)), "n_seeds": len(frame)})
    return pd.DataFrame(rows)


def add_stage1_comparison(stage2: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(STAGE1_ROOT / "PASS_TO_CHAT" / "SURROGATE_STAGE1_MODEL_COMPARISON.csv")
    rows = []
    for _, r in old.iterrows():
        if r.time_group not in {"training_grid", "heldout_interpolation"}:
            continue
        metrics = ["phase_mae", "phase_rmse", "phase_relative_l2", "delta_phase_relative_l2", "dice", "iou", "mean_depth_error_m", "p95_depth_error_m", "max_depth_error_m"] if r.task == "phase" else ["mae", "rmse", "r2", "relative_mae", "mean", "median", "p95", "max", "teacher_mean"]
        for metric in metrics:
            if metric in r and pd.notna(r[metric]):
                rows.append({"source": "Stage1", "task": r.task, "model": r.model, "time_group": r.time_group, "metric": metric, "value_mean": float(r[metric]), "value_sd": np.nan, "n_seeds": 1})
    return pd.concat([pd.DataFrame(rows), stage2], ignore_index=True)


def time_generalization(final_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame in final_frames:
        seed = int(frame.seed.iloc[0])
        for hour, f in frame.groupby("time_h"):
            use = f if hour == 0 else f
            current_use = f[f.time_h > 0]
            row = {
                "seed": seed,
                "time_h": hour,
                "time_group": group_name(hour),
                "phase_mae": float(f.phase_mae.mean()),
                "phase_rmse": float(f.phase_rmse.mean()),
                "phase_relative_l2": float(f.phase_relative_l2.mean()),
                "delta_phase_relative_l2": float(f.delta_phase_relative_l2.mean()) if hour > 0 else np.nan,
                "max_depth_error_m": float(f.max_depth_error_m.mean()),
                "mass_relative_mae": regression_metrics(f.true_mass.to_numpy(), f.pred_mass.to_numpy())["relative_mae"],
                "current_relative_mae": np.nan,
            }
            if len(current_use):
                a = regression_metrics(current_use.true_I_alpha.to_numpy(), current_use.pred_I_alpha.to_numpy())["relative_mae"]
                b = regression_metrics(current_use.true_I_beta.to_numpy(), current_use.pred_I_beta.to_numpy())["relative_mae"]
                row["current_relative_mae"] = 0.5 * (a + b)
            rows.append(row)
    return pd.DataFrame(rows)


def stratified_metrics(final_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame in final_frames:
        seed = int(frame.seed.iloc[0])
        for dim in ["f_beta", "topology", "time_h"]:
            for value, f in frame.groupby(dim):
                nonzero = f[f.time_h > 0]
                current_relative = float("nan")
                if len(nonzero):
                    current_relative = 0.5 * (
                        regression_metrics(nonzero.true_I_alpha.to_numpy(), nonzero.pred_I_alpha.to_numpy())["relative_mae"]
                        + regression_metrics(nonzero.true_I_beta.to_numpy(), nonzero.pred_I_beta.to_numpy())["relative_mae"]
                    )
                rows.append({
                    "seed": seed,
                    "stratifier": dim,
                    "stratum": value,
                    "n": len(f),
                    "phase_mae": float(f.phase_mae.mean()),
                    "delta_phase_relative_l2": float(f.delta_phase_relative_l2.mean()),
                    "max_depth_error_m": float(f.max_depth_error_m.mean()),
                    "mass_relative_mae": regression_metrics(f.true_mass.to_numpy(), f.pred_mass.to_numpy())["relative_mae"],
                    "current_relative_mae": current_relative,
                })
    return pd.DataFrame(rows)


def load_stage1_fields(store: DataStore, run_id: str, indices: list[int]):
    device = torch.device("cuda")
    mean_std = torch.load(STAGE1_ROOT / "checkpoints" / "plain_unet_best.pt", map_location="cpu", weights_only=False)
    plain = PlainUNet().to(device)
    plain.load_state_dict(mean_std["state_dict"])
    film_ck = torch.load(STAGE1_ROOT / "checkpoints" / "time_conditioned_unet_best.pt", map_location="cpu", weights_only=False)
    film = FilmUNet().to(device)
    film.load_state_dict(film_ck["state_dict"])
    output = {"plain": {}, "film": {}}
    micro = torch.from_numpy(store.micro(run_id)[None, None]).to(device)
    with torch.no_grad():
        for ti in indices:
            tau = torch.tensor([[store.times[ti] / 86400.0]], device=device, dtype=torch.float32)
            for name, model in [("plain", plain), ("film", film)]:
                with torch.autocast("cuda", dtype=torch.float16):
                    delta, _ = model(micro, tau)
                output[name][ti] = np.clip(store.phase0(run_id) + delta[0, 0].float().cpu().numpy(), 0, 1)
    return output


def make_plots(store, final_frames, seed0_fields, rep_id, comparison, time_df):
    hours = [3.0, 14.0, 22.0]
    indices = [int(np.flatnonzero(np.isclose(store.hours, h))[0]) for h in hours]
    old_fields = load_stage1_fields(store, rep_id, indices)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)
    for row, (hour, ti) in enumerate(zip(hours, indices)):
        fields = [("COMSOL", store.phase(rep_id, ti)), ("Stage1 plain", old_fields["plain"][ti]), ("Stage1 FiLM", old_fields["film"][ti]), ("Stage2", seed0_fields[(rep_id, ti)])]
        for col, (title, field) in enumerate(fields):
            axes[row, col].imshow(field, origin="lower", extent=[-100, 100, -40, 100], vmin=0, vmax=1, cmap="viridis", aspect="equal")
            axes[row, col].set_title(f"{hour:g} h | {title}")
    fig.savefig(PASS / "01_comsol_stage1_stage2_phase.png", dpi=180)
    plt.close(fig)

    s1_phase = pd.read_csv(STAGE1_ROOT / "test_phase_predictions_metrics.csv")
    s2 = final_frames[0]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    teacher = s2[s2.run_id == rep_id]
    ax.plot(teacher.time_h, teacher.true_max_depth_m * 1e9, "k--", label="COMSOL")
    for model in ["plain_unet", "time_conditioned_unet"]:
        f = s1_phase[(s1_phase.run_id == rep_id) & (s1_phase.model == model)]
        ax.plot(f.time_h, f.pred_max_depth_m * 1e9, label=f"Stage1 {model}")
    f = s2[s2.run_id == rep_id]
    ax.plot(f.time_h, f.pred_max_depth_m * 1e9, label="Stage2 seed0")
    ax.set(xlabel="Time (h)", ylabel="Maximum depth (nm)", title=f"Interface depth: {rep_id}")
    ax.legend(fontsize=8)
    fig.savefig(PASS / "02_interface_depth_trajectory.png", dpi=180)
    plt.close(fig)

    s1_scalar = pd.read_csv(STAGE1_ROOT / "test_scalar_predictions.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(f.time_h, f.true_mass * 1e9, "k--", label="COMSOL")
    for model in ["extra_trees", "plain_unet", "time_conditioned_unet"]:
        g = s1_scalar[(s1_scalar.run_id == rep_id) & (s1_scalar.model == model)]
        ax.plot(g.time_h, g.pred_Faraday_mass_kg_per_m * 1e9, label=f"Stage1 {model}")
    ax.plot(f.time_h, f.pred_mass * 1e9, label="Stage2 seed0")
    ax.set(xlabel="Time (h)", ylabel="Faraday mass (1e-9 kg/m)", title=f"Mass trajectory: {rep_id}")
    ax.legend(fontsize=8)
    fig.savefig(PASS / "03_mass_trajectory.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(f.time_h, f.true_I_alpha * 1e6, "k--", label="COMSOL I_alpha")
    ax.plot(f.time_h, f.pred_I_alpha * 1e6, label="Stage2 I_alpha")
    ax.plot(f.time_h, f.true_I_beta * 1e6, "k:", label="COMSOL I_beta")
    ax.plot(f.time_h, f.pred_I_beta * 1e6, label="Stage2 I_beta")
    ax.set(xlabel="Time (h)", ylabel="Current (uA/m)", title=f"Constrained current: {rep_id}")
    ax.legend(fontsize=8)
    fig.savefig(PASS / "04_current_trajectory.png", dpi=180)
    plt.close(fig)

    held = time_df[time_df.time_group == "heldout_interpolation"].groupby("time_h").agg(mean=("delta_phase_relative_l2", "mean"), sd=("delta_phase_relative_l2", "std")).reset_index()
    old = pd.read_csv(STAGE1_ROOT / "PASS_TO_CHAT" / "SURROGATE_STAGE1_TIME_GENERALIZATION.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for model in ["plain_unet", "time_conditioned_unet", "zero_change"]:
        g = old[(old.task == "phase") & (old.model == model) & (old.time_group == "heldout_interpolation")]
        ax.plot(g.time_h, g.delta_relative_l2, marker="o", label=f"Stage1 {model}")
    ax.errorbar(held.time_h, held["mean"], yerr=held.sd, marker="o", capsize=3, label="Stage2 mean +/- SD")
    ax.set(xlabel="Held-out time (h)", ylabel="Delta-phase relative L2", title="Held-out-time error")
    ax.legend(fontsize=8)
    fig.savefig(PASS / "05_heldout_time_error.png", dpi=180)
    plt.close(fig)

    seed_metrics = pd.read_csv(PASS / "SURROGATE_STAGE2_SEED_STABILITY.csv")
    held_seed = seed_metrics[seed_metrics.time_group == "heldout_interpolation"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for ax, col, title in zip(axes, ["delta_phase_relative_l2", "max_depth_error_m", "mass_relative_mae", "current_relative_mae_mean"], ["Delta phase L2", "Max-depth error (m)", "Mass relative MAE", "Current relative MAE"]):
        ax.scatter(held_seed.seed, held_seed[col])
        ax.axhline(held_seed[col].mean(), color="k", linestyle="--")
        ax.set(xlabel="Training seed", ylabel=title)
    fig.savefig(PASS / "06_five_seed_stability.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for path in [OUT, PASS, CHECKPOINTS, SELECTION_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    store = DataStore(DATA_ROOT)
    train_ids, val_ids = store.ids("train"), store.ids("validation")
    if (len(train_ids), len(val_ids), len(store.ids("test"))) != (72, 18, 18):
        raise RuntimeError("Frozen split mismatch")
    train_idx = time_indices(store, TRAIN_HOURS, include_zero=True)
    held_idx = time_indices(store, HELDOUT_HOURS)
    scales = train_scales(store, train_ids, train_idx)
    reference = dynamic_reference(store, train_ids, train_idx)
    selection_epochs = 1 if args.smoke else 22
    final_epochs = 1 if args.smoke else 35
    patience = 1 if args.smoke else 6
    selection = []

    base_loss = {"interface_lambda": 5.0, "gradient_weight": 0.03}
    for mode in ["T1_tau", "T2_tau_tau2", "T3_mlp"]:
        cfg = Stage2Config(mode, **base_loss, scalar_transform="raw_standardized")
        selection.append(run_candidate(store, train_ids, val_ids, train_idx, held_idx, cfg, scales, reference, f"time_{mode}", selection_epochs, patience))
    chosen_time = min(selection[-3:], key=lambda x: x["phase"]["selection_score"])["config"]["time_mode"]

    loss_candidates = [(2.0, 0.01), (5.0, 0.03), (8.0, 0.05)]
    loss_results = []
    for lam, grad in loss_candidates:
        cfg = Stage2Config(chosen_time, lam, grad, "raw_standardized")
        result = run_candidate(store, train_ids, val_ids, train_idx, held_idx, cfg, scales, reference, f"loss_l{lam}_g{grad}", selection_epochs, patience)
        selection.append(result)
        loss_results.append(result)
    chosen_loss = min(loss_results, key=lambda x: x["phase"]["selection_score"])["config"]

    scalar_results = []
    for transform in ["raw_standardized", "asinh_scaled"]:
        cfg = Stage2Config(chosen_time, chosen_loss["interface_lambda"], chosen_loss["gradient_weight"], transform)
        result = run_candidate(store, train_ids, val_ids, train_idx, held_idx, cfg, scales, reference, f"scalar_{transform}", selection_epochs, patience)
        selection.append(result)
        scalar_results.append(result)
    chosen_scalar = min(scalar_results, key=lambda x: x["scalar"]["selection_score"])["config"]["scalar_transform"]
    frozen = Stage2Config(chosen_time, chosen_loss["interface_lambda"], chosen_loss["gradient_weight"], chosen_scalar)
    write_json(SELECTION_DIR / "STAGE2_VALIDATION_SELECTION.json", {"train_only_scales": scales, "train_only_dynamic_reference": reference, "candidate_results": selection})
    freeze_path = OUT / "FROZEN_STAGE2_CONFIG.json"
    write_json(freeze_path, {"status": "FROZEN_BEFORE_TEST", "config": frozen.__dict__, "selection_source": "validation trajectories at training-grid and held-out times", "test_accessed_during_selection": False, "scales": scales, "dynamic_reference": reference})
    print(json.dumps({"event": "stage2_config_frozen", "config": frozen.__dict__}), flush=True)

    seeds = FINAL_SEEDS[:2] if args.smoke else FINAL_SEEDS
    final_models, training_meta = [], []
    for seed in seeds:
        checkpoint = CHECKPOINTS / f"stage2_seed{seed}.pt" if not args.smoke else None
        model, meta = train_model(store, train_ids, val_ids, train_idx, held_idx, frozen, scales, reference, seed, final_epochs, patience, checkpoint)
        final_models.append(model)
        training_meta.append(meta)

    # Fixed test is first read only after the configuration above has been frozen.
    test_ids = store.ids("test")
    all_idx = list(range(store.header.ntimes))
    rep_id = test_ids[0]
    final_frames, seed0_fields = [], {}
    for seed, model in zip(seeds, final_models):
        frame, saved = evaluate_model(model, store, test_ids, all_idx, "constrained_smooth_time_unet", seed, rep_id if seed == seeds[0] else None)
        final_frames.append(frame)
        if seed == seeds[0]:
            seed0_fields = saved
    detailed = pd.concat(final_frames, ignore_index=True)
    detailed.to_csv(OUT / "stage2_test_predictions.csv", index=False)
    seed_rows = []
    for seed, frame in zip(seeds, final_frames):
        seed_rows.extend(metric_rows(frame, seed))
    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(PASS / "SURROGATE_STAGE2_SEED_STABILITY.csv", index=False)
    stage2_comparison = aggregate_stage2(seed_metrics)
    comparison = add_stage1_comparison(stage2_comparison)
    comparison.to_csv(PASS / "SURROGATE_STAGE2_MODEL_COMPARISON.csv", index=False)
    time_df = time_generalization(final_frames)
    time_df.to_csv(PASS / "SURROGATE_STAGE2_TIME_GENERALIZATION.csv", index=False)
    stratified = stratified_metrics(final_frames)
    stratified.to_csv(PASS / "SURROGATE_STAGE2_STRATIFIED_METRICS.csv", index=False)
    if not args.smoke:
        make_plots(store, final_frames, seed0_fields, rep_id, comparison, time_df)

    held = seed_metrics[seed_metrics.time_group == "heldout_interpolation"]
    train_grid = seed_metrics[seed_metrics.time_group == "training_grid"]
    old = pd.read_csv(STAGE1_ROOT / "PASS_TO_CHAT" / "SURROGATE_STAGE1_MODEL_COMPARISON.csv")
    def old_value(model, task, metric):
        r = old[(old.model == model) & (old.task == task) & (old.time_group == "heldout_interpolation")]
        return float(r.iloc[0][metric])
    report = {
        "title": "Formal V1.1 Surrogate Stage 2 constrained smooth-time surrogate",
        "status": "SMOKE" if args.smoke else "COMPLETE",
        "frozen_config": frozen.__dict__,
        "selection": {"rule": "validation only; 35% training-grid + 65% held-out-time composite", "candidate_results": selection, "frozen_before_test": True},
        "data_integrity": {"trajectories": 108, "split": {"train": 72, "validation": 18, "test": 18}, "split_unit": "complete trajectory", "new_comsol_data": False, "test_used_for_selection": False},
        "analytic_constraints": {"delta_pf_t0_exact_zero": True, "mass_t0_exact_zero": True, "mass_nonnegative": True, "current_magnitude_positive": True, "I_alpha": "+I", "I_beta": "-I", "surrogate_closure": "machine zero by construction, not independently learned"},
        "train_only_scaling": scales,
        "train_only_dynamic_reference": reference,
        "training_seeds": training_meta,
        "test_summary": {
            "training_grid_mean_sd": train_grid.select_dtypes(include=[np.number]).agg(["mean", "std"]).replace({np.nan: None}).to_dict(),
            "heldout_mean_sd": held.select_dtypes(include=[np.number]).agg(["mean", "std"]).replace({np.nan: None}).to_dict(),
            "stage1_heldout": {
                "plain_phase_delta_relative_l2": old_value("plain_unet", "phase", "delta_phase_relative_l2"),
                "film_phase_delta_relative_l2": old_value("time_conditioned_unet", "phase", "delta_phase_relative_l2"),
                "plain_mass_relative_mae": old_value("plain_unet", "Faraday_mass_kg_per_m", "relative_mae"),
                "extra_trees_I_alpha_relative_mae": old_value("extra_trees", "I_alpha_A_per_m", "relative_mae"),
                "extra_trees_I_beta_relative_mae": old_value("extra_trees", "I_beta_A_per_m", "relative_mae"),
            },
        },
        "decisions": {
            "stage2_improves_heldout_phase_over_stage1_plain": bool(held.delta_phase_relative_l2.mean() < old_value("plain_unet", "phase", "delta_phase_relative_l2")),
            "stage2_improves_heldout_max_depth": bool(held.max_depth_error_m.mean() < old_value("plain_unet", "phase", "max_depth_error_m")),
            "stage2_improves_heldout_mass": bool(held.mass_relative_mae.mean() < old_value("plain_unet", "Faraday_mass_kg_per_m", "relative_mae")),
            "stage2_current_beats_extra_trees": bool(held.I_alpha_relative_mae.mean() < old_value("extra_trees", "I_alpha_A_per_m", "relative_mae") and held.I_beta_relative_mae.mean() < old_value("extra_trees", "I_beta_A_per_m", "relative_mae")),
            "learning_curve_rerun": False,
            "stage3_started": False,
        },
        "prohibitions_honored": {"new_comsol": False, "split_change": False, "test_selection": False, "FNO": False, "Transformer": False, "ensemble": False, "q_training": False, "raw_phi_training": False},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "files": [p.name for p in sorted(PASS.glob("*"))],
    }
    write_json(PASS / "SURROGATE_STAGE2_REPORT.json", report)
    print(json.dumps({"event": "stage2_complete", "pass_to_chat": str(PASS), "decisions": report["decisions"]}), flush=True)


if __name__ == "__main__":
    main()
