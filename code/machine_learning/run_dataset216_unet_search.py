from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from surrogate_stage1 import BinaryHeader, ConvBlock, crossings, phase_metrics
from surrogate_stage2 import Stage2Config, dynamic_reference, field_training_loss, scalar_training_loss

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AZ91D_DATA_ROOT", REPO_ROOT / "data"))
FREEZE = Path(os.environ.get("AZ91D_FREEZE_ROOT", DATA_ROOT / "dataset216_freeze"))
OUT = Path(os.environ.get("AZ91D_OUTPUT_ROOT", REPO_ROOT / "outputs" / "hyperparameter_search"))
CKPT = OUT / "checkpoints"
STATE_PATH = OUT / "search_state.json"
ACCESS_LOG = OUT / "TEST_ACCESS_GUARD_LOG.txt"
STATUS = OUT / "TRAINING_RUNTIME_STATUS.md"
TRAIN_HOURS = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24]
MAX_EPOCHS, PATIENCE, WEIGHT_DECAY = 35, 6, 1e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_header(path: Path) -> BinaryHeader:
    with path.open("rb") as handle:
        magic, version, nx, ny, nf, nt = struct.unpack(">6i", handle.read(24))
        if magic != 0x415A5632 or version != 3:
            raise ValueError(f"Unsupported fields file: {path}")
        xmin, xmax, ymin, ymax = struct.unpack(">4d", handle.read(32))
        times = list(struct.unpack(f">{nt}d", handle.read(8 * nt)))
    return BinaryHeader(nx, ny, nf, nt, xmin, xmax, ymin, ymax, times, 56 + 8 * nt)


class Store216:
    """Strict reader: only train and validation truth may be opened."""
    def __init__(self) -> None:
        manifest = pd.read_csv(FREEZE / "COMBINED216_PRODUCTION_MANIFEST.csv")
        self.rows = manifest.set_index("production_id")
        self.allowed = set(self.rows.index[self.rows.split_role.isin(["train", "validation"])])
        self.forbidden = set(self.rows.index) - self.allowed
        self.split = manifest[["production_id", "split_role", "f_beta", "morphology", "dataset_generation"]].rename(columns={"production_id": "run_id"})
        first = self.ids("train")[0]
        self.header = read_header(self.field_path(first))
        self.times = np.asarray(self.header.times, dtype=np.float64)
        self.hours = self.times / 3600.0
        self._maps: OrderedDict[str, np.memmap] = OrderedDict()
        self._micro: dict[str, np.ndarray] = {}
        self._metrics: dict[str, pd.DataFrame] = {}
        self._pf0: dict[str, np.ndarray] = {}

    def ids(self, role: str) -> list[str]:
        return self.split.loc[self.split.split_role == role, "run_id"].tolist()

    def _guard(self, run_id: str, resource: str) -> None:
        if run_id in self.allowed:
            return
        ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ACCESS_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{pd.Timestamp.now().isoformat()} BLOCKED {run_id} {resource}\n")
        raise RuntimeError(f"Forbidden {self.rows.loc[run_id, 'split_role']} truth access: {run_id}")

    def field_path(self, run_id: str) -> Path:
        self._guard(run_id, "fields.bin")
        return Path(self.rows.loc[run_id, "trajectory_path"]) / "fields.bin"

    def metric_path(self, run_id: str) -> Path:
        self._guard(run_id, "metrics.csv")
        return Path(self.rows.loc[run_id, "trajectory_path"]) / "metrics.csv"

    def micro_path(self, run_id: str) -> Path:
        self._guard(run_id, "smooth_microstructure")
        return Path(self.rows.loc[run_id, "microstructure_path"])

    def mmap(self, run_id: str) -> np.memmap:
        self._guard(run_id, "fields.bin")
        if run_id in self._maps:
            self._maps.move_to_end(run_id)
            return self._maps[run_id]
        header = read_header(self.field_path(run_id))
        if asdict(header) != asdict(self.header):
            raise ValueError(f"Header mismatch: {run_id}")
        mapping = np.memmap(self.field_path(run_id), dtype=">f8", mode="r", offset=header.offset,
                            shape=(header.nfields, header.ntimes, header.ny, header.nx))
        self._maps[run_id] = mapping
        if len(self._maps) > 8:
            self._maps.popitem(last=False)
        return mapping

    def phase(self, run_id: str, ti: int) -> np.ndarray:
        return np.asarray(self.mmap(run_id)[3, ti], dtype=np.float32)

    def phase0(self, run_id: str) -> np.ndarray:
        if run_id not in self._pf0:
            self._pf0[run_id] = self.phase(run_id, 0)
        return self._pf0[run_id]

    def micro(self, run_id: str) -> np.ndarray:
        if run_id not in self._micro:
            source = np.load(self.micro_path(run_id), allow_pickle=False).astype(np.float32)
            ys = np.linspace(self.header.ymin, self.header.ymax, self.header.ny)
            source_y = np.linspace(-25e-6, 0.0, source.shape[0])
            full = np.full((self.header.ny, self.header.nx), -1.0, dtype=np.float32)
            metal = ys <= 0.0
            for ix in range(self.header.nx):
                full[metal, ix] = np.interp(ys[metal], source_y, source[:, ix], left=source[0, ix], right=source[-1, ix])
            self._micro[run_id] = full
        return self._micro[run_id]

    def metrics(self, run_id: str) -> pd.DataFrame:
        if run_id not in self._metrics:
            frame = pd.read_csv(self.metric_path(run_id))
            if len(frame) != 27 or not np.allclose(frame.time_s, self.times):
                raise ValueError(f"Metrics time mismatch: {run_id}")
            self._metrics[run_id] = frame
        return self._metrics[run_id]

    def scalars(self, run_id: str, ti: int) -> np.ndarray:
        row = self.metrics(run_id).iloc[ti]
        return np.asarray([row.Faraday_mass_kg_per_m, row.I_alpha_A_per_m, row.I_beta_A_per_m], dtype=np.float32)


class PhaseDataset(Dataset):
    def __init__(self, store: Store216, ids: list[str], indices: list[int]):
        self.store, self.items = store, [(run_id, ti) for run_id in ids for ti in indices]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        run_id, ti = self.items[index]
        pf0 = self.store.phase0(run_id)
        pf = self.store.phase(run_id, ti)
        scalar = self.store.scalars(run_id, ti)
        return {
            "micro": torch.from_numpy(self.store.micro(run_id)[None]),
            "pf0": torch.from_numpy(pf0[None]),
            "delta": torch.from_numpy((pf - pf0)[None]),
            "tau": torch.tensor([self.store.times[ti] / 86400.0], dtype=torch.float32),
            "mass": torch.tensor(scalar[0], dtype=torch.float32),
            "current": torch.tensor(0.5 * (scalar[1] - scalar[2]), dtype=torch.float32),
            "current_mask": torch.tensor(1.0, dtype=torch.float32),
            "run_id": run_id,
            "time_index": ti,
        }


class TimeUNet(torch.nn.Module):
    """Existing constrained U-Net, with analytic tau multiplication optionally disabled."""
    def __init__(self, mass_scale: float, current_scale: float, base: int, constrained: bool):
        super().__init__()
        self.constrained = constrained
        self.register_buffer("mass_scale", torch.tensor(float(mass_scale)))
        self.register_buffer("current_scale", torch.tensor(float(current_scale)))
        self.pool = torch.nn.MaxPool2d(2)
        self.e1 = ConvBlock(2, base); self.e2 = ConvBlock(base, base * 2); self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8); self.mid = ConvBlock(base * 8, base * 16)
        self.d4 = ConvBlock(base * 24, base * 8); self.d3 = ConvBlock(base * 12, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2); self.d1 = ConvBlock(base * 3, base)
        self.field = torch.nn.Conv2d(base, 1, 1)
        self.scalar = torch.nn.Sequential(torch.nn.Linear(base * 16 + 1, base * 8), torch.nn.SiLU(), torch.nn.Linear(base * 8, 2))

    def forward(self, micro: torch.Tensor, tau: torch.Tensor):
        x = F.pad(micro, (0, 0, 5, 5), value=-1.0)
        tmap = tau[:, :, None, None].expand(-1, 1, x.shape[-2], x.shape[-1])
        e1 = self.e1(torch.cat([x, tmap], dim=1)); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3)); mid = self.mid(self.pool(e4))
        d4 = self.d4(torch.cat([F.interpolate(mid, size=e4.shape[-2:], mode="bilinear", align_corners=False), e4], dim=1))
        d3 = self.d3(torch.cat([F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        raw = self.field(d1)[..., 5:-5, :]
        delta = tau[:, :, None, None] * raw if self.constrained else raw
        latent = F.adaptive_avg_pool2d(mid, 1).flatten(1)
        scalar = self.scalar(torch.cat([latent, tau], dim=1))
        return delta, tau[:, 0] * F.softplus(scalar[:, 0]) * self.mass_scale, F.softplus(scalar[:, 1]) * self.current_scale


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def indices(store: Store216) -> list[int]:
    result = []
    for hour in TRAIN_HOURS:
        found = np.flatnonzero(np.isclose(store.hours, hour))
        if len(found) != 1:
            raise RuntimeError(f"Missing training time {hour} h")
        result.append(int(found[0]))
    return result


def scales(store: Store216, ids: list[str], time_indices: list[int]) -> dict[str, float]:
    mass, current = [], []
    for run_id in ids:
        for ti in time_indices:
            values = store.scalars(run_id, ti)
            mass.append(float(values[0])); current.append(float(0.5 * (values[1] - values[2])))
    mass_a, current_a = np.asarray(mass), np.asarray(current)
    return {"mass_scale": float(mass_a.max()), "current_scale": float(np.median(current_a)), "mass_mean": float(mass_a.mean()), "mass_std": float(max(mass_a.std(), 1e-30)), "current_mean": float(current_a.mean()), "current_std": float(max(current_a.std(), 1e-30))}


def state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"completed": {}, "started": pd.Timestamp.now().isoformat()}


def save_state(value: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def runtime(label: str, done: int, planned: int, epoch: int | str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = subprocess_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
    STATUS.write_text(f"# Dataset216 U-Net search status\n\n- Updated: {pd.Timestamp.now().isoformat()}\n- Stage/job: {label}\n- Epoch: {epoch}\n- Completed jobs: {done}\n- Planned jobs: {planned}\n- GPU status: {gpu}\n- historical_audit_test truth accessed: NO\n- pristine_final_test truth accessed: NO\n", encoding="utf-8")


def subprocess_output(command: list[str]) -> str:
    import subprocess
    try:
        return subprocess.check_output(command, text=True, timeout=5).strip()
    except Exception:
        return "unavailable"


def validation_metrics(model: TimeUNet, store: Store216, ids: list[str], time_indices: list[int], batch_size: int) -> dict[str, float]:
    model.eval(); device = torch.device("cuda"); loader = DataLoader(PhaseDataset(store, ids, time_indices), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    y = np.linspace(store.header.ymin, store.header.ymax, store.header.ny)
    records = []
    with torch.no_grad():
        for batch in loader:
            micro, tau = batch["micro"].to(device), batch["tau"].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                pred, _, _ = model(micro, tau)
            pred = pred.float().cpu().numpy()[:, 0]
            for bi, run_id in enumerate(batch["run_id"]):
                ti = int(batch["time_index"][bi]); pf0 = store.phase0(run_id); truth = store.phase(run_id, ti); estimate = np.clip(pf0 + pred[bi], 0, 1)
                metric = phase_metrics(estimate, truth, pf0, y)
                ref, true_i, pred_i = crossings(pf0, y), crossings(truth, y, crossings(pf0, y)), crossings(estimate, y, crossings(pf0, y))
                valid = np.isfinite(true_i) & np.isfinite(pred_i)
                metric["interface_mae_m"] = float(np.mean(np.abs(pred_i[valid] - true_i[valid]))) if valid.any() else float("nan")
                records.append(metric)
    frame = pd.DataFrame(records)
    return {"delta_phase_relative_l2": float(frame.delta_phase_relative_l2.mean()), "interface_mae_nm": float(frame.interface_mae_m.mean() * 1e9), "max_depth_mae_nm": float(frame.max_depth_error_m.mean() * 1e9)}


def train_job(store: Store216, train_ids: list[str], val_ids: list[str], config: dict, seed: int, label: str, constrained: bool, planned: int) -> dict:
    set_seed(seed); time_indices = indices(store); train_scales = scales(store, train_ids, time_indices)
    cfg = Stage2Config("T1_tau", config["interface_lambda"], config["gradient_weight"], "raw_standardized", 4.0, config["base_channels"])
    reference = dynamic_reference(store, train_ids, time_indices)
    model = TimeUNet(train_scales["mass_scale"], train_scales["current_scale"], config["base_channels"], constrained).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    amp = torch.amp.GradScaler("cuda")
    train_loader = DataLoader(PhaseDataset(store, train_ids, time_indices), batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(PhaseDataset(store, val_ids, time_indices), batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    best, best_state, best_epoch, stale, history = float("inf"), None, 0, 0, []
    started = time.time(); last_path, best_path = CKPT / f"{label}_seed{seed}_last.pt", CKPT / f"{label}_seed{seed}_best.pt"
    if last_path.exists():
        resume = torch.load(last_path, map_location="cuda", weights_only=False)
        model.load_state_dict(resume["state_dict"]); optimizer.load_state_dict(resume["optimizer"]); scheduler.load_state_dict(resume["scheduler"]); amp.load_state_dict(resume["amp"])
        best, best_state, best_epoch, stale, history = resume["best"], resume["best_state"], resume["best_epoch"], resume["stale"], resume["history"]
    for epoch in range(len(history) + 1, MAX_EPOCHS + 1):
        model.train(); total = 0.0
        for batch in train_loader:
            micro, tau, pf0, delta = (batch[key].cuda(non_blocking=True) for key in ("micro", "tau", "pf0", "delta"))
            mass, current, mask = (batch[key].cuda(non_blocking=True) for key in ("mass", "current", "current_mask"))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred, pm, pc = model(micro, tau)
                loss = field_training_loss(pred, delta, pf0, reference, cfg) + 0.25 * scalar_training_loss(pm, pc, mass, current, mask, train_scales, "raw_standardized")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {label} seed {seed}")
            amp.scale(loss).backward(); amp.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); amp.step(optimizer); amp.update(); total += float(loss.detach()) * len(micro)
        scheduler.step(); model.eval(); val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                micro, tau, pf0, delta = (batch[key].cuda(non_blocking=True) for key in ("micro", "tau", "pf0", "delta"))
                mass, current, mask = (batch[key].cuda(non_blocking=True) for key in ("mass", "current", "current_mask"))
                with torch.autocast("cuda", dtype=torch.float16):
                    pred, pm, pc = model(micro, tau)
                    loss = field_training_loss(pred, delta, pf0, reference, cfg) + 0.25 * scalar_training_loss(pm, pc, mass, current, mask, train_scales, "raw_standardized")
                val_total += float(loss) * len(micro)
        val_loss = val_total / len(val_loader.dataset); history.append({"epoch": epoch, "train_loss": total / len(train_loader.dataset), "validation_loss": val_loss})
        if val_loss < best - 1e-5:
            best, best_state, best_epoch, stale = val_loss, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, epoch, 0
        else:
            stale += 1
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "amp": amp.state_dict(), "best": best, "best_state": best_state, "best_epoch": best_epoch, "stale": stale, "history": history}, last_path)
        runtime(label, len(state()["completed"]), planned, epoch)
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state); torch.save({"state_dict": best_state, "config": config, "seed": seed, "best_epoch": best_epoch, "best_validation_loss": best, "scales": train_scales, "dynamic_reference": reference, "constrained": constrained}, best_path)
    pd.DataFrame(history).to_csv(CKPT / f"{label}_seed{seed}_history.csv", index=False)
    metrics = validation_metrics(model, store, val_ids, time_indices, config["batch_size"])
    return {"status": "OK", "label": label, "seed": seed, "constrained": constrained, **config, **metrics, "best_epoch": best_epoch, "best_validation_loss": best, "parameters": sum(p.numel() for p in model.parameters()), "wall_s": time.time() - started, "checkpoint": str(best_path)}


def run(store: Store216, train_ids: list[str], val_ids: list[str], config: dict, seed: int, label: str, constrained: bool, planned: int) -> dict:
    key = f"{label}|seed{seed}|{'constrained' if constrained else 'plain'}"
    saved = state()
    if key in saved["completed"]:
        return saved["completed"][key]
    try:
        result = train_job(store, train_ids, val_ids, config, seed, label, constrained, planned)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); result = {"status": "OOM", "label": label, "seed": seed, "constrained": constrained, **config}
    saved = state(); saved["completed"][key] = result; save_state(saved); torch.cuda.empty_cache()
    return result


def average(rows: list[dict]) -> dict:
    usable = [row for row in rows if row["status"] == "OK"]
    if not usable:
        return {"status": "NO_VALID_RUN"}
    fields = ["delta_phase_relative_l2", "interface_mae_nm", "max_depth_mae_nm", "best_epoch", "parameters"]
    value = {"status": "OK", "n_seeds": len(usable)}
    for field in fields:
        data = np.asarray([float(row[field]) for row in usable])
        value[f"{field}_mean"] = float(data.mean()); value[f"{field}_sd"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
    return value


def config_label(config: dict) -> str:
    return f"c{config['base_channels']}_lr{config['learning_rate']:.0e}_b{config['batch_size']}_i{config['interface_lambda']:g}_g{config['gradient_weight']:g}".replace("+", "")


def summarize(rows: list[dict], ref: dict) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    output = []
    for label, group in grouped.items():
        base = {key: group[0][key] for key in ("label", "base_channels", "learning_rate", "batch_size", "interface_lambda", "gradient_weight")}
        agg = average(group); base.update(agg)
        if agg["status"] == "OK":
            base["score"] = 0.60 * agg["delta_phase_relative_l2_mean"] / ref["delta_phase_relative_l2_mean"] + 0.25 * agg["interface_mae_nm_mean"] / ref["interface_mae_nm_mean"] + 0.15 * agg["max_depth_mae_nm_mean"] / ref["max_depth_mae_nm_mean"]
            base["hard_constraint_pass"] = bool(agg["interface_mae_nm_mean"] <= 1.15 * ref["interface_mae_nm_mean"] and agg["max_depth_mae_nm_mean"] <= 1.15 * ref["max_depth_mae_nm_mean"])
        else:
            base["score"], base["hard_constraint_pass"] = float("inf"), False
        output.append(base)
    return sorted(output, key=lambda row: (not row["hard_constraint_pass"], row["score"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(exist_ok=True)
    ACCESS_LOG.write_text("# Final-test guard log\n\nNo historical or pristine truth access is permitted during search.\n", encoding="utf-8") if not ACCESS_LOG.exists() else None
    store = Store216(); train_ids, val_ids = store.ids("train"), store.ids("validation")
    if len(train_ids) != 144 or len(val_ids) != 36 or len(store.ids("historical_audit_test")) != 18 or len(store.ids("pristine_final_test")) != 18:
        raise RuntimeError("Frozen 144/36/18/18 split mismatch")
    time_indices = indices(store)
    reference_config = {"base_channels": 16, "learning_rate": 2e-4, "batch_size": 2, "interface_lambda": 5.0, "gradient_weight": 0.03}
    planned = 36 + 54 + 5 + 12 + 5 + 3
    reference_label = config_label(reference_config)
    baseline = [run(store, train_ids, val_ids, reference_config, seed, reference_label, True, planned) for seed in (0, 1)]
    ref = average(baseline)
    if ref["status"] != "OK":
        raise RuntimeError("Baseline reference did not complete")

    # The frozen constrained form must be identically zero at the analytic origin.
    probe = TimeUNet(1.0, 1.0, 16, True).cuda().eval()
    with torch.no_grad():
        delta0, mass0, _ = probe(torch.from_numpy(store.micro(train_ids[0])[None, None]).cuda(), torch.zeros((1, 1), device="cuda"))
    t0 = {"max_abs_delta_phase": float(delta0.abs().max().cpu()), "max_abs_faraday_mass": float(mass0.abs().max().cpu())}
    if t0["max_abs_delta_phase"] != 0.0 or t0["max_abs_faraday_mass"] != 0.0:
        raise RuntimeError(f"Analytic t=0 constraint failed: {t0}")
    (OUT / "ANALYTIC_T0_CHECK.json").write_text(json.dumps(t0, indent=2) + "\n", encoding="utf-8")
    del probe

    stage_a_rows = []
    for base in (16, 24, 32):
        for lr in (1e-4, 2e-4, 4e-4):
            for batch in (2, 4):
                cfg = {"base_channels": base, "learning_rate": lr, "batch_size": batch, "interface_lambda": 5.0, "gradient_weight": 0.03}
                label = config_label(cfg)
                for seed in (0, 1): stage_a_rows.append(run(store, train_ids, val_ids, cfg, seed, label, True, planned))
    stage_a = summarize(stage_a_rows, ref); write_csv(OUT / "STAGE_A_GRID_RESULTS.csv", stage_a)
    top3 = [row for row in stage_a if row["hard_constraint_pass"]][:3]
    if len(top3) != 3: raise RuntimeError("Stage A did not yield three admissible configurations")

    stage_b_rows = []
    for candidate in top3:
        for interface_lambda in (2.0, 5.0, 8.0):
            for gradient_weight in (0.01, 0.03, 0.06):
                cfg = {"base_channels": candidate["base_channels"], "learning_rate": candidate["learning_rate"], "batch_size": candidate["batch_size"], "interface_lambda": interface_lambda, "gradient_weight": gradient_weight}
                label = "stageb_" + config_label(cfg)
                for seed in (0, 1): stage_b_rows.append(run(store, train_ids, val_ids, cfg, seed, label, True, planned))
    stage_b = summarize(stage_b_rows, ref); write_csv(OUT / "STAGE_B_LOSS_GRID_RESULTS.csv", stage_b)
    top5 = [row for row in stage_b if row["hard_constraint_pass"]][:5]
    if len(top5) != 5: raise RuntimeError("Stage B did not yield five admissible configurations")
    replication_rows = []
    for candidate in top5:
        cfg = {key: candidate[key] for key in ("base_channels", "learning_rate", "batch_size", "interface_lambda", "gradient_weight")}; label = candidate["label"]
        for seed in (0, 1, 2): replication_rows.append(run(store, train_ids, val_ids, cfg, seed, label, True, planned))
    replicated = summarize(replication_rows, ref); write_csv(OUT / "TOP5_CONFIG_REPLICATION.csv", replication_rows)
    final = next(row for row in replicated if row["hard_constraint_pass"])
    final_cfg = {key: final[key] for key in ("base_channels", "learning_rate", "batch_size", "interface_lambda", "gradient_weight")}

    yaml = "\n".join(["time_mode: T1_tau", "analytic_time_constraint: delta_phase_hat = tau * F_theta(M, tau)", *[f"{key}: {value}" for key, value in final_cfg.items()], f"weight_decay: {WEIGHT_DECAY}", f"max_epochs: {MAX_EPOCHS}", f"early_stopping_patience: {PATIENCE}", "optimizer: AdamW", "scheduler: CosineAnnealingLR", "selection_split: validation", "selection_time_grid_h: [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24]"]) + "\n"
    config_path = OUT / "FINAL_UNET_CONFIG.yaml"; config_path.write_text(yaml, encoding="utf-8"); (OUT / "FINAL_UNET_CONFIG_HASH.txt").write_text(sha256(config_path) + "\n", encoding="utf-8")
    (OUT / "HYPERPARAMETER_SEARCH_COMPLETE.flag").write_text("validation-only selection complete\n", encoding="utf-8")

    cells = {(fb, topology): sorted(frame.index.tolist(), key=lambda pid: (store.rows.loc[pid, "dataset_generation"], int(store.rows.loc[pid, "seed"]))) for (fb, topology), frame in store.rows.loc[train_ids].groupby(["f_beta", "morphology"])}
    subset_rows, curve_rows = [], []
    for size in (36, 72, 108, 144):
        per_cell = size // 9; selected = sorted(pid for ids in cells.values() for pid in ids[:per_cell])
        for pid in selected: subset_rows.append({"n_train": size, "production_id": pid})
        for seed in (0, 1, 2):
            record = run(store, selected, val_ids, final_cfg, seed, f"learning_n{size}", True, planned)
            curve_rows.append({"n_train": size, **record})
    write_csv(OUT / "LEARNING_CURVE_SUBSETS.csv", subset_rows); write_csv(OUT / "LEARNING_CURVE_36_72_108_144.csv", curve_rows)
    curve = pd.DataFrame([row for row in curve_rows if row["status"] == "OK"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for axis, metric, title in zip(axes, ["delta_phase_relative_l2", "interface_mae_nm", "max_depth_mae_nm"], ["Delta-phase relative L2", "Interface MAE (nm)", "Maximum-depth MAE (nm)"]):
        values = curve.groupby("n_train")[metric].agg(["mean", "std"]); axis.errorbar(values.index, values["mean"], yerr=values["std"], marker="o", capsize=3); axis.set(xlabel="Training trajectories", ylabel=title)
    fig.savefig(OUT / "LEARNING_CURVE_36_72_108_144.png", dpi=180); plt.close(fig)

    final_rows = [run(store, train_ids, val_ids, final_cfg, seed, "final_constrained", True, planned) for seed in range(5)]
    write_csv(OUT / "FINAL_5SEED_VALIDATION_METRICS.csv", final_rows)
    ablation = [run(store, train_ids, val_ids, final_cfg, seed, "baseline_unconstrained", False, planned) for seed in range(3)]
    write_csv(OUT / "BASELINE_ABLATION_VALIDATION.csv", ablation)

    checkpoints = []
    for row in final_rows:
        if row["status"] == "OK": checkpoints.append({"seed": row["seed"], "checkpoint": row["checkpoint"], "sha256": sha256(Path(row["checkpoint"])), "best_epoch": row["best_epoch"]})
    write_csv(OUT / "LOCKED_CHECKPOINT_MANIFEST.csv", checkpoints)
    (OUT / "FINAL_MODEL_EVALUATION_PROTOCOL.md").write_text("# Final-model evaluation protocol\n\nBefore opening the pristine final test, the main result will be reported as the mean ± SD across five independently trained constrained U-Net seeds. A separate five-seed mean prediction field will be reported as the ensemble prediction. Best-test-seed reporting is prohibited.\n", encoding="utf-8")
    final_avg, ablation_avg = average(final_rows), average(ablation)
    report = "\n".join(["# Dataset216 U-Net hyperparameter search report", "", "## Frozen data boundary", "- Training trajectories: 144", "- Validation trajectories: 36", "- historical_audit_test truth access: none", "- pristine_final_test truth access: none", "", "## Reference and final validation metrics", "| Metric | Reference | Final constrained | Plain ablation |", "| --- | ---: | ---: | ---: |", f"| Delta-phase relative L2 | {ref['delta_phase_relative_l2_mean']:.6g} | {final_avg['delta_phase_relative_l2_mean']:.6g} ± {final_avg['delta_phase_relative_l2_sd']:.3g} | {ablation_avg['delta_phase_relative_l2_mean']:.6g} ± {ablation_avg['delta_phase_relative_l2_sd']:.3g} |", f"| Interface MAE (nm) | {ref['interface_mae_nm_mean']:.6g} | {final_avg['interface_mae_nm_mean']:.6g} ± {final_avg['interface_mae_nm_sd']:.3g} | {ablation_avg['interface_mae_nm_mean']:.6g} ± {ablation_avg['interface_mae_nm_sd']:.3g} |", f"| Maximum-depth MAE (nm) | {ref['max_depth_mae_nm_mean']:.6g} | {final_avg['max_depth_mae_nm_mean']:.6g} ± {final_avg['max_depth_mae_nm_sd']:.3g} | {ablation_avg['max_depth_mae_nm_mean']:.6g} ± {ablation_avg['max_depth_mae_nm_sd']:.3g} |", "", "## Final configuration", *[f"- {key}: {value}" for key, value in final_cfg.items()], f"- weight_decay: {WEIGHT_DECAY}", f"- parameters: {int(final_avg['parameters_mean'])}", "", "## Lock", "- Final configuration and five checkpoint hashes are recorded before any pristine-final-test access."])
    (OUT / "HYPERPARAMETER_SEARCH_REPORT.md").write_text(report + "\n", encoding="utf-8")
    code_hash = sha256(Path(__file__)); split_hash = sha256(FREEZE / "COMBINED216_SPLIT_MANIFEST.csv")
    (OUT / "LOCKED_FINAL_MODEL.flag").write_text(f"locked\nconfig_sha256={sha256(config_path)}\nsplit_sha256={split_hash}\ncode_sha256={code_hash}\n", encoding="utf-8")
    runtime("COMPLETE", len(state()["completed"]), planned)
    print("DATASET216_UNET_SEARCH=complete", flush=True)


if __name__ == "__main__":
    main()
