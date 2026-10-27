from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AZ91D_DATA_ROOT", REPO_ROOT / "data"))
PROJECT = Path(os.environ.get("AZ91D_PROJECT_ROOT", DATA_ROOT / "project"))
EXT = Path(os.environ.get("AZ91D_EXTENSION_ROOT", DATA_ROOT / "extension"))
FROZEN = PROJECT / "dataset216_freeze_audit_20260912"
SEARCH = PROJECT / "az91d_216_unet_hyperparameter_search_20260912"
SOURCE = Path(os.environ.get("AZ91D_TRAINING_SCRIPT", Path(__file__).with_name("run_dataset216_unet_search.py")))
OUT = EXT
SIZES = (144, 180, 216, 252, 288)
SEEDS = range(5)


def load_training_module():
    spec = importlib.util.spec_from_file_location("dataset216_training", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.OUT = OUT / "learning_curve_work"
    module.CKPT = module.OUT / "checkpoints"
    module.STATE_PATH = module.OUT / "state.json"
    module.ACCESS_LOG = OUT / "FINAL_TEST_ACCESS_GUARD.log"
    module.STATUS = OUT / "TRAIN288_RUNTIME_STATUS.md"
    module.CKPT.mkdir(parents=True, exist_ok=True)
    return module


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def monotonic(values: list[float], tolerance: float = 1e-18) -> bool:
    return all(b + tolerance >= a for a, b in zip(values, values[1:]))


def audit_extension() -> tuple[list[dict], dict]:
    extension = read_csv(EXT / "TRAIN_EXTENSION144_MANIFEST.csv")
    status = {row["production_id"]: row for row in read_csv(EXT / "logs" / "TRAIN_EXTENSION144_CASE_STATUS.csv")}
    existing = pd.read_csv(FROZEN / "COMBINED216_PRODUCTION_MANIFEST.csv")
    existing_hashes = set(existing.microstructure_sha256.astype(str))
    hashes = [row["microstructure_sha256"] for row in extension]
    failures, qc_failures, wall_times = [], [], []
    max_closure = max_mg_error = max_phase_error = 0.0
    for row in extension:
        pid = row["production_id"]
        case = EXT / "comsol_runs" / pid
        metrics_path, fields_path = case / "metrics.csv", case / "fields.bin"
        reason = []
        if status.get(pid, {}).get("status") != "COMPLETED":
            reason.append("status_not_completed")
        if not metrics_path.exists() or not fields_path.exists():
            reason.append("missing_output")
        else:
            frame = pd.read_csv(metrics_path)
            if len(frame) != 27 or not math.isclose(float(frame.time_s.iloc[-1]), 86400.0, abs_tol=1e-6):
                reason.append("time_grid")
            numeric = frame.select_dtypes(include=[np.number]).to_numpy()
            if not np.isfinite(numeric).all():
                reason.append("nonfinite")
            if not frame.concentration_nonnegative.astype(str).str.lower().eq("true").all():
                reason.append("negative_concentration")
            if not ((frame.alpha_range_status == "inside") & (frame.beta_range_status == "inside")).all():
                reason.append("kinetic_range")
            for column in ("Faraday_mass_kg_per_m", "phase_relative_L2_change", "mean_depth_m", "p95_depth_m", "max_depth_m"):
                if not monotonic(frame[column].astype(float).tolist()):
                    reason.append(f"nonmonotonic_{column}")
            nonzero = frame.loc[frame.time_s > 0]
            max_closure = max(max_closure, float(nonzero.current_closure_error.max()))
            max_mg_error = max(max_mg_error, float(frame.Mg_Faraday_conservation_error.iloc[-1]))
            max_phase_error = max(max_phase_error, float(frame.phase_Faraday_relative_difference.iloc[-1]))
        if reason:
            qc_failures.append({"production_id": pid, "reason": ";".join(reason)})
        state = status.get(pid, {})
        if state.get("status") == "FAILED":
            failures.append(pid)
        try:
            wall_times.append(float(state.get("wall_time_s", "")))
        except ValueError:
            pass
    overlap = sorted(set(hashes) & existing_hashes)
    duplicates = len(hashes) - len(set(hashes))
    result = {
        "extension_trajectories": len(extension),
        "completed": sum(status.get(row["production_id"], {}).get("status") == "COMPLETED" for row in extension),
        "failed": len(failures),
        "retried": sum(int(status.get(row["production_id"], {}).get("retry_count", 0) or 0) > 0 for row in extension),
        "qc_failures": len(qc_failures),
        "duplicate_with_existing216": len(overlap),
        "duplicate_within_extension": duplicates,
        "max_current_closure_percent": 100 * max_closure,
        "max_mg_faraday_error_percent": 100 * max_mg_error,
        "max_phase_faraday_difference_percent": 100 * max_phase_error,
        "total_wall_hours": sum(wall_times) / 3600,
        "mean_wall_minutes": float(np.mean(wall_times) / 60) if wall_times else float("nan"),
        "qc_failure_details": qc_failures,
    }
    if len(extension) != 144 or result["completed"] != 144 or failures or qc_failures or overlap or duplicates:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return extension, result


def combined_store_class(module, extension_rows):
    class Store288(module.Store216):
        def __init__(self) -> None:
            base = pd.read_csv(FROZEN / "COMBINED216_PRODUCTION_MANIFEST.csv")
            extra = pd.DataFrame(extension_rows)
            extra["trajectory_path"] = extra.production_id.map(lambda pid: str(EXT / "comsol_runs" / pid))
            extra["microstructure_path"] = extra.smooth_npy_path
            extra["dataset_generation"] = "train_extension144"
            extra["split_role"] = "train"
            extra["f_beta"] = pd.to_numeric(extra.f_beta)
            extra["seed"] = pd.to_numeric(extra.seed)
            required = list(base.columns)
            for column in required:
                if column not in extra:
                    extra[column] = ""
            manifest = pd.concat([base, extra[required]], ignore_index=True)
            self.rows = manifest.set_index("production_id")
            self.allowed = set(self.rows.index[self.rows.split_role.isin(["train", "validation"])])
            self.forbidden = set(self.rows.index) - self.allowed
            self.split = manifest[["production_id", "split_role", "f_beta", "morphology", "dataset_generation"]].rename(columns={"production_id": "run_id"})
            first = self.ids("train")[0]
            self.header = module.read_header(self.field_path(first))
            self.times = np.asarray(self.header.times, dtype=np.float64)
            self.hours = self.times / 3600.0
            from collections import OrderedDict
            self._maps = OrderedDict()
            self._micro, self._metrics, self._pf0 = {}, {}, {}
    return Store288


def aggregate(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    output = []
    metrics = ("delta_phase_relative_l2", "interface_mae_nm", "max_depth_mae_nm")
    previous = None
    for size in SIZES:
        group = frame[(frame.n_train == size) & (frame.status == "OK")]
        row = {"n_train": size, "n_seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
            row[f"{metric}_improvement_from_previous_percent"] = "" if previous is None else 100 * (previous[metric] - row[f"{metric}_mean"]) / previous[metric]
        output.append(row)
        previous = {metric: row[f"{metric}_mean"] for metric in metrics}
    return output


def main() -> None:
    extension, qc = audit_extension()
    module = load_training_module()
    module.ACCESS_LOG.write_text("# Truth-access guard\n\nNo historical or independent-final-test truth was accessed.\n", encoding="utf-8")
    store = combined_store_class(module, extension)()
    train_ids, val_ids = store.ids("train"), store.ids("validation")
    if len(train_ids) != 288 or len(val_ids) != 36:
        raise RuntimeError(f"Unexpected train/validation counts: {len(train_ids)}/{len(val_ids)}")
    cells = {}
    train_frame = store.rows.loc[train_ids]
    for (fraction, morphology), frame in train_frame.groupby(["f_beta", "morphology"]):
        old = sorted(frame.index[frame.dataset_generation != "train_extension144"].tolist(), key=lambda pid: int(store.rows.loc[pid, "seed"]))
        new = sorted(frame.index[frame.dataset_generation == "train_extension144"].tolist(), key=lambda pid: int(store.rows.loc[pid, "seed"]))
        if len(old) != 16 or len(new) != 16:
            raise RuntimeError(f"Unbalanced cell {fraction}/{morphology}: {len(old)}+{len(new)}")
        cells[(float(fraction), morphology)] = old + new
    subset_rows, training_rows = [], []
    config = {"base_channels": 32, "learning_rate": 2e-4, "batch_size": 2, "interface_lambda": 2.0, "gradient_weight": 0.03}
    existing144 = pd.read_csv(SEARCH / "FINAL_5SEED_VALIDATION_METRICS.csv").to_dict("records")
    for row in existing144:
        training_rows.append({"n_train": 144, "source": "existing_exact_configuration", **row})
    for size in SIZES:
        selected = sorted(pid for ids in cells.values() for pid in ids[: size // 9])
        for pid in selected:
            subset_rows.append({"n_train": size, "production_id": pid})
        if size == 144:
            continue
        for seed in SEEDS:
            result = module.run(store, selected, val_ids, config, seed, f"scaling_n{size}", True, 20)
            training_rows.append({"n_train": size, "source": "new_training", **result})
    if any(row.get("status") != "OK" for row in training_rows) or len(training_rows) != 25:
        raise RuntimeError("One or more learning-curve jobs failed")
    write_csv(OUT / "LEARNING_CURVE_NESTED_SUBSETS.csv", subset_rows)
    summary = aggregate(training_rows)
    write_csv(OUT / "LEARNING_CURVE_144_180_216_252_288.csv", summary)
    final_rows = [row for row in training_rows if row["n_train"] == 288]
    write_csv(OUT / "TRAIN288_FINAL_VALIDATION_METRICS.csv", final_rows)
    metrics = ("delta_phase_relative_l2", "interface_mae_nm", "max_depth_mae_nm")
    labels = ("Delta-phase relative L2", "Interface MAE (nm)", "Maximum-depth MAE (nm)")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for axis, metric, label in zip(axes, metrics, labels):
        axis.errorbar([row["n_train"] for row in summary], [row[f"{metric}_mean"] for row in summary], yerr=[row[f"{metric}_sd"] for row in summary], marker="o", capsize=3)
        axis.set(xlabel="Training trajectories", ylabel=label, xticks=SIZES)
    fig.savefig(OUT / "LEARNING_CURVE_144_180_216_252_288.png", dpi=220)
    plt.close(fig)
    access = module.ACCESS_LOG.read_text(encoding="utf-8")
    if "BLOCKED" in access:
        raise RuntimeError("A forbidden truth-access attempt was recorded")
    audit_lines = ["# DATA SCALING AUDIT", "", "- Nested subsets: 144 ⊂ 180 ⊂ 216 ⊂ 252 ⊂ 288", "- Each subset is balanced across all nine design cells.", "- Validation trajectories: fixed 36", "- Independent final-test truth accessed: no", "- Hyperparameters: unchanged", "", "## Relative improvement from previous scale"]
    for row in summary[1:]:
        audit_lines.append(f"- {SIZES[SIZES.index(row['n_train'])-1]}→{row['n_train']}: phase {row['delta_phase_relative_l2_improvement_from_previous_percent']:.3f}%, interface {row['interface_mae_nm_improvement_from_previous_percent']:.3f}%, max depth {row['max_depth_mae_nm_improvement_from_previous_percent']:.3f}%")
    (OUT / "DATA_SCALING_AUDIT.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    report = ["# TRAIN EXTENSION144 FINAL REPORT", "", f"- Successful trajectories: {qc['completed']}/144", f"- Failed trajectories: {qc['failed']}", f"- Retried trajectories: {qc['retried']}", f"- Duplicate hashes against existing 216: {qc['duplicate_with_existing216']}", f"- Internal duplicate hashes: {qc['duplicate_within_extension']}", "- Total training trajectories: 288", "- Validation trajectories: fixed 36", "- Independent final-test truth accessed: no", "- Hyperparameter search performed: no", "", "## Learning curve"]
    for row in summary:
        report.append(f"- n={row['n_train']}: delta-phase L2 {row['delta_phase_relative_l2_mean']:.6g} ± {row['delta_phase_relative_l2_sd']:.3g}; interface MAE {row['interface_mae_nm_mean']:.4g} ± {row['interface_mae_nm_sd']:.3g} nm; maximum-depth MAE {row['max_depth_mae_nm_mean']:.4g} ± {row['max_depth_mae_nm_sd']:.3g} nm")
    (OUT / "TRAIN_EXTENSION144_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "TRAIN_EXTENSION144_SCALING_COMPLETE.flag").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
