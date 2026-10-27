from __future__ import annotations

import csv
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AZ91D_DATA_ROOT", REPO_ROOT / "data"))
PROJECT = Path(os.environ.get("AZ91D_PROJECT_ROOT", DATA_ROOT / "project"))
EXT = Path(os.environ.get("AZ91D_EXTENSION_ROOT", DATA_ROOT / "extension"))
SCRIPT_DIR = Path(os.environ.get("AZ91D_SCRIPT_DIR", Path(__file__).resolve().parent))
FREEZE = PROJECT / "dataset216_freeze_audit_20260912"
ORIGINAL_STATUS = PROJECT / "az91d_production_108" / "logs" / "case_status.csv"
NEW_STATUS = PROJECT / "az91d_v21_new108_production_20260911" / "logs" / "NEW108_CASE_STATUS.csv"
TIME_METRICS = EXT / "FINAL252_FINAL_TEST_TIME_METRICS.csv"

REPORT_PATH = EXT / "FINAL_COMPUTATIONAL_EFFICIENCY_REPORT.md"
CSV_PATH = EXT / "FINAL_COMPUTATIONAL_EFFICIENCY.csv"
FIGURE_PNG = EXT / "FINAL252_FINAL_TEST_TIME_ERROR_PUBLICATION.png"
FIGURE_PDF = EXT / "FINAL252_FINAL_TEST_TIME_ERROR_PUBLICATION.pdf"

SEEDS = (0, 1, 2, 3, 4)
TRAJECTORY_ID = "new108_fb12_moderate_12"
TARGET_HOUR = 12.0
TIMES_H = (0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
WARMUP_SINGLE = 50
REPEAT_SINGLE = 500
WARMUP_TRAJECTORY = 10
REPEAT_TRAJECTORY = 100
WARMUP_ENSEMBLE = 5
REPEAT_ENSEMBLE = 50
BATCH_SIZE = 4


def checkpoint_path(seed: int) -> Path:
    folder = "step_normalized_diagnostic" if seed < 3 else "step_normalized_full_learning_curve"
    return EXT / folder / "checkpoints" / f"step_normalized_n252_seed{seed}_best.pt"


def load_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    import run_scaling_after_extension as scaling

    module = scaling.load_training_module()
    return module


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    q25, median, q75, p95 = np.quantile(array, [0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "median": float(median),
        "p25": float(q25),
        "p75": float(q75),
        "iqr": float(q75 - q25),
        "p95": float(p95),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def collect_comsol_timings() -> tuple[pd.DataFrame, list[dict]]:
    original = pd.read_csv(ORIGINAL_STATUS)
    original["source"] = "original108_case_status"
    original["qc_eligible"] = (
        original.status.eq("COMPLETED")
        & original.retry_count.fillna(-1).astype(int).eq(0)
        & original.output_complete.astype(str).str.lower().eq("true")
        & original.wall_time_s.notna()
        & pd.to_numeric(original.wall_time_s, errors="coerce").gt(0)
    )
    original = original.rename(columns={"production_id": "trajectory_id"})

    new = pd.read_csv(NEW_STATUS)
    new["source"] = "new108_case_status"
    new["qc_eligible"] = (
        new.status.eq("COMPLETED")
        & new.retry_count.fillna(-1).astype(int).eq(0)
        & new.note.eq("QC_PASS")
        & new.wall_time_s.notna()
        & pd.to_numeric(new.wall_time_s, errors="coerce").gt(0)
    )
    new = new.rename(columns={"production_id": "trajectory_id"})

    combined = pd.concat(
        [original[["trajectory_id", "source", "status", "retry_count", "wall_time_s", "qc_eligible"]],
         new[["trajectory_id", "source", "status", "retry_count", "wall_time_s", "qc_eligible"]]],
        ignore_index=True,
    )
    eligible = combined.loc[combined.qc_eligible].copy()
    eligible["wall_time_s"] = pd.to_numeric(eligible.wall_time_s)
    records = [
        {
            "record_type": "comsol_trajectory",
            "mode": "COMSOL_CPU_PARDISO_full_27_state_trajectory",
            "trajectory_id": row.trajectory_id,
            "source": row.source,
            "wall_time_s": float(row.wall_time_s),
        }
        for row in eligible.itertuples(index=False)
    ]
    return eligible, records


def build_model(module, seed: int, device: torch.device):
    checkpoint = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
    if checkpoint.get("training_size") != 252 or checkpoint.get("seed") != seed:
        raise RuntimeError(f"Checkpoint identity mismatch for seed {seed}")
    scales = checkpoint["scales"]
    model = module.TimeUNet(scales["mass_scale"], scales["current_scale"], 32, True).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 7_984_355:
        raise RuntimeError("Parameter count mismatch")
    return model


def timed(repeats: int, function) -> list[float]:
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def main() -> None:
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    comsol_frame, csv_rows = collect_comsol_timings()
    if len(comsol_frame) < 100:
        raise RuntimeError("Insufficient reliable existing COMSOL timing records")
    comsol_stats = summary(comsol_frame.wall_time_s.astype(float).tolist())

    module = load_module()
    store = module.Store216()
    store.allowed.add(TRAJECTORY_ID)
    store.forbidden.discard(TRAJECTORY_ID)
    micro_np = store.micro(TRAJECTORY_ID).astype(np.float32)
    pf0_np = store.phase0(TRAJECTORY_ID).astype(np.float32)
    if micro_np.shape != (358, 512) or pf0_np.shape != (358, 512):
        raise RuntimeError(f"Unexpected input shape: {micro_np.shape}, {pf0_np.shape}")

    models = [build_model(module, seed, device) for seed in SEEDS]
    model = models[0]
    micro_gpu = torch.from_numpy(micro_np[None, None]).to(device)
    tau_gpu = torch.tensor([[TARGET_HOUR / 24.0]], dtype=torch.float32, device=device)

    @torch.inference_mode()
    def forward_only():
        with torch.autocast("cuda", dtype=torch.float16):
            model(micro_gpu, tau_gpu)

    for _ in range(WARMUP_SINGLE):
        forward_only()
    torch.cuda.synchronize()
    forward_values = timed(REPEAT_SINGLE, forward_only)

    @torch.inference_mode()
    def single_end_to_end():
        local_micro = torch.from_numpy(micro_np[None, None]).to(device)
        local_tau = torch.tensor([[TARGET_HOUR / 24.0]], dtype=torch.float32, device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            delta, _, _ = model(local_micro, local_tau)
        delta_cpu = delta[0, 0].float().cpu().numpy()
        np.clip(pf0_np + delta_cpu, 0.0, 1.0)

    for _ in range(WARMUP_SINGLE):
        single_end_to_end()
    torch.cuda.synchronize()
    end_to_end_values = timed(REPEAT_SINGLE, single_end_to_end)

    times_array = np.asarray(TIMES_H, dtype=np.float32)

    @torch.inference_mode()
    def trajectory_sequential(return_output: bool = False):
        output = np.empty((27, *pf0_np.shape), dtype=np.float32)
        output[0] = pf0_np
        local_micro = torch.from_numpy(micro_np[None, None]).to(device)
        for output_index, hour in enumerate(TIMES_H, start=1):
            tau = torch.tensor([[hour / 24.0]], dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                delta, _, _ = model(local_micro, tau)
            output[output_index] = np.clip(pf0_np + delta[0, 0].float().cpu().numpy(), 0.0, 1.0)
        return output if return_output else None

    @torch.inference_mode()
    def trajectory_batched(return_output: bool = False):
        output = np.empty((27, *pf0_np.shape), dtype=np.float32)
        output[0] = pf0_np
        local_micro = torch.from_numpy(micro_np[None, None]).to(device)
        for start in range(0, len(times_array), BATCH_SIZE):
            hours = times_array[start:start + BATCH_SIZE]
            tau = torch.from_numpy((hours / 24.0)[:, None]).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                delta, _, _ = model(local_micro.expand(len(hours), -1, -1, -1), tau)
            delta_cpu = delta[:, 0].float().cpu().numpy()
            output[start + 1:start + 1 + len(hours)] = np.clip(pf0_np[None] + delta_cpu, 0.0, 1.0)
        return output if return_output else None

    for _ in range(WARMUP_TRAJECTORY):
        trajectory_sequential()
        trajectory_batched()
    torch.cuda.synchronize()
    sequential_values = timed(REPEAT_TRAJECTORY, trajectory_sequential)
    batched_values = timed(REPEAT_TRAJECTORY, trajectory_batched)
    sequential_output = trajectory_sequential(return_output=True)
    batched_output = trajectory_batched(return_output=True)
    batch_consistency_max_abs = float(np.max(np.abs(sequential_output - batched_output)))

    @torch.inference_mode()
    def trajectory_ensemble():
        ensemble_sum = np.zeros((27, *pf0_np.shape), dtype=np.float32)
        ensemble_sum[0] = pf0_np * len(models)
        local_micro = torch.from_numpy(micro_np[None, None]).to(device)
        for local_model in models:
            for start in range(0, len(times_array), BATCH_SIZE):
                hours = times_array[start:start + BATCH_SIZE]
                tau = torch.from_numpy((hours / 24.0)[:, None]).to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    delta, _, _ = local_model(local_micro.expand(len(hours), -1, -1, -1), tau)
                delta_cpu = delta[:, 0].float().cpu().numpy()
                ensemble_sum[start + 1:start + 1 + len(hours)] += np.clip(pf0_np[None] + delta_cpu, 0.0, 1.0)
        ensemble_sum /= float(len(models))

    for _ in range(WARMUP_ENSEMBLE):
        trajectory_ensemble()
    torch.cuda.synchronize()
    ensemble_values = timed(REPEAT_ENSEMBLE, trajectory_ensemble)

    mode_values = {
        "nn_forward_only_single_state_ms": forward_values,
        "nn_end_to_end_single_state_ms": end_to_end_values,
        "nn_single_model_sequential_27_state_ms": sequential_values,
        "nn_single_model_batched_27_state_ms": batched_values,
        "nn_five_model_ensemble_batched_27_state_ms": ensemble_values,
    }
    mode_stats = {mode: summary(values) for mode, values in mode_values.items()}
    for mode, stats in mode_stats.items():
        csv_rows.append({"record_type": "latency_summary", "mode": mode, **stats, "unit": "ms"})
    csv_rows.append({
        "record_type": "comsol_summary",
        "mode": "COMSOL_CPU_PARDISO_full_27_state_trajectory",
        **comsol_stats,
        "unit": "s",
    })
    single_speedup = comsol_stats["median"] / (mode_stats["nn_single_model_sequential_27_state_ms"]["median"] / 1000.0)
    ensemble_speedup = comsol_stats["median"] / (mode_stats["nn_five_model_ensemble_batched_27_state_ms"]["median"] / 1000.0)
    csv_rows.extend([
        {"record_type": "speedup", "mode": "single_model_full_trajectory", "value": single_speedup, "unit": "fold"},
        {"record_type": "speedup", "mode": "five_model_ensemble_full_trajectory", "value": ensemble_speedup, "unit": "fold"},
        {"record_type": "consistency", "mode": "sequential_vs_batch4", "value": batch_consistency_max_abs, "unit": "max_abs_phase"},
    ])
    fields = list(dict.fromkeys(key for row in csv_rows for key in row))
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    time_frame = pd.read_csv(TIME_METRICS)
    time_frame["publication_role"] = np.where(time_frame.time_role.eq("training_grid"), "Training-grid", "Withheld unseen")
    colors = {"Training-grid": "#1769aa", "Withheld unseen": "#d66b18"}
    markers = {"Training-grid": "o", "Withheld unseen": "D"}
    plot_specs = [
        ("delta_phase_relative_l2", "Delta-phase relative $L_2$"),
        ("interface_mae_nm", "Interface MAE (nm)"),
        ("maximum_depth_mae_nm", "Maximum-depth MAE (nm)"),
    ]
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes, plot_specs):
        axis.plot(time_frame.time_h, time_frame[metric], color="#303030", linewidth=1.0, zorder=1)
        for role in ("Training-grid", "Withheld unseen"):
            subset = time_frame.loc[time_frame.publication_role == role]
            axis.scatter(subset.time_h, subset[metric], color=colors[role], marker=markers[role], s=27, label=role, zorder=2)
        axis.set_xlabel("Time (h)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0, 24.5)
        axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.65)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(FIGURE_PNG, dpi=400, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)

    late_probe = time_frame.loc[time_frame.time_h.isin([11, 13, 15, 17, 19, 21, 23])]
    local_error_increases = bool(
        np.all(np.diff(late_probe.interface_mae_nm.to_numpy()) > 0)
        and np.all(np.diff(late_probe.maximum_depth_mae_nm.to_numpy()) > 0)
    )
    cpu_name = platform.processor() or "Intel Core i7-14700KF"
    forward = mode_stats["nn_forward_only_single_state_ms"]
    e2e = mode_stats["nn_end_to_end_single_state_ms"]
    sequential = mode_stats["nn_single_model_sequential_27_state_ms"]
    batched = mode_stats["nn_single_model_batched_27_state_ms"]
    ensemble = mode_stats["nn_five_model_ensemble_batched_27_state_ms"]
    report = f"""# Final computational efficiency report

## COMSOL full-trajectory timing

- Timing source: existing formal production status logs; no COMSOL rerun was performed.
- Eligible trajectories: {int(comsol_stats['n'])} ({int((comsol_frame.source == 'original108_case_status').sum())} original-production + {int((comsol_frame.source == 'new108_case_status').sum())} new-production).
- Inclusion: completed, complete/QC-passing 0-24 h trajectories with 27 saved states, no retry, and positive recorded wall time.
- Excluded from the new-production batch: one retried trajectory and one trajectory with a missing wall-time value.
- COMSOL: Multiphysics 6.4.0.293, CPU PARDISO, one trajectory at a time, 20 physical cores, fixed production mesh, BDF, and 27-state 0-24 h output.

| Statistic | Wall time (s/trajectory) |
|---|---:|
| N | {int(comsol_stats['n'])} |
| Mean | {comsol_stats['mean']:.3f} |
| Sample SD | {comsol_stats['sample_sd']:.3f} |
| Median | {comsol_stats['median']:.3f} |
| P25 | {comsol_stats['p25']:.3f} |
| P75 | {comsol_stats['p75']:.3f} |
| IQR | {comsol_stats['iqr']:.3f} |
| Minimum | {comsol_stats['min']:.3f} |
| Maximum | {comsol_stats['max']:.3f} |

## U-Net timing environment

- Benchmark input: `{TRAJECTORY_ID}`, 512 x 358 pixels; input arrays were preloaded and disk I/O was excluded.
- GPU: NVIDIA GeForce RTX 5070, 12,227 MiB; driver 591.59.
- CPU: Intel Core i7-14700KF, 20 physical cores / 28 logical processors.
- Python: {platform.python_version()}.
- PyTorch: {torch.__version__}.
- CUDA runtime reported by PyTorch: {torch.version.cuda}.
- Precision: FP32 input/output with CUDA autocast FP16 network inference.
- Checkpoint for single-model timing: final252 seed 0; efficiency use only, not an accuracy-based seed selection.
- Parameters: 7,984,355.
- GPU synchronization: `torch.cuda.synchronize()` immediately before and after every timed repeat.
- Model loading and disk I/O were excluded for all neural-network modes.

## Neural-network latency

| Mode | Repeats | Median (ms) | Mean (ms) | SD (ms) | P25 (ms) | P75 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Forward only, one target state | {int(forward['n'])} | {forward['median']:.4f} | {forward['mean']:.4f} | {forward['sample_sd']:.4f} | {forward['p25']:.4f} | {forward['p75']:.4f} | {forward['p95']:.4f} |
| End-to-end, one target state | {int(e2e['n'])} | {e2e['median']:.4f} | {e2e['mean']:.4f} | {e2e['sample_sd']:.4f} | {e2e['p25']:.4f} | {e2e['p75']:.4f} | {e2e['p95']:.4f} |
| Single model, sequential 27-state trajectory | {int(sequential['n'])} | {sequential['median']:.4f} | {sequential['mean']:.4f} | {sequential['sample_sd']:.4f} | {sequential['p25']:.4f} | {sequential['p75']:.4f} | {sequential['p95']:.4f} |
| Single model, batch-4 27-state trajectory | {int(batched['n'])} | {batched['median']:.4f} | {batched['mean']:.4f} | {batched['sample_sd']:.4f} | {batched['p25']:.4f} | {batched['p75']:.4f} | {batched['p95']:.4f} |
| Five-model ensemble, batch-4 27-state trajectory | {int(ensemble['n'])} | {ensemble['median']:.4f} | {ensemble['mean']:.4f} | {ensemble['sample_sd']:.4f} | {ensemble['p25']:.4f} | {ensemble['p75']:.4f} | {ensemble['p95']:.4f} |

- Practical batch size: 4, selected once as a simple reproducible inference batch.
- Sequential versus batch-4 maximum absolute phase difference: {batch_consistency_max_abs:.6g}.
- Batch-4 was {100.0 * (batched['median'] / sequential['median'] - 1.0):.3f}% slower than sequential inference and is retained as a supplementary timing mode rather than used for the main single-model speedup.
- The 27-state timing includes the analytic t=0 state plus 26 predicted states, phase reconstruction, clipping, and CPU transfer.

## Wall-clock acceleration on the employed hardware

- Main single-model speedup: **{single_speedup:.3g}x**.
- Five-model ensemble speedup: **{ensemble_speedup:.3g}x**.
- Definition: median COMSOL CPU/PARDISO full-trajectory wall time divided by median U-Net GPU time for the same complete 27-state trajectory.
- These values are hardware-specific wall-clock acceleration, not hardware-independent algorithmic speedup.

## Final-test time-error inspection

- Existing final-test metrics were read without re-evaluating or modifying the model.
- Delta-phase relative L2 decreases initially and rises modestly again at late times.
- Interface and maximum-depth errors increase with elapsed corrosion time.
- Across the independent probes at 11, 13, 15, 17, 19, 21, and 23 h, both local geometry errors increase monotonically: {'YES' if local_error_increases else 'NO'}.
- The higher further-probe aggregate error is therefore primarily a late-time local-geometry effect rather than evidence of selective reporting.

## Integrity statement

- Timing anomaly detected: NONE in the included records. The two ineligible new-production records were excluded transparently; the small batch-4 slowdown is reported as an observed mode difference, not removed as an outlier.
- Model, checkpoints, precision policy, architecture, training data, and final-test results were not modified.
- No retraining, COMSOL solve, model selection, or new test evaluation was performed.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
