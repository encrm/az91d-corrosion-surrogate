from __future__ import annotations

import csv
import hashlib
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AZ91D_DATA_ROOT", REPO_ROOT / "data"))
PROJECT = Path(os.environ.get("AZ91D_PROJECT_ROOT", DATA_ROOT / "project"))
EXT = Path(os.environ.get("AZ91D_EXTENSION_ROOT", DATA_ROOT / "extension"))
SCRIPT_DIR = Path(os.environ.get("AZ91D_SCRIPT_DIR", Path(__file__).resolve().parent))
OUT_DIR = EXT / "validation_only_252_vs_288_robustness"
REPORT_PATH = EXT / "TRAIN_SIZE_252_VS_288_ROBUSTNESS.md"
TRAJECTORY_PATH = EXT / "TRAIN_SIZE_252_VS_288_TRAJECTORY_METRICS.csv"
ENSEMBLE_PATH = EXT / "TRAIN_SIZE_252_VS_288_ENSEMBLE_METRICS.csv"

SIZES = (252, 288)
SEEDS = (0, 1, 2, 3, 4)
METRICS = ("delta_phase_relative_l2", "interface_mae_nm", "maximum_depth_mae_nm")
TRAIN_HOURS = (0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24)
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20_260_917


def checkpoint_path(size: int, seed: int) -> Path:
    folder = "step_normalized_diagnostic" if seed < 3 else "step_normalized_full_learning_curve"
    return EXT / folder / "checkpoints" / f"step_normalized_n{size}_seed{seed}_best.pt"


def load_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    import run_scaling_after_extension as scaling

    module = scaling.load_training_module()
    module.ACCESS_LOG = OUT_DIR / "INDEPENDENT_TEST_ACCESS_GUARD.log"
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    batch = 10_000
    for start in range(0, BOOTSTRAP_REPLICATES, batch):
        stop = min(start + batch, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def sample_metrics(module, estimate: np.ndarray, truth: np.ndarray, pf0: np.ndarray, y: np.ndarray) -> dict[str, float]:
    metric = module.phase_metrics(estimate, truth, pf0, y)
    reference = module.crossings(pf0, y)
    true_interface = module.crossings(truth, y, reference)
    pred_interface = module.crossings(estimate, y, reference)
    valid = np.isfinite(true_interface) & np.isfinite(pred_interface)
    interface = float(np.mean(np.abs(pred_interface[valid] - true_interface[valid]))) if valid.any() else float("nan")
    return {
        "delta_phase_relative_l2": float(metric["delta_phase_relative_l2"]),
        "interface_mae_nm": interface * 1e9,
        "maximum_depth_mae_nm": float(metric["max_depth_error_m"]) * 1e9,
    }


def predict_model(module, store, validation_ids: list[str], time_indices: list[int], checkpoint: dict) -> np.ndarray:
    scales = checkpoint["scales"]
    model = module.TimeUNet(scales["mass_scale"], scales["current_scale"], 32, True).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    predictions = []
    with torch.no_grad():
        for run_id in validation_ids:
            micro = torch.from_numpy(store.micro(run_id)[None, None]).cuda()
            pf0 = store.phase0(run_id)
            for time_index in time_indices:
                tau = torch.tensor([[store.times[time_index] / 86400.0]], dtype=torch.float32, device="cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    delta, _, _ = model(micro, tau)
                phase = np.clip(pf0 + delta[0, 0].float().cpu().numpy(), 0.0, 1.0).astype(np.float32)
                predictions.append(phase)
    del model
    torch.cuda.empty_cache()
    return np.stack(predictions)


def prepare_truth(store, validation_ids: list[str], time_indices: list[int]):
    truths, initial, run_ids, times = [], [], [], []
    for run_id in validation_ids:
        pf0 = store.phase0(run_id).astype(np.float32)
        for time_index in time_indices:
            truths.append(store.phase(run_id, time_index).astype(np.float32))
            initial.append(pf0)
            run_ids.append(run_id)
            times.append(float(store.hours[time_index]))
    return np.stack(truths), np.stack(initial), run_ids, times


def seed_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    q1, q3 = np.quantile(array, [0.25, 0.75])
    median = float(np.median(array))
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "median": median,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "mad": float(np.median(np.abs(array - median))),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guard = OUT_DIR / "INDEPENDENT_TEST_ACCESS_GUARD.log"
    guard.write_text("# Validation-only robustness audit guard\nNo independent evaluation truth was accessed.\n", encoding="utf-8")
    module = load_module()
    store = module.Store216()
    validation_ids = store.ids("validation")
    if len(validation_ids) != 36:
        raise RuntimeError(f"Expected 36 validation trajectories, found {len(validation_ids)}")
    time_indices = module.indices(store)
    actual_hours = tuple(float(store.hours[index]) for index in time_indices)
    if actual_hours != TRAIN_HOURS:
        raise RuntimeError(f"Training-time grid mismatch: {actual_hours}")

    truths, initial, state_run_ids, state_hours = prepare_truth(store, validation_ids, time_indices)
    y = np.linspace(store.header.ymin, store.header.ymax, store.header.ny)
    state_index_by_run = {
        run_id: [index for index, value in enumerate(state_run_ids) if value == run_id]
        for run_id in validation_ids
    }

    per_seed_state: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    per_seed_global: dict[tuple[int, int], dict[str, float]] = {}
    per_seed_trajectory: dict[tuple[int, int, str], dict[str, float]] = {}
    ensemble_global: dict[int, dict[str, float]] = {}
    ensemble_trajectory: dict[tuple[int, str], dict[str, float]] = {}
    checkpoint_hashes: dict[tuple[int, int], str] = {}

    started = time.time()
    for size in SIZES:
        prediction_sum = np.zeros_like(truths, dtype=np.float32)
        for seed in SEEDS:
            path = checkpoint_path(size, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            checkpoint_hashes[(size, seed)] = sha256_file(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("training_size") != size or checkpoint.get("seed") != seed:
                raise RuntimeError(f"Checkpoint identity mismatch: {path}")
            predictions = predict_model(module, store, validation_ids, time_indices, checkpoint)
            prediction_sum += predictions
            state_metrics = {metric: np.empty(len(predictions), dtype=np.float64) for metric in METRICS}
            for index in range(len(predictions)):
                values = sample_metrics(module, predictions[index], truths[index], initial[index], y)
                for metric in METRICS:
                    state_metrics[metric][index] = values[metric]
            per_seed_state[(size, seed)] = state_metrics
            per_seed_global[(size, seed)] = {metric: float(state_metrics[metric].mean()) for metric in METRICS}
            for run_id in validation_ids:
                indices = state_index_by_run[run_id]
                per_seed_trajectory[(size, seed, run_id)] = {
                    metric: float(state_metrics[metric][indices].mean()) for metric in METRICS
                }
            del predictions

        ensemble = prediction_sum / float(len(SEEDS))
        ensemble_state_metrics = {metric: np.empty(len(ensemble), dtype=np.float64) for metric in METRICS}
        for index in range(len(ensemble)):
            values = sample_metrics(module, ensemble[index], truths[index], initial[index], y)
            for metric in METRICS:
                ensemble_state_metrics[metric][index] = values[metric]
        ensemble_global[size] = {metric: float(ensemble_state_metrics[metric].mean()) for metric in METRICS}
        for run_id in validation_ids:
            indices = state_index_by_run[run_id]
            ensemble_trajectory[(size, run_id)] = {
                metric: float(ensemble_state_metrics[metric][indices].mean()) for metric in METRICS
            }
        del ensemble, prediction_sum

    trajectory_rows: list[dict] = []
    for size in SIZES:
        for seed in SEEDS:
            for run_id in validation_ids:
                for metric in METRICS:
                    trajectory_rows.append({
                        "record_type": "seed_trajectory",
                        "training_size": size,
                        "seed": seed,
                        "trajectory_id": run_id,
                        "metric": metric,
                        "value": per_seed_trajectory[(size, seed, run_id)][metric],
                    })
        for run_id in validation_ids:
            for metric in METRICS:
                trajectory_rows.append({
                    "record_type": "ensemble_trajectory",
                    "training_size": size,
                    "seed": "ensemble_mean",
                    "trajectory_id": run_id,
                    "metric": metric,
                    "value": ensemble_trajectory[(size, run_id)][metric],
                })

    paired_summaries = {}
    for metric in METRICS:
        e252 = np.asarray([
            np.mean([per_seed_trajectory[(252, seed, run_id)][metric] for seed in SEEDS])
            for run_id in validation_ids
        ])
        e288 = np.asarray([
            np.mean([per_seed_trajectory[(288, seed, run_id)][metric] for seed in SEEDS])
            for run_id in validation_ids
        ])
        difference = e288 - e252
        ci_low, ci_high = bootstrap_mean_ci(difference)
        nonzero = difference[~np.isclose(difference, 0.0, rtol=0.0, atol=1e-15)]
        p_value = float(wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue) if len(nonzero) else 1.0
        paired_summaries[metric] = {
            "count_252_better": int(np.sum(difference > 0)),
            "count_288_better": int(np.sum(difference < 0)),
            "count_tied": int(np.sum(difference == 0)),
            "mean_difference_288_minus_252": float(difference.mean()),
            "median_difference_288_minus_252": float(np.median(difference)),
            "bootstrap_95_ci_low": ci_low,
            "bootstrap_95_ci_high": ci_high,
            "wilcoxon_p": p_value,
        }
        for index, run_id in enumerate(validation_ids):
            trajectory_rows.append({
                "record_type": "paired_trajectory",
                "training_size": "252_vs_288",
                "seed": "five_seed_mean",
                "trajectory_id": run_id,
                "metric": metric,
                "E252": float(e252[index]),
                "E288": float(e288[index]),
                "difference_288_minus_252": float(difference[index]),
                "better_size": 252 if difference[index] > 0 else (288 if difference[index] < 0 else "tie"),
            })
    write_csv(TRAJECTORY_PATH, trajectory_rows)

    summary_rows: list[dict] = []
    seed_summaries = {}
    for size in SIZES:
        for metric in METRICS:
            values = [per_seed_global[(size, seed)][metric] for seed in SEEDS]
            summary = seed_summary(values)
            seed_summaries[(size, metric)] = summary
            summary_rows.append({
                "record_type": "seed_summary",
                "training_size": size,
                "metric": metric,
                **summary,
            })
            summary_rows.append({
                "record_type": "ensemble_global",
                "training_size": size,
                "metric": metric,
                "value": ensemble_global[size][metric],
            })
    for seed in SEEDS:
        for metric in METRICS:
            value252 = per_seed_global[(252, seed)][metric]
            value288 = per_seed_global[(288, seed)][metric]
            summary_rows.append({
                "record_type": "seed_pair_change",
                "seed": seed,
                "metric": metric,
                "value_252": value252,
                "value_288": value288,
                "difference_288_minus_252": value288 - value252,
                "relative_change_percent": (value288 - value252) / value252 * 100.0,
            })
    for metric in METRICS:
        paired = paired_summaries[metric]
        summary_rows.append({"record_type": "paired_summary", "metric": metric, **paired})
        full252 = np.mean([per_seed_global[(252, seed)][metric] for seed in SEEDS])
        full288 = np.mean([per_seed_global[(288, seed)][metric] for seed in SEEDS])
        no3_252 = np.mean([per_seed_global[(252, seed)][metric] for seed in SEEDS if seed != 3])
        no3_288 = np.mean([per_seed_global[(288, seed)][metric] for seed in SEEDS if seed != 3])
        worst252 = max(SEEDS, key=lambda seed: per_seed_global[(252, seed)][metric])
        worst288 = max(SEEDS, key=lambda seed: per_seed_global[(288, seed)][metric])
        trimmed252 = np.mean([per_seed_global[(252, seed)][metric] for seed in SEEDS if seed != worst252])
        trimmed288 = np.mean([per_seed_global[(288, seed)][metric] for seed in SEEDS if seed != worst288])
        summary_rows.append({
            "record_type": "seed3_sensitivity",
            "metric": metric,
            "full_mean_252": full252,
            "full_mean_288": full288,
            "full_difference_288_minus_252": full288 - full252,
            "exclude_seed3_mean_252": no3_252,
            "exclude_seed3_mean_288": no3_288,
            "exclude_seed3_difference_288_minus_252": no3_288 - no3_252,
            "worst_seed_252": worst252,
            "worst_seed_288": worst288,
            "remove_worst_mean_252": trimmed252,
            "remove_worst_mean_288": trimmed288,
            "remove_worst_difference_288_minus_252": trimmed288 - trimmed252,
        })
    write_csv(ENSEMBLE_PATH, summary_rows)

    central_votes_252 = sum(seed_summaries[(252, metric)]["median"] < seed_summaries[(288, metric)]["median"] for metric in METRICS)
    central_votes_288 = sum(seed_summaries[(288, metric)]["median"] < seed_summaries[(252, metric)]["median"] for metric in METRICS)
    ensemble_votes_252 = sum(ensemble_global[252][metric] < ensemble_global[288][metric] for metric in METRICS)
    ensemble_votes_288 = sum(ensemble_global[288][metric] < ensemble_global[252][metric] for metric in METRICS)
    paired_votes_252 = sum(
        paired_summaries[metric]["mean_difference_288_minus_252"] > 0
        and paired_summaries[metric]["median_difference_288_minus_252"] > 0
        and paired_summaries[metric]["count_252_better"] > paired_summaries[metric]["count_288_better"]
        for metric in METRICS
    )
    paired_votes_288 = sum(
        paired_summaries[metric]["mean_difference_288_minus_252"] < 0
        and paired_summaries[metric]["median_difference_288_minus_252"] < 0
        and paired_summaries[metric]["count_288_better"] > paired_summaries[metric]["count_252_better"]
        for metric in METRICS
    )
    if central_votes_252 >= 2 and ensemble_votes_252 >= 2 and paired_votes_252 >= 2:
        decision = "SUPPORTS_252"
    elif central_votes_288 >= 2 and ensemble_votes_288 >= 2 and paired_votes_288 >= 2:
        decision = "SUPPORTS_288"
    else:
        decision = "AMBIGUOUS"

    lines = [
        "# Validation-only robustness check for training size 252 versus 288",
        "",
        "## Scope and safeguards",
        "",
        "- Data: fixed 36 validation trajectories only.",
        f"- Times (h): {', '.join(map(str, TRAIN_HOURS))}.",
        "- Independent evaluation/final-test target truth accessed: **NO**.",
        "- Checkpoints were loaded read-only; no training, COMSOL solve, or parameter modification was performed.",
        "- Ensemble field: arithmetic mean of the five individually clipped absolute phase predictions.",
        "- Bootstrap: 100,000 trajectory-level resamples, percentile 95% CI, deterministic seed 20260917.",
        "- Wilcoxon unit: trajectory; seed and time states are not treated as independent replicates.",
        "",
        "## Ensemble mean-field metrics",
        "",
        "| Training size | delta-phase relative L2 | Interface MAE (nm) | Maximum-depth MAE (nm) |",
        "|---:|---:|---:|---:|",
        f"| 252 | {ensemble_global[252]['delta_phase_relative_l2']:.10f} | {ensemble_global[252]['interface_mae_nm']:.6f} | {ensemble_global[252]['maximum_depth_mae_nm']:.6f} |",
        f"| 288 | {ensemble_global[288]['delta_phase_relative_l2']:.10f} | {ensemble_global[288]['interface_mae_nm']:.6f} | {ensemble_global[288]['maximum_depth_mae_nm']:.6f} |",
        "",
        "## Trajectory-level paired analysis",
        "",
        "| Metric | 252 better | 288 better | Median D | Mean D | Bootstrap 95% CI of mean D | Wilcoxon p |",
        "|:---|---:|---:|---:|---:|:---|---:|",
    ]
    for metric in METRICS:
        value = paired_summaries[metric]
        lines.append(
            f"| {metric} | {value['count_252_better']} | {value['count_288_better']} | "
            f"{value['median_difference_288_minus_252']:.10g} | {value['mean_difference_288_minus_252']:.10g} | "
            f"[{value['bootstrap_95_ci_low']:.10g}, {value['bootstrap_95_ci_high']:.10g}] | {value['wilcoxon_p']:.6g} |"
        )
    lines.extend(["", "## Seed robustness", ""])
    for metric in METRICS:
        lines.extend([
            f"### {metric}",
            "",
            "| Size | Mean | Sample SD | Median | IQR | MAD |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for size in SIZES:
            value = seed_summaries[(size, metric)]
            lines.append(
                f"| {size} | {value['mean']:.10g} | {value['sample_sd']:.10g} | {value['median']:.10g} | {value['iqr']:.10g} | {value['mad']:.10g} |"
            )
        seed3 = next(row for row in summary_rows if row.get("record_type") == "seed3_sensitivity" and row.get("metric") == metric)
        lines.extend([
            "",
            f"- Full five-seed difference (288 - 252): {seed3['full_difference_288_minus_252']:.10g}",
            f"- Excluding seed 3, four-seed difference (288 - 252): {seed3['exclude_seed3_difference_288_minus_252']:.10g}",
            f"- Removing each size's worst seed, four-seed difference (288 - 252): {seed3['remove_worst_difference_288_minus_252']:.10g}",
            "",
        ])
    lines.extend([
        "## Integrated judgment",
        "",
        f"- Central-tendency votes favoring 252/288: {central_votes_252}/{central_votes_288}",
        f"- Ensemble-field votes favoring 252/288: {ensemble_votes_252}/{ensemble_votes_288}",
        f"- Trajectory-paired votes favoring 252/288: {paired_votes_252}/{paired_votes_288}",
        "- The diagnostic considers central tendency, ensemble behavior, trajectory pairing, and seed dispersion jointly; no single p-value determines the result.",
        "",
        f"**ROBUST_SIZE_SELECTION: {decision}**",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"REPORT={REPORT_PATH}")
    print(f"TRAJECTORY={TRAJECTORY_PATH}")
    print(f"ENSEMBLE={ENSEMBLE_PATH}")
    print(f"DECISION={decision}")
    print(f"WALL_S={time.time() - started:.3f}")
    for size in SIZES:
        print(f"ENSEMBLE_{size}={ensemble_global[size]}")
    for metric in METRICS:
        print(f"PAIRED_{metric}={paired_summaries[metric]}")


if __name__ == "__main__":
    main()
