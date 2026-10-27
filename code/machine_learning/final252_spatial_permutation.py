from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AZ91D_DATA_ROOT", REPO_ROOT / "data"))
EXT = Path(os.environ.get("AZ91D_EXTENSION_ROOT", DATA_ROOT / "extension"))
SCRIPT_DIR = Path(os.environ.get("AZ91D_SCRIPT_DIR", Path(__file__).resolve().parent))
OUT_DIR = EXT / "final252_spatial_permutation_work"
METRICS_PATH = EXT / "FINAL252_SPATIAL_PERMUTATION_METRICS.csv"
TRAJECTORY_PATH = EXT / "FINAL252_SPATIAL_PERMUTATION_TRAJECTORY.csv"
GROUPED_PATH = EXT / "FINAL252_SPATIAL_PERMUTATION_GROUPED.csv"
REPORT_PATH = EXT / "FINAL252_SPATIAL_PERMUTATION_REPORT.md"
FIGURE_PATH = EXT / "FINAL252_SPATIAL_PERMUTATION.png"

SEEDS = (0, 1, 2, 3, 4)
CONDITIONS = ("normal", "permuted")
METRICS = ("delta_phase_relative_l2", "interface_mae_nm", "maximum_depth_mae_nm")
TRAIN_HOURS = (0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24)
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20_260_917


def checkpoint_path(seed: int) -> Path:
    folder = "step_normalized_diagnostic" if seed < 3 else "step_normalized_full_learning_curve"
    return EXT / folder / "checkpoints" / f"step_normalized_n252_seed{seed}_best.pt"


def load_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    import run_scaling_after_extension as scaling

    module = scaling.load_training_module()
    module.ACCESS_LOG = OUT_DIR / "INDEPENDENT_TEST_ACCESS_GUARD.log"
    return module


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


def build_permutation(store, validation_ids: list[str]) -> tuple[dict[str, str], pd.DataFrame]:
    frame = store.split.loc[store.split.run_id.isin(validation_ids)].copy()
    frame = frame.sort_values(["f_beta", "morphology", "run_id"]).reset_index(drop=True)
    mapping = {}
    for _, group in frame.groupby(["f_beta", "morphology"], sort=True):
        ids = group.run_id.tolist()
        if len(ids) != 4:
            raise RuntimeError(f"Expected four validation trajectories per design cell, found {len(ids)}")
        for index, run_id in enumerate(ids):
            mapping[run_id] = ids[(index + 1) % len(ids)]
    if any(source == target for target, source in mapping.items()):
        raise RuntimeError("Permutation contains a self-match")
    return mapping, frame.set_index("run_id")


def prepare_truth(store, validation_ids: list[str], time_indices: list[int]):
    truths, initial, state_run_ids = [], [], []
    for run_id in validation_ids:
        pf0 = store.phase0(run_id).astype(np.float32)
        for time_index in time_indices:
            truths.append(store.phase(run_id, time_index).astype(np.float32))
            initial.append(pf0)
            state_run_ids.append(run_id)
    return np.stack(truths), np.stack(initial), state_run_ids


def predict_model(module, store, validation_ids, time_indices, checkpoint, input_mapping) -> np.ndarray:
    scales = checkpoint["scales"]
    model = module.TimeUNet(scales["mass_scale"], scales["current_scale"], 32, True).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    predictions = []
    with torch.no_grad():
        for target_id in validation_ids:
            input_id = input_mapping[target_id]
            micro = torch.from_numpy(store.micro(input_id)[None, None]).cuda()
            target_pf0 = store.phase0(target_id)
            for time_index in time_indices:
                tau = torch.tensor([[store.times[time_index] / 86400.0]], dtype=torch.float32, device="cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    delta, _, _ = model(micro, tau)
                phase = np.clip(target_pf0 + delta[0, 0].float().cpu().numpy(), 0.0, 1.0).astype(np.float32)
                predictions.append(phase)
    del model
    torch.cuda.empty_cache()
    return np.stack(predictions)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sample_sd": float(array.std(ddof=1))}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guard = OUT_DIR / "INDEPENDENT_TEST_ACCESS_GUARD.log"
    guard.write_text("# Final252 validation-only spatial-permutation guard\nNo independent evaluation truth was accessed.\n", encoding="utf-8")
    module = load_module()
    store = module.Store216()
    validation_ids = store.ids("validation")
    if len(validation_ids) != 36:
        raise RuntimeError(f"Expected 36 validation trajectories, found {len(validation_ids)}")
    time_indices = module.indices(store)
    if tuple(float(store.hours[index]) for index in time_indices) != TRAIN_HOURS:
        raise RuntimeError("Training-time grid mismatch")
    permutation, design = build_permutation(store, validation_ids)
    normal_mapping = {run_id: run_id for run_id in validation_ids}
    mappings = {"normal": normal_mapping, "permuted": permutation}

    truths, initial, state_run_ids = prepare_truth(store, validation_ids, time_indices)
    state_index_by_run = {
        run_id: [index for index, value in enumerate(state_run_ids) if value == run_id]
        for run_id in validation_ids
    }
    y = np.linspace(store.header.ymin, store.header.ymax, store.header.ny)
    state_metrics: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    seed_global: dict[tuple[str, int], dict[str, float]] = {}
    seed_trajectory: dict[tuple[str, int, str], dict[str, float]] = {}
    ensemble_global: dict[str, dict[str, float]] = {}
    ensemble_trajectory: dict[tuple[str, str], dict[str, float]] = {}

    started = time.time()
    for condition in CONDITIONS:
        prediction_sum = np.zeros_like(truths, dtype=np.float32)
        for seed in SEEDS:
            path = checkpoint_path(seed)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("training_size") != 252 or checkpoint.get("seed") != seed:
                raise RuntimeError(f"Checkpoint identity mismatch: {path}")
            predictions = predict_model(module, store, validation_ids, time_indices, checkpoint, mappings[condition])
            prediction_sum += predictions
            values = {metric: np.empty(len(predictions), dtype=np.float64) for metric in METRICS}
            for index in range(len(predictions)):
                metric_values = sample_metrics(module, predictions[index], truths[index], initial[index], y)
                for metric in METRICS:
                    values[metric][index] = metric_values[metric]
            state_metrics[(condition, seed)] = values
            seed_global[(condition, seed)] = {metric: float(values[metric].mean()) for metric in METRICS}
            for run_id in validation_ids:
                indices = state_index_by_run[run_id]
                seed_trajectory[(condition, seed, run_id)] = {
                    metric: float(values[metric][indices].mean()) for metric in METRICS
                }
            del predictions

        ensemble = prediction_sum / float(len(SEEDS))
        ensemble_values = {metric: np.empty(len(ensemble), dtype=np.float64) for metric in METRICS}
        for index in range(len(ensemble)):
            metric_values = sample_metrics(module, ensemble[index], truths[index], initial[index], y)
            for metric in METRICS:
                ensemble_values[metric][index] = metric_values[metric]
        ensemble_global[condition] = {metric: float(ensemble_values[metric].mean()) for metric in METRICS}
        for run_id in validation_ids:
            indices = state_index_by_run[run_id]
            ensemble_trajectory[(condition, run_id)] = {
                metric: float(ensemble_values[metric][indices].mean()) for metric in METRICS
            }
        del ensemble, prediction_sum

    metric_rows = []
    summaries = {}
    for condition in CONDITIONS:
        for metric in METRICS:
            summary = summarize([seed_global[(condition, seed)][metric] for seed in SEEDS])
            summaries[(condition, metric)] = summary
            metric_rows.append({
                "record_type": "five_seed_summary",
                "condition": condition,
                "metric": metric,
                **summary,
            })
            metric_rows.append({
                "record_type": "ensemble_global",
                "condition": condition,
                "metric": metric,
                "value": ensemble_global[condition][metric],
            })
        for seed in SEEDS:
            for metric in METRICS:
                metric_rows.append({
                    "record_type": "individual_seed",
                    "condition": condition,
                    "seed": seed,
                    "metric": metric,
                    "value": seed_global[(condition, seed)][metric],
                })

    trajectory_rows = []
    paired = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            for run_id in validation_ids:
                for metric in METRICS:
                    trajectory_rows.append({
                        "record_type": "seed_trajectory",
                        "condition": condition,
                        "seed": seed,
                        "trajectory_id": run_id,
                        "input_trajectory_id": mappings[condition][run_id],
                        "f_beta": design.loc[run_id, "f_beta"],
                        "morphology": design.loc[run_id, "morphology"],
                        "metric": metric,
                        "value": seed_trajectory[(condition, seed, run_id)][metric],
                    })
        for run_id in validation_ids:
            for metric in METRICS:
                trajectory_rows.append({
                    "record_type": "ensemble_trajectory",
                    "condition": condition,
                    "seed": "ensemble_mean",
                    "trajectory_id": run_id,
                    "input_trajectory_id": mappings[condition][run_id],
                    "f_beta": design.loc[run_id, "f_beta"],
                    "morphology": design.loc[run_id, "morphology"],
                    "metric": metric,
                    "value": ensemble_trajectory[(condition, run_id)][metric],
                })

    for metric in METRICS:
        normal = np.asarray([
            np.mean([seed_trajectory[("normal", seed, run_id)][metric] for seed in SEEDS])
            for run_id in validation_ids
        ])
        permuted = np.asarray([
            np.mean([seed_trajectory[("permuted", seed, run_id)][metric] for seed in SEEDS])
            for run_id in validation_ids
        ])
        difference = permuted - normal
        ci_low, ci_high = bootstrap_mean_ci(difference)
        nonzero = difference[~np.isclose(difference, 0.0, rtol=0.0, atol=1e-15)]
        p_value = float(wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue) if len(nonzero) else 1.0
        paired[metric] = {
            "count_error_increased": int(np.sum(difference > 0)),
            "count_error_decreased": int(np.sum(difference < 0)),
            "count_tied": int(np.sum(difference == 0)),
            "mean_difference": float(difference.mean()),
            "median_difference": float(np.median(difference)),
            "bootstrap_95_ci_low": ci_low,
            "bootstrap_95_ci_high": ci_high,
            "wilcoxon_p": p_value,
        }
        metric_rows.append({"record_type": "paired_summary", "metric": metric, **paired[metric]})
        for index, run_id in enumerate(validation_ids):
            trajectory_rows.append({
                "record_type": "paired_trajectory",
                "condition": "permuted_minus_normal",
                "seed": "five_seed_mean",
                "trajectory_id": run_id,
                "input_trajectory_id": permutation[run_id],
                "f_beta": design.loc[run_id, "f_beta"],
                "morphology": design.loc[run_id, "morphology"],
                "metric": metric,
                "normal": float(normal[index]),
                "permuted": float(permuted[index]),
                "difference_permuted_minus_normal": float(difference[index]),
            })
    write_csv(METRICS_PATH, metric_rows)
    write_csv(TRAJECTORY_PATH, trajectory_rows)

    grouped_rows = []
    for grouping, column in (("f_beta", "f_beta"), ("morphology", "morphology")):
        for group_value in sorted(design[column].unique(), key=str):
            run_ids = design.index[design[column] == group_value].tolist()
            for condition in CONDITIONS:
                for metric in METRICS:
                    values = [
                        seed_trajectory[(condition, seed, run_id)][metric]
                        for seed in SEEDS
                        for run_id in run_ids
                    ]
                    grouped_rows.append({
                        "grouping": grouping,
                        "group": group_value,
                        "condition": condition,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "trajectory_count": len(run_ids),
                        "seed_count": len(SEEDS),
                    })
    write_csv(GROUPED_PATH, grouped_rows)

    central_votes = sum(summaries[("permuted", metric)]["mean"] > summaries[("normal", metric)]["mean"] for metric in METRICS)
    ensemble_votes = sum(ensemble_global["permuted"][metric] > ensemble_global["normal"][metric] for metric in METRICS)
    paired_votes = sum(
        paired[metric]["mean_difference"] > 0
        and paired[metric]["median_difference"] > 0
        and paired[metric]["count_error_increased"] > paired[metric]["count_error_decreased"]
        for metric in METRICS
    )
    grouped_votes = 0
    grouped_total = 0
    grouped_frame = pd.DataFrame(grouped_rows)
    for grouping in grouped_frame.grouping.unique():
        for group_value in grouped_frame.loc[grouped_frame.grouping == grouping, "group"].unique():
            for metric in METRICS:
                subset = grouped_frame.loc[
                    (grouped_frame.grouping == grouping)
                    & (grouped_frame.group.astype(str) == str(group_value))
                    & (grouped_frame.metric == metric)
                ]
                normal_value = float(subset.loc[subset.condition == "normal", "mean"].iloc[0])
                permuted_value = float(subset.loc[subset.condition == "permuted", "mean"].iloc[0])
                grouped_votes += int(permuted_value > normal_value)
                grouped_total += 1
    if central_votes == 3 and ensemble_votes == 3 and paired_votes >= 2 and grouped_votes >= int(0.75 * grouped_total):
        decision = "SUPPORTED"
    elif central_votes >= 2 or ensemble_votes >= 2 or paired_votes >= 2:
        decision = "PARTIALLY_SUPPORTED"
    else:
        decision = "NOT_SUPPORTED"

    labels = ["delta-phase\nrelative L2", "Interface MAE\n(nm)", "Maximum-depth\nMAE (nm)"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    colors = ["#3B6FB6", "#D97835"]
    for index, metric in enumerate(METRICS):
        means = [summaries[(condition, metric)]["mean"] for condition in CONDITIONS]
        errors = [summaries[(condition, metric)]["sample_sd"] for condition in CONDITIONS]
        axes[index].bar([0, 1], means, yerr=errors, capsize=4, color=colors, width=0.62)
        axes[index].set_xticks([0, 1], ["Normal", "Permuted"])
        axes[index].set_ylabel(labels[index])
        axes[index].spines[["top", "right"]].set_visible(False)
        axes[index].grid(axis="y", alpha=0.2)
    fig.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)

    lines = [
        "# Final252 validation-only spatial microstructure permutation control",
        "",
        "## Scope",
        "",
        "- Final 252-trajectory, five-seed step-normalized best checkpoints.",
        "- Fixed 36 validation trajectories and 12 training-grid times only.",
        "- Within each beta-fraction × morphology cell, the four microstructures were cyclically shifted by one position after sorting by trajectory ID.",
        "- Target COMSOL fields and times remained attached to the original trajectory.",
        "- Independent evaluation/final-test target truth accessed: **NO**.",
        "",
        "## Five-seed results",
        "",
        "| Condition | Metric | Mean | Sample SD |",
        "|:---|:---|---:|---:|",
    ]
    for condition in CONDITIONS:
        for metric in METRICS:
            value = summaries[(condition, metric)]
            lines.append(f"| {condition} | {metric} | {value['mean']:.10g} | {value['sample_sd']:.10g} |")
    lines.extend([
        "",
        "## Ensemble mean-field results",
        "",
        "| Condition | delta-phase relative L2 | Interface MAE (nm) | Maximum-depth MAE (nm) |",
        "|:---|---:|---:|---:|",
        f"| normal | {ensemble_global['normal']['delta_phase_relative_l2']:.10g} | {ensemble_global['normal']['interface_mae_nm']:.10g} | {ensemble_global['normal']['maximum_depth_mae_nm']:.10g} |",
        f"| permuted | {ensemble_global['permuted']['delta_phase_relative_l2']:.10g} | {ensemble_global['permuted']['interface_mae_nm']:.10g} | {ensemble_global['permuted']['maximum_depth_mae_nm']:.10g} |",
        "",
        "## Trajectory-level paired analysis",
        "",
        "| Metric | Error increased | Error decreased | Mean D | Median D | Bootstrap 95% CI | Wilcoxon p |",
        "|:---|---:|---:|---:|---:|:---|---:|",
    ])
    for metric in METRICS:
        value = paired[metric]
        lines.append(
            f"| {metric} | {value['count_error_increased']} | {value['count_error_decreased']} | "
            f"{value['mean_difference']:.10g} | {value['median_difference']:.10g} | "
            f"[{value['bootstrap_95_ci_low']:.10g}, {value['bootstrap_95_ci_high']:.10g}] | {value['wilcoxon_p']:.6g} |"
        )
    lines.extend([
        "",
        "## Integrated judgment",
        "",
        f"- Five-seed central-tendency metrics degraded: {central_votes}/3.",
        f"- Ensemble metrics degraded: {ensemble_votes}/3.",
        f"- Trajectory-paired criteria supported degradation: {paired_votes}/3.",
        f"- Descriptive subgroup comparisons showing degradation: {grouped_votes}/{grouped_total}.",
        "- The conclusion integrates seed-level central tendency, ensemble fields, trajectory-level pairing, and descriptive subgroup consistency; no single seed or p-value determines the result.",
        "",
        f"**SPATIAL_INFORMATION_CONTROL: {decision}**",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"METRICS={METRICS_PATH}")
    print(f"TRAJECTORY={TRAJECTORY_PATH}")
    print(f"GROUPED={GROUPED_PATH}")
    print(f"REPORT={REPORT_PATH}")
    print(f"FIGURE={FIGURE_PATH}")
    print(f"DECISION={decision}")
    print(f"WALL_S={time.time() - started:.3f}")
    for condition in CONDITIONS:
        print(f"SUMMARY_{condition}={{{', '.join(f'{metric}: {summaries[(condition, metric)]}' for metric in METRICS)}}}")
        print(f"ENSEMBLE_{condition}={ensemble_global[condition]}")
    for metric in METRICS:
        print(f"PAIRED_{metric}={paired[metric]}")


if __name__ == "__main__":
    main()
