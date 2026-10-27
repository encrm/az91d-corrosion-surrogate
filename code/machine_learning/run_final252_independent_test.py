from __future__ import annotations

import csv
import hashlib
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AZ91D_DATA_ROOT", REPO_ROOT / "data"))
PROJECT = Path(os.environ.get("AZ91D_PROJECT_ROOT", DATA_ROOT / "project"))
EXT = Path(os.environ.get("AZ91D_EXTENSION_ROOT", DATA_ROOT / "extension"))
SCRIPT_DIR = Path(os.environ.get("AZ91D_SCRIPT_DIR", Path(__file__).resolve().parent))
FREEZE = PROJECT / "dataset216_freeze_audit_20260912"

ACCESS_RECORD = EXT / "FINAL_TEST_ACCESS_RECORD.md"
SEED_METRICS_PATH = EXT / "FINAL252_FINAL_TEST_SEED_METRICS.csv"
SUMMARY_PATH = EXT / "FINAL252_FINAL_TEST_SUMMARY.csv"
ENSEMBLE_PATH = EXT / "FINAL252_FINAL_TEST_ENSEMBLE_METRICS.csv"
TRAJECTORY_PATH = EXT / "FINAL252_FINAL_TEST_TRAJECTORY_METRICS.csv"
TIME_PATH = EXT / "FINAL252_FINAL_TEST_TIME_METRICS.csv"
REPORT_PATH = EXT / "FINAL252_FINAL_TEST_REPORT.md"
FIGURE_PATH = EXT / "FINAL252_FINAL_TEST_TIME_ERROR.png"
WORK = EXT / "final252_final_test_work"

SEEDS = (0, 1, 2, 3, 4)
FINAL_ROLE = "pristine_final_test"
OTHER_ROLE = "historical_audit_test"
METRICS = ("delta_phase_relative_l2", "interface_mae_nm", "maximum_depth_mae_nm")
TRAIN_HOURS = (0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24)
INTERPOLATION_HOURS = (3, 5, 7, 9, 14, 18, 22)
PROBE_HOURS = (11, 13, 15, 17, 19, 21, 23)
GROUP_HOURS = {
    "all_nonzero": TRAIN_HOURS + INTERPOLATION_HOURS + PROBE_HOURS,
    "training_grid": TRAIN_HOURS,
    "withheld_interpolation": INTERPOLATION_HOURS,
    "further_time_probes": PROBE_HOURS,
}
EXPECTED_PROTOCOL_HASH = "8d6687c6ffd47306e00fd4bcc73133de97167553a09bfd28aa4bcbc677829c7a"
EXPECTED_CHECKPOINT_HASHES = {
    0: "e6eec858705352d006880880b104481caa004d56883c6d88677e7a0d54930f5f",
    1: "84d3eb644e6033c0c1fc0ba23dd3ec8e123db49fa7ac28d7257144cfbcd08f54",
    2: "2e22049d8f7dc032ced2e40f5233ecfa3ac4fb71864b953c5bc053d4271acc29",
    3: "708222d478f3d680a62074616b69142e23a506aaaedc370d7c4f261a8703a066",
    4: "4786e90ce557753f6b6efc9d0e2eaf93a81ec0ae31a43426c292109512f72764",
}


def checkpoint_path(seed: int) -> Path:
    folder = "step_normalized_diagnostic" if seed < 3 else "step_normalized_full_learning_curve"
    return EXT / folder / "checkpoints" / f"step_normalized_n252_seed{seed}_best.pt"


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


def load_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    import run_scaling_after_extension as scaling

    module = scaling.load_training_module()
    module.ACCESS_LOG = WORK / "FINAL_TEST_AUTHORIZED_ACCESS.log"
    return module


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


def hour_index(hours: np.ndarray, target: float) -> int:
    found = np.flatnonzero(np.isclose(hours, target))
    if len(found) != 1:
        raise RuntimeError(f"Expected one state at {target} h, found {len(found)}")
    return int(found[0])


def summarize_rows(rows: list[dict], metric: str) -> tuple[float, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1))


def main() -> None:
    started = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    if not ACCESS_RECORD.exists():
        raise RuntimeError("FINAL_TEST_ACCESS_RECORD.md must exist before truth access")
    if sha256_file(EXT / "FINAL_MODEL_PROTOCOL_LOCK.md") != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("Final protocol hash mismatch")
    for seed in SEEDS:
        path = checkpoint_path(seed)
        if sha256_file(path) != EXPECTED_CHECKPOINT_HASHES[seed]:
            raise RuntimeError(f"Checkpoint hash mismatch for seed {seed}")

    manifest = pd.read_csv(FREEZE / "COMBINED216_PRODUCTION_MANIFEST.csv")
    final_rows = manifest.loc[manifest.split_role == FINAL_ROLE].copy()
    other_rows = manifest.loc[manifest.split_role == OTHER_ROLE].copy()
    if len(final_rows) != 18 or len(other_rows) != 18:
        raise RuntimeError("The two independent 18-trajectory roles are not uniquely defined")
    final_ids = final_rows.production_id.tolist()

    module = load_module()
    store = module.Store216()
    store.allowed = set(final_ids)
    store.forbidden = set(store.rows.index) - store.allowed
    if set(store.ids(FINAL_ROLE)) != set(final_ids):
        raise RuntimeError("Final-test role mismatch between metadata sources")

    expected_hours = (0.0,) + tuple(sorted(GROUP_HOURS["all_nonzero"]))
    actual_hours = tuple(float(value) for value in store.hours)
    if actual_hours != expected_hours:
        raise RuntimeError(f"Unexpected 27-state time grid: {actual_hours}")
    group_indices = {
        group: [hour_index(store.hours, hour) for hour in hours]
        for group, hours in GROUP_HOURS.items()
    }
    all_indices = [0] + group_indices["all_nonzero"]
    if len(group_indices["all_nonzero"]) != 26:
        raise RuntimeError("Expected 26 nonzero evaluation times")

    y = np.linspace(store.header.ymin, store.header.ymax, store.header.ny)
    ensemble_sum_path = WORK / "ensemble_absolute_phase_sum_float32.dat"
    ensemble_sum = np.memmap(
        ensemble_sum_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(final_ids), len(actual_hours), store.header.ny, store.header.nx),
    )
    ensemble_sum[:] = 0.0

    state_rows: list[dict] = []
    t0_rows: list[dict] = []
    device = torch.device("cuda")
    for seed in SEEDS:
        path = checkpoint_path(seed)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("training_size") != 252 or checkpoint.get("seed") != seed:
            raise RuntimeError(f"Checkpoint identity mismatch for seed {seed}")
        scales = checkpoint["scales"]
        model = module.TimeUNet(scales["mass_scale"], scales["current_scale"], 32, True).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if sum(parameter.numel() for parameter in model.parameters()) != 7_984_355:
            raise RuntimeError(f"Parameter count mismatch for seed {seed}")

        with torch.no_grad():
            for run_index, run_id in enumerate(final_ids):
                micro_np = store.micro(run_id).astype(np.float32)
                pf0 = store.phase0(run_id).astype(np.float32)
                micro = torch.from_numpy(micro_np[None, None]).to(device)
                for start in range(0, len(all_indices), 2):
                    time_indices = all_indices[start:start + 2]
                    batch_micro = micro.expand(len(time_indices), -1, -1, -1)
                    tau = torch.tensor(
                        [[store.times[index] / 86400.0] for index in time_indices],
                        dtype=torch.float32,
                        device=device,
                    )
                    with torch.autocast("cuda", dtype=torch.float16):
                        delta, _, _ = model(batch_micro, tau)
                    delta_np = delta[:, 0].float().cpu().numpy()
                    for local_index, time_index in enumerate(time_indices):
                        estimate = np.clip(pf0 + delta_np[local_index], 0.0, 1.0).astype(np.float32)
                        ensemble_sum[run_index, time_index] += estimate
                        hour = float(store.hours[time_index])
                        if time_index == 0:
                            t0_rows.append({
                                "seed": seed,
                                "trajectory_id": run_id,
                                "max_abs_predicted_delta_phase": float(np.max(np.abs(delta_np[local_index]))),
                            })
                            continue
                        truth = store.phase(run_id, time_index).astype(np.float32)
                        values = sample_metrics(module, estimate, truth, pf0, y)
                        state_rows.append({
                            "seed": seed,
                            "trajectory_id": run_id,
                            "f_beta": float(store.rows.loc[run_id, "f_beta"]),
                            "morphology_class": str(store.rows.loc[run_id, "morphology"]),
                            "time_h": hour,
                            "time_role": next(group for group, hours in GROUP_HOURS.items() if group != "all_nonzero" and hour in hours),
                            **values,
                        })
        del model
        torch.cuda.empty_cache()
        ensemble_sum.flush()

    ensemble_state_rows: list[dict] = []
    for run_index, run_id in enumerate(final_ids):
        pf0 = store.phase0(run_id).astype(np.float32)
        for time_index in group_indices["all_nonzero"]:
            estimate = np.asarray(ensemble_sum[run_index, time_index] / float(len(SEEDS)), dtype=np.float32)
            truth = store.phase(run_id, time_index).astype(np.float32)
            values = sample_metrics(module, estimate, truth, pf0, y)
            hour = float(store.hours[time_index])
            ensemble_state_rows.append({
                "trajectory_id": run_id,
                "f_beta": float(store.rows.loc[run_id, "f_beta"]),
                "morphology_class": str(store.rows.loc[run_id, "morphology"]),
                "time_h": hour,
                "time_role": next(group for group, hours in GROUP_HOURS.items() if group != "all_nonzero" and hour in hours),
                **values,
            })

    state_frame = pd.DataFrame(state_rows)
    ensemble_frame = pd.DataFrame(ensemble_state_rows)
    if len(state_frame) != 5 * 18 * 26 or len(ensemble_frame) != 18 * 26:
        raise RuntimeError("Unexpected number of evaluated states")

    seed_metric_rows: list[dict] = []
    for seed in SEEDS:
        seed_frame = state_frame.loc[state_frame.seed == seed]
        for group, hours in GROUP_HOURS.items():
            subset = seed_frame.loc[seed_frame.time_h.isin(hours)]
            seed_metric_rows.append({
                "seed": seed,
                "time_group": group,
                "trajectory_count": 18,
                "state_count": len(subset),
                **{metric: float(subset[metric].mean()) for metric in METRICS},
            })
    write_csv(SEED_METRICS_PATH, seed_metric_rows)

    summary_rows: list[dict] = []
    for group in GROUP_HOURS:
        group_rows = [row for row in seed_metric_rows if row["time_group"] == group]
        for metric in METRICS:
            mean, sd = summarize_rows(group_rows, metric)
            summary_rows.append({
                "time_group": group,
                "metric": metric,
                "five_seed_mean": mean,
                "sample_sd_ddof1": sd,
                "seed_count": 5,
            })
    write_csv(SUMMARY_PATH, summary_rows)

    ensemble_metric_rows: list[dict] = []
    for group, hours in GROUP_HOURS.items():
        subset = ensemble_frame.loc[ensemble_frame.time_h.isin(hours)]
        ensemble_metric_rows.append({
            "time_group": group,
            "trajectory_count": 18,
            "state_count": len(subset),
            **{metric: float(subset[metric].mean()) for metric in METRICS},
        })
    write_csv(ENSEMBLE_PATH, ensemble_metric_rows)

    trajectory_rows: list[dict] = []
    for run_id in final_ids:
        run_frame = state_frame.loc[state_frame.trajectory_id == run_id]
        for group, hours in GROUP_HOURS.items():
            subset = run_frame.loc[run_frame.time_h.isin(hours)]
            trajectory_rows.append({
                "trajectory_id": run_id,
                "f_beta": float(store.rows.loc[run_id, "f_beta"]),
                "morphology_class": str(store.rows.loc[run_id, "morphology"]),
                "time_group": group,
                "time_count": len(hours),
                "seed_count": 5,
                **{metric: float(subset[metric].mean()) for metric in METRICS},
            })
    write_csv(TRAJECTORY_PATH, trajectory_rows)

    time_rows: list[dict] = []
    for hour in sorted(GROUP_HOURS["all_nonzero"]):
        subset = state_frame.loc[np.isclose(state_frame.time_h, hour)]
        ensemble_subset = ensemble_frame.loc[np.isclose(ensemble_frame.time_h, hour)]
        role = str(subset.time_role.iloc[0])
        time_rows.append({
            "time_h": hour,
            "time_role": role,
            "trajectory_count": 18,
            "seed_count": 5,
            **{metric: float(subset[metric].mean()) for metric in METRICS},
            **{f"ensemble_{metric}": float(ensemble_subset[metric].mean()) for metric in METRICS},
        })
    write_csv(TIME_PATH, time_rows)

    ensemble_all = ensemble_frame.groupby("trajectory_id", sort=False)["delta_phase_relative_l2"].mean().sort_values()
    ordered_ids = ensemble_all.index.tolist()
    lower_id = ordered_ids[int(round(0.25 * (len(ordered_ids) - 1)))]
    median_id = ordered_ids[int(round(0.50 * (len(ordered_ids) - 1)))]
    upper_id = ordered_ids[int(round(0.75 * (len(ordered_ids) - 1)))]

    colors = {
        "training_grid": "#2468a2",
        "withheld_interpolation": "#e07a1f",
        "further_time_probes": "#2f8f5b",
    }
    markers = {"training_grid": "o", "withheld_interpolation": "s", "further_time_probes": "^"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    plot_specs = [
        ("delta_phase_relative_l2", "Delta-phase relative L2"),
        ("interface_mae_nm", "Interface MAE (nm)"),
        ("maximum_depth_mae_nm", "Maximum-depth MAE (nm)"),
    ]
    time_frame = pd.DataFrame(time_rows)
    for axis, (metric, label) in zip(axes, plot_specs):
        for role in ("training_grid", "withheld_interpolation", "further_time_probes"):
            subset = time_frame.loc[time_frame.time_role == role]
            axis.scatter(subset.time_h, subset[metric], color=colors[role], marker=markers[role], s=42, label=role.replace("_", " "))
            axis.plot(subset.time_h, subset[metric], color=colors[role], linewidth=0.8, alpha=0.5)
        axis.set_xlabel("Time (h)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)

    finite_seed = bool(np.isfinite(state_frame[list(METRICS)].to_numpy()).all())
    finite_ensemble = bool(np.isfinite(ensemble_frame[list(METRICS)].to_numpy()).all())
    t0_max = float(pd.DataFrame(t0_rows).max_abs_predicted_delta_phase.max())
    summary_lookup = {(row["time_group"], row["metric"]): row for row in summary_rows}
    ensemble_lookup = {row["time_group"]: row for row in ensemble_metric_rows}

    def summary_line(group: str, metric: str) -> str:
        row = summary_lookup[(group, metric)]
        return f"{row['five_seed_mean']:.10g} ± {row['sample_sd_ddof1']:.10g}"

    report_lines = [
        "# Final252 final independent-test report",
        "",
        "## Locked evaluation state",
        "",
        "- Final-test role: `pristine_final_test` (18 trajectories).",
        "- The separate `historical_audit_test` role was not included.",
        "- Final training size: 252 trajectories.",
        "- Five locked checkpoints: seeds 0, 1, 2, 3, and 4.",
        "- Model parameter count: 7,984,355.",
        "- No retraining, fine-tuning, checkpoint reselection, hyperparameter change, loss change, or post-test model modification was performed.",
        "- Final-test truth was accessed only after training size, architecture, hyperparameters, checkpoints, evaluation metrics and spatial-information control had been locked.",
        "",
        "## Validation model-selection performance (locked; not recomputed)",
        "",
        "| Metric | Five-seed mean ± sample SD |",
        "|---|---:|",
        "| Delta-phase relative L2 | 0.0461006 ± 0.0102263 |",
        "| Interface MAE (nm) | 4.48013 ± 1.44883 |",
        "| Maximum-depth MAE (nm) | 6.86713 ± 2.66121 |",
        "",
        "## Final independent-test performance",
        "",
        "Five-seed summaries average state-level metrics within each time group for each seed, then report mean ± sample SD across the five independent training seeds.",
        "",
        "| Time group | States | Delta-phase relative L2 | Interface MAE (nm) | Maximum-depth MAE (nm) |",
        "|---|---:|---:|---:|---:|",
    ]
    state_counts = {row["time_group"]: row["state_count"] for row in seed_metric_rows if row["seed"] == 0}
    for group in GROUP_HOURS:
        report_lines.append(
            f"| {group.replace('_', ' ')} | {state_counts[group]} | {summary_line(group, 'delta_phase_relative_l2')} | "
            f"{summary_line(group, 'interface_mae_nm')} | {summary_line(group, 'maximum_depth_mae_nm')} |"
        )
    report_lines.extend([
        "",
        "## Ensemble mean-field performance",
        "",
        "| Time group | States | Delta-phase relative L2 | Interface MAE (nm) | Maximum-depth MAE (nm) |",
        "|---|---:|---:|---:|---:|",
    ])
    for group in GROUP_HOURS:
        row = ensemble_lookup[group]
        report_lines.append(
            f"| {group.replace('_', ' ')} | {row['state_count']} | {row['delta_phase_relative_l2']:.10g} | "
            f"{row['interface_mae_nm']:.10g} | {row['maximum_depth_mae_nm']:.10g} |"
        )
    report_lines.extend([
        "",
        "## Initial condition and integrity checks",
        "",
        f"- Maximum absolute predicted delta phase at t = 0 across all trajectories and seeds: {t0_max:.10g}.",
        f"- Single-seed state metrics finite: {'YES' if finite_seed else 'NO'}.",
        f"- Ensemble state metrics finite: {'YES' if finite_ensemble else 'NO'}.",
        f"- Evaluated nonzero states per seed: {18 * 26}.",
        "- Auxiliary scalar heads were not assigned new post hoc metrics; the locked three spatial metrics remain the formal final-test endpoints.",
        "",
        "## Predefined representative-case rule",
        "",
        "Trajectories were ranked by ensemble all-26-time delta-phase relative L2. No visual case selection was used.",
        "",
        f"- Lower-quartile-error trajectory: `{lower_id}`",
        f"- Median-error trajectory: `{median_id}`",
        f"- Upper-quartile-error trajectory: `{upper_id}`",
        "",
        "## Status",
        "",
        "**FINAL_TEST_STATUS = COMPLETE**",
        "",
        f"- Evaluation wall time: {time.time() - started:.1f} s",
        "- Post-test model modification: NONE",
    ])
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (WORK / "FINAL_TEST_STATUS_COMPLETE.flag").write_text("FINAL_TEST_STATUS = COMPLETE\n", encoding="ascii")
    (WORK / "FINAL_TEST_AUTHORIZED_ACCESS.log").write_text(
        "Final-test truth access authorized by FINAL_TEST_ACCESS_RECORD.md.\n"
        f"Role: {FINAL_ROLE}\nCount: {len(final_ids)}\nStatus: COMPLETE\n",
        encoding="utf-8",
    )
    del ensemble_sum


if __name__ == "__main__":
    main()
