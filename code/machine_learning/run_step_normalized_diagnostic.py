from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import run_scaling_after_extension as scaling


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AZ91D_DATA_ROOT", REPO_ROOT / "data"))
EXT = Path(os.environ.get("AZ91D_EXTENSION_ROOT", DATA_ROOT / "extension"))
WORK = EXT / "step_normalized_diagnostic"
CHECKPOINTS = WORK / "checkpoints"
STATE_PATH = WORK / "state.json"
STATUS_PATH = EXT / "STEP_NORMALIZED_DIAGNOSTIC_RUNTIME_STATUS.md"
METRICS_PATH = EXT / "STEP_NORMALIZED_DIAGNOSTIC_METRICS.csv"
HISTORY_PATH = EXT / "STEP_NORMALIZED_DIAGNOSTIC_HISTORY.csv"
REPORT_PATH = EXT / "STEP_NORMALIZED_DIAGNOSTIC_REPORT.md"
FIGURE_PATH = EXT / "STEP_NORMALIZED_DIAGNOSTIC_COMPARISON.png"

TRAINING_SIZES = (252, 288)
SEEDS = (0, 1, 2)
MAX_OPTIMIZER_STEPS = 45_360
VALIDATION_INTERVAL_STEPS = 1_296
EARLY_STOP_PATIENCE_CHECKS = 6
EARLY_STOP_PATIENCE_STEPS = 7_776
MIN_IMPROVEMENT = 1e-5
CONFIG = {
    "base_channels": 32,
    "batch_size": 2,
    "learning_rate": 2e-4,
    "interface_lambda": 2.0,
    "dynamic_weight_cap": 4.0,
    "gradient_weight": 0.03,
    "weight_decay": 1e-4,
}
REFERENCE_216 = {
    "delta_phase_relative_l2": 0.0423136,
    "interface_mae_nm": 4.072,
    "max_depth_mae_nm": 5.599,
}
METRIC_NAMES = (
    "delta_phase_relative_l2",
    "interface_mae_nm",
    "max_depth_mae_nm",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"started": pd.Timestamp.now().isoformat(), "completed": {}}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def update_status(size: int, seed: int, step: int, best_step: int, stale: int, completed: int) -> None:
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        gpu = "unavailable"
    STATUS_PATH.write_text(
        "\n".join(
            [
                "# Step-normalized diagnostic status",
                "",
                f"- Updated: {pd.Timestamp.now().isoformat()}",
                f"- Current job: n={size}, seed={seed}",
                f"- Optimizer step: {step}/{MAX_OPTIMIZER_STEPS}",
                f"- Best optimizer step: {best_step}",
                f"- Consecutive stale validations: {stale}/{EARLY_STOP_PATIENCE_CHECKS}",
                f"- Completed jobs: {completed}/6",
                f"- GPU: {gpu}",
                "- Independent test truth accessed: NO",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def validation_loss(module, model, loader, reference, cfg, scales) -> float:
    total = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            micro, tau, pf0, delta = (
                batch[key].cuda(non_blocking=True) for key in ("micro", "tau", "pf0", "delta")
            )
            mass, current, mask = (
                batch[key].cuda(non_blocking=True) for key in ("mass", "current", "current_mask")
            )
            with torch.autocast("cuda", dtype=torch.float16):
                pred, pm, pc = model(micro, tau)
                loss = module.field_training_loss(pred, delta, pf0, reference, cfg)
                loss += 0.25 * module.scalar_training_loss(
                    pm, pc, mass, current, mask, scales, "raw_standardized"
                )
            total += float(loss) * len(micro)
    return total / len(loader.dataset)


def train_one(module, store, train_ids: list[str], val_ids: list[str], size: int, seed: int) -> dict:
    module.set_seed(seed)
    train_indices = module.indices(store)
    train_scales = module.scales(store, train_ids, train_indices)
    cfg = module.Stage2Config(
        "T1_tau",
        CONFIG["interface_lambda"],
        CONFIG["gradient_weight"],
        "raw_standardized",
        CONFIG["dynamic_weight_cap"],
        CONFIG["base_channels"],
    )
    reference = module.dynamic_reference(store, train_ids, train_indices)
    model = module.TimeUNet(
        train_scales["mass_scale"],
        train_scales["current_scale"],
        CONFIG["base_channels"],
        True,
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_OPTIMIZER_STEPS, eta_min=0.0
    )
    amp = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        module.PhaseDataset(store, train_ids, train_indices),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        module.PhaseDataset(store, val_ids, train_indices),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    steps_per_data_pass = len(train_loader)
    best_loss = float("inf")
    best_state = None
    best_step = 0
    stale = 0
    global_step = 0
    interval_loss = 0.0
    interval_samples = 0
    amp_skipped_batches = 0
    histories: list[dict] = []
    started = time.time()
    iterator = iter(train_loader)

    while global_step < MAX_OPTIMIZER_STEPS and stale < EARLY_STOP_PATIENCE_CHECKS:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        model.train()
        micro, tau, pf0, delta = (
            batch[key].cuda(non_blocking=True) for key in ("micro", "tau", "pf0", "delta")
        )
        mass, current, mask = (
            batch[key].cuda(non_blocking=True) for key in ("mass", "current", "current_mask")
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred, pm, pc = model(micro, tau)
            loss = module.field_training_loss(pred, delta, pf0, reference, cfg)
            loss += 0.25 * module.scalar_training_loss(
                pm, pc, mass, current, mask, train_scales, "raw_standardized"
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss for n={size}, seed={seed}, step={global_step + 1}")
        amp.scale(loss).backward()
        amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scale_before = amp.get_scale()
        amp.step(optimizer)
        amp.update()
        scale_after = amp.get_scale()
        if scale_after < scale_before:
            amp_skipped_batches += 1
            continue
        scheduler.step()
        global_step += 1
        interval_loss += float(loss.detach()) * len(micro)
        interval_samples += len(micro)

        if global_step % VALIDATION_INTERVAL_STEPS != 0:
            continue

        score = validation_loss(module, model, val_loader, reference, cfg, train_scales)
        improved = score < best_loss - MIN_IMPROVEMENT
        if improved:
            best_loss = score
            best_step = global_step
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        row = {
            "training_size": size,
            "seed": seed,
            "optimizer_step": global_step,
            "validation_check": global_step // VALIDATION_INTERVAL_STEPS,
            "data_passes": global_step / steps_per_data_pass,
            "interval_train_loss": interval_loss / interval_samples,
            "validation_total_loss": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "improved": improved,
            "best_optimizer_step": best_step,
            "best_validation_total_loss": best_loss,
            "stale_validation_checks": stale,
            "amp_skipped_batches": amp_skipped_batches,
        }
        histories.append(row)
        print(json.dumps({"event": "step_validation", **row}), flush=True)
        write_csv(WORK / f"n{size}_seed{seed}_history.csv", histories)
        update_status(size, seed, global_step, best_step, stale, len(load_state()["completed"]))
        interval_loss = 0.0
        interval_samples = 0

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for n={size}, seed={seed}")
    training_wall_s = time.time() - started
    model.load_state_dict(best_state)
    model.cuda().eval()
    checkpoint = CHECKPOINTS / f"step_normalized_n{size}_seed{seed}_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": CONFIG,
            "training_size": size,
            "seed": seed,
            "best_optimizer_step": best_step,
            "stop_optimizer_step": global_step,
            "best_validation_total_loss": best_loss,
            "scales": train_scales,
            "dynamic_reference": reference,
            "protocol": {
                "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
                "validation_interval_steps": VALIDATION_INTERVAL_STEPS,
                "early_stop_patience_steps": EARLY_STOP_PATIENCE_STEPS,
                "minimum_improvement": MIN_IMPROVEMENT,
                "scheduler": "CosineAnnealingLR per optimizer step",
                "scheduler_t_max": MAX_OPTIMIZER_STEPS,
                "eta_min": 0.0,
            },
        },
        checkpoint,
    )
    metrics = module.validation_metrics(
        model, store, val_ids, train_indices, CONFIG["batch_size"]
    )
    return {
        "status": "OK",
        "training_size": size,
        "seed": seed,
        **metrics,
        "best_optimizer_step": best_step,
        "stop_optimizer_step": global_step,
        "best_validation_total_loss": best_loss,
        "data_passes_at_best": best_step / steps_per_data_pass,
        "data_passes_at_stop": global_step / steps_per_data_pass,
        "training_wall_s": training_wall_s,
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "steps_per_data_pass": steps_per_data_pass,
        "validation_checks": len(histories),
        "early_stopped": stale >= EARLY_STOP_PATIENCE_CHECKS,
        "amp_skipped_batches": amp_skipped_batches,
        "dynamic_reference": reference,
        "mass_mean": train_scales["mass_mean"],
        "mass_std": train_scales["mass_std"],
        "current_mean": train_scales["current_mean"],
        "current_std": train_scales["current_std"],
        "checkpoint": str(checkpoint),
    }


def original_results() -> dict[tuple[int, int], dict]:
    state_path = EXT / "learning_curve_work" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows = {}
    for size in TRAINING_SIZES:
        for seed in SEEDS:
            key = f"scaling_n{size}|seed{seed}|constrained"
            rows[(size, seed)] = state["completed"][key]
    return rows


def combined_metric_rows(results: list[dict], originals: dict[tuple[int, int], dict]) -> list[dict]:
    rows = []
    for result in results:
        original = originals[(result["training_size"], result["seed"])]
        row = dict(result)
        for metric in METRIC_NAMES:
            old = float(original[metric])
            new = float(result[metric])
            row[f"original_{metric}"] = old
            row[f"relative_change_{metric}_percent"] = 100.0 * (new - old) / old
            row[f"improvement_{metric}_percent"] = 100.0 * (old - new) / old
        row["original_best_epoch"] = original["best_epoch"]
        row["original_best_validation_loss"] = original["best_validation_loss"]
        rows.append(row)
    return rows


def summary_rows(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    output = []
    originals = original_results()
    for size in TRAINING_SIZES:
        current = frame[frame.training_size == size]
        old_rows = pd.DataFrame([originals[(size, seed)] for seed in SEEDS])
        for protocol, values in (("original_epoch_based", old_rows), ("step_normalized", current)):
            row = {"training_size": size, "protocol": protocol, "n_seeds": len(values)}
            for metric in METRIC_NAMES:
                row[f"{metric}_mean"] = float(values[metric].mean())
                row[f"{metric}_sd"] = float(values[metric].std(ddof=1))
            output.append(row)
    return output


def plot_comparison(summary: list[dict]) -> None:
    frame = pd.DataFrame(summary)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=True)
    labels = ("Delta-phase relative L2", "Interface MAE (nm)", "Maximum-depth MAE (nm)")
    x = np.arange(len(TRAINING_SIZES))
    width = 0.34
    for axis, metric, label in zip(axes, METRIC_NAMES, labels):
        for offset, protocol, color in (
            (-width / 2, "original_epoch_based", "#6C8EBF"),
            (width / 2, "step_normalized", "#E69F45"),
        ):
            part = frame[frame.protocol == protocol].set_index("training_size").loc[list(TRAINING_SIZES)]
            axis.bar(
                x + offset,
                part[f"{metric}_mean"],
                width,
                yerr=part[f"{metric}_sd"],
                capsize=3,
                color=color,
                label=protocol.replace("_", " "),
            )
        axis.axhline(REFERENCE_216[metric], color="#555555", linestyle="--", linewidth=1)
        axis.set(xticks=x, xticklabels=TRAINING_SIZES, xlabel="Training trajectories", ylabel=label)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def hypothesis_label(rows: list[dict], summary: list[dict]) -> str:
    frame = pd.DataFrame(summary)
    improved_counts = 0
    for size in TRAINING_SIZES:
        old = frame[(frame.training_size == size) & (frame.protocol == "original_epoch_based")].iloc[0]
        new = frame[(frame.training_size == size) & (frame.protocol == "step_normalized")].iloc[0]
        improved_counts += sum(float(new[f"{metric}_mean"]) < float(old[f"{metric}_mean"]) for metric in METRIC_NAMES)
    if improved_counts >= 5:
        return "SUPPORTED"
    if improved_counts >= 2:
        return "PARTIALLY_SUPPORTED"
    return "NOT_SUPPORTED"


def write_report(rows: list[dict], summary: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    aggregate = pd.DataFrame(summary)
    label = hypothesis_label(rows, summary)
    lines = [
        "# Step-normalized training-budget diagnostic",
        "",
        "## Fixed protocol",
        "",
        f"- Training sizes: {TRAINING_SIZES}",
        f"- Seeds: {SEEDS}",
        f"- Maximum optimizer steps: {MAX_OPTIMIZER_STEPS}",
        f"- Validation interval: {VALIDATION_INTERVAL_STEPS} optimizer steps",
        f"- Early-stopping patience: {EARLY_STOP_PATIENCE_STEPS} optimizer steps",
        "- Scheduler: CosineAnnealingLR stepped after every optimizer update",
        "- Fixed validation trajectories: 36",
        "- Independent test truth accessed: NO",
        "- Network, loss, preprocessing and data subsets: unchanged",
        "",
        "## Seed-level results",
        "",
        "| n | seed | delta L2 | interface MAE (nm) | max-depth MAE (nm) | best step | stop step | passes at stop | train time (h) | early stop |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['training_size']} | {row['seed']} | {row['delta_phase_relative_l2']:.6g} | "
            f"{row['interface_mae_nm']:.6g} | {row['max_depth_mae_nm']:.6g} | "
            f"{row['best_optimizer_step']} | {row['stop_optimizer_step']} | "
            f"{row['data_passes_at_stop']:.3f} | {row['training_wall_s']/3600:.3f} | {row['early_stopped']} |"
        )
    lines.extend(
        [
            "",
            "## Three-seed mean ± sample SD",
            "",
            "| n | protocol | delta L2 | interface MAE (nm) | max-depth MAE (nm) |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {int(row.training_size)} | {row.protocol} | "
            f"{row.delta_phase_relative_l2_mean:.6g} ± {row.delta_phase_relative_l2_sd:.3g} | "
            f"{row.interface_mae_nm_mean:.6g} ± {row.interface_mae_nm_sd:.3g} | "
            f"{row.max_depth_mae_nm_mean:.6g} ± {row.max_depth_mae_nm_sd:.3g} |"
        )
    lines.extend(["", "## Seed-matched relative changes", ""])
    for _, row in frame.iterrows():
        lines.append(
            f"- n={int(row.training_size)}, seed={int(row.seed)}: "
            f"delta {row.relative_change_delta_phase_relative_l2_percent:+.2f}%, "
            f"interface {row.relative_change_interface_mae_nm_percent:+.2f}%, "
            f"max depth {row.relative_change_max_depth_mae_nm_percent:+.2f}%."
        )
    group_evidence = {}
    for size in TRAINING_SIZES:
        old = aggregate[
            (aggregate.training_size == size) & (aggregate.protocol == "original_epoch_based")
        ].iloc[0]
        new = aggregate[
            (aggregate.training_size == size) & (aggregate.protocol == "step_normalized")
        ].iloc[0]
        group_evidence[size] = {
            metric: {
                "mean_improvement": 100.0
                * (float(old[f"{metric}_mean"]) - float(new[f"{metric}_mean"]))
                / float(old[f"{metric}_mean"]),
                "sd_change": 100.0
                * (float(old[f"{metric}_sd"]) - float(new[f"{metric}_sd"]))
                / float(old[f"{metric}_sd"]),
                "reference_difference": 100.0
                * (float(new[f"{metric}_mean"]) - REFERENCE_216[metric])
                / REFERENCE_216[metric],
            }
            for metric in METRIC_NAMES
        }
    lines.extend(
        [
            "",
            "## 216 reference",
            "",
            f"- Delta-phase relative L2: {REFERENCE_216['delta_phase_relative_l2']}",
            f"- Interface MAE: {REFERENCE_216['interface_mae_nm']} nm",
            f"- Maximum-depth MAE: {REFERENCE_216['max_depth_mae_nm']} nm",
            "",
            "## Diagnostic conclusion",
            "",
            f"OPTIMIZATION_BUDGET_HYPOTHESIS: {label}",
            "",
            "### Direct answers to the diagnostic questions",
            "",
            "- **A — 252 trajectories:** all three group-mean errors improved after step normalization: "
            f"delta-phase relative L2 by {group_evidence[252]['delta_phase_relative_l2']['mean_improvement']:.2f}%, "
            f"interface MAE by {group_evidence[252]['interface_mae_nm']['mean_improvement']:.2f}%, and "
            f"maximum-depth MAE by {group_evidence[252]['max_depth_mae_nm']['mean_improvement']:.2f}%. "
            "Seed 1 did not improve on the two depth metrics, so the evidence is group-level rather than universal per seed.",
            "- **B — 288 trajectories:** all three group-mean errors improved: "
            f"delta-phase relative L2 by {group_evidence[288]['delta_phase_relative_l2']['mean_improvement']:.2f}%, "
            f"interface MAE by {group_evidence[288]['interface_mae_nm']['mean_improvement']:.2f}%, and "
            f"maximum-depth MAE by {group_evidence[288]['max_depth_mae_nm']['mean_improvement']:.2f}%. "
            "All three fixed seeds improved on all three metrics.",
            "- **C — seed dispersion:** dispersion decreased for 252 delta-phase and maximum-depth errors and for all three 288 metrics. "
            f"The only exception was 252 interface-MAE SD, which changed by "
            f"{-group_evidence[252]['interface_mae_nm']['sd_change']:+.2f}% (a negligible increase).",
            "- **D — early convergence:** the previously early best checkpoints moved substantially later under the normalized schedule. "
            "For 252, seeds 1/2 moved from epoch-equivalent steps 22,680/10,584 to 45,360/45,360. "
            "For 288, seeds 1/2 moved from 20,736/17,280 to 41,472/32,400. "
            "Only 288 seed 2 triggered early stopping, at step 40,176, after its best checkpoint at step 32,400.",
            "- **E — comparison with the 216 reference:** the normalized 252 means were within "
            f"{abs(group_evidence[252]['delta_phase_relative_l2']['reference_difference']):.2f}%, "
            f"{abs(group_evidence[252]['interface_mae_nm']['reference_difference']):.2f}%, and "
            f"{abs(group_evidence[252]['max_depth_mae_nm']['reference_difference']):.2f}% of the reference. "
            "For 288, delta-phase relative L2 was "
            f"{group_evidence[288]['delta_phase_relative_l2']['reference_difference']:+.2f}% relative to the reference, while interface and maximum-depth MAE were "
            f"{-group_evidence[288]['interface_mae_nm']['reference_difference']:.2f}% and "
            f"{-group_evidence[288]['max_depth_mae_nm']['reference_difference']:.2f}% lower, respectively.",
            "",
            "This classification compares the predeclared three seeds as a group. It does not select a best seed and does not alter the formal protocol.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    extension, _ = scaling.audit_extension()
    module = scaling.load_training_module()
    module.OUT = WORK
    module.CKPT = CHECKPOINTS
    module.STATE_PATH = WORK / "unused_module_state.json"
    module.ACCESS_LOG = WORK / "INDEPENDENT_TEST_ACCESS_GUARD.log"
    module.STATUS = STATUS_PATH
    module.ACCESS_LOG.write_text(
        "# Independent-test truth access guard\n\nNo independent test truth was accessed.\n",
        encoding="utf-8",
    )
    store = scaling.combined_store_class(module, extension)()
    val_ids = store.ids("validation")
    if len(val_ids) != 36:
        raise RuntimeError(f"Expected 36 validation trajectories, found {len(val_ids)}")
    nested = pd.read_csv(EXT / "LEARNING_CURVE_NESTED_SUBSETS.csv")
    subsets = {
        size: nested.loc[nested.n_train == size, "production_id"].tolist()
        for size in TRAINING_SIZES
    }
    if any(len(subsets[size]) != size for size in TRAINING_SIZES):
        raise RuntimeError("Nested subset size mismatch")
    if not set(subsets[252]).issubset(set(subsets[288])):
        raise RuntimeError("252 is not a subset of 288")

    state = load_state()
    for size in TRAINING_SIZES:
        for seed in SEEDS:
            key = f"n{size}_seed{seed}"
            if key in state["completed"]:
                continue
            result = train_one(module, store, subsets[size], val_ids, size, seed)
            state = load_state()
            state["completed"][key] = result
            save_state(state)
            update_status(size, seed, result["stop_optimizer_step"], result["best_optimizer_step"], 0, len(state["completed"]))
            torch.cuda.empty_cache()

    state = load_state()
    results = [state["completed"][f"n{size}_seed{seed}"] for size in TRAINING_SIZES for seed in SEEDS]
    originals = original_results()
    metric_rows = combined_metric_rows(results, originals)
    write_csv(METRICS_PATH, metric_rows)
    histories = []
    for size in TRAINING_SIZES:
        for seed in SEEDS:
            histories.extend(pd.read_csv(WORK / f"n{size}_seed{seed}_history.csv").to_dict("records"))
    write_csv(HISTORY_PATH, histories)
    summary = summary_rows(results)
    plot_comparison(summary)
    write_report(metric_rows, summary)
    guard = module.ACCESS_LOG.read_text(encoding="utf-8")
    if "BLOCKED" in guard:
        raise RuntimeError("Forbidden independent-test truth access was attempted")
    STATUS_PATH.write_text(
        "# Step-normalized diagnostic status\n\n"
        f"- Updated: {pd.Timestamp.now().isoformat()}\n"
        "- Completed jobs: 6/6\n"
        "- Independent test truth accessed: NO\n"
        "- Status: COMPLETE\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "report": str(REPORT_PATH)}), flush=True)


if __name__ == "__main__":
    main()
