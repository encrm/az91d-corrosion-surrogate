from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import run_scaling_after_extension as scaling
import run_step_normalized_diagnostic as diagnostic


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AZ91D_DATA_ROOT", REPO_ROOT / "data"))
EXT = Path(os.environ.get("AZ91D_EXTENSION_ROOT", DATA_ROOT / "extension"))
WORK = EXT / "step_normalized_full_learning_curve"
CHECKPOINTS = WORK / "checkpoints"
STATE_PATH = WORK / "state.json"
STATUS_PATH = EXT / "STEP_NORMALIZED_RUNTIME_STATUS.md"
STDOUT_PATH = EXT / "STEP_NORMALIZED_FULL_STDOUT.log"
STDERR_PATH = EXT / "STEP_NORMALIZED_FULL_STDERR.log"
METRICS_PATH = EXT / "STEP_NORMALIZED_FULL_SEED_METRICS.csv"
CURVE_PATH = EXT / "STEP_NORMALIZED_FULL_LEARNING_CURVE.csv"
FIGURE_PATH = EXT / "STEP_NORMALIZED_FULL_LEARNING_CURVE.png"
REPORT_PATH = EXT / "STEP_NORMALIZED_FULL_TRAINING_REPORT.md"
ACCESS_LOG = WORK / "INDEPENDENT_TEST_ACCESS_GUARD.log"

TRAINING_SIZES = (144, 180, 216, 252, 288)
SEEDS = (0, 1, 2, 3, 4)
REUSED = {(size, seed) for size in (252, 288) for seed in (0, 1, 2)}
MAX_OPTIMIZER_STEPS = 45_360
VALIDATION_INTERVAL_STEPS = 1_296
EARLY_STOP_PATIENCE_CHECKS = 6
EARLY_STOP_PATIENCE_STEPS = 7_776
MIN_IMPROVEMENT = 1e-5
CONFIG = dict(diagnostic.CONFIG)
METRICS = (
    "delta_phase_relative_l2",
    "interface_mae_nm",
    "max_depth_mae_nm",
)


def emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, "time": pd.Timestamp.now().isoformat(), **fields}), flush=True)


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "started": pd.Timestamp.now().isoformat(),
        "protocol": {
            "training_sizes": TRAINING_SIZES,
            "seeds": SEEDS,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "validation_interval_steps": VALIDATION_INTERVAL_STEPS,
            "early_stop_patience_steps": EARLY_STOP_PATIENCE_STEPS,
            "minimum_improvement": MIN_IMPROVEMENT,
            "config": CONFIG,
        },
        "completed": {},
        "failed": {},
        "retries": {},
        "current": None,
    }


def save_state(state: dict) -> None:
    atomic_json(STATE_PATH, state)


def update_status(
    state: dict,
    *,
    size: int | None,
    seed: int | None,
    step: int,
    last_validation_step: int,
    best_step: int,
    best_loss: float,
    error_status: str,
) -> None:
    lines = [
        f"current_training_size: {'' if size is None else size}",
        f"current_seed: {'' if seed is None else seed}",
        f"current_step: {step}",
        f"last_validation_step: {last_validation_step}",
        f"best_step: {best_step}",
        f"best_validation_loss: {best_loss if math.isfinite(best_loss) else ''}",
        f"completed_jobs: {len(state['completed'])}",
        f"total_jobs: {len(TRAINING_SIZES) * len(SEEDS)}",
        f"last_update_time: {pd.Timestamp.now().isoformat()}",
        f"error_status: {error_status}",
    ]
    temp = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(STATUS_PATH)


def cpu_state_dict(model: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def optimizer_to_cuda(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.cuda()


def save_resume_checkpoint(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def validation_loss(module, model, loader, reference, cfg, scales) -> float:
    return diagnostic.validation_loss(module, model, loader, reference, cfg, scales)


def train_one(module, store, train_ids: list[str], val_ids: list[str], size: int, seed: int) -> dict:
    key = f"n{size}_seed{seed}"
    history_path = WORK / f"{key}_history.csv"
    best_path = CHECKPOINTS / f"step_normalized_{key}_best.pt"
    last_path = CHECKPOINTS / f"step_normalized_{key}_last.pt"

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
    best_step = 0
    stale = 0
    global_step = 0
    amp_skipped_batches = 0
    histories: list[dict] = []
    elapsed_before = 0.0
    batches_in_current_pass = 0
    pass_start_generator_state = generator.get_state()
    resume_used = False

    if last_path.exists():
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume.get("completed"):
            raise RuntimeError(f"Completed last-state exists without completed registry for {key}")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        optimizer_to_cuda(optimizer)
        scheduler.load_state_dict(resume["scheduler_state"])
        amp.load_state_dict(resume["amp_state"])
        global_step = int(resume["global_step"])
        best_loss = float(resume["best_loss"])
        best_step = int(resume["best_step"])
        stale = int(resume["stale"])
        amp_skipped_batches = int(resume["amp_skipped_batches"])
        elapsed_before = float(resume["elapsed_s"])
        histories = list(resume["histories"])
        pass_start_generator_state = resume["pass_start_generator_state"]
        batches_in_current_pass = int(resume["batches_in_current_pass"])
        generator.set_state(pass_start_generator_state)
        iterator = iter(train_loader)
        for _ in range(batches_in_current_pass):
            next(iterator)
        random.setstate(resume["python_rng_state"])
        np.random.set_state(resume["numpy_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
        resume_used = True
        emit("job_resume", key=key, optimizer_step=global_step, best_step=best_step)
    else:
        iterator = iter(train_loader)

    state = load_state()
    state["current"] = {"key": key, "training_size": size, "seed": seed, "step": global_step}
    save_state(state)
    update_status(
        state,
        size=size,
        seed=seed,
        step=global_step,
        last_validation_step=global_step - global_step % VALIDATION_INTERVAL_STEPS,
        best_step=best_step,
        best_loss=best_loss,
        error_status="",
    )
    emit("job_start", key=key, resumed=resume_used, optimizer_step=global_step)

    interval_loss = 0.0
    interval_samples = 0
    started = time.time()
    while global_step < MAX_OPTIMIZER_STEPS and stale < EARLY_STOP_PATIENCE_CHECKS:
        try:
            batch = next(iterator)
            batches_in_current_pass += 1
        except StopIteration:
            pass_start_generator_state = generator.get_state()
            iterator = iter(train_loader)
            batches_in_current_pass = 0
            batch = next(iterator)
            batches_in_current_pass = 1

        model.train()
        micro, tau, pf0, delta = (
            batch[name].cuda(non_blocking=True) for name in ("micro", "tau", "pf0", "delta")
        )
        mass, current, mask = (
            batch[name].cuda(non_blocking=True) for name in ("mass", "current", "current_mask")
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred, pm, pc = model(micro, tau)
            loss = module.field_training_loss(pred, delta, pf0, reference, cfg)
            loss += 0.25 * module.scalar_training_loss(
                pm, pc, mass, current, mask, train_scales, "raw_standardized"
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss for {key} at optimizer step {global_step + 1}")
        amp.scale(loss).backward()
        amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scale_before = amp.get_scale()
        amp.step(optimizer)
        amp.update()
        if amp.get_scale() < scale_before:
            amp_skipped_batches += 1
            continue

        scheduler.step()
        global_step += 1
        interval_loss += float(loss.detach()) * len(micro)
        interval_samples += len(micro)
        if global_step % VALIDATION_INTERVAL_STEPS:
            continue

        score = validation_loss(module, model, val_loader, reference, cfg, train_scales)
        improved = score < best_loss - MIN_IMPROVEMENT
        if improved:
            best_loss = score
            best_step = global_step
            stale = 0
            torch.save(
                {
                    "state_dict": cpu_state_dict(model),
                    "config": CONFIG,
                    "training_size": size,
                    "seed": seed,
                    "best_optimizer_step": best_step,
                    "best_validation_total_loss": best_loss,
                    "scales": train_scales,
                    "dynamic_reference": reference,
                    "protocol": {
                        "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
                        "validation_interval_steps": VALIDATION_INTERVAL_STEPS,
                        "early_stop_patience_steps": EARLY_STOP_PATIENCE_STEPS,
                        "minimum_improvement": MIN_IMPROVEMENT,
                        "scheduler": "CosineAnnealingLR per successful optimizer update",
                        "scheduler_t_max": MAX_OPTIMIZER_STEPS,
                        "eta_min": 0.0,
                    },
                },
                best_path,
            )
            emit("best_update", key=key, optimizer_step=global_step, validation_loss=score)
        else:
            stale += 1

        row = {
            "training_size": size,
            "seed": seed,
            "optimizer_step": global_step,
            "validation_check": global_step // VALIDATION_INTERVAL_STEPS,
            "data_passes": global_step / steps_per_data_pass,
            "interval_train_loss": interval_loss / max(interval_samples, 1),
            "validation_total_loss": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "improved": improved,
            "best_optimizer_step": best_step,
            "best_validation_total_loss": best_loss,
            "stale_validation_checks": stale,
            "amp_skipped_batches": amp_skipped_batches,
        }
        histories.append(row)
        write_csv(history_path, histories)
        elapsed = elapsed_before + time.time() - started
        save_resume_checkpoint(
            last_path,
            {
                "completed": False,
                "model_state": cpu_state_dict(model),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "amp_state": amp.state_dict(),
                "global_step": global_step,
                "best_loss": best_loss,
                "best_step": best_step,
                "stale": stale,
                "amp_skipped_batches": amp_skipped_batches,
                "elapsed_s": elapsed,
                "histories": histories,
                "pass_start_generator_state": pass_start_generator_state,
                "batches_in_current_pass": batches_in_current_pass,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
            },
        )
        state = load_state()
        state["current"] = {"key": key, "training_size": size, "seed": seed, "step": global_step}
        save_state(state)
        update_status(
            state,
            size=size,
            seed=seed,
            step=global_step,
            last_validation_step=global_step,
            best_step=best_step,
            best_loss=best_loss,
            error_status="",
        )
        emit(
            "validation",
            key=key,
            optimizer_step=global_step,
            validation_loss=score,
            best_step=best_step,
            stale=stale,
        )
        interval_loss = 0.0
        interval_samples = 0

    if not best_path.exists():
        raise RuntimeError(f"No valid best checkpoint for {key}")
    training_wall_s = elapsed_before + time.time() - started
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["state_dict"])
    model.cuda().eval()
    metrics = module.validation_metrics(model, store, val_ids, train_indices, CONFIG["batch_size"])
    result = {
        "status": "OK",
        "source": "new_step_normalized_full_curve",
        "training_size": size,
        "seed": seed,
        **metrics,
        "best_optimizer_step": best_step,
        "stop_optimizer_step": global_step,
        "best_validation_total_loss": best_loss,
        "data_passes_at_best": best_step / steps_per_data_pass,
        "data_passes_at_stop": global_step / steps_per_data_pass,
        "training_wall_s": training_wall_s,
        "training_time_h": training_wall_s / 3600.0,
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "steps_per_data_pass": steps_per_data_pass,
        "validation_checks": len(histories),
        "early_stopped": stale >= EARLY_STOP_PATIENCE_CHECKS,
        "amp_skipped_batches": amp_skipped_batches,
        "resume_used": resume_used,
        "checkpoint": str(best_path),
    }
    last_path.unlink(missing_ok=True)
    emit("job_complete", key=key, best_step=best_step, stop_step=global_step)
    return result


def register_reused(state: dict) -> None:
    source = json.loads(diagnostic.STATE_PATH.read_text(encoding="utf-8"))["completed"]
    for size, seed in sorted(REUSED):
        key = f"n{size}_seed{seed}"
        result = dict(source[key])
        checkpoint = Path(result["checkpoint"])
        history = diagnostic.WORK / f"{key}_history.csv"
        if result.get("status") != "OK" or not checkpoint.exists() or not history.exists():
            raise RuntimeError(f"Reusable task is incomplete: {key}")
        result["source"] = "reused_step_normalized_diagnostic"
        result["training_time_h"] = float(result["training_wall_s"]) / 3600.0
        state["completed"].setdefault(key, result)


def metric_rows(state: dict) -> list[dict]:
    rows = []
    for size in TRAINING_SIZES:
        for seed in SEEDS:
            key = f"n{size}_seed{seed}"
            if key in state["completed"]:
                rows.append(state["completed"][key])
    return rows


def learning_curve(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    output = []
    previous = None
    for size in TRAINING_SIZES:
        group = frame[frame.training_size == size]
        if len(group) != len(SEEDS):
            continue
        row = {"training_size": size, "n_seeds": len(group)}
        for metric in METRICS:
            mean = float(group[metric].mean())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
            row[f"{metric}_improvement_from_previous_percent"] = (
                "" if previous is None else 100.0 * (previous[metric] - mean) / previous[metric]
            )
        output.append(row)
        previous = {metric: row[f"{metric}_mean"] for metric in METRICS}
    return output


def plot_curve(curve: list[dict]) -> None:
    frame = pd.DataFrame(curve)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=True)
    labels = ("Delta-phase relative L2", "Interface MAE (nm)", "Maximum-depth MAE (nm)")
    for axis, metric, label in zip(axes, METRICS, labels):
        axis.errorbar(
            frame.training_size,
            frame[f"{metric}_mean"],
            yerr=frame[f"{metric}_sd"],
            marker="o",
            capsize=4,
            color="#276FBF",
            linewidth=1.8,
        )
        axis.set_xlabel("Training trajectories")
        axis.set_ylabel(label)
        axis.set_xticks(TRAINING_SIZES)
        axis.grid(alpha=0.2)
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def trend(values: list[float]) -> str:
    decreases = sum(right < left for left, right in zip(values, values[1:]))
    increases = sum(right > left for left, right in zip(values, values[1:]))
    if decreases == len(values) - 1:
        return "decreased monotonically"
    if increases == len(values) - 1:
        return "increased monotonically"
    return "showed no monotonic trend"


def write_report(rows: list[dict], curve: list[dict], state: dict) -> None:
    frame = pd.DataFrame(rows)
    curve_frame = pd.DataFrame(curve).set_index("training_size")
    lines = [
        "# Unified optimizer-step full learning curve",
        "",
        "## Protocol integrity",
        "",
        "- All five sizes use the same 45,360-update maximum budget, 1,296-update validation interval, and six-check early stopping patience.",
        "- Fixed validation trajectories: 36.",
        "- Independent test truth accessed: NO.",
        "- Model, loss, preprocessing, nested subsets and hyperparameters were unchanged.",
        f"- Completed jobs: {len(rows)}/25; failed jobs: {len(state['failed'])}; reused jobs: {len(REUSED)}.",
        "",
        "## Five-seed learning curve",
        "",
        "| n | delta L2 | interface MAE (nm) | maximum-depth MAE (nm) | best steps | early stops |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for size in TRAINING_SIZES:
        summary = curve_frame.loc[size]
        group = frame[frame.training_size == size].sort_values("seed")
        lines.append(
            f"| {size} | {summary.delta_phase_relative_l2_mean:.6g} ± {summary.delta_phase_relative_l2_sd:.3g} | "
            f"{summary.interface_mae_nm_mean:.6g} ± {summary.interface_mae_nm_sd:.3g} | "
            f"{summary.max_depth_mae_nm_mean:.6g} ± {summary.max_depth_mae_nm_sd:.3g} | "
            f"{', '.join(str(int(value)) for value in group.best_optimizer_step)} | "
            f"{int(group.early_stopped.astype(str).str.lower().eq('true').sum())} |"
        )
    lines.extend(["", "## Adjacent-size improvements", ""])
    for size in TRAINING_SIZES[1:]:
        row = curve_frame.loc[size]
        previous = TRAINING_SIZES[TRAINING_SIZES.index(size) - 1]
        lines.append(
            f"- {previous}→{size}: delta {float(row.delta_phase_relative_l2_improvement_from_previous_percent):+.2f}%, "
            f"interface {float(row.interface_mae_nm_improvement_from_previous_percent):+.2f}%, "
            f"maximum depth {float(row.max_depth_mae_nm_improvement_from_previous_percent):+.2f}%."
        )
    lines.extend(["", "## Required conclusions", ""])
    for metric, label in zip(METRICS, ("delta L2", "interface MAE", "maximum-depth MAE")):
        lines.append(
            f"- Five-seed dispersion for {label} {trend(curve_frame[f'{metric}_sd'].tolist())}."
        )
    for previous, current in ((216, 252), (252, 288)):
        values = [
            float(curve_frame.loc[current, f"{metric}_improvement_from_previous_percent"])
            for metric in METRICS
        ]
        if all(value > 2.0 for value in values):
            classification = "improvement"
        elif all(value < -2.0 for value in values):
            classification = "degradation"
        else:
            classification = "mixed or approximately flat"
        lines.append(
            f"- {previous}→{current}: {classification}; metric improvements were "
            + ", ".join(f"{value:+.2f}%" for value in values)
            + "."
        )
    final_improvements = [
        float(curve_frame.loc[288, f"{metric}_improvement_from_previous_percent"])
        for metric in METRICS
    ]
    benefit = sum(value > 5.0 for value in final_improvements) >= 2
    lines.append(
        "- Additional-data signal at 288: "
        + ("present in at least two core metrics (>5%)." if benefit else "not clearly present in at least two core metrics (>5%).")
    )
    lines.append(
        "- Recommendation based only on the fixed validation learning curve: "
        + ("further balanced training-data expansion may be considered, but is not started automatically." if benefit else "do not expand the training data further on the present evidence.")
    )
    lines.append(
        "- Training anomalies or protocol inconsistencies: "
        + ("none recorded." if not state["failed"] else f"{len(state['failed'])} failed task(s) recorded; see state.json.")
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(state: dict) -> None:
    rows = metric_rows(state)
    if len(rows) != 25:
        return
    if any(row.get("status") != "OK" for row in rows):
        raise RuntimeError("At least one registered result is not OK")
    write_csv(METRICS_PATH, rows)
    curve = learning_curve(rows)
    if len(curve) != len(TRAINING_SIZES):
        raise RuntimeError("Incomplete five-seed learning curve")
    write_csv(CURVE_PATH, curve)
    plot_curve(curve)
    write_report(rows, curve, state)
    if "BLOCKED" in ACCESS_LOG.read_text(encoding="utf-8"):
        raise RuntimeError("Forbidden independent-test truth access was attempted")
    update_status(
        state,
        size=None,
        seed=None,
        step=MAX_OPTIMIZER_STEPS,
        last_validation_step=MAX_OPTIMIZER_STEPS,
        best_step=0,
        best_loss=float("inf"),
        error_status="COMPLETE",
    )
    emit("all_complete", report=str(REPORT_PATH))


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ACCESS_LOG.write_text(
        "# Independent-test truth access guard\n\nNo independent test truth was accessed.\n",
        encoding="utf-8",
    )
    extension, _ = scaling.audit_extension()
    module = scaling.load_training_module()
    module.OUT = WORK
    module.CKPT = CHECKPOINTS
    module.STATE_PATH = WORK / "unused_module_state.json"
    module.ACCESS_LOG = ACCESS_LOG
    module.STATUS = STATUS_PATH
    store = scaling.combined_store_class(module, extension)()
    val_ids = store.ids("validation")
    if len(val_ids) != 36:
        raise RuntimeError(f"Expected 36 validation trajectories, found {len(val_ids)}")

    nested = pd.read_csv(EXT / "LEARNING_CURVE_NESTED_SUBSETS.csv")
    subsets = {
        size: nested.loc[nested.n_train == size, "production_id"].tolist()
        for size in TRAINING_SIZES
    }
    for size in TRAINING_SIZES:
        if len(subsets[size]) != size:
            raise RuntimeError(f"Nested subset size mismatch for {size}")
    for left, right in zip(TRAINING_SIZES, TRAINING_SIZES[1:]):
        if not set(subsets[left]).issubset(set(subsets[right])):
            raise RuntimeError(f"Nested subset violation: {left} is not contained in {right}")

    state = load_state()
    register_reused(state)
    save_state(state)
    emit("scheduler_start", reusable_jobs=len(REUSED), pending_jobs=25 - len(state["completed"]))

    for size in TRAINING_SIZES:
        for seed in SEEDS:
            key = f"n{size}_seed{seed}"
            if key in state["completed"]:
                continue
            attempts = int(state["retries"].get(key, 0))
            while attempts <= 1:
                try:
                    result = train_one(module, store, subsets[size], val_ids, size, seed)
                    state = load_state()
                    state["completed"][key] = result
                    state["current"] = None
                    state["failed"].pop(key, None)
                    save_state(state)
                    write_csv(METRICS_PATH, metric_rows(state))
                    torch.cuda.empty_cache()
                    break
                except Exception as exc:
                    attempts += 1
                    state = load_state()
                    state["retries"][key] = attempts
                    state["failed"][key] = {
                        "attempts": attempts,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "time": pd.Timestamp.now().isoformat(),
                    }
                    save_state(state)
                    update_status(
                        state,
                        size=size,
                        seed=seed,
                        step=int(state.get("current", {}).get("step", 0)) if state.get("current") else 0,
                        last_validation_step=0,
                        best_step=0,
                        best_loss=float("inf"),
                        error_status=repr(exc),
                    )
                    emit("job_error", key=key, attempt=attempts, error=repr(exc))
                    torch.cuda.empty_cache()
                    if attempts > 1:
                        break
                    emit("job_retry", key=key, attempt=attempts + 1)
            state = load_state()
            completed_size = sum(f"n{size}_seed{seed_value}" in state["completed"] for seed_value in SEEDS)
            if completed_size == len(SEEDS):
                emit("training_size_complete", training_size=size)

    state = load_state()
    finalize(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        emit("fatal_error", error=repr(exc))
        raise
