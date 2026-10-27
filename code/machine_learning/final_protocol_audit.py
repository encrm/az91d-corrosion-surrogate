from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AZ91D_DATA_ROOT", REPO_ROOT / "data"))
PROJECT = Path(os.environ.get("AZ91D_PROJECT_ROOT", DATA_ROOT / "project"))
EXT = Path(os.environ.get("AZ91D_EXTENSION_ROOT", DATA_ROOT / "extension"))
FROZEN = Path(os.environ.get("AZ91D_FREEZE_ROOT", PROJECT / "dataset216_freeze_audit_20260912"))
SCRIPT_DIR = Path(os.environ.get("AZ91D_SCRIPT_DIR", Path(__file__).resolve().parent))

METRICS_PATH = EXT / "STEP_NORMALIZED_FULL_SEED_METRICS.csv"
STATE_PATH = EXT / "step_normalized_full_learning_curve" / "state.json"
NESTED_PATH = EXT / "LEARNING_CURVE_NESTED_SUBSETS.csv"
BASE_MANIFEST_PATH = FROZEN / "COMBINED216_PRODUCTION_MANIFEST.csv"
SPLIT_PATH = FROZEN / "COMBINED216_SPLIT_MANIFEST.csv"
EXT_MANIFEST_PATH = EXT / "TRAIN_EXTENSION144_MANIFEST.csv"
TIME_PATH = PROJECT / "az91d_formal_surrogate_training" / "manifests" / "TIME_SPLIT_MANIFEST.csv"

REPORT_PATH = EXT / "FINAL_MODEL_PROTOCOL_LOCK.md"
AUDIT_PATH = EXT / "FINAL252_CHECKPOINT_AUDIT.csv"

EXPECTED_PARAMETERS = 7_984_355
EXPECTED_BEST_STEPS = {0: 42_768, 1: 45_360, 2: 45_360, 3: 44_064, 4: 38_880}
EXPECTED_CONFIG = {
    "base_channels": 32,
    "batch_size": 2,
    "learning_rate": 2e-4,
    "interface_lambda": 2.0,
    "dynamic_weight_cap": 4.0,
    "gradient_weight": 0.03,
    "weight_decay": 1e-4,
}
EXPECTED_PROTOCOL = {
    "max_optimizer_steps": 45_360,
    "validation_interval_steps": 1_296,
    "early_stop_patience_steps": 7_776,
    "minimum_improvement": 1e-5,
    "scheduler_t_max": 45_360,
    "eta_min": 0.0,
}
TRAIN_HOURS = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24]
WITHHELD_HOURS = [3, 5, 7, 9, 14, 18, 22]
PROBE_HOURS = [11, 13, 15, 17, 19, 21, 23]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_micro_hash(path: Path) -> str:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        array = np.loadtxt(path)
    array = np.ascontiguousarray(array.astype(np.float32, copy=False))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def checkpoint_paths() -> dict[int, Path]:
    diagnostic = EXT / "step_normalized_diagnostic" / "checkpoints"
    full = EXT / "step_normalized_full_learning_curve" / "checkpoints"
    return {
        0: diagnostic / "step_normalized_n252_seed0_best.pt",
        1: diagnostic / "step_normalized_n252_seed1_best.pt",
        2: diagnostic / "step_normalized_n252_seed2_best.pt",
        3: full / "step_normalized_n252_seed3_best.pt",
        4: full / "step_normalized_n252_seed4_best.pt",
    }


def history_path(seed: int) -> Path:
    folder = "step_normalized_diagnostic" if seed < 3 else "step_normalized_full_learning_curve"
    return EXT / folder / f"n252_seed{seed}_history.csv"


def load_model_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    import run_scaling_after_extension as scaling

    return scaling.load_training_module()


def close(a: float, b: float, atol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=atol)


def audit_checkpoints(metrics: pd.DataFrame) -> tuple[list[dict], list[str]]:
    module = load_model_module()
    rows: list[dict] = []
    issues: list[str] = []
    paths = checkpoint_paths()

    for seed in range(5):
        metric = metrics.loc[metrics.seed == seed].iloc[0]
        path = paths[seed]
        result = {
            "seed": seed,
            "checkpoint_path": str(path),
            "sha256": "",
            "parameter_count": "",
            "best_step": int(metric.best_optimizer_step),
            "stop_step": int(metric.stop_optimizer_step),
            "early_stop": str(metric.early_stopped).lower() == "true",
            "delta_phase_relative_L2": float(metric.delta_phase_relative_l2),
            "interface_MAE_nm": float(metric.interface_mae_nm),
            "maximum_depth_MAE_nm": float(metric.max_depth_mae_nm),
            "best_validation_L_total": float(metric.best_validation_total_loss),
            "load_status": "FAIL",
            "architecture_match": False,
            "finite_weights": False,
            "history_traceable": False,
            "best_checkpoint_verified": False,
            "resume_status": "",
            "audit_status": "FAIL",
        }
        seed_issues: list[str] = []
        if not path.exists():
            seed_issues.append("checkpoint missing")
            rows.append(result)
            issues.append(f"seed {seed}: checkpoint missing")
            continue

        result["sha256"] = sha256_file(path)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            scales = checkpoint["scales"]
            model = module.TimeUNet(scales["mass_scale"], scales["current_scale"], 32, True)
            incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
            result["load_status"] = "PASS"
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            result["parameter_count"] = parameter_count
            result["finite_weights"] = all(
                bool(torch.isfinite(tensor).all())
                for tensor in checkpoint["state_dict"].values()
                if torch.is_tensor(tensor)
            )
            config_match = checkpoint.get("config") == EXPECTED_CONFIG
            protocol = checkpoint.get("protocol", {})
            protocol_match = all(close(protocol.get(key, float("nan")), value) for key, value in EXPECTED_PROTOCOL.items())
            scheduler_match = str(protocol.get("scheduler", "")).startswith("CosineAnnealingLR")
            identity_match = checkpoint.get("training_size") == 252 and checkpoint.get("seed") == seed
            keys_match = not incompatible.missing_keys and not incompatible.unexpected_keys
            result["architecture_match"] = bool(config_match and protocol_match and scheduler_match and identity_match and keys_match)
            if parameter_count != EXPECTED_PARAMETERS:
                seed_issues.append(f"parameter count {parameter_count}")
            if not result["finite_weights"]:
                seed_issues.append("non-finite weight")
            if not result["architecture_match"]:
                seed_issues.append("architecture/config/protocol mismatch")
            if checkpoint.get("best_optimizer_step") != EXPECTED_BEST_STEPS[seed]:
                seed_issues.append("best step mismatch")
            if checkpoint.get("best_optimizer_step") != int(metric.best_optimizer_step):
                seed_issues.append("checkpoint/metrics best-step mismatch")
            if not close(checkpoint.get("best_validation_total_loss"), metric.best_validation_total_loss):
                seed_issues.append("checkpoint/metrics validation-loss mismatch")

            history = pd.read_csv(history_path(seed))
            best_history = history.loc[history.optimizer_step == int(metric.best_optimizer_step)]
            history_ok = (
                len(best_history) == 1
                and close(best_history.iloc[0].validation_total_loss, metric.best_validation_total_loss)
                and bool(best_history.iloc[0].improved)
            )
            result["history_traceable"] = bool(history_ok)
            result["best_checkpoint_verified"] = bool(
                history_ok
                and path.name.endswith("_best.pt")
                and checkpoint.get("best_optimizer_step") == int(metric.best_optimizer_step)
            )
            if not history_ok:
                seed_issues.append("best metric not traceable in history")
            if not result["best_checkpoint_verified"]:
                seed_issues.append("best checkpoint identity not verified")

            if seed < 3:
                result["resume_status"] = "direct diagnostic run; no resume path used"
            else:
                result["resume_status"] = "resume_used=false" if str(metric.get("resume_used", "false")).lower() == "false" else "resume_used=true"
                if result["resume_status"] != "resume_used=false":
                    seed_issues.append("unexpected resume use")
        except Exception as exc:
            seed_issues.append(f"load exception: {exc!r}")

        if seed_issues:
            issues.extend(f"seed {seed}: {issue}" for issue in seed_issues)
        else:
            result["audit_status"] = "PASS"
        rows.append(result)
    return rows, issues


def audit_metrics(metrics: pd.DataFrame) -> tuple[dict, list[str]]:
    expected = {
        "delta_phase_relative_l2": (0.0461006, 7, 0.0102, 4),
        "interface_mae_nm": (4.48013, 5, 1.45, 2),
        "max_depth_mae_nm": (6.86713, 5, 2.66, 2),
    }
    summary = {}
    issues = []
    for column, (target_mean, mean_digits, target_sd, sd_digits) in expected.items():
        values = metrics[column].astype(float).to_numpy()
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        summary[column] = {"mean": mean, "sample_sd": sd}
        if round(mean, mean_digits) != target_mean:
            issues.append(f"{column} mean differs: {mean} vs {target_mean}")
        if round(sd, sd_digits) != target_sd:
            issues.append(f"{column} SD differs: {sd} vs {target_sd}")
    return summary, issues


def audit_data_roles() -> tuple[dict, list[str]]:
    nested = pd.read_csv(NESTED_PATH)
    train_ids = nested.loc[nested.n_train == 252, "production_id"].astype(str).tolist()
    base = pd.read_csv(BASE_MANIFEST_PATH)
    split = pd.read_csv(SPLIT_PATH)
    extension = pd.read_csv(EXT_MANIFEST_PATH)
    extension = extension.copy()
    extension["dataset_generation"] = "extension144"
    extension["trajectory_path"] = extension.production_id.map(lambda value: str(EXT / "comsol_runs" / value))
    extension["microstructure_file"] = extension.smooth_npy_path
    base = base.copy()
    base["microstructure_file"] = base.microstructure_path
    metadata = pd.concat(
        [
            base[["production_id", "microstructure_sha256", "microstructure_file", "trajectory_path", "split_role", "f_beta", "morphology"]],
            extension[["production_id", "microstructure_sha256", "microstructure_file", "trajectory_path", "split_role", "f_beta", "morphology"]],
        ],
        ignore_index=True,
    )
    metadata.production_id = metadata.production_id.astype(str)
    by_id = metadata.set_index("production_id", drop=False)

    validation = split.loc[split.split_role == "validation", "production_id"].astype(str).tolist()
    historical = split.loc[split.split_role == "historical_audit_test", "production_id"].astype(str).tolist()
    pristine = split.loc[split.split_role == "pristine_final_test", "production_id"].astype(str).tolist()
    role_sets = {
        "train252": set(train_ids),
        "validation36": set(validation),
        "independent18_a": set(historical),
        "independent18_b": set(pristine),
    }
    issues: list[str] = []
    expected_sizes = {"train252": 252, "validation36": 36, "independent18_a": 18, "independent18_b": 18}
    for role, expected in expected_sizes.items():
        if len(role_sets[role]) != expected:
            issues.append(f"{role} has {len(role_sets[role])} unique IDs, expected {expected}")
    roles = list(role_sets)
    id_overlaps = {}
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlap = sorted(role_sets[left] & role_sets[right])
            id_overlaps[f"{left}__{right}"] = overlap
            if overlap:
                issues.append(f"ID overlap {left}/{right}: {len(overlap)}")

    selected_ids = set().union(*role_sets.values())
    missing_metadata = sorted(selected_ids - set(by_id.index))
    if missing_metadata:
        issues.append(f"missing metadata for {len(missing_metadata)} IDs")

    selected = by_id.loc[sorted(selected_ids)].copy()
    selected["normalized_path"] = selected.microstructure_file.map(lambda value: str(Path(str(value)).resolve()).lower())
    selected["normalized_trajectory_path"] = selected.trajectory_path.map(lambda value: str(Path(str(value)).resolve()).lower())
    path_duplicates = selected[selected.duplicated("normalized_path", keep=False)]
    trajectory_path_duplicates = selected[selected.duplicated("normalized_trajectory_path", keep=False)]
    recorded_hash_duplicates = selected[selected.duplicated("microstructure_sha256", keep=False)]
    if len(path_duplicates):
        issues.append(f"duplicate microstructure paths: {len(path_duplicates)} rows")
    if len(trajectory_path_duplicates):
        issues.append(f"duplicate trajectory paths: {len(trajectory_path_duplicates)} rows")
    if len(recorded_hash_duplicates):
        issues.append(f"duplicate recorded microstructure hashes: {len(recorded_hash_duplicates)} rows")

    missing_files = []
    canonical_hashes = {}
    for production_id, row in selected.iterrows():
        path = Path(str(row.microstructure_file))
        if not path.exists():
            missing_files.append(production_id)
            continue
        canonical_hashes[production_id] = canonical_micro_hash(path)
    if missing_files:
        issues.append(f"missing microstructure files: {len(missing_files)}")
    canonical_counts = Counter(canonical_hashes.values())
    canonical_duplicates = {key: count for key, count in canonical_counts.items() if count > 1}
    if canonical_duplicates:
        issues.append(f"canonical smooth-microstructure duplicates: {len(canonical_duplicates)} groups")

    train_meta = by_id.loc[train_ids]
    invalid_train_roles = train_meta.loc[~train_meta.split_role.isin(["train", "train_extension"]), "production_id"].tolist()
    if invalid_train_roles:
        issues.append(f"train252 contains invalid source roles: {len(invalid_train_roles)}")
    cell_counts = train_meta.groupby(["f_beta", "morphology"]).size().to_dict()
    if set(cell_counts.values()) != {28} or len(cell_counts) != 9:
        issues.append(f"train252 is not balanced across 9 cells: {cell_counts}")

    pool288 = set(nested.loc[nested.n_train == 288, "production_id"].astype(str))
    excluded36 = pool288 - role_sets["train252"]
    if len(excluded36) != 36:
        issues.append(f"288-minus-252 pool has {len(excluded36)} IDs, expected 36")
    if excluded36 & (role_sets["validation36"] | role_sets["independent18_a"] | role_sets["independent18_b"]):
        issues.append("excluded 36 training candidates overlap evaluation roles")

    result = {
        "role_sizes": {key: len(value) for key, value in role_sets.items()},
        "id_overlap_counts": {key: len(value) for key, value in id_overlaps.items()},
        "recorded_hash_duplicate_rows": int(len(recorded_hash_duplicates)),
        "canonical_smooth_duplicate_groups": len(canonical_duplicates),
        "microstructure_path_duplicate_rows": int(len(path_duplicates)),
        "trajectory_path_duplicate_rows": int(len(trajectory_path_duplicates)),
        "missing_metadata": len(missing_metadata),
        "missing_microstructure_files": len(missing_files),
        "train252_cell_counts": {f"{fb}|{morph}": int(count) for (fb, morph), count in cell_counts.items()},
        "excluded_training_candidates": len(excluded36),
        "truth_labels_read": False,
    }
    return result, issues


def audit_time_protocol() -> tuple[dict, list[str]]:
    time_manifest = pd.read_csv(TIME_PATH)
    groups = {
        group: time_manifest.loc[time_manifest.group == group, "time_h"].astype(float).tolist()
        for group in time_manifest.group.unique()
    }
    issues = []
    if groups.get("TRAINING_GRID") != TRAIN_HOURS:
        issues.append(f"training times differ: {groups.get('TRAINING_GRID')}")
    if groups.get("FORMAL_WITHHELD") != WITHHELD_HOURS:
        issues.append(f"withheld times differ: {groups.get('FORMAL_WITHHELD')}")
    if groups.get("AUXILIARY_PRISTINE_CANDIDATES") != PROBE_HOURS:
        issues.append(f"probe times differ: {groups.get('AUXILIARY_PRISTINE_CANDIDATES')}")
    if groups.get("INITIAL_CONDITION_ONLY") != [0.0]:
        issues.append("t=0 role differs from INITIAL_CONDITION_ONLY")

    source = (CODE / "new108_production" / "run_dataset216_unet_search.py").read_text(encoding="utf-8")
    source_train_literal = "TRAIN_HOURS = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24]"
    if source_train_literal not in source:
        issues.append("training source does not contain the expected TRAIN_HOURS literal")
    return {
        "training_hours": TRAIN_HOURS,
        "withheld_hours": WITHHELD_HOURS,
        "probe_hours": PROBE_HOURS,
        "t0_handling": "analytic initial-condition constraint only",
        "selection_uses_training_grid_only": source_train_literal in source,
    }, issues


def write_audit_csv(rows: list[dict]) -> None:
    fields = list(rows[0])
    with AUDIT_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_metrics = pd.read_csv(METRICS_PATH)
    metrics = all_metrics.loc[all_metrics.training_size == 252].copy().sort_values("seed")
    if list(metrics.seed.astype(int)) != [0, 1, 2, 3, 4]:
        raise RuntimeError("The five 252-seed metric rows are incomplete")

    metric_summary, metric_issues = audit_metrics(metrics)
    checkpoint_rows, checkpoint_issues = audit_checkpoints(metrics)
    data_result, data_issues = audit_data_roles()
    time_result, time_issues = audit_time_protocol()
    all_issues = metric_issues + checkpoint_issues + data_issues + time_issues
    checkpoint_pass = all(row["audit_status"] == "PASS" for row in checkpoint_rows)
    gate_ready = checkpoint_pass and not all_issues

    write_audit_csv(checkpoint_rows)

    lines = [
        "# Final model protocol lock",
        "",
        "## Decision",
        "",
        "- Final training size: **252 trajectories**",
        "- Fixed validation size: **36 trajectories**",
        "- Independent evaluation roles: **18 + 18 trajectories**",
        "- The 36 trajectories in the 288-candidate pool but outside the 252 nested subset remain training candidates only; they are not reassigned or deleted.",
        "- No additional COMSOL data generation or training-size search is authorized by this protocol.",
        "",
        "## Model and optimization protocol",
        "",
        "- Architecture: continuous-time analytically constrained U-Net with two auxiliary scalar heads",
        "- Parameters: 7,984,355 trainable parameters",
        "- base_channels: 32",
        "- batch_size: 2",
        "- learning_rate: 2e-4",
        "- interface_lambda: 2.0",
        "- dynamic_weight_cap: 4.0",
        "- gradient_weight: 0.03",
        "- weight_decay: 1e-4",
        "- Optimizer: AdamW",
        "- AMP: enabled",
        "- Gradient clipping: 2.0",
        "- MAX_OPTIMIZER_STEPS: 45,360",
        "- VALIDATION_INTERVAL_STEPS: 1,296",
        "- EARLY_STOP_PATIENCE_STEPS: 7,776",
        "- Minimum improvement: 1e-5",
        "- Scheduler: CosineAnnealingLR, stepped after each successful optimizer update",
        "- T_max: 45,360; eta_min: 0; warm-up: none",
        "",
        "## Loss configuration",
        "",
        "- Field loss: global L1/0.01 + 0.05 × capped relative L2 + 0.35 × dynamically weighted L1/0.01 + 0.03 × spatial-gradient L1/0.01.",
        "- Relative-L2 cap: 20; dynamic-weight cap: 4.0; interface multiplier: 2.0.",
        "- Scalar auxiliary loss: standardized smooth-L1 mass loss + standardized smooth-L1 galvanic-current loss.",
        "- Total loss: field loss + 0.25 × scalar auxiliary loss.",
        "",
        "## Time protocol",
        "",
        f"- Training grid (h): {', '.join(map(str, TRAIN_HOURS))}",
        "- t = 0: handled only by the analytic initial-condition constraint.",
        f"- Withheld interpolation times not used for model selection (h): {', '.join(map(str, WITHHELD_HOURS))}",
        f"- Further independent time probes not used for model selection (h): {', '.join(map(str, PROBE_HOURS))}",
        "",
        "## Five-seed checkpoint audit",
        "",
        "| Seed | Best step | Stop step | Early stop | delta-phase relative L2 | Interface MAE (nm) | Maximum-depth MAE (nm) | Best validation L_total | SHA-256 | Status |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|:---|:---:|",
    ]
    for row in checkpoint_rows:
        lines.append(
            f"| {row['seed']} | {row['best_step']} | {row['stop_step']} | {str(row['early_stop']).upper()} | "
            f"{row['delta_phase_relative_L2']:.10f} | {row['interface_MAE_nm']:.6f} | "
            f"{row['maximum_depth_MAE_nm']:.6f} | {row['best_validation_L_total']:.10f} | "
            f"`{row['sha256']}` | {row['audit_status']} |"
        )
    lines.extend(
        [
            "",
            "## Five-seed validation summary",
            "",
            f"- delta-phase relative L2: {metric_summary['delta_phase_relative_l2']['mean']:.7f} ± {metric_summary['delta_phase_relative_l2']['sample_sd']:.7f}",
            f"- Interface MAE: {metric_summary['interface_mae_nm']['mean']:.5f} ± {metric_summary['interface_mae_nm']['sample_sd']:.5f} nm",
            f"- Maximum-depth MAE: {metric_summary['max_depth_mae_nm']['mean']:.5f} ± {metric_summary['max_depth_mae_nm']['sample_sd']:.5f} nm",
            "- Standard deviation: sample SD (ddof = 1).",
            "",
            "## Data-role independence",
            "",
            f"- Role sizes: `{json.dumps(data_result['role_sizes'], ensure_ascii=False)}`",
            f"- ID overlaps: `{json.dumps(data_result['id_overlap_counts'], ensure_ascii=False)}`",
            f"- Duplicate recorded microstructure hashes: {data_result['recorded_hash_duplicate_rows']}",
            f"- Duplicate canonical smooth microstructures: {data_result['canonical_smooth_duplicate_groups']}",
            f"- Duplicate microstructure paths: {data_result['microstructure_path_duplicate_rows']}",
            f"- Duplicate trajectory paths: {data_result['trajectory_path_duplicate_rows']}",
            f"- Missing metadata/files: {data_result['missing_metadata']}/{data_result['missing_microstructure_files']}",
            f"- Final 252 training subset balance: 28 trajectories in each of the 9 beta-fraction × morphology cells.",
            f"- Excluded training candidates retained: {data_result['excluded_training_candidates']}",
            "- Independent-evaluation target truth read during this audit: **NO**",
            "",
            "## Paper-method conflict record",
            "",
            "> The final formal training protocol must replace the previous 35-epoch description with the unified optimizer-step protocol. The Methods section will be updated before final-test evaluation.",
            "",
            "## Final-test gate",
            "",
            f"**FINAL_TEST_GATE = {'READY' if gate_ready else 'NOT READY'}**",
        ]
    )
    if all_issues:
        lines.extend(["", "## Unresolved issues", ""] + [f"- {issue}" for issue in all_issues])
    else:
        lines.extend(["", "No unresolved checkpoint, metric, data-role, or time-protocol anomaly was found."])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "report": str(REPORT_PATH),
        "audit_csv": str(AUDIT_PATH),
        "checkpoint_pass": checkpoint_pass,
        "metric_summary": metric_summary,
        "data_result": data_result,
        "time_result": time_result,
        "issues": all_issues,
        "final_test_gate": "READY" if gate_ready else "NOT READY",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
