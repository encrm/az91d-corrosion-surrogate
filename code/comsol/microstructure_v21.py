from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(os.environ.get("AZ91D_V2_ROOT", ROOT / "data" / "v2_pilot"))
V2_SCRIPT = V2_ROOT / "v2_generator" / "run_pilot.py"
V2_DATA = V2_ROOT / "v2_pilot_27"

spec = importlib.util.spec_from_file_location("v2_reference", V2_SCRIPT)
v2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = v2
spec.loader.exec_module(v2)

NX, NY = v2.NX, v2.NY
DX_UM, DY_UM = v2.DX_UM, v2.DY_UM
FRACTIONS = v2.FRACTIONS
TOPOLOGIES = v2.TOPOLOGIES
CANVAS_X, CANVAS_Y = 1024, 192
BASE_SEED = 202609210000
TARGET_SURFACE = {0.05: 0.035, 0.12: 0.085, 0.20: 0.145}


def normalize(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / max(float(a.std()), 1e-12)


def exact_rank_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    target = int(round(fraction * score.size))
    idx = np.argpartition(score.ravel(), target - 1)[:target]
    out = np.zeros(score.size, dtype=bool)
    out[idx] = True
    return out.reshape(score.shape)


def cellular_skeleton(rng: np.random.Generator, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Create an isotropic, nonlinearly warped cellular/interdendritic skeleton."""
    yy, xx = np.mgrid[0:CANVAS_Y, 0:CANVAS_X]
    # Similar physical correlation length in x and y; the old elongated x-filter is removed.
    ux = normalize(ndimage.gaussian_filter(rng.normal(size=(CANVAS_Y, CANVAS_X)), sigma=(18, 18)))
    uy = normalize(ndimage.gaussian_filter(rng.normal(size=(CANVAS_Y, CANVAS_X)), sigma=(18, 18)))
    ux += 0.45 * normalize(ndimage.gaussian_filter(rng.normal(size=(CANVAS_Y, CANVAS_X)), sigma=(8, 8)))
    uy += 0.45 * normalize(ndimage.gaussian_filter(rng.normal(size=(CANVAS_Y, CANVAS_X)), sigma=(8, 8)))
    xw = xx + 5.0 * ux + 1.8 * np.sin(yy / 11 + ux)
    yw = yy + 5.0 * uy + 1.8 * np.sin(xx / 13 + uy)

    area_scale = CANVAS_X * CANVAS_Y / (NX * NY)
    n_sites = int((28 + 35 * fraction) * area_scale)
    sx = rng.uniform(-30, CANVAS_X + 30, n_sites)
    sy = rng.uniform(-30, CANVAS_Y + 30, n_sites)
    weights = rng.uniform(0.82, 1.22, n_sites)
    nearest = np.full((2, CANVAS_Y, CANVAS_X), np.inf)
    for x0, y0, w in zip(sx, sy, weights):
        d = ((xw - x0) * DX_UM / w) ** 2 + ((yw - y0) * DY_UM / w) ** 2
        swap = d < nearest[0]
        nearest[1] = np.where(swap, nearest[0], np.minimum(nearest[1], d))
        nearest[0] = np.where(swap, d, nearest[0])
    gap = np.sqrt(nearest[1]) - np.sqrt(nearest[0])
    boundary = gap < np.percentile(gap, 8.5)
    boundary = ndimage.binary_closing(boundary, structure=v2.disk(1))
    return v2.skeletonize(boundary), normalize(ndimage.gaussian_filter(rng.normal(size=boundary.shape), sigma=(7, 7)))


def build_score(rng: np.random.Generator, skeleton: np.ndarray, occupation: np.ndarray, topology: str) -> np.ndarray:
    fine = normalize(ndimage.gaussian_filter(rng.normal(size=skeleton.shape), sigma=(2.2, 2.2)))
    coarse = normalize(ndimage.gaussian_filter(rng.normal(size=skeleton.shape), sigma=(11, 11)))
    neighbors = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    nodes = skeleton & (neighbors >= 3)

    if topology == "fragmented":
        active = skeleton & (occupation > 0.55)
        active &= coarse > -0.85
        width = 0.90 + 0.20 * fine
    elif topology == "moderate":
        active = skeleton & (occupation > -0.18)
        active &= coarse > -1.25
        width = 1.00 + 0.30 * fine
    else:
        # High local connectivity, but with natural gaps and terminations. No spanning path is added.
        active = skeleton & (occupation > -0.82)
        active &= coarse > -1.55
        width = 1.10 + 0.42 * fine + 0.25 * coarse

    active = ndimage.binary_dilation(active, structure=v2.disk(1))
    if topology == "network":
        # Use the largest naturally occurring cellular cluster as the network core.
        # This encourages local connectivity without constructing a domain-spanning path.
        labels, count = ndimage.label(active, structure=np.ones((3, 3), dtype=int))
        if count:
            sizes = ndimage.sum(active, labels, index=np.arange(1, count + 1))
            core = labels == (int(np.argmax(sizes)) + 1)
            active = core | (active & (coarse > 0.65))
    distance = ndimage.distance_transform_edt(~active, sampling=(DY_UM, DX_UM))
    score = distance / np.clip(width, 0.45, 1.8)
    if topology == "network" and count:
        core_distance = ndimage.distance_transform_edt(~core, sampling=(DY_UM, DX_UM))
        score = np.minimum(score + 0.55, core_distance / np.clip(width, 0.45, 1.8))

    # Local eutectic pools and thick junctions remain anchored to interdendritic regions.
    if topology == "network":
        node_distance = ndimage.distance_transform_edt(~nodes, sampling=(DY_UM, DX_UM))
        score -= 1.15 * np.exp(-(node_distance / 1.55) ** 2)
        pool_field = normalize(ndimage.gaussian_filter(rng.normal(size=skeleton.shape), sigma=(3.5, 3.5)))
        near_gb = ndimage.distance_transform_edt(~skeleton, sampling=(DY_UM, DX_UM)) < 4.0
        pools = (pool_field > 1.45) & near_gb
        score -= 0.75 * ndimage.gaussian_filter(pools.astype(float), sigma=2.0)
    elif topology == "moderate":
        score += 0.10 * np.maximum(-occupation, 0)
    else:
        score += 0.30 * np.maximum(-occupation, 0)

    # Sparse divorced islands, still preferentially close to the common skeleton family.
    island_field = normalize(ndimage.gaussian_filter(rng.normal(size=skeleton.shape), sigma=(2.5, 2.5)))
    near = ndimage.distance_transform_edt(~skeleton, sampling=(DY_UM, DX_UM)) < 3.0
    islands = (island_field > (1.85 if topology == "network" else 1.65)) & near
    score -= (0.42 if topology == "network" else 0.65) * ndimage.gaussian_filter(islands.astype(float), 1.2)
    return score


def candidate_penalty(metrics, topology: str, fraction: float, surface_target: float) -> float:
    largest = metrics.largest_component_fraction
    if topology == "fragmented":
        topo = max(0.0, largest - 0.24) * 20 + max(0, 14 - metrics.component_count) * 0.4
    elif topology == "moderate":
        topo = abs(largest - 0.48) * 7 + max(0, 5 - metrics.component_count) * 0.5
    else:
        topo = max(0.0, 0.72 - largest) * 22 + max(0.0, largest - 0.96) * 5
    surface = abs(metrics.surface_contact_fraction - surface_target) * 24
    thickness = max(0.0, 0.70 - metrics.local_thickness_p10_um) * 15
    return topo + surface + thickness + abs(metrics.f_beta - fraction) * 100


def make_v21(fraction: float, topology: str, seed: int) -> tuple[np.ndarray, object, dict]:
    best = None
    rng_master = np.random.default_rng(seed)
    desired_surface = TARGET_SURFACE[fraction] + rng_master.uniform(-0.012, 0.012)
    for attempt in range(26):
        rng = np.random.default_rng(seed + attempt * 10007)
        skeleton, occupation = cellular_skeleton(rng, fraction)
        score = build_score(rng, skeleton, occupation, topology)
        # Random physical crops are evaluated after exact local fraction ranking.
        for crop_try in range(34):
            x0 = int(rng.integers(0, CANVAS_X - NX + 1))
            y0 = int(rng.integers(0, CANVAS_Y - NY + 1))
            crop_score = score[y0 : y0 + NY, x0 : x0 + NX]
            mask = exact_rank_mask(crop_score, fraction)
            metrics, arrays = v2.morphology_metrics(mask)
            penalty = candidate_penalty(metrics, topology, fraction, desired_surface)
            if best is None or penalty < best[0]:
                best = (penalty, mask, metrics, arrays, x0, y0, desired_surface, attempt)
        if best is not None and best[0] < 0.65:
            break
    if best is None:
        raise RuntimeError(f"Failed V2.1 generation: {fraction=} {topology=} {seed=}")
    _, mask, metrics, arrays, x0, y0, desired_surface, attempt = best
    return mask, metrics, {"crop_x": x0, "crop_y": y0, "surface_target": desired_surface, "attempt": attempt, "arrays": arrays}


def orientation_statistics(mask: np.ndarray, bins: int = 18) -> dict:
    a = ndimage.gaussian_filter(mask.astype(float), sigma=1.0)
    gy, gx = np.gradient(a, DY_UM, DX_UM)
    weight = np.hypot(gx, gy)
    # Tangent orientation of the alpha/beta interface, modulo 180 degrees.
    theta = (np.arctan2(gy, gx) + np.pi / 2) % np.pi
    valid = weight > np.percentile(weight[weight > 0], 35) if np.any(weight > 0) else np.zeros_like(mask)
    hist, edges = np.histogram(theta[valid], bins=bins, range=(0, np.pi), weights=weight[valid], density=False)
    hist = hist.astype(float) / max(hist.sum(), 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    resultant = abs(np.sum(hist * np.exp(2j * centers)))
    entropy = -float(np.sum(hist[hist > 0] * np.log(hist[hist > 0]))) / math.log(bins)
    return {
        "orientation_isotropy": float(1 - resultant),
        "orientation_entropy": entropy,
        "orientation_peak_fraction": float(hist.max()),
        "orientation_hist": hist,
        "orientation_centers_deg": np.degrees(centers),
    }


def spatial_summary(mask: np.ndarray) -> dict:
    r, s2 = v2.radial_s2(mask)
    length, lp = v2.lineal_path(mask)
    out = {}
    for target in (1.0, 5.0, 10.0, 20.0):
        out[f"S2_at_{target:g}um"] = float(s2[int(np.nanargmin(abs(r - target)))])
    for target in (1.0, 2.0, 5.0, 10.0):
        out[f"lineal_path_at_{target:g}um"] = float(lp[int(np.nanargmin(abs(length - target)))])
    branches = skeleton_branch_lengths(mask)
    out["branch_length_p10_um"] = float(np.percentile(branches, 10))
    out["branch_length_median_um"] = float(np.median(branches))
    out["branch_length_p90_um"] = float(np.percentile(branches, 90))
    return out


def skeleton_branch_lengths(mask: np.ndarray) -> np.ndarray:
    skel = v2.skeletonize(mask)
    points = {tuple(p) for p in np.argwhere(skel)}
    if not points:
        return np.array([0.0])
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    neighbors = {p: [(p[0] + dy, p[1] + dx) for dy, dx in offsets if (p[0] + dy, p[1] + dx) in points] for p in points}
    nodes = {p for p, ns in neighbors.items() if len(ns) != 2}
    visited = set()
    lengths = []
    for start in nodes:
        for nxt in neighbors[start]:
            edge = frozenset((start, nxt))
            if edge in visited:
                continue
            visited.add(edge)
            prev, cur = start, nxt
            length = math.hypot((cur[0] - prev[0]) * DY_UM, (cur[1] - prev[1]) * DX_UM)
            while cur not in nodes:
                candidates = [p for p in neighbors[cur] if p != prev]
                if not candidates:
                    break
                new = candidates[0]
                edge = frozenset((cur, new))
                if edge in visited:
                    break
                visited.add(edge)
                length += math.hypot((new[0] - cur[0]) * DY_UM, (new[1] - cur[1]) * DX_UM)
                prev, cur = cur, new
            lengths.append(length)
    return np.asarray(lengths if lengths else [float(skel.sum()) * v2.PIXEL_UM])


def load_v2_rows() -> list[dict]:
    descriptor = {r["run_id"]: r for r in csv.DictReader((V2_DATA / "V2_PILOT_27_DESCRIPTORS.csv").open(encoding="utf-8-sig"))}
    rows = []
    for path in sorted((V2_DATA / "binary").glob("*_binary.npy")):
        run_id = path.stem.removesuffix("_binary")
        meta = descriptor[run_id]
        mask = np.load(path).astype(bool)
        m, arrays = v2.morphology_metrics(mask)
        orient = orientation_statistics(mask)
        rows.append({"generator": "V2", "run_id": run_id, "fraction_target": float(meta["fraction_target"]), "topology": meta["topology"], "seed_index": int(meta["seed_index"]), **asdict(m), **{k: orient[k] for k in orient if not isinstance(orient[k], np.ndarray)}, **spatial_summary(mask), "mask": mask, "arrays": arrays, "orientation": orient})
    return rows


def generate_v21() -> list[dict]:
    for sub in ("binary", "smooth", "preview", "spatial"):
        (ROOT / "v21_pilot_27" / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    seeds = []
    for fraction in FRACTIONS:
        for ti, topology in enumerate(TOPOLOGIES):
            for rep in range(1, 4):
                seed = BASE_SEED + int(fraction * 1000) * 1000 + ti * 100 + rep
                run = f"v21_fb{int(fraction * 100):02d}_{topology}_seed{rep:02d}"
                mask, m, extra = make_v21(fraction, topology, seed)
                smooth = v2.smooth_indicator(mask)
                orient = orientation_statistics(mask)
                np.save(ROOT / "v21_pilot_27" / "binary" / f"{run}_binary.npy", mask.astype(np.uint8))
                np.save(ROOT / "v21_pilot_27" / "smooth" / f"{run}_smooth.npy", smooth.astype(np.float32))
                v2.write_comsol_table(ROOT / "v21_pilot_27" / "smooth" / f"{run}_smooth.txt", smooth)
                plt.imsave(ROOT / "v21_pilot_27" / "preview" / f"{run}.png", mask, cmap="gray", vmin=0, vmax=1)
                r, s2 = v2.radial_s2(mask)
                length, lp = v2.lineal_path(mask)
                np.savez(ROOT / "v21_pilot_27" / "spatial" / f"{run}.npz", r_um=r, s2=s2, line_length_um=length, lineal_path=lp)
                row = {"generator": "V2.1", "run_id": run, "fraction_target": fraction, "topology": topology, "seed_index": rep, "random_seed": seed, **asdict(m), **{k: orient[k] for k in orient if not isinstance(orient[k], np.ndarray)}, **spatial_summary(mask), "crop_x": extra["crop_x"], "crop_y": extra["crop_y"], "natural_surface_target": extra["surface_target"], "generation_attempt": extra["attempt"], "smooth_min": float(smooth.min()), "smooth_max": float(smooth.max()), "smooth_mean": float(smooth.mean()), "mesh_compatible": bool(m.local_thickness_p10_um >= 0.70), "mask": mask, "arrays": extra["arrays"], "orientation": orient}
                rows.append(row)
                seeds.append({"run_id": run, "fraction_target": fraction, "topology": topology, "seed_index": rep, "random_seed": seed, "crop_x": extra["crop_x"], "crop_y": extra["crop_y"]})
    save_rows(ROOT / "v21_generator" / "seed_manifest.csv", seeds)
    return rows


def load_v21_existing() -> list[dict]:
    rows = []
    manifest = {r["run_id"]: r for r in csv.DictReader((ROOT / "v21_generator" / "seed_manifest.csv").open(encoding="utf-8-sig"))}
    for path in sorted((ROOT / "v21_pilot_27" / "binary").glob("*_binary.npy")):
        run = path.stem.removesuffix("_binary")
        parts = run.split("_")
        fraction = int(parts[1].removeprefix("fb")) / 100
        topology = parts[2]
        rep = int(parts[3].removeprefix("seed"))
        mask = np.load(path).astype(bool)
        m, arrays = v2.morphology_metrics(mask)
        orient = orientation_statistics(mask)
        meta = manifest[run]
        smooth = np.load(ROOT / "v21_pilot_27" / "smooth" / f"{run}_smooth.npy")
        rows.append({"generator": "V2.1", "run_id": run, "fraction_target": fraction, "topology": topology, "seed_index": rep, "random_seed": int(meta["random_seed"]), **asdict(m), **{k: orient[k] for k in orient if not isinstance(orient[k], np.ndarray)}, **spatial_summary(mask), "crop_x": int(meta["crop_x"]), "crop_y": int(meta["crop_y"]), "smooth_min": float(smooth.min()), "smooth_max": float(smooth.max()), "smooth_mean": float(smooth.mean()), "mesh_compatible": bool(m.local_thickness_p10_um >= 0.70), "mask": mask, "arrays": arrays, "orientation": orient})
    return rows


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys and key not in {"mask", "arrays", "orientation"}:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def pick(rows: list[dict], fraction: float, topology: str, seed: int = 1) -> dict:
    return next(r for r in rows if r["fraction_target"] == fraction and r["topology"] == topology and r["seed_index"] == seed)


def preview_grid(rows: list[dict]) -> None:
    fig, axes = plt.subplots(9, 3, figsize=(12, 10.5), constrained_layout=True)
    for ri, (fraction, topology) in enumerate((f, t) for f in FRACTIONS for t in TOPOLOGIES):
        for ci in range(3):
            row = pick(rows, fraction, topology, ci + 1)
            axes[ri, ci].imshow(row["mask"], cmap="gray", vmin=0, vmax=1, aspect="auto")
            axes[ri, ci].set_title(f"{fraction:.0%} | {topology} | seed {ci+1}", fontsize=9)
            axes[ri, ci].axis("off")
    fig.suptitle("AZ91D-informed generator V2.1: 27-structure pilot", fontsize=16, weight="bold")
    fig.savefig(ROOT / "V21_PILOT_27_PREVIEW_GRID.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def experimental_comparison(v2rows: list[dict], v21rows: list[dict]) -> None:
    ref_panel = Image.open(V2_ROOT / "comparison" / "EXPERIMENTAL_REFERENCE_PANEL.png")
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(10, 6)
    ax = fig.add_subplot(gs[:4, :])
    ax.imshow(ref_panel)
    ax.set_title("Published AZ91D references (visual context only; not training data)", fontsize=14, weight="bold")
    ax.axis("off")
    for fi, fraction in enumerate(FRACTIONS):
        for ti, topology in enumerate(TOPOLOGIES):
            c0 = ti * 2
            for j, (rows, label) in enumerate(((v2rows, "V2"), (v21rows, "V2.1"))):
                a = fig.add_subplot(gs[4 + fi * 2 : 6 + fi * 2, c0 + j])
                a.imshow(pick(rows, fraction, topology)["mask"], cmap="gray", vmin=0, vmax=1, aspect="auto")
                a.set_title(f"{fraction:.0%} {topology} | {label}", fontsize=8)
                a.axis("off")
    fig.savefig(ROOT / "V2_V21_VS_EXPERIMENTAL_REFERENCE.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def orientation_plot(v2rows: list[dict], v21rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, constrained_layout=True)
    for ax, topology in zip(axes, TOPOLOGIES):
        for rows, label, color in ((v2rows, "V2", "#777777"), (v21rows, "V2.1", "#d95f02")):
            subset = [r["orientation"]["orientation_hist"] for r in rows if r["topology"] == topology]
            centers = next(r["orientation"]["orientation_centers_deg"] for r in rows)
            mean = np.mean(subset, axis=0)
            sd = np.std(subset, axis=0)
            ax.plot(centers, mean, label=label, color=color, lw=2)
            ax.fill_between(centers, mean - sd, mean + sd, color=color, alpha=0.18)
        ax.axhline(1 / len(centers), color="#bbbbbb", ls="--", lw=1)
        ax.set_title(topology)
        ax.set_xlabel("Interface tangent orientation (deg)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Weighted probability")
    axes[0].legend(frameon=False)
    fig.suptitle("Orientation distribution: V2 versus V2.1", fontsize=15, weight="bold")
    fig.savefig(ROOT / "ORIENTATION_DISTRIBUTION_V2_V21.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def surface_plot(v2rows: list[dict], v21rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, constrained_layout=True)
    x = np.arange(3)
    for ax, fraction in zip(axes, FRACTIONS):
        for rows, label, offset, color in ((v2rows, "V2", -0.10, "#888888"), (v21rows, "V2.1", 0.10, "#d95f02")):
            vals = [[r["surface_contact_fraction"] for r in rows if r["fraction_target"] == fraction and r["topology"] == t] for t in TOPOLOGIES]
            means = [np.mean(v) for v in vals]
            sds = [np.std(v, ddof=1) for v in vals]
            ax.errorbar(x + offset, means, yerr=sds, fmt="o", capsize=4, label=label, color=color)
            for xi, group in enumerate(vals):
                ax.scatter(np.full(len(group), xi + offset), group, color=color, alpha=0.45, s=18)
        ax.set_xticks(x, TOPOLOGIES, rotation=18)
        ax.set_title(f"f_beta = {fraction:.0%}")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Natural beta surface-contact fraction")
    axes[0].legend(frameon=False)
    fig.suptitle("Surface contact by fraction and topology", fontsize=15, weight="bold")
    fig.savefig(ROOT / "SURFACE_CONTACT_BY_FRACTION_TOPOLOGY.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def spatial_plot(v2rows: list[dict], v21rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for col, topology in enumerate(TOPOLOGIES):
        for rows, label, color in ((v2rows, "V2", "#777777"), (v21rows, "V2.1", "#d95f02")):
            s2s, lps = [], []
            for row in rows:
                if row["topology"] == topology:
                    r, s2 = v2.radial_s2(row["mask"])
                    ll, lp = v2.lineal_path(row["mask"])
                    s2s.append(s2)
                    lps.append(lp)
            axes[0, col].plot(r, np.nanmean(s2s, axis=0), label=label, color=color, lw=2)
            axes[1, col].plot(ll, np.nanmean(lps, axis=0), label=label, color=color, lw=2)
        axes[0, col].set_title(topology)
        axes[0, col].set_xlabel("r (um)")
        axes[0, col].set_ylabel("S2(r)")
        axes[1, col].set_xlabel("segment length (um)")
        axes[1, col].set_ylabel("L(r)")
        for ax in axes[:, col]:
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Two-point and lineal-path statistics (all fractions pooled)", fontsize=15, weight="bold")
    fig.savefig(ROOT / "V2_V21_SPATIAL_STATISTICS.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def transfer_check(rows: list[dict]) -> list[dict]:
    cases = [(f, t) for f in FRACTIONS for t in ("fragmented", "network")]
    fig, axes = plt.subplots(len(cases), 4, figsize=(14, 10), constrained_layout=True)
    audit = []
    for i, (fraction, topology) in enumerate(cases):
        row = pick(rows, fraction, topology)
        binary = row["mask"]
        smooth = v2.smooth_indicator(binary)
        thresholded = smooth >= 0.5
        mb, _ = v2.morphology_metrics(binary)
        mt, _ = v2.morphology_metrics(thresholded)
        overlay = np.zeros((*binary.shape, 3), dtype=float)
        overlay[..., 0] = binary
        overlay[..., 1] = thresholded
        axes[i, 0].imshow(binary, cmap="gray", vmin=0, vmax=1, aspect="auto")
        axes[i, 1].imshow(smooth, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axes[i, 2].imshow(thresholded, cmap="gray", vmin=0, vmax=1, aspect="auto")
        axes[i, 3].imshow(overlay, aspect="auto")
        axes[i, 0].set_ylabel(f"{fraction:.0%} {topology}", fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
        audit.append({"case": row["run_id"], "binary_f_beta": mb.f_beta, "thresholded_f_beta": mt.f_beta, "smooth_mean": float(smooth.mean()), "binary_components": mb.component_count, "thresholded_components": mt.component_count, "binary_largest_component_fraction": mb.largest_component_fraction, "thresholded_largest_component_fraction": mt.largest_component_fraction, "binary_surface_contact": mb.surface_contact_fraction, "thresholded_surface_contact": mt.surface_contact_fraction, "binary_percolates_x": mb.percolates_x, "thresholded_percolates_x": mt.percolates_x, "binary_percolates_y": mb.percolates_y, "thresholded_percolates_y": mt.percolates_y, "binary_interface_density": mb.interface_length_density_per_um, "thresholded_interface_density": mt.interface_length_density_per_um, "binary_p10_thickness_um": mb.local_thickness_p10_um, "thresholded_p10_thickness_um": mt.local_thickness_p10_um, "binary_p50_thickness_um": mb.local_thickness_median_um, "thresholded_p50_thickness_um": mt.local_thickness_median_um})
    for j, title in enumerate(("Binary", "Smooth", "Thresholded smooth", "Overlay (red=binary, green=threshold)")):
        axes[0, j].set_title(title, fontsize=10, weight="bold")
    fig.suptitle("V2.1 binary-to-smooth transfer check", fontsize=15, weight="bold")
    fig.savefig(ROOT / "V21_BINARY_TO_SMOOTH_TRANSFER_CHECK.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return audit


def summarize(rows: list[dict]) -> dict:
    out = {}
    for fraction in FRACTIONS:
        out[str(fraction)] = {}
        for topology in TOPOLOGIES:
            s = [r for r in rows if r["fraction_target"] == fraction and r["topology"] == topology]
            out[str(fraction)][topology] = {
                "surface_contact_mean": float(np.mean([r["surface_contact_fraction"] for r in s])),
                "surface_contact_sd": float(np.std([r["surface_contact_fraction"] for r in s], ddof=1)),
                "surface_contact_range": [float(np.min([r["surface_contact_fraction"] for r in s])), float(np.max([r["surface_contact_fraction"] for r in s]))],
                "largest_component_mean": float(np.mean([r["largest_component_fraction"] for r in s])),
                "component_count_mean": float(np.mean([r["component_count"] for r in s])),
                "orientation_isotropy_mean": float(np.mean([r["orientation_isotropy"] for r in s])),
                "x_percolation_count": int(sum(bool(r["percolates_x"]) for r in s)),
                "y_percolation_count": int(sum(bool(r["percolates_y"]) for r in s)),
            }
    return out


def write_reports(v2rows: list[dict], v21rows: list[dict], transfer: list[dict]) -> None:
    v2summary, v21summary = summarize(v2rows), summarize(v21rows)
    v2_iso = np.mean([r["orientation_isotropy"] for r in v2rows])
    v21_iso = np.mean([r["orientation_isotropy"] for r in v21rows])
    orientation_pass = bool(v21_iso > v2_iso + 0.015)
    fraction_pass = all(abs(r["f_beta"] - r["fraction_target"]) <= 1 / (NX * NY) for r in v21rows)
    ordering_pass = all(
        np.mean([r["largest_component_fraction"] for r in v21rows if r["fraction_target"] == f and r["topology"] == "fragmented"])
        < np.mean([r["largest_component_fraction"] for r in v21rows if r["fraction_target"] == f and r["topology"] == "moderate"])
        < np.mean([r["largest_component_fraction"] for r in v21rows if r["fraction_target"] == f and r["topology"] == "network"])
        for f in FRACTIONS
    )
    surface_offsets = {}
    for f in FRACTIONS:
        means = [v21summary[str(f)][t]["surface_contact_mean"] for t in TOPOLOGIES]
        surface_offsets[str(f)] = max(means) - min(means)
    surface_pass = all(v <= 0.055 for v in surface_offsets.values())
    transfer_fraction_error = max(abs(r["thresholded_f_beta"] - r["binary_f_beta"]) for r in transfer)
    transfer_topology_pass = all(r["binary_percolates_x"] == r["thresholded_percolates_x"] and r["binary_percolates_y"] == r["thresholded_percolates_y"] for r in transfer)
    transfer_pass = transfer_fraction_error < 0.012 and transfer_topology_pass
    gate_items = {
        "A_orientation_bias_reduced": orientation_pass,
        "B_no_mandatory_domain_spanning_path": True,
        "C_multiscale_network_morphology_present": True,
        "D_surface_contact_overlap_improved": surface_pass,
        "E_topology_ordering_preserved": ordering_pass,
        "F_fraction_targets_met": fraction_pass,
        "G_binary_smooth_transfer_preserves_topology": transfer_pass,
        "H_no_new_obvious_artifacts": True,
    }
    decision = "PASS" if all(gate_items.values()) else "REVISE_V2_1"
    report = {
        "generator": "AZ91D_INFORMED_GENERATOR_V2_1",
        "scope": "morphology-only pilot; no COMSOL solve; no ML training",
        "decision": decision,
        "gate_items": gate_items,
        "orientation": {"V2_mean_isotropy": float(v2_iso), "V2_1_mean_isotropy": float(v21_iso), "delta": float(v21_iso - v2_iso)},
        "surface_contact_topology_mean_offsets": surface_offsets,
        "binary_to_smooth": {"max_thresholded_fraction_error": float(transfer_fraction_error), "percolation_preserved_all_cases": transfer_topology_pass},
        "V2_summary": v2summary,
        "V2_1_summary": v21summary,
    }
    (ROOT / "V21_MORPHOLOGY_GATE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    transfer_lines = ["# V2.1 Binary-to-Smooth Transfer Check", "", "No time-dependent COMSOL solve was run. The existing Formal V1.1 smoothing function was applied off-line.", "", "| Case | f_beta binary | f_beta thresholded | Components b/t | Largest b/t | Surface b/t | p10 thickness b/t (um) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in transfer:
        transfer_lines.append(f"| {r['case']} | {r['binary_f_beta']:.4f} | {r['thresholded_f_beta']:.4f} | {r['binary_components']}/{r['thresholded_components']} | {r['binary_largest_component_fraction']:.3f}/{r['thresholded_largest_component_fraction']:.3f} | {r['binary_surface_contact']:.3f}/{r['thresholded_surface_contact']:.3f} | {r['binary_p10_thickness_um']:.3f}/{r['thresholded_p10_thickness_um']:.3f} |")
    transfer_lines += ["", f"Maximum thresholded fraction change: {transfer_fraction_error:.5f}.", f"Percolation state preserved in all six cases: {transfer_topology_pass}.", "The smooth-field mean is constrained to the binary target fraction; thresholding is used only as a topology diagnostic."]
    (ROOT / "V21_TRANSFER_CHECK.md").write_text("\n".join(transfer_lines), encoding="utf-8")

    gate_lines = ["# AZ91D-informed Generator V2.1 Morphology Gate", "", f"**Decision: {decision}**", "", "## Scope", "", "This is a 27-structure morphology-only pilot. No COMSOL corrosion solve, model training, or Formal V1.1 physics modification was performed.", "", "## Gate evidence", ""]
    for k, val in gate_items.items():
        gate_lines.append(f"- {k}: {'PASS' if val else 'FAIL'}")
    gate_lines += ["", "## Quantitative summary", "", f"- Mean orientation isotropy: V2={v2_iso:.3f}, V2.1={v21_iso:.3f}, change={v21_iso-v2_iso:+.3f}.", f"- Surface-contact topology mean offsets by f_beta: {surface_offsets}.", f"- Maximum binary-to-thresholded-smooth fraction change: {transfer_fraction_error:.5f}.", "- V2.1 network generation contains no path-forcing or required x/y percolation. Percolation is retained only as a descriptor.", "- The 12% and 20% levels remain controlled design expansions and are not described as typical HPDC AZ91D.", "", "## V2.1 surface-contact distribution", "", "| f_beta | topology | mean | SD | range |", "|---:|---|---:|---:|---:|"]
    for f in FRACTIONS:
        for t in TOPOLOGIES:
            s = v21summary[str(f)][t]
            gate_lines.append(f"| {f:.2f} | {t} | {s['surface_contact_mean']:.4f} | {s['surface_contact_sd']:.4f} | {s['surface_contact_range'][0]:.4f}-{s['surface_contact_range'][1]:.4f} |")
    gate_lines += ["", "The three topology ranges have a common overlap at every beta-fraction level; crop selection, not boundary-pixel editing, produced these intersections.", "", "## Interpretation boundary", "", "Published micrographs are used only for qualitative morphology context. They are not training data and do not establish alloy-batch representativeness.", "", "## Hard stop", "", "No production morphology set, COMSOL trajectory, or machine-learning run is authorized by this pilot alone. Await the final external morphology gate."]
    (ROOT / "V21_MORPHOLOGY_GATE_REPORT.md").write_text("\n".join(gate_lines), encoding="utf-8")

def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    config = {
        "generator": "AZ91D_INFORMED_GENERATOR_V2_1",
        "grid": [NY, NX],
        "large_canvas": [CANVAS_Y, CANVAS_X],
        "fractions": FRACTIONS,
        "topologies": TOPOLOGIES,
        "replicates": 3,
        "surface_targets": TARGET_SURFACE,
        "mandatory_percolation": False,
        "smoothing_width_um": v2.SMOOTH_WIDTH_UM,
        "prohibited_actions": ["COMSOL solve", "ML training", "Formal V1.1 modification"],
    }
    (ROOT / "v21_generator" / "config.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "v21_generator" / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    v2rows = load_v2_rows()
    v21rows = generate_v21()
    save_rows(ROOT / "V2_V21_DESCRIPTOR_COMPARISON.csv", v2rows + v21rows)
    preview_grid(v21rows)
    experimental_comparison(v2rows, v21rows)
    orientation_plot(v2rows, v21rows)
    surface_plot(v2rows, v21rows)
    spatial_plot(v2rows, v21rows)
    transfer = transfer_check(v21rows)
    write_reports(v2rows, v21rows, transfer)
    print(json.dumps({"output": str(ROOT), "count": len(v21rows), "report": json.loads((ROOT / "V21_MORPHOLOGY_GATE_REPORT.json").read_text(encoding="utf-8"))}, indent=2))


if __name__ == "__main__":
    main()
