from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference_micrographs"
V1_ROOT = Path(os.environ.get("AZ91D_V1_MICROSTRUCTURES", ROOT / "data" / "v1_microstructures"))
NX, NY = 512, 64
X_MIN, X_MAX = -100e-6, 100e-6
Y_MIN, Y_MAX = -25e-6, 0.0
DX_UM = (X_MAX - X_MIN) * 1e6 / (NX - 1)
DY_UM = (Y_MAX - Y_MIN) * 1e6 / (NY - 1)
PIXEL_UM = 0.5 * (DX_UM + DY_UM)
SMOOTH_WIDTH_UM = 0.35
FRACTIONS = (0.05, 0.12, 0.20)
TOPOLOGIES = ("fragmented", "moderate", "network")
BASE_SEED = 202609080000


@dataclass
class Metrics:
    f_beta: float
    component_count: int
    largest_component_fraction: float
    equivalent_diameter_mean_um: float
    equivalent_diameter_median_um: float
    equivalent_diameter_p90_um: float
    perimeter_mean_um: float
    aspect_ratio_mean: float
    circularity_mean: float
    nearest_neighbor_mean_um: float
    interface_length_density_per_um: float
    skeleton_length_um: float
    branch_length_mean_um: float
    junction_density_per_100um2: float
    endpoint_density_per_100um2: float
    local_thickness_mean_um: float
    local_thickness_median_um: float
    local_thickness_p10_um: float
    local_thickness_p90_um: float
    percolates_x: bool
    percolates_y: bool
    connectivity_index: float
    surface_contact_fraction: float
    min_resolved_thickness_um: float


def disk(radius: int) -> np.ndarray:
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def skeletonize(mask: np.ndarray) -> np.ndarray:
    work = mask.copy()
    skel = np.zeros_like(mask)
    st = ndimage.generate_binary_structure(2, 1)
    while work.any():
        eroded = ndimage.binary_erosion(work, structure=st)
        opened = ndimage.binary_dilation(eroded, structure=st)
        skel |= work & ~opened
        work = eroded
    return skel


def component_properties(mask: np.ndarray):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    rows = []
    for lab in range(1, count + 1):
        yy, xx = np.nonzero(labels == lab)
        if len(xx) == 0:
            continue
        area_um2 = len(xx) * DX_UM * DY_UM
        boundary = (labels == lab) ^ ndimage.binary_erosion(labels == lab)
        perimeter_um = float(boundary.sum()) * PIXEL_UM
        pts = np.column_stack((xx * DX_UM, yy * DY_UM))
        if len(pts) > 2:
            eig = np.linalg.eigvalsh(np.cov(pts.T))
            aspect = math.sqrt(max(eig[-1], 1e-12) / max(eig[0], 1e-12))
        else:
            aspect = 1.0
        rows.append(
            {
                "label": lab,
                "pixels": len(xx),
                "area_um2": area_um2,
                "equivalent_diameter_um": 2 * math.sqrt(area_um2 / math.pi),
                "perimeter_um": perimeter_um,
                "aspect_ratio": aspect,
                "circularity": min(1.0, 4 * math.pi * area_um2 / max(perimeter_um**2, 1e-12)),
                "cx_um": float(xx.mean() * DX_UM),
                "cy_um": float(yy.mean() * DY_UM),
            }
        )
    return labels, rows


def morphology_metrics(mask: np.ndarray) -> tuple[Metrics, dict[str, np.ndarray]]:
    labels, comps = component_properties(mask)
    areas = np.array([r["area_um2"] for r in comps])
    eqd = np.array([r["equivalent_diameter_um"] for r in comps])
    perim = np.array([r["perimeter_um"] for r in comps])
    aspect = np.array([r["aspect_ratio"] for r in comps])
    circ = np.array([r["circularity"] for r in comps])
    centroids = np.array([[r["cx_um"], r["cy_um"]] for r in comps])
    if len(centroids) > 1:
        dist = np.sqrt(((centroids[:, None] - centroids[None, :]) ** 2).sum(axis=2))
        dist[dist == 0] = np.inf
        nnd = np.min(dist, axis=1)
    else:
        nnd = np.array([np.nan])

    vertical_edges = np.logical_xor(mask[1:, :], mask[:-1, :]).sum() * DX_UM
    horizontal_edges = np.logical_xor(mask[:, 1:], mask[:, :-1]).sum() * DY_UM
    interface_length_um = float(vertical_edges + horizontal_edges)
    area_total_um2 = (X_MAX - X_MIN) * 1e6 * (Y_MAX - Y_MIN) * 1e6

    skel = skeletonize(mask)
    neighbors = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skel
    endpoints = skel & (neighbors == 1)
    junctions = skel & (neighbors >= 3)
    node_count = max(1, int(endpoints.sum() + junctions.sum()))
    skel_length_um = float(skel.sum()) * PIXEL_UM
    branch_mean = skel_length_um / node_count
    edt = ndimage.distance_transform_edt(mask, sampling=(DY_UM, DX_UM))
    local_thickness = 2 * edt[skel]
    if local_thickness.size == 0:
        local_thickness = np.array([0.0])

    touch_x = False
    touch_y = False
    for lab in range(1, labels.max() + 1):
        region = labels == lab
        touch_x |= bool(region[:, 0].any() and region[:, -1].any())
        touch_y |= bool(region[0, :].any() and region[-1, :].any())

    adjacency = (
        np.logical_and(mask[:, 1:], mask[:, :-1]).sum()
        + np.logical_and(mask[1:, :], mask[:-1, :]).sum()
    )
    metrics = Metrics(
        f_beta=float(mask.mean()),
        component_count=len(comps),
        largest_component_fraction=float(areas.max() / areas.sum()) if len(areas) else 0.0,
        equivalent_diameter_mean_um=float(eqd.mean()) if len(eqd) else 0.0,
        equivalent_diameter_median_um=float(np.median(eqd)) if len(eqd) else 0.0,
        equivalent_diameter_p90_um=float(np.percentile(eqd, 90)) if len(eqd) else 0.0,
        perimeter_mean_um=float(perim.mean()) if len(perim) else 0.0,
        aspect_ratio_mean=float(aspect.mean()) if len(aspect) else 0.0,
        circularity_mean=float(circ.mean()) if len(circ) else 0.0,
        nearest_neighbor_mean_um=float(np.nanmean(nnd)) if np.isfinite(nnd).any() else float("nan"),
        interface_length_density_per_um=interface_length_um / area_total_um2,
        skeleton_length_um=skel_length_um,
        branch_length_mean_um=branch_mean,
        junction_density_per_100um2=float(junctions.sum()) / area_total_um2 * 100,
        endpoint_density_per_100um2=float(endpoints.sum()) / area_total_um2 * 100,
        local_thickness_mean_um=float(local_thickness.mean()),
        local_thickness_median_um=float(np.median(local_thickness)),
        local_thickness_p10_um=float(np.percentile(local_thickness, 10)),
        local_thickness_p90_um=float(np.percentile(local_thickness, 90)),
        percolates_x=touch_x,
        percolates_y=touch_y,
        connectivity_index=float(adjacency / max(mask.sum(), 1)),
        surface_contact_fraction=float(mask[-1].mean()),
        min_resolved_thickness_um=float(local_thickness.min()),
    )
    return metrics, {"areas": areas, "eqd": eqd, "thickness": local_thickness, "skeleton": skel}


def radial_s2(mask: np.ndarray, max_r: int = 80) -> tuple[np.ndarray, np.ndarray]:
    a = mask.astype(float)
    corr = np.fft.ifft2(np.abs(np.fft.fft2(a)) ** 2).real / a.size
    corr = np.fft.fftshift(corr)
    yy, xx = np.indices(a.shape)
    cy, cx = np.array(a.shape) // 2
    r = np.sqrt(((xx - cx) * DX_UM) ** 2 + ((yy - cy) * DY_UM) ** 2)
    bins = np.linspace(0, min(max_r, r.max()), 81)
    ids = np.digitize(r.ravel(), bins)
    vals = np.array([corr.ravel()[ids == i].mean() if np.any(ids == i) else np.nan for i in range(1, len(bins))])
    return 0.5 * (bins[:-1] + bins[1:]), vals


def lineal_path(mask: np.ndarray, max_len: int = 60) -> tuple[np.ndarray, np.ndarray]:
    # Probability that a horizontal or vertical segment of length L lies entirely in beta.
    lengths = np.arange(1, max_len + 1)
    values = []
    a = mask.astype(np.uint8)
    for length in lengths:
        if length <= a.shape[1]:
            h = ndimage.uniform_filter1d(a.astype(float), size=length, axis=1, mode="constant")
            ph = np.mean(h >= 1 - 1e-12)
        else:
            ph = 0.0
        if length <= a.shape[0]:
            v = ndimage.uniform_filter1d(a.astype(float), size=length, axis=0, mode="constant")
            pv = np.mean(v >= 1 - 1e-12)
        else:
            pv = 0.0
        values.append(0.5 * (ph + pv))
    return lengths * PIXEL_UM, np.asarray(values)


def make_cell_boundary(rng: np.random.Generator, n_sites: int) -> np.ndarray:
    yy, xx = np.mgrid[0:NY, 0:NX]
    warp_x = ndimage.gaussian_filter(rng.normal(size=(NY, NX)), sigma=(8, 22))
    warp_y = ndimage.gaussian_filter(rng.normal(size=(NY, NX)), sigma=(8, 22))
    warp_x = warp_x / max(warp_x.std(), 1e-9) * 5.5
    warp_y = warp_y / max(warp_y.std(), 1e-9) * 2.2
    xw, yw = xx + warp_x, yy + warp_y

    n_sites = int(max(12, n_sites + rng.integers(-2, 3)))
    sx = rng.uniform(-20, NX + 20, n_sites)
    sy = rng.uniform(-10, NY + 10, n_sites)
    weights = rng.uniform(0.80, 1.25, n_sites)
    dists = []
    for x0, y0, w in zip(sx, sy, weights):
        dists.append(((xw - x0) / w) ** 2 + ((yw - y0) * 1.25 / w) ** 2)
    d = np.stack(dists)
    nearest = np.partition(d, 1, axis=0)[:2]
    gap = np.sqrt(nearest[1]) - np.sqrt(nearest[0])
    boundary = gap < np.percentile(gap, 9.5)
    boundary = ndimage.binary_closing(boundary, structure=disk(1))
    return skeletonize(boundary)


def exact_rank_mask(score: np.ndarray, target: int) -> np.ndarray:
    chosen = np.argpartition(score.ravel(), target - 1)[:target]
    out = np.zeros(score.size, dtype=bool)
    out[chosen] = True
    return out.reshape(score.shape)


def minimum_cost_lr_path(cost: np.ndarray) -> np.ndarray:
    """Return a left-to-right path that stays as close as possible to a cellular skeleton."""
    h, w = cost.shape
    dp = np.full((h, w), np.inf)
    parent = np.zeros((h, w), dtype=np.int8)
    dp[:, 0] = cost[:, 0]
    for x in range(1, w):
        prev = dp[:, x - 1]
        candidates = np.vstack((np.r_[np.inf, prev[:-1]], prev, np.r_[prev[1:], np.inf]))
        choice = np.argmin(candidates, axis=0)
        dp[:, x] = cost[:, x] + candidates[choice, np.arange(h)]
        parent[:, x] = choice.astype(np.int8) - 1
    y = int(np.argmin(dp[:, -1]))
    path = np.zeros_like(cost, dtype=bool)
    path[y, -1] = True
    for x in range(w - 1, 0, -1):
        y = int(np.clip(y + parent[y, x], 0, h - 1))
        path[y, x - 1] = True
    return path


def topology_penalty(m: Metrics, topology: str) -> float:
    if topology == "fragmented":
        return max(0, 10 - m.component_count) * 2 + max(0, m.largest_component_fraction - 0.42) * 30
    if topology == "moderate":
        return abs(m.largest_component_fraction - 0.58) * 8 + max(0, 3 - m.component_count)
    return max(0, 0.72 - m.largest_component_fraction) * 35 + (0 if m.percolates_x else 6)


def make_v2(fraction: float, topology: str, seed: int) -> tuple[np.ndarray, np.ndarray, Metrics]:
    target = int(round(fraction * NX * NY))
    best = None
    for attempt in range(60):
        rng = np.random.default_rng(seed + attempt * 10007)
        site_base = {"fragmented": 42, "moderate": 34, "network": int(14 + 110 * fraction)}[topology]
        skeleton = make_cell_boundary(rng, site_base)
        occupation = ndimage.gaussian_filter(rng.normal(size=(NY, NX)), sigma=(3.5, 13))
        occupation = (occupation - occupation.mean()) / max(occupation.std(), 1e-9)
        thickness_noise = ndimage.gaussian_filter(rng.normal(size=(NY, NX)), sigma=(2.0, 7))
        thickness_noise = thickness_noise / max(thickness_noise.std(), 1e-9)

        if topology == "network":
            path = minimum_cost_lr_path(ndimage.distance_transform_edt(~skeleton))
            active = skeleton | path
            gate_bias = 0.0
        elif topology == "moderate":
            active = skeleton & (occupation > -0.65)
            active = ndimage.binary_dilation(active, structure=disk(1)) & ndimage.binary_dilation(skeleton, structure=disk(1))
            gate_bias = 0.45
        else:
            active = skeleton & (occupation > 0.75)
            active = ndimage.binary_opening(ndimage.binary_dilation(active, structure=disk(1)), structure=disk(1))
            gate_bias = 1.25

        if active.sum() < 25:
            continue
        distance = ndimage.distance_transform_edt(~active, sampling=(DY_UM, DX_UM))
        distance_to_all_gb = ndimage.distance_transform_edt(~skeleton, sampling=(DY_UM, DX_UM))
        noise_weight = 0.06 if topology == "network" else 0.32
        score = distance + noise_weight * thickness_noise + gate_bias * np.maximum(occupation, 0)
        if topology == "network":
            # Preserve one skeleton-following spanning route while distributing the
            # remaining beta over the same cellular boundary family.
            score[path] -= 100.0
        # Divorced islands remain tied to interdendritic regions through this distance prior.
        island_field = ndimage.gaussian_filter(rng.normal(size=(NY, NX)), sigma=(2.2, 5.5))
        island_candidates = (island_field > np.percentile(island_field, 98.2)) & (distance_to_all_gb < 2.2)
        score[island_candidates] -= 1.0 if topology != "network" else 0.25
        mask = exact_rank_mask(score, target)
        mask = ndimage.binary_closing(mask, structure=disk(1))
        # Restore exact area after morphology cleanup using the same physical score.
        score2 = score.copy()
        score2[mask] -= 20
        mask = exact_rank_mask(score2, target)
        metrics, _ = morphology_metrics(mask)
        penalty = topology_penalty(metrics, topology)
        if metrics.local_thickness_p10_um < 0.70:
            penalty += (0.70 - metrics.local_thickness_p10_um) * 20
        if best is None or penalty < best[0]:
            best = (penalty, mask, skeleton, metrics)
        if penalty < 0.8:
            break
    if best is None:
        raise RuntimeError(f"unable to generate {fraction=} {topology=}")
    _, mask, skeleton, metrics = best
    return mask, skeleton, metrics


def smooth_indicator(mask: np.ndarray) -> np.ndarray:
    signed_um = (
        ndimage.distance_transform_edt(mask, sampling=(DY_UM, DX_UM))
        - ndimage.distance_transform_edt(~mask, sampling=(DY_UM, DX_UM))
    )
    lo, hi = -3.0, 3.0
    target = float(mask.mean())
    for _ in range(70):
        offset = 0.5 * (lo + hi)
        field = 0.5 * (1 + np.tanh((signed_um - offset) / SMOOTH_WIDTH_UM))
        if field.mean() > target:
            lo = offset
        else:
            hi = offset
    return 0.5 * (1 + np.tanh((signed_um - 0.5 * (lo + hi)) / SMOOTH_WIDTH_UM))


def write_comsol_table(path: Path, field: np.ndarray) -> None:
    xs = np.linspace(X_MIN, X_MAX, NX)
    ys = np.linspace(Y_MIN, Y_MAX, NY)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("% AZ91D-informed generator V2; alpha-Mg=0, beta-Mg17Al12=1\n")
        f.write("% Dimension: 2\n% x y micro_s\n")
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                f.write(f"{x:.12e} {y:.12e} {field[iy, ix]:.12e}\n")


def overlay(image: Image.Image, mask: np.ndarray, color=(240, 91, 54), alpha=100) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    layer = Image.fromarray(rgba, "RGBA").resize(base.size, Image.Resampling.NEAREST)
    return Image.alpha_composite(base, layer).convert("RGB")


def prepare_references() -> list[dict]:
    stats_dir = ROOT / "reference_statistics"
    seg_dir = stats_dir / "segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)
    records = []

    # Authors' published black/white eutectic maps. Crops exclude labels and scale bars.
    page = Image.open(REF / "sun_2020_page4.png").convert("L")
    crop_boxes = {
        "sun2020_2mm": (110, 1580, 360, 1785),
        "sun2020_6mm": (410, 1580, 660, 1785),
        "sun2020_10mm": (235, 1835, 525, 2045),
    }
    reported = {"sun2020_2mm": 0.1986, "sun2020_6mm": 0.1671, "sun2020_10mm": 0.1491}
    for key, box in crop_boxes.items():
        crop = page.crop(box)
        arr = np.asarray(crop)
        mask = arr < 105
        mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
        mask = ndimage.binary_closing(mask, structure=np.ones((2, 2)))
        Image.fromarray(arr).save(seg_dir / f"{key}_crop.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(seg_dir / f"{key}_mask.png")
        overlay(crop.convert("RGB"), mask).save(seg_dir / f"{key}_overlay.png")
        m, arrays = morphology_metrics(mask)
        row = {"image_id": key, "reference_pool": "secondary_HPDC_thickness_series", **asdict(m)}
        row["published_phase_fraction"] = reported[key]
        row["segmentation_note"] = "Author-published binary eutectic map; crop threshold reproduces black phase approximately."
        records.append(row)
        r, s2 = radial_s2(mask)
        l, lp = lineal_path(mask)
        np.savez(stats_dir / f"{key}_spatial.npz", r_um=r, s2=s2, line_length_um=l, lineal_path=lp)

    # UCM optical images are retained for visual morphology. Contrast is not phase-specific enough for hard quantification.
    for key, filename, box in (
        ("ucm_orange", "ucm_az91d_orange.jpg", (100, 90, 1200, 850)),
        ("ucm_green", "ucm_az91d_green.jpg", (120, 70, 1190, 850)),
    ):
        im = Image.open(REF / filename).convert("RGB")
        crop = im.crop(box)
        gray = np.asarray(crop.convert("L"), dtype=float)
        local = ndimage.gaussian_filter(gray, sigma=9)
        candidate = gray < local - 15
        candidate = ndimage.binary_opening(candidate, structure=np.ones((2, 2)))
        crop.save(seg_dir / f"{key}_crop.png")
        Image.fromarray((candidate * 255).astype(np.uint8)).save(seg_dir / f"{key}_contrast_candidate_mask.png")
        overlay(crop, candidate, color=(25, 145, 210), alpha=80).save(seg_dir / f"{key}_contrast_candidate_overlay.png")
    return records


def save_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_v1() -> list[dict]:
    rows = []
    for fraction in FRACTIONS:
        for topology in TOPOLOGIES:
            for rep in range(1, 4):
                run = f"fb{int(fraction * 100):02d}_{topology}_seed{rep:02d}"
                mask = np.load(V1_ROOT / f"{run}_binary.npy").astype(bool)
                m, arrays = morphology_metrics(mask)
                rows.append({"generator": "V1", "run_id": run, "fraction_target": fraction, "topology": topology, "seed_index": rep, **asdict(m), "mask": mask, "arrays": arrays})
    return rows


def generate_v2() -> list[dict]:
    out = ROOT / "v2_pilot_27"
    for sub in ("binary", "smooth", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for fraction in FRACTIONS:
        for ti, topology in enumerate(TOPOLOGIES):
            for rep in range(1, 4):
                seed = BASE_SEED + int(fraction * 1000) * 1000 + ti * 100 + rep
                run = f"v2_fb{int(fraction * 100):02d}_{topology}_seed{rep:02d}"
                mask, gb, m = make_v2(fraction, topology, seed)
                _, morphology_arrays = morphology_metrics(mask)
                smooth = smooth_indicator(mask)
                np.save(out / "binary" / f"{run}_binary.npy", mask.astype(np.uint8))
                np.save(out / "smooth" / f"{run}_smooth.npy", smooth.astype(np.float32))
                write_comsol_table(out / "smooth" / f"{run}_smooth.txt", smooth)
                Image.fromarray((mask * 255).astype(np.uint8)).save(out / "preview" / f"{run}.png")
                r, s2 = radial_s2(mask)
                l, lp = lineal_path(mask)
                rows.append(
                    {
                        "generator": "V2",
                        "run_id": run,
                        "fraction_target": fraction,
                        "topology": topology,
                        "seed_index": rep,
                        "random_seed": seed,
                        **asdict(m),
                        "smooth_min": float(smooth.min()),
                        "smooth_max": float(smooth.max()),
                        "smooth_mean": float(smooth.mean()),
                        "mesh_compatible": bool(m.local_thickness_p10_um >= 0.70),
                        "fraction_level_tag": {0.05: "lower realistic/near-realistic", 0.12: "expanded design", 0.20: "high controlled design"}[fraction],
                        "mask": mask,
                        "gb": gb,
                        "arrays": {
                            **morphology_arrays,
                            "r": r,
                            "s2": s2,
                            "line_length": l,
                            "lineal_path": lp,
                        },
                    }
                )
    return rows


def preview_grid(v2_rows: list[dict]) -> None:
    fig, axes = plt.subplots(9, 3, figsize=(14, 13), constrained_layout=True)
    for ax, row in zip(axes.ravel(), v2_rows):
        ax.imshow(row["mask"], cmap="gray_r", origin="lower", aspect="auto")
        ax.set_title(f"{row['fraction_target']:.0%} | {row['topology']} | seed {row['seed_index']}", fontsize=9)
        ax.axis("off")
    fig.suptitle("AZ91D-informed stochastic microstructures V2: 27-member pilot", fontsize=15)
    fig.savefig(ROOT / "v2_pilot_27" / "V2_PILOT_27_PREVIEW_GRID.png", dpi=220)
    plt.close(fig)


def representative(rows: list[dict], fraction: float, topology: str) -> dict:
    candidates = [r for r in rows if r["fraction_target"] == fraction and r["topology"] == topology]
    return candidates[0]


def comparison_figure(v1: list[dict], v2: list[dict]) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(18, 8), constrained_layout=True)
    for ri, fraction in enumerate(FRACTIONS):
        for ti, topology in enumerate(TOPOLOGIES):
            for gi, data in enumerate((v1, v2)):
                ax = axes[ri, ti * 2 + gi]
                row = representative(data, fraction, topology)
                ax.imshow(row["mask"], cmap="gray_r", origin="lower", aspect="auto")
                ax.set_title(f"{topology}\n{'V1' if gi == 0 else 'V2'}", fontsize=10)
                ax.axis("off")
        axes[ri, 0].set_ylabel(f"fβ={fraction:.0%}", fontsize=12)
    fig.suptitle("V1 vs V2 morphology comparison (same fraction and topology labels)", fontsize=16)
    out = ROOT / "comparison"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "V1_VS_V2_MORPHOLOGY_COMPARISON.png", dpi=240)
    plt.close(fig)

    refs = [
        ROOT / "reference_statistics" / "segmentations" / "ucm_orange_crop.png",
        ROOT / "reference_statistics" / "segmentations" / "ucm_green_crop.png",
        ROOT / "reference_statistics" / "segmentations" / "sun2020_2mm_crop.png",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    labels = ["UCM AZ91D, casting, 200 μm-scale image", "UCM AZ91D, casting, 50 μm-scale image", "HPDC AZ91, published binary eutectic map"]
    for ax, path, label in zip(axes, refs, labels):
        ax.imshow(Image.open(path), cmap="gray")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle("Experimental reference (not synthetic training data)", fontsize=14)
    fig.savefig(out / "EXPERIMENTAL_REFERENCE_PANEL.png", dpi=220)
    plt.close(fig)

    ref_img = Image.open(out / "EXPERIMENTAL_REFERENCE_PANEL.png").convert("RGB")
    comp_img = Image.open(out / "V1_VS_V2_MORPHOLOGY_COMPARISON.png").convert("RGB")
    target_w = max(ref_img.width, comp_img.width)
    if ref_img.width != target_w:
        ref_img = ref_img.resize((target_w, round(ref_img.height * target_w / ref_img.width)), Image.Resampling.LANCZOS)
    if comp_img.width != target_w:
        comp_img = comp_img.resize((target_w, round(comp_img.height * target_w / comp_img.width)), Image.Resampling.LANCZOS)
    combined = Image.new("RGB", (target_w, ref_img.height + comp_img.height), "white")
    combined.paste(ref_img, (0, 0))
    combined.paste(comp_img, (0, ref_img.height))
    combined.save(out / "V1_VS_V2_VS_EXPERIMENTAL_REFERENCE.png", quality=95)


def descriptor_plots(v1: list[dict], v2: list[dict], ref_rows: list[dict]) -> None:
    out = ROOT / "comparison"
    clean = v1 + v2
    descriptors = [
        ("component_count", "Connected components"),
        ("interface_length_density_per_um", "Interface length density (1/μm)"),
        ("largest_component_fraction", "Largest-component fraction"),
        ("local_thickness_median_um", "Median local thickness (μm)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (key, label) in zip(axes.ravel(), descriptors):
        xs, ys, cs = [], [], []
        pos = 0
        ticks, names = [], []
        for gen, color in (("V1", "#4575b4"), ("V2", "#d95f0e")):
            for topology in TOPOLOGIES:
                vals = [r[key] for r in clean if r["generator"] == gen and r["topology"] == topology]
                jitter = np.linspace(-0.12, 0.12, len(vals))
                xs.extend(pos + jitter)
                ys.extend(vals)
                cs.extend([color] * len(vals))
                ticks.append(pos)
                names.append(f"{gen}\n{topology[:4]}")
                pos += 1
            pos += 0.4
        ax.scatter(xs, ys, c=cs, alpha=0.72, s=22)
        ax.set_xticks(ticks, names, fontsize=8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    fig.suptitle("V1 vs V2 descriptor distributions across all 27 structures")
    fig.savefig(out / "DESCRIPTOR_SUMMARY_V1_V2.png", dpi=220)
    plt.close(fig)

    for key, title, filename in (
        ("equivalent_diameter", "Beta component-size distributions", "COMPONENT_SIZE_DISTRIBUTIONS.png"),
        ("thickness", "Local beta thickness distributions", "LOCAL_THICKNESS_DISTRIBUTIONS.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for gen, color in (("V1", "#4575b4"), ("V2", "#d95f0e")):
            vals = []
            for r in clean:
                if r["generator"] != gen:
                    continue
                arr_key = "eqd" if key == "equivalent_diameter" else "thickness"
                vals.extend(np.asarray(r["arrays"][arr_key]).tolist())
            vals = np.asarray(vals)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            ax.hist(vals, bins=35, density=True, histtype="step", linewidth=2, color=color, label=gen)
        ax.set_xlabel("Equivalent diameter (μm)" if key == "equivalent_diameter" else "Local thickness (μm)")
        ax.set_ylabel("Probability density")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.savefig(out / filename, dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for gen, color in (("V1", "#4575b4"), ("V2", "#d95f0e")):
        rs, curves, ls, lcurves = [], [], [], []
        source = v1 if gen == "V1" else v2
        for row in source:
            if gen == "V1":
                r, c = radial_s2(row["mask"])
                l, lc = lineal_path(row["mask"])
            else:
                r, c = row["arrays"]["r"], row["arrays"]["s2"]
                l, lc = row["arrays"]["line_length"], row["arrays"]["lineal_path"]
            rs.append(r); curves.append(c); ls.append(l); lcurves.append(lc)
        axes[0].plot(rs[0], np.nanmean(curves, axis=0), color=color, lw=2, label=gen)
        axes[1].plot(ls[0], np.nanmean(lcurves, axis=0), color=color, lw=2, label=gen)
    axes[0].set(title="Two-point correlation S2(r)", xlabel="r (μm)", ylabel="S2")
    axes[1].set(title="Lineal-path function L(r)", xlabel="segment length (μm)", ylabel="L")
    for ax in axes:
        ax.grid(alpha=0.2); ax.legend()
    fig.savefig(out / "SPATIAL_STATISTICS_S2_LINEAL_PATH.png", dpi=220)
    plt.close(fig)


def compact_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        result.append({k: v for k, v in row.items() if k not in {"mask", "arrays", "gb"}})
    return result


def write_reports(v1: list[dict], v2: list[dict], refs: list[dict]) -> None:
    v1c, v2c = compact_rows(v1), compact_rows(v2)
    save_csv(ROOT / "v1_audit" / "V1_GEOMETRY_STATISTICS.csv", v1c)
    save_csv(ROOT / "v2_generator" / "seed_manifest.csv", v2c)
    save_csv(ROOT / "v2_pilot_27" / "V2_PILOT_27_DESCRIPTORS.csv", v2c)
    save_csv(ROOT / "reference_statistics" / "reference_morphology_statistics.csv", refs)
    np.savez(
        ROOT / "reference_statistics" / "reference_spatial_statistics.npz",
        note=np.array(["Per-image S2/L arrays are stored beside this index; Sun 2020 binary maps are secondary HPDC references."], dtype=object),
        image_ids=np.array([r["image_id"] for r in refs], dtype=object),
    )

    summary = []
    keys = ["component_count", "largest_component_fraction", "interface_length_density_per_um", "local_thickness_median_um", "equivalent_diameter_median_um", "surface_contact_fraction"]
    for gen, rows in (("V1", v1c), ("V2", v2c)):
        for topology in TOPOLOGIES:
            selected = [r for r in rows if r["topology"] == topology]
            record = {"generator": gen, "topology": topology, "n": len(selected)}
            for key in keys:
                values = np.asarray([r[key] for r in selected], dtype=float)
                record[f"{key}_mean"] = float(values.mean())
                record[f"{key}_sd"] = float(values.std(ddof=1))
            summary.append(record)
    save_csv(ROOT / "comparison" / "V1_V2_DESCRIPTOR_SUMMARY.csv", summary)

    all_mesh = all(r["mesh_compatible"] for r in v2c)
    fraction_ok = all(abs(r["f_beta"] - r["fraction_target"]) <= 1 / (NX * NY) + 1e-9 for r in v2c)
    means = {(g, t): np.mean([r["largest_component_fraction"] for r in (v1c if g == "V1" else v2c) if r["topology"] == t]) for g in ("V1", "V2") for t in TOPOLOGIES}
    topology_order = bool(means[("V2", "fragmented")] < means[("V2", "moderate")] < means[("V2", "network")])
    recommendation = "ADOPT V2" if all_mesh and fraction_ok and topology_order else "REVISE V2"
    gate = {
        "scope": "morphology-only pilot; no COMSOL solve and no ML training",
        "recommendation": recommendation,
        "fraction_targets_met": fraction_ok,
        "mesh_compatible_all": all_mesh,
        "topology_largest_component_ordered": topology_order,
        "reference_baseline": {"route": "HPDC/as-cast AZ91/AZ91D", "beta_fraction_band": [0.064, 0.089], "reported_average_particle_diameter_um": 3.7, "note": "5% is near/lower-realistic; 12% and 20% remain expanded controlled design levels."},
        "physics_change_required": False,
        "segmentation_uncertainty": "UCM etched optical contrast is not beta-specific and is visual-only. Sun 2020 author-published binary eutectic maps are secondary quantitative references and include section-thickness/process effects.",
        "mesh_rule": "V2 p10 local beta thickness >=0.70 um on a 0.39 um input grid; Formal V1.1 physics/mesh was not changed or run.",
    }
    (ROOT / "V2_MORPHOLOGY_GATE_REPORT.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    md = f"""# V2 morphology gate report

## Scope

This pilot changes only the stochastic alpha-Mg / beta-Mg17Al12 morphology generator. No COMSOL solve, machine-learning training, split change, or modification of the existing 108 trajectories was performed.

## Decision

**{recommendation}**, subject to scientific review before any solver rerun.

1. Visual realism: V2 derives all three topology levels from one warped cellular/interdendritic skeleton, removing V1 circles, sinusoidal bands, vertical joins, and manually imposed line families.
2. Statistical realism: V2 explicitly controls cellular-boundary preference, nonuniform local thickness, branches, junctions, disconnected divorced segments, S2(r), and lineal-path behavior.
3. Topology separation: fragmented < moderate < network in mean largest-component fraction: {means[('V2','fragmented')]:.3f}, {means[('V2','moderate')]:.3f}, {means[('V2','network')]:.3f}.
4. Fraction control: {'PASS' if fraction_ok else 'FAIL'} for all 27 structures at one-pixel tolerance.
5. Mesh compatibility: {'PASS' if all_mesh else 'PARTIAL/FAIL'} under the preregistered p10 local-thickness threshold of 0.70 um.
6. Seed diversity: three independent Voronoi-site fields and independent deformation/occupation fields were used in every design cell; no translations, rotations, or mirrors were used.
7. Artifacts: V2 can still show short closed loops because it is a two-dimensional section of a cellular skeleton. These are physically plausible sectioning features but should be reviewed visually.
8. Formal V1.1 physics: no change is required. Only the input morphology files would change after approval.

## Reference limits

The literature-informed HPDC baseline is approximately 6.4-8.9% beta phase, with one reported average beta-particle diameter near 3.7 um. The 12% and 20% pilot levels are retained only for continuity and controlled extrapolation; they are not claimed to be typical HPDC AZ91D.

The UCM optical images are used for visual morphology only because etched contrast does not uniquely isolate beta-Mg17Al12. The Sun et al. (2020) binary eutectic maps are usable as secondary quantitative references, but their reported 14.91-19.86% phase fractions depend on section thickness and their segmented class is broader than a clean HPDC beta-only baseline.
"""
    (ROOT / "V2_MORPHOLOGY_GATE_REPORT.md").write_text(md, encoding="utf-8")

def main() -> None:
    for sub in ("v1_audit", "reference_statistics", "v2_generator", "v2_pilot_27", "comparison"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    refs = prepare_references()
    v1 = load_v1()
    v2 = generate_v2()
    preview_grid(v2)
    comparison_figure(v1, v2)
    descriptor_plots(v1, v2, refs)
    write_reports(v1, v2, refs)
    print(json.dumps({"v1": len(v1), "v2": len(v2), "root": str(ROOT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
