#!/usr/bin/env python3
"""Rebuild Fig6 sampling-variability panels with shared ticks/xlim and a square left axes."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[1]
FIG6 = ROOT / "figures" / "Fig6.svg"
OUT_DIR = ROOT / "figures"
PANEL_STEM = OUT_DIR / "Fig6_variability_panels"


def local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def gid(el) -> str:
    return el.attrib.get("id", "")


def parse_path_points(d: str):
    tokens = re.findall(r"[MmLl]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    pts = []
    i = 0
    mode = None
    cx = cy = 0.0
    while i < len(tokens):
        token = tokens[i]
        if token in "MmLl":
            mode = token
            i += 1
            continue
        x = float(token)
        y = float(tokens[i + 1])
        i += 2
        if mode in ("M", "L"):
            cx, cy = x, y
        elif mode in ("m", "l"):
            cx += x
            cy += y
        else:
            cx, cy = x, y
            mode = "L"
        pts.append((cx, cy))
        if mode == "M":
            mode = "L"
        if mode == "m":
            mode = "l"
    return pts


def invert_y(svg_y, y0, y1, v0, v1):
    return v0 + (y0 - svg_y) * (v1 - v0) / (y0 - y1)


def parse_rect(d: str):
    nums = list(map(float, re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)))
    if len(nums) < 4:
        return None
    x, y, dx, dy = nums[:4]
    return x, y + dy if dy < 0 else y, abs(dx), abs(dy)


def color_to_value(hexcol: str, vmin: float, vmax: float) -> float:
    cmap = colormaps["viridis"]
    rgb = np.array(to_rgb(hexcol))
    lut = np.array([cmap(i / 255.0)[:3] for i in range(256)])
    idx = int(((lut - rgb) ** 2).sum(axis=1).argmin())
    return vmin + (vmax - vmin) * idx / 255.0


def extract_mean_curve(root, path_predicate, x_t1, step, y0_svg, y1_svg, v0, v1):
    for el in root.iter():
        if local(el.tag) != "path":
            continue
        style = el.attrib.get("style", "")
        if not path_predicate(style, el.attrib.get("d", "")):
            continue
        pts = parse_path_points(el.attrib["d"])
        if len(pts) < 2:
            continue
        timesteps = np.array([1 + round((x - x_t1) / step) for x, _ in pts], dtype=float)
        values = np.array([invert_y(y, y0_svg, y1_svg, v0, v1) for _, y in pts], dtype=float)
        return timesteps, values
    raise RuntimeError("mean curve not found")


def extract_heatmap(root, mesh_id: str, x0: float, width: float, y_bottom: float, height: float,
                    evidence_min: float, evidence_max: float, vmin: float, vmax: float):
    mesh = None
    for el in root.iter():
        if gid(el) == mesh_id:
            mesh = el
            break
    if mesh is None:
        raise RuntimeError(f"missing mesh {mesh_id}")

    xs = []
    ys = []
    fills = []
    for path in mesh:
        if local(path.tag) != "path":
            continue
        rect = parse_rect(path.attrib.get("d", ""))
        if rect is None:
            continue
        x, y, w, h = rect
        style = path.attrib.get("style", "")
        match = re.search(r"fill:([^;]+)", style)
        fill = match.group(1) if match else "none"
        xs.append(x)
        ys.append(y)
        fills.append(fill)

    xs_u = np.array(sorted({round(v, 5) for v in xs}))
    ys_u = np.array(sorted({round(v, 5) for v in ys}))
    n_t = len(xs_u)
    n_e = len(ys_u)
    values = np.full((n_t, n_e), np.nan)
    x_index = {round(v, 5): i for i, v in enumerate(xs_u)}
    y_index = {round(v, 5): i for i, v in enumerate(ys_u)}
    for x, y, fill in zip(xs, ys, fills):
        if fill == "none":
            continue
        i = x_index[round(x, 5)]
        j = y_index[round(y, 5)]
        values[i, j] = color_to_value(fill, vmin, vmax)

    # Cell centers in data coordinates.
    dx = width / n_t
    dy = height / n_e
    t_centers = 0.5 + (xs_u - x0) / dx + 0.5
    # y SVG increases downward; top of axes is evidence_max.
    e_centers = evidence_max - ((ys_u - (y_bottom - height)) / dy + 0.5) * (
        (evidence_max - evidence_min) / n_e
    )
    # Prefer exact linspace edges matching original binning intent.
    e_edges = np.linspace(evidence_min, evidence_max, n_e + 1)
    t_edges = np.linspace(0.5, 0.5 + n_t, n_t + 1)
    return values, t_edges, e_edges, t_centers, e_centers


def plot_panel(ax_left, ax_right, mean_t, mean_y, heat, t_edges, e_edges, timesteps):
    t = np.asarray(timesteps, dtype=float)
    xlim = (float(t_edges[0]), float(t_edges[-1]))

    ax_left.grid(False)
    ax_right.grid(False)
    ax_left.plot(mean_t, mean_y, "s--", color="black", lw=2, ms=5)
    ax_left.set(xlabel="Timestep", ylabel="Variability of p(sample) [std]")
    heat_mesh = ax_right.pcolormesh(t_edges, e_edges, heat.T, cmap="viridis", shading="flat")
    ax_right.set(xlabel="Timestep", ylabel="Cumulative evidence (LLR)")
    for ax in (ax_left, ax_right):
        ax.set_xlim(xlim)
        ax.set_xticks(t)
    return heat_mesh


def build_square_pair_figure(mean_t, mean_y, heat, t_edges, e_edges, timesteps, out_stem: Path):
    """Left axes is physically square; both panels share the same axes width, xlim, and ticks."""
    fig_w, fig_h = 7.4, 3.55
    panel = 2.55  # inches; left panel is panel x panel
    fig = plt.figure(figsize=(fig_w, fig_h))

    left = 0.10
    bottom = 0.18
    w = panel / fig_w
    h = panel / fig_h
    ax_left = fig.add_axes([left, bottom, w, h])
    gap = 0.11
    ax_right = fig.add_axes([left + w + gap, bottom, w, h])
    cax = fig.add_axes([left + 2 * w + gap + 0.015, bottom, 0.025, h])

    mesh = plot_panel(ax_left, ax_right, mean_t, mean_y, heat, t_edges, e_edges, timesteps)
    fig.colorbar(mesh, cax=cax).set_label("Variability of p(sample) [std]")

    # Hard-verify geometry in display coordinates.
    fig.canvas.draw()
    left_bbox = ax_left.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    right_bbox = ax_right.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    left_ratio = left_bbox.width / left_bbox.height
    assert abs(left_ratio - 1.0) < 1e-6, f"left axes not square: {left_ratio}"
    assert abs(left_bbox.width - right_bbox.width) < 1e-6, "panel widths differ"
    assert ax_left.get_xlim() == ax_right.get_xlim()
    assert np.allclose(ax_left.get_xticks(), ax_right.get_xticks())

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=500, facecolor="white")
    fig.savefig(out_stem.with_suffix(".svg"), dpi=500, facecolor="white")
    plt.close(fig)
    return left_bbox, right_bbox


def build_fig6(top, bottom, out_path: Path):
    """Stack the two monkey panels into a single portrait figure."""
    fig_w, fig_h = 7.4, 7.2
    panel = 2.55
    fig = plt.figure(figsize=(fig_w, fig_h))
    left = 0.10
    w = panel / fig_w
    h = panel / fig_h
    gap_x = 0.11
    gap_y = 0.10
    bottom_y = 0.08
    top_y = bottom_y + h + gap_y

    for y0, panel_data in ((top_y, top), (bottom_y, bottom)):
        mean_t, mean_y, heat, t_edges, e_edges, timesteps = panel_data
        ax_left = fig.add_axes([left, y0, w, h])
        ax_right = fig.add_axes([left + w + gap_x, y0, w, h])
        cax = fig.add_axes([left + 2 * w + gap_x + 0.015, y0, 0.025, h])
        mesh = plot_panel(ax_left, ax_right, mean_t, mean_y, heat, t_edges, e_edges, timesteps)
        fig.colorbar(mesh, cax=cax).set_label("Variability of p(sample) [std]")

    fig.canvas.draw()
    # Check the first (top) left axes.
    ax0 = fig.axes[0]
    bbox = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    assert abs(bbox.width / bbox.height - 1.0) < 1e-6, bbox

    out_path = Path(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=500, facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), dpi=500, facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    root = ET.parse(FIG6).getroot()

    top_t, top_y = extract_mean_curve(
        root,
        lambda style, d: (
            "stroke-dasharray" in style
            and "stroke:#000000" in style
            and "stroke-width:2.2" in style
            and "h 8" not in d
            and d.strip().startswith("M 49.775537")
        ),
        x_t1=49.775537,
        step=13.84494,
        y0_svg=135.23254,
        y1_svg=34.89983,
        v0=0.0,
        v1=0.08,
    )
    bot_t, bot_y = extract_mean_curve(
        root,
        lambda style, d: (
            "stroke-dasharray" in style
            and "stroke:#000000" in style
            and "stroke-width:2.2" in style
            and "h 8" not in d
            and d.strip().startswith("m 43.816926")
        ),
        x_t1=43.816926,
        step=14.581376,
        y0_svg=135.23254,
        y1_svg=92.365257,
        v0=0.0,
        v1=0.1,
    )

    # Discover bottom QuadMesh id.
    mesh_ids = []
    for el in root.iter():
        if gid(el).startswith("QuadMesh"):
            mesh_ids.append(gid(el))
    print("meshes", mesh_ids)

    top_heat, top_te, top_ee, _, _ = extract_heatmap(
        root,
        mesh_ids[0],
        x0=265.84531,
        width=402.91025 - 265.84531,
        y_bottom=141.15891,
        height=141.15891 - 10.778906,
        evidence_min=-3.0,
        evidence_max=3.0,
        vmin=0.0,
        vmax=0.08,
    )
    bot_heat, bot_te, bot_ee, _, _ = extract_heatmap(
        root,
        mesh_ids[1],
        x0=260.28436,
        width=137.064938,
        y_bottom=141.15891,
        height=141.15891 - 10.778906,
        evidence_min=-2.0,
        evidence_max=2.0,
        vmin=0.0,
        vmax=0.25,
    )

    timesteps = list(range(1, 11))
    # Pad bottom mean to length 10 with nan if needed for consistent xticks only;
    # plot uses the extracted points directly.
    build_square_pair_figure(top_t, top_y, top_heat, top_te, top_ee, timesteps, PANEL_STEM.with_name("Fig6_panel_top"))
    build_square_pair_figure(bot_t, bot_y, bot_heat, bot_te, bot_ee, timesteps, PANEL_STEM.with_name("Fig6_panel_bottom"))
    build_fig6(
        (top_t, top_y, top_heat, top_te, top_ee, timesteps),
        (bot_t, bot_y, bot_heat, bot_te, bot_ee, timesteps),
        OUT_DIR / "Fig6",
    )
    print("wrote", OUT_DIR / "Fig6.png", "and", OUT_DIR / "Fig6.svg")


if __name__ == "__main__":
    main()
