#!/usr/bin/env python3
"""Render matplotlib marker symbols and save each as a PNG with transparent background."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STIM_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]

# Filenames safe on all platforms (avoid raw "*" etc.)
MARKER_TO_STEM = {
    "o": "marker_o",
    "s": "marker_s",
    "^": "marker_caret_up",
    "D": "marker_diamond_thin",
    "P": "marker_plus_filled",
    "X": "marker_x_filled",
    "v": "marker_caret_down",
    "*": "marker_star",
}


def save_marker_png(
    marker: str,
    out_path: Path,
    *,
    color: str = "black",
    size: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(1.2, 1.2), dpi=dpi)
    ax.scatter(
        [0],
        [0],
        marker=marker,
        s=size,
        c=color,
        edgecolors=color,
        linewidths=0.8,
        zorder=3,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "marker_shape_pngs",
        help="Directory to write PNGs (created if missing).",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=400.0,
        help="Matplotlib scatter s=... (marker area in points^2).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for m in STIM_MARKERS:
        stem = MARKER_TO_STEM.get(m, f"marker_{ord(m):#x}")
        path = outdir / f"{stem}.png"
        save_marker_png(
            m,
            path,
            color="black" if m == "*" else "black",
            size=args.marker_size,
            dpi=args.dpi,
        )
        written.append(path)

    print(f"Wrote {len(written)} PNGs to {outdir.resolve()}")
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
