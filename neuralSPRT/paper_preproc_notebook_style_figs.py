from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings


def load_mat73(file_path: Path) -> dict:
    data = {}
    with h5py.File(file_path, "r") as f:
        def load_group(group):
            result = {}
            for key in group.keys():
                if key != "#refs#":
                    obj = group[key]
                    if isinstance(obj, h5py.Dataset):
                        result[key] = np.array(obj)
                    elif isinstance(obj, h5py.Group):
                        result[key] = load_group(obj)
            return result

        for key in f.keys():
            if key != "#refs#":
                if isinstance(f[key], h5py.Group):
                    data[key] = load_group(f[key])
                else:
                    data[key] = np.array(f[key])
    return data


def apply_notebook_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.loc": "upper right",
    })


def preprocess(TM: np.ndarray, monkey_id: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Paper-consistent behavioral preprocessing based on neuralSPRT_BEH.m:
    - remove trump-shape trials
    - compute N* (num_accum) using non-decision-time rule
    - keep only accumulated shapes for decision evidence
    - compute cumulative evidence at decision (cum_loglr)
    """
    TM = np.asarray(TM, dtype=float)

    # ---- monkey parameters
    if monkey_id == 1:
        lastAccumShape2Sac = 270.0
        stim_int = 250.0
    elif monkey_id == 2:
        lastAccumShape2Sac = 180.0
        stim_int = 270.0
    else:
        raise ValueError("monkey_id must be 1 or 2")

    # Remove trials with trump shapes (monkey J)
    # MATLAB neuralSPRT_BEH uses TM(:,11:30)+1 (1-indexed) -> Python [10:30]+1
    shapeMtrx = TM[:, 10:30].copy() + 1
    pick_trump = np.sum((shapeMtrx == 1) | (shapeMtrx == 2), axis=1).astype(bool)
    kept = ~pick_trump
    trial_index = np.nonzero(kept)[0]
    shapeMtrx = shapeMtrx[kept, :]
    TM = TM[kept, :]

    # True assigned weights (logLR) * 10
    trueDeciWOE = np.array([9999, -9999, 9, -9, 7, -7, 5, -5, 3, -3, np.nan], dtype=float)

    # number of shapes shown
    num_shown = np.sum(np.isfinite(shapeMtrx), axis=1).astype(int)

    # reward-assigned target (1:left, 2:right) initially
    rew_targ = TM[:, 0].astype(int)

    # reward-assigned color (Joey only): (rew_targ ~= TM(:,9)) + 1
    rew_color = None
    if monkey_id == 2 and TM.shape[1] > 8:
        rew_color = ((rew_targ != TM[:, 8].astype(int)).astype(int) + 1)

    # choice recoding
    if monkey_id == 1:
        # choice = 3 - TM(:,2); rew_targ = 3 - rew_targ; sign flip later
        choice = (3 - TM[:, 1].astype(int))
        rew_targ = (3 - rew_targ)
    else:
        if TM.shape[1] <= 9:
            raise ValueError("TM must have at least 10 columns for monkey_id=2 choice recode.")
        choice = TM[:, 9].astype(int)  # TM(:,10) -> col 9

    RT = TM[:, 2].copy()          # TM(:,3)
    lastStim2Sac = TM[:, 3].copy()  # TM(:,4)

    correct = (TM[:, 0].astype(int) == TM[:, 1].astype(int)).astype(int)

    # Compute N* (num_accum)
    num_accum = num_shown.copy()
    t_cutoff = lastAccumShape2Sac * np.ones_like(num_shown, dtype=float)
    t_cutoff = t_cutoff - lastStim2Sac  # - sac_latency

    pick = t_cutoff > 0
    num_accum[pick] = num_accum[pick] - 1
    while np.any(t_cutoff > 0):
        t_cutoff = t_cutoff - stim_int
        pick = t_cutoff > 0
        num_accum[pick] = num_accum[pick] - 1
    num_accum = np.maximum(num_accum, 0)

    # truncate shapeMtrx/locMtrx after N*
    for i in range(TM.shape[0]):
        # MATLAB: shapeMtrx(i, num_accum(i)+1:num_shown(i)) = nan;
        start = int(num_accum[i])
        end = int(num_shown[i])
        if end > start:
            shapeMtrx[i, start:end] = np.nan

    # IMPORTANT: sign convention flip for monkey E
    shape_idx = shapeMtrx.copy()
    shape_idx[~np.isfinite(shape_idx)] = 11
    shape_idx = shape_idx.astype(int)
    llr10 = trueDeciWOE[shape_idx - 1]  # still *10
    if monkey_id == 1:
        llr10 = -llr10

    cum_llr10 = np.cumsum(llr10, axis=1)
    trial_row = np.arange(TM.shape[0])
    end_idx = np.maximum(num_accum - 1, 0)
    cum_loglr = np.where(num_accum > 0, cum_llr10[trial_row, end_idx] / 10.0, 0.0)
    cum_loglr = cum_loglr.astype(float)
    cum_loglr[~np.isfinite(cum_loglr)] = np.nan

    if monkey_id == 1:
        reward = rew_targ
    else:
        reward = rew_color if rew_color is not None else rew_targ
        if rew_color is None:
            warnings.warn("TM missing column 9; using rew_targ as reward proxy for monkey J.", RuntimeWarning)

    df = pd.DataFrame({
        "trial_index": trial_index,
        "choice": choice.astype(int),
        "correct_choice": correct.astype(int),
        "reward": np.asarray(reward, dtype=int),
        "RT": RT.astype(float),
        "lastStim2Sac": lastStim2Sac.astype(float),
        "nums_shape": num_accum.astype(int),
        "num_shown": num_shown.astype(int),
        "cum_loglr": cum_loglr,
    })

    # Return accumulated shapes matrix (after N* truncation) for any downstream use.
    shape_accum = shapeMtrx
    return df, shape_accum



def plot_cumloglr_vs_num_shapes(
    df: pd.DataFrame,
    monkey_id: int,
    out_dir: Path,
) -> None:

    work = df.copy()
    work = work[(work["correct_choice"] == 1) & np.isfinite(work["cum_loglr"])].copy()

    x_col = "nums_shape"
    y_col = "cum_loglr"
    saccade_col = "choice"

    agg = (
        work.groupby([saccade_col, x_col])[y_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    if monkey_id == 1:
        agg = agg[(agg["count"] > 5) & (agg["nums_shape"] >= 1) & (agg["nums_shape"] <= 10)].copy()
        # agg = agg[(agg["nums_shape"] <= 10)].copy()
        colors = {1: "red", 2: "blue"}
        labels = {1: "Choice 1", 2: "Choice 2"}

        pos_bar_color = "blue"
        neg_bar_color = "red"
        xlim = (1.5, 10.5)
        ylim = (-3.0, 3.0)
    else:
        agg = agg[(agg["count"] > 5) & (agg["nums_shape"] >= 1) & (agg["nums_shape"] <= 8)].copy()
        # agg = agg[(agg["nums_shape"] <= 10)].copy()
        colors = {1: "red", 2: "blue"}
        labels = {1: "Choice 1", 2: "Choice 2"}
        pos_bar_color = "red"
        neg_bar_color = "green"
        xlim = (0, 8.5)
        ylim = (-2.1, 2.1)

    freq_pos = work[work[y_col] > 0].groupby(x_col, observed=True).size().reindex(range(1, 15)).fillna(0).astype(int)
    freq_neg = work[work[y_col] < 0].groupby(x_col, observed=True).size().reindex(range(1, 15)).fillna(0).astype(int)
    max_freq = max(freq_pos.max(), freq_neg.max(), 1)
    y_range = max(agg["mean"].max() - agg["mean"].min(), 0.5) if len(agg) else 1.0
    scale = 0.7 * y_range / max_freq

    plt.figure(figsize=(3.85, 2.5))
    for direction in [1, 2]:
        sub = agg[agg[saccade_col] == direction].sort_values(x_col)
        if len(sub) == 0:
            continue
        plt.plot(sub[x_col], sub["mean"], marker="o", label=labels[direction], color=colors[direction])
        plt.errorbar(
            sub[x_col],
            sub["mean"],
            yerr=sub["std"],
            fmt="none",
            capsize=5,
            color=colors[direction],
        )

    plt.bar(freq_pos.index, freq_pos.values * scale, width=0.8, bottom=0, color=pos_bar_color, alpha=0.25)
    plt.bar(freq_neg.index, -freq_neg.values * scale, width=0.8, bottom=0, color=neg_bar_color, alpha=0.25)
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("Number of shapes")
    plt.ylabel("Cumulative logLR")
    plt.legend(title="Monkey saccade")
    plt.tight_layout()

    monkey = "E" if monkey_id == 1 else "J"
    plt.savefig(out_dir / f"cumloglr_vs_numshapes_monkey{monkey}.png", dpi=300)
    plt.savefig(out_dir / f"cumloglr_vs_numshapes_monkey{monkey}.svg")
    plt.close()


def plot_num_shapes_frequency_by_choice(
    df: pd.DataFrame,
    monkey_id: int,
    out_dir: Path,
    *,
    max_timestep: int = 15,
    correct_only: bool = True,
    figsize: tuple[float, float] = (2.85, 2.5),
) -> pd.DataFrame:
    """
    Plot frequency distribution of number of shapes (N*) for the two choices.

    Uses df columns:
      - nums_shape: N*
      - choice: {1,2}
      - correct_choice (optional): used if correct_only=True

    Returns a tidy dataframe with columns: choice, nums_shape, count.
    """
    work = df.copy()
    if correct_only and "correct_choice" in work.columns:
        work = work[work["correct_choice"] == 1].copy()

    work = work[np.isfinite(work["nums_shape"]) & np.isfinite(work["choice"])].copy()
    work["nums_shape"] = work["nums_shape"].astype(int)
    work["choice"] = work["choice"].astype(int)
    work = work[(work["nums_shape"] >= 1) & (work["nums_shape"] <= max_timestep)].copy()
    if work.empty:
        return pd.DataFrame(columns=["choice", "nums_shape", "count"])

    counts = (
        work.groupby(["choice", "nums_shape"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )

    x = np.arange(1, max_timestep + 1, dtype=int)
    c1 = counts[counts["choice"] == 1].set_index("nums_shape")["count"].reindex(x).fillna(0).to_numpy()
    c2 = counts[counts["choice"] == 2].set_index("nums_shape")["count"].reindex(x).fillna(0).to_numpy()

    if monkey_id == 1:
        colors = {1: "red", 2: "blue"}
        labels = {1: "Choice 1", 2: "Choice 2"}
        xlim = (0.5, max_timestep + 0.5)
    else:
        colors = {1: "red", 2: "blue"}
        labels = {1: "Choice 1", 2: "Choice 2"}
        xlim = (0.5, max_timestep + 0.5)

    plt.figure(figsize=figsize)
    plt.bar(x, c1, width=0.8, bottom=0, color=colors[1], alpha=0.35, label=labels[1])
    plt.bar(x, -c2, width=0.8, bottom=0, color=colors[2], alpha=0.35, label=labels[2])
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.xlim(*xlim)
    plt.xlabel("Number of shapes (N*)")
    plt.ylabel("Trial count")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()

    monkey = "E" if monkey_id == 1 else "J"
    suffix = "correct_only" if correct_only else "all_trials"
    plt.savefig(out_dir / f"numshapes_frequency_by_choice_monkey{monkey}_{suffix}.png", dpi=300)
    plt.savefig(out_dir / f"numshapes_frequency_by_choice_monkey{monkey}_{suffix}.svg")
    plt.close()

    return counts


def plot_decision_threshold_distribution(
    df: pd.DataFrame,
    shape_accum: np.ndarray,
    monkey_id: int,
    out_dir: Path,
    max_timestep: int = 10,
    figsize: tuple[float, float] = (5, 2.5),
) -> None:

    work = df.dropna(subset=["nums_shape", "cum_loglr", "choice", "correct_choice"]).copy()
    work = work[(work["correct_choice"] == 1) & (work["nums_shape"] >= 1) & (work["nums_shape"] <= max_timestep)].copy()
    if work.empty:
        return

    row_idx = work.index.to_numpy()
    work = work.reset_index(drop=True)

    shape_vals = shape_accum[row_idx, :]
    valid_mask = np.isfinite(shape_vals)
    logLR_table = np.array([np.inf, -np.inf, 0.9, -0.9, 0.7, -0.7, 0.5, -0.5, 0.3, -0.3])

    def partial_cumloglr(i: int, t: int) -> float:
        ids = shape_vals[i, valid_mask[i]].astype(int)[:t]
        # Safety: keep only known shape IDs (1..10) for lookup.
        ids = ids[(ids >= 1) & (ids <= 10)]
        if ids.size == 0:
            return np.nan
        return float(np.sum(logLR_table[ids - 1]))

    rows = []
    for i in range(len(work)):
        n = int(work.iloc[i]["nums_shape"])
        for t in range(1, min(n, max_timestep) + 1):
            if t < n:
                action_type = "sampling"
                ev = partial_cumloglr(i, t)
            else:
                action_type = int(work.iloc[i]["choice"])
                ev = float(work.iloc[i]["cum_loglr"])
            if np.isfinite(ev):
                rows.append({"t": t, "action": action_type, "evidence": ev})

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    if monkey_id == 1:
        y_lim = (-4.0, 4.0) 
    else:
        y_lim = (-2.5, 2.5)

    fig, axes = plt.subplots(1, max_timestep, figsize=figsize, sharey=True)
    if max_timestep == 1:
        axes = [axes]

    # blue = chosen A, red = chosen B, green = sampling
    colors = {"sampling": "green", 1.0: "blue", 2.0: "red", 1: "blue", 2: "red"}
    labels = {"sampling": "Sampling", 1.0: "Chosen A", 2.0: "Chosen B", 1: "Chosen A", 2: "Chosen B"}
    # Build a stable legend that doesn't depend on which actions appear at t=1.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors[1], edgecolor="none", alpha=0.3, label=labels[1]),
        Patch(facecolor=colors[2], edgecolor="none", alpha=0.3, label=labels[2]),
        Patch(facecolor=colors["sampling"], edgecolor="none", alpha=0.3, label=labels["sampling"]),
    ]
    ev_grid = np.arange(y_lim[0], y_lim[1] + 1e-9, 0.2)

    def _smooth_counts(counts: np.ndarray) -> np.ndarray:
        # Smooth along evidence axis within each timestep.
        kernel = np.array([1, 4, 6, 4, 1], dtype=float)
        kernel = kernel / kernel.sum()
        if counts.size < kernel.size:
            return counts.astype(float)
        return np.convolve(counts.astype(float), kernel, mode="same")

    for t in range(max_timestep):
        ax = axes[t]
        timestep_data = plot_df[plot_df["t"] == t + 1]

        for action in ["sampling", 1.0, 2.0]:
            if action == "sampling":
                action_data = timestep_data[timestep_data["action"] == "sampling"]
            else:
                action_data = timestep_data[timestep_data["action"].isin([action, int(action)])]
            if action_data.empty:
                continue
            # Count threshold: only plot when enough observations exist.
            if len(action_data) <= 5:
                continue

            ev = action_data["evidence"].to_numpy(dtype=float)
            # Bin onto the canonical evidence grid (0.2 steps), then smooth.
            bin_idx = np.round((ev - ev_grid[0]) / 0.2).astype(int)
            good = (bin_idx >= 0) & (bin_idx < ev_grid.size)
            if not np.any(good):
                continue
            counts = np.zeros(ev_grid.size, dtype=float)
            np.add.at(counts, bin_idx[good], 1.0)
            counts = _smooth_counts(counts)
            if counts.max() <= 0:
                continue

            ax.fill_betweenx(
                ev_grid, 0, counts,
                color=colors[action], alpha=0.3,
                label=None,
            )
            if action != "sampling":
                mean_ev = action_data["evidence"].mean()
                ax.axhline(mean_ev, color=colors[action], linewidth=2)

        ax.set_xlabel(str(t + 1))
        ax.set_xlim(0, None)
        ax.set_ylim(y_lim)
        ax.set_xticks([])

    # Match OriginalSPRT text positions/sizes.
    fig.text(0.54, 0.03, "Timestep", ha="center", fontsize=10)
    fig.text(0.1, 0.5, "Cumulative Evidence", va="center", rotation="vertical", fontsize=10)
    for ax in axes:
        ax.label_outer()

    axes[0].legend(handles=legend_handles, loc="best")
    plt.tight_layout()

    monkey = "E" if monkey_id == 1 else "J"
    plt.savefig(out_dir / f"decision_threshold_distribution_monkey{monkey}.png", dpi=300)
    plt.savefig(out_dir / f"decision_threshold_distribution_monkey{monkey}.svg")
    plt.close()





def run_for_monkey(monkey_id: int, out_dir: Path) -> None:
    monkey = "E" if monkey_id == 1 else "J"
    mat_path = Path("neuralSPRT/data") / f"data{monkey}.mat"
    raw = load_mat73(mat_path)
    TM = np.array(raw["info"]["TM"]).T

    df, shape_accum = preprocess(TM, monkey_id)
    plot_num_shapes_frequency_by_choice(df, monkey_id, out_dir, max_timestep=10)
    plot_cumloglr_vs_num_shapes(df, monkey_id, out_dir)
    plot_decision_threshold_distribution(df, shape_accum, monkey_id, out_dir, max_timestep=10)
    stat_evidence_vs_timestep(df, shape_accum, monkey_id, out_dir, max_timestep=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monkey", choices=["E", "J", "both"], default="both")
    parser.add_argument("--outdir", default="runs/notebook_style_paper_preproc")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_notebook_style()

    if args.monkey in ("E", "both"):
        run_for_monkey(1, out_dir)
    if args.monkey in ("J", "both"):
        run_for_monkey(2, out_dir)

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
