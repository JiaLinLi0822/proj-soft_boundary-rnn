from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.stats as st
import statsmodels.api as sm
from typing import Dict, Any, List, Union, Optional


# -----------------------------
# Helpers (MATLAB equivalents)
# -----------------------------
def nanse(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = np.sum(np.isfinite(x), axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    return sd / np.sqrt(np.maximum(n, 1))

def stderr(x: np.ndarray) -> float:
    return float(nanse(np.asarray(x, dtype=float)))

def histc(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """MATLAB histc(x, edges) analogue for monotonic edges."""
    x = np.asarray(x)
    edges = np.asarray(edges)
    counts = np.zeros(edges.size, dtype=int)
    valid = np.isfinite(x) & (x >= edges[0]) & (x <= edges[-1])
    if not np.any(valid):
        return counts
    idx = np.searchsorted(edges, x[valid], side="right") - 1
    idx[x[valid] == edges[-1]] = edges.size - 1
    np.add.at(counts, idx, 1)
    return counts

def fillTrace(x: np.ndarray, y: np.ndarray, yerr: np.ndarray, color, alpha: float = 0.25, ax=None):
    """Filled mean±err band."""
    if ax is None:
        ax = plt.gca()
    x = np.asarray(x)
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha, linewidth=0)

def ploterr(x, y, yerr=None, xerr=None, ax=None, fmt="o", **kwargs):
    if ax is None:
        ax = plt.gca()
    return ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=fmt, **kwargs)

def fillRect(xy0, xy1, color, ax=None, alpha=1.0, lw=0):
    """MATLAB-ish fillRect([x0,y0],[x1,y1],color)."""
    if ax is None:
        ax = plt.gca()
    x0, y0 = xy0
    x1, y1 = xy1
    rect = Rectangle((x0, y0), width=(x1 - x0), height=(y1 - y0),
                     facecolor=color, edgecolor=color, alpha=alpha, linewidth=lw)
    ax.add_patch(rect)
    return rect

def glmfit_normal(X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None):
    """
    MATLAB glmfit(...,'normal','weights',...) equivalent.
    Returns params, bse, pvalues, fitted_result.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_ = sm.add_constant(X, has_constant="add")
    if weights is None:
        model = sm.GLM(y, X_, family=sm.families.Gaussian())
        res = model.fit()
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        model = sm.GLM(y, X_, family=sm.families.Gaussian(), var_weights=w)
        res = model.fit()
    return res.params, res.bse, res.pvalues, res

def ttest2(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    return st.ttest_ind(a, b, equal_var=False)

def ttest1(a: np.ndarray):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return st.ttest_1samp(a, popmean=0.0)


# -----------------------------
# Main function
# -----------------------------
def neuralSPRT_PHY(info: Dict[str, Any], monkey_id: int, fig_switch: Union[int, List[int]]) -> Dict[str, Any]:
    """
    Python translation of MATLAB:
        function info = neuralSPRT_PHY(info,id,fig_switch)

    Required in info:
      - "TM": trial matrix
      - "KM": neural matrix (trial x time) (FRgain-scaled ints in original code)
      - "FRgain": scalar (e.g., 100)
      - "fig": starting figure index (int)

    Optional:
      - "popID": vector length n_trials
      - "num_cell": if exists, triggers popID filtering
      - "delay": used in Fig 4A (default 0 if missing)
    """

    # fig_switch -> boolean array indices 1..6 (MATLAB uses 1..4 here)
    if isinstance(fig_switch, int):
        fig_temp = [fig_switch]
    else:
        fig_temp = list(fig_switch)
    fig_flags = np.zeros(7, dtype=int)
    for k in fig_temp:
        fig_flags[k] = 1

    # monkey params
    if monkey_id == 1:  # monkey E
        Pick_RF_flag = 0
        t_div = 200.0
        lastAccumShape2Sac = 270.0
        stim_int = 250.0
    elif monkey_id == 2:  # monkey J
        Pick_RF_flag = 1
        t_div = 130.0
        lastAccumShape2Sac = 180.0
        stim_int = 270.0
    else:
        raise ValueError("monkey_id must be 1 or 2")

    total_shape = 8
    shape_offset = 2

    fig = int(info.get("fig", 1))

    TM = np.asarray(info["TM"], dtype=float).copy()
    KM = np.asarray(info["KM"], dtype=float).copy()
    FRgain = float(info["FRgain"])
    popID = info.get("popID", None)

    # -----------------------------
    # Joey filtering: RF + trump
    # -----------------------------
    if monkey_id == 2:
        pick = np.zeros(TM.shape[0], dtype=bool)

        if Pick_RF_flag:
            # red in RF trials only
            # MATLAB: pick = logical(TM(:,9)==1) | pick;
            pick = pick | (TM[:, 8].astype(int) == 1)

        # trump removal needs shapeMtrx
        shapeMtrx_tmp = TM[:, 10:30] + 1  # 11:30 -> 10:30
        no_trump = ~np.sum((shapeMtrx_tmp == 1) | (shapeMtrx_tmp == 2), axis=1).astype(bool)
        pick = pick & no_trump

        TM = TM[pick, :]
        KM = KM[pick, :]
        if ("num_cell" in info) and (popID is not None):
            popID = np.asarray(popID)[pick]

    # -----------------------------
    # Core variables
    # -----------------------------
    trueDeciWOE = np.array([9999, -9999, 9, -9, 7, -7, 5, -5, 3, -3, np.nan], dtype=float)
    shapeMtrx = TM[:, 10:30] + 1
    frMtrx1 = TM[:, 30:50].copy()  # MATLAB 31:50 -> Python 30:50
    num_shown = np.sum(np.isfinite(shapeMtrx), axis=1).astype(int)

    rew_targ = TM[:, 0].astype(int)
    if monkey_id == 2:
        # rew_color = (rew_targ ~= TM(:,9)) + 1, but TM(:,9) is col 8
        rew_color = ((rew_targ != TM[:, 8].astype(int)).astype(int) + 1)

    choice = TM[:, 1].astype(int)
    RT = TM[:, 2].astype(float)
    correct = (TM[:, 0].astype(int) == TM[:, 1].astype(int)).astype(int)
    T_in_out = TM[:, 7].astype(int)  # Tin/Tout in col 8 -> index 7

    # raw LLR and raw FR
    rawShapeMtrx = shapeMtrx.copy()
    rawShapeMtrx[~np.isfinite(rawShapeMtrx)] = 11
    rawShapeMtrx = rawShapeMtrx.astype(int)
    rawLLR1 = trueDeciWOE[rawShapeMtrx - 1]
    rawFrMtrx1 = frMtrx1.copy()

    # Random order (fixed seed like MATLAB rng(1))
    # MATLAB rng(1); randperm(...) behavior is closer to RandomState MT19937.
    rand_trial_order = np.random.RandomState(1).permutation(TM.shape[0])

    # -----------------------------
    # Compute N* (num_accum)
    # -----------------------------
    num_accum = num_shown.copy()
    t_cutoff = lastAccumShape2Sac * np.ones_like(num_shown, dtype=float)
    t_cutoff = t_cutoff - TM[:, 3]  # sac_latency is col 4 -> index 3

    pick = t_cutoff > 0
    num_accum[pick] -= 1
    while np.any(t_cutoff > 0):
        t_cutoff -= stim_int
        pick = t_cutoff > 0
        num_accum[pick] -= 1

    # apply truncation beyond N*
    for i in range(TM.shape[0]):
        if num_accum[i] < 0:
            num_accum[i] = 0
        start = num_accum[i]
        end = num_shown[i]
        if end > start:
            shapeMtrx[i, start:end] = np.nan
            frMtrx1[i, start:end] = np.nan

    # LLR for accumulated shapes
    shapeMtrx2 = shapeMtrx.copy()
    shapeMtrx2[~np.isfinite(shapeMtrx2)] = 11
    shapeMtrx2 = shapeMtrx2.astype(int)
    LLR1 = trueDeciWOE[shapeMtrx2 - 1]

    # delta FR matrices (diff along epochs; MATLAB uses TM(:,51) as baseline)
    # MATLAB: temp=[TM(:,51), frMtrx{1}]; deltaFrMtrx{1}=diff(temp,1,2)
    baseline_fr = TM[:, 50].reshape(-1, 1)  # col 51 -> index 50
    temp = np.concatenate([baseline_fr, frMtrx1], axis=1)
    deltaFrMtrx1 = np.diff(temp, axis=1)

    temp_raw = np.concatenate([baseline_fr, rawFrMtrx1], axis=1)
    rawDeltaFrMtrx1 = np.diff(temp_raw, axis=1)

    # -----------------------------
    # Sign convention: positive WOE favors Tin
    # -----------------------------
    if monkey_id == 1:
        # pick = TM(:,7)==1 (RF side)
        pick_signflip = (TM[:, 6].astype(int) == 1)
    else:
        # pick = TM(:,9)==2 (Target color config)
        pick_signflip = (TM[:, 8].astype(int) == 2)

    LLR1[pick_signflip, :] *= -1
    rawLLR1[pick_signflip, :] *= -1

    cumLLR1 = np.cumsum(LLR1, axis=1)
    rawCumLLR1 = np.cumsum(rawLLR1, axis=1)

    # -----------------------------
    # Backward-aligned versions (index 2 in MATLAB cells)
    # -----------------------------
    def backward_align(M: np.ndarray, n_used: np.ndarray, n_shown: np.ndarray):
        M2 = np.full_like(M, np.nan, dtype=float)
        for ci in range(1, 21):
            num_shift = 20 - ci
            pick = (n_used == ci)
            if np.any(pick):
                M2[pick, :] = np.fliplr(np.roll(M[pick, :], shift=num_shift, axis=1))
        return M2

    LLR2 = backward_align(LLR1, num_accum, num_shown)
    cumLLR2 = backward_align(cumLLR1, num_accum, num_shown)
    frMtrx2 = backward_align(frMtrx1, num_accum, num_shown)
    deltaFrMtrx2 = backward_align(deltaFrMtrx1, num_accum, num_shown)

    rawLLR2 = backward_align(rawLLR1, num_shown, num_shown)
    rawCumLLR2 = backward_align(rawCumLLR1, num_shown, num_shown)
    rawFrMtrx2 = backward_align(rawFrMtrx1, num_shown, num_shown)
    rawDeltaFrMtrx2 = backward_align(rawDeltaFrMtrx1, num_shown, num_shown)

    # store core outputs (like MATLAB's cell arrays)
    info["LLR"] = {1: LLR1, 2: LLR2}
    info["cumLLR"] = {1: cumLLR1, 2: cumLLR2}
    info["rawLLR"] = {1: rawLLR1, 2: rawLLR2}
    info["rawCumLLR"] = {1: rawCumLLR1, 2: rawCumLLR2}
    info["frMtrx"] = {1: frMtrx1, 2: frMtrx2}
    info["deltaFrMtrx"] = {1: deltaFrMtrx1, 2: deltaFrMtrx2}
    info["rawFrMtrx"] = {1: rawFrMtrx1, 2: rawFrMtrx2}
    info["rawDeltaFrMtrx"] = {1: rawDeltaFrMtrx1, 2: rawDeltaFrMtrx2}
    info["num_accum"] = num_accum
    info["num_shown"] = num_shown
    info["RT"] = RT
    info["correct"] = correct
    info["T_in_out"] = T_in_out
    if popID is not None:
        info["popID"] = popID

    # ============================================================
    # Fig 3A / 3B: decision formation
    # ============================================================
    if fig_flags[1] or fig_flags[2]:
        t_axis = np.arange(0, 5000, 1)  # 5000 pts to match KM columns

        # shuffle trials to avoid baseline FR confounds
        KM_shuffled = KM[rand_trial_order, :]
        woeMtrx_shuffled = LLR1[rand_trial_order, :]
        rawWoeMtrx_shuffled = rawLLR1[rand_trial_order, :]
        cumWoeMtrx_shuffled = cumLLR1[rand_trial_order, :]
        num_accum_shuffled = num_accum[rand_trial_order]
        RT_shuffled = RT[rand_trial_order]

        firstShapeKM = KM_shuffled[:, :5000] / FRgain  # FR in sp/s

        uniqueWoe = np.array([-9, -7, -5, -3, 3, 5, 7, 9], dtype=int)

        t_start = 100.0
        t_end = stim_int
        t_gap = 20
        num_epoch = 18

        # Keep MATLAB precedence:
        # if fig_switch(1) sortBy='cumLLR'; if fig_switch(2) sortBy='LLR';
        if fig_flags[1]:
            sortBy = "cumLLR"
        if fig_flags[2]:
            sortBy = "LLR"

        # containers for Fig 3B aggregation
        allEpochDeltaFR = {w: [] for w in uniqueWoe}

        # plotting setup
        plt.figure(fig); plt.clf()
        plt.figure(fig + 1); plt.clf()
        plt.figure(fig + 2); plt.clf()

        for ei in range(1, num_epoch + 1):
            if sortBy == "cumLLR":
                group = 5
                cmap = plt.get_cmap("jet")
                color_map = [cmap(i / (group - 1)) for i in range(group)]

                # pick trials with finite cumWoe at epoch ei and N* >= ei+1
                pick = np.isfinite(cumWoeMtrx_shuffled[:, ei - 1]) & (num_accum_shuffled >= (ei + 1))
                if np.sum(pick) == 0:
                    continue

                sorted_cumWoe = cumWoeMtrx_shuffled[pick, ei - 1]
                sort_order = np.argsort(sorted_cumWoe)
                sorted_cumWoe = sorted_cumWoe[sort_order]
                sortedFirstShape = firstShapeKM[pick, :][sort_order, :]

                num_per_group = int(np.floor(len(sorted_cumWoe) / group))
                if num_per_group < 1:
                    continue

                # time windows
                if ei == 1:
                    plot_pick = (t_axis <= (t_div + stim_int))
                else:
                    plot_pick = (t_axis >= (stim_int * (ei - 1) + t_div + t_gap)) & (t_axis <= (stim_int * ei + t_div))
                t_pick = (t_axis >= (stim_int * (ei - 1) + t_div + t_start)) & (t_axis <= (stim_int * (ei - 1) + t_div + t_end))

                fr_group = []
                woe_mean = np.zeros(group)

                ax = plt.figure(fig).gca()
                for gi in range(group):
                    idx0 = num_per_group * gi
                    idx1 = num_per_group * (gi + 1)
                    pick_group = np.arange(idx0, idx1)

                    woe_mean[gi] = np.mean(sorted_cumWoe[pick_group])
                    meanFR = np.mean(sortedFirstShape[pick_group, :], axis=0)
                    seFR = np.std(sortedFirstShape[pick_group, :], axis=0, ddof=1) / np.sqrt(num_per_group)

                    fr_vals = np.mean(sortedFirstShape[pick_group][:, t_pick], axis=1)
                    fr_group.append(fr_vals)

                    # plot band
                    fillTrace(t_axis[plot_pick], meanFR[plot_pick], seFR[plot_pick], color_map[gi], ax=ax, alpha=0.25)

                    # annotate windows like MATLAB (optional)
                    if gi == group - 1:
                        analysis_range = t_axis[t_pick]
                        fillRect((analysis_range[0], 0.1), (analysis_range[-1], 3.0), color=(0.8, 0.8, 0.8), ax=ax, alpha=1.0)
                        plot_range = t_axis[plot_pick]
                        fillRect((plot_range[-1], 0.0), (plot_range[-1] + t_gap, 3.0), color="white", ax=ax, alpha=1.0)
                        if ei > 1:
                            ax.plot([plot_range[0], plot_range[0]], [0, 70], "k-", lw=1)

                ax.set_xlabel("Time from first shape onset (ms)")
                ax.set_ylabel("Firing rate (mean ± s.e.)")
                ax.set_xlim(0, 1500)
                ax.set_ylim(0, 70)

                # save quintiles
                info.setdefault("fr_woe", np.full((num_epoch, group), np.nan))
                info.setdefault("fr_mean", np.full((num_epoch, group), np.nan))
                info.setdefault("fr_se", np.full((num_epoch, group), np.nan))
                for gi in range(group):
                    info["fr_woe"][ei - 1, gi] = woe_mean[gi] / 10.0
                    info["fr_mean"][ei - 1, gi] = float(np.mean(fr_group[gi]))
                    info["fr_se"][ei - 1, gi] = float(stderr(fr_group[gi]))

                # GLM leverage of shapes on FR (normal)
                y = np.mean(firstShapeKM[pick][:, t_pick], axis=1)
                X = woeMtrx_shuffled[pick, :ei] / 10.0
                beta, bse, pvals, _ = glmfit_normal(X, y)

                info.setdefault("beta_Mtrx", np.zeros((10, 11)) * np.nan)
                info.setdefault("beta_se_Mtrx", np.zeros((10, 11)) * np.nan)
                info.setdefault("beta_p_Mtrx", np.zeros((10, 11)) * np.nan)

                if ei <= 10:
                    info["beta_Mtrx"][ei - 1, : (ei + 1)] = beta[: (ei + 1)]
                    info["beta_se_Mtrx"][ei - 1, : (ei + 1)] = bse[: (ei + 1)]
                    info["beta_p_Mtrx"][ei - 1, : (ei + 1)] = pvals[: (ei + 1)]

                    axb = plt.figure(fig + 2).add_subplot(2, 5, ei)
                    axb.bar(np.arange(0, ei + 1), beta[: (ei + 1)], width=0.7, color="k")
                    axb.errorbar(np.arange(0, ei + 1), beta[: (ei + 1)], yerr=bse[: (ei + 1)], fmt="k.", lw=1)
                    axb.set_xlim(-1, 11)
                    axb.set_ylim(0, 40)
                    axb.set_xticks(np.arange(0, ei + 1))
                    axb.tick_params(length=0)

            elif sortBy == "LLR":
                group = 8
                cmap = plt.get_cmap("jet")
                color_map = [cmap(i / (group - 1)) for i in range(group)]

                # time windows
                t_plot_pick = (t_axis >= (stim_int * (ei - 1) + t_div)) & (t_axis <= (stim_int * ei + t_div))
                t_pick = (t_axis >= (stim_int * (ei - 1) + t_div + t_start)) & (t_axis <= (stim_int * (ei - 1) + t_div + t_end))
                if ei == 1:
                    t_pre_pick = (t_axis <= t_div)
                else:
                    t_pre_pick = (t_axis >= (stim_int * (ei - 2) + t_div + t_start)) & (t_axis <= (stim_int * (ei - 2) + t_div + t_end))

                deltaFR = np.mean(firstShapeKM[:, t_pick], axis=1) - np.mean(firstShapeKM[:, t_pre_pick], axis=1)

                pick_epoch = (num_accum_shuffled >= (ei + 1))
                if np.sum(pick_epoch) < 1:
                    continue

                # glmfit(deltaFR ~ woe_of_epoch)
                x = woeMtrx_shuffled[pick_epoch, ei - 1] / 10.0
                y = deltaFR[pick_epoch]
                beta, bse, pvals, _ = glmfit_normal(x, y)

                # accumulate per uniqueWoe for the “all epochs” delta FR plot
                for wi, w in enumerate(uniqueWoe):
                    pick_group = pick_epoch & (woeMtrx_shuffled[:, ei - 1] == w)
                    if np.sum(pick_group) <= 1:
                        continue
                    allEpochDeltaFR[w].append(deltaFR[pick_group])

                    # PSTH by weight
                    ax = plt.figure(fig).gca()
                    t_plot = t_axis[t_plot_pick]
                    meanFR = np.mean(firstShapeKM[pick_group, :], axis=0)
                    seFR = np.std(firstShapeKM[pick_group, :], axis=0, ddof=1) / np.sqrt(np.sum(pick_group))
                    fillTrace(t_plot[:-t_gap], meanFR[t_plot_pick][:-t_gap], seFR[t_plot_pick][:-t_gap],
                              color_map[wi], ax=ax, alpha=0.25)
                    if ei > 1:
                        ax.plot([t_plot[0], t_plot[0]], [0, 70], "k-", lw=1)

        # After epochs: plot delta FR (Fig 3B)
        if sortBy == "LLR":
            allDeltaFR = []
            allWOE = []
            means = []
            ses = []
            for w in uniqueWoe:
                if len(allEpochDeltaFR[w]) == 0:
                    means.append(np.nan)
                    ses.append(np.nan)
                    continue
                vals = np.concatenate(allEpochDeltaFR[w])
                vals = vals[np.isfinite(vals)]
                means.append(np.nanmean(vals))
                ses.append(nanse(vals))
                allDeltaFR.append(vals)
                allWOE.append(np.ones_like(vals) * w)
            means = np.asarray(means)
            ses = np.asarray(ses)

            ax = plt.figure(fig + 2).gca()
            ploterr(uniqueWoe / 10.0, means, yerr=ses, ax=ax, fmt="ko", color="k")
            ax.set_xlim(-1.1, 1.1)
            if monkey_id == 1:
                ax.set_ylim(-5, 15)
                ax.set_yticks(np.arange(-5, 16, 5))
            else:
                ax.set_ylim(-30, 30)
                ax.set_yticks(np.arange(-30, 31, 10))
            ax.set_xlabel("Evidence for Tin (logLR)")
            ax.set_ylabel("Change in firing rate (sp/s)")
            ax.tick_params(direction="out")
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)

        fig += 3

    # ============================================================
    # Fig 4A: peri-saccadic FR vs cumWoe slope over time
    # ============================================================
    if fig_flags[3]:
        num_fig = 5
        group = 5
        cmap = plt.get_cmap("jet")
        color_map = [cmap(i / (group - 1)) for i in range(group)]

        t_pre_sac = -500
        t_post_sac = 150
        t_axis = np.arange(t_pre_sac, t_post_sac + 1)
        t_center = t_axis.copy()

        # Extract peri-saccadic KM
        sacKM = np.full((KM.shape[0], len(t_axis)), np.nan, dtype=float)
        for ti in range(KM.shape[0]):
            RT_ind = int(round(RT[ti]))  # RT is in ms; KM indexed by ms
            t_min = RT_ind + t_pre_sac
            t_max = RT_ind + t_post_sac
            if t_min < 0:
                # pad on left
                sacKM[ti, (-t_min):] = KM[ti, 0:(t_max + 1)]
            else:
                if t_max < KM.shape[1]:
                    sacKM[ti, :] = KM[ti, t_min:(t_max + 1)]
                else:
                    # pad right
                    avail = KM.shape[1] - t_min
                    sacKM[ti, :avail] = KM[ti, t_min:]

        lastStim2Sac = TM[:, 3].copy()

        sacKM_shuffled = sacKM[rand_trial_order, :] / FRgain
        T_in_out_shuffled = T_in_out[rand_trial_order]
        correct_shuffled = correct[rand_trial_order]
        rawCumLLR_shuffled = rawCumLLR2[rand_trial_order, :]
        lastStim2Sac_shuffled = lastStim2Sac[rand_trial_order]

        extra_delay_list = [100.0]  # MATLAB loops but only 100
        delay = float(info.get("delay", 0.0))

        for extra_delay in extra_delay_list:
            # cumWoeKM: trial x time in peri-saccade window
            cumWoeKM = np.full((KM.shape[0], len(t_axis)), np.nan, dtype=float)
            sac_ind = int(round(-t_pre_sac + delay + extra_delay))  # index into t_axis (0-based-ish)
            # Convert to python index in [0..len-1]
            sac_ind = np.clip(sac_ind, 0, len(t_axis) - 1)

            for ti in range(KM.shape[0]):
                t_lastStim_ind = sac_ind - int(round(lastStim2Sac_shuffled[ti]))
                if t_lastStim_ind < 0:
                    continue
                # last (most recent) cumLLR at column 1 in backward matrix is rawCumLLR_shuffled(ti,1) in MATLAB
                # In Python: rawCumLLR_shuffled[ti, 0]
                cumWoeKM[ti, t_lastStim_ind:] = rawCumLLR_shuffled[ti, 0]
                for si in range(1, 6):
                    left = t_lastStim_ind - int(round(stim_int * si))
                    right = t_lastStim_ind - int(round(stim_int * (si - 1)))
                    if left > 0:
                        cumWoeKM[ti, left:right] = rawCumLLR_shuffled[ti, si]
                    else:
                        if right > 0:
                            cumWoeKM[ti, 0:right] = rawCumLLR_shuffled[ti, si]
                        break

            plt.figure(fig); plt.clf()
            plt.gcf().set_size_inches(7.5, 3.5)

            slope_v = np.full(len(t_center), np.nan)
            slope_se_v = np.full(len(t_center), np.nan)
            slope_p_v = np.full(len(t_center), np.nan)

            fr_mean_group = {1: np.full((group, len(t_center)), np.nan),
                             2: np.full((group, len(t_center)), np.nan)}
            fr_se_group = {1: np.full((group, len(t_center)), np.nan),
                           2: np.full((group, len(t_center)), np.nan)}
            fr_raw_correct = {}
            fr_raw_error = {}

            for i_inout in [2, 1]:  # Tout then Tin (matches MATLAB loop)
                pick = (T_in_out_shuffled == i_inout)

                for ti in range(len(t_center)):
                    cumWoe_pick = cumWoeKM[pick, ti]
                    fr_pick = sacKM_shuffled[pick, ti]
                    corr_pick = correct_shuffled[pick]

                    pick_finite = np.isfinite(cumWoe_pick) & np.isfinite(fr_pick)
                    if np.sum(pick_finite) <= 1:
                        continue

                    cumWoe_finite = cumWoe_pick[pick_finite]
                    fr_finite = fr_pick[pick_finite]
                    corr_finite = corr_pick[pick_finite]

                    # weights by inverse variance per unique cumWoe
                    uniqueCumWoe = np.unique(cumWoe_finite)
                    bigS = np.full_like(cumWoe_finite, np.nan, dtype=float)
                    for w in uniqueCumWoe:
                        pick_w = (cumWoe_finite == w)
                        if np.sum(pick_w) >= 10:
                            bigS[pick_w] = np.std(fr_finite[pick_w], ddof=1)
                    ok = np.isfinite(bigS) & (bigS > 0)
                    if np.sum(ok) < 5:
                        continue

                    beta, bse, pvals, _ = glmfit_normal(cumWoe_finite[ok] / 10.0, fr_finite[ok], weights=(bigS[ok] ** -2))
                    slope_v[ti] = beta[1]
                    slope_se_v[ti] = bse[1]
                    slope_p_v[ti] = pvals[1]

                    # quantiles by cumWoe
                    order = np.argsort(cumWoe_finite)
                    fr_sorted = fr_finite[order]
                    n = len(fr_sorted)
                    num_per_group = int(np.floor(n / group))
                    if num_per_group < 1:
                        continue
                    for gi in range(group):
                        idx0 = num_per_group * gi
                        idx1 = num_per_group * (gi + 1)
                        seg = fr_sorted[idx0:idx1]
                        fr_mean_group[i_inout][gi, ti] = np.nanmean(seg)
                        fr_se_group[i_inout][gi, ti] = np.nanstd(seg, ddof=1) / np.sqrt(num_per_group)

                    # store correct/error raw at each time
                    fr_raw_correct[(i_inout, ti)] = fr_finite[corr_finite.astype(bool)]
                    fr_raw_error[(i_inout, ti)] = fr_finite[(~corr_finite.astype(bool))]

                # plot PSTH quantiles
                ax = plt.figure(fig).gca()
                for gi in range(group):
                    # tint Tout
                    c = np.array(color_map[gi])
                    if i_inout == 2:
                        c = 1 - (1 - c) / (1 + 0.5 * (i_inout - 1))
                    fillTrace(t_center[2:], fr_mean_group[i_inout][gi, 2:], fr_se_group[i_inout][gi, 2:], color=c, ax=ax, alpha=0.25)

            ax = plt.figure(fig).gca()
            ax.plot([0, 0], [0, 100], "k--", lw=1)
            ax.set_ylim(0, 80)
            ax.set_xlim(t_pre_sac, t_post_sac)
            ax.set_xticks(np.arange(-500, 101, 100))
            ax.set_yticks(np.arange(0, 71, 10))
            ax.set_xlabel("Time from saccade (ms)")
            ax.set_ylabel("Firing rate (sp/s)")
            ax.tick_params(direction="out")
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)

            # p-values plot
            axp = plt.figure(fig + 1).gca()
            axp.plot(t_center, slope_p_v, "bo-", ms=3)
            axp.plot(t_center, 0.05 * np.ones_like(t_center), "k--", lw=1)
            axp.set_xlim(-500, 150)
            axp.set_xlabel("Time from saccade (ms)")
            axp.set_ylabel("p-val")
            axp.tick_params(direction="out")
            for s in ["top", "right"]:
                axp.spines[s].set_visible(False)

            # slope inset
            axs = plt.figure(fig + 2).gca()
            fillTrace(t_center, slope_v, slope_se_v, color=(0, 0, 0), ax=axs, alpha=0.3)
            axs.plot([0, 0], [-10, 30], "k--", lw=1)
            axs.plot(t_center, np.zeros_like(t_center), "k--", lw=1)
            axs.set_xlim(-500, 150)
            if monkey_id == 1:
                axs.set_ylim(-3, 4)
            else:
                axs.set_ylim(-5, 20)
            axs.set_xlabel("Time from saccade (ms)")
            axs.set_ylabel("Regression slope")
            axs.tick_params(direction="out")
            for s in ["top", "right"]:
                axs.spines[s].set_visible(False)

            # presaccadic correct vs error Tin at t>=-50
            t_ind = int(np.argmax(t_center >= -50))
            a = fr_raw_correct.get((1, t_ind), np.array([]))
            b = fr_raw_error.get((1, t_ind), np.array([]))
            if len(a) > 2 and len(b) > 2:
                res = ttest2(a, b)
                print("H0: Presaccadic FR same for correct vs error Tin trials")
                print("p =", res.pvalue)

        fig += num_fig

    # ============================================================
    # Fig 4B: delta FR associated with N*th shapes
    # ============================================================
    if fig_flags[4]:
        lastStim2Sac = TM[:, 3].copy()

        # while sum(lastStim2Sac < lastAccumShape2Sac): add stim_int for those
        while np.sum(lastStim2Sac < lastAccumShape2Sac) > 0:
            pick = (lastStim2Sac < lastAccumShape2Sac)
            lastStim2Sac[pick] += stim_int

        KM_shuffled = KM[rand_trial_order, :]
        LLR_shuffled = LLR2[rand_trial_order, :]
        num_accum_shuffled = num_accum[rand_trial_order]
        T_in_out_shuffled = T_in_out[rand_trial_order]
        lastStim2Sac_shuffled = lastStim2Sac[rand_trial_order]

        woe = np.array([-9, -7, -5, -3, 3, 5, 7, 9], dtype=int)

        plot_start = int(t_div)
        plot_end = int(t_div + 500)
        t_temp = np.arange(plot_start, plot_end + 1)

        t_start = 100
        t_end = int(stim_int)

        # Extract N*th epoch window and previous epoch window
        oneShapeKM = np.full((KM_shuffled.shape[0], plot_end - plot_start + 1), np.nan, dtype=float)
        oneShapeKM_pre = np.full_like(oneShapeKM, np.nan)

        for i in range(KM_shuffled.shape[0]):
            if num_accum_shuffled[i] <= 0:
                continue
            # current N*th epoch
            base = int(stim_int * (num_accum_shuffled[i] - 1)) + 1
            t_label = np.arange(base + plot_start, base + plot_end + 1)
            t_label = t_label[(t_label >= 0) & (t_label < KM_shuffled.shape[1])]
            if len(t_label) == oneShapeKM.shape[1]:
                oneShapeKM[i, :] = KM_shuffled[i, t_label]

            # previous epoch
            if num_accum_shuffled[i] == 1:
                # shift so that pre-window includes [0..t_div]
                t_label2 = np.arange(0, int(t_div) + 1)
                # place into positions t_start..t_start+t_div
                if (t_start + len(t_label2)) <= oneShapeKM_pre.shape[1]:
                    oneShapeKM_pre[i, t_start:(t_start + len(t_label2))] = KM_shuffled[i, t_label2]
            else:
                base2 = int(stim_int * (num_accum_shuffled[i] - 2)) + 1
                t_label2 = np.arange(base2 + plot_start, base2 + plot_end + 1)
                t_label2 = t_label2[(t_label2 >= 0) & (t_label2 < KM_shuffled.shape[1])]
                if len(t_label2) == oneShapeKM_pre.shape[1]:
                    oneShapeKM_pre[i, :] = KM_shuffled[i, t_label2]

        oneShapeKM /= FRgain
        oneShapeKM_pre /= FRgain

        # time window to compute pre FR
        t_pick_pre = (t_temp >= (t_div + t_start)) & (t_temp <= (t_div + t_end))

        for i_inout in [1, 2]:
            pick_in_out = (T_in_out_shuffled == i_inout)
            pick_sign = ((-1) ** (i_inout - 1) * LLR_shuffled[:, 0] > 0)
            pick_opposite_sign = ((-1) ** (i_inout - 1) * LLR_shuffled[:, 0] < 0)

            if i_inout == 1:
                woe2regress = [4, 5, 6, 7]  # indices 5:8 in MATLAB -> 0-based 4..7
            else:
                woe2regress = [0, 1, 2, 3]  # 1:4 -> 0..3

            epochGroupDeltaFR_mean = np.full(8, np.nan)
            epochGroupDeltaFR_se = np.full(8, np.nan)

            # delta FR conditioned on logLR at N*
            for gi in range(8):
                pick_woe = (LLR_shuffled[:, 0] == woe[gi])
                pick = pick_in_out & pick_woe
                fr_picked = oneShapeKM[pick, :]
                fr_picked_pre = oneShapeKM_pre[pick, :]
                lastStim_picked = lastStim2Sac_shuffled[pick]

                epochGroupDeltaFR = np.full(lastStim_picked.shape[0], np.nan)
                for ti in range(lastStim_picked.shape[0]):
                    t_pick_Nstar = (t_temp <= lastStim_picked[ti])
                    epochGroupDeltaFR[ti] = np.nanmean(fr_picked[ti, t_pick_Nstar]) - np.nanmean(fr_picked_pre[ti, t_pick_pre])

                epochGroupDeltaFR_mean[gi] = np.nanmean(epochGroupDeltaFR)
                epochGroupDeltaFR_se[gi] = nanse(epochGroupDeltaFR)

            # delta FR for all trials (for regression)
            epochDeltaFR_all = np.full(KM_shuffled.shape[0], np.nan)
            for ti in range(KM_shuffled.shape[0]):
                t_pick_Nstar = (t_temp <= lastStim2Sac_shuffled[ti])
                epochDeltaFR_all[ti] = np.nanmean(oneShapeKM[ti, t_pick_Nstar]) - np.nanmean(oneShapeKM_pre[ti, t_pick_pre])

            pick = pick_in_out & pick_sign & np.isfinite(epochDeltaFR_all)
            epochDeltaFR = epochDeltaFR_all[pick]
            LLR_picked = LLR_shuffled[pick, 0]

            # weights by std per LLR level
            bigS = np.full_like(epochDeltaFR, np.nan, dtype=float)
            for idx in woe2regress:
                wval = woe[idx]
                pick_w = (LLR_picked == wval)
                if np.sum(pick_w) > 1:
                    bigS[pick_w] = np.nanstd(epochDeltaFR[pick_w], ddof=1)

            ok = np.isfinite(bigS) & (bigS > 0)
            beta, bse, pvals, res = glmfit_normal(LLR_picked[ok] / 10.0, epochDeltaFR[ok], weights=(bigS[ok] ** -2))

            if i_inout == 1:
                print(f"slope for delta FR vs logLR: {beta[1]:.1f} ± {bse[1]:.1f} (p={pvals[1]:.3g})")
                epochDeltaFR_opposite = epochDeltaFR_all[pick_in_out & pick_opposite_sign & np.isfinite(epochDeltaFR_all)]
                if len(epochDeltaFR_opposite) > 2:
                    tres = ttest1(epochDeltaFR_opposite)
                    print(f"delta FR for negative shapes: mean={np.nanmean(epochDeltaFR_opposite):.1f}, se={nanse(epochDeltaFR_opposite):.1f}, p={tres.pvalue:.3g}")

            # plot mean delta FR vs logLR
            plt.figure(fig + (i_inout - 1))
            ax = plt.gca()
            for gi in range(8):
                ploterr(woe[gi] / 10.0, epochGroupDeltaFR_mean[gi], yerr=epochGroupDeltaFR_se[gi],
                        ax=ax, fmt="ko", color="k")
            # regression line over woe2regress range
            wL = woe[woe2regress[0]] / 10.0
            wR = woe[woe2regress[-1]] / 10.0
            ax.plot([wL, wR], [beta[0] + beta[1] * wL, beta[0] + beta[1] * wR], "k--", lw=1)

            if monkey_id == 1:
                if i_inout == 1:
                    ax.set_ylim(0, 20)
                else:
                    ax.set_ylim(-10, 10)
            else:
                ax.set_ylim(-10, 10)

            ax.set_xlabel("Evidence for Tin (logLR)")
            ax.set_ylabel("Δ firing rate (sp/s)")
            ax.tick_params(direction="out")
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)

        fig += 2

    info["fig"] = fig
    return info


def load_mat73(file_path: str) -> dict:
    """Load MATLAB v7.3 .mat file and return nested dictionary of numpy arrays."""
    import h5py
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


# -----------------------------
# Example usage: load Monkey E data and plot all figures
# -----------------------------
if __name__ == "__main__":
    # Load Monkey E (Eli) data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mat_path = os.path.join(script_dir, "neuralSPRT", "data", "dataJ.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Data file not found: {mat_path}")

    raw_data = load_mat73(mat_path)
    info_dict = raw_data["info"]
    TM = np.array(info_dict["TM"]).T  # Trial matrix: (n_trials, n_cols)
    KM = np.array(info_dict["KM"]).T  # Neural matrix: (n_trials, n_time)

    info = {
        "TM": TM,
        "KM": KM,
        "FRgain": 100.0,  # KM stored as FR * 100 (sp/s)
        "fig": 1,
    }
    monkey_id = 2  # 1=Eli, 2=Joey
    fig_switch = [1, 2, 3, 4]  # All figures (3A/3B, 4A, 4B)

    info = neuralSPRT_PHY(info, monkey_id, fig_switch)

    # Save figures to current folder
    save_dir = script_dir
    for i in range(1, info["fig"]):
        plt.figure(i)
        plt.savefig(os.path.join(save_dir, f"neuralSPRT_PHY_MonkeyE_fig{i}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(save_dir, f"neuralSPRT_PHY_MonkeyE_fig{i}.svg"), bbox_inches="tight")
        print(f"Saved fig{i} to {save_dir}")
    plt.show()
