# neural_sprt_beh.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union

import scipy.optimize as opt
import scipy.stats as st

import statsmodels.api as sm

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.labelsize': 9,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.75,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.loc': 'upper right',
})


# -----------------------------
# Helper functions (MATLAB equivalents)
# -----------------------------

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


def nanse(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Standard error of the mean ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    n = np.sum(np.isfinite(x), axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    return sd / np.sqrt(np.maximum(n, 1))


def ploterr(x, y, yerr=None, xerr=None, ax=None, marker="o", **kwargs):
    """
    Simplified MATLAB ploterr: returns handles for (marker, errorbar).
    """
    if ax is None:
        ax = plt.gca()
    h = ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=marker, **kwargs)
    return h


def fillTrace(x_range, y_center, y_se, c=None, c_alpha=0.8, ax=None):
    """MATLAB-like fillTrace(X,Y,err,color,c_alpha): filled band + center line.

    MATLAB convention:
      - default color c = 'b'
      - default c_alpha = 0.8
      - fill color = 1 - (1-c) * c_alpha  (lighter as c_alpha decreases)
      - fill is opaque (FaceAlpha = 1), border drawn with original c.
    """
    if ax is None:
        ax = plt.gca()
    if c is None:
        c = np.array([0.0, 0.0, 1.0])  # default 'b'
    c_arr = np.asarray(c, dtype=float).reshape(-1)
    if c_arr.size == 1:
        c_arr = np.array([c_arr.item()] * 3, dtype=float)
    fill_c = 1.0 - (1.0 - c_arr) * c_alpha
    fill_c = np.clip(fill_c, 0.0, 1.0)

    X = np.asarray(x_range, dtype=float).reshape(-1)
    Y = np.asarray(y_center, dtype=float).reshape(-1)
    err = np.asarray(y_se, dtype=float).reshape(-1)
    if Y.size == 1:
        Y = np.full(X.size, Y.item(), dtype=float)
    if err.size == 1:
        err = np.full(X.size, err.item(), dtype=float)
    if X.size != Y.size:
        if X.size == 2 and Y.size == 2:
            pass
        else:
            raise ValueError("fillTrace: X and Y size mismatch")
    if err.size != Y.size:
        raise ValueError("fillTrace: err and Y size mismatch")
    pick = np.isfinite(Y) & np.isfinite(X)
    if not np.any(pick):
        return
    X = X[pick]
    Y = Y[pick]
    err = err[pick]
    err[np.isnan(err)] = 0
    ax.fill_between(X, Y - err, Y + err, color=fill_c, linewidth=0)
    ax.plot(X, Y, color=c_arr, linewidth=1)


def glmfit_binomial_logit(X: np.ndarray, y01: np.ndarray, add_const: bool = True):
    """
    MATLAB glmfit(..., 'binomial','link','logit','constant', const) equivalent.
    y01 must be 0/1.
    """
    X = np.asarray(X, dtype=float)
    y01 = np.asarray(y01, dtype=float)

    if add_const:
        X_ = sm.add_constant(X, has_constant="add")
    else:
        X_ = X

    model = sm.GLM(y01, X_, family=sm.families.Binomial())
    res = model.fit()

    coef = res.params
    se = res.bse
    p = res.pvalues
    return coef, se, p, res


def glmfit_normal_weighted(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """
    MATLAB glmfit(x, y, 'normal','weights', w) equivalent (WLS).
    Returns coef (intercept, slope), se, p.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)

    X = sm.add_constant(x)
    model = sm.WLS(y, X, weights=w)
    res = model.fit()
    return res.params, res.bse, res.pvalues, res


def makeStimProb(monkey_id: int, _dummy_flag: int = 0) -> np.ndarray:
    """Exact translation of MATLAB makeStimProb(id, trump)."""
    trump = int(_dummy_flag)

    if monkey_id == 1:
        stim_ratio = np.array([0, 1, 2, 4, 8], dtype=float)
    elif monkey_id == 2:
        stim_ratio = np.array([0.1, 1, 2, 4, 8], dtype=float)
    else:
        raise ValueError("monkey_id must be 1 or 2")

    llr = np.array([9999, 9, 7, 5, 3], dtype=float) / 10.0
    sampling_p = stim_ratio / stim_ratio.sum()
    with np.errstate(over="ignore", invalid="ignore"):
        p1 = sampling_p * (10.0**llr) / (1.0 + 10.0**llr)
    pick_nan = np.isnan(p1)
    p1[pick_nan] = sampling_p[pick_nan]
    p2 = sampling_p - p1
    pp1 = np.r_[p1, p2[::-1]]
    pR = pp1[::-1]

    if trump:
        shape_prob = np.r_[pR[:5], 0.0, 0.0, pR[5:10]]
    else:
        shape_prob = np.r_[pR[1:5], 0.0, 0.0, pR[5:9]]
    return shape_prob.astype(float)


def linearFitErr(beta: np.ndarray, x: np.ndarray, y: np.ndarray, y_se: np.ndarray) -> float:
    """Exact translation of MATLAB linearFitErr(beta, x, y, y_se).

    MATLAB only uses `pick = (y_se > 0)` (NaN comparisons are false so NaN
    y_se entries are excluded implicitly). We replicate that behavior here.
    """
    beta = np.asarray(beta, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    y_se = np.asarray(y_se, dtype=float).reshape(-1)

    with np.errstate(invalid="ignore"):
        pick = y_se > 0
    x = x[pick]
    y = y[pick]
    y_se = y_se[pick]

    if beta.size < 2:
        yhat = x * beta[0]
    else:
        yhat = beta[0] + x * beta[1]
    return float(np.sum(((y - yhat) / y_se) ** 2 / 2.0))


# -----------------------------
# Main function: neuralSPRT_BEH
# -----------------------------

def neuralSPRT_BEH(info: Dict[str, Any], monkey_id: int, fig_switch: Union[int, List[int]]) -> Dict[str, Any]:
    """
    Python translation of MATLAB:
        function info = neuralSPRT_BEH(info,id,fig_switch)

    info must include:
      - info["TM"] : trial matrix (n_trials x n_cols)
      - info["fig"]: starting figure index (int)

    monkey_id: 1 (Eli) or 2 (Joey)
    fig_switch: int or list of ints in {1..6}
    """

    # ---- fig_switch logic: allow single int or list
    if isinstance(fig_switch, int):
        fig_temp = [fig_switch]
    else:
        fig_temp = list(fig_switch)

    fig_flags = np.zeros(7, dtype=int)  # 1..6 used
    for k in fig_temp:
        fig_flags[k] = 1

    # ---- monkey parameters
    if monkey_id == 1:
        lastAccumShape2Sac = 270.0
        stim_int = 250.0
    elif monkey_id == 2:
        lastAccumShape2Sac = 180.0
        stim_int = 270.0
    else:
        raise ValueError("monkey_id must be 1 or 2")

    if fig_flags[2]:
        # Set "lastAccumShape2Sac" to 0 (ms) to include all the observed shapes
        # (as opposed to accumulated shapes) in order to plot Fig. 2B
        lastAccumShape2Sac = 0.0

    fig = int(info.get("fig", 1))

    # ---- load TM
    TM = np.asarray(info["TM"], dtype=float).copy()

    total_shape = 8
    shape_offset = 2

    # Remove trials with trump shapes (monkey J)
    shapeMtrx = TM[:, 10:30] + 1  # MATLAB 11:30 -> Python 10:30 (exclusive)
    pick_trump = np.sum((shapeMtrx == 1) | (shapeMtrx == 2), axis=1).astype(bool)
    num_trump_trial = int(np.sum(pick_trump))
    shapeMtrx = shapeMtrx[~pick_trump, :]
    TM = TM[~pick_trump, :]

    # True assigned weights (logLR) * 10
    trueDeciWOE = np.array([9999, -9999, 9, -9, 7, -7, 5, -5, 3, -3, np.nan], dtype=float)

    # Location matrix if present
    locMtrx = None
    if TM.shape[1] > 70:
        locMtrx = TM[:, 70:90].copy()  # MATLAB 71:90 -> Python 70:90

    # number of shapes shown
    num_shown = np.sum(np.isfinite(shapeMtrx), axis=1).astype(int)

    # reward-assigned target (1:left, 2:right) initially
    rew_targ = TM[:, 0].astype(int)

    if monkey_id == 2:
        # reward-assigned color (Joey only): (rew_targ ~= TM(:,9)) + 1
        rew_color = ((rew_targ != TM[:, 8].astype(int)).astype(int) + 1)

    # choice recoding
    if monkey_id == 1:
        # choice = 3 - TM(:,2); rew_targ = 3 - rew_targ; sign flip later
        choice = (3 - TM[:, 1].astype(int))
        rew_targ = (3 - rew_targ)
    else:
        choice = TM[:, 9].astype(int)  # TM(:,10) -> col 9

    RT = TM[:, 2].copy()          # TM(:,3)
    lastStim2Sac = TM[:, 3].copy()  # TM(:,4)

    correct = (TM[:, 0].astype(int) == TM[:, 1].astype(int)).astype(int)

    # rawLLR
    rawShapeMtrx = shapeMtrx.copy()
    rawShapeMtrx[~np.isfinite(rawShapeMtrx)] = 11
    rawShapeMtrx = rawShapeMtrx.astype(int)
    rawLLR1 = trueDeciWOE[rawShapeMtrx - 1]  # MATLAB indices start at 1
    rawLLR = {1: rawLLR1}

    # Compute N* (num_accum)
    print("compute N*\n")
    num_accum = num_shown.copy()
    t_cutoff = lastAccumShape2Sac * np.ones_like(num_shown, dtype=float)
    t_cutoff = t_cutoff - TM[:, 3]  # - sac_latency

    pick = t_cutoff > 0
    num_accum[pick] = num_accum[pick] - 1
    while np.any(t_cutoff > 0):
        t_cutoff = t_cutoff - stim_int
        pick = t_cutoff > 0
        num_accum[pick] = num_accum[pick] - 1

    # truncate shapeMtrx/locMtrx after N*
    for i in range(TM.shape[0]):
        if num_accum[i] < 0:
            num_accum[i] = 0
        # MATLAB: shapeMtrx(i, num_accum(i)+1:num_shown(i)) = nan;
        start = num_accum[i]
        end = num_shown[i]
        if end > start:
            shapeMtrx[i, start:end] = np.nan
            if locMtrx is not None:
                locMtrx[i, start:end] = np.nan

    shapeMtrx2 = shapeMtrx.copy()
    shapeMtrx2[~np.isfinite(shapeMtrx2)] = 11
    shapeMtrx2 = shapeMtrx2.astype(int)
    LLR1 = trueDeciWOE[shapeMtrx2 - 1]
    LLR = {1: LLR1}

    # IMPORTANT: sign convention flip for monkey E
    if monkey_id == 1:
        rawLLR[1] = -rawLLR[1]
        LLR[1] = -LLR[1]

    cumLLR = {1: np.cumsum(LLR[1], axis=1)}
    rawCumLLR = {1: np.cumsum(rawLLR[1], axis=1)}

    # reverse element order in each row depending on N
    LLR2 = np.full_like(LLR[1], np.nan)
    cumLLR2 = np.full_like(cumLLR[1], np.nan)
    rawLLR2 = np.full_like(rawLLR[1], np.nan)
    rawCumLLR2 = np.full_like(rawCumLLR[1], np.nan)

    for ci in range(1, 21):
        # accumulated shapes
        pick = (num_accum == ci)
        num_shift = 20 - ci
        if np.any(pick):
            LLR2[pick, :] = np.fliplr(np.roll(LLR[1][pick, :], shift=num_shift, axis=1))
            cumLLR2[pick, :] = np.fliplr(np.roll(cumLLR[1][pick, :], shift=num_shift, axis=1))

        # shown shapes
        pick = (num_shown == ci)
        if np.any(pick):
            rawLLR2[pick, :] = np.fliplr(np.roll(rawLLR[1][pick, :], shift=num_shift, axis=1))
            rawCumLLR2[pick, :] = np.fliplr(np.roll(rawCumLLR[1][pick, :], shift=num_shift, axis=1))

    LLR[2] = LLR2
    cumLLR[2] = cumLLR2
    rawLLR[2] = rawLLR2
    rawCumLLR[2] = rawCumLLR2

    totalLLR = np.nansum(LLR[1], axis=1)

    # -----------------------------
    # Fig 2A: RT histogram
    # -----------------------------
    if fig_flags[1]:
        t_bin = np.arange(0, 5000 + 1, 10)

        rt_hist = histc(RT, np.arange(0, 5000 + 1))
        rt_cum_hist = np.cumsum(rt_hist)
        rt_hist_10ms = np.diff(np.r_[0, rt_cum_hist[t_bin]])

        p_rt_hist = rt_hist / np.sum(rt_hist)
        cum_p_rt_hist = np.cumsum(p_rt_hist)
        p_rt_hist_10ms = np.diff(np.r_[0, cum_p_rt_hist[t_bin]])

        rt_hist_correct = histc(RT[correct == 1], np.arange(0, 5000 + 1))
        rt_cum_hist_correct = np.cumsum(rt_hist_correct)
        rt_hist_10ms_correct = np.diff(np.r_[0, rt_cum_hist_correct[t_bin]])

        p_rt_hist_correct = rt_hist_correct / np.sum(rt_hist)
        cum_p_rt_hist_correct = np.cumsum(p_rt_hist_correct)
        p_rt_hist_10ms_correct = np.diff(np.r_[0, cum_p_rt_hist_correct[t_bin]])

        rt_hist_error = histc(RT[correct == 0], np.arange(0, 5000 + 1))
        rt_cum_hist_error = np.cumsum(rt_hist_error)
        rt_hist_10ms_error = np.diff(np.r_[0, rt_cum_hist_error[t_bin]])

        p_rt_hist_error = rt_hist_error / np.sum(rt_hist)
        cum_p_rt_hist_error = np.cumsum(p_rt_hist_error)
        p_rt_hist_10ms_error = np.diff(np.r_[0, cum_p_rt_hist_error[t_bin]])

        plt.figure(fig)
        plt.clf()
        plt.bar(t_bin / 1e3, rt_hist_10ms, width=0.01, color="k")
        plt.xlim(0, 3)
        plt.ylim(0, 600)
        ax = plt.gca()
        ax.tick_params(direction="out")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

        info["rt_hist"] = rt_hist
        info["rt_hist_correct"] = rt_hist_correct
        info["rt_hist_error"] = rt_hist_error
        info["p_rt_hist_10ms"] = p_rt_hist_10ms
        info["p_rt_hist_10ms_correct"] = p_rt_hist_10ms_correct
        info["p_rt_hist_10ms_error"] = p_rt_hist_10ms_error
        info["rt_hist_10ms"] = rt_hist_10ms
        info["rt_hist_10ms_correct"] = rt_hist_10ms_correct
        info["rt_hist_10ms_error"] = rt_hist_10ms_error

        fig += 1

    # -----------------------------
    # Fig 2B: GLM timing leverage (Forward + Backward)
    # -----------------------------
    if fig_flags[2]:
        # *********** Forward Analysis ***********
        LLR_forward = rawLLR[1] / 10.0
        num_epoch = 10

        combine_choice_flag = 1
        offset_flag = 0
        add_const = bool(offset_flag)

        # With constant='off', MATLAB glmfit returns two coefficients.
        glm_w = np.zeros((num_epoch, 2), dtype=float)
        glm_w_se = np.zeros_like(glm_w)
        glm_p = np.zeros_like(glm_w)
        glm_w_ci = np.zeros_like(glm_w)
        glm_w_lo = np.zeros_like(glm_w)
        glm_w_hi = np.zeros_like(glm_w)

        for ei in range(num_epoch):
            pick = np.isfinite(LLR_forward[:, ei])
            LLR_picked = LLR_forward[pick, ei]
            res_cols = [c for c in range(num_epoch) if c != ei]
            resLLR_picked = np.nansum(LLR_forward[pick][:, res_cols], axis=1)
            choice_picked = choice[pick]  # {1,2}

            X = np.c_[LLR_picked, resLLR_picked]
            coef, se, pvals, _res = glmfit_binomial_logit(X, (choice_picked - 1), add_const=add_const)

            # Convert log base from exp to 10: divide by ln(10)
            coef = coef / np.log(10)
            se = se / np.log(10)

            glm_w[ei, :] = coef
            glm_w_se[ei, :] = se
            glm_p[ei, :] = pvals

            z975 = st.norm.ppf(0.975)
            glm_w_ci[ei, :] = glm_w_se[ei, :] * z975
            glm_w_lo[ei, :] = glm_w[ei, :] - glm_w_se[ei, :] * z975
            glm_w_hi[ei, :] = glm_w[ei, :] + glm_w_se[ei, :] * z975

        info["glm_w_init"] = glm_w
        info["glm_w_se_init"] = glm_w_se
        info["glm_w_ci_init"] = glm_w_ci
        info["glm_w_lo_init"] = glm_w_lo
        info["glm_w_hi_init"] = glm_w_hi
        info["glm_p_end"] = glm_p

        # Plot forward (left panel)
        plt.figure(fig)
        plt.clf()
        ax1 = plt.axes([0.1, 0.1, 0.8 * 6 / 11, 0.85])
        t_onset = (np.arange(1, num_epoch + 1) - 1) * stim_int

        # MATLAB uses glm_w(:,2) here.
        shape_coef = glm_w[:, 1]
        shape_se = glm_w_se[:, 1]
        for bi in range(5):  # first 5 epochs
            fillTrace([t_onset[bi] - 10, t_onset[bi] + 10], shape_coef[bi], shape_se[bi], 0.2 * np.ones(3), ax=ax1)
            ax1.plot([t_onset[bi], t_onset[bi + 1]], [shape_coef[bi], shape_coef[bi]], "k--", lw=1)
            ax1.plot([t_onset[bi + 1], t_onset[bi + 1]], [shape_coef[bi], shape_coef[bi + 1]], "k--", lw=1)

        print("leverage of kth shape")
        for bi in range(num_epoch):
            print(
                f"Shape {bi + 1}: {glm_w[bi, 1]:.2f} \u00b1 {glm_w_se[bi, 1]:.2f} "
                f"(p = {glm_p[bi, 1]:f})"
            )

        ax1.set_xlim(-50, 550)
        if monkey_id == 1:
            ax1.set_ylim(-0.5, 2)
        else:
            ax1.set_ylim(-2, 10)
        ax1.set_xticks([0, 100, 200, 300, 400, 500])
        ax1.set_xticklabels(["0", "", "0.2", "", "0.4", ""])
        ax1.tick_params(direction="out")
        for s in ["top", "right"]:
            ax1.spines[s].set_visible(False)

        # *********** Backward Analysis ***********
        LLR_back = LLR[2] / 10.0
        shift_size = 10
        bin_size = 20
        num_bin = 100
        _woe_set = np.arange(-9, 10, 2)

        glm_w = np.zeros((num_bin, 2), dtype=float)
        glm_w_se = np.zeros_like(glm_w)
        glm_p = np.zeros_like(glm_w)
        glm_w_ci = np.zeros_like(glm_w)
        glm_w_lo = np.zeros_like(glm_w)
        glm_w_hi = np.zeros_like(glm_w)

        lastStim2Sac_back = lastStim2Sac.copy()
        for ti in range(num_bin):
            pick_before = (lastStim2Sac_back >= 0)
            pick_after = ((lastStim2Sac_back - bin_size) >= 0)
            pickDiff = (pick_before.astype(int) - pick_after.astype(int)) == 1

            lastStim2Sac_back = lastStim2Sac_back - shift_size
            pick_res = ((lastStim2Sac_back - shift_size) >= 0)
            pick_remove = (pick_before.astype(int) - pick_res.astype(int)) == 1
            lastStim2Sac_back[pick_remove] = lastStim2Sac_back[pick_remove] + stim_int

            pick = pickDiff & np.isfinite(LLR_back[:, 0])
            LLR_picked = LLR_back[pick, 0]
            resLLR_picked = np.nansum(LLR_back[pick, 1:], axis=1)
            choice_picked = choice[pick]
            X = np.c_[LLR_picked, resLLR_picked]
            coef, se, pvals, _ = glmfit_binomial_logit(X, (choice_picked - 1), add_const=add_const)
            coef = coef / np.log(10)
            se = se / np.log(10)
            glm_w[ti, :] = coef
            glm_w_se[ti, :] = se
            glm_p[ti, :] = pvals
            z975 = st.norm.ppf(0.975)
            glm_w_ci[ti, :] = se * z975
            glm_w_lo[ti, :] = coef - se * z975
            glm_w_hi[ti, :] = coef + se * z975

            # shift the matrix for trials that were picked and replace with NaN.
            if np.any(pick_remove):
                LLR_back[pick_remove, :] = np.roll(LLR_back[pick_remove, :], shift=-1, axis=1)
                LLR_back[pick_remove, -1] = np.nan

        ax2 = plt.axes([0.11 + 0.8 * 6 / 11, 0.1, 0.8 / 11 * 5, 0.85])
        t_bin_center = bin_size / 2 + shift_size * np.arange(num_bin)
        fillTrace(-t_bin_center, glm_w[:, offset_flag], glm_w_se[:, offset_flag], 0.2 * np.ones(3), ax=ax2)
        ax2.set_xlim(-500, 0)
        if monkey_id == 1:
            ax2.set_ylim(-0.5, 2)
        else:
            ax2.set_ylim(-2, 10)
        ax2.set_xticks([-500, -400, -300, -200, -100, 0])
        ax2.set_xticklabels(["", "-0.4", "", "-0.2", "", "0"])
        ax2.set_yticks([])
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.tick_params(direction="out")
        for s in ["top", "left"]:
            ax2.spines[s].set_visible(False)

        info["glm_w_end"] = glm_w
        info["glm_w_se_end"] = glm_w_se
        info["glm_w_ci_end"] = glm_w_ci
        info["glm_w_lo_end"] = glm_w_lo
        info["glm_w_hi_end"] = glm_w_hi
        info["glm_p_end"] = glm_p
        fig += 2  # matches MATLAB fig = fig+2

    # -----------------------------
    # Fig 2C: N* histogram
    # -----------------------------
    if fig_flags[3]:
        max_num_accum = int(np.max(num_accum))
        n_accum = histc(num_accum, np.arange(1, max_num_accum + 1))

        plt.figure(fig)
        plt.clf()
        plt.bar(np.arange(1, max_num_accum + 1), n_accum, color="k")
        mean_num_accum = np.mean(num_accum)
        std_num_accum = np.std(num_accum, ddof=1)
        print(f"mean N* = {mean_num_accum:.1f} \u00b1 {std_num_accum:.1f}\n")
        plt.xlim(0, 17)
        if monkey_id == 1:
            plt.ylim(0, 9000)
            plt.yticks(np.arange(0, 8001, 2000))
        else:
            plt.ylim(0, 7000)
            plt.yticks(np.arange(0, 6001, 2000))
        plt.xticks([0, 5, 10, 15], [0, 5, 10, 15])
        ax = plt.gca()
        ax.tick_params(direction="out")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

        info["n_accum"] = n_accum
        info["mean_num_accum"] = mean_num_accum
        info["std_num_accum"] = std_num_accum
        fig += 1

    # -----------------------------
    # Fig 2D: Psychometric function
    # -----------------------------
    if fig_flags[4]:
        # Only ci=1 (all trials) translated (same as your MATLAB)
        totalLLR = np.nansum(LLR[1], axis=1)
        choice_picked = choice.copy()

        LLR_edges = np.arange(np.floor(np.min(totalLLR)), np.ceil(np.max(totalLLR)) + 1, 1)
        LLR_all_dist = histc(totalLLR, LLR_edges)
        LLR_choice2 = totalLLR[choice_picked == 2]
        LLR_choice2_dist = histc(LLR_choice2, LLR_edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_choice2 = LLR_choice2_dist / LLR_all_dist

        Y = np.c_[LLR_choice2_dist, LLR_all_dist]  # [#choice2, #total]

        # logistic fit: glmfit(totalLLR, choice-1, binomial logit)
        coef, se, pvals, _res = glmfit_binomial_logit(totalLLR.reshape(-1, 1), (choice_picked - 1), add_const=True)
        coef = coef / np.log(10)
        se = se / np.log(10)

        # predicted
        # MATLAB: p_pred = 1./(1+10.^(-(coef(2).*LLR_edges + coef(1))))
        p_pred = 1.0 / (1.0 + 10.0 ** (-(coef[1] * LLR_edges + coef[0])))

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_y = Y[:, 0] / Y[:, 1]

        # Expected fraction of correct trials given a single shape
        shape_prob_tmp = makeStimProb(monkey_id, 0)
        woe_vec = np.array([-9, -7, -5, -3, 0, 0, 3, 5, 7, 9], dtype=float) / 10.0
        mean_woe = float(np.sum(shape_prob_tmp * woe_vec))
        p_correct_single = 1.0 / (1.0 + 10.0 ** (-mean_woe))

        # Fraction of correct (rewarded) trials
        p_correct_overall = float(np.sum(correct)) / float(np.sum(Y[:, 1]))

        # Fraction of rational trials (choosing target supported by cumulative logLR)
        pick_neg = (LLR_edges < 0)
        pick_zero = (LLR_edges == 0)
        pick_pos = (LLR_edges > 0)
        n_rational = (
            np.sum(Y[pick_neg, 1] - Y[pick_neg, 0]) + np.sum(Y[pick_pos, 0])
        )
        n_total = np.sum(Y[:, 1]) - np.sum(Y[pick_zero, 1])
        p_rational = float(n_rational) / float(n_total) if n_total > 0 else np.nan

        print("Expected fraction of correct trials given a single shape")
        print(f"p = {p_correct_single:.2f}\n")
        print("Fraction of correct (rewarded) trials")
        print(f"p = {p_correct_overall:.2f}\n")
        print("Fraction of trials in which the monkey chose the target supported by cumulative logLR")
        print(f"p = {p_rational:.2f}\n")

        plt.figure(fig)
        plt.clf()
        ax1 = plt.axes([0.3, 0.45, 0.6, 0.5])
        LLRaxis = LLR_edges / 10.0
        ax1.plot(LLRaxis, mean_y, "ko", ms=8, mfc="k")
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Proportion of choice A")
        ax1.tick_params(direction="out")
        for s in ["top", "right"]:
            ax1.spines[s].set_visible(False)

        ax2 = plt.axes([0.3, 0.2, 0.6, 0.15])
        ax2.bar(LLRaxis, Y[:, 1], width=0.1)
        ax2.set_xlim(-3, 3)
        ax2.set_xlabel("Evidence for target A (LLR)")
        ax2.set_ylabel("Trial count")
        ax2.tick_params(direction="out")
        for s in ["top", "right"]:
            ax2.spines[s].set_visible(False)

        info["uniqueLLR"] = LLR_edges
        info["NA"] = Y
        info["PA"] = mean_y

        fig += 1

    # -----------------------------
    # Fig 2E: End cumulative evidence
    # -----------------------------
    if fig_flags[5]:
        num_fig = 1
        shape_prob = makeStimProb(monkey_id, 0)
        p = {1: np.array([1.0])}
        A = 9999

        meanWOE = np.zeros(20, dtype=float)
        varWOE = np.zeros(20, dtype=float)
        stdWOE = np.zeros(20, dtype=float)
        p_upper_bound = np.zeros(21, dtype=float)
        p_lower_bound = np.zeros(21, dtype=float)
        p_correct = np.zeros(21, dtype=float)

        for i in range(1, 21):
            WOE = np.arange(-0.9 * i, 0.9 * i + 1e-9, 0.2)
            p[i + 1] = np.convolve(p[i], shape_prob)
            pick_hi = (WOE >= A)
            pick_lo = (WOE <= -A)
            p_upper_bound[i] = np.sum(p[i + 1][pick_hi])
            p_lower_bound[i] = np.sum(p[i + 1][pick_lo])
            p[i + 1][pick_hi | pick_lo] = 0
            pick_finite = np.isfinite(WOE)
            P = p[i + 1][pick_finite] / np.sum(p[i + 1][pick_finite])
            meanWOE[i - 1] = np.sum(WOE * P)
            varWOE[i - 1] = np.sum((WOE ** 2) * P) - (np.sum(WOE * P) ** 2)
            stdWOE[i - 1] = np.sqrt(varWOE[i - 1])
            p_correct_WOE = 10.0 ** WOE / (1.0 + 10.0 ** WOE)
            p_correct_WOE = np.abs(p_correct_WOE - 0.5) + 0.5
            p_correct[i] = np.sum(p_correct_WOE * p[i + 1])

        cumLLR_all = [[None for _ in range(20)] for _ in range(1)]
        cumLLR_correct = [[None for _ in range(20)] for _ in range(1)]
        cumLLR_error = [[None for _ in range(20)] for _ in range(1)]
        meanCumLLR = np.full((20, 20, 2), np.nan)
        stdCumLLR = np.full((20, 20, 2), np.nan)
        seCumLLR = np.full((20, 20, 2), np.nan)
        varCumLLR = np.full((20, 20, 2), np.nan)

        max_num_accum = int(np.max(num_accum))
        separate_plot_flag = 1
        sortBy = "choice"

        if monkey_id == 1:
            rew = rew_targ
        else:
            rew = rew_color

        i_case = 1
        cumLLR_bigM1 = np.empty((0, 2), dtype=float)
        cumLLR_bigM2 = np.empty((0, 2), dtype=float)

        for j in [1, 2]:  # target index
            for k in range(1, 19):  # epochs to plot
                if separate_plot_flag:
                    if sortBy == "choice":
                        if i_case == 1:
                            pick = np.where((choice == j) & (num_accum == k))[0]
                        elif i_case == 2:
                            pick = np.where((choice == j) & (num_accum == k) & (correct == 1))[0]
                        else:
                            pick = np.where((choice == j) & (num_accum == k) & (correct == 0))[0]
                    else:
                        if i_case == 1:
                            pick = np.where((rew == j) & (num_accum == k))[0]
                        elif i_case == 2:
                            pick = np.where((rew == j) & (num_accum == k) & (correct == 1))[0]
                        else:
                            pick = np.where((rew == j) & (num_accum == k) & (correct == 0))[0]

                    if pick.size < 5:
                        continue

                    pickedCumLLR = cumLLR[1][pick, :] / 10.0
                    for ei in range(1, k + 1):
                        vals = pickedCumLLR[:, ei - 1]
                        pick_finite = np.where((vals > -900) & (vals < 900) & np.isfinite(vals))[0]
                        if pick_finite.size == 0:
                            continue
                        if ei == k:
                            endCumLLR = vals[pick_finite]
                            block = np.c_[np.ones_like(endCumLLR) * ei, endCumLLR]
                            if j == 1:
                                cumLLR_bigM1 = np.vstack([cumLLR_bigM1, block])
                            else:
                                cumLLR_bigM2 = np.vstack([cumLLR_bigM2, block])

                        meanCumLLR[k - 1, ei - 1, j - 1] = np.mean(vals[pick_finite])
                        varCumLLR[k - 1, ei - 1, j - 1] = np.nanvar(vals[pick_finite], ddof=1)
                        stdCumLLR[k - 1, ei - 1, j - 1] = np.nanstd(vals[pick_finite], ddof=1)
                        seCumLLR[k - 1, ei - 1, j - 1] = nanse(vals[pick_finite])

                    if k <= 10:
                        plt.figure(fig)
                        ax = plt.gca()
                        if j == 1:
                            x_offset = 0.05
                            marker = "s"
                        else:
                            x_offset = -0.05
                            marker = "o"
                        x = k + x_offset
                        y = meanCumLLR[k - 1, k - 1, j - 1]
                        yerr = stdCumLLR[k - 1, k - 1, j - 1]
                        if np.isfinite(y) and np.isfinite(yerr):
                            ploterr([x], [y], yerr=[yerr], ax=ax, marker=marker, color="k", mfc="k", mec="k")

        ERR0 = np.full((1, 2), np.nan)
        ERR1 = np.full((1, 2), np.nan)
        n_data = np.full((1, 2), np.nan)
        endCumLLR_beta = np.full((1, 2, 2), np.nan)
        delta_AIC = np.full(1, np.nan)
        delta_BIC = np.full(1, np.nan)

        for j in [1, 2]:
            y_diag = np.diag(meanCumLLR[:, :, j - 1])
            y_se_diag = np.diag(seCumLLR[:, :, j - 1])
            x_axis = np.arange(1, meanCumLLR.shape[0] + 1, dtype=float)
            init_beta = np.array([0.1, 0.0], dtype=float)
            res0 = opt.minimize(
                lambda b: linearFitErr(np.array([b[0], 0.0]), x_axis, y_diag, y_se_diag),
                x0=init_beta,
                method="BFGS",
            )
            beta_fit0 = res0.x
            err0 = res0.fun

            y = y_diag
            y_pred = meanWOE * ((-1) ** j)
            y_se = y_se_diag
            pick_nonzero = (np.abs(y) > 0)
            err1 = np.sum(((y[pick_nonzero] - y_pred[pick_nonzero]) / y_se[pick_nonzero]) ** 2 / 2.0)

            ERR0[0, j - 1] = err0
            ERR1[0, j - 1] = err1
            n_data[0, j - 1] = np.sum(pick_nonzero)
            endCumLLR_beta[0, j - 1, :] = beta_fit0

        delta_AIC[0] = 2 * (1 - 0) - 2 * (np.sum(-ERR0[0, :]) - np.sum(-ERR1[0, :]))
        num_data = np.sum(n_data[0, :])
        delta_BIC[0] = (1 * (np.log(num_data) + np.log(2 * np.pi)) - 0) - 2 * (np.sum(-ERR0[0, :]) - np.sum(-ERR1[0, :]))

        print(f"delta AIC = {int(round(delta_AIC[0]))}\n")
        print(f"delta BIC = {int(round(delta_BIC[0]))}\n")

        plt.figure(fig)
        plt.gcf().set_size_inches(5.05, 2.55)
        plt.xlim(0, 11)
        if separate_plot_flag:
            plt.ylim(-3, 3)
        else:
            plt.ylim(0, 3)
        ax = plt.gca()
        ax.set_xlabel("Time step")
        ax.set_ylabel("Cumulative evidence")
        ax.tick_params(direction="out")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

        betaGLM = np.zeros((2, 2))
        betaGLM_se = np.zeros((2, 2))
        betaGLM_p = np.zeros((2, 2))

        for j in [1, 2]:
            tempM = cumLLR_bigM1 if j == 1 else cumLLR_bigM2
            if tempM.size == 0:
                continue
            unique_N = np.unique(np.round(tempM[:, 0]).astype(int))
            bigS = np.full(tempM.shape[0], np.nan)

            # Match MATLAB: for ni = 1:max(unique_N)
            max_unique_N = int(np.max(unique_N)) if unique_N.size > 0 else 0
            for ni in range(1, max_unique_N + 1):
                pick_n = (np.round(tempM[:, 0]).astype(int) == ni)
                if not np.any(pick_n):
                    continue
                std_n_val = np.std(tempM[pick_n, 1], ddof=1)
                bigS[pick_n] = std_n_val

            x = tempM[:, 0]
            y = tempM[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                w = bigS ** -2
            pick_ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
            x = x[pick_ok]
            y = y[pick_ok]
            w = w[pick_ok]

            params, bse, pvals, _ = glmfit_normal_weighted(x, y, w)
            betaGLM[:, j - 1] = params
            betaGLM_se[:, j - 1] = bse
            betaGLM_p[:, j - 1] = pvals

            print(
                f"slope for cumulative logLR at N*: "
                f"{params[1]:.2f} \u00b1 {bse[1]:.2f} (logLR/epoch) "
                f"(p = {pvals[1]:f})\n"
            )

            N_axis = np.arange(1, 11)
            pred = params[1] * N_axis + params[0]
            plt.plot(N_axis, pred, "k--", lw=1)

        info["cumLLR_all"] = cumLLR_all
        info["meanCumLLR"] = meanCumLLR
        info["stdCumLLR"] = stdCumLLR
        info["seCumLLR"] = seCumLLR
        info["betaGLM"] = betaGLM
        info["betaGLM_se"] = betaGLM_se
        info["betaGLM_p"] = betaGLM_p
        info["delta_AIC"] = delta_AIC
        info["delta_BIC"] = delta_BIC
        fig += num_fig

    # -----------------------------
    # Fig 2F: Subjective weights (SWOE)
    # -----------------------------
    if fig_flags[6]:
        EM = np.zeros((TM.shape[0], total_shape), dtype=float)

        # shapeMtrx currently has NaNs beyond N*; use first num_accum per trial
        for ind in range(TM.shape[0]):
            k = int(num_accum[ind])
            shapes = shapeMtrx[ind, :k]
            # MATLAB histc(shapes, (1+shape_offset):10) with shape_offset=2 => edges 3..10 inclusive
            edges = np.arange(1 + shape_offset, 10 + 1)
            # shapes are in 1..11 coding; ignore NaNs
            shapes = shapes[np.isfinite(shapes)].astype(int)
            # count exact matches to edges (discrete categories)
            counts = np.array([(shapes == e).sum() for e in edges], dtype=float)
            EM[ind, :] = counts  # total_shape=8 (edges length 8)

        print("compute SWOE\n")
        # compute the subjective weight of evidence (Fig. 2F, see also Equation 3)
        coef, se, pvals, _res = glmfit_binomial_logit(EM, (choice - 1), add_const=False)

        # convert the log base from exp to 10
        coef = coef / np.log(10)
        se = se / np.log(10)

        # reorder as in MATLAB
        shapeSWOE = coef.copy()
        shapeSWOE_se = se.copy()

        x_fit = shapeSWOE.copy()
        stdErr = shapeSWOE_se.copy()

        for i in range(1, total_shape + 1):
            if i <= total_shape / 2:
                shapeSWOE[i - 1] = x_fit[2 * i - 1]      # MATLAB 2*i
                shapeSWOE_se[i - 1] = stdErr[2 * i - 1]
            else:
                idx = total_shape - 2 * (i - total_shape // 2)  # MATLAB indexing converted
                shapeSWOE[i - 1] = x_fit[idx]
                shapeSWOE_se[i - 1] = stdErr[idx]

        if monkey_id == 1:
            woe_axis = -np.array([-9, -7, -5, -3, 3, 5, 7, 9]) * 0.1
        else:
            woe_axis = np.array([-9, -7, -5, -3, 3, 5, 7, 9]) * 0.1

        rho, pval = st.spearmanr(woe_axis, shapeSWOE, nan_policy="omit")
        print(f"r = {rho:.2f} (p = {pval:f})")

        plt.figure(fig)
        plt.clf()
        ax = plt.gca()
        ploterr(woe_axis, shapeSWOE, yerr=shapeSWOE_se, ax=ax, marker="o", color="k")
        ax.set_xlim(-1.1, 1.1)
        if monkey_id == 1:
            ax.set_ylim(-1, 1.2)
        else:
            ax.set_ylim(-6, 4)
        ax.tick_params(direction="out")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.set_xlabel("True Weight (LLR)")
        ax.set_ylabel("Subjective Weight (LLR)")
        ax.set_title(f"Spearman r={rho:.2f}, p={pval:.3g}")

        info["shapeSWOE"] = shapeSWOE
        info["shapeSWOE_se"] = shapeSWOE_se

        fig += 1

    info["fig"] = fig
    info["num_trump_trial_removed"] = num_trump_trial
    info["num_accum"] = num_accum
    info["num_shown"] = num_shown
    info["choice"] = choice
    info["correct"] = correct
    info["totalLLR"] = totalLLR

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


if __name__ == "__main__":
    # Load Monkey J (Joey) data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mat_path = os.path.join(script_dir, "data", "dataE.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Data file not found: {mat_path}")

    raw_data = load_mat73(mat_path)
    info_dict = raw_data["info"]
    TM = np.array(info_dict["TM"]).T  # Trial matrix: (n_trials, n_cols)

    info = {"TM": TM, "fig": 1}
    monkey_id = 1  # 1=Eli, 2=Joey
    # fig_switch = [1, 2, 3, 4, 5, 6]  # All figures
    fig_switch = [1, 3, 4, 5, 6]

    info = neuralSPRT_BEH(info, monkey_id, fig_switch)

    # Save figures under neuralSPRT/figures/
    save_dir = Path(script_dir) / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, info["fig"]):
        plt.figure(i)
        plt.savefig(save_dir / f"neuralSPRT_BEH_MonkeyJ_fig{i}.png", dpi=300, bbox_inches="tight")
        plt.savefig(save_dir / f"neuralSPRT_BEH_MonkeyJ_fig{i}.svg", bbox_inches="tight")
        print(f"Saved fig{i} to {save_dir}")
    plt.show()
