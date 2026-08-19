from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from ecodeRTshape import ecodeRTshape
from neuralSPRT import _load_mat73


def tick_raster(spike_time: np.ndarray, trial_num: int, color: str = "k") -> None:
    if spike_time.ndim != 1:
        spike_time = spike_time.reshape(-1)
    s = np.vstack([spike_time, spike_time, np.full_like(spike_time, np.nan)]).reshape(-1, order="F")
    t = np.vstack([
        np.full_like(spike_time, trial_num - 0.3, dtype=float),
        np.full_like(spike_time, trial_num + 0.3, dtype=float),
        np.full_like(spike_time, np.nan, dtype=float),
    ]).reshape(-1, order="F")
    plt.plot(s, t, "-", color=color)


def plot_lfp_data(info: Dict[str, Any] | None = None) -> None:
    if info is None:
        root = Path(__file__).resolve().parent.parent
        raw = _load_mat73(root / "neuralSPRT" / "data" / "dataE_LFP.mat")
        info = raw["info"]

    trial_id = 0

    trial_lfp = np.asarray(info["LM"][trial_id]).reshape(-1)
    trial_time = np.asarray(info["LMe"][trial_id])[:, 0]
    trial_events = np.asarray(info["LMe"][trial_id])[:, 1]
    trial_v = np.asarray(info["TM"])[trial_id, :]

    codes, event_name, event_code = ecodeRTshape()

    plt.figure(1)
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.title("Trial Events and Spikes")
    for i in range(len(trial_events)):
        skip = {
            codes["E_SPIKE"], codes["E_FP_OFF"], codes["E_FP2_ON"], codes["E_STIM_OFF"],
            codes["E_CLEAR_SCREEN"], codes["E_SHAPE_TASK"], codes["E_PARAM"],
            codes["E_FEEDBACK_ON"], codes["E_TRIAL_INFO"], codes["E_TARGET1_ACQUIRED"],
            codes["E_TARGET2_ACQUIRED"],
        }
        if trial_events[i] in skip:
            continue
        idx = [j for j, ec in enumerate(event_code) if ec == int(trial_events[i])]
        if not idx:
            continue
        en = event_name[idx[0]].replace("E_", "").replace("E-", "").replace("STIM", "SHAPE")
        plt.text(trial_time[i], 2, en, rotation=90)

    spike_time = trial_time[trial_events == codes["E_SPIKE"]]
    tick_raster(spike_time, 1, "k")
    plt.xlim(trial_time[0], trial_time[-1])
    plt.ylim(0, 3)
    plt.xlabel("Elapsed trial time (ms)")
    plt.yticks([1, 2], ["Spikes", "Event Name"])

    t_stim_on = trial_time[trial_events == codes["E_STIM_ON"]]
    n_shape_used = int(np.sum(np.isfinite(trial_v[10:30])))
    t_stim_on = np.r_[0, t_stim_on, trial_time[trial_events == codes["E_SACCADE"]][0]]

    woe = np.array([np.inf, -np.inf, 0.9, -0.9, 0.7, -0.7, 0.5, -0.5, 0.3, -0.3, 0.1, -0.1])
    idx = trial_v[10:10 + n_shape_used].astype(int)
    w = np.r_[0.0, woe[idx]]
    cum_w = np.cumsum(w)

    plt.subplot(3, 1, 2)
    for si in range(n_shape_used + 1):
        plt.plot(t_stim_on[si:si + 2], [cum_w[si], cum_w[si]], "k-")
    plt.xlim(trial_time[0], trial_time[-1])
    plt.ylim(-2, 2)
    plt.xlabel("Elapsed trial time (ms)")
    plt.ylabel("Cumulative evidence (logLR)")

    plt.subplot(3, 1, 3)
    plt.plot(trial_lfp)
    plt.xlim(trial_time[0], trial_time[-1])
    plt.xlabel("Elapsed trial time (ms)")
    plt.ylabel("LFP (a.u.)")
    plt.tight_layout()


if __name__ == "__main__":
    plot_lfp_data()
    plt.show()
