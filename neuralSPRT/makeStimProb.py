import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.75,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.loc": "upper right",
        }
    )

def make_stim_prob(stim_id: int, trump: bool = False) -> np.ndarray:

    if stim_id == 1:

        stim_ratio = np.array([0, 1, 2, 4, 8], dtype=float)

    elif stim_id == 2:

        stim_ratio = np.array([0.1, 1, 2, 4, 8], dtype=float)

    llr = np.array([9999, 9, 7, 5, 3], dtype=float) / 10

    sampling_p = stim_ratio / stim_ratio.sum()

    p1 = sampling_p * (10 ** llr) / (1 + 10 ** llr)

    p1[np.isnan(p1)] = sampling_p[np.isnan(p1)]

    p2 = sampling_p - p1

    pp1 = np.concatenate([p1, p2[::-1]])

    pp2 = pp1[::-1]

    pR = pp2

    if trump:

        shape_prob = np.concatenate([pR[:5], [0, 0], pR[5:10]])

    else:

        shape_prob = np.concatenate([pR[1:5], [0, 0], pR[5:9]])

    return shape_prob

# compute

p1 = make_stim_prob(1)

p2 = make_stim_prob(2)

# remove zeros

mask = np.ones_like(p1, dtype=bool)

mask[4] = False

mask[5] = False

p1 = p1[mask]

p2 = p2[mask]

# enforce symmetry: mirror one distribution

p2_sym = p2[::-1]

labels = ["-0.9","-0.7","-0.5","-0.3","0.3","0.5","0.7","0.9"]

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(4,2.25))

width = 0.38

ax.bar(x - width/2, p1, width, color='red', label='Target A')

ax.bar(x + width/2, p2_sym, width, color='blue', label='Target B')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.set_xlabel("Log-likelihood ratio")

ax.set_ylabel("Sampling probability")

# ax.set_title("Symmetric stimulus distributions")

ax.legend(frameon=False)

fig.tight_layout()

out_path = Path("neuralSPRT/distribution.svg")

fig.savefig(out_path, dpi=300, bbox_inches="tight")

plt.show()

out_path