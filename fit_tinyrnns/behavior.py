"""Monkey/RNN behavior processing and plots."""

import os
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from plotting_style import set_style, _save
from simulate import Trial


def trials_from_dataset(dataset):
    evidence = dict(zip(
        [4, 6, 8, 10, 9, 7, 5, 3],
        [-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9],
    ))
    trials = []
    for shapes, length, choice in zip(
        dataset.shapeMtrx, dataset.lengths, dataset.choice
    ):
        shapes = shapes[:length]
        if length <= 0 or not np.all(np.isfinite(shapes)):
            continue
        shape_ids = shapes.astype(int)
        if all(shape in evidence for shape in shape_ids):
            trials.append(Trial(
                shape_ids,
                np.cumsum([evidence[shape] for shape in shape_ids]),
                int(choice),
                int(length),
            ))
    return trials


def extract_decided_arrays(trials):
    trials = [trial for trial in trials if trial.choice is not None]
    num_shapes = np.asarray([trial.decision_step for trial in trials])
    evidence = np.asarray(
        [trial.cum_loglr_path[length - 1] for trial, length in zip(trials, num_shapes)]
    )
    choices = np.asarray([trial.choice for trial in trials])
    prefixes = [trial.shape_ids[:length] for trial, length in zip(trials, num_shapes)]
    return num_shapes, evidence, choices, prefixes


def _histc(values, edges):
    counts = np.zeros(len(edges), dtype=int)
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = len(edges) - 1
    np.add.at(counts, indices[(indices >= 0) & (indices < len(edges))], 1)
    return counts


def compute_num_shapes_hist(num_shapes):
    x = np.arange(1, np.max(num_shapes) + 1)
    return {"x": x, "count": _histc(num_shapes, x)}


def compute_psychometric(evidence, choices, monkey_id):
    evidence10 = np.rint(evidence * 10).astype(int)
    x = np.arange(np.floor(evidence10.min()), np.ceil(evidence10.max()) + 1).astype(int)
    counts = _histc(evidence10, x)
    choice_a = _histc(evidence10[choices == 2], x)
    with np.errstate(divide="ignore", invalid="ignore"):
        probability = choice_a / counts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        fit = sm.GLM(
            (choices == 2).astype(float),
            sm.add_constant(evidence10),
            family=sm.families.Binomial(),
        ).fit()
    coefficients = fit.params / np.log(10)
    prediction = 1 / (1 + 10 ** -(coefficients[1] * x + coefficients[0]))
    return {"x": x / 10, "count": counts, "p_choice_a": probability, "p_fit": prediction}


def compute_subjective_weight(prefixes, choices, monkey_id):
    counts = np.asarray(
        [
            [np.sum(prefix == shape) for shape in [4, 6, 8, 10, 9, 7, 5, 3]]
            for prefix in prefixes
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        fit = sm.GLM(
            (choices == 2).astype(float),
            counts,
            family=sm.families.Binomial(),
        ).fit()
    return {
        "x": np.array([-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9]),
        "coef": fit.params / np.log(10),
        "se": fit.bse / np.log(10),
    }


def _decision_threshold(trials, max_timestep=10, min_count=5):
    rows = {1: [], 2: []}
    for trial in trials:
        if trial.choice is not None and trial.decision_step <= max_timestep:
            rows[trial.choice].append(
                (trial.decision_step, trial.cum_loglr_path[trial.decision_step - 1])
            )
    output = {}
    for choice, values in rows.items():
        result = []
        for timestep in range(1, max_timestep + 1):
            evidence = np.asarray([value for t, value in values if t == timestep])
            if len(evidence) >= min_count:
                result.append((timestep, evidence.mean(), evidence.std(ddof=1)))
        output[choice] = np.asarray(result)
    return output


def plot_decision_threshold_distribution(
    monkey_trials, monkey_id, out_path, max_timestep=10, min_count=5
):
    """Plot monkey evidence distributions before sampling or choosing."""
    set_style()
    y_lim = (-4, 4) if monkey_id == 1 else (-2.5, 2.5)
    evidence_grid = np.arange(y_lim[0], y_lim[1] + 0.1, 0.2)
    colors = {"sampling": "green", 1: "blue", 2: "red"}
    kernel = np.array([1, 4, 6, 4, 1]) / 16
    fig, axes = plt.subplots(
        1, max_timestep, figsize=(5, 3), sharey=True,
    )
    axes = np.atleast_1d(axes)

    for timestep, ax in enumerate(axes, start=1):
        values = {"sampling": [], 1: [], 2: []}
        for trial in monkey_trials:
            if trial.decision_step is None or trial.decision_step < timestep:
                continue
            action = "sampling" if trial.decision_step > timestep else trial.choice
            if action in values:
                values[action].append(trial.cum_loglr_path[timestep - 1])

        max_count = 0
        for action in ("sampling", 1, 2):
            evidence = np.asarray(values[action])
            if len(evidence) <= min_count:
                continue
            indices = np.rint((evidence - evidence_grid[0]) / 0.2).astype(int)
            indices = indices[(indices >= 0) & (indices < len(evidence_grid))]
            counts = np.bincount(indices, minlength=len(evidence_grid))
            counts = np.convolve(counts, kernel, mode="same")
            max_count = max(max_count, counts.max())
            ax.fill_betweenx(
                evidence_grid, 0, counts,
                color=colors[action], alpha=0.3,
            )
            if action != "sampling":
                ax.axhline(evidence.mean(), color=colors[action], lw=2)

        ax.set(
            xlabel=str(timestep), xlim=(0, max_count * 1.05 if max_count else 1),
            ylim=y_lim, xticks=[],
        )
        ax.tick_params(axis="y", labelleft=timestep == 1)

    axes[0].set_ylabel("Cumulative Evidence")
    axes[0].set_yticks(np.arange(np.ceil(y_lim[0]), y_lim[1] + 0.1, 2))
    fig.supxlabel("Timestep", y=0.03)
    fig.subplots_adjust(left=0.14, bottom=0.2, right=0.99, top=0.98, wspace=0.08)
    _save(fig, out_path)


def plot_behavior(monkey_trials, model_trials, monkey_id, out_path, title=None):
    # Shared setup
    set_style()
    monkey = extract_decided_arrays(monkey_trials)
    model = extract_decided_arrays(model_trials)
    monkey_label = f"Monkey {'E' if monkey_id == 1 else 'J'}"
    model_label = "tinyRNN"
    fig, axes = plt.subplots(
        1, 4, figsize=(12, 2.55), gridspec_kw={"width_ratios": [1.1, 1.1, 1.1, 1.5]}
    )
    for ax in axes[:3]:
        ax.set_box_aspect(0.88)
    axes[3].set_box_aspect(0.55)

    # Panel A: number of shapes used
    histograms = [
        compute_num_shapes_hist(monkey[0]),
        compute_num_shapes_hist(model[0]),
    ]
    max_shapes = max(histogram["x"].max() for histogram in histograms)
    for histogram, color, label, style in zip(
        histograms, ["black", "red"], [monkey_label, model_label], ["bar", "line"]
    ):
        counts = np.zeros(max_shapes)
        counts[histogram["x"] - 1] = histogram["count"]
        frequency = counts / counts.sum()
        if style == "bar":
            axes[0].bar(
                range(1, max_shapes + 1), frequency,
                color=color, alpha=0.22, label=label,
            )
        else:
            axes[0].plot(
                range(1, max_shapes + 1), frequency, "-o",
                color=color, label=label, ms=3,
            )
    axes[0].set(
        xlabel="Number of shapes used", ylabel="Frequency", xlim=(0.5, 15.5)
    )

    # Panel B: psychometric curve
    for values, color, label in zip(
        [monkey, model], ["black", "red"], [monkey_label, model_label]
    ):
        psychometric = compute_psychometric(values[1], values[2], monkey_id)
        axes[1].plot(
            psychometric["x"], psychometric["p_choice_a"], "o",
            color=color, ms=2, label=label,
        )
        axes[1].plot(psychometric["x"], psychometric["p_fit"], color=color)
    axes[1].set(
        xlabel="Cumulative Evidence (LLR)", ylabel="Probability of choice A",
        xlim=(-3.1, 3.1), ylim=(-0.02, 1.02), xticks=[-3, -2, -1, 0, 1, 2, 3],
    )

    # Panel C: subjective shape weights
    for values, color in zip([monkey, model], ["black", "red"]):
        weights = compute_subjective_weight(values[3], values[2], monkey_id)
        axes[2].errorbar(
            weights["x"], weights["coef"], yerr=weights["se"],
            fmt="o-", color=color, ms=3, capsize=3,
        )
    axes[2].set(
        xlabel="True weight (LLR)", ylabel="Subjective weight (LLR)",
        ylim=(-5, 5), xlim=(-1.05, 1.05),
    )

    # Panel D: decision boundaries
    for trials, color, label, marker in zip(
        [monkey_trials, model_trials],
        ["black", "red"],
        [monkey_label, model_label],
        ["o", "s"],
    ):
        for choice, values in _decision_threshold(trials).items():
            if len(values):
                axes[3].errorbar(
                    values[:, 0], values[:, 1], yerr=values[:, 2],
                    fmt=marker + "-", color=color, ms=3,
                    label=label if choice == 2 else None,
                )
    axes[3].axhline(0, color="0.45", ls=":")
    axes[3].set(
        xlabel="Decision timestep", ylabel="Decision threshold (LLR)",
        xlim=(0.5, 10.5), ylim=(-3, 3), yticks=[-3, -2, -1, 0, 1, 2, 3],
    )

    # Shared legend and output
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center", ncol=2, frameon=False,
    )
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_path)
