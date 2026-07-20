import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from collections import Counter
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm, to_rgba
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
import seaborn as sns
from sklearn.decomposition import PCA

def set_style():
    """
    Adjusts matplotlib figure and axes style.

    Parameters:
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes or np.ndarray of Axes
    """

    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.labelsize': 10,
        'axes.titlesize': 8,
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
        'legend.loc': 'upper right'
    })


def add_info(df, target_log_ratios):
    
    # decision time
    episode_lengths = df.groupby('episode').size().reset_index(name='decision_time')
    df = df.merge(episode_lengths, on='episode', how='left')
    df["decision_time"] = df["decision_time"] - 1
    df["action"] = df["action"].astype(int)

    # timesteps
    df['time_step'] = df['stimuli_so_far'].apply(len)

    # evidence
    df["evidences"] = df.apply(lambda row: [target_log_ratios[s] for s in row["stimuli_so_far"][:row["decision_time"]]], axis=1)
    df["evidence_sum"] = df["evidences"].apply(sum)

    # policy
    df['p_chooseA'] = df['policy'].apply(lambda p: p[0])
    df['p_chooseB'] = df['policy'].apply(lambda p: p[1])
    df['p_sample'] = df['policy'].apply(lambda p: p[2])
    df['log_ratio'] = np.log10(df['p_chooseA'] / df['p_chooseB'].replace(0, np.nan)) # replace 0 with nan to avoid log(0)

    # calculate the accuracy
    correct_data = df[df["correct_answer"] == df["action"]].copy()

    print('total correct trials:', len(correct_data))
    correct_data_0 = correct_data[correct_data["correct_answer"] == 0]
    correct_data_1 = correct_data[correct_data["correct_answer"] == 1]
    print('correct trials for A:', len(correct_data_0))
    print('correct trials for B:', len(correct_data_1))
    
    # calculate the averagereward
    reward_sum = df['reward'].sum()
    print("Sum of reward column:", reward_sum/df['episode'].nunique())
    
    return df



def plot_decision_time_distribution(df, xlim=(0, 16), savePNG=True, saveSVG=False, save_path=None):
    df_decision = df[df["action"] != 2].dropna(subset=["decision_time"])
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.hist(df_decision["decision_time"], 
            bins=range(1, int(df_decision["decision_time"].max()) + 2),
            edgecolor='white', align='left', color = 'black', alpha = 1.0)
    
    # ax.vlines(10, 0, 8000, color = 'red', linestyle = '--', linewidth = 0.5)
    ax.set_xlabel("Number of Shapes (N*)")
    ax.set_ylabel("Frequency")
    # ax.set_title("Distribution of Decision Times")
    ax.set_xlim(xlim)
    ax.set_xticks(range(xlim[0], xlim[1], 3))
    # ax.set_yticks(range(0, 8200, 2000))

    if saveSVG:
        plt.savefig(save_path + 'dt_dist.svg', dpi=300)
    if savePNG:
        plt.savefig(save_path + 'dt_dist.png', dpi=300)
    plt.show()

    # count how many trials decision time is above 10
    print('total trials:', len(df_decision))
    print('trials with decision time above 10:', len(df_decision[df_decision["decision_time"] > 10]))


def plot_num_shapes_frequency_by_choice(
    df,
    max_timestep=10,
    correct_only=True,
    figsize=(2.85, 2.5),
    savePNG=True,
    saveSVG=False,
    save_path=None,
):
    """
    Plot the frequency distribution of decision time / number of sampled shapes
    for the task-optimized RNN, split by final choice.

    Parameters:
        df : pd.DataFrame
            Dataframe returned by the task analysis pipeline. It should already
            contain one final decision row per episode with `action != 2` and a
            `decision_time` column (for example after `add_info`).
        max_timestep : int
            Maximum number of sampled shapes (N*) to include on the x-axis.
        correct_only : bool
            If True, keep only trials where the final choice matches the
            correct answer.
    Returns:
        pd.DataFrame with columns ["action", "decision_time", "count"].
    """

    df_decision = df[df["action"] != 2].dropna(subset=["decision_time"]).copy()
    if correct_only:
        df_decision = df_decision[df_decision["action"] == df_decision["correct_answer"]].copy()

    if df_decision.empty:
        return pd.DataFrame(columns=["action", "decision_time", "count"])

    df_decision["decision_time"] = df_decision["decision_time"].astype(int)
    df_decision["action"] = df_decision["action"].astype(int)
    df_decision = df_decision[
        (df_decision["decision_time"] >= 1) & (df_decision["decision_time"] <= max_timestep)
    ].copy()

    if df_decision.empty:
        return pd.DataFrame(columns=["action", "decision_time", "count"])

    counts = (
        df_decision.groupby(["action", "decision_time"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )

    x = np.arange(1, max_timestep + 1, dtype=int)
    choose_a = (
        counts[counts["action"] == 0]
        .set_index("decision_time")["count"]
        .reindex(x)
        .fillna(0)
        .to_numpy()
    )
    choose_b = (
        counts[counts["action"] == 1]
        .set_index("decision_time")["count"]
        .reindex(x)
        .fillna(0)
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, choose_a, width=0.8, bottom=0, color="blue", alpha=0.35, label="Choose A")
    ax.bar(x, -choose_b, width=0.8, bottom=0, color="red", alpha=0.35, label="Choose B")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlim(0.5, max_timestep + 0.5)
    ax.set_xticks(range(0, max_timestep + 1, 3))
    ax.set_xlabel("Number of shapes (N*)")
    ax.set_ylabel("Trial count")
    ax.legend()
    plt.tight_layout()

    prefix = save_path or ""
    suffix = "correct_only" if correct_only else "all_trials"
    if savePNG:
        plt.savefig(prefix + f"numshapes_frequency_by_choice_{suffix}.png", dpi=300)
    if saveSVG:
        plt.savefig(prefix + f"numshapes_frequency_by_choice_{suffix}.svg", dpi=300)
    plt.show()

    return counts


def plot_decision_threshold(df, figsize=(3.85, 2.5), ylim=(-3, 3), savePNG=True, saveSVG=False, save_path=None):

    df_decision = df[df["action"] != 2].dropna(subset=["decision_time"])
    # remove trials when agent responded incorrectly
    correct_data = df_decision[df_decision["correct_answer"] == df_decision["action"]].copy()

    # separate into groups based on correct_answer value
    correct_data_0 = correct_data[correct_data["correct_answer"] == 0]
    correct_data_1 = correct_data[correct_data["correct_answer"] == 1]

    grouped0_stats = correct_data_0.groupby("decision_time")["evidence_sum"].agg(["mean", "std", "count"])
    grouped1_stats = correct_data_1.groupby("decision_time")["evidence_sum"].agg(["mean", "std", "count"])

    # remove the decision time where the count is less than 5
    # grouped0_stats = grouped0_stats[grouped0_stats["count"] >= 10]
    # grouped1_stats = grouped1_stats[grouped1_stats["count"] >= 10]

    # remove the decision time more than 20
    grouped0_stats = grouped0_stats[grouped0_stats.index <= 15]
    grouped1_stats = grouped1_stats[grouped1_stats.index <= 15]

    grouped0_stats = grouped0_stats[grouped0_stats.index > 0]
    grouped1_stats = grouped1_stats[grouped1_stats.index > 0]

    # compute 95% confidence intervals
    grouped0_stats["ci95"] = 1.96 * grouped0_stats["std"] / np.sqrt(grouped0_stats["count"])
    grouped1_stats["ci95"] = 1.96 * grouped1_stats["std"] / np.sqrt(grouped1_stats["count"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(grouped0_stats.index, grouped0_stats["mean"], 
                yerr=grouped0_stats["std"], fmt='o-', capsize=5, label="Choose A", color='blue', markersize=5, elinewidth=2, linewidth=2)
    ax.errorbar(grouped1_stats.index, grouped1_stats["mean"], 
                yerr=grouped1_stats["std"], fmt='s-', capsize=5, label="Choose B", color='red', markersize=5, elinewidth=2, linewidth=2)
    ax.hlines(0, 0, 10, color='black', linestyle='--', linewidth=0.5)
    ax.vlines(10, -3, 3, color = 'black', linestyle = '--', linewidth = 0.5)
    ax.set_xlim(0, 15.5)
    ax.set_xticks(range(0, 16, 3))
    ax.set_ylim(ylim)
    ax.set_xlabel("Decision Time")
    ax.set_ylabel("Cumulative Evidence")
    # ax.set_title("Decision Boundary vs Decision Time")
    ax.legend()
    # ax.grid(True)
    
    if saveSVG:
        plt.savefig(save_path + 'logLR_bound.svg', dpi=300)
    if savePNG:
        plt.savefig(save_path + 'logLR_bound.png', dpi=300)
    plt.show()


def plot_psychometric_curve(df, bin_start=-3, bin_end=3.1, bin_width=0.1, savePNG=True, saveSVG=False, save_path=None):

    df_decision = df[df["action"] != 2].dropna(subset=["decision_time"])

    psychometric_data = df_decision.groupby(['episode'], sort=False).last().reset_index()

    psychometric_data["evidence_sum_binned"] = pd.cut(
        psychometric_data["evidence_sum"],
        bins=np.arange(bin_start, bin_end, bin_width),
        labels=np.arange(bin_start + bin_width/2, bin_end - bin_width/2, bin_width),
        include_lowest=True,
    )

    psychometric_data = (
        psychometric_data
        .dropna(subset=["evidence_sum_binned"])
        .groupby("evidence_sum_binned", observed=True)["action"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.scatter(psychometric_data["evidence_sum_binned"], psychometric_data["action"], s = 10, color = 'black')
    # ax.set_xlim(-3, 3.1)
    # ax.set_xticks(np.arange(-3, 3.1, 1))
    ax.set_xlabel("Cumulative Evidence")
    ax.set_ylabel("Choice Probability")
    # ax.set_title("Choice Probability vs Accumulated Evidence")

    # Fit sigmoid: y = L / (1 + exp(-k*(x - x0))) + b
    def sigmoid(x, L, k, x0, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    x_data = psychometric_data["evidence_sum_binned"].astype(float).values
    y_data = psychometric_data["action"].values
    try:
        p0 = [y_data.max() - y_data.min(), 1.0, 0.0, y_data.min()]
        popt, _ = curve_fit(sigmoid, x_data, y_data, p0=p0, maxfev=5000)
        x_smooth = np.linspace(bin_start, bin_end - bin_width, 200)
        ax.plot(x_smooth, sigmoid(x_smooth, *popt), color='black', linestyle='--', alpha=0.8)
    except Exception:
        pass

    if saveSVG:
        plt.savefig(save_path + 'psychMetric.svg', dpi=300)
    if savePNG:
        plt.savefig(save_path + 'psychMetric.png', dpi=300)
    plt.show()


def plot_decision_accuracy(df, bin_width=0.2, savePNG=True, saveSVG=False, save_path=None):

    df_decision = df[df["action"] != 2].dropna(subset=["decision_time"])
    # Calculate accuracy for each row: 1 if action == correct_answer, else 0
    df_decision["accuracy"] = (df_decision["action"] == df_decision["correct_answer"]).astype(float)

    # Bin absolute cumulative evidence (logLRs)
    abs_evidence = df_decision["evidence_sum"].abs()
    bin_edges = np.arange(0, abs_evidence.max() + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width/2

    df_decision["abs_evidence_bin"] = pd.cut(
        abs_evidence,
        bins=bin_edges,
        labels=bin_centers,
        include_lowest=False
    )

    mean_logLR_by_bin = (
        df_decision
        .dropna(subset=["abs_evidence_bin"])
        .groupby("abs_evidence_bin", observed=True)["evidence_sum"]
        .apply(lambda x: x.abs().mean())
        .reset_index()
        .rename(columns={"evidence_sum": "mean_abs_logLR"})
    )

    mean_map = (
        df_decision
        .dropna(subset=["abs_evidence_bin"])
        .groupby("abs_evidence_bin", observed=True)["evidence_sum"]
        .apply(lambda x: x.abs().mean())
    )

    df_decision["mean_abs_logLR"] = df_decision["abs_evidence_bin"].map(mean_map)

    accuracy_by_bin = (
        df_decision
        .dropna(subset=["mean_abs_logLR"])
        .groupby("mean_abs_logLR", observed=True)["accuracy"]
        .mean()
        .reset_index()
    )


    def theoretical_accuracy(x):
        return 1 / (1 + 10 ** (-x))

    x = np.linspace(0, 4, 100)
    y = theoretical_accuracy(x)
    # Plot
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(
        accuracy_by_bin["mean_abs_logLR"].astype(float),
        accuracy_by_bin["accuracy"],
        marker='o',
        color='black',
        linestyle='-',
        markersize=3
    )
    ax.plot(x, y, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Absolute Cumulative Evidence (|logLRs|)")
    ax.set_ylabel("Accuracy")
    # ax.set_title("Accuracy vs. |Cumulative Evidence|")

    # ax.set_xlim(0, 3.1)
    # ax.set_xticks(np.arange(0, 3.1, 1))
    ax.set_ylim(0.49, 1.05)
    ax.set_yticks(np.arange(0.5, 1.05, 0.1))

    plt.tight_layout()
    if saveSVG:
        plt.savefig(save_path + 'accuracy_vs_abs_logLR.svg', dpi=300)
    if savePNG:
        plt.savefig(save_path + 'accuracy_vs_abs_logLR.png', dpi=300)
    
    plt.show()


def plot_decision_weights(df, target_log_ratios, savePNG=True, saveSVG=False, save_path=None):
    df_last = df.groupby("episode").last()
    stimuli_counts = df_last["stimuli_so_far"].apply(Counter)
    stimuli_df = pd.DataFrame.from_records(stimuli_counts).fillna(0).astype(int)

    stimuli_df = stimuli_df[sorted(stimuli_df.columns)]
    stimuli_df["action"] = df_last["action"]

    # X is the features, y is the target
    X = stimuli_df.drop(columns=['action'])
    y = stimuli_df['action']

    # fit the logistic regression model
    model = LogisticRegression()
    model.fit(X, y)

    # get the coefficients and intercept
    coef_natural = pd.Series(model.coef_[0], index=X.columns)
    intercept_natural = model.intercept_[0]

    # sort the coefficients by stimulus value
    coef_natural = coef_natural.sort_index()

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(target_log_ratios, coef_natural, 'o-', color = 'black', markersize = 4)
    ax.vlines(0, -8, 8, color='black', linestyle='--', linewidth = 0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xticks(np.arange(-1.0, 1.1, 0.5))
    ax.set_xlabel("Stimuli logLR")
    ax.set_ylabel(r"Decision weight ($\beta_i$)")
    # ax.set_title("Subjective Weights of stimulus")
    # ax.grid(True)

    if saveSVG:
        plt.savefig(save_path + 'subjWeight.svg', dpi=300)
    if savePNG:
        plt.savefig(save_path + 'subjWeight.png', dpi=300)
    plt.show()


def plot_psample(df, figsize=(2.85, 2.5), max_samples=20, bin_width=0.05, bin_start=-3, bin_end=3.1, xlim=(-3.1, 3.1), savePNG=True, saveSVG=False, save_path=None):

    df_decision = df[df["action"] != 2].dropna(subset=["decision_time"])

    df_decision["evidence_sum_bin"] = pd.cut(
        df_decision["evidence_sum"], 
        bins=np.arange(bin_start, bin_end, bin_width), 
        include_lowest=True
    )

    df_binned = (
        df_decision.dropna(subset=['evidence_sum_bin'])
        .groupby(['decision_time', 'evidence_sum_bin'], observed=True)['p_sample']
        .mean()
        .reset_index()
    )
    df_binned['bin_center'] = df_binned['evidence_sum_bin'].apply(lambda iv: iv.mid)

    dt_values = np.arange(1, max_samples)
    cmap = cm.plasma
    norm = Normalize(vmin=min(dt_values), vmax=max(dt_values))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    df_binned['bin_center'] = df_binned['evidence_sum_bin'].apply(lambda iv: (iv.left + iv.right) / 2).astype(float)
    df_binned = df_binned[df_binned['bin_center'].between(-3, 3)]
    # df_binned = df_binned[df_binned['decision_time'] <= 15]

    fig, ax = plt.subplots(figsize=figsize)
    for dt in dt_values:

        subset = df_binned[df_binned['decision_time'] == dt]
        ax.plot(subset['bin_center'],
                subset['p_sample'],
                color=cmap(norm(dt)),
                label=f'dt={dt}')

    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Cumulative Evidence')
    ax.set_ylabel('Probability of sampling')
    # ax.set_title('Sampling Probability vs Cumulative Evidence by Decision Time')
    ax.grid(False)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(xlim[0] + 0.1, xlim[1] - 0.1, 1))
    # ax.set_aspect('equal', adjustable='box')

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Timestep')

    # ax.legend(title='decision_time', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if saveSVG:
        plt.savefig(save_path + 'sampling_probability_vs_evidence_sum_softBound.svg', dpi=300)
    if savePNG:
        plt.savefig(save_path + 'sampling_probability_vs_evidence_sum_softBound.png', dpi=300)
    plt.show()


def plot_psample_flip(
    df,
    figsize=(2.05, 2.55),
    max_samples=20,
    bin_width=0.05,
    bin_start=-3,
    bin_end=3.1,
    xlim=(-0.05, 1.05),
    ylim=None,
    savePNG=True,
    saveSVG=False,
    save_path=None,
):

    df_decision = df[df["action"] != 2].dropna(subset=["decision_time"]).copy()
    if ylim is None:
        if xlim is not None and (xlim[0] < -0.05 or xlim[1] > 1.05):
            ylim = xlim
            xlim = (-0.05, 1.05)
        else:
            ylim = (bin_start, bin_end - bin_width)

    df_decision["evidence_sum_bin"] = pd.cut(
        df_decision["evidence_sum"],
        bins=np.arange(bin_start, bin_end, bin_width),
        include_lowest=True
    )

    df_binned = (
        df_decision.dropna(subset=['evidence_sum_bin'])
        .groupby(['decision_time', 'evidence_sum_bin'], observed=True)['p_sample']
        .mean()
        .reset_index()
    )

    dt_values = np.arange(1, max_samples)
    time_colors = plt.cm.plasma(np.linspace(0.1, 0.95, len(dt_values)))

    df_binned['bin_center'] = df_binned['evidence_sum_bin'].apply(lambda iv: (iv.left + iv.right) / 2).astype(float)
    df_binned = df_binned[df_binned['bin_center'].between(ylim[0], ylim[1])]

    fig, ax = plt.subplots(figsize=figsize)
    for idx, dt in enumerate(dt_values):

        subset = df_binned[df_binned['decision_time'] == dt]
        ax.plot(subset['p_sample'],
                subset['bin_center'],
                color=time_colors[idx],
                linewidth=1.6,
                label=f't={dt}')

    ax.set_xlim(xlim)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$p(\mathrm{sample})$", fontsize=10)
    ax.set_ylabel("Cumulative Evidence", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(False)

    sm = plt.cm.ScalarMappable(
        cmap="plasma",
        norm=plt.Normalize(vmin=1, vmax=max_samples - 1),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.08, pad=0.08)
    cbar.set_label("Time step", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()

    prefix = save_path or ""
    if saveSVG:
        plt.savefig(prefix + 'sampling_probability_vs_evidence_sum_softBound_flip.svg', dpi=300)
    if savePNG:
        plt.savefig(prefix + 'sampling_probability_vs_evidence_sum_softBound_flip.png', dpi=300)
    plt.show()


# def plot_decision_threshold_distribution(df, target_log_ratios, max_timestep=15, figsize=(5, 3), savePNG=True, saveSVG=False, save_path=None):

#     fig, axes = plt.subplots(1, max_timestep, figsize=figsize, sharey=True)
#     if max_timestep == 1:
#         axes = [axes]

#     df["evidences"] = df.apply(
#         lambda row: [target_log_ratios[s] for s in row["stimuli_so_far"][:row["decision_time"]]],
#         axis=1
#     )
#     df["evidence_sum"] = df["evidences"].apply(sum)

#     colors = {0:'blue', 1:'red', 2:'green'}
#     labels = {0:'Chosen A', 1:'Chosen B', 2:'Sampling'}

#     for t in range(max_timestep):
#         ax = axes[t]

#         timestep_data = df[df['time_step'] == t+1]

#         for action in [0,1,2]:
#             action_data = timestep_data[timestep_data['action'] == action]
#             if action_data.empty:
#                 continue

#             ev = action_data['evidence_sum'].round(10).to_numpy()
#             uniq_vals, counts = np.unique(ev, return_counts=True)

#             ax.fill_betweenx(uniq_vals, 0, counts,
#                             color=colors[action], alpha=0.3,
#                             label=labels[action] if t==0 else None)

#             if action in [0,1]:
#                 mean_ev = action_data['evidence_sum'].mean()
#                 ax.axhline(mean_ev, color=colors[action], linewidth=2)

#         ax.set_xticks([0, max(counts)]) 
#         ax.set_xlabel(str(t+1))
#         ax.set_xlim(0, None)
#         # ax.set_ylim(-2.2, 2.2)
#         ax.set_ylim(-2.0, 2.0)

#     fig.text(0.54, 0.03, 'Timestep', ha='center', fontsize=10)
#     fig.text(0.1, 0.5, 'Cumulative Evidence', va='center', rotation='vertical', fontsize=10)

#     for ax in axes:
#         ax.set_xticks([])
#         ax.label_outer()

#     axes[0].legend(loc='best')

#     # plt.tight_layout(rect=[0.05, 0.05, 1, 1])
#     plt.tight_layout()
#     plt.rcParams.update({
#         'axes.spines.top': True,
#         'axes.spines.right': True,
#     })

#     fig.subplots_adjust(wspace=0.00)

#     if savePNG:
#         plt.savefig(save_path + 'decision_threshold_distribution.png', dpi=300)
#     if saveSVG:
#         plt.savefig(save_path + 'decision_threshold_distribution.svg', dpi=300)
    
#     plt.show()


import numpy as np
import matplotlib.pyplot as plt

def _compute_evidence_distribution(
    ev,
    distribution,
    y_min,
    y_max,
    kernel_sigma=1.0,
    kernel_bin_width=0.05,
):
    ev = np.asarray(ev, dtype=float)
    if ev.size == 0:
        return np.array([]), np.array([])

    if distribution == "discrete":
        return np.unique(np.round(ev, 10), return_counts=True)

    if distribution != "kernel":
        raise ValueError(
            f"distribution must be 'discrete' or 'kernel', got {distribution!r}"
        )

    edges = np.arange(y_min, y_max + kernel_bin_width, kernel_bin_width)
    counts, edges = np.histogram(ev, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    if kernel_sigma > 0:
        counts = gaussian_filter(
            counts.astype(float),
            sigma=kernel_sigma,
            mode="nearest",
        )
    return centers, counts


def plot_decision_threshold_distribution(
    df,
    target_log_ratios,
    max_timestep=15,
    figsize=(4.85, 2.05),
    savePNG=False,
    saveSVG=True,
    save_path="",
    filename="Ideal_Infinite_Distribution",
    y_min=-2.0,
    y_max=2.0,
    lower_log_boundary=None,
    upper_log_boundary=None,
    distribution="discrete",
    kernel_sigma=1.0,
    kernel_bin_width=0.05,
):
    fig_dist, dist_axes = plt.subplots(
        1,
        max_timestep,
        figsize=figsize,
        sharey=True
    )

    if max_timestep == 1:
        dist_axes = [dist_axes]

    fig_dist.subplots_adjust(wspace=0.02)

    df = df.copy()

    df["evidences"] = df.apply(
        lambda row: [
            target_log_ratios[s]
            for s in row["stimuli_so_far"][:row["decision_time"]]
        ],
        axis=1
    )

    df["evidence_sum"] = df["evidences"].apply(sum)

    colors = {
        0: "blue",
        1: "red",
        2: "green"
    }

    labels = {
        0: "Chosen A",
        1: "Chosen B",
        2: "Sampling"
    }

    for t in range(1, max_timestep + 1):

        ax = dist_axes[t - 1]
        timestep_data = df[df["time_step"] == t]

        max_count = 0.0

        for action in [0, 1, 2]:

            action_data = timestep_data[
                timestep_data["action"] == action
            ]

            if action_data.empty:
                continue

            ev = action_data["evidence_sum"].to_numpy()

            y_vals, counts = _compute_evidence_distribution(
                ev,
                distribution=distribution,
                y_min=y_min,
                y_max=y_max,
                kernel_sigma=kernel_sigma,
                kernel_bin_width=kernel_bin_width,
            )

            if y_vals.size == 0:
                continue

            max_count = max(
                max_count,
                counts.max()
            )

            fill_kwargs = {
                "color": colors[action],
                "alpha": 0.3,
                "label": labels[action] if t == 1 else None,
                "linewidth": 0,
            }
            if distribution == "discrete":
                fill_kwargs["step"] = "mid"

            ax.fill_betweenx(
                y_vals,
                0,
                counts,
                **fill_kwargs,
            )

            if action in (0, 1):
                mean_ev = action_data["evidence_sum"].mean()
                ax.axhline(mean_ev, color=colors[action], linewidth=1)

        if lower_log_boundary is not None:
            ax.axhline(
                lower_log_boundary,
                color="black",
                linestyle=":",
                linewidth=1
            )

        if upper_log_boundary is not None:
            ax.axhline(
                upper_log_boundary,
                color="black",
                linestyle=":",
                linewidth=1
            )

        ax.set_ylim(y_min, y_max)

        if max_count > 0:
            ax.set_xlim(0, max_count * 1.05)
        else:
            ax.set_xlim(0, 1)

        ax.set_xticks([])
        ax.set_xlabel(str(t), fontsize=8)
        ax.label_outer()

        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    fig_dist.text(
        0.54,
        0.03,
        "Timestep",
        ha="center",
        fontsize=8
    )

    fig_dist.text(
        0.04,
        0.5,
        "Cumulative evidence",
        va="center",
        rotation="vertical",
        fontsize=8
    )

    dist_axes[0].legend(loc="best", fontsize=7)

    plt.tight_layout(rect=[0.05, 0.08, 1, 1])
    fig_dist.subplots_adjust(wspace=0.00)

    if save_path is None:
        save_path = ""

    if savePNG:
        plt.savefig(
            save_path + filename + ".png",
            dpi=300
        )

    if saveSVG:
        plt.savefig(
            save_path + filename + ".svg",
            dpi=300
        )

    plt.show()

def plot_psample_dt(df, dt, max_steps=20, bin_width=0.05, bin_start=-3, bin_end=3.1, savePNG=True, saveSVG=False, save_path=None):
    
    df_dt = df[df['decision_time'] == dt]

    df_dt["evidence_sum_bin"] = pd.cut(df_dt["evidence_sum"], bins=np.arange(bin_start, bin_end, bin_width), include_lowest=True)

    df_binned = (df_dt.dropna(subset=['evidence_sum_bin'])
        .groupby(['evidence_sum_bin'], observed=True)['p_sample']
        .mean()
        .reset_index()
    )

    df_binned['bin_center'] = df_binned['evidence_sum_bin'].apply(lambda iv: (iv.left + iv.right) / 2).astype(float)
    df_binned = df_binned[df_binned['bin_center'].between(-3, 3)]
    
    # Create color gradient from light to dark based on dt within max_steps range
    norm = Normalize(vmin=1, vmax=max_steps)
    color = cm.viridis(norm(dt))  # Use viridis colormap for gradient

    fig, ax = plt.subplots(figsize=(3.55, 3))
    ax.plot(df_binned['bin_center'],
            df_binned['p_sample'],
            marker='o',
            label=f'dt={dt}',
            color=color,
            markersize=5)

    ax.legend()
    ax.set_xlabel('Cumulative Evidence')
    ax.set_ylabel('Probability of sampling')
    # ax.set_title('Sampling Probability vs Cumulative Evidence by Decision Time')
    ax.grid(False)
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(np.arange(-3, 4, 1))

    # Add colorbar to show the gradient
    sm = ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=15, shrink=0.8)
    cbar.set_label('Decision Time', rotation=270, labelpad=8)
    cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
    cbar.set_ticks(np.arange(1, max_steps+1, 2))
    cbar.set_ticklabels([f'{i}' for i in np.arange(1, max_steps+1, 2)])

    plt.tight_layout()
    
    if savePNG:
        plt.savefig(save_path + f'psample_dt{dt}.png', dpi=300)
    if saveSVG:
        plt.savefig(save_path + f'psample_dt{dt}.svg', dpi=300)
    plt.show()

def regression_weights(df):

    decision_times = sorted(df['decision_time'].unique())
    df_decision = df[df['action'] == df['correct_answer']]

    results = {}

    for dt in decision_times:

        if dt > 10:
            continue
            
        # Filter data for this decision time
        dt_data = df_decision[df_decision['decision_time'] == dt].copy()
        
        if len(dt_data) < 10:  # Skip if too few samples
            print(f"  Skipping decision_time {dt}: insufficient data ({len(dt_data)} samples)")
            continue
            
        # Create feature matrix: each column represents a position in the sequence
        X = np.stack(dt_data['evidences'].values)
        y = np.where(dt_data['action'].values == 1, 1, 0)

        if X.shape[1] == 0:
            print(f"  Skipping decision_time {dt}: empty evidence sequences")
            continue
        
        # Check if we have variation in the target variable
        if len(np.unique(y)) < 2:
            print(f"  Skipping decision_time {dt}: no variation in decisions")
            continue
            
        # Fit logistic regression
        model = LogisticRegression(fit_intercept=True, max_iter=2000)
        model.fit(X, y)
        
        # Store results
        results[dt] = {
            'coefficients': model.coef_[0],
            'intercept': model.intercept_[0],
            'n_samples': len(dt_data),
            'positions': list(range(1, dt + 1)),
            'accuracy': model.score(X, y)
        }

    return results

def plot_decision_weights_heatmap(df, savePNG=True, saveSVG=False, save_path=None):
    """Create a heatmap showing decision weights across positions and decision times"""
    results = regression_weights(df)

    # Determine the maximum position across all decision times
    max_position = max(max(data['positions']) for data in results.values()) + 1
    decision_times = sorted(results.keys())
    
    # Create matrix for heatmap
    weight_matrix = np.full((len(decision_times), max_position), np.nan)
    
    for i, dt in enumerate(decision_times):    
        if dt in results:
            positions = results[dt]['positions']
            weights = results[dt]['coefficients']
            for pos, weight in zip(positions, weights):
                weight_matrix[i, pos] = weight
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(2.85, 2.5))
    
    # Create mask for missing values
    mask = np.isnan(weight_matrix)
    
    sns.heatmap(weight_matrix, 
                xticklabels=range(max_position),
                yticklabels=[f'DT = {dt}' for dt in decision_times],
                cmap='RdBu_r', center=0, 
                annot=False, fmt='.2f',
                mask=mask,
                cbar_kws={'label': 'Decision Weight'})
    
    ax.set_xlabel('Stimulus Position in Sequence')
    ax.set_ylabel('Decision Time')
    ax.set_xlim(1, 11)
    # plt.title('Decision Weights Heatmap: Position × Decision Time', fontsize=14)
    plt.tight_layout()

    if savePNG:
        plt.savefig(save_path + 'decisionWeights_heatmap_inf.png', dpi=300)
    if saveSVG:
        plt.savefig(save_path + 'decisionWeights_heatmap_inf.svg', dpi=300)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(2.5, 2.5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    max_position = 0
    handles = []
    labels = []
    
    for i, (dt, data) in enumerate(results.items()):
        positions = np.array(data['positions'])
        weights = np.array(data['coefficients'])
        
        min_len = min(len(positions), len(weights))
        positions = positions[:min_len]
        weights = weights[:min_len]
        
        line, = ax2.plot(positions, weights, 'o-', color=colors[i], 
                        label=f'RT = {dt} (n={data["n_samples"]})', 
                        linewidth=2, markersize=4)
        handles.append(line)
        labels.append(f'RT = {dt} (n={data["n_samples"]})')
        
        max_position = max(max_position, max(positions))
    
    ax2.set_xlabel('Stimuli Position in Sequence')
    ax2.set_ylabel(r'Decision Weight ($\beta_t$)')
    ax2.set_xticks(np.arange(0, 11, 2))
    ax2.set_xlim(-0.5, 10.5)
    plt.tight_layout()

    if saveSVG:
        plt.savefig(save_path + 'decisionWeights.svg', dpi=300)
    if savePNG:
        plt.savefig(save_path + 'decisionWeights.png', dpi=300)
    plt.show()

    fig_leg = plt.figure(figsize=(1, 2.5))
    fig_leg.legend(handles, labels, loc='center', frameon=False, fontsize=6)
    plt.axis('off')
    plt.tight_layout()

    if saveSVG:
        plt.savefig(save_path + 'decisionWeights_legend.svg', dpi=300)
    if savePNG:
        plt.savefig(save_path + 'decisionWeights_legend.png', dpi=300)
    plt.show()
    

def plot_scree_plot(df, savePNG=True, saveSVG=False, save_path=None, figsize=(2.5, 2.5)):
    
    D = len(df.hidden_state.iloc[0])
    hmat = np.vstack(df.hidden_state.values)
    X = hmat
    pca_full = PCA(n_components=10)
    pca_full.fit(X)

    Z = pca_full.fit_transform(X)
    df[['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10']] = Z

    ratios = pca_full.explained_variance_ratio_

    # ---- Scree plot ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(1, 10 + 1), ratios[:10], align='center', color='black')
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    # plt.title("Scree Plot")
    plt.xticks(np.arange(1, 10 + 1))

    if savePNG:
        plt.savefig(save_path + 'scree_plot.png', dpi=300)
    if saveSVG:
        plt.savefig(save_path + 'scree_plot.svg', dpi=300)
    plt.show()

    # ---- Cumulative explained variance ----
    cum_ratios = np.cumsum(ratios)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(1, 10 + 1), cum_ratios[:10], marker='o', color='black')
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    # ax.set_title("Cumulative Explained Variance")
    ax.set_xticks(np.arange(1, 10 + 1))
    ax.set_ylim(0,1.05)
    if savePNG:
        plt.savefig(save_path + 'cumulative_explained_variance.png', dpi=300)
    if saveSVG:
        plt.savefig(save_path + 'cumulative_explained_variance.svg', dpi=300)
    plt.show()

    # Step 2: Remove time_step == 0
    df_remove0 = df[df['time_step'] != 0].copy()

    # Step 3: Bin evidence_sum (you can change `n_bins` or use qcut for quantiles)
    n_bins = 20
    df_remove0['evidence_bin'] = pd.cut(df_remove0['evidence_sum'], bins=n_bins)

    # Step 4: Group by bins and average PCA activations
    grouped = df_remove0.groupby('evidence_bin')[['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10']].mean()

    # Compute bin centers for plotting on x-axis
    bin_centers = df_remove0.groupby('evidence_bin')['evidence_sum'].mean()

    # Step 5: Plot
    plt.figure(figsize=figsize)

    plt.plot(bin_centers, grouped['PCA1'], label='PC1', marker='o')
    plt.plot(bin_centers, grouped['PCA2'], label='PC2', marker='o')
    plt.plot(bin_centers, grouped['PCA3'], label='PC3', marker='o')
    plt.plot(bin_centers, grouped['PCA4'], label='PC4', marker='o')
    plt.plot(bin_centers, grouped['PCA5'], label='PC5', marker='o')
    # plt.plot(bin_centers, grouped['PCA6'], label='PC6', marker='o')
    # plt.plot(bin_centers, grouped['PCA7'], label='PC7', marker='o')
    # plt.plot(bin_centers, grouped['PCA8'], label='PC8', marker='o')
    # plt.plot(bin_centers, grouped['PCA9'], label='PC9', marker='o')
    # plt.plot(bin_centers, grouped['PCA10'], label='PC10', marker='o')

    plt.xlabel('Cumulative Evidence (logLR)')
    plt.ylabel('Mean PCA Activation')
    # plt.title('PCA Activation vs. Evidence (Binned)')
    # plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if savePNG:
        plt.savefig(save_path + 'pca_activation_vs_evidence.png', dpi=300)
    if saveSVG:
        plt.savefig(save_path + 'pca_activation_vs_evidence.svg', dpi=300)
    plt.show()


def _ensure_policy_columns(df):
    if {'p_chooseA', 'p_chooseB', 'p_sample'}.issubset(df.columns):
        return df

    df = df.copy()
    df['p_chooseA'] = df['policy'].apply(lambda p: p[0])
    df['p_chooseB'] = df['policy'].apply(lambda p: p[1])
    df['p_sample'] = df['policy'].apply(lambda p: p[2])
    return df


def _prepare_pca_policy_field(
    df,
    xlim=None,
    ylim=(-3.5, 3.5),
    max_decision_time=15,
    smooth_sigma=2.0,
):
    df = _ensure_policy_columns(df)
    field_df = df[df['decision_time'] <= max_decision_time].copy()
    field_df = field_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['PCA1', 'PCA2', 'p_chooseA', 'p_chooseB', 'p_sample']
    )

    if field_df.empty:
        raise ValueError("No valid rows available to infer the PCA policy field.")

    x = field_df['PCA2'].to_numpy(dtype=float)
    y = field_df['PCA1'].to_numpy(dtype=float)
    points = np.column_stack([x, y])

    x_pad = 0.05 * max(float(np.ptp(x)), 1e-6)
    y_pad = 0.05 * max(float(np.ptp(y)), 1e-6)
    x_limits = xlim if xlim is not None else (float(np.min(x) - x_pad), float(np.max(x) + x_pad))
    y_limits = ylim if ylim is not None else (float(np.min(y) - y_pad), float(np.max(y) + y_pad))
    x_grid = np.linspace(float(x_limits[0]), float(x_limits[1]), 220)
    y_grid = np.linspace(float(y_limits[0]), float(y_limits[1]), 220)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)

    def interpolate_prob(values):
        values = np.asarray(values, dtype=float)
        grid_linear = griddata(points, values, (grid_x, grid_y), method='linear')
        grid_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
        return np.where(np.isfinite(grid_linear), grid_linear, grid_nearest)

    p_choose_a = interpolate_prob(field_df['p_chooseA'].to_numpy(dtype=float))
    p_choose_b = interpolate_prob(field_df['p_chooseB'].to_numpy(dtype=float))
    p_sample = interpolate_prob(field_df['p_sample'].to_numpy(dtype=float))
    rgb = np.stack([p_choose_b, p_sample, p_choose_a], axis=-1)
    if smooth_sigma and smooth_sigma > 0:
        rgb = gaussian_filter(
            rgb,
            sigma=(smooth_sigma, smooth_sigma, 0),
            mode='reflect',
        )
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb, x_limits, y_limits


def plot_inferred_policy_in_pca_space(
    df,
    savePNG=True,
    saveSVG=False,
    save_path=None,
    flip_x_axis=False,
    xlim=None,
    ylim=(-3.5, 3.5),
    max_decision_time=15,
    smooth_sigma=2.0,
    figsize=(3.3, 2.7),
):
    set_style()

    rgb, x_limits, y_limits = _prepare_pca_policy_field(
        df,
        xlim=xlim,
        ylim=ylim,
        max_decision_time=max_decision_time,
        smooth_sigma=smooth_sigma,
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        rgb,
        extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
        origin='lower',
        interpolation='bicubic',
        alpha=0.62,
        zorder=0,
        aspect='auto',
    )

    action_cmap = ListedColormap(['red', 'green', 'blue'])
    action_norm = BoundaryNorm([0, 1, 2, 3], action_cmap.N)
    action_sm = ScalarMappable(cmap=action_cmap, norm=action_norm)
    action_sm.set_array([])

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    if flip_x_axis:
        ax.invert_xaxis()
    ax.set_xlabel('PCA Dimension 2')
    ax.set_ylabel('PCA Dimension 1')
    plt.tight_layout()

    prefix = save_path or ''
    if savePNG:
        plt.savefig(prefix + 'inferred_policy_pca_space.png', dpi=300)
    if saveSVG:
        plt.savefig(prefix + 'inferred_policy_pca_space.svg', dpi=300)
    plt.show()



def plot_example_pca_trajectories(
    df,
    savePNG=True,
    saveSVG=False,
    save_path=None,
    trial_length=None,
    n=None,
    n_trajectories=20,
    n_examples=None,
    random_state=80,
    flip_x_axis=False,
    xlim=None,
    ylim=(-3.5, 3.5),
    max_decision_time=15,
    figsize=(2.85, 2.5),
):
    set_style()

    # Only include episodes that made the decision within the requested time window.
    eligible_df = df[df['decision_time'] <= max_decision_time].copy()
    eligible_episodes = eligible_df['episode'].drop_duplicates()
    if len(eligible_episodes) == 0:
        print("No eligible episodes found for plot_example_pca_trajectories.")
        return

    # Optionally restrict to one example trial with an exact decision time.
    if trial_length is not None:
        eligible_exact = eligible_df[eligible_df['decision_time'] == trial_length]['episode'].drop_duplicates()
        if len(eligible_exact) == 0:
            raise ValueError(f"No eligible episodes found with decision_time == {trial_length}.")
        sample_trials = eligible_exact.sample(n=1, random_state=random_state)
    else:
        if n is not None:
            n_trajectories = n
        elif n_examples is not None:
            n_trajectories = n_examples
        n_trajectories = min(int(n_trajectories), len(eligible_episodes))
        sample_trials = eligible_episodes.sample(n=n_trajectories, random_state=random_state)

    fig, ax = plt.subplots(figsize=figsize)

    traj_df = eligible_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['PCA1', 'PCA2']).copy()
    if traj_df.empty:
        raise ValueError("No valid PCA coordinates available for plot_example_pca_trajectories.")

    x = traj_df['PCA2'].to_numpy(dtype=float)
    y = traj_df['PCA1'].to_numpy(dtype=float)
    x_pad = 0.05 * max(float(np.ptp(x)), 1e-6)
    y_pad = 0.05 * max(float(np.ptp(y)), 1e-6)
    x_limits = xlim if xlim is not None else (float(np.min(x) - x_pad), float(np.max(x) + x_pad))
    y_limits = ylim if ylim is not None else (float(np.min(y) - y_pad), float(np.max(y) + y_pad))

    # Same timestep across trajectories uses the same plasma color.
    sample_df = traj_df[traj_df['episode'].isin(sample_trials)].copy()
    timestep_min = max(1, int(sample_df['time_step'].min()))
    timestep_max = int(sample_df['time_step'].max())
    norm = plt.Normalize(timestep_min, timestep_max)
    cmap = cm.get_cmap('plasma')

    for ep in sample_trials:
        this_trial = sample_df[sample_df['episode'] == ep].sort_values('time_step')
        if this_trial.empty:
            continue
        ax.scatter(
            this_trial['PCA2'].iloc[0],
            this_trial['PCA1'].iloc[0],
            color='black',
            s=30,
            zorder=10,
        )
        for idx in range(1, len(this_trial)):
            prev_row = this_trial.iloc[idx - 1]
            curr_row = this_trial.iloc[idx]
            color = cmap(norm(int(curr_row['time_step'])))
            ax.annotate(
                '',
                xy=(curr_row['PCA2'], curr_row['PCA1']),
                xytext=(prev_row['PCA2'], prev_row['PCA1']),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8, alpha=0.95),
                zorder=3,
            )
            ax.scatter(
                curr_row['PCA2'],
                curr_row['PCA1'],
                s=18,
                color=color,
                edgecolors='none',
                zorder=4,
            )

    # Add a colorbar to explain color encoding across timesteps.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Time step')

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    if flip_x_axis:
        ax.invert_xaxis()
    ax.set_xlabel('PCA Dimension 2')
    ax.set_ylabel('PCA Dimension 1')
    plt.tight_layout()

    prefix = save_path or ''
    if savePNG:
        plt.savefig(prefix + 'example_pca_trajectories.png', dpi=300)
    if saveSVG:
        plt.savefig(prefix + 'example_pca_trajectories.svg', dpi=300)
    plt.show()


def plot_example_pca_trajectories_choose_ab_longshort(
    df,
    decision_time_short,
    decision_time_medium,
    decision_time_long,
    savePNG=True,
    saveSVG=False,
    save_path=None,
    random_state=80,
    flip_x_axis=False,
    xlim=None,
    ylim=(-3.5, 3.5),
    max_decision_time=15,
    choose_a_action=0,
    choose_b_action=1,
    color_choose_a='blue',
    color_choose_b='red',
    alpha_short=0.42,
    alpha_medium=0.72,
    alpha_long=1.0,
    figsize=(2.85, 2.5),
):
    """
    For choose A and choose B, randomly pick three trials each (short / medium / long RT), matched
    exactly on ``decision_time_{short,medium,long}``. Early rows may be sample actions; episode
    label uses the **last** committed ``action`` (see ``choose_a_action`` / ``choose_b_action``).

    Legend shows only Choose A vs Choose B. RT length is encoded by ``alpha_short`` / ``alpha_medium``
    / ``alpha_long`` (multipliers on time-varying opacity along each trajectory; short more
    transparent, long more opaque by default).
    """
    set_style()

    if choose_a_action == choose_b_action:
        raise ValueError("choose_a_action and choose_b_action must differ.")

    eligible_df = df[df['decision_time'] <= max_decision_time].copy()
    eligible_episodes = eligible_df['episode'].drop_duplicates()
    if len(eligible_episodes) == 0:
        print("No eligible episodes found for plot_example_pca_trajectories_choose_ab_longshort.")
        return

    if 'action' not in eligible_df.columns:
        raise ValueError("DataFrame must contain an 'action' column.")

    traj_df = eligible_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['PCA1', 'PCA2']).copy()
    if traj_df.empty:
        raise ValueError("No valid PCA coordinates available.")

    committed = {int(choose_a_action), int(choose_b_action)}
    ep_rows = []
    for ep, g in traj_df.groupby('episode', sort=False):
        g = g.sort_values('time_step')
        mask = g['action'].isin([choose_a_action, choose_b_action])
        chosen = g.loc[mask]
        if chosen.empty:
            continue
        final = chosen.iloc[-1]
        ep_rows.append(
            {
                'episode': ep,
                'decision_time': int(final['decision_time']),
                'action': int(final['action']),
            }
        )
    ep_meta = pd.DataFrame(ep_rows)
    if ep_meta.empty:
        raise ValueError(
            f"No episodes whose final committed action is in {sorted(committed)}. "
            "Check choose_a_action / choose_b_action or data."
        )

    rng = np.random.default_rng(random_state)

    def _pick_episode(action_val, dt):
        cand = ep_meta[(ep_meta['action'] == action_val) & (ep_meta['decision_time'] == dt)]['episode']
        cand = cand.drop_duplicates()
        if len(cand) == 0:
            side = (
                f'choose A (action=={choose_a_action})'
                if action_val == choose_a_action
                else f'choose B (action=={choose_b_action})'
            )
            raise ValueError(
                f"No episode with {side} and decision_time == {dt}. "
                "Try other decision_time_short / decision_time_medium / decision_time_long values."
            )
        return cand.iloc[int(rng.integers(0, len(cand)))]

    episodes = [
        (_pick_episode(choose_a_action, decision_time_short), color_choose_a, alpha_short),
        (_pick_episode(choose_a_action, decision_time_medium), color_choose_a, alpha_medium),
        (_pick_episode(choose_a_action, decision_time_long), color_choose_a, alpha_long),
        (_pick_episode(choose_b_action, decision_time_short), color_choose_b, alpha_short),
        (_pick_episode(choose_b_action, decision_time_medium), color_choose_b, alpha_medium),
        (_pick_episode(choose_b_action, decision_time_long), color_choose_b, alpha_long),
    ]

    x = traj_df['PCA2'].to_numpy(dtype=float)
    y = traj_df['PCA1'].to_numpy(dtype=float)
    x_pad = 0.05 * max(float(np.ptp(x)), 1e-6)
    y_pad = 0.05 * max(float(np.ptp(y)), 1e-6)
    x_limits = xlim if xlim is not None else (float(np.min(x) - x_pad), float(np.max(x) + x_pad))
    y_limits = ylim if ylim is not None else (float(np.min(y) - y_pad), float(np.max(y) + y_pad))

    fig, ax = plt.subplots(figsize=figsize)
    sample_df = traj_df[traj_df['episode'].isin([e[0] for e in episodes])].copy()
    timestep_min = max(1, int(sample_df['time_step'].min()))
    timestep_max = int(sample_df['time_step'].max())
    t_den = max(float(timestep_max - timestep_min), 1.0)

    for ep, base_color, length_alpha in episodes:
        this_trial = sample_df[sample_df['episode'] == ep].sort_values('time_step')
        if this_trial.empty:
            continue
        br, bg, bb, _ = to_rgba(base_color)
        t0 = int(this_trial['time_step'].iloc[0])
        tnorm0 = (t0 - timestep_min) / t_den
        a0 = float(np.clip(length_alpha * (0.35 + 0.65 * tnorm0), 0.0, 1.0))
        ax.scatter(
            this_trial['PCA2'].iloc[0],
            this_trial['PCA1'].iloc[0],
            facecolors=(br, bg, bb, a0),
            edgecolors='black',
            linewidths=0.6,
            s=30,
            zorder=10,
        )
        for idx in range(1, len(this_trial)):
            prev_row = this_trial.iloc[idx - 1]
            curr_row = this_trial.iloc[idx]
            ti = int(curr_row['time_step'])
            tnorm = (ti - timestep_min) / t_den
            alpha = float(np.clip(length_alpha * (0.35 + 0.65 * tnorm), 0.0, 1.0))
            seg_rgba = (br, bg, bb, alpha)
            ax.annotate(
                '',
                xy=(curr_row['PCA2'], curr_row['PCA1']),
                xytext=(prev_row['PCA2'], prev_row['PCA1']),
                arrowprops=dict(arrowstyle='->', color=seg_rgba, lw=1.8),
                zorder=3,
            )
            ax.scatter(
                curr_row['PCA2'],
                curr_row['PCA1'],
                s=18,
                facecolors=seg_rgba,
                edgecolors='none',
                zorder=4,
            )

    leg_handles = [
        plt.Line2D([0], [0], color=color_choose_a, lw=2.5, label='Choose A'),
        plt.Line2D([0], [0], color=color_choose_b, lw=2.5, label='Choose B'),
    ]
    ax.legend(handles=leg_handles, loc='upper right', frameon=False, fontsize=8)

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    if flip_x_axis:
        ax.invert_xaxis()
    ax.set_xlabel('PCA Dimension 2')
    ax.set_ylabel('PCA Dimension 1')
    plt.tight_layout()

    prefix = save_path or ''
    if savePNG:
        plt.savefig(prefix + 'example_pca_trajectories_ab_longshort.png', dpi=300)
    if saveSVG:
        plt.savefig(prefix + 'example_pca_trajectories_ab_longshort.svg', dpi=300)
    plt.show()


def plot_pca_activation_vs_evidence_sum(df, dt, savePNG=True, saveSVG=False, save_path=None):
    """
    Plot 2D histogram of PC1 activity vs cumulative evidence sum for a single decision time.
    
    Parameters:
    df: DataFrame containing the data
    dt: Decision time to plot
    savePNG: Whether to save as PNG
    saveSVG: Whether to save as SVG
    save_path: Path to save the figure
    """
    
    fig, ax = plt.subplots(figsize=(2.85, 2.5))
    
    # Filter data for this decision time
    dt_data = df[df['decision_time'] == dt]
    
    if len(dt_data) == 0:
        print(f"No data found for decision time {dt}")
        return
    
    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(
        dt_data['evidence_sum'],
        dt_data['PCA1'],
        bins=30
    )
    
    # Calculate correlation coefficient
    corr = np.corrcoef(dt_data['evidence_sum'], dt_data['PCA1'])[0,1]
    
    # Create heatmap
    im = ax.imshow(hist.T, 
                   origin='lower',
                   aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='viridis')
    
    ax.set_xlabel('Cumulative Log LR')
    ax.set_ylabel('PC1 Activity')
    # ax.set_title(f'Decision Time = {dt}\nCorr = {corr:.3f}')

    # ax.set_xlim(-3, 3)
    # ax.set_ylim(-3, 3)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Count')

    plt.tight_layout()
    
    if savePNG and save_path:
        plt.savefig(save_path + f'pca_vs_evidence_dt{dt}.png', dpi=300)
    if saveSVG and save_path:
        plt.savefig(save_path + f'pca_vs_evidence_dt{dt}.svg', dpi=300)
    plt.show()
    return corr

def plot_pc1_loglr_timestep_correlation(df, figsize=(2.85, 2.5), decision_time=10, pc_col='PCA1', loglr_col='evidence_sum', time_col='time_step', savePNG=True, saveSVG=False, save_path=None):
    """
    Plots a heatmap of correlations between PC1 activity and cumulative logLR
    across all timestep pairs for trials with a fixed decision time.

    Parameters:
    - df: pandas DataFrame containing PCA and logLR data
    - decision_time: the fixed decision time to filter trials (default: 10)
    - pc_col: the name of the PCA column to correlate (default: 'PCA1')
    - loglr_col: the name of the logLR column (default: 'evidence_sum')
    - time_col: the name of the time step column (default: 'time_step')
    """
    
    # Filter data for the specified decision time
    df_filtered = df[df['decision_time'] == decision_time].copy()
    max_timestep = df_filtered[time_col].max()

    # Initialize correlation matrix
    corr_matrix = np.zeros((max_timestep, max_timestep))

    # Compute correlations between PC1 and logLR across all timestep pairs
    for i in range(1, max_timestep + 1):  # PC1 timestep
        pca_data = df_filtered[df_filtered[time_col] == i][pc_col]
        
        for j in range(1, max_timestep + 1):  # logLR timestep
            lr_data = df_filtered[df_filtered[time_col] == j][loglr_col]
            
            # Only compute if both have data
            if len(pca_data) > 0 and len(lr_data) > 0:
                corr = np.corrcoef(pca_data, lr_data)[0, 1]
                corr_matrix[i - 1, j - 1] = corr

    # Plot heatmap
    plt.figure(figsize=figsize)
    im = plt.imshow(corr_matrix, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation')

    plt.xlabel('Cumulative LogLR at timestep t')
    plt.ylabel(f'PC1 activation at timestep t')
    plt.xticks(range(max_timestep), range(1, max_timestep + 1))
    plt.yticks(range(max_timestep), range(1, max_timestep + 1))

    plt.tight_layout()
    if savePNG:
        plt.savefig(save_path + f'pc1_loglr_timestep_correlation_dt{decision_time}.png', dpi=300)
    if saveSVG:
        plt.savefig(save_path + f'pc1_loglr_timestep_correlation_dt{decision_time}.svg', dpi=300)
    plt.show()


def plot_pc2_timestep_correlation(
    df,
    figsize=(2.85, 2.5),
    pc_col='PCA2',
    time_col='time_step',
    max_timestep=None,
    drop_time_step_zero=True,
    savePNG=True,
    saveSVG=False,
    save_path=None,
):
    """
    Mean PC activation per timestep (pooled over all trials) vs timestep.

    max_timestep: if set, only rows with time_step <= max_timestep are used.
    """
    df_plot = df.copy()
    if drop_time_step_zero:
        df_plot = df_plot[df_plot[time_col] != 0]
    sub = df_plot[[time_col, pc_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        raise ValueError(f"No valid rows for {time_col} vs {pc_col}.")

    if max_timestep is not None:
        max_timestep = int(max_timestep)
        sub = sub[sub[time_col] <= max_timestep].copy()
        if sub.empty:
            raise ValueError(f"No rows with {time_col} <= {max_timestep}.")

    mean_by_t = sub.groupby(time_col, sort=True)[pc_col].mean()
    t_steps = mean_by_t.index.to_numpy(dtype=float)
    y_mean = mean_by_t.to_numpy(dtype=float)

    plt.figure(figsize=figsize)
    plt.plot(t_steps, y_mean, marker='o', ms=4, lw=1, color='black')
    plt.xlabel('Timestep')
    plt.ylabel(f'Mean PC2 activation')

    plt.tight_layout()
    prefix = save_path or ''
    stem = 'pc2_vs_timestep_mean'
    if max_timestep is not None:
        stem += f'_maxt{max_timestep}'
    if savePNG:
        plt.savefig(prefix + f'{stem}.png', dpi=300)
    if saveSVG:
        plt.savefig(prefix + f'{stem}.svg', dpi=300)
    plt.show()


def _compute_binned_indicator_correlation_matrix(
    df,
    x_col,
    y_col,
    x_bins=20,
    y_bins=20,
    drop_time_step_zero=False,
):
    """
    Bin x/y values and compute a bin-pair correlation matrix.

    Each cell is the Pearson correlation between two binary indicator vectors:
    whether each sample falls into a given x-bin and whether it falls into a
    given y-bin. For binary indicators, this is the phi coefficient.
    """

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Missing required columns: {x_col}, {y_col}")

    data = df[[x_col, y_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if drop_time_step_zero and x_col == 'time_step':
        data = data[data[x_col] != 0].copy()

    if data.empty:
        raise ValueError(f"No valid rows available for {x_col} vs {y_col}.")

    data['x_bin'] = pd.cut(data[x_col], bins=x_bins, include_lowest=True, duplicates='drop')
    data['y_bin'] = pd.cut(data[y_col], bins=y_bins, include_lowest=True, duplicates='drop')
    data = data.dropna(subset=['x_bin', 'y_bin']).copy()

    if data.empty:
        raise ValueError(f"No rows remained after binning {x_col} and {y_col}.")

    x_categories = data['x_bin'].cat.categories
    y_categories = data['y_bin'].cat.categories
    corr_matrix = np.full((len(y_categories), len(x_categories)), np.nan, dtype=float)

    for x_idx, x_bin in enumerate(x_categories):
        x_indicator = (data['x_bin'] == x_bin).astype(float).to_numpy()
        x_std = float(np.std(x_indicator))
        if np.isclose(x_std, 0.0):
            continue

        for y_idx, y_bin in enumerate(y_categories):
            y_indicator = (data['y_bin'] == y_bin).astype(float).to_numpy()
            y_std = float(np.std(y_indicator))
            if np.isclose(y_std, 0.0):
                continue
            corr_matrix[y_idx, x_idx] = np.corrcoef(x_indicator, y_indicator)[0, 1]

    return corr_matrix, x_categories, y_categories


def _format_interval_labels(categories, precision=2):
    labels = []
    for interval in categories:
        left = round(interval.left, precision)
        right = round(interval.right, precision)
        labels.append(f'[{left}, {right}]')
    return labels


def _format_bin_indices(n_bins):
    return [str(i) for i in range(1, n_bins + 1)]


def plot_pc1_evidence_bin_correlation_heatmap(
    df,
    x_col='evidence_sum',
    y_col='PCA1',
    n_bins=10,
    max_timestep=None,
    x_range=(-3, 3),
    y_range=None,
    figsize=(4.2, 3.4),
    cmap='viridis',
    savePNG=True,
    saveSVG=False,
    save_path=None,
):
    """
    Plot a heatmap of within-bin correlations between cumulative evidence and PC1.

    Collect the current cumulative evidence and PC1 activation from all
    timesteps, then divide both dimensions into `n_bins` bins. Within each
    2D bin, compute the Pearson correlation between cumulative evidence and PC1
    activation across all samples in that subset. The heatmap shows correlation
    on real-valued cumulative-evidence (x-axis) and PC1-activation (y-axis)
    coordinates.
    """

    set_style()
    required_cols = {x_col, y_col, 'time_step'}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    data = df[['time_step', x_col, y_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data['time_step'] != 0].copy()

    if data.empty:
        raise ValueError(f"No valid rows available for {x_col} vs {y_col}.")

    if max_timestep is None:
        max_timestep = int(data['time_step'].max())
    else:
        max_timestep = int(max_timestep)

    if max_timestep < 1:
        raise ValueError("max_timestep must be at least 1.")

    data = data[(data['time_step'] >= 1) & (data['time_step'] <= max_timestep)].copy()
    if data.empty:
        raise ValueError(f"No valid rows remained after filtering to timesteps <= {max_timestep}.")

    x_min, x_max = float(x_range[0]), float(x_range[1])
    if x_max <= x_min:
        raise ValueError("x_range must satisfy x_range[1] > x_range[0].")

    if y_range is None:
        y_min = float(data[y_col].min())
        y_max = float(data[y_col].max())
    else:
        y_min, y_max = float(y_range[0]), float(y_range[1])
    if y_max <= y_min:
        raise ValueError("y_range must satisfy y_range[1] > y_range[0].")

    x_edges = np.linspace(x_min, x_max, int(n_bins) + 1)
    y_edges = np.linspace(y_min, y_max, int(n_bins) + 1)

    data['evidence_bin'] = pd.cut(
        data[x_col],
        bins=x_edges,
        labels=False,
        include_lowest=True,
        duplicates='drop',
    )
    data['pc1_bin'] = pd.cut(
        data[y_col],
        bins=y_edges,
        labels=False,
        include_lowest=True,
        duplicates='drop',
    )
    data = data.dropna(subset=['evidence_bin', 'pc1_bin']).copy()

    if data.empty:
        raise ValueError("No valid rows remained after 2D binning.")

    data['evidence_bin'] = data['evidence_bin'].astype(int)
    data['pc1_bin'] = data['pc1_bin'].astype(int)

    corr_matrix = np.full((int(n_bins), int(n_bins)), np.nan, dtype=float)

    for pc1_bin_idx in range(int(n_bins)):
        for evidence_bin_idx in range(int(n_bins)):
            bin_data = data[
                (data['evidence_bin'] == evidence_bin_idx)
                & (data['pc1_bin'] == pc1_bin_idx)
            ]
            if len(bin_data) < 2:
                continue
            if np.isclose(bin_data[x_col].std(ddof=0), 0.0) or np.isclose(bin_data[y_col].std(ddof=0), 0.0):
                continue
            corr_matrix[pc1_bin_idx, evidence_bin_idx] = np.corrcoef(
                bin_data[x_col],
                bin_data[y_col],
            )[0, 1]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        corr_matrix,
        origin='lower',
        aspect='auto',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set_xlabel('Cumulative Evidence')
    ax.set_ylabel('PC1 Activation')
    ax.set_xticks(np.arange(-3, 4, 1))
    plt.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()

    prefix = save_path or ''
    if savePNG:
        plt.savefig(prefix + 'pc1_evidence_bin_correlation_heatmap.png', dpi=300)
    if saveSVG:
        plt.savefig(prefix + 'pc1_evidence_bin_correlation_heatmap.svg', dpi=300)
    plt.show()

    return corr_matrix


def plot_pc2_timestep_bin_correlation_heatmap(
    df,
    x_col='time_step',
    y_col='PCA2',
    max_timestep=None,
    y_bins=20,
    figsize=(4.2, 3.4),
    cmap='RdBu_r',
    savePNG=True,
    saveSVG=False,
    save_path=None,
):
    """
    Plot a heatmap of bin-pair correlations between timestep and PC2 activation.

    After binning the x/y axes, each heatmap cell shows the Pearson correlation
    between the binary indicator for one timestep bin and the binary indicator
    for one PC2-activation bin across samples.
    """

    set_style()

    df_plot = df.copy()
    valid_time_steps = (
        df_plot[x_col]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    valid_time_steps = valid_time_steps[valid_time_steps != 0]
    if valid_time_steps.empty:
        raise ValueError(f"No valid rows available for {x_col} vs {y_col}.")

    if max_timestep is None:
        max_timestep = int(valid_time_steps.max())
    else:
        max_timestep = int(max_timestep)

    if max_timestep < 1:
        raise ValueError("max_timestep must be at least 1.")

    df_plot = df_plot[(df_plot[x_col] >= 1) & (df_plot[x_col] <= max_timestep)].copy()
    x_bins = np.arange(0.5, max_timestep + 1.5, 1.0)

    corr_matrix, x_categories, y_categories = _compute_binned_indicator_correlation_matrix(
        df=df_plot,
        x_col=x_col,
        y_col=y_col,
        x_bins=x_bins,
        y_bins=y_bins,
        drop_time_step_zero=True,
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=[str(i) for i in range(1, len(x_categories) + 1)],
        yticklabels=_format_bin_indices(len(y_categories)),
        cbar_kws={'label': 'Bin-pair correlation'},
    )
    ax.set_xlabel('Timestep')
    ax.set_ylabel('PC2 Activation Bin Index')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    prefix = save_path or ''
    if savePNG:
        plt.savefig(prefix + 'pc2_timestep_bin_correlation_heatmap.png', dpi=300)
    if saveSVG:
        plt.savefig(prefix + 'pc2_timestep_bin_correlation_heatmap.svg', dpi=300)
    plt.show()

    return corr_matrix, x_categories, y_categories


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

def plot_example_pca_tree(
    df,
    cum_col='cumulative_logLR',
    savePNG=True,
    saveSVG=False,
    save_path=None,
    max_time_step=15,
    random_state=80,
    flip_x_axis=False,
    xlim=None,
    ylim=None,
    figsize=(3.2, 2.8),
    node_size=40,
    edge_lw=1.6,
    alpha_edge=0.9,
    alpha_node=1.0,
    average_nodes=True,
):
    """
    Plot PCA state-space as a branching tree based on cumulative evidence.

    Tree definition:
    - each node = one unique cumulative logLR value at one timestep
    - node position = mean PCA position of all rows with that (time_step, cumulative_logLR)
    - each edge = observed transition from (t-1, cum1) to (t, cum2) across episodes

    Required columns in df:
        ['episode', 'time_step', 'PCA1', 'PCA2', cum_col]
    """

    set_style()

    required_cols = ['episode', 'time_step', 'PCA1', 'PCA2', cum_col]
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns: {missing}")

    # --------------------------------------------------
    # 1. Clean data
    # --------------------------------------------------
    plot_df = df.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=['PCA1', 'PCA2', 'time_step', cum_col, 'episode']).copy()

    if plot_df.empty:
        raise ValueError("No valid rows available after dropping NaN/inf values.")

    plot_df['time_step'] = plot_df['time_step'].astype(int)
    plot_df = plot_df[plot_df['time_step'] <= max_time_step].copy()

    if plot_df.empty:
        raise ValueError(f"No rows with time_step <= {max_time_step}.")

    # --------------------------------------------------
    # 2. Build node table:
    #    one node per (time_step, cumulative logLR)
    # --------------------------------------------------
    if average_nodes:
        node_df = (
            plot_df
            .groupby(['time_step', cum_col], as_index=False)
            .agg(
                PCA1=('PCA1', 'mean'),
                PCA2=('PCA2', 'mean'),
                n_obs=('episode', 'count')
            )
            .sort_values(['time_step', cum_col])
        )
    else:
        # keep first occurrence if you do not want averaging
        node_df = (
            plot_df
            .sort_values(['time_step', cum_col])
            .drop_duplicates(subset=['time_step', cum_col])
            [['time_step', cum_col, 'PCA1', 'PCA2']]
            .copy()
        )
        node_df['n_obs'] = 1

    # dictionary: (t, cum) -> (x, y)
    node_pos = {
        (int(row['time_step']), row[cum_col]): (float(row['PCA2']), float(row['PCA1']))
        for _, row in node_df.iterrows()
    }

    # --------------------------------------------------
    # 3. Build edges from observed transitions within episodes
    # --------------------------------------------------
    edge_counter = defaultdict(int)

    for ep, ep_df in plot_df.groupby('episode'):
        ep_df = ep_df.sort_values('time_step')

        # collapse duplicate rows within same episode/timestep if needed
        ep_df = ep_df.drop_duplicates(subset=['time_step'], keep='first')

        rows = ep_df[['time_step', cum_col]].to_records(index=False)
        for i in range(1, len(rows)):
            t_prev, cum_prev = rows[i - 1]
            t_curr, cum_curr = rows[i]

            # only connect consecutive timesteps
            if int(t_curr) == int(t_prev) + 1:
                edge_counter[((int(t_prev), cum_prev), (int(t_curr), cum_curr))] += 1

    if len(edge_counter) == 0:
        raise ValueError("No valid parent-child transitions found.")

    # --------------------------------------------------
    # 4. Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    timestep_min = int(node_df['time_step'].min())
    timestep_max = int(node_df['time_step'].max())
    norm = plt.Normalize(timestep_min, timestep_max)
    cmap = cm.get_cmap('plasma')

    # ---- draw edges first
    max_edge_count = max(edge_counter.values())

    for (parent, child), count in edge_counter.items():
        if parent not in node_pos or child not in node_pos:
            continue

        x0, y0 = node_pos[parent]
        x1, y1 = node_pos[child]

        color = cmap(norm(child[0]))
        lw = edge_lw * (0.6 + 0.8 * count / max_edge_count)

        ax.annotate(
            '',
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=lw,
                alpha=alpha_edge,
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=2,
        )

    # ---- draw nodes
    for _, row in node_df.iterrows():
        t = int(row['time_step'])
        cum_val = row[cum_col]
        x, y = node_pos[(t, cum_val)]
        color = cmap(norm(t))

        # optionally scale node size by occupancy
        size = node_size * (1.0 + 0.25 * np.log1p(row['n_obs']))

        ax.scatter(
            x, y,
            s=size,
            color=color,
            edgecolors='black' if t == timestep_min else 'none',
            linewidths=0.6,
            alpha=alpha_node,
            zorder=4,
        )

    # --------------------------------------------------
    # 5. Axis limits
    # --------------------------------------------------
    x_all = node_df['PCA2'].to_numpy(dtype=float)
    y_all = node_df['PCA1'].to_numpy(dtype=float)

    x_pad = 0.08 * max(float(np.ptp(x_all)), 1e-6)
    y_pad = 0.08 * max(float(np.ptp(y_all)), 1e-6)

    x_limits = xlim if xlim is not None else (float(np.min(x_all) - x_pad), float(np.max(x_all) + x_pad))
    y_limits = ylim if ylim is not None else (float(np.min(y_all) - y_pad), float(np.max(y_all) + y_pad))

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    if flip_x_axis:
        ax.invert_xaxis()

    ax.set_xlabel('PCA Dimension 2')
    ax.set_ylabel('PCA Dimension 1')

    # --------------------------------------------------
    # 6. Colorbar
    # --------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Time step')

    plt.tight_layout()

    prefix = save_path or ''
    if savePNG:
        plt.savefig(prefix + 'example_pca_tree.png', dpi=300, bbox_inches='tight')
    if saveSVG:
        plt.savefig(prefix + 'example_pca_tree.svg', dpi=300, bbox_inches='tight')

    plt.show()
