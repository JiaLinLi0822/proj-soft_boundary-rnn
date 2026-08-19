import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm

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
    'legend.loc': 'upper right'
})

# target log ratios
target_log_ratios = np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

root_path = '/Users/lijialin/Desktop/Research/proj-rnn-sprt/opt_taskrnns/results/Cond2/'

experiments = [
    # {'path': 'exp_004_reward=1.0_sample_cost=0.01_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10_max_steps=10_epNum=1000000/', 'sample_cost': 0.01, 'label': 'Cost=0.01'},
    # {'path': 'exp_005_reward=1.0_sample_cost=0.02_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10_max_steps=10_epNum=1000000/', 'sample_cost': 0.02, 'label': 'Cost=0.02'},
    # {'path': 'exp_006_reward=1.0_sample_cost=0.03_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10_max_steps=10_epNum=1000000/', 'sample_cost': 0.03, 'label': 'Cost=0.03'},
    # {'path': 'exp_007_reward=1.0_sample_cost=0.04_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10_max_steps=10_epNum=1000000/', 'sample_cost': 0.04, 'label': 'Cost=0.04'},
    # {'path': 'exp_008_reward=1.0_sample_cost=0.05_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10_max_steps=10_epNum=1000000/', 'sample_cost': 0.05, 'label': 'Cost=0.05'},
    {'path': 'exp_106_reward=1.0_sample_cost=0.01_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1000000/', 'sample_cost': 0.01, 'label': 'Cost=0.01'},
    {'path': 'exp_107_reward=1.0_sample_cost=0.02_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1000000/', 'sample_cost': 0.02, 'label': 'Cost=0.02'},
    {'path': 'exp_108_reward=1.0_sample_cost=0.03_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1000000/', 'sample_cost': 0.03, 'label': 'Cost=0.03'},
    {'path': 'exp_109_reward=1.0_sample_cost=0.04_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1000000/', 'sample_cost': 0.04, 'label': 'Cost=0.04'},
    {'path': 'exp_110_reward=1.0_sample_cost=0.05_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1000000/', 'sample_cost': 0.05, 'label': 'Cost=0.05'},
]

sample_costs = [exp['sample_cost'] for exp in experiments]

def plot_psample_dt_single(df, dt, max_steps=20, bin_width=0.05, bin_start=-3, bin_end=3.1, savePNG=True, saveSVG=False, save_path=None):
    """
    Plot psample as a function of cumulative evidence for a specific decision time
    """
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
    fig.subplots_adjust(wspace=0.02)
    
    if savePNG:
        filename = save_path + f'psample_dt{dt}.png' if save_path else f'psample_dt{dt}.png'
        plt.savefig(filename, dpi=300)
    if saveSVG:
        filename = save_path + f'psample_dt{dt}.svg' if save_path else f'psample_dt{dt}.svg'
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_psample_dt_multiple_costs(df, dt, bin_width=0.05, bin_start=-3, bin_end=3.1, savePNG=True, saveSVG=False, save_path=None):
    """
    Plot psample as a function of cumulative evidence for a specific decision time
    across different sample costs
    """
    df_dt = df[df['decision_time'] == dt]

    df_dt["evidence_sum_bin"] = pd.cut(df_dt["evidence_sum"], bins=np.arange(bin_start, bin_end, bin_width), include_lowest=True)

    df_binned = (df_dt.dropna(subset=['evidence_sum_bin'])
        .groupby(['model', 'evidence_sum_bin'], observed=True)['p_sample']
        .mean()
        .reset_index()
    )

    df_binned['bin_center'] = df_binned['evidence_sum_bin'].apply(lambda iv: (iv.left + iv.right) / 2).astype(float)
    df_binned = df_binned[df_binned['bin_center'].between(-3, 3)]

    fig, ax = plt.subplots(figsize=(2.85, 2.5))
    
    # Create color gradient for different sample costs
    n_models = len(df_binned['model'].unique())
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, n_models))
    
    ax.vlines(0, 0, 1, color='black', linestyle='--', linewidth=0.75)

    for i, model in enumerate(sorted(df_binned['model'].unique())):
        subset = df_binned[df_binned['model'] == model].sort_values('bin_center')
        ax.plot(subset['bin_center'],
                subset['p_sample'],
                marker='o',
                linewidth=0.75,
                markersize=3,
                color=colors[i],
                alpha=0.8,
                label=f'Cost={sample_costs[i]:.2f}')

    ax.set_xlabel('Cumulative Evidence')
    ax.set_ylabel('Probability of Sampling')
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(-3, 4, 1))
    ax.grid(False)
    ax.set_title(f'Decision Time = {dt}', fontweight='bold')
    
    # Add colorbar for sample cost gradient
    norm = Normalize(vmin=min(sample_costs), vmax=max(sample_costs))
    sm = ScalarMappable(cmap=plt.cm.Greens, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=15, shrink=0.8)
    cbar.set_label('Sample Cost', rotation=270, labelpad=8)
    cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
    cbar.set_ticks(sample_costs)
    cbar.set_ticklabels([f'{c:.2f}' for c in sample_costs])
    
    plt.tight_layout()
    
    if savePNG:
        filename = save_path + f'psample_dt{dt}_multiple_costs.png' if save_path else f'psample_dt{dt}_multiple_costs_0318.png'
        plt.savefig(filename, dpi=300)
    if saveSVG:
        filename = save_path + f'psample_dt{dt}_multiple_costs.svg' if save_path else f'psample_dt{dt}_multiple_costs_0318.svg'
        plt.savefig(filename, dpi=300)
    plt.show()


def plot_psample_dt_multiple_costs_grid(df, decision_times, bin_width=0.05, bin_start=-3, bin_end=3.1, savePNG=True, saveSVG=False, save_path=None):
    """
    Plot psample as a function of cumulative evidence across different sample costs
    for multiple decision times in a 1xN subplot layout.
    """
    fig, axes = plt.subplots(1, len(decision_times), figsize=(2.85 * len(decision_times), 2.5), sharey=True)

    if len(decision_times) == 1:
        axes = [axes]

    norm = Normalize(vmin=min(sample_costs), vmax=max(sample_costs))
    sm = ScalarMappable(cmap=plt.cm.Greens, norm=norm)
    sm.set_array([])
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(sample_costs)))

    for ax, dt in zip(axes, decision_times):
        df_dt = df[df['decision_time'] == dt].copy()
        df_dt["evidence_sum_bin"] = pd.cut(
            df_dt["evidence_sum"],
            bins=np.arange(bin_start, bin_end, bin_width),
            include_lowest=True,
        )

        df_binned = (
            df_dt.dropna(subset=['evidence_sum_bin'])
            .groupby(['model', 'evidence_sum_bin'], observed=True)['p_sample']
            .mean()
            .reset_index()
        )

        df_binned['bin_center'] = df_binned['evidence_sum_bin'].apply(lambda iv: (iv.left + iv.right) / 2).astype(float)
        df_binned = df_binned[df_binned['bin_center'].between(-3, 3)]

        ax.vlines(0, 0, 1, color='black', linestyle='--', linewidth=0.75)

        for i, model in enumerate(sorted(df_binned['model'].unique())):
            subset = df_binned[df_binned['model'] == model].sort_values('bin_center')
            ax.plot(
                subset['bin_center'],
                subset['p_sample'],
                marker='o',
                linewidth=0.75,
                markersize=3,
                color=colors[i],
                alpha=0.8,
                label=f'Cost={sample_costs[i]:.2f}',
            )

        ax.set_xlabel('Cumulative Evidence', fontsize=10)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(np.arange(-3, 4, 1))
        ax.grid(False)
        ax.set_title(f'Decision Time = {dt}', fontweight='bold')
        ax.set_box_aspect(1)

    axes[0].set_ylabel('Probability of Sampling', fontsize=10)

    cbar = fig.colorbar(sm, ax=axes, pad=0.02, aspect=25, shrink=0.85)
    cbar.set_label('Sample Cost', rotation=270, labelpad=8)
    cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
    cbar.set_ticks(sample_costs)
    cbar.set_ticklabels([f'{c:.2f}' for c in sample_costs])

    plt.tight_layout()

    prefix = save_path or ""
    if savePNG:
        plt.savefig(prefix + 'psample_multiple_costs_grid_0318.png', dpi=300)
    if saveSVG:
        plt.savefig(prefix + 'psample_multiple_costs_grid_0318.svg', dpi=300)
    plt.show()

# Load and process data from multiple experiments
all_dfs = []

for idx, exp in enumerate(experiments):
    print(f"\nProcessing {exp['label']}...")
    
    try:
        path = root_path + exp['path']
        df = pd.read_json(path + 'data.json', lines=True)
        
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
        
        # policy - extract p_sample
        df['p_sample'] = df['policy'].apply(lambda p: p[2])
        
        # Add model identifier (1-indexed)
        df['model'] = idx + 1
        
        # Filter to decision points only (exclude sampling actions)
        df_decision = df[df["action"] != 2].dropna(subset=["decision_time"]).copy()
        
        all_dfs.append(df_decision)
        
        print(f'Total trials: {len(df_decision)}')
        
    except Exception as e:
        print(f"Error processing {exp['label']}: {e}")

# Combine all dataframes
if all_dfs:
    total_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined trials: {len(total_df)}")
    
    # Plot psample as a function of cumulative evidence for specific decision times
    decision_times = [2, 4, 6, 8, 10]
    
    print(f"\nGenerating plots for decision times: {decision_times}")

    plot_psample_dt_multiple_costs_grid(
        total_df,
        decision_times,
        savePNG=True,
        saveSVG=True,
        save_path="",
    )
        
else:
    print("No data loaded!")