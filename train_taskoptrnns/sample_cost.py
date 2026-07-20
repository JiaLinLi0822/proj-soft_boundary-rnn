import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

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

# target log ratios
target_log_ratios = np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# root_path = '/Users/lijialin/Desktop/Research/proj-rnn-sprt/opt_taskrnns/results/Cond1/'
print(os.getcwd())
root_path = os.getcwd() + '/train_taskoptrnns/results/Cond2/'

experiments = [
    # {'path': 'exp_001_reward=1.0_sample_cost=0.0001_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10_max_steps=10_epNum=1000000/', 'sample_cost': 0.0001, 'label': 'Cost=0.0001'},
    # {'path': 'exp_002_reward=1.0_sample_cost=0.001_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10_max_steps=10_epNum=1000000/', 'sample_cost': 0.001, 'label': 'Cost=0.001'},
    # {'path': 'exp_003_reward=1.0_sample_cost=0.005_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10_max_steps=10_epNum=1000000/', 'sample_cost': 0.005, 'label': 'Cost=0.005'},
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
    # {'path': 'exp_112_reward=1.0_sample_cost=0.001_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1000000/', 'sample_cost': 0.001, 'label': 'Cost=0.001'},
    # {'path': 'exp_113_reward=1.0_sample_cost=0.0001_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1000000/', 'sample_cost': 0.0001, 'label': 'Cost=0.0001'},
    # {'path': 'exp_114_reward=1.0_sample_cost=0.008_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1000000/', 'sample_cost': 0.008, 'label': 'Cost=0.008'},
    # {'path': 'exp_115_reward=1.0_sample_cost=0.005_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1000000/', 'sample_cost': 0.005, 'label': 'Cost=0.005'},
    # {'path': 'exp_116_reward=1.0_sample_cost=0.002_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1000000/', 'sample_cost': 0.002, 'label': 'Cost=0.002'},
    # {'path': 'exp_001_reward=1.0_sample_cost=0.01_urgency_cost=0.00_logLR=[-0.8,0.8]_max_samples=10_max_steps=10000_epNum=900000/', 'sample_cost': 0.01, 'label': 'Cost=0.01'},
    # {'path': 'exp_002_reward=1.0_sample_cost=0.01_urgency_cost=0.00_logLR=[-0.8,0.8]_max_samples=10_max_steps=10000_epNum=1000000/', 'sample_cost': 0.01, 'label': 'Cost=0.01'},
    # {'path': 'exp_003_reward=1.0_sample_cost=0.01_urgency_cost=0.00_logLR=[-0.8,0.8]_max_samples=10_max_steps=10000_epNum=1100000/', 'sample_cost': 0.01, 'label': 'Cost=0.01'},
    # {'path': 'exp_004_reward=1.0_sample_cost=0.01_urgency_cost=0.00_logLR=[-0.8,0.8]_max_samples=10_max_steps=10000_epNum=1200000/', 'sample_cost': 0.01, 'label': 'Cost=0.01'},
    # {'path': 'exp_005_reward=1.0_sample_cost=0.01_urgency_cost=0.00_logLR=[-0.8,0.8]_max_samples=10_max_steps=10000_epNum=1300000/', 'sample_cost': 0.01, 'label': 'Cost=0.01'},
    # {'path': 'exp_006_reward=1.0_sample_cost=0.01_urgency_cost=0.00_logLR=[-0.8,0.8]_max_samples=10_max_steps=10000_epNum=1500000/', 'sample_cost': 0.01, 'label': 'Cost=0.01'},
]

all_stats = []

# Define line styles for different experiments
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2)), (0, (3, 5, 1, 5))]
markers_A = ['o', 'v', '^', '<', '>', 's', 'd']
markers_B = ['s', 'D', 'p', '*', 'h', 'H', 'X']

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

        reward_sum = df['reward'].sum()
        print(f"Sum of reward column: {reward_sum/50000:.4f}")
        
        df_decision = df[df["action"] != 2].dropna(subset=["decision_time"])
        
        # remove trials when agent responded incorrectly
        correct_data = df_decision[df_decision["correct_answer"] == df_decision["action"]].copy()
        
        # separate into groups based on correct_answer value
        correct_data_0 = correct_data[correct_data["correct_answer"] == 0]
        correct_data_1 = correct_data[correct_data["correct_answer"] == 1]
        
        print(f'Total correct trials: {len(correct_data)}')
        print(f'Correct trials for A: {len(correct_data_0)}')
        print(f'Correct trials for B: {len(correct_data_1)}')
        
        grouped0_stats = correct_data_0.groupby("decision_time")["evidence_sum"].agg(["mean", "std", "count"])
        grouped1_stats = correct_data_1.groupby("decision_time")["evidence_sum"].agg(["mean", "std", "count"])
        
        # remove the decision time more than 15
        grouped0_stats = grouped0_stats[grouped0_stats.index <= 10]
        grouped1_stats = grouped1_stats[grouped1_stats.index <= 10]
        
        grouped0_stats = grouped0_stats[grouped0_stats.index > 0]
        grouped1_stats = grouped1_stats[grouped1_stats.index > 0]
        
        # compute 95% confidence intervals
        grouped0_stats["ci95"] = 1.96 * grouped0_stats["std"] / np.sqrt(grouped0_stats["count"])
        grouped1_stats["ci95"] = 1.96 * grouped1_stats["std"] / np.sqrt(grouped1_stats["count"])
        
        # Store the statistics for later analysis
        all_stats.append({
            'sample_cost': exp['sample_cost'],
            'label': exp['label'],
            'grouped0_stats': grouped0_stats,
            'grouped1_stats': grouped1_stats
        })
        
    except Exception as e:
        print(f"Error processing {exp['label']}: {e}")

# ========== Figure 1: Original plot with error bars (no ribbon) ==========
fig1, ax = plt.subplots(figsize=(2.85, 2.05))

# Create color gradients for different sample costs
n_experiments = len(all_stats)
# Blue gradient for Choose A (light to dark)
blues = plt.cm.Blues(np.linspace(0.4, 0.9, n_experiments))
# Red gradient for Choose B (light to dark)
reds = plt.cm.Reds(np.linspace(0.4, 0.9, n_experiments))

# Get sample costs for colorbar
sample_costs = [stats_dict['sample_cost'] for stats_dict in all_stats]

for idx, stats_dict in enumerate(all_stats):
    grouped0_stats = stats_dict['grouped0_stats']
    grouped1_stats = stats_dict['grouped1_stats']
    
    # Plot Choose A in blue gradient with error bars
    ax.errorbar(grouped0_stats.index, grouped0_stats["mean"], 
               yerr=grouped0_stats["std"], fmt='o', linestyle='-', capsize=2, 
               color=blues[idx], elinewidth=0.5, linewidth=1, alpha=0.8, markersize=3)
    
    # Plot Choose B in red gradient with error bars
    ax.errorbar(grouped1_stats.index, grouped1_stats["mean"], 
               yerr=grouped1_stats["std"], fmt='s', linestyle='-', capsize=2, 
               color=reds[idx], elinewidth=0.5, linewidth=1, alpha=0.8, markersize=3)

ax.axhline(0, color='black', linestyle='--', linewidth=0.75)
# Add vertical dashed lines at decision times 3, 8, 12
# for dt in [2, 4, 6, 8, 10]:
#     ax.axvline(dt, color='gray', linestyle='--', linewidth=0.75, alpha=0.6)
ax.set_xlim(0, 10.5)
ax.set_xticks(range(0, 11, 2))
ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("Decision Time")
ax.set_ylabel("Cumulative Evidence")
# ax.set_title("Decision Boundary vs Decision Time", fontweight='bold')
ax.grid(False)

# Add gradient colorbar legend for sample cost
from matplotlib.lines import Line2D

# Add a simple legend for Choose A and Choose B markers
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
           markersize=4, label='Chosen $H_1$', markeredgewidth=0),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
           markersize=4, label='Chosen $H_0$', markeredgewidth=0),
]
ax.legend(handles=legend_elements, loc='upper left', frameon=False, handletextpad=0.5)

# Create two separate colorbars for blue and red gradients
# Blue gradient colorbar
# norm_blue = Normalize(vmin=min(sample_costs), vmax=max(sample_costs))
# sm_blue = ScalarMappable(cmap=plt.cm.Blues, norm=norm_blue)
# sm_blue.set_array([])
# cbar_blue = plt.colorbar(sm_blue, ax=ax, pad=0.02, aspect=15, shrink=0.4, location='right')
# cbar_blue.set_label('Sample Cost', rotation=270, labelpad=8)
# cbar_blue.ax.tick_params(labelsize=6, width=0.5, length=2)
# cbar_blue.set_ticks(sample_costs)
# cbar_blue.set_ticklabels([f'{c:.2f}' for c in sample_costs])

# # Red gradient colorbar
# norm_red = Normalize(vmin=min(sample_costs), vmax=max(sample_costs))
# sm_red = ScalarMappable(cmap=plt.cm.Reds, norm=norm_red)
# sm_red.set_array([])
# cbar_red = plt.colorbar(sm_red, ax=ax, pad=0.02, aspect=15, shrink=0.4, location='right')
# cbar_red.set_label('Sample Cost', rotation=270, labelpad=8)
# cbar_red.ax.tick_params(labelsize=6, width=0.5, length=2)
# cbar_red.set_ticks(sample_costs)
# cbar_red.set_ticklabels([f'{c:.2f}' for c in sample_costs])

# # Adjust the position of the second colorbar
# cbar_red.ax.set_position([cbar_blue.ax.get_position().x0 + 0.05, 
#                           cbar_blue.ax.get_position().y0, 
#                           cbar_blue.ax.get_position().width, 
#                           cbar_blue.ax.get_position().height])

plt.tight_layout()
plt.savefig('logLR_bound_multiple.svg', dpi=300, bbox_inches='tight')
print("\nFigure 1 saved as 'logLR_bound_multiple.svg'")

# # ========== Figure 2: 2x3 subplot - Threshold and Error bars vs Sample Cost ==========
# fig2, axes = plt.subplots(2, 3, figsize=(7, 5))

# # Decision times to analyze
# decision_times = [2, 6, 10]

# # Extract data for each decision time
# for col_idx, dt in enumerate(decision_times):
#     sample_costs = []
#     thresholds_A = []
#     thresholds_B = []
#     std_A = []
#     std_B = []
    
#     for stats_dict in all_stats:
#         sample_costs.append(stats_dict['sample_cost'])
        
#         # Get threshold (mean) for Choose A at this decision time
#         if dt in stats_dict['grouped0_stats'].index:
#             thresholds_A.append(stats_dict['grouped0_stats'].loc[dt, 'mean'])
#             std_A.append(stats_dict['grouped0_stats'].loc[dt, 'std'])
#         else:
#             thresholds_A.append(np.nan)
#             std_A.append(np.nan)
        
#         # Get threshold (mean) for Choose B at this decision time
#         if dt in stats_dict['grouped1_stats'].index:
#             thresholds_B.append(stats_dict['grouped1_stats'].loc[dt, 'mean'])
#             std_B.append(stats_dict['grouped1_stats'].loc[dt, 'std'])
#         else:
#             thresholds_B.append(np.nan)
#             std_B.append(np.nan)
    
#     # Row 1: Threshold vs Sample Cost
#     ax1 = axes[0, col_idx]
#     ax1.plot(sample_costs, thresholds_A, 'o-', color='blue', label='Choose A', 
#              markersize=4, linewidth=0.75, alpha=0.8)
#     ax1.plot(sample_costs, thresholds_B, 's-', color='red', label='Choose B', 
#              markersize=4, linewidth=0.75, alpha=0.8)
#     ax1.axhline(0, color='black', linestyle='--', linewidth=0.75, alpha=0.5)
#     ax1.set_xlabel("Sample Cost")
#     ax1.set_ylabel("Threshold (Mean Evidence)")
#     ax1.set_title(f"Decision Time = {dt}", fontweight='bold')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3, linewidth=0.5)
    
#     # Row 2: Standard Deviation vs Sample Cost
#     ax2 = axes[1, col_idx]
#     ax2.plot(sample_costs, std_A, 'o-', color='blue', label='Choose A', 
#              markersize=4, linewidth=0.75, alpha=0.8)
#     ax2.plot(sample_costs, std_B, 's-', color='red', label='Choose B', 
#              markersize=4, linewidth=0.75, alpha=0.8)
#     ax2.set_xlabel("Sample Cost")
#     ax2.set_ylabel("Standard Deviation")
#     ax2.set_title(f"Std Dev at Decision Time = {dt}", fontweight='bold')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3, linewidth=0.5)

# plt.tight_layout()
# plt.savefig('threshold_std_vs_sample_cost.svg', dpi=300, bbox_inches='tight')
# print("Figure 2 saved as 'threshold_std_vs_sample_cost.svg'")

# plt.show()