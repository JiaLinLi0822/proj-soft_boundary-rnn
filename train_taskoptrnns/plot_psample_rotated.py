from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


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

# target log ratios
target_log_ratios = np.array([
    -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
])

# Switch between 'Cond1' (max_steps=10) and 'Cond2' (max_steps=10000)
CONDITION = 'Cond2'

SAMPLE_COSTS = [0.01, 0.02, 0.03, 0.04, 0.05]
CONDITION_CONFIG = {
    'Cond1': {'exp_ids': [4, 5, 6, 7, 8], 'max_samples': 10, 'max_steps': 10},
    'Cond2': {'exp_ids': [106, 107, 108, 109, 110], 'max_samples': 10000, 'max_steps': 10000},
}

script_dir = Path(__file__).resolve().parent


def resolve_root_path(condition):
    for candidate in (
        script_dir / 'results' / condition,
        Path('/Users/lijialin/Desktop/Research/proj-rnn-sprt/opt_taskrnns/results') / condition,
    ):
        if candidate.exists():
            return candidate
    return script_dir / 'results' / condition


def build_experiments(condition):
    cfg = CONDITION_CONFIG[condition]
    experiments = []
    for exp_id, sample_cost in zip(cfg['exp_ids'], SAMPLE_COSTS):
        path = (
            f'exp_{exp_id:03d}_reward=1.0_sample_cost={sample_cost}_'
            f'urgency_cost=0.00_logLR=[-0.9,0.9]_'
            f'max_samples={cfg["max_samples"]}_max_steps={cfg["max_steps"]}_epNum=1000000'
        )
        experiments.append({
            'path': path,
            'sample_cost': sample_cost,
            'label': f'Cost={sample_cost:.2f}',
        })
    return experiments


root_path = resolve_root_path(CONDITION)
experiments = build_experiments(CONDITION)
sample_costs = [exp['sample_cost'] for exp in experiments]


def _save_figure(fig, prefix, stem, savePNG, saveSVG):
    if savePNG:
        fig.savefig(prefix / f'{stem}.png', dpi=300, bbox_inches='tight')
    if saveSVG:
        fig.savefig(prefix / f'{stem}.svg', dpi=300, bbox_inches='tight')


def save_sample_cost_legend(
    save_path=None,
    output_stem='psample_sample_cost_legend',
    savePNG=True,
    saveSVG=False,
    show=False,
):
    """Save sample-cost colorbar as a standalone figure."""
    norm = Normalize(vmin=min(sample_costs), vmax=max(sample_costs))
    sm = ScalarMappable(cmap=plt.cm.Greens, norm=norm)
    sm.set_array([])

    fig = plt.figure(figsize=(0.45, 2.85))
    ax = fig.add_axes([0.35, 0.05, 0.25, 0.9])
    cbar = fig.colorbar(sm, cax=ax)
    cbar.set_label('Sample Cost', rotation=270, labelpad=12)
    cbar.ax.tick_params(labelsize=8, width=0.5, length=2)
    cbar.set_ticks(sample_costs)
    cbar.set_ticklabels([f'{c:.2f}' for c in sample_costs])

    prefix = Path(save_path) if save_path else Path('')
    _save_figure(fig, prefix, output_stem, savePNG, saveSVG)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_rotated_psample_dt_multiple_costs_grid(
    df,
    decision_times,
    bin_width=0.05,
    bin_start=-3,
    bin_end=3.1,
    savePNG=True,
    saveSVG=False,
    save_path=None,
    output_stem='psample_multiple_costs_grid_rotated',
    save_legend=True,
    legend_stem=None,
    show=False,
):
    """
    Plot cumulative evidence as a function of psample across sample costs
    for multiple decision times in a 1xN shared-y layout.
    """
    if legend_stem is None:
        legend_stem = f'{output_stem}_legend'

    fig, axes = plt.subplots(
        1,
        len(decision_times),
        figsize=(1.25 * len(decision_times), 2.85),
        sharey=True,
        constrained_layout=True,
    )

    if len(decision_times) == 1:
        axes = [axes]

    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(sample_costs)))

    for ax_idx, (ax, dt) in enumerate(zip(axes, decision_times)):
        df_dt = df[df['decision_time'] == dt].copy()
        df_dt['evidence_sum_bin'] = pd.cut(
            df_dt['evidence_sum'],
            bins=np.arange(bin_start, bin_end, bin_width),
            include_lowest=True,
        )

        df_binned = (
            df_dt.dropna(subset=['evidence_sum_bin'])
            .groupby(['model', 'evidence_sum_bin'], observed=True)['p_sample']
            .mean()
            .reset_index()
        )

        df_binned['bin_center'] = (
            df_binned['evidence_sum_bin']
            .apply(lambda iv: (iv.left + iv.right) / 2)
            .astype(float)
        )
        df_binned = df_binned[df_binned['bin_center'].between(-3, 3)]

        ax.hlines(0, 0, 1, color='black', linestyle='--', linewidth=0.75)

        for i, model in enumerate(sorted(df_binned['model'].unique())):
            subset = df_binned[df_binned['model'] == model].sort_values('bin_center')
            ax.plot(
                subset['p_sample'],
                subset['bin_center'],
                linewidth=2,
                color=colors[i],
                alpha=1,
            )

        ax.set_xlabel(r"$p(\mathrm{sample})$", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-3.1, 3.1)
        ax.set_xticks(np.arange(0, 1.1, 0.5))
        ax.set_yticks(np.arange(-3, 4, 1))
        ax.grid(False)
        ax.set_title(f'Decision Time = {dt}', fontweight='bold', fontsize=8)
        ax.set_box_aspect(1.25)

        if ax_idx > 0:
            ax.tick_params(labelleft=False)

    axes[0].set_ylabel("Cumulative Evidence", fontsize=10)

    prefix = Path(save_path) if save_path else Path('')
    _save_figure(fig, prefix, output_stem, savePNG, saveSVG)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if save_legend:
        save_sample_cost_legend(
            save_path=save_path,
            output_stem=legend_stem,
            savePNG=savePNG,
            saveSVG=saveSVG,
            show=show,
        )


def load_decision_data():
    all_dfs = []

    for idx, exp in enumerate(experiments):
        print(f"\nProcessing {exp['label']}...")

        try:
            path = root_path / exp['path']
            df = pd.read_json(path / 'data.json', lines=True)

            episode_lengths = df.groupby('episode').size().reset_index(name='decision_time')
            df = df.merge(episode_lengths, on='episode', how='left')
            df['decision_time'] = df['decision_time'] - 1
            df['action'] = df['action'].astype(int)

            df['time_step'] = df['stimuli_so_far'].apply(len)
            df['evidences'] = df.apply(
                lambda row: [
                    target_log_ratios[s]
                    for s in row['stimuli_so_far'][:row['decision_time']]
                ],
                axis=1,
            )
            df['evidence_sum'] = df['evidences'].apply(sum)
            df['p_sample'] = df['policy'].apply(lambda p: p[2])
            df['model'] = idx + 1

            df_decision = df[df['action'] != 2].dropna(subset=['decision_time']).copy()
            all_dfs.append(df_decision)

            print(f'Total trials: {len(df_decision)}')

        except Exception as exc:
            print(f"Error processing {exp['label']}: {exc}")

    if not all_dfs:
        return None

    total_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined trials: {len(total_df)}")
    return total_df


if __name__ == '__main__':
    print(f'Condition: {CONDITION}')
    print(f'Data root: {root_path}')

    total_df = load_decision_data()

    if total_df is not None:
        decision_times = [2, 6, 10]
        print(f"\nGenerating rotated plots for decision times: {decision_times}")

        plot_rotated_psample_dt_multiple_costs_grid(
            total_df,
            decision_times,
            savePNG=True,
            saveSVG=True,
            save_path='',
            output_stem=f'psample_multiple_costs_grid_rotated_{CONDITION}',
        )
    else:
        print('No data loaded!')
