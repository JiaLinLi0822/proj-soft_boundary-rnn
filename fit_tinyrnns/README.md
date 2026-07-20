# fit_tinyrnns

Fit small **GRU policies** (`TinyGRUPolicy`) to monkey choice data from the
sequential probability ratio test (SPRT) experiments. A trial is a sequence of
shapes presented one-per-timestep; the model outputs an action distribution
over `{choose_A, choose_B, continue}` at every step. The final-step action is
compared against the monkey's actual choice.

The package covers the full workflow:

1. Build datasets from Kira-lab `.mat` files.
2. Train one or a grid of models (with optional nested cross-validation).
3. Load fitted checkpoints to run diagnostics, PCA, behaviour reproduction,
   and likelihood-vs-capacity analyses.

---

## Directory layout

```
fit_tinyrnns/
├── config.py                  # TrainConfig dataclass
├── networks.py                # TinyGRUPolicy (GRUCell + linear / diagonal readout)
├── dataset.py                 # TrialSeq, KiraDataset, split_dataset, stratified_kfold, ...
├── trainer.py                 # Trainer wrapper class + module-level model ops (build_*/rollout/loss/evaluate)
├── nested_cv.py               # NestedCV(...).fit(data)
├── checkpoints.py             # save/load and loss-curve plotting
├── simulate.py                # Generate trained-RNN rollout data
├── behavior.py                # Monkey/RNN behavior processing and plots
├── policy.py                  # RNN policy processing and plots
├── representation.py          # RNN hidden-state and representation-noise plots
├── environment.py             # Kira-style trial environment
├── utils.py                   # get_device / set_seed / shape encoding / load_mat73
│
├── train_single.py            # CLI: single model or hidden-size sweep
├── train_nested_cv.py         # Separate nested CV for each hidden size
│
├── analysis.py                # CLI: complete one-shot report for a single checkpoint
│
├── runs/                      # Training checkpoints & plots (created on first run)
└── results/                   # Secondary analysis artefacts
```

### Module responsibilities

| Module | Job |
|---|---|
| `config.py` | `TrainConfig` specifies one `Trainer.fit()` run. |
| `networks.py` | `TinyGRUPolicy` — a single `GRUCell` plus either a dense or diagonal readout over `{A, B, continue}`. |
| `dataset.py` | `TrialSeq`, list-based `KiraDataset`, train/validation splitting, nested-CV folds, and batch collation. |
| `trainer.py` | Model operations (`rollout_sequence`, `compute_batch_loss`, `recurrent_l1_penalty`, `evaluate`) plus `Trainer`. Callers use PyTorch `DataLoader` directly. |
| `nested_cv.py` | `NestedCV(...).fit(data)` returns a flat `DataFrame` and saves per-inner plus per-outer checkpoints. |
| `checkpoints.py` | `save_checkpoint`, `load_policy_checkpoint`, `save_loss_curve_png`. |
| `simulate.py` | `Trial`, normal rollouts, and matched-evidence rollouts; no analysis or plotting. |
| `behavior.py` | Monkey/RNN behavior conversion, summaries, and plots. |
| `policy.py` | Policy tables, evidence binning, and p(sample) plots. |
| `representation.py` | Hidden-state, state-space, and representation-noise plots. |
| `utils.py` | `get_device("auto"|"cpu"|"cuda"|"mps")`, `set_seed`, `load_mat73`, and shape-encoding helpers. |

---

## Installation

This folder shares a Python environment with the rest of the `proj-rnn-sprt`
repository. Required packages:

```
torch, numpy, scipy, h5py, matplotlib, tqdm, scikit-learn
```

All scripts are designed to be launched **from the `fit_tinyrnns/` directory**
(top-level `sys.path` is flat — `from dataset import ...`).

---

## Data

Input is a Kira-lab MATLAB file (`dataE.mat` or `dataJ.mat`) placed at
`proj-rnn-sprt/neuralSPRT/data/`. Use
`KiraDataset(mat_path, monkey_id, ...)` to build a list of
`TrialSeq`s. Each trial's input `x` is a per-timestep one-hot over 8 shape
categories, and `y` encodes the target action per step (`continue` until the
final step, then `choose_A` / `choose_B`).

`monkey_id`: `1 = Eli`, `2 = Joey`.

---

## Training

### Single model or hidden-size sweep — `train_single.py`

```bash
python train_single.py \
    --monkey-id 1 \
    --hidden 2 \
    --epochs 500 --lr 1e-3 --batch-size 128 \
    --val-frac 0.2 --seed 30 --device auto \
    --diagonal-readout --recurrent-l1-lambda 1e-5 \
    --outdir runs/tiny_gru_single
```

Pass several values to `--hidden 1 2 3 5 10` to train a sweep. Per run the
script writes under `runs/.../MonkeyE|MonkeyJ/h{H}_l1{L}_seed{S}_{timestamp}/`:

- `model.pt` — checkpoint and final metrics
- `loss_curve.png` — training/validation loss curve

### Nested cross-validation — `train_nested_cv.py`

Runs stratified *K*-fold outer CV; for each outer fold, a `(K−1)`-fold inner CV
is run over the `(hidden_dim, recurrent_l1)` grid; the best config per outer
fold is re-evaluated on the held-out outer test split.

```bash
python train_nested_cv.py \
    --monkey-id 1 \
    --hidden 1 2 3 4 5 10 20 \
    --recurrent-l1 1e-5 \
    --outer-folds 5 \
    --inner-epochs 1000 --batch-size 1024 --lr 3e-3 \
    --early-stopping-patience 100 \
    --num-workers 8 \
    --seed 20260409 \
    --outdir runs/tiny_gru_choice_nestedcv
```

**Parallelism.** Each hidden size runs a separate nested CV whose inner loop is
`outer × inner × (l1, initialization seed)`. `--num-workers N` (default `0` =
serial) dispatches those fits to a pool of `N`
processes; each worker pins BLAS to a single thread to avoid oversubscription.
Because every fit is fully seeded by its own task, the results (and every
on-disk checkpoint) are **bit-identical to the serial run** — only faster.
Set `N` to the number of CPU cores you want to use; speedup is close to linear.
Prefer `--device cpu` with parallel workers (the models are tiny; many
processes sharing one GPU would contend). Selection, outer-test evaluation, and
checkpoint I/O all run in the main process.

Output layout:

```
runs/tiny_gru_choice_nestedcv/monkeyE/nestedcv_{timestamp}_seed{S}/
├── cv_models/outerNN/hH_l1_{slug}_seedS/innerMM.pt
├── inner_results_hH.csv                         # initialization robustness
├── outerN_besthH_l1_{L}_seedS.pt                # selected model per outer fold
└── summary.csv                                  # one row per outer fold and hidden size
```

---

## Analysis

Analysis entry points require a checkpoint `.pt` file (or a run directory).
The monkey id is inferred from `MonkeyE` or `MonkeyJ` in its path when omitted.

`behavior.trials_from_dataset()` converts `KiraDataset` into monkey trials;
`simulate.simulate_model()` generates model trials with policy and hidden-state paths.

### `analysis.py` — one-shot report

```bash
python analysis.py --ckpt runs/.../model.pt
```

Loads the checkpoint once, simulates one shared set of model trials, and writes
the complete figure suite (PNG + SVG) plus one `analysis_summary.json` into the
checkpoint folder (or `--outdir`):

| File | Content |
|---|---|
| `behavior.{png,svg}` | Monkey vs RNN: #shapes-used, psychometric, subjective weight. |
| `psample_avg_policy.{png,svg}` | Mean P(sample) vs cumulative log-LR by time step. |
| `psample_vs_evidence_scatter.{png,svg}` | Raw P(sample) observations vs evidence with one highlighted trial. |
| `sampling_variability.{png,svg}` | Variability of P(sample) across trials at matched timestep and evidence. |
| `hidden_by_timestep.{png,svg}` | Raw h0/h1 scatter coloured by time step. |
| `hidden_by_evidence.{png,svg}` | Raw h0/h1 scatter coloured by cumulative evidence. |
| `representation_noise_over_time.{png,svg}` | Hidden-state variability by timestep. |
| `representation_noise_heatmap.{png,svg}` | Variability by timestep and evidence. |
| `policy_representation.{png,svg}` | Action policy represented in two-dimensional hidden space. |
| `permuted_evidence_hidden_trajectories.{png,svg}` | Hidden trajectories for reordered stimuli with matched evidence. |

Plot-specific settings are edited directly at the corresponding calls in
`analysis.py`; they are not command-line arguments. The old standalone `psample_*` and
`permuted_evidence_trajectories.py` entry points have been removed; their
functionality now runs through `analysis.py` and the domain modules.

Nested-CV training automatically writes `test_trajectory_nll_vs_hidden.png`
and `.svg` beside `summary.csv`. Each outer-fold value is the mean, across
test trials, of the summed action NLL over that trial's whole trajectory.

### Programmatic API for custom plots

```python
from checkpoints import load_policy_checkpoint
from policy import plot_policy
from simulate import simulate_model

model, ckpt, _ = load_policy_checkpoint("runs/.../model.pt", "auto")
rollouts = simulate_model(model, monkey_id=1, n_trials=20000, rollout_seed=0)

plot_policy(rollouts, out_path="figs/policy")
```

---

## Checkpoint format

`torch.save` dict with at least these keys (produced by `Trainer` /
`NestedCV.fit`):

```
state_dict, hidden_dim, input_dim (=8), num_actions (=3),
diagonal_readout,
best_epoch, epochs_trained, stopped_early,
final_train {loss, final_acc, ...}, final_val {...},
split / outer_fold / inner_fold (depending on entry point), ...
```

Load anywhere with `checkpoints.load_policy_checkpoint(path, device)` → returns
`(model, ckpt_dict, device)` already `.to(device).eval()`.

---

## Design notes

The codebase was refactored to be intentionally small and flat:

- **Dataclasses over long kwarg lists.** `TrainConfig` configures one training
  run; `NestedCV` owns its CV settings and exposes one `fit(data)` entry point.
- **Stateless model ops are module-level functions in `trainer.py`** —
  `rollout_sequence`, `compute_batch_loss`, `recurrent_l1_penalty`,
  `evaluate`. Plot scripts, `nested_cv`, and `checkpoints` import them
  directly (`from trainer import rollout_sequence, ...`).
- **`Trainer` is a thin wrapper.** The caller is responsible for building
  the model and native PyTorch DataLoaders externally, then passing them in:
  `Trainer(model, train_loader, val_loader,
  device=..., config=..., train_eval_loader=...).fit()`. `Trainer` owns
  only the optimizer and the epoch loop; no `@staticmethod`s required.
- **`Trainer` only trains.** It owns a single `(train, val)` pair and a
  single model + optimiser. Data splitting lives in `dataset.py`; multi-run
  orchestration lives in `nested_cv.py`; persistence lives in
  `checkpoints.py`.
- **Two CLI entry points.** `train_single.py` covers both single-model and
  hidden-sweep use cases; `train_nested_cv.py` covers the full nested CV.
