# Soft-boundary RNNs for sequential evidence accumulation

Code for studying **soft decision boundaries** in recurrent networks on a sequential probability ratio test (SPRT) task. Monkeys view a sequence of shapes and choose target A or B (or keep sampling). This repo covers three complementary pieces:

1. **Behavioral / physiology analysis** of the original monkey SPRT data (`neuralSPRT/`)
2. **Task-optimized RNNs** trained with A2C on a matching sequential-inference environment (`train_taskoptrnns/`)
3. **Data-fitted tiny GRUs** that imitate monkey choices trial-by-trial (`fit_tinyrnns/`)

Together they support comparing hard vs soft stopping policies, psychometric behavior, and low-dimensional latent dynamics.

---

## Repository layout

```
proj-soft_boundary-rnn/
├── neuralSPRT/           # Monkey SPRT data analysis (+ MATLAB originals)
│   ├── data/             # dataE.mat, dataJ.mat (not tracked; see Data)
│   ├── neuralSPRT_BEH.py # Behavioral panels
│   ├── neuralSPRT_PHY.py # Physiology panels
│   └── paper_reproduction/
├── train_taskoptrnns/    # RL-trained LSTM policies on SequentialInferenceEnv
├── fit_tinyrnns/         # Supervised TinyGRUPolicy fits to monkey choice data
├── figures/              # Paper / poster figure assets
└── presentation/         # Talks and posters
```

Package-specific detail lives in [`fit_tinyrnns/README.md`](fit_tinyrnns/README.md) and [`neuralSPRT/paper_reproduction/README.md`](neuralSPRT/paper_reproduction/README.md).

---

## Setup

Python 3.10+ recommended. Create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch numpy scipy h5py matplotlib tqdm scikit-learn \
            gymnasium statsmodels
```

- `fit_tinyrnns` and `train_taskoptrnns` use **PyTorch**.
- `train_taskoptrnns` additionally needs **Gymnasium**.
- `neuralSPRT` analysis uses **SciPy / statsmodels / matplotlib**; `.mat` files are read via `scipy` / `h5py` (`mat73`-style loaders in `fit_tinyrnns`).

Scripts generally assume you run them **from inside their package directory** (flat imports such as `from dataset import ...`).

---

## Data

Place Kira-lab monkey files under `neuralSPRT/data/` (gitignored):

| File | Monkey |
|------|--------|
| `dataE.mat` | Eli (`monkey_id = 1`) |
| `dataJ.mat` | Joey (`monkey_id = 2`) |

Optional LFP: `dataE_LFP.mat`. Contact the lab / authors if you need access to the recordings.

Each trial is a sequence of shape categories. Fitted models take per-timestep one-hot inputs over 8 shapes and predict actions in `{choose_A, choose_B, continue}`.

---

## Quick start

### 1. Monkey analysis (`neuralSPRT`)

Reproduce paper-style behavioral / physiology figures:

```bash
cd neuralSPRT/paper_reproduction
python neuralSPRT.py --id 2 --beh 1 2 3 4 5 6 --phy 1 2 3 4
```

Or work from the notebooks `analysis_MonkeyE.ipynb` / `analysis_MonkeyJ.ipynb` in `neuralSPRT/`.

### 2. Task-optimized RNN (`train_taskoptrnns`)

Train an A2C agent on `SequentialInferenceEnv` (sample vs decide, with sampling / urgency costs):

```bash
cd train_taskoptrnns
python training.py \
  --hidden_size 64 \
  --reward 1.0 --sampling_cost 0.01 --urgency_cost 0.0 \
  --num_episodes 1500000 \
  --path ./results
```

Cluster helpers: `submit.sh`, `submit.slurm`. Post-training plots: `plotting.py`, `plot_psample.py`, `sample_cost.py`.

### 3. Fit tiny GRUs to monkey data (`fit_tinyrnns`)

Single model or hidden-size sweep:

```bash
cd fit_tinyrnns
python train_single.py \
  --monkey-id 1 \
  --hidden 2 \
  --epochs 500 --lr 1e-3 --batch-size 128 \
  --diagonal-readout --recurrent-l1-lambda 1e-5 \
  --outdir runs/tiny_gru_single
```

Nested cross-validation over capacity / regularization:

```bash
python train_nested_cv.py \
  --monkey-id 1 \
  --hidden 1 2 3 4 5 10 20 \
  --recurrent-l1 1e-5 \
  --outer-folds 5 \
  --outdir runs/tiny_gru_choice_nestedcv
```

One-shot analysis report from a checkpoint:

```bash
python analysis.py --ckpt runs/.../model.pt
```

See [`fit_tinyrnns/README.md`](fit_tinyrnns/README.md) for checkpoint format, nested-CV layout, and the full figure suite.

---

## Task in brief

At each step the agent observes a shape (evidence sample) and may:

- **continue / sample** — pay a cost and see another shape, or
- **choose A / B** — commit to a target and end the trial.

Hard-bound SPRT stops at fixed evidence thresholds; **soft-boundary** policies (and fitted tiny RNNs) produce graded, noisy stopping that better matches monkey psychometric and sampling curves.

---

## Citation / contact

If you use this code, please cite the associated presentation or paper materials under `presentation/` and `figures/`. For data access or questions, open an issue on the repository or contact the authors.
