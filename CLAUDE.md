# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PointWorld: a 3D world model that predicts full-scene 3D point flows from partially-observable
RGB-D captures and robot actions (also represented as 3D point flows), for in-the-wild robotic
manipulation. This `main` branch is the **training/evaluation code** release; a separate `data`
branch (not this checkout) holds the dataset preparation pipeline (`data_integrity_check.py`,
`convert_wds.py`). Prepare data on `data` first, then use `main` to train/evaluate.

Paper: https://arxiv.org/abs/2601.03782

## Environment setup

The `pointworld-env` conda env already exists on this machine — just `conda activate pointworld-env`
rather than re-running `conda env create`. Only recreate it (or `conda env update --prune`) if
dependencies are missing/out of date.

```bash
conda env create -n pointworld-env -f environments/train_eval.yml
conda activate pointworld-env
python -m pip install huggingface_hub==0.26.2
python -m pip install timm==1.0.19 --no-deps          # PTv3 DropPath
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install networkx==3.4.2 --no-deps       # urdfpy-compatible pin
```

Add `environments/train_eval_viz.yml` (via `conda env update --prune`) for visualization extras
(`matplotlib`, `open3d`, `viser`).

`environments/requirements.txt` is the canonical pinned dependency list (torch 2.5.1 / cu124,
torch-scatter, spconv, numba, urdfpy, webdataset, etc.) — check it before adding/upgrading deps.

DINOv3 is a git submodule and is a hard runtime dependency of the scene encoder
(`third_party/dinov3/`, requires `git submodule update --init --recursive` plus a downloaded
`.pth` checkpoint placed under `third_party/dinov3/checkpoints/`). `scene_featurizer.py` raises a
clear error at load time if the submodule or weights are missing — don't work around it, fix the
submodule/checkpoint.

There is no test suite in this repo; correctness is validated by running training/eval smoke
tests against small dataset subsets.

## Common commands

Dataset directories are expected at `<LOCAL_DATASET_DIR>/{droid,behavior}/wds` (see
`arguments.py: DOMAIN_TO_DATA_DIR`), and pretrained checkpoints under `pretrained_checkpoints/`
(download via `huggingface-cli download nvidia/PointWorld_models --local-dir pretrained_checkpoints ...`).

Train (single domain):
```bash
python train.py --domains=droid --data_dirs=/path/to/droid/wds \
  --norm_stats_path=stats/droid --batch_size=<B> --num_workers=<NW> \
  --eval_num_workers=<ENW> --eval_freq=-1
```

Train (multi-domain, DROID + BEHAVIOR): pass `--domains=droid,behavior`,
`--data_dirs=<droid_wds>,<behavior_wds>`, `--norm_stats_path=stats/droid_behavior`.

DDP: `torchrun --standalone --nproc_per_node=<N> train.py --distributed=true <args>`.

Eval:
```bash
python eval.py --model_path pretrained_checkpoints/large-droid/model-best.pt \
  --domains=droid --data_dirs=/path/to/droid/wds \
  --confidence_thres=0.8 --batch_size=1 --eval_num_batches=-1   # -1 = full dataset, or e.g. 100 for quick iteration
```
BEHAVIOR eval needs a checkpoint trained on both domains (`large-droid+behavior/model-best.pt`)
and `--norm_stats_path=stats/droid_behavior`; it skips the expert-confidence filtering DROID uses
since BEHAVIOR (simulated) data is noiseless.

Eval-time visualization (viser, opens `http://localhost:<viewer_port>`):
add `--eval_viz_num=8 --viewer_port=8080` to the eval command (`--eval_skip_viz=true` to force off).
The visualizer prompts `Press ENTER to continue` between samples — requires a real TTY stdin.

`--ptv3_size` selects the PTv3 backbone variant (`small|base|large`, default `base`); architecture
per size lives in `ptv3/ptv3_arch.yaml`, not in Python.

Eval outputs land in `eval_logs/<timestamp>-<exp_name>-split=<split>-seed=<seed>/`; train logs in
`train_logs/<exp_name>/`. Both are gitignored working directories, not part of the release itself.

## Architecture

**Entry points**: `train.py` and `eval.py` are thin wrappers — both call `arguments.parse_args()`
then hand off to `training/trainer.py::Trainer` (`eval.py` uses `evaluation/tester.py::Tester`,
which subclasses `Trainer` and overrides dataset/domain resolution for eval-only semantics).
`arguments.py` is the single source of CLI args; it also converts `'true'/'false'`/`'none'`
strings to real bool/None and validates domain↔data_dir↔feature-set consistency. Read it before
guessing what a flag does.

**Checkpoint contract (`pointworld/checkpoint_contract.py`)**: every saved checkpoint embeds a
`model_contract` (architecture-defining args like `ptv3_size`, `grid_size`, `max_scene_points`,
camera-count bounds, `norm_stats_path`) and `data_contract` (`domains` the model was trained on).
On load, `Trainer`/`Tester` apply the checkpoint's contract onto `args`, and **raise** if an
explicit CLI flag conflicts with what the checkpoint requires — checkpoints are the source of
truth for model shape, not the CLI. When adding a new architecture-defining arg, register it in
`MODEL_CONTRACT_KEYS` (and a legacy default if old checkpoints won't have it).

**Data pipeline (`dataset_components/`)**: WebDataset (`.tar`) shards →
`dataloader.py::build_dataset` chains: `decode_data` (raw sample decode) → `build_flow_sample`
(uses `robot_sampler.py::RobotSampler`, a URDF-driven robot point sampler, to compute robot point
flows from joint states) → `sample_cameras` → `pipeline.py::sample_transform_pipeline` (grid
sampling / sphere-crop / rotation / scale / flip / chromatic augmentation, train vs test mode
differ) → `gather_features` (selects the `--robot_features`/`--scene_features` subset) →
`convert_to_tensors` → `collate.py::custom_collate_fn`. `constants.py` holds all the
`RELEASE_*` fixed augmentation hyperparameters (release CLI trims most of these from flags).
Domains are `droid` (real-robot) and `behavior` (BEHAVIOR sim benchmark, bimanual-capable);
`has_bimanual_robot` toggles extra fields (`left/right_gripper_pose`, `base_pose`, etc.).

**Model (`pointworld/base.py::BaseModel`)**:
1. `SceneFeatureEncoder` (`scene_featurizer.py`) fuses two scene-feature sources: a frozen
   DINOv3 ViT-L16 backbone (`SceneEncoder2D`) that projects 3D scene points into each camera view,
   samples multi-layer patch tokens (layers 4/11/17/23) via `grid_sample`, and aggregates across
   visible cameras with a depth-consistency visibility mask; plus the raw precomputed
   `scene_features` from the dataset. Both are LayerNorm'd and concatenated/projected.
2. Robot features are projected and combined with a sinusoidal `TemporalEmbedding`
   (`pointworld/embeddings.py`) and a learned robot-type embedding.
3. Scene + robot (all T timesteps) tokens are packed into one point cloud and passed through
   `DynamicsPredictor`, which runs a vendored **PointTransformerV3** (`ptv3/ptv3.py`, architecture
   sizes in `ptv3/ptv3_arch.yaml`) and predicts, per scene point, a 3-vector displacement for each
   of `PRED_HORIZON=10` future steps (`CONTEXT_HORIZON=1` input step) plus a heteroscedastic
   log-variance head used to derive a `confidence` score. Skip connections use FiLM modulation
   from both the input scene features and a max-pooled global robot-feature summary.
4. Normalization stats (per-domain, per-timestep mean/var for flows; per-domain mean/var for
   robot/scene raw features) are loaded from JSON in `--norm_stats_path` (see `stats/droid`,
   `stats/droid_behavior`) via `pointworld/norm_stats.py` and applied/inverted around the model
   boundary (`normalize`/`unnormalize`).
5. `pointworld/losses.py` and `pointworld/metrics.py` hold the Huber + uncertainty loss and the
   metric aggregation (including domain-split and moved-vs-static point breakdowns); `BaseModel`
   just delegates to them. `behavior` (sim) domains get a fixed/clamped variance
   (`SIM_VAR_CONST`) instead of learned uncertainty, since sim data is noiseless.

**Training loop (`training/trainer.py::Trainer`)**: single AdamW optimizer, AMP autocast
(bf16 if supported), gradient clipping, DDP via `torchrun`+`DistributedDataParallel`. Cadence for
eval/checkpoint-save is driven by **batch count**, not wall-clock time (`--eval_freq`/`--save_freq`
are batch intervals). Extensive NaN-guarding throughout (`utils.py::handle_nan_outputs`,
`handle_nan_grad_norm`, `check_model_parameters_for_nan`) skips bad batches and, in DDP, uses an
`all_reduce(MAX)` over a CPU process group (`self.cpu_pg`) so all ranks skip in lockstep — never
let one rank diverge from another here. On repeated consecutive NaNs training raises
`NaNDetectionError` and saves an emergency checkpoint before exiting.

**Evaluation (`evaluation/tester.py::Tester`)**: reconstructs `args.domains`/`data_dirs` from the
checkpoint's data contract (not from the CLI), then restricts to the domains actually requested
via `--domains` (must be a subset of what the checkpoint was trained on). DROID evaluation can use
an expert-confidence H5 artifact (`droid/wds/test/expert_confidence-seed=42.h5`) to compute
`full_eval/test/filtered_l2_moved/mean`, the primary DROID metric — filters to reliable
moving-point regions. `evaluation/metrics.py`, `evaluation/meta.py`, and
`evaluation/annotation.py::ConfidenceHelper` support this. Eval outputs are **not** perfectly
deterministic on GPU even with fixed seeds; partial (`eval_num_batches < full`) runs are also
sensitive to `num_workers`/`eval_num_workers` — match these when comparing runs.

**Visualization (`visualization/`)**: `viser`-based. `visualization/prediction_viz/` renders
eval-time predictions (invoked from `evaluation/tester.py` when `--eval_viz_num > 0`);
`visualization/viser_flow/` and `visualization/viser_tools/` provide lower-level scene/robot-flow
viser building blocks (camera layers, robot overlays, timeline scrubbing, upsampling).

**Vendored/adapted third-party code** (see `THIRD_PARTY_LICENSES.md` for full attribution —
check it before modifying these, license terms apply):
- `ptv3/` — Point Transformer V3, adapted from Pointcept/PointTransformerV3 (MIT), with sonata
  lineage influence.
- `third_party/dinov3/` — Meta DINOv3, git submodule, frozen scene-encoder backbone.
- `transform_utils.py` — geometry/quaternion utilities adapted from OmniGibson (MIT) and
  deoxys_control (Apache-2.0).

## Working practices

- Commit work to git regularly as you go, with clean, descriptive commit messages — don't let a
  long working session accumulate into one giant uncommitted diff.
- `main` is protected: don't push directly to it. Work on a feature branch, push that branch
  regularly (so work-in-progress isn't only sitting locally and can't be lost), and open a PR for
  review/merge into `main`.
- `origin` is the read-only upstream `NVlabs/PointWorld` repo — pushing there will fail with a
  permissions error. Push feature branches to the `fork` remote instead
  (`https://github.com/SergioMOrozco/PointWorld.git`), and open PRs from there.

## Conventions worth knowing

- Boolean/optional CLI flags are passed as strings (`--distributed=true`, `--domains=droid,behavior`)
  and normalized inside `arguments.parse_args`; don't assume argparse `bool`/`None` types directly
  on the raw `ArgumentParser` actions.
- `--domains` and `--data_dirs` must be parallel comma-separated lists of equal length (or omit
  `--data_dirs` to use the `DOMAIN_TO_DATA_DIR` convention-based default).
- `args.robot_features`/`args.scene_features` are validated against a fixed expected list for this
  release — arbitrary custom feature sets will raise, since the model shape is baked to them.
- Restored dataset directories (`dataset_components/*-subset*/`, `pointworld_droid_subset_restored/`)
  and downloaded checkpoints (`pretrained_checkpoints/*/model-*.pt`) are large, git-ignored,
  locally-restored artifacts, not part of the tracked source tree.
