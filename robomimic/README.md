# RoboMimic reproduction

This component covers the low-dimensional Square, Can, and Lift multi-human (MH) experiments in *Controlled Execution-Style Heterogeneity in Federated Imitation Learning*. It includes the federated trainer, global learning-rate wrapper, split memberships, action conversion, and switch-aware checkpoint evaluation. Obtain demonstrations from RoboMimic and generate checkpoints locally.

## Environment

Run the commands from this `robomimic` directory. Use Python 3.11 on Linux. The example selects PyTorch 2.5.1 and torchvision 0.20.1 with CUDA 12.1, the pair observed in the evaluation environment. For another CUDA version, use the corresponding PyTorch installation instructions.

```bash
python3.11 -m venv ../.venv-robomimic
source ../.venv-robomimic/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

git clone https://github.com/siddancha/streaming-flow-policy.git ../streaming-flow-policy-upstream
git -C ../streaming-flow-policy-upstream checkout --detach c0458f1a228b8e26490707347f45e224bbca5f34
export PYTHONPATH="$(pwd)/../streaming-flow-policy-upstream${PYTHONPATH:+:$PYTHONPATH}"
python -m scripts.robomimic_sfp.fedavg_glr --help
```

The original Streaming Flow Policy checkout supplies `ConditionalUnet1D`; the study's RoboMimic modules are included here. Install RoboMimic and robosuite as external dependencies. Their upstream terms apply. [Streaming Flow Policy](https://github.com/siddancha/streaming-flow-policy), [RoboMimic installation](https://robomimic.github.io/docs/introduction/installation.html), [PyTorch builds](https://pytorch.org/get-started/previous-versions/)

## Demonstrations and client memberships

Download the **robosuite 1.5.1, low-dimensional MH** files for all three tasks. Check that the HDF5 data contain `demo_0` through `demo_299`, simulator states, environment metadata, and the original demonstration ordering. The split IDs are those public dataset indices. [RoboMimic v0.1 datasets](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html)

```bash
python -m robomimic.scripts.download_datasets \
  --download_dir data/source --tasks square can lift --dataset_types mh --hdf5_types low_dim
```

Training expects seven action values per frame: world-frame EEF position, axis-angle orientation, and gripper open fraction in `[0, 1]`. The default observation keys are `object`, `robot0_eef_pos`, `robot0_eef_quat`, and `robot0_gripper_qpos`. The historical Can export used live sequential conversion, Lift used the earlier absolute-action converter, and the archived Square performance arms used a state-conditioned export. The state-conditioned script here implements the recorded-state conversion path. The example regenerates a Can-format file from a user-supplied source.

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
mkdir -p data runs
python -m scripts.robomimic_sfp.convert_table3_absolute_actions_live_v2 \
  --input /path/to/can/mh/low_dim_v15.hdf5 --output data/can_abs.hdf5 \
  --report data/can_conversion.json
```

Use `convert_table3_absolute_actions` for the Lift format and `convert_absolute_actions_state_v2` for the archived Square format. Both take `--input`, `--output`, `--report`, and `--workers`. Inspect the conversion report before training. The exact converter version for the historical Square export has not been recovered. Regenerated HDF5 files have not been compared byte-for-byte to the historical exports.

| Task | Main performance memberships |
| --- | --- |
| Can | `splits/can_pld_h15_k{2,4,8}.json` |
| Lift | `splits/lift_pld_h15_k{2,4,8}.json`, identical by client to the historical `h15_median` memberships |
| Square | `splits/square_apprc_k2.json`, `splits/square_apprh_k4.json`, `splits/square_apprh_k8.json` |
| Centralized control | `splits/pooled_k1.json` for each task |

The three `square_pld_h15_*` files are the unified PLD-H15 reference. The archived Square performance memberships differ from those references by 3, 7, and 14 demonstrations at `K=2,4,8`, after optimal client relabeling. To rebuild a reference partition from matching observations, use the same lift-height detector, diameter split, and recursive circular-median refinement as the registered constructor.

```bash
python -m scripts.robomimic_sfp.build_pld_h15 \
  --task can --dataset-path data/can_abs.hdf5 --clients 2 \
  --output data/can_rebuilt_k2.json
```

## Train a paired arm

Use the global-cosine wrapper so AdamW resets each round and the learning-rate schedule advances across rounds. The command below is one Can GAB recipe. For FedAvg, use `--global-anchor-frac 0` and a new output directory. For the pooled control, also use the `pooled_k1.json` membership file. The recipe uses full-pool normalization and weights federated aggregation by the number of training windows.

```bash
python -m scripts.robomimic_sfp.fedavg_glr \
  --dataset-path data/can_abs.hdf5 --split-json splits/can_pld_h15_k2.json \
  --output-dir runs/can-k2-gab-s46 --seed 46 --optimizer-mode raw \
  --rounds 250 --local-epochs 7 --batch-size 256 --num-workers 2 \
  --lr 1e-4 --weight-decay 1e-6 --warmup-steps 500 --save-every-rounds 2 \
  --sigma0 0.4 --k 10 --global-anchor-frac 0.5 --global-anchor-refresh 1
```

The paper's main Square and Lift recipes use `--sigma0 0.8 --warmup-steps 50 --global-anchor-refresh 5`. The paper's current-export Square setting uses `--sigma0 0.4`. The Can recipe uses refresh 1 and warmup 500. Can's refresh cadence was selected using the reported `K=8` evaluation. Use the seed and partition of the arm you want to recreate. The trainer is the audited implementation, with historical per-checkpoint source identity and numerical reproduction still unverified. Use the default observation keys. The archived optional `--obs-keys` switch does not propagate through the dataset constructor.

## Evaluate and aggregate

Use checkpoints you produced or trust. The evaluator loads serialized normalization arrays alongside model weights. The command below evaluates rounds 228–250 in steps of two, uses seeds 100000–100049, caps each episode at 500 steps, executes eight actions per prediction, and carries the previous action. It writes a per-round JSON/log and `summary.json` with Avg5 and mean12. Avg5 uses the five best scores from these same evaluations.

```bash
python -m scripts.robomimic_sfp.evaluate_last12 \
  --dataset-path data/can_abs.hdf5 --checkpoint-dir runs/can-k2-gab-s46 \
  --output-dir runs/can-k2-gab-s46-eval
```

Use a new training or evaluation output directory for every run. Run `python -m scripts.robomimic_sfp.rollout_table3_paper_action_anchor_v3_switchaware --help` for single-checkpoint evaluation. Cross-seed means and standard deviations require the paper's corresponding arm cohort. This component covers the last-12 Avg5 protocol. Historical held-out checks and probes follow their own protocols.
