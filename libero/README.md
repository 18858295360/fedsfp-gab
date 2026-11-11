# LIBERO-10

This component contains the FedAvg and GAB training entry point, the SFP rollout evaluator, and the recorded LIBERO-10 demo assignments. The supported recipe is image observations and recorded action targets with the v2 image encoder. Run the commands below from this directory on Linux.

## Dependencies

The archived training used Python 3.10, PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128`, RoboMimic `0.2.0`, NumPy `1.26.4`, h5py `3.11.0`, and diffusers `0.11.1`. Install the matched CUDA wheels for your driver, for example:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 h5py==3.11.0 robomimic==0.2.0 \
  diffusers==0.11.1 huggingface-hub==0.11.1
```

The SFP UNet uses a module from the [Streaming Flow Policy implementation](https://github.com/siddancha/streaming-flow-policy). Fetch that dependency into a separate directory. The `ConditionalUnet1D` class and its five dependent classes at the pinned revision match the original training dependency.

```bash
export SFP_ROOT=/absolute/path/to/streaming-flow-policy
git clone https://github.com/siddancha/streaming-flow-policy.git "$SFP_ROOT"
git -C "$SFP_ROOT" checkout ea08aa9bf24fe9960abc85d5ed69023ce7a91cc8
export PYTHONPATH="$PWD:$SFP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Rollout additionally requires a working [LIBERO simulator installation](https://github.com/Lifelong-Robot-Learning/LIBERO#installtion) with the `libero_10` suite, assets and prescribed initial states. The archived rollout environment used a separate Python 3.10 environment with the same PyTorch and torchvision wheels, LIBERO revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`, RoboMimic `0.3.0`, Robosuite `1.4.0`, MuJoCo `2.3.7`, BDDL `1.0.1`, and Gym `0.25.2`. Install LIBERO from that revision and follow its simulator setup with those versions. Keep `SFP_ROOT` on `PYTHONPATH` and set `MUJOCO_GL=egl` for headless GPU rendering.

## Data and frozen assignments

Get the original LIBERO-10 HDF5 demonstrations from the [LIBERO dataset release](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets). Point `DATA` at the directory with the ten `*_demo.hdf5` files and no other HDF5 files. File index is the sorted basename order in [`libero10_task_order.json`](libero10_task_order.json). Each file contains `data/demo_0` through `data/demo_49`, 7D `actions`, and `obs/agentview_rgb`, `obs/eye_in_hand_rgb`, `obs/ee_states`, `obs/gripper_states` and `obs/joint_states`. Both RGB cameras have dimensions 128 × 128 × 3. The three proprioception arrays contain 6, 2, and 7 components.

Supply the original task-language NPZ as `LANG_NPZ`. Its `tasks` entries match the HDF5 basenames with `_demo.hdf5` removed, and `emb` contains one float32 vector of width 384 per task. The archived NPZ has SHA-256 `c392dc42debb7ca1eb5fbf1d87870386d45f0e5f191fa162dce72e7f19c3e56f`. The original language encoder and its generation settings were not established in the audit. An independently generated NPZ with the same shape can exercise the pipeline but changes the training/evaluation input. The demonstrations and NPZ are external inputs to this release.

The JSON files in [`splits`](splits) preserve the archived membership order and `file_index|demo_key` values. [`split_provenance.json`](split_provenance.json) gives the hashes before and after removal of unrelated metadata. The `closelift_S` files are the structured endpoint and centralized control. The five `sev*_PLD_K2` files and the four `pathPLD_K*_s*` files at each of K=4 and K=8, together with their `closelift_S` endpoints, supply the diagnostic paths. All splits cover each of the 500 demonstrations once. The K=4 and K=8 structured assignments preserve the previous level's parent memberships.

```bash
export DATA=/absolute/path/to/libero_10
export LANG_NPZ=/absolute/path/to/libero_10_lang.npz
K=2
SPLIT="$PWD/splits/libero10_closelift_S_K${K}.json"
python validate_inputs.py --libero-root "$DATA" --lang-npz "$LANG_NPZ" --split "$SPLIT"
```

This check reads HDF5 schema and frame counts. It verifies 130,590 full 16-step windows and the language asset hash. It cannot authenticate individual HDF5 image or action values.

## Train one arm

Use `Q=0` for FedAvg and `Q=0.5` for GAB. For the pooled reference, use `K=1`, `Q=0`, and its single-client file. The main comparison uses seeds `43 44 45` at K=2 and `42 43 44` at K=4 and K=8. The centralized reference uses `42 43 44`. For the K=2 path-zero comparison, set `SPLIT="$PWD/splits/libero10_sev0p00_PLD_K2.json"` and keep the K=2 seed set.

```bash
Q=0.5
SEED=43
SPLIT_NAME="${SPLIT##*/}"
OUT="$PWD/runs/${SPLIT_NAME%.json}_q${Q}_seed${SEED}"
test ! -e "$OUT" || exit 1

python -m scripts.robomimic_sfp.train_adroit_federated \
  --dataset-class libero --policy sfp --libero-root "$DATA" --libero-lang "$LANG_NPZ" \
  --libero-partition mode --libero-mode-clients "$K" --libero-mode-feature "file:$SPLIT" \
  --obs-mode image --imitation-mode action --image-encoder v2 --img-size 128 --num-kp 32 \
  --gripper-source binary_command --normalizer-kind legacy --optimizer-mode sched_only \
  --rounds 20 --local-epochs 1 --steps-per-epoch 2000 --batch-size 256 --num-workers 2 \
  --lr 1e-4 --lr-shape cosine --warmup-steps 50 --weight-decay 1e-6 --sigma0 0.4 --k 10 \
  --global-anchor-frac "$Q" --global-anchor-refresh 10 --global-anchor-subsample 0.3 \
  --seed "$SEED" --save-every-rounds 1 --output-dir "$OUT"
```

The round budget is 40,000 optimizer steps per client. AdamW moments restart per round; the warmup/cosine clock continues per client. Aggregation weights count prediction windows. The v2 encoder uses both cameras, 15D proprioception and 384D language projected to 64D. Keep the default UNet widths `[256, 512, 1024]` and time-embedding dimension 256 to match the reported recipe.

## Evaluate one arm

```bash
export MUJOCO_GL=egl
python evaluate_selected.py --checkpoint-dir "$OUT" --lang-npz "$LANG_NPZ" \
  --output-dir "${OUT}_evaluation"
```

The runner evaluates rounds 4–20 on two episodes per task, selects the five highest pooled success rates with earlier rounds breaking ties, and re-evaluates those five rounds on 20 episodes per task. It writes each rollout summary and the arm mean across all five re-evaluations. Each rollout uses eight actions per policy query, a 400-step cap, and the benchmark's supplied initial states. Table cells use the mean and sample standard deviation across the stated training seeds.

For a single checkpoint instead:

```bash
python -m scripts.robomimic_sfp.rollout_libero \
  --checkpoint "$OUT/global_round_0020.pt" --benchmark libero_10 --tasks 10 \
  --rollouts-per-task 20 --action-horizon 8 --max-steps 400 --img-size 128 \
  --lang-npz "$LANG_NPZ" --output "${OUT}_r20.json"
```

## Source and verification limits

The training, data, image-model and rollout files come from the audited LIBERO implementation. The release replaces cluster-specific path defaults with local input options, keeps the LIBERO imports separate from unrelated Adroit/RoboMimic training branches, and lets `--lang-npz` point to a relocated copy of the training embeddings. `evaluate_selected.py` transcribes the archived two-stage launch procedure. The CLI help and input validator were checked locally. The input check used archived dataset shape metadata, not the demonstration arrays. The released source has not been used to repeat full training or simulator rollouts.

- The archived default `--k 10` is effective for this recipe. Non-default values are lost in the dataset-construction path. The archived GAB run uses `--global-anchor-subsample 0.3`; a value of zero falls back to full subsampling in the original code. The LIBERO path also omits the client size field needed by the optional inverse-size anchor weighting.
- Training-cache construction starts from a noised expert action and uses 15 Euler intervals at horizon 16. The rollout starts from the supplied previous action without an extra noise draw and integrates in steps of 1/16. The environment receives clipped commands while the next chunk carries the unclipped integral endpoint.
- The released CLI reproduces the recorded cohort and recipe. Archived audits validated the K=2/K=4 data indexing, model reconstruction and bounded offline checkpoint inference. Full trained success rates, exact simulator setup and bitwise reproducibility remain separate checks.
