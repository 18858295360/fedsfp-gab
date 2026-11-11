# Push-T experiments and archived Table 2 metrics

## Recompute the archived table

Run `python recompute_table2.py` from this directory. This uses only the Python standard library. The included CSV contains 450 checkpoint metrics from 18 archived evaluation runs. The script selects the five highest coverage scores out of the 25 even checkpoints in each run, averages the scores and corresponding success rates, and reports the mean and sample standard deviation over three seeds per method and client count. The result matches the archived Table 2 numbers.

`evidence/table2_saved_configs.json` contains the relevant saved arguments for each evaluation record. The paper specifies an anchor refresh interval of **one round**. The nine archived GAB configurations record **five rounds** and four workers. The historical loader could retain an earlier anchor cache in persistent workers. The archive lacks the checkpoint-specific training logs and source hashes needed to determine which runs used that implementation or those exact settings. The saved scripts could also evaluate existing checkpoints before writing their configuration. Recomputing the table verifies the aggregation of recorded scores.

The three split files contain demonstration indices, with identical membership across the six archived runs for each client count. They reference the same ordered 200 demonstrations. They contain no demonstration samples or model weights.

## Prepare the training and evaluation environment

The study scripts use a compatible Streaming Flow Policy source tree. `source_requirements.json` identifies the twelve external modules used by the audited Push-T runtime. Four are byte-identical at the pinned upstream revision. Seven modules differ and the required `trajectory.py` is absent at its expected path. Obtain the eight remaining files from the compatible study source tree before using the training and evaluation commands below. The four matching files can be taken from that tree or the pinned upstream revision. [Streaming Flow Policy upstream revision](https://github.com/siddancha/streaming-flow-policy/tree/ea08aa9bf24fe9960abc85d5ed69023ce7a91cc8)

Use Python 3.10 on Linux with matching CUDA PyTorch and torchvision wheels if training on GPU. The requirements describe API-compatible ranges, including the archived scheduler dependency `diffusers==0.11.1`. The exact historical environment was not recovered for all 18 runs.

```bash
RELEASE="$(pwd -P)"
SOURCE=/path/to/compatible-sfp-source
WORK="$HOME/pusht-repro-work"

python -m venv "$HOME/pusht-repro-venv"
source "$HOME/pusht-repro-venv/bin/activate"
python -m pip install -r "$RELEASE/requirements.txt"
python "$RELEASE/prepare_runtime.py" --source-root "$SOURCE" --work-dir "$WORK"
```

The preparation script checks hashes and Python syntax before it creates a new work directory. It accepts the audited or fixed dataset source and applies the two state-window index edits when given the audited version. The assembled runtime uses the verified trainer that recreates workers when GAB is enabled. The evaluator and its external dependencies come from the compatible source tree. Keep that assembled work directory outside this release. You can use `--verify-only` in place of `--work-dir` to check an external tree in memory.

## Run a fixed-source pair

Get the Push-T dataset from the [Diffusion Policy training data distribution](https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip). Extract it and set `DATA_DIR` to the directory containing `pusht_cchi_v7_replay.zarr`. The loader uses a Zarr 2 replay with `data/action`, `data/state`, `data/img` and `meta/episode_ends`. The split files depend on the ordering of the demonstrations. The archived run configurations do not provide a per-run dataset digest.

The commands use the fixed trainer, the paper's one-round refresh interval and the saved configuration for the remaining listed settings. Use fresh output directories. To compare the archived five-round setting under the fixed implementation, change both refresh arguments to `5` and use a different output location.

```bash
DATA_DIR=/path/to/directory/containing/pusht_cchi_v7_replay.zarr
OUT="$HOME/pusht-repro-outputs"
K=2
SEED=16
BASE="$OUT/k${K}_seed${SEED}_fedavg"
GAB="$OUT/k${K}_seed${SEED}_gab"
cd "$DATA_DIR"

common=(
  --base-split-json "$RELEASE/splits/rotation_k${K}.json" --custom-split
  --split-seed 20260709 --seeds "$SEED" --device cuda
  --obs-mode image --imitation-mode action --methods persistent_both
  --rounds 50 --local-epochs 20 --batch-size 128 --lr 0.0001
  --warmup-steps 500 --num-workers 4
  --eval-num-tests 50 --env-start-seed 1300 --max-rollout-steps 300
)
python "$WORK/scripts/pusht_sfp/p64_v23.py" "${common[@]}" \
  --out-dir "$BASE" --global-anchor-frac 0 --global-anchor-refresh 1
python "$WORK/scripts/pusht_sfp/p64_v23.py" "${common[@]}" \
  --out-dir "$GAB" --global-anchor-frac 0.5 --global-anchor-refresh 1
```

Repeat for `K=2,4,8` and training seeds `16,42,43`. The trainer seeds both arms identically before initializing the same architecture. The recipe retains optimizer state across rounds and weights clients by their episode counts, as recorded in these saved configurations. The trainer evaluates every even checkpoint from round 2 to 50. It uses deterministic initial noise, Euler integration, a prediction horizon of 16, an execution horizon of 8, and the current pusher as the action start. FLARE variants are outside this component's Table 2 workflow.

To evaluate a checkpoint generated by that run independently, use the following command from `DATA_DIR`. The train-and-evaluate script above invokes the same external evaluator.

```bash
python "$WORK/streaming_flow_policy/pusht/evaluate.py" \
  --model-type sfps --obs-mode image --imitation-mode action \
  --ckpt-path "$GAB/seed${SEED}_split20260709_p1p0/persistent_both/round_0050.pth" \
  --save-dir "$OUT/k${K}_seed${SEED}_eval" --name round_0050 --device cuda \
  --solver euler --integration-steps-per-action 1 --deterministic-z0-eval \
  --seed "$SEED" --num-tests 50 --env-start-seed 1300 --max-rollout-steps 300 \
  --success-score-threshold 0.95 --obs-horizon 2 --action-horizon 8 \
  --image-encoder final_v1 --image-cond-dim 256 --image-agent-pos-cond --image-crop-size 84 \
  --image-key img --rollout-start-mode current_pusher \
  --episode-indices-json "$RELEASE/splits/rotation_k${K}.json" \
  --stats-episode-indices-key union_200
```

## Validation boundary

The release preparation verified the source hashes, checked Python syntax, and recalculated the archived table. Full GPU training and environment rollouts with this packaged workflow are unverified. The fixed trainer is intended for new runs. Its results may differ from the archived Table 2 outcomes.
