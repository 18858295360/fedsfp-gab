# Partition diagnostics

Install NumPy and h5py in a separate Python environment. Push-T also requires Zarr 2 and a compatible NumCodecs.

```bash
python -m pip install 'numpy>=1.24,<2' h5py 'zarr>=2.16,<3' 'numcodecs<0.16'
```

## RoboMimic

```bash
python diagnostics/robomimic_diagnostics.py square --clients 2 --dataset /path/to/square_mh_state_v2.hdf5
python diagnostics/robomimic_diagnostics.py can --clients 4 --dataset /path/to/can_abs_actions.hdf5
python diagnostics/robomimic_diagnostics.py lift --clients 8 --dataset /path/to/lift_abs_actions.hdf5
```

The included files identify five levels for each task and each client count. Use the input export corresponding to each diagnostic. Square uses the state-v2 observation export; Can uses the continuous-gripper absolute-action export; Lift uses the continuous-gripper absolute-action export with the Lift object field layout. The 60-draw, size-matched reference is shared across the five levels of each ladder.

## LIBERO-10

```bash
python diagnostics/libero_peak_ratio.py --dataset-dir /path/to/libero_10 \
  --split-dir libero/splits --output peak-ratios.json
python diagnostics/libero_trajectory_separation.py --dataset-dir /path/to/libero_10 \
  --split-dir libero/splits
```

The scripts use lexicographically sorted task-file order and the frozen `<task index>|<demo name>` IDs. The trajectory calculation aligns end-effector and task-object paths at 11 arc-length phases. The matched-action diagnostic compares flattened 16-step action chunks within each task and relative-time bin, excluding windows from the same demonstration. It redoes the matching for 1,000 size-preserving whole-demonstration permutations. `peakS` is the maximum raw cross/within distance ratio across bins. The `obs_bad` and `perm_bad` fields record non-computable observed bins and permutation rows, respectively. An archived log in the expected format may be supplied with `--reference-log` to check the standardized readout.

For the earlier standardized phase-scan view of one split, use `libero_dcond.py --dataset-dir /path/to/libero_10 --split /path/to/split.json`. It reports `D` as the ratio divided by the permutation mean and `max-z` as the maximum standardized value.

## Push-T

```bash
python diagnostics/pusht_diagnostics.py --dataset /path/to/pusht_cchi_v7_replay.zarr.zip --control
```

The control checks the recovered K=2 structured endpoint against the recorded values. The script accepts additional assignment paths as JSON arguments. The historical intermediate Push-T diagnostic path inputs are not included here.
