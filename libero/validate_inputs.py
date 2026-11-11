"""Check LIBERO-10 demo indexing, observation schema and embedding identity."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py


LANGUAGE_SHA256 = "c392dc42debb7ca1eb5fbf1d87870386d45f0e5f191fa162dce72e7f19c3e56f"
OBS_SHAPES = {"agentview_rgb": (128, 128, 3), "eye_in_hand_rgb": (128, 128, 3),
              "ee_states": (6,), "gripper_states": (2,), "joint_states": (7,)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--lang-npz", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()

    task_file = Path(__file__).with_name("libero10_task_order.json")
    expected = json.loads(task_file.read_text(encoding="utf-8"))
    paths = sorted(args.libero_root.glob("*.hdf5"))
    if [path.name for path in paths] != expected:
        parser.error(f"expected the exact HDF5 basename order in {task_file.name}")
    if hashlib.sha256(args.lang_npz.read_bytes()).hexdigest() != LANGUAGE_SHA256:
        parser.error("language NPZ does not match the recorded training asset hash")

    windows = {}
    expected_keys = {f"demo_{i}" for i in range(50)}
    for file_index, path in enumerate(paths):
        with h5py.File(path, "r") as h5:
            data = h5["data"]
            if set(data) != expected_keys:
                raise ValueError(f"expected demo_0 through demo_49 in {path}")
            for key, demo in data.items():
                action_shape = demo["actions"].shape
                if len(action_shape) != 2 or action_shape[1] != 7 or action_shape[0] < 16:
                    raise ValueError(f"invalid actions shape for {path.name}/{key}: {action_shape}")
                frames = action_shape[0]
                for name, shape in OBS_SHAPES.items():
                    if demo["obs"][name].shape != (frames, *shape):
                        raise ValueError(f"unexpected {name} shape for {path.name}/{key}")
                windows[f"{file_index}|{key}"] = frames - 15

    groups = json.loads(args.split.read_text(encoding="utf-8"))["clients"]
    flattened = [entry for group in groups.values() for entry in group]
    if len(flattened) != len(windows) or set(flattened) != windows.keys():
        raise ValueError("split must assign every demonstration exactly once")
    if sum(windows.values()) != 130590:
        raise ValueError("prediction-window count differs from the archived training data")
    counts = {name: sum(windows[entry] for entry in group) for name, group in groups.items()}
    print(json.dumps({"demonstrations": len(windows), "training_windows_by_client": counts}, indent=2))


if __name__ == "__main__":
    main()
