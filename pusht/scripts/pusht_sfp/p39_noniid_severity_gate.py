"""Helpers from the study module p39_noniid_severity_gate.py used by the Push-T trainer."""

import numpy as np

from streaming_flow_policy.pusht.dataset import PushTStateActionDataset


def initial_x_by_episode():
    root = PushTStateActionDataset.GetDatasetRoot()
    state = root["data"]["state"][:]
    episode_ends = root["meta"]["episode_ends"][:]
    starts = np.concatenate(([0], episode_ends[:-1]))
    return {episode: float(state[int(starts[episode]), 0]) for episode in range(len(episode_ends))}


def make_bins(base_split, x_values):
    union = [int(ep) for ep in base_split["union_200"]]
    ranked = sorted(((x_values[ep], ep) for ep in union), key=lambda item: item[0])
    sizes = [len(base_split[f"client_{idx}"]) for idx in range(3)]
    bins = []
    cursor = 0
    for size in sizes:
        eps = [int(ep) for _, ep in ranked[cursor: cursor + size]]
        bins.append(eps)
        cursor += size
    return union, ranked, sizes, bins


def make_severity_split(base_split, severity_p, split_seed):
    if not (0.0 <= severity_p <= 1.0):
        raise ValueError("severity_p must be in [0, 1]")
    rng = np.random.default_rng(split_seed)
    x_values = initial_x_by_episode()
    union, ranked, sizes, bins = make_bins(base_split, x_values)
    ep_to_bin = {int(ep): int(bin_idx) for bin_idx, eps in enumerate(bins) for ep in eps}

    client_eps = {idx: [] for idx in range(3)}
    used = set()
    private_counts = []
    for idx, size in enumerate(sizes):
        count = int(np.floor(severity_p * size + 0.5))
        count = min(count, size)
        private_counts.append(count)
        own_pool = np.asarray(bins[idx], dtype=np.int64)
        picked = rng.choice(own_pool, size=count, replace=False).astype(int).tolist() if count else []
        client_eps[idx].extend(picked)
        used.update(picked)

    slots = []
    for idx, size in enumerate(sizes):
        slots.extend([idx] * (size - len(client_eps[idx])))
    remaining = [int(ep) for ep in union if int(ep) not in used]
    rng.shuffle(remaining)
    rng.shuffle(slots)
    if len(remaining) != len(slots):
        raise RuntimeError(f"remaining episodes {len(remaining)} != remaining slots {len(slots)}")
    for episode, idx in zip(remaining, slots):
        client_eps[idx].append(int(episode))

    split = {
        "split": "Non-IID severity p-mix initial-x",
        "split_rule": "first200 initial pusher-x bins; each client gets p private own-bin samples, remaining slots filled globally without replacement",
        "severity_p": float(severity_p),
        "split_seed": int(split_seed),
        "num_demos": 200,
        "num_clients": 3,
        "union_200": union,
        "base_split": base_split.get("split", "unknown"),
        "base_split_seed": base_split.get("split_seed"),
        "feature": "initial_state[0]",
        "feature_min": float(ranked[0][0]),
        "feature_max": float(ranked[-1][0]),
        "bin_sizes": sizes,
        "private_own_counts": private_counts,
        "omitted_episodes": base_split.get("omitted_episodes", []),
    }
    split_rows = []
    for idx, size in enumerate(sizes):
        eps = sorted(int(ep) for ep in client_eps[idx])
        values = [x_values[ep] for ep in eps]
        origin_counts = {f"bin_{j}": int(sum(ep_to_bin[ep] == j for ep in eps)) for j in range(3)}
        split[f"client_{idx}"] = eps
        split[f"client_{idx}_feature_min"] = float(np.min(values))
        split[f"client_{idx}_feature_max"] = float(np.max(values))
        split[f"client_{idx}_feature_mean"] = float(np.mean(values))
        split[f"client_{idx}_own_bin_count"] = origin_counts[f"bin_{idx}"]
        split[f"client_{idx}_own_bin_ratio"] = float(origin_counts[f"bin_{idx}"] / size)
        split[f"client_{idx}_origin_counts"] = origin_counts
        split_rows.append({
            "client": idx,
            "size": size,
            "private_own_count": private_counts[idx],
            "own_bin_count": origin_counts[f"bin_{idx}"],
            "own_bin_ratio": origin_counts[f"bin_{idx}"] / size,
            "x_min": float(np.min(values)),
            "x_max": float(np.max(values)),
            "x_mean": float(np.mean(values)),
            **origin_counts,
        })
    return split, split_rows
