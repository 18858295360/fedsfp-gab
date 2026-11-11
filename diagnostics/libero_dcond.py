"""Conditional audit for the LIBERO azimuth partition.

Design points forced by what the partition is:

* Matching happens WITHIN A TASK; libero_10 is ten different tasks and a
  neighbour from another task shares nothing meaningful.
* The matching variable is the SITUATION rather than the robot's own pose.  The
  split groups demonstrations by which side the arm approaches from, so matching
  on end-effector position would discard exactly the pairs in question.
  `robot_states` does not align with any offset of `states` (best mean abs error
  0.67), so the situation columns are selected empirically instead: an object
  column barely moves during the opening of an episode while differing a lot
  between episodes, whereas a robot column starts moving immediately.
* Split ids are "<task>|<demo>" strings, not global integers.

Action is the policy's chunk a_{t:t+H}.  Reported per task, at 1000
permutations, with the discard rate as the support-overlap diagnostic.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from action_chunks import chunk_actions
from dbeh_fast import phase_scan_fast

H_CHUNK, M = 16, 20


def run(dataset_dir, split):
    with open(split, encoding="utf-8") as handle:
        j = json.load(handle)
    cl = j.get("clients", j)
    ks = sorted([k for k in cl if k.startswith("client_") and isinstance(cl[k], list)],
                key=lambda k: int(k.split("_")[1]))
    assign = {}
    for gi, k in enumerate(ks):
        for e in cl[k]:
            assign[str(e)] = gi
    print("  split %s: sizes %s" % (split.name, [len(cl[k]) for k in ks]))
    print("  id format sample: %s" % list(assign)[:3])

    files = sorted(dataset_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files in {dataset_dir}")
    per_task = []
    for ti, p in enumerate(files):
        fh = h5py.File(p, "r")
        d = fh["data"]
        names = sorted(d.keys(), key=lambda k: int(k.split("_")[1]))
        raw = {}
        for nm in names:
            key = "%d|%s" % (ti, nm)
            if key not in assign:
                continue
            g = d[nm]
            raw[nm] = (np.asarray(g["states"][:], dtype=float),
                       np.asarray(g["actions"][:], dtype=float), assign[key])
        fh.close()
        if len(raw) < 10 or len({v[2] for v in raw.values()}) < 2:
            print("  %-46s skipped (%d demos, %d clients)"
                  % (os.path.basename(p)[:46], len(raw), len({v[2] for v in raw.values()})))
            continue

        # situation columns: quiet inside an episode's opening, different between episodes
        S = list(raw.values())
        early = np.array([v[0][:max(3, len(v[0]) // 5)].std(0) for v in S]).mean(0)
        across = np.array([v[0][0] for v in S]).std(0)
        sit = (early < np.median(early)) & (across > np.median(across))
        if sit.sum() < 3:
            sit = across > np.percentile(across, 60)

        O, A, demo, bins, groups = [], [], [], [], {}
        for i, (nm, (st, a, gi)) in enumerate(sorted(raw.items())):
            T = min(len(st), len(a))
            if T < M:
                continue
            idx = np.linspace(0, T - 1, M).astype(int)
            O.append(st[idx][:, sit])
            A.append(chunk_actions(a, idx, H_CHUNK))
            demo.append(np.full(M, i))
            bins.append(np.minimum((idx / max(T - 1, 1) * 5).astype(int), 4))
            groups.setdefault(gi, []).append(i)
        O = np.concatenate(O); A = np.concatenate(A)
        demo = np.concatenate(demo); bins = np.concatenate(bins)
        keep = O.std(0) > 1e-8
        O = (O[:, keep] - O[:, keep].mean(0)) / (O[:, keep].std(0) + 1e-8)
        A = (A - A.mean(0)) / (A.std(0) + 1e-8)
        gl = [groups[k] for k in sorted(groups)]
        r = phase_scan_fast(O, A, demo, bins, gl, k=6, perms=1000)
        tot = len(O)
        print("  %-46s sizes=%-9s dims=%-3d peak D=%6.3f max-z=%6.2f p=%.4f usable=%d/%d"
              % (os.path.basename(p)[:46], str([len(x) for x in gl]), int(keep.sum()),
                 max(r["D"]), r["max_z"], r["p"], sum(r["used"]), tot))
        per_task.append((os.path.basename(p), max(r["D"]), r["max_z"], r["p"],
                         sum(r["used"]) / tot))

    print()
    if per_task:
        sig = [t for t in per_task if t[3] < 0.05]
        print("  tasks with p<0.05: %d of %d" % (len(sig), len(per_task)))
        print("  median peak D_cond %.3f   median usable fraction %.2f"
              % (float(np.median([t[1] for t in per_task])),
                 float(np.median([t[4] for t in per_task]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset_dir, args.split)


if __name__ == "__main__":
    main()
