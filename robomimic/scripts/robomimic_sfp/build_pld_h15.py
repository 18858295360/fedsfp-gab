"""Rebuild the frozen RoboMimic PLD-H15 reference memberships."""

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


HOR, BLO, BHI, TMIN = 15, 10, 20, 20
HALF = 0.5 * math.pi
OBJECT_Z_COLUMNS = {"square": 9, "can": 9, "lift": 2}


def Rvec(a):
    return float(np.hypot(np.cos(a).sum(), np.sin(a).sum()))


def phi_table(h5path, zcol):
    out, skipped = {}, []
    with h5py.File(h5path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for k in keys:
            g = f["data"][k]["obs"]
            z = np.asarray(g["object"][:, zcol], dtype=float)
            e = np.asarray(g["robot0_eef_pos"][:, :2], dtype=float)
            did = int(k.split("_")[1])
            if len(z) <= BHI:
                skipped.append(did); continue
            rest = float(np.median(z[BLO:BHI])); up = z > rest + 0.02
            t = None
            for i in range(TMIN, len(up) - 2):
                if up[i] and up[i + 1] and up[i + 2]:
                    t = i; break
            if t is None or t - HOR < 0:
                skipped.append(did); continue
            d = e[t] - e[t - HOR]; m = float(np.hypot(d[0], d[1]))
            if m < 1e-12:
                skipped.append(did); continue
            out[did] = math.atan2(d[1], d[0]) % (2 * math.pi)
    return out, skipped


def best_diameter(ids, phi):
    a = np.asarray([phi[i] for i in ids]); ids = np.asarray(ids)
    o = np.sort(a)
    pts = list(o) + [float(0.5 * (o[i] + o[i + 1])) for i in range(len(o) - 1)]
    pts.append(float(0.5 * (o[-1] + o[0] + 2.0 * np.pi)))
    best, bs = None, -1.0
    for p in pts:
        for th in (p + HALF, p - HALF):
            s = np.cos(a - th) >= 0
            if s.sum() == 0 or (~s).sum() == 0:
                continue
            sc = Rvec(a[s]) + Rvec(a[~s])
            if sc > bs:
                bs, best = sc, s.copy()
    if best is None:
        raise ValueError("No nonempty diameter partition exists")
    return sorted(ids[best].tolist()), sorted(ids[~best].tolist())


def median_split(ids, phi):
    items = sorted(((phi[i], i) for i in ids)); n = len(items)
    if n < 2:
        return sorted(ids), []
    gaps = []
    for j in range(n):
        a0 = items[j][0]
        a1 = items[(j + 1) % n][0] + (2 * math.pi if j == n - 1 else 0.0)
        gaps.append((a1 - a0, j))
    gmax = max(g for g, _ in gaps)
    j = min([j for g, j in gaps if abs(g - gmax) < 1e-12], key=lambda jj: items[(jj + 1) % n])
    start = (j + 1) % n
    ordered = items[start:] + items[:start]
    origin = ordered[0][0]
    unwrapped = sorted(((a - origin) % (2 * math.pi), i) for a, i in ordered)
    return sorted(i for _, i in unwrapped[: n // 2]), sorted(i for _, i in unwrapped[n // 2:])


def refine(parents, phi):
    out = []
    for p in parents:
        l, r = median_split(p, phi); out += [l, r]
    return out


def ladder(phi):
    lv = {2: list(best_diameter(sorted(phi), phi))}
    lv[4] = refine(lv[2], phi); lv[8] = refine(lv[4], phi)
    return lv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=OBJECT_Z_COLUMNS, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--clients", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    phi, skipped = phi_table(args.dataset_path, OBJECT_Z_COLUMNS[args.task])
    if skipped or sorted(phi) != list(range(300)):
        parser.error(
            f"Expected all 300 MH demonstrations to have a valid lift event; "
            f"found {len(phi)} valid demonstrations and skipped IDs {skipped}"
        )

    groups = ladder(phi)[args.clients]
    split = {
        "split": f"{args.task}_pld_h15_k{args.clients}",
        "clients": {f"client_{index}": ids for index, ids in enumerate(groups)},
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(split, output, separators=(",", ":"))
        output.write("\n")


if __name__ == "__main__":
    main()
