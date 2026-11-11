#!/usr/bin/env python
"""FedAvg with the optimizer reset every round but a learning rate schedule that is global.

The trainer only offers the two ends: `raw` reinitialises AdamW each round *and* restarts
the cosine within that round, while `persistent` keeps per-client AdamW state across
rounds *and* runs one cosine over the whole run. The baseline wanted here is the middle:
textbook FedAvg, where nothing but the averaged weights crosses a round boundary, paired
with a single global schedule so the learning rate decays across the run instead of
sawtoothing 300 times.

Implemented by intercepting the scheduler the `raw` branch builds. The optimizer is still
recreated per round -- untouched -- but its scheduler is rebuilt against the full horizon
and fast-forwarded to the round's starting step, which reproduces exactly the LR a single
global cosine would be at.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.robomimic_sfp import train_table3_federated as F

_orig_get_scheduler = F.get_scheduler
_state = {"calls": 0, "clients": None, "rounds": None, "warmup": None}


def global_get_scheduler(name, optimizer=None, num_warmup_steps=0, num_training_steps=0, **kw):
    n_clients = _state["clients"]
    rounds = _state["rounds"]
    if not n_clients or not rounds:
        return _orig_get_scheduler(name, optimizer=optimizer,
                                   num_warmup_steps=num_warmup_steps,
                                   num_training_steps=num_training_steps, **kw)

    steps_per_round = max(int(num_training_steps), 1)
    total = steps_per_round * int(rounds)
    rnd = _state["calls"] // n_clients          # zero-based round index
    _state["calls"] += 1

    sched = _orig_get_scheduler(name, optimizer=optimizer,
                                num_warmup_steps=_state["warmup"],
                                num_training_steps=total, **kw)
    # advance to where a single continuous cosine would be at the start of this round
    for _ in range(rnd * steps_per_round):
        sched.step()
    if rnd == 0:
        print("[lr] global cosine: warmup=%d total_steps=%d steps_per_round=%d"
              % (_state["warmup"], total, steps_per_round), flush=True)
    return sched


def main():
    args = F.parse_args()
    import json
    cfg = json.loads(Path(args.split_json).read_text())
    clients = cfg.get("clients", cfg)
    _state["clients"] = len([k for k in clients if str(k).startswith("client_")])
    _state["rounds"] = int(args.rounds)
    _state["warmup"] = int(args.warmup_steps)
    F.get_scheduler = global_get_scheduler
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    F.main()


if __name__ == "__main__":
    main()
