"""Helpers from the study module phase1_iid_action_fedavg.py used by the Push-T trainer."""

import torch


def average_state_dicts(weighted_state_dicts):
    total_weight = float(sum(weight for _, weight in weighted_state_dicts))
    result = {}
    first_state = weighted_state_dicts[0][0]
    for key, first_value in first_state.items():
        if torch.is_floating_point(first_value):
            acc = torch.zeros_like(first_value, dtype=first_value.dtype)
            for state, weight in weighted_state_dicts:
                acc.add_(state[key].to(acc.device), alpha=float(weight) / total_weight)
            result[key] = acc
        else:
            result[key] = first_value.clone()
    return result
