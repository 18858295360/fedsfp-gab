import math

import torch


def reset_adam_first_moment(optimizer):
    """Clear Adam's first moment while retaining variance and step state."""
    for state in optimizer.state.values():
        exp_avg = state.get("exp_avg")
        if exp_avg is not None:
            exp_avg.zero_()


def scale_adam_first_moment(optimizer, retention):
    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must be in [0, 1]")
    for state in optimizer.state.values():
        exp_avg = state.get("exp_avg")
        if exp_avg is not None:
            exp_avg.mul_(retention)


@torch.no_grad()
def project_adam_first_moment_against_gradients(
    optimizer, parameters, epsilon=1e-12, projection_scale=1.0
):
    """Remove a configurable share of the first-moment component opposed to gradients."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < projection_scale <= 1.0:
        raise ValueError("projection_scale must be in (0, 1]")

    pairs = []
    dot = 0.0
    moment_norm_sq = 0.0
    gradient_norm_sq = 0.0
    for parameter in parameters:
        gradient = parameter.grad
        state = optimizer.state.get(parameter)
        exp_avg = None if state is None else state.get("exp_avg")
        if gradient is None or exp_avg is None:
            continue
        gradient_value = gradient.detach().to(device=exp_avg.device, dtype=exp_avg.dtype)
        pairs.append((exp_avg, gradient_value))
        dot += float(torch.sum(exp_avg.float() * gradient_value.float()).item())
        moment_norm_sq += float(torch.sum(exp_avg.float().square()).item())
        gradient_norm_sq += float(torch.sum(gradient_value.float().square()).item())

    coefficient = dot / (gradient_norm_sq + epsilon) if pairs else 0.0
    projected = bool(pairs and gradient_norm_sq > 0.0 and dot < 0.0)
    if projected:
        for exp_avg, gradient in pairs:
            exp_avg.sub_(gradient, alpha=coefficient * projection_scale)

    dot_after = 0.0
    for exp_avg, gradient in pairs:
        dot_after += float(torch.sum(exp_avg.float() * gradient.float()).item())

    denominator = math.sqrt(moment_norm_sq * gradient_norm_sq)
    cosine = dot / denominator if denominator > 0.0 else 0.0
    removed_norm = (
        abs(coefficient) * projection_scale * math.sqrt(gradient_norm_sq)
        if projected
        else 0.0
    )
    return {
        "projected": projected,
        "moment_tensor_count": len(pairs),
        "dot_before": dot,
        "dot_after": dot_after,
        "cosine_before": cosine,
        "coefficient": coefficient,
        "projection_scale": projection_scale,
        "moment_norm": math.sqrt(moment_norm_sq),
        "gradient_norm": math.sqrt(gradient_norm_sq),
        "removed_norm": removed_norm,
    }

def moment_retention(alignment):
    if alignment <= 0.0:
        return 0.0
    if alignment < 0.5:
        return 0.5
    return 1.0

def leave_one_out_cosines(global_parameters, local_parameters, weights):
    if len(local_parameters) != len(weights) or len(local_parameters) < 2:
        raise ValueError("leave-one-out alignment needs at least two weighted clients")
    total_weight = float(sum(weights))
    dots = [0.0] * len(local_parameters)
    own_norms = [0.0] * len(local_parameters)
    loo_norms = [0.0] * len(local_parameters)
    for parameter_name, global_value in global_parameters.items():
        updates = [local[parameter_name].detach() - global_value.detach() for local in local_parameters]
        weighted_total = updates[0] * float(weights[0])
        for update, weight in zip(updates[1:], weights[1:]):
            weighted_total = weighted_total + update * float(weight)
        for index, (update, weight) in enumerate(zip(updates, weights)):
            denominator = total_weight - float(weight)
            loo_update = (weighted_total - update * float(weight)) / denominator
            dots[index] += float((update * loo_update).sum().item())
            own_norms[index] += float((update * update).sum().item())
            loo_norms[index] += float((loo_update * loo_update).sum().item())
    return [0.0 if own <= 0.0 or loo <= 0.0 else dot / (own * loo) ** 0.5 for dot, own, loo in zip(dots, own_norms, loo_norms)]
