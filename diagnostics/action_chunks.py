"""Action-chunk extraction for the LIBERO diagnostic."""

import numpy as np


def chunk_actions(acts, idx, H):
    """a_{t:t+H} flattened, clamped at the episode end by repeating the last."""
    out = []
    for t in idx:
        c = acts[t:t + H]
        if len(c) < H:
            c = np.concatenate([c, np.repeat(c[-1:], H - len(c), axis=0)], 0)
        out.append(c.reshape(-1))
    return np.stack(out)
