"""Metric helpers shared by the reproducible paper experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def count_turn_events(kdir: Sequence[float], threshold: float = 0.1) -> int:
    """Count maneuver events, not every active control sample."""
    values = np.asarray(kdir, dtype=float)
    if values.size == 0:
        return 0
    active = np.abs(values) > threshold
    signs = np.sign(values)
    previous_active = False
    previous_sign = 0.0
    events = 0
    for is_active, sign in zip(active, signs):
        if is_active and (not previous_active or sign != previous_sign):
            events += 1
        previous_active = bool(is_active)
        if is_active:
            previous_sign = sign
    return int(events)
