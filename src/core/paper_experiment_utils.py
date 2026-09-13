from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def scene_level_values(
    rows: Iterable[Dict[str, Any]],
    metric: str,
    case_field: str = "case_number",
) -> np.ndarray:
    by_case: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            by_case[int(row[case_field])].append(numeric)
    return np.asarray(
        [float(np.mean(by_case[case_number])) for case_number in sorted(by_case)],
        dtype=float,
    )


def add_scene_summary(
    destination: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
) -> None:
    for metric in metrics:
        values = scene_level_values(rows, metric)
        destination[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
        destination[f"{metric}_std"] = float(np.std(values)) if values.size else float("nan")
