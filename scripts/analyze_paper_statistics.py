from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper's scene-level paired statistical analysis.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-column", default="provider")
    parser.add_argument("--case-column", default="case_number")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--comparators", nargs="+", default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["R_max", "R_avg", "min_dcpa_nm", "A_turn_pct", "delta_D_pct", "N_call"],
    )
    parser.add_argument("--higher-is-better", nargs="+", default=["min_dcpa_nm", "A_turn_pct"])
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def scene_means(
    rows: List[Dict[str, str]], method_column: str, case_column: str, metric: str
) -> Dict[str, Dict[int, float]]:
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in rows:
        try:
            value = float(row[metric])
            case_number = int(float(row[case_column]))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            grouped[(row[method_column], case_number)].append(value)

    result: Dict[str, Dict[int, float]] = defaultdict(dict)
    for (method, case_number), values in grouped.items():
        result[method][case_number] = float(np.mean(values))
    return result


def holm_adjust(p_values: List[float]) -> List[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_ci(values: np.ndarray, resamples: int, rng: np.random.Generator) -> Tuple[float, float]:
    samples = rng.choice(values, size=(resamples, values.size), replace=True)
    means = np.mean(samples, axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def paired_test(differences: np.ndarray) -> Tuple[str, float, float]:
    try:
        from scipy import stats
    except ImportError as exc:
        raise SystemExit("Statistical analysis requires scipy; install the project requirements first.") from exc
    if differences.size >= 3:
        normality_p = float(stats.shapiro(differences).pvalue)
    else:
        normality_p = float("nan")
    if differences.size >= 3 and normality_p >= 0.05:
        return "paired_t", normality_p, float(stats.ttest_1samp(differences, 0.0).pvalue)
    if np.allclose(differences, 0.0):
        return "wilcoxon", normality_p, 1.0
    return "wilcoxon", normality_p, float(stats.wilcoxon(differences).pvalue)


def main() -> int:
    args = parse_args()
    rows = [row for path in args.input for row in read_rows(path)]
    rng = np.random.default_rng(args.seed)
    results: List[Dict[str, object]] = []

    for metric in args.metrics:
        by_method = scene_means(rows, args.method_column, args.case_column, metric)
        if args.reference not in by_method:
            raise SystemExit(f"Reference method {args.reference!r} is absent for metric {metric!r}.")
        comparators = args.comparators or sorted(method for method in by_method if method != args.reference)
        for comparator in comparators:
            if comparator not in by_method:
                continue
            common_cases = sorted(set(by_method[args.reference]) & set(by_method[comparator]))
            if not common_cases:
                continue
            reference = np.asarray([by_method[args.reference][case] for case in common_cases])
            compared = np.asarray([by_method[comparator][case] for case in common_cases])
            paired_difference = reference - compared
            improvement = paired_difference if metric in args.higher_is_better else -paired_difference
            test_name, normality_p, p_value = paired_test(paired_difference)
            ci_low, ci_high = bootstrap_ci(paired_difference, args.bootstrap_resamples, rng)
            tolerance = 1e-12
            results.append(
                {
                    "metric": metric,
                    "reference": args.reference,
                    "comparator": comparator,
                    "n_scenes": len(common_cases),
                    "reference_mean": float(np.mean(reference)),
                    "comparator_mean": float(np.mean(compared)),
                    "mean_difference": float(np.mean(paired_difference)),
                    "difference_definition": "reference - comparator",
                    "bootstrap_95_ci_low": ci_low,
                    "bootstrap_95_ci_high": ci_high,
                    "test": test_name,
                    "normality_p": normality_p,
                    "p_value": p_value,
                    "wins": int(np.sum(improvement > tolerance)),
                    "ties": int(np.sum(np.abs(improvement) <= tolerance)),
                    "losses": int(np.sum(improvement < -tolerance)),
                }
            )

    for metric in args.metrics:
        metric_rows = [row for row in results if row["metric"] == metric]
        adjusted = holm_adjust([float(row["p_value"]) for row in metric_rows]) if metric_rows else []
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["holm_adjusted_p"] = adjusted_p

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "paper_statistical_tests.csv"
    if results:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    json_path = args.output_dir / "paper_statistical_tests.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} comparisons to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
