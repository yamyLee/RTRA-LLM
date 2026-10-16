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


METRIC_SCHEMA = "rtra-2026-10-16-v4-paper-distance"


def final_sample_distance_m(result):
    """Paper experiment distance: radius from origin at the last stored sample.

    The original experiments use hypot(x[-1], y[-1]) on the [0,T) trace.
    This is endpoint displacement, not sailed length or distance to the waypoint;
    do not append the separately stored state at T when calculating this metric.
    """
    return float(np.hypot(result['x'][-1], result['y'][-1]))


def distance_deviation_pct(distance_m, reference_distance_m):
    """Original paper experiment Delta D, including its rounding/zero guard."""
    if abs(reference_distance_m) < 1e-6:
        return 0.0
    return round((distance_m - reference_distance_m) / reference_distance_m * 100, 3)


def count_reverse_switches(kdir, threshold=0.1):
    """Ignore holds; count changes between the last nonzero turn directions."""
    values = np.asarray(kdir, float)
    signs = np.sign(values[np.abs(values) > threshold])
    return int(np.count_nonzero(np.diff(signs)))


def rectangular_hull_overlap(positions, headings, targets, target_headings):
    """Sampled overlap of oriented rectangular hulls, using separating axes."""
    from src.config.paper_parameters import (
        OWN_SHIP_LENGTH_M, OWN_SHIP_BEAM_M, TARGET_SHIP_LENGTH_M, TARGET_SHIP_BEAM_M,
    )
    positions, targets = np.asarray(positions), np.asarray(targets)
    own_x = np.column_stack((np.cos(headings), np.sin(headings)))[:, None, :]
    own_y = np.stack((-own_x[..., 1], own_x[..., 0]), axis=-1)
    other_x = np.column_stack((np.cos(target_headings), np.sin(target_headings)))[None, :, :]
    other_y = np.stack((-other_x[..., 1], other_x[..., 0]), axis=-1)
    separation = targets - positions[:, None, :]
    overlap = np.ones(separation.shape[:2], dtype=bool)
    for axis in (own_x, own_y, other_x, other_y):
        own_extent = (OWN_SHIP_LENGTH_M * np.abs(np.sum(own_x * axis, axis=-1)) +
                      OWN_SHIP_BEAM_M * np.abs(np.sum(own_y * axis, axis=-1))) / 2
        other_extent = (TARGET_SHIP_LENGTH_M * np.abs(np.sum(other_x * axis, axis=-1)) +
                        TARGET_SHIP_BEAM_M * np.abs(np.sum(other_y * axis, axis=-1))) / 2
        overlap &= np.abs(np.sum(separation * axis, axis=-1)) <= own_extent + other_extent
    return overlap


def trajectory_metrics(result, baseline=None):
    """Shared metrics for VO, LLM and PPO. No model-specific metric aliases.

    Risk/CPA/turn signals use synchronous left endpoints [0,T). The final state
    is appended for path length, progress, and sampled geometric safety. Domain
    duration is a left-rectangle estimate, with episodes counted per target.
    A_turn is ONLY similarity to a paired VO trajectory, never correctness.
    Delta D retains the original experiment's last-sample radial-distance
    comparison with paired VO; actual sailed length is path_length_m only.
    """
    from src.config.paper_parameters import VO_SAFETY_RADIUS_M
    positions = np.column_stack((result['x'], result['y']))
    endpoints = np.vstack((positions, result['final_position_m']))
    headings = np.append(result['psi'], result['final_heading_rad'])
    targets = np.stack((result['obstacles_x_history'], result['obstacles_y_history']), axis=-1)
    targets = np.concatenate((targets, np.asarray(result['final_target_positions_m'])[None]), axis=0)
    distance = np.linalg.norm(targets - endpoints[:, None, :], axis=-1)
    inside = distance < VO_SAFETY_RADIUS_M
    starts = inside & ~np.vstack((np.zeros((1, inside.shape[1]), dtype=bool), inside[:-1]))
    collision = rectangular_hull_overlap(endpoints, headings, targets, result['obstacles_heading_rad'])
    length = float(np.linalg.norm(np.diff(endpoints, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(endpoints[-1] - endpoints[0]))
    route_start = np.array([result['waypoints_x'][0], result['waypoints_y'][0]]) * 1852
    route_end = np.array([result['waypoints_x'][-1], result['waypoints_y'][-1]]) * 1852
    route_vector = route_end - route_start
    route_length = float(np.linalg.norm(route_vector))
    direction = route_vector / max(route_length, 1e-12)
    rel = endpoints - route_start
    cross_track = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
    risk = np.asarray(result['risk'], float)
    dcpa, tcpa = np.asarray(result['dcpa'], float), np.asarray(result['tcpa'], float)
    approaching_dcpa = np.abs(dcpa[np.isfinite(dcpa) & (tcpa >= 0) & np.isfinite(tcpa)])
    turns = np.asarray(result['kdir'], float)
    output = {
        'metric_schema': METRIC_SCHEMA,
        'R_max': float(np.max(risk)) if risk.size else 0.0,
        'R_avg': float(np.mean(risk)) if risk.size else 0.0,
        'min_dcpa_nm': float(np.min(approaching_dcpa) / 1852) if approaching_dcpa.size else float('nan'),
        'min_actual_separation_nm': float(np.min(distance) / 1852) if distance.size else float('inf'),
        'safety_domain_intrusion_count': int(starts.sum()),
        'safety_domain_intrusion_duration_s': float(np.any(inside[:-1], axis=1).sum() * result['dt']),
        'collision': bool(np.any(collision)),
        'path_length_m': length,
        'route_excess_pct': 100 * (length / chord - 1) if chord > 1e-9 else float('nan'),
        'max_cross_track_m': float(cross_track.max()),
        'rms_cross_track_m': float(np.sqrt(np.mean(cross_track ** 2))),
        'route_progress_m': float((endpoints[-1] - endpoints[0]) @ direction),
        'waypoint_distance_m': float(np.linalg.norm(endpoints[-1] - route_end)),
        'final_dist_nm': final_sample_distance_m(result) / 1852,
        'total_turns': count_turn_events(turns),
        'reverse_switches': count_reverse_switches(turns),
        'A_turn_pct': float('nan'),
        'delta_D_pct': float('nan'),
        # Consistency with this implementation's instruction sets only.
        'execution_rule_violation_pct': (100.0 * np.count_nonzero(result.get('rule_violation', [])) /
            np.count_nonzero(result.get('rule_evaluable', []))
            if np.count_nonzero(result.get('rule_evaluable', [])) else float('nan')),
    }
    if baseline is not None:
        if (len(result['time']) != len(baseline['time']) or
                not np.allclose(result['time'], baseline['time'], atol=1e-9, rtol=0)):
            raise ValueError('Paired trajectory time grids must match')
        reference_turns = np.asarray(baseline['kdir'])
        output['A_turn_pct'] = float(100 * np.mean(np.sign(turns) == np.sign(reference_turns)))
        reference_distance = final_sample_distance_m(baseline)
        output['baseline_dist_nm'] = reference_distance / 1852
        output['delta_D_pct'] = distance_deviation_pct(
            final_sample_distance_m(result), reference_distance)
    return output


def hull_collision_at_state(position, heading, targets, target_headings):
    """Exact same rectangular footprint test as offline evaluation, with a cheap
    circumscribed-circle rejection for distant ships. No planning-radius proxy.
    """
    from src.config.paper_parameters import (OWN_SHIP_LENGTH_M, OWN_SHIP_BEAM_M,
        TARGET_SHIP_LENGTH_M, TARGET_SHIP_BEAM_M)
    targets = np.asarray(targets, float).reshape(-1, 2)
    bound = (np.hypot(OWN_SHIP_LENGTH_M, OWN_SHIP_BEAM_M) +
             np.hypot(TARGET_SHIP_LENGTH_M, TARGET_SHIP_BEAM_M)) / 2
    close = np.linalg.norm(targets - position, axis=1) <= bound
    if not np.any(close):
        return False
    return bool(rectangular_hull_overlap(np.asarray(position)[None], [heading],
        targets[None, close, :], np.asarray(target_headings)[close]).any())
