"""Finite-horizon, constant-speed VO search with direction constraints.

Candidates can be checked with the nominal closed-loop vessel response.
Neither prediction model is a robust safety certificate for stochastic motion.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from src.config.paper_parameters import (
    OWN_SHIP_SPEED_MPS, TARGET_SHIP_SPEED_MPS, VO_HEADING_SEARCH_DEG,
    VO_HEADING_STEP_DEG, VO_SAFETY_RADIUS_M, VO_TIME_HORIZON_S,
)
from src.navigation.encounters import wrap_angle

METERS_PER_NAUTICAL_MILE = 1852.0


@dataclass(frozen=True)
class VOPlan:
    heading: float
    correction: float
    predicted_clearance_m: float
    prediction_safe: bool
    direction_overridden: bool
    status: str
    prediction_model: str = "constant_heading"
    evaluated_candidates: int = 0


def apply_direction_constraint(correction: float, kdir: int) -> float:
    """Legacy reactive-planner mapping; NEVER apply to a VO search result."""
    magnitude = max(abs(float(correction)), float(np.deg2rad(VO_HEADING_STEP_DEG)))
    return -magnitude if kdir > 0 else magnitude if kdir < 0 else 0.0


def plan_velocity_obstacle(x_ob, y_ob, x, y, psi, v_ob=None, psi_ob=None,
                           own_speed_mps=OWN_SHIP_SPEED_MPS,
                           safety_radius_m=VO_SAFETY_RADIUS_M,
                           horizon_s=VO_TIME_HORIZON_S, *,
                           desired_heading=None, kdir=None, dynamics_state=None,
                           controller_integral=0., response_dt_s=.5, response_transient_s=60.) -> VOPlan:
    """Search absolute headings; positions are nmi, speeds m/s, headings rad.

    kdir +1/-1 permits starboard/port or holding the present heading. kdir 0
    requests holding heading; None follows desired_heading without a side limit.
    If no safe constrained candidate exists, search all sides and explicitly
    record the override. If none is safe, return maximum-clearance best effort
    with prediction_safe=False; never label that fallback safe.
    """
    if kdir not in (None, -1, 0, 1):
        raise ValueError('kdir must be None, -1, 0, or 1')
    if not np.all(np.isfinite([horizon_s,safety_radius_m,own_speed_mps])) or horizon_s <= 0 or safety_radius_m <= 0 or own_speed_mps < 0:
        raise ValueError('Invalid VO prediction parameters')
    xo, yo = np.asarray(x_ob, float), np.asarray(y_ob, float)
    n = xo.size
    speeds = np.full(n, TARGET_SHIP_SPEED_MPS) if v_ob is None else np.asarray(v_ob, float)
    headings = np.zeros(n) if psi_ob is None else np.asarray(psi_ob, float)
    if not (xo.shape == yo.shape == speeds.shape == headings.shape) or xo.ndim != 1:
        raise ValueError('Target arrays must be one-dimensional and have equal length')
    if not all(np.all(np.isfinite(a)) for a in (xo, yo, speeds, headings, [x, y, psi, own_speed_mps])):
        raise ValueError('VO inputs must be finite')
    goal = float(psi if desired_heading is None else desired_heading)
    if not np.isfinite(goal):
        raise ValueError('desired_heading must be finite')
    limit = np.deg2rad(VO_HEADING_SEARCH_DEG)
    offsets = np.deg2rad(np.arange(-VO_HEADING_SEARCH_DEG,
                                   VO_HEADING_SEARCH_DEG + VO_HEADING_STEP_DEG,
                                   VO_HEADING_STEP_DEG))
    offsets = np.unique(np.append(offsets, np.clip(wrap_angle(goal - psi), -limit, limit)))
    candidates = psi + offsets
    velocity = own_speed_mps * np.column_stack((np.cos(candidates), np.sin(candidates)))
    clearance = np.full(len(candidates), np.inf)
    for px, py, speed, heading in zip(xo, yo, speeds, headings):
        position = METERS_PER_NAUTICAL_MILE * np.array([px - x, py - y])
        rel_velocity = speed * np.array([np.cos(heading), np.sin(heading)]) - velocity
        speed_sq = np.sum(rel_velocity ** 2, axis=1)
        tcpa = np.divide(-np.sum(position * rel_velocity, axis=1), speed_sq,
                         out=np.zeros(len(candidates)), where=speed_sq > 1e-12)
        # Evaluate the entire closed time interval, including an initial
        # intrusion, a diverging close vessel, and CPA beyond the horizon.
        tcpa = np.clip(tcpa, 0.0, horizon_s)
        distances = np.linalg.norm(position + rel_velocity * tcpa[:, None], axis=1)
        clearance = np.minimum(clearance, distances)
    safe = clearance > safety_radius_m
    allowed = np.ones(len(offsets), dtype=bool)
    if kdir == 1:
        allowed = offsets <= 1e-12
    elif kdir == -1:
        allowed = offsets >= -1e-12
    elif kdir == 0:
        allowed = np.abs(offsets) <= 1e-12
    model = 'constant_heading'
    evaluated = len(candidates)
    if dynamics_state is not None:
        from src.navigation.response_prediction import response_clearance
        model = 'nomoto_closed_loop'
        goal_errors = np.abs(wrap_angle(candidates-goal))
        order = np.lexsort((np.abs(offsets), goal_errors))
        clearance = np.full(len(candidates), np.nan)
        selected = None
        evaluated = 0
        status = 'safe'
        target_positions = METERS_PER_NAUTICAL_MILE*np.column_stack((xo-x, yo-y))
        # Predictor uses world positions, while this relative translation keeps
        # small coordinates without changing headings or target velocities.
        state = np.asarray(dynamics_state, float).copy()
        if state.shape != (6,) or abs(float(wrap_angle(state[2]-psi))) > 1e-9:
            raise ValueError('Dynamics state and VO heading must agree')
        state[:2] = 0.
        target_velocity = speeds[:, None]*np.column_stack((np.cos(headings),np.sin(headings)))
        for phase in (True, False):
            indices = order[allowed[order] == phase]
            for start in range(0, len(indices), 8):
                batch = indices[start:start+8]
                clearance[batch] = response_clearance(candidates[batch], state, controller_integral,
                    target_positions, target_velocity, own_speed_mps, horizon_s,
                    response_dt_s, response_transient_s)
                evaluated += len(batch)
                feasible = batch[clearance[batch] > safety_radius_m]
                if feasible.size:
                    selected = int(feasible[0])
                    status = 'safe' if phase else 'direction_override'
                    break
            if selected is not None:
                break
        if selected is None:
            best = np.nanmax(clearance)
            pool = np.flatnonzero(np.isclose(clearance,best,rtol=0,atol=1e-6))
            if np.any(allowed[pool]):pool=pool[allowed[pool]]
            errors = np.abs(wrap_angle(candidates[pool]-goal))
            selected = int(pool[np.lexsort((np.abs(offsets[pool]),errors))[0]])
            status = 'no_safe_heading'
        safe = clearance > safety_radius_m
    else:
        pool = np.flatnonzero(safe & allowed)
        status = 'safe'
        if not pool.size:
            pool = np.flatnonzero(safe)
            status = 'direction_override'
        if not pool.size:
            best = np.max(clearance)
            pool = np.flatnonzero(np.isclose(clearance,best,rtol=0,atol=1e-6))
            if np.any(allowed[pool]):pool=pool[allowed[pool]]
            status = 'no_safe_heading'
        errors = np.abs(wrap_angle(candidates[pool]-goal))
        selected = int(pool[np.lexsort((np.abs(offsets[pool]),errors))[0]])
    return VOPlan(float(wrap_angle(candidates[selected])), float(offsets[selected]),
                  float(clearance[selected]), bool(safe[selected]),
                  bool(not allowed[selected]), status, model, evaluated)


def velocity_obstacle_avoidance(x_ob, y_ob, x, y, psi, v_ob=None, psi_ob=None,
                               own_speed_mps=OWN_SHIP_SPEED_MPS,
                               safety_radius_m=VO_SAFETY_RADIUS_M,
                               horizon_s=VO_TIME_HORIZON_S, **kwargs):
    """Compatibility API: correction is relative to CURRENT heading."""
    plan = plan_velocity_obstacle(x_ob, y_ob, x, y, psi, v_ob, psi_ob,
                                 own_speed_mps, safety_radius_m, horizon_s, **kwargs)
    xo, yo = np.asarray(x_ob, float), np.asarray(y_ob, float)
    bearing = wrap_angle(psi - np.arctan2(yo - y, xo - x))
    distance = np.hypot(xo - x, yo - y)
    return (plan.correction, -np.cos(bearing),
            (distance <= safety_radius_m / METERS_PER_NAUTICAL_MILE).astype(float),
            distance, bearing)
