"""Two-dimensional velocity-obstacle baseline used by the paper experiments.

The baseline searches constant-speed candidate headings around the current
heading. A candidate is rejected when the relative trajectory enters the
specified safety radius within the finite prediction horizon. The returned
correction keeps the same interface as the legacy local-avoidance module so it
can be used by the existing simulation loop.
"""

from __future__ import annotations

import numpy as np

from src.config.paper_parameters import (
    OWN_SHIP_SPEED_MPS,
    TARGET_SHIP_SPEED_MPS,
    VO_HEADING_SEARCH_DEG,
    VO_HEADING_STEP_DEG,
    VO_SAFETY_RADIUS_M,
    VO_TIME_HORIZON_S,
)

METERS_PER_NAUTICAL_MILE = 1852.0


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def velocity_obstacle_avoidance(
    x_ob,
    y_ob,
    x,
    y,
    psi,
    v_ob=None,
    psi_ob=None,
    own_speed_mps: float = OWN_SHIP_SPEED_MPS,
    safety_radius_m: float = VO_SAFETY_RADIUS_M,
    horizon_s: float = VO_TIME_HORIZON_S,
):
    """Return a VO heading correction and legacy diagnostic arrays.

    Positions are in nautical miles and headings are in radians. Target
    speeds are in m/s. If target velocities are omitted, the paper's common
    target speed is used and targets are assumed to maintain heading zero.
    """
    x_ob = np.asarray(x_ob, dtype=float)
    y_ob = np.asarray(y_ob, dtype=float)
    n_targets = x_ob.size
    if n_targets == 0:
        empty = np.zeros(0, dtype=float)
        return 0.0, empty, empty, empty, empty

    if v_ob is None:
        v_ob = np.full(n_targets, TARGET_SHIP_SPEED_MPS, dtype=float)
    else:
        v_ob = np.asarray(v_ob, dtype=float)
    if psi_ob is None:
        psi_ob = np.zeros(n_targets, dtype=float)
    else:
        psi_ob = np.asarray(psi_ob, dtype=float)

    relative_position = np.column_stack((x_ob - x, y_ob - y))
    target_velocity = np.column_stack((
        v_ob * np.cos(psi_ob) / METERS_PER_NAUTICAL_MILE,
        v_ob * np.sin(psi_ob) / METERS_PER_NAUTICAL_MILE,
    ))
    candidate_offsets = np.deg2rad(
        np.arange(-VO_HEADING_SEARCH_DEG,
                  VO_HEADING_SEARCH_DEG + VO_HEADING_STEP_DEG,
                  VO_HEADING_STEP_DEG)
    )
    candidate_headings = float(psi) + candidate_offsets
    own_speed_nmi_s = own_speed_mps / METERS_PER_NAUTICAL_MILE
    candidate_velocity = np.column_stack((
        own_speed_nmi_s * np.cos(candidate_headings),
        own_speed_nmi_s * np.sin(candidate_headings),
    ))

    safety_radius_nmi = safety_radius_m / METERS_PER_NAUTICAL_MILE
    free = np.ones(candidate_headings.size, dtype=bool)
    clearance = np.full(candidate_headings.size, np.inf, dtype=float)
    for target_position, target_velocity in zip(relative_position, target_velocity):
        relative_velocity = target_velocity[None, :] - candidate_velocity
        speed_sq = np.sum(relative_velocity * relative_velocity, axis=1)
        tcpa = np.divide(
            -np.sum(target_position[None, :] * relative_velocity, axis=1),
            speed_sq,
            out=np.full(candidate_headings.size, np.inf),
            where=speed_sq > 1e-12,
        )
        tcpa_clip = np.clip(tcpa, 0.0, horizon_s)
        closest_vector = target_position[None, :] + relative_velocity * tcpa_clip[:, None]
        dcpa = np.linalg.norm(closest_vector, axis=1)
        collision = (tcpa >= 0.0) & (tcpa <= horizon_s) & (dcpa <= safety_radius_nmi)
        free &= ~collision
        clearance = np.minimum(clearance, dcpa)

    if np.any(free):
        scores = np.abs(candidate_offsets)
        scores[~free] = np.inf
        selected = int(np.argmin(scores))
    else:
        # If every candidate intersects a VO, select the heading with the
        # greatest predicted clearance and then the smallest deviation.
        best_clearance = np.max(clearance)
        candidates = np.flatnonzero(np.isclose(clearance, best_clearance))
        selected = int(candidates[np.argmin(np.abs(candidate_offsets[candidates]))])

    correction = float(_wrap_angle(candidate_headings[selected] - float(psi)))
    line_of_sight = np.arctan2(y_ob - y, x_ob - x)
    bearing_ob = float(psi) - line_of_sight
    distance_ob = np.sqrt((x_ob - x) ** 2 + (y_ob - y) ** 2)
    w_r = (distance_ob <= safety_radius_nmi).astype(float)
    w_b = -np.cos(bearing_ob)
    return correction, w_b, w_r, distance_ob, bearing_ob
