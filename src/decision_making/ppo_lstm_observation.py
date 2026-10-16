"""Sawada-style OZT grid sensor and numerical observation construction.

The paper represents a variable number of target ships with a fixed-size binary
polar grid.  This method-level reimplementation uses time-matched occupancy:
each grid-cell centre is occupied when the own ship and a target ship would be
inside the configured safe-passing domain at the own ship's arrival time.  A
forward-elongated domain supplies the paper's bow-crossing bias, and the same
calculation remains defined when a target is already inside the nominal OZT.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


NMI_TO_METERS = 1852.0


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def grid_shape(config: Dict) -> Tuple[int, int]:
    obs = config["observation"]
    radial = int(round(obs["sensor_radius_nm"] / obs["radial_resolution_nm"]))
    angular = int(round(360.0 / obs["angle_resolution_deg"]))
    return radial, angular


def build_ozt_grid(
    own_position_m: np.ndarray,
    own_heading_rad: float,
    own_speed_mps: float,
    target_positions_m: np.ndarray,
    target_speeds_mps: np.ndarray,
    target_headings_rad: np.ndarray,
    config: Dict,
) -> np.ndarray:
    """Return a binary radial-by-angular OZT occupancy grid."""
    obs = config["observation"]
    radial_count, angular_count = grid_shape(config)
    radial_step_m = float(obs["radial_resolution_nm"]) * NMI_TO_METERS
    angular_step = math.radians(float(obs["angle_resolution_deg"]))
    safe_distance_m = float(obs["safe_passing_distance_nm"]) * NMI_TO_METERS
    bow_factor = float(obs.get("bow_crossing_factor", 2.0))

    radii = (np.arange(radial_count, dtype=float) + 0.5) * radial_step_m
    bearings = -np.pi + (np.arange(angular_count, dtype=float) + 0.5) * angular_step
    rr, aa = np.meshgrid(radii, bearings, indexing="ij")
    world_angle = own_heading_rad + aa
    points = np.stack(
        (own_position_m[0] + rr * np.cos(world_angle),
         own_position_m[1] + rr * np.sin(world_angle)),
        axis=-1,
    )
    arrival_time = rr / max(float(own_speed_mps), 0.1)
    occupied = np.zeros((radial_count, angular_count), dtype=bool)

    for target_position, target_speed, target_heading in zip(
        np.asarray(target_positions_m, dtype=float),
        np.asarray(target_speeds_mps, dtype=float),
        np.asarray(target_headings_rad, dtype=float),
    ):
        direction = np.array([math.cos(target_heading), math.sin(target_heading)])
        starboard = np.array([direction[1], -direction[0]])
        target_at_arrival = target_position + arrival_time[..., None] * target_speed * direction
        relative = points - target_at_arrival
        longitudinal = relative @ direction
        lateral = relative @ starboard

        # Ahead of the target the protected domain is elongated, discouraging
        # bow crossing.  Astern and abeam it retains the circular 0.5 NM OZT.
        longitudinal_scale = np.where(longitudinal >= 0.0, safe_distance_m * bow_factor, safe_distance_m)
        domain_value = (longitudinal / longitudinal_scale) ** 2 + (lateral / safe_distance_m) ** 2
        occupied |= domain_value <= 1.0

        # Inside-OZT continuation: when already too close, retain a capsule in
        # the target's direction so the observation never becomes falsely empty.
        current_delta = own_position_m - target_position
        if np.linalg.norm(current_delta) <= safe_distance_m:
            projection = (points - target_position) @ direction
            nearest = target_position + np.maximum(projection, 0.0)[..., None] * direction
            occupied |= np.linalg.norm(points - nearest, axis=-1) <= safe_distance_m

    return occupied.astype(np.float32)


def build_numeric_state(
    *,
    own_position_m: np.ndarray,
    own_heading_rad: float,
    yaw_rate_rad_s: float,
    own_speed_mps: float,
    speed_command_mps: float,
    waypoint_m: np.ndarray,
    current_action_deg: float,
    max_action_deg: float,
    sensor_radius_nm: float,
) -> np.ndarray:
    """Construct normalized own-ship, waypoint and current-command features."""
    delta = np.asarray(waypoint_m, dtype=float) - np.asarray(own_position_m, dtype=float)
    waypoint_bearing = math.atan2(delta[1], delta[0])
    relative_bearing = float(wrap_angle(waypoint_bearing - own_heading_rad))
    distance_nm = float(np.linalg.norm(delta) / NMI_TO_METERS)
    autopilot_delta = float(wrap_angle(waypoint_bearing - own_heading_rad)) / math.radians(max_action_deg)
    return np.asarray(
        [
            math.sin(own_heading_rad),
            math.cos(own_heading_rad),
            np.clip(yaw_rate_rad_s / math.radians(5.0), -1.0, 1.0),
            np.clip(own_speed_mps / max(speed_command_mps, 0.1), 0.0, 2.0) - 1.0,
            math.sin(relative_bearing),
            math.cos(relative_bearing),
            np.clip(distance_nm / sensor_radius_nm, 0.0, 4.0) / 2.0 - 1.0,
            np.clip(current_action_deg / max_action_deg, -1.0, 1.0),
            np.clip(autopilot_delta, -1.0, 1.0),
            np.clip(own_position_m[1] / (sensor_radius_nm * NMI_TO_METERS), -1.0, 1.0),
        ],
        dtype=np.float32,
    )
