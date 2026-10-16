"""Encounter geometry for powered vessels in sight of one another.

Angles are radians, absolute headings are counterclockwise from east, and
relative bearings are clockwise (positive to starboard). These engineering
categories are not a complete implementation of COLREGs.
"""
from __future__ import annotations

import numpy as np


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def classify_encounter(bearing, target_heading, own_heading, own_speed, target_speed):
    if target_heading is None or own_heading is None:
        return "unknown"
    values = [bearing, target_heading, own_heading, own_speed, target_speed]
    if not np.all(np.isfinite(values)):
        return "unknown"
    b = float(wrap_angle(bearing))
    relative_heading = float(wrap_angle(target_heading - own_heading))
    target_bearing = float(wrap_angle(relative_heading + b - np.pi))
    closing_speed = own_speed * np.cos(b) - target_speed * np.cos(relative_heading + b)
    if closing_speed <= 1e-6:
        return "none"
    stern_sector = np.deg2rad(112.5)
    # Rule 13 takes precedence over crossing classification. Identify which
    # ship approaches the other's stern, not merely which ship sees a target aft.
    if abs(target_bearing) > stern_sector:
        return "overtaking"
    if abs(b) > stern_sector:
        return "being_overtaken"
    if (abs(b) <= np.deg2rad(6) and abs(target_bearing) <= np.deg2rad(6)
            and abs(abs(relative_heading) - np.pi) <= np.deg2rad(12)):
        return "head_on"
    if abs(relative_heading) <= np.deg2rad(6):
        return "none"
    return "crossing_give_way" if b >= 0 else "crossing_stand_on"


def relative_metrics(own_position, own_heading, own_speed, target_positions,
                     target_speeds, target_headings):
    """CPA and range from one synchronous SI state; stationary TCPA is +inf."""
    target_positions = np.asarray(target_positions, dtype=float).reshape(-1, 2)
    target_speeds = np.asarray(target_speeds, dtype=float)
    target_headings = np.asarray(target_headings, dtype=float)
    position = target_positions - np.asarray(own_position)
    own_velocity = own_speed * np.array([np.cos(own_heading), np.sin(own_heading)])
    velocity = target_speeds[:, None] * np.column_stack(
        (np.cos(target_headings), np.sin(target_headings))) - own_velocity
    speed_sq = np.sum(velocity ** 2, axis=1)
    tcpa = np.divide(-np.sum(position * velocity, axis=1), speed_sq,
                     out=np.full(len(position), np.inf), where=speed_sq > 1e-12)
    projection_time = np.where(np.isfinite(tcpa), tcpa, 0.0)
    dcpa = np.linalg.norm(position + projection_time[:, None] * velocity, axis=1)
    distance = np.linalg.norm(position, axis=1)
    bearing = wrap_angle(own_heading - np.arctan2(position[:, 1], position[:, 0]))
    return dcpa, tcpa, distance, bearing, np.sqrt(speed_sq)
