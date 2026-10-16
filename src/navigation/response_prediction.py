"""Nominal closed-loop heading response used to screen final VO commands.

Uses the project's controller, saturation, bias decay and commanded translation
speed. Future stochastic disturbances are zero, so this is a model prediction,
not a robust safety certificate. Closest range is checked along each line segment.
"""
from functools import lru_cache
import numpy as np
from src.config.paper_parameters import (CONTROL_KP_YAW, CONTROL_KD_YAW, CONTROL_KI_YAW,
    ACTUATOR_SATURATION, DYNAMICS_K_PSI, DYNAMICS_T_PSI_S, DYNAMICS_BIAS_TIME_FACTOR)
from src.navigation.encounters import wrap_angle


@lru_cache(maxsize=16)
def _transitions(times):
    # Stable linear closed-loop tail after the saturated transient. The bias
    # state is included; unlike a straight-line tail, yaw response continues.
    gain, tau = DYNAMICS_K_PSI, DYNAMICS_T_PSI_S
    matrix = np.array([[0, 1, 0], [-gain*CONTROL_KP_YAW/tau,
        (-1+gain*CONTROL_KD_YAW)/tau, -gain/tau],
        [0, 0, -1/(DYNAMICS_BIAS_TIME_FACTOR*tau)]])
    values, vectors = np.linalg.eig(matrix)
    return np.real(np.einsum('ij,tj,jk->tik', vectors,
        np.exp(np.asarray(times)[:, None]*values), np.linalg.inv(vectors)))


def _segment_clearance(own_start, own_end, target_start, target_end):
    # Shape (..., candidates, targets, xy), projected onto each relative segment.
    start = target_start - own_start
    change = (target_end - own_end) - start
    speed_sq = np.sum(change*change, axis=-1)
    fraction = np.divide(-np.sum(start*change, axis=-1), speed_sq,
        out=np.zeros_like(speed_sq), where=speed_sq > 1e-15)
    closest = start + np.clip(fraction, 0, 1)[..., None] * change
    return np.linalg.norm(closest, axis=-1)


def _linear_clearance(headings, state, targets, target_velocity, speed, horizon, dt):
    times = np.unique(np.append(np.arange(0, horizon, dt), horizon))
    initial = np.column_stack((wrap_angle(state[2]-headings),
                              np.full(len(headings),state[3]), np.full(len(headings),state[4])))
    response = np.einsum('tij,cj->tci', _transitions(tuple(times)), initial)
    torque = -CONTROL_KP_YAW*response[...,0] + CONTROL_KD_YAW*response[...,1]
    valid = np.all(np.abs(torque) <= ACTUATOR_SATURATION+1e-8,axis=0)
    clearance = np.full(len(headings), np.nan)
    if not valid.any():return valid,clearance
    psi = headings[None,valid] + response[:,valid,0]
    velocity = speed*np.stack((np.cos(psi),np.sin(psi)),axis=-1)
    increments = .5*(velocity[:-1]+velocity[1:])*np.diff(times)[:,None,None]
    positions = np.concatenate((np.broadcast_to(state[:2],(1,valid.sum(),2)),
                               state[:2][None,None]+np.cumsum(increments,axis=0)))
    other = targets[None]+times[:,None,None]*target_velocity[None]
    ranges = _segment_clearance(positions[:-1,:,None],positions[1:,:,None],other[:-1,None],other[1:,None])
    clearance[valid] = ranges.min(axis=(0,2))
    return valid,clearance


def response_clearance(headings, state, integral_error, target_positions,
                       target_velocity, own_speed, horizon, dt=.5, transient_s=60.):
    headings, state = np.asarray(headings, float), np.asarray(state, float)
    target_positions, target_velocity = np.asarray(target_positions), np.asarray(target_velocity)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError('Heading response requires a finite six-state vessel vector')
    if dt <= 0 or not np.isfinite(dt) or not np.isfinite(transient_s) or transient_s <= 0:
        raise ValueError('Prediction integration settings must be positive and finite')
    if len(target_positions) == 0:
        return np.full(len(headings), np.inf)
    if not CONTROL_KI_YAW:
        linear, fast_clearance = _linear_clearance(headings, state, target_positions,
            target_velocity, own_speed, horizon, dt)
        if linear.all():return fast_clearance
        if linear.any():
            fast_clearance[~linear] = response_clearance(headings[~linear], state, integral_error,
                target_positions, target_velocity, own_speed, horizon, dt, transient_s)
            return fast_clearance
    count = len(headings)
    position = np.broadcast_to(state[:2], (count, 2)).copy()
    psi, yaw, bias = (np.full(count, value) for value in state[2:5])
    integral = np.full(count, integral_error, dtype=float)
    clearance = np.full(count, np.inf)
    time = 0.
    # Nonzero integral gain needs explicit simulation throughout.
    transient = horizon if CONTROL_KI_YAW else min(horizon, transient_s)
    while time < transient - 1e-10:
        step = min(dt, transient-time)
        error = wrap_angle(headings - psi)
        integral += step*error
        torque = np.clip(CONTROL_KP_YAW*error + CONTROL_KD_YAW*yaw + CONTROL_KI_YAW*integral,
                         -ACTUATOR_SATURATION, ACTUATOR_SATURATION)
        next_position = position + own_speed*np.column_stack((np.cos(psi), np.sin(psi)))*step
        ranges = _segment_clearance(position[:, None, :], next_position[:, None, :],
            (target_positions + time*target_velocity)[None],
            (target_positions + (time+step)*target_velocity)[None])
        clearance = np.minimum(clearance, ranges.min(axis=1))
        position = next_position
        psi = psi + step*yaw
        yaw = yaw + step*(-yaw + DYNAMICS_K_PSI*(torque-bias))/DYNAMICS_T_PSI_S
        bias = bias*(1-step/(DYNAMICS_BIAS_TIME_FACTOR*DYNAMICS_T_PSI_S))
        time += step
    if time >= horizon - 1e-10:
        return clearance
    remaining = horizon-time
    times = np.unique(np.append(np.arange(0, remaining, 1.), remaining))
    error = wrap_angle(psi-headings)
    initial = np.column_stack((error, yaw, bias))
    tail = np.einsum('tij,cj->tci', _transitions(tuple(times)), initial)
    torque = -CONTROL_KP_YAW*tail[..., 0] + CONTROL_KD_YAW*tail[..., 1]
    if np.any(np.abs(torque) > ACTUATOR_SATURATION + 1e-8):
        # A pathological unsaturated-tail assumption must not silently certify
        # safety. Fall back to explicit saturated simulation for the full horizon.
        return response_clearance(headings, state, integral_error, target_positions,
                                  target_velocity, own_speed, horizon, dt, horizon)
    tail_psi = headings[None] + tail[..., 0]
    velocity = own_speed*np.stack((np.cos(tail_psi), np.sin(tail_psi)), axis=-1)
    step_sizes = np.diff(times)
    # Trapezoid position integration of the continued linear heading response.
    increments = .5*(velocity[:-1] + velocity[1:])*step_sizes[:, None, None]
    positions = np.concatenate((position[None], position[None] + np.cumsum(increments, axis=0)))
    targets = target_positions[None] + (time+times)[:, None, None]*target_velocity[None]
    ranges = _segment_clearance(positions[:-1, :, None], positions[1:, :, None],
                                targets[:-1, None], targets[1:, None])
    return np.minimum(clearance, ranges.min(axis=(0, 2)))
