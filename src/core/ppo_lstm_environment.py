"""Unified RTRA simulation environment for the Sawada-style PPO-LSTM baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.config.paper_parameters import (
    ACTUATOR_SATURATION,
    OWN_SHIP_LENGTH_M,
    OWN_SHIP_SPEED_MPS,
    TARGET_SHIP_LENGTH_M,
)
from src.core.integration import integration
from src.navigation.encounters import relative_metrics
from src.core.experiment_metrics import trajectory_metrics, hull_collision_at_state
from src.decision_making.rtra_llm_supervisor import RiskTriggeredLLMSupervisor, Action
from src.decision_making.ppo_lstm_observation import (
    NMI_TO_METERS,
    build_numeric_state,
    build_ozt_grid,
    wrap_angle,
)
from src.dynamics.actuator_modeling import actuator_modeling
from src.dynamics.controller import controller
from src.dynamics.vessel_dynamics import vessel_dynamics
from src.navigation.obstacle_sim import obstacle_sim
from src.risk_assessment.risk_calculations import risk_calculations
from src.utils.imazu_cases import get_obstacle_data


@dataclass
class PpoLstmObservation:
    grid: np.ndarray
    numeric: np.ndarray


def _relative_metrics(
    own_position: np.ndarray,
    own_heading: float,
    own_speed: float,
    target_positions: np.ndarray,
    target_speeds: np.ndarray,
    target_headings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shared convention: positive relative bearing means starboard.
    return relative_metrics(own_position, own_heading, own_speed, target_positions,
                            target_speeds, target_headings)


class SawadaPpoLstmEnv:
    """A lightweight recurrent-RL environment using the paper's simulation model.

    One call to :meth:`step` applies a heading-increment command and advances the
    common vessel model for 10 seconds (100 integrations at 0.1 seconds by
    default).  It deliberately bypasses VO: PPO-LSTM directly supplies the
    desired heading to the same controller used by the other methods.
    """

    def __init__(self, config: Dict[str, Any], *, training: bool = False):
        self.config = config
        self.training = bool(training)
        self.simulation = config["simulation"]
        self.observation_config = config["observation"]
        self.action_config = config["action"]
        self.reward_config = config["reward"]
        self.dt = float(self.simulation["dt_s"])
        self.policy_interval_steps = int(config["policy_interval_steps"])
        self.max_policy_steps = int(
            round(float(self.simulation["episode_time_s"]) / float(self.simulation["policy_interval_s"]))
        )
        self.rng = np.random.default_rng()
        self.case_number = 1
        self._reset_state()

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return tuple(self.config["grid_shape"])

    @property
    def numeric_state_dim(self) -> int:
        return int(self.observation_config["numeric_state_dim"])

    @property
    def action_limit_deg(self) -> float:
        return float(self.action_config["maximum_heading_increment_deg"])

    def _reset_state(self) -> None:
        self.state = np.zeros(6, dtype=float)
        self.target_x = np.empty(0, dtype=float)
        self.target_y = np.empty(0, dtype=float)
        self.target_speeds = np.empty(0, dtype=float)
        self.target_headings = np.empty(0, dtype=float)
        self.waypoint = np.asarray(
            [
                float(self.simulation["waypoint_x_nm"]) * NMI_TO_METERS,
                float(self.simulation["waypoint_y_nm"]) * NMI_TO_METERS,
            ],
            dtype=float,
        )
        self.integral_heading_error = 0.0
        self.current_action_deg = 0.0
        self.desired_heading = 0.0
        self.policy_step = 0
        self.elapsed_s = 0.0
        self.done = False
        self.termination_reason = "running"
        self.episode_return = 0.0
        self.collision_occurred = False
        self.unsafe_passing_occurred = False
        self.path_length_m = 0.0
        self.min_actual_separation_m = float("inf")
        self.min_dcpa_m = float("inf")
        self.max_risk = 0.0
        self.risk_sum = 0.0
        self.risk_samples = 0
        self.action_history = []
        self.inference_latency_s = []
        self.trace = []
        self.evaluation_samples = []
        self.rule_observer = (None if self.training else RiskTriggeredLLMSupervisor(provider="offline-rules", rule_only=True))

    def reset(
        self,
        *,
        case_number: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> PpoLstmObservation:
        self._reset_state()
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            # The shared vessel model draws its disturbance through numpy's
            # legacy global generator; seed both sources for reproducibility.
            np.random.seed(int(seed))
        cases = self.config["case_numbers"]
        self.case_number = int(case_number if case_number is not None else self.rng.choice(cases))
        x_ob, y_ob, speeds, headings = get_obstacle_data(self.case_number)
        self.target_x = np.asarray(x_ob, dtype=float)
        self.target_y = np.asarray(y_ob, dtype=float)
        self.target_speeds = np.asarray(speeds, dtype=float)
        self.target_headings = np.asarray(headings, dtype=float)
        if self.training:
            jitter = float(self.simulation.get("initial_heading_jitter_deg", 0.0))
            self.state[2] = math.radians(float(self.rng.uniform(-jitter, jitter)))
        self.desired_heading = float(self.state[2])
        self.collision_occurred = hull_collision_at_state(self.state[:2],self.state[2],
            self._target_positions(),self.target_headings)
        self._record_trace()
        return self._observation()

    def _target_positions(self) -> np.ndarray:
        return np.stack((self.target_x, self.target_y), axis=1)

    def _observation(self) -> PpoLstmObservation:
        position = self.state[:2]
        grid = build_ozt_grid(
            position,
            float(self.state[2]),
            OWN_SHIP_SPEED_MPS,
            self._target_positions(),
            self.target_speeds,
            self.target_headings,
            self.config,
        )
        numeric = build_numeric_state(
            own_position_m=position,
            own_heading_rad=float(self.state[2]),
            yaw_rate_rad_s=float(self.state[3]),
            own_speed_mps=float(self.state[5]),
            speed_command_mps=OWN_SHIP_SPEED_MPS,
            waypoint_m=self.waypoint,
            current_action_deg=self.current_action_deg,
            max_action_deg=self.action_limit_deg,
            sensor_radius_nm=float(self.observation_config["sensor_radius_nm"]),
        )
        return PpoLstmObservation(grid=grid, numeric=numeric)

    def _metrics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dcpa, tcpa, distance, _bearing, relative_speed = _relative_metrics(
            self.state[:2],
            float(self.state[2]),
            OWN_SHIP_SPEED_MPS,
            self._target_positions(),
            self.target_speeds,
            self.target_headings,
        )
        risk = np.asarray(risk_calculations(dcpa, tcpa, distance, relative_speed), dtype=float)
        return dcpa, tcpa, distance, risk

    def _record_trace(self) -> None:
        if self.target_x.size:
            dcpa, tcpa, distance, risk = self._metrics()
            min_dcpa = float(np.min(np.abs(dcpa)))
            min_distance = float(np.min(distance))
            max_risk = float(np.max(risk))
        else:
            tcpa = np.empty(0)
            min_dcpa = min_distance = float("inf")
            max_risk = 0.0
        self.trace.append(
            {
                "time_s": self.elapsed_s,
                "x_m": float(self.state[0]),
                "y_m": float(self.state[1]),
                "heading_rad": float(self.state[2]),
                "yaw_rate_rad_s": float(self.state[3]),
                "action_deg": float(self.current_action_deg),
                "desired_heading_rad": float(self.desired_heading),
                "min_distance_m": min_distance,
                "min_dcpa_m": min_dcpa,
                "min_tcpa_s": float(np.min(tcpa)) if tcpa.size else float("nan"),
                "max_risk": max_risk,
            }
        )

    def step(self, action_deg: float) -> Tuple[PpoLstmObservation, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called after episode termination; call reset()")
        lower = float(self.action_config["minimum_heading_increment_deg"])
        upper = float(self.action_config["maximum_heading_increment_deg"])
        self.current_action_deg = float(np.clip(action_deg, lower, upper))
        self.action_history.append(self.current_action_deg)
        self.desired_heading = float(wrap_angle(self.state[2] + math.radians(self.current_action_deg)))
        previous_waypoint_distance = float(np.linalg.norm(self.waypoint - self.state[:2]))
        interval_unsafe = False
        interval_collision = self.collision_occurred if self.policy_step == 0 else False

        safe_distance = float(self.observation_config["safe_passing_distance_nm"]) * NMI_TO_METERS

        for _ in range(self.policy_interval_steps):
            # Store the same left-endpoint SI state used by the VO/LLM runner.
            # Training does not retain dense traces for an entire run.
            if not self.training:
                dcpa_before, tcpa_before, distance_before, risk_before = self._metrics()
                _, _, _, bearing_before, _ = relative_metrics(
                    self.state[:2], self.state[2], OWN_SHIP_SPEED_MPS, self._target_positions(),
                    self.target_speeds, self.target_headings)
                self.rule_observer.update(risk=risk_before, distance=distance_before/1852,
                    bearing=bearing_before, dcpa=dcpa_before/1852, tcpa=tcpa_before,
                    step=len(self.evaluation_samples), target_heading=self.target_headings,
                    own_heading=self.state[2], target_speed=self.target_speeds)
                offset = float(wrap_angle(self.desired_heading - self.state[2]))
                executed = Action.AR if offset < -1e-10 else Action.AL if offset > 1e-10 else Action.A0
                targets = self.rule_observer._last_validation_targets
                safe_prediction = None
                if executed == Action.A0 and self.rule_observer.memory.active_maneuver_action is not None:
                    from src.navigation.response_prediction import response_clearance
                    from src.config.paper_parameters import VO_TIME_HORIZON_S, VO_SAFETY_RADIUS_M
                    target_velocity = self.target_speeds[:,None]*np.column_stack(
                        (np.cos(self.target_headings),np.sin(self.target_headings)))
                    predicted = response_clearance(np.asarray([self.desired_heading]),self.state,
                        self.integral_heading_error,self._target_positions(),target_velocity,
                        OWN_SHIP_SPEED_MPS,VO_TIME_HORIZON_S)
                    safe_prediction = bool(predicted[0] > VO_SAFETY_RADIUS_M)
                self.rule_observer.feedback_execution(self.desired_heading,self.state[2],
                    len(self.evaluation_samples),safe_prediction=safe_prediction)
                maneuver = self.rule_observer.memory.active_maneuver_action or executed
                self.evaluation_samples.append({
                    'time': self.elapsed_s, 'state': self.state.copy(),
                    'targets': self._target_positions().copy(), 'risk': risk_before.copy(),
                    'dcpa': dcpa_before.copy(), 'tcpa': tcpa_before.copy(),
                    'distance': distance_before.copy(),
                    'rule_evaluable': bool(targets),
                    'rule_violation': any(not self.rule_observer._is_action_legal_for_target(maneuver, t) for t in targets),
                    'kdir': 1 if offset < -1e-10 else -1 if offset > 1e-10 else 0,
                })
            old_position = self.state[:2].copy()
            tau, speed_command, self.integral_heading_error = controller(
                self.desired_heading,
                float(self.state[2]),
                float(self.state[3]),
                OWN_SHIP_SPEED_MPS,
                float(self.state[4]),
                self.integral_heading_error,
                self.dt,
            )
            applied_tau = actuator_modeling(tau, ACTUATOR_SATURATION)
            derivative = vessel_dynamics(self.state.copy(), [applied_tau, speed_command])
            self.state = integration(self.state, derivative, self.dt)
            self.state[2] = float(wrap_angle(self.state[2]))
            self.path_length_m += float(np.linalg.norm(self.state[:2] - old_position))
            self.target_x, self.target_y, _vx, _vy = obstacle_sim(
                self.target_x,
                self.target_y,
                self.target_speeds,
                self.target_headings,
                self.dt,
            )
            self.elapsed_s += self.dt

            dcpa, _tcpa, distance, risk = self._metrics()
            min_distance = float(np.min(distance))
            self.min_actual_separation_m = min(self.min_actual_separation_m, min_distance)
            self.min_dcpa_m = min(self.min_dcpa_m, float(np.min(np.abs(dcpa))))
            self.max_risk = max(self.max_risk, float(np.max(risk)))
            self.risk_sum += float(np.mean(risk))
            self.risk_samples += 1
            interval_unsafe |= min_distance <= safe_distance
            interval_collision |= hull_collision_at_state(self.state[:2], self.state[2],
                self._target_positions(), self.target_headings)

        self.policy_step += 1
        self.unsafe_passing_occurred |= interval_unsafe
        self.collision_occurred |= interval_collision
        waypoint_distance = float(np.linalg.norm(self.waypoint - self.state[:2]))
        distance_nm = waypoint_distance / NMI_TO_METERS
        progress_nm = (previous_waypoint_distance - waypoint_distance) / NMI_TO_METERS

        reward = float(self.reward_config["waypoint_inverse_distance_scale"]) * math.tanh(
            1.0 / max(distance_nm, 1e-6)
        )
        reward += float(self.reward_config["waypoint_progress_scale"]) * progress_nm
        # Route is +x and starboard is -y in this coordinate convention.
        if self.state[1] <= 0.0:
            reward += float(self.reward_config["starboard_reward"])
        reward -= float(self.reward_config["heading_stability_scale"]) * abs(self.state[3] / math.pi)
        if interval_unsafe:
            reward += float(self.reward_config["unsafe_passing_penalty"])

        reached_waypoint = waypoint_distance <= float(self.simulation["waypoint_tolerance_m"])
        out_of_bounds = abs(float(self.state[1])) > float(self.simulation["cross_track_limit_nm"]) * NMI_TO_METERS
        time_limit = self.policy_step >= self.max_policy_steps
        collision_terminal = interval_collision and bool(self.simulation.get("terminate_on_collision", False))
        self.done = reached_waypoint or out_of_bounds or time_limit or collision_terminal
        termination_reason = "running"
        if reached_waypoint:
            reward += float(self.reward_config["waypoint_reward"])
            termination_reason = "waypoint"
        elif out_of_bounds:
            reward += float(self.reward_config["out_of_bounds_penalty"])
            termination_reason = "out_of_bounds"
        elif collision_terminal:
            reward += float(self.reward_config["collision_penalty"])
            termination_reason = "collision"
        elif time_limit:
            termination_reason = "time_limit"
            if self.collision_occurred:
                reward += float(self.reward_config["collision_penalty"])

        self.termination_reason = termination_reason
        self.episode_return += float(reward)

        self._record_trace()
        info = {
            "case_number": self.case_number,
            "policy_step": self.policy_step,
            "low_level_steps": self.policy_step * self.policy_interval_steps,
            "elapsed_s": self.elapsed_s,
            "collision": self.collision_occurred,
            "unsafe_passing": self.unsafe_passing_occurred,
            "success": reached_waypoint and not self.collision_occurred,
            "termination_reason": termination_reason,
            "min_actual_separation_m": self.min_actual_separation_m,
            "min_dcpa_m": self.min_dcpa_m,
            "max_risk": self.max_risk,
            "avg_risk": self.risk_sum / max(self.risk_samples, 1),
            "path_length_m": self.path_length_m,
            "waypoint_distance_m": waypoint_distance,
        }
        return self._observation(), float(reward), self.done, info

    def summary(self) -> Dict[str, Any]:
        final_distance = float(np.linalg.norm(self.state[:2]))
        straight_distance = max(final_distance, 1e-9)
        nonzero = np.asarray(self.action_history, dtype=float)
        nonzero = nonzero[np.abs(nonzero) > 0.1]
        if nonzero.size < 2:
            consistency = 100.0
        else:
            consistency = 100.0 * float(np.mean(np.sign(nonzero[1:]) == np.sign(nonzero[:-1])))
        summary = {
            "case_number": self.case_number,
            "policy_steps": self.policy_step,
            "low_level_steps": self.policy_step * self.policy_interval_steps,
            "elapsed_s": self.elapsed_s,
            "collision": self.collision_occurred,
            "success": (not self.collision_occurred) and self.termination_reason == "waypoint",
            "collision_free_episode": not self.collision_occurred,
            "termination_reason": self.termination_reason,
            "episode_return": self.episode_return,
            "R_max": self.max_risk,
            "R_avg": self.risk_sum / max(self.risk_samples, 1),
            "min_dcpa_nm": self.min_dcpa_m / NMI_TO_METERS,
            "min_actual_separation_nm": self.min_actual_separation_m / NMI_TO_METERS,
            # Paired VO metrics are added by the common evaluation runner.
            "A_turn_pct": float("nan"),
            "delta_D_pct": float("nan"),
            "action_sign_persistence_pct": consistency,
            "route_excess_pct": 100.0 * (self.path_length_m / straight_distance - 1.0),
            "path_length_m": self.path_length_m,
            "final_distance_nm": final_distance / NMI_TO_METERS,
            "action_count": len(self.action_history),
        }

        if self.evaluation_samples:
            summary.update(trajectory_metrics(self.evaluation_trajectory()))
            summary['success'] = self.termination_reason == 'waypoint' and not summary['collision']
            summary['collision_free_episode'] = not summary['collision']
        return summary

    def evaluation_trajectory(self):
        """Dense trace for common metrics; unavailable during training."""
        if not self.evaluation_samples:
            raise ValueError("No evaluation samples; run an evaluation episode first")
        samples = self.evaluation_samples
        states = np.asarray([row['state'] for row in samples])
        targets = np.asarray([row['targets'] for row in samples])
        return {
            'time': np.asarray([row['time'] for row in samples]),
            'x': states[:, 0], 'y': states[:, 1], 'psi': states[:, 2],
            'kdir': np.asarray([row['kdir'] for row in samples]),
            'risk': np.asarray([row['risk'] for row in samples]),
            'dcpa': np.asarray([row['dcpa'] for row in samples]),
            'tcpa': np.asarray([row['tcpa'] for row in samples]),
            'distance_ob': np.asarray([row['distance'] for row in samples]),
            'obstacles_x_history': targets[:, :, 0],
            'obstacles_y_history': targets[:, :, 1],
            'obstacles_heading_rad': self.target_headings.copy(),
            'target_speed_mps': self.target_speeds.copy(),
            'final_position_m': self.state[:2].copy(),
            'final_heading_rad': float(self.state[2]),
            'final_target_positions_m': self._target_positions().copy(),
            'waypoints_x': np.asarray([0, self.waypoint[0] / NMI_TO_METERS]),
            'waypoints_y': np.asarray([0, self.waypoint[1] / NMI_TO_METERS]),
            'dt': self.dt,
            'rule_evaluable': np.asarray([row['rule_evaluable'] for row in samples]),
            'rule_violation': np.asarray([row['rule_violation'] for row in samples]),
        }
