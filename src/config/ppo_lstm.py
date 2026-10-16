"""Configuration loading and validation for the Sawada-style PPO-LSTM baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.config.paper_parameters import SIMULATION_DT_S, SIMULATION_TIME_S
from src.utils.imazu_cases import get_case_numbers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "ppo_lstm.json"


def _positive(mapping: Dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"ppo_lstm.{key} must be a positive number")
    return float(value)


def load_ppo_lstm_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load a JSON config and enforce the paper-comparison invariants."""
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("PPO-LSTM config must be a JSON object")

    required_sections = ("simulation", "observation", "action", "reward", "network", "ppo")
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"PPO-LSTM config section {section!r} is required")

    cases = config.get("case_numbers")
    expected_cases = list(get_case_numbers())
    if cases != expected_cases:
        raise ValueError(f"ppo_lstm.case_numbers must equal {expected_cases}")

    simulation = config["simulation"]
    dt = _positive(simulation, "dt_s")
    episode_time = _positive(simulation, "episode_time_s")
    policy_interval = _positive(simulation, "policy_interval_s")
    if abs(dt - SIMULATION_DT_S) > 1e-12:
        raise ValueError(f"PPO-LSTM dt_s must match the paper simulation ({SIMULATION_DT_S})")
    if abs(episode_time - SIMULATION_TIME_S) > 1e-12:
        raise ValueError(
            f"PPO-LSTM episode_time_s must match the paper simulation ({SIMULATION_TIME_S})"
        )
    ratio = policy_interval / dt
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("policy_interval_s must be an integer number of simulation steps")

    observation = config["observation"]
    sensor_radius = _positive(observation, "sensor_radius_nm")
    radial_resolution = _positive(observation, "radial_resolution_nm")
    angle_resolution = _positive(observation, "angle_resolution_deg")
    if abs(360.0 / angle_resolution - round(360.0 / angle_resolution)) > 1e-9:
        raise ValueError("angle_resolution_deg must divide 360 degrees")
    if abs(sensor_radius / radial_resolution - round(sensor_radius / radial_resolution)) > 1e-9:
        raise ValueError("radial_resolution_nm must divide sensor_radius_nm")
    if int(observation.get("numeric_state_dim", 0)) != 10:
        raise ValueError("numeric_state_dim must be 10 for the implemented observation schema")

    action = config["action"]
    minimum = float(action.get("minimum_heading_increment_deg", 0.0))
    maximum = float(action.get("maximum_heading_increment_deg", 0.0))
    if minimum >= maximum or minimum != -10.0 or maximum != 10.0:
        raise ValueError("Sawada autopilot action range must be exactly -10 to +10 degrees")

    ppo = config["ppo"]
    for key in (
        "actor_learning_rate", "critic_learning_rate", "gamma", "gae_lambda",
        "clip_epsilon", "gradient_norm_clip", "rollout_steps", "sequence_length",
        "minibatch_sequences", "epochs", "total_policy_steps",
    ):
        _positive(ppo, key)
    if int(ppo["rollout_steps"]) < int(ppo["sequence_length"]):
        raise ValueError("rollout_steps must be at least sequence_length")

    validated = dict(config)
    validated["config_path"] = config_path
    validated["policy_interval_steps"] = int(round(ratio))
    validated["grid_shape"] = (
        int(round(sensor_radius / radial_resolution)),
        int(round(360.0 / angle_resolution)),
    )
    return validated
