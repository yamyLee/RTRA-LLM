from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.config.ppo_lstm import load_ppo_lstm_config
from src.core.ppo_lstm_environment import SawadaPpoLstmEnv
from src.core.simulation import parse_args, run_simulation
from src.decision_making.ppo_lstm_observation import build_ozt_grid

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class PpoLstmIntegrationTest(unittest.TestCase):
    def test_default_config_matches_paper_comparison_protocol(self):
        config = load_ppo_lstm_config()
        self.assertEqual(config["case_numbers"], list(range(1, 24)))
        self.assertEqual(config["policy_interval_steps"], 100)
        self.assertEqual(config["grid_shape"], (60, 180))
        self.assertEqual(config["action"]["minimum_heading_increment_deg"], -10.0)
        self.assertEqual(config["action"]["maximum_heading_increment_deg"], 10.0)

    def test_cli_exposes_ppo_lstm_and_removes_ppo_corall(self):
        args = parse_args(["--method", "ppo_lstm", "--ppo_checkpoint", "model.pt"])
        self.assertEqual(args.method, "ppo_lstm")
        self.assertEqual(args.ppo_checkpoint, "model.pt")
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--method", "ppo_corall"])

    def test_main_dispatches_to_ppo_lstm_runner(self):
        args = parse_args(["--method", "ppo_lstm", "--ppo_checkpoint", "model.pt"])
        expected = {"method": "ppo_lstm", "case_number": 1}
        with patch("src.core.ppo_lstm_simulation.run_ppo_lstm", return_value=expected) as runner:
            result = run_simulation(args)
        self.assertEqual(result, expected)
        runner.assert_called_once()

    def test_ozt_grid_has_fixed_shape_and_detects_collision_course(self):
        config = load_ppo_lstm_config()
        grid = build_ozt_grid(
            own_position_m=np.asarray([0.0, 0.0]),
            own_heading_rad=0.0,
            own_speed_mps=10.0,
            target_positions_m=np.asarray([[2000.0, 0.0]]),
            target_speeds_mps=np.asarray([10.0]),
            target_headings_rad=np.asarray([np.pi]),
            config=config,
        )
        self.assertEqual(grid.shape, (60, 180))
        self.assertGreater(float(grid.sum()), 0.0)

    def test_environment_applies_one_action_for_configured_low_level_steps(self):
        config = load_ppo_lstm_config()
        config = {**config, "simulation": dict(config["simulation"])}
        config["simulation"]["episode_time_s"] = 10.0
        env = SawadaPpoLstmEnv(config, training=False)
        observation = env.reset(case_number=1, seed=42)
        self.assertEqual(observation.grid.shape, (60, 180))
        next_observation, reward, done, info = env.step(5.0)
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(done)
        self.assertEqual(info["low_level_steps"], 100)
        self.assertEqual(next_observation.numeric.shape, (10,))

    def test_evaluation_requires_an_explicit_checkpoint(self):
        from src.core.ppo_lstm_simulation import run_ppo_lstm

        args = parse_args(["--method", "ppo_lstm"])
        with tempfile.TemporaryDirectory() as temp_dir:
            args.output_dir = temp_dir
            with self.assertRaisesRegex(ValueError, "--ppo_checkpoint"):
                run_ppo_lstm(args)

    @unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
    def test_recurrent_ppo_collects_and_updates_sequences(self):
        from src.core.ppo_lstm_training import collect_rollout, update_agent
        from src.decision_making.ppo_lstm_policy import SawadaPpoLstmAgent

        config = load_ppo_lstm_config()
        config = {**config, "case_numbers": [1], "ppo": dict(config["ppo"])}
        config["ppo"].update(sequence_length=2, minibatch_sequences=2, epochs=1)
        env = SawadaPpoLstmEnv(config, training=True)
        agent = SawadaPpoLstmAgent(config, device="cpu")
        observation = env.reset(seed=42)
        state = agent.initial_state()
        rollout, _observation, _state, _episode, _summaries = collect_rollout(
            agent,
            env,
            observation,
            state,
            steps=4,
            seed=42,
            episode_index=0,
        )
        actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=3e-4)
        metrics = update_agent(agent, rollout, actor_optimizer, critic_optimizer)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
