"""Recurrent actor-critic networks for the Sawada-style PPO-LSTM baseline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # Keep the non-RL simulator usable without the optional training stack.
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError:  # pragma: no cover - exercised through require_torch
    torch = None
    nn = None
    Normal = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PPO-LSTM requires PyTorch. Install it with "
            "`python -m pip install -r requirements-ppo-lstm.txt`."
        )


if nn is not None:

    class _GridNumericLstm(nn.Module):
        def __init__(
            self,
            grid_shape: Tuple[int, int],
            numeric_dim: int,
            conv_channels: list[int],
            network: Dict[str, Any],
        ) -> None:
            super().__init__()
            layers = []
            in_channels = 1
            for index, out_channels in enumerate(conv_channels):
                kernel = 8 if index == 0 else 4
                stride = 4 if index == 0 else 2
                padding = 2 if index == 0 else 1
                layers.extend(
                    [
                        nn.Conv2d(in_channels, int(out_channels), kernel, stride, padding),
                        nn.ReLU(),
                    ]
                )
                in_channels = int(out_channels)
            self.grid_encoder = nn.Sequential(*layers)
            with torch.no_grad():
                probe = torch.zeros(1, 1, *grid_shape)
                conv_size = int(self.grid_encoder(probe).numel())
            numeric_hidden = int(network["numeric_hidden_size"])
            lstm_input = int(network["lstm_input_size"])
            self.numeric_encoder = nn.Sequential(
                nn.Linear(numeric_dim, numeric_hidden),
                nn.ReLU(),
            )
            self.merge = nn.Sequential(
                nn.Linear(conv_size + numeric_hidden, lstm_input),
                nn.ReLU(),
            )
            self.lstm = nn.LSTM(lstm_input, int(network["lstm_hidden_size"]), batch_first=True)
            self.output = nn.Sequential(
                nn.Linear(int(network["lstm_hidden_size"]), int(network["merged_hidden_size"])),
                nn.ReLU(),
            )

        def initial_state(self, batch_size: int, device: "torch.device") -> Tuple["torch.Tensor", "torch.Tensor"]:
            hidden = self.lstm.hidden_size
            return (
                torch.zeros(1, batch_size, hidden, device=device),
                torch.zeros(1, batch_size, hidden, device=device),
            )

        def forward(
            self,
            grid: "torch.Tensor",
            numeric: "torch.Tensor",
            state: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,
        ) -> Tuple["torch.Tensor", Tuple["torch.Tensor", "torch.Tensor"]]:
            # Accept BxHxW, BxTxHxW, or BxTx1xHxW.
            if grid.ndim == 3:
                grid = grid[:, None, None, :, :]
                numeric = numeric[:, None, :]
            elif grid.ndim == 4:
                grid = grid[:, :, None, :, :]
            batch, steps = grid.shape[:2]
            flat_grid = grid.reshape(batch * steps, 1, grid.shape[-2], grid.shape[-1])
            flat_numeric = numeric.reshape(batch * steps, numeric.shape[-1])
            encoded_grid = self.grid_encoder(flat_grid).reshape(batch * steps, -1)
            encoded_numeric = self.numeric_encoder(flat_numeric)
            merged = self.merge(torch.cat((encoded_grid, encoded_numeric), dim=-1))
            merged = merged.reshape(batch, steps, -1)
            recurrent, new_state = self.lstm(merged, state)
            return self.output(recurrent), new_state


    class SawadaActor(nn.Module):
        def __init__(self, config: Dict[str, Any]) -> None:
            super().__init__()
            network = config["network"]
            self.body = _GridNumericLstm(
                tuple(config["grid_shape"]),
                int(config["observation"]["numeric_state_dim"]),
                list(network["actor_conv_channels"]),
                network,
            )
            self.mean = nn.Linear(int(network["merged_hidden_size"]), 1)
            self.log_std = nn.Parameter(torch.tensor([float(network["initial_log_std"])], dtype=torch.float32))
            self.action_limit = float(config["action"]["maximum_heading_increment_deg"])

        def initial_state(self, batch_size: int, device: "torch.device"):
            return self.body.initial_state(batch_size, device)

        def distribution(self, grid, numeric, state=None):
            features, new_state = self.body(grid, numeric, state)
            mean = self.mean(features)
            std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
            return Normal(mean, std), new_state

        @staticmethod
        def _squashed_log_prob(distribution, raw_action):
            squashed = torch.tanh(raw_action)
            correction = torch.log(1.0 - squashed.pow(2) + 1e-6)
            return distribution.log_prob(raw_action).sum(-1) - correction.sum(-1)

        def act(self, grid, numeric, state=None, deterministic: bool = False):
            distribution, new_state = self.distribution(grid, numeric, state)
            raw_action = distribution.mean if deterministic else distribution.rsample()
            action = torch.tanh(raw_action) * self.action_limit
            log_prob = self._squashed_log_prob(distribution, raw_action)
            return action, raw_action, log_prob, new_state

        def evaluate(self, grid, numeric, raw_action, state=None):
            distribution, _new_state = self.distribution(grid, numeric, state)
            log_prob = self._squashed_log_prob(distribution, raw_action)
            entropy = distribution.entropy().sum(-1)
            return log_prob, entropy


    class SawadaCritic(nn.Module):
        def __init__(self, config: Dict[str, Any]) -> None:
            super().__init__()
            network = config["network"]
            self.body = _GridNumericLstm(
                tuple(config["grid_shape"]),
                int(config["observation"]["numeric_state_dim"]),
                list(network["critic_conv_channels"]),
                network,
            )
            self.value = nn.Linear(int(network["merged_hidden_size"]), 1)

        def initial_state(self, batch_size: int, device: "torch.device"):
            return self.body.initial_state(batch_size, device)

        def forward(self, grid, numeric, state=None):
            features, new_state = self.body(grid, numeric, state)
            return self.value(features).squeeze(-1), new_state


    class SawadaPpoLstmAgent:
        def __init__(self, config: Dict[str, Any], device: str = "auto") -> None:
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            self.device = torch.device(device)
            self.config = config
            self.actor = SawadaActor(config).to(self.device)
            self.critic = SawadaCritic(config).to(self.device)

        def initial_state(self, batch_size: int = 1):
            return (
                self.actor.initial_state(batch_size, self.device),
                self.critic.initial_state(batch_size, self.device),
            )

        def observation_tensors(self, observation):
            grid = torch.as_tensor(observation.grid, dtype=torch.float32, device=self.device).unsqueeze(0)
            numeric = torch.as_tensor(observation.numeric, dtype=torch.float32, device=self.device).unsqueeze(0)
            return grid, numeric

        def act(self, observation, state=None, deterministic: bool = False):
            if state is None:
                state = self.initial_state()
            actor_state, critic_state = state
            grid, numeric = self.observation_tensors(observation)
            with torch.no_grad():
                action, raw_action, log_prob, actor_state = self.actor.act(
                    grid, numeric, actor_state, deterministic=deterministic
                )
                value, critic_state = self.critic(grid, numeric, critic_state)
            next_state = (
                tuple(item.detach() for item in actor_state),
                tuple(item.detach() for item in critic_state),
            )
            return (
                float(action.reshape(-1)[0].cpu()),
                float(raw_action.reshape(-1)[0].cpu()),
                float(log_prob.reshape(-1)[0].cpu()),
                float(value.reshape(-1)[0].cpu()),
                next_state,
            )

        def save(self, path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            serializable_config = dict(self.config)
            serializable_config.pop("config_path", None)
            torch.save(
                {
                    "format": "rtra_sawada_ppo_lstm_v1",
                    "config": serializable_config,
                    "actor_state_dict": self.actor.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "metadata": metadata or {},
                },
                path,
            )

        @classmethod
        def load(cls, path: str | Path, device: str = "auto"):
            checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
            if checkpoint.get("format") != "rtra_sawada_ppo_lstm_v1":
                raise ValueError("Unsupported PPO-LSTM checkpoint format")
            config = checkpoint["config"]
            config["grid_shape"] = tuple(config["grid_shape"])
            agent = cls(config, device=device)
            agent.actor.load_state_dict(checkpoint["actor_state_dict"])
            agent.critic.load_state_dict(checkpoint["critic_state_dict"])
            agent.actor.eval()
            agent.critic.eval()
            return agent, checkpoint.get("metadata", {})

else:

    class SawadaPpoLstmAgent:  # pragma: no cover - simple missing-dependency guard
        def __init__(self, *_args, **_kwargs):
            require_torch()

        @classmethod
        def load(cls, *_args, **_kwargs):
            require_torch()


def hidden_state_to_numpy(state):
    """Detach nested actor/critic LSTM states for rollout storage."""
    require_torch()
    return tuple(tuple(item.detach().cpu().numpy() for item in branch) for branch in state)


def hidden_state_from_numpy(state, device):
    require_torch()
    return tuple(torch.as_tensor(item, dtype=torch.float32, device=device) for item in state)
