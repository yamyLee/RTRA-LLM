"""On-policy recurrent PPO training for the Sawada-style baseline."""

from __future__ import annotations

import csv
import json
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.ppo_lstm_environment import SawadaPpoLstmEnv
from src.decision_making.ppo_lstm_policy import (
    SawadaPpoLstmAgent,
    hidden_state_from_numpy,
    hidden_state_to_numpy,
    require_torch,
    torch,
)


@dataclass
class Rollout:
    grid: np.ndarray
    numeric: np.ndarray
    raw_action: np.ndarray
    old_log_prob: np.ndarray
    value: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    advantage: np.ndarray
    returns: np.ndarray
    actor_h: np.ndarray
    actor_c: np.ndarray
    critic_h: np.ndarray
    critic_c: np.ndarray


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _branch_numpy(branch) -> tuple[np.ndarray, np.ndarray]:
    return tuple(item.detach().cpu().numpy() for item in branch)


def collect_rollout(
    agent: SawadaPpoLstmAgent,
    env: SawadaPpoLstmEnv,
    observation,
    recurrent_state,
    *,
    steps: int,
    seed: int,
    episode_index: int,
) -> tuple[Rollout, Any, Any, int, List[Dict[str, Any]]]:
    records: Dict[str, List[Any]] = {
        key: []
        for key in (
            "grid", "numeric", "raw_action", "old_log_prob", "value", "reward", "done",
            "actor_h", "actor_c", "critic_h", "critic_c",
        )
    }
    episode_summaries: List[Dict[str, Any]] = []

    for _ in range(steps):
        actor_before, critic_before = recurrent_state
        action, raw_action, log_prob, value, next_state = agent.act(
            observation, recurrent_state, deterministic=False
        )
        next_observation, reward, done, info = env.step(action)
        actor_h, actor_c = _branch_numpy(actor_before)
        critic_h, critic_c = _branch_numpy(critic_before)
        records["grid"].append(observation.grid)
        records["numeric"].append(observation.numeric)
        records["raw_action"].append(raw_action)
        records["old_log_prob"].append(log_prob)
        records["value"].append(value)
        records["reward"].append(reward)
        records["done"].append(done)
        records["actor_h"].append(actor_h)
        records["actor_c"].append(actor_c)
        records["critic_h"].append(critic_h)
        records["critic_c"].append(critic_c)

        observation = next_observation
        recurrent_state = next_state
        if done:
            summary = env.summary()
            episode_summaries.append(summary)
            episode_index += 1
            observation = env.reset(seed=seed + episode_index)
            recurrent_state = agent.initial_state()

    if records["done"][-1]:
        next_value = 0.0
    else:
        _action, _raw, _log_prob, next_value, _next_state = agent.act(
            observation, recurrent_state, deterministic=True
        )

    rewards = np.asarray(records["reward"], dtype=np.float32)
    values = np.asarray(records["value"], dtype=np.float32)
    dones = np.asarray(records["done"], dtype=np.float32)
    advantages = np.zeros_like(rewards)
    gamma = float(agent.config["ppo"]["gamma"])
    gae_lambda = float(agent.config["ppo"]["gae_lambda"])
    last_gae = 0.0
    for index in reversed(range(len(rewards))):
        if index == len(rewards) - 1:
            following_value = float(next_value)
        else:
            following_value = float(values[index + 1])
        nonterminal = 1.0 - float(dones[index])
        delta = float(rewards[index]) + gamma * following_value * nonterminal - float(values[index])
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[index] = last_gae
    returns = advantages + values

    rollout = Rollout(
        grid=np.asarray(records["grid"], dtype=np.float32),
        numeric=np.asarray(records["numeric"], dtype=np.float32),
        raw_action=np.asarray(records["raw_action"], dtype=np.float32),
        old_log_prob=np.asarray(records["old_log_prob"], dtype=np.float32),
        value=values,
        reward=rewards,
        done=dones,
        advantage=advantages,
        returns=returns,
        actor_h=np.concatenate(records["actor_h"], axis=1).transpose(1, 0, 2),
        actor_c=np.concatenate(records["actor_c"], axis=1).transpose(1, 0, 2),
        critic_h=np.concatenate(records["critic_h"], axis=1).transpose(1, 0, 2),
        critic_c=np.concatenate(records["critic_c"], axis=1).transpose(1, 0, 2),
    )
    return rollout, observation, recurrent_state, episode_index, episode_summaries


def _sequence_starts(rollout: Rollout, sequence_length: int) -> List[tuple[int, int]]:
    sequences = []
    start = 0
    total = len(rollout.reward)
    while start < total:
        end = min(start + sequence_length, total)
        done_positions = np.flatnonzero(rollout.done[start:end] > 0.5)
        if done_positions.size:
            end = start + int(done_positions[0]) + 1
        sequences.append((start, end))
        start = end
    return sequences


def _batch_sequences(rollout: Rollout, spans: List[tuple[int, int]], indices: np.ndarray, device):
    selected = [spans[int(index)] for index in indices]
    max_length = max(end - start for start, end in selected)
    batch = len(selected)
    grid = np.zeros((batch, max_length, *rollout.grid.shape[1:]), dtype=np.float32)
    numeric = np.zeros((batch, max_length, rollout.numeric.shape[1]), dtype=np.float32)
    raw_action = np.zeros((batch, max_length, 1), dtype=np.float32)
    old_log_prob = np.zeros((batch, max_length), dtype=np.float32)
    advantage = np.zeros((batch, max_length), dtype=np.float32)
    returns = np.zeros((batch, max_length), dtype=np.float32)
    mask = np.zeros((batch, max_length), dtype=np.float32)
    actor_h, actor_c, critic_h, critic_c = [], [], [], []
    for row, (start, end) in enumerate(selected):
        length = end - start
        grid[row, :length] = rollout.grid[start:end]
        numeric[row, :length] = rollout.numeric[start:end]
        raw_action[row, :length, 0] = rollout.raw_action[start:end]
        old_log_prob[row, :length] = rollout.old_log_prob[start:end]
        advantage[row, :length] = rollout.advantage[start:end]
        returns[row, :length] = rollout.returns[start:end]
        mask[row, :length] = 1.0
        actor_h.append(rollout.actor_h[start])
        actor_c.append(rollout.actor_c[start])
        critic_h.append(rollout.critic_h[start])
        critic_c.append(rollout.critic_c[start])

    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    actor_state = (
        tensor(np.stack(actor_h, axis=1)),
        tensor(np.stack(actor_c, axis=1)),
    )
    critic_state = (
        tensor(np.stack(critic_h, axis=1)),
        tensor(np.stack(critic_c, axis=1)),
    )
    return {
        "grid": tensor(grid),
        "numeric": tensor(numeric),
        "raw_action": tensor(raw_action),
        "old_log_prob": tensor(old_log_prob),
        "advantage": tensor(advantage),
        "returns": tensor(returns),
        "mask": tensor(mask),
        "actor_state": actor_state,
        "critic_state": critic_state,
    }


def update_agent(agent: SawadaPpoLstmAgent, rollout: Rollout, actor_optimizer, critic_optimizer) -> Dict[str, float]:
    ppo = agent.config["ppo"]
    valid_advantage = rollout.advantage
    rollout.advantage = (valid_advantage - valid_advantage.mean()) / (valid_advantage.std() + 1e-8)
    spans = _sequence_starts(rollout, int(ppo["sequence_length"]))
    minibatch = int(ppo["minibatch_sequences"])
    metrics = {"actor_loss": [], "critic_loss": [], "entropy": [], "clip_fraction": []}

    for _epoch in range(int(ppo["epochs"])):
        order = np.random.permutation(len(spans))
        for offset in range(0, len(order), minibatch):
            data = _batch_sequences(rollout, spans, order[offset : offset + minibatch], agent.device)
            mask = data["mask"]
            count = mask.sum().clamp_min(1.0)

            new_log_prob, entropy = agent.actor.evaluate(
                data["grid"], data["numeric"], data["raw_action"], data["actor_state"]
            )
            ratio = torch.exp(new_log_prob - data["old_log_prob"])
            unclipped = ratio * data["advantage"]
            clipped = torch.clamp(
                ratio,
                1.0 - float(ppo["clip_epsilon"]),
                1.0 + float(ppo["clip_epsilon"]),
            ) * data["advantage"]
            actor_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / count
            entropy_mean = (entropy * mask).sum() / count
            total_actor_loss = actor_loss - float(ppo["entropy_coefficient"]) * entropy_mean
            actor_optimizer.zero_grad(set_to_none=True)
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), float(ppo["gradient_norm_clip"]))
            actor_optimizer.step()

            predicted_value, _ = agent.critic(
                data["grid"], data["numeric"], data["critic_state"]
            )
            critic_loss = (((predicted_value - data["returns"]) ** 2) * mask).sum() / count
            critic_optimizer.zero_grad(set_to_none=True)
            (float(ppo["value_coefficient"]) * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), float(ppo["gradient_norm_clip"]))
            critic_optimizer.step()

            metrics["actor_loss"].append(float(actor_loss.detach().cpu()))
            metrics["critic_loss"].append(float(critic_loss.detach().cpu()))
            metrics["entropy"].append(float(entropy_mean.detach().cpu()))
            clipped_fraction = ((torch.abs(ratio - 1.0) > float(ppo["clip_epsilon"])).float() * mask).sum() / count
            metrics["clip_fraction"].append(float(clipped_fraction.detach().cpu()))

    return {key: float(np.mean(values)) for key, values in metrics.items()}


def evaluate_agent(agent: SawadaPpoLstmAgent, config: Dict[str, Any], seed: int) -> Dict[str, float]:
    summaries = []
    for case_number in config["case_numbers"]:
        env = SawadaPpoLstmEnv(config, training=False)
        observation = env.reset(case_number=case_number, seed=seed)
        state = agent.initial_state()
        while not env.done:
            action, _raw, _log_prob, _value, state = agent.act(
                observation, state, deterministic=True
            )
            observation, _reward, _done, _info = env.step(action)
        summaries.append(env.summary())
    return {
        "validation_R_avg": float(np.mean([row["R_avg"] for row in summaries])),
        "validation_R_max": float(np.mean([row["R_max"] for row in summaries])),
        "validation_min_dcpa_nm": float(np.mean([row["min_dcpa_nm"] for row in summaries])),
        "validation_collision_rate": float(np.mean([row["collision"] for row in summaries])),
    }


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_ppo_lstm(
    config: Dict[str, Any],
    output_dir: str | Path,
    *,
    seed: int = 42,
    total_policy_steps: Optional[int] = None,
    device: str = "auto",
) -> Path:
    """Train from scratch and return the final checkpoint path."""
    require_torch()
    _seed_everything(seed)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ppo = config["ppo"]
    total_steps = int(total_policy_steps or ppo["total_policy_steps"])
    agent = SawadaPpoLstmAgent(config, device=device)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=float(ppo["actor_learning_rate"]))
    critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=float(ppo["critic_learning_rate"]))
    env = SawadaPpoLstmEnv(config, training=True)
    episode_index = 0
    observation = env.reset(seed=seed)
    recurrent_state = agent.initial_state()
    completed_steps = 0
    started = time.perf_counter()
    best_validation = float("inf")
    from src.core.run_archive import source_manifest
    training_source = source_manifest()
    log_path = output_dir / "training_log.csv"
    episode_path = output_dir / "episode_log.csv"

    while completed_steps < total_steps:
        rollout_steps = min(int(ppo["rollout_steps"]), total_steps - completed_steps)
        rollout, observation, recurrent_state, episode_index, episode_summaries = collect_rollout(
            agent,
            env,
            observation,
            recurrent_state,
            steps=rollout_steps,
            seed=seed,
            episode_index=episode_index,
        )
        losses = update_agent(agent, rollout, actor_optimizer, critic_optimizer)
        previous_steps = completed_steps
        completed_steps += rollout_steps
        row: Dict[str, Any] = {
            "policy_steps": completed_steps,
            "low_level_steps": completed_steps * int(config["policy_interval_steps"]),
            "elapsed_s": time.perf_counter() - started,
            "mean_rollout_reward": float(np.mean(rollout.reward)),
            "mean_rollout_return": float(np.mean(rollout.returns)),
            **losses,
        }
        for summary in episode_summaries:
            _append_csv(episode_path, {"policy_steps": completed_steps, **summary})

        validation_interval = int(ppo["validation_interval_steps"])
        crossed_validation = completed_steps // validation_interval > previous_steps // validation_interval
        if crossed_validation or completed_steps == total_steps:
            validation = evaluate_agent(agent, config, seed=seed + 100000)
            row.update(validation)
            if validation["validation_R_avg"] < best_validation:
                best_validation = validation["validation_R_avg"]
                agent.save(
                    output_dir / "best.pt",
                    metadata={"source": training_source, "seed": seed, "policy_steps": completed_steps, **validation},
                )

        _append_csv(log_path, row)
        checkpoint_interval = int(ppo["checkpoint_interval_steps"])
        crossed_checkpoint = completed_steps // checkpoint_interval > previous_steps // checkpoint_interval
        if crossed_checkpoint or completed_steps == total_steps:
            agent.save(
                output_dir / f"checkpoint_{completed_steps:09d}.pt",
                metadata={"source": training_source, "seed": seed, "policy_steps": completed_steps, "elapsed_s": row["elapsed_s"]},
            )
        print(
            f"PPO-LSTM steps={completed_steps}/{total_steps} "
            f"reward={row['mean_rollout_reward']:.4f} "
            f"actor={row['actor_loss']:.4f} critic={row['critic_loss']:.4f}"
        )

    final_path = output_dir / "final.pt"
    metadata = {
        "source": training_source,
        "seed": seed,
        "policy_steps": completed_steps,
        "low_level_steps": completed_steps * int(config["policy_interval_steps"]),
        "training_time_s": time.perf_counter() - started,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(agent.device),
    }
    agent.save(final_path, metadata=metadata)
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final_path
