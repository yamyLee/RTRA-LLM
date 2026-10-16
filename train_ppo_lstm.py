#!/usr/bin/env python3
"""Train the Sawada-style PPO-LSTM baseline in the unified RTRA environment."""

from __future__ import annotations

import argparse

from src.config.ppo_lstm import load_ppo_lstm_config
from src.core.ppo_lstm_training import train_ppo_lstm


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="PPO-LSTM JSON configuration")
    parser.add_argument("--output-dir", default="output/ppo_lstm/training/seed_42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-policy-steps", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or an explicit PyTorch device")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_ppo_lstm_config(args.config)
    checkpoint = train_ppo_lstm(
        config,
        args.output_dir,
        seed=args.seed,
        total_policy_steps=args.total_policy_steps,
        device=args.device,
    )
    print(f"Final checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
