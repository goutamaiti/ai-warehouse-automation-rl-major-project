"""Train a PPO policy on the warehouse environment.

Example::

    python -m rl_agent.train --scenario default --timesteps 500000

Outputs (all under paths from :class:`rl_agent.config.PPOConfig`):

* ``rl_agent/models/<tag>.zip``        - final policy
* ``rl_agent/models/<tag>_best/``      - best policy found by periodic evaluation
* ``data/logs/<tag>/``                 - Monitor CSVs and TensorBoard traces
* ``data/logs/<tag>/run_metadata.json``- config, versions and wall-clock time

The metadata file exists so that every reported training curve can be traced
back to the exact configuration that produced it.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from functools import partial
from pathlib import Path

from environment import make_env

from .config import PPOConfig


def _import_sb3():
    """Import Stable-Baselines3 with an actionable error message."""
    try:
        import stable_baselines3 as sb3
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "stable-baselines3 (and torch) are required for training.\n"
            "Install them with:  pip install -r requirements.txt"
        ) from exc
    return sb3, PPO, make_vec_env, Monitor, DummyVecEnv, EvalCallback, CheckpointCallback


def build_config(args: argparse.Namespace) -> PPOConfig:
    config = PPOConfig(
        scenario=args.scenario,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        eval_freq=args.eval_freq,
    )
    config.validate()
    return config


def train(config: PPOConfig, tag: str = "ppo_warehouse") -> Path:
    """Run PPO training and return the path of the saved model."""
    sb3, PPO, make_vec_env, Monitor, DummyVecEnv, EvalCallback, CheckpointCallback = _import_sb3()

    log_dir = Path(config.log_dir) / tag
    model_dir = Path(config.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env_fn = partial(make_env, config.scenario)
    train_env = make_vec_env(
        env_fn,
        n_envs=config.n_envs,
        seed=config.seed,
        vec_env_cls=DummyVecEnv,
        monitor_dir=str(log_dir / "monitor"),
    )
    # A separate environment for evaluation: mixing evaluation episodes into the
    # training rollouts would bias both the policy and the reported score.
    eval_env = Monitor(env_fn(), filename=str(log_dir / "eval_monitor"))

    # Ten checkpoints per run is enough to recover from a crash without filling
    # the disk with hundreds of copies of the network.
    checkpoint_every = max(config.total_timesteps // (10 * max(config.n_envs, 1)), 1)
    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_every,
            save_path=str(model_dir / f"{tag}_checkpoints"),
            name_prefix=tag,
        )
    ]
    if config.eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(model_dir / f"{tag}_best"),
                log_path=str(log_dir / "eval"),
                eval_freq=max(config.eval_freq // max(config.n_envs, 1), 1),
                n_eval_episodes=config.n_eval_episodes,
                deterministic=True,
                render=False,
            )
        )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=config.policy_kwargs(),
        seed=config.seed,
        tensorboard_log=str(log_dir / "tensorboard"),
        verbose=1,
    )

    started = time.time()
    model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
    duration = time.time() - started

    model_path = config.model_path(tag)
    model.save(model_path)

    metadata = {
        "tag": tag,
        "config": config.to_dict(),
        "wall_clock_seconds": round(duration, 1),
        "stable_baselines3": sb3.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model_path": str(model_path),
    }
    (log_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    train_env.close()
    eval_env.close()
    print(f"\nsaved model to {model_path}")
    print(f"training metadata written to {log_dir / 'run_metadata.json'}")
    return model_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = PPOConfig()
    parser = argparse.ArgumentParser(description="Train PPO on the warehouse environment")
    parser.add_argument("--scenario", default=defaults.scenario)
    parser.add_argument("--timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--n-envs", type=int, default=defaults.n_envs)
    parser.add_argument("--n-steps", type=int, default=defaults.n_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--eval-freq", type=int, default=defaults.eval_freq)
    parser.add_argument("--tag", default="ppo_warehouse", help="name for model and log files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train(build_config(args), tag=args.tag)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
