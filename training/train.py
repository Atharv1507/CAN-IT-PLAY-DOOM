"""Run from the repository root with python -m training.train."""
import argparse
from contextlib import closing
from functools import partial
import json
import os
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np
import torch

from training.ppo import ReferencePolicy, advantages, distributions, log_probability, tree_map, update

ROOT = Path(__file__).resolve().parents[1]
# Python on macOS can skip hidden editable-install path files.
sys.path.append(str(ROOT / "env"))

from doom_env import action_space, make_env, observation_space


def obs_to_stub(obs, device):
    sensors = {key: obs[key] for key in ("rays", "visible_entities", "vitals")}
    return tree_map(lambda array: torch.as_tensor(array, device=device), sensors)


def stub_buttons(move, turn, attack, use, weapon):
    moves = ((), (0,), (1,), (2,), (3,), (0, 2), (0, 3))
    buttons = np.zeros(9, dtype=np.int8)
    buttons[list(moves[move])] = 1
    if turn:
        buttons[3 + turn] = 1
    buttons[6:] = (attack, use, weapon)
    return buttons


class CurriculumEnv(gym.Env):
    def __init__(self, maps, pixels, exploration_bonus):
        self.maps = maps
        self.probabilities = np.array([entry.get("weight", 1) for entry in maps], dtype=float)
        self.probabilities /= self.probabilities.sum()
        self.pixels = pixels
        self.exploration_bonus = exploration_bonus
        self.observation_space = observation_space()
        self.action_space = action_space()
        self.env = None
        self.current = None

    def _info(self, info, state):
        radians = np.deg2rad(state["angle"])
        info["priv"] = np.array([
            state["x"] / 4096, state["y"] / 4096, np.sin(radians), np.cos(radians),
            state["health"] / 100, state["damage_taken"] / 100,
            state["kills"] / 10, state["items"] / 10, *state["keys"],
            state["tic"] / state["episode_ticks"], len(self.visited) / 100,
        ], dtype=np.float32)
        return info

    def _cell(self, state):
        return (int(state["x"] // 128), int(state["y"] // 128), int(state["z"] // 64))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = self.np_random.choice(len(self.maps), p=self.probabilities)
        entry = self.maps[index]
        if index != self.current:
            self.close()
            self.env = make_env(
                entry["wad"], entry.get("map", "MAP01"),
                episode_ticks=entry.get("ticks", 2100), render_buffers=self.pixels,
            )
            self.current = index
        obs, info = self.env.reset(seed=int(self.np_random.integers(2**31)))
        state = self.env.privileged_state()
        self.visited = {self._cell(state)}
        self.keys = int(np.count_nonzero(state["keys"]))
        return obs, self._info(info, state)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self.env.privileged_state()
        cell = self._cell(state)
        keys = int(np.count_nonzero(state["keys"]))
        terms = {"base": reward, "novelty": self.exploration_bonus * (cell not in self.visited),
                 "key": 0.5 * max(0, keys - self.keys)}
        reward = sum(terms.values())
        info["terms"] = terms
        self.visited.add(cell)
        self.keys = keys
        if terminated or truncated:
            entry = self.maps[self.current]
            info["episode_stats"] = {
                "kills": state["kills"], "keys": keys, "cells": len(self.visited),
                "exit": state["exit_reached"], "dead": state["dead"], "timeout": state["timeout"],
                "doors": 0, "unlocks": 0, "damage_taken": state["damage_taken"],
                "steps": (state["tic"] + 3) // 4,
                "map": Path(entry["wad"]).stem, "stage": 1,
            }
        return obs, reward, terminated, truncated, self._info(info, state)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.current = None


def checkpoint(path, model, optimizer, steps, config):
    payload = {
        "model": tree_map(lambda tensor: tensor.detach().cpu(), dict(model.state_dict())),
        "optimizer": tree_map(
            lambda item: item.detach().cpu() if torch.is_tensor(item) else item,
            optimizer.state_dict(),
        ),
        "step": steps,
        "cfg": config.get("cfg", {}),
        "stage_mix": config["stage_mix"],
        "reward_weights": config["reward_weights"],
        "run": config["run"],
        "smoke_test": config.get("smoke_test", False),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def benchmark(args, maps):
    for workers in (1, 4, 8):
        with closing(make_vector(args, maps, workers)) as env:
            env.reset(seed=args.seed)
            rng = np.random.default_rng(args.seed)
            actions = rng.integers(0, 2, (args.benchmark + 20, workers, 9), dtype=np.int8)
            for action in actions[:20]:
                env.step(action)
            start = time.perf_counter()
            for action in actions[20:]:
                env.step(action)
            elapsed = time.perf_counter() - start
            print(f"{workers} copies: {workers * args.benchmark / elapsed:.1f} decisions/s "
                  f"({elapsed:.2f}s measured)", flush=True)


def make_vector(args, maps, workers):
    if maps is not None:
        factories = [partial(CurriculumEnv, maps, False, 0.0)] * workers
    else:
        from training.envs import make_training_env

        factories = [partial(make_training_env, args.config["stage_mix"],
                             args.config["reward_weights"], args.seed + rank,
                             heldout=False) for rank in range(workers)]
    return gym.vector.AsyncVectorEnv(
        factories, context="spawn", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def train(args, maps, env, path):
    from torch.utils.tensorboard import SummaryWriter

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    config = args.config
    if maps is not None:
        model = ReferencePolicy(env.single_observation_space, env.single_action_space)
        obs_to_tensors, heads_to_buttons = obs_to_stub, stub_buttons
    else:
        from submission.model import DoomNet, heads_to_buttons, obs_to_tensors

        model = DoomNet(config.get("cfg", {}))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    steps = 0
    if args.resume or (path.exists() and args.init is None):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if any(saved[key] != config.get(key, {}) for key in ("cfg", "stage_mix", "reward_weights", "run")):
            raise ValueError("Checkpoint configuration differs; use --init in a new run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        steps = saved["step"]
        print(f"Resuming from {steps} decisions", flush=True)
    elif args.init:
        saved = torch.load(args.init, map_location="cpu", weights_only=True)
        if saved.get("smoke_test", False) != (maps is not None):
            raise ValueError("Smoke and team checkpoints are incompatible")
        if saved["cfg"] != config.get("cfg", {}):
            raise ValueError("Checkpoint model cfg differs from this run")
        model.load_state_dict(saved["model"])

    def tensor(array):
        return torch.as_tensor(array, device=device)

    def observe(obs):
        return obs_to_tensors(obs, device=device)

    obs, info = env.reset(seed=(args.seed + steps) % (2**31))
    state = tree_map(lambda x: x.to(device), model.initial_state(args.workers))
    episode_returns = np.zeros(args.workers)
    starts = torch.ones(args.workers, dtype=torch.bool, device=device)
    checkpoint_at = time.monotonic() + args.save_seconds
    began = time.monotonic()
    initial_steps = steps
    with SummaryWriter(str(args.log_dir), purge_step=steps or None) as writer:
        while args.steps is None or steps < args.steps:
            sequence_states = []
            term_totals = {}
            rollout = []
            episodes = []
            with torch.no_grad():
                for offset in range(args.horizon):
                    if offset % args.seq_len == 0:
                        sequence_states.append(tree_map(lambda value: value.detach().clone(), state))
                    observation = observe(obs)
                    privileged = tensor(info["priv"])
                    logits, values, state = model(observation, state, starts, privileged)
                    values = values.reshape(-1)
                    dists = distributions(logits)
                    action = torch.stack([dist.sample() for dist in dists], dim=-1)
                    buttons = np.stack([heads_to_buttons(*row.tolist()) for row in action.cpu().numpy()])
                    obs, reward, terminated, truncated, info = env.step(buttons)
                    done = terminated | truncated
                    episode_returns += reward
                    for source, mask in ((info, ~done), (info.get("final_info", {}), done)):
                        terms = source.get("terms", {})
                        for key, value in terms.items():
                            if not key.startswith("_"):
                                subtotal = np.asarray(value)[mask].sum()
                                term_totals[key] = term_totals.get(key, 0.0) + subtotal
                    # The official wrapper repeats its last sensor reading at episode end.
                    # Treat its tick cap as an absorbing horizon for this challenge.
                    rollout.append({
                        "obs": observation, "privileged": privileged, "starts": starts,
                        "values": values, "actions": action, "rewards": tensor((reward * args.reward_scale).astype(np.float32)),
                        "log_probs": log_probability(dists, action), "done": tensor(done),
                    })
                    starts = rollout[-1]["done"]
                    if done.any():
                        final = info["final_info"]["episode_stats"]
                        for index in np.flatnonzero(done):
                            stats = {key: values[index] for key, values in final.items()
                                     if not key.startswith("_")}
                            stats["episode_return"] = episode_returns[index]
                            episode_returns[index] = 0
                            episodes.append(stats)
                bootstrap_state = tree_map(lambda value: value.clone(), state)
                _, bootstrap, _ = model(observe(obs), bootstrap_state, starts, tensor(info["priv"]))
                bootstrap = bootstrap.reshape(-1)

            def stack(records):
                if isinstance(records[0], dict):
                    return {key: stack([record[key] for record in records]) for key in records[0]}
                return torch.stack(records)

            batches = stack(rollout)
            next_values = torch.cat((batches["values"][1:], bootstrap[None]))
            next_values = next_values.masked_fill(batches["done"], 0)
            batches["advantages"] = advantages(
                batches["rewards"], batches["values"], next_values, batches["done"],
            )
            policy_loss, value_loss, entropy = update(
                model, optimizer, batches, sequence_states, args.seq_len,
            )
            steps += args.horizon * args.workers
            for name, value in zip(("policy_loss", "value_loss", "entropy"),
                                   (policy_loss, value_loss, entropy), strict=True):
                writer.add_scalar(f"ppo/{name}", value, steps)
            for name, value in term_totals.items():
                writer.add_scalar(f"reward/{name}", value / (args.horizon * args.workers), steps)
            for stage in sorted({episode["stage"] for episode in episodes}):
                selected = [episode for episode in episodes if episode["stage"] == stage]
                for name in selected[0]:
                    if name not in {"map", "stage"}:
                        writer.add_scalar(f"stage_{stage}/{name}",
                                          np.mean([ep[name] for ep in selected]), steps)
            reached_steps = args.steps is not None and steps >= args.steps
            reached_time = args.seconds is not None and time.monotonic() - began >= args.seconds
            if time.monotonic() >= checkpoint_at or reached_steps or reached_time:
                hour = config.get("start_hour", 0) + int((time.monotonic() - began) // 3600)
                milestone = path.with_name(f"{config['run']}_h{hour}_{steps}.pt")
                checkpoint(milestone, model, optimizer, steps, config)
                # Publish the same immutable checkpoint under the automatic resume name.
                temporary = path.with_suffix(".link")
                temporary.unlink(missing_ok=True)
                os.link(milestone, temporary)
                temporary.replace(path)
                writer.flush()
                checkpoint_at = time.monotonic() + args.save_seconds
            rate = (steps - initial_steps) / (time.monotonic() - began)
            writer.add_scalar("performance/decisions_per_second", rate, steps)
            print(f"steps={steps} episodes={len(episodes)} value_loss={value_loss:.4f} "
                  f"decisions/s={rate:.1f}", flush=True)
            if reached_time:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    startup = parser.add_mutually_exclusive_group()
    startup.add_argument("--init", type=Path)
    startup.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", choices=("two_rooms", "defend_the_center"))
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--benchmark", type=int, metavar="VECTOR_STEPS")
    parser.add_argument("--output", type=Path, help="override the checkpoint and log directory")
    args = parser.parse_args()
    if args.config is None and args.smoke is None and args.benchmark is None:
        parser.error("--config is required for team training")
    config = json.loads(args.config.read_text()) if args.config else {
        "run": "smoke", "seed": 1, "num_envs": 6, "device": "cpu", "stage_mix": {1: 1},
        "reward_weights": {}, "lr": 2.5e-4, "rollout_len": 128, "seq_len": 32,
        "hours": 0.5,
    }
    config["stage_mix"] = {int(stage): weight for stage, weight in config["stage_mix"].items()}
    args.config = config
    args.seed = config["seed"]
    args.device = args.device or config["device"]
    args.workers = args.workers if args.workers is not None else config["num_envs"]
    args.horizon, args.seq_len = config["rollout_len"], config["seq_len"]
    args.seconds = args.seconds if args.seconds is not None else config["hours"] * 3600
    args.save_seconds = config.get("save_seconds", 1800)
    # PPO learns from reward * reward_scale (TensorBoard still shows unscaled rewards). With exit = 40
    # the critic's squared error dominated the shared network's update; 0.1 keeps it in the old range.
    args.reward_scale = float(config.get("reward_scale", 1.0))
    if not 1 <= args.workers <= 8:
        parser.error("Worker count must be in 1..8 on Laptop A")
    counts = (args.horizon, args.seq_len, args.save_seconds, args.steps, args.seconds, args.benchmark)
    if any(value is not None and (not np.isfinite(value) or value <= 0) for value in counts):
        parser.error("Training counts and durations must be positive")
    if args.horizon % args.seq_len:
        parser.error("rollout_len must be divisible by seq_len")
    maps = None
    if args.smoke == "defend_the_center":
        import vizdoom

        maps = [{"wad": str(Path(vizdoom.scenarios_path) / "defend_the_center.wad")}]
    elif args.smoke or args.benchmark:
        maps = [{"wad": str(ROOT / "maps" / "two_rooms.wad")}]
    config["smoke_test"] = maps is not None
    if args.benchmark is not None:
        benchmark(args, maps)
        return
    if maps is None:
        missing = [str(path.relative_to(ROOT)) for path in (
            ROOT / "training/envs.py", ROOT / "training/maps/maps.csv", ROOT / "submission/model.py",
        ) if not path.is_file()]
        if missing:
            parser.error(f"Team integration files are missing: {', '.join(missing)}. Use --smoke for a stub run.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("PyTorch MPS is unavailable in this session")

    import fcntl

    label = args.config["run"] + (f"_h{config['start_hour']}" if "start_hour" in config else "")
    checkpoint_dir = args.output or ROOT / "checkpoints" / label
    args.log_dir = args.output or ROOT / "runs" / label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "latest.pt"
    with (checkpoint_dir / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another process is writing to this run directory")
        if args.init and path.exists():
            parser.error("Use a new run directory for --init")
        if args.resume and not path.is_file():
            parser.error(f"No checkpoint to resume at {path}")
        with closing(make_vector(args, maps, args.workers)) as env:
            train(args, maps, env, path)


if __name__ == "__main__":
    main()
