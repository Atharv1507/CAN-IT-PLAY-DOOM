"""Training environment (Person 2): official doom_env + map sampling by curriculum stage + our reward.

Interface frozen in PLAN.md section 4.3:

    env = make_training_env(stage_mix={1: 0.5, 2: 0.5}, reward_weights={}, seed=0)
    obs, info = env.reset()            # obs = official observation dict (same as at evaluation)
    obs, r, term, trunc, info = env.step(buttons)   # r = shaped reward
    info["priv"]          (16,) float32 for the privileged critic (training only)
    info["terms"]         each reward term this step (for TensorBoard)
    info["episode_stats"] on the final step: exit, dead, timeout, kills, keys, doors, unlocks, cells, ...

For gymnasium.vector.AsyncVectorEnv pass a factory: lambda: make_training_env(...). Note that
gymnasium >= 1.0 vector envs auto-reset on the step AFTER an episode ends (that step's action is ignored).
"""

from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import vizdoom

from doom_env import make_env
from doom_env import spaces as S
from doom_env.wrapper import IWAD

from training.rewards import RewardShaper, priv_vector

REPO = Path(__file__).resolve().parent.parent
MAPS_CSV = REPO / "training" / "maps" / "maps.csv"


@dataclass(frozen=True)
class MapRow:
    wad: str  # as written in maps.csv
    map: str
    stage: int
    episode_ticks: int
    heldout: bool

    @property
    def path(self) -> str:
        return resolve_wad(self.wad)

    @property
    def name(self) -> str:
        return f"{Path(self.wad).name}:{self.map}"


def resolve_wad(wad: str) -> str:
    """'@freedoom2' -> bundled IWAD, '@vizdoom/<file>' -> ViZDoom scenario, else path relative to repo root."""
    if wad == "@freedoom2":
        return IWAD
    if wad.startswith("@vizdoom/"):
        return os.path.join(vizdoom.scenarios_path, wad.split("/", 1)[1])
    p = Path(wad)
    return str(p if p.is_absolute() else REPO / p)


def load_maps(csv_path: str | Path = MAPS_CSV) -> list[MapRow]:
    with open(csv_path, newline="") as f:
        return [MapRow(r["wad"], r["map"], int(r["stage"]), int(r["episode_ticks"]), r["heldout"].strip() == "1")
                for r in csv.DictReader(f)]


class TrainingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, stage_mix: dict, reward_weights: dict | None = None, seed: int = 0,
                 heldout: bool = False, maps: list[MapRow] | None = None, render_buffers: bool = False):
        rows = maps if maps is not None else [m for m in load_maps() if m.heldout == heldout]
        mix = {int(k): float(v) for k, v in stage_mix.items() if float(v) > 0}
        self.by_stage = {s: [m for m in rows if m.stage == s] for s in mix}
        missing = [s for s, ms in self.by_stage.items() if not ms]
        if missing:
            raise ValueError(f"no maps for stage(s) {missing} (heldout={heldout})")
        self.stages = list(mix)
        p = np.array([mix[s] for s in self.stages])
        self.stage_p = p / p.sum()
        self.observation_space = S.observation_space()
        self.action_space = S.action_space()
        self.rng = np.random.default_rng(seed)
        self.shaper = RewardShaper(reward_weights)
        self.render_buffers = render_buffers
        self._env = None
        self._row: MapRow | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        row = (options or {}).get("map")
        if row is None:
            stage = self.stages[self.rng.choice(len(self.stages), p=self.stage_p)]
            maps = self.by_stage[stage]
            row = maps[self.rng.integers(len(maps))]
        if row != self._row or self._env is None:
            if self._env is not None:
                self._env.close()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._env = make_env(row.path, row.map, episode_ticks=row.episode_ticks,
                                     render_buffers=self.render_buffers)
            self._row = row
        obs, _ = self._env.reset(seed=int(self.rng.integers(2**31)),
                                 options={"record_path": options["record_path"]} if options and "record_path" in options else None)
        priv = self._env.privileged_state()
        self.shaper.reset(priv)
        return obs, {"priv": priv_vector(priv, self.shaper.cells_visited), "map": row.name, "stage": row.stage}

    def step(self, action):
        obs, _default_reward, terminated, truncated, _ = self._env.step(action)
        priv = self._env.privileged_state()
        reward, terms = self.shaper.step(priv)
        info = {"priv": priv_vector(priv, self.shaper.cells_visited), "terms": terms}
        if terminated or truncated:
            info["episode_stats"] = {**self.shaper.stats, "map": self._row.name, "stage": self._row.stage}
        return obs, reward, terminated, truncated, info

    def privileged_state(self) -> dict:
        return self._env.privileged_state()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def make_training_env(stage_mix: dict, reward_weights: dict | None = None, seed: int = 0,
                      heldout: bool = False, **kwargs) -> TrainingEnv:
    return TrainingEnv(stage_mix, reward_weights, seed, heldout, **kwargs)
