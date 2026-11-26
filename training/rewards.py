"""Reward shaping (Person 2). Turns ground truth from env.privileged_state() into a reward that copies
what the official scorer pays for (docs/EVALUATION.md): exit > progression > exploration > combat.

Training only: privileged_state() does not exist at evaluation, and agent.py never uses this file.
Interface frozen in PLAN.md section 4.4.
"""

from __future__ import annotations

import math

import numpy as np

CELL = 64  # exploration grid: one "area" = 64x64 map units
DOOR_OPEN_GAP = 56  # a door counts as open once ceiling - floor >= player height
PRIV_DIM = 16

DEFAULT_WEIGHTS = {
    "exit": 10.0,  # reaching the exit (biggest scorer category)
    "exit_bonus": 2.0,  # up to +2 more for remaining health and time
    "key": 3.0,  # each key picked up
    "door": 0.5,  # first time each normal door opens
    "unlock": 3.0,  # first time each locked door opens
    "cell": 0.1,  # first visit to each 64x64 area
    "kill": 1.0,  # each monster killed
    "damage": 0.01,  # per HP of damage taken (penalty)
    "death": 5.0,  # dying (penalty)
    "time": 0.001,  # per step (penalty, so it hurries)
}
TERMS = ("exit", "key", "door", "unlock", "cell", "kill", "damage", "death", "time")


class RewardShaper:
    """Call reset(priv) after env.reset(), then step(priv) after every env.step()."""

    def __init__(self, weights: dict | None = None):
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}

    def reset(self, priv: dict) -> None:
        self.kills = priv["kills"]
        self.damage = priv["damage_taken"]
        self.keys = float(np.sum(priv["keys"]))
        self.cells = {self._cell(priv)}
        lock = priv["door_lock"]
        gap = self._gap(priv)
        # only doors that start closed can be "opened"
        self.closed_doors = set(np.flatnonzero((lock >= 0) & (gap < DOOR_OPEN_GAP)).tolist()) if lock is not None else set()
        self.lock = lock
        self.stats = {"exit": 0, "dead": 0, "timeout": 0, "kills": 0, "keys": 0, "doors": 0, "unlocks": 0,
                      "cells": 1, "damage_taken": 0.0, "steps": 0, "return": 0.0}

    def step(self, priv: dict) -> tuple[float, dict]:
        w = self.w
        t = dict.fromkeys(TERMS, 0.0)

        if priv["exit_reached"]:
            frac_left = 1.0 - priv["tic"] / max(1, priv["episode_ticks"])
            t["exit"] = w["exit"] + w["exit_bonus"] * (0.5 * max(0.0, priv["health"]) / 100.0 + 0.5 * frac_left)
            self.stats["exit"] = 1

        keys = float(np.sum(priv["keys"]))
        if keys > self.keys:
            t["key"] = w["key"] * (keys - self.keys)
            self.stats["keys"] += int(keys - self.keys)
        self.keys = keys

        if self.closed_doors:
            gap = self._gap(priv)
            opened = [s for s in self.closed_doors if gap[s] >= DOOR_OPEN_GAP]
            for s in opened:
                self.closed_doors.discard(s)
                if self.lock[s] > 0:
                    t["unlock"] += w["unlock"]
                    self.stats["unlocks"] += 1
                else:
                    t["door"] += w["door"]
                    self.stats["doors"] += 1

        cell = self._cell(priv)
        if cell not in self.cells:
            self.cells.add(cell)
            t["cell"] = w["cell"]
            self.stats["cells"] += 1

        kills = priv["kills"]
        if kills > self.kills:
            t["kill"] = w["kill"] * (kills - self.kills)
            self.stats["kills"] += kills - self.kills
        self.kills = kills

        dmg = priv["damage_taken"]
        if dmg > self.damage:
            t["damage"] = -w["damage"] * (dmg - self.damage)
            self.stats["damage_taken"] += dmg - self.damage
        self.damage = dmg

        if priv["dead"]:
            t["death"] = -w["death"]
            self.stats["dead"] = 1
        if priv["timeout"]:
            self.stats["timeout"] = 1
        t["time"] = -w["time"]

        r = float(sum(t.values()))
        self.stats["steps"] += 1
        self.stats["return"] += r
        return r, t

    @property
    def cells_visited(self) -> int:
        return len(self.cells)

    @staticmethod
    def _cell(priv: dict) -> tuple[int, int]:
        return int(math.floor(priv["x"] / CELL)), int(math.floor(priv["y"] / CELL))

    @staticmethod
    def _gap(priv: dict) -> np.ndarray:
        if priv["sector_ceiling"] is None:
            return np.zeros(0)
        return priv["sector_ceiling"] - priv["sector_floor"]


def priv_vector(priv: dict, cells_visited: int) -> np.ndarray:
    """16 ground-truth numbers for the privileged critic (training only). Order frozen in PLAN.md 4.4."""
    a = math.radians(priv["angle"])
    v = [priv["x"] / 4096.0, priv["y"] / 4096.0, math.sin(a), math.cos(a), priv["health"] / 100.0,
         priv["damage_taken"] / 100.0, priv["kills"] / 10.0, priv["items"] / 10.0, *np.asarray(priv["keys"], np.float32),
         priv["tic"] / max(1, priv["episode_ticks"]), cells_visited / 100.0]
    out = np.asarray(v, np.float32)
    assert out.shape == (PRIV_DIM,)
    return out
