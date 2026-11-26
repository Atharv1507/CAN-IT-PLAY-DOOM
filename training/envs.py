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

ENEMY_TYPES = {i for i, name in enumerate(S.ENTITY_TYPES) if name.startswith("enemy")}
KEY_THING_LOCK = {13: 129, 38: 129, 5: 130, 40: 130, 6: 131, 39: 131}  # thing type -> ZDoom lock it opens
_GOALS_CACHE: dict = {}


def map_goals(path: str, map_name: str, any_path: bool = False) -> dict | None:
    """Key / locked-door / exit positions of one of OUR generated UDMF maps (None for other maps).
    Training only: used by the "progress" reward term. any_path=True is for tests with temp maps."""
    p = Path(path).resolve()
    if not any_path and (REPO / "training" / "maps" / "wads") not in p.parents:
        return None
    k = (str(p), map_name)
    if k not in _GOALS_CACHE:
        from doom_env.mapinfo import parse_udmf
        from doom_env.wad import map_lumps, read_wad

        b = parse_udmf(map_lumps(read_wad(p), map_name)["TEXTMAP"].decode())
        verts = [(float(v.get("x", 0)), float(v.get("y", 0))) for v in b.get("vertex", [])]
        sides = b.get("sidedef", [])
        mid = lambda ln: ((verts[ln["v1"]][0] + verts[ln["v2"]][0]) / 2, (verts[ln["v1"]][1] + verts[ln["v2"]][1]) / 2)
        keys = [(float(t["x"]), float(t["y"]), KEY_THING_LOCK[t["type"]]) for t in b.get("thing", [])
                if t.get("type") in KEY_THING_LOCK]
        doors: dict = {}
        exits = []
        for ln in b.get("linedef", []):
            sp = int(ln.get("special", 0) or 0)
            if sp == 13:
                d = doors.setdefault(int(ln.get("arg3", 0)), {"pts": [], "sectors": set()})
                d["pts"].append(mid(ln))
                if "sideback" in ln:
                    d["sectors"].add(int(sides[ln["sideback"]]["sector"]))
            elif sp == 243:
                exits.append(mid(ln))
        for d in doors.values():
            d["pos"] = tuple(np.mean(d.pop("pts"), axis=0))
        exit_lines = list(exits)
        if exits:  # one exit: an exit pad is surrounded by 4 exit lines, a switch has 1-3 faces
            exits = [tuple(np.mean(exits, axis=0))]
        goals = {"keys": keys, "doors": doors, "exits": exits, "_exit_lines": exit_lines}
        try:
            _add_walk_distances(goals, b, verts)
        except Exception:  # noqa: BLE001 - fall back to straight-line distance
            goals["grid"] = None
        _GOALS_CACHE[k] = goals
    return _GOALS_CACHE[k]


CELL = 128  # our generator's grid size (training/maps/udmf_gen.py)


def _add_walk_distances(goals: dict, b: dict, verts: list) -> None:
    """Walking distance (in 128-unit cells, around walls, through doors) from every cell to each goal, for the
    geodesic "progress" reward. Training-time scaffolding only - the agent never sees it."""
    xs, ys = [v[0] for v in verts], [v[1] for v in verts]
    minx, maxy = min(xs), max(ys)
    ncols, nrows = int(round((max(xs) - minx) / CELL)), int(round((maxy - min(ys)) / CELL))
    blocked = set()  # frozenset({cell_a, cell_b}) edges crossed by a one-sided (solid) line
    for ln in b.get("linedef", []):
        if "sideback" in ln:
            continue
        (x1, y1), (x2, y2) = verts[ln["v1"]], verts[ln["v2"]]
        if x1 == x2:
            c = int(round((x1 - minx) / CELL))
            for r in range(int(round((maxy - max(y1, y2)) / CELL)), int(round((maxy - min(y1, y2)) / CELL))):
                blocked.add(frozenset({(r, c - 1), (r, c)}))
        else:
            r = int(round((maxy - y1) / CELL))
            for c in range(int(round((min(x1, x2) - minx) / CELL)), int(round((max(x1, x2) - minx) / CELL))):
                blocked.add(frozenset({(r - 1, c), (r, c)}))
    cell_of = lambda x, y: (min(nrows - 1, max(0, int((maxy - y) // CELL))), min(ncols - 1, max(0, int((x - minx) // CELL))))
    start = [t for t in b.get("thing", []) if t.get("type") == 1]
    if not start:
        raise ValueError("no player start")
    walk = np.zeros((nrows, ncols), bool)
    todo = [cell_of(float(start[0]["x"]), float(start[0]["y"]))]
    walk[todo[0]] = True
    nbrs = lambda r, c: [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                         if 0 <= r + dr < nrows and 0 <= c + dc < ncols and frozenset({(r, c), (r + dr, c + dc)}) not in blocked]
    while todo:
        cur = todo.pop()
        for n in nbrs(*cur):
            if not walk[n]:
                walk[n] = True
                todo.append(n)

    def field(*points):
        cands = [cell_of(x + dx, y + dy) for x, y in points for dx in (-1, 1) for dy in (-1, 1)]  # edge point -> walkable side
        cands = [c for c in dict.fromkeys(cands) if walk[c]]
        if not cands:
            return None
        dist = np.full((nrows, ncols), -1, np.int32)
        queue = list(cands)
        for c in cands:
            dist[c] = 0
        for cur in queue:  # BFS (list grows while iterating)
            for n in nbrs(*cur):
                if walk[n] and dist[n] < 0:
                    dist[n] = dist[cur] + 1
                    queue.append(n)
        return dist

    goals["grid"] = (minx, maxy, nrows, ncols)
    goals["key_fields"] = [field((x, y)) for x, y, _ in goals["keys"]]
    for d in goals["doors"].values():
        d["field"] = field(d["pos"])
    goals["exit_fields"] = [field(*goals["_exit_lines"])] if goals["exits"] else []  # every exit line is a target

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
                 heldout: bool = False, maps: list[MapRow] | None = None, render_buffers: bool = False,
                 goals_any_path: bool = False):
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
        self.goals_any_path = goals_any_path
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
        self.shaper.reset(priv, map_goals(row.path, row.map, self.goals_any_path) if self.shaper.w["progress"] else None)
        self._obs = obs
        return obs, {"priv": priv_vector(priv, self.shaper.cells_visited), "map": row.name, "stage": row.stage}

    def step(self, action):
        ents = self._obs["visible_entities"]
        enemy_seen = any(int(t) in ENEMY_TYPES for t, m in zip(ents["entity_type"], ents["mask"]) if m)
        blind = bool(np.asarray(action)[6]) and not enemy_seen  # ATTACK with no monster in the view it acted on
        obs, _default_reward, terminated, truncated, _ = self._env.step(action)
        self._obs = obs
        priv = self._env.privileged_state()
        reward, terms = self.shaper.step(priv, blind_fire=blind)
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
