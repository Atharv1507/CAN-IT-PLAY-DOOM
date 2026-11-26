"""Builds the whole map pool (Person 2): generated .wad files + training/maps/maps.csv.

    python -m training.maps.build_pool            # (re)generate everything, deterministic
    python -m training.maps.build_pool --check    # also load every map and play 100 random steps

Stages (PLAN.md section 3):
    1 combat      ViZDoom scenarios + small arenas full of monsters
    2 navigation  3-6 rooms with doors, few monsters, exit in the farthest room
    3 keycards    2-4 rooms, 1-2 locked doors, key usually in a side room (backtracking)
    4 full maps   Freedoom 2 maps + bigger generated maps with 2-3 locks and more monsters
Held-out maps (heldout=1) use different seeds / different Freedoom maps and are NEVER trained on.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

from doom_env.wad import list_maps, map_lumps, read_wad
from doom_env.wrapper import IWAD

from training.maps.udmf_gen import random_layout, write_map

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WADS = HERE / "wads"
CSV = HERE / "maps.csv"

SCENARIOS_STAGE1 = ["defend_the_center.wad", "defend_the_line.wad", "deadly_corridor.wad", "health_gathering.wad"]
SCENARIOS_STAGE2 = ["my_way_home.wad"]
FREEDOOM_HELDOUT = {"MAP02", "MAP06", "MAP14", "MAP31"}
FREEDOOM_MAX_SECTORS = 700

# stage -> (count train, count heldout, layout kwargs, episode_ticks)
GENERATED = {
    1: (30, 5, dict(n_rooms=(1, 2), n_locks=(0, 0), monsters=(3, 8), items=(2, 4), exit_p=0.5), 2100),
    2: (40, 6, dict(n_rooms=(3, 6), n_locks=(0, 0), monsters=(0, 3), items=(0, 3), exit_p=1.0), 2800),
    3: (50, 8, dict(n_rooms=(2, 4), n_locks=(1, 2), monsters=(0, 2), items=(0, 2), exit_p=1.0), 3500),
    4: (20, 5, dict(n_rooms=(5, 9), n_locks=(2, 3), monsters=(4, 10), items=(2, 5), exit_p=1.0), 4200),
}


def generate(stage: int, idx: int, heldout: bool, kw: dict) -> list[str]:
    seed = 1_000_000 * (2 if heldout else 1) + 1000 * stage + idx
    rng = np.random.default_rng(seed)
    n_rooms = int(rng.integers(kw["n_rooms"][0], kw["n_rooms"][1] + 1))
    n_locks = int(rng.integers(kw["n_locks"][0], kw["n_locks"][1] + 1))
    if n_rooms < 4:
        n_locks = min(n_locks, 1)
    exit_kind = None if rng.random() < kw["exit_p"] else "none"
    grid, angle = random_layout(rng, n_rooms, n_locks, kw["monsters"], kw["items"], exit_kind=exit_kind)
    name = f"{'heldout' if heldout else 'gen'}_s{stage}_{idx:03d}.wad"
    write_map(WADS / name, grid, rng, angle)
    return grid


def freedoom_rows() -> list[dict]:
    lumps = read_wad(IWAD)
    rows = []
    for m in list_maps(lumps):
        n_sec = len(map_lumps(lumps, m)["SECTORS"]) // 26
        if n_sec <= FREEDOOM_MAX_SECTORS:
            rows.append(dict(wad="@freedoom2", map=m, stage=4, episode_ticks=4200, heldout=int(m in FREEDOOM_HELDOUT)))
    return rows


def build() -> list[dict]:
    WADS.mkdir(parents=True, exist_ok=True)
    for old in WADS.glob("*.wad"):
        old.unlink()
    rows = [dict(wad=f"@vizdoom/{w}", map="MAP01", stage=1, episode_ticks=2100, heldout=0) for w in SCENARIOS_STAGE1]
    rows += [dict(wad=f"@vizdoom/{w}", map="MAP01", stage=2, episode_ticks=2100, heldout=0) for w in SCENARIOS_STAGE2]
    for stage, (n_train, n_held, kw, ticks) in GENERATED.items():
        for heldout, n in ((0, n_train), (1, n_held)):
            for i in range(n):
                generate(stage, i, bool(heldout), kw)
                name = f"{'heldout' if heldout else 'gen'}_s{stage}_{i:03d}.wad"
                rows.append(dict(wad=f"training/maps/wads/{name}", map="MAP01", stage=stage, episode_ticks=ticks, heldout=heldout))
    rows += freedoom_rows()
    with open(CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["wad", "map", "stage", "episode_ticks", "heldout"])
        wr.writeheader()
        wr.writerows(rows)
    return rows


def check(rows: list[dict]) -> None:
    from doom_env import make_env
    from training.envs import resolve_wad

    bad = 0
    for r in rows:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                env = make_env(resolve_wad(r["wad"]), r["map"], episode_ticks=r["episode_ticks"], render_buffers=False)
            env.reset(seed=0)
            priv = env.privileged_state()
            for _ in range(100):
                _, _, te, tr, _ = env.step(env.action_space.sample())
                if te or tr:
                    break
            locks = int((priv["door_lock"] > 0).sum()) if priv["door_lock"] is not None else -1
            env.close()
            print(f"ok   s{r['stage']} held={r['heldout']} {r['wad']}:{r['map']}  locked doors={locks}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {r['wad']}:{r['map']}: {e}")
    print(f"\n{len(rows) - bad}/{len(rows)} maps load and run")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="load every map and play 100 random steps")
    args = ap.parse_args()
    rows = build()
    counts = {}
    for r in rows:
        k = (r["stage"], r["heldout"])
        counts[k] = counts.get(k, 0) + 1
    for (s, h), n in sorted(counts.items()):
        print(f"stage {s} {'heldout' if h else 'train  '}: {n} maps")
    print(f"wrote {CSV.relative_to(REPO)}")
    if args.check:
        check(rows)
