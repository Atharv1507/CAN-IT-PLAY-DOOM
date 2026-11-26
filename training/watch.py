"""Watch the bot play (Person 2). Graphs hide reward hacking; watching the bot exposes it.

    python -m training.watch replays/A_h10_..._heldout_s3_000_MAP01.lmp      # replay a recording from evaluate --record
    python -m training.watch --checkpoint ~/doom-checkpoints/A_h10.pt --stage 3   # play a held-out stage-3 map now, then show it
    python -m training.watch --checkpoint A.pt --map heldout_s3_002 --speed 2

Look for: spinning in place, hugging walls, spamming USE at a door (it bounces), walking past keys,
standing still, fighting forever instead of heading for the exit.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import vizdoom as vd

from doom_env.wrapper import IWAD, SKILL

from training.envs import load_maps, resolve_wad


def replay(lmp: Path, wad: str, map_name: str, speed: float = 1.0, headless: bool = False) -> int:
    g = vd.DoomGame()
    g.set_doom_game_path(IWAD)
    g.set_doom_scenario_path(resolve_wad(wad))
    g.set_doom_map(map_name)
    g.set_doom_skill(SKILL)
    g.set_mode(vd.Mode.PLAYER)
    g.set_screen_resolution(vd.ScreenResolution.RES_640X480)
    g.set_render_hud(True)
    g.set_sound_enabled(False)
    g.set_window_visible(not headless)
    g.init()
    g.replay_episode(str(lmp))
    tics = 0
    while not g.is_episode_finished():
        g.advance_action()
        tics += 1
        if not headless:
            time.sleep(1 / (35 * speed))
    g.close()
    return tics


def record_one(checkpoint: Path, row, out: Path, sample: bool | None) -> dict:
    from training.envs import TrainingEnv
    from training.evaluate import _load_agent

    agent = _load_agent(str(checkpoint), sample)
    env = TrainingEnv({row.stage: 1}, maps=[row])
    obs, _ = env.reset(seed=0, options={"record_path": str(out)})
    done, stats = False, {}
    while not done:
        obs, _, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
        if done:
            stats = info["episode_stats"]
    env.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lmp", nargs="?", type=Path, help=".lmp from evaluate --record (needs its .json next to it)")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--map", help="part of a map name from maps.csv, e.g. heldout_s3_002 or MAP14")
    ap.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    ap.add_argument("--sample", action="store_true", help="sample actions instead of agent.py's default")
    ap.add_argument("--headless", action="store_true", help="no window (for testing)")
    args = ap.parse_args()

    if args.lmp:
        meta = json.loads(args.lmp.with_suffix(".json").read_text())
        tics = replay(args.lmp, meta["wad"], meta["map"], args.speed, args.headless)
        print(f"replayed {tics} tics ({tics / 35:.0f} s of game time)")
        return
    if not args.checkpoint:
        ap.error("give a .lmp file or --checkpoint")
    rows = [m for m in load_maps() if m.heldout]
    rows = [m for m in rows if args.map in m.name] if args.map else [m for m in rows if m.stage == args.stage]
    if not rows:
        ap.error("no held-out map matches")
    row = rows[int(np.random.default_rng().integers(len(rows)))]
    with tempfile.TemporaryDirectory() as d:
        lmp = Path(d) / "episode.lmp"
        print(f"playing {row.name} (stage {row.stage}) with {args.checkpoint.name} ...")
        stats = record_one(args.checkpoint.expanduser(), row, lmp, True if args.sample else None)
        print({k: stats[k] for k in ("exit", "keys", "unlocks", "doors", "cells", "kills", "dead", "steps")})
        tics = replay(lmp, row.wad, row.map, args.speed, args.headless)
        print(f"replayed {tics} tics ({tics / 35:.0f} s of game time)")


if __name__ == "__main__":
    main()
