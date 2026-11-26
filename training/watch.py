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


def replay(lmp: Path, wad: str, map_name: str, speed: float = 1.0, headless: bool = False,
           gif: Path | None = None, every: int = 8) -> int:
    """Replay an .lmp in a window, or (gif=...) save it as an animated GIF: game view + overhead map."""
    if gif is not None:
        headless = True
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
    if gif is not None:
        g.set_screen_resolution(vd.ScreenResolution.RES_320X240)
        g.set_screen_format(vd.ScreenFormat.RGB24)
        g.set_automap_buffer_enabled(True)
        g.set_automap_mode(vd.AutomapMode.OBJECTS_WITH_SIZE)
        g.set_automap_rotate(False)
        g.set_automap_render_textures(False)
    g.init()
    g.replay_episode(str(lmp))
    tics, frames = 0, []
    while not g.is_episode_finished():
        if gif is not None and tics % every == 0:
            st = g.get_state()
            if st is not None:
                frames.append(np.concatenate([st.screen_buffer, st.automap_buffer], axis=1))
        try:
            g.advance_action()
        except vd.ViZDoomErrorException:  # automap rendering can crash on the very last tic; frames are kept
            break
        tics += 1
        if not headless:
            time.sleep(1 / (35 * speed))
    try:
        g.close()
    except Exception:  # noqa: BLE001
        pass
    if gif is not None and frames:
        from PIL import Image

        imgs = [Image.fromarray(f).quantize(colors=128) for f in frames]
        ms = int(1000 * every / (35 * speed))
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=ms, loop=0, optimize=True)
        print(f"saved {gif} ({len(frames)} frames)")
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
    ap.add_argument("--gif", type=Path, help="save an animated GIF (game view + overhead map) instead of a window")
    args = ap.parse_args()

    if args.lmp:
        meta = json.loads(args.lmp.with_suffix(".json").read_text())
        tics = replay(args.lmp, meta["wad"], meta["map"], args.speed, args.headless, args.gif)
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
        tics = replay(lmp, row.wad, row.map, args.speed, args.headless, args.gif)
        print(f"replayed {tics} tics ({tics / 35:.0f} s of game time)")


if __name__ == "__main__":
    main()
