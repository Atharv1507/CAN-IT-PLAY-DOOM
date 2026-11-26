"""Checkpoint evaluator (Person 2). Plays a checkpoint on maps it never trained on and prints a score table.

    python -m training.evaluate --checkpoint ~/doom-checkpoints/A_h10_12000000.pt
    python -m training.evaluate --checkpoint A.pt B.pt            # compare several, prints a ranking
    python -m training.evaluate --checkpoint A.pt --maps train --limit 5 --stages 3
    python -m training.evaluate --checkpoint A.pt --record replays/   # save .lmp replays (watch with training.watch)

- The bot is driven by submission/agent.py's DoomAgent, so we measure exactly what we would submit
  (same preprocessing, same greedy/sampled action choice, 1 CPU thread).
- Every checkpoint plays the same maps with the same seeds, so scores are comparable.
- "proxy" is OUR approximation of the official score (docs/EVALUATION.md categories), not the real one:
    per episode: 100*exit + 15*keys + 15*unlocks + 5*doors + 0.2*min(cells, 300) + 3*kills
                 - 0.05*damage_taken - 20*dead;   overall = stage-weighted mean (stage 1..4 -> weight 1..4)
- Results are appended to training/results.csv (one row per checkpoint and stage, plus an "all" row).
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "training" / "results.csv"
STAGE_WEIGHT = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
COLUMNS = ["exit", "keys", "unlocks", "doors", "cells", "kills", "dead", "timeout", "damage_taken", "return", "proxy"]


def proxy_score(s: dict) -> float:
    return (100 * s["exit"] + 15 * s["keys"] + 15 * s["unlocks"] + 5 * s["doors"] + 0.2 * min(s["cells"], 300)
            + 3 * s["kills"] - 0.05 * s["damage_taken"] - 20 * s["dead"])


# ----------------------------------------------------------------------------- worker process
_AGENT = None


def _load_agent(checkpoint: str, sample: bool | None):
    import torch

    torch.set_num_threads(1)
    from doom_env import spaces as S
    from submission.agent import DoomAgent
    from submission.model import DoomNet

    agent = DoomAgent(S.observation_space(), S.action_space(), "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("smoke_test"):
        raise ValueError(f"{checkpoint} is a smoke-test checkpoint (stub model), not a DoomNet checkpoint")
    cfg = ckpt.get("cfg") or {}
    agent.model = DoomNet(cfg)
    agent.model.load_state_dict(ckpt["model"])
    agent.model.eval()
    agent.cfg = cfg
    agent.eval_sample = bool(cfg.get("eval_sample", False)) if sample is None else sample
    agent.reset()
    return agent


def _init_worker(checkpoint: str, sample: bool | None):
    global _AGENT
    _AGENT = _load_agent(checkpoint, sample)


def _play(job: dict) -> dict:
    """One episode on one map. Runs inside a worker process."""
    import torch

    from training.envs import MapRow, TrainingEnv

    torch.manual_seed(job["seed"])
    row = MapRow(**job["row"])
    env = TrainingEnv({row.stage: 1}, maps=[row])
    options = {"record_path": job["record"]} if job.get("record") else None
    obs, _ = env.reset(seed=job["seed"], options=options)
    _AGENT.reset()
    times, done, stats = [], False, None
    while not done:
        t0 = time.perf_counter()
        action = _AGENT.act(obs)
        times.append((time.perf_counter() - t0) * 1000)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        if done:
            stats = info["episode_stats"]
    env.close()  # also finishes the .lmp recording
    if job.get("record"):
        Path(job["record"]).with_suffix(".json").write_text(json.dumps({"wad": row.wad, "map": row.map}))
    stats = {k: (float(v) if k not in ("map", "stage") else v) for k, v in stats.items()}
    stats["proxy"] = proxy_score(stats)
    stats["act_ms"] = times
    return stats


# ----------------------------------------------------------------------------- main process
def evaluate(checkpoint: Path, rows, episodes: int, workers: int, sample: bool | None,
             record_dir: Path | None) -> list[dict]:
    jobs = []
    for i, row in enumerate(rows):
        for ep in range(episodes):
            rec = None
            if record_dir is not None and ep == 0:
                rec = str(record_dir / f"{checkpoint.stem}__{Path(row.wad).stem.lstrip('@')}_{row.map}.lmp")
            jobs.append({"row": row.__dict__, "seed": 10_000 * ep + i, "record": rec})
    _load_agent(str(checkpoint), sample)  # fail fast in the main process if the checkpoint is broken
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(checkpoint), sample)) as pool:
        results = []
        for k, res in enumerate(pool.imap_unordered(_play, jobs), 1):
            results.append(res)
            print(f"\r  played {k}/{len(jobs)} episodes", end="", flush=True)
    print()
    return results


def summarize(results: list[dict]) -> tuple[dict, float]:
    by_stage = {}
    for s in sorted({r["stage"] for r in results}):
        rs = [r for r in results if r["stage"] == s]
        by_stage[s] = {c: float(np.mean([r[c] for r in rs])) for c in COLUMNS} | {"n": len(rs)}
    total_w = sum(STAGE_WEIGHT[s] for s in by_stage)
    overall = sum(STAGE_WEIGHT[s] * v["proxy"] for s, v in by_stage.items()) / total_w
    return by_stage, overall


def print_table(name: str, by_stage: dict, overall: float, results: list[dict], per_map: bool) -> None:
    times = np.concatenate([r["act_ms"] for r in results])
    print(f"\n{name}")
    print(f"{'stage':>5} {'eps':>4} {'exit%':>6} {'keys':>5} {'unlck':>5} {'doors':>5} {'cells':>6} {'kills':>5} "
          f"{'dead%':>6} {'tout%':>6} {'return':>7} {'proxy':>7}")
    for s, v in by_stage.items():
        print(f"{s:>5} {v['n']:>4} {100 * v['exit']:>6.1f} {v['keys']:>5.2f} {v['unlocks']:>5.2f} {v['doors']:>5.2f} "
              f"{v['cells']:>6.1f} {v['kills']:>5.2f} {100 * v['dead']:>6.1f} {100 * v['timeout']:>6.1f} "
              f"{v['return']:>7.2f} {v['proxy']:>7.1f}")
    print(f"overall proxy (stage-weighted): {overall:.1f}   act(): mean {times.mean():.2f} ms, "
          f"p99 {np.percentile(times, 99):.2f} ms, max {times.max():.1f} ms  (limit 40 ms avg)")
    if per_map:
        for r in sorted(results, key=lambda r: (r["stage"], r["map"])):
            print(f"   s{r['stage']} {r['map']:32s} exit={int(r['exit'])} keys={int(r['keys'])} unlocks={int(r['unlocks'])} "
                  f"cells={int(r['cells'])} kills={int(r['kills'])} dead={int(r['dead'])} proxy={r['proxy']:.1f}")


def append_csv(checkpoint: Path, maps_set: str, sample: bool | None, by_stage: dict, overall: float, results) -> None:
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    times = np.concatenate([r["act_ms"] for r in results])
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["time", "checkpoint", "run", "step", "maps", "sample", "stage", "episodes", *COLUMNS,
                         "overall_proxy", "act_ms_mean", "act_ms_p99"])
        base = [datetime.now().isoformat(timespec="seconds"), checkpoint.name, ck.get("run", ""), ck.get("step", ""),
                maps_set, sample]
        tail = [round(overall, 2), round(float(times.mean()), 3), round(float(np.percentile(times, 99)), 3)]
        for s, v in by_stage.items():
            wr.writerow(base + [s, v["n"]] + [round(v[c], 4) for c in COLUMNS] + tail)
        allv = {c: float(np.mean([r[c] for r in results])) for c in COLUMNS}
        wr.writerow(base + ["all", len(results)] + [round(allv[c], 4) for c in COLUMNS] + tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    ap.add_argument("--maps", choices=("heldout", "train", "all"), default="heldout")
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--limit", type=int, default=None, help="at most N maps per stage (quick check)")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per map (different seeds)")
    ap.add_argument("--workers", type=int, default=4, help="parallel games (M1 8 GB: max 4)")
    ap.add_argument("--sample", action="store_true", help="sample actions instead of agent.py's default")
    ap.add_argument("--record", type=Path, default=None, help="folder for .lmp replays (first episode per map)")
    ap.add_argument("--per-map", action="store_true", help="print one line per map")
    ap.add_argument("--no-save", action="store_true", help="don't append to training/results.csv")
    args = ap.parse_args()

    from training.envs import load_maps

    rows = [m for m in load_maps() if m.stage in args.stages
            and (args.maps == "all" or m.heldout == (args.maps == "heldout"))]
    if args.limit:
        rows = [m for s in args.stages for m in [r for r in rows if r.stage == s][:args.limit]]
    if not rows:
        raise SystemExit("no maps selected")
    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)
    sample = True if args.sample else None
    print(f"{len(rows)} maps x {args.episodes} episode(s), {args.workers} workers, maps={args.maps}")

    ranking = []
    for ck in args.checkpoint:
        ck = ck.expanduser()
        t0 = time.time()
        results = evaluate(ck, rows, args.episodes, args.workers, sample, args.record)
        by_stage, overall = summarize(results)
        print_table(f"{ck.name}  ({time.time() - t0:.0f} s)", by_stage, overall, results, args.per_map)
        if not args.no_save:
            append_csv(ck, args.maps, args.sample, by_stage, overall, results)
        ranking.append((overall, ck.name))
    if len(ranking) > 1:
        print("\nRanking (best first):")
        for i, (score, name) in enumerate(sorted(ranking, reverse=True), 1):
            print(f"  {i}. {name}  proxy {score:.1f}")
    if not args.no_save:
        print(f"\nsaved to {RESULTS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
