"""Reward check (Person 2, H5): proves every reward term fires and the training env works.

    python -m training.check_rewards              # scripted tests + random play on every stage
    python -m training.check_rewards --episodes 3

Part 1 uses tiny hand-made corridor maps and fixed button sequences (test scaffolding only, never
used by the agent) so each term MUST fire. Part 2 plays random-ish actions on each stage and prints
how often each term fired, so we can spot terms that are too big or never happen.
"""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.envs import MapRow, TrainingEnv
from training.maps.udmf_gen import write_map

# buttons: 0 fwd, 1 back, 2 strafe L, 3 strafe R, 4 turn L, 5 turn R, 6 attack, 7 use, 8 next weapon
FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
FWD_USE = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0], np.int8)
ATTACK = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0], np.int8)

TESTS = {
    # name: (grid, button policy, terms that must fire)
    # USE fires on button-down and re-pressing reverses a moving door, so pulse it every 30 steps
    "key + locked door + exit": (["#########", "#P.r.1.E#", "#########"],
                                 lambda t: FWD_USE if t % 30 == 0 else FWD, {"key", "unlock", "cell", "exit", "progress"}),
    "normal door": (["#######", "#P.D..#", "#######"],
                    lambda t: FWD_USE if t % 30 == 0 else FWD, {"door", "cell"}),
    "exit switch": (["#####", "#P.X#", "#####"],
                    lambda t: FWD_USE if t % 2 == 0 else FWD, {"exit"}),
    "kill": (["#########", "#P....z.#", "#########"], lambda t: ATTACK, {"kill"}),
    # firing wakes the monsters up; too many to survive with a pistol
    "damage + death": (["#########", "#P..dddd#", "#...dddd#", "#...gggg#", "#########"], lambda t: ATTACK,
                       {"damage", "death"}),
    # opt-in terms (H12): firing at nothing is penalised; walking toward key -> door -> exit earns progress
    "blind fire": (["#######", "#P....#", "#######"], lambda t: ATTACK, {"blind_fire"}),
}


OPT_IN = {"progress": 0.1, "blind_fire": 0.005}


def scripted(tmp: Path) -> bool:
    ok_all = True
    for name, (grid, policy, must) in TESTS.items():
        wad = write_map(tmp / f"{abs(hash(name))}.wad", grid)
        env = TrainingEnv({9: 1}, OPT_IN, maps=[MapRow(str(wad), "MAP01", 9, 1400, False)], goals_any_path=True)
        env.reset(seed=0)
        fired, total, t, prog = set(), 0.0, 0, 0.0
        while True:
            _, r, te, tr, info = env.step(policy(t))
            total += r
            prog += info["terms"]["progress"]
            t += 1
            fired |= {k for k, v in info["terms"].items() if v != 0 and k != "time"}
            if te or tr:
                break
        env.close()
        missing = must - fired
        if "progress" in must and prog <= 0:
            missing = missing | {"progress>0"}
        ok_all &= not missing
        print(f"{'PASS' if not missing else 'FAIL'}  {name:26s} fired={sorted(fired)}  return={total:+.2f}  "
              f"steps={t}" + (f"  progress={prog:+.2f}" if "progress" in must else "") + (f"  MISSING={sorted(missing)}" if missing else ""))
    return ok_all


def random_play(episodes: int) -> None:
    rng = np.random.default_rng(0)
    print("\nRandom play (forward-biased random buttons, USE pulsed):")
    print(f"{'stage':6s} {'return':>8s} " + " ".join(f"{k:>7s}" for k in ("exit", "key", "door", "unlock", "cell", "kill", "damage", "death")))
    for stage in (1, 2, 3, 4):
        env = TrainingEnv({stage: 1}, seed=stage)
        sums, rets = defaultdict(float), []
        for ep in range(episodes):
            env.reset()
            ret, t, done = 0.0, 0, False
            while not done:
                a = (rng.random(9) < [0.6, 0.05, 0.1, 0.1, 0.25, 0.25, 0.3, 0.0, 0.02]).astype(np.int8)
                a[7] = int(t % 8 == 0)
                _, r, te, tr, info = env.step(a)
                ret += r
                t += 1
                done = te or tr
                for k, v in info["terms"].items():
                    sums[k] += v
            rets.append(ret)
        env.close()
        print(f"{stage:<6d} {np.mean(rets):8.2f} " + " ".join(
            f"{sums[k] / episodes:7.2f}" for k in ("exit", "key", "door", "unlock", "cell", "kill", "damage", "death")))
    print("(numbers = average reward per episode from each term)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        ok = scripted(Path(d))
    random_play(args.episodes)
    print("\nALL SCRIPTED TESTS PASSED" if ok else "\nSOME SCRIPTED TESTS FAILED")
    raise SystemExit(0 if ok else 1)
