# Team Plan — Track 1 "Can It Play DOOM?" (20 hours, 3 people, 3 MacBooks)

This file is the single source of truth for **what we build, who builds what, and how the pieces connect**.
AI assistants: read `AGENTS.md` first for how to guide your person.

`H0` = hackathon start. `H6` = 6 hours after start, and so on.

---

## 1. Main idea

- We train a bot that plays **unseen** DOOM maps using a small neural network trained with **our own PPO** (reinforcement learning).
- Inputs: only `rays`, `visible_entities` and `vitals` from the official wrapper. **No RGB, no depth** (too slow on laptops).
- Memory (both are **core and non-negotiable**): a **GRU** (short-term) **+ episodic memory** (snapshots it can look back at, so it can remember a locked door and come back with the key).
- Training: rewards that copy what the scorer pays for + a 4-stage curriculum. **Stage 3 (keycards) starts at H10** and is **non-negotiable**.
- Everything runs locally on 3 Macs. **No cloud, no Colab.**

## 2. People, laptops, roles

| Person | Laptop | Role | Owns (only this person edits these files) |
| --- | --- | --- | --- |
| Person 1 | Laptop A — MacBook Air M4 #1 | Training lead | `training/ppo.py`, `training/train.py`, `training/configs/*.json`, run A |
| Person 2 | Laptop C — MacBook Air M1, 8 GB | Environment, maps, evaluation | `training/rewards.py`, `training/envs.py`, `training/maps/`, `training/evaluate.py`, `training/results.csv` |
| Person 3 | Laptop B — MacBook Air M4 #2 | Model and submission | `submission/model.py`, `submission/agent.py`, `submission/weights/`, `README.md`, run B (hosted) |

- Both M4s train from H6. The M1 does coding, maps, and scoring of checkpoints (at most 4 game copies, 8 GB RAM).
- Person 1 writes run B's config; Person 3 just launches it on Laptop B.

## 3. The solution, piece by piece

| Piece | What it is | Owner |
| --- | --- | --- |
| Ray encoder | 1D conv over 64 rays (distance + one-hot hit type, circular padding) | P3 |
| Entity encoder | Attention pooling over up to 32 visible entities (features + type embedding, uses `mask`) | P3 |
| Vitals encoder | Small MLP over the 17 normalized vitals | P3 |
| GRU | Hidden size 256, carries memory between steps | P3 |
| Episodic memory | Every 8 steps write a 128-d snapshot into a ring buffer (max 64 slots); every step attend over it with the GRU output as the query | P3 |
| Action heads | 5 separate choices: move (7), turn (3), attack (2), use (2), next weapon (2) | P3 |
| Privileged critic | Value head that also receives a 16-number ground-truth vector — **training only, never used in `agent.py`** | P3 (head), P2 (vector) |
| Own recurrent PPO | Rollouts from parallel games, GAE, clipped loss, sequences for the GRU + memory, resumable | P1 |
| Reward shaping | Exit, keycard, door, unlock, new 64×64 cell, kills; damage, death, time penalties | P2 |
| Curriculum | Stage 1 combat, 2 navigation, 3 keycards (from H10), 4 full maps; always mix ~30% earlier stages | P1 (schedule), P2 (maps) |
| Small keycard maps | 2–4 rooms: key → locked door → exit; target 20+ | P2 |
| Held-out maps | Never trained on; our stand-in for the hidden evaluation | P2 |
| Inference safety | `torch.set_num_threads(1)`, small model, `try/except` in `act()` | P3 |

Model size target: 1–3 M parameters, well under 40 ms per decision on 1 CPU thread.

## 4. Interfaces (frozen at H1 — changing one = edit this file + tell the team)

### 4.1 File layout

```
submission/model.py        DoomNet + obs_to_tensors + heads_to_buttons. Self-contained: imports only torch/numpy.
submission/agent.py        DoomAgent: loads submission/weights/model.pt, runs DoomNet on CPU, 1 thread.
submission/weights/model.pt   final weights only (Git LFS)
training/envs.py           make_training_env(...): official doom_env + map sampling + reward shaping
training/rewards.py        RewardShaper + priv_vector
training/maps/maps.csv     map list
training/maps/wads/        our .wad files
training/ppo.py            recurrent PPO update
training/train.py          entry point: parallel games, logging, checkpoints, resume
training/configs/*.json    one config per run / fork
training/evaluate.py       per-tier score table for a checkpoint
training/results.csv       every evaluation result (written only by evaluate.py)
```

- Training code imports the model with `from submission.model import DoomNet, obs_to_tensors, heads_to_buttons` (run everything **from the repo root** as `python -m training.train ...`).
- `agent.py` imports `model.py` from its own folder (add `Path(__file__).parent` to `sys.path`). Never import from `training/` in `submission/`.

### 4.2 Map list — `training/maps/maps.csv`

```
wad,map,stage,episode_ticks,heldout
training/maps/wads/key_small_01.wad,MAP01,3,2100,0
```

- `stage`: 1 combat, 2 navigation, 3 keycards, 4 full maps. `heldout`: 1 = never used for training.
- `wad` is a path relative to the repo root, or `@vizdoom/<file>.wad` (ViZDoom built-in scenario) or `@freedoom2` (the bundled IWAD), so the file works on every laptop. `training.envs.resolve_wad()` turns these into real paths.
- Built by `python -m training.maps.build_pool [--check]` (deterministic). Generator: `training/maps/udmf_gen.py` (ASCII grid → UDMF map; `random_layout()` makes rooms, doors, locked doors, keys, exits, monsters).
- Reward check: `python -m training.check_rewards` (scripted tests that every term fires + random play per stage).

### 4.3 Training env — `training/envs.py` (Person 2)

```python
make_training_env(stage_mix: dict[int, float], reward_weights: dict, seed: int,
                  heldout: bool = False) -> gymnasium.Env
```

- On every `reset()` it samples a map from `maps.csv` by `stage_mix` (e.g. `{1: 0.15, 2: 0.15, 3: 0.7}`) and builds `doom_env.make_env(wad, map, episode_ticks=..., render_buffers=False)`.
- Observations: unchanged official dict (so training matches evaluation).
- `reward`: shaped reward from `RewardShaper` (replaces the wrapper's default reward).
- `info["priv"]`: `np.float32` array of shape `(16,)` from `priv_vector()` — for the critic only.
- `info["terms"]`: dict of each reward term this step (for TensorBoard).
- `info["episode_stats"]` on the final step: `exit, dead, timeout, kills, keys, doors, unlocks, cells, damage_taken, steps, map, stage`.

### 4.4 Rewards — `training/rewards.py` (Person 2)

Ground truth comes from `env.privileged_state()` (position, kills, items, keys, damage_taken, dead, timeout, exit_reached, sector_floor, sector_ceiling, door_lock). Starting weights (Person 2 tunes):

| Term | Start value |
| --- | --- |
| exit reached | +10 (+ up to +2 for remaining health/time); **set to 40 in all configs from H7** (with exploration paying ~10-20 per episode, +10 made wandering until timeout worth more than exiting) |
| keycard picked up (each) | +3 |
| door opened (first time per door sector) | +0.5 |
| locked door opened (first time) | +3 |
| new 64×64 cell visited (first visit) | +0.1 |
| kill | +1 |
| damage taken | −0.01 per HP |
| death | −5 |
| time | −0.001 per step |

- Door opened = a sector with `door_lock >= 0` whose `ceiling − floor` rises above 56 for the first time. Locked = `door_lock > 0`.
- Run B variant weights: novelty ×2, kill ×0.5 (Person 1 puts these in `runB.json`).

`priv_vector(priv: dict, cells_visited: int) -> np.ndarray (16,)`, in this order:
`x/4096, y/4096, sin(angle), cos(angle), health/100, damage_taken/100, kills/10, items/10, keys[6], tic/episode_ticks, cells_visited/100`.

### 4.5 Model — `submission/model.py` (Person 3)

```python
class DoomNet(torch.nn.Module):
    def __init__(self, cfg: dict): ...
    def initial_state(self, batch: int) -> dict: ...
        # {"h": [1,B,256], "mem": [B,64,128], "mem_mask": [B,64], "t": [B]}
    def forward(self, obs: dict, state: dict, done: torch.Tensor, priv: torch.Tensor | None = None):
        # obs: batched tensors from obs_to_tensors; done [B] resets state where 1
        # returns (logits: dict[str, Tensor], value: Tensor | None, new_state: dict)

def obs_to_tensors(obs: dict, device="cpu") -> dict[str, torch.Tensor]   # batched or single obs
def heads_to_buttons(move: int, turn: int, attack: int, use: int, weapon: int) -> np.ndarray  # (9,) int8
```

- Heads → buttons (CONTRACT.md §3): move `0 none, 1 fwd, 2 back, 3 strafe L, 4 strafe R, 5 fwd+L, 6 fwd+R`; turn `0 none, 1 left, 2 right`; attack/use `0/1`. **Weapon head is ignored from H8** (`SELECT_NEXT_WEAPON` always 0: the bot kept cycling to fists; Doom auto-switches on pickup). `agent.py` samples actions by default (greedy got stuck facing walls).
- `value` is computed only when `priv` is given (training). `agent.py` never passes `priv`.

### 4.6 Checkpoints

```python
torch.save({"model": state_dict_on_cpu, "cfg": cfg, "optimizer": opt_state,
            "step": env_steps, "stage_mix": ..., "reward_weights": ..., "run": "A"}, path)
```

- Name: `<run>_h<hour>_<steps>.pt`, e.g. `A_h10_12000000.pt`. Saved every 30 min to the local `checkpoints/` folder (gitignored) and copied to `~/doom-checkpoints/` / AirDropped to Laptop C.
- Final submission file `submission/weights/model.pt` keeps only `model` + `cfg` (no optimizer), pushed with Git LFS.

### 4.7 Commands

```bash
python -m training.train --config training/configs/runA.json                 # start
python -m training.train --config training/configs/runA.json --resume        # after a crash/sleep
python -m training.train --config training/configs/runA_fork1.json --init ~/doom-checkpoints/B_h10_12000000.pt   # fork
python -m training.evaluate --checkpoint ~/doom-checkpoints/A_h10_12000000.pt --maps heldout --workers 4
python -m training.evaluate --checkpoint A.pt B.pt --record replays/          # compare + ranking + replays
python -m training.watch replays/<file>.lmp                                      # watch a replay in a window
python -m training.watch --checkpoint ~/doom-checkpoints/A_h10_12000000.pt --stage 3   # play + watch now
python validate.py
```

Config (`training/configs/runA.json`): `run, seed, num_envs (6–8), device (cpu|mps), stage_mix, reward_weights, lr, rollout_len, seq_len, hours`.
PPO starting values: lr 2.5e-4, gamma 0.995, GAE λ 0.95, clip 0.2, entropy 0.01, 4 epochs, 4 minibatches, rollout 128 steps, sequence length 32 (store the GRU + memory state at the start of each sequence).

## 5. Timeline

| Hours | Person 1 (Laptop A) | Person 2 (Laptop C) | Person 3 (Laptop B) |
| --- | --- | --- | --- |
| H0–1 | Setup; M4 prep; measure steps/sec with 1, 4, 8 game copies | Setup done; create `~/doom-checkpoints/`; read `docs/` | Setup on M4 #2 (Python 3.13); M4 prep; run `pytest` |
| **H1 sync** | Agree this file's §4 | Agree §4 | Agree §4 |
| H1–3 | Research recurrent PPO (CleanRL as reference); count-based novelty | Find keycard maps; Obsidian (time-box 1 h); UDMF script plan; check `privileged_state()` fields | Research attention pooling, GRU, episodic memory; design heads |
| **H3 sync** | | | |
| H3–6 | Build `ppo.py`, `train.py` (AsyncVectorEnv, TensorBoard, save/resume) | Build `rewards.py`, `envs.py`, `maps.csv`, small keycard maps, held-out set | Build `model.py` incl. episodic memory; `agent.py` loading a checkpoint; `reset()` clears GRU + memory |
| H5 | Sanity check with P3 on `defend_the_center` (reward must rise in ~30 min); CPU vs MPS test | Random-action check: every reward term fires | Sanity check with P1 |
| **H6 sync** | Launch run A (stages 1–2); commit `runB.json` | Push maps + rewards to `main` | Launch run B (stages 1–2) |
| H6–10 | Monitor TensorBoard; fix reward hacking; copy checkpoints | Build `evaluate.py`; tiny keycard-map test (≤4 games); 20+ keycard maps | Latency check (1 thread, < 40 ms); `try/except` in `act()`; `validate.py` with real checkpoint |
| **H10 sync (fork 1)** | Load picked checkpoint; Stage 2+3 mix | Score A and B; pick fork source | Load picked checkpoint; Stage 3-heavy (~30% stages 1–2) |
| H10–14 | Monitor; draft training section of brief | Score every 30–60 min; record episode videos | README; brief layout; architecture diagram |
| **H14 sync (fork 2)** | Best checkpoint; Stage 3+4 mix, seed 1 | Score; pick fork source | Best checkpoint; Stage 3+4 mix, seed 2 |
| H14–17 | Monitor; checkpoint every 30 min | Score + rank in `results.csv` | `validate.py` on each new best |
| **H17 freeze** | Stop training | Pick final checkpoint → send to P3 | Copy to `submission/weights/model.pt`; LFS push; validate on fresh clone |
| H18–20 | Brief, demo clips, Q&A rehearsal | Same | Same |

## 6. Git workflow

- Branches: `p1-training`, `p2-env`, `p3-model`. Commit often to your own branch.
- At each sync point: `git pull origin main`, merge your branch into `main`, run `python validate.py`, push. **`main` must always pass `validate.py`.**
- Never commit: `.venv/`, `runs/`, `checkpoints/`, `*.pt` outside `submission/weights/`, `.DS_Store`.
- Only `submission/weights/model.pt` goes through Git LFS, once, at H17.

## 7. Mac rules

- Python 3.13 venv with **pip** (not uv): `python3.13 -m venv .venv && source .venv/bin/activate && pip install -e ./env && pip install torch pytest tensorboard`.
- Never `pip install -e .` at the repo root (that config is uv-only).
- Training Macs: plugged in, Low Power Mode off, `caffeinate -dimsu` running, 6–8 game copies (not 10).
- Every script that starts parallel games needs `if __name__ == "__main__":`.
- Save checkpoints as CPU tensors so they load on the organizers' Linux machine.
- `validate.py` warnings about pinning to 1 CPU and peak RAM are normal on macOS.

## 8. Competition rules we must never break (docs/RULES.md)

- Actions must come from the trained network. **No** hand-written if/else action rules, **no** A*/Dijkstra/BFS, **no** `atan2` aimbots.
- **No** off-the-shelf Stable-Baselines3 (or similar) policies — our PPO and architecture are our own.
- `agent.py` gets only the observation dict: no game engine, no `privileged_state()`, no network, repo is read-only (scratch files to `/tmp` only).
- Limits: 40 ms per decision on average, 1 CPU thread, 2 GB VRAM, 8 GB RAM, repo ≤ 1 GB, extra packages pinned in `submission/requirements.txt`.
- A crash, timeout or invalid action freezes our player for the rest of the map → always `try/except` in `act()` with a safe fallback action.
