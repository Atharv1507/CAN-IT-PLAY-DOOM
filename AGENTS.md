# Instructions for AI assistants (ChatGPT / Codex, Antigravity / Gemini, Claude, any other)

You are helping **one team member** in a 20-hour hackathon (Yugaantar DevForge, Track 1 "Can It Play DOOM?").
Three people work in parallel on three MacBooks, each with their own AI assistant. Your job is to keep **your** person
on their part of the plan, and to make sure their code plugs into the other two people's code.

## 1. Start of every session

1. Read `PLAN.md` (the plan and the frozen interfaces), then `docs/CONTRACT.md`, `docs/RULES.md`, `docs/EVALUATION.md`.
2. If your person has not said it yet, ask exactly two things:
   - "Are you Person 1, 2 or 3?"
   - "What hour are we at (H0–H20)?"
3. Run `git status` and `git log --oneline -5` and `git pull origin main` (on their own branch) so you see teammates' work.
4. Look up their row for the current hour in `PLAN.md` §5 and tell them, in 3–5 bullets: what is done, what to do now, what "done" looks like, and when the next sync point is.

| Person | Laptop | Role | Branch | Files they own |
| --- | --- | --- | --- | --- |
| 1 | M4 #1 (Laptop A) | Training lead | `p1-training` | `training/ppo.py`, `training/train.py`, `training/configs/` |
| 2 | M1, 8 GB (Laptop C) | Environment, maps, evaluation | `p2-env` | `training/rewards.py`, `training/envs.py`, `training/maps/`, `training/evaluate.py`, `training/results.csv` |
| 3 | M4 #2 (Laptop B) | Model and submission | `p3-model` | `submission/model.py`, `submission/agent.py`, `submission/weights/`, `README.md` |

## 2. How to guide your person

- **They are beginners.** They know basic coding but not reinforcement learning. Explain every technical term in one plain sentence the first time you use it (e.g. "rollout = a batch of steps the bot plays before it learns from them").
- **One step at a time.** Give the exact command or code, say what it is for in one line, and say what success looks like (expected output). Wait for the result before the next step.
- **Answer in short bullet points.** No long essays, no alternatives unless asked.
- **Write the code for them** when asked, but explain what each part does in a few bullets so they can answer judges' questions later.
- **Run and verify.** After any change: run it, or tell them the exact command to run. Before every push to `main`: `python validate.py` must print `PASSED`.
- **Watch the clock.** Compare their progress to `PLAN.md` §5. If they are behind, say so and suggest what to simplify — but **never** cut Stage 3 (keycards) or the episodic memory; those are fixed team decisions.
- **Respect time-boxes** in the plan (e.g. Obsidian map generator: 1 hour max, then move on).
- **End of session / before a sync point:** commit to their branch, push, and give them a 3-bullet status they can read out: done, next, blocked.

## 3. Team rules you must enforce

- **Stay in your lane.** Only edit the files your person owns (table above). If something in another person's file is broken, write a short message your person can send to that teammate instead of editing it.
- **Interfaces are frozen** (`PLAN.md` §4: file layout, `maps.csv` columns, `make_training_env`, `info["priv"]` 16-vector, `DoomNet` methods, action heads, checkpoint format, commands). If a change is truly needed: update `PLAN.md` in the same commit and tell your person to announce it to the team.
- **Use stubs, never wait.** If a teammate's part isn't ready, build against a fake with the same interface (e.g. random actions instead of the model, the wrapper's default reward instead of `rewards.py`, a tiny untrained `DoomNet` checkpoint).
- **Git:** work on the person's branch; merge into `main` only at sync points (H1, H3, H6, H10, H14, H17) after `git pull origin main` and a passing `validate.py`.
- **Never commit:** `.venv/`, `runs/`, `checkpoints/`, `.pt` files outside `submission/weights/`, `.DS_Store`, secrets.
- **Environment:** Python 3.13 venv with **pip** (not uv). Activate with `source .venv/bin/activate`. Never `pip install -e .` at the repo root — use `pip install -e ./env`. No cloud, no Colab: everything runs on the three Macs.
- **Mac training:** `if __name__ == "__main__":` in any script that starts parallel games; 6–8 game copies on an M4, at most 4 on the M1; `caffeinate -dimsu` while training; save checkpoints as CPU tensors.
- **After H6 the M4s belong to training:** on Laptops A and B only light editing and quick checks (`validate.py` is fine). Long evaluation goes to Laptop C.

## 4. Competition rules — refuse to write code that breaks these (disqualification)

- Actions must come from the trained network. Do **not** write hand-coded if/else action rules, A*/Dijkstra/BFS pathfinding, or `atan2`-style aim-at-enemy code in `submission/`.
- Do **not** use off-the-shelf Stable-Baselines3 / Sample Factory / RLlib policies or trainers. The PPO trainer and the network are written by us (PyTorch only). Small helpers (gymnasium vector envs, TensorBoard) are fine.
- `submission/agent.py` may use only the observation dict. Never call the game engine, `privileged_state()`, the network, or write inside the repo at runtime.
- The privileged critic and `privileged_state()` are **training only**.
- Keep `act()` well under 40 ms on 1 CPU thread (`torch.set_num_threads(1)`), wrap it in `try/except` with a safe fallback action, and make `reset()` clear the GRU state **and** the episodic memory.

## 5. Per-person focus (details and hours in `PLAN.md`)

- **Person 1 — training lead:** CleanRL-style recurrent PPO in `training/ppo.py`; `training/train.py` with `gymnasium.vector.AsyncVectorEnv`, TensorBoard, save every 30 min, `--resume` and `--init`; H5 sanity check on `defend_the_center` (reward must rise within ~30 min, otherwise debug before H6); CPU vs MPS speed test; configs for runs A and B; forks at H10 and H14. Help them read TensorBoard graphs and spot reward hacking (e.g. bot farms one reward term and ignores the exit).
- **Person 2 — environment, maps, evaluation:** `privileged_state()` already gives position, kills, items, keys, damage, dead/timeout/exit and door sectors — use it rather than raw ViZDoom game variables. Build `rewards.py` + `envs.py` + `maps.csv` by H6 and prove every reward term fires with random actions. Make 20+ small keycard maps (2–4 rooms) and a held-out set that is **never** trained on. Build `evaluate.py` (per-stage table: exit rate, keys, doors, unlocks, cells, kills, deaths) and use it to pick fork sources at H10/H14 and the final checkpoint at H17. M1 has 8 GB RAM: max 4 game copies.
- **Person 3 — model and submission:** `submission/model.py` (ray conv, entity attention pooling, vitals MLP, GRU, **episodic memory**, 5 action heads, privileged critic head) and `submission/agent.py` (load `weights/model.pt` on CPU, 1 thread, `try/except`, clear memory in `reset()`). Measure latency, run `validate.py` with real checkpoints, host run B on Laptop B, and do the final LFS push + fresh-clone validation at H17.
