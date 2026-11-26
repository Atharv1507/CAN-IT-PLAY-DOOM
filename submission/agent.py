"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"

if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

try:
    from .model import DoomNet, heads_to_buttons, obs_to_tensors
except ImportError:
    from model import DoomNet, heads_to_buttons, obs_to_tensors  # type: ignore[import-not-found]  # noqa: E402

class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once. Load weights from WEIGHTS_DIR onto `device` ("cuda" or "cpu")."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)
        torch.set_num_threads(1)

        weights_path = WEIGHTS_DIR / "model.pt"
        if weights_path.is_file():
            try:
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=True)
            except Exception:
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)

            if isinstance(ckpt, dict) and "model" in ckpt:
                cfg = ckpt.get("cfg") or {}
                state_dict = ckpt["model"]
            else:
                cfg = {}
                state_dict = ckpt
            self.cfg = cfg
            self.model = DoomNet(cfg).to(self.device)
            self.model.load_state_dict(state_dict)
        else:
            # Untrained fallback before final weights exist so validate.py works now.
            self.cfg = {}
            self.model = DoomNet(self.cfg).to(self.device)

        self.model.eval()
        # Sample from the policy by default: greedy (argmax) got stuck facing walls in held-out tests.
        self.eval_sample = bool(self.cfg.get("eval_sample", True))
        self._done_zero = torch.zeros(1, dtype=torch.float32, device=self.device)
        self._fallback = np.zeros(self.action_space.shape, dtype=np.int8)
        self.state: dict[str, torch.Tensor] = self.model.initial_state(1, device=self.device)

    def reset(self) -> None:
        """Called before every map. Clear both GRU and episodic memory."""
        self.state = self.model.initial_state(1, device=self.device)

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int array of shape (9,) with values in {0, 1}."""
        try:
            with torch.no_grad():
                obs_t = obs_to_tensors(observation, device=self.device)
                logits, _, self.state = self.model(obs_t, self.state, self._done_zero)
                if self.eval_sample:
                    picks = {
                        k: int(torch.distributions.Categorical(logits=v).sample().item())
                        for k, v in logits.items()
                    }
                else:
                    picks = {k: int(v.argmax(dim=-1).item()) for k, v in logits.items()}
                return heads_to_buttons(
                    move=picks["move"],
                    turn=picks["turn"],
                    attack=picks["attack"],
                    use=picks["use"],
                    weapon=picks["weapon"],
                )
        except Exception:
            return self._fallback.copy()
