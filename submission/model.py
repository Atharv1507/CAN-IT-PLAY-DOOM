"""Neural network architecture, observation preprocessing, and action mapping.

Self-contained module (imports only torch, numpy, and stdlib) shared between
training (`from submission.model import ...`) and evaluation (`agent.py`).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Contract constants (mirrors env/doom_env/spaces.py and docs/CONTRACT.md)
NUM_RAYS = 64
NUM_RAY_HIT_TYPES = 6
MAX_ENTITIES = 32
NUM_ENTITY_FEATURES = 5
NUM_ENTITY_TYPES = 12
NUM_VITALS = 17
NUM_BUTTONS = 9
PRIV_DIM = 16

HEAD_SIZES: dict[str, int] = {
    "move": 7,
    "turn": 3,
    "attack": 2,
    "use": 2,
    "weapon": 2,
}

# Move head index -> (MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT)
_MOVE_TABLE = (
    (0, 0, 0, 0),  # 0: none
    (1, 0, 0, 0),  # 1: forward
    (0, 1, 0, 0),  # 2: backward
    (0, 0, 1, 0),  # 3: strafe left
    (0, 0, 0, 1),  # 4: strafe right
    (1, 0, 1, 0),  # 5: forward + strafe left
    (1, 0, 0, 1),  # 6: forward + strafe right
)


def obs_to_tensors(obs: dict, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Convert a single or batched observation dict into batched PyTorch tensors.

    Ignores `rgb` and `depth` buffers for speed; extracts only `vitals`,
    `rays`, and `visible_entities`.
    """
    vitals = obs["vitals"]
    single = np.asarray(vitals).ndim == 1

    def _to_tensor(val, dtype: torch.dtype) -> torch.Tensor:
        t = torch.as_tensor(val, dtype=dtype, device=device)
        return t.unsqueeze(0) if single else t

    rays = obs["rays"]
    entities = obs["visible_entities"]

    return {
        "vitals": _to_tensor(vitals, torch.float32),
        "ray_distance": _to_tensor(rays["distance"], torch.float32),
        "ray_hit_type": _to_tensor(rays["hit_type"], torch.long),
        "entity_features": _to_tensor(entities["features"], torch.float32),
        "entity_type": _to_tensor(entities["entity_type"], torch.long),
        "entity_mask": _to_tensor(entities["mask"], torch.float32),
    }


def heads_to_buttons(move: int, turn: int, attack: int, use: int, weapon: int) -> np.ndarray:
    """Map the 5 discrete action heads to a (9,) int8 MultiBinary button array."""
    buttons = np.zeros(NUM_BUTTONS, dtype=np.int8)
    fwd, back, left, right = _MOVE_TABLE[int(move)]
    buttons[0] = fwd
    buttons[1] = back
    buttons[2] = left
    buttons[3] = right

    t = int(turn)
    if t == 1:
        buttons[4] = 1  # TURN_LEFT
    elif t == 2:
        buttons[5] = 1  # TURN_RIGHT

    buttons[6] = 1 if int(attack) else 0
    buttons[7] = 1 if int(use) else 0
    # SELECT_NEXT_WEAPON is never pressed: the bot kept cycling pistol -> fists. Doom already switches
    # to a better weapon automatically when one is picked up. (The weapon head is kept so old
    # checkpoints still load; its output is ignored.)
    buttons[8] = 0
    return buttons


class RayEncoder(nn.Module):
    """1D circular-padded CNN over 64 ego-centric rays."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        in_channels = 1 + NUM_RAY_HIT_TYPES  # distance + one-hot hit_type
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.GELU(),
        )
        # 64 -> 32 -> 16 spatial positions * 128 channels = 2048
        self.proj = nn.Sequential(
            nn.Linear(128 * 16, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, distance: torch.Tensor, hit_type: torch.Tensor) -> torch.Tensor:
        hit_oh = F.one_hot(hit_type.clamp(0, NUM_RAY_HIT_TYPES - 1), NUM_RAY_HIT_TYPES).float()
        x = torch.cat([distance.unsqueeze(-1), hit_oh], dim=-1)  # [B, 64, 7]
        x = x.transpose(1, 2)  # [B, 7, 64]
        x = self.conv(x).flatten(1)  # [B, 2048]
        return self.proj(x)


class EntityEncoder(nn.Module):
    """Attention pooling over up to 32 visible entities with padding mask."""

    def __init__(self, out_dim: int = 128, type_embed_dim: int = 32) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, type_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(NUM_ENTITY_FEATURES + type_embed_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )
        self.attn_score = nn.Linear(out_dim, 1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self, features: torch.Tensor, entity_type: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        emb = self.type_emb(entity_type.clamp(0, NUM_ENTITY_TYPES - 1))
        # Soft-compress relative velocities while preserving distance/angles
        feat_scaled = torch.cat(
            [features[..., :3], torch.tanh(features[..., 3:] / 20.0)], dim=-1
        )
        h = self.mlp(torch.cat([feat_scaled, emb], dim=-1))  # [B, 32, D]

        scores = self.attn_score(h).squeeze(-1)  # [B, 32]
        valid = mask > 0.5
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=-1)
        # Zero out weights completely when no entities are visible
        has_any = valid.any(dim=-1, keepdim=True)
        weights = torch.where(has_any, weights, torch.zeros_like(weights))

        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # [B, D]
        return self.norm(pooled)


class VitalsEncoder(nn.Module):
    """Small MLP over the 17 normalized player vitals."""

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor(
                [
                    100.0, 100.0, 7.0, 100.0,  # health, armor, weapon, weapon_ammo
                    100.0, 100.0, 50.0, 100.0,  # bullets, shells, rockets, cells
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # 6 keycards/skulls
                    20.0, 20.0, 20.0,  # vel_forward, vel_right, vel_up
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.net = nn.Sequential(
            nn.Linear(NUM_VITALS, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, vitals: torch.Tensor) -> torch.Tensor:
        return self.net(vitals / self.scale)


class DoomNet(nn.Module):
    """Recurrent actor-critic with episodic memory and privileged critic head."""

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__()
        cfg = cfg or {}
        self.hidden_size: int = int(cfg.get("hidden_size", 256))
        self.mem_slots: int = int(cfg.get("mem_slots", 64))
        self.mem_dim: int = int(cfg.get("mem_dim", 128))
        self.mem_write_every: int = int(cfg.get("mem_write_every", 8))
        self.priv_dim: int = int(cfg.get("priv_dim", PRIV_DIM))

        ray_dim = int(cfg.get("ray_dim", 128))
        entity_dim = int(cfg.get("entity_dim", 128))
        vitals_dim = int(cfg.get("vitals_dim", 64))

        self.ray_encoder = RayEncoder(ray_dim)
        self.entity_encoder = EntityEncoder(entity_dim)
        self.vitals_encoder = VitalsEncoder(vitals_dim)

        fused_dim = ray_dim + entity_dim + vitals_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
        )
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # Episodic memory projections
        self.mem_write = nn.Linear(self.hidden_size, self.mem_dim)
        self.mem_q = nn.Linear(self.hidden_size, self.mem_dim)
        self.mem_k = nn.Linear(self.mem_dim, self.mem_dim)
        self.mem_v = nn.Linear(self.mem_dim, self.mem_dim)
        self.mem_scale = 1.0 / math.sqrt(self.mem_dim)

        self.policy_proj = nn.Sequential(
            nn.Linear(self.hidden_size + self.mem_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
        )

        self.heads = nn.ModuleDict(
            {name: nn.Linear(self.hidden_size, size) for name, size in HEAD_SIZES.items()}
        )

        # Privileged critic: training only, skipped when priv is None
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size + self.priv_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def initial_state(self, batch: int, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        """Return zeroed recurrent and episodic memory state for `batch` envs."""
        return {
            "h": torch.zeros(1, batch, self.hidden_size, dtype=torch.float32, device=device),
            "mem": torch.zeros(batch, self.mem_slots, self.mem_dim, dtype=torch.float32, device=device),
            "mem_mask": torch.zeros(batch, self.mem_slots, dtype=torch.float32, device=device),
            "t": torch.zeros(batch, dtype=torch.long, device=device),
        }

    def _reset_done(self, state: dict[str, torch.Tensor], done: torch.Tensor) -> dict[str, torch.Tensor]:
        done_bool = done.bool()
        if not done_bool.any():
            return state
        keep_f = (~done_bool).float()
        return {
            "h": state["h"] * keep_f.view(1, -1, 1),
            "mem": state["mem"] * keep_f.view(-1, 1, 1),
            "mem_mask": state["mem_mask"] * keep_f.view(-1, 1),
            "t": state["t"] * (~done_bool).long(),
        }

    def _query_memory(
        self, gru_out: torch.Tensor, mem: torch.Tensor, mem_mask: torch.Tensor
    ) -> torch.Tensor:
        q = self.mem_q(gru_out).unsqueeze(1)  # [B, 1, M]
        k = self.mem_k(mem)  # [B, S, M]
        v = self.mem_v(mem)  # [B, S, M]

        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) * self.mem_scale  # [B, S]
        valid = mem_mask > 0.5
        scores = scores.masked_fill(~valid, -1e9)
        attn = torch.softmax(scores, dim=-1)

        has_mem = valid.any(dim=-1, keepdim=True)
        attn = torch.where(has_mem, attn, torch.zeros_like(attn))
        return torch.bmm(attn.unsqueeze(1), v).squeeze(1)  # [B, M]

    def _write_memory(
        self,
        gru_out: torch.Tensor,
        mem: torch.Tensor,
        mem_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        should_write = (t % self.mem_write_every) == 0  # [B]
        if should_write.any():
            slot = (t // self.mem_write_every) % self.mem_slots  # [B]
            slot_oh = F.one_hot(slot, self.mem_slots).float()  # [B, S]
            write_mask = slot_oh * should_write.float().unsqueeze(-1)  # [B, S]

            snapshot = torch.tanh(self.mem_write(gru_out)).unsqueeze(1)  # [B, 1, M]
            w = write_mask.unsqueeze(-1)  # [B, S, 1]
            mem = mem * (1.0 - w) + snapshot * w
            mem_mask = torch.maximum(mem_mask, write_mask)

        return mem, mem_mask, t + 1

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        state: dict[str, torch.Tensor],
        done: torch.Tensor,
        priv: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, dict[str, torch.Tensor]]:
        """Run one step across a batch of `B` environments."""
        state = self._reset_done(state, done)

        ray_feat = self.ray_encoder(obs["ray_distance"], obs["ray_hit_type"])
        ent_feat = self.entity_encoder(
            obs["entity_features"], obs["entity_type"], obs["entity_mask"]
        )
        vit_feat = self.vitals_encoder(obs["vitals"])

        fused = self.fusion(torch.cat([ray_feat, ent_feat, vit_feat], dim=-1))
        gru_seq, new_h = self.gru(fused.unsqueeze(1), state["h"])
        gru_out = gru_seq.squeeze(1)

        mem_read = self._query_memory(gru_out, state["mem"], state["mem_mask"])
        new_mem, new_mem_mask, new_t = self._write_memory(
            gru_out, state["mem"], state["mem_mask"], state["t"]
        )

        policy_feat = self.policy_proj(torch.cat([gru_out, mem_read], dim=-1))
        logits = {name: head(policy_feat) for name, head in self.heads.items()}

        value = None
        if priv is not None:
            value = self.critic(torch.cat([policy_feat, priv], dim=-1)).squeeze(-1)

        new_state = {
            "h": new_h,
            "mem": new_mem,
            "mem_mask": new_mem_mask,
            "t": new_t,
        }
        return logits, value, new_state
