"""Recurrent PPO and a small training-only reference policy."""
import torch
from torch import nn


class ReferencePolicy(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        rays = observation_space["rays"]["distance"].shape[0]
        entities, features = observation_space["visible_entities"]["features"].shape
        vitals = observation_space["vitals"].shape[0]
        self.encoder = nn.Sequential(nn.Linear(vitals + 2 * rays + entities * (features + 2), 128), nn.Tanh())
        self.memory = nn.GRUCell(128, 128)
        self.actor = nn.Linear(128, 16)
        self.critic = nn.Sequential(nn.Linear(128 + 16, 128), nn.Tanh(), nn.Linear(128, 1))

    def initial_state(self, batch):
        return {"h": self.actor.weight.new_zeros(1, batch, 128)}

    def forward(self, obs, state, done, priv=None):
        rays = obs["rays"]
        entities = obs["visible_entities"]
        mask = entities["mask"].float()
        features = torch.cat([
            obs["vitals"].float().clamp(-200, 200) / 100,
            rays["distance"].float(),
            rays["hit_type"].float() / 5,
            (entities["features"].float().clamp(-10, 10) * mask[..., None]).flatten(1),
            entities["entity_type"].float() / 12,
            mask,
        ], dim=-1)
        hidden = state["h"][0].masked_fill(done[:, None], 0)
        hidden = self.memory(self.encoder(features), hidden)
        chunks = self.actor(hidden).split(HEAD_SIZES, dim=-1)
        logits = dict(zip(HEADS, chunks, strict=True))
        value = None if priv is None else self.critic(torch.cat((hidden, priv), dim=-1)).squeeze(-1)
        return logits, value, {"h": hidden[None]}


def tree_map(fn, tree):
    if isinstance(tree, dict):
        return {key: tree_map(fn, value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [tree_map(fn, value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, value) for value in tree)
    return fn(tree)


HEADS = ("move", "turn", "attack", "use", "weapon")
HEAD_SIZES = (7, 3, 2, 2, 2)


def distributions(logits):
    return [torch.distributions.Categorical(logits=logits[head], validate_args=False)
            for head in HEADS]


def log_probability(dists, actions):
    return torch.stack([dist.log_prob(action) for dist, action in zip(
        dists, actions.unbind(-1), strict=True,
    )]).sum(dim=0)


def advantages(rewards, values, next_values, boundaries, gamma=0.995, decay=0.95):
    result = torch.empty_like(values)
    carry = torch.zeros_like(values[0])
    for step in reversed(range(len(values))):
        delta = rewards[step] + gamma * next_values[step] - values[step]
        carry = delta + gamma * decay * (~boundaries[step]) * carry
        result[step] = carry
    return result


def update(model, optimizer, batches, sequence_states, seq_len, epochs=4):
    """Shuffle fixed-length sequences and retain their starting GRU and episodic state."""
    adv = batches["advantages"]
    returns = adv + batches["values"]
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    sequence_count, workers = len(sequence_states), adv.shape[1]
    saved = {
        key: torch.cat([state[key] for state in sequence_states], dim=1 if key == "h" else 0)
        for key in sequence_states[0]
    }
    metrics = []
    for _ in range(epochs):
        shuffled = torch.randperm(sequence_count * workers, device=adv.device)
        for indices in shuffled.tensor_split(min(4, len(shuffled))):
            state = {key: (value[:, indices] if key == "h" else value[indices]).clone()
                     for key, value in saved.items()}
            columns = indices % workers
            starts = indices // workers * seq_len
            log_probs, entropies, values = [], [], []
            for offset in range(seq_len):
                rows = starts + offset
                obs = tree_map(lambda tensor: tensor[rows, columns], batches["obs"])
                logits, value, state = model(obs, state, batches["starts"][rows, columns],
                                             batches["privileged"][rows, columns])
                dists = distributions(logits)
                log_probs.append(log_probability(dists, batches["actions"][rows, columns]))
                entropies.append(torch.stack([dist.entropy() for dist in dists]).sum(dim=0))
                values.append(value.reshape(-1))
            rows = starts[None, :] + torch.arange(seq_len, device=adv.device)[:, None]
            ratio = (torch.stack(log_probs) - batches["log_probs"][rows, columns]).exp()
            advantage = adv[rows, columns]
            policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
            value_loss = (torch.stack(values) - returns[rows, columns]).square().mean()
            entropy = torch.stack(entropies).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            metrics.append(torch.stack((policy_loss, value_loss, entropy)).detach())
    return torch.stack(metrics).mean(dim=0).tolist()
