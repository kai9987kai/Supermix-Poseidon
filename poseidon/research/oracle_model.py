"""Small ensemble world model used by the opt-in ORACLE research pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..world import N_ACTIONS, OBS_SIZE

ORACLE_SCHEMA = "poseidon-oracle-ensemble-v1"


@dataclass(frozen=True)
class OracleConfig:
    hidden_size: int = 128
    ensemble_size: int = 5
    observation_size: int = OBS_SIZE
    action_count: int = N_ACTIONS

    def validate(self) -> None:
        if type(self.hidden_size) is not int or not 16 <= self.hidden_size <= 1024:
            raise ValueError("hidden_size must be an integer in [16, 1024]")
        if type(self.ensemble_size) is not int or not 2 <= self.ensemble_size <= 16:
            raise ValueError("ensemble_size must be an integer in [2, 16]")
        if (type(self.observation_size) is not int or type(self.action_count) is not int
                or self.observation_size != OBS_SIZE or self.action_count != N_ACTIONS):
            raise ValueError("ORACLE config does not match the registered TidePool contract")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


class OracleMember(nn.Module):
    """Predicts observation delta, scalar reward, and terminal-death probability."""

    def __init__(self, config: OracleConfig):
        super().__init__()
        self.action_embedding = nn.Embedding(config.action_count, 16)
        self.encoder = nn.Sequential(
            nn.Linear(config.observation_size + 16, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(config.hidden_size, 2 * config.observation_size)
        self.reward_head = nn.Linear(config.hidden_size, 2)
        self.death_head = nn.Linear(config.hidden_size, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.action_embedding(action)
        hidden = self.encoder(torch.cat((observation, embedded), dim=-1))
        delta = self.delta_head(hidden).reshape(-1, 2, observation.shape[-1])
        reward = self.reward_head(hidden)
        return {
            "delta_mean": delta[:, 0],
            "delta_log_variance": delta[:, 1].clamp(-8.0, 3.0),
            "reward_mean": reward[:, 0],
            "reward_log_variance": reward[:, 1].clamp(-8.0, 3.0),
            "death_logit": self.death_head(hidden).squeeze(-1),
        }


class OracleEnsemble(nn.Module):
    """Episode-bootstrap ensemble with aleatoric and between-member variance."""

    def __init__(self, config: OracleConfig, seed: int):
        super().__init__()
        config.validate()
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be a nonnegative integer")
        self.config = config
        members = []
        # Initialization is isolated so fitting ORACLE never perturbs global RNG state.
        with torch.random.fork_rng(devices=[]):
            for index in range(config.ensemble_size):
                torch.manual_seed(seed + 104729 * index)
                members.append(OracleMember(config))
        self.members = nn.ModuleList(members)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[1] != self.config.observation_size:
            raise ValueError("observation must have shape [batch, observation_size]")
        if not observation.is_floating_point() or not bool(torch.isfinite(observation).all()):
            raise ValueError("observation must contain finite floating point values")
        if action.ndim != 1 or action.shape[0] != observation.shape[0]:
            raise ValueError("action must have shape [batch]")
        if action.dtype != torch.long or bool(((action < 0) | (action >= self.config.action_count)).any()):
            raise ValueError("action tensor contains an invalid action")
        outputs = [member(observation, action) for member in self.members]
        return {name: torch.stack([output[name] for output in outputs], dim=0)
                for name in outputs[0]}

    @staticmethod
    def summarize(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        delta_mean = output["delta_mean"].mean(dim=0)
        reward_mean = output["reward_mean"].mean(dim=0)
        delta_epistemic = output["delta_mean"].var(dim=0, unbiased=False)
        reward_epistemic = output["reward_mean"].var(dim=0, unbiased=False)
        delta_aleatoric = output["delta_log_variance"].exp().mean(dim=0)
        reward_aleatoric = output["reward_log_variance"].exp().mean(dim=0)
        return {
            "delta_mean": delta_mean,
            "delta_epistemic_variance": delta_epistemic,
            "delta_aleatoric_variance": delta_aleatoric,
            "delta_total_variance": (delta_epistemic + delta_aleatoric).clamp_min(1e-8),
            "reward_mean": reward_mean,
            "reward_epistemic_variance": reward_epistemic,
            "reward_aleatoric_variance": reward_aleatoric,
            "reward_total_variance": (reward_epistemic + reward_aleatoric).clamp_min(1e-8),
            "death_probability": output["death_logit"].sigmoid().mean(dim=0),
        }
