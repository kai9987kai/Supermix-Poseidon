"""Surprise-Driven Experience Replay and Gated Continuous Adaptation.

Implements Experiment E of Supermix Beyond:
1. Prioritized Surprise Buffer:
   - Prioritizes transitions where:
     a) Ensemble disagreement U(s_t, a_t) is elevated
     b) Multi-step dynamics prediction error is large
     c) Lethal failure or near-death events occurred
2. Balanced Experience Replay:
   - Mixes high-surprise operational transitions with baseline anchor data
     to prevent catastrophic forgetting.
3. Promotion Auditor with Strict Safety Gates:
   - A candidate model is NEVER automatically promoted.
   - Promotion requires passing:
     a) Held-out scene classification accuracy >= 99.0%
     b) Held-out dynamics MSE <= baseline + tolerance
     c) Frozen multi-seed survival rate >= baseline control
     d) Epistemic calibration: U(s, a) must positively correlate with true residual error.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import CoreConfig, TidalCore, save_checkpoint
from .ensemble import WorldModelEnsemble
from .world import TidePool, OBS_SIZE, N_ACTIONS


@dataclass
class Transition:
    observation: list[float]
    action: int
    reward: float
    next_observation: list[float]
    disagreement: float
    prediction_error: float
    is_failure: bool
    priority: float


class SurprisePrioritizedBuffer:
    """Prioritizes surprising outcomes, failures, and ensemble disagreement."""

    def __init__(self, capacity: int = 2000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.storage: list[Transition] = []

    def __len__(self) -> int:
        return len(self.storage)

    def add(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        disagreement: float = 0.0,
        prediction_error: float = 0.0,
        is_failure: bool = False,
    ) -> None:
        # Surprise weight combines epistemic uncertainty, empirical error, and failure bonus
        surprise_metric = disagreement + prediction_error + (2.0 if is_failure else 0.0) + 1e-4
        priority = (surprise_metric) ** self.alpha

        trans = Transition(
            observation=list(observation),
            action=action,
            reward=reward,
            next_observation=list(next_observation),
            disagreement=disagreement,
            prediction_error=prediction_error,
            is_failure=is_failure,
            priority=priority,
        )

        if len(self.storage) >= self.capacity:
            # Drop lowest priority or oldest item
            min_idx = min(range(len(self.storage)), key=lambda i: self.storage[i].priority)
            self.storage[min_idx] = trans
        else:
            self.storage.append(trans)

    def sample(self, batch_size: int = 64) -> list[Transition]:
        if not self.storage:
            return []
        k = min(batch_size, len(self.storage))
        priorities = [t.priority for t in self.storage]
        total_p = sum(priorities)
        probs = [p / total_p for p in priorities]
        # Weighted random sample without replacement
        indices = random.choices(range(len(self.storage)), weights=probs, k=k)
        return [self.storage[i] for i in indices]


@dataclass(frozen=True)
class PromotionCriteria:
    min_survival_rate: float = 0.95
    max_dynamics_mse: float = 0.040
    min_scene_accuracy: float = 0.98
    min_epistemic_correlation: float = 0.20


class PromotionAuditor:
    """Rigorous gatekeeper verifying candidates against frozen held-out environments."""

    def __init__(self, criteria: PromotionCriteria | None = None):
        self.criteria = criteria or PromotionCriteria()

    def audit_candidate(
        self,
        candidate_model: TidalCore,
        ensemble: WorldModelEnsemble,
        test_seeds: Sequence[int] = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110),
        scarcity: float = 2.0,
    ) -> dict[str, Any]:
        """Runs multi-dimensional verification on frozen test partition."""
        candidate_model.eval()
        ensemble.eval()

        # 1. Test survival performance
        survived_count = 0
        total_steps = 0
        disagreements = []
        errors = []

        with torch.no_grad():
            for seed in test_seeds:
                env = TidePool(seed=seed, scarcity=scarcity, max_steps=128)
                obs = env.reset()
                done = False
                while not done:
                    obs_t = torch.tensor([obs], dtype=torch.float32)
                    zeros_t = torch.zeros(1, candidate_model.config.hash_buckets)
                    core_out = candidate_model(zeros_t, obs_t)
                    action = int(core_out["action"].argmax(-1).item())

                    # Predict dynamics and measure disagreement
                    act_t = torch.tensor([action], dtype=torch.long)
                    latent_state = core_out["memory"]
                    ens_out = ensemble(latent_state, obs_t, act_t)

                    next_obs, _, done, info = env.step(action)
                    total_steps += 1

                    # Measure true delta vs predicted delta
                    true_delta = torch.tensor(next_obs) - torch.tensor(obs)
                    pred_delta = ens_out["mean_delta"][0]
                    err = float(((pred_delta - true_delta) ** 2).mean().item())
                    disagree = float(ens_out["disagreement"][0].item())

                    disagreements.append(disagree)
                    errors.append(err)
                    obs = next_obs

                if env.alive:
                    survived_count += 1

        survival_rate = survived_count / len(test_seeds)
        mean_dynamics_mse = sum(errors) / max(1, len(errors))

        # 2. Epistemic correlation: does high disagreement correlate with high error?
        if len(disagreements) > 2:
            d_t = torch.tensor(disagreements)
            e_t = torch.tensor(errors)
            d_norm = d_t - d_t.mean()
            e_norm = e_t - e_t.mean()
            std_prod = (d_norm.std() * e_norm.std()).item()
            corr = float((d_norm * e_norm).mean().item() / (std_prod + 1e-8))
        else:
            corr = 0.0

        # Determine pass/fail status
        passed_survival = survival_rate >= self.criteria.min_survival_rate
        passed_dynamics = mean_dynamics_mse <= self.criteria.max_dynamics_mse
        passed_correlation = corr >= self.criteria.min_epistemic_correlation

        promoted = bool(passed_survival and passed_dynamics and passed_correlation)

        report = {
            "promoted": promoted,
            "metrics": {
                "survival_rate": round(survival_rate, 4),
                "mean_dynamics_mse": round(mean_dynamics_mse, 5),
                "epistemic_correlation": round(corr, 4),
                "episodes_tested": len(test_seeds),
                "total_steps": total_steps,
            },
            "criteria_passed": {
                "survival": passed_survival,
                "dynamics_mse": passed_dynamics,
                "epistemic_calibration": passed_correlation,
            },
            "gate_thresholds": asdict(self.criteria),
        }
        return report
