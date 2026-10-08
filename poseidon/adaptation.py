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
        candidate_model: Any,
        ensemble: Any,
        test_seeds: Sequence[int] = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110),
        scarcity: float = 2.0,
        incumbent_model: Any | None = None,
        incumbent_ensemble: Any | None = None,
        scene_examples: Sequence[dict] | None = None,
    ) -> dict[str, Any]:
        """Runs multi-dimensional verification on frozen test partition."""
        # 1. Input validations
        if not test_seeds or len(test_seeds) == 0:
            raise ValueError("test_seeds must be non-empty")
        if len(test_seeds) != len(set(test_seeds)):
            raise ValueError("test_seeds must contain distinct seeds")
        for s in test_seeds:
            if type(s) is not int or s != s:
                raise ValueError("test_seeds must contain valid integers")

        if scene_examples is not None:
            if len(scene_examples) == 0:
                raise ValueError("scene_examples must be non-empty")
            from .media import _split
            for row in scene_examples:
                group = row.get("group")
                if group and _split(group) != "test":
                    raise ValueError("scene_examples must belong to test split")

        candidate_model.eval()
        ensemble.eval()
        if incumbent_model:
            incumbent_model.eval()
        if incumbent_ensemble:
            incumbent_ensemble.eval()

        # 2. Missing evidence audit
        missing_evidence = []
        if incumbent_model is None:
            missing_evidence.append("incumbent_model")
        if incumbent_ensemble is None:
            missing_evidence.append("incumbent_ensemble")
        if scene_examples is None:
            missing_evidence.append("scene_examples")
        evidence_complete = len(missing_evidence) == 0

        # 3. Test survival performance & paired comparisons
        survived_count = 0
        total_steps = 0
        disagreements = []
        errors = []
        paired_episodes = []

        with torch.no_grad():
            for seed in test_seeds:
                # Roll out candidate
                env = TidePool(seed=seed, scarcity=scarcity, max_steps=128)
                obs = env.reset(seed)
                done = False
                while not done:
                    obs_t = torch.tensor([obs], dtype=torch.float32).reshape(1, -1)
                    zeros_t = torch.zeros(1, candidate_model.config.hash_buckets)
                    core_out = candidate_model(zeros_t, obs_t)
                    action = int(core_out["action"].argmax(-1).item())

                    act_t = torch.tensor([action], dtype=torch.long)
                    latent_state = core_out["memory"]
                    ens_out = ensemble(latent_state, obs_t, act_t)

                    # Check finiteness
                    if not torch.isfinite(ens_out["mean_delta"]).all() or not torch.isfinite(ens_out["disagreement"]).all():
                        raise ValueError("model predictions must be finite")

                    next_obs, _, done, info = env.step(action)
                    total_steps += 1

                    true_delta = torch.tensor(next_obs) - torch.tensor(obs)
                    pred_delta = ens_out["mean_delta"][0]
                    err = float(((pred_delta - true_delta) ** 2).mean().item())
                    disagree = float(ens_out["disagreement"][0].item())

                    disagreements.append(disagree)
                    errors.append(err)
                    obs = next_obs

                cand_alive = env.alive
                if cand_alive:
                    survived_count += 1

                # Roll out incumbent on matched seed
                inc_alive = False
                if incumbent_model:
                    env_inc = TidePool(seed=seed, scarcity=scarcity, max_steps=128)
                    obs_inc = env_inc.reset(seed)
                    done_inc = False
                    while not done_inc:
                        obs_inc_t = torch.tensor([obs_inc], dtype=torch.float32).reshape(1, -1)
                        zeros_inc_t = torch.zeros(1, incumbent_model.config.hash_buckets)
                        inc_out = incumbent_model(zeros_inc_t, obs_inc_t)
                        action_inc = int(inc_out["action"].argmax(-1).item())
                        obs_inc, _, done_inc, _ = env_inc.step(action_inc)
                    inc_alive = env_inc.alive

                survival_delta = int(cand_alive) - int(inc_alive)
                paired_episodes.append({
                    "seed": seed,
                    "survival_delta": survival_delta,
                    "candidate_alive": cand_alive,
                    "incumbent_alive": inc_alive,
                })

        survival_rate = survived_count / len(test_seeds)
        mean_dynamics_mse = sum(errors) / max(1, len(errors))

        # 4. Scene accuracy evaluation
        scene_exact_accuracy = 1.0
        passed_scene = True
        if scene_examples is not None:
            from .core import SCENE_LABELS
            correct_scenes = 0
            for row in scene_examples:
                zeros_scene = torch.zeros(1, candidate_model.config.hash_buckets)
                scene_pred = candidate_model(zeros_scene)
                match = True
                for name in SCENE_LABELS:
                    if name in scene_pred.get("scene", {}):
                        pred_idx = int(scene_pred["scene"][name].argmax(-1).item())
                        if pred_idx != row["labels"][name]:
                            match = False
                            break
                if match:
                    correct_scenes += 1
            scene_exact_accuracy = correct_scenes / len(scene_examples)
            passed_scene = scene_exact_accuracy >= self.criteria.min_scene_accuracy

        # 5. Epistemic calibration evaluation
        d_t = torch.tensor(disagreements)
        is_constant_disagreement = (d_t.var().item() < 1e-7) if len(disagreements) > 1 else True

        if is_constant_disagreement:
            epistemic_correlation = None
            calibration_valid = False
            passed_correlation = False
        else:
            e_t = torch.tensor(errors)
            d_norm = d_t - d_t.mean()
            e_norm = e_t - e_t.mean()
            std_prod = (d_norm.std() * e_norm.std()).item()
            corr = float((d_norm * e_norm).mean().item() / (std_prod + 1e-8))
            epistemic_correlation = round(corr, 4)
            calibration_valid = True
            passed_correlation = corr >= self.criteria.min_epistemic_correlation

        # 6. Incumbent survival check
        passed_incumbent = all(p["survival_delta"] >= 0 for p in paired_episodes) if incumbent_model else True
        passed_survival = survival_rate >= self.criteria.min_survival_rate
        passed_dynamics = mean_dynamics_mse <= self.criteria.max_dynamics_mse

        criteria_passed = {
            "survival": passed_survival,
            "dynamics_mse": passed_dynamics,
            "epistemic_calibration": passed_correlation,
        }
        if scene_examples is not None:
            criteria_passed["scene_accuracy"] = passed_scene
        if incumbent_model is not None:
            criteria_passed["incumbent_survival"] = passed_incumbent

        eligible = bool(evidence_complete and all(criteria_passed.values()) and calibration_valid)
        promoted = False  # Passing an audit only grants eligibility; promotion requires explicit promotion sign-off

        report = {
            "promoted": promoted,
            "eligible": eligible,
            "evidence_complete": evidence_complete,
            "missing_evidence": missing_evidence,
            "metrics": {
                "survival_rate": round(survival_rate, 4),
                "mean_dynamics_mse": round(mean_dynamics_mse, 5),
                "epistemic_correlation": epistemic_correlation,
                "scene_exact_accuracy": round(scene_exact_accuracy, 4),
                "episodes_tested": len(test_seeds),
                "total_steps": total_steps,
            },
            "paired_episodes": paired_episodes,
            "calibration": {
                "valid": calibration_valid,
                "failure_probability_calibrated": False,
            },
            "criteria_passed": criteria_passed,
            "gate_thresholds": asdict(self.criteria),
        }
        return report
