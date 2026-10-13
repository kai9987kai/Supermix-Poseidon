"""PROJECT NEXUS-Ω: Guided Counterfactual Trajectory Diffusion Kernel.

Implements score-based trajectory imagination over multi-step horizons,
combining Invariant Causal Mechanism (ICM) dynamics scores with homeostatic
vital guidance and multi-step conformal risk bounds.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .contracts import FactoredState, ImaginedTrajectory, MacroIntent
from .causal_world_model import FactoredCausalWorldModel


class TrajectoryDiffusionImaginer:
    """Generates and refines coherent counterfactual multi-step trajectory rollouts."""

    def __init__(
        self,
        world_model: FactoredCausalWorldModel,
        default_horizon: int = 8,
        diffusion_steps: int = 5,
        guidance_scale: float = 1.25,
        conformal_quantile: float = 1.96,
        base_seed: int = 42,
    ):
        self.world_model = world_model
        self.default_horizon = default_horizon
        self.diffusion_steps = diffusion_steps
        self.guidance_scale = guidance_scale
        self.conformal_quantile = conformal_quantile
        self.rng = np.random.RandomState(base_seed)

        # Homeostatic target vector: health=1.0, energy=0.9, hydration=0.9, exposure=0.0
        self.s_homeo_target = np.array([1.0, 0.90, 0.90, 0.0], dtype=np.float32)

    def _intent_to_action_prior(self, intent: MacroIntent) -> np.ndarray:
        """Returns action selection probability distribution for a given macro-intent."""
        # Actions: 0: Forage, 1: Shelter, 2: Explore, 3: Rest
        if intent == MacroIntent.FORAGE_EXPEDITION:
            return np.array([0.70, 0.05, 0.15, 0.10], dtype=np.float32)
        elif intent == MacroIntent.DIAPAUSE_HIBERNATION:
            return np.array([0.05, 0.70, 0.05, 0.20], dtype=np.float32)
        elif intent == MacroIntent.HABITAT_RELOCATION:
            return np.array([0.15, 0.05, 0.70, 0.10], dtype=np.float32)
        elif intent == MacroIntent.EMERGENCY_FORTIFICATION:
            return np.array([0.05, 0.80, 0.05, 0.10], dtype=np.float32)
        return np.ones(4, dtype=np.float32) / 4.0

    def compute_homeostatic_guidance(self, state: np.ndarray) -> np.ndarray:
        """Computes gradient guidance direction pushing state toward homeostatic envelope."""
        grad = np.zeros_like(state)
        # Endogenous vitals: health, energy, hydration
        for i in range(min(3, len(state))):
            diff = self.s_homeo_target[i] - state[i]
            grad[i] = diff * (2.0 if i == 0 else 1.0)
        # Exogenous exposure (index 4 in 8-D obs, or index 0 in s_exo)
        if len(state) > 4:
            # Penalize exposure (index 4)
            grad[4] = -1.5 * max(0.0, state[4] - 0.2)
        return grad

    def imagine_trajectory(
        self,
        current_obs: np.ndarray | Sequence[float],
        current_mem: np.ndarray | Sequence[float],
        target_intent: MacroIntent = MacroIntent.FORAGE_EXPEDITION,
        horizon: Optional[int] = None,
    ) -> ImaginedTrajectory:
        """Synthesizes score-guided counterfactual rollout over horizon H."""
        H = horizon or self.default_horizon
        obs_curr = np.asarray(current_obs, dtype=np.float32)[:self.world_model.d_obs]
        mem_curr = np.asarray(current_mem, dtype=np.float32)

        action_prior = self._intent_to_action_prior(target_intent)

        # 1. Initialize candidate actions across horizon
        # For storm or severe threat, prioritize shelter
        has_storm = (
            (len(obs_curr) > 3 and obs_curr[3] > 0.8)
            or (len(mem_curr) > 0 and float(mem_curr[0]) > 0.1)
        )
        if has_storm or target_intent in (MacroIntent.DIAPAUSE_HIBERNATION, MacroIntent.EMERGENCY_FORTIFICATION):
            action_seq = [1 if t < 3 else int(np.argmax(action_prior)) for t in range(H)]
        else:
            action_seq = [int(np.argmax(action_prior)) for _ in range(H)]

        # 2. Sequential multi-step rollout using Factored Causal World Model
        states: List[np.ndarray] = [obs_curr.copy()]
        rewards: List[float] = []
        epistemic_vars: List[float] = []

        s_t = obs_curr.copy()
        mem_t = mem_curr.copy()

        total_guidance_norm = 0.0

        for t in range(H):
            act = action_seq[t]
            pred = self.world_model.predict_factored(s_t, mem_t, act)

            # Score-guided diffusion update on predicted state
            guidance_vec = self.compute_homeostatic_guidance(pred.predicted_obs)
            total_guidance_norm += float(np.linalg.norm(guidance_vec))

            # Apply reverse diffusion score adjustment:
            # s_{t+1} = s_pred + alpha * guidance
            alpha = 0.05 * self.guidance_scale
            refined_next_obs = pred.predicted_obs + alpha * guidance_vec

            # Clip vitals to physically valid ranges [0, 1]
            refined_next_obs[0] = max(0.0, min(1.0, refined_next_obs[0]))
            refined_next_obs[1] = max(0.0, min(1.0, refined_next_obs[1]))
            refined_next_obs[2] = max(0.0, min(1.0, refined_next_obs[2]))

            states.append(refined_next_obs.copy())
            rewards.append(float(pred.predicted_reward))
            epistemic_vars.append(float(pred.epistemic_uncertainty_total))

            s_t = refined_next_obs.copy()
            # Advance working memory decay
            if len(mem_t) > 0:
                mem_t[0] = max(0.0, float(mem_t[0]) - 1.0)

        cumulative_r = float(np.sum(rewards))

        # 3. Compute multi-step Conformal Lower Confidence Bound
        total_epistemic_sigma = math.sqrt(max(1e-6, float(np.sum(epistemic_vars))))
        conformal_lcb = cumulative_r - self.conformal_quantile * total_epistemic_sigma

        return ImaginedTrajectory(
            horizon=H,
            macro_intent=target_intent,
            states=states[1:],  # H predicted future states
            actions=action_seq,
            rewards=rewards,
            epistemic_uncertainties=epistemic_vars,
            cumulative_reward=cumulative_r,
            conformal_trajectory_lcb=conformal_lcb,
            score_guidance_norm=float(total_guidance_norm / max(1, H)),
        )

    def evaluate_all_intents(
        self,
        obs: np.ndarray,
        mem: np.ndarray,
        horizon: Optional[int] = None,
    ) -> Dict[str, ImaginedTrajectory]:
        """Imagines and compares counterfactual trajectories across all four macro-intents."""
        results: Dict[str, ImaginedTrajectory] = {}
        for intent in MacroIntent:
            traj = self.imagine_trajectory(obs, mem, target_intent=intent, horizon=horizon)
            results[intent.value] = traj
        return results
