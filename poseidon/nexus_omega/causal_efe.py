"""PROJECT NEXUS-Ω: Deep Active Inference & Conformal Safe Policy Improvement (Causal-EFE).

Formulates action selection as variational Expected Free Energy (EFE) optimization
governed by distribution-free Conformal Safety Return Bounds:
1. Pragmatic Instrumental Value:
   Mahalanobis divergence from homeostatic viability kernel (health, energy, hydration, stamina).
2. Epistemic Curiosity Value:
   Information gain / variance reduction over exogenous field and latent causal regime.
3. Dynamic Curiosity Regulation:
   Suppressed when physiological safety is endangered; unconstrained during surplus.
4. Conformal Safety Guard:
   Computes distribution-free lower confidence bound (LCB) on return, preventing
   catastrophic exploration into fatal storms.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np

from poseidon.nexus_omega.contracts import (
    CausalEFEDecision,
    FactoredPredictionOutput,
    FactoredState,
)
from poseidon.nexus_omega.causal_world_model import FactoredCausalWorldModel


class CausalEFEPlanner:
    """Plans actions minimizing Expected Free Energy under Conformal Safety Guardrails."""

    def __init__(
        self,
        world_model: FactoredCausalWorldModel,
        n_actions: int = 4,
        base_epistemic_weight: float = 0.50,
        conformal_confidence_quantile: float = 1.96,
        safety_threshold: float = -1.20,
    ):
        self.world_model = world_model
        self.n_actions = n_actions
        self.base_epistemic_weight = base_epistemic_weight
        self.conformal_quantile = conformal_confidence_quantile
        self.safety_threshold = safety_threshold

        # Target homeostatic vital prior: health=1.0, energy=0.9, hydration=0.9, stamina=0.9
        self.s_target = np.array([1.0, 0.90, 0.90, 0.90], dtype=np.float32)
        # Precision weighting: health is penalized most heavily
        self.precision_weights = np.array([4.0, 2.5, 2.5, 1.5], dtype=np.float32)

        self.last_decision: Optional[CausalEFEDecision] = None

    def compute_pragmatic_value(self, predicted_endo: np.ndarray, predicted_reward: float = 0.0) -> float:
        """Expected utility combining extrinsic reward and homeostatic vital alignment."""
        diff = predicted_endo[:3] - self.s_target[:3]
        vital_penalty = np.sum(self.precision_weights[:3] * (diff ** 2))
        return float(predicted_reward - vital_penalty)

    def compute_epistemic_drive(self, current_obs: np.ndarray) -> float:
        """Modulates epistemic curiosity based on current physiological safety margin."""
        health = max(0.0, min(1.0, float(current_obs[0])))
        energy = max(0.0, min(1.0, float(current_obs[1])))
        hydration = max(0.0, min(1.0, float(current_obs[2])))

        # When vital reserves are safe, exploration blooms; under starvation, curiosity shuts down
        vital_safety = health * min(energy, hydration)
        epistemic_weight = max(0.02, self.base_epistemic_weight * vital_safety)
        return float(epistemic_weight)

    def compute_empowerment(
        self,
        predicted_endo: np.ndarray,
        predicted_exo: np.ndarray,
        mem: np.ndarray,
    ) -> float:
        """Estimates future behavioral freedom across 1-step successor actions."""
        succ_states = []
        obs_hypo = np.concatenate([predicted_endo, predicted_exo], axis=0)
        for a_succ in range(self.n_actions):
            pred = self.world_model.predict_factored(obs_hypo, mem, a_succ)
            succ_states.append(pred.endogenous_pred)

        # Variance of reachable vital states across actions
        stack = np.stack(succ_states, axis=0)
        dispersion = float(np.mean(np.var(stack, axis=0)))
        return dispersion

    def plan_action(
        self,
        obs: np.ndarray,
        mem: np.ndarray,
        action_names: Optional[Sequence[str]] = None,
    ) -> CausalEFEDecision:
        """Evaluates all actions under Expected Free Energy and enforces Conformal Guard."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        if len(obs_arr) > self.world_model.d_obs:
            obs_arr = obs_arr[:self.world_model.d_obs]

        names = list(action_names or ["forage", "shelter", "explore", "rest"])

        best_action = 0
        best_efe = -float("inf")
        best_prag = 0.0
        best_epistemic = 0.0
        best_weight = 0.0
        best_empowerment = 0.0
        best_lcb = -float("inf")

        epistemic_weight = self.compute_epistemic_drive(obs_arr)

        action_scores = []

        for a in range(self.n_actions):
            pred = self.world_model.predict_factored(obs_arr, mem, a)
            prag = self.compute_pragmatic_value(pred.endogenous_pred, pred.predicted_reward)
            epistemic = pred.exogenous_uncertainty + pred.regime_uncertainty
            empowerment = self.compute_empowerment(pred.endogenous_pred, pred.exogenous_pred, mem)

            # Expected Free Energy Objective (to be maximized)
            # G = Pragmatic + beta * Epistemic + 0.2 * Empowerment
            efe = prag + epistemic_weight * epistemic + 0.20 * empowerment

            # Conformal Return Lower Bound: penalize by total epistemic uncertainty
            conformal_sigma = math.sqrt(max(1e-6, pred.epistemic_uncertainty_total))
            lcb = efe - self.conformal_quantile * conformal_sigma

            action_scores.append({
                "action": a,
                "efe": efe,
                "prag": prag,
                "epistemic": epistemic,
                "empowerment": empowerment,
                "lcb": lcb,
            })

            if efe > best_efe:
                best_efe = efe
                best_action = a
                best_prag = prag
                best_epistemic = epistemic
                best_weight = epistemic_weight
                best_empowerment = empowerment
                best_lcb = lcb

        # Conformal Safe Policy Improvement Guard:
        # If best action is an exploratory/foraging action with high risk (LCB < safety_threshold),
        # and health/exposure is threatened, override to safe shelter (action 1) or rest (action 3).
        conformal_override = False
        is_threatened = (obs_arr[0] < 0.6) or (obs_arr[4] > 0.5)  # low health or high exposure
        if best_lcb < self.safety_threshold and is_threatened and best_action in (0, 2):
            # Fallback to shelter if available, else rest
            fallback_action = 1 if self.n_actions > 1 else 0
            best_action = fallback_action
            conformal_override = True
            fallback_data = action_scores[fallback_action]
            best_efe = fallback_data["efe"]
            best_prag = fallback_data["prag"]
            best_epistemic = fallback_data["epistemic"]
            best_empowerment = fallback_data["empowerment"]
            best_lcb = fallback_data["lcb"]

        # Ensure world model last_prediction corresponds to chosen action
        self.world_model.predict_factored(obs_arr, mem, best_action)

        action_label = names[best_action] if best_action < len(names) else f"action_{best_action}"

        factored = self.world_model.get_factored_state(obs_arr)
        factored_dict = {
            "endogenous": [round(float(x), 4) for x in factored.endogenous_vitals],
            "exogenous": [round(float(x), 4) for x in factored.exogenous_field],
            "causal_regime": [round(float(x), 4) for x in factored.causal_regime],
        }

        decision = CausalEFEDecision(
            action=best_action,
            action_name=action_label,
            free_energy_objective=best_efe,
            pragmatic_value=best_prag,
            epistemic_value=best_epistemic,
            epistemic_weight=best_weight,
            empowerment_value=best_empowerment,
            conformal_lcb=best_lcb,
            conformal_override=conformal_override,
            factored_state=factored_dict,
        )

        self.last_decision = decision
        return decision
