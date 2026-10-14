"""PROJECT NEXUS-Ω: Autonomous Agent with Causal Memory & Uncertainty-Aware Planning.

Integrates CausalMemoryFabric, EnsembleWorldModel, and Comparator Error Feedback
into an active decision loop.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from poseidon.nexus_omega.contracts import (
    CarrierState,
    CausalEFEDecision,
    FactoredState,
    InterventionSpec,
    InterventionType,
    MemoryCarrierType,
    PredictionOutput,
    MacroIntent,
    ImaginedTrajectory,
    HierarchicalEFEDecision,
)
from poseidon.nexus_omega.memory import CausalMemoryFabric
from poseidon.nexus_omega.world_model import EnsembleWorldModel
from poseidon.nexus_omega.causal_world_model import FactoredCausalWorldModel
from poseidon.nexus_omega.causal_efe import CausalEFEPlanner
from poseidon.nexus_omega.hierarchical_efe import HierarchicalEFEPlanner


class NexusOmegaAgent:
    """Adaptive agent coordinating causal memory, world model, and comparator updates."""

    V2_SURPRISE_STORM_PERSISTENCE = 3.0

    def __init__(
        self,
        d_obs: int = 8,
        d_working: int = 16,
        n_actions: int = 4,
        uncertainty_penalty_lambda: float = 0.5,
        use_causal_efe: bool = True,
        use_hierarchical_efe: bool = False,
        base_seed: int = 42,
        residual_adapter_path: Optional[str] = None,
        exposure_index: Optional[int] = None,
        dynamics_profile: str = "nexus-phaseworld-v1",
    ):
        if dynamics_profile not in {"nexus-phaseworld-v1", "nexus-phaseworld-v2"}:
            raise ValueError("dynamics_profile must be nexus-phaseworld-v1 or nexus-phaseworld-v2")
        if residual_adapter_path is not None and dynamics_profile != "nexus-phaseworld-v1":
            raise ValueError("the contextual residual adapter is admitted only for nexus-phaseworld-v1")
        self.d_obs = d_obs
        self.d_working = d_working
        self.n_actions = n_actions
        self.lmbda = uncertainty_penalty_lambda
        self.use_causal_efe = use_causal_efe
        self.use_hierarchical_efe = use_hierarchical_efe
        self.dynamics_profile = dynamics_profile

        self.memory = CausalMemoryFabric(d_working=d_working, d_key=d_obs, d_value=d_obs)
        # Memory combined vector dimension: d_working + 1 (struct) + 1 (ep) = d_working + 2
        d_mem = d_working + 2
        self.world_model = EnsembleWorldModel(
            d_obs=d_obs,
            d_mem=d_mem,
            n_actions=n_actions,
            n_members=3,
            base_seed=base_seed,
        )
        model_type = FactoredCausalWorldModel
        if dynamics_profile == "nexus-phaseworld-v2":
            from poseidon.nexus_omega.causal_world_model_v2 import ViablePhaseWorldFactoredModel
            model_type = ViablePhaseWorldFactoredModel
        self.factored_world_model = model_type(
            d_obs=d_obs,
            d_mem=d_mem,
            n_actions=n_actions,
            n_members=3,
            base_seed=base_seed + 300,
        )
        if residual_adapter_path is not None:
            from poseidon.nexus_omega.residual_adapter import (
                CalibratedFactoredWorldModel,
                ContextualResidualAdapter,
            )
            adapter = ContextualResidualAdapter.from_receipt(residual_adapter_path)
            self.factored_world_model = CalibratedFactoredWorldModel(self.factored_world_model, adapter)
        self.efe_planner = CausalEFEPlanner(
            world_model=self.factored_world_model,
            n_actions=n_actions,
            base_epistemic_weight=uncertainty_penalty_lambda,
            exposure_index=exposure_index,
            safety_fallback_actions=(1, 3) if dynamics_profile == "nexus-phaseworld-v2" else None,
        )
        self.hierarchical_planner = HierarchicalEFEPlanner(
            world_model=self.factored_world_model,
            n_actions=n_actions,
            base_epistemic_weight=uncertainty_penalty_lambda,
            base_seed=base_seed + 500,
            exposure_index=exposure_index,
        )

        # Comparator projection: projects d_obs residual error into d_working memory delta
        rng = np.random.RandomState(base_seed + 99)
        self.W_comp = rng.randn(d_working, d_obs) * 0.15

        self.last_obs: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_mem: Optional[np.ndarray] = None
        self.last_decision: Optional[CausalEFEDecision | HierarchicalEFEDecision] = None
        self.last_hierarchical_decision: Optional[HierarchicalEFEDecision] = None
        self.last_health_shock_update = False

    def reset(self) -> None:
        self.memory.reset()
        self.hierarchical_planner.reset()
        self.last_obs = None
        self.last_action = None
        self.last_mem = None
        self.last_decision = None
        self.last_hierarchical_decision = None
        self.last_health_shock_update = False
        self.world_model.last_prediction = None
        self.factored_world_model.last_prediction = None
        self.factored_world_model.z_regime = np.array([0.0, 1.0], dtype=np.float32)

    def act(self, obs: np.ndarray | Sequence[float], evaluate_planning: bool = True) -> int:
        """Selects action via Deep Active Inference (EFE) or heuristic optimization."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        if len(obs_arr) > self.d_obs:
            obs_arr = obs_arr[:self.d_obs]
        mem_vec = self.memory.get_combined_vector()
        self.last_obs = obs_arr.copy()
        self.last_mem = mem_vec.copy()

        if not evaluate_planning:
            # Default reactive heuristic: forage if obs_arr[0] (health) > 0.5, else shelter
            self.last_action = 0 if obs_arr[0] > 0.5 else 1
            return self.last_action

        # Stage 10 Frontier: Hierarchical Multi-Scale Active Inference with Score Diffusion
        if self.use_hierarchical_efe:
            hier_dec = self.hierarchical_planner.plan(obs_arr, mem_vec)
            self.last_hierarchical_decision = hier_dec
            self.last_decision = hier_dec
            self.last_action = hier_dec.micro_action
            return hier_dec.micro_action

        # Primary Cognitive Path: Deep Active Inference with Conformal SPI
        if self.use_causal_efe:
            decision = self.efe_planner.plan_action(obs_arr, mem_vec)
            self.last_decision = decision
            self.last_action = decision.action
            return decision.action

        # Fallback Baseline Heuristic Path (for ablations and lesion controls)
        has_memory = (np.linalg.norm(mem_vec[:self.d_working]) > 1e-4)
        best_action = 0
        best_score = -float("inf")

        cue = float(obs_arr[3])
        remaining_storm_ticks = float(mem_vec[0]) if has_memory else 0.0
        is_predicted_storm = (cue > 0.8) or (has_memory and remaining_storm_ticks > 0.1)

        for a in range(self.n_actions):
            pred = self.world_model.predict(obs_arr, mem_vec, a)
            if a == 0:  # Forage
                expected_r = -0.6 if is_predicted_storm else 0.4
            elif a == 1:  # Shelter
                expected_r = 0.35 if is_predicted_storm else 0.05
            elif a == 2:  # Explore
                expected_r = 0.1
            else:  # Rest
                expected_r = 0.05

            score = expected_r - self.lmbda * pred.epistemic_uncertainty
            if score > best_score:
                best_score = score
                best_action = a

        self.last_action = best_action
        return best_action

    def observe_transition(self, next_obs: np.ndarray | Sequence[float], reward: float = 0.0, done: bool = False) -> np.ndarray:
        """Performs comparator update: computes residual e_t and updates causal memory."""
        if self.last_obs is None or self.last_action is None:
            return np.zeros(self.d_obs, dtype=np.float32)

        next_arr = np.asarray(next_obs, dtype=np.float32)
        if len(next_arr) > self.d_obs:
            next_arr = next_arr[:self.d_obs]

        # 1. Compute realized prediction error across active model
        if self.use_causal_efe:
            e_t = self.factored_world_model.compute_comparator_residual(next_arr)
        else:
            e_t = self.world_model.compute_comparator_residual(next_arr)

        self.last_health_shock_update = False
        if (
            self.dynamics_profile == "nexus-phaseworld-v2"
            and self.use_causal_efe
            and len(e_t) > 0
            and float(e_t[0]) <= -0.25
        ):
            # A large unexplained health loss is a one-step storm witness, even
            # when a pulse began before the first observation was available.
            # Retain a short multi-step belief so longer hidden hazards can be
            # survived long enough for repeated observations to correct it.
            self.factored_world_model.z_regime[0] = max(
                self.V2_SURPRISE_STORM_PERSISTENCE,
                float(self.factored_world_model.z_regime[0]),
            )
            self.last_health_shock_update = True

        # 2. Project comparator error into working memory update
        delta_mem = np.dot(self.W_comp, e_t)
        
        # Working memory pulse integration:
        # If pulse occurred during this transition, storm has 2 more steps remaining
        current_storm_ticks = float(self.memory.state.working_state[0])
        if self.last_obs[3] > 0.8 or next_arr[3] > 0.8:
            new_counter = 2.0
        else:
            new_counter = max(0.0, current_storm_ticks - 1.0)
            
        delta_mem[0] = new_counter - 0.85 * current_storm_ticks
        self.memory.write_working(delta_mem)

        # 3. Store episodic transition trace
        self.memory.store_episode(key=self.last_obs, value=next_arr)

        return e_t

    def observe(self, next_obs: np.ndarray | Sequence[float], reward: float = 0.0, done: bool = False) -> np.ndarray:
        """Ergonomic alias for observe_transition."""
        return self.observe_transition(next_obs, reward, done)

    def get_last_decision(self) -> Optional[CausalEFEDecision | HierarchicalEFEDecision]:
        return self.last_decision

    def get_last_hierarchical_decision(self) -> Optional[HierarchicalEFEDecision]:
        return self.last_hierarchical_decision

    def imagine_trajectory(
        self,
        intent: MacroIntent = MacroIntent.FORAGE_EXPEDITION,
        horizon: int = 8,
    ) -> ImaginedTrajectory:
        obs = self.last_obs if self.last_obs is not None else np.zeros(self.d_obs, dtype=np.float32)
        mem = self.last_mem if self.last_mem is not None else self.memory.get_combined_vector()
        return self.hierarchical_planner.imaginer.imagine_trajectory(obs, mem, target_intent=intent, horizon=horizon)

    def get_factored_state(self) -> Optional[FactoredState]:
        if self.last_obs is None:
            return None
        return self.factored_world_model.get_factored_state(self.last_obs)

    def comparator_update(self, predicted_state: np.ndarray, realized_state: np.ndarray) -> None:
        """Applies direct comparator residual error update from counterfactual or external prediction."""
        error = realized_state - predicted_state
        delta_mem = np.dot(self.W_comp, error[:self.d_obs])
        self.memory.write_working(delta_mem)

    observe = observe_transition

    def apply_intervention(self, spec: InterventionSpec) -> None:
        """Applies causal intervention operator directly to memory fabric."""
        self.memory.apply_intervention(spec)
