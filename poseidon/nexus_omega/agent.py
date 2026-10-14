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
    QuantumCausalState,
    HarmonicSpectralProfile,
    SynthesizedOpcode,
    ThermodynamicState,
    DreamConsolidationReceipt,
    UltimaDecision,
)
from poseidon.nexus_omega.memory import CausalMemoryFabric
from poseidon.nexus_omega.world_model import EnsembleWorldModel
from poseidon.nexus_omega.causal_world_model import FactoredCausalWorldModel
from poseidon.nexus_omega.causal_efe import CausalEFEPlanner
from poseidon.nexus_omega.hierarchical_efe import HierarchicalEFEPlanner
from poseidon.nexus_omega.spectral_resonator import BioharmonicResonator
from poseidon.nexus_omega.quantum_causal import QuantumCausalKernel
from poseidon.nexus_omega.bytecode_weaver import MetamorphicBytecodeWeaver
from poseidon.nexus_omega.autopoiesis_field import AutopoiesisThermodynamicField
from poseidon.nexus_omega.morpheus_dream import MorpheusDreamEngine


class NexusOmegaAgent:
    """Adaptive agent coordinating causal memory, world model, and comparator updates."""

    V2_DEFAULT_STORM_DURATION = 3.0
    V2_MAX_STORM_DURATION = 5.0

    def __init__(
        self,
        d_obs: int = 8,
        d_working: int = 16,
        n_actions: int = 4,
        uncertainty_penalty_lambda: float = 0.5,
        use_causal_efe: bool = True,
        use_hierarchical_efe: bool = False,
        use_ultima_engine: bool = False,
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
        self.use_causal_efe = use_causal_efe or use_ultima_engine
        self.use_hierarchical_efe = use_hierarchical_efe or use_ultima_engine
        self.use_ultima_engine = use_ultima_engine
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
            calm_recovery_action=0 if dynamics_profile == "nexus-phaseworld-v2" else None,
        )
        self.hierarchical_planner = HierarchicalEFEPlanner(
            world_model=self.factored_world_model,
            n_actions=n_actions,
            base_epistemic_weight=uncertainty_penalty_lambda,
            base_seed=base_seed + 500,
            exposure_index=exposure_index,
        )

        # Stage 11 Frontier Subsystems
        self.spectral_resonator = BioharmonicResonator()
        self.quantum_kernel = QuantumCausalKernel(tunneling_rate=0.18)
        self.bytecode_weaver = MetamorphicBytecodeWeaver(world_model=self.factored_world_model)
        self.autopoiesis_field = AutopoiesisThermodynamicField(base_curiosity_temp=uncertainty_penalty_lambda)
        self.dream_engine = MorpheusDreamEngine()

        self.last_ultima_decision: Optional[UltimaDecision] = None
        self.active_opcode_sequence: List[int] = []
        self.active_opcode: Optional[SynthesizedOpcode] = None
        self.high_regret_buffer: List[Dict[str, Any]] = []
        self.dream_receipt: Optional[DreamConsolidationReceipt] = None

        # Comparator projection: projects d_obs residual error into d_working memory delta
        rng = np.random.RandomState(base_seed + 99)
        self.W_comp = rng.randn(d_working, d_obs) * 0.15

        self.last_obs: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_mem: Optional[np.ndarray] = None
        self.last_decision: Optional[CausalEFEDecision | HierarchicalEFEDecision] = None
        self.last_hierarchical_decision: Optional[HierarchicalEFEDecision] = None
        self.last_health_shock_update = False
        self.v2_storm_duration_estimate = self.V2_DEFAULT_STORM_DURATION
        self.v2_ticks_since_pulse: Optional[int] = None

    def reset(self) -> None:
        self.memory.reset()
        self.hierarchical_planner.reset()
        self.spectral_resonator = BioharmonicResonator()
        self.quantum_kernel = QuantumCausalKernel(tunneling_rate=0.18)
        self.last_obs = None
        self.last_action = None
        self.last_mem = None
        self.last_decision = None
        self.last_hierarchical_decision = None
        self.last_ultima_decision = None
        self.active_opcode_sequence = []
        self.active_opcode = None
        self.high_regret_buffer = []
        self.dream_receipt = None
        self.last_health_shock_update = False
        self.v2_storm_duration_estimate = self.V2_DEFAULT_STORM_DURATION
        self.v2_ticks_since_pulse = None
        self.world_model.last_prediction = None
        self.factored_world_model.last_prediction = None
        self.factored_world_model.z_regime = np.array([0.0, 1.0], dtype=np.float32)

    def act(self, obs: np.ndarray | Sequence[float], evaluate_planning: bool = True) -> int:
        """Selects action via Deep Active Inference (EFE) or heuristic optimization."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        if len(obs_arr) > self.d_obs:
            obs_arr = obs_arr[:self.d_obs]
        if self.dynamics_profile == "nexus-phaseworld-v2" and len(obs_arr) > 3:
            if float(obs_arr[3]) > 0.8:
                self.v2_ticks_since_pulse = 0
            elif self.v2_ticks_since_pulse is not None:
                self.v2_ticks_since_pulse += 1
        mem_vec = self.memory.get_combined_vector()
        self.last_obs = obs_arr.copy()
        self.last_mem = mem_vec.copy()

        if not evaluate_planning:
            # Default reactive heuristic: forage if obs_arr[0] (health) > 0.5, else shelter
            self.last_action = 0 if obs_arr[0] > 0.5 else 1
            return self.last_action

        # Stage 11 Singularity Frontier: POSEIDON NEXUS-ULTIMA Omni-Resonant Autopoietic Engine
        if self.use_ultima_engine:
            # 1. Bioharmonic Spectral Fourier Resonator
            cue_val = float(obs_arr[3]) if len(obs_arr) > 3 else float(obs_arr[0])
            harmonic_profile = self.spectral_resonator.update(cue_val)
            anticipated_hazard = self.spectral_resonator.anticipate_hazard(horizon=4)

            # 2. Thermodynamic Autopoiesis & PAD Affect Field
            pred_surprise = 0.05
            if self.world_model.last_prediction is not None:
                pred_surprise = float(self.world_model.last_prediction.epistemic_uncertainty)
            thermo_state = self.autopoiesis_field.evaluate_state(
                obs_arr,
                epistemic_surprise=pred_surprise,
                empowerment=0.50,
            )
            # Modulate EFE curiosity scale dynamically by PAD affect
            if hasattr(self.hierarchical_planner, "micro_planner"):
                self.hierarchical_planner.micro_planner.base_epistemic_weight = thermo_state.effective_curiosity_temp

            # 3. Quantum Causal Superposition Evolution
            evaluated_trajs = self.hierarchical_planner.imaginer.evaluate_all_intents(obs_arr, mem_vec)
            macro_fes = []
            for m in QuantumCausalKernel.INTENT_ORDER:
                traj = evaluated_trajs.get(m)
                fe = -float(traj.conformal_trajectory_lcb) if traj else 1.0
                if anticipated_hazard > 0.60 and m in (MacroIntent.FORAGE_EXPEDITION, MacroIntent.HABITAT_RELOCATION):
                    fe += 3.5 * anticipated_hazard
                macro_fes.append(fe)

            quantum_state = self.quantum_kernel.evolve_step(macro_fes, dt=0.25)

            # 4. Holonomic Intent Selection
            holonomic_override = False
            if anticipated_hazard > 0.70 or obs_arr[0] < 0.35:
                collapsed_intent, _ = self.quantum_kernel.collapse_state(force_index=1)
                holonomic_override = True
            elif quantum_state.purity > 0.75:
                collapsed_intent, _ = self.quantum_kernel.collapse_state()
            else:
                collapsed_intent, _ = self.quantum_kernel.get_highest_probability_intent()

            # 5. Metamorphic Bytecode Synthesis / Execution
            active_opcode_obj = None
            if len(self.active_opcode_sequence) > 0:
                micro_act = self.active_opcode_sequence.pop(0)
                active_opcode_obj = self.active_opcode
            elif anticipated_hazard > 0.65 or obs_arr[0] < 0.35:
                opcode = self.bytecode_weaver.synthesize(collapsed_intent, obs_arr, mem_vec)
                self.active_opcode = opcode
                self.active_opcode_sequence = list(opcode.micro_action_sequence)
                micro_act = self.active_opcode_sequence.pop(0)
                active_opcode_obj = opcode
            else:
                hier_dec = self.hierarchical_planner.plan(obs_arr, mem_vec)
                micro_act = hier_dec.micro_action

            # 6. Morpheus Dream Sleep Trigger
            if collapsed_intent == MacroIntent.DIAPAUSE_HIBERNATION:
                self.dream_receipt = self.dream_engine.consolidate(
                    self.memory, self.high_regret_buffer, num_dream_rollouts=4
                )

            hier_decision = getattr(self.hierarchical_planner, "last_decision", None)
            if hier_decision is None:
                hier_decision = self.hierarchical_planner.plan(obs_arr, mem_vec)

            action_names = ["forage", "shelter", "explore", "rest"]
            act_name = action_names[micro_act] if 0 <= micro_act < len(action_names) else "unknown"

            ultima_dec = UltimaDecision(
                quantum_state=quantum_state,
                harmonic_profile=harmonic_profile,
                thermodynamic_state=thermo_state,
                hierarchical_decision=hier_decision,
                active_opcode=active_opcode_obj,
                dream_receipt=self.dream_receipt,
                final_action=micro_act,
                final_action_name=act_name,
                holonomic_override=holonomic_override,
            )
            self.last_ultima_decision = ultima_dec
            self.last_hierarchical_decision = hier_decision
            self.last_decision = hier_decision
            self.last_action = micro_act
            return micro_act

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
        if self.dynamics_profile == "nexus-phaseworld-v2" and self.use_causal_efe:
            if len(e_t) > 0 and float(e_t[0]) <= -0.25:
                # A large health residual lengthens the inferred storm only if
                # it outlasts the cue-relative duration currently expected.
                pulse_age = self.v2_ticks_since_pulse
                if pulse_age is not None:
                    self.v2_storm_duration_estimate = min(
                        self.V2_MAX_STORM_DURATION,
                        max(self.v2_storm_duration_estimate, float(pulse_age + 2)),
                    )
                    remaining_ticks = max(
                        1.0,
                        self.v2_storm_duration_estimate - float(pulse_age + 1),
                    )
                else:
                    remaining_ticks = self.V2_DEFAULT_STORM_DURATION
                self.factored_world_model.z_regime[0] = max(
                    remaining_ticks,
                    float(self.factored_world_model.z_regime[0]),
                )
                self.last_health_shock_update = True
            elif len(self.last_obs) > 3 and float(self.last_obs[3]) > 0.8:
                # Keep the learned remaining duration after acting on a pulse.
                remaining_ticks = max(1.0, self.v2_storm_duration_estimate - 1.0)
                self.factored_world_model.z_regime[0] = max(
                    remaining_ticks,
                    float(self.factored_world_model.z_regime[0]),
                )

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

        if reward < 0.0 or (self.last_obs is not None and self.last_obs[0] - next_arr[0] > 0.15):
            self.high_regret_buffer.append({
                "obs": self.last_obs.tolist() if self.last_obs is not None else [],
                "action": self.last_action,
                "reward": float(reward),
                "regret": float(max(0.1, abs(reward))),
            })
            if len(self.high_regret_buffer) > 32:
                self.high_regret_buffer.pop(0)

        return e_t

    def observe(self, next_obs: np.ndarray | Sequence[float], reward: float = 0.0, done: bool = False) -> np.ndarray:
        """Ergonomic alias for observe_transition."""
        return self.observe_transition(next_obs, reward, done)

    def get_last_decision(self) -> Optional[CausalEFEDecision | HierarchicalEFEDecision]:
        return self.last_decision

    def get_last_hierarchical_decision(self) -> Optional[HierarchicalEFEDecision]:
        return self.last_hierarchical_decision

    def get_last_ultima_decision(self) -> Optional[UltimaDecision]:
        return self.last_ultima_decision

    def get_spectral_profile(self) -> HarmonicSpectralProfile:
        cue = float(self.last_obs[3]) if (self.last_obs is not None and len(self.last_obs) > 3) else 0.0
        return self.spectral_resonator.update(cue)

    def get_quantum_state(self) -> QuantumCausalState:
        return self.quantum_kernel.evolve_step([1.0, 1.0, 1.0, 1.0])

    def dream_consolidate(self, num_rollouts: int = 4) -> DreamConsolidationReceipt:
        receipt = self.dream_engine.consolidate(self.memory, self.high_regret_buffer, num_dream_rollouts=num_rollouts)
        self.dream_receipt = receipt
        return receipt

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
