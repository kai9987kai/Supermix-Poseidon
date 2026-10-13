"""PROJECT NEXUS-Ω: Core Interfaces, Data Contracts, and Formal Protocols.

Defines typed schemas for observations, memory carriers, interventions,
world models, audit receipts, and scientific hypotheses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Protocol
import numpy as np


class InterventionType(str, Enum):
    INTACT = "intact"
    ABLATE_CARRIER = "ablate_carrier"
    SHUFFLE_CARRIER = "shuffle_carrier"
    TRANSPLANT_CARRIER = "transplant_carrier"
    RECONSTRUCT_CARRIER = "reconstruct_carrier"
    CLAMP_INPUT = "clamp_input"


class MemoryCarrierType(str, Enum):
    WORKING = "working"         # Short-horizon bounded scratchpad
    EPISODIC = "episodic"       # Key-value trajectory store
    STRUCTURAL = "structural"   # Latent topology / causal adjacency
    PARAMETRIC = "parametric"   # Slow synaptic / weight reserves


@dataclass
class StructuredObservation:
    """Layer 1: Structured Observation Contract."""
    source_id: str
    step: int
    data: np.ndarray
    uncertainty: np.ndarray
    timestamp: float = field(default_factory=time.time)
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            h = hashlib.sha256()
            h.update(self.source_id.encode("utf-8"))
            h.update(str(self.step).encode("utf-8"))
            h.update(self.data.tobytes())
            self.provenance_hash = h.hexdigest()[:16]


@dataclass
class CarrierState:
    """Layer 2: Multi-substrate memory carrier state."""
    working_state: np.ndarray        # Shape: (d_working,)
    episodic_keys: List[np.ndarray] = field(default_factory=list)
    episodic_values: List[np.ndarray] = field(default_factory=list)
    structural_graph: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))  # Adjacency matrix
    step_count: int = 0
    intervention_log: List[str] = field(default_factory=list)

    def clone(self) -> CarrierState:
        return CarrierState(
            working_state=self.working_state.copy(),
            episodic_keys=[k.copy() for k in self.episodic_keys],
            episodic_values=[v.copy() for v in self.episodic_values],
            structural_graph=self.structural_graph.copy(),
            step_count=self.step_count,
            intervention_log=list(self.intervention_log),
        )


@dataclass
class PredictionOutput:
    """Layer 4: World model transition prediction with epistemic uncertainty."""
    predicted_obs: np.ndarray
    predicted_reward: float
    predicted_done: float
    epistemic_uncertainty: float
    ensemble_variance: np.ndarray
    comparator_error: Optional[np.ndarray] = None


@dataclass
class InterventionSpec:
    """Layer 5: Formal Causal Intervention Specification."""
    intervention_type: InterventionType
    target_carrier: Optional[MemoryCarrierType] = None
    target_indices: Optional[List[int]] = None
    donor_state: Optional[np.ndarray] = None
    description: str = ""


@dataclass
class HypothesisSpec:
    """Scientific hypothesis with explicit falsification conditions."""
    hypothesis_id: str
    statement: str
    null_hypothesis: str
    independent_variable: str
    dependent_variable: str
    alpha: float = 0.05
    minimum_effect_size: float = 0.05
    falsification_condition: str = ""


@dataclass
class AuditReceipt:
    """Layer 7: Independent Scientific Audit Receipt."""
    audit_id: str
    hypothesis_id: str
    timestamp: str
    seeds_evaluated: List[int]
    baseline_metrics: Dict[str, float]
    candidate_metrics: Dict[str, float]
    p_value: float
    effect_size: float
    retained_in_library: bool
    verdict: str  # "ACCEPTED", "FALSIFIED", "INCONCLUSIVE"
    receipt_hash: str = ""

    def seal(self) -> str:
        payload = {
            "audit_id": self.audit_id,
            "hypothesis_id": self.hypothesis_id,
            "timestamp": self.timestamp,
            "seeds": self.seeds_evaluated,
            "baseline": self.baseline_metrics,
            "candidate": self.candidate_metrics,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "verdict": self.verdict,
        }
        self.receipt_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.receipt_hash


class DevelopmentalStage(str, Enum):
    PROTOCELL = "protocell"
    LARVAL_FORAGER = "larval_forager"
    PUPAL_DIAPAUSE = "pupal_diapause"
    IMAGO_NAVIGATOR = "imago_navigator"
    SENESCENT_ARCHIVE = "senescent_archive"


@dataclass(frozen=True)
class EpigeneticProfile:
    stage: DevelopmentalStage
    basal_metabolism: float        # Multiplier on vital decay rates
    sensory_gain: float            # Multiplier on resource / gradient cues
    risk_penalty_scale: float      # Horizon-X lambda risk factor
    plasticity_rate: float         # Weight/adapter adaptation learning rate
    mobility_efficiency: float     # Stamina translation efficiency
    receptive_horizon: float       # Spatial perception radius scale


@dataclass
class ExuviaShell:
    """Cryptographic trace of an ecdysis event shedding an old morphological profile."""
    from_stage: DevelopmentalStage
    to_stage: DevelopmentalStage
    step: int
    biomass_accumulated: float
    health_at_ecdysis: float
    stamina_at_ecdysis: float
    shell_sha256: str = ""

    def __post_init__(self):
        if not self.shell_sha256:
            payload = {
                "from_stage": self.from_stage.value,
                "to_stage": self.to_stage.value,
                "step": self.step,
                "biomass": round(float(self.biomass_accumulated), 4),
                "health": round(float(self.health_at_ecdysis), 4),
                "stamina": round(float(self.stamina_at_ecdysis), 4),
            }
            self.shell_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class CounterfactualQuery:
    factual_state: np.ndarray
    factual_action: int
    factual_next_state: np.ndarray
    counterfactual_action: int
    exogenous_noise: Optional[np.ndarray] = None


@dataclass
class CounterfactualResult:
    query: CounterfactualQuery
    counterfactual_predicted_state: np.ndarray
    regret: float
    preventable_by_policy: bool
    attribution: str  # "policy_error", "exogenous_shock", "unavoidable_attrition"


@dataclass
class DiscoveryResult:
    hypothesis: HypothesisSpec
    baseline_reward: float
    candidate_reward: float
    reward_advantage: float
    p_value: float
    effect_size: float
    verdict: str  # "ACCEPTED", "REFUTED"
    seeds_evaluated: List[int]
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class DiscoveryLedger:
    ledger_id: str
    timestamp: str
    ratified_laws: List[Dict[str, Any]] = field(default_factory=list)
    refutation_graveyard: List[Dict[str, Any]] = field(default_factory=list)
    ledger_sha256: str = ""

    def seal(self) -> str:
        payload = {
            "ledger_id": self.ledger_id,
            "timestamp": self.timestamp,
            "ratified": self.ratified_laws,
            "refutations": self.refutation_graveyard,
        }
        self.ledger_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.ledger_sha256


@dataclass(frozen=True)
class FactoredState:
    """Factored Causal Latent Space separating physiological and environmental invariants."""
    endogenous_vitals: np.ndarray      # Shape (4,): health, energy, hydration, stamina
    exogenous_field: np.ndarray        # Shape (4,): exposure, storm_cue, food_scarcity, water_scarcity
    causal_regime: np.ndarray          # Shape (2,): storm_persistence, replenishment_rate


@dataclass
class FactoredPredictionOutput:
    """Prediction output with partitioned epistemic uncertainty."""
    predicted_obs: np.ndarray
    predicted_reward: float
    endogenous_pred: np.ndarray
    exogenous_pred: np.ndarray
    regime_pred: np.ndarray
    epistemic_uncertainty_total: float
    endogenous_uncertainty: float
    exogenous_uncertainty: float
    regime_uncertainty: float
    comparator_error: Optional[np.ndarray] = None


@dataclass
class CausalEFEDecision:
    """Rich decision telemetry under Deep Active Inference & Conformal SPI."""
    action: int
    action_name: str
    free_energy_objective: float
    pragmatic_value: float
    epistemic_value: float
    epistemic_weight: float
    empowerment_value: float
    conformal_lcb: float
    conformal_override: bool
    factored_state: Optional[Dict[str, List[float]]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": int(self.action),
            "action_name": str(self.action_name),
            "free_energy_objective": round(float(self.free_energy_objective), 6),
            "pragmatic_value": round(float(self.pragmatic_value), 6),
            "epistemic_value": round(float(self.epistemic_value), 6),
            "epistemic_weight": round(float(self.epistemic_weight), 6),
            "empowerment_value": round(float(self.empowerment_value), 6),
            "conformal_lcb": round(float(self.conformal_lcb), 6),
            "conformal_override": bool(self.conformal_override),
            "factored_state": self.factored_state or {},
            "timestamp": round(float(self.timestamp), 4),
        }


