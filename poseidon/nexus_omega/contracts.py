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
