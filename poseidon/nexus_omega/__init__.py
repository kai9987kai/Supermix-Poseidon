"""PROJECT NEXUS-Ω Research Platform.

A Causally Audited, Developmentally Adaptive, Self-Experimenting Computational System.
"""

from poseidon.nexus_omega.contracts import (
    StructuredObservation,
    CarrierState,
    PredictionOutput,
    InterventionSpec,
    InterventionType,
    MemoryCarrierType,
    HypothesisSpec,
    AuditReceipt,
    DevelopmentalStage,
    EpigeneticProfile,
    ExuviaShell,
    CounterfactualQuery,
    CounterfactualResult,
    DiscoveryResult,
    DiscoveryLedger,
    FactoredState,
    FactoredPredictionOutput,
    CausalEFEDecision,
)
from poseidon.nexus_omega.memory import CausalMemoryFabric
from poseidon.nexus_omega.world_model import EnsembleWorldModel
from poseidon.nexus_omega.causal_world_model import FactoredCausalWorldModel
from poseidon.nexus_omega.causal_efe import CausalEFEPlanner
from poseidon.nexus_omega.agent import NexusOmegaAgent
from poseidon.nexus_omega.benchmark import POMDPPhaseWorld
from poseidon.nexus_omega.auditor import IndependentScientificAuditor
from poseidon.nexus_omega.counterfactual import CounterfactualEngine
from poseidon.nexus_omega.epigenetics import (
    EpigeneticMorphogenesisSystem,
    EPIGENETIC_PROFILES,
)
from poseidon.nexus_omega.discovery import (
    AutonomousDiscoveryEngine,
    verify_discovery_receipt,
)

__all__ = [
    "StructuredObservation",
    "CarrierState",
    "PredictionOutput",
    "InterventionSpec",
    "InterventionType",
    "MemoryCarrierType",
    "HypothesisSpec",
    "AuditReceipt",
    "DevelopmentalStage",
    "EpigeneticProfile",
    "ExuviaShell",
    "CounterfactualQuery",
    "CounterfactualResult",
    "DiscoveryResult",
    "DiscoveryLedger",
    "CausalMemoryFabric",
    "EnsembleWorldModel",
    "NexusOmegaAgent",
    "POMDPPhaseWorld",
    "IndependentScientificAuditor",
    "CounterfactualEngine",
    "EpigeneticMorphogenesisSystem",
    "EPIGENETIC_PROFILES",
    "AutonomousDiscoveryEngine",
    "verify_discovery_receipt",
    "FactoredState",
    "FactoredPredictionOutput",
    "CausalEFEDecision",
    "FactoredCausalWorldModel",
    "CausalEFEPlanner",
]
