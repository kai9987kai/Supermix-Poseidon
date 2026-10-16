"""Supermix Poseidon: explicitly bounded, locally trained research system."""
__version__ = "2.0.0"

from .provenance import capture_sources as _capture_sources
from .cognitive_agent import CognitiveAgent, CognitiveDecision
from .oracle import OracleConfig, OracleEnsemble, OracleRuntime
from .mnemosyne import MNEMOSYNE_MODES, MnemosyneConfig, MnemosyneMemory
from .horizon_x import HorizonXConfig, HorizonXPlanner
from .adapt import AdaptConfig, ContinualAdapter, ResidualPolicyAdapter
from .worldlab import POMDPConfig, POMDPTidePool, StepResult, WorldBridgeProtocol
from .neuro import NeuroConfig, NeuromodulatedRouter
from .active_inference import ActiveInferenceConfig, ActiveInferenceEvaluator, HomeostaticTarget
from .dream import DreamConfig, DreamConsolidator, DreamSummary
from .metacognition import MetacognitionConfig, MetacognitiveMonitor, MetacognitiveState
from .guardrails import FormalSafetyGuard, GuardrailAudit, InvariantResult
from .information_seeking import EpistemicInformationProber, CandidateAdapterSession, RetentionAndAdaptationAuditor
from .external_adapter import (
    ExternalAdapterIdentity,
    AdapterHandshake,
    ExternalTickMessage,
    MemoryInfluenceTelemetry,
    ExternalActionResponse,
    ExternalAdapterBridge,
    Compact3DReplayVerifier,
)
from .nexus_omega import (
    NexusOmegaAgent,
    POMDPPhaseWorld,
    AutonomousDiscoveryEngine,
    verify_discovery_receipt,
    FactoredCausalWorldModel,
    CausalEFEPlanner,
    FactoredState,
    CausalEFEDecision,
    MacroIntent,
    ImaginedTrajectory,
    HierarchicalEFEDecision,
    TrajectoryDiffusionImaginer,
    HierarchicalEFEPlanner,
    QuantumCausalState,
    HarmonicSpectralProfile,
    SynthesizedOpcode,
    ThermodynamicState,
    DreamConsolidationReceipt,
    UltimaDecision,
    BioharmonicResonator,
    QuantumCausalKernel,
    MetamorphicBytecodeWeaver,
    LTLSafetyShield,
    AutopoiesisThermodynamicField,
    MorpheusDreamEngine,
    CognitiveMonologue,
    ApexDecision,
    NonstationaryPhaseFieldResonator,
    NonstationaryPhaseState,
    ContrastiveCausalMemoryFabric,
    ContrastiveMemoryDecision,
    ApexReasoningEngine,
    NexusApexEngine,
    POMDPPhaseWorldV2,
    POMDPPhaseWorldV3,
)
from .hypothesis_engine import (
    ScientificHypothesis,
    CrucialExperiment,
    HypothesisFalsificationReceipt,
    FalsificationOutcome,
    HypothesisEngine,
)

_SOURCE_SNAPSHOT = _capture_sources()
