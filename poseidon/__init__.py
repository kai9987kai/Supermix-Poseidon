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

_SOURCE_SNAPSHOT = _capture_sources()
