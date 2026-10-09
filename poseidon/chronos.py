"""Chronos & Ghost: Periodic Frame Beaconing and Phantom Trace Introspection.

Synthesizing hardware-clock cyclic beacon synchronization (Flipper-Zero NTAG / svideo)
and autonomous latent ghost trace introspection (GhostInTheMachine).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class BeaconPulse:
    cycle: int
    tick: int
    phase: int
    syn_pulse: bool
    pulse_sha256: str


class ChronosBeaconClock:
    """Deterministic cyclic beacon clock providing phase synchronization across modules."""

    def __init__(self, period: int = 8):
        self.period = int(period)
        self.tick = 0
        self.cycle = 0

    def reset(self) -> None:
        self.tick = 0
        self.cycle = 0

    def step(self) -> BeaconPulse:
        """Advance one clock tick and evaluate periodic synchronization pulse."""
        self.tick += 1
        phase = self.tick % self.period
        if phase == 0:
            self.cycle += 1
            syn_pulse = True
        else:
            syn_pulse = False

        payload = f"{self.cycle}:{self.tick}:{phase}:{syn_pulse}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        return BeaconPulse(
            cycle=self.cycle,
            tick=self.tick,
            phase=phase,
            syn_pulse=syn_pulse,
            pulse_sha256=digest,
        )


class GhostTraceAuditor:
    """Introspective auditor tracking counterfactual shadow traces and Spectral Divergence."""

    def __init__(self, trace_decay: float = 0.88, n_actions: int = 6):
        self.trace_decay = float(trace_decay)
        self.n_actions = int(n_actions)
        self.actual_history: List[int] = []
        self.phantom_history: List[int] = []
        self.sdi_history: List[float] = []
        self.actual_distribution = np.zeros(self.n_actions, dtype=np.float32)
        self.phantom_distribution = np.zeros(self.n_actions, dtype=np.float32)

    def reset(self) -> None:
        self.actual_history.clear()
        self.phantom_history.clear()
        self.sdi_history.clear()
        self.actual_distribution.fill(0.0)
        self.phantom_distribution.fill(0.0)

    def record_step(self, actual_action: int, phantom_action: int) -> float:
        """Record executed vs counterfactual shadow action, updating Spectral Divergence Index."""
        self.actual_history.append(int(actual_action))
        self.phantom_history.append(int(phantom_action))

        # Exponential decay update of empirical action probabilities
        self.actual_distribution *= self.trace_decay
        self.actual_distribution[actual_action] += 1.0 - self.trace_decay

        self.phantom_distribution *= self.trace_decay
        self.phantom_distribution[phantom_action] += 1.0 - self.trace_decay

        # Normalize distributions
        sum_act = np.sum(self.actual_distribution)
        sum_phan = np.sum(self.phantom_distribution)

        p_act = self.actual_distribution / max(sum_act, 1e-6)
        p_phan = self.phantom_distribution / max(sum_phan, 1e-6)

        # Spectral Divergence Index: mean squared difference across action spectrum
        sdi = float(np.mean((p_act - p_phan) ** 2))
        self.sdi_history.append(round(sdi, 6))
        return round(sdi, 6)

    def summary(self) -> Dict[str, Any]:
        mean_sdi = float(np.mean(self.sdi_history)) if self.sdi_history else 0.0
        max_sdi = float(np.max(self.sdi_history)) if self.sdi_history else 0.0
        return {
            "steps_recorded": len(self.actual_history),
            "mean_spectral_divergence": round(mean_sdi, 6),
            "max_spectral_divergence": round(max_sdi, 6),
            "latest_sdi": self.sdi_history[-1] if self.sdi_history else 0.0,
        }
