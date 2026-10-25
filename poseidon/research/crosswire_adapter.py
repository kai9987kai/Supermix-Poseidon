"""Project CROSSWIRE Adapter for Supermix-Poseidon.

Bridges CROSSWIRE's verified interoperability, active diagnostic probing,
and closed-loop runtime shielding with Poseidon's ExternalAdapterBridge
(poseidon/external_adapter.py) and TidePool continuous simulation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from ..crosswire_engine import (
    ActiveDiagnosticProber,
    CandidateMapping,
    CandidateSpace,
    ClosedLoopRuntimeShield,
    CrosswireEngine,
    InterfaceContractSpec,
    InterfaceContractVerifier,
    TransitionTrace,
    VersionSpaceTracker,
)
from ..external_adapter import (
    AdapterHandshake,
    ExternalActionResponse,
    ExternalAdapterBridge,
    ExternalAdapterIdentity,
    ExternalTickMessage,
)
from ..world import N_ACTIONS, OBS_SIZE


# Observation Discretization Predicates
STATE_READY = 0
STATE_NOMINAL = 1
STATE_GOAL = 2
STATE_HAZARD = 3


def discretize_tidepool_obs_16d(obs: Sequence[float]) -> int:
    """Discretizes continuous 16-D TidePool observation into semantic discrete states."""
    if len(obs) < 16:
        return STATE_NOMINAL
    vitality = float(obs[0])
    depth = float(obs[1])
    salinity = float(obs[2])
    dist_goal = float(obs[3])

    # Hazard checks: depleted vital, crushing depth, toxic salinity
    if vitality <= 0.15 or depth >= 0.90 or salinity >= 0.85:
        return STATE_HAZARD
    # Goal check: close to resource and healthy
    if dist_goal <= 0.25 and vitality >= 0.60:
        return STATE_GOAL
    # Ready check: initial state
    if dist_goal >= 0.70 and vitality >= 0.85 and 0.20 <= depth <= 0.50:
        return STATE_READY
    return STATE_NOMINAL


class CrosswireAdapterVerifier:
    """Connects CROSSWIRE active probing and runtime shielding to ExternalAdapterBridge."""

    def __init__(
        self,
        bridge: Optional[ExternalAdapterBridge] = None,
        num_actions: int = 4,
        seed: int = 42,
    ):
        self.bridge = bridge or ExternalAdapterBridge()
        self.num_actions = num_actions
        self.seed = seed
        self.engine = CrosswireEngine(num_actions=num_actions)
        self.hypotheses = CandidateSpace.generate_permutations(num_actions)
        self.tracker = VersionSpaceTracker(self.hypotheses, self.engine.transition_model)
        self.shield = ClosedLoopRuntimeShield(
            fallback_action_int=0,
            hazard_states={STATE_HAZARD},
            transition_model=self.engine.transition_model,
        )
        self.prober = ActiveDiagnosticProber(num_actions, self.engine.transition_model)
        self.verified_audit_receipt: Optional[Dict[str, Any]] = None

    def create_mock_handshake(self, client_id: str = "crosswire-unity-bridge") -> AdapterHandshake:
        """Establishes a typed adapter handshake for diagnostic sessions."""
        identity = ExternalAdapterIdentity(
            adapter_id="unfamiliar-engine-v1",
            units="NORMALIZED_01",
            ticks_per_second=20.0,
            coordinate_frame="FLU",
            spatial_bounds=(1.0, 1.0, 1.0),
            max_linear_velocity=5.0,
        )
        return self.bridge.negotiate_handshake(
            client_id=client_id,
            engine_name="Unity3D-Unfamiliar",
            identity=identity,
        )

    def run_interoperability_audit(
        self,
        true_permutation: Optional[Sequence[int]] = None,
        probing_budget: int = 12,
        eval_ticks: int = 24,
    ) -> Dict[str, Any]:
        """Runs end-to-end active diagnostic discovery and closed-loop shielded execution."""
        k = self.num_actions
        # Ground truth mapping: external code -> internal core action
        if true_permutation is None:
            # Deterministic permutation for reproducibility
            true_perm_tuple = tuple((i + 1) % k for i in range(k))
        else:
            true_perm_tuple = tuple(true_permutation)

        true_h = CandidateMapping("h_true", true_perm_tuple)
        handshake = self.create_mock_handshake()
        token = handshake.session_token

        # Step 1: Active Diagnostic Probing to contract version space
        probing_session = self.prober.generate_probing_sequence(
            start_state=0,
            candidate_hypotheses=self.hypotheses,
            true_hypothesis=true_h,
            budget=probing_budget,
        )

        recovered_mapping = (
            CandidateMapping("h_recovered", tuple(probing_session["identified_mapping"]["permutation"]))
            if probing_session["identified_mapping"]
            else None
        )

        # Step 2: Test Closed-Loop Runtime Shield on synthetic agent stream
        shield_history = []
        curr_state = STATE_READY
        hazard_violations = 0

        for t in range(eval_ticks):
            # Propose candidate action (intended internal action)
            intended_a_int = (t % (k - 1)) + 1
            surviving = [recovered_mapping] if recovered_mapping else self.hypotheses

            # Shield evaluation
            shield_res = self.shield.evaluate_and_filter(
                current_state=curr_state,
                intended_action_int=intended_a_int,
                version_space=surviving,
            )
            dispatched_ext = shield_res["dispatched_action_ext"]

            # Simulator executes dispatched_ext under ground truth
            simulated_act_int = true_h.map_action(dispatched_ext)
            next_state = self.engine.transition_model.get((curr_state, simulated_act_int), curr_state)

            if next_state == STATE_HAZARD:
                hazard_violations += 1

            shield_history.append({
                "tick": t,
                "curr_state": curr_state,
                "intended_int": intended_a_int,
                "dispatched_ext": dispatched_ext,
                "simulated_int": simulated_act_int,
                "next_state": next_state,
                "decision": shield_res["decision"],
                "is_fallback": shield_res["is_fallback"],
            })
            curr_state = next_state if next_state != STATE_HAZARD else STATE_READY

        is_match = (
            recovered_mapping.permutation == true_h.permutation
            if recovered_mapping is not None
            else False
        )

        receipt = {
            "session_token": token,
            "actions_count": k,
            "true_permutation": list(true_h.permutation),
            "recovered_permutation": list(recovered_mapping.permutation) if recovered_mapping else None,
            "exact_permutation_match": is_match,
            "probing_steps": probing_session["steps_executed"],
            "probing_budget": probing_budget,
            "shield_eval_ticks": eval_ticks,
            "shield_intercepts": self.shield.intercept_count,
            "shield_admitted": self.shield.admitted_count,
            "hazard_violations": hazard_violations,
            "zero_hazard_violations": hazard_violations == 0,
            "interoperability_verified": is_match and (hazard_violations == 0),
        }

        # SHA-256 fingerprint
        blob = json.dumps(receipt, sort_keys=True).encode("utf-8")
        receipt["audit_sha256"] = hashlib.sha256(blob).hexdigest()
        self.verified_audit_receipt = receipt
        return receipt
