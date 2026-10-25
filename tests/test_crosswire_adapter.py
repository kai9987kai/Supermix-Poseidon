"""Tests for Project CROSSWIRE Adapter."""

import pytest

from poseidon.crosswire_engine import CandidateMapping
from poseidon.research.crosswire_adapter import (
    CrosswireAdapterVerifier,
    discretize_tidepool_obs_16d,
    STATE_GOAL,
    STATE_HAZARD,
    STATE_NOMINAL,
    STATE_READY,
)


def test_discretize_tidepool_obs():
    # Depleted vitality -> HAZARD
    obs_hazard1 = [0.10, 0.5, 0.5, 0.5] + [0.0] * 12
    assert discretize_tidepool_obs_16d(obs_hazard1) == STATE_HAZARD

    # High depth -> HAZARD
    obs_hazard2 = [0.8, 0.95, 0.5, 0.5] + [0.0] * 12
    assert discretize_tidepool_obs_16d(obs_hazard2) == STATE_HAZARD

    # Close to goal and healthy -> GOAL
    obs_goal = [0.8, 0.4, 0.2, 0.15] + [0.0] * 12
    assert discretize_tidepool_obs_16d(obs_goal) == STATE_GOAL

    # Initial state -> READY
    obs_ready = [0.9, 0.3, 0.2, 0.85] + [0.0] * 12
    assert discretize_tidepool_obs_16d(obs_ready) == STATE_READY

    # Default intermediate -> NOMINAL
    obs_nom = [0.5, 0.6, 0.5, 0.5] + [0.0] * 12
    assert discretize_tidepool_obs_16d(obs_nom) == STATE_NOMINAL


def test_crosswire_adapter_verifier_audit():
    verifier = CrosswireAdapterVerifier(num_actions=3, seed=42)
    receipt = verifier.run_interoperability_audit(
        true_permutation=[1, 2, 0],
        probing_budget=8,
        eval_ticks=16,
    )

    assert receipt["exact_permutation_match"] is True
    assert receipt["recovered_permutation"] == [1, 2, 0]
    assert receipt["zero_hazard_violations"] is True
    assert receipt["interoperability_verified"] is True
    assert "audit_sha256" in receipt
    assert verifier.verified_audit_receipt is not None
