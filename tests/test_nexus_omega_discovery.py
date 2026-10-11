"""Tests for PROJECT NEXUS-Ω: Counterfactual SCM, Epigenetics, and Autonomous Discovery."""
import copy
import hashlib
import json
import pytest
import numpy as np

from poseidon.nexus_omega import (
    CounterfactualEngine,
    EpigeneticMorphogenesisSystem,
    DevelopmentalStage,
    EPIGENETIC_PROFILES,
    AutonomousDiscoveryEngine,
    verify_discovery_receipt,
    HypothesisSpec,
)


def test_counterfactual_engine_abduction_and_regret():
    engine = CounterfactualEngine()
    state = np.array([0.9, 0.8, 0.8, 0.7, 0.1, 0.0, 0.6, 0.5] + [0.0] * 8, dtype=np.float32)

    # 1. Prediction
    pred = engine.predict_factual(state, action=1)  # Forage
    assert len(pred) == 16
    assert all(0.0 <= x <= 1.0 for x in pred)

    # 2. Noise Abduction
    realized_next = pred + np.array([0.0, -0.05, 0.0, 0.0] + [0.0] * 12, dtype=np.float32)
    noise = engine.abduct_exogenous_noise(state, 1, realized_next)
    assert np.isclose(noise[1], -0.05, atol=1e-5)

    # 3. Counterfactual Evaluation
    cf_res = engine.evaluate_counterfactual(state, factual_action=4, realized_next_state=pred, counterfactual_action=1)
    assert cf_res.query.counterfactual_action == 1
    assert isinstance(cf_res.regret, float)
    assert cf_res.attribution in ("policy_error", "exogenous_shock", "unavoidable_attrition")

    # 4. Optimal Counterfactual Search
    best_a, best_res = engine.find_optimal_counterfactual(state, factual_action=4, realized_next_state=pred)
    assert 0 <= best_a < 6
    assert isinstance(best_res.regret, float)


def test_epigenetic_morphogenesis_and_ecdysis():
    system = EpigeneticMorphogenesisSystem(biomass_threshold=1.0, pupa_duration_ticks=2)
    assert system.stage == DevelopmentalStage.PROTOCELL

    # Feed healthy states to transition to Larval
    shell1 = None
    for _ in range(3):
        obs = np.array([0.95, 0.9, 0.9, 0.8, 0.1] + [0.5] * 11)
        shell = system.update_development(obs)
        if shell:
            shell1 = shell

    assert system.stage == DevelopmentalStage.LARVAL_FORAGER
    assert shell1 is not None
    assert len(shell1.shell_sha256) == 64

    # Accumulate biomass to trigger Pupal Diapause
    shell2 = None
    for _ in range(15):
        obs = np.array([0.95, 0.95, 0.95, 0.8, 0.1] + [0.5] * 11)
        shell = system.update_development(obs)
        if shell:
            shell2 = shell
            break

    assert system.stage == DevelopmentalStage.PUPAL_DIAPAUSE
    assert shell2 is not None

    # Test observation and action modulation during pupal diapause
    obs_test = np.array([0.5] * 16)
    obs_filtered = system.filter_observation(obs_test)
    assert len(obs_filtered) == 16

    action_scores = np.zeros(6)
    modulated = system.modulate_action_potentials(action_scores)
    # Pupal diapause should privilege Rest (action 0) over Explore (action 4)
    assert modulated[0] > modulated[4]

    # After pupa duration, emerges as Imago
    shell3 = None
    for _ in range(3):
        shell = system.update_development(obs_test)
        if shell:
            shell3 = shell

    assert system.stage == DevelopmentalStage.IMAGO_NAVIGATOR
    assert shell3 is not None


def test_autonomous_discovery_cycle_and_receipt_verification():
    engine = AutonomousDiscoveryEngine(max_steps=8, min_effect_size=0.1)
    # Run a short discovery cycle across 4 seeds
    ledger = engine.run_discovery_cycle(seed_base=60000, num_seeds=4)

    assert len(ledger.ledger_sha256) == 64
    assert len(ledger.ratified_laws) > 0
    # Negative test hypothesis (noise injection) must be refuted and placed in graveyard
    assert len(ledger.refutation_graveyard) > 0

    receipt_dict = {
        "ledger_id": ledger.ledger_id,
        "timestamp": ledger.timestamp,
        "ratified_laws": ledger.ratified_laws,
        "refutation_graveyard": ledger.refutation_graveyard,
        "ledger_sha256": ledger.ledger_sha256,
    }

    # Verify genuine receipt
    assert verify_discovery_receipt(receipt_dict) is True

    # Mutated receipt with forged claims must be rejected
    mutated = copy.deepcopy(receipt_dict)
    mutated["ratified_laws"][0]["reward_advantage"] = -99.0
    with pytest.raises(ValueError):
        verify_discovery_receipt(mutated)

    # Tampered ledger hash must be rejected
    tampered_hash = copy.deepcopy(receipt_dict)
    tampered_hash["ledger_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_discovery_receipt(tampered_hash)
