import pytest
import torch

from poseidon.research.candidate_evaluation import evaluate_oracle, verify_evaluation_receipt
from poseidon.research.contracts import EpisodeSplit, assert_disjoint_splits, tidepool_contract
from poseidon.research.oracle_training import (collect_episode, conformal_order_statistic,
                                                fit_oracle, verify_fit_receipt)


def test_tidepool_contract_and_episode_collection_are_exact():
    contract = tidepool_contract()
    assert contract.world_version == "tidepool-v1"
    first = collect_episode(91_001, scarcity=1.25, max_steps=12)
    second = collect_episode(91_001, scarcity=1.25, max_steps=12)
    assert first == second
    assert first.transitions[-1].terminal
    assert all(row.episode_seed == first.seed for row in first.transitions)
    with pytest.raises(ValueError, match="overlaps"):
        assert_disjoint_splits({
            "train": EpisodeSplit("train", (first,)),
            "test": EpisodeSplit("test", (second,)),
        })


def test_episode_conformal_quantile_requires_finite_sample_rank():
    assert conformal_order_statistic([1.0, 3.0, 2.0, 4.0], 0.8) == 4.0
    with pytest.raises(ValueError, match="too small"):
        conformal_order_statistic([1.0, 2.0, 3.0], 0.8)


def test_oracle_fit_keeps_splits_disjoint_and_receipts_replay(tmp_path):
    rng_before = torch.random.get_rng_state().clone()
    output = tmp_path / "oracle-fit"
    fit = fit_oracle(
        output,
        base_seed=552_000_000,
        train_episodes=4,
        selection_episodes=2,
        calibration_episodes=4,
        max_steps=10,
        training_updates=2,
        batch_size=8,
        hidden_size=16,
        ensemble_size=3,
        coverage=0.8,
    )
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert fit["activated"] is False
    seeds = fit["training_manifest"]["seed_sets"]
    assert not set(seeds["train"]) & set(seeds["selection"])
    assert not set(seeds["train"]) & set(seeds["calibration"])
    assert len(fit["training"]["bootstrap_episode_draws"]) == 3
    assert all(len(draws) == 4 for draws in fit["training"]["bootstrap_episode_draws"])
    assert verify_fit_receipt(output / "fit_receipt.json", replay_data=True)["valid"]

    evaluation = evaluate_oracle(
        fit["artifact_path"], tmp_path / "oracle-eval", base_seed=552_000_000,
        episodes_per_condition=2, bootstrap_reps=100,
    )
    assert evaluation["promoted"] is False
    assert evaluation["seed_disjointness_verified"] is True
    assert set(evaluation["results"]) == {"id_test", "ood_scarcity_test"}
    for result in evaluation["results"].values():
        metrics = result["metrics"]
        assert 0 <= metrics["state_episode_interval_coverage"] <= 1
        assert 0 <= metrics["reward_episode_interval_coverage"] <= 1
        assert len(metrics["paired_state_mse_gain_95pct_bootstrap_interval"]) == 2
    assert verify_evaluation_receipt(tmp_path / "oracle-eval" / "evaluation_receipt.json", replay=True)["valid"]


def test_oracle_refuses_to_overwrite_existing_experiment(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("prior experiment", encoding="utf-8")
    with pytest.raises(FileExistsError):
        fit_oracle(output, train_episodes=2, selection_episodes=2,
                   calibration_episodes=10, training_updates=1)
    assert marker.read_text(encoding="utf-8") == "prior experiment"
