from pathlib import Path

import pytest
import torch

from poseidon.research.oracle_adapter import ResearchOracleAdapter
from poseidon.research.oracle_training import _load_model, fit_oracle, load_oracle_artifact


@pytest.fixture(scope="module")
def oracle_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("adapter-fit")
    fit_oracle(directory, base_seed=631000000, train_episodes=4, selection_episodes=2,
               calibration_episodes=4, max_steps=12, training_updates=2, batch_size=8,
               hidden_size=16, ensemble_size=3, coverage=0.8)
    return directory / "oracle.pt"


def test_adapter_predictions_equal_loaded_research_model(oracle_artifact):
    adapter = ResearchOracleAdapter(oracle_artifact)
    model = _load_model(load_oracle_artifact(oracle_artifact))
    observations = torch.full((2, 16), 0.5)
    actions = torch.tensor([0, 2])
    with torch.inference_mode():
        expected = model.summarize(model(observations, actions))
        actual = adapter(torch.zeros(2, 128), observations, actions)
    assert torch.equal(actual["mean_delta"], expected["delta_mean"])
    assert torch.equal(actual["mean_reward"], expected["reward_mean"])
    assert torch.equal(actual["mean_failure"], expected["death_probability"])
    status = adapter.status()
    assert status["trained"] is True
    assert status["failure_probability_calibrated"] is False
    assert len(status["artifact_sha256"]) == 64


def test_adapter_rejects_invalid_observations_and_missing_artifact(oracle_artifact, tmp_path):
    adapter = ResearchOracleAdapter(oracle_artifact)
    for invalid in (float("nan"), float("inf"), -0.1, 1.1):
        obs = torch.full((1, 16), 0.5)
        obs[0, 0] = invalid
        with pytest.raises(ValueError, match="observation"):
            adapter(None, obs, torch.tensor([0]))
    with pytest.raises((ValueError, FileNotFoundError)):
        ResearchOracleAdapter(tmp_path / "absent.pt")


def test_imagination_has_no_fabricated_survival_certificate(oracle_artifact):
    adapter = ResearchOracleAdapter(oracle_artifact)
    result = adapter.imagine_trajectory(torch.zeros(128), torch.full((16,), 0.5), [0, 1])
    assert result["survival_probability"] is None
    assert result["failure_probability_calibrated"] is False
    assert result["calibration_applies_to_rollout"] is False
    assert len(result["trajectory"]) == 2
    assert adapter.status()["member_forward_calls"] == 6
    with pytest.raises(ValueError):
        adapter.imagine_trajectory(None, torch.full((16,), 0.5), [True])
