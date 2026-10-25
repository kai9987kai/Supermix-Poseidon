"""Tests for FaultlineOracleAdapter."""

import pytest
import torch

from poseidon.research.faultline_adapter import FaultlineOracleAdapter
from poseidon.research.tether_gate import TetherReliabilityGate
from poseidon.world import N_ACTIONS, OBS_SIZE


def test_faultline_oracle_adapter_initialization():
    adapter = FaultlineOracleAdapter(panel_size=3, primary_policy="joint", seed=42)
    assert adapter.panel_size == 3
    assert adapter.primary_policy == "joint"
    assert len(adapter.active_panel) == 3
    assert len(adapter.active_panel_ids) == 3

    receipt = adapter.audit_receipt
    assert receipt is not None
    assert receipt["schema"] == "poseidon-faultline-oracle-audit-v1"
    assert "sha256" in receipt
    assert receipt["joint_advantage"] >= 0.0


def test_faultline_oracle_adapter_set_active_panel():
    adapter = FaultlineOracleAdapter(panel_size=3, primary_policy="joint", seed=42)
    adapter.set_active_panel([0, 1, 4])
    assert adapter.active_panel == [0, 1, 4]
    assert adapter.active_panel_ids == ["m1", "m2", "m5"]

    with pytest.raises(ValueError, match="Invalid member index"):
        adapter.set_active_panel([0, 1, 99])

    with pytest.raises(ValueError, match="Panel size mismatch"):
        adapter.set_active_panel([0, 1])


def test_faultline_oracle_adapter_status():
    adapter = FaultlineOracleAdapter(panel_size=3, primary_policy="joint", seed=42)
    st = adapter.status()
    assert st["faultline_enabled"] is True
    assert st["faultline_protocol"] == "faultline-oracle-panel-v1"
    assert st["panel_size"] == 3
    assert st["n_available_members"] == 5
    assert len(st["active_panel"]) == 3
    assert st["sha256_receipt"] is not None


def test_faultline_oracle_adapter_forward():
    adapter = FaultlineOracleAdapter(panel_size=3, primary_policy="joint", seed=42)
    batch_size = 4
    obs = torch.randn(batch_size, OBS_SIZE)
    actions = torch.randint(0, N_ACTIONS, (batch_size,))

    out = adapter(None, obs, actions)
    assert "mean_delta" in out
    assert out["mean_delta"].shape == (batch_size, OBS_SIZE)
    assert "mean_reward" in out
    assert "mean_failure" in out
    assert "raw_disagreement" in out
    assert "faultline_active_panel" in out
    assert torch.equal(out["faultline_active_panel"], torch.tensor(adapter.active_panel))


def test_faultline_oracle_adapter_planning_support():
    adapter = FaultlineOracleAdapter(panel_size=3, primary_policy="joint", seed=42)
    obs = [0.1] * OBS_SIZE
    ok, reason = adapter.planning_support(obs)
    assert ok is True
    assert reason is None

    bad_obs = [0.1] * 5
    ok_bad, reason_bad = adapter.planning_support(bad_obs)
    assert ok_bad is False
    assert "invalid_obs_size" in reason_bad
