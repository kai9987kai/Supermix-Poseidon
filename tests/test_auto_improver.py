"""Regression coverage for candidate-local parameter search, without a real search."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import pytest

from tools import auto_improver as search


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    (root / "poseidon").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "runs").mkdir()
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads" / "main").write_text("1" * 40 + "\n")
    (root / "poseidon" / "active_inference.py").write_text(
        "base_epistemic_weight: float = 0.30\n# uncommitted source change\n"
    )
    (root / "dirty.txt").write_bytes(b"unrelated uncommitted work\r\n")
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (root / "tests" / "test_gate.py").write_text(
        "from pathlib import Path\n"
        "def test_workspace_isolation():\n"
        "    assert not Path('.git').exists()\n"
        "    assert not Path('dirty.txt').exists()\n"
        "    assert Path('runs/search_core.pt').read_bytes() == b'frozen core'\n"
    )
    (root / "runs" / "core.pt").write_bytes(b"frozen core")
    (root / "runs" / "active_core.json").write_text(json.dumps({"checkpoint": "runs/core.pt"}))
    monkeypatch.setattr(search, "SEARCH_SPACE", [search.SEARCH_SPACE[0]])
    monkeypatch.setattr(search, "TEST_FILES", ["tests/test_gate.py"], raising=False)
    return root


def tree_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def benchmark_result(seeds, reward):
    rows = [{"seed": seed, "total_reward": reward, "survived": True,
             "steps": 8, "overrides": 0} for seed in seeds]
    return {"per_seed": rows, "mean_reward": reward, "survival_rate": 1.0,
            "overrides": 0, "total_steps": 8 * len(seeds),
            "rewards": [reward] * len(seeds)}


def run_one(root, out, **kwargs):
    return search.run_search(source_root=root, output_root=out,
                             end_time=datetime.now() + timedelta(seconds=30),
                             seed=14, max_iterations=1, **kwargs)


def test_rejection_preserves_dirty_checkout_head_and_bound_core(source_tree, tmp_path, monkeypatch):
    before = tree_bytes(source_tree)

    def benchmark(workspace, *, seeds, deadline, **kwargs):
        assert workspace != source_tree
        reward = 1.0 if workspace.name == "baseline" else -1.0
        return benchmark_result(seeds, reward)

    monkeypatch.setattr(search, "run_benchmark", benchmark)
    result = run_one(source_tree, tmp_path / "artifacts")
    assert tree_bytes(source_tree) == before
    assert result["status"] == "completed"
    assert result["selected_candidate"] is None
    record = result["candidates"][0]
    assert record["status"] == "REJECTED_NO_IMPROVEMENT"
    assert Path(record["workspace"]).joinpath("poseidon/active_inference.py").exists()
    assert record["baseline_binding"]["core_sha256"] == hashlib.sha256(b"frozen core").hexdigest()
    assert record["candidate_binding"]["source_sha256"] != record["baseline_binding"]["source_sha256"]
    assert [row["seed"] for row in record["selection"]["candidate"]["per_seed"]] == search.SELECTION_SEEDS
    assert json.loads(Path(result["report_path"]).read_text())["status"] == "completed"


def test_selection_remains_inactive_and_audit_is_separate(source_tree, tmp_path, monkeypatch):
    before = tree_bytes(source_tree)
    seen = []

    def benchmark(workspace, *, seeds, deadline, **kwargs):
        seen.append((workspace.name, tuple(seeds)))
        reward = 1.0 if workspace.name == "baseline" else 2.0
        if seeds == search.AUDIT_SEEDS and workspace.name != "baseline":
            reward = -2.0
        return benchmark_result(seeds, reward)

    monkeypatch.setattr(search, "run_benchmark", benchmark)
    result = run_one(source_tree, tmp_path / "artifacts")
    assert tree_bytes(source_tree) == before
    assert result["selected_candidate"]["active"] is False
    assert result["audit"]["status"] == "completed"
    assert result["audit"]["improvement_supported"] is False
    assert result["audit"]["delta_reward"] == -3.0
    assert not set(search.SELECTION_SEEDS) & set(search.AUDIT_SEEDS)
    assert seen == [("baseline", tuple(search.SELECTION_SEEDS)),
                    ("candidate_0001", tuple(search.SELECTION_SEEDS)),
                    ("baseline", tuple(search.AUDIT_SEEDS)),
                    ("candidate_0001", tuple(search.AUDIT_SEEDS))]


def test_proposal_seed_is_independent_of_global_random_state(source_tree):
    left = random.Random(123)
    right = random.Random(123)
    expected = [search.propose_mutation(source_tree, left) for _ in range(8)]
    random.seed(999)
    actual = [search.propose_mutation(source_tree, right) for _ in range(8)]
    assert actual == expected
    assert len({row["new"] for row in actual}) > 1
    assert all(row["old"] == 0.30 for row in actual)


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_interrupted_candidate_preserves_incumbent(source_tree, tmp_path, monkeypatch, interruption):
    before = tree_bytes(source_tree)

    def benchmark(workspace, *, seeds, deadline, **kwargs):
        if workspace.name == "baseline":
            return benchmark_result(seeds, 1.0)
        if interruption == "cancel":
            raise KeyboardInterrupt()
        search.run_command([sys.executable, "-c", "import time; time.sleep(5)"],
                           workspace=workspace, deadline=time.monotonic() + 0.1)

    monkeypatch.setattr(search, "run_benchmark", benchmark)
    result = run_one(source_tree, tmp_path / "artifacts")
    assert tree_bytes(source_tree) == before
    assert result["status"] == {"timeout": "timed_out", "cancel": "cancelled"}[interruption]
    assert result["selected_candidate"] is None
    baseline = Path(result["baseline_workspace"])
    assert baseline.joinpath("poseidon/active_inference.py").read_bytes() == before["poseidon\\active_inference.py"]
    assert result["candidates"][0]["status"] in {"TIMED_OUT", "CANCELLED"}


def test_expired_deadline_never_starts_benchmark(source_tree, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("an expired run must not start a benchmark")

    monkeypatch.setattr(search, "run_benchmark", fail)
    result = search.run_search(source_root=source_tree, output_root=tmp_path / "artifacts",
                               end_time=datetime.now() - timedelta(seconds=1), seed=1)
    assert result["status"] == "timed_out"
    assert result["selected_candidate"] is None


@pytest.mark.parametrize("hours", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_is_rejected(hours):
    with pytest.raises(ValueError):
        search.parse_target_end_time(None, hours)
