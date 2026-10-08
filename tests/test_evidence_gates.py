"""Behavioral checks for incomplete audit evidence and controlled episodic recall."""
from types import SimpleNamespace

import pytest
import torch

from poseidon import adaptation
from poseidon.adaptation import PromotionAuditor, PromotionCriteria
from poseidon.core import CoreConfig, SCENE_LABELS
from poseidon.media import generate_scene_examples
from poseidon.persistent_memory import EpisodicMemoryStore, run_delayed_recall_benchmark


class AuditWorld:
    def __init__(self, seed=0, scarcity=2.0, max_steps=128):
        self.alive = True
        self.reset(seed)

    def reset(self, seed=None):
        self.tick = 0
        self.alive = True
        return [0.2] * 16

    def step(self, action):
        self.tick += 1
        self.alive = action == 0
        return [0.2 + self.tick * 0.1] * 16, float(self.alive), self.tick == 3, {"death": None if self.alive else "hazard"}


class AuditCore(torch.nn.Module):
    def __init__(self, labels, action=0, wrong_scene=False):
        super().__init__()
        self.config = CoreConfig(hash_buckets=32, hidden_size=16)
        self.marker = torch.nn.Parameter(torch.tensor(float(action)))
        self.labels = labels
        self.action = action
        self.wrong_scene = wrong_scene

    def forward(self, features, observations=None):
        n = len(features)
        scene = {}
        for name, choices in SCENE_LABELS.items():
            logits = torch.zeros(n, len(choices))
            label = (self.labels[name] + int(self.wrong_scene)) % len(choices)
            logits[:, label] = 1
            scene[name] = logits
        actions = torch.zeros(n, 6)
        actions[:, self.action] = 1
        return {"scene": scene, "action": actions, "memory": torch.zeros(n, 16)}


class AuditEnsemble(torch.nn.Module):
    def __init__(self, constant=False, nonfinite=False):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(1.0))
        self.constant = constant
        self.nonfinite = nonfinite

    def forward(self, latent, observation, action):
        disagreement = (observation[:, 0] - 0.1).square()
        if self.constant:
            disagreement = torch.ones_like(disagreement)
        if self.nonfinite:
            disagreement[:] = float("nan")
        return {"mean_delta": observation.clone(), "disagreement": disagreement}


@pytest.fixture
def audit_setup(monkeypatch):
    monkeypatch.setattr(adaptation, "TidePool", AuditWorld)
    rows = generate_scene_examples(1, seed=123, split="test")
    core = AuditCore(rows[0]["labels"])
    ens = AuditEnsemble()
    criteria = PromotionCriteria(min_survival_rate=0.9, max_dynamics_mse=1.0, min_epistemic_correlation=0.1)
    return PromotionAuditor(criteria), core, ens, rows


def complete_audit(setup, **kwargs):
    auditor, core, ens, rows = setup
    options = {"incumbent_model": core, "incumbent_ensemble": ens, "scene_examples": rows}
    options.update(kwargs)
    return auditor.audit_candidate(core, ens, test_seeds=[101, 102], **options)


def test_missing_incumbent_and_scenes_never_claim_promotion(audit_setup):
    auditor, core, ens, _ = audit_setup
    report = auditor.audit_candidate(core, ens, test_seeds=[101])
    assert report["promoted"] is False
    assert report["eligible"] is False
    assert report["evidence_complete"] is False
    assert {"incumbent_model", "incumbent_ensemble", "scene_examples"} <= set(report["missing_evidence"])


def test_complete_passing_audit_only_grants_eligibility(audit_setup):
    report = complete_audit(audit_setup)
    assert report["eligible"] is True
    assert report["evidence_complete"] is True
    assert report["promoted"] is False
    assert report["metrics"]["scene_exact_accuracy"] == 1.0
    assert len(report["paired_episodes"]) == 2
    assert report["paired_episodes"][0]["survival_delta"] == 0
    assert report["calibration"]["valid"] is True
    assert report["calibration"]["failure_probability_calibrated"] is False


def test_scene_regression_is_actually_measured(audit_setup):
    auditor, core, ens, rows = audit_setup
    candidate = AuditCore(rows[0]["labels"], wrong_scene=True)
    report = auditor.audit_candidate(candidate, ens, test_seeds=[101], incumbent_model=core,
                                     incumbent_ensemble=ens, scene_examples=rows)
    assert report["metrics"]["scene_exact_accuracy"] == 0.0
    assert report["criteria_passed"]["scene_accuracy"] is False
    assert report["eligible"] is False


def test_incumbent_survival_cannot_be_ignored(audit_setup):
    auditor, core, ens, rows = audit_setup
    auditor.criteria = PromotionCriteria(min_survival_rate=0.0, max_dynamics_mse=1.0, min_scene_accuracy=0.0)
    candidate = AuditCore(rows[0]["labels"], action=1)
    report = auditor.audit_candidate(candidate, ens, test_seeds=[101], incumbent_model=core,
                                     incumbent_ensemble=ens, scene_examples=rows)
    assert report["criteria_passed"]["incumbent_survival"] is False
    assert report["paired_episodes"][0]["survival_delta"] == -1
    assert report["eligible"] is False


def test_constant_disagreement_does_not_pass_calibration(audit_setup):
    auditor, core, _, rows = audit_setup
    ens = AuditEnsemble(constant=True)
    auditor.criteria = PromotionCriteria(min_survival_rate=0, max_dynamics_mse=1, min_epistemic_correlation=0)
    report = auditor.audit_candidate(core, ens, test_seeds=[101], incumbent_model=core,
                                     incumbent_ensemble=ens, scene_examples=rows)
    assert report["calibration"]["valid"] is False
    assert report["eligible"] is False
    assert report["metrics"]["epistemic_correlation"] is None


@pytest.mark.parametrize("seeds", [[], [1, 1], [float("nan")], [True]])
def test_invalid_audit_seed_sets_fail_early(audit_setup, seeds):
    auditor, core, ens, _ = audit_setup
    with pytest.raises(ValueError):
        auditor.audit_candidate(core, ens, test_seeds=seeds)


def test_nonfinite_predictions_and_wrong_scene_split_are_rejected(audit_setup):
    auditor, core, ens, rows = audit_setup
    with pytest.raises(ValueError, match="finite"):
        auditor.audit_candidate(core, AuditEnsemble(nonfinite=True), test_seeds=[101])
    with pytest.raises(ValueError, match="test"):
        complete_audit(audit_setup, scene_examples=generate_scene_examples(1, split="train"))
    with pytest.raises(ValueError, match="non-empty"):
        complete_audit(audit_setup, scene_examples=[])


def test_episodic_assay_uses_retrieval_not_unused_neural_state(monkeypatch):
    intact = run_delayed_recall_benchmark(delays=[3], episodes_per_condition=40, seed=5)
    monkeypatch.setattr(EpisodicMemoryStore, "retrieve_by_resource", lambda self, resource: [])
    erased = run_delayed_recall_benchmark(delays=[3], episodes_per_condition=40, seed=5)
    assert intact["persistent_memory"][3] == 1.0
    assert erased["persistent_memory"][3] < 1.0


def test_episodic_assay_receipts_show_matched_action_support_and_donor_records():
    report = run_delayed_recall_benchmark(delays=[3], episodes_per_condition=60, seed=8, include_receipts=True)
    assert report["neural_memory_evaluated"] is False
    assert report["action_support"] == [1, 2, 3]
    assert len(report["episodes"]) == 180
    shuffled = [row for row in report["episodes"] if row["condition"] == "shuffled_memory"]
    assert all(row["action"] in [1, 2, 3] for row in report["episodes"])
    assert all(row["source_episode"] is not None for row in shuffled)
    assert all(row["action"] == row["retrieved_action"] for row in shuffled)
    assert report == run_delayed_recall_benchmark(delays=[3], episodes_per_condition=60, seed=8, include_receipts=True)
