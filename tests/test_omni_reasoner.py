"""Comprehensive tests for Stage 14 Omni-Reasoner & Cognitive Coprocessor."""
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest

from poseidon.omni_reasoner import OmniReasoner, OmniReasoningResult
from poseidon.runtime import Poseidon
from poseidon.server import make_handler
from http.server import ThreadingHTTPServer
import threading


@pytest.fixture
def test_server(tmp_path):
    runtime = Poseidon(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, 8787))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}", tmp_path
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=2)


def post_json(url, data):
    headers = {"Content-Type": "application/json"}
    return urlopen(Request(url, json.dumps(data).encode(), headers))


def test_intent_classification():
    reasoner = OmniReasoner(seed=42)
    assert reasoner.classify_intent("How do we cross a river when stepping stones sink?") == "physical_discovery"
    assert reasoner.classify_intent("Synthesize an enzyme for polyester cleavage") == "biocatalytic_synthesis"
    assert reasoner.classify_intent("Verify formal safety contracts under LTL model checking") == "formal_verification"
    assert reasoner.classify_intent("Calculate Clifford Cl(3,0) multivector rotor rotation") == "clifford_geometric"
    assert reasoner.classify_intent("Evaluate Causeway quantum superposition wavefunction collapse") == "causal_superposition"
    assert reasoner.classify_intent("What is the capital of France?") == "general_hyper_reasoning"


def test_cognitive_drives():
    reasoner = OmniReasoner(seed=42)
    drives = reasoner.compute_cognitive_drives("Cross wide river with tools", "physical_discovery")
    assert 0.0 <= drives["curiosity"] <= 1.0
    assert 0.0 <= drives["epistemic_uncertainty"] <= 1.0
    assert 0.0 <= drives["rigor_pressure"] <= 1.0
    assert 0.0 <= drives["causal_drive"] <= 1.0
    assert 0.0 <= drives["entropy"] <= 1.0


def test_physical_discovery_simulation():
    reasoner = OmniReasoner(seed=42)
    sim = reasoner._simulate_physical_discovery(seed=42)
    assert sim["narrow_channel"]["stepping_stones_viable"] is True
    assert sim["wide_channel"]["stepping_stones_viable"] is False
    assert sim["granite_physics"]["floats"] is False
    assert sim["granite_physics"]["net_buoyant_force_n"] < 0
    assert sim["pine_log_physics"]["net_buoyant_force_n"] > 0
    assert sim["pine_log_physics"]["single_log_payload_kg"] == 36.0
    assert sim["pine_log_physics"]["raft_3log_payload_kg"] == 108.0
    assert sim["pine_log_physics"]["raft_supports_agent"] is True
    assert sim["biocatalytic_binding"]["curing_ticks"] <= 5


def test_biocatalytic_enzyme_simulation():
    reasoner = OmniReasoner(seed=42)
    sim = reasoner._simulate_biocatalytic_synthesis(seed=42)
    assert "enzyme_name" in sim
    assert len(sim["active_site"]["residues"]) == 3
    assert sim["michaelis_menten_kinetics"]["delta_delta_g_barrier_kcal"] <= -8.0
    assert sim["michaelis_menten_kinetics"]["v_max_umol_min_mg"] > 0.0
    assert sim["curing_properties"]["bond_tensile_strength"] >= 0.95


def test_clifford_geometric_simulation():
    reasoner = OmniReasoner(seed=42)
    sim = reasoner._simulate_clifford_geometric()
    assert sim["isometry_invariant_preserved"] is True
    assert abs(sim["initial_velocity"]["norm"] - sim["rotated_velocity"]["norm"]) < 1e-5
    assert "vortex_lorentz_force" in sim


def test_causal_superposition_simulation():
    reasoner = OmniReasoner(seed=42)
    sim = reasoner._simulate_causal_superposition()
    assert sim["num_superposed_states"] == 6
    assert sim["evolved_entropy_nats"] <= sim["initial_entropy_nats"]
    assert 0.0 <= sim["optimal_action_probability"] <= 1.0
    assert sim["optimal_collapsed_action"] in ("REST", "FORAGE", "DRINK", "SHELTER", "EXPLORE", "FLEE")


def test_formal_ltl_safety_verification():
    reasoner = OmniReasoner(seed=42)
    proof = reasoner._verify_formal_safety("physical_discovery")
    assert proof["verdict"] == "PROVEN_SAFE"
    assert proof["contract_satisfied"] is True
    assert proof["shield_intervened"] is False
    assert len(proof["certificate_sha256"]) == 64


def test_full_reasoning_pipeline(tmp_path):
    runtime = Poseidon(tmp_path)
    res = runtime.reason("How do I cross a 10m wide river when stepping stones submerge?")
    assert res["intent"] == "physical_discovery"
    assert res["verified"] is True
    assert len(res["thought_steps"]) == 5
    assert len(res["certificate_sha256"]) == 64
    assert res["formal_proof"]["verdict"] == "PROVEN_SAFE"
    assert "Autonomous Cognitive Discovery" in res["text"]

    # Test via respond with mode="reason"
    resp = runtime.respond("Synthesize an enzyme for wood crosslinking", mode="reason")
    assert resp["intent"] == "biocatalytic_synthesis"
    assert resp["verified"] is True


def test_server_reasoning_endpoints(test_server):
    url, _ = test_server

    # 1. Test GET /api/reasoning-status
    status_res = json.load(urlopen(url + "/api/reasoning-status"))
    assert status_res["ready"] is True
    assert "GhostInTheMachine" in status_res["deliberation"]
    assert "physical_discovery" in status_res["intents"]

    # 2. Test POST /api/reason
    payload = {
        "prompt": "Construct a buoyant raft across a wide river channel",
        "seed": 42
    }
    reason_res = json.load(post_json(url + "/api/reason", payload))
    assert reason_res["verified"] is True
    assert reason_res["intent"] == "physical_discovery"
    assert len(reason_res["thought_steps"]) == 5
    assert len(reason_res["certificate_sha256"]) == 64
