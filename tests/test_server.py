import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
import pytest
from poseidon.server import make_handler
from poseidon.runtime import Poseidon

@pytest.fixture
def server(tmp_path):
    runtime = Poseidon(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, 8787))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}", tmp_path
    srv.shutdown(); srv.server_close(); thread.join(timeout=2)

def post(url, data, origin=None):
    headers = {"Content-Type": "application/json"}
    if origin: headers["Origin"] = origin
    return urlopen(Request(url, json.dumps(data).encode(), headers))

def test_math_and_status_work_without_model(server):
    url, _ = server
    status = json.load(urlopen(url+"/api/status"))
    assert not status["core_ready"]
    result = json.load(post(url+"/api/respond", {"prompt":"3*x+7=22","mode":"math"}))
    assert result["answer"] == "5"
    assert result["backend"] == "exact-rational-linear-solver"

def test_missing_model_fails_explicitly(server):
    url, _ = server
    with pytest.raises(HTTPError) as error:
        post(url+"/api/respond", {"prompt":"a red cube", "mode":"image"})
    assert error.value.code == 500
    assert "checkpoint" in json.load(error.value)["error"].lower()

def test_artifacts_cannot_escape_or_expose_memory(server):
    url, root = server
    (root/"outputs").mkdir()
    (root/"secret.json").write_text('{"private":true}')
    (root/"outputs/memory.json").write_text('{"fact":"private"}')
    for path in ("/artifacts/../secret.json", "/artifacts/memory.json"):
        with pytest.raises(HTTPError) as error:
            urlopen(url+path)
        assert error.value.code == 403

def test_cross_origin_mutation_rejected(server):
    url, _ = server
    with pytest.raises(HTTPError) as error:
        post(url+"/api/remember", {"text":"unexpected"}, "https://example.org")
    assert error.value.code == 403


def test_experiment_request_is_bounded_and_atlas_is_explicit(server):
    url, _ = server
    for settings in ({"episodes": 9}, {"max_steps": 1000}, {"episodes": True}, {"unknown": 1}):
        with pytest.raises(HTTPError) as error:
            post(url+"/api/experiment", settings)
        assert error.value.code == 400
    with pytest.raises(HTTPError) as error:
        post(url+"/api/experiment", {"episodes": 1, "max_steps": 32})
    assert error.value.code == 500
    assert "checkpoint" in json.load(error.value)["error"].lower()


def test_unknown_planner_is_rejected_without_loading_model(server):
    url, _ = server
    with pytest.raises(HTTPError) as error:
        post(url+"/api/respond", {"prompt": "Survive", "mode": "world", "planner": "invented"})
    assert error.value.code == 400


def test_odyssey_experiment_request_is_bounded_and_requires_checkpoint(server):
    url, _ = server
    for settings in ({"episodes": 9}, {"max_steps": 1000}, {"episodes": True}, {"unknown": 1}):
        with pytest.raises(HTTPError) as error:
            post(url+"/api/odyssey-experiment", settings)
        assert error.value.code == 400
    with pytest.raises(HTTPError) as error:
        post(url+"/api/odyssey-experiment", {"episodes": 1, "max_steps": 32})
    assert error.value.code == 500
    assert "checkpoint" in json.load(error.value)["error"].lower()


def test_odysseus_and_mco_experiment_request_validation(server):
    url, _ = server
    for settings in ({"episodes": 9}, {"max_steps": 1000}, {"episodes": True}, {"unknown": 1}):
        with pytest.raises(HTTPError) as error:
            post(url+"/api/odysseus-experiment", settings)
        assert error.value.code == 400
    with pytest.raises(HTTPError) as error:
        post(url+"/api/odysseus-experiment", {"episodes": 1, "max_steps": 32})
    assert error.value.code == 500
    assert "checkpoint" in json.load(error.value)["error"].lower()


def test_metamorph_experiment_request_validation(server):
    url, _ = server
    for settings in ({"episodes": 9}, {"max_steps": 1000}, {"episodes": True}, {"unknown": 1}):
        with pytest.raises(HTTPError) as error:
            post(url+"/api/metamorph-experiment", settings)
        assert error.value.code == 400
    with pytest.raises(HTTPError) as error:
        post(url+"/api/metamorph-experiment", {"episodes": 1, "max_steps": 32})
    assert error.value.code == 500
    assert "checkpoint" in json.load(error.value)["error"].lower()


def test_ecosystem_and_memory_endpoints(server):
    url, _ = server
    # Test GET /api/ecosystem/telemetry
    res = json.load(urlopen(url+"/api/ecosystem/telemetry"))
    assert "alive_counts" in res
    assert res["tick"] >= 1

    # Test GET /api/memory/topology
    top_res = json.load(urlopen(url+"/api/memory/topology"))
    assert "metrics" in top_res
    assert "betti_0" in top_res["metrics"]
    assert top_res["metrics"]["betti_0"] >= 1

    # Test POST /api/tessera/shield
    shield_res = json.load(post(url+"/api/tessera/shield", {
        "obs": [1.0, 0.05, 0.8, 0.5, 0.1, 0.05, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.0, 0.0, 0.5, 0.5],
        "action": 4,
    }))
    assert shield_res["shield_triggered"] is True
    assert shield_res["certified_action"] == 1
    assert len(shield_res["certificate_sha256"]) == 64

    # Test GET /api/autopoiesis/affect
    aff_res = json.load(urlopen(url+"/api/autopoiesis/affect"))
    assert "affect" in aff_res
    assert "curiosity_drive" in aff_res

    # Test GET /api/hypergraph/theorems
    thm_res = json.load(urlopen(url+"/api/hypergraph/theorems"))
    assert "@graph" in thm_res
    assert len(thm_res["@graph"]) >= 4


def test_external_adapter_server_endpoints(server):
    url, _ = server
    # 1. GET /api/adapter/status
    stat_res = json.load(urlopen(url+"/api/adapter/status"))
    assert stat_res["ready"] is True
    assert stat_res["protocol_version"] == "poseidon-external-adapter-v1"
    assert "SI_METRIC" in stat_res["supported_units"]

    # 2. GET /api/telemetry/memory-influence
    tel_res = json.load(urlopen(url+"/api/telemetry/memory-influence"))
    assert "dv_gem_active" in tel_res
    assert len(tel_res["dv_gem_deltas"]) == 6

    # 3. POST /api/adapter/handshake
    handshake_payload = {
        "client_id": "test-suite-runner",
        "engine_name": "pytest-sim",
        "identity": {
            "adapter_id": "sim-adapter-01",
            "units": "SI_METRIC",
            "ticks_per_second": 30.0,
            "coordinate_frame": "FLU",
            "spatial_bounds": [100.0, 100.0, 20.0],
            "max_linear_velocity": 15.0,
        },
    }
    hs_res = json.load(post(url+"/api/adapter/handshake", handshake_payload))
    assert hs_res["accepted"] is True
    assert hs_res["client_id"] == "test-suite-runner"
    token = hs_res["session_token"]

    # 4. POST /api/adapter/verify-3d-replay
    replay_payload = {
        "schema": "poseidon-3d-replay-v1",
        "adapter_identity": {"adapter_id": "sim-adapter-01", "max_linear_velocity": 15.0},
        "total_reward": 1.0,
        "keyframes": [
            {"tick": 0, "delta_time": 0.05, "position": [0.0, 0.0, 0.0], "action": 0, "reward": 1.0},
        ],
    }
    rep_res = json.load(post(url+"/api/adapter/verify-3d-replay", replay_payload))
    assert rep_res["verified"] is True
    assert rep_res["transitions_verified"] == 1
    assert len(rep_res["receipt_sha256"]) == 64

    # 5. POST /api/adapter/cancel
    cancel_res = json.load(post(url+"/api/adapter/cancel", {"session_token": token}))
    assert cancel_res["cancelled"] is True
    assert cancel_res["session_token"] == token



