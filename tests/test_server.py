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
