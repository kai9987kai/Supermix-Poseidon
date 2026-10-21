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

    # 4. POST /api/adapter/tick with controller="nexus_omega"
    tick_payload = {
        "session_token": token,
        "tick": 0,
        "delta_time": 0.05,
        "observation_16d": [0.5] * 16,
        "controller": "nexus_omega",
    }
    tick_res = json.load(post(url+"/api/adapter/tick", tick_payload))
    assert tick_res["tick"] == 0
    assert "action" in tick_res
    assert tick_res["action"] in range(4)

    # 5. POST /api/adapter/verify-3d-replay
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

    # 6. POST /api/adapter/cancel
    cancel_res = json.load(post(url+"/api/adapter/cancel", {"session_token": token}))
    assert cancel_res["cancelled"] is True
    assert cancel_res["session_token"] == token


def test_nexus_omega_server_endpoints(server):
    url, _ = server
    # 1. GET /api/nexus-omega/status
    stat = json.load(urlopen(url + "/api/nexus-omega/status"))
    assert stat["ready"] is True
    assert stat["version"] in ("1.0.0", "2.0.0")
    assert "causal_memory_fabric" in stat["subsystems"]
    assert "quantum_causal_kernel" in stat["subsystems"]
    assert stat.get("nexus_ultima") is True
    assert "PROTOCELL" in stat["developmental_stages"]

    # 2. POST /api/nexus-omega/discover
    disc_payload = {
        "seed_base": 50000,
        "num_seeds": 2,
        "max_steps": 8,
    }
    disc_res = json.load(post(url + "/api/nexus-omega/discover", disc_payload))
    assert disc_res["verified"] is True
    assert "ledger_id" in disc_res
    assert len(disc_res["ledger_sha256"]) == 64
    assert disc_res["ratified_laws_count"] >= 0

    # 3. GET /api/nexus-omega/factored-state
    fac_res = json.load(urlopen(url + "/api/nexus-omega/factored-state"))
    assert "action" in fac_res
    assert "decision" in fac_res
    assert "factored_state" in fac_res
    assert "endogenous" in fac_res["factored_state"]
    assert "exogenous" in fac_res["factored_state"]
    assert "causal_regime" in fac_res["factored_state"]
    assert "free_energy_objective" in fac_res["decision"]

    # 4. GET /api/nexus-omega/hierarchical-state
    hier_res = json.load(urlopen(url + "/api/nexus-omega/hierarchical-state"))
    assert "micro_action" in hier_res
    assert "macro_intent" in hier_res
    assert hier_res["micro_action"] in range(4)
    assert hier_res["macro_intent"] in [
        "forage_expedition",
        "diapause_hibernation",
        "habitat_relocation",
        "emergency_fortification",
    ]
    assert "decision" in hier_res
    assert "imagined_trajectory" in hier_res

    # 5. POST /api/nexus-omega/imagine
    imagine_payload = {
        "intent": "forage_expedition",
        "horizon": 6,
        "seed": 42,
    }
    img_res = json.load(post(url + "/api/nexus-omega/imagine", imagine_payload))
    assert img_res["horizon"] == 6
    assert img_res["macro_intent"] == "forage_expedition"
    assert len(img_res["states"]) == 6
    assert len(img_res["actions"]) == 6
    assert "conformal_return_lcb" in img_res

    # 6. GET /api/nexus-omega/ultima-state
    ult_res = json.load(urlopen(url + "/api/nexus-omega/ultima-state"))
    assert "action" in ult_res
    assert "decision" in ult_res
    assert "quantum_state" in ult_res
    assert "spectral_profile" in ult_res
    assert "thermodynamic_state" in ult_res
    assert "active_opcode" in ult_res
    assert ult_res["quantum_state"]["purity"] > 0.0
    assert ult_res["spectral_profile"]["dominant_period"] > 0.0
    assert ult_res["thermodynamic_state"]["autopoietic_viability"] > 0.0

    # 7. POST /api/nexus-omega/dream
    dream_res = json.load(post(url + "/api/nexus-omega/dream", {"num_rollouts": 2, "seed": 42}))
    assert "replayed_transitions" in dream_res
    assert "mean_regret_mitigation" in dream_res
    assert "synaptic_pruning_ratio" in dream_res

    # 8. POST /api/nexus-omega/synthesize-opcode
    op_res = json.load(post(url + "/api/nexus-omega/synthesize-opcode", {
        "intent": "forage_expedition",
        "seed": 42
    }))
    assert "opcode_name" in op_res
    assert "execution_steps" in op_res
    assert "ltl_safe" in op_res
    assert len(op_res["verification_certificate"]) == 64


def test_context_expansion_and_token_budget(server):
    url, root = server
    runtime = Poseidon(root)

    # 1. Test 32000-character prompt boundary on runtime
    padded_math = "3*x + 7 = 22" + " " * (32000 - len("3*x + 7 = 22"))
    assert len(padded_math) == 32000
    res = runtime.respond(padded_math, mode="math")
    assert res["answer"] == "5"

    with pytest.raises(ValueError, match="1–32000"):
        runtime.respond("x" * 32001, mode="math")

    # 2. Test server HTTP endpoint with 32000-char prompt and max_new_tokens parameter
    res = json.load(post(url + "/api/respond", {
        "prompt": padded_math,
        "mode": "math",
        "max_new_tokens": 1024
    }))
    assert res["answer"] == "5"

    # 3. Test HTTP 400 rejection when prompt exceeds 32000 characters
    with pytest.raises(HTTPError) as err:
        post(url + "/api/respond", {"prompt": "x" * 32001, "mode": "math"})
    assert err.value.code == 400


def test_discovery_lab_server_endpoints(server):
    url, _ = server

    # 1. GET /api/discovery-lab/status
    status = json.load(urlopen(url + "/api/discovery-lab/status"))
    assert status["ready"] is True
    assert "river_world_simulation" in status["subsystems"]

    # 2. GET /api/discovery-lab/river-state
    state = json.load(urlopen(url + "/api/discovery-lab/river-state?seed=42&river_width=3"))
    assert state["river_width"] == 3
    assert "agent" in state
    assert "objects" in state
    assert len(state["objects"]) > 0
    assert len(state["state_hash"]) == 64

    # 3. POST /api/discovery-lab/step
    step_res = json.load(post(url + "/api/discovery-lab/step", {"action": 1, "seed": 42, "river_width": 3}))
    assert "action_executed" in step_res
    assert len(step_res["observation"]) == 24
    assert "state_hash" in step_res


def test_problem_solver_server_endpoints(server):
    url, _ = server

    # 1. GET /api/problem/capabilities
    caps = json.load(urlopen(url + "/api/problem/capabilities"))
    assert caps["ready"] is True
    assert len(caps["capabilities"]) >= 10
    domain_names = [c["domain"] for c in caps["capabilities"]]
    assert "water_jugs" in domain_names
    assert "sudoku" in domain_names
    assert "cryptarithm" in domain_names

    # 2. POST /api/problem/solve
    solve_res = json.load(post(url + "/api/problem/solve", {
        "prompt": "Solve SEND + MORE = MONEY"
    }))
    assert solve_res["domain"] == "cryptarithm"
    assert solve_res["verified"] is True
    assert "9567 + 1085 = 10652" in solve_res["answer"]

    # 3. POST /api/respond with mode="problem"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "Solve Tower of Hanoi with 3 disks",
        "mode": "problem"
    }))
    assert resp["domain"] == "tower_of_hanoi"
    assert resp["answer"] == "7"
    assert resp["verified"] is True


def test_hypothesis_server_endpoints(server):
    url, _ = server

    # 1. GET /api/hypothesis/status
    stat = json.load(urlopen(url + "/api/hypothesis/status"))
    assert stat["ready"] is True
    assert stat["popperian_demarcation"] is True
    assert stat["lakatos_msrp_evaluator"] is True

    # 2. POST /api/hypothesis/formulate
    res = json.load(post(url + "/api/hypothesis/formulate", {
        "conjecture": "Energy conservation holds across closed thermodynamic transformations",
        "domain": "thermodynamics",
        "hard_core_assumptions": ["First Law of Thermodynamics"],
        "falsification_criteria": ["Net energy generated ex nihilo in closed cycle"],
        "novel_predictions": ["Delta U = Q - W"],
    }))
    assert res["id"].startswith("hyp-")
    assert res["domain"] == "thermodynamics"
    assert res["status"] == "PROPOSED"

    # 3. POST /api/respond with mode="hypothesis"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "General Relativity curved spacetime predicts light deflection",
        "mode": "hypothesis"
    }))
    assert resp["id"].startswith("hyp-")


def test_proof_server_endpoints(server):
    url, _ = server

    # 1. GET /api/proof/status
    stat = json.load(urlopen(url + "/api/proof/status"))
    assert stat["ready"] is True
    assert stat["natural_deduction"] is True
    assert stat["mathematical_induction"] is True

    # 2. POST /api/proof/prove with algebraic identity
    res_alg = json.load(post(url + "/api/proof/prove", {
        "lhs": "(a - b) * (a + b)",
        "rhs": "a^2 - b^2",
    }))
    assert res_alg["proved"] is True
    assert res_alg["certificate"]["is_valid"] is True

    # 3. POST /api/proof/prove with natural deduction
    res_prop = json.load(post(url + "/api/proof/prove", {
        "premises": ["P", "P -> Q"],
        "goal": "Q",
    }))
    assert res_prop["proved"] is True
    assert res_prop["goal"] == "Q"

    # 4. POST /api/respond with mode="proof"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "(x + 1)^2 == x^2 + 2*x + 1",
        "mode": "proof",
    }))
    assert resp["proved"] is True


def test_creativity_server_endpoints(server):
    url, _ = server

    # 1. GET /api/creativity/status
    stat = json.load(urlopen(url + "/api/creativity/status"))
    assert stat["ready"] is True
    assert stat["koestler_bisociation"] is True
    assert stat["triz_contradiction_engine"] is True

    # 2. GET /api/creativity/matrices
    matrices = json.load(urlopen(url + "/api/creativity/matrices"))
    assert len(matrices) >= 8

    # 3. GET /api/creativity/principles
    principles = json.load(urlopen(url + "/api/creativity/principles"))
    assert len(principles) >= 11

    # 4. POST /api/creativity/invent
    res_inv = json.load(post(url + "/api/creativity/invent", {
        "problem": "Fault-tolerant swarm coordination under adversarial conditions",
        "domain_a": "quantum_mechanics",
        "domain_b": "distributed_consensus",
        "improving_param": "speed",
        "deteriorating_param": "precision",
    }))
    assert "Project DA VINCI" in res_inv["title"]
    assert res_inv["pareto_score"] > 0.8
    assert res_inv["triz_resolution"]["contradiction_resolved"] is True

    # 5. POST /api/respond with mode="invent"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "Scalable neural-symbolic memory consolidation",
        "mode": "invent",
    }))
    assert "Project DA VINCI" in resp["title"]


def test_athenaeum_server_endpoints(server):
    url, _ = server

    # 1. GET /api/athenaeum/status
    stat = json.load(urlopen(url + "/api/athenaeum/status"))
    assert stat["ready"] is True
    assert stat["egraph_equality_saturation"] is True
    assert stat["prm_mcts_test_time_compute"] is True
    assert stat["sindy_sparse_identification"] is True
    assert stat["rewrite_rules_count"] >= 20

    # 2. POST /api/athenaeum/synthesize
    res_ath = json.load(post(url + "/api/athenaeum/synthesize", {
        "expression": "(x + 0) * 1 + (x - x)",
        "target_expression": "x",
        "budget_rollouts": 12,
    }))
    assert res_ath["project"] == "ATHENAEUM"
    assert res_ath["verified"] is True
    assert res_ath["egraph_saturation"]["simplified_expression"] == "x"
    assert len(res_ath["receipt_sha256"]) == 64

    # 3. POST /api/respond with mode="athenaeum"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "(x + 0) * 1 + (x - x) == x",
        "mode": "athenaeum",
    }))
    assert resp["project"] == "ATHENAEUM"
    assert resp["verified"] is True


def test_hephaestus_server_endpoints(server):
    url, _ = server

    # 1. GET /api/hephaestus/status
    stat = json.load(urlopen(url + "/api/hephaestus/status"))
    assert stat["ready"] is True
    assert stat["wake_sleep_library_learning"] is True
    assert stat["gflownet_trajectory_balance"] is True
    assert stat["koopman_edmd_linearizer"] is True
    assert stat["buchberger_groebner_prover"] is True

    # 2. POST /api/hephaestus/synthesize
    res_hep = json.load(post(url + "/api/hephaestus/synthesize", {
        "terminal_rewards": {"M1": 4.0, "M2": 12.0, "M3": 8.0, "M4": 16.0},
    }))
    assert res_hep["project"] == "HEPHAESTUS"
    assert res_hep["verified"] is True
    assert res_hep["wake_sleep_library"]["compression_ratio"] > 1.2
    assert abs(res_hep["gflownet_sampler"]["partition_function_Z"] - 40.0) < 1e-9
    assert len(res_hep["receipt_sha256"]) == 64

    # 3. POST /api/respond with mode="hephaestus"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "(x ^ 2 + 2 * x * y) + 1; (a ^ 2 + 2 * a * b) - 3; (u ^ 2 + 2 * u * v) * 5",
        "mode": "hephaestus",
    }))
    assert resp["project"] == "HEPHAESTUS"
    assert resp["verified"] is True


def test_hyperborea_server_endpoints(server):
    url, _ = server

    # 1. GET /api/hyperborea/status
    stat = json.load(urlopen(url + "/api/hyperborea/status"))
    assert stat["ready"] is True
    assert stat["cellular_sheaf_cohomology"] is True
    assert stat["port_hamiltonian_dirac_control"] is True
    assert stat["renormalization_group_criticality"] is True
    assert stat["topological_anyon_braiding"] is True

    # 2. POST /api/hyperborea/synthesize
    res_hyp = json.load(post(url + "/api/hyperborea/synthesize", {
        "rg_initial_K": 0.5,
        "braid_word": ["sigma_1", "sigma_2", "sigma_1"],
    }))
    assert res_hyp["project"] == "HYPERBOREA"
    assert res_hyp["verified"] is True
    assert res_hyp["cellular_sheaf"]["dim_H0_global_sections"] == 2
    assert abs(res_hyp["renormalization_group"]["critical_coupling_Kc"] - 0.60937786) < 1e-6
    assert res_hyp["anyon_braiding"]["hexagon_artin_braid_residual"] < 1e-12
    assert len(res_hyp["receipt_sha256"]) == 64

    # 3. POST /api/respond with mode="hyperborea"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "Synthesize Project HYPERBOREA",
        "mode": "hyperborea",
    }))
    assert resp["project"] == "HYPERBOREA"
    assert resp["verified"] is True


def test_neurograft_server_endpoints(server):
    url, _ = server

    # 1. GET /api/neurograft/status
    stat = json.load(urlopen(url + "/api/neurograft/status"))
    assert stat["ready"] is True
    assert stat["species_connectomes"] == ["worm", "fly_female", "fly_male", "human"]
    assert stat["ltc_continuous_integration"] is True
    assert stat["tri_monoaminergic_neuromodulation"] is True
    assert stat["virtual_lesion_causal_attribution"] is True

    # 2. POST /api/neurograft/synthesize
    res_ng = json.load(post(url + "/api/neurograft/synthesize", {
        "seed": 42,
        "horizon": 4,
    }))
    assert res_ng["project"] == "NEUROGRAFT"
    assert res_ng["verified"] is True
    assert res_ng["tri_neuromodulation"]["behavioral_regime"] in ("goal-exploitation", "exploration", "hazard-avoidance", "homeostatic-rest")
    assert res_ng["virtual_lesion_audit"]["critical_lesion_target"] in ("worm", "fly_female", "fly_male", "human")
    assert len(res_ng["receipt_sha256"]) == 64

    # 3. POST /api/respond with mode="neurograft"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "Synthesize Project NEUROGRAFT",
        "mode": "neurograft",
    }))
    assert resp["project"] == "NEUROGRAFT"
    assert resp["verified"] is True


def test_aether_server_endpoints(server):
    url, _ = server

    # 1. GET /api/aether/status
    stat = json.load(urlopen(url + "/api/aether/status"))
    assert stat["ready"] is True
    assert stat["version"] == "v3.5.0-AETHER-60"
    assert stat["subsystems"]["gauge_field_holonomy"] is True
    assert stat["subsystems"]["jarzynski_non_equilibrium_thermodynamics"] is True
    assert stat["subsystems"]["symplectic_contact_mechanics"] is True
    assert stat["total_verified_domains"] == 60

    # 2. POST /api/aether/synthesize
    res_ae = json.load(post(url + "/api/aether/synthesize", {
        "grid_coords": [0, 0],
        "thermo_center_end": 2.0,
        "mu": 1.0,
        "sigma": 1.5,
        "symplectic_steps": 50,
    }))
    assert res_ae["project"] == "AETHER"
    assert res_ae["verified"] is True
    assert res_ae["gauge_holonomy"]["yang_mills_action"] >= 0.0
    assert res_ae["non_equilibrium_thermodynamics"]["second_law_satisfied"] is True
    assert res_ae["information_geometry"]["gaussian_curvature"] == -0.5
    assert res_ae["symplectic_mechanics"]["symplectic_conserved"] is True
    assert len(res_ae["receipt_sha256"]) == 64

    # 3. POST /api/respond with mode="aether"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "Synthesize Project AETHER",
        "mode": "aether",
    }))
    assert resp["project"] == "AETHER"
    assert resp["verified"] is True


def test_chronos_server_endpoints(server):
    url, _ = server

    # 1. GET /api/chronos/status
    stat = json.load(urlopen(url + "/api/chronos/status"))
    assert stat["ready"] is True
    assert stat["version"] == "v3.6.0-CHRONOS-65"
    assert stat["subsystems"]["causal_dynamical_triangulations"] is True
    assert stat["subsystems"]["noncommutative_spectral_geometry"] is True
    assert stat["subsystems"]["tropical_algebraic_geometry"] is True
    assert stat["subsystems"]["stiefel_grassmannian_optimization"] is True
    assert stat["subsystems"]["coalgebraic_bisimulation_metric"] is True
    assert stat["total_verified_domains"] == 65

    # 2. POST /api/chronos/synthesize
    res_ch = json.load(post(url + "/api/chronos/synthesize", {
        "seed": 42,
    }))
    assert res_ch["project"] == "CHRONOS"
    assert res_ch["verified"] is True
    assert res_ch["causal_dynamical_triangulation"]["time_slices"] == 4
    assert res_ch["spectral_geometry"]["spectral_action_tr_d2"] >= 0.0
    assert res_ch["stiefel_optimization"]["orthogonality_residual"] < 1e-12
    assert res_ch["coalgebraic_bisimulation"]["dist_s0_s1"] < res_ch["coalgebraic_bisimulation"]["dist_s0_s2"]
    assert len(res_ch["receipt_sha256"]) == 64

    # 3. POST /api/respond with mode="chronos"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "Synthesize Project CHRONOS",
        "mode": "chronos",
    }))
    assert resp["project"] == "CHRONOS"
    assert resp["verified"] is True


def test_noether_server_endpoints(server):
    url, _ = server

    # 1. GET /api/noether/status
    stat = json.load(urlopen(url + "/api/noether/status"))
    assert stat["ready"] is True
    assert stat["version"] == "v3.7.0-NOETHER-70"
    assert stat["subsystems"]["noether_symmetry_conservation"] is True
    assert stat["subsystems"]["modular_tensor_category"] is True
    assert stat["subsystems"]["clifford_spacetime_algebra"] is True
    assert stat["subsystems"]["batalin_vilkovisky_quantization"] is True
    assert stat["subsystems"]["active_matter_nonequilibrium"] is True
    assert stat["total_verified_domains"] == 70

    # 2. POST /api/noether/synthesize
    res_noe = json.load(post(url + "/api/noether/synthesize", {
        "seed": 42,
    }))
    assert res_noe["project"] == "NOETHER"
    assert res_noe["verified"] is True
    assert res_noe["noether_symmetry"]["symmetry_group"] == "SO(2)"
    assert res_noe["modular_tensor_category"]["verlinde_algebra_valid"] is True
    assert res_noe["clifford_spacetime_algebra"]["lorentz_invariance_verified"] is True
    assert res_noe["batalin_vilkovisky"]["cme_satisfied"] is True
    assert res_noe["active_matter"]["verified"] is True
    assert len(res_noe["receipt_sha256"]) == 64

    # 3. POST /api/respond with mode="noether"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "Synthesize Project NOETHER",
        "mode": "noether",
    }))
    assert resp["project"] == "NOETHER"
    assert resp["verified"] is True




def test_cartan_server_endpoints(server):
    url, _ = server

    # 1. GET /api/cartan/status
    stat = json.load(urlopen(url + "/api/cartan/status"))
    assert stat["ready"] is True
    assert stat["version"] == "v3.8.0-CARTAN-75"
    assert stat["subsystems"]["cartan_exterior_systems"] is True
    assert stat["subsystems"]["non_hermitian_pt_symmetry"] is True
    assert stat["subsystems"]["exceptional_jordan_algebra"] is True
    assert stat["subsystems"]["holographic_ryu_takayanagi"] is True
    assert stat["subsystems"]["morphogenetic_turing_reaction"] is True
    assert stat["total_verified_domains"] == 75

    # 2. POST /api/cartan/synthesize
    res_car = json.load(post(url + "/api/cartan/synthesize", {
        "seed": 42,
    }))
    assert res_car["project"] == "CARTAN"
    assert res_car["verified"] is True
    assert res_car["cartan_exterior"]["is_involutive_system"] is True
    assert res_car["non_hermitian_pt"]["pt_commutator_zero"] is True
    assert res_car["exceptional_jordan"]["hurwitz_norm_identity_verified"] is True
    assert res_car["holographic_entanglement"]["verified"] is True
    assert res_car["morphogenetic_turing"]["turing_instability_active"] is True
    assert len(res_car["receipt_sha256"]) == 64

    # 3. POST /api/respond with mode="cartan"
    resp = json.load(post(url + "/api/respond", {
        "prompt": "Synthesize Project CARTAN",
        "mode": "cartan",
    }))
    assert resp["project"] == "CARTAN"
    assert resp["verified"] is True
