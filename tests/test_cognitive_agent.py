import pytest

from poseidon.cognitive_agent import CognitiveAgent, CognitiveDecision
from poseidon.core import CoreRuntime, active_core_path
from poseidon.mnemosyne import MNEMOSYNE_MODES
from poseidon.world import N_ACTIONS, OBS_SIZE, TidePool
from poseidon.worldlab import POMDPTidePool


@pytest.fixture(scope="module")
def core():
    return CoreRuntime(active_core_path())


def test_cognitive_agent_act_and_observe(core):
    agent = CognitiveAgent(core=core, mode="intact")
    agent.reset()

    obs = [0.5] * OBS_SIZE
    decision = agent.act(obs)

    assert isinstance(decision, CognitiveDecision)
    assert 0 <= decision.action < N_ACTIONS
    assert isinstance(decision.override, bool)
    assert isinstance(decision.margin, float)
    assert decision.mode == "intact"
    assert 0.0 <= decision.active_sparsity <= 1.0
    assert 0.5 <= decision.neuromodulatory_gain <= 2.0

    d_dict = decision.to_dict()
    assert d_dict["action"] == decision.action

    # Observe transition
    next_obs = [0.52] * OBS_SIZE
    agent.observe_transition(obs, decision.action, 0.5, next_obs, done=False)
    assert agent.predicted_next_obs is not None
    assert len(agent.predicted_next_obs) == OBS_SIZE

    # Next step should have surprise calculated from the prediction
    decision2 = agent.act(next_obs)
    assert decision2.surprise > 0.0


def test_cognitive_agent_rollout_tidepool(core):
    agent = CognitiveAgent(core=core, mode="intact")
    env = TidePool(seed=42, max_steps=8)
    res = agent.rollout(env, max_steps=8)

    assert res["schema"] == "poseidon-cognitive-agent-v2"
    assert res["steps"] > 0
    assert isinstance(res["survived"], bool)
    assert isinstance(res["total_reward"], float)
    assert "override_rate" in res
    assert len(res["decisions"]) == res["steps"]


def test_cognitive_agent_rollout_pomdp(core):
    agent = CognitiveAgent(core=core, mode="intact")
    env = POMDPTidePool(max_steps=8)
    res = agent.rollout(env, max_steps=8)

    assert res["steps"] > 0
    assert isinstance(res["total_reward"], float)


def test_cognitive_agent_matched_lesion_modes(core):
    agent = CognitiveAgent(core=core)
    for mode in MNEMOSYNE_MODES:
        env = TidePool(seed=101, max_steps=4)
        res = agent.rollout(env, max_steps=4, mode=mode)
        assert res["mode"] == mode
        assert res["steps"] > 0
