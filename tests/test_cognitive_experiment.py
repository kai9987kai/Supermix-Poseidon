import pytest
import copy
import hashlib
import json

from poseidon.cognitive_experiment import (
    ARMS,
    COGNITIVE_EXPERIMENT_SCHEMA,
    run_cognitive_benchmark,
    verify_cognitive_receipt,
)
from poseidon.core import active_core_path


def test_micro_cognitive_benchmark():
    result = run_cognitive_benchmark(
        core_path=active_core_path(),
        seed_base=90000000,
        episodes=2,
        max_steps=4,
        scarcity=2.5,
        use_pomdp=False,
    )

    assert result["schema"] == COGNITIVE_EXPERIMENT_SCHEMA
    assert "sha256" in result
    for arm in ARMS:
        assert arm in result["arms"]
        assert result["arms"][arm]["episodes"] == 2

    assert "intact_vs_frozen_tidal" in result["paired"]
    assert "intact_vs_memory_free" in result["paired"]

    # Verify receipt hash
    assert verify_cognitive_receipt(result) is True


def _reseal(receipt):
    receipt.pop("sha256", None)
    receipt["sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    return receipt


def test_empty_resealed_receipt_rejected():
    with pytest.raises(ValueError):
        verify_cognitive_receipt(_reseal({"schema": COGNITIVE_EXPERIMENT_SCHEMA,
                                        "episodes_data": {}}))


@pytest.fixture(scope="module", params=[False, True], ids=["normal", "noisy"])
def replay_receipt(request):
    return run_cognitive_benchmark(episodes=1, max_steps=2, seed_base=91234567,
                                   use_pomdp=request.param)


def test_cognitive_full_replay(replay_receipt):
    # Exercise the persisted JSON representation, including sensor RNG tuples.
    assert verify_cognitive_receipt(json.loads(json.dumps(replay_receipt))) is True


@pytest.mark.parametrize("mutation", ["schema", "action", "reward", "death", "missing_pair",
                                       "duplicate_pair", "summary", "model", "source", "extra_field"])
def test_cognitive_resealed_mutants_fail(replay_receipt, mutation):
    receipt = copy.deepcopy(replay_receipt)
    episode = receipt["episodes_data"]["intact"][0]
    if mutation == "schema":
        receipt["schema"] = "unsupported"
    elif mutation == "action":
        episode["trajectory"][0]["action"] = (episode["trajectory"][0]["action"] + 1) % 6
    elif mutation == "reward":
        episode["trajectory"][0]["reward"] += 1.0
    elif mutation == "death":
        episode["survived"] = not episode["survived"]
    elif mutation == "missing_pair":
        receipt["episodes_data"]["intact"] = []
    elif mutation == "duplicate_pair":
        receipt["episodes_data"]["intact"].append(copy.deepcopy(episode))
    elif mutation == "summary":
        receipt["arms"]["intact"]["mean_reward"] += 1.0
    elif mutation == "model":
        receipt["model"]["checkpoint_sha256"] = "0" * 64
    elif mutation == "source":
        receipt["source_sha256"]["world.py"] = "0" * 64
    else:
        receipt["fabricated_claim"] = 42
    with pytest.raises(ValueError):
        verify_cognitive_receipt(_reseal(receipt))
