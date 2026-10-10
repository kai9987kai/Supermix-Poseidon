import pytest

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
