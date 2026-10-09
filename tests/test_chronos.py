import pytest

from poseidon.chronos import ChronosBeaconClock, GhostTraceAuditor


def test_chronos_beacon_clock_cycles():
    clock = ChronosBeaconClock(period=4)
    assert clock.tick == 0
    assert clock.cycle == 0

    p1 = clock.step()
    assert p1.phase == 1
    assert not p1.syn_pulse

    p2 = clock.step()
    p3 = clock.step()
    assert not p3.syn_pulse

    p4 = clock.step()
    assert p4.phase == 0
    assert p4.cycle == 1
    assert p4.syn_pulse
    assert len(p4.pulse_sha256) == 64


def test_ghost_trace_auditor_sdi():
    auditor = GhostTraceAuditor(trace_decay=0.8, n_actions=6)
    
    # Step where actual == phantom
    sdi_0 = auditor.record_step(actual_action=1, phantom_action=1)
    assert sdi_0 == 0.0

    # Step where actual != phantom
    sdi_1 = auditor.record_step(actual_action=2, phantom_action=4)
    assert sdi_1 > 0.0

    summary = auditor.summary()
    assert summary["steps_recorded"] == 2
    assert summary["mean_spectral_divergence"] > 0.0
