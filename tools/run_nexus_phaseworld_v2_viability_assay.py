"""Evaluate whether a versioned NEXUS POMDP supports informative survival tests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.nexus_omega.agent import NexusOmegaAgent
from poseidon.nexus_omega.benchmark import POMDPPhaseWorld
from poseidon.nexus_omega.benchmark_v2 import POMDPPhaseWorldV2


SEEDS = tuple(range(99101, 99165))
AGENT_SEEDS = (55001, 55002)
HORIZON = 32
BOOTSTRAP_REPLICATES = 20_000
ARM_NAMES = (
    "legacy_v1_trace_memory",
    "v2_reactive_cue_only",
    "v2_trace_memory",
    "v2_agent_v1_prior_55001",
    "v2_agent_v2_prior_55001",
    "v2_agent_v1_prior_55002",
    "v2_agent_v2_prior_55002",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        means[i] = float(np.mean(values[rng.randint(0, len(values), size=len(values))]))
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _sign_test(values: np.ndarray) -> dict:
    nonzero = [float(x) for x in values if abs(float(x)) > 1e-12]
    n = len(nonzero)
    positive = sum(x > 0 for x in nonzero)
    tail = min(positive, n - positive)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)) if n else 1.0
    return {"n_nonzero": n, "positive": positive, "p_value": p}


def _policy_action(name: str, obs: np.ndarray, memory: dict) -> int:
    if name == "reactive":
        return 3 if float(obs[3]) > 0.8 else 0
    if name == "trace_memory":
        if float(obs[3]) > 0.8:
            memory["storm_ticks"] = 3
        if memory["storm_ticks"] > 0:
            memory["storm_ticks"] -= 1
            return 3
        return 0
    raise ValueError(f"unknown non-agent policy: {name}")


def _run_episode(seed: int, arm: str, agent_seed: int | None = None, horizon: int = HORIZON) -> dict:
    legacy = arm == "legacy_v1_trace_memory"
    env = POMDPPhaseWorld(seed=seed, max_steps=horizon) if legacy else POMDPPhaseWorldV2(seed=seed, max_steps=horizon)
    obs = env.reset()
    memory = {"storm_ticks": 0}
    agent = None
    profile = None
    policy = None
    if arm.startswith("v2_agent_"):
        if agent_seed is None:
            raise ValueError("agent arms require an agent seed")
        profile = "nexus-phaseworld-v2" if "v2_prior" in arm else "nexus-phaseworld-v1"
        agent = NexusOmegaAgent(base_seed=agent_seed, dynamics_profile=profile)
        agent.reset()
    elif arm == "v2_trace_memory" or legacy:
        policy = "trace_memory"
    elif arm == "v2_reactive_cue_only":
        policy = "reactive"
    else:
        raise ValueError(f"unknown arm: {arm}")

    observations = [[float(x) for x in obs]]
    actions: list[int] = []
    rewards: list[float] = []
    transitions: list[dict] = []
    for _ in range(horizon):
        action = int(agent.act(obs)) if agent is not None else _policy_action(policy, obs, memory)
        next_obs, reward, done, info = env.step(action)
        if agent is not None:
            agent.observe_transition(next_obs, reward, done)
        actions.append(action)
        rewards.append(float(reward))
        transitions.append({
            "is_storm": bool(info["is_storm"]),
            "death": bool(info["death"]),
            "truncated": bool(info.get("truncated", done and not info["death"])),
            "done": bool(done),
            "health": float(info["health"]),
            "energy": float(env.energy),
            "hydration": float(env.hydration),
            "version": info.get("version", "nexus-phaseworld-v1"),
            "health_shock_update": bool(agent.last_health_shock_update) if agent is not None else False,
            "regime_storm_ticks": float(agent.factored_world_model.z_regime[0]) if agent is not None else 0.0,
        })
        obs = next_obs
        observations.append([float(x) for x in obs])
        if done:
            break

    terminal = transitions[-1]
    return {
        "seed": seed,
        "arm": arm,
        "agent_seed": agent_seed,
        "dynamics_profile": profile,
        "survived": not terminal["death"],
        "death": terminal["death"],
        "truncated": terminal["truncated"],
        "steps": len(actions),
        "total_reward": float(sum(rewards)),
        "storm_forage_count": sum(
            1 for action, row in zip(actions, transitions) if row["is_storm"] and action == 0
        ),
        "terminal_health": terminal["health"],
        "terminal_energy": terminal["energy"],
        "terminal_hydration": terminal["hydration"],
        "actions": actions,
        "rewards": rewards,
        "observations": observations,
        "transitions": transitions,
    }


def _episode_metrics(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "survival_rate": float(np.mean([row["survived"] for row in rows])),
        "mean_reward": float(np.mean([row["total_reward"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_storm_forage_count": float(np.mean([row["storm_forage_count"] for row in rows])),
        "mean_terminal_health": float(np.mean([row["terminal_health"] for row in rows])),
        "mean_terminal_energy": float(np.mean([row["terminal_energy"] for row in rows])),
        "mean_terminal_hydration": float(np.mean([row["terminal_hydration"] for row in rows])),
    }


def _replay_check(episode: dict, horizon: int) -> bool:
    replayed = _run_episode(episode["seed"], episode["arm"], episode["agent_seed"], horizon)
    keys = ("survived", "death", "truncated", "steps", "total_reward", "actions", "rewards", "observations", "transitions")
    return all(replayed[key] == episode[key] for key in keys)


def run(
    environment_seeds: tuple[int, ...] = SEEDS,
    agent_seeds: tuple[int, ...] = AGENT_SEEDS,
    horizon: int = HORIZON,
) -> dict:
    episodes: list[dict] = []
    arm_names = (
        "legacy_v1_trace_memory",
        "v2_reactive_cue_only",
        "v2_trace_memory",
        *(f"v2_agent_v1_prior_{seed}" for seed in agent_seeds),
        *(f"v2_agent_v2_prior_{seed}" for seed in agent_seeds),
    )
    for seed in environment_seeds:
        episodes.append(_run_episode(seed, "legacy_v1_trace_memory", horizon=horizon))
        episodes.append(_run_episode(seed, "v2_reactive_cue_only", horizon=horizon))
        episodes.append(_run_episode(seed, "v2_trace_memory", horizon=horizon))
        for agent_seed in agent_seeds:
            episodes.append(_run_episode(seed, f"v2_agent_v1_prior_{agent_seed}", agent_seed, horizon))
            episodes.append(_run_episode(seed, f"v2_agent_v2_prior_{agent_seed}", agent_seed, horizon))

    replay_verified = sum(_replay_check(episode, horizon) for episode in episodes)
    if replay_verified != len(episodes):
        raise RuntimeError(f"trajectory replay failed for {len(episodes) - replay_verified} episodes")

    by_arm = {
        arm: [row for row in episodes if row["arm"] == arm]
        for arm in arm_names
    }
    summaries = {arm: _episode_metrics(rows) for arm, rows in by_arm.items()}

    paired = {}
    for agent_seed in agent_seeds:
        before = by_arm[f"v2_agent_v1_prior_{agent_seed}"]
        after = by_arm[f"v2_agent_v2_prior_{agent_seed}"]
        reward_delta = np.asarray([b["total_reward"] - a["total_reward"] for a, b in zip(before, after)], dtype=np.float64)
        survival_delta = np.asarray([int(b["survived"]) - int(a["survived"]) for a, b in zip(before, after)], dtype=np.float64)
        paired[str(agent_seed)] = {
            "v2_minus_v1_reward_mean": float(reward_delta.mean()),
            "v2_minus_v1_reward_95pct_ci": _bootstrap_ci(reward_delta, 1_009_001 + agent_seed),
            "v2_minus_v1_reward_sign_test": _sign_test(reward_delta),
            "v2_minus_v1_survival_delta": float(survival_delta.mean()),
            "v2_minus_v1_survival_95pct_ci": _bootstrap_ci(survival_delta, 1_009_101 + agent_seed),
            "survival_pairs": {
                "both_survived": int(sum(a["survived"] and b["survived"] for a, b in zip(before, after))),
                "v2_only_survived": int(sum(not a["survived"] and b["survived"] for a, b in zip(before, after))),
                "v1_only_survived": int(sum(a["survived"] and not b["survived"] for a, b in zip(before, after))),
                "neither_survived": int(sum(not a["survived"] and not b["survived"] for a, b in zip(before, after))),
            },
        }

    trace_v_reactive = np.asarray([
        int(trace["survived"]) - int(reactive["survived"])
        for trace, reactive in zip(by_arm["v2_trace_memory"], by_arm["v2_reactive_cue_only"])
    ], dtype=np.float64)
    viability_passed = summaries["v2_trace_memory"]["survival_rate"] >= 0.95

    source_paths = {
        "assay": Path(__file__).resolve(),
        "agent": ROOT / "poseidon/nexus_omega/agent.py",
        "benchmark_v1": ROOT / "poseidon/nexus_omega/benchmark.py",
        "benchmark_v2": ROOT / "poseidon/nexus_omega/benchmark_v2.py",
        "causal_world_model_v1": ROOT / "poseidon/nexus_omega/causal_world_model.py",
        "causal_world_model_v2": ROOT / "poseidon/nexus_omega/causal_world_model_v2.py",
        "causal_efe": ROOT / "poseidon/nexus_omega/causal_efe.py",
    }
    return {
        "schema": "poseidon-nexus-phaseworld-v2-viability-assay-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "environment_seeds": list(environment_seeds),
            "agent_seeds": list(agent_seeds),
            "horizon": horizon,
            "version_change": "v2 adds +0.12 hydration to Rest before the unchanged -0.05 basal hydration drain; v1 remains untouched",
            "v2_prior_refinement": "corrects the inherited storm-shelter health gain and selects the higher-LCB action from shelter/rest when the safety guard overrides exploration",
            "v2_shock_update": "unexpected health residual <= -0.25 sets latent storm ticks to at least one; the update is recorded on each transition",
            "observation_contract": "same 8 float32 channels; channel 3 transient storm pulse and channel 4 episode progress",
            "action_contract": "same four actions: forage, shelter, explore, rest",
            "arms": list(ARM_NAMES),
            "primary_task_validity_gate": "three-tick pulse-trace controller survival >= 0.95 on v2",
            "agent_comparison": "same random initialization and same environment seed, v1 structured world prior versus v2 hydration-aligned prior",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "promotion": False,
        },
        "results": {
            "by_arm": summaries,
            "paired_agent_profile_comparisons": paired,
            "trace_memory_minus_reactive_survival_delta": float(trace_v_reactive.mean()),
            "trace_memory_vs_reactive_survival_pairs": {
                "trace_only": int(sum(x > 0 for x in trace_v_reactive)),
                "reactive_only": int(sum(x < 0 for x in trace_v_reactive)),
                "tie": int(sum(x == 0 for x in trace_v_reactive)),
            },
            "viability_positive_control_passed": bool(viability_passed),
            "complete_episode_replay_checks": int(replay_verified),
            "replay_failures": 0,
            "episodes": episodes,
        },
        "limits": [
            "Both environment versions and policies are synthetic; survival is only within this simulator.",
            "The pulse-trace controller is a hand-coded positive control, not a learned policy or proof of memory emergence.",
            "The NEXUS agent comparison uses two fixed random initializations; environment seeds do not substitute for additional trained lineages.",
            "This assay is not a controller promotion gate and does not test the residual adapter.",
            "The v2 environment changes only rest hydration recovery; it is not a general ecology or real-world safety model.",
        ],
        "identity": {
            "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/nexus_omega/phaseworld_v2_viability_assay.json")
    parser.add_argument("--seed-start", type=int, default=SEEDS[0])
    parser.add_argument("--seed-count", type=int, default=len(SEEDS))
    parser.add_argument("--agent-seeds", type=int, nargs=2, default=AGENT_SEEDS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {args.output}")
    if args.seed_count < 2:
        raise ValueError("seed-count must be at least 2 for paired evaluation")
    receipt = run(
        environment_seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
        agent_seeds=tuple(args.agent_seeds),
        horizon=args.horizon,
    )
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "receipt_sha256": receipt["receipt_sha256"],
        "results": {key: value for key, value in receipt["results"].items() if key != "episodes"},
        "arms": receipt["results"]["by_arm"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
