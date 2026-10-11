"""POSEIDON COGNITIVE EXPERIMENT: Rigorous Matched-Seed Lesion Assay & POMDP Benchmark.

Evaluates Supermix Poseidon v2.0 Cognitive Platform across 6 matched arms:
1. intact: Full CognitiveAgent v2.0 with Oracle, 3-tier Mnemosyne, Horizon-X, Neuro.
2. erased: Working memory cleared at each step, episodic store cleared.
3. shifted: Circularly permuted episodic keys (norm-preserving semantic lesion).
4. random_retrieval: Uncorrelated episodic memory recall.
5. memory_free: Ablated memory subsystems (pure reactive baseline).
6. frozen_tidal: Pure un-augmented frozen baseline policy.

Evaluations run across identical paired seeds under standard and POMDP environments.
Produces an auditable cryptographically-hashed receipt.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean, stdev
from typing import Any, Callable

import torch

from .cognitive_agent import CognitiveAgent
from .core import CoreRuntime, active_core_path
from .provenance import assert_source_current
from .world import ACTIONS, N_ACTIONS, OBS_SIZE, TidePool, WORLD_VERSION
from .worldlab import POMDPConfig, POMDPTidePool

COGNITIVE_EXPERIMENT_SCHEMA = "poseidon-cognitive-experiment-v3"
ARMS = ("intact", "erased", "shifted", "random_retrieval", "memory_free", "frozen_tidal")
CONTROLLER_RANDOMIZATION = "python-torch-per-episode-seed-v1"
_RESEARCH_DIRECTORY = Path(__file__).parent / "research"
_RESEARCH_SOURCES = {f"research/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(_RESEARCH_DIRECTORY.glob("*.py"))}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _source_identity() -> dict[str, str]:
    sources = assert_source_current()
    research = {f"research/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(_RESEARCH_DIRECTORY.glob("*.py"))}
    if research != _RESEARCH_SOURCES:
        raise ValueError("Research source changed since import; restart before collecting or replaying evidence")
    return {**sources, **research}


def _file_digest(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _model_identity(core: CoreRuntime) -> dict[str, Any]:
    digest = hashlib.sha256()
    for name, tensor in sorted(core.model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return {"checkpoint_sha256": _file_digest(core.path), "state_dict_sha256": digest.hexdigest(),
            "config": asdict(core.model.config)}


def _load_bound_core(path: str | Path) -> tuple[CoreRuntime, dict[str, Any]]:
    before = _file_digest(path)
    core = CoreRuntime(path)
    identity = _model_identity(core)
    if before != identity["checkpoint_sha256"]:
        raise ValueError("Core checkpoint changed while loading")
    return core, identity


def _oracle_identity(path: str | Path | None) -> dict[str, str] | None:
    return None if path is None else {"path": str(Path(path).resolve()), "sha256": _file_digest(path)}


def _validate_protocol(seed_base: int, episodes: int, max_steps: int, scarcity: float, use_pomdp: bool) -> None:
    if type(seed_base) is not int or not 0 <= seed_base < 2**63:
        raise ValueError("Invalid cognitive seed_base")
    if type(episodes) is not int or not 1 <= episodes <= 256 or seed_base + episodes > 2**63:
        raise ValueError("Cognitive episodes must contain 1 to 256 unique seeds")
    if (type(max_steps) is not int or not 1 <= max_steps <= 10000
            or episodes * max_steps * len(ARMS) > 100000):
        raise ValueError("Invalid or excessive cognitive transition budget")
    if type(scarcity) not in (float, int) or not math.isfinite(scarcity) or not 0.5 <= scarcity <= 4.0:
        raise ValueError("Invalid cognitive scarcity")
    if type(use_pomdp) is not bool:
        raise ValueError("use_pomdp must be boolean")


@contextmanager
def _episode_randomness(seed: int):
    # Existing dream/retrieval code uses Python's generator and neural constructors
    # use Torch's generator. Scope both so paired runs and replay are repeatable.
    python_state = random.getstate()
    with torch.random.fork_rng(devices=[]):
        try:
            random.seed(seed)
            torch.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)


def _environment(seed: int, scarcity: float, max_steps: int, use_pomdp: bool):
    if use_pomdp:
        return POMDPTidePool(POMDPConfig(scarcity=scarcity, seed=seed), max_steps=max_steps)
    return TidePool(scarcity=scarcity, seed=seed, max_steps=max_steps)


def _run_single_seed(
    arm: str,
    seed: int,
    max_steps: int,
    scarcity: float,
    use_pomdp: bool = False,
    core: CoreRuntime | None = None,
    oracle_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("Unsupported cognitive arm")
    with _episode_randomness(seed):
        core = core or CoreRuntime(active_core_path())
        env = _environment(seed, scarcity, max_steps, use_pomdp)
        if arm != "frozen_tidal":
            kwargs = {"oracle_checkpoint": oracle_checkpoint} if oracle_checkpoint is not None else {}
            agent = CognitiveAgent(core=core, mode=arm, **kwargs)
            rollout = agent.rollout(env, max_steps=max_steps, mode=arm)
            return {"arm": arm, "seed": seed, **{key: rollout[key] for key in (
                "steps", "total_reward", "survived", "death", "death_reason", "truncated", "terminal_reason",
                "final_health", "overrides", "override_rate", "initial_snapshot", "trajectory")}}

        obs = env.reset(seed=seed)
        initial_snap = env.snapshot()
        total_reward = 0.0
        steps = 0
        trajectory = []
        for _ in range(max_steps):
            act = core.act(obs)
            if use_pomdp:
                res = env.step(act)
                next_obs = res.observation
                reward = res.reward
                done = res.done
                info = res.info
            else:
                next_obs, reward, done, info = env.step(act)

            total_reward += reward
            steps += 1
            dec = {"action": act, "action_name": ACTIONS[act], "override": False}
            trajectory.append({
                "observation": obs,
                "action": act,
                "action_name": ACTIONS[act],
                "reward": reward,
                "done": bool(done),
                "next_observation": next_obs,
                "state": env.state(),
                "info": info,
                "decision": dec,
            })

            obs = next_obs
            if done:
                break

        return {
            "arm": arm,
            "seed": seed,
            "steps": steps,
            "total_reward": float(total_reward),
            "survived": bool(env.alive),
            "death": not env.alive,
            "death_reason": env.death_reason,
            "truncated": bool(env.alive and env.done),
            "terminal_reason": "death" if not env.alive else "horizon",
            "final_health": float(env.health),
            "overrides": 0,
            "override_rate": 0.0,
            "initial_snapshot": initial_snap,
            "trajectory": trajectory,
        }

def run_cognitive_benchmark(
    core_path: str | Path | None = None,
    seed_base: int = 90000000,
    episodes: int = 4,
    max_steps: int = 32,
    scarcity: float = 2.5,
    use_pomdp: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
    oracle_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Runs complete 6-arm matched lesion and POMDP benchmark."""
    _validate_protocol(seed_base, episodes, max_steps, scarcity, use_pomdp)
    sources = _source_identity()
    core, model_identity = _load_bound_core(core_path or active_core_path())
    oracle_identity = _oracle_identity(oracle_checkpoint)
    seeds = [seed_base + i for i in range(episodes)]

    arm_results: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}

    total_tasks = len(ARMS) * episodes
    completed = 0

    for arm in ARMS:
        for seed in seeds:
            res = _run_single_seed(
                arm=arm,
                seed=seed,
                max_steps=max_steps,
                scarcity=scarcity,
                use_pomdp=use_pomdp,
                core=core,
                oracle_checkpoint=oracle_checkpoint,
            )
            arm_results[arm].append(res)
            completed += 1
            if progress:
                progress({
                    "phase": "running",
                    "arm": arm,
                    "seed": seed,
                    "completed": completed,
                    "total": total_tasks,
                })

    arm_summaries, paired_comparisons = _summarize(arm_results)
    if (_source_identity() != sources or _model_identity(core) != model_identity
            or _oracle_identity(oracle_checkpoint) != oracle_identity):
        raise ValueError("Source or model identity changed during benchmark")
    receipt_body = {
        "schema": COGNITIVE_EXPERIMENT_SCHEMA,
        "world_version": WORLD_VERSION,
        "controller_randomization": CONTROLLER_RANDOMIZATION,
        "model": model_identity,
        "oracle_artifact": oracle_identity,
        "source_sha256": sources,
        "seed_base": seed_base,
        "episodes": episodes,
        "max_steps": max_steps,
        "scarcity": scarcity,
        "use_pomdp": use_pomdp,
        "arms": arm_summaries,
        "paired": paired_comparisons,
        "episodes_data": arm_results,
    }
    receipt_body = json.loads(json.dumps(receipt_body))
    receipt_body["sha256"] = _digest(receipt_body)
    return receipt_body


def _summarize(arm_results: dict[str, list[dict[str, Any]]]) -> tuple[dict, dict]:
    """Recompute all outcome claims from complete seed-aligned episode records."""
    arm_summaries = {}
    for arm, records in arm_results.items():
        rewards = [r["total_reward"] for r in records]
        survivals = [1.0 if r["survived"] else 0.0 for r in records]
        steps_list = [r["steps"] for r in records]
        overrides_list = [r["overrides"] for r in records]

        m_rew = fmean(rewards)
        s_rew = stdev(rewards) if len(rewards) > 1 else 0.0
        surv_rate = fmean(survivals)
        m_steps = fmean(steps_list)

        arm_summaries[arm] = {
            "episodes": len(records),
            "survival_rate": round(surv_rate, 4),
            "mean_reward": round(m_rew, 4),
            "std_reward": round(s_rew, 4),
            "mean_steps": round(m_steps, 2),
            "total_overrides": sum(overrides_list),
            "mean_override_rate": round(fmean([r["override_rate"] for r in records]), 4),
        }

    # Paired comparisons vs intact
    intact_rewards = [r["total_reward"] for r in arm_results["intact"]]
    paired_comparisons = {}
    for arm in ARMS:
        if arm == "intact":
            continue
        deltas = [
            intact_r - other_r
            for intact_r, other_r in zip(intact_rewards, [r["total_reward"] for r in arm_results[arm]])
        ]
        mean_delta = fmean(deltas)
        wins = sum(d > 1e-6 for d in deltas)
        ties = sum(abs(d) <= 1e-6 for d in deltas)
        losses = sum(d < -1e-6 for d in deltas)

        paired_comparisons[f"intact_vs_{arm}"] = {
            "mean_reward_advantage": round(mean_delta, 4),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "deltas": [round(d, 4) for d in deltas],
        }

    return arm_summaries, paired_comparisons


def verify_cognitive_receipt(receipt: dict[str, Any], core_path: str | Path | None = None) -> bool:
    """Verifies cryptographic hash, provenance, schema, metrics, and exact physical replay of an experiment receipt."""
    if not isinstance(receipt, dict):
        raise ValueError("Receipt must be a dictionary")

    receipt_keys = set(receipt.keys())
    expected_keys = {
        "schema",
        "world_version",
        "controller_randomization",
        "model",
        "oracle_artifact",
        "source_sha256",
        "seed_base",
        "episodes",
        "max_steps",
        "scarcity",
        "use_pomdp",
        "arms",
        "paired",
        "episodes_data",
        "sha256",
    }
    if receipt_keys != expected_keys:
        raise ValueError(
            f"Receipt keys mismatch: missing {expected_keys - receipt_keys}, unexpected {receipt_keys - expected_keys}"
        )

    # 1. Verify cryptographic hash
    claimed_sha = receipt["sha256"]
    body = {k: v for k, v in receipt.items() if k != "sha256"}
    canonical_body = json.loads(json.dumps(body))
    computed_sha = hashlib.sha256(json.dumps(canonical_body, sort_keys=True).encode("utf-8")).hexdigest()
    if claimed_sha != computed_sha:
        raise ValueError(f"Receipt hash mismatch: claimed {claimed_sha}, calculated {computed_sha}")

    # 2. Verify schema and environment constants
    if receipt["schema"] != COGNITIVE_EXPERIMENT_SCHEMA:
        raise ValueError(f"Unsupported schema {receipt['schema']}, expected {COGNITIVE_EXPERIMENT_SCHEMA}")
    if receipt["world_version"] != WORLD_VERSION:
        raise ValueError(f"World version mismatch: {receipt['world_version']} != {WORLD_VERSION}")
    if receipt["controller_randomization"] != CONTROLLER_RANDOMIZATION:
        raise ValueError(
            f"Controller randomization mismatch: {receipt['controller_randomization']} != {CONTROLLER_RANDOMIZATION}"
        )

    # 3. Verify protocol parameters
    _validate_protocol(
        receipt["seed_base"],
        receipt["episodes"],
        receipt["max_steps"],
        receipt["scarcity"],
        receipt["use_pomdp"],
    )

    # 4. Verify source code provenance
    current_sources = _source_identity()
    if receipt["source_sha256"] != current_sources:
        raise ValueError("Source file hash mismatch with active environment")

    # 5. Verify core model identity
    core, model_identity = _load_bound_core(core_path or active_core_path())
    if receipt["model"] != model_identity:
        raise ValueError("Core model state or configuration mismatch")

    # 6. Verify oracle artifact if present
    if receipt["oracle_artifact"] is not None:
        oracle_info = receipt["oracle_artifact"]
        if not isinstance(oracle_info, dict) or "path" not in oracle_info or "sha256" not in oracle_info:
            raise ValueError("Malformed oracle artifact specification")
        if _oracle_identity(oracle_info["path"]) != oracle_info:
            raise ValueError("Oracle artifact hash mismatch")

    # 7. Verify episodes_data structure and seed alignments
    episodes_data = receipt["episodes_data"]
    if not isinstance(episodes_data, dict) or set(episodes_data.keys()) != set(ARMS):
        raise ValueError(f"episodes_data must contain all arms: {ARMS}")

    seed_base = receipt["seed_base"]
    episodes_count = receipt["episodes"]
    max_steps = receipt["max_steps"]
    scarcity = receipt["scarcity"]
    use_pomdp = receipt["use_pomdp"]
    expected_seeds = [seed_base + i for i in range(episodes_count)]

    for arm in ARMS:
        records = episodes_data[arm]
        if not isinstance(records, list) or len(records) != episodes_count:
            raise ValueError(f"Arm '{arm}' must have exactly {episodes_count} episode records")
        seeds = [r.get("seed") for r in records if isinstance(r, dict)]
        if seeds != expected_seeds:
            raise ValueError(f"Arm '{arm}' seeds {seeds} do not match expected sequence {expected_seeds}")

    # 8. Verify statistical summaries and paired comparisons
    expected_summaries, expected_paired = _summarize(episodes_data)
    if receipt["arms"] != expected_summaries:
        raise ValueError("Summary arm metrics disagree with episode records")
    if receipt["paired"] != expected_paired:
        raise ValueError("Paired comparison statistics disagree with episode records")

    # 9. Verify step-by-step physical replay of trajectories
    for arm in ARMS:
        for ep_idx, ep in enumerate(episodes_data[arm]):
            if not isinstance(ep, dict):
                raise ValueError(f"Episode {ep_idx} in arm '{arm}' is not a dict")

            if ep.get("arm") != arm:
                raise ValueError(f"Episode arm tag '{ep.get('arm')}' does not match arm '{arm}'")
            if ep.get("death") != (not ep.get("survived")):
                raise ValueError("Death and survived flags are logically inconsistent")

            trajectory = ep.get("trajectory")
            if not isinstance(trajectory, list):
                raise ValueError(f"Episode trajectory missing or not a list in arm '{arm}'")
            if len(trajectory) != ep.get("steps"):
                raise ValueError(f"Trajectory length {len(trajectory)} does not match recorded steps {ep.get('steps')}")

            init_snap = ep.get("initial_snapshot")
            if not isinstance(init_snap, dict):
                raise ValueError("Initial snapshot missing or not a dict")

            # Restore environment from initial snapshot
            try:
                if use_pomdp:
                    env = POMDPTidePool.from_snapshot(init_snap)
                else:
                    env = TidePool.from_snapshot(init_snap)
            except Exception as e:
                raise ValueError(f"Failed to restore environment snapshot: {e}") from e

            total_reward = 0.0
            for step_idx, step_record in enumerate(trajectory):
                action = step_record.get("action")
                if type(action) is not int or not 0 <= action < N_ACTIONS:
                    raise ValueError(f"Invalid action {action} at step {step_idx}")

                # Check decision field if present
                decision = step_record.get("decision")
                if isinstance(decision, dict) and decision.get("action") != action:
                    raise ValueError(f"Decision action {decision.get('action')} does not match executed action {action}")

                # Step the restored environment
                if use_pomdp:
                    step_res = env.step(action)
                    real_obs = step_res.observation
                    real_reward = step_res.reward
                    real_done = step_res.done
                else:
                    real_obs, real_reward, real_done, _ = env.step(action)

                # Check reward
                if abs(real_reward - step_record.get("reward", 0.0)) > 1e-6:
                    raise ValueError(
                        f"Replayed reward {real_reward} != claimed reward {step_record.get('reward')} at step {step_idx}"
                    )

                # Check done flag
                if bool(real_done) != bool(step_record.get("done")):
                    raise ValueError(f"Replayed done flag {real_done} != claimed {step_record.get('done')} at step {step_idx}")

                # Check next observation
                claimed_next_obs = step_record.get("next_observation")
                if not isinstance(claimed_next_obs, list) or len(claimed_next_obs) != len(real_obs):
                    raise ValueError(f"Observation dimension mismatch at step {step_idx}")
                if any(abs(a - b) > 1e-5 for a, b in zip(real_obs, claimed_next_obs)):
                    raise ValueError(f"Replayed next observation does not match claimed observation at step {step_idx}")

                # Check state consistency
                real_state = env.state()
                claimed_state = step_record.get("state")
                if isinstance(claimed_state, dict):
                    if real_state.get("x") != claimed_state.get("x") or real_state.get("y") != claimed_state.get("y"):
                        raise ValueError(f"Agent position mismatch at step {step_idx}")
                    if abs(real_state.get("health", 0.0) - claimed_state.get("health", 0.0)) > 1e-6:
                        raise ValueError(f"Health mismatch at step {step_idx}")

                total_reward += real_reward
                if real_done:
                    break

            # Verify terminal state
            is_alive = env.alive
            if bool(is_alive) != bool(ep.get("survived")):
                raise ValueError(f"Replayed survival {is_alive} != claimed {ep.get('survived')}")

            if abs(total_reward - ep.get("total_reward", 0.0)) > 1e-5:
                raise ValueError(f"Replayed total reward {total_reward} != claimed {ep.get('total_reward')}")

            final_h = env.health
            if abs(final_h - ep.get("final_health", 0.0)) > 1e-5:
                raise ValueError(f"Replayed final health {final_h} != claimed {ep.get('final_health')}")

    return True
