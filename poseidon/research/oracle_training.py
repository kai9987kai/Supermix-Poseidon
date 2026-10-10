"""Train and calibrate ORACLE without changing Poseidon's active policy.

Each model member bootstraps complete training episodes. Checkpoint selection
uses disjoint episodes, uncertainty intervals use a second disjoint calibration
split, and final ID/OOD behavior belongs to candidate_evaluation.py.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from ..world import N_ACTIONS, TidePool, WORLD_VERSION, event_uniform, teacher_action
from .contracts import (Episode, EpisodeSplit, TransitionRecord, assert_disjoint_splits,
                        canonical_bytes, split_summary, tidepool_contract)
from .oracle_model import ORACLE_SCHEMA, OracleConfig, OracleEnsemble

FIT_RECEIPT_SCHEMA = "poseidon-oracle-fit-receipt-v1"
TRAIN_SCARCITY = (0.8, 1.0, 1.25, 1.5)
EXPLORATION_RATE = 0.18


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    paths = {
        "world.py": package / "world.py",
        "contracts.py": package / "research" / "contracts.py",
        "oracle_model.py": package / "research" / "oracle_model.py",
        "oracle_training.py": package / "research" / "oracle_training.py",
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def collect_episode(seed: int, scarcity: float, max_steps: int = 96,
                    exploration_rate: float = EXPLORATION_RATE) -> Episode:
    """Collect one complete, observation-only teacher-plus-exploration episode."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if isinstance(scarcity, bool) or not math.isfinite(scarcity) or not 0.5 <= scarcity <= 4:
        raise ValueError("scarcity must be finite and in [0.5, 4]")
    if type(max_steps) is not int or not 1 <= max_steps <= 10000:
        raise ValueError("max_steps must be an integer in [1, 10000]")
    if isinstance(exploration_rate, bool) or not math.isfinite(exploration_rate) or not 0 <= exploration_rate <= 1:
        raise ValueError("exploration_rate must be finite and in [0, 1]")
    env = TidePool(seed=seed, scarcity=float(scarcity), max_steps=max_steps)
    observation = env.reset(seed)
    rows = []
    done = False
    while not done:
        if event_uniform(seed, env.tick, "oracle-data-policy", "explore") < exploration_rate:
            draw = event_uniform(seed, env.tick, "oracle-data-policy", "action")
            action = min(int(draw * N_ACTIONS), N_ACTIONS - 1)
        else:
            action = teacher_action(observation)
        next_observation, reward, done, info = env.step(action)
        row = TransitionRecord(
            episode_seed=seed,
            tick=env.tick - 1,
            scarcity=float(scarcity),
            observation=tuple(float(value) for value in observation),
            action=int(action),
            next_observation=tuple(float(value) for value in next_observation),
            reward=float(reward),
            death=bool(info["death"]),
            truncated=bool(info["truncated"]),
        )
        row.validate()
        rows.append(row)
        observation = next_observation
    episode = Episode(seed=seed, scarcity=float(scarcity), transitions=tuple(rows))
    episode.validate()
    return episode


def collect_split(name: str, seeds: Sequence[int], *, max_steps: int,
                  scarcity_values: Sequence[float] = TRAIN_SCARCITY,
                  fixed_scarcity: float | None = None,
                  exploration_rate: float = EXPLORATION_RATE) -> EpisodeSplit:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"{name} seeds must be nonempty and unique")
    if not scarcity_values:
        raise ValueError("scarcity_values cannot be empty")
    episodes = []
    for index, seed in enumerate(seeds):
        scarcity = float(fixed_scarcity if fixed_scarcity is not None else scarcity_values[index % len(scarcity_values)])
        episodes.append(collect_episode(seed, scarcity, max_steps, exploration_rate))
    split = EpisodeSplit(name, tuple(episodes))
    split.validate()
    return split


def _flatten(split: EpisodeSplit) -> list[TransitionRecord]:
    return [row for episode in split.episodes for row in episode.transitions]


def _batch(rows: Sequence[TransitionRecord]) -> tuple[torch.Tensor, ...]:
    obs = torch.tensor([row.observation for row in rows], dtype=torch.float32)
    actions = torch.tensor([row.action for row in rows], dtype=torch.long)
    deltas = torch.tensor([row.next_observation for row in rows], dtype=torch.float32) - obs
    rewards = torch.tensor([row.reward for row in rows], dtype=torch.float32)
    deaths = torch.tensor([row.death for row in rows], dtype=torch.float32)
    return obs, actions, deltas, rewards, deaths


def _loss(output: dict[str, torch.Tensor], deltas: torch.Tensor, rewards: torch.Tensor,
          deaths: torch.Tensor, positive_weight: float) -> torch.Tensor:
    delta_logv = output["delta_log_variance"]
    delta_nll = 0.5 * (delta_logv + (deltas - output["delta_mean"]).square() * torch.exp(-delta_logv))
    reward_logv = output["reward_log_variance"]
    reward_nll = 0.5 * (reward_logv + (rewards - output["reward_mean"]).square() * torch.exp(-reward_logv))
    death_loss = F.binary_cross_entropy_with_logits(
        output["death_logit"], deaths, pos_weight=torch.tensor(positive_weight, dtype=torch.float32))
    return delta_nll.mean() + 0.35 * reward_nll.mean() + 0.20 * death_loss


@torch.inference_mode()
def _selection_loss(model: OracleEnsemble, split: EpisodeSplit, positive_weight: float) -> float:
    rows = _flatten(split)
    obs, actions, deltas, rewards, deaths = _batch(rows)
    prediction = model(obs, actions)
    losses = [_loss({name: prediction[name][index] for name in prediction}, deltas, rewards, deaths, positive_weight)
              for index in range(model.config.ensemble_size)]
    result = float(torch.stack(losses).mean())
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite ORACLE selection loss")
    return result


@torch.inference_mode()
def episode_calibration_scores(model: OracleEnsemble, episode: Episode) -> tuple[float, float]:
    rows = episode.transitions
    obs, actions, deltas, rewards, _ = _batch(rows)
    summary = model.summarize(model(obs, actions))
    delta_scale = summary["delta_total_variance"].sqrt().clamp_min(1e-4)
    reward_scale = summary["reward_total_variance"].sqrt().clamp_min(1e-4)
    state_score = ((deltas - summary["delta_mean"]).abs() / delta_scale).max()
    reward_score = ((rewards - summary["reward_mean"]).abs() / reward_scale).max()
    result = float(state_score), float(reward_score)
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise FloatingPointError("invalid episode calibration score")
    return result


def conformal_order_statistic(scores: Sequence[float], coverage: float) -> float:
    if not scores or any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("calibration scores must be finite, nonnegative, and nonempty")
    if isinstance(coverage, bool) or not math.isfinite(coverage) or not 0.5 <= coverage < 1:
        raise ValueError("coverage must be in [0.5, 1)")
    rank = math.ceil((len(scores) + 1) * coverage)
    if rank > len(scores):
        raise ValueError("calibration split is too small for the requested finite-sample coverage")
    return float(sorted(scores)[rank - 1])


def _state_to_cpu(model: OracleEnsemble) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _derive_seeds(base_seed: int, train_episodes: int, selection_episodes: int,
                  calibration_episodes: int) -> dict[str, list[int]]:
    if type(base_seed) is not int or not 0 <= base_seed <= 2**63 - 3_000_000:
        raise ValueError("base_seed must leave room for four disjoint million-seed namespaces")
    result = {
        "train": list(range(base_seed, base_seed + train_episodes)),
        "selection": list(range(base_seed + 1_000_000, base_seed + 1_000_000 + selection_episodes)),
        "calibration": list(range(base_seed + 2_000_000, base_seed + 2_000_000 + calibration_episodes)),
    }
    if any(seed >= 2**63 for values in result.values() for seed in values):
        raise ValueError("derived episode seed exceeds the contract")
    if len({seed for values in result.values() for seed in values}) != sum(map(len, result.values())):
        raise ValueError("derived episode splits overlap")
    return result


def fit_oracle(
    output_dir: str | Path,
    *,
    base_seed: int = 442_000_000,
    train_episodes: int = 32,
    selection_episodes: int = 8,
    calibration_episodes: int = 12,
    max_steps: int = 96,
    training_updates: int = 256,
    batch_size: int = 128,
    hidden_size: int = 128,
    ensemble_size: int = 5,
    coverage: float = 0.90,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit an independently initialized, episode-bootstrap predictive ensemble.

    Only train episodes optimize weights. Selection episodes choose the saved
    update, and calibration episodes set episode-level split-conformal widths.
    No test episodes are collected or accessed here.
    """
    for name, value, low, high in (
        ("train_episodes", train_episodes, 2, 10000),
        ("selection_episodes", selection_episodes, 2, 1000),
        ("calibration_episodes", calibration_episodes, 2, 1000),
        ("max_steps", max_steps, 2, 10000),
        ("training_updates", training_updates, 1, 100000),
        ("batch_size", batch_size, 8, 8192),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    config = OracleConfig(hidden_size=hidden_size, ensemble_size=ensemble_size)
    config.validate()
    if type(base_seed) is not int or not 0 <= base_seed <= 2**63 - 3_000_000:
        raise ValueError("base_seed must leave room for four disjoint million-seed namespaces")
    if isinstance(coverage, bool) or not math.isfinite(coverage) or not 0.5 <= coverage < 1:
        raise ValueError("coverage must be in [0.5, 1)")
    minimum_calibration_episodes = math.ceil(coverage / (1 - coverage) - 1e-12)
    if calibration_episodes < minimum_calibration_episodes:
        raise ValueError(f"coverage {coverage} requires at least {minimum_calibration_episodes} calibration episodes")
    seed_groups = _derive_seeds(base_seed, train_episodes, selection_episodes, calibration_episodes)
    out_path = Path(output_dir)
    if out_path.exists() and any(out_path.iterdir()):
        raise FileExistsError("ORACLE output directory must be empty; existing candidates are preserved")
    contract = tidepool_contract()
    if progress:
        progress({"phase": "collect", "split": "train", "completed": 0, "total": train_episodes})
    splits = {
        "train": collect_split("train", seed_groups["train"], max_steps=max_steps),
        "selection": collect_split("selection", seed_groups["selection"], max_steps=max_steps),
        "calibration": collect_split("calibration", seed_groups["calibration"], max_steps=max_steps),
    }
    split_manifest = assert_disjoint_splits(splits)
    train_rows = _flatten(splits["train"])
    selection_rows = _flatten(splits["selection"])
    calibration_rows = _flatten(splits["calibration"])
    positive_count = sum(row.death for row in train_rows)
    negative_count = len(train_rows) - positive_count
    positive_weight = min(30.0, negative_count / max(1, positive_count)) if positive_count else 1.0

    model = OracleEnsemble(config, seed=base_seed + 71_003)
    optimizers = [torch.optim.AdamW(member.parameters(), lr=3e-4, weight_decay=1e-5)
                  for member in model.members]
    training_groups = {episode.seed: list(episode.transitions) for episode in splits["train"].episodes}
    bootstrap_pools: list[list[TransitionRecord]] = []
    bootstrap_draws: list[list[int]] = []
    member_rngs: list[random.Random] = []
    for member_index in range(ensemble_size):
        rng = random.Random(base_seed + 83_117 + member_index * 1_000_003)
        sampled_episode_seeds = [rng.choice(seed_groups["train"]) for _ in seed_groups["train"]]
        bootstrap_draws.append(sampled_episode_seeds)
        bootstrap_pools.append([row for seed in sampled_episode_seeds for row in training_groups[seed]])
        member_rngs.append(rng)

    best_score = math.inf
    best_update = 0
    best_state: dict[str, torch.Tensor] | None = None
    selection_history = []
    eval_interval = max(1, min(32, training_updates // 8 or 1))
    model.train()
    for update in range(1, training_updates + 1):
        member_losses = []
        for index, member in enumerate(model.members):
            pool = bootstrap_pools[index]
            rng = member_rngs[index]
            rows = [pool[rng.randrange(len(pool))] for _ in range(batch_size)]
            obs, actions, deltas, rewards, deaths = _batch(rows)
            prediction = member(obs, actions)
            loss = _loss(prediction, deltas, rewards, deaths, positive_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite ORACLE training loss; no candidate was written")
            optimizers[index].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(member.parameters(), max_norm=2.0, error_if_nonfinite=True)
            optimizers[index].step()
            member_losses.append(float(loss.detach()))
        if update % eval_interval == 0 or update == training_updates:
            model.eval()
            score = _selection_loss(model, splits["selection"], positive_weight)
            selection_history.append({"update": update, "selection_nll": score,
                                     "mean_bootstrap_train_loss": sum(member_losses) / len(member_losses)})
            if score < best_score:
                best_score = score
                best_update = update
                best_state = _state_to_cpu(model)
            if progress:
                progress({"phase": "train", "completed": update, "total": training_updates,
                          "selection_nll": score, "best_update": best_update})
            model.train()
    if best_state is None:
        raise RuntimeError("ORACLE selection did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    state_scores, reward_scores = [], []
    calibration_episode_rows = []
    for episode in splits["calibration"].episodes:
        state_score, reward_score = episode_calibration_scores(model, episode)
        state_scores.append(state_score)
        reward_scores.append(reward_score)
        calibration_episode_rows.append({"seed": episode.seed, "state_max_score": state_score,
                                         "reward_max_score": reward_score})
    state_quantile = conformal_order_statistic(state_scores, coverage)
    reward_quantile = conformal_order_statistic(reward_scores, coverage)

    reward_sums = [0.0] * config.action_count
    reward_counts = [0] * config.action_count
    death_count_by_action = [0] * config.action_count
    for row in train_rows:
        reward_sums[row.action] += row.reward
        reward_counts[row.action] += 1
        death_count_by_action[row.action] += int(row.death)
    global_reward_mean = sum(row.reward for row in train_rows) / len(train_rows)
    reward_mean_by_action = [reward_sums[action] / reward_counts[action] if reward_counts[action] else global_reward_mean
                             for action in range(config.action_count)]
    death_prior = (positive_count + 1) / (len(train_rows) + 2)
    summaries = {name: split_summary(split) for name, split in splits.items()}
    source_digest = source_hashes()
    training_manifest = {
        "base_seed": base_seed,
        "seed_sets": split_manifest,
        "split_summaries": summaries,
        "max_steps": max_steps,
        "train_scarcity_cycle": list(TRAIN_SCARCITY),
        "exploration_rate": EXPLORATION_RATE,
        "data_policy": "teacher_action with deterministic event-keyed uniform-action exploration",
        "target_semantics": {"state": "next_observation minus observation",
                             "reward": "TidePool realized one-step reward",
                             "death": "death on this transition; horizon truncation excluded"},
        "source_sha256": source_digest,
        "environment_contract": {"version": contract.version, "world_version": contract.world_version,
                                  "observation_names": list(contract.observation_names),
                                  "actions": list(contract.actions)},
    }
    model_sha = model_state_sha256(best_state)
    artifact_payload = {
        "schema": ORACLE_SCHEMA,
        "config": config.to_dict(),
        "model_seed": base_seed + 71_003,
        "model_state_dict": best_state,
        "model_sha256": model_sha,
        "training_manifest": training_manifest,
        "training": {"updates_requested": training_updates, "selected_update": best_update,
                     "selection_nll": best_score, "selection_history": selection_history,
                     "training_positive_death_labels": positive_count,
                     "death_positive_weight": positive_weight,
                     "bootstrap_episode_draws": bootstrap_draws,
                     "reward_mean_by_action": reward_mean_by_action,
                     "death_probability_prior": death_prior,
                     "death_count_by_action": death_count_by_action},
        "calibration": {"method": "episode_max_split_conformal_v1",
                        "coverage": coverage,
                        "exchangeability_unit": "complete episode seed",
                        "calibration_episode_count": calibration_episodes,
                        "state_standardized_error_quantile": state_quantile,
                        "reward_standardized_error_quantile": reward_quantile,
                        "state_episode_scores": state_scores,
                        "reward_episode_scores": reward_scores,
                        "episode_scores": calibration_episode_rows,
                        "minimum_calibration_episodes": minimum_calibration_episodes,
                        "scope": "ID calibration distribution only; no OOD coverage guarantee"},
        "activated": False,
    }
    out_path.mkdir(parents=True, exist_ok=True)
    artifact_path = out_path / "oracle.pt"
    artifact_tmp = out_path / "oracle.pt.tmp"
    try:
        torch.save(artifact_payload, artifact_tmp)
        os.replace(artifact_tmp, artifact_path)
    finally:
        if artifact_tmp.exists():
            artifact_tmp.unlink()
    receipt = {
        "schema": FIT_RECEIPT_SCHEMA,
        "status": "trained_and_calibrated_candidate",
        "activated": False,
        "world_version": WORLD_VERSION,
        "model_sha256": model_sha,
        "artifact_sha256": _file_sha256(artifact_path),
        "artifact_path": str(artifact_path.resolve()),
        "config": config.to_dict(),
        "training_manifest": training_manifest,
        "training": artifact_payload["training"],
        "calibration": artifact_payload["calibration"],
        "limits": [
            "This artifact predicts TidePool transitions; it does not control or improve the active policy.",
            "Calibration is episode-level and applies to the registered ID distribution, not shifted scarcity.",
            "The synthetic TidePool benchmark is not evidence of biological, physical-world, or Unity transfer.",
            "Death scores are reported with Brier metrics later and are not probability-calibrated by this fit.",
        ],
    }
    receipt["receipt_sha256"] = _sha256_bytes(canonical_bytes(receipt))
    _atomic_json(out_path / "fit_receipt.json", receipt)
    if progress:
        progress({"phase": "completed", "completed": 1, "total": 1,
                  "artifact_sha256": receipt["artifact_sha256"], "model_sha256": model_sha})
    return receipt


def load_oracle_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.is_file() or artifact_path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("ORACLE artifact is missing or exceeds the 512 MiB limit")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    required = {"schema", "config", "model_seed", "model_state_dict", "model_sha256",
                "training_manifest", "training", "calibration", "activated"}
    if not isinstance(payload, dict) or set(payload) != required or payload["schema"] != ORACLE_SCHEMA:
        raise ValueError("unsupported ORACLE artifact schema")
    if payload["activated"] is not False:
        raise ValueError("ORACLE candidate must remain opt-in")
    config_data = payload["config"]
    if not isinstance(config_data, dict) or set(config_data) != {"hidden_size", "ensemble_size", "observation_size", "action_count"}:
        raise ValueError("invalid ORACLE config")
    config = OracleConfig(**config_data)
    config.validate()
    if model_state_sha256(payload["model_state_dict"]) != payload["model_sha256"]:
        raise ValueError("ORACLE model state hash mismatch")
    return payload


def _load_model(payload: dict[str, Any]) -> OracleEnsemble:
    config = OracleConfig(**payload["config"])
    model = OracleEnsemble(config, seed=int(payload["model_seed"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def verify_fit_receipt(receipt_path: str | Path, *, replay_data: bool = True) -> dict[str, Any]:
    path = Path(receipt_path)
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("ORACLE fit receipt is missing or too large")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != FIT_RECEIPT_SCHEMA:
        raise ValueError("unsupported ORACLE fit receipt")
    expected_receipt_hash = receipt.pop("receipt_sha256", None)
    if not isinstance(expected_receipt_hash, str) or _sha256_bytes(canonical_bytes(receipt)) != expected_receipt_hash:
        raise ValueError("ORACLE fit receipt digest mismatch")
    artifact_path = Path(receipt.get("artifact_path", ""))
    if not artifact_path.is_absolute():
        artifact_path = path.parent / artifact_path
    if not artifact_path.is_file() or _file_sha256(artifact_path) != receipt.get("artifact_sha256"):
        raise ValueError("ORACLE artifact file hash mismatch")
    payload = load_oracle_artifact(artifact_path)
    if payload["model_sha256"] != receipt.get("model_sha256"):
        raise ValueError("ORACLE model identity mismatch")
    if replay_data:
        protocol = receipt["training_manifest"]
        if source_hashes() != protocol["source_sha256"]:
            raise ValueError("ORACLE source fingerprints changed since fitting")
        seed_sets = protocol["seed_sets"]
        max_steps = int(protocol["max_steps"])
        replayed = {
            name: collect_split(name, seeds, max_steps=max_steps)
            for name, seeds in seed_sets.items()
        }
        assert_disjoint_splits(replayed)
        for name, split in replayed.items():
            actual = split_summary(split)
            expected = protocol["split_summaries"].get(name)
            if actual != expected:
                raise ValueError(f"ORACLE {name} transitions do not replay from the recorded seeds")
        model = _load_model(payload)
        replayed_calibration = [episode_calibration_scores(model, episode) for episode in replayed["calibration"].episodes]
        state_scores = [score[0] for score in replayed_calibration]
        reward_scores = [score[1] for score in replayed_calibration]
        calibration = payload["calibration"]
        if state_scores != calibration["state_episode_scores"] or reward_scores != calibration["reward_episode_scores"]:
            raise ValueError("ORACLE calibration scores do not replay")
        if conformal_order_statistic(state_scores, calibration["coverage"]) != calibration["state_standardized_error_quantile"]:
            raise ValueError("ORACLE state conformal quantile does not replay")
        if conformal_order_statistic(reward_scores, calibration["coverage"]) != calibration["reward_standardized_error_quantile"]:
            raise ValueError("ORACLE reward conformal quantile does not replay")
    receipt["receipt_sha256"] = expected_receipt_hash
    return {"valid": True, "replayed_data": replay_data, "artifact_sha256": receipt["artifact_sha256"],
            "model_sha256": receipt["model_sha256"], "episode_splits": list(receipt["training_manifest"]["seed_sets"])}
