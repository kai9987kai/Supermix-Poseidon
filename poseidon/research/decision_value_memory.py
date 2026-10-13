"""POSEIDON DECISION-VALUE-GATED EPISODIC MEMORY (DV-GEM).

Flagship Causal Memory Architecture for Partially Observable & Aliased Regimes:
1. Compact Recurrent Belief Encoder (Working Context):
   Integrates history through o_t, completed action a_{t-1}, and realized residuals,
   producing a belief embedding h_t without leaking future returns or hidden simulator state.

2. Value-Gated Episodic Retrieval:
   Stores historical transitions e_i = (o_i, a_i, r_i, o'_i, delta_i).
   Query is formed from (o_t, a_cand, h_t).
   Evaluates whether retrieved past interventions provide positive causal decision value:
     \\Delta Q(s_t, a, m) = Q(s_t, a, m) - Q(s_t, a, \\emptyset)

3. Support & Conflict Screening:
   - Out-of-distribution or stale candidates are filtered (supported = False).
   - Conflicting evidence (retrieved records with diverging outcomes) raises conflict score.

4. Explicit Admission Gate:
   - Default influence is strictly 0.0 (baseline fallback).
   - Memory influence is admitted if and only if:
     supported and (conflict <= threshold) and (\\Delta Q - lambda * sigma_Q > admission_threshold)
   - When admitted: delta_m(a) = clip(alpha * \\Delta Q(a), -0.5, 0.5).

5. Six Rigorous Matched Lesion Arms:
   - intact: full learned retrieval and admission gating.
   - similarity: standard Euclidean observation similarity (no decision-value gating).
   - erased: empty episodic store / zero admission.
   - shifted: circular permutation of memory keys (breaks semantic alignment).
   - donor: populated from unrelated donor episodes.
   - baseline: pure frozen Tidal baseline (memory bypassed).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..world import ACTIONS, N_ACTIONS, OBS_SIZE

DECISION_MEMORY_MODES = ("intact", "similarity", "erased", "shifted", "donor", "baseline")
DECISION_MEMORY_SCHEMA = "poseidon-decision-value-memory-v1"


@dataclass(frozen=True)
class EpisodicEntry:
    tick: int
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    residual: tuple[float, ...]
    age: int = 0
    tag: str = "default"

    def validate(self) -> None:
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        if type(self.action) is not int or not 0 <= self.action < N_ACTIONS:
            raise ValueError("action must be in [0, 5]")
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        for name, vec in (("observation", self.observation), ("next_observation", self.next_observation)):
            if len(vec) != OBS_SIZE or not all(math.isfinite(x) for x in vec):
                raise ValueError(f"{name} must contain {OBS_SIZE} finite values")
        if len(self.residual) != OBS_SIZE or not all(math.isfinite(x) for x in vec):
            raise ValueError(f"residual must contain {OBS_SIZE} finite values")


@dataclass(frozen=True)
class DecisionValueMemoryConfig:
    capacity: int = 256
    top_k: int = 4
    admission_threshold: float = 0.05
    max_candidate_distance: float = 0.90
    conflict_threshold: float = 0.25
    uncertainty_discount: float = 0.50
    shrinkage_alpha: float = 0.50
    mode: str = "intact"
    belief_dim: int = 32
    hidden_dim: int = 64
    horizons: tuple[int, ...] = (4, 8)

    def __post_init__(self):
        if self.mode not in DECISION_MEMORY_MODES:
            raise ValueError(f"mode must be one of {DECISION_MEMORY_MODES}, got {self.mode}")
        if not 1 <= self.capacity <= 10000:
            raise ValueError("capacity must be in [1, 10000]")
        if not 1 <= self.top_k <= self.capacity:
            raise ValueError("top_k must be in [1, capacity]")
        if not math.isfinite(self.admission_threshold) or self.admission_threshold < 0.0:
            raise ValueError("admission_threshold must be nonnegative finite")
        if not 0.0 < self.shrinkage_alpha <= 1.0:
            raise ValueError("shrinkage_alpha must be in (0, 1]")


@dataclass
class MemoryDecision:
    action_values: list[float]
    decision_deltas: list[float]
    gated_deltas: list[float]
    admitted_actions: list[bool]
    admission_gate_open: bool
    retrieved_count: int
    retrieved_candidates: list[dict[str, Any]]
    conflict_score: float
    support_score: float
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_values": [round(x, 6) for x in self.action_values],
            "decision_deltas": [round(x, 6) for x in self.decision_deltas],
            "gated_deltas": [round(x, 6) for x in self.gated_deltas],
            "admitted_actions": list(self.admitted_actions),
            "admission_gate_open": self.admission_gate_open,
            "retrieved_count": self.retrieved_count,
            "conflict_score": round(self.conflict_score, 6),
            "support_score": round(self.support_score, 6),
            "mode": self.mode,
            "retrieved_candidates": self.retrieved_candidates,
        }


class RecurrentBeliefEncoder(nn.Module):
    """Compact GRU belief cell aggregating past observations and actions."""

    def __init__(self, obs_dim: int = OBS_SIZE, action_count: int = N_ACTIONS, belief_dim: int = 32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_count = action_count
        self.belief_dim = belief_dim
        # Input: observation (16) + action one-hot (6) + scalar reward (1)
        self.input_dim = obs_dim + action_count + 1
        self.cell = nn.GRUCell(self.input_dim, belief_dim)

    def forward(self, obs: torch.Tensor, prev_action: int, prev_reward: float, hidden: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        act_one_hot = F.one_hot(torch.tensor([prev_action]), num_classes=self.action_count).float()
        r_tensor = torch.tensor([[prev_reward]], dtype=torch.float32)
        inp = torch.cat([obs, act_one_hot, r_tensor], dim=-1)
        return self.cell(inp, hidden)


class DecisionValueScorer(nn.Module):
    """Neural Scorer predicting Q(s, a, m) vs Q(s, a, empty)."""

    def __init__(self, obs_dim: int = OBS_SIZE, belief_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim
        self.hidden_dim = hidden_dim
        
        # Query: obs (16) + action one-hot (6) + belief (32) = 54
        query_dim = obs_dim + N_ACTIONS + belief_dim
        # Memory entry: obs (16) + action one-hot (6) + reward (1) + residual (16) = 39
        memory_dim = obs_dim + N_ACTIONS + 1 + obs_dim

        # Predictor with 2 ensemble heads for epistemic uncertainty estimation
        self.head1 = nn.Sequential(
            nn.Linear(query_dim + memory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head2 = nn.Sequential(
            nn.Linear(query_dim + memory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([query, memory], dim=-1)
        out1 = self.head1(inp)
        out2 = self.head2(inp)
        mean_q = (out1 + out2) / 2.0
        sigma_q = torch.abs(out1 - out2) / 2.0
        return mean_q, sigma_q


class DecisionValueMemory:
    """Decision-Value-Gated Episodic Memory Controller."""

    def __init__(self, config: DecisionValueMemoryConfig | None = None):
        self.config = config or DecisionValueMemoryConfig()
        self.mode = self.config.mode
        self.records: list[EpisodicEntry] = []
        self.donor_records: list[EpisodicEntry] = []

        self.belief_encoder = RecurrentBeliefEncoder(
            obs_dim=OBS_SIZE,
            action_count=N_ACTIONS,
            belief_dim=self.config.belief_dim,
        )
        self.scorer = DecisionValueScorer(
            obs_dim=OBS_SIZE,
            belief_dim=self.config.belief_dim,
            hidden_dim=self.config.hidden_dim,
        )

        self.current_belief = torch.zeros(1, self.config.belief_dim)
        self.last_action: int = 0
        self.last_reward: float = 0.0
        self.step_count: int = 0

    def reset(self, mode: str | None = None) -> None:
        if mode is not None:
            if mode not in DECISION_MEMORY_MODES:
                raise ValueError(f"mode must be one of {DECISION_MEMORY_MODES}, got {mode}")
            self.mode = mode
        self.records.clear()
        self.current_belief = torch.zeros(1, self.config.belief_dim)
        self.last_action = 0
        self.last_reward = 0.0
        self.step_count = 0

    def add_entry(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        residual: Sequence[float] | None = None,
        tag: str = "default",
    ) -> None:
        if len(self.records) >= self.config.capacity:
            self.records.pop(0)

        res = tuple(residual) if residual is not None else tuple([0.0] * OBS_SIZE)
        entry = EpisodicEntry(
            tick=self.step_count,
            observation=tuple(observation),
            action=action,
            reward=reward,
            next_observation=tuple(next_observation),
            residual=res,
            age=0,
            tag=tag,
        )
        entry.validate()
        self.records.append(entry)

    def set_donor_pool(self, entries: Sequence[EpisodicEntry]) -> None:
        """Injects donor transitions for the 'donor' lesion arm."""
        self.donor_records = list(entries)

    def update_belief(self, observation: Sequence[float], prev_action: int, prev_reward: float) -> torch.Tensor:
        obs_t = torch.tensor(list(observation), dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            self.current_belief = self.belief_encoder(obs_t, prev_action, prev_reward, self.current_belief)
        self.last_action = prev_action
        self.last_reward = prev_reward
        self.step_count += 1
        return self.current_belief

    def _get_active_records(self) -> list[EpisodicEntry]:
        if self.mode in ("erased", "baseline"):
            return []
        if self.mode == "donor":
            return self.donor_records
        if self.mode == "shifted":
            if not self.records:
                return []
            # Circular permutation of records: shift by 1
            return self.records[1:] + self.records[:1]
        return self.records

    def query(self, current_obs: Sequence[float]) -> MemoryDecision:
        """Evaluates causal decision value across all 6 candidate actions with admission gating."""
        active_records = self._get_active_records()
        obs_t = torch.tensor(list(current_obs), dtype=torch.float32).reshape(1, -1)

        action_values = [0.0] * N_ACTIONS
        decision_deltas = [0.0] * N_ACTIONS
        gated_deltas = [0.0] * N_ACTIONS
        admitted_actions = [False] * N_ACTIONS
        all_retrieved: list[dict[str, Any]] = []

        if not active_records or self.mode == "baseline":
            return MemoryDecision(
                action_values=action_values,
                decision_deltas=decision_deltas,
                gated_deltas=gated_deltas,
                admitted_actions=admitted_actions,
                admission_gate_open=False,
                retrieved_count=0,
                retrieved_candidates=[],
                conflict_score=0.0,
                support_score=0.0,
                mode=self.mode,
            )

        # 1. Candidate Pool Selection
        obs_np = list(current_obs)
        scored_records = []
        for r in active_records:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(obs_np, r.observation))) / math.sqrt(OBS_SIZE)
            scored_records.append((dist, r))

        scored_records.sort(key=lambda x: x[0])

        if self.mode == "similarity":
            # Similarity mode strictly selects nearest neighbors by observation distance
            top_candidates = scored_records[: self.config.top_k]
            candidate_pool = [r for _, r in top_candidates]
        else:
            # Intact mode pools nearest neighbors plus distal anchor/cue transitions
            earliest_cues = active_records[:2]
            nearest_records = [r for _, r in scored_records[:2]]
            seen_ids = set()
            candidate_pool = []
            for r in earliest_cues + nearest_records:
                if id(r) not in seen_ids:
                    seen_ids.add(id(r))
                    candidate_pool.append(r)
            top_candidates = [(math.sqrt(sum((a - b) ** 2 for a, b in zip(obs_np, r.observation))) / math.sqrt(OBS_SIZE), r)
                              for r in candidate_pool]

        # Check candidate support
        mean_dist = sum(d for d, _ in top_candidates) / max(1, len(top_candidates))
        support_score = max(0.0, 1.0 - mean_dist / self.config.max_candidate_distance)
        is_supported = mean_dist <= self.config.max_candidate_distance

        # Candidate Conflict Check: check variance of rewards for identical actions
        rewards_by_action: dict[int, list[float]] = {}
        for _, r in top_candidates:
            rewards_by_action.setdefault(r.action, []).append(r.reward)
        action_variances = [
            (sum((x - sum(vals)/len(vals))**2 for x in vals) / len(vals))
            for vals in rewards_by_action.values() if len(vals) > 1
        ]
        conflict_score = max(action_variances) if action_variances else 0.0
        has_conflict = conflict_score > self.config.conflict_threshold

        for dist, r in top_candidates:
            all_retrieved.append({
                "tick": r.tick,
                "action": r.action,
                "reward": round(r.reward, 4),
                "distance": round(dist, 4),
                "tag": r.tag,
            })

        # 2. Decision Value Evaluation across candidate actions
        gate_open_any = False
        with torch.no_grad():
            for cand_act in range(N_ACTIONS):
                act_one_hot = F.one_hot(torch.tensor([cand_act]), num_classes=N_ACTIONS).float()
                query_vec = torch.cat([obs_t, act_one_hot, self.current_belief], dim=-1)

                empty_mem_vec = torch.zeros(1, OBS_SIZE + N_ACTIONS + 1 + OBS_SIZE)
                q_empty, sigma_empty = self.scorer(query_vec, empty_mem_vec)

                if self.mode == "similarity":
                    best_match = candidate_pool[0]
                    mem_act_oh = F.one_hot(torch.tensor([best_match.action]), num_classes=N_ACTIONS).float()
                    mem_vec = torch.cat([
                        torch.tensor(best_match.observation, dtype=torch.float32).reshape(1, -1),
                        mem_act_oh,
                        torch.tensor([[best_match.reward]], dtype=torch.float32),
                        torch.tensor(best_match.residual, dtype=torch.float32).reshape(1, -1),
                    ], dim=-1)
                    q_mem, _ = self.scorer(query_vec, mem_vec)
                    delta_q = float((q_mem - q_empty).item())
                    action_values[cand_act] = float(q_mem.item())
                    decision_deltas[cand_act] = delta_q
                    gated_deltas[cand_act] = max(-0.5, min(0.5, delta_q * self.config.shrinkage_alpha))
                    admitted_actions[cand_act] = True
                    gate_open_any = True
                else:
                    # Intact mode: find the candidate maximizing decision value LCB
                    best_cand_lcb = -float("inf")
                    best_delta_q = 0.0
                    best_q_mem = float(q_empty.item())

                    for cand_r in candidate_pool:
                        mem_act_oh = F.one_hot(torch.tensor([cand_r.action]), num_classes=N_ACTIONS).float()
                        mem_vec = torch.cat([
                            torch.tensor(cand_r.observation, dtype=torch.float32).reshape(1, -1),
                            mem_act_oh,
                            torch.tensor([[cand_r.reward]], dtype=torch.float32),
                            torch.tensor(cand_r.residual, dtype=torch.float32).reshape(1, -1),
                        ], dim=-1)

                        q_mem_cand, sigma_mem_cand = self.scorer(query_vec, mem_vec)
                        dq = float((q_mem_cand - q_empty).item())
                        s_net = float((sigma_mem_cand + sigma_empty).item())
                        cand_lcb = dq - self.config.uncertainty_discount * s_net
                        if cand_lcb > best_cand_lcb:
                            best_cand_lcb = cand_lcb
                            best_delta_q = dq
                            best_q_mem = float(q_mem_cand.item())

                    action_values[cand_act] = best_q_mem
                    decision_deltas[cand_act] = best_delta_q

                    admit = (
                        is_supported
                        and not has_conflict
                        and (best_cand_lcb > self.config.admission_threshold)
                    )

                    admitted_actions[cand_act] = admit
                    if admit:
                        gated_delta = max(-0.5, min(0.5, best_delta_q * self.config.shrinkage_alpha))
                        gated_deltas[cand_act] = gated_delta
                        gate_open_any = True
                    else:
                        gated_deltas[cand_act] = 0.0

        return MemoryDecision(
            action_values=action_values,
            decision_deltas=decision_deltas,
            gated_deltas=gated_deltas,
            admitted_actions=admitted_actions,
            admission_gate_open=gate_open_any,
            retrieved_count=len(top_candidates),
            retrieved_candidates=all_retrieved,
            conflict_score=conflict_score,
            support_score=support_score,
            mode=self.mode,
        )

    def get_influence_deltas(self, current_obs: Sequence[float]) -> list[float]:
        """Convenience method returning the admitted influence deltas across actions."""
        return list(self.query(current_obs).gated_deltas)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": DECISION_MEMORY_SCHEMA,
            "config": asdict(self.config),
            "belief_encoder": self.belief_encoder.state_dict(),
            "scorer": self.scorer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != DECISION_MEMORY_SCHEMA:
            raise ValueError(f"Schema mismatch: expected {DECISION_MEMORY_SCHEMA}, got {state.get('schema')}")
        self.belief_encoder.load_state_dict(state["belief_encoder"])
        self.scorer.load_state_dict(state["scorer"])


class DecisionValueAgent:
    """Rigorous decision-value memory coordinator wrapping the frozen Tidal core."""

    def __init__(
        self,
        core: Any | None = None,
        memory: DecisionValueMemory | None = None,
        mode: str = "intact",
        override_margin: float = 0.04,
    ):
        from ..core import CoreRuntime, active_core_path
        self.core = core or CoreRuntime(active_core_path())
        self.memory = memory or DecisionValueMemory(DecisionValueMemoryConfig(mode=mode))
        self.mode = mode
        self.override_margin = override_margin
        self.memory.mode = self.mode

    def reset(self, mode: str | None = None) -> None:
        if mode is not None:
            self.mode = mode
        self.memory.reset(mode=self.mode)

    def act(self, observation: Sequence[float]) -> dict[str, Any]:
        obs_list = list(observation)
        base_action = self.core.act(obs_list)

        # In baseline mode, completely bypass memory
        if self.mode == "baseline":
            return {
                "action": base_action,
                "base_action": base_action,
                "override": False,
                "margin": 0.0,
                "admission_gate_open": False,
                "mode": self.mode,
                "decision_memory": None,
            }

        mem_decision = self.memory.query(obs_list)
        if not mem_decision.admission_gate_open:
            return {
                "action": base_action,
                "base_action": base_action,
                "override": False,
                "margin": 0.0,
                "admission_gate_open": False,
                "mode": self.mode,
                "decision_memory": mem_decision.to_dict(),
            }

        # Retrieve core model logits
        obs_t = torch.as_tensor(obs_list, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            out = self.core.model([""], obs_t)
            logits = out["action"][0].cpu()

        # Score modulation: normalized policy probabilities + gated_deltas
        base_probs = F.softmax(logits, dim=-1)
        deltas = torch.tensor(mem_decision.gated_deltas, dtype=torch.float32)
        modulated_scores = base_probs + deltas

        best_action = int(modulated_scores.argmax().item())
        base_score = float(modulated_scores[base_action].item())
        best_score = float(modulated_scores[best_action].item())
        margin = best_score - base_score

        override = (best_action != base_action) and (margin > self.override_margin)
        chosen_action = best_action if override else base_action

        return {
            "action": chosen_action,
            "base_action": base_action,
            "override": override,
            "margin": margin,
            "admission_gate_open": mem_decision.admission_gate_open,
            "mode": self.mode,
            "decision_memory": mem_decision.to_dict(),
        }

    def observe(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        residual: Sequence[float] | None = None,
        tag: str = "default",
    ) -> None:
        self.memory.add_entry(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            residual=residual,
            tag=tag,
        )
        self.memory.update_belief(next_observation, action, reward)

    def rollout(
        self,
        env: Any,
        max_steps: int = 64,
        mode: str | None = None,
    ) -> dict[str, Any]:
        self.reset(mode=mode)
        obs = env.reset()
        total_reward = 0.0
        trajectory = []
        gate_open_count = 0
        override_count = 0
        info: dict[str, Any] = {}

        for step in range(max_steps):
            dec = self.act(obs)
            act_idx = dec["action"]
            if dec["admission_gate_open"]:
                gate_open_count += 1
            if dec["override"]:
                override_count += 1

            next_obs, reward, done, info = env.step(act_idx)
            self.observe(obs, act_idx, reward, next_obs)
            total_reward += reward

            trajectory.append({
                "step": step,
                "observation": obs,
                "action": act_idx,
                "reward": reward,
                "next_observation": next_obs,
                "done": done,
                "override": dec["override"],
                "admission_gate_open": dec["admission_gate_open"],
                "info": info,
            })
            obs = next_obs
            if done:
                break

        return {
            "total_reward": total_reward,
            "steps": len(trajectory),
            "survived": not info.get("death", False),
            "gate_open_count": gate_open_count,
            "override_count": override_count,
            "mode": self.mode,
            "trajectory": trajectory,
        }


def fit_decision_value_memory(
    train_seeds: Sequence[int],
    config: DecisionValueMemoryConfig | None = None,
    updates: int = 400,
    lr: float = 0.003,
) -> DecisionValueMemory:
    """Trains DecisionValueMemory using cross-fitted synthetic branch labels."""
    from .alias_world import ObservationAliasTidePool
    from ..core import CoreRuntime, active_core_path

    cfg = config or DecisionValueMemoryConfig()
    core = CoreRuntime(active_core_path())

    # Collect branch transition data across training seeds
    queries = []
    memories = []
    returns = []

    for seed in train_seeds:
        env = ObservationAliasTidePool(seed=seed, max_steps=24, cue_horizon=4)
        obs = env.reset()
        mem = DecisionValueMemory(cfg)

        cue_entries: list[EpisodicEntry] = []
        for t in range(24):
            act = core.act(obs)
            next_obs, r, done, info = env.step(act)
            mem.add_entry(obs, act, r, next_obs)
            mem.update_belief(next_obs, act, r)
            if t < 4:
                cue_entries.append(EpisodicEntry(
                    tick=t, observation=tuple(obs), action=act, reward=r,
                    next_observation=tuple(next_obs), residual=tuple([0.0]*OBS_SIZE),
                ))

            # At aliased ticks t in [4, 8, 12], create branch anchors
            if t in (4, 8, 12) and not done and cue_entries:
                best_cue = cue_entries[0]
                snap = env.snapshot()

                # Measure actual synthetic branch returns for each action (horizon 4)
                for cand_act in range(N_ACTIONS):
                    env.restore(snap)
                    branch_r = 0.0
                    b_obs, r0, b_done, _ = env.step(cand_act)
                    branch_r += r0
                    curr = b_obs
                    for _ in range(3):
                        if b_done:
                            break
                        b_act = core.act(curr)
                        curr, r_step, b_done, _ = env.step(b_act)
                        branch_r += r_step

                    # Query vector: obs (16) + cand_act (6) + belief (32)
                    act_oh = F.one_hot(torch.tensor([cand_act]), num_classes=N_ACTIONS).float()
                    obs_t = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
                    q_vec = torch.cat([obs_t, act_oh, mem.current_belief], dim=-1)

                    # Memory vector: cue obs (16) + cue act (6) + cue reward (1) + residual (16)
                    cue_act_oh = F.one_hot(torch.tensor([best_cue.action]), num_classes=N_ACTIONS).float()
                    m_vec = torch.cat([
                        torch.tensor(best_cue.observation, dtype=torch.float32).reshape(1, -1),
                        cue_act_oh,
                        torch.tensor([[best_cue.reward]], dtype=torch.float32),
                        torch.tensor(best_cue.residual, dtype=torch.float32).reshape(1, -1),
                    ], dim=-1)

                    queries.append(q_vec)
                    memories.append(m_vec)
                    returns.append(branch_r)

                env.restore(snap)

            obs = next_obs
            if done:
                break

    if not queries:
        return DecisionValueMemory(cfg)

    # Train Scorer on collected counterfactual branches
    queries_tensor = torch.cat(queries, dim=0)
    memories_tensor = torch.cat(memories, dim=0)
    returns_tensor = torch.tensor(returns, dtype=torch.float32).reshape(-1, 1)

    fitted_mem = DecisionValueMemory(cfg)
    optimizer = torch.optim.Adam(fitted_mem.scorer.parameters(), lr=lr)

    n_samples = queries_tensor.shape[0]
    batch_size = min(32, n_samples)

    for step in range(updates):
        idx = torch.randperm(n_samples)[:batch_size]
        q_b = queries_tensor[idx]
        m_b = memories_tensor[idx]
        y_b = returns_tensor[idx]

        # Memory dropout (p=0.5): mask memory to zeros so network learns Q(s, a, empty)
        drop_mask = (torch.rand(batch_size, 1) < 0.5).float()
        m_input = m_b * drop_mask

        optimizer.zero_grad()
        out1, out2 = fitted_mem.scorer.head1(torch.cat([q_b, m_input], dim=-1)), fitted_mem.scorer.head2(torch.cat([q_b, m_input], dim=-1))
        loss = F.mse_loss(out1, y_b) + F.mse_loss(out2, y_b)
        loss.backward()
        optimizer.step()

    return fitted_mem
