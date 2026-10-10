"""POSEIDON COGNITIVE AGENT: Unified v2.0 Cognitive Autonomous Agent Architecture.

Unifies all v2.0 cognitive subsystems around the immutable, frozen Tidal core:
1. ORACLE: Bootstrapped calibrated dynamics ensemble for multi-step trajectory imagination.
2. MNEMOSYNE: 3-tier memory hierarchy (working, value-gated episodic, spatial cognitive map)
   supporting 5 matched lesion assays (intact, erased, shifted, random_retrieval, memory_free).
3. HORIZON-X: Multi-horizon (H in {1, 4, 8, 16}) risk-sensitive planning with dynamic
   metabolic reserve modulation, preventing the starvation pessimism trap.
4. ADAPT: Continual residual policy adaptation with anti-forgetting anchor regularization.
5. NEURO: Neuromodulated sparse connectome routing with metabolic gain control.
6. WORLDLAB: High-fidelity partially observable POMDP environments and simulation bridge.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .adapt import AdaptConfig, ContinualAdapter
from .core import CoreRuntime, active_core_path
from .horizon_x import HorizonXConfig, HorizonXPlanner
from .mnemosyne import MNEMOSYNE_MODES, MnemosyneConfig, MnemosyneMemory
from .neuro import NeuroConfig, NeuromodulatedRouter
from .oracle import OracleConfig, OracleEnsemble
from .world import ACTIONS, N_ACTIONS, OBS_SIZE, TidePool
from .worldlab import POMDPConfig, POMDPTidePool, StepResult


@dataclass
class CognitiveDecision:
    """Rich audit decision output from the CognitiveAgent."""
    action: int
    action_name: str
    base_action: int
    base_action_name: str
    override: bool
    margin: float
    surprise: float
    reserve_index: float
    dynamic_lambda: float
    active_sparsity: float
    neuromodulatory_gain: float
    memory_patch: str | None
    food_lcb: float
    water_lcb: float
    mode: str
    evaluations: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "action_name": self.action_name,
            "base_action": self.base_action,
            "base_action_name": self.base_action_name,
            "override": self.override,
            "margin": round(self.margin, 6),
            "surprise": round(self.surprise, 6),
            "reserve_index": round(self.reserve_index, 6),
            "dynamic_lambda": round(self.dynamic_lambda, 6),
            "active_sparsity": round(self.active_sparsity, 6),
            "neuromodulatory_gain": round(self.neuromodulatory_gain, 6),
            "memory_patch": self.memory_patch,
            "food_lcb": round(self.food_lcb, 4),
            "water_lcb": round(self.water_lcb, 4),
            "mode": self.mode,
            "evaluations": self.evaluations,
        }


class CognitiveAgent:
    """Supermix Poseidon v2.0 Cognitive Agent Coordinator."""

    SCHEMA = "poseidon-cognitive-agent-v2"

    def __init__(
        self,
        core: CoreRuntime | None = None,
        mode: str = "intact",
        oracle_config: OracleConfig | None = None,
        mnemosyne_config: MnemosyneConfig | None = None,
        horizon_config: HorizonXConfig | None = None,
        adapt_config: AdaptConfig | None = None,
        neuro_config: NeuroConfig | None = None,
    ):
        self.core = core or CoreRuntime(active_core_path())
        self.mode = mode
        if self.mode not in MNEMOSYNE_MODES:
            raise ValueError(f"mode must be one of {MNEMOSYNE_MODES}, got {mode}")

        # Subsystems
        self.oracle = OracleEnsemble(oracle_config or OracleConfig())
        mnemo_cfg = mnemosyne_config or MnemosyneConfig(mode=self.mode)
        mnemo_cfg.mode = self.mode
        self.mnemosyne = MnemosyneMemory(mnemo_cfg)
        self.planner = HorizonXPlanner(core=self.core, oracle=self.oracle, config=horizon_config)
        self.adapter = ContinualAdapter(core=self.core, config=adapt_config)
        self.router = NeuromodulatedRouter(neuro_config or NeuroConfig(dim=self.core.model.config.hidden_size))

        # Runtime transition memory
        self.last_observation: list[float] | None = None
        self.predicted_next_obs: list[float] | None = None
        self.step_count: int = 0

    def reset(self, mode: str | None = None) -> None:
        """Resets the agent's internal state for a new episode."""
        if mode is not None:
            if mode not in MNEMOSYNE_MODES:
                raise ValueError(f"mode must be one of {MNEMOSYNE_MODES}, got {mode}")
            self.mode = mode

        self.mnemosyne.reset(mode=self.mode)
        self.last_observation = None
        self.predicted_next_obs = None
        self.step_count = 0

    def act(self, observation: Sequence[float]) -> CognitiveDecision:
        """Computes action decision through integrated cognitive hierarchy."""
        obs_list = list(observation)
        obs_t = torch.tensor(obs_list, dtype=torch.float32).reshape(1, -1)

        # 1. Update 3-Tier Memory
        mem_info = self.mnemosyne.step_observation(
            observation=obs_list,
            predicted_observation=self.predicted_next_obs,
        )

        # 2. Neuromodulated routing
        vitals = torch.tensor([[
            obs_list[0],  # health
            obs_list[1],  # energy
            obs_list[2],  # hydration
            obs_list[5],  # threat
        ]], dtype=torch.float32)

        core_out = self.core.model(["Survive"], obs_t)
        latent_state = core_out["memory"]
        neuro_out = self.router(latent_state, vitals)

        # 3. Multi-Horizon Risk-Sensitive Planning
        plan_out = self.planner.plan(obs_list)

        # Create rich decision
        decision = CognitiveDecision(
            action=plan_out["selected_action"],
            action_name=plan_out["selected_action_name"],
            base_action=plan_out["base_action"],
            base_action_name=plan_out["base_action_name"],
            override=plan_out["override"],
            margin=plan_out["margin"],
            surprise=mem_info["surprise"],
            reserve_index=plan_out["reserve_index"],
            dynamic_lambda=plan_out["dynamic_lambda"],
            active_sparsity=float(neuro_out["active_sparsity"].item()),
            neuromodulatory_gain=float(neuro_out["gain"].item()),
            memory_patch=mem_info["patch_id"],
            food_lcb=mem_info["estimated_food_lcb"],
            water_lcb=mem_info["estimated_water_lcb"],
            mode=self.mode,
            evaluations=plan_out["candidate_evaluations"],
        )

        self.last_observation = obs_list
        return decision

    def plan(self, observation: Sequence[float]) -> dict[str, Any]:
        """Planner interface returning decision dictionary with action and telemetry."""
        return self.act(observation).to_dict()

    def observe_transition(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        info: Any = None,
        done: bool = False,
    ) -> None:
        """Records transition across memory, replay buffer, and computes next prediction."""
        obs_list = list(observation)
        next_obs_list = list(next_observation)

        # Record in Mnemosyne episodic store
        self.mnemosyne.record_transition(
            observation=obs_list,
            action=action,
            reward=reward,
            next_observation=next_obs_list,
        )

        # Record in ContinualAdapter replay buffer
        self.adapter.replay.add(
            obs=obs_list,
            action=action,
            reward=reward,
            next_obs=next_obs_list,
        )

        # Generate world-model prediction for next step surprise calculation
        obs_t = torch.tensor(obs_list, dtype=torch.float32).reshape(1, -1)
        act_t = torch.tensor([action], dtype=torch.long)
        core_out = self.core.model(["Survive"], obs_t)
        lat_t = core_out["memory"]

        with torch.no_grad():
            wm_out = self.oracle(lat_t, obs_t, act_t)
            delta = wm_out["mean_delta"].squeeze(0)
            pred_next = torch.clamp(obs_t.squeeze(0) + delta, 0.0, 1.0)
            self.predicted_next_obs = pred_next.tolist()

        self.step_count += 1

    def rollout(
        self,
        env: TidePool | POMDPTidePool,
        max_steps: int = 64,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Rolls out a full episode recording transitions, metrics, and overrides."""
        self.reset(mode=mode)
        obs = env.reset() if hasattr(env, "reset") else env.observe()
        total_reward = 0.0
        decisions: list[dict[str, Any]] = []
        survived = True

        for step in range(max_steps):
            decision = self.act(obs)
            decisions.append(decision.to_dict())

            act = decision.action
            if isinstance(env, POMDPTidePool):
                res = env.step(act)
                next_obs = res.observation
                reward = res.reward
                done = res.done
            else:
                next_obs, reward, done, info = env.step(act)

            total_reward += reward
            self.observe_transition(obs, act, reward, next_obs, done=done)

            if next_obs[0] <= 0.0 or done:
                survived = (next_obs[0] > 0.0)
                break
            obs = next_obs

        overrides = sum(1 for d in decisions if d["override"])
        return {
            "schema": self.SCHEMA,
            "mode": self.mode,
            "steps": len(decisions),
            "total_reward": float(total_reward),
            "survived": bool(survived),
            "overrides": overrides,
            "override_rate": float(overrides / len(decisions)) if decisions else 0.0,
            "final_health": float(obs[0]),
            "decisions": decisions,
        }
