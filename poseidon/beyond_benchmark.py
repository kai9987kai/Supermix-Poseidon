"""Empirical Benchmark Suite for Supermix Beyond.

Compares:
1. Reactive Policy (Baseline behavior cloning / DAgger prior)
2. Single-Model MPC (Poseidon v0.1 lookahead)
3. Uncertainty-Aware Ensemble MPC (Supermix Beyond Risk-Sensitive Planner)

Across 100 independently seeded held-out test episodes using paired seeds,
measuring survival rate, step duration, cumulative return, catastrophic error rate,
and per-step CPU latency.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from .core import CoreRuntime
from .ensemble import RiskConfig, UncertaintyAwarePlanner, WorldModelEnsemble
from .planning import ModelPredictivePlanner, PlanningConfig
from .world import TidePool, N_ACTIONS


@dataclass
class BenchmarkSummary:
    agent_name: str
    episodes: int
    scarcity: float
    survival_rate: float
    survival_ci_95: float
    mean_steps: float
    mean_reward: float
    catastrophic_deaths: int
    catastrophic_rate: float
    mean_latency_ms: float


def run_paired_benchmark(
    core_path: str = "runs/tidal_dagger/core.pt",
    n_episodes: int = 100,
    base_seed: int = 1000,
    scarcity: float = 2.5,
    max_steps: int = 256,
) -> dict[str, BenchmarkSummary]:
    """Runs matched, seed-paired evaluation across the three planning paradigms."""
    runtime = CoreRuntime(core_path)
    single_mpc = ModelPredictivePlanner(runtime, PlanningConfig(horizon=2))
    
    # Beyond uncertainty planner with 3-model ensemble
    risk_cfg = RiskConfig(
        ensemble_size=3,
        horizon=2,
        uncertainty_penalty=1.5,
        failure_penalty=5.0
    )
    beyond_planner = UncertaintyAwarePlanner(runtime, config=risk_cfg)

    agents = {
        "ReactivePolicy": lambda obs: runtime.act(obs),
        "SingleModelMPC": lambda obs: single_mpc.act(obs),
        "UncertaintyAwareMPC": lambda obs: beyond_planner.act(obs),
    }

    results: dict[str, BenchmarkSummary] = {}

    for name, act_fn in agents.items():
        survived = 0
        total_steps = 0
        total_reward = 0.0
        catastrophic_deaths = 0
        latencies = []

        for i in range(n_episodes):
            seed = base_seed + i
            env = TidePool(seed=seed, scarcity=scarcity, max_steps=max_steps)
            obs = env.reset(seed)
            done = False
            ep_steps = 0
            ep_reward = 0.0

            while not done:
                t0 = time.perf_counter()
                action = act_fn(obs)
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)

                obs, reward, done, info = env.step(action)
                ep_steps += 1
                ep_reward += reward

            if env.alive:
                survived += 1
            else:
                # Death analysis: check if death was sudden/unmitigated
                if info.get("death") in ("starvation", "dehydration", "hazard"):
                    catastrophic_deaths += 1

            total_steps += ep_steps
            total_reward += ep_reward

        surv_rate = survived / n_episodes
        # Wilson / Normal approx 95% confidence interval
        ci_95 = 1.96 * math.sqrt(max(0.0, surv_rate * (1.0 - surv_rate) / n_episodes))
        mean_steps = total_steps / n_episodes
        mean_rew = total_reward / n_episodes
        cat_rate = catastrophic_deaths / n_episodes
        mean_lat = sum(latencies) / max(1, len(latencies))

        summary = BenchmarkSummary(
            agent_name=name,
            episodes=n_episodes,
            scarcity=scarcity,
            survival_rate=round(surv_rate, 4),
            survival_ci_95=round(ci_95, 4),
            mean_steps=round(mean_steps, 2),
            mean_reward=round(mean_rew, 2),
            catastrophic_deaths=catastrophic_deaths,
            catastrophic_rate=round(cat_rate, 4),
            mean_latency_ms=round(mean_lat, 3),
        )
        results[name] = summary

    return results
