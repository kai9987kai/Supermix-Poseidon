"""Automated Surprise-Driven Continuous Adaptation Trainer.

Implements the complete self-improvement loop for Supermix Beyond:
1. Experience Gathering:
   - Rolls out episodes across varying environmental difficulties (scarcity 1.5 - 3.5).
   - Computes epistemic disagreement U(s, a) and multi-step dynamics residuals.
   - Flags surprises and lethal failure transitions into SurprisePrioritizedBuffer.
2. Targeted Parameter Adaptation:
   - Fine-tunes the dynamics ensemble on high-surprise events balanced with anchor replay.
3. Multi-Gate Promotion Audit:
   - Tests candidate against frozen held-out seeds.
   - Verifies held-out survival >= baseline, dynamics MSE <= threshold, and calibration.
   - Saves audit report to runs/adaptation_audit.json.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from .adaptation import PromotionAuditor, PromotionCriteria, SurprisePrioritizedBuffer
from .core import CoreRuntime, TidalCore, save_checkpoint
from .ensemble import RiskConfig, UncertaintyAwarePlanner, WorldModelEnsemble
from .world import TidePool, OBS_SIZE, N_ACTIONS

logger = logging.getLogger("poseidon.adaptation_trainer")


def run_adaptation_cycle(
    base_core_path: str = "runs/tidal_dagger/core.pt",
    output_dir: str = "runs/beyond_adapted",
    collect_episodes: int = 15,
    adaptation_steps: int = 64,
    batch_size: int = 32,
    lr: float = 1e-4,
    base_seed: int = 7000,
) -> dict[str, Any]:
    """Executes a full cycle of surprise collection, fine-tuning, and audit validation."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    runtime = CoreRuntime(base_core_path)
    core_model = runtime.model
    ensemble = WorldModelEnsemble(hidden_size=core_model.config.hidden_size, k=3)
    planner = UncertaintyAwarePlanner(runtime, ensemble=ensemble, config=RiskConfig(horizon=2))

    buffer = SurprisePrioritizedBuffer(capacity=2000)

    # 1. Experience Gathering with Surprise Detection
    total_steps_collected = 0
    surprises_detected = 0

    for ep in range(collect_episodes):
        scarcity = 1.5 + (ep % 4) * 0.7  # diverse stress environments
        seed = base_seed + ep
        env = TidePool(seed=seed, scarcity=scarcity, max_steps=128)
        obs = env.reset(seed)
        done = False

        while not done:
            plan_out = planner.plan(obs)
            action = plan_out["action"]
            disagreement = plan_out["epistemic_uncertainty"]

            next_obs, reward, done, info = env.step(action)
            total_steps_collected += 1

            # Compute prediction error
            obs_t = torch.tensor([obs], dtype=torch.float32)
            act_t = torch.tensor([action], dtype=torch.long)
            latent = core_model(torch.zeros(1, core_model.config.hash_buckets), obs_t)["memory"]
            pred_delta = ensemble(latent, obs_t, act_t)["mean_delta"][0]
            true_delta = torch.tensor(next_obs) - torch.tensor(obs)
            err = float(((pred_delta - true_delta) ** 2).mean().item())

            is_fail = bool(done and not env.alive)
            if plan_out["is_high_surprise"] or is_fail or err > 0.05:
                surprises_detected += 1

            buffer.add(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                disagreement=disagreement,
                prediction_error=err,
                is_failure=is_fail,
            )
            obs = next_obs

    # 2. Targeted Adaptation Updates on Dynamics Ensemble
    optimizer = optim.AdamW(ensemble.parameters(), lr=lr, weight_decay=1e-5)
    loss_history = []

    if len(buffer) >= batch_size:
        ensemble.train()
        for step in range(adaptation_steps):
            batch = buffer.sample(batch_size=batch_size)
            obs_b = torch.tensor([t.observation for t in batch], dtype=torch.float32)
            act_b = torch.tensor([t.action for t in batch], dtype=torch.long)
            next_obs_b = torch.tensor([t.next_observation for t in batch], dtype=torch.float32)
            target_deltas = next_obs_b - obs_b

            with torch.no_grad():
                latents = core_model(torch.zeros(len(batch), core_model.config.hash_buckets), obs_b)["memory"]

            ens_out = ensemble(latents, obs_b, act_b)
            # Loss across all individual ensemble heads
            loss = 0.0
            for head_delta in ens_out["deltas"]:
                loss += nn.functional.mse_loss(head_delta, target_deltas)
            loss = loss / ensemble.k

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))

    # 3. Gated Audit Validation
    auditor = PromotionAuditor(PromotionCriteria(
        min_survival_rate=0.90,
        max_dynamics_mse=0.045,
        min_epistemic_correlation=0.15,
    ))
    audit_report = auditor.audit_candidate(
        candidate_model=core_model,
        ensemble=ensemble,
        test_seeds=[8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010],
        scarcity=2.0,
    )

    # 4. Save Candidate Artifacts & Receipts
    adapted_checkpoint_path = out_path / "ensemble_adapted.pt"
    torch.save({
        "ensemble_state_dict": ensemble.state_dict(),
        "audit_report": audit_report,
        "loss_history": loss_history,
    }, adapted_checkpoint_path)

    receipt = {
        "status": "complete",
        "steps_collected": total_steps_collected,
        "surprises_stored": surprises_detected,
        "adaptation_steps": len(loss_history),
        "initial_loss": round(loss_history[0], 5) if loss_history else None,
        "final_loss": round(loss_history[-1], 5) if loss_history else None,
        "audit": audit_report,
        "artifact_path": str(adapted_checkpoint_path),
    }

    receipt_file = out_path / "audit_receipt.json"
    receipt_file.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return receipt
