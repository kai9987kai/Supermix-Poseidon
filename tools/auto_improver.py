"""Supermix Poseidon Autonomous Overnight Experimentation & Continuous Improvement Loop.

Runs continuous matched-seed empirical experiments until the target hour (06:00 AM),
evaluating hypothesis mutations across Active Inference, Horizon-X planning,
Metacognitive monitoring, Memory geometry, and Formal Safety contracts.

Decision Gating:
1. Syntax & Unit Test Integrity (pytest verification).
2. Paired Matched-Seed POMDP Benchmark (4 difficult seeds).
3. Improvement Criterion:
   - Survival rate must not regress (candidate_survival >= champion_survival).
   - Paired reward gain must be strictly positive (delta_reward >= +0.005) OR
     survival must strictly increase with delta_reward >= -0.05.
4. If Accepted: Commits cleanly to git and updates champion baseline.
5. If Rejected: Automatically restores working tree via git revert.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

# Ensure CPU safety
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TORCH_NUM_THREADS"] = "2"

ROOT = Path(__file__).resolve().parent.parent

# Output logs
OUT_DIR = ROOT / "outputs" / "auto_improvements"
HISTORY_FILE = OUT_DIR / "history.jsonl"
SUMMARY_FILE = OUT_DIR / "OVERNIGHT_REPORT.md"
RECEIPTS_DIR = OUT_DIR / "receipts"

# Target evaluation seeds
EVAL_SEEDS = [90500000, 90500001, 90500002, 90500003]
EVAL_MAX_STEPS = 32
EVAL_SCARCITY = 2.8

# Search Space Definitions: (filepath, regex_pattern, min_val, max_val, format_str)
SEARCH_SPACE = [
    # Active Inference
    {
        "file": "poseidon/active_inference.py",
        "name": "base_epistemic_weight",
        "pattern": r"(base_epistemic_weight:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.10, "max": 0.65, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "min_epistemic_weight",
        "pattern": r"(min_epistemic_weight:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.01, "max": 0.12, "type": "float", "fmt": "{:.3f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "w_health",
        "pattern": r"(w_health:\s*float\s*=\s*)([0-9\.]+)",
        "min": 3.0, "max": 8.0, "type": "float", "fmt": "{:.1f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "w_energy",
        "pattern": r"(w_energy:\s*float\s*=\s*)([0-9\.]+)",
        "min": 1.5, "max": 5.0, "type": "float", "fmt": "{:.1f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "w_hydration",
        "pattern": r"(w_hydration:\s*float\s*=\s*)([0-9\.]+)",
        "min": 2.0, "max": 5.5, "type": "float", "fmt": "{:.1f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "w_stamina",
        "pattern": r"(w_stamina:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.8, "max": 2.5, "type": "float", "fmt": "{:.1f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "w_threat",
        "pattern": r"(w_threat:\s*float\s*=\s*)([0-9\.]+)",
        "min": 2.0, "max": 6.0, "type": "float", "fmt": "{:.1f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "stamina_depletion_cost",
        "pattern": r"(stamina_depletion_cost:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.01, "max": 0.10, "type": "float", "fmt": "{:.3f}",
    },
    # Horizon-X Planner
    {
        "file": "poseidon/horizon_x.py",
        "name": "base_lambda",
        "pattern": r"(base_lambda:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.05, "max": 0.35, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/horizon_x.py",
        "name": "base_beta",
        "pattern": r"(base_beta:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.60, "max": 2.50, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/horizon_x.py",
        "name": "override_threshold",
        "pattern": r"(override_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.005, "max": 0.050, "type": "float", "fmt": "{:.3f}",
    },
    {
        "file": "poseidon/horizon_x.py",
        "name": "horizon_decay",
        "pattern": r"(horizon_decay:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.65, "max": 0.95, "type": "float", "fmt": "{:.2f}",
    },
    # Formal Safety Guardrails
    {
        "file": "poseidon/guardrails.py",
        "name": "critical_hydration_threshold",
        "pattern": r"(critical_hydration_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.10, "max": 0.22, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/guardrails.py",
        "name": "critical_energy_threshold",
        "pattern": r"(critical_energy_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.10, "max": 0.22, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/guardrails.py",
        "name": "critical_stamina_threshold",
        "pattern": r"(critical_stamina_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.04, "max": 0.14, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/guardrails.py",
        "name": "severe_storm_threshold",
        "pattern": r"(severe_storm_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.60, "max": 0.85, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/guardrails.py",
        "name": "safe_shelter_threshold",
        "pattern": r"(safe_shelter_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.30, "max": 0.55, "type": "float", "fmt": "{:.2f}",
    },
    # Metacognitive Monitor
    {
        "file": "poseidon/metacognition.py",
        "name": "volatility_threshold_elevated",
        "pattern": r"(volatility_threshold_elevated:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.10, "max": 0.24, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/metacognition.py",
        "name": "volatility_threshold_critical",
        "pattern": r"(volatility_threshold_critical:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.25, "max": 0.45, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/metacognition.py",
        "name": "energy_deep_horizon_threshold",
        "pattern": r"(energy_deep_horizon_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.35, "max": 0.65, "type": "float", "fmt": "{:.2f}",
    },
    # Spatial Cognitive Map
    {
        "file": "poseidon/mnemosyne.py",
        "name": "lcb_z",
        "pattern": r"(lcb_z:\s*float\s*=\s*)([0-9\.]+)",
        "min": 1.0, "max": 2.5, "type": "float", "fmt": "{:.2f}",
    },
]


def git_revert() -> None:
    """Restores working tree to last clean git commit."""
    subprocess.run(["git", "restore", "poseidon/", "tests/"], cwd=ROOT, check=False)


def run_tests() -> bool:
    """Verifies that unit tests and integrity invariants pass."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_cognitive_agent.py",
        "tests/test_active_inference.py",
        "tests/test_horizon_x.py",
        "tests/test_guardrails.py",
        "tests/test_metacognition.py",
        "-q",
    ]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return res.returncode == 0


def run_benchmark() -> dict:
    """Runs matched-seed evaluation across EVAL_SEEDS."""
    script = f"""
import torch, json
from poseidon.cognitive_agent import CognitiveAgent
from poseidon.worldlab import POMDPTidePool, POMDPConfig

torch.set_num_threads(2)
agent = CognitiveAgent()
rewards = []
survivals = []
overrides = 0
total_steps = 0

for s in {EVAL_SEEDS}:
    env = POMDPTidePool(POMDPConfig(seed=s, scarcity={EVAL_SCARCITY}, sensor_noise_std=0.02, fog_threshold=0.65), max_steps={EVAL_MAX_STEPS})
    res = agent.rollout(env, max_steps={EVAL_MAX_STEPS})
    rewards.append(res['total_reward'])
    survivals.append(1.0 if res['survived'] else 0.0)
    overrides += res['overrides']
    total_steps += res['steps']

out = {{
    'mean_reward': sum(rewards) / len(rewards),
    'survival_rate': sum(survivals) / len(survivals),
    'overrides': overrides,
    'total_steps': total_steps,
    'rewards': rewards,
}}
print('__RESULT__' + json.dumps(out))
"""
    cmd = [sys.executable, "-c", script]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Benchmark failed: {res.stderr}")
    for line in res.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    raise RuntimeError(f"Could not parse benchmark output: {res.stdout}")


def read_param_value(spec: dict) -> float:
    path = ROOT / spec["file"]
    content = path.read_text(encoding="utf-8")
    m = re.search(spec["pattern"], content)
    if not m:
        raise ValueError(f"Pattern {spec['pattern']} not found in {spec['file']}")
    return float(m.group(2))


def write_param_value(spec: dict, val: float) -> None:
    path = ROOT / spec["file"]
    content = path.read_text(encoding="utf-8")
    formatted = spec["fmt"].format(val)
    new_content, count = re.subn(spec["pattern"], r"\g<1>" + formatted, content, count=1)
    if count == 0:
        raise ValueError(f"Could not replace {spec['name']} in {spec['file']}")
    path.write_text(new_content, encoding="utf-8")


def parse_target_end_time(until_str: str | None, hours: float | None) -> datetime:
    now = datetime.now()
    if hours is not None:
        return now + timedelta(hours=hours)
    target_hour, target_min = 6, 0
    if until_str:
        parts = until_str.split(":")
        target_hour = int(parts[0])
        target_min = int(parts[1]) if len(parts) > 1 else 0
    target = now.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def main():
    parser = argparse.ArgumentParser(description="Autonomous Overnight Poseidon Improver")
    parser.add_argument("--until", type=str, default="06:00", help="Stop time HH:MM (default 06:00)")
    parser.add_argument("--duration-hours", type=float, default=None, help="Optional duration in hours")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)

    end_time = parse_target_end_time(args.until, args.duration_hours)
    print(f"================================================================")
    print(f" SUPERMIX POSEIDON AUTONOMOUS IMPROVEMENT DAEMON")
    print(f" Target End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Root Directory:  {ROOT}")
    print(f" Log File:        {HISTORY_FILE}")
    print(f"================================================================")

    # 1. Establish Initial Baseline
    print("\n[Step 1] Establishing initial champion baseline...")
    base_res = run_benchmark()
    champ_reward = base_res["mean_reward"]
    champ_survival = base_res["survival_rate"]
    print(f"Initial Champion: Mean Reward = {champ_reward:.4f}, Survival = {champ_survival:.1%}")

    iteration = 0
    accepted_count = 0
    rejected_count = 0
    start_time = datetime.now()

    while True:
        now = datetime.now()
        if now >= end_time:
            print(f"\n[Terminating] Reached scheduled stop time {end_time.strftime('%H:%M:%S')}.")
            break

        iteration += 1
        spec = random.choice(SEARCH_SPACE)
        curr_val = read_param_value(spec)

        # Perturbation: +/- 10% to 25% Gaussian noise or random range exploration
        if random.random() < 0.70:
            scale = random.uniform(0.10, 0.25)
            direction = 1.0 if random.random() < 0.5 else -1.0
            new_val = curr_val + direction * scale * (spec["max"] - spec["min"])
        else:
            new_val = random.uniform(spec["min"], spec["max"])

        new_val = max(spec["min"], min(spec["max"], new_val))
        if abs(new_val - curr_val) < 1e-4:
            continue

        print(f"\n--- [Iteration {iteration}] Testing Hypothesis on {spec['name']} ---")
        print(f"    Target: {spec['file']}")
        print(f"    Mutation: {curr_val} -> {spec['fmt'].format(new_val)}")

        # Apply mutation to disk
        write_param_value(spec, new_val)

        # Test verification gate
        test_passed = run_tests()
        if not test_passed:
            print("    [Gating 1 FAILED] Unit tests failed. Reverting...")
            git_revert()
            rejected_count += 1
            record = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "name": spec["name"],
                "file": spec["file"],
                "old": curr_val,
                "new": new_val,
                "status": "REJECTED_TESTS_FAILED",
                "delta_reward": 0.0,
            }
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            continue

        # Benchmark gate
        try:
            cand_res = run_benchmark()
        except Exception as e:
            print(f"    [Gating 2 FAILED] Benchmark crashed: {e}. Reverting...")
            git_revert()
            rejected_count += 1
            continue

        cand_reward = cand_res["mean_reward"]
        cand_survival = cand_res["survival_rate"]
        delta_rew = cand_reward - champ_reward
        delta_surv = cand_survival - champ_survival

        print(f"    Candidate: Reward={cand_reward:.4f} (Δ={delta_rew:+.4f}), Survival={cand_survival:.1%} (Δ={delta_surv:+.1%})")

        # Acceptance Criteria
        is_improved = False
        if cand_survival > champ_survival and delta_rew >= -0.05:
            is_improved = True
            reason = "Higher survival rate with non-catastrophic reward"
        elif cand_survival >= champ_survival and delta_rew > 0.005:
            is_improved = True
            reason = f"Positive paired reward gain (+{delta_rew:.4f})"

        if is_improved:
            print(f"    >>> [ACCEPTED IMPROVEMENT] {reason} <<<")
            champ_reward = cand_reward
            champ_survival = cand_survival
            accepted_count += 1

            # Commit improvement to git
            commit_msg = (
                f"auto-improve: [ACCEPTED #{accepted_count}] {spec['name']} "
                f"({curr_val} -> {spec['fmt'].format(new_val)}) ΔR={delta_rew:+.4f}"
            )
            subprocess.run(["git", "add", spec["file"]], cwd=ROOT, check=True)
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=ROOT, check=True)

            # Record receipt
            receipt_path = RECEIPTS_DIR / f"receipt_iter_{iteration:04d}.json"
            receipt_data = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "mutation": {"name": spec["name"], "file": spec["file"], "old": curr_val, "new": new_val},
                "champion": {"reward": champ_reward, "survival": champ_survival},
                "delta_reward": delta_rew,
            }
            receipt_path.write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")

            record = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "name": spec["name"],
                "file": spec["file"],
                "old": curr_val,
                "new": new_val,
                "status": "ACCEPTED",
                "delta_reward": delta_rew,
                "champ_reward": champ_reward,
                "champ_survival": champ_survival,
            }
        else:
            print("    [REJECTED] Performance did not exceed champion baseline. Reverting...")
            git_revert()
            rejected_count += 1
            record = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "name": spec["name"],
                "file": spec["file"],
                "old": curr_val,
                "new": new_val,
                "status": "REJECTED_NO_IMPROVEMENT",
                "delta_reward": delta_rew,
                "cand_reward": cand_reward,
                "cand_survival": cand_survival,
            }

        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        time.sleep(1)

    # Final Morning Summary Generation
    elapsed = (datetime.now() - start_time).total_seconds() / 3600
    report_md = f"""# Supermix Poseidon Overnight Auto-Improvement Report
- **Run Period:** {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({elapsed:.2f} hours)
- **Total Hypotheses Tested:** {iteration}
- **Accepted Improvements Committed:** {accepted_count}
- **Rejected Mutations (Clean Reverts):** {rejected_count}
- **Initial Champion Reward:** {base_res['mean_reward']:.4f} (Survival: {base_res['survival_rate']:.1%})
- **Final Champion Reward:** {champ_reward:.4f} (Survival: {champ_survival:.1%})
- **Net Reward Delta:** {champ_reward - base_res['mean_reward']:+.4f}

## Verification Receipts
Receipts stored in `outputs/auto_improvements/receipts/`.
History log recorded in `outputs/auto_improvements/history.jsonl`.
"""
    SUMMARY_FILE.write_text(report_md, encoding="utf-8")
    print(f"\n================================================================")
    print(f" OVERNIGHT RUN COMPLETED!")
    print(f" Report written to: {SUMMARY_FILE}")
    print(f"================================================================")


if __name__ == "__main__":
    main()
