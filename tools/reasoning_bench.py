"""Ground-truth reasoning benchmark for Poseidon's symbolic tool stack.

Each case is a natural-language (or semi-formal) prompt with a known correct answer.
The router mirrors ``Poseidon.respond(mode="chat")``: exact maths first, then the
Problem Forge. A case scores only if a tool fires AND its answer contains every
expected token (whitespace/case-insensitive). Unanswered cases are reported
separately from wrong answers so routing gaps and solver errors stay distinguishable.

Usage: python tools/reasoning_bench.py [--json] [--failures]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (id, category, prompt, expected tokens that must all appear in the normalized answer)
CASES = [
    # --- Formal maths (already-supported syntax) ---
    ("m01", "formal_math", "(17 + 5) * 3", ["66"]),
    ("m02", "formal_math", "3*x + 7 = 22", ["5"]),
    ("m03", "formal_math", "x^2 - 5*x + 6 = 0", ["2", "3"]),
    ("m04", "formal_math", "x^3 - 6*x^2 + 11*x - 6 = 0", ["1", "2", "3"]),
    ("m05", "formal_math", "det([[1, 2], [3, 4]])", ["-2"]),
    ("m06", "formal_math", "factor(360)", ["2^3", "3^2", "5"]),
    ("m07", "formal_math", "totient(36)", ["12"]),
    ("m08", "formal_math", "modinv(3, 11)", ["4"]),
    ("m09", "formal_math", "integrate(3*x^2 + 2*x, x, 0, 2)", ["12"]),
    ("m10", "formal_math", "diff(x^3 - 4*x, x, at=2)", ["8"]),
    # --- Natural-language maths (how people actually ask) ---
    ("n01", "nl_math", "If 2x + 3 = 11, what is x?", ["4"]),
    ("n02", "nl_math", "What is 15% of 240?", ["36"]),
    ("n03", "nl_math", "What is the derivative of x^3 - 4x?", ["3*x^2", "-4"]),
    ("n04", "nl_math", "Find the integral of 2x from 0 to 3", ["9"]),
    ("n05", "nl_math", "Solve x + y = 10 and x - y = 2", ["6", "4"]),
    ("n06", "nl_math", "What is the greatest common divisor of 84 and 36?", ["12"]),
    ("n07", "nl_math", "Factorize 1001", ["7", "11", "13"]),
    ("n08", "nl_math", "What is the determinant of [[2, 1], [5, 3]]?", ["1"]),
    ("n09", "nl_math", "What is 2 to the power of 10?", ["1024"]),
    ("n10", "nl_math", "What is the square root of 144?", ["12"]),
    ("n11", "nl_math", "Calculate 7 times 8 plus 6", ["62"]),
    ("n12", "nl_math", "What is the least common multiple of 4, 6 and 10?", ["60"]),
    # --- Problem Forge domains ---
    ("p01", "problem", "Tower of Hanoi with 3 disks", ["7"]),
    ("p02", "problem", "Solve the cryptarithm SEND + MORE = MONEY", ["9567", "1085", "10652"]),
    ("p03", "problem", "What is the edit distance between 'kitten' and 'sitting'?", ["3"]),
    ("p04", "problem", "Find the Nash equilibrium of the prisoner's dilemma", ["defect"]),
    ("p05", "problem", "A disease has prevalence 1%, test sensitivity 99% and false positive rate 5%. Using Bayes' theorem, what is the posterior probability of disease after a positive test?", ["1/6"]),
    ("p06", "problem", "Run Grover's algorithm for N=1024 items with M=1 marked state", ["25"]),
    ("p07", "problem", "Shannon-Hartley channel capacity for bandwidth B = 1 MHz and SNR = 20 dB", ["6.65"]),
    ("p08", "problem", "Hawk-Dove evolutionarily stable strategy with V=4 and C=6", ["2/3"]),
    ("p09", "problem", "Solve the 6-queens problem", ["["]),
    ("p10", "problem", "Kalman filter gain for A=1.0, H=1.0, Q=0.1, R=0.5", ["0.358"]),
    ("p11", "problem", "Is the CNF formula (A or B) and (not A) and (not B) satisfiable?", ["unsatisfiable"]),
    ("p12", "problem", "Make 24 from the numbers 4, 7, 8, 8", ["24"]),
]


def _norm(text: str) -> str:
    return "".join(str(text).lower().split())


def route(prompt: str):
    """Same precedence as runtime chat routing: exact maths, then Problem Forge."""
    from poseidon.reasoning import extract_math, solve
    from poseidon.problem_solver import detect_problem, solve_problem

    expr = extract_math(prompt)
    if expr:
        try:
            return "math", solve(expr)
        except Exception as error:  # solver refused: report, don't crash the bench
            return "math_error", {"answer": "", "text": f"{type(error).__name__}: {error}"}
    domain = detect_problem(prompt)
    if domain is not None:
        try:
            return f"problem:{domain}", solve_problem(prompt, domain=domain)
        except Exception as error:
            return "problem_error", {"answer": "", "text": f"{type(error).__name__}: {error}"}
    return None, None


def run(cases=CASES):
    rows = []
    for case_id, category, prompt, expected in cases:
        t0 = time.perf_counter()
        tool, result = route(prompt)
        ms = (time.perf_counter() - t0) * 1000.0
        answer = "" if result is None else f"{result.get('answer', '')} {result.get('text', '')}"
        if tool is None:
            status = "unanswered"
        elif all(_norm(tok) in _norm(answer) for tok in expected):
            status = "correct"
        else:
            status = "wrong"
        rows.append({
            "id": case_id, "category": category, "prompt": prompt, "tool": tool,
            "status": status, "expected": expected,
            "answer": (result or {}).get("answer"), "ms": round(ms, 1),
        })
    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"total": 0, "correct": 0, "wrong": 0, "unanswered": 0})
        c["total"] += 1
        c[r["status"]] += 1
    total = len(rows)
    correct = sum(r["status"] == "correct" for r in rows)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "wrong": sum(r["status"] == "wrong" for r in rows),
        "unanswered": sum(r["status"] == "unanswered" for r in rows),
        "by_category": by_cat,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--failures", action="store_true")
    args = ap.parse_args()
    report = run()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    print(f"Accuracy: {report['correct']}/{report['total']} = {report['accuracy']:.1%}  "
          f"(wrong {report['wrong']}, unanswered {report['unanswered']})")
    for cat, c in report["by_category"].items():
        print(f"  {cat:12s} {c['correct']}/{c['total']}  wrong={c['wrong']} unanswered={c['unanswered']}")
    if args.failures:
        for r in report["rows"]:
            if r["status"] != "correct":
                print(f"[{r['status'].upper():10s}] {r['id']} tool={r['tool']} expected={r['expected']} "
                      f"got={str(r['answer'])[:90]!r}\n             prompt={r['prompt']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
