from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from bankassist import bot
from guardrails.core import Context
from guardrails.pipeline import GuardrailPipeline
from guardrails.policy import load_policy

E = Path("evals")
LLM_CHECKS = ("hallucination", "schema")


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (E / name).read_text().splitlines() if line.strip()]


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


async def _noop_repair(text: str, error: str) -> str:
    return text


async def run_all(policy_path: str) -> dict:
    policy, h = load_policy(policy_path)
    pipe = GuardrailPipeline(policy, h)
    lat: dict[str, list[float]] = defaultdict(list)
    pos: dict[str, dict] = defaultdict(lambda: {"positives": 0, "caught": 0, "misses": []})
    fp: dict[str, dict] = defaultdict(lambda: {"benign": 0, "false_positives": 0})
    items: list[dict] = []
    in_over: list[float] = []
    out_local: list[float] = []
    out_llm: list[float] = []

    def score_positive(row: dict, verdicts) -> None:
        c = row["expected_check"]
        got = next((v for v in verdicts if v.check == c), None)
        ok = got is not None and got.action == row["expected_action"]
        pos[c]["positives"] += 1
        pos[c]["caught"] += ok
        if not ok:
            pos[c]["misses"].append(row["id"])
        for v in verdicts:
            lat[v.check].append(v.latency_ms)
        items.append({
            "id": row["id"], "expected_check": c, "expected_action": row["expected_action"],
            "got_action": got.action if got else None, "pass": ok,
        })

    for row in _rows("redteam_input.jsonl"):
        res = await pipe.run_input(row["text"], Context(policy_hash=h))
        in_over.append(res.overhead_ms)
        score_positive(row, res.verdicts)

    for row in _rows("benign.jsonl"):
        res = await pipe.run_input(row["text"], Context(policy_hash=h))
        in_over.append(res.overhead_ms)
        for v in res.verdicts:
            lat[v.check].append(v.latency_ms)
            fp[v.check]["benign"] += 1
            fp[v.check]["false_positives"] += v.action == "block"
        bad = [v.check for v in res.verdicts if v.action == "block"]
        items.append({
            "id": row["id"], "expected_check": None, "expected_action": "allow",
            "got_action": "block" if bad else "allow", "pass": not bad, "blocked_by": bad,
        })

    for row in _rows("redteam_output.jsonl"):
        use_schema = bool(row.get("schema_name"))
        ctx = Context(
            policy_hash=h, retrieved_chunks=row["context_chunks"],
            schema=bot.BotAnswer if use_schema else None,
            schema_fallback=bot.FALLBACK if use_schema else None,
            repair_fn=_noop_repair if use_schema else None,
        )
        t0 = time.perf_counter()
        res = await pipe.run_output(row["text"], ctx)
        out_llm.append((time.perf_counter() - t0) * 1000)
        out_local.append(sum(v.latency_ms for v in res.verdicts if v.check not in LLM_CHECKS))
        score_positive(row, res.verdicts)

    checks = {}
    for name in sorted(set(pos) | set(fp) | set(lat)):
        p, f = pos[name], fp[name]
        checks[name] = {
            **p,
            **f,
            "catch_rate": p["caught"] / p["positives"] if p["positives"] else None,
            "false_positive_rate": f["false_positives"] / f["benign"] if f["benign"] else None,
            "p50_ms": round(_pct(lat[name], 0.5), 2),
            "p99_ms": round(_pct(lat[name], 0.99), 2),
        }
    total_pos = sum(p["positives"] for p in pos.values())
    total_caught = sum(p["caught"] for p in pos.values())
    return {
        "policy": {"path": policy_path, "version": policy.version, "hash": h},
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cassette_mode": os.getenv("LLM_CASSETTE_MODE", "off"),
        "checks": checks,
        "overall": {
            "catch_rate": total_caught / total_pos if total_pos else None,
            "input_overhead_p50_ms": round(_pct(in_over, 0.5), 1),
            "input_overhead_p99_ms": round(_pct(in_over, 0.99), 1),
            "output_overhead_local_p50_ms": round(_pct(out_local, 0.5), 1),
            "output_overhead_local_p99_ms": round(_pct(out_local, 0.99), 1),
            "output_overhead_with_llm_p50_ms": round(_pct(out_llm, 0.5), 1),
            "output_overhead_with_llm_p99_ms": round(_pct(out_llm, 0.99), 1),
        },
        "items": items,
    }


def summarize(results: dict) -> str:
    lines = [
        "| check | catch rate | false-positive rate | p50 ms | p99 ms | misses |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in results["checks"].items():
        cr = c.get("catch_rate")
        fr = c.get("false_positive_rate")
        cr_s = "n/a" if cr is None else f"{cr:.1%} ({c['caught']}/{c['positives']})"
        fr_s = "n/a" if fr is None else f"{fr:.1%} ({c['false_positives']}/{c['benign']})"
        misses = ", ".join(c.get("misses", [])[:5])
        lines.append(f"| {name} | {cr_s} | {fr_s} | {c['p50_ms']} | {c['p99_ms']} | {misses} |")
    o = results["overall"]
    lines.append("")
    lines.append(
        f"Overall catch rate: {o['catch_rate']:.1%}. Input overhead p50/p99: "
        f"{o.get('input_overhead_p50_ms')}/{o.get('input_overhead_p99_ms')} ms."
    )
    return "\n".join(lines)


def gate(results: dict, thresholds: dict, baseline: dict | None) -> list[str]:
    msgs = []
    default = thresholds.get("default", {})
    for name, c in results["checks"].items():
        t = {**default, **thresholds.get("checks", {}).get(name, {})}
        cr, fr = c.get("catch_rate"), c.get("false_positive_rate")
        if cr is not None and cr < t.get("min_catch_rate", 0):
            msgs.append(f"{name}: catch_rate {cr:.3f} < {t['min_catch_rate']}")
        if fr is not None and fr > t.get("max_false_positive_rate", 1):
            msgs.append(f"{name}: false_positive_rate {fr:.3f} > {t['max_false_positive_rate']}")
    base = (baseline or {}).get("overall", {}).get("catch_rate")
    if base is not None and results["overall"]["catch_rate"] is not None:
        drop = base - results["overall"]["catch_rate"]
        if drop > thresholds.get("max_catch_rate_drop_vs_main", 1):
            msgs.insert(
                0,
                f"overall catch_rate regressed by {drop:.3f} vs main "
                f"({base:.3f} -> {results['overall']['catch_rate']:.3f})",
            )
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.getenv("GUARDRAILS_POLICY", "policies/v1.yaml"))
    ap.add_argument("--out", default="evals/results/latest.json")
    ap.add_argument("--baseline", default="evals/results/main.json")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    if a.record:
        os.environ["LLM_CASSETTE_MODE"] = "record"
    elif a.replay:
        os.environ["LLM_CASSETTE_MODE"] = "replay"

    results = asyncio.run(run_all(a.policy))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(summarize(results))
    if a.no_gate:
        return 0
    thresholds = yaml.safe_load((E / "thresholds.yaml").read_text())
    baseline = json.loads(Path(a.baseline).read_text()) if Path(a.baseline).exists() else None
    failures = gate(results, thresholds, baseline)
    if failures:
        print("\nEVAL GATE FAILED:\n- " + "\n- ".join(failures), file=sys.stderr)
        return 1
    print("\nEVAL GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
