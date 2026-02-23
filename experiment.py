"""
Plumber Runtime — Real-World Experiment Harness
=================================================
Runs live Gemini calls against real behavior YAMLs and validates that the
full pipeline — lint → model call → contract enforcement → trace — works
end-to-end on authentic inputs.

Prerequisites:
    export GEMINI_API_KEY=your_key_here

    behaviors/ folder must contain:
        summarize_ticket.yaml
        summarize_ticket.v2.yaml
        classify_incident.v1.yaml
        extract_meeting.v1.yaml

Run:
    python experiment.py                # all experiments
    python experiment.py ticket         # ticket summarisation only
    python experiment.py incident       # incident classification only
    python experiment.py meeting        # meeting extraction only
    python experiment.py version        # version evolution comparison only
    python experiment.py lint           # lint + diff only (no model calls)
    python experiment.py error          # contract violation paths only
    python experiment.py determinism    # determinism spot-check only
"""

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import (
    run_behavior,
    lint_behavior,
    diff_behaviors,
    BehaviorDefinitionError,
    InputContractError,
    OutputContractError,
    ValidationRuleError,
    ModelExecutionError,
)

# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

BOLD   = "\033[1m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"
RESET  = "\033[0m"

WIDTH = 68


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'━' * WIDTH}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'━' * WIDTH}{RESET}")


def subheader(title: str) -> None:
    print(f"\n{BOLD}  ── {title}{RESET}")


def show_result(result: dict, trace: dict) -> None:
    print(f"\n  {BOLD}Result:{RESET}")
    for k, v in result.items():
        print(f"    {YELLOW}{k}{RESET}: {repr(v)}")
    print(f"\n  {DIM}Trace:{RESET}")
    print(f"  {DIM}  behavior      : {trace['behavior']}{RESET}")
    print(f"  {DIM}  behavior_hash : {trace['behavior_hash'][:16]}…{RESET}")
    print(f"  {DIM}  model         : {trace['model']}{RESET}")
    print(f"  {DIM}  latency_ms    : {trace['latency_ms']}{RESET}")
    print(f"  {DIM}  validated     : {trace['validated']}{RESET}")


def passed(msg: str) -> None:
    print(f"  {GREEN}{BOLD}✓{RESET}  {msg}")


def failed(msg: str) -> None:
    print(f"  {RED}{BOLD}✗{RESET}  {msg}")


def caught(error_type: str, msg: str) -> None:
    print(f"  {YELLOW}{BOLD}⚑{RESET}  [{error_type}] caught as expected")
    print(f"  {DIM}  {str(msg)[:120]}{RESET}")


def assert_field(result: dict, field: str, allowed: list = None,
                 required_type=None) -> bool:
    if field not in result:
        failed(f"Field '{field}' missing from result")
        return False
    val = result[field]
    if required_type is not None and not isinstance(val, required_type):
        failed(f"'{field}' wrong type: expected {required_type.__name__}, "
               f"got {type(val).__name__}")
        return False
    if allowed is not None and val not in allowed:
        failed(f"'{field}' = {repr(val)}, not in allowed set {allowed}")
        return False
    passed(f"'{field}' = {repr(val)}")
    return True


def assert_trace(trace: dict, behavior: str) -> bool:
    ok = True
    if trace.get("behavior") != behavior:
        failed(f"trace.behavior = {trace.get('behavior')!r}, expected {behavior!r}")
        ok = False
    if not isinstance(trace.get("behavior_hash"), str) or \
            len(trace["behavior_hash"]) != 64:
        failed("trace.behavior_hash missing or wrong length")
        ok = False
    if not isinstance(trace.get("latency_ms"), int) or trace["latency_ms"] <= 0:
        failed(f"trace.latency_ms invalid: {trace.get('latency_ms')!r}")
        ok = False
    if trace.get("validated") is not True:
        failed("trace.validated is not True")
        ok = False
    if ok:
        passed(
            f"trace ok  [hash={trace['behavior_hash'][:12]}…  "
            f"{trace['latency_ms']}ms]"
        )
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Ticket Summarisation
# ─────────────────────────────────────────────────────────────────────────────

def experiment_ticket():
    header("Experiment 1 — Ticket Summarisation  (summarize_ticket)")

    cases = [
        (
            "Clear login failure",
            "User reports they cannot log in during checkout. "
            "Error message says 'invalid credentials' but password was reset yesterday.",
        ),
        (
            "Vague performance complaint",
            "The app feels slow sometimes, not sure which page. "
            "Started a couple of days ago.",
        ),
        (
            "Critical data loss",
            "Customer exported their data and the CSV is empty. "
            "This is blocking their finance team from month-end close. Urgent.",
        ),
        (
            "Feature request disguised as a bug",
            "The dashboard doesn't show last 90 days of data, only 30. "
            "We need more history for compliance reporting.",
        ),
    ]

    behavior = "summarize_ticket"
    hashes = set()

    for label, ticket_text in cases:
        subheader(label)
        print(f"  {DIM}Input: {ticket_text[:80]}…{RESET}")
        try:
            result, trace = run_behavior(behavior, {"ticket_text": ticket_text})
            show_result(result, trace)

            assert_field(result, "summary", required_type=str)
            assert_field(result, "priority", allowed=["low", "medium", "high"])
            assert_trace(trace, behavior)

            # max_length=120 validation rule
            if len(result["summary"]) <= 120:
                passed(f"summary length ok ({len(result['summary'])} ≤ 120 chars)")
            else:
                failed(f"summary too long: {len(result['summary'])} chars")

            # normalization: strip
            if result["summary"] == result["summary"].strip():
                passed("summary has no surrounding whitespace (strip applied)")
            else:
                failed("summary has surrounding whitespace — normalization not applied")

            hashes.add(trace["behavior_hash"])

        except Exception as e:
            failed(f"Unexpected exception: {type(e).__name__}: {e}")
            traceback.print_exc()

    subheader("Behavior hash stability across all runs")
    if len(hashes) == 1:
        passed(f"behavior_hash stable across {len(cases)} runs: {list(hashes)[0][:16]}…")
    else:
        failed(f"behavior_hash changed mid-run — file was modified: {hashes}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Incident Classification
# ─────────────────────────────────────────────────────────────────────────────

def experiment_incident():
    header("Experiment 2 — Incident Classification  (classify_incident.v1)")

    behavior = "classify_incident.v1"

    VALID_CATEGORIES = ["outage", "degradation", "security", "billing", "other"]
    VALID_SEVERITIES = ["sev1", "sev2", "sev3", "sev4"]

    cases = [
        (
            "Full EU payment outage",
            "All payment processing is down across the EU region. "
            "No transactions completing. Revenue impact ongoing.",
            "outage",
        ),
        (
            "Elevated API latency — system still up",
            "API p99 latency increased from 120ms to 850ms. "
            "System is operational but degraded.",
            "degradation",
        ),
        (
            "Credential stuffing attack",
            "Automated alerts firing for 400+ failed login attempts "
            "from a single IP range in the past 10 minutes.",
            "security",
        ),
        (
            "Wrong currency on invoice",
            "Customer in Canada sees USD on invoice instead of CAD. "
            "Minor display issue, payments processing correctly.",
            "billing",
        ),
        (
            "Low-priority UI glitch",
            "The scheduled maintenance notification banner did not appear "
            "on the dashboard today. No customer impact.",
            "other",
        ),
    ]

    for label, description, expected_category in cases:
        subheader(label)
        print(f"  {DIM}Input: {description[:80]}…{RESET}")
        print(f"  {DIM}Expected category: {expected_category}{RESET}")
        try:
            result, trace = run_behavior(behavior, {"description": description})
            show_result(result, trace)

            assert_field(result, "category", allowed=VALID_CATEGORIES)
            assert_field(result, "severity", allowed=VALID_SEVERITIES)
            assert_trace(trace, behavior)

            # normalization: lowercase applied to both enum fields
            for field in ("category", "severity"):
                val = result.get(field, "")
                if val == val.lower():
                    passed(f"'{field}' is lowercase (normalization applied)")
                else:
                    failed(f"'{field}' not lowercased: {val!r}")

            # Soft assertion — model judgment may legitimately vary on severity
            if result["category"] == expected_category:
                passed(f"category matches expected: {expected_category!r}")
            else:
                print(
                    f"  {YELLOW}⚠{RESET}  category={result['category']!r} differs from "
                    f"expected={expected_category!r} — model judgment may vary"
                )

        except Exception as e:
            failed(f"Unexpected exception: {type(e).__name__}: {e}")
            traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 — Meeting Extraction
# ─────────────────────────────────────────────────────────────────────────────

def experiment_meeting():
    header("Experiment 3 — Structured Meeting Extraction  (extract_meeting.v1)")

    behavior = "extract_meeting.v1"

    cases = [
        (
            "Explicit all-fields email",
            "Hi team, let's meet on Tuesday March 5th at 2:30pm with "
            "Alice, Bob, and Carol to discuss the Q2 roadmap planning.",
        ),
        (
            "Ambiguous timing",
            "Can we sync tomorrow afternoon? I'm thinking 3 or 4pm. "
            "Jordan and Sam should join — topic is the API migration.",
        ),
        (
            "Sparse details",
            "Let's find some time this week to talk about onboarding.",
        ),
        (
            "Complex multi-sentence email",
            "Thanks for the intro, Marcus. I'd love to connect with your team. "
            "How about Wednesday at 10am? We could cover the integration design "
            "and get Kim and Priya on the call too.",
        ),
    ]

    for label, email_text in cases:
        subheader(label)
        print(f"  {DIM}Input: {email_text[:80]}…{RESET}")
        try:
            result, trace = run_behavior(behavior, {"email_text": email_text})
            show_result(result, trace)

            for field in ("date", "time", "participants", "topic"):
                assert_field(result, field, required_type=str)

            # validation: max_length topic=80
            topic_len = len(result.get("topic", ""))
            if topic_len <= 80:
                passed(f"topic length ok ({topic_len} ≤ 80 chars)")
            else:
                failed(f"topic too long: {topic_len} chars")

            # normalization: strip all four fields
            for field in ("date", "time", "participants", "topic"):
                val = result.get(field, "")
                if isinstance(val, str) and val == val.strip():
                    passed(f"'{field}' is stripped")
                elif isinstance(val, str):
                    failed(f"'{field}' has surrounding whitespace: {val!r}")

            assert_trace(trace, behavior)

        except Exception as e:
            failed(f"Unexpected exception: {type(e).__name__}: {e}")
            traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4 — Version Evolution (summarize_ticket v1 → v2)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_versioning():
    header("Experiment 4 — Version Evolution  (summarize_ticket v1 ↔ v2)")

    payload = {
        "ticket_text": (
            "User cannot access the billing portal after a password reset. "
            "Affects 3 enterprise accounts. No known workaround."
        )
    }

    trace_v1 = trace_v2 = None

    subheader("Run v1 (summarize_ticket)")
    try:
        result_v1, trace_v1 = run_behavior("summarize_ticket", payload)
        show_result(result_v1, trace_v1)

        assert_field(result_v1, "summary", required_type=str)
        assert_field(result_v1, "priority", allowed=["low", "medium", "high"])
        assert_trace(trace_v1, "summarize_ticket")

        if "category" not in result_v1:
            passed("v1 correctly has no 'category' field")
        else:
            failed(f"v1 unexpectedly returned 'category': {result_v1['category']!r}")

    except Exception as e:
        failed(f"v1 failed: {type(e).__name__}: {e}")
        traceback.print_exc()

    subheader("Run v2 (summarize_ticket.v2)")
    try:
        result_v2, trace_v2 = run_behavior("summarize_ticket.v2", payload)
        show_result(result_v2, trace_v2)

        assert_field(result_v2, "summary", required_type=str)
        assert_field(result_v2, "priority", allowed=["low", "medium", "high"])
        assert_trace(trace_v2, "summarize_ticket.v2")

        if "category" in result_v2:
            passed(f"v2 correctly includes 'category': {result_v2['category']!r}")
        else:
            failed("v2 missing expected 'category' field")

    except Exception as e:
        failed(f"v2 failed: {type(e).__name__}: {e}")
        traceback.print_exc()

    subheader("Hash comparison (v1 ≠ v2)")
    if trace_v1 and trace_v2:
        h1 = trace_v1["behavior_hash"]
        h2 = trace_v2["behavior_hash"]
        if h1 != h2:
            passed("v1 hash ≠ v2 hash — distinct contracts correctly fingerprinted")
            print(f"  {DIM}  v1: {h1[:32]}…{RESET}")
            print(f"  {DIM}  v2: {h2[:32]}…{RESET}")
        else:
            failed("v1 and v2 share the same hash — YAML files appear identical")

    subheader("Schema diff output (v1 → v2)")
    try:
        diff_behaviors("summarize_ticket", "summarize_ticket.v2")
        passed("diff produced output without raising")
    except Exception as e:
        failed(f"diff raised: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5 — Lint + Diff (no model calls)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_lint():
    header("Experiment 5 — Lint + Diff  (no model calls)")

    behaviors_to_lint = [
        "summarize_ticket",
        "summarize_ticket.v2",
        "classify_incident.v1",
        "extract_meeting.v1",
    ]

    subheader("Lint every known behavior")
    for name in behaviors_to_lint:
        try:
            lint_behavior(name)
            passed(f"lint passed: {name}")
        except BehaviorDefinitionError as e:
            failed(f"lint failed: {name}\n    {e}")
        except Exception as e:
            failed(f"unexpected error linting {name}: {type(e).__name__}: {e}")

    subheader("Diff: summarize_ticket v1 → v2")
    try:
        diff_behaviors("summarize_ticket", "summarize_ticket.v2")
        passed("diff completed without errors")
    except Exception as e:
        failed(f"diff raised: {type(e).__name__}: {e}")

    subheader("Diff: same file against itself (should show no changes)")
    try:
        diff_behaviors("summarize_ticket", "summarize_ticket")
        passed("diff of identical files completed")
    except Exception as e:
        failed(f"diff raised: {type(e).__name__}: {e}")

    subheader("Lint a non-existent behavior (expected failure)")
    try:
        lint_behavior("does_not_exist")
        failed("Expected BehaviorDefinitionError — none raised")
    except BehaviorDefinitionError as e:
        caught("BehaviorDefinitionError", e)
        passed("missing behavior correctly rejected")
    except Exception as e:
        failed(f"Wrong exception type: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 6 — Contract Violation Paths
# ─────────────────────────────────────────────────────────────────────────────

def experiment_error_paths():
    header("Experiment 6 — Contract Enforcement  (deliberate violations)")

    # These all reject before reaching the model — no API cost
    subheader("Missing required input field → InputContractError")
    try:
        run_behavior("summarize_ticket", {})
        failed("Expected InputContractError — none raised")
    except InputContractError as e:
        caught("InputContractError", e)
        passed("correctly rejected before model call")
    except Exception as e:
        failed(f"Wrong exception: {type(e).__name__}: {e}")

    subheader("Extra undeclared input field → InputContractError")
    try:
        run_behavior("summarize_ticket", {
            "ticket_text": "Login broken",
            "injected": "extra field",
        })
        failed("Expected InputContractError — none raised")
    except InputContractError as e:
        caught("InputContractError", e)
        passed("extra input field correctly rejected")
    except Exception as e:
        failed(f"Wrong exception: {type(e).__name__}: {e}")

    subheader("Input field with wrong type (int instead of string) → InputContractError")
    try:
        run_behavior("summarize_ticket", {"ticket_text": 99999})
        failed("Expected InputContractError — none raised")
    except InputContractError as e:
        caught("InputContractError", e)
        passed("wrong-type input correctly rejected")
    except Exception as e:
        failed(f"Wrong exception: {type(e).__name__}: {e}")

    subheader("Behavior file does not exist → BehaviorDefinitionError")
    try:
        run_behavior("ghost_behavior", {"text": "irrelevant"})
        failed("Expected BehaviorDefinitionError — none raised")
    except BehaviorDefinitionError as e:
        caught("BehaviorDefinitionError", e)
        passed("missing behavior file correctly rejected")
    except Exception as e:
        failed(f"Wrong exception: {type(e).__name__}: {e}")

    # Live model call — verify max_length rule fires or model self-complies
    subheader("max_length enforcement via live model call (summarize_ticket)")
    try:
        result, trace = run_behavior(
            "summarize_ticket",
            {"ticket_text": (
                "Critical multi-region failure affecting all enterprise customers. "
                "Authentication service, payment gateway, data pipeline, notification "
                "system, and admin portal are all simultaneously down due to an upstream "
                "infrastructure provider incident. Estimated 50,000 users impacted."
            )},
        )
        summary_len = len(result.get("summary", ""))
        if summary_len <= 120:
            passed(f"Model self-complied with max_length=120 (summary={summary_len} chars)")
        else:
            failed(f"Summary exceeded 120 chars ({summary_len}) — validation gap")
    except ValidationRuleError as e:
        caught("ValidationRuleError", e)
        passed("ValidationRuleError fired correctly — max_length enforced")
    except Exception as e:
        failed(f"Unexpected exception: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 7 — Determinism Spot-Check
# ─────────────────────────────────────────────────────────────────────────────

def experiment_determinism():
    header("Experiment 7 — Determinism Spot-Check  (temperature=0.0)")

    # classify_incident.v1 uses temperature=0.0 — expect identical results
    payload = {
        "description": (
            "Payment gateway returning 503 errors for all EU transactions. "
            "Confirmed by multiple customers. Ongoing for 8 minutes."
        )
    }
    behavior = "classify_incident.v1"

    subheader("Running same input twice at temperature=0.0")
    results = []
    for i in range(2):
        try:
            result, trace = run_behavior(behavior, payload)
            results.append(result)
            passed(
                f"Run {i+1}: category={result['category']!r}  "
                f"severity={result['severity']!r}"
            )
        except Exception as e:
            failed(f"Run {i+1} failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    if len(results) == 2:
        if results[0] == results[1]:
            passed("Both runs returned identical output ✓ deterministic at temp=0.0")
        else:
            print(
                f"  {YELLOW}⚠{RESET}  Outputs differ across two runs at temperature=0.0\n"
                f"  {DIM}  Run 1: {results[0]}{RESET}\n"
                f"  {DIM}  Run 2: {results[1]}{RESET}\n"
                f"  {DIM}  Minor non-determinism in LLM APIs can occur even at temp=0.{RESET}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "ticket":      experiment_ticket,
    "incident":    experiment_incident,
    "meeting":     experiment_meeting,
    "version":     experiment_versioning,
    "lint":        experiment_lint,
    "error":       experiment_error_paths,
    "determinism": experiment_determinism,
}

if __name__ == "__main__":
    filter_arg = sys.argv[1].lower() if len(sys.argv) > 1 else None

    if filter_arg and filter_arg not in EXPERIMENTS:
        print(f"{RED}Unknown experiment: {filter_arg!r}{RESET}")
        print(f"Available: {', '.join(EXPERIMENTS)}")
        sys.exit(1)

    t0 = time.time()
    selected = (
        {filter_arg: EXPERIMENTS[filter_arg]}
        if filter_arg
        else EXPERIMENTS
    )

    print(f"\n{BOLD}{'━' * WIDTH}{RESET}")
    print(f"{BOLD}  Plumber Runtime — Experiment Harness{RESET}")
    print(f"{BOLD}  Running: {', '.join(selected)}{RESET}")
    print(f"{BOLD}{'━' * WIDTH}{RESET}")

    for name, fn in selected.items():
        try:
            fn()
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Interrupted.{RESET}")
            sys.exit(0)
        except Exception as e:
            print(f"\n{RED}{BOLD}Experiment '{name}' crashed:{RESET} {e}")
            traceback.print_exc()

    elapsed = int((time.time() - t0) * 1000)
    print(f"\n{BOLD}{'━' * WIDTH}{RESET}")
    print(f"{BOLD}  All experiments complete  [{elapsed}ms total]{RESET}")
    print(f"{BOLD}{'━' * WIDTH}{RESET}\n")
