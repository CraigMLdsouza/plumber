"""
Plumber Runtime Integration Tests

Run:
    python test.py

Make sure:
    export GEMINI_API_KEY=your_key
    behaviors/ folder exists
"""

import json
import traceback

from app import (
    run_behavior,
    lint_behavior,
    InputContractError,
    OutputContractError,
    ValidationRuleError,
    BehaviorDefinitionError,
)

# ─────────────────────────────────────────────────────────────
# Test Runner Utilities
# ─────────────────────────────────────────────────────────────

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_test(name, fn, expect_error=None):
    print(f"\n[Test] {name}")

    try:
        fn()

        if expect_error:
            print(f"→ FAIL (expected {expect_error.__name__}, but no error raised)")
        else:
            print("→ PASS")

    except Exception as e:
        if expect_error and isinstance(e, expect_error):
            print(f"→ PASS (caught expected {expect_error.__name__})")
        else:
            print("→ FAIL")
            print(f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────

def test_lint_behavior():
    """Behavior definition should validate without calling model."""
    lint_behavior("summarize_ticket")


def test_valid_execution():
    """Happy path execution."""
    data = {"ticket_text": "User cannot log in during checkout."}
    result, trace = run_behavior("summarize_ticket", data)

    print("\nResult:")
    print(json.dumps(result, indent=2))

    print("\nTrace:")
    print(json.dumps(trace, indent=2))

    assert "summary" in result
    assert "priority" in result
    assert trace["validated"] is True


def test_missing_input():
    """Missing required input must be rejected."""
    data = {}
    run_behavior("summarize_ticket", data)


def test_extra_input():
    """Undeclared fields must be rejected."""
    data = {
        "ticket_text": "Login failure",
        "unexpected": "should not be allowed"
    }
    run_behavior("summarize_ticket", data)


def test_invalid_behavior_reference():
    """Referencing non-existent behavior must fail."""
    lint_behavior("does_not_exist")


def test_versioned_behavior_if_present():
    """
    If summarize_ticket.v2.yaml exists, run it.
    Safe to skip if not present.
    """
    try:
        lint_behavior("summarize_ticket.v2")

        data = {"ticket_text": "Checkout login blocked"}
        result, trace = run_behavior("summarize_ticket.v2", data)

        print("\n[v2 Result]")
        print(json.dumps(result, indent=2))

        assert trace["behavior"] == "summarize_ticket.v2"

    except BehaviorDefinitionError:
        print("→ Skipping (no v2 behavior defined)")


# ─────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_section("Plumber Runtime Integration Tests")

    run_test("Lint Behavior", test_lint_behavior)
    run_test("Valid Execution", test_valid_execution)

    run_test("Reject Missing Input", test_missing_input, expect_error=InputContractError)
    run_test("Reject Extra Input", test_extra_input, expect_error=InputContractError)

    run_test("Reject Invalid Behavior", test_invalid_behavior_reference, expect_error=BehaviorDefinitionError)

    run_test("Versioned Behavior (Optional)", test_versioned_behavior_if_present)

    print("\nAll tests completed.")