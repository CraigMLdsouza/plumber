"""
Plumber Runtime — Unit Test Suite
===================================
Tests every internal function without making any model calls.
Temporary YAML files are written to a temp directory and cleaned up automatically.

Run:
    python test.py
    python test.py -v          # verbose: show each test name as it runs
    python test.py ClassName   # run one group, e.g. python test.py TestLintBehavior

No external dependencies beyond those already required by app.py.
No GEMINI_API_KEY needed.
"""

import os
import sys
import json
import unittest
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

# ── Make sure app.py is importable from the same directory ───────────────────
sys.path.insert(0, str(Path(__file__).parent))

from app import (
    # helpers
    _get_field_type,
    _is_optional,
    _validate_scalar_value,
    # loading / prompts
    load_behavior,
    build_system_prompt,
    build_user_prompt,
    parse_output,
    # validation pipeline
    validate_input,
    validate_output,
    apply_missing_and_defaults,
    apply_normalization,
    apply_validation_rules,
    # linting
    lint_behavior,
    # error types
    BehaviorDefinitionError,
    InputContractError,
    OutputContractError,
    ValidationRuleError,
    ModelExecutionError,
)

# ─────────────────────────────────────────────────────────────────────────────
# Test Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
RESET = "\033[0m"


def section(title: str) -> None:
    width = 60
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


class PlumberTestCase(unittest.TestCase):
    """Base class — provides a temp behaviors/ directory for each test."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._behaviors_path = Path(self._tmpdir.name) / "behaviors"
        self._behaviors_path.mkdir()
        # Patch Path("behaviors") to point to our temp dir inside load_behavior
        self._patcher = patch(
            "app.Path",
            side_effect=lambda *args: (
                self._behaviors_path if args == ("behaviors",)
                else Path(*args)
            ),
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def write_behavior(self, name: str, content: str) -> None:
        (self._behaviors_path / f"{name}.yaml").write_text(textwrap.dedent(content))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema Helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFieldType(unittest.TestCase):

    def test_bare_string_type(self):
        self.assertEqual(_get_field_type("string"), "string")

    def test_bare_integer_type(self):
        self.assertEqual(_get_field_type("integer"), "integer")

    def test_dict_with_type_key(self):
        self.assertEqual(_get_field_type({"type": "enum", "values": ["a"]}), "enum")

    def test_dict_without_type_key_defaults_to_string(self):
        self.assertEqual(_get_field_type({"optional": True}), "string")

    def test_dict_array_type(self):
        self.assertEqual(_get_field_type({"type": "array", "items": "float"}), "array")

    def test_none_defaults_to_string(self):
        self.assertEqual(_get_field_type(None), "string")


class TestIsOptional(unittest.TestCase):

    def test_bare_string_is_not_optional(self):
        self.assertFalse(_is_optional("string"))

    def test_dict_optional_true(self):
        self.assertTrue(_is_optional({"type": "string", "optional": True}))

    def test_dict_optional_false(self):
        self.assertFalse(_is_optional({"type": "string", "optional": False}))

    def test_dict_no_optional_key(self):
        self.assertFalse(_is_optional({"type": "string"}))

    def test_none_is_not_optional(self):
        self.assertFalse(_is_optional(None))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scalar Type Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateScalarValue(unittest.TestCase):

    # ── string ──────────────────────────────────────────────
    def test_string_accepts_str(self):
        _validate_scalar_value("f", "hello", "string")  # no raise

    def test_string_rejects_int(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", 42, "string")

    def test_string_rejects_none(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", None, "string")

    # ── integer ─────────────────────────────────────────────
    def test_integer_accepts_int(self):
        _validate_scalar_value("f", 5, "integer")

    def test_integer_rejects_bool_true(self):
        # bool is a subclass of int — must be explicitly rejected
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", True, "integer")

    def test_integer_rejects_bool_false(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", False, "integer")

    def test_integer_rejects_float(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", 5.0, "integer")

    def test_integer_rejects_string(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", "5", "integer")

    def test_integer_rejects_negative_float(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", -3.5, "integer")

    # ── float ────────────────────────────────────────────────
    def test_float_accepts_float(self):
        _validate_scalar_value("f", 3.14, "float")

    def test_float_accepts_negative_float(self):
        _validate_scalar_value("f", -0.5, "float")

    def test_float_rejects_int(self):
        # Strict: int is not float
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", 3, "float")

    def test_float_rejects_bool(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", True, "float")

    def test_float_rejects_string(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", "3.14", "float")

    # ── boolean ──────────────────────────────────────────────
    def test_boolean_accepts_true(self):
        _validate_scalar_value("f", True, "boolean")

    def test_boolean_accepts_false(self):
        _validate_scalar_value("f", False, "boolean")

    def test_boolean_rejects_int_one(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", 1, "boolean")

    def test_boolean_rejects_int_zero(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", 0, "boolean")

    def test_boolean_rejects_string_true(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", "true", "boolean")

    # ── datetime ─────────────────────────────────────────────
    def test_datetime_accepts_iso8601_with_time(self):
        _validate_scalar_value("f", "2024-01-15T14:30:00", "datetime")

    def test_datetime_accepts_date_only(self):
        _validate_scalar_value("f", "2024-01-15", "datetime")

    def test_datetime_accepts_with_timezone(self):
        _validate_scalar_value("f", "2024-01-15T14:30:00+05:30", "datetime")

    def test_datetime_rejects_human_format(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", "January 15, 2024", "datetime")

    def test_datetime_rejects_unix_timestamp(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", 1705324200, "datetime")

    def test_datetime_rejects_partial_string(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", "not-a-date", "datetime")

    # ── enum ─────────────────────────────────────────────────
    def test_enum_passes_through(self):
        # enum validation is handled in validate_output, not _validate_scalar_value
        _validate_scalar_value("f", "anything", "enum")  # no raise

    # ── unknown type ─────────────────────────────────────────
    def test_unknown_type_raises(self):
        with self.assertRaises(OutputContractError):
            _validate_scalar_value("f", "x", "uuid")


# ─────────────────────────────────────────────────────────────────────────────
# 3. parse_output
# ─────────────────────────────────────────────────────────────────────────────

class TestParseOutput(unittest.TestCase):

    def test_valid_json_object(self):
        result = parse_output('{"a": 1, "b": "hello"}')
        self.assertEqual(result, {"a": 1, "b": "hello"})

    def test_json_with_surrounding_whitespace(self):
        result = parse_output('  {"x": true}  ')
        self.assertEqual(result, {"x": True})

    def test_empty_string_raises(self):
        with self.assertRaises(ModelExecutionError):
            parse_output("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ModelExecutionError):
            parse_output("   ")

    def test_malformed_json_raises(self):
        with self.assertRaises(ModelExecutionError):
            parse_output("{not valid json}")

    def test_markdown_fenced_json_raises(self):
        # Model should never return markdown fences
        with self.assertRaises(ModelExecutionError):
            parse_output("```json\n{\"a\": 1}\n```")

    def test_truncated_json_raises(self):
        with self.assertRaises(ModelExecutionError):
            parse_output('{"a": 1, "b":')

    def test_none_input_raises(self):
        with self.assertRaises(ModelExecutionError):
            parse_output(None)


# ─────────────────────────────────────────────────────────────────────────────
# 4. validate_input
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateInput(unittest.TestCase):

    SCHEMA = {"ticket_text": "string", "priority": "string"}

    def test_valid_input_passes(self):
        validate_input({"ticket_text": "hello", "priority": "high"}, self.SCHEMA)

    def test_missing_one_field_raises(self):
        with self.assertRaises(InputContractError) as ctx:
            validate_input({"ticket_text": "hello"}, self.SCHEMA)
        self.assertIn("priority", str(ctx.exception))

    def test_missing_all_fields_raises(self):
        with self.assertRaises(InputContractError):
            validate_input({}, self.SCHEMA)

    def test_extra_field_raises(self):
        with self.assertRaises(InputContractError) as ctx:
            validate_input(
                {"ticket_text": "hello", "priority": "high", "injected": "bad"},
                self.SCHEMA,
            )
        self.assertIn("injected", str(ctx.exception))

    def test_multiple_extra_fields_all_reported(self):
        with self.assertRaises(InputContractError) as ctx:
            validate_input(
                {"ticket_text": "hi", "priority": "low", "a": 1, "b": 2},
                self.SCHEMA,
            )
        msg = str(ctx.exception)
        self.assertIn("a", msg)
        self.assertIn("b", msg)

    def test_wrong_type_string_field_raises(self):
        with self.assertRaises(InputContractError):
            validate_input({"ticket_text": 123, "priority": "high"}, self.SCHEMA)

    def test_empty_input_against_empty_schema(self):
        validate_input({}, {})  # no raise

    def test_missing_takes_priority_over_extra(self):
        with self.assertRaises(InputContractError) as ctx:
            validate_input({"extra": "val"}, {"required": "string"})
        self.assertIn("required", str(ctx.exception))


# ─────────────────────────────────────────────────────────────────────────────
# 5. validate_output
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateOutput(unittest.TestCase):

    def test_valid_string_and_integer_passes(self):
        validate_output({"summary": "ok", "count": 3}, {"summary": "string", "count": "integer"})

    def test_missing_required_field_raises(self):
        with self.assertRaises(OutputContractError) as ctx:
            validate_output({"summary": "ok"}, {"summary": "string", "count": "integer"})
        self.assertIn("count", str(ctx.exception))

    def test_extra_field_raises(self):
        with self.assertRaises(OutputContractError) as ctx:
            validate_output({"summary": "ok", "extra": "bad"}, {"summary": "string"})
        self.assertIn("extra", str(ctx.exception))

    def test_optional_field_may_be_absent(self):
        schema = {
            "summary": "string",
            "notes": {"type": "string", "optional": True},
        }
        validate_output({"summary": "ok"}, schema)  # no raise

    def test_optional_field_when_present_is_type_validated(self):
        schema = {"notes": {"type": "string", "optional": True}}
        with self.assertRaises(OutputContractError):
            validate_output({"notes": 999}, schema)

    def test_enum_valid_value_passes(self):
        schema = {"priority": {"type": "enum", "values": ["low", "medium", "high"]}}
        validate_output({"priority": "low"}, schema)

    def test_enum_invalid_value_raises(self):
        schema = {"priority": {"type": "enum", "values": ["low", "medium", "high"]}}
        with self.assertRaises(OutputContractError) as ctx:
            validate_output({"priority": "critical"}, schema)
        self.assertIn("critical", str(ctx.exception))

    def test_array_of_strings_passes(self):
        schema = {"tags": {"type": "array", "items": "string"}}
        validate_output({"tags": ["a", "b", "c"]}, schema)

    def test_array_of_floats_passes(self):
        schema = {"scores": {"type": "array", "items": "float"}}
        validate_output({"scores": [1.0, 2.5, 0.9]}, schema)

    def test_array_element_wrong_type_raises(self):
        schema = {"scores": {"type": "array", "items": "float"}}
        with self.assertRaises(OutputContractError) as ctx:
            validate_output({"scores": [1.0, "bad", 3.0]}, schema)
        self.assertIn("element 1", str(ctx.exception))

    def test_empty_array_passes(self):
        schema = {"tags": {"type": "array", "items": "string"}}
        validate_output({"tags": []}, schema)

    def test_array_not_a_list_raises(self):
        schema = {"tags": {"type": "array", "items": "string"}}
        with self.assertRaises(OutputContractError):
            validate_output({"tags": "not a list"}, schema)

    def test_boolean_true_passes(self):
        validate_output({"escalated": True}, {"escalated": "boolean"})

    def test_boolean_int_zero_raises(self):
        with self.assertRaises(OutputContractError):
            validate_output({"escalated": 0}, {"escalated": "boolean"})

    def test_integer_rejects_bool(self):
        with self.assertRaises(OutputContractError):
            validate_output({"count": True}, {"count": "integer"})

    def test_float_rejects_int(self):
        with self.assertRaises(OutputContractError):
            validate_output({"confidence": 1}, {"confidence": "float"})

    def test_float_accepts_float(self):
        validate_output({"confidence": 0.95}, {"confidence": "float"})

    def test_datetime_valid_string_passes(self):
        validate_output({"created_at": "2024-06-01T10:00:00"}, {"created_at": "datetime"})

    def test_datetime_invalid_string_raises(self):
        with self.assertRaises(OutputContractError):
            validate_output({"created_at": "June 1st, 2024"}, {"created_at": "datetime"})

    def test_multiple_required_missing_all_reported(self):
        schema = {"a": "string", "b": "string", "c": "string"}
        with self.assertRaises(OutputContractError) as ctx:
            validate_output({}, schema)
        msg = str(ctx.exception)
        self.assertIn("a", msg)
        self.assertIn("b", msg)
        self.assertIn("c", msg)


# ─────────────────────────────────────────────────────────────────────────────
# 6. apply_missing_and_defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyMissingAndDefaults(unittest.TestCase):

    SCHEMA = {
        "summary":  "string",
        "location": {"type": "string", "optional": True},
        "score":    {"type": "float",  "optional": True},
    }

    def _run(self, data, missing_cfg=None, defaults=None):
        return apply_missing_and_defaults(
            dict(data),
            self.SCHEMA,
            missing_cfg or {},
            defaults or {},
        )

    def test_no_config_returns_data_unchanged(self):
        result = apply_missing_and_defaults({"summary": "ok"}, self.SCHEMA, {}, {})
        self.assertNotIn("location", result)

    def test_strategy_null_fills_none_for_all_optionals(self):
        result = self._run({"summary": "ok"}, {"strategy": "null"})
        self.assertIsNone(result["location"])
        self.assertIsNone(result["score"])

    def test_strategy_unknown_fills_unknown_for_string_field(self):
        schema = {"notes": {"type": "string", "optional": True}}
        result = apply_missing_and_defaults(
            {}, schema, {"strategy": "unknown"}, {}
        )
        self.assertEqual(result["notes"], "unknown")

    def test_strategy_unknown_raises_for_float_field(self):
        with self.assertRaises(BehaviorDefinitionError) as ctx:
            self._run({"summary": "ok"}, {"strategy": "unknown"})
        self.assertIn("score", str(ctx.exception))

    def test_strategy_unknown_raises_for_integer_field(self):
        schema = {"count": {"type": "integer", "optional": True}}
        with self.assertRaises(BehaviorDefinitionError):
            apply_missing_and_defaults({}, schema, {"strategy": "unknown"}, {})

    def test_strategy_unknown_raises_for_boolean_field(self):
        schema = {"flag": {"type": "boolean", "optional": True}}
        with self.assertRaises(BehaviorDefinitionError):
            apply_missing_and_defaults({}, schema, {"strategy": "unknown"}, {})

    def test_strategy_reject_raises_when_field_missing(self):
        with self.assertRaises(ValidationRuleError) as ctx:
            self._run({"summary": "ok"}, {"strategy": "reject"})
        self.assertIn("location", str(ctx.exception))

    def test_override_takes_priority_over_strategy(self):
        result = self._run(
            {"summary": "ok"},
            {"strategy": "null", "overrides": {"location": "Paris"}},
        )
        self.assertEqual(result["location"], "Paris")

    def test_default_takes_priority_over_strategy(self):
        result = self._run(
            {"summary": "ok"},
            {"strategy": "null"},
            {"location": "Berlin"},
        )
        self.assertEqual(result["location"], "Berlin")

    def test_override_takes_priority_over_default(self):
        result = self._run(
            {"summary": "ok"},
            {"strategy": "null", "overrides": {"location": "Paris"}},
            {"location": "Berlin"},
        )
        self.assertEqual(result["location"], "Paris")

    def test_field_already_present_not_overwritten(self):
        result = self._run(
            {"summary": "ok", "location": "Tokyo"},
            {"strategy": "null"},
        )
        self.assertEqual(result["location"], "Tokyo")

    def test_required_fields_never_touched(self):
        # summary is required, should never be inserted by missing logic
        result = self._run({}, {"strategy": "null"})
        self.assertNotIn("summary", result)

    def test_override_with_wrong_type_raises(self):
        with self.assertRaises(OutputContractError):
            self._run(
                {"summary": "ok"},
                {"strategy": "null", "overrides": {"location": 999}},
            )

    def test_default_with_wrong_type_raises(self):
        with self.assertRaises(OutputContractError):
            self._run({"summary": "ok"}, {}, {"location": 42})

    def test_null_strategy_does_not_raise_for_non_string(self):
        # null is always safe — resolves to None for any type
        result = self._run({"summary": "ok"}, {"strategy": "null"})
        self.assertIsNone(result["score"])


# ─────────────────────────────────────────────────────────────────────────────
# 7. apply_normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyNormalization(unittest.TestCase):

    SCHEMA = {
        "summary":  "string",
        "category": "string",
        "count":    "integer",
    }

    def test_lowercase_applied_to_multiple_fields(self):
        data = {"summary": "HELLO World", "category": "BUG", "count": 1}
        result = apply_normalization(data, {"lowercase": ["summary", "category"]}, self.SCHEMA)
        self.assertEqual(result["summary"], "hello world")
        self.assertEqual(result["category"], "bug")

    def test_strip_removes_surrounding_whitespace(self):
        data = {"summary": "  hello  ", "category": "bug", "count": 1}
        result = apply_normalization(data, {"strip": ["summary"]}, self.SCHEMA)
        self.assertEqual(result["summary"], "hello")

    def test_strip_and_lowercase_combined(self):
        data = {"summary": "  HELLO  ", "category": "x", "count": 0}
        result = apply_normalization(
            data, {"lowercase": ["summary"], "strip": ["summary"]}, self.SCHEMA
        )
        # lowercase then strip — order depends on dict iteration, both should be applied
        self.assertIn(result["summary"], ["hello", "  hello  ".lower().strip()])

    def test_none_value_skipped_silently(self):
        data = {"summary": None, "category": "bug", "count": 1}
        result = apply_normalization(data, {"lowercase": ["summary"]}, self.SCHEMA)
        self.assertIsNone(result["summary"])  # no crash, value unchanged

    def test_non_string_field_raises(self):
        data = {"summary": "ok", "category": "bug", "count": 1}
        with self.assertRaises(OutputContractError):
            apply_normalization(data, {"lowercase": ["count"]}, self.SCHEMA)

    def test_unknown_field_raises(self):
        data = {"summary": "ok", "category": "bug", "count": 1}
        with self.assertRaises(BehaviorDefinitionError):
            apply_normalization(data, {"strip": ["nonexistent"]}, self.SCHEMA)

    def test_empty_normalization_returns_unchanged(self):
        data = {"summary": "Hello", "category": "Bug", "count": 1}
        result = apply_normalization(data, {}, self.SCHEMA)
        self.assertEqual(result["summary"], "Hello")
        self.assertEqual(result["category"], "Bug")

    def test_original_dict_mutated_in_place(self):
        data = {"summary": "  HELLO  ", "category": "x", "count": 0}
        result = apply_normalization(data, {"strip": ["summary"]}, self.SCHEMA)
        # result and data are the same object
        self.assertIs(result, data)


# ─────────────────────────────────────────────────────────────────────────────
# 8. apply_validation_rules
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyValidationRules(unittest.TestCase):

    SCHEMA = {
        "summary": "string",
        "topic":   "string",
        "tags":    {"type": "array", "items": "string"},
        "status":  "string",
        "count":   "integer",
    }

    def _run(self, data, rules):
        apply_validation_rules(data, rules, self.SCHEMA)

    def test_no_rules_passes(self):
        self._run({"summary": "ok"}, {})

    # ── max_length ────────────────────────────────────────────
    def test_max_length_within_limit_passes(self):
        self._run({"summary": "short"}, {"max_length": {"summary": 100}})

    def test_max_length_at_exact_limit_passes(self):
        self._run({"summary": "12345"}, {"max_length": {"summary": 5}})

    def test_max_length_exceeded_raises(self):
        with self.assertRaises(ValidationRuleError):
            self._run({"summary": "x" * 50}, {"max_length": {"summary": 10}})

    def test_max_length_bare_integer_raises_definition_error(self):
        with self.assertRaises(BehaviorDefinitionError):
            self._run({"summary": "hi"}, {"max_length": 100})

    def test_max_length_unknown_field_raises_definition_error(self):
        with self.assertRaises(BehaviorDefinitionError):
            self._run({"summary": "hi"}, {"max_length": {"ghost": 10}})

    def test_max_length_skips_none_value(self):
        self._run({"summary": None}, {"max_length": {"summary": 5}})  # no raise

    # ── pattern ───────────────────────────────────────────────
    def test_pattern_matching_value_passes(self):
        self._run({"topic": "2024-01-15"}, {"pattern": {"topic": r"\d{4}-\d{2}-\d{2}"}})

    def test_pattern_non_matching_value_raises(self):
        with self.assertRaises(ValidationRuleError):
            self._run({"topic": "not-a-date"}, {"pattern": {"topic": r"\d{4}-\d{2}-\d{2}"}})

    def test_pattern_uses_fullmatch_not_search(self):
        # "abc123" contains digits but should fail if pattern is just r"\d+"
        with self.assertRaises(ValidationRuleError):
            self._run({"topic": "abc123"}, {"pattern": {"topic": r"\d+"}})

    def test_pattern_skips_none_value(self):
        self._run({"topic": None}, {"pattern": {"topic": r"\d+"}})  # no raise

    def test_pattern_unknown_field_raises(self):
        with self.assertRaises(BehaviorDefinitionError):
            self._run({"topic": "x"}, {"pattern": {"ghost": r"\d+"}})

    # ── min_items ─────────────────────────────────────────────
    def test_min_items_sufficient_passes(self):
        self._run({"tags": ["a", "b", "c"]}, {"min_items": {"tags": 2}})

    def test_min_items_exact_minimum_passes(self):
        self._run({"tags": ["a"]}, {"min_items": {"tags": 1}})

    def test_min_items_empty_array_fails(self):
        with self.assertRaises(ValidationRuleError):
            self._run({"tags": []}, {"min_items": {"tags": 1}})

    def test_min_items_too_few_raises(self):
        with self.assertRaises(ValidationRuleError):
            self._run({"tags": ["a"]}, {"min_items": {"tags": 3}})

    def test_min_items_none_treated_as_empty(self):
        with self.assertRaises(ValidationRuleError):
            self._run({"tags": None}, {"min_items": {"tags": 1}})

    def test_min_items_unknown_field_raises(self):
        with self.assertRaises(BehaviorDefinitionError):
            self._run({"tags": []}, {"min_items": {"ghost": 1}})

    # ── forbidden_values ──────────────────────────────────────
    def test_forbidden_values_clean_value_passes(self):
        self._run({"status": "active"}, {"forbidden_values": {"status": ["unknown", "n/a"]}})

    def test_forbidden_values_exact_match_raises(self):
        with self.assertRaises(ValidationRuleError):
            self._run({"status": "unknown"}, {"forbidden_values": {"status": ["unknown", "n/a"]}})

    def test_forbidden_values_skips_none(self):
        self._run({"status": None}, {"forbidden_values": {"status": ["unknown"]}})

    def test_forbidden_values_unknown_field_raises(self):
        with self.assertRaises(BehaviorDefinitionError):
            self._run({"status": "x"}, {"forbidden_values": {"ghost": ["bad"]}})

    # ── required_keywords ─────────────────────────────────────
    def test_required_keywords_present_passes(self):
        self._run({"summary": "login failure bug"}, {"required_keywords": {"summary": ["login"]}})

    def test_required_keywords_missing_raises(self):
        with self.assertRaises(ValidationRuleError):
            self._run(
                {"summary": "database timeout"},
                {"required_keywords": {"summary": ["login"]}},
            )

    def test_required_keywords_case_insensitive(self):
        self._run({"summary": "LOGIN FAILURE"}, {"required_keywords": {"summary": ["login"]}})

    def test_required_keywords_multiple_all_required(self):
        self._run(
            {"summary": "login failure on checkout"},
            {"required_keywords": {"summary": ["login", "checkout"]}},
        )

    def test_required_keywords_one_missing_raises(self):
        with self.assertRaises(ValidationRuleError):
            self._run(
                {"summary": "login failure"},
                {"required_keywords": {"summary": ["login", "checkout"]}},
            )

    # ── multiple rules together ───────────────────────────────
    def test_multiple_rules_all_pass(self):
        self._run(
            {"summary": "login error", "tags": ["auth", "checkout"], "status": "open"},
            {
                "max_length": {"summary": 50},
                "required_keywords": {"summary": ["login"]},
                "min_items": {"tags": 1},
                "forbidden_values": {"status": ["unknown"]},
            },
        )

    def test_multiple_rules_one_fails_raises(self):
        with self.assertRaises(ValidationRuleError):
            self._run(
                {"summary": "x" * 200, "tags": ["a"], "status": "open"},
                {
                    "max_length": {"summary": 50},
                    "min_items": {"tags": 1},
                },
            )


# ─────────────────────────────────────────────────────────────────────────────
# 9. build_system_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSystemPrompt(unittest.TestCase):

    def test_contains_required_field_names(self):
        schema = {"summary": "string", "priority": {"type": "enum", "values": ["low", "high"]}}
        prompt = build_system_prompt(schema)
        self.assertIn("summary", prompt)
        self.assertIn("priority", prompt)

    def test_enum_values_listed_in_prompt(self):
        schema = {"priority": {"type": "enum", "values": ["low", "medium", "high"]}}
        prompt = build_system_prompt(schema)
        self.assertIn("low", prompt)
        self.assertIn("medium", prompt)
        self.assertIn("high", prompt)

    def test_optional_marker_present_for_optional_fields(self):
        schema = {"location": {"type": "string", "optional": True}}
        prompt = build_system_prompt(schema)
        self.assertIn("optional", prompt.lower())

    def test_integer_type_described(self):
        schema = {"count": "integer"}
        prompt = build_system_prompt(schema)
        self.assertIn("integer", prompt)

    def test_float_type_described(self):
        schema = {"confidence": "float"}
        prompt = build_system_prompt(schema)
        self.assertIn("float", prompt)

    def test_boolean_type_described(self):
        schema = {"escalated": "boolean"}
        prompt = build_system_prompt(schema)
        self.assertIn("boolean", prompt)

    def test_datetime_mentions_iso8601(self):
        schema = {"created_at": "datetime"}
        prompt = build_system_prompt(schema)
        self.assertIn("ISO8601", prompt)

    def test_array_mentions_items_type(self):
        schema = {"tags": {"type": "array", "items": "string"}}
        prompt = build_system_prompt(schema)
        self.assertIn("array", prompt)

    def test_required_fields_section_present(self):
        schema = {"summary": "string"}
        prompt = build_system_prompt(schema)
        self.assertIn("Required fields", prompt)

    def test_optional_fields_note_present_when_optionals_exist(self):
        schema = {
            "summary": "string",
            "notes": {"type": "string", "optional": True},
        }
        prompt = build_system_prompt(schema)
        self.assertIn("Optional fields", prompt)

    def test_no_optional_note_when_no_optionals(self):
        schema = {"summary": "string", "count": "integer"}
        prompt = build_system_prompt(schema)
        self.assertNotIn("Optional fields:", prompt)

    def test_prompt_contains_json_only_instruction(self):
        schema = {"summary": "string"}
        prompt = build_system_prompt(schema)
        self.assertIn("JSON", prompt)


class TestBuildUserPrompt(unittest.TestCase):

    def test_input_data_serialized(self):
        prompt = build_user_prompt({"ticket_text": "User locked out"})
        self.assertIn("ticket_text", prompt)
        self.assertIn("User locked out", prompt)

    def test_prompt_contains_transform_instruction(self):
        prompt = build_user_prompt({"x": "y"})
        self.assertIn("Transform", prompt)

    def test_multi_field_input_all_present(self):
        prompt = build_user_prompt({"a": "1", "b": "2", "c": "3"})
        self.assertIn('"a"', prompt)
        self.assertIn('"b"', prompt)
        self.assertIn('"c"', prompt)


# ─────────────────────────────────────────────────────────────────────────────
# 10. load_behavior
# ─────────────────────────────────────────────────────────────────────────────

MINIMAL_YAML = """\
    name: test
    model:
      name: gemini-2.5-flash
      temperature: 0.0
    input:
      schema:
        text: string
    output:
      schema:
        result: string
"""


class TestLoadBehavior(PlumberTestCase):

    def test_loads_valid_behavior(self):
        self.write_behavior("test", MINIMAL_YAML)
        b = load_behavior("test")
        self.assertEqual(b["name"], "test")

    def test_attaches_sha256_hash(self):
        self.write_behavior("test", MINIMAL_YAML)
        b = load_behavior("test")
        self.assertIn("__hash__", b)
        self.assertEqual(len(b["__hash__"]), 64)  # SHA256 hex = 64 chars

    def test_hash_is_stable_across_loads(self):
        self.write_behavior("test", MINIMAL_YAML)
        h1 = load_behavior("test")["__hash__"]
        h2 = load_behavior("test")["__hash__"]
        self.assertEqual(h1, h2)

    def test_hash_changes_when_content_changes(self):
        self.write_behavior("test", MINIMAL_YAML)
        h1 = load_behavior("test")["__hash__"]
        self.write_behavior("test", MINIMAL_YAML + "\n# changed")
        h2 = load_behavior("test")["__hash__"]
        self.assertNotEqual(h1, h2)

    def test_missing_file_raises_behavior_definition_error(self):
        with self.assertRaises(BehaviorDefinitionError):
            load_behavior("does_not_exist")

    def test_missing_input_section_raises(self):
        self.write_behavior("bad", textwrap.dedent("""\
            name: bad
            model:
              name: x
              temperature: 0.0
            output:
              schema:
                result: string
        """))
        with self.assertRaises(BehaviorDefinitionError) as ctx:
            load_behavior("bad")
        self.assertIn("input", str(ctx.exception).lower())

    def test_versioned_name_loads_correctly(self):
        self.write_behavior("test.v2", MINIMAL_YAML)
        b = load_behavior("test.v2")
        self.assertEqual(b["name"], "test")

    def test_hash_excludes_double_underscore_from_user_keys(self):
        self.write_behavior("test", MINIMAL_YAML)
        b = load_behavior("test")
        # __hash__ must not appear in top-level YAML keys linter sees
        user_keys = {k for k in b.keys() if not k.startswith("__")}
        self.assertNotIn("__hash__", user_keys)


# ─────────────────────────────────────────────────────────────────────────────
# 11. lint_behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestLintBehavior(PlumberTestCase):

    def _lint_ok(self, yaml_content: str) -> None:
        self.write_behavior("test", yaml_content)
        lint_behavior("test")  # must not raise

    def _lint_fails(self, yaml_content: str, *substrings: str) -> None:
        self.write_behavior("test", yaml_content)
        with self.assertRaises(BehaviorDefinitionError) as ctx:
            lint_behavior("test")
        msg = str(ctx.exception)
        for s in substrings:
            self.assertIn(s, msg)

    # ── valid behaviors ───────────────────────────────────────────────────────

    def test_minimal_valid_behavior_passes(self):
        self._lint_ok(MINIMAL_YAML)

    def test_enum_with_values_passes(self):
        self._lint_ok("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                status:
                  type: enum
                  values: [ok, fail]
        """)

    def test_array_with_items_passes(self):
        self._lint_ok("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                tags:
                  type: array
                  items: string
        """)

    def test_optional_field_passes(self):
        self._lint_ok("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
                notes:
                  type: string
                  optional: true
        """)

    def test_missing_strategy_null_passes(self):
        self._lint_ok("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
                notes:
                  type: string
                  optional: true
            missing:
              strategy: null
        """)

    def test_missing_strategy_reject_passes(self):
        self._lint_ok("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
                notes:
                  type: string
                  optional: true
            missing:
              strategy: reject
        """)

    def test_missing_strategy_unknown_on_string_fields_passes(self):
        self._lint_ok("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
                notes:
                  type: string
                  optional: true
            missing:
              strategy: unknown
        """)

    def test_valid_defaults_for_optional_field_passes(self):
        self._lint_ok("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
                severity:
                  type: string
                  optional: true
            defaults:
              severity: sev3
        """)

    def test_all_validation_rules_valid_passes(self):
        self._lint_ok("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                summary: string
                tags:
                  type: array
                  items: string
                status: string
            validation:
              max_length:
                summary: 120
              pattern:
                status: "^(open|closed)$"
              min_items:
                tags: 1
              forbidden_values:
                status: [unknown]
        """)

    def test_normalization_on_string_field_passes(self):
        self._lint_ok("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                summary: string
            normalization:
              strip: [summary]
              lowercase: [summary]
        """)

    def test_all_new_scalar_types_pass(self):
        self._lint_ok("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                count: integer
                score: float
                active: boolean
                created_at: datetime
        """)

    # ── failures: schema ──────────────────────────────────────────────────────

    def test_unknown_top_level_key_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
            unknown_key: bad
        """, "unknown_key")

    def test_unknown_schema_type_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: uuid
        """, "uuid")

    def test_enum_without_values_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                status:
                  type: enum
        """, "missing 'values'")

    def test_array_without_items_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                tags:
                  type: array
        """, "missing 'items'")

    def test_nested_array_items_rejected(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                matrix:
                  type: array
                  items: array
        """, "unsupported type")

    # ── failures: missing config ───────────────────────────────────────────────

    def test_invalid_missing_strategy_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
            missing:
              strategy: coerce
        """, "coerce")

    def test_unknown_strategy_incompatible_with_integer_optional_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
                count:
                  type: integer
                  optional: true
            missing:
              strategy: unknown
        """, "incompatible")

    def test_unknown_strategy_incompatible_with_float_optional_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
                score:
                  type: float
                  optional: true
            missing:
              strategy: unknown
        """, "incompatible")

    def test_overrides_unknown_field_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
            missing:
              strategy: null
              overrides:
                ghost: value
        """, "ghost")

    def test_overrides_non_optional_field_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
            missing:
              strategy: null
              overrides:
                result: something
        """, "not declared optional")

    # ── failures: defaults ────────────────────────────────────────────────────

    def test_defaults_unknown_field_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
            defaults:
              ghost: value
        """, "ghost")

    def test_defaults_non_optional_field_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                result: string
            defaults:
              result: fallback
        """, "not declared optional")

    def test_defaults_wrong_type_integer_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                count:
                  type: integer
                  optional: true
            defaults:
              count: "not_an_integer"
        """, "invalid default")

    def test_defaults_wrong_type_boolean_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                active:
                  type: boolean
                  optional: true
            defaults:
              active: "yes"
        """, "invalid default")

    # ── failures: validation rules ────────────────────────────────────────────

    def test_max_length_bare_integer_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                summary: string
            validation:
              max_length: 100
        """, "not a bare integer")

    def test_max_length_unknown_field_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                summary: string
            validation:
              max_length:
                ghost: 50
        """, "ghost")

    def test_min_items_on_non_array_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                summary: string
            validation:
              min_items:
                summary: 1
        """, "not an array type")

    def test_invalid_regex_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                summary: string
            validation:
              pattern:
                summary: "[invalid(regex"
        """, "invalid regex")

    def test_pattern_unknown_field_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                summary: string
            validation:
              pattern:
                ghost: ".*"
        """, "ghost")

    def test_forbidden_values_unknown_field_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                summary: string
            validation:
              forbidden_values:
                ghost: [bad]
        """, "ghost")

    # ── failures: normalization ───────────────────────────────────────────────

    def test_normalization_on_integer_field_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                count: integer
            normalization:
              lowercase: [count]
        """, "non-string field")

    def test_normalization_on_boolean_field_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                active: boolean
            normalization:
              strip: [active]
        """, "non-string field")

    def test_normalization_unknown_field_fails(self):
        self._lint_fails("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                summary: string
            normalization:
              strip: [ghost]
        """, "ghost")

    # ── multiple errors accumulated ───────────────────────────────────────────

    def test_multiple_errors_all_reported(self):
        self.write_behavior("test", textwrap.dedent("""\
            name: test
            model:
              name: gemini-2.5-flash
              temperature: 0.0
            input:
              schema:
                text: string
            output:
              schema:
                bad_type: uuid
                status:
                  type: enum
            validation:
              max_length: 100
        """))
        with self.assertRaises(BehaviorDefinitionError) as ctx:
            lint_behavior("test")
        msg = str(ctx.exception)
        self.assertIn("[1]", msg)
        self.assertIn("[2]", msg)
        self.assertIn("[3]", msg)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Execution Pipeline Ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionPipelineOrder(unittest.TestCase):
    """
    Validates that the pipeline order is:
      validate_output → apply_missing_and_defaults → apply_normalization → apply_validation_rules

    Key invariant: normalization runs AFTER missing resolution, so None values
    from strategy=null don't crash lowercase/strip operations.
    """

    def test_normalization_runs_after_missing_resolution_no_crash(self):
        schema = {
            "summary": "string",
            "notes":   {"type": "string", "optional": True},
        }
        # Model returned only the required field
        data = {"summary": "  Hello World  "}

        validate_output(data, schema)

        data = apply_missing_and_defaults(data, schema, {"strategy": "null"}, {})
        self.assertIsNone(data["notes"])  # notes filled with None

        # normalization must skip None silently — not crash
        data = apply_normalization(
            data, {"lowercase": ["notes"], "strip": ["summary"]}, schema
        )
        self.assertIsNone(data["notes"])  # still None
        self.assertEqual(data["summary"], "Hello World")  # stripped

        apply_validation_rules(data, {}, schema)  # no crash

    def test_validation_rules_run_after_normalization(self):
        # If normalization lowercases "OPEN" → "open", then forbidden_values check
        # against "OPEN" should not trigger (because normalization ran first).
        schema = {"status": "string"}
        data = {"status": "OPEN"}

        validate_output(data, schema)
        data = apply_missing_and_defaults(data, schema, {}, {})
        data = apply_normalization(data, {"lowercase": ["status"]}, schema)

        self.assertEqual(data["status"], "open")

        # forbidden check is on the POST-normalized value
        apply_validation_rules(
            data, {"forbidden_values": {"status": ["OPEN"]}}, schema
        )  # "OPEN" is forbidden but value is now "open" — passes

    def test_validation_rules_catch_post_normalization_value(self):
        schema = {"status": "string"}
        data = {"status": "OPEN"}

        validate_output(data, schema)
        data = apply_missing_and_defaults(data, schema, {}, {})
        data = apply_normalization(data, {"lowercase": ["status"]}, schema)

        self.assertEqual(data["status"], "open")

        with self.assertRaises(ValidationRuleError):
            apply_validation_rules(
                data, {"forbidden_values": {"status": ["open"]}}, schema
            )  # "open" (post-normalize) IS in forbidden list — fails


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()

    groups = [
        ("Schema Helpers",          [TestGetFieldType, TestIsOptional]),
        ("Scalar Type Validation",  [TestValidateScalarValue]),
        ("Output Parser",           [TestParseOutput]),
        ("Input Validation",        [TestValidateInput]),
        ("Output Validation",       [TestValidateOutput]),
        ("Missing / Defaults",      [TestApplyMissingAndDefaults]),
        ("Normalization",           [TestApplyNormalization]),
        ("Validation Rules",        [TestApplyValidationRules]),
        ("Prompt Construction",     [TestBuildSystemPrompt, TestBuildUserPrompt]),
        ("Behavior Loading",        [TestLoadBehavior]),
        ("Lint",                    [TestLintBehavior]),
        ("Pipeline Ordering",       [TestExecutionPipelineOrder]),
    ]

    verbose = "-v" in sys.argv
    width = 60
    total_run = total_fail = total_error = 0

    for group_name, classes in groups:
        section(group_name)
        group_suite = unittest.TestSuite()
        for cls in classes:
            group_suite.addTests(loader.loadTestsFromTestCase(cls))

        runner = unittest.TextTestRunner(
            verbosity=2 if verbose else 1,
            stream=sys.stdout,
        )
        result = runner.run(group_suite)
        total_run   += result.testsRun
        total_fail  += len(result.failures)
        total_error += len(result.errors)

    print(f"\n{BOLD}{'═' * width}{RESET}")
    if (total_fail + total_error) == 0:
        status = f"{GREEN}ALL PASSED{RESET}"
    else:
        status = f"{RED}FAILURES DETECTED{RESET}"
    print(
        f"{BOLD}  {status}  —  "
        f"{total_run} tests  |  "
        f"{total_fail} failures  |  "
        f"{total_error} errors{RESET}"
    )
    print(f"{BOLD}{'═' * width}{RESET}\n")

    sys.exit(0 if (total_fail + total_error) == 0 else 1)
