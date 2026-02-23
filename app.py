"""
Plumber Runtime — deterministic, contract-enforced LLM function execution.

Usage:
    export GEMINI_API_KEY=your_key_here
    python app.py                        # run default test harness
    python app.py lint summarize_ticket  # lint a behavior YAML without calling Gemini

Dependencies:
    pip install google-genai pyyaml pydantic
"""

import os
import sys
import re
import json
import time
import hashlib
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from google import genai
from google.genai import types

# ── Error Types ───────────────────────────────────────────────────────────────

class BehaviorDefinitionError(Exception): pass  # Bad YAML structure / lint failure
class InputContractError(Exception):      pass  # Input schema violation
class OutputContractError(Exception):     pass  # Output schema violation
class ValidationRuleError(Exception):     pass  # Post-output validation rule violation
class ModelExecutionError(Exception):     pass  # Gemini call or JSON parse failure

# ── Logging ───────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
DIM    = "\033[2m"

def log(step: str, msg: str = "", color: str = CYAN) -> None:
    label = f"{color}{BOLD}[Plumber]{RESET}"
    step_str = f"{BOLD}{step}{RESET}"
    suffix = f" {DIM}→ {msg}{RESET}" if msg else ""
    print(f"{label} {step_str}{suffix}", flush=True)

def log_error(msg: str) -> None:
    print(f"{RED}{BOLD}[Plumber] ERROR:{RESET} {RED}{msg}{RESET}", file=sys.stderr, flush=True)

def log_field(key: str, value: Any) -> None:
    print(f"  {DIM}|{RESET}  {YELLOW}{key}{RESET}: {value}")

# ── Schema Helpers ────────────────────────────────────────────────────────────

def _get_field_type(spec: Any) -> str:
    """Return the declared type string for a field spec."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return spec.get("type", "string")
    return "string"

def _is_optional(spec: Any) -> bool:
    """Return True if field is declared optional."""
    if isinstance(spec, dict):
        return bool(spec.get("optional", False))
    return False

def _validate_scalar_value(field: str, value: Any, type_name: str) -> None:
    """
    Strict type validation — no coercion (Change 9).
    Raises OutputContractError on mismatch.
    """
    if type_name == "string":
        if not isinstance(value, str):
            raise OutputContractError(
                f"Field '{field}' must be string, got {type(value).__name__}: {repr(value)}"
            )
    elif type_name == "integer":
        # bool is a subclass of int in Python — reject it explicitly
        if isinstance(value, bool) or not isinstance(value, int):
            raise OutputContractError(
                f"Field '{field}' must be integer, got {type(value).__name__}: {repr(value)}"
            )
    elif type_name == "float":
        # Strict: model must emit a JSON float. int is a different type — reject it.
        if isinstance(value, bool) or not isinstance(value, float):
            raise OutputContractError(
                f"Field '{field}' must be float, got {type(value).__name__}: {repr(value)}"
            )
    elif type_name == "boolean":
        if not isinstance(value, bool):
            raise OutputContractError(
                f"Field '{field}' must be boolean (true/false), "
                f"got {type(value).__name__}: {repr(value)}"
            )
    elif type_name == "datetime":
        if not isinstance(value, str):
            raise OutputContractError(
                f"Field '{field}' must be an ISO8601 datetime string, "
                f"got {type(value).__name__}: {repr(value)}"
            )
        try:
            datetime.fromisoformat(value)
        except ValueError:
            raise OutputContractError(
                f"Field '{field}' value {repr(value)} is not a valid ISO8601 datetime"
            )
    elif type_name == "enum":
        pass  # enum validation handled separately in validate_output
    else:
        raise OutputContractError(
            f"Field '{field}' has unknown type '{type_name}'"
        )

# ── Behavior Loading ──────────────────────────────────────────────────────────

def load_behavior(name: str) -> dict:
    # name is treated literally as the filename stem, supporting e.g. "summarize_ticket.v2"
    path = Path("behaviors") / f"{name}.yaml"
    if not path.exists():
        raise BehaviorDefinitionError(
            f"Behavior file not found: {path}\n"
            f"  Create it at: behaviors/{name}.yaml"
        )
    raw_bytes = path.read_bytes()
    behavior  = yaml.safe_load(raw_bytes)

    # Attach fingerprint — SHA256 of the raw YAML bytes (Change 5)
    behavior["__hash__"] = hashlib.sha256(raw_bytes).hexdigest()

    required_sections = {"name", "model", "input", "output"}
    missing = required_sections - set(behavior.keys())
    if missing:
        raise BehaviorDefinitionError(f"Behavior YAML missing required sections: {missing}")

    log("Behavior loaded", f"{name}.yaml  [{behavior['model'].get('name', '?')}]", GREEN)
    return behavior

# ── Input Validation ──────────────────────────────────────────────────────────

def validate_input(input_data: dict, schema: dict) -> None:
    required = set(schema.keys())
    provided = set(input_data.keys())

    missing = required - provided
    if missing:
        raise InputContractError(f"Input missing required fields: {sorted(missing)}")

    extra = provided - required
    if extra:
        raise InputContractError(f"Input contains unexpected fields (rejected): {sorted(extra)}")

    for field, spec in schema.items():
        declared_type = spec if isinstance(spec, str) else spec.get("type", "string")
        value = input_data[field]
        if declared_type == "string" and not isinstance(value, str):
            raise InputContractError(
                f"Input field '{field}' must be a string, got {type(value).__name__}"
            )

    log("Input validated", f"{len(provided)} field(s): {', '.join(sorted(provided))}")
    for k, v in input_data.items():
        preview = str(v)[:80] + ("..." if len(str(v)) > 80 else "")
        log_field(k, f'"{preview}"')

# ── Prompt Construction ───────────────────────────────────────────────────────

def build_system_prompt(output_schema: dict) -> str:
    field_lines = []
    required_keys = []
    optional_keys = []

    for key, spec in output_schema.items():
        field_type = _get_field_type(spec)
        optional   = _is_optional(spec)

        if field_type == "enum":
            values = spec["values"] if isinstance(spec, dict) else []
            desc = f'string — MUST be exactly one of: {values}'
        elif field_type == "array":
            items_type = spec.get("items", "string") if isinstance(spec, dict) else "string"
            desc = f'array of {items_type}s — JSON array'
        elif field_type == "integer":
            desc = "integer — JSON number with no decimal point"
        elif field_type == "float":
            desc = "float — JSON number (decimal allowed)"
        elif field_type == "boolean":
            desc = "boolean — JSON true or false (no quotes)"
        elif field_type == "datetime":
            desc = "string — ISO8601 datetime (e.g. 2024-01-15T14:30:00)"
        else:
            desc = field_type  # string

        opt_marker = " (optional — omit if unknown)" if optional else ""
        field_lines.append(f'  "{key}": {desc}{opt_marker}')

        if optional:
            optional_keys.append(key)
        else:
            required_keys.append(key)

    optional_note = ""
    if optional_keys:
        optional_note = (
            f"\nOptional fields: {optional_keys}. "
            "Include them only if the input contains clear evidence. "
            "Omit them entirely (do NOT emit null) if uncertain."
        )

    return textwrap.dedent(f"""
        You are a deterministic data transformation engine. You do NOT converse.
        You receive structured input and return ONLY a JSON object.

        OUTPUT CONTRACT:
        {{
        {chr(10).join(field_lines)}
        }}

        Required fields: {required_keys}
        {optional_note}

        STRICT RULES:
        1. Return ONLY the raw JSON object. Nothing else.
        2. No markdown, no code fences, no backticks, no commentary.
        3. Include every REQUIRED field — no omissions.
        4. Enum fields must use exactly one of the allowed values — no deviation.
        5. integer fields must be JSON integers (no quotes, no decimals).
        6. boolean fields must be JSON true or false (no quotes).
        7. float fields must be JSON numbers.
        8. datetime fields must be ISO8601 strings.
        9. Do not explain your reasoning.
    """).strip()

def build_user_prompt(input_data: dict) -> str:
    return (
        f"INPUT:\n{json.dumps(input_data, indent=2)}\n\n"
        "Transform this input and return the JSON object."
    )

# ── Gemini Call ───────────────────────────────────────────────────────────────

def call_gemini(model_cfg: dict, system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "  Run: export GEMINI_API_KEY=your_key_here"
        )

    client = genai.Client(api_key=api_key)
    model_name  = model_cfg.get("name", "gemini-2.5-flash")
    temperature = float(model_cfg.get("temperature", 0.2))

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        response_mime_type="application/json",
        # Disable thinking for deterministic structured output (faster + cheaper)
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    log("Calling model", f"{model_name}  [temp={temperature}]", YELLOW)
    t0 = time.time()

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=config,
    )

    elapsed_ms = int((time.time() - t0) * 1000)
    usage = response.usage_metadata
    log(
        "Model responded",
        f"{elapsed_ms}ms  "
        f"[in={usage.prompt_token_count} tok  out={usage.candidates_token_count} tok]",
        GREEN,
    )
    return response.text

# ── Output Parsing ────────────────────────────────────────────────────────────

def parse_output(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ModelExecutionError("Model returned an empty response.")
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise ModelExecutionError(
            f"Model returned invalid JSON.\n"
            f"  Parse error: {e}\n"
            f"  Raw output (first 500 chars):\n    {raw[:500]}"
        )

# ── Output Validation ─────────────────────────────────────────────────────────

def validate_output(data: dict, output_schema: dict) -> None:
    required_fields  = {k for k, v in output_schema.items() if not _is_optional(v)}
    optional_fields  = {k for k, v in output_schema.items() if _is_optional(v)}
    all_schema_fields = set(output_schema.keys())
    provided = set(data.keys())

    # Required fields must be present
    missing = required_fields - provided
    if missing:
        raise OutputContractError(f"Output is missing required fields: {sorted(missing)}")

    # No extra keys allowed
    extra = provided - all_schema_fields
    if extra:
        raise OutputContractError(
            f"Output contains unexpected fields not in schema: {sorted(extra)}"
        )

    # Type-validate every present field
    for key in provided:
        spec       = output_schema[key]
        field_type = _get_field_type(spec)
        value      = data[key]

        if field_type == "enum":
            allowed = spec["values"] if isinstance(spec, dict) else []
            if value not in allowed:
                raise OutputContractError(
                    f"Field '{key}' has value {repr(value)} "
                    f"but must be one of: {allowed}"
                )

        elif field_type == "array":
            if not isinstance(value, list):
                raise OutputContractError(
                    f"Field '{key}' must be an array, got {type(value).__name__}"
                )
            items_type = spec.get("items", "string") if isinstance(spec, dict) else "string"
            for i, elem in enumerate(value):
                try:
                    _validate_scalar_value(f"{key}[{i}]", elem, items_type)
                except OutputContractError as e:
                    raise OutputContractError(
                        f"Array field '{key}' element {i} failed type check: {e}"
                    )
        else:
            _validate_scalar_value(key, value, field_type)

    log("Output schema validated", f"{len(provided)} field(s) match contract")
    for k, v in data.items():
        log_field(k, repr(v))

# ── Missing Value Strategy + Defaults ────────────────────────────────────────

def apply_missing_and_defaults(
    data: dict,
    output_schema: dict,
    missing_cfg: dict,
    defaults: dict,
) -> dict:
    """
    Runs AFTER validate_output. Fills in missing optional fields deterministically.
    Priority: field override > default > strategy (Change 11).
    """
    if not missing_cfg and not defaults:
        return data

    # YAML `null` deserialises to Python None — normalise to the string "null"
    _raw_strategy = missing_cfg.get("strategy", "null") if missing_cfg else "null"
    strategy = "null" if _raw_strategy is None else _raw_strategy
    overrides = missing_cfg.get("overrides", {}) if missing_cfg else {}

    for field, spec in output_schema.items():
        if not _is_optional(spec):
            continue
        if field in data:
            continue  # already present — nothing to do

        # Priority 1: field-level override
        if field in overrides:
            data[field] = overrides[field]

        # Priority 2: behavior-level default
        elif field in defaults:
            data[field] = defaults[field]

        # Priority 3: strategy
        elif strategy == "null":
            data[field] = None
        elif strategy == "unknown":
            field_type = _get_field_type(spec)
            if field_type not in ("string", "enum"):
                raise BehaviorDefinitionError(
                    f"missing.strategy 'unknown' cannot be used for non-string field "
                    f"'{field}' (type={field_type}). Use null or an override instead."
                )
            data[field] = "unknown"
        elif strategy == "reject":
            raise ValidationRuleError(
                f"Optional field '{field}' is missing and strategy is 'reject'"
            )
        else:
            raise BehaviorDefinitionError(
                f"Unknown missing strategy: '{strategy}'. "
                "Supported: null, unknown, reject"
            )

        # Type-validate the resolved value (override or default may be stale — Issue 2)
        resolved = data.get(field)
        if resolved is not None:
            field_type = _get_field_type(spec)
            if field_type == "array":
                if not isinstance(resolved, list):
                    raise OutputContractError(
                        f"Resolved value for optional field '{field}' must be a list, "
                        f"got {type(resolved).__name__}: {repr(resolved)}"
                    )
            elif field_type != "enum":
                _validate_scalar_value(field, resolved, field_type)

    filled = [f for f in output_schema if _is_optional(output_schema[f])]
    if filled:
        log("Missing/defaults applied", f"optional fields resolved: {filled}")
    return data

# ── Normalization ─────────────────────────────────────────────────────────────

def apply_normalization(data: dict, normalization: dict, output_schema: dict) -> dict:
    """Mutates and returns data after applying normalization rules."""
    if not normalization:
        return data

    known_fields = set(output_schema.keys())

    for rule, fields in normalization.items():
        unknown = [f for f in fields if f not in known_fields]
        if unknown:
            raise BehaviorDefinitionError(
                f"Normalization rule '{rule}' references unknown field(s): {unknown}"
            )

    for field in normalization.get("lowercase", []):
        val = data.get(field)
        if val is None:
            continue  # optional field resolved to null — skip
        if not isinstance(val, str):
            raise OutputContractError(
                f"Cannot apply 'lowercase' to non-string field '{field}'"
            )
        data[field] = val.lower()

    for field in normalization.get("strip", []):
        val = data.get(field)
        if val is None:
            continue  # optional field resolved to null — skip
        if not isinstance(val, str):
            raise OutputContractError(
                f"Cannot apply 'strip' to non-string field '{field}'"
            )
        data[field] = val.strip()

    applied = {rule: fields for rule, fields in normalization.items() if fields}
    log("Normalization applied", "  ".join(f"{r}={v}" for r, v in applied.items()))
    return data

# ── Validation Rules ──────────────────────────────────────────────────────────

def apply_validation_rules(data: dict, rules: dict, output_schema: dict) -> None:
    if not rules:
        return

    known_fields = set(output_schema.keys())
    violations = []

    # max_length — field-specific dict only
    if "max_length" in rules:
        spec = rules["max_length"]
        if not isinstance(spec, dict):
            raise BehaviorDefinitionError(
                "Invalid 'max_length' format. Must be field-specific:\n"
                "  max_length:\n"
                "    summary: 120\n"
                "Got a bare value instead."
            )
        for field, limit in spec.items():
            if field not in known_fields:
                raise BehaviorDefinitionError(
                    f"Validation rule 'max_length' references unknown field '{field}'"
                )
            val = data.get(field)
            if val is not None and isinstance(val, str) and len(val) > limit:
                violations.append(
                    f"Field '{field}' exceeds max_length={limit} "
                    f"(got {len(val)} chars)"
                )

    # required_keywords
    if "required_keywords" in rules:
        for field, keywords in rules["required_keywords"].items():
            if field not in known_fields:
                raise BehaviorDefinitionError(
                    f"Validation rule 'required_keywords' references unknown field '{field}'"
                )
            val = data.get(field)
            if val is not None:
                text = str(val).lower()
                missing_kw = [kw for kw in keywords if kw.lower() not in text]
                if missing_kw:
                    violations.append(
                        f"Field '{field}' must contain keywords: {missing_kw}"
                    )

    # pattern — regex via re.fullmatch (Change 4)
    if "pattern" in rules:
        for field, pattern in rules["pattern"].items():
            if field not in known_fields:
                raise BehaviorDefinitionError(
                    f"Validation rule 'pattern' references unknown field '{field}'"
                )
            val = data.get(field)
            if val is not None:
                if not isinstance(val, str):
                    violations.append(
                        f"Field '{field}' pattern check requires string, "
                        f"got {type(val).__name__}"
                    )
                elif not re.fullmatch(pattern, val):
                    violations.append(
                        f"Field '{field}' value {repr(val)} "
                        f"does not match pattern '{pattern}'"
                    )

    # min_items — for array fields (Change 4)
    if "min_items" in rules:
        for field, minimum in rules["min_items"].items():
            if field not in known_fields:
                raise BehaviorDefinitionError(
                    f"Validation rule 'min_items' references unknown field '{field}'"
                )
            val = data.get(field)
            if val is None:
                val = []  # treat missing optional array as empty
            if not isinstance(val, list):
                violations.append(
                    f"Field '{field}' min_items check requires array, "
                    f"got {type(val).__name__}"
                )
            elif len(val) < minimum:
                violations.append(
                    f"Field '{field}' has {len(val)} item(s) "
                    f"but min_items={minimum}"
                )

    # forbidden_values (Change 4)
    if "forbidden_values" in rules:
        for field, forbidden in rules["forbidden_values"].items():
            if field not in known_fields:
                raise BehaviorDefinitionError(
                    f"Validation rule 'forbidden_values' references unknown field '{field}'"
                )
            val = data.get(field)
            if val is not None and val in forbidden:
                violations.append(
                    f"Field '{field}' has forbidden value {repr(val)}"
                )

    if violations:
        raise ValidationRuleError(
            "Validation rule violations:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    log("Validation rules passed", f"{len(rules)} rule(s) applied")

# ── Behavior Linter ───────────────────────────────────────────────────────────

VALID_TOP_LEVEL_KEYS = {
    "name", "model", "input", "output",
    "validation", "normalization", "missing", "defaults",
}
VALID_SCHEMA_TYPES = {
    "string", "integer", "float", "boolean", "datetime", "enum", "array",
}

def lint_behavior(name: str) -> None:
    """
    Validate a behavior YAML file without calling Gemini.
    Raises BehaviorDefinitionError on any structural problem.
    """
    behavior = load_behavior(name)
    errors = []

    # Unknown top-level keys (exclude internal __hash__)
    user_keys = {k for k in behavior.keys() if not k.startswith("__")}
    unknown_keys = user_keys - VALID_TOP_LEVEL_KEYS
    if unknown_keys:
        errors.append(f"Unknown top-level keys: {sorted(unknown_keys)}")

    output_schema = behavior.get("output", {}).get("schema", {})
    known_fields  = set(output_schema.keys())

    # ── Output schema field type validation ───────────────────────────────────
    for field, spec in output_schema.items():
        if isinstance(spec, str):
            if spec not in VALID_SCHEMA_TYPES:
                errors.append(f"Output field '{field}' has unknown type '{spec}'")
        elif isinstance(spec, dict):
            field_type = spec.get("type")
            if field_type not in VALID_SCHEMA_TYPES:
                errors.append(f"Output field '{field}' has unknown type '{field_type}'")
            if field_type == "enum" and not spec.get("values"):
                errors.append(f"Output field '{field}' is enum but missing 'values'")
            if field_type == "array":
                items_type = spec.get("items")
                if items_type is None:
                    errors.append(f"Output field '{field}' is array but missing 'items'")
                elif items_type not in VALID_SCHEMA_TYPES - {"array", "enum"}:
                    errors.append(
                        f"Output field '{field}' array items has unsupported type '{items_type}'"
                    )
        else:
            errors.append(f"Output field '{field}' has unparseable spec: {spec}")

    # ── missing config validation ─────────────────────────────────────────────
    missing_cfg = behavior.get("missing", {})
    if missing_cfg:
        # YAML `null` deserialises to Python None — normalise to the string "null"
        _raw = missing_cfg.get("strategy")
        strategy = "null" if _raw is None else _raw
        valid_strategies = {"null", "unknown", "reject"}
        if strategy not in valid_strategies:
            errors.append(
                f"missing.strategy '{strategy}' is invalid. "
                f"Must be one of: {sorted(valid_strategies)}"
            )
        # Check that 'unknown' is only used when all optional fields are string/enum
        if strategy == "unknown":
            for field, spec in output_schema.items():
                if _is_optional(spec):
                    ft = _get_field_type(spec)
                    if ft not in ("string", "enum"):
                        errors.append(
                            f"missing.strategy 'unknown' is incompatible with optional "
                            f"field '{field}' (type={ft}). Use null or an override instead."
                        )
        overrides = missing_cfg.get("overrides", {})
        for field in overrides:
            if field not in known_fields:
                errors.append(
                    f"missing.overrides references unknown field '{field}'"
                )
            elif not _is_optional(output_schema.get(field, "string")):
                errors.append(
                    f"missing.overrides field '{field}' is not declared optional"
                )

    # ── defaults validation ───────────────────────────────────────────────────
    defaults = behavior.get("defaults", {})
    for field, default_value in defaults.items():
        if field not in known_fields:
            errors.append(f"defaults references unknown field '{field}'")
        elif not _is_optional(output_schema.get(field, "string")):
            errors.append(f"defaults field '{field}' is not declared optional")
        else:
            # Validate the default value matches the declared type
            spec = output_schema[field]
            field_type = _get_field_type(spec)
            try:
                if field_type == "array":
                    if not isinstance(default_value, list):
                        errors.append(
                            f"defaults field '{field}' must be a list, "
                            f"got {type(default_value).__name__}"
                        )
                elif field_type not in ("enum",):
                    _validate_scalar_value(field, default_value, field_type)
            except OutputContractError as e:
                errors.append(f"defaults field '{field}' has invalid default: {e}")

    # ── validation rules ──────────────────────────────────────────────────────
    rules = behavior.get("validation", {})

    if "max_length" in rules:
        spec = rules["max_length"]
        if not isinstance(spec, dict):
            errors.append(
                "'max_length' must be field-specific (e.g. max_length: {summary: 120}), "
                "not a bare integer"
            )
        else:
            for field in spec:
                if field not in known_fields:
                    errors.append(f"validation.max_length references unknown field '{field}'")

    if "required_keywords" in rules:
        for field in rules["required_keywords"]:
            if field not in known_fields:
                errors.append(
                    f"validation.required_keywords references unknown field '{field}'"
                )

    if "pattern" in rules:
        for field, pattern in rules["pattern"].items():
            if field not in known_fields:
                errors.append(f"validation.pattern references unknown field '{field}'")
            else:
                try:
                    re.compile(pattern)
                except re.error as e:
                    errors.append(f"validation.pattern for '{field}' is invalid regex: {e}")

    if "min_items" in rules:
        for field in rules["min_items"]:
            if field not in known_fields:
                errors.append(f"validation.min_items references unknown field '{field}'")
            elif _get_field_type(output_schema[field]) != "array":
                errors.append(
                    f"validation.min_items field '{field}' is not an array type"
                )

    if "forbidden_values" in rules:
        for field in rules["forbidden_values"]:
            if field not in known_fields:
                errors.append(f"validation.forbidden_values references unknown field '{field}'")

    # ── normalization references valid fields ─────────────────────────────────
    normalization = behavior.get("normalization", {})
    for rule, fields in normalization.items():
        for field in (fields or []):
            if field not in known_fields:
                errors.append(f"normalization.{rule} references unknown field '{field}'")
            else:
                field_type = _get_field_type(output_schema[field])
                if field_type not in ("string", "enum"):
                    errors.append(
                        f"normalization.{rule} cannot target non-string field "
                        f"'{field}' (type={field_type})"
                    )

    if errors:
        formatted = "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        raise BehaviorDefinitionError(
            f"Behavior '{name}' failed lint with {len(errors)} error(s):\n{formatted}"
        )

    log("Lint passed", f"'{name}' is a valid behavior definition", GREEN)

# ── Core Runtime ──────────────────────────────────────────────────────────────

def run_behavior(name: str, input_data: dict) -> tuple[dict, dict]:
    """
    Execute a named behavior against the given input.
    Returns (output_data, trace). Raises loudly on any violation.
    """
    wall_start = time.time()

    print()
    print(f"{BOLD}{'─' * 55}{RESET}")
    print(f"{BOLD}  Plumber Runtime  |  behavior: {name}{RESET}")
    print(f"{BOLD}{'─' * 55}{RESET}")

    behavior      = load_behavior(name)
    input_schema  = behavior["input"]["schema"]
    output_schema = behavior["output"]["schema"]
    model_cfg     = behavior.get("model", {})
    rules         = behavior.get("validation", {})
    normalization = behavior.get("normalization", {})
    missing_cfg   = behavior.get("missing", {})
    defaults      = behavior.get("defaults", {})
    behavior_hash = behavior["__hash__"]

    validate_input(input_data, input_schema)

    system_prompt = build_system_prompt(output_schema)
    user_prompt   = build_user_prompt(input_data)
    raw           = call_gemini(model_cfg, system_prompt, user_prompt)

    data = parse_output(raw)

    # Execution order (Change 11):
    # 1. type/schema validation
    validate_output(data, output_schema)
    # 2. apply missing strategy + defaults
    data = apply_missing_and_defaults(data, output_schema, missing_cfg, defaults)
    # 3. normalization
    data = apply_normalization(data, normalization, output_schema)
    # 4. validation rules
    apply_validation_rules(data, rules, output_schema)

    total_ms = int((time.time() - wall_start) * 1000)
    print(f"{BOLD}{'─' * 55}{RESET}")
    log("Completed", f"total: {total_ms}ms", GREEN)
    print(f"{BOLD}{'─' * 55}{RESET}")
    print()

    trace = {
        "behavior":      name,
        "behavior_hash": behavior_hash,
        "model":         model_cfg.get("name", "gemini-2.5-flash"),
        "latency_ms":    total_ms,
        "validated":     True,
    }
    return data, trace

def diff_behaviors(name_a: str, name_b: str):
    a = load_behavior(name_a)
    b = load_behavior(name_b)

    print("\n" + "─" * 55)
    print(f"{BOLD}  Plumber Runtime  |  diff: {name_a} ↔ {name_b}{RESET}")
    print("─" * 55)
    print(f"  Hash A: {DIM}{a['__hash__']}{RESET}")
    print(f"  Hash B: {DIM}{b['__hash__']}{RESET}")

    def show(title):
        print(f"\n{BOLD}{title}{RESET}")
        print("-" * len(title))

    # ── Output Schema Diff ─────────────────────────────
    a_schema = a["output"]["schema"]
    b_schema = b["output"]["schema"]

    added   = set(b_schema) - set(a_schema)
    removed = set(a_schema) - set(b_schema)
    common  = set(a_schema) & set(b_schema)

    show("Schema Changes")

    if not added and not removed:
        print("  No structural field changes.")
    else:
        for f in sorted(added):
            print(f"  + Added field: {f}")
        for f in sorted(removed):
            print(f"  - Removed field: {f}")

    # ── Enum Changes ───────────────────────────────────
    show("Enum Changes")

    enum_changed = False
    for field in common:
        a_spec = a_schema[field]
        b_spec = b_schema[field]

        if (
            isinstance(a_spec, dict)
            and isinstance(b_spec, dict)
            and a_spec.get("type") == "enum"
            and b_spec.get("type") == "enum"
        ):
            if a_spec["values"] != b_spec["values"]:
                enum_changed = True
                print(f"  {field}:")
                print(f"    before → {a_spec['values']}")
                print(f"    after  → {b_spec['values']}")

    if not enum_changed:
        print("  No enum changes.")

    # ── Validation Diff ────────────────────────────────
    show("Validation Changes")

    if a.get("validation") == b.get("validation"):
        print("  No validation changes.")
    else:
        print(f"  before → {a.get('validation')}")
        print(f"  after  → {b.get('validation')}")

    # ── Normalization Diff ─────────────────────────────
    show("Normalization Changes")

    if a.get("normalization") == b.get("normalization"):
        print("  No normalization changes.")
    else:
        print(f"  before → {a.get('normalization')}")
        print(f"  after  → {b.get('normalization')}")

    print("\n" + "─" * 55)
# ── Test Harness / CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # CLI: python app.py lint <behavior_name>
    if len(sys.argv) == 3 and sys.argv[1] == "lint":
        behavior_name = sys.argv[2]
        try:
            lint_behavior(behavior_name)
        except BehaviorDefinitionError as e:
            log_error(str(e))
            sys.exit(1)
        sys.exit(0)

    # CLI: python app.py diff <behavior_a> <behavior_b>
    if len(sys.argv) == 4 and sys.argv[1] == "diff":
        try:
            diff_behaviors(sys.argv[2], sys.argv[3])
        except BehaviorDefinitionError as e:
            log_error(str(e))
            sys.exit(1)
        sys.exit(0)

    # CLI: python app.py simulate <behavior_name>  (Change 6)
    if len(sys.argv) == 3 and sys.argv[1] == "simulate":
        behavior_name = sys.argv[2]
        try:
            lint_behavior(behavior_name)
            behavior      = load_behavior(behavior_name)
            output_schema = behavior["output"]["schema"]
            system_prompt = build_system_prompt(output_schema)
            user_prompt   = build_user_prompt({"<field>": "<value>"})

            print()
            print(f"{BOLD}{'─' * 55}{RESET}")
            print(f"{BOLD}  Plumber Simulate  |  behavior: {behavior_name}{RESET}")
            print(f"{BOLD}{'─' * 55}{RESET}")
            print(f"\n{BOLD}System Prompt:{RESET}\n")
            print(system_prompt)
            print(f"\n{BOLD}User Prompt Template:{RESET}\n")
            print(user_prompt)
            print(f"\n{DIM}Behavior hash: {behavior['__hash__']}{RESET}")
            print(f"{BOLD}{'─' * 55}{RESET}")
            print(f"{YELLOW}{BOLD}[Simulate] No model call was made.{RESET}")

        except BehaviorDefinitionError as e:
            log_error(str(e))
            sys.exit(1)
        sys.exit(0)

    # Default: run the sample behavior
    sample_ticket = "User cannot log in during checkout."

    try:
        result, trace = run_behavior(
            "summarize_ticket",
            {"ticket_text": sample_ticket},
        )
        print(f"{GREEN}{BOLD}Result:{RESET}")
        print(json.dumps(result, indent=2))
        print(f"\n{DIM}Trace:{RESET}")
        print(json.dumps(trace, indent=2))

    except (
        BehaviorDefinitionError,
        InputContractError,
        OutputContractError,
        ValidationRuleError,
        ModelExecutionError,
    ) as e:
        log_error(f"[{type(e).__name__}] {e}")
        sys.exit(1)