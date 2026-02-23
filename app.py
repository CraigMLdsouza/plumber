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
import json
import time
import textwrap
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

# ── Behavior Loading ──────────────────────────────────────────────────────────

def load_behavior(name: str) -> dict:
    # name is treated literally as the filename stem, supporting e.g. "summarize_ticket.v2"
    path = Path("behaviors") / f"{name}.yaml"
    if not path.exists():
        raise BehaviorDefinitionError(
            f"Behavior file not found: {path}\n"
            f"  Create it at: behaviors/{name}.yaml"
        )
    with path.open() as f:
        behavior = yaml.safe_load(f)

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
    for key, spec in output_schema.items():
        if isinstance(spec, dict) and spec.get("type") == "enum":
            field_lines.append(f'  "{key}": string — MUST be exactly one of: {spec["values"]}')
        else:
            t = spec if isinstance(spec, str) else spec.get("type", "string")
            field_lines.append(f'  "{key}": {t}')

    return textwrap.dedent(f"""
        You are a deterministic data transformation engine. You do NOT converse.
        You receive structured input and return ONLY a JSON object.

        OUTPUT CONTRACT:
        {{
        {chr(10).join(field_lines)}
        }}

        STRICT RULES:
        1. Return ONLY the raw JSON object. Nothing else.
        2. No markdown, no code fences, no backticks, no commentary.
        3. Include EVERY key listed above — no extras, no omissions.
        4. Enum fields must use exactly one of the allowed values — no deviation.
        5. Do not explain your reasoning.
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
    required = set(output_schema.keys())
    provided = set(data.keys())

    missing = required - provided
    if missing:
        raise OutputContractError(f"Output is missing required fields: {sorted(missing)}")

    extra = provided - required
    if extra:
        raise OutputContractError(
            f"Output contains unexpected fields not in schema: {sorted(extra)}"
        )

    for key, spec in output_schema.items():
        if isinstance(spec, dict) and spec.get("type") == "enum":
            allowed = spec["values"]
            actual = data.get(key)
            if actual not in allowed:
                raise OutputContractError(
                    f"Field '{key}' has value {repr(actual)} "
                    f"but must be one of: {allowed}"
                )

    log("Output schema validated", f"{len(provided)} field(s) match contract")
    for k, v in data.items():
        log_field(k, repr(v))

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
        if not isinstance(data.get(field), str):
            raise OutputContractError(
                f"Cannot apply 'lowercase' to non-string field '{field}'"
            )
        data[field] = data[field].lower()

    for field in normalization.get("strip", []):
        if not isinstance(data.get(field), str):
            raise OutputContractError(
                f"Cannot apply 'strip' to non-string field '{field}'"
            )
        data[field] = data[field].strip()

    applied = {rule: fields for rule, fields in normalization.items() if fields}
    log("Normalization applied", "  ".join(f"{r}={v}" for r, v in applied.items()))
    return data

# ── Validation Rules ──────────────────────────────────────────────────────────

def apply_validation_rules(data: dict, rules: dict, output_schema: dict) -> None:
    if not rules:
        return

    known_fields = set(output_schema.keys())
    violations = []

    if "max_length" in rules:
        spec = rules["max_length"]
        # Reject the flat integer format: max_length: 120
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
            if field in data and isinstance(data[field], str):
                if len(data[field]) > limit:
                    violations.append(
                        f"Field '{field}' exceeds max_length={limit} "
                        f"(got {len(data[field])} chars)"
                    )

    if "required_keywords" in rules:
        for field, keywords in rules["required_keywords"].items():
            if field not in known_fields:
                raise BehaviorDefinitionError(
                    f"Validation rule 'required_keywords' references unknown field '{field}'"
                )
            if field in data:
                text = str(data[field]).lower()
                missing_kw = [kw for kw in keywords if kw.lower() not in text]
                if missing_kw:
                    violations.append(
                        f"Field '{field}' must contain keywords: {missing_kw}"
                    )

    if violations:
        raise ValidationRuleError(
            "Validation rule violations:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    log("Validation rules passed", f"{len(rules)} rule(s) applied")

# ── Behavior Linter ───────────────────────────────────────────────────────────

VALID_TOP_LEVEL_KEYS = {"name", "model", "input", "output", "validation", "normalization"}
VALID_SCHEMA_TYPES   = {"string", "integer", "number", "boolean", "enum"}

def lint_behavior(name: str) -> None:
    """
    Validate a behavior YAML file without calling Gemini.
    Raises BehaviorDefinitionError on any structural problem.
    """
    behavior = load_behavior(name)
    errors = []

    # Unknown top-level keys
    unknown_keys = set(behavior.keys()) - VALID_TOP_LEVEL_KEYS
    if unknown_keys:
        errors.append(f"Unknown top-level keys: {sorted(unknown_keys)}")

    output_schema = behavior.get("output", {}).get("schema", {})

    # Validate output schema field types and enum definitions
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
        else:
            errors.append(f"Output field '{field}' has unparseable spec: {spec}")

    # Validation rules reference valid fields
    rules = behavior.get("validation", {})
    known_fields = set(output_schema.keys())

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
                    errors.append(
                        f"validation.max_length references unknown field '{field}'"
                    )

    if "required_keywords" in rules:
        for field in rules["required_keywords"]:
            if field not in known_fields:
                errors.append(
                    f"validation.required_keywords references unknown field '{field}'"
                )

    # Normalization references valid fields
    normalization = behavior.get("normalization", {})
    for rule, fields in normalization.items():
        for field in (fields or []):
            if field not in known_fields:
                errors.append(
                    f"normalization.{rule} references unknown field '{field}'"
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

    validate_input(input_data, input_schema)

    system_prompt = build_system_prompt(output_schema)
    user_prompt   = build_user_prompt(input_data)
    raw           = call_gemini(model_cfg, system_prompt, user_prompt)

    data = parse_output(raw)

    # Normalization runs BEFORE output validation
    data = apply_normalization(data, normalization, output_schema)

    validate_output(data, output_schema)
    apply_validation_rules(data, rules, output_schema)

    total_ms = int((time.time() - wall_start) * 1000)
    print(f"{BOLD}{'─' * 55}{RESET}")
    log("Completed", f"total: {total_ms}ms", GREEN)
    print(f"{BOLD}{'─' * 55}{RESET}")
    print()

    trace = {
        "behavior": name,
        "model":    model_cfg.get("name", "gemini-2.5-flash"),
        "latency_ms": total_ms,
        "validated": True,
    }
    return data, trace

def diff_behaviors(name_a: str, name_b: str):
    a = load_behavior(name_a)
    b = load_behavior(name_b)

    print("\n" + "─" * 55)
    print(f"{BOLD}  Plumber Runtime  |  diff: {name_a} ↔ {name_b}{RESET}")
    print("─" * 55)

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