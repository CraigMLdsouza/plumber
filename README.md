# Plumber 👨‍🔧

## Summary
Plumber is a deterministic AI execution engine and compliance middleware. LLMs are treated as untrusted, probabilistic components; the runtime is a strict contract enforcer that guarantees deterministic schema, routing, cache, serialization, and float math. This document describes the complete system as implemented.

## Core Thesis
- **LLM = untrusted component**: semantic misclassification is outside the deterministic SLA.
- **Runtime = trusted contract enforcer**: schema, routing, state, cache, and float math are deterministic.

## High-Level Capabilities
- Deterministic schema validation and strict type enforcement.
- Canonical JSON serialization for idempotency keys.
- Multi-tenant isolation for cache, circuit breaker, and rate limiting.
- Circuit breaker with tenant-scoped failure tracking.
- Idempotency cache with deterministic keys.
- Audit log with hash chaining for integrity.
- Explicit failure on non-JSON or malformed outputs.
- Provider abstraction for OpenAI, Gemini, and Ollama.
- Optional API key authentication.
- Optional Redis-backed cache and circuit breaker.
- Gemini key rotation with retry/backoff for rate limits.
- License enforcement with signed tokens, tenant binding, and monthly quotas.
- Optional billing webhook after successful execution.
- License enforcement with signed tokens, tenant binding, and monthly quotas.
- Optional billing webhook after successful execution.

## Runtime Flow (Execution Pipeline)
1. Verify tenant ID presence.
2. Enforce per-tenant rate limiting.
3. Enforce queue limits and backpressure (reject or wait).
4. Load behavior YAML and compute content hash.
5. Canonicalize input JSON.
6. Check idempotency cache.
7. Validate input schema (strict).
8. Evaluate deterministic rules (bypass LLM if matched).
9. Build prompt with strict schema instructions.
10. Check tenant circuit breaker state.
11. Enforce timeout budget.
12. Call primary model with retry/backoff.
13. Parse JSON response and fail-fast on non-JSON.
14. Apply missing-field strategy, then strict schema validation.
15. Apply field constraints and cross-field rules.
16. Append audit log hash chain.
17. Cache successful response.
18. Return validated payload.

## Determinism Guarantees
- Canonical JSON serialization with sorted keys.
- UTF-8 NFC normalization.
- Float rounding uses configurable precision and half-even rounding.
- Strict schema enforcement rejects extra fields and type mismatches.
- JSON parsing fails if output is not a JSON object (no partial parsing).
- Deterministic rule outputs are validated with the same schema enforcement.

## Project Structure
- `plumber/app.py`: CLI entry point.
- `plumber/server.py`: FastAPI server for `/health`, `/ready`, `/metrics`, `/execute`.
- `plumber/config.py`: Environment configuration.
- `plumber/logging.py`: Structured logging with optional file sink.
- `plumber/licensing/*`: License token validation and errors.
- `plumber/tenancy/*`: Tenant registry, usage tracking, concurrency.
- `plumber/licensing/*`: License token validation and errors.
- `plumber/tenancy/*`: Tenant registry, usage tracking, concurrency.
- `plumber/metrics.py`: In-memory metrics and Prometheus rendering.
- `plumber/audit.py`: Hash-chained audit logging.
- `plumber/batching.py`: CSV batch execution helper.
- `plumber/runtime/engine.py`: Core pipeline orchestration.
- `plumber/runtime/prompt.py`: PromptBuilder and schema injection.
- `plumber/runtime/validation.py`: Strict schema validation, constraints, and cross-field rules.
- `plumber/runtime/normalization.py`: Canonical JSON and normalization.
- `plumber/runtime/retry.py`: Retry policy with backoff.
- `plumber/runtime/circuit_breaker.py`: Tenant-isolated breaker (memory or Redis).
- `plumber/runtime/cache.py`: Idempotency cache (memory or Redis).
- `plumber/runtime/quota.py`: Token bucket rate limiting.
- `plumber/runtime/redis_backend.py`: Minimal Redis client wrapper.
- `plumber/runtime/providers/*`: Provider integrations.
- `tools/system_test.py`: System test runner with logging.

## Configuration
Environment variables are supported via `plumber/config.py`. Key values:
- `PLUMBER_GLOBAL_TIMEOUT_MS`
- `PLUMBER_PROVIDER_TIMEOUT_MS`
- `PLUMBER_RETRY_BUDGET_MS`
- `PLUMBER_MAX_CONCURRENCY`
- `PLUMBER_QUEUE_LIMIT`
- `PLUMBER_OVERFLOW_BEHAVIOR` (`reject` or `wait`)
- `PLUMBER_RATE_LIMIT_PER_MIN`
- `PLUMBER_FLOAT_PRECISION`
- `PLUMBER_FLOAT_ROUNDING`
- `PLUMBER_CACHE_TTL`
- `PLUMBER_CB_FAILURES`
- `PLUMBER_CB_WINDOW`
- `PLUMBER_CB_COOLDOWN`
- `PLUMBER_LOG_PATH` (optional file sink for logs)
- `PLUMBER_REDIS_URL` (enables Redis backends when set)
- `PLUMBER_CACHE_BACKEND` (`memory` or `redis`)
- `PLUMBER_CB_BACKEND` (`memory` or `redis`)
- `LICENSE_PUBLIC_KEY`
- `PLUMBER_REQUEST_TIMEOUT_MS`
- `PLUMBER_ENV_FILE`
- `PLUMBER_CONFIG_YAML`
- `BILLING_WEBHOOK_URL`
- `LICENSE_PUBLIC_KEY`
- `PLUMBER_REQUEST_TIMEOUT_MS`
- `PLUMBER_ENV_FILE`
- `PLUMBER_CONFIG_YAML`
- `BILLING_WEBHOOK_URL`
- `GEMINI_API_KEYS` (comma-separated rotation list)
- `GEMINI_API_VERSION` (default `v1beta`)
- `GEMINI_MODEL` (default `gemini-2.0-flash`)
- `GEMINI_RETRY_ATTEMPTS` (default `2`)
- `GEMINI_RETRY_BACKOFF_MS` (default `500`)

## Provider Abstraction
All providers implement:
```
generate(system: str, user: str, cfg: dict) -> ProviderResponse
```
ProviderResponse fields:
- `raw_text`
- `token_usage`
- `latency_ms`
- `provider_metadata`

## Behavior Files
Behavior definitions live in `behaviors/*.yaml`. Each behavior declares:
- `name`
- `model` (provider + model)
- `input.schema`
- `output.schema`
- optional `deterministic_rules`
- optional `validation` rules and constraints
- optional `model.fallback` for one-step fallback

## Licensing
License tokens are HMAC-signed JSON with:
- `tenant_id`
- `expiration_date` (ISO8601)
- `max_executions_per_month`
- `allowed_features`
- `version`

All API routes except `/health` require `Authorization: Bearer <license_key>`. The CLI requires `LICENSE_KEY` or `--license`.

## Tenant Quotas
- Monthly execution quota enforced per tenant.
- Per-tenant concurrency enforced at API entry.
- Usage stored in Redis when configured, otherwise in-memory.

## Licensing
License tokens are HMAC-signed JSON with:
- `tenant_id`
- `expiration_date` (ISO8601)
- `max_executions_per_month`
- `allowed_features`
- `version`

All API routes except `/health` require `Authorization: Bearer <license_key>`. The CLI requires `LICENSE_KEY` or `--license`.

## Tenant Quotas
- Monthly execution quota enforced per tenant.
- Per-tenant concurrency enforced at API entry.
- Usage stored in Redis when configured, otherwise in-memory.

## Test Behaviors
These behaviors stress strict validation and schema enforcement:
- `behaviors/summarize_ticket.yaml` (LLM output JSON)
- `behaviors/strict_types.yaml` (strict types)
- `behaviors/array_enum.yaml` (arrays + enum)
- `behaviors/floats.yaml` (float enforcement)
- `behaviors/cross_field.yaml` (cross-field validation)
- `behaviors/missing_nullable.yaml` (missing + nullable)

## System Test
`tools/system_test.py` performs:
- `/health`
- `/execute` across all behaviors
- `/metrics`
Logs are written to `system_test.log`.

### Example Test Log Highlights (2026-02-25)
- `summarize_ticket` runs via Ollama to avoid free-tier Gemini limits.
- Deterministic rules allow full-pass system tests without external calls.

## Billing Hook
When `BILLING_WEBHOOK_URL` is set, a POST is sent after successful execution:
```
{
  "tenant_id": "...",
  "execution_id": "...",
  "behavior_name": "...",
  "cost_estimate": 0.0,
  "latency_ms": 1234,
  "timestamp": 1700000000
}
```
Webhook failures do not affect execution responses.

## Billing Hook
When `BILLING_WEBHOOK_URL` is set, a POST is sent after successful execution:
```
{
  "tenant_id": "...",
  "execution_id": "...",
  "behavior_name": "...",
  "cost_estimate": 0.0,
  "latency_ms": 1234,
  "timestamp": 1700000000
}
```
Webhook failures do not affect execution responses.

## Operational Notes
- Strict JSON parsing intentionally fails when models echo the schema or include non-JSON text.
- If high compliance is required, use a more reliable model or deterministic rules.
- For production, enable Redis to share cache and breaker state across processes.
- Gemini key rotation is supported via `GEMINI_API_KEYS` (comma-separated) with retry/backoff.

## Known Limitations
- Some local models may not reliably emit strict JSON without retries or higher-fidelity prompting.
- This runtime enforces deterministic outputs but does not correct model behavior.

## How To Run
```
python -m plumber.app serve --host 127.0.0.1 --port 8080
python tools/system_test.py
```

## Deployment (Minimal)
- `Dockerfile` for container build.
- `docker-compose.yml` for app + Redis.

## Security and Compliance
- Audit records are hash-chained for tamper-evidence.
- Deterministic redaction support is available in `plumber/logging.py`.
- Tenant isolation for cache, circuit breaker, and rate limiting.
