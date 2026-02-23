import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from app import run_behavior, OutputContractError, ValidationRuleError, ModelExecutionError

INPUT_FILE   = "leads.csv"
OUTPUT_FILE  = "leads_scored.csv"
REJECT_FILE  = "leads_rejected.csv"

# ── Performance Tuning ──────────────────────────────────────
MAX_WORKERS = 6      # good for phi3:mini / 6–8 CPU cores
BATCH_SIZE  = 20     # number of concurrent jobs to queue

accepted_rows = []
rejected_rows = []

start_time = time.time()

# ── Worker Function ─────────────────────────────────────────
def process_lead(row, index):
    try:
        result, trace = run_behavior(
            "triage_lead",
            {
                "source": row["source"],
                "content": row["content"]
            }
        )

        enriched = {**row, **result}

        return ("accepted", enriched)

    except (OutputContractError, ValidationRuleError) as e:
        return ("rejected", {
            **row,
            "error": str(e),
            "reason": "contract_violation"
        })

    except ModelExecutionError as e:
        return ("rejected", {
            **row,
            "error": str(e),
            "reason": "model_failure"
        })

    except Exception as e:
        return ("rejected", {
            **row,
            "error": str(e),
            "reason": "unknown_error"
        })


# ── Load Input ──────────────────────────────────────────────
with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))

print(f"\nLoaded {len(all_rows)} leads")

# ── Batched Parallel Execution ──────────────────────────────
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    for i in range(0, len(all_rows), BATCH_SIZE):
        batch = all_rows[i:i+BATCH_SIZE]
        print(f"\nProcessing batch {i//BATCH_SIZE + 1} ({len(batch)} leads)")

        futures = {
            executor.submit(process_lead, row, i+j): row
            for j, row in enumerate(batch)
        }

        for future in as_completed(futures):
            status, data = future.result()

            if status == "accepted":
                accepted_rows.append(data)
            else:
                rejected_rows.append(data)


# ── Write Accepted Output ───────────────────────────────────
if accepted_rows:
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=accepted_rows[0].keys())
        writer.writeheader()
        writer.writerows(accepted_rows)

# ── Write Rejected Output ───────────────────────────────────
if rejected_rows:
    with open(REJECT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rejected_rows[0].keys())
        writer.writeheader()
        writer.writerows(rejected_rows)

# ── Stats ───────────────────────────────────────────────────
elapsed = time.time() - start_time

print("\n────────────────────────────────────────")
print("Plumber Lead Processing Complete")
print("────────────────────────────────────────")
print(f"Accepted : {len(accepted_rows)}")
print(f"Rejected : {len(rejected_rows)}")
print(f"Time     : {elapsed:.2f}s")
print(f"Rate     : {len(all_rows)/elapsed:.2f} leads/sec")

if accepted_rows:
    print(f"\nScored leads → {OUTPUT_FILE}")

if rejected_rows:
    print(f"Rejected leads → {REJECT_FILE}")