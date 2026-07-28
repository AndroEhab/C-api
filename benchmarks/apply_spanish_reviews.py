"""Apply review ledger events to derive candidate review state.

Usage:
    python -m benchmarks.apply_spanish_reviews

Reads:
    benchmarks/spanish_boundary_candidates.json
    benchmarks/spanish_boundary_reviews.jsonl

Writes:
    benchmarks/spanish_boundary_candidates.json  (updated)
    benchmarks/spanish_boundary_dataset_report.json  (updated)
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

FIXTURE_PATH = BENCHMARK_DIR / "spanish_boundary_candidates.json"
LEDGER_PATH = BENCHMARK_DIR / "spanish_boundary_reviews.jsonl"
MANIFEST_PATH = BENCHMARK_DIR / "spanish_source_manifest.json"
REFERENCE_PATH = BENCHMARK_DIR / "spanish_reference_manifest.json"
REPORT_PATH = BENCHMARK_DIR / "spanish_boundary_dataset_report.json"
POLICY_FREEZE_PATH = BENCHMARK_DIR / "spanish_policy_freeze.json"

from benchmarks.spanish_benchmark_lib import (
    build_boundary_key,
    derive_review_state,
    generate_dataset_report,
    load_policy_freeze,
    validate_ledger_events,
    validate_label_origin_consistency,
)


def load_ledger(path: Path) -> list[dict]:
    """Load review ledger entries, preserving order."""
    events: list[dict] = []
    if not path.exists() or path.stat().st_size == 0:
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def group_events_by_boundary(
    events: list[dict],
) -> dict[str, list[dict]]:
    """Group review events by boundaryId, preserving chronological order."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        bid = ev.get("boundaryId", "")
        grouped[bid].append(ev)
    return dict(grouped)


def apply_reviews() -> None:
    """Read everything, validate, derive state, write updated fixture and report."""
    # Load fixture
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        entries: list[dict] = json.load(f)

    # Load manifest and reference
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data: dict = json.load(f)
    manifest = manifest_data.get("sources", [])

    with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
        reference: list[dict] = json.load(f)

    # Load policy freeze
    policy_freeze = load_policy_freeze(POLICY_FREEZE_PATH)

    # Build valid boundary IDs
    valid_boundary_ids: set[str] = set()
    for entry in entries:
        valid_boundary_ids.add(build_boundary_key(
            entry["sourceId"], entry["leftCueId"], entry["rightCueId"],
        ))

    # Load and validate ledger
    events = load_ledger(LEDGER_PATH)
    if events:
        validation_errors = validate_ledger_events(events, valid_boundary_ids)
        if validation_errors:
            for err in validation_errors:
                print(f"Ledger validation error: {err}", file=sys.stderr)
            print("ERROR: Ledger validation failed. Fix events before applying.", file=sys.stderr)
            sys.exit(1)

    grouped = group_events_by_boundary(events)

    # Apply derived state
    for entry in entries:
        key = build_boundary_key(
            entry["sourceId"], entry["leftCueId"], entry["rightCueId"]
        )
        boundary_events = grouped.get(key, [])
        state = derive_review_state(boundary_events)

        # Update entry fields
        entry["goldLabel"] = state["goldLabel"]
        entry["labelConfidence"] = state["labelConfidence"]
        entry["reviewerCount"] = state["reviewerCount"]
        entry["needsSecondReview"] = state["needsSecondReview"]
        entry["reviewReason"] = state["reviewReason"]
        entry["reviewStatus"] = state["reviewStatus"]
        entry["labelOrigin"] = state["labelOrigin"]

    # Validate label-origin consistency
    lo_errors = validate_label_origin_consistency(entries)
    if lo_errors:
        for err in lo_errors:
            print(f"Label-origin error: {err}", file=sys.stderr)
        print("ERROR: Label-origin consistency check failed.", file=sys.stderr)
        sys.exit(1)

    # Write updated fixture
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"Updated {len(entries)} entries in {FIXTURE_PATH.name}")

    # Regenerate report using shared function
    report = generate_dataset_report(entries, manifest, reference, policy_freeze)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Regenerated {REPORT_PATH.name}")


def main() -> None:
    apply_reviews()


if __name__ == "__main__":
    main()
