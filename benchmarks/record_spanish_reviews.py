"""Import completed review queue CSV into the review ledger.

Usage:
    python -m benchmarks.record_spanish_reviews \\
        --input review_queue_completed.csv \\
        --reviewer reviewer-a \\
        --round 1

    python -m benchmarks.record_spanish_reviews \\
        --adjudicate \\
        --input adjudication_queue.csv \\
        --reviewer adjudicator-1

Validates every row, builds canonical boundaryId, ensures the boundary
belongs to the supplied queue and split, requires label/confidence/reason,
creates unique reviewId values, and appends events to
spanish_boundary_reviews.jsonl atomically.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

FIXTURE_PATH = BENCHMARK_DIR / "spanish_boundary_candidates.json"
LEDGER_PATH = BENCHMARK_DIR / "spanish_boundary_reviews.jsonl"

from benchmarks.spanish_benchmark_lib import (
    build_boundary_key,
    derive_review_state,
    validate_ledger_events,
)


def load_ledger(path: Path) -> list[dict]:
    """Load ledger entries."""
    events: list[dict] = []
    if not path.exists() or path.stat().st_size == 0:
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_boundary_ids(path: Path) -> set[str]:
    """Load all valid boundary IDs from the fixture."""
    with open(path, "r", encoding="utf-8") as f:
        entries: list[dict] = json.load(f)
    ids: set[str] = set()
    for e in entries:
        ids.add(build_boundary_key(
            e["sourceId"], e["leftCueId"], e["rightCueId"],
        ))
    return ids


def record_reviews(args: argparse.Namespace) -> None:
    # Load valid boundary IDs
    valid_boundary_ids = load_boundary_ids(FIXTURE_PATH)

    # Load existing ledger (for duplicate checking and boundary context)
    existing_events = load_ledger(LEDGER_PATH)
    existing_review_ids: set[str] = set()
    existing_boundary_reviewers: dict[str, set[str]] = {}
    for ev in existing_events:
        rid = ev.get("reviewId", "")
        if rid:
            existing_review_ids.add(rid)
        bid = ev.get("boundaryId", "")
        rev = ev.get("reviewerId", "")
        if bid and rev:
            if bid not in existing_boundary_reviewers:
                existing_boundary_reviewers[bid] = set()
            existing_boundary_reviewers[bid].add(rev)

    # Read CSV
    with open(args.input, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("ERROR: CSV file is empty")
        sys.exit(1)

    # Determine if this is adjudication mode
    is_adjudication = args.adjudicate

    # Validate and build events
    events: list[dict] = []
    errors: list[str] = []
    line_num = 0

    for row in rows:
        line_num += 1
        source_id = row.get("sourceId", "").strip()
        left_cue_id = row.get("leftCueId", "").strip()
        right_cue_id = row.get("rightCueId", "").strip()
        boundary_id = build_boundary_key(source_id, left_cue_id, right_cue_id)
        label = row.get("goldLabel", "").strip() or row.get("label", "").strip()
        confidence = row.get("labelConfidence", "").strip() or row.get("confidence", "").strip()
        reason = row.get("reviewReason", "").strip() or row.get("reason", "").strip()
        created_at = row.get("createdAt", "").strip()

        if not source_id or not left_cue_id or not right_cue_id:
            errors.append(f"Row {line_num}: missing sourceId, leftCueId, or rightCueId")
            continue

        if boundary_id not in valid_boundary_ids:
            errors.append(
                f"Row {line_num}: boundaryId={boundary_id} is not a valid candidate boundary"
            )
            continue

        if not label:
            errors.append(f"Row {line_num}: missing label for boundaryId={boundary_id}")
            continue
        if label not in ("JOIN", "BREAK", "AMBIGUOUS"):
            errors.append(f"Row {line_num}: invalid label={label!r} for boundaryId={boundary_id}")
            continue

        if not confidence:
            errors.append(f"Row {line_num}: missing confidence for boundaryId={boundary_id}")
            continue
        if confidence not in ("high", "medium", "low"):
            errors.append(f"Row {line_num}: invalid confidence={confidence!r} for boundaryId={boundary_id}")
            continue

        if not reason:
            errors.append(f"Row {line_num}: missing reason for boundaryId={boundary_id}")
            continue

        # Generate unique reviewId
        review_id = str(uuid.uuid4())

        if not created_at:
            created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Detect split from CSV or derive from fixture
        csv_split = row.get("split", "").strip()

        if is_adjudication:
            event: dict = {
                "eventType": "adjudication",
                "boundaryId": boundary_id,
                "reviewId": review_id,
                "reviewerId": args.reviewer,
                "label": label,
                "confidence": confidence,
                "reason": reason,
                "createdAt": created_at,
                "labelOrigin": "human",
            }
        else:
            review_round = args.round
            event = {
                "eventType": "review",
                "boundaryId": boundary_id,
                "reviewId": review_id,
                "reviewerId": args.reviewer,
                "reviewRound": review_round,
                "label": label,
                "confidence": confidence,
                "reason": reason,
                "createdAt": created_at,
                "labelOrigin": "human",
            }

        events.append(event)

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        print(f"Failed with {len(errors)} error(s). No events recorded.", file=sys.stderr)
        sys.exit(1)

    # Validate the new events together with existing events
    all_events = existing_events + events
    validation_errors = validate_ledger_events(all_events, valid_boundary_ids)
    if validation_errors:
        for err in validation_errors:
            print(f"Validation error: {err}", file=sys.stderr)
        print("Ledger validation failed. No events recorded.", file=sys.stderr)
        sys.exit(1)

    # Atomic append: write to temp file, then rename
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(LEDGER_PATH.parent),
            prefix=f".{LEDGER_PATH.name}.",
            suffix=".tmp",
        )
        with os.fdopen(fd, "a", encoding="utf-8") as tmp:
            # Copy existing events
            if LEDGER_PATH.exists():
                with open(LEDGER_PATH, "r", encoding="utf-8") as orig:
                    for line in orig:
                        tmp.write(line)

            # Append new events
            for ev in events:
                tmp.write(json.dumps(ev, ensure_ascii=False) + "\n")

        # Atomic rename (os.replace is atomic on same filesystem)
        os.replace(tmp_path, LEDGER_PATH)
    except Exception as exc:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        print(f"ERROR: Failed to write ledger: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Recorded {len(events)} events to {LEDGER_PATH}")
    if is_adjudication:
        print(f"  Mode: adjudication")
    else:
        print(f"  Mode: review (round {args.round})")
    print(f"  Reviewer: {args.reviewer}")
    print(f"  Batch complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import completed review queue CSV into review ledger"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Completed review queue CSV",
    )
    parser.add_argument(
        "--reviewer", type=str, required=True,
        help="Reviewer identifier",
    )
    parser.add_argument(
        "--round", type=int, choices=[1, 2], default=1,
        help="Review round (default: 1, not used with --adjudicate)",
    )
    parser.add_argument(
        "--adjudicate", action="store_true",
        help="Adjudication mode (creates adjudication events instead of review events)",
    )
    args = parser.parse_args()

    if args.adjudicate and args.round != 1:
        print("ERROR: --round must not be used with --adjudicate", file=sys.stderr)
        sys.exit(1)

    record_reviews(args)


if __name__ == "__main__":
    main()
