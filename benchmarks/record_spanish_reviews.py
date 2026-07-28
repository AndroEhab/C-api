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

Validates every row against the queue manifest, builds canonical boundaryId,
ensures the boundary belongs to the supplied queue and split, requires
label/confidence/reason, creates unique reviewId values, and appends events
to spanish_boundary_reviews.jsonl with concurrency-safe writes.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
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
    is_valid_frozen_policy,
    validate_ledger_events,
)


# ── helpers ────────────────────────────────────────────────────────────────


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


def load_fixture_map(path: Path) -> dict[str, dict]:
    """Load fixture entries keyed by boundaryId."""
    with open(path, "r", encoding="utf-8") as f:
        entries: list[dict] = json.load(f)
    result: dict[str, dict] = {}
    for e in entries:
        bid = build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"])
        result[bid] = e
    return result


def load_queue_manifest(manifest_path: Path) -> dict:
    """Load a queue manifest JSON file."""
    if not manifest_path.exists():
        print(f"ERROR: Queue manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_queue_content_hash(rows: list[dict]) -> str:
    """Compute SHA-256 of ordered queue content, excluding mutable review fields.

    Must match the hash computed by prepare_spanish_review_queue.
    """
    canonical_rows: list[dict] = []
    for row in rows:
        r = dict(row)
        # Strip mutable review fields
        r.pop("goldLabel", None)
        r.pop("labelConfidence", None)
        r.pop("reviewReason", None)
        canonical_rows.append({k: r[k] for k in sorted(r.keys())})
    content = json.dumps(canonical_rows, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _lock_path() -> Path:
    """Return the lock file path for the ledger."""
    return LEDGER_PATH.with_suffix(".jsonl.lock")


def _acquire_lock(lock_path: Path, timeout: float = 5.0) -> bool:
    """Try to acquire an exclusive lock via mkdir (atomic on all platforms).

    Returns True if the lock was acquired, False if timeout expired.
    """
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            lock_path.mkdir(exist_ok=False)
            return True
        except FileExistsError:
            time.sleep(0.05)
    return False


def _release_lock(lock_path: Path) -> None:
    """Release the lock."""
    try:
        lock_path.rmdir()
    except OSError:
        pass


# ── main ────────────────────────────────────────────────────────────────────


def record_reviews(args: argparse.Namespace) -> None:
    # Load valid boundary IDs and fixture data
    valid_boundary_ids = load_boundary_ids(FIXTURE_PATH)
    fixture_map = load_fixture_map(FIXTURE_PATH)

    # Derive manifest path from input CSV
    input_path = Path(args.input)
    manifest_path = input_path.with_suffix(".manifest.json")
    manifest = load_queue_manifest(manifest_path)

    queue_id = manifest.get("queueId", "")
    manifest_reviewer = manifest.get("reviewerId", "")
    manifest_round = manifest.get("reviewRound", 1)
    manifest_split = manifest.get("split", "")
    manifest_boundary_ids = manifest.get("boundaryIds", [])
    manifest_hash = manifest.get("contentSha256", "")
    manifest_row_count = manifest.get("rowCount", 0)

    # validate queue identity
    if not queue_id:
        print("ERROR: Queue manifest has no queueId", file=sys.stderr)
        sys.exit(1)

    if manifest_reviewer != args.reviewer:
        print(
            f"ERROR: Manifest reviewer ({manifest_reviewer!r}) does not match "
            f"--reviewer ({args.reviewer!r})",
            file=sys.stderr,
        )
        sys.exit(1)

    if manifest_round != args.round:
        print(
            f"ERROR: Manifest round ({manifest_round}) does not match "
            f"--round ({args.round})",
            file=sys.stderr,
        )
        sys.exit(1)

    is_adjudication = args.adjudicate

    # Read CSV
    with open(input_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("ERROR: CSV file is empty", file=sys.stderr)
        sys.exit(1)

    # Validate and build events
    events: list[dict] = []
    errors: list[str] = []
    line_num = 0
    seen_boundary_ids_in_batch: set[str] = set()

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
        row_queue_id = row.get("queueId", "").strip()
        row_queue_reviewer = row.get("queueReviewer", "").strip()
        row_queue_round = row.get("queueRound", "").strip()
        row_queue_split = row.get("queueSplit", "").strip()
        row_boundary_id = row.get("boundaryId", "").strip()

        # 1. Check queue identity in row matches manifest
        if row_queue_id and row_queue_id != queue_id:
            errors.append(f"Row {line_num}: queueId {row_queue_id!r} does not match manifest {queue_id!r}")
            continue

        if row_queue_reviewer and row_queue_reviewer != args.reviewer:
            errors.append(f"Row {line_num}: queueReviewer {row_queue_reviewer!r} does not match --reviewer {args.reviewer!r}")
            continue

        if row_queue_round:
            try:
                row_round = int(row_queue_round)
                if row_round != args.round:
                    errors.append(f"Row {line_num}: queueRound {row_round} does not match --round {args.round}")
                    continue
            except ValueError:
                pass

        if row_queue_split and row_queue_split != manifest_split:
            errors.append(f"Row {line_num}: queueSplit {row_queue_split!r} does not match manifest split {manifest_split!r}")
            continue

        # 2. Check boundaryId consistency
        if row_boundary_id and row_boundary_id != boundary_id:
            errors.append(
                f"Row {line_num}: boundaryId {row_boundary_id!r} does not match "
                f"sourceId:leftCueId:rightCueId ({boundary_id!r})"
            )
            continue

        # 3. Boundary must be in the fixture
        if boundary_id not in valid_boundary_ids:
            errors.append(
                f"Row {line_num}: boundaryId={boundary_id} is not a valid candidate boundary"
            )
            continue

        # 4. Boundary must be in this exact queue manifest
        if boundary_id not in set(manifest_boundary_ids):
            errors.append(
                f"Row {line_num}: boundaryId={boundary_id} is not in queue {queue_id}"
            )
            continue

        # 5. No duplicate boundary rows in this batch
        if boundary_id in seen_boundary_ids_in_batch:
            errors.append(
                f"Row {line_num}: duplicate boundaryId={boundary_id} in this batch"
            )
            continue
        seen_boundary_ids_in_batch.add(boundary_id)

        # 6. sourceId, leftCueId, rightCueId must be consistent with boundaryId
        if not source_id or not left_cue_id or not right_cue_id:
            errors.append(f"Row {line_num}: missing sourceId, leftCueId, or rightCueId")
            continue

        # 7. Verify immutable cue text and context match fixture
        fixture_entry = fixture_map.get(boundary_id)
        if fixture_entry:
            for field in ("leftRawText", "rightRawText", "leftNormalized", "rightNormalized",
                          "leftLines", "rightLines"):
                csv_val = row.get(field, "").strip()
                fixture_val = str(json.dumps(fixture_entry.get(field), ensure_ascii=False) if isinstance(fixture_entry.get(field), (list, dict))
                                  else (fixture_entry.get(field) or ""))
                # Compare after stripping JSON encoding
                clean_csv = csv_val.strip('"')
                if csv_val and clean_csv != fixture_val.strip('"'):
                    errors.append(
                        f"Row {line_num}: {field} modified for boundaryId={boundary_id}. "
                        f"CSV value does not match fixture."
                    )
                    break

        if errors:
            continue

        # 8. Label, confidence, reason
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

    # Acquire exclusive lock for concurrency-safe write
    lock_path = _lock_path()
    if not _acquire_lock(lock_path, timeout=10.0):
        print(
            "ERROR: Could not acquire ledger lock (concurrent write in progress).",
            file=sys.stderr,
        )
        sys.exit(1)

    tmp_path: Path | None = None
    try:
        # Load existing ledger under lock to prevent TOCTOU
        existing_events = load_ledger(LEDGER_PATH)

        # Check that no boundary in this batch was already submitted by same reviewer/round
        for ev in existing_events:
            if ev.get("reviewerId") == args.reviewer and ev.get("eventType") == "review":
                if ev.get("reviewRound") == args.round:
                    ev_bid = ev.get("boundaryId", "")
                    for new_ev in events:
                        if new_ev.get("boundaryId") == ev_bid:
                            errors.append(
                                f"boundaryId={ev_bid}: already submitted by reviewer "
                                f"{args.reviewer} for round {args.round}"
                            )

        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            print("Duplicate review detected. No events recorded.", file=sys.stderr)
            sys.exit(1)

        # Check queue row count vs expected
        if manifest_row_count and len(events) != manifest_row_count:
            print(
                f"ERROR: CSV has {len(events)} rows but manifest declares {manifest_row_count}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Validate content hash against manifest
        computed_hash = compute_queue_content_hash(rows)
        if manifest_hash and computed_hash != manifest_hash:
            print(
                f"ERROR: Queue content hash mismatch. Manifest: {manifest_hash}, "
                f"computed: {computed_hash}. Queue content may have been modified.",
                file=sys.stderr,
            )
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
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(LEDGER_PATH.parent),
            prefix=f".{LEDGER_PATH.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_path_str)
        with os.fdopen(fd, "a", encoding="utf-8") as tmp:
            # Copy existing events byte-for-byte
            if LEDGER_PATH.exists():
                with open(LEDGER_PATH, "r", encoding="utf-8") as orig:
                    for line in orig:
                        tmp.write(line)

            # Append new events
            for ev in events:
                tmp.write(json.dumps(ev, ensure_ascii=False) + "\n")

        # Atomic rename (os.replace is atomic on same filesystem)
        os.replace(tmp_path_str, LEDGER_PATH)
        tmp_path = None  # successfully written, prevent cleanup
    except Exception as exc:
        # Clean up temp file on failure
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        print(f"ERROR: Failed to write ledger: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        _release_lock(lock_path)

    print(f"Recorded {len(events)} events to {LEDGER_PATH}")
    if is_adjudication:
        print(f"  Mode: adjudication")
    else:
        print(f"  Mode: review (round {args.round})")
    print(f"  Reviewer: {args.reviewer}")
    print(f"  Queue: {queue_id}")
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
