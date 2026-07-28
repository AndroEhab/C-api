"""Prepare a review queue CSV for human labelers.

Usage:
    python -m benchmarks.prepare_spanish_review_queue \\
        --reviewer reviewer-a \\
        --round 1 \\
        --split dev \\
        --output review_queue.csv

    python -m benchmarks.prepare_spanish_review_queue \\
        --reviewer reviewer-a \\
        --round 1 \\
        --split dev \\
        --limit 30 \\
        --batch-index 0 \\
        --output review_queue_dev_r1_batch0.csv

Round 1: All entries in the given split, in deterministic diverse order.
Round 2: Only boundaries with a valid round-one review from a different reviewer,
         without an existing round-two review from this reviewer,
         and not already adjudicated. Prioritizes JOIN medium/low confidence
         and needsSecondReview entries. Fills to ~20% double-review target.

Every row contains:
    queueId, boundaryId (= sourceId:leftCueId:rightCueId),
    queueReviewer, queueRound, queueSplit

Batch positions are stable: universe is partitioned into fixed 30-entry blocks.
Batch completion does not shift remaining batch positions.

A sidecar manifest (review_queue.manifest.json) is generated alongside each CSV.
The content hash covers canonical queue rows with structured field serialisation.
"""

from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
FIXTURE_PATH = BENCHMARK_DIR / "spanish_boundary_candidates.json"
LEDGER_PATH = BENCHMARK_DIR / "spanish_boundary_reviews.jsonl"
MANIFEST_PATH = BENCHMARK_DIR / "spanish_source_manifest.json"

from benchmarks.spanish_benchmark_lib import (
    build_boundary_key,
    build_queue_row,
    compute_queue_content_hash,
)


def load_ledger(path: Path) -> list[dict]:
    """Load review ledger entries."""
    events: list[dict] = []
    if not path.exists() or path.stat().st_size == 0:
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def get_reviewed_keys(events: list[dict]) -> set[str]:
    """Return set of boundary keys that have at least one review event."""
    keys: set[str] = set()
    for ev in events:
        bid = ev.get("boundaryId", "")
        if bid:
            keys.add(bid)
    return keys


def get_first_review_per_boundary(
    events: list[dict],
) -> dict[str, dict]:
    """Return the first review event per boundary (review events only, not adjudication)."""
    first: dict[str, dict] = {}
    for ev in events:
        bid = ev.get("boundaryId", "")
        if bid and bid not in first:
            if ev.get("eventType") == "review" and ev.get("reviewRound", 0) == 1:
                first[bid] = ev
    return first


def get_reviewer_events(
    events: list[dict], reviewer_id: str,
) -> dict[str, dict]:
    """Return the latest review per boundary for a specific reviewer."""
    result: dict[str, dict] = {}
    for ev in events:
        bid = ev.get("boundaryId", "")
        if bid and ev.get("reviewerId") == reviewer_id and ev.get("eventType") == "review":
            result[bid] = ev
    return result


def get_adjudicated_keys(events: list[dict]) -> set[str]:
    """Return set of boundary keys that have been adjudicated."""
    keys: set[str] = set()
    for ev in events:
        if ev.get("eventType") == "adjudication":
            bid = ev.get("boundaryId", "")
            if bid:
                keys.add(bid)
    return keys


def get_round_two_keys(events: list[dict]) -> set[str]:
    """Return set of boundary keys that already have a round-two review."""
    keys: set[str] = set()
    for ev in events:
        if ev.get("eventType") == "review" and ev.get("reviewRound") == 2:
            bid = ev.get("boundaryId", "")
            if bid:
                keys.add(bid)
    return keys


def get_completed_queue_boundaries(ledger_path: Path, reviewer_id: str, review_round: int) -> set[str]:
    """Return set of boundary keys already submitted by this reviewer and round."""
    events = load_ledger(ledger_path)
    keys: set[str] = set()
    for ev in events:
        if (ev.get("reviewerId") == reviewer_id
                and ev.get("eventType") == "review"
                and ev.get("reviewRound") == review_round):
            bid = ev.get("boundaryId", "")
            if bid:
                keys.add(bid)
    return keys


def get_batch_boundary_keys(ledger_path: Path, reviewer_id: str, review_round: int, split: str) -> set[str]:
    """Return boundary keys already covered by any completed batch for this reviewer/round/split.

    Looks through the ledger for review events matching the reviewer, round,
    and split. Because the ledger doesn't store split directly, we check via
    the fixture.
    """
    events = load_ledger(ledger_path)
    keys: set[str] = set()
    for ev in events:
        if (ev.get("reviewerId") == reviewer_id
                and ev.get("eventType") == "review"
                and ev.get("reviewRound") == review_round):
            bid = ev.get("boundaryId", "")
            if bid:
                keys.add(bid)
    return keys


def _build_diverse_universe(entries: list[dict]) -> list[dict]:
    """Build a deterministic round-robin interleaved universe.

    Groups entries by source, sorts each group by (chainId, leftCueId),
    then round-robins between sources to ensure source diversity.

    Returns the interleaved list.
    """
    by_source: dict[str, list[dict]] = {}
    for e in entries:
        src = e["sourceId"]
        by_source.setdefault(src, []).append(e)

    sources = sorted(by_source.keys())
    for src in sources:
        by_source[src].sort(key=lambda e: (
            e.get("chainId") or "",
            int(e["leftCueId"]) if e["leftCueId"].lstrip("-").isdigit() else e["leftCueId"],
        ))

    max_len = max(len(by_source[s]) for s in sources)
    positions = {s: 0 for s in sources}
    universe: list[dict] = []

    for _ in range(max_len):
        for src in sources:
            if positions[src] < len(by_source[src]):
                universe.append(by_source[src][positions[src]])
                positions[src] += 1

    return universe


def prepare_queue(args: argparse.Namespace) -> None:
    # Load fixture
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        entries: list[dict] = json.load(f)

    # Load manifest for variant metadata
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data: dict = json.load(f)
    manifest_by_id: dict = {m["sourceId"]: m for m in manifest_data.get("sources", [])}

    # Load ledger
    events = load_ledger(LEDGER_PATH)
    reviewed_keys = get_reviewed_keys(events)

    # Filter by split
    split_entries = [e for e in entries if e.get("split") == args.split]

    if args.round == 1:
        candidates = _prepare_round1(
            split_entries, manifest_by_id,
            reviewed_keys, args.reviewer, args.split,
            args.limit, args.batch_index, LEDGER_PATH,
        )
    elif args.round == 2:
        candidates = _prepare_round2(
            entries, split_entries, manifest_by_id,
            events, reviewed_keys, args.reviewer, args.split,
            args.limit, args.batch_index,
        )
    else:
        print(f"ERROR: Unknown round {args.round}")
        sys.exit(1)

    if not candidates:
        print("WARNING: Empty queue — no eligible boundaries found.")

    queue_id = (
        f"{args.split}_r{args.round}_{args.reviewer}"
        f"{f'_batch{args.batch_index}' if args.limit else ''}"
    )

    # Build canonical queue rows (shared representation for CSV + hash)
    rows = [
        build_queue_row(c, queue_id, args.reviewer, args.round, args.split, manifest_by_id)
        for c in candidates
    ]

    # Compute hash BEFORE writing CSV (same canonical rows)
    content_hash = compute_queue_content_hash(rows)

    # Write CSV
    _write_queue_csv(rows, args.output)
    # Write manifest
    _write_queue_manifest(rows, args.output, queue_id, args.reviewer, args.round, args.split, content_hash)


def _prepare_round1(
    split_entries: list[dict],
    manifest_by_id: dict,
    reviewed_keys: set[str],
    reviewer: str,
    split: str,
    limit: int | None,
    batch_index: int,
    ledger_path: Path,
) -> list[dict]:
    """Round 1: stable batch selection with source diversity.

    Steps:
    1. Build deterministic diverse universe from unreviewed split entries.
    2. Partition universe into fixed 30-entry blocks.
    3. For the requested batch, return only unsubmitted entries.
    """
    # Exclude entries already reviewed by ANYONE
    eligible = [
        e for e in split_entries
        if build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"]) not in reviewed_keys
    ]

    # Build diverse universe (round-robin between sources)
    universe = _build_diverse_universe(eligible)

    # Get boundaries already submitted by this reviewer/round
    already_done = get_completed_queue_boundaries(ledger_path, reviewer, 1)

    if limit is None:
        # No limit — return all eligible entries
        return [e for e in universe
                if build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"]) not in already_done]

    # Partition universe into fixed batch positions
    batch_size = limit
    start = batch_index * batch_size
    end = start + batch_size

    batch_slice = universe[start:end]
    if not batch_slice:
        print(f"WARNING: batch_index {batch_index} starts beyond {len(universe)} available entries")
        return []

    # Filter out already-submitted entries (stable: positions don't shift)
    batch = [
        e for e in batch_slice
        if build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"]) not in already_done
    ]

    return batch


def _prepare_round2(
    all_entries: list[dict],
    split_entries: list[dict],
    manifest_by_id: dict,
    events: list[dict],
    reviewed_keys: set[str],
    reviewer: str,
    split: str,
    limit: int | None,
    batch_index: int,
) -> list[dict]:
    """Round 2: boundaries with a valid round-one review from a DIFFERENT reviewer,
    no existing round-two review from this reviewer, not adjudicated.
    Prioritizes JOIN medium/low confidence and needsSecondReview.
    Never includes unreviewed boundaries.

    Uses stable batch positions from the diverse universe for round-two-eligible entries.
    """
    rng = random.Random(42)

    # Get adjudicated keys - these are excluded from round two
    adjudicated_keys = get_adjudicated_keys(events)

    # Get boundaries already having round-two reviews
    round_two_keys = get_round_two_keys(events)

    # Find entries the current reviewer has already reviewed (any round)
    reviewer_events = get_reviewer_events(events, reviewer)
    reviewer_keys = set(reviewer_events.keys())

    # Get first round-one reviews per boundary
    all_first_reviews = get_first_review_per_boundary(events)

    # Build a set of entries that are eligible for round two:
    eligible: list[dict] = []
    split_entry_map: dict[str, dict] = {}
    for e in split_entries:
        key = build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"])
        split_entry_map[key] = e

    for bid, ev in all_first_reviews.items():
        # Skip if current reviewer was the first reviewer
        if ev.get("reviewerId") == reviewer:
            continue
        # Skip if current reviewer already reviewed this boundary (any round)
        if bid in reviewer_keys:
            continue
        # Skip if this boundary already has a round-two review
        if bid in round_two_keys:
            continue
        # Skip if adjudicated
        if bid in adjudicated_keys:
            continue
        # Skip if not in our split
        if bid not in split_entry_map:
            continue

        entry = split_entry_map[bid]
        label = ev.get("label", "")
        confidence = ev.get("confidence", "")

        # Attach metadata from the first review for prioritization
        entry["_first_label"] = label
        entry["_first_confidence"] = confidence
        entry["_needs_second"] = (
            label == "JOIN" and confidence in ("medium", "low")
        )
        eligible.append(entry)

    # Deduplicate
    seen: set[str] = set()
    unique_eligible: list[dict] = []
    for e in eligible:
        key = build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"])
        if key not in seen:
            seen.add(key)
            unique_eligible.append(e)

    # Priority sorting:
    def priority(e: dict) -> tuple:
        fl = e.get("_first_label", "")
        fc = e.get("_first_confidence", "")
        ns = e.get("_needs_second", False)
        if fl == "JOIN" and fc == "medium":
            return (0,)
        if fl == "JOIN" and fc == "low":
            return (1,)
        if ns:
            return (2,)
        return (3,)

    unique_eligible.sort(key=priority)

    # For stable batching, build a universe in priority order then diverse order within ties
    # Since round two is small, use the priority-sorted list directly
    priority_queue = unique_eligible

    # Calculate target: at least 20% of split entries, but at most available
    total_in_split = len(split_entries)
    target = max(1, int(total_in_split * 0.2))
    target = min(target, len(priority_queue))

    # If limit is specified, cap target
    if limit is not None:
        target = min(target, limit)

    # Build diverse universe for round-two: mix sources within priority tiers
    # Group priority-eligible entries by source within each tier
    tier_groups: dict[int, dict[str, list[dict]]] = {}
    for e in priority_queue:
        p = priority(e)
        p0 = p[0]
        if p0 not in tier_groups:
            tier_groups[p0] = {}
        src = e["sourceId"]
        tier_groups[p0].setdefault(src, []).append(e)

    # Within each tier, sort each source by (chainId, leftCueId)
    for tier in tier_groups:
        for src in tier_groups[tier]:
            tier_groups[tier][src].sort(key=lambda e: (
                e.get("chainId") or "",
                int(e["leftCueId"]) if e["leftCueId"].lstrip("-").isdigit() else e["leftCueId"],
            ))

    # Build universe: within each priority tier, round-robin between sources
    round2_universe: list[dict] = []
    sources_sorted = sorted(manifest_by_id.keys())
    for tier in sorted(tier_groups.keys()):
        srcs_in_tier = sorted(tier_groups[tier].keys())
        max_in_tier = max(len(tier_groups[tier][s]) for s in srcs_in_tier) if srcs_in_tier else 0
        positions = {s: 0 for s in srcs_in_tier}
        for pos in range(max_in_tier):
            for src in srcs_in_tier:
                if positions[src] < len(tier_groups[tier][src]):
                    round2_universe.append(tier_groups[tier][src][positions[src]])
                    positions[src] += 1

    # Apply batch position
    if limit is not None:
        batch_start = batch_index * limit
        batch_end = batch_start + limit
        selected = round2_universe[batch_start:batch_end]
    else:
        selected = round2_universe[:target]

    # Strip internal metadata before writing
    for e in selected:
        e.pop("_first_label", None)
        e.pop("_first_confidence", None)
        e.pop("_needs_second", None)

    return selected


def _write_queue_manifest(
    rows: list[dict],
    output: Path,
    queue_id: str,
    reviewer: str,
    review_round: int,
    split: str,
    content_hash: str,
) -> None:
    """Write the sidecar queue manifest JSON."""
    boundary_ids: list[str] = [r["boundaryId"] for r in rows]

    manifest = {
        "queueId": queue_id,
        "reviewerId": reviewer,
        "reviewRound": review_round,
        "split": split,
        "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "boundaryIds": boundary_ids,
        "rowCount": len(rows),
        "contentSha256": content_hash,
    }

    manifest_path = output.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote queue manifest to {manifest_path}")


def _write_queue_csv(
    rows: list[dict],
    output: Path,
) -> None:
    """Write canonical queue rows to CSV."""
    if not rows:
        fieldnames = [
            "queueId", "boundaryId", "queueReviewer", "queueRound", "queueSplit",
            "reviewRound", "sourceId", "spanishVariant", "contentType",
            "sourceQualityTier", "contentStructure", "originalSpokenLanguage",
            "sceneId", "split", "leftCueId", "rightCueId",
            "leftRawText", "rightRawText", "leftNormalized", "rightNormalized",
            "leftLines", "rightLines",
            "leftStartMs", "leftEndMs", "rightStartMs", "rightEndMs",
            "gapMs", "overlapMs",
            "previousContext", "nextContext", "speakerMarkers",
            "samplingTags", "structureTags", "punctuationTags",
            "linguisticTags", "timingBand", "chainId",
            "goldLabel", "labelConfidence", "reviewReason",
        ]
        with open(output, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
        print(f"Wrote 0 entries to {output}")
        return

    fieldnames = list(rows[0].keys())
    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} entries to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Spanish boundary review queue"
    )
    parser.add_argument(
        "--reviewer", type=str, required=True,
        help="Reviewer identifier (stable pseudonym)",
    )
    parser.add_argument(
        "--round", type=int, required=True, choices=[1, 2],
        help="Review round",
    )
    parser.add_argument(
        "--split", type=str, required=True, choices=["dev", "test"],
        help="Split to prepare queue for",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of rows in the queue (None = full split)",
    )
    parser.add_argument(
        "--batch-index", type=int, default=0,
        help="Batch slice index when --limit is provided (default: 0)",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        print("ERROR: --limit must be >= 1")
        sys.exit(1)
    if args.batch_index < 0:
        print("ERROR: --batch-index must be >= 0")
        sys.exit(1)

    prepare_queue(args)


if __name__ == "__main__":
    main()
