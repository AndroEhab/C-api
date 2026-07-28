"""Prepare a review queue CSV for human labelers.

Usage:
    python -m benchmarks.prepare_spanish_review_queue \\
        --reviewer reviewer-a \\
        --round 1 \\
        --split dev \\
        --output review_queue.csv

Round 1: All entries in the given split, in deterministic order.
Round 2: Only boundaries with a valid round-one review from a different reviewer,
         without an existing round-two review from this reviewer,
         and not already adjudicated. Prioritizes JOIN medium/low confidence
         and needsSecondReview entries. Fills to ~20% double-review target.
"""

from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
FIXTURE_PATH = BENCHMARK_DIR / "spanish_boundary_candidates.json"
LEDGER_PATH = BENCHMARK_DIR / "spanish_boundary_reviews.jsonl"
MANIFEST_PATH = BENCHMARK_DIR / "spanish_source_manifest.json"


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


def build_boundary_key(source_id: str, left_cue_id: str, right_cue_id: str) -> str:
    return f"{source_id}:{left_cue_id}:{right_cue_id}"


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
        _prepare_round1(
            split_entries, manifest_by_id,
            reviewed_keys, args.reviewer, args.output,
        )
    elif args.round == 2:
        _prepare_round2(
            entries, split_entries, manifest_by_id,
            events, reviewed_keys, args.reviewer, args.output,
        )
    else:
        print(f"ERROR: Unknown round {args.round}")
        sys.exit(1)


def _prepare_round1(
    split_entries: list[dict],
    manifest_by_id: dict,
    reviewed_keys: set[str],
    reviewer: str,
    output: Path,
) -> None:
    """Round 1: all unreviewed entries in split, deterministic order."""
    # Exclude already reviewed
    queue = [
        e for e in split_entries
        if build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"]) not in reviewed_keys
    ]

    # Sort by source, then leftCueId for deterministic order
    queue.sort(key=lambda e: (e["sourceId"], int(e["leftCueId"]) if e["leftCueId"].lstrip("-").isdigit() else e["leftCueId"]))

    _write_queue_csv(queue, manifest_by_id, output, round_num=1, hide_previous=False)


def _prepare_round2(
    all_entries: list[dict],
    split_entries: list[dict],
    manifest_by_id: dict,
    events: list[dict],
    reviewed_keys: set[str],
    reviewer: str,
    output: Path,
) -> None:
    """Round 2: boundaries with a valid round-one review from a DIFFERENT reviewer,
    no existing round-two review from this reviewer, not adjudicated.
    Prioritizes JOIN medium/low confidence and needsSecondReview.
    Never includes unreviewed boundaries.
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
    # 1. Have a valid round-one review
    # 2. Reviewed by a DIFFERENT reviewer (not the current one)
    # 3. Current reviewer hasn't already done round two on this
    # 4. Not already adjudicated
    # 5. In the target split
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
        # (the first reviewer's label/confidence/reason/identity are NOT exposed in output)
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
    # 1. JOIN + medium confidence
    # 2. JOIN + low confidence
    # 3. needsSecondReview (other reasons)
    # 4. All other eligible boundaries
    def priority(e: dict) -> tuple:
        fl = e.get("_first_label", "")
        fc = e.get("_first_confidence", "")
        ns = e.get("_needs_second", False)
        # Priority 0: JOIN + medium
        if fl == "JOIN" and fc == "medium":
            return (0,)
        # Priority 1: JOIN + low
        if fl == "JOIN" and fc == "low":
            return (1,)
        # Priority 2: needsSecondReview (other)
        if ns:
            return (2,)
        # Priority 3: everything else eligible
        return (3,)

    unique_eligible.sort(key=priority)

    # Calculate target: at least 20% of split entries, but at most available
    total_in_split = len(split_entries)
    target = max(1, int(total_in_split * 0.2))
    target = min(target, len(unique_eligible))

    # Select the top-priority entries
    selected_keys: set[str] = set()
    queue: list[dict] = []

    for e in unique_eligible:
        if len(queue) >= target:
            break
        key = build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"])
        if key not in selected_keys:
            selected_keys.add(key)
            queue.append(e)

    # If we still have room, add more eligible boundaries (deterministic)
    remaining = [e for e in unique_eligible
                 if build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"]) not in selected_keys]
    rng.shuffle(remaining)

    for e in remaining:
        if len(queue) >= target:
            break
        key = build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"])
        if key not in selected_keys:
            selected_keys.add(key)
            queue.append(e)

    # Strip internal metadata before writing
    for e in queue:
        e.pop("_first_label", None)
        e.pop("_first_confidence", None)
        e.pop("_needs_second", None)

    # Sort for deterministic output
    queue.sort(key=lambda e: (e["sourceId"], int(e["leftCueId"]) if e["leftCueId"].lstrip("-").isdigit() else e["leftCueId"]))

    _write_queue_csv(queue, manifest_by_id, output, round_num=2, hide_previous=True)


def _write_queue_csv(
    entries: list[dict],
    manifest_by_id: dict,
    output: Path,
    round_num: int,
    hide_previous: bool,
) -> None:
    """Write the review queue CSV with all context needed for decisions."""
    fieldnames = [
        "reviewRound",
        "sourceId", "spanishVariant", "contentType", "sourceQualityTier",
        "contentStructure", "originalSpokenLanguage",
        "sceneId", "split",
        "leftCueId", "rightCueId",
        "leftRawText", "rightRawText",
        "leftNormalized", "rightNormalized",
        "leftLines", "rightLines",
        "leftStartMs", "leftEndMs", "rightStartMs", "rightEndMs",
        "gapMs", "overlapMs",
        "previousContext", "nextContext",
        "speakerMarkers",
        "samplingTags", "structureTags", "punctuationTags",
        "linguisticTags", "timingBand",
        "chainId",
        # Review fields - empty for the labeler to fill
        "goldLabel", "labelConfidence", "reviewReason",
    ]

    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for e in entries:
            row = dict(e)
            row["reviewRound"] = round_num

            # Add manifest metadata
            src = e["sourceId"]
            m = manifest_by_id.get(src, {})
            row["spanishVariant"] = m.get("spanishVariant", "unknown")
            row["contentType"] = m.get("contentType", "other")
            row["sourceQualityTier"] = m.get("sourceQualityTier", "unknown")

            # JSON-encode structured fields
            for field in ("previousContext", "nextContext",
                          "speakerMarkers", "samplingTags",
                          "structureTags", "punctuationTags",
                          "linguisticTags", "leftLines", "rightLines"):
                if field in row:
                    row[field] = json.dumps(row[field], ensure_ascii=False)

            # Empty review fields for labeler
            row["goldLabel"] = ""
            row["labelConfidence"] = ""
            row["reviewReason"] = ""

            if row.get("chainId") is None:
                row["chainId"] = ""

            writer.writerow(row)

    print(f"Wrote {len(entries)} entries to {output}")
    print(f"  Round: {round_num}")
    print(f"  Previous labels hidden: {hide_previous}")


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
    args = parser.parse_args()
    prepare_queue(args)


if __name__ == "__main__":
    main()
