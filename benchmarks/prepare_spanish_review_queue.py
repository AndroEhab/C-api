"""Prepare a review queue CSV for human labelers.

Usage:
    python -m benchmarks.prepare_spanish_review_queue \\
        --reviewer reviewer-a \\
        --round 1 \\
        --split dev \\
        --output review_queue.csv

Round 1: All entries in the given split, in deterministic order.
Round 2: Prioritizes JOIN entries with medium/low confidence from round 1,
         hides the first reviewer's label/confidence, fills to 20% coverage.

Round 1 excludes entries already reviewed in the ledger.
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
    """Return the first review event per boundary."""
    first: dict[str, dict] = {}
    for ev in events:
        bid = ev.get("boundaryId", "")
        if bid and bid not in first:
            # Only regular reviews, not adjudications
            if ev.get("reviewRound", 0) >= 1:
                first[bid] = ev
    return first


def get_reviewer_events(
    events: list[dict], reviewer_id: str,
) -> dict[str, dict]:
    """Return the latest review per boundary for a specific reviewer."""
    result: dict[str, dict] = {}
    for ev in events:
        bid = ev.get("boundaryId", "")
        if bid and ev.get("reviewerId") == reviewer_id and ev.get("reviewRound", 0) >= 1:
            result[bid] = ev
    return result


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
    """Round 2: prioritize JOIN medium/low confidence, hide first reviewer's labels."""
    rng = random.Random(42)

    # Find entries the current reviewer has already reviewed
    reviewer_events = get_reviewer_events(events, reviewer)
    reviewer_keys = set(reviewer_events.keys())

    # Find entries that need second review:
    # JOIN entries with medium/low confidence from OTHER reviewers
    all_first_reviews = get_first_review_per_boundary(events)

    needs_second: list[dict] = []
    for bid, ev in all_first_reviews.items():
        # Skip if current reviewer already reviewed this
        if bid in reviewer_keys:
            continue
        # Skip if not in our split
        matching = [
            e for e in split_entries
            if build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"]) == bid
        ]
        if not matching:
            continue
        label = ev.get("label", "")
        confidence = ev.get("confidence", "")
        if label == "JOIN" and confidence in ("medium", "low"):
            needs_second.append(matching[0])

    # Deduplicate
    seen: set[str] = set()
    unique_needs_second: list[dict] = []
    for e in needs_second:
        key = build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"])
        if key not in seen:
            seen.add(key)
            unique_needs_second.append(e)

    # Calculate coverage target: at least 20% of all entries (approx)
    total_entries = len(split_entries)
    target = max(len(unique_needs_second), int(total_entries * 0.2) + 1)
    target = min(target, total_entries)  # cap at total entries available

    # Start with prioritized entries
    selected_keys: set[str] = set()
    queue: list[dict] = []

    for e in unique_needs_second:
        key = build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"])
        if key not in selected_keys:
            selected_keys.add(key)
            queue.append(e)

    # Fill remaining with random entries (deterministic)
    remaining_candidates = [
        e for e in split_entries
        if build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"]) not in selected_keys
        and build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"]) not in reviewer_keys
    ]
    rng.shuffle(remaining_candidates)

    for e in remaining_candidates:
        if len(queue) >= target:
            break
        key = build_boundary_key(e["sourceId"], e["leftCueId"], e["rightCueId"])
        if key not in selected_keys:
            selected_keys.add(key)
            queue.append(e)

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
