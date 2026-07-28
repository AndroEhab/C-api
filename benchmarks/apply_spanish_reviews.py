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
from datetime import datetime, timezone
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
from build_spanish_boundary_candidates import (  # type: ignore[import-not-found]
    _compute_maturity_level,
    _is_operationally_ready,
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


def derive_review_state(
    events: list[dict],
) -> dict:
    """Derive final review state from a boundary's events.

    Returns dict with keys:
        goldLabel, labelConfidence, reviewerCount, needsSecondReview,
        reviewReason, reviewStatus, labelOrigin
    """
    # Default unreviewed state
    state: dict = {
        "goldLabel": None,
        "labelConfidence": None,
        "reviewerCount": 0,
        "needsSecondReview": False,
        "reviewReason": "",
        "reviewStatus": "unreviewed",
        "labelOrigin": None,
    }

    if not events:
        return state

    # Separate regular reviews from adjudications
    reviews = [e for e in events if e.get("reviewRound", 0) >= 1 and e.get("label")]
    adjudications = [
        e for e in events
        if e.get("reviewRound", 0) == 0  # adjudication marker
        and e.get("label")
    ]

    # If there's an adjudication event, use it
    if adjudications:
        adj = adjudications[-1]  # last adjudication wins
        confidence = adj.get("confidence", "medium")
        label = adj.get("label", "AMBIGUOUS")
        # Low-confidence final decisions become AMBIGUOUS
        if confidence == "low" and label != "AMBIGUOUS":
            label = "AMBIGUOUS"
            confidence = "low"

        state.update({
            "goldLabel": label,
            "labelConfidence": confidence,
            "reviewerCount": len(reviews) + len(adjudications),
            "needsSecondReview": False,
            "reviewReason": adj.get("reason", ""),
            "reviewStatus": "adjudicated",
            "labelOrigin": "human",
        })
        return state

    # Count unique reviewers
    reviewer_ids: list[str] = []
    seen_reviewers: set[str] = set()
    for e in reviews:
        rid = e.get("reviewerId", "")
        if rid and rid not in seen_reviewers:
            seen_reviewers.add(rid)
            reviewer_ids.append(rid)

    reviewer_count = len(seen_reviewers)

    if reviewer_count == 0:
        return state

    # Get the latest review per unique reviewer
    latest_per_reviewer: dict[str, dict] = {}
    for e in reviews:
        rid = e.get("reviewerId", "")
        if rid:
            latest_per_reviewer[rid] = e

    review_list = list(latest_per_reviewer.values())

    if reviewer_count == 1:
        r = review_list[0]
        label = r.get("label")
        confidence = r.get("confidence", "medium")
        reason = r.get("reason", "")

        # Low confidence -> AMBIGUOUS
        if confidence == "low" and label != "AMBIGUOUS":
            label = "AMBIGUOUS"
            confidence = "low"

        needs_second = (
            label == "JOIN" and confidence in ("medium", "low")
        )

        state.update({
            "goldLabel": label,
            "labelConfidence": confidence,
            "reviewerCount": reviewer_count,
            "needsSecondReview": needs_second,
            "reviewReason": reason,
            "reviewStatus": "reviewed",
            "labelOrigin": "human",
        })
        return state

    # Two or more reviews - check agreement
    labels = set(r.get("label") for r in review_list)

    if len(labels) == 1:
        # Agreeing reviews
        r = review_list[0]
        label = r.get("label")
        confidence = r.get("confidence", "medium")
        reason = r.get("reason", "")

        if confidence == "low" and label != "AMBIGUOUS":
            label = "AMBIGUOUS"
            confidence = "low"

        state.update({
            "goldLabel": label,
            "labelConfidence": confidence,
            "reviewerCount": reviewer_count,
            "needsSecondReview": False,
            "reviewReason": f"Agreed: {reason}" if reason else "Agreed",
            "reviewStatus": "reviewed",
            "labelOrigin": "human",
        })
        return state

    # Disagreement
    reason = "; ".join(
        f"R{e.get('reviewerId', '?')}: {e.get('label', '?')} ({e.get('reason', '')})"
        for e in review_list
    )
    state.update({
        "goldLabel": None,
        "labelConfidence": None,
        "reviewerCount": reviewer_count,
        "needsSecondReview": True,
        "reviewReason": f"Disagreement: {reason}",
        "reviewStatus": "needs_adjudication",
        "labelOrigin": "human",
    })
    return state


def build_boundary_key(
    source_id: str, left_cue_id: str, right_cue_id: str,
) -> str:
    return f"{source_id}:{left_cue_id}:{right_cue_id}"


def apply_reviews() -> None:
    """Read everything, derive state, write updated fixture and report."""
    # Load fixture
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        entries: list[dict] = json.load(f)

    # Load ledger
    events = load_ledger(LEDGER_PATH)
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

    # Write updated fixture
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"Updated {len(entries)} entries in {FIXTURE_PATH.name}")

    # Regenerate report
    _regenerate_report(entries)
    print(f"Regenerated {REPORT_PATH.name}")


def _regenerate_report(entries: list[dict]) -> None:
    """Regenerate dataset report to reflect updated review state."""
    from collections import Counter

    # Load manifest and reference
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data: dict = json.load(f)
    manifest = manifest_data.get("sources", [])

    with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
        reference: list[dict] = json.load(f)

    manifest_by_id: dict = {m["sourceId"]: m for m in manifest}
    source_counts: Counter = Counter()
    for e in entries:
        source_counts[e["sourceId"]] += 1

    # Per region/variant
    region_counts: Counter = Counter()
    for e in entries:
        src = e["sourceId"]
        var = manifest_by_id.get(src, {}).get("spanishVariant", "unknown")
        region_counts[var] += 1

    # Per content type
    type_counts: Counter = Counter()
    for e in entries:
        src = e["sourceId"]
        ctype = manifest_by_id.get(src, {}).get("contentType", "other")
        type_counts[ctype] += 1

    # Per source quality tier
    tier_counts: Counter = Counter()
    for e in entries:
        tier_counts[e.get("sourceQualityTier", "unknown")] += 1

    # Per content structure
    structure_counts: Counter = Counter()
    for e in entries:
        structure_counts[e.get("contentStructure", "unknown")] += 1

    # Per original spoken language
    lang_counts: Counter = Counter()
    for e in entries:
        lang_counts[e.get("originalSpokenLanguage", "unknown")] += 1

    # Sampling tag counts
    tag_counts: Counter = Counter()
    for e in entries:
        for tag in e.get("samplingTags", []):
            tag_counts[tag] += 1

    # Timing band counts
    timing_counts: Counter = Counter()
    for e in entries:
        timing_counts[e.get("timingBand", "unknown")] += 1

    # Split counts
    dev_count = sum(1 for e in entries if e.get("split") == "dev")
    test_count = sum(1 for e in entries if e.get("split") == "test")

    # Dev/test tag coverage
    dev_tags: Counter = Counter()
    test_tags: Counter = Counter()
    for e in entries:
        split = e.get("split", "")
        for tag in e.get("samplingTags", []):
            if split == "dev":
                dev_tags[tag] += 1
            elif split == "test":
                test_tags[tag] += 1

    # Multiline count
    multiline_left = sum(
        1 for e in entries if "multiline_cue_left" in e.get("structureTags", [])
    )
    multiline_right = sum(
        1 for e in entries if "multiline_cue_right" in e.get("structureTags", [])
    )
    multiline = sum(
        1 for e in entries
        if "multiline_cue_left" in e.get("structureTags", [])
        or "multiline_cue_right" in e.get("structureTags", [])
    )

    # Multiple speaker count
    multi_speaker = sum(
        1 for e in entries
        if e.get("speakerMarkers", {})
        .get("left", {})
        .get("contains_multiple_speakers", False)
        or e.get("speakerMarkers", {})
        .get("right", {})
        .get("contains_multiple_speakers", False)
    )

    # Chain count
    chain_ids = set()
    for e in entries:
        cid = e.get("chainId")
        if cid:
            chain_ids.add(cid)

    unreviewed = sum(1 for e in entries if e.get("reviewStatus") == "unreviewed")
    human_labeled = sum(1 for e in entries if e.get("labelOrigin") == "human")

    # Review progress
    reviewed = sum(
        1 for e in entries if e.get("reviewStatus") in ("reviewed", "adjudicated")
    )
    needs_second = sum(1 for e in entries if e.get("needsSecondReview"))
    second_done = sum(1 for e in entries if e.get("reviewerCount", 0) >= 2)
    ambiguous = sum(1 for e in entries if e.get("goldLabel") == "AMBIGUOUS")

    # Maturity and readiness
    maturity = _compute_maturity_level(entries)
    op_ready = _is_operationally_ready(entries, manifest, reference)

    # Source provenance
    source_provenance = []
    for m in manifest:
        sid = m["sourceId"]
        source_provenance.append({
            "sourceId": sid,
            "contentType": m.get("contentType", "other"),
            "contentStructure": m.get("contentStructure", "unknown"),
            "originalSpokenLanguage": m.get("originalSpokenLanguage", "unknown"),
            "spanishVariant": m.get("spanishVariant", "unknown"),
            "sourceQualityTier": m.get("sourceQualityTier", "unknown"),
            "translationType": m.get("translationType", "unknown"),
            "candidatesInBenchmark": source_counts.get(sid, 0),
        })

    report: dict = {
        "datasetReport": {
            "reportVersion": "2.0",
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        },
        "sourceProvenance": source_provenance,
        "overview": {
            "totalCandidates": len(entries),
            "totalSources": len(source_counts),
            "devCount": dev_count,
            "testCount": test_count,
            "devRatio": round(dev_count / len(entries), 3) if entries else 0,
            "unreviewed": unreviewed,
            "humanLabeledCount": human_labeled,
            "multilineCues": multiline,
            "multilineLeft": multiline_left,
            "multilineRight": multiline_right,
            "multipleSpeakerCues": multi_speaker,
            "chainCount": len(chain_ids),
        },
        "reviewProgress": {
            "unreviewed": unreviewed,
            "reviewed": reviewed,
            "needsSecondReview": needs_second,
            "secondReviewCompleted": second_done,
            "ambiguousExcluded": ambiguous,
        },
        "metricStrata": {
            "candidatesPerSource": dict(source_counts.most_common()),
            "candidatesPerRegion": dict(region_counts.most_common()),
            "candidatesPerContentType": dict(type_counts.most_common()),
            "candidatesPerQualityTier": dict(tier_counts.most_common()),
            "candidatesPerContentStructure": dict(structure_counts.most_common()),
            "candidatesPerOriginalLanguage": dict(lang_counts.most_common()),
        },
        "samplingTagCounts": dict(tag_counts.most_common()),
        "timingBandCounts": dict(timing_counts.most_common()),
        "devTestCoverage": {
            "devCount": dev_count,
            "testCount": test_count,
            "devTagCoverage": dict(dev_tags.most_common()),
            "testTagCoverage": dict(test_tags.most_common()),
        },
        "benchmarkOperationallyReady": op_ready,
        "nativeCoverageTargetMet": False,
        "knownLimitations": [
            "No dialogue-heavy source originally spoken in Spanish from Spain exists.",
            "No dialogue-heavy source originally spoken in Spanish from Latin America exists.",
            "The only originally-Spanish source (ted_tales_es) is a TED monologue, not dialogue-heavy.",
            "The only dialogue-heavy source (the_goat_life_es) is translated from Malayalam; provenance is community translation, not professionally verified.",
        ],
        "maturity": maturity,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    apply_reviews()


if __name__ == "__main__":
    main()
