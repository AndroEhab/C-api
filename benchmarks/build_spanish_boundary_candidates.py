"""Deterministic candidate boundary extraction for Spanish subtitle benchmark.

Usage:
    python -m benchmarks.build_spanish_boundary_candidates \\
        --input benchmark-source --output benchmarks/spanish_boundary_candidates.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is on sys.path for app imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.language_profile import (
    visible_text,
    ends_strong_sentence,
    text_contains_question,
)
from app.subtitles import (
    SubtitleSegment,
    parse_srt_file,
    _detect_cue_speakers,
    _normalise_speaker,
)


# ── constants ──────────────────────────────────────────────────────────────

_ELLIPSIS_RE = re.compile(r"\.\.\.|…")
_DIALOGUE_DASH_RE = re.compile(r"^\s*(?:--?|[–—])\s+")
_SHORT_RESPONSES = frozenset({
    "sí", "no", "claro", "vale", "bueno", "ok", "okay",
    "sip", "nop", "ya", "dale", "listo", "hecho",
})
_CAPTION_KEYWORDS = frozenset({
    "risas", "música", "aplausos", "gritos", "llora", "suspiros",
    "teléfono", "timbre", "disparo", "explosión", "ruido",
    "sonido", "canción", "voz", "narrador", "subtítulos",
})
_CONJUNCTIONS = frozenset({
    "y", "e", "ni", "que", "pero", "mas", "sino", "aunque",
    "porque", "pues", "como", "cuando", "mientras", "si",
    "ya que", "debido a", "así que", "de modo que", "de manera que",
    "o", "u", "sea", "es decir", "o sea",
})
_SUBORDINATE_MARKERS = frozenset({
    "que", "cuando", "como", "donde", "mientras", "aunque",
    "porque", "si", "para que", "a fin de que", "antes de que",
    "después de que", "apenas", "tan pronto como", "en cuanto",
    "hasta que", "siempre que", "con tal de que", "a menos que",
    "sin que", "cuyo", "quien", "el cual", "la cual",
})

# ── helpers ────────────────────────────────────────────────────────────────


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _gap_ms(left: SubtitleSegment, right: SubtitleSegment) -> int:
    """Return the gap between left end and right start in ms."""
    gap = right.start_ms - left.end_ms
    return max(0, gap)


def _normalised_text(text: str) -> str:
    """Return a normalised version of visible text."""
    vt = visible_text(text)
    vt = re.sub(r"\s+", " ", vt)
    return vt.strip()


def _punctuation_metadata(text: str) -> dict[str, Any]:
    vt = visible_text(text)
    return {
        "endsWithPeriod": vt.rstrip().endswith(".") if vt else False,
        "endsWithQuestion": text_contains_question(text),
        "endsWithExclamation": bool(re.search(r"[!¡]$", vt.rstrip())),
        "endsWithEllipsis": bool(_ELLIPSIS_RE.search(vt.rstrip())),
        "startsWithQuestion": bool(re.match(r"^\s*¿", vt)),
        "startsWithExclamation": bool(re.match(r"^\s*¡", vt)),
        "startsWithDash": bool(_DIALOGUE_DASH_RE.match(text)),
    }


def _speaker_has_dash(speaker_info: dict) -> bool:
    """Return True if the cue has a dash speaker marker."""
    return "-" in speaker_info.get("speaker_markers", ())


def _speaker_explicit(speaker_info: dict) -> str | None:
    """Return the explicit speaker name or None."""
    return speaker_info.get("speaker")


def _candidate_category(
    left: SubtitleSegment,
    right: SubtitleSegment,
    left_text: str,
    right_text: str,
    left_norm: str,
    right_norm: str,
    gap: int,
    left_speaker_info: dict,
    right_speaker_info: dict,
    left_punct: dict,
    right_punct: dict,
) -> str:
    """Classify a boundary into its primary category string.

    Returns one category label; boundaries frequently satisfy multiple
    categories – the first applicable rule is returned.
    """
    left_vt = _normalised_text(left_text)
    right_vt = _normalised_text(right_text)

    left_explicit = _speaker_explicit(left_speaker_info)
    right_explicit = _speaker_explicit(right_speaker_info)
    left_dash = _speaker_has_dash(left_speaker_info)
    right_dash = _speaker_has_dash(right_speaker_info)

    # 1. Explicit speaker changes
    if left_explicit and right_explicit and _normalise_speaker(left_explicit) != _normalise_speaker(right_explicit):
        return "explicit_speaker_change"

    # 2. Sound / caption descriptions
    if any(kw in right_vt.lower() for kw in _CAPTION_KEYWORDS):
        return "caption_or_sound"

    # 3. Short responses
    first_word_right = right_vt.split()[0].lower().strip("¿¡.!?…,;:") if right_vt.split() else ""
    if first_word_right in _SHORT_RESPONSES:
        return "short_response"

    # 4. Inverted Spanish punctuation
    if right_punct.get("startsWithQuestion", False) or right_punct.get("startsWithExclamation", False):
        return "inverted_punctuation"

    # 5. Dialogue dashes (unknown-speaker turn)
    if right_dash:
        return "dialogue_dash"

    # 6. Unknown-speaker turns (dialogue dash without explicit speaker)
    if right_dash and not right_explicit:
        return "unknown_speaker_turn"

    # 7. Ellipsis / interrupted speech
    if _ELLIPSIS_RE.search(left_vt.rstrip()) or _ELLIPSIS_RE.search(right_vt):
        return "ellipsis_or_interrupted"

    # 8. Incorrect terminal period (sentence continues despite period)
    if left_punct.get("endsWithPeriod", False) and not ends_strong_sentence(left_vt) and right_vt[0:1].islower() if right_vt else False:
        return "misleading_period"

    # 9. Continuations without punctuation
    if left_vt and right_vt and not left_punct.get("endsWithPeriod", False) and not left_punct.get("endsWithQuestion", False) and not left_punct.get("endsWithExclamation", False) and not left_punct.get("endsWithEllipsis", False):
        if left_vt[-1].islower() or left_vt[-1].isalpha():
            return "continuation_without_punctuation"

    # 10. Independent sentences lacking punctuation
    if right_vt and not right_punct.get("endsWithPeriod", False) and not right_punct.get("endsWithQuestion", False) and not right_punct.get("endsWithExclamation", False):
        if right_vt[0:1].isupper() if right_vt else False:
            return "independent_without_punctuation"

    # 11. Subordinate clause
    first_word = right_vt.split()[0].lower() if right_vt.split() else ""
    if first_word in _SUBORDINATE_MARKERS:
        return "subordinate_clause"

    # 12. Coordinated clause
    if first_word in _CONJUNCTIONS and first_word not in _SUBORDINATE_MARKERS:
        return "coordinated_clause"

    # 13. Multiline cues
    if "\n" in left.text or "\n" in right.text:
        return "multiline_cue"

    # 14. Timing-based
    if 500 <= gap <= 1500:
        return "medium_gap"
    if 1500 < gap <= 3000:
        return "long_gap"

    # 15. Sentence continuation vs break — basic heuristic
    if right_vt and right_vt[0:1].isupper():
        if ends_strong_sentence(left_vt):
            return "clear_independent_sentence"
        else:
            return "clear_sentence_continuation"
    elif left_vt and ends_strong_sentence(left_vt) and right_vt and right_vt[0:1].islower():
        return "clear_sentence_continuation"

    # Default fallback
    return "clear_sentence_continuation"


def _extract_boundaries(
    segments: Sequence[SubtitleSegment],
    source_id: str,
    source_checksum: str,
) -> list[dict[str, Any]]:
    """Extract every adjacent cue boundary from parsed segments."""
    boundaries: list[dict[str, Any]] = []

    for i in range(len(segments) - 1):
        left = segments[i]
        right = segments[i + 1]
        gap = _gap_ms(left, right)

        # Context cues
        prev_context = [segments[j].text for j in range(max(0, i - 3), i)]
        next_context = [segments[j].text for j in range(i + 2, min(len(segments), i + 5))]

        left_speaker = _detect_cue_speakers(left.text.split("\n"), _normalised_text(left.text))
        right_speaker = _detect_cue_speakers(right.text.split("\n"), _normalised_text(right.text))
        left_punct = _punctuation_metadata(left.text)
        right_punct = _punctuation_metadata(right.text)
        left_norm = _normalised_text(left.text)
        right_norm = _normalised_text(right.text)

        category = _candidate_category(
            left, right,
            left.text, right.text,
            left_norm, right_norm,
            gap,
            left_speaker, right_speaker,
            left_punct, right_punct,
        )

        boundaries.append({
            "sourceId": source_id,
            "sourceChecksum": source_checksum,
            "leftCueId": left.segment_id,
            "rightCueId": right.segment_id,
            "left": left.text,
            "right": right.text,
            "leftNormalized": left_norm,
            "rightNormalized": right_norm,
            "leftStartMs": left.start_ms,
            "leftEndMs": left.end_ms,
            "rightStartMs": right.start_ms,
            "rightEndMs": right.end_ms,
            "gapMs": gap,
            "previousContext": prev_context,
            "nextContext": next_context,
            "originalLineStructure": {
                "leftLines": left.text.split("\n"),
                "rightLines": right.text.split("\n"),
            },
            "speakerMarkers": {
                "left": left_speaker,
                "right": right_speaker,
            },
            "punctuationMetadata": {
                "left": left_punct,
                "right": right_punct,
            },
            "category": category,
        })

    return boundaries


def _validate_extraction(
    boundaries: list[dict[str, Any]],
    segments: list[SubtitleSegment],
    path: Path,
) -> None:
    """Validate every extracted boundary exactly matches the source."""
    by_id = {s.segment_id: s for s in segments}

    for b in boundaries:
        left = by_id.get(b["leftCueId"])
        right = by_id.get(b["rightCueId"])
        assert left is not None, f"leftCueId {b['leftCueId']} not found in {path}"
        assert right is not None, f"rightCueId {b['rightCueId']} not found in {path}"
        assert left.text == b["left"], (
            f"left text mismatch for {b['leftCueId']} in {path}: "
            f"{left.text!r} != {b['left']!r}"
        )
        assert right.text == b["right"], (
            f"right text mismatch for {b['rightCueId']} in {path}: "
            f"{right.text!r} != {b['right']!r}"
        )
        assert left.start_ms == b["leftStartMs"], (
            f"leftStartMs mismatch for {b['leftCueId']}: "
            f"{left.start_ms} != {b['leftStartMs']}"
        )
        assert left.end_ms == b["leftEndMs"], (
            f"leftEndMs mismatch for {b['leftCueId']}: "
            f"{left.end_ms} != {b['leftEndMs']}"
        )
        assert right.start_ms == b["rightStartMs"]
        assert right.end_ms == b["rightEndMs"]

        # Verify adjacency
        idx_left = segments.index(left)
        idx_right = segments.index(right)
        assert idx_right == idx_left + 1, (
            f"Boundary {b['leftCueId']}-{b['rightCueId']} not adjacent "
            f"(positions {idx_left}, {idx_right})"
        )


def _load_sources(input_path: Path) -> list[tuple[str, list[SubtitleSegment], str]]:
    """Load SRT files and return (source_id, segments, checksum) tuples."""
    sources: list[tuple[str, list[SubtitleSegment], str]] = []

    paths = sorted(input_path.rglob("*.srt")) if input_path.is_dir() else [input_path]

    for p in paths:
        csum = _checksum(p)
        segments = parse_srt_file(p)
        segments = [s for s in segments if s.text.strip()]  # skip empty cues
        if len(segments) < 2:
            print(f"  skipping {p.name}: only {len(segments)} non-empty cues")
            continue
        sources.append((p.stem, segments, csum))
        print(f"  {p.name}: {len(segments)} cues, checksum={csum}")

    return sources


def _stratified_sample(
    all_boundaries: list[dict[str, Any]],
    target: int = 250,
) -> list[dict[str, Any]]:
    """Deterministic stratified sampling across categories and sources."""
    import random

    rng = random.Random(42)  # deterministic seed

    # Group by category
    by_category: dict[str, list[dict[str, Any]]] = {}
    for b in all_boundaries:
        cat = b["category"]
        by_category.setdefault(cat, []).append(b)

    _ = len(all_boundaries)  # total_candidates tracked via report

    # Distribute slots proportionally across categories, min 2 per category
    cat_counts: dict[str, int] = {}
    categories_sorted = sorted(by_category.keys())
    remaining = target

    # Give every category at least 2
    for cat in categories_sorted:
        count = min(2, len(by_category[cat]))
        cat_counts[cat] = count
        remaining -= count

    # Distribute remaining proportionally
    if remaining > 0 and categories_sorted:
        # Weight by sqrt of category size to avoid drowning small cats
        total_weight = sum(
            max(1, len(by_category[c]) ** 0.5) for c in categories_sorted
        )
        for cat in categories_sorted:
            if remaining <= 0:
                break
            weight = max(1, len(by_category[cat]) ** 0.5)
            extra = max(0, min(
                remaining,
                int(remaining * weight / total_weight),
                len(by_category[cat]) - cat_counts[cat],
            ))
            cat_counts[cat] += extra
            remaining -= extra

        # Distribute any leftovers
        for cat in categories_sorted:
            if remaining <= 0:
                break
            available = len(by_category[cat]) - cat_counts[cat]
            if available > 0:
                give = min(remaining, available)
                cat_counts[cat] += give
                remaining -= give

    # Sample from each category
    sampled: list[dict[str, Any]] = []
    for cat in categories_sorted:
        pool = list(by_category[cat])
        count = cat_counts.get(cat, 0)
        # Sort for determinism, then shuffle with seed
        pool.sort(key=lambda x: (x["sourceId"], x["leftCueId"]))
        rng.shuffle(pool)
        selected = pool[:count]
        sampled.extend(selected)

    # Sort by source and cue order
    sampled.sort(key=lambda x: (x["sourceId"], int(x["leftCueId"]) if x["leftCueId"].isdigit() else x["leftCueId"]))
    return sampled


def _category_and_timing_report(boundaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate category and timing band counts for the report."""
    from collections import Counter

    category_counts: Counter = Counter()
    timing_bands: Counter = Counter()

    for b in boundaries:
        category_counts[b["category"]] += 1
        gap = b["gapMs"]
        if gap <= 100:
            timing_bands["0-100ms"] += 1
        elif gap <= 300:
            timing_bands["101-300ms"] += 1
        elif gap <= 500:
            timing_bands["301-500ms"] += 1
        elif gap <= 1500:
            timing_bands["501-1500ms"] += 1
        elif gap <= 3000:
            timing_bands["1501-3000ms"] += 1
        else:
            timing_bands["3000+ms"] += 1

    return {
        "totalBoundaries": len(boundaries),
        "categoryCounts": dict(category_counts.most_common()),
        "timingBands": dict(timing_bands.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Spanish subtitle boundary candidates for benchmark"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="SRT file or directory of SRT files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=250,
        help="Target number of stratified candidates (default: 250)",
    )
    args = parser.parse_args()

    print(f"Loading sources from {args.input}")
    sources = _load_sources(args.input)

    if not sources:
        print("ERROR: No usable source files found.")
        sys.exit(1)

    all_boundaries: list[dict[str, Any]] = []
    for source_id, segments, csum in sources:
        print(f"  extracting boundaries from {source_id}...")
        boundaries = _extract_boundaries(segments, source_id, csum)
        print(f"    {len(boundaries)} boundaries")
        _validate_extraction(boundaries, segments, args.input / f"{source_id}.srt" if args.input.is_dir() else args.input)
        all_boundaries.extend(boundaries)

    print(f"\nTotal raw boundaries: {len(all_boundaries)}")

    sampled = _stratified_sample(all_boundaries, target=args.target)
    print(f"Sampled candidates: {len(sampled)}")

    report = _category_and_timing_report(sampled)
    print("\nCategory distribution:")
    for cat, count in report["categoryCounts"].items():
        print(f"  {cat}: {count}")
    print("\nTiming bands:")
    for band, count in report["timingBands"].items():
        print(f"  {band}: {count}")

    # Prepare output with review fields
    output_entries = []
    for b in sampled:
        entry = {
            "sourceId": b["sourceId"],
            "sourceChecksum": b["sourceChecksum"],
            "leftCueId": b["leftCueId"],
            "rightCueId": b["rightCueId"],
            "left": b["left"],
            "right": b["right"],
            "leftNormalized": b["leftNormalized"],
            "rightNormalized": b["rightNormalized"],
            "leftStartMs": b["leftStartMs"],
            "leftEndMs": b["leftEndMs"],
            "rightStartMs": b["rightStartMs"],
            "rightEndMs": b["rightEndMs"],
            "gapMs": b["gapMs"],
            "previousContext": b["previousContext"],
            "nextContext": b["nextContext"],
            "originalLineStructure": b["originalLineStructure"],
            "speakerMarkers": b["speakerMarkers"],
            "punctuationMetadata": b["punctuationMetadata"],
            "category": b["category"],
            "draftLabel": None,
            "goldLabel": None,
            "reviewReason": "",
            "reviewStatus": "unreviewed",
        }
        output_entries.append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_entries, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(output_entries)} candidates to {args.output}")

    # Write CSV review sheet
    csv_path = args.output.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sourceId", "leftCueId", "rightCueId",
            "left", "right",
            "gapMs", "category",
            "draftLabel", "goldLabel", "reviewReason", "reviewStatus",
        ])
        for e in output_entries:
            writer.writerow([
                e["sourceId"],
                e["leftCueId"],
                e["rightCueId"],
                e["left"],
                e["right"],
                e["gapMs"],
                e["category"],
                e["draftLabel"] if e["draftLabel"] else "",
                e["goldLabel"] if e["goldLabel"] else "",
                e["reviewReason"],
                e["reviewStatus"],
            ])
    print(f"Saved CSV review to {csv_path}")

    # Print summary
    unreviewed = sum(1 for e in output_entries if e["reviewStatus"] == "unreviewed")
    print(f"Total candidates: {len(output_entries)}")
    print(f"Unreviewed: {unreviewed}")
    print(f"Sources: {len(sources)}")
    for source_id, segments, csum in sources:
        print(f"  {source_id}: {len([b for b in output_entries if b['sourceId'] == source_id])} candidates")


if __name__ == "__main__":
    main()
