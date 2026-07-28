"""Deterministic candidate boundary extraction for Spanish subtitle benchmark.

Usage:
    python -m benchmarks.build_spanish_boundary_candidates \\
        --input benchmark-source \\
        --output benchmarks/spanish_boundary_candidates.json

Corrections applied over the original build_spanish_boundary_candidates.py:
  1. Source-provenance manifest instead of blind-loading all SRTs
  2. Uses SubtitleSegment.raw_text and .lines throughout
  3. Multi-label sampling dimensions (tags) instead of single priority category
  4. Classification defects fixed (unknown_speaker_turn reachable,
     misleading_period independent of terminal punct, multiline from lines,
     caption detection requires caption structure, short_response genuine)
  5. Signed timing gaps (no clamping overlaps to zero)
  6. Source-aware deterministic sampling with quotas
  7. Deterministic scene grouping and split assignment
  8. Portable validation support
  9. Rich review CSV
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.language_profile import (
    visible_text,
    ends_strong_sentence,
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
_SRT_PATH = re.compile(r"^\d+\s*$")

# Caption detection: requires caption-like structure
_CAPTION_STRUCTURE_RE = re.compile(
    r"^[\(\[【]\s*(?:risas|música|aplausos|gritos|llora|suspiros|"
    r"teléfono|timbre|disparo|explosión|ruido|sonido|canción|"
    r"voz|narrador|subtítulos|lágrimas|tose|susurra|gime|respira|"
    r"murmura|tartamudea|sollozando|tosiendo|bostezo|estornudo|"
    r"pita|pitido|alarma|sirena|trueno|lluvia|olas|viento|"
    r"música|aplausos|risas)\s*[\)\]】]?\s*$",
    re.IGNORECASE,
)
_CAPTION_MUSIC_RE = re.compile(r"[♫♪🎵🎶]+")
_CAPTION_BRACKET_RE = re.compile(
    r"^[\(\[【][^\)\]】]{1,60}[\)\]】]\s*$"
)

# Punctuation that can be misleading
_MISLEADING_PERIOD_RE = re.compile(r"""
    (?:
        [.][.][.]       # trailing ellipsis
        |
        \b(?:Dr|Dra|Sr|Sra|Sta|etc|ej|pág|Vol|vs|mons|ilmo|ud)\.  # abbreviations
        |
        (?:^|[.?!])\s+(?:y|e|ni|o|u|que|pero|mas|sino|aunque|porque|pues)  # conjunction after terminator
    )
""", re.IGNORECASE | re.VERBOSE)

# Genuinely short responses: must be short standalone utterances
_SHORT_RESPONSE_WORDS = frozenset({
    "sí", "no", "sip", "nop", "vale", "ok", "okay",
})
_SHORT_RESPONSE_PHRASES = frozenset({
    "sí", "no", "claro", "vale", "bueno", "ok", "okay",
    "sip", "nop", "ya", "dale", "listo", "hecho",
    "por supuesto", "desde luego", "está bien", "de acuerdo",
    "tal vez", "quizás", "quizá",
})

_TIMING_BAND_DEFS = [
    ("0-100ms", 0, 100),
    ("101-300ms", 101, 300),
    ("301-500ms", 301, 500),
    ("501-1500ms", 501, 1500),
    ("1501-3000ms", 1501, 3000),
    ("3000+ms", 3001, float("inf")),
]

SCENE_BREAK_GAP_MS = 5000  # gaps >= this start a new scene
MAX_SCENE_CUES = 200        # max cues per scene (forced break)


# ── helpers ────────────────────────────────────────────────────────────────


def _full_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _short_checksum(path: Path) -> str:
    return _full_checksum(path)[:16]


def _normalised_text(text: str) -> str:
    vt = visible_text(text)
    vt = re.sub(r"\s+", " ", vt)
    return vt.strip()


def _timing_band(gap_ms: int) -> str:
    for label, lo, hi in _TIMING_BAND_DEFS:
        if lo <= gap_ms <= hi:
            return label
    return "3000+ms"


def _speaker_has_dash(speaker_info: dict) -> bool:
    return "-" in speaker_info.get("speaker_markers", ())


def _speaker_explicit(speaker_info: dict) -> str | None:
    return speaker_info.get("speaker")


def _has_caption_structure(text: str) -> bool:
    """Check if text looks like a caption/sound descriptor, not normal dialogue."""
    vt = visible_text(text).strip()
    if not vt:
        return False
    # Music symbols are clear caption indicators
    if _CAPTION_MUSIC_RE.search(vt):
        return True
    # Bracket/parenthesis structure
    if _CAPTION_BRACKET_RE.match(text.strip()):
        return True
    # Specific caption patterns like (risas), [música]
    if _CAPTION_STRUCTURE_RE.match(text.strip()):
        return True
    return False


def _is_genuine_short_response(right_text: str, right_vt: str) -> bool:
    """Check if right cue is a genuinely short standalone response.

    Must be short (<=3 words) and not just a sentence starting with a word
    like 'no', 'bueno', 'claro' that happens to be in the short-response set.
    """
    words = right_vt.split()
    if len(words) == 0:
        return False
    if len(words) > 3:
        return False  # too long for a standalone response
    first = words[0].lower().strip("¿¡.!?…,;:\"'")
    if first not in _SHORT_RESPONSE_WORDS:
        return False
    # Must be the whole utterance or a genuine response fragment
    text_lower = right_vt.lower().strip("¿¡.!?…")
    if text_lower in _SHORT_RESPONSE_PHRASES:
        return True
    # Single-word responses
    if first in _SHORT_RESPONSE_WORDS and len(words) <= 2:
        return True
    return False


def _detect_misleading_period(left_text: str, left_vt: str, right_vt: str) -> bool:
    """Detect a period that looks terminal but isn't (abbrev, ellipsis, etc.)."""
    vt = left_vt.rstrip()
    if not vt.endswith("."):
        return False
    # Ellipsis-looking pattern (... at end)
    if _ELLIPSIS_RE.search(vt):
        return False  # this is an ellipsis, not misleading period
    # Check for abbreviation
    trailing = vt.split()[-1] if vt.split() else ""
    if re.match(r"^[A-Za-z]\.$", trailing) or trailing in (
        "Dr.", "Dra.", "Sr.", "Sra.", "Sta.",
        "etc.", "ej.", "pág.", "Vol.", "vs.",
    ):
        return True
    # Sentence continues (next starts lowercase or conjunction)
    if right_vt and right_vt[0:1].islower():
        return True
    first_word = right_vt.split()[0].lower() if right_vt.split() else ""
    if first_word in ("y", "e", "ni", "o", "u", "que", "pero", "mas", "porque"):
        return True
    return False


# ── multi-label tagging ────────────────────────────────────────────────────


def _compute_sampling_tags(
    left: SubtitleSegment,
    right: SubtitleSegment,
    left_text: str,
    right_text: str,
    left_norm: str,
    right_norm: str,
    gap: int,
    left_speaker_info: dict,
    right_speaker_info: dict,
    left_lines: tuple[str, ...],
    right_lines: tuple[str, ...],
) -> dict[str, Any]:
    """Compute multi-label sampling tags across independent dimensions."""
    left_vt = _normalised_text(left_text)
    right_vt = _normalised_text(right_text)

    left_explicit = _speaker_explicit(left_speaker_info)
    right_explicit = _speaker_explicit(right_speaker_info)
    right_dash = _speaker_has_dash(right_speaker_info)

    sampling_tags: list[str] = []
    structure_tags: list[str] = []
    punctuation_tags: list[str] = []
    linguistic_tags: list[str] = []

    # ── sampling tags (used for quota selection) ──

    # Speaker changes
    if left_explicit and right_explicit and _normalise_speaker(left_explicit) != _normalise_speaker(right_explicit):
        sampling_tags.append("explicit_speaker_change")

    # Dialogue dash (unknown-speaker turn) — must come BEFORE generic dash check
    if right_dash and not right_explicit:
        sampling_tags.append("unknown_speaker_turn")

    # Dialogue dash (any)
    if right_dash:
        sampling_tags.append("dialogue_dash")
        structure_tags.append("dialogue_dash")

    # Caption/sound descriptors (requires caption structure)
    if _has_caption_structure(right_text):
        sampling_tags.append("caption")

    # Genuine short responses
    if _is_genuine_short_response(right_text, right_vt):
        sampling_tags.append("short_response")

    # Inverted punctuation
    if re.match(r"^\s*¿", right_vt):
        sampling_tags.append("inverted_question")
        punctuation_tags.append("inverted_question")
    if re.match(r"^\s*¡", right_vt):
        sampling_tags.append("inverted_exclamation")
        punctuation_tags.append("inverted_exclamation")

    # Ellipsis / interruption
    if _ELLIPSIS_RE.search(left_vt.rstrip()) or _ELLIPSIS_RE.search(right_vt):
        sampling_tags.append("ellipsis_or_interruption")
        punctuation_tags.append("ellipsis_or_interruption")

    # Misleading period
    if _detect_misleading_period(left_text, left_vt, right_vt):
        sampling_tags.append("misleading_period")
        punctuation_tags.append("misleading_period")

    # Continuation signals
    if right_vt and right_vt[0:1].islower() if right_vt else False:
        sampling_tags.append("join_like")
        structure_tags.append("lowercase_start")

    # Independent utterances
    if right_vt and right_vt[0:1].isupper() and ends_strong_sentence(left_vt):
        sampling_tags.append("independent_utterance")
        structure_tags.append("uppercase_start")

    # Punctuation boundaries
    if left_vt.rstrip()[-1:] in (".", "?", "!", "…") if left_vt else False:
        punctuation_tags.append("terminal_punctuation_left")
    if right_vt.rstrip()[-1:] in (".", "?", "!", "…") if right_vt else False:
        punctuation_tags.append("terminal_punctuation_right")

    # Multiline detection from SubtitleSegment.lines
    if len(left_lines) > 1:
        structure_tags.append("multiline_cue_left")
    if len(right_lines) > 1:
        structure_tags.append("multiline_cue_right")

    # Linguistic: conjunctions
    if right_vt:
        first_word = right_vt.split()[0].lower().strip("¿¡.!?…")
        conj = {"y", "e", "ni", "que", "pero", "mas", "sino", "aunque",
                "porque", "pues", "como", "mientras", "si", "o", "u"}
        if first_word in conj:
            linguistic_tags.append("starts_with_conjunction")

    # Timing band
    timing_band = _timing_band(gap)

    return {
        "samplingTags": sorted(set(sampling_tags)),
        "structureTags": sorted(set(structure_tags)),
        "punctuationTags": sorted(set(punctuation_tags)),
        "linguisticTags": sorted(set(linguistic_tags)),
        "timingBand": timing_band,
    }


# ── chain detection ────────────────────────────────────────────────────────


def _detect_chains(
    boundaries: list[dict[str, Any]],
    source_id: str,
) -> list[dict[str, Any]]:
    """Detect 3+ cue sentence chains from consecutive boundaries.

    A chain is a sequence of boundaries where each adjacent pair looks like
    a continuation (lowercase start, conjunction, ellipsis, etc.).
    """
    chain_id_counter: int = 0
    chain_start: int | None = None

    for i, b in enumerate(boundaries):
        tags = b.get("samplingTags", [])
        is_cont = any(t in tags for t in ("join_like", "ellipsis_or_interruption"))

        if is_cont and chain_start is None:
            chain_start = i
        elif not is_cont and chain_start is not None:
            if i - chain_start >= 3:  # 3+ consecutive continuation boundaries
                chain_id_counter += 1
                for j in range(chain_start, i):
                    boundaries[j]["chainId"] = f"{source_id}-ch{chain_id_counter}"
            chain_start = None

    # Handle chain at end
    if chain_start is not None and len(boundaries) - chain_start >= 3:
        chain_id_counter += 1
        for j in range(chain_start, len(boundaries)):
            boundaries[j]["chainId"] = f"{source_id}-ch{chain_id_counter}"

    return boundaries


# ── scene detection ────────────────────────────────────────────────────────


def _detect_scenes(
    segments: Sequence[SubtitleSegment],
    source_id: str,
) -> list[dict[str, Any]]:
    """Partition source cues into deterministic scenes using timing gaps.

    A scene break is declared when:
    - gap between cues >= SCENE_BREAK_GAP_MS (5s), or
    - MAX_SCENE_CUES cues accumulated in current scene
    """
    scenes: list[dict[str, Any]] = []
    current_start = 0

    for i in range(1, len(segments)):
        gap = segments[i].start_ms - segments[i - 1].end_ms
        is_break = (
            gap >= SCENE_BREAK_GAP_MS
            or (i - current_start) >= MAX_SCENE_CUES
        )
        if is_break:
            scenes.append({
                "sceneId": f"{source_id}-s{len(scenes) + 1:03d}",
                "sourceId": source_id,
                "cueRangeStart": segments[current_start].segment_id,
                "cueRangeEnd": segments[i - 1].segment_id,
                "startIdx": current_start,
                "endIdx": i - 1,
                "cueCount": i - current_start,
            })
            current_start = i

    # Last scene
    if current_start < len(segments):
        scenes.append({
            "sceneId": f"{source_id}-s{len(scenes) + 1:03d}",
            "sourceId": source_id,
            "cueRangeStart": segments[current_start].segment_id,
            "cueRangeEnd": segments[-1].segment_id,
            "startIdx": current_start,
            "endIdx": len(segments) - 1,
            "cueCount": len(segments) - current_start,
        })

    return scenes


# ── split assignment ────────────────────────────────────────────────────────


def _assign_splits(
    scenes: list[dict[str, Any]],
    rng: Any,
    dev_ratio: float = 0.7,
) -> dict[str, str]:
    """Assign whole scenes to dev or test, targeting dev_ratio.

    Deterministic allocation: group scenes by source, sort by cue index,
    then round-robin assign scenes to dev/test maintaining the ratio.
    Every source contributes to both splits.
    """
    from collections import defaultdict
    by_source: dict[str, list[dict]] = defaultdict(list)
    for s in scenes:
        by_source[s["sourceId"]].append(s)

    scene_split: dict[str, str] = {}
    total_cues = sum(s["cueCount"] for s in scenes)
    target_dev_cues = int(total_cues * dev_ratio)
    target_test_cues = total_cues - target_dev_cues
    dev_cues = 0
    test_cues = 0

    # Process sources in deterministic order
    for source_id in sorted(by_source.keys()):
        source_scenes = by_source[source_id]
        # Sort by source cue index for determinism
        source_scenes.sort(key=lambda s: s["startIdx"])
        # For each source, alternate starting assignment to achieve balance
        assign_to_dev = rng.random() < dev_ratio

        for sc in source_scenes:
            # Don't overshoot target if other split still needs cues
            would_dev = dev_cues + sc["cueCount"]
            would_test = test_cues + sc["cueCount"]

            # Force to test if dev would exceed target by too much
            if (would_dev > target_dev_cues + 5
                    and test_cues < target_test_cues):
                assign_to_dev = False
            # Force to dev if test would exceed target by too much
            elif (would_test > target_test_cues + 5
                  and dev_cues < target_dev_cues):
                assign_to_dev = True

            if assign_to_dev:
                scene_split[sc["sceneId"]] = "dev"
                dev_cues += sc["cueCount"]
            else:
                scene_split[sc["sceneId"]] = "test"
                test_cues += sc["cueCount"]

            # Toggle for next scene from this source
            assign_to_dev = not assign_to_dev

    return scene_split


# ── boundary extraction ────────────────────────────────────────────────────


def _extract_boundaries(
    segments: Sequence[SubtitleSegment],
    source_id: str,
    source_checksum: str,
) -> list[dict[str, Any]]:
    """Extract every adjacent cue boundary from parsed segments.

    Uses SubtitleSegment.raw_text and .lines for canonical source data.
    """
    boundaries: list[dict[str, Any]] = []

    for i in range(len(segments) - 1):
        left = segments[i]
        right = segments[i + 1]

        # Signed gap (negative = overlap)
        gap = right.start_ms - left.end_ms
        overlap = max(0, left.end_ms - right.start_ms)

        # Context cues with full structure
        prev_ctx: list[dict] = []
        for j in range(max(0, i - 3), i):
            ctx = segments[j]
            prev_ctx.append({
                "cueId": ctx.segment_id,
                "rawText": ctx.raw_text or ctx.text,
                "lines": list(ctx.lines),
                "startMs": ctx.start_ms,
                "endMs": ctx.end_ms,
            })

        next_ctx: list[dict] = []
        for j in range(i + 2, min(len(segments), i + 5)):
            ctx = segments[j]
            next_ctx.append({
                "cueId": ctx.segment_id,
                "rawText": ctx.raw_text or ctx.text,
                "lines": list(ctx.lines),
                "startMs": ctx.start_ms,
                "endMs": ctx.end_ms,
            })

        # Speaker detection from original lines (not normalized_text.split)
        left_speaker = _detect_cue_speakers(list(left.lines), left.text)
        right_speaker = _detect_cue_speakers(list(right.lines), right.text)

        left_norm = _normalised_text(left.text)
        right_norm = _normalised_text(right.text)

        # Multi-label tags
        tags = _compute_sampling_tags(
            left, right,
            left.text, right.text,
            left_norm, right_norm,
            gap,
            left_speaker, right_speaker,
            left.lines, right.lines,
        )

        entry: dict[str, Any] = {
            "sourceId": source_id,
            "sourceChecksum": source_checksum,
            "leftCueId": left.segment_id,
            "rightCueId": right.segment_id,

            # Canonical source data (not reconstructed from normalized text)
            "leftRawText": left.raw_text,
            "rightRawText": right.raw_text,
            "leftLines": list(left.lines),
            "rightLines": list(right.lines),

            # Normalized forms for model input
            "leftNormalized": left_norm,
            "rightNormalized": right_norm,

            # Timing
            "leftStartMs": left.start_ms,
            "leftEndMs": left.end_ms,
            "rightStartMs": right.start_ms,
            "rightEndMs": right.end_ms,
            "gapMs": gap,
            "overlapMs": overlap,

            # Context
            "previousContext": prev_ctx,
            "nextContext": next_ctx,

            # Speaker metadata (from original lines)
            "speakerMarkers": {
                "left": left_speaker,
                "right": right_speaker,
            },

            # Multi-label tags (independent dimensions)
            **tags,

            # Review fields (all unreviewed, no model data)
            "goldLabel": None,
            "reviewReason": "",
            "reviewStatus": "unreviewed",

            # Chain and scene (filled later)
            "chainId": None,
            "sceneId": None,

            # Optional primary stratum (for deterministic quota)
            "primarySamplingStratum": None,
        }
        boundaries.append(entry)

    # Detect chains
    boundaries = _detect_chains(boundaries, source_id)

    return boundaries


# ── source manifest ────────────────────────────────────────────────────────


def build_source_manifest(input_path: Path) -> list[dict[str, Any]]:
    """Build source-provenance manifest from available SRT files."""
    manifest_path = Path(__file__).parent / "spanish_source_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest.get("sources", [])
    return []


# ── source loading ────────────────────────────────────────────────────────


def _load_sources(input_path: Path) -> list[tuple[str, list[SubtitleSegment], str]]:
    """Load SRT files and return (source_id, segments, checksum) tuples."""
    sources: list[tuple[str, list[SubtitleSegment], str]] = []
    manifest_sources = build_source_manifest(input_path)
    manifest_by_id: dict = {m["sourceId"]: m for m in manifest_sources}
    paths = sorted(input_path.rglob("*.srt")) if input_path.is_dir() else [input_path]

    for p in paths:
        csum = _full_checksum(p)
        short_csum = csum[:16]
        segments = parse_srt_file(p)
        segments = [s for s in segments if s.text.strip()]
        if len(segments) < 2:
            print(f"  skipping {p.name}: only {len(segments)} non-empty cues")
            continue

        src_id = p.stem

        # Verify against manifest
        if src_id in manifest_by_id:
            expected = manifest_by_id[src_id]["fullSha256"]
            if csum != expected:
                print(f"  WARNING: {p.name} checksum mismatch manifest!\n"
                      f"    expected: {expected}\n"
                      f"    actual:   {csum}")
            print(f"  {p.name}: {len(segments)} cues, "
                  f"variant={manifest_by_id[src_id].get('spanishVariant', '?')}, "
                  f"type={manifest_by_id[src_id].get('contentType', '?')}, "
                  f"lang={manifest_by_id[src_id].get('originalSpokenLanguage', '?')}")
        else:
            print(f"  {p.name}: {len(segments)} cues (no manifest entry)")

        sources.append((src_id, segments, short_csum))

    return sources


# ── source-aware sampling ──────────────────────────────────────────────────


def _sample_candidates(
    all_boundaries: list[dict[str, Any]],
    sources: list[tuple[str, list[SubtitleSegment], str]],
    manifest: list[dict[str, Any]],
    target: int = 250,
) -> list[dict[str, Any]]:
    """Deterministic source-aware sampling with minimum coverage quotas."""
    import random
    rng = random.Random(42)
    manifest_by_id = {m["sourceId"]: m for m in manifest}
    source_list = [s[0] for s in sources]
    from collections import defaultdict
    by_tag: dict[str, list[dict]] = defaultdict(list)
    by_source: dict[str, list[dict]] = defaultdict(list)
    by_variant: dict[str, list[dict]] = defaultdict(list)
    by_type: dict[str, list[dict]] = defaultdict(list)

    for b in all_boundaries:
        src = b["sourceId"]
        by_source[src].append(b)
        m = manifest_by_id.get(src, {})
        var = m.get("spanishVariant", "unknown")
        ctype = m.get("contentType", "other")
        by_variant[var].append(b)
        by_type[ctype].append(b)

        for tag in b.get("samplingTags", []):
            by_tag[tag].append(b)

    # Minimum quotas per tag (must have at least N)
    MIN_QUOTAS: dict[str, int] = {
        "dialogue_dash": 10,
        "unknown_speaker_turn": 5,
        "caption": 5,
        "short_response": 8,
        "inverted_question": 8,
        "inverted_exclamation": 5,
        "ellipsis_or_interruption": 5,
        "misleading_period": 5,
        "join_like": 15,
        "independent_utterance": 15,
        "explicit_speaker_change": 5,
    }

    # Timing band quotas
    TIMING_QUOTAS: dict[str, int] = {
        "501-1500ms": 20,
        "1501-3000ms": 10,
    }

    # Source and variant quotas
    SOURCE_MIN: int = 20  # min per source
    VARIANT_MIN: int = 15  # min per variant
    TYPE_MIN: int = 10  # min per content type

    selected: set[int] = set()  # indices into all_boundaries
    index_of: dict[str, int] = {}
    for i, b in enumerate(all_boundaries):
        key = (b["sourceId"], b["leftCueId"], b["rightCueId"])
        index_of[str(key)] = i

    def _selected_count_for_tag(tag: str) -> int:
        return sum(1 for i in selected if tag in all_boundaries[i].get("samplingTags", []))

    def _selected_for_timing(band: str) -> int:
        return sum(1 for i in selected if all_boundaries[i].get("timingBand") == band)

    def _selected_for_source(src: str) -> int:
        return sum(1 for i in selected if all_boundaries[i]["sourceId"] == src)

    def _selected_for_variant(var: str) -> int:
        return sum(1 for i in selected if manifest_by_id.get(all_boundaries[i]["sourceId"], {}).get("spanishVariant", "unknown") == var)

    def _selected_for_type(ctype: str) -> int:
        return sum(1 for i in selected if manifest_by_id.get(all_boundaries[i]["sourceId"], {}).get("contentType", "other") == ctype)

    def _select(b: dict) -> bool:
        """Attempt to select a boundary. Returns True if newly selected."""
        for i, cand in enumerate(all_boundaries):
            if (cand["sourceId"] == b["sourceId"]
                    and cand["leftCueId"] == b["leftCueId"]
                    and cand["rightCueId"] == b["rightCueId"]):
                if i not in selected:
                    selected.add(i)
                    return True
                return False
        return False

    # Phase 1: Fulfill tag minimums
    for tag, needed in sorted(MIN_QUOTAS.items()):
        pool = list(by_tag.get(tag, []))
        rng.shuffle(pool)
        for b in pool:
            if _selected_count_for_tag(tag) >= needed:
                break
            _select(b)

    # Phase 2: Timing band quotas
    for band, needed in sorted(TIMING_QUOTAS.items()):
        pool = [b for b in all_boundaries if b.get("timingBand") == band]
        rng.shuffle(pool)
        for b in pool:
            if _selected_for_timing(band) >= needed:
                break
            _select(b)

    # Phase 3: Source minimums
    for src in source_list:
        pool = list(by_source.get(src, []))
        rng.shuffle(pool)
        for b in pool:
            if _selected_for_source(src) >= SOURCE_MIN:
                break
            _select(b)

    # Phase 4: Variant minimums
    for var in set(m.get("spanishVariant", "unknown") for m in manifest):
        pool = list(by_variant.get(var, []))
        rng.shuffle(pool)
        for b in pool:
            if _selected_for_variant(var) >= VARIANT_MIN:
                break
            _select(b)

    # Phase 5: Content type minimums
    for ctype in set(m.get("contentType", "other") for m in manifest):
        pool = list(by_type.get(ctype, []))
        rng.shuffle(pool)
        for b in pool:
            if _selected_for_type(ctype) >= TYPE_MIN:
                break
            _select(b)

    # Phase 6: Fill to target with proportional source distribution
    remaining = target - len(selected)
    if remaining > 0:
        # Weight by how far each source is from its minimum
        source_remaining = {src: max(0, SOURCE_MIN - _selected_for_source(src))
                           for src in source_list}
        total_deficit = sum(source_remaining.values())
        if total_deficit == 0:
            # Distribute evenly
            per_source = max(1, remaining // len(source_list))
            for src in source_list:
                pool = list(by_source.get(src, []))
                rng.shuffle(pool)
                for b in pool:
                    if _selected_for_source(src) >= per_source or remaining <= 0:
                        break
                    if _select(b):
                        remaining -= 1
        else:
            # Fill deficits first
            for src in sorted(source_list, key=lambda s: -source_remaining[s]):
                pool = list(by_source.get(src, []))
                rng.shuffle(pool)
                for b in pool:
                    deficit = source_remaining[src]
                    if _selected_for_source(src) >= deficit or remaining <= 0:
                        break
                    if _select(b):
                        remaining -= 1

    # Phase 7: If still under, add more proportionally
    remaining = target - len(selected)
    if remaining > 0:
        for src in source_list:
            pool = list(by_source.get(src, []))
            rng.shuffle(pool)
            for b in pool:
                if remaining <= 0:
                    break
                if _select(b):
                    remaining -= 1

    # Build result in source order
    result = sorted(
        [all_boundaries[i] for i in sorted(selected)],
        key=lambda x: (x["sourceId"], int(x["leftCueId"]) if x["leftCueId"].lstrip("-").isdigit() else x["leftCueId"]),
    )

    # Assign scene IDs
    # Build segment list per source
    for src in source_list:
        segs = [s for sid, segs, _ in sources if sid == src for s in segs]
        if not segs:
            continue
        scenes = _detect_scenes(segs, src)
        # Build cue->scene mapping
        cue_to_scene: dict[str, str] = {}
        for sc in scenes:
            in_scene = False
            for s in segs:
                if s.segment_id == sc["cueRangeStart"]:
                    in_scene = True
                if in_scene:
                    cue_to_scene[s.segment_id] = sc["sceneId"]
                if s.segment_id == sc["cueRangeEnd"]:
                    in_scene = False

        for b in result:
            if b["sourceId"] == src:
                b["sceneId"] = cue_to_scene.get(b["leftCueId"], "")

    # Assign scene splits
    all_scene_entries: list[dict[str, Any]] = []
    seen_scene_ids: set[str] = set()
    for src in source_list:
        segs = [s for sid, segs, _ in sources if sid == src for s in segs]
        if segs:
            scenes = _detect_scenes(segs, src)
            for sc in scenes:
                if sc["sceneId"] not in seen_scene_ids:
                    seen_scene_ids.add(sc["sceneId"])
                    all_scene_entries.append(sc)

    scene_split_map = _assign_splits(all_scene_entries, rng)

    for b in result:
        b["split"] = scene_split_map.get(b.get("sceneId", ""), "dev")
    return result

# ── output generation ──────────────────────────────────────────────────────


def _generate_review_csv(
    entries: list[dict[str, Any]],
    csv_path: Path,
) -> None:
    """Generate a rich review CSV with all fields needed for labeling."""
    fieldnames = [
        "sourceId", "spanishVariant", "contentType",
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
        "goldLabel", "reviewReason", "reviewStatus",
    ]

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for e in entries:
            row = dict(e)

            # JSON-encode structured fields for CSV readability
            for field in ("previousContext", "nextContext",
                          "speakerMarkers", "samplingTags",
                          "structureTags", "punctuationTags",
                          "linguisticTags", "leftLines", "rightLines"):
                if field in row:
                    row[field] = json.dumps(row[field], ensure_ascii=False)

            # Null labels as empty strings
            if row.get("goldLabel") is None:
                row["goldLabel"] = ""
            if row.get("chainId") is None:
                row["chainId"] = ""

            writer.writerow(row)


def _generate_dataset_report(
    entries: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate comprehensive dataset report."""
    from collections import Counter, defaultdict
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
    multiline_left = sum(1 for e in entries if "multiline_cue_left" in e.get("structureTags", []))
    multiline_right = sum(1 for e in entries if "multiline_cue_right" in e.get("structureTags", []))
    multiline = sum(1 for e in entries if "multiline_cue_left" in e.get("structureTags", [])
                    or "multiline_cue_right" in e.get("structureTags", []))

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

    # Source provenance
    source_provenance = []
    for m in manifest:
        sid = m["sourceId"]
        source_provenance.append({
            "sourceId": sid,
            "contentType": m.get("contentType", "other"),
            "originalSpokenLanguage": m.get("originalSpokenLanguage", "unknown"),
            "spanishVariant": m.get("spanishVariant", "unknown"),
            "translationType": m.get("translationType", "unknown"),
            "candidatesInBenchmark": source_counts.get(sid, 0),
        })

    return {
        "datasetReport": {
            "reportVersion": "1.0",
            "created": "2026-07-28",
        },
        "sourceProvenance": source_provenance,
        "overview": {
            "totalCandidates": len(entries),
            "totalSources": len(source_counts),
            "devCount": dev_count,
            "testCount": test_count,
            "devRatio": round(dev_count / len(entries), 3) if entries else 0,
            "unreviewed": unreviewed,
            "multilineCues": multiline,
            "multilineLeft": multiline_left,
            "multilineRight": multiline_right,
            "multipleSpeakerCues": multi_speaker,
            "chainCount": len(chain_ids),
        },
        "candidatesPerSource": dict(source_counts.most_common()),
        "candidatesPerRegion": dict(region_counts.most_common()),
        "candidatesPerContentType": dict(type_counts.most_common()),
        "samplingTagCounts": dict(tag_counts.most_common()),
        "timingBandCounts": dict(timing_counts.most_common()),
        "devTestCoverage": {
            "devCount": dev_count,
            "testCount": test_count,
            "devTagCoverage": dict(dev_tags.most_common()),
            "testTagCoverage": dict(test_tags.most_common()),
        },
        "minimumRequirementsMet": False,
        "minimumRequirementsNotes": (
            "NO dialogue-heavy source originally spoken in Spanish from Spain exists. "
            "NO dialogue-heavy source originally spoken in Spanish from Latin America exists. "
            "The only originally-Spanish source (ted_tales_es) is a TED monologue. "
            "The only dialogue-heavy source (the_goat_life_es) is translated from Malayalam. "
            "Additional Spanish-original dialogue-heavy sources required before labeling."
        ),
    }


def _validate_extraction(
    boundaries: list[dict[str, Any]],
    segments: list[SubtitleSegment],
    path: Path,
) -> None:
    """Validate every extracted boundary exactly matches the source."""
    by_id: dict = {s.segment_id: s for s in segments}
    for b in boundaries:
        left = by_id.get(b["leftCueId"])
        right = by_id.get(b["rightCueId"])
        assert left is not None, f"leftCueId {b['leftCueId']} not found in {path}"
        assert right is not None, f"rightCueId {b['rightCueId']} not found in {path}"

        # Verify raw text
        assert left.raw_text == b["leftRawText"], (
            f"left raw_text mismatch for {b['leftCueId']} in {path}: "
            f"{left.raw_text!r} != {b['leftRawText']!r}"
        )
        assert right.raw_text == b["rightRawText"], (
            f"right raw_text mismatch for {b['rightCueId']} in {path}: "
            f"{right.raw_text!r} != {b['rightRawText']!r}"
        )

        # Verify lines
        assert list(left.lines) == b["leftLines"], (
            f"left lines mismatch for {b['leftCueId']} in {path}"
        )
        assert list(right.lines) == b["rightLines"], (
            f"right lines mismatch for {b['rightCueId']} in {path}"
        )

        # Verify timings
        assert left.start_ms == b["leftStartMs"]
        assert left.end_ms == b["leftEndMs"]
        assert right.start_ms == b["rightStartMs"]
        assert right.end_ms == b["rightEndMs"]

        # Verify adjacency
        idx_left = segments.index(left)
        idx_right = segments.index(right)
        assert idx_right == idx_left + 1, (
            f"Boundary {b['leftCueId']}-{b['rightCueId']} not adjacent "
            f"(positions {idx_left}, {idx_right})"
        )

        # Verify normalized text is a processed form - for multiline cues,
        # raw text must preserve newlines while normalized must be flat
        if "\n" in (b.get("leftRawText") or ""):
            assert "\n" not in (b.get("leftNormalized") or ""), (
                f"leftNormalized for {b['leftCueId']} contains newlines — "
                f"normalized text must be a processed single-line form"
            )


# ── reference manifest (for portable tests) ────────────────────────────────


def build_reference_manifest(
    entries: list[dict[str, Any]],
    sources: list[tuple[str, list[SubtitleSegment], str]],
) -> list[dict[str, Any]]:
    """Build a minimal reference manifest for portable test validation.

    Contains only the cue records needed to validate selected excerpts,
    without including copyrighted source content.
    """
    ref: list[dict[str, Any]] = []
    for sid, segs, csum in sources:
        by_id = {s.segment_id: s for s in segs}
        involved_ids: set[str] = set()
        for e in entries:
            if e["sourceId"] == sid:
                involved_ids.add(e["leftCueId"])
                involved_ids.add(e["rightCueId"])
                for ctx in e.get("previousContext", []):
                    involved_ids.add(ctx["cueId"])
                for ctx in e.get("nextContext", []):
                    involved_ids.add(ctx["cueId"])

        for seg in segs:
            if seg.segment_id in involved_ids:
                ref.append({
                    "sourceId": sid,
                    "sourceChecksum": csum,
                    "cueId": seg.segment_id,
                    "rawText": seg.raw_text,
                    "lines": list(seg.lines),
                    "text": seg.text,
                    "startMs": seg.start_ms,
                    "endMs": seg.end_ms,
                    "speaker": seg.speaker,
                    "speakerMarkers": list(seg.speaker_markers),
                    "containsMultipleSpeakers": seg.contains_multiple_speakers,
                })

    return ref


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Spanish subtitle boundary candidates for benchmark"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="SRT file or directory of SRT files",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--target", type=int, default=250,
        help="Target number of candidates (default: 250)",
    )
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="Only build source provenance manifest, exit",
    )
    args = parser.parse_args()

    print(f"Loading sources from {args.input}")
    sources = _load_sources(args.input)

    if not sources:
        print("ERROR: No usable source files found.")
        sys.exit(1)

    # Build and save source manifest reference (already exists as JSON)
    manifest_sources = build_source_manifest(args.input)
    if not manifest_sources:
        print("WARNING: No source manifest found at benchmarks/spanish_source_manifest.json")

    if args.manifest_only:
        return

    all_boundaries: list[dict[str, Any]] = []
    for source_id, segments, csum in sources:
        print(f"  extracting boundaries from {source_id}...")
        boundaries = _extract_boundaries(segments, source_id, csum)
        print(f"    {len(boundaries)} boundaries extracted")
        _validate_extraction(
            boundaries, segments,
            args.input / f"{source_id}.srt" if args.input.is_dir() else args.input,
        )
        all_boundaries.extend(boundaries)

    print(f"\nTotal raw boundaries: {len(all_boundaries)}")

    # Source-aware sampling with quotas
    sampled = _sample_candidates(
        all_boundaries, sources, manifest_sources, target=args.target,
    )
    print(f"Sampled candidates: {len(sampled)}")

    # Remove fields we said we'd never have
    for b in sampled:
        b.pop("primarySamplingStratum", None)

    # Report stats
    tag_counts: Counter = Counter()
    for b in sampled:
        for tag in b.get("samplingTags", []):
            tag_counts[tag] += 1
    print("\nSampling tag distribution:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")

    timing_counts: Counter = Counter()
    for b in sampled:
        timing_counts[b.get("timingBand", "unknown")] += 1
    print("\nTiming bands:")
    for band, count in timing_counts.most_common():
        print(f"  {band}: {count}")

    dev_count = sum(1 for b in sampled if b.get("split") == "dev")
    test_count = sum(1 for b in sampled if b.get("split") == "test")
    print(f"\nSplit: {dev_count} dev / {test_count} test")

    multiline = sum(
        1 for b in sampled
        if "multiline_cue_left" in b.get("structureTags", [])
        or "multiline_cue_right" in b.get("structureTags", [])
    )
    print(f"Multiline cues: {multiline}")

    chain_ids = set()
    for b in sampled:
        if b.get("chainId"):
            chain_ids.add(b["chainId"])
    print(f"Chains: {len(chain_ids)}")

    unreviewed = sum(1 for b in sampled if b.get("reviewStatus") == "unreviewed")
    print(f"Unreviewed: {unreviewed}")

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(sampled)} candidates to {args.output}")

    # Save reference manifest for portable tests
    ref = build_reference_manifest(sampled, sources)
    ref_path = args.output.parent / "spanish_reference_manifest.json"
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False, indent=2)
    print(f"Saved reference manifest ({len(ref)} cue records) to {ref_path}")

    csv_path = args.output.with_suffix(".csv")
    _generate_review_csv(sampled, csv_path)
    print(f"Saved CSV review to {csv_path}")

    # Dataset report
    report = _generate_dataset_report(sampled, manifest_sources)
    report_path = args.output.parent / "spanish_boundary_dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset report to {report_path}")

    # Summary
    for source_id, _, _ in sources:
        src_count = len([b for b in sampled if b["sourceId"] == source_id])
        print(f"  {source_id}: {src_count} candidates")


if __name__ == "__main__":
    main()
