"""Evaluate the production subtitle boundary decisions with the real SaT model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sat import SaTSentenceReconstructor, join_segments  # noqa: E402
from app.subtitles import (  # noqa: E402
    SubtitleSegment,
    SubtitleSentenceReconstructor,
    parse_srt_file,
)


DEFAULT_FIXTURE = PROJECT_ROOT / "benchmarks" / "subtitle_boundary_benchmark.json"
DEFAULT_WINDOW_SRT = PROJECT_ROOT / "Murder Game 2026 iQY WEB.srt"
DEFAULT_WINDOW_CUES = ("168", "169", "170")
REQUIRED_KEYS = frozenset(
    {"left", "right", "previousContext", "nextContext", "gapMs", "expected"}
)
OPTIONAL_REVIEW_KEYS = frozenset({"leftCueId", "rightCueId", "reviewCategory"})
LABELS = frozenset({"join", "break", "ambiguous"})

def _validate_example(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"case {index} must be an object")
    fields = set(value)
    invalid_fields = fields - REQUIRED_KEYS - OPTIONAL_REVIEW_KEYS
    missing = sorted(REQUIRED_KEYS - fields)
    details = []
    if missing:
        details.append(f"missing={missing}")
    if invalid_fields:
        details.append(f"extra={sorted(invalid_fields)}")
    if details:
        raise ValueError(f"case {index} has invalid fields ({', '.join(details)})")

    result = dict(value)
    for field in ("left", "right"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"case {index} {field} must be non-empty text")
    for field in ("previousContext", "nextContext"):
        context = result[field]
        if not isinstance(context, list) or any(
            not isinstance(item, str) or not item.strip() for item in context
        ):
            raise ValueError(f"case {index} {field} must be a list of non-empty strings")
    gap_ms = result["gapMs"]
    if isinstance(gap_ms, bool) or not isinstance(gap_ms, int) or gap_ms < 0:
        raise ValueError(f"case {index} gapMs must be a non-negative integer")
    if result["expected"] not in LABELS:
        raise ValueError(f"case {index} expected must be one of {sorted(LABELS)}")
    for field in OPTIONAL_REVIEW_KEYS & fields:
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"case {index} {field} must be non-empty text")
    return result


def load_fixture(path: Path = DEFAULT_FIXTURE) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("boundary fixture must be a non-empty JSON array")
    return [_validate_example(value, index) for index, value in enumerate(payload, start=1)]


def load_source_srt(path: Path = DEFAULT_WINDOW_SRT) -> dict[str, str]:
    """Load source SRT text by cue ID for fixture validation."""
    return {s.segment_id: s.text for s in parse_srt_file(path)}
def validate_fixture_texts(
    cases: Sequence[Mapping[str, Any]],
    source_texts: dict[str, str],
) -> None:
    """Verify fixture left/right text matches the source SRT for each case."""
    mismatches: list[str] = []
    for index, case in enumerate(cases, start=1):
        for side in ("left", "right"):
            cue_key = f"{side}CueId"
            cue_id = case.get(cue_key)
            if cue_id is None:
                continue
            fixture_text = case[side]
            source_text = source_texts.get(cue_id)
            if source_text is None:
                mismatches.append(
                    f"case {index}: {cue_id} not found in source SRT"
                )
            elif fixture_text.strip() != source_text.strip():
                mismatches.append(
                    f"case {index}: {cue_id} fixture text {fixture_text!r} "
                    f"differs from source {source_text!r}"
                )
    if mismatches:
        raise AssertionError(
            "fixture text validation failed:\n" + "\n".join(mismatches)
        )


def _find_boundary_index(
    segments: Sequence[SubtitleSegment],
    left_cue_id: str,
    right_cue_id: str,
) -> int | None:
    for i in range(len(segments) - 1):
        if segments[i].segment_id == left_cue_id and segments[i + 1].segment_id == right_cue_id:
            return i
    return None


def _build_decisions_from_full_file(
    full_segments: list[SubtitleSegment],
    reconstructor: SubtitleSentenceReconstructor,
) -> list[dict | None]:
    """Run the production windowed pipeline on the full file once."""
    try:
        return reconstructor.evaluate_boundaries(full_segments)
    except Exception:
        return [None] * max(0, len(full_segments) - 1)


def _segments_for_case(case: Mapping[str, Any], index: int) -> tuple[list[SubtitleSegment], str, str]:
    previous_context = list(case["previousContext"])
    texts = previous_context + [case["left"], case["right"]] + list(case["nextContext"])
    left_position = len(previous_context)
    right_position = left_position + 1
    left_id = f"case-{index}-left"
    right_id = f"case-{index}-right"
    ids = [
        *(f"case-{index}-previous-{position}" for position in range(left_position)),
        left_id,
        right_id,
        *(f"case-{index}-next-{position}" for position in range(len(case["nextContext"]))),
    ]

    segments: list[SubtitleSegment] = []
    cursor = 0
    for position, (segment_id, text) in enumerate(zip(ids, texts)):
        if position == right_position:
            cursor += case["gapMs"]
        end = cursor + 1_000
        segments.append(SubtitleSegment(segment_id, text, cursor, end))
        cursor = end
    return segments, left_id, right_id


def _evaluate_case(
    case: Mapping[str, Any],
    index: int,
    reconstructor: SubtitleSentenceReconstructor,
    *,
    full_file_decisions: list[dict] | None = None,
    full_file_segments: list[SubtitleSegment] | None = None,
) -> dict[str, Any]:
    left_cue_id = case.get("leftCueId")
    right_cue_id = case.get("rightCueId")

    # Murder Game cases: use pre-computed full-file decisions
    if left_cue_id and right_cue_id and full_file_decisions is not None and full_file_segments is not None:
        boundary_index = _find_boundary_index(full_file_segments, left_cue_id, right_cue_id)
        if boundary_index is None:
            raise ValueError(
                f"case {index}: boundary ({left_cue_id}/{right_cue_id}) "
                f"not found in full file"
            )
        decision = full_file_decisions[boundary_index]
        if decision is None:
            raise ValueError(f"case {index}: full-file evaluation failed for boundary {boundary_index}")
        predicted = "join" if decision["decision"] == "join" else "break"
        result = {
            "case": index,
            "left": case["left"],
            "right": case["right"],
            "gapMs": case["gapMs"],
            "expected": case["expected"],
            "predicted": predicted,
            "decision": decision["decision"],
            "modelProbability": decision["modelProbability"],
            "reason": decision["reason"],
            "correct": predicted == case["expected"],
        }
        for field in OPTIONAL_REVIEW_KEYS & set(case):
            result[field] = case[field]
        return result

    # Synthetic cases: construct segments and evaluate normally
    segments, left_id, right_id = _segments_for_case(case, index)
    decisions = reconstructor.evaluate_boundaries(segments)
    boundary_index = next(
        (
            position
            for position, dec in enumerate(decisions)
            if dec["leftSegmentId"] == left_id and dec["rightSegmentId"] == right_id
        ),
        None,
    )
    if boundary_index is None:
        raise ValueError(f"case {index} did not return its expected boundary decision")

    decision = decisions[boundary_index]
    predicted = "join" if decision["decision"] == "join" else "break"
    result = {
        "case": index,
        "left": case["left"],
        "right": case["right"],
        "gapMs": case["gapMs"],
        "expected": case["expected"],
        "predicted": predicted,
        "decision": decision["decision"],
        "modelProbability": decision["modelProbability"],
        "reason": decision["reason"],
        "correct": predicted == case["expected"],
    }
    for field in OPTIONAL_REVIEW_KEYS & set(case):
        result[field] = case[field]
    return result


def classify_boundary(
    case: Mapping[str, Any],
    index: int,
    reconstructor: SubtitleSentenceReconstructor,
    *,
    full_file_decisions: list[dict] | None = None,
    full_file_segments: list[SubtitleSegment] | None = None,
) -> str:
    """Classify one fixture boundary using the production windowed path."""
    return _evaluate_case(
        case, index, reconstructor,
        full_file_decisions=full_file_decisions,
        full_file_segments=full_file_segments,
    )["predicted"]


def evaluate_fixture(
    cases: Sequence[Mapping[str, Any]],
    reconstructor: SubtitleSentenceReconstructor | None = None,
    *,
    source_srt_path: Path | None = None,
) -> dict[str, Any]:
    current = reconstructor or SubtitleSentenceReconstructor(SaTSentenceReconstructor())

    # Pre-compute full-file decisions for Murder Game cases
    full_file_decisions: list[dict] | None = None
    full_file_segments: list[SubtitleSegment] | None = None
    if source_srt_path is not None:
        full_file_segments = parse_srt_file(source_srt_path)
        full_file_decisions = _build_decisions_from_full_file(full_file_segments, current)

    case_results = [
        _evaluate_case(
            case, index, current,
            full_file_decisions=full_file_decisions,
            full_file_segments=full_file_segments,
        )
        for index, case in enumerate(cases, start=1)
    ]

    # Separate ambiguous cases
    labeled_results = [
        r for r in case_results
        if cases[r["case"] - 1].get("expected") != "ambiguous"
    ]
    ambiguous_results = [
        r for r in case_results
        if cases[r["case"] - 1].get("expected") == "ambiguous"
    ]

    expected_join = sum(cases[r["case"] - 1]["expected"] == "join" for r in labeled_results)
    predicted_join = sum(r["predicted"] == "join" for r in labeled_results)
    true_join = sum(
        cases[r["case"] - 1]["expected"] == "join" and r["predicted"] == "join"
        for r in labeled_results
    )
    expected_break = sum(cases[r["case"] - 1]["expected"] == "break" for r in labeled_results)
    true_break = sum(
        cases[r["case"] - 1]["expected"] == "break" and r["predicted"] == "break"
        for r in labeled_results
    )
    false_merges = sum(
        cases[r["case"] - 1]["expected"] == "break" and r["predicted"] == "join"
        for r in labeled_results
    )
    false_negatives = sum(
        cases[r["case"] - 1]["expected"] == "join" and r["predicted"] != "join"
        for r in labeled_results
    )
    total = len(case_results)
    misclassified = [r for r in case_results if not r["correct"]]

    return {
        "metrics": {
            "totalBoundaries": total,
            "cases": total,
            "ambiguousCount": len(ambiguous_results),
            "expectedJoin": expected_join,
            "expectedBreak": expected_break,
            "predictedJoin": predicted_join,
            "trueJoin": true_join,
            "trueBreak": true_break,
            "joinPrecision": true_join / predicted_join if predicted_join else 0.0,
            "joinRecall": true_join / expected_join if expected_join else 0.0,
            "breakAccuracy": true_break / expected_break if expected_break else 0.0,
            "falseMergeCount": false_merges,
            "falseNegativeCount": false_negatives,
        },
        "cases": case_results,
        "misclassified": misclassified,
        "ambiguous": ambiguous_results,
    }


def load_real_window(
    path: Path = DEFAULT_WINDOW_SRT,
    cue_ids: Sequence[str] = DEFAULT_WINDOW_CUES,
) -> list[SubtitleSegment]:
    requested_ids = [str(cue_id).strip() for cue_id in cue_ids if str(cue_id).strip()]
    if len(requested_ids) < 2 or len(set(requested_ids)) != len(requested_ids):
        raise ValueError("the real subtitle window must contain at least two unique cue IDs")

    source_segments = parse_srt_file(path)
    by_id = {segment.segment_id: (position, segment) for position, segment in enumerate(source_segments)}
    missing = [cue_id for cue_id in requested_ids if cue_id not in by_id]
    if missing:
        raise ValueError(f"window cue IDs were not found in {path}: {missing}")

    selected_positions = [by_id[cue_id][0] for cue_id in requested_ids]
    if selected_positions != list(range(selected_positions[0], selected_positions[0] + len(selected_positions))):
        raise ValueError("window cue IDs must identify consecutive cues in source order")
    return [by_id[cue_id][1] for cue_id in requested_ids]


def _output_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return [int(dimension) for dimension in tuple(shape)]
        except (TypeError, ValueError):
            pass
    try:
        return [len(value)]
    except TypeError:
        return None


def _expected_boundary_offsets(segments: Sequence[SubtitleSegment]) -> list[int]:
    return [len(join_segments([segment.text for segment in segments[:index + 1]])) - 1 for index in range(len(segments) - 1)]
def _boundary_offsets_match(
    segments: Sequence[SubtitleSegment],
    joined_text: str,
    actual_offsets: Sequence[int],
    expected_offsets: Sequence[int],
) -> bool:
    if list(actual_offsets) != list(expected_offsets):
        return False
    for index, offset in enumerate(actual_offsets):
        left_text = segments[index].text.strip()
        right_text = segments[index + 1].text.strip()
        pair_prefix = join_segments(
            [segment.text for segment in segments[: index + 2]]
        )
        if (
            not left_text
            or not right_text
            or offset < 0
            or offset >= len(joined_text)
            or not joined_text.startswith(pair_prefix)
            or joined_text[offset] != left_text[-1]
        ):
            return False
        right_start = len(pair_prefix) - len(right_text)
        if (
            right_start <= offset
            or joined_text[right_start : right_start + len(right_text)] != right_text
        ):
            return False
    return True




def verify_real_window(
    sat_service: SaTSentenceReconstructor,
    segments: Sequence[SubtitleSegment],
) -> dict[str, Any]:
    texts = [segment.text for segment in segments]
    joined_text = join_segments(texts)
    model = sat_service._get_model()
    evidence = sat_service.score_boundaries(texts)
    raw_probabilities = model.predict_proba(joined_text)
    raw_shape = _output_shape(raw_probabilities)
    expected_count = max(0, len(segments) - 1)
    expected_offsets = _expected_boundary_offsets(segments)
    actual_offsets = [entry["characterOffset"] for entry in evidence]
    probabilities = [entry["boundaryProbability"] for entry in evidence]

    scores_count_ok = len(evidence) == expected_count
    probabilities_valid = all(
        isinstance(probability, (int, float))
        and not isinstance(probability, bool)
        and math.isfinite(float(probability))
        and 0.0 <= float(probability) <= 1.0
        for probability in probabilities
    )
    offsets_match = _boundary_offsets_match(
        segments,
        joined_text,
        actual_offsets,
        expected_offsets,
    )
    shape_handled = raw_shape in (
        [len(joined_text)],
        [len(joined_text), 1],
    )
    checks = {
        "exactlyNMinusOneScores": scores_count_ok,
        "probabilitiesFiniteAndInRange": probabilities_valid,
        "characterOffsetsMatchCueBoundaries": offsets_match,
        "predictProbaShapeHandled": shape_handled,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"real score_boundaries verification failed: {failed}")

    return {
        "cueIds": [segment.segment_id for segment in segments],
        "texts": texts,
        "joinedText": joined_text,
        "scoreCount": len(evidence),
        "expectedScoreCount": expected_count,
        "predictProbaOutputShape": raw_shape,
        "joinedTextCharacterCount": len(joined_text),
        "checks": checks,
        "boundaries": [
            {
                "leftCueId": segments[index].segment_id,
                "rightCueId": segments[index + 1].segment_id,
                "characterOffset": entry["characterOffset"],
                "expectedCharacterOffset": expected_offsets[index],
                "boundaryProbability": entry["boundaryProbability"],
            }
            for index, entry in enumerate(evidence)
        ],
    }


def _print_report(report: Mapping[str, Any]) -> None:
    metrics = report["metrics"]
    print("Subtitle boundary classification benchmark")
    print(f"model={report['model']} fixture={report['fixture']}")
    print(
        f"total_boundaries={metrics['totalBoundaries']} "
        f"expected_join={metrics['expectedJoin']} expected_break={metrics['expectedBreak']}"
    )
    if metrics.get("ambiguousCount"):
        print(f"ambiguous={metrics['ambiguousCount']}")
    print(f"JOIN precision={metrics['joinPrecision']:.3f} recall={metrics['joinRecall']:.3f}")
    print(f"BREAK accuracy={metrics['breakAccuracy']:.3f}")
    print(
        f"false_merges={metrics['falseMergeCount']} "
        f"false_negatives={metrics['falseNegativeCount']}"
    )
    if report["misclassified"]:
        print("misclassified:")
        for result in report["misclassified"]:
            print(
                f"  case={result['case']} expected={result['expected']} "
                f"predicted={result['predicted']} "
                f"model_probability={result['modelProbability']} "
                f"left={result['left']!r} right={result['right']!r} "
                f"reason={result['reason']}"
            )
    else:
        print("misclassified: none")

    if report.get("ambiguous"):
        print("ambiguous:")
        for result in report["ambiguous"]:
            print(
                f"  case={result['case']} expected={result['expected']} "
                f"predicted={result['predicted']} "
                f"model_probability={result['modelProbability']} "
                f"left={result['left']!r} right={result['right']!r}"
            )

    if "realWindow" in report:
        window = report["realWindow"]
        print("real score_boundaries verification")
        print(
            f"  cues={','.join(window['cueIds'])} scores={window['scoreCount']} "
            f"expected={window['expectedScoreCount']} "
            f"predict_proba_shape={tuple(window['predictProbaOutputShape'])} "
            f"checks={all(window['checks'].values())}"
        )


def _fixture_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, help="write the detailed baseline report as JSON")
    parser.add_argument("--window-srt", type=Path, default=DEFAULT_WINDOW_SRT)
    parser.add_argument(
        "--window-cues",
        default=",".join(DEFAULT_WINDOW_CUES),
        help="comma-separated consecutive cue IDs for the real score_boundaries check",
    )
    args = parser.parse_args()

    cases = load_fixture(args.fixture)
    source_texts = load_source_srt(args.window_srt)
    validate_fixture_texts(cases, source_texts)
    sat_service = SaTSentenceReconstructor()
    report = evaluate_fixture(
        cases,
        SubtitleSentenceReconstructor(sat_service),
        source_srt_path=args.window_srt,
    )
    window_segments = load_real_window(args.window_srt, args.window_cues.split(","))
    report = {
        "implementation": "SubtitleSentenceReconstructor(SaTSentenceReconstructor).evaluate_boundaries",
        "model": sat_service.model_name,
        "fixture": _fixture_name(args.fixture),
        "windowSrt": _fixture_name(args.window_srt),
        **report,
        "realWindow": verify_real_window(sat_service, window_segments),
    }
    _print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"saved={args.output}")
