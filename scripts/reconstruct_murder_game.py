"""Reconstruct the Murder Game subtitle file and emit policy diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sat import SaTSentenceReconstructor, join_segments  # noqa: E402
from app.subtitles import (  # noqa: E402
    BoundaryDecision,
    ReconstructedSentence,
    SubtitleSegment,
    SubtitleSentenceReconstructor,
    _ends_strong_sentence,
    _likely_complete_clause,
    _looks_like_question,
    _continuation_structure_score,
    parse_srt_file,
)


DEFAULT_SOURCE = PROJECT_ROOT / "Murder Game 2026 iQY WEB.srt"
DEFAULT_OUTPUT = PROJECT_ROOT / "Murder Game 2026 iQY WEB_reconstructed.srt"
DEFAULT_REPORT = PROJECT_ROOT / "benchmarks" / "murder_game_reconstruction_task4.json"


def _srt_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _write_srt(path: Path, sentences: list[ReconstructedSentence]) -> None:
    blocks = []
    for output_index, sentence in enumerate(sentences, start=1):
        blocks.append(
            "\n".join(
                [
                    str(output_index),
                    f"{_srt_timestamp(sentence.start_ms)} --> {_srt_timestamp(sentence.end_ms)}",
                    sentence.text,
                ]
            )
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _non_space_text(text: str) -> str:
    return re.sub(r"\s+", "", text)

def _appears_independently_complete(text: str) -> bool:
    if _ends_strong_sentence(text) or _looks_like_question(text):
        return True
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)
    if len(words) == 1:
        return True
    first_letter = re.search(r"[A-Za-z]", text)
    return (
        first_letter is not None
        and first_letter.group(0).isupper()
        and _likely_complete_clause(text)
    )


def _decision_diagnostics(
    segments: list[SubtitleSegment],
    decisions: list[BoundaryDecision],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        if decision["decision"] != "join":
            continue
        left = segments[index]
        right = segments[index + 1]
        diagnostics.append(
            {
                "leftCueId": left.segment_id,
                "rightCueId": right.segment_id,
                "leftText": left.text,
                "rightText": right.text,
                "gapMs": decision["gapMs"],
                "satProbability": decision["modelProbability"],
                "continuationScore": _continuation_structure_score(left.text, right.text),
                "finalReason": decision["reason"],
            }
        )
    return diagnostics


def _risk_record(
    left: SubtitleSegment,
    right: SubtitleSegment,
    decision: BoundaryDecision,
    continuation_score: int,
) -> dict[str, Any]:
    left_terminal = _ends_strong_sentence(left.text)
    both_complete = (
        continuation_score == 0
        and _appears_independently_complete(left.text)
        and _appears_independently_complete(right.text)
    )
    return {
        "leftCueId": left.segment_id,
        "rightCueId": right.segment_id,
        "leftText": left.text,
        "rightText": right.text,
        "gapMs": decision["gapMs"],
        "satProbability": decision["modelProbability"],
        "continuationScore": continuation_score,
        "finalReason": decision["reason"],
        "leftEndsTerminalPunctuation": left_terminal,
        "bothCuesAppearIndependentlyComplete": both_complete,
    }


def _source_comparison(
    segments: list[SubtitleSegment],
    sentences: list[ReconstructedSentence],
    decisions: list[BoundaryDecision],
) -> dict[str, Any]:
    source_ids = [segment.segment_id for segment in segments]
    source_by_id = {segment.segment_id: segment for segment in segments}
    reconstructed_ids = [segment_id for sentence in sentences for segment_id in sentence.segment_ids]
    source_text = join_segments([segment.text for segment in segments])
    reconstructed_text = join_segments([sentence.text for sentence in sentences])
    duplicate_ids = sorted(
        segment_id for segment_id, count in Counter(reconstructed_ids).items() if count > 1
    )
    missing_ids = [segment_id for segment_id in source_ids if segment_id not in reconstructed_ids]
    reordered = reconstructed_ids != source_ids
    text_loss = _non_space_text(source_text) != _non_space_text(reconstructed_text)
    timing_loss = any(
        part.start_ms != next(
            segment.start_ms for segment in segments if segment.segment_id == part.segment_id
        )
        or part.end_ms != next(
            segment.end_ms for segment in segments if segment.segment_id == part.segment_id
        )
        for sentence in sentences
        for part in sentence.parts
    )
    sentence_range_timing_loss = any(
        sentence.start_ms != source_by_id[sentence.parts[0].segment_id].start_ms
        or sentence.end_ms != source_by_id[sentence.parts[-1].segment_id].end_ms
        for sentence in sentences
    )
    internal_cue_timings_collapsed = any(len(sentence.parts) > 1 for sentence in sentences)
    return {
        "originalCueCount": len(segments),
        "boundaryCount": len(decisions),
        "boundariesJoined": sum(decision["decision"] == "join" for decision in decisions),
        "boundariesBroken": sum(decision["decision"] != "join" for decision in decisions),
        "sentencesCreated": len(sentences),
        "sourceCueIdsPreservedInOrder": not reordered and not missing_ids and not duplicate_ids,
        "missingCueIds": missing_ids,
        "duplicateCueIds": duplicate_ids,
        "cueOrderChanged": reordered,
        "textLoss": text_loss,
        "textDuplication": bool(duplicate_ids),
        "textReordering": reordered,
        "timingLossInRetainedCueParts": timing_loss,
        "timingLossInOutputSentenceRanges": sentence_range_timing_loss,
        "internalCueTimingsCollapsedInSrt": internal_cue_timings_collapsed,
        "sourceNonSpaceTextLength": len(_non_space_text(source_text)),
        "reconstructedNonSpaceTextLength": len(_non_space_text(reconstructed_text)),
    }


def generate(source: Path, output: Path, report_path: Path) -> dict[str, Any]:
    segments = parse_srt_file(source)
    sat_service = SaTSentenceReconstructor()
    reconstructor = SubtitleSentenceReconstructor(sat_service)
    decisions = reconstructor.evaluate_boundaries(segments)
    sentences = reconstructor.build_sentences(segments, decisions)
    _write_srt(output, sentences)

    join_diagnostics = _decision_diagnostics(segments, decisions)
    high_risk_joins: list[dict[str, Any]] = []
    joins_over_1500ms: list[dict[str, Any]] = []
    joins_with_terminal_left: list[dict[str, Any]] = []
    joins_with_both_complete: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        if decision["decision"] != "join":
            continue
        left = segments[index]
        right = segments[index + 1]
        continuation_score = _continuation_structure_score(left.text, right.text)
        record = _risk_record(left, right, decision, continuation_score)
        if decision["gapMs"] > 1_500 or record["leftEndsTerminalPunctuation"] or record[
            "bothCuesAppearIndependentlyComplete"
        ] or (decision["modelProbability"] is not None and decision["modelProbability"] > 0.5):
            high_risk_joins.append(record)
        if decision["gapMs"] > 1_500:
            joins_over_1500ms.append(record)
        if record["leftEndsTerminalPunctuation"]:
            joins_with_terminal_left.append(record)
        if record["bothCuesAppearIndependentlyComplete"]:
            joins_with_both_complete.append(record)

    report = {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "output": str(output.relative_to(PROJECT_ROOT)),
        "model": sat_service.model_name,
        "policy": "BoundaryPolicyConfig defaults",
        "comparison": _source_comparison(segments, sentences, decisions),
        "highRiskJoins": high_risk_joins,
        "joinsAcrossMoreThan1500ms": joins_over_1500ms,
        "joinsWhereLeftEndsWithTerminalPunctuation": joins_with_terminal_left,
        "joinsWhereBothCuesAppearIndependentlyComplete": joins_with_both_complete,
        "joinDiagnostics": join_diagnostics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = generate(args.source, args.output, args.report)
    comparison = report["comparison"]
    print(
        f"model={report['model']} cues={comparison['originalCueCount']} "
        f"joined={comparison['boundariesJoined']} sentences={comparison['sentencesCreated']}"
    )
    print(
        f"high_risk={len(report['highRiskJoins'])} "
        f"over_1500ms={len(report['joinsAcrossMoreThan1500ms'])} "
        f"terminal_left={len(report['joinsWhereLeftEndsWithTerminalPunctuation'])} "
        f"both_complete={len(report['joinsWhereBothCuesAppearIndependentlyComplete'])}"
    )
    print(
        f"text_loss={comparison['textLoss']} duplication={comparison['textDuplication']} "
        f"reordering={comparison['textReordering']} timing_loss={comparison['timingLossInRetainedCueParts']}"
    )
    print(f"saved={args.output}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
