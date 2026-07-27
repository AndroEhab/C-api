from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

from .sat import SaTSentenceReconstructor, join_segments, text_join_issue


_TIMESTAMP_RE = re.compile(
    r"^(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)[,.](?P<millis>\d{3})$"
)
_CUE_TIMING_RE = re.compile(
    r"^(?P<start>\d+:[0-5]\d:[0-5]\d[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d+:[0-5]\d:[0-5]\d[,.]\d{3})(?:\s+.*)?$"
)

DEFAULT_MAX_MERGE_GAP_MS = 1_500
_LOGGER = logging.getLogger(__name__)
_STRONG_SENTENCE_END_RE = re.compile(r"""[.!?…]+(?:["'’”»)\]}]+)?$""")
_DIALOGUE_DASH_RE = re.compile(r"^\s*(?:--?|[–—]|>>)\s+")
_VOICE_TAG_RE = re.compile(
    r"^\s*(?:<v(?:\s+[^>]*)?>|\[[A-Z][A-Z0-9 ._-]{1,30}\]\s*|"
    r"[A-Z][A-Z0-9 ._-]{1,30}:)"
)
_FORMATTING_TAG_RE = re.compile(r"</?[^>]+>|\{\\[^}]+\}")
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_QUESTION_STARTS = frozenset(
    {
        "am",
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "how",
        "is",
        "may",
        "must",
        "should",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "would",
    }
)
_QUESTION_AUXILIARIES = frozenset(
    {
        "am",
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "may",
        "must",
        "should",
        "was",
        "were",
        "will",
        "would",
    }
)
_CONTINUATION_STARTS = frozenset(
    {
        "and",
        "as",
        "because",
        "but",
        "could",
        "for",
        "from",
        "if",
        "is",
        "nor",
        "of",
        "or",
        "so",
        "than",
        "that",
        "to",
        "unless",
        "until",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "with",
        "without",
        "would",
        "yet",
    }
)
_COMPLEMENT_ENDINGS = frozenset(
    {
        "about",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "onto",
        "than",
        "to",
        "with",
        "without",
    }
)
_RESPONSE_START_RE = re.compile(
    r"^(?:yes|no|nope|yeah|okay|ok|sure|well|of course|I think|I will|I'll)\b",
    re.IGNORECASE,
)
_FINITE_VERB_RE = re.compile(
    r"\b(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would|won't|can't|don't|doesn't|didn't|"
    r"i'm|you're|he's|she's|it's|we're|they're|i'll|you'll|we'll|they'll|"
    r"think|thinks|thought|suspect|suspects|tell|tells|told|want|wants|"
    r"need|needs|know|knows|knew|escaped|issue|live|die)\b",
    re.IGNORECASE,
)


def _visible_text(text: str) -> str:
    return _FORMATTING_TAG_RE.sub("", text).strip()


def _words(text: str) -> list[str]:
    return [match.group(0) for match in _WORD_RE.finditer(_visible_text(text))]


def _ends_strong_sentence(text: str) -> bool:
    return _STRONG_SENTENCE_END_RE.search(_visible_text(text)) is not None


def _has_speaker_marker(text: str) -> bool:
    return _DIALOGUE_DASH_RE.match(text) is not None or _VOICE_TAG_RE.match(text) is not None


def _starts_new_sentence(text: str) -> bool:
    if _has_speaker_marker(text):
        return True
    first_letter = re.search(r"[A-Za-z]", _visible_text(text))
    return first_letter is not None and first_letter.group(0).isupper()


def _looks_like_question(text: str) -> bool:
    visible = _visible_text(text)
    if visible.rstrip(""" "'’”»)]}""").endswith("?"):
        return True
    words = _words(visible)
    if not words:
        return False
    first = words[0].lower()
    return first in _QUESTION_STARTS or (first == "any" and len(words) > 1)


def _likely_complete_clause(text: str) -> bool:
    visible = _visible_text(text)
    return (
        _ends_strong_sentence(visible)
        or _looks_like_question(visible)
        or _FINITE_VERB_RE.search(visible) is not None
    )


def _clear_continuation_boundary(left: str, right: str) -> bool:
    left_words = _words(left)
    right_words = _words(right)
    if not left_words or not right_words:
        return False

    first = right_words[0].lower()
    second = right_words[1].lower() if len(right_words) > 1 else ""
    left_last = left_words[-1].lower()
    first_letter = re.search(r"[A-Za-z]", _visible_text(right))
    starts_lowercase = first_letter is not None and first_letter.group(0).islower()

    if first == "than":
        return True
    if first in {"how", "what", "when", "where", "which", "who", "whom", "whose", "why"}:
        if starts_lowercase or second not in _QUESTION_AUXILIARIES:
            return True
    if left_last in _COMPLEMENT_ENDINGS and (
        starts_lowercase
        or first in _CONTINUATION_STARTS
        or first.endswith("ing")
    ):
        return True
    if _ends_strong_sentence(left):
        return False
    if starts_lowercase or first in _CONTINUATION_STARTS:
        return True
    return not _likely_complete_clause(left) and first in _QUESTION_AUXILIARIES


def _looks_like_answer(text: str) -> bool:
    visible = _visible_text(text)
    return _RESPONSE_START_RE.match(visible) is not None or _starts_new_sentence(visible)


@dataclass(frozen=True, slots=True)
class SubtitleSegment:
    """One original subtitle cue. Its text and timing are never reconstructed."""

    segment_id: str
    text: str
    start_ms: int
    end_ms: int
    speaker: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, str) or not self.segment_id.strip():
            raise ValueError("segment_id must contain non-whitespace text")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must contain non-whitespace text")
        if not isinstance(self.start_ms, int) or not isinstance(self.end_ms, int):
            raise ValueError("subtitle timings must be integers in milliseconds")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("subtitle timing range is invalid")


@dataclass(frozen=True, slots=True)
class ReconstructedSentencePart:
    segment_id: str
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class ReconstructedSentence:
    text: str
    start_ms: int
    end_ms: int
    parts: list[ReconstructedSentencePart]

    @property
    def segment_ids(self) -> list[str]:
        return [part.segment_id for part in self.parts]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "segmentIds": self.segment_ids,
            "parts": [
                {
                    "segmentId": part.segment_id,
                    "text": part.text,
                    "startMs": part.start_ms,
                    "endMs": part.end_ms,
                }
                for part in self.parts
            ],
        }


class InvalidGroupingResponse(ValueError):
    """Raised when the grouping API cannot be mapped to the original cues."""


class SegmentGroupingApi(Protocol):
    def group(self, segments: Sequence[Mapping[str, str]]) -> Any:
        """Group ordered segments and return groups containing segment IDs."""


def _timestamp_to_ms(timestamp: str) -> int:
    match = _TIMESTAMP_RE.fullmatch(timestamp.replace(",", "."))
    if match is None:
        raise ValueError(f"invalid SRT timestamp: {timestamp!r}")
    return (
        int(match["hours"]) * 3_600_000
        + int(match["minutes"]) * 60_000
        + int(match["seconds"]) * 1_000
        + int(match["millis"])
    )


def parse_srt(content: str) -> list[SubtitleSegment]:
    """Parse SRT cues while assigning stable IDs from their cue numbers."""
    if not isinstance(content, str):
        raise ValueError("SRT content must be text")

    segments: list[SubtitleSegment] = []
    seen_ids: set[str] = set()
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip()) if content.strip() else []
    for position, block in enumerate(blocks, start=1):
        lines = [line.rstrip("\r") for line in block.splitlines()]
        if not lines:
            continue
        if _CUE_TIMING_RE.match(lines[0].strip()):
            segment_id = f"cue-{position}"
            timing_line_index = 0
        elif len(lines) >= 2 and _CUE_TIMING_RE.match(lines[1].strip()):
            segment_id = lines[0].strip()
            timing_line_index = 1
        else:
            raise ValueError(f"SRT cue {position} has no valid timing line")

        if not segment_id or segment_id in seen_ids:
            raise ValueError(f"duplicate or empty SRT cue ID: {segment_id!r}")
        timing = _CUE_TIMING_RE.fullmatch(lines[timing_line_index].strip())
        assert timing is not None
        start_ms = _timestamp_to_ms(timing["start"])
        end_ms = _timestamp_to_ms(timing["end"])
        text = " ".join(line.strip() for line in lines[timing_line_index + 1 :] if line.strip())
        if not text:
            raise ValueError(f"SRT cue {segment_id!r} has no text")
        segments.append(SubtitleSegment(segment_id, text, start_ms, end_ms))
        seen_ids.add(segment_id)

    _validate_segments(segments)
    return segments


def parse_srt_file(path: str | Path, *, encoding: str = "utf-8-sig") -> list[SubtitleSegment]:
    return parse_srt(Path(path).read_text(encoding=encoding))


def _validate_segments(segments: Sequence[SubtitleSegment]) -> list[SubtitleSegment]:
    result = list(segments)
    if not result:
        raise ValueError("at least one subtitle segment is required")
    if any(not isinstance(segment, SubtitleSegment) for segment in result):
        raise TypeError("segments must be SubtitleSegment objects")

    ids = [segment.segment_id for segment in result]
    if len(set(ids)) != len(ids):
        raise ValueError("subtitle segment IDs must be unique")
    if any(result[index].start_ms < result[index - 1].start_ms for index in range(1, len(result))):
        raise ValueError("subtitle segments must be ordered by start time")
    return result


def _fallback_sentence(segment: SubtitleSegment) -> ReconstructedSentence:
    part = ReconstructedSentencePart(
        segment_id=segment.segment_id,
        text=segment.text,
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
    )
    return ReconstructedSentence(
        text=segment.text.strip(),
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        parts=[part],
    )


def _segment_ids_from_value(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise InvalidGroupingResponse("group segment IDs must be an array")
    ids: list[str] = []
    for segment_id in value:
        if isinstance(segment_id, Mapping):
            segment_id = segment_id.get("segmentId", segment_id.get("segment_id"))
        if not isinstance(segment_id, str) or not segment_id:
            raise InvalidGroupingResponse("group contains an invalid segment ID")
        ids.append(segment_id)
    if not ids:
        raise InvalidGroupingResponse("groups must not be empty")
    return ids


def _segment_ids_from_indexes(value: Any, segments: Sequence[SubtitleSegment]) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise InvalidGroupingResponse("group segmentIndexes must be an array")
    ids: list[str] = []
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int):
            raise InvalidGroupingResponse("group contains an invalid segment index")
        if index < 0 or index >= len(segments):
            raise InvalidGroupingResponse("group contains an out-of-range segment index")
        ids.append(segments[index].segment_id)
    if not ids:
        raise InvalidGroupingResponse("groups must not be empty")
    return ids


def _ids_from_group(group: Any, segments: Sequence[SubtitleSegment]) -> list[str]:
    if not isinstance(group, Mapping):
        return _segment_ids_from_value(group)

    indexed_ids: list[str] | None = None
    explicit_ids: list[str] | None = None
    for key in ("segmentIndexes", "segment_indexes"):
        if key in group:
            indexed_ids = _segment_ids_from_indexes(group[key], segments)
            break
    for key in ("segmentIds", "segment_ids", "segments", "ids"):
        if key in group:
            explicit_ids = _segment_ids_from_value(group[key])
            break
    if indexed_ids is None and explicit_ids is None:
        raise InvalidGroupingResponse("group has no source segment indexes or IDs")
    if indexed_ids is not None and explicit_ids is not None and indexed_ids != explicit_ids:
        raise InvalidGroupingResponse("group segment indexes and IDs disagree")
    return indexed_ids if indexed_ids is not None else explicit_ids or []


def _groups_from_response(
    response: Any,
    segments: Sequence[SubtitleSegment],
) -> list[list[str]]:
    if isinstance(response, Mapping):
        response = response.get("groups")
    if not isinstance(response, Sequence) or isinstance(response, (str, bytes)):
        raise InvalidGroupingResponse("grouping response must contain groups")
    return [_ids_from_group(group, segments) for group in response]


def _validate_groups(groups: Sequence[Sequence[str]], segments: Sequence[SubtitleSegment]) -> None:
    expected = [segment.segment_id for segment in segments]
    expected_set = set(expected)
    flattened = [segment_id for group in groups for segment_id in group]
    if len(flattened) != len(set(flattened)):
        raise InvalidGroupingResponse("grouping response duplicates or reuses a source cue")
    if any(segment_id not in expected_set for segment_id in flattened):
        raise InvalidGroupingResponse("grouping response contains an unknown source cue")
    if set(flattened) != expected_set:
        raise InvalidGroupingResponse("grouping response is missing a source cue")
    if flattened != expected:
        raise InvalidGroupingResponse(
            "grouping response changes cue order or contains a nonconsecutive group"
        )


def validate_grouping_response(
    response: Any,
    segments: Sequence[SubtitleSegment],
) -> list[list[str]]:
    """Return a complete, ordered source partition or reject it atomically."""
    ordered = _validate_segments(segments)
    if isinstance(response, Mapping) and "groups" not in response:
        groups = _groups_from_sentence_text(response, ordered)
    else:
        groups = _groups_from_response(response, ordered)
    _validate_groups(groups, ordered)
    return groups


def _parts_for_group(group: Sequence[str], by_id: Mapping[str, SubtitleSegment]) -> list[ReconstructedSentencePart]:
    return [
        ReconstructedSentencePart(
            segment_id=segment_id,
            text=by_id[segment_id].text,
            start_ms=by_id[segment_id].start_ms,
            end_ms=by_id[segment_id].end_ms,
        )
        for segment_id in group
    ]


def _merge_rejection_reason(
    group: Sequence[str],
    by_id: Mapping[str, SubtitleSegment],
    max_gap_ms: int,
) -> str | None:
    if len(group) < 2:
        return None

    for left_id, right_id in zip(group, group[1:]):
        left = by_id[left_id]
        right = by_id[right_id]
        gap_ms = right.start_ms - left.end_ms
        if gap_ms > max_gap_ms:
            return f"cue gap {gap_ms}ms exceeds maximum {max_gap_ms}ms"
        if left.speaker != right.speaker and (left.speaker is not None or right.speaker is not None):
            return "adjacent cues have different explicit speakers"
        if _has_speaker_marker(right.text):
            return "next cue has a dialogue or speaker marker"

        join_issue = text_join_issue(left.text, right.text)
        if join_issue is not None:
            return f"unsafe text boundary: {join_issue}"

        clear_continuation = _clear_continuation_boundary(left.text, right.text)
        if (
            _looks_like_question(left.text)
            and _looks_like_answer(right.text)
            and not clear_continuation
        ):
            return "question appears to be followed by an answer"
        if (
            _RESPONSE_START_RE.match(_visible_text(right.text)) is not None
            and _likely_complete_clause(left.text)
            and not clear_continuation
        ):
            return "next cue looks like a conversational response"
        if (
            _likely_complete_clause(left.text)
            and _likely_complete_clause(right.text)
            and (
                _starts_new_sentence(right.text)
                or (
                    _ends_strong_sentence(left.text)
                    and _ends_strong_sentence(right.text)
                )
            )
            and not clear_continuation
        ):
            return "adjacent cues look like independently complete statements"
        if (
            _ends_strong_sentence(left.text)
            and _starts_new_sentence(right.text)
            and not clear_continuation
        ):
            return "first cue ends a sentence and the next starts a new one"

    source_non_space = "".join(
        re.sub(r"\s+", "", by_id[segment_id].text) for segment_id in group
    )
    joined = join_segments([by_id[segment_id].text for segment_id in group])
    if re.sub(r"\s+", "", joined) != source_non_space:
        return "display join would lose or reorder source text"
    return None


class SubtitleSentenceReconstructor:
    """Add validated sentence context without replacing original subtitle cues."""

    def __init__(
        self,
        grouping_api: SegmentGroupingApi | SaTSentenceReconstructor,
        *,
        max_gap_ms: int = DEFAULT_MAX_MERGE_GAP_MS,
        debug: bool = False,
    ) -> None:
        if isinstance(max_gap_ms, bool) or not isinstance(max_gap_ms, int) or max_gap_ms < 0:
            raise ValueError("max_gap_ms must be a non-negative integer")
        self.grouping_api = grouping_api
        self.max_gap_ms = max_gap_ms
        self.debug = debug

    def _log_rejection(self, group: Sequence[str], reason: str) -> None:
        if self.debug:
            _LOGGER.debug("Rejected subtitle group %s: %s", list(group), reason)

    def reconstruct(self, segments: Sequence[SubtitleSegment]) -> list[ReconstructedSentence]:
        ordered = _validate_segments(segments)
        sentences: list[ReconstructedSentence] = []
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end].speaker == ordered[start].speaker:
                end += 1
            sentences.extend(self._reconstruct_run(ordered[start:end]))
            start = end
        return sentences

    def reconstruct_timeline(self, segments: Sequence[SubtitleSegment]) -> "SubtitleTimeline":
        ordered = _validate_segments(segments)
        return SubtitleTimeline(segments=ordered, sentences=self.reconstruct(ordered))

    def _validated_groups(
        self,
        groups: Sequence[Sequence[str]],
        by_id: Mapping[str, SubtitleSegment],
    ) -> list[list[str]]:
        safe_groups: list[list[str]] = []
        blocked_boundaries: set[tuple[str, str]] = set()
        for proposed_group in groups:
            group = list(proposed_group)
            reason = _merge_rejection_reason(group, by_id, self.max_gap_ms)
            if reason is None:
                safe_groups.append(group)
                continue

            self._log_rejection(group, reason)
            blocked_boundaries.update(zip(group, group[1:]))
            safe_groups.extend([[segment_id] for segment_id in group])

        coalesced: list[list[str]] = []
        for group in safe_groups:
            if coalesced:
                boundary = (coalesced[-1][-1], group[0])
                if (
                    boundary not in blocked_boundaries
                    and _clear_continuation_boundary(
                        by_id[boundary[0]].text,
                        by_id[boundary[1]].text,
                    )
                ):
                    candidate = [*coalesced[-1], *group]
                    reason = _merge_rejection_reason(candidate, by_id, self.max_gap_ms)
                    if reason is None:
                        coalesced[-1] = candidate
                        continue
                    self._log_rejection(candidate, reason)
            coalesced.append(group)
        return coalesced

    def _reconstruct_run(self, segments: Sequence[SubtitleSegment]) -> list[ReconstructedSentence]:
        payload = [
            {"segmentId": segment.segment_id, "text": segment.text}
            for segment in segments
        ]
        try:
            if hasattr(self.grouping_api, "group"):
                response = self.grouping_api.group(payload)
            else:
                response = self.grouping_api.reconstruct(
                    [segment.text for segment in segments]
                )
            groups = validate_grouping_response(response, segments)
        except Exception as exc:
            self._log_rejection(
                [segment.segment_id for segment in segments],
                f"invalid or unavailable grouping response ({type(exc).__name__}: {exc})",
            )
            return [_fallback_sentence(segment) for segment in segments]

        by_id = {segment.segment_id: segment for segment in segments}
        groups = self._validated_groups(groups, by_id)
        return [
            ReconstructedSentence(
                text=join_segments([by_id[segment_id].text for segment_id in group]),
                start_ms=by_id[group[0]].start_ms,
                end_ms=by_id[group[-1]].end_ms,
                parts=_parts_for_group(group, by_id),
            )
            for group in groups
        ]


def _groups_from_sentence_texts(
    sentence_texts: Sequence[Any],
    segments: Sequence[SubtitleSegment],
) -> list[list[str]]:
    """Map sentence spans to cues without splitting or duplicating timed cues.

    A detected boundary may fall inside one cue because a cue can contain
    multiple sentences. Sentence ranges that share such a cue are coalesced.
    This keeps every cue in exactly one group while retaining valid cross-cue
    boundaries elsewhere in the same API response.
    """
    if not isinstance(sentence_texts, Sequence) or isinstance(sentence_texts, (str, bytes)):
        raise InvalidGroupingResponse("sentence response must contain sentence text")

    joined = join_segments([segment.text for segment in segments])
    cue_spans: list[tuple[int, int]] = []
    cursor = 0
    for segment in segments:
        text = segment.text.strip()
        start = joined.find(text, cursor)
        if start < 0:
            raise InvalidGroupingResponse("segment text cannot be aligned")
        cue_spans.append((start, start + len(text)))
        cursor = start + len(text)

    sentence_cue_ranges: list[tuple[int, int]] = []
    cursor = 0
    for sentence_text in sentence_texts:
        if not isinstance(sentence_text, str) or not sentence_text.strip():
            raise InvalidGroupingResponse("sentence response contains invalid text")
        sentence = sentence_text.strip()
        while cursor < len(joined) and joined[cursor].isspace():
            cursor += 1
        if not joined.startswith(sentence, cursor):
            raise InvalidGroupingResponse(
                "sentence text cannot be aligned exactly to consecutive source text"
            )
        start = cursor
        end = start + len(sentence)
        overlapping = [
            index
            for index, (cue_start, cue_end) in enumerate(cue_spans)
            if cue_start < end and start < cue_end
        ]
        if not overlapping:
            raise InvalidGroupingResponse("sentence response cannot be mapped to a subtitle segment")
        sentence_cue_ranges.append((overlapping[0], overlapping[-1]))
        cursor = end

    if joined[cursor:].strip():
        raise InvalidGroupingResponse("sentence response loses source text")

    merged_ranges: list[tuple[int, int]] = []
    for first, last in sentence_cue_ranges:
        if merged_ranges and first <= merged_ranges[-1][1]:
            previous_first, previous_last = merged_ranges[-1]
            merged_ranges[-1] = (previous_first, max(previous_last, last))
        else:
            merged_ranges.append((first, last))

    groups = [
        [segments[index].segment_id for index in range(first, last + 1)]
        for first, last in merged_ranges
    ]
    _validate_groups(groups, segments)
    return groups


def _groups_from_sentence_text(
    response: Any,
    segments: Sequence[SubtitleSegment],
) -> list[list[str]]:
    if not isinstance(response, Mapping):
        raise InvalidGroupingResponse("sentence response must be an object")

    sentence_texts = response.get("sentences")
    full_sentence = response.get("full_sentence")
    is_single_sentence = response.get("is_single_sentence")
    if sentence_texts is None:
        if is_single_sentence is True and isinstance(full_sentence, str):
            sentence_texts = [full_sentence]
        else:
            raise InvalidGroupingResponse(
                "legacy sentence response has no unambiguous source sentences"
            )
    if is_single_sentence is not None and not isinstance(is_single_sentence, bool):
        raise InvalidGroupingResponse("is_single_sentence must be a boolean")
    if full_sentence is not None:
        if (
            not isinstance(full_sentence, str)
            or not isinstance(sentence_texts, Sequence)
            or isinstance(sentence_texts, (str, bytes))
            or len(sentence_texts) != 1
            or sentence_texts[0].strip() != full_sentence.strip()
        ):
            raise InvalidGroupingResponse("full_sentence conflicts with sentences")
    if is_single_sentence is True and len(sentence_texts) != 1:
        raise InvalidGroupingResponse("is_single_sentence conflicts with sentences")
    if is_single_sentence is False and full_sentence is not None:
        raise InvalidGroupingResponse("multi-sentence response cannot have full_sentence")
    return _groups_from_sentence_texts(sentence_texts, segments)


@dataclass(slots=True)
class SubtitleTimeline:
    """Player-facing view retaining cue-level timing inside each sentence."""

    segments: list[SubtitleSegment]
    sentences: list[ReconstructedSentence]

    def active_part_at(self, time_ms: int) -> SubtitleSegment | None:
        for segment in self.segments:
            if segment.start_ms <= time_ms < segment.end_ms:
                return segment
        return None

    def active_sentence_at(self, time_ms: int) -> ReconstructedSentence | None:
        active = self.active_part_at(time_ms)
        if active is None:
            return None
        for sentence in self.sentences:
            if active.segment_id in sentence.segment_ids:
                return sentence
        return None

    active_segment_at = active_part_at
