from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Protocol, Sequence, TypedDict

from .sat import SaTSentenceReconstructor, join_segments, text_join_issue


_TIMESTAMP_RE = re.compile(
    r"^(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)[,.](?P<millis>\d{3})$"
)
_CUE_TIMING_RE = re.compile(
    r"^(?P<start>\d+:[0-5]\d:[0-5]\d[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d+:[0-5]\d:[0-5]\d[,.]\d{3})(?:\s+.*)?$"
)

@dataclass(frozen=True, slots=True)
class BoundaryPolicyConfig:
    """Initial defaults; normal/medium/large/extreme ranges require benchmark calibration."""

    normal_gap_max_ms: int = 500
    medium_gap_max_ms: int = 1_500
    extreme_gap_ms: int = 30_000
    normal_model_join_max_probability: float = 0.50
    medium_model_join_max_probability: float = 0.20
    large_model_join_max_probability: float = 0.05
    continuation_override_max_probability: float = 0.95
    min_continuation_score: int = 3

    def __post_init__(self) -> None:
        for name in ("normal_gap_max_ms", "medium_gap_max_ms", "extreme_gap_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.normal_gap_max_ms > self.medium_gap_max_ms:
            raise ValueError("normal_gap_max_ms must not exceed medium_gap_max_ms")
        if self.medium_gap_max_ms >= self.extreme_gap_ms:
            raise ValueError("medium_gap_max_ms must be less than extreme_gap_ms")

        for name in (
            "normal_model_join_max_probability",
            "medium_model_join_max_probability",
            "large_model_join_max_probability",
            "continuation_override_max_probability",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if (
            self.normal_model_join_max_probability
            < self.medium_model_join_max_probability
            or self.medium_model_join_max_probability
            < self.large_model_join_max_probability
        ):
            raise ValueError(
                "model join thresholds must become stricter as the timing gap grows"
            )
        if (
            isinstance(self.min_continuation_score, bool)
            or not isinstance(self.min_continuation_score, int)
            or self.min_continuation_score < 1
        ):
            raise ValueError("min_continuation_score must be a positive integer")


DEFAULT_BOUNDARY_POLICY_CONFIG = BoundaryPolicyConfig()
_LOGGER = logging.getLogger(__name__)
_STRONG_SENTENCE_END_RE = re.compile(r"""[.!?…]+(?:["'’”»)\]}]+)?$""")
_DIALOGUE_DASH_RE = re.compile(r"^\s*(?:--?|[–—]|>>)\s+")
_VOICE_TAG_RE = re.compile(
    r"^\s*(?:<v(?:\s+[^>]*)?>|\[[A-Z][A-Z0-9 ._-]{1,30}\]\s*|"
    r"[A-Z][A-Z0-9 ._-]{1,30}:)"
)
_FORMATTING_TAG_RE = re.compile(r"</?[^>]+>|\{\\[^}]+\}")
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_CUE_ID_SUFFIX_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")
_SUBORDINATE_STARTS = frozenset(
    {"although", "because", "if", "since", "though", "unless", "when", "while"}
)
_CONTINUATION_STARTS = frozenset(
    {"and", "as", "because", "but", "for", "if", "nor", "or", "so", "that", "while"}
)
_CONTINUATION_PRONOUNS = frozenset(
    {"her", "his", "its", "my", "our", "their", "the", "this", "that", "your"}
)
_WH_STARTS = frozenset({"how", "what", "when", "where", "which", "who", "whom", "whose", "why"})
_INCOMPLETE_ENDINGS = frozenset(
    {"a", "an", "and", "as", "at", "because", "but", "for", "if", "of", "or", "than", "to", "with"}
)
_COMPLEMENT_CONTINUATION_RE = re.compile(
    r"\b(?:i['’]?ll|i\s+will|i\s+can)\s+"
    r"(?:tell|show|explain)\s+(?:you|him|her|them)\b",
    re.IGNORECASE,
)
_COMPARATIVE_CONTINUATION_RE = re.compile(r"^\s*better\s+to\b", re.IGNORECASE)
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
    if _ends_strong_sentence(visible) or _looks_like_question(visible):
        return True
    words = _words(visible)
    return len(words) >= 2 and _FINITE_VERB_RE.search(visible) is not None


def _appears_incomplete(text: str) -> bool:
    """Return weak evidence that a cue is a fragment of a larger clause."""
    visible = _visible_text(text)
    words = _words(visible)
    if not words or _ends_strong_sentence(visible) or _looks_like_question(visible):
        return False
    first = words[0].casefold()
    last = words[-1].casefold()
    return (
        len(words) == 1
        or last in _INCOMPLETE_ENDINGS
        or first in _SUBORDINATE_STARTS
        or re.search(r"[a-z]$", visible) is not None
    )


def _continuation_structure_score(left: str, right: str) -> int:
    """Score multi-signal continuation patterns without keyword-only joins."""
    left_visible = _visible_text(left)
    right_visible = _visible_text(right)
    left_words = _words(left_visible)
    right_words = _words(right_visible)
    if not left_words or not right_words:
        return 0

    first_right = right_words[0].casefold()
    first_left = left_words[0].casefold()
    score = 0
    first_letter = re.search(r"[A-Za-z]", right_visible)
    if first_letter is not None and first_letter.group(0).islower() and len(right_words) >= 2:
        score += 2

    if (
        first_right == "than"
        and len(right_words) >= 3
        and _COMPARATIVE_CONTINUATION_RE.search(left_visible) is not None
    ):
        score += 3

    if (
        first_left in _SUBORDINATE_STARTS
        and first_right in _CONTINUATION_PRONOUNS
        and len(left_words) >= 3
        and len(right_words) >= 3
        and _likely_complete_clause(right_visible)
    ):
        score += 3

    if (
        first_right in _WH_STARTS
        and len(right_words) >= 3
        and _starts_new_sentence(right_visible)
        and _COMPLEMENT_CONTINUATION_RE.search(left_visible) is not None
    ):
        score += 4

    if (
        first_right in _CONTINUATION_STARTS
        and not _ends_strong_sentence(left_visible)
        and len(right_words) >= 2
    ):
        score += 2
    return score


def _looks_like_independent_statements(
    left: str,
    right: str,
    continuation_score: int,
) -> bool:
    if continuation_score:
        return False
    if not (_likely_complete_clause(left) and _likely_complete_clause(right)):
        return False

    left_incomplete = _appears_incomplete(left)
    right_incomplete = _appears_incomplete(right)
    right_visible = _visible_text(right)
    right_words = _words(right_visible)
    right_starts_lowercase_question = (
        bool(right_words)
        and right_words[0].casefold() in _WH_STARTS
        and _ends_strong_sentence(right_visible)
    )
    if left_incomplete and not (
        _ends_strong_sentence(right_visible)
        and (_starts_new_sentence(right_visible) or right_starts_lowercase_question)
    ):
        return False
    if right_incomplete and not _ends_strong_sentence(right_visible):
        return False

    right_starts_sentence = _starts_new_sentence(right_visible)
    return right_starts_sentence or right_starts_lowercase_question

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
    """Raised when a model response cannot be mapped to the original cues."""


BoundaryDecisionValue = Literal["join", "break", "uncertain"]


class BoundaryEvidence(TypedDict):
    """Model evidence for one original subtitle cue boundary."""

    leftSegmentId: str
    rightSegmentId: str
    modelProbability: float | None


class BoundaryDecision(TypedDict):
    """One independent decision for an original subtitle cue boundary."""

    leftSegmentId: str
    rightSegmentId: str
    decision: BoundaryDecisionValue
    reason: str
    modelProbability: float | None
    gapMs: int


class BoundaryScoringApi(Protocol):
    def score_boundaries(self, segments: Sequence[str]) -> Any:
        """Return model evidence for every original cue boundary."""


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


def _boundary_value(entry: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in entry:
            return entry[key]
    return None


def _normalise_model_probability(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, str, bytes)):
        raise InvalidGroupingResponse("boundary model probability must be numeric")
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidGroupingResponse("boundary model probability must be numeric") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise InvalidGroupingResponse("boundary model probability must be within [0, 1]")
    return probability


def _normalise_boundary_evidence(
    response: Any,
    segments: Sequence[SubtitleSegment],
) -> list[BoundaryEvidence]:
    if isinstance(response, Mapping):
        for key in ("boundaries", "evidence", "scores"):
            if key in response:
                response = response[key]
                break
        else:
            raise InvalidGroupingResponse("boundary response must contain evidence")
    if not isinstance(response, Sequence) or isinstance(response, (str, bytes)):
        raise InvalidGroupingResponse("boundary response must contain an array of evidence")

    entries = list(response)
    expected_count = max(0, len(segments) - 1)
    if len(entries) != expected_count:
        raise InvalidGroupingResponse(
            f"expected {expected_count} boundary evidence items, received {len(entries)}"
        )

    evidence: list[BoundaryEvidence] = []
    for boundary_index, entry in enumerate(entries):
        left_segment_id = segments[boundary_index].segment_id
        right_segment_id = segments[boundary_index + 1].segment_id
        if isinstance(entry, Mapping):
            supplied_left_id = _boundary_value(entry, "leftSegmentId", "left_segment_id")
            supplied_right_id = _boundary_value(entry, "rightSegmentId", "right_segment_id")
            if supplied_left_id is not None and supplied_left_id != left_segment_id:
                raise InvalidGroupingResponse("boundary evidence changes cue order")
            if supplied_right_id is not None and supplied_right_id != right_segment_id:
                raise InvalidGroupingResponse("boundary evidence changes cue order")

            supplied_left_index = _boundary_value(entry, "leftIndex", "left_index")
            supplied_right_index = _boundary_value(entry, "rightIndex", "right_index")
            if supplied_left_index is not None and supplied_left_index != boundary_index:
                raise InvalidGroupingResponse("boundary evidence changes cue order")
            if supplied_right_index is not None and supplied_right_index != boundary_index + 1:
                raise InvalidGroupingResponse("boundary evidence changes cue order")
            raw_probability = _boundary_value(
                entry,
                "modelProbability",
                "model_probability",
                "boundaryProbability",
                "boundary_probability",
                "probability",
            )
        else:
            raw_probability = entry

        try:
            model_probability = _normalise_model_probability(raw_probability)
        except InvalidGroupingResponse:
            # Keep neighboring boundaries usable when one score is malformed.
            model_probability = None
        evidence.append(
            {
                "leftSegmentId": left_segment_id,
                "rightSegmentId": right_segment_id,
                "modelProbability": model_probability,
            }
        )
    return evidence


def _source_cue_order_reason(left: SubtitleSegment, right: SubtitleSegment) -> str | None:
    if right.start_ms < left.start_ms:
        return "cue order is invalid"

    left_match = _CUE_ID_SUFFIX_RE.fullmatch(left.segment_id.strip())
    right_match = _CUE_ID_SUFFIX_RE.fullmatch(right.segment_id.strip())
    if left_match is None or right_match is None:
        return None
    if left_match["prefix"] != right_match["prefix"]:
        return None

    left_number = int(left_match["number"])
    right_number = int(right_match["number"])
    if right_number <= left_number:
        return "cue order is invalid"
    if right_number != left_number + 1:
        return "source cues are not consecutive"
    return None


def _text_corruption_reason(left: SubtitleSegment, right: SubtitleSegment) -> str | None:
    join_issue = text_join_issue(left.text, right.text)
    if join_issue is not None:
        return f"joining would corrupt text: {join_issue}"

    source_non_space = re.sub(r"\s+", "", left.text + right.text)
    joined = join_segments([left.text, right.text])
    if re.sub(r"\s+", "", joined) != source_non_space:
        return "joining would corrupt text: display join would lose or reorder source text"
    return None


def _boundary_hard_break_reason(
    left: SubtitleSegment,
    right: SubtitleSegment,
    gap_ms: int,
    config: BoundaryPolicyConfig,
) -> str | None:
    if left.speaker != right.speaker and (left.speaker is not None or right.speaker is not None):
        return "adjacent cues have different explicit speakers"
    if _has_speaker_marker(right.text):
        return "next cue has a dialogue or voice marker"

    source_order_reason = _source_cue_order_reason(left, right)
    if source_order_reason is not None:
        return source_order_reason
    if gap_ms > config.extreme_gap_ms:
        return (
            f"cue gap {gap_ms}ms exceeds extreme-gap threshold "
            f"{config.extreme_gap_ms}ms"
        )
    return _text_corruption_reason(left, right)


def _gap_band(gap_ms: int, config: BoundaryPolicyConfig) -> str:
    if gap_ms <= config.normal_gap_max_ms:
        return "normal"
    if gap_ms <= config.medium_gap_max_ms:
        return "medium"
    if gap_ms < config.extreme_gap_ms:
        return "large"
    return "extreme"


def _model_join_threshold(gap_band: str, config: BoundaryPolicyConfig) -> float:
    if gap_band == "normal":
        return config.normal_model_join_max_probability
    if gap_band == "medium":
        return config.medium_model_join_max_probability
    return config.large_model_join_max_probability


def _soft_boundary_decision(
    left: SubtitleSegment,
    right: SubtitleSegment,
    gap_ms: int,
    model_probability: float,
    config: BoundaryPolicyConfig,
) -> tuple[BoundaryDecisionValue, str]:
    gap_band = _gap_band(gap_ms, config)
    continuation_score = _continuation_structure_score(left.text, right.text)
    strong_continuation = continuation_score >= config.min_continuation_score
    independent_statements = _looks_like_independent_statements(
        left.text,
        right.text,
        continuation_score if strong_continuation else 0,
    )

    if (
        strong_continuation
        and model_probability <= config.continuation_override_max_probability
        and (
            gap_band == "normal"
            or model_probability < _model_join_threshold(gap_band, config)
        )
    ):
        return (
            "join",
            "continuation structure outweighs advisory punctuation, capitalization, "
            f"and model boundary evidence (score={continuation_score})",
        )

    if independent_statements:
        return (
            "break",
            "soft evidence indicates independently complete statements "
            "(terminal punctuation/capitalization)",
        )

    if gap_band == "large":
        return "break", f"break by default for large cue gap {gap_ms}ms"
    if gap_band == "extreme":
        return "break", f"break for extreme cue gap {gap_ms}ms"

    threshold = _model_join_threshold(gap_band, config)
    if model_probability < threshold:
        return "join", "model probability favors continuation"
    if model_probability > threshold:
        return "break", "model probability favors a sentence boundary"
    return "uncertain", "soft boundary evidence is exactly ambiguous"


class SubtitleSentenceReconstructor:
    """Reconstruct sentences from independent decisions at cue boundaries."""

    def __init__(
        self,
        boundary_api: BoundaryScoringApi | SaTSentenceReconstructor,
        *,
        policy_config: BoundaryPolicyConfig | None = None,
        max_gap_ms: int | None = None,
        debug: bool = False,
    ) -> None:
        if policy_config is not None and not isinstance(policy_config, BoundaryPolicyConfig):
            raise TypeError("policy_config must be a BoundaryPolicyConfig")
        if max_gap_ms is not None:
            if isinstance(max_gap_ms, bool) or not isinstance(max_gap_ms, int) or max_gap_ms < 0:
                raise ValueError("max_gap_ms must be a non-negative integer")
            if policy_config is not None:
                raise ValueError("max_gap_ms cannot be combined with policy_config")
            default = DEFAULT_BOUNDARY_POLICY_CONFIG
            policy_config = BoundaryPolicyConfig(
                normal_gap_max_ms=min(default.normal_gap_max_ms, max_gap_ms),
                medium_gap_max_ms=max_gap_ms,
                extreme_gap_ms=max(default.extreme_gap_ms, max_gap_ms + 1),
                normal_model_join_max_probability=default.normal_model_join_max_probability,
                medium_model_join_max_probability=default.medium_model_join_max_probability,
                large_model_join_max_probability=default.large_model_join_max_probability,
                continuation_override_max_probability=default.continuation_override_max_probability,
                min_continuation_score=default.min_continuation_score,
            )
        self.boundary_api = boundary_api
        self.policy_config = policy_config or DEFAULT_BOUNDARY_POLICY_CONFIG
        self.debug = debug

    def _log_boundary_issue(
        self,
        left_segment_id: str,
        right_segment_id: str,
        reason: str,
    ) -> None:
        if self.debug:
            _LOGGER.debug(
                "Rejected subtitle boundary %s/%s: %s",
                left_segment_id,
                right_segment_id,
                reason,
            )

    def _score_boundary_evidence(
        self,
        segments: Sequence[SubtitleSegment],
    ) -> tuple[list[BoundaryEvidence | None], str | None]:
        boundary_count = max(0, len(segments) - 1)
        if not boundary_count:
            return [], None

        texts = [segment.text for segment in segments]
        windowed = getattr(self.boundary_api, "windowed_score_boundaries", None)
        if callable(windowed):
            response = windowed(texts)
        else:
            score_boundaries = getattr(self.boundary_api, "score_boundaries", None)
            if not callable(score_boundaries):
                return (
                    [None] * boundary_count,
                    "boundary scoring is unavailable",
                )
            response = score_boundaries(texts)

        try:
            evidence = _normalise_boundary_evidence(response, segments)
        except Exception as exc:
            return (
                [None] * boundary_count,
                f"invalid or unavailable boundary evidence ({type(exc).__name__}: {exc})",
            )
        return evidence, None

    def evaluate_boundaries(
        self,
        segments: Sequence[SubtitleSegment],
    ) -> list[BoundaryDecision]:
        """Return one hard/soft policy decision for every adjacent cue pair."""
        ordered = _validate_segments(segments)
        evidence, evidence_error = self._score_boundary_evidence(ordered)
        decisions: list[BoundaryDecision] = []

        for boundary_index, (left, right) in enumerate(zip(ordered, ordered[1:])):
            current_evidence = evidence[boundary_index]
            model_probability = (
                current_evidence["modelProbability"] if current_evidence is not None else None
            )
            gap_ms = right.start_ms - left.end_ms
            hard_reason = _boundary_hard_break_reason(
                left,
                right,
                gap_ms,
                self.policy_config,
            )
            if hard_reason is not None:
                decision: BoundaryDecisionValue = "break"
                reason = hard_reason
            elif (
                current_evidence is None
                and evidence_error
                and "changes cue order" in evidence_error
            ):
                decision = "break"
                reason = f"cue order is invalid: {evidence_error}"
            elif current_evidence is None:
                decision = "uncertain"
                reason = evidence_error or "boundary model evidence is unavailable"
            elif model_probability is None:
                decision = "uncertain"
                reason = "boundary model probability is unavailable"
            else:
                decision, reason = _soft_boundary_decision(
                    left,
                    right,
                    gap_ms,
                    model_probability,
                    self.policy_config,
                )

            if decision != "join" and self.debug:
                self._log_boundary_issue(left.segment_id, right.segment_id, reason)
            decisions.append(
                {
                    "leftSegmentId": left.segment_id,
                    "rightSegmentId": right.segment_id,
                    "decision": decision,
                    "reason": reason,
                    "modelProbability": model_probability,
                    "gapMs": gap_ms,
                }
            )
        return decisions

    def build_sentences(
        self,
        segments: Sequence[SubtitleSegment],
        decisions: Sequence[BoundaryDecision],
    ) -> list[ReconstructedSentence]:
        """Build sentence groups using only JOIN decisions."""
        ordered = _validate_segments(segments)
        expected_count = max(0, len(ordered) - 1)
        if len(decisions) != expected_count:
            raise ValueError(
                f"expected {expected_count} boundary decisions, received {len(decisions)}"
            )

        groups: list[list[str]] = [[ordered[0].segment_id]]
        for boundary_index, decision in enumerate(decisions):
            if not isinstance(decision, Mapping):
                raise ValueError("boundary decisions must be objects")
            left = ordered[boundary_index]
            right = ordered[boundary_index + 1]
            if (
                decision.get("leftSegmentId") != left.segment_id
                or decision.get("rightSegmentId") != right.segment_id
            ):
                raise ValueError("boundary decisions must retain cue order and IDs")
            decision_value = decision.get("decision")
            if decision_value not in {"join", "break", "uncertain"}:
                raise ValueError("boundary decision must be join, break, or uncertain")

            # UNCERTAIN is deliberately handled like BREAK.
            if decision_value == "join":
                groups[-1].append(right.segment_id)
            else:
                groups.append([right.segment_id])

        by_id = {segment.segment_id: segment for segment in ordered}
        return [
            ReconstructedSentence(
                text=join_segments([by_id[segment_id].text for segment_id in group]),
                start_ms=by_id[group[0]].start_ms,
                end_ms=by_id[group[-1]].end_ms,
                parts=_parts_for_group(group, by_id),
            )
            for group in groups
        ]

    def reconstruct(self, segments: Sequence[SubtitleSegment]) -> list[ReconstructedSentence]:
        ordered = _validate_segments(segments)
        return self.build_sentences(ordered, self.evaluate_boundaries(ordered))

    def reconstruct_timeline(self, segments: Sequence[SubtitleSegment]) -> "SubtitleTimeline":
        ordered = _validate_segments(segments)
        return SubtitleTimeline(segments=ordered, sentences=self.reconstruct(ordered))




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
