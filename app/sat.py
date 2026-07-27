from __future__ import annotations

import math
import re
import threading
from typing import Any, Protocol, Sequence, TypedDict

SAT_MODEL_NAME = "sat-3l-sm"
_MAX_SEGMENT_LENGTH = 5000
_NO_SPACE_BEFORE = frozenset(",.!?;:%)]}»”")
_NO_SPACE_AFTER = frozenset("([{«“")
_CLOSING_TAG_RE = re.compile(r"^</[^>]+>")
_LEADING_CONTRACTION_RE = re.compile(
    r"^(?P<apostrophe>['’])(?P<suffix>m|re|ve|ll|d|s|t)\b",
    re.IGNORECASE,
)
_LAST_WORD_RE = re.compile(r"(?P<word>[A-Za-z]+)$")
_INVALID_CONTRACTION_BASES = {
    "ve": frozenset({"her", "him", "them", "us", "me", "my", "your", "our", "their", "his", "its"}),
    "re": frozenset({"her", "him", "us", "me", "my", "your", "our", "their", "his", "its", "he", "she", "it"}),
    "ll": frozenset({"her", "him", "them", "us", "me", "my", "your", "our", "their", "his", "its"}),
    "d": frozenset({"her", "him", "them", "us", "me", "my", "your", "our", "their", "his", "its"}),
}


class SaTUnavailableError(RuntimeError):
    """Raised when the SaT model cannot be loaded or queried."""


class BoundaryEvidence(TypedDict):
    """Model evidence for one original subtitle cue boundary."""

    leftIndex: int
    rightIndex: int
    characterOffset: int
    boundaryProbability: float


class SaTModel(Protocol):
    def split(self, text: str) -> Sequence[str]:
        ...


class SaTProbabilityModel(Protocol):
    def predict_proba(self, text: str) -> Any:
        ...


def _contraction_attachment_issue(left: str, right: str) -> str | None:
    fragment = right.lstrip()
    contraction = _LEADING_CONTRACTION_RE.match(fragment)
    if contraction is None:
        return None

    preceding_word = _LAST_WORD_RE.search(left.rstrip())
    if preceding_word is None:
        return "apostrophe-leading contraction has no preceding word"

    base = preceding_word["word"].lower()
    suffix = contraction["suffix"].lower()
    if suffix == "m" and base != "i":
        return f"{contraction.group(0)!r} can only attach safely to 'I'"
    if suffix == "t" and base not in {
        "aren",
        "can",
        "couldn",
        "daren",
        "didn",
        "doesn",
        "don",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "mightn",
        "mustn",
        "needn",
        "shan",
        "shouldn",
        "wasn",
        "weren",
        "won",
        "wouldn",
    }:
        return f"{contraction.group(0)!r} cannot safely attach to {preceding_word.group(0)!r}"
    if base in _INVALID_CONTRACTION_BASES.get(suffix, ()):
        return f"{contraction.group(0)!r} cannot safely attach to {preceding_word.group(0)!r}"
    return None


def text_join_issue(left: str, right: str) -> str | None:
    """Return why joining two source fragments could corrupt their boundary."""
    if not isinstance(left, str) or not isinstance(right, str):
        return "subtitle fragments must be text"
    if not left.strip() or not right.strip():
        return "subtitle fragments must contain non-whitespace text"
    return _contraction_attachment_issue(left, right)


def _join_segments_with_boundary_offsets(
    segments: Sequence[str],
) -> tuple[str, list[int]]:
    """Join fragments and retain SaT's raw character boundary indexes."""
    joined = ""
    boundary_offsets: list[int] = []
    for segment in segments:
        if not isinstance(segment, str):
            raise ValueError("each segment must be text")
        part = segment.strip()
        if not part:
            continue

        contraction = _LEADING_CONTRACTION_RE.match(part)
        contraction_is_safe = (
            contraction is not None and _contraction_attachment_issue(joined, part) is None
        )
        if not joined:
            joined = part
        else:
            # wtpsplit assigns each character probability to the character
            # ending a possible boundary, so this is the raw index before
            # the next fragment starts in the joined model input.
            boundary_offsets.append(len(joined) - 1)
            if (
                joined[-1].isspace()
                or part[0] in _NO_SPACE_BEFORE
                or _CLOSING_TAG_RE.match(part)
                or joined[-1] in _NO_SPACE_AFTER
                or contraction_is_safe
            ):
                joined += part
            else:
                joined += f" {part}"
    return joined, boundary_offsets


def join_segments(segments: Sequence[str]) -> str:
    """Create a lossless display join while retaining every non-space character."""
    return _join_segments_with_boundary_offsets(segments)[0]


def _validate_segments(segments: Sequence[str]) -> None:
    if not segments:
        raise ValueError("at least one segment is required")
    if any(not isinstance(segment, str) for segment in segments):
        raise ValueError("each segment must be text")
    if any(not segment.strip() for segment in segments):
        raise ValueError("segments must contain non-empty text")
    if any(len(segment.strip()) > _MAX_SEGMENT_LENGTH for segment in segments):
        raise ValueError(f"each segment must be at most {_MAX_SEGMENT_LENGTH} characters")


def _normalise_boundary_probabilities(
    raw_probabilities: Any,
    expected_count: int,
) -> list[float]:
    """Validate and flatten wtpsplit's per-character probability array."""
    if isinstance(raw_probabilities, (str, bytes)):
        raise ValueError("probabilities must be a numeric sequence")

    tolist = getattr(raw_probabilities, "tolist", None)
    try:
        values = tolist() if callable(tolist) else list(raw_probabilities)
    except Exception as exc:
        raise ValueError("probabilities must be a numeric sequence") from exc

    if not isinstance(values, (list, tuple)):
        raise ValueError("probabilities must be a numeric sequence")
    if len(values) != expected_count:
        raise ValueError(
            f"expected {expected_count} character probabilities, received {len(values)}"
        )

    probabilities: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, (str, bytes)):
            raise ValueError(f"probability at index {index} is not numeric")
        try:
            row = list(value)
        except TypeError:
            scalar = value
        else:
            if len(row) != 1:
                raise ValueError(
                    f"probability at index {index} must contain exactly one value"
                )
            scalar = row[0]
        if isinstance(scalar, (str, bytes, bool)):
            raise ValueError(f"probability at index {index} is not numeric")
        try:
            probability = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"probability at index {index} is not numeric") from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"probability at index {index} must be within [0, 1]"
            )
        probabilities.append(probability)
    return probabilities


def _validate_boundary_offsets(
    boundary_offsets: Sequence[int],
    segment_count: int,
    joined_text_length: int,
) -> None:
    expected_count = max(0, segment_count - 1)
    if len(boundary_offsets) != expected_count:
        raise ValueError(
            f"expected {expected_count} boundary offsets, received {len(boundary_offsets)}"
        )
    if list(boundary_offsets) != sorted(set(boundary_offsets)):
        raise ValueError("boundary offsets must be strictly increasing")
    if any(offset < 0 or offset >= joined_text_length for offset in boundary_offsets):
        raise ValueError("boundary offsets must point inside the joined text")


class SaTSentenceReconstructor:
    """Reconstruct ordered fragments and segment them with Segment Any Text."""

    def __init__(
        self,
        model: SaTModel | SaTProbabilityModel | None = None,
        model_name: str = SAT_MODEL_NAME,
    ) -> None:
        self.model_name = model_name
        self._model = model
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def score_boundaries(self, segments: Sequence[str]) -> list[BoundaryEvidence]:
        """Return SaT's raw probability for every original cue boundary.

        ``characterOffset`` is the zero-based index consumed by wtpsplit's
        character probability vector. It identifies the last character of the
        left segment; the boundary is immediately after that character.
        """
        _validate_segments(segments)
        if len(segments) < 2:
            return []

        joined_text, boundary_offsets = _join_segments_with_boundary_offsets(segments)
        _validate_boundary_offsets(
            boundary_offsets,
            segment_count=len(segments),
            joined_text_length=len(joined_text),
        )

        model = self._get_model()
        predict_proba = getattr(model, "predict_proba", None)
        if not callable(predict_proba):
            raise SaTUnavailableError(
                "SaT raw boundary probability scoring is unavailable"
            )

        try:
            raw_probabilities = predict_proba(joined_text)
        except SaTUnavailableError:
            raise
        except Exception as exc:
            raise SaTUnavailableError(
                "Unable to score subtitle boundaries with SaT"
            ) from exc

        try:
            probabilities = _normalise_boundary_probabilities(
                raw_probabilities,
                expected_count=len(joined_text),
            )
        except Exception as exc:
            raise SaTUnavailableError(
                f"Invalid SaT boundary probabilities: {exc}"
            ) from exc

        return [
            {
                "leftIndex": boundary_index,
                "rightIndex": boundary_index + 1,
                "characterOffset": character_offset,
                "boundaryProbability": probabilities[character_offset],
            }
            for boundary_index, character_offset in enumerate(boundary_offsets)
        ]

    def _get_model(self) -> SaTModel | SaTProbabilityModel:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is None:
                try:
                    from wtpsplit import SaT

                    self._model = SaT(self.model_name)
                except Exception as exc:
                    raise SaTUnavailableError(
                        f"Unable to load SaT model '{self.model_name}'"
                    ) from exc
        return self._model

    def group(self, segments: Sequence[dict[str, str]]) -> dict[str, Any]:
        """Group ordered segment payloads while retaining their stable IDs."""
        if not segments:
            raise ValueError("at least one segment is required")
        ids: list[str] = []
        texts: list[str] = []
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("each segment must be an object")
            segment_id = segment.get("segmentId")
            text = segment.get("text")
            if not isinstance(segment_id, str) or not segment_id:
                raise ValueError("each segment must have a stable segmentId")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("each segment must contain non-empty text")
            ids.append(segment_id)
            texts.append(text)
        if len(ids) != len(set(ids)):
            raise ValueError("segment IDs must be unique")

        from .subtitles import (
            SubtitleSegment,
            _groups_from_sentence_texts,
            validate_grouping_response,
        )

        source_segments = [
            SubtitleSegment(
                segment_id=segment_id,
                text=text,
                start_ms=0,
                end_ms=0,
            )
            for segment_id, text in zip(ids, texts)
        ]
        model = self._get_model()
        model_group = getattr(model, "group", None)
        if callable(model_group):
            groups = validate_grouping_response(model_group(segments), source_segments)
        else:
            reconstruction = self.reconstruct(texts)
            groups = _groups_from_sentence_texts(
                reconstruction["sentences"],
                source_segments,
            )

        index_by_id = {segment_id: index for index, segment_id in enumerate(ids)}
        return {
            "model": self.model_name,
            "groups": [
                {
                    "segmentIndexes": [index_by_id[segment_id] for segment_id in group],
                    "segmentIds": group,
                    "text": join_segments(
                        [texts[index_by_id[segment_id]] for segment_id in group]
                    ),
                }
                for group in groups
            ],
        }


    def reconstruct(self, segments: Sequence[str]) -> dict[str, Any]:
        _validate_segments(segments)

        joined_text = join_segments(segments)

        try:
            split_result = self._get_model().split(joined_text)
            sentences = [sentence.strip() for sentence in split_result if sentence.strip()]
        except SaTUnavailableError:
            raise
        except Exception as exc:
            raise SaTUnavailableError("Unable to segment text with SaT") from exc

        if not sentences:
            raise SaTUnavailableError("SaT returned no sentences")

        return {
            "model": self.model_name,
            "segments": list(segments),
            "joined_text": joined_text,
            "sentences": sentences,
            "full_sentence": sentences[0] if len(sentences) == 1 else None,
            "is_single_sentence": len(sentences) == 1,
        }
