from __future__ import annotations

import re
import threading
from typing import Any, Protocol, Sequence

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


class SaTModel(Protocol):
    def split(self, text: str) -> Sequence[str]:
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


def join_segments(segments: Sequence[str]) -> str:
    """Create a lossless display join while retaining every non-space character."""
    joined = ""
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
        elif (
            joined[-1].isspace()
            or part[0] in _NO_SPACE_BEFORE
            or _CLOSING_TAG_RE.match(part)
            or joined[-1] in _NO_SPACE_AFTER
            or contraction_is_safe
        ):
            joined += part
        else:
            joined += f" {part}"
    return joined


class SaTSentenceReconstructor:
    """Reconstruct ordered fragments and segment them with Segment Any Text."""

    def __init__(
        self,
        model: SaTModel | None = None,
        model_name: str = SAT_MODEL_NAME,
    ) -> None:
        self.model_name = model_name
        self._model = model
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _get_model(self) -> SaTModel:
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
        if not segments:
            raise ValueError("at least one segment is required")
        if any(not isinstance(segment, str) for segment in segments):
            raise ValueError("each segment must be text")
        if any(not segment.strip() for segment in segments):
            raise ValueError("segments must contain non-empty text")
        if any(len(segment.strip()) > _MAX_SEGMENT_LENGTH for segment in segments):
            raise ValueError(f"each segment must be at most {_MAX_SEGMENT_LENGTH} characters")

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
