from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import Any, NotRequired, Protocol, Sequence, TypedDict

from .language_profile import BoundaryLanguageProfile



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
    boundaryProbability: float | None
    status: NotRequired[str]  # "unavailable" when model could not score this boundary


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
    profile: BoundaryLanguageProfile | None = None,
) -> tuple[str, list[int]]:
    """Join fragments and retain SaT's raw character boundary indexes.

    When *profile* is provided, delegates to its ``join_segments`` method
    so that boundary offsets use language‑appropriate joining rules.
    Otherwise uses the default English joining behaviour (backward compat).
    """
    if profile is not None:
        return profile.join_segments(segments)

    # Default English joining behaviour (backward compatible).
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



@dataclass(frozen=True, slots=True)
class WindowedScoringConfig:
    """Configuration for contextual windowing in boundary scoring.

    ``window_size`` controls how many consecutive cues form one model call.
    ``min_context`` controls how many extra cues on each side frame each
    owned boundary.  The constraint ``window_size >= 2 * min_context + 2``
    guarantees at least one owned boundary per interior window.
    """

    window_size: int = 48
    min_context: int = 6

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_size, bool)
            or not isinstance(self.window_size, int)
            or self.window_size < 1
        ):
            raise ValueError("window_size must be a positive integer")
        if (
            isinstance(self.min_context, bool)
            or not isinstance(self.min_context, int)
            or self.min_context < 1
        ):
            raise ValueError("min_context must be a positive integer")
        if self.window_size < 2 * self.min_context + 2:
            raise ValueError(
                f"window_size ({self.window_size}) must be >= "
                f"2 * min_context + 2 ({2 * self.min_context + 2})"
            )


_WindowPlan = dict[str, int]  # windowStart, windowEnd, ownedStart, ownedEnd


def _build_window_plan(
    segment_count: int,
    config: WindowedScoringConfig,
) -> list[_WindowPlan]:
    """Build a deterministic window plan before calling the model.

    Each plan item records the segment range (windowStart, windowEnd) and
    the boundary index range (ownedStart, ownedEnd) that the window owns.

    The first window owns from boundary 0.  The last window owns through
    boundary N - 2.  Interior windows own only their central boundaries,
    leaving the overlap region to adjacent windows.  Every boundary index
    from 0 through N - 2 has exactly one owner.
    """
    n = segment_count
    num_boundaries = n - 1

    if n <= config.window_size:
        return [
            {
                "windowStart": 0,
                "windowEnd": n,
                "ownedStart": 0,
                "ownedEnd": num_boundaries - 1,
            }
        ]

    hop = config.window_size - 2 * config.min_context
    plan: list[_WindowPlan] = []
    start = 0

    while start < n:
        window_end = min(n, start + config.window_size)

        # First window: own from boundary 0.
        # Interior windows: own central region.
        # Last window: own through the final boundary.
        owned_start = start + config.min_context if start > 0 else 0
        owned_end = min(
            num_boundaries - 1,
            start + config.window_size - config.min_context - 1,
        )
        if window_end >= n:
            owned_end = num_boundaries - 1

        plan.append(
            {
                "windowStart": start,
                "windowEnd": window_end,
                "ownedStart": owned_start,
                "ownedEnd": owned_end,
            }
        )

        if window_end >= n:
            break
        start += hop

    return plan


def _validate_window_plan(plan: list[_WindowPlan], num_boundaries: int) -> None:
    """Validate coverage and ordering invariants of a window plan."""
    assert plan, "window plan must not be empty"
    assert plan[0]["ownedStart"] == 0, "first owned boundary must be 0"
    assert plan[-1]["ownedEnd"] == num_boundaries - 1, "final owned boundary must be N-2"

    for i in range(len(plan)):
        p = plan[i]
        assert 0 <= p["ownedStart"] <= p["ownedEnd"] < num_boundaries

    for i in range(len(plan) - 1):
        assert plan[i]["windowStart"] < plan[i + 1]["windowStart"], (
            "windows must be in source order"
        )
        assert plan[i]["ownedEnd"] < plan[i + 1]["ownedStart"], (
            "owned ranges must not overlap"
        )
        assert plan[i]["ownedEnd"] + 1 == plan[i + 1]["ownedStart"], (
            "owned ranges must have no gaps"
        )

    owned_count = sum(
        p["ownedEnd"] - p["ownedStart"] + 1 for p in plan
    )
    assert owned_count == num_boundaries, (
        f"expected {num_boundaries} owned boundaries, plan covers {owned_count}"
    )

class SaTSentenceReconstructor:
    """Reconstruct ordered fragments and segment them with Segment Any Text."""

    def __init__(
        self,
        model: SaTModel | SaTProbabilityModel | None = None,
        model_name: str = SAT_MODEL_NAME,
        profile: BoundaryLanguageProfile | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model
        self._load_lock = threading.Lock()
        self.profile = profile

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

        joined_text, boundary_offsets = _join_segments_with_boundary_offsets(segments, profile=self.profile)
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

    def windowed_score_boundaries(
        self,
        segments: Sequence[str],
        *,
        config: WindowedScoringConfig | None = None,
    ) -> list[BoundaryEvidence]:
        """Score boundaries using overlapping contextual windows.

        Uses a deterministic window plan created before any model call.
        Each window is scored independently.  A failure in one window
        marks only its owned boundaries as unavailable and does not
        affect other windows.

        Unavailable evidence is reported with ``boundaryProbability=None``
        and ``status="unavailable"``.  The subtitle policy treats
        unavailable evidence as UNCERTAIN, which is safe (BREAK).
        """
        _validate_segments(segments)
        if len(segments) < 2:
            return []

        cfg = config or WindowedScoringConfig()
        n = len(segments)
        num_boundaries = n - 1

        if n <= cfg.window_size:
            # Single window covering all segments
            try:
                return self.score_boundaries(segments)
            except Exception:
                return [
                    {
                        "leftIndex": i,
                        "rightIndex": i + 1,
                        "characterOffset": 0,
                        "boundaryProbability": None,
                        "status": "unavailable",
                    }
                    for i in range(num_boundaries)
                ]

        plan = _build_window_plan(n, cfg)
        _validate_window_plan(plan, num_boundaries)

        results: list[BoundaryEvidence | None] = [None] * num_boundaries

        for wp in plan:
            window_segments = segments[wp["windowStart"] : wp["windowEnd"]]
            try:
                window_evidence = self.score_boundaries(window_segments)
            except Exception:
                # Mark only this window's owned boundaries as unavailable.
                for b in range(wp["ownedStart"], wp["ownedEnd"] + 1):
                    results[b] = {
                        "leftIndex": b,
                        "rightIndex": b + 1,
                        "characterOffset": 0,
                        "boundaryProbability": None,
                        "status": "unavailable",
                    }
                continue

            for offset, evidence in enumerate(window_evidence):
                boundary_idx = wp["windowStart"] + offset
                if wp["ownedStart"] <= boundary_idx <= wp["ownedEnd"]:
                    results[boundary_idx] = {
                        "leftIndex": boundary_idx,
                        "rightIndex": boundary_idx + 1,
                        "characterOffset": evidence["characterOffset"],
                        "boundaryProbability": evidence["boundaryProbability"],
                    }

        # The plan guarantee every boundary has been assigned.
        assert all(r is not None for r in results), (
            "window plan did not cover every boundary"
        )
        return results  # type: ignore[return-value]

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
