from __future__ import annotations

import threading
from typing import Any, Protocol, Sequence

SAT_MODEL_NAME = "sat-3l-sm"
_MAX_SEGMENT_LENGTH = 5000
_NO_SPACE_BEFORE = frozenset(",.!?;:%)]}»”'’")
_NO_SPACE_AFTER = frozenset("([{«“")


class SaTUnavailableError(RuntimeError):
    """Raised when the SaT model cannot be loaded or queried."""


class SaTModel(Protocol):
    def split(self, text: str) -> Sequence[str]:
        ...


def join_segments(segments: Sequence[str]) -> str:
    """Join text fragments without adding spaces before punctuation."""
    joined = ""
    for segment in segments:
        part = segment.strip()
        if not part:
            continue
        if not joined:
            joined = part
        elif (
            joined[-1].isspace()
            or part[0] in _NO_SPACE_BEFORE
            or joined[-1] in _NO_SPACE_AFTER
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
