from __future__ import annotations

import re
import threading
from typing import Any, Protocol, Sequence

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.80
_ACCURACY_MODEL_NAME = "all-mpnet-base-v2"
_ACCURACY_COSINE_CUTOFF = 0.630
_ACCURACY_ENCODE_COSINE_CUTOFF = 0.355
_TARGET_TOKEN_WEIGHT = 0.5
_CONTEXT_WINDOW = 256

# A target must be one lexical word, with optional internal apostrophes or hyphens.
WORD_PATTERN = re.compile(r"^[^\W\d_]+(?:['’\-][^\W\d_]+)*$", re.UNICODE)


class ModelUnavailableError(RuntimeError):
    """Raised when the embedding model cannot be loaded or queried."""


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any:
        ...


_LEMMATIZER: Any | None = None
_LEMMATIZER_LOAD_LOCK = threading.Lock()
_LEMMATIZER_CALL_LOCK = threading.Lock()


def _get_lemmatizer() -> Any:
    global _LEMMATIZER

    if _LEMMATIZER is None:
        with _LEMMATIZER_LOAD_LOCK:
            if _LEMMATIZER is None:
                try:
                    import spacy

                    nlp = spacy.blank("en")
                    nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
                    nlp.initialize()
                    _LEMMATIZER = nlp
                except Exception as exc:
                    raise ModelUnavailableError("Unable to initialize English lemmatizer") from exc

    return _LEMMATIZER


_SURFACE_TOKEN_PATTERN = re.compile(
    r"[^\W\d_]+(?:['’\-][^\W\d_]+)*",
    re.UNICODE,
)


def _lemmatize_surface(surface: str) -> str:
    normalized_surface = surface.casefold()
    with _LEMMATIZER_CALL_LOCK:
        doc = _get_lemmatizer()(normalized_surface)
        tokens = [token for token in doc if not token.is_space]
        if len(tokens) == 1:
            return (tokens[0].lemma_ or tokens[0].text).casefold()
    return normalized_surface


def _lemma_for(text: str) -> str:
    if not WORD_PATTERN.fullmatch(text):
        return text.casefold()
    return _lemmatize_surface(text)


def find_word_occurrences(word: str, sentence: str) -> list[dict[str, int]]:
    """Return offsets for sentence tokens whose lemma matches the target lemma."""
    target_lemma = _lemma_for(word)
    return [
        {"start": match.start(), "end": match.end()}
        for match in _SURFACE_TOKEN_PATTERN.finditer(sentence)
        if _lemmatize_surface(match.group()) == target_lemma
    ]


def _target_sentence_context(word: str, sentence: str) -> str:
    """Build a bounded, target-centered sentence context."""
    occurrences = find_word_occurrences(word, sentence)
    if not occurrences:
        raise ValueError("word must occur in the sentence")

    start = occurrences[0]["start"]
    end = occurrences[0]["end"]
    context_start = max(0, start - _CONTEXT_WINDOW)
    context_end = min(len(sentence), end + _CONTEXT_WINDOW)
    context = sentence[context_start:context_end]

    if context_start:
        context = f"... {context}"
    if context_end < len(sentence):
        context = f"{context} ..."
    return context


def _target_frame(word: str, sentence: str) -> tuple[str | None, str | None]:
    """Return lexical neighbors immediately around the first target."""
    occurrence = find_word_occurrences(word, sentence)[0]
    matches = list(_SURFACE_TOKEN_PATTERN.finditer(sentence))
    index = next(
        index
        for index, match in enumerate(matches)
        if match.start() == occurrence["start"] and match.end() == occurrence["end"]
    )
    previous = matches[index - 1].group().casefold() if index else None
    following = matches[index + 1].group().casefold() if index + 1 < len(matches) else None
    return previous, following


def _target_frame_bonus(word: str, sentence1: str, sentence2: str) -> float:
    """Reward a shared noun-complement frame that clarifies a target sense."""
    _, following1 = _target_frame(word, sentence1)
    _, following2 = _target_frame(word, sentence2)
    return 0.25 if following1 == following2 == "of" else 0.0


def _target_context(word: str, sentence: str) -> str:
    """Build a bounded context with target occurrences masked.

    Masking the target keeps injected or fallback encoders from scoring the
    identical lexical token more heavily than the surrounding sense-bearing
    context.
    """
    occurrences = find_word_occurrences(word, sentence)
    if not occurrences:
        raise ValueError("word must occur in the sentence")

    start = occurrences[0]["start"]
    end = occurrences[0]["end"]
    context_start = max(0, start - _CONTEXT_WINDOW)
    context_end = min(len(sentence), end + _CONTEXT_WINDOW)

    pieces: list[str] = []
    cursor = context_start
    for occurrence in occurrences:
        occurrence_start = occurrence["start"]
        occurrence_end = occurrence["end"]
        if occurrence_start < context_start or occurrence_end > context_end:
            continue
        pieces.append(sentence[cursor:occurrence_start])
        pieces.append("[TARGET]")
        cursor = occurrence_end
    pieces.append(sentence[cursor:context_end])
    context = "".join(pieces)

    if context_start:
        context = f"... {context}"
    if context_end < len(sentence):
        context = f"{context} ..."

    return f'Target word: "{word}". Sentence context (target masked): {context}'


def _cosine_similarity(vectors: np.ndarray) -> float:
    if (
        vectors.ndim != 2
        or vectors.shape[0] != 2
        or vectors.shape[1] == 0
        or not np.all(np.isfinite(vectors))
    ):
        raise ModelUnavailableError("Embedding model returned an invalid result")

    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ModelUnavailableError("Embedding model returned a zero or non-finite vector")

    similarity = float(np.dot(vectors[0], vectors[1]) / (norms[0] * norms[1]))
    return max(-1.0, min(1.0, similarity))


def _calibrate_accuracy_similarity(
    cosine: float,
    *,
    cutoff: float = _ACCURACY_COSINE_CUTOFF,
) -> float:
    """Map an encoder's cosine boundary onto the public cutoff."""
    if cosine <= cutoff:
        similarity = -1.0 + (cosine + 1.0) * (
            (DEFAULT_THRESHOLD + 1.0) / (cutoff + 1.0)
        )
    else:
        similarity = DEFAULT_THRESHOLD + (cosine - cutoff) * (
            (1.0 - DEFAULT_THRESHOLD) / (1.0 - cutoff)
        )
    return max(-1.0, min(1.0, similarity))


def _contextual_embeddings(
    model: Any,
    contexts: Sequence[str],
    word: str,
) -> np.ndarray | None:
    """Pool contextual tokens while retaining a target-aware signal."""
    try:
        tokenizer = model.tokenizer
        transformer = model[0].auto_model
    except (AttributeError, IndexError, TypeError):
        return None

    try:
        import torch

        encoded = tokenizer(
            list(contexts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        offset_mapping = encoded.pop("offset_mapping")
        special_tokens_mask = encoded.pop("special_tokens_mask")
        device = next(transformer.parameters()).device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        offset_mapping = offset_mapping.to(device)
        special_tokens_mask = special_tokens_mask.to(device)

        transformer.eval()
        with torch.no_grad():
            output = transformer(**encoded, output_hidden_states=True)
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) < 2:
            raise ModelUnavailableError("Embedding model did not return contextual states")

        hidden = hidden_states[-2]
        token_mask = encoded["attention_mask"].bool() & ~special_tokens_mask.bool()
        target_mask = torch.zeros_like(token_mask)
        for row, context in enumerate(contexts):
            occurrence = find_word_occurrences(word, context)[0]
            target_mask[row] = (
                (offset_mapping[row, :, 0] < occurrence["end"])
                & (offset_mapping[row, :, 1] > occurrence["start"])
                & token_mask[row]
            )

        target_counts = target_mask.sum(dim=1, keepdim=True)
        if torch.any(target_counts == 0):
            raise ModelUnavailableError("Embedding model did not retain the target token")
        target_vectors = (hidden * target_mask.unsqueeze(-1)).sum(dim=1) / target_counts

        context_mask = token_mask & ~target_mask
        context_counts = context_mask.sum(dim=1, keepdim=True)
        context_vectors = (
            (hidden * context_mask.unsqueeze(-1)).sum(dim=1)
            / context_counts.clamp_min(1)
        )
        context_vectors = torch.where(
            context_counts > 0,
            context_vectors,
            target_vectors,
        )
        vectors = context_vectors + _TARGET_TOKEN_WEIGHT * target_vectors
        return np.asarray(vectors.detach().cpu(), dtype=np.float32)
    except ModelUnavailableError:
        raise
    except Exception as exc:
        raise ModelUnavailableError("Unable to compute contextual embeddings") from exc


class MiniLMSimilarity:
    """Compare target-word meanings with a calibrated contextual encoder.

    The public model identifier remains stable for API compatibility. The
    default service prefers a higher-capacity encoder and falls back to the
    public model when that optional backend is unavailable. Injected models
    retain the direct cosine behavior used by callers and tests.
    """

    def __init__(
        self,
        model: EmbeddingModel | None = None,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.model_name = model_name
        self._model = model
        self._prefer_accuracy_model = model is None and model_name == MODEL_NAME
        self._using_accuracy_model = self._prefer_accuracy_model
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _get_model(self) -> EmbeddingModel:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is None:
                preferred_name = (
                    _ACCURACY_MODEL_NAME if self._prefer_accuracy_model else self.model_name
                )
                try:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(preferred_name)
                except Exception as preferred_exc:
                    if not self._prefer_accuracy_model:
                        raise ModelUnavailableError(
                            f"Unable to load embedding model '{self.model_name}'"
                        ) from preferred_exc
                    try:
                        from sentence_transformers import SentenceTransformer

                        self._model = SentenceTransformer(self.model_name)
                    except Exception as fallback_exc:
                        raise ModelUnavailableError(
                            f"Unable to load embedding model '{self.model_name}'"
                        ) from fallback_exc
                    self._using_accuracy_model = False

        return self._model

    def _score_contexts(
        self,
        model: EmbeddingModel,
        contexts: Sequence[str],
        *,
        calibrated: bool,
        calibration_cutoff: float = _ACCURACY_COSINE_CUTOFF,
    ) -> float:
        try:
            embeddings = model.encode(
                contexts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            vectors = np.asarray(embeddings, dtype=np.float32)
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError("Unable to compute sentence embeddings") from exc

        similarity = _cosine_similarity(vectors)
        return (
            _calibrate_accuracy_similarity(similarity, cutoff=calibration_cutoff)
            if calibrated
            else similarity
        )

    def _score_accuracy_contexts(
        self,
        model: EmbeddingModel,
        contexts: Sequence[str],
        word: str,
        *,
        frame_bonus: float = 0.0,
    ) -> float:
        vectors = _contextual_embeddings(model, contexts, word)
        if vectors is None:
            return self._score_contexts(
                model,
                contexts,
                calibrated=True,
                calibration_cutoff=_ACCURACY_ENCODE_COSINE_CUTOFF,
            )
        similarity = min(1.0, _cosine_similarity(vectors) + frame_bonus)
        return _calibrate_accuracy_similarity(similarity)

    def compare(self, word: str, sentence1: str, sentence2: str) -> float:
        """Return calibrated similarity for the target word's two contexts."""
        model = self._get_model()
        if self._using_accuracy_model:
            contexts = [
                _target_sentence_context(word, sentence1),
                _target_sentence_context(word, sentence2),
            ]
            return self._score_accuracy_contexts(
                model,
                contexts,
                word,
                frame_bonus=_target_frame_bonus(word, sentence1, sentence2),
            )

        contexts = [
            _target_context(word, sentence1),
            _target_context(word, sentence2),
        ]
        return self._score_contexts(model, contexts, calibrated=False)
