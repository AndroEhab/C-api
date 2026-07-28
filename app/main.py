from __future__ import annotations

from typing import Literal

from fastapi import Depends, FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .model import (
    DEFAULT_THRESHOLD,
    WORD_PATTERN,
    MiniLMSimilarity,
    ModelUnavailableError,
    find_word_occurrences,
)

from .sat import SaTSentenceReconstructor, SaTUnavailableError
from .subtitles import (
    SubtitleSegment,
    SubtitleSentenceReconstructor,
)


class SentenceFromSegmentsRequest(BaseModel):
    """Input containing one or more ordered text fragments to reconstruct."""

    model_config = ConfigDict(extra="forbid")

    segments: list[str] = Field(
        ...,
        min_length=1,
        description="One or more ordered text fragments.",
        examples=[["The quick brown", "fox jumps", "over the lazy dog."]],
    )

    @field_validator("segments", mode="before")
    @classmethod
    def strip_segments(cls, value: object) -> object:
        if isinstance(value, list):
            return [segment.strip() if isinstance(segment, str) else segment for segment in value]
        return value

    @field_validator("segments")
    @classmethod
    def require_non_empty_segments(cls, value: list[str]) -> list[str]:
        if any(not segment for segment in value):
            raise ValueError("each segment must contain non-whitespace text")
        if any(len(segment) > 5000 for segment in value):
            raise ValueError("each segment must be at most 5000 characters")
        return value


class SentenceFromSegmentsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    segments: list[str]
    joined_text: str
    sentences: list[str] = Field(..., min_length=1)
    full_sentence: str | None
    is_single_sentence: bool



class SubtitleSegmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    segment_id: str = Field(..., alias="segmentId", min_length=1)
    text: str = Field(..., min_length=1)
    start_ms: int = Field(..., alias="startMs", ge=0)
    end_ms: int = Field(..., alias="endMs", ge=0)
    speaker: str | None = Field(default=None)
    raw_text: str | None = Field(default=None, alias="rawText")
    lines: list[str] = Field(default_factory=list)
    speaker_markers: list[str] = Field(default_factory=list, alias="speakerMarkers")
    contains_multiple_speakers: bool = Field(default=False, alias="containsMultipleSpeakers")

    @field_validator("text")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace text")
        return value

    @model_validator(mode="after")
    def require_valid_timing(self) -> "SubtitleSegmentRequest":
        if self.end_ms < self.start_ms:
            raise ValueError("endMs must be greater than or equal to startMs")
        return self

    def to_domain(self) -> SubtitleSegment:
        return SubtitleSegment(
            segment_id=self.segment_id,
            text=self.text,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            speaker=self.speaker,
            raw_text=self.raw_text,
            lines=tuple(self.lines),
            speaker_markers=tuple(self.speaker_markers),
            contains_multiple_speakers=self.contains_multiple_speakers,
        )

class ReconstructedSentencePartResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    segment_id: str = Field(..., alias="segmentId")
    text: str
    start_ms: int = Field(..., alias="startMs")
    end_ms: int = Field(..., alias="endMs")
    raw_text: str | None = Field(default=None, alias="rawText")
    lines: list[str] | None = Field(default=None)
    speaker: str | None = Field(default=None)
    speaker_markers: list[str] | None = Field(default=None, alias="speakerMarkers")
    contains_multiple_speakers: bool | None = Field(default=None, alias="containsMultipleSpeakers")

class ReconstructedSentenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    text: str
    start_ms: int = Field(..., alias="startMs")
    end_ms: int = Field(..., alias="endMs")
    segment_ids: list[str] = Field(..., alias="segmentIds")
    parts: list[ReconstructedSentencePartResponse]


class SubtitleReconstructionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[SubtitleSegmentRequest] = Field(..., min_length=1)


class SubtitleReconstructionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[SubtitleSegmentRequest]
    sentences: list[ReconstructedSentenceResponse]


class SimilarityRequest(BaseModel):
    """Input for comparing one target word in two sentences."""

    model_config = ConfigDict(extra="forbid")

    word: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="One word whose contextual meaning should be compared.",
        examples=["bank"],
    )
    sentence1: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="First sentence containing the target word.",
        examples=["I deposited my paycheck at the bank."],
    )
    sentence2: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Second sentence containing the target word.",
        examples=["The hikers rested on the bank of the river."],
    )
    threshold: float = Field(
        DEFAULT_THRESHOLD,
        ge=-1.0,
        le=1.0,
        description="Calibrated semantic-similarity cutoff used for is_similar.",
        examples=[0.80],
    )

    @field_validator("word", "sentence1", "sentence2", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("word")
    @classmethod
    def require_single_word(cls, value: str) -> str:
        if not WORD_PATTERN.fullmatch(value):
            raise ValueError("word must contain one alphabetic word, with optional internal hyphens or apostrophes")
        return value

    @model_validator(mode="after")
    def require_word_in_both_sentences(self) -> "SimilarityRequest":
        missing = []
        if not find_word_occurrences(self.word, self.sentence1):
            missing.append("sentence1")
        if not find_word_occurrences(self.word, self.sentence2):
            missing.append("sentence2")
        if missing:
            joined = " and ".join(missing)
            raise ValueError(f"word must occur as a matching lemma in {joined}")
        return self


class Occurrence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)


class Occurrences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence1: list[Occurrence]
    sentence2: list[Occurrence]


class SimilarityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    word: str
    similarity: float = Field(..., ge=-1.0, le=1.0)
    threshold: float = Field(..., ge=-1.0, le=1.0)
    is_similar: bool
    occurrences: Occurrences


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    model: str
    model_loaded: bool


app = FastAPI(
    title="Contextual Word Similarity API",
    version="1.0.0",
    description=(
        "Compares the meaning of one word as used in two sentences with "
        "a calibrated contextual embedding score, and reconstructs sentence "
        "boundaries from ordered text fragments with Segment Any Text."
    ),
)
similarity_service = MiniLMSimilarity()
sat_service = SaTSentenceReconstructor()


def get_similarity_service() -> MiniLMSimilarity:
    return similarity_service


def get_sat_service() -> SaTSentenceReconstructor:
    return sat_service


def get_subtitle_service(
    service: SaTSentenceReconstructor = Depends(get_sat_service),
) -> SubtitleSentenceReconstructor:
    return SubtitleSentenceReconstructor(service)




@app.exception_handler(ModelUnavailableError)
async def model_unavailable_handler(_, exc: ModelUnavailableError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )



@app.exception_handler(SaTUnavailableError)
async def sat_unavailable_handler(_, exc: SaTUnavailableError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: MiniLMSimilarity = Depends(get_similarity_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=service.model_name,
        model_loaded=service.is_loaded,
    )


@app.post("/similarity", response_model=SimilarityResponse, tags=["similarity"])
def similarity(
    request: SimilarityRequest,
    service: MiniLMSimilarity = Depends(get_similarity_service),
) -> SimilarityResponse:
    score = service.compare(request.word, request.sentence1, request.sentence2)
    return SimilarityResponse(
        model=service.model_name,
        word=request.word,
        similarity=score,
        threshold=request.threshold,
        is_similar=score >= request.threshold,
        occurrences=Occurrences(
            sentence1=[Occurrence(**item) for item in find_word_occurrences(request.word, request.sentence1)],
            sentence2=[Occurrence(**item) for item in find_word_occurrences(request.word, request.sentence2)],
        ),
    )


@app.post(
    "/sentence-from-segments",
    response_model=SentenceFromSegmentsResponse,
    tags=["segmentation"],
)
def sentence_from_segments(
    request: SentenceFromSegmentsRequest,
    service: SaTSentenceReconstructor = Depends(get_sat_service),
) -> SentenceFromSegmentsResponse:
    return SentenceFromSegmentsResponse(**service.reconstruct(request.segments))


@app.post(
    "/reconstruct-subtitles",
    response_model=SubtitleReconstructionResponse,
    tags=["segmentation"],
)
def reconstruct_subtitles(
    request: SubtitleReconstructionRequest,
    service: SubtitleSentenceReconstructor = Depends(get_subtitle_service),
) -> SubtitleReconstructionResponse:
    """Reconstruct sentence context while retaining every original cue."""
    segments = [segment.to_domain() for segment in request.segments]
    sentences = service.reconstruct(segments)
    return SubtitleReconstructionResponse(
        segments=request.segments,
        sentences=[ReconstructedSentenceResponse(**sentence.to_dict()) for sentence in sentences],
    )
