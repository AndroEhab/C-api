from __future__ import annotations

from typing import Sequence

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_sat_service
from app.sat import SAT_MODEL_NAME, SaTSentenceReconstructor


class FakeSaTModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def split(self, text: str) -> Sequence[str]:
        self.calls.append(text)
        if "MULTI_SENTENCE" in text:
            return ["First detected sentence.", "Second detected sentence."]
        return [text]


class FailingSaTModel:
    def split(self, text: str) -> Sequence[str]:
        raise RuntimeError("controlled SaT failure")


@pytest.fixture
def api_and_model() -> tuple[TestClient, FakeSaTModel]:
    model = FakeSaTModel()
    service = SaTSentenceReconstructor(model=model)
    previous = app.dependency_overrides.get(get_sat_service)
    app.dependency_overrides[get_sat_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client, model
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_sat_service, None)
        else:
            app.dependency_overrides[get_sat_service] = previous


@pytest.fixture
def api(api_and_model: tuple[TestClient, FakeSaTModel]) -> TestClient:
    return api_and_model[0]


def test_three_segments_are_joined_and_presented_as_one_sentence(
    api_and_model: tuple[TestClient, FakeSaTModel],
) -> None:
    api, model = api_and_model
    response = api.post(
        "/sentence-from-segments",
        json={
            "segments": [
                "The quick brown",
                "fox jumps",
                "over the lazy dog.",
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": SAT_MODEL_NAME,
        "segments": ["The quick brown", "fox jumps", "over the lazy dog."],
        "joined_text": "The quick brown fox jumps over the lazy dog.",
        "sentences": ["The quick brown fox jumps over the lazy dog."],
        "full_sentence": "The quick brown fox jumps over the lazy dog.",
        "is_single_sentence": True,
    }
    assert model.calls == ["The quick brown fox jumps over the lazy dog."]


@pytest.mark.parametrize(
    "segments",
    [
        ["only one fragment."],
        ["one", "two"],
        ["one", "two", "three"],
        ["one", "two", "three", "four"],
        [f"fragment {index}" for index in range(12)],
    ],
    ids=["one", "two", "three", "four", "twelve"],
)
def test_any_positive_number_of_fragments_is_accepted(
    api: TestClient,
    segments: list[str],
) -> None:
    response = api.post("/sentence-from-segments", json={"segments": segments})

    assert response.status_code == 200
    body = response.json()
    expected_joined = " ".join(segments)
    assert body["segments"] == segments
    assert body["joined_text"] == expected_joined
    assert body["sentences"] == [expected_joined]
    assert body["full_sentence"] == expected_joined
    assert body["is_single_sentence"] is True


def test_joining_does_not_insert_spaces_before_punctuation(api: TestClient) -> None:
    response = api.post(
        "/sentence-from-segments",
        json={"segments": ["Hello", ", world", "!"]},
    )

    assert response.status_code == 200
    assert response.json()["joined_text"] == "Hello, world!"
    assert response.json()["full_sentence"] == "Hello, world!"


def test_multiple_detected_sentences_are_returned_without_single_full_sentence(
    api: TestClient,
) -> None:
    response = api.post(
        "/sentence-from-segments",
        json={"segments": ["MULTI_SENTENCE", "fragment", "tail"]},
    )

    assert response.status_code == 200
    assert response.json()["sentences"] == [
        "First detected sentence.",
        "Second detected sentence.",
    ]
    assert response.json()["full_sentence"] is None
    assert response.json()["is_single_sentence"] is False


@pytest.mark.parametrize(
    "segments",
    [
        [],
        ["one", "", "three"],
        ["one", "   ", "three"],
        ["one", None, "three"],
        ["one", ["nested"], "three"],
        ["one", "two", "x" * 5001],
    ],
    ids=[
        "zero-segments",
        "empty-segment",
        "whitespace-segment",
        "null-segment",
        "array-segment",
        "oversized-segment",
    ],
)
def test_segment_shape_and_content_are_validated(api: TestClient, segments: object) -> None:
    response = api.post("/sentence-from-segments", json={"segments": segments})

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_extra_request_fields_are_rejected(api: TestClient) -> None:
    response = api.post(
        "/sentence-from-segments",
        json={"segments": ["one", "two", "three"], "model": "other"},
    )

    assert response.status_code == 422


def test_sat_failure_returns_503() -> None:
    service = SaTSentenceReconstructor(model=FailingSaTModel())
    previous = app.dependency_overrides.get(get_sat_service)
    app.dependency_overrides[get_sat_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/sentence-from-segments",
            json={"segments": ["one", "two", "three"]},
        )
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_sat_service, None)
        else:
            app.dependency_overrides[get_sat_service] = previous

    assert response.status_code == 503
    assert response.json() == {"detail": "Unable to segment text with SaT"}


def test_group_contract_returns_explicit_source_indexes_and_ids() -> None:
    service = SaTSentenceReconstructor(model=FakeSaTModel())

    response = service.group(
        [
            {"segmentId": "144", "text": "This case"},
            {"segmentId": "145", "text": "could be a breakthrough."},
        ]
    )

    assert response == {
        "model": SAT_MODEL_NAME,
        "groups": [
            {
                "segmentIndexes": [0, 1],
                "segmentIds": ["144", "145"],
                "text": "This case could be a breakthrough.",
            }
        ],
    }


def test_segmentation_route_is_in_openapi(api: TestClient) -> None:
    response = api.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert "/sentence-from-segments" in document["paths"]
    assert "post" in document["paths"]["/sentence-from-segments"]
