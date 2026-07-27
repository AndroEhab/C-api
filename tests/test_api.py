from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_similarity_service
from app.model import MiniLMSimilarity


class FakeEmbeddingModel:
    """Deterministic vectors that keep API tests independent of model downloads."""

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        vectors = []
        for sentence in sentences:
            text = sentence.lower()
            if any(token in text for token in ("money", "paycheck", "deposit")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(token in text for token in ("river", "shore", "water")):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([1.0, 1.0, 0.0])
        return vectors


class FailingEmbeddingModel:
    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        raise RuntimeError("inference failed")


@pytest.fixture
def client() -> TestClient:
    service = MiniLMSimilarity(model=FakeEmbeddingModel())
    app.dependency_overrides[get_similarity_service] = lambda: service
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_same_contextual_meaning_is_similar(client: TestClient) -> None:
    response = client.post(
        "/similarity",
        json={
            "word": "bank",
            "sentence1": "I deposited money at the bank.",
            "sentence2": "The bank handled my paycheck deposit.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "all-MiniLM-L6-v2"
    assert body["word"] == "bank"
    assert body["similarity"] == pytest.approx(1.0)
    assert body["threshold"] == pytest.approx(0.80)
    assert body["is_similar"] is True
    assert body["occurrences"]["sentence1"] == [{"start": 25, "end": 29}]
    assert body["occurrences"]["sentence2"] == [{"start": 4, "end": 8}]


def test_inflected_forms_match_the_entered_lemma(client: TestClient) -> None:
    response = client.post(
        "/similarity",
        json={
            "word": "book",
            "sentence1": "I read many books.",
            "sentence2": "I need to book a room.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["is_similar"] is True
    assert body["occurrences"]["sentence1"] == [{"start": 12, "end": 17}]
    assert body["occurrences"]["sentence2"] == [{"start": 10, "end": 14}]

def test_different_contextual_meaning_is_not_similar(client: TestClient) -> None:
    response = client.post(
        "/similarity",
        json={
            "word": "bank",
            "sentence1": "I deposited money at the bank.",
            "sentence2": "The hikers rested on the bank of the river.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["similarity"] == pytest.approx(0.0)
    assert body["is_similar"] is False


def test_target_word_must_appear_in_both_sentences(client: TestClient) -> None:
    response = client.post(
        "/similarity",
        json={
            "word": "bank",
            "sentence1": "I deposited money at the bank.",
            "sentence2": "The hikers rested beside the river.",
        },
    )

    assert response.status_code == 422
    assert "word must occur" in response.json()["detail"][0]["msg"]


def test_target_must_be_a_single_word(client: TestClient) -> None:
    response = client.post(
        "/similarity",
        json={
            "word": "credit card",
            "sentence1": "I used my credit card at the bank.",
            "sentence2": "The bank issued a credit card.",
        },
    )

    assert response.status_code == 422


def test_model_failures_return_service_unavailable() -> None:
    service = MiniLMSimilarity(model=FailingEmbeddingModel())
    app.dependency_overrides[get_similarity_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/similarity",
            json={
                "word": "bank",
                "sentence1": "I deposited money at the bank.",
                "sentence2": "The bank handled my paycheck deposit.",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Unable to compute sentence embeddings"}
