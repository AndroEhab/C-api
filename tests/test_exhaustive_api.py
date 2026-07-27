from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_similarity_service
from app.model import MODEL_NAME, MiniLMSimilarity


class ControlledEmbeddingModel:
    """Deterministic vectors for testing API behavior without model inference noise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        texts = tuple(sentences)
        self.calls.append(texts)
        lowered = [text.lower() for text in texts]

        if any("encode-failure" in text for text in lowered):
            raise RuntimeError("controlled inference failure")
        if any("invalid-shape" in text for text in lowered):
            return [[1.0, 0.0, 0.0]]

        vectors: list[list[float]] = []
        for text in lowered:
            if "zero-vector" in text:
                vectors.append([0.0, 0.0, 0.0])
            elif "nan-vector" in text:
                vectors.append([float("nan"), 0.0, 0.0])
            elif "opposite" in text:
                vectors.append([-1.0, 0.0, 0.0])
            elif "orthogonal" in text:
                vectors.append([0.0, 1.0, 0.0])
            elif "moderate" in text:
                vectors.append([0.6, 0.8, 0.0])
            elif "diagonal" in text:
                vectors.append([1.0, 1.0, 0.0])
            else:
                vectors.append([1.0, 0.0, 0.0])
        return vectors


@pytest.fixture
def api_and_model() -> tuple[TestClient, ControlledEmbeddingModel]:
    model = ControlledEmbeddingModel()
    service = MiniLMSimilarity(model=model)
    previous = app.dependency_overrides.get(get_similarity_service)
    app.dependency_overrides[get_similarity_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client, model
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_similarity_service, None)
        else:
            app.dependency_overrides[get_similarity_service] = previous


@pytest.fixture
def api(api_and_model: tuple[TestClient, ControlledEmbeddingModel]) -> TestClient:
    return api_and_model[0]


def valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "word": "book",
        "sentence1": "finance book sales increased.",
        "sentence2": "finance books sold quickly.",
    }
    payload.update(overrides)
    return payload


LEMMA_CASES = [
    ("book", "books", "book"),
    ("run", "running", "ran"),
    ("study", "studies", "studied"),
    ("try", "tries", "tried"),
    ("carry", "carried", "carries"),
    ("go", "went", "going"),
    ("be", "was", "are"),
    ("have", "had", "has"),
    ("mouse", "mice", "mouse"),
    ("child", "children", "child"),
    ("woman", "women", "woman"),
    ("foot", "feet", "foot"),
    ("tooth", "teeth", "tooth"),
    ("goose", "geese", "goose"),
    ("man", "men", "man"),
    ("leave", "leaves", "left"),
    ("city", "cities", "city"),
    ("box", "boxes", "box"),
    ("stop", "stopped", "stops"),
    ("write", "writing", "written"),
    ("eat", "ate", "eating"),
    ("drive", "drove", "driven"),
    ("analysis", "analyses", "analysis"),
    ("crisis", "crises", "crisis"),
    ("knife", "knives", "knife"),
    ("baby", "babies", "baby"),
    ("dance", "danced", "dancing"),
    ("see", "saw", "seen"),
    ("take", "took", "taken"),
    ("speak", "spoke", "spoken"),
    ("person", "persons", "person"),
    ("wolf", "wolves", "wolf"),
]


@pytest.mark.parametrize("lemma,form1,form2", LEMMA_CASES)
def test_common_lemma_forms_are_accepted(
    api: TestClient,
    lemma: str,
    form1: str,
    form2: str,
) -> None:
    sentence1 = f"finance {form1} matter."
    sentence2 = f"finance {form2} matter."
    response = api.post(
        "/similarity",
        json={"word": lemma, "sentence1": sentence1, "sentence2": sentence2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["word"] == lemma
    assert body["is_similar"] is True
    assert body["occurrences"]["sentence1"] == [
        {"start": sentence1.index(form1), "end": sentence1.index(form1) + len(form1)}
    ]
    assert body["occurrences"]["sentence2"] == [
        {"start": sentence2.index(form2), "end": sentence2.index(form2) + len(form2)}
    ]


def test_inflected_input_is_normalized_to_its_lemma(api: TestClient) -> None:
    response = api.post(
        "/similarity",
        json={
            "word": "books",
            "sentence1": "finance book sales increased.",
            "sentence2": "finance books sold quickly.",
        },
    )

    assert response.status_code == 200
    assert response.json()["word"] == "books"
    assert response.json()["occurrences"]["sentence1"] == [{"start": 8, "end": 12}]
    assert response.json()["occurrences"]["sentence2"] == [{"start": 8, "end": 13}]


def test_matching_is_case_insensitive_and_preserves_surface_offsets(api: TestClient) -> None:
    sentence1 = "“BOOKS”—matter."
    sentence2 = "(Book) matters."
    response = api.post(
        "/similarity",
        json={"word": "Book", "sentence1": sentence1, "sentence2": sentence2},
    )

    assert response.status_code == 200
    assert response.json()["occurrences"] == {
        "sentence1": [{"start": 1, "end": 6}],
        "sentence2": [{"start": 1, "end": 5}],
    }


def test_input_and_sentence_whitespace_is_trimmed(api: TestClient) -> None:
    response = api.post(
        "/similarity",
        json={
            "word": "  book  ",
            "sentence1": "  finance books.  ",
            "sentence2": " finance book ",
        },
    )

    assert response.status_code == 200
    assert response.json()["word"] == "book"
    assert response.json()["occurrences"] == {
        "sentence1": [{"start": 8, "end": 13}],
        "sentence2": [{"start": 8, "end": 12}],
    }


def test_all_matching_occurrences_are_returned(api: TestClient) -> None:
    sentence1 = "finance Books, books, and books."
    sentence2 = "finance book books."
    response = api.post(
        "/similarity",
        json={"word": "book", "sentence1": sentence1, "sentence2": sentence2},
    )

    assert response.status_code == 200
    first = sentence1.index("Books")
    second = sentence1.index("books", first + 1)
    third = sentence1.index("books", second + 1)
    assert response.json()["occurrences"] == {
        "sentence1": [
            {"start": first, "end": first + 5},
            {"start": second, "end": second + 5},
            {"start": third, "end": third + 5},
        ],
        "sentence2": [
            {"start": 8, "end": 12},
            {"start": 13, "end": 18},
        ],
    }


def test_substrings_are_not_treated_as_whole_lemma_matches(api: TestClient) -> None:
    response = api.post(
        "/similarity",
        json={
            "word": "book",
            "sentence1": "The bookstore is open.",
            "sentence2": "I read books.",
        },
    )

    assert response.status_code == 422
    assert "sentence1" in str(response.json()["detail"])


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "not an object",
        {},
        {"sentence1": "book", "sentence2": "books"},
        {"word": "book", "sentence1": "book"},
        {"word": "book", "sentence2": "books"},
        valid_payload(word=""),
        valid_payload(word="   "),
        valid_payload(word="credit card"),
        valid_payload(word="123"),
        valid_payload(word="book_thing"),
        valid_payload(word="a" * 101),
        valid_payload(word=None),
        valid_payload(word=True),
        valid_payload(sentence1=""),
        valid_payload(sentence2="   "),
        valid_payload(sentence1=["book"]),
        valid_payload(sentence2=None),
        valid_payload(sentence1="book" + "x" * 5000),
        valid_payload(sentence2="book" + "x" * 5000),
        valid_payload(threshold=-1.01),
        valid_payload(threshold=1.01),
        valid_payload(threshold="not a number"),
        valid_payload(threshold=None),
        {**valid_payload(), "unexpected": True},
    ],
    ids=[
        "null-body",
        "array-body",
        "scalar-body",
        "empty-object",
        "missing-word",
        "missing-sentence2",
        "missing-sentence1",
        "empty-word",
        "whitespace-word",
        "phrase-word",
        "numeric-word",
        "underscore-word",
        "word-too-long",
        "null-word",
        "boolean-word",
        "empty-sentence1",
        "whitespace-sentence2",
        "array-sentence1",
        "null-sentence2",
        "sentence1-too-long",
        "sentence2-too-long",
        "threshold-too-low",
        "threshold-too-high",
        "threshold-text",
        "null-threshold",
        "unknown-field",
    ],
)
def test_malformed_requests_return_422(api: TestClient, payload: Any) -> None:
    response = api.post("/similarity", json=payload)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize(
    "sentence1,sentence2,missing",
    [
        ("bookstore", "books", "sentence1"),
        ("book", "notebook", "sentence2"),
        ("bookstore", "notebook", "sentence1 and sentence2"),
        ("reader", "books", "sentence1"),
    ],
)
def test_missing_lemma_matches_are_reported(
    api: TestClient,
    sentence1: str,
    sentence2: str,
    missing: str,
) -> None:
    response = api.post(
        "/similarity",
        json={"word": "book", "sentence1": sentence1, "sentence2": sentence2},
    )

    assert response.status_code == 422
    error_text = str(response.json()["detail"])
    assert "matching lemma" in error_text
    assert missing in error_text


@pytest.mark.parametrize(
    "first_marker,second_marker,expected_score,threshold,expected_flag",
    [
        ("", "", 1.0, 0.80, True),
        ("", "orthogonal", 0.0, 0.80, False),
        ("", "moderate", 0.6, 0.60, True),
        ("", "moderate", 0.6, 0.600001, False),
        ("", "diagonal", 2**-0.5, 0.70, True),
        ("", "diagonal", 2**-0.5, 0.71, False),
        ("", "opposite", -1.0, -1.0, True),
        ("", "opposite", -1.0, -0.999, False),
    ],
)
def test_similarity_scores_and_threshold_boundaries(
    api: TestClient,
    first_marker: str,
    second_marker: str,
    expected_score: float,
    threshold: float,
    expected_flag: bool,
) -> None:
    first = f"{first_marker} book".strip()
    second = f"{second_marker} book".strip()
    response = api.post(
        "/similarity",
        json={
            "word": "book",
            "sentence1": first,
            "sentence2": second,
            "threshold": threshold,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["similarity"] == pytest.approx(expected_score, abs=1e-6)
    assert body["threshold"] == pytest.approx(threshold)
    assert body["is_similar"] is expected_flag


@pytest.mark.parametrize("marker", ["encode-failure", "invalid-shape", "zero-vector", "nan-vector"])
def test_embedding_failures_return_503(api: TestClient, marker: str) -> None:
    response = api.post(
        "/similarity",
        json={
            "word": "book",
            "sentence1": f"{marker} book",
            "sentence2": "book",
        },
    )

    assert response.status_code == 503
    assert isinstance(response.json()["detail"], str)


def test_embedding_model_is_called_once_with_two_target_contexts(
    api_and_model: tuple[TestClient, ControlledEmbeddingModel],
) -> None:
    api, model = api_and_model
    response = api.post(
        "/similarity",
        json={
            "word": "book",
            "sentence1": "finance books are popular.",
            "sentence2": "finance book sales grew.",
        },
    )

    assert response.status_code == 200
    assert len(model.calls) == 1
    assert len(model.calls[0]) == 2
    assert all('Target word: "book"' in context for context in model.calls[0])
    assert all("finance" in context for context in model.calls[0])


def test_success_response_has_stable_contract(api: TestClient) -> None:
    response = api.post("/similarity", json=valid_payload())

    assert response.status_code == 200
    assert set(response.json()) == {
        "model",
        "word",
        "similarity",
        "threshold",
        "is_similar",
        "occurrences",
    }
    body = response.json()
    assert body["model"] == MODEL_NAME
    assert body["word"] == "book"
    assert isinstance(body["similarity"], float)
    assert -1.0 <= body["similarity"] <= 1.0
    assert isinstance(body["threshold"], float)
    assert isinstance(body["is_similar"], bool)
    for sentence_occurrences in body["occurrences"].values():
        for occurrence in sentence_occurrences:
            assert occurrence["start"] >= 0
            assert occurrence["end"] > occurrence["start"]


def test_health_reports_loaded_injected_model(api: TestClient) -> None:
    response = api.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": True,
    }


def test_openapi_describes_similarity_and_health_routes(api: TestClient) -> None:
    response = api.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert "/similarity" in document["paths"]
    assert "post" in document["paths"]["/similarity"]
    assert "/health" in document["paths"]
    assert document["info"]["title"] == "Contextual Word Similarity API"


def test_interactive_docs_are_available(api: TestClient) -> None:
    response = api.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "swagger" in response.text.lower()


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/similarity"),
        ("put", "/similarity"),
        ("delete", "/similarity"),
        ("post", "/health"),
        ("put", "/health"),
    ],
)
def test_unsupported_methods_are_rejected(api: TestClient, method: str, path: str) -> None:
    response = getattr(api, method)(path)

    assert response.status_code == 405


def test_unknown_route_returns_404(api: TestClient) -> None:
    response = api.get("/does-not-exist")

    assert response.status_code == 404


def test_lemma_matching_is_stable_under_concurrent_requests(api: TestClient) -> None:
    payloads = [
        valid_payload(sentence1="finance books are popular."),
        {
            "word": "child",
            "sentence1": "finance children are learning.",
            "sentence2": "finance child is learning.",
        },
        {
            "word": "run",
            "sentence1": "finance running is healthy.",
            "sentence2": "finance ran yesterday.",
        },
    ]

    def post(payload: dict[str, Any]) -> Any:
        return api.post("/similarity", json=payload)

    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(post, payloads * 8))

    assert len(responses) == 24
    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["occurrences"]["sentence1"] for response in responses)
    assert all(response.json()["occurrences"]["sentence2"] for response in responses)
