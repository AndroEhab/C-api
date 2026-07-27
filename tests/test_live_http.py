from __future__ import annotations

import os

import httpx
import pytest

from app.model import MODEL_NAME


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_API") != "1",
    reason="set RUN_LIVE_API=1 to exercise a running API server",
)

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

LIVE_CASES = [
    {
        "word": "book",
        "sentence1": "I read many books.",
        "sentence2": "I need to book a room.",
    },
    {
        "word": "bank",
        "sentence1": "I deposited money at the bank.",
        "sentence2": "The hikers rested on the bank of the river.",
    },
    {
        "word": "run",
        "sentence1": "Running every morning helps.",
        "sentence2": "She ran five miles yesterday.",
    },
    {
        "word": "child",
        "sentence1": "The children played outside.",
        "sentence2": "A child waved at us.",
    },
    {
        "word": "analysis",
        "sentence1": "The analysts completed several analyses.",
        "sentence2": "The analysis was reviewed today.",
    },
    {
        "word": "city",
        "sentence1": "Several cities hosted the event.",
        "sentence2": "The city hosted the final event.",
    },
    {
        "word": "book",
        "sentence1": "“BOOKS”—filled the shelves.",
        "sentence2": "(Book) clubs met weekly.",
    },
]


def test_live_http_api_matrix() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=180.0) as client:
        initial_health = client.get("/health")
        assert initial_health.status_code == 200
        assert initial_health.json()["model"] == MODEL_NAME

        for payload in LIVE_CASES:
            response = client.post("/similarity", json=payload)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["model"] == MODEL_NAME
            assert body["word"] == payload["word"]
            assert -1.0 <= body["similarity"] <= 1.0
            assert body["is_similar"] == (body["similarity"] >= body["threshold"])
            assert body["occurrences"]["sentence1"]
            assert body["occurrences"]["sentence2"]

        missing_target = client.post(
            "/similarity",
            json={
                "word": "book",
                "sentence1": "The bookstore is closed.",
                "sentence2": "I read books.",
            },
        )
        assert missing_target.status_code == 422

        threshold_override = client.post(
            "/similarity",
            json={
                **LIVE_CASES[0],
                "threshold": 1.0,
            },
        )
        assert threshold_override.status_code == 200
        override_body = threshold_override.json()
        assert override_body["threshold"] == 1.0
        assert override_body["is_similar"] == (override_body["similarity"] >= 1.0)

        openapi = client.get("/openapi.json")
        assert openapi.status_code == 200
        assert "/similarity" in openapi.json()["paths"]

        final_health = client.get("/health")
        assert final_health.status_code == 200
        assert final_health.json()["model_loaded"] is True
