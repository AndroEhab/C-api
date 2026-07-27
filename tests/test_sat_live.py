from __future__ import annotations

import os

import httpx
import pytest

from app.sat import SAT_MODEL_NAME


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_API") != "1",
    reason="set RUN_LIVE_API=1 to exercise a running SaT server",
)

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def test_live_sat_reconstructs_and_detects_sentence_boundaries() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=180.0) as client:
        one_sentence = client.post(
            "/sentence-from-segments",
            json={
                "segments": [
                    "This is the first part of",
                    "one complete sentence that",
                    "ends here.",
                ]
            },
        )
        assert one_sentence.status_code == 200, one_sentence.text
        one_body = one_sentence.json()
        assert one_body["model"] == SAT_MODEL_NAME
        assert one_body["joined_text"] == (
            "This is the first part of one complete sentence that ends here."
        )
        assert one_body["is_single_sentence"] is True
        assert one_body["full_sentence"] == one_body["joined_text"]

        two_sentences = client.post(
            "/sentence-from-segments",
            json={
                "segments": [
                    "First sentence ends here.",
                    "Second sentence starts",
                    "and ends here.",
                ]
            },
        )
        assert two_sentences.status_code == 200, two_sentences.text
        two_body = two_sentences.json()
        assert two_body["model"] == SAT_MODEL_NAME
        assert two_body["is_single_sentence"] is False
        assert two_body["full_sentence"] is None
        assert len(two_body["sentences"]) >= 2


def test_live_sat_accepts_arbitrary_fragment_counts() -> None:
    cases = [
        ["A single sentence."],
        ["Two fragments", "make one sentence."],
        ["Four", "small", "fragments", "make one sentence."],
    ]

    with httpx.Client(base_url=BASE_URL, timeout=180.0) as client:
        for segments in cases:
            response = client.post(
                "/sentence-from-segments",
                json={"segments": segments},
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["model"] == SAT_MODEL_NAME
            assert body["segments"] == segments
            assert body["joined_text"] == " ".join(segments)
            assert body["sentences"]
