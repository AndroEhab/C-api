from __future__ import annotations

from typing import Any, Sequence

import pytest
from fastapi.testclient import TestClient

from app.language_profile import EnglishBoundaryProfile, NeutralBoundaryProfile
from app.main import app, get_sat_service
from app.sat import SAT_MODEL_NAME, SaTUnavailableError, SaTSentenceReconstructor, WindowedScoringConfig


class FakeSaTModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def split(self, text: str) -> Sequence[str]:
        self.calls.append(text)
        if "MULTI_SENTENCE" in text:
            return ["First detected sentence.", "Second detected sentence."]
        return [text]


class FakeProbabilitySaTModel(FakeSaTModel):
    def __init__(self, probabilities: Sequence[float] | Sequence[Sequence[float]] | None = None) -> None:
        super().__init__()
        self.probability_calls: list[str] = []
        self._probabilities = probabilities

    def predict_proba(self, text: str) -> Sequence[float] | Sequence[Sequence[float]]:
        self.probability_calls.append(text)
        if self._probabilities is not None:
            return self._probabilities
        return [[index / 100] for index in range(len(text))]


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


def test_score_boundaries_returns_one_ordered_score_per_original_boundary() -> None:
    model = FakeProbabilitySaTModel()
    service = SaTSentenceReconstructor(model=model)

    evidence = service.score_boundaries(["one", "two", "three"])

    assert model.probability_calls == ["one two three"]
    assert evidence == [
        {
            "leftIndex": 0,
            "rightIndex": 1,
            "characterOffset": 2,
            "boundaryProbability": 0.02,
        },
        {
            "leftIndex": 1,
            "rightIndex": 2,
            "characterOffset": 6,
            "boundaryProbability": 0.06,
        },
    ]
    assert len(evidence) == 2
    assert [item["leftIndex"] for item in evidence] == [0, 1]
    assert [item["rightIndex"] for item in evidence] == [1, 2]
    assert [item["characterOffset"] for item in evidence] == [2, 6]


def test_score_boundaries_preserves_offsets_when_joining_without_punctuation_space() -> None:
    model = FakeProbabilitySaTModel()
    service = SaTSentenceReconstructor(model=model)

    evidence = service.score_boundaries(["Hello", ", world", "again"])

    assert model.probability_calls == ["Hello, world again"]
    assert [item["characterOffset"] for item in evidence] == [4, 11]
    assert [item["boundaryProbability"] for item in evidence] == [0.04, 0.11]


def test_score_boundaries_rejects_wrong_probability_count() -> None:
    service = SaTSentenceReconstructor(model=FakeProbabilitySaTModel([0.5]))

    with pytest.raises(SaTUnavailableError, match="expected 7 character probabilities"):
        service.score_boundaries(["one", "two"])


@pytest.mark.parametrize("bad_probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_score_boundaries_rejects_probability_out_of_range(bad_probability: float) -> None:
    service = SaTSentenceReconstructor(
        model=FakeProbabilitySaTModel([bad_probability] * len("one two"))
    )

    with pytest.raises(SaTUnavailableError, match="within \\[0, 1\\]"):
        service.score_boundaries(["one", "two"])


def test_score_boundaries_fails_safely_without_raw_probability_support() -> None:
    service = SaTSentenceReconstructor(model=FakeSaTModel())

    with pytest.raises(SaTUnavailableError, match="raw boundary probability"):
        service.score_boundaries(["one", "two"])


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


class WindowTraceModel:
    """Model that encodes character position into probability and can fail on
    specific calls.  Each probability = (index + 1) / 10000, which is always
    non-zero and within [0, 1].  A zero probability or None is therefore
    never a legitimate model result — it indicates a fallback or unavailable
    boundary."""

    def __init__(self, fail_on_calls: set[int] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on_calls: set[int] = fail_on_calls or set()

    def predict_proba(self, text: str) -> list[list[float]]:
        self.calls.append(text)
        call_index = len(self.calls) - 1
        if call_index in self.fail_on_calls:
            raise RuntimeError(f"controlled model failure on call {call_index}")
        return [[(index + 1) / 10000] for index in range(len(text))]

    def split(self, text: str) -> Sequence[str]:
        return [text]


def _cues(count: int, prefix: str = "cue") -> list[str]:
    return [f"{prefix}-{i}" for i in range(count)]


class TestWindowedScoreBoundaries:
    """Direct tests for SaTSentenceReconstructor.windowed_score_boundaries()."""

    def test_100_cues_yields_99_evidence_entries(self) -> None:
        model = WindowTraceModel()
        service = SaTSentenceReconstructor(model=model)
        cues = _cues(100)
        evidence = service.windowed_score_boundaries(cues)
        assert len(evidence) == 99

    def test_every_left_right_index_is_unique_and_correct(self) -> None:
        model = WindowTraceModel()
        service = SaTSentenceReconstructor(model=model)
        cues = _cues(200)
        evidence = service.windowed_score_boundaries(cues)
        seen_pairs: set[tuple[int, int]] = set()
        for entry in evidence:
            pair = (entry["leftIndex"], entry["rightIndex"])
            assert pair not in seen_pairs, f"duplicate boundary pair {pair}"
            seen_pairs.add(pair)
            assert entry["rightIndex"] == entry["leftIndex"] + 1
        assert len(seen_pairs) == 199
        assert min(p[0] for p in seen_pairs) == 0
        assert max(p[0] for p in seen_pairs) == 198

    def test_first_boundary_receives_real_model_score(self) -> None:
        model = WindowTraceModel()
        service = SaTSentenceReconstructor(model=model)
        cues = _cues(200)
        evidence = service.windowed_score_boundaries(cues)
        first = evidence[0]
        assert first["leftIndex"] == 0
        assert first["rightIndex"] == 1
        assert first["boundaryProbability"] is not None
        assert first["boundaryProbability"] > 0.0
        assert "status" not in first or first.get("status") != "unavailable"

    def test_final_boundary_receives_real_model_score(self) -> None:
        model = WindowTraceModel()
        service = SaTSentenceReconstructor(model=model)
        cues = _cues(200)
        evidence = service.windowed_score_boundaries(cues)
        last = evidence[-1]
        assert last["leftIndex"] == 198
        assert last["rightIndex"] == 199
        assert last["boundaryProbability"] is not None
        assert last["boundaryProbability"] > 0.0
        assert "status" not in last or last.get("status") != "unavailable"

    def test_no_successful_boundary_has_synthetic_fallback(self) -> None:
        model = WindowTraceModel()
        service = SaTSentenceReconstructor(model=model)
        cues = _cues(200)
        evidence = service.windowed_score_boundaries(cues)
        for entry in evidence:
            assert entry["boundaryProbability"] is not None, (
                f"boundary {entry['leftIndex']} has fallback None"
            )
            assert entry["boundaryProbability"] > 0.0, (
                f"boundary {entry['leftIndex']} has fallback 0.0"
            )

    def test_failed_middle_window_marks_only_its_owned_as_unavailable(self) -> None:
        model = WindowTraceModel(fail_on_calls={1})  # second window call fails
        service = SaTSentenceReconstructor(model=model)
        # 100 cues, W=48, C=6 → 3 windows
        # Window 0: segments [0, 48), owns boundaries [0, 41]
        # Window 1: segments [36, 84), owns boundaries [42, 77]
        # Window 2: segments [72, 100), owns boundaries [78, 98]
        cfg = WindowedScoringConfig(window_size=48, min_context=6)
        cues = _cues(100)
        evidence = service.windowed_score_boundaries(cues, config=cfg)

        # Boundaries 0-41 should be available (window 0 succeeded)
        for i in range(0, 42):
            assert evidence[i]["boundaryProbability"] is not None, (
                f"boundary {i} should have real probability"
            )
            assert evidence[i]["boundaryProbability"] > 0.0
            assert evidence[i].get("status") != "unavailable"

        # Boundaries 42-77 should be unavailable (window 1 failed)
        for i in range(42, 78):
            assert evidence[i].get("boundaryProbability") is None, (
                f"boundary {i} should be None"
            )
            assert evidence[i].get("status") == "unavailable", (
                f"boundary {i} should have status unavailable"
            )

        # Boundaries 78-98 should be available (window 2 succeeded)
        for i in range(78, 99):
            assert evidence[i]["boundaryProbability"] is not None, (
                f"boundary {i} should have real probability"
            )
            assert evidence[i]["boundaryProbability"] > 0.0
            assert evidence[i].get("status") != "unavailable"

    def test_successful_windows_before_and_after_failure_remain_usable(self) -> None:
        model = WindowTraceModel(fail_on_calls={1})
        service = SaTSentenceReconstructor(model=model)
        cfg = WindowedScoringConfig(window_size=48, min_context=6)
        cues = _cues(100)
        evidence = service.windowed_score_boundaries(cues, config=cfg)

        # Successful window 0: all owned boundaries should have valid probabilities
        window0_probs = [evidence[i]["boundaryProbability"] for i in range(0, 42)]
        assert all(p is not None and p > 0.0 for p in window0_probs)

        # Failed window 1: all owned boundaries should be None
        failed_probs = [evidence[i]["boundaryProbability"] for i in range(42, 78)]
        assert all(p is None for p in failed_probs)

        # Successful window 2: all owned boundaries should have valid probabilities
        window2_probs = [evidence[i]["boundaryProbability"] for i in range(78, 99)]
        assert all(p is not None and p > 0.0 for p in window2_probs)

    def test_unavailable_evidence_results_in_uncertain_break_in_policy(self) -> None:
        """Unavailable probability → modelProbability=None → UNCERTAIN → treated as BREAK."""
        from app.subtitles import (
            SubtitleSentenceReconstructor,
            SubtitleSegment,
        )

        class WindowWithFailures:
            """A boundary API that returns unavailable evidence for all boundaries."""
            def windowed_score_boundaries(
                self,
                segments: list[str],
                *,
                profile: Any = None,
            ) -> list[dict]:
                n = len(segments)
                return [
                    {
                        "leftIndex": i,
                        "rightIndex": i + 1,
                        "characterOffset": 0,
                        "boundaryProbability": None,
                        "status": "unavailable",
                    }
                    for i in range(n - 1)
                ]

        api = WindowWithFailures()
        reconstructor = SubtitleSentenceReconstructor(api)
        segments = [
            SubtitleSegment("a", "Hello", 0, 500),
            SubtitleSegment("b", "world", 520, 1000),
        ]
        decisions = reconstructor.evaluate_boundaries(segments)
        assert len(decisions) == 1
        # None probability → uncertain (which is treated as BREAK)
        assert decisions[0]["decision"] in ("uncertain", "break")
        assert decisions[0]["decision"] != "join"
        assert decisions[0]["modelProbability"] is None

    def test_overlap_does_not_produce_duplicate_decisions(self) -> None:
        model = WindowTraceModel()
        service = SaTSentenceReconstructor(model=model)
        # 10 cues, W=8, C=3 → 2 windows, each boundary has 1 owner
        cfg = WindowedScoringConfig(window_size=8, min_context=3)
        cues = _cues(10)
        evidence = service.windowed_score_boundaries(cues, config=cfg)
        assert len(evidence) == 9
        # Verify plan coverage
        left_indices = [e["leftIndex"] for e in evidence]
        assert left_indices == list(range(9))

    def test_invalid_window_configuration_raises_clear_error(self) -> None:
        with pytest.raises(ValueError, match="window_size .* must be >= 2 \\* min_context \\+ 2"):
            WindowedScoringConfig(window_size=5, min_context=3)

        with pytest.raises(ValueError, match="window_size must be a positive integer"):
            WindowedScoringConfig(window_size=-1)

        with pytest.raises(ValueError, match="min_context must be a positive integer"):
            WindowedScoringConfig(min_context=0)

    def test_small_input_uses_normal_scoring(self) -> None:
        model = WindowTraceModel()
        service = SaTSentenceReconstructor(model=model)
        # 3 cues is smaller than one window → single-call scoring
        cues = _cues(3)
        evidence = service.windowed_score_boundaries(cues)
        assert len(evidence) == 2
        assert len(model.calls) == 1  # single predict_proba call
        for entry in evidence:
            assert entry["boundaryProbability"] is not None
            assert entry["boundaryProbability"] > 0.0

    def test_large_file_returns_exact_boundary_count(self) -> None:
        model = WindowTraceModel()
        service = SaTSentenceReconstructor(model=model)
        # 651 cues → 650 boundaries
        cues = _cues(651)
        evidence = service.windowed_score_boundaries(cues)
        assert len(evidence) == 650
        left_indices = [e["leftIndex"] for e in evidence]
        assert left_indices == list(range(650))
        # Verify the first and last have real scores
        assert evidence[0]["boundaryProbability"] is not None
        assert evidence[0]["boundaryProbability"] > 0.0
        assert evidence[-1]["boundaryProbability"] is not None
        assert evidence[-1]["boundaryProbability"] > 0.0

    def test_config_is_used_and_validated(self) -> None:
        model = WindowTraceModel()
        service = SaTSentenceReconstructor(model=model)
        cfg = WindowedScoringConfig(window_size=24, min_context=4)
        cues = _cues(200)
        evidence = service.windowed_score_boundaries(cues, config=cfg)
        assert len(evidence) == 199
        # All boundaries have valid probabilities (non-zero)
        assert all(e["boundaryProbability"] is not None and e["boundaryProbability"] > 0.0 for e in evidence)


# ──── Task 7: Profile propagation through SaT ─────────────────────────────────


class RecordingProbabilityModel:
    """A fake SaT probability model that records every predict_proba() input.

    Supports optional synchronisation via a ``barrier`` so two requests can
    overlap during scoring for concurrency tests.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.barrier: threading.Barrier | None = None

    def predict_proba(self, text: str) -> list[list[float]]:
        self.calls.append(text)
        if self.barrier is not None:
            self.barrier.wait()
        return [[min(0.01 * (i + 1), 0.99)] for i in range(len(text))]

    def split(self, text: str) -> Sequence[str]:
        return [text]


def test_score_boundaries_model_input_uses_profile() -> None:
    """score_boundaries feeds profile-joined text into predict_proba."""

    # ENGLISH: contraction attached -> "I'm ready."
    eng_model = RecordingProbabilityModel()
    eng_svc = SaTSentenceReconstructor(model=eng_model)
    eng_evidence = eng_svc.score_boundaries(
        ["I", "'m ready."],
        profile=EnglishBoundaryProfile(),
    )
    assert len(eng_model.calls) == 1
    assert eng_model.calls[0] == "I'm ready.", (
        f"English model input should be joined, got {eng_model.calls[0]!r}"
    )
    # Character offset = index of last char of left segment in joined text.
    # English joined="I'm ready.", last char of "I" is at index 0.
    assert eng_evidence[0]["characterOffset"] == 0, (
        f"English offset {eng_evidence[0]['characterOffset']} != 0"
    )

    # NEUTRAL: space kept -> "I 'm ready."
    neutral_model = RecordingProbabilityModel()
    neutral_svc = SaTSentenceReconstructor(model=neutral_model)
    neutral_evidence = neutral_svc.score_boundaries(
        ["I", "'m ready."],
        profile=NeutralBoundaryProfile(),
    )
    assert len(neutral_model.calls) == 1
    assert neutral_model.calls[0] == "I 'm ready.", (
        f"Neutral model input should keep space, got {neutral_model.calls[0]!r}"
    )
    # Neutral joined="I 'm ready.", last char of "I" is at index 0.
    assert neutral_evidence[0]["characterOffset"] == 0, (
        f"Neutral offset {neutral_evidence[0]['characterOffset']} != 0"
    )


def test_concurrent_english_and_spanish_are_isolated() -> None:
    """Simultaneous English and Spanish requests receive correct profile-joined model input."""
    import threading

    barrier = threading.Barrier(2)  # both threads meet at predict_proba
    eng_model = RecordingProbabilityModel()
    eng_model.barrier = barrier
    spa_model = RecordingProbabilityModel()
    spa_model.barrier = barrier

    results: dict[str, Any] = {}
    errors: list[str] = []

    def english_request() -> None:
        try:
            svc = SaTSentenceReconstructor(model=eng_model)
            evidence = svc.score_boundaries(
                ["I", "'m ready."],
                profile=EnglishBoundaryProfile(),
            )
            results["eng_input"] = list(eng_model.calls)
            results["eng_evidence"] = evidence
        except Exception as exc:
            errors.append(f"English: {exc!r}")

    def spanish_request() -> None:
        try:
            svc = SaTSentenceReconstructor(model=spa_model)
            evidence = svc.score_boundaries(
                ["I", "'m ready."],
                profile=NeutralBoundaryProfile(),
            )
            results["spa_input"] = list(spa_model.calls)
            results["spa_evidence"] = evidence
        except Exception as exc:
            errors.append(f"Spanish: {exc!r}")

    t1 = threading.Thread(target=english_request)
    t2 = threading.Thread(target=spanish_request)

    t1.start()
    t2.start()

    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"
    assert "eng_input" in results, f"English didn't complete. results keys: {list(results)}"
    assert "spa_input" in results, f"Spanish didn't complete. results keys: {list(results)}"

    # English always receives English-joined model input
    assert results["eng_input"] == ["I'm ready."], (
        f"English model input wrong: {results['eng_input']}"
    )
    # Spanish always receives neutral-joined model input
    assert results["spa_input"] == ["I 'm ready."], (
        f"Spanish model input wrong: {results['spa_input']}"
    )
    # Character offsets — both joined strings start with "I" at index 0
    eng_offset = results["eng_evidence"][0]["characterOffset"]
    assert eng_offset == 0, f"English offset {eng_offset}"
    spa_offset = results["spa_evidence"][0]["characterOffset"]
    assert spa_offset == 0, f"Spanish offset {spa_offset}"


def test_concurrent_isolation_reversed_order() -> None:
    """Reversed start order — Spanish then English — produces the same isolation."""
    import threading

    barrier = threading.Barrier(2)  # both threads meet at predict_proba
    eng_model = RecordingProbabilityModel()
    eng_model.barrier = barrier
    spa_model = RecordingProbabilityModel()
    spa_model.barrier = barrier

    results: dict[str, Any] = {}
    errors: list[str] = []

    def english_request() -> None:
        try:
            svc = SaTSentenceReconstructor(model=eng_model)
            evidence = svc.score_boundaries(
                ["I", "'m ready."],
                profile=EnglishBoundaryProfile(),
            )
            results["eng_input"] = list(eng_model.calls)
            results["eng_evidence"] = evidence
        except Exception as exc:
            errors.append(f"English: {exc!r}")

    def spanish_request() -> None:
        try:
            svc = SaTSentenceReconstructor(model=spa_model)
            evidence = svc.score_boundaries(
                ["I", "'m ready."],
                profile=NeutralBoundaryProfile(),
            )
            results["spa_input"] = list(spa_model.calls)
            results["spa_evidence"] = evidence
        except Exception as exc:
            errors.append(f"Spanish: {exc!r}")

    # Spanish starts first, then English
    t2 = threading.Thread(target=spanish_request)
    t1 = threading.Thread(target=english_request)

    t2.start()
    t1.start()

    t2.join(timeout=10)
    t1.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"
    assert "eng_input" in results, f"English didn't complete. results keys: {list(results)}"
    assert "spa_input" in results, f"Spanish didn't complete. results keys: {list(results)}"

    assert results["eng_input"] == ["I'm ready."], (
        f"English model input wrong: {results['eng_input']}"
    )
    assert results["spa_input"] == ["I 'm ready."], (
        f"Spanish model input wrong: {results['spa_input']}"
    )


def test_sequential_spanish_requests_produce_identical_output() -> None:
    """Spanish -> English -> Spanish: both Spanish responses must be identical."""

    svc = SaTSentenceReconstructor(model=RecordingProbabilityModel())

    # Spanish first
    spa1_evidence = svc.score_boundaries(
        ["I", "'m ready."],
        profile=NeutralBoundaryProfile(),
    )
    spa1_input = svc._model.calls[:]  # type: ignore[union-attr]
    svc._model.calls.clear()  # type: ignore[union-attr]

    # English
    eng_evidence = svc.score_boundaries(
        ["I", "'m ready."],
        profile=EnglishBoundaryProfile(),
    )
    eng_input = svc._model.calls[:]  # type: ignore[union-attr]
    svc._model.calls.clear()  # type: ignore[union-attr]

    # Spanish again
    spa2_evidence = svc.score_boundaries(
        ["I", "'m ready."],
        profile=NeutralBoundaryProfile(),
    )
    spa2_input = list(svc._model.calls)  # type: ignore[union-attr]

    assert spa1_input == ["I 'm ready."], (
        f"First Spanish input: {spa1_input}"
    )
    assert spa2_input == ["I 'm ready."], (
        f"Second Spanish input: {spa2_input}"
    )
    assert eng_input == ["I'm ready."], (
        f"English input: {eng_input}"
    )
    # Both Spanish evidence must be structurally identical
    assert len(spa1_evidence) == len(spa2_evidence)
    for e1, e2 in zip(spa1_evidence, spa2_evidence):
        assert e1["characterOffset"] == e2["characterOffset"]
        assert e1["boundaryProbability"] == e2["boundaryProbability"]


def test_shared_service_concurrency_isolation() -> None:
    """One shared SaTSentenceReconstructor + one shared model, simultaneous calls.

    Both English and neutral (Spanish) scoring run on the same service
    instance and same model. The barrier forces both predict_proba calls
    to overlap. Each must still receive the correctly profile-joined text
    and return the correct character offsets.
    """
    import threading

    barrier = threading.Barrier(2)
    shared_model = RecordingProbabilityModel()
    shared_model.barrier = barrier

    eng_evidence: list[dict] = []
    neu_evidence: list[dict] = []
    errors: list[str] = []

    svc = SaTSentenceReconstructor(model=shared_model)

    def english_request() -> None:
        try:
            evidence = svc.score_boundaries(
                ["I", "'m ready."],
                profile=EnglishBoundaryProfile(),
            )
            eng_evidence.extend(evidence)
        except Exception as exc:
            errors.append(f"English: {exc!r}")

    def neutral_request() -> None:
        try:
            evidence = svc.score_boundaries(
                ["I", "'m ready."],
                profile=NeutralBoundaryProfile(),
            )
            neu_evidence.extend(evidence)
        except Exception as exc:
            errors.append(f"Neutral: {exc!r}")

    t1 = threading.Thread(target=english_request)
    t2 = threading.Thread(target=neutral_request)

    t1.start()
    t2.start()

    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"

    # The shared model saw exactly two calls — one English-joined, one neutral-joined.
    assert "I'm ready." in shared_model.calls, (
        f"English-joined input missing from model calls: {shared_model.calls}"
    )
    assert "I 'm ready." in shared_model.calls, (
        f"Neutral-joined input missing from model calls: {shared_model.calls}"
    )
    assert len(shared_model.calls) == 2, (
        f"Expected exactly 2 model calls, got {len(shared_model.calls)}: {shared_model.calls}"
    )

    # Both got one boundary (between 'I' and "'m ready.")
    assert len(eng_evidence) == 1, f"English evidence count: {len(eng_evidence)}"
    assert len(neu_evidence) == 1, f"Neutral evidence count: {len(neu_evidence)}"

    # Both joined strings start at offset 0 for the first boundary
    eng_offset = eng_evidence[0]["characterOffset"]
    assert eng_offset == 0, f"English offset {eng_offset}"
    neu_offset = neu_evidence[0]["characterOffset"]
    assert neu_offset == 0, f"Neutral offset {neu_offset}"

    # Both boundaries should have a valid probability
    assert 0.0 <= eng_evidence[0]["boundaryProbability"] <= 1.0
    assert 0.0 <= neu_evidence[0]["boundaryProbability"] <= 1.0
