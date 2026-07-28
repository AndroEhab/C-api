from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_subtitle_service
from app.sat import join_segments
from app.subtitles import (
    BoundaryPolicyConfig,
    ReconstructedSentence,
    SubtitleSegment,
    SubtitleSentenceReconstructor,
    parse_srt,
)


@dataclass
class FakeBoundaryApi:
    probabilities: Any
    calls: list[list[str]]

    def score_boundaries(self, segments: Sequence[str]) -> Any:
        self.calls.append(list(segments))
        return self.probabilities


@dataclass
class FakeGroupingApi:
    groups: Any
    calls: list[list[Mapping[str, str]]]
    score_calls: list[list[str]] | None = None

    def __post_init__(self) -> None:
        if self.score_calls is None:
            self.score_calls = []

    def score_boundaries(self, segments: Sequence[str]) -> Any:
        assert self.score_calls is not None
        self.score_calls.append(list(segments))
        response = self.groups.get("groups") if isinstance(self.groups, Mapping) else self.groups
        if not isinstance(response, Sequence) or isinstance(response, (str, bytes)):
            return self.groups

        probabilities: list[float] = []
        source_count = 0
        for group in response:
            if isinstance(group, Mapping):
                members = group.get(
                    "segmentIndexes",
                    group.get("segment_indexes", group.get("segmentIds", group.get("segment_ids"))),
                )
            else:
                members = group
            if not isinstance(members, Sequence) or isinstance(members, (str, bytes)) or not members:
                return self.groups
            probabilities.extend([0.0] * (len(members) - 1))
            source_count += len(members)
        if source_count != len(segments):
            return self.groups
        return probabilities

    def group(self, segments: Sequence[Mapping[str, str]]) -> Any:
        self.calls.append(list(segments))
        return self.groups



def segment(segment_id: str, text: str, start: int, end: int, speaker: str | None = None) -> SubtitleSegment:
    return SubtitleSegment(segment_id, text, start, end, speaker)


def test_one_sentence_across_segments_preserves_parts_and_active_timing() -> None:
    api = FakeGroupingApi(
        {"groups": [{"segmentIds": ["cue-1", "cue-2", "cue-3"]}]},
        [],
    )
    cues = [
        segment("cue-1", "I never thought", 10_000, 11_200),
        segment("cue-2", "that we would end up", 11_200, 12_500),
        segment("cue-3", "living here.", 12_500, 13_800),
    ]

    timeline = SubtitleSentenceReconstructor(api).reconstruct_timeline(cues)

    assert timeline.sentences == [
        ReconstructedSentence(
            text="I never thought that we would end up living here.",
            start_ms=10_000,
            end_ms=13_800,
            parts=[
                # The exact cue text and timing remain available to the player.
                timeline.sentences[0].parts[0],
                timeline.sentences[0].parts[1],
                timeline.sentences[0].parts[2],
            ],
        )
    ]
    assert [part.segment_id for part in timeline.sentences[0].parts] == ["cue-1", "cue-2", "cue-3"]
    assert timeline.active_part_at(11_300).segment_id == "cue-2"
    assert timeline.active_sentence_at(12_600) is timeline.sentences[0]
    assert api.calls == []
    assert api.score_calls == [
        ["I never thought", "that we would end up", "living here."]
    ]
def test_cues_are_decided_only_at_original_boundaries() -> None:
    api = FakeBoundaryApi([0.9, 0.1], [])
    cues = [
        segment("143", "First. Second.", 1_109_040, 1_111_250),
        segment("144", "This case", 1_111_620, 1_112_620),
        segment("145", "could be a breakthrough.", 1_112_710, 1_114_790),
    ]

    timeline = SubtitleSentenceReconstructor(api).reconstruct_timeline(cues)

    assert [sentence.segment_ids for sentence in timeline.sentences] == [
        ["143"],
        ["144", "145"],
    ]
    reconstructed = timeline.sentences[1]
    assert reconstructed.text == "This case could be a breakthrough."
    assert reconstructed.start_ms == 1_111_620
    assert reconstructed.end_ms == 1_114_790
    assert [
        (part.segment_id, part.text, part.start_ms, part.end_ms)
        for part in reconstructed.parts
    ] == [
        ("144", "This case", 1_111_620, 1_112_620),
        ("145", "could be a breakthrough.", 1_112_710, 1_114_790),
    ]




def test_multiple_sentences_in_one_segment_falls_back_without_losing_the_cue() -> None:
    api = FakeGroupingApi({"groups": [["cue-1"], ["cue-1"]]}, [])
    cue = segment("cue-1", "Hello. Goodbye.", 100, 900)

    sentences = SubtitleSentenceReconstructor(api).reconstruct([cue])

    assert len(sentences) == 1
    assert sentences[0].text == cue.text
    assert sentences[0].segment_ids == ["cue-1"]
    assert sentences[0].parts[0].start_ms == 100
    assert sentences[0].parts[0].end_ms == 900


def test_missing_punctuation_still_reconstructs_exact_spoken_text() -> None:
    api = FakeGroupingApi({"groups": [["a", "b"]]}, [])
    cues = [segment("a", "We can go", 0, 500), segment("b", "when you are ready", 500, 1_000)]

    sentence = SubtitleSentenceReconstructor(api).reconstruct(cues)[0]

    assert sentence.text == "We can go when you are ready"
    assert sentence.parts[0].text == "We can go"
    assert sentence.parts[1].text == "when you are ready"


def test_different_speakers_are_never_grouped_together() -> None:
    api = FakeGroupingApi({"groups": [["a"], ["b"]]}, [])
    cues = [
        segment("a", "Hello.", 0, 500, "Alice"),
        segment("b", "Hello.", 500, 1_000, "Bob"),
    ]

    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)

    assert [sentence.segment_ids for sentence in sentences] == [["a"], ["b"]]
    assert api.calls == []
    assert api.score_calls == [["Hello.", "Hello."]]


@pytest.mark.parametrize(
    "response",
    [
        [],
        {"boundaries": [0.2, 0.3]},
        {"boundaries": [{"modelProbability": 2.0}]},
        {"notEvidence": []},
    ],
)
def test_malformed_boundary_evidence_falls_back_to_original_cues(response: Any) -> None:
    api = FakeBoundaryApi(response, [])
    cues = [segment("a", "First", 0, 100), segment("b", "Second", 100, 200)]

    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)

    assert [(sentence.text, sentence.segment_ids) for sentence in sentences] == [
        ("First", ["a"]),
        ("Second", ["b"]),
    ]


def test_grouping_api_failure_falls_back_to_original_cues() -> None:
    class FailingApi:
        def group(self, segments: Sequence[Mapping[str, str]]) -> Any:
            raise RuntimeError("controlled failure")

    cues = [segment("a", "First", 0, 100), segment("b", "Second", 100, 200)]

    sentences = SubtitleSentenceReconstructor(FailingApi()).reconstruct(cues)

    assert [sentence.segment_ids for sentence in sentences] == [["a"], ["b"]]


def test_srt_parser_assigns_stable_ids_and_millisecond_timings() -> None:
    cues = parse_srt(
        "1\n00:00:10,000 --> 00:00:11,200\nI never thought\n\n"
        "2\n00:00:11,200 --> 00:00:12,500\nthat we would end up\n"
    )

    assert [(cue.segment_id, cue.start_ms, cue.end_ms, cue.text) for cue in cues] == [
        ("1", 10_000, 11_200, "I never thought"),
        ("2", 11_200, 12_500, "that we would end up"),
    ]


def test_subtitle_endpoint_returns_original_cues_and_sentence_parts() -> None:
    api = FakeGroupingApi({"groups": [["a", "b"]]}, [])
    cues = [
        segment("a", "First part", 0, 100),
        segment("b", "second part.", 100, 200),
    ]
    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)
    body = {
        "segments": cues,
        "sentences": [sentence.to_dict() for sentence in sentences],
    }
    assert [cue.segment_id for cue in body["segments"]] == ["a", "b"]
    sentence = body["sentences"][0]
    assert sentence["text"] == "First part second part."
    assert sentence["startMs"] == 0
    assert sentence["endMs"] == 200
    assert sentence["segmentIds"] == ["a", "b"]
    assert len(sentence["parts"]) == 2
    assert sentence["parts"][0]["segmentId"] == "a"
    assert sentence["parts"][0]["text"] == "First part"
    assert sentence["parts"][0]["startMs"] == 0
    assert sentence["parts"][0]["endMs"] == 100
    assert sentence["parts"][1]["segmentId"] == "b"
    assert sentence["parts"][1]["text"] == "second part."
    assert sentence["parts"][1]["startMs"] == 100
    assert sentence["parts"][1]["endMs"] == 200
    # New fields should be present with default values
    assert sentence["parts"][0]["rawText"] is None
    assert sentence["parts"][0]["lines"] is None
    assert sentence["parts"][0]["speaker"] is None
    assert sentence["parts"][0]["speakerMarkers"] is None
    assert sentence["parts"][0]["containsMultipleSpeakers"] is None


def test_explicit_indexes_merge_clear_two_cue_continuation() -> None:
    api = FakeGroupingApi(
        {
            "groups": [
                {
                    "segmentIndexes": [0, 1],
                    "text": "This case could be a breakthrough.",
                }
            ]
        },
        [],
    )
    cues = [
        segment("144", "This case", 1_111_620, 1_112_620),
        segment("145", "could be a breakthrough.", 1_112_710, 1_114_790),
    ]

    sentence = SubtitleSentenceReconstructor(api).reconstruct(cues)[0]

    assert sentence.text == "This case could be a breakthrough."
    assert sentence.segment_ids == ["144", "145"]


def test_model_breaks_are_not_coalesced_into_other_cues() -> None:
    api = FakeGroupingApi({"groups": [["168"], ["169"], ["170"]]}, [])
    cues = [
        segment("168", "Police suspect a professional", 0, 1_000),
        segment("169", "assassin organization", 1_020, 2_000),
        segment("170", "is behind these murders.", 2_020, 3_000),
    ]

    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)

    assert [sentence.segment_ids for sentence in sentences] == [
        ["168"],
        ["169"],
        ["170"],
    ]


def test_merge_across_thirty_seven_second_gap_is_rejected() -> None:
    api = FakeGroupingApi({"groups": [["a", "b"]]}, [])
    cues = [
        segment("a", "This case", 0, 1_000),
        segment("b", "could be a breakthrough.", 38_000, 39_000),
    ]

    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)

    assert [sentence.segment_ids for sentence in sentences] == [["a"], ["b"]]


def test_two_independently_complete_statements_are_rejected() -> None:
    api = FakeGroupingApi({"groups": [["101", "102"]]}, [])
    cues = [
        segment("101", "The one who escaped is Poison Rat.", 0, 1_000),
        segment("102", "I think we can issue a warrant immediately.", 1_020, 2_000),
    ]

    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)

    assert [sentence.segment_ids for sentence in sentences] == [["101"], ["102"]]


def test_question_followed_by_another_speakers_answer_is_rejected() -> None:
    api = FakeGroupingApi({"groups": [["540", "541"]]}, [])
    cues = [
        segment("540", "Where is Mingzhu?", 0, 1_000),
        segment("541", "I'll take you to her.", 1_100, 2_000),
    ]

    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)

    assert [sentence.segment_ids for sentence in sentences] == [["540"], ["541"]]


def test_grouping_preserves_every_source_part_timing() -> None:
    api = FakeGroupingApi({"groups": [["a", "b", "c"]]}, [])
    cues = [
        segment("a", "Police suspect a professional", 100, 900),
        segment("b", "assassin organization", 950, 1_600),
        segment("c", "is behind these murders.", 1_650, 2_700),
    ]

    sentence = SubtitleSentenceReconstructor(api).reconstruct(cues)[0]

    assert sentence.start_ms == 100
    assert sentence.end_ms == 2_700
    assert [
        (part.segment_id, part.text, part.start_ms, part.end_ms)
        for part in sentence.parts
    ] == [
        ("a", "Police suspect a professional", 100, 900),
        ("b", "assassin organization", 950, 1_600),
        ("c", "is behind these murders.", 1_650, 2_700),
    ]


def test_apostrophe_leading_fragment_never_corrupts_preceding_word() -> None:
    api = FakeGroupingApi({"groups": [["541", "543"]]}, [])
    cues = [
        segment("541", "I'll take you to her", 0, 1_000),
        segment("543", "'ve taken good care of your sister.", 1_100, 2_000),
    ]

    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)

    assert join_segments(["I", "'ve taken care."]) == "I've taken care."
    assert join_segments([cues[0].text, cues[1].text]) == (
        "I'll take you to her 've taken good care of your sister."
    )
    assert "her've" not in " ".join(sentence.text for sentence in sentences)
    assert [sentence.segment_ids for sentence in sentences] == [["541"], ["543"]]


def test_grouping_preserves_original_cue_objects_text_and_ids() -> None:
    api = FakeGroupingApi({"groups": [["a", "b"]]}, [])
    cues = [
        segment("a", "<i>This case", 10, 100),
        segment("b", "could be a breakthrough.</i>", 120, 300),
    ]
    original = list(cues)

    sentence = SubtitleSentenceReconstructor(api).reconstruct(cues)[0]

    assert cues == original
    assert [(part.segment_id, part.text) for part in sentence.parts] == [
        ("a", "<i>This case"),
        ("b", "could be a breakthrough.</i>"),
    ]
    assert sentence.text == "<i>This case could be a breakthrough.</i>"


@pytest.mark.parametrize(
    "response",
    [
        {"sentences": ["rewritten model output"]},
        {"full_sentence": "First Second", "is_single_sentence": False},
        {"groups": [{"segmentIndexes": [0], "segmentIds": ["b"]}, ["b"]]},
    ],
)
def test_invalid_or_ambiguous_api_response_keeps_original_cues(response: Any) -> None:
    api = FakeGroupingApi(response, [])
    cues = [segment("a", "First", 0, 100), segment("b", "Second", 120, 200)]

    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)

    assert [(sentence.text, sentence.segment_ids) for sentence in sentences] == [
        ("First", ["a"]),
        ("Second", ["b"]),
    ]


def test_api_timeout_keeps_original_cues() -> None:
    class TimeoutApi:
        def group(self, segments: Sequence[Mapping[str, str]]) -> Any:
            raise TimeoutError("controlled timeout")

    cues = [segment("a", "First", 0, 100), segment("b", "Second", 120, 200)]

    sentences = SubtitleSentenceReconstructor(TimeoutApi()).reconstruct(cues)

    assert [sentence.segment_ids for sentence in sentences] == [["a"], ["b"]]


def test_model_break_is_not_overridden_by_lexical_continuation() -> None:
    api = FakeBoundaryApi([0.9], [])
    cues = [
        segment("a", "I told you", 0, 500),
        segment("b", "why this matters.", 520, 1_200),
    ]

    decisions = SubtitleSentenceReconstructor(api).evaluate_boundaries(cues)

    assert decisions[0]["decision"] == "break"
    assert decisions[0]["modelProbability"] == pytest.approx(0.9)
@pytest.mark.parametrize(
    "texts",
    [
        ["I'll tell you", "who your sister is."],
        ["Better to die quick.", "Than live in pain."],
        [
            "They're also suspected of.",
            "Colluding with local criminal gangs",
            "to carry out targeted assassinations.",
        ],
    ],
)
def test_separate_model_breaks_remain_separate(texts: list[str]) -> None:
    ids = [str(index) for index in range(len(texts))]
    api = FakeGroupingApi({"groups": [[segment_id] for segment_id in ids]}, [])
    cues = [
        segment(segment_id, text, index * 1_000, index * 1_000 + 900)
        for index, (segment_id, text) in enumerate(zip(ids, texts))
    ]

    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)

    assert [sentence.segment_ids for sentence in sentences] == [[segment_id] for segment_id in ids]
    assert [sentence.text for sentence in sentences] == texts




@pytest.mark.parametrize(
    "second_text",
    ["- I'll go.", "<v Mingzhu>I'll go.", "MINGZHU: I'll go."],
)
def test_dialogue_markers_protect_probable_speaker_boundary(second_text: str) -> None:
    api = FakeGroupingApi({"groups": [["a", "b"]]}, [])
    cues = [
        segment("a", "Wait here.", 0, 500),
        segment("b", second_text, 600, 1_100),
    ]

    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)

    assert [sentence.segment_ids for sentence in sentences] == [["a"], ["b"]]
    assert sentences[1].text == second_text


def test_configurable_gap_threshold_is_applied() -> None:
    api = FakeGroupingApi({"groups": [["a", "b"]]}, [])
    cues = [
        segment("a", "This case", 0, 1_000),
        segment("b", "could be a breakthrough.", 2_600, 3_000),
    ]

    default_groups = SubtitleSentenceReconstructor(api).reconstruct(cues)
    configured_groups = SubtitleSentenceReconstructor(
        api,
        max_gap_ms=2_000,
    ).reconstruct(cues)

    assert [sentence.segment_ids for sentence in default_groups] == [["a"], ["b"]]
    assert [sentence.segment_ids for sentence in configured_groups] == [["a", "b"]]


def test_boundary_policy_config_uses_calibrated_gap_bands() -> None:
    config = BoundaryPolicyConfig(
        normal_gap_max_ms=100,
        medium_gap_max_ms=500,
        extreme_gap_ms=2_000,
    )

    def decide(gap_ms: int, probability: float) -> Mapping[str, Any]:
        api = FakeBoundaryApi([probability], [])
        cues = [
            segment("a", "This case", 0, 1_000),
            segment("b", "could be a breakthrough.", 1_000 + gap_ms, 2_000 + gap_ms),
        ]
        return SubtitleSentenceReconstructor(api, policy_config=config).evaluate_boundaries(cues)[0]

    assert decide(100, 0.49)["decision"] == "join"
    assert decide(500, 0.25)["decision"] == "break"
    assert decide(500, 0.10)["decision"] == "join"
    assert decide(1_000, 0.0)["reason"] == "break by default for large cue gap 1000ms"
    assert decide(2_001, 0.0)["reason"] == (
        "cue gap 2001ms exceeds extreme-gap threshold 2000ms"
    )


@pytest.mark.parametrize(
    ("left", "right", "speaker", "expected_reason"),
    [
        ("Hello.", "Hi.", ("Alice", "Bob"), "different explicit speakers"),
        ("Wait here.", "- I'll go.", (None, None), "next cue has a dialogue marker"),
        ("her", "'ve taken care.", (None, None), "joining would corrupt text"),
    ],
)
def test_hard_boundary_constraints_always_break(
    left: str,
    right: str,
    speaker: tuple[str | None, str | None],
    expected_reason: str,
) -> None:
    api = FakeBoundaryApi([0.0], [])
    cues = [
        segment("a", left, 0, 500, speaker[0]),
        segment("b", right, 500, 1_000, speaker[1]),
    ]

    decision = SubtitleSentenceReconstructor(api).evaluate_boundaries(cues)[0]

    assert decision["decision"] == "break"
    assert expected_reason in decision["reason"]


@pytest.mark.parametrize(
    ("left_id", "right_id", "expected_reason"),
    [
        ("2", "1", "cue order is invalid"),
        ("10", "12", "source cues are not consecutive"),
    ],
)
def test_invalid_source_adjacency_is_a_hard_break(
    left_id: str,
    right_id: str,
    expected_reason: str,
) -> None:
    api = FakeBoundaryApi([0.0], [])
    cues = [
        segment(left_id, "This case", 0, 500),
        segment(right_id, "could continue.", 500, 1_000),
    ]

    decision = SubtitleSentenceReconstructor(api).evaluate_boundaries(cues)[0]

    assert decision["decision"] == "break"
    assert decision["reason"] == expected_reason


def test_boundary_evidence_that_changes_cue_order_is_a_hard_break() -> None:
    api = FakeBoundaryApi(
        [{"leftSegmentId": "b", "rightSegmentId": "a", "boundaryProbability": 0.0}],
        [],
    )
    cues = [
        segment("a", "This case", 0, 500),
        segment("b", "could continue.", 500, 1_000),
    ]

    decision = SubtitleSentenceReconstructor(api).evaluate_boundaries(cues)[0]

    assert decision["decision"] == "break"
    assert decision["reason"].startswith("cue order is invalid:")


@pytest.mark.parametrize(
    ("left", "right", "probability", "expected"),
    [
        ("Better to die quick.", "Than live in pain.", 0.90, "join"),
        ("Fame is a good thing.", "Why refuse.", 0.10, "break"),
        ("Better to die quick", "I love ice cream.", 0.10, "break"),
    ],
)
def test_punctuation_and_capitalization_are_advisory_soft_evidence(
    left: str,
    right: str,
    probability: float,
    expected: str,
) -> None:
    api = FakeBoundaryApi([probability], [])
    cues = [segment("a", left, 0, 500), segment("b", right, 500, 1_000)]

    decision = SubtitleSentenceReconstructor(api).evaluate_boundaries(cues)[0]

    assert decision["decision"] == expected


def test_rejected_boundary_does_not_destroy_valid_joins(caplog: pytest.LogCaptureFixture) -> None:
    api = FakeBoundaryApi([0.1, 0.9, 0.1], [])
    cues = [
        segment("a", "A", 0, 500),
        segment("b", "B", 520, 1_000),
        segment("c", "C", 1_020, 1_500),
        segment("d", "D", 1_520, 2_000),
    ]

    reconstructor = SubtitleSentenceReconstructor(api, debug=True)
    with caplog.at_level("DEBUG", logger="app.subtitles"):
        decisions = reconstructor.evaluate_boundaries(cues)
        sentences = reconstructor.build_sentences(cues, decisions)

    assert len(decisions) == 3
    assert [
        (item["leftSegmentId"], item["rightSegmentId"], item["decision"], item["gapMs"])
        for item in decisions
    ] == [
        ("a", "b", "join", 20),
        ("b", "c", "break", 20),
        ("c", "d", "join", 20),
    ]

    assert [sentence.segment_ids for sentence in sentences] == [["a", "b"], ["c", "d"]]
    assert "model probability favors a sentence boundary" in caplog.text


def test_sentence_construction_treats_uncertain_as_break() -> None:
    reconstructor = SubtitleSentenceReconstructor(FakeBoundaryApi([], []))
    cues = [
        segment("a", "A", 0, 500),
        segment("b", "B", 520, 1_000),
        segment("c", "C", 1_020, 1_500),
        segment("d", "D", 1_520, 2_000),
    ]
    decisions = [
        {
            "leftSegmentId": "a",
            "rightSegmentId": "b",
            "decision": "join",
            "reason": "test join",
            "modelProbability": 0.1,
            "gapMs": 20,
        },
        {
            "leftSegmentId": "b",
            "rightSegmentId": "c",
            "decision": "uncertain",
            "reason": "test uncertainty",
            "modelProbability": None,
            "gapMs": 20,
        },
        {
            "leftSegmentId": "c",
            "rightSegmentId": "d",
            "decision": "join",
            "reason": "test join",
            "modelProbability": 0.1,
            "gapMs": 20,
        },
    ]

    sentences = reconstructor.build_sentences(cues, decisions)

    assert [sentence.segment_ids for sentence in sentences] == [["a", "b"], ["c", "d"]]


# ---- Task 5: Subtitle structural preservation tests ----

def test_multiline_srt_preserves_lines_and_raw_text() -> None:
    """Multiline SRT text is preserved in lines and raw_text."""
    cues = parse_srt(
        "1\n00:00:10,000 --> 00:00:12,500\nWhere are you going?\nHome.\n\n"
        "2\n00:00:13,000 --> 00:00:14,000\nSingle line."
    )
    assert len(cues) == 2
    # First cue had two text lines
    assert cues[0].lines == ("Where are you going?", "Home.")
    assert cues[0].raw_text == "Where are you going?\nHome."
    assert cues[0].text == "Where are you going? Home."
    # Second cue had single line
    assert cues[1].lines == ("Single line.",)
    assert cues[1].raw_text == "Single line."
    assert cues[1].text == "Single line."


def test_existing_text_field_remains_backward_compatible() -> None:
    """Existing text field still works as before."""
    s = segment("a", "Hello world", 0, 1000)
    assert s.text == "Hello world"
    assert s.raw_text is None
    assert s.lines == ()
    assert s.speaker is None


def test_two_dash_lines_detected_as_multiple_speakers() -> None:
    """Two dash-prefixed dialogue lines are detected as multiple speakers."""
    s = segment("a", "- Where are you going? - Home.", 0, 1000)
    assert s.contains_multiple_speakers is False  # segment() doesn't set it

    # Via parse_srt
    cues = parse_srt(
        "1\n00:00:10,000 --> 00:00:12,500\n- Where are you going?\n- Home.\n"
    )
    assert cues[0].contains_multiple_speakers is True
    assert "-" in cues[0].speaker_markers
    assert cues[0].speaker is None  # No named speaker


def test_single_dash_not_assigned_named_speaker() -> None:
    """A single dash-prefixed line is not automatically assigned a named speaker."""
    cues = parse_srt("1\n00:00:10,000 --> 00:00:11,000\n- I'll go.\n")
    assert cues[0].contains_multiple_speakers is False
    assert "-" in cues[0].speaker_markers
    assert cues[0].speaker is None


def test_webvtt_voice_tag_extracts_speaker() -> None:
    """WebVTT <v Alice> tags extract Alice as the speaker."""
    cues = parse_srt("1\n00:00:10,000 --> 00:00:11,000\n<v Alice>Hello there.\n")
    assert cues[0].speaker == "Alice"
    assert "Alice" in cues[0].speaker_markers


def test_two_different_voice_tags_cause_protected_boundary() -> None:
    """Two different voice tags cause a hard break between adjacent cues."""
    left = segment("a", "<v Alice>Hello", 0, 500, speaker="Alice")
    right = segment("b", "<v Bob>Hi", 500, 1000, speaker="Bob")
    api = FakeBoundaryApi([0.0], [])
    decisions = SubtitleSentenceReconstructor(api).evaluate_boundaries([left, right])
    assert decisions[0]["decision"] == "break"
    assert "different explicit speakers" in decisions[0]["reason"]


def test_colon_labels_detected_conservatively() -> None:
    """ALICE: and BOB: labels are detected but do not trigger false positives."""
    cues = parse_srt(
        "1\n00:00:10,000 --> 00:00:11,000\nALICE: Hello\n\n"
        "2\n00:00:11,500 --> 00:00:12,500\nBOB: Hi there.\n"
    )
    assert cues[0].speaker == "ALICE"
    assert "ALICE" in cues[0].speaker_markers
    assert cues[1].speaker == "BOB"
    assert "BOB" in cues[1].speaker_markers


def test_formatting_tags_are_preserved() -> None:
    """Formatting tags remain preserved in the text and raw_text."""
    cues = parse_srt(
        "1\n00:00:10,000 --> 00:00:12,000\n<i>This is italic</i>\n\n"
        "2\n00:00:13,000 --> 00:00:14,000\nPlain text.\n"
    )
    assert "<i>This is italic</i>" in cues[0].text
    assert cues[0].lines[0] == "<i>This is italic</i>"


def test_multiple_speakers_in_one_cue_not_split() -> None:
    """Multiple speakers inside one timed cue do not cause it to be split."""
    cues = parse_srt(
        "1\n00:00:10,000 --> 00:00:12,500\nALICE: Hello\nBOB: Hi.\n"
    )
    assert len(cues) == 1
    assert cues[0].contains_multiple_speakers is True
    assert cues[0].speaker is None  # Two named speakers, so ambiguous overall
    assert len(cues[0].lines) == 2


def test_different_explicit_speakers_cannot_be_joined() -> None:
    """Adjacent cues with different explicit speakers cannot be joined."""
    left = segment("a", "Hello", 0, 500, speaker="Alice")
    right = segment("b", "Hi", 500, 1000, speaker="Bob")
    api = FakeBoundaryApi([0.0], [])
    decisions = SubtitleSentenceReconstructor(api).evaluate_boundaries([left, right])
    assert decisions[0]["decision"] == "break"
    assert "different explicit speakers" in decisions[0]["reason"]

    # Even with 0.0 model probability (favors join), still breaks
    api2 = FakeBoundaryApi([0.0], [])
    decisions2 = SubtitleSentenceReconstructor(api2).evaluate_boundaries([left, right])
    assert decisions2[0]["decision"] == "break"


def test_same_speaker_evaluated_by_normal_policy() -> None:
    """Same explicit speaker may still be evaluated by the normal boundary policy."""
    api = FakeBoundaryApi([0.9], [])
    left = segment("a", "Hello.", 0, 500, speaker="Alice")
    right = segment("b", "How are you?", 500, 1000, speaker="Alice")
    decisions = SubtitleSentenceReconstructor(api).evaluate_boundaries([left, right])
    # Same speaker, model says join (0.9) -> should not be a hard break
    assert decisions[0]["decision"] != "break" or "different explicit speakers" not in decisions[0]["reason"]


def test_unmarked_subtitles_through_sat_path() -> None:
    """Unmarked subtitles continue through the existing SaT windowed path."""
    api = FakeBoundaryApi([0.6], [])
    cues = [
        segment("a", "The quick brown fox", 0, 1000),
        segment("b", "jumps over the lazy dog.", 1000, 2000),
    ]
    decisions = SubtitleSentenceReconstructor(api).evaluate_boundaries(cues)
    # No speaker info, normal soft boundary
    assert decisions[0]["decision"] in ("join", "break", "uncertain")


def test_api_request_with_only_old_fields_is_valid() -> None:
    """API requests containing only the old fields remain valid."""
    from app.main import SubtitleSegmentRequest
    req = SubtitleSegmentRequest(
        segmentId="test",
        text="Hello world",
        startMs=0,
        endMs=1000,
    )
    domain = req.to_domain()
    assert domain.segment_id == "test"
    assert domain.text == "Hello world"
    assert domain.start_ms == 0
    assert domain.end_ms == 1000
    assert domain.speaker is None
    assert domain.raw_text is None
    assert domain.lines == ()


def test_api_response_preserves_source_lines() -> None:
    """API responses preserve source lines without losing or rewriting text."""
    api = FakeGroupingApi({"groups": [["a"]]}, [])
    cues = [
        segment("a", "- Hello.\n- Hi.", 0, 1000),
    ]
    sentences = SubtitleSentenceReconstructor(api).reconstruct(cues)
    parts = sentences[0].parts
    assert len(parts) == 1
    assert parts[0].text == "- Hello.\n- Hi."
    # to_dict preserves the text
    out = sentences[0].to_dict()
    assert out["parts"][0]["text"] == "- Hello.\n- Hi."


def test_boundary_protected_by_structural_speaker_evidence() -> None:
    """A boundary that would otherwise be joined is protected by speaker evidence."""
    api = FakeBoundaryApi([0.0], [])
    left = segment("a", "Hello.", 0, 500)
    # Right cue has a voice tag -> protected boundary
    right = segment("b", "<v Alice>Hi!", 500, 1000, speaker="Alice")
    decisions = SubtitleSentenceReconstructor(api).evaluate_boundaries([left, right])
    assert decisions[0]["decision"] == "break"
    assert "explicit speaker" in decisions[0]["reason"]


def test_both_dialogue_markers_protect_boundary() -> None:
    """Both cues having dialogue markers protects the boundary even with no named speaker."""
    api = FakeBoundaryApi([0.0], [])
    left = segment("a", "- Hello.", 0, 500)
    right = segment("b", "- Hi!", 500, 1000)
    decisions = SubtitleSentenceReconstructor(api).evaluate_boundaries([left, right])
    assert decisions[0]["decision"] == "break"
    assert "both cues have dialogue markers" in decisions[0]["reason"]
