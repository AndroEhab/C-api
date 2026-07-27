from __future__ import annotations

from typing import Sequence

from app.subtitles import SubtitleSentenceReconstructor
from scripts.evaluate_subtitle_boundaries import evaluate_fixture, load_fixture


class ConstantBoundaryApi:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def score_boundaries(self, segments: Sequence[str]) -> list[dict[str, float]]:
        return [
            {"boundaryProbability": self.probability}
            for _ in range(max(0, len(segments) - 1))
        ]


def test_boundary_fixture_contains_the_identified_cases() -> None:
    cases = load_fixture()

    assert len(cases) == 71
    assert [case["expected"] for case in cases].count("join") == 31
    assert [case["expected"] for case in cases].count("break") == 38
    assert [case["expected"] for case in cases].count("ambiguous") == 2
    assert cases[0]["left"] == "This case"
    assert cases[-1]["right"] == "you to end up like your father."
    reviewed = cases[11:61]
    assert len(reviewed) == 50
    assert all(
        {"leftCueId", "rightCueId", "reviewCategory"} <= set(case)
        for case in reviewed
    )
    assert len({(case["leftCueId"], case["rightCueId"]) for case in reviewed}) == 50
    assert "grammatically_compatible_semantically_unrelated" in {
        case["reviewCategory"] for case in reviewed
    }
    # Regression cases (indices 61-70)
    regression = cases[61:]
    assert len(regression) == 10
    assert all({"left", "right", "expected"} <= set(case) for case in regression)
    # BREAK regressions have cue IDs for full-file evaluation
    break_regs = regression[:5]
    join_regs = regression[5:]
    assert all(case.get("leftCueId") and case.get("rightCueId") for case in break_regs)
    assert all(case["expected"] == "break" for case in break_regs)
    assert all(case["expected"] == "join" for case in join_regs)
    assert break_regs[0]["left"] == "I wanted to surprise you"
    assert join_regs[0]["left"] == "Police suspect a professional"


def test_fixture_report_keeps_model_probability_on_misclassified_boundaries() -> None:
    cases = [
        {
            "left": "unfinished",
            "right": "continuation",
            "previousContext": [],
            "nextContext": [],
            "gapMs": 0,
            "expected": "join",
        },
        {
            "left": "one",
            "right": "two",
            "previousContext": [],
            "nextContext": [],
            "gapMs": 0,
            "expected": "break",
        },
    ]
    report = evaluate_fixture(
        cases,
        SubtitleSentenceReconstructor(ConstantBoundaryApi(0.1)),
    )

    assert report["metrics"]["totalBoundaries"] == 2
    assert report["metrics"]["joinPrecision"] == 0.5
    assert report["metrics"]["joinRecall"] == 1.0
    assert report["metrics"]["breakAccuracy"] == 0.0
    assert report["metrics"]["falseMergeCount"] == 1
    assert report["metrics"]["falseNegativeCount"] == 0
    assert report["misclassified"] == [
        {
            "case": 2,
            "left": "one",
            "right": "two",
            "gapMs": 0,
            "expected": "break",
            "predicted": "join",
            "decision": "join",
            "modelProbability": 0.1,
            "reason": "model probability favors continuation",
            "correct": False,
        }
    ]
