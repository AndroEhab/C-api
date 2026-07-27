from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import httpx
import pytest

from app.model import DEFAULT_THRESHOLD, MODEL_NAME


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ACCURACY_BENCHMARK") != "1",
    reason="set RUN_ACCURACY_BENCHMARK=1 to run the live accuracy benchmark",
)

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    word: str
    sentence1: str
    sentence2: str
    same_meaning: bool


# 52 hand-labeled cases: 26 same-sense pairs and 26 deliberately different-sense
# pairs covering inflection, physical objects, actions, institutions, and polysemy.
ACCURACY_CASES = [
    AccuracyCase("book_same", "book", "I read many books.", "The book was fascinating.", True),
    AccuracyCase("book_different", "book", "I read a book.", "Please book a room.", False),
    AccuracyCase("bank_finance_same", "bank", "I deposited cash at the bank.", "The bank approved my mortgage.", True),
    AccuracyCase("bank_different", "bank", "I deposited cash at the bank.", "The hikers rested on the bank of the river.", False),
    AccuracyCase("bat_animal_same", "bat", "A bat flew from the cave.", "Bats hunt insects at dusk.", True),
    AccuracyCase("bat_different", "bat", "A bat flew from the cave.", "He swung the bat at the ball.", False),
    AccuracyCase("bark_tree_same", "bark", "The tree's bark was rough.", "We peeled bark from the trunk.", True),
    AccuracyCase("bark_different", "bark", "The tree's bark was rough.", "The dog's bark woke me.", False),
    AccuracyCase("crane_machine_same", "crane", "The crane lifted steel beams.", "Workers operated the crane at the site.", True),
    AccuracyCase("crane_different", "crane", "The crane lifted steel beams.", "A crane stood beside the wetland.", False),
    AccuracyCase("seal_animal_same", "seal", "A seal surfaced near the boat.", "The seal swam beside the aquarium glass.", True),
    AccuracyCase("seal_different", "seal", "A seal surfaced near the boat.", "Please seal the envelope.", False),
    AccuracyCase("spring_season_same", "spring", "Flowers bloom in spring.", "Spring arrives after winter.", True),
    AccuracyCase("spring_different", "spring", "Flowers bloom in spring.", "The metal spring was compressed.", False),
    AccuracyCase("light_illumination_same", "light", "Turn on the light in the hallway.", "The lamp provided light.", True),
    AccuracyCase("light_different", "light", "Turn on the light in the hallway.", "The light suitcase was easy to carry.", False),
    AccuracyCase("match_contest_same", "match", "Our team won the match.", "The match ended in a draw.", True),
    AccuracyCase("match_different", "match", "Our team won the match.", "The match ignited the candle.", False),
    AccuracyCase("paper_material_same", "paper", "She folded the paper.", "The paper tore easily.", True),
    AccuracyCase("paper_different", "paper", "She folded the paper into a boat.", "The paper analyzed voting patterns.", False),
    AccuracyCase("file_computer_same", "file", "Save the report as a file.", "Open the file from the folder.", True),
    AccuracyCase("file_different", "file", "Save the report as a file.", "Use a file to smooth the edge.", False),
    AccuracyCase("watch_timepiece_same", "watch", "My watch stopped.", "The watch needs a new battery.", True),
    AccuracyCase("watch_different", "watch", "My watch stopped.", "We watch birds from the hide.", False),
    AccuracyCase("key_lock_same", "key", "The key opened the door.", "I lost the key to the apartment.", True),
    AccuracyCase("key_different", "key", "The key opened the door.", "The key to the puzzle was patience.", False),
    AccuracyCase("charge_payment_same", "charge", "The store will charge my card.", "A charge appeared on my statement.", True),
    AccuracyCase("charge_different", "charge", "The store will charge my card.", "The police will charge the suspect.", False),
    AccuracyCase("date_calendar_same", "date", "Enter the date on the form.", "The date is printed at the top.", True),
    AccuracyCase("date_different", "date", "Enter the date on the form.", "We ate sweet dates after dinner.", False),
    AccuracyCase("rock_stone_same", "rock", "A rock blocked the path.", "She skipped a rock across the pond.", True),
    AccuracyCase("rock_different", "rock", "A rock blocked the path.", "The band played rock music.", False),
    AccuracyCase("wave_ocean_same", "wave", "A large wave hit the shore.", "Surfers waited for a wave.", True),
    AccuracyCase("wave_different", "wave", "A large wave hit the shore.", "She used a wave to say goodbye.", False),
    AccuracyCase("current_electric_same", "current", "The current flows through the wire.", "Measure the current in the circuit.", True),
    AccuracyCase("current_different", "current", "The current flows through the wire.", "Current events dominated the news.", False),
    AccuracyCase("jam_traffic_same", "jam", "Traffic was stuck in a jam.", "A traffic jam blocked the highway.", True),
    AccuracyCase("jam_different", "jam", "Traffic was stuck in a jam.", "She spread berry jam on toast.", False),
    AccuracyCase("plane_aircraft_same", "plane", "The plane landed safely.", "The plane crossed the ocean.", True),
    AccuracyCase("plane_different", "plane", "The plane landed safely.", "A plane is a flat geometric surface.", False),
    AccuracyCase("trunk_car_same", "trunk", "The luggage is in the trunk.", "Open the trunk of the car.", True),
    AccuracyCase("trunk_different", "trunk", "The luggage is in the trunk.", "The elephant lifted its trunk.", False),
    AccuracyCase("club_organization_same", "club", "She joined the chess club.", "The club elected a president.", True),
    AccuracyCase("club_different", "club", "She joined the chess club.", "He swung the club at the ball.", False),
    AccuracyCase("case_legal_same", "case", "The lawyer argued the case.", "The court dismissed the case.", True),
    AccuracyCase("case_different", "case", "The lawyer argued the case.", "The phone case was red.", False),
    AccuracyCase("kind_type_same", "kind", "What kind of plant is this?", "This kind of fabric is durable.", True),
    AccuracyCase("kind_different", "kind", "What kind of plant is this?", "She was kind to the visitor.", False),
    AccuracyCase("scale_weighing_same", "scale", "The scale showed my weight.", "Put the package on the scale.", True),
    AccuracyCase("scale_different", "scale", "The scale showed my weight.", "Fish have scales on their bodies.", False),
    AccuracyCase("run_action_same", "run", "She runs every morning.", "He ran five miles yesterday.", True),
    AccuracyCase("run_different", "run", "She runs every morning.", "The program runs quickly.", False),
]


def _classification_metrics(labels: Iterable[bool], predictions: Iterable[bool]) -> dict[str, float | int]:
    actual = list(labels)
    predicted = list(predictions)
    true_positive = sum(expected and result for expected, result in zip(actual, predicted))
    true_negative = sum(not expected and not result for expected, result in zip(actual, predicted))
    false_positive = sum(not expected and result for expected, result in zip(actual, predicted))
    false_negative = sum(expected and not result for expected, result in zip(actual, predicted))
    total = len(actual)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": total,
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _roc_auc(labels: list[bool], scores: list[float]) -> float:
    positive_scores = [score for label, score in zip(labels, scores) if label]
    negative_scores = [score for label, score in zip(labels, scores) if not label]
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positive_scores
        for negative in negative_scores
    )
    denominator = len(positive_scores) * len(negative_scores)
    return wins / denominator if denominator else 0.0


def _best_observed_threshold(labels: list[bool], scores: list[float]) -> tuple[float, dict[str, float | int]]:
    candidates = sorted({DEFAULT_THRESHOLD, *scores})
    scored = [
        (
            threshold,
            _classification_metrics(labels, [score >= threshold for score in scores]),
        )
        for threshold in candidates
    ]
    return max(scored, key=lambda item: (item[1]["accuracy"], item[1]["f1"]))


def test_live_accuracy_benchmark() -> None:
    assert len(ACCURACY_CASES) == 52
    assert sum(case.same_meaning for case in ACCURACY_CASES) == 26
    assert sum(not case.same_meaning for case in ACCURACY_CASES) == 26

    labels: list[bool] = []
    predictions: list[bool] = []
    scores: list[float] = []
    failures: list[str] = []

    with httpx.Client(base_url=BASE_URL, timeout=180.0) as client:
        for case in ACCURACY_CASES:
            response = client.post(
                "/similarity",
                json={
                    "word": case.word,
                    "sentence1": case.sentence1,
                    "sentence2": case.sentence2,
                },
            )
            assert response.status_code == 200, f"{case.name}: {response.text}"
            body = response.json()
            assert body["model"] == MODEL_NAME
            assert body["occurrences"]["sentence1"]
            assert body["occurrences"]["sentence2"]
            assert -1.0 <= body["similarity"] <= 1.0

            score = float(body["similarity"])
            prediction = bool(body["is_similar"])
            labels.append(case.same_meaning)
            predictions.append(prediction)
            scores.append(score)
            if prediction != case.same_meaning:
                failures.append(f"{case.name}: expected={case.same_meaning} score={score:.4f}")

    metrics = _classification_metrics(labels, predictions)
    auc = _roc_auc(labels, scores)
    best_threshold, best_metrics = _best_observed_threshold(labels, scores)

    print("\nSemantic accuracy benchmark")
    print(f"cases={metrics['count']} positives=26 negatives=26")
    print(
        "default_threshold="
        f"{DEFAULT_THRESHOLD:.2f} accuracy={metrics['accuracy']:.3f} "
        f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
        f"f1={metrics['f1']:.3f} roc_auc={auc:.3f}"
    )
    print(
        "confusion_matrix="
        f"TP:{metrics['true_positive']} TN:{metrics['true_negative']} "
        f"FP:{metrics['false_positive']} FN:{metrics['false_negative']}"
    )
    print(
        "best_observed_threshold="
        f"{best_threshold:.4f} accuracy={best_metrics['accuracy']:.3f} "
        f"f1={best_metrics['f1']:.3f}"
    )
    if failures:
        print("misclassified_cases:")
        print("\n".join(f"- {failure}" for failure in failures))
    else:
        print("misclassified_cases: none")

    assert len(labels) == 52
    assert len(predictions) == 52
    assert len(scores) == 52
