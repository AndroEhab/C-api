"""Run the single frozen Spanish held-out evaluation with resumable checkpoints.

This utility never changes production configuration. It validates the frozen
policy and finalized silver hashes, invokes the production
``/reconstruct-subtitles`` function in-process once per held-out boundary, and
records raw predictions for metric calculation without rescoring.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from app.main import (  # noqa: E402
    SubtitleReconstructionRequest,
    reconstruct_subtitles,
)
from app.sat import (  # noqa: E402
    SAT_MODEL_NAME,
    SaTSentenceReconstructor,
)
from benchmarks.spanish_benchmark_lib import (  # noqa: E402
    load_policy_freeze,
    require_policy_freeze_for_test_evaluation,
)

CANDIDATES_PATH = BENCHMARK_DIR / "spanish_boundary_candidates.json"
LABELS_PATH = BENCHMARK_DIR / "spanish_boundary_test_silver_labels.json"
HASH_MANIFEST_PATH = BENCHMARK_DIR / "spanish_boundary_test_silver_hashes.json"
FREEZE_PATH = BENCHMARK_DIR / "spanish_policy_freeze.json"
LEDGER_PATH = BENCHMARK_DIR / "spanish_boundary_reviews.jsonl"
OUTPUT_PATH = BENCHMARK_DIR / "spanish_boundary_test_predictions.json"
TASK_8D_SHA = "dc9bdb69b80ab6d22037097dce429334ae09da2f"
EXPECTED_DEV_HASHES = {
    "spanish_boundary_dev_silver_labels.json": "2284b5e35ad7de7e4ea02c5bfec1ebd0d29fa6ac0aabe81a3bafef61955838ca",
    "spanish_boundary_dev_silver_labels.csv": "33b4b5ab5e1e03600d707cae6c26a23e7ac041c39b77da6625e52acc037fb843",
    "spanish_boundary_dev_silver_report.json": "b6bc3d0e5843f17e1f05f0428a827ec8a1eb31dd59dce558bf53e0eb3d36a2cc",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boundary_id(candidate: dict[str, Any]) -> str:
    return f"{candidate['sourceId']}:{candidate['leftCueId']}:{candidate['rightCueId']}"


def _request_segment(cue: dict[str, Any]) -> dict[str, Any]:
    return {
        "segmentId": str(cue["cueId"]),
        "text": cue["rawText"],
        "startMs": cue["startMs"],
        "endMs": cue["endMs"],
        "rawText": cue["rawText"],
        "lines": cue["lines"],
    }


def _evaluation_window(candidate: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    left = {
        "cueId": candidate["leftCueId"],
        "rawText": candidate["leftRawText"],
        "lines": candidate["leftLines"],
        "startMs": candidate["leftStartMs"],
        "endMs": candidate["leftEndMs"],
    }
    right = {
        "cueId": candidate["rightCueId"],
        "rawText": candidate["rightRawText"],
        "lines": candidate["rightLines"],
        "startMs": candidate["rightStartMs"],
        "endMs": candidate["rightEndMs"],
    }
    return (
        [*candidate["previousContext"], left, right, *candidate["nextContext"]],
        len(candidate["previousContext"]),
    )


class RecordingSaTService(SaTSentenceReconstructor):
    """Production SaT service with behavior-preserving evidence capture."""

    def __init__(self) -> None:
        super().__init__()
        self.evaluation_calls: list[dict[str, Any]] = []

    def windowed_score_boundaries(
        self,
        segments: Sequence[str],
        *,
        profile: Any = None,
        config: Any = None,
    ) -> list[dict[str, Any]]:
        evidence = super().windowed_score_boundaries(
            segments,
            profile=profile,
            config=config,
        )
        self.evaluation_calls.append(
            {
                "profile": type(profile).__name__ if profile is not None else None,
                "profileCode": getattr(profile, "code", None),
                "evidence": evidence,
            }
        )
        return evidence


def _write_checkpoint(payload: dict[str, Any]) -> None:
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT_PATH)


def main() -> None:
    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    test_candidates = [entry for entry in candidates if entry.get("split") == "test"]
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    hash_manifest = json.loads(HASH_MANIFEST_PATH.read_text(encoding="utf-8"))
    freeze = load_policy_freeze(FREEZE_PATH)

    require_policy_freeze_for_test_evaluation(freeze)
    if freeze["commitSha"] != TASK_8D_SHA:
        raise RuntimeError("frozen policy target does not match the Task 8D commit")
    if len(test_candidates) != 74 or len(labels) != 74:
        raise RuntimeError("held-out evaluation requires exactly 74 candidates and labels")
    if any(entry.get("goldLabel") is not None for entry in candidates):
        raise RuntimeError("official candidate goldLabel fields must remain null")
    if LEDGER_PATH.read_bytes():
        raise RuntimeError("Spanish review ledger must remain empty")
    for name, expected_hash in EXPECTED_DEV_HASHES.items():
        if _sha256(BENCHMARK_DIR / name) != expected_hash:
            raise RuntimeError(f"development artifact changed: {name}")
    for name, expected_hash in hash_manifest["artifacts"].items():
        if _sha256(BENCHMARK_DIR / name) != expected_hash:
            raise RuntimeError(f"finalized test silver artifact changed: {name}")

    expected_ids = [_boundary_id(candidate) for candidate in test_candidates]
    label_ids = [label["boundaryId"] for label in labels]
    if expected_ids != label_ids or len(set(expected_ids)) != 74:
        raise RuntimeError("test labels do not match canonical held-out boundaries")

    run_id = hashlib.sha256(
        (freeze["commitSha"] + json.dumps(hash_manifest["artifacts"], sort_keys=True)).encode("utf-8")
    ).hexdigest()
    predictions: list[dict[str, Any]] = []
    if OUTPUT_PATH.exists():
        checkpoint = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if checkpoint.get("runId") != run_id:
            raise RuntimeError("existing prediction checkpoint belongs to another frozen run")
        predictions = checkpoint.get("predictions", [])
        if checkpoint.get("complete"):
            raise RuntimeError("frozen held-out evaluation is already complete; refusing to rescore")
        if [item["boundaryId"] for item in predictions] != expected_ids[: len(predictions)]:
            raise RuntimeError("prediction checkpoint is not a canonical prefix")

    service = RecordingSaTService()
    if service.model_name != SAT_MODEL_NAME or SAT_MODEL_NAME != "sat-3l-sm":
        raise RuntimeError("unexpected SaT model")

    checkpoint_payload = {
        "reportType": "Raw frozen Spanish held-out predictions",
        "runId": run_id,
        "complete": False,
        "frozenPolicyTargetSha": freeze["commitSha"],
        "satModel": service.model_name,
        "testSilverArtifactHashes": hash_manifest["artifacts"],
        "predictions": predictions,
    }
    _write_checkpoint(checkpoint_payload)
    print(f"EVALUATION_START completed={len(predictions)} total=74", flush=True)

    for canonical_index in range(len(predictions), len(test_candidates)):
        candidate = test_candidates[canonical_index]
        window, target_index = _evaluation_window(candidate)
        calls_before = len(service.evaluation_calls)
        request = SubtitleReconstructionRequest(
            segments=[_request_segment(cue) for cue in window],
            language="es",
        )
        response = reconstruct_subtitles(request, service=service)
        if len(service.evaluation_calls) != calls_before + 1:
            raise RuntimeError("production reconstruction did not issue exactly one scoring call")
        target_evidence = service.evaluation_calls[-1]["evidence"][target_index]
        left_id = str(candidate["leftCueId"])
        right_id = str(candidate["rightCueId"])
        sentence_ids = [sentence.segment_ids for sentence in response.sentences]
        left_sentence_index = next(index for index, ids in enumerate(sentence_ids) if left_id in ids)
        right_sentence_index = next(index for index, ids in enumerate(sentence_ids) if right_id in ids)
        predicted = "JOIN" if left_sentence_index == right_sentence_index else "BREAK"
        diagnostics = (
            response.diagnostics.model_dump(by_alias=True)
            if response.diagnostics is not None
            else None
        )
        predictions.append(
            {
                "canonicalIndex": canonical_index,
                "boundaryId": _boundary_id(candidate),
                "modelPrediction": predicted,
                "satProbability": target_evidence.get("boundaryProbability"),
                "satStatus": target_evidence.get("status", "available"),
                "reconstructedText": (
                    response.sentences[left_sentence_index].text
                    if predicted == "JOIN"
                    else None
                ),
                "diagnostics": diagnostics,
            }
        )
        checkpoint_payload["predictions"] = predictions
        _write_checkpoint(checkpoint_payload)
        print(f"EVALUATION_PROGRESS completed={len(predictions)} boundaryId={predictions[-1]['boundaryId']}", flush=True)

    diagnostics = {json.dumps(item["diagnostics"], sort_keys=True) for item in predictions}
    expected_diagnostics = json.dumps(
        {
            "requestedLanguage": "es",
            "resolvedLanguage": "es",
            "profile": "NeutralBoundaryProfile",
            "profileCode": "und",
        },
        sort_keys=True,
    )
    if diagnostics != {expected_diagnostics}:
        raise RuntimeError(f"unexpected reconstruction diagnostics: {sorted(diagnostics)}")
    if any(item["satProbability"] is None for item in predictions):
        raise RuntimeError("SaT probability unavailable for one or more held-out boundaries")

    checkpoint_payload["complete"] = True
    checkpoint_payload["completedBoundaries"] = len(predictions)
    checkpoint_payload["observedDiagnostics"] = [json.loads(expected_diagnostics)]
    _write_checkpoint(checkpoint_payload)
    print("EVALUATION_COMPLETE completed=74", flush=True)


if __name__ == "__main__":
    main()
