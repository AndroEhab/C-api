"""Shared library for Spanish boundary benchmark: report generation, maturity
computation, operational readiness, ledger validation, state derivation,
policy freeze, and label-origin consistency.

This is the single source of truth for:
    benchmarkOperationallyReady
    nativeCoverageTargetMet
    knownLimitations
    maturity
    reviewProgress
    policy freeze state
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────
#  Policy freeze
# ──────────────────────────────────────────────────────────────────────

FREEZE_SCHEMA = frozenset({"frozen", "commitSha", "frozenAt", "notes"})

VALID_CONFIDENCE = frozenset({"high", "medium", "low"})
VALID_LABELS = frozenset({"JOIN", "BREAK", "AMBIGUOUS"})
VALID_EVENT_TYPES = frozenset({"review", "adjudication"})
VALID_LABEL_ORIGINS = frozenset({"human"})
VALID_REVIEW_STATUSES = frozenset({
    "unreviewed", "reviewed", "needs_adjudication", "adjudicated",
})


def load_policy_freeze(path: Path) -> dict[str, Any]:
    """Load the policy freeze record from a JSON file."""
    if not path.exists():
        return {"frozen": False, "commitSha": None, "frozenAt": None, "notes": "File not found"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_policy_freeze(freeze: dict[str, Any]) -> list[str]:
    """Validate a policy freeze record. Returns list of error messages (empty = valid)."""
    errors: list[str] = []
    # Must contain only known keys
    unknown_keys = set(freeze.keys()) - FREEZE_SCHEMA
    if unknown_keys:
        errors.append(f"Unknown freeze keys: {sorted(unknown_keys)}")

    if not isinstance(freeze.get("frozen"), bool):
        errors.append("frozen must be a boolean")
    else:
        if freeze["frozen"]:
            sha = freeze.get("commitSha")
            if not sha or not isinstance(sha, str) or len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha):
                errors.append(f"frozen is true but commitSha is invalid: {sha!r}")
            fa = freeze.get("frozenAt")
            if not fa or not isinstance(fa, str):
                errors.append("frozen is true but frozenAt is missing or not a string")
            else:
                try:
                    datetime.fromisoformat(fa)
                except ValueError:
                    errors.append(f"frozenAt is not a valid ISO-8601 timestamp: {fa!r}")
    return errors


def require_policy_freeze_for_test_evaluation(freeze: dict[str, Any]) -> None:
    """Raise if the policy freeze does not permit test evaluation."""
    if not freeze.get("frozen"):
        raise RuntimeError(
            "Test evaluation blocked: policy is not frozen. "
            "Set frozen=true with a valid commitSha and frozenAt."
        )
    sha = freeze.get("commitSha")
    if not sha or not isinstance(sha, str) or len(sha) != 40:
        raise RuntimeError(
            "Test evaluation blocked: invalid or missing commitSha in freeze record."
        )
    fa = freeze.get("frozenAt")
    if not fa or not isinstance(fa, str):
        raise RuntimeError(
            "Test evaluation blocked: missing frozenAt in freeze record."
        )


# ──────────────────────────────────────────────────────────────────────
#  Label-origin consistency
# ──────────────────────────────────────────────────────────────────────

def validate_label_origin_consistency(entries: list[dict[str, Any]]) -> list[str]:
    """Validate label-origin consistency across all entries.

    Returns list of error messages (empty = all consistent).
    """
    errors: list[str] = []
    for i, e in enumerate(entries):
        key = _entry_key(e)
        gl = e.get("goldLabel")
        lo = e.get("labelOrigin")
        rs = e.get("reviewStatus", "unreviewed")

        # goldLabel non-null with null labelOrigin
        if gl is not None and lo is None:
            errors.append(f"Entry {key} (idx {i}): goldLabel={gl!r} but labelOrigin is null")

        # goldLabel non-null with labelOrigin other than human
        if gl is not None and lo is not None and lo != "human":
            errors.append(f"Entry {key} (idx {i}): goldLabel={gl!r} but labelOrigin={lo!r} (must be 'human')")

        # unreviewed entry with non-null labelOrigin
        if rs == "unreviewed" and lo is not None:
            errors.append(f"Entry {key} (idx {i}): unreviewed but labelOrigin={lo!r}")

        # reviewed/adjudicated entry without human labelOrigin
        if rs in ("reviewed", "adjudicated") and lo != "human":
            errors.append(f"Entry {key} (idx {i}): reviewStatus={rs!r} but labelOrigin={lo!r} (must be 'human')")

        # needs_adjudication must have null goldLabel
        if rs == "needs_adjudication":
            if gl is not None:
                errors.append(f"Entry {key} (idx {i}): needs_adjudication but goldLabel={gl!r} (must be null)")

        # invalid goldLabel values
        if gl is not None and gl not in VALID_LABELS:
            errors.append(f"Entry {key} (idx {i}): invalid goldLabel={gl!r}")

        # invalid reviewStatus values
        if rs not in VALID_REVIEW_STATUSES:
            errors.append(f"Entry {key} (idx {i}): invalid reviewStatus={rs!r}")

    return errors


# ──────────────────────────────────────────────────────────────────────
#  Ledger validation
# ──────────────────────────────────────────────────────────────────────

def build_boundary_key(source_id: str, left_cue_id: str, right_cue_id: str) -> str:
    return f"{source_id}:{left_cue_id}:{right_cue_id}"


def validate_ledger_events(
    events: list[dict[str, Any]],
    valid_boundary_ids: set[str],
) -> list[str]:
    """Validate every ledger event against the required schema and business rules.

    Returns list of error messages. Empty list means all events are valid.
    """
    errors: list[str] = []
    seen_review_ids: set[str] = set()
    # Track per-boundary: {boundaryId: {reviewerId: set of rounds}}
    boundary_reviewers: dict[str, dict[str, set[int]]] = {}
    # Track per-boundary: event objects for cross-reference
    boundary_events: dict[str, list[dict]] = {}

    for line_idx, ev in enumerate(events, start=1):
        event_type = ev.get("eventType", "")
        boundary_id = ev.get("boundaryId", "")
        review_id = ev.get("reviewId", "")
        reviewer_id = ev.get("reviewerId", "")

        # 1. unknown eventType
        if event_type not in VALID_EVENT_TYPES:
            errors.append(
                f"Line {line_idx}: unknown eventType={event_type!r} "
                f"(reviewId={review_id!r})"
            )
            continue

        # 2. unknown boundaryId
        if not boundary_id:
            errors.append(f"Line {line_idx}: missing boundaryId (reviewId={review_id!r})")
            continue
        if boundary_id not in valid_boundary_ids:
            errors.append(
                f"Line {line_idx}: unknown boundaryId={boundary_id!r} "
                f"(reviewId={review_id!r})"
            )
            continue

        # 3. duplicate reviewId
        if review_id:
            if review_id in seen_review_ids:
                errors.append(
                    f"Line {line_idx}: duplicate reviewId={review_id!r}"
                )
            seen_review_ids.add(review_id)
        else:
            errors.append(f"Line {line_idx}: missing reviewId")

        # 4. missing reviewerId
        if not reviewer_id:
            errors.append(
                f"Line {line_idx}: missing reviewerId (reviewId={review_id!r})"
            )

        # 5. invalid reviewRound (only for review events)
        review_round = ev.get("reviewRound")
        if event_type == "review":
            if not isinstance(review_round, int) or review_round < 1:
                errors.append(
                    f"Line {line_idx}: reviewRound={review_round!r} invalid "
                    f"(reviewId={review_id!r})"
                )
            elif review_round not in (1, 2):
                errors.append(
                    f"Line {line_idx}: reviewRound={review_round} not 1 or 2 "
                    f"(reviewId={review_id!r})"
                )
        elif event_type == "adjudication":
            # Adjudication must not have reviewRound
            if review_round is not None:
                errors.append(
                    f"Line {line_idx}: adjudication must not have reviewRound "
                    f"(reviewId={review_id!r})"
                )

        # 6. invalid label
        label = ev.get("label")
        if label not in VALID_LABELS:
            errors.append(
                f"Line {line_idx}: invalid label={label!r} "
                f"(reviewId={review_id!r})"
            )

        # 7. invalid confidence
        confidence = ev.get("confidence")
        if confidence not in VALID_CONFIDENCE:
            errors.append(
                f"Line {line_idx}: invalid confidence={confidence!r} "
                f"(reviewId={review_id!r})"
            )

        # 8. labelOrigin other than human
        label_origin = ev.get("labelOrigin")
        if label_origin not in VALID_LABEL_ORIGINS:
            errors.append(
                f"Line {line_idx}: invalid labelOrigin={label_origin!r} "
                f"(reviewId={review_id!r})"
            )

        # 9. malformed createdAt
        created_at = ev.get("createdAt")
        if created_at:
            try:
                datetime.fromisoformat(created_at)
            except (ValueError, TypeError):
                errors.append(
                    f"Line {line_idx}: malformed createdAt={created_at!r} "
                    f"(reviewId={review_id!r})"
                )
        else:
            errors.append(
                f"Line {line_idx}: missing createdAt "
                f"(reviewId={review_id!r})"
            )

        # Business rules (track state for cross-event checks)
        if boundary_id not in boundary_events:
            boundary_events[boundary_id] = []
        boundary_events[boundary_id].append(ev)

        if reviewer_id and boundary_id:
            if boundary_id not in boundary_reviewers:
                boundary_reviewers[boundary_id] = {}
            if reviewer_id not in boundary_reviewers[boundary_id]:
                boundary_reviewers[boundary_id][reviewer_id] = set()
            if event_type == "review" and isinstance(review_round, int):
                boundary_reviewers[boundary_id][reviewer_id].add(review_round)

    # Cross-event validation
    for bid, evs in boundary_events.items():
        by_type: dict[str, list[dict]] = {}
        for e in evs:
            by_type.setdefault(e.get("eventType", ""), []).append(e)

        # - a reviewer performing both independent reviews on one boundary
        for reviewer, rounds in boundary_reviewers.get(bid, {}).items():
            if 1 in rounds and 2 in rounds:
                errors.append(
                    f"boundaryId={bid!r}: reviewer={reviewer!r} performed both "
                    f"round 1 and round 2 reviews"
                )

        # - a round-two event without an existing round-one event
        round_two_events = [e for e in evs if e.get("reviewRound") == 2]
        round_one_events = [e for e in evs if e.get("reviewRound") == 1]
        if round_two_events and not round_one_events:
            for e in round_two_events:
                errors.append(
                    f"boundaryId={bid!r}: round-two event (reviewId={e.get('reviewId', '?')}) "
                    f"without existing round-one event"
                )

        # - an adjudication without a disagreement
        adjudications = by_type.get("adjudication", [])
        reviews = by_type.get("review", [])
        if adjudications:
            # Check that there's a disagreement among unique reviewers' labels
            # At least 2 reviews with different labels
            reviewer_labels: dict[str, str] = {}
            for e in reviews:
                rid = e.get("reviewerId", "")
                lbl = e.get("label", "")
                if rid and lbl:
                    reviewer_labels[rid] = lbl
            if len(set(reviewer_labels.values())) < 2:
                errors.append(
                    f"boundaryId={bid!r}: adjudication without disagreement "
                    f"(reviewer labels: {reviewer_labels})"
                )

        # - duplicate active reviews from the same reviewer and round
        reviewer_round_count: dict[tuple[str, int], int] = {}
        for e in evs:
            if e.get("eventType") == "review":
                rid = e.get("reviewerId", "")
                rr = e.get("reviewRound")
                if rid and isinstance(rr, int):
                    key = (rid, rr)
                    reviewer_round_count[key] = reviewer_round_count.get(key, 0) + 1
        for (rid, rr), count in reviewer_round_count.items():
            if count > 1:
                errors.append(
                    f"boundaryId={bid!r}: reviewer={rid!r} has {count} reviews "
                    f"for round {rr}"
                )

    return errors


# ──────────────────────────────────────────────────────────────────────
#  State derivation
# ──────────────────────────────────────────────────────────────────────

def derive_review_state(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive final review state from a boundary's events.

    Returns dict with keys:
        goldLabel, labelConfidence, reviewerCount, needsSecondReview,
        reviewReason, reviewStatus, labelOrigin
    """
    # Default unreviewed state
    state: dict[str, Any] = {
        "goldLabel": None,
        "labelConfidence": None,
        "reviewerCount": 0,
        "needsSecondReview": False,
        "reviewReason": "",
        "reviewStatus": "unreviewed",
        "labelOrigin": None,
    }

    if not events:
        return state

    # Separate regular reviews (round 1 or 2) from adjudications
    reviews = [e for e in events
               if e.get("eventType") == "review"
               and e.get("reviewRound") in (1, 2)]
    adjudications = [e for e in events
                     if e.get("eventType") == "adjudication"]

    # If there's an adjudication event, use it
    if adjudications:
        adj = adjudications[-1]  # last adjudication wins
        label = adj.get("label", "AMBIGUOUS")
        confidence = adj.get("confidence", "medium")
        # Low-confidence final decisions become AMBIGUOUS
        if confidence == "low" and label != "AMBIGUOUS":
            label = "AMBIGUOUS"
            confidence = "low"

        # Count unique human reviewers from ALL events
        reviewer_ids = _unique_reviewer_ids(events)
        state.update({
            "goldLabel": label,
            "labelConfidence": confidence,
            "reviewerCount": len(reviewer_ids),
            "needsSecondReview": False,
            "reviewReason": adj.get("reason", ""),
            "reviewStatus": "adjudicated",
            "labelOrigin": "human",
        })
        return state

    if not reviews:
        return state

    # Count unique reviewers from review events
    reviewer_ids = _unique_reviewer_ids(reviews)
    reviewer_count = len(reviewer_ids)

    # Get latest review per unique reviewer
    latest_per_reviewer: dict[str, dict] = {}
    for e in reviews:
        rid = e.get("reviewerId", "")
        if rid:
            # Later events override earlier ones for the same reviewer
            latest_per_reviewer[rid] = e

    review_list = list(latest_per_reviewer.values())

    if reviewer_count == 1:
        return _derive_single_review_state(review_list[0], reviewer_count, state)
    else:
        return _derive_multi_review_state(review_list, reviewer_count, state)


def _unique_reviewer_ids(events: list[dict]) -> list[str]:
    """Return ordered unique reviewer IDs."""
    seen: set[str] = set()
    ids: list[str] = []
    for e in events:
        rid = e.get("reviewerId", "")
        if rid and rid not in seen:
            seen.add(rid)
            ids.append(rid)
    return ids


def _derive_single_review_state(
    r: dict, reviewer_count: int, state: dict,
) -> dict:
    """Derive state for a single review (possibly needs second review)."""
    label = r.get("label")
    confidence = r.get("confidence", "medium")
    reason = r.get("reason", "")

    needs_second = False

    # Rule: JOIN + high confidence -> reviewed, no mandatory second review
    # Rule: JOIN + medium/low -> retain provisional JOIN, needsSecondReview = true
    if label == "JOIN":
        if confidence in ("medium", "low"):
            needs_second = True
        # else high confidence: no second review needed

    # Rule: BREAK + high/medium -> reviewed (unless explicitly marked)
    # That's already the default (needs_second stays False)

    # Rule: AMBIGUOUS -> reviewed and excluded from primary metrics
    # Already handled - reviewer count is set, status is reviewed

    # Rule: Do NOT automatically convert low-confidence JOIN to AMBIGUOUS
    # (low confidence JOIN retains the JOIN label but needs second review)

    state.update({
        "goldLabel": label,
        "labelConfidence": confidence,
        "reviewerCount": reviewer_count,
        "needsSecondReview": needs_second,
        "reviewReason": reason,
        "reviewStatus": "reviewed",
        "labelOrigin": "human",
    })
    return state


def _derive_multi_review_state(
    review_list: list[dict], reviewer_count: int, state: dict,
) -> dict:
    """Derive state for two or more reviews."""
    labels = set(r.get("label") for r in review_list)

    if len(labels) == 1:
        # Agreeing labels -> reviewed
        r = review_list[0]
        label = r.get("label")
        confidence = r.get("confidence", "medium")
        reason = r.get("reason", "")

        # If the agreed final result remains low confidence, convert to AMBIGUOUS
        if confidence == "low" and label != "AMBIGUOUS":
            label = "AMBIGUOUS"
            confidence = "low"

        state.update({
            "goldLabel": label,
            "labelConfidence": confidence,
            "reviewerCount": reviewer_count,
            "needsSecondReview": False,
            "reviewReason": f"Agreed: {reason}" if reason else "Agreed",
            "reviewStatus": "reviewed",
            "labelOrigin": "human",
        })
        return state

    # Disagreeing labels
    reason = "; ".join(
        f"R{e.get('reviewerId', '?')}: {e.get('label', '?')} ({e.get('reason', '')})"
        for e in review_list
    )
    state.update({
        "goldLabel": None,
        "labelConfidence": None,
        "reviewerCount": reviewer_count,
        "needsSecondReview": False,  # cleared; adjudication handles it
        "reviewReason": f"Disagreement: {reason}",
        "reviewStatus": "needs_adjudication",
        "labelOrigin": "human",
    })
    return state


# ──────────────────────────────────────────────────────────────────────
#  Maturity levels
# ──────────────────────────────────────────────────────────────────────

def compute_maturity_level(
    entries: list[dict[str, Any]],
    policy_freeze: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate benchmark maturity level based on review progress and coverage.

    Rules:
      exploratory: 75+ reviewed non-ambiguous, 5+ JOIN, 5+ BREAK
      usable: 150+ reviewed non-ambiguous, meaningful coverage in dev and test,
              10+ JOIN, 10+ BREAK, 2+ productions, some second review,
              all accepted labels human origin
      validated: 200+ reviewed non-ambiguous, 80%+ dev reviewed, 80%+ test reviewed,
                 20+ JOIN, 20+ BREAK, 20%+ second-reviewed, no unresolved disagreements,
                 all human origin, valid frozen policy
    """
    reviewed_non_ambig = sum(
        1 for e in entries
        if e.get("reviewStatus") in ("reviewed", "adjudicated")
        and e.get("goldLabel") not in (None, "AMBIGUOUS")
    )

    source_ids = {e["sourceId"] for e in entries}
    num_productions = len(source_ids)

    all_tags: set[str] = set()
    for e in entries:
        all_tags.update(e.get("samplingTags", []))

    break_like_tags = {"independent_utterance", "short_response",
                        "dialogue_dash", "unknown_speaker_turn",
                        "explicit_speaker_change", "inverted_question",
                        "inverted_exclamation", "ellipsis_or_interruption",
                        "caption"}
    join_like_tags = {"join_like", "misleading_period"}
    has_break_like = bool(all_tags & break_like_tags)
    has_join_like = bool(all_tags & join_like_tags)

    timing_bands = {e.get("timingBand") for e in entries}
    has_timing_spread = len(timing_bands) >= 3

    second_reviewed = sum(1 for e in entries if e.get("reviewerCount", 0) >= 2)
    total_reviewed = sum(1 for e in entries
                         if e.get("reviewStatus") in ("reviewed", "adjudicated"))
    second_review_pct = second_reviewed / total_reviewed if total_reviewed else 0.0

    has_held_out = any(e.get("split") == "test" for e in entries)
    has_dev = any(e.get("split") == "dev" for e in entries)

    content_types_actual: set[str] = set()
    for e in entries:
        ct = e.get("contentStructure", "")
        if ct and ct != "unknown":
            content_types_actual.add(ct)

    reviewed_join = sum(
        1 for e in entries
        if e.get("reviewStatus") in ("reviewed", "adjudicated")
        and e.get("goldLabel") == "JOIN"
    )
    reviewed_break = sum(
        1 for e in entries
        if e.get("reviewStatus") in ("reviewed", "adjudicated")
        and e.get("goldLabel") == "BREAK"
    )

    all_reviewed_human = all(
        e.get("labelOrigin") == "human"
        for e in entries
        if e.get("reviewStatus") in ("reviewed", "adjudicated")
    )

    needs_adjudication_count = sum(
        1 for e in entries if e.get("reviewStatus") == "needs_adjudication"
    )
    all_disagreements_adjudicated = needs_adjudication_count == 0

    native_count = sum(
        1 for e in entries if e.get("originalSpokenLanguage") == "es"
    )

    # Dev/test specific counts for validated
    dev_reviewed = sum(
        1 for e in entries
        if e.get("split") == "dev"
        and e.get("reviewStatus") in ("reviewed", "adjudicated")
    )
    test_reviewed = sum(
        1 for e in entries
        if e.get("split") == "test"
        and e.get("reviewStatus") in ("reviewed", "adjudicated")
    )
    dev_total = sum(1 for e in entries if e.get("split") == "dev")
    test_total = sum(1 for e in entries if e.get("split") == "test")

    dev_reviewed_pct = dev_reviewed / dev_total if dev_total else 0.0
    test_reviewed_pct = test_reviewed / test_total if test_total else 0.0

    # Policy freeze state
    if policy_freeze is not None:
        is_frozen = bool(policy_freeze.get("frozen", False))
    else:
        is_frozen = False

    # Exploratory
    is_exploratory = (
        reviewed_non_ambig >= 75
        and reviewed_join >= 5
        and reviewed_break >= 5
        and has_break_like
        and has_join_like
    )

    # Usable
    # Concrete minimum: at least 20 reviewed non-ambiguous test boundaries
    test_reviewed_non_ambig = sum(
        1 for e in entries
        if e.get("split") == "test"
        and e.get("reviewStatus") in ("reviewed", "adjudicated")
        and e.get("goldLabel") not in (None, "AMBIGUOUS")
    )
    is_usable = (
        reviewed_non_ambig >= 150
        and num_productions >= 2
        and reviewed_join >= 10
        and reviewed_break >= 10
        and has_break_like
        and has_join_like
        and has_timing_spread
        and has_dev
        and has_held_out
        and test_reviewed_non_ambig >= 20
        and second_review_pct > 0
        and all_reviewed_human
    )

    # Validated
    is_validated = (
        reviewed_non_ambig >= 200
        and dev_reviewed_pct >= 0.8
        and test_reviewed_pct >= 0.8
        and reviewed_join >= 20
        and reviewed_break >= 20
        and num_productions >= 2
        and content_types_actual.issuperset({"dialogue", "monologue"})
        and has_held_out
        and second_review_pct >= 0.2
        and all_disagreements_adjudicated
        and native_count >= 50
        and all_reviewed_human
        and is_frozen
    )

    if is_validated:
        level = "validated"
    elif is_usable:
        level = "usable"
    elif is_exploratory:
        level = "exploratory"
    else:
        level = "none"

    return {
        "maturityLevel": level,
        "reviewedNonAmbiguous": reviewed_non_ambig,
        "numProductions": num_productions,
        "hasBreakLikeCategories": has_break_like,
        "hasJoinLikeCategories": has_join_like,
        "hasTimingSpread": has_timing_spread,
        "hasHeldOutTest": has_held_out,
        "hasDevSplit": has_dev,
        "secondReviewPercentage": round(second_review_pct, 3),
        "nativeSourceEntries": native_count,
        "contentStructures": sorted(content_types_actual),
        "reviewedJoin": reviewed_join,
        "reviewedBreak": reviewed_break,
        "needsAdjudicationCount": needs_adjudication_count,
        "isFrozen": is_frozen,
    }


# ──────────────────────────────────────────────────────────────────────
#  Operational readiness
# ──────────────────────────────────────────────────────────────────────

def is_operationally_ready(
    entries: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> bool:
    """Check if the benchmark meets operational readiness criteria.

    Independent of native-source coverage completeness.
    """
    if len(entries) < 150:
        return False

    source_ids = {e["sourceId"] for e in entries}
    if len(source_ids) < 2:
        return False

    all_tags: set[str] = set()
    for e in entries:
        all_tags.update(e.get("samplingTags", []))

    break_like = {"independent_utterance", "short_response",
                   "dialogue_dash", "unknown_speaker_turn",
                   "explicit_speaker_change"}
    join_like = {"join_like", "misleading_period"}
    if not (all_tags & break_like) or not (all_tags & join_like):
        return False

    timing_bands = {e.get("timingBand") for e in entries}
    if len(timing_bands) < 2:
        return False

    has_dialogue = any(e.get("contentStructure") == "dialogue" for e in entries)
    has_dialogue_or_punctuation = has_dialogue or any(
        "inverted_question" in e.get("samplingTags", []) or
        "inverted_exclamation" in e.get("samplingTags", [])
        for e in entries
    )
    if not has_dialogue_or_punctuation:
        return False

    # Source manifest validation
    manifest_by_id: dict = {m["sourceId"]: m for m in manifest}
    for e in entries:
        src = e["sourceId"]
        if src not in manifest_by_id:
            return False

    required_prov = {"sourceId", "title", "contentType", "contentStructure",
                     "sourceQualityTier", "originalSpokenLanguage",
                     "subtitleLanguage", "fullSha256", "cueCount"}
    for m in manifest:
        missing = required_prov - set(m.keys())
        if missing:
            return False

    valid_tiers = {"native_original", "professional_translation",
                   "community_translation", "reviewed_machine_transcription",
                   "unknown"}
    for m in manifest:
        tier = m.get("sourceQualityTier", "")
        if tier not in valid_tiers:
            return False

    # Reference cue validation
    by_source_cue: dict[str, set[str]] = {}
    for ref in reference:
        src = ref["sourceId"]
        if src not in by_source_cue:
            by_source_cue[src] = set()
        by_source_cue[src].add(ref["cueId"])

    for e in entries:
        src = e["sourceId"]
        ref_cues = by_source_cue.get(src, set())
        for side in ("left", "right"):
            cid = e[f"{side}CueId"]
            if cid not in ref_cues:
                return False
        for ctx_list_key in ("previousContext", "nextContext"):
            for ctx in e.get(ctx_list_key, []):
                ctx_id = ctx["cueId"]
                if ctx_id not in ref_cues:
                    return False

    # No model prediction/probability fields
    forbidden = {"modelProbability", "modelPrediction", "modelScore",
                 "saTScore", "prediction", "model", "evidence"}
    for e in entries:
        found = forbidden & set(e.keys())
        if found:
            return False

    # Label-origin consistency check (non-human labels block readiness)
    lo_errors = validate_label_origin_consistency(entries)
    if lo_errors:
        return False

    return True


# ──────────────────────────────────────────────────────────────────────
#  Dataset report generation
# ──────────────────────────────────────────────────────────────────────

def generate_dataset_report(
    entries: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    policy_freeze: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate comprehensive dataset report.

    This is the single source of truth for:
        benchmarkOperationallyReady
        nativeCoverageTargetMet
        knownLimitations
        maturity
        reviewProgress
        policy freeze state
    """
    manifest_by_id: dict = {m["sourceId"]: m for m in manifest}
    source_counts: Counter = Counter()
    for e in entries:
        source_counts[e["sourceId"]] += 1

    # Per region/variant
    region_counts: Counter = Counter()
    for e in entries:
        src = e["sourceId"]
        var = manifest_by_id.get(src, {}).get("spanishVariant", "unknown")
        region_counts[var] += 1

    # Per content type
    type_counts: Counter = Counter()
    for e in entries:
        src = e["sourceId"]
        ctype = manifest_by_id.get(src, {}).get("contentType", "other")
        type_counts[ctype] += 1

    # Per source quality tier
    tier_counts: Counter = Counter()
    for e in entries:
        tier_counts[e.get("sourceQualityTier", "unknown")] += 1

    # Per content structure
    structure_counts: Counter = Counter()
    for e in entries:
        structure_counts[e.get("contentStructure", "unknown")] += 1

    # Per original spoken language
    lang_counts: Counter = Counter()
    for e in entries:
        lang_counts[e.get("originalSpokenLanguage", "unknown")] += 1

    # Sampling tag counts
    tag_counts: Counter = Counter()
    for e in entries:
        for tag in e.get("samplingTags", []):
            tag_counts[tag] += 1

    # Timing band counts
    timing_counts: Counter = Counter()
    for e in entries:
        timing_counts[e.get("timingBand", "unknown")] += 1

    # Split counts
    dev_count = sum(1 for e in entries if e.get("split") == "dev")
    test_count = sum(1 for e in entries if e.get("split") == "test")

    # Dev/test tag coverage
    dev_tags: Counter = Counter()
    test_tags: Counter = Counter()
    for e in entries:
        split = e.get("split", "")
        for tag in e.get("samplingTags", []):
            if split == "dev":
                dev_tags[tag] += 1
            elif split == "test":
                test_tags[tag] += 1

    # Multiline count
    multiline_left = sum(
        1 for e in entries if "multiline_cue_left" in e.get("structureTags", [])
    )
    multiline_right = sum(
        1 for e in entries if "multiline_cue_right" in e.get("structureTags", [])
    )
    multiline = sum(
        1 for e in entries
        if "multiline_cue_left" in e.get("structureTags", [])
        or "multiline_cue_right" in e.get("structureTags", [])
    )

    # Multiple speaker count
    multi_speaker = sum(
        1 for e in entries
        if e.get("speakerMarkers", {})
        .get("left", {})
        .get("contains_multiple_speakers", False)
        or e.get("speakerMarkers", {})
        .get("right", {})
        .get("contains_multiple_speakers", False)
    )

    # Chain count
    chain_ids: set[str] = set()
    for e in entries:
        cid = e.get("chainId")
        if cid:
            chain_ids.add(cid)

    unreviewed = sum(1 for e in entries if e.get("reviewStatus") == "unreviewed")
    human_labeled = sum(1 for e in entries if e.get("labelOrigin") == "human")

    # Review progress
    reviewed = sum(
        1 for e in entries if e.get("reviewStatus") in ("reviewed", "adjudicated")
    )
    needs_second = sum(1 for e in entries if e.get("needsSecondReview"))
    second_done = sum(1 for e in entries if e.get("reviewerCount", 0) >= 2)
    ambiguous = sum(1 for e in entries if e.get("goldLabel") == "AMBIGUOUS")

    # Collect known limitations from manifest sources
    manifest_limitations: list[str] = []
    for m in manifest:
        src_lims = m.get("knownLimitations", [])
        if isinstance(src_lims, list):
            manifest_limitations.extend(src_lims)
    known_limitations = manifest_limitations or [
        "No dialogue-heavy source originally spoken in Spanish from Spain exists.",
        "No dialogue-heavy source originally spoken in Spanish from Latin America exists.",
        "The only originally-Spanish source (ted_tales_es) is a TED monologue, not dialogue-heavy.",
        "The only dialogue-heavy source (the_goat_life_es) is translated from Malayalam; provenance is community translation, not professionally verified.",
    ]

    # Maturity calculation
    maturity = compute_maturity_level(entries, policy_freeze)

    # Operational readiness
    op_ready = is_operationally_ready(entries, manifest, reference)

    # Source provenance
    source_provenance = []
    for m in manifest:
        sid = m["sourceId"]
        source_provenance.append({
            "sourceId": sid,
            "contentType": m.get("contentType", "other"),
            "contentStructure": m.get("contentStructure", "unknown"),
            "originalSpokenLanguage": m.get("originalSpokenLanguage", "unknown"),
            "spanishVariant": m.get("spanishVariant", "unknown"),
            "sourceQualityTier": m.get("sourceQualityTier", "unknown"),
            "translationType": m.get("translationType", "unknown"),
            "candidatesInBenchmark": source_counts.get(sid, 0),
        })

    report: dict[str, Any] = {
        "datasetReport": {
            "reportVersion": "2.0",
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        },
        "sourceProvenance": source_provenance,
        "overview": {
            "totalCandidates": len(entries),
            "totalSources": len(source_counts),
            "devCount": dev_count,
            "testCount": test_count,
            "devRatio": round(dev_count / len(entries), 3) if entries else 0,
            "unreviewed": unreviewed,
            "humanLabeledCount": human_labeled,
            "multilineCues": multiline,
            "multilineLeft": multiline_left,
            "multilineRight": multiline_right,
            "multipleSpeakerCues": multi_speaker,
            "chainCount": len(chain_ids),
        },
        "reviewProgress": {
            "unreviewed": unreviewed,
            "reviewed": reviewed,
            "needsSecondReview": needs_second,
            "secondReviewCompleted": second_done,
            "ambiguousExcluded": ambiguous,
        },
        "metricStrata": {
            "candidatesPerSource": dict(source_counts.most_common()),
            "candidatesPerRegion": dict(region_counts.most_common()),
            "candidatesPerContentType": dict(type_counts.most_common()),
            "candidatesPerQualityTier": dict(tier_counts.most_common()),
            "candidatesPerContentStructure": dict(structure_counts.most_common()),
            "candidatesPerOriginalLanguage": dict(lang_counts.most_common()),
        },
        "samplingTagCounts": dict(tag_counts.most_common()),
        "timingBandCounts": dict(timing_counts.most_common()),
        "devTestCoverage": {
            "devCount": dev_count,
            "testCount": test_count,
            "devTagCoverage": dict(dev_tags.most_common()),
            "testTagCoverage": dict(test_tags.most_common()),
        },
        "benchmarkOperationallyReady": op_ready,
        "nativeCoverageTargetMet": False,
        "knownLimitations": known_limitations,
        "maturity": maturity,
    }
    return report


def _entry_key(entry: dict) -> str:
    """Build a readable key for an entry for error messages."""
    return f"{entry.get('sourceId', '?')}:{entry.get('leftCueId', '?')}:{entry.get('rightCueId', '?')}"
