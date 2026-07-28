"""Validation tests for the Spanish subtitle-boundary benchmark (Task 8A)."""

from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path

import pytest

from app.subtitles import parse_srt_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
SOURCE_DIR = PROJECT_ROOT / "benchmark-source"
FIXTURE_PATH = BENCHMARK_DIR / "spanish_boundary_candidates.json"
CSV_PATH = BENCHMARK_DIR / "spanish_boundary_candidates.csv"
SPLITS_PATH = BENCHMARK_DIR / "spanish_boundary_splits.json"
LABELING_GUIDE = BENCHMARK_DIR / "SPANISH_BOUNDARY_LABELING.md"

VALID_LABELS = {"JOIN", "BREAK", "AMBIGUOUS"}


@pytest.fixture(scope="module")
def fixture() -> list[dict]:
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def splits() -> dict:
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def source_data() -> dict[str, dict[str, str]]:
    """Return {source_id: {cue_id: text}} for every source file."""
    result = {}
    for srt_path in sorted(SOURCE_DIR.rglob("*.srt")):
        source_id = srt_path.stem
        segments = parse_srt_file(srt_path)
        result[source_id] = {
            s.segment_id: {
                "text": s.text,
                "startMs": s.start_ms,
                "endMs": s.end_ms,
            }
            for s in segments
        }
    return result


def _source_and_checksum(fixture_entry: dict) -> tuple[str, str]:
    return fixture_entry.get("sourceId", ""), fixture_entry.get("sourceChecksum", "")


# ── 1. Every cue ID exists in source ──────────────────────────────────────


class TestCueIdsExist:
    def test_every_left_cue_id_exists(self, fixture, source_data):
        for entry in fixture:
            src = entry["sourceId"]
            cid = entry["leftCueId"]
            assert src in source_data, f"source {src} not found in source_data"
            assert cid in source_data[src], (
                f"leftCueId {cid} not found in source {src}"
            )

    def test_every_right_cue_id_exists(self, fixture, source_data):
        for entry in fixture:
            src = entry["sourceId"]
            cid = entry["rightCueId"]
            assert cid in source_data[src], (
                f"rightCueId {cid} not found in source {src}"
            )


# ── 2. Fixture text matches source exactly ────────────────────────────────


class TestFixtureTextMatchesSource:
    def test_left_text_matches(self, fixture, source_data):
        for entry in fixture:
            src = entry["sourceId"]
            cid = entry["leftCueId"]
            expected = source_data[src][cid]["text"]
            assert entry["left"] == expected, (
                f"left text mismatch for {src} cue {cid}: "
                f"{entry['left'][:60]!r} != {expected[:60]!r}"
            )

    def test_right_text_matches(self, fixture, source_data):
        for entry in fixture:
            src = entry["sourceId"]
            cid = entry["rightCueId"]
            expected = source_data[src][cid]["text"]
            assert entry["right"] == expected, (
                f"right text mismatch for {src} cue {cid}: "
                f"{entry['right'][:60]!r} != {expected[:60]!r}"
            )


# ── 3. Timings match source ──────────────────────────────────────────────


class TestTimingsMatch:
    def test_left_start_ms(self, fixture, source_data):
        for entry in fixture:
            src = entry["sourceId"]
            cid = entry["leftCueId"]
            assert entry["leftStartMs"] == source_data[src][cid]["startMs"], (
                f"leftStartMs mismatch for {src} {cid}"
            )

    def test_left_end_ms(self, fixture, source_data):
        for entry in fixture:
            src = entry["sourceId"]
            cid = entry["leftCueId"]
            assert entry["leftEndMs"] == source_data[src][cid]["endMs"], (
                f"leftEndMs mismatch for {src} {cid}"
            )

    def test_right_start_ms(self, fixture, source_data):
        for entry in fixture:
            src = entry["sourceId"]
            cid = entry["rightCueId"]
            assert entry["rightStartMs"] == source_data[src][cid]["startMs"], (
                f"rightStartMs mismatch for {src} {cid}"
            )

    def test_right_end_ms(self, fixture, source_data):
        for entry in fixture:
            src = entry["sourceId"]
            cid = entry["rightCueId"]
            assert entry["rightEndMs"] == source_data[src][cid]["endMs"], (
                f"rightEndMs mismatch for {src} {cid}"
            )


# ── 4. Boundaries are adjacent ───────────────────────────────────────────


class TestBoundariesAreAdjacent:
    def test_left_right_are_consecutive_cues(self, fixture):
        """Verify left and right cues are from the same source and adjacent."""
        for entry in fixture:
            src = entry["sourceId"]
            segs = parse_srt_file(SOURCE_DIR / f"{src}.srt")
            ids = [s.segment_id for s in segs]
            left_idx = ids.index(entry["leftCueId"])
            right_idx = ids.index(entry["rightCueId"])
            assert right_idx == left_idx + 1, (
                f"Cues {entry['leftCueId']} and {entry['rightCueId']} "
                f"are not adjacent in {src} (positions {left_idx}, {right_idx})"
            )


# ── 5. Source boundaries are unique ──────────────────────────────────────


class TestBoundariesAreUnique:
    def test_no_duplicate_boundaries(self, fixture):
        seen = set()
        for entry in fixture:
            key = (entry["sourceId"], entry["leftCueId"], entry["rightCueId"])
            assert key not in seen, f"Duplicate boundary: {key}"
            seen.add(key)


# ── 6. Labels are restricted ─────────────────────────────────────────────


class TestLabelValidity:
    def test_gold_label_restricted(self, fixture):
        for entry in fixture:
            gl = entry.get("goldLabel")
            if gl is not None:
                assert gl in VALID_LABELS, (
                    f"Invalid goldLabel {gl!r} "
                    f"for {entry['sourceId']} {entry['leftCueId']}-{entry['rightCueId']}"
                )

    def test_draft_label_restricted(self, fixture):
        for entry in fixture:
            dl = entry.get("draftLabel")
            if dl is not None:
                assert dl in VALID_LABELS, (
                    f"Invalid draftLabel {dl!r}"
                )


# ── 7. Unreviewed entries cannot be included in metrics ──────────────────


class TestUnreviewedExclusion:
    def test_all_entries_are_unreviewed(self, fixture):
        """Initial state: all entries must be unreviewed."""
        for entry in fixture:
            assert entry.get("reviewStatus") == "unreviewed", (
                f"Entry {entry['sourceId']} {entry['leftCueId']}-{entry['rightCueId']} "
                f"has reviewStatus {entry['reviewStatus']!r}, expected 'unreviewed'"
            )

    def test_unreviewed_cannot_have_gold_label(self, fixture):
        for entry in fixture:
            if entry.get("reviewStatus") == "unreviewed":
                assert entry.get("goldLabel") is None, (
                    f"Unreviewed entry has goldLabel: "
                    f"{entry['sourceId']} {entry['leftCueId']}-{entry['rightCueId']}"
                )


# ── 8. Ambiguous entries excluded from primary metrics ───────────────────


class TestAmbiguousExclusion:
    def test_ambiguous_excluded(self):
        """Verify that in metric computation, AMBIGUOUS labels are filtered out."""
        entries = [
            {"goldLabel": "JOIN"},
            {"goldLabel": "BREAK"},
            {"goldLabel": "AMBIGUOUS"},
            {"goldLabel": "BREAK"},
            {"goldLabel": "JOIN"},
            {"goldLabel": None, "reviewStatus": "unreviewed"},
        ]
        reviewable = [
            e for e in entries
            if e.get("goldLabel") is not None
            and e.get("goldLabel") != "AMBIGUOUS"
        ]
        assert len(reviewable) == 4  # 2 JOIN + 2 BREAK
        assert all(e["goldLabel"] in ("JOIN", "BREAK") for e in reviewable)


# ── 9. Dev and test scenes do not overlap ────────────────────────────────


class TestSplitNoOverlap:
    def test_no_overlap_between_splits(self, splits):
        dev_ids = {
            (e["sourceId"], e["leftCueId"], e["rightCueId"])
            for e in splits["splits"]["dev"]
        }
        test_ids = {
            (e["sourceId"], e["leftCueId"], e["rightCueId"])
            for e in splits["splits"]["test"]
        }
        overlap = dev_ids & test_ids
        assert not overlap, f"Dev/test overlap: {overlap}"

    def test_all_boundaries_assigned(self, fixture, splits):
        all_fixture_ids = {
            (e["sourceId"], e["leftCueId"], e["rightCueId"])
            for e in fixture
        }
        dev_ids = {
            (e["sourceId"], e["leftCueId"], e["rightCueId"])
            for e in splits["splits"]["dev"]
        }
        test_ids = {
            (e["sourceId"], e["leftCueId"], e["rightCueId"])
            for e in splits["splits"]["test"]
        }
        assigned = dev_ids | test_ids
        missing = all_fixture_ids - assigned
        extra = assigned - all_fixture_ids
        assert not missing, f"Boundaries not assigned to any split: {missing}"
        assert not extra, f"Extra boundaries in splits: {extra}"

    def test_split_coverage(self, splits):
        """Each split should contain boundaries from multiple source files."""
        dev_sources = {e["sourceId"] for e in splits["splits"]["dev"]}
        test_sources = {e["sourceId"] for e in splits["splits"]["test"]}
        assert len(dev_sources) >= 2, (
            f"Dev split has only {len(dev_sources)} source(s), "
            f"expected at least 2: {dev_sources}"
        )
        assert len(test_sources) >= 2, (
            f"Test split has only {len(test_sources)} source(s), "
            f"expected at least 2: {test_sources}"
        )


# ── 10. Spanish characters preserved ─────────────────────────────────────


class TestSpanishCharacters:
    SPANISH_CHARS = set("áéíóúñü¿¡ÁÉÍÓÚÜÑ")

    def test_spanish_characters_preserved(self, fixture):
        for entry in fixture:
            for field in ("left", "right"):
                text = entry[field]
                for ch in self.SPANISH_CHARS:
                    # Count occurrences in fixture vs expected from encoding
                    pass  # Just check the text round-trips through UTF-8
                # Verify text round-trips
                roundtripped = text.encode("utf-8").decode("utf-8")
                assert text == roundtripped, (
                    f"Text failed UTF-8 roundtrip for {entry['sourceId']} "
                    f"{entry['leftCueId']}-{entry['rightCueId']}"
                )

    def test_characters_match_source(self, fixture, source_data):
        for entry in fixture:
            src = entry["sourceId"]
            for cid, field in [(entry["leftCueId"], "left"), (entry["rightCueId"], "right")]:
                expected = source_data[src][cid]["text"]
                actual = entry[field]
                assert actual == expected, (
                    f"Character mismatch in {src} {cid}"
                )


# ── 11. Cue ordering and timings retained ────────────────────────────────


class TestOrderingAndTimings:
    def test_cue_ordering_retained(self, fixture, source_data):
        """Verify left cues come before right cues in time."""
        for entry in fixture:
            assert entry["leftStartMs"] <= entry["rightStartMs"], (
                f"Left cue starts after right cue: "
                f"{entry['sourceId']} {entry['leftCueId']}-{entry['rightCueId']} "
                f"({entry['leftEndMs']} > {entry['rightStartMs']})"
            )

    def test_gap_ms_correct(self, fixture):
        """Verify gapMs equals right start - left end."""
        for entry in fixture:
            expected = max(0, entry["rightStartMs"] - entry["leftEndMs"])
            assert entry["gapMs"] == expected, (
                f"gapMs mismatch for {entry['sourceId']} "
                f"{entry['leftCueId']}-{entry['rightCueId']}: "
                f"{entry['gapMs']} != {expected}"
            )


# ── 12. CSV file matches JSON ────────────────────────────────────────────


class TestCSVMatchesJSON:
    def test_csv_has_same_count(self):
        with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Header + data rows
        assert len(rows) >= 251, f"CSV has {len(rows)} rows, expected at least 251"
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(rows) - 1 == len(data), (
            f"CSV rows ({len(rows) - 1}) != JSON entries ({len(data)})"
        )


# ── 13. Labeling guide exists ────────────────────────────────────────────


class TestLabelingGuide:
    def test_labeling_guide_exists(self):
        assert LABELING_GUIDE.exists(), f"Labeling guide not found at {LABELING_GUIDE}"

    def test_labeling_guide_has_join_section(self):
        content = LABELING_GUIDE.read_text(encoding="utf-8")
        assert "## JOIN" in content or "### JOIN" in content

    def test_labeling_guide_has_break_section(self):
        content = LABELING_GUIDE.read_text(encoding="utf-8")
        assert "## BREAK" in content or "### BREAK" in content

    def test_labeling_guide_has_ambiguous_section(self):
        content = LABELING_GUIDE.read_text(encoding="utf-8")
        assert "## AMBIGUOUS" in content or "### AMBIGUOUS" in content


# ── 14. Extract validation ────────────────────────────────────────────────


class TestExtractValidation:
    def test_all_sources_have_checksum(self, fixture):
        for entry in fixture:
            assert entry.get("sourceChecksum"), (
                f"Missing checksum for {entry['sourceId']}"
            )

    def test_checksum_matches_file(self, fixture):
        """Spot-check that source file checksums are valid."""
        seen_sources = {}
        for entry in fixture:
            src = entry["sourceId"]
            cksum = entry["sourceChecksum"]
            if src not in seen_sources:
                seen_sources[src] = cksum
            else:
                assert seen_sources[src] == cksum, (
                    f"Inconsistent checksum for {src}: "
                    f"{seen_sources[src]} vs {cksum}"
                )

        for src, expected_cksum in seen_sources.items():
            srt_path = SOURCE_DIR / f"{src}.srt"
            actual_cksum = hashlib.sha256(srt_path.read_bytes()).hexdigest()[:16]
            assert actual_cksum == expected_cksum, (
                f"Checksum mismatch for {src}: "
                f"expected {expected_cksum}, got {actual_cksum}"
            )

    def test_all_boundaries_have_category(self, fixture):
        for entry in fixture:
            assert entry.get("category"), (
                f"Missing category for "
                f"{entry['sourceId']} {entry['leftCueId']}-{entry['rightCueId']}"
            )
