"""Migrate Spanish benchmark candidates and report to new schema.

Adds:
- sourceQualityTier, contentStructure, originalSpokenLanguage per candidate
- labelConfidence, reviewerCount, needsSecondReview per candidate
- Regenerates dataset report with new fields and maturity calculation.

Usage:
    python benchmarks/migrate_spanish_schema.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the builder's report function
from benchmarks.spanish_benchmark_lib import generate_dataset_report, load_policy_freeze

BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
CANDIDATES_PATH = BENCHMARK_DIR / "spanish_boundary_candidates.json"
MANIFEST_PATH = BENCHMARK_DIR / "spanish_source_manifest.json"
REPORT_PATH = BENCHMARK_DIR / "spanish_boundary_dataset_report.json"
CSV_PATH = BENCHMARK_DIR / "spanish_boundary_candidates.csv"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("Loading candidates...")
    candidates = load_json(CANDIDATES_PATH)
    print(f"  {len(candidates)} candidates loaded")

    print("Loading source manifest...")
    manifest = load_json(MANIFEST_PATH)
    sources = manifest.get("sources", [])
    manifest_by_id = {m["sourceId"]: m for m in sources}

    # Add new fields to every candidate
    updated = 0
    for entry in candidates:
        src = entry.get("sourceId", "")
        m = manifest_by_id.get(src, {})

        changed = False

        # Source quality tier
        if "sourceQualityTier" not in entry:
            entry["sourceQualityTier"] = m.get("sourceQualityTier", "unknown")
            changed = True

        # Content structure
        if "contentStructure" not in entry:
            entry["contentStructure"] = m.get("contentStructure", "unknown")
            changed = True

        # Original spoken language
        if "originalSpokenLanguage" not in entry:
            entry["originalSpokenLanguage"] = m.get("originalSpokenLanguage", "unknown")
            changed = True

        # Label confidence fields
        if "labelConfidence" not in entry:
            entry["labelConfidence"] = None
            changed = True

        if "reviewerCount" not in entry:
            entry["reviewerCount"] = 0
            changed = True

        if "needsSecondReview" not in entry:
            entry["needsSecondReview"] = False
            changed = True

        if changed:
            updated += 1

    print(f"  {updated} candidates updated")

    # Save updated candidates
    print(f"\nSaving {len(candidates)} candidates to {CANDIDATES_PATH}")
    save_json(CANDIDATES_PATH, candidates)

    # Regenerate dataset report
    from pathlib import Path as _Path
    policy_freeze_path = BENCHMARK_DIR / "spanish_policy_freeze.json"
    policy_freeze = load_policy_freeze(_Path(policy_freeze_path))
    print("\nRegenerating dataset report...")
    report = generate_dataset_report(candidates, sources, [], policy_freeze)
    print(f"  benchmarkOperationallyReady: {report.get('benchmarkOperationallyReady')}")
    print(f"  nativeCoverageTargetMet: {report.get('nativeCoverageTargetMet')}")
    maturity = report.get("maturity", {})
    print(f"  maturity: {maturity.get('maturityLevel')} ({maturity.get('reviewedNonAmbiguous')} reviewed non-ambiguous)")

    print(f"\nSaving report to {REPORT_PATH}")
    save_json(REPORT_PATH, report)

    # Do not modify manifest with readiness fields — source manifest
    # contains immutable source provenance only.

    # Print stratum summary
    strata = report.get("metricStrata", {})
    print("\nMetric strata:")
    for key, values in strata.items():
        print(f"  {key}: {values}")


if __name__ == "__main__":
    main()
