"""Strong source verification for Spanish benchmark sources.

Usage:
    python -m benchmarks.verify_spanish_sources --input benchmark-source

Verifies local source files against the provenance manifest using
full SHA-256 checksums. Fails clearly when a source is missing or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _full_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(
    source_dir: Path,
    manifest_path: Path,
    verbose: bool = True,
) -> bool:
    """Verify all manifest entries exist and match checksums.

    Returns True if all sources pass, False otherwise.
    """
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return False

    if not source_dir.is_dir():
        print(f"ERROR: Source directory not found at {source_dir}")
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_version = manifest.get("manifestVersion", "unknown")
    if verbose:
        print(f"Spanish Source Verification v{manifest_version}")
        print(f"  Manifest: {manifest_path}")
        print(f"  Source dir: {source_dir}")
        print()

    sources = manifest.get("sources", [])
    if not sources:
        print("ERROR: No sources in manifest")
        return False

    all_ok = True

    for src in sources:
        source_id = src["sourceId"]
        expected_checksum = src["fullSha256"]
        expected_cue_count = src.get("cueCount", 0)

        srt_path = source_dir / f"{source_id}.srt"

        if not srt_path.exists():
            print(f"  FAIL: {source_id} — source file MISSING at {srt_path}")
            all_ok = False
            continue

        actual_checksum = _full_checksum(srt_path)
        if actual_checksum != expected_checksum:
            print(f"  FAIL: {source_id} — checksum MISMATCH")
            print(f"         expected: {expected_checksum}")
            print(f"         actual:   {actual_checksum}")
            all_ok = False
            continue

        # Count cue blocks
        content = srt_path.read_text(encoding="utf-8-sig")
        import re
        blocks = re.split(r"\r?\n\s*\r?\n", content.strip("\r\n"))
        actual_cue_count = len(blocks)

        cue_ok = actual_cue_count == expected_cue_count
        if not cue_ok:
            print(f"  WARN: {source_id} — cue count MISMATCH "
                  f"(expected {expected_cue_count}, got {actual_cue_count})")
            all_ok = False

        if verbose:
            status = "PASS" if actual_checksum == expected_checksum else "FAIL"
            print(f"  {status}: {source_id}")
            print(f"         SHA256: {actual_checksum[:20]}...")
            print(f"         Cues: {actual_cue_count}")
            print(f"         Type: {src.get('contentType', '?')}")
            print(f"         Lang: {src.get('originalSpokenLanguage', '?')}")
            print(f"         Variant: {src.get('spanishVariant', '?')}")

    # Check for unregistered source files
    for srt_path in sorted(source_dir.rglob("*.srt")):
        src_id = srt_path.stem
        if src_id not in {s["sourceId"] for s in sources}:
            print(f"  WARN: {src_id} — source file has NO manifest entry")

    print()
    if all_ok:
        print("All sources verified successfully.")
    else:
        print("Some sources FAILED verification.")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify benchmark source files against provenance manifest",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmark-source"),
        help="Directory containing source SRT files (default: benchmark-source)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent / "spanish_source_manifest.json",
        help="Path to provenance manifest (default: auto)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    success = verify_sources(
        source_dir=args.input,
        manifest_path=args.manifest,
        verbose=not args.quiet,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
