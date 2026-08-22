#!/usr/bin/env python3
"""Build IMMER's final offline bundle without touching the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from immer.hf_export import HfExportError, export_hf_poc
except ModuleNotFoundError:  # direct checkout execution without installation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from immer.hf_export import HfExportError, export_hf_poc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build and offline-verify an HF-shaped IMMER PoC. "
            "This command never authenticates or uploads."
        )
    )
    parser.add_argument("output", type=Path, help="new bundle directory")
    parser.add_argument("--manifest", type=Path, help="SHIP-v6 source manifest")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="explicit root containing the digest-matching source artifacts",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="move an existing target to a recoverable .backup path",
    )
    args = parser.parse_args(argv)
    try:
        report = export_hf_poc(
            args.output,
            manifest=args.manifest,
            artifact_root=args.artifact_root,
            replace=args.replace,
        )
    except HfExportError as exc:
        print(
            json.dumps(
                {"status": "error", "reason": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
