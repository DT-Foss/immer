#!/usr/bin/env python3
"""Check local IMMER capabilities without downloading models or caches."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from immer.adapters import DEFAULT_FERTIG_ROOT, DEFAULT_FLCA_ROOT, DEFAULT_O1_ROOT, FertigAdapter


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def check(label: str, ok: bool, detail: str = "") -> bool:
    marker = "✓" if ok else "○"
    suffix = f" — {detail}" if detail else ""
    print(f"{marker} {label}{suffix}")
    return ok


def main() -> int:
    fertig_root = _path_from_env("IMMER_FERTIG_ROOT", DEFAULT_FERTIG_ROOT)
    o1_root = _path_from_env("IMMER_O1_ROOT", DEFAULT_O1_ROOT)
    flca_root = _path_from_env("IMMER_FLCA_ROOT", DEFAULT_FLCA_ROOT)
    fable_root = Path(os.environ.get("IMMER_FABLE_ROOT", "/Users/bhkmie/self-verification_fable")).expanduser()
    organ_root = fable_root / "mirkoNN-analysis" / "s3" / "wesen_organbank.py"
    crsa_root = fable_root / "neue attention" / "operators.py"
    qad_root = Path(os.environ.get("IMMER_QAD_ROOT", "/Users/bhkmie/Downloads/Forschung/Liquid-QAD")).expanduser()

    core = check("FERTIG core available", FertigAdapter(fertig_root).available, str(fertig_root))
    check("o1-state runtime available", o1_root.exists(), str(o1_root))
    check("CRSA operators available", crsa_root.is_file(), str(crsa_root))
    check("OrganBank available", organ_root.is_file(), str(organ_root))
    check("FLCA compiler available", (flca_root / "components").is_dir(), str(flca_root))
    check("FOSS-QAD adapter available", (qad_root / "qad").is_dir(), str(qad_root))

    model = os.environ.get("IMMER_LFM_MODEL")
    if model:
        check("LFM model configured", Path(model).expanduser().exists(), model)
    else:
        print("○ LFM2.5 model not configured — set IMMER_LFM_MODEL=/path/to/model")
    print("○ Donor model optional")
    print("○ Redmi backend offline")
    print("IMMER core ready." if core else "IMMER core ready in bootstrap-only mode; FERTIG is external.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
