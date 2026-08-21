"""
fertig.code — die Code-Schicht (aus FORGE übernommen, schlank nachgebaut).

FORGEs legacy code_assembler (2546 Zeilen, tief im Autonomie-Kern verdrahtet)
wird hier als eigenständiger, leichter Pfad abgebildet — gleiche Semantik,
gleiche Daten, keine Fremd-Abhängigkeiten:

  .causal-Wissen (python_stdlib/python_libraries/code_patterns/error_patterns)
        + Fragment-Templates (stdlib_fragments.json)
        -> Assemblierung: Prompt-Token -> Jaro-Winkler-Match gegen
           Fragment-IDs und Triplett-Trigger -> Import-Dedupe -> main()-Wrapper
        -> Sandbox-Ausführung (subprocess, timeout, tempdir)

Der Compounding-Loop: fehlgeschlagene Aufgaben -> Fragment-Lernen aus
Referenzlösungen (lern_triplets/lern_fragment) -> nächster Lauf misst den
Fortschritt. Deterministisch, offline, kein LLM.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import inference
from ._vendor.dotcausal import CausalReader
from .pipeline import _toks

CODE_DIR = Path(__file__).resolve().parent.parent / "data" / "code"

_FRAGMENT_FILES = ["stdlib_fragments.json", "stdlib_network_fragments.json"]
_KNOWLEDGE_FILES = ["python_stdlib.causal", "python_libraries.causal",
                    "code_patterns.causal", "error_patterns.causal"]


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------

def load_fragments(code_dir: Path = CODE_DIR) -> Dict[str, str]:
    """Fragment-ID -> Code-Template (mit {platzhaltern})."""
    out: Dict[str, str] = {}
    for fn in _FRAGMENT_FILES:
        p = code_dir / fn
        if p.exists():
            out.update(json.loads(p.read_text(encoding="utf-8")))
    return out


def _read_forge_causal(path: Path) -> List[dict]:
    """FORGE-Layout: 6-Byte-Magic 'CAUSAL', 2 Bytes Skip, zlib, msgpack."""
    import zlib
    try:
        import msgpack
    except ImportError:  # pragma: no cover
        return []
    raw = path.read_bytes()
    if raw[:6] != b"CAUSAL":
        return []
    try:
        data = msgpack.unpackb(zlib.decompress(raw[8:]), raw=False)
    except Exception:
        return []
    if isinstance(data, dict):
        return data.get("triplets", [])
    if isinstance(data, list):
        return data
    return []


def load_triplets(code_dir: Path = CODE_DIR) -> List[dict]:
    """Alle Code-Wissens-Tripletts (dotcausal- UND FORGE-Format)."""
    out: List[dict] = []
    for fn in _KNOWLEDGE_FILES:
        p = code_dir / fn
        if not p.exists():
            continue
        try:
            trips = CausalReader(str(p)).get_all_triplets()
        except Exception:
            trips = []
        if not trips:
            trips = _read_forge_causal(p)
        out.extend(trips)
    return out


# ---------------------------------------------------------------------------
# Assemblierung
# ---------------------------------------------------------------------------

def _match_score(prompt_tokens: List[str], key: str) -> float:
    """Prompt-Token gegen Fragment-Key/Trigger: Jaro-Winkler, beste Phrase."""
    key_toks = key.replace("_", " ").split()
    best = 0.0
    for k in key_toks:
        for t in prompt_tokens:
            best = max(best, inference.jaro_winkler(t, k))
    return best


def _fragment_code(template) -> str:
    """Fragment-Wert -> Code-String (String direkt, dict via 'code'-Key)."""
    if isinstance(template, str):
        return template
    if isinstance(template, dict):
        return template.get("code", "")
    return str(template)


def assemble(prompt: str, fragments: Dict[str, str],
             triplets: List[dict], top_k: int = 3,
             threshold: float = 0.8) -> Tuple[str, List[str]]:
    """Prompt -> Python-Skript.

    Liefert (code, verwendete_fragment_ids). Deterministisch.
    """
    toks = _toks(prompt)
    scored: List[Tuple[float, str]] = []
    for fid, template in fragments.items():
        s = _match_score(toks, fid)
        if s >= threshold:
            scored.append((s, fid))
    for t in triplets:
        s = _match_score(toks, str(t.get("trigger", "")))
        if s >= threshold:
            scored.append((s, str(t.get("outcome", ""))))
    scored.sort(key=lambda x: -x[0])
    picked = [fid for _, fid in scored[:top_k]]

    imports: List[str] = []
    body: List[str] = []
    used: List[str] = []
    for _, fid in scored[:top_k]:
        if fid in fragments:
            template = _fragment_code(fragments[fid])
            # Platzhalter mit vernünftigen Defaults füllen
            code = re.sub(r"\{(\w+)\}", lambda m: f'"{m.group(1)}"', template)
            for line in code.splitlines():
                if line.startswith("import ") or line.startswith("from "):
                    if line not in imports:
                        imports.append(line)
                else:
                    body.append(line)
            used.append(fid)
        else:
            # Triplett-Outcome ist ein Mechanismus-Text -> als Kommentar
            body.append(f"# {fid}")

    if not used:
        # ehrliche Sackgasse: kein Fragment über der Schwelle
        return ("", [])

    # main()-Wrapper, damit das Skript lauffähig ist
    lines = ["#!/usr/bin/env python3"]
    lines += imports
    lines += [""]
    lines += ["def main():"]
    for line in body:
        lines.append("    " + line if line.strip() else "")
    lines += [""]
    lines += ['if __name__ == "__main__":']
    lines += ["    main()"]
    return "\n".join(lines), used


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

def run_sandbox(code: str, timeout: int = 15,
                stdin_data: str = "") -> Tuple[int, str, str]:
    """Code in einer Sandbox ausführen: tempdir, timeout, kein Netz.

    Rückgabe: (exit_code, stdout, stderr).
    """
    with tempfile.TemporaryDirectory(prefix="fertig_code_") as td:
        script = Path(td) / "main.py"
        script.write_text(code, encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "HOME": td,
               "PYTHONNOUSERSITE": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                input=stdin_data, capture_output=True, text=True,
                timeout=timeout, env=env, cwd=td)
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "TIMEOUT"
        except Exception as e:  # pragma: no cover
            return -2, "", str(e)


# ---------------------------------------------------------------------------
# Compounding: Lernen aus Referenzlösungen
# ---------------------------------------------------------------------------

def learn_fragment(name: str, solution: str, fragments: Dict[str, str]) -> None:
    """Referenzlösung als neues Fragment registrieren (in-memory).

    Der Loop: Aufgabe fehlgeschlagen -> Lösung gelernt -> beim nächsten Mal
    wird das Fragment gematcht. Erst nach dem Evaluations-Split anwenden,
    sonst ist es Leckage.
    """
    fragments[name] = textwrap.dedent(solution).strip()
