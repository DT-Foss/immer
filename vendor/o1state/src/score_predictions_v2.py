#!/usr/bin/env python3
"""
Auto-Scorer v2 — audit analysis/PREDICTIONS.md against results/*.json over the
FULL register (P1..P60+), mechanically.
============================================================================
Unlike src/score_predictions.py (a hand-curated P1..P20 registry, kept intact
as the reference), this scorer PARSES PREDICTIONS.md:

  1. every P-number, its registration text, and its scoring/verdict blocks;
  2. its status (registered / scored / withdrawn / amended / in_flight);
  3. the result-artifact paths its text names.

For every scored P whose artifact is machine-readable, it then checks the
CLAUSE-BEARING fields the artifacts carry by convention (p52c_pass, p46a_delta,
p33_scoring.a_*.pass, gate_frac_cum, ...) against the numbers the register text
claims. Three outcomes per check:

  MATCH        — register value and artifact value agree (within tolerance);
  MISMATCH     — they genuinely diverge (BOTH values reported — a first-class
                 finding, never smoothed over);
  UNPARSEABLE  — the clause is not machine-checkable from the text (no number
                 could be tied to the field) — listed honestly, never guessed.

Runs in flight (a status/phase == "running" artifact — e.g. pos_d1024) are
marked IN_FLIGHT and skipped, not scored.

Writes results/scorer_audit.json and prints a compact table (MISMATCHES first).
Exit code is always 0 — this scorer reports, it does not gate.

Usage:  python src/score_predictions_v2.py [--predictions analysis/PREDICTIONS.md]
                                            [--results-dir results]
                                            [--out results/scorer_audit.json]
"""
import argparse
import json
import os
import re
import sys


# ─────────────────────────────────────────────────────────────────────────
#  Markdown parsing
# ─────────────────────────────────────────────────────────────────────────
# A P-number token, e.g. "P52", "P39a", "P11-revision". We normalise scoring
# blocks to their integer base (P52a SCORED -> P52) for status roll-up, but we
# keep sub-labels for the record.
P_TOKEN = r"P\d+[a-z]?(?:-[a-z]+)?"

# A registration bullet: line begins (after "- ") with **P52 — ...** or
# **P52 ...:** . We treat "— <title>" style as the registration anchor.
# The bold title can wrap across newlines ("**P56 — the width law's fourth\n
# point, ...**"), so the title body uses [\s\S] (dot that also spans \n) and is
# non-greedy up to the closing **.
RE_REGISTER = re.compile(
    r"^[ \t]*[-*][ \t]+\*\*(?P<pid>P\d+[a-z]?)[ \t]*(?:—|-|–)[ \t]*"
    r"(?P<title>[\s\S]+?)\*\*",
    re.MULTILINE,
)

# A scoring/verdict block header, appearing anywhere (often NOT under its own
# registration bullet). Examples seen in the file:
#   **P47 SCORED (2026-08-05 night, results/knowledge_file.json...):**
#   **P45 FINAL (2026-08-05 evening, ...):**
#   **P45 DECISIVE 2 — THE WIDTH LAW STANDS (...)**
#   **P51 WITHDRAWN (2026-08-05...)**
#   **P54 AMENDED before the run (...)**
#   **P39a SCORED (day 4 ~09:15): CONFIRMED ...**
#   **P23 SCORED (results/state_weight_swap.json, ...):**
RE_VERDICT_HEAD = re.compile(
    r"\*\*(?P<pid>P\d+[a-z]?)\s+"
    r"(?P<kind>SCORED|FINAL|DECISIVE|INTERIM|WITHDRAWN|AMENDED|EXCEEDED|RE-LOCK|SCORED FINAL)"
    r"(?P<rest>[^*]*)\*\*",
    re.MULTILINE,
)

# Headerless verdict blocks: "**SCORED 2026-07-24 (...)**" or
# "**AMENDED before the sweep (...)**" with NO P-number in the bold header.
# These belong to the immediately preceding registration bullet, and dropping
# them silently miscounts scored predictions as merely registered.
RE_VERDICT_HEADLESS = re.compile(
    r"\*\*(?P<kind>SCORED|FINAL|DECISIVE|INTERIM|WITHDRAWN|AMENDED|EXCEEDED)"
    r"(?P<rest>[^*]*)\*\*",
    re.MULTILINE,
)

# result-artifact paths anywhere in a chunk of text.
RE_ARTIFACT = re.compile(r"results/[A-Za-z0-9_./-]+\.jsonl?")

# Status verdict words we roll up from a scoring block's body.
VERDICT_WORDS = ["CONFIRMED", "FALSIFIED", "PARTIAL", "WITHDRAWN", "AMENDED",
                 "EXCEEDED", "PASS", "PENDING"]


def base_pid(pid):
    """P52a -> 52 ; P11-revision -> 11 ; P39 -> 39."""
    m = re.match(r"P(\d+)", pid)
    return int(m.group(1)) if m else None


def split_blocks(text):
    """Return a list of dicts (pid, kind, start, end, body) for every verdict
    block, where body runs until the next verdict head / registration / EOF.

    Two header forms are captured:
      - explicit "**P37 SCORED ...**" (pid taken from the header);
      - headerless "**SCORED ...**" (pid inherited from the nearest preceding
        registration bullet — these sit directly under their own bullet).
    """
    regs = list(RE_REGISTER.finditer(text))
    reg_starts = [(m.start(), m.group("pid")) for m in regs]

    def preceding_pid(pos):
        pid = None
        for s, p in reg_starts:
            if s < pos:
                pid = p
            else:
                break
        return pid

    explicit = list(RE_VERDICT_HEAD.finditer(text))
    explicit_spans = {(m.start(), m.end()) for m in explicit}

    headless = []
    for m in RE_VERDICT_HEADLESS.finditer(text):
        # skip if this is actually the tail of an explicit "P37 SCORED" match
        if any(s <= m.start() < e for (s, e) in explicit_spans):
            continue
        headless.append(m)

    all_heads = [("explicit", m) for m in explicit] + [("headless", m) for m in headless]
    all_heads.sort(key=lambda t: t[1].start())

    head_starts = [m.start() for _, m in all_heads]
    boundaries = sorted(set(head_starts + [m.start() for m in regs] + [len(text)]))

    blocks = []
    for kind_tag, h in all_heads:
        start = h.start()
        nxt = min((b for b in boundaries if b > start), default=len(text))
        body = text[h.end():nxt]
        if kind_tag == "explicit":
            pid = h.group("pid")
        else:
            pid = preceding_pid(start)
            if pid is None:
                continue  # a verdict word with no owning registration; skip
        blocks.append({
            "pid": pid,
            "kind": h.group("kind"),
            "head_rest": h.group("rest").strip(" :—-–\t"),
            "start": start,
            "end": nxt,
            "body": body,
        })
    return blocks


def registration_spans(text):
    """Map base P-number -> the registration text (from its bullet to the next
    bullet/head). Used to pull the CLAIMED numbers and the artifact paths."""
    regs = list(RE_REGISTER.finditer(text))
    heads = list(RE_VERDICT_HEAD.finditer(text))
    boundaries = sorted(set([m.start() for m in regs] + [m.start() for m in heads] + [len(text)]))
    out = {}
    for r in regs:
        pid = r.group("pid")
        n = base_pid(pid)
        start = r.start()
        nxt = min((b for b in boundaries if b > start), default=len(text))
        span_text = text[start:nxt]
        # keep the FIRST registration for a base number (amendments come later)
        if n not in out:
            out[n] = {"pid": pid, "title": r.group("title").strip(),
                      "text": span_text}
    return out


def rollup_status(base_n, blocks_for_n):
    """Given all verdict blocks whose base P-number == base_n, decide a status."""
    if not blocks_for_n:
        return "registered"
    kinds = {b["kind"] for b in blocks_for_n}
    if "WITHDRAWN" in kinds:
        return "withdrawn"
    # AMENDED is a pre-data edit; if the same P was later SCORED, scored wins.
    scored_like = {"SCORED", "FINAL", "DECISIVE", "EXCEEDED", "SCORED FINAL",
                   "RE-LOCK", "INTERIM"}
    if kinds & scored_like:
        return "scored"
    if "AMENDED" in kinds:
        return "amended"
    return "scored"


def verdict_label(blocks_for_n):
    """Extract the human verdict words present in the scoring bodies+heads."""
    labels = []
    for b in blocks_for_n:
        hay = (b["head_rest"] + " " + b["body"]).upper()
        for w in VERDICT_WORDS:
            if re.search(r"\b" + w + r"\b", hay):
                labels.append(w)
    # de-dup preserving order
    seen, out = set(), []
    for w in labels:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Artifact clause extraction
# ─────────────────────────────────────────────────────────────────────────
CLAUSE_SUFFIXES = ("pass", "delta", "ratio", "found", "exact", "drop",
                   "frac", "gap", "recovery", "forgetting", "plasticity")


def flatten(obj, prefix=""):
    """Yield (dotted_path, value) for every scalar leaf in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}{i}.")
    else:
        yield prefix.rstrip("."), obj


RE_ARRAY_INDEX = re.compile(r"\.\d+\.")


def clause_fields_for_p(artifact, base_n):
    """Find clause-bearing leaves that belong to prediction base_n.

    Two conventions in the artifacts:
      flat:    p52c_pass, p46a_delta, p48c_found, gate_frac_cum, ...
      nested:  p33_scoring.a_forgetting_leq_R3....pass, p20_scoring.*, p49.p49a_pass
    A leaf belongs to base_n if its path contains a "p<base_n>" token (with an
    optional single sub-letter) AND its final key ends in a clause suffix.

    Array elements (paths with a ".<int>." segment, e.g. the per-entry
    provenance rows p52c.picks.0.exact) are EXCLUDED — only the aggregated,
    named clause fields are checkable against a register sentence. Scoring
    per row against one text token is what produced false mismatches in v1.
    """
    pnum = str(base_n)
    # match p52, p52c, p52a_pass, p52_scoring, p52. ...
    ptok = re.compile(rf"(^|[^0-9a-z])p{pnum}(?![0-9])", re.IGNORECASE)
    out = {}
    for path, val in flatten(artifact):
        if not ptok.search(path):
            continue
        if RE_ARRAY_INDEX.search("." + path + "."):
            continue  # skip per-element array leaves; keep only aggregates
        last = path.split(".")[-1]
        if last.endswith(CLAUSE_SUFFIXES) or last in ("pass",):
            out[path] = val
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Claim verification (text number  vs  artifact field)
# ─────────────────────────────────────────────────────────────────────────
# a signed decimal, tolerating a leading + or − (unicode minus too)
NUMBER = r"[+\-−]?\d+(?:\.\d+)?"

# The guiding rule of the verifier: report MATCH only when the artifact value
# is unambiguously present in the register text; report MISMATCH only when a
# number is clearly bound to THIS clause and genuinely differs; otherwise
# UNPARSEABLE. Never pick a "nearest" number and call the gap a mismatch —
# register sentences routinely carry the bar AND the measured value side by
# side, and guessing between them manufactures false findings.


def parse_number(s):
    return float(s.replace("−", "-"))


def value_tokens(value):
    """Printed forms of a numeric artifact value that might appear in prose,
    specific enough not to collide with unrelated numbers:
      - decimal at 4,3,2 dp  (NOT 1 dp: '0.1' is too short and matches noise
        like '0.011 GB' or '0.1×'); an integer-valued number also yields its
        bare integer form;
      - the ×100 percent form at 2,1,0 dp (0.6332 -> 63.32/63.3/63), which is
        only accepted when adjacent to a '%' sign, so it is self-guarding.
    Returns a list of (token_string, kind)."""
    toks = []
    av = abs(value)
    for nd in (4, 3, 2):
        toks.append((f"{av:.{nd}f}", "decimal"))
    if float(av).is_integer():
        toks.append((str(int(av)), "decimal"))
    pct = av * 100.0
    for nd in (2, 1, 0):
        toks.append((f"{pct:.{nd}f}", "percent"))
    # de-dup preserving order
    seen, out = set(), []
    for t, k in toks:
        if t not in seen:
            seen.add(t)
            out.append((t, k))
    return out


def text_contains_token(text, token, kind):
    """True if `token` appears in text as a standalone number (percent form
    must be adjacent to a % sign to count, so '13.5' matches only for
    '13.5%')."""
    norm = text.replace("−", "-")
    if kind == "percent":
        # require a % within a few chars after the number, and no digit/decimal
        # immediately before (so '63' does not match inside '1963' or '0.63').
        for m in re.finditer(re.escape(token), norm):
            before = norm[m.start() - 1] if m.start() > 0 else " "
            tail = norm[m.end():m.end() + 3]
            if before not in "0123456789." and "%" in tail:
                return True
        return False
    # decimal: not immediately adjacent to another digit on either side
    for m in re.finditer(re.escape(token), norm):
        before = norm[m.start() - 1] if m.start() > 0 else " "
        after = norm[m.end():m.end() + 1]
        if before not in "0123456789" and not after.isdigit():
            return True
    return False


def clause_context(text, clause_letter, span=160):
    """Return the substring around an explicit '(<letter>)' marker, so number
    lookups are scoped to the right clause. Falls back to whole text."""
    if not clause_letter:
        return text
    m = re.search(rf"\(\s*{clause_letter}\s*\)", text)
    if not m:
        # some artifacts label clauses R1/R2/R3 or a_/b_ — try those too
        m = re.search(rf"\b{clause_letter.upper()}\d?\b", text)
    if not m:
        return None
    return text[m.start():m.start() + span]


# Verdict vocabulary, mapped to a canonical polarity. FAILED/FAIL are common
# informal synonyms for FALSIFIED in the scoring prose ("(a) FAILED as
# measured"); PASS/CONFIRMED/EXCEEDED are positive.
VERDICT_CANON = {
    "CONFIRMED": "CONFIRMED", "PASS": "PASS", "EXCEEDED": "EXCEEDED",
    "FALSIFIED": "FALSIFIED", "FAILED": "FALSIFIED", "FAIL": "FALSIFIED",
    "PARTIAL": "PARTIAL",
}
VERDICT_ALT = r"(CONFIRMED|FALSIFIED|FAILED|FAIL|PARTIAL|PASS|EXCEEDED)"


def clause_verdict_word(text, clause_letter):
    """Find the verdict word bound to a clause, e.g. '(a) PASS', '(b)+(d)
    FALSIFIED', '(c) CONFIRMED', '(a) FAILED'. Returns a CANONICAL word
    (FAILED->FALSIFIED) or None. Ambiguity (>1 distinct global verdict and no
    clause binding) yields None."""
    if clause_letter:
        # a clause letter may share a verdict with siblings: "(b)+(d) FALSIFIED"
        for m in re.finditer(r"\(([a-z](?:\)\s*[+,/]?\s*\([a-z])*)\)\s*"
                             r"(?:[A-Za-z ,\-]{0,24}?)\b" + VERDICT_ALT + r"\b", text):
            letters = re.findall(r"[a-z]", m.group(1))
            if clause_letter in letters:
                return VERDICT_CANON[m.group(2)]
        # regime label form: "R3 (gating+dosed replay) forgets ... CONFIRMED"
        # or "Clause FALSIFIED: R2's ..." — bind by the label token proximity.
        if re.match(r"r\d", clause_letter):
            lab = clause_letter.upper()
            for m in re.finditer(VERDICT_ALT + r"[^.]{0,80}?\b" + lab + r"\b", text):
                return VERDICT_CANON[m.group(1)]
            for m in re.finditer(r"\b" + lab + r"\b[^.]{0,80}?" + VERDICT_ALT, text):
                return VERDICT_CANON[m.group(2)]
    # global fallback: exactly one DISTINCT canonical verdict in the whole block
    found = {VERDICT_CANON[w] for w in VERDICT_CANON
             if re.search(r"\b" + w + r"\b", text)}
    if len(found) == 1:
        return next(iter(found))
    return None


def clause_letter_of(path):
    """Infer the clause label a field belongs to, across conventions:
      p52c_pass                          -> 'c'
      p33_scoring.a_forgetting_leq...pass-> 'a'   (named sub-dict starts a_/b_)
      ....R2_forgets_less_than_R1.pass   -> 'r2'  (regime label)
    Returns a lowercase label or None.
    """
    # direct p<N><letter>
    m = re.search(r"p\d+([a-z])(?:_|\.|$)", path, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # any dotted segment that starts with a single letter + underscore (a_, b_)
    for seg in path.split("."):
        m = re.match(r"([a-z])_[a-z]", seg)
        if m:
            return m.group(1).lower()
    # a regime label like R2, R3, W, C as its own segment prefix
    for seg in path.split("."):
        m = re.match(r"(R\d|[WC])(?:_|$)", seg)
        if m:
            return m.group(1).lower()
    return None


def verify_field(path, value, reg_text, score_text):
    """Return (verdict, register_value, note) for one clause field.

    Conservative by design: MATCH on unambiguous presence, MISMATCH only on a
    clause-bound contradiction, UNPARSEABLE otherwise.
    """
    last = path.split(".")[-1]
    text = (score_text or "") + "\n" + (reg_text or "")
    clause_letter = clause_letter_of(path)

    # ---- boolean pass fields -------------------------------------------------
    if last == "pass" or last.endswith("_pass"):
        want = bool(value)
        vw = clause_verdict_word(text, clause_letter)
        if vw is None:
            return "UNPARSEABLE", None, "no verdict word bound to this clause"
        if vw == "PARTIAL":
            return "UNPARSEABLE", "PARTIAL", "text says PARTIAL — clause truth ambiguous"
        expect_true = vw in ("CONFIRMED", "PASS", "EXCEEDED")
        if expect_true == want:
            return "MATCH", f"{vw}->{expect_true}", ""
        return "MISMATCH", f"{vw}->{expect_true}", f"artifact pass={want}"

    # ---- provenance "N/M" fields (found / exact) ----------------------------
    if last in ("found", "exact") or last.endswith(("_found", "_exact")):
        ctx = clause_context(text, clause_letter) or text
        m = re.search(r"(\d+)\s*/\s*(\d+)", ctx)
        if not m:
            return "UNPARSEABLE", None, "no N/M token near this clause"
        claim = f"{m.group(1)}/{m.group(2)}"
        art = value.strip() if isinstance(value, str) else str(value)
        if claim == art:
            return "MATCH", claim, ""
        # if artifact is an int count and claim is k/M with k == count
        if not isinstance(value, str) and str(value) == m.group(1):
            return "MATCH", claim, f"artifact count {value} == {claim}"
        return "MISMATCH", claim, f"artifact={art}"

    # ---- numeric fields (delta/ratio/drop/frac/...) -------------------------
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # A verbatim appearance of the value (decimal or ×100 percent form) is
        # an unambiguous MATCH. Search the clause context first (tighter
        # provenance), then the whole block. We never pick a nearest number, so
        # widening the search cannot manufacture a mismatch — only find a hit.
        ctx = clause_context(text, clause_letter, span=320)
        for scope, scope_name in ((ctx, "clause context"), (text, "block")):
            if scope is None:
                continue
            for token, kind in value_tokens(value):
                if text_contains_token(scope, token, kind):
                    shown = token + ("%" if kind == "percent" else "")
                    return "MATCH", shown, f"artifact {value} present as {shown} in {scope_name}"
        return "UNPARSEABLE", None, f"artifact {value} not found verbatim in text"

    return "UNPARSEABLE", None, f"field type {type(value).__name__} not checkable"


# ─────────────────────────────────────────────────────────────────────────
#  IN_FLIGHT detection
# ─────────────────────────────────────────────────────────────────────────
def scan_running_runs(results_dir):
    """One pass over results/*_status.json: return {tag: filename} for every
    run whose phase/status reports running. Used to flag a prediction whose
    text names a still-in-flight run by tag (e.g. 'd1024') even when it cites
    no results/*.json path yet."""
    running = {}
    if not os.path.isdir(results_dir):
        return running
    for fn in os.listdir(results_dir):
        if not fn.endswith("_status.json"):
            continue
        try:
            with open(os.path.join(results_dir, fn)) as f:
                d = json.load(f)
        except Exception:
            continue
        state = str(d.get("phase", d.get("status", ""))).lower()
        if state in ("running", "in_flight", "in-flight"):
            tag = (d.get("config", {}) or {}).get("tag") or fn[:-len("_status.json")]
            running[str(tag)] = fn
    return running


def text_names_running_run(text, running):
    """Return the filename of a running run whose tag is named in text, else
    None. Tags are matched as whole tokens (so 'd1024' does not match inside
    'd10240')."""
    for tag, fn in running.items():
        if re.search(rf"(^|[^0-9A-Za-z]){re.escape(tag)}([^0-9A-Za-z]|$)", text):
            return fn
    return None


def artifact_in_flight(results_dir, path):
    """path is like 'results/pos_d1024_status.json' or a bare stem match.
    Returns True if the artifact (or a sibling *_status.json) reports running."""
    candidates = [path]
    stem = os.path.basename(path)
    # a run named results/foo.json often has a results/foo_status.json partner
    if stem.endswith(".json") and not stem.endswith("_status.json"):
        candidates.append(path[:-5] + "_status.json")
    for c in candidates:
        full = c if os.path.isabs(c) else os.path.join(os.getcwd(), c)
        if not os.path.exists(full):
            full = os.path.join(results_dir, os.path.basename(c))
        if os.path.exists(full):
            try:
                with open(full) as f:
                    d = json.load(f)
                if str(d.get("phase", "")).lower() in ("running", "in_flight", "in-flight"):
                    return True, os.path.basename(full)
                if str(d.get("status", "")).lower() in ("running", "in_flight", "in-flight"):
                    return True, os.path.basename(full)
            except Exception:
                pass
    return False, None


def load_artifact(results_dir, path):
    """Load a results/*.json (jsonl not loaded as a dict — returned as marker)."""
    full = path if os.path.isabs(path) else path
    if not os.path.exists(full):
        full = os.path.join(results_dir, os.path.basename(path))
    if not os.path.exists(full):
        return None, "missing"
    if full.endswith(".jsonl"):
        return None, "jsonl"          # traces: not a clause-bearing dict
    try:
        with open(full) as f:
            return json.load(f), "ok"
    except Exception as e:
        return None, f"error: {e}"


# ─────────────────────────────────────────────────────────────────────────
#  Main audit
# ─────────────────────────────────────────────────────────────────────────
def audit(predictions_path, results_dir):
    with open(predictions_path) as f:
        text = f.read()

    regs = registration_spans(text)
    blocks = split_blocks(text)
    running_runs = scan_running_runs(results_dir)

    # group scoring blocks by base P-number
    by_n = {}
    for b in blocks:
        n = base_pid(b["pid"])
        by_n.setdefault(n, []).append(b)

    all_ns = sorted(set(regs) | set(by_n))
    audit_rows = {}

    for n in all_ns:
        reg = regs.get(n, {})
        reg_text = reg.get("text", "")
        blks = by_n.get(n, [])
        status = rollup_status(n, blks)
        verdicts = verdict_label(blks)

        # artifacts named in registration + scoring text
        pool_text = reg_text + "\n" + "\n".join(b["body"] + " " + b["head_rest"] for b in blks)
        artifacts = sorted(set(RE_ARTIFACT.findall(pool_text)))

        # IN_FLIGHT check: either a cited artifact reports running, or the
        # prediction's text names a still-running run by tag (e.g. 'd1024').
        in_flight_hit = None
        for a in artifacts:
            hit, which = artifact_in_flight(results_dir, a)
            if hit:
                in_flight_hit = which
                break
        if in_flight_hit is None:
            in_flight_hit = text_names_running_run(pool_text, running_runs)

        row = {
            "pid": f"P{n}",
            "title": reg.get("title", ""),
            "status": status,
            "verdict_words": verdicts,
            "artifacts_named": artifacts,
            "artifacts_found": [],
            "checks": [],
            "check_summary": {"MATCH": 0, "MISMATCH": 0, "UNPARSEABLE": 0},
            "coverage": "none",
        }

        if in_flight_hit and status in ("registered", "amended"):
            row["status"] = "in_flight"
            row["note"] = f"artifact {in_flight_hit} reports phase=running — skipped"
            audit_rows[n] = row
            continue

        if status in ("withdrawn",):
            row["coverage"] = "n/a"
            audit_rows[n] = row
            continue

        if status in ("registered", "amended"):
            # no data expected yet
            row["coverage"] = "prose-only (not yet scored)"
            audit_rows[n] = row
            continue

        # SCORED: try to verify against a machine-readable artifact
        score_text = "\n".join(b["body"] for b in blks)
        json_artifacts = [a for a in artifacts if a.endswith(".json")]
        # If a non-smoke artifact exists for this P, drop the *_smoke.json
        # partners: they are interim runs and their numbers do not match the
        # full-run scoring text (double-checking them only manufactures noise).
        if any("smoke" not in os.path.basename(a) for a in json_artifacts):
            json_artifacts = [a for a in json_artifacts
                              if "smoke" not in os.path.basename(a)]
        machine_checked = 0
        any_json_loaded = False

        for a in json_artifacts:
            data, st = load_artifact(results_dir, a)
            if st == "ok":
                any_json_loaded = True
                row["artifacts_found"].append(os.path.basename(a))
                fields = clause_fields_for_p(data, n)
                for path, val in sorted(fields.items()):
                    verdict, reg_val, note = verify_field(path, val, reg_text, score_text)
                    row["checks"].append({
                        "claim": path.split(".")[-1],   # the clause being checked
                        "artifact": os.path.basename(a),
                        "artifact_field": path,
                        "artifact_value": val,
                        "register_value": reg_val,
                        "verdict": verdict,
                        "note": note,
                    })
                    if verdict in ("MATCH", "MISMATCH"):
                        machine_checked += 1
            elif st == "jsonl":
                row["artifacts_found"].append(os.path.basename(a) + " (jsonl trace)")

        n_checks = len(row["checks"])
        n_match = sum(1 for c in row["checks"] if c["verdict"] == "MATCH")
        n_mis = sum(1 for c in row["checks"] if c["verdict"] == "MISMATCH")
        n_unp = sum(1 for c in row["checks"] if c["verdict"] == "UNPARSEABLE")

        if machine_checked > 0 and n_unp == 0:
            row["coverage"] = "full"
        elif machine_checked > 0:
            row["coverage"] = "partial"
        elif any_json_loaded and n_checks > 0:
            row["coverage"] = "prose-only (clauses not machine-readable)"
        elif any_json_loaded:
            row["coverage"] = "prose-only (no clause fields in artifact)"
        else:
            row["coverage"] = "prose-only (no loadable json artifact)"

        row["check_summary"] = {"MATCH": n_match, "MISMATCH": n_mis,
                                "UNPARSEABLE": n_unp}
        audit_rows[n] = row

    return audit_rows


def summarize(rows):
    cov = {"full": 0, "partial": 0, "prose": 0, "in_flight": 0,
           "withdrawn": 0, "registered": 0}
    for r in rows.values():
        c = r["coverage"]
        s = r["status"]
        if s == "in_flight":
            cov["in_flight"] += 1
        elif s == "withdrawn":
            cov["withdrawn"] += 1
        elif s in ("registered", "amended"):
            cov["registered"] += 1
        elif c == "full":
            cov["full"] += 1
        elif c == "partial":
            cov["partial"] += 1
        else:
            cov["prose"] += 1
    return cov


def main():
    ap = argparse.ArgumentParser(
        description="Audit PREDICTIONS.md against results/*.json over the full register.")
    ap.add_argument("--predictions", default=os.path.join("analysis", "PREDICTIONS.md"))
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=os.path.join("results", "scorer_audit.json"))
    args = ap.parse_args()

    rows = audit(args.predictions, args.results_dir)
    cov = summarize(rows)

    # collect all mismatches
    mismatches = []
    for n, r in rows.items():
        for c in r.get("checks", []):
            if c["verdict"] == "MISMATCH":
                mismatches.append((r["pid"], c))

    # ── stdout: MISMATCHES first ──
    print("=" * 82)
    print("AUTO-SCORER v2 — PREDICTIONS.md audited against results/*.json (full register)")
    print("=" * 82)
    print(f"predictions parsed: {len(rows)} P-numbers")
    print(f"coverage: full={cov['full']}  partial={cov['partial']}  "
          f"prose-only={cov['prose']}  in_flight={cov['in_flight']}  "
          f"withdrawn={cov['withdrawn']}  registered/amended={cov['registered']}")
    print("-" * 82)

    if mismatches:
        print(f"\n### MISMATCHES ({len(mismatches)}) — register text vs artifact ###")
        for pid, c in mismatches:
            print(f"  [{pid}] {c['artifact']} :: {c['artifact_field']}")
            print(f"        register={c['register_value']!r}  "
                  f"artifact={c['artifact_value']!r}  ({c['note']})")
    else:
        print("\n### MISMATCHES: none ###")

    print("\n### PER-PREDICTION ###")
    for n in sorted(rows):
        r = rows[n]
        cs = r.get("check_summary")
        cs_str = ""
        if cs:
            cs_str = f"  checks[M{cs['MATCH']}/X{cs['MISMATCH']}/U{cs['UNPARSEABLE']}]"
        verd = ",".join(r["verdict_words"]) if r["verdict_words"] else "-"
        print(f"  {r['pid']:<5} {r['status']:<11} cov={r['coverage']:<42} "
              f"verdict={verd:<24}{cs_str}")

    # ── write audit json ──
    payload = {
        "generated_by": "src/score_predictions_v2.py",
        "predictions_file": args.predictions,
        "n_predictions": len(rows),
        "coverage_summary": cov,
        "n_mismatches": len(mismatches),
        "mismatches": [{"pid": pid, **c} for pid, c in mismatches],
        "predictions": {r["pid"]: r for r in
                        (rows[n] for n in sorted(rows))},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote {args.out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
