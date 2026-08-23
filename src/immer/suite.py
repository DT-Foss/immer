"""Suite: one metrics heart for the human UI and the machine feed.

- status.json  — full snapshot, rewritten every turn (for agents: me)
- metrics.jsonl — append-only event line per turn (for graphs + audits)
- dashboard     — stdlib HTTP server: /status JSON, / dark HTML for David
"""

from __future__ import annotations

import json
import os
import threading
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable


class Metrics:
    def __init__(self, *, status_path: str | Path, jsonl_path: str | Path | None = None) -> None:
        self.status_path = Path(status_path).expanduser()
        self.jsonl_path = Path(jsonl_path).expanduser() if jsonl_path else None
        self.counters: dict[str, int] = {}
        self.gauges_override: dict[str, Any] = {}
        self.started_at = time.time()
        self._lock = threading.RLock()

    def bump(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def emit_gauge(self, name: str, value: Any) -> None:
        with self._lock:
            self.gauges_override[name] = value

    def snapshot(self, gauges: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            merged = dict(self.gauges_override)
            merged.update(gauges or {})
            return {
                "ts": time.time(),
                "uptime_s": round(time.time() - self.started_at, 1),
                "counters": dict(self.counters),
                "gauges": merged,
            }

    def emit(self, gauges: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            document = self.snapshot(gauges)
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
            fd, temporary = tempfile.mkstemp(
                dir=self.status_path.parent,
                prefix=f".{self.status_path.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self.status_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            if self.jsonl_path is not None:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(document, ensure_ascii=False) + "\n")
                    handle.flush()
            return document


_DARK_HTML = """<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<title>IMMER — Lebenszeichen</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root { color-scheme: dark; }
 body { background:#0b0e14; color:#c9d1d9; font:14px/1.5 ui-monospace,Menlo,monospace;
        margin:0; padding:24px; }
 h1 { font-size:18px; color:#58a6ff; margin:0 0 4px; }
 .sub { color:#8b949e; margin-bottom:20px; }
 .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }
 .card { background:#11161f; border:1px solid #21262d; border-radius:10px; padding:14px; }
 .card h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
            color:#8b949e; margin:0 0 10px; }
 .big { font-size:26px; color:#f0f6fc; }
 .row { display:flex; justify-content:space-between; padding:2px 0; }
 .row span:first-child { color:#8b949e; }
 .ok { color:#3fb950; } .warn { color:#d29922; } .off { color:#484f58; }
</style></head><body>
<h1>IMMER — Lebenszeichen</h1>
<div class="sub" id="sub">verbinde…</div>
<div class="grid" id="grid"></div>
<script>
const CARDS = [
  ["Leben", s => [
    ["turns", s.gauges.turns],
    ["tokens", s.gauges.tokens],
    ["loss_ema", fmt(s.gauges.loss_ema)],
    ["uptime", s.uptime_s + "s"],
  ]],
  ["Lernen (Plastizität)", s => [
    ["surprises", s.gauges.surprises],
    ["updates", s.gauges.updates],
    ["sleeps", s.gauges.sleeps],
    ["span_buffer", s.gauges.span_buffer],
  ]],
  ["Gehirne", s => [
    ["donor_27b", s.gauges.donor ? "✓ configured" : "· offline", s.gauges.donor ? "ok" : "off"],
    ["donor_modell", short(s.gauges.donor_modell)],
    ["lokal_fallback", s.gauges.lokal_gehirn ? "✓ geladen" : "aus", s.gauges.lokal_gehirn ? "warn" : "off"],
    ["fertig", "✓ vendored", "ok"],
    ["crsa", "✓ v0.5", "ok"],
  ]],
  ["Erinnerung", s => [
    ["spans", s.gauges.spans],
    ["lehr-stücke", s.counters.teaches || 0],
    ["abrufe", s.counters.recalls || 0],
  ]],
  ["Absichten", s => [
    ["chat", s.counters.chat || 0],
    ["math_ok", s.counters.math_ok || 0],
    ["math_abstain", s.counters.math_abstained || 0],
    ["status_fragen", s.counters.status || 0],
  ]],
];
function fmt(v){ return (v==null)?"—":Number(v).toFixed(4); }
function short(v){ return v?String(v).split("/").pop().slice(0,22):"—"; }
function esc(v){ return String(v??"—").replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"})[c]); }
async function tick(){
  try {
    const s = await (await fetch("/status")).json();
    document.getElementById("sub").textContent =
      "lebt seit " + s.uptime_s + "s · Quelle: status.json · Aktualisierung alle 2s";
    document.getElementById("grid").innerHTML = CARDS.map(([title, rows]) =>
      `<div class="card"><h2>${title}</h2>` +
      rows(s).map(([k,v,cls]) =>
        `<div class="row"><span>${esc(k)}</span><span class="${cls||""}">${esc(v)}</span></div>`
      ).join("") + "</div>"
    ).join("");
  } catch(e) {
    document.getElementById("sub").textContent = "kein Kontakt zum Leben: " + e;
  }
}
tick(); setInterval(tick, 2000);
</script></body></html>"""


def start_dashboard(metrics: Metrics, gauges: Callable[[], dict[str, Any]], port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            if self.path == "/status":
                body = json.dumps(metrics.emit(gauges()), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = _DARK_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:  # silence request spam
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
