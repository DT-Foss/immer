# O1-STATE — Server-Betrieb (Forschungsläufe)

*Stand: 2026-08-15 aktualisiert (Aero-Teardown + Beast-Entschlackung); Zugangsdaten und Regeln unverändert gültig.*

> **Diese Maschinen tragen ZWEI Projekte.** Das maritime Sicherheits-/Honeypot-Projekt
> **Konpeki** ist History auf aero/beast (Teardown 2026-08-15, Archiv in
> `aero:/storage/archive/`); Core läuft noch, Intel unverändert. O1-Regeln gelten weiter:
> Konpeki-Reste (Core) nicht mit O1-Läufen vermischen; O1-Läufe weiterhin nie auf aero
> (aero ist jetzt reiner Storage-Server — kein Python-Stack, keine Container).

## Projekt-Anker

- **Lokales Repo (Mac):** `/Users/bhkmie/Documents/Forschung/O1_juli`
  — Git-Remote `o1state` → `github.com/DT-Foss/o1-state`
- **Wichtigste Runner:** `src/pos_run.py` (POS/Gate-Läufe, 3-Arm-Rezept),
  `src/portable_organism.py` (Organismus-Primitive: Migration, Resume, Spans, Sleep).
  Experiment-Harnesses: `src/hub_n5_run.py`, `src/knowledge_file_run.py`,
  `src/source_swap_run.py`, `src/five_brain_run.py`, `src/pixel_body_run.py`.
- **Prediction-Register:** `analysis/PREDICTIONS.md` — jeder Lauf wird VOR dem Build
  registriert. Ergebnisse landen als `results/*.json` + `*.log` im Repo.
- SSH-Zugangsdaten/Keys: siehe `CREDENTIALS.md` (hier nicht dupliziert).

## Server-Landkarte (Stand 2026-08-15, live verifiziert)

| Alias | IP | CPU | RAM | Disk frei | O1-Rolle |
|---|---|---|---|---|---|
| **intel** (`ki`) | 89.167.47.205 | 4 | 7 GB | 32 G (~56%) | **DER LIFETIME-LAUF — NIE ANFASSEN** (2026-08-15: 12.07B tokens, RSS 0.82 GB). Achtung: davidfoss-Website läuft auf demselben Server |
| **core** (`kc`) | 89.167.31.243 | 4 | 7 GB | 49 G (36%) + leeres Volume | **Freie Experiment-Maschine** — Konpeki komplett entfernt (2026-08-15), nur nginx-default/SSH. Volle O1-Nutzung möglich |
| **beast** (`kb`) | 89.167.35.196 | 16 | 30 GB | 163 G (44%) | Experiment-Runner (stärkste Maschine; C4-Streaming historisch unzuverlässig — synthetische/lokale Workloads bevorzugen). Konpeki-Reste entfernt, aktive Projekte (tzofeh/jacsi/channel-bot/forge/forward2) teilen die Maschine |
| **aero** (`ka`) | 89.167.35.24 | 4 | 7 GB | 27 G (64%) | **KEINE O1-Nutzung** — jetzt reiner Storage-Server (`/storage/archive`, `/storage/mac_offload`) |
| gottformel (`fl`) | 10.99.0.1 (VPN) | — | — | — | nicht erreichbar (VPN down) — kein O1-Bestandteil |
| **Mac** (lokal) | — | 10 | — | — | Referenzmaschine für saubere Einzelläufe (Ignition-/Gate-Messungen sind co-load-empfindlich) |

## Was wo läuft (Momentaufnahme 2026-08-15)

- **intel:** `/root/o1_lifetime/` — der ununterbrochene Lifetime-Organismus
  (PID 2095339, seit 2026-07-24 in einem Prozess; Leben insgesamt seit ~20.–22. Juli).
  Live gelesen: **7.134.694.400 streamed tokens**, loss_ema 4.155, **RSS 0.782 GB**
  (Peak 0.834), 5.973 tok/s, d_model=128, chunk=64, batch=8.
  Status: `/root/o1_lifetime/results_lifetime/status.json` ·
  Log: `/root/o1_lifetime/lifetime.log` · venv: `/root/o1_lifetime/.venv/bin/python`.
  Ein Stall-Guard-Cron überwacht Hänger (`stall_guard.log`, leer = keine Eingriffe).
- **Mac:** DECISIVE-Läufe (d512 @50M Repeat/Seed-43 — Lotterie-vs-Breite-Urteil).
- **core / beast:** frei zwischen Experimenten; zuletzt P46 (hub_n5) auf core,
  P48 (pixel_body) auf beast.
- Hinweis: `beast:/root/o1_lifetime/` existiert ebenfalls, ist aber nur eine
  **Code-Kopie** (src/, reference/, requirements.txt) — dort läuft kein Leben.

## Verzeichnis- & Python-Struktur pro Server

| Server | O1-Arbeitsverzeichnis | Python für O1-Läufe |
|---|---|---|
| core | `/root/o1lab/` (src/, results/) | `/root/o1lab-venv/bin/python` — torch 2.13.0+cpu, datasets 2.19.0 (gepinnt!) |
| beast | `/root/o1lab/` (src/, results/) | `/usr/bin/python3` — torch 2.10.0+cu128 |
| intel | `/root/o1_lifetime/` | `/root/o1_lifetime/.venv/bin/python` |

## Das Standard-Rezept: Lauf starten & ernten

```bash
# 1) LOKAL Syntax prüfen — immer vor dem scp (Hausregel)
python3 -m py_compile src/<harness>.py

# 2) Deployen (portable_organism.py mitschicken, wenn der Harness es importiert)
scp src/<harness>.py src/portable_organism.py <server>:/root/o1lab/src/

# 3) Remote Syntax + Start (stdin-detach ist PFLICHT, sonst hängt die SSH-Session)
ssh -n <server> "cd /root/o1lab && <venv-python> -m py_compile src/<harness>.py \
  && nohup nice -n 10 <venv-python> -u src/<harness>.py [--smoke] \
  > results/<name>.log 2>&1 < /dev/null & echo GESTARTET"

# 4) Beobachten (read-only)
ssh -n <server> "tail -5 /root/o1lab/results/<name>.log"

# 5) Ernten
scp <server>:/root/o1lab/results/<name>.json results/
```

## Betriebsregeln (aus Messungen, nicht aus Vorsicht)

1. **intel nie anfassen.** Der Lifetime-Lauf ist das teuerste Artefakt des Projekts
   (Milliarden Token in einem Leben). Kein Deploy, kein zweiter Job, nichts.
2. **torch threads=1** in jedem Harness (`torch.set_num_threads(1)`) — nur so sind
   Zahlen zwischen Maschinen vergleichbar.
3. **Ein rechenintensiver O1-Job pro Maschine, sequenziell.** Co-Load verschiebt
   Gate-/Ignition-Messungen messbar (P34/P45c: Run-vs-Run-Instabilität).
4. **Kadenz explizit setzen und im Artefakt aufzeichnen** (batch/chunk/d_model) —
   eine Kosten-Ratio ohne Kadenz im selben JSON ist eine Zahl über die Defaults.
5. **Erst Smoke, dann Full.** Jeder Harness hat `--smoke`.
6. **Immer registrieren vor dem Bauen** (`analysis/PREDICTIONS.md`).
7. SSH-Aufrufe in Skripten **stdin-detached** (`ssh -n` bzw. `< /dev/null`) —
   sonst hält die Session den Kanal und Timeouts fressen die Orchestrierung.

## So kommst du drauf

SSH-Aliase liegen in `~/.ssh/config` (Mac): `core`/`kc`, `beast`/`kb`, `intel`/`ki`,
`aero`/`ka`, `gottformel`/`fl`. Beispiel: `ssh core`. Keys/Details: `CREDENTIALS.md`.
- SSH-Regel ergänzt: nohup-startende ssh-Kommandos NIE durch `| tail` pipen (puffert bis Kanal-EOF); Verifikation immer als separates Lese-ssh.
