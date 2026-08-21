Pro thinking

Assessing the integration

I’m reviewing the supplied integration, changelog, and draft material  before extending anything. I’ll verify what’s actually available,  identify missing implementation pieces, then continue with calculations  and tests where the evidence supports it.



Inspecting Repository and Searching Plans

```
bash -lc set -euo pipefail
printf 'PWD=%s\n' "$PWD"
find /mnt/data -maxdepth 3 -type f \( -name 'pyproject.toml' -o -name 'pytest.ini' -o -name 'PLAN*.md' -o -path '*/docs/superpowers/plans/*.md' \) -print | sed -n '1,160p'
printf '\nTop-level /mnt/data:\n'
find /mnt/data -maxdepth 2 -mindepth 1 -printf '%y %p\n' | sed -n '1,220p'
PWD=/
/mnt/data/_stage_attention_moonshot_full/attention-moonshot-final/pyproject.toml
/mnt/data/attention-moonshot-final/pyproject.toml
/mnt/data/_stage_causal_prefix_v05_ai/attention-moonshot-final/pyproject.toml

Top-level /mnt/data:
d /mnt/data/attention_v0.5_clean_integration
d /mnt/data/attention_v0.5_clean_integration/implementation
f /mnt/data/attention_v0.5_clean_integration/README_INTEGRATION.md
d /mnt/data/attention_v0.5_clean_integration/paper
f /mnt/data/attention_v0.5_clean_integration/RELEASE_NOTES.md
f /mnt/data/attention_v0.5_clean_integration/RELEASE_INDEX.json
f /mnt/data/attention_v0.5_clean_integration/AI_CONTEXT.md
f /mnt/data/attention_v0.5_clean_integration/CLAIM_LEDGER.json
d /mnt/data/attention_v0.5_clean_integration/source_materials
f /mnt/data/attention_v0.5_clean_integration/FILE_MANIFEST.csv
d /mnt/data/attention_v0.5_clean_integration/meta
f /mnt/data/attention_v0.5_clean_integration/SHA256SUMS.txt
f /mnt/data/Causal_Prefix_Sinkhorn_Attention_v0.5_IEEE_CONCISE.pdf
d /mnt/data/_verify_v05_concise
f /mnt/data/_verify_v05_concise/page-5.png
f /mnt/data/_verify_v05_concise/page-3.png
f /mnt/data/_verify_v05_concise/page-6.png
f /mnt/data/_verify_v05_concise/page-1.png
f /mnt/data/_verify_v05_concise/page-2.png
f /mnt/data/_verify_v05_concise/contact-1-3.png
f /mnt/data/_verify_v05_concise/page-4.png
f /mnt/data/_verify_v05_concise/contact-4-6.png
d /mnt/data/_stage_attention_moonshot_full
f /mnt/data/_stage_attention_moonshot_full/README_AI_INTEGRATION.md
d /mnt/data/_stage_attention_moonshot_full/source-material
d /mnt/data/_stage_attention_moonshot_full/attention-moonshot-final
d /mnt/data/_stage_attention_moonshot_full/paper
f /mnt/data/_stage_attention_moonshot_full/TEST_RESULTS.txt
f /mnt/data/_stage_attention_moonshot_full/BUNDLE_INFO.md
f /mnt/data/_stage_attention_moonshot_full/CLAIMS_AND_EVIDENCE.md
d /mnt/data/_stage_attention_moonshot_full/causal-marginal-attention-lab
f /mnt/data/_stage_attention_moonshot_full/FILE_MANIFEST.csv
f /mnt/data/_stage_attention_moonshot_full/INTEGRATION_CONTEXT.json
f /mnt/data/_stage_attention_moonshot_full/SHA256SUMS.txt
d /mnt/data/_renders_v04
f /mnt/data/_renders_v04/page-5.png
f /mnt/data/_renders_v04/page-3.png
f /mnt/data/_renders_v04/page-6.png
f /mnt/data/_renders_v04/page-1.png
f /mnt/data/_renders_v04/page-2.png
f /mnt/data/_renders_v04/page-4.png
f /mnt/data/causal-prefix-routing-v0.5-ai-integration.zip
f /mnt/data/attention_v0.5_DELIVERABLES_SHA256.txt
d /mnt/data/_render_v05_final
f /mnt/data/_render_v05_final/page-5.png
f /mnt/data/_render_v05_final/page-3.png
f /mnt/data/_render_v05_final/page-6.png
f /mnt/data/_render_v05_final/page-1.png
f /mnt/data/_render_v05_final/page-2.png
f /mnt/data/_render_v05_final/page-4.png
f /mnt/data/causal_prefix_sinkhorn_v0.4_polished(2).pdf
d /mnt/data/attention-moonshot-final
f /mnt/data/attention-moonshot-final/.gitignore
d /mnt/data/attention-moonshot-final/src
d /mnt/data/attention-moonshot-final/results
d /mnt/data/attention-moonshot-final/scripts
d /mnt/data/attention-moonshot-final/.git
d /mnt/data/attention-moonshot-final/corpus
f /mnt/data/attention-moonshot-final/pyproject.toml
d /mnt/data/attention-moonshot-final/.pytest_cache
d /mnt/data/attention-moonshot-final/tests
d /mnt/data/attention-moonshot-final/docs
f /mnt/data/attention_v0.5_clean_integration.zip
f /mnt/data/attention.zip
f /mnt/data/temporal_attention_2016.pdf
f /mnt/data/Causal_Prefix_Sinkhorn_Attention_v0.5_FULL.pdf
d /mnt/data/attention_source
d /mnt/data/attention_source/attention
d /mnt/data/attention_source/__MACOSX
f /mnt/data/ORGAN-GRAFTING-DRAFT.md
f /mnt/data/attention-moonshot-full-snapshot-2026-08-20.zip
d /mnt/data/user-voiFubHOHvfOs8kdPXDb71Ac
d /mnt/data/user-voiFubHOHvfOs8kdPXDb71Ac/1e3e6125d74a4e1895ee5b07ef840ad3
d /mnt/data/attention_v0.5_release
f /mnt/data/attention_v0.5_release/README_INTEGRATION.md
d /mnt/data/attention_v0.5_release/paper
f /mnt/data/attention_v0.5_release/RELEASE_NOTES.md
f /mnt/data/attention_v0.5_release/AI_CONTEXT.md
f /mnt/data/attention_v0.5_release/CLAIM_LEDGER.json
d /mnt/data/attention_v0.5_release/source_materials
d /mnt/data/attention_v0.5_release/meta
d /mnt/data/attention_v0.5_complete_snapshot
d /mnt/data/attention_v0.5_complete_snapshot/rendered_previews
d /mnt/data/attention_v0.5_complete_snapshot/working
f /mnt/data/attention_v0.5_complete_snapshot/RELEASE_INDEX.json
d /mnt/data/attention_v0.5_complete_snapshot/source_materials
d /mnt/data/attention_v0.5_complete_snapshot/release
f /mnt/data/attention_v0.5_complete_snapshot/FILE_MANIFEST.csv
f /mnt/data/attention_v0.5_complete_snapshot/README_COMPLETE_SNAPSHOT.md
f /mnt/data/attention_v0.5_complete_snapshot/SHA256SUMS.txt
d /mnt/data/causal-prefix-routing-v0.5
f /mnt/data/causal-prefix-routing-v0.5/README_AI_INTEGRATION.md
f /mnt/data/causal-prefix-routing-v0.5/causal_prefix_routing_v0.5.tex
f /mnt/data/causal-prefix-routing-v0.5/causal_prefix_routing_v0.5.pdf
f /mnt/data/causal-prefix-routing-v0.5/TEST_RESULTS.txt
d /mnt/data/causal-prefix-routing-v0.5/data
d /mnt/data/causal-prefix-routing-v0.5/build
d /mnt/data/causal-prefix-routing-v0.5/figures
f /mnt/data/causal-prefix-routing-v0.5/CLAIMS_AND_EVIDENCE.md
f /mnt/data/causal-prefix-routing-v0.5/make_figures.py
f /mnt/data/causal-prefix-routing-v0.5/causal_prefix_routing_v0.5.md
d /mnt/data/causal-prefix-routing-v0.5/renders
f /mnt/data/causal-prefix-routing-v0.5/INTEGRATION_CONTEXT.json
d /mnt/data/_verify_v05_full
f /mnt/data/_verify_v05_full/page-5.png
f /mnt/data/_verify_v05_full/page-7.png
f /mnt/data/_verify_v05_full/page-3.png
f /mnt/data/_verify_v05_full/page-6.png
f /mnt/data/_verify_v05_full/page-1.png
f /mnt/data/_verify_v05_full/contact-5-8.png
f /mnt/data/_verify_v05_full/page-2.png
f /mnt/data/_verify_v05_full/page-8.png
f /mnt/data/_verify_v05_full/contact-1-4.png
f /mnt/data/_verify_v05_full/page-4.png
d /mnt/data/causal-marginal-attention-lab
d /mnt/data/causal-marginal-attention-lab/results
f /mnt/data/attention_v0.5_complete_snapshot.zip
d /mnt/data/attention-source-text
d /mnt/data/attention-source-text/attention
d /mnt/data/attention-source-text/__MACOSX
d /mnt/data/causal-prefix-routing-v0.5-final
f /mnt/data/causal-prefix-routing-v0.5-final/README_AI_INTEGRATION.md
f /mnt/data/causal-prefix-routing-v0.5-final/causal_prefix_routing_v0.5.tex
f /mnt/data/causal-prefix-routing-v0.5-final/causal_prefix_routing_v0.5.pdf
f /mnt/data/causal-prefix-routing-v0.5-final/TEST_RESULTS.txt
f /mnt/data/causal-prefix-routing-v0.5-final/make_pressure_figure.py
d /mnt/data/causal-prefix-routing-v0.5-final/data
d /mnt/data/causal-prefix-routing-v0.5-final/build
d /mnt/data/causal-prefix-routing-v0.5-final/figures
f /mnt/data/causal-prefix-routing-v0.5-final/CLAIMS_AND_EVIDENCE.md
f /mnt/data/causal-prefix-routing-v0.5-final/make_figures.py
f /mnt/data/causal-prefix-routing-v0.5-final/causal_prefix_routing_v0.5.md
d /mnt/data/causal-prefix-routing-v0.5-final/renders
f /mnt/data/causal-prefix-routing-v0.5-final/INTEGRATION_CONTEXT.json
d /mnt/data/causal-prefix-sinkhorn-v0.5-concise-final
f /mnt/data/causal-prefix-sinkhorn-v0.5-concise-final/causal_prefix_sinkhorn_v0.5_ieee_concise.tex
f /mnt/data/causal-prefix-sinkhorn-v0.5-concise-final/references.bib
f /mnt/data/attention_v0.5_paper_sources.zip
d /mnt/data/_stage_causal_prefix_v05_ai
f /mnt/data/_stage_causal_prefix_v05_ai/README_AI_INTEGRATION.md
d /mnt/data/_stage_causal_prefix_v05_ai/source-material
d /mnt/data/_stage_causal_prefix_v05_ai/attention-moonshot-final
d /mnt/data/_stage_causal_prefix_v05_ai/paper
f /mnt/data/_stage_causal_prefix_v05_ai/TEST_RESULTS.txt
f /mnt/data/_stage_causal_prefix_v05_ai/BUNDLE_INFO.md
f /mnt/data/_stage_causal_prefix_v05_ai/CLAIMS_AND_EVIDENCE.md
d /mnt/data/_stage_causal_prefix_v05_ai/causal-marginal-attention-lab
f /mnt/data/_stage_causal_prefix_v05_ai/FILE_MANIFEST.csv
f /mnt/data/_stage_causal_prefix_v05_ai/INTEGRATION_CONTEXT.json
f /mnt/data/_stage_causal_prefix_v05_ai/SHA256SUMS.txt
f /mnt/data/CHANGELOG.md
```



Inspecting Git status and worktrees

```
bash -lc set -euo pipefail
cd /mnt/data/attention-moonshot-final
printf 'repo=%s\n' "$(pwd -P)"
GIT_DIR=$(cd "$(git rev-parse --git-dir)" && pwd -P)
GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)
BRANCH=$(git branch --show-current)
SUPER=$(git rev-parse --show-superproject-working-tree 2>/dev/null || true)
printf 'git_dir=%s\ngit_common=%s\nbranch=%s\nsuper=%s\n' "$GIT_DIR" "$GIT_COMMON" "$BRANCH" "$SUPER"
printf '\nstatus:\n'; git status --short
printf '\nbranches:\n'; git branch --all --no-color
printf '\nworktrees:\n'; git worktree list --porcelain
printf '\nlocal worktree dirs:\n'; ls -ld .worktrees worktrees 2>/dev/null || true
printf '\n.gitignore:\n'; cat .gitignore
repo=/mnt/data/attention-moonshot-final
git_dir=/mnt/data/attention-moonshot-final/.git
git_common=/mnt/data/attention-moonshot-final/.git
branch=moonshot/full-program
super=

status:
 M results/role_complete_screen/REPORT.md
 M results/role_complete_screen/results.json
 M src/attention_moonshot/candidates.py
 M src/attention_moonshot/experiment.py
 M src/attention_moonshot/model.py
 M src/attention_moonshot/operators.py
 M src/attention_moonshot/streaming.py
 M src/attention_moonshot/synthetic.py
 M tests/test_candidates.py
 M tests/test_model_training.py
?? results/PARALLEL_MERGE_2026-08-20.md
?? results/adaptive_archive_arm1.log
?? results/adaptive_archive_arm2.log
?? results/adaptive_archive_arm3.log
?? results/adaptive_archive_arm4.log
?? results/adaptive_archive_arm5.log
?? results/adaptive_depth_screen/
?? results/adaptive_route_screen/
?? results/adaptive_specialist_stage1/
?? results/adaptive_stage1/
?? results/anchor_foundation_stage/
?? results/anchor_residual_stage1/
?? results/architecture_scale/
?? results/depth_confirmation/
?? results/depth_program_stage/
?? results/generation14_stage1/
?? results/geometric_q_variable_lag/
?? results/length_extrapolation/
?? results/long_horizon_confirmation/
?? results/long_horizon_foundation/
?? results/marginal_residual_stage/
?? results/marginal_variable_lag/
?? results/q_residual_variable_lag/
?? results/q_spectrum_variable_lag/
?? results/recall_calibration/
?? results/role_complete_screen/run.log
?? results/role_complete_screen/run.pid
?? results/role_synthetic/
?? results/route_confirmation/
?? results/route_screen/
?? results/selected_confirmation/
?? results/specialist_mqar_stage/
?? results/specialist_stage1/
?? results/variable_lag_screen/
?? scripts/run_adaptive_depth_screen.py
?? scripts/run_adaptive_screen.py
?? scripts/run_anchor_confirmation.py
?? scripts/run_anchor_foundation_screen.py
?? scripts/run_anchor_residual_screen.py
?? scripts/run_architecture_scale.py
?? scripts/run_depth_confirmation.py
?? scripts/run_depth_program_atomic.py
?? scripts/run_depth_program_screen.py
?? scripts/run_geometric_q_variable_lag.py
?? scripts/run_headsplit_screen.py
?? scripts/run_length_extrapolation.py
?? scripts/run_long_horizon_confirmation.py
?? scripts/run_marginal_residual_screen.py
?? scripts/run_marginal_variable_lag.py
?? scripts/run_q_residual_variable_lag.py
?? scripts/run_q_spectrum_variable_lag.py
?? scripts/run_role_synthetic.py
?? scripts/run_variable_lag_screen.py
?? tests/test_anchor_confirmation_script.py
?? tests/test_geometric_prefix.py
?? tests/test_geometric_q_script.py
?? tests/test_marginal_residual.py
?? tests/test_marginal_screen_script.py
?? tests/test_marginal_variable_lag_script.py
?? tests/test_q_residual.py
?? tests/test_q_residual_script.py
?? tests/test_q_spectrum_script.py

branches:
* moonshot/full-program

worktrees:
worktree /mnt/data/attention-moonshot-final
HEAD c478d5380305969bf6a8d547584f4fadf02a76d5
branch refs/heads/moonshot/full-program


local worktree dirs:

.gitignore:
__pycache__/
*.py[cod]
.pytest_cache/
*.tmp
.DS_Store
```

Reconstructing the workspace

I’m continuing from the available materials, checking the current  implementation and draft context without assuming missing files exist.  I’ll separate verified integration behavior from placeholders, then  extend calculations and tests only where the evidence is sufficient.



Inspecting Project Documentation and Recent Commits

```
bash -lc set -euo pipefail
cd /mnt/data/attention-moonshot-final
find docs -maxdepth 4 -type f -print | sort | sed -n '1,240p'
printf '\nRecent commits:\n'; git log --oneline --decorate -12
docs/superpowers/plans/2026-08-20-causal-marginal-attention-moonshot.md
docs/superpowers/specs/2026-08-20-causal-marginal-attention-moonshot-design.md

Recent commits:
c478d53 (HEAD -> moonshot/full-program) feat: bound adaptive routing around causal role anchors
60ea765 feat: add whole-head adaptive specialists
61e1d6b feat: seed adaptive specialists from fixed-route winners
611a13c feat: anchor adaptive specialists with a whole free head
a044be0 feat: stream adaptive causal routes exactly
cb371a2 feat: report and screen adaptive route usage
f000b53 feat: instrument adaptive route utilization
e35c233 feat: add three-seed route confirmation harness
d7c66c4 feat: add causal multi-query recall stress task
cfa723c feat: add floor-gated adaptive causal routing
c91aecc feat: add three-local free-route candidate
2132eec feat: support layerwise attention programs
```



Analyzed

```
bash -lc set -euo pipefail
cd /mnt/data/attention-moonshot-final
du -sh . corpus results .git 2>/dev/null
WORK=/mnt/data/attention-moonshot-nope-v1
if [ -e "$WORK" ]; then echo "existing $WORK"; fi
14M	.
```



Analyzed

```
bash -lc set -euo pipefail
SRC=/mnt/data/attention-moonshot-final
WORK=/mnt/data/attention-moonshot-nope-v1
cd "$SRC"
if git show-ref --verify --quiet refs/heads/moonshot/nope-suite-v1; then
  echo 'branch already exists'; exit 3
fi
rm -rf "$WORK"
git worktree add "$WORK" -b moonshot/nope-suite-v1
rsync -a --exclude='.git' --exclude='.pytest_cache' "$SRC"/ "$WORK"/
cd "$WORK"
printf '\nWorktree status after snapshot sync:\n'
git status --short | sed -n '1,260p'
printf '\npyproject:\n'; sed -n '1,220p' pyproject.toml
Preparing worktree (new branch 'moonshot/nope-suite-v1')
HEAD is now at c478d53 feat: bound adaptive routing around causal role anchors

Worktree status after snapshot sync:
 M results/role_complete_screen/REPORT.md
 M results/role_complete_screen/results.json
 M src/attention_moonshot/candidates.py
 M src/attention_moonshot/experiment.py
 M src/attention_moonshot/model.py
 M src/attention_moonshot/operators.py
 M src/attention_moonshot/streaming.py
 M src/attention_moonshot/synthetic.py
 M tests/test_candidates.py
 M tests/test_model_training.py
?? results/PARALLEL_MERGE_2026-08-20.md
?? results/adaptive_archive_arm1.log
?? results/adaptive_archive_arm2.log
?? results/adaptive_archive_arm3.log
?? results/adaptive_archive_arm4.log
?? results/adaptive_archive_arm5.log
?? results/adaptive_depth_screen/
?? results/adaptive_route_screen/
?? results/adaptive_specialist_stage1/
?? results/adaptive_stage1/
?? results/anchor_foundation_stage/
?? results/anchor_residual_stage1/
?? results/architecture_scale/
?? results/depth_confirmation/
?? results/depth_program_stage/
?? results/generation14_stage1/
?? results/geometric_q_variable_lag/
?? results/length_extrapolation/
?? results/long_horizon_confirmation/
?? results/long_horizon_foundation/
?? results/marginal_residual_stage/
?? results/marginal_variable_lag/
?? results/q_residual_variable_lag/
?? results/q_spectrum_variable_lag/
?? results/recall_calibration/
?? results/role_complete_screen/run.log
?? results/role_complete_screen/run.pid
?? results/role_synthetic/
?? results/route_confirmation/
?? results/route_screen/
?? results/selected_confirmation/
?? results/specialist_mqar_stage/
?? results/specialist_stage1/
?? results/variable_lag_screen/
?? scripts/run_adaptive_depth_screen.py
?? scripts/run_adaptive_screen.py
?? scripts/run_anchor_confirmation.py
?? scripts/run_anchor_foundation_screen.py
?? scripts/run_anchor_residual_screen.py
?? scripts/run_architecture_scale.py
?? scripts/run_depth_confirmation.py
?? scripts/run_depth_program_atomic.py
?? scripts/run_depth_program_screen.py
?? scripts/run_geometric_q_variable_lag.py
?? scripts/run_headsplit_screen.py
?? scripts/run_length_extrapolation.py
?? scripts/run_long_horizon_confirmation.py
?? scripts/run_marginal_residual_screen.py
?? scripts/run_marginal_variable_lag.py
?? scripts/run_q_residual_variable_lag.py
?? scripts/run_q_spectrum_variable_lag.py
?? scripts/run_role_synthetic.py
?? scripts/run_variable_lag_screen.py
?? tests/test_anchor_confirmation_script.py
?? tests/test_geometric_prefix.py
?? tests/test_geometric_q_script.py
?? tests/test_marginal_residual.py
?? tests/test_marginal_screen_script.py
?? tests/test_marginal_variable_lag_script.py
?? tests/test_q_residual.py
?? tests/test_q_residual_script.py
?? tests/test_q_spectrum_script.py

pyproject:
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "attention-moonshot"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["torch", "numpy"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```



Analyzed

```
bash -lc set -euo pipefail
cd /mnt/data/attention-moonshot-nope-v1
printf '%s\n' '--- SPEC ---'
sed -n '1,260p' docs/superpowers/specs/2026-08-20-causal-marginal-attention-moonshot-design.md
printf '%s\n' '--- PLAN ---'
sed -n '1,360p' docs/superpowers/plans/2026-08-20-causal-marginal-attention-moonshot.md
--- SPEC ---
# Causal Marginal Attention Moonshot Design

## Objective

Turn Causal Prefix–Sinkhorn from a single normalization into a causal routing architecture that preserves its byte-model optimization gain while retaining an explicit unrestricted long-range channel.

## Core architecture

The four head roles are fixed and independently testable:

1. **Self** — identity attention for exact current-token transport.
2. **Local** — strictly-past recency attention, excluding the diagonal.
3. **Balanced** — residual-aware causal Prefix–Sinkhorn (RAPS) with log-domain streaming state.
4. **Free** — ordinary causal softmax, left untouched for unrestricted long-range retrieval.

The initial four-head form assigns one head to each role. Candidate grids may change role counts while keeping at least one free head during long-range screening.

## Acceptance gates

- Exact causal support and zero future-input gradient.
- Batch/streaming equality within 2e-6 in float32.
- All source-based unit tests pass from a clean cache.
- Long-lag copy no longer collapses as dual-route did.
- Associative recall remains trainable.
- Archive-corpus validation beats softmax under identical budget.
- Finalists receive multi-seed test confirmation and length-extrapolation stress tests.

## Experimental order

1. Stabilize and audit the four-route implementation.
2. Cheap synthetic kill-screen across role-count variants.
3. One-seed archive-corpus screen for survivors.
4. Multi-seed confirmation on test split.
5. Length extrapolation, mechanism ablations, and throughput/Pareto analysis.
6. Promote the winning architecture into the paper and distributable repository.
--- PLAN ---
# Causal Marginal Attention Moonshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and independently verify a four-route causal attention architecture that combines self, local, Prefix–Sinkhorn-balanced, and unrestricted softmax heads.

**Architecture:** Extend the existing attention operator family with an exact four-role router and streaming state that updates only balanced heads. Screen role allocations first on synthetic long-range tasks, then on the held-out archive byte corpus, and confirm finalists across seeds and extrapolated contexts.

**Tech Stack:** Python 3.13, PyTorch 2.10 CPU, pytest, JSON/Markdown experiment reports.

**Spec:** `docs/superpowers/specs/2026-08-20-causal-marginal-attention-moonshot-design.md`

## Global Constraints

- Strict autoregressive causality: future-input gradient exactly zero.
- Deterministic seeds and identical data order across arms.
- Test split remains unopened during screening.
- At least one unrestricted causal-softmax head in long-range candidates.
- CPU-only execution with no external network dependency.

---

### Task 1: Stabilize four-route batch and streaming semantics

**Files:**
- Modify: `src/attention_moonshot/operators.py`
- Modify: `src/attention_moonshot/streaming.py`
- Modify: `tests/test_operators.py`

- [ ] Add a failing test proving only balanced heads accumulate Prefix state.
- [ ] Run the targeted test and verify the expected failure.
- [ ] Remove duplicate streaming definitions and implement one exact state update.
- [ ] Run targeted and full unit tests.
- [ ] Commit the verified operator.

### Task 2: Build a deterministic role-allocation screening harness

**Files:**
- Create: `src/attention_moonshot/candidates.py`
- Create: `scripts/run_route_screen.py`
- Create: `tests/test_candidates.py`

- [ ] Add failing tests for unique labels, free-head preservation, and canonical candidates.
- [ ] Implement candidate grids and report generation.
- [ ] Run tests and commit.

### Task 3: Run synthetic kill-screen

**Files:**
- Create: `results/route_screen_synthetic/results.json`
- Create: `results/route_screen_synthetic/REPORT.md`

- [ ] Run delayed-copy and associative-recall screens for all route candidates.
- [ ] Reject candidates that lose the explicit long-range gate.
- [ ] Promote the Pareto survivors.

### Task 4: Run archive-corpus screen and confirmation

**Files:**
- Create: `results/route_screen_archive/results.json`
- Create: `results/route_screen_archive/REPORT.md`
- Create: `results/route_confirmation/results.json`
- Create: `results/route_confirmation/REPORT.md`

- [ ] Run one-seed validation-only archive screen.
- [ ] Run three-seed validation/test confirmation for finalists and baselines.
- [ ] Compute mean, standard deviation, effect sizes, and throughput.

### Task 5: Length extrapolation and mechanism audit

**Files:**
- Create: `scripts/run_extrapolation.py`
- Create: `results/extrapolation/results.json`
- Create: `results/extrapolation/REPORT.md`

- [ ] Add tests for evaluation at contexts longer than training context using a configurable position extension.
- [ ] Run copy/recall at increasing lag and context.
- [ ] Audit diagonal, age, long-range mass, Gini, effective rank, and free-head utilization.

### Task 6: Consolidate claims and package

**Files:**
- Create: `CLAIM_LEDGER.md`
- Create: `FINAL_REPORT.md`
- Create: distributable ZIP artifact.

- [ ] Record every theorem, test, empirical result, and exact limitation.
- [ ] Run clean-cache unit tests and all report validators.
- [ ] Package code, reports, manifests, and reproducibility commands.
```



Analyzed

```
bash -lc set -euo pipefail
cd /mnt/data/attention-moonshot-nope-v1
python3 --version
python3 - <<'PY'
import torch, numpy
print('torch', torch.__version__)
print('numpy', numpy.__version__)
PY
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q
Python 3.13.5
torch 2.10.0+cpu
numpy 2.3.5
........................................................................ [ 81%]
................                                                         [100%]
88 passed in 2.50s
```



Analyzed

```
bash -lc set -euo pipefail
cd /mnt/data/attention-moonshot-nope-v1
printf '%s\n' '--- tree ---'
find src/attention_moonshot -maxdepth 2 -type f -print | sort
printf '%s\n' '--- scripts ---'
find scripts -maxdepth 1 -type f -print | sort
printf '%s\n' '--- current model/program APIs ---'
rg -n "class ByteLM|class.*Model|PROGRAM|programme|nope|position|pos_emb|AttentionConfig|route|Q_RESIDUAL|RAPS|LOCAL|FREE" src tests scripts | sed -n '1,320p'
--- tree ---
src/attention_moonshot/__init__.py
src/attention_moonshot/__pycache__/__init__.cpython-313.pyc
src/attention_moonshot/__pycache__/candidates.cpython-313.pyc
src/attention_moonshot/__pycache__/data.cpython-313.pyc
src/attention_moonshot/__pycache__/experiment.cpython-313.pyc
src/attention_moonshot/__pycache__/metrics.cpython-313.pyc
src/attention_moonshot/__pycache__/model.cpython-313.pyc
src/attention_moonshot/__pycache__/operators.cpython-313.pyc
src/attention_moonshot/__pycache__/streaming.cpython-313.pyc
src/attention_moonshot/__pycache__/synthetic.cpython-313.pyc
src/attention_moonshot/candidates.py
src/attention_moonshot/data.py
src/attention_moonshot/experiment.py
src/attention_moonshot/metrics.py
src/attention_moonshot/model.py
src/attention_moonshot/operators.py
src/attention_moonshot/streaming.py
src/attention_moonshot/synthetic.py
--- scripts ---
scripts/run_adaptive_depth_screen.py
scripts/run_adaptive_screen.py
scripts/run_anchor_confirmation.py
scripts/run_anchor_foundation_screen.py
scripts/run_anchor_residual_screen.py
scripts/run_architecture_scale.py
scripts/run_archive_screen.py
scripts/run_confirmation.py
scripts/run_depth_confirmation.py
scripts/run_depth_program_atomic.py
scripts/run_depth_program_screen.py
scripts/run_geometric_q_variable_lag.py
scripts/run_headsplit_screen.py
scripts/run_length_extrapolation.py
scripts/run_long_horizon_confirmation.py
scripts/run_marginal_residual_screen.py
scripts/run_marginal_variable_lag.py
scripts/run_q_residual_variable_lag.py
scripts/run_q_spectrum_variable_lag.py
scripts/run_role_complete_screen.py
scripts/run_role_synthetic.py
scripts/run_route_confirmation.py
scripts/run_route_screen.py
scripts/run_specialist_screen.py
scripts/run_synthetic_suite.py
scripts/run_theory_audit.py
scripts/run_variable_lag_screen.py
--- current model/program APIs ---
src/attention_moonshot/streaming.py:22:    if spec.kind not in {'prefix_log','raps','geometric_prefix','dual_route','slg','quad_route'}:raise ValueError('prefix-log compatible spec required')
src/attention_moonshot/streaming.py:52:def dual_route_stream_step(logits_row:Tensor,state:PrefixLogState,spec:AttentionSpec)->tuple[Tensor,PrefixLogState]:
src/attention_moonshot/streaming.py:53:    if spec.kind!='dual_route':raise ValueError("kind='dual_route' required")
src/attention_moonshot/streaming.py:71:def quad_route_stream_step(logits_row:Tensor,state:PrefixLogState,spec:AttentionSpec)->tuple[Tensor,PrefixLogState]:
src/attention_moonshot/streaming.py:74:    Head order is fixed: self, strictly-past local, RAPS-balanced, then
src/attention_moonshot/streaming.py:78:    if spec.kind!='quad_route':raise ValueError("kind='quad_route' required")
src/attention_moonshot/streaming.py:113:def adaptive_route_stream_step(
src/attention_moonshot/streaming.py:120:    if spec.kind != "adaptive_route":
src/attention_moonshot/streaming.py:121:        raise ValueError("kind='adaptive_route' required")
tests/test_candidates.py:1:from attention_moonshot.candidates import free_head_count, route_screen_candidates
tests/test_candidates.py:4:def test_route_screen_candidate_labels_are_unique():
tests/test_candidates.py:5:    specs=route_screen_candidates(n_heads=4)
tests/test_candidates.py:10:def test_every_four_route_candidate_preserves_a_free_softmax_head():
tests/test_candidates.py:11:    specs=route_screen_candidates(n_heads=4)
tests/test_candidates.py:12:    quads=[spec for spec in specs if spec.kind=='quad_route']
tests/test_candidates.py:18:    labels={spec.label() for spec in route_screen_candidates(n_heads=4)}
tests/test_candidates.py:19:    assert 'quad_route[dd=3,sh=1,lh=1,bh=1,s=0.8]' in labels
tests/test_candidates.py:23:    kinds={spec.kind for spec in route_screen_candidates(n_heads=4)}
tests/test_candidates.py:24:    assert {'softmax','raps','recency','dual_route','slg'} <= kinds
tests/test_candidates.py:28:    labels={spec.label() for spec in route_screen_candidates(n_heads=4)}
tests/test_candidates.py:29:    assert 'quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]' in labels
tests/test_candidates.py:32:def test_confirmation_set_contains_baseline_and_pareto_routes():
tests/test_candidates.py:33:    from attention_moonshot.candidates import route_confirmation_candidates
tests/test_candidates.py:34:    labels=[spec.label() for spec in route_confirmation_candidates(n_heads=4)]
tests/test_candidates.py:36:    assert 'quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]' in labels
tests/test_candidates.py:37:    assert 'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]' in labels
tests/test_candidates.py:38:    assert 'quad_route[dd=3,sh=0,lh=1,bh=1,s=0.8]' in labels
tests/test_candidates.py:42:def test_adaptive_route_candidates_are_unique_and_keep_a_hard_free_path():
tests/test_candidates.py:43:    from attention_moonshot.candidates import adaptive_route_candidates
tests/test_candidates.py:45:    specs = adaptive_route_candidates()
tests/test_candidates.py:50:    adaptive = [spec for spec in specs if spec.kind == "adaptive_route"]
tests/test_candidates.py:57:def test_adaptive_specialist_candidates_anchor_to_both_fixed_route_winners() -> None:
tests/test_candidates.py:78:    assert specs[1].kind == "quad_route" and specs[1].local_heads == 2 and specs[1].balanced_heads == 1
tests/test_candidates.py:79:    assert specs[2].kind == "quad_route" and specs[2].local_heads == 3 and specs[2].balanced_heads == 0
tests/test_candidates.py:87:def test_crystallized_depth_programs_cover_route_placement_and_preserve_free_heads() -> None:
tests/test_candidates.py:120:    assert "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax" in labels
tests/test_candidates.py:121:    assert "quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax" in labels
tests/test_candidates.py:124:    routed = programs[1:3]
tests/test_candidates.py:125:    assert all(free_head_count(program[0], 4) == 1 for program in routed)
tests/test_candidates.py:135:        "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:136:        "quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax",
tests/test_candidates.py:149:        "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:150:        "quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax",
tests/test_candidates.py:163:    assert labels[1].startswith("quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax")
tests/test_candidates.py:164:    assert labels[2].startswith("quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax")
tests/test_candidates.py:165:    assert labels[3].count("quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]") == 4
tests/test_candidates.py:201:    crsa_label = "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]"
tests/test_candidates.py:206:        "quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax -> softmax -> softmax",
tests/test_candidates.py:210:        sum(spec.kind == "quad_route" and spec.balanced_heads == 1 for spec in program)
tests/test_candidates.py:211:        for program in programs if all(spec.kind in {"softmax", "quad_route"} for spec in program)
tests/test_candidates.py:212:        and not any(spec.kind == "quad_route" and spec.balanced_heads == 0 for spec in program)
tests/test_candidates.py:224:        "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:225:        "quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax",
tests/test_candidates.py:242:        "quad_route[a=0.5,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:243:        "quad_route[a=0.75,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:244:        "quad_route[dd=2,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:245:        "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:246:        "quad_route[dd=4,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:247:        "quad_route[a=1.25,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:248:        "quad_route[a=1.5,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:260:        "quad_route[lam=0.9,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:261:        "quad_route[lam=0.95,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:262:        "quad_route[lam=0.98,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:263:        "quad_route[lam=0.99,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:264:        "quad_route[lam=0.995,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_candidates.py:265:        "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
tests/test_marginal_residual.py:19:def test_marginal_residual_zero_budget_is_exact_three_local_one_free_route() -> None:
tests/test_marginal_residual.py:31:    actual, balance = marginal_residual_attention(logits, gate, spec, return_routes=True)
tests/test_marginal_residual.py:34:        S(kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0, slope=0.8),
tests/test_marginal_residual.py:102:        weight = block.attn.route_gate_weight
tests/test_marginal_residual.py:103:        bias = block.attn.route_gate_bias
tests/test_marginal_residual.py:162:    from attention_moonshot.experiment import adaptive_route_metrics
tests/test_marginal_residual.py:176:    route = adaptive_route_metrics(model)
tests/test_marginal_residual.py:177:    assert route is not None
tests/test_marginal_residual.py:178:    assert route["names"] == ["self", "local", "balanced", "free"]
tests/test_marginal_residual.py:179:    assert route["free_heads"] == 1
tests/test_marginal_residual.py:180:    assert route["effective_mean"][3] == 0.25
tests/test_marginal_residual.py:181:    assert 0.0 <= route["effective_mean"][2] <= 0.375
tests/test_marginal_residual.py:182:    assert abs(sum(route["effective_mean"]) - 1.0) < 1e-6
tests/test_marginal_residual.py:183:    assert route["per_head_effective"][-1] == [0.0, 0.0, 0.0, 1.0]
tests/test_route_screen_reporting.py:7:def _load_route_screen_module():
tests/test_route_screen_reporting.py:8:    script = Path(__file__).resolve().parents[1] / "scripts[... ELLIPSIZATION ...].25, diagonal_debit=3),
scripts/run_role_complete_screen.py:27:    routed = [
scripts/run_role_complete_screen.py:29:        S(kind="quad_route", self_heads=1, local_heads=1, balanced_heads=1, slope=.8, diagonal_debit=3),
scripts/run_role_complete_screen.py:31:        S(kind="quad_route", self_heads=1, local_heads=1, balanced_heads=1, slope=.8, diagonal_debit=2),
scripts/run_role_complete_screen.py:32:        S(kind="quad_route", self_heads=1, local_heads=1, balanced_heads=1, slope=.8, diagonal_debit=4),
scripts/run_role_complete_screen.py:33:        S(kind="quad_route", self_heads=1, local_heads=1, balanced_heads=1, slope=1.25, diagonal_debit=3),
scripts/run_role_complete_screen.py:35:        S(kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1, slope=.8, diagonal_debit=3),
scripts/run_role_complete_screen.py:36:        S(kind="quad_route", self_heads=0, local_heads=1, balanced_heads=2, slope=.8, diagonal_debit=3),
scripts/run_role_complete_screen.py:37:        S(kind="quad_route", self_heads=1, local_heads=0, balanced_heads=2, slope=.8, diagonal_debit=3),
scripts/run_role_complete_screen.py:38:        S(kind="quad_route", self_heads=2, local_heads=0, balanced_heads=1, slope=.8, diagonal_debit=3),
scripts/run_role_complete_screen.py:39:        S(kind="quad_route", self_heads=0, local_heads=0, balanced_heads=3, slope=.8, diagonal_debit=3),
scripts/run_role_complete_screen.py:40:        S(kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0, slope=.8, diagonal_debit=0),
scripts/run_role_complete_screen.py:41:        S(kind="quad_route", self_heads=1, local_heads=2, balanced_heads=0, slope=.8, diagonal_debit=0),
scripts/run_role_complete_screen.py:43:        S(kind="quad_route", self_heads=0, local_heads=1, balanced_heads=1, slope=.8, diagonal_debit=3),
scripts/run_role_complete_screen.py:44:        S(kind="quad_route", self_heads=1, local_heads=0, balanced_heads=1, slope=.8, diagonal_debit=3),
scripts/run_role_complete_screen.py:45:        S(kind="quad_route", self_heads=1, local_heads=1, balanced_heads=0, slope=.8, diagonal_debit=0),
scripts/run_role_complete_screen.py:46:        S(kind="quad_route", self_heads=0, local_heads=0, balanced_heads=1, slope=.8, diagonal_debit=3),
scripts/run_role_complete_screen.py:47:        S(kind="quad_route", self_heads=0, local_heads=1, balanced_heads=0, slope=.8, diagonal_debit=0),
scripts/run_role_complete_screen.py:49:    out = controls + routed
scripts/run_role_complete_screen.py:68:        f"Completed **{len(results)}/{len(candidates())}** arms; seed **9137**; 220 steps. Test split remained unopened.",
scripts/run_role_complete_screen.py:70:        "`quad_route[sh,lh,bh]` assigns exact self heads, past-only local heads, prefix-balanced heads, and leaves all remaining heads as unrestricted causal softmax.",
tests/test_operators.py:4:from attention_moonshot.streaming import PrefixLogState,prefix_log_stream_step,reservoir_stream_step,dual_route_stream_step,slg_stream_step,quad_route_stream_step
tests/test_operators.py:32:    specs=[S(),S(kind='prefix_log'),S(kind='raps',diagonal_debit=3),S(kind='reservoir_prefix',diagonal_debit=3),S(kind='recency',slope=1.25),S(kind='past_recency',slope=.8),S(kind='dual_route',local_heads=3,slope=1.25,diagonal_debit=3),S(kind='slg',self_heads=1,local_heads=1,slope=.8,diagonal_debit=3),S(kind='quad_route',self_heads=1,local_heads=1,balanced_heads=1,slope=.8,diagonal_debit=3)]
tests/test_operators.py:36:    for s in [S(),S(kind='prefix_log'),S(kind='raps',diagonal_debit=3),S(kind='reservoir_prefix',diagonal_debit=3),S(kind='dual_route',local_heads=2,slope=.8,diagonal_debit=3),S(kind='slg',self_heads=1,local_heads=1,slope=.8,diagonal_debit=3),S(kind='quad_route',self_heads=1,local_heads=1,balanced_heads=1,slope=.8,diagonal_debit=3)]:
tests/test_operators.py:44:def test_head_routes_assign_exact_roles():
tests/test_operators.py:45:    x=logits();dual=S(kind='dual_route',local_heads=3,slope=1.25,diagonal_debit=3);w=apply_attention(x,dual)
tests/test_operators.py:62:    _stream_check(S(kind='dual_route',local_heads=2,slope=.8,diagonal_debit=3),dual_route_stream_step)
tests/test_operators.py:64:    _stream_check(S(kind='quad_route',self_heads=1,local_heads=1,balanced_heads=1,slope=.8,diagonal_debit=3),quad_route_stream_step)
tests/test_operators.py:72:def test_self_local_balanced_free_route_assigns_exact_roles():
tests/test_operators.py:74:    spec=S(kind='quad_route',self_heads=1,local_heads=1,balanced_heads=1,slope=.8,diagonal_debit=3)
tests/test_operators.py:82:def test_quad_route_stream_state_is_owned_only_by_balanced_heads():
tests/test_operators.py:83:    spec=S(kind='quad_route',self_heads=1,local_heads=1,balanced_heads=1,slope=.8,diagonal_debit=3)
tests/test_operators.py:86:    _,next_state=quad_route_stream_step(row,state,spec)
scripts/run_marginal_residual_screen.py:44:def _route_text(group: list[dict[str, Any]]) -> str:
scripts/run_marginal_residual_screen.py:45:    routes = [row.get("route") for row in group if row.get("route") is not None]
scripts/run_marginal_residual_screen.py:46:    if not routes:
scripts/run_marginal_residual_screen.py:48:    dimensions = len(routes[0]["effective_mean"])
scripts/run_marginal_residual_screen.py:50:        statistics.mean(float(route["effective_mean"][index]) for route in routes)
scripts/run_marginal_residual_screen.py:80:                ranked.append((score, operator, len(group), score_sd, speed, speed_sd, _route_text(group)))
scripts/run_marginal_residual_screen.py:85:                "Test split remained unopened.",
scripts/run_marginal_residual_screen.py:90:            for rank, (score, operator, count, score_sd, speed, speed_sd, route) in enumerate(ranked, 1):
scripts/run_marginal_residual_screen.py:94:                    f"{delta:+.6f}|{route}|{speed:.0f} ± {speed_sd:.0f}|"
scripts/run_marginal_residual_screen.py:102:                ranked.append((-accuracy, bits, operator, len(group), accuracy, accuracy_sd, bits_sd, speed, speed_sd, _route_text(group)))
scripts/run_marginal_residual_screen.py:111:            for rank, (_, bits, operator, count, accuracy, accuracy_sd, bits_sd, speed, speed_sd, route) in enumerate(ranked, 1):
scripts/run_marginal_residual_screen.py:114:                    f"{bits:.4f} ± {bits_sd:.4f}|{route}|{speed:.0f} ± {speed_sd:.0f}|"
scripts/run_anchor_confirmation.py:26:        kind="quad_route",
scripts/run_anchor_confirmation.py:33:        kind="quad_route",
scripts/run_anchor_confirmation.py:80:def _route_text(group: list[dict[str, Any]]) -> str:
scripts/run_anchor_confirmation.py:81:    routes = [row.get("route") for row in group if row.get("route") is not None]
scripts/run_anchor_confirmation.py:82:    if not routes:
scripts/run_anchor_confirmation.py:84:    dimensions = len(routes[0]["effective_mean"])
scripts/run_anchor_confirmation.py:86:        statistics.mean(float(route["effective_mean"][index]) for route in routes)
scripts/run_anchor_confirmation.py:131:            _route_text(group),
scripts/run_anchor_confirmation.py:148:            wins, paired_count, speed, speed_sd, route_text,
scripts/run_anchor_confirmation.py:154:            f"{wins}/{paired_count}|{route_text}|{speed:.0f} ± {speed_sd:.0f}|"
scripts/run_role_synthetic.py:25:        S(kind="dual_route", local_heads=3, slope=1.25, diagonal_debit=3),
scripts/run_role_synthetic.py:28:        S(kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0, slope=.8),
scripts/run_role_synthetic.py:29:        S(kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1, slope=.8, diagonal_debit=3),
scripts/run_role_synthetic.py:30:        S(kind="quad_route", self_heads=1, local_heads=2, balanced_heads=0, slope=.8),
scripts/run_role_synthetic.py:32:        S(kind="quad_route", self_heads=0, local_heads=1, balanced_heads=1, slope=.8, diagonal_debit=3),
scripts/run_role_synthetic.py:34:        S(kind="quad_route", self_heads=1, local_heads=1, balanced_heads=1, slope=.8, diagonal_debit=3),
scripts/run_role_synthetic.py:36:        S(kind="quad_route", self_heads=0, local_heads=1, balanced_heads=0, slope=.8),
scripts/run_role_synthetic.py:37:        S(kind="quad_route", self_heads=0, local_heads=0, balanced_heads=1, slope=.8, diagonal_debit=3),
scripts/run_role_synthetic.py:53:        "Delayed copy uses lag 48 in a 96-token context. Associative recall uses 16 unique key-value pairs and queries one key at the final position.",
scripts/run_route_confirmation.py:14:from attention_moonshot.candidates import route_confirmation_candidates
scripts/run_route_confirmation.py:79:    candidates = route_confirmation_candidates(n_heads=4)
scripts/run_route_confirmation.py:90:    output_dir = ROOT / "results" / "route_confirmation"
scripts/run_route_screen.py:14:from attention_moonshot.candidates import free_head_count, route_screen_candidates
scripts/run_route_screen.py:24:    return root / "results" / "route_screen" / profile
scripts/run_route_screen.py:56:        f"Completed **{len(results)}** atomic runs. Every four-route candidate preserves at least one untouched causal-softmax head.",
scripts/run_route_screen.py:109:                "The file-hash test split remains unopened in this screen.",
scripts/run_route_screen.py:127:    candidates = route_screen_candidates(n_heads=4)
tests/test_geometric_prefix.py:58:def test_quad_route_can_use_geometric_prefix_for_only_the_balanced_head() -> None:
tests/test_geometric_prefix.py:61:        kind="quad_route",
tests/test_model_training.py:13:    for spec in [S(kind='prefix_log'),S(kind='raps',diagonal_debit=3),S(kind='reservoir_prefix',diagonal_debit=3),S(kind='slg',self_heads=1,local_heads=1,slope=.8,diagonal_debit=3),S(kind='quad_route',self_heads=1,local_heads=1,balanced_heads=1,slope=.8,diagonal_debit=3)]:
tests/test_model_training.py:49:    program=(S(kind='quad_route',self_heads=0,local_heads=2,balanced_heads=1,slope=.8,diagonal_debit=3),S())
tests/test_model_training.py:75:def test_adaptive_training_reports_learned_route_usage(tmp_path: Path):
```



Analyzed

```
bash -lc set -euo pipefail
cd /mnt/data/attention-moonshot-nope-v1
sed -n '1,320p' src/attention_moonshot/model.py
printf '\n--- operators 1 ---\n'; sed -n '1,420p' src/attention_moonshot/operators.py
printf '\n--- synthetic ---\n'; sed -n '1,420p' src/attention_moonshot/synthetic.py
printf '\n--- experiment ---\n'; sed -n '1,360p' src/attention_moonshot/experiment.py
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .operators import AttentionSpec, adaptive_route_attention, adaptive_specialist_attention, anchor_residual_attention, marginal_residual_attention, q_residual_attention, apply_attention


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 256
    context: int = 128
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.0
    position_mode: str = "learned"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.context < 2:
            raise ValueError("context must be at least 2")
        if self.position_mode not in {"learned", "none"}:
            raise ValueError("position_mode must be 'learned' or 'none'")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, spec: AttentionSpec) -> None:
        super().__init__()
        self.spec = spec
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        if spec.kind == "adaptive_route":
            self.route_gate_weight = nn.Parameter(torch.zeros(cfg.n_heads, self.head_dim, 4))
            initial_bias = torch.tensor((0.0, 0.0, 0.0, -4.0)).repeat(cfg.n_heads, 1)
            self.route_gate_bias = nn.Parameter(initial_bias)
        elif spec.kind == "adaptive_specialists":
            if not 1 <= spec.free_heads < cfg.n_heads:
                raise ValueError("free_heads must be in 1..n_heads-1")
            specialist_heads = cfg.n_heads - spec.free_heads
            self.route_gate_weight = nn.Parameter(torch.zeros(specialist_heads, self.head_dim, 3))
            if spec.init_strength < 0:
                raise ValueError("init_strength must be nonnegative")
            if spec.specialist_init == "uniform":
                initial_bias = torch.zeros(specialist_heads, 3)
            elif spec.specialist_init in {"llb", "lll"}:
                strength = float(spec.init_strength)
                initial_bias = torch.full((specialist_heads, 3), -strength)
                for head in range(specialist_heads):
                    role = 1
                    if spec.specialist_init == "llb" and head == specialist_heads - 1:
                        role = 2
                    initial_bias[head, role] = strength
            else:
                raise ValueError("specialist_init must be 'llb', 'lll', or 'uniform'")
            self.route_gate_bias = nn.Parameter(initial_bias)
        elif spec.kind == "anchor_residual":
            if not 1 <= spec.free_heads < cfg.n_heads:
                raise ValueError("free_heads must be in 1..n_heads-1")
            if not 0.0 <= spec.adapt_budget <= 1.0:
                raise ValueError("adapt_budget must be in [0, 1]")
            if spec.anchor_pattern not in {"lll", "llb"}:
                raise ValueError("anchor_pattern must be 'lll' or 'llb'")
            specialist_heads = cfg.n_heads - spec.free_heads
            self.route_gate_weight = nn.Parameter(torch.zeros(specialist_heads, self.head_dim, 3))
            self.route_gate_bias = nn.Parameter(torch.zeros(specialist_heads, 3))
        elif spec.kind == "marginal_residual":
            if not 1 <= spec.free_heads < cfg.n_heads:
                raise ValueError("free_heads must be in 1..n_heads-1")
            if not 0.0 <= spec.adapt_budget <= 1.0:
                raise ValueError("adapt_budget must be in [0, 1]")
            specialist_heads = cfg.n_heads - spec.free_heads
            self.route_gate_weight = nn.Parameter(torch.zeros(specialist_heads, self.head_dim))
            self.route_gate_bias = nn.Parameter(torch.zeros(specialist_heads))
        elif spec.kind == "q_residual":
            if not 1 <= spec.free_heads < cfg.n_heads:
                raise ValueError("free_heads must be in 1..n_heads-1")
            if not 0.0 <= spec.adapt_budget <= 1.0:
                raise ValueError("adapt_budget must be in [0, 1]")
            if spec.init_strength < 0:
                raise ValueError("init_strength must be nonnegative")
            specialist_heads = cfg.n_heads - spec.free_heads
            self.route_gate_weight = nn.Parameter(torch.zeros(specialist_heads, self.head_dim))
            initial_bias = torch.full((specialist_heads,), -float(spec.init_strength))
            initial_bias[-1] = float(spec.init_strength)
            self.route_gate_bias = nn.Parameter(initial_bias)
        else:
            self.register_parameter("route_gate_weight", None)
            self.register_parameter("route_gate_bias", None)
        self.last_weights: Tensor | None = None
        self.last_route_probs: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        b,t,c = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        def heads(z: Tensor) -> Tensor:
            return z.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        q,k,v = heads(q),heads(k),heads(v)
        logits = q @ k.transpose(-2,-1) / math.sqrt(self.head_dim)
        if self.spec.kind == "adaptive_route":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("adaptive route gate parameters are unavailable")
            gate_logits = torch.einsum("bhtd,hdr->bhtr", q, self.route_gate_weight)
            gate_logits = gate_logits + self.route_gate_bias.view(1, self.n_heads, 1, 4)
            route_probs = torch.softmax(gate_logits, dim=-1)
            weights = adaptive_route_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        elif self.spec.kind == "adaptive_specialists":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("adaptive specialist gate parameters are unavailable")
            specialist_heads = self.n_heads - self.spec.free_heads
            gate_logits = torch.einsum(
                "bhtd,hdr->bhtr", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1, 3)
            route_probs = torch.softmax(gate_logits, dim=-1)
            weights = adaptive_specialist_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        elif self.spec.kind == "anchor_residual":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("anchor-residual gate parameters are unavailable")
            specialist_heads = self.n_heads - self.spec.free_heads
            gate_logits = torch.einsum(
                "bhtd,hdr->bhtr", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1, 3)
            route_probs = torch.softmax(gate_logits, dim=-1)
            weights = anchor_residual_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        elif self.spec.kind == "marginal_residual":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("marginal-residual gate parameters are unavailable")
            specialist_heads = self.n_heads - self.spec.free_heads
            gate_logits = torch.einsum(
                "bhtd,hd->bht", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1)
            route_probs = torch.sigmoid(gate_logits)
            weights = marginal_residual_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        elif self.spec.kind == "q_residual":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("Q-residual gate parameters are unavailable")
            specialist_heads = self.n_heads - self.spec.free_heads
            gate_logits = torch.einsum(
                "bhtd,hd->bht", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1)
            route_probs = torch.sigmoid(gate_logits)
            weights = q_residual_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        else:
            weights = apply_attention(logits, self.spec)
            self.last_route_probs = None
        self.last_weights = weights.detach()
        y = weights @ v
        y = y.transpose(1,2).contiguous().view(b,t,c)
        return self.dropout(self.out(y))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, spec: AttentionSpec) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg,spec)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.ff_mult * cfg.d_model
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model,hidden),nn.GELU(),nn.Linear(hidden,cfg.d_model),nn.Dropout(cfg.dropout))

    def forward(self,x:Tensor)->Tensor:
        x=x+self.attn(self.ln1(x))
        return x+self.mlp(self.ln2(x))


class ByteGPT(nn.Module):
    def __init__(self,cfg:ModelConfig,spec:AttentionSpec|Sequence[AttentionSpec])->None:
        super().__init__();self.cfg=cfg
        specs=(spec,)*cfg.n_layers if isinstance(spec,AttentionSpec) else tuple(spec)
        if len(specs)!=cfg.n_layers:raise ValueError("one attention spec per layer required")
        self.token=nn.Emb[... ELLIPSIZATION ...]            probabilities.new_tensor(0.0),
            specialist_local.sum() / attn.n_heads,
            specialist_balance.sum() / attn.n_heads,
            probabilities.new_tensor(free_heads / attn.n_heads),
        ))
        specialist_rows = torch.stack((
            torch.zeros_like(specialist_local),
            specialist_local,
            specialist_balance,
            torch.zeros_like(specialist_local),
        ), dim=-1)
        free_rows = torch.zeros(free_heads, 4, device=probabilities.device)
        free_rows[:, 3] = 1.0
        per_head = torch.cat((specialist_rows, free_rows), dim=0)
        clipped = probabilities.clamp(min=tiny, max=1.0 - torch.finfo(probabilities.dtype).eps)
        entropy = -(
            clipped * clipped.log()
            + (1.0 - clipped) * (1.0 - clipped).clamp_min(tiny).log()
        ).mean()
        raw_balance = probabilities.mean()
        return {
            "names": ["self", "local", "balanced", "free"],
            "raw_mean": [0.0, float(1.0 - raw_balance), float(raw_balance), 0.0],
            "effective_mean": [float(value) for value in aggregate],
            "per_head_effective": [[float(value) for value in row] for row in per_head],
            "gate_entropy_nats": float(entropy),
            "free_floor": 0.0,
            "free_heads": free_heads,
            "adapt_budget": budget,
        }

    if attn.spec.kind == "q_residual":
        free_heads = int(attn.spec.free_heads)
        specialist_heads = attn.n_heads - free_heads
        budget = float(attn.spec.adapt_budget)
        anchor = probabilities.new_zeros(specialist_heads)
        anchor[-1] = 1.0
        balance = (
            (1.0 - budget) * anchor.view(1, specialist_heads, 1)
            + budget * probabilities
        )
        specialist_balance = balance.mean(dim=(0, 2))
        specialist_local = 1.0 - specialist_balance
        aggregate = torch.stack((
            probabilities.new_tensor(0.0),
            specialist_local.sum() / attn.n_heads,
            specialist_balance.sum() / attn.n_heads,
            probabilities.new_tensor(free_heads / attn.n_heads),
        ))
        specialist_rows = torch.stack((
            torch.zeros_like(specialist_local),
            specialist_local,
            specialist_balance,
            torch.zeros_like(specialist_local),
        ), dim=-1)
        free_rows = torch.zeros(free_heads, 4, device=probabilities.device)
        free_rows[:, 3] = 1.0
        per_head = torch.cat((specialist_rows, free_rows), dim=0)
        clipped = probabilities.clamp(min=tiny, max=1.0 - torch.finfo(probabilities.dtype).eps)
        entropy = -(
            clipped * clipped.log()
            + (1.0 - clipped) * (1.0 - clipped).clamp_min(tiny).log()
        ).mean()
        raw_balance = probabilities.mean()
        return {
            "names": ["self", "local", "balanced", "free"],
            "raw_mean": [0.0, float(1.0 - raw_balance), float(raw_balance), 0.0],
            "effective_mean": [float(value) for value in aggregate],
            "per_head_effective": [[float(value) for value in row] for row in per_head],
            "gate_entropy_nats": float(entropy),
            "free_floor": 0.0,
            "free_heads": free_heads,
            "anchor_pattern": "llb",
            "adapt_budget": budget,
            "init_strength": float(attn.spec.init_strength),
        }

    entropy = -(probabilities * probabilities.clamp_min(tiny).log()).sum(-1).mean()

    if attn.spec.kind == "adaptive_route":
        floor = float(attn.spec.free_floor)
        effective = (1.0 - floor) * probabilities
        effective = effective.clone()
        effective[..., 3] = effective[..., 3] + floor
        raw_mean = probabilities.mean(dim=(0, 1, 2))
        effective_mean = effective.mean(dim=(0, 1, 2))
        per_head = effective.mean(dim=(0, 2))
        return {
            "names": ["self", "local", "balanced", "free"],
            "raw_mean": [float(value) for value in raw_mean],
            "effective_mean": [float(value) for value in effective_mean],
            "per_head_effective": [[float(value) for value in row] for row in per_head],
            "gate_entropy_nats": float(entropy),
            "free_floor": floor,
            "free_heads": 0,
        }

    if attn.spec.kind == "adaptive_specialists":
        free_heads = int(attn.spec.free_heads)
        specialist_heads = attn.n_heads - free_heads
        specialist_fraction = specialist_heads / attn.n_heads
        raw_specialist = probabilities.mean(dim=(0, 1, 2))
        aggregate = torch.cat(
            (specialist_fraction * raw_specialist, probabilities.new_tensor([free_heads / attn.n_heads]))
        )
        specialist_per_head = probabilities.mean(dim=(0, 2))
        specialist_per_head = torch.cat(
            (specialist_per_head, torch.zeros(specialist_heads, 1, device=probabilities.device)), dim=-1
        )
        free_rows = torch.zeros(free_heads, 4, device=probabilities.device)
        free_rows[:, 3] = 1.0
        per_head = torch.cat((specialist_per_head, free_rows), dim=0)
        return {
            "names": ["self", "local", "balanced", "free"],
            "raw_mean": [float(value) for value in raw_specialist] + [0.0],
            "effective_mean": [float(value) for value in aggregate],
            "per_head_effective": [[float(value) for value in row] for row in per_head],
            "gate_entropy_nats": float(entropy),
            "free_floor": 0.0,
            "free_heads": free_heads,
        }

    if attn.spec.kind == "anchor_residual":
        free_heads = int(attn.spec.free_heads)
        specialist_heads = attn.n_heads - free_heads
        anchor = probabilities.new_zeros(specialist_heads, 3)
        anchor[:, 1] = 1.0
        if attn.spec.anchor_pattern == "llb":
            anchor[-1, 1] = 0.0
            anchor[-1, 2] = 1.0
        budget = float(attn.spec.adapt_budget)
        effective = (1.0 - budget) * anchor.view(1, specialist_heads, 1, 3) + budget * probabilities
        specialist_per_head = effective.mean(dim=(0, 2))
        aggregate_specialist = specialist_per_head.sum(0) / attn.n_heads
        aggregate = torch.cat((aggregate_specialist, probabilities.new_tensor([free_heads / attn.n_heads])))
        specialist_rows = torch.cat((
            specialist_per_head,
            torch.zeros(specialist_heads, 1, device=probabilities.device),
        ), dim=-1)
        free_rows = torch.zeros(free_heads, 4, device=probabilities.device)
        free_rows[:, 3] = 1.0
        per_head = torch.cat((specialist_rows, free_rows), dim=0)
        raw_mean = probabilities.mean(dim=(0, 1, 2))
        return {
            "names": ["self", "local", "balanced", "free"],
            "raw_mean": [float(value) for value in raw_mean] + [0.0],
            "effective_mean": [float(value) for value in aggregate],
            "per_head_effective": [[float(value) for value in row] for row in per_head],
            "gate_entropy_nats": float(entropy),
            "free_floor": 0.0,
            "free_heads": free_heads,
            "anchor_pattern": attn.spec.anchor_pattern,
            "adapt_budget": budget,
        }
    return None


@dataclass(frozen=True,slots=True)
class TrainConfig:
    steps:int=260;batch_size:int=16;learning_rate:float=3e-4;weight_decay:float=.01;grad_clip:float=1.0;eval_batches:int=12;log_every:int=130;torch_threads:int=4

def seed_all(seed:int,threads:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass

@torch.no_grad()
def evaluate(model:ByteGPT,corpus:ByteCorpus,split:str,cfg:TrainConfig,seed:int)->float:
    model.eval();g=torch.Generator().manual_seed(seed);total=0.0
    for _ in range(cfg.eval_batches):
        x,y=corpus.sample(split,batch_size=cfg.batch_size,context=model.cfg.context,generator=g);total+=model.loss(x,y).item()
    model.train();return total/cfg.eval_batches/math.log(2)

def train_lm(corpus:ByteCorpus,model_cfg:ModelConfig,spec:AttentionProgram,cfg:TrainConfig,*,seed:int,verbose:bool=False,evaluate_test:bool=False)->dict[str,Any]:
    seed_all(seed,cfg.torch_threads);model=ByteGPT(model_cfg,spec);opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay);g=torch.Generator().manual_seed(seed+17171)
    curve=[];tokens=0;start=time.perf_counter();model.train();label=attention_program_label(spec)
    for step in range(1,cfg.steps+1):
        x,y=corpus.sample('train',batch_size=cfg.batch_size,context=model_cfg.context,generator=g);opt.zero_grad(set_to_none=True);loss=model.loss(x,y)
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at step {step}')
        loss.backward();gn=float(clip_grad_norm_(model.parameters(),cfg.grad_clip));opt.step();tokens+=x.numel()
        if step==1 or step==cfg.steps or step%cfg.log_every==0:
            curve.append({'step':step,'train_bpb':loss.item()/math.log(2),'grad_norm':gn})
            if verbose:print(f'{label:60s} seed={seed} step={step:4d} bpb={curve[-1]["train_bpb"]:.4f}',flush=True)
    elapsed=time.perf_counter()-start
    val=evaluate(model,corpus,'val',cfg,seed+9001);test=evaluate(model,corpus,'test',cfg,seed+19001) if evaluate_test else None
    pg=torch.Generator().manual_seed(seed+55);px,_=corpus.sample('val',batch_size=min(4,cfg.batch_size),context=model_cfg.context,generator=pg);w=model.attention_weights(px,0)
    result={'operator':label,'spec':attention_program_payload(spec),'seed':seed,'steps':cfg.steps,'val_bpb':val,'test_bpb':test,'tokens_per_second':tokens/max(elapsed,1e-9),'elapsed_seconds':elapsed,'parameters':sum(p.numel() for p in model.parameters()),'curve':curve,'attention':attention_metrics(w),'route':adaptive_route_metrics(model,0),'model_config':asdict(model_cfg),'train_config':asdict(cfg)}
    if verbose:print(f'DONE {label:60s} val={val:.6f} test={test}',flush=True)
    return result
```



Analyzed

```
bash -lc cd /mnt/data/attention-moonshot-nope-v1 && sed -n '120,300p' src/attention_moonshot/model.py && sed -n '1,260p' src/attention_moonshot/operators.py && sed -n '260,620p' src/attention_moonshot/operators.py
                "bhtd,hdr->bhtr", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1, 3)
            route_probs = torch.softmax(gate_logits, dim=-1)
            weights = adaptive_specialist_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        elif self.spec.kind == "anchor_residual":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("anchor-residual gate parameters are unavailable")
            specialist_heads = self.n_heads - self.spec.free_heads
            gate_logits = torch.einsum(
                "bhtd,hdr->bhtr", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1, 3)
            route_probs = torch.softmax(gate_logits, dim=-1)
            weights = anchor_residual_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        elif self.spec.kind == "marginal_residual":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("marginal-residual gate parameters are unavailable")
            specialist_heads = self.n_heads - self.spec.free_heads
            gate_logits = torch.einsum(
                "bhtd,hd->bht", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1)
            route_probs = torch.sigmoid(gate_logits)
            weights = marginal_residual_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        elif self.spec.kind == "q_residual":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("Q-residual gate parameters are unavailable")
            specialist_heads = self.n_heads - self.spec.free_heads
            gate_logits = torch.einsum(
                "bhtd,hd->bht", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1)
            route_probs = torch.sigmoid(gate_logits)
            weights = q_residual_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        else:
            weights = apply_attention(logits, self.spec)
            self.last_route_probs = None
        self.last_weights = weights.detach()
        y = weights @ v
        y = y.transpose(1,2).contiguous().view(b,t,c)
        return self.dropout(self.out(y))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, spec: AttentionSpec) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg,spec)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.ff_mult * cfg.d_model
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model,hidden),nn.GELU(),nn.Linear(hidden,cfg.d_model),nn.Dropout(cfg.dropout))

    def forward(self,x:Tensor)->Tensor:
        x=x+self.attn(self.ln1(x))
        return x+self.mlp(self.ln2(x))


class ByteGPT(nn.Module):
    def __init__(self,cfg:ModelConfig,spec:AttentionSpec|Sequence[AttentionSpec])->None:
        super().__init__();self.cfg=cfg
        specs=(spec,)*cfg.n_layers if isinstance(spec,AttentionSpec) else tuple(spec)
        if len(specs)!=cfg.n_layers:raise ValueError("one attention spec per layer required")
        self.token=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.position=nn.Embedding(cfg.context,cfg.d_model) if cfg.position_mode == "learned" else None
        self.blocks=nn.ModuleList(Block(cfg,s) for s in specs)
        self.final_ln=nn.LayerNorm(cfg.d_model)
        self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module:nn.Module)->None:
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,tokens:Tensor)->Tensor:
        b,t=tokens.shape
        if t>self.cfg.context and self.cfg.position_mode == "learned":
            raise ValueError("sequence exceeds configured context")
        x=self.token(tokens)
        if self.position is not None:
            pos=torch.arange(t,device=tokens.device)
            x=x+self.position(pos)
        for block in self.blocks:x=block(x)
        return self.lm_head(self.final_ln(x))

    def loss(self,x:Tensor,y:Tensor,loss_mask:Tensor|None=None)->Tensor:
        logits=self(x)
        per=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='none').view_as(y)
        if loss_mask is None:return per.mean()
        return (per*loss_mask).sum()/loss_mask.sum().clamp_min(1)

    def attention_weights(self,x:Tensor,layer:int=0)->Tensor:
        self(x)
        w=self.blocks[layer].attn.last_weights
        if w is None:raise RuntimeError("attention weights unavailable")
        return w
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class AttentionSpec:
    kind: str = "softmax"
    alpha: float = 1.0
    usage_decay: float = 1.0
    eps: float = 0.0
    diagonal_debit: float = 0.0
    slope: float = 1.0
    local_heads: int = 0
    self_heads: int = 0
    balanced_heads: int = 0
    reservoir_logit: float = -1.0
    free_floor: float = 0.25
    free_heads: int = 1
    specialist_init: str = "llb"
    init_strength: float = 4.0
    anchor_pattern: str = "llb"
    adapt_budget: float = 0.25

    def label(self) -> str:
        p: list[str] = []
        if self.kind in {"prefix", "prefix_log", "raps", "reservoir_prefix", "geometric_prefix"}:
            p.append(f"a={self.alpha:g}")
        if self.kind == "quad_route" and self.alpha != 1.0:
            p.append(f"a={self.alpha:g}")
        if self.kind in {"geometric_prefix", "quad_route"} and self.usage_decay != 1.0:
            p.append(f"lam={self.usage_decay:g}")
        if self.diagonal_debit:
            p.append(f"dd={self.diagonal_debit:g}")
        if self.kind in {"recency", "past_recency"}:
            p.append(f"s={self.slope:g}")
        if self.kind == "dual_route":
            p.extend((f"lh={self.local_heads}", f"s={self.slope:g}"))
        if self.kind == "slg":
            p.extend((f"sh={self.self_heads}", f"lh={self.local_heads}", f"s={self.slope:g}"))
        if self.kind == "quad_route":
            p.extend((f"sh={self.self_heads}", f"lh={self.local_heads}", f"bh={self.balanced_heads}", f"s={self.slope:g}"))
        if self.kind == "reservoir_prefix":
            p.append(f"rho={self.reservoir_logit:g}")
        if self.kind == "adaptive_route":
            p.extend((f"s={self.slope:g}", f"ff={self.free_floor:g}"))
        if self.kind == "adaptive_specialists":
            p.extend((
                f"fh={self.free_heads}",
                f"s={self.slope:g}",
                f"si={self.specialist_init}",
                f"is={self.init_strength:g}",
            ))
        if self.kind == "anchor_residual":
            p.extend((
                f"fh={self.free_heads}",
                f"s={self.slope:g}",
                f"ap={self.anchor_pattern}",
                f"ab={self.adapt_budget:g}",
            ))
        if self.kind == "marginal_residual":
            p.extend((
                f"fh={self.free_heads}",
                f"s={self.slope:g}",
                f"ab={self.adapt_budget:g}",
            ))
        if self.kind == "q_residual":
            p.extend((
                f"fh={self.free_heads}",
                f"s={self.slope:g}",
                f"is={self.init_strength:g}",
                f"ab={self.adapt_budget:g}",
            ))
        return self.kind if not p else f"{self.kind}[{','.join(p)}]"


def causal_mask(length: int, device: torch.device | str) -> Tensor:
    if length < 1:
        raise ValueError("length must be positive")
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()


def _validate_logits(logits: Tensor) -> Tensor:
    if logits.ndim != 4 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("expected square [batch, heads, query, key] logits")
    return causal_mask(logits.shape[-1], logits.device)


def _masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    return logits.masked_fill(~mask, -torch.inf)


def _base(logits: Tensor) -> tuple[Tensor, Tensor]:
    mask = _validate_logits(logits)
    weights = torch.softmax(_masked_logits(logits, mask), dim=-1).masked_fill(~mask, 0.0)
    return weights, mask


def _log_base(logits: Tensor) -> tuple[Tensor, Tensor]:
    mask = _validate_logits(logits)
    return torch.log_softmax(_masked_logits(logits, mask), dim=-1).masked_fill(~mask, -torch.inf), mask


def _row_normalize(raw: Tensor, mask: Tensor) -> Tensor:
    raw = raw.masked_fill(~mask, 0.0)
    floor = torch.finfo(raw.dtype).tiny
    total = raw.sum(-1, keepdim=True).clamp_min(floor)
    return (raw / total).masked_fill(~mask, 0.0)


def _debit_log_diagonal(log_weights: Tensor, debit: float, *, preserve_first: bool = True) -> Tensor:
    if debit <= 0:
        return log_weights
    t = log_weights.shape[-1]
    eye = torch.eye(t, dtype=torch.bool, device=log_weights.device).view(1, 1, t, t)
    out = torch.where(eye, log_weights - float(debit), log_weights)
    if preserve_first:
        out = out.clone()
        out[..., 0, 0] = log_weights[..., 0, 0]
    return out


def _debit_probability(weights: Tensor, debit: float, mask: Tensor) -> Tensor:
    if debit <= 0:
        return weights
    t = weights.shape[-1]
    eye = torch.eye(t, dtype=torch.bool, device=weights.device).view(1, 1, t, t)
    raw = torch.where(eye, weights * ma[... ELLIPSIZATION ...]e ValueError(f"expected gate probabilities with shape {expected}")

    specialist_logits = logits[:, :specialist_heads]
    self_route = identity_attention(specialist_logits)
    local_route = recency_attention(specialist_logits, spec.slope, exclude_self=True)
    balanced_route = prefix_log(specialist_logits, replace(spec, kind="raps"))
    branches = torch.stack((self_route, local_route, balanced_route), dim=-2)
    specialist = (branches * gate_probs.unsqueeze(-1)).sum(dim=-2)
    free = _base(logits[:, specialist_heads:])[0]
    return torch.cat((specialist, free), dim=1)


def specialist_anchor(
    pattern: str,
    specialist_heads: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Return one-hot self/local/balanced anchors for specialist heads."""
    if specialist_heads < 1:
        raise ValueError("specialist_heads must be positive")
    if pattern not in {"lll", "llb"}:
        raise ValueError("anchor_pattern must be 'lll' or 'llb'")
    anchor = torch.zeros(specialist_heads, 3, device=device, dtype=dtype)
    anchor[:, 1] = 1.0
    if pattern == "llb":
        anchor[-1, 1] = 0.0
        anchor[-1, 2] = 1.0
    return anchor


def anchor_residual_attention(
    logits: Tensor,
    gate_probs: Tensor,
    spec: AttentionSpec,
    *,
    return_routes: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Anchor-preserving residual routing plus whole free softmax heads.

    Each specialist keeps ``1-adapt_budget`` of a fixed LLL/LLB role and
    allocates only the residual budget through a query-dependent gate.
    """
    if spec.kind != "anchor_residual":
        raise ValueError("kind='anchor_residual' required")
    if not 0.0 <= spec.adapt_budget <= 1.0:
        raise ValueError("adapt_budget must be in [0, 1]")
    _validate_logits(logits)
    heads = logits.shape[1]
    if not 1 <= spec.free_heads < heads:
        raise ValueError("free_heads must be in 1..heads-1")
    specialist_heads = heads - spec.free_heads
    expected = (logits.shape[0], specialist_heads, logits.shape[-2], 3)
    if gate_probs.shape != expected:
        raise ValueError(f"expected gate probabilities with shape {expected}")

    anchor = specialist_anchor(
        spec.anchor_pattern,
        specialist_heads,
        device=logits.device,
        dtype=logits.dtype,
    ).view(1, specialist_heads, 1, 3)
    budget = float(spec.adapt_budget)
    effective = (1.0 - budget) * anchor + budget * gate_probs

    specialist_logits = logits[:, :specialist_heads]
    branches = torch.stack((
        identity_attention(specialist_logits),
        recency_attention(specialist_logits, spec.slope, exclude_self=True),
        prefix_log(specialist_logits, replace(spec, kind="raps")),
    ), dim=-2)
    specialist = (branches * effective.unsqueeze(-1)).sum(dim=-2)
    free = _base(logits[:, specialist_heads:])[0]
    weights = torch.cat((specialist, free), dim=1)
    if return_routes:
        return weights, effective.expand(logits.shape[0], -1, logits.shape[-2], -1)
    return weights


def marginal_residual_attention(
    logits: Tensor,
    gate_probs: Tensor,
    spec: AttentionSpec,
    *,
    return_routes: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Bounded query-dependent interpolation from local to prefix-balanced routing.

    Specialist heads remain local by default and may divert at most
    ``adapt_budget`` of each query into the causal Prefix-Sinkhorn branch.
    The final ``free_heads`` remain exact causal-softmax heads.
    """
    if spec.kind != "marginal_residual":
        raise ValueError("kind='marginal_residual' required")
    if not 0.0 <= spec.adapt_budget <= 1.0:
        raise ValueError("adapt_budget must be in [0, 1]")
    _validate_logits(logits)
    heads = logits.shape[1]
    if not 1 <= spec.free_heads < heads:
        raise ValueError("free_heads must be in 1..heads-1")
    specialist_heads = heads - spec.free_heads
    expected = (logits.shape[0], specialist_heads, logits.shape[-2])
    if gate_probs.shape != expected:
        raise ValueError(f"expected gate probabilities with shape {expected}")
    if torch.any((gate_probs < 0.0) | (gate_probs > 1.0)):
        raise ValueError("gate probabilities must be in [0, 1]")

    specialist_logits = logits[:, :specialist_heads]
    local = recency_attention(specialist_logits, spec.slope, exclude_self=True)
    balanced = prefix_log(specialist_logits, replace(spec, kind="raps"))
    balance = float(spec.adapt_budget) * gate_probs
    specialist = local + balance.unsqueeze(-1) * (balanced - local)
    free = _base(logits[:, specialist_heads:])[0]
    weights = torch.cat((specialist, free), dim=1)
    if return_routes:
        return weights, balance
    return weights


def q_residual_attention(
    logits: Tensor,
    gate_probs: Tensor,
    spec: AttentionSpec,
    *,
    return_routes: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """LLB-anchored local/Prefix interpolation plus whole free heads.

    The final specialist head is anchored to Prefix-Sinkhorn balance and all
    preceding specialist heads are anchored to strictly-past local attention.
    ``adapt_budget`` bounds how far each query may move away from that fixed Q
    foundation, while the final ``free_heads`` remain exact causal softmax.
    """
    if spec.kind != "q_residual":
        raise ValueError("kind='q_residual' required")
    if not 0.0 <= spec.adapt_budget <= 1.0:
        raise ValueError("adapt_budget must be in [0, 1]")
    _validate_logits(logits)
    heads = logits.shape[1]
    if not 1 <= spec.free_heads < heads:
        raise ValueError("free_heads must be in 1..heads-1")
    specialist_heads = heads - spec.free_heads
    expected = (logits.shape[0], specialist_heads, logits.shape[-2])
    if gate_probs.shape != expected:
        raise ValueError(f"expected gate probabilities with shape {expected}")
    if torch.any((gate_probs < 0.0) | (gate_probs > 1.0)):
        raise ValueError("gate probabilities must be in [0, 1]")

    specialist_logits = logits[:, :specialist_heads]
    local = recency_attention(specialist_logits, spec.slope, exclude_self=True)
    balanced = prefix_log(specialist_logits, replace(spec, kind="raps"))
    anchor = torch.zeros_like(gate_probs)
    anchor[:, -1] = 1.0
    budget = float(spec.adapt_budget)
    balance = (1.0 - budget) * anchor + budget * gate_probs
    if budget == 0.0:
        specialist = torch.cat((local[:, :-1], balanced[:, -1:]), dim=1)
    else:
        specialist = local + balance.unsqueeze(-1) * (balanced - local)
    free = _base(logits[:, specialist_heads:])[0]
    weights = torch.cat((specialist, free), dim=1)
    if return_routes:
        return weights, balance
    return weights


def leaky_masked_sinkhorn(logits: Tensor, iterations: int = 1) -> Tensor:
    base, mask = _base(logits)
    out = base
    for _ in range(iterations):
        out = out / out.sum(-2, keepdim=True).clamp_min(torch.finfo(out.dtype).tiny)
        out = _row_normalize(out, mask)
    return out


def apply_attention(logits: Tensor, spec: AttentionSpec) -> Tensor:
    if spec.kind == "softmax":
        return _base(logits)[0]
    if spec.kind == "prefix":
        return prefix_probability(logits, spec)
    if spec.kind == "prefix_log":
        return prefix_log(logits, spec)
    if spec.kind == "raps":
        return prefix_log(logits, spec)
    if spec.kind == "geometric_prefix":
        return geometric_prefix_log(logits, spec)
    if spec.kind == "reservoir_prefix":
        return reservoir_prefix(logits, spec)
    if spec.kind == "recency":
        return recency_attention(logits, spec.slope)
    if spec.kind == "past_recency":
        return recency_attention(logits, spec.slope, exclude_self=True)
    if spec.kind == "identity":
        return identity_attention(logits)
    if spec.kind == "dual_route":
        h = logits.shape[1]
        if not 0 <= spec.local_heads <= h:
            raise ValueError("local_heads out of range")
        n = spec.local_heads
        parts = []
        if n:
            parts.append(recency_attention(logits[:, :n], spec.slope))
        if n < h:
            parts.append(prefix_log(logits[:, n:], replace(spec, kind="prefix_log")))
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    if spec.kind == "slg":
        h = logits.shape[1]
        if spec.self_heads < 0 or spec.local_heads < 0 or spec.self_heads + spec.local_heads > h:
            raise ValueError("self_heads + local_heads out of range")
        a = spec.self_heads; b = a + spec.local_heads
        parts = []
        if a:
            parts.append(identity_attention(logits[:, :a]))
        if b > a:
            parts.append(recency_attention(logits[:, a:b], spec.slope, exclude_self=True))
        if b < h:
            parts.append(prefix_log(logits[:, b:], replace(spec, kind="prefix_log")))
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    if spec.kind == "quad_route":
        h = logits.shape[1]
        counts = (spec.self_heads, spec.local_heads, spec.balanced_heads)
        if any(n < 0 for n in counts) or sum(counts) > h:
            raise ValueError("self_heads + local_heads + balanced_heads out of range")
        a = spec.self_heads
        b = a + spec.local_heads
        c = b + spec.balanced_heads
        parts = []
        if a:
            parts.append(identity_attention(logits[:, :a]))
        if b > a:
            parts.append(recency_attention(logits[:, a:b], spec.slope, exclude_self=True))
        if c > b:
            balanced_spec = replace(spec, kind="geometric_prefix")
            parts.append(geometric_prefix_log(logits[:, b:c], balanced_spec))
        if c < h:
            parts.append(_base(logits[:, c:])[0])
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    if spec.kind == "leaky_sinkhorn":
        return leaky_masked_sinkhorn(logits)
    raise ValueError(f"unsupported attention kind: {spec.kind}")
```



Analyzed

```
bash -lc cd /mnt/data/attention-moonshot-nope-v1 && nl -ba src/attention_moonshot/synthetic.py | sed -n '1,420p'
     1	from __future__ import annotations
     2	from dataclasses import asdict,dataclass
     3	import math,time
     4	from typing import Any
     5	import torch
     6	from torch.nn.utils import clip_grad_norm_
     7	from .experiment import AttentionProgram,adaptive_route_metrics,attention_program_label,attention_program_payload,seed_all
     8	from .model import ByteGPT,ModelConfig
     9	from .operators import AttentionSpec
    10	
    11	@dataclass(frozen=True,slots=True)
    12	class SyntheticConfig:
    13	    task:str='copy';steps:int=300;batch_size:int=32;learning_rate:float=1e-3;weight_decay:float=0.0;grad_clip:float=1.0;eval_batches:int=20;lag:int=64;pairs:int=16;queries:int=4;torch_threads:int=4
    14	
    15	def delayed_copy_batch(*,batch_size:int,context:int,vocab_size:int,lag:int,generator:torch.Generator):
    16	    if lag>=context:raise ValueError('lag must be smaller than context')
    17	    seed=torch.randint(1,vocab_size,(batch_size,lag),generator=generator);seq=seed.repeat(1,(context+lag-1)//lag)[:,:context]
    18	    x=seq[:,:-1];y=seq[:,1:];m=torch.zeros_like(y,dtype=torch.float32);m[:,lag-1:]=1;return x,y,m
    19	
    20	def random_lag_copy_batch(
    21	    *,
    22	    batch_size: int,
    23	    context: int,
    24	    vocab_size: int,
    25	    min_lag: int,
    26	    max_lag: int,
    27	    generator: torch.Generator,
    28	):
    29	    """Sample one deterministic lag for a whole batch and build periodic copy data."""
    30	    if min_lag < 1:
    31	        raise ValueError("min_lag must be positive")
    32	    if min_lag > max_lag:
    33	        raise ValueError("min_lag must not exceed max_lag")
    34	    if max_lag >= context:
    35	        raise ValueError("max_lag must be smaller than context")
    36	    lag = int(torch.randint(min_lag, max_lag + 1, (1,), generator=generator).item())
    37	    x, y, mask = delayed_copy_batch(
    38	        batch_size=batch_size,
    39	        context=context,
    40	        vocab_size=vocab_size,
    41	        lag=lag,
    42	        generator=generator,
    43	    )
    44	    return x, y, mask, lag
    45	
    46	
    47	def associative_recall_batch(*,batch_size:int,context:int,pairs:int,generator:torch.Generator):
    48	    if 2*pairs+2>context or pairs>31:raise ValueError('invalid pairs/context')
    49	    keys=torch.stack([torch.randperm(31,generator=generator)[:pairs]+1 for _ in range(batch_size)])
    50	    vals=torch.randint(32,64,(batch_size,pairs),generator=generator);qi=torch.randint(0,pairs,(batch_size,),generator=generator)
    51	    seq=torch.zeros(batch_size,context,dtype=torch.long);seq[:,:2*pairs:2]=keys;seq[:,1:2*pairs:2]=vals
    52	    seq[:,2*pairs:-1]=torch.randint(1,64,(batch_size,context-2*pairs-1),generator=generator)
    53	    rows=torch.arange(batch_size);seq[:,-1]=keys[rows,qi];target=vals[rows,qi]
    54	    y=torch.zeros_like(seq);y[:,-1]=target;m=torch.zeros_like(seq,dtype=torch.float32);m[:,-1]=1;return seq,y,m
    55	
    56	def multi_query_recall_batch(*,batch_size:int,context:int,pairs:int,queries:int,generator:torch.Generator):
    57	    if pairs < 1 or pairs > 31:
    58	        raise ValueError("pairs must be in 1..31")
    59	    if queries < 1 or queries > pairs:
    60	        raise ValueError("queries must be in 1..pairs")
    61	    if 2*pairs+queries>context:
    62	        raise ValueError("context too short for pair table and queries")
    63	    keys=torch.stack([torch.randperm(31,generator=generator)[:pairs]+1 for _ in range(batch_size)])
    64	    vals=torch.randint(32,64,(batch_size,pairs),generator=generator)
    65	    query_indices=torch.stack([torch.randperm(pairs,generator=generator)[:queries] for _ in range(batch_size)])
    66	    rows=torch.arange(batch_size).unsqueeze(1)
    67	    query_keys=keys[rows,query_indices]
    68	    targets=vals[rows,query_indices]
    69	    x=torch.zeros(batch_size,context,dtype=torch.long)
    70	    x[:,:2*pairs:2]=keys
    71	    x[:,1:2*pairs:2]=vals
    72	    x[:,-queries:]=query_keys
    73	    y=torch.zeros_like(x)
    74	    y[:,-queries:]=targets
    75	    m=torch.zeros_like(x,dtype=torch.float32)
    76	    m[:,-queries:]=1
    77	    return x,y,m
    78	
    79	
    80	def _batch(mc:ModelConfig,cfg:SyntheticConfig,g:torch.Generator):
    81	    if cfg.task=='copy':return delayed_copy_batch(batch_size=cfg.batch_size,context=mc.context,vocab_size=mc.vocab_size,lag=cfg.lag,generator=g)
    82	    if cfg.task=='recall':return associative_recall_batch(batch_size=cfg.batch_size,context=mc.context,pairs=cfg.pairs,generator=g)
    83	    if cfg.task=='mqar':return multi_query_recall_batch(batch_size=cfg.batch_size,context=mc.context,pairs=cfg.pairs,queries=cfg.queries,generator=g)
    84	    raise ValueError(cfg.task)
    85	
    86	@torch.no_grad()
    87	def _eval(model:ByteGPT,cfg:SyntheticConfig,seed:int):
    88	    model.eval();g=torch.Generator().manual_seed(seed);loss=correct=count=0.0
    89	    for _ in range(cfg.eval_batches):
    90	        x,y,m=_batch(model.cfg,cfg,g);logits=model(x);per=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='none').view_as(y);loss+=(per*m).sum().item();correct+=((logits.argmax(-1)==y)&m.bool()).sum().item();count+=m.sum().item()
    91	    model.train();return loss/count/math.log(2),correct/count
    92	
    93	def train_synthetic(mc:ModelConfig,spec:AttentionProgram,cfg:SyntheticConfig,*,seed:int,verbose:bool=False)->dict[str,Any]:
    94	    seed_all(seed,cfg.torch_threads);model=ByteGPT(mc,spec);opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay);g=torch.Generator().manual_seed(seed+171);start=time.perf_counter();tokens=0
    95	    for step in range(1,cfg.steps+1):
    96	        x,y,m=_batch(mc,cfg,g);opt.zero_grad(set_to_none=True);loss=model.loss(x,y,m);loss.backward();clip_grad_norm_(model.parameters(),cfg.grad_clip);opt.step();tokens+=x.numel()
    97	        if verbose and (step==1 or step==cfg.steps or step%max(1,cfg.steps//4)==0):print(cfg.task,attention_program_label(spec),seed,step,loss.item()/math.log(2),flush=True)
    98	    bits,acc=_eval(model,cfg,seed+913)
    99	    probe_generator=torch.Generator().manual_seed(seed+1913)
   100	    probe_x,_,_=_batch(mc,cfg,probe_generator)
   101	    with torch.no_grad():model(probe_x)
   102	    return {'task':cfg.task,'operator':attention_program_label(spec),'spec':attention_program_payload(spec),'seed':seed,'eval_bits':bits,'eval_accuracy':acc,'tokens_per_second':tokens/max(time.perf_counter()-start,1e-9),'route':adaptive_route_metrics(model,0),'model_config':asdict(mc),'train_config':asdict(cfg)}
   103	
   104	
   105	@torch.no_grad()
   106	def evaluate_copy_lag(
   107	    model: ByteGPT,
   108	    cfg: SyntheticConfig,
   109	    *,
   110	    lag: int,
   111	    seed: int,
   112	    context: int | None = None,
   113	) -> tuple[float, float]:
   114	    """Evaluate a trained model on one exact delayed-copy lag."""
   115	    evaluation_context = model.cfg.context if context is None else int(context)
   116	    if lag >= evaluation_context:
   117	        raise ValueError("lag must be smaller than evaluation context")
   118	    model.eval()
   119	    generator = torch.Generator().manual_seed(seed)
   120	    loss = correct = count = 0.0
   121	    for _ in range(cfg.eval_batches):
   122	        x, y, mask = delayed_copy_batch(
   123	            batch_size=cfg.batch_size,
   124	            context=evaluation_context,
   125	            vocab_size=model.cfg.vocab_size,
   126	            lag=lag,
   127	            generator=generator,
   128	        )
   129	        logits = model(x)
   130	        per = torch.nn.functional.cross_entropy(
   131	            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none"
   132	        ).view_as(y)
   133	        loss += (per * mask).sum().item()
   134	        correct += ((logits.argmax(-1) == y) & mask.bool()).sum().item()
   135	        count += mask.sum().item()
   136	    model.train()
   137	    return loss / count / math.log(2), correct / count
   138	
   139	
   140	def train_variable_lag_copy(
   141	    mc: ModelConfig,
   142	    spec: AttentionProgram,
   143	    cfg: SyntheticConfig,
   144	    *,
   145	    train_lag_min: int,
   146	    train_lag_max: int,
   147	    eval_lags: tuple[int, ...],
   148	    seed: int,
   149	    verbose: bool = False,
   150	) -> dict[str, Any]:
   151	    """Train across a lag interval, then measure every requested fixed lag."""
   152	    if not eval_lags:
   153	        raise ValueError("eval_lags must not be empty")
   154	    if any(lag < 1 or lag >= mc.context for lag in eval_lags):
   155	        raise ValueError("every eval lag must be in 1..context-1")
   156	    seed_all(seed, cfg.torch_threads)
   157	    model = ByteGPT(mc, spec)
   158	    optimizer = torch.optim.AdamW(
   159	        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
   160	    )
   161	    generator = torch.Generator().manual_seed(seed + 171)
   162	    start = time.perf_counter()
   163	    tokens = 0
   164	    lag_counts = {lag: 0 for lag in range(train_lag_min, train_lag_max + 1)}
   165	    model.train()
   166	    for step in range(1, cfg.steps + 1):
   167	        x, y, mask, lag = random_lag_copy_batch(
   168	            batch_size=cfg.batch_size,
   169	            context=mc.context,
   170	            vocab_size=mc.vocab_size,
   171	            min_lag=train_lag_min,
   172	            max_lag=train_lag_max,
   173	            generator=generator,
   174	        )
   175	        lag_counts[lag] += 1
   176	        optimizer.zero_grad(set_to_none=True)
   177	        loss = model.loss(x, y, mask)
   178	        loss.backward()
   179	        clip_grad_norm_(model.parameters(), cfg.grad_clip)
   180	        optimizer.step()
   181	        tokens += x.numel()
   182	        if verbose and (step == 1 or step == cfg.steps or step % max(1, cfg.steps // 4) == 0):
   183	            print(
   184	                "copy_variable", attention_program_label(spec), seed, step, lag,
   185	                loss.item() / math.log(2), flush=True,
   186	            )
   187	    sweep = []
   188	    for offset, lag in enumerate(sorted(set(eval_lags))):
   189	        bits, accuracy = evaluate_copy_lag(model, cfg, lag=lag, seed=seed + 913 + offset)
   190	        sweep.append({"lag": lag, "eval_bits": bits, "eval_accuracy": accuracy})
   191	    probe_generator = torch.Generator().manual_seed(seed + 1913)
   192	    probe_x, _, _, _ = random_lag_copy_batch(
   193	        batch_size=cfg.batch_size, context=mc.context, vocab_size=mc.vocab_size,
   194	        min_lag=train_lag_min, max_lag=train_lag_max, generator=probe_generator,
   195	    )
   196	    with torch.no_grad():
   197	        model(probe_x)
   198	    return {
   199	        "task": "copy_variable",
   200	        "operator": attention_program_label(spec),
   201	        "spec": attention_program_payload(spec),
   202	        "seed": seed,
   203	        "train_lag_range": [train_lag_min, train_lag_max],
   204	        "train_lag_counts": lag_counts,
   205	        "lag_sweep": sweep,
   206	        "tokens_per_second": tokens / max(time.perf_counter() - start, 1e-9),
   207	        "route": adaptive_route_metrics(model, 0),
   208	        "model_config": asdict(mc),
   209	        "train_config": asdict(cfg),
   210	    }
   211	
   212	
   213	def train_length_extrapolation_copy(
   214	    mc: ModelConfig,
   215	    spec: AttentionProgram,
   216	    cfg: SyntheticConfig,
   217	    *,
   218	    train_lag_min: int,
   219	    train_lag_max: int,
   220	    eval_cases: tuple[tuple[int, int], ...],
   221	    seed: int,
   222	    verbose: bool = False,
   223	) -> dict[str, Any]:
   224	    """Train at ``mc.context`` and evaluate longer position-free sequences."""
   225	    if not eval_cases:
   226	        raise ValueError("eval_cases must not be empty")
   227	    if any(context < 2 or lag < 1 or lag >= context for context, lag in eval_cases):
   228	        raise ValueError("each eval case requires 1 <= lag < context")
   229	    if any(context > mc.context for context, _ in eval_cases) and mc.position_mode != "none":
   230	        raise ValueError("longer-context evaluation requires position_mode='none'")
   231	    seed_all(seed, cfg.torch_threads)
   232	    model = ByteGPT(mc, spec)
   233	    optimizer = torch.optim.AdamW(
   234	        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
   235	    )
   236	    generator = torch.Generator().manual_seed(seed + 171)
   237	    start = time.perf_counter()
   238	    tokens = 0
   239	    lag_counts = {lag: 0 for lag in range(train_lag_min, train_lag_max + 1)}
   240	    model.train()
   241	    for step in range(1, cfg.steps + 1):
   242	        x, y, mask, lag = random_lag_copy_batch(
   243	            batch_size=cfg.batch_size,
   244	            context=mc.context,
   245	            vocab_size=mc.vocab_size,
   246	            min_lag=train_lag_min,
   247	            max_lag=train_lag_max,
   248	            generator=generator,
   249	        )
   250	        lag_counts[lag] += 1
   251	        optimizer.zero_grad(set_to_none=True)
   252	        loss = model.loss(x, y, mask)
   253	        loss.backward()
   254	        clip_grad_norm_(model.parameters(), cfg.grad_clip)
   255	        optimizer.step()
   256	        tokens += x.numel()
   257	        if verbose and (step == 1 or step == cfg.steps or step % max(1, cfg.steps // 4) == 0):
   258	            print(
   259	                "copy_length", attention_program_label(spec), seed, step, lag,
   260	                loss.item() / math.log(2), flush=True,
   261	            )
   262	    sweep = []
   263	    for offset, (context, lag) in enumerate(eval_cases):
   264	        bits, accuracy = evaluate_copy_lag(
   265	            model, cfg, lag=lag, context=context, seed=seed + 913 + offset
   266	        )
   267	        sweep.append({
   268	            "context": int(context), "lag": int(lag),
   269	            "eval_bits": bits, "eval_accuracy": accuracy,
   270	        })
   271	    probe_generator = torch.Generator().manual_seed(seed + 1913)
   272	    probe_x, _, _, _ = random_lag_copy_batch(
   273	        batch_size=cfg.batch_size, context=mc.context, vocab_size=mc.vocab_size,
   274	        min_lag=train_lag_min, max_lag=train_lag_max, generator=probe_generator,
   275	    )
   276	    with torch.no_grad():
   277	        model(probe_x)
   278	    return {
   279	        "task": "copy_length_extrapolation",
   280	        "operator": attention_program_label(spec),
   281	        "spec": attention_program_payload(spec),
   282	        "seed": seed,
   283	        "train_context": mc.context,
   284	        "train_lag_range": [train_lag_min, train_lag_max],
   285	        "train_lag_counts": lag_counts,
   286	        "context_sweep": sweep,
   287	        "tokens_per_second": tokens / max(time.perf_counter() - start, 1e-9),
   288	        "route": adaptive_route_metrics(model, 0),
   289	        "model_config": asdict(mc),
   290	        "train_config": asdict(cfg),
   291	    }
```



Analyzed

```
bash -lc cd /mnt/data/attention-moonshot-nope-v1 && nl -ba src/attention_moonshot/synthetic.py | sed -n '290,520p'
   290	        "train_config": asdict(cfg),
   291	    }
```

Auditing task design

I’m checking the experiment for hidden confounds before extending it.  Boundary clamping can duplicate the target key, making some trials  ambiguous, while padding, sequence length, and distractor placement may  also affect interpretation.



Inspecting Candidate Attention Programs

```
bash -lc cd /mnt/data/attention-moonshot-nope-v1 && nl -ba src/attention_moonshot/candidates.py | sed -n '1,380p' && printf '\n--- depth scripts ---\n' && nl -ba scripts/run_depth_program_screen.py | sed -n '1,340p' && printf '\n--- atomic ---\n' && nl -ba scripts/run_depth_program_atomic.py | sed -n '1,320p'
```



Analyzed

```
     1	from __future__ import annotations
     2	
     3	from .operators import AttentionSpec
     4	
     5	
     6	def free_head_count(spec: AttentionSpec, n_heads: int) -> int:
     7	    """Return the number of untouched causal-softmax heads in a route spec."""
     8	    if n_heads < 1:
     9	        raise ValueError("n_heads must be positive")
    10	    if spec.kind == "softmax":
    11	        return n_heads
    12	    if spec.kind == "quad_route":
    13	        used = spec.self_heads + spec.local_heads + spec.balanced_heads
    14	        if min(spec.self_heads, spec.local_heads, spec.balanced_heads) < 0 or used > n_heads:
    15	            raise ValueError("invalid quad-route head allocation")
    16	        return n_heads - used
    17	    if spec.kind in {"adaptive_specialists", "anchor_residual", "marginal_residual", "q_residual"}:
    18	        if not 1 <= spec.free_heads < n_heads:
    19	            raise ValueError("invalid specialist free-head allocation")
    20	        return spec.free_heads
    21	    return 0
    22	
    23	
    24	def route_screen_candidates(n_heads: int = 4) -> list[AttentionSpec]:
    25	    """Canonical baselines plus a free-head-preserving four-route grid.
    26	
    27	    The four-route candidates deliberately keep at least one ordinary causal
    28	    softmax head. This turns the long-range path into an architectural invariant
    29	    rather than a behavior the optimizer must rediscover.
    30	    """
    31	    if n_heads != 4:
    32	        raise ValueError("the preregistered route screen uses exactly four heads")
    33	
    34	    specs = [
    35	        AttentionSpec(),
    36	        AttentionSpec(kind="raps", diagonal_debit=3.0),
    37	        AttentionSpec(kind="recency", slope=1.25),
    38	        AttentionSpec(kind="dual_route", local_heads=3, slope=1.25, diagonal_debit=3.0),
    39	        AttentionSpec(kind="slg", self_heads=1, local_heads=2, slope=0.8, diagonal_debit=3.0),
    40	    ]
    41	
    42	    allocations = (
    43	        # canonical one-head-per-role architecture
    44	        (1, 1, 1, 0.8),
    45	        (1, 1, 1, 1.25),
    46	        # two free heads: isolate whether explicit self is necessary
    47	        (0, 1, 1, 0.8),
    48	        (0, 1, 1, 1.25),
    49	        # one free head with extra local or balanced capacity
    50	        (0, 2, 1, 0.8),
    51	        (0, 2, 1, 1.25),
    52	        (0, 1, 2, 0.8),
    53	        (1, 0, 2, 0.8),
    54	        # causal controls that retain free heads but remove balancing
    55	        (1, 1, 0, 0.8),
    56	        (0, 2, 0, 1.25),
    57	        # archive winner among free-head controls: three local specialists
    58	        (0, 3, 0, 0.8),
    59	    )
    60	    specs.extend(
    61	        AttentionSpec(
    62	            kind="quad_route",
    63	            self_heads=self_heads,
    64	            local_heads=local_heads,
    65	            balanced_heads=balanced_heads,
    66	            slope=slope,
    67	            diagonal_debit=3.0,
    68	        )
    69	        for self_heads, local_heads, balanced_heads, slope in allocations
    70	    )
    71	
    72	    labels = [spec.label() for spec in specs]
    73	    if len(labels) != len(set(labels)):
    74	        raise RuntimeError("route screen contains duplicate operator labels")
    75	    if any(free_head_count(spec, n_heads) < 1 for spec in specs if spec.kind == "quad_route"):
    76	        raise RuntimeError("every quad-route screen candidate must preserve a free head")
    77	    return specs
    78	
    79	
    80	def route_confirmation_candidates(n_heads: int = 4) -> list[AttentionSpec]:
    81	    """Baseline and fixed-route Pareto finalists for three-seed confirmation."""
    82	    if n_heads != 4:
    83	        raise ValueError("the preregistered confirmation uses exactly four heads")
    84	    specs = [
    85	        AttentionSpec(),
    86	        AttentionSpec(kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0, slope=0.8, diagonal_debit=3.0),
    87	        AttentionSpec(kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1, slope=0.8, diagonal_debit=3.0),
    88	        AttentionSpec(kind="quad_route", self_heads=0, local_heads=1, balanced_heads=1, slope=0.8, diagonal_debit=3.0),
    89	    ]
    90	    labels = [spec.label() for spec in specs]
    91	    if len(labels) != len(set(labels)):
    92	        raise RuntimeError("confirmation set contains duplicate labels")
    93	    if any(free_head_count(spec, n_heads) < 1 for spec in specs):
    94	        raise RuntimeError("every confirmation candidate must preserve a free head")
    95	    return specs
    96	
    97	
    98	
    99	def adaptive_route_candidates(n_heads: int = 4) -> list[AttentionSpec]:
   100	    """Generation-9 floor-gated adaptive routes and fixed-route controls."""
   101	    if n_heads != 4:
   102	        raise ValueError("the adaptive route screen uses exactly four heads")
   103	    specs = [
   104	        AttentionSpec(),
   105	        AttentionSpec(
   106	            kind="quad_route",
   107	            self_heads=0,
   108	            local_heads=2,
   109	            balanced_heads=1,
   110	            slope=0.8,
   111	            diagonal_debit=3.0,
   112	        ),
   113	        AttentionSpec(
   114	            kind="quad_route",
   115	            self_heads=0,
   116	            local_heads=3,
   117	            balanced_heads=0,
   118	            slope=0.8,
   119	        ),
   120	        AttentionSpec(kind="adaptive_route", slope=0.8, diagonal_debit=3.0, free_floor=0.125),
   121	        AttentionSpec(kind="adaptive_route", slope=0.8, diagonal_debit=3.0, free_floor=0.25),
   122	        AttentionSpec(kind="adaptive_route", slope=0.8, diagonal_debit=3.0, free_floor=0.5),
   123	        AttentionSpec(kind="adaptive_route", slope=1.25, diagonal_debit=3.0, free_floor=0.25),
   124	        AttentionSpec(kind="adaptive_route", slope=0.8, diagonal_debit=0.0, free_floor=0.25),
   125	    ]
   126	    labels = [spec.label() for spec in specs]
   127	    if len(labels) != len(set(labels)):
   128	        raise RuntimeError("adaptive route screen contains duplicate labels")
   129	    return specs
   130	
   131	
   132	def adaptive_specialist_candidates(n_heads: int = 4) -> list[AttentionSpec]:
   133	    """Generation-10 adaptive specialists with one untouched free head.
   134	
   135	    The adaptive heads are initialized at the two fixed-route winners: three
   136	    local specialists (LLL) or two local plus one balanced specialist (LLB).
   137	    Uniform initialization is retained as the mechanism control.
   138	    """
   139	    if n_heads != 4:
   140	        raise ValueError("the adaptive specialist screen uses exactly four heads")
   141	    specs = [
   142	        AttentionSpec(),
   143	        AttentionSpec(
   144	            kind="quad_route",
   145	            self_heads=0,
   146	            local_heads=2,
   147	            balanced_heads=1,
   148	            slope=0.8,
   149	            diagonal_debit=3.0,
   150	        ),
   151	        AttentionSpec(
   152	            kind="quad_route",
   153	            self_heads=0,
   154	            local_heads=3,
   155	            balanced_heads=0,
   156	            slope=0.8,
   157	            diagonal_debit=3.0,
   158	        ),
   159	        AttentionSpec(
   160	            kind="adaptive_specialists",
   161	            slope=0.8,
   162	            diagonal_debit=3.0,
   163	            free_heads=1,
   164	            specialist_init="lll",
   165	            init_strength=4.0,
   166	        ),
   167	        AttentionSpec(
   168	            kind="adaptive_specialists",
   169	            slope=0.8,
   170	            diagonal_debit=3.0,
   171	            free_heads=1,
   172	            specialist_init="llb",
   173	            init_strength=4.0,
   174	        ),
   175	        AttentionSpec(
   176	            kind="adaptive_specialists",
   177	            slope=0.8,
   178	            diagonal_debit=3.0,
   179	            free_heads=1,
   180	            specialist_init="uniform",
   181	            init_strength=0.0,
   182	        ),
   183	        AttentionSpec(
   184	            kind="adaptive_specialists",
   185	            slope=0.8,
   186	            diagonal_debit=3.0,
   187	            free_heads=1,
   188	            specialist_init="lll",
   189	            init_strength=2.0,
   190	        ),
   191	        AttentionSpec(
   192	            kind="adaptive_specialists",
   193	            slope=0.8,
   194	            diagonal_debit=3.0,
   195	            free_heads=1,
   196	            specialist_init="llb",
   197	            init_strength=2.0,
   198	        ),
   199	    ]
   200	    labels = [spec.label() for spec in specs]
   201	    if len(labels) != len(set(labels)):
   202	        raise RuntimeError("adaptive specialist screen contains duplicate labels")
   203	    if any(free_head_count(spec, n_heads) < 1 for spec in specs):
   204	        raise RuntimeError("every adaptive specialist candidate must preserve a whole free head")
   205	    return specs
   206	
   207	
   208	def crystallized_depth_programs(
   209	    n_heads: int = 4,
   210	    n_layers: int = 2,
   211	) -> list[tuple[AttentionSpec, ...]]:
   212	    """Free-head-preserving layer programs for CRSA placement tests.
   213	
   214	    ``R`` is the archive-leading three-local/one-free control. ``Q`` is the
   215	    canonical two-local/one-RAPS/one-free CRSA layer. The cross-order programs
   216	    determine whether balancing belongs below or above the maximally local
   217	    layer, while F/Q and F/R controls isolate specialization depth.
   218	    """
   219	    if n_heads != 4 or n_layers != 2:
   220	        raise ValueError("the preregistered depth screen uses two four-head layers")
   221	    free = AttentionSpec()
   222	    local = AttentionSpec(
   223	        kind="qu[... ELLIPSIZATION ...](kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1, slope=.8, diagonal_debit=3),
    29	        "R": S(kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0, slope=.8),
    30	        "C": S(kind="quad_route", self_heads=1, local_heads=1, balanced_heads=1, slope=.8, diagonal_debit=3),
    31	    }
    32	
    33	
    34	def programs() -> list[Program]:
    35	    s = specs()
    36	    codes = [
    37	        "FF", "LL", "DD", "QQ", "RR", "CC",
    38	        "LF", "DF", "QF", "RF", "GF", "BF",
    39	        "FL", "FD", "FQ", "FR", "FG", "FB",
    40	        "DQ", "QD", "RQ", "QR", "LQ", "QL",
    41	    ]
    42	    out = [(s[a], s[b]) for a, b in codes]
    43	    labels = [attention_program_label(p) for p in out]
    44	    if len(labels) != len(set(labels)):
    45	        raise RuntimeError("duplicate program labels")
    46	    return out
    47	
    48	
    49	def write_report(results: list[dict], out: Path) -> None:
    50	    if not results:
    51	        return
    52	    soft = next(r["val_bpb"] for r in results if r["operator"] == "softmax -> softmax")
    53	    ranked = sorted(results, key=lambda r: r["val_bpb"])
    54	    lines = [
    55	        "# Depth-Routed Causal Attention — Generation 8 Screen",
    56	        "",
    57	        f"Completed **{len(results)}/{len(programs())}** two-layer programs; seed **27183**; 240 steps. Test split remained unopened.",
    58	        "",
    59	        "The arrow is bottom layer → top layer. This isolates whether local/balanced routing belongs below or above unrestricted retrieval.",
    60	        "",
    61	        "|#|bottom → top|validation bpb|Δ vs F→F|diag|age|long|Gini|e-rank|tok/s|",
    62	        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    63	    ]
    64	    for rank, result in enumerate(ranked, 1):
    65	        a = result["attention"]
    66	        lines.append(
    67	            f"|{rank}|`{result['operator']}`|{result['val_bpb']:.6f}|{result['val_bpb']-soft:+.6f}|"
    68	            f"{a['diagonal_mass']:.3f}|{a['mean_age']:.2f}|{a['long_range_mass']:.3f}|"
    69	            f"{a['column_gini']:.3f}|{a['effective_rank_fraction']:.3f}|{result['tokens_per_second']:.0f}|"
    70	        )
    71	    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    72	
    73	
    74	def main() -> None:
    75	    ap = argparse.ArgumentParser()
    76	    ap.add_argument("--start", type=int, default=1)
    77	    ap.add_argument("--end", type=int)
    78	    args = ap.parse_args()
    79	    arms = programs()
    80	    end = args.end or len(arms)
    81	    out = ROOT / "results" / "depth_program_screen"
    82	    out.mkdir(parents=True, exist_ok=True)
    83	    path = out / "results.json"
    84	    results = json.loads(path.read_text()) if path.exists() else []
    85	    done = {r["operator"] for r in results}
    86	    corpus = ByteCorpus(ROOT / "corpus/train.bin", ROOT / "corpus/val.bin", ROOT / "corpus/test.bin")
    87	    model_cfg = ModelConfig(context=96, d_model=48, n_heads=4, n_layers=2)
    88	    train_cfg = TrainConfig(steps=240, batch_size=12, eval_batches=12, log_every=120, torch_threads=4)
    89	    for index in range(args.start - 1, min(end, len(arms))):
    90	        program = arms[index]
    91	        label = attention_program_label(program)
    92	        if label in done:
    93	            continue
    94	        print(f"=== {index+1}/{len(arms)} {label} ===", flush=True)
    95	        result = train_lm(corpus, model_cfg, program, train_cfg, seed=27183, verbose=True)
    96	        results.append(result)
    97	        path.write_text(json.dumps(results, indent=2))
    98	        write_report(results, out)
    99	    write_report(results, out)
   100	
   101	
   102	if __name__ == "__main__":
   103	    main()

--- atomic ---
     1	from __future__ import annotations
     2	
     3	import argparse
     4	import hashlib
     5	import json
     6	import sys
     7	from pathlib import Path
     8	from typing import Any
     9	
    10	ROOT = Path(__file__).resolve().parents[1]
    11	sys.path.insert(0, str(ROOT / "src"))
    12	
    13	from attention_moonshot.data import ByteCorpus
    14	from attention_moonshot.experiment import TrainConfig, attention_program_label, train_lm
    15	from attention_moonshot.model import ModelConfig
    16	from attention_moonshot.operators import AttentionSpec as S
    17	
    18	Program = tuple[S, S]
    19	
    20	
    21	def _specs() -> dict[str, S]:
    22	    return {
    23	        "F": S(),
    24	        "B": S(kind="raps", diagonal_debit=3.0),
    25	        "L": S(kind="recency", slope=1.25),
    26	        "D": S(kind="dual_route", local_heads=3, slope=1.25, diagonal_debit=3.0),
    27	        "G": S(kind="slg", self_heads=1, local_heads=2, slope=0.8, diagonal_debit=3.0),
    28	        "Q": S(kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1, slope=0.8, diagonal_debit=3.0),
    29	        "R": S(kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0, slope=0.8),
    30	        "C": S(kind="quad_route", self_heads=1, local_heads=1, balanced_heads=1, slope=0.8, diagonal_debit=3.0),
    31	    }
    32	
    33	
    34	def programs() -> list[Program]:
    35	    s = _specs()
    36	    codes = (
    37	        "FF", "LL", "DD", "QQ", "RR", "CC",
    38	        "LF", "DF", "QF", "RF", "GF", "BF",
    39	        "FL", "FD", "FQ", "FR", "FG", "FB",
    40	        "DQ", "QD", "RQ", "QR", "LQ", "QL",
    41	    )
    42	    result = [(s[code[0]], s[code[1]]) for code in codes]
    43	    labels = [attention_program_label(program) for program in result]
    44	    if len(labels) != len(set(labels)):
    45	        raise RuntimeError("depth program labels must be unique")
    46	    return result
    47	
    48	
    49	def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    50	    path.parent.mkdir(parents=True, exist_ok=True)
    51	    temporary = path.with_suffix(path.suffix + ".tmp")
    52	    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    53	    temporary.replace(path)
    54	
    55	
    56	def _run_id(seed: int, arm: int, label: str) -> str:
    57	    digest = hashlib.sha256(label.encode()).hexdigest()[:10]
    58	    return f"archive-seed{seed}-arm{arm:02d}-{digest}.json"
    59	
    60	
    61	def write_report(output: Path) -> None:
    62	    rows = [json.loads(path.read_text()) for path in sorted((output / "raw").glob("*.json"))]
    63	    if not rows:
    64	        return
    65	    soft = next((float(row["val_bpb"]) for row in rows if row["operator"] == "softmax -> softmax"), None)
    66	    ranked = sorted(rows, key=lambda row: float(row["val_bpb"]))
    67	    lines = [
    68	        "# Depth-Routed Causal Attention — Generation 11",
    69	        "",
    70	        f"Completed **{len(rows)}/{len(programs())}** atomic two-layer programs. Arrow order is bottom → top. Test split remained unopened.",
    71	        "",
    72	        "|#|bottom → top|validation bpb|Δ vs F→F|diag|age|long|Gini|e-rank|tokens/s|",
    73	        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    74	    ]
    75	    for rank, row in enumerate(ranked, 1):
    76	        attention = row["attention"]
    77	        delta = float(row["val_bpb"]) - soft if soft is not None else 0.0
    78	        lines.append(
    79	            f"|{rank}|`{row['operator']}`|{float(row['val_bpb']):.6f}|{delta:+.6f}|"
    80	            f"{float(attention['diagonal_mass']):.3f}|{float(attention['mean_age']):.2f}|"
    81	            f"{float(attention['long_range_mass']):.3f}|{float(attention['column_gini']):.3f}|"
    82	            f"{float(attention['effective_rank_fraction']):.3f}|{float(row['tokens_per_second']):.0f}|"
    83	        )
    84	    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    85	
    86	
    87	def main() -> None:
    88	    parser = argparse.ArgumentParser()
    89	    parser.add_argument("--arm", type=int, default=1)
    90	    parser.add_argument("--seed", type=int, default=27183)
    91	    parser.add_argument("--threads", type=int, default=1)
    92	    parser.add_argument("--steps", type=int, default=240)
    93	    parser.add_argument("--output", default="results/depth_program_atomic")
    94	    parser.add_argument("--force", action="store_true")
    95	    parser.add_argument("--list", action="store_true")
    96	    args = parser.parse_args()
    97	
    98	    arms = programs()
    99	    if args.list:
   100	        for index, program in enumerate(arms, 1):
   101	            print(f"{index:2d} {attention_program_label(program)}")
   102	        return
   103	    if not 1 <= args.arm <= len(arms):
   104	        raise SystemExit(f"arm must be in 1..{len(arms)}")
   105	
   106	    program = arms[args.arm - 1]
   107	    label = attention_program_label(program)
   108	    output = ROOT / args.output
   109	    path = output / "raw" / _run_id(args.seed, args.arm, label)
   110	    if path.exists() and not args.force:
   111	        print(f"SKIP {path.name}")
   112	        write_report(output)
   113	        return
   114	
   115	    corpus = ByteCorpus(ROOT / "corpus/train.bin", ROOT / "corpus/val.bin", ROOT / "corpus/test.bin")
   116	    result = train_lm(
   117	        corpus,
   118	        ModelConfig(context=96, d_model=48, n_heads=4, n_layers=2),
   119	        program,
   120	        TrainConfig(
   121	            steps=args.steps,
   122	            batch_size=12,
   123	            eval_batches=12,
   124	            log_every=max(1, args.steps // 2),
   125	            torch_threads=args.threads,
   126	        ),
   127	        seed=args.seed,
   128	        verbose=True,
   129	    )
   130	    result["arm"] = args.arm
   131	    result["domain"] = "archive"
   132	    _atomic_json(path, result)
   133	    write_report(output)
   134	    print(f"WROTE {path}")
   135	
   136	
   137	if __name__ == "__main__":
   138	    main()
```

Continuing the experiments

I’m extending the integrated NoPE suite now, starting with ambiguity checks and adversarial controls before interpreting gains. I’m also comparing  the implementation against the documented routing and crystallization  design, then I’ll report concrete results and integration judgments.

Extending the validation

I’m treating the current suite as a promising integration with one  important generator flaw to resolve first: duplicate distractor keys can make labels ambiguous. I’ll compare the original and hardened versions  before drawing conclusions.



Inspecting Repository Files and Status

```
bash -lc pwd && git status --short --branch && find . -maxdepth 2 -type f | sort | sed -n '1,220p'
/mnt/data/attention-moonshot-nope-v1
## moonshot/nope-suite-v1
 M results/role_complete_screen/REPORT.md
 M results/role_complete_screen/results.json
 M src/attention_moonshot/candidates.py
 M src/attention_moonshot/experiment.py
 M src/attention_moonshot/model.py
 M src/attention_moonshot/operators.py
 M src/attention_moonshot/streaming.py
 M src/attention_moonshot/synthetic.py
 M tests/test_candidates.py
 M tests/test_model_training.py
?? results/PARALLEL_MERGE_2026-08-20.md
?? results/adaptive_archive_arm1.log
?? results/adaptive_archive_arm2.log
?? results/adaptive_archive_arm3.log
?? results/adaptive_archive_arm4.log
?? results/adaptive_archive_arm5.log
?? results/adaptive_depth_screen/
?? results/adaptive_route_screen/
?? results/adaptive_specialist_stage1/
?? results/adaptive_stage1/
?? results/anchor_foundation_stage/
?? results/anchor_residual_stage1/
?? results/architecture_scale/
?? results/depth_confirmation/
?? results/depth_program_stage/
?? results/generation14_stage1/
?? results/geometric_q_variable_lag/
?? results/length_extrapolation/
?? results/long_horizon_confirmation/
?? results/long_horizon_foundation/
?? results/marginal_residual_stage/
?? results/marginal_variable_lag/
?? results/q_residual_variable_lag/
?? results/q_spectrum_variable_lag/
?? results/recall_calibration/
?? results/role_complete_screen/run.log
?? results/role_complete_screen/run.pid
?? results/role_synthetic/
?? results/route_confirmation/
?? results/route_screen/
?? results/selected_confirmation/
?? results/specialist_mqar_stage/
?? results/specialist_stage1/
?? results/variable_lag_screen/
?? scripts/run_adaptive_depth_screen.py
?? scripts/run_adaptive_screen.py
?? scripts/run_anchor_confirmation.py
?? scripts/run_anchor_foundation_screen.py
?? scripts/run_anchor_residual_screen.py
?? scripts/run_architecture_scale.py
?? scripts/run_depth_confirmation.py
?? scripts/run_depth_program_atomic.py
?? scripts/run_depth_program_screen.py
?? scripts/run_geometric_q_variable_lag.py
?? scripts/run_headsplit_screen.py
?? scripts/run_length_extrapolation.py
?? scripts/run_long_horizon_confirmation.py
?? scripts/run_marginal_residual_screen.py
?? scripts/run_marginal_variable_lag.py
?? scripts/run_q_residual_variable_lag.py
?? scripts/run_q_spectrum_variable_lag.py
?? scripts/run_role_synthetic.py
?? scripts/run_variable_lag_screen.py
?? tests/test_anchor_confirmation_script.py
?? tests/test_geometric_prefix.py
?? tests/test_geometric_q_script.py
?? tests/test_marginal_residual.py
?? tests/test_marginal_screen_script.py
?? tests/test_marginal_variable_lag_script.py
?? tests/test_q_residual.py
?? tests/test_q_residual_script.py
?? tests/test_q_spectrum_script.py
./.git
./.gitignore
./.pytest_cache/.gitignore
./.pytest_cache/CACHEDIR.TAG
./.pytest_cache/README.md
./corpus/manifest.json
./corpus/test.bin
./corpus/train.bin
./corpus/val.bin
./pyproject.toml
./results/PARALLEL_MERGE_2026-08-20.md
./results/adaptive_archive_arm1.log
./results/adaptive_archive_arm2.log
./results/adaptive_archive_arm3.log
./results/adaptive_archive_arm4.log
./results/adaptive_archive_arm5.log
./scripts/run_adaptive_depth_screen.py
./scripts/run_adaptive_screen.py
./scripts/run_anchor_confirmation.py
./scripts/run_anchor_foundation_screen.py
./scripts/run_anchor_residual_screen.py
./scripts/run_architecture_scale.py
./scripts/run_archive_screen.py
./scripts/run_confirmation.py
./scripts/run_depth_confirmation.py
./scripts/run_depth_program_atomic.py
./scripts/run_depth_program_screen.py
./scripts/run_geometric_q_variable_lag.py
./scripts/run_headsplit_screen.py
./scripts/run_length_extrapolation.py
./scripts/run_long_horizon_confirmation.py
./scripts/run_marginal_residual_screen.py
./scripts/run_marginal_variable_lag.py
./scripts/run_q_residual_variable_lag.py
./scripts/run_q_spectrum_variable_lag.py
./scripts/run_role_complete_screen.py
./scripts/run_role_synthetic.py
./scripts/run_route_confirmation.py
./scripts/run_route_screen.py
./scripts/run_specialist_screen.py
./scripts/run_synthetic_suite.py
./scripts/run_theory_audit.py
./scripts/run_variable_lag_screen.py
./tests/test_adaptive_route.py
./tests/test_anchor_confirmation_script.py
./tests/test_anchor_residual.py
./tests/test_candidates.py
./tests/test_geometric_prefix.py
./tests/test_geometric_q_script.py
./tests/test_marginal_residual.py
./tests/test_marginal_screen_script.py
./tests/test_marginal_variable_lag_script.py
./tests/test_model_training.py
./tests/test_operators.py
./tests/test_q_residual.py
./tests/test_q_residual_script.py
./tests/test_q_spectrum_script.py
./tests/test_route_screen_reporting.py
```



Analyzed

```
bash -lc sed -n '1,260p' src/attention_moonshot/synthetic.py && printf '\n--- model ---\n' && sed -n '1,320p' src/attention_moonshot/model.py && printf '\n--- candidates ---\n' && sed -n '1,360p' src/attention_moonshot/candidates.py
from __future__ import annotations
from dataclasses import asdict,dataclass
import math,time
from typing import Any
import torch
from torch.nn.utils import clip_grad_norm_
from .experiment import AttentionProgram,adaptive_route_metrics,attention_program_label,attention_program_payload,seed_all
from .model import ByteGPT,ModelConfig
from .operators import AttentionSpec

@dataclass(frozen=True,slots=True)
class SyntheticConfig:
    task:str='copy';steps:int=300;batch_size:int=32;learning_rate:float=1e-3;weight_decay:float=0.0;grad_clip:float=1.0;eval_batches:int=20;lag:int=64;pairs:int=16;queries:int=4;torch_threads:int=4

def delayed_copy_batch(*,batch_size:int,context:int,vocab_size:int,lag:int,generator:torch.Generator):
    if lag>=context:raise ValueError('lag must be smaller than context')
    seed=torch.randint(1,vocab_size,(batch_size,lag),generator=generator);seq=seed.repeat(1,(context+lag-1)//lag)[:,:context]
    x=seq[:,:-1];y=seq[:,1:];m=torch.zeros_like(y,dtype=torch.float32);m[:,lag-1:]=1;return x,y,m

def random_lag_copy_batch(
    *,
    batch_size: int,
    context: int,
    vocab_size: int,
    min_lag: int,
    max_lag: int,
    generator: torch.Generator,
):
    """Sample one deterministic lag for a whole batch and build periodic copy data."""
    if min_lag < 1:
        raise ValueError("min_lag must be positive")
    if min_lag > max_lag:
        raise ValueError("min_lag must not exceed max_lag")
    if max_lag >= context:
        raise ValueError("max_lag must be smaller than context")
    lag = int(torch.randint(min_lag, max_lag + 1, (1,), generator=generator).item())
    x, y, mask = delayed_copy_batch(
        batch_size=batch_size,
        context=context,
        vocab_size=vocab_size,
        lag=lag,
        generator=generator,
    )
    return x, y, mask, lag


def associative_recall_batch(*,batch_size:int,context:int,pairs:int,generator:torch.Generator):
    if 2*pairs+2>context or pairs>31:raise ValueError('invalid pairs/context')
    keys=torch.stack([torch.randperm(31,generator=generator)[:pairs]+1 for _ in range(batch_size)])
    vals=torch.randint(32,64,(batch_size,pairs),generator=generator);qi=torch.randint(0,pairs,(batch_size,),generator=generator)
    seq=torch.zeros(batch_size,context,dtype=torch.long);seq[:,:2*pairs:2]=keys;seq[:,1:2*pairs:2]=vals
    seq[:,2*pairs:-1]=torch.randint(1,64,(batch_size,context-2*pairs-1),generator=generator)
    rows=torch.arange(batch_size);seq[:,-1]=keys[rows,qi];target=vals[rows,qi]
    y=torch.zeros_like(seq);y[:,-1]=target;m=torch.zeros_like(seq,dtype=torch.float32);m[:,-1]=1;return seq,y,m

def multi_query_recall_batch(*,batch_size:int,context:int,pairs:int,queries:int,generator:torch.Generator):
    if pairs < 1 or pairs > 31:
        raise ValueError("pairs must be in 1..31")
    if queries < 1 or queries > pairs:
        raise ValueError("queries must be in 1..pairs")
    if 2*pairs+queries>context:
        raise ValueError("context too short for pair table and queries")
    keys=torch.stack([torch.randperm(31,generator=generator)[:pairs]+1 for _ in range(batch_size)])
    vals=torch.randint(32,64,(batch_size,pairs),generator=generator)
    query_indices=torch.stack([torch.randperm(pairs,generator=generator)[:queries] for _ in range(batch_size)])
    rows=torch.arange(batch_size).unsqueeze(1)
    query_keys=keys[rows,query_indices]
    targets=vals[rows,query_indices]
    x=torch.zeros(batch_size,context,dtype=torch.long)
    x[:,:2*pairs:2]=keys
    x[:,1:2*pairs:2]=vals
    x[:,-queries:]=query_keys
    y=torch.zeros_like(x)
    y[:,-queries:]=targets
    m=torch.zeros_like(x,dtype=torch.float32)
    m[:,-queries:]=1
    return x,y,m


def _batch(mc:ModelConfig,cfg:SyntheticConfig,g:torch.Generator):
    if cfg.task=='copy':return delayed_copy_batch(batch_size=cfg.batch_size,context=mc.context,vocab_size=mc.vocab_size,lag=cfg.lag,generator=g)
    if cfg.task=='recall':return associative_recall_batch(batch_size=cfg.batch_size,context=mc.context,pairs=cfg.pairs,generator=g)
    if cfg.task=='mqar':return multi_query_recall_batch(batch_size=cfg.batch_size,context=mc.context,pairs=cfg.pairs,queries=cfg.queries,generator=g)
    raise ValueError(cfg.task)

@torch.no_grad()
def _eval(model:ByteGPT,cfg:SyntheticConfig,seed:int):
    model.eval();g=torch.Generator().manual_seed(seed);loss=correct=count=0.0
    for _ in range(cfg.eval_batches):
        x,y,m=_batch(model.cfg,cfg,g);logits=model(x);per=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='none').view_as(y);loss+=(per*m).sum().item();correct+=((logits.argmax(-1)==y)&m.bool()).sum().item();count+=m.sum().item()
    model.train();return loss/count/math.log(2),correct/count

def train_synthetic(mc:ModelConfig,spec:AttentionProgram,cfg:SyntheticConfig,*,seed:int,verbose:bool=False)->dict[str,Any]:
    seed_all(seed,cfg.torch_threads);model=ByteGPT(mc,spec);opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay);g=torch.Generator().manual_seed(seed+171);start=time.perf_counter();tokens=0
    for step in range(1,cfg.steps+1):
        x,y,m=_batch(mc,cfg,g);opt.zero_grad(set_to_none=True);loss=model.loss(x,y,m);loss.backward();clip_grad_norm_(model.parameters(),cfg.grad_clip);opt.step();tokens+=x.numel()
        if verbose and (step==1 or step==cfg.steps or step%max(1,cfg.steps//4)==0):print(cfg.task,attention_program_label(spec),seed,step,loss.item()/math.log(2),flush=True)
    bits,acc=_eval(model,cfg,seed+913)
    probe_generator=torch.Generator().manual_seed(seed+1913)
    probe_x,_,_=_batch(mc,cfg,probe_generator)
    with torch.no_grad():model(probe_x)
    return {'task':cfg.task,'operator':attention_program_label(spec),'spec':attention_program_payload(spec),'seed':seed,'eval_bits':bits,'eval_accuracy':acc,'tokens_per_second':tokens/max(time.perf_counter()-start,1e-9),'route':adaptive_route_metrics(model,0),'model_config':asdict(mc),'train_config':asdict(cfg)}


@torch.no_grad()
def evaluate_copy_lag(
    model: ByteGPT,
    cfg: SyntheticConfig,
    *,
    lag: int,
    seed: int,
    context: int | None = None,
) -> tuple[float, float]:
    """Evaluate a trained model on one exact delayed-copy lag."""
    evaluation_context = model.cfg.context if context is None else int(context)
    if lag >= evaluation_context:
        raise ValueError("lag must be smaller than evaluation context")
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    loss = correct = count = 0.0
    for _ in range(cfg.eval_batches):
        x, y, mask = delayed_copy_batch(
            batch_size=cfg.batch_size,
            context=evaluation_context,
            vocab_size=model.cfg.vocab_size,
            lag=lag,
            generator=generator,
        )
        logits = model(x)
        per = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none"
        ).view_as(y)
        loss += (per * mask).sum().item()
        correct += ((logits.argmax(-1) == y) & mask.bool()).sum().item()
        count += mask.sum().item()
    model.train()
    return loss / count / math.log(2), correct / count


def train_variable_lag_copy(
    mc: ModelConfig,
    spec: AttentionProgram,
    cfg: SyntheticConfig,
    *,
    train_lag_min: int,
    train_lag_max: int,
    eval_lags: tuple[int, ...],
    seed: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train across a lag interval, then measure every requested fixed lag."""
    if not eval_lags:
        raise ValueError("eval_lags must not be empty")
    if any(lag < 1 or lag >= mc.context for lag in eval_lags):
        raise ValueError("every eval lag must be in 1..context-1")
    seed_all(seed, cfg.torch_threads)
    model = ByteGPT(mc, spec)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 171)
    start = time.perf_counter()
    tokens = 0
    lag_counts = {lag: 0 for lag in range(train_lag_min, train_lag_max + 1)}
    model.train()
    for step in range(1, cfg.steps + 1):
        x, y, mask, lag = random_lag_copy_batch(
            batch_size=cfg.batch_size,
            context=mc.context,
            vocab_size=mc.vocab_size,
            min_lag=train_lag_min,
            max_lag=train_lag_max,
            generator=generator,
        )
        lag_counts[lag] += 1
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x, y, mask)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        tokens += x.numel()
        if verbose and (step == 1 or step == cfg.steps or step % max(1, cfg.steps // 4) == 0):
            print(
                "copy_variable", attention_program_label(spec), seed, step, lag,
                loss.item() / math.log(2), flush=True,
            )
    sweep = []
    for offset, lag in enumerate(sorted(set(eval_lags))):
        bits, accuracy = evaluate_copy_lag(model, cfg, lag=lag, seed=seed + 913 + offset)
        sweep.append({"lag": lag, "eval_bits": bits, "eval_accuracy": accuracy})
    probe_generator = torch.Generator().manual_seed(seed + 1913)
    probe_x, _, _, _ = random_lag_copy_batch(
        batch_size=cfg.batch_size, context=mc.context, vocab_size=mc.vocab_size,
        min_lag=train_lag_min, max_lag=train_lag_max, generator=probe_generator,
    )
    with torch.no_grad():
        model(probe_x)
    return {
        "task": "copy_variable",
        "operator": attention_program_label(spec),
        "spec": attention_program_payload(spec),
        "seed": seed,
        "train_lag_range": [train_lag_min, train_lag_max],
        "train_lag_counts": lag_counts,
        "lag_sweep": sweep,
        "tokens_per_second": tokens / max(time.perf_counter() - start, 1e-9),
        "route": adaptive_route_metrics(model, 0),
        "model_config": asdict(mc),
        "train_confi[... ELLIPSIZATION ...]ation uses exactly four heads")
    specs = [
        AttentionSpec(),
        AttentionSpec(kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0, slope=0.8, diagonal_debit=3.0),
        AttentionSpec(kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1, slope=0.8, diagonal_debit=3.0),
        AttentionSpec(kind="quad_route", self_heads=0, local_heads=1, balanced_heads=1, slope=0.8, diagonal_debit=3.0),
    ]
    labels = [spec.label() for spec in specs]
    if len(labels) != len(set(labels)):
        raise RuntimeError("confirmation set contains duplicate labels")
    if any(free_head_count(spec, n_heads) < 1 for spec in specs):
        raise RuntimeError("every confirmation candidate must preserve a free head")
    return specs



def adaptive_route_candidates(n_heads: int = 4) -> list[AttentionSpec]:
    """Generation-9 floor-gated adaptive routes and fixed-route controls."""
    if n_heads != 4:
        raise ValueError("the adaptive route screen uses exactly four heads")
    specs = [
        AttentionSpec(),
        AttentionSpec(
            kind="quad_route",
            self_heads=0,
            local_heads=2,
            balanced_heads=1,
            slope=0.8,
            diagonal_debit=3.0,
        ),
        AttentionSpec(
            kind="quad_route",
            self_heads=0,
            local_heads=3,
            balanced_heads=0,
            slope=0.8,
        ),
        AttentionSpec(kind="adaptive_route", slope=0.8, diagonal_debit=3.0, free_floor=0.125),
        AttentionSpec(kind="adaptive_route", slope=0.8, diagonal_debit=3.0, free_floor=0.25),
        AttentionSpec(kind="adaptive_route", slope=0.8, diagonal_debit=3.0, free_floor=0.5),
        AttentionSpec(kind="adaptive_route", slope=1.25, diagonal_debit=3.0, free_floor=0.25),
        AttentionSpec(kind="adaptive_route", slope=0.8, diagonal_debit=0.0, free_floor=0.25),
    ]
    labels = [spec.label() for spec in specs]
    if len(labels) != len(set(labels)):
        raise RuntimeError("adaptive route screen contains duplicate labels")
    return specs


def adaptive_specialist_candidates(n_heads: int = 4) -> list[AttentionSpec]:
    """Generation-10 adaptive specialists with one untouched free head.

    The adaptive heads are initialized at the two fixed-route winners: three
    local specialists (LLL) or two local plus one balanced specialist (LLB).
    Uniform initialization is retained as the mechanism control.
    """
    if n_heads != 4:
        raise ValueError("the adaptive specialist screen uses exactly four heads")
    specs = [
        AttentionSpec(),
        AttentionSpec(
            kind="quad_route",
            self_heads=0,
            local_heads=2,
            balanced_heads=1,
            slope=0.8,
            diagonal_debit=3.0,
        ),
        AttentionSpec(
            kind="quad_route",
            self_heads=0,
            local_heads=3,
            balanced_heads=0,
            slope=0.8,
            diagonal_debit=3.0,
        ),
        AttentionSpec(
            kind="adaptive_specialists",
            slope=0.8,
            diagonal_debit=3.0,
            free_heads=1,
            specialist_init="lll",
            init_strength=4.0,
        ),
        AttentionSpec(
            kind="adaptive_specialists",
            slope=0.8,
            diagonal_debit=3.0,
            free_heads=1,
            specialist_init="llb",
            init_strength=4.0,
        ),
        AttentionSpec(
            kind="adaptive_specialists",
            slope=0.8,
            diagonal_debit=3.0,
            free_heads=1,
            specialist_init="uniform",
            init_strength=0.0,
        ),
        AttentionSpec(
            kind="adaptive_specialists",
            slope=0.8,
            diagonal_debit=3.0,
            free_heads=1,
            specialist_init="lll",
            init_strength=2.0,
        ),
        AttentionSpec(
            kind="adaptive_specialists",
            slope=0.8,
            diagonal_debit=3.0,
            free_heads=1,
            specialist_init="llb",
            init_strength=2.0,
        ),
    ]
    labels = [spec.label() for spec in specs]
    if len(labels) != len(set(labels)):
        raise RuntimeError("adaptive specialist screen contains duplicate labels")
    if any(free_head_count(spec, n_heads) < 1 for spec in specs):
        raise RuntimeError("every adaptive specialist candidate must preserve a whole free head")
    return specs


def crystallized_depth_programs(
    n_heads: int = 4,
    n_layers: int = 2,
) -> list[tuple[AttentionSpec, ...]]:
    """Free-head-preserving layer programs for CRSA placement tests.

    ``R`` is the archive-leading three-local/one-free control. ``Q`` is the
    canonical two-local/one-RAPS/one-free CRSA layer. The cross-order programs
    determine whether balancing belongs below or above the maximally local
    layer, while F/Q and F/R controls isolate specialization depth.
    """
    if n_heads != 4 or n_layers != 2:
        raise ValueError("the preregistered depth screen uses two four-head layers")
    free = AttentionSpec()
    local = AttentionSpec(
        kind="quad_route",
        self_heads=0,
        local_heads=3,
        balanced_heads=0,
        slope=0.8,
    )
    crsa = AttentionSpec(
        kind="quad_route",
        self_heads=0,
        local_heads=2,
        balanced_heads=1,
        slope=0.8,
        diagonal_debit=3.0,
    )
    programs = [
        (free, free),
        (crsa, crsa),
        (local, local),
        (local, crsa),
        (crsa, local),
        (crsa, free),
        (free, crsa),
        (local, free),
        (free, local),
    ]
    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
    if len(labels) != len(set(labels)):
        raise RuntimeError("depth screen contains duplicate programs")
    if any(free_head_count(spec, n_heads) < 1 for program in programs for spec in program):
        raise RuntimeError("every depth-program layer must preserve a whole free head")
    return programs

def anchor_residual_candidates(n_heads: int = 4) -> list[AttentionSpec]:
    """Generation-11 anchor-residual specialists with exact free heads."""
    if n_heads != 4:
        raise ValueError("the anchor-residual screen uses exactly four heads")
    specs = [
        AttentionSpec(),
        AttentionSpec(
            kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1,
            slope=0.8, diagonal_debit=3.0,
        ),
        AttentionSpec(
            kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0,
            slope=0.8, diagonal_debit=3.0,
        ),
    ]
    for pattern in ("lll", "llb"):
        for budget in (0.125, 0.25, 0.5):
            specs.append(AttentionSpec(
                kind="anchor_residual",
                slope=0.8,
                diagonal_debit=3.0,
                free_heads=1,
                anchor_pattern=pattern,
                adapt_budget=budget,
            ))
    labels = [spec.label() for spec in specs]
    if len(labels) != len(set(labels)):
        raise RuntimeError("anchor-residual screen contains duplicate labels")
    if any(free_head_count(spec, n_heads) < 1 for spec in specs):
        raise RuntimeError("every anchor-residual candidate must preserve a whole free head")
    return specs




def marginal_residual_candidates(n_heads: int = 4) -> list[AttentionSpec]:
    """Generation-13 bounded local-to-Prefix residual candidates.

    Every learned candidate keeps three local specialist heads, permits only a
    bounded query-dependent diversion into Prefix-Sinkhorn balancing, and keeps
    one whole causal-softmax head untouched.
    """
    if n_heads != 4:
        raise ValueError("the marginal-residual screen uses exactly four heads")
    specs = [
        AttentionSpec(),
        AttentionSpec(
            kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1,
            slope=0.8, diagonal_debit=3.0,
        ),
        AttentionSpec(
            kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0,
            slope=0.8,
        ),
        AttentionSpec(
            kind="anchor_residual", slope=0.8, diagonal_debit=3.0,
            free_heads=1, anchor_pattern="lll", adapt_budget=0.5,
        ),
    ]
    specs.extend(
        AttentionSpec(
            kind="marginal_residual", slope=0.8, diagonal_debit=3.0,
            free_heads=1, adapt_budget=budget,
        )
        for budget in (0.125, 0.25, 0.5, 0.75)
    )
    labels = [spec.label() for spec in specs]
    if len(labels) != len(set(labels)):
        raise RuntimeError("marginal-residual screen contains duplicate labels")
    if any(free_head_count(spec, n_heads) < 1 for spec in specs):
        raise RuntimeError("every marginal-residual candidate must preserve a whole free head")
    return specs




def geometric_q_programs(
    n_heads: int = 4,
    n_layers: int = 2,
) -> list[tuple[AttentionSpec, ...]]:
    """Q foundations with bounded geometric memory in the Prefix head."""
    if n_heads != 4 or n_layers != 2:
        raise ValueError("the geometric-Q screen uses two four-head layers")
    free = AttentionSpec()
    programs = [
        (
            AttentionSpec(
                kind="quad_route",
                usage_decay=decay,
                diagonal_debit=3.0,
                self_heads=0,
                local_heads=2,
                balanced_heads=1,
                slope=0.8,
            ),
            free,
        )
        for decay in (0.9, 0.95, 0.98, 0.99, 0.995, 1.0)
    ]
    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
    if len(labels) != len(set(labels)):
        raise RuntimeError("geometric-Q program labels must be unique")
    if any(free_head_count(program[0], n_heads) != 1 for program in programs):
        raise RuntimeError("every geometric-Q candidate must preserve one whole free head")
    return programs
```



Analyzed

```
bash -lc grep -R "position_mode\|class ByteGPT\|class ModelConfig\|def forward" -n src/attention_moonshot/model.py src/attention_moonshot/operators.py src/attention_moonshot/experiment.py | head -80 && sed -n '1,240p' src/attention_moonshot/model.py && sed -n '240,520p' src/attention_moonshot/model.py
src/attention_moonshot/model.py:12:class ModelConfig:
src/attention_moonshot/model.py:20:    position_mode: str = "learned"
src/attention_moonshot/model.py:27:        if self.position_mode not in {"learned", "none"}:
src/attention_moonshot/model.py:28:            raise ValueError("position_mode must be 'learned' or 'none'")
src/attention_moonshot/model.py:100:    def forward(self, x: Tensor) -> Tensor:
src/attention_moonshot/model.py:177:    def forward(self,x:Tensor)->Tensor:
src/attention_moonshot/model.py:182:class ByteGPT(nn.Module):
src/attention_moonshot/model.py:188:        self.position=nn.Embedding(cfg.context,cfg.d_model) if cfg.position_mode == "learned" else None
src/attention_moonshot/model.py:199:    def forward(self,tokens:Tensor)->Tensor:
src/attention_moonshot/model.py:201:        if t>self.cfg.context and self.cfg.position_mode == "learned":
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .operators import AttentionSpec, adaptive_route_attention, adaptive_specialist_attention, anchor_residual_attention, marginal_residual_attention, q_residual_attention, apply_attention


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 256
    context: int = 128
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.0
    position_mode: str = "learned"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.context < 2:
            raise ValueError("context must be at least 2")
        if self.position_mode not in {"learned", "none"}:
            raise ValueError("position_mode must be 'learned' or 'none'")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, spec: AttentionSpec) -> None:
        super().__init__()
        self.spec = spec
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        if spec.kind == "adaptive_route":
            self.route_gate_weight = nn.Parameter(torch.zeros(cfg.n_heads, self.head_dim, 4))
            initial_bias = torch.tensor((0.0, 0.0, 0.0, -4.0)).repeat(cfg.n_heads, 1)
            self.route_gate_bias = nn.Parameter(initial_bias)
        elif spec.kind == "adaptive_specialists":
            if not 1 <= spec.free_heads < cfg.n_heads:
                raise ValueError("free_heads must be in 1..n_heads-1")
            specialist_heads = cfg.n_heads - spec.free_heads
            self.route_gate_weight = nn.Parameter(torch.zeros(specialist_heads, self.head_dim, 3))
            if spec.init_strength < 0:
                raise ValueError("init_strength must be nonnegative")
            if spec.specialist_init == "uniform":
                initial_bias = torch.zeros(specialist_heads, 3)
            elif spec.specialist_init in {"llb", "lll"}:
                strength = float(spec.init_strength)
                initial_bias = torch.full((specialist_heads, 3), -strength)
                for head in range(specialist_heads):
                    role = 1
                    if spec.specialist_init == "llb" and head == specialist_heads - 1:
                        role = 2
                    initial_bias[head, role] = strength
            else:
                raise ValueError("specialist_init must be 'llb', 'lll', or 'uniform'")
            self.route_gate_bias = nn.Parameter(initial_bias)
        elif spec.kind == "anchor_residual":
            if not 1 <= spec.free_heads < cfg.n_heads:
                raise ValueError("free_heads must be in 1..n_heads-1")
            if not 0.0 <= spec.adapt_budget <= 1.0:
                raise ValueError("adapt_budget must be in [0, 1]")
            if spec.anchor_pattern not in {"lll", "llb"}:
                raise ValueError("anchor_pattern must be 'lll' or 'llb'")
            specialist_heads = cfg.n_heads - spec.free_heads
            self.route_gate_weight = nn.Parameter(torch.zeros(specialist_heads, self.head_dim, 3))
            self.route_gate_bias = nn.Parameter(torch.zeros(specialist_heads, 3))
        elif spec.kind == "marginal_residual":
            if not 1 <= spec.free_heads < cfg.n_heads:
                raise ValueError("free_heads must be in 1..n_heads-1")
            if not 0.0 <= spec.adapt_budget <= 1.0:
                raise ValueError("adapt_budget must be in [0, 1]")
            specialist_heads = cfg.n_heads - spec.free_heads
            self.route_gate_weight = nn.Parameter(torch.zeros(specialist_heads, self.head_dim))
            self.route_gate_bias = nn.Parameter(torch.zeros(specialist_heads))
        elif spec.kind == "q_residual":
            if not 1 <= spec.free_heads < cfg.n_heads:
                raise ValueError("free_heads must be in 1..n_heads-1")
            if not 0.0 <= spec.adapt_budget <= 1.0:
                raise ValueError("adapt_budget must be in [0, 1]")
            if spec.init_strength < 0:
                raise ValueError("init_strength must be nonnegative")
            specialist_heads = cfg.n_heads - spec.free_heads
            self.route_gate_weight = nn.Parameter(torch.zeros(specialist_heads, self.head_dim))
            initial_bias = torch.full((specialist_heads,), -float(spec.init_strength))
            initial_bias[-1] = float(spec.init_strength)
            self.route_gate_bias = nn.Parameter(initial_bias)
        else:
            self.register_parameter("route_gate_weight", None)
            self.register_parameter("route_gate_bias", None)
        self.last_weights: Tensor | None = None
        self.last_route_probs: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        b,t,c = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        def heads(z: Tensor) -> Tensor:
            return z.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        q,k,v = heads(q),heads(k),heads(v)
        logits = q @ k.transpose(-2,-1) / math.sqrt(self.head_dim)
        if self.spec.kind == "adaptive_route":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("adaptive route gate parameters are unavailable")
            gate_logits = torch.einsum("bhtd,hdr->bhtr", q, self.route_gate_weight)
            gate_logits = gate_logits + self.route_gate_bias.view(1, self.n_heads, 1, 4)
            route_probs = torch.softmax(gate_logits, dim=-1)
            weights = adaptive_route_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        elif self.spec.kind == "adaptive_specialists":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("adaptive specialist gate parameters are unavailable")
            specialist_heads = self.n_heads - self.spec.free_heads
            gate_logits = torch.einsum(
                "bhtd,hdr->bhtr", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1, 3)
            route_probs = torch.softmax(gate_logits, dim=-1)
            weights = adaptive_specialist_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        elif self.spec.kind == "anchor_residual":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("anchor-residual gate parameters are unavailable")
            specialist_heads = self.n_heads - self.spec.free_heads
            gate_logits = torch.einsum(
                "bhtd,hdr->bhtr", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1, 3)
            route_probs = torch.softmax(gate_logits, dim=-1)
            weights = anchor_residual_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        elif self.spec.kind == "marginal_residual":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("marginal-residual gate parameters are unavailable")
            specialist_heads = self.n_heads - self.spec.free_heads
            gate_logits = torch.einsum(
                "bhtd,hd->bht", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1)
            route_probs = torch.sigmoid(gate_logits)
            weights = marginal_residual_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        elif self.spec.kind == "q_residual":
            if self.route_gate_weight is None or self.route_gate_bias is None:
                raise RuntimeError("Q-residual gate parameters are unavailable")
            specialist_heads = self.n_heads - self.spec.free_heads
            gate_logits = torch.einsum(
                "bhtd,hd->bht", q[:, :specialist_heads], self.route_gate_weight
            )
            gate_logits = gate_logits + self.route_gate_bias.view(1, specialist_heads, 1)
            route_probs = torch.sigmoid(gate_logits)
            weights = q_residual_attention(logits, route_probs, self.spec)
            self.last_route_probs = route_probs.detach()
        else:
            weights = apply_attention(logits, self.spec)
            self.last_route_probs = None
        self.last_weights = weights.detach()
        y = weights @ v
        y = y.transpose(1,2).contiguous().view(b,t,c)
        return self.dropout(self.out(y))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, spec: AttentionSpec) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg,spec)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.ff_mult * cfg.d_model
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model,hidden),nn.GELU(),nn.Linear(hidden,cfg.d_model),nn.Dropout(cfg.dropout))

    def forward(self,x:Tensor)->Tensor:
        x=x+self.attn(self.ln1(x))
        return x+self.mlp(self.ln2(x))


class ByteGPT(nn.Module):
    def __init__(self,cfg:ModelConfig,spec:AttentionSpec|Sequence[AttentionSpec])->None:
        super().__init__();self.cfg=cfg
        specs=(spec,)*cfg.n_layers if isinstance(spec,AttentionSpec) else tuple(spec)
        if len(specs)!=cfg.n_layers:raise ValueError("one attention spec per layer required")
        self.token=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.position=nn.Embedding(cfg.context,cfg.d_model) if cfg.position_mode == "learned" else None
        self.blocks=nn.ModuleList(Block(cfg,s) for s in specs)
        self.final_ln=nn.LayerNorm(cfg.d_model)
        self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module:nn.Module)->None:
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,tokens:Tensor)->Tensor:
        b,t=tokens.shape
        if t>self.cfg.context and self.cfg.position_mode == "learned":
            raise ValueError("sequence exceeds configured context")
        x=self.token(tokens)
        if self.position is not None:
            pos=torch.arange(t,device=tokens.device)
            x=x+self.position(pos)
        for block in self.blocks:x=block(x)
        return self.lm_head(self.final_ln(x))

    def loss(self,x:Tensor,y:Tensor,loss_mask:Tensor|None=None)->Tensor:
        logits=self(x)
        per=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='none').view_as(y)
        if loss_mask is None:return per.mean()
        return (per*loss_mask).sum()/loss_mask.sum().clamp_min(1)

    def attention_weights(self,x:Tensor,layer:int=0)->Tensor:
        self(x)
        w=self.blocks[layer].attn.last_weights
        if w is None:raise RuntimeError("attention weights unavailable")
        return w
```



Analyzed

```
bash -lc cat pyproject.toml && sed -n '1,340p' src/attention_moonshot/experiment.py && sed -n '1,300p' scripts/run_depth_program_atomic.py && sed -n '1,260p' scripts/run_variable_lag_screen.py
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "attention-moonshot"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["torch", "numpy"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import asdict,dataclass
import math,random,time
from typing import Any
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from .data import ByteCorpus
from .metrics import attention_metrics
from .model import ByteGPT,ModelConfig
from .operators import AttentionSpec

AttentionProgram = AttentionSpec | Sequence[AttentionSpec]

def attention_program_label(program: AttentionProgram) -> str:
    if isinstance(program, AttentionSpec):
        return program.label()
    return " -> ".join(spec.label() for spec in program)

def attention_program_payload(program: AttentionProgram):
    if isinstance(program, AttentionSpec):
        return asdict(program)
    return [asdict(spec) for spec in program]

def adaptive_route_metrics(model: ByteGPT, layer: int = 0) -> dict[str, Any] | None:
    """Summarize learned route usage for both adaptive routing families."""
    attn = model.blocks[layer].attn
    probabilities = attn.last_route_probs
    if probabilities is None:
        return None
    tiny = torch.finfo(probabilities.dtype).tiny

    if attn.spec.kind == "marginal_residual":
        free_heads = int(attn.spec.free_heads)
        specialist_heads = attn.n_heads - free_heads
        budget = float(attn.spec.adapt_budget)
        balance = budget * probabilities
        specialist_balance = balance.mean(dim=(0, 2))
        specialist_local = 1.0 - specialist_balance
        aggregate = torch.stack((
            probabilities.new_tensor(0.0),
            specialist_local.sum() / attn.n_heads,
            specialist_balance.sum() / attn.n_heads,
            probabilities.new_tensor(free_heads / attn.n_heads),
        ))
        specialist_rows = torch.stack((
            torch.zeros_like(specialist_local),
            specialist_local,
            specialist_balance,
            torch.zeros_like(specialist_local),
        ), dim=-1)
        free_rows = torch.zeros(free_heads, 4, device=probabilities.device)
        free_rows[:, 3] = 1.0
        per_head = torch.cat((specialist_rows, free_rows), dim=0)
        clipped = probabilities.clamp(min=tiny, max=1.0 - torch.finfo(probabilities.dtype).eps)
        entropy = -(
            clipped * clipped.log()
            + (1.0 - clipped) * (1.0 - clipped).clamp_min(tiny).log()
        ).mean()
        raw_balance = probabilities.mean()
        return {
            "names": ["self", "local", "balanced", "free"],
            "raw_mean": [0.0, float(1.0 - raw_balance), float(raw_balance), 0.0],
            "effective_mean": [float(value) for value in aggregate],
            "per_head_effective": [[float(value) for value in row] for row in per_head],
            "gate_entropy_nats": float(entropy),
            "free_floor": 0.0,
            "free_heads": free_heads,
            "adapt_budget": budget,
        }

    if attn.spec.kind == "q_residual":
        free_heads = int(attn.spec.free_heads)
        specialist_heads = attn.n_heads - free_heads
        budget = float(attn.spec.adapt_budget)
        anchor = probabilities.new_zeros(specialist_heads)
        anchor[-1] = 1.0
        balance = (
            (1.0 - budget) * anchor.view(1, specialist_heads, 1)
            + budget * probabilities
        )
        specialist_balance = balance.mean(dim=(0, 2))
        specialist_local = 1.0 - specialist_balance
        aggregate = torch.stack((
            probabilities.new_tensor(0.0),
            specialist_local.sum() / attn.n_heads,
            specialist_balance.sum() / attn.n_heads,
            probabilities.new_tensor(free_heads / attn.n_heads),
        ))
        specialist_rows = torch.stack((
            torch.zeros_like(specialist_local),
            specialist_local,
            specialist_balance,
            torch.zeros_like(specialist_local),
        ), dim=-1)
        free_rows = torch.zeros(free_heads, 4, device=probabilities.device)
        free_rows[:, 3] = 1.0
        per_head = torch.cat((specialist_rows, free_rows), dim=0)
        clipped = probabilities.clamp(min=tiny, max=1.0 - torch.finfo(probabilities.dtype).eps)
        entropy = -(
            clipped * clipped.log()
            + (1.0 - clipped) * (1.0 - clipped).clamp_min(tiny).log()
        ).mean()
        raw_balance = probabilities.mean()
        return {
            "names": ["self", "local", "balanced", "free"],
            "raw_mean": [0.0, float(1.0 - raw_balance), float(raw_balance), 0.0],
            "effective_mean": [float(value) for value in aggregate],
            "per_head_effective": [[float(value) for value in row] for row in per_head],
            "gate_entropy_nats": float(entropy),
            "free_floor": 0.0,
            "free_heads": free_heads,
            "anchor_pattern": "llb",
            "adapt_budget": budget,
            "init_strength": float(attn.spec.init_strength),
        }

    entropy = -(probabilities * probabilities.clamp_min(tiny).log()).sum(-1).mean()

    if attn.spec.kind == "adaptive_route":
        floor = float(attn.spec.free_floor)
        effective = (1.0 - floor) * probabilities
        effective = effective.clone()
        effective[..., 3] = effective[..., 3] + floor
        raw_mean = probabilities.mean(dim=(0, 1, 2))
        effective_mean = effective.mean(dim=(0, 1, 2))
        per_head = effective.mean(dim=(0, 2))
        return {
            "names": ["self", "local", "balanced", "free"],
            "raw_mean": [float(value) for value in raw_mean],
            "effective_mean": [float(value) for value in effective_mean],
            "per_head_effective": [[float(value) for value in row] for row in per_head],
            "gate_entropy_nats": float(entropy),
            "free_floor": floor,
            "free_heads": 0,
        }

    if attn.spec.kind == "adaptive_specialists":
        free_heads = int(attn.spec.free_heads)
        specialist_heads = attn.n_heads - free_heads
        specialist_fraction = specialist_heads / attn.n_heads
        raw_specialist = probabilities.mean(dim=(0, 1, 2))
        aggregate = torch.cat(
            (specialist_fraction * raw_specialist, probabilities.new_tensor([free_heads / attn.n_heads]))
        )
        specialist_per_head = probabilities.mean(dim=(0, 2))
        specialist_per_head = torch.cat(
            (specialist_per_head, torch.zeros(specialist_heads, 1, device=probabilities.device)), dim=-1
        )
        free_rows = torch.zeros(free_heads, 4, device=probabilities.device)
        free_rows[:, 3] = 1.0
        per_head = torch.cat((specialist_per_head, free_rows), dim=0)
        return {
            "names": ["self", "local", "balanced", "free"],
            "raw_mean": [float(value) for value in raw_specialist] + [0.0],
            "effective_mean": [float(value) for value in aggregate],
            "per_head_effective": [[float(value) for value in row] for row in per_head],
            "gate_entropy_nats": float(entropy),
            "free_floor": 0.0,
            "free_heads": free_heads,
        }

    if attn.spec.kind == "anchor_residual":
        free_heads = int(attn.spec.free_heads)
        specialist_heads = attn.n_heads - free_heads
        anchor = probabilities.new_zeros(specialist_heads, 3)
        anchor[:, 1] = 1.0
        if attn.spec.anchor_pattern == "llb":
            anchor[-1, 1] = 0.0
            anchor[-1, 2] = 1.0
        budget = float(attn.spec.adapt_budget)
        effective = (1.0 - budget) * anchor.view(1, specialist_heads, 1, 3) + budget * probabilities
        specialist_per_head = effective.mean(dim=(0, 2))
        aggregate_specialist = specialist_per_head.sum(0) / attn.n_heads
        aggregate = torch.cat((aggregate_specialist, probabilities.new_tensor([free_heads / attn.n_heads])))
        specialist_rows = torch.cat((
            specialist_per_head,
            torch.zeros(specialist_heads, 1, device=probabilities.device),
        ), dim=-1)
        free_rows = torch.zeros(free_heads, 4, device=probabilities.device)
        free_rows[:, 3] = 1.0
        per_head = torch.cat((specialist_rows, free_rows), dim=0)
        raw_mean = probabilities.mean(dim=(0, 1, 2))
        return {
            "names": ["self", "local", "balanced", "free"],
            "raw_mean": [float(value) for value in raw_mean] + [0.0],
            "effective_mean": [float(value) for value in aggregate],
            "per_head_effective": [[float(value) for value in row] for row in per_head],
            "gate_entropy_nats": float(entropy),
            "free_floor": 0.0,
            "free_heads": free_heads,
            "anchor_pattern": attn.spec.anchor_pattern,
            "adapt_budget": budget,
        }
    return None


@dataclass(frozen=True,slots=True)
class TrainConfig:
    steps:int=260;batch_size:int=16;learning_rate:float=3e-4;weight_decay:float=.01;grad_clip:float=1.0;eval_batches:int=12;log_every:int=130;torch_threads:int=4

def seed_all(seed:int,threads:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass

@torch.no_grad()
def evaluate(model:ByteGPT,corpus:ByteCorpus,split:str,cfg:TrainConfig,seed:int)->float:
    model.eval();g=torch.Generator().manual_seed(seed);total=0.0
    for _ in range(cfg.eval_batches):
        x,y=corpus.sample(split,batch_size=cfg.batch_size,context=model.cfg.context,generator=g);total+=model.loss(x,y).item()
    model.train();return total/cfg.eval_batches/math.log(2)

def train_lm(corpus:ByteCorpus,model_cfg:ModelConfig,spec:AttentionProgram,cfg:TrainConfig,*,seed:int,verbose:bool=False,evaluate_test:bool=False)->dict[str,Any]:
    seed_all([... ELLIPSIZATION ...]  result = [(s[code[0]], s[code[1]]) for code in codes]
    labels = [attention_program_label(program) for program in result]
    if len(labels) != len(set(labels)):
        raise RuntimeError("depth program labels must be unique")
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_id(seed: int, arm: int, label: str) -> str:
    digest = hashlib.sha256(label.encode()).hexdigest()[:10]
    return f"archive-seed{seed}-arm{arm:02d}-{digest}.json"


def write_report(output: Path) -> None:
    rows = [json.loads(path.read_text()) for path in sorted((output / "raw").glob("*.json"))]
    if not rows:
        return
    soft = next((float(row["val_bpb"]) for row in rows if row["operator"] == "softmax -> softmax"), None)
    ranked = sorted(rows, key=lambda row: float(row["val_bpb"]))
    lines = [
        "# Depth-Routed Causal Attention — Generation 11",
        "",
        f"Completed **{len(rows)}/{len(programs())}** atomic two-layer programs. Arrow order is bottom → top. Test split remained unopened.",
        "",
        "|#|bottom → top|validation bpb|Δ vs F→F|diag|age|long|Gini|e-rank|tokens/s|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, 1):
        attention = row["attention"]
        delta = float(row["val_bpb"]) - soft if soft is not None else 0.0
        lines.append(
            f"|{rank}|`{row['operator']}`|{float(row['val_bpb']):.6f}|{delta:+.6f}|"
            f"{float(attention['diagonal_mass']):.3f}|{float(attention['mean_age']):.2f}|"
            f"{float(attention['long_range_mass']):.3f}|{float(attention['column_gini']):.3f}|"
            f"{float(attention['effective_rank_fraction']):.3f}|{float(row['tokens_per_second']):.0f}|"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=int, default=1)
    parser.add_argument("--seed", type=int, default=27183)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--output", default="results/depth_program_atomic")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    arms = programs()
    if args.list:
        for index, program in enumerate(arms, 1):
            print(f"{index:2d} {attention_program_label(program)}")
        return
    if not 1 <= args.arm <= len(arms):
        raise SystemExit(f"arm must be in 1..{len(arms)}")

    program = arms[args.arm - 1]
    label = attention_program_label(program)
    output = ROOT / args.output
    path = output / "raw" / _run_id(args.seed, args.arm, label)
    if path.exists() and not args.force:
        print(f"SKIP {path.name}")
        write_report(output)
        return

    corpus = ByteCorpus(ROOT / "corpus/train.bin", ROOT / "corpus/val.bin", ROOT / "corpus/test.bin")
    result = train_lm(
        corpus,
        ModelConfig(context=96, d_model=48, n_heads=4, n_layers=2),
        program,
        TrainConfig(
            steps=args.steps,
            batch_size=12,
            eval_batches=12,
            log_every=max(1, args.steps // 2),
            torch_threads=args.threads,
        ),
        seed=args.seed,
        verbose=True,
    )
    result["arm"] = args.arm
    result["domain"] = "archive"
    _atomic_json(path, result)
    write_report(output)
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_moonshot.candidates import variable_lag_programs
from attention_moonshot.experiment import attention_program_label
from attention_moonshot.model import ModelConfig
from attention_moonshot.synthetic import SyntheticConfig, train_variable_lag_copy

TRAIN_MIN = 8
TRAIN_MAX = 48
EVAL_LAGS = (8, 16, 24, 32, 40, 48, 56, 64, 72, 80)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def write_report(output: Path) -> None:
    rows = [json.loads(path.read_text()) for path in sorted((output / "raw").glob("*.json"))]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["operator"], []).append(row)

    summary = []
    for operator, runs in groups.items():
        by_lag: dict[int, list[dict[str, float]]] = {lag: [] for lag in EVAL_LAGS}
        for run in runs:
            for item in run["lag_sweep"]:
                by_lag[int(item["lag"])].append(item)
        lag_means = {
            lag: {
                "accuracy": statistics.mean(float(item["eval_accuracy"]) for item in items),
                "bits": statistics.mean(float(item["eval_bits"]) for item in items),
            }
            for lag, items in by_lag.items() if items
        }
        interpolation = [value for lag, value in lag_means.items() if lag <= TRAIN_MAX]
        extrapolation = [value for lag, value in lag_means.items() if lag > TRAIN_MAX]
        speed, speed_sd = _mean_sd([float(run["tokens_per_second"]) for run in runs])
        summary.append({
            "operator": operator,
            "runs": len(runs),
            "interpolation_accuracy": statistics.mean(item["accuracy"] for item in interpolation),
            "extrapolation_accuracy": statistics.mean(item["accuracy"] for item in extrapolation),
            "worst_accuracy": min(item["accuracy"] for item in lag_means.values()),
            "mean_bits": statistics.mean(item["bits"] for item in lag_means.values()),
            "speed": speed,
            "speed_sd": speed_sd,
            "lag_means": lag_means,
        })
    summary.sort(key=lambda item: (-item["extrapolation_accuracy"], item["mean_bits"]))

    lines = [
        "# Variable-Lag Copy Interpolation and Extrapolation",
        "",
        f"Completed **{len(rows)}** atomic runs. Training lags were sampled uniformly from **{TRAIN_MIN}..{TRAIN_MAX}**; evaluation used fixed lags **{', '.join(map(str, EVAL_LAGS))}** at context 96.",
        "",
        "## Aggregate",
        "",
        "|#|foundation → readout|runs|train-range accuracy|unseen-lag accuracy|worst-lag accuracy|mean bits|tokens/s|",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(summary, 1):
        lines.append(
            f"|{rank}|`{item['operator']}`|{item['runs']}|"
            f"{item['interpolation_accuracy']:.4f}|{item['extrapolation_accuracy']:.4f}|"
            f"{item['worst_accuracy']:.4f}|{item['mean_bits']:.4f}|"
            f"{item['speed']:.0f} ± {item['speed_sd']:.0f}|"
        )

    lines += ["", "## Accuracy by lag", ""]
    header = "|operator|" + "|".join(str(lag) for lag in EVAL_LAGS) + "|"
    divider = "|---|" + "|".join("---:" for _ in EVAL_LAGS) + "|"
    lines += [header, divider]
    for item in summary:
        cells = [f"{item['lag_means'][lag]['accuracy']:.4f}" if lag in item["lag_means"] else "—" for lag in EVAL_LAGS]
        lines.append(f"|`{item['operator']}`|" + "|".join(cells) + "|")

    lines += ["", "## Bits by lag", "", header, divider]
    for item in summary:
        cells = [f"{item['lag_means'][lag]['bits']:.4f}" if lag in item["lag_means"] else "—" for lag in EVAL_LAGS]
        lines.append(f"|`{item['operator']}`|" + "|".join(cells) + "|")

    (output / "REPORT.md").write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8317)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="results/variable_lag_screen")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    programs = variable_lag_programs()
    if args.list:
        for index, program in enumerate(programs, 1):
            print(f"{index:2d} {attention_program_label(program)}")
        return
    if not 1 <= args.arm <= len(programs):
        raise SystemExit(f"arm must be in 1..{len(programs)}")
    program = programs[args.arm - 1]
    label = attention_program_label(program)
    digest = hashlib.sha256(label.encode()).hexdigest()[:10]
    output = ROOT / args.output
    path = output / "raw" / f"seed{args.seed}-arm{args.arm:02d}-{digest}.json"
    if path.exists() and not args.force:
        print(f"SKIP {path.name}")
        write_report(output)
        return

    result = train_variable_lag_copy(
        ModelConfig(vocab_size=64, context=96, d_model=48, n_heads=4, n_layers=2),
        program,
        SyntheticConfig(
            task="copy", steps=args.steps, batch_size=32, learning_rate=1e-3,
            eval_batches=24, torch_threads=args.threads,
        ),
        train_lag_min=TRAIN_MIN,
        train_lag_max=TRAIN_MAX,
        eval_lags=EVAL_LAGS,
        seed=args.seed,
        verbose=True,
    )
    result["arm"] = args.arm
    _atomic_json(path, result)
    write_report(output)
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
```



Analyzed

```
bash -lc find results -maxdepth 3 -name 'REPORT.md' -print | sort | while read f; do echo '===== '"$f"; sed -n '1,120p' "$f"; done
===== results/adaptive_route_screen/REPORT.md
# Floor-Gated Adaptive Causal Routing — Generation 9

Completed **3** atomic runs. Adaptive heads mix self, strictly-past local, RAPS-balanced, and free softmax routes per query while retaining a hard free-route floor.

## Archive byte corpus

Test split remained unopened.

|#|operator|runs|validation bpb|Δ vs softmax|effective S/L/B/F|tokens/s|
|---:|---|---:|---:|---:|---:|---:|
|1|`quad_route[sh=0,lh=3,bh=0,s=0.8]`|1|4.661228 ± 0.000000|-0.072131|—|12086 ± 0|
|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|1|4.662617 ± 0.000000|-0.070743|—|12210 ± 0|
|3|`softmax`|1|4.733360 ± 0.000000|+0.000000|—|33339 ± 0|
===== results/adaptive_stage1/REPORT.md
# Floor-Gated Adaptive Causal Routing — Generation 9

Completed **6** atomic runs. Adaptive heads mix self, strictly-past local, RAPS-balanced, and free softmax routes per query while retaining a hard free-route floor.

## Copy

|#|operator|runs|accuracy|bits|effective S/L/B/F|tokens/s|
|---:|---|---:|---:|---:|---:|---:|
|1|`adaptive_route[dd=3,s=0.8,ff=0.25]`|1|0.0380 ± 0.0000|5.8648 ± 0.0000|0.207/0.269/0.267/0.256|4309 ± 0|

## Mqar

|#|operator|runs|accuracy|bits|effective S/L/B/F|tokens/s|
|---:|---|---:|---:|---:|---:|---:|
|1|`softmax`|1|0.1998 ± 0.0000|3.2427 ± 0.0000|—|21357 ± 0|
|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|1|0.1855 ± 0.0000|3.6444 ± 0.0000|—|13922 ± 0|
|3|`adaptive_route[dd=3,s=0.8,ff=0.25]`|1|0.1842 ± 0.0000|3.5242 ± 0.0000|0.394/0.262/0.091/0.253|9801 ± 0|
|4|`quad_route[sh=0,lh=3,bh=0,s=0.8]`|1|0.1758 ± 0.0000|3.7924 ± 0.0000|—|46587 ± 0|

## Archive byte corpus

Test split remained unopened.

|#|operator|runs|validation bpb|Δ vs softmax|effective S/L/B/F|tokens/s|
|---:|---|---:|---:|---:|---:|---:|
|1|`adaptive_route[dd=3,s=0.8,ff=0.25]`|1|4.710329 ± 0.000000|+0.000000|0.236/0.279/0.231/0.254|2870 ± 0|
===== results/anchor_foundation_stage/REPORT.md
# Anchor-Residual Sinkhorn Foundation — Generation 12

Completed **18** atomic runs. Every candidate keeps a pure-softmax top layer and a whole free head inside the adaptive foundation layer.

## Delayed copy

|#|foundation → readout|runs|accuracy|bits|effective S/L/B/F|tokens/s|
|---:|---|---:|---:|---:|---:|---:|
|1|`quad_route[sh=0,lh=3,bh=0,s=0.8] -> softmax`|1|1.0000 ± 0.0000|0.0272 ± 0.0000|—|25870 ± 0|
|2|`anchor_residual[dd=3,fh=1,s=0.8,ap=lll,ab=0.125] -> softmax`|1|1.0000 ± 0.0000|0.0278 ± 0.0000|0.031/0.684/0.035/0.250|12837 ± 0|
|3|`anchor_residual[dd=3,fh=1,s=0.8,ap=lll,ab=0.25] -> softmax`|1|1.0000 ± 0.0000|0.0279 ± 0.0000|0.062/0.617/0.070/0.250|18505 ± 0|
|4|`anchor_residual[dd=3,fh=1,s=0.8,ap=lll,ab=0.5] -> softmax`|1|1.0000 ± 0.0000|0.0295 ± 0.0000|0.128/0.482/0.140/0.250|19430 ± 0|
|5|`anchor_residual[dd=3,fh=1,s=0.8,ap=llb,ab=0.5] -> softmax`|1|1.0000 ± 0.0000|0.0314 ± 0.0000|0.125/0.354/0.271/0.250|11306 ± 0|
|6|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|1|1.0000 ± 0.0000|0.0324 ± 0.0000|—|17471 ± 0|
|7|`anchor_residual[dd=3,fh=1,s=0.8,ap=llb,ab=0.125] -> softmax`|1|1.0000 ± 0.0000|0.0324 ± 0.0000|0.031/0.463/0.256/0.250|10548 ± 0|
|8|`anchor_residual[dd=3,fh=1,s=0.8,ap=llb,ab=0.25] -> softmax`|1|1.0000 ± 0.0000|0.0325 ± 0.0000|0.060/0.427/0.263/0.250|10991 ± 0|
|9|`softmax -> softmax`|1|1.0000 ± 0.0000|0.0345 ± 0.0000|—|37545 ± 0|

## Archive byte corpus

Test split remained unopened.

|#|foundation → readout|runs|validation bpb|Δ vs F→F|effective S/L/B/F|tokens/s|
|---:|---|---:|---:|---:|---:|---:|
|1|`anchor_residual[dd=3,fh=1,s=0.8,ap=lll,ab=0.5] -> softmax`|1|4.609256 ± 0.000000|-0.070662|0.123/0.509/0.118/0.250|7982 ± 0|
|2|`anchor_residual[dd=3,fh=1,s=0.8,ap=lll,ab=0.25] -> softmax`|1|4.610521 ± 0.000000|-0.069396|0.062/0.629/0.059/0.250|12173 ± 0|
|3|`anchor_residual[dd=3,fh=1,s=0.8,ap=lll,ab=0.125] -> softmax`|1|4.610585 ± 0.000000|-0.069333|0.031/0.689/0.030/0.250|9869 ± 0|
|4|`quad_route[sh=0,lh=3,bh=0,s=0.8] -> softmax`|1|4.610730 ± 0.000000|-0.069188|—|45669 ± 0|
|5|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|1|4.612393 ± 0.000000|-0.067524|—|21410 ± 0|
|6|`anchor_residual[dd=3,fh=1,s=0.8,ap=llb,ab=0.125] -> softmax`|1|4.612522 ± 0.000000|-0.067396|0.031/0.471/0.248/0.250|9780 ± 0|
|7|`anchor_residual[dd=3,fh=1,s=0.8,ap=llb,ab=0.25] -> softmax`|1|4.612742 ± 0.000000|-0.067175|0.061/0.443/0.246/0.250|10161 ± 0|
|8|`anchor_residual[dd=3,fh=1,s=0.8,ap=llb,ab=0.5] -> softmax`|1|4.613013 ± 0.000000|-0.066905|0.120/0.388/0.241/0.250|9378 ± 0|
|9|`softmax -> softmax`|1|4.679917 ± 0.000000|+0.000000|—|35003 ± 0|
===== results/architecture_scale/REPORT.md
# CRSA Architecture Scaling

Completed **46** validation-only atomic runs. Every comparison is paired by seed and scale value; the test split remained unopened.

## Width scale

|width|#|program|runs|validation bpb|paired Δ vs all-softmax|wins|parameters|tokens/s|
|---:|---:|---|---:|---:|---:|---:|---:|---:|
|32|1|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax`|2|4.918414 ± 0.105558|-0.036541 ± 0.001800|2/2|44672|40883 ± 8098|
|32|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|2|4.921675 ± 0.107673|-0.033281 ± 0.003916|2/2|44672|9607 ± 592|
|32|3|`softmax -> softmax`|2|4.954956 ± 0.103757|+0.000000 ± 0.000000|0/2|44672|26193 ± 14343|
|64|1|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax`|2|4.220917 ± 0.121794|-0.179606 ± 0.030003|2/2|138496|22319 ± 3032|
|64|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|2|4.224777 ± 0.118852|-0.175746 ± 0.027061|2/2|138496|12503 ± 367|
|64|3|`softmax -> softmax`|2|4.400523 ± 0.091791|+0.000000 ± 0.000000|0/2|138496|17363 ± 3349|
|96|1|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax`|2|4.007860 ± 0.120939|-0.233798 ± 0.044517|2/2|281472|11329 ± 1293|
|96|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|2|4.017937 ± 0.114749|-0.223721 ± 0.038327|2/2|281472|9054 ± 4416|
|96|3|`softmax -> softmax`|2|4.241658 ± 0.076422|+0.000000 ± 0.000000|0/2|281472|12739 ± 1156|

## Depth scale

|depth|#|program|runs|validation bpb|paired Δ vs all-softmax|wins|parameters|tokens/s|
|---:|---:|---|---:|---:|---:|---:|---:|---:|
|1|1|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|2|4.464803 ± 0.113203|-0.119118 ± 0.006082|2/2|57360|16860 ± 9503|
|1|2|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]`|2|4.465942 ± 0.110242|-0.117978 ± 0.003120|2/2|57360|31684 ± 22464|
|1|3|`softmax`|2|4.583920 ± 0.107122|+0.000000 ± 0.000000|0/2|57360|38047 ± 14030|
|3|1|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|2|4.463059 ± 0.108198|-0.139196 ± 0.000676|2/2|113520|5169 ± 195|
|3|2|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax -> softmax`|2|4.463736 ± 0.105447|-0.138518 ± 0.003427|2/2|113520|13566 ± 6910|
|3|3|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|2|4.464999 ± 0.108349|-0.137255 ± 0.000525|2/2|113520|4734 ± 1213|
|3|4|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax -> softmax`|2|4.467041 ± 0.105522|-0.135213 ± 0.003351|2/2|113520|7229 ± 1157|
|3|5|`softmax -> softmax -> softmax`|2|4.602254 ± 0.108874|+0.000000 ± 0.000000|0/2|113520|21456 ± 2250|
|4|1|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax -> softmax -> softmax`|2|4.482323 ± 0.137448|-0.131720 ± 0.025061|2/2|141600|11352 ± 4296|
|4|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax -> softmax`|2|4.484059 ± 0.134872|-0.129983 ± 0.022485|2/2|141600|7134 ± 3692|
|4|3|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax -> softmax -> softmax`|2|4.487916 ± 0.134044|-0.126126 ± 0.021657|2/2|141600|9487 ± 3106|
|4|4|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|2|4.494590 ± 0.117491|-0.119452 ± 0.005104|2/2|141600|8270 ± 2882|
|4|5|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|2|4.495004 ± 0.116846|-0.119038 ± 0.004459|2/2|141600|5374 ± 1000|
|4|6|`softmax -> softmax -> softmax -> softmax`|2|4.614043 ± 0.112387|+0.000000 ± 0.000000|0/2|141600|8002 ± 242|
===== results/archive_screen/REPORT.md
# Attention Archive Corpus — Generation 6 Screen

Completed **18/18** arms; seed **8317**; 160 steps. Test split remained unopened.

|#|operator|validation bpb|Δ vs softmax|row mass|diag|age|long|Gini|e-rank|tok/s|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|`recency[s=1.25]`|5.055577|-0.028189|1.000|0.689|0.44|0.000|0.045|0.942|34972|
|2|`recency[s=0.8]`|5.055704|-0.028061|1.000|0.541|0.84|0.000|0.051|0.876|57735|
|3|`dual_route[dd=3,lh=3,s=1.25]`|5.056185|-0.027580|1.000|0.524|2.75|0.029|0.067|0.899|25718|
|4|`recency[s=1.5]`|5.056470|-0.027295|1.000|0.750|0.33|0.000|0.042|0.962|25639|
|5|`slg[dd=3,sh=1,lh=2,s=0.8]`|5.059336|-0.024430|1.000|0.263|3.32|0.029|0.074|0.635|21787|
|6|`slg[dd=3,sh=1,lh=1,s=0.8]`|5.060661|-0.023104|1.000|0.268|5.28|0.059|0.111|0.663|25520|
|7|`dual_route[dd=3,lh=2,s=0.8]`|5.062323|-0.021442|1.000|0.282|5.26|0.059|0.113|0.684|25474|
|8|`identity`|5.064848|-0.018917|1.000|1.000|0.00|0.000|0.000|1.000|111909|
|9|`past_recency[s=0.8]`|5.067269|-0.016497|1.000|0.010|1.78|0.000|0.059|0.868|24460|
|10|`prefix_log[a=1]`|5.071805|-0.011961|1.000|0.297|7.31|0.091|0.152|0.731|14442|
|11|`prefix[a=1]`|5.071805|-0.011960|1.000|0.297|7.31|0.091|0.152|0.731|48548|
|12|`reservoir_prefix[a=1,dd=3,rho=0]`|5.073501|-0.010264|0.973|0.026|9.73|0.119|0.187|0.509|16106|
|13|`reservoir_prefix[a=1,dd=3,rho=-1]`|5.073670|-0.010096|0.987|0.028|9.67|0.118|0.193|0.511|14857|
|14|`raps[a=1,dd=4]`|5.073776|-0.009989|1.000|0.018|9.74|0.118|0.203|0.527|10213|
|15|`reservoir_prefix[a=1,dd=3,rho=-2]`|5.073820|-0.009946|0.994|0.030|9.65|0.117|0.197|0.512|14905|
|16|`raps[a=1,dd=2]`|5.073883|-0.009882|1.000|0.065|9.37|0.114|0.195|0.472|25676|
|17|`raps[a=1,dd=3]`|5.073914|-0.009852|1.000|0.032|9.64|0.117|0.201|0.513|33180|
|18|`softmax`|5.083765|+0.000000|1.000|0.055|23.66|0.405|0.493|0.135|75649|
===== results/confirmation/REPORT.m[... ELLIPSIZATION ...]13|`raps[a=1,dd=3]`|1|0.0647 ± 0.0000|5.4435 ± 0.0000|7280 ± 0|
|14|`dual_route[dd=3,lh=3,s=1.25]`|1|0.0152 ± 0.0000|5.9799 ± 0.0000|12196 ± 0|
|15|`slg[dd=3,sh=1,lh=2,s=0.8]`|1|0.0152 ± 0.0000|5.9799 ± 0.0000|18074 ± 0|
|16|`recency[s=1.25]`|1|0.0152 ± 0.0000|5.9799 ± 0.0000|22090 ± 0|

## Synthetic Recall

|#|operator|runs|accuracy|bits|tokens/s|
|---:|---|---:|---:|---:|---:|
|1|`softmax`|1|0.0326 ± 0.0000|5.0139 ± 0.0000|51474 ± 0|
===== results/route_screen/quick/REPORT.md
# Self–Local–Balanced–Free Route Screen

Completed **19** atomic runs. Every four-route candidate preserves at least one untouched causal-softmax head.

## Synthetic Copy

|#|operator|runs|accuracy|bits|tokens/s|
|---:|---|---:|---:|---:|---:|
|1|`softmax`|1|0.1816 ± 0.0000|5.2712 ± 0.0000|133555 ± 0|
|2|`quad_route[dd=3,sh=0,lh=1,bh=1,s=0.8]`|1|0.0285 ± 0.0000|5.9234 ± 0.0000|13707 ± 0|
|3|`quad_route[dd=3,sh=0,lh=1,bh=1,s=1.25]`|1|0.0267 ± 0.0000|5.9272 ± 0.0000|41323 ± 0|
|4|`recency[s=1.25]`|1|0.0171 ± 0.0000|5.9793 ± 0.0000|23893 ± 0|
|5|`quad_route[dd=3,sh=1,lh=1,bh=1,s=1.25]`|1|0.0171 ± 0.0000|5.9786 ± 0.0000|34032 ± 0|
|6|`quad_route[dd=3,sh=1,lh=1,bh=1,s=0.8]`|1|0.0170 ± 0.0000|5.9786 ± 0.0000|25790 ± 0|
|7|`raps[a=1,dd=3]`|1|0.0170 ± 0.0000|5.9792 ± 0.0000|8268 ± 0|
|8|`quad_route[dd=3,sh=0,lh=2,bh=1,s=1.25]`|1|0.0169 ± 0.0000|5.9784 ± 0.0000|59897 ± 0|
|9|`dual_route[dd=3,lh=3,s=1.25]`|1|0.0169 ± 0.0000|5.9793 ± 0.0000|9496 ± 0|
|10|`slg[dd=3,sh=1,lh=2,s=0.8]`|1|0.0169 ± 0.0000|5.9793 ± 0.0000|12480 ± 0|
|11|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|1|0.0168 ± 0.0000|5.9783 ± 0.0000|76111 ± 0|

## Synthetic Recall

|#|operator|runs|accuracy|bits|tokens/s|
|---:|---|---:|---:|---:|---:|
|1|`softmax`|1|0.0234 ± 0.0000|5.0221 ± 0.0000|126653 ± 0|
|2|`raps[a=1,dd=3]`|1|0.0234 ± 0.0000|5.0226 ± 0.0000|5774 ± 0|
|3|`quad_route[dd=3,sh=1,lh=1,bh=1,s=0.8]`|1|0.0234 ± 0.0000|5.0228 ± 0.0000|77383 ± 0|
|4|`quad_route[dd=3,sh=0,lh=1,bh=1,s=0.8]`|1|0.0234 ± 0.0000|5.0228 ± 0.0000|81387 ± 0|
|5|`quad_route[dd=3,sh=1,lh=1,bh=1,s=1.25]`|1|0.0234 ± 0.0000|5.0229 ± 0.0000|43212 ± 0|
|6|`recency[s=1.25]`|1|0.0234 ± 0.0000|5.0229 ± 0.0000|21444 ± 0|
|7|`dual_route[dd=3,lh=3,s=1.25]`|1|0.0234 ± 0.0000|5.0230 ± 0.0000|18377 ± 0|
|8|`slg[dd=3,sh=1,lh=2,s=0.8]`|1|0.0234 ± 0.0000|5.0232 ± 0.0000|32413 ± 0|
===== results/specialist_mqar_stage/REPORT.md
# Whole-Head-Anchored Adaptive Specialists — Generation 10

Completed **6** atomic runs. Three specialist heads learn per-query self/local/balanced routing while one or more entire causal-softmax heads remain untouched as an exact long-range subspace.

## Mqar

|#|operator|runs|accuracy|bits|effective S/L/B/F|tokens/s|
|---:|---|---:|---:|---:|---:|---:|
|1|`softmax`|1|0.1998 ± 0.0000|3.2427 ± 0.0000|—|59610 ± 0|
|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|1|0.1855 ± 0.0000|3.6444 ± 0.0000|—|31077 ± 0|
|3|`adaptive_specialists[dd=3,fh=1,s=0.8,si=lll,is=2]`|1|0.1840 ± 0.0000|3.6815 ± 0.0000|0.039/0.659/0.052/0.250|12177 ± 0|
|4|`adaptive_specialists[dd=3,fh=1,s=0.8,si=uniform,is=0]`|1|0.1830 ± 0.0000|3.6678 ± 0.0000|0.169/0.165/0.417/0.250|10494 ± 0|
|5|`adaptive_specialists[dd=3,fh=1,s=0.8,si=llb,is=2]`|1|0.1828 ± 0.0000|3.6779 ± 0.0000|0.017/0.468/0.265/0.250|10773 ± 0|
|6|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]`|1|0.1758 ± 0.0000|3.7924 ± 0.0000|—|37394 ± 0|
===== results/specialist_stage1/REPORT.md
# Whole-Head-Anchored Adaptive Specialists — Generation 10

Completed **16** atomic runs. Three specialist heads learn per-query self/local/balanced routing while one or more entire causal-softmax heads remain untouched as an exact long-range subspace.

## Copy

|#|operator|runs|accuracy|bits|effective S/L/B/F|tokens/s|
|---:|---|---:|---:|---:|---:|---:|
|1|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|1|1.0000 ± 0.0000|0.0273 ± 0.0000|—|16997 ± 0|
|2|`adaptive_specialists[dd=3,fh=1,s=0.8,si=llb,is=4]`|1|1.0000 ± 0.0000|0.0273 ± 0.0000|0.000/0.500/0.250/0.250|6897 ± 0|
|3|`adaptive_specialists[dd=3,fh=1,s=0.8,si=llb,is=2]`|1|1.0000 ± 0.0000|0.0274 ± 0.0000|0.014/0.482/0.254/0.250|8597 ± 0|
|4|`adaptive_specialists[dd=3,fh=1,s=0.8,si=uniform,is=0]`|1|1.0000 ± 0.0000|0.0274 ± 0.0000|0.259/0.220/0.270/0.250|8819 ± 0|
|5|`adaptive_specialists[dd=3,fh=1,s=0.8,si=lll,is=4]`|1|1.0000 ± 0.0000|0.0282 ± 0.0000|0.000/0.749/0.000/0.250|7627 ± 0|
|6|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]`|1|1.0000 ± 0.0000|0.0282 ± 0.0000|—|36667 ± 0|
|7|`adaptive_specialists[dd=3,fh=1,s=0.8,si=lll,is=2]`|1|1.0000 ± 0.0000|0.0283 ± 0.0000|0.015/0.718/0.017/0.250|9437 ± 0|
|8|`softmax`|1|1.0000 ± 0.0000|0.0345 ± 0.0000|—|67538 ± 0|

## Archive byte corpus

Test split remained unopened.

|#|operator|runs|validation bpb|Δ vs softmax|effective S/L/B/F|tokens/s|
|---:|---|---:|---:|---:|---:|---:|
|1|`adaptive_specialists[dd=3,fh=1,s=0.8,si=lll,is=2]`|1|4.699248 ± 0.000000|-0.056859|0.013/0.723/0.013/0.250|8233 ± 0|
|2|`adaptive_specialists[dd=3,fh=1,s=0.8,si=lll,is=4]`|1|4.699710 ± 0.000000|-0.056396|0.000/0.749/0.000/0.250|10321 ± 0|
|3|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]`|1|4.699718 ± 0.000000|-0.056388|—|59247 ± 0|
|4|`adaptive_specialists[dd=3,fh=1,s=0.8,si=uniform,is=0]`|1|4.701755 ± 0.000000|-0.054351|0.240/0.277/0.234/0.250|8160 ± 0|
|5|`adaptive_specialists[dd=3,fh=1,s=0.8,si=llb,is=2]`|1|4.704106 ± 0.000000|-0.052000|0.013/0.488/0.249/0.250|10212 ± 0|
|6|`adaptive_specialists[dd=3,fh=1,s=0.8,si=llb,is=4]`|1|4.704572 ± 0.000000|-0.051534|0.000/0.500/0.250/0.250|11362 ± 0|
|7|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|1|4.704581 ± 0.000000|-0.051526|—|21848 ± 0|
|8|`softmax`|1|4.756106 ± 0.000000|+0.000000|—|42463 ± 0|
===== results/synthetic/REPORT.md
# Long-Range Synthetic Stress Test

Delayed copy uses lag 48 in a 96-token context. Associative recall uses 16 unique key-value pairs and queries one key at the final position.

## Copy

|#|operator|seeds|accuracy|bits|
|---:|---|---:|---:|---:|
|1|`softmax`|1|1.0000 ± 0.0000|0.0266 ± 0.0000|
|2|`dual_route[dd=3,lh=3,s=1.25]`|1|0.0158 ± 0.0000|5.9794 ± 0.0000|
|3|`recency[s=1.25]`|1|0.0156 ± 0.0000|5.9794 ± 0.0000|

## Recall

|#|operator|seeds|accuracy|bits|
|---:|---|---:|---:|---:|
===== results/theory_audit/REPORT.md
# Theory, Causality, Precision, and Streaming Audit

## Forced diagonal mode

Float64 regular audit: **4096 matrices / 173950 rows / 0 violations**.
Float32 stress audit: probability-domain implementation had **9** violations from numerical underflow; Log-Prefix had **0**.

Extreme witness: probability-domain row `[1.0, 0.0]`; Log-Prefix row `[0.3333331048488617, 0.6666668653488159]`.

## Future-gradient probe

|operator|future-gradient norm|
|---|---:|
|`softmax`|0|
|`prefix_log`|0|
|`raps`|0|
|`reservoir_prefix`|0|
|`dual_route`|0|
|`slg`|0|
|`leaky_full_support`|0.504455864429|

## Streaming equality

|operator|max absolute error|
|---|---:|
|`prefix_log`|1.192e-07|
|`reservoir_prefix`|1.192e-07|
|`dual_route`|1.192e-07|
|`slg`|1.192e-07|

## Residual-Aware Prefix–Sinkhorn identity

Diagonal debit δ=3 is exactly equivalent to virtual prior exposure `(exp(δ)-1)A_ii`; multiplier **19.085537**, matrix error **4.441e-16**.

## Causal doubly-stochastic obstruction

A nonnegative lower-triangular matrix with unit row and column sums is the identity. Numerical alternating normalization follows that unique boundary point:

|iterations|max |A-I||Frobenius |A-I||
|---:|---:|---:|
|0|1.005222|5.290163|
|1|0.955255|2.970624|
|2|0.915459|2.764601|
|5|0.843856|2.441011|
|10|0.731570|2.107271|
|50|0.291538|0.944126|
|200|0.083476|0.293955|
|1000|0.017670|0.062934|
|2000|0.008908|0.031793|

## Causal Prefix Reservoir

The new token block is strictly causal and sub-stochastic. In the random audit, mean abstention mass was **0.0351** (range **0.0015–0.6660**). This extra reservoir dimension escapes the triangular identity obstruction without importing future rows.
===== results/variable_lag_screen/REPORT.md
# Variable-Lag Copy Interpolation and Extrapolation

Completed **10** atomic runs. Training lags were sampled uniformly from **8..48**; evaluation used fixed lags **8, 16, 24, 32, 40, 48, 56, 64, 72, 80** at context 96.

## Aggregate

|#|foundation → readout|runs|train-range accuracy|unseen-lag accuracy|worst-lag accuracy|mean bits|tokens/s|
|---:|---|---:|---:|---:|---:|---:|---:|
|1|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|2|0.9622|0.9293|0.9020|0.5872|19308 ± 1480|
|2|`past_recency[s=0.8] -> softmax`|2|0.9572|0.9220|0.8890|0.6250|50923 ± 7394|
|3|`raps[a=1,dd=3] -> softmax`|2|0.9397|0.8832|0.8442|0.8762|11355 ± 2074|
|4|`softmax -> softmax`|2|0.8514|0.7745|0.6982|1.1481|35309 ± 8039|
|5|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax`|2|0.5739|0.4791|0.4563|3.0924|30888 ± 4074|

## Accuracy by lag

|operator|8|16|24|32|40|48|56|64|72|80|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|0.9815|0.9735|0.9674|0.9604|0.9560|0.9342|0.9458|0.9398|0.9298|0.9020|
|`past_recency[s=0.8] -> softmax`|0.9775|0.9673|0.9592|0.9532|0.9492|0.9366|0.9415|0.9347|0.9228|0.8890|
|`raps[a=1,dd=3] -> softmax`|0.9642|0.9529|0.9429|0.9368|0.9300|0.9113|0.9086|0.8989|0.8810|0.8442|
|`softmax -> softmax`|0.9487|0.9034|0.8617|0.8300|0.8666|0.6982|0.7399|0.7543|0.7865|0.8173|
|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax`|0.7198|0.6172|0.5690|0.5458|0.5355|0.4563|0.4886|0.4855|0.4799|0.4624|

## Bits by lag

|operator|8|16|24|32|40|48|56|64|72|80|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|0.2305|0.2969|0.3437|0.4118|0.4659|0.6972|0.6335|0.7170|0.8552|1.2204|
|`past_recency[s=0.8] -> softmax`|0.2691|0.3382|0.3915|0.4472|0.4819|0.6304|0.6630|0.7603|0.9274|1.3406|
|`raps[a=1,dd=3] -> softmax`|0.3732|0.4472|0.5265|0.5985|0.6802|0.8755|0.9849|1.1147|1.3452|1.8155|
|`softmax -> softmax`|0.3208|0.5297|0.7580|0.9101|0.8554|2.4361|1.5442|1.5079|1.3719|1.2468|
|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax`|1.8771|2.3097|2.6153|2.8125|2.8898|3.7543|3.4161|3.5743|3.7176|3.9576|
```



Analyzed

```
bash -lc sed -n '220,520p' src/attention_moonshot/synthetic.py && sed -n '1,320p' tests/test_model_training.py && sed -n '1,300p' tests/test_candidates.py && grep -R "nope\|position_mode" -n tests scripts src | sort
    eval_cases: tuple[tuple[int, int], ...],
    seed: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train at ``mc.context`` and evaluate longer position-free sequences."""
    if not eval_cases:
        raise ValueError("eval_cases must not be empty")
    if any(context < 2 or lag < 1 or lag >= context for context, lag in eval_cases):
        raise ValueError("each eval case requires 1 <= lag < context")
    if any(context > mc.context for context, _ in eval_cases) and mc.position_mode != "none":
        raise ValueError("longer-context evaluation requires position_mode='none'")
    seed_all(seed, cfg.torch_threads)
    model = ByteGPT(mc, spec)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 171)
    start = time.perf_counter()
    tokens = 0
    lag_counts = {lag: 0 for lag in range(train_lag_min, train_lag_max + 1)}
    model.train()
    for step in range(1, cfg.steps + 1):
        x, y, mask, lag = random_lag_copy_batch(
            batch_size=cfg.batch_size,
            context=mc.context,
            vocab_size=mc.vocab_size,
            min_lag=train_lag_min,
            max_lag=train_lag_max,
            generator=generator,
        )
        lag_counts[lag] += 1
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x, y, mask)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        tokens += x.numel()
        if verbose and (step == 1 or step == cfg.steps or step % max(1, cfg.steps // 4) == 0):
            print(
                "copy_length", attention_program_label(spec), seed, step, lag,
                loss.item() / math.log(2), flush=True,
            )
    sweep = []
    for offset, (context, lag) in enumerate(eval_cases):
        bits, accuracy = evaluate_copy_lag(
            model, cfg, lag=lag, context=context, seed=seed + 913 + offset
        )
        sweep.append({
            "context": int(context), "lag": int(lag),
            "eval_bits": bits, "eval_accuracy": accuracy,
        })
    probe_generator = torch.Generator().manual_seed(seed + 1913)
    probe_x, _, _, _ = random_lag_copy_batch(
        batch_size=cfg.batch_size, context=mc.context, vocab_size=mc.vocab_size,
        min_lag=train_lag_min, max_lag=train_lag_max, generator=probe_generator,
    )
    with torch.no_grad():
        model(probe_x)
    return {
        "task": "copy_length_extrapolation",
        "operator": attention_program_label(spec),
        "spec": attention_program_payload(spec),
        "seed": seed,
        "train_context": mc.context,
        "train_lag_range": [train_lag_min, train_lag_max],
        "train_lag_counts": lag_counts,
        "context_sweep": sweep,
        "tokens_per_second": tokens / max(time.perf_counter() - start, 1e-9),
        "route": adaptive_route_metrics(model, 0),
        "model_config": asdict(mc),
        "train_config": asdict(cfg),
    }
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from attention_moonshot.data import ByteCorpus
from attention_moonshot.experiment import TrainConfig,train_lm
from attention_moonshot.model import ByteGPT,ModelConfig
from attention_moonshot.operators import AttentionSpec as S
from attention_moonshot.synthetic import associative_recall_batch,delayed_copy_batch,multi_query_recall_batch,random_lag_copy_batch,SyntheticConfig,train_synthetic,train_variable_lag_copy,train_length_extrapolation_copy

def test_full_model_prefix_outputs_ignore_future_suffix():
    torch.manual_seed(3);mc=ModelConfig(vocab_size=64,context=24,d_model=32,n_heads=4,n_layers=2)
    for spec in [S(kind='prefix_log'),S(kind='raps',diagonal_debit=3),S(kind='reservoir_prefix',diagonal_debit=3),S(kind='slg',self_heads=1,local_heads=1,slope=.8,diagonal_debit=3),S(kind='quad_route',self_heads=1,local_heads=1,balanced_heads=1,slope=.8,diagonal_debit=3)]:
        model=ByteGPT(mc,spec).eval();x=torch.randint(0,64,(2,24));y=x.clone();y[:,13:]=torch.randint(0,64,(2,11))
        with torch.no_grad():a=model(x)[:,:13];b=model(y)[:,:13]
        torch.testing.assert_close(a,b,atol=0,rtol=0)

def test_training_smoke(tmp_path:Path):
    raw=bytes((i*17+i//7)%256 for i in range(100000));paths=[]
    for n,(a,b) in enumerate(((0,60000),(60000,85000),(85000,100000))):p=tmp_path/f'{n}.bin';p.write_bytes(raw[a:b]);paths.append(p)
    c=ByteCorpus(*paths);r=train_lm(c,ModelConfig(context=32,d_model=32,n_heads=4,n_layers=1),S(kind='reservoir_prefix',diagonal_debit=3),TrainConfig(steps=4,batch_size=2,eval_batches=1,torch_threads=1),seed=8,evaluate_test=True)
    assert np.isfinite(r['val_bpb']) and np.isfinite(r['test_bpb'])

def test_synthetic_generators_and_training():
    g=torch.Generator().manual_seed(8);x,y,m=delayed_copy_batch(batch_size=3,context=32,vocab_size=64,lag=16,generator=g);seq=torch.cat((x[:,:1],y),1);assert torch.equal(seq[:,16:],seq[:,:-16])
    x,y,m=associative_recall_batch(batch_size=4,context=48,pairs=8,generator=g)
    for row in range(4):assert (int(x[row,-1]),int(y[row,-1])) in {(int(x[row,2*k]),int(x[row,2*k+1])) for k in range(8)}
    r=train_synthetic(ModelConfig(vocab_size=64,context=32,d_model=32,n_heads=4,n_layers=1),S(kind='prefix_log'),SyntheticConfig(task='copy',steps=3,batch_size=4,eval_batches=1,lag=16,torch_threads=1),seed=4)
    assert np.isfinite(r['eval_bits'])

def test_layerwise_attention_program_trains_and_serializes(tmp_path:Path):
    raw=bytes((i*29+i//11)%64 for i in range(20000));paths=[]
    for n,(a,b) in enumerate(((0,12000),(12000,17000),(17000,20000))):
        p=tmp_path/f'program-{n}.bin';p.write_bytes(raw[a:b]);paths.append(p)
    corpus=ByteCorpus(*paths)
    program=(S(kind='recency',slope=.8),S(kind='softmax'))
    result=train_lm(
        corpus,
        ModelConfig(vocab_size=64,context=16,d_model=16,n_heads=4,n_layers=2),
        program,
        TrainConfig(steps=1,batch_size=1,eval_batches=1,torch_threads=1),
        seed=17,
    )
    assert result['operator']=='recency[s=0.8] -> softmax'
    assert isinstance(result['spec'],list)
    assert [item['kind'] for item in result['spec']]==['recency','softmax']

def test_layerwise_attention_program_runs_on_synthetic_task():
    program=(S(kind='quad_route',self_heads=0,local_heads=2,balanced_heads=1,slope=.8,diagonal_debit=3),S())
    result=train_synthetic(
        ModelConfig(vocab_size=64,context=24,d_model=16,n_heads=4,n_layers=2),
        program,
        SyntheticConfig(task='copy',steps=1,batch_size=2,eval_batches=1,lag=12,torch_threads=1),
        seed=23,
    )
    assert result['operator'].endswith(' -> softmax')
    assert isinstance(result['spec'],list)


def test_multi_query_recall_batch_has_causal_queries_and_exact_targets():
    g=torch.Generator().manual_seed(29)
    x,y,m=multi_query_recall_batch(batch_size=5,context=64,pairs=8,queries=4,generator=g)
    assert x.shape==y.shape==m.shape==(5,64)
    assert torch.equal(m[:,:-4],torch.zeros_like(m[:,:-4]))
    assert torch.equal(m[:,-4:],torch.ones_like(m[:,-4:]))
    assert torch.equal(x[:,16:-4],torch.zeros_like(x[:,16:-4]))
    assert torch.all((1<=x[:,-4:])&(x[:,-4:]<=31))
    for row in range(5):
        table={int(x[row,2*k]):int(x[row,2*k+1]) for k in range(8)}
        for offset in range(4):
            query=int(x[row,-4+offset])
            assert int(y[row,-4+offset])==table[query]


def test_adaptive_training_reports_learned_route_usage(tmp_path: Path):
    raw = bytes((i * 7 + i // 5) % 64 for i in range(16000))
    paths = []
    for n, (a, b) in enumerate(((0, 10000), (10000, 13500), (13500, 16000))):
        path = tmp_path / f"adaptive-{n}.bin"
        path.write_bytes(raw[a:b])
        paths.append(path)
    corpus = ByteCorpus(*paths)
    result = train_lm(
        corpus,
        ModelConfig(vocab_size=64, context=16, d_model=16, n_heads=4, n_layers=1),
        S(kind="adaptive_route", slope=0.8, diagonal_debit=3.0, free_floor=0.25),
        TrainConfig(steps=2, batch_size=2, eval_batches=1, torch_threads=1),
        seed=29,
    )
    route = result["route"]
    assert route is not None
    assert route["names"] == ["self", "local", "balanced", "free"]
    assert len(route["raw_mean"]) == 4
    assert len(route["effective_mean"]) == 4
    assert abs(sum(route["effective_mean"]) - 1.0) < 1e-6
    assert route["effective_mean"][3] >= 0.25
    assert len(route["per_head_effective"]) == 4


def test_random_lag_copy_batch_is_seeded_and_respects_sampled_period():
    generator=torch.Generator().manual_seed(1234)
    seen=set()
    for _ in range(32):
        x,y,m,lag=random_lag_copy_batch(
            batch_size=4,context=40,vocab_size=64,min_lag=8,max_lag=16,generator=generator,
        )
        seen.add(lag)
        seq=torch.cat((x[:,:1],y),dim=1)
        assert 8<=lag<=16
        assert torch.equal(seq[:,lag:],seq[:,:-lag])
        assert torch.equal(m[:,:lag-1],torch.zeros_like(m[:,:lag-1]))
        assert torch.equal(m[:,lag-1:],torch.ones_like(m[:,lag-1:]))
    assert len(seen)>=5


def test_variable_lag_training_reports_each_fixed_eval_lag():
    result=train_variable_lag_copy(
        ModelConfig(vocab_size=64,context=32,d_model=16,n_heads=4,n_layers=1),
        (S(kind='quad_route',self_heads=0,local_heads=2,balanced_heads=1,slope=.8,diagonal_debit=3),S())[:1],
        SyntheticConfig(task='copy',steps=1,batch_size=2,eval_batches=1,torch_threads=1),
        train_lag_min=8,train_lag_max=12,eval_lags=(8,12,16),seed=31,
    )
    assert result['train_lag_range']==[8,12]
    assert [item['lag'] for item in result['lag_sweep']]==[8,12,16]
    assert all(np.isfinite(item['eval_bits']) for item in result['lag_sweep'])


def test_position_free_model_extends_context_without_future_leakage():
    torch.manual_seed(101)
    model=ByteGPT(
        ModelConfig(vocab_size=64,context=16,d_model=16,n_heads=4,n_layers=1,position_mode='none'),
        S(ki[... ELLIPSIZATION ...]].count("quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]") == 4
    assert all(len(program) == 4 for program in depth_four)


def test_marginal_residual_candidates_are_unique_bounded_and_keep_whole_free_heads() -> None:
    from attention_moonshot.candidates import marginal_residual_candidates

    specs = marginal_residual_candidates(n_heads=4)
    labels = [spec.label() for spec in specs]
    assert len(specs) == 8
    assert len(labels) == len(set(labels))
    assert labels[0] == "softmax"
    marginal = [spec for spec in specs if spec.kind == "marginal_residual"]
    assert {spec.adapt_budget for spec in marginal} == {0.125, 0.25, 0.5, 0.75}
    assert all(spec.free_heads == 1 for spec in marginal)
    assert all(free_head_count(spec, 4) >= 1 for spec in specs)


def test_marginal_variable_lag_programs_cover_the_full_budget_curve() -> None:
    from attention_moonshot.candidates import marginal_variable_lag_programs

    programs = marginal_variable_lag_programs(n_heads=4, n_layers=2)
    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
    assert labels == [
        "marginal_residual[dd=3,fh=1,s=0.8,ab=0.125] -> softmax",
        "marginal_residual[dd=3,fh=1,s=0.8,ab=0.25] -> softmax",
        "marginal_residual[dd=3,fh=1,s=0.8,ab=0.5] -> softmax",
        "marginal_residual[dd=3,fh=1,s=0.8,ab=0.75] -> softmax",
    ]
    assert all(program[0].free_heads == 1 and program[1].kind == "softmax" for program in programs)


def test_depth_scaling_programs_cover_every_crsa_density_without_reindexing_existing_arms() -> None:
    from attention_moonshot.candidates import depth_scaling_programs

    programs = depth_scaling_programs(n_heads=4, n_layers=4)
    crsa_label = "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]"
    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
    assert labels[:4] == [
        "softmax -> softmax -> softmax -> softmax",
        f"{crsa_label} -> softmax -> softmax -> softmax",
        "quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax -> softmax -> softmax",
        " -> ".join([crsa_label] * 4),
    ]
    densities = sorted(
        sum(spec.kind == "quad_route" and spec.balanced_heads == 1 for spec in program)
        for program in programs if all(spec.kind in {"softmax", "quad_route"} for spec in program)
        and not any(spec.kind == "quad_route" and spec.balanced_heads == 0 for spec in program)
    )
    assert densities == [0, 1, 2, 3, 4]


def test_variable_lag_programs_append_marginal_residuals_without_reindexing_controls() -> None:
    from attention_moonshot.candidates import variable_lag_programs

    programs = variable_lag_programs(n_heads=4, n_layers=2)
    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
    assert labels[:5] == [
        "softmax -> softmax",
        "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax",
        "raps[a=1,dd=3] -> softmax",
        "past_recency[s=0.8] -> softmax",
    ]
    assert labels[5:] == [
        "marginal_residual[dd=3,fh=1,s=0.8,ab=0.125] -> softmax",
        "marginal_residual[dd=3,fh=1,s=0.8,ab=0.25] -> softmax",
        "marginal_residual[dd=3,fh=1,s=0.8,ab=0.5] -> softmax",
    ]


def test_q_balance_spectrum_programs_sweep_strength_and_diagonal_debit() -> None:
    from attention_moonshot.candidates import q_balance_spectrum_programs

    programs = q_balance_spectrum_programs(n_heads=4, n_layers=2)
    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
    assert labels == [
        "quad_route[a=0.5,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[a=0.75,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[dd=2,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[dd=4,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[a=1.25,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[a=1.5,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
    ]
    assert all(program[0].balanced_heads == 1 for program in programs)
    assert all(program[0].local_heads == 2 and program[1].kind == "softmax" for program in programs)


def test_geometric_q_programs_span_prefix_memory_scales_and_keep_free_head() -> None:
    from attention_moonshot.candidates import geometric_q_programs

    programs = geometric_q_programs(n_heads=4, n_layers=2)
    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
    assert labels == [
        "quad_route[lam=0.9,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[lam=0.95,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[lam=0.98,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[lam=0.99,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[lam=0.995,dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
        "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax",
    ]
    assert all(program[0].balanced_heads == 1 for program in programs)
    assert all(free_head_count(program[0], 4) == 1 for program in programs)


def test_q_residual_programs_keep_llb_anchor_and_whole_free_head() -> None:
    from attention_moonshot.candidates import q_residual_programs

    programs = q_residual_programs(n_heads=4, n_layers=2)
    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
    assert len(labels) == len(set(labels)) == 4
    assert labels == [
        f"q_residual[dd=3,fh=1,s=0.8,is=4,ab={budget}] -> softmax"
        for budget in ("0.125", "0.25", "0.5", "0.75")
    ]
    assert all(free_head_count(program[0], 4) == 1 for program in programs)
grep: tests/__pycache__/test_model_training.cpython-313.pyc: binary file matches
grep: tests/__pycache__/test_model_training.cpython-313-pytest-9.0.2.pyc: binary file matches
grep: scripts/__pycache__/run_specialist_screen.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_adaptive_depth_screen.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_anchor_foundation_screen.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_archive_screen.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_route_screen.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_marginal_residual_screen.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_role_complete_screen.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_architecture_scale.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_length_extrapolation.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_adaptive_screen.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_depth_program_screen.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_depth_program_atomic.cpython-313.pyc: binary file matches
grep: scripts/__pycache__/run_anchor_residual_screen.cpython-313.pyc: binary file matches
grep: src/attention_moonshot/__pycache__/model.cpython-313.pyc: binary file matches
scripts/run_adaptive_depth_screen.py:68:        f"Completed **{len(rows)}/{len(programs())}** atomic runs at seed 27183. Test split remained unopened.",
scripts/run_adaptive_screen.py:68:                "Test split remained unopened.",
scripts/run_anchor_foundation_screen.py:106:                "Test split remained unopened.",
scripts/run_anchor_residual_screen.py:68:                "Test split remained unopened.",
scripts/run_architecture_scale.py:42:        f"Completed **{len(rows)}** validation-only atomic runs. Every comparison is paired by seed and scale value; the test split remained unopened.",
scripts/run_archive_screen.py:25:    lines=['# Attention Archive Corpus — Generation 6 Screen','',f'Completed **{len(rs)}/{len(arms())}** arms; seed **8317**; 160 steps. Test split remained unopened.','',
scripts/run_depth_program_atomic.py:70:        f"Completed **{len(rows)}/{len(programs())}** atomic two-layer programs. Arrow order is bottom → top. Test split remained unopened.",
scripts/run_depth_program_screen.py:57:        f"Completed **{len(results)}/{len(programs())}** two-layer programs; seed **27183**; 240 steps. Test split remained unopened.",
scripts/run_length_extrapolation.py:127:            n_heads=4, n_layers=2, position_mode="none",
scripts/run_marginal_residual_screen.py:85:                "Test split remained unopened.",
scripts/run_role_complete_screen.py:68:        f"Completed **{len(results)}/{len(candidates())}** arms; seed **9137**; 220 steps. Test split remained unopened.",
scripts/run_route_screen.py:109:                "The file-hash test split remains unopened in this screen.",
scripts/run_specialist_screen.py:68:                "Test split remained unopened.",
src/attention_moonshot/model.py:188:        self.position=nn.Embedding(cfg.context,cfg.d_model) if cfg.position_mode == "learned" else None
src/attention_moonshot/model.py:201:        if t>self.cfg.context and self.cfg.position_mode == "learned":
src/attention_moonshot/model.py:20:    position_mode: str = "learned"
src/attention_moonshot/model.py:27:        if self.position_mode not in {"learned", "none"}:
src/attention_moonshot/model.py:28:            raise ValueError("position_mode must be 'learned' or 'none'")
src/attention_moonshot/synthetic.py:229:    if any(context > mc.context for context, _ in eval_cases) and mc.position_mode != "none":
src/attention_moonshot/synthetic.py:230:        raise ValueError("longer-context evaluation requires position_mode='none'")
tests/test_model_training.py:131:        ModelConfig(vocab_size=64,context=16,d_model=16,n_heads=4,n_layers=1,position_mode='none'),
tests/test_model_training.py:143:def test_learned_position_model_still_rejects_context_extension():
tests/test_model_training.py:151:        ModelConfig(vocab_size=64,context=32,d_model=16,n_heads=4,n_layers=1,position_mode='none'),
grep: src/attention_moonshot/__pycache__/synthetic.cpython-313.pyc: binary file matches
```



Analyzed

```
bash -lc sed -n '115,190p' tests/test_model_training.py && sed -n '1,220p' scripts/run_length_extrapolation.py && grep -n "def variable_lag_programs\|def q_residual_programs\|def depth_scaling_programs" -A100 src/attention_moonshot/candidates.py
def test_variable_lag_training_reports_each_fixed_eval_lag():
    result=train_variable_lag_copy(
        ModelConfig(vocab_size=64,context=32,d_model=16,n_heads=4,n_layers=1),
        (S(kind='quad_route',self_heads=0,local_heads=2,balanced_heads=1,slope=.8,diagonal_debit=3),S())[:1],
        SyntheticConfig(task='copy',steps=1,batch_size=2,eval_batches=1,torch_threads=1),
        train_lag_min=8,train_lag_max=12,eval_lags=(8,12,16),seed=31,
    )
    assert result['train_lag_range']==[8,12]
    assert [item['lag'] for item in result['lag_sweep']]==[8,12,16]
    assert all(np.isfinite(item['eval_bits']) for item in result['lag_sweep'])


def test_position_free_model_extends_context_without_future_leakage():
    torch.manual_seed(101)
    model=ByteGPT(
        ModelConfig(vocab_size=64,context=16,d_model=16,n_heads=4,n_layers=1,position_mode='none'),
        S(kind='quad_route',self_heads=0,local_heads=2,balanced_heads=1,slope=.8,diagonal_debit=3),
    ).eval()
    x=torch.randint(0,64,(2,28))
    y=x.clone();y[:,19:]=torch.randint(0,64,(2,9))
    with torch.no_grad():
        full=model(x)
        changed=model(y)
    assert full.shape==(2,28,64)
    torch.testing.assert_close(full[:,:19],changed[:,:19],atol=0,rtol=0)


def test_learned_position_model_still_rejects_context_extension():
    model=ByteGPT(ModelConfig(vocab_size=64,context=16,d_model=16,n_heads=4,n_layers=1),S())
    with __import__('pytest').raises(ValueError,match='sequence exceeds'):
        model(torch.randint(0,64,(1,17)))


def test_length_extrapolation_training_reports_unseen_contexts():
    result=train_length_extrapolation_copy(
        ModelConfig(vocab_size=64,context=32,d_model=16,n_heads=4,n_layers=1,position_mode='none'),
        S(kind='quad_route',self_heads=0,local_heads=2,balanced_heads=1,slope=.8,diagonal_debit=3),
        SyntheticConfig(task='copy',steps=1,batch_size=2,eval_batches=1,torch_threads=1),
        train_lag_min=8,train_lag_max=12,eval_cases=((32,12),(48,20)),seed=37,
    )
    assert result['train_context']==32
    assert [(item['context'],item['lag']) for item in result['context_sweep']]==[(32,12),(48,20)]
    assert all(np.isfinite(item['eval_bits']) for item in result['context_sweep'])
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_moonshot.candidates import variable_lag_programs
from attention_moonshot.experiment import attention_program_label
from attention_moonshot.model import ModelConfig
from attention_moonshot.synthetic import SyntheticConfig, train_length_extrapolation_copy

TRAIN_CONTEXT = 96
TRAIN_MIN = 8
TRAIN_MAX = 48
EVAL_CASES = ((96, 48), (128, 64), (160, 80), (192, 96), (256, 128))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def write_report(output: Path) -> None:
    rows = [json.loads(path.read_text()) for path in sorted((output / "raw").glob("*.json"))]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["operator"], []).append(row)
    summaries = []
    for operator, runs in groups.items():
        by_context: dict[int, list[dict[str, float]]] = {context: [] for context, _ in EVAL_CASES}
        for run in runs:
            for item in run["context_sweep"]:
                by_context[int(item["context"])].append(item)
        means = {
            context: {
                "accuracy": statistics.mean(float(item["eval_accuracy"]) for item in items),
                "bits": statistics.mean(float(item["eval_bits"]) for item in items),
            }
            for context, items in by_context.items() if items
        }
        unseen = [value for context, value in means.items() if context > TRAIN_CONTEXT]
        speed, speed_sd = _mean_sd([float(run["tokens_per_second"]) for run in runs])
        summaries.append({
            "operator": operator,
            "runs": len(runs),
            "base_accuracy": means.get(TRAIN_CONTEXT, {}).get("accuracy", 0.0),
            "unseen_accuracy": statistics.mean(item["accuracy"] for item in unseen),
            "max_context_accuracy": means[max(means)]["accuracy"],
            "mean_bits": statistics.mean(item["bits"] for item in means.values()),
            "means": means,
            "speed": speed,
            "speed_sd": speed_sd,
        })
    summaries.sort(key=lambda item: (-item["unseen_accuracy"], item["mean_bits"]))

    contexts = [context for context, _ in EVAL_CASES]
    lines = [
        "# Position-Free Length Extrapolation",
        "",
        f"Completed **{len(rows)}** atomic runs. Models trained only at context **{TRAIN_CONTEXT}** with random lags **{TRAIN_MIN}..{TRAIN_MAX}**, then ran without retraining at contexts **{', '.join(map(str, contexts))}**. Position embeddings were disabled for every arm so this isolates causal routing rather than unseen absolute-position vectors.",
        "",
        "## Aggregate",
        "",
        "|#|foundation → readout|runs|accuracy @96|mean accuracy >96|accuracy @256|mean bits|tokens/s|",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(summaries, 1):
        lines.append(
            f"|{rank}|`{item['operator']}`|{item['runs']}|{item['base_accuracy']:.4f}|"
            f"{item['unseen_accuracy']:.4f}|{item['max_context_accuracy']:.4f}|"
            f"{item['mean_bits']:.4f}|{item['speed']:.0f} ± {item['speed_sd']:.0f}|"
        )
    for metric, title in (("accuracy", "Accuracy"), ("bits", "Bits")):
        lines += ["", f"## {title} by context", ""]
        lines.append("|operator|" + "|".join(str(context) for context in contexts) + "|")
        lines.append("|---|" + "|".join("---:" for _ in contexts) + "|")
        for item in summaries:
            cells = [f"{item['means'][context][metric]:.4f}" for context in contexts]
            lines.append(f"|`{item['operator']}`|" + "|".join(cells) + "|")
    (output / "REPORT.md").write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8317)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="results/length_extrapolation")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    programs = variable_lag_programs()
    if args.list:
        for index, program in enumerate(programs, 1):
            print(f"{index:2d} {attention_program_label(program)}")
        return
    if not 1 <= args.arm <= len(programs):
        raise SystemExit(f"arm must be in 1..{len(programs)}")
    program = programs[args.arm - 1]
    label = attention_program_label(program)
    digest = hashlib.sha256(label.encode()).hexdigest()[:10]
    output = ROOT / args.output
    path = output / "raw" / f"seed{args.seed}-arm{args.arm:02d}-{digest}.json"
    if path.exists() and not args.force:
        print(f"SKIP {path.name}")
        write_report(output)
        return

    result = train_length_extrapolation_copy(
        ModelConfig(
            vocab_size=64, context=TRAIN_CONTEXT, d_model=48,
            n_heads=4, n_layers=2, position_mode="none",
        ),
        program,
        SyntheticConfig(
            task="copy", steps=args.steps, batch_size=32, learning_rate=1e-3,
            eval_batches=12, torch_threads=args.threads,
        ),
        train_lag_min=TRAIN_MIN,
        train_lag_max=TRAIN_MAX,
        eval_cases=EVAL_CASES,
        seed=args.seed,
        verbose=True,
    )
    result["arm"] = args.arm
    _atomic_json(path, result)
    write_report(output)
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
429:def q_residual_programs(
430-    n_heads: int = 4,
431-    n_layers: int = 2,
432-) -> list[tuple[AttentionSpec, ...]]:
433-    """LLB-anchored Q-residual budget curve under lag stress."""
434-    if n_heads != 4 or n_layers != 2:
435-        raise ValueError("the Q-residual screen uses two four-head layers")
436-    free = AttentionSpec()
437-    programs = [
438-        (
439-            AttentionSpec(
440-                kind="q_residual",
441-                slope=0.8,
442-                diagonal_debit=3.0,
443-                free_heads=1,
444-                init_strength=4.0,
445-                adapt_budget=budget,
446-            ),
447-            free,
448-        )
449-        for budget in (0.125, 0.25, 0.5, 0.75)
450-    ]
451-    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
452-    if len(labels) != len(set(labels)):
453-        raise RuntimeError("Q-residual program labels must be unique")
454-    if any(free_head_count(program[0], n_heads) != 1 for program in programs):
455-        raise RuntimeError("every Q-residual candidate must keep one whole free head")
456-    return programs
457-
458:def variable_lag_programs(
459-    n_heads: int = 4,
460-    n_layers: int = 2,
461-) -> list[tuple[AttentionSpec, ...]]:
462-    """Foundation/readout programs for lag interpolation and extrapolation.
463-
464-    The two routed candidates preserve one complete unrestricted softmax head.
465-    All-head RAPS and strictly-past recency are retained as mechanism controls.
466-    """
467-    if n_heads != 4 or n_layers != 2:
468-        raise ValueError("the variable-lag screen uses two four-head layers")
469-    free = AttentionSpec()
470-    crsa = AttentionSpec(
471-        kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1,
472-        slope=0.8, diagonal_debit=3.0,
473-    )
474-    local = AttentionSpec(
475-        kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0,
476-        slope=0.8, diagonal_debit=3.0,
477-    )
478-    programs = [
479-        (free, free),
480-        (crsa, free),
481-        (local, free),
482-        (AttentionSpec(kind="raps", diagonal_debit=3.0), free),
483-        (AttentionSpec(kind="past_recency", slope=0.8), free),
484-    ]
485-    programs.extend(
486-        (
487-            AttentionSpec(
488-                kind="marginal_residual",
489-                slope=0.8,
490-                diagonal_debit=3.0,
491-                free_heads=1,
492-                adapt_budget=budget,
493-            ),
494-            free,
495-        )
496-        for budget in (0.125, 0.25, 0.5)
497-    )
498-    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
499-    if len(labels) != len(set(labels)):
500-        raise RuntimeError("variable-lag program labels must be unique")
501-    if any(free_head_count(program[0], n_heads) != 1 for program in programs[1:3]):
502-        raise RuntimeError("routed variable-lag candidates must keep one whole free head")
503-    return programs
504-
505-
506-def long_horizon_confirmation_programs(
507-    n_heads: int = 4,
508-    n_layers: int = 2,
509-) -> list[tuple[AttentionSpec, ...]]:
510-    """Frozen test-opening set after the 1,200-step validation screen."""
511-    if n_heads != 4 or n_layers != 2:
512-        raise ValueError("the long-horizon confirmation uses two four-head layers")
513-    free = AttentionSpec()
514-    crsa = AttentionSpec(
515-        kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1,
516-        slope=0.8, diagonal_debit=3.0,
517-    )
518-    local = AttentionSpec(
519-        kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0,
520-        slope=0.8, diagonal_debit=3.0,
521-    )
522-    anchored = AttentionSpec(
523-        kind="anchor_residual", slope=0.8, diagonal_debit=3.0,
524-        free_heads=1, anchor_pattern="llb", adapt_budget=0.125,
525-    )
526-    programs = [(free, free), (crsa, free), (local, free), (anchored, free)]
527-    if any(free_head_count(spec, n_heads) < 1 for program in programs for spec in program):
528-        raise RuntimeError("every long-horizon confirmation layer must preserve a free head")
529-    return programs
530-
531-
532-def foundation_scale_programs(
533-    n_heads: int = 4,
534-    n_layers: int = 2,
535-) -> list[tuple[AttentionSpec, ...]]:
536-    """Frozen width-scale trio: free baseline, CRSA, and no-balance local control."""
537-    if n_heads != 4 or n_layers != 2:
538-        raise ValueError("the width scale uses two four-head layers")
539-    free = AttentionSpec()
540-    crsa = AttentionSpec(
541-        kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1,
542-        slope=0.8, diagonal_debit=3.0,
543-    )
544-    local = AttentionSpec(
545-        kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0,
546-        slope=0.8, diagonal_debit=3.0,
547-    )
548-    return [(free, free), (crsa, free), (local, free)]
549-
550-
551:def depth_scaling_programs(
552-    n_heads: int = 4,
553-    n_layers: int = 2,
554-) -> list[tuple[AttentionSpec, ...]]:
555-    """Programs testing foundation-only routing against routing at every depth."""
556-    if n_heads != 4 or n_layers < 1:
557-        raise ValueError("depth scaling requires four heads and at least one layer")
558-    free = AttentionSpec()
559-    crsa = AttentionSpec(
560-        kind="quad_route", self_heads=0, local_heads=2, balanced_heads=1,
561-        slope=0.8, diagonal_debit=3.0,
562-    )
563-    local = AttentionSpec(
564-        kind="quad_route", self_heads=0, local_heads=3, balanced_heads=0,
565-        slope=0.8, diagonal_debit=3.0,
566-    )
567-    programs = [
568-        (free,) * n_layers,
569-        (crsa,) + (free,) * (n_layers - 1),
570-        (local,) + (free,) * (n_layers - 1),
571-    ]
572-    if n_layers > 1:
573-        programs.append((crsa,) * n_layers)
574-        # Keep the four preregistered arms at stable indices, then fill every
575-        # intermediate bottom-up CRSA density for schedule discovery.
576-        programs.extend(
577-            (crsa,) * count + (free,) * (n_layers - count)
578-            for count in range(2, n_layers)
579-        )
580-    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
581-    if len(labels) != len(set(labels)):
582-        raise RuntimeError("depth scaling contains duplicate programs")
583-    return programs
```



Analyzed

```
bash -lc sed -n '1,260p' docs/superpowers/specs/2026-08-20-causal-marginal-attention-moonshot-design.md && printf '\n--- PLAN TAIL ---\n' && tail -220 docs/superpowers/plans/2026-08-20-causal-marginal-attention-moonshot.md
# Causal Marginal Attention Moonshot Design

## Objective

Turn Causal Prefix–Sinkhorn from a single normalization into a causal routing architecture that preserves its byte-model optimization gain while retaining an explicit unrestricted long-range channel.

## Core architecture

The four head roles are fixed and independently testable:

1. **Self** — identity attention for exact current-token transport.
2. **Local** — strictly-past recency attention, excluding the diagonal.
3. **Balanced** — residual-aware causal Prefix–Sinkhorn (RAPS) with log-domain streaming state.
4. **Free** — ordinary causal softmax, left untouched for unrestricted long-range retrieval.

The initial four-head form assigns one head to each role. Candidate grids may change role counts while keeping at least one free head during long-range screening.

## Acceptance gates

- Exact causal support and zero future-input gradient.
- Batch/streaming equality within 2e-6 in float32.
- All source-based unit tests pass from a clean cache.
- Long-lag copy no longer collapses as dual-route did.
- Associative recall remains trainable.
- Archive-corpus validation beats softmax under identical budget.
- Finalists receive multi-seed test confirmation and length-extrapolation stress tests.

## Experimental order

1. Stabilize and audit the four-route implementation.
2. Cheap synthetic kill-screen across role-count variants.
3. One-seed archive-corpus screen for survivors.
4. Multi-seed confirmation on test split.
5. Length extrapolation, mechanism ablations, and throughput/Pareto analysis.
6. Promote the winning architecture into the paper and distributable repository.

--- PLAN TAIL ---
# Causal Marginal Attention Moonshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and independently verify a four-route causal attention architecture that combines self, local, Prefix–Sinkhorn-balanced, and unrestricted softmax heads.

**Architecture:** Extend the existing attention operator family with an exact four-role router and streaming state that updates only balanced heads. Screen role allocations first on synthetic long-range tasks, then on the held-out archive byte corpus, and confirm finalists across seeds and extrapolated contexts.

**Tech Stack:** Python 3.13, PyTorch 2.10 CPU, pytest, JSON/Markdown experiment reports.

**Spec:** `docs/superpowers/specs/2026-08-20-causal-marginal-attention-moonshot-design.md`

## Global Constraints

- Strict autoregressive causality: future-input gradient exactly zero.
- Deterministic seeds and identical data order across arms.
- Test split remains unopened during screening.
- At least one unrestricted causal-softmax head in long-range candidates.
- CPU-only execution with no external network dependency.

---

### Task 1: Stabilize four-route batch and streaming semantics

**Files:**
- Modify: `src/attention_moonshot/operators.py`
- Modify: `src/attention_moonshot/streaming.py`
- Modify: `tests/test_operators.py`

- [ ] Add a failing test proving only balanced heads accumulate Prefix state.
- [ ] Run the targeted test and verify the expected failure.
- [ ] Remove duplicate streaming definitions and implement one exact state update.
- [ ] Run targeted and full unit tests.
- [ ] Commit the verified operator.

### Task 2: Build a deterministic role-allocation screening harness

**Files:**
- Create: `src/attention_moonshot/candidates.py`
- Create: `scripts/run_route_screen.py`
- Create: `tests/test_candidates.py`

- [ ] Add failing tests for unique labels, free-head preservation, and canonical candidates.
- [ ] Implement candidate grids and report generation.
- [ ] Run tests and commit.

### Task 3: Run synthetic kill-screen

**Files:**
- Create: `results/route_screen_synthetic/results.json`
- Create: `results/route_screen_synthetic/REPORT.md`

- [ ] Run delayed-copy and associative-recall screens for all route candidates.
- [ ] Reject candidates that lose the explicit long-range gate.
- [ ] Promote the Pareto survivors.

### Task 4: Run archive-corpus screen and confirmation

**Files:**
- Create: `results/route_screen_archive/results.json`
- Create: `results/route_screen_archive/REPORT.md`
- Create: `results/route_confirmation/results.json`
- Create: `results/route_confirmation/REPORT.md`

- [ ] Run one-seed validation-only archive screen.
- [ ] Run three-seed validation/test confirmation for finalists and baselines.
- [ ] Compute mean, standard deviation, effect sizes, and throughput.

### Task 5: Length extrapolation and mechanism audit

**Files:**
- Create: `scripts/run_extrapolation.py`
- Create: `results/extrapolation/results.json`
- Create: `results/extrapolation/REPORT.md`

- [ ] Add tests for evaluation at contexts longer than training context using a configurable position extension.
- [ ] Run copy/recall at increasing lag and context.
- [ ] Audit diagonal, age, long-range mass, Gini, effective rank, and free-head utilization.

### Task 6: Consolidate claims and package

**Files:**
- Create: `CLAIM_LEDGER.md`
- Create: `FINAL_REPORT.md`
- Create: distributable ZIP artifact.

- [ ] Record every theorem, test, empirical result, and exact limitation.
- [ ] Run clean-cache unit tests and all report validators.
- [ ] Package code, reports, manifests, and reproducibility commands.
```



Analyzed

```
bash -lc nproc && sysctl -n hw.ncpu 2>/dev/null || true && free -h 2>/dev/null || true
5
               total        used        free      shared  buff/cache   available
Mem:           5.9Gi       893Mi       3.9Gi        16Ki       1.4Gi       5.1Gi
Swap:             0B          0B          0B
```



Analyzed

```
bash -lc find /mnt/data -iname 'crsa_harness.py' -o -iname 'crsa_suite_v1.py' -o -iname '*crsa*' | sed -n '1,120p'
/mnt/data/attention-crsa-moonshot-next
```



Analyzed

```
bash -lc find /mnt/data/attention-crsa-moonshot-next -maxdepth 4 -type f | sort | sed -n '1,240p' && du -sh /mnt/data/attention-crsa-moonshot-next
/mnt/data/attention-crsa-moonshot-next/.git/COMMIT_EDITMSG
/mnt/data/attention-crsa-moonshot-next/.git/HEAD
/mnt/data/attention-crsa-moonshot-next/.git/config
/mnt/data/attention-crsa-moonshot-next/.git/description
/mnt/data/attention-crsa-moonshot-next/.git/hooks/applypatch-msg.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/commit-msg.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/fsmonitor-watchman.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/post-update.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/pre-applypatch.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/pre-commit.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/pre-merge-commit.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/pre-push.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/pre-rebase.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/pre-receive.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/prepare-commit-msg.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/push-to-checkout.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/sendemail-validate.sample
/mnt/data/attention-crsa-moonshot-next/.git/hooks/update.sample
/mnt/data/attention-crsa-moonshot-next/.git/index
/mnt/data/attention-crsa-moonshot-next/.git/info/exclude
/mnt/data/attention-crsa-moonshot-next/.git/logs/HEAD
/mnt/data/attention-crsa-moonshot-next/.git/objects/03/3a27c8366d5a2282d8a1a0bfba3bf645b72f62
/mnt/data/attention-crsa-moonshot-next/.git/objects/04/99335aeda987956069f89d773be0deee657fed
/mnt/data/attention-crsa-moonshot-next/.git/objects/06/d28007df9ee95c2f59cc41aad69d385dc2690a
/mnt/data/attention-crsa-moonshot-next/.git/objects/0a/421acf1e84b4423c7d8df83790562edda06d20
/mnt/data/attention-crsa-moonshot-next/.git/objects/0a/888a4459a5fdf65be5144d5a6db0bd0544550f
/mnt/data/attention-crsa-moonshot-next/.git/objects/0b/5a8970e87476ef64da9a58ca0dde428e499625
/mnt/data/attention-crsa-moonshot-next/.git/objects/0d/6e29a1c7e731a6f55257b03792fb3839041d9b
/mnt/data/attention-crsa-moonshot-next/.git/objects/0e/e7990c7aa3aa14c5459030b16b91b8a3e9beda
/mnt/data/attention-crsa-moonshot-next/.git/objects/10/c23e22f2f39e287db8b8c7ed725b762f246c42
/mnt/data/attention-crsa-moonshot-next/.git/objects/12/bf2f094b6b6ffda7f688bb261a3cd6affbfe48
/mnt/data/attention-crsa-moonshot-next/.git/objects/16/e31ed09a5b1d56570702bba703de639e7a378d
/mnt/data/attention-crsa-moonshot-next/.git/objects/17/e7d5a583a516df0efc138644a329f66b5dbc6b
/mnt/data/attention-crsa-moonshot-next/.git/objects/18/1956ffa522b10c7fe703f1b4d2a0e600f50f4f
/mnt/data/attention-crsa-moonshot-next/.git/objects/18/f6c1997b3b91d7db70bbfb93c59ac06ded8908
/mnt/data/attention-crsa-moonshot-next/.git/objects/1a/417c66eca69bff5c1e644d89c2bcafd3b4874c
/mnt/data/attention-crsa-moonshot-next/.git/objects/1a/9fe39c5c5ba9f11ee742dfe0c14ff9dd2c4201
/mnt/data/attention-crsa-moonshot-next/.git/objects/1b/9eba518a19d1384f5efac17fdbb0656889d9a5
/mnt/data/attention-crsa-moonshot-next/.git/objects/1e/dfdd082f6c31753ea256692cc804585837e6d2
/mnt/data/attention-crsa-moonshot-next/.git/objects/1f/670c4571fc0026eb39b8daa7f97610871401de
/mnt/data/attention-crsa-moonshot-next/.git/objects/21/32eec42a4c04d23ba1e7e09b2d6f59bd88981c
/mnt/data/attention-crsa-moonshot-next/.git/objects/22/5bf782852d52cb253fec0921e9b5799851640d
/mnt/data/attention-crsa-moonshot-next/.git/objects/22/e346c3fab4f075fc9e88ff59a9d8d37b1bb5f5
/mnt/data/attention-crsa-moonshot-next/.git/objects/25/6ca8c7ed788c73f8be365862fe51009983204c
/mnt/data/attention-crsa-moonshot-next/.git/objects/26/3531d5b50a1d0a3b4f5cab72b1b1436e6214dd
/mnt/data/attention-crsa-moonshot-next/.git/objects/28/6cba43dbcaf31e4929c4735bab830f207d8910
/mnt/data/attention-crsa-moonshot-next/.git/objects/2a/76dc02b1c5c290f5cf49837c234f9e28b7658b
/mnt/data/attention-crsa-moonshot-next/.git/objects/2e/41870286ace6960da617f348a007a3021332e3
/mnt/data/attention-crsa-moonshot-next/.git/objects/2f/8cdbe10aaf02315ef44ab48fef86db14c3705f
/mnt/data/attention-crsa-moonshot-next/.git/objects/30/444be0a81ed8a63ee08aa081903c3ef11eb8fe
/mnt/data/attention-crsa-moonshot-next/.git/objects/33/4c38c0bdb611cf9c7c763f5b0831afd669ba52
/mnt/data/attention-crsa-moonshot-next/.git/objects/34/f632c3bb703a841935d36d2fbc1d626741c546
/mnt/data/attention-crsa-moonshot-next/.git/objects/35/c555d2dcf68849979d8a85822f0058718df2d3
/mnt/data/attention-crsa-moonshot-next/.git/objects/37/824d29aed29351347e364c461fa24b0e48b377
/mnt/data/attention-crsa-moonshot-next/.git/objects/39/932ce83e340c8fd32233926e600735d3584ee3
/mnt/data/attention-crsa-moonshot-next/.git/objects/3b/9324753dcbb81e4b8d270f612e963b6c90d206
/mnt/data/attention-crsa-moonshot-next/.git/objects/3c/4bb60777b9370307195d46218bf2c3234589c0
/mnt/data/attention-crsa-moonshot-next/.git/objects/3d/0ca3fb71ddbeee096273c169a0904ebab81379
/mnt/data/attention-crsa-moonshot-next/.git/objects/3d/5e7d0aa94b78df24f882311e433628e27d9e40
/mnt/data/attention-crsa-moonshot-next/.git/objects/3e/2980971cd4d03ff2e788e24d4bc305bb3c3b59
/mnt/data/attention-crsa-moonshot-next/.git/objects/3e/c7c9805fda1c5b91019255e46cd43b61b71727
/mnt/data/attention-crsa-moonshot-next/.git/objects/40/f39978f650d8d9ac9fdc4bc91ec01ad7895c63
/mnt/data/attention-crsa-moonshot-next/.git/objects/47/fb8ab1f96f3e316f6e21cd05243467919648bb
/mnt/data/attention-crsa-moonshot-next/.git/objects/4c/9a7fc673a22899e127a1d42c2387773c400f80
/mnt/data/attention-crsa-moonshot-next/.git/objects/4c/c0d60fd81052e55c0091da5d626da62f45c20f
/mnt/data/attention-crsa-moonshot-next/.git/objects/4d/718731e19dff0abd0c19d49737b9e737a07c10
/mnt/data/attention-crsa-moonshot-next/.git/objects/4f/403c2c84232f2fcfbe71d27e5ddbdca3bfd1fb
/mnt/data/attention-crsa-moonshot-next/.git/objects/4f/4d906bd364eae8989f0d934c79ec5123b70167
/mnt/data/attention-crsa-moonshot-next/.git/objects/50/2cdb05f479f81551d3d430d1a61924e7e0e7fa
/mnt/data/attention-crsa-moonshot-next/.git/objects/50/357fd76f3320ea4b908f7a8fd85636577a0245
/mnt/data/attention-crsa-moonshot-next/.git/objects/51/1c86d523f9eaa571215b7c1f8227ff49ca8cf2
/mnt/data/attention-crsa-moonshot-next/.git/objects/52/374d9f438141ec9776e90e9b5fef2bb2105b4b
/mnt/data/attention-crsa-moonshot-next/.git/objects/52/bae799fd2b6fa0fa7e509630e5f5646b604caa
/mnt/data/attention-crsa-moonshot-next/.git/objects/54/79f1362dfe6596529939ffa9aec77aa7d4b4dd
/mnt/data/attention-crsa-moonshot-next/.git/objects/57/a9d0a20ca705380fafdcfad5a0814171070b04
/mnt/data/attention-crsa-moonshot-next/.git/objects/59/3fc5eba723d2718e00dfbf0985e1e1b5a3aa7b
/mnt/data/attention-crsa-moonshot-next/.git/objects/5b/7e947c764b46a1ee11a59e232843e28e3d3b58
/mnt/data/attention-crsa-moonshot-next/.git/objects/5e/2eaca2eab2e5f942031125cfb453ba2f963e16
/mnt/data/attention-crsa-moonshot-next/.git/objects/60/ea765f84b6a7a63265ee7a26d518867dc12446
/mnt/data/attention-crsa-moonshot-next/.git/objects/61/1a13cf1c5ccfb8c75c36b47128caa6cd5bc5bf
/mnt/data/attention-crsa-moonshot-next/.git/objects/61/1a4d8dcf8c4ee34bd637a9164fed23e9bec55c
/mnt/data/attention-crsa-moonshot-next/.git/objects/61/43a3e34fad5ca559be3a537385120438562385
/mnt/data/attention-crsa-moonshot-next/.git/objects/61/e1d6bf1ddff7c5b17c3871ef534942bc776943
/mnt/data/attention-crsa-moonshot-next/.git/objects/63/ed54f8362752fb96108d3bc1d8131927898554
/mnt/data/attention-crsa-moonshot-next/.git/objects/65/020e15ab4e5c9f84c3fca8a00e0e3ff085d104
/mnt/data/attention-crsa-moonshot-next/.git/objects/67/9cc7049b140962d75d1ce0bc24c0292684a884
/mnt/data/attention-crsa-moonshot-next/.git/objects/68/c62e2bc3e1bc6f0e76e80798eb5dbcffef3879
/mnt/data/attention-crsa-moonshot-next/.git/objects/6a/2c224da4e01b73490f1cb6dcec1ca19a2b0d84
/mnt/data/attention-crsa-moonshot-next/.git/objects/6d/0802cba05b90bb45169ca2c75bc86b17b135e8
/mnt/data/attention-crsa-moonshot-next/.git/objects/6d/11a495a2a5c861bc20573758d3329597be6632
/mnt/data/attention-crsa-moonshot-next/.git/objects/6d/6a9fc87f6effba9381d28e42b76d4c33a02d70
/mnt/data/attention-crsa-moonshot-next/.git/objects/6f/9b2729e84c5e30fdb8bcec42ba8cc37b24f4e8
/mnt/data/attention-crsa-moonshot-next/.git/objects/6f/bb40aebd7da3135853a3a866a6bcf541294032
/mnt/data/attention-crsa-moonshot-next/.git/objects/6f/f3b5ef19f57ef61f214f0e7a80302b5b50ed4d
/mnt/data/attention-crsa-moonshot-next/.git/objects/73/934e198de784d6e2ea25666b367f847b04a895
/mnt/data/attention-crsa-moonshot-next/.git/objects/74/d168d6f0a0e302de4a2557913996c9197cd870
/mnt/data/attention-crsa-moonshot-next/.git/objects/75/cf1a4203b011ea25d58ab9bd244410358e9351
/mnt/data/attention-crsa-moonshot-next/.git/objects/7a/f7a968df5726f07d6adc879c1d2b6a65361696
/mnt/data/attention-crsa-moonshot-next/.git/objects/7b/7c5b1730d38510bccf4d40a76bcc85254f4001
/mnt/data/attention-crsa-moonshot-next/.git/objects/7b/9df88b973a126a525763f876e5dc38ee5f2230
/mnt/data/attention-crsa-moonshot-next/.git/objects/7b/cec26b1c48aff1f2cefe1a2a943b6e1444eb82
/mnt/data/attention-crsa-moonshot-next/.git/objects/7f/4401bbafbbf9f268588ff1375cc8fdfc47636a
/mnt/data/attention-crsa-moonshot-next/.git/objects/81/bf20fe7dc3573e1ff28a60e38bdfacf0cfe26e
/mnt/data/attention-crsa-moonshot-next/.git/objects/82/209cd2949812421c7a3ebc77b57db3589bb4b1
/mnt/data/attention-crsa-moonshot-next/.git/objects/83/a5e9dfe589578a0d5bd246087a45e2671381e3
/mnt/data/attention-crsa-moonshot-next/.git/objects/83/fe0fb0fce23d4ed6852e4cd38804081adf1060
/mnt/data/attention-crsa-moonshot-next/.git/objects/8a/628328bd10a464c2ea9fecba7cbc8bfc7b4d67
/mnt/data/attention-crsa-moonshot-next/.git/objects/8e/25a201914b40b9b5d0256d8f8b3fbfd5d2673a
/mnt/data/attention-crsa-moonshot-next/.git/objects/8f/9aa7aedc0db508833190cda7bd256fac27a2ea
/mnt/data/attention-crsa-moonshot-next/.git/objects/92/c5de4f557e70be7fd45acade72d419ff5c5442
/mnt/data/attention-crsa-moonshot-next/.git/objects/95/1b9f1034236f8064385ad6b638745d7af3228a
/mnt/data/attention-crsa-moonshot-next/.git/objects/98/e68dcaa1c2df53fbef4273ce8694b839963f24
/mnt/data/attention-crsa-moonshot-next/.git/objects/99/b9046166e494749f87710b5[... ELLIPSIZATION ...]objects/ac/fbd1fc4cf2124a414f778ae5f4f863d9e0566d
/mnt/data/attention-crsa-moonshot-next/.git/objects/ad/54adce5b8f1d29d0cd7bcf1a62539690af169c
/mnt/data/attention-crsa-moonshot-next/.git/objects/ae/f275e7a376b33a9caaf75f536b048862e76b61
/mnt/data/attention-crsa-moonshot-next/.git/objects/b0/8c564037324b2862a20b83bc85b48f2ba19e79
/mnt/data/attention-crsa-moonshot-next/.git/objects/b3/fb55a9ba69ecd1e8f4f639822b8063e88ec7dd
/mnt/data/attention-crsa-moonshot-next/.git/objects/b7/fc4d3d228af225d1dbd56743ec92ffdb54c54b
/mnt/data/attention-crsa-moonshot-next/.git/objects/ba/3db0861f0c3b332cf0806e79c0999008f20330
/mnt/data/attention-crsa-moonshot-next/.git/objects/ba/4f4770c46364de6df7ea9cc5bbac882d71ecf9
/mnt/data/attention-crsa-moonshot-next/.git/objects/bc/0c0fcdceec3fd62fe89a2912a46f58812496fd
/mnt/data/attention-crsa-moonshot-next/.git/objects/bd/280af0c9ef695e5baa1d5cc8791fb14864e977
/mnt/data/attention-crsa-moonshot-next/.git/objects/bd/64db2baf46958e746d01fdad5c636c110526de
/mnt/data/attention-crsa-moonshot-next/.git/objects/bf/28e8f84aae774ceaf7d61a0c24462371bcf675
/mnt/data/attention-crsa-moonshot-next/.git/objects/bf/bc7f4aa15cae26586ddcf134f30448fef59b33
/mnt/data/attention-crsa-moonshot-next/.git/objects/c1/ffe879ee704573202725dfed670ddaab4bf896
/mnt/data/attention-crsa-moonshot-next/.git/objects/c4/78d5380305969bf6a8d547584f4fadf02a76d5
/mnt/data/attention-crsa-moonshot-next/.git/objects/c5/f38eb4b44489c00fa1e1314054cd05514665c4
/mnt/data/attention-crsa-moonshot-next/.git/objects/c7/5f79bdf67ed280bcf7f196229e1e2b1aa439da
/mnt/data/attention-crsa-moonshot-next/.git/objects/c7/a3594ea751bf4d80a841e89cab6b9d9cb8edf7
/mnt/data/attention-crsa-moonshot-next/.git/objects/c7/c147dbf36582be94d9bd9aeb3e0e7d09ea9a31
/mnt/data/attention-crsa-moonshot-next/.git/objects/c9/0f1b2377ea5bf094f6e7a0787a6ae0270727fb
/mnt/data/attention-crsa-moonshot-next/.git/objects/c9/1aeccee18670b0ac1f648b8950e49780562188
/mnt/data/attention-crsa-moonshot-next/.git/objects/ca/78b7279ffa6537dfc7f691b76f4b8d9b6dd649
/mnt/data/attention-crsa-moonshot-next/.git/objects/cb/371a2d08e2de38c5d793ce8d68eb351bda65c8
/mnt/data/attention-crsa-moonshot-next/.git/objects/cc/7b221e8669b3a9da6c9330437c1da178820400
/mnt/data/attention-crsa-moonshot-next/.git/objects/cd/2326389017025217b6eaa2fc168702d14a1fdb
/mnt/data/attention-crsa-moonshot-next/.git/objects/ce/ac7d63f9c7c7fe9a15ebe1fad5e8200a6a45ad
/mnt/data/attention-crsa-moonshot-next/.git/objects/ce/e3a926f034c81a349062d211fcc6e4997146ca
/mnt/data/attention-crsa-moonshot-next/.git/objects/cf/a723c19f928e107c114526ea89f3a71bd305c0
/mnt/data/attention-crsa-moonshot-next/.git/objects/d1/25662dde573858c18e7de1c4cfb717e9ec5e94
/mnt/data/attention-crsa-moonshot-next/.git/objects/d2/56da4bcf53ef1f9b056bc5db19f257da916969
/mnt/data/attention-crsa-moonshot-next/.git/objects/d5/8215fd3f2855458cd8f51b7a3cfe58d00850f6
/mnt/data/attention-crsa-moonshot-next/.git/objects/d7/c66c431083324d817baa1673ab346be3ed3e1e
/mnt/data/attention-crsa-moonshot-next/.git/objects/db/dff10a83441f00054edda6565c7204871e729b
/mnt/data/attention-crsa-moonshot-next/.git/objects/dd/132425baa208a90f51307ac8aedb07539cf481
/mnt/data/attention-crsa-moonshot-next/.git/objects/dd/497c8b490c21469324909cfb06bb7313202a1d
/mnt/data/attention-crsa-moonshot-next/.git/objects/de/85ade4998b4b8d15d7527f761abd10a8e1b76e
/mnt/data/attention-crsa-moonshot-next/.git/objects/df/3cb64afee38cf401ed9499916e3a7c873d9ae7
/mnt/data/attention-crsa-moonshot-next/.git/objects/df/6e5059af721b09c02dfb19d1f3b9fae2285d23
/mnt/data/attention-crsa-moonshot-next/.git/objects/df/c410a2e2804aad3f7afca6b3b0a4e632b92167
/mnt/data/attention-crsa-moonshot-next/.git/objects/e1/7a6a37c24e26bcb27ca4ea6f9a3fa81b05911f
/mnt/data/attention-crsa-moonshot-next/.git/objects/e3/5c233f3a5f26a8f92c63601a15d4b9c26fc296
/mnt/data/attention-crsa-moonshot-next/.git/objects/e3/be557f237293e1a80dfd87c864335cad1eb2a9
/mnt/data/attention-crsa-moonshot-next/.git/objects/e4/30b36566490d862f353f28af6050b50796f47c
/mnt/data/attention-crsa-moonshot-next/.git/objects/e6/30bad1285173705aa44822c708f765abf06275
/mnt/data/attention-crsa-moonshot-next/.git/objects/e6/a15d53ee186000958550e8aacc83fcd74649da
/mnt/data/attention-crsa-moonshot-next/.git/objects/e9/57981e2629bad0e405da41aa6d83f8e722791d
/mnt/data/attention-crsa-moonshot-next/.git/objects/ea/4fd7ca91b30cd6f5537748cf0bb9d735ad7656
/mnt/data/attention-crsa-moonshot-next/.git/objects/ec/d972fd4f328fe9883a8ad9df08fcb0eff7bfa3
/mnt/data/attention-crsa-moonshot-next/.git/objects/ed/04ad4965ecd858248313090f75b7bf7d969240
/mnt/data/attention-crsa-moonshot-next/.git/objects/ee/d188c7d6af35841ab778a7a78fb20f587e639f
/mnt/data/attention-crsa-moonshot-next/.git/objects/f0/00b53a450998a8eb7dee31697e3990dfdcfc17
/mnt/data/attention-crsa-moonshot-next/.git/objects/f0/05bb206b55549a696a334f05ee2376b05ef160
/mnt/data/attention-crsa-moonshot-next/.git/objects/f0/0b70de3512b71ddd61337cef9fc6ba39fc5f73
/mnt/data/attention-crsa-moonshot-next/.git/objects/f1/afea47ca54f42916025e4a5408803804776405
/mnt/data/attention-crsa-moonshot-next/.git/objects/f2/8d4d4b552b5a250b28d164ff37c35d8dcbacd8
/mnt/data/attention-crsa-moonshot-next/.git/objects/f2/d96c5fc4683441c1c76bb396d38415a9b182eb
/mnt/data/attention-crsa-moonshot-next/.git/objects/f3/4baa9575e29ac89e56142ef02115eafe98d991
/mnt/data/attention-crsa-moonshot-next/.git/objects/f4/f00d06c1c14c8b66f95fd8aac28e14b418da08
/mnt/data/attention-crsa-moonshot-next/.git/objects/f5/4958f4d7e70d68214d98370d5788ca6a74de9a
/mnt/data/attention-crsa-moonshot-next/.git/objects/f7/af13807415772a0629268e8e30634a2e6c7b32
/mnt/data/attention-crsa-moonshot-next/.git/objects/f8/4fbf62cd4ebe2a52893de7fcbb22dd1607a867
/mnt/data/attention-crsa-moonshot-next/.git/objects/f8/8d1e84519260f8eaa701eee6320b359676e782
/mnt/data/attention-crsa-moonshot-next/.git/objects/f8/8d8263ee30e2fdc4197e987ba8a99335cb773d
/mnt/data/attention-crsa-moonshot-next/.git/objects/f9/4880ad1d48fb2bf70ac78ce6acfdbc59a1205b
/mnt/data/attention-crsa-moonshot-next/.git/objects/f9/efb3d6e28632f51e97bad7aa7571ace22b60f8
/mnt/data/attention-crsa-moonshot-next/.git/objects/fc/0ceed9757c30becb9c2978d55efffc8f093d0d
/mnt/data/attention-crsa-moonshot-next/.git/objects/fd/b5bf179cf1edc4e04209ed878afe72930914bd
/mnt/data/attention-crsa-moonshot-next/.git/objects/fd/bc4d2e228fd1371a9237d2d039f8cebe3883ca
/mnt/data/attention-crsa-moonshot-next/.git/objects/fe/67d78bc290cc92b3641411c5e9fd8522e7976c
/mnt/data/attention-crsa-moonshot-next/.git/objects/fe/e2a8158090c53c988573845230fe459d5870c5
/mnt/data/attention-crsa-moonshot-next/.git/worktrees/attention-moonshot-nope-v1/HEAD
/mnt/data/attention-crsa-moonshot-next/.git/worktrees/attention-moonshot-nope-v1/ORIG_HEAD
/mnt/data/attention-crsa-moonshot-next/.git/worktrees/attention-moonshot-nope-v1/commondir
/mnt/data/attention-crsa-moonshot-next/.git/worktrees/attention-moonshot-nope-v1/gitdir
/mnt/data/attention-crsa-moonshot-next/.git/worktrees/attention-moonshot-nope-v1/index
/mnt/data/attention-crsa-moonshot-next/.gitignore
/mnt/data/attention-crsa-moonshot-next/.pytest_cache/.gitignore
/mnt/data/attention-crsa-moonshot-next/.pytest_cache/CACHEDIR.TAG
/mnt/data/attention-crsa-moonshot-next/.pytest_cache/README.md
/mnt/data/attention-crsa-moonshot-next/.pytest_cache/v/cache/lastfailed
/mnt/data/attention-crsa-moonshot-next/.pytest_cache/v/cache/nodeids
/mnt/data/attention-crsa-moonshot-next/corpus/manifest.json
/mnt/data/attention-crsa-moonshot-next/corpus/test.bin
/mnt/data/attention-crsa-moonshot-next/corpus/train.bin
/mnt/data/attention-crsa-moonshot-next/corpus/val.bin
/mnt/data/attention-crsa-moonshot-next/docs/superpowers/plans/2026-08-20-causal-marginal-attention-moonshot.md
/mnt/data/attention-crsa-moonshot-next/docs/superpowers/specs/2026-08-20-causal-marginal-attention-moonshot-design.md
/mnt/data/attention-crsa-moonshot-next/pyproject.toml
/mnt/data/attention-crsa-moonshot-next/results/PARALLEL_MERGE_2026-08-20.md
/mnt/data/attention-crsa-moonshot-next/results/adaptive_archive_arm1.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_archive_arm2.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_archive_arm3.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_archive_arm4.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_archive_arm5.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_depth_screen/logs/arm01.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_depth_screen/logs/arm02.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_depth_screen/logs/arm03.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_depth_screen/logs/arm04.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_depth_screen/run.sh
/mnt/data/attention-crsa-moonshot-next/results/adaptive_depth_screen/supervisor.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_depth_screen/supervisor.pid
/mnt/data/attention-crsa-moonshot-next/results/adaptive_route_screen/REPORT.md
/mnt/data/attention-crsa-moonshot-next/results/adaptive_route_screen/archive-supervisor.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_route_screen/archive-supervisor.pid
/mnt/data/attention-crsa-moonshot-next/results/adaptive_route_screen/logs/archive-arm02-seed8317.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_route_screen/logs/archive-arm04-seed8317.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_route_screen/logs/archive-arm05-seed8317.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_route_screen/logs/archive-arm05-stage1.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_route_screen/logs/copy-arm05-stage1.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_route_screen/logs/mqar-arm01-stage1.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_route_screen/logs/mqar-arm02-stage1.log
/mnt/data/attention-crsa-moonshot-next/results/adaptive_route_screen/raw/archive-seed8317-arm01-9a5fd614ff.json
14M	/mnt/data/attention-crsa-moonshot-next
```



Analyzed

```
bash -lc python3 - <<'PY'
import torch
KEYS=list(range(10,42))


def sample(n, ndlo, ndhi):
 g=torch.Generator().manual_seed(999)
 target_dup=0; any_dup=0; conflict=0; total=0
 for _ in range(n):
  k=KEYS[int(torch.randint(len(KEYS),(1,),generator=g))]
  v=50+int(torch.randint(32,(1,),generator=g))
  keys=[k]; vals=[v]
  nd=int(torch.randint(ndlo,ndhi+1,(1,),generator=g))
  for _ in range(nd):
   dk=k+int(torch.randint(1,4,(1,),generator=g))*(1 if torch.rand(1,generator=g)<.5 else -1)
   dk=min(max(dk,KEYS[0]),KEYS[-1])
   dv=50+int(torch.randint(32,(1,),generator=g))
   keys.append(dk);vals.append(dv)
  total+=1
  if keys[1:].count(k)>0: target_dup+=1
  if len(set(keys))<len(keys): any_dup+=1
  # conflicting duplicate key values
  d={}
  bad=False
  for kk,vv in zip(keys,vals):
   if kk in d and d[kk]!=vv: bad=True
   d.setdefault(kk,vv)
  conflict+=bad
 return target_dup/total,any_dup/total,conflict/total
for rr in [(8,14),(20,30)]:
 print(rr,sample(100000,*rr))
PY
Command failed because it timed out.
```



Analyzed

```
bash -lc python3 - <<'PY'
import numpy as np
rng=np.random.default_rng(999)
for lo,hi in [(8,14),(20,30)]:
 n=200000
 td=ad=cf=0
 for _ in range(n):
  k=int(rng.integers(10,42)); v=int(rng.integers(50,82)); nd=int(rng.integers(lo,hi+1))
  keys=[k]; vals=[v]
  for __ in range(nd):
   dk=k+int(rng.integers(1,4))*(1 if rng.random()<.5 else -1)
   dk=min(max(dk,10),41); dv=int(rng.integers(50,82)); keys.append(dk);vals.append(dv)
  td += k in keys[1:]
  ad += len(set(keys))<len(keys)
  d={};bad=False
  for kk,vv in zip(keys,vals):
   if kk in d and d[kk]!=vv: bad=True
   d.setdefault(kk,vv)
  cf += bad
 print((lo,hi),td/n,ad/n,cf/n)
PY
(8, 14) 0.061605 1.0 0.99998
(20, 30) 0.062015 1.0 1.0
```



Analyzed

```
bash -lc cat > docs/superpowers/plans/2026-08-20-nope-adversarial-crystallization.md <<'EOF'
# NoPE Adversarial Retrieval and Route Crystallization Implementation Plan


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task. Steps use checkbox syntax for tracking.


**Goal:** Reproduce the external CRSA Suite v1 exactly, eliminate its retrieval confounds, then use the hardened NoPE task to discover and crystallize causal head roles.


**Architecture:** Preserve the external v1 as a legacy comparator. Add a collision-free marked associative-recall generator with randomized target rank and counterfactual same-table queries, train paired two-layer F/R/Q programs with identical initialization and batches, then screen bounded Q-residual scouts and convert measured route usage into a fixed free-head-preserving program.


**Tech Stack:** Python 3.13, PyTorch 2.10 CPU, pytest, JSON/Markdown atomic reports.


**Spec:** `docs/superpowers/specs/2026-08-20-causal-marginal-attention-moonshot-design.md`


## Global Constraints


- No learned absolute positional embeddings in every robustness arm.
- Identical initialization and batch stream within each paired seed.
- One untouched causal-softmax head in every routed layer.
- Exact external-v1 mode remains available and unmodified for comparison.
- Hardened mode uses unique keys, unique values, randomized queried rank, and same-table counterfactual queries.
- All result claims come from persisted atomic JSON and generated reports.


---


### Task 1: Build and falsify the marked-retrieval generators


- [ ] Write failing tests for exact-v1 reproduction, collision-free tables, randomized rank, determinism, and counterfactual query pairs.
- [ ] Run tests and observe the missing API failure.
- [ ] Implement the minimum generator API.
- [ ] Re-run targeted tests.


### Task 2: Build the paired NoPE training harness


- [ ] Write failing tests for program order, identical initialization digests, condition reporting, and context extension.
- [ ] Implement atomic runner and report generator.
- [ ] Run smoke tests and the full unit suite.


### Task 3: Reproduce Suite v1 and run the hardened Suite v1.1


- [ ] Run F→F, Q→Q, Q→R, R→Q, and R→R for seeds 7, 8, and 9 at 1,200 steps.
- [ ] Evaluate exact-v1 in-distribution and 20–30-distractor shift.
- [ ] Evaluate hardened random-rank, first-rank, middle-rank, last-rank, and counterfactual conditions.
- [ ] Generate paired effect tables and reject shortcut-only conclusions.


### Task 4: Adaptive scout and role crystallization


- [ ] Screen bounded Q-residual budgets on the hardened task.
- [ ] Measure query-conditioned per-head Local/Balance usage.
- [ ] Convert the measured role pattern into a fixed program with a whole free head.
- [ ] Measure zero-shot weight transfer, short recovery, and paired-from-scratch fixed deployment.


### Task 5: Verify, package, and feed the paper


- [ ] Run all tests from a clean bytecode cache.
- [ ] Write a claim ledger and integration review.
- [ ] Package scripts, source, atomic results, and reports into a ZIP.
EOF
sed -n '1,240p' docs/superpowers/plans/2026-08-20-nope-adversarial-crystallization.md
# NoPE Adversarial Retrieval and Route Crystallization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Reproduce the external CRSA Suite v1 exactly, eliminate its retrieval confounds, then use the hardened NoPE task to discover and crystallize causal head roles.

**Architecture:** Preserve the external v1 as a legacy comparator. Add a collision-free marked associative-recall generator with randomized target rank and counterfactual same-table queries, train paired two-layer F/R/Q programs with identical initialization and batches, then screen bounded Q-residual scouts and convert measured route usage into a fixed free-head-preserving program.

**Tech Stack:** Python 3.13, PyTorch 2.10 CPU, pytest, JSON/Markdown atomic reports.

**Spec:** `docs/superpowers/specs/2026-08-20-causal-marginal-attention-moonshot-design.md`

## Global Constraints

- No learned absolute positional embeddings in every robustness arm.
- Identical initialization and batch stream within each paired seed.
- One untouched causal-softmax head in every routed layer.
- Exact external-v1 mode remains available and unmodified for comparison.
- Hardened mode uses unique keys, unique values, randomized queried rank, and same-table counterfactual queries.
- All result claims come from persisted atomic JSON and generated reports.

---

### Task 1: Build and falsify the marked-retrieval generators

- [ ] Write failing tests for exact-v1 reproduction, collision-free tables, randomized rank, determinism, and counterfactual query pairs.
- [ ] Run tests and observe the missing API failure.
- [ ] Implement the minimum generator API.
- [ ] Re-run targeted tests.

### Task 2: Build the paired NoPE training harness

- [ ] Write failing tests for program order, identical initialization digests, condition reporting, and context extension.
- [ ] Implement atomic runner and report generator.
- [ ] Run smoke tests and the full unit suite.

### Task 3: Reproduce Suite v1 and run the hardened Suite v1.1

- [ ] Run F→F, Q→Q, Q→R, R→Q, and R→R for seeds 7, 8, and 9 at 1,200 steps.
- [ ] Evaluate exact-v1 in-distribution and 20–30-distractor shift.
- [ ] Evaluate hardened random-rank, first-rank, middle-rank, last-rank, and counterfactual conditions.
- [ ] Generate paired effect tables and reject shortcut-only conclusions.

### Task 4: Adaptive scout and role crystallization

- [ ] Screen bounded Q-residual budgets on the hardened task.
- [ ] Measure query-conditioned per-head Local/Balance usage.
- [ ] Convert the measured role pattern into a fixed program with a whole free head.
- [ ] Measure zero-shot weight transfer, short recovery, and paired-from-scratch fixed deployment.

### Task 5: Verify, package, and feed the paper

- [ ] Run all tests from a clean bytecode cache.
- [ ] Write a claim ledger and integration review.
- [ ] Package scripts, source, atomic results, and reports into a ZIP.
```



Analyzed

```
bash -lc cat > tests/test_marked_retrieval.py <<'PY'
from __future__ import annotations


import torch


import attention_moonshot.synthetic as synthetic


KM, VM, SEP = 200, 201, 254




def _parse_table(row: torch.Tensor) -> tuple[list[int], list[int], int]:
    nonpad = row[row.ne(0)].tolist()
    sep_index = nonpad.index(SEP)
    table = nonpad[:sep_index]
    query = nonpad[sep_index:]
    assert query[:2] == [SEP, KM]
    assert len(table) % 4 == 0
    keys: list[int] = []
    values: list[int] = []
    for offset in range(0, len(table), 4):
        assert table[offset] == KM
        assert table[offset + 2] == VM
        keys.append(table[offset + 1])
        values.append(table[offset + 3])
    return keys, values, query[2]




def test_legacy_marked_recall_reproduces_target_first_layout() -> None:
    function = getattr(synthetic, "legacy_marked_recall_batch", None)
    assert function is not None, "legacy marked-recall API is missing"
    batch = function(
        batch_size=8,
        context=160,
        distractor_min=8,
        distractor_max=8,
        generator=torch.Generator().manual_seed(7),
    )
    assert batch.tokens.shape == (8, 160)
    assert batch.targets.shape == (8,)
    assert torch.equal(batch.target_rank, torch.zeros(8, dtype=torch.long))
    for row, target in zip(batch.tokens, batch.targets, strict=True):
        keys, values, query_key = _parse_table(row)
        assert query_key == keys[0]
        assert int(target) == values[0]




def test_collision_free_marked_recall_has_unique_keys_values_and_random_target_rank() -> None:
    function = getattr(synthetic, "collision_free_marked_recall_batch", None)
    assert function is not None, "collision-free marked-recall API is missing"
    batch = function(
        batch_size=96,
        context=160,
        distractor_min=20,
        distractor_max=30,
        target_mode="random",
        generator=torch.Generator().manual_seed(11),
    )
    ranks = set(batch.target_rank.tolist())
    assert len(ranks) >= 10
    for row, target, rank in zip(
        batch.tokens, batch.targets, batch.target_rank, strict=True
    ):
        keys, values, query_key = _parse_table(row)
        assert len(keys) == len(set(keys))
        assert len(values) == len(set(values))
        assert keys.count(query_key) == 1
        assert keys[int(rank)] == query_key
        assert values[int(rank)] == int(target)




def test_collision_free_marked_recall_is_seed_deterministic() -> None:
    function = getattr(synthetic, "collision_free_marked_recall_batch", None)
    assert function is not None, "collision-free marked-recall API is missing"
    kwargs = dict(
        batch_size=16,
        context=160,
        distractor_min=8,
        distractor_max=14,
        target_mode="random",
    )
    first = function(**kwargs, generator=torch.Generator().manual_seed(19))
    second = function(**kwargs, generator=torch.Generator().manual_seed(19))
    torch.testing.assert_close(first.tokens, second.tokens, atol=0, rtol=0)
    torch.testing.assert_close(first.targets, second.targets, atol=0, rtol=0)
    torch.testing.assert_close(first.target_rank, second.target_rank, atol=0, rtol=0)




def test_counterfactual_queries_share_table_but_require_different_values() -> None:
    function = getattr(synthetic, "counterfactual_marked_recall_batch", None)
    assert function is not None, "counterfactual marked-recall API is missing"
    first, second = function(
        batch_size=32,
        context=160,
        distractor_min=20,
        distractor_max=30,
FFFFF                                                                    [100%]
=================================== FAILURES ===================================
___________ test_legacy_marked_recall_reproduces_target_first_layout ___________

    def test_legacy_marked_recall_reproduces_target_first_layout() -> None:
        function = getattr(synthetic, "legacy_marked_recall_batch", None)
>       assert function is not None, "legacy marked-recall API is missing"
E       AssertionError: legacy marked-recall API is missing
E       assert None is not None

tests/test_marked_retrieval.py:29: AssertionError
_ test_collision_free_marked_recall_has_unique_keys_values_and_random_target_rank _

    def test_collision_free_marked_recall_has_unique_keys_values_and_random_target_rank() -> None:
        function = getattr(synthetic, "collision_free_marked_recall_batch", None)
>       assert function is not None, "collision-free marked-recall API is missing"
E       AssertionError: collision-free marked-recall API is missing
E       assert None is not None

tests/test_marked_retrieval.py:48: AssertionError
___________ test_collision_free_marked_recall_is_seed_deterministic ____________

    def test_collision_free_marked_recall_is_seed_deterministic() -> None:
        function = getattr(synthetic, "collision_free_marked_recall_batch", None)
>       assert function is not None, "collision-free marked-recall API is missing"
E       AssertionError: collision-free marked-recall API is missing
E       assert None is not None

tests/test_marked_retrieval.py:72: AssertionError
_____ test_counterfactual_queries_share_table_but_require_different_values _____

    def test_counterfactual_queries_share_table_but_require_different_values() -> None:
        function = getattr(synthetic, "counterfactual_marked_recall_batch", None)
>       assert function is not None, "counterfactual marked-recall API is missing"
E       AssertionError: counterfactual marked-recall API is missing
E       assert None is not None

tests/test_marked_retrieval.py:89: AssertionError
______ test_target_modes_place_query_at_first_middle_and_last_table_rank _______

    def test_target_modes_place_query_at_first_middle_and_last_table_rank() -> None:
        function = getattr(synthetic, "collision_free_marked_recall_batch", None)
>       assert function is not None, "collision-free marked-recall API is missing"
E       AssertionError: collision-free marked-recall API is missing
E       assert None is not None

tests/test_marked_retrieval.py:109: AssertionError
=========================== short test summary info ============================
FAILED tests/test_marked_retrieval.py::test_legacy_marked_recall_reproduces_target_first_layout - AssertionError: legacy marked-recall API is missing
assert None is not None
FAILED tests/test_marked_retrieval.py::test_collision_free_marked_recall_has_unique_keys_values_and_random_target_rank - AssertionError: collision-free marked-recall API is missing
assert None is not None
FAILED tests/test_marked_retrieval.py::test_collision_free_marked_recall_is_seed_deterministic - AssertionError: collision-free marked-recall API is missing
assert None is not None
FAILED tests/test_marked_retrieval.py::test_counterfactual_queries_share_table_but_require_different_values - AssertionError: counterfactual marked-recall API is missing
assert None is not None
FAILED tests/test_marked_retrieval.py::test_target_modes_place_query_at_first_middle_and_last_table_rank - AssertionError: collision-free marked-recall API is missing
assert None is not None
5 failed in 1.73s


Command 'bash -lc 'cat > tests/test_marked_retrieval.py <<'"'"'PY'"'"'
from __future__ import annotations

import torch

import attention_moonshot.synthetic as synthetic

KM, VM, SEP = 200, 201, 254


def _parse_table(row: torch.Tensor) -> tuple[list[int], list[int], int]:
    nonpad = row[row.ne(0)].tolist()
    sep_index = nonpad.index(SEP)
    table = nonpad[:sep_index]
    query = nonpad[sep_index:]
    assert query[:2] == [SEP, KM]
    assert len(table) % 4 == 0
    keys: list[int] = []
    values: list[int] = []
    for offset in range(0, len(table), 4):
        assert table[offset] == KM
        assert table[offset + 2] == VM
        keys.append(table[offset + 1])
        values.append(table[offset + 3])
    return keys, values, query[2]


def test_legacy_marked_recall_reproduces_target_first_layout() -> None:
    function = getattr(synthetic, "legacy_marked_recall_batch", None)
    assert function is not None, "legacy marked-recall API is missing"
    batch = function(
        batch_size=8,
        context=160,
        distractor_min=8,
        distractor_max=8,
        generator=torch.Generator().manual_seed(7),
    )
    assert batch.tokens.shape == (8, 160)
    assert batch.targets.shape == (8,)
    assert torch.equal(batch.target_rank, torch.zeros(8, dtype=torch.long))
    for row, target in zip(batch.tokens, batch.targets, strict=True):
        keys, values, query_key = _parse_table(row)
        assert query_key == keys[0]
        assert int(target) == values[0]


def test_collision_free_marked_recall_has_unique_keys_values_and_random_target_rank() -> None:
    function = getattr(synthetic, "collision_free_marked_recall_batch", None)
    assert function is not None, "collision-free marked-recall API is missing"
    batch = function(
        batch_size=96,
        context=160,
        distractor_min=20,
        distractor_max=30,
        target_mode="random",
        generator=torch.Generator().manual_seed(11),
    )
    ranks = set(batch.target_rank.tolist())
    assert len(ranks) >= 10
    for row, target, rank in zip(
        batch.tokens, batch.targets, batch.target_rank, strict=True
    ):
        keys, values, query_key = _parse_table(row)
        assert len(keys) == len(set(keys))
        assert len(values) == len(set(values))
        assert keys.count(query_key) == 1
        assert keys[int(rank)] == query_key
        assert values[int(rank)] == int(target)


def test_collision_free_marked_recall_is_seed_deterministic() -> None:
    function = getattr(synthetic, "collision_free_marked_recall_batch", None)
    assert function is not None, "collision-free marked-recall API is missing"
    kwargs = dict(
        batch_size=16,
        context=160,
        distractor_min=8,
        distractor_max=14,
        target_mode="random",
    )
    first = function(**kwargs, generator=torch.Generator().manual_seed(19))
    second = function(**kwargs, generator=torch.Generator().manual_seed(19))
    torch.testing.assert_close(first.tokens, second.tokens, atol=0, rtol=0)
    torch.testing.assert_close(first.targets, second.targets, atol=0, rtol=0)
    torch.testing.assert_close(first.target_rank, second.target_rank, atol=0, rtol=0)


def test_counterfactual_queries_share_table_but_require_different_values() -> None:
    function = getattr(synthetic, "counterfactual_marked_recall_batch", None)
    assert function is not None, "counterfactual marked-recall API is missing"
    first, second = function(
        batch_size=32,
        context=160,
        distractor_min=20,
        distractor_max=30,
        generator=torch.Generator().manual_seed(23),
    )
    assert torch.all(first.targets.ne(second.targets))
    for row_a, row_b in zip(first.tokens, second.tokens, strict=True):
        keys_a, values_a, query_a = _parse_table(row_a)
        keys_b, values_b, query_b = _parse_table(row_b)
        assert keys_a == keys_b
        assert values_a == values_b
        assert query_a != query_b
        assert torch.equal(row_a[:-1], row_b[:-1])


def test_target_modes_place_query_at_first_middle_and_last_table_rank() -> None:
    function = getattr(synthetic, "collision_free_marked_recall_batch", None)
    assert function is not None, "collision-free marked-recall API is missing"
    for mode in ("first", "middle", "last"):
        batch = function(
            batch_size=12,
            context=160,
            distractor_min=20,
            distractor_max=30,
            target_mode=mode,
            generator=torch.Generator().manual_seed(29),
        )
        for distractors, rank in zip(
            batch.distractor_count, batch.target_rank, strict=True
        ):
            count = int(distractors)
            expected = {"first": 0, "middle": count // 2, "last": count}[mode]
            assert int(rank) == expected
PY
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_marked_retrieval.py -q'' failed with status 1.
```



Analyzed

```
bash -lc python3 - <<'PY'
from pathlib import Path
path=Path('src/attention_moonshot/synthetic.py')
text=path.read_text()
anchor='def _batch(mc:ModelConfig,cfg:SyntheticConfig,g:torch.Generator):\n'
insert=r'''


@dataclass(frozen=True, slots=True)
class MarkedRecallBatch:
    """A fixed-context marked key/value table and one final query."""


    tokens: torch.Tensor
    targets: torch.Tensor
    target_rank: torch.Tensor
    distractor_count: torch.Tensor




_MARKED_KEYS = tuple(range(10, 42))
_MARKED_VALUES = tuple(range(50, 82))
_MARKED_KEY = 200
_MARKED_VALUE = 201
_MARKED_SEPARATOR = 254




def _validate_marked_recall_request(
    *, batch_size: int, context: int, distractor_min: int, distractor_max: int
) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if distractor_min < 0 or distractor_min > distractor_max:
        raise ValueError("invalid distractor range")
    if distractor_max >= len(_MARKED_KEYS):
        raise ValueError("not enough unique marked-recall keys")
    required = 4 * (distractor_max + 1) + 3
    if required > context:
        raise ValueError(f"context {context} is shorter than required length {required}")




def _pack_marked_rows(rows: list[list[int]], *, context: int) -> torch.Tensor:
    tokens = torch.zeros(len(rows), context, dtype=torch.long)
    for index, row in enumerate(rows):
        tokens[index, context - len(row) :] = torch.tensor(row, dtype=torch.long)
    return tokens




def legacy_marked_recall_batch(
    *,
    batch_size: int,
    context: int,
    distractor_min: int,
    distractor_max: int,
    generator: torch.Generator,
) -> MarkedRecallBatch:
    """Reproduce the external Suite-v1 generator byte for byte in semantics.


    The target pair is always first. Distractor keys are sampled as clamped
    offsets around the target, intentionally preserving the original duplicate
    and collision behavior for an apples-to-apples comparator.
    """
    _validate_marked_recall_request(
        batch_size=batch_size,
        context=context,
        distractor_min=distractor_min,
        distractor_max=distractor_max,
    )
    rows: list[list[int]] = []
    targets: list[int] = []
    counts: list[int] = []
    for _ in range(batch_size):
        key = _MARKED_KEYS[
            int(torch.randint(len(_MARKED_KEYS), (1,), generator=generator).item())
        ]
        value = _MARKED_VALUES[
            int(torch.randint(len(_MARKED_VALUES), (1,), generator=generator).item())
        ]
        row = [_MARKED_KEY, key, _MARKED_VALUE, value]
        distractors = int(
            torch.randint(
                distractor_min, distractor_max + 1, (1,), generator=generator
            ).item()
        )
        for _ in range(distractors):
            offset = int(torch.randint(1, 4, (1,), generator=generator).item())
            direction = 1 if float(torch.rand(1, generator=generator).item()) < 0.5 else -1
            distractor_key = min(max(key + direction * offset, _MARKED_KEYS[0]), _MARKED_KEYS[-1])
            distractor_value = _MARKED_VALUES[
                int(torch.randint(len(_MARKED_VALUES), (1,), generator=generator).item())
            ]
            row.extend((_MARKED_KEY, distractor_key, _MARKED_VALUE, distractor_value))
        row.extend((_MARKED_SEPARATOR, _MARKED_KEY, key))
        rows.append(row)
        targets.append(value)
        counts.append(distractors)
    return MarkedRecallBatch(
        tokens=_pack_marked_rows(rows, context=context),
        targets=torch.tensor(targets, dtype=torch.long),
        target_rank=torch.zeros(batch_size, dtype=torch.long),
        distractor_count=torch.tensor(counts, dtype=torch.long),
    )
.....                                                                    [100%]
5 passed in 2.45s
```



Analyzed

```
bash -lc cat >> tests/test_marked_retrieval.py <<'PY'




def test_marked_recall_programs_match_preregistered_depth_order() -> None:
    import attention_moonshot.candidates as candidates
    from attention_moonshot.experiment import attention_program_label


    function = getattr(candidates, "marked_recall_programs", None)
    assert function is not None, "marked-recall program set is missing"
    labels = [attention_program_label(program) for program in function()]
    q = "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]"
    r = "quad_route[sh=0,lh=3,bh=0,s=0.8]"
    assert labels == [
        "softmax -> softmax",
        f"{q} -> {q}",
        f"{q} -> {r}",
        f"{r} -> {q}",
        f"{r} -> {r}",
    ]




def test_marked_recall_training_is_nope_paired_and_reports_hardened_conditions() -> None:
    from attention_moonshot.candidates import marked_recall_programs
    from attention_moonshot.model import ModelConfig


    config_type = getattr(synthetic, "MarkedRecallConfig", None)
    function = getattr(synthetic, "train_marked_recall", None)
    assert config_type is not None, "marked-recall training config is missing"
    assert function is not None, "marked-recall training API is missing"
    model_config = ModelConfig(
        vocab_size=256,
        context=40,
        d_model=16,
        n_heads=4,
        n_layers=2,
        position_mode="none",
    )
    train_config = config_type(
        steps=1,
        batch_size=2,
        eval_batches=1,
        eval_batch_size=2,
        train_distractor_min=1,
        train_distractor_max=2,
        shift_distractor_min=3,
        shift_distractor_max=4,
        torch_threads=1,
    )
    first = function(
        model_config,
        marked_recall_programs()[0],
        train_config,
        seed=31,
        mode="hardened",
    )
    second = function(
        model_config,
        marked_recall_programs()[1],
        train_config,
        seed=31,
        mode="hardened",
    )
    assert first["initialization_digest"] == second["initialization_digest"]
    assert first["batch_stream_digest"] == second["batch_stream_digest"]
    assert set(first["conditions"]) == {
        "in_dist_random",
        "shift_random",
        "shift_first",
        "shift_middle",
        "shift_last",
        "counterfactual",
    }
    assert 0.0 <= first["conditions"]["counterfactual"]["both_correct"] <= 1.0




def test_marked_recall_training_rejects_learned_absolute_positions() -> None:
    import pytest
    from attention_moonshot.candidates import marked_recall_programs
    from attention_moonshot.model import ModelConfig


    config_type = getattr(synthetic, "MarkedRecallConfig", None)
    function = getattr(synthetic, "train_marked_recall", None)
    assert config_type is not None and function is not None
    with pytest.raises(ValueError, match="position_mode='none'"):
        function(
            ModelConfig(
                vocab_size=256,
                context=40,
                d_model=16,
                n_heads=4,
                n_layers=2,
                position_mode="learned",
            ),
            marked_recall_programs()[0],
            config_type(
                steps=1,
                batch_size=2,
                eval_batches=1,
                eval_batch_size=2,
                train_distractor_min=1,
                train_distractor_max=2,
                shift_distractor_min=3,
```



Analyzed

```
.....FFF                                                                 [100%]
=================================== FAILURES ===================================
_________ test_marked_recall_programs_match_preregistered_depth_order __________

    def test_marked_recall_programs_match_preregistered_depth_order() -> None:
        import attention_moonshot.candidates as candidates
        from attention_moonshot.experiment import attention_program_label
    
        function = getattr(candidates, "marked_recall_programs", None)
>       assert function is not None, "marked-recall program set is missing"
E       AssertionError: marked-recall program set is missing
E       assert None is not None

tests/test_marked_retrieval.py:132: AssertionError
__ test_marked_recall_training_is_nope_paired_and_reports_hardened_conditions __

    def test_marked_recall_training_is_nope_paired_and_reports_hardened_conditions() -> None:
>       from attention_moonshot.candidates import marked_recall_programs
E       ImportError: cannot import name 'marked_recall_programs' from 'attention_moonshot.candidates' (/mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/candidates.py)

tests/test_marked_retrieval.py:146: ImportError
________ test_marked_recall_training_rejects_learned_absolute_positions ________

    def test_marked_recall_training_rejects_learned_absolute_positions() -> None:
        import pytest
>       from attention_moonshot.candidates import marked_recall_programs
E       ImportError: cannot import name 'marked_recall_programs' from 'attention_moonshot.candidates' (/mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/candidates.py)

tests/test_marked_retrieval.py:201: ImportError
=========================== short test summary info ============================
FAILED tests/test_marked_retrieval.py::test_marked_recall_programs_match_preregistered_depth_order - AssertionError: marked-recall program set is missing
assert None is not None
FAILED tests/test_marked_retrieval.py::test_marked_recall_training_is_nope_paired_and_reports_hardened_conditions - ImportError: cannot import name 'marked_recall_programs' from 'attention_moonshot.candidates' (/mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/candidates.py)
FAILED tests/test_marked_retrieval.py::test_marked_recall_training_rejects_learned_absolute_positions - ImportError: cannot import name 'marked_recall_programs' from 'attention_moonshot.candidates' (/mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/candidates.py)
3 failed, 5 passed in 2.69s


Command 'bash -lc 'cat >> tests/test_marked_retrieval.py <<'"'"'PY'"'"'


def test_marked_recall_programs_match_preregistered_depth_order() -> None:
    import attention_moonshot.candidates as candidates
    from attention_moonshot.experiment import attention_program_label

    function = getattr(candidates, "marked_recall_programs", None)
    assert function is not None, "marked-recall program set is missing"
    labels = [attention_program_label(program) for program in function()]
    q = "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]"
    r = "quad_route[sh=0,lh=3,bh=0,s=0.8]"
    assert labels == [
        "softmax -> softmax",
        f"{q} -> {q}",
        f"{q} -> {r}",
        f"{r} -> {q}",
        f"{r} -> {r}",
    ]


def test_marked_recall_training_is_nope_paired_and_reports_hardened_conditions() -> None:
    from attention_moonshot.candidates import marked_recall_programs
    from attention_moonshot.model import ModelConfig

    config_type = getattr(synthetic, "MarkedRecallConfig", None)
    function = getattr(synthetic, "train_marked_recall", None)
    assert config_type is not None, "marked-recall training config is missing"
    assert function is not None, "marked-recall training API is missing"
    model_config = ModelConfig(
        vocab_size=256,
        context=40,
        d_model=16,
        n_heads=4,
        n_layers=2,
        position_mode="none",
    )
    train_config = config_type(
        steps=1,
        batch_size=2,
        eval_batches=1,
        eval_batch_size=2,
        train_distractor_min=1,
        train_distractor_max=2,
        shift_distractor_min=3,
        shift_distractor_max=4,
        torch_threads=1,
    )
    first = function(
        model_config,
        marked_recall_programs()[0],
        train_config,
        seed=31,
        mode="hardened",
    )
    second = function(
        model_config,
        marked_recall_programs()[1],
        train_config,
        seed=31,
        mode="hardened",
    )
    assert first["initialization_digest"] == second["initialization_digest"]
    assert first["batch_stream_digest"] == second["batch_stream_digest"]
    assert set(first["conditions"]) == {
        "in_dist_random",
        "shift_random",
        "shift_first",
        "shift_middle",
        "shift_last",
        "counterfactual",
    }
    assert 0.0 <= first["conditions"]["counterfactual"]["both_correct"] <= 1.0


def test_marked_recall_training_rejects_learned_absolute_positions() -> None:
    import pytest
    from attention_moonshot.candidates import marked_recall_programs
    from attention_moonshot.model import ModelConfig

    config_type = getattr(synthetic, "MarkedRecallConfig", None)
    function = getattr(synthetic, "train_marked_recall", None)
    assert config_type is not None and function is not None
    with pytest.raises(ValueError, match="position_mode='"'"'none'"'"'"):
        function(
            ModelConfig(
                vocab_size=256,
                context=40,
                d_model=16,
                n_heads=4,
                n_layers=2,
                position_mode="learned",
            ),
            marked_recall_programs()[0],
            config_type(
                steps=1,
                batch_size=2,
                eval_batches=1,
                eval_batch_size=2,
                train_distractor_min=1,
                train_distractor_max=2,
                shift_distractor_min=3,
                shift_distractor_max=4,
                torch_threads=1,
            ),
            seed=37,
            mode="hardened",
        )
PY
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_marked_retrieval.py -q'' failed with status 1.
```



Analyzed

```
bash -lc cat >> src/attention_moonshot/candidates.py <<'PY'




def marked_recall_programs(
    n_heads: int = 4,
    n_layers: int = 2,
) -> list[tuple[AttentionSpec, ...]]:
    """Preregistered F/R/Q depth programmes for marked NoPE retrieval."""
    if n_heads != 4 or n_layers != 2:
        raise ValueError("marked recall uses two four-head layers")
    free = AttentionSpec()
    local = AttentionSpec(
        kind="quad_route",
        self_heads=0,
        local_heads=3,
        balanced_heads=0,
        slope=0.8,
        diagonal_debit=3.0,
    )
    crsa = AttentionSpec(
        kind="quad_route",
        self_heads=0,
        local_heads=2,
        balanced_heads=1,
        slope=0.8,
        diagonal_debit=3.0,
    )
    programs = [
        (free, free),
        (crsa, crsa),
        (crsa, local),
        (local, crsa),
        (local, local),
    ]
    labels = [" -> ".join(spec.label() for spec in program) for program in programs]
    if len(labels) != len(set(labels)):
        raise RuntimeError("marked-recall program labels must be unique")
    if any(free_head_count(spec, n_heads) < 1 for program in programs for spec in program):
        raise RuntimeError("every marked-recall layer must retain a whole free head")
    return programs
PY
```



Analyzed

```
bash -lc python3 - <<'PY'
from pathlib import Path
path=Path('src/attention_moonshot/synthetic.py')
text=path.read_text()
text=text.replace('from dataclasses import asdict,dataclass\nimport math,time\n', 'from dataclasses import asdict,dataclass\nimport hashlib,math,time\n')
anchor='def _batch(mc:ModelConfig,cfg:SyntheticConfig,g:torch.Generator):\n'
insert=r'''


@dataclass(frozen=True, slots=True)
class MarkedRecallConfig:
    steps: int = 1200
    batch_size: int = 24
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_batches: int = 8
    eval_batch_size: int = 48
    train_distractor_min: int = 8
    train_distractor_max: int = 14
    shift_distractor_min: int = 20
    shift_distractor_max: int = 30
    torch_threads: int = 1




def _tensor_digest(hasher: Any, tensor: torch.Tensor) -> None:
    compact = tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    hasher.update(compact.numpy().tobytes())




def _model_initialization_digest(model: ByteGPT) -> str:
    hasher = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        hasher.update(name.encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()




def _make_marked_batch(
    *,
    mode: str,
    batch_size: int,
    context: int,
    distractor_min: int,
    distractor_max: int,
    target_mode: str,
    generator: torch.Generator,
) -> MarkedRecallBatch:
    if mode == "legacy":
        if target_mode != "first":
            raise ValueError("legacy mode supports only target_mode='first'")
        return legacy_marked_recall_batch(
            batch_size=batch_size,
            context=context,
            distractor_min=distractor_min,
            distractor_max=distractor_max,
            generator=generator,
        )
    if mode == "hardened":
        return collision_free_marked_recall_batch(
            batch_size=batch_size,
            context=context,
            distractor_min=distractor_min,
            distractor_max=distractor_max,
            target_mode=target_mode,
            generator=generator,
        )
    raise ValueError("mode must be 'legacy' or 'hardened'")




@torch.no_grad()
def _evaluate_marked_condition(
    model: ByteGPT,
    cfg: MarkedRecallConfig,
    *,
    mode: str,
    distractor_min: int,
    distractor_max: int,
    target_mode: str,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total_correct = 0
    count = 0
    rank_correct = {"early": 0, "middle": 0, "late": 0}
    rank_count = {"early": 0, "middle": 0, "late": 0}
    model.eval()
    for _ in range(cfg.eval_batches):
        batch = _make_marked_batch(
            mode=mode,
            batch_size=cfg.eval_batch_size,
            context=model.cfg.context,
            distractor_min=distractor_min,
            distractor_max=distractor_max,
            target_mode=target_mode,
            generator=generator,
        )
        logits = model(batch.tokens)[:, -1]
        per = torch.nn.functional.cross_entropy(logits, batch.targets, reduction="none")
        predictions = logits.argmax(-1)
.....F..                                                                 [100%]
=================================== FAILURES ===================================
_________ test_marked_recall_programs_match_preregistered_depth_order __________

    def test_marked_recall_programs_match_preregistered_depth_order() -> None:
        import attention_moonshot.candidates as candidates
        from attention_moonshot.experiment import attention_program_label
    
        function = getattr(candidates, "marked_recall_programs", None)
        assert function is not None, "marked-recall program set is missing"
        labels = [attention_program_label(program) for program in function()]
        q = "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]"
        r = "quad_route[sh=0,lh=3,bh=0,s=0.8]"
>       assert labels == [
            "softmax -> softmax",
            f"{q} -> {q}",
            f"{q} -> {r}",
            f"{r} -> {q}",
            f"{r} -> {r}",
        ]
E       AssertionError: assert ['softmax -> ...,bh=0,s=0.8]'] == ['softmax -> ...,bh=0,s=0.8]']
E         
E         At index 2 diff: 'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]' != 'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[sh=0,lh=3,bh=0,s=0.8]'
E         
E         Full diff:
E           [
E               'softmax -> softmax',
E               'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> '
E               'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]',
E         -     'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[sh=0,lh=3,bh=0,s=0.8]',
E         -     'quad_route[sh=0,lh=3,bh=0,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]',
E         -     'quad_route[sh=0,lh=3,bh=0,s=0.8] -> quad_route[sh=0,lh=3,bh=0,s=0.8]',
E         +     'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> '
E         +     'quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]',
E         +     'quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> '
E         +     'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]',
E         +     'quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> '
E         +     'quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]',
E           ]

tests/test_marked_retrieval.py:136: AssertionError
=========================== short test summary info ============================
FAILED tests/test_marked_retrieval.py::test_marked_recall_programs_match_preregistered_depth_order - AssertionError: assert ['softmax -> ...,bh=0,s=0.8]'] == ['softmax -> ...,bh=0,s=0.8]']
  
  At index 2 diff: 'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]' != 'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[sh=0,lh=3,bh=0,s=0.8]'
  
  Full diff:
    [
        'softmax -> softmax',
        'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> '
        'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]',
  -     'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[sh=0,lh=3,bh=0,s=0.8]',
  -     'quad_route[sh=0,lh=3,bh=0,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]',
  -     'quad_route[sh=0,lh=3,bh=0,s=0.8] -> quad_route[sh=0,lh=3,bh=0,s=0.8]',
  +     'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> '
  +     'quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]',
  +     'quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> '
  +     'quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]',
  +     'quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> '
  +     'quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]',
    ]
1 failed, 7 passed in 2.69s


Command 'bash -lc 'python3 - <<'"'"'PY'"'"'
from pathlib import Path
path=Path('"'"'src/attention_moonshot/synthetic.py'"'"')
text=path.read_text()
text=text.replace('"'"'from dataclasses import asdict,dataclass\nimport math,time\n'"'"', '"'"'from dataclasses import asdict,dataclass\nimport hashlib,math,time\n'"'"')
anchor='"'"'def _batch(mc:ModelConfig,cfg:SyntheticConfig,g:torch.Generator):\n'"'"'
insert=r'"'"''"'"''"'"'

@dataclass(frozen=True, slots=True)
class MarkedRecallConfig:
    steps: int = 1200
    batch_size: int = 24
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_batches: int = 8
    eval_batch_size: int = 48
    train_distractor_min: int = 8
    train_distractor_max: int = 14
    shift_distractor_min: int = 20
    shift_distractor_max: int = 30
    torch_threads: int = 1


def _tensor_digest(hasher: Any, tensor: torch.Tensor) -> None:
    compact = tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    hasher.update(compact.numpy().tobytes())


def _model_initialization_digest(model: ByteGPT) -> str:
    hasher = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        hasher.update(name.encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _make_marked_batch(
    *,
    mode: str,
    batch_size: int,
    context: int,
    distractor_min: int,
    distractor_max: int,
    target_mode: str,
    generator: torch.Generator,
) -> MarkedRecallBatch:
    if mode == "legacy":
        if target_mode != "first":
            raise ValueError("legacy mode supports only target_mode='"'"'first'"'"'")
        return legacy_marked_recall_batch(
            batch_size=batch_size,
            context=context,
            distractor_min=distractor_min,
            distractor_max=distractor_max,
            generator=generator,
        )
    if mode == "hardened":
        return collision_free_marked_recall_batch(
            batch_size=batch_size,
            context=context,
            distractor_min=distractor_min,
            distractor_max=distractor_max,
            target_mode=target_mode,
            generator=generator,
        )
    raise ValueError("mode must be '"'"'legacy'"'"' or '"'"'hardened'"'"'")


@torch.no_grad()
def _evaluate_marked_condition(
    model: ByteGPT,
    cfg: MarkedRecallConfig,
    *,
    mode: str,
    distractor_min: int,
    distractor_max: int,
    target_mode: str,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total_correct = 0
    count = 0
    rank_correct = {"early": 0, "middle": 0, "late": 0}
    rank_count = {"early": 0, "middle": 0, "late": 0}
    model.eval()
    for _ in range(cfg.eval_batches):
        batch = _make_marked_batch(
            mode=mode,
            batch_size=cfg.eval_batch_size,
            context=model.cfg.context,
            distractor_min=distractor_min,
            distractor_max=distractor_max,
            target_mode=target_mode,
            generator=generator,
        )
        logits = model(batch.tokens)[:, -1]
        per = torch.nn.functional.cross_entropy(logits, batch.targets, reduction="none")
        predictions = logits.argmax(-1)
        correct = predictions.eq(batch.targets)
        total_loss += float(per.sum())
        total_correct += int(correct.sum())
        count += batch.targets.numel()
        fractions = batch.target_rank.to(torch.float32) / batch.distractor_count.clamp_min(1)
        for label, mask in (
            ("early", fractions < 1.0 / 3.0),
            ("middle", (fractions >= 1.0 / 3.0) & (fractions < 2.0 / 3.0)),
            ("late", fractions >= 2.0 / 3.0),
        ):
            rank_correct[label] += int((correct & mask).sum())
            rank_count[label] += int(mask.sum())
    model.train()
    rank_accuracy = {
        label: rank_correct[label] / rank_count[label]
        for label in rank_count
        if rank_count[label]
    }
    return {
        "accuracy": total_correct / count,
        "bits": total_loss / count / math.log(2),
        "count": count,
        "rank_accuracy": rank_accuracy,
    }


@torch.no_grad()
def _evaluate_counterfactual_marked_recall(
    model: ByteGPT,
    cfg: MarkedRecallConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    correct_a = correct_b = both_correct = changed = count = 0
    loss = 0.0
    model.eval()
    for _ in range(cfg.eval_batches):
        first, second = counterfactual_marked_recall_batch(
            batch_size=cfg.eval_batch_size,
            context=model.cfg.context,
            distractor_min=cfg.shift_distractor_min,
            distractor_max=cfg.shift_distractor_max,
            generator=generator,
        )
        logits_a = model(first.tokens)[:, -1]
        logits_b = model(second.tokens)[:, -1]
        pred_a = logits_a.argmax(-1)
        pred_b = logits_b.argmax(-1)
        ok_a = pred_a.eq(first.targets)
        ok_b = pred_b.eq(second.targets)
        correct_a += int(ok_a.sum())
        correct_b += int(ok_b.sum())
        both_correct += int((ok_a & ok_b).sum())
        changed += int(pred_a.ne(pred_b).sum())
        loss += float(torch.nn.functional.cross_entropy(logits_a, first.targets, reduction="sum"))
        loss += float(torch.nn.functional.cross_entropy(logits_b, second.targets, reduction="sum"))
        count += first.targets.numel()
    model.train()
    return {
        "accuracy_first": correct_a / count,
        "accuracy_second": correct_b / count,
        "both_correct": both_correct / count,
        "prediction_change_rate": changed / count,
        "bits": loss / (2 * count) / math.log(2),
        "count": count,
    }


def train_marked_recall(
    mc: ModelConfig,
    spec: AttentionProgram,
    cfg: MarkedRecallConfig,
    *,
    seed: int,
    mode: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train a paired NoPE marked-retrieval model and evaluate its shift suite."""
    if mc.position_mode != "none":
        raise ValueError("marked retrieval requires position_mode='"'"'none'"'"'")
    if mode not in {"legacy", "hardened"}:
        raise ValueError("mode must be '"'"'legacy'"'"' or '"'"'hardened'"'"'")
    _validate_marked_recall_request(
        batch_size=cfg.batch_size,
        context=mc.context,
        distractor_min=cfg.train_distractor_min,
        distractor_max=max(cfg.train_distractor_max, cfg.shift_distractor_max),
    )
    seed_all(seed, cfg.torch_threads)
    model = ByteGPT(mc, spec)
    initialization_digest = _model_initialization_digest(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator().manual_seed(3000 + seed)
    stream_hasher = hashlib.sha256()
    start = time.perf_counter()
    tokens = 0
    model.train()
    for step in range(1, cfg.steps + 1):
        batch = _make_marked_batch(
            mode=mode,
            batch_size=cfg.batch_size,
            context=mc.context,
            distractor_min=cfg.train_distractor_min,
            distractor_max=cfg.train_distractor_max,
            target_mode="first" if mode == "legacy" else "random",
            generator=generator,
        )
        _tensor_digest(stream_hasher, batch.tokens)
        _tensor_digest(stream_hasher, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.tokens)[:, -1]
        loss = torch.nn.functional.cross_entropy(logits, batch.targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        tokens += batch.tokens.numel()
        if verbose and (
            step == 1 or step == cfg.steps or step % max(1, cfg.steps // 4) == 0
        ):
            print(
                "marked_recall", mode, attention_program_label(spec), seed,
                step, float(loss / math.log(2)), flush=True,
            )

    if mode == "legacy":
        conditions = {
            "in_dist": _evaluate_marked_condition(
                model, cfg, mode="legacy",
                distractor_min=cfg.train_distractor_min,
                distractor_max=cfg.train_distractor_max,
                target_mode="first", seed=9999,
            ),
            "shift_20_30": _evaluate_marked_condition(
                model, cfg, mode="legacy",
                distractor_min=cfg.shift_distractor_min,
                distractor_max=cfg.shift_distractor_max,
                target_mode="first", seed=10009,
            ),
        }
    else:
        conditions = {
            "in_dist_random": _evaluate_marked_condition(
                model, cfg, mode="hardened",
                distractor_min=cfg.train_distractor_min,
                distractor_max=cfg.train_distractor_max,
                target_mode="random", seed=9999,
            ),
            "shift_random": _evaluate_marked_condition(
                model, cfg, mode="hardened",
                distractor_min=cfg.shift_distractor_min,
                distractor_max=cfg.shift_distractor_max,
                target_mode="random", seed=10009,
            ),
            "shift_first": _evaluate_marked_condition(
                model, cfg, mode="hardened",
                distractor_min=cfg.shift_distractor_min,
                distractor_max=cfg.shift_distractor_max,
                target_mode="first", seed=10019,
            ),
            "shift_middle": _evaluate_marked_condition(
                model, cfg, mode="hardened",
                distractor_min=cfg.shift_distractor_min,
                distractor_max=cfg.shift_distractor_max,
                target_mode="middle", seed=10029,
            ),
            "shift_last": _evaluate_marked_condition(
                model, cfg, mode="hardened",
                distractor_min=cfg.shift_distractor_min,
                distractor_max=cfg.shift_distractor_max,
                target_mode="last", seed=10039,
            ),
            "counterfactual": _evaluate_counterfactual_marked_recall(
                model, cfg, seed=10049,
            ),
        }

    probe = _make_marked_batch(
        mode=mode,
        batch_size=cfg.eval_batch_size,
        context=mc.context,
        distractor_min=cfg.shift_distractor_min,
        distractor_max=cfg.shift_distractor_max,
        target_mode="first" if mode == "legacy" else "random",
        generator=torch.Generator().manual_seed(11001),
    )
    with torch.no_grad():
        model(probe.tokens)
    route_layers = [adaptive_route_metrics(model, layer) for layer in range(mc.n_layers)]
    return {
        "task": "marked_recall",
        "mode": mode,
        "operator": attention_program_label(spec),
        "spec": attention_program_payload(spec),
        "seed": seed,
        "initialization_digest": initialization_digest,
        "batch_stream_digest": stream_hasher.hexdigest(),
        "conditions": conditions,
        "tokens_per_second": tokens / max(time.perf_counter() - start, 1e-9),
        "route_layers": route_layers,
        "model_config": asdict(mc),
        "train_config": asdict(cfg),
    }

'"'"''"'"''"'"'
if anchor not in text:
    raise SystemExit('"'"'anchor missing'"'"')
text=text.replace(anchor,insert+anchor)
path.write_text(text)
PY
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_marked_retrieval.py -q'' failed with status 1.
```



Analyzed

```
bash -lc python3 - <<'PY'
from pathlib import Path
p=Path('src/attention_moonshot/candidates.py')
s=p.read_text()
old='''    local = AttentionSpec(
        kind="quad_route",
        self_heads=0,
        local_heads=3,
        balanced_heads=0,
        slope=0.8,
        diagonal_debit=3.0,
    )
    crsa = AttentionSpec(
'''
new='''    local = AttentionSpec(
        kind="quad_route",
        self_heads=0,
        local_heads=3,
        balanced_heads=0,
        slope=0.8,
    )
    crsa = AttentionSpec(
'''
# replace last occurrence? There are many similar. We appended at end, use rsplit
idx=s.rfind(old)
if idx<0: raise SystemExit('pattern not found')
s=s[:idx]+s[idx:].replace(old,new,1)
p.write_text(s)
PY
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_marked_retrieval.py -q
........                                                                 [100%]
8 passed in 3.64s
```



Analyzed

```
bash -lc rm -rf .pytest_cache src/attention_moonshot/__pycache__ tests/__pycache__ && PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q
........................................................................ [ 75%]
........................                                                 [100%]
96 passed in 15.36s
```



Analyzed

```
bash -lc cat > scripts/run_marked_recall_suite.py <<'PY'
from __future__ import annotations


import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from attention_moonshot.candidates import marked_recall_programs
from attention_moonshot.experiment import attention_program_label
from attention_moonshot.model import ModelConfig
from attention_moonshot.synthetic import MarkedRecallConfig, train_marked_recall


SEEDS = (7, 8, 9)




def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)




def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0




def _condition_mean(runs: list[dict[str, Any]], condition: str, metric: str) -> tuple[float, float]:
    values = [float(run["conditions"][condition][metric]) for run in runs]
    return _mean_sd(values)




def _short_label(label: str) -> str:
    q = "quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]"
    r = "quad_route[sh=0,lh=3,bh=0,s=0.8]"
    return label.replace("softmax", "F").replace(q, "Q").replace(r, "R")




def _paired_integrity(rows: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    for mode in sorted({row["mode"] for row in rows}):
        for seed in sorted({int(row["seed"]) for row in rows if row["mode"] == mode}):
            group = [row for row in rows if row["mode"] == mode and int(row["seed"]) == seed]
            if len(group) < 2:
                continue
            init = {row["initialization_digest"] for row in group}
            batches = {row["batch_stream_digest"] for row in group}
            if len(init) != 1:
                problems.append(f"{mode}/seed {seed}: initialization digests differ")
            if len(batches) != 1:
                problems.append(f"{mode}/seed {seed}: batch stream digests differ")
    return problems




def write_report(output: Path) -> None:
    rows = [json.loads(path.read_text()) for path in sorted((output / "raw").glob("*.json"))]
    if not rows:
        return
    problems = _paired_integrity(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["mode"], row["operator"]), []).append(row)


    lines = [
        "# NoPE Marked Retrieval — External v1 Reproduction and Hardened v1.1",
        "",
        f"Completed **{len(rows)}** atomic runs. Every arm uses NoPE, two layers, four heads, identical initialization and batch stream within each mode/seed pair.",
        "",
        "## Pairing audit",
        "",
        "**PASS:** all comparable arms have identical initialization and batch-stream digests."
        if not problems
        else "**FAIL:** " + "; ".join(problems),
    ]


    legacy = [(operator, runs) for (mode, operator), runs in grouped.items() if mode == "legacy"]
    if legacy:
        summary = []
        for operator, runs in legacy:
            in_acc, in_sd = _condition_mean(runs, "in_dist", "accuracy")
            sh_acc, sh_sd = _condition_mean(runs, "shift_20_30", "accuracy")
            sh_bits, _ = _condition_mean(runs, "shift_20_30", "bits")
            speed, speed_sd = _mean_sd([float(run["tokens_per_second"]) for run in runs])
            summary.append((sh_acc, operator, len(runs), in_acc, in_sd, sh_sd, sh_bits, speed, speed_sd))
        summary.sort(reverse=True)
        lines += [
            "",
            "## Exact external Suite v1 semantics",
            "",
            "Target pair fixed at rank 0; clamped ±1..3 distractor keys retained exactly for comparability.",
            "",
            "|#|program|runs|in-dist accuracy|shift 20–30 accuracy|shift bits|tokens/s|",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for rank, item in enumerate(summary, 1):
            sh_acc, operator, count, in_acc, in_sd, sh_sd, sh_bits, speed, speed_sd = item
            lines.append(
                f"|{rank}|`{_short_label(operator)}`|{count}|{in_acc:.4f} ± {in_sd:.4f}|"
                f"{sh_acc:.4f} ± {sh_sd:.4f}|{sh_bits:.4f}|{speed:.0f} ± {speed_sd:.0f}|"
            )
 1 F -> F: softmax -> softmax
 2 Q -> Q: quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]
 3 Q -> R: quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[sh=0,lh=3,bh=0,s=0.8]
 4 R -> Q: quad_route[sh=0,lh=3,bh=0,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]
 5 R -> R: quad_route[sh=0,lh=3,bh=0,s=0.8] -> quad_route[sh=0,lh=3,bh=0,s=0.8]
```



Analyzed

```
bash -lc /usr/bin/time -p python3 scripts/run_marked_recall_suite.py --mode legacy --arm 1 --seed 7 --steps 50 --threads 2 --width 48 --output results/nope_marked_recall_calibration --force
/mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/synthetic.py:517: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:836.)
  step, float(loss / math.log(2)), flush=True,
marked_recall legacy softmax -> softmax 7 1 8.023406028747559
marked_recall legacy softmax -> softmax 7 12 5.9156622886657715
marked_recall legacy softmax -> softmax 7 24 5.1874895095825195
marked_recall legacy softmax -> softmax 7 36 5.153146266937256
marked_recall legacy softmax -> softmax 7 48 4.9785566329956055
marked_recall legacy softmax -> softmax 7 50 4.969729423522949
WROTE /mnt/data/attention-moonshot-nope-v1/results/nope_marked_recall_calibration/raw/legacy-seed7-arm01-fa95773798.json
real 7.48
user 10.53
sys 1.37
```



Analyzed

```
bash -lc /usr/bin/time -p python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 300 --threads 2 --width 48 --output results/nope_marked_recall_calibration --force
/mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/synthetic.py:517: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:836.)
  step, float(loss / math.log(2)), flush=True,
marked_recall hardened softmax -> softmax 7 1 8.021169662475586
marked_recall hardened softmax -> softmax 7 75 5.021873950958252
marked_recall hardened softmax -> softmax 7 150 5.063988208770752


Command failed because it timed out.
```



Profiling Legacy Versus Collision-Free Generation

```
bash -lc ps -ef | grep run_marked_recall | grep -v grep || true
python3 - <<'PY'
import cProfile,pstats,io,time,torch
from attention_moonshot.synthetic import collision_free_marked_recall_batch, legacy_marked_recall_batch


def bench(fn,name):
 g=torch.Generator().manual_seed(7)
 t=time.perf_counter()
 for _ in range(100):
  fn(batch_size=24,context=160,distractor_min=8,distractor_max=14,generator=g,**({'target_mode':'random'} if 'collision' in name else {}))
 print(name,time.perf_counter()-t)
bench(legacy_marked_recall_batch,'legacy')
bench(collision_free_marked_recall_batch,'collision')


pr=cProfile.Profile();g=torch.Generator().manual_seed(7);pr.enable()
for _ in range(20):
 collision_free_marked_recall_batch(batch_size=24,context=160,distractor_min=8,distractor_max=14,target_mode='random',generator=g)
pr.disable();s=io.StringIO();pstats.Stats(pr,stream=s).sort_stats('cumtime').print_stats(20);print(s.getvalue())
PY
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ModuleNotFoundError: No module named 'attention_moonshot'


Command 'bash -lc 'ps -ef | grep run_marked_recall | grep -v grep || true
python3 - <<'"'"'PY'"'"'
import cProfile,pstats,io,time,torch
from attention_moonshot.synthetic import collision_free_marked_recall_batch, legacy_marked_recall_batch

def bench(fn,name):
 g=torch.Generator().manual_seed(7)
 t=time.perf_counter()
 for _ in range(100):
  fn(batch_size=24,context=160,distractor_min=8,distractor_max=14,generator=g,**({'"'"'target_mode'"'"':'"'"'random'"'"'} if '"'"'collision'"'"' in name else {}))
 print(name,time.perf_counter()-t)
bench(legacy_marked_recall_batch,'"'"'legacy'"'"')
bench(collision_free_marked_recall_batch,'"'"'collision'"'"')

pr=cProfile.Profile();g=torch.Generator().manual_seed(7);pr.enable()
for _ in range(20):
 collision_free_marked_recall_batch(batch_size=24,context=160,distractor_min=8,distractor_max=14,target_mode='"'"'random'"'"',generator=g)
pr.disable();s=io.StringIO();pstats.Stats(pr,stream=s).sort_stats('"'"'cumtime'"'"').print_stats(20);print(s.getvalue())
PY'' failed with status 1.
```



Analyzed

```
bash -lc PYTHONPATH=src python3 - <<'PY'
import cProfile,pstats,io,time,torch
from attention_moonshot.synthetic import collision_free_marked_recall_batch, legacy_marked_recall_batch


def bench(fn,name):
 g=torch.Generator().manual_seed(7)
 t=time.perf_counter()
 for _ in range(100):
  kw={'batch_size':24,'context':160,'distractor_min':8,'distractor_max':14,'generator':g}
  if name=='collision': kw['target_mode']='random'
  fn(**kw)
 print(name,time.perf_counter()-t)
bench(legacy_marked_recall_batch,'legacy')
bench(collision_free_marked_recall_batch,'collision')


pr=cProfile.Profile();g=torch.Generator().manual_seed(7);pr.enable()
for _ in range(20):
 collision_free_marked_recall_batch(batch_size=24,context=160,distractor_min=8,distractor_max=14,target_mode='random',generator=g)
pr.disable();s=io.StringIO();pstats.Stats(pr,stream=s).sort_stats('cumtime').print_stats(20);print(s.getvalue())
PY
legacy 0.22888430300008622
collision 0.12327124500006903
         18498 function calls in 0.052 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       20    0.011    0.001    0.052    0.003 /mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/synthetic.py:188(collision_free_marked_recall_batch)
      960    0.001    0.000    0.030    0.000 /opt/pyvenv/lib/python3.13/site-packages/torch/_tensor.py:1203(__iter__)
      960    0.028    0.000    0.028    0.000 {method 'unbind' of 'torch._C.TensorBase' objects}
       20    0.002    0.000    0.004    0.000 /mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/synthetic.py:113(_pack_marked_rows)
      960    0.003    0.000    0.003    0.000 {built-in method torch.randint}
      960    0.002    0.000    0.002    0.000 {built-in method torch.randperm}
      540    0.002    0.000    0.002    0.000 {built-in method torch.tensor}
      480    0.000    0.000    0.001    0.000 /mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/synthetic.py:176(_target_rank)
     6297    0.001    0.000    0.001    0.000 {method 'extend' of 'list' objects}
      960    0.001    0.000    0.001    0.000 {built-in method torch._C._get_tracing_state}
      960    0.000    0.000    0.000    0.000 {method 'item' of 'torch._C.TensorBase' objects}
     1920    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     1480    0.000    0.000    0.000    0.000 {built-in method builtins.len}
      960    0.000    0.000    0.000    0.000 {method 'dim' of 'torch._C.TensorBase' objects}
      960    0.000    0.000    0.000    0.000 {built-in method builtins.iter}
       20    0.000    0.000    0.000    0.000 {built-in method torch.zeros}
       20    0.000    0.000    0.000    0.000 <string>:2(__init__)
       20    0.000    0.000    0.000    0.000 /mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/synthetic.py:99(_validate_marked_recall_request)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```



Analyzed

```
bash -lc PYTHONPATH=src python3 - <<'PY'
import time
from attention_moonshot.synthetic import MarkedRecallConfig,train_marked_recall
from attention_moonshot.model import ModelConfig
from attention_moonshot.candidates import marked_recall_programs
mc=ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode='none')
for mode in ['legacy','hardened']:
 cfg=MarkedRecallConfig(steps=10,batch_size=24,eval_batches=1,eval_batch_size=8,torch_threads=2)
 t=time.perf_counter();r=train_marked_recall(mc,marked_recall_programs()[0],cfg,seed=7,mode=mode);print(mode,time.perf_counter()-t,r['tokens_per_second'])
PY
legacy 6.0828693200001 9441.22001880161
hardened 3.0110496589986724 12813.526394117538
```



Analyzed

```
bash -lc PYTHONPATH=src /usr/bin/time -p python3 - <<'PY'
from attention_moonshot.synthetic import MarkedRecallConfig,train_marked_recall
from attention_moonshot.model import ModelConfig
from attention_moonshot.candidates import marked_recall_programs
mc=ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode='none')
cfg=MarkedRecallConfig(steps=100,batch_size=24,eval_batches=1,eval_batch_size=8,torch_threads=2)
r=train_marked_recall(mc,marked_recall_programs()[0],cfg,seed=7,mode='hardened',verbose=True)
print(r['tokens_per_second'],r['conditions']['shift_random'])
PY
/mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/synthetic.py:517: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:836.)
  step, float(loss / math.log(2)), flush=True,
marked_recall hardened softmax -> softmax 7 1 8.021169662475586
marked_recall hardened softmax -> softmax 7 25 5.165452003479004
marked_recall hardened softmax -> softmax 7 50 5.204441070556641
marked_recall hardened softmax -> softmax 7 75 5.021873950958252
marked_recall hardened softmax -> softmax 7 100 4.964351654052734
32385.562642982182 {'accuracy': 0.125, 'bits': 5.13060295087379, 'count': 8, 'rank_accuracy': {'early': 0.0, 'middle': 0.25, 'late': 0.0}}
real 15.51
user 19.69
sys 1.28
```



Analyzed

```
bash -lc set -e
mkdir -p results/nope_marked_recall_calibration/logs
(
  python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --threads 2 --width 48 --output results/nope_marked_recall_calibration --force > results/nope_marked_recall_calibration/logs/hard-f-600.log 2>&1
) & p1=$!
(
  python3 scripts/run_marked_recall_suite.py --mode hardened --arm 2 --seed 7 --steps 600 --threads 2 --width 48 --output results/nope_marked_recall_calibration --force > results/nope_marked_recall_calibration/logs/hard-q-600.log 2>&1
) & p2=$!
wait $p1
wait $p2
cat results/nope_marked_recall_calibration/logs/hard-f-600.log
cat results/nope_marked_recall_calibration/logs/hard-q-600.log
python3 - <<'PY'
import json,glob
for p in sorted(glob.glob('results/nope_marked_recall_calibration/raw/hardened*.json')):
 r=json.load(open(p));print(r['short_program'],r['conditions']['in_dist_random']['accuracy'],r['conditions']['shift_random']['accuracy'],r['conditions']['counterfactual']['both_correct'],r['tokens_per_second'])
PY
Command failed because it timed out.
```



Analyzed

```
bash -lc ps -ef | grep run_marked_recall | grep -v grep || true; tail -30 results/nope_marked_recall_calibration/logs/hard-f-600.log 2>/dev/null; echo ---; tail -30 results/nope_marked_recall_calibration/logs/hard-q-600.log 2>/dev/null; ls -l results/nope_marked_recall_calibration/raw | tail
/mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/synthetic.py:517: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:836.)
  step, float(loss / math.log(2)), flush=True,
marked_recall hardened softmax -> softmax 7 1 8.021169662475586
---
/mnt/data/attention-moonshot-nope-v1/src/attention_moonshot/synthetic.py:517: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:836.)
  step, float(loss / math.log(2)), flush=True,
marked_recall hardened quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] 7 1 8.037687301635742
total 4
-rw-r--r-- 1 root oai_shared 2181 Aug 20 06:15 legacy-seed7-arm01-fa95773798.json
```



Analyzed

```
bash -lc for th in 1 2 4; do echo THREADS=$th; PYTHONPATH=src /usr/bin/time -p python3 - <<PY
from attention_moonshot.synthetic import MarkedRecallConfig,train_marked_recall
from attention_moonshot.model import ModelConfig
from attention_moonshot.candidates import marked_recall_programs
mc=ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode='none')
cfg=MarkedRecallConfig(steps=50,batch_size=24,eval_batches=1,eval_batch_size=8,torch_threads=$th)
r=train_marked_recall(mc,marked_recall_programs()[0],cfg,seed=7,mode='hardened')
print(r['tokens_per_second'])
PY
done
THREADS=1
44939.934887517666
real 7.44
user 7.06
sys 0.83
THREADS=2
58911.65164061828
real 6.70
user 9.54
sys 0.65
THREADS=4
25955.288345040124
real 10.81
user 20.97
sys 3.15
```



Analyzed

```
bash -lc rm -rf /tmp/parallelbench; mkdir -p /tmp/parallelbench
/usr/bin/time -p bash -c '
PYTHONPATH=src python3 - <<"PY" > /tmp/parallelbench/a 2>&1 &
from attention_moonshot.synthetic import MarkedRecallConfig,train_marked_recall
from attention_moonshot.model import ModelConfig
from attention_moonshot.candidates import marked_recall_programs
r=train_marked_recall(ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode="none"),marked_recall_programs()[0],MarkedRecallConfig(steps=50,batch_size=24,eval_batches=1,eval_batch_size=8,torch_threads=1),seed=7,mode="hardened")
print(r["tokens_per_second"])
PY
p1=$!
PYTHONPATH=src python3 - <<"PY" > /tmp/parallelbench/b 2>&1 &
from attention_moonshot.synthetic import MarkedRecallConfig,train_marked_recall
from attention_moonshot.model import ModelConfig
from attention_moonshot.candidates import marked_recall_programs
r=train_marked_recall(ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode="none"),marked_recall_programs()[1],MarkedRecallConfig(steps=50,batch_size=24,eval_batches=1,eval_batch_size=8,torch_threads=1),seed=7,mode="hardened")
print(r["tokens_per_second"])
PY
p2=$!
wait $p1; wait $p2
'
cat /tmp/parallelbench/a /tmp/parallelbench/b
real 26.55
user 30.42
sys 1.81
11165.923743568057
9049.955612159254
```



Analyzed

```
bash -lc python3 - <<'PY'
from pathlib import Path
p=Path('src/attention_moonshot/synthetic.py')
s=p.read_text().replace('step, float(loss / math.log(2)), flush=True,','step, float(loss.detach() / math.log(2)), flush=True,')
p.write_text(s)
PY
/usr/bin/time -p python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --threads 2 --width 48 --output results/nope_marked_recall_calibration --force
python3 - <<'PY'
import json,glob
p=sorted(glob.glob('results/nope_marked_recall_calibration/raw/hardened-seed7-arm01-*.json'))[-1]
r=json.load(open(p));print(r['short_program'],r['conditions'])
PY
marked_recall hardened softmax -> softmax 7 1 8.021169662475586


Command failed because it timed out.
```



Analyzed

```
bash -lc PYTHONPATH=src python3 -u - <<'PY'
import os,time,resource,torch,math
from attention_moonshot.synthetic import collision_free_marked_recall_batch
from attention_moonshot.model import ModelConfig,ByteGPT
from attention_moonshot.candidates import marked_recall_programs
from attention_moonshot.experiment import seed_all
seed_all(7,2)
mc=ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode='none')
model=ByteGPT(mc,marked_recall_programs()[0]);opt=torch.optim.AdamW(model.parameters(),lr=3e-3)
g=torch.Generator().manual_seed(3007)
t0=last=time.perf_counter()
for step in range(1,301):
 b=collision_free_marked_recall_batch(batch_size=24,context=160,distractor_min=8,distractor_max=14,target_mode='random',generator=g)
 opt.zero_grad(set_to_none=True);z=model(b.tokens)[:,-1];loss=torch.nn.functional.cross_entropy(z,b.targets);loss.backward();opt.step()
 if step%10==0:
  now=time.perf_counter();print(step,'chunk',now-last,'total',now-t0,'loss',loss.item(),'rssMB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,flush=True);last=now
PY
10 chunk 13.998081407000427 total 13.998081407000427 loss 4.337468147277832 rssMB 518.8046875
20 chunk 4.58413537999877 total 18.582216786999197 loss 3.6876118183135986 rssMB 526.68359375
30 chunk 1.1107197279998218 total 19.69293651499902 loss 3.5470073223114014 rssMB 534.81640625
40 chunk 0.9604526840012113 total 20.65338919900023 loss 3.524030923843384 rssMB 544.12890625
50 chunk 1.0113540009988355 total 21.664743199999066 loss 3.6056911945343018 rssMB 544.12890625
60 chunk 0.9844719180000538 total 22.64921511799912 loss 3.4694411754608154 rssMB 544.12890625
70 chunk 0.8668249570000626 total 23.516040074999182 loss 3.4498918056488037 rssMB 544.12890625
80 chunk 0.7810099279995484 total 24.29705000299873 loss 3.408740758895874 rssMB 544.12890625
90 chunk 0.6994485210016137 total 24.996498524000344 loss 3.498499631881714 rssMB 544.12890625
100 chunk 0.5996508059997723 total 25.596149330000117 loss 3.455610990524292 rssMB 544.12890625


Command failed because it timed out.
```



Analyzed

```
bash -lc PYTHONPATH=src python3 -u - <<'PY'
import time,torch
from attention_moonshot.synthetic import collision_free_marked_recall_batch
g=torch.Generator().manual_seed(3007)
last=time.perf_counter()
for step in range(1,501):
 b=collision_free_marked_recall_batch(batch_size=24,context=160,distractor_min=8,distractor_max=14,target_mode='random',generator=g)
 if step%10==0:
  now=time.perf_counter();print(step,now-last);last=now
PY
10 0.01733037600024545
20 0.014870078999592806
30 0.011673183000311838
40 0.013952706000054604
50 0.011704262999046477
60 0.014183744000547449
70 0.015946602999974857
80 0.014881294999213424
90 0.014659272001154022
100 0.013137371999619063
110 0.012247446000401396
120 0.014160611999614048
130 0.016356475000065984
140 0.021154760999706923
150 0.01342871499946341
160 0.018200554000941338
170 0.019944842999393586
180 0.013832402999469195
190 0.018913183001131983
200 0.017473605999839492
210 0.013751548000072944
220 0.02051931299865828
230 0.015698198001700803
240 0.013566362998972181
250 0.01337661300021864
260 0.015688904000853654
270 0.012205067998365848
280 0.014055386000109138
290 0.012330164001468802
300 0.012331819998507854
310 0.014536202999806846
320 0.012121733001549728
330 0.011925943999813171
340 0.010558345999015728
350 0.014200132000041776
360 0.013751602000411367
370 0.010182289999647764
380 0.011175229999935254
390 0.019912024999939604
400 0.022173330000441638
410 0.012173029999758
420 0.01145736400030728
430 0.012152892999438336
440 0.013392793000093661
450 0.00972985000043991
460 0.012361925000732299
470 0.012273873999220086
480 0.015364565000709263
490 0.013704773999052122
500 0.012983923999854596
```



Analyzed

```
bash -lc PYTHONPATH=src python3 -u - <<'PY'
import time,torch
from attention_moonshot.synthetic import collision_free_marked_recall_batch
from attention_moonshot.model import ModelConfig,ByteGPT
from attention_moonshot.candidates import marked_recall_programs
from attention_moonshot.experiment import seed_all
seed_all(7,2);mc=ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode='none');model=ByteGPT(mc,marked_recall_programs()[0]);opt=torch.optim.AdamW(model.parameters(),lr=3e-3);g=torch.Generator().manual_seed(3007)
for step in range(1,106):
 if step>=98: print(step,'gen-start',flush=True)
 b=collision_free_marked_recall_batch(batch_size=24,context=160,distractor_min=8,distractor_max=14,target_mode='random',generator=g)
 if step>=98: print(step,'zero',flush=True)
 opt.zero_grad(set_to_none=True)
 if step>=98: print(step,'forward',flush=True)
 z=model(b.tokens)[:,-1]
 if step>=98: print(step,'loss',flush=True)
 loss=torch.nn.functional.cross_entropy(z,b.targets)
 if step>=98: print(step,'backward',loss.item(),flush=True)
 loss.backward()
 if step>=98: print(step,'step',flush=True)
 opt.step()
 if step>=98: print(step,'done',flush=True)
PY
98 gen-start
98 zero
98 forward
98 loss
98 backward 3.498006582260132
98 step
98 done
99 gen-start
99 zero
99 forward
99 loss
99 backward 3.44966197013855
99 step
99 done
100 gen-start
100 zero
100 forward
100 loss
100 backward 3.455610990524292
100 step
100 done
101 gen-start
101 zero
101 forward
101 loss
101 backward 3.6440303325653076
101 step
101 done
102 gen-start
102 zero
102 forward
102 loss
102 backward 3.4919164180755615
102 step
102 done
103 gen-start
103 zero
103 forward
103 loss
103 backward 3.5650112628936768
103 step
103 done
104 gen-start
104 zero
104 forward
104 loss
104 backward 3.556170701980591
104 step
104 done
105 gen-start
105 zero
105 forward
105 loss
105 backward 3.530378580093384
105 step
105 done
```



Analyzed

```
bash -lc PYTHONPATH=src python3 -u - <<'PY'
import time,resource,torch
from attention_moonshot.synthetic import collision_free_marked_recall_batch
from attention_moonshot.model import ModelConfig,ByteGPT
from attention_moonshot.candidates import marked_recall_programs
from attention_moonshot.experiment import seed_all
seed_all(7,2);mc=ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode='none');model=ByteGPT(mc,marked_recall_programs()[0]);opt=torch.optim.AdamW(model.parameters(),lr=3e-3);g=torch.Generator().manual_seed(3007)
t0=last=time.perf_counter()
for step in range(1,301):
 b=collision_free_marked_recall_batch(batch_size=24,context=160,distractor_min=8,distractor_max=14,target_mode='random',generator=g)
 opt.zero_grad(set_to_none=True);z=model(b.tokens)[:,-1];loss=torch.nn.functional.cross_entropy(z,b.targets);loss.backward();opt.step()
 if step%25==0:
  now=time.perf_counter();print(step,now-last,now-t0,loss.item(),flush=True);last=now
PY
25 3.362858401000267 3.362858401000267 3.589797258377075
50 1.5246758129997033 4.88753421399997 3.6056911945343018
75 1.5410504810006387 6.428584695000609 3.4861514568328857
100 1.473844005000501 7.90242870000111 3.455610990524292
125 1.4217181899985007 9.32414688999961 3.565643310546875
150 1.334227721001298 10.658374611000909 3.5277280807495117
175 3.1967641779992846 13.855138789000193 3.5164995193481445
200 5.7561065909994795 19.611245379999673 3.5245840549468994
225 2.430144937001387 22.04139031700106 3.4831669330596924
250 1.6599867019995145 23.701377019000574 3.4872570037841797
275 1.3043021430003137 25.005679162000888 3.4834940433502197
300 1.3400547679993906 26.34573393000028 3.475923776626587
```



Analyzed

```
bash -lc /usr/bin/time -p python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --threads 2 --width 48 --output results/nope_marked_recall_calibration --force
python3 - <<'PY'
import json,glob
p=sorted(glob.glob('results/nope_marked_recall_calibration/raw/hardened-seed7-arm01-*.json'))[-1]
r=json.load(open(p));print(r['short_program']);
for k,v in r['conditions'].items(): print(k,v)
PY
marked_recall hardened softmax -> softmax 7 1 8.021169662475586
marked_recall hardened softmax -> softmax 7 150 5.063988208770752
marked_recall hardened softmax -> softmax 7 300 5.011213779449463


Command failed because it timed out.
```



Analyzed

```
bash -lc PYTHONPATH=src python3 -u - <<'PY'
import time,torch,hashlib,math
from torch.nn.utils import clip_grad_norm_
from attention_moonshot.synthetic import collision_free_marked_recall_batch,_tensor_digest
from attention_moonshot.model import ModelConfig,ByteGPT
from attention_moonshot.candidates import marked_recall_programs
from attention_moonshot.experiment import seed_all
seed_all(7,2);mc=ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode='none');model=ByteGPT(mc,marked_recall_programs()[0]);opt=torch.optim.AdamW(model.parameters(),lr=3e-3);g=torch.Generator().manual_seed(3007);h=hashlib.sha256()
t0=last=time.perf_counter()
for step in range(1,601):
 b=collision_free_marked_recall_batch(batch_size=24,context=160,distractor_min=8,distractor_max=14,target_mode='random',generator=g)
 _tensor_digest(h,b.tokens);_tensor_digest(h,b.targets)
 opt.zero_grad(set_to_none=True);z=model(b.tokens)[:,-1];loss=torch.nn.functional.cross_entropy(z,b.targets);loss.backward();clip_grad_norm_(model.parameters(),1.0);opt.step()
 if step%50==0:
  now=time.perf_counter();print(step,now-last,now-t0,loss.item(),flush=True);last=now
PY
50 5.0228862010008015 5.0228862010008015 3.607419967651367
100 3.941757377999238 8.96464357900004 3.441190481185913
150 2.7398122289996536 11.704455807999693 3.5102145671844482
200 2.756421322999813 14.460877130999506 3.5131618976593018
250 3.5215717789997143 17.98244890999922 3.489500045776367
300 3.5954168180014676 21.57786572800069 3.472109794616699
350 6.076449845999377 27.654315574000066 3.5328989028930664


Command failed because it timed out.
```



Analyzed

```
bash -lc cat >> tests/test_marked_retrieval.py <<'PY'




def test_marked_recall_checkpoint_resume_matches_uninterrupted_training() -> None:
    from attention_moonshot.candidates import marked_recall_programs
    from attention_moonshot.model import ModelConfig


    advance = getattr(synthetic, "advance_marked_recall_checkpoint", None)
    assert advance is not None, "resumable marked-recall trainer is missing"
    mc = ModelConfig(
        vocab_size=256,
        context=32,
        d_model=16,
        n_heads=4,
        n_layers=2,
        position_mode="none",
    )
    cfg = synthetic.MarkedRecallConfig(
        steps=4,
        batch_size=2,
        eval_batches=1,
        eval_batch_size=2,
        train_distractor_min=1,
        train_distractor_max=2,
        shift_distractor_min=3,
        shift_distractor_max=4,
        torch_threads=1,
    )
    program = marked_recall_programs()[1]
    uninterrupted = advance(
        mc, program, cfg, seed=41, mode="hardened", checkpoint=None, segment_steps=4
    )
    first_half = advance(
        mc, program, cfg, seed=41, mode="hardened", checkpoint=None, segment_steps=2
    )
    resumed = advance(
        mc, program, cfg, seed=41, mode="hardened", checkpoint=first_half, segment_steps=2
    )
    assert uninterrupted["step"] == resumed["step"] == 4
    assert uninterrupted["batch_stream_digest"] == resumed["batch_stream_digest"]
    for name, tensor in uninterrupted["model_state"].items():
        torch.testing.assert_close(tensor, resumed["model_state"][name], atol=0, rtol=0)
PY
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_marked_retrieval.py::test_marked_recall_checkpoint_resume_matches_uninterrupted_training -q
F                                                                        [100%]
=================================== FAILURES ===================================
_____ test_marked_recall_checkpoint_resume_matches_uninterrupted_training ______

    def test_marked_recall_checkpoint_resume_matches_uninterrupted_training() -> None:
        from attention_moonshot.candidates import marked_recall_programs
        from attention_moonshot.model import ModelConfig
    
        advance = getattr(synthetic, "advance_marked_recall_checkpoint", None)
>       assert advance is not None, "resumable marked-recall trainer is missing"
E       AssertionError: resumable marked-recall trainer is missing
E       assert None is not None

tests/test_marked_retrieval.py:239: AssertionError
=========================== short test summary info ============================
FAILED tests/test_marked_retrieval.py::test_marked_recall_checkpoint_resume_matches_uninterrupted_training - AssertionError: resumable marked-recall trainer is missing
assert None is not None
1 failed in 2.79s


Command 'bash -lc 'cat >> tests/test_marked_retrieval.py <<'"'"'PY'"'"'


def test_marked_recall_checkpoint_resume_matches_uninterrupted_training() -> None:
    from attention_moonshot.candidates import marked_recall_programs
    from attention_moonshot.model import ModelConfig

    advance = getattr(synthetic, "advance_marked_recall_checkpoint", None)
    assert advance is not None, "resumable marked-recall trainer is missing"
    mc = ModelConfig(
        vocab_size=256,
        context=32,
        d_model=16,
        n_heads=4,
        n_layers=2,
        position_mode="none",
    )
    cfg = synthetic.MarkedRecallConfig(
        steps=4,
        batch_size=2,
        eval_batches=1,
        eval_batch_size=2,
        train_distractor_min=1,
        train_distractor_max=2,
        shift_distractor_min=3,
        shift_distractor_max=4,
        torch_threads=1,
    )
    program = marked_recall_programs()[1]
    uninterrupted = advance(
        mc, program, cfg, seed=41, mode="hardened", checkpoint=None, segment_steps=4
    )
    first_half = advance(
        mc, program, cfg, seed=41, mode="hardened", checkpoint=None, segment_steps=2
    )
    resumed = advance(
        mc, program, cfg, seed=41, mode="hardened", checkpoint=first_half, segment_steps=2
    )
    assert uninterrupted["step"] == resumed["step"] == 4
    assert uninterrupted["batch_stream_digest"] == resumed["batch_stream_digest"]
    for name, tensor in uninterrupted["model_state"].items():
        torch.testing.assert_close(tensor, resumed["model_state"][name], atol=0, rtol=0)
PY
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_marked_retrieval.py::test_marked_recall_checkpoint_resume_matches_uninterrupted_training -q'' failed with status 1.
```



Analyzed

```
bash -lc python3 - <<'PY'
from pathlib import Path
p=Path('src/attention_moonshot/synthetic.py')
s=p.read_text()
anchor='def train_marked_recall(\n'
insert=r'''


def _chained_batch_digest(
    previous: str,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> str:
    hasher = hashlib.sha256(bytes.fromhex(previous))
    _tensor_digest(hasher, tokens)
    _tensor_digest(hasher, targets)
    return hasher.hexdigest()




def advance_marked_recall_checkpoint(
    mc: ModelConfig,
    spec: AttentionProgram,
    cfg: MarkedRecallConfig,
    *,
    seed: int,
    mode: str,
    checkpoint: dict[str, Any] | None,
    segment_steps: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Advance a marked-recall training run without changing its RNG stream."""
    if segment_steps < 1:
        raise ValueError("segment_steps must be positive")
    if mc.position_mode != "none":
        raise ValueError("marked retrieval requires position_mode='none'")
    if mode not in {"legacy", "hardened"}:
        raise ValueError("mode must be 'legacy' or 'hardened'")
    _validate_marked_recall_request(
        batch_size=cfg.batch_size,
        context=mc.context,
        distractor_min=cfg.train_distractor_min,
        distractor_max=max(cfg.train_distractor_max, cfg.shift_distractor_max),
    )
    seed_all(seed, cfg.torch_threads)
    model = ByteGPT(mc, spec)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator().manual_seed(3000 + seed)
    label = attention_program_label(spec)
    initial_digest = _model_initialization_digest(model)
    current_step = 0
    stream_digest = "00" * 32
    elapsed = 0.0
    if checkpoint is not None:
        if checkpoint["operator"] != label or checkpoint["mode"] != mode:
            raise ValueError("checkpoint operator or mode mismatch")
        if int(checkpoint["seed"]) != seed:
            raise ValueError("checkpoint seed mismatch")
        if checkpoint["model_config"] != asdict(mc) or checkpoint["train_config"] != asdict(cfg):
            raise ValueError("checkpoint configuration mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        generator.set_state(checkpoint["generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        current_step = int(checkpoint["step"])
        stream_digest = str(checkpoint["batch_stream_digest"])
        initial_digest = str(checkpoint["initialization_digest"])
        elapsed = float(checkpoint.get("elapsed_training_s", 0.0))


    end_step = min(cfg.steps, current_step + segment_steps)
    started = time.perf_counter()
    model.train()
    last_loss_bits = float("nan")
    for step in range(current_step + 1, end_step + 1):
        batch = _make_marked_batch(
            mode=mode,
            batch_size=cfg.batch_size,
            context=mc.context,
            distractor_min=cfg.train_distractor_min,
            distractor_max=cfg.train_distractor_max,
            target_mode="first" if mode == "legacy" else "random",
            generator=generator,
        )
        stream_digest = _chained_batch_digest(stream_digest, batch.tokens, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.tokens)[:, -1]
        loss = torch.nn.functional.cross_entropy(logits, batch.targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        last_loss_bits = float(loss.detach() / math.log(2))
        if verbose and (
            step == current_step + 1
            or step == end_step
            or step % max(1, segment_steps // 2) == 0
        ):
            print("marked_segment", mode, label, seed, step, last_loss_bits, flush=True)
    elapsed += time.perf_counter() - started
    return {
        "version": 1,
        "task": "marked_recall",
        "mode": mode,
        "operator": label,
.                                                                        [100%]
1 passed in 6.39s
```



Analyzed

```
bash -lc python3 - <<'PY'
from pathlib import Path
p=Path('scripts/run_marked_recall_suite.py')
s=p.read_text()
s=s.replace('import statistics\nimport sys\n', 'import statistics\nimport sys\n\nimport torch\n')
s=s.replace('from attention_moonshot.synthetic import MarkedRecallConfig, train_marked_recall\n', 'from attention_moonshot.synthetic import (\n    MarkedRecallConfig,\n    advance_marked_recall_checkpoint,\n    finalize_marked_recall_checkpoint,\n)\n')
s=s.replace('    parser.add_argument("--output", default="results/nope_marked_recall")\n', '    parser.add_argument("--output", default="results/nope_marked_recall")\n    parser.add_argument("--segment-steps", type=int, default=250)\n')
old='''    if path.exists() and not args.force:
        print(f"SKIP {path.name}")
        write_report(output)
        return


    result = train_marked_recall(
        ModelConfig(
            vocab_size=256,
            context=160,
            d_model=args.width,
            n_heads=4,
            n_layers=2,
            position_mode="none",
        ),
        program,
        MarkedRecallConfig(
            steps=args.steps,
            batch_size=24,
            learning_rate=3e-3,
            eval_batches=8,
            eval_batch_size=48,
            train_distractor_min=8,
            train_distractor_max=14,
            shift_distractor_min=20,
            shift_distractor_max=30,
            torch_threads=args.threads,
        ),
        seed=args.seed,
        mode=args.mode,
        verbose=True,
    )
    result["arm"] = args.arm
    result["short_program"] = _short_label(label)
    _atomic_json(path, result)
    write_report(output)
    print(f"WROTE {path}")
'''
new='''    checkpoint_path = output / "checkpoints" / f"{args.mode}-seed{args.seed}-arm{args.arm:02d}-{digest}.pt"
    if args.force:
        path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
    if path.exists():
        print(f"SKIP {path.name}")
        write_report(output)
        return


    model_config = ModelConfig(
        vocab_size=256,
        context=160,
        d_model=args.width,
        n_heads=4,
        n_layers=2,
        position_mode="none",
    )
    train_config = MarkedRecallConfig(
        steps=args.steps,
        batch_size=24,
        learning_rate=3e-3,
        eval_batches=8,
        eval_batch_size=48,
        train_distractor_min=8,
        train_distractor_max=14,
        shift_distractor_min=20,
        shift_distractor_max=30,
        torch_threads=args.threads,
    )
    checkpoint = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = advance_marked_recall_checkpoint(
        model_config,
        program,
        train_config,
        seed=args.seed,
        mode=args.mode,
        checkpoint=checkpoint,
        segment_steps=args.segment_steps,
        verbose=True,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(checkpoint_path)
    if int(state["step"]) < args.steps:
        print(f"CHECKPOINT {checkpoint_path} step={state['step']}/{args.steps}")
        return


    result = finalize_marked_recall_checkpoint(
        model_config, program, train_config, checkpoint=state
    )
    result["arm"] = args.arm
marked_segment hardened softmax -> softmax 7 1 8.031614303588867
marked_segment hardened softmax -> softmax 7 2 7.946512699127197
marked_segment hardened softmax -> softmax 7 4 7.7642645835876465
marked_segment hardened softmax -> softmax 7 5 7.777144432067871
CHECKPOINT /mnt/data/attention-moonshot-nope-v1/results/nope_marked_recall_chunk_smoke/checkpoints/hardened-seed7-arm01-fa95773798.pt step=5/10
marked_segment hardened softmax -> softmax 7 6 7.637188911437988
marked_segment hardened softmax -> softmax 7 8 7.428893089294434
marked_segment hardened softmax -> softmax 7 10 7.290082931518555
WROTE /mnt/data/attention-moonshot-nope-v1/results/nope_marked_recall_chunk_smoke/raw/hardened-seed7-arm01-fa95773798.json
results/nope_marked_recall_chunk_smoke/checkpoints:
total 216
-rw-r--r-- 1 root oai_shared 218101 Aug 20 06:27 hardened-seed7-arm01-fa95773798.pt

results/nope_marked_recall_chunk_smoke/raw:
total 4
-rw-r--r-- 1 root oai_shared 3150 Aug 20 06:27 hardened-seed7-arm01-fa95773798.json
# NoPE Marked Retrieval — External v1 Reproduction and Hardened v1.1

Completed **1** atomic runs. Every arm uses NoPE, two layers, four heads, identical initialization and batch stream within each mode/seed pair.

## Pairing audit

**PASS:** all comparable arms have identical initialization and batch-stream digests.

## Hardened v1.1: unique table, randomized query rank

Every table uses unique keys and values. Training queries a random table row. Counterfactual evaluation changes only the final query key while keeping the entire table fixed.

|#|program|runs|shift random|first|middle|last|counterfactual both|query-change|shift bits|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|`F -> F`|1|0.0339 ± 0.0000|0.0286|0.0417|0.0469|0.0000 ± 0.0000|0.4818|7.2213|

## Hardened rank bins

### `F -> F`

|seed|early|middle|late|
|---:|---:|---:|---:|
|7|0.0222|0.0388|0.0411|
```



Analyzed

```
bash -lc set -e
out=results/nope_marked_recall_calibration_chunked
rm -rf "$out"
for i in 1 2 3 4; do
  python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 200 --threads 2 --width 48 --output "$out"
  test -f "$out"/raw/hardened-seed7-arm01-fa95773798.json && break
done
python3 - <<'PY'
import json,glob
p=glob.glob('results/nope_marked_recall_calibration_chunked/raw/*.json')[0]
r=json.load(open(p));print(r['short_program']);
for k,v in r['conditions'].items():print(k,v)
print('speed',r['tokens_per_second'])
PY
marked_segment hardened softmax -> softmax 7 1 8.021169662475586
marked_segment hardened softmax -> softmax 7 100 4.964351654052734
marked_segment hardened softmax -> softmax 7 200 5.068321704864502
CHECKPOINT /mnt/data/attention-moonshot-nope-v1/results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt step=200/600
marked_segment hardened softmax -> softmax 7 201 4.9805121421813965
marked_segment hardened softmax -> softmax 7 300 5.011213779449463


Command failed because it timed out.
```



Analyzed

```
bash -lc python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 100 --threads 2 --width 48 --output results/nope_marked_recall_calibration_chunked
python3 - <<'PY'
import torch
p='results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt'
x=torch.load(p,weights_only=False);print('step',x['step'],'loss',x['last_loss_bits'])
PY
marked_segment hardened softmax -> softmax 7 201 4.9805121421813965
marked_segment hardened softmax -> softmax 7 250 5.034108638763428
marked_segment hardened softmax -> softmax 7 300 5.011213779449463
CHECKPOINT /mnt/data/attention-moonshot-nope-v1/results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt step=300/600
step 300 loss 5.011213779449463
```



Analyzed

```
bash -lc for mode in false true; do echo FLUSH=$mode; PYTHONPATH=src /usr/bin/time -p python3 -u - <<PY
import time,torch
from attention_moonshot.synthetic import MarkedRecallConfig,advance_marked_recall_checkpoint
from attention_moonshot.model import ModelConfig
from attention_moonshot.candidates import marked_recall_programs
torch.set_flush_denormal(${mode^})
ck=torch.load('results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt',weights_only=False)
mc=ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode='none')
cfg=MarkedRecallConfig(steps=600,batch_size=24,learning_rate=3e-3,eval_batches=8,eval_batch_size=48,train_distractor_min=8,train_distractor_max=14,shift_distractor_min=20,shift_distractor_max=30,torch_threads=2)
t=time.perf_counter();x=advance_marked_recall_checkpoint(mc,marked_recall_programs()[0],cfg,seed=7,mode='hardened',checkpoint=ck,segment_steps=20,verbose=True);print('elapsed',time.perf_counter()-t,x['step'])
PY
done
FLUSH=false
marked_segment hardened softmax -> softmax 7 301 4.998706817626953
marked_segment hardened softmax -> softmax 7 310 5.098697185516357
marked_segment hardened softmax -> softmax 7 320 5.019438743591309
elapsed 1.7459088780015009 320
real 4.13
user 5.08
sys 0.45
FLUSH=true
marked_segment hardened softmax -> softmax 7 301 4.998706817626953
marked_segment hardened softmax -> softmax 7 310 5.098697185516357
marked_segment hardened softmax -> softmax 7 320 5.019438743591309
elapsed 3.4372481099999277 320
real 5.62
user 7.69
sys 0.62
```



Analyzed

```
bash -lc set -e
out=results/nope_marked_recall_calibration_chunked
for i in 1 2 3 4; do
  timeout 180s python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 100 --threads 2 --width 48 --output "$out" || true
  test -f "$out"/raw/hardened-seed7-arm01-fa95773798.json && break
done
python3 - <<'PY'
import json,glob
p=glob.glob('results/nope_marked_recall_calibration_chunked/raw/*.json')[0]
r=json.load(open(p));print(r['short_program']);
for k,v in r['conditions'].items():print(k,v)
print('speed',r['tokens_per_second'])
PY
Command failed due to container ClientError.
```



Analyzed

```
bash -lc ps -ef | grep run_marked_recall | grep -v grep || true; python3 - <<'PY'
import torch,glob,os,json
for p in glob.glob('results/nope_marked_recall_calibration_chunked/checkpoints/*.pt'):
 x=torch.load(p,weights_only=False);print(p,x['step'])
for p in glob.glob('results/nope_marked_recall_calibration_chunked/raw/*.json'):
 print('raw',p)
PY
root       37902       1  0 06:30 ?        00:00:00 timeout 180s python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 100 --threads 2 --width 48 --output results/nope_marked_recall_calibration_chunked
root       37903   37902 72 06:30 ?        00:00:36 python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 100 --threads 2 --width 48 --output results/nope_marked_recall_calibration_chunked
results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt 300
```



Analyzed

```
bash -lc while ps -p 37903 >/dev/null 2>&1; do sleep 5; done; echo done; python3 - <<'PY'
import torch,glob,os,json
for p in glob.glob('results/nope_marked_recall_calibration_chunked/checkpoints/*.pt'):
 x=torch.load(p,weights_only=False);print(p,x['step'])
for p in glob.glob('results/nope_marked_recall_calibration_chunked/raw/*.json'):
 print('raw',p)
PY
done
results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt 400
```



Analyzed

```
bash -lc ps -ef | grep -E 'run_marked_recall|nope_marked' | grep -v grep || true
```



Analyzed

```
bash -lc timeout 180s python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 100 --threads 2 --width 48 --output results/nope_marked_recall_calibration_chunked; echo rc=$?
Command failed due to container ClientError.
```



Analyzed

```
bash -lc ps -ef | grep run_marked_recall | grep -v grep || true
root       38100       1  0 06:32 ?        00:00:00 timeout 180s python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 100 --threads 2 --width 48 --output results/nope_marked_recall_calibration_chunked
root       38101   38100 66 06:32 ?        00:00:35 python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 100 --threads 2 --width 48 --output results/nope_marked_recall_calibration_chunked
```



Analyzed

```
bash -lc while ps -p 38101 >/dev/null 2>&1; do sleep 5; done; python3 - <<'PY'
import torch,glob
for p in glob.glob('results/nope_marked_recall_calibration_chunked/checkpoints/*.pt'):
 x=torch.load(p,weights_only=False);print(x['step'])
PY
500
```



Analyzed

```
bash -lc nohup timeout 300s python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 100 --threads 2 --width 48 --output results/nope_marked_recall_calibration_chunked > /tmp/mr-final.log 2>&1 & echo $!
38305
```



Analyzed

```
bash -lc for i in $(seq 1 60); do if ! ps -p 38305 >/dev/null 2>&1; then break; fi; sleep 5; done; ps -p 38305 -o pid,stat,etime,cmd || true; cat /tmp/mr-final.log; python3 - <<'PY'
import glob,json,torch,os
print('raw',glob.glob('results/nope_marked_recall_calibration_chunked/raw/*.json'))
for p in glob.glob('results/nope_marked_recall_calibration_chunked/checkpoints/*.pt'):
 x=torch.load(p,weights_only=False);print('ckpt',x['step'])
for p in glob.glob('results/nope_marked_recall_calibration_chunked/raw/*.json'):
 r=json.load(open(p));print(r['short_program'],r['conditions'])
PY
Command failed because it timed out.
```



Analyzed

```
bash -lc ps -p 38305 -o pid,ppid,stat,etime,cmd || true; ps -ef | grep run_marked_recall | grep -v grep || true; tail -50 /tmp/mr-final.log; python3 - <<'PY'
import glob,json,torch,os
print('raw',glob.glob('results/nope_marked_recall_calibration_chunked/raw/*.json'))
for p in glob.glob('results/nope_marked_recall_calibration_chunked/checkpoints/*.pt'):
 x=torch.load(p,weights_only=False);print('ckpt',x['step'])
PY
    PID    PPID STAT     ELAPSED CMD
  38305       1 S          00:54 timeout 300s python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 100 --threads 2 --width 48 --output results/nope_marked_recall_calibration_chunked
root       38305       1  0 06:33 ?        00:00:00 timeout 300s python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 100 --threads 2 --width 48 --output results/nope_marked_recall_calibration_chunked
root       38306   38305 79 06:33 ?        00:00:42 python3 scripts/run_marked_recall_suite.py --mode hardened --arm 1 --seed 7 --steps 600 --segment-steps 100 --threads 2 --width 48 --output results/nope_marked_recall_calibration_chunked
marked_segment hardened softmax -> softmax 7 501 5.182204723358154
raw []
ckpt 500
```



Analyzed

```
bash -lc kill 38305 38306 2>/dev/null || true; sleep 1; ps -ef | grep run_marked_recall | grep -v grep || true
```



Analyzed

```
bash -lc for flush in 0 1; do echo FLUSH=$flush; PYTHONPATH=src python3 -u - <<PY
import time,torch
from torch.nn.utils import clip_grad_norm_
from attention_moonshot.synthetic import MarkedRecallConfig,collision_free_marked_recall_batch
from attention_moonshot.model import ModelConfig,ByteGPT
from attention_moonshot.candidates import marked_recall_programs
from attention_moonshot.experiment import seed_all
seed_all(7,2);torch.set_flush_denormal(bool($flush))
ck=torch.load('results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt',weights_only=False)
print('ck step',ck['step'])
mc=ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode='none');m=ByteGPT(mc,marked_recall_programs()[0]);m.load_state_dict(ck['model_state']);o=torch.optim.AdamW(m.parameters(),lr=3e-3);o.load_state_dict(ck['optimizer_state']);g=torch.Generator();g.set_state(ck['generator_state'])
for step in range(3):
 t=time.perf_counter();b=collision_free_marked_recall_batch(batch_size=24,context=160,distractor_min=8,distractor_max=14,target_mode='random',generator=g);print('gen',time.perf_counter()-t)
 t=time.perf_counter();o.zero_grad(set_to_none=True);print('zero',time.perf_counter()-t)
 t=time.perf_counter();z=m(b.tokens)[:,-1];print('fwd',time.perf_counter()-t)
 t=time.perf_counter();loss=torch.nn.functional.cross_entropy(z,b.targets);print('loss',time.perf_counter()-t,loss.item())
 t=time.perf_counter();loss.backward();print('bwd',time.perf_counter()-t)
 t=time.perf_counter();print('norm',float(clip_grad_norm_(m.parameters(),1.0)),time.perf_counter()-t)
 t=time.perf_counter();o.step();print('opt',time.perf_counter()-t)
PY
done
FLUSH=0
ck step 600
gen 0.0015511239998886595
zero 0.0002406089988653548
fwd 0.09090896199995768
loss 0.00034814700120477937 3.5102529525756836
bwd 0.09975764999944658
norm 0.9966464042663574 0.000953873999606003
opt 0.0019626089997473173
gen 0.0017626200005906867
zero 0.0001444509998691501
fwd 0.03523879800013674
loss 0.0002401750007265946 3.5141496658325195
bwd 0.06887760199970216
norm 0.9420972466468811 0.0009785879992705304
opt 0.001452839000194217
gen 0.0012547370006359415
zero 0.00010403499982203357
fwd 0.025131096999757574
loss 0.00015865099885559175 3.5085134506225586
bwd 0.07746620299985807
norm 0.9294064044952393 0.0006915100002515828
opt 0.0019543630005500745
FLUSH=1
ck step 600
gen 0.0015909909998299554
zero 0.0002637210000102641
fwd 0.09718718899966916
loss 0.0002364169995416887 3.5102529525756836
bwd 0.08394820499961497
norm 0.9966464042663574 0.002163479999580886
opt 0.002144247000614996
gen 0.0017862699987745145
zero 0.00014209099936124403
fwd 0.04862478799987002
loss 0.0003750080013560364 3.5141496658325195
bwd 0.05289453399927879
norm 0.9420972466468811 0.0005242229999566916
opt 0.007614782000018749
gen 0.0010968309998133918
zero 0.00010219499927188735
fwd 0.049823755000034
loss 0.0001516940010333201 3.5085134506225586
bwd 0.026980960999935633
norm 0.9294064044952393 0.00043069899948022794
opt 0.001690598001005128
```



Analyzed

```
bash -lc PYTHONPATH=src python3 -u - <<'PY'
import time,torch
from attention_moonshot.synthetic import MarkedRecallConfig,finalize_marked_recall_checkpoint
from attention_moonshot.model import ModelConfig
from attention_moonshot.candidates import marked_recall_programs
p='results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt';ck=torch.load(p,weights_only=False);print('step',ck['step'])
mc=ModelConfig(vocab_size=256,context=160,d_model=48,n_heads=4,n_layers=2,position_mode='none');cfg=MarkedRecallConfig(steps=600,batch_size=24,learning_rate=3e-3,eval_batches=8,eval_batch_size=48,train_distractor_min=8,train_distractor_max=14,shift_distractor_min=20,shift_distractor_max=30,torch_threads=2)
t=time.perf_counter();r=finalize_marked_recall_checkpoint(mc,marked_recall_programs()[0],cfg,checkpoint=ck);print('elapsed',time.perf_counter()-t);print(r['conditions'])
PY
step 600
elapsed 3.6135230529998807
{'in_dist_random': {'accuracy': 0.0078125, 'bits': 5.089716770026021, 'count': 384, 'rank_accuracy': {'early': 0.008771929824561403, 'middle': 0.007692307692307693, 'late': 0.007142857142857143}}, 'shift_random': {'accuracy': 0.033854166666666664, 'bits': 5.053072541261782, 'count': 384, 'rank_accuracy': {'early': 0.014814814814814815, 'middle': 0.019417475728155338, 'late': 0.06164383561643835}}, 'shift_first': {'accuracy': 0.049479166666666664, 'bits': 5.033233425243953, 'count': 384, 'rank_accuracy': {'early': 0.049479166666666664}}, 'shift_middle': {'accuracy': 0.018229166666666668, 'bits': 5.072750337553104, 'count': 384, 'rank_accuracy': {'middle': 0.018229166666666668}}, 'shift_last': {'accuracy': 0.036458333333333336, 'bits': 5.057832791727464, 'count': 384, 'rank_accuracy': {'late': 0.036458333333333336}}, 'counterfactual': {'accuracy_first': 0.010416666666666666, 'accuracy_second': 0.026041666666666668, 'both_correct': 0.0, 'prediction_change_rate': 0.515625, 'bits': 5.074703114562325, 'count': 384}}
```



Analyzed

```
bash -lc cat >> tests/test_marked_retrieval.py <<'PY'




def test_marked_recall_checkpoint_can_extend_the_training_horizon() -> None:
    from dataclasses import replace
    from attention_moonshot.candidates import marked_recall_programs
    from attention_moonshot.model import ModelConfig


    mc = ModelConfig(
        vocab_size=256, context=32, d_model=16, n_heads=4, n_layers=2,
        position_mode="none",
    )
    short = synthetic.MarkedRecallConfig(
        steps=2, batch_size=2, eval_batches=1, eval_batch_size=2,
        train_distractor_min=1, train_distractor_max=2,
        shift_distractor_min=3, shift_distractor_max=4, torch_threads=1,
    )
    checkpoint = synthetic.advance_marked_recall_checkpoint(
        mc, marked_recall_programs()[0], short, seed=43, mode="hardened",
        checkpoint=None, segment_steps=2,
    )
    extended = synthetic.advance_marked_recall_checkpoint(
        mc, marked_recall_programs()[0], replace(short, steps=4), seed=43,
        mode="hardened", checkpoint=checkpoint, segment_steps=2,
    )
    assert extended["step"] == 4
PY
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_marked_retrieval.py::test_marked_recall_checkpoint_can_extend_the_training_horizon -q
F                                                                        [100%]
=================================== FAILURES ===================================
________ test_marked_recall_checkpoint_can_extend_the_training_horizon _________

    def test_marked_recall_checkpoint_can_extend_the_training_horizon() -> None:
        from dataclasses import replace
        from attention_moonshot.candidates import marked_recall_programs
        from attention_moonshot.model import ModelConfig
    
        mc = ModelConfig(
            vocab_size=256, context=32, d_model=16, n_heads=4, n_layers=2,
            position_mode="none",
        )
        short = synthetic.MarkedRecallConfig(
            steps=2, batch_size=2, eval_batches=1, eval_batch_size=2,
            train_distractor_min=1, train_distractor_max=2,
            shift_distractor_min=3, shift_distractor_max=4, torch_threads=1,
        )
        checkpoint = synthetic.advance_marked_recall_checkpoint(
            mc, marked_recall_programs()[0], short, seed=43, mode="hardened",
            checkpoint=None, segment_steps=2,
        )
>       extended = synthetic.advance_marked_recall_checkpoint(
            mc, marked_recall_programs()[0], replace(short, steps=4), seed=43,
            mode="hardened", checkpoint=checkpoint, segment_steps=2,
        )

tests/test_marked_retrieval.py:293: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

mc = ModelConfig(vocab_size=256, context=32, d_model=16, n_heads=4, n_layers=2, ff_mult=4, dropout=0.0, position_mode='none')
spec = (AttentionSpec(kind='softmax', alpha=1.0, usage_decay=1.0, eps=0.0, diagonal_debit=0.0, slope=1.0, local_heads=0, self...1.0, free_floor=0.25, free_heads=1, specialist_init='llb', init_strength=4.0, anchor_pattern='llb', adapt_budget=0.25))
cfg = MarkedRecallConfig(steps=4, batch_size=2, learning_rate=0.003, weight_decay=0.0, grad_clip=1.0, eval_batches=1, eval_b...ize=2, train_distractor_min=1, train_distractor_max=2, shift_distractor_min=3, shift_distractor_max=4, torch_threads=1)

    def advance_marked_recall_checkpoint(
        mc: ModelConfig,
        spec: AttentionProgram,
        cfg: MarkedRecallConfig,
        *,
        seed: int,
        mode: str,
        checkpoint: dict[str, Any] | None,
        segment_steps: int,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Advance a marked-recall training run without changing its RNG stream."""
        if segment_steps < 1:
            raise ValueError("segment_steps must be positive")
        if mc.position_mode != "none":
            raise ValueError("marked retrieval requires position_mode='none'")
        if mode not in {"legacy", "hardened"}:
            raise ValueError("mode must be 'legacy' or 'hardened'")
        _validate_marked_recall_request(
            batch_size=cfg.batch_size,
            context=mc.context,
            distractor_min=cfg.train_distractor_min,
            distractor_max=max(cfg.train_distractor_max, cfg.shift_distractor_max),
        )
        seed_all(seed, cfg.torch_threads)
        model = ByteGPT(mc, spec)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        generator = torch.Generator().manual_seed(3000 + seed)
        label = attention_program_label(spec)
        initial_digest = _model_initialization_digest(model)
        current_step = 0
        stream_digest = "00" * 32
        elapsed = 0.0
        if checkpoint is not None:
            if checkpoint["operator"] != label or checkpoint["mode"] != mode:
                raise ValueError("checkpoint operator or mode mismatch")
            if int(checkpoint["seed"]) != seed:
                raise ValueError("checkpoint seed mismatch")
            if checkpoint["model_config"] != asdict(mc) or checkpoint["train_config"] != asdict(cfg):
>               raise ValueError("checkpoint configuration mismatch")
E               ValueError: checkpoint configuration mismatch

src/attention_moonshot/synthetic.py:516: ValueError
=========================== short test summary info ============================
FAILED tests/test_marked_retrieval.py::test_marked_recall_checkpoint_can_extend_the_training_horizon - ValueError: checkpoint configuration mismatch
1 failed in 2.20s


Command 'bash -lc 'cat >> tests/test_marked_retrieval.py <<'"'"'PY'"'"'


def test_marked_recall_checkpoint_can_extend_the_training_horizon() -> None:
    from dataclasses import replace
    from attention_moonshot.candidates import marked_recall_programs
    from attention_moonshot.model import ModelConfig

    mc = ModelConfig(
        vocab_size=256, context=32, d_model=16, n_heads=4, n_layers=2,
        position_mode="none",
    )
    short = synthetic.MarkedRecallConfig(
        steps=2, batch_size=2, eval_batches=1, eval_batch_size=2,
        train_distractor_min=1, train_distractor_max=2,
        shift_distractor_min=3, shift_distractor_max=4, torch_threads=1,
    )
    checkpoint = synthetic.advance_marked_recall_checkpoint(
        mc, marked_recall_programs()[0], short, seed=43, mode="hardened",
        checkpoint=None, segment_steps=2,
    )
    extended = synthetic.advance_marked_recall_checkpoint(
        mc, marked_recall_programs()[0], replace(short, steps=4), seed=43,
        mode="hardened", checkpoint=checkpoint, segment_steps=2,
    )
    assert extended["step"] == 4
PY
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_marked_retrieval.py::test_marked_recall_checkpoint_can_extend_the_training_horizon -q'' failed with status 1.
```



Analyzed

```
bash -lc python3 - <<'PY'
from pathlib import Path
p=Path('src/attention_moonshot/synthetic.py')
s=p.read_text()
old='''        if checkpoint["model_config"] != asdict(mc) or checkpoint["train_config"] != asdict(cfg):
            raise ValueError("checkpoint configuration mismatch")
'''
new='''        saved_train_config = dict(checkpoint["train_config"])
        current_train_config = asdict(cfg)
        saved_train_config.pop("steps", None)
        current_train_config.pop("steps", None)
        if checkpoint["model_config"] != asdict(mc) or saved_train_config != current_train_config:
            raise ValueError("checkpoint configuration mismatch")
        if cfg.steps < int(checkpoint["step"]):
            raise ValueError("configured horizon is shorter than checkpoint step")
'''
if old not in s: raise SystemExit('pattern missing')
s=s.replace(old,new)
p.write_text(s)
PY
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_marked_retrieval.py::test_marked_recall_checkpoint_can_extend_the_training_horizon -q
.                                                                        [100%]
1 passed in 2.64s
```



Analyzed

```
bash -lc cat > scripts/supervise_marked_recall_campaign.py <<'PY'
from __future__ import annotations


import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from attention_moonshot.candidates import marked_recall_programs
from attention_moonshot.experiment import attention_program_label




def result_path(output: Path, mode: str, seed: int, arm: int) -> Path:
    label = attention_program_label(marked_recall_programs()[arm - 1])
    digest = hashlib.sha256(label.encode()).hexdigest()[:10]
    return output / "raw" / f"{mode}-seed{seed}-arm{arm:02d}-{digest}.json"




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=("legacy", "hardened"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--arms", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--segment-steps", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", default="results/nope_marked_recall")
    args = parser.parse_args()


    output = ROOT / args.output
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(mode, seed, arm) for mode in args.modes for seed in args.seeds for arm in args.arms]
    completed = 0
    for mode, seed, arm in tasks:
        result = result_path(output, mode, seed, arm)
        log = log_dir / f"{mode}-seed{seed}-arm{arm:02d}.log"
        attempts = 0
        while not result.exists():
            attempts += 1
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_marked_recall_suite.py"),
                "--mode", mode,
                "--arm", str(arm),
                "--seed", str(seed),
                "--steps", str(args.steps),
                "--segment-steps", str(args.segment_steps),
                "--threads", str(args.threads),
                "--width", str(args.width),
                "--output", str(output.relative_to(ROOT)),
            ]
            with log.open("a") as handle:
                handle.write(f"\n=== attempt {attempts} {time.strftime('%F %T')} ===\n")
                handle.flush()
                try:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=args.timeout,
                    )
                except subprocess.TimeoutExpired:
                    handle.write(f"TIMEOUT after {args.timeout}s; retrying from atomic checkpoint\n")
                except subprocess.CalledProcessError as error:
                    handle.write(f"EXIT {error.returncode}; retrying from atomic checkpoint\n")
            if attempts >= 30 and not result.exists():
                raise SystemExit(f"failed after {attempts} attempts: {mode} seed={seed} arm={arm}")
        completed += 1
        print(f"COMPLETE {completed}/{len(tasks)} {mode} seed={seed} arm={arm}", flush=True)
    print(f"ALL COMPLETE {len(tasks)}/{len(tasks)}", flush=True)




if __name__ == "__main__":
    main()
PY
python3 scripts/supervise_marked_recall_campaign.py --modes hardened --seeds 7 --arms 1 --steps 600 --segment-steps 100 --threads 2 --width 48 --timeout 180 --output results/nope_marked_recall_calibration_chunked
COMPLETE 1/1 hardened seed=7 arm=1
ALL COMPLETE 1/1


Command failed because it timed out.
```



Analyzed

```
bash -lc python3 - <<'PY'
import glob,json
print(glob.glob('results/nope_marked_recall_calibration_chunked/raw/*.json'))
for p in glob.glob('results/nope_marked_recall_calibration_chunked/raw/*.json'):
 r=json.load(open(p));print(r['short_program'],r['conditions'],r['tokens_per_second'])
PY
sed -n '1,160p' results/nope_marked_recall_calibration_chunked/REPORT.md
['results/nope_marked_recall_calibration_chunked/raw/hardened-seed7-arm01-fa95773798.json']
F -> F {'counterfactual': {'accuracy_first': 0.010416666666666666, 'accuracy_second': 0.026041666666666668, 'bits': 5.074703114562325, 'both_correct': 0.0, 'count': 384, 'prediction_change_rate': 0.515625}, 'in_dist_random': {'accuracy': 0.0078125, 'bits': 5.089716770026021, 'count': 384, 'rank_accuracy': {'early': 0.008771929824561403, 'late': 0.007142857142857143, 'middle': 0.007692307692307693}}, 'shift_first': {'accuracy': 0.049479166666666664, 'bits': 5.033233425243953, 'count': 384, 'rank_accuracy': {'early': 0.049479166666666664}}, 'shift_last': {'accuracy': 0.036458333333333336, 'bits': 5.057832791727464, 'count': 384, 'rank_accuracy': {'late': 0.036458333333333336}}, 'shift_middle': {'accuracy': 0.018229166666666668, 'bits': 5.072750337553104, 'count': 384, 'rank_accuracy': {'middle': 0.018229166666666668}}, 'shift_random': {'accuracy': 0.033854166666666664, 'bits': 5.053072541261782, 'count': 384, 'rank_accuracy': {'early': 0.014814814814814815, 'late': 0.06164383561643835, 'middle': 0.019417475728155338}}} 10063.62070694336
# NoPE Marked Retrieval — External v1 Reproduction and Hardened v1.1

Completed **1** atomic runs. Every arm uses NoPE, two layers, four heads, identical initialization and batch stream within each mode/seed pair.

## Pairing audit

**PASS:** all comparable arms have identical initialization and batch-stream digests.

## Hardened v1.1: unique table, randomized query rank

Every table uses unique keys and values. Training queries a random table row. Counterfactual evaluation changes only the final query key while keeping the entire table fixed.

|#|program|runs|shift random|first|middle|last|counterfactual both|query-change|shift bits|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|`F -> F`|1|0.0339 ± 0.0000|0.0495|0.0182|0.0365|0.0000 ± 0.0000|0.5156|5.0531|

## Hardened rank bins

### `F -> F`

|seed|early|middle|late|
|---:|---:|---:|---:|
|7|0.0148|0.0194|0.0616|
```



Analyzed

```
bash -lc rm -f results/nope_marked_recall_calibration_chunked/raw/hardened-seed7-arm01-fa95773798.json
nohup python3 scripts/supervise_marked_recall_campaign.py --modes hardened --seeds 7 --arms 1 --steps 1200 --segment-steps 100 --threads 2 --width 48 --timeout 180 --output results/nope_marked_recall_calibration_chunked > /tmp/supervise1200.log 2>&1 & echo $!
```



Analyzed

```
38924
```



Analyzed

```
bash -lc cat /tmp/supervise1200.log; ps -p 38924 -o pid,stat,etime,cmd || true; python3 - <<'PY'
import torch,glob,os
for p in glob.glob('results/nope_marked_recall_calibration_chunked/checkpoints/*.pt'):
 x=torch.load(p,weights_only=False);print('step',x['step'])
PY
    PID STAT     ELAPSED CMD
  38924 Sl         00:11 python3 scripts/supervise_marked_recall_campaign.py --modes hardened --seeds 7 --arms 1 --steps 1200 --segment-steps 100 --threads 2 --width 48 --timeout 180 --output results/nope_marked_recall_calibration_chunked
step 600
```



Analyzed

```
bash -lc cat /tmp/supervise1200.log; ps -p 38924 -o pid,stat,etime || true; python3 - <<'PY'
import torch,glob
for p in glob.glob('results/nope_marked_recall_calibration_chunked/checkpoints/*.pt'):
 x=torch.load(p,weights_only=False);print('step',x['step'])
PY
    PID STAT     ELAPSED
  38924 Sl         01:00
step 1000
```



Analyzed

```
bash -lc tail -80 results/nope_marked_recall_calibration_chunked/logs/hardened-seed7-arm01.log
=== attempt 1 2026-08-20 06:37:29 ===
WROTE /mnt/data/attention-moonshot-nope-v1/results/nope_marked_recall_calibration_chunked/raw/hardened-seed7-arm01-fa95773798.json

=== attempt 1 2026-08-20 06:38:25 ===
marked_segment hardened softmax -> softmax 7 601 5.064224720001221
marked_segment hardened softmax -> softmax 7 650 4.99555778503418
marked_segment hardened softmax -> softmax 7 700 4.983762741088867
CHECKPOINT /mnt/data/attention-moonshot-nope-v1/results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt step=700/1200

=== attempt 2 2026-08-20 06:38:48 ===
marked_segment hardened softmax -> softmax 7 701 5.1102824211120605
marked_segment hardened softmax -> softmax 7 750 5.123144626617432
marked_segment hardened softmax -> softmax 7 800 5.068881988525391
CHECKPOINT /mnt/data/attention-moonshot-nope-v1/results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt step=800/1200

=== attempt 3 2026-08-20 06:38:58 ===
marked_segment hardened softmax -> softmax 7 801 5.098690986633301
marked_segment hardened softmax -> softmax 7 850 5.046379089355469
marked_segment hardened softmax -> softmax 7 900 5.065659046173096
CHECKPOINT /mnt/data/attention-moonshot-nope-v1/results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt step=900/1200

=== attempt 4 2026-08-20 06:39:10 ===
marked_segment hardened softmax -> softmax 7 901 5.0423455238342285
marked_segment hardened softmax -> softmax 7 950 4.9549560546875
marked_segment hardened softmax -> softmax 7 1000 5.010983943939209
CHECKPOINT /mnt/data/attention-moonshot-nope-v1/results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt step=1000/1200

=== attempt 5 2026-08-20 06:39:22 ===
marked_segment hardened softmax -> softmax 7 1001 5.001674652099609
marked_segment hardened softmax -> softmax 7 1050 4.99770975112915
marked_segment hardened softmax -> softmax 7 1100 5.067416191101074
CHECKPOINT /mnt/data/attention-moonshot-nope-v1/results/nope_marked_recall_calibration_chunked/checkpoints/hardened-seed7-arm01-fa95773798.pt step=1100/1200
```



Analyzed

```
bash -lc sleep 25; cat /tmp/supervise1200.log; ps -p 38924 -o pid,stat,etime || true; python3 - <<'PY'
import glob,json,torch
print(glob.glob('results/nope_marked_recall_calibration_chunked/raw/*.json'))
for p in glob.glob('results/nope_marked_recall_calibration_chunked/checkpoints/*.pt'):
 x=torch.load(p,weights_only=False);print('step',x['step'])
PY
```