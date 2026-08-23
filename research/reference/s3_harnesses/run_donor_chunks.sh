#!/usr/bin/env bash
set -euo pipefail

export DONOR_TASK="add2"
export DONOR_OUTPUT_PREFIX="donor_chunk"
unset DONOR_CANDIDATES
exec "$(dirname "$0")/run_donor_chunks_common.sh"
