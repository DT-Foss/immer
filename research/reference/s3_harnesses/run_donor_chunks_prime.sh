#!/usr/bin/env bash
set -euo pipefail

export DONOR_TASK="prime"
export DONOR_OUTPUT_PREFIX="donor_chunk_prime"
export DONOR_CANDIDATES="yes,no"
exec "$(dirname "$0")/run_donor_chunks_common.sh"
