#!/usr/bin/env bash
set -euo pipefail

export DONOR_TASK="prime"
exec "$(dirname "$0")/analyse_binary_donor.sh"
