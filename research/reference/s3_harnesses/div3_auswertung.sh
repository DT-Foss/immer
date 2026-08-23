#!/usr/bin/env bash
set -euo pipefail

export DONOR_TASK="div3"
exec "$(dirname "$0")/analyse_binary_donor.sh"
