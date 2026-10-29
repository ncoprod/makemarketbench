#!/usr/bin/env bash
# Cross-repository mutex for a whole sequential NIM batch or heavy ML job.
set -euo pipefail
resource=$1
shift
mkdir -p "$HOME/.cache"
while :; do
  if [ "$resource" = nim ]; then
    candidates=("$HOME/.cache/nim-slot-1" "$HOME/.cache/nim-slot-2")
  elif [ "$resource" = heavy-ml ]; then
    candidates=("$HOME/.cache/heavy-ml.lock")
  else
    echo 'resource must be nim or heavy-ml' >&2
    exit 2
  fi
  for candidate in "${candidates[@]}"; do
    if mkdir "$candidate" 2>/dev/null; then
      lock=$candidate
      printf '%s\n' "pid=$$ repo=makemarketbench resource=$resource" > "$lock/owner"
      trap 'rm -f "$lock/owner"; rmdir "$lock"' EXIT
      "$@"
      exit
    fi
  done
  if [ "$resource" = nim ]; then sleep 5; else sleep 1; fi
done
