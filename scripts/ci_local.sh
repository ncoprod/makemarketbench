#!/usr/bin/env bash
# Same checks as hosted CI, on the installed local Python/macOS toolchain.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
export ASAN_OPTIONS=detect_leaks=0
{
  set -x
  date -u
  git rev-parse HEAD
  .venv/bin/python --version
  .venv/bin/ruff check src tests scripts train
  .venv/bin/python -c 'from makemarket.engine import cpp_available; assert cpp_available()'
  .venv/bin/python -m pytest -q
  cmake -S . -B build/native -DMMB_BUILD_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build/native -j 2
  ctest --test-dir build/native --output-on-failure
  cmake -S . -B build/sanitize -DMMB_BUILD_NATIVE=ON -DMMB_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
  cmake --build build/sanitize -j 2
  ctest --test-dir build/sanitize --output-on-failure
  .venv/bin/python train/build_kaggle_kernel.py --out build/kaggle_ci
  MMB_WORK=build/kaggle_dry .venv/bin/python build/kaggle_ci/grpo_unsloth_kaggle.py --dry-run
  date -u
} 2>&1 | tee runs/ci_local.log
