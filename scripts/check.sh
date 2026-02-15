#!/usr/bin/env bash
# scripts/check.sh — Run the same checks as CI locally.
#
# Usage:
#   ./scripts/check.sh          # run all checks
#   ./scripts/check.sh --quick  # skip slow checks (coverage, WASM, MSRV)
#
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
RESET='\033[0m'

QUICK=false
FAILED=()

for arg in "$@"; do
  case "$arg" in
    --quick|-q) QUICK=true ;;
    --help|-h)
      echo "Usage: $0 [--quick]"
      echo "  --quick, -q   Skip slow checks (coverage, WASM, MSRV, no_std)"
      exit 0
      ;;
  esac
done

step() {
  echo -e "\n${BOLD}▸ $1${RESET}"
}

pass() {
  echo -e "  ${GREEN}✓ $1${RESET}"
}

fail() {
  echo -e "  ${RED}✗ $1${RESET}"
  FAILED+=("$1")
}

run_check() {
  local name="$1"
  shift
  step "$name"
  if "$@"; then
    pass "$name"
  else
    fail "$name"
  fi
}

# ── Format ─────────────────────────────────────────────────
run_check "rustfmt" cargo fmt --all -- --check

# ── Clippy ─────────────────────────────────────────────────
run_check "clippy" cargo clippy --workspace --all-features -- -D warnings

# ── Tests ──────────────────────────────────────────────────
run_check "tests" cargo test --workspace --all-features

# ── Documentation ──────────────────────────────────────────
run_check "docs" env RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

# ── Benchmarks (compile only) ──────────────────────────────
run_check "bench compile" cargo bench --workspace --all-features --no-run

# ── Feature subsets ────────────────────────────────────────
FEATURE_SETS=(
  "x86_64"
  "x86"
  "aarch64"
  "arm"
  "riscv"
  "x86_64,avx,avx512"
  "aarch64,neon,sve"
  "riscv,riscv_f,riscv_v"
)
step "feature subsets"
ALL_FEATURES_OK=true
for features in "${FEATURE_SETS[@]}"; do
  if cargo check -p asm-rs --no-default-features --features "std,$features" 2>/dev/null; then
    echo -e "  ${GREEN}✓${RESET} $features"
  else
    echo -e "  ${RED}✗${RESET} $features"
    ALL_FEATURES_OK=false
  fi
done
if $ALL_FEATURES_OK; then
  pass "feature subsets"
else
  fail "feature subsets"
fi

# ── Publish dry-run ────────────────────────────────────────
run_check "publish dry-run" cargo publish -p asm-rs --dry-run --allow-dirty

# ── Slow checks (skip with --quick) ───────────────────────
if ! $QUICK; then

  # ── no_std build ───────────────────────────────────────
  if rustup target list --installed | grep -q thumbv7em-none-eabihf; then
    run_check "no_std build" cargo build -p asm-rs --no-default-features \
      --features "x86,x86_64,arm,aarch64,riscv" --target thumbv7em-none-eabihf
  else
    step "no_std build"
    echo -e "  ${YELLOW}⊘ skipped (install target: rustup target add thumbv7em-none-eabihf)${RESET}"
  fi

  # ── WASM tests ─────────────────────────────────────────
  if rustup target list --installed | grep -q wasm32-unknown-unknown && command -v wasm-bindgen &>/dev/null; then
    run_check "WASM tests" env WASM_BINDGEN_TEST_ONLY_NODE=1 \
      cargo test -p asm-rs --target wasm32-unknown-unknown \
      --test wasm_integration --features "x86,x86_64,arm,aarch64,riscv"
  else
    step "WASM tests"
    echo -e "  ${YELLOW}⊘ skipped (need: rustup target add wasm32-unknown-unknown && cargo install wasm-bindgen-cli)${RESET}"
  fi

  # ── MSRV ───────────────────────────────────────────────
  if rustup toolchain list | grep -q "1\.75"; then
    run_check "MSRV (1.75)" cargo +1.75 check --workspace --all-features
  elif rustup toolchain list | grep -q "1\.75\.0"; then
    run_check "MSRV (1.75)" cargo +1.75.0 check --workspace --all-features
  else
    step "MSRV (1.75)"
    echo -e "  ${YELLOW}⊘ skipped (install: rustup toolchain install 1.75)${RESET}"
  fi

  # ── Coverage ───────────────────────────────────────────
  if command -v cargo-llvm-cov &>/dev/null; then
    run_check "coverage (≥80%)" cargo llvm-cov --workspace --all-features --fail-under-lines 80
  else
    step "coverage"
    echo -e "  ${YELLOW}⊘ skipped (install: cargo install cargo-llvm-cov)${RESET}"
  fi

else
  echo -e "\n${YELLOW}--quick: skipping no_std, WASM, MSRV, coverage${RESET}"
fi

# ── Summary ────────────────────────────────────────────────
echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
  echo -e "${GREEN}${BOLD}All checks passed!${RESET}"
  exit 0
else
  echo -e "${RED}${BOLD}${#FAILED[@]} check(s) failed:${RESET}"
  for f in "${FAILED[@]}"; do
    echo -e "  ${RED}✗ $f${RESET}"
  done
  exit 1
fi
