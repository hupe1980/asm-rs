#!/usr/bin/env bash
# scripts/check.sh — Run the same checks as CI locally.
#
# Usage:
#   ./scripts/check.sh          # run all checks
#   ./scripts/check.sh --quick  # skip slow checks (coverage, MSRV)
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
      echo "  --quick, -q   Skip slow checks (coverage, MSRV)"
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

skip() {
  echo -e "  ${YELLOW}⊘ $1${RESET}"
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

ensure_target() {
  local target="$1"
  if ! rustup target list --installed | grep -q "$target"; then
    echo -e "  ${YELLOW}Installing target $target...${RESET}"
    rustup target add "$target"
  fi
}

# ── Match CI: RUSTFLAGS=-Dwarnings ─────────────────────────
export RUSTFLAGS="${RUSTFLAGS:--Dwarnings}"

# ── Format ─────────────────────────────────────────────────
run_check "rustfmt" cargo fmt --all -- --check

# ── Clippy ─────────────────────────────────────────────────
run_check "clippy" cargo clippy --workspace --all-features -- -D warnings

# ── Tests ──────────────────────────────────────────────────
run_check "tests" cargo test --workspace --all-features

# ── no_std build ───────────────────────────────────────────
step "no_std build"
ensure_target thumbv7em-none-eabihf
if cargo build -p asm-rs --no-default-features \
    --features "x86,x86_64,arm,aarch64,riscv" \
    --target thumbv7em-none-eabihf; then
  pass "no_std build"
else
  fail "no_std build"
fi

# ── WASM build ─────────────────────────────────────────────
step "WASM build"
ensure_target wasm32-unknown-unknown
if cargo check -p asm-rs --target wasm32-unknown-unknown --all-features; then
  pass "WASM build"
else
  fail "WASM build"
fi

# ── WASM tests ─────────────────────────────────────────────
step "WASM tests"
if command -v wasm-bindgen &>/dev/null; then
  if WASM_BINDGEN_TEST_ONLY_NODE=1 \
      cargo test -p asm-rs --target wasm32-unknown-unknown \
      --test wasm_integration --features "x86,x86_64,arm,aarch64,riscv"; then
    pass "WASM tests"
  else
    fail "WASM tests"
  fi
else
  skip "skipped (install: cargo install wasm-bindgen-cli)"
fi

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
  if cargo check -p asm-rs --no-default-features --features "std,$features"; then
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

  # ── MSRV ───────────────────────────────────────────────
  step "MSRV (1.75)"
  if rustup toolchain list | grep -q "1\.75"; then
    if cargo +1.75 check --workspace --all-features; then
      pass "MSRV (1.75)"
    else
      fail "MSRV (1.75)"
    fi
  elif rustup toolchain list | grep -q "1\.75\.0"; then
    if cargo +1.75.0 check --workspace --all-features; then
      pass "MSRV (1.75)"
    else
      fail "MSRV (1.75)"
    fi
  else
    skip "skipped (install: rustup toolchain install 1.75)"
  fi

  # ── Coverage ───────────────────────────────────────────
  step "coverage"
  if command -v cargo-llvm-cov &>/dev/null; then
    if cargo llvm-cov --workspace --all-features --fail-under-lines 80; then
      pass "coverage (≥80%)"
    else
      fail "coverage (≥80%)"
    fi
  else
    skip "skipped (install: cargo install cargo-llvm-cov)"
  fi

else
  echo -e "\n${YELLOW}--quick: skipping MSRV, coverage${RESET}"
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