# Performance

Benchmark results for `asm-rs`, measured with [Criterion.rs](https://github.com/bheisler/criterion.rs).

**Hardware**: Apple Silicon (M-series), macOS, Rust 1.92.0.  
**Run**: `cargo bench --bench throughput`

## Single Instruction Latency

| Benchmark | Time | Notes |
|---|---|---|
| `x86_64_nop` | ~260 ns | Simplest instruction |
| `x86_64_mov_reg_imm` | ~385 ns | REX + ModR/M + imm |
| `x86_64_add_reg_reg` | ~363 ns | REX + ModR/M |
| `x86_64_mov_mem` | ~565 ns | SIB + displacement |
| `x86_64_vaddps_avx` | ~490 ns | VEX 3-operand |
| `x86_64_vaddps_avx512` | ~490 ns | EVEX 3-operand |
| `aarch64_add` | ~425 ns | Fixed-width encoding |
| `aarch64_ldr` | ~445 ns | Load/store with offset |
| `arm_add` | ~405 ns | ARM A32 |
| `riscv64_add` | ~455 ns | R-type |
| `riscv64_lw` | ~425 ns | I-type load |

**Target**: < 1 µs for simple instructions — **achieved** for all cases.

## Throughput (Multi-Instruction)

| Benchmark | Time | Throughput |
|---|---|---|
| `x86_64_100_insn` | ~24 µs | ~50 MiB/s |
| `x86_64_1000_insn` | ~230 µs | ~53 MiB/s |
| `x86_64_5000_insn` | ~1.2 ms | ~50 MiB/s |
| `aarch64_1000_insn` | ~240 µs | ~62 MiB/s |
| `riscv_1000_insn` | ~230 µs | ~64 MiB/s |

Throughput is consistent at ~50–64 MiB/s across instruction counts and architectures, demonstrating linear scaling.

## Label Resolution

| Benchmark | Time |
|---|---|
| `50_labels` | ~53 µs |
| `200_labels` | ~140 µs |
| `500_labels` | ~300 µs |

Near-linear scaling — no quadratic blowup in label resolution.

## Branch Relaxation

| Benchmark | Time | Notes |
|---|---|---|
| `short_branch_10_nop` | ~5.2 µs | No relaxation needed |
| `edge_branch_120_nop` | ~30 µs | Near boundary |
| `long_branch_200_nop` | ~50 µs | Relaxation triggered |

Relaxation adds minimal overhead — proportional to code size, not branch count.

## Preprocessor

| Benchmark | Time |
|---|---|
| `rept_100_nop` | ~24 µs |
| `macro_push_pair_7x` | ~9.6 µs |
| `conditional_assembly` | ~2.6 µs |

Macro expansion and conditional assembly are extremely fast.

## API Comparison

| Benchmark | Time |
|---|---|
| `oneshot_4_insn` | ~2.25 µs |
| `builder_4_insn` | ~2.57 µs |
| `builder_4_insn_at_base` | ~2.58 µs |

The `assemble()` one-shot API is slightly faster than the builder for small sequences due to reduced overhead.

## Realistic Workloads

| Benchmark | Time |
|---|---|
| `syscall_stub` (5 insn) | ~2.9 µs |
| `function_prolog_epilog` (9 insn) | ~5.5 µs |
| `avx512_vector_body` (6 insn) | ~5.0 µs |
| `aarch64_function` (7 insn) | ~5.0 µs |
| `riscv_function` (10 insn) | ~6.7 µs |

## Running Benchmarks

```bash
# Full benchmark suite
cargo bench --bench throughput

# Quick validation run
cargo bench --bench throughput -- --quick

# Specific group
cargo bench --bench throughput -- single_instruction
```

## Tracking Regressions

Criterion automatically compares against previous runs and reports statistically significant changes. Results are stored in `target/criterion/`.
