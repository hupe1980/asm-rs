---
layout: default
title: ISA Reference
---

# Instruction Set Architecture Reference

Complete instruction references for each supported architecture.

---

Each reference page lists every supported instruction with its syntax, operand forms,
and encoding notes. Click an architecture below to view its full reference.

| Architecture | Page | Scope |
|:-------------|:-----|:------|
| [x86-64]({{ site.baseurl }}/reference/x86_64) | x86 / x86-64 | Data movement, ALU, SIMD (SSE–AVX-512), AES-NI, SHA, BMI, FMA, TSX |
| [AArch64]({{ site.baseurl }}/reference/aarch64) | ARM64 / A64 | Data processing, load/store, NEON/AdvSIMD, SVE, LSE atomics |
| [ARM32]({{ site.baseurl }}/reference/arm32) | ARMv7 / A32 | Data processing, barrel shifter, condition codes, load/store |
| [Thumb]({{ site.baseurl }}/reference/thumb) | Thumb / T32 | 16-bit Thumb-1, 32-bit Thumb-2, IT blocks |
| [RISC-V]({{ site.baseurl }}/reference/riscv) | RV32I / RV64I | Base ISA, M/A/F/D/V/C extensions |

## Notation Conventions

All reference pages use a consistent notation:

| Symbol | Meaning |
|:-------|:--------|
| `r64` / `r32` / `r16` / `r8` | Register of the specified bit width |
| `[mem]` | Memory operand |
| `imm` / `#imm` | Immediate value |
| `label` | Label reference (resolved by the linker) |
| `cond` | Condition code |
