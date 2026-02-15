---
layout: default
title: Home
nav_order: 1
permalink: /
---

# asm-rs
{: .fs-9 .fw-700 }

A pure Rust multi-architecture assembly engine for offensive security.
{: .fs-6 .fw-300 }

[Get Started]({{ site.baseurl }}/getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/hupe1980/asm-rs){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Why asm-rs?

**asm-rs** is a pure Rust assembler that turns assembly source text into machine code bytes at runtime (or compile time via proc-macros). It supports multiple architectures, requires zero `unsafe` code, and works anywhere Rust does — including `no_std` and WebAssembly.

{: .note }
> Designed for offensive security: shellcode generation, exploit compilers, JIT engines, and security tooling.

## Key Features

| Feature | Description |
|:--------|:------------|
| **Pure Rust** | `#![forbid(unsafe_code)]`, no C dependencies |
| **`no_std` support** | Works in embedded and WASM environments (with `alloc`) |
| **Multi-architecture** | x86, x86-64, ARM32, Thumb/Thumb-2, AArch64, RISC-V |
| **Intel & AT&T syntax** | Full GAS-compatible AT&T syntax with `.syntax att`/`.syntax intel` |
| **Branch relaxation** | Szymanski's algorithm for optimal branch encoding |
| **Preprocessor** | `.macro`/`.rept`/`.irp`/`.if`/`.ifdef` directives |
| **Peephole optimizer** | Zero-idiom, MOV narrowing, REX elimination |
| **Compile-time macros** | `asm_bytes!`/`asm_array!` for zero-overhead assembly |
| **Fuzz & property tested** | cargo-fuzz, proptest, Miri clean |
| **Cross-validated** | Verified against llvm-mc, iced-x86, yaxpeax-arm, riscv-decode |

## Supported Architectures

| Architecture | Variants | Features |
|:-------------|:---------|:---------|
| **x86** | 32-bit, 16-bit real mode | Full ISA, `.code16`/`.code32` |
| **x86-64** | 64-bit | SSE–SSE4.2, AVX/AVX2, AVX-512, AES-NI, SHA, BMI1/2, FMA3, TSX |
| **ARM32 (A32)** | ARMv7 | Condition codes, barrel shifter, MOVW/MOVT, literal pools |
| **Thumb/Thumb-2** | T16/T32 | Auto 16/32-bit encoding, IT blocks, `.thumb_func` |
| **AArch64 (A64)** | ARMv8+ | NEON/AdvSIMD, SVE, LSE atomics, literal pools |
| **RISC-V** | RV32I, RV64I | M/A/F/D/V/C extensions, auto-compression |

## Quick Example

```rust
use asm_rs::{assemble, Arch};

let bytes = assemble("mov eax, 42\nret", Arch::X86_64).unwrap();
assert_eq!(bytes[0], 0xB8); // mov eax, imm32
```

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::Aarch64);
asm.emit(r#"
    mov x0, #0          // status = 0
    mov x8, #93         // __NR_exit
    svc #0              // syscall
"#).unwrap();
let result = asm.finish().unwrap();
```

## Performance

| Metric | Value |
|:-------|:------|
| Single-instruction latency | ~260–565 ns |
| Throughput | ~50–64 MiB/s |
| Per-instruction allocations | Zero (stack-allocated IR) |

---

## About the Project

asm-rs is &copy; 2026 by [hupe1980](https://github.com/hupe1980).

### License

Licensed under either of [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) or [MIT License](http://opensource.org/licenses/MIT), at your option.
