# asm-rs

A pure Rust multi-architecture assembly engine for offensive security.

[![Crates.io](https://img.shields.io/crates/v/asm-rs.svg)](https://crates.io/crates/asm-rs)
[![docs.rs](https://docs.rs/asm-rs/badge.svg)](https://docs.rs/asm-rs)
[![CI](https://github.com/hupe1980/asm-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/hupe1980/asm-rs/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

[📖 Documentation](https://hupe1980.github.io/asm-rs/) · [📚 API Reference](https://docs.rs/asm-rs) · [📦 Crate](https://crates.io/crates/asm-rs)

> Zero `unsafe`, `no_std`-compatible, designed for embedding in exploit compilers, JIT engines, security tools, and shellcode generators.

## ✨ Features

- 🦀 **Pure Rust** — `#![forbid(unsafe_code)]`, no C dependencies
- 📦 **`no_std` support** — embedded and WASM environments (with `alloc`)
- 🏗️ **Multi-architecture** — x86, x86-64, ARM32, Thumb/Thumb-2, AArch64, RISC-V
- 🔄 **Intel & AT&T syntax** — full GAS-compatible AT&T with `.syntax att`/`.syntax intel`
- 🏷️ **Labels & constants** — forward/backward references, numeric labels, `.equ`/`.set`
- 🔀 **Branch relaxation** — Szymanski's algorithm for optimal branch encoding
- 📐 **Preprocessor** — `.macro`/`.rept`/`.irp`/`.if`/`.ifdef` directives
- ⚡ **Peephole optimizer** — zero-idiom, MOV narrowing, REX elimination
- 🔧 **Compile-time macros** — `asm_bytes!`/`asm_array!` for zero-overhead assembly
- 🎯 **Literal pools** — `LDR Xn/Rn, =value` with automatic pool management (AArch64/ARM32)
- 📋 **Listing output** — human-readable address/hex/source listing for debugging
- 🔗 **Applied relocations** — full relocation info exposed for tooling
- 🧬 **Serde support** — optional serialization for all public types

## 🏛️ Supported Architectures

| Architecture | Variants | Highlights |
|:---|:---|:---|
| **x86** | 32-bit, 16-bit real mode | Full ISA, `.code16`/`.code32` |
| **x86-64** | 64-bit | SSE–SSE4.2, AVX/AVX2, AVX-512, AES-NI, BMI1/2, FMA3 |
| **ARM32** | A32 (ARMv7) | Condition codes, barrel shifter, literal pools |
| **Thumb** | T16/T32 | Auto 16/32-bit encoding, IT blocks |
| **AArch64** | A64 (ARMv8+) | NEON/AdvSIMD, LSE atomics, literal pools |
| **RISC-V** | RV32I, RV64I | M/A/C extensions, auto-compression |

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
asm-rs = "0.1"
```

### One-Shot Assembly

```rust
use asm_rs::{assemble, Arch};

let bytes = assemble("mov eax, 42\nret", Arch::X86_64).unwrap();
assert_eq!(bytes[0], 0xB8); // mov eax, imm32
```

### Builder API

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.emit("push rbp").unwrap();
asm.emit("mov rbp, rsp").unwrap();
asm.emit("sub rsp, 0x20").unwrap();
// ... function body ...
asm.emit("add rsp, 0x20").unwrap();
asm.emit("pop rbp").unwrap();
asm.emit("ret").unwrap();

let result = asm.finish().unwrap();
println!("Generated {} bytes", result.len());
```

### AT&T / GAS Syntax

```rust
use asm_rs::{Assembler, Arch, Syntax};

let mut asm = Assembler::new(Arch::X86_64);
asm.syntax(Syntax::Att);
asm.emit(r#"
    pushq %rbp
    movq %rsp, %rbp
    movl $42, %eax
    popq %rbp
    ret
"#).unwrap();
```

### Multi-Architecture Shellcode

```rust
use asm_rs::{Assembler, Arch};

// x86-64
let bytes = asm_rs::assemble("xor edi, edi; mov eax, 60; syscall", Arch::X86_64).unwrap();

// AArch64
let mut asm = Assembler::new(Arch::Aarch64);
asm.emit("mov x0, #0; mov x8, #93; svc #0").unwrap();

// ARM32
let mut asm = Assembler::new(Arch::Arm);
asm.emit("mov r0, #0; mov r7, #1; svc #0").unwrap();

// RISC-V
let bytes = asm_rs::assemble("li a7, 93; li a0, 0; ecall", Arch::Rv32).unwrap();
```

### Compile-Time Assembly

```rust
use asm_rs_macros::{asm_bytes, asm_array};

const SHELLCODE: &[u8] = asm_bytes!(x86_64, "xor eax, eax; inc eax; ret");
const NOP: [u8; 1] = asm_array!(x86_64, "nop");
const ARM_CODE: &[u8] = asm_bytes!(arm, "bx lr");
```

> See [`crates/asm-rs-macros/README.md`](crates/asm-rs-macros/README.md) for full proc-macro documentation.

## 🧪 Testing

Extensive test suite covering unit, integration, cross-validation (llvm-mc, iced-x86, yaxpeax-arm, riscv-decode), property-based (proptest), and fuzz testing (`cargo-fuzz`). Zero warnings, Miri clean.

## ⚙️ Configuration

### Cargo Features

| Feature | Default | Description |
|:---|:---:|:---|
| `std` | ✅ | Standard library support |
| `x86` | ✅ | x86 (32-bit) backend |
| `x86_64` | ✅ | x86-64 backend |
| `arm` | ✅ | ARM32 + Thumb/Thumb-2 backend |
| `aarch64` | ✅ | AArch64 backend |
| `riscv` | ✅ | RISC-V backend |
| `avx` | ✅ | AVX/AVX2/FMA |
| `avx512` | ✅ | AVX-512/EVEX |
| `neon` | ✅ | AArch64 NEON/AdvSIMD |
| `sve` | ❌ | AArch64 SVE |
| `riscv_f` | ✅ | RISC-V F/D floating-point |
| `riscv_v` | ❌ | RISC-V V vector |
| `serde` | ❌ | Serialize/Deserialize for public types |

### MSRV

Rust **1.75** or later.

## 📖 Learn More

For the full reference — architecture details, ISA instruction tables, directives, API docs, and configuration options — visit the **[documentation site](https://hupe1980.github.io/asm-rs/)**.

## 📄 License

Licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
