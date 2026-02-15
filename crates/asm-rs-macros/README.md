# asm-rs-macros

Compile-time assembly proc-macros for [`asm-rs`](https://crates.io/crates/asm-rs).

## Features

- **`asm_bytes!`** — Assemble at compile time → `&'static [u8]`
- **`asm_array!`** — Assemble at compile time → `[u8; N]`
- **Zero runtime overhead** — all assembly happens during compilation
- **All architectures** — x86, x86-64, ARM, Thumb, AArch64, RISC-V
- **Full assembler features** — labels, directives, multi-line source
- **Optional base address** — `asm_bytes!(arch, 0x400000, "source")`

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
asm-rs = "0.1"
asm-rs-macros = "0.1"
```

## Examples

### Basic Usage

```rust
use asm_rs_macros::asm_bytes;

// x86-64 shellcode assembled at compile time
const SHELLCODE: &[u8] = asm_bytes!(x86_64, "
    xor eax, eax
    inc eax
    ret
");

// Fixed-size array variant
use asm_rs_macros::asm_array;
const NOP: [u8; 1] = asm_array!(x86_64, "nop");
```

### Multi-Architecture

```rust
use asm_rs_macros::asm_bytes;

const X86_CODE: &[u8]    = asm_bytes!(x86, "nop; ret");
const X64_CODE: &[u8]    = asm_bytes!(x86_64, "nop; ret");
const ARM_CODE: &[u8]    = asm_bytes!(arm, "bx lr");
const THUMB_CODE: &[u8]  = asm_bytes!(thumb, "nop");
const A64_CODE: &[u8]    = asm_bytes!(aarch64, "nop\nret");
const RV32_CODE: &[u8]   = asm_bytes!(rv32, "nop");
const RV64_CODE: &[u8]   = asm_bytes!(rv64, "nop\nret");
```

### With Base Address

```rust
use asm_rs_macros::asm_bytes;

const CODE: &[u8] = asm_bytes!(x86_64, 0x400000, "
    start:
        nop
        jmp start
");
```

### Labels and Directives

```rust
use asm_rs_macros::asm_bytes;

const DATA: &[u8] = asm_bytes!(x86_64, "
    .byte 0xCC
    .byte 0x90
");
```

## Supported Architectures

| Identifier | Architecture |
|------------|-------------|
| `x86` | x86 (32-bit) |
| `x86_64` | x86-64 (64-bit) |
| `arm` | ARM (32-bit) |
| `thumb` | ARM Thumb |
| `aarch64` | AArch64 (ARM 64-bit) |
| `rv32` | RISC-V 32-bit |
| `rv64` | RISC-V 64-bit |

## Error Handling

Assembly errors become compile-time errors with full diagnostics:

```rust,compile_fail
// This produces a compile-time error:
const BAD: &[u8] = asm_bytes!(x86_64, "invalid_mnemonic");
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
