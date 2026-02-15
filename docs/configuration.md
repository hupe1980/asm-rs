---
layout: default
title: Configuration
nav_order: 4
---

# Configuration
{: .fs-8 }

Cargo features, runtime options, and build configuration for asm-rs.
{: .fs-5 .fw-300 }

---

## Cargo Features

### Architecture Backends

| Feature | Default | Description |
|:--------|:--------|:------------|
| `x86` | Yes | Enable x86 (32-bit) backend |
| `x86_64` | Yes | Enable x86-64 backend |
| `arm` | Yes | Enable ARM32 (A32) + Thumb/Thumb-2 (T32) backend |
| `aarch64` | Yes | Enable AArch64 (A64) backend |
| `riscv` | Yes | Enable RISC-V (RV32I/RV64I + M + A + C extensions) backend |

### SIMD / Extension Features

| Feature | Default | Description |
|:--------|:--------|:------------|
| `avx` | Yes | AVX/AVX2/FMA instructions (requires `x86_64`) |
| `avx512` | Yes | AVX-512/EVEX instructions with opmask + broadcast (requires `x86_64`) |
| `neon` | Yes | AArch64 Advanced SIMD (NEON) instructions (requires `aarch64`) |
| `sve` | No | AArch64 Scalable Vector Extension (requires `aarch64`) |
| `riscv_f` | Yes | RISC-V F/D floating-point extensions (requires `riscv`) |
| `riscv_v` | No | RISC-V V vector extension (requires `riscv`) |

### Other Features

| Feature | Default | Description |
|:--------|:--------|:------------|
| `std` | Yes | Enable standard library support (`std::error::Error`) |
| `serde` | No | Enable `Serialize`/`Deserialize` derives for all public types |

### Usage Examples

```toml
# Default: all architectures, all default SIMD extensions
[dependencies]
asm-rs = "0.1"

# x86-64 only (minimal binary size)
[dependencies]
asm-rs = { version = "0.1", default-features = false, features = ["std", "x86_64", "avx", "avx512"] }

# no_std for embedded/WASM
[dependencies]
asm-rs = { version = "0.1", default-features = false, features = ["x86_64"] }

# With serde support
[dependencies]
asm-rs = { version = "0.1", features = ["serde"] }

# ARM only
[dependencies]
asm-rs = { version = "0.1", default-features = false, features = ["std", "arm", "aarch64", "neon"] }
```

---

## Runtime Options

### Optimization Level

Peephole optimizations are controlled at runtime via `OptLevel`:

```rust
use asm_rs::{Assembler, Arch, OptLevel};

let mut asm = Assembler::new(Arch::X86_64);
asm.opt_level(OptLevel::Size);  // Enable optimizations (default)
asm.opt_level(OptLevel::None);  // Disable optimizations
```

The default is `OptLevel::Size`, which enables these transformations:

| Pattern | Replacement | Savings |
|:--------|:------------|:--------|
| `mov reg64, 0` | `xor reg32, reg32` | 5–7 → 2 bytes |
| `mov reg64, small_imm` | `mov reg32, imm32` | 7 → 5 bytes |
| `and reg64, u32_imm` | `and reg32, u32_imm` | 1 byte (REX removed) |
| `and reg, reg` | `test reg, reg` | Same size, better for flags |

### Syntax

```rust
use asm_rs::{Assembler, Arch, Syntax};

let mut asm = Assembler::new(Arch::X86_64);
asm.syntax(Syntax::Att);   // AT&T / GAS syntax
asm.syntax(Syntax::Intel); // Intel syntax (default for x86)
```

You can also switch syntax mid-stream:

```asm
.syntax att
movq $42, %rax
.syntax intel
mov rbx, 42
```

### Base Address

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.base_address(0x401000); // Set the base address for label resolution
```

### External Labels

Pre-define label addresses for linking against external code:

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.define_external("printf", 0x401000);
asm.emit("mov rax, printf").unwrap();
```

### Constants

Define assembly-time constants:

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.define_constant("SYS_EXIT", 60);
asm.define_constant("BUFFER_SIZE", 4096);
asm.emit("mov eax, SYS_EXIT").unwrap();
```

### Preprocessor Symbols

Define symbols for conditional assembly:

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.define_preprocessor_symbol("DEBUG", 1);
asm.define_preprocessor_symbol("VERSION", 2);
asm.emit(r#"
.ifdef DEBUG
    int 3
.endif
"#).unwrap();
```

### Listing Output

Enable human-readable listing output for debugging:

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.base_address(0x401000);
asm.enable_listing();
asm.emit("entry:\npush rbp\nmov rbp, rsp\nret").unwrap();
let result = asm.finish().unwrap();

println!("{}", result.listing());
// 00401000                  entry:
// 00401000  55                push rbp
// 00401001  4889E5            mov rbp, rsp
// 00401004  C3                ret
```

---

## Directives Reference

| Directive | Description |
|:----------|:------------|
| `.byte` / `.db` | Emit raw bytes |
| `.word` / `.dw` / `.short` | Emit 16-bit values |
| `.long` / `.dd` / `.int` | Emit 32-bit values |
| `.quad` / `.dq` | Emit 64-bit values |
| `.ascii` | Emit string (no terminator) |
| `.asciz` / `.string` | Emit null-terminated string |
| `.equ` / `.set` | Define named constant |
| `name = value` | Alternative constant syntax |
| `.align` / `.balign` | Align to byte boundary |
| `.p2align` | Align to power-of-2 boundary |
| `.fill` | Fill with repeated pattern |
| `.space` / `.skip` | Reserve zero-filled space |
| `.org` | Set origin address (with optional fill byte) |
| `.global` / `.globl` | Declare global symbol (accepted, no-op) |
| `.section` | Declare section (accepted, no-op) |
| `.macro` / `.endm` | Define and end a macro |
| `.rept` / `.endr` | Repeat block N times |
| `.irp` / `.endr` | Iterate over value list |
| `.irpc` / `.endr` | Iterate over characters |
| `.if` / `.else` / `.elseif` / `.endif` | Conditional assembly |
| `.ifdef` / `.ifndef` | Test if symbol is defined |
| `.code16` / `.code32` / `.code64` | Switch operand/address size mode |
| `.syntax att` / `.syntax intel` | Switch assembly syntax |
| `.option rvc` / `.option norvc` | Enable/disable RISC-V C extension |
| `.ltorg` / `.pool` | Flush literal pool (AArch64/ARM) |
| `.thumb` / `.arm` | Switch ARM/Thumb mode |
| `.thumb_func` | Mark next label as Thumb function |

---

## Comments & Separators

```asm
# Hash comments (all architectures)
mov rax, rbx    # inline comment

@ At-sign comments (ARM/Thumb)
add r0, r1, r2  @ inline comment

// Double-slash comments (AArch64)
add x0, x1, x2  // inline comment

; Semicolons are statement separators
nop; nop; ret
```

---

## Minimum Supported Rust Version

**Rust 1.75** or later.

The MSRV is enforced in CI and verified on every push.
