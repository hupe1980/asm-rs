---
layout: default
title: API Reference
---

# API Reference

Complete reference for the asm-rs public API.

For auto-generated Rust API docs, see [docs.rs/asm-rs](https://docs.rs/asm-rs).

---

## Core Types

### `Arch`

Target architecture selection:

```rust
pub enum Arch {
    X86,       // x86 (32-bit)
    X86_64,    // x86-64 (64-bit)
    Arm,       // ARM32 (A32)
    Thumb,     // Thumb / Thumb-2 (T32)
    Aarch64,   // AArch64 (ARM64 / A64)
    Rv32,      // RISC-V 32-bit (RV32I)
    Rv64,      // RISC-V 64-bit (RV64I)
}
```

### `Syntax`

Assembly syntax flavor:

```rust
pub enum Syntax {
    Intel,  // Intel syntax (default for x86/x86-64)
    Att,    // AT&T / GAS syntax
}
```

### `OptLevel`

Peephole optimization level:

```rust
pub enum OptLevel {
    None,  // No optimizations
    Size,  // Optimize for code size (default)
}
```

---

## One-Shot Functions

### `assemble`

Assemble source text into machine code bytes:

```rust
pub fn assemble(source: &str, arch: Arch) -> Result<Vec<u8>, AsmError>
```

```rust
use asm_rs::{assemble, Arch};
let bytes = assemble("nop\nret", Arch::X86_64).unwrap();
```

### `assemble_at`

Assemble with a base address:

```rust
pub fn assemble_at(source: &str, arch: Arch, base: u64) -> Result<Vec<u8>, AsmError>
```

```rust
use asm_rs::{assemble_at, Arch};
let bytes = assemble_at("nop\nret", Arch::X86_64, 0x400000).unwrap();
```

---

## `Assembler` (Builder API)

### Construction

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
```

### Configuration Methods

| Method | Description |
|:-------|:------------|
| `base_address(addr: u64)` | Set the base address for label resolution |
| `syntax(syntax: Syntax)` | Set the assembly syntax |
| `opt_level(level: OptLevel)` | Set the optimization level |
| `enable_listing()` | Enable listing output generation |
| `define_constant(name, value)` | Define an assembly-time constant |
| `define_external(name, addr)` | Define an external label address |
| `define_preprocessor_symbol(name, value)` | Define a preprocessor symbol |

### Assembly Methods

| Method | Description |
|:-------|:------------|
| `emit(source: &str)` | Assemble source text (can be called multiple times) |
| `finish()` | Finalize and return `AssemblyResult` |

### Data Methods

| Method | Returns | Description |
|:-------|:--------|:------------|
| `ascii(s)` | `Result<&mut Self>` | Emit raw string bytes |
| `asciz(s)` | `Result<&mut Self>` | Emit NUL-terminated string |
| `align(n)` | `&mut Self` | Align to n-byte boundary (multi-byte NOPs) |
| `align_with_fill(n, fill)` | `&mut Self` | Align with explicit fill byte |
| `org(offset)` | `Result<&mut Self>` | Advance to byte offset |
| `org_with_fill(offset, fill)` | `Result<&mut Self>` | Advance with fill byte |
| `fill(count, size, value)` | `Result<&mut Self>` | Emit repeated pattern |
| `space(n)` | `Result<&mut Self>` | Reserve n zero bytes |

### Example

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.base_address(0x401000);
asm.enable_listing();
asm.define_constant("EXIT_CODE", 0);

asm.emit(r#"
    _start:
        xor edi, edi
        mov eax, 60
        syscall
"#).unwrap();

let result = asm.finish().unwrap();
println!("{}", result.listing());
```

---

## `AssemblyResult`

Returned by `Assembler::finish()`:

| Method | Returns | Description |
|:-------|:--------|:------------|
| `bytes()` | `&[u8]` | Final machine code bytes |
| `len()` | `usize` | Output size in bytes |
| `is_empty()` | `bool` | Whether output is empty |
| `labels()` | `&HashMap<String, u64>` | All label names and addresses |
| `label_address(name)` | `Option<u64>` | Look up a specific label |
| `relocations()` | `&[AppliedRelocation]` | Applied relocation records |
| `listing()` | `String` | Human-readable hex dump |
| `base_address()` | `u64` | Base address used |

### `AppliedRelocation`

```rust
pub struct AppliedRelocation {
    pub offset: usize,         // Byte offset in output
    pub size: usize,           // Size of the relocation in bytes
    pub label: String,         // Label name
    pub kind: RelocKind,       // Relocation type
    pub rip_relative: bool,    // Whether RIP-relative
    pub addend: i64,           // Addend value
}
```

---

## `AsmError`

Error types with source location information:

```rust
pub enum AsmError {
    UnknownMnemonic { mnemonic: String, span: Option<Span> },
    InvalidOperands { detail: String, span: Option<Span> },
    UndefinedLabel { label: String, span: Option<Span> },
    DuplicateLabel { label: String, span: Option<Span> },
    BranchOutOfRange { label: String, span: Option<Span> },
    InvalidImmediate { detail: String, span: Option<Span> },
    ParseError { message: String, span: Option<Span> },
    Multiple { errors: Vec<AsmError> },
    // ... additional variants
}
```

### `Span`

Source location tracking:

```rust
pub struct Span {
    pub line: usize,     // 1-based line number
    pub column: usize,   // 1-based column
    pub offset: usize,   // Byte offset from start
    pub length: usize,   // Span length in bytes
}
```

---

## Compile-Time Macros (`asm-rs-macros`)

### `asm_bytes!`

Assemble at compile time, returning `&'static [u8]`:

```rust
use asm_rs_macros::asm_bytes;

const CODE: &[u8] = asm_bytes!(x86_64, "nop; ret");
const BASED: &[u8] = asm_bytes!(x86_64, 0x400000, "nop; ret");
```

### `asm_array!`

Assemble at compile time, returning `[u8; N]`:

```rust
use asm_rs_macros::asm_array;

const NOP: [u8; 1] = asm_array!(x86_64, "nop");
```

### Supported Architecture Tokens

| Token | Architecture |
|:------|:-------------|
| `x86` | x86 (32-bit) |
| `x86_64` | x86-64 |
| `arm` | ARM32 (A32) |
| `thumb` | Thumb/Thumb-2 |
| `aarch64` | AArch64 (A64) |
| `rv32` | RISC-V 32-bit |
| `rv64` | RISC-V 64-bit |

Assembly errors become `compile_error!()` with full diagnostics.
