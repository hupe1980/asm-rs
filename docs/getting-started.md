---
layout: default
title: Getting Started
nav_order: 2
---

# Getting Started
{: .fs-8 }

Get up and running with asm-rs in minutes.
{: .fs-5 .fw-300 }

---

## Installation

Add asm-rs to your `Cargo.toml`:

```toml
[dependencies]
asm-rs = "0.1"
```

**Minimum Supported Rust Version**: 1.75

### Optional: Compile-Time Macros

For zero-overhead compile-time assembly:

```toml
[dependencies]
asm-rs = "0.1"
asm-rs-macros = "0.1"
```

---

## One-Shot Assembly

The simplest way to assemble code — pass source text and get machine code bytes:

```rust
use asm_rs::{assemble, Arch};

let bytes = assemble("mov eax, 42\nret", Arch::X86_64).unwrap();
assert_eq!(bytes[0], 0xB8); // mov eax, imm32
```

### With Base Address

```rust
use asm_rs::{assemble_at, Arch};

let bytes = assemble_at("nop\nret", Arch::X86_64, 0x400000).unwrap();
```

---

## Builder API

The builder API offers fine-grained control over the assembly process:

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

### AssemblyResult

The `finish()` method returns an `AssemblyResult` with rich output:

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.base_address(0x401000);
asm.emit("entry:\npush rbp\nmov rbp, rsp\npop rbp\nret").unwrap();
let result = asm.finish().unwrap();

// Machine code bytes
let bytes: &[u8] = result.bytes();

// Label addresses
assert_eq!(result.label_address("entry"), Some(0x401000));

// Applied relocations
for reloc in result.relocations() {
    println!("offset={} label={} kind={:?}",
        reloc.offset, reloc.label, reloc.kind);
}

// Human-readable listing
println!("{}", result.listing());
// 00401000                  entry:
// 00401000  55                push rbp
// 00401001  4889E5            mov rbp, rsp
// 00401004  5D                pop rbp
// 00401005  C3                ret
```

---

## AT&T / GAS Syntax

asm-rs supports full AT&T/GAS syntax (`%reg`, `$imm`, reversed operand order):

```rust
use asm_rs::{Assembler, Arch, Syntax};

let mut asm = Assembler::new(Arch::X86_64);
asm.syntax(Syntax::Att);
asm.emit(r#"
    pushq %rbp
    movq %rsp, %rbp
    subq $0x20, %rsp
    movl $42, %eax
    addq $0x20, %rsp
    popq %rbp
    ret
"#).unwrap();
let result = asm.finish().unwrap();
```

You can also switch syntax mid-stream with `.syntax att` / `.syntax intel` directives.

---

## Labels & Control Flow

Forward and backward references are resolved automatically:

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.emit(r#"
    mov ecx, 10
loop_top:
    dec ecx
    jnz loop_top
    ret
"#).unwrap();
let result = asm.finish().unwrap();
```

### Branch Relaxation

Short branches (rel8, 2 bytes) are preferred; they are automatically promoted to
long form (rel32, 5–6 bytes) only when the target is out of range:

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.emit("je done\ndone:\nret").unwrap();
let result = asm.finish().unwrap();
// Short form: 74 00 C3 (3 bytes, not 0F 84 xx xx xx xx C3)
assert_eq!(result.bytes(), &[0x74, 0x00, 0xC3]);
```

---

## Constants

Define constants with `.equ`, `.set`, or `name = expression`:

```rust
use asm_rs::{assemble, Arch};

let bytes = assemble(r#"
    .equ SYS_EXIT, 60
    mov eax, SYS_EXIT
    xor edi, edi
    syscall
"#, Arch::X86_64).unwrap();
```

Or via the builder API:

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.define_constant("SYS_EXIT", 60);
asm.emit("mov eax, SYS_EXIT\nsyscall").unwrap();
let result = asm.finish().unwrap();
```

---

## Data Directives

Embed raw data alongside code:

```rust
use asm_rs::{assemble, Arch};

let bytes = assemble(r#".asciz "Hello, World!""#, Arch::X86_64).unwrap();
assert_eq!(&bytes[..13], b"Hello, World!");

let bytes = assemble(".byte 0xDE, 0xAD, 0xBE, 0xEF", Arch::X86_64).unwrap();
assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]);
```

### Builder Convenience Methods

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.emit("push rbp").unwrap();
asm.ascii("Hello");                      // raw string bytes
asm.asciz("World");                      // NUL-terminated string
asm.align(16);                           // align to 16-byte boundary
asm.align_with_fill(8, 0xCC);           // align with explicit fill byte
asm.org(0x100);                          // advance to offset
asm.fill(4, 1, 0x90);                   // emit 4 NOP bytes
asm.space(16);                           // reserve 16 zero bytes
let result = asm.finish().unwrap();
```

---

## Preprocessor

The built-in preprocessor supports macros, loops, and conditional assembly:

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.emit(r#"
.macro prologue frame_size=0
    push rbp
    mov rbp, rsp
    .if \frame_size
    sub rsp, \frame_size
    .endif
.endm

.macro epilogue frame_size=0
    .if \frame_size
    add rsp, \frame_size
    .endif
    pop rbp
    ret
.endm

func:
    prologue 0x20
    xor eax, eax
    epilogue 0x20
"#).unwrap();
let result = asm.finish().unwrap();
```

### Repeat Loops & Conditional Assembly

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.define_preprocessor_symbol("DEBUG", 1);
asm.emit(r#"
.ifdef DEBUG
    int 3
.endif

.irp reg, rbx, r12, r13, r14, r15
    push \reg
.endr
    nop
.irp reg, r15, r14, r13, r12, rbx
    pop \reg
.endr
    ret
"#).unwrap();
let result = asm.finish().unwrap();
```

---

## Compile-Time Assembly

The `asm-rs-macros` crate provides proc-macros for zero-overhead compile-time assembly:

```rust
use asm_rs_macros::{asm_bytes, asm_array};

// Assemble at compile time → &'static [u8]
const SHELLCODE: &[u8] = asm_bytes!(x86_64, "
    xor eax, eax
    inc eax
    ret
");

// Fixed-size array variant → [u8; N]
const NOP: [u8; 1] = asm_array!(x86_64, "nop");

// All architectures supported
const ARM_CODE: &[u8] = asm_bytes!(arm, "bx lr");
const RV_CODE: &[u8] = asm_bytes!(rv64, "nop\nret");
```

Assembly errors become compile-time errors with full diagnostics.

---

## Shellcode Examples

### x86-64

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::X86_64);
asm.emit(r#"
    xor eax, eax
    xor edi, edi
    mov al, 60          # __NR_exit
    syscall
"#).unwrap();
let result = asm.finish().unwrap();
```

### ARM32

```rust
use asm_rs::{Assembler, Arch};

let mut asm = Assembler::new(Arch::Arm);
asm.emit(r#"
    mov r0, #0          @ status = 0
    mov r7, #1          @ __NR_exit
    svc #0              @ syscall
"#).unwrap();
let result = asm.finish().unwrap();
```

### AArch64

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

### RISC-V

```rust
use asm_rs::{assemble, Arch};

let bytes = assemble(r#"
    li a7, 93           # SYS_exit = 93
    li a0, 0            # exit code = 0
    ecall               # syscall
"#, Arch::Rv32).unwrap();
```

---

## Error Handling

All errors include source location information:

```rust
use asm_rs::{assemble, Arch, AsmError};

match assemble("foobar", Arch::X86_64) {
    Err(AsmError::UnknownMnemonic { mnemonic, .. }) => {
        println!("Unknown: {}", mnemonic);
    }
    Err(AsmError::InvalidOperands { detail, .. }) => {
        println!("Bad operands: {}", detail);
    }
    Err(AsmError::UndefinedLabel { label, .. }) => {
        println!("Undefined: {}", label);
    }
    Err(AsmError::Multiple { errors }) => {
        println!("{} errors collected:", errors.len());
        for e in &errors { println!("  - {}", e); }
    }
    Err(e) => println!("Error: {}", e),
    Ok(bytes) => println!("OK: {} bytes", bytes.len()),
}
```

---

## Next Steps

- [Architecture Guide]({{ site.baseurl }}/architecture) — deep dive into the assembler pipeline
- [Configuration]({{ site.baseurl }}/configuration) — Cargo features and runtime options
- [ISA References]({{ site.baseurl }}/reference/) — instruction references for all architectures
- [API Reference]({{ site.baseurl }}/api-reference) — complete API documentation
