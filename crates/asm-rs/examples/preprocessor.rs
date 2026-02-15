//! Preprocessor example — demonstrates macros, loops, and conditional assembly.
//!
//! Run with: `cargo run --example preprocessor`

use asm_rs::{Arch, Assembler};

fn main() {
    println!("=== asm_rs preprocessor example ===\n");

    // --- Macro definitions ---
    println!("1. Macro-generated function prologue/epilogue:");
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit(
        r#"
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
    # ... function body ...
    xor eax, eax
    epilogue 0x20
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    println!("   Listing:");
    for line in result.listing().lines() {
        println!("   {}", line);
    }

    // --- .rept for NOP sleds ---
    println!("\n2. .rept for padding:");
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit(
        r#"
start:
    nop
.rept 5
    nop
.endr
    ret
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    print_hex("   ", result.bytes());
    println!("   ({} bytes)", result.len());

    // --- .irp for register push/pop ---
    println!("\n3. .irp for register save/restore:");
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit(
        r#"
.macro save_regs regs:vararg
.irp reg, \regs
    push \reg
.endr
.endm

.macro restore_regs regs:vararg
.irp reg, \regs
    pop \reg
.endr
.endm

save_regs rbx, r12, r13, r14, r15
    # ... callee-saved registers saved ...
    nop
restore_regs r15, r14, r13, r12, rbx
    ret
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    println!("   Listing:");
    for line in result.listing().lines() {
        println!("   {}", line);
    }

    // --- Conditional assembly ---
    println!("\n4. Conditional assembly with .ifdef:");
    let mut asm = Assembler::new(Arch::X86_64);
    asm.define_preprocessor_symbol("DEBUG", 1);
    asm.emit(
        r#"
.ifdef DEBUG
    int 3       # breakpoint in debug builds
.endif
    xor eax, eax
    ret
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    println!("   With DEBUG defined:");
    print_hex("   ", result.bytes());

    let mut asm = Assembler::new(Arch::X86_64);
    // No DEBUG symbol defined
    asm.emit(
        r#"
.ifdef DEBUG
    int 3
.endif
    xor eax, eax
    ret
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    println!("   Without DEBUG:");
    print_hex("   ", result.bytes());

    // --- Nested constructs ---
    println!("\n5. Nested .rept inside .irp:");
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit(
        r#"
.irp reg, rax, rbx
.rept 2
    push \reg
.endr
.endr
    ret
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    print_hex("   ", result.bytes());
    println!("   ({} bytes: 2x push rax, 2x push rbx, ret)", result.len());

    println!("\n=== Done! ===");
}

fn print_hex(prefix: &str, bytes: &[u8]) {
    print!("{}", prefix);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && i % 16 == 0 {
            println!();
            print!("{}", prefix);
        }
        print!("{:02X} ", b);
    }
    println!();
}
