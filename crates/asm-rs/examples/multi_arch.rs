//! Multi-architecture shellcode example — ARM32 and AArch64 payloads.
//!
//! Demonstrates assembling common shellcode patterns across architectures
//! using the same `asm_rs` API.
//!
//! Run with: `cargo run --example multi_arch`

use asm_rs::{assemble, Arch, Assembler};

fn main() {
    println!("=== asm_rs multi-architecture assembler ===\n");

    // ── AArch64 (ARM64) ─────────────────────────────────────────────────

    println!("1. AArch64 Linux exit(0) shellcode:");
    let code = assemble(
        r#"
    mov x0, 0          # exit code = 0
    mov x8, 93          # SYS_exit = 93
    svc 0               # syscall
"#,
        Arch::Aarch64,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes", code.len());

    println!("\n2. AArch64 function prologue/epilogue:");
    let code = assemble(
        r#"
    stp x29, x30, [sp, -16]!   # push frame pointer + link register
    mov x29, sp                  # set up frame pointer
    # ... function body ...
    mov x0, 0                   # return value
    ldp x29, x30, [sp], 16     # pop frame pointer + link register
    ret                          # return to caller
"#,
        Arch::Aarch64,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes", code.len());

    println!("\n3. AArch64 branch and conditional pattern:");
    let code = assemble(
        r#"
    cmp x0, 0
    b.eq zero_case
    mov x1, 1
    b done
zero_case:
    mov x1, 0
done:
    ret
"#,
        Arch::Aarch64,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes", code.len());

    println!("\n4. AArch64 bitwise operations:");
    let code = assemble(
        r#"
    mvn x1, x0          # bitwise NOT
    clz x2, x0          # count leading zeros
    rbit x3, x0         # reverse bits
    rev x4, x0          # reverse bytes
    and x5, x0, 0xff    # mask lower byte
    uxtb w6, w0         # zero-extend byte
    cset x7, eq         # conditional set
"#,
        Arch::Aarch64,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes", code.len());

    println!("\n5. AArch64 arithmetic:");
    let code = assemble(
        r#"
    mul x2, x0, x1       # multiply
    sdiv x3, x2, x1      # signed divide
    udiv x4, x2, x1      # unsigned divide
    madd x5, x0, x1, x2  # multiply-add: x5 = x0*x1 + x2
"#,
        Arch::Aarch64,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes", code.len());

    println!("\n6. AArch64 system register access:");
    let code = assemble(
        r#"
    mrs x0, nzcv         # read condition flags
    dmb ish               # data memory barrier (inner shareable)
    isb                   # instruction synchronization barrier
    msr nzcv, x0         # write condition flags
"#,
        Arch::Aarch64,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes", code.len());

    // ── ARM32 ────────────────────────────────────────────────────────────

    println!("\n7. ARM32 Linux exit(0) shellcode:");
    let code = assemble(
        r#"
    mov r0, 0           # exit code = 0
    mov r7, 1           # SYS_exit = 1
    svc 0               # syscall
"#,
        Arch::Arm,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes", code.len());

    println!("\n8. ARM32 branches with labels:");
    let code = assemble(
        r#"
    cmp r0, 0
    beq zero
    mov r1, 1
    b done
zero:
    mov r1, 0
done:
    bx lr
"#,
        Arch::Arm,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes", code.len());

    println!("\n9. ARM32 bit manipulation:");
    let code = assemble(
        r#"
    clz r1, r0          # count leading zeros
    rev r2, r0          # reverse bytes
    rev16 r3, r0        # reverse bytes in halfwords
    rbit r4, r0         # reverse bits
    uxtb r5, r0         # zero-extend byte
    sxtb r6, r0         # sign-extend byte
"#,
        Arch::Arm,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes", code.len());

    println!("\n10. ARM32 load/store halfword:");
    let code = assemble(
        r#"
    ldrh r0, [r1]          # load unsigned halfword
    strh r0, [r1, 4]       # store halfword at offset
    ldrsb r2, [r1]         # load signed byte
    ldrsh r3, [r1, 2]      # load signed halfword at offset
"#,
        Arch::Arm,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes", code.len());

    // ── Multi-arch comparison ────────────────────────────────────────────

    println!("\n11. Same operation, three architectures:");
    let arches = [
        ("x86-64", Arch::X86_64, "xor eax, eax\nret"),
        ("AArch64", Arch::Aarch64, "mov x0, 0\nret"),
        ("ARM32", Arch::Arm, "mov r0, 0\nbx lr"),
    ];
    for (name, arch, src) in &arches {
        let code = assemble(src, *arch).unwrap();
        print!("   {:<10} return 0: ", name);
        for b in &code {
            print!("{:02X} ", b);
        }
        println!("({} bytes)", code.len());
    }

    // ── Using the Assembler API with macros ──────────────────────────────

    println!("\n12. AArch64 with preprocessor macros:");
    let mut asm = Assembler::new(Arch::Aarch64);
    asm.emit(
        r#"
.macro syscall_exit code
    mov x0, \code
    mov x8, 93
    svc 0
.endm

.macro push_pair r1, r2
    stp \r1, \r2, [sp, -16]!
.endm

.macro pop_pair r1, r2
    ldp \r1, \r2, [sp], 16
.endm

    push_pair x29, x30
    mov x29, sp
    # ... do work ...
    pop_pair x29, x30
    syscall_exit 0
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    print_hex("   ", result.bytes());
    println!("   Size: {} bytes", result.bytes().len());

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
