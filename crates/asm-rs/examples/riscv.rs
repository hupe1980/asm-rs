//! RISC-V assembly example — RV32I and RV64I payloads.
//!
//! Demonstrates assembling RISC-V instructions using `asm_rs`, including
//! function prologues/epilogues, loops, system calls, pseudo-instructions,
//! atomics (A extension), named CSRs, and branch relaxation.
//!
//! Run with: `cargo run --example riscv`

use asm_rs::{assemble, Arch, Assembler};

fn main() {
    println!("=== asm_rs RISC-V assembler ===\n");

    // ── RV32I basic instructions ─────────────────────────────────────────

    println!("1. RV32I Linux exit(0) shellcode:");
    let code = assemble(
        r#"
    li a7, 93           # SYS_exit = 93
    li a0, 0            # exit code = 0
    ecall               # invoke syscall
"#,
        Arch::Rv32,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes\n", code.len());

    // ── RV32I function prologue/epilogue ─────────────────────────────────

    println!("2. RV32I function prologue/epilogue:");
    let code = assemble(
        r#"
    addi sp, sp, -16    # allocate stack frame
    sw   ra, 12(sp)     # save return address
    sw   s0,  8(sp)     # save frame pointer
    addi s0, sp, 16     # set frame pointer

    # ... function body ...
    nop

    lw   s0,  8(sp)     # restore frame pointer
    lw   ra, 12(sp)     # restore return address
    addi sp, sp, 16     # deallocate stack frame
    ret                 # return
"#,
        Arch::Rv32,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes\n", code.len());

    // ── RV32I loop ───────────────────────────────────────────────────────

    println!("3. RV32I loop (sum 1..10):");
    let code = assemble(
        r#"
    li   a0, 0          # accumulator = 0
    li   a1, 1          # counter = 1
    li   a2, 11         # limit = 11
loop:
    add  a0, a0, a1     # accumulator += counter
    addi a1, a1, 1      # counter++
    bne  a1, a2, loop   # while counter != limit
    ret
"#,
        Arch::Rv32,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes\n", code.len());

    // ── RV64I with 64-bit operations ─────────────────────────────────────

    println!("4. RV64I function with 64-bit ops:");
    let code = assemble(
        r#"
    addi sp, sp, -32    # allocate stack frame
    sd   ra, 24(sp)     # save return address (64-bit)
    sd   s0, 16(sp)     # save frame pointer

    # 64-bit arithmetic
    add  a0, a1, a2     # a0 = a1 + a2 (64-bit)
    mul  a0, a0, a3     # a0 *= a3

    ld   s0, 16(sp)     # restore frame pointer
    ld   ra, 24(sp)     # restore return address
    addi sp, sp, 32     # deallocate stack frame
    ret
"#,
        Arch::Rv64,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes\n", code.len());

    // ── Builder API with labels and branches ─────────────────────────────

    println!("5. RV32I Builder API — conditional branch:");
    let mut asm = Assembler::new(Arch::Rv32);
    asm.emit("beqz a0, is_zero").unwrap();
    asm.emit("li a0, 1").unwrap();
    asm.emit("j done").unwrap();
    asm.label("is_zero").unwrap();
    asm.emit("li a0, 0").unwrap();
    asm.label("done").unwrap();
    asm.emit("ret").unwrap();
    let result = asm.finish().unwrap();
    print_hex("   ", result.bytes());
    println!("   Size: {} bytes\n", result.bytes().len());

    // ── Pseudo-instructions ──────────────────────────────────────────────

    println!("6. RV32I pseudo-instructions:");
    let code = assemble(
        r#"
    nop                 # addi x0, x0, 0
    mv   a0, a1        # addi a0, a1, 0
    not  a0, a0        # xori a0, a0, -1
    neg  a0, a0        # sub  a0, x0, a0
    seqz a0, a1        # sltiu a0, a1, 1
    snez a0, a1        # sltu  a0, x0, a1
    jr   ra            # jalr  x0, ra, 0
"#,
        Arch::Rv32,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes\n", code.len());

    // ── M extension (multiply/divide) ────────────────────────────────────

    println!("7. RV32M multiply/divide:");
    let code = assemble(
        r#"
    mul  a0, a1, a2     # a0 = a1 * a2
    div  a3, a0, a4     # a3 = a0 / a4
    rem  a5, a0, a4     # a5 = a0 % a4
"#,
        Arch::Rv32,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes\n", code.len());

    // ── A Extension (Atomics) ────────────────────────────────────────────

    println!("8. RV32A atomic operations:");
    let code = assemble(
        r#"
    lr.w     a0, (a1)          # load-reserved word
    sc.w     a2, a3, (a1)      # store-conditional word
    amoswap.w.aq a0, a3, (a1)  # atomic swap (acquire)
    amoadd.w.rl  a0, a3, (a1)  # atomic add (release)
    amoand.w.aqrl a0, a3, (a1) # atomic AND (acquire+release)
"#,
        Arch::Rv32,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes\n", code.len());

    // ── Named CSR & pseudo-instructions ──────────────────────────────────

    println!("9. RV32I CSR operations with named registers:");
    let code = assemble(
        r#"
    csrr  a0, mstatus          # read mstatus → a0
    csrw  mstatus, a1          # write a1 → mstatus
    csrs  mie, a2              # set bits in mie
    csrc  mip, a3              # clear bits in mip
    csrwi mstatus, 0           # clear mstatus via immediate
"#,
        Arch::Rv32,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes\n", code.len());

    // ── LA pseudo & branch relaxation ────────────────────────────────────

    println!("10. RV32I la pseudo and branch relaxation:");
    let code = assemble(
        r#"
    la   a0, data              # load address (AUIPC + ADDI pair)
    beq  a0, zero, skip        # branch — will be relaxed if needed
    nop
skip:
    ret
data:
"#,
        Arch::Rv32,
    )
    .unwrap();
    print_hex("   ", &code);
    println!("   Size: {} bytes\n", code.len());

    println!("=== Done! ===");
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
