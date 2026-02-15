//! Regression tests for bug fixes.
//!
//! Each test documents a specific bug that was found and fixed, ensuring the
//! fix is never accidentally reverted.

use asm_rs::{assemble, Arch, Assembler};

/// Regression: x86-32 INC/DEC short forms (0x40–0x4F) must use the
/// single-byte opcode, not the ModR/M form used in x86-64 (where 0x40–0x4F
/// are REX prefixes).
#[test]
#[cfg(feature = "x86")]
fn x86_32_inc_dec_short_forms() {
    let mut asm = Assembler::new(Arch::X86);
    asm.emit("inc eax\ndec ebx").unwrap();
    let result = asm.finish().unwrap();
    // INC EAX = 0x40, DEC EBX = 0x4B (short forms)
    assert_eq!(result.bytes(), &[0x40, 0x4B]);
}

/// Regression: MOV reg, 0 is optimized to XOR reg, reg by the peephole
/// optimizer (zero idiom). This is the default behavior — the encoder
/// produces the shorter 2-byte form [0x31, 0xC9] instead of the 5-byte
/// MOV ECX, imm32 encoding.
#[test]
#[cfg(feature = "x86_64")]
fn x86_64_mov_reg_zero_optimized_to_xor() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit("mov ecx, 0").unwrap();
    let result = asm.finish().unwrap();
    // Peephole: XOR ECX, ECX = 31 C9 (2 bytes, zero idiom)
    assert_eq!(result.bytes(), &[0x31, 0xC9]);
}

/// Regression: RIP-relative addressing must correctly compute displacement
/// from the end of the current instruction, not from the start.
#[test]
#[cfg(feature = "x86_64")]
fn rip_relative_displacement_from_instruction_end() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit("lea rax, [rip + 0]\nnop").unwrap();
    let result = asm.finish().unwrap();
    // LEA RAX, [RIP+0] = 48 8D 05 00 00 00 00 (7 bytes), NOP = 90
    assert_eq!(
        &result.bytes()[..7],
        &[0x48, 0x8D, 0x05, 0x00, 0x00, 0x00, 0x00]
    );
}

/// Regression: ARM32 NOP must encode as MOV R0, R0 (pre-ARMv6K encoding),
/// not the ARMv6K+ hint NOP.
#[test]
#[cfg(feature = "arm")]
fn arm32_nop_is_mov_r0_r0() {
    let code = assemble("nop", Arch::Arm).unwrap();
    assert_eq!(code, &[0x00, 0x00, 0xA0, 0xE1]);
}
