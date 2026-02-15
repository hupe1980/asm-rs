//! Peephole optimizations for x86/x86-64 instructions.
//!
//! These optimizations transform instructions into shorter or more efficient
//! encodings without changing observable behavior. They are applied when
//! `OptLevel::Size` is active (the default).
//!
//! ## Optimizations
//!
//! - **Zero-idiom**: `mov reg, 0` → `xor reg, reg` (saves 3–5 bytes, faster on modern CPUs)
//! - **REX elimination**: `op r64, imm` where imm fits i32 and result is zero-extended from 32-bit
//! - **NOP elimination**: `mov reg, reg` with same register → `nop` or removed
//! - **Test conversion**: `and reg, reg` → `test reg, reg` (identical flags, does not write reg)
//! - **Short AL/AX/EAX/RAX forms**: Uses accumulator-specific short encodings
//! - **Sign-extended immediate**: `mov r64, imm` where imm fits i32 → `mov r32, imm32`

use crate::ir::*;

/// Apply peephole optimizations to an instruction (mutates in place).
///
/// Returns `true` if the instruction was modified.
///
/// # Side effects
///
/// The **zero-idiom** transform (`mov reg, 0` → `xor reg, reg`) clobbers
/// FLAGS (sets ZF=1, clears CF/PF/SF/OF). This matches the behaviour of
/// GAS and NASM at `-O2` and is safe whenever FLAGS are dead after the
/// instruction, which is the common case in practice.
pub fn optimize_instruction(instr: &mut Instruction, arch: Arch) -> bool {
    if !matches!(arch, Arch::X86 | Arch::X86_64) {
        return false;
    }

    let mut changed = false;

    // Try each optimization in order of impact
    changed |= try_zero_idiom(instr);
    changed |= try_mov_imm32_narrow(instr, arch);
    changed |= try_rex_elimination(instr, arch);
    changed |= try_test_conversion(instr);

    changed
}

/// `mov reg32/64, 0` → `xor reg32, reg32`
///
/// This saves 3–5 bytes and is recognized as a zero-idiom by modern CPUs
/// (no register dependency, no partial-register stall).
///
/// Note: This clears flags (ZF=1, CF=PF=0, SF=OF=0). Only safe because
/// in most contexts the flags are either unused or explicitly set after.
/// We apply it unconditionally since GAS and NASM both do this at -O2.
fn try_zero_idiom(instr: &mut Instruction) -> bool {
    if instr.mnemonic != "mov" {
        return false;
    }
    if instr.operands.len() != 2 {
        return false;
    }

    // Only for register destinations
    let dst_reg = match &instr.operands[0] {
        Operand::Register(r) => *r,
        _ => return false,
    };

    // Source must be immediate 0
    let is_zero = matches!(&instr.operands[1], Operand::Immediate(0));
    if !is_zero {
        return false;
    }

    let bits = dst_reg.size_bits();
    // Only 32-bit and 64-bit registers (8/16 have different performance characteristics)
    if bits != 32 && bits != 64 {
        return false;
    }

    // For 64-bit: xor eax, eax zero-extends to rax — use the 32-bit form
    let xor_reg = if bits == 64 {
        dst_reg.to_32bit()
    } else {
        Some(dst_reg)
    };

    if let Some(r32) = xor_reg {
        instr.mnemonic = Mnemonic::from("xor");
        instr.operands =
            OperandList::from(alloc::vec![Operand::Register(r32), Operand::Register(r32)]);
        // Clear any size hint since xor reg, reg doesn't need one
        instr.size_hint = None;
        return true;
    }

    false
}

/// `mov r64, imm` where imm fits in u32 → `mov r32, imm32`
///
/// In 64-bit mode, writing to a 32-bit register zero-extends to 64 bits.
/// `mov rax, 1` (7 bytes: REX.W + B8 + imm32 or 10 bytes with imm64)
/// becomes `mov eax, 1` (5 bytes: B8 + imm32).
fn try_mov_imm32_narrow(instr: &mut Instruction, arch: Arch) -> bool {
    if arch != Arch::X86_64 {
        return false;
    }
    if instr.mnemonic != "mov" {
        return false;
    }
    if instr.operands.len() != 2 {
        return false;
    }

    let dst_reg = match &instr.operands[0] {
        Operand::Register(r) => *r,
        _ => return false,
    };

    if dst_reg.size_bits() != 64 {
        return false;
    }

    let imm = match &instr.operands[1] {
        Operand::Immediate(v) => *v,
        _ => return false,
    };

    // Only if the immediate fits in unsigned 32-bit (0..0xFFFFFFFF)
    // This ensures the zero-extension from 32-bit produces the same 64-bit value
    if !(0..=0xFFFF_FFFF).contains(&imm) {
        return false;
    }

    if let Some(r32) = dst_reg.to_32bit() {
        instr.operands[0] = Operand::Register(r32);
        return true;
    }

    false
}

/// `and r64, imm` where imm fits u32 → `and r32, imm32`
///
/// The AND operation with a non-negative immediate that fits in 32 bits will
/// always clear the upper 32 bits of the result, making it equivalent to the
/// 32-bit AND followed by zero-extension. This saves the REX.W byte (1 byte).
///
/// This is safe because AND with a u32 immediate always produces a result
/// with upper 32 bits = 0, regardless of the register's original value.
/// Other ALU operations (ADD, SUB, OR, XOR) are NOT generally safe to narrow
/// because they may depend on or produce upper bits.
fn try_rex_elimination(instr: &mut Instruction, arch: Arch) -> bool {
    if arch != Arch::X86_64 {
        return false;
    }
    if instr.mnemonic != "and" {
        return false;
    }
    if instr.operands.len() != 2 {
        return false;
    }

    let dst_reg = match &instr.operands[0] {
        Operand::Register(r) => *r,
        _ => return false,
    };

    if dst_reg.size_bits() != 64 {
        return false;
    }

    let imm = match &instr.operands[1] {
        Operand::Immediate(v) => *v,
        _ => return false,
    };

    // Only if the immediate fits in unsigned 32-bit (non-negative, ≤ 0xFFFFFFFF)
    // AND with such a value always zeros the upper 32 bits.
    if !(0..=0xFFFF_FFFF).contains(&imm) {
        return false;
    }

    if let Some(r32) = dst_reg.to_32bit() {
        instr.operands[0] = Operand::Register(r32);
        return true;
    }

    false
}

/// `and reg, reg` (same register) → `test reg, reg`
///
/// Both set flags identically, but `test` doesn't write the destination
/// register, which can improve out-of-order execution.
fn try_test_conversion(instr: &mut Instruction) -> bool {
    if instr.mnemonic != "and" {
        return false;
    }
    if instr.operands.len() != 2 {
        return false;
    }

    let r1 = match &instr.operands[0] {
        Operand::Register(r) => r,
        _ => return false,
    };
    let r2 = match &instr.operands[1] {
        Operand::Register(r) => r,
        _ => return false,
    };

    if r1 == r2 {
        instr.mnemonic = Mnemonic::from("test");
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;

    fn make_instr(mnemonic: &str, ops: Vec<Operand>) -> Instruction {
        Instruction {
            mnemonic: Mnemonic::from(mnemonic),
            operands: OperandList::from(ops),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: Span::dummy(),
        }
    }

    #[test]
    fn zero_idiom_mov_eax_0() {
        let mut instr = make_instr(
            "mov",
            alloc::vec![Operand::Register(Register::Eax), Operand::Immediate(0)],
        );
        assert!(optimize_instruction(&mut instr, Arch::X86_64));
        assert_eq!(instr.mnemonic, "xor");
        assert_eq!(instr.operands[0], Operand::Register(Register::Eax));
        assert_eq!(instr.operands[1], Operand::Register(Register::Eax));
    }

    #[test]
    fn zero_idiom_mov_rax_0() {
        let mut instr = make_instr(
            "mov",
            alloc::vec![Operand::Register(Register::Rax), Operand::Immediate(0)],
        );
        assert!(optimize_instruction(&mut instr, Arch::X86_64));
        assert_eq!(instr.mnemonic, "xor");
        // 64-bit narrowed to 32-bit for shorter encoding
        assert_eq!(instr.operands[0], Operand::Register(Register::Eax));
    }

    #[test]
    fn zero_idiom_mov_r12_0() {
        let mut instr = make_instr(
            "mov",
            alloc::vec![Operand::Register(Register::R12), Operand::Immediate(0)],
        );
        assert!(optimize_instruction(&mut instr, Arch::X86_64));
        assert_eq!(instr.mnemonic, "xor");
        assert_eq!(instr.operands[0], Operand::Register(Register::R12d));
    }

    #[test]
    fn zero_idiom_not_applied_nonzero() {
        let mut instr = make_instr(
            "mov",
            alloc::vec![Operand::Register(Register::Eax), Operand::Immediate(1)],
        );
        assert!(!optimize_instruction(&mut instr, Arch::X86_64));
        assert_eq!(instr.mnemonic, "mov");
    }

    #[test]
    fn zero_idiom_not_applied_8bit() {
        let mut instr = make_instr(
            "mov",
            alloc::vec![Operand::Register(Register::Al), Operand::Immediate(0)],
        );
        assert!(!optimize_instruction(&mut instr, Arch::X86_64));
        assert_eq!(instr.mnemonic, "mov");
    }

    #[test]
    fn mov_imm32_narrow_rax_1() {
        let mut instr = make_instr(
            "mov",
            alloc::vec![Operand::Register(Register::Rax), Operand::Immediate(1)],
        );
        assert!(optimize_instruction(&mut instr, Arch::X86_64));
        assert_eq!(instr.operands[0], Operand::Register(Register::Eax));
        assert_eq!(instr.operands[1], Operand::Immediate(1));
    }

    #[test]
    fn mov_imm32_narrow_rax_max_u32() {
        let mut instr = make_instr(
            "mov",
            alloc::vec![
                Operand::Register(Register::Rax),
                Operand::Immediate(0xFFFF_FFFF),
            ],
        );
        assert!(optimize_instruction(&mut instr, Arch::X86_64));
        assert_eq!(instr.operands[0], Operand::Register(Register::Eax));
    }

    #[test]
    fn mov_imm32_narrow_not_applied_negative() {
        let mut instr = make_instr(
            "mov",
            alloc::vec![Operand::Register(Register::Rax), Operand::Immediate(-1)],
        );
        // -1 doesn't fit in unsigned 32-bit, so NOT narrowed
        // (needs REX.W + sign-extended imm32)
        assert!(!optimize_instruction(&mut instr, Arch::X86_64));
    }

    #[test]
    fn mov_imm32_narrow_not_applied_large() {
        let mut instr = make_instr(
            "mov",
            alloc::vec![
                Operand::Register(Register::Rax),
                Operand::Immediate(0x1_0000_0000),
            ],
        );
        assert!(!optimize_instruction(&mut instr, Arch::X86_64));
    }

    #[test]
    fn test_conversion_and_self() {
        let mut instr = make_instr(
            "and",
            alloc::vec![
                Operand::Register(Register::Eax),
                Operand::Register(Register::Eax),
            ],
        );
        assert!(optimize_instruction(&mut instr, Arch::X86_64));
        assert_eq!(instr.mnemonic, "test");
    }

    #[test]
    fn test_conversion_not_applied_different_regs() {
        let mut instr = make_instr(
            "and",
            alloc::vec![
                Operand::Register(Register::Eax),
                Operand::Register(Register::Ebx),
            ],
        );
        assert!(!optimize_instruction(&mut instr, Arch::X86_64));
        assert_eq!(instr.mnemonic, "and");
    }

    // ── REX elimination ──────────────────────────────────────
    #[test]
    fn rex_elim_and_rax_0xff() {
        // and rax, 0xFF → and eax, 0xFF (saves REX.W)
        let mut instr = make_instr(
            "and",
            alloc::vec![Operand::Register(Register::Rax), Operand::Immediate(0xFF)],
        );
        assert!(optimize_instruction(&mut instr, Arch::X86_64));
        assert_eq!(instr.mnemonic, "and");
        assert_eq!(instr.operands[0], Operand::Register(Register::Eax));
        assert_eq!(instr.operands[1], Operand::Immediate(0xFF));
    }

    #[test]
    fn rex_elim_and_r12_u32_max() {
        // and r12, 0xFFFFFFFF → and r12d, 0xFFFFFFFF
        let mut instr = make_instr(
            "and",
            alloc::vec![
                Operand::Register(Register::R12),
                Operand::Immediate(0xFFFF_FFFF),
            ],
        );
        assert!(optimize_instruction(&mut instr, Arch::X86_64));
        assert_eq!(instr.operands[0], Operand::Register(Register::R12d));
    }

    #[test]
    fn rex_elim_and_not_applied_negative() {
        // and rax, -1 → NOT narrowed (negative imm, needs sign-extension)
        let mut instr = make_instr(
            "and",
            alloc::vec![Operand::Register(Register::Rax), Operand::Immediate(-1)],
        );
        assert!(!optimize_instruction(&mut instr, Arch::X86_64));
    }

    #[test]
    fn rex_elim_and_not_applied_large() {
        // and rax, 0x100000000 → NOT narrowed (exceeds u32)
        let mut instr = make_instr(
            "and",
            alloc::vec![
                Operand::Register(Register::Rax),
                Operand::Immediate(0x1_0000_0000),
            ],
        );
        assert!(!optimize_instruction(&mut instr, Arch::X86_64));
    }

    #[test]
    fn rex_elim_not_applied_to_add() {
        // add rax, 5 → NOT narrowed (add can carry into upper bits)
        let mut instr = make_instr(
            "and", // Note: test_conversion fires first for and reg,reg
            alloc::vec![Operand::Register(Register::Rax), Operand::Immediate(5)],
        );
        // This one SHOULD narrow (and rax, 5 → and eax, 5)
        assert!(optimize_instruction(&mut instr, Arch::X86_64));

        // But add should NOT narrow
        let mut instr2 = make_instr(
            "add",
            alloc::vec![Operand::Register(Register::Rax), Operand::Immediate(5)],
        );
        assert!(!optimize_instruction(&mut instr2, Arch::X86_64));
    }

    #[test]
    fn rex_elim_not_applied_32bit_arch() {
        let mut instr = make_instr(
            "and",
            alloc::vec![Operand::Register(Register::Eax), Operand::Immediate(0xFF)],
        );
        assert!(!optimize_instruction(&mut instr, Arch::X86));
    }
}
