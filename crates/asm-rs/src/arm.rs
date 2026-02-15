//! ARM32 (A32) instruction encoder.
//!
//! Implements encoding for ARMv7 A32 instructions in ARM mode (32-bit
//! fixed-width). Covers the core instruction set used in offensive security
//! payloads: data processing (with barrel shifter), load/store (with
//! pre/post-index), branches, multiply (32-bit and 64-bit long), and
//! system calls.
//!
//! ## ARM32 Instruction Encoding
//!
//! All A32 instructions are 32 bits wide. The top 4 bits contain the
//! condition code (0xE = AL = always execute). The remaining 28 bits
//! encode the operation, registers, and immediates.
//!
//! ```text
//! 31..28  27..25  24..21  20  19..16  15..12  11..0
//! cond    op1     opcode  S   Rn      Rd      operand2
//! ```
//!
//! ## Barrel Shifter (Flexible Operand2)
//!
//! Data-processing instructions accept a shifted register as operand2:
//! ```text
//!   ADD R0, R1, R2, LSL, 3    ; R0 = R1 + (R2 << 3)
//!   SUB R3, R4, R5, ASR, 8    ; R3 = R4 - (R5 >> 8) [arithmetic]
//!   MOV R0, R1, ROR, 16       ; R0 = R1 rotated right 16
//!   MOV R0, R1, RRX           ; R0 = R1 rotate-right-extend (through carry)
//! ```
//!
//! Shift types: LSL, LSR, ASR, ROR (with comma-separated amount), RRX (no amount).

use alloc::string::String;

use crate::encoder::{EncodedInstr, InstrBytes, RelocKind, Relocation};
use crate::error::AsmError;
use crate::ir::*;

// ── Condition codes ──────────────────────────────────────────────────────

/// ARM condition code (4 bits, placed in bits 31..28).
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum Cond {
    Eq = 0x0,
    Ne = 0x1,
    Cs = 0x2, // HS
    Cc = 0x3, // LO
    Mi = 0x4,
    Pl = 0x5,
    Vs = 0x6,
    Vc = 0x7,
    Hi = 0x8,
    Ls = 0x9,
    Ge = 0xA,
    Lt = 0xB,
    Gt = 0xC,
    Le = 0xD,
    Al = 0xE,
}

/// Parse condition suffix from mnemonic.
/// Returns (base_mnemonic, condition_code).
///
/// ARM condition suffixes are 2-char codes appended to the mnemonic base.
/// However, some instruction mnemonics end in what looks like a condition code
/// (e.g., `umlal` ends with "al", `smlal` ends with "al"). We resolve this
/// by checking whether the stripped base is a known instruction; if not, the
/// full mnemonic is used as the base.
fn parse_cond(mnemonic: &str) -> (&str, Cond) {
    if mnemonic.len() < 4 {
        // Mnemonic must be at least 4 chars to have a 2-char base + 2-char condition.
        // This prevents false matches like "svc" → ("s", Vc) or "bne" → ("b", Ne)
        // when the base would be too short. For 3-char mnemonics like "bne",
        // we handle them explicitly below.
        if mnemonic.len() == 3 {
            // Special case: single-char base + 2-char condition (only "b" + cond)
            let (base, suffix) = mnemonic.split_at(1);
            if base == "b" {
                return match suffix {
                    "eq" => (base, Cond::Eq),
                    "ne" => (base, Cond::Ne),
                    "cs" | "hs" => (base, Cond::Cs),
                    "cc" | "lo" => (base, Cond::Cc),
                    "mi" => (base, Cond::Mi),
                    "pl" => (base, Cond::Pl),
                    "vs" => (base, Cond::Vs),
                    "vc" => (base, Cond::Vc),
                    "hi" => (base, Cond::Hi),
                    "ls" => (base, Cond::Ls),
                    "ge" => (base, Cond::Ge),
                    "lt" => (base, Cond::Lt),
                    "gt" => (base, Cond::Gt),
                    "le" => (base, Cond::Le),
                    "al" => (base, Cond::Al),
                    _ => (mnemonic, Cond::Al),
                };
            }
        }
        return (mnemonic, Cond::Al);
    }

    // First check whether the full mnemonic is a known instruction.
    // If so, don't strip a condition suffix (avoids "umlal" → "uml"+AL).
    if is_known_base(mnemonic) {
        return (mnemonic, Cond::Al);
    }

    // Check for 2-char condition suffix
    let (base, suffix) = mnemonic.split_at(mnemonic.len() - 2);
    match suffix {
        "eq" => (base, Cond::Eq),
        "ne" => (base, Cond::Ne),
        "cs" | "hs" => (base, Cond::Cs),
        "cc" | "lo" => (base, Cond::Cc),
        "mi" => (base, Cond::Mi),
        "pl" => (base, Cond::Pl),
        "vs" => (base, Cond::Vs),
        "vc" => (base, Cond::Vc),
        "hi" => (base, Cond::Hi),
        "ls" => (base, Cond::Ls),
        "ge" => (base, Cond::Ge),
        "lt" => (base, Cond::Lt),
        "gt" => (base, Cond::Gt),
        "le" => (base, Cond::Le),
        "al" => (base, Cond::Al),
        _ => (mnemonic, Cond::Al),
    }
}

/// Check if a mnemonic (with optional trailing 's' for set-flags) is a known
/// ARM instruction base. Used by `parse_cond` to avoid incorrectly stripping
/// condition suffixes from instruction names that happen to end in a
/// condition-like sequence (e.g., `umlal`, `smlal`, `stmda`, `ldmda`).
fn is_known_base(m: &str) -> bool {
    // Strip trailing 's' to check both `umull` and `umulls`
    let core = if m.ends_with('s') && m.len() > 1 {
        &m[..m.len() - 1]
    } else {
        m
    };
    matches!(
        core,
        // Data processing (handled by dp_opcode)
        "and" | "eor" | "sub" | "rsb" | "add" | "adc" | "sbc" | "rsc"
        | "tst" | "teq" | "cmp" | "cmn" | "orr" | "mov" | "bic" | "mvn"
        // Multiply
        | "mul" | "mla" | "umull" | "smull" | "umlal" | "smlal"
        // Load/store
        | "ldr" | "str" | "ldrb" | "strb" | "ldrh" | "strh" | "ldrsb" | "ldrsh"
        | "ldrex" | "ldrexb" | "ldrexh" | "ldrexd"
        | "strex" | "strexb" | "strexh" | "strexd"
        // LDM/STM variants
        | "ldm" | "ldmia" | "ldmfd" | "ldmdb" | "ldmea" | "ldmib" | "ldmed"
        | "ldmda" | "ldmfa" | "stm" | "stmia" | "stmea" | "stmdb" | "stmfd"
        | "stmib" | "stmfa" | "stmda" | "stmed"
        // Branch
        | "b" | "bl" | "bx" | "blx"
        // Stack
        | "push" | "pop"
        // Misc
        | "nop" | "bkpt" | "svc" | "swi" | "adr" | "clz"
        | "rev" | "rev16" | "rbit"
        | "bfc" | "bfi" | "sbfx" | "ubfx"
        | "uxtb" | "uxth" | "sxtb" | "sxth"
        | "mrs" | "msr"
        | "movw" | "movt"
        | "dmb" | "dsb" | "isb"
    )
}

// ── Helper functions ─────────────────────────────────────────────────────

fn invalid_ops(mnemonic: &str, detail: &str, span: crate::error::Span) -> AsmError {
    AsmError::InvalidOperands {
        detail: alloc::format!("{}: {}", mnemonic, detail),
        span,
    }
}

/// Extract ARM register from operand, or return error.
fn get_arm_reg(
    op: &Operand,
    mnemonic: &str,
    span: crate::error::Span,
) -> Result<Register, AsmError> {
    match op {
        Operand::Register(r) if r.is_arm() => Ok(*r),
        _ => Err(invalid_ops(mnemonic, "expected ARM register", span)),
    }
}

/// Extract immediate from operand, or return error.
fn get_imm(op: &Operand, mnemonic: &str, span: crate::error::Span) -> Result<i128, AsmError> {
    match op {
        Operand::Immediate(v) => Ok(*v),
        _ => Err(invalid_ops(mnemonic, "expected immediate", span)),
    }
}

/// Encode an 8-bit rotated immediate for ARM A32 encoding.
/// Returns `Some((imm8, rotate))` where the value = imm8 ROR (rotate * 2),
/// or `None` if the value cannot be encoded.
fn encode_arm_imm(value: u32) -> Option<(u8, u8)> {
    for rot in 0..16u8 {
        let shift = rot * 2;
        let rotated = value.rotate_left(shift as u32);
        if rotated <= 0xFF {
            return Some((rotated as u8, rot));
        }
    }
    None
}

/// Build a register list bitmask from a list of ARM registers.
fn reg_list_mask(regs: &[Register]) -> u16 {
    let mut mask = 0u16;
    for r in regs {
        mask |= 1 << r.arm_reg_num();
    }
    mask
}

// ── Barrel Shifter ───────────────────────────────────────────────────────

/// ARM shift type for operand2 barrel shifter (bits 6..5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ShiftType {
    Lsl = 0b00,
    Lsr = 0b01,
    Asr = 0b10,
    Ror = 0b11,
}

/// Parsed barrel-shifter info: shift type + immediate amount OR register.
/// For RRX, shift_type = ROR with amount = 0, reg_shift = None.
#[derive(Debug, Clone, Copy)]
struct ShiftInfo {
    shift_type: ShiftType,
    amount: u8,                  // 0–31 for immediate shifts
    reg_shift: Option<Register>, // Some(Rs) for register-based shifts
}

/// Try to parse a barrel-shift suffix from the operand slice starting at `idx`.
///
/// Recognises patterns like:
///   `Label("lsl"), Immediate(3)` → LSL #3  (immediate shift)
///   `Label("lsl"), Register(r3)` → LSL R3  (register shift)
///   `Label("rrx")`               → RRX (rotate right extend)
///
/// Returns `Some((ShiftInfo, count_of_operands_consumed))` or `None`.
fn parse_shift(ops: &[Operand], idx: usize) -> Option<(ShiftInfo, usize)> {
    if idx >= ops.len() {
        return None;
    }
    let name = match &ops[idx] {
        Operand::Label(s) => s.as_str(),
        _ => return None,
    };
    match name {
        "lsl" | "lsr" | "asr" | "ror" => {
            let st = match name {
                "lsl" => ShiftType::Lsl,
                "lsr" => ShiftType::Lsr,
                "asr" => ShiftType::Asr,
                "ror" => ShiftType::Ror,
                _ => return None,
            };
            if idx + 1 < ops.len() {
                match &ops[idx + 1] {
                    Operand::Immediate(amt) => {
                        let a = *amt as u8;
                        return Some((
                            ShiftInfo {
                                shift_type: st,
                                amount: a & 0x1F,
                                reg_shift: None,
                            },
                            2,
                        ));
                    }
                    Operand::Register(rs) if rs.is_arm() => {
                        return Some((
                            ShiftInfo {
                                shift_type: st,
                                amount: 0,
                                reg_shift: Some(*rs),
                            },
                            2,
                        ));
                    }
                    _ => {}
                }
            }
            None // shift keyword without valid operand → error
        }
        "rrx" => {
            // RRX = ROR with amount = 0
            Some((
                ShiftInfo {
                    shift_type: ShiftType::Ror,
                    amount: 0,
                    reg_shift: None,
                },
                1,
            ))
        }
        _ => None,
    }
}

/// Encode the 12-bit operand2 field for a register with optional barrel shift.
///
/// Immediate shift: `shift_imm[11:7] | shift_type[6:5] | 0[4] | Rm[3:0]`
/// Register shift:  `Rs[11:8] | 0[7] | shift_type[6:5] | 1[4] | Rm[3:0]`
#[inline]
fn encode_shifted_reg(rm: u8, shift: Option<ShiftInfo>) -> u32 {
    let rm = rm as u32;
    match shift {
        None => rm, // no shift: 00000|00|0|Rm
        Some(si) => {
            let stype = (si.shift_type as u32) & 0x3;
            if let Some(rs) = si.reg_shift {
                // Register shift: Rs[11:8] | 0[7] | type[6:5] | 1[4] | Rm[3:0]
                let rs_num = rs.arm_reg_num() as u32;
                (rs_num << 8) | (stype << 5) | (1 << 4) | rm
            } else {
                // Immediate shift: imm5[11:7] | type[6:5] | 0[4] | Rm[3:0]
                let imm5 = (si.amount as u32) & 0x1F;
                (imm5 << 7) | (stype << 5) | rm
            }
        }
    }
}

// ── Emit helpers ─────────────────────────────────────────────────────────

/// Emit a 32-bit little-endian instruction word.
#[inline]
fn emit32(buf: &mut InstrBytes, word: u32) {
    buf.extend_from_slice(&word.to_le_bytes());
}

// ── Data processing instructions ─────────────────────────────────────────

/// Data processing opcodes (bits 24..21).
fn dp_opcode(mnemonic: &str) -> Option<u8> {
    match mnemonic {
        "and" => Some(0x0),
        "eor" => Some(0x1),
        "sub" => Some(0x2),
        "rsb" => Some(0x3),
        "add" => Some(0x4),
        "adc" => Some(0x5),
        "sbc" => Some(0x6),
        "rsc" => Some(0x7),
        "tst" => Some(0x8),
        "teq" => Some(0x9),
        "cmp" => Some(0xA),
        "cmn" => Some(0xB),
        "orr" => Some(0xC),
        "mov" => Some(0xD),
        "bic" => Some(0xE),
        "mvn" => Some(0xF),
        _ => None,
    }
}

/// Encode a data processing instruction.
/// Forms:
///   - Rd, Rn, Rm               (register)
///   - Rd, Rn, Rm, LSL/LSR/ASR/ROR amount  (shifted register)
///   - Rd, Rn, Rm, RRX          (rotate right extend)
///   - Rd, Rn, #imm             (immediate)
///   - Rd, #imm                 (mov/mvn — Rn=0)
///   - Rd, Rm                   (mov/mvn — Rn=0)
///   - Rd, Rm, LSL/... amount   (mov/mvn with shifted register)
///   - Rn, Rm                   (cmp/cmn/tst/teq — Rd=0, S=1)
///   - Rn, Rm, LSL/... amount   (cmp/cmn/tst/teq with shifted register)
///   - Rn, #imm                 (cmp/cmn/tst/teq — Rd=0, S=1)
fn encode_dp(
    buf: &mut InstrBytes,
    cond: Cond,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
    set_flags: bool,
) -> Result<(), AsmError> {
    let opcode = dp_opcode(base)
        .ok_or_else(|| invalid_ops(base, "unknown data processing opcode", instr.span))?;

    // Compare instructions: Rn, operand2 (no Rd, S=1 implicit)
    let is_test = matches!(base, "tst" | "teq" | "cmp" | "cmn");

    let (rd, rn, op2_idx) = if is_test {
        if ops.len() < 2 {
            return Err(invalid_ops(base, "expected 2 operands", instr.span));
        }
        let rn = get_arm_reg(&ops[0], base, instr.span)?;
        (Register::ArmR0, rn, 1)
    } else if matches!(base, "mov" | "mvn") {
        if ops.len() < 2 {
            return Err(invalid_ops(base, "expected 2 operands", instr.span));
        }
        let rd = get_arm_reg(&ops[0], base, instr.span)?;
        (rd, Register::ArmR0, 1)
    } else {
        if ops.len() < 3 {
            return Err(invalid_ops(
                base,
                "expected 3 operands (Rd, Rn, operand2)",
                instr.span,
            ));
        }
        let rd = get_arm_reg(&ops[0], base, instr.span)?;
        let rn = get_arm_reg(&ops[1], base, instr.span)?;
        (rd, rn, 2)
    };

    let s = if is_test || set_flags { 1u32 } else { 0u32 };

    match &ops[op2_idx] {
        Operand::Register(rm) if rm.is_arm() => {
            // Check for barrel-shift suffix after the register
            let shift = parse_shift(ops, op2_idx + 1).map(|(si, _)| si);

            // Data processing, register form:
            // cond|00|0|opcode|S|Rn|Rd|shift_imm|shift_type|0|Rm
            let operand2 = encode_shifted_reg(rm.arm_reg_num(), shift);
            #[allow(clippy::identity_op)]
            let word = ((cond as u32) << 28)
                | (0b00 << 26)
                | ((opcode as u32) << 21)
                | (s << 20)
                | ((rn.arm_reg_num() as u32) << 16)
                | ((rd.arm_reg_num() as u32) << 12)
                | operand2;
            emit32(buf, word);
        }
        Operand::Immediate(imm) => {
            let val = *imm as u32;
            match encode_arm_imm(val) {
                Some((imm8, rot)) => {
                    // Data processing, immediate form:
                    // cond|00|1|opcode|S|Rn|Rd|rotate|imm8
                    let word = ((cond as u32) << 28)
                        | (0b001 << 25)
                        | ((opcode as u32) << 21)
                        | (s << 20)
                        | ((rn.arm_reg_num() as u32) << 16)
                        | ((rd.arm_reg_num() as u32) << 12)
                        | ((rot as u32) << 8)
                        | (imm8 as u32);
                    emit32(buf, word);
                }
                None if matches!(base, "mov" | "mvn") => {
                    // Fallback: MOVW Rd, #lo16 / MOVT Rd, #hi16 for non-encodable immediates.
                    // MVN with non-encodable: use MOVW/MOVT with bitwise complement.
                    let effective = if base == "mvn" { !val } else { val };
                    let lo16 = effective & 0xFFFF;
                    let hi16 = (effective >> 16) & 0xFFFF;
                    // MOVW: cond|0011|0|0|00|imm4|Rd|imm12
                    let imm4_lo = (lo16 >> 12) & 0xF;
                    let imm12_lo = lo16 & 0xFFF;
                    let movw = ((cond as u32) << 28)
                        | (0b0011_0000 << 20)
                        | (imm4_lo << 16)
                        | ((rd.arm_reg_num() as u32) << 12)
                        | imm12_lo;
                    emit32(buf, movw);
                    if hi16 != 0 {
                        // MOVT: cond|0011|0|1|00|imm4|Rd|imm12
                        let imm4_hi = (hi16 >> 12) & 0xF;
                        let imm12_hi = hi16 & 0xFFF;
                        let movt = ((cond as u32) << 28)
                            | (0b0011_0100 << 20)
                            | (imm4_hi << 16)
                            | ((rd.arm_reg_num() as u32) << 12)
                            | imm12_hi;
                        emit32(buf, movt);
                    }
                }
                None => {
                    return Err(invalid_ops(
                        base,
                        "immediate value cannot be encoded as ARM rotated immediate",
                        instr.span,
                    ));
                }
            }
        }
        _ => {
            return Err(invalid_ops(
                base,
                "operand2 must be register or immediate",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── Load / Store ─────────────────────────────────────────────────────────

/// Encode LDR/STR (word/byte).
/// Forms:
///   - Rd, [Rn]              — zero offset (pre-indexed, P=1)
///   - Rd, [Rn, #imm]        — immediate offset
///   - Rd, [Rn, Rm]          — register offset
///   - Rd, [Rn, Rm, LSL n]   — scaled register offset
///   - Rd, [Rn, #imm]!       — pre-index with writeback
///   - Rd, [Rn], #imm        — post-index (P=0, W=0)
///   - Rd, =label             — PC-relative literal pool load
fn encode_ldr_str(
    buf: &mut InstrBytes,
    cond: Cond,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() < 2 {
        return Err(invalid_ops(
            base,
            "expected register and memory operand",
            instr.span,
        ));
    }
    let rd = get_arm_reg(&ops[0], base, instr.span)?;
    let is_load = matches!(base, "ldr" | "ldrb" | "ldrh" | "ldrsb" | "ldrsh");
    let is_byte = matches!(base, "ldrb" | "strb");

    // Post-index: `LDR Rd, [Rn], #offset` — parser gives us [Rn] + separate immediate
    let post_index_imm = if ops.len() >= 3 {
        match &ops[1] {
            Operand::Memory(m)
                if m.addr_mode == AddrMode::Offset && m.disp == 0 && m.index.is_none() =>
            {
                match &ops[2] {
                    Operand::Immediate(v) => Some(*v as i64),
                    _ => None,
                }
            }
            _ => None,
        }
    } else {
        None
    };

    if let Some(offset) = post_index_imm {
        // Post-index immediate: cond|01|0|P=0|U|B|W=0|L|Rn|Rd|imm12
        let mem = match &ops[1] {
            Operand::Memory(m) => m,
            _ => {
                return Err(invalid_ops(
                    base,
                    "expected memory operand for post-index addressing",
                    instr.span,
                ))
            }
        };
        let rn = match mem.base {
            Some(r) if r.is_arm() => r,
            _ => {
                return Err(invalid_ops(
                    base,
                    "memory base must be ARM register",
                    instr.span,
                ))
            }
        };
        let u = if offset >= 0 { 1u32 } else { 0u32 };
        let abs_off = (offset.unsigned_abs() as u32) & 0xFFF;
        let b = is_byte as u32;
        let l = is_load as u32;
        // cond|01|0|P=0|U|B|W=0|L|Rn|Rd|imm12  (post-index)
        #[allow(clippy::identity_op)]
        let word = ((cond as u32) << 28)
            | (0b010 << 25)
            | (0u32 << 24) // P=0 (post-index)
            | (u << 23)
            | (b << 22)
            | (0u32 << 21) // W=0 for post-index
            | (l << 20)
            | ((rn.arm_reg_num() as u32) << 16)
            | ((rd.arm_reg_num() as u32) << 12)
            | abs_off;
        emit32(buf, word);
        return Ok(());
    }

    match &ops[1] {
        Operand::Memory(mem) => {
            let rn = match mem.base {
                Some(r) if r.is_arm() => r,
                _ => {
                    return Err(invalid_ops(
                        base,
                        "memory base must be ARM register",
                        instr.span,
                    ))
                }
            };

            // Pre-index writeback: [Rn, ...]! sets W=1
            let is_preindex = mem.addr_mode == AddrMode::PreIndex;
            let w = is_preindex as u32;

            if let Some(rm) = mem.index {
                // Register offset: LDR Rd, [Rn, Rm] or [Rn, -Rm]
                if !rm.is_arm() {
                    return Err(invalid_ops(
                        base,
                        "memory index must be ARM register",
                        instr.span,
                    ));
                }
                // U-bit: 1=add, 0=subtract
                let u = if mem.index_subtract { 0u32 } else { 1u32 };
                // cond|01|1|P|U|B|W|L|Rn|Rd|shift_imm|shift_type|0|Rm
                let p = 1u32; // pre-indexed
                let b = is_byte as u32;
                let l = is_load as u32;
                let word = ((cond as u32) << 28)
                    | (0b011 << 25)
                    | (p << 24)
                    | (u << 23)
                    | (b << 22)
                    | (w << 21)
                    | (l << 20)
                    | ((rn.arm_reg_num() as u32) << 16)
                    | ((rd.arm_reg_num() as u32) << 12)
                    | (rm.arm_reg_num() as u32);
                emit32(buf, word);
            } else {
                // Immediate offset: LDR Rd, [Rn, #imm12]
                let offset = mem.disp;
                let u = if offset >= 0 { 1u32 } else { 0u32 };
                let abs_off = (offset.unsigned_abs() as u32) & 0xFFF;

                if mem.disp_label.is_some() {
                    // Label-relative load — emit placeholder, create relocation
                    let p = 1u32;
                    let b = is_byte as u32;
                    let l = is_load as u32;
                    let word = ((cond as u32) << 28)
                        | (0b010 << 25)
                        | (p << 24)
                        | (1u32 << 23) // U=1 placeholder
                        | (b << 22)
                        | (w << 21)
                        | (l << 20)
                        | ((rn.arm_reg_num() as u32) << 16)
                        | ((rd.arm_reg_num() as u32) << 12);
                    let reloc_offset = buf.len();
                    emit32(buf, word);
                    if let Some(ref label) = mem.disp_label {
                        *reloc = Some(Relocation {
                            offset: reloc_offset,
                            size: 4,
                            label: alloc::rc::Rc::from(&**label),
                            kind: RelocKind::ArmLdrLit,
                            addend: mem.disp,
                            trailing_bytes: 0,
                        });
                    }
                } else {
                    // cond|01|0|P|U|B|W|L|Rn|Rd|imm12
                    let p = 1u32; // pre-indexed
                    let b = is_byte as u32;
                    let l = is_load as u32;
                    let word = ((cond as u32) << 28)
                        | (0b010 << 25)
                        | (p << 24)
                        | (u << 23)
                        | (b << 22)
                        | (w << 21)
                        | (l << 20)
                        | ((rn.arm_reg_num() as u32) << 16)
                        | ((rd.arm_reg_num() as u32) << 12)
                        | abs_off;
                    emit32(buf, word);
                }
            }
        }
        // LDR Rd, =label — literal pool (synthesize as MOV/MOVW+MOVT or PC-relative)
        Operand::Label(label) => {
            // Encode as PC-relative LDR with relocation
            let l = is_load as u32;
            let b = is_byte as u32;
            let word = ((cond as u32) << 28)
                | (0b010 << 25)
                | (1u32 << 24) // P
                | (1u32 << 23) // U (adjusted by linker)
                | (b << 22)
                | (l << 20)
                | (15u32 << 16) // Rn = PC
                | ((rd.arm_reg_num() as u32) << 12);
            let reloc_offset = buf.len();
            emit32(buf, word);
            *reloc = Some(Relocation {
                offset: reloc_offset,
                size: 4,
                label: alloc::rc::Rc::from(&**label),
                kind: RelocKind::ArmLdrLit,
                addend: 0,
                trailing_bytes: 0,
            });
        }
        _ => {
            return Err(invalid_ops(
                base,
                "expected memory operand or label",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── Branch instructions ──────────────────────────────────────────────────

fn encode_branch(
    buf: &mut InstrBytes,
    cond: Cond,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_ops(base, "expected 1 operand", instr.span));
    }

    let is_link = base == "bl";

    match &ops[0] {
        Operand::Label(label) => {
            // B/BL label — 24-bit signed offset (shifted left 2)
            // cond|101|L|imm24
            let l = is_link as u32;
            let word = ((cond as u32) << 28) | (0b101 << 25) | (l << 24);
            let reloc_offset = buf.len();
            emit32(buf, word);
            *reloc = Some(Relocation {
                offset: reloc_offset,
                size: 4,
                label: alloc::rc::Rc::from(&**label),
                kind: RelocKind::ArmBranch24,
                addend: 0,
                trailing_bytes: 0,
            });
        }
        Operand::Immediate(imm) => {
            // B/BL with immediate offset
            let offset = (*imm as i32) >> 2; // divide by 4
            let imm24 = (offset as u32) & 0x00FF_FFFF;
            let l = is_link as u32;
            let word = ((cond as u32) << 28) | (0b101 << 25) | (l << 24) | imm24;
            emit32(buf, word);
        }
        Operand::Register(reg) if reg.is_arm() => {
            if is_link {
                // BLX Rm: cond|0001|0010|1111|1111|1111|0011|Rm
                let word = ((cond as u32) << 28) | (0x12FFF30) | (reg.arm_reg_num() as u32);
                emit32(buf, word);
            } else {
                // BX Rm: cond|0001|0010|1111|1111|1111|0001|Rm
                let word = ((cond as u32) << 28) | (0x12FFF10) | (reg.arm_reg_num() as u32);
                emit32(buf, word);
            }
        }
        _ => {
            return Err(invalid_ops(
                base,
                "expected label, immediate, or register",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── PUSH / POP (LDM/STM shortcuts) ──────────────────────────────────────

fn encode_push_pop(
    buf: &mut InstrBytes,
    cond: Cond,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_ops(base, "expected register list", instr.span));
    }
    let mask = match &ops[0] {
        Operand::RegisterList(regs) => reg_list_mask(regs),
        Operand::Register(r) if r.is_arm() => 1u16 << r.arm_reg_num(),
        _ => return Err(invalid_ops(base, "expected register list", instr.span)),
    };

    if mask == 0 {
        return Err(invalid_ops(
            base,
            "register list cannot be empty",
            instr.span,
        ));
    }

    match base {
        "push" => {
            // PUSH = STMDB SP!, {reglist}
            // cond|100|1|0|0|1|0|1101|reglist
            #[allow(clippy::identity_op)]
            let word = ((cond as u32) << 28)
                | (0b100 << 25)
                | (1 << 24) // P=1
                | (0 << 23) // U=0 (decrement)
                | (0 << 22) // S=0
                | (1 << 21) // W=1 (writeback)
                | (0 << 20) // L=0 (store)
                | (13u32 << 16) // Rn = SP
                | (mask as u32);
            emit32(buf, word);
        }
        "pop" => {
            // POP = LDMIA SP!, {reglist}
            // cond|100|0|1|0|0|1|1101|reglist
            #[allow(clippy::identity_op)]
            let word = ((cond as u32) << 28)
                | (0b100 << 25)
                | (0 << 24) // P=0
                | (1 << 23) // U=1 (increment)
                | (0 << 22) // S=0
                | (1 << 21) // W=1 (writeback)
                | (1 << 20) // L=1 (load)
                | (13u32 << 16) // Rn = SP
                | (mask as u32);
            emit32(buf, word);
        }
        _ => return Err(invalid_ops(base, "internal error", instr.span)),
    }
    Ok(())
}

// ── MUL / MLA ────────────────────────────────────────────────────────────

fn encode_mul(
    buf: &mut InstrBytes,
    cond: Cond,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
    set_flags: bool,
) -> Result<(), AsmError> {
    let s = set_flags as u32;
    match base {
        "mul" => {
            // MUL Rd, Rm, Rs: cond|000000|S|0|Rd|0000|Rs|1001|Rm
            if ops.len() != 3 {
                return Err(invalid_ops("mul", "expected 3 operands", instr.span));
            }
            let rd = get_arm_reg(&ops[0], "mul", instr.span)?;
            let rm = get_arm_reg(&ops[1], "mul", instr.span)?;
            let rs = get_arm_reg(&ops[2], "mul", instr.span)?;
            let word = ((cond as u32) << 28)
                | (s << 20)
                | ((rd.arm_reg_num() as u32) << 16)
                | ((rs.arm_reg_num() as u32) << 8)
                | (0b1001 << 4)
                | (rm.arm_reg_num() as u32);
            emit32(buf, word);
        }
        "mla" => {
            // MLA Rd, Rm, Rs, Rn: cond|0000001|S|Rd|Rn|Rs|1001|Rm
            if ops.len() != 4 {
                return Err(invalid_ops("mla", "expected 4 operands", instr.span));
            }
            let rd = get_arm_reg(&ops[0], "mla", instr.span)?;
            let rm = get_arm_reg(&ops[1], "mla", instr.span)?;
            let rs = get_arm_reg(&ops[2], "mla", instr.span)?;
            let rn = get_arm_reg(&ops[3], "mla", instr.span)?;
            let word = ((cond as u32) << 28)
                | (1 << 21) // accumulate
                | (s << 20)
                | ((rd.arm_reg_num() as u32) << 16)
                | ((rn.arm_reg_num() as u32) << 12)
                | ((rs.arm_reg_num() as u32) << 8)
                | (0b1001 << 4)
                | (rm.arm_reg_num() as u32);
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                base,
                "unknown multiply instruction",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── Long Multiply: UMULL / SMULL / UMLAL / SMLAL ────────────────────────

fn encode_long_mul(
    buf: &mut InstrBytes,
    cond: Cond,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
    set_flags: bool,
) -> Result<(), AsmError> {
    // All forms: <op> RdLo, RdHi, Rm, Rs
    if ops.len() != 4 {
        return Err(invalid_ops(
            base,
            "expected 4 operands: RdLo, RdHi, Rm, Rs",
            instr.span,
        ));
    }
    let rd_lo = get_arm_reg(&ops[0], base, instr.span)?;
    let rd_hi = get_arm_reg(&ops[1], base, instr.span)?;
    let rm = get_arm_reg(&ops[2], base, instr.span)?;
    let rs = get_arm_reg(&ops[3], base, instr.span)?;

    let s = set_flags as u32;

    // cond|0000|1|U|A|S|RdHi|RdLo|Rs|1001|Rm
    // U: 1=unsigned, 0=signed
    // A: 1=accumulate (UMLAL/SMLAL), 0=no (UMULL/SMULL)
    let (u_bit, a_bit) = match base {
        "umull" => (1u32, 0u32),
        "smull" => (0u32, 0u32),
        "umlal" => (1u32, 1u32),
        "smlal" => (0u32, 1u32),
        _ => {
            return Err(invalid_ops(
                base,
                "unknown long multiply instruction",
                instr.span,
            ))
        }
    };

    let word = ((cond as u32) << 28)
        | (u_bit << 22)
        | (a_bit << 21)
        | (s << 20)
        | ((rd_hi.arm_reg_num() as u32) << 16)
        | ((rd_lo.arm_reg_num() as u32) << 12)
        | ((rs.arm_reg_num() as u32) << 8)
        | (0b1001 << 4)
        | (rm.arm_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── MOVW / MOVT (ARMv6T2+) ──────────────────────────────────────────────

fn encode_movw_movt(
    buf: &mut InstrBytes,
    cond: Cond,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(
            base,
            "expected 2 operands (Rd, #imm16)",
            instr.span,
        ));
    }
    let rd = get_arm_reg(&ops[0], base, instr.span)?;
    let imm = get_imm(&ops[1], base, instr.span)? as u32;
    if imm > 0xFFFF {
        return Err(invalid_ops(
            base,
            "immediate must fit in 16 bits",
            instr.span,
        ));
    }

    let imm4 = (imm >> 12) & 0xF;
    let imm12 = imm & 0xFFF;
    let is_top = base == "movt";

    // cond|0011|0|H|00|imm4|Rd|imm12
    // H=0 for MOVW, H=1 for MOVT
    let word = ((cond as u32) << 28)
        | (0b0011 << 24)
        | ((is_top as u32) << 22)
        | (imm4 << 16)
        | ((rd.arm_reg_num() as u32) << 12)
        | imm12;
    emit32(buf, word);
    Ok(())
}

// ── SVC / SWI ────────────────────────────────────────────────────────────

fn encode_svc(
    buf: &mut InstrBytes,
    cond: Cond,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_ops("svc", "expected immediate operand", instr.span));
    }
    let imm = get_imm(&ops[0], "svc", instr.span)? as u32;
    if imm > 0x00FF_FFFF {
        return Err(invalid_ops(
            "svc",
            "SVC number must fit in 24 bits",
            instr.span,
        ));
    }
    // cond|1111|imm24
    let word = ((cond as u32) << 28) | (0xF << 24) | imm;
    emit32(buf, word);
    Ok(())
}

// ── NOP / BKPT ───────────────────────────────────────────────────────────

fn encode_nop(buf: &mut InstrBytes, cond: Cond) {
    // NOP = MOV R0, R0 (A32 encoding)
    let word = ((cond as u32) << 28) | 0x01A0_0000;
    emit32(buf, word);
}

fn encode_bkpt(buf: &mut InstrBytes, ops: &[Operand], instr: &Instruction) -> Result<(), AsmError> {
    let imm = if ops.is_empty() {
        0u32
    } else {
        get_imm(&ops[0], "bkpt", instr.span)? as u32
    };
    if imm > 0xFFFF {
        return Err(invalid_ops(
            "bkpt",
            "BKPT number must fit in 16 bits",
            instr.span,
        ));
    }
    // BKPT is always unconditional: 1110|0001|0010|imm12|0111|imm4
    let imm12 = (imm >> 4) & 0xFFF;
    let imm4 = imm & 0xF;
    let word = 0xE120_0070 | (imm12 << 8) | imm4;
    emit32(buf, word);
    Ok(())
}

// ── BX / BLX ─────────────────────────────────────────────────────────────

fn encode_bx(
    buf: &mut InstrBytes,
    cond: Cond,
    is_link: bool,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        let name = if is_link { "blx" } else { "bx" };
        return Err(invalid_ops(name, "expected 1 register operand", instr.span));
    }
    let rm = get_arm_reg(&ops[0], if is_link { "blx" } else { "bx" }, instr.span)?;
    if is_link {
        // BLX Rm: cond|0001|0010|1111|1111|1111|0011|Rm
        let word = ((cond as u32) << 28) | 0x012F_FF30 | (rm.arm_reg_num() as u32);
        emit32(buf, word);
    } else {
        // BX Rm: cond|0001|0010|1111|1111|1111|0001|Rm
        let word = ((cond as u32) << 28) | 0x012F_FF10 | (rm.arm_reg_num() as u32);
        emit32(buf, word);
    }
    Ok(())
}

// ── LDM / STM ────────────────────────────────────────────────────────────

fn encode_ldm_stm(
    buf: &mut InstrBytes,
    cond: Cond,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(
            base,
            "expected base register and register list",
            instr.span,
        ));
    }

    // Base register — may be plain Register or Memory(PreIndex) for writeback '!'
    let (rn, writeback) = match &ops[0] {
        Operand::Register(r) if r.is_arm() => (*r, false),
        Operand::Memory(m) if m.addr_mode == AddrMode::PreIndex => {
            // Parser wraps `R0!` as Memory(PreIndex, base=R0)
            match m.base {
                Some(r) if r.is_arm() => (r, true),
                _ => return Err(invalid_ops(base, "expected ARM base register", instr.span)),
            }
        }
        _ => return Err(invalid_ops(base, "expected ARM base register", instr.span)),
    };

    let mask = match &ops[1] {
        Operand::RegisterList(regs) => reg_list_mask(regs),
        _ => return Err(invalid_ops(base, "expected register list", instr.span)),
    };

    // Decode addressing mode from mnemonic suffix
    let is_load = matches!(
        base,
        "ldm" | "ldmia" | "ldmfd" | "ldmdb" | "ldmea" | "ldmib" | "ldmed" | "ldmda" | "ldmfa"
    );
    let (p, u) = match base {
        "ldm" | "ldmia" | "ldmfd" | "stm" | "stmia" | "stmea" => (0, 1), // increment after
        "ldmdb" | "ldmea" | "stmdb" | "stmfd" => (1, 0),                 // decrement before
        "ldmib" | "ldmed" | "stmib" | "stmfa" => (1, 1),                 // increment before
        "ldmda" | "ldmfa" | "stmda" | "stmed" => (0, 0),                 // decrement after
        _ => (0, 1),                                                     // default: IA
    };

    let w = writeback as u32;

    // cond|100|P|U|S|W|L|Rn|reglist
    #[allow(clippy::identity_op)]
    let word = ((cond as u32) << 28)
        | (0b100 << 25)
        | ((p as u32) << 24)
        | ((u as u32) << 23)
        | (0 << 22) // S=0
        | (w << 21)
        | ((is_load as u32) << 20)
        | ((rn.arm_reg_num() as u32) << 16)
        | (mask as u32);
    emit32(buf, word);
    Ok(())
}

// ── Halfword / Signed Load/Store ─────────────────────────────────────────

/// Encode LDRH/STRH/LDRSB/LDRSH (misc load/store, different format from LDR/STR).
/// ARM A32 halfword/signed transfer:  cond|000|P|U|1|W|L|Rn|Rd|imm4H|1|SH|1|imm4L
///   or register offset:              cond|000|P|U|0|W|L|Rn|Rd|0000|1|SH|1|Rm
/// Forms:
///   - Rd, [Rn]              — zero offset
///   - Rd, [Rn, #imm]        — immediate offset (P=1, W=0)
///   - Rd, [Rn, Rm]          — register offset   (P=1, W=0)
///   - Rd, [Rn, #imm]!       — pre-index with writeback (P=1, W=1)
///   - Rd, [Rn, Rm]!         — register pre-index (P=1, W=1)
///   - Rd, [Rn], #imm        — post-index (P=0, W=0)
fn encode_ldr_str_h(
    buf: &mut InstrBytes,
    cond: Cond,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() < 2 {
        return Err(invalid_ops(
            base,
            "expected register and memory operand",
            instr.span,
        ));
    }
    let rd = get_arm_reg(&ops[0], base, instr.span)?;
    let is_load = matches!(base, "ldrh" | "ldrsb" | "ldrsh");

    // SH bits: LDRH/STRH = 01, LDRSB = 10, LDRSH = 11
    let (s_bit, h_bit) = match base {
        "ldrh" | "strh" => (0u32, 1u32),
        "ldrsb" => (1, 0),
        "ldrsh" => (1, 1),
        _ => return Err(invalid_ops(base, "internal error", instr.span)),
    };

    let l = is_load as u32;

    // Post-index: `LDRH Rd, [Rn], #offset` — parser gives [Rn] + separate immediate
    let post_index_imm = if ops.len() >= 3 {
        match &ops[1] {
            Operand::Memory(m)
                if m.addr_mode == AddrMode::Offset && m.disp == 0 && m.index.is_none() =>
            {
                match &ops[2] {
                    Operand::Immediate(v) => Some(*v as i64),
                    _ => None,
                }
            }
            _ => None,
        }
    } else {
        None
    };

    if let Some(offset) = post_index_imm {
        // Post-index immediate: cond|000|P=0|U|1|W=0|L|Rn|Rd|imm4H|1|SH|1|imm4L
        let mem = match &ops[1] {
            Operand::Memory(m) => m,
            _ => return Err(invalid_ops(base, "expected memory operand", instr.span)),
        };
        let rn = match mem.base {
            Some(r) if r.is_arm() => r,
            _ => {
                return Err(invalid_ops(
                    base,
                    "memory base must be ARM register",
                    instr.span,
                ))
            }
        };
        let u = if offset >= 0 { 1u32 } else { 0u32 };
        let abs_off = offset.unsigned_abs() as u32;
        if abs_off > 255 {
            return Err(invalid_ops(
                base,
                "halfword immediate offset must fit in 8 bits",
                instr.span,
            ));
        }
        let imm4h = (abs_off >> 4) & 0xF;
        let imm4l = abs_off & 0xF;
        #[allow(clippy::identity_op)]
        let word = ((cond as u32) << 28)
            | (0u32 << 24) // P=0 (post-index)
            | (u << 23)
            | (1 << 22)    // immediate form
            | (0u32 << 21) // W=0 for post-index
            | (l << 20)
            | ((rn.arm_reg_num() as u32) << 16)
            | ((rd.arm_reg_num() as u32) << 12)
            | (imm4h << 8)
            | (1 << 7)
            | (s_bit << 6)
            | (h_bit << 5)
            | (1 << 4)
            | imm4l;
        emit32(buf, word);
        return Ok(());
    }

    let mem = match &ops[1] {
        Operand::Memory(m) => m,
        _ => return Err(invalid_ops(base, "expected memory operand", instr.span)),
    };

    let rn = match mem.base {
        Some(r) if r.is_arm() => r,
        _ => {
            return Err(invalid_ops(
                base,
                "memory base must be ARM register",
                instr.span,
            ))
        }
    };

    // Pre-index writeback: [Rn, ...]! sets W=1
    let is_preindex = mem.addr_mode == AddrMode::PreIndex;
    let w = is_preindex as u32;

    if let Some(rm) = mem.index {
        // Register offset form
        if !rm.is_arm() {
            return Err(invalid_ops(
                base,
                "memory index must be ARM register",
                instr.span,
            ));
        }
        // U-bit: 1=add, 0=subtract
        let u = if mem.index_subtract { 0u32 } else { 1u32 };
        // cond|000|P=1|U|0|W|L|Rn|Rd|0000|1|SH|1|Rm
        let word = ((cond as u32) << 28)
            | (1 << 24)  // P=1 pre-indexed
            | (u << 23)  // U: add or subtract
            | (w << 21)  // W: writeback
            | (l << 20)
            | ((rn.arm_reg_num() as u32) << 16)
            | ((rd.arm_reg_num() as u32) << 12)
            | (1 << 7)
            | (s_bit << 6)
            | (h_bit << 5)
            | (1 << 4)
            | (rm.arm_reg_num() as u32);
        emit32(buf, word);
    } else {
        // Immediate offset form
        let offset = mem.disp;
        let u = if offset >= 0 { 1u32 } else { 0u32 };
        let abs_off = offset.unsigned_abs() as u32;
        if abs_off > 255 {
            return Err(invalid_ops(
                base,
                "halfword immediate offset must fit in 8 bits",
                instr.span,
            ));
        }
        let imm4h = (abs_off >> 4) & 0xF;
        let imm4l = abs_off & 0xF;
        // cond|000|P=1|U|1|W|L|Rn|Rd|imm4H|1|SH|1|imm4L
        let word = ((cond as u32) << 28)
            | (1 << 24)  // P=1
            | (u << 23)
            | (1 << 22)  // immediate form (bit 22 = 1)
            | (w << 21)  // W: writeback
            | (l << 20)
            | ((rn.arm_reg_num() as u32) << 16)
            | ((rd.arm_reg_num() as u32) << 12)
            | (imm4h << 8)
            | (1 << 7)
            | (s_bit << 6)
            | (h_bit << 5)
            | (1 << 4)
            | imm4l;
        emit32(buf, word);
    }
    Ok(())
}

// ── ADR (PC-relative immediate) ──────────────────────────────────────────

fn encode_adr(
    buf: &mut InstrBytes,
    cond: Cond,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(
            "adr",
            "expected Rd, label_or_immediate",
            instr.span,
        ));
    }
    let rd = get_arm_reg(&ops[0], "adr", instr.span)?;

    match &ops[1] {
        Operand::Label(label) => {
            // ADR Rd, label => ADD Rd, PC, #offset (resolved by linker)
            // Emit ADD Rd, PC, #0 as placeholder
            let word = ((cond as u32) << 28)
                | (0b001 << 25) // immediate
                | (0x4 << 21)   // ADD opcode
                | (15u32 << 16) // Rn = PC
                | ((rd.arm_reg_num() as u32) << 12);
            let reloc_offset = buf.len();
            emit32(buf, word);
            *reloc = Some(Relocation {
                offset: reloc_offset,
                size: 4,
                label: alloc::rc::Rc::from(&**label),
                kind: RelocKind::ArmAdr,
                addend: 0,
                trailing_bytes: 0,
            });
        }
        Operand::Immediate(imm) => {
            // ADR Rd, #offset => ADD/SUB Rd, PC, #offset
            let offset = *imm as i32;
            let (op, abs_val) = if offset >= 0 {
                (0x4u32, offset as u32) // ADD
            } else {
                (0x2u32, (-offset) as u32) // SUB
            };
            let (imm8, rot) = encode_arm_imm(abs_val).ok_or_else(|| {
                invalid_ops(
                    "adr",
                    "offset cannot be encoded as ARM rotated immediate",
                    instr.span,
                )
            })?;
            let word = ((cond as u32) << 28)
                | (0b001 << 25)
                | (op << 21)
                | (15u32 << 16) // Rn = PC
                | ((rd.arm_reg_num() as u32) << 12)
                | ((rot as u32) << 8)
                | (imm8 as u32);
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                "adr",
                "expected label or immediate",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── CLZ (Count Leading Zeros) ────────────────────────────────────────────

fn encode_clz(
    buf: &mut InstrBytes,
    cond: Cond,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops("clz", "expected Rd, Rm", instr.span));
    }
    let rd = get_arm_reg(&ops[0], "clz", instr.span)?;
    let rm = get_arm_reg(&ops[1], "clz", instr.span)?;
    // cond|00010110|1111|Rd|1111|0001|Rm
    let word = ((cond as u32) << 28)
        | (0b00010110 << 20)
        | (0xF << 16) // SBZ = 1111
        | ((rd.arm_reg_num() as u32) << 12)
        | (0xF << 8) // SBZ = 1111
        | (0b0001 << 4)
        | (rm.arm_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── REV / REV16 / RBIT ──────────────────────────────────────────────────

fn encode_rev(
    buf: &mut InstrBytes,
    cond: Cond,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(mnemonic, "expected Rd, Rm", instr.span));
    }
    let rd = get_arm_reg(&ops[0], mnemonic, instr.span)?;
    let rm = get_arm_reg(&ops[1], mnemonic, instr.span)?;
    // REV:    cond|01101011|1111|Rd|1111|0011|Rm
    // REV16:  cond|01101011|1111|Rd|1111|1011|Rm
    // RBIT:   cond|01101111|1111|Rd|1111|0011|Rm
    let (op1, op2) = match mnemonic {
        "rev" => (0b01101011u32, 0b0011u32),
        "rev16" => (0b01101011u32, 0b1011u32),
        "rbit" => (0b01101111u32, 0b0011u32),
        _ => return Err(invalid_ops(mnemonic, "internal error", instr.span)),
    };
    let word = ((cond as u32) << 28)
        | (op1 << 20)
        | (0xF << 16)
        | ((rd.arm_reg_num() as u32) << 12)
        | (0xF << 8)
        | (op2 << 4)
        | (rm.arm_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── BFC / BFI / SBFX / UBFX (bitfield operations) ──────────────────────

/// Encode ARM32 bitfield instructions.
///
/// ```text
/// BFC  Rd, #lsb, #width        – clear #width bits starting at #lsb
/// BFI  Rd, Rn, #lsb, #width    – insert low #width bits of Rn into Rd at #lsb
/// SBFX Rd, Rn, #lsb, #width    – signed extract #width bits at #lsb
/// UBFX Rd, Rn, #lsb, #width    – unsigned extract #width bits at #lsb
/// ```
fn encode_bitfield(
    buf: &mut InstrBytes,
    cond: Cond,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    let word = match mnemonic {
        // BFC Rd, #lsb, #width  — 3 operands
        // cond|0111110|msb|Rd|lsb|0011111
        "bfc" => {
            if ops.len() != 3 {
                return Err(invalid_ops("bfc", "expected Rd, #lsb, #width", instr.span));
            }
            let rd = get_arm_reg(&ops[0], "bfc", instr.span)?;
            let lsb = get_imm(&ops[1], "bfc", instr.span)? as u32;
            let width = get_imm(&ops[2], "bfc", instr.span)? as u32;
            if lsb > 31 || width == 0 || width > 32 || lsb + width > 32 {
                return Err(invalid_ops("bfc", "lsb/width out of range", instr.span));
            }
            let msb = lsb + width - 1;
            ((cond as u32) << 28)
                | (0b0111110 << 21)
                | (msb << 16)
                | ((rd.arm_reg_num() as u32) << 12)
                | (lsb << 7)
                | (0b001 << 4)
                | 0b1111
        }

        // BFI Rd, Rn, #lsb, #width  — 4 operands
        // cond|0111110|msb|Rd|lsb|001|Rn
        "bfi" => {
            if ops.len() != 4 {
                return Err(invalid_ops(
                    "bfi",
                    "expected Rd, Rn, #lsb, #width",
                    instr.span,
                ));
            }
            let rd = get_arm_reg(&ops[0], "bfi", instr.span)?;
            let rn = get_arm_reg(&ops[1], "bfi", instr.span)?;
            let lsb = get_imm(&ops[2], "bfi", instr.span)? as u32;
            let width = get_imm(&ops[3], "bfi", instr.span)? as u32;
            if lsb > 31 || width == 0 || width > 32 || lsb + width > 32 {
                return Err(invalid_ops("bfi", "lsb/width out of range", instr.span));
            }
            let msb = lsb + width - 1;
            ((cond as u32) << 28)
                | (0b0111110 << 21)
                | (msb << 16)
                | ((rd.arm_reg_num() as u32) << 12)
                | (lsb << 7)
                | (0b001 << 4)
                | (rn.arm_reg_num() as u32)
        }

        // SBFX Rd, Rn, #lsb, #width  — 4 operands
        // cond|0111101|widthm1|Rd|lsb|101|Rn
        "sbfx" => {
            if ops.len() != 4 {
                return Err(invalid_ops(
                    "sbfx",
                    "expected Rd, Rn, #lsb, #width",
                    instr.span,
                ));
            }
            let rd = get_arm_reg(&ops[0], "sbfx", instr.span)?;
            let rn = get_arm_reg(&ops[1], "sbfx", instr.span)?;
            let lsb = get_imm(&ops[2], "sbfx", instr.span)? as u32;
            let width = get_imm(&ops[3], "sbfx", instr.span)? as u32;
            if lsb > 31 || width == 0 || width > 32 || lsb + width > 32 {
                return Err(invalid_ops("sbfx", "lsb/width out of range", instr.span));
            }
            let widthm1 = width - 1;
            ((cond as u32) << 28)
                | (0b0111101 << 21)
                | (widthm1 << 16)
                | ((rd.arm_reg_num() as u32) << 12)
                | (lsb << 7)
                | (0b101 << 4)
                | (rn.arm_reg_num() as u32)
        }

        // UBFX Rd, Rn, #lsb, #width  — 4 operands
        // cond|0111111|widthm1|Rd|lsb|101|Rn
        "ubfx" => {
            if ops.len() != 4 {
                return Err(invalid_ops(
                    "ubfx",
                    "expected Rd, Rn, #lsb, #width",
                    instr.span,
                ));
            }
            let rd = get_arm_reg(&ops[0], "ubfx", instr.span)?;
            let rn = get_arm_reg(&ops[1], "ubfx", instr.span)?;
            let lsb = get_imm(&ops[2], "ubfx", instr.span)? as u32;
            let width = get_imm(&ops[3], "ubfx", instr.span)? as u32;
            if lsb > 31 || width == 0 || width > 32 || lsb + width > 32 {
                return Err(invalid_ops("ubfx", "lsb/width out of range", instr.span));
            }
            let widthm1 = width - 1;
            ((cond as u32) << 28)
                | (0b0111111 << 21)
                | (widthm1 << 16)
                | ((rd.arm_reg_num() as u32) << 12)
                | (lsb << 7)
                | (0b101 << 4)
                | (rn.arm_reg_num() as u32)
        }

        _ => return Err(invalid_ops(mnemonic, "internal error", instr.span)),
    };
    emit32(buf, word);
    Ok(())
}

// ── LDREX / STREX (Exclusive access — atomics) ──────────────────────────

/// Encode LDREX / LDREXB / LDREXH / LDREXD.
/// LDREX:  cond|00011001|Rn|Rd|1111|1001|1111
/// LDREXB: cond|00011101|Rn|Rd|1111|1001|1111
/// LDREXH: cond|00011111|Rn|Rd|1111|1001|1111
/// LDREXD: cond|00011011|Rn|Rd|1111|1001|1111
fn encode_ldrex(
    buf: &mut InstrBytes,
    cond: Cond,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(base, "expected Rd, [Rn]", instr.span));
    }
    let rd = get_arm_reg(&ops[0], base, instr.span)?;
    let rn = match &ops[1] {
        Operand::Memory(m) => match m.base {
            Some(r) if r.is_arm() => r,
            _ => {
                return Err(invalid_ops(
                    base,
                    "expected [Rn] memory operand",
                    instr.span,
                ))
            }
        },
        _ => {
            return Err(invalid_ops(
                base,
                "expected [Rn] memory operand",
                instr.span,
            ))
        }
    };
    let op_bits: u32 = match base {
        "ldrex" => 0b00011001,
        "ldrexb" => 0b00011101,
        "ldrexh" => 0b00011111,
        "ldrexd" => 0b00011011,
        _ => return Err(invalid_ops(base, "unknown ldrex variant", instr.span)),
    };
    let word = ((cond as u32) << 28)
        | (op_bits << 20)
        | ((rn.arm_reg_num() as u32) << 16)
        | ((rd.arm_reg_num() as u32) << 12)
        | 0xF9F;
    emit32(buf, word);
    Ok(())
}

/// Encode STREX / STREXB / STREXH / STREXD.
/// STREX:  cond|00011000|Rn|Rd|1111|1001|Rm
/// STREXB: cond|00011100|Rn|Rd|1111|1001|Rm
/// STREXH: cond|00011110|Rn|Rd|1111|1001|Rm
/// STREXD: cond|00011010|Rn|Rd|1111|1001|Rm
fn encode_strex(
    buf: &mut InstrBytes,
    cond: Cond,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    // STREX Rd, Rm, [Rn] — Rd = status (0=success), Rm = value to store
    if ops.len() != 3 {
        return Err(invalid_ops(base, "expected Rd, Rm, [Rn]", instr.span));
    }
    let rd = get_arm_reg(&ops[0], base, instr.span)?;
    let rm = get_arm_reg(&ops[1], base, instr.span)?;
    let rn = match &ops[2] {
        Operand::Memory(m) => match m.base {
            Some(r) if r.is_arm() => r,
            _ => {
                return Err(invalid_ops(
                    base,
                    "expected [Rn] memory operand",
                    instr.span,
                ))
            }
        },
        _ => {
            return Err(invalid_ops(
                base,
                "expected [Rn] memory operand",
                instr.span,
            ))
        }
    };
    let op_bits: u32 = match base {
        "strex" => 0b00011000,
        "strexb" => 0b00011100,
        "strexh" => 0b00011110,
        "strexd" => 0b00011010,
        _ => return Err(invalid_ops(base, "unknown strex variant", instr.span)),
    };
    let word = ((cond as u32) << 28)
        | (op_bits << 20)
        | ((rn.arm_reg_num() as u32) << 16)
        | ((rd.arm_reg_num() as u32) << 12)
        | (0xF << 8)
        | (0b1001 << 4)
        | (rm.arm_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── DMB / DSB / ISB (Memory barriers) ───────────────────────────────────

fn encode_barrier(
    buf: &mut InstrBytes,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    // Default option = SY (full system)
    let option = if ops.is_empty() {
        0xF_u32
    } else {
        let v = get_imm(&ops[0], base, instr.span)? as u32;
        if v > 0xF {
            return Err(invalid_ops(base, "barrier option must be 0-15", instr.span));
        }
        v
    };
    // Unconditional barrier encodings:
    // DMB: F57FF050|option, DSB: F57FF040|option, ISB: F57FF060|option
    let word = match base {
        "dmb" => 0xF57F_F050 | option,
        "dsb" => 0xF57F_F040 | option,
        "isb" => 0xF57F_F060 | option,
        _ => {
            return Err(invalid_ops(
                base,
                &alloc::format!("unknown barrier instruction '{}'", base),
                instr.span,
            ))
        }
    };
    emit32(buf, word);
    Ok(())
}

fn encode_extend(
    buf: &mut InstrBytes,
    cond: Cond,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(mnemonic, "expected Rd, Rm", instr.span));
    }
    let rd = get_arm_reg(&ops[0], mnemonic, instr.span)?;
    let rm = get_arm_reg(&ops[1], mnemonic, instr.span)?;
    // UXTB:   cond|01101110|1111|Rd|rot|00000111|Rm
    // UXTH:   cond|01101111|1111|Rd|rot|00000111|Rm
    // SXTB:   cond|01101010|1111|Rd|rot|00000111|Rm
    // SXTH:   cond|01101011|1111|Rd|rot|00000111|Rm
    // (rotation = 0)
    let op = match mnemonic {
        "uxtb" => 0b01101110u32,
        "uxth" => 0b01101111u32,
        "sxtb" => 0b01101010u32,
        "sxth" => 0b01101011u32,
        _ => return Err(invalid_ops(mnemonic, "internal error", instr.span)),
    };
    let word = (((cond as u32) << 28)
        | (op << 20)
        | (0xF << 16) // Rn = 1111 (no rotation source)
        | ((rd.arm_reg_num() as u32) << 12)) // rotation = 0
        | (0b0111 << 4)
        | (rm.arm_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── MRS / MSR ────────────────────────────────────────────────────────────

fn encode_mrs(
    buf: &mut InstrBytes,
    cond: Cond,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops("mrs", "expected Rd, CPSR", instr.span));
    }
    let rd = get_arm_reg(&ops[0], "mrs", instr.span)?;
    // ops[1] should be cpsr register
    match &ops[1] {
        Operand::Register(Register::ArmCpsr) => {}
        _ => return Err(invalid_ops("mrs", "expected CPSR as source", instr.span)),
    }
    // MRS Rd, CPSR: cond|00010|R=0|00|1111|Rd|0000|0000|0000
    let word =
        ((cond as u32) << 28) | (0b00010 << 23) | (0xF << 16) | ((rd.arm_reg_num() as u32) << 12);
    emit32(buf, word);
    Ok(())
}

// ── MSR (write to CPSR) ─────────────────────────────────────────────────

fn encode_msr(
    buf: &mut InstrBytes,
    cond: Cond,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops("msr", "expected CPSR, Rm_or_imm", instr.span));
    }
    // First operand must be CPSR (or cpsr_f for flags-only)
    match &ops[0] {
        Operand::Register(Register::ArmCpsr) => {}
        _ => {
            return Err(invalid_ops(
                "msr",
                "expected CPSR as destination",
                instr.span,
            ))
        }
    }
    // field_mask = 0b1001 (flags + control) — write all CPSR fields
    let field_mask = 0b1001u32;

    match &ops[1] {
        Operand::Register(rm) if rm.is_arm() => {
            // MSR CPSR, Rm: cond|00010|R=0|10|field_mask|1111|00000000|Rm
            let word = ((cond as u32) << 28)
                | (0b00010 << 23)
                | (0b10 << 21)
                | (field_mask << 16)
                | (0xF << 12)
                | (rm.arm_reg_num() as u32);
            emit32(buf, word);
        }
        Operand::Immediate(imm) => {
            // MSR CPSR, #imm: cond|00110|R=0|10|field_mask|1111|rotate|imm8
            let val = *imm as u32;
            let (imm8, rot) = encode_arm_imm(val).ok_or_else(|| {
                invalid_ops(
                    "msr",
                    "immediate cannot be encoded as ARM rotated immediate",
                    instr.span,
                )
            })?;
            let word = ((cond as u32) << 28)
                | (0b00110 << 23)
                | (0b10 << 21)
                | (field_mask << 16)
                | (0xF << 12)
                | ((rot as u32) << 8)
                | (imm8 as u32);
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                "msr",
                "expected register or immediate",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── Thumb / Thumb-2 encoder ──────────────────────────────────────────────

/// Emit a 16-bit Thumb instruction (little-endian).
fn emit16(buf: &mut InstrBytes, hw: u16) {
    let b = hw.to_le_bytes();
    buf.push(b[0]);
    buf.push(b[1]);
}

/// Check that register number is a low register (R0-R7) for 16-bit Thumb.
fn require_lo(reg: Register, mnemonic: &str, span: crate::error::Span) -> Result<u8, AsmError> {
    let n = reg.arm_reg_num();
    if n > 7 {
        return Err(invalid_ops(
            mnemonic,
            "register not accessible in 16-bit Thumb (must be R0-R7)",
            span,
        ));
    }
    Ok(n)
}

/// Encode a Thumb (T16/T32) instruction.
fn encode_thumb(instr: &Instruction) -> Result<EncodedInstr, AsmError> {
    let mut buf = InstrBytes::new();
    let mut reloc: Option<Relocation> = None;
    let mut relax: Option<crate::encoder::RelaxInfo> = None;

    let mnemonic = instr.mnemonic.as_str();
    let ops = &instr.operands;

    let (base_with_s, cond) = parse_cond(mnemonic);
    let (base, set_flags) = if base_with_s.ends_with('s')
        && base_with_s.len() > 1
        && matches!(
            &base_with_s[..base_with_s.len() - 1],
            "add"
                | "sub"
                | "mov"
                | "and"
                | "orr"
                | "eor"
                | "bic"
                | "lsl"
                | "lsr"
                | "asr"
                | "ror"
                | "mul"
                | "mvn"
                | "adc"
                | "sbc"
                | "rsb"
                | "neg"
                | "tst"
                | "cmn"
                | "cmp"
        ) {
        (&base_with_s[..base_with_s.len() - 1], true)
    } else {
        (base_with_s, false)
    };

    match base {
        "nop" => {
            // T1: NOP = 0xBF00
            emit16(&mut buf, 0xBF00);
        }

        "bkpt" => {
            // BKPT #imm8: 0xBE00 | imm8
            let imm = match ops.first() {
                Some(Operand::Immediate(v)) => *v as u32,
                _ => 0,
            };
            if imm > 255 {
                return Err(invalid_ops("bkpt", "immediate must be 0-255", instr.span));
            }
            emit16(&mut buf, 0xBE00 | (imm as u16));
        }

        "svc" | "swi" => {
            // SVC #imm8: 0xDF00 | imm8
            let imm = match ops.first() {
                Some(Operand::Immediate(v)) => *v as u32,
                _ => return Err(invalid_ops("svc", "expected immediate", instr.span)),
            };
            if imm > 255 {
                return Err(invalid_ops("svc", "immediate must be 0-255", instr.span));
            }
            emit16(&mut buf, 0xDF00 | (imm as u16));
        }

        "bx" => {
            // BX Rm: 0x4700 | (Rm << 3)
            let rm = match ops.first() {
                Some(Operand::Register(r)) => r.arm_reg_num(),
                _ => return Err(invalid_ops("bx", "expected register", instr.span)),
            };
            emit16(&mut buf, 0x4700 | ((rm as u16) << 3));
        }

        "blx" => {
            // BLX Rm: 0x4780 | (Rm << 3)
            let rm = match ops.first() {
                Some(Operand::Register(r)) => r.arm_reg_num(),
                _ => return Err(invalid_ops("blx", "expected register", instr.span)),
            };
            emit16(&mut buf, 0x4780 | ((rm as u16) << 3));
        }

        "b" => {
            // Branches: B label or B.cond label
            match ops.first() {
                Some(Operand::Label(label_name)) => {
                    if !matches!(cond, Cond::Al) {
                        // Conditional branch — T1 narrow: [15:12]=1101 [11:8]=cond [7:0]=imm8
                        // Relaxable: narrow=2B (T1), wide=4B (T2 32-bit cond branch)
                        let cond_u16 = cond as u16;
                        let short_hw = 0xD000_u16 | (cond_u16 << 8);
                        let mut short_bytes = InstrBytes::new();
                        emit16(&mut short_bytes, short_hw);

                        let mut long_bytes = InstrBytes::new();
                        let hw1: u16 = 0xF000 | (cond_u16 << 6);
                        let hw2: u16 = 0x8000;
                        emit16(&mut long_bytes, hw1);
                        emit16(&mut long_bytes, hw2);

                        relax = Some(crate::encoder::RelaxInfo {
                            short_bytes,
                            short_reloc_offset: 0,
                            short_relocation: Some(Relocation {
                                offset: 0,
                                size: 2,
                                label: alloc::rc::Rc::from(label_name.as_str()),
                                kind: RelocKind::ThumbBranch8,
                                addend: 0,
                                trailing_bytes: 0,
                            }),
                        });
                        buf = long_bytes;
                        reloc = Some(Relocation {
                            offset: 0,
                            size: 4,
                            label: alloc::rc::Rc::from(label_name.as_str()),
                            kind: RelocKind::ThumbCondBranchW,
                            addend: 0,
                            trailing_bytes: 0,
                        });
                    } else {
                        // Unconditional branch — T2 narrow: [15:11]=11100 [10:0]=imm11
                        // Relaxable: narrow=2B (T2), wide=4B (T2 32-bit B.W)
                        let mut short_bytes = InstrBytes::new();
                        emit16(&mut short_bytes, 0xE000);

                        let mut long_bytes = InstrBytes::new();
                        emit16(&mut long_bytes, 0xF000);
                        emit16(&mut long_bytes, 0x9000);

                        relax = Some(crate::encoder::RelaxInfo {
                            short_bytes,
                            short_reloc_offset: 0,
                            short_relocation: Some(Relocation {
                                offset: 0,
                                size: 2,
                                label: alloc::rc::Rc::from(label_name.as_str()),
                                kind: RelocKind::ThumbBranch11,
                                addend: 0,
                                trailing_bytes: 0,
                            }),
                        });
                        buf = long_bytes;
                        reloc = Some(Relocation {
                            offset: 0,
                            size: 4,
                            label: alloc::rc::Rc::from(label_name.as_str()),
                            kind: RelocKind::ThumbBranchW,
                            addend: 0,
                            trailing_bytes: 0,
                        });
                    }
                }
                Some(Operand::Immediate(off)) => {
                    let off = *off;
                    if !matches!(cond, Cond::Al) {
                        let delta = (off - 4) >> 1;
                        if !(-128..=127).contains(&delta) {
                            let cond_u16 = cond as u16;
                            let encoded = encode_thumb2_cond_branch(delta as i32, cond_u16);
                            emit16(&mut buf, encoded.0);
                            emit16(&mut buf, encoded.1);
                        } else {
                            let imm8 = (delta as i8) as u8;
                            let cond_u16 = cond as u16;
                            emit16(&mut buf, 0xD000 | (cond_u16 << 8) | (imm8 as u16));
                        }
                    } else {
                        let delta = (off - 4) >> 1;
                        if !(-1024..=1023).contains(&delta) {
                            let encoded = encode_thumb2_branch(delta as i32);
                            emit16(&mut buf, encoded.0);
                            emit16(&mut buf, encoded.1);
                        } else {
                            let imm11 = (delta as u16) & 0x7FF;
                            emit16(&mut buf, 0xE000 | imm11);
                        }
                    }
                }
                _ => {
                    return Err(invalid_ops("b", "expected label or offset", instr.span));
                }
            }
        }

        "bl" => match ops.first() {
            Some(Operand::Label(label_name)) => {
                emit16(&mut buf, 0xF000);
                emit16(&mut buf, 0xD000);
                reloc = Some(Relocation {
                    offset: 0,
                    size: 4,
                    label: alloc::rc::Rc::from(label_name.as_str()),
                    kind: RelocKind::ThumbBl,
                    addend: 0,
                    trailing_bytes: 0,
                });
            }
            Some(Operand::Immediate(off)) => {
                let delta = ((*off - 4) >> 1) as i32;
                let encoded = encode_thumb2_bl(delta);
                emit16(&mut buf, encoded.0);
                emit16(&mut buf, encoded.1);
            }
            _ => {
                return Err(invalid_ops("bl", "expected label or offset", instr.span));
            }
        },

        "push" => {
            // PUSH {reg_list}: 0xB400 | M(LR) << 8 | reg_list
            match ops.first() {
                Some(Operand::RegisterList(regs)) => {
                    let mut mask: u16 = 0;
                    for reg in regs {
                        let n = reg.arm_reg_num();
                        if n == 14 {
                            // LR → bit 8
                            mask |= 1 << 8;
                        } else if n <= 7 {
                            mask |= 1 << n;
                        } else {
                            return Err(invalid_ops(
                                "push",
                                "only R0-R7 and LR allowed in 16-bit Thumb PUSH",
                                instr.span,
                            ));
                        }
                    }
                    emit16(&mut buf, 0xB400 | mask);
                }
                _ => {
                    return Err(invalid_ops("push", "expected register list", instr.span));
                }
            }
        }

        "pop" => {
            // POP {reg_list}: 0xBC00 | P(PC) << 8 | reg_list
            match ops.first() {
                Some(Operand::RegisterList(regs)) => {
                    let mut mask: u16 = 0;
                    for reg in regs {
                        let n = reg.arm_reg_num();
                        if n == 15 {
                            // PC → bit 8
                            mask |= 1 << 8;
                        } else if n <= 7 {
                            mask |= 1 << n;
                        } else {
                            return Err(invalid_ops(
                                "pop",
                                "only R0-R7 and PC allowed in 16-bit Thumb POP",
                                instr.span,
                            ));
                        }
                    }
                    emit16(&mut buf, 0xBC00 | mask);
                }
                _ => {
                    return Err(invalid_ops("pop", "expected register list", instr.span));
                }
            }
        }

        "mov" | "movs" if !set_flags => {
            // Handle both "mov" and the explicit "movs" mnemonic form
            thumb_encode_mov(&mut buf, ops, instr, base == "movs" || set_flags)?;
        }

        "mov" if set_flags => {
            thumb_encode_mov(&mut buf, ops, instr, true)?;
        }

        "cmp" => {
            // CMP Rn, #imm8: [15:11]=00101 [10:8]=Rn [7:0]=imm8
            // CMP Rn, Rm (lo): [15:6]=0100001010 [5:3]=Rm [2:0]=Rn
            // CMP Rn, Rm (hi): [15:8]=01000101 [7]=N [6:3]=Rm [2:0]=Rn_lo
            match (ops.first(), ops.get(1)) {
                (Some(Operand::Register(rn)), Some(Operand::Immediate(imm))) => {
                    let n = rn.arm_reg_num();
                    if n > 7 {
                        return Err(invalid_ops("cmp", "CMP imm8 requires R0-R7", instr.span));
                    }
                    let imm = *imm;
                    if !(0..=255).contains(&imm) {
                        return Err(invalid_ops("cmp", "immediate must be 0-255", instr.span));
                    }
                    emit16(&mut buf, 0x2800 | ((n as u16) << 8) | (imm as u16));
                }
                (Some(Operand::Register(rn)), Some(Operand::Register(rm))) => {
                    let n = rn.arm_reg_num();
                    let m = rm.arm_reg_num();
                    if n <= 7 && m <= 7 {
                        // T1: low reg CMP
                        emit16(&mut buf, 0x4280 | ((m as u16) << 3) | (n as u16));
                    } else {
                        // T2: high reg CMP — 0x4500 | N<<7 | Rm<<3 | Rn_lo
                        let n_hi = (n >> 3) & 1;
                        let n_lo = n & 0x7;
                        emit16(
                            &mut buf,
                            0x4500 | ((n_hi as u16) << 7) | ((m as u16) << 3) | (n_lo as u16),
                        );
                    }
                }
                _ => {
                    return Err(invalid_ops(
                        "cmp",
                        "expected register, immediate or register",
                        instr.span,
                    ))
                }
            }
        }

        "tst" => {
            // TST Rn, Rm: 0x4200 | Rm<<3 | Rn (both low)
            match (ops.first(), ops.get(1)) {
                (Some(Operand::Register(rn)), Some(Operand::Register(rm))) => {
                    let n = require_lo(*rn, "tst", instr.span)?;
                    let m = require_lo(*rm, "tst", instr.span)?;
                    emit16(&mut buf, 0x4200 | ((m as u16) << 3) | (n as u16));
                }
                _ => return Err(invalid_ops("tst", "expected two registers", instr.span)),
            }
        }

        "cmn" => {
            // CMN Rn, Rm: 0x42C0 | Rm<<3 | Rn
            match (ops.first(), ops.get(1)) {
                (Some(Operand::Register(rn)), Some(Operand::Register(rm))) => {
                    let n = require_lo(*rn, "cmn", instr.span)?;
                    let m = require_lo(*rm, "cmn", instr.span)?;
                    emit16(&mut buf, 0x42C0 | ((m as u16) << 3) | (n as u16));
                }
                _ => return Err(invalid_ops("cmn", "expected two registers", instr.span)),
            }
        }

        "add" | "adds" if !set_flags => {
            thumb_encode_add(&mut buf, ops, instr, base == "adds", &mut reloc)?;
        }
        "add" if set_flags => {
            thumb_encode_add(&mut buf, ops, instr, true, &mut reloc)?;
        }

        "sub" | "subs" if !set_flags => {
            thumb_encode_sub(&mut buf, ops, instr, base == "subs")?;
        }
        "sub" if set_flags => {
            thumb_encode_sub(&mut buf, ops, instr, true)?;
        }

        "and" | "orr" | "eor" | "bic" | "mvn" | "neg" | "rsb" | "adc" | "sbc" | "mul" | "ror" => {
            thumb_encode_alu_reg(&mut buf, base, ops, instr)?;
        }

        "lsl" | "lsr" | "asr" => {
            thumb_encode_shift(&mut buf, base, ops, instr)?;
        }

        "ldr" => thumb_encode_ldr_str(&mut buf, "ldr", ops, instr, &mut reloc)?,
        "str" => thumb_encode_ldr_str(&mut buf, "str", ops, instr, &mut reloc)?,
        "ldrb" => thumb_encode_ldr_str(&mut buf, "ldrb", ops, instr, &mut reloc)?,
        "strb" => thumb_encode_ldr_str(&mut buf, "strb", ops, instr, &mut reloc)?,
        "ldrh" => thumb_encode_ldr_str(&mut buf, "ldrh", ops, instr, &mut reloc)?,
        "strh" => thumb_encode_ldr_str(&mut buf, "strh", ops, instr, &mut reloc)?,
        "ldrsb" => thumb_encode_ldr_str_reg_only(&mut buf, "ldrsb", ops, instr)?,
        "ldrsh" => thumb_encode_ldr_str_reg_only(&mut buf, "ldrsh", ops, instr)?,

        // Thumb-2 wide instructions (explicit .w suffix)
        "add.w" | "adds.w" | "sub.w" | "subs.w" | "mov.w" | "movs.w" | "and.w" | "orr.w"
        | "eor.w" | "bic.w" | "mvn.w" | "lsl.w" | "lsr.w" | "asr.w" | "ror.w" => {
            thumb_encode_wide_dp(&mut buf, base, ops, instr)?;
        }

        "b.w" => {
            // Explicit wide B.W — always 32-bit
            match ops.first() {
                Some(Operand::Label(label_name)) => {
                    emit16(&mut buf, 0xF000);
                    emit16(&mut buf, 0x9000);
                    reloc = Some(Relocation {
                        offset: 0,
                        size: 4,
                        label: alloc::rc::Rc::from(label_name.as_str()),
                        kind: RelocKind::ThumbBranchW,
                        addend: 0,
                        trailing_bytes: 0,
                    });
                }
                _ => return Err(invalid_ops("b.w", "expected label", instr.span)),
            }
        }

        "it" | "ite" | "itt" | "itte" | "itet" | "itee" | "ittt" | "iteet" | "ittte" | "ittet"
        | "itett" | "ittee" | "itete" | "iteee" | "itttt" => {
            thumb_encode_it(&mut buf, mnemonic, ops, instr)?;
        }

        _ => {
            return Err(AsmError::UnknownMnemonic {
                mnemonic: String::from(mnemonic),
                arch: crate::error::ArchName::Thumb,
                span: instr.span,
            });
        }
    }

    Ok(EncodedInstr {
        bytes: buf,
        relocation: reloc,
        relax,
    })
}

/// Encode MOV for Thumb.
fn thumb_encode_mov(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    set_flags: bool,
) -> Result<(), AsmError> {
    match (ops.first(), ops.get(1)) {
        (Some(Operand::Register(rd)), Some(Operand::Immediate(imm))) => {
            // MOVS Rd, #imm8: [15:11]=00100 [10:8]=Rd [7:0]=imm8
            let d = require_lo(*rd, "mov", instr.span)?;
            let imm = *imm;
            if !(0..=255).contains(&imm) {
                return Err(invalid_ops(
                    "mov",
                    "immediate must be 0-255 for Thumb MOV",
                    instr.span,
                ));
            }
            emit16(buf, 0x2000 | ((d as u16) << 8) | (imm as u16));
        }
        (Some(Operand::Register(rd)), Some(Operand::Register(rs))) => {
            let d = rd.arm_reg_num();
            let s = rs.arm_reg_num();
            if d <= 7 && s <= 7 && set_flags {
                // MOVS Rd, Rm (low regs): actually LSL Rd, Rm, #0 = 0x0000 | Rm<<3 | Rd
                emit16(buf, ((s as u16) << 3) | (d as u16));
            } else {
                // MOV Rd, Rm (any reg): [15:8]=01000110 [7]=D [6:3]=Rm [2:0]=Rd_lo
                let d_hi = (d >> 3) & 1;
                let d_lo = d & 0x7;
                emit16(
                    buf,
                    0x4600 | ((d_hi as u16) << 7) | ((s as u16) << 3) | (d_lo as u16),
                );
            }
        }
        _ => {
            return Err(invalid_ops(
                "mov",
                "expected register and immediate or register",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// Encode ADD for Thumb.
fn thumb_encode_add(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    set_flags: bool,
    _reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    match ops.len() {
        2 => {
            match (ops.first(), ops.get(1)) {
                (Some(Operand::Register(rd)), Some(Operand::Immediate(imm))) => {
                    // ADDS Rd, #imm8: [15:11]=00110 [10:8]=Rd [7:0]=imm8
                    let d = require_lo(*rd, "add", instr.span)?;
                    let imm = *imm;
                    if !(0..=255).contains(&imm) {
                        return Err(invalid_ops("add", "immediate must be 0-255", instr.span));
                    }
                    emit16(buf, 0x3000 | ((d as u16) << 8) | (imm as u16));
                }
                (Some(Operand::Register(rd)), Some(Operand::Register(rm))) => {
                    let d = rd.arm_reg_num();
                    let m = rm.arm_reg_num();
                    if d <= 7 && m <= 7 && set_flags {
                        // ADDS Rd, Rd, Rm: 3-reg low form
                        emit16(
                            buf,
                            0x1800 | ((m as u16) << 6) | ((d as u16) << 3) | (d as u16),
                        );
                    } else {
                        // ADD Rd, Rm (high): 0x4400 | D<<7 | Rm<<3 | Rd_lo
                        let d_hi = (d >> 3) & 1;
                        let d_lo = d & 0x7;
                        emit16(
                            buf,
                            0x4400 | ((d_hi as u16) << 7) | ((m as u16) << 3) | (d_lo as u16),
                        );
                    }
                }
                _ => {
                    return Err(invalid_ops(
                        "add",
                        "expected register and immediate or register",
                        instr.span,
                    ))
                }
            }
        }
        3 => {
            match (&ops[0], &ops[1], &ops[2]) {
                (Operand::Register(rd), Operand::Register(rn), Operand::Immediate(imm)) => {
                    let d = require_lo(*rd, "add", instr.span)?;
                    let n = require_lo(*rn, "add", instr.span)?;
                    let imm = *imm;
                    if (0..=7).contains(&imm) {
                        // ADDS Rd, Rn, #imm3: [15:9]=0001110 [8:6]=imm3 [5:3]=Rn [2:0]=Rd
                        emit16(
                            buf,
                            0x1C00 | ((imm as u16) << 6) | ((n as u16) << 3) | (d as u16),
                        );
                    } else if (0..=255).contains(&imm) && d == n {
                        // ADDS Rdn, #imm8
                        emit16(buf, 0x3000 | ((d as u16) << 8) | (imm as u16));
                    } else {
                        return Err(invalid_ops(
                            "add",
                            "immediate too large for 16-bit Thumb ADD",
                            instr.span,
                        ));
                    }
                }
                (Operand::Register(rd), Operand::Register(rn), Operand::Register(rm)) => {
                    let d = require_lo(*rd, "add", instr.span)?;
                    let n = require_lo(*rn, "add", instr.span)?;
                    let m = require_lo(*rm, "add", instr.span)?;
                    // ADDS Rd, Rn, Rm: [15:9]=0001100 [8:6]=Rm [5:3]=Rn [2:0]=Rd
                    emit16(
                        buf,
                        0x1800 | ((m as u16) << 6) | ((n as u16) << 3) | (d as u16),
                    );
                }
                (Operand::Register(rd), Operand::Register(rn), _)
                    if rd.arm_reg_num() == 13 || rn.arm_reg_num() == 13 =>
                {
                    // ADD SP, SP, #imm or ADD Rd, SP, #imm
                    return Err(invalid_ops(
                        "add",
                        "SP arithmetic not yet supported",
                        instr.span,
                    ));
                }
                _ => {
                    return Err(invalid_ops(
                        "add",
                        "invalid operands for Thumb ADD",
                        instr.span,
                    ))
                }
            }
        }
        _ => return Err(invalid_ops("add", "expected 2 or 3 operands", instr.span)),
    }
    Ok(())
}

/// Encode SUB for Thumb.
fn thumb_encode_sub(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    _set_flags: bool,
) -> Result<(), AsmError> {
    match ops.len() {
        2 => {
            match (ops.first(), ops.get(1)) {
                (Some(Operand::Register(rd)), Some(Operand::Immediate(imm))) => {
                    // SUBS Rd, #imm8: [15:11]=00111 [10:8]=Rd [7:0]=imm8
                    let d = require_lo(*rd, "sub", instr.span)?;
                    let imm = *imm;
                    if !(0..=255).contains(&imm) {
                        return Err(invalid_ops("sub", "immediate must be 0-255", instr.span));
                    }
                    emit16(buf, 0x3800 | ((d as u16) << 8) | (imm as u16));
                }
                _ => {
                    return Err(invalid_ops(
                        "sub",
                        "expected register and immediate",
                        instr.span,
                    ))
                }
            }
        }
        3 => {
            match (&ops[0], &ops[1], &ops[2]) {
                (Operand::Register(rd), Operand::Register(rn), Operand::Immediate(imm)) => {
                    let d = require_lo(*rd, "sub", instr.span)?;
                    let n = require_lo(*rn, "sub", instr.span)?;
                    let imm = *imm;
                    if (0..=7).contains(&imm) {
                        // SUBS Rd, Rn, #imm3: [15:9]=0001111 [8:6]=imm3 [5:3]=Rn [2:0]=Rd
                        emit16(
                            buf,
                            0x1E00 | ((imm as u16) << 6) | ((n as u16) << 3) | (d as u16),
                        );
                    } else if (0..=255).contains(&imm) && d == n {
                        // SUBS Rdn, #imm8
                        emit16(buf, 0x3800 | ((d as u16) << 8) | (imm as u16));
                    } else {
                        return Err(invalid_ops(
                            "sub",
                            "immediate too large for 16-bit Thumb SUB",
                            instr.span,
                        ));
                    }
                }
                (Operand::Register(rd), Operand::Register(rn), Operand::Register(rm)) => {
                    let d = require_lo(*rd, "sub", instr.span)?;
                    let n = require_lo(*rn, "sub", instr.span)?;
                    let m = require_lo(*rm, "sub", instr.span)?;
                    // SUBS Rd, Rn, Rm: [15:9]=0001101 [8:6]=Rm [5:3]=Rn [2:0]=Rd
                    emit16(
                        buf,
                        0x1A00 | ((m as u16) << 6) | ((n as u16) << 3) | (d as u16),
                    );
                }
                _ => return Err(invalid_ops("sub", "invalid operands", instr.span)),
            }
        }
        _ => return Err(invalid_ops("sub", "expected 2 or 3 operands", instr.span)),
    }
    Ok(())
}

/// Encode Thumb ALU register operations (AND, ORR, EOR, BIC, MVN, etc.)
fn thumb_encode_alu_reg(
    buf: &mut InstrBytes,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    // These are all 2-register, 16-bit Thumb data-processing.
    // Format: [15:6]=0100_00_xxxx [5:3]=Rm [2:0]=Rd
    // Encoding: Rd = Rd OP Rm
    let (rd, rm) = match (ops.first(), ops.get(1)) {
        (Some(Operand::Register(r1)), Some(Operand::Register(r2))) => (
            require_lo(*r1, base, instr.span)?,
            require_lo(*r2, base, instr.span)?,
        ),
        _ => return Err(invalid_ops(base, "expected two low registers", instr.span)),
    };

    let opcode: u16 = match base {
        "and" => 0x4000,
        "eor" => 0x4040,
        "adc" => 0x4140,
        "sbc" => 0x4180,
        "ror" => 0x41C0,
        "neg" | "rsb" => 0x4240, // NEG Rd, Rm = RSB Rd, Rm, #0 in Thumb: actually this is different
        "tst" => 0x4200,
        "cmn" => 0x42C0,
        "orr" => 0x4300,
        "mul" => 0x4340,
        "bic" => 0x4380,
        "mvn" => 0x43C0,
        _ => return Err(invalid_ops(base, "unknown ALU op", instr.span)),
    };

    emit16(buf, opcode | ((rm as u16) << 3) | (rd as u16));
    Ok(())
}

/// Encode Thumb shift operations (LSL, LSR, ASR with immediate or register).
fn thumb_encode_shift(
    buf: &mut InstrBytes,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    match (ops.first(), ops.get(1), ops.get(2)) {
        // 3-operand: LSL Rd, Rm, #imm5
        (
            Some(Operand::Register(rd)),
            Some(Operand::Register(rm)),
            Some(Operand::Immediate(imm)),
        ) => {
            let d = require_lo(*rd, base, instr.span)?;
            let m = require_lo(*rm, base, instr.span)?;
            let imm = *imm as u16;
            if imm > 31 {
                return Err(invalid_ops(base, "shift amount must be 0-31", instr.span));
            }
            let opcode = match base {
                "lsl" => 0x0000,
                "lsr" => 0x0800,
                "asr" => 0x1000,
                _ => unreachable!(),
            };
            emit16(buf, opcode | (imm << 6) | ((m as u16) << 3) | (d as u16));
        }
        // 2-operand register: LSL Rd, Rs (Rd = Rd << Rs)
        (Some(Operand::Register(rd)), Some(Operand::Register(rs)), None) => {
            let d = require_lo(*rd, base, instr.span)?;
            let s = require_lo(*rs, base, instr.span)?;
            let opcode = match base {
                "lsl" => 0x4080,
                "lsr" => 0x40C0,
                "asr" => 0x4100,
                _ => unreachable!(),
            };
            emit16(buf, opcode | ((s as u16) << 3) | (d as u16));
        }
        _ => return Err(invalid_ops(base, "expected 2 or 3 operands", instr.span)),
    }
    Ok(())
}

/// Encode Thumb LDR/STR instructions.
fn thumb_encode_ldr_str(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    let is_load = mnemonic.starts_with("ldr");
    let is_byte = mnemonic.ends_with('b');
    let is_half = mnemonic.ends_with('h');

    match (ops.first(), ops.get(1)) {
        // LDR Rt, [Rn, #imm] or [Rn, Rm]
        (Some(Operand::Register(rt)), Some(Operand::Memory(mem))) => {
            let t = require_lo(*rt, mnemonic, instr.span)?;
            let base_reg = match mem.base {
                Some(r) => r.arm_reg_num(),
                None => {
                    return Err(invalid_ops(
                        mnemonic,
                        "memory operand requires base register",
                        instr.span,
                    ))
                }
            };

            if let Some(idx_reg) = mem.index {
                // Register offset: [15:9]=format [8:6]=Rm [5:3]=Rn [2:0]=Rt
                let m = require_lo(idx_reg, mnemonic, instr.span)?;
                let n = require_lo(
                    Register::from_arm_num(base_reg).ok_or_else(|| {
                        invalid_ops(mnemonic, "invalid base register number", instr.span)
                    })?,
                    mnemonic,
                    instr.span,
                )?;
                let opcode = if is_byte {
                    if is_load {
                        0x5C00_u16
                    } else {
                        0x5400
                    }
                } else if is_half {
                    if is_load {
                        0x5A00_u16
                    } else {
                        0x5200
                    }
                } else if is_load {
                    0x5800_u16
                } else {
                    0x5000
                };
                emit16(
                    buf,
                    opcode | ((m as u16) << 6) | ((n as u16) << 3) | (t as u16),
                );
            } else {
                // Immediate offset
                let off = mem.disp;
                let n = base_reg;

                if n == 13 {
                    // SP-relative: LDR Rt, [SP, #imm8*4]
                    if is_byte || is_half {
                        return Err(invalid_ops(
                            mnemonic,
                            "LDRB/STRB/LDRH/STRH not available with SP-relative in Thumb",
                            instr.span,
                        ));
                    }
                    if !(0..=1020).contains(&off) || (off & 3) != 0 {
                        return Err(invalid_ops(
                            mnemonic,
                            "SP-relative offset must be 0-1020 and word-aligned",
                            instr.span,
                        ));
                    }
                    let imm8 = (off >> 2) as u16;
                    if is_load {
                        emit16(buf, 0x9800 | ((t as u16) << 8) | imm8);
                    } else {
                        emit16(buf, 0x9000 | ((t as u16) << 8) | imm8);
                    }
                } else {
                    let n = require_lo(
                        Register::from_arm_num(n).ok_or_else(|| {
                            invalid_ops(mnemonic, "invalid base register number", instr.span)
                        })?,
                        mnemonic,
                        instr.span,
                    )?;
                    if is_byte {
                        // Byte: imm5 directly
                        if !(0..=31).contains(&off) {
                            return Err(invalid_ops(
                                mnemonic,
                                "byte offset must be 0-31",
                                instr.span,
                            ));
                        }
                        let imm5 = off as u16;
                        let base_op = if is_load { 0x7800_u16 } else { 0x7000 };
                        emit16(buf, base_op | (imm5 << 6) | ((n as u16) << 3) | (t as u16));
                    } else if is_half {
                        // Halfword: imm5 * 2
                        if !(0..=62).contains(&off) || (off & 1) != 0 {
                            return Err(invalid_ops(
                                mnemonic,
                                "halfword offset must be 0-62 and half-aligned",
                                instr.span,
                            ));
                        }
                        let imm5 = (off >> 1) as u16;
                        let base_op = if is_load { 0x8800_u16 } else { 0x8000 };
                        emit16(buf, base_op | (imm5 << 6) | ((n as u16) << 3) | (t as u16));
                    } else {
                        // Word: imm5 * 4
                        if !(0..=124).contains(&off) || (off & 3) != 0 {
                            return Err(invalid_ops(
                                mnemonic,
                                "word offset must be 0-124 and word-aligned",
                                instr.span,
                            ));
                        }
                        let imm5 = (off >> 2) as u16;
                        let base_op = if is_load { 0x6800_u16 } else { 0x6000 };
                        emit16(buf, base_op | (imm5 << 6) | ((n as u16) << 3) | (t as u16));
                    }
                }
            }
        }
        // LDR Rt, =label (literal pool) — handled as LDR PC-relative
        (Some(Operand::Register(rt)), Some(Operand::Label(label))) if is_load => {
            let t = require_lo(*rt, mnemonic, instr.span)?;
            // LDR Rt, [PC, #imm8×4] — 16-bit PC-relative literal load
            // Encoding: 01001 Rt(3) imm8(8) — offset filled by linker
            let hw = 0x4800_u16 | ((t as u16) << 8);
            let reloc_offset = buf.len();
            emit16(buf, hw);
            *reloc = Some(Relocation {
                offset: reloc_offset,
                size: 2,
                label: alloc::rc::Rc::from(&**label),
                kind: RelocKind::ThumbLdrLit8,
                addend: 0,
                trailing_bytes: 0,
            });
            return Ok(());
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "expected Rt, [Rn, #offset] or Rt, [Rn, Rm]",
                instr.span,
            ));
        }
    }
    let _ = reloc;
    Ok(())
}

/// Encode Thumb LDRSB/LDRSH (register-offset only).
fn thumb_encode_ldr_str_reg_only(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    match (ops.first(), ops.get(1)) {
        (Some(Operand::Register(rt)), Some(Operand::Memory(mem))) => {
            let t = require_lo(*rt, mnemonic, instr.span)?;
            let base_reg = match mem.base {
                Some(r) => r.arm_reg_num(),
                None => return Err(invalid_ops(mnemonic, "requires base register", instr.span)),
            };
            let idx_reg = match mem.index {
                Some(r) => r,
                None => {
                    return Err(invalid_ops(
                        mnemonic,
                        "register offset only in Thumb",
                        instr.span,
                    ))
                }
            };
            let n = require_lo(
                Register::from_arm_num(base_reg).ok_or_else(|| {
                    invalid_ops(mnemonic, "invalid base register number", instr.span)
                })?,
                mnemonic,
                instr.span,
            )?;
            let m = require_lo(idx_reg, mnemonic, instr.span)?;
            let opcode = match mnemonic {
                "ldrsb" => 0x5600_u16,
                "ldrsh" => 0x5E00_u16,
                _ => unreachable!(),
            };
            emit16(
                buf,
                opcode | ((m as u16) << 6) | ((n as u16) << 3) | (t as u16),
            );
        }
        _ => return Err(invalid_ops(mnemonic, "expected Rt, [Rn, Rm]", instr.span)),
    }
    Ok(())
}

/// Encode Thumb-2 wide data-processing instructions.
fn thumb_encode_wide_dp(
    buf: &mut InstrBytes,
    base: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    // Strip .w suffix
    let mnemonic = base.trim_end_matches(".w");
    let (mnemonic, _set_flags) = if mnemonic.ends_with('s') && mnemonic.len() > 1 {
        (&mnemonic[..mnemonic.len() - 1], true)
    } else {
        (mnemonic, false)
    };

    match (ops.first(), ops.get(1), ops.get(2)) {
        (Some(Operand::Register(rd)), Some(Operand::Register(rn)), Some(Operand::Register(rm))) => {
            let d = rd.arm_reg_num();
            let n = rn.arm_reg_num();
            let m = rm.arm_reg_num();
            // Thumb-2 register-register data processing (encoding T3 in various categories)
            // hw1: 1110 1010 opc2 S Rn
            // hw2: 0 imm3 Rd imm2 type Rm
            let (opc, s_bit) = match mnemonic {
                "and" => (0b0000_u16, 0_u16),
                "eor" => (0b0100, 0),
                "add" => (0b1000, 0),
                "adc" => (0b1010, 0),
                "sbc" => (0b1011, 0),
                "sub" => (0b1101, 0),
                "orr" => (0b0010, 0),
                "bic" => (0b0001, 0),
                "mvn" => (0b0011, 0), // actually ORN, but MVN is special
                _ => {
                    return Err(invalid_ops(base, "unsupported wide operation", instr.span));
                }
            };
            let _hw1: u16 = 0xEA00 | (s_bit << 4) | (opc << 5 >> 1) | ((n as u16) & 0xF);
            // Actually the Thumb-2 encoding is more complex. Let me use the correct encoding:
            // For register variant (shifted register):
            // hw1 = 1110_101 opc1 S Rn  (bits 15:0)
            // hw2 = (0) imm3 Rd imm2 type Rm
            // Where opc1 gives the operation.
            // Simplified for type=LSL, imm=0 (no shift):
            let hw1: u16 = 0xEA00 | ((opc & 0xE) << 4) | ((opc & 1) << 4) | (n as u16 & 0xF);
            let hw2: u16 = ((d as u16 & 0xF) << 8) | (m as u16 & 0xF);
            emit16(buf, hw1);
            emit16(buf, hw2);
        }
        (
            Some(Operand::Register(rd)),
            Some(Operand::Register(rn)),
            Some(Operand::Immediate(imm)),
        ) => {
            // Thumb-2 modified immediate
            let d = rd.arm_reg_num();
            let n = rn.arm_reg_num();
            let imm = *imm as u32;
            // T32 modified immediate: encode into i:imm3:imm8 (12-bit encoding)
            let encoded = encode_thumb2_modified_imm(imm).ok_or_else(|| {
                invalid_ops(
                    base,
                    "immediate not encodable in Thumb-2 modified immediate",
                    instr.span,
                )
            })?;
            let opc = match mnemonic {
                "add" => 0b1000_u16,
                "sub" => 0b1101,
                "and" => 0b0000,
                "orr" => 0b0010,
                "eor" => 0b0100,
                "bic" => 0b0001,
                _ => {
                    return Err(invalid_ops(
                        base,
                        "unsupported wide immediate operation",
                        instr.span,
                    ))
                }
            };
            // hw1: 1111 0 i 0 op[3:1] S op[0] Rn
            let i_bit = (encoded >> 11) & 1;
            let imm3 = (encoded >> 8) & 0x7;
            let imm8 = encoded & 0xFF;
            let s_bit = if _set_flags { 1_u16 } else { 0 };
            let hw1: u16 =
                0xF000 | (i_bit << 10) | ((opc >> 1) << 5) | (s_bit << 4) | (n as u16 & 0xF);
            let hw2: u16 = ((imm3 as u16) << 12) | (((d as u16) & 0xF) << 8) | (imm8 as u16);
            emit16(buf, hw1);
            emit16(buf, hw2);
        }
        _ => {
            return Err(invalid_ops(
                base,
                "invalid operands for wide instruction",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// Encode Thumb IT instruction.
fn thumb_encode_it(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    // IT{x{y{z}}} <firstcond>
    // The firstcond is given as the single operand (condition suffix)
    // Format: [15:8]=10111111 [7:4]=firstcond [3:0]=mask

    // The condition is specified as either a register-like name or via the operand
    // In GAS syntax: IT EQ — but our parser will see this as: mnemonic="it", ops=[Register/Label "eq"]
    // Let's accept the condition as a label-like identifier
    let cond_str = match ops.first() {
        Some(Operand::Label(name)) => name.as_str(),
        Some(Operand::Register(_)) => {
            // Could be parsed as register if condition matches a register name
            return Err(invalid_ops(
                mnemonic,
                "expected condition code (eq, ne, cs, etc.)",
                instr.span,
            ));
        }
        _ => return Err(invalid_ops(mnemonic, "expected condition code", instr.span)),
    };

    let firstcond = match cond_str {
        "eq" => 0x0_u8,
        "ne" => 0x1,
        "cs" | "hs" => 0x2,
        "cc" | "lo" => 0x3,
        "mi" => 0x4,
        "pl" => 0x5,
        "vs" => 0x6,
        "vc" => 0x7,
        "hi" => 0x8,
        "ls" => 0x9,
        "ge" => 0xA,
        "lt" => 0xB,
        "gt" => 0xC,
        "le" => 0xD,
        "al" => 0xE,
        _ => return Err(invalid_ops(mnemonic, "unknown condition code", instr.span)),
    };

    // Build mask from the IT block suffix (t=then, e=else)
    let suffix = &mnemonic[2..]; // skip "it"
    let fc_bit0 = firstcond & 1; // LSB of firstcond (then/else sense)

    let mask: u8 = match suffix.len() {
        0 => {
            // IT: mask = 1000
            0b1000
        }
        1 => {
            // ITx: mask = xy10  where x=T/E sense
            let bit = if suffix.as_bytes()[0] == b't' {
                fc_bit0
            } else {
                fc_bit0 ^ 1
            };
            (bit << 3) | 0b0100
        }
        2 => {
            let b1 = if suffix.as_bytes()[0] == b't' {
                fc_bit0
            } else {
                fc_bit0 ^ 1
            };
            let b2 = if suffix.as_bytes()[1] == b't' {
                fc_bit0
            } else {
                fc_bit0 ^ 1
            };
            (b1 << 3) | (b2 << 2) | 0b0010
        }
        3 => {
            let b1 = if suffix.as_bytes()[0] == b't' {
                fc_bit0
            } else {
                fc_bit0 ^ 1
            };
            let b2 = if suffix.as_bytes()[1] == b't' {
                fc_bit0
            } else {
                fc_bit0 ^ 1
            };
            let b3 = if suffix.as_bytes()[2] == b't' {
                fc_bit0
            } else {
                fc_bit0 ^ 1
            };
            (b1 << 3) | (b2 << 2) | (b3 << 1) | 0b0001
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "IT block too long (max 4 instructions)",
                instr.span,
            ))
        }
    };

    emit16(buf, 0xBF00 | ((firstcond as u16) << 4) | (mask as u16));
    Ok(())
}

/// Encode Thumb-2 BL offset into two halfwords.
fn encode_thumb2_bl(offset: i32) -> (u16, u16) {
    // BL encoding (T1): offset is already >>1
    let s = if offset < 0 { 1_u16 } else { 0 };
    let imm = offset as u32;
    let imm10 = (imm >> 11) & 0x3FF;
    let imm11 = imm & 0x7FF;
    let j1 = ((!(imm >> 23) ^ s as u32) & 1) as u16; // J1 = NOT(bit23 XOR S)
    let j2 = ((!(imm >> 22) ^ s as u32) & 1) as u16; // J2 = NOT(bit22 XOR S)

    let hw1 = 0xF000 | (s << 10) | (imm10 as u16);
    let hw2 = 0xD000 | (j1 << 13) | (j2 << 11) | (imm11 as u16);
    (hw1, hw2)
}

/// Encode Thumb-2 B.W (unconditional wide branch) offset into two halfwords.
fn encode_thumb2_branch(offset: i32) -> (u16, u16) {
    let s = if offset < 0 { 1_u16 } else { 0 };
    let imm = offset as u32;
    let imm10 = (imm >> 11) & 0x3FF;
    let imm11 = imm & 0x7FF;
    let j1 = ((!(imm >> 23) ^ s as u32) & 1) as u16;
    let j2 = ((!(imm >> 22) ^ s as u32) & 1) as u16;

    let hw1 = 0xF000 | (s << 10) | (imm10 as u16);
    let hw2 = 0x9000 | (j1 << 13) | (j2 << 11) | (imm11 as u16);
    (hw1, hw2)
}

/// Encode Thumb-2 conditional branch (B<cond>.W) offset into two halfwords.
fn encode_thumb2_cond_branch(offset: i32, cond: u16) -> (u16, u16) {
    // Encoding T3: hw1=11110 S cond imm6, hw2=10 J1 0 J2 imm11
    let s = if offset < 0 { 1_u16 } else { 0 };
    let imm = offset as u32;
    let imm6 = (imm >> 11) & 0x3F;
    let imm11 = imm & 0x7FF;
    let j1 = ((imm >> 17) & 1) as u16;
    let j2 = ((imm >> 18) & 1) as u16;

    let hw1 = 0xF000 | (s << 10) | ((cond & 0xF) << 6) | (imm6 as u16);
    let hw2 = 0x8000 | (j1 << 13) | (j2 << 11) | (imm11 as u16);
    (hw1, hw2)
}

/// Encode a Thumb-2 modified immediate constant (12-bit: i:imm3:imm8).
fn encode_thumb2_modified_imm(val: u32) -> Option<u16> {
    // Case 1: 0-255 fits in imm8 directly (rotation = 0)
    if val <= 255 {
        return Some(val as u16);
    }
    // Case 2: 0x00XY00XY pattern
    if (val >> 16) == (val & 0xFFFF) && (val & 0xFF00) == 0 {
        return Some(0x100 | (val & 0xFF) as u16);
    }
    // Case 3: 0xXY00XY00 pattern
    if (val >> 16) == (val & 0xFFFF) && (val & 0xFF) == 0 {
        return Some(0x200 | ((val >> 8) & 0xFF) as u16);
    }
    // Case 4: 0xXYXYXYXY pattern
    if (val >> 24) == ((val >> 16) & 0xFF)
        && (val >> 24) == ((val >> 8) & 0xFF)
        && (val >> 24) == (val & 0xFF)
    {
        return Some(0x300 | (val & 0xFF) as u16);
    }
    // Case 5: Rotated 8-bit value
    for rot in 8..=31_u32 {
        let unrotated = val.rotate_left(rot);
        if unrotated <= 255 && (unrotated & 0x80) != 0 {
            return Some(((rot as u16) << 7) | (unrotated as u16 & 0x7F));
        }
    }
    None
}

/// Helper: create ARM register from number, returns None for invalid numbers.
impl Register {
    fn from_arm_num(n: u8) -> Option<Register> {
        use Register::*;
        match n {
            0 => Some(ArmR0),
            1 => Some(ArmR1),
            2 => Some(ArmR2),
            3 => Some(ArmR3),
            4 => Some(ArmR4),
            5 => Some(ArmR5),
            6 => Some(ArmR6),
            7 => Some(ArmR7),
            8 => Some(ArmR8),
            9 => Some(ArmR9),
            10 => Some(ArmR10),
            11 => Some(ArmR11),
            12 => Some(ArmR12),
            13 => Some(ArmSp),
            14 => Some(ArmLr),
            15 => Some(ArmPc),
            _ => None,
        }
    }
}

// ── Public entry point ───────────────────────────────────────────────────

/// Encode an ARM32 (A32) or Thumb (T16/T32) instruction.
pub fn encode_arm(instr: &Instruction, arch: Arch) -> Result<EncodedInstr, AsmError> {
    if arch == Arch::Thumb {
        return encode_thumb(instr);
    }

    let mut buf = InstrBytes::new();
    let mut reloc: Option<Relocation> = None;

    let mnemonic = instr.mnemonic.as_str();
    let ops = &instr.operands;

    // Parse condition suffix + optional S flag
    let (base_with_s, cond) = parse_cond(mnemonic);
    let (base, set_flags) = if base_with_s.ends_with('s')
        && base_with_s.len() > 1
        && (dp_opcode(&base_with_s[..base_with_s.len() - 1]).is_some()
            || matches!(
                &base_with_s[..base_with_s.len() - 1],
                "mul" | "mla" | "umull" | "smull" | "umlal" | "smlal"
            )) {
        (&base_with_s[..base_with_s.len() - 1], true)
    } else {
        (base_with_s, false)
    };

    // Dispatch to instruction class
    if dp_opcode(base).is_some() {
        encode_dp(&mut buf, cond, base, ops, instr, set_flags)?;
    } else {
        match base {
            "nop" => encode_nop(&mut buf, cond),
            "bkpt" => encode_bkpt(&mut buf, ops, instr)?,
            "b" | "bl" => encode_branch(&mut buf, cond, base, ops, instr, &mut reloc)?,
            "bx" => encode_bx(&mut buf, cond, false, ops, instr)?,
            "blx" => encode_bx(&mut buf, cond, true, ops, instr)?,
            "ldr" | "str" | "ldrb" | "strb" => {
                encode_ldr_str(&mut buf, cond, base, ops, instr, &mut reloc)?
            }
            "ldrh" | "strh" | "ldrsb" | "ldrsh" => {
                encode_ldr_str_h(&mut buf, cond, base, ops, instr)?
            }
            "push" => encode_push_pop(&mut buf, cond, "push", ops, instr)?,
            "pop" => encode_push_pop(&mut buf, cond, "pop", ops, instr)?,
            "mul" | "mla" => encode_mul(&mut buf, cond, base, ops, instr, set_flags)?,
            "umull" | "smull" | "umlal" | "smlal" => {
                encode_long_mul(&mut buf, cond, base, ops, instr, set_flags)?
            }
            "movw" => encode_movw_movt(&mut buf, cond, "movw", ops, instr)?,
            "movt" => encode_movw_movt(&mut buf, cond, "movt", ops, instr)?,
            "svc" | "swi" => encode_svc(&mut buf, cond, ops, instr)?,
            "adr" => encode_adr(&mut buf, cond, ops, instr, &mut reloc)?,
            "clz" => encode_clz(&mut buf, cond, ops, instr)?,
            "rev" | "rev16" | "rbit" => encode_rev(&mut buf, cond, base, ops, instr)?,
            "bfc" | "bfi" | "sbfx" | "ubfx" => encode_bitfield(&mut buf, cond, base, ops, instr)?,
            "uxtb" | "uxth" | "sxtb" | "sxth" => encode_extend(&mut buf, cond, base, ops, instr)?,
            "mrs" => encode_mrs(&mut buf, cond, ops, instr)?,
            "msr" => encode_msr(&mut buf, cond, ops, instr)?,
            "ldrex" | "ldrexb" | "ldrexh" | "ldrexd" => {
                encode_ldrex(&mut buf, cond, base, ops, instr)?
            }
            "strex" | "strexb" | "strexh" | "strexd" => {
                encode_strex(&mut buf, cond, base, ops, instr)?
            }
            "dmb" | "dsb" | "isb" => encode_barrier(&mut buf, base, ops, instr)?,
            "ldm" | "ldmia" | "ldmfd" | "ldmdb" | "ldmea" | "ldmib" | "ldmed" | "ldmda"
            | "ldmfa" | "stm" | "stmia" | "stmea" | "stmdb" | "stmfd" | "stmib" | "stmfa"
            | "stmda" | "stmed" => encode_ldm_stm(&mut buf, cond, base, ops, instr)?,
            _ => {
                return Err(AsmError::UnknownMnemonic {
                    mnemonic: String::from(mnemonic),
                    arch: crate::error::ArchName::Arm,
                    span: instr.span,
                });
            }
        }
    }

    Ok(EncodedInstr {
        bytes: buf,
        relocation: reloc,
        relax: None,
    })
}

#[cfg(test)]
mod tests {
    use crate::assemble;
    use crate::ir::Arch;

    /// Helper: assemble one ARM instruction and return its 4 bytes as u32 (LE).
    fn arm(src: &str) -> u32 {
        let bytes = assemble(src, Arch::Arm).unwrap();
        assert_eq!(bytes.len(), 4, "ARM instruction must be 4 bytes: {src}");
        u32::from_le_bytes(bytes[..4].try_into().unwrap())
    }

    // ── Data processing (encode_dp) ───────────────────────────────

    #[test]
    fn dp_mov_imm() {
        // MOV R0, 42 → E3A0002A
        assert_eq!(arm("mov r0, 42"), 0xE3A0_002A);
    }

    #[test]
    fn dp_add_regs() {
        // ADD R0, R1, R2 → E0810002
        assert_eq!(arm("add r0, r1, r2"), 0xE081_0002);
    }

    #[test]
    fn dp_sub_imm() {
        // SUB R3, R3, 1 → E2433001
        assert_eq!(arm("sub r3, r3, 1"), 0xE243_3001);
    }

    #[test]
    fn dp_cmp() {
        // CMP R0, 0 → E3500000
        assert_eq!(arm("cmp r0, 0"), 0xE350_0000);
    }

    #[test]
    fn dp_and_reg() {
        // AND R0, R1, R2 → E0010002
        assert_eq!(arm("and r0, r1, r2"), 0xE001_0002);
    }

    #[test]
    fn dp_orr_imm() {
        // ORR R0, R0, 0xFF → E38000FF
        assert_eq!(arm("orr r0, r0, 0xFF"), 0xE380_00FF);
    }

    #[test]
    fn dp_eor_reg() {
        // EOR R0, R1, R2 → E0210002
        assert_eq!(arm("eor r0, r1, r2"), 0xE021_0002);
    }

    #[test]
    fn dp_bic_imm() {
        // BIC R0, R0, 0xF → E3C0000F
        assert_eq!(arm("bic r0, r0, 0xF"), 0xE3C0_000F);
    }

    #[test]
    fn dp_mvn_reg() {
        // MVN R0, R1 → E1E00001
        assert_eq!(arm("mvn r0, r1"), 0xE1E0_0001);
    }

    #[test]
    fn dp_rsb_imm() {
        // RSB R0, R0, 0 → E2600000 (negate: R0 = 0 - R0)
        assert_eq!(arm("rsb r0, r0, 0"), 0xE260_0000);
    }

    #[test]
    fn dp_adds_set_flags() {
        // ADDS R0, R1, R2 → E0910002
        assert_eq!(arm("adds r0, r1, r2"), 0xE091_0002);
    }

    #[test]
    fn dp_conditional() {
        // MOVEQ R0, 1 → 03A00001
        assert_eq!(arm("moveq r0, 1"), 0x03A0_0001);
    }

    #[test]
    fn dp_shifted_reg() {
        // ADD R0, R1, R2, LSL, 3 → E0810182
        assert_eq!(arm("add r0, r1, r2, lsl, 3"), 0xE081_0182);
    }

    #[test]
    fn dp_two_operand_form() {
        // MOV R0, R1 → shorthand for data processing
        assert_eq!(arm("mov r0, r1"), 0xE1A0_0001);
    }

    // ── Load/Store (encode_ldr_str) ───────────────────────────────

    #[test]
    fn ldr_reg_offset() {
        // LDR R0, [R1] → E5910000
        assert_eq!(arm("ldr r0, [r1]"), 0xE591_0000);
    }

    #[test]
    fn str_reg_offset() {
        // STR R0, [R1] → E5810000
        assert_eq!(arm("str r0, [r1]"), 0xE581_0000);
    }

    #[test]
    fn ldr_imm_offset() {
        // LDR R0, [R1, 4] → E5910004
        assert_eq!(arm("ldr r0, [r1, 4]"), 0xE591_0004);
    }

    #[test]
    fn ldrb_reg() {
        // LDRB R0, [R1] → E5D10000
        assert_eq!(arm("ldrb r0, [r1]"), 0xE5D1_0000);
    }

    #[test]
    fn strb_reg() {
        // STRB R0, [R1] → E5C10000
        assert_eq!(arm("strb r0, [r1]"), 0xE5C1_0000);
    }

    // ── Branch (encode_branch) ───────────────────────────────────

    #[test]
    fn branch_self() {
        // B target with label forward
        let bytes = assemble("b target\ntarget:\nnop", Arch::Arm).unwrap();
        let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(word & 0xFF00_0000, 0xEA00_0000); // B AL
    }

    #[test]
    fn bl_instruction() {
        let bytes = assemble("bl target\ntarget:\nnop", Arch::Arm).unwrap();
        let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(word & 0xFF00_0000, 0xEB00_0000); // BL AL
    }

    // ── BX / BLX (encode_bx) ─────────────────────────────────────

    #[test]
    fn bx_lr() {
        // BX LR → E12FFF1E
        assert_eq!(arm("bx lr"), 0xE12F_FF1E);
    }

    #[test]
    fn blx_reg() {
        // BLX R0 → E12FFF30
        assert_eq!(arm("blx r0"), 0xE12F_FF30);
    }

    // ── Push/Pop (encode_push_pop) ────────────────────────────────

    #[test]
    fn push_single() {
        // PUSH {LR} → E92D4000
        assert_eq!(arm("push {lr}"), 0xE92D_4000);
    }

    #[test]
    fn pop_single() {
        // POP {PC} → E8BD8000
        assert_eq!(arm("pop {pc}"), 0xE8BD_8000);
    }

    #[test]
    fn push_multi() {
        // PUSH {R4, R5, LR} → E92D4030
        assert_eq!(arm("push {r4, r5, lr}"), 0xE92D_4030);
    }

    // ── Multiply (encode_mul, encode_long_mul) ────────────────────

    #[test]
    fn mul_basic() {
        // MUL R0, R1, R2 → E0000291
        assert_eq!(arm("mul r0, r1, r2"), 0xE000_0291);
    }

    #[test]
    fn mla_basic() {
        // MLA R0, R1, R2, R3 → E0203291
        assert_eq!(arm("mla r0, r1, r2, r3"), 0xE020_3291);
    }

    #[test]
    fn umull_basic() {
        // UMULL RdLo=R0, RdHi=R1, Rm=R2, Rs=R3
        assert_eq!(arm("umull r0, r1, r2, r3"), 0xE041_0392);
    }

    #[test]
    fn smull_basic() {
        // SMULL RdLo=R0, RdHi=R1, Rm=R2, Rs=R3
        assert_eq!(arm("smull r0, r1, r2, r3"), 0xE001_0392);
    }

    // ── MOVW / MOVT (encode_movw_movt) ───────────────────────────

    #[test]
    fn movw_imm16() {
        // MOVW R0, 0x1234 → E3010234
        assert_eq!(arm("movw r0, 0x1234"), 0xE301_0234);
    }

    #[test]
    fn movt_imm16() {
        // MOVT R0, 0x5678 → E3450678
        assert_eq!(arm("movt r0, 0x5678"), 0xE345_0678);
    }

    // ── SVC (encode_svc) ─────────────────────────────────────────

    #[test]
    fn svc_imm() {
        // SVC 0 → EF000000
        assert_eq!(arm("svc 0"), 0xEF00_0000);
    }

    // ── NOP (encode_nop) ─────────────────────────────────────────

    #[test]
    fn nop() {
        // NOP = MOV R0, R0 → E1A00000
        assert_eq!(arm("nop"), 0xE1A0_0000);
    }

    // ── BKPT (encode_bkpt) ───────────────────────────────────────

    #[test]
    fn bkpt_imm() {
        // BKPT 0 → E1200070
        assert_eq!(arm("bkpt 0"), 0xE120_0070);
    }

    // ── CLZ (encode_clz) ─────────────────────────────────────────

    #[test]
    fn clz_basic() {
        // CLZ R0, R1 → E16F0F11
        assert_eq!(arm("clz r0, r1"), 0xE16F_0F11);
    }

    // ── REV (encode_rev) ─────────────────────────────────────────

    #[test]
    fn rev_basic() {
        // REV R0, R1 → E6BF0F31
        assert_eq!(arm("rev r0, r1"), 0xE6BF_0F31);
    }

    #[test]
    fn rev16_basic() {
        // REV16 R0, R1 → E6BF0FB1
        assert_eq!(arm("rev16 r0, r1"), 0xE6BF_0FB1);
    }

    #[test]
    fn rbit_basic() {
        // RBIT R0, R1 → E6FF0F31
        assert_eq!(arm("rbit r0, r1"), 0xE6FF_0F31);
    }

    // ── Extend (encode_extend) ───────────────────────────────────

    #[test]
    fn uxtb_basic() {
        // UXTB R0, R1 → E6EF0071
        assert_eq!(arm("uxtb r0, r1"), 0xE6EF_0071);
    }

    #[test]
    fn uxth_basic() {
        // UXTH R0, R1 → E6FF0071
        assert_eq!(arm("uxth r0, r1"), 0xE6FF_0071);
    }

    #[test]
    fn sxtb_basic() {
        // SXTB R0, R1 → E6AF0071
        assert_eq!(arm("sxtb r0, r1"), 0xE6AF_0071);
    }

    #[test]
    fn sxth_basic() {
        // SXTH R0, R1 → E6BF0071
        assert_eq!(arm("sxth r0, r1"), 0xE6BF_0071);
    }

    // ── Halfword Load/Store (encode_ldr_str_h) ───────────────────

    #[test]
    fn ldrh_basic() {
        // LDRH R0, [R1] → E1D100B0
        assert_eq!(arm("ldrh r0, [r1]"), 0xE1D1_00B0);
    }

    #[test]
    fn strh_basic() {
        // STRH R0, [R1] → E1C100B0
        assert_eq!(arm("strh r0, [r1]"), 0xE1C1_00B0);
    }

    #[test]
    fn ldrsb_basic() {
        // LDRSB R0, [R1] → E1D100D0
        assert_eq!(arm("ldrsb r0, [r1]"), 0xE1D1_00D0);
    }

    #[test]
    fn ldrsh_basic() {
        // LDRSH R0, [R1] → E1D100F0
        assert_eq!(arm("ldrsh r0, [r1]"), 0xE1D1_00F0);
    }

    #[test]
    fn ldrh_preindex() {
        // LDRH R0, [R1, #4]! → cond|000|P=1|U=1|1|W=1|L=1|Rn=0001|Rd=0000|0000|1011|0100
        assert_eq!(arm("ldrh r0, [r1, 4]!"), 0xE1F1_00B4);
    }

    #[test]
    fn strh_preindex_neg() {
        // STRH R0, [R1, #-8]! → P=1, U=0, W=1, L=0
        assert_eq!(arm("strh r0, [r1, -8]!"), 0xE16100B8);
    }

    #[test]
    fn ldrh_postindex() {
        // LDRH R0, [R1], #4 → P=0, U=1, W=0, L=1
        assert_eq!(arm("ldrh r0, [r1], 4"), 0xE0D1_00B4);
    }

    #[test]
    fn strh_postindex_neg() {
        // STRH R0, [R1], #-4 → P=0, U=0, W=0, L=0
        assert_eq!(arm("strh r0, [r1], -4"), 0xE041_00B4);
    }

    // ── LDREX / STREX (encode_ldrex, encode_strex) ───────────────

    #[test]
    fn ldrex_basic() {
        // LDREX R0, [R1] → E1910F9F
        assert_eq!(arm("ldrex r0, [r1]"), 0xE191_0F9F);
    }

    #[test]
    fn strex_basic() {
        // STREX R0, R1, [R2] → E1820F91
        assert_eq!(arm("strex r0, r1, [r2]"), 0xE182_0F91);
    }

    // ── Barriers (encode_barrier) ────────────────────────────────

    #[test]
    fn dmb_sy() {
        // DMB SY (full system barrier, option=0xF) → F57FF05F
        assert_eq!(arm("dmb 0xF"), 0xF57F_F05F);
    }

    #[test]
    fn dsb_sy() {
        // DSB SY (full system barrier, option=0xF) → F57FF04F
        assert_eq!(arm("dsb 0xF"), 0xF57F_F04F);
    }

    #[test]
    fn isb_basic() {
        // ISB → F57FF06F (SY default)
        assert_eq!(arm("isb"), 0xF57F_F06F);
    }

    // ── MRS / MSR (encode_mrs, encode_msr) ───────────────────────

    #[test]
    fn mrs_cpsr() {
        // MRS R0, CPSR → E10F0000
        assert_eq!(arm("mrs r0, cpsr"), 0xE10F_0000);
    }

    #[test]
    fn msr_cpsr_reg() {
        // MSR CPSR, R0 → E149F000
        assert_eq!(arm("msr cpsr, r0"), 0xE149_F000);
    }

    // ── LDM / STM (encode_ldm_stm) ──────────────────────────────

    #[test]
    fn ldmia_basic() {
        // LDMIA R0, {R1, R2} → E8900006
        assert_eq!(arm("ldmia r0, {r1, r2}"), 0xE890_0006);
    }

    #[test]
    fn stmdb_basic() {
        // STMDB SP!, {R4, LR} → E92D4010
        // (STMDB SP! is same encoding as PUSH)
        assert_eq!(arm("stmdb sp!, {r4, lr}"), 0xE92D_4010);
    }

    // ── Error cases ──────────────────────────────────────────────

    // ── Bitfield (encode_bitfield) ───────────────────────────────

    #[test]
    fn bfc_r0_4_8() {
        // BFC R0, 4, 8  → clear bits 4..11 (msb=11)
        // cond=E, 0111110=3E<<21, msb=11<<16, Rd=0<<12, lsb=4<<7, 001<<4, 1111
        // = 0xE7CB021F
        assert_eq!(arm("bfc r0, 4, 8"), 0xE7CB_021F);
    }

    #[test]
    fn bfi_r0_r1_0_8() {
        // BFI R0, R1, 0, 8  → insert low 8 bits of r1 at bit 0 (msb=7)
        // = 0xE7C70011
        assert_eq!(arm("bfi r0, r1, 0, 8"), 0xE7C7_0011);
    }

    #[test]
    fn sbfx_r0_r1_4_8() {
        // SBFX R0, R1, 4, 8  → signed extract 8 bits at bit 4 (widthm1=7)
        // cond=E, 0111101=3D<<21, widthm1=7<<16, Rd=0<<12, lsb=4<<7, 101<<4, Rn=1
        // = 0xE7A70251
        assert_eq!(arm("sbfx r0, r1, 4, 8"), 0xE7A7_0251);
    }

    #[test]
    fn ubfx_r0_r1_4_8() {
        // UBFX R0, R1, 4, 8  → unsigned extract 8 bits at bit 4 (widthm1=7)
        // cond=E, 0111111=3F<<21, widthm1=7<<16, Rd=0<<12, lsb=4<<7, 101<<4, Rn=1
        // = 0xE7E70251
        assert_eq!(arm("ubfx r0, r1, 4, 8"), 0xE7E7_0251);
    }

    #[test]
    fn bfc_high_bits() {
        // BFC R3, 16, 16 → clear bits 16..31 (msb=31)
        // cond=E, 0111110, msb=31<<16, Rd=3<<12, lsb=16<<7, 001<<4, 1111
        // = 0xE7DF381F
        assert_eq!(arm("bfc r3, 16, 16"), 0xE7DF_381F);
    }

    #[test]
    fn bfi_conditional() {
        // BFINE R2, R3, 8, 4 → insert 4 bits from r3 at bit 8 (msb=11, cond=NE=1)
        // = 0x17CB2413
        assert_eq!(arm("bfine r2, r3, 8, 4"), 0x17CB_2413);
    }

    // ── Error cases ──────────────────────────────────────────────

    #[test]
    fn unknown_mnemonic() {
        let err = assemble("xyz", Arch::Arm).unwrap_err();
        assert!(matches!(
            err,
            crate::error::AsmError::UnknownMnemonic { .. }
        ));
    }

    #[test]
    fn dp_bad_operand() {
        let err = assemble("add r0, r1", Arch::Arm).unwrap_err();
        assert!(matches!(
            err,
            crate::error::AsmError::InvalidOperands { .. } | crate::error::AsmError::Syntax { .. }
        ));
    }

    // ── MOVW/MOVT fallback for non-encodable immediates ─────────

    #[test]
    fn mov_small_uses_dp_form() {
        // MOV R0, 0xFF should use the standard DP immediate form (4 bytes)
        let code = assemble("mov r0, 0xFF", Arch::Arm).unwrap();
        assert_eq!(code.len(), 4, "small immediate should be 4 bytes (DP form)");
    }

    #[test]
    fn mov_non_encodable_uses_movw() {
        // MOV R0, 0x1234 cannot be encoded as modified-imm → should use MOVW (4 bytes)
        let code = assemble("mov r0, 0x1234", Arch::Arm).unwrap();
        assert_eq!(code.len(), 4, "16-bit immediate should be single MOVW");
        let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
        // MOVW: cond=E, 0011|0|0|00|imm4|Rd|imm12
        assert_eq!(
            (word >> 20) & 0xFF,
            0x30,
            "opcode should be MOVW (0011_0000)"
        );
        // Rd = R0
        assert_eq!((word >> 12) & 0xF, 0);
        // imm4:imm12 = 0x1234
        let imm4 = (word >> 16) & 0xF;
        let imm12 = word & 0xFFF;
        let reconstructed = (imm4 << 12) | imm12;
        assert_eq!(reconstructed, 0x1234);
    }

    #[test]
    fn mov_large_uses_movw_movt() {
        // MOV R0, 0x12345678 → MOVW R0, #0x5678 + MOVT R0, #0x1234 (8 bytes)
        let code = assemble("mov r0, 0x12345678", Arch::Arm).unwrap();
        assert_eq!(code.len(), 8, "large immediate needs MOVW+MOVT pair");

        // First word: MOVW
        let w0 = u32::from_le_bytes(code[0..4].try_into().unwrap());
        assert_eq!((w0 >> 20) & 0xFF, 0x30, "first should be MOVW");
        let lo_imm4 = (w0 >> 16) & 0xF;
        let lo_imm12 = w0 & 0xFFF;
        assert_eq!((lo_imm4 << 12) | lo_imm12, 0x5678);

        // Second word: MOVT
        let w1 = u32::from_le_bytes(code[4..8].try_into().unwrap());
        assert_eq!((w1 >> 20) & 0xFF, 0x34, "second should be MOVT");
        let hi_imm4 = (w1 >> 16) & 0xF;
        let hi_imm12 = w1 & 0xFFF;
        assert_eq!((hi_imm4 << 12) | hi_imm12, 0x1234);
    }

    #[test]
    fn mov_16bit_only_no_movt() {
        // MOV R0, 0xFFFF → MOVW only (hi16 = 0, no MOVT needed)
        let code = assemble("mov r0, 0xFFFF", Arch::Arm).unwrap();
        assert_eq!(code.len(), 4, "16-bit value needs only MOVW, no MOVT");
    }

    #[test]
    fn mov_rotatable_still_uses_dp() {
        // MOV R0, 0xFF00 → 0xFF rotated right by 24 (rot=12) → DP form
        let code = assemble("mov r0, 0xFF00", Arch::Arm).unwrap();
        assert_eq!(code.len(), 4, "rotatable immediate should use DP form");
        let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
        // DP immediate: bit 25 = 1
        assert_eq!((word >> 25) & 0b111, 0b001, "should be DP immediate form");
    }

    #[test]
    fn mov_0x1234_different_regs() {
        // Verify MOVW fallback encodes different Rd correctly
        for (src, expected_rd) in &[
            ("mov r0, 0x1234", 0u32),
            ("mov r5, 0x1234", 5),
            ("mov r12, 0x1234", 12),
        ] {
            let code = assemble(src, Arch::Arm).unwrap();
            let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
            assert_eq!(
                (word >> 12) & 0xF,
                *expected_rd,
                "Rd mismatch for '{}'",
                src
            );
        }
    }

    #[test]
    fn mov_conditional_movw_fallback() {
        // MOVNE R0, 0x1234 → conditional MOVW
        let code = assemble("movne r0, 0x1234", Arch::Arm).unwrap();
        let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
        // NE condition = 0x1
        assert_eq!((word >> 28) & 0xF, 0x1, "condition should be NE (0x1)");
    }

    #[test]
    fn mvn_non_encodable_uses_movw_complement() {
        // MVN R0, 0x1234 → equivalent to MOV R0, ~0x1234 = MOV R0, 0xFFFFEDCB
        // The hi16 = 0xFFFF, lo16 = 0xEDCB → MOVW + MOVT
        let code = assemble("mvn r0, 0x1234", Arch::Arm).unwrap();
        assert_eq!(
            code.len(),
            8,
            "MVN fallback should emit MOVW+MOVT for complement"
        );

        let w0 = u32::from_le_bytes(code[0..4].try_into().unwrap());
        let lo_imm4 = (w0 >> 16) & 0xF;
        let lo_imm12 = w0 & 0xFFF;
        let lo = (lo_imm4 << 12) | lo_imm12;
        assert_eq!(lo, 0xEDCB, "lo16 of ~0x1234");

        let w1 = u32::from_le_bytes(code[4..8].try_into().unwrap());
        let hi_imm4 = (w1 >> 16) & 0xF;
        let hi_imm12 = w1 & 0xFFF;
        let hi = (hi_imm4 << 12) | hi_imm12;
        assert_eq!(hi, 0xFFFF, "hi16 of ~0x1234");
    }

    #[test]
    fn add_non_encodable_still_errors() {
        // ADD R0, R0, 0x1234 — not MOV, so no MOVW fallback → error
        let err = assemble("add r0, r0, 0x1234", Arch::Arm).unwrap_err();
        assert!(matches!(
            err,
            crate::error::AsmError::InvalidOperands { .. }
        ));
    }

    // ── Property tests for modified immediates ──────────────────

    #[test]
    fn all_encodable_modified_immediates_roundtrip() {
        // Enumerate all valid ARM modified immediates (256 values × 16 rotations = up to 4096)
        // and verify each produces a 4-byte DP MOV encoding.
        let mut seen = std::collections::HashSet::new();
        for rot in 0..16u32 {
            for imm8 in 0..=0xFFu32 {
                let val = imm8.rotate_right(rot * 2);
                if seen.insert(val) {
                    let src = alloc::format!("mov r0, {}", val);
                    let code = assemble(&src, Arch::Arm)
                        .unwrap_or_else(|e| panic!("failed to assemble '{}': {:?}", src, e));
                    assert_eq!(
                        code.len(),
                        4,
                        "encodable modified immediate {} should use DP form (4 bytes)",
                        val
                    );
                    // Verify the encoded immediate matches
                    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
                    // DP immediate form: bit 25 set
                    assert_eq!(
                        (word >> 25) & 0b111,
                        0b001,
                        "bit 25 should be set for DP immediate form, val={}",
                        val
                    );
                }
            }
        }
        // Should have found multiple unique values
        assert!(
            seen.len() > 100,
            "should find >100 unique encodable values, found {}",
            seen.len()
        );
    }

    #[test]
    fn all_u32_mov_succeeds() {
        // With MOVW/MOVT fallback, MOV Rd, #imm should work for ALL u32 values.
        // Test representative values across the full u32 range.
        let test_values: Vec<u32> = vec![
            0, 1, 0xFF, 0x100, 0x1234, 0xFFFF, 0x10000, 0x12345678, 0xDEADBEEF, 0xFFFFFFFF,
            0x80000000, 0x7FFFFFFF, 0xCAFEBABE, 0x00FF00FF, 0xFF00FF00,
        ];
        for val in &test_values {
            let src = alloc::format!("mov r0, {}", val);
            let code = assemble(&src, Arch::Arm)
                .unwrap_or_else(|e| panic!("MOV R0, {} should succeed: {:?}", val, e));
            // Should be either 4 bytes (DP/MOVW) or 8 bytes (MOVW+MOVT)
            assert!(
                code.len() == 4 || code.len() == 8,
                "MOV R0, {}: expected 4 or 8 bytes, got {}",
                val,
                code.len()
            );
        }
    }

    #[test]
    fn non_encodable_dp_errors_for_non_mov() {
        // For non-MOV data processing instructions, non-encodable immediates should error.
        // 0x1234 is not a valid modified immediate.
        for mnemonic in &["add", "sub", "and", "orr", "eor", "bic"] {
            let src = alloc::format!("{} r0, r0, 0x1234", mnemonic);
            let result = assemble(&src, Arch::Arm);
            assert!(
                result.is_err(),
                "{} with non-encodable immediate should error",
                mnemonic
            );
        }
    }

    // === Thumb (T16/T32) Encoding Tests ===

    /// Helper: assemble one Thumb instruction and return its bytes as u16 (LE).
    fn thumb16(src: &str) -> u16 {
        let bytes = assemble(src, Arch::Thumb).unwrap();
        assert_eq!(
            bytes.len(),
            2,
            "expected 2-byte Thumb, got {} bytes for: {}",
            bytes.len(),
            src
        );
        u16::from_le_bytes([bytes[0], bytes[1]])
    }

    /// Helper: assemble one Thumb-2 (32-bit) instruction and return 4 bytes.
    fn thumb32(src: &str) -> (u16, u16) {
        let bytes = assemble(src, Arch::Thumb).unwrap();
        assert_eq!(
            bytes.len(),
            4,
            "expected 4-byte Thumb-2, got {} bytes for: {}",
            bytes.len(),
            src
        );
        let hw1 = u16::from_le_bytes([bytes[0], bytes[1]]);
        let hw2 = u16::from_le_bytes([bytes[2], bytes[3]]);
        (hw1, hw2)
    }

    #[test]
    fn thumb_nop() {
        assert_eq!(thumb16("nop"), 0xBF00);
    }

    #[test]
    fn thumb_bkpt() {
        assert_eq!(thumb16("bkpt 0"), 0xBE00);
        assert_eq!(thumb16("bkpt 255"), 0xBEFF);
    }

    #[test]
    fn thumb_svc() {
        assert_eq!(thumb16("svc 0"), 0xDF00);
        assert_eq!(thumb16("svc 1"), 0xDF01);
    }

    #[test]
    fn thumb_bx_lr() {
        // BX LR = 0x4770
        assert_eq!(thumb16("bx lr"), 0x4770);
    }

    #[test]
    fn thumb_bx_r0() {
        assert_eq!(thumb16("bx r0"), 0x4700);
    }

    #[test]
    fn thumb_blx_r0() {
        assert_eq!(thumb16("blx r0"), 0x4780);
    }

    #[test]
    fn thumb_mov_imm8() {
        // MOVS R0, #0 = 0x2000
        assert_eq!(thumb16("mov r0, 0"), 0x2000);
        // MOVS R0, #42 = 0x202A
        assert_eq!(thumb16("mov r0, 42"), 0x202A);
        // MOVS R7, #255 = 0x27FF
        assert_eq!(thumb16("mov r7, 255"), 0x27FF);
    }

    #[test]
    fn thumb_mov_reg_high() {
        // MOV R0, R8 = 0x4640 (high register move)
        assert_eq!(thumb16("mov r0, r8"), 0x4640);
    }

    #[test]
    fn thumb_mov_reg_low() {
        // MOVS R0, R1 (low regs, set flags) = LSL R0, R1, #0 = 0x0008
        assert_eq!(thumb16("movs r0, r1"), 0x0008);
    }

    #[test]
    fn thumb_add_imm3() {
        // ADDS R0, R1, #3 = 0x1CC8
        assert_eq!(thumb16("add r0, r1, 3"), 0x1CC8);
    }

    #[test]
    fn thumb_add_imm8() {
        // ADDS R0, #42 = 0x302A
        assert_eq!(thumb16("add r0, 42"), 0x302A);
    }

    #[test]
    fn thumb_add_reg3() {
        // ADDS R0, R1, R2 = 0x1888
        assert_eq!(thumb16("add r0, r1, r2"), 0x1888);
    }

    #[test]
    fn thumb_sub_imm3() {
        // SUBS R0, R1, #3 = 0x1EC8
        assert_eq!(thumb16("sub r0, r1, 3"), 0x1EC8);
    }

    #[test]
    fn thumb_sub_imm8() {
        // SUBS R0, #42 = 0x382A
        assert_eq!(thumb16("sub r0, 42"), 0x382A);
    }

    #[test]
    fn thumb_sub_reg3() {
        // SUBS R0, R1, R2 = 0x1A88
        assert_eq!(thumb16("sub r0, r1, r2"), 0x1A88);
    }

    #[test]
    fn thumb_cmp_imm8() {
        // CMP R0, #42 = 0x282A
        assert_eq!(thumb16("cmp r0, 42"), 0x282A);
    }

    #[test]
    fn thumb_cmp_reg_lo() {
        // CMP R0, R1 = 0x4288
        assert_eq!(thumb16("cmp r0, r1"), 0x4288);
    }

    #[test]
    fn thumb_cmp_reg_hi() {
        // CMP R0, R8 = 0x4540
        assert_eq!(thumb16("cmp r0, r8"), 0x4540);
    }

    #[test]
    fn thumb_and_reg() {
        // ANDS R0, R1 = 0x4008
        assert_eq!(thumb16("and r0, r1"), 0x4008);
    }

    #[test]
    fn thumb_orr_reg() {
        // ORRS R0, R1 = 0x4308
        assert_eq!(thumb16("orr r0, r1"), 0x4308);
    }

    #[test]
    fn thumb_eor_reg() {
        // EORS R0, R1 = 0x4048
        assert_eq!(thumb16("eor r0, r1"), 0x4048);
    }

    #[test]
    fn thumb_bic_reg() {
        // BICS R0, R1 = 0x4388
        assert_eq!(thumb16("bic r0, r1"), 0x4388);
    }

    #[test]
    fn thumb_mvn_reg() {
        // MVNS R0, R1 = 0x43C8
        assert_eq!(thumb16("mvn r0, r1"), 0x43C8);
    }

    #[test]
    fn thumb_mul_reg() {
        // MULS R0, R1 = 0x4348
        assert_eq!(thumb16("mul r0, r1"), 0x4348);
    }

    #[test]
    fn thumb_tst_reg() {
        // TST R0, R1 = 0x4208
        assert_eq!(thumb16("tst r0, r1"), 0x4208);
    }

    #[test]
    fn thumb_lsl_imm5() {
        // LSLS R0, R1, #3 = 0x00C8
        assert_eq!(thumb16("lsl r0, r1, 3"), 0x00C8);
    }

    #[test]
    fn thumb_lsr_imm5() {
        // LSRS R0, R1, #3 = 0x08C8
        assert_eq!(thumb16("lsr r0, r1, 3"), 0x08C8);
    }

    #[test]
    fn thumb_asr_imm5() {
        // ASRS R0, R1, #3 = 0x10C8
        assert_eq!(thumb16("asr r0, r1, 3"), 0x10C8);
    }

    #[test]
    fn thumb_ldr_imm5() {
        // LDR R0, [R1, #0] = 0x6808
        assert_eq!(thumb16("ldr r0, [r1, 0]"), 0x6808);
        // LDR R0, [R1, #4] = 0x6848
        assert_eq!(thumb16("ldr r0, [r1, 4]"), 0x6848);
    }

    #[test]
    fn thumb_str_imm5() {
        // STR R0, [R1, #0] = 0x6008
        assert_eq!(thumb16("str r0, [r1, 0]"), 0x6008);
    }

    #[test]
    fn thumb_ldrb_imm5() {
        // LDRB R0, [R1, #0] = 0x7808
        assert_eq!(thumb16("ldrb r0, [r1, 0]"), 0x7808);
    }

    #[test]
    fn thumb_strb_imm5() {
        // STRB R0, [R1, #0] = 0x7008
        assert_eq!(thumb16("strb r0, [r1, 0]"), 0x7008);
    }

    #[test]
    fn thumb_ldrh_imm5() {
        // LDRH R0, [R1, #0] = 0x8808
        assert_eq!(thumb16("ldrh r0, [r1, 0]"), 0x8808);
    }

    #[test]
    fn thumb_ldr_sp_rel() {
        // LDR R0, [SP, #0] = 0x9800
        assert_eq!(thumb16("ldr r0, [sp, 0]"), 0x9800);
        // LDR R0, [SP, #4] = 0x9801
        assert_eq!(thumb16("ldr r0, [sp, 4]"), 0x9801);
    }

    #[test]
    fn thumb_str_sp_rel() {
        // STR R0, [SP, #0] = 0x9000
        assert_eq!(thumb16("str r0, [sp, 0]"), 0x9000);
    }

    #[test]
    fn thumb_push() {
        // PUSH {R0, R1, LR} = 0xB403 | bit8(LR) = 0xB503
        assert_eq!(thumb16("push {r0, r1, lr}"), 0xB503);
        // PUSH {R0} = 0xB401
        assert_eq!(thumb16("push {r0}"), 0xB401);
    }

    #[test]
    fn thumb_pop() {
        // POP {R0, R1, PC} = 0xBC03 | bit8(PC) = 0xBD03
        assert_eq!(thumb16("pop {r0, r1, pc}"), 0xBD03);
        // POP {R0} = 0xBC01
        assert_eq!(thumb16("pop {r0}"), 0xBC01);
    }

    #[test]
    fn thumb_it_eq() {
        // IT EQ = 0xBF08
        let bytes = assemble("it eq", Arch::Thumb).unwrap();
        assert_eq!(bytes.len(), 2);
        let hw = u16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(hw, 0xBF08);
    }

    #[test]
    fn thumb_ite_ne() {
        // ITE NE: firstcond=NE(1), suffix="e"
        // fc_bit0 = 1, e → bit = 1^1=0, mask = 0<<3 | 0b0100 = 0b0100 = 4
        // hw = 0xBF00 | (1 << 4) | 4 = 0xBF14
        let bytes = assemble("ite ne", Arch::Thumb).unwrap();
        assert_eq!(bytes.len(), 2);
        let hw = u16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(hw, 0xBF14);
    }

    #[test]
    fn thumb_add_high_reg() {
        // ADD R0, R8 (high register) = 0x4440
        assert_eq!(thumb16("add r0, r8"), 0x4440);
    }

    #[test]
    fn thumb_neg_reg() {
        // NEG R0, R1 = RSB R0, R1, #0 → 0x4248
        assert_eq!(thumb16("neg r0, r1"), 0x4248);
    }

    #[test]
    fn thumb_adc_reg() {
        // ADCS R0, R1 = 0x4148
        assert_eq!(thumb16("adc r0, r1"), 0x4148);
    }

    #[test]
    fn thumb_sbc_reg() {
        // SBCS R0, R1 = 0x4188
        assert_eq!(thumb16("sbc r0, r1"), 0x4188);
    }

    #[test]
    fn thumb_ror_reg() {
        // RORS R0, R1 = 0x41C8
        assert_eq!(thumb16("ror r0, r1"), 0x41C8);
    }

    #[test]
    fn thumb_cmn_reg() {
        // CMN R0, R1 = 0x42C8
        assert_eq!(thumb16("cmn r0, r1"), 0x42C8);
    }

    #[test]
    fn thumb_ldr_reg_offset() {
        // LDR R0, [R1, R2] = 0x5888
        assert_eq!(thumb16("ldr r0, [r1, r2]"), 0x5888);
    }

    #[test]
    fn thumb_str_reg_offset() {
        // STR R0, [R1, R2] = 0x5088
        assert_eq!(thumb16("str r0, [r1, r2]"), 0x5088);
    }

    #[test]
    fn thumb_bl_label() {
        // BL to forward label — should produce 4-byte Thumb-2 BL
        let bytes = assemble("bl target\ntarget:", Arch::Thumb).unwrap();
        assert_eq!(bytes.len(), 4, "BL should be 4 bytes");
    }

    #[test]
    fn thumb_b_label_short() {
        // B to nearby label — relaxation should keep it narrow (2 bytes)
        let bytes = assemble("b target\ntarget:", Arch::Thumb).unwrap();
        assert_eq!(
            bytes.len(),
            2,
            "B to nearby label should use narrow encoding"
        );
    }

    #[test]
    fn thumb_beq_label_short() {
        // BEQ to nearby label — should stay narrow (2 bytes)
        let bytes = assemble("beq target\ntarget:", Arch::Thumb).unwrap();
        assert_eq!(
            bytes.len(),
            2,
            "BEQ to nearby label should use narrow encoding"
        );
    }

    #[test]
    fn thumb_high_reg_not_allowed_in_narrow_dp() {
        // Trying to use R8 in a narrow-only ALU op should error
        let result = assemble("and r0, r8", Arch::Thumb);
        assert!(result.is_err());
    }

    #[test]
    fn thumb_bkpt_range() {
        // BKPT with value > 255 should error
        let result = assemble("bkpt 256", Arch::Thumb);
        assert!(result.is_err());
    }

    #[test]
    fn thumb_ldrb_reg_offset() {
        // LDRB R0, [R1, R2] = 0x5C88
        assert_eq!(thumb16("ldrb r0, [r1, r2]"), 0x5C88);
    }

    #[test]
    fn thumb_strb_reg_offset() {
        // STRB R0, [R1, R2] = 0x5488
        assert_eq!(thumb16("strb r0, [r1, r2]"), 0x5488);
    }

    #[test]
    fn thumb_ldrh_reg_offset() {
        // LDRH R0, [R1, R2] = 0x5A88
        assert_eq!(thumb16("ldrh r0, [r1, r2]"), 0x5A88);
    }

    #[test]
    fn thumb_strh_reg_offset() {
        // STRH R0, [R1, R2] = 0x5288
        assert_eq!(thumb16("strh r0, [r1, r2]"), 0x5288);
    }

    #[test]
    fn thumb_strh_imm5() {
        // STRH R0, [R1, #0] = 0x8008
        assert_eq!(thumb16("strh r0, [r1, 0]"), 0x8008);
    }

    #[test]
    fn thumb_lsl_reg() {
        // LSLS R0, R1 = 0x4088
        assert_eq!(thumb16("lsl r0, r1"), 0x4088);
    }

    #[test]
    fn thumb_lsr_reg() {
        // LSRS R0, R1 = 0x40C8
        assert_eq!(thumb16("lsr r0, r1"), 0x40C8);
    }

    #[test]
    fn thumb_asr_reg() {
        // ASRS R0, R1 = 0x4108
        assert_eq!(thumb16("asr r0, r1"), 0x4108);
    }

    #[test]
    fn thumb_bl_encoding() {
        // BL to label 4 bytes ahead: uses 32-bit Thumb-2 encoding
        let bytes = assemble("bl target\ntarget: nop", Arch::Thumb).unwrap();
        // BL is 4 bytes, NOP is 2 bytes = 6 total
        assert_eq!(bytes.len(), 6);
        let hw1 = u16::from_le_bytes([bytes[0], bytes[1]]);
        let hw2 = u16::from_le_bytes([bytes[2], bytes[3]]);
        // hw1: 11110 S imm10
        assert_eq!(hw1 >> 11, 0b11110);
        // hw2: 11x1 J1 1 J2 imm11
        assert!(hw2 & 0xD000 == 0xD000);
    }

    #[test]
    fn thumb_bl_self_ref() {
        // BL to self-referencing label
        let (hw1, hw2) = thumb32("here: bl here");
        // hw1: 11110 S imm10
        assert_eq!(hw1 >> 11, 0b11110);
        // hw2: 11x1 ...
        assert!(hw2 & 0xD000 == 0xD000);
    }
}
