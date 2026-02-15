//! AArch64 (ARM64) instruction encoder.
//!
//! Implements encoding for ARMv8-A AArch64 instructions — all 32-bit
//! fixed-width. Covers the core instruction set for offensive security
//! payloads: data processing, load/store, branches, moves, and system calls.
//!
//! ## AArch64 Instruction Encoding
//!
//! All A64 instructions are 32 bits. The top bits encode the instruction
//! class. Register fields are always 5 bits. The `sf` bit selects between
//! 32-bit (W) and 64-bit (X) register variants.
//!
//! ## Encoding Classes
//!
//! - **Data processing (immediate)**: ADD/SUB/AND/ORR with shifted imm12
//! - **Data processing (register)**: ADD/SUB/AND/ORR/EOR with shifted Rm
//! - **Move wide**: MOVZ/MOVN/MOVK with 16-bit immediate + shift
//! - **Branch**: B/BL (26-bit offset), BR/BLR/RET (register), B.cond
//! - **Load/Store**: LDR/STR with unsigned offset, pre/post-index
//! - **System**: SVC, BRK, NOP

use alloc::string::String;

use crate::encoder::{EncodedInstr, InstrBytes, RelaxInfo, RelocKind, Relocation};
use crate::error::AsmError;
use crate::ir::*;

// ── Helpers ──────────────────────────────────────────────────────────────

fn invalid_ops(mnemonic: &str, detail: &str, span: crate::error::Span) -> AsmError {
    AsmError::InvalidOperands {
        detail: alloc::format!("{}: {}", mnemonic, detail),
        span,
    }
}

fn get_a64_reg(
    op: &Operand,
    mnemonic: &str,
    span: crate::error::Span,
) -> Result<Register, AsmError> {
    match op {
        Operand::Register(r) if r.is_aarch64() => Ok(*r),
        _ => Err(invalid_ops(mnemonic, "expected AArch64 register", span)),
    }
}

fn get_imm(op: &Operand, mnemonic: &str, span: crate::error::Span) -> Result<i128, AsmError> {
    match op {
        Operand::Immediate(v) => Ok(*v),
        _ => Err(invalid_ops(mnemonic, "expected immediate", span)),
    }
}

#[inline]
fn emit32(buf: &mut InstrBytes, word: u32) {
    buf.extend_from_slice(&word.to_le_bytes());
}

/// `sf` bit: 1 for 64-bit (X registers), 0 for 32-bit (W registers).
fn sf(reg: Register) -> u32 {
    if reg.is_a64_64bit() {
        1
    } else {
        0
    }
}

// ── Condition codes ──────────────────────────────────────────────────────

fn cond_code(name: &str) -> Option<u32> {
    match name {
        "eq" => Some(0x0),
        "ne" => Some(0x1),
        "cs" | "hs" => Some(0x2),
        "cc" | "lo" => Some(0x3),
        "mi" => Some(0x4),
        "pl" => Some(0x5),
        "vs" => Some(0x6),
        "vc" => Some(0x7),
        "hi" => Some(0x8),
        "ls" => Some(0x9),
        "ge" => Some(0xA),
        "lt" => Some(0xB),
        "gt" => Some(0xC),
        "le" => Some(0xD),
        "al" => Some(0xE),
        "nv" => Some(0xF),
        _ => None,
    }
}

// ── NEON / AdvSIMD helpers ───────────────────────────────────────────────

/// Extract a vector register with arrangement from an operand.
fn get_a64_vreg(
    op: &Operand,
    mnemonic: &str,
    span: crate::error::Span,
) -> Result<(Register, VectorArrangement), AsmError> {
    match op {
        Operand::VectorRegister(r, arr) if r.is_a64_vector() => Ok((*r, *arr)),
        _ => Err(invalid_ops(
            mnemonic,
            "expected vector register with arrangement (e.g. v0.4s)",
            span,
        )),
    }
}

/// Q bit: 1 for 128-bit arrangements, 0 for 64-bit.
#[inline]
fn neon_q(arr: VectorArrangement) -> u32 {
    if arr.total_bits() == 128 {
        1
    } else {
        0
    }
}

/// AdvSIMD "size" field from arrangement element size:
/// 8→0b00, 16→0b01, 32→0b10, 64→0b11.
#[inline]
fn neon_size(arr: VectorArrangement) -> u32 {
    match arr.element_bits() {
        8 => 0b00,
        16 => 0b01,
        32 => 0b10,
        64 => 0b11,
        _ => 0b00,
    }
}

/// Encode an AdvSIMD three-same instruction.
///
/// Format: `0|Q|U|01110|size|1|Rm|opcode(5)|1|Rn|Rd`
#[inline]
fn neon_3same(q: u32, u: u32, size: u32, rm: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
    (q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (size << 22)
        | (1 << 21)
        | (rm << 16)
        | (opcode << 11)
        | (1 << 10)
        | (rn << 5)
        | rd
}

/// Encode an AdvSIMD two-register miscellaneous instruction.
///
/// Format: `0|Q|U|01110|size|10000|opcode(5)|10|Rn|Rd`
#[inline]
fn neon_2misc(q: u32, u: u32, size: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
    (q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (size << 22)
        | (0b10000 << 17)
        | (opcode << 12)
        | (0b10 << 10)
        | (rn << 5)
        | rd
}

// AdvSIMD copy encoding reference:
// DUP (element): 0|Q|0|01110000|imm5|0|0000|1|Rn|Rd
// DUP (general): 0|Q|0|01110000|imm5|0|0011|1|Rn|Rd
// INS (general): 0|1|0|01110000|imm5|0|0111|1|Rn|Rd
// UMOV:          0|Q|1|01110000|imm5|0|0111|1|Rn|Rd
// SMOV:          0|Q|0|01110000|imm5|0|0101|1|Rn|Rd

/// Encode AdvSIMD load/store multiple structures (no offset).
///
/// Format: `0|Q|0|01100|L|0|00000|opcode(4)|size(2)|Rn(5)|Rt(5)`
#[inline]
fn neon_ld_st_multiple(q: u32, l: u32, opcode: u32, size: u32, rn: u32, rt: u32) -> u32 {
    // AdvSIMD load/store multiple structures (no offset):
    // [31]0 [30]Q [29:23]0011000 [22]L [21:16]000000 [15:12]opcode [11:10]size [9:5]Rn [4:0]Rt
    (q << 30) | (0b0011000 << 23) | (l << 22) | (opcode << 12) | (size << 10) | (rn << 5) | rt
}

/// Encode NEON 3-same vector data-processing instruction.
fn encode_neon_3same(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    span: crate::error::Span,
    u: u32,
    opcode: u32,
) -> Result<(), AsmError> {
    if ops.len() != 3 {
        return Err(invalid_ops(mnemonic, "expected 3 vector operands", span));
    }
    let (rd, arr_d) = get_a64_vreg(&ops[0], mnemonic, span)?;
    let (rn, arr_n) = get_a64_vreg(&ops[1], mnemonic, span)?;
    let (rm, arr_m) = get_a64_vreg(&ops[2], mnemonic, span)?;
    if arr_d != arr_n || arr_n != arr_m {
        return Err(invalid_ops(
            mnemonic,
            "all vector operands must have the same arrangement",
            span,
        ));
    }
    let q = neon_q(arr_d);
    let size = neon_size(arr_d);
    emit32(
        buf,
        neon_3same(
            q,
            u,
            size,
            rm.a64_reg_num() as u32,
            opcode,
            rn.a64_reg_num() as u32,
            rd.a64_reg_num() as u32,
        ),
    );
    Ok(())
}

// ── NEON / AdvSIMD dispatch ──────────────────────────────────────────────
//
// Called from encode_aarch64 BEFORE the main scalar match block when the
// first operand is a VectorRegister.  Returns Ok(true) if the instruction
// was handled, Ok(false) if it should fall through to scalar handling.
fn encode_neon_dispatch(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<bool, AsmError> {
    let span = instr.span;
    match mnemonic {
        // ── Three-same arithmetic ────────────────────────────
        "add" => {
            encode_neon_3same(buf, mnemonic, ops, span, 0, 0b10000)?;
            Ok(true)
        }
        "sub" => {
            encode_neon_3same(buf, mnemonic, ops, span, 1, 0b10000)?;
            Ok(true)
        }
        "mul" => {
            encode_neon_3same(buf, mnemonic, ops, span, 0, 0b10011)?;
            Ok(true)
        }

        // ── Three-same bitwise ───────────────────────────────
        // AND/ORR/EOR/BIC/ORN: size field encodes the specific operation
        "and" => {
            neon_3same_bitwise(buf, mnemonic, ops, span, 0, 0b00) // U=0 size=00
        }
        "orr" => {
            neon_3same_bitwise(buf, mnemonic, ops, span, 0, 0b10) // U=0 size=10
        }
        "eor" => {
            neon_3same_bitwise(buf, mnemonic, ops, span, 1, 0b00) // U=1 size=00
        }
        "bic" => {
            neon_3same_bitwise(buf, mnemonic, ops, span, 0, 0b01) // U=0 size=01
        }
        "orn" => {
            neon_3same_bitwise(buf, mnemonic, ops, span, 0, 0b11) // U=0 size=11
        }

        // ── Three-same compare ───────────────────────────────
        "cmeq" => {
            encode_neon_3same(buf, mnemonic, ops, span, 1, 0b10001)?;
            Ok(true)
        }
        "cmhi" => {
            encode_neon_3same(buf, mnemonic, ops, span, 1, 0b00110)?;
            Ok(true)
        }
        "cmhs" => {
            encode_neon_3same(buf, mnemonic, ops, span, 1, 0b00111)?;
            Ok(true)
        }
        "cmge" => {
            encode_neon_3same(buf, mnemonic, ops, span, 0, 0b00111)?;
            Ok(true)
        }
        "cmgt" => {
            encode_neon_3same(buf, mnemonic, ops, span, 0, 0b00110)?;
            Ok(true)
        }

        // ── Three-same misc ─────────────────────────────────
        "addp" => {
            encode_neon_3same(buf, mnemonic, ops, span, 0, 0b10111)?;
            Ok(true)
        }
        "smax" => {
            encode_neon_3same(buf, mnemonic, ops, span, 0, 0b01100)?;
            Ok(true)
        }
        "smin" => {
            encode_neon_3same(buf, mnemonic, ops, span, 0, 0b01101)?;
            Ok(true)
        }
        "umax" => {
            encode_neon_3same(buf, mnemonic, ops, span, 1, 0b01100)?;
            Ok(true)
        }
        "umin" => {
            encode_neon_3same(buf, mnemonic, ops, span, 1, 0b01101)?;
            Ok(true)
        }

        // ── Two-register misc ────────────────────────────────
        "neg" => encode_neon_2misc(buf, mnemonic, ops, span, 1, 0b01011),
        "abs" => encode_neon_2misc(buf, mnemonic, ops, span, 0, 0b01011),
        "not" => {
            // NOT is bitwise – always size=00
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected 2 vector operands", span));
            }
            let (rd, arr_d) = get_a64_vreg(&ops[0], mnemonic, span)?;
            let (rn, _) = get_a64_vreg(&ops[1], mnemonic, span)?;
            let q = neon_q(arr_d);
            emit32(
                buf,
                neon_2misc(
                    q,
                    1,
                    0b00,
                    0b00101,
                    rn.a64_reg_num() as u32,
                    rd.a64_reg_num() as u32,
                ),
            );
            Ok(true)
        }
        "cnt" => {
            // CNT is only valid for 8B/16B → size=00
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected 2 vector operands", span));
            }
            let (rd, arr_d) = get_a64_vreg(&ops[0], mnemonic, span)?;
            let (rn, _) = get_a64_vreg(&ops[1], mnemonic, span)?;
            let q = neon_q(arr_d);
            emit32(
                buf,
                neon_2misc(
                    q,
                    0,
                    0b00,
                    0b00101,
                    rn.a64_reg_num() as u32,
                    rd.a64_reg_num() as u32,
                ),
            );
            Ok(true)
        }

        // ── MOV (vector alias for ORR Vd, Vn, Vn) ───────────
        "mov" => {
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected 2 vector operands", span));
            }
            let (rd, arr_d) = get_a64_vreg(&ops[0], mnemonic, span)?;
            let (rn, _) = get_a64_vreg(&ops[1], mnemonic, span)?;
            let q = neon_q(arr_d);
            emit32(
                buf,
                neon_3same(
                    q,
                    0,
                    0b10,
                    rn.a64_reg_num() as u32,
                    0b00011,
                    rn.a64_reg_num() as u32,
                    rd.a64_reg_num() as u32,
                ),
            );
            Ok(true)
        }

        // ── DUP (general register → all lanes) ──────────────
        "dup" => {
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected 2 operands", span));
            }
            let (rd, arr) = get_a64_vreg(&ops[0], mnemonic, span)?;
            let rn = get_a64_reg(&ops[1], mnemonic, span)?;
            let q = neon_q(arr);
            let imm5: u32 = match arr.element_bits() {
                8 => 0b00001,
                16 => 0b00010,
                32 => 0b00100,
                64 => 0b01000,
                _ => return Err(invalid_ops(mnemonic, "invalid arrangement", span)),
            };
            let word = (q << 30)
                | (0b001110000 << 21)
                | (imm5 << 16)
                | (0b00011 << 11)
                | (1 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
            Ok(true)
        }

        // ── INS (general register → element lane 0) ─────────
        "ins" => {
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected 2 operands", span));
            }
            let (rd, arr) = get_a64_vreg(&ops[0], mnemonic, span)?;
            let rn = get_a64_reg(&ops[1], mnemonic, span)?;
            let imm5: u32 = match arr.element_bits() {
                8 => 0b00001,
                16 => 0b00010,
                32 => 0b00100,
                64 => 0b01000,
                _ => return Err(invalid_ops(mnemonic, "invalid arrangement", span)),
            };
            let word = (1u32 << 30)
                | (0b001110000 << 21)
                | (imm5 << 16)
                | (0b00111 << 11)
                | (1 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
            Ok(true)
        }

        // ── LD1 / ST1 (single structure, no offset) ─────────
        "ld1" => {
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected {Vt.T}, [Xn]", span));
            }
            let (rt, arr) = get_a64_vreg(&ops[0], mnemonic, span)?;
            let rn = match &ops[1] {
                Operand::Memory(m) => m.base.ok_or_else(|| {
                    invalid_ops(mnemonic, "missing base register in memory operand", span)
                })?,
                Operand::Register(r) => *r,
                _ => return Err(invalid_ops(mnemonic, "expected base register", span)),
            };
            let q = neon_q(arr);
            let size = neon_size(arr);
            emit32(
                buf,
                neon_ld_st_multiple(
                    q,
                    1,
                    0b0111,
                    size,
                    rn.a64_reg_num() as u32,
                    rt.a64_reg_num() as u32,
                ),
            );
            Ok(true)
        }
        "st1" => {
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected {Vt.T}, [Xn]", span));
            }
            let (rt, arr) = get_a64_vreg(&ops[0], mnemonic, span)?;
            let rn = match &ops[1] {
                Operand::Memory(m) => m.base.ok_or_else(|| {
                    invalid_ops(mnemonic, "missing base register in memory operand", span)
                })?,
                Operand::Register(r) => *r,
                _ => return Err(invalid_ops(mnemonic, "expected base register", span)),
            };
            let q = neon_q(arr);
            let size = neon_size(arr);
            emit32(
                buf,
                neon_ld_st_multiple(
                    q,
                    0,
                    0b0111,
                    size,
                    rn.a64_reg_num() as u32,
                    rt.a64_reg_num() as u32,
                ),
            );
            Ok(true)
        }

        // Not a NEON instruction – fall through to scalar match
        _ => Ok(false),
    }
}

/// Helper for NEON three-same bitwise ops (AND/ORR/EOR/BIC/ORN).
/// These use the `size` field to encode the specific operation rather than
/// element width.
fn neon_3same_bitwise(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    span: crate::error::Span,
    u: u32,
    size: u32,
) -> Result<bool, AsmError> {
    if ops.len() != 3 {
        return Err(invalid_ops(mnemonic, "expected 3 vector operands", span));
    }
    let (rd, arr_d) = get_a64_vreg(&ops[0], mnemonic, span)?;
    let (rn, _) = get_a64_vreg(&ops[1], mnemonic, span)?;
    let (rm, _) = get_a64_vreg(&ops[2], mnemonic, span)?;
    let q = neon_q(arr_d);
    emit32(
        buf,
        neon_3same(
            q,
            u,
            size,
            rm.a64_reg_num() as u32,
            0b00011,
            rn.a64_reg_num() as u32,
            rd.a64_reg_num() as u32,
        ),
    );
    Ok(true)
}

/// Helper for NEON two-register misc ops (NEG, ABS, etc.)
fn encode_neon_2misc(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    span: crate::error::Span,
    u: u32,
    opcode: u32,
) -> Result<bool, AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(mnemonic, "expected 2 vector operands", span));
    }
    let (rd, arr_d) = get_a64_vreg(&ops[0], mnemonic, span)?;
    let (rn, arr_n) = get_a64_vreg(&ops[1], mnemonic, span)?;
    if arr_d != arr_n {
        return Err(invalid_ops(mnemonic, "arrangement mismatch", span));
    }
    let q = neon_q(arr_d);
    let size = neon_size(arr_d);
    emit32(
        buf,
        neon_2misc(
            q,
            u,
            size,
            opcode,
            rn.a64_reg_num() as u32,
            rd.a64_reg_num() as u32,
        ),
    );
    Ok(true)
}

// ── Data Processing (Immediate) ──────────────────────────────────────────

/// ADD/SUB (immediate): sf|op|S|10001|shift|imm12|Rn|Rd
fn encode_addsub_imm(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 3 {
        // 2-operand form: CMP Xn, #imm → SUBS XZR, Xn, #imm
        if ops.len() == 2 && matches!(mnemonic, "cmp" | "cmn") {
            let rn = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let imm = get_imm(&ops[1], mnemonic, instr.span)? as u64;
            let sf_bit = sf(rn);
            let (op, s) = match mnemonic {
                "cmp" => (1u32, 1u32), // SUBS
                "cmn" => (0u32, 1u32), // ADDS
                _ => return Err(invalid_ops(mnemonic, "expected cmp or cmn", instr.span)),
            };
            if imm > 0xFFF {
                return Err(invalid_ops(
                    mnemonic,
                    "immediate must fit in 12 bits",
                    instr.span,
                ));
            }
            let word = (sf_bit << 31)
                | (op << 30)
                | (s << 29)
                | (0b10001 << 24)
                | ((imm as u32) << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | 0b11111; // Rd = XZR/WZR
            emit32(buf, word);
            return Ok(());
        }
        return Err(invalid_ops(
            mnemonic,
            "expected 3 operands (Rd, Rn, #imm)",
            instr.span,
        ));
    }

    let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
    let imm = get_imm(&ops[2], mnemonic, instr.span)? as u64;
    let sf_bit = sf(rd);

    let (op, s) = match mnemonic {
        "add" => (0u32, 0u32),
        "adds" => (0u32, 1u32),
        "sub" => (1u32, 0u32),
        "subs" => (1u32, 1u32),
        _ => return Err(invalid_ops(mnemonic, "unknown add/sub variant", instr.span)),
    };

    if imm > 0xFFF {
        return Err(invalid_ops(
            mnemonic,
            "immediate must fit in 12 bits",
            instr.span,
        ));
    }

    // sf|op|S|10001|sh(0)|imm12|Rn|Rd
    let word = (sf_bit << 31)
        | (op << 30)
        | (s << 29)
        | (0b10001 << 24)
        | ((imm as u32) << 10)
        | ((rn.a64_reg_num() as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── Data Processing (Register) ───────────────────────────────────────────

/// ADD/SUB (shifted register): sf|op|S|01011|shift|0|Rm|imm6|Rn|Rd
fn encode_addsub_reg(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() < 3 {
        return Err(invalid_ops(
            mnemonic,
            "expected 3 operands (Rd, Rn, Rm)",
            instr.span,
        ));
    }
    let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
    let rm = get_a64_reg(&ops[2], mnemonic, instr.span)?;
    let sf_bit = sf(rd);

    let (op, s) = match mnemonic {
        "add" => (0u32, 0u32),
        "adds" => (0u32, 1u32),
        "sub" => (1u32, 0u32),
        "subs" => (1u32, 1u32),
        "neg" => (1u32, 0u32), // NEG Xd, Xm → SUB Xd, XZR, Xm
        "negs" => (1u32, 1u32),
        _ => return Err(invalid_ops(mnemonic, "unknown add/sub variant", instr.span)),
    };

    // sf|op|S|01011|00|0|Rm|000000|Rn|Rd
    #[allow(clippy::identity_op)]
    let word = (sf_bit << 31)
        | (op << 30)
        | (s << 29)
        | (0b01011 << 24)
        | (0 << 22) // shift = LSL
        | (0 << 21)
        | ((rm.a64_reg_num() as u32) << 16)
        | (0 << 10) // imm6 = 0
        | ((rn.a64_reg_num() as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── Logical (Register) ──────────────────────────────────────────────────

fn encode_logical_reg(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    // TST Rn, Rm → ANDS XZR, Rn, Rm
    if matches!(mnemonic, "tst") {
        if ops.len() < 2 {
            return Err(invalid_ops(mnemonic, "expected 2 operands", instr.span));
        }
        let rn = get_a64_reg(&ops[0], mnemonic, instr.span)?;
        let rm = get_a64_reg(&ops[1], mnemonic, instr.span)?;
        let sf_bit = sf(rn);
        // ANDS: sf|11|01010|00|0|Rm|000000|Rn|11111
        let word = (sf_bit << 31)
            | (0b11 << 29)
            | (0b01010 << 24)
            | ((rm.a64_reg_num() as u32) << 16)
            | ((rn.a64_reg_num() as u32) << 5)
            | 0b11111;
        emit32(buf, word);
        return Ok(());
    }

    if ops.len() < 3 {
        return Err(invalid_ops(
            mnemonic,
            "expected 3 operands (Rd, Rn, Rm)",
            instr.span,
        ));
    }
    let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
    let rm = get_a64_reg(&ops[2], mnemonic, instr.span)?;
    let sf_bit = sf(rd);

    // opc: 00=AND, 01=ORR, 10=EOR, 11=ANDS
    let (opc, n) = match mnemonic {
        "and" => (0b00u32, 0u32),
        "bic" => (0b00, 1),
        "orr" => (0b01, 0),
        "orn" => (0b01, 1),
        "eor" => (0b10, 0),
        "eon" => (0b10, 1),
        "ands" => (0b11, 0),
        "bics" => (0b11, 1),
        _ => return Err(invalid_ops(mnemonic, "unknown logical op", instr.span)),
    };

    // sf|opc|01010|00|N|Rm|000000|Rn|Rd
    #[allow(clippy::identity_op)]
    let word = (sf_bit << 31)
        | (opc << 29)
        | (0b01010 << 24)
        | (0 << 22) // shift = LSL
        | (n << 21)
        | ((rm.a64_reg_num() as u32) << 16)
        | (0 << 10) // imm6 = 0
        | ((rn.a64_reg_num() as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── Move Wide ────────────────────────────────────────────────────────────

fn encode_movz_movn_movk(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() < 2 {
        return Err(invalid_ops(
            mnemonic,
            "expected Rd, #imm16[, LSL #shift]",
            instr.span,
        ));
    }
    let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let imm = get_imm(&ops[1], mnemonic, instr.span)? as u64;
    let sf_bit = sf(rd);

    // Optional shift: 3rd operand as immediate (shift amount: 0, 16, 32, 48)
    let hw = if ops.len() >= 3 {
        let shift = get_imm(&ops[2], mnemonic, instr.span)? as u64;
        match shift {
            0 => 0u32,
            16 => 1,
            32 => 2,
            48 => 3,
            _ => {
                return Err(invalid_ops(
                    mnemonic,
                    "shift must be 0, 16, 32, or 48",
                    instr.span,
                ))
            }
        }
    } else {
        0
    };

    if imm > 0xFFFF {
        return Err(invalid_ops(
            mnemonic,
            "immediate must fit in 16 bits",
            instr.span,
        ));
    }

    let opc = match mnemonic {
        "movn" => 0b00u32,
        "movz" => 0b10,
        "movk" => 0b11,
        _ => return Err(invalid_ops(mnemonic, "unknown move wide op", instr.span)),
    };

    // sf|opc|100101|hw|imm16|Rd
    let word = (sf_bit << 31)
        | (opc << 29)
        | (0b100101 << 23)
        | (hw << 21)
        | ((imm as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

/// MOV pseudo-instruction: MOV Xd, Xm → ORR Xd, XZR, Xm
///                         MOV Xd, #imm → MOVZ/MOVN
fn encode_mov(buf: &mut InstrBytes, ops: &[Operand], instr: &Instruction) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops("mov", "expected 2 operands", instr.span));
    }

    match (&ops[0], &ops[1]) {
        (Operand::Register(rd), Operand::Register(rm)) if rd.is_aarch64() && rm.is_aarch64() => {
            // MOV Xd, Xm → ORR Xd, XZR, Xm
            let sf_bit = sf(*rd);
            // sf|01|01010|00|0|Rm|000000|11111|Rd
            let word = (sf_bit << 31)
                | (0b01 << 29)
                | (0b01010 << 24)
                | ((rm.a64_reg_num() as u32) << 16)
                | (0b11111 << 5) // Rn = XZR
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        (Operand::Register(rd), Operand::Immediate(imm)) if rd.is_aarch64() => {
            let val = *imm as u64;
            let sf_bit = sf(*rd);

            // Try MOVZ with optimal shift
            for hw in 0u32..4 {
                if sf_bit == 0 && hw >= 2 {
                    break;
                }
                let shifted = val >> (hw * 16);
                let mask = if sf_bit == 1 { u64::MAX } else { 0xFFFF_FFFF };
                let reconstructed = (shifted & 0xFFFF) << (hw * 16);
                if reconstructed == (val & mask) {
                    // MOVZ
                    let word = (sf_bit << 31)
                        | (0b10 << 29)
                        | (0b100101 << 23)
                        | (hw << 21)
                        | (((shifted & 0xFFFF) as u32) << 5)
                        | (rd.a64_reg_num() as u32);
                    emit32(buf, word);
                    return Ok(());
                }
            }

            // Try MOVN (bitwise NOT)
            let inv = !val;
            for hw in 0u32..4 {
                if sf_bit == 0 && hw >= 2 {
                    break;
                }
                let shifted = inv >> (hw * 16);
                let mask = if sf_bit == 1 { u64::MAX } else { 0xFFFF_FFFF };
                let reconstructed = (shifted & 0xFFFF) << (hw * 16);
                if reconstructed == (inv & mask) {
                    // MOVN
                    #[allow(clippy::identity_op)]
                    let word = (sf_bit << 31)
                        | (0b00 << 29)
                        | (0b100101 << 23)
                        | (hw << 21)
                        | (((shifted & 0xFFFF) as u32) << 5)
                        | (rd.a64_reg_num() as u32);
                    emit32(buf, word);
                    return Ok(());
                }
            }

            return Err(invalid_ops(
                "mov",
                "immediate cannot be encoded in a single instruction; use movz+movk",
                instr.span,
            ));
        }
        // MOV to/from SP: ADD Xd, Xn, #0  (SP is special — can only appear in ADD/SUB forms)
        (Operand::Register(rd), Operand::Register(rn))
            if rd.is_aarch64()
                && (matches!(rd, Register::A64Sp) || matches!(rn, Register::A64Sp)) =>
        {
            let sf_bit = sf(*rd);
            // ADD: sf|0|0|10001|00|000000000000|Rn|Rd
            let word = (sf_bit << 31)
                | (0b0010001 << 24)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                "mov",
                "unsupported operand combination",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── Branch ───────────────────────────────────────────────────────────────

fn encode_branch(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_ops(mnemonic, "expected 1 operand", instr.span));
    }

    let is_link = mnemonic == "bl";

    match &ops[0] {
        Operand::Label(label) => {
            // B/BL label: op|imm26
            // B:  000101|imm26
            // BL: 100101|imm26
            let op = if is_link { 0b100101u32 } else { 0b000101u32 };
            let word = op << 26;
            let reloc_offset = buf.len();
            emit32(buf, word);
            *reloc = Some(Relocation {
                offset: reloc_offset,
                size: 4,
                label: alloc::rc::Rc::from(&**label),
                kind: RelocKind::Aarch64Jump26,
                addend: 0,
                trailing_bytes: 0,
            });
        }
        Operand::Immediate(imm) => {
            let offset = (*imm as i32) >> 2;
            let imm26 = (offset as u32) & 0x03FF_FFFF;
            let op = if is_link { 0b100101u32 } else { 0b000101u32 };
            let word = (op << 26) | imm26;
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "expected label or immediate",
                instr.span,
            ))
        }
    }
    Ok(())
}

fn encode_br_blr_ret(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    match mnemonic {
        "ret" => {
            // RET {Xn} — default X30
            let rn = if ops.is_empty() {
                Register::A64X30
            } else {
                get_a64_reg(&ops[0], "ret", instr.span)?
            };
            // 1101011|0|0|10|11111|0000|0|0|Rn|00000
            let word = 0xD65F_0000 | ((rn.a64_reg_num() as u32) << 5);
            emit32(buf, word);
        }
        "br" => {
            if ops.len() != 1 {
                return Err(invalid_ops("br", "expected 1 register", instr.span));
            }
            let rn = get_a64_reg(&ops[0], "br", instr.span)?;
            // 1101011|0|0|00|11111|0000|0|0|Rn|00000
            let word = 0xD61F_0000 | ((rn.a64_reg_num() as u32) << 5);
            emit32(buf, word);
        }
        "blr" => {
            if ops.len() != 1 {
                return Err(invalid_ops("blr", "expected 1 register", instr.span));
            }
            let rn = get_a64_reg(&ops[0], "blr", instr.span)?;
            // 1101011|0|0|01|11111|0000|0|0|Rn|00000
            let word = 0xD63F_0000 | ((rn.a64_reg_num() as u32) << 5);
            emit32(buf, word);
        }
        _ => return Err(invalid_ops(mnemonic, "unknown branch", instr.span)),
    }
    Ok(())
}

/// B.cond label — with relaxation for out-of-range targets.
///
/// Short form (4 bytes): `B.{cond} label` (±1 MB, 19-bit offset)
/// Long form (8 bytes): `B.{!cond} +8; B label` (±128 MB via unconditional B)
fn encode_bcond(
    buf: &mut InstrBytes,
    cond_name: &str,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
    relax: &mut Option<RelaxInfo>,
) -> Result<(), AsmError> {
    let cc = cond_code(cond_name).ok_or_else(|| {
        invalid_ops(
            &alloc::format!("b.{}", cond_name),
            "unknown condition",
            instr.span,
        )
    })?;

    if ops.len() != 1 {
        return Err(invalid_ops(
            &alloc::format!("b.{}", cond_name),
            "expected 1 operand",
            instr.span,
        ));
    }

    match &ops[0] {
        Operand::Label(label) => {
            // ── Long form: B.{!cond} +8; B label ──
            // Inverted condition skips over the unconditional B.
            let inv_cc = cc ^ 1; // AArch64 condition inversion: flip bit 0
                                 // B.{!cond} +8: imm19 = 2 (skip 2 words = 8 bytes)
            let skip_word = (0b01010100u32 << 24) | (2u32 << 5) | inv_cc;
            emit32(buf, skip_word);
            // B label: unconditional branch placeholder
            let b_word = 0b000101u32 << 26; // B with imm26 = 0
            let reloc_offset = buf.len();
            emit32(buf, b_word);

            *reloc = Some(Relocation {
                offset: reloc_offset,
                size: 4,
                label: alloc::rc::Rc::from(&**label),
                kind: RelocKind::Aarch64Jump26,
                addend: 0,
                trailing_bytes: 0,
            });

            // ── Short form: B.{cond} label ──
            let mut short = InstrBytes::new();
            let short_word = (0b01010100u32 << 24) | cc;
            emit32(&mut short, short_word);

            *relax = Some(RelaxInfo {
                short_bytes: short,
                short_reloc_offset: 0,
                short_relocation: Some(Relocation {
                    offset: 0,
                    size: 4,
                    label: alloc::rc::Rc::from(&**label),
                    kind: RelocKind::Aarch64Branch19,
                    addend: 0,
                    trailing_bytes: 0,
                }),
            });
        }
        Operand::Immediate(imm) => {
            let offset = ((*imm as i32) >> 2) & 0x7FFFF;
            let word = (0b01010100 << 24) | ((offset as u32) << 5) | cc;
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                &alloc::format!("b.{}", cond_name),
                "expected label or immediate",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// CBZ/CBNZ Rt, label — with relaxation for out-of-range targets.
///
/// Short form (4 bytes): `CBZ/CBNZ Rt, label` (±1 MB, 19-bit offset)
/// Long form (8 bytes): `CBNZ/CBZ Rt, +8; B label` (±128 MB via B)
fn encode_cbz(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
    relax: &mut Option<RelaxInfo>,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(mnemonic, "expected Rt, label", instr.span));
    }
    let rt = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let sf_bit = sf(rt);
    let op = if mnemonic == "cbnz" { 1u32 } else { 0u32 };

    match &ops[1] {
        Operand::Label(label) => {
            // ── Long form: {inverted} Rt, +8; B label ──
            let inv_op = op ^ 1; // CBZ↔CBNZ inversion
                                 // Inverted CBZ/CBNZ Rt, +8: imm19 = 2 (skip 8 bytes)
            let skip_word = (sf_bit << 31)
                | (0b011010 << 25)
                | (inv_op << 24)
                | (2u32 << 5)
                | (rt.a64_reg_num() as u32);
            emit32(buf, skip_word);
            // B label: unconditional branch placeholder
            let b_word = 0b000101u32 << 26;
            let reloc_offset = buf.len();
            emit32(buf, b_word);

            *reloc = Some(Relocation {
                offset: reloc_offset,
                size: 4,
                label: alloc::rc::Rc::from(&**label),
                kind: RelocKind::Aarch64Jump26,
                addend: 0,
                trailing_bytes: 0,
            });

            // ── Short form: original CBZ/CBNZ Rt, label ──
            let mut short = InstrBytes::new();
            let short_word =
                (sf_bit << 31) | (0b011010 << 25) | (op << 24) | (rt.a64_reg_num() as u32);
            emit32(&mut short, short_word);

            *relax = Some(RelaxInfo {
                short_bytes: short,
                short_reloc_offset: 0,
                short_relocation: Some(Relocation {
                    offset: 0,
                    size: 4,
                    label: alloc::rc::Rc::from(&**label),
                    kind: RelocKind::Aarch64Branch19,
                    addend: 0,
                    trailing_bytes: 0,
                }),
            });
        }
        Operand::Immediate(imm) => {
            let off19 = ((*imm as i32) >> 2) & 0x7FFFF;
            let word = (sf_bit << 31)
                | (0b011010 << 25)
                | (op << 24)
                | ((off19 as u32) << 5)
                | (rt.a64_reg_num() as u32);
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "expected label or immediate",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── Load / Store ─────────────────────────────────────────────────────────

/// LDR/STR (unsigned offset): size|11|1|00|01|opc|imm12|Rn|Rt
fn encode_ldr_str(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() < 2 {
        return Err(invalid_ops(
            mnemonic,
            "expected Rt, [Xn, #imm] or Rt, label",
            instr.span,
        ));
    }
    let rt = get_a64_reg(&ops[0], mnemonic, instr.span)?;

    // Determine size and opc
    let (size_bits, is_load) = match mnemonic {
        "ldr" => (if rt.is_a64_64bit() { 3u32 } else { 2u32 }, true),
        "str" => (if rt.is_a64_64bit() { 3u32 } else { 2u32 }, false),
        "ldrb" => (0, true),
        "strb" => (0, false),
        "ldrh" => (1, true),
        "strh" => (1, false),
        "ldrsb" => (0, true), // sign-extend handled separately
        "ldrsh" => (1, true),
        "ldrsw" => (2, true),
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "unknown load/store variant",
                instr.span,
            ))
        }
    };

    match &ops[1] {
        Operand::Memory(mem) => {
            let rn = match mem.base {
                Some(r) if r.is_aarch64() => r,
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "memory base must be AArch64 register",
                        instr.span,
                    ))
                }
            };

            // Register offset: LDR Rt, [Xn, Xm] or [Xn, Xm, LSL #n]
            // Encoding: size|111|V=0|00|opc|1|Rm|option|S|10|Rn|Rt
            if let Some(rm) = mem.index {
                if !rm.is_aarch64() {
                    return Err(invalid_ops(
                        mnemonic,
                        "index register must be AArch64 register",
                        instr.span,
                    ));
                }
                let opc = if is_load { 0b01u32 } else { 0b00u32 };
                let opc = match mnemonic {
                    "ldrsb" => {
                        if rt.is_a64_64bit() {
                            0b10
                        } else {
                            0b11
                        }
                    }
                    "ldrsh" => {
                        if rt.is_a64_64bit() {
                            0b10
                        } else {
                            0b11
                        }
                    }
                    "ldrsw" => 0b10,
                    _ => opc,
                };
                let option = if rm.is_a64_64bit() {
                    0b011u32
                } else {
                    0b010u32
                }; // LSL/UXTW
                   // S bit: scale > 1 implies shift by log2(size)
                let s = if mem.scale > 1 { 1u32 } else { 0u32 };
                let word = (size_bits << 30)
                    | (0b111000 << 24)
                    | (opc << 22)
                    | (1u32 << 21)
                    | ((rm.a64_reg_num() as u32) << 16)
                    | (option << 13)
                    | (s << 12)
                    | (0b10u32 << 10)
                    | ((rn.a64_reg_num() as u32) << 5)
                    | (rt.a64_reg_num() as u32);
                emit32(buf, word);
                return Ok(());
            }

            let offset = mem.disp;
            let scale = 1u32 << size_bits;

            if offset >= 0 && (offset as u64) % (scale as u64) == 0 {
                let scaled = (offset as u64) / (scale as u64);
                if scaled <= 0xFFF {
                    // Unsigned offset: size|11|1|00|01|opc|imm12|Rn|Rt
                    let opc = if is_load { 0b01u32 } else { 0b00u32 };
                    // Sign-extending loads use different opc
                    let opc = match mnemonic {
                        "ldrsb" => {
                            if rt.is_a64_64bit() {
                                0b10
                            } else {
                                0b11
                            }
                        }
                        "ldrsh" => {
                            if rt.is_a64_64bit() {
                                0b10
                            } else {
                                0b11
                            }
                        }
                        "ldrsw" => 0b10,
                        _ => opc,
                    };
                    let word = (size_bits << 30)
                        | (0b111001 << 24)
                        | (opc << 22)
                        | ((scaled as u32) << 10)
                        | ((rn.a64_reg_num() as u32) << 5)
                        | (rt.a64_reg_num() as u32);
                    emit32(buf, word);
                    return Ok(());
                }
            }

            // Unscaled offset: size|11|1|00|00|opc|imm9|00|Rn|Rt (LDUR/STUR)
            if (-256..=255).contains(&offset) {
                let opc = if is_load { 0b01u32 } else { 0b00u32 };
                let opc = match mnemonic {
                    "ldrsb" => {
                        if rt.is_a64_64bit() {
                            0b10
                        } else {
                            0b11
                        }
                    }
                    "ldrsh" => {
                        if rt.is_a64_64bit() {
                            0b10
                        } else {
                            0b11
                        }
                    }
                    "ldrsw" => 0b10,
                    _ => opc,
                };
                let imm9 = (offset as u32) & 0x1FF;
                #[allow(clippy::identity_op)]
                let word = (size_bits << 30)
                    | (0b111000 << 24)
                    | (opc << 22)
                    | (0 << 21)
                    | (imm9 << 12)
                    | (0b00 << 10) // unscaled
                    | ((rn.a64_reg_num() as u32) << 5)
                    | (rt.a64_reg_num() as u32);
                emit32(buf, word);
                return Ok(());
            }

            return Err(invalid_ops(mnemonic, "offset out of range", instr.span));
        }
        Operand::Label(label) => {
            // LDR Rt, label → LDR (literal): opc|01|1|0|00|imm19|Rt
            let opc = if rt.is_a64_64bit() { 0b01u32 } else { 0b00u32 };
            let word = (opc << 30) | (0b011000 << 24) | (rt.a64_reg_num() as u32);
            let reloc_offset = buf.len();
            emit32(buf, word);
            *reloc = Some(Relocation {
                offset: reloc_offset,
                size: 4,
                label: alloc::rc::Rc::from(&**label),
                kind: RelocKind::Aarch64LdrLit19,
                addend: 0,
                trailing_bytes: 0,
            });
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "expected memory or label operand",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// STP/LDP (store/load pair): opc|10|1|0|0|L|imm7|Rt2|Rn|Rt
fn encode_stp_ldp(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() < 3 {
        return Err(invalid_ops(
            mnemonic,
            "expected Rt, Rt2, [Xn, #imm]",
            instr.span,
        ));
    }
    let rt = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let rt2 = get_a64_reg(&ops[1], mnemonic, instr.span)?;
    let is_load = matches!(mnemonic, "ldp");
    let opc = if rt.is_a64_64bit() { 0b10u32 } else { 0b00u32 };
    let scale = if rt.is_a64_64bit() { 8i64 } else { 4i64 };

    match &ops[2] {
        Operand::Memory(mem) => {
            let rn = match mem.base {
                Some(r) if r.is_aarch64() => r,
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "memory base must be AArch64 register",
                        instr.span,
                    ))
                }
            };
            let offset = mem.disp;
            if offset % scale != 0 {
                return Err(invalid_ops(
                    mnemonic,
                    "offset must be aligned to register size",
                    instr.span,
                ));
            }
            let imm7 = ((offset / scale) as u32) & 0x7F;

            // Signed offset form: opc|10|1|0|0|1|0|imm7|Rt2|Rn|Rt
            let l = is_load as u32;
            #[allow(clippy::identity_op)]
            let word = (opc << 30)
                | (0b101 << 27)
                | (0 << 26) // pre/post
                | (0b01 << 24) // signed offset
                | (0 << 23)
                | (l << 22)
                | (imm7 << 15)
                | ((rt2.a64_reg_num() as u32) << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rt.a64_reg_num() as u32);
            emit32(buf, word);
        }
        _ => return Err(invalid_ops(mnemonic, "expected memory operand", instr.span)),
    }
    Ok(())
}

// ── Shift instructions (aliases) ─────────────────────────────────────────

fn encode_shift(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 3 {
        return Err(invalid_ops(
            mnemonic,
            "expected Rd, Rn, Rm or Rd, Rn, #imm",
            instr.span,
        ));
    }
    let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
    let sf_bit = sf(rd);

    match &ops[2] {
        Operand::Register(rm) if rm.is_aarch64() => {
            // Variable shift: sf|0|0|11010110|Rm|0010|op2|Rn|Rd
            let op2 = match mnemonic {
                "lsl" | "lslv" => 0b00u32,
                "lsr" | "lsrv" => 0b01,
                "asr" | "asrv" => 0b10,
                "ror" | "rorv" => 0b11,
                _ => return Err(invalid_ops(mnemonic, "unknown shift type", instr.span)),
            };
            let word = (sf_bit << 31)
                | (0b0011010110 << 21)
                | ((rm.a64_reg_num() as u32) << 16)
                | (0b0010 << 12)
                | (op2 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        Operand::Immediate(imm) => {
            // Immediate shift — alias for UBFM/SBFM
            let bits = if sf_bit == 1 { 64u32 } else { 32u32 };
            let amount = (*imm as u32) & (bits - 1);

            match mnemonic {
                "lsl" => {
                    // LSL Rd, Rn, #n → UBFM Rd, Rn, #(-n mod bits), #(bits-1-n)
                    let immr = (bits.wrapping_sub(amount)) & (bits - 1);
                    let imms = bits - 1 - amount;
                    let n = sf_bit;
                    let word = (sf_bit << 31)
                        | (0b10 << 29)
                        | (0b100110 << 23)
                        | (n << 22)
                        | (immr << 16)
                        | (imms << 10)
                        | ((rn.a64_reg_num() as u32) << 5)
                        | (rd.a64_reg_num() as u32);
                    emit32(buf, word);
                }
                "lsr" => {
                    // LSR Rd, Rn, #n → UBFM Rd, Rn, #n, #(bits-1)
                    let n = sf_bit;
                    let word = (sf_bit << 31)
                        | (0b10 << 29)
                        | (0b100110 << 23)
                        | (n << 22)
                        | (amount << 16)
                        | ((bits - 1) << 10)
                        | ((rn.a64_reg_num() as u32) << 5)
                        | (rd.a64_reg_num() as u32);
                    emit32(buf, word);
                }
                "asr" => {
                    // ASR Rd, Rn, #n → SBFM Rd, Rn, #n, #(bits-1)
                    let n = sf_bit;
                    #[allow(clippy::identity_op)]
                    let word = (sf_bit << 31)
                        | (0b00 << 29)
                        | (0b100110 << 23)
                        | (n << 22)
                        | (amount << 16)
                        | ((bits - 1) << 10)
                        | ((rn.a64_reg_num() as u32) << 5)
                        | (rd.a64_reg_num() as u32);
                    emit32(buf, word);
                }
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "immediate shift not supported for this op",
                        instr.span,
                    ))
                }
            }
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "expected register or immediate",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── System / NOP / SVC / BRK ─────────────────────────────────────────────

fn encode_svc(buf: &mut InstrBytes, ops: &[Operand], instr: &Instruction) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_ops("svc", "expected immediate", instr.span));
    }
    let imm = get_imm(&ops[0], "svc", instr.span)? as u32;
    if imm > 0xFFFF {
        return Err(invalid_ops(
            "svc",
            "SVC number must fit in 16 bits",
            instr.span,
        ));
    }
    // 11010100|000|imm16|000|01
    let word = 0xD400_0001 | (imm << 5);
    emit32(buf, word);
    Ok(())
}

fn encode_brk(buf: &mut InstrBytes, ops: &[Operand], instr: &Instruction) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_ops("brk", "expected immediate", instr.span));
    }
    let imm = get_imm(&ops[0], "brk", instr.span)? as u32;
    if imm > 0xFFFF {
        return Err(invalid_ops(
            "brk",
            "BRK number must fit in 16 bits",
            instr.span,
        ));
    }
    // 11010100|001|imm16|000|00
    let word = 0xD420_0000 | (imm << 5);
    emit32(buf, word);
    Ok(())
}

fn encode_hlt(buf: &mut InstrBytes, ops: &[Operand], instr: &Instruction) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_ops("hlt", "expected immediate", instr.span));
    }
    let imm = get_imm(&ops[0], "hlt", instr.span)? as u32;
    if imm > 0xFFFF {
        return Err(invalid_ops(
            "hlt",
            "HLT number must fit in 16 bits",
            instr.span,
        ));
    }
    // 11010100|010|imm16|000|00
    let word = 0xD440_0000 | (imm << 5);
    emit32(buf, word);
    Ok(())
}

// ── CSEL / CSINC / CSINV / CSNEG ────────────────────────────────────────

fn encode_csel(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() < 3 {
        return Err(invalid_ops(
            mnemonic,
            "expected Rd, Rn, Rm, cond",
            instr.span,
        ));
    }
    let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
    let rm = get_a64_reg(&ops[2], mnemonic, instr.span)?;
    let sf_bit = sf(rd);

    // Condition from 4th operand (immediate or condition name)
    let cc = if ops.len() >= 4 {
        get_cond(&ops[3], mnemonic, instr.span)?
    } else {
        return Err(invalid_ops(
            mnemonic,
            "expected condition code as 4th operand",
            instr.span,
        ));
    };

    let (op, op2) = match mnemonic {
        "csel" => (0u32, 0u32),
        "csinc" => (0, 1),
        "csinv" => (1, 0),
        "csneg" => (1, 1),
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "unknown conditional select",
                instr.span,
            ))
        }
    };

    // sf|op|0|11010100|Rm|cond|0|op2|Rn|Rd
    let word = (sf_bit << 31)
        | (op << 30)
        | (0b011010100 << 21)
        | ((rm.a64_reg_num() as u32) << 16)
        | (cc << 12)
        | (op2 << 10)
        | ((rn.a64_reg_num() as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── ADR / ADRP ───────────────────────────────────────────────────────────

fn encode_adr(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
    relax: &mut Option<RelaxInfo>,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(mnemonic, "expected Rd, label", instr.span));
    }
    let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let is_page = mnemonic == "adrp";
    let rd_num = rd.a64_reg_num() as u32;

    match &ops[1] {
        Operand::Label(label) => {
            if is_page {
                // ADRP — no relaxation, already ±4 GB range
                let word = (1u32 << 31) | (0b10000 << 24) | rd_num;
                let reloc_offset = buf.len();
                emit32(buf, word);
                *reloc = Some(Relocation {
                    offset: reloc_offset,
                    size: 4,
                    label: alloc::rc::Rc::from(&**label),
                    kind: RelocKind::Aarch64Adrp,
                    addend: 0,
                    trailing_bytes: 0,
                });
            } else {
                // ADR with relaxation: long form is ADRP+ADD (8 bytes)
                // Long form: ADRP Xd, label; ADD Xd, Xd, :lo12:label
                // Use 64-bit register for ADRP (sf=1, always Xd)
                let sf = 1u32;
                let adrp_word = (sf << 31) | (0b10000 << 24) | rd_num;
                let reloc_offset = buf.len();
                emit32(buf, adrp_word);

                // ADD Xd, Xd, #0  (placeholder, linker fills lo12)
                // sf|0|0|100010|sh=0|imm12|Rn|Rd
                let add_word = (sf << 31) | (0b00100010 << 23) | (rd_num << 5) | rd_num;
                emit32(buf, add_word);

                *reloc = Some(Relocation {
                    offset: reloc_offset,
                    size: 8,
                    label: alloc::rc::Rc::from(&**label),
                    kind: RelocKind::Aarch64AdrpAddPair,
                    addend: 0,
                    trailing_bytes: 0,
                });

                // Short form: ADR Xd, label (4 bytes, ±1 MB)
                let mut short = InstrBytes::new();
                let adr_word = (0b10000 << 24) | rd_num;
                emit32(&mut short, adr_word);

                *relax = Some(RelaxInfo {
                    short_bytes: short,
                    short_reloc_offset: 0,
                    short_relocation: Some(Relocation {
                        offset: 0,
                        size: 4,
                        label: alloc::rc::Rc::from(&**label),
                        kind: RelocKind::Aarch64Adr21,
                        addend: 0,
                        trailing_bytes: 0,
                    }),
                });
            }
        }
        Operand::Immediate(imm) => {
            let offset = *imm as i32;
            let immhi = ((offset >> 2) as u32) & 0x7FFFF;
            let immlo = (offset as u32) & 0x3;
            let op = is_page as u32;
            let word = (op << 31) | (immlo << 29) | (0b10000 << 24) | (immhi << 5) | rd_num;
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "expected label or immediate",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── MUL / MADD / MSUB / SDIV / UDIV ─────────────────────────────────────

/// MUL Xd, Xn, Xm → MADD Xd, Xn, Xm, XZR
/// MADD/MSUB: sf|00|11011|000|Rm|o0|Ra|Rn|Rd
/// SDIV/UDIV: sf|0|0|11010110|Rm|00001|o1|Rn|Rd
fn encode_mul_div(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    match mnemonic {
        "mul" | "mneg" => {
            // MUL Xd, Xn, Xm → MADD Xd, Xn, Xm, XZR (o0=0)
            // MNEG Xd, Xn, Xm → MSUB Xd, Xn, Xm, XZR (o0=1)
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected Rd, Rn, Rm", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let rm = get_a64_reg(&ops[2], mnemonic, instr.span)?;
            let sf_bit = sf(rd);
            let o0 = if mnemonic == "mneg" { 1u32 } else { 0u32 };
            // Ra = XZR (31)
            let word = (sf_bit << 31)
                | (0b0011011000u32 << 21)
                | ((rm.a64_reg_num() as u32) << 16)
                | (o0 << 15)
                | (0b11111 << 10) // Ra = XZR
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        "madd" | "msub" => {
            // MADD/MSUB Xd, Xn, Xm, Xa
            if ops.len() != 4 {
                return Err(invalid_ops(mnemonic, "expected Rd, Rn, Rm, Ra", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let rm = get_a64_reg(&ops[2], mnemonic, instr.span)?;
            let ra = get_a64_reg(&ops[3], mnemonic, instr.span)?;
            let sf_bit = sf(rd);
            let o0 = if mnemonic == "msub" { 1u32 } else { 0u32 };
            let word = (sf_bit << 31)
                | (0b0011011000u32 << 21)
                | ((rm.a64_reg_num() as u32) << 16)
                | (o0 << 15)
                | ((ra.a64_reg_num() as u32) << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        "smull" | "umull" => {
            // SMULL Xd, Wn, Wm → SMADDL Xd, Wn, Wm, XZR
            // sf=1|00|11011|U01|Rm|0|11111|Rn|Rd
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected Xd, Wn, Wm", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let rm = get_a64_reg(&ops[2], mnemonic, instr.span)?;
            let u = if mnemonic == "umull" { 1u32 } else { 0u32 };
            // 1|00|11011|U|01|Rm|0|11111|Rn|Rd
            let word = ((1u32 << 31)
                | (0b0011011u32 << 24)
                | (u << 23)
                | (0b01 << 21)
                | ((rm.a64_reg_num() as u32) << 16))
                | (0b11111 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        // ── Widening multiply-add / subtract (4 operands) ──
        "smaddl" | "umaddl" | "smsubl" | "umsubl" => {
            // SMADDL Xd, Wn, Wm, Xa  →  Xd = (i64)(Wn) * (i64)(Wm) + Xa
            // UMADDL Xd, Wn, Wm, Xa  →  Xd = (u64)(Wn) * (u64)(Wm) + Xa
            // SMSUBL Xd, Wn, Wm, Xa  →  Xd = Xa - (i64)(Wn) * (i64)(Wm)
            // UMSUBL Xd, Wn, Wm, Xa  →  Xd = Xa - (u64)(Wn) * (u64)(Wm)
            // Encoding: 1|00|11011|U|01|Rm|o0|Ra|Rn|Rd
            if ops.len() != 4 {
                return Err(invalid_ops(mnemonic, "expected Xd, Wn, Wm, Xa", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let rm = get_a64_reg(&ops[2], mnemonic, instr.span)?;
            let ra = get_a64_reg(&ops[3], mnemonic, instr.span)?;
            let u: u32 = if mnemonic.starts_with('u') { 1 } else { 0 };
            let o0: u32 = if mnemonic.contains("sub") { 1 } else { 0 };
            let word = (1u32 << 31)
                | (0b0011011u32 << 24)
                | (u << 23)
                | (0b01 << 21)
                | ((rm.a64_reg_num() as u32) << 16)
                | (o0 << 15)
                | ((ra.a64_reg_num() as u32) << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        // ── Widening multiply-negate (3-op aliases) ──────────
        "smnegl" | "umnegl" => {
            // SMNEGL Xd, Wn, Wm  →  SMSUBL Xd, Wn, Wm, XZR
            // UMNEGL Xd, Wn, Wm  →  UMSUBL Xd, Wn, Wm, XZR
            // Encoding: 1|00|11011|U|01|Rm|1|11111|Rn|Rd  (o0=1, Ra=XZR)
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected Xd, Wn, Wm", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let rm = get_a64_reg(&ops[2], mnemonic, instr.span)?;
            let u: u32 = if mnemonic == "umnegl" { 1 } else { 0 };
            let word = (1u32 << 31)
                | (0b0011011u32 << 24)
                | (u << 23)
                | (0b01 << 21)
                | ((rm.a64_reg_num() as u32) << 16)
                | (1u32 << 15) // o0=1 (subtract)
                | (0b11111u32 << 10) // Ra=XZR
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        // ── Multiply high (upper 64 bits of 64×64→128) ──────
        "smulh" | "umulh" => {
            // SMULH Xd, Xn, Xm  →  Xd = (Xn * Xm)[127:64]  (signed)
            // UMULH Xd, Xn, Xm  →  Xd = (Xn * Xm)[127:64]  (unsigned)
            // Encoding: 1|00|11011|U|10|Rm|0|11111|Rn|Rd
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected Xd, Xn, Xm", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let rm = get_a64_reg(&ops[2], mnemonic, instr.span)?;
            let u: u32 = if mnemonic == "umulh" { 1 } else { 0 };
            let word = (1u32 << 31)
                | (0b0011011u32 << 24)
                | (u << 23)
                | (0b10 << 21) // op31[1:0]=10 for MULH
                | ((rm.a64_reg_num() as u32) << 16)
                | (0b11111u32 << 10) // Ra=11111 (ignored for MULH)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        "sdiv" | "udiv" => {
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected Rd, Rn, Rm", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let rm = get_a64_reg(&ops[2], mnemonic, instr.span)?;
            let sf_bit = sf(rd);
            // Data-processing (2 source): sf|0|0|11010110|Rm|opcode|Rn|Rd
            // UDIV opcode=000010, SDIV opcode=000011
            let opcode = if mnemonic == "sdiv" {
                0b000011u32
            } else {
                0b000010u32
            };
            let word = (sf_bit << 31)
                | (0b0011010110u32 << 21)
                | ((rm.a64_reg_num() as u32) << 16)
                | (opcode << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        _ => return Err(invalid_ops(mnemonic, "unknown mul/div variant", instr.span)),
    }
    Ok(())
}

// ── MVN ──────────────────────────────────────────────────────────────────

/// MVN Xd, Xm → ORN Xd, XZR, Xm
fn encode_mvn(buf: &mut InstrBytes, ops: &[Operand], instr: &Instruction) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops("mvn", "expected Rd, Rm", instr.span));
    }
    let rd = get_a64_reg(&ops[0], "mvn", instr.span)?;
    let rm = get_a64_reg(&ops[1], "mvn", instr.span)?;
    let sf_bit = sf(rd);
    // ORN: sf|01|01010|00|1|Rm|000000|11111|Rd  (Rn=XZR)
    let word = (sf_bit << 31)
        | (0b01 << 29)
        | (0b01010 << 24)
        | (1 << 21) // N=1 (ORN)
        | ((rm.a64_reg_num() as u32) << 16)
        | (0b11111 << 5) // Rn = XZR
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── Conditional aliases: CSET / CSETM / CINC / CNEG ─────────────────────

/// Invert a condition code (EQ↔NE, CS↔CC, etc.)
fn invert_cond(cc: u32) -> u32 {
    cc ^ 1
}

/// Parse a condition code from an operand (either label name or immediate).
fn get_cond(op: &Operand, mnemonic: &str, span: crate::error::Span) -> Result<u32, AsmError> {
    match op {
        Operand::Immediate(v) => Ok((*v as u32) & 0xF),
        Operand::Label(name) => {
            cond_code(name).ok_or_else(|| invalid_ops(mnemonic, "unknown condition code", span))
        }
        _ => Err(invalid_ops(mnemonic, "expected condition code", span)),
    }
}

fn encode_cond_alias(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    match mnemonic {
        "cset" | "csetm" => {
            // CSET  Xd, cond → CSINC Xd, XZR, XZR, invert(cond)
            // CSETM Xd, cond → CSINV Xd, XZR, XZR, invert(cond)
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected Rd, cond", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let cc = invert_cond(get_cond(&ops[1], mnemonic, instr.span)?);
            let sf_bit = sf(rd);
            let (op, op2) = if mnemonic == "cset" {
                (0u32, 1u32)
            } else {
                (1u32, 0u32)
            };
            // sf|op|0|11010100|Rm(=11111)|cond|0|op2|Rn(=11111)|Rd
            let word = (sf_bit << 31)
                | (op << 30)
                | (0b011010100 << 21)
                | (0b11111 << 16) // Rm = XZR
                | (cc << 12)
                | (op2 << 10)
                | (0b11111 << 5)  // Rn = XZR
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        "cinc" | "cneg" => {
            // CINC Xd, Xn, cond → CSINC Xd, Xn, Xn, invert(cond)
            // CNEG Xd, Xn, cond → CSNEG Xd, Xn, Xn, invert(cond)
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected Rd, Rn, cond", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let cc = invert_cond(get_cond(&ops[2], mnemonic, instr.span)?);
            let sf_bit = sf(rd);
            let (op, op2) = if mnemonic == "cinc" {
                (0u32, 1u32)
            } else {
                (1u32, 1u32)
            };
            let word = (sf_bit << 31)
                | (op << 30)
                | (0b011010100 << 21)
                | ((rn.a64_reg_num() as u32) << 16) // Rm = Rn
                | (cc << 12)
                | (op2 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "unknown conditional alias",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── TBZ / TBNZ ──────────────────────────────────────────────────────────

/// TBZ/TBNZ Rt, #bit, label — with relaxation for out-of-range targets.
///
/// Short form (4 bytes): `TBZ/TBNZ Rt, #bit, label` (±32 KB, 14-bit offset)
/// Long form (8 bytes): `TBNZ/TBZ Rt, #bit, +8; B label` (±128 MB via B)
fn encode_tbz(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
    relax: &mut Option<RelaxInfo>,
) -> Result<(), AsmError> {
    if ops.len() != 3 {
        return Err(invalid_ops(
            mnemonic,
            "expected Rt, #bit, label",
            instr.span,
        ));
    }
    let rt = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let bit = get_imm(&ops[1], mnemonic, instr.span)? as u32;
    if bit > 63 {
        return Err(invalid_ops(mnemonic, "bit number must be 0-63", instr.span));
    }
    let op = if mnemonic == "tbnz" { 1u32 } else { 0u32 };
    let b5 = (bit >> 5) & 1;
    let b40 = bit & 0x1F;

    match &ops[2] {
        Operand::Label(label) => {
            // ── Long form: {inverted} Rt, #bit, +8; B label ──
            let inv_op = op ^ 1; // TBZ↔TBNZ inversion
                                 // Inverted TBZ/TBNZ Rt, #bit, +8: imm14 = 2 (skip 8 bytes)
            let skip_word = (b5 << 31)
                | (0b011011 << 25)
                | (inv_op << 24)
                | (b40 << 19)
                | (2u32 << 5)
                | (rt.a64_reg_num() as u32);
            emit32(buf, skip_word);
            // B label: unconditional branch placeholder
            let b_word = 0b000101u32 << 26;
            let reloc_offset = buf.len();
            emit32(buf, b_word);

            *reloc = Some(Relocation {
                offset: reloc_offset,
                size: 4,
                label: alloc::rc::Rc::from(&**label),
                kind: RelocKind::Aarch64Jump26,
                addend: 0,
                trailing_bytes: 0,
            });

            // ── Short form: original TBZ/TBNZ Rt, #bit, label ──
            let mut short = InstrBytes::new();
            let short_word = (b5 << 31)
                | (0b011011 << 25)
                | (op << 24)
                | (b40 << 19)
                | (rt.a64_reg_num() as u32);
            emit32(&mut short, short_word);

            *relax = Some(RelaxInfo {
                short_bytes: short,
                short_reloc_offset: 0,
                short_relocation: Some(Relocation {
                    offset: 0,
                    size: 4,
                    label: alloc::rc::Rc::from(&**label),
                    kind: RelocKind::Aarch64Branch14,
                    addend: 0,
                    trailing_bytes: 0,
                }),
            });
        }
        Operand::Immediate(imm) => {
            let off14 = ((*imm as i32) >> 2) & 0x3FFF;
            let word = (b5 << 31)
                | (0b011011 << 25)
                | (op << 24)
                | (b40 << 19)
                | ((off14 as u32) << 5)
                | (rt.a64_reg_num() as u32);
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "expected label or immediate",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── Bit manipulation: CLZ / CLS / RBIT / REV / REV16 / REV32 ────────────

/// Data-processing (1 source): sf|1|0|11010110|00000|opcode[5:0]|Rn|Rd
fn encode_bitmanip(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(mnemonic, "expected Rd, Rn", instr.span));
    }
    let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
    let sf_bit = sf(rd);

    let opcode = match mnemonic {
        "rbit" => 0b000000u32,
        "rev16" => 0b000001,
        "rev32" => {
            if sf_bit == 0 {
                return Err(invalid_ops(
                    mnemonic,
                    "REV32 requires 64-bit registers",
                    instr.span,
                ));
            }
            0b000010
        }
        "rev" => {
            if sf_bit == 1 {
                0b000011
            } else {
                0b000010
            }
        }
        "clz" => 0b000100,
        "cls" => 0b000101,
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "unknown bit manipulation",
                instr.span,
            ))
        }
    };

    // sf|1|0|11010110|00000|opcode|Rn|Rd
    let word = (sf_bit << 31)
        | (1 << 30)
        | (0b0011010110 << 21)
        | (opcode << 10)
        | ((rn.a64_reg_num() as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── Extend: UXTB / UXTH / SXTB / SXTH / SXTW ──────────────────────────

/// UXTB Wd, Wn  → UBFM Wd, Wn, #0, #7
/// UXTH Wd, Wn  → UBFM Wd, Wn, #0, #15
/// SXTB Xd, Wn  → SBFM Xd, Xn, #0, #7
/// SXTH Xd, Wn  → SBFM Xd, Xn, #0, #15
/// SXTW Xd, Wn  → SBFM Xd, Xn, #0, #31
fn encode_extend(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(mnemonic, "expected Rd, Rn", instr.span));
    }
    let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;

    let (opc, _n, imms, use_sf) = match mnemonic {
        "uxtb" => (0b10u32, 0u32, 7u32, 0u32), // UBFM Wd, Wn, #0, #7 (sf=0)
        "uxth" => (0b10, 0, 15, 0),            // UBFM Wd, Wn, #0, #15 (sf=0)
        "sxtb" => (0b00, 0, 7, sf(rd)),        // SBFM, sf from dest
        "sxth" => (0b00, 0, 15, sf(rd)),
        "sxtw" => (0b00, 1, 31, 1), // SBFM Xd, Xn, #0, #31 (sf=1, N=1)
        _ => return Err(invalid_ops(mnemonic, "unknown extend op", instr.span)),
    };

    let n_bit = if mnemonic == "sxtw" { 1u32 } else { use_sf };

    // sf|opc|100110|N|immr(=0)|imms|Rn|Rd
    let word = ((use_sf << 31)
        | (opc << 29)
        | (0b100110 << 23)
        | (n_bit << 22))   // immr = 0
        | (imms << 10)
        | ((rn.a64_reg_num() as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── MRS / MSR ────────────────────────────────────────────────────────────

/// Map system register names to their 16-bit encoding: op0:op1:CRn:CRm:op2
#[allow(clippy::unusual_byte_groupings)]
fn sysreg_encoding(name: &str) -> Option<u32> {
    match name {
        "nzcv" => Some(0b11_011_0100_0010_000),        // S3_3_C4_C2_0
        "fpcr" => Some(0b11_011_0100_0100_000),        // S3_3_C4_C4_0
        "fpsr" => Some(0b11_011_0100_0100_001),        // S3_3_C4_C4_1
        "currentel" => Some(0b11_000_0100_0010_010),   // S3_0_C4_C2_2
        "daif" => Some(0b11_011_0100_0010_001),        // S3_3_C4_C2_1
        "tpidr_el0" => Some(0b11_011_1101_0000_010),   // S3_3_C13_C0_2
        "tpidrro_el0" => Some(0b11_011_1101_0000_011), // S3_3_C13_C0_3
        "ctr_el0" => Some(0b11_011_0000_0000_001),     // S3_3_C0_C0_1
        "dczid_el0" => Some(0b11_011_0000_0000_111),   // S3_3_C0_C0_7
        "cntvct_el0" => Some(0b11_011_1110_0000_010),  // S3_3_C14_C0_2
        "cntfrq_el0" => Some(0b11_011_1110_0000_000),  // S3_3_C14_C0_0
        "sp_el0" => Some(0b11_000_0100_0001_000),      // S3_0_C4_C1_0
        "spsel" => Some(0b11_000_0100_0010_000),       // S3_0_C4_C2_0
        _ => None,
    }
}

fn encode_mrs_msr(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_ops(mnemonic, "expected 2 operands", instr.span));
    }

    match mnemonic {
        "mrs" => {
            // MRS Xt, <sysreg>
            let rt = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let encoding = match &ops[1] {
                Operand::Label(name) => sysreg_encoding(&name.to_ascii_lowercase())
                    .ok_or_else(|| invalid_ops("mrs", "unknown system register", instr.span))?,
                Operand::Immediate(v) => *v as u32,
                _ => {
                    return Err(invalid_ops(
                        "mrs",
                        "expected system register name or encoding",
                        instr.span,
                    ))
                }
            };
            // 1101010100|1|1|op0|op1|CRn|CRm|op2|Rt
            let word = 0xD530_0000 | (encoding << 5) | (rt.a64_reg_num() as u32);
            emit32(buf, word);
        }
        "msr" => {
            // MSR <sysreg>, Xt
            let encoding = match &ops[0] {
                Operand::Label(name) => sysreg_encoding(&name.to_ascii_lowercase())
                    .ok_or_else(|| invalid_ops("msr", "unknown system register", instr.span))?,
                Operand::Immediate(v) => *v as u32,
                _ => {
                    return Err(invalid_ops(
                        "msr",
                        "expected system register name or encoding",
                        instr.span,
                    ))
                }
            };
            let rt = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            // 1101010100|0|1|op0|op1|CRn|CRm|op2|Rt
            let word = 0xD510_0000 | (encoding << 5) | (rt.a64_reg_num() as u32);
            emit32(buf, word);
        }
        _ => return Err(invalid_ops(mnemonic, "unknown system op", instr.span)),
    }
    Ok(())
}

// ── Barrier instructions: DMB / DSB / ISB ────────────────────────────────

fn barrier_option(name: &str) -> Option<u32> {
    match name {
        "sy" => Some(0xF),
        "ish" => Some(0xB),
        "ishld" => Some(0x9),
        "ishst" => Some(0xA),
        "nsh" => Some(0x7),
        "nshld" => Some(0x5),
        "nshst" => Some(0x6),
        "osh" => Some(0x3),
        "oshld" => Some(0x1),
        "oshst" => Some(0x2),
        "ld" => Some(0xD),
        "st" => Some(0xE),
        _ => None,
    }
}

fn encode_barrier(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    // Default CRm = SY (0xF) if no operand
    let crm = if ops.is_empty() {
        0xFu32
    } else {
        match &ops[0] {
            Operand::Immediate(v) => (*v as u32) & 0xF,
            Operand::Label(name) => barrier_option(&name.to_ascii_lowercase())
                .ok_or_else(|| invalid_ops(mnemonic, "unknown barrier option", instr.span))?,
            _ => return Err(invalid_ops(mnemonic, "expected barrier option", instr.span)),
        }
    };

    let word = match mnemonic {
        // DMB: 11010101000000110011|CRm|1|01|11111
        "dmb" => 0xD503_30BF | (crm << 8),
        // DSB: 11010101000000110011|CRm|1|00|11111
        "dsb" => 0xD503_309F | (crm << 8),
        // ISB: 11010101000000110011|CRm|1|10|11111
        "isb" => 0xD503_30DF | (crm << 8),
        _ => return Err(invalid_ops(mnemonic, "unknown barrier", instr.span)),
    };
    emit32(buf, word);
    Ok(())
}

// ── Logical Immediate ────────────────────────────────────────────────────

/// Attempt to encode a bitmask immediate for AArch64 logical instructions.
/// Returns `Some((n, immr, imms))` or `None` if not encodable.
fn encode_bitmask_imm(value: u64, reg_size: u32) -> Option<(u32, u32, u32)> {
    if reg_size == 32 {
        let w = value as u32;
        if w == 0 || w == 0xFFFF_FFFF {
            return None;
        }
        // Replicate to 64 bits for the encoding algorithm
        let val64 = (w as u64) | ((w as u64) << 32);
        return encode_bitmask_imm_inner(val64);
    }
    if value == 0 || value == u64::MAX {
        return None;
    }
    encode_bitmask_imm_inner(value)
}

fn encode_bitmask_imm_inner(value: u64) -> Option<(u32, u32, u32)> {
    // Find the smallest repeating element size
    let mut imm = value;
    let mut size = 64u32;

    loop {
        let half = size >> 1;
        if half < 2 {
            break;
        }
        let mask = (1u64 << half) - 1;
        if (imm & mask) == ((imm >> half) & mask) {
            size = half;
            imm &= mask;
        } else {
            break;
        }
    }

    let ones = imm.count_ones();
    if ones == 0 || ones == size {
        return None;
    }

    // Find rotation: rotate element right by r to get contiguous 1s at LSB
    let mask = if size == 64 {
        u64::MAX
    } else {
        (1u64 << size) - 1
    };
    let mut immr = 0u32;

    for r in 0..size {
        let rot = if size == 64 {
            imm.rotate_right(r)
        } else {
            let doubled = imm | (imm << size);
            (doubled >> r) & mask
        };
        if rot.trailing_ones() == ones && (rot >> ones) == 0 {
            immr = r;
            break;
        }
    }

    // N bit: 1 for 64-bit element, 0 otherwise
    let n = if size == 64 { 1u32 } else { 0u32 };

    // Compute imms
    let imms = if size == 64 {
        (ones - 1) & 0x3F
    } else {
        // Upper bits encode element size:
        // size=32 → 0x00, size=16 → 0x20, size=8 → 0x30, size=4 → 0x38, size=2 → 0x3C
        let upper = (!(size * 2 - 1)) & 0x3F;
        upper | ((ones - 1) & (size - 1))
    };

    Some((n, immr, imms))
}

/// AND/ORR/EOR/ANDS (immediate): sf|opc|100100|N|immr|imms|Rn|Rd
fn encode_logical_imm(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    let (rd, rn, imm_val) = if matches!(mnemonic, "tst") {
        // TST Rn, #imm → ANDS XZR, Rn, #imm
        if ops.len() != 2 {
            return Err(invalid_ops(mnemonic, "expected Rn, #imm", instr.span));
        }
        let rn = get_a64_reg(&ops[0], mnemonic, instr.span)?;
        let imm = get_imm(&ops[1], mnemonic, instr.span)? as u64;
        let zr = if rn.is_a64_64bit() {
            Register::A64Xzr
        } else {
            Register::A64Wzr
        };
        (zr, rn, imm)
    } else {
        if ops.len() != 3 {
            return Err(invalid_ops(mnemonic, "expected Rd, Rn, #imm", instr.span));
        }
        let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
        let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
        let imm = get_imm(&ops[2], mnemonic, instr.span)? as u64;
        (rd, rn, imm)
    };

    let sf_bit = sf(rn);
    let reg_size = if sf_bit == 1 { 64 } else { 32 };

    let (n, immr, imms) = encode_bitmask_imm(imm_val, reg_size).ok_or_else(|| {
        invalid_ops(
            mnemonic,
            "immediate cannot be encoded as bitmask",
            instr.span,
        )
    })?;

    let opc = match mnemonic {
        "and" => 0b00u32,
        "orr" => 0b01,
        "eor" => 0b10,
        "ands" | "tst" => 0b11,
        _ => return Err(invalid_ops(mnemonic, "unknown logical op", instr.span)),
    };

    // sf|opc|100100|N|immr|imms|Rn|Rd
    let word = (sf_bit << 31)
        | (opc << 29)
        | (0b100100 << 23)
        | (n << 22)
        | (immr << 16)
        | (imms << 10)
        | ((rn.a64_reg_num() as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── Pre-index / Post-index LDR/STR ──────────────────────────────────────

/// Encode LDR/STR with pre-index or post-index addressing.
/// Pre-index:  size|111|V|00|opc|imm9|11|Rn|Rt
/// Post-index: size|111|V|00|opc|imm9|01|Rn|Rt
fn encode_ldr_str_idx(
    buf: &mut InstrBytes,
    mnemonic: &str,
    rt: Register,
    mem: &MemoryOperand,
    post_imm: Option<i64>,
    instr: &Instruction,
) -> Result<(), AsmError> {
    let rn = match mem.base {
        Some(r) if r.is_aarch64() => r,
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "memory base must be AArch64 register",
                instr.span,
            ))
        }
    };

    let (is_pre, offset) = if mem.addr_mode == AddrMode::PreIndex {
        (true, mem.disp)
    } else if let Some(imm) = post_imm {
        (false, imm)
    } else {
        return Err(invalid_ops(
            mnemonic,
            "expected pre-index or post-index addressing",
            instr.span,
        ));
    };

    if !(-256..=255).contains(&offset) {
        return Err(invalid_ops(
            mnemonic,
            "offset must be in range -256..255 for pre/post-index",
            instr.span,
        ));
    }

    let (size_bits, is_load) = match mnemonic {
        "ldr" => (if rt.is_a64_64bit() { 3u32 } else { 2u32 }, true),
        "str" => (if rt.is_a64_64bit() { 3u32 } else { 2u32 }, false),
        "ldrb" => (0, true),
        "strb" => (0, false),
        "ldrh" => (1, true),
        "strh" => (1, false),
        "ldrsb" => (0, true),
        "ldrsh" => (1, true),
        "ldrsw" => (2, true),
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "unknown load/store variant",
                instr.span,
            ))
        }
    };

    let opc = if is_load { 0b01u32 } else { 0b00u32 };
    let opc = match mnemonic {
        "ldrsb" => {
            if rt.is_a64_64bit() {
                0b10
            } else {
                0b11
            }
        }
        "ldrsh" => {
            if rt.is_a64_64bit() {
                0b10
            } else {
                0b11
            }
        }
        "ldrsw" => 0b10,
        _ => opc,
    };

    let idx_bits = if is_pre { 0b11u32 } else { 0b01u32 };
    let imm9 = (offset as u32) & 0x1FF;

    // size|111|V=0|00|opc|0|imm9|idx|Rn|Rt
    let word = (size_bits << 30)
        | (0b111000 << 24)
        | (opc << 22)
        | (imm9 << 12)
        | (idx_bits << 10)
        | ((rn.a64_reg_num() as u32) << 5)
        | (rt.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

/// STP/LDP pre-index: opc|101|0|011|L|imm7|Rt2|Rn|Rt
/// STP/LDP post-index: opc|101|0|001|L|imm7|Rt2|Rn|Rt
fn encode_stp_ldp_idx(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() < 3 {
        return Err(invalid_ops(
            mnemonic,
            "expected Rt, Rt2, [Xn, #imm]! or [Xn], #imm",
            instr.span,
        ));
    }
    let rt = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let rt2 = get_a64_reg(&ops[1], mnemonic, instr.span)?;
    let is_load = matches!(mnemonic, "ldp");
    let opc = if rt.is_a64_64bit() { 0b10u32 } else { 0b00u32 };
    let scale = if rt.is_a64_64bit() { 8i64 } else { 4i64 };

    let mem = match &ops[2] {
        Operand::Memory(m) => m,
        _ => return Err(invalid_ops(mnemonic, "expected memory operand", instr.span)),
    };

    let rn = match mem.base {
        Some(r) if r.is_aarch64() => r,
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "memory base must be AArch64 register",
                instr.span,
            ))
        }
    };

    let (is_pre, offset) = if mem.addr_mode == AddrMode::PreIndex {
        (true, mem.disp)
    } else if ops.len() >= 4 {
        // Post-index: 4th operand is the immediate
        let imm = get_imm(&ops[3], mnemonic, instr.span)?;
        (false, imm as i64)
    } else {
        return Err(invalid_ops(
            mnemonic,
            "expected pre-index or post-index",
            instr.span,
        ));
    };

    if offset % scale != 0 {
        return Err(invalid_ops(
            mnemonic,
            "offset must be aligned to register size",
            instr.span,
        ));
    }
    let imm7 = ((offset / scale) as u32) & 0x7F;
    let l = is_load as u32;

    // Pre-index:  opc|101|0|011|L|imm7|Rt2|Rn|Rt
    // Post-index: opc|101|0|001|L|imm7|Rt2|Rn|Rt
    let mode_bits = if is_pre { 0b011u32 } else { 0b001u32 };
    let word = ((opc << 30) | (0b101 << 27))
        | (mode_bits << 23)
        | (l << 22)
        | (imm7 << 15)
        | ((rt2.a64_reg_num() as u32) << 10)
        | ((rn.a64_reg_num() as u32) << 5)
        | (rt.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── Atomic (LSE) instructions ────────────────────────────────────────────

/// Encode LSE atomic operations: LDADD, LDCLR, LDSET, LDEOR, SWP, CAS.
/// Each has variants: plain, A (acquire), L (release), AL (acquire+release).
///
/// LDADD/LDCLR/LDSET/LDEOR: size|111|0|00|A|R|1|Rs|o3|opc|00|Rn|Rt
/// SWP:                       size|111|0|00|A|R|1|Rs|1|000|00|Rn|Rt
/// CAS:                       size|001|0|00|A|1|Rs|R|11111|Rn|Rt
#[allow(clippy::identity_op)]
fn encode_atomic(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    // Parse ordering suffix (a/l/al) and size suffix (b/h)
    // E.g., "ldaddal" → base="ldadd", acquire=1, release=1
    // E.g., "swpalb" → base="swp", acquire=1, release=1, size=00 (byte)
    let (base_op, acquire, release, size_bits) = parse_atomic_suffixes(mnemonic)?;

    match base_op {
        "ldadd" | "ldclr" | "ldset" | "ldeor" => {
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected Rs, Rt, [Rn]", instr.span));
            }
            let rs = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rt = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let rn = match &ops[2] {
                Operand::Memory(m) => match m.base {
                    Some(r) if r.is_aarch64() => r,
                    _ => {
                        return Err(invalid_ops(
                            mnemonic,
                            "expected [Xn] memory operand",
                            instr.span,
                        ))
                    }
                },
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "expected [Xn] memory operand",
                        instr.span,
                    ))
                }
            };
            let opc = match base_op {
                "ldadd" => 0b000u32,
                "ldclr" => 0b001u32,
                "ldset" => 0b011u32,
                "ldeor" => 0b010u32,
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        &alloc::format!("unknown atomic load op '{}'", base_op),
                        instr.span,
                    ))
                }
            };
            // size|111|0|00|A|R|1|Rs|0|opc|00|Rn|Rt
            let word = (size_bits << 30)
                | (0b111_000 << 24)
                | (acquire << 23)
                | (release << 22)
                | (1u32 << 21)
                | ((rs.a64_reg_num() as u32) << 16)
                | (0u32 << 15) // o3=0
                | (opc << 12)
                | (0b00u32 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rt.a64_reg_num() as u32);
            emit32(buf, word);
        }
        "swp" => {
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected Rs, Rt, [Rn]", instr.span));
            }
            let rs = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rt = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let rn = match &ops[2] {
                Operand::Memory(m) => match m.base {
                    Some(r) if r.is_aarch64() => r,
                    _ => {
                        return Err(invalid_ops(
                            mnemonic,
                            "expected [Xn] memory operand",
                            instr.span,
                        ))
                    }
                },
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "expected [Xn] memory operand",
                        instr.span,
                    ))
                }
            };
            // size|111|0|00|A|R|1|Rs|1|000|00|Rn|Rt
            let word = (size_bits << 30)
                | (0b111_000 << 24)
                | (acquire << 23)
                | (release << 22)
                | (1u32 << 21)
                | ((rs.a64_reg_num() as u32) << 16)
                | (1u32 << 15) // o3=1 for SWP
                | (0b000u32 << 12)
                | (0b00u32 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rt.a64_reg_num() as u32);
            emit32(buf, word);
        }
        "cas" => {
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected Rs, Rt, [Rn]", instr.span));
            }
            let rs = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rt = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let rn = match &ops[2] {
                Operand::Memory(m) => match m.base {
                    Some(r) if r.is_aarch64() => r,
                    _ => {
                        return Err(invalid_ops(
                            mnemonic,
                            "expected [Xn] memory operand",
                            instr.span,
                        ))
                    }
                },
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "expected [Xn] memory operand",
                        instr.span,
                    ))
                }
            };
            // size|00|1000|A|1|Rs|R|11111|Rn|Rt
            let word = (size_bits << 30)
                | (0b001000 << 24)
                | (acquire << 23)
                | (1u32 << 22)
                | ((rs.a64_reg_num() as u32) << 16)
                | (release << 15)
                | (0b11111u32 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rt.a64_reg_num() as u32);
            emit32(buf, word);
        }
        "stadd" | "stclr" | "stset" | "steor" => {
            // Store atomics: like LD variants but Rt=XZR
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected Rs, [Rn]", instr.span));
            }
            let rs = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rn = match &ops[1] {
                Operand::Memory(m) => match m.base {
                    Some(r) if r.is_aarch64() => r,
                    _ => {
                        return Err(invalid_ops(
                            mnemonic,
                            "expected [Xn] memory operand",
                            instr.span,
                        ))
                    }
                },
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "expected [Xn] memory operand",
                        instr.span,
                    ))
                }
            };
            let opc = match base_op {
                "stadd" => 0b000u32,
                "stclr" => 0b001u32,
                "stset" => 0b011u32,
                "steor" => 0b010u32,
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        &alloc::format!("unknown atomic store op '{}'", base_op),
                        instr.span,
                    ))
                }
            };
            // size|111|0|00|A|R|1|Rs|0|opc|00|Rn|Rt=11111
            let word = (size_bits << 30)
                | (0b111_000 << 24)
                | (acquire << 23)
                | (release << 22)
                | (1u32 << 21)
                | ((rs.a64_reg_num() as u32) << 16)
                | (0u32 << 15) // o3=0
                | (opc << 12)
                | (0b00u32 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | 0b11111u32; // Rt = XZR (store-only variant)
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "unknown atomic operation",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// Parse atomic instruction suffixes: ordering (a/l/al) and size (b/h).
/// Returns (base_op, acquire_bit, release_bit, size_bits).
fn parse_atomic_suffixes(mnemonic: &str) -> Result<(&str, u32, u32, u32), AsmError> {
    // Known base operations
    let bases = [
        "ldadd", "ldclr", "ldset", "ldeor", "stadd", "stclr", "stset", "steor", "swp", "cas",
    ];

    for base in &bases {
        if let Some(suffix) = mnemonic.strip_prefix(base) {
            let (acquire, release, rest) = match suffix {
                s if s.starts_with("al") => (1u32, 1u32, &s[2..]),
                s if s.starts_with('a') => (1u32, 0u32, &s[1..]),
                s if s.starts_with('l') => (0u32, 1u32, &s[1..]),
                s => (0u32, 0u32, s),
            };
            let size = match rest {
                "" => 0b11u32,  // 64-bit (X registers)
                "b" => 0b00u32, // byte
                "h" => 0b01u32, // halfword
                _ => continue,  // not a valid suffix, try next base
            };
            return Ok((base, acquire, release, size));
        }
    }

    // If no base matched, return error
    Err(AsmError::UnknownMnemonic {
        mnemonic: String::from(mnemonic),
        arch: crate::error::ArchName::Aarch64,
        span: crate::error::Span::new(0, 0, 0, 0),
    })
}

// ── Load/Store Exclusive (LDXR / STXR) ──────────────────────────────────

/// Encode load-exclusive / store-exclusive instructions.
///
/// Load-exclusive:
///   LDXR  Rt, [Rn]          — size|001000|0|1|0|11111|0|11111|Rn|Rt
///   LDAXR Rt, [Rn]          — size|001000|0|1|0|11111|1|11111|Rn|Rt
///
/// Store-exclusive:
///   STXR  Ws, Rt, [Rn]     — size|001000|0|0|0|Rs|0|11111|Rn|Rt
///   STLXR Ws, Rt, [Rn]     — size|001000|0|0|0|Rs|1|11111|Rn|Rt
///
/// Size variants via suffix: b=00, h=01, (none/w)=10, (none/x)=11
#[allow(clippy::identity_op)]
fn encode_exclusive(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    // Determine base and attributes from mnemonic
    let (is_store, acquire, release, suffix_size) = parse_exclusive_mnemonic(mnemonic, instr.span)?;

    if is_store {
        // STXR/STLXR: Ws, Rt, [Xn]
        if ops.len() != 3 {
            return Err(invalid_ops(mnemonic, "expected Ws, Rt, [Xn]", instr.span));
        }
        let rs = get_a64_reg(&ops[0], mnemonic, instr.span)?;
        let rt = get_a64_reg(&ops[1], mnemonic, instr.span)?;
        let rn = match &ops[2] {
            Operand::Memory(m) => match m.base {
                Some(r) if r.is_aarch64() => r,
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "expected [Xn] memory operand",
                        instr.span,
                    ))
                }
            },
            _ => {
                return Err(invalid_ops(
                    mnemonic,
                    "expected [Xn] memory operand",
                    instr.span,
                ))
            }
        };
        // Infer size from register width for no-suffix variant
        let size_bits = match suffix_size {
            Some(s) => s,
            None => {
                if rt.is_a64_64bit() {
                    0b11u32
                } else {
                    0b10u32
                }
            }
        };
        // size|001000|0|0|0|Rs|o0|11111|Rn|Rt
        let word = (size_bits << 30)
            | (0b001000u32 << 24)
            | (0u32 << 23) // o2=0
            | (0u32 << 22) // L=0 (store)
            | (0u32 << 21) // o1=0
            | ((rs.a64_reg_num() as u32) << 16)
            | (release << 15) // o0: 1 for STLXR
            | (0b11111u32 << 10) // Rt2=11111
            | ((rn.a64_reg_num() as u32) << 5)
            | (rt.a64_reg_num() as u32);
        emit32(buf, word);
    } else {
        // LDXR/LDAXR: Rt, [Xn]
        if ops.len() != 2 {
            return Err(invalid_ops(mnemonic, "expected Rt, [Xn]", instr.span));
        }
        let rt = get_a64_reg(&ops[0], mnemonic, instr.span)?;
        let rn = match &ops[1] {
            Operand::Memory(m) => match m.base {
                Some(r) if r.is_aarch64() => r,
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "expected [Xn] memory operand",
                        instr.span,
                    ))
                }
            },
            _ => {
                return Err(invalid_ops(
                    mnemonic,
                    "expected [Xn] memory operand",
                    instr.span,
                ))
            }
        };
        // Infer size from register width for no-suffix variant
        let size_bits = match suffix_size {
            Some(s) => s,
            None => {
                if rt.is_a64_64bit() {
                    0b11u32
                } else {
                    0b10u32
                }
            }
        };
        // size|001000|0|1|0|11111|o0|11111|Rn|Rt
        let word = (size_bits << 30)
            | (0b001000u32 << 24)
            | (0u32 << 23) // o2=0
            | (1u32 << 22) // L=1 (load)
            | (0u32 << 21) // o1=0
            | (0b11111u32 << 16) // Rs=11111
            | (acquire << 15) // o0: 1 for LDAXR
            | (0b11111u32 << 10) // Rt2=11111
            | ((rn.a64_reg_num() as u32) << 5)
            | (rt.a64_reg_num() as u32);
        emit32(buf, word);
    }
    Ok(())
}

/// Parse exclusive instruction mnemonic to extract attributes.
/// Returns (is_store, acquire_bit, release_bit, Option<size_bits>).
/// None size means infer from register width.
fn parse_exclusive_mnemonic(
    mnemonic: &str,
    span: crate::error::Span,
) -> Result<(bool, u32, u32, Option<u32>), AsmError> {
    // Load variants: ldxr, ldaxr, ldxrb, ldxrh, ldaxrb, ldaxrh
    // Store variants: stxr, stlxr, stxrb, stxrh, stlxrb, stlxrh
    let (is_store, acquire, release, rest) = if let Some(s) = mnemonic.strip_prefix("stlxr") {
        (true, 0u32, 1u32, s)
    } else if let Some(s) = mnemonic.strip_prefix("stxr") {
        (true, 0u32, 0u32, s)
    } else if let Some(s) = mnemonic.strip_prefix("ldaxr") {
        (false, 1u32, 0u32, s)
    } else if let Some(s) = mnemonic.strip_prefix("ldxr") {
        (false, 0u32, 0u32, s)
    } else {
        return Err(invalid_ops(mnemonic, "unknown exclusive variant", span));
    };

    let size_bits = match rest {
        "" => None,           // infer from register width
        "b" => Some(0b00u32), // byte
        "h" => Some(0b01u32), // halfword
        _ => return Err(invalid_ops(mnemonic, "unknown size suffix", span)),
    };

    Ok((is_store, acquire, release, size_bits))
}

// ── Load-Acquire / Store-Release (non-exclusive) ─────────────────────────

/// Encode non-exclusive ordered memory operations.
///
/// Load-acquire:
///   LDAR   Xt, [Xn]   — 11|001000|1|1|0|11111|1|11111|Rn|Rt   (64-bit)
///   LDAR   Wt, [Xn]   — 10|001000|1|1|0|11111|1|11111|Rn|Rt   (32-bit)
///   LDARB  Wt, [Xn]   — 00|001000|1|1|0|11111|1|11111|Rn|Rt   (byte)
///   LDARH  Wt, [Xn]   — 01|001000|1|1|0|11111|1|11111|Rn|Rt   (halfword)
///
/// Store-release:
///   STLR   Xt, [Xn]   — 11|001000|1|0|0|11111|1|11111|Rn|Rt   (64-bit)
///   STLR   Wt, [Xn]   — 10|001000|1|0|0|11111|1|11111|Rn|Rt   (32-bit)
///   STLRB  Wt, [Xn]   — 00|001000|1|0|0|11111|1|11111|Rn|Rt   (byte)
///   STLRH  Wt, [Xn]   — 01|001000|1|0|0|11111|1|11111|Rn|Rt   (halfword)
fn encode_ordered(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    let (is_store, suffix_size) = parse_ordered_mnemonic(mnemonic, instr.span)?;

    // Both LDAR and STLR take: Rt, [Xn]
    if ops.len() != 2 {
        return Err(invalid_ops(mnemonic, "expected Rt, [Xn]", instr.span));
    }
    let rt = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let rn = match &ops[1] {
        Operand::Memory(m) => match m.base {
            Some(r) if r.is_aarch64() => r,
            _ => {
                return Err(invalid_ops(
                    mnemonic,
                    "expected [Xn] memory operand",
                    instr.span,
                ))
            }
        },
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "expected [Xn] memory operand",
                instr.span,
            ))
        }
    };

    // For no-suffix variant, detect size from register width
    let size_bits = match suffix_size {
        Some(s) => s,
        None => {
            if rt.is_a64_64bit() {
                0b11u32
            } else {
                0b10u32
            }
        }
    };

    let l = if is_store { 0u32 } else { 1u32 };

    // size|001000|1|L|0|11111|1|11111|Rn|Rt
    #[allow(clippy::identity_op)]
    let word = (size_bits << 30)
        | (0b001000u32 << 24)
        | (1u32 << 23)           // o2=1 (ordered, non-exclusive)
        | (l << 22)              // L: 1=load, 0=store
        | (0u32 << 21)           // o1=0
        | (0b11111u32 << 16)     // Rs=11111
        | (1u32 << 15)           // o0=1 (acquire/release)
        | (0b11111u32 << 10)     // Rt2=11111
        | ((rn.a64_reg_num() as u32) << 5)
        | (rt.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

/// Parse LDAR/STLR mnemonic. Returns (is_store, Option<size_bits>).
/// None size means infer from register width.
fn parse_ordered_mnemonic(
    mnemonic: &str,
    span: crate::error::Span,
) -> Result<(bool, Option<u32>), AsmError> {
    let (is_store, rest) = if let Some(s) = mnemonic.strip_prefix("stlr") {
        (true, s)
    } else if let Some(s) = mnemonic.strip_prefix("ldar") {
        (false, s)
    } else {
        return Err(invalid_ops(mnemonic, "unknown ordered variant", span));
    };

    let size = match rest {
        "" => None,           // infer from register
        "b" => Some(0b00u32), // byte
        "h" => Some(0b01u32), // halfword
        _ => return Err(invalid_ops(mnemonic, "unknown size suffix", span)),
    };

    Ok((is_store, size))
}

// ── Bitfield instructions ────────────────────────────────────────────────

/// Encode bitfield instructions: BFM, BFI, BFXIL, UBFM, UBFX, SBFM, SBFX.
///
/// BFM/UBFM/SBFM: sf|opc|100110|N|immr|imms|Rn|Rd
/// BFI:   alias for BFM Rd, Rn, #(-lsb mod 32/64), #(width-1)
/// BFXIL: alias for BFM Rd, Rn, #lsb, #(lsb+width-1)
/// UBFX:  alias for UBFM Rd, Rn, #lsb, #(lsb+width-1)
/// SBFX:  alias for SBFM Rd, Rn, #lsb, #(lsb+width-1)
fn encode_bitfield(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 4 {
        return Err(invalid_ops(
            mnemonic,
            "expected Rd, Rn, #immr, #imms",
            instr.span,
        ));
    }
    let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
    let imm1 = get_imm(&ops[2], mnemonic, instr.span)? as u32;
    let imm2 = get_imm(&ops[3], mnemonic, instr.span)? as u32;

    let sf = if rd.is_a64_64bit() { 1u32 } else { 0u32 };
    let bits = if sf == 1 { 64u32 } else { 32u32 };
    let n = sf; // N = sf for 64-bit

    // Determine opc and compute immr/imms from the alias form
    let (opc, immr, imms) = match mnemonic {
        "bfm" => (0b01u32, imm1 & 0x3F, imm2 & 0x3F),
        "ubfm" => (0b10u32, imm1 & 0x3F, imm2 & 0x3F),
        "sbfm" => (0b00u32, imm1 & 0x3F, imm2 & 0x3F),
        "bfi" => {
            // BFI Rd, Rn, #lsb, #width → BFM Rd, Rn, #(-lsb mod bits), #(width-1)
            let lsb = imm1;
            let width = imm2;
            let immr = (bits.wrapping_sub(lsb)) & (bits - 1);
            let imms = width.wrapping_sub(1);
            (0b01u32, immr, imms)
        }
        "bfxil" => {
            // BFXIL Rd, Rn, #lsb, #width → BFM Rd, Rn, #lsb, #(lsb+width-1)
            let lsb = imm1;
            let width = imm2;
            (0b01u32, lsb, lsb + width - 1)
        }
        "ubfx" => {
            // UBFX Rd, Rn, #lsb, #width → UBFM Rd, Rn, #lsb, #(lsb+width-1)
            let lsb = imm1;
            let width = imm2;
            (0b10u32, lsb, lsb + width - 1)
        }
        "sbfx" => {
            // SBFX Rd, Rn, #lsb, #width → SBFM Rd, Rn, #lsb, #(lsb+width-1)
            let lsb = imm1;
            let width = imm2;
            (0b00u32, lsb, lsb + width - 1)
        }
        "ubfiz" => {
            // UBFIZ Rd, Rn, #lsb, #width → UBFM Rd, Rn, #(-lsb mod bits), #(width-1)
            let lsb = imm1;
            let width = imm2;
            let immr = (bits.wrapping_sub(lsb)) & (bits - 1);
            let imms = width.wrapping_sub(1);
            (0b10u32, immr, imms)
        }
        "sbfiz" => {
            // SBFIZ Rd, Rn, #lsb, #width → SBFM Rd, Rn, #(-lsb mod bits), #(width-1)
            let lsb = imm1;
            let width = imm2;
            let immr = (bits.wrapping_sub(lsb)) & (bits - 1);
            let imms = width.wrapping_sub(1);
            (0b00u32, immr, imms)
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "unknown bitfield instruction",
                instr.span,
            ))
        }
    };

    // sf|opc|100110|N|immr|imms|Rn|Rd
    let word = (sf << 31)
        | (opc << 29)
        | (0b100110 << 23)
        | (n << 22)
        | ((immr & 0x3F) << 16)
        | ((imms & 0x3F) << 10)
        | ((rn.a64_reg_num() as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── CCMP / CCMN (Conditional Compare) ────────────────────────────────────

/// Encode CCMP/CCMN: conditional compare (register or immediate).
/// CCMP:  sf|1|1|1010010|Rm|cond|1|0|Rn|o3|o2|nzcv  (register)
/// CCMP:  sf|1|1|1010010|imm5|cond|1|0|Rn|o3|o2|nzcv (immediate)
/// CCMN uses o2=0 (same encoding but op=0 instead of 1 for CCMP).
#[allow(clippy::identity_op)]
fn encode_ccmp(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    // CCMP Rn, Rm_or_imm5, #nzcv, cond
    if ops.len() != 4 {
        return Err(invalid_ops(
            mnemonic,
            "expected Rn, Rm/#imm5, #nzcv, cond_label",
            instr.span,
        ));
    }
    let rn = get_a64_reg(&ops[0], mnemonic, instr.span)?;
    let nzcv = get_imm(&ops[2], mnemonic, instr.span)? as u32;
    if nzcv > 0xF {
        return Err(invalid_ops(mnemonic, "nzcv must be 0-15", instr.span));
    }

    let cond_code_val = match &ops[3] {
        Operand::Label(s) => cond_code(s).ok_or_else(|| {
            invalid_ops(
                mnemonic,
                "expected condition code (eq/ne/cs/cc/mi/pl/vs/vc/hi/ls/ge/lt/gt/le/al)",
                instr.span,
            )
        })?,
        Operand::Immediate(v) => (*v as u32) & 0xF,
        _ => return Err(invalid_ops(mnemonic, "expected condition code", instr.span)),
    };

    let sf = if rn.is_a64_64bit() { 1u32 } else { 0u32 };
    let op = if mnemonic.starts_with("ccmp") {
        1u32
    } else {
        0u32
    }; // CCMP=1, CCMN=0

    match &ops[1] {
        Operand::Register(rm) if rm.is_aarch64() => {
            // Register form: sf|op|1|11010010|Rm|cond|0|0|Rn|0|nzcv
            let word = (sf << 31)
                | (op << 30)
                | (0b1_11010010 << 21)
                | ((rm.a64_reg_num() as u32) << 16)
                | (cond_code_val << 12)
                | (0b00 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (0u32 << 4) // o3=0
                | nzcv;
            emit32(buf, word);
        }
        Operand::Immediate(imm5) => {
            // Immediate form: sf|op|1|11010010|imm5|cond|1|0|Rn|0|nzcv
            let imm = (*imm5 as u32) & 0x1F;
            let word = (sf << 31)
                | (op << 30)
                | (0b1_11010010 << 21)
                | (imm << 16)
                | (cond_code_val << 12)
                | (0b10 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (0u32 << 4) // o3=0
                | nzcv;
            emit32(buf, word);
        }
        _ => {
            return Err(invalid_ops(
                mnemonic,
                "expected register or 5-bit immediate",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ── EXTR (Extract register) ─────────────────────────────────────────────

/// Encode EXTR Rd, Rn, Rm, #lsb
/// sf|00|100111|N0|Rm|imms|Rn|Rd
#[allow(clippy::identity_op)]
fn encode_extr(buf: &mut InstrBytes, ops: &[Operand], instr: &Instruction) -> Result<(), AsmError> {
    if ops.len() != 4 {
        return Err(invalid_ops("extr", "expected Rd, Rn, Rm, #lsb", instr.span));
    }
    let rd = get_a64_reg(&ops[0], "extr", instr.span)?;
    let rn = get_a64_reg(&ops[1], "extr", instr.span)?;
    let rm = get_a64_reg(&ops[2], "extr", instr.span)?;
    let lsb = get_imm(&ops[3], "extr", instr.span)? as u32;

    let sf = if rd.is_a64_64bit() { 1u32 } else { 0u32 };
    let n = sf;
    let max_shift = if sf == 1 { 63 } else { 31 };
    if lsb > max_shift {
        return Err(invalid_ops("extr", "lsb out of range", instr.span));
    }

    // sf|00|100111|N|0|Rm|imms|Rn|Rd
    let word = (sf << 31)
        | (0b00_100111 << 23)
        | (n << 22)
        | (0u32 << 21)
        | ((rm.a64_reg_num() as u32) << 16)
        | ((lsb & 0x3F) << 10)
        | ((rn.a64_reg_num() as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── SVE (Scalable Vector Extension) helpers ──────────────────────────────

/// Extract an SVE Z register from a VectorRegister operand.
/// Returns (register, arrangement).
fn get_sve_zreg(
    op: &Operand,
    mnemonic: &str,
    span: crate::error::Span,
) -> Result<(Register, VectorArrangement), AsmError> {
    match op {
        Operand::VectorRegister(r, arr) if r.is_a64_sve_z() => Ok((*r, *arr)),
        _ => Err(invalid_ops(mnemonic, "expected SVE Z register", span)),
    }
}

/// Extract an SVE predicate from a VectorRegister operand (e.g., p0.b).
fn get_sve_preg_arr(
    op: &Operand,
    mnemonic: &str,
    span: crate::error::Span,
) -> Result<(Register, VectorArrangement), AsmError> {
    match op {
        Operand::VectorRegister(r, arr) if r.is_a64_sve_p() => Ok((*r, *arr)),
        _ => Err(invalid_ops(
            mnemonic,
            "expected SVE predicate register with arrangement",
            span,
        )),
    }
}

/// Extract an SVE predicate with qualifier from an SvePredicate operand (e.g., p0/m).
fn get_sve_pred_qual(
    op: &Operand,
    mnemonic: &str,
    span: crate::error::Span,
) -> Result<(Register, SvePredQual), AsmError> {
    match op {
        Operand::SvePredicate(r, q) if r.is_a64_sve_p() => Ok((*r, *q)),
        _ => Err(invalid_ops(
            mnemonic,
            "expected SVE predicate with /m or /z",
            span,
        )),
    }
}

/// SVE element size encoding (2 bits): B=0, H=1, S=2, D=3.
///
/// Returns an error for non-SVE arrangements.
fn sve_size(arr: VectorArrangement, span: crate::error::Span) -> Result<u32, AsmError> {
    arr.sve_size().ok_or_else(|| AsmError::InvalidOperands {
        detail: String::from("non-SVE vector arrangement in SVE instruction"),
        span,
    })
}

/// Dispatch SVE instructions whose mnemonics overlap with scalar/NEON ops.
///
/// Returns `Ok(true)` if the instruction was encoded as SVE, `Ok(false)` to
/// fall through to the scalar match.
fn encode_sve_dispatch(
    buf: &mut InstrBytes,
    mnemonic: &str,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<bool, AsmError> {
    match mnemonic {
        // ── Unpredicated integer ADD / SUB ────────────────────
        "add" => {
            // 3-operand Z,Z,Z: add zd.T, zn.T, zm.T  (unpredicated)
            // 3-operand Z,Z,#imm: add zdn.T, zdn.T, #imm  (immediate)
            // 4-operand: add zd.T, pg/m, zdn.T, zm.T  (predicated)
            if ops.len() == 3 {
                if matches!(&ops[2], Operand::Immediate(_)) {
                    // SVE ADD (immediate): 0x2520C000 | (size<<22) | (sh<<13) | (imm8<<5) | Zdn
                    let (zdn, arr) = get_sve_zreg(&ops[0], mnemonic, instr.span)?;
                    let imm = get_imm(&ops[2], mnemonic, instr.span)?;
                    let size = sve_size(arr, instr.span)?;
                    let (sh, imm8) = if (0..=255).contains(&imm) {
                        (0u32, imm as u32)
                    } else if imm > 255 && imm <= 0xFF00 && (imm & 0xFF) == 0 {
                        (1u32, (imm >> 8) as u32)
                    } else {
                        return Err(invalid_ops(
                            mnemonic,
                            "SVE immediate out of range",
                            instr.span,
                        ));
                    };
                    let word = 0x2520C000
                        | (size << 22)
                        | (sh << 13)
                        | (imm8 << 5)
                        | (zdn.a64_reg_num() as u32);
                    emit32(buf, word);
                    return Ok(true);
                }
                encode_sve_arith_unpred(buf, 0x04200000, ops, instr)?;
                return Ok(true);
            }
            if ops.len() == 4 {
                encode_sve_arith_pred(buf, 0x04000000, ops, instr)?;
                return Ok(true);
            }
            Ok(false)
        }
        "sub" => {
            if ops.len() == 4 {
                encode_sve_arith_pred(buf, 0x04010000, ops, instr)?;
                return Ok(true);
            }
            // SVE has no unpredicated SUB — only predicated
            Ok(false)
        }
        "mul" => {
            if ops.len() == 4 {
                encode_sve_arith_pred(buf, 0x04100000, ops, instr)?;
                return Ok(true);
            }
            Ok(false)
        }
        // ── Unpredicated bitwise logical ──────────────────────
        "and" => {
            if ops.len() == 3 {
                let (_, arr) = get_sve_zreg(&ops[0], mnemonic, instr.span)?;
                if arr.sve_size().is_some() {
                    encode_sve_logical_unpred(buf, 0x04203000, ops, instr)?;
                    return Ok(true);
                }
            }
            if ops.len() == 4 {
                encode_sve_logical_pred(buf, 0x041A0000, ops, instr)?;
                return Ok(true);
            }
            Ok(false)
        }
        "orr" => {
            if ops.len() == 3 {
                let (_, arr) = get_sve_zreg(&ops[0], mnemonic, instr.span)?;
                if arr.sve_size().is_some() {
                    encode_sve_logical_unpred(buf, 0x04603000, ops, instr)?;
                    return Ok(true);
                }
            }
            if ops.len() == 4 {
                encode_sve_logical_pred(buf, 0x04180000, ops, instr)?;
                return Ok(true);
            }
            Ok(false)
        }
        "eor" => {
            if ops.len() == 3 {
                let (_, arr) = get_sve_zreg(&ops[0], mnemonic, instr.span)?;
                if arr.sve_size().is_some() {
                    encode_sve_logical_unpred(buf, 0x04A03000, ops, instr)?;
                    return Ok(true);
                }
            }
            if ops.len() == 4 {
                encode_sve_logical_pred(buf, 0x04190000, ops, instr)?;
                return Ok(true);
            }
            Ok(false)
        }
        // ── SVE-unique mnemonics with vector first operand ────
        "dup" | "ptrue" | "pfalse" | "ld1b" | "ld1h" | "ld1w" | "ld1d" | "st1b" | "st1h"
        | "st1w" | "st1d" => {
            // These are SVE-unique mnemonics; return false to let them
            // be handled by the main match block.
            Ok(false)
        }
        _ => Ok(false),
    }
}

/// Encode unpredicated SVE integer arithmetic: op Zd.T, Zn.T, Zm.T
///
/// Word layout: `base | (size << 22) | (Zm << 16) | (Zn << 5) | Zd`
fn encode_sve_arith_unpred(
    buf: &mut InstrBytes,
    base: u32,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    let mnemonic = instr.mnemonic.as_str();
    let (rd, arr) = get_sve_zreg(&ops[0], mnemonic, instr.span)?;
    let (rn, _) = get_sve_zreg(&ops[1], mnemonic, instr.span)?;
    let (rm, _) = get_sve_zreg(&ops[2], mnemonic, instr.span)?;
    let size = sve_size(arr, instr.span)?;
    let word = base
        | (size << 22)
        | ((rm.a64_reg_num() as u32) << 16)
        | ((rn.a64_reg_num() as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

/// Encode predicated SVE integer arithmetic: op Zdn.T, Pg/M, Zdn.T, Zm.T
///
/// Word layout: `base | (size << 22) | (Pg << 10) | (Zm << 5) | Zdn`
fn encode_sve_arith_pred(
    buf: &mut InstrBytes,
    base: u32,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    let mnemonic = instr.mnemonic.as_str();
    let (zdn, arr) = get_sve_zreg(&ops[0], mnemonic, instr.span)?;
    let (pg, _qual) = get_sve_pred_qual(&ops[1], mnemonic, instr.span)?;
    // ops[2] is the same register as ops[0] (zdn) — validate but skip
    let (_zdn2, _) = get_sve_zreg(&ops[2], mnemonic, instr.span)?;
    let (zm, _) = get_sve_zreg(&ops[3], mnemonic, instr.span)?;
    let size = sve_size(arr, instr.span)?;
    let word = base
        | (size << 22)
        | ((pg.a64_p_num() as u32) << 10)
        | ((zm.a64_reg_num() as u32) << 5)
        | (zdn.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

/// Encode unpredicated SVE bitwise logical: op Zd.D, Zn.D, Zm.D
///
/// Note: for unpredicated logical, size is NOT encoded (operates on full width).
/// Word layout: `base | (Zm << 16) | (Zn << 5) | Zd`
fn encode_sve_logical_unpred(
    buf: &mut InstrBytes,
    base: u32,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    let mnemonic = instr.mnemonic.as_str();
    let (rd, _) = get_sve_zreg(&ops[0], mnemonic, instr.span)?;
    let (rn, _) = get_sve_zreg(&ops[1], mnemonic, instr.span)?;
    let (rm, _) = get_sve_zreg(&ops[2], mnemonic, instr.span)?;
    let word = base
        | ((rm.a64_reg_num() as u32) << 16)
        | ((rn.a64_reg_num() as u32) << 5)
        | (rd.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

/// Encode predicated SVE bitwise logical: op Zdn.T, Pg/M, Zdn.T, Zm.T
///
/// Word layout: `base | (size << 22) | (Pg << 10) | (Zm << 5) | Zdn`
fn encode_sve_logical_pred(
    buf: &mut InstrBytes,
    base: u32,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    let mnemonic = instr.mnemonic.as_str();
    let (zdn, arr) = get_sve_zreg(&ops[0], mnemonic, instr.span)?;
    let (pg, _qual) = get_sve_pred_qual(&ops[1], mnemonic, instr.span)?;
    let (_zdn2, _) = get_sve_zreg(&ops[2], mnemonic, instr.span)?;
    let (zm, _) = get_sve_zreg(&ops[3], mnemonic, instr.span)?;
    let size = sve_size(arr, instr.span)?;
    let word = base
        | (size << 22)
        | ((pg.a64_p_num() as u32) << 10)
        | ((zm.a64_reg_num() as u32) << 5)
        | (zdn.a64_reg_num() as u32);
    emit32(buf, word);
    Ok(())
}

// ── Public entry point ───────────────────────────────────────────────────

/// Encode an AArch64 (ARM64) instruction.
pub fn encode_aarch64(instr: &Instruction) -> Result<EncodedInstr, AsmError> {
    let mut buf = InstrBytes::new();
    let mut reloc: Option<Relocation> = None;
    let mut relax_info: Option<RelaxInfo> = None;

    let mnemonic = instr.mnemonic.as_str();
    let ops = &instr.operands;

    // Check for B.cond (mnemonic like "b.eq", "b.ne", etc.)
    if let Some(cond_name) = mnemonic.strip_prefix("b.") {
        encode_bcond(&mut buf, cond_name, ops, instr, &mut reloc, &mut relax_info)?;
        return Ok(EncodedInstr {
            bytes: buf,
            relocation: reloc,
            relax: relax_info,
        });
    }

    // ── NEON / AdvSIMD early dispatch ─────────────────────────
    // Mnemonics that share names with scalar ops (add, sub, mul, etc.)
    // are routed to NEON when the first operand is a vector register.
    let has_vec_operand = ops
        .first()
        .is_some_and(|o| matches!(o, Operand::VectorRegister(..)));
    if has_vec_operand {
        // ── SVE early dispatch ────────────────────────────────
        // SVE Z/P registers take priority over NEON when first operand is SVE.
        let is_sve = ops.first().is_some_and(|o| match o {
            Operand::VectorRegister(r, _) => r.is_a64_sve_z() || r.is_a64_sve_p(),
            _ => false,
        });
        if is_sve {
            let sve_handled = encode_sve_dispatch(&mut buf, mnemonic, ops, instr)?;
            if sve_handled {
                return Ok(EncodedInstr {
                    bytes: buf,
                    relocation: reloc,
                    relax: relax_info,
                });
            }
            // SVE-unique mnemonics fall through to main match (skip NEON)
        } else {
            let neon_handled = encode_neon_dispatch(&mut buf, mnemonic, ops, instr)?;
            if neon_handled {
                return Ok(EncodedInstr {
                    bytes: buf,
                    relocation: reloc,
                    relax: relax_info,
                });
            }
        }
    }

    match mnemonic {
        // ── NOP / Hint ───────────────────────────────────────
        "nop" => emit32(&mut buf, 0xD503_201F),
        "wfi" => emit32(&mut buf, 0xD503_207F),
        "wfe" => emit32(&mut buf, 0xD503_205F),
        "sev" => emit32(&mut buf, 0xD503_209F),
        "sevl" => emit32(&mut buf, 0xD503_20BF),
        "yield" => emit32(&mut buf, 0xD503_203F),

        // ── System ───────────────────────────────────────────
        "svc" => encode_svc(&mut buf, ops, instr)?,
        "brk" => encode_brk(&mut buf, ops, instr)?,
        "hlt" => encode_hlt(&mut buf, ops, instr)?,
        "mrs" | "msr" => encode_mrs_msr(&mut buf, mnemonic, ops, instr)?,
        "dmb" | "dsb" | "isb" => encode_barrier(&mut buf, mnemonic, ops, instr)?,

        // ── Branch ───────────────────────────────────────────
        "b" | "bl" => encode_branch(&mut buf, mnemonic, ops, instr, &mut reloc)?,
        "br" | "blr" | "ret" => encode_br_blr_ret(&mut buf, mnemonic, ops, instr)?,
        "cbz" | "cbnz" => encode_cbz(&mut buf, mnemonic, ops, instr, &mut reloc, &mut relax_info)?,
        "tbz" | "tbnz" => encode_tbz(&mut buf, mnemonic, ops, instr, &mut reloc, &mut relax_info)?,

        // ── Move ─────────────────────────────────────────────
        "mov" => encode_mov(&mut buf, ops, instr)?,
        "movz" | "movn" | "movk" => encode_movz_movn_movk(&mut buf, mnemonic, ops, instr)?,
        "mvn" => encode_mvn(&mut buf, ops, instr)?,

        // ── Add / Sub ────────────────────────────────────────
        "add" | "adds" | "sub" | "subs" => {
            // Route to immediate or register form
            if ops.len() >= 3 {
                match &ops[2] {
                    Operand::Immediate(_) => encode_addsub_imm(&mut buf, mnemonic, ops, instr)?,
                    Operand::Register(_) => encode_addsub_reg(&mut buf, mnemonic, ops, instr)?,
                    _ => {
                        return Err(invalid_ops(
                            mnemonic,
                            "expected register or immediate",
                            instr.span,
                        ))
                    }
                }
            } else {
                return Err(invalid_ops(mnemonic, "expected 3 operands", instr.span));
            }
        }
        "neg" | "negs" => {
            // NEG Xd, Xm → SUB Xd, XZR, Xm
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected 2 operands", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let rm = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let sf_bit = sf(rd);
            let s = if mnemonic == "negs" { 1u32 } else { 0 };
            let word = (sf_bit << 31) | (1 << 30) | (s << 29) | (0b01011 << 24)
                | ((rm.a64_reg_num() as u32) << 16)
                | (0b11111 << 5) // Rn = XZR
                | (rd.a64_reg_num() as u32);
            emit32(&mut buf, word);
        }

        // ── Multiply / Divide ────────────────────────────────
        "mul" | "mneg" | "madd" | "msub" | "sdiv" | "udiv" | "smull" | "umull" | "smaddl"
        | "umaddl" | "smsubl" | "umsubl" | "smnegl" | "umnegl" | "smulh" | "umulh" => {
            encode_mul_div(&mut buf, mnemonic, ops, instr)?
        }

        // ── Compare ──────────────────────────────────────────
        "cmp" | "cmn" => {
            if ops.len() == 2 {
                match &ops[1] {
                    Operand::Immediate(_) => encode_addsub_imm(&mut buf, mnemonic, ops, instr)?,
                    Operand::Register(rm) if rm.is_aarch64() => {
                        // CMP Rn, Rm → SUBS XZR, Rn, Rm
                        let rn = get_a64_reg(&ops[0], mnemonic, instr.span)?;
                        let sf_bit = sf(rn);
                        let op = if mnemonic == "cmp" { 1u32 } else { 0 };
                        let word = (sf_bit << 31)
                            | (op << 30)
                            | (1 << 29)
                            | (0b01011 << 24)
                            | ((rm.a64_reg_num() as u32) << 16)
                            | ((rn.a64_reg_num() as u32) << 5)
                            | 0b11111; // Rd = XZR
                        emit32(&mut buf, word);
                    }
                    _ => {
                        return Err(invalid_ops(
                            mnemonic,
                            "expected register or immediate",
                            instr.span,
                        ))
                    }
                }
            } else {
                return Err(invalid_ops(mnemonic, "expected 2 operands", instr.span));
            }
        }
        "tst" => {
            // TST Rn, Rm (register) or TST Rn, #imm (logical immediate)
            if ops.len() >= 2 {
                match &ops[1] {
                    Operand::Immediate(_) => encode_logical_imm(&mut buf, "tst", ops, instr)?,
                    _ => encode_logical_reg(&mut buf, "tst", ops, instr)?,
                }
            } else {
                return Err(invalid_ops("tst", "expected 2 operands", instr.span));
            }
        }

        // ── Logical ──────────────────────────────────────────
        "and" | "ands" | "orr" | "eor" => {
            // Route to immediate or register form
            if ops.len() >= 3 {
                match &ops[2] {
                    Operand::Immediate(_) => encode_logical_imm(&mut buf, mnemonic, ops, instr)?,
                    _ => encode_logical_reg(&mut buf, mnemonic, ops, instr)?,
                }
            } else if ops.len() == 2 {
                // 2-operand TST form handled above
                return Err(invalid_ops(mnemonic, "expected 3 operands", instr.span));
            } else {
                return Err(invalid_ops(mnemonic, "expected 3 operands", instr.span));
            }
        }
        "orn" | "eon" | "bic" | "bics" => encode_logical_reg(&mut buf, mnemonic, ops, instr)?,

        // ── Shift ────────────────────────────────────────────
        "lsl" | "lsr" | "asr" | "ror" | "lslv" | "lsrv" | "asrv" | "rorv" => {
            encode_shift(&mut buf, mnemonic, ops, instr)?
        }

        // ── Bit manipulation ─────────────────────────────────
        "clz" | "cls" | "rbit" | "rev" | "rev16" | "rev32" => {
            encode_bitmanip(&mut buf, mnemonic, ops, instr)?
        }

        // ── Extend ───────────────────────────────────────────
        "uxtb" | "uxth" | "sxtb" | "sxth" | "sxtw" => {
            encode_extend(&mut buf, mnemonic, ops, instr)?
        }

        // ── Load / Store ─────────────────────────────────────
        "ldr" | "str" | "ldrb" | "strb" | "ldrh" | "strh" | "ldrsb" | "ldrsh" | "ldrsw" => {
            // Check for pre-index or post-index addressing modes
            if ops.len() >= 2 {
                let is_preindex =
                    matches!(&ops[1], Operand::Memory(m) if m.addr_mode == AddrMode::PreIndex);
                let is_postindex = ops.len() >= 3
                    && matches!(&ops[1], Operand::Memory(m) if m.addr_mode == AddrMode::Offset)
                    && matches!(&ops[2], Operand::Immediate(_));
                if is_preindex || is_postindex {
                    let rt = get_a64_reg(&ops[0], mnemonic, instr.span)?;
                    let mem = match &ops[1] {
                        Operand::Memory(m) => m,
                        _ => {
                            return Err(invalid_ops(
                                mnemonic,
                                "expected memory operand for indexed addressing",
                                instr.span,
                            ))
                        }
                    };
                    let post_imm = if is_postindex {
                        Some(get_imm(&ops[2], mnemonic, instr.span)? as i64)
                    } else {
                        None
                    };
                    encode_ldr_str_idx(&mut buf, mnemonic, rt, mem, post_imm, instr)?;
                } else {
                    encode_ldr_str(&mut buf, mnemonic, ops, instr, &mut reloc)?;
                }
            } else {
                encode_ldr_str(&mut buf, mnemonic, ops, instr, &mut reloc)?;
            }
        }
        "stp" | "ldp" => {
            // Check for pre-index or post-index pair addressing
            let is_preindex = ops.len() >= 3
                && matches!(&ops[2], Operand::Memory(m) if m.addr_mode == AddrMode::PreIndex);
            let is_postindex = ops.len() >= 4
                && matches!(&ops[3], Operand::Immediate(_))
                && matches!(&ops[2], Operand::Memory(m) if m.addr_mode == AddrMode::Offset);
            if is_preindex || is_postindex {
                encode_stp_ldp_idx(&mut buf, mnemonic, ops, instr)?;
            } else {
                encode_stp_ldp(&mut buf, mnemonic, ops, instr)?;
            }
        }

        // ── Conditional select ───────────────────────────────
        "csel" | "csinc" | "csinv" | "csneg" => encode_csel(&mut buf, mnemonic, ops, instr)?,

        // ── Conditional aliases ──────────────────────────────
        "cset" | "csetm" | "cinc" | "cneg" => encode_cond_alias(&mut buf, mnemonic, ops, instr)?,

        // ── ADR / ADRP ──────────────────────────────────────
        "adr" | "adrp" => encode_adr(&mut buf, mnemonic, ops, instr, &mut reloc, &mut relax_info)?,

        // ── Bitfield ─────────────────────────────────────────
        "bfm" | "ubfm" | "sbfm" | "bfi" | "bfxil" | "ubfx" | "sbfx" | "ubfiz" | "sbfiz" => {
            encode_bitfield(&mut buf, mnemonic, ops, instr)?
        }

        // ── Conditional compare ──────────────────────────────
        "ccmp" | "ccmn" => encode_ccmp(&mut buf, mnemonic, ops, instr)?,

        // ── EXTR ─────────────────────────────────────────────
        "extr" => encode_extr(&mut buf, ops, instr)?,

        // ── Load/Store Exclusive ─────────────────────────────
        "ldxr" | "ldxrb" | "ldxrh" | "ldaxr" | "ldaxrb" | "ldaxrh" | "stxr" | "stxrb" | "stxrh"
        | "stlxr" | "stlxrb" | "stlxrh" => encode_exclusive(&mut buf, mnemonic, ops, instr)?,

        // ── Load-Acquire / Store-Release (non-exclusive) ─────
        "ldar" | "ldarb" | "ldarh" | "stlr" | "stlrb" | "stlrh" => {
            encode_ordered(&mut buf, mnemonic, ops, instr)?
        }

        // ── Atomics (LSE) ────────────────────────────────────
        m if m.starts_with("ldadd")
            || m.starts_with("ldclr")
            || m.starts_with("ldset")
            || m.starts_with("ldeor")
            || m.starts_with("stadd")
            || m.starts_with("stclr")
            || m.starts_with("stset")
            || m.starts_with("steor")
            || m.starts_with("swp")
            || m.starts_with("cas") =>
        {
            encode_atomic(&mut buf, mnemonic, ops, instr)?
        }

        // ── NEON ops whose first operand is a scalar register ──
        "umov" => {
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected 2 operands", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let (rn, arr) = get_a64_vreg(&ops[1], mnemonic, instr.span)?;
            let q: u32 = if arr.element_bits() == 64 { 1 } else { 0 };
            let imm5: u32 = match arr.element_bits() {
                8 => 0b00001,
                16 => 0b00010,
                32 => 0b00100,
                64 => 0b01000,
                _ => return Err(invalid_ops(mnemonic, "invalid arrangement", instr.span)),
            };
            let word = (q << 30)
                | (1u32 << 29)
                | (0b01110000 << 21)
                | (imm5 << 16)
                | (0b00111 << 11)
                | (1 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(&mut buf, word);
        }
        "smov" => {
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected 2 operands", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let (rn, arr) = get_a64_vreg(&ops[1], mnemonic, instr.span)?;
            let q: u32 = if rd.is_a64_64bit() { 1 } else { 0 };
            let imm5: u32 = match arr.element_bits() {
                8 => 0b00001,
                16 => 0b00010,
                32 => 0b00100,
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "invalid arrangement for smov",
                        instr.span,
                    ))
                }
            };
            let word = (q << 30)
                | (0b001110000 << 21)
                | (imm5 << 16)
                | (0b00101 << 11)
                | (1 << 10)
                | ((rn.a64_reg_num() as u32) << 5)
                | (rd.a64_reg_num() as u32);
            emit32(&mut buf, word);
        }

        // ── SVE predicate operations ─────────────────────────
        "ptrue" => {
            // PTRUE Pd.T{, pattern}
            // Encoding: 0x2518E000 | (size << 22) | (pattern << 5) | Pd
            // Default pattern = 0b11111 (all elements)
            if ops.is_empty() {
                return Err(invalid_ops(
                    mnemonic,
                    "expected predicate register",
                    instr.span,
                ));
            }
            let (pd, arr) = get_sve_preg_arr(&ops[0], mnemonic, instr.span)?;
            let size = sve_size(arr, instr.span)?;
            let pattern: u32 = if ops.len() > 1 {
                get_imm(&ops[1], mnemonic, instr.span)? as u32 & 0x1F
            } else {
                0b11111 // ALL
            };
            let word = 0x2518E000 | (size << 22) | (pattern << 5) | (pd.a64_p_num() as u32);
            emit32(&mut buf, word);
        }
        "pfalse" => {
            // PFALSE Pd.B — fixed encoding, always .B
            // Encoding: 0x2518E400 | Pd
            if ops.is_empty() {
                return Err(invalid_ops(
                    mnemonic,
                    "expected predicate register",
                    instr.span,
                ));
            }
            let (pd, _arr) = get_sve_preg_arr(&ops[0], mnemonic, instr.span)?;
            let word = 0x2518E400 | (pd.a64_p_num() as u32);
            emit32(&mut buf, word);
        }
        "whilelt" => {
            // WHILELT Pd.T, Xn, Xm
            // Encoding: 0x25200400 | (size << 22) | (sf << 12) | (Rm << 16) | (Rn << 5) | Pd
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected 3 operands", instr.span));
            }
            let (pd, arr) = get_sve_preg_arr(&ops[0], mnemonic, instr.span)?;
            let rn = get_a64_reg(&ops[1], mnemonic, instr.span)?;
            let rm = get_a64_reg(&ops[2], mnemonic, instr.span)?;
            let size = sve_size(arr, instr.span)?;
            let sf: u32 = if rn.is_a64_64bit() { 1 } else { 0 };
            let word = 0x25200400
                | (size << 22)
                | ((rm.a64_reg_num() as u32) << 16)
                | (sf << 12)
                | ((rn.a64_reg_num() as u32) << 5)
                | (pd.a64_p_num() as u32);
            emit32(&mut buf, word);
        }

        // ── SVE contiguous load/store ────────────────────────
        "ld1b" | "ld1h" | "ld1w" | "ld1d" => {
            // LD1x {Zt.T}, Pg/Z, [Xn]
            // Base encodings (zero-offset scalar+imm forms):
            //   ld1b: 0xA400A000, ld1h: 0xA4A0A000, ld1w: 0xA540A000, ld1d: 0xA5E0A000
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected 3 operands", instr.span));
            }
            let (zt, _arr) = get_sve_zreg(&ops[0], mnemonic, instr.span)?;
            let (pg, _qual) = get_sve_pred_qual(&ops[1], mnemonic, instr.span)?;
            let base_reg = match &ops[2] {
                Operand::Memory(m) => m.base.ok_or_else(|| {
                    invalid_ops(mnemonic, "expected base register in [Xn]", instr.span)
                })?,
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "expected [Xn] memory operand",
                        instr.span,
                    ))
                }
            };
            let base_enc = match mnemonic {
                "ld1b" => 0xA400A000u32,
                "ld1h" => 0xA4A0A000u32,
                "ld1w" => 0xA540A000u32,
                "ld1d" => 0xA5E0A000u32,
                _ => unreachable!(),
            };
            let word = base_enc
                | ((pg.a64_p_num() as u32) << 10)
                | ((base_reg.a64_reg_num() as u32) << 5)
                | (zt.a64_reg_num() as u32);
            emit32(&mut buf, word);
        }
        "st1b" | "st1h" | "st1w" | "st1d" => {
            // ST1x {Zt.T}, Pg, [Xn]
            // Base encodings (zero-offset scalar+imm forms):
            //   st1b: 0xE400E000, st1h: 0xE4A0E000, st1w: 0xE540E000, st1d: 0xE5E0E000
            if ops.len() != 3 {
                return Err(invalid_ops(mnemonic, "expected 3 operands", instr.span));
            }
            let (zt, _arr) = get_sve_zreg(&ops[0], mnemonic, instr.span)?;
            let (pg, _qual) = get_sve_pred_qual(&ops[1], mnemonic, instr.span)?;
            let base_reg = match &ops[2] {
                Operand::Memory(m) => m.base.ok_or_else(|| {
                    invalid_ops(mnemonic, "expected base register in [Xn]", instr.span)
                })?,
                _ => {
                    return Err(invalid_ops(
                        mnemonic,
                        "expected [Xn] memory operand",
                        instr.span,
                    ))
                }
            };
            let base_enc = match mnemonic {
                "st1b" => 0xE400E000u32,
                "st1h" => 0xE4A0E000u32,
                "st1w" => 0xE540E000u32,
                "st1d" => 0xE5E0E000u32,
                _ => unreachable!(),
            };
            let word = base_enc
                | ((pg.a64_p_num() as u32) << 10)
                | ((base_reg.a64_reg_num() as u32) << 5)
                | (zt.a64_reg_num() as u32);
            emit32(&mut buf, word);
        }

        // ── SVE element count ────────────────────────────────
        "cntb" | "cnth" | "cntw" | "cntd" => {
            // CNTx Xd{, pattern{, MUL #imm}}
            // Encoding: 0x0420E000 | (size << 22) | (pattern << 5) | Rd
            // Size: cntb=0, cnth=1, cntw=2, cntd=3
            if ops.is_empty() {
                return Err(invalid_ops(mnemonic, "expected register", instr.span));
            }
            let rd = get_a64_reg(&ops[0], mnemonic, instr.span)?;
            let size: u32 = match mnemonic {
                "cntb" => 0,
                "cnth" => 1,
                "cntw" => 2,
                "cntd" => 3,
                _ => unreachable!(),
            };
            let pattern: u32 = 0b11111; // ALL
            let word = 0x0420E000 | (size << 22) | (pattern << 5) | (rd.a64_reg_num() as u32);
            emit32(&mut buf, word);
        }

        // ── SVE DUP (broadcast immediate) ────────────────────
        "dup" => {
            // DUP Zd.T, #imm8{, LSL #8}
            // Encoding: 0x2538C000 | (size << 22) | (sh << 13) | (imm8 << 5) | Zd
            if ops.len() != 2 {
                return Err(invalid_ops(mnemonic, "expected 2 operands", instr.span));
            }
            let (zd, arr) = get_sve_zreg(&ops[0], mnemonic, instr.span)?;
            let imm = get_imm(&ops[1], mnemonic, instr.span)?;
            let size = sve_size(arr, instr.span)?;
            // imm8 is a signed 8-bit value (range -128..127)
            let (sh, imm8) = if (-128..=127).contains(&imm) {
                (0u32, (imm as i8 as u8) as u32)
            } else if (-32768..=32512).contains(&imm) && (imm & 0xFF) == 0 {
                (1u32, ((imm >> 8) as i8 as u8) as u32)
            } else {
                return Err(invalid_ops(
                    mnemonic,
                    "immediate out of range for DUP",
                    instr.span,
                ));
            };
            let word =
                0x2538C000 | (size << 22) | (sh << 13) | (imm8 << 5) | (zd.a64_reg_num() as u32);
            emit32(&mut buf, word);
        }

        _ => {
            return Err(AsmError::UnknownMnemonic {
                mnemonic: String::from(mnemonic),
                arch: crate::error::ArchName::Aarch64,
                span: instr.span,
            });
        }
    }

    Ok(EncodedInstr {
        bytes: buf,
        relocation: reloc,
        relax: relax_info,
    })
}

// ── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create an instruction with given mnemonic and operands.
    fn make_instr(mnemonic: &str, operands: Vec<Operand>) -> Instruction {
        Instruction {
            mnemonic: Mnemonic::from(mnemonic),
            operands: OperandList::from(operands),
            span: crate::error::Span::default(),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
        }
    }

    /// Helper: encode and return the 4-byte LE word.
    fn enc(mnemonic: &str, ops: Vec<Operand>) -> u32 {
        let instr = make_instr(mnemonic, ops);
        let result = encode_aarch64(&instr).unwrap();
        assert_eq!(result.bytes.len(), 4);
        u32::from_le_bytes(result.bytes[..4].try_into().unwrap())
    }

    fn xreg(n: u8) -> Operand {
        Operand::Register(match n {
            0 => Register::A64X0,
            1 => Register::A64X1,
            2 => Register::A64X2,
            3 => Register::A64X3,
            4 => Register::A64X4,
            8 => Register::A64X8,
            29 => Register::A64X29,
            30 => Register::A64X30,
            _ => Register::A64X0,
        })
    }

    fn wreg(n: u8) -> Operand {
        Operand::Register(match n {
            0 => Register::A64W0,
            1 => Register::A64W1,
            2 => Register::A64W2,
            3 => Register::A64W3,
            _ => Register::A64W0,
        })
    }

    fn imm(v: i128) -> Operand {
        Operand::Immediate(v)
    }

    fn label(s: &str) -> Operand {
        Operand::Label(String::from(s))
    }

    // ── MUL / DIV ────────────────────────────────────────────

    #[test]
    fn mul_x0_x1_x2() {
        // MUL X0, X1, X2 → MADD X0, X1, X2, XZR
        // sf=1|00|11011|000|Rm=00010|0|Ra=11111|Rn=00001|Rd=00000
        let w = enc("mul", vec![xreg(0), xreg(1), xreg(2)]);
        assert_eq!(w, 0x9B02_7C20);
    }

    #[test]
    fn mul_w0_w1_w2() {
        let w = enc("mul", vec![wreg(0), wreg(1), wreg(2)]);
        assert_eq!(w, 0x1B02_7C20);
    }

    #[test]
    fn sdiv_x0_x1_x2() {
        // SDIV X0, X1, X2: sf=1|00|11010110|Rm=00010|000011|Rn=00001|Rd=00000
        let w = enc("sdiv", vec![xreg(0), xreg(1), xreg(2)]);
        assert_eq!(w, 0x9AC2_0C20);
    }

    #[test]
    fn udiv_x0_x1_x2() {
        // UDIV: same but opcode=000010
        let w = enc("udiv", vec![xreg(0), xreg(1), xreg(2)]);
        assert_eq!(w, 0x9AC2_0820);
    }

    #[test]
    fn madd_x0_x1_x2_x3() {
        let w = enc("madd", vec![xreg(0), xreg(1), xreg(2), xreg(3)]);
        // sf=1|00|11011|000|Rm=00010|0|Ra=00011|Rn=00001|Rd=00000
        assert_eq!(w, 0x9B02_0C20);
    }

    #[test]
    fn smaddl_x0_w1_w2_x3() {
        // SMADDL Xd,Wn,Wm,Xa: 1|00|11011|001|Rm|0|Ra|Rn|Rd
        let w = enc("smaddl", vec![xreg(0), wreg(1), wreg(2), xreg(3)]);
        assert_eq!(w, 0x9B22_0C20);
    }

    #[test]
    fn umaddl_x0_w1_w2_x3() {
        let w = enc("umaddl", vec![xreg(0), wreg(1), wreg(2), xreg(3)]);
        assert_eq!(w, 0x9BA2_0C20);
    }

    #[test]
    fn smsubl_x0_w1_w2_x3() {
        // o0=1 for SUB
        let w = enc("smsubl", vec![xreg(0), wreg(1), wreg(2), xreg(3)]);
        assert_eq!(w, 0x9B22_8C20);
    }

    #[test]
    fn umsubl_x0_w1_w2_x3() {
        let w = enc("umsubl", vec![xreg(0), wreg(1), wreg(2), xreg(3)]);
        assert_eq!(w, 0x9BA2_8C20);
    }

    #[test]
    fn smnegl_x0_w1_w2() {
        // SMNEGL = SMSUBL with Ra=XZR
        let w = enc("smnegl", vec![xreg(0), wreg(1), wreg(2)]);
        assert_eq!(w, 0x9B22_FC20);
    }

    #[test]
    fn umnegl_x0_w1_w2() {
        let w = enc("umnegl", vec![xreg(0), wreg(1), wreg(2)]);
        assert_eq!(w, 0x9BA2_FC20);
    }

    #[test]
    fn smulh_x0_x1_x2() {
        // SMULH: op31=010, U=0
        let w = enc("smulh", vec![xreg(0), xreg(1), xreg(2)]);
        assert_eq!(w, 0x9B42_7C20);
    }

    #[test]
    fn umulh_x0_x1_x2() {
        // UMULH: op31=110, U=1
        let w = enc("umulh", vec![xreg(0), xreg(1), xreg(2)]);
        assert_eq!(w, 0x9BC2_7C20);
    }

    // ── MVN ──────────────────────────────────────────────────

    #[test]
    fn mvn_x0_x1() {
        // MVN X0, X1 → ORN X0, XZR, X1
        // sf=1|01|01010|00|1|Rm=00001|000000|Rn=11111|Rd=00000
        let w = enc("mvn", vec![xreg(0), xreg(1)]);
        assert_eq!(w, 0xAA21_03E0);
    }

    // ── CSET / CSETM / CINC / CNEG ──────────────────────────

    #[test]
    fn cset_x0_eq() {
        // CSET X0, EQ → CSINC X0, XZR, XZR, NE
        // sf=1|0|0|11010100|Rm=11111|cond=0001|0|op2=1|Rn=11111|Rd=00000
        let w = enc("cset", vec![xreg(0), label("eq")]);
        assert_eq!(w, 0x9A9F_17E0);
    }

    #[test]
    fn csetm_x0_ne() {
        // CSETM X0, NE → CSINV X0, XZR, XZR, EQ
        // sf=1|1|0|11010100|Rm=11111|cond=0000|0|op2=0|Rn=11111|Rd=00000
        let w = enc("csetm", vec![xreg(0), label("ne")]);
        assert_eq!(w, 0xDA9F_03E0);
    }

    #[test]
    fn cinc_x0_x1_eq() {
        // CINC X0, X1, EQ → CSINC X0, X1, X1, NE
        let w = enc("cinc", vec![xreg(0), xreg(1), label("eq")]);
        assert_eq!(w, 0x9A81_1420);
    }

    // ── Bit manipulation ─────────────────────────────────────

    #[test]
    fn clz_x0_x1() {
        // CLZ X0, X1: sf=1|1|0|11010110|00000|000100|Rn=00001|Rd=00000
        let w = enc("clz", vec![xreg(0), xreg(1)]);
        assert_eq!(w, 0xDAC0_1020);
    }

    #[test]
    fn rbit_x0_x1() {
        let w = enc("rbit", vec![xreg(0), xreg(1)]);
        assert_eq!(w, 0xDAC0_0020);
    }

    #[test]
    fn rev_x0_x1() {
        // REV X0, X1 (64-bit): opcode=000011
        let w = enc("rev", vec![xreg(0), xreg(1)]);
        assert_eq!(w, 0xDAC0_0C20);
    }

    #[test]
    fn rev_w0_w1() {
        // REV W0, W1 (32-bit): opcode=000010
        let w = enc("rev", vec![wreg(0), wreg(1)]);
        assert_eq!(w, 0x5AC0_0820);
    }

    #[test]
    fn rev16_x0_x1() {
        let w = enc("rev16", vec![xreg(0), xreg(1)]);
        assert_eq!(w, 0xDAC0_0420);
    }

    #[test]
    fn cls_x0_x1() {
        let w = enc("cls", vec![xreg(0), xreg(1)]);
        assert_eq!(w, 0xDAC0_1420);
    }

    // ── Extend ───────────────────────────────────────────────

    #[test]
    fn uxtb_w0_w1() {
        // UXTB W0, W1 → UBFM W0, W1, #0, #7
        // sf=0|10|100110|0|immr=000000|imms=000111|Rn=00001|Rd=00000
        let w = enc("uxtb", vec![wreg(0), wreg(1)]);
        assert_eq!(w, 0x5300_1C20);
    }

    #[test]
    fn sxtb_x0_w1() {
        // SXTB X0, W1 → SBFM X0, X1, #0, #7
        // sf=1|00|100110|1|000000|000111|Rn=00001|Rd=00000
        let w = enc("sxtb", vec![xreg(0), wreg(1)]);
        assert_eq!(w, 0x9340_1C20);
    }

    #[test]
    fn sxtw_x0_w1() {
        // SXTW X0, W1 → SBFM X0, X1, #0, #31
        // sf=1|00|100110|1|000000|011111|Rn=00001|Rd=00000
        let w = enc("sxtw", vec![xreg(0), wreg(1)]);
        assert_eq!(w, 0x9340_7C20);
    }

    // ── MRS / MSR ────────────────────────────────────────────

    #[test]
    fn mrs_x0_nzcv() {
        // MRS X0, NZCV
        let w = enc("mrs", vec![xreg(0), label("nzcv")]);
        // nzcv = 0b11_011_0100_0010_000 = 0xDA10
        // 0xD530_0000 | (0xDA10 << 5) | 0 = 0xD53B_4200
        assert_eq!(w, 0xD53B_4200);
    }

    #[test]
    fn msr_nzcv_x0() {
        let w = enc("msr", vec![label("nzcv"), xreg(0)]);
        assert_eq!(w, 0xD51B_4200);
    }

    // ── Barriers ─────────────────────────────────────────────

    #[test]
    fn dmb_sy() {
        let w = enc("dmb", vec![label("sy")]);
        assert_eq!(w, 0xD503_3FBF);
    }

    #[test]
    fn dmb_ish() {
        let w = enc("dmb", vec![label("ish")]);
        assert_eq!(w, 0xD503_3BBF);
    }

    #[test]
    fn dsb_sy() {
        let w = enc("dsb", vec![label("sy")]);
        assert_eq!(w, 0xD503_3F9F);
    }

    #[test]
    fn isb_default() {
        // ISB with no operand defaults to SY
        let w = enc("isb", vec![]);
        assert_eq!(w, 0xD503_3FDF);
    }

    // ── Logical Immediate ────────────────────────────────────

    #[test]
    fn and_x0_x1_0xff() {
        // AND X0, X1, #0xFF
        // 0xFF is 8 consecutive 1s → N=1, immr=0, imms=7 (0b000111)
        let w = enc("and", vec![xreg(0), xreg(1), imm(0xFF)]);
        // sf=1|00|100100|1|000000|000111|Rn=00001|Rd=00000
        assert_eq!(w, 0x9240_1C20);
    }

    #[test]
    fn orr_x0_x1_0xf0f0f0f0f0f0f0f0() {
        // 0xF0F0...F0F0 is a repeating 8-bit pattern 0xF0
        let w = enc(
            "orr",
            vec![xreg(0), xreg(1), imm(0xF0F0_F0F0_F0F0_F0F0u64 as i128)],
        );
        assert!(w != 0); // Just verify it encodes
    }

    #[test]
    fn tst_x0_imm() {
        // TST X0, #0xFF → ANDS XZR, X0, #0xFF
        let w = enc("tst", vec![xreg(0), imm(0xFF)]);
        assert_eq!(w, 0xF240_1C1F);
    }

    // ── Bitmask immediate encoder ────────────────────────────

    #[test]
    fn bitmask_0xff() {
        let (n, immr, imms) = encode_bitmask_imm(0xFF, 64).unwrap();
        assert_eq!((n, immr, imms), (1, 0, 7));
    }

    #[test]
    fn bitmask_0xffff() {
        let (n, immr, imms) = encode_bitmask_imm(0xFFFF, 64).unwrap();
        assert_eq!((n, immr, imms), (1, 0, 15));
    }

    #[test]
    fn bitmask_all_ones_rejected() {
        assert!(encode_bitmask_imm(u64::MAX, 64).is_none());
        assert!(encode_bitmask_imm(0, 64).is_none());
    }

    #[test]
    fn bitmask_0x5555() {
        // Alternating bits: 01 repeated → element size = 2
        let result = encode_bitmask_imm(0x5555_5555_5555_5555, 64);
        assert!(result.is_some());
    }

    /// Decode a bitmask immediate from (N, immr, imms) back to the 64-bit value.
    /// This is the inverse of `encode_bitmask_imm_inner`.
    fn decode_bitmask(n: u32, immr: u32, imms: u32) -> u64 {
        // Determine element size from N and imms
        let len = if n == 1 {
            6 // 64-bit element
        } else if imms & 0x20 == 0 {
            5 // 32-bit element
        } else if imms & 0x10 == 0 {
            4 // 16-bit element
        } else if imms & 0x08 == 0 {
            3 // 8-bit element
        } else if imms & 0x04 == 0 {
            2 // 4-bit element
        } else {
            1 // 2-bit element
        };

        let size = 1u32 << len;
        let s = (imms & (size - 1)) + 1; // number of ones
        let r = immr & (size - 1); // rotation amount

        // Build element: s consecutive 1-bits at LSB, then rotate LEFT by r
        // (encoder found r such that value.rotate_right(r) == contiguous_ones)
        let elem_ones: u64 = if s == 64 { u64::MAX } else { (1u64 << s) - 1 };

        let mask: u64 = if size == 64 {
            u64::MAX
        } else {
            (1u64 << size) - 1
        };
        let elem = if r == 0 {
            elem_ones
        } else {
            ((elem_ones << r) | (elem_ones >> (size - r))) & mask
        };

        // Replicate element across 64 bits
        let mut result = elem;
        let mut sz = size as u64;
        while sz < 64 {
            result |= result << sz;
            sz <<= 1;
        }
        result
    }

    /// Generate all valid 64-bit bitmask immediates by enumeration.
    fn enumerate_all_valid_bitmasks() -> alloc::vec::Vec<u64> {
        let mut values = alloc::vec::Vec::new();
        // For each element size: 2, 4, 8, 16, 32, 64
        for log_size in 1..=6u32 {
            let size = 1u64 << log_size;
            // For each number of ones (1 to size-1, excluding 0 and size)
            for ones in 1..size {
                // For each rotation (0 to size-1)
                for rot in 0..size {
                    // Build element: `ones` consecutive 1-bits, rotated right by `rot`
                    let elem_ones = (1u64 << ones) - 1;
                    let elem = if rot == 0 {
                        elem_ones
                    } else {
                        let mask = if size == 64 {
                            u64::MAX
                        } else {
                            (1u64 << size) - 1
                        };
                        ((elem_ones >> rot) | (elem_ones.wrapping_shl((size - rot) as u32))) & mask
                    };
                    // Replicate across 64 bits
                    let mut value = elem;
                    let mut sz = size;
                    while sz < 64 {
                        value |= value << sz;
                        sz <<= 1;
                    }
                    values.push(value);
                }
            }
        }
        values.sort_unstable();
        values.dedup();
        values
    }

    #[test]
    fn bitmask_exhaustive_all_5334_patterns_accepted() {
        let all_valid = enumerate_all_valid_bitmasks();
        // AArch64 has exactly 5,334 valid 64-bit logical immediate values
        assert_eq!(
            all_valid.len(),
            5334,
            "expected 5,334 valid bitmask patterns, got {}",
            all_valid.len()
        );

        let mut failures = alloc::vec::Vec::new();
        for &val in &all_valid {
            if encode_bitmask_imm(val, 64).is_none() {
                failures.push(val);
            }
        }
        assert!(
            failures.is_empty(),
            "encoder rejected {} valid bitmask patterns (first 5: {:?})",
            failures.len(),
            &failures[..core::cmp::min(5, failures.len())]
        );
    }

    #[test]
    fn bitmask_exhaustive_round_trip() {
        // Verify that encoding and decoding round-trips: for every valid bitmask,
        // encode it, decode the (N,immr,imms) back, and verify we get the same value.
        let all_valid = enumerate_all_valid_bitmasks();
        let mut failures = alloc::vec::Vec::new();
        for &val in &all_valid {
            if let Some((n, immr, imms)) = encode_bitmask_imm(val, 64) {
                let decoded = decode_bitmask(n, immr, imms);
                if decoded != val {
                    failures.push((val, n, immr, imms, decoded));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "round-trip failed for {} patterns (first 5: {:?})",
            failures.len(),
            &failures[..core::cmp::min(5, failures.len())]
        );
    }

    #[test]
    fn bitmask_32bit_patterns_accepted() {
        // Test 32-bit bitmask patterns
        let test_cases: &[(u64, bool)] = &[
            (0x0000_0001, true),  // single bit
            (0x0000_00FF, true),  // 8 consecutive bits
            (0x5555_5555, true),  // alternating
            (0xFFFF_0000, true),  // upper half
            (0x0000_FFFF, true),  // lower half
            (0x0000_0000, false), // all zeros
            (0xFFFF_FFFF, false), // all ones
        ];
        for &(val, expected_valid) in test_cases {
            let result = encode_bitmask_imm(val, 32);
            assert_eq!(
                result.is_some(),
                expected_valid,
                "32-bit bitmask 0x{:08X}: expected valid={}, got {:?}",
                val,
                expected_valid,
                result
            );
        }
    }

    // ── TBZ / TBNZ ──────────────────────────────────────────

    #[test]
    fn tbz_immediate() {
        // TBZ X0, #5, #0x10 → b5=0|011011|0|b40=00101|imm14|Rt=00000
        let w = enc("tbz", vec![xreg(0), imm(5), imm(0x10)]);
        let b5 = (w >> 31) & 1;
        let op = (w >> 24) & 1;
        let b40 = (w >> 19) & 0x1F;
        assert_eq!(b5, 0);
        assert_eq!(op, 0);
        assert_eq!(b40, 5);
    }

    #[test]
    fn tbnz_high_bit() {
        // TBNZ X0, #63, #0x10 → b5=1|011011|1|b40=11111|imm14|Rt=00000
        let w = enc("tbnz", vec![xreg(0), imm(63), imm(0x10)]);
        let b5 = (w >> 31) & 1;
        let op = (w >> 24) & 1;
        let b40 = (w >> 19) & 0x1F;
        assert_eq!(b5, 1);
        assert_eq!(op, 1);
        assert_eq!(b40, 31);
    }

    // ── CSEL with condition name ─────────────────────────────

    #[test]
    fn csel_with_cond_name() {
        // CSEL X0, X1, X2, EQ
        let w = enc("csel", vec![xreg(0), xreg(1), xreg(2), label("eq")]);
        let cc_field = (w >> 12) & 0xF;
        assert_eq!(cc_field, 0x0); // EQ = 0
    }

    // ── Helper: build memory operand ─────────────────────────

    fn mem_simple(base: Register) -> Operand {
        Operand::Memory(Box::new(MemoryOperand {
            base: Some(base),
            ..Default::default()
        }))
    }

    fn mem_pre(base: Register, disp: i64) -> Operand {
        Operand::Memory(Box::new(MemoryOperand {
            base: Some(base),
            disp,
            addr_mode: AddrMode::PreIndex,
            ..Default::default()
        }))
    }

    fn mem_offset(base: Register) -> Operand {
        Operand::Memory(Box::new(MemoryOperand {
            base: Some(base),
            addr_mode: AddrMode::Offset,
            ..Default::default()
        }))
    }

    // ── LDR/STR pre/post-index ──────────────────────────────

    #[test]
    fn str_x0_x1_pre_index_neg16() {
        // STR X0, [X1, #-16]!
        // size=11, opc=00, imm9=(-16 & 0x1FF)=0x1F0, idx=11, Rn=1, Rt=0
        let w = enc("str", vec![xreg(0), mem_pre(Register::A64X1, -16)]);
        assert_eq!(w & (0b11 << 30), 0b11 << 30); // size=64-bit
        assert_eq!((w >> 22) & 0b11, 0b00); // opc=store
        assert_eq!((w >> 12) & 0x1FF, 0x1F0); // imm9=-16
        assert_eq!((w >> 10) & 0b11, 0b11); // idx=pre
        assert_eq!((w >> 5) & 0x1F, 1); // Rn=X1
        assert_eq!(w & 0x1F, 0); // Rt=X0
    }

    #[test]
    fn ldr_x0_x1_post_index_16() {
        // LDR X0, [X1], #16
        let w = enc("ldr", vec![xreg(0), mem_offset(Register::A64X1), imm(16)]);
        assert_eq!(w & (0b11 << 30), 0b11 << 30); // size=64-bit
        assert_eq!((w >> 22) & 0b11, 0b01); // opc=load
        assert_eq!((w >> 12) & 0x1FF, 16); // imm9=16
        assert_eq!((w >> 10) & 0b11, 0b01); // idx=post
    }

    #[test]
    fn ldrb_w0_x2_pre_index_1() {
        // LDRB W0, [X2, #1]!
        let w = enc("ldrb", vec![wreg(0), mem_pre(Register::A64X2, 1)]);
        assert_eq!((w >> 30) & 0b11, 0b00); // size=byte
        assert_eq!((w >> 22) & 0b11, 0b01); // opc=load
        assert_eq!((w >> 12) & 0x1FF, 1); // imm9=1
        assert_eq!((w >> 10) & 0b11, 0b11); // idx=pre
    }

    // ── STP/LDP pre/post-index ──────────────────────────────

    #[test]
    fn stp_x29_x30_sp_pre_neg16() {
        // STP X29, X30, [SP, #-16]!
        // opc=10, mode=011 (pre), L=0, imm7=(-16/8)&0x7F=0x7E
        let sp = Register::A64Sp;
        let w = enc("stp", vec![xreg(29), xreg(30), mem_pre(sp, -16)]);
        assert_eq!((w >> 30) & 0b11, 0b10); // opc=64-bit pair
        assert_eq!((w >> 23) & 0b111, 0b011); // mode=pre-index
        assert_eq!((w >> 22) & 1, 0); // L=0 (store)
        assert_eq!((w >> 15) & 0x7F, 0x7E); // imm7=(-2 & 0x7F)
        assert_eq!((w >> 5) & 0x1F, 31); // Rn=SP
    }

    #[test]
    fn ldp_x29_x30_sp_post_16() {
        // LDP X29, X30, [SP], #16
        let sp = Register::A64Sp;
        let w = enc("ldp", vec![xreg(29), xreg(30), mem_offset(sp), imm(16)]);
        assert_eq!((w >> 30) & 0b11, 0b10); // opc=64-bit pair
        assert_eq!((w >> 23) & 0b111, 0b001); // mode=post-index
        assert_eq!((w >> 22) & 1, 1); // L=1 (load)
        assert_eq!((w >> 15) & 0x7F, 2); // imm7=(16/8)=2
    }

    // ── Atomic (LSE) instructions ───────────────────────────

    #[test]
    fn ldadd_x0_x1_x2() {
        // LDADD X0, X1, [X2]
        // size=11, A=0, R=0, Rs=0, o3=0, opc=000, Rn=2, Rt=1
        let w = enc("ldadd", vec![xreg(0), xreg(1), mem_simple(Register::A64X2)]);
        assert_eq!((w >> 30) & 0b11, 0b11); // size=64-bit
        assert_eq!((w >> 23) & 1, 0); // A=0
        assert_eq!((w >> 22) & 1, 0); // R=0
        assert_eq!((w >> 16) & 0x1F, 0); // Rs=X0
        assert_eq!((w >> 12) & 0b111, 0b000); // opc=ldadd
        assert_eq!((w >> 5) & 0x1F, 2); // Rn=X2
        assert_eq!(w & 0x1F, 1); // Rt=X1
    }

    #[test]
    fn ldaddal_x0_x1_x2() {
        // LDADDAL X0, X1, [X2] — acquire+release
        let w = enc(
            "ldaddal",
            vec![xreg(0), xreg(1), mem_simple(Register::A64X2)],
        );
        assert_eq!((w >> 23) & 1, 1); // A=1 (acquire)
        assert_eq!((w >> 22) & 1, 1); // R=1 (release)
        assert_eq!((w >> 12) & 0b111, 0b000); // opc=ldadd
    }

    #[test]
    fn swp_x0_x1_x2() {
        // SWP X0, X1, [X2]
        let w = enc("swp", vec![xreg(0), xreg(1), mem_simple(Register::A64X2)]);
        assert_eq!((w >> 30) & 0b11, 0b11); // size=64-bit
        assert_eq!((w >> 15) & 1, 1); // o3=1 (SWP)
        assert_eq!((w >> 16) & 0x1F, 0); // Rs=X0
        assert_eq!(w & 0x1F, 1); // Rt=X1
    }

    #[test]
    fn cas_x0_x1_x2() {
        // CAS X0, X1, [X2]
        // size=11, A=0, Rs=0, R=0, Rn=2, Rt=1
        let w = enc("cas", vec![xreg(0), xreg(1), mem_simple(Register::A64X2)]);
        assert_eq!((w >> 30) & 0b11, 0b11); // size=64-bit
        assert_eq!((w >> 24) & 0b111111, 0b001000); // CAS opcode
        assert_eq!((w >> 22) & 1, 1); // fixed=1
        assert_eq!((w >> 16) & 0x1F, 0); // Rs=X0
        assert_eq!((w >> 5) & 0x1F, 2); // Rn=X2
        assert_eq!(w & 0x1F, 1); // Rt=X1
    }

    #[test]
    fn staddl_x0_x1() {
        // STADDL X0, [X1] — store-only variant, Rt=XZR, release ordering
        let w = enc("staddl", vec![xreg(0), mem_simple(Register::A64X1)]);
        assert_eq!((w >> 22) & 1, 1); // R=1 (release)
        assert_eq!((w >> 16) & 0x1F, 0); // Rs=X0
        assert_eq!((w >> 12) & 0b111, 0b000); // opc=add
        assert_eq!(w & 0x1F, 0b11111); // Rt=XZR
        assert_eq!((w >> 5) & 0x1F, 1); // Rn=X1
    }

    // ── Load/Store-Exclusive ────────────────────────────────

    #[test]
    fn ldxr_x0_x1() {
        // LDXR X0, [X1]
        // size=11, o2=0, L=1, o1=0, Rs=11111, o0=0, Rt2=11111, Rn=1, Rt=0
        let w = enc("ldxr", vec![xreg(0), mem_simple(Register::A64X1)]);
        assert_eq!((w >> 30) & 0b11, 0b11); // size=64-bit
        assert_eq!((w >> 22) & 1, 1); // L=1 (load)
        assert_eq!((w >> 16) & 0x1F, 0x1F); // Rs=11111
        assert_eq!((w >> 15) & 1, 0); // o0=0 (no acquire)
        assert_eq!((w >> 5) & 0x1F, 1); // Rn=X1
        assert_eq!(w & 0x1F, 0); // Rt=X0
    }

    #[test]
    fn ldaxr_x0_x1() {
        // LDAXR X0, [X1] — with acquire semantics
        let w = enc("ldaxr", vec![xreg(0), mem_simple(Register::A64X1)]);
        assert_eq!((w >> 15) & 1, 1); // o0=1 (acquire)
        assert_eq!((w >> 22) & 1, 1); // L=1 (load)
    }

    #[test]
    fn stxr_w3_x0_x1() {
        // STXR W3, X0, [X1]
        // Rs=W3(=3), Rt=X0, Rn=X1
        let w = enc("stxr", vec![wreg(3), xreg(0), mem_simple(Register::A64X1)]);
        assert_eq!((w >> 30) & 0b11, 0b11); // size=64-bit (from Rt width)
        assert_eq!((w >> 22) & 1, 0); // L=0 (store)
        assert_eq!((w >> 16) & 0x1F, 3); // Rs=W3
        assert_eq!((w >> 15) & 1, 0); // o0=0 (no release)
        assert_eq!(w & 0x1F, 0); // Rt=X0
    }

    #[test]
    fn stlxr_w3_x0_x1() {
        // STLXR W3, X0, [X1] — with release semantics
        let w = enc("stlxr", vec![wreg(3), xreg(0), mem_simple(Register::A64X1)]);
        assert_eq!((w >> 15) & 1, 1); // o0=1 (release)
        assert_eq!((w >> 22) & 1, 0); // L=0 (store)
    }

    // ── Load-Acquire / Store-Release (ordered) ──────────────

    #[test]
    fn ldar_x0_x1() {
        // LDAR X0, [X1]
        // size=11, o2=1, L=1, o1=0, Rs=11111, o0=1, Rt2=11111, Rn=1, Rt=0
        let w = enc("ldar", vec![xreg(0), mem_simple(Register::A64X1)]);
        assert_eq!((w >> 30) & 0b11, 0b11); // size=64-bit
        assert_eq!((w >> 23) & 1, 1); // o2=1 (ordered)
        assert_eq!((w >> 22) & 1, 1); // L=1 (load)
        assert_eq!((w >> 15) & 1, 1); // o0=1 (acquire)
        assert_eq!((w >> 5) & 0x1F, 1); // Rn=X1
        assert_eq!(w & 0x1F, 0); // Rt=X0
    }

    #[test]
    fn stlr_x0_x1() {
        // STLR X0, [X1]
        let w = enc("stlr", vec![xreg(0), mem_simple(Register::A64X1)]);
        assert_eq!((w >> 30) & 0b11, 0b11); // size=64-bit
        assert_eq!((w >> 23) & 1, 1); // o2=1 (ordered)
        assert_eq!((w >> 22) & 1, 0); // L=0 (store)
        assert_eq!((w >> 15) & 1, 1); // o0=1 (release)
    }

    #[test]
    fn ldarb_w0_x1() {
        // LDARB W0, [X1] — byte-size load-acquire
        let w = enc("ldarb", vec![wreg(0), mem_simple(Register::A64X1)]);
        assert_eq!((w >> 30) & 0b11, 0b00); // size=byte
        assert_eq!((w >> 22) & 1, 1); // L=1 (load)
    }

    // ── Bitfield instructions ───────────────────────────────

    #[test]
    fn ubfx_x0_x1_4_8() {
        // UBFX X0, X1, #4, #8  → UBFM X0, X1, #4, #(4+8-1)=#11
        // sf=1, opc=10, N=1, immr=4, imms=11, Rn=1, Rd=0
        let w = enc("ubfx", vec![xreg(0), xreg(1), imm(4), imm(8)]);
        assert_eq!((w >> 31) & 1, 1); // sf=1 (64-bit)
        assert_eq!((w >> 29) & 0b11, 0b10); // opc=UBFM
        assert_eq!((w >> 22) & 1, 1); // N=1
        assert_eq!((w >> 16) & 0x3F, 4); // immr=4
        assert_eq!((w >> 10) & 0x3F, 11); // imms=4+8-1=11
        assert_eq!((w >> 5) & 0x1F, 1); // Rn=X1
        assert_eq!(w & 0x1F, 0); // Rd=X0
    }

    #[test]
    fn bfi_x0_x1_8_4() {
        // BFI X0, X1, #8, #4  → BFM X0, X1, #(-8 mod 64)=#56, #(4-1)=#3
        // sf=1, opc=01, N=1, immr=56, imms=3
        let w = enc("bfi", vec![xreg(0), xreg(1), imm(8), imm(4)]);
        assert_eq!((w >> 29) & 0b11, 0b01); // opc=BFM
        assert_eq!((w >> 16) & 0x3F, 56); // immr=(-8 mod 64)=56
        assert_eq!((w >> 10) & 0x3F, 3); // imms=4-1=3
    }

    #[test]
    fn sbfx_x0_x1_0_16() {
        // SBFX X0, X1, #0, #16 → SBFM X0, X1, #0, #15
        // sf=1, opc=00, immr=0, imms=15
        let w = enc("sbfx", vec![xreg(0), xreg(1), imm(0), imm(16)]);
        assert_eq!((w >> 29) & 0b11, 0b00); // opc=SBFM
        assert_eq!((w >> 16) & 0x3F, 0); // immr=0
        assert_eq!((w >> 10) & 0x3F, 15); // imms=0+16-1=15
    }

    #[test]
    fn ubfiz_w0_w1_4_8() {
        // UBFIZ W0, W1, #4, #8 → UBFM W0, W1, #(-4 mod 32)=#28, #(8-1)=#7
        // sf=0, opc=10, N=0, immr=28, imms=7
        let w = enc("ubfiz", vec![wreg(0), wreg(1), imm(4), imm(8)]);
        assert_eq!((w >> 31) & 1, 0); // sf=0 (32-bit)
        assert_eq!((w >> 29) & 0b11, 0b10); // opc=UBFM
        assert_eq!((w >> 22) & 1, 0); // N=0
        assert_eq!((w >> 16) & 0x3F, 28); // immr=(-4 mod 32)=28
        assert_eq!((w >> 10) & 0x3F, 7); // imms=8-1=7
    }

    // ── CCMP / CCMN ────────────────────────────────────────

    #[test]
    fn ccmp_x0_x1_0_eq() {
        // CCMP X0, X1, #0, EQ  (register form)
        // sf=1, op=1, Rm=1, cond=0(EQ), 00, Rn=0, 0, nzcv=0
        let w = enc("ccmp", vec![xreg(0), xreg(1), imm(0), label("eq")]);
        assert_eq!((w >> 31) & 1, 1); // sf=1
        assert_eq!((w >> 30) & 1, 1); // op=1 (CCMP)
        assert_eq!((w >> 16) & 0x1F, 1); // Rm=X1
        assert_eq!((w >> 12) & 0xF, 0); // cond=EQ=0
        assert_eq!((w >> 10) & 0b11, 0b00); // register form
        assert_eq!((w >> 5) & 0x1F, 0); // Rn=X0
        assert_eq!(w & 0xF, 0); // nzcv=0
    }

    #[test]
    fn ccmp_x0_imm5_4_ne() {
        // CCMP X0, #5, #4, NE  (immediate form)
        let w = enc("ccmp", vec![xreg(0), imm(5), imm(4), label("ne")]);
        assert_eq!((w >> 30) & 1, 1); // op=1 (CCMP)
        assert_eq!((w >> 16) & 0x1F, 5); // imm5=5
        assert_eq!((w >> 12) & 0xF, 1); // cond=NE=1
        assert_eq!((w >> 10) & 0b11, 0b10); // immediate form
        assert_eq!(w & 0xF, 4); // nzcv=4
    }

    #[test]
    fn ccmn_x0_x1_0_ge() {
        // CCMN X0, X1, #0, GE
        let w = enc("ccmn", vec![xreg(0), xreg(1), imm(0), label("ge")]);
        assert_eq!((w >> 30) & 1, 0); // op=0 (CCMN)
        assert_eq!((w >> 12) & 0xF, 0xA); // cond=GE=10
    }

    // ── EXTR ────────────────────────────────────────────────

    #[test]
    fn extr_x0_x1_x2_4() {
        // EXTR X0, X1, X2, #4
        // sf=1, 00|100111, N=1, 0, Rm=2, imms=4, Rn=1, Rd=0
        let w = enc("extr", vec![xreg(0), xreg(1), xreg(2), imm(4)]);
        assert_eq!((w >> 31) & 1, 1); // sf=1
        assert_eq!((w >> 22) & 1, 1); // N=1
        assert_eq!((w >> 16) & 0x1F, 2); // Rm=X2
        assert_eq!((w >> 10) & 0x3F, 4); // imms=4
        assert_eq!((w >> 5) & 0x1F, 1); // Rn=X1
        assert_eq!(w & 0x1F, 0); // Rd=X0
    }

    #[test]
    fn extr_w0_w1_w2_16() {
        // EXTR W0, W1, W2, #16
        // sf=0, N=0
        let w = enc("extr", vec![wreg(0), wreg(1), wreg(2), imm(16)]);
        assert_eq!((w >> 31) & 1, 0); // sf=0 (32-bit)
        assert_eq!((w >> 22) & 1, 0); // N=0
        assert_eq!((w >> 16) & 0x1F, 2); // Rm=W2
        assert_eq!((w >> 10) & 0x3F, 16); // imms=16
    }

    // ── Hint instructions ────────────────────────────────────

    #[test]
    fn hint_wfi() {
        let w = enc("wfi", vec![]);
        assert_eq!(w, 0xD503_207F);
    }

    #[test]
    fn hint_wfe() {
        let w = enc("wfe", vec![]);
        assert_eq!(w, 0xD503_205F);
    }

    #[test]
    fn hint_sev() {
        let w = enc("sev", vec![]);
        assert_eq!(w, 0xD503_209F);
    }

    #[test]
    fn hint_sevl() {
        let w = enc("sevl", vec![]);
        assert_eq!(w, 0xD503_20BF);
    }

    #[test]
    fn hint_yield() {
        let w = enc("yield", vec![]);
        assert_eq!(w, 0xD503_203F);
    }

    // ─── Branch relaxation tests ─────────────────────────────

    #[test]
    fn bcond_label_emits_long_form_with_relax() {
        let instr = make_instr("b.eq", vec![Operand::Label(String::from("target"))]);
        let result = encode_aarch64(&instr).unwrap();

        // Long form: B.NE +8 (inverted) + B target
        assert_eq!(result.bytes.len(), 8);
        let skip_word = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let b_word = u32::from_le_bytes(result.bytes[4..8].try_into().unwrap());

        // B.NE +8: 01010100 | imm19=2 | 0 | cond=1 (NE)
        assert_eq!(skip_word, (0b01010100 << 24) | (2 << 5) | 0x1);
        // B target: 000101 | imm26=0
        assert_eq!(b_word, 0b000101 << 26);

        // Relocation on the B instruction
        let reloc = result.relocation.unwrap();
        assert_eq!(reloc.offset, 4);
        assert_eq!(reloc.kind, RelocKind::Aarch64Jump26);

        // Short form: B.EQ label
        let ri = result.relax.unwrap();
        assert_eq!(ri.short_bytes.len(), 4);
        let short_word = u32::from_le_bytes(ri.short_bytes[0..4].try_into().unwrap());
        assert_eq!(short_word, 0b01010100 << 24); // B.EQ, imm19=0
        let sr = ri.short_relocation.unwrap();
        assert_eq!(sr.kind, RelocKind::Aarch64Branch19);
    }

    #[test]
    fn cbz_label_emits_long_form_with_relax() {
        let instr = make_instr("cbz", vec![xreg(0), Operand::Label(String::from("target"))]);
        let result = encode_aarch64(&instr).unwrap();

        // Long form: CBNZ X0, +8 (inverted) + B target
        assert_eq!(result.bytes.len(), 8);
        let skip_word = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());

        // CBNZ X0, +8: sf=1 | 011010 | op=1 | imm19=2 | Rt=0
        let expected = (1u32 << 31) | (0b011010 << 25) | (1 << 24) | (2 << 5);
        assert_eq!(skip_word, expected);

        let reloc = result.relocation.unwrap();
        assert_eq!(reloc.offset, 4);
        assert_eq!(reloc.kind, RelocKind::Aarch64Jump26);

        let ri = result.relax.unwrap();
        assert_eq!(ri.short_bytes.len(), 4);
        let sr = ri.short_relocation.unwrap();
        assert_eq!(sr.kind, RelocKind::Aarch64Branch19);
    }

    #[test]
    fn cbnz_label_emits_long_form_inverted() {
        let instr = make_instr(
            "cbnz",
            vec![wreg(1), Operand::Label(String::from("target"))],
        );
        let result = encode_aarch64(&instr).unwrap();

        // Long form: CBZ W1, +8 (inverted CBNZ) + B target
        assert_eq!(result.bytes.len(), 8);
        let skip_word = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());

        // CBZ W1, +8: sf=0 | 011010 | op=0 (inverted from 1) | imm19=2 | Rt=1
        let expected = (0b011010u32 << 25) | (2 << 5) | 1;
        assert_eq!(skip_word, expected);
    }

    #[test]
    fn tbz_label_emits_long_form_with_relax() {
        let instr = make_instr(
            "tbz",
            vec![
                xreg(2),
                Operand::Immediate(5),
                Operand::Label(String::from("target")),
            ],
        );
        let result = encode_aarch64(&instr).unwrap();

        // Long form: TBNZ X2, #5, +8 (inverted) + B target
        assert_eq!(result.bytes.len(), 8);
        let skip_word = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());

        // TBNZ: b5=0 (bit 5 of bitno), 011011, op=1 (inverted), b40=5, imm14=2, Rt=2
        let expected = (0b011011u32 << 25) | (1 << 24) | (5 << 19) | (2 << 5) | 2;
        assert_eq!(skip_word, expected);

        let ri = result.relax.unwrap();
        let sr = ri.short_relocation.unwrap();
        assert_eq!(sr.kind, RelocKind::Aarch64Branch14);
    }

    #[test]
    fn tbnz_label_emits_long_form_inverted() {
        let instr = make_instr(
            "tbnz",
            vec![
                xreg(3),
                Operand::Immediate(32),
                Operand::Label(String::from("target")),
            ],
        );
        let result = encode_aarch64(&instr).unwrap();

        // Long form: TBZ X3, #32, +8 (inverted) + B target
        assert_eq!(result.bytes.len(), 8);
        let skip_word = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());

        // TBZ: b5=1 (bit 32), 011011, op=0 (inverted from 1), b40=0, imm14=2, Rt=3
        let expected = (1u32 << 31) | (0b011011 << 25) | (2 << 5) | 3;
        assert_eq!(skip_word, expected);
    }

    #[test]
    fn bcond_immediate_no_relax() {
        // Immediate operand → no relaxation
        let instr = make_instr("b.ne", vec![Operand::Immediate(0x100)]);
        let result = encode_aarch64(&instr).unwrap();
        assert_eq!(result.bytes.len(), 4);
        assert!(result.relax.is_none());
        assert!(result.relocation.is_none());
    }

    // ── ADR relaxation to ADRP+ADD ──────────────────────────

    #[test]
    fn adr_label_has_relaxation() {
        // ADR X0, label → long form (ADRP+ADD, 8 bytes) with short form (ADR, 4 bytes)
        let instr = make_instr(
            "adr",
            vec![
                Operand::Register(Register::A64X0),
                Operand::Label(String::from("target")),
            ],
        );
        let result = encode_aarch64(&instr).unwrap();
        // Long form: 8 bytes (ADRP + ADD)
        assert_eq!(result.bytes.len(), 8);
        // Should have relaxation info
        assert!(
            result.relax.is_some(),
            "ADR with label should have relaxation"
        );
        let relax = result.relax.as_ref().unwrap();
        // Short form: 4 bytes (ADR)
        assert_eq!(relax.short_bytes.len(), 4);
        // Short form relocation uses Aarch64Adr21
        assert!(relax.short_relocation.is_some());
        assert_eq!(
            relax.short_relocation.as_ref().unwrap().kind,
            RelocKind::Aarch64Adr21
        );
        // Long form relocation uses Aarch64AdrpAddPair
        assert!(result.relocation.is_some());
        assert_eq!(
            result.relocation.as_ref().unwrap().kind,
            RelocKind::Aarch64AdrpAddPair
        );
    }

    #[test]
    fn adr_long_form_encoding() {
        // Verify the ADRP+ADD encoding structure
        let instr = make_instr(
            "adr",
            vec![
                Operand::Register(Register::A64X5),
                Operand::Label(String::from("target")),
            ],
        );
        let result = encode_aarch64(&instr).unwrap();
        let adrp_word = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let add_word = u32::from_le_bytes(result.bytes[4..8].try_into().unwrap());

        // ADRP: op=1 | immlo | 10000 | immhi | Rd=5
        assert_eq!((adrp_word >> 31) & 1, 1, "ADRP op bit");
        assert_eq!((adrp_word >> 24) & 0b11111, 0b10000, "ADRP opcode");
        assert_eq!(adrp_word & 0x1F, 5, "ADRP Rd=X5");

        // ADD: sf=1 | 00100010 | sh=0 | imm12=0 | Rn=X5 | Rd=X5
        assert_eq!((add_word >> 31) & 1, 1, "ADD sf bit");
        assert_eq!((add_word >> 23) & 0xFF, 0b00100010, "ADD opcode");
        assert_eq!((add_word >> 5) & 0x1F, 5, "ADD Rn=X5");
        assert_eq!(add_word & 0x1F, 5, "ADD Rd=X5");
    }

    #[test]
    fn adrp_no_relaxation() {
        // ADRP should not have relaxation (already has ±4 GB range)
        let instr = make_instr(
            "adrp",
            vec![
                Operand::Register(Register::A64X0),
                Operand::Label(String::from("target")),
            ],
        );
        let result = encode_aarch64(&instr).unwrap();
        assert_eq!(result.bytes.len(), 4);
        assert!(result.relax.is_none(), "ADRP should not have relaxation");
        assert_eq!(
            result.relocation.as_ref().unwrap().kind,
            RelocKind::Aarch64Adrp
        );
    }

    #[test]
    fn adr_immediate_no_relaxation() {
        // ADR with immediate operand → no relaxation or relocation
        let instr = make_instr(
            "adr",
            vec![
                Operand::Register(Register::A64X0),
                Operand::Immediate(0x100),
            ],
        );
        let result = encode_aarch64(&instr).unwrap();
        assert_eq!(result.bytes.len(), 4);
        assert!(result.relax.is_none());
        assert!(result.relocation.is_none());
    }

    // ── NEON / AdvSIMD tests ─────────────────────────────────

    fn vreg(n: u8, arr: VectorArrangement) -> Operand {
        use Register::*;
        let r = match n {
            0 => A64V0,
            1 => A64V1,
            2 => A64V2,
            3 => A64V3,
            4 => A64V4,
            5 => A64V5,
            6 => A64V6,
            7 => A64V7,
            8 => A64V8,
            9 => A64V9,
            10 => A64V10,
            11 => A64V11,
            12 => A64V12,
            13 => A64V13,
            14 => A64V14,
            15 => A64V15,
            16 => A64V16,
            17 => A64V17,
            18 => A64V18,
            19 => A64V19,
            20 => A64V20,
            21 => A64V21,
            22 => A64V22,
            23 => A64V23,
            24 => A64V24,
            25 => A64V25,
            26 => A64V26,
            27 => A64V27,
            28 => A64V28,
            29 => A64V29,
            30 => A64V30,
            31 => A64V31,
            _ => panic!("invalid vector register"),
        };
        Operand::VectorRegister(r, arr)
    }

    // ── Three-same arithmetic ────────────────────────────────

    #[test]
    fn neon_add_v0_4s_v1_4s_v2_4s() {
        // ADD V0.4S, V1.4S, V2.4S
        // Q=1, U=0, size=10, Rm=2, opcode=10000, Rn=1, Rd=0
        let w = enc(
            "add",
            vec![
                vreg(0, VectorArrangement::S4),
                vreg(1, VectorArrangement::S4),
                vreg(2, VectorArrangement::S4),
            ],
        );
        assert_eq!(w, 0x4EA2_8420);
    }

    #[test]
    fn neon_add_v3_8b_v4_8b_v5_8b() {
        // ADD V3.8B, V4.8B, V5.8B
        // Q=0, U=0, size=00, Rm=5, opcode=10000, Rn=4, Rd=3
        let w = enc(
            "add",
            vec![
                vreg(3, VectorArrangement::B8),
                vreg(4, VectorArrangement::B8),
                vreg(5, VectorArrangement::B8),
            ],
        );
        assert_eq!(w, 0x0E25_8483);
    }

    #[test]
    fn neon_sub_v0_2d_v1_2d_v2_2d() {
        // SUB V0.2D, V1.2D, V2.2D
        // Q=1, U=1, size=11, Rm=2, opcode=10000, Rn=1, Rd=0
        let w = enc(
            "sub",
            vec![
                vreg(0, VectorArrangement::D2),
                vreg(1, VectorArrangement::D2),
                vreg(2, VectorArrangement::D2),
            ],
        );
        assert_eq!(w, 0x6EE2_8420);
    }

    #[test]
    fn neon_mul_v0_4h_v1_4h_v2_4h() {
        // MUL V0.4H, V1.4H, V2.4H
        // Q=0, U=0, size=01, Rm=2, opcode=10011, Rn=1, Rd=0
        let w = enc(
            "mul",
            vec![
                vreg(0, VectorArrangement::H4),
                vreg(1, VectorArrangement::H4),
                vreg(2, VectorArrangement::H4),
            ],
        );
        assert_eq!(w, 0x0E62_9C20);
    }

    // ── Three-same bitwise ───────────────────────────────────

    #[test]
    fn neon_and_v0_16b_v1_16b_v2_16b() {
        // AND V0.16B, V1.16B, V2.16B
        // Q=1, U=0, size=00, Rm=2, opcode=00011, Rn=1, Rd=0
        let w = enc(
            "and",
            vec![
                vreg(0, VectorArrangement::B16),
                vreg(1, VectorArrangement::B16),
                vreg(2, VectorArrangement::B16),
            ],
        );
        assert_eq!(w, 0x4E22_1C20);
    }

    #[test]
    fn neon_orr_v0_16b_v1_16b_v2_16b() {
        // ORR V0.16B, V1.16B, V2.16B
        // Q=1, U=0, size=10, Rm=2, opcode=00011, Rn=1, Rd=0
        let w = enc(
            "orr",
            vec![
                vreg(0, VectorArrangement::B16),
                vreg(1, VectorArrangement::B16),
                vreg(2, VectorArrangement::B16),
            ],
        );
        assert_eq!(w, 0x4EA2_1C20);
    }

    #[test]
    fn neon_eor_v0_16b_v1_16b_v2_16b() {
        // EOR V0.16B, V1.16B, V2.16B
        // Q=1, U=1, size=00, Rm=2, opcode=00011, Rn=1, Rd=0
        let w = enc(
            "eor",
            vec![
                vreg(0, VectorArrangement::B16),
                vreg(1, VectorArrangement::B16),
                vreg(2, VectorArrangement::B16),
            ],
        );
        assert_eq!(w, 0x6E22_1C20);
    }

    #[test]
    fn neon_bic_v0_8b_v1_8b_v2_8b() {
        // BIC V0.8B, V1.8B, V2.8B
        // Q=0, U=0, size=01, Rm=2, opcode=00011, Rn=1, Rd=0
        let w = enc(
            "bic",
            vec![
                vreg(0, VectorArrangement::B8),
                vreg(1, VectorArrangement::B8),
                vreg(2, VectorArrangement::B8),
            ],
        );
        assert_eq!(w, 0x0E62_1C20);
    }

    #[test]
    fn neon_orn_v0_8b_v1_8b_v2_8b() {
        // ORN V0.8B, V1.8B, V2.8B
        // Q=0, U=0, size=11, Rm=2, opcode=00011, Rn=1, Rd=0
        let w = enc(
            "orn",
            vec![
                vreg(0, VectorArrangement::B8),
                vreg(1, VectorArrangement::B8),
                vreg(2, VectorArrangement::B8),
            ],
        );
        assert_eq!(w, 0x0EE2_1C20);
    }

    // ── Three-same compare ───────────────────────────────────

    #[test]
    fn neon_cmeq_v0_4s_v1_4s_v2_4s() {
        // CMEQ V0.4S, V1.4S, V2.4S
        // Q=1, U=1, size=10, Rm=2, opcode=10001, Rn=1, Rd=0
        let w = enc(
            "cmeq",
            vec![
                vreg(0, VectorArrangement::S4),
                vreg(1, VectorArrangement::S4),
                vreg(2, VectorArrangement::S4),
            ],
        );
        assert_eq!(w, 0x6EA2_8C20);
    }

    #[test]
    fn neon_cmgt_v0_4s_v1_4s_v2_4s() {
        // CMGT V0.4S, V1.4S, V2.4S
        // Q=1, U=0, size=10, Rm=2, opcode=00110, Rn=1, Rd=0
        let w = enc(
            "cmgt",
            vec![
                vreg(0, VectorArrangement::S4),
                vreg(1, VectorArrangement::S4),
                vreg(2, VectorArrangement::S4),
            ],
        );
        assert_eq!(w, 0x4EA2_3420);
    }

    #[test]
    fn neon_cmge_v0_8h_v1_8h_v2_8h() {
        // CMGE V0.8H, V1.8H, V2.8H
        // Q=1, U=0, size=01, Rm=2, opcode=00111, Rn=1, Rd=0
        let w = enc(
            "cmge",
            vec![
                vreg(0, VectorArrangement::H8),
                vreg(1, VectorArrangement::H8),
                vreg(2, VectorArrangement::H8),
            ],
        );
        assert_eq!(w, 0x4E62_3C20);
    }

    // ── Three-same misc ──────────────────────────────────────

    #[test]
    fn neon_addp_v0_4s_v1_4s_v2_4s() {
        // ADDP V0.4S, V1.4S, V2.4S
        // Q=1, U=0, size=10, Rm=2, opcode=10111, Rn=1, Rd=0
        let w = enc(
            "addp",
            vec![
                vreg(0, VectorArrangement::S4),
                vreg(1, VectorArrangement::S4),
                vreg(2, VectorArrangement::S4),
            ],
        );
        assert_eq!(w, 0x4EA2_BC20);
    }

    #[test]
    fn neon_smax_v0_4s_v1_4s_v2_4s() {
        // SMAX V0.4S, V1.4S, V2.4S
        // Q=1, U=0, size=10, Rm=2, opcode=01100, Rn=1, Rd=0
        let w = enc(
            "smax",
            vec![
                vreg(0, VectorArrangement::S4),
                vreg(1, VectorArrangement::S4),
                vreg(2, VectorArrangement::S4),
            ],
        );
        assert_eq!(w, 0x4EA2_6420);
    }

    #[test]
    fn neon_umin_v0_16b_v1_16b_v2_16b() {
        // UMIN V0.16B, V1.16B, V2.16B
        // Q=1, U=1, size=00, Rm=2, opcode=01101, Rn=1, Rd=0
        let w = enc(
            "umin",
            vec![
                vreg(0, VectorArrangement::B16),
                vreg(1, VectorArrangement::B16),
                vreg(2, VectorArrangement::B16),
            ],
        );
        assert_eq!(w, 0x6E22_6C20);
    }

    // ── Two-register misc ────────────────────────────────────

    #[test]
    fn neon_neg_v0_4s_v1_4s() {
        // NEG V0.4S, V1.4S
        // Q=1, U=1, size=10, opcode2=01011, Rn=1, Rd=0
        let w = enc(
            "neg",
            vec![
                vreg(0, VectorArrangement::S4),
                vreg(1, VectorArrangement::S4),
            ],
        );
        assert_eq!(w, 0x6EA0_B820);
    }

    #[test]
    fn neon_abs_v0_4s_v1_4s() {
        // ABS V0.4S, V1.4S
        // Q=1, U=0, size=10, opcode2=01011, Rn=1, Rd=0
        let w = enc(
            "abs",
            vec![
                vreg(0, VectorArrangement::S4),
                vreg(1, VectorArrangement::S4),
            ],
        );
        assert_eq!(w, 0x4EA0_B820);
    }

    #[test]
    fn neon_not_v0_16b_v1_16b() {
        // NOT V0.16B, V1.16B
        // Q=1, U=1, size=00, opcode2=00101, Rn=1, Rd=0
        let w = enc(
            "not",
            vec![
                vreg(0, VectorArrangement::B16),
                vreg(1, VectorArrangement::B16),
            ],
        );
        assert_eq!(w, 0x6E20_5820);
    }

    #[test]
    fn neon_cnt_v0_16b_v1_16b() {
        // CNT V0.16B, V1.16B
        // Q=1, U=0, size=00, opcode2=00101, Rn=1, Rd=0
        let w = enc(
            "cnt",
            vec![
                vreg(0, VectorArrangement::B16),
                vreg(1, VectorArrangement::B16),
            ],
        );
        assert_eq!(w, 0x4E20_5820);
    }

    // ── DUP / INS / UMOV / SMOV ──────────────────────────────

    #[test]
    fn neon_dup_v0_4s_w0() {
        // DUP V0.4S, W0
        // Q=1, 001110000, imm5=00100, 00011, 1, Rn=0, Rd=0
        let w = enc("dup", vec![vreg(0, VectorArrangement::S4), wreg(0)]);
        assert_eq!(w, 0x4E04_1C00);
    }

    #[test]
    fn neon_ins_v0_s4_w0() {
        // INS V0.S[0], W0  (encoded as INS V0.4S, W0 in our API)
        // 0|1|001110000|imm5=00100|00111|1|Rn=0|Rd=0
        let w = enc("ins", vec![vreg(0, VectorArrangement::S4), wreg(0)]);
        assert_eq!(w, 0x4E04_3C00);
    }

    #[test]
    fn neon_umov_w0_v0_s4() {
        // UMOV W0, V0.S[0]
        // Q=0, 1, 01110000, imm5=00100, 00111, 1, Rn=0, Rd=0
        let w = enc("umov", vec![wreg(0), vreg(0, VectorArrangement::S4)]);
        assert_eq!(w, 0x2E04_3C00);
    }

    #[test]
    fn neon_smov_x0_v0_s4() {
        // SMOV X0, V0.S[0]
        // Q=1 (64-bit dest), 0, 01110000, imm5=00100, 00101, 1, Rn=0, Rd=0
        let w = enc("smov", vec![xreg(0), vreg(0, VectorArrangement::S4)]);
        assert_eq!(w, 0x4E04_2C00);
    }

    // ── MOV (vector alias) ───────────────────────────────────

    #[test]
    fn neon_mov_v0_16b_v1_16b() {
        // MOV V0.16B, V1.16B → ORR V0, V1, V1
        // Q=1, U=0, size=10, Rm=1, opcode=00011, Rn=1, Rd=0
        let w = enc(
            "mov",
            vec![
                vreg(0, VectorArrangement::B16),
                vreg(1, VectorArrangement::B16),
            ],
        );
        assert_eq!(w, 0x4EA1_1C20);
    }

    // ── LD1 / ST1 ────────────────────────────────────────────

    #[test]
    fn neon_ld1_v0_4s_x0() {
        // LD1 {V0.4S}, [X0]
        // [31]0 [30]Q=1 [29:23]0011000 [22]L=1 [21:16]000000 [15:12]opcode=0111 [11:10]size=10 [9:5]Rn=0 [4:0]Rt=0
        let w = enc(
            "ld1",
            vec![
                vreg(0, VectorArrangement::S4),
                Operand::Memory(Box::new(crate::ir::MemoryOperand {
                    base: Some(Register::A64X0),
                    ..Default::default()
                })),
            ],
        );
        assert_eq!(w, 0x4C40_7800);
    }

    #[test]
    fn neon_st1_v0_4s_x0() {
        // ST1 {V0.4S}, [X0]
        // Q=1, 0011010, L=0, 0, 00000, opcode=0111, size=10, Rn=0, Rt=0
        let w = enc(
            "st1",
            vec![
                vreg(0, VectorArrangement::S4),
                Operand::Memory(Box::new(crate::ir::MemoryOperand {
                    base: Some(Register::A64X0),
                    ..Default::default()
                })),
            ],
        );
        assert_eq!(w, 0x4C00_7800);
    }

    // ── Arrangement variants ─────────────────────────────────

    #[test]
    fn neon_add_v0_8h_v1_8h_v2_8h() {
        // ADD V0.8H, V1.8H, V2.8H  (Q=1, size=01)
        let w = enc(
            "add",
            vec![
                vreg(0, VectorArrangement::H8),
                vreg(1, VectorArrangement::H8),
                vreg(2, VectorArrangement::H8),
            ],
        );
        assert_eq!(w, 0x4E62_8420);
    }

    #[test]
    fn neon_add_v0_16b_v1_16b_v2_16b() {
        // ADD V0.16B, V1.16B, V2.16B  (Q=1, size=00)
        let w = enc(
            "add",
            vec![
                vreg(0, VectorArrangement::B16),
                vreg(1, VectorArrangement::B16),
                vreg(2, VectorArrangement::B16),
            ],
        );
        assert_eq!(w, 0x4E22_8420);
    }

    #[test]
    fn neon_sub_v0_4s_v1_4s_v2_4s() {
        // SUB V0.4S, V1.4S, V2.4S  (Q=1, size=10)
        let w = enc(
            "sub",
            vec![
                vreg(0, VectorArrangement::S4),
                vreg(1, VectorArrangement::S4),
                vreg(2, VectorArrangement::S4),
            ],
        );
        assert_eq!(w, 0x6EA2_8420);
    }

    // ── Scalar ops still work with NEON dispatch ─────────────

    #[test]
    fn scalar_add_still_works() {
        // Scalar ADD X0, X1, X2 must not be routed to NEON
        let w = enc("add", vec![xreg(0), xreg(1), xreg(2)]);
        assert_eq!(w, 0x8B02_0020);
    }

    #[test]
    fn scalar_sub_still_works() {
        // Scalar SUB X0, X1, X2 must not be routed to NEON
        let w = enc("sub", vec![xreg(0), xreg(1), xreg(2)]);
        assert_eq!(w, 0xCB02_0020);
    }

    #[test]
    fn scalar_mov_still_works() {
        // Scalar MOV X0, X1 must not be routed to NEON
        let w = enc("mov", vec![xreg(0), xreg(1)]);
        assert_eq!(w, 0xAA01_03E0);
    }

    // ── SVE helper constructors ──────────────────────────────

    fn zreg(n: u8, arr: VectorArrangement) -> Operand {
        use Register::*;
        let r = match n {
            0 => A64Z0,
            1 => A64Z1,
            2 => A64Z2,
            3 => A64Z3,
            4 => A64Z4,
            5 => A64Z5,
            6 => A64Z6,
            7 => A64Z7,
            8 => A64Z8,
            9 => A64Z9,
            10 => A64Z10,
            11 => A64Z11,
            12 => A64Z12,
            13 => A64Z13,
            14 => A64Z14,
            15 => A64Z15,
            16 => A64Z16,
            17 => A64Z17,
            18 => A64Z18,
            19 => A64Z19,
            20 => A64Z20,
            21 => A64Z21,
            22 => A64Z22,
            23 => A64Z23,
            24 => A64Z24,
            25 => A64Z25,
            26 => A64Z26,
            27 => A64Z27,
            28 => A64Z28,
            29 => A64Z29,
            30 => A64Z30,
            31 => A64Z31,
            _ => panic!("invalid Z register {}", n),
        };
        Operand::VectorRegister(r, arr)
    }

    fn preg_arr(n: u8, arr: VectorArrangement) -> Operand {
        use Register::*;
        let r = match n {
            0 => A64P0,
            1 => A64P1,
            2 => A64P2,
            3 => A64P3,
            4 => A64P4,
            5 => A64P5,
            6 => A64P6,
            7 => A64P7,
            8 => A64P8,
            9 => A64P9,
            10 => A64P10,
            11 => A64P11,
            12 => A64P12,
            13 => A64P13,
            14 => A64P14,
            15 => A64P15,
            _ => panic!("invalid P register {}", n),
        };
        Operand::VectorRegister(r, arr)
    }

    fn preg_m(n: u8) -> Operand {
        use Register::*;
        let r = match n {
            0 => A64P0,
            1 => A64P1,
            2 => A64P2,
            3 => A64P3,
            4 => A64P4,
            5 => A64P5,
            6 => A64P6,
            7 => A64P7,
            _ => panic!("invalid P register {}", n),
        };
        Operand::SvePredicate(r, SvePredQual::Merging)
    }

    fn preg_z(n: u8) -> Operand {
        use Register::*;
        let r = match n {
            0 => A64P0,
            1 => A64P1,
            2 => A64P2,
            3 => A64P3,
            4 => A64P4,
            5 => A64P5,
            6 => A64P6,
            7 => A64P7,
            _ => panic!("invalid P register {}", n),
        };
        Operand::SvePredicate(r, SvePredQual::Zeroing)
    }

    fn mem_base(reg: Register) -> Operand {
        Operand::Memory(Box::new(MemoryOperand {
            base: Some(reg),
            ..Default::default()
        }))
    }

    // ── SVE tests (cross-validated against llvm-mc) ──────────

    #[test]
    fn sve_ptrue_p0_b() {
        // ptrue p0.b → [0xe0,0xe3,0x18,0x25] = 0x2518E3E0
        let w = enc("ptrue", vec![preg_arr(0, VectorArrangement::SveB)]);
        assert_eq!(w, 0x2518_E3E0);
    }

    #[test]
    fn sve_pfalse_p0_b() {
        // pfalse p0.b → [0x00,0xe4,0x18,0x25] = 0x2518E400
        let w = enc("pfalse", vec![preg_arr(0, VectorArrangement::SveB)]);
        assert_eq!(w, 0x2518_E400);
    }

    #[test]
    fn sve_add_unpred_z0_z1_z2_s() {
        // add z0.s, z1.s, z2.s → [0x20,0x00,0xa2,0x04] = 0x04A20020
        let arr = VectorArrangement::SveS;
        let w = enc("add", vec![zreg(0, arr), zreg(1, arr), zreg(2, arr)]);
        assert_eq!(w, 0x04A2_0020);
    }

    #[test]
    fn sve_add_pred_z0_p0m_z0_z1_s() {
        // add z0.s, p0/m, z0.s, z1.s → [0x20,0x00,0x80,0x04] = 0x04800020
        let arr = VectorArrangement::SveS;
        let w = enc(
            "add",
            vec![zreg(0, arr), preg_m(0), zreg(0, arr), zreg(1, arr)],
        );
        assert_eq!(w, 0x0480_0020);
    }

    #[test]
    fn sve_sub_pred_z0_p0m_z0_z1_s() {
        // sub z0.s, p0/m, z0.s, z1.s → [0x20,0x00,0x81,0x04] = 0x04810020
        let arr = VectorArrangement::SveS;
        let w = enc(
            "sub",
            vec![zreg(0, arr), preg_m(0), zreg(0, arr), zreg(1, arr)],
        );
        assert_eq!(w, 0x0481_0020);
    }

    #[test]
    fn sve_mul_pred_z0_p0m_z0_z1_s() {
        // mul z0.s, p0/m, z0.s, z1.s → [0x20,0x00,0x90,0x04] = 0x04900020
        let arr = VectorArrangement::SveS;
        let w = enc(
            "mul",
            vec![zreg(0, arr), preg_m(0), zreg(0, arr), zreg(1, arr)],
        );
        assert_eq!(w, 0x0490_0020);
    }

    #[test]
    fn sve_and_unpred_z0_z1_z2_d() {
        // and z0.d, z1.d, z2.d → [0x20,0x30,0x22,0x04] = 0x04223020
        let arr = VectorArrangement::SveD;
        let w = enc("and", vec![zreg(0, arr), zreg(1, arr), zreg(2, arr)]);
        assert_eq!(w, 0x0422_3020);
    }

    #[test]
    fn sve_orr_unpred_z0_z1_z2_d() {
        // orr z0.d, z1.d, z2.d → [0x20,0x30,0x62,0x04] = 0x04623020
        let arr = VectorArrangement::SveD;
        let w = enc("orr", vec![zreg(0, arr), zreg(1, arr), zreg(2, arr)]);
        assert_eq!(w, 0x0462_3020);
    }

    #[test]
    fn sve_eor_unpred_z0_z1_z2_d() {
        // eor z0.d, z1.d, z2.d → [0x20,0x30,0xa2,0x04] = 0x04A23020
        let arr = VectorArrangement::SveD;
        let w = enc("eor", vec![zreg(0, arr), zreg(1, arr), zreg(2, arr)]);
        assert_eq!(w, 0x04A2_3020);
    }

    #[test]
    fn sve_and_pred_z0_p0m_z0_z1_b() {
        // and z0.b, p0/m, z0.b, z1.b → [0x20,0x00,0x1a,0x04] = 0x041A0020
        let arr = VectorArrangement::SveB;
        let w = enc(
            "and",
            vec![zreg(0, arr), preg_m(0), zreg(0, arr), zreg(1, arr)],
        );
        assert_eq!(w, 0x041A_0020);
    }

    #[test]
    fn sve_orr_pred_z0_p0m_z0_z1_b() {
        // orr z0.b, p0/m, z0.b, z1.b → [0x20,0x00,0x18,0x04] = 0x04180020
        let arr = VectorArrangement::SveB;
        let w = enc(
            "orr",
            vec![zreg(0, arr), preg_m(0), zreg(0, arr), zreg(1, arr)],
        );
        assert_eq!(w, 0x0418_0020);
    }

    #[test]
    fn sve_eor_pred_z0_p0m_z0_z1_b() {
        // eor z0.b, p0/m, z0.b, z1.b → [0x20,0x00,0x19,0x04] = 0x04190020
        let arr = VectorArrangement::SveB;
        let w = enc(
            "eor",
            vec![zreg(0, arr), preg_m(0), zreg(0, arr), zreg(1, arr)],
        );
        assert_eq!(w, 0x0419_0020);
    }

    #[test]
    fn sve_whilelt_p0_s_x0_x1() {
        // whilelt p0.s, x0, x1 → [0x00,0x14,0xa1,0x25] = 0x25A11400
        let w = enc(
            "whilelt",
            vec![preg_arr(0, VectorArrangement::SveS), xreg(0), xreg(1)],
        );
        assert_eq!(w, 0x25A1_1400);
    }

    #[test]
    fn sve_dup_z0_s_imm1() {
        // dup z0.s, #1 → [0x20,0xc0,0xb8,0x25] = 0x25B8C020
        let w = enc("dup", vec![zreg(0, VectorArrangement::SveS), imm(1)]);
        assert_eq!(w, 0x25B8_C020);
    }

    #[test]
    fn sve_cntb_x0() {
        // cntb x0 → [0xe0,0xe3,0x20,0x04] = 0x0420E3E0
        let w = enc("cntb", vec![xreg(0)]);
        assert_eq!(w, 0x0420_E3E0);
    }

    #[test]
    fn sve_cnth_x0() {
        // cnth x0 → [0xe0,0xe3,0x60,0x04] = 0x0460E3E0
        let w = enc("cnth", vec![xreg(0)]);
        assert_eq!(w, 0x0460_E3E0);
    }

    #[test]
    fn sve_cntw_x0() {
        // cntw x0 → [0xe0,0xe3,0xa0,0x04] = 0x04A0E3E0
        let w = enc("cntw", vec![xreg(0)]);
        assert_eq!(w, 0x04A0_E3E0);
    }

    #[test]
    fn sve_cntd_x0() {
        // cntd x0 → [0xe0,0xe3,0xe0,0x04] = 0x04E0E3E0
        let w = enc("cntd", vec![xreg(0)]);
        assert_eq!(w, 0x04E0_E3E0);
    }

    #[test]
    fn sve_add_imm_z0_s_1() {
        // add z0.s, z0.s, #1 → [0x20,0xc0,0xa0,0x25] = 0x25A0C020
        let arr = VectorArrangement::SveS;
        let w = enc("add", vec![zreg(0, arr), zreg(0, arr), imm(1)]);
        assert_eq!(w, 0x25A0_C020);
    }

    #[test]
    fn sve_ld1w_z0_p0z_x0() {
        // ld1w {z0.s}, p0/z, [x0] → [0x00,0xa0,0x40,0xa5] = 0xA540A000
        let w = enc(
            "ld1w",
            vec![
                zreg(0, VectorArrangement::SveS),
                preg_z(0),
                mem_base(Register::A64X0),
            ],
        );
        assert_eq!(w, 0xA540_A000);
    }

    #[test]
    fn sve_st1w_z0_p0m_x0() {
        // st1w {z0.s}, p0, [x0] → [0x00,0xe0,0x40,0xe5] = 0xE540E000
        // Note: ST1W uses bare predicate (parsed as p0/m or just p0,
        //       but the qualifier doesn't affect encoding for stores)
        let w = enc(
            "st1w",
            vec![
                zreg(0, VectorArrangement::SveS),
                preg_m(0),
                mem_base(Register::A64X0),
            ],
        );
        assert_eq!(w, 0xE540_E000);
    }
}
