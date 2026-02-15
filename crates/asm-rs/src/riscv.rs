//! RISC-V (RV32I / RV64I) instruction encoder.
//!
//! Implements encoding for the RISC-V base integer instruction sets (RV32I
//! and RV64I) plus the M (multiply/divide), A (atomics), and C (compressed)
//! extensions. Standard instructions are 32 bits wide; the C extension
//! provides 16-bit compressed forms for common operations.
//!
//! ## RISC-V Instruction Formats
//!
//! ```text
//! R-type:  [funct7 | rs2 | rs1 | funct3 | rd  | opcode]
//! I-type:  [  imm[11:0]  | rs1 | funct3 | rd  | opcode]
//! S-type:  [imm[11:5]|rs2| rs1 | funct3 |imm[4:0]|opcode]
//! B-type:  [imm[12|10:5]|rs2|rs1|funct3|imm[4:1|11]|opcode]
//! U-type:  [      imm[31:12]             | rd  | opcode]
//! J-type:  [imm[20|10:1|11|19:12]        | rd  | opcode]
//! ```
//!
//! ## Compressed (C) Instruction Formats (16-bit)
//!
//! ```text
//! CR:  [funct4 | rd/rs1 | rs2 | op]        (register)
//! CI:  [funct3 | imm | rd/rs1 | imm | op]  (immediate)
//! CSS: [funct3 | imm | rs2 | op]           (stack-store)
//! CIW: [funct3 | imm | rd' | op]           (wide immediate)
//! CL:  [funct3 | imm | rs1' | imm | rd' | op] (load)
//! CS:  [funct3 | imm | rs1' | imm | rs2'| op] (store)
//! CA:  [funct6 | rd'/rs1' | funct2 | rs2' | op] (arithmetic)
//! CB:  [funct3 | offset | rs1' | offset | op]  (branch)
//! CJ:  [funct3 | jump target | op]         (jump)
//! ```

use alloc::string::String;

use crate::encoder::{EncodedInstr, InstrBytes, RelocKind, Relocation};
use crate::error::{AsmError, Span};
use crate::ir::*;

// ── Opcodes ─────────────────────────────────────────────────────────────

const OP_LUI: u32 = 0b011_0111;
const OP_AUIPC: u32 = 0b001_0111;
const OP_JAL: u32 = 0b110_1111;
const OP_JALR: u32 = 0b110_0111;
const OP_BRANCH: u32 = 0b110_0011;
const OP_LOAD: u32 = 0b000_0011;
const OP_STORE: u32 = 0b010_0011;
const OP_IMM: u32 = 0b001_0011;
const OP_REG: u32 = 0b011_0011;
const OP_IMM_W: u32 = 0b001_1011; // RV64I W-suffix immediate ops
const OP_REG_W: u32 = 0b011_1011; // RV64I W-suffix register ops
const OP_SYSTEM: u32 = 0b111_0011;
const OP_FENCE: u32 = 0b000_1111;
const OP_AMO: u32 = 0b010_1111; // A extension: atomic memory operations

// ── F/D extension opcodes ───────────────────────────────────────────────

const OP_LOAD_FP: u32 = 0b000_0111; // I-type: FLW, FLD
const OP_STORE_FP: u32 = 0b010_0111; // S-type: FSW, FSD
const OP_MADD: u32 = 0b100_0011; // R4-type: FMADD
const OP_MSUB: u32 = 0b100_0111; // R4-type: FMSUB
const OP_NMSUB: u32 = 0b100_1011; // R4-type: FNMSUB
const OP_NMADD: u32 = 0b100_1111; // R4-type: FNMADD
const OP_FP: u32 = 0b101_0011; // R-type: all FP compute (FADD, FSUB, …)

// ── C-extension opcodes (quadrant bits [1:0]) ───────────────────────────

const C_OP_Q0: u16 = 0b00; // Quadrant 0
const C_OP_Q1: u16 = 0b01; // Quadrant 1
const C_OP_Q2: u16 = 0b10; // Quadrant 2

// ── V-extension opcodes ─────────────────────────────────────────────────

const OP_V_LOAD: u32 = 0b000_0111; // Vector load (same encoding space as FP loads)
const OP_V_STORE: u32 = 0b010_0111; // Vector store (same encoding space as FP stores)
const OP_V: u32 = 0b101_0111; // Vector arithmetic / vset{i}vli / vsetvl

// ── Compressed encoding helpers ─────────────────────────────────────────

/// Map full register number (x8–x15) to compressed 3-bit encoding (0–7).
/// Returns `None` if the register is not in the "compact" set.
#[inline]
fn compact_reg(r: u32) -> Option<u32> {
    if (8..=15).contains(&r) {
        Some(r - 8)
    } else {
        None
    }
}

/// Encode a CR-type compressed instruction.
///   `[funct4(4) | rd/rs1(5) | rs2(5) | op(2)]`
#[inline]
fn cr_type(funct4: u16, rd_rs1: u16, rs2: u16, op: u16) -> u16 {
    (funct4 << 12) | (rd_rs1 << 7) | (rs2 << 2) | op
}

/// Encode a CI-type compressed instruction.
///   `[funct3(3) | imm[5](1) | rd/rs1(5) | imm[4:0](5) | op(2)]`
#[inline]
fn ci_type(funct3: u16, imm_bit5: u16, rd_rs1: u16, imm_lo5: u16, op: u16) -> u16 {
    (funct3 << 13) | ((imm_bit5 & 1) << 12) | (rd_rs1 << 7) | ((imm_lo5 & 0x1F) << 2) | op
}

/// Encode a CSS-type compressed instruction (stack-relative store).
///   `[funct3(3) | imm(6) | rs2(5) | op(2)]`
#[inline]
fn css_type(funct3: u16, imm6: u16, rs2: u16, op: u16) -> u16 {
    (funct3 << 13) | ((imm6 & 0x3F) << 7) | (rs2 << 2) | op
}

/// Encode a CIW-type compressed instruction (wide immediate).
///   `[funct3(3) | imm(8) | rd'(3) | op(2)]`
#[inline]
fn ciw_type(funct3: u16, imm8: u16, rd_p: u16, op: u16) -> u16 {
    (funct3 << 13) | ((imm8 & 0xFF) << 5) | ((rd_p & 7) << 2) | op
}

/// Encode a CL-type compressed instruction (load from base+offset).
///   `[funct3(3) | imm_hi(3) | rs1'(3) | imm_lo(2) | rd'(3) | op(2)]`
#[inline]
fn cl_type(funct3: u16, imm_hi3: u16, rs1_p: u16, imm_lo2: u16, rd_p: u16, op: u16) -> u16 {
    (funct3 << 13)
        | ((imm_hi3 & 7) << 10)
        | ((rs1_p & 7) << 7)
        | ((imm_lo2 & 3) << 5)
        | ((rd_p & 7) << 2)
        | op
}

/// Encode a CS-type compressed instruction (store to base+offset).
///   `[funct3(3) | imm_hi(3) | rs1'(3) | imm_lo(2) | rs2'(3) | op(2)]`
#[inline]
fn cs_type(funct3: u16, imm_hi3: u16, rs1_p: u16, imm_lo2: u16, rs2_p: u16, op: u16) -> u16 {
    (funct3 << 13)
        | ((imm_hi3 & 7) << 10)
        | ((rs1_p & 7) << 7)
        | ((imm_lo2 & 3) << 5)
        | ((rs2_p & 7) << 2)
        | op
}

/// Encode a CA-type compressed instruction (register arithmetic).
///   `[funct6(6) | rd'/rs1'(3) | funct2(2) | rs2'(3) | op(2)]`
#[inline]
fn ca_type(funct6: u16, rd_rs1_p: u16, funct2: u16, rs2_p: u16, op: u16) -> u16 {
    (funct6 << 10) | ((rd_rs1_p & 7) << 7) | ((funct2 & 3) << 5) | ((rs2_p & 7) << 2) | op
}

/// Encode a CB-type compressed branch.
///   `[funct3(3) | offset[8|4:3](3) | rs1'(3) | offset[7:6|2:1|5](5) | op(2)]`
#[inline]
fn cb_type(funct3: u16, rs1_p: u16, offset: i32) -> u16 {
    let off = offset as u16;
    let bit8 = (off >> 8) & 1;
    let bits4_3 = (off >> 3) & 3;
    let bits7_6 = (off >> 6) & 3;
    let bits2_1 = (off >> 1) & 3;
    let bit5 = (off >> 5) & 1;
    (funct3 << 13)
        | (bit8 << 12)
        | (bits4_3 << 10)
        | ((rs1_p & 7) << 7)
        | (bits7_6 << 5)
        | (bits2_1 << 3)
        | (bit5 << 2)
        | C_OP_Q1
}

/// Encode a CJ-type compressed jump.
///   `[funct3(3) | jump_target[11|4|9:8|10|6|7|3:1|5](11) | op(2)]`
#[inline]
fn cj_type(funct3: u16, offset: i32) -> u16 {
    let off = offset as u16;
    let bit11 = (off >> 11) & 1;
    let bit4 = (off >> 4) & 1;
    let bits9_8 = (off >> 8) & 3;
    let bit10 = (off >> 10) & 1;
    let bit6 = (off >> 6) & 1;
    let bit7 = (off >> 7) & 1;
    let bits3_1 = (off >> 1) & 7;
    let bit5 = (off >> 5) & 1;
    let target = (bit11 << 10)
        | (bit4 << 9)
        | (bits9_8 << 7)
        | (bit10 << 6)
        | (bit6 << 5)
        | (bit7 << 4)
        | (bits3_1 << 1)
        | bit5;
    (funct3 << 13) | (target << 2) | C_OP_Q1
}

// ── Encoding helpers ────────────────────────────────────────────────────

/// Encode an R-type instruction.
#[inline]
fn r_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, rs2: u32, funct7: u32) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

/// Encode an I-type instruction.
#[inline]
fn i_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, imm: i32) -> u32 {
    let imm = (imm as u32) & 0xFFF;
    (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

/// Encode an S-type instruction.
#[inline]
fn s_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    let imm_hi = (imm >> 5) & 0x7F;
    let imm_lo = imm & 0x1F;
    (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | opcode
}

/// Encode a B-type instruction.
#[inline]
fn b_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    let bit12 = (imm >> 12) & 1;
    let bit11 = (imm >> 11) & 1;
    let bits10_5 = (imm >> 5) & 0x3F;
    let bits4_1 = (imm >> 1) & 0xF;
    (bit12 << 31)
        | (bits10_5 << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (bits4_1 << 8)
        | (bit11 << 7)
        | opcode
}

/// Encode a U-type instruction.
#[inline]
fn u_type(opcode: u32, rd: u32, imm: u32) -> u32 {
    (imm & 0xFFFF_F000) | (rd << 7) | opcode
}

/// Encode a J-type instruction.
#[inline]
fn j_type(opcode: u32, rd: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    let bit20 = (imm >> 20) & 1;
    let bits10_1 = (imm >> 1) & 0x3FF;
    let bit11 = (imm >> 11) & 1;
    let bits19_12 = (imm >> 12) & 0xFF;
    (bit20 << 31) | (bits10_1 << 21) | (bit11 << 20) | (bits19_12 << 12) | (rd << 7) | opcode
}

/// Encode an AMO (atomic memory operation) instruction.
///
/// Format: `[funct5 | aq | rl | rs2 | rs1 | funct3 | rd | opcode]`
#[inline]
fn amo_type(funct5: u32, aq: bool, rl: bool, rs2: u32, rs1: u32, funct3: u32, rd: u32) -> u32 {
    (funct5 << 27)
        | ((aq as u32) << 26)
        | ((rl as u32) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (rd << 7)
        | OP_AMO
}

/// Encode an R4-type (fused multiply-add) instruction.
///
/// Format: `[rs3 | fmt | rs2 | rs1 | rm | rd | opcode]`
///
/// Used by `FMADD`, `FMSUB`, `FNMSUB`, `FNMADD`.
#[inline]
fn r4_type(opcode: u32, rd: u32, rm: u32, rs1: u32, rs2: u32, fmt: u32, rs3: u32) -> u32 {
    (rs3 << 27) | (fmt << 25) | (rs2 << 20) | (rs1 << 15) | (rm << 12) | (rd << 7) | opcode
}

// ── RV64 multi-instruction LI ───────────────────────────────────────────

/// Sign-extend a 12-bit value embedded in the low 12 bits of an `i32`.
#[inline]
fn sign_extend_12(val: i32) -> i32 {
    (val << 20) >> 20
}

/// Encode a full 64-bit immediate load for RV64.
///
/// Produces at most 8 instructions (LUI/ADDI/SLLI combinations), matching
/// the standard GAS expansion for `li` on RV64.  The recursive algorithm
/// peels off 12-bit signed chunks from the bottom and shifts the remainder
/// upward until the value fits in a 32-bit LUI+ADDI or a 12-bit ADDI.
fn encode_li_rv64(rd: u32, val: i64) -> EncodedInstr {
    let mut bytes = InstrBytes::new();
    emit_li_rv64(rd, val, &mut bytes);
    EncodedInstr {
        bytes,
        relocation: None,
        relax: None,
    }
}

/// Recursively emit the instruction words for an RV64 immediate load.
///
/// The first emitted instruction is always either `LUI` (no source
/// register) or `ADDI rd, x0, imm` — both initialise `rd` from scratch.
/// All subsequent instructions use `rd` as both source and destination.
fn emit_li_rv64(rd: u32, val: i64, bytes: &mut InstrBytes) {
    // Base: fits in 12-bit signed → single ADDI rd, x0, imm
    if (-2048..=2047).contains(&val) {
        let w = i_type(OP_IMM, rd, 0, 0, val as i32);
        bytes.extend_from_slice(&w.to_le_bytes());
        return;
    }

    // Fits in 32-bit signed → LUI + optional ADDI
    if (-2_147_483_648..=2_147_483_647).contains(&val) {
        let lo12 = sign_extend_12(val as i32);
        let hi20 = ((val as i32).wrapping_sub(lo12)) as u32;
        let w = u_type(OP_LUI, rd, hi20);
        bytes.extend_from_slice(&w.to_le_bytes());
        if lo12 != 0 {
            let w = i_type(OP_IMM, rd, 0, rd, lo12);
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        return;
    }

    // Full 64-bit: peel off low 12 bits, shift the remainder, recurse.
    let lo12 = sign_extend_12(val as i32);
    let remaining = val.wrapping_sub(lo12 as i64);

    // `remaining` is a multiple of 2^12 (at minimum); use trailing zeros
    // as the shift amount so the upper part is as small as possible.
    let shamt = (remaining as u64).trailing_zeros().clamp(12, 63);
    let upper = remaining >> shamt;

    // Emit instructions for the upper part first (recursion).
    emit_li_rv64(rd, upper, bytes);

    // SLLI rd, rd, shamt
    let w = i_type(OP_IMM, rd, 1, rd, shamt as i32);
    bytes.extend_from_slice(&w.to_le_bytes());

    // ADDI rd, rd, lo12  (only when the low chunk is non-zero)
    if lo12 != 0 {
        let w = i_type(OP_IMM, rd, 0, rd, lo12);
        bytes.extend_from_slice(&w.to_le_bytes());
    }
}

// ── Named CSR mapping ───────────────────────────────────────────────────

/// Resolve a CSR name to its 12-bit address.
fn csr_by_name(name: &str) -> Option<u32> {
    Some(match name {
        // Machine-level CSRs
        "mstatus" => 0x300,
        "misa" => 0x301,
        "medeleg" => 0x302,
        "mideleg" => 0x303,
        "mie" => 0x304,
        "mtvec" => 0x305,
        "mcounteren" => 0x306,
        "mstatush" => 0x310,
        "mscratch" => 0x340,
        "mepc" => 0x341,
        "mcause" => 0x342,
        "mtval" => 0x343,
        "mip" => 0x344,
        "mhartid" => 0xF14,
        "mvendorid" => 0xF11,
        "marchid" => 0xF12,
        "mimpid" => 0xF13,
        // Machine memory protection
        "pmpcfg0" => 0x3A0,
        "pmpcfg1" => 0x3A1,
        "pmpcfg2" => 0x3A2,
        "pmpcfg3" => 0x3A3,
        "pmpaddr0" => 0x3B0,
        "pmpaddr1" => 0x3B1,
        "pmpaddr2" => 0x3B2,
        "pmpaddr3" => 0x3B3,
        // Machine counters
        "mcycle" => 0xB00,
        "minstret" => 0xB02,
        "mcycleh" => 0xB80,
        "minstreth" => 0xB82,
        "mcountinhibit" => 0x320,
        // Supervisor-level CSRs
        "sstatus" => 0x100,
        "sie" => 0x104,
        "stvec" => 0x105,
        "scounteren" => 0x106,
        "sscratch" => 0x140,
        "sepc" => 0x141,
        "scause" => 0x142,
        "stval" => 0x143,
        "sip" => 0x144,
        "satp" => 0x180,
        // User-level CSRs (read-only counters)
        "cycle" => 0xC00,
        "time" => 0xC01,
        "instret" => 0xC02,
        "cycleh" => 0xC80,
        "timeh" => 0xC81,
        "instreth" => 0xC82,
        // Floating-point CSRs
        "fflags" => 0x001,
        "frm" => 0x002,
        "fcsr" => 0x003,
        _ => return None,
    })
}

// ── Register extraction helpers ──────────────────────────────────────────

fn reg(op: &Operand, span: Span) -> Result<u32, AsmError> {
    match op {
        Operand::Register(r) if r.is_riscv() => Ok(r.rv_reg_num() as u32),
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected RISC-V register"),
            span,
        }),
    }
}

/// Extract a RISC-V floating-point register number (f0–f31) from an operand.
fn fpreg(op: &Operand, span: Span) -> Result<u32, AsmError> {
    match op {
        Operand::Register(r) if r.is_riscv_fp() => Ok(r.rv_fp_reg_num() as u32),
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected RISC-V FP register (f0-f31)"),
            span,
        }),
    }
}

fn imm_i(op: &Operand, span: Span) -> Result<i32, AsmError> {
    match op {
        Operand::Immediate(v) => {
            let v = *v as i64;
            if !(-2048..=2047).contains(&v) {
                return Err(AsmError::ImmediateOverflow {
                    value: v as i128,
                    min: -(1 << 11),
                    max: (1 << 11) - 1,
                    span,
                });
            }
            Ok(v as i32)
        }
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected 12-bit immediate"),
            span,
        }),
    }
}

fn imm_u(op: &Operand, span: Span) -> Result<u32, AsmError> {
    match op {
        Operand::Immediate(v) => {
            let v = *v as i64;
            // LUI/AUIPC take a 20-bit value that gets placed in bits 31:12.
            // The assembler convention is the immediate is given as the raw
            // upper 20 bits (0–0xFFFFF), which we shift left by 12.
            if !(0..=0xFFFFF).contains(&v) && !(-524288..=-1).contains(&v) {
                return Err(AsmError::ImmediateOverflow {
                    value: v as i128,
                    min: 0,
                    max: (1 << 20) - 1,
                    span,
                });
            }
            Ok(((v as u32) & 0xFFFFF) << 12)
        }
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected 20-bit immediate for U-type"),
            span,
        }),
    }
}

fn mem(op: &Operand, span: Span) -> Result<(u32, i32), AsmError> {
    match op {
        Operand::Memory(m) => {
            let base = m.base.map_or(0, |r| r.rv_reg_num() as u32);
            let disp = m.disp as i32;
            Ok((base, disp))
        }
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected memory operand offset(reg)"),
            span,
        }),
    }
}

fn extract_label(op: &Operand) -> Option<(&str, i64)> {
    crate::encoder::extract_label(op)
}

/// Extract the base register from a memory operand `(rs1)` for atomic instructions.
/// The displacement must be zero.
fn amo_addr(op: &Operand, span: Span) -> Result<u32, AsmError> {
    match op {
        Operand::Memory(m) => {
            if m.disp != 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("atomic instructions require zero-offset address: (rs1)"),
                    span,
                });
            }
            Ok(m.base.map_or(0, |r| r.rv_reg_num() as u32))
        }
        // Also allow bare register for AMO (common in some assemblers)
        Operand::Register(r) if r.is_riscv() => Ok(r.rv_reg_num() as u32),
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected address operand (rs1) for atomic instruction"),
            span,
        }),
    }
}

/// Resolve a CSR operand: either an immediate number or a named CSR register.
fn csr_operand(op: &Operand, span: Span) -> Result<u32, AsmError> {
    match op {
        Operand::Immediate(v) => Ok((*v as u32) & 0xFFF),
        Operand::Label(name) => csr_by_name(name).ok_or_else(|| AsmError::InvalidOperands {
            detail: alloc::format!("unknown CSR name '{}'", name),
            span,
        }),
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected CSR address (immediate or name)"),
            span,
        }),
    }
}

// ── V-extension helpers ─────────────────────────────────────────────────

/// Extract a RISC-V vector register number (0–31) from an operand.
fn vreg(op: &Operand, span: Span) -> Result<u32, AsmError> {
    match op {
        Operand::Register(r) if r.is_riscv_vec() => Ok(r.rv_vec_num() as u32),
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected RISC-V vector register (v0–v31)"),
            span,
        }),
    }
}

/// Parse vtype components from label operands.
///
/// Parses `e{sew}, m{lmul}, ta|tu, ma|mu` and returns the vtype immediate.
/// The vtype field layout (bits [7:0]):
///   [7] = vma (0=mu, 1=ma)
///   [6] = vta (0=tu, 1=ta)
///   [5:3] = vsew (e8=0, e16=1, e32=2, e64=3)
///   [2:0] = vlmul (m1=0, m2=1, m4=2, m8=3, mf8=5, mf4=6, mf2=7)
fn parse_vtype(ops: &[Operand], start: usize, span: Span) -> Result<u32, AsmError> {
    if ops.len() < start + 4 {
        return Err(AsmError::InvalidOperands {
            detail: String::from("expected vtype: e{sew}, m{lmul}, ta|tu, ma|mu"),
            span,
        });
    }

    let sew = match &ops[start] {
        Operand::Label(s) => match s.as_str() {
            "e8" => 0u32,
            "e16" => 1,
            "e32" => 2,
            "e64" => 3,
            _ => {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!("invalid SEW '{}', expected e8/e16/e32/e64", s),
                    span,
                })
            }
        },
        _ => {
            return Err(AsmError::InvalidOperands {
                detail: String::from("expected SEW (e8/e16/e32/e64)"),
                span,
            })
        }
    };

    let lmul = match &ops[start + 1] {
        Operand::Label(s) => match s.as_str() {
            "m1" => 0u32,
            "m2" => 1,
            "m4" => 2,
            "m8" => 3,
            "mf8" => 5,
            "mf4" => 6,
            "mf2" => 7,
            _ => {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!(
                        "invalid LMUL '{}', expected m1/m2/m4/m8/mf2/mf4/mf8",
                        s
                    ),
                    span,
                })
            }
        },
        _ => {
            return Err(AsmError::InvalidOperands {
                detail: String::from("expected LMUL (m1/m2/m4/m8/mf2/mf4/mf8)"),
                span,
            })
        }
    };

    let vta = match &ops[start + 2] {
        Operand::Label(s) => match s.as_str() {
            "ta" => 1u32,
            "tu" => 0,
            _ => {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!("invalid tail agnostic '{}', expected ta/tu", s),
                    span,
                })
            }
        },
        _ => {
            return Err(AsmError::InvalidOperands {
                detail: String::from("expected tail policy (ta/tu)"),
                span,
            })
        }
    };

    let vma = match &ops[start + 3] {
        Operand::Label(s) => match s.as_str() {
            "ma" => 1u32,
            "mu" => 0,
            _ => {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!("invalid mask agnostic '{}', expected ma/mu", s),
                    span,
                })
            }
        },
        _ => {
            return Err(AsmError::InvalidOperands {
                detail: String::from("expected mask policy (ma/mu)"),
                span,
            })
        }
    };

    Ok((vma << 7) | (vta << 6) | (sew << 3) | lmul)
}

// ── Compressed (C-extension) instruction encoder ────────────────────────

/// Encode an explicit C-extension instruction (e.g., `c.addi`, `c.mv`).
///
/// Returns a 2-byte `EncodedInstr` on success.
fn encode_rvc_explicit(
    mnemonic: &str,
    ops: &[Operand],
    is_rv64: bool,
    span: Span,
) -> Result<EncodedInstr, AsmError> {
    let hw = match mnemonic {
        // ── CR-type ─────────────────────────────────────────────
        "c.mv" => {
            let rd = reg(&ops[0], span)? as u16;
            let rs2 = reg(&ops[1], span)? as u16;
            if rd == 0 || rs2 == 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.mv: rd and rs2 must not be x0"),
                    span,
                });
            }
            cr_type(0b1000, rd, rs2, C_OP_Q2)
        }
        "c.add" => {
            let rd = reg(&ops[0], span)? as u16;
            let rs2 = reg(&ops[1], span)? as u16;
            if rd == 0 || rs2 == 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.add: rd and rs2 must not be x0"),
                    span,
                });
            }
            cr_type(0b1001, rd, rs2, C_OP_Q2)
        }
        "c.jr" => {
            let rs1 = reg(&ops[0], span)? as u16;
            if rs1 == 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.jr: rs1 must not be x0"),
                    span,
                });
            }
            cr_type(0b1000, rs1, 0, C_OP_Q2)
        }
        "c.jalr" => {
            let rs1 = reg(&ops[0], span)? as u16;
            if rs1 == 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.jalr: rs1 must not be x0"),
                    span,
                });
            }
            cr_type(0b1001, rs1, 0, C_OP_Q2)
        }

        // ── CI-type ─────────────────────────────────────────────
        "c.li" => {
            let rd = reg(&ops[0], span)? as u16;
            let imm = ci_imm6(&ops[1], span)?;
            if rd == 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.li: rd must not be x0"),
                    span,
                });
            }
            ci_type(0b010, (imm >> 5) & 1, rd, imm & 0x1F, C_OP_Q1)
        }
        "c.lui" => {
            let rd = reg(&ops[0], span)? as u16;
            let imm = ci_imm6(&ops[1], span)?;
            if rd == 0 || rd == 2 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.lui: rd must not be x0 or x2 (sp)"),
                    span,
                });
            }
            if imm == 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.lui: immediate must not be zero"),
                    span,
                });
            }
            ci_type(0b011, (imm >> 5) & 1, rd, imm & 0x1F, C_OP_Q1)
        }
        "c.addi" => {
            let rd = reg(&ops[0], span)? as u16;
            let imm = ci_imm6(&ops[1], span)?;
            if rd == 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.addi: rd must not be x0"),
                    span,
                });
            }
            ci_type(0b000, (imm >> 5) & 1, rd, imm & 0x1F, C_OP_Q1)
        }
        "c.addiw" if is_rv64 => {
            let rd = reg(&ops[0], span)? as u16;
            let imm = ci_imm6(&ops[1], span)?;
            if rd == 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.addiw: rd must not be x0"),
                    span,
                });
            }
            ci_type(0b001, (imm >> 5) & 1, rd, imm & 0x1F, C_OP_Q1)
        }
        "c.addi16sp" => {
            // Adds sign-extended immediate*16 to SP. Immediate must be non-zero.
            let imm = ci_imm_addi16sp(&ops[0], span)?;
            ci_addi16sp(imm)
        }
        "c.slli" => {
            let rd = reg(&ops[0], span)? as u16;
            let shamt = ci_shamt(&ops[1], is_rv64, span)?;
            if rd == 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.slli: rd must not be x0"),
                    span,
                });
            }
            ci_type(0b000, (shamt >> 5) & 1, rd, shamt & 0x1F, C_OP_Q2)
        }
        "c.lwsp" => {
            let rd = reg(&ops[0], span)? as u16;
            let off = ci_uimm_sp_lw(&ops[1], span)?;
            if rd == 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.lwsp: rd must not be x0"),
                    span,
                });
            }
            // offset[5] in bit 12, offset[4:2|7:6] in bits 6:2
            let bit5 = (off >> 5) & 1;
            let bits4_2 = (off >> 2) & 7;
            let bits7_6 = (off >> 6) & 3;
            ci_type(0b010, bit5, rd, (bits4_2 << 2) | bits7_6, C_OP_Q2)
        }
        "c.ldsp" if is_rv64 => {
            let rd = reg(&ops[0], span)? as u16;
            let off = ci_uimm_sp_ld(&ops[1], span)?;
            if rd == 0 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("c.ldsp: rd must not be x0"),
                    span,
                });
            }
            // offset[5] in bit 12, offset[4:3|8:6] in bits 6:2
            let bit5 = (off >> 5) & 1;
            let bits4_3 = (off >> 3) & 3;
            let bits8_6 = (off >> 6) & 7;
            ci_type(0b011, bit5, rd, (bits4_3 << 3) | bits8_6, C_OP_Q2)
        }

        // ── CSS-type (stack-relative stores) ────────────────────
        "c.swsp" => {
            let rs2 = reg(&ops[0], span)? as u16;
            let off = ci_uimm_sp_lw(&ops[1], span)?;
            // offset[5:2|7:6]
            let bits5_2 = (off >> 2) & 0xF;
            let bits7_6 = (off >> 6) & 3;
            css_type(0b110, (bits5_2 << 2) | bits7_6, rs2, C_OP_Q2)
        }
        "c.sdsp" if is_rv64 => {
            let rs2 = reg(&ops[0], span)? as u16;
            let off = ci_uimm_sp_ld(&ops[1], span)?;
            // offset[5:3|8:6]
            let bits5_3 = (off >> 3) & 7;
            let bits8_6 = (off >> 6) & 7;
            css_type(0b111, (bits5_3 << 3) | bits8_6, rs2, C_OP_Q2)
        }

        // ── CL-type (load from base+offset, compact regs) ──────
        "c.lw" => {
            let rd_p =
                compact_reg(reg(&ops[0], span)?).ok_or_else(|| AsmError::InvalidOperands {
                    detail: String::from("c.lw: rd must be x8-x15"),
                    span,
                })? as u16;
            let (rs1, off) = compact_mem(&ops[1], span)?;
            let rs1_p = rs1 as u16;
            let off_u = cl_offset_w(off, span)?;
            // offset[5:3] in bits 12:10, offset[2|6] in bits 6:5
            let bits5_3 = (off_u >> 3) & 7;
            let bit2 = (off_u >> 2) & 1;
            let bit6 = (off_u >> 6) & 1;
            cl_type(0b010, bits5_3, rs1_p, (bit6 << 1) | bit2, rd_p, C_OP_Q0)
        }
        "c.ld" if is_rv64 => {
            let rd_p =
                compact_reg(reg(&ops[0], span)?).ok_or_else(|| AsmError::InvalidOperands {
                    detail: String::from("c.ld: rd must be x8-x15"),
                    span,
                })? as u16;
            let (rs1, off) = compact_mem(&ops[1], span)?;
            let rs1_p = rs1 as u16;
            let off_u = cl_offset_d(off, span)?;
            // offset[5:3] in bits 12:10, offset[7:6] in bits 6:5
            let bits5_3 = (off_u >> 3) & 7;
            let bits7_6 = (off_u >> 6) & 3;
            cl_type(0b011, bits5_3, rs1_p, bits7_6, rd_p, C_OP_Q0)
        }

        // ── CS-type (store to base+offset, compact regs) ────────
        "c.sw" => {
            let rs2_p =
                compact_reg(reg(&ops[0], span)?).ok_or_else(|| AsmError::InvalidOperands {
                    detail: String::from("c.sw: rs2 must be x8-x15"),
                    span,
                })? as u16;
            let (rs1, off) = compact_mem(&ops[1], span)?;
            let rs1_p = rs1 as u16;
            let off_u = cl_offset_w(off, span)?;
            let bits5_3 = (off_u >> 3) & 7;
            let bit2 = (off_u >> 2) & 1;
            let bit6 = (off_u >> 6) & 1;
            cs_type(0b110, bits5_3, rs1_p, (bit6 << 1) | bit2, rs2_p, C_OP_Q0)
        }
        "c.sd" if is_rv64 => {
            let rs2_p =
                compact_reg(reg(&ops[0], span)?).ok_or_else(|| AsmError::InvalidOperands {
                    detail: String::from("c.sd: rs2 must be x8-x15"),
                    span,
                })? as u16;
            let (rs1, off) = compact_mem(&ops[1], span)?;
            let rs1_p = rs1 as u16;
            let off_u = cl_offset_d(off, span)?;
            let bits5_3 = (off_u >> 3) & 7;
            let bits7_6 = (off_u >> 6) & 3;
            cs_type(0b111, bits5_3, rs1_p, bits7_6, rs2_p, C_OP_Q0)
        }

        // ── CA-type (register-register, compact regs) ───────────
        "c.sub" => ca_arith(ops, 0b100011, 0b00, span)?,
        "c.xor" => ca_arith(ops, 0b100011, 0b01, span)?,
        "c.or" => ca_arith(ops, 0b100011, 0b10, span)?,
        "c.and" => ca_arith(ops, 0b100011, 0b11, span)?,
        "c.subw" if is_rv64 => ca_arith(ops, 0b100111, 0b00, span)?,
        "c.addw" if is_rv64 => ca_arith(ops, 0b100111, 0b01, span)?,

        // ── CB-type (compressed branch) ─────────────────────────
        "c.beqz" => {
            let rs1_p =
                compact_reg(reg(&ops[0], span)?).ok_or_else(|| AsmError::InvalidOperands {
                    detail: String::from("c.beqz: rs1 must be x8-x15"),
                    span,
                })? as u16;
            if let Some((label, addend)) = extract_label(&ops[1]) {
                return Ok(rvc_branch_reloc(0b110, rs1_p, label, addend));
            }
            let off = cb_offset(&ops[1], span)?;
            cb_type(0b110, rs1_p, off)
        }
        "c.bnez" => {
            let rs1_p =
                compact_reg(reg(&ops[0], span)?).ok_or_else(|| AsmError::InvalidOperands {
                    detail: String::from("c.bnez: rs1 must be x8-x15"),
                    span,
                })? as u16;
            if let Some((label, addend)) = extract_label(&ops[1]) {
                return Ok(rvc_branch_reloc(0b111, rs1_p, label, addend));
            }
            let off = cb_offset(&ops[1], span)?;
            cb_type(0b111, rs1_p, off)
        }

        // ── CJ-type (compressed jump) ──────────────────────────
        "c.j" => {
            if let Some((label, addend)) = extract_label(&ops[0]) {
                return Ok(rvc_jump_reloc(label, addend));
            }
            let off = cj_offset(&ops[0], span)?;
            cj_type(0b101, off)
        }

        // ── Misc ────────────────────────────────────────────────
        "c.nop" => ci_type(0b000, 0, 0, 0, C_OP_Q1),
        "c.ebreak" => cr_type(0b1001, 0, 0, C_OP_Q2),

        // ── CB-type shifts/andi (compact regs) ──────────────────
        "c.srli" => {
            let rd_p =
                compact_reg(reg(&ops[0], span)?).ok_or_else(|| AsmError::InvalidOperands {
                    detail: String::from("c.srli: rd must be x8-x15"),
                    span,
                })? as u16;
            let shamt = ci_shamt(&ops[1], is_rv64, span)?;
            cb_shift(0b00, rd_p, shamt)
        }
        "c.srai" => {
            let rd_p =
                compact_reg(reg(&ops[0], span)?).ok_or_else(|| AsmError::InvalidOperands {
                    detail: String::from("c.srai: rd must be x8-x15"),
                    span,
                })? as u16;
            let shamt = ci_shamt(&ops[1], is_rv64, span)?;
            cb_shift(0b01, rd_p, shamt)
        }
        "c.andi" => {
            let rd_p =
                compact_reg(reg(&ops[0], span)?).ok_or_else(|| AsmError::InvalidOperands {
                    detail: String::from("c.andi: rd must be x8-x15"),
                    span,
                })? as u16;
            let imm = ci_imm6(&ops[1], span)?;
            // same encoding as CB-type with funct2=10
            let bit5 = (imm >> 5) & 1;
            (0b100u16 << 13)
                | (bit5 << 12)
                | (0b10u16 << 10)
                | (rd_p << 7)
                | ((imm & 0x1F) << 2)
                | C_OP_Q1
        }

        // ── CIW-type (c.addi4spn) ──────────────────────────────
        "c.addi4spn" => {
            let rd_p =
                compact_reg(reg(&ops[0], span)?).ok_or_else(|| AsmError::InvalidOperands {
                    detail: String::from("c.addi4spn: rd must be x8-x15"),
                    span,
                })? as u16;
            let imm = match &ops[1] {
                Operand::Immediate(v) => {
                    let v = *v as i64;
                    if v <= 0 || v > 1020 || v % 4 != 0 {
                        return Err(AsmError::ImmediateOverflow {
                            value: v as i128,
                            min: 4,
                            max: 1020,
                            span,
                        });
                    }
                    v as u16
                }
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("expected unsigned immediate multiple of 4 (4..1020)"),
                        span,
                    })
                }
            };
            // CIW immediate encodes nzuimm[5:4|9:6|2|3]
            let bits5_4 = (imm >> 4) & 3;
            let bits9_6 = (imm >> 6) & 0xF;
            let bit2 = (imm >> 2) & 1;
            let bit3 = (imm >> 3) & 1;
            ciw_type(
                0b000,
                (bits5_4 << 6) | (bits9_6 << 2) | (bit2 << 1) | bit3,
                rd_p,
                C_OP_Q0,
            )
        }

        _ => {
            return Err(AsmError::UnknownMnemonic {
                mnemonic: String::from(mnemonic),
                arch: if is_rv64 {
                    crate::error::ArchName::Rv64
                } else {
                    crate::error::ArchName::Rv32
                },
                span,
            });
        }
    };

    let mut bytes = InstrBytes::new();
    bytes.extend_from_slice(&hw.to_le_bytes());
    Ok(EncodedInstr {
        bytes,
        relocation: None,
        relax: None,
    })
}

// ── C-extension immediate helpers ───────────────────────────────────────

/// Extract a 6-bit sign-extended immediate for CI-type: range −32..31.
fn ci_imm6(op: &Operand, span: Span) -> Result<u16, AsmError> {
    match op {
        Operand::Immediate(v) => {
            let v = *v as i64;
            if !(-32..=31).contains(&v) {
                return Err(AsmError::ImmediateOverflow {
                    value: v as i128,
                    min: -32,
                    max: 31,
                    span,
                });
            }
            Ok((v as u16) & 0x3F)
        }
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected 6-bit immediate"),
            span,
        }),
    }
}

/// Extract shift amount for C-extension: 1..31 (RV32), 1..63 (RV64).
fn ci_shamt(op: &Operand, is_rv64: bool, span: Span) -> Result<u16, AsmError> {
    match op {
        Operand::Immediate(v) => {
            let v = *v as i64;
            let max: i64 = if is_rv64 { 63 } else { 31 };
            if v < 1 || v > max {
                return Err(AsmError::ImmediateOverflow {
                    value: v as i128,
                    min: 1,
                    max: max as i128,
                    span,
                });
            }
            Ok(v as u16)
        }
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected shift amount"),
            span,
        }),
    }
}

/// Extract immediate for c.addi16sp: must be a non-zero multiple of 16, range ±496.
fn ci_imm_addi16sp(op: &Operand, span: Span) -> Result<i32, AsmError> {
    match op {
        Operand::Immediate(v) => {
            let v = *v as i64;
            if v == 0 || v % 16 != 0 || !(-512..=496).contains(&v) {
                return Err(AsmError::ImmediateOverflow {
                    value: v as i128,
                    min: -512,
                    max: 496,
                    span,
                });
            }
            Ok(v as i32)
        }
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected immediate multiple of 16"),
            span,
        }),
    }
}

/// Encode c.addi16sp: CI-type with special immediate layout.
fn ci_addi16sp(imm: i32) -> u16 {
    let v = (imm as u16) & 0x3FF;
    let bit9 = (v >> 9) & 1;
    let bit4 = (v >> 4) & 1;
    let bit6 = (v >> 6) & 1;
    let bits8_7 = (v >> 7) & 3;
    let bit5 = (v >> 5) & 1;
    ci_type(
        0b011,
        bit9,
        2,
        (bit4 << 4) | (bit6 << 3) | (bits8_7 << 1) | bit5,
        C_OP_Q1,
    )
}

/// Extract unsigned word-aligned offset for c.lwsp: 0..252, multiple of 4.
fn ci_uimm_sp_lw(op: &Operand, span: Span) -> Result<u16, AsmError> {
    match op {
        Operand::Immediate(v) => {
            let v = *v as i64;
            if !(0..=252).contains(&v) || v % 4 != 0 {
                return Err(AsmError::ImmediateOverflow {
                    value: v as i128,
                    min: 0,
                    max: 252,
                    span,
                });
            }
            Ok(v as u16)
        }
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected word-aligned unsigned offset (0..252)"),
            span,
        }),
    }
}

/// Extract unsigned double-aligned offset for c.ldsp: 0..504, multiple of 8.
fn ci_uimm_sp_ld(op: &Operand, span: Span) -> Result<u16, AsmError> {
    match op {
        Operand::Immediate(v) => {
            let v = *v as i64;
            if !(0..=504).contains(&v) || v % 8 != 0 {
                return Err(AsmError::ImmediateOverflow {
                    value: v as i128,
                    min: 0,
                    max: 504,
                    span,
                });
            }
            Ok(v as u16)
        }
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected double-aligned unsigned offset (0..504)"),
            span,
        }),
    }
}

/// Extract a compact register + offset from a Memory operand for CL/CS-type.
/// The base register must be x8–x15.
fn compact_mem(op: &Operand, span: Span) -> Result<(u32, i32), AsmError> {
    match op {
        Operand::Memory(m) => {
            let base = m.base.ok_or_else(|| AsmError::InvalidOperands {
                detail: String::from("memory operand requires base register"),
                span,
            })?;
            if !base.is_riscv() {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("expected RISC-V register"),
                    span,
                });
            }
            let rn = base.rv_reg_num() as u32;
            let rp = compact_reg(rn).ok_or_else(|| AsmError::InvalidOperands {
                detail: String::from("base register must be x8-x15 for compressed load/store"),
                span,
            })?;
            let disp = m.disp;
            Ok((rp, disp as i32))
        }
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected memory operand"),
            span,
        }),
    }
}

/// Validate and return word-aligned CL/CS offset: 0..124, multiple of 4.
fn cl_offset_w(off: i32, span: Span) -> Result<u16, AsmError> {
    if !(0..=124).contains(&off) || off % 4 != 0 {
        return Err(AsmError::ImmediateOverflow {
            value: off as i128,
            min: 0,
            max: 124,
            span,
        });
    }
    Ok(off as u16)
}

/// Validate and return double-aligned CL/CS offset: 0..248, multiple of 8.
fn cl_offset_d(off: i32, span: Span) -> Result<u16, AsmError> {
    if !(0..=248).contains(&off) || off % 8 != 0 {
        return Err(AsmError::ImmediateOverflow {
            value: off as i128,
            min: 0,
            max: 248,
            span,
        });
    }
    Ok(off as u16)
}

/// Encode a CA-type (register arithmetic) instruction with compact reg validation.
fn ca_arith(ops: &[Operand], funct6: u16, funct2: u16, span: Span) -> Result<u16, AsmError> {
    let rd_p = compact_reg(reg(&ops[0], span)?).ok_or_else(|| AsmError::InvalidOperands {
        detail: String::from("rd/rs1 must be x8-x15 for compressed arithmetic"),
        span,
    })? as u16;
    let rs2_p = compact_reg(reg(&ops[1], span)?).ok_or_else(|| AsmError::InvalidOperands {
        detail: String::from("rs2 must be x8-x15 for compressed arithmetic"),
        span,
    })? as u16;
    Ok(ca_type(funct6, rd_p, funct2, rs2_p, C_OP_Q1))
}

/// Encode a CB-type shift instruction (c.srli / c.srai).
fn cb_shift(funct2: u16, rd_p: u16, shamt: u16) -> u16 {
    let bit5 = (shamt >> 5) & 1;
    (0b100u16 << 13) | (bit5 << 12) | (funct2 << 10) | (rd_p << 7) | ((shamt & 0x1F) << 2) | C_OP_Q1
}

/// Extract CB-type branch offset: ±256 bytes, must be even.
fn cb_offset(op: &Operand, span: Span) -> Result<i32, AsmError> {
    match op {
        Operand::Immediate(v) => {
            let v = *v as i64;
            if v % 2 != 0 || !(-256..=254).contains(&v) {
                return Err(AsmError::ImmediateOverflow {
                    value: v as i128,
                    min: -256,
                    max: 254,
                    span,
                });
            }
            Ok(v as i32)
        }
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected branch offset"),
            span,
        }),
    }
}

/// Extract CJ-type jump offset: ±2 KB, must be even.
fn cj_offset(op: &Operand, span: Span) -> Result<i32, AsmError> {
    match op {
        Operand::Immediate(v) => {
            let v = *v as i64;
            if v % 2 != 0 || !(-2048..=2046).contains(&v) {
                return Err(AsmError::ImmediateOverflow {
                    value: v as i128,
                    min: -2048,
                    max: 2046,
                    span,
                });
            }
            Ok(v as i32)
        }
        _ => Err(AsmError::InvalidOperands {
            detail: String::from("expected jump offset"),
            span,
        }),
    }
}

/// Build a relaxable compressed branch with label relocation (CB-type).
///
/// Short form (2 bytes, ±256 B): CB-type `c.beqz`/`c.bnez` with `RvCBranch8`.
/// Long form  (8 bytes, ±1 MiB): inverted B-type (`+8`) + `JAL x0, target`
/// with `RvJal20` relocation at offset 4.
///
/// `funct3`: 0b110 for c.beqz, 0b111 for c.bnez.
/// `rs1_p`: compact register index (0–7, maps to x8–x15).
fn rvc_branch_reloc(funct3: u16, rs1_p: u16, label: &str, addend: i64) -> EncodedInstr {
    // Short form: 2-byte CB-type placeholder.
    let hw = cb_type(funct3, rs1_p, 0);
    let mut short_bytes = InstrBytes::new();
    short_bytes.extend_from_slice(&hw.to_le_bytes());
    let short_relocation = Some(Relocation {
        offset: 0,
        size: 2,
        label: alloc::rc::Rc::from(label),
        kind: RelocKind::RvCBranch8,
        addend,
        trailing_bytes: 0,
    });

    // Long form: inverted B-type branch + JAL.
    // c.beqz (funct3=0b110) → BNE rs1,x0,+8 (funct3=0b001)
    // c.bnez (funct3=0b111) → BEQ rs1,x0,+8 (funct3=0b000)
    let rs1 = (rs1_p as u32) + 8; // un-compact: x8–x15
    let b_funct3: u32 = if funct3 == 0b110 { 0b001 } else { 0b000 }; // invert sense
    let inv_branch = b_type(OP_BRANCH, b_funct3, rs1, 0, 8); // skip over JAL
    let jal_placeholder = j_type(OP_JAL, 0, 0); // JAL x0, <target>
    let mut long_bytes = InstrBytes::new();
    long_bytes.extend_from_slice(&inv_branch.to_le_bytes());
    long_bytes.extend_from_slice(&jal_placeholder.to_le_bytes());

    EncodedInstr {
        bytes: long_bytes,
        relocation: Some(Relocation {
            offset: 4,
            size: 4,
            label: alloc::rc::Rc::from(label),
            kind: RelocKind::RvJal20,
            addend,
            trailing_bytes: 0,
        }),
        relax: Some(crate::encoder::RelaxInfo {
            short_bytes,
            short_reloc_offset: 0,
            short_relocation,
        }),
    }
}

/// Build a relaxable compressed jump with label relocation (CJ-type).
///
/// Short form (2 bytes, ±2 KB): CJ-type `c.j` with `RvCJump11`.
/// Long form  (4 bytes, ±1 MiB): `JAL x0, target` with `RvJal20`.
fn rvc_jump_reloc(label: &str, addend: i64) -> EncodedInstr {
    // Short form: 2-byte CJ-type placeholder.
    let hw = cj_type(0b101, 0);
    let mut short_bytes = InstrBytes::new();
    short_bytes.extend_from_slice(&hw.to_le_bytes());
    let short_relocation = Some(Relocation {
        offset: 0,
        size: 2,
        label: alloc::rc::Rc::from(label),
        kind: RelocKind::RvCJump11,
        addend,
        trailing_bytes: 0,
    });

    // Long form: JAL x0, <target> (4 bytes, ±1 MiB).
    let jal_placeholder = j_type(OP_JAL, 0, 0);
    let mut long_bytes = InstrBytes::new();
    long_bytes.extend_from_slice(&jal_placeholder.to_le_bytes());

    EncodedInstr {
        bytes: long_bytes,
        relocation: Some(Relocation {
            offset: 0,
            size: 4,
            label: alloc::rc::Rc::from(label),
            kind: RelocKind::RvJal20,
            addend,
            trailing_bytes: 0,
        }),
        relax: Some(crate::encoder::RelaxInfo {
            short_bytes,
            short_reloc_offset: 0,
            short_relocation,
        }),
    }
}

// ── Auto-narrowing: try to compress a 32-bit instruction to 16-bit ──────

/// Attempt to compress a standard 32-bit RISC-V instruction into a 16-bit
/// C-extension equivalent. Returns `Some(halfword)` if compression is
/// possible, `None` otherwise.
///
/// This implements automatic narrowing: when `.option rvc` is active (the
/// default), the assembler tries to compress every instruction before
/// falling through to the 32-bit encoder.
pub(crate) fn try_compress(
    mnemonic: &str,
    ops: &[Operand],
    is_rv64: bool,
    _span: Span,
) -> Option<u16> {
    match (mnemonic, ops) {
        // nop → c.nop
        ("nop", []) => Some(ci_type(0b000, 0, 0, 0, C_OP_Q1)),

        // addi rd, rd, imm → c.addi rd, imm  (rd != x0, imm != 0)
        ("addi", [Operand::Register(rd), Operand::Register(rs1), Operand::Immediate(imm)])
            if rd == rs1 && rd.is_riscv() && rd.rv_reg_num() != 0 =>
        {
            let v = *imm as i64;
            if v == 0 || !(-32..=31).contains(&v) {
                return None;
            }
            let rd_n = rd.rv_reg_num() as u16;
            let imm6 = (v as u16) & 0x3F;
            Some(ci_type(0b000, (imm6 >> 5) & 1, rd_n, imm6 & 0x1F, C_OP_Q1))
        }

        // addiw rd, rd, imm → c.addiw rd, imm (RV64, rd != x0)
        ("addiw", [Operand::Register(rd), Operand::Register(rs1), Operand::Immediate(imm)])
            if is_rv64 && rd == rs1 && rd.is_riscv() && rd.rv_reg_num() != 0 =>
        {
            let v = *imm as i64;
            if !(-32..=31).contains(&v) {
                return None;
            }
            let rd_n = rd.rv_reg_num() as u16;
            let imm6 = (v as u16) & 0x3F;
            Some(ci_type(0b001, (imm6 >> 5) & 1, rd_n, imm6 & 0x1F, C_OP_Q1))
        }

        // li (addi rd, x0, imm) → c.li rd, imm  (rd != x0)
        ("addi", [Operand::Register(rd), Operand::Register(rs1), Operand::Immediate(imm)])
            if rs1.is_riscv() && rs1.rv_reg_num() == 0 && rd.is_riscv() && rd.rv_reg_num() != 0 =>
        {
            let v = *imm as i64;
            if !(-32..=31).contains(&v) {
                return None;
            }
            let rd_n = rd.rv_reg_num() as u16;
            let imm6 = (v as u16) & 0x3F;
            Some(ci_type(0b010, (imm6 >> 5) & 1, rd_n, imm6 & 0x1F, C_OP_Q1))
        }

        // lui rd, imm → c.lui rd, imm  (rd != x0, x2, imm != 0)
        ("lui", [Operand::Register(rd), Operand::Immediate(imm)])
            if rd.is_riscv() && rd.rv_reg_num() != 0 && rd.rv_reg_num() != 2 =>
        {
            // LUI imm is already shifted left 12 in our representation? No—
            // the operand for LUI is the 20-bit upper value. For C.LUI, the
            // immediate is a 6-bit sign-extended value for nzimm[17:12].
            let v = *imm as i64;
            if v == 0 || !(-32..=31).contains(&v) {
                return None;
            }
            let rd_n = rd.rv_reg_num() as u16;
            let imm6 = (v as u16) & 0x3F;
            Some(ci_type(0b011, (imm6 >> 5) & 1, rd_n, imm6 & 0x1F, C_OP_Q1))
        }

        // mv rd, rs2 (add rd, x0, rs2) → c.mv rd, rs2  (rd != x0, rs2 != x0)
        ("add", [Operand::Register(rd), Operand::Register(rs1), Operand::Register(rs2)])
            if rs1.is_riscv()
                && rs1.rv_reg_num() == 0
                && rd.is_riscv()
                && rd.rv_reg_num() != 0
                && rs2.is_riscv()
                && rs2.rv_reg_num() != 0 =>
        {
            Some(cr_type(
                0b1000,
                rd.rv_reg_num() as u16,
                rs2.rv_reg_num() as u16,
                C_OP_Q2,
            ))
        }

        // add rd, rd, rs2 → c.add rd, rs2  (rd != x0, rs2 != x0)
        ("add", [Operand::Register(rd), Operand::Register(rs1), Operand::Register(rs2)])
            if rd == rs1
                && rd.is_riscv()
                && rd.rv_reg_num() != 0
                && rs2.is_riscv()
                && rs2.rv_reg_num() != 0 =>
        {
            Some(cr_type(
                0b1001,
                rd.rv_reg_num() as u16,
                rs2.rv_reg_num() as u16,
                C_OP_Q2,
            ))
        }

        // sub/xor/or/and rd, rd, rs2 → c.sub/c.xor/c.or/c.and (compact regs only)
        (
            "sub" | "xor" | "or" | "and",
            [Operand::Register(rd), Operand::Register(rs1), Operand::Register(rs2)],
        ) if rd == rs1 && rd.is_riscv() && rs2.is_riscv() => {
            let rd_p = compact_reg(rd.rv_reg_num() as u32)?;
            let rs2_p = compact_reg(rs2.rv_reg_num() as u32)?;
            let funct2 = match mnemonic {
                "sub" => 0b00,
                "xor" => 0b01,
                "or" => 0b10,
                "and" => 0b11,
                _ => return None,
            };
            Some(ca_type(
                0b100011,
                rd_p as u16,
                funct2,
                rs2_p as u16,
                C_OP_Q1,
            ))
        }

        // subw/addw rd, rd, rs2 → c.subw/c.addw (RV64, compact regs)
        (
            "subw" | "addw",
            [Operand::Register(rd), Operand::Register(rs1), Operand::Register(rs2)],
        ) if is_rv64 && rd == rs1 && rd.is_riscv() && rs2.is_riscv() => {
            let rd_p = compact_reg(rd.rv_reg_num() as u32)?;
            let rs2_p = compact_reg(rs2.rv_reg_num() as u32)?;
            let funct2 = if mnemonic == "subw" { 0b00 } else { 0b01 };
            Some(ca_type(
                0b100111,
                rd_p as u16,
                funct2,
                rs2_p as u16,
                C_OP_Q1,
            ))
        }

        // slli rd, rd, shamt → c.slli rd, shamt  (rd != x0, shamt > 0)
        ("slli", [Operand::Register(rd), Operand::Register(rs1), Operand::Immediate(shamt)])
            if rd == rs1 && rd.is_riscv() && rd.rv_reg_num() != 0 =>
        {
            let shamt_max: i64 = if is_rv64 { 63 } else { 31 };
            let s = *shamt as i64;
            if s < 1 || s > shamt_max {
                return None;
            }
            let rd_n = rd.rv_reg_num() as u16;
            let sv = s as u16;
            Some(ci_type(0b000, (sv >> 5) & 1, rd_n, sv & 0x1F, C_OP_Q2))
        }

        // srli rd, rd, shamt → c.srli (compact regs)
        ("srli", [Operand::Register(rd), Operand::Register(rs1), Operand::Immediate(shamt)])
            if rd == rs1 && rd.is_riscv() =>
        {
            let rd_p = compact_reg(rd.rv_reg_num() as u32)?;
            let shamt_max: i64 = if is_rv64 { 63 } else { 31 };
            let s = *shamt as i64;
            if s < 1 || s > shamt_max {
                return None;
            }
            Some(cb_shift(0b00, rd_p as u16, s as u16))
        }

        // srai rd, rd, shamt → c.srai (compact regs)
        ("srai", [Operand::Register(rd), Operand::Register(rs1), Operand::Immediate(shamt)])
            if rd == rs1 && rd.is_riscv() =>
        {
            let rd_p = compact_reg(rd.rv_reg_num() as u32)?;
            let shamt_max: i64 = if is_rv64 { 63 } else { 31 };
            let s = *shamt as i64;
            if s < 1 || s > shamt_max {
                return None;
            }
            Some(cb_shift(0b01, rd_p as u16, s as u16))
        }

        // andi rd, rd, imm → c.andi (compact regs, −32..31)
        ("andi", [Operand::Register(rd), Operand::Register(rs1), Operand::Immediate(imm)])
            if rd == rs1 && rd.is_riscv() =>
        {
            let rd_p = compact_reg(rd.rv_reg_num() as u32)?;
            let v = *imm as i64;
            if !(-32..=31).contains(&v) {
                return None;
            }
            let imm6 = (v as u16) & 0x3F;
            let bit5 = (imm6 >> 5) & 1;
            Some(
                (0b100u16 << 13)
                    | (bit5 << 12)
                    | (0b10u16 << 10)
                    | ((rd_p as u16) << 7)
                    | ((imm6 & 0x1F) << 2)
                    | C_OP_Q1,
            )
        }

        // jalr x0, rs1, 0 → c.jr rs1  (rs1 != x0)
        ("jalr", [Operand::Register(rd), Operand::Register(rs1), Operand::Immediate(imm)])
            if rd.is_riscv()
                && rd.rv_reg_num() == 0
                && rs1.is_riscv()
                && rs1.rv_reg_num() != 0
                && *imm == 0 =>
        {
            Some(cr_type(0b1000, rs1.rv_reg_num() as u16, 0, C_OP_Q2))
        }

        // jalr ra, rs1, 0 → c.jalr rs1  (rs1 != x0)
        ("jalr", [Operand::Register(rd), Operand::Register(rs1), Operand::Immediate(imm)])
            if rd.is_riscv()
                && rd.rv_reg_num() == 1
                && rs1.is_riscv()
                && rs1.rv_reg_num() != 0
                && *imm == 0 =>
        {
            Some(cr_type(0b1001, rs1.rv_reg_num() as u16, 0, C_OP_Q2))
        }

        // lw rd, offset(rs1) → c.lw rd', offset(rs1')  (compact regs, offset 0..124 word-aligned)
        ("lw", [Operand::Register(rd), Operand::Memory(m)])
            if rd.is_riscv() && m.base.is_some_and(|b| b.is_riscv()) =>
        {
            let rd_n = rd.rv_reg_num() as u32;
            let base = m.base?;
            let base_n = base.rv_reg_num() as u32;
            // SP-relative → c.lwsp (any rd except x0)
            if base_n == 2 && rd_n != 0 {
                let off = m.disp;
                if (0..=252).contains(&off) && off % 4 == 0 {
                    let off_u = off as u16;
                    let bit5 = (off_u >> 5) & 1;
                    let bits4_2 = (off_u >> 2) & 7;
                    let bits7_6 = (off_u >> 6) & 3;
                    return Some(ci_type(
                        0b010,
                        bit5,
                        rd_n as u16,
                        (bits4_2 << 2) | bits7_6,
                        C_OP_Q2,
                    ));
                }
            }
            // Compact regs only for non-SP path
            let rd_p = compact_reg(rd_n)?;
            let rs1_p = compact_reg(base_n)?;
            let off = m.disp as i32;
            if !(0..=124).contains(&off) || off % 4 != 0 {
                return None;
            }
            let off_u = off as u16;
            let bits5_3 = (off_u >> 3) & 7;
            let bit2 = (off_u >> 2) & 1;
            let bit6 = (off_u >> 6) & 1;
            Some(cl_type(
                0b010,
                bits5_3,
                rs1_p as u16,
                (bit6 << 1) | bit2,
                rd_p as u16,
                C_OP_Q0,
            ))
        }

        // sw rs2, offset(rs1) → c.sw/c.swsp
        ("sw", [Operand::Register(rs2), Operand::Memory(m)])
            if rs2.is_riscv() && m.base.is_some_and(|b| b.is_riscv()) =>
        {
            let rs2_n = rs2.rv_reg_num() as u32;
            let base = m.base?;
            let base_n = base.rv_reg_num() as u32;
            // SP-relative → c.swsp
            if base_n == 2 {
                let off = m.disp;
                if (0..=252).contains(&off) && off % 4 == 0 {
                    let off_u = off as u16;
                    let bits5_2 = (off_u >> 2) & 0xF;
                    let bits7_6 = (off_u >> 6) & 3;
                    return Some(css_type(
                        0b110,
                        (bits5_2 << 2) | bits7_6,
                        rs2_n as u16,
                        C_OP_Q2,
                    ));
                }
            }
            let rs2_p = compact_reg(rs2_n)?;
            let rs1_p = compact_reg(base_n)?;
            let off = m.disp as i32;
            if !(0..=124).contains(&off) || off % 4 != 0 {
                return None;
            }
            let off_u = off as u16;
            let bits5_3 = (off_u >> 3) & 7;
            let bit2 = (off_u >> 2) & 1;
            let bit6 = (off_u >> 6) & 1;
            Some(cs_type(
                0b110,
                bits5_3,
                rs1_p as u16,
                (bit6 << 1) | bit2,
                rs2_p as u16,
                C_OP_Q0,
            ))
        }

        // ld rd, offset(rs1) → c.ld/c.ldsp (RV64 only)
        ("ld", [Operand::Register(rd), Operand::Memory(m)])
            if is_rv64 && rd.is_riscv() && m.base.is_some_and(|b| b.is_riscv()) =>
        {
            let rd_n = rd.rv_reg_num() as u32;
            let base = m.base?;
            let base_n = base.rv_reg_num() as u32;
            if base_n == 2 && rd_n != 0 {
                let off = m.disp;
                if (0..=504).contains(&off) && off % 8 == 0 {
                    let off_u = off as u16;
                    let bit5 = (off_u >> 5) & 1;
                    let bits4_3 = (off_u >> 3) & 3;
                    let bits8_6 = (off_u >> 6) & 7;
                    return Some(ci_type(
                        0b011,
                        bit5,
                        rd_n as u16,
                        (bits4_3 << 3) | bits8_6,
                        C_OP_Q2,
                    ));
                }
            }
            let rd_p = compact_reg(rd_n)?;
            let rs1_p = compact_reg(base_n)?;
            let off = m.disp as i32;
            if !(0..=248).contains(&off) || off % 8 != 0 {
                return None;
            }
            let off_u = off as u16;
            let bits5_3 = (off_u >> 3) & 7;
            let bits7_6 = (off_u >> 6) & 3;
            Some(cl_type(
                0b011,
                bits5_3,
                rs1_p as u16,
                bits7_6,
                rd_p as u16,
                C_OP_Q0,
            ))
        }

        // sd rs2, offset(rs1) → c.sd/c.sdsp (RV64 only)
        ("sd", [Operand::Register(rs2), Operand::Memory(m)])
            if is_rv64 && rs2.is_riscv() && m.base.is_some_and(|b| b.is_riscv()) =>
        {
            let rs2_n = rs2.rv_reg_num() as u32;
            let base = m.base?;
            let base_n = base.rv_reg_num() as u32;
            if base_n == 2 {
                let off = m.disp;
                if (0..=504).contains(&off) && off % 8 == 0 {
                    let off_u = off as u16;
                    let bits5_3 = (off_u >> 3) & 7;
                    let bits8_6 = (off_u >> 6) & 7;
                    return Some(css_type(
                        0b111,
                        (bits5_3 << 3) | bits8_6,
                        rs2_n as u16,
                        C_OP_Q2,
                    ));
                }
            }
            let rs2_p = compact_reg(rs2_n)?;
            let rs1_p = compact_reg(base_n)?;
            let off = m.disp as i32;
            if !(0..=248).contains(&off) || off % 8 != 0 {
                return None;
            }
            let off_u = off as u16;
            let bits5_3 = (off_u >> 3) & 7;
            let bits7_6 = (off_u >> 6) & 3;
            Some(cs_type(
                0b111,
                bits5_3,
                rs1_p as u16,
                bits7_6,
                rs2_p as u16,
                C_OP_Q0,
            ))
        }

        // ebreak → c.ebreak
        ("ebreak", []) => Some(cr_type(0b1001, 0, 0, C_OP_Q2)),

        // ── Pseudo-instructions ──────────────────────────────────
        // jr rs1 → c.jr rs1 (rs1 != x0)
        ("jr", [Operand::Register(rs1)]) if rs1.is_riscv() && rs1.rv_reg_num() != 0 => {
            Some(cr_type(0b1000, rs1.rv_reg_num() as u16, 0, C_OP_Q2))
        }

        // ret → c.jr ra
        ("ret", []) => Some(cr_type(0b1000, 1, 0, C_OP_Q2)),

        // li rd, imm → c.li rd, imm (rd != x0, imm in [-32,31])
        ("li", [Operand::Register(rd), Operand::Immediate(imm)])
            if rd.is_riscv() && rd.rv_reg_num() != 0 =>
        {
            let v = *imm as i64;
            if !(-32..=31).contains(&v) {
                return None;
            }
            let rd_n = rd.rv_reg_num() as u16;
            let imm6 = (v as u16) & 0x3F;
            Some(ci_type(0b010, (imm6 >> 5) & 1, rd_n, imm6 & 0x1F, C_OP_Q1))
        }

        // mv rd, rs2 → c.mv rd, rs2 (rd != x0, rs2 != x0)
        ("mv", [Operand::Register(rd), Operand::Register(rs2)])
            if rd.is_riscv() && rd.rv_reg_num() != 0 && rs2.is_riscv() && rs2.rv_reg_num() != 0 =>
        {
            Some(cr_type(
                0b1000,
                rd.rv_reg_num() as u16,
                rs2.rv_reg_num() as u16,
                C_OP_Q2,
            ))
        }

        // not rd, rs → no compressed form
        // neg rd, rs → no compressed form
        // seqz/snez/sltz/sgtz → no compressed form
        _ => None,
    }
}

/// Build a 2-byte `EncodedInstr` from a compressed halfword.
#[inline]
pub(crate) fn rvc_instr(hw: u16) -> EncodedInstr {
    let mut bytes = InstrBytes::new();
    bytes.extend_from_slice(&hw.to_le_bytes());
    EncodedInstr {
        bytes,
        relocation: None,
        relax: None,
    }
}

/// Build a relaxable B-type branch targeting a label.
///
/// Short form (4 bytes, ±4 KiB): plain B-type with `RvBranch12` relocation.
/// Long form  (8 bytes, ±1 MiB): inverted B-type (`+8`) + `JAL x0, target`
/// with `RvJal20` relocation at offset 4.
fn relaxable_branch(funct3: u32, rs1: u32, rs2: u32, label: &str, addend: i64) -> EncodedInstr {
    // Short form: B-type branch placeholder (offset patched by linker).
    let short_word = b_type(OP_BRANCH, funct3, rs1, rs2, 0);
    let mut short_bytes = InstrBytes::new();
    short_bytes.extend_from_slice(&short_word.to_le_bytes());
    let short_relocation = Some(Relocation {
        offset: 0,
        size: 4,
        label: alloc::rc::Rc::from(label),
        kind: RelocKind::RvBranch12,
        addend,
        trailing_bytes: 0,
    });
    // Long form: inverted condition skips over a JAL.
    let inv_funct3 = funct3 ^ 1; // beq↔bne, blt↔bge, bltu↔bgeu
    let inv_branch = b_type(OP_BRANCH, inv_funct3, rs1, rs2, 8);
    let jal_placeholder = j_type(OP_JAL, 0, 0); // JAL x0, <target>
    let mut long_bytes = InstrBytes::new();
    long_bytes.extend_from_slice(&inv_branch.to_le_bytes());
    long_bytes.extend_from_slice(&jal_placeholder.to_le_bytes());
    EncodedInstr {
        bytes: long_bytes,
        relocation: Some(Relocation {
            offset: 4,
            size: 4,
            label: alloc::rc::Rc::from(label),
            kind: RelocKind::RvJal20,
            addend,
            trailing_bytes: 0,
        }),
        relax: Some(crate::encoder::RelaxInfo {
            short_bytes,
            short_reloc_offset: 0,
            short_relocation,
        }),
    }
}

// ── Shifts (I-type with special immediate encoding) ─────────────────────

#[allow(clippy::too_many_arguments)]
fn encode_shift_imm(
    rd: u32,
    rs1: u32,
    shamt: i32,
    funct3: u32,
    high_bits: u32,
    opcode: u32,
    is_rv64: bool,
    span: Span,
) -> Result<u32, AsmError> {
    let max_shamt = if is_rv64 { 63 } else { 31 };
    if shamt < 0 || shamt > max_shamt {
        return Err(AsmError::ImmediateOverflow {
            value: shamt as i128,
            min: 0,
            max: if is_rv64 { 63 } else { 31 },
            span,
        });
    }
    // high_bits is the raw upper immediate bits (e.g. 0x400 for SRAI, bit 10).
    let shamt_mask: u32 = if is_rv64 { 0x3F } else { 0x1F };
    let imm = high_bits | (shamt as u32 & shamt_mask);
    Ok(i_type(opcode, rd, funct3, rs1, imm as i32))
}

// ── Main encoder ─────────────────────────────────────────────────────────

/// Encode a single RISC-V instruction.
///
/// # Errors
///
/// Returns `Err(AsmError)` if the mnemonic is unknown, operand combination
/// is invalid, or an immediate is out of range.
pub fn encode_riscv(instr: &Instruction, arch: Arch) -> Result<EncodedInstr, AsmError> {
    let span = instr.span;
    let mnemonic = instr.mnemonic.as_str();
    let ops = &instr.operands;
    let is_rv64 = arch == Arch::Rv64;

    // ── Explicit C-extension mnemonics (c.xxx) ──────────────
    if mnemonic.starts_with("c.") {
        return encode_rvc_explicit(mnemonic, ops, is_rv64, span);
    }

    // ── Operand count validation ────────────────────────────
    // Guard against index-out-of-bounds panics: validate that the
    // operand count meets minimum requirements for the mnemonic.
    let min_ops = match mnemonic {
        "ecall" | "ebreak" | "mret" | "sret" | "wfi" | "nop" | "ret" | "fence" | "fence.i"
        | "sfence.vma" => 0,
        "j" | "jr" | "call" | "tail" | "rdcycle" | "rdtime" | "rdinstret" | "rdcycleh"
        | "rdtimeh" | "rdinstreth" => 1,
        _ if mnemonic.starts_with("csrr") && mnemonic.len() == 4 => 2,
        _ => {
            // Most instructions need at least 1 operand; R-type need 3.
            // We conservatively require 1. Individual arms handle specific
            // counts where the helper functions (`reg`, `imm_*`) produce
            // proper errors for wrong operand types.
            1
        }
    };
    if ops.len() < min_ops {
        return Err(AsmError::InvalidOperands {
            detail: alloc::format!(
                "'{}' requires at least {} operand(s), got {}",
                mnemonic,
                min_ops,
                ops.len()
            ),
            span,
        });
    }

    let word = match mnemonic {
        // ── U-type ────────────────────────────────────────────
        "lui" => {
            let rd = reg(&ops[0], span)?;
            let imm = imm_u(&ops[1], span)?;
            u_type(OP_LUI, rd, imm)
        }

        "auipc" => {
            let rd = reg(&ops[0], span)?;
            if let Some((label, addend)) = extract_label(&ops[1]) {
                // AUIPC with label — emit relocation
                let word = u_type(OP_AUIPC, rd, 0);
                let bytes = InstrBytes::from_slice(&word.to_le_bytes());
                return Ok(EncodedInstr {
                    bytes,
                    relocation: Some(Relocation {
                        offset: 0,
                        size: 4,
                        label: alloc::rc::Rc::from(label),
                        kind: RelocKind::RvAuipc20,
                        addend,
                        trailing_bytes: 0,
                    }),
                    relax: None,
                });
            }
            let imm = imm_u(&ops[1], span)?;
            u_type(OP_AUIPC, rd, imm)
        }

        // ── J-type (JAL) ──────────────────────────────────────
        "jal" => {
            match ops.len() {
                // jal label → rd=ra(x1)
                1 => {
                    if let Some((label, addend)) = extract_label(&ops[0]) {
                        let word = j_type(OP_JAL, 1, 0); // rd=x1 (ra)
                        let bytes = InstrBytes::from_slice(&word.to_le_bytes());
                        return Ok(EncodedInstr {
                            bytes,
                            relocation: Some(Relocation {
                                offset: 0,
                                size: 4,
                                label: alloc::rc::Rc::from(label),
                                kind: RelocKind::RvJal20,
                                addend,
                                trailing_bytes: 0,
                            }),
                            relax: None,
                        });
                    }
                    let imm = imm_i(&ops[0], span)?; // small immediate jump
                    j_type(OP_JAL, 1, imm)
                }
                // jal rd, label
                2 => {
                    let rd = reg(&ops[0], span)?;
                    if let Some((label, addend)) = extract_label(&ops[1]) {
                        let word = j_type(OP_JAL, rd, 0);
                        let bytes = InstrBytes::from_slice(&word.to_le_bytes());
                        return Ok(EncodedInstr {
                            bytes,
                            relocation: Some(Relocation {
                                offset: 0,
                                size: 4,
                                label: alloc::rc::Rc::from(label),
                                kind: RelocKind::RvJal20,
                                addend,
                                trailing_bytes: 0,
                            }),
                            relax: None,
                        });
                    }
                    let imm = imm_i(&ops[1], span)?;
                    j_type(OP_JAL, rd, imm)
                }
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("jal expects 1 or 2 operands"),
                        span,
                    })
                }
            }
        }

        // ── I-type (JALR) ─────────────────────────────────────
        "jalr" => {
            match ops.len() {
                // jalr rs1 → jalr ra, rs1, 0
                1 => {
                    let rs1 = reg(&ops[0], span)?;
                    i_type(OP_JALR, 1, 0, rs1, 0)
                }
                // jalr rd, rs1 → jalr rd, rs1, 0
                // jalr rd, offset(rs1)
                2 => {
                    let rd = reg(&ops[0], span)?;
                    // Try register first (jalr rd, rs1)
                    if let Ok(rs1) = reg(&ops[1], span) {
                        i_type(OP_JALR, rd, 0, rs1, 0)
                    } else {
                        // Memory operand: jalr rd, offset(rs1)
                        let (rs1, imm) = mem(&ops[1], span)?;
                        if !(-2048..=2047).contains(&imm) {
                            return Err(AsmError::ImmediateOverflow {
                                value: imm as i128,
                                min: -(1 << 11),
                                max: (1 << 11) - 1,
                                span,
                            });
                        }
                        i_type(OP_JALR, rd, 0, rs1, imm)
                    }
                }
                // jalr rd, rs1, imm
                3 => {
                    let rd = reg(&ops[0], span)?;
                    let rs1 = reg(&ops[1], span)?;
                    let imm = imm_i(&ops[2], span)?;
                    i_type(OP_JALR, rd, 0, rs1, imm)
                }
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("jalr expects 1, 2, or 3 operands"),
                        span,
                    })
                }
            }
        }

        // ── B-type branches ──────────────────────────────────
        "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" => {
            let funct3 = match mnemonic {
                "beq" => 0b000,
                "bne" => 0b001,
                "blt" => 0b100,
                "bge" => 0b101,
                "bltu" => 0b110,
                "bgeu" => 0b111,
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!("unhandled branch mnemonic '{}'", mnemonic),
                        span,
                    })
                }
            };
            let rs1 = reg(&ops[0], span)?;
            let rs2 = reg(&ops[1], span)?;
            if let Some((label, addend)) = extract_label(&ops[2]) {
                return Ok(relaxable_branch(funct3, rs1, rs2, label, addend));
            }
            let imm = imm_i(&ops[2], span)?;
            b_type(OP_BRANCH, funct3, rs1, rs2, imm)
        }

        // ── Loads (I-type) ───────────────────────────────────
        "lb" | "lh" | "lw" | "ld" | "lbu" | "lhu" | "lwu" => {
            let funct3 = match mnemonic {
                "lb" => 0b000,
                "lh" => 0b001,
                "lw" => 0b010,
                "ld" => 0b011,
                "lbu" => 0b100,
                "lhu" => 0b101,
                "lwu" => 0b110,
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!("unhandled load mnemonic '{}'", mnemonic),
                        span,
                    })
                }
            };
            // ld is RV64I only
            if matches!(mnemonic, "ld" | "lwu") && !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!("'{}' is only available in RV64I", mnemonic),
                    span,
                });
            }
            let rd = reg(&ops[0], span)?;
            let (rs1, imm) = mem(&ops[1], span)?;
            if !(-2048..=2047).contains(&imm) {
                return Err(AsmError::ImmediateOverflow {
                    value: imm as i128,
                    min: -(1 << 11),
                    max: (1 << 11) - 1,
                    span,
                });
            }
            i_type(OP_LOAD, rd, funct3, rs1, imm)
        }

        // ── Stores (S-type) ──────────────────────────────────
        "sb" | "sh" | "sw" | "sd" => {
            let funct3 = match mnemonic {
                "sb" => 0b000,
                "sh" => 0b001,
                "sw" => 0b010,
                "sd" => 0b011,
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!("unhandled store mnemonic '{}'", mnemonic),
                        span,
                    })
                }
            };
            if mnemonic == "sd" && !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'sd' is only available in RV64I"),
                    span,
                });
            }
            let rs2 = reg(&ops[0], span)?;
            let (rs1, imm) = mem(&ops[1], span)?;
            if !(-2048..=2047).contains(&imm) {
                return Err(AsmError::ImmediateOverflow {
                    value: imm as i128,
                    min: -(1 << 11),
                    max: (1 << 11) - 1,
                    span,
                });
            }
            s_type(OP_STORE, funct3, rs1, rs2, imm)
        }

        // ── I-type ALU immediates ────────────────────────────
        "addi" | "slti" | "sltiu" | "xori" | "ori" | "andi" => {
            let funct3 = match mnemonic {
                "addi" => 0b000,
                "slti" => 0b010,
                "sltiu" => 0b011,
                "xori" => 0b100,
                "ori" => 0b110,
                "andi" => 0b111,
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!("unhandled ALU-immediate mnemonic '{}'", mnemonic),
                        span,
                    })
                }
            };
            let rd = reg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            let imm = imm_i(&ops[2], span)?;
            i_type(OP_IMM, rd, funct3, rs1, imm)
        }

        // ── Shift immediates ─────────────────────────────────
        "slli" | "srli" | "srai" => {
            let (funct3, high_bits) = match mnemonic {
                "slli" => (0b001, 0x000),
                "srli" => (0b101, 0x000),
                "srai" => (0b101, 0x400),
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!("unhandled shift-immediate mnemonic '{}'", mnemonic),
                        span,
                    })
                }
            };
            let rd = reg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            let shamt = imm_i(&ops[2], span)?;
            encode_shift_imm(rd, rs1, shamt, funct3, high_bits, OP_IMM, is_rv64, span)?
        }

        // ── R-type ALU register ──────────────────────────────
        "add" | "sub" | "sll" | "slt" | "sltu" | "xor" | "srl" | "sra" | "or" | "and" => {
            let (funct3, funct7) = match mnemonic {
                "add" => (0b000, 0b000_0000),
                "sub" => (0b000, 0b010_0000),
                "sll" => (0b001, 0b000_0000),
                "slt" => (0b010, 0b000_0000),
                "sltu" => (0b011, 0b000_0000),
                "xor" => (0b100, 0b000_0000),
                "srl" => (0b101, 0b000_0000),
                "sra" => (0b101, 0b010_0000),
                "or" => (0b110, 0b000_0000),
                "and" => (0b111, 0b000_0000),
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!("unhandled ALU-register mnemonic '{}'", mnemonic),
                        span,
                    })
                }
            };
            let rd = reg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            let rs2 = reg(&ops[2], span)?;
            r_type(OP_REG, rd, funct3, rs1, rs2, funct7)
        }

        // ── RV64I W-suffix immediates ────────────────────────
        "addiw" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'addiw' is only available in RV64I"),
                    span,
                });
            }
            let rd = reg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            let imm = imm_i(&ops[2], span)?;
            i_type(OP_IMM_W, rd, 0b000, rs1, imm)
        }

        "slliw" | "srliw" | "sraiw" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!("'{}' is only available in RV64I", mnemonic),
                    span,
                });
            }
            let (funct3, high_bits) = match mnemonic {
                "slliw" => (0b001, 0x000),
                "srliw" => (0b101, 0x000),
                "sraiw" => (0b101, 0x400),
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!(
                            "unhandled RV64I shift-immediate mnemonic '{}'",
                            mnemonic
                        ),
                        span,
                    })
                }
            };
            let rd = reg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            let shamt = imm_i(&ops[2], span)?;
            // W-suffix shift uses 5-bit shamt always
            encode_shift_imm(rd, rs1, shamt, funct3, high_bits, OP_IMM_W, false, span)?
        }

        // ── RV64I W-suffix register ──────────────────────────
        "addw" | "subw" | "sllw" | "srlw" | "sraw" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!("'{}' is only available in RV64I", mnemonic),
                    span,
                });
            }
            let (funct3, funct7) = match mnemonic {
                "addw" => (0b000, 0b000_0000),
                "subw" => (0b000, 0b010_0000),
                "sllw" => (0b001, 0b000_0000),
                "srlw" => (0b101, 0b000_0000),
                "sraw" => (0b101, 0b010_0000),
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!("unhandled RV64I register mnemonic '{}'", mnemonic),
                        span,
                    })
                }
            };
            let rd = reg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            let rs2 = reg(&ops[2], span)?;
            r_type(OP_REG_W, rd, funct3, rs1, rs2, funct7)
        }

        // ── M extension (multiply/divide) ────────────────────
        "mul" | "mulh" | "mulhsu" | "mulhu" | "div" | "divu" | "rem" | "remu" => {
            let funct3 = match mnemonic {
                "mul" => 0b000,
                "mulh" => 0b001,
                "mulhsu" => 0b010,
                "mulhu" => 0b011,
                "div" => 0b100,
                "divu" => 0b101,
                "rem" => 0b110,
                "remu" => 0b111,
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!("unhandled M-extension mnemonic '{}'", mnemonic),
                        span,
                    })
                }
            };
            let rd = reg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            let rs2 = reg(&ops[2], span)?;
            r_type(OP_REG, rd, funct3, rs1, rs2, 0b000_0001)
        }

        // ── RV64M W-suffix multiply/divide ───────────────────
        "mulw" | "divw" | "divuw" | "remw" | "remuw" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!("'{}' is only available in RV64", mnemonic),
                    span,
                });
            }
            let funct3 = match mnemonic {
                "mulw" => 0b000,
                "divw" => 0b100,
                "divuw" => 0b101,
                "remw" => 0b110,
                "remuw" => 0b111,
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!("unhandled RV64M mnemonic '{}'", mnemonic),
                        span,
                    })
                }
            };
            let rd = reg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            let rs2 = reg(&ops[2], span)?;
            r_type(OP_REG_W, rd, funct3, rs1, rs2, 0b000_0001)
        }

        // ── System ───────────────────────────────────────────
        "ecall" => i_type(OP_SYSTEM, 0, 0, 0, 0),
        "ebreak" => i_type(OP_SYSTEM, 0, 0, 0, 1),

        // ── Privileged instructions ──────────────────────────
        "mret" => {
            // MRET: 0011000_00010_00000_000_00000_1110011
            i_type(OP_SYSTEM, 0, 0, 0, 0x302)
        }
        "sret" => {
            // SRET: 0001000_00010_00000_000_00000_1110011
            i_type(OP_SYSTEM, 0, 0, 0, 0x102)
        }
        "wfi" => {
            // WFI: 0001000_00101_00000_000_00000_1110011
            i_type(OP_SYSTEM, 0, 0, 0, 0x105)
        }
        "sfence.vma" => {
            // SFENCE.VMA rs1, rs2
            // 0001001_rs2_rs1_000_00000_1110011
            let (rs1, rs2) = if ops.is_empty() {
                (0u32, 0u32)
            } else if ops.len() == 2 {
                (reg(&ops[0], span)?, reg(&ops[1], span)?)
            } else {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("sfence.vma expects 0 or 2 register operands"),
                    span,
                });
            };
            // Encoded as R-type: funct7=0001001, rs2, rs1, funct3=000, rd=0, opcode=SYSTEM
            r_type(OP_SYSTEM, 0, 0, rs1, rs2, 0b000_1001)
        }

        // ── FENCE ────────────────────────────────────────────
        "fence" => {
            // Default fence: iorw, iorw
            // Full fence: pred=0b1111, succ=0b1111
            i_type(OP_FENCE, 0, 0b000, 0, 0x0FF)
        }
        "fence.i" => i_type(OP_FENCE, 0, 0b001, 0, 0),

        // ── NOP (pseudo) ─────────────────────────────────────
        "nop" => {
            i_type(OP_IMM, 0, 0, 0, 0) // addi x0, x0, 0
        }

        // ── Pseudo-instructions ──────────────────────────────

        // li rd, imm → for small values: addi rd, x0, imm
        //              for large values: lui + addi pair
        "li" => {
            let rd = reg(&ops[0], span)?;
            let val = match &ops[1] {
                Operand::Immediate(v) => *v as i64,
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("li expects an immediate value"),
                        span,
                    })
                }
            };

            // Small immediate fits in 12-bit signed: addi rd, x0, imm
            if (-2048..=2047).contains(&val) {
                i_type(OP_IMM, rd, 0, 0, val as i32)
            } else if (-2_147_483_648i64..=2_147_483_647i64).contains(&val)
                || (!is_rv64 && (0i64..=4_294_967_295i64).contains(&val))
            {
                // Fits in 32-bit signed, or unsigned 32-bit on RV32 where
                // registers don't sign-extend past bit 31.  Values in
                // [2^31, 2^32-1] on RV32 are treated as their two's
                // complement equivalent (e.g. 0xFFFFFFFF → -1).
                let v32 = val as i32;
                let lo12 = (v32 << 20) >> 20;
                let hi20 = v32.wrapping_sub(lo12) as u32 & 0xFFFF_F000;
                if hi20 == 0 {
                    // Value fits in 12-bit signed after truncation to 32 bits.
                    i_type(OP_IMM, rd, 0, 0, lo12)
                } else if lo12 == 0 {
                    // Exact multiple of 0x1000 — LUI alone suffices.
                    u_type(OP_LUI, rd, hi20)
                } else {
                    let w1 = u_type(OP_LUI, rd, hi20);
                    let w2 = i_type(OP_IMM, rd, 0, rd, lo12);
                    let mut bytes = InstrBytes::new();
                    bytes.extend_from_slice(&w1.to_le_bytes());
                    bytes.extend_from_slice(&w2.to_le_bytes());
                    return Ok(EncodedInstr {
                        bytes,
                        relocation: None,
                        relax: None,
                    });
                }
            } else if !is_rv64 {
                return Err(AsmError::ImmediateOverflow {
                    value: val as i128,
                    min: i32::MIN as i128,
                    max: u32::MAX as i128,
                    span,
                });
            } else {
                // RV64: full 64-bit immediate loading via multi-instruction sequence.
                // Strategy: load upper 32 bits, shift left 32, add lower 32 bits.
                // This matches the GAS li expansion for RV64.
                return Ok(encode_li_rv64(rd, val));
            }
        }

        // mv rd, rs → addi rd, rs, 0
        "mv" => {
            let rd = reg(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            i_type(OP_IMM, rd, 0, rs, 0)
        }

        // not rd, rs → xori rd, rs, -1
        "not" => {
            let rd = reg(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            i_type(OP_IMM, rd, 0b100, rs, -1)
        }

        // neg rd, rs → sub rd, x0, rs
        "neg" => {
            let rd = reg(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            r_type(OP_REG, rd, 0, 0, rs, 0b010_0000)
        }

        // negw rd, rs → subw rd, x0, rs (RV64)
        "negw" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'negw' is only available in RV64"),
                    span,
                });
            }
            let rd = reg(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            r_type(OP_REG_W, rd, 0, 0, rs, 0b010_0000)
        }

        // seqz rd, rs → sltiu rd, rs, 1
        "seqz" => {
            let rd = reg(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            i_type(OP_IMM, rd, 0b011, rs, 1)
        }

        // snez rd, rs → sltu rd, x0, rs
        "snez" => {
            let rd = reg(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            r_type(OP_REG, rd, 0b011, 0, rs, 0)
        }

        // sltz rd, rs → slt rd, rs, x0
        "sltz" => {
            let rd = reg(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            r_type(OP_REG, rd, 0b010, rs, 0, 0)
        }

        // sgtz rd, rs → slt rd, x0, rs
        "sgtz" => {
            let rd = reg(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            r_type(OP_REG, rd, 0b010, 0, rs, 0)
        }

        // j offset → jal x0, offset
        "j" => {
            if let Some((label, addend)) = extract_label(&ops[0]) {
                let word = j_type(OP_JAL, 0, 0);
                let bytes = InstrBytes::from_slice(&word.to_le_bytes());
                return Ok(EncodedInstr {
                    bytes,
                    relocation: Some(Relocation {
                        offset: 0,
                        size: 4,
                        label: alloc::rc::Rc::from(label),
                        kind: RelocKind::RvJal20,
                        addend,
                        trailing_bytes: 0,
                    }),
                    relax: None,
                });
            }
            let imm = imm_i(&ops[0], span)?;
            j_type(OP_JAL, 0, imm)
        }

        // jr rs → jalr x0, rs, 0
        "jr" => {
            let rs = reg(&ops[0], span)?;
            i_type(OP_JALR, 0, 0, rs, 0)
        }

        // ret → jalr x0, ra, 0
        "ret" => {
            i_type(OP_JALR, 0, 0, 1, 0) // x0, x1(ra), 0
        }

        // call label → auipc ra, 0 + jalr ra, ra, 0 (2-instruction sequence)
        "call" => {
            if let Some((label, addend)) = extract_label(&ops[0]) {
                let w1 = u_type(OP_AUIPC, 1, 0); // auipc ra, %hi(label)
                let w2 = i_type(OP_JALR, 1, 0, 1, 0); // jalr ra, ra, %lo(label)
                let mut bytes = InstrBytes::new();
                bytes.extend_from_slice(&w1.to_le_bytes());
                bytes.extend_from_slice(&w2.to_le_bytes());
                return Ok(EncodedInstr {
                    bytes,
                    relocation: Some(Relocation {
                        offset: 0,
                        size: 4,
                        label: alloc::rc::Rc::from(label),
                        kind: RelocKind::RvAuipc20,
                        addend,
                        trailing_bytes: 4, // the JALR follows
                    }),
                    relax: None,
                });
            }
            return Err(AsmError::InvalidOperands {
                detail: String::from("'call' requires a label operand"),
                span,
            });
        }

        // tail label → auipc t1, 0 + jalr x0, t1, 0 (2-instruction sequence)
        "tail" => {
            if let Some((label, addend)) = extract_label(&ops[0]) {
                let w1 = u_type(OP_AUIPC, 6, 0); // auipc t1(x6), %hi(label)
                let w2 = i_type(OP_JALR, 0, 0, 6, 0); // jalr x0, t1(x6), %lo(label)
                let mut bytes = InstrBytes::new();
                bytes.extend_from_slice(&w1.to_le_bytes());
                bytes.extend_from_slice(&w2.to_le_bytes());
                return Ok(EncodedInstr {
                    bytes,
                    relocation: Some(Relocation {
                        offset: 0,
                        size: 4,
                        label: alloc::rc::Rc::from(label),
                        kind: RelocKind::RvAuipc20,
                        addend,
                        trailing_bytes: 4,
                    }),
                    relax: None,
                });
            }
            return Err(AsmError::InvalidOperands {
                detail: String::from("'tail' requires a label operand"),
                span,
            });
        }

        // Branch pseudo-instructions
        // beqz rs, label → beq rs, x0, label
        "beqz" => {
            let rs = reg(&ops[0], span)?;
            if let Some((label, addend)) = extract_label(&ops[1]) {
                return Ok(relaxable_branch(0b000, rs, 0, label, addend));
            }
            let imm = imm_i(&ops[1], span)?;
            b_type(OP_BRANCH, 0b000, rs, 0, imm)
        }

        // bnez rs, label → bne rs, x0, label
        "bnez" => {
            let rs = reg(&ops[0], span)?;
            if let Some((label, addend)) = extract_label(&ops[1]) {
                return Ok(relaxable_branch(0b001, rs, 0, label, addend));
            }
            let imm = imm_i(&ops[1], span)?;
            b_type(OP_BRANCH, 0b001, rs, 0, imm)
        }

        // blez rs, label → bge x0, rs, label
        "blez" => {
            let rs = reg(&ops[0], span)?;
            if let Some((label, addend)) = extract_label(&ops[1]) {
                return Ok(relaxable_branch(0b101, 0, rs, label, addend));
            }
            let imm = imm_i(&ops[1], span)?;
            b_type(OP_BRANCH, 0b101, 0, rs, imm)
        }

        // bgez rs, label → bge rs, x0, label
        "bgez" => {
            let rs = reg(&ops[0], span)?;
            if let Some((label, addend)) = extract_label(&ops[1]) {
                return Ok(relaxable_branch(0b101, rs, 0, label, addend));
            }
            let imm = imm_i(&ops[1], span)?;
            b_type(OP_BRANCH, 0b101, rs, 0, imm)
        }

        // bltz rs, label → blt rs, x0, label
        "bltz" => {
            let rs = reg(&ops[0], span)?;
            if let Some((label, addend)) = extract_label(&ops[1]) {
                return Ok(relaxable_branch(0b100, rs, 0, label, addend));
            }
            let imm = imm_i(&ops[1], span)?;
            b_type(OP_BRANCH, 0b100, rs, 0, imm)
        }

        // bgtz rs, label → blt x0, rs, label
        "bgtz" => {
            let rs = reg(&ops[0], span)?;
            if let Some((label, addend)) = extract_label(&ops[1]) {
                return Ok(relaxable_branch(0b100, 0, rs, label, addend));
            }
            let imm = imm_i(&ops[1], span)?;
            b_type(OP_BRANCH, 0b100, 0, rs, imm)
        }

        // Reverse-comparison branch aliases
        // bgt rs, rt, label → blt rt, rs, label
        "bgt" => {
            let rs = reg(&ops[0], span)?;
            let rt = reg(&ops[1], span)?;
            if let Some((label, addend)) = extract_label(&ops[2]) {
                return Ok(relaxable_branch(0b100, rt, rs, label, addend));
            }
            let imm = imm_i(&ops[2], span)?;
            b_type(OP_BRANCH, 0b100, rt, rs, imm)
        }

        // ble rs, rt, label → bge rt, rs, label
        "ble" => {
            let rs = reg(&ops[0], span)?;
            let rt = reg(&ops[1], span)?;
            if let Some((label, addend)) = extract_label(&ops[2]) {
                return Ok(relaxable_branch(0b101, rt, rs, label, addend));
            }
            let imm = imm_i(&ops[2], span)?;
            b_type(OP_BRANCH, 0b101, rt, rs, imm)
        }

        // bgtu rs, rt, label → bltu rt, rs, label
        "bgtu" => {
            let rs = reg(&ops[0], span)?;
            let rt = reg(&ops[1], span)?;
            if let Some((label, addend)) = extract_label(&ops[2]) {
                return Ok(relaxable_branch(0b110, rt, rs, label, addend));
            }
            let imm = imm_i(&ops[2], span)?;
            b_type(OP_BRANCH, 0b110, rt, rs, imm)
        }

        // bleu rs, rt, label → bgeu rt, rs, label
        "bleu" => {
            let rs = reg(&ops[0], span)?;
            let rt = reg(&ops[1], span)?;
            if let Some((label, addend)) = extract_label(&ops[2]) {
                return Ok(relaxable_branch(0b111, rt, rs, label, addend));
            }
            let imm = imm_i(&ops[2], span)?;
            b_type(OP_BRANCH, 0b111, rt, rs, imm)
        }

        // CSR instructions (with named CSR support)
        "csrrw" | "csrrs" | "csrrc" => {
            let funct3 = match mnemonic {
                "csrrw" => 0b001,
                "csrrs" => 0b010,
                "csrrc" => 0b011,
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!("unhandled CSR mnemonic '{}'", mnemonic),
                        span,
                    })
                }
            };
            let rd = reg(&ops[0], span)?;
            let csr = csr_operand(&ops[1], span)?;
            let rs1 = reg(&ops[2], span)?;
            (csr << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OP_SYSTEM
        }

        "csrrwi" | "csrrsi" | "csrrci" => {
            let funct3 = match mnemonic {
                "csrrwi" => 0b101,
                "csrrsi" => 0b110,
                "csrrci" => 0b111,
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: alloc::format!("unhandled CSR-immediate mnemonic '{}'", mnemonic),
                        span,
                    })
                }
            };
            let rd = reg(&ops[0], span)?;
            let csr = csr_operand(&ops[1], span)?;
            let uimm = match &ops[2] {
                Operand::Immediate(v) => {
                    let v = *v as u32;
                    if v > 31 {
                        return Err(AsmError::ImmediateOverflow {
                            value: v as i128,
                            min: 0,
                            max: 31,
                            span,
                        });
                    }
                    v
                }
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("expected 5-bit unsigned immediate for CSR"),
                        span,
                    })
                }
            };
            (csr << 20) | (uimm << 15) | (funct3 << 12) | (rd << 7) | OP_SYSTEM
        }

        // ── CSR pseudo-instructions ──────────────────────────
        // csrr rd, csr → csrrs rd, csr, x0
        "csrr" => {
            let rd = reg(&ops[0], span)?;
            let csr = csr_operand(&ops[1], span)?;
            (csr << 20) | (0b010 << 12) | (rd << 7) | OP_SYSTEM
        }
        // csrw csr, rs → csrrw x0, csr, rs
        "csrw" => {
            let csr = csr_operand(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            (csr << 20) | (rs << 15) | (0b001 << 12) | OP_SYSTEM
        }
        // csrs csr, rs → csrrs x0, csr, rs
        "csrs" => {
            let csr = csr_operand(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            (csr << 20) | (rs << 15) | (0b010 << 12) | OP_SYSTEM
        }
        // csrc csr, rs → csrrc x0, csr, rs
        "csrc" => {
            let csr = csr_operand(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            (csr << 20) | (rs << 15) | (0b011 << 12) | OP_SYSTEM
        }
        // csrwi csr, imm → csrrwi x0, csr, imm
        "csrwi" => {
            let csr = csr_operand(&ops[0], span)?;
            let uimm = match &ops[1] {
                Operand::Immediate(v) => {
                    let v = *v as u32;
                    if v > 31 {
                        return Err(AsmError::ImmediateOverflow {
                            value: v as i128,
                            min: 0,
                            max: 31,
                            span,
                        });
                    }
                    v
                }
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("expected 5-bit unsigned immediate for CSR"),
                        span,
                    })
                }
            };
            (csr << 20) | (uimm << 15) | (0b101 << 12) | OP_SYSTEM
        }
        // csrsi csr, imm → csrrsi x0, csr, imm
        "csrsi" => {
            let csr = csr_operand(&ops[0], span)?;
            let uimm = match &ops[1] {
                Operand::Immediate(v) => {
                    let v = *v as u32;
                    if v > 31 {
                        return Err(AsmError::ImmediateOverflow {
                            value: v as i128,
                            min: 0,
                            max: 31,
                            span,
                        });
                    }
                    v
                }
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("expected 5-bit unsigned immediate for CSR"),
                        span,
                    })
                }
            };
            (csr << 20) | (uimm << 15) | (0b110 << 12) | OP_SYSTEM
        }
        // csrci csr, imm → csrrci x0, csr, imm
        "csrci" => {
            let csr = csr_operand(&ops[0], span)?;
            let uimm = match &ops[1] {
                Operand::Immediate(v) => {
                    let v = *v as u32;
                    if v > 31 {
                        return Err(AsmError::ImmediateOverflow {
                            value: v as i128,
                            min: 0,
                            max: 31,
                            span,
                        });
                    }
                    v
                }
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("expected 5-bit unsigned immediate for CSR"),
                        span,
                    })
                }
            };
            (csr << 20) | (uimm << 15) | (0b111 << 12) | OP_SYSTEM
        }

        // sext.w rd, rs → addiw rd, rs, 0 (RV64 only)
        "sext.w" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'sext.w' is only available in RV64"),
                    span,
                });
            }
            let rd = reg(&ops[0], span)?;
            let rs = reg(&ops[1], span)?;
            i_type(OP_IMM_W, rd, 0b000, rs, 0)
        }

        // la rd, label → auipc rd, %hi(label) + addi rd, rd, %lo(label)
        "la" => {
            if let Some((label, addend)) = extract_label(&ops[1]) {
                let rd = reg(&ops[0], span)?;
                let w1 = u_type(OP_AUIPC, rd, 0); // auipc rd, %hi(label)
                let w2 = i_type(OP_IMM, rd, 0b000, rd, 0); // addi rd, rd, %lo(label)
                let mut bytes = InstrBytes::new();
                bytes.extend_from_slice(&w1.to_le_bytes());
                bytes.extend_from_slice(&w2.to_le_bytes());
                return Ok(EncodedInstr {
                    bytes,
                    relocation: Some(Relocation {
                        offset: 0,
                        size: 4,
                        label: alloc::rc::Rc::from(label),
                        kind: RelocKind::RvAuipc20,
                        addend,
                        trailing_bytes: 4,
                    }),
                    relax: None,
                });
            }
            return Err(AsmError::InvalidOperands {
                detail: String::from("'la' requires a label operand"),
                span,
            });
        }

        // ── A extension (atomics) ────────────────────────────
        //
        // Format: AMO rd, rs2, (rs1)  with optional .aq, .rl, .aqrl suffixes
        // LR.W/D rd, (rs1)
        // SC.W/D rd, rs2, (rs1)
        // AMOxxx.W/D rd, rs2, (rs1)
        "lr.w" | "lr.w.aq" | "lr.w.rl" | "lr.w.aqrl" | "lr.d" | "lr.d.aq" | "lr.d.rl"
        | "lr.d.aqrl" => {
            let is_d = mnemonic.starts_with("lr.d");
            if is_d && !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!("'{}' is only available in RV64", mnemonic),
                    span,
                });
            }
            let aq = mnemonic.ends_with(".aq") || mnemonic.ends_with(".aqrl");
            let rl = mnemonic.ends_with(".rl") || mnemonic.ends_with(".aqrl");
            let funct3: u32 = if is_d { 0b011 } else { 0b010 };
            let rd = reg(&ops[0], span)?;
            let rs1 = amo_addr(&ops[1], span)?;
            amo_type(0b00010, aq, rl, 0, rs1, funct3, rd)
        }

        "sc.w" | "sc.w.aq" | "sc.w.rl" | "sc.w.aqrl" | "sc.d" | "sc.d.aq" | "sc.d.rl"
        | "sc.d.aqrl" => {
            let is_d = mnemonic.starts_with("sc.d");
            if is_d && !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!("'{}' is only available in RV64", mnemonic),
                    span,
                });
            }
            let aq = mnemonic.ends_with(".aq") || mnemonic.ends_with(".aqrl");
            let rl = mnemonic.ends_with(".rl") || mnemonic.ends_with(".aqrl");
            let funct3: u32 = if is_d { 0b011 } else { 0b010 };
            let rd = reg(&ops[0], span)?;
            let rs2 = reg(&ops[1], span)?;
            let rs1 = amo_addr(&ops[2], span)?;
            amo_type(0b00011, aq, rl, rs2, rs1, funct3, rd)
        }

        "amoswap.w" | "amoswap.w.aq" | "amoswap.w.rl" | "amoswap.w.aqrl" | "amoswap.d"
        | "amoswap.d.aq" | "amoswap.d.rl" | "amoswap.d.aqrl" | "amoadd.w" | "amoadd.w.aq"
        | "amoadd.w.rl" | "amoadd.w.aqrl" | "amoadd.d" | "amoadd.d.aq" | "amoadd.d.rl"
        | "amoadd.d.aqrl" | "amoand.w" | "amoand.w.aq" | "amoand.w.rl" | "amoand.w.aqrl"
        | "amoand.d" | "amoand.d.aq" | "amoand.d.rl" | "amoand.d.aqrl" | "amoor.w"
        | "amoor.w.aq" | "amoor.w.rl" | "amoor.w.aqrl" | "amoor.d" | "amoor.d.aq"
        | "amoor.d.rl" | "amoor.d.aqrl" | "amoxor.w" | "amoxor.w.aq" | "amoxor.w.rl"
        | "amoxor.w.aqrl" | "amoxor.d" | "amoxor.d.aq" | "amoxor.d.rl" | "amoxor.d.aqrl"
        | "amomax.w" | "amomax.w.aq" | "amomax.w.rl" | "amomax.w.aqrl" | "amomax.d"
        | "amomax.d.aq" | "amomax.d.rl" | "amomax.d.aqrl" | "amomaxu.w" | "amomaxu.w.aq"
        | "amomaxu.w.rl" | "amomaxu.w.aqrl" | "amomaxu.d" | "amomaxu.d.aq" | "amomaxu.d.rl"
        | "amomaxu.d.aqrl" | "amomin.w" | "amomin.w.aq" | "amomin.w.rl" | "amomin.w.aqrl"
        | "amomin.d" | "amomin.d.aq" | "amomin.d.rl" | "amomin.d.aqrl" | "amominu.w"
        | "amominu.w.aq" | "amominu.w.rl" | "amominu.w.aqrl" | "amominu.d" | "amominu.d.aq"
        | "amominu.d.rl" | "amominu.d.aqrl" => {
            // Parse the base name to get funct5, and extract .w/.d and aq/rl
            let is_d = mnemonic.contains(".d");
            if is_d && !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!("'{}' is only available in RV64", mnemonic),
                    span,
                });
            }
            let aq = mnemonic.ends_with(".aq") || mnemonic.ends_with(".aqrl");
            let rl = mnemonic.ends_with(".rl") || mnemonic.ends_with(".aqrl");
            let funct3: u32 = if is_d { 0b011 } else { 0b010 };
            // Extract funct5 from the base operation name
            let funct5: u32 = if mnemonic.starts_with("amoswap") {
                0b00001
            } else if mnemonic.starts_with("amoadd") {
                0b00000
            } else if mnemonic.starts_with("amoxor") {
                0b00100
            } else if mnemonic.starts_with("amoand") {
                0b01100
            } else if mnemonic.starts_with("amoor") {
                0b01000
            } else if mnemonic.starts_with("amomin.") {
                0b10000
            } else if mnemonic.starts_with("amomax.") {
                0b10100
            } else if mnemonic.starts_with("amominu") {
                0b11000
            } else if mnemonic.starts_with("amomaxu") {
                0b11100
            } else {
                return Err(AsmError::InvalidOperands {
                    detail: alloc::format!("unhandled atomic mnemonic '{}'", mnemonic),
                    span,
                });
            };

            let rd = reg(&ops[0], span)?;
            let rs2 = reg(&ops[1], span)?;
            let rs1 = amo_addr(&ops[2], span)?;
            amo_type(funct5, aq, rl, rs2, rs1, funct3, rd)
        }

        // ── F extension (single-precision floating-point) ─────
        // ── D extension (double-precision floating-point) ──────

        // FP loads: FLW rd, offset(rs1)  /  FLD rd, offset(rs1)
        "flw" => {
            let rd = fpreg(&ops[0], span)?;
            let (rs1, off) = mem(&ops[1], span)?;
            i_type(OP_LOAD_FP, rd, 0b010, rs1, off)
        }
        "fld" => {
            let rd = fpreg(&ops[0], span)?;
            let (rs1, off) = mem(&ops[1], span)?;
            i_type(OP_LOAD_FP, rd, 0b011, rs1, off)
        }

        // FP stores: FSW rs2, offset(rs1)  /  FSD rs2, offset(rs1)
        "fsw" => {
            let rs2 = fpreg(&ops[0], span)?;
            let (rs1, off) = mem(&ops[1], span)?;
            s_type(OP_STORE_FP, 0b010, rs1, rs2, off)
        }
        "fsd" => {
            let rs2 = fpreg(&ops[0], span)?;
            let (rs1, off) = mem(&ops[1], span)?;
            s_type(OP_STORE_FP, 0b011, rs1, rs2, off)
        }

        // ── R-type FP arithmetic (fmt=0b00 for .S, fmt=0b01 for .D) ──
        // rm defaults to 0b111 (dynamic) for all rounding-mode instructions.

        // FADD.S / FADD.D
        "fadd.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b111, rs1, rs2, 0b000_0000)
        }
        "fadd.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b111, rs1, rs2, 0b000_0001)
        }

        // FSUB.S / FSUB.D
        "fsub.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b111, rs1, rs2, 0b000_0100)
        }
        "fsub.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b111, rs1, rs2, 0b000_0101)
        }

        // FMUL.S / FMUL.D
        "fmul.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b111, rs1, rs2, 0b000_1000)
        }
        "fmul.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b111, rs1, rs2, 0b000_1001)
        }

        // FDIV.S / FDIV.D
        "fdiv.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b111, rs1, rs2, 0b000_1100)
        }
        "fdiv.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b111, rs1, rs2, 0b000_1101)
        }

        // FSQRT.S / FSQRT.D  (rs2=0)
        "fsqrt.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 0, 0b010_1100)
        }
        "fsqrt.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 0, 0b010_1101)
        }

        // FSGNJ.S / FSGNJN.S / FSGNJX.S  (sign-injection)
        "fsgnj.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b000, rs1, rs2, 0b001_0000)
        }
        "fsgnjn.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b001, rs1, rs2, 0b001_0000)
        }
        "fsgnjx.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b010, rs1, rs2, 0b001_0000)
        }

        // FSGNJ.D / FSGNJN.D / FSGNJX.D
        "fsgnj.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b000, rs1, rs2, 0b001_0001)
        }
        "fsgnjn.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b001, rs1, rs2, 0b001_0001)
        }
        "fsgnjx.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b010, rs1, rs2, 0b001_0001)
        }

        // FMIN.S / FMAX.S
        "fmin.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b000, rs1, rs2, 0b001_0100)
        }
        "fmax.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b001, rs1, rs2, 0b001_0100)
        }

        // FMIN.D / FMAX.D
        "fmin.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b000, rs1, rs2, 0b001_0101)
        }
        "fmax.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b001, rs1, rs2, 0b001_0101)
        }

        // ── FP compare → integer rd ──────────────────────────
        "feq.s" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b010, rs1, rs2, 0b101_0000)
        }
        "flt.s" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b001, rs1, rs2, 0b101_0000)
        }
        "fle.s" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b000, rs1, rs2, 0b101_0000)
        }
        "feq.d" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b010, rs1, rs2, 0b101_0001)
        }
        "flt.d" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b001, rs1, rs2, 0b101_0001)
        }
        "fle.d" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            r_type(OP_FP, rd, 0b000, rs1, rs2, 0b101_0001)
        }

        // ── FCLASS (classify FP → integer rd, rs2=0) ────────
        "fclass.s" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b001, rs1, 0, 0b111_0000)
        }
        "fclass.d" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b001, rs1, 0, 0b111_0001)
        }

        // ── FP ↔ integer conversions ─────────────────────────
        // FCVT.W.S: float→signed int32 (rs2=0)
        "fcvt.w.s" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 0, 0b110_0000)
        }
        // FCVT.WU.S: float→unsigned int32 (rs2=1)
        "fcvt.wu.s" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 1, 0b110_0000)
        }
        // FCVT.S.W: signed int32→float (rs2=0)
        "fcvt.s.w" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 0, 0b110_1000)
        }
        // FCVT.S.WU: unsigned int32→float (rs2=1)
        "fcvt.s.wu" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 1, 0b110_1000)
        }
        // FCVT.W.D / FCVT.WU.D: double→int32
        "fcvt.w.d" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 0, 0b110_0001)
        }
        "fcvt.wu.d" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 1, 0b110_0001)
        }
        // FCVT.D.W / FCVT.D.WU: int32→double
        "fcvt.d.w" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 0, 0b110_1001)
        }
        "fcvt.d.wu" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 1, 0b110_1001)
        }
        // FCVT.S.D: double→float (rs2=1)
        "fcvt.s.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 1, 0b010_0000)
        }
        // FCVT.D.S: float→double (rs2=0)
        "fcvt.d.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 0, 0b010_0001)
        }

        // ── RV64 FP↔int conversions (64-bit integer) ────────
        // FCVT.L.S / FCVT.LU.S: float→int64
        "fcvt.l.s" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'fcvt.l.s' requires RV64"),
                    span,
                });
            }
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 2, 0b110_0000)
        }
        "fcvt.lu.s" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'fcvt.lu.s' requires RV64"),
                    span,
                });
            }
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 3, 0b110_0000)
        }
        // FCVT.S.L / FCVT.S.LU: int64→float
        "fcvt.s.l" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'fcvt.s.l' requires RV64"),
                    span,
                });
            }
            let rd = fpreg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 2, 0b110_1000)
        }
        "fcvt.s.lu" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'fcvt.s.lu' requires RV64"),
                    span,
                });
            }
            let rd = fpreg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 3, 0b110_1000)
        }
        // FCVT.L.D / FCVT.LU.D: double→int64
        "fcvt.l.d" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'fcvt.l.d' requires RV64"),
                    span,
                });
            }
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 2, 0b110_0001)
        }
        "fcvt.lu.d" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'fcvt.lu.d' requires RV64"),
                    span,
                });
            }
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 3, 0b110_0001)
        }
        // FCVT.D.L / FCVT.D.LU: int64→double
        "fcvt.d.l" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'fcvt.d.l' requires RV64"),
                    span,
                });
            }
            let rd = fpreg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 2, 0b110_1001)
        }
        "fcvt.d.lu" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'fcvt.d.lu' requires RV64"),
                    span,
                });
            }
            let rd = fpreg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b111, rs1, 3, 0b110_1001)
        }

        // ── FP move (bitwise copy, no conversion) ───────────
        // FMV.X.W: FP bits→integer (rs2=0, funct3=000)
        "fmv.x.w" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b000, rs1, 0, 0b111_0000)
        }
        // FMV.W.X: integer bits→FP (rs2=0, funct3=000)
        "fmv.w.x" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b000, rs1, 0, 0b111_1000)
        }
        // FMV.X.D: FP bits→integer (RV64 only)
        "fmv.x.d" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'fmv.x.d' requires RV64"),
                    span,
                });
            }
            let rd = reg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b000, rs1, 0, 0b111_0001)
        }
        // FMV.D.X: integer bits→FP (RV64 only)
        "fmv.d.x" => {
            if !is_rv64 {
                return Err(AsmError::InvalidOperands {
                    detail: String::from("'fmv.d.x' requires RV64"),
                    span,
                });
            }
            let rd = fpreg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b000, rs1, 0, 0b111_1001)
        }

        // ── R4-type fused multiply-add ──────────────────────
        // FMADD.S: rd = rs1*rs2 + rs3
        "fmadd.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            let rs3 = fpreg(&ops[3], span)?;
            r4_type(OP_MADD, rd, 0b111, rs1, rs2, 0b00, rs3)
        }
        "fmadd.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            let rs3 = fpreg(&ops[3], span)?;
            r4_type(OP_MADD, rd, 0b111, rs1, rs2, 0b01, rs3)
        }
        // FMSUB.S: rd = rs1*rs2 − rs3
        "fmsub.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            let rs3 = fpreg(&ops[3], span)?;
            r4_type(OP_MSUB, rd, 0b111, rs1, rs2, 0b00, rs3)
        }
        "fmsub.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            let rs3 = fpreg(&ops[3], span)?;
            r4_type(OP_MSUB, rd, 0b111, rs1, rs2, 0b01, rs3)
        }
        // FNMSUB.S: rd = −(rs1*rs2) + rs3
        "fnmsub.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            let rs3 = fpreg(&ops[3], span)?;
            r4_type(OP_NMSUB, rd, 0b111, rs1, rs2, 0b00, rs3)
        }
        "fnmsub.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            let rs3 = fpreg(&ops[3], span)?;
            r4_type(OP_NMSUB, rd, 0b111, rs1, rs2, 0b01, rs3)
        }
        // FNMADD.S: rd = −(rs1*rs2) − rs3
        "fnmadd.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            let rs3 = fpreg(&ops[3], span)?;
            r4_type(OP_NMADD, rd, 0b111, rs1, rs2, 0b00, rs3)
        }
        "fnmadd.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs1 = fpreg(&ops[1], span)?;
            let rs2 = fpreg(&ops[2], span)?;
            let rs3 = fpreg(&ops[3], span)?;
            r4_type(OP_NMADD, rd, 0b111, rs1, rs2, 0b01, rs3)
        }

        // ── FP pseudo-instructions ──────────────────────────
        // FMV.S rd, rs  →  FSGNJ.S rd, rs, rs
        "fmv.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b000, rs, rs, 0b001_0000)
        }
        // FMV.D rd, rs  →  FSGNJ.D rd, rs, rs
        "fmv.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b000, rs, rs, 0b001_0001)
        }
        // FNEG.S rd, rs  →  FSGNJN.S rd, rs, rs
        "fneg.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b001, rs, rs, 0b001_0000)
        }
        // FNEG.D rd, rs  →  FSGNJN.D rd, rs, rs
        "fneg.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b001, rs, rs, 0b001_0001)
        }
        // FABS.S rd, rs  →  FSGNJX.S rd, rs, rs
        "fabs.s" => {
            let rd = fpreg(&ops[0], span)?;
            let rs = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b010, rs, rs, 0b001_0000)
        }
        // FABS.D rd, rs  →  FSGNJX.D rd, rs, rs
        "fabs.d" => {
            let rd = fpreg(&ops[0], span)?;
            let rs = fpreg(&ops[1], span)?;
            r_type(OP_FP, rd, 0b010, rs, rs, 0b001_0001)
        }

        // ── V-extension (vector) ────────────────────────────────

        // VSETVLI rd, rs1, vtypei  (vtype from e/m/ta/ma labels)
        // vsetvli a0, a1, e32, m1, ta, ma
        // Encoding: [31]=0 | [30:20]=zimm[10:0] | [19:15]=rs1 | [14:12]=111 | [11:7]=rd | [6:0]=1010111
        "vsetvli" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            let vtype = parse_vtype(ops, 2, span)?;
            // bit 31 = 0 for vsetvli
            (vtype << 20) | (rs1 << 15) | (0b111 << 12) | (rd << 7) | OP_V
        }

        // VSETIVLI rd, uimm, vtypei  (AVL from immediate)
        // vsetivli a0, 16, e32, m1, ta, ma
        // Encoding: [31:30]=11 | [29:20]=zimm[9:0] | [19:15]=uimm5 | [14:12]=111 | [11:7]=rd | [6:0]=1010111
        "vsetivli" => {
            let rd = reg(&ops[0], span)?;
            let avl = match &ops[1] {
                Operand::Immediate(v) => (*v as u32) & 0x1F,
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("expected immediate AVL for vsetivli"),
                        span,
                    })
                }
            };
            let vtype = parse_vtype(ops, 2, span)?;
            // bits [31:30] = 11
            (0b11 << 30) | ((vtype & 0x3FF) << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | OP_V
        }

        // VSETVL rd, rs1, rs2  (vtype from register)
        // vsetvl a0, a1, a2
        // Encoding: [31]=1 | [30:25]=000000 | [24:20]=rs2 | [19:15]=rs1 | [14:12]=111 | [11:7]=rd | [6:0]=1010111
        "vsetvl" => {
            let rd = reg(&ops[0], span)?;
            let rs1 = reg(&ops[1], span)?;
            let rs2 = reg(&ops[2], span)?;
            (1u32 << 31) | (rs2 << 20) | (rs1 << 15) | (0b111 << 12) | (rd << 7) | OP_V
        }

        // Vector loads: vle{8,16,32,64}.v vd, (rs1) [, v0.t]
        "vle8.v" | "vle16.v" | "vle32.v" | "vle64.v" => {
            let vd = vreg(&ops[0], span)?;
            let rs1 = match &ops[1] {
                Operand::Memory(m) => {
                    let base = m.base.ok_or_else(|| AsmError::InvalidOperands {
                        detail: String::from("expected base register for vector load"),
                        span,
                    })?;
                    base.rv_reg_num() as u32
                }
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("expected (rs1) memory operand"),
                        span,
                    })
                }
            };
            // Determine if masked: check for v0.t operand
            let vm: u32 = if ops.len() > 2 {
                // v0.t → masked (vm=0)
                match &ops[2] {
                    Operand::Register(r) if r.is_riscv_vec() && r.rv_vec_num() == 0 => 0,
                    _ => 1,
                }
            } else {
                1 // unmasked (vm=1)
            };
            let width: u32 = match mnemonic {
                "vle8.v" => 0b000,
                "vle16.v" => 0b101,
                "vle32.v" => 0b110,
                "vle64.v" => 0b111,
                _ => unreachable!(),
            };
            // Unit-stride load: nf=000, mew=0, mop=00, vm, lumop=00000
            // [31:29]=nf | [28]=mew | [27:26]=mop | [25]=vm | [24:20]=lumop | [19:15]=rs1 | [14:12]=width | [11:7]=vd | [6:0]=0000111
            (vm << 25) | (rs1 << 15) | (width << 12) | (vd << 7) | OP_V_LOAD
        }

        // Vector stores: vse{8,16,32,64}.v vs3, (rs1) [, v0.t]
        "vse8.v" | "vse16.v" | "vse32.v" | "vse64.v" => {
            let vs3 = vreg(&ops[0], span)?;
            let rs1 = match &ops[1] {
                Operand::Memory(m) => {
                    let base = m.base.ok_or_else(|| AsmError::InvalidOperands {
                        detail: String::from("expected base register for vector store"),
                        span,
                    })?;
                    base.rv_reg_num() as u32
                }
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("expected (rs1) memory operand"),
                        span,
                    })
                }
            };
            let vm: u32 = if ops.len() > 2 {
                match &ops[2] {
                    Operand::Register(r) if r.is_riscv_vec() && r.rv_vec_num() == 0 => 0,
                    _ => 1,
                }
            } else {
                1
            };
            let width: u32 = match mnemonic {
                "vse8.v" => 0b000,
                "vse16.v" => 0b101,
                "vse32.v" => 0b110,
                "vse64.v" => 0b111,
                _ => unreachable!(),
            };
            // Unit-stride store: nf=000, mew=0, mop=00, vm, sumop=00000
            (vm << 25) | (rs1 << 15) | (width << 12) | (vs3 << 7) | OP_V_STORE
        }

        // Vector arithmetic: vadd/vsub/vand/vor/vxor.vv vd, vs2, vs1
        // funct6 | vm | vs2 | vs1 | funct3 | vd | opcode
        "vadd.vv" => {
            let vd = vreg(&ops[0], span)?;
            let vs2 = vreg(&ops[1], span)?;
            let vs1 = vreg(&ops[2], span)?;
            (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | OP_V
        }
        "vsub.vv" => {
            let vd = vreg(&ops[0], span)?;
            let vs2 = vreg(&ops[1], span)?;
            let vs1 = vreg(&ops[2], span)?;
            (0b000010 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | OP_V
        }
        "vand.vv" => {
            let vd = vreg(&ops[0], span)?;
            let vs2 = vreg(&ops[1], span)?;
            let vs1 = vreg(&ops[2], span)?;
            (0b001001 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | OP_V
        }
        "vor.vv" => {
            let vd = vreg(&ops[0], span)?;
            let vs2 = vreg(&ops[1], span)?;
            let vs1 = vreg(&ops[2], span)?;
            (0b001010 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | OP_V
        }
        "vxor.vv" => {
            let vd = vreg(&ops[0], span)?;
            let vs2 = vreg(&ops[1], span)?;
            let vs1 = vreg(&ops[2], span)?;
            (0b001011 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | OP_V
        }
        "vmul.vv" => {
            let vd = vreg(&ops[0], span)?;
            let vs2 = vreg(&ops[1], span)?;
            let vs1 = vreg(&ops[2], span)?;
            // funct6=100101, funct3=OPMVV=010
            (0b100101 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (vs1 << 15)
                | (0b010 << 12)
                | (vd << 7)
                | OP_V
        }

        // Vector-scalar arithmetic: vadd.vx vd, vs2, rs1
        "vadd.vx" => {
            let vd = vreg(&ops[0], span)?;
            let vs2 = vreg(&ops[1], span)?;
            let rs1 = reg(&ops[2], span)?;
            // funct6=000000, funct3=OPIVX=100
            (1 << 25) | (vs2 << 20) | (rs1 << 15) | (0b100 << 12) | (vd << 7) | OP_V
        }
        "vsub.vx" => {
            let vd = vreg(&ops[0], span)?;
            let vs2 = vreg(&ops[1], span)?;
            let rs1 = reg(&ops[2], span)?;
            (0b000010 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (rs1 << 15)
                | (0b100 << 12)
                | (vd << 7)
                | OP_V
        }

        // Vector-immediate: vadd.vi vd, vs2, simm5
        "vadd.vi" => {
            let vd = vreg(&ops[0], span)?;
            let vs2 = vreg(&ops[1], span)?;
            let simm5 = match &ops[2] {
                Operand::Immediate(v) => (*v as u32) & 0x1F,
                _ => {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("expected immediate for vadd.vi"),
                        span,
                    })
                }
            };
            // funct6=000000, funct3=OPIVI=011
            (1 << 25) | (vs2 << 20) | (simm5 << 15) | (0b011 << 12) | (vd << 7) | OP_V
        }

        _ => {
            return Err(AsmError::UnknownMnemonic {
                mnemonic: instr.mnemonic.to_string(),
                arch: arch.to_arch_name(),
                span,
            });
        }
    };

    let mut bytes = InstrBytes::new();
    bytes.extend_from_slice(&word.to_le_bytes());
    Ok(EncodedInstr {
        bytes,
        relocation: None,
        relax: None,
    })
}

// ── Unit tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;

    fn span() -> Span {
        Span::dummy()
    }

    fn make_instr(mnemonic: &str, operands: Vec<Operand>) -> Instruction {
        Instruction {
            mnemonic: Mnemonic::from(mnemonic),
            operands: OperandList::from(operands),
            prefixes: PrefixList::new(),
            size_hint: None,
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        }
    }

    fn encode32(mnemonic: &str, ops: Vec<Operand>) -> u32 {
        let instr = make_instr(mnemonic, ops);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        u32::from_le_bytes(result.bytes[..4].try_into().unwrap())
    }

    fn encode64(mnemonic: &str, ops: Vec<Operand>) -> u32 {
        let instr = make_instr(mnemonic, ops);
        let result = encode_riscv(&instr, Arch::Rv64).unwrap();
        u32::from_le_bytes(result.bytes[..4].try_into().unwrap())
    }

    fn r(n: u8) -> Operand {
        use Register::*;
        let reg = match n {
            0 => RvX0,
            1 => RvX1,
            2 => RvX2,
            3 => RvX3,
            4 => RvX4,
            5 => RvX5,
            6 => RvX6,
            7 => RvX7,
            8 => RvX8,
            9 => RvX9,
            10 => RvX10,
            11 => RvX11,
            12 => RvX12,
            13 => RvX13,
            14 => RvX14,
            15 => RvX15,
            16 => RvX16,
            17 => RvX17,
            18 => RvX18,
            19 => RvX19,
            20 => RvX20,
            21 => RvX21,
            22 => RvX22,
            23 => RvX23,
            24 => RvX24,
            25 => RvX25,
            26 => RvX26,
            27 => RvX27,
            28 => RvX28,
            29 => RvX29,
            30 => RvX30,
            31 => RvX31,
            _ => panic!("invalid register number"),
        };
        Operand::Register(reg)
    }

    fn imm(v: i128) -> Operand {
        Operand::Immediate(v)
    }

    fn memop(base: u8, disp: i64) -> Operand {
        use Register::*;
        let reg = match base {
            0 => RvX0,
            1 => RvX1,
            2 => RvX2,
            3 => RvX3,
            4 => RvX4,
            5 => RvX5,
            6 => RvX6,
            7 => RvX7,
            8 => RvX8,
            9 => RvX9,
            10 => RvX10,
            11 => RvX11,
            12 => RvX12,
            13 => RvX13,
            14 => RvX14,
            15 => RvX15,
            16 => RvX16,
            17 => RvX17,
            18 => RvX18,
            19 => RvX19,
            20 => RvX20,
            21 => RvX21,
            22 => RvX22,
            23 => RvX23,
            24 => RvX24,
            25 => RvX25,
            26 => RvX26,
            27 => RvX27,
            28 => RvX28,
            29 => RvX29,
            30 => RvX30,
            31 => RvX31,
            _ => panic!("invalid register number"),
        };
        Operand::Memory(Box::new(MemoryOperand {
            base: Some(reg),
            disp,
            ..Default::default()
        }))
    }

    // ── R-type encoding tests ───────────────────────────────

    #[test]
    fn test_add() {
        // add x1, x2, x3 → R-type: funct7=0, rs2=3, rs1=2, funct3=0, rd=1, op=0x33
        let w = encode32("add", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x003100B3);
    }

    #[test]
    fn test_sub() {
        // sub x1, x2, x3
        let w = encode32("sub", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x403100B3);
    }

    #[test]
    fn test_and() {
        // and x5, x6, x7
        let w = encode32("and", vec![r(5), r(6), r(7)]);
        assert_eq!(w, 0x007372B3);
    }

    #[test]
    fn test_or() {
        // or x5, x6, x7
        let w = encode32("or", vec![r(5), r(6), r(7)]);
        assert_eq!(w, 0x007362B3);
    }

    #[test]
    fn test_xor() {
        // xor x5, x6, x7
        let w = encode32("xor", vec![r(5), r(6), r(7)]);
        assert_eq!(w, 0x007342B3);
    }

    #[test]
    fn test_sll() {
        // sll x1, x2, x3
        let w = encode32("sll", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x003110B3);
    }

    #[test]
    fn test_srl() {
        let w = encode32("srl", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x003150B3);
    }

    #[test]
    fn test_sra() {
        let w = encode32("sra", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x403150B3);
    }

    #[test]
    fn test_slt() {
        let w = encode32("slt", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x003120B3);
    }

    #[test]
    fn test_sltu() {
        let w = encode32("sltu", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x003130B3);
    }

    // ── I-type encoding tests ───────────────────────────────

    #[test]
    fn test_addi() {
        // addi x1, x2, 5 → I-type: imm=5, rs1=2, funct3=0, rd=1, op=0x13
        let w = encode32("addi", vec![r(1), r(2), imm(5)]);
        assert_eq!(w, 0x00510093);
    }

    #[test]
    fn test_addi_negative() {
        // addi x1, x0, -1
        let w = encode32("addi", vec![r(1), r(0), imm(-1)]);
        assert_eq!(w, 0xFFF00093);
    }

    #[test]
    fn test_andi() {
        let w = encode32("andi", vec![r(1), r(2), imm(0xFF)]);
        assert_eq!(w, 0x0FF17093);
    }

    #[test]
    fn test_ori() {
        let w = encode32("ori", vec![r(1), r(2), imm(0x12)]);
        assert_eq!(w, 0x01216093);
    }

    #[test]
    fn test_xori() {
        let w = encode32("xori", vec![r(1), r(2), imm(1)]);
        assert_eq!(w, 0x00114093);
    }

    #[test]
    fn test_slti() {
        let w = encode32("slti", vec![r(1), r(2), imm(10)]);
        assert_eq!(w, 0x00A12093);
    }

    #[test]
    fn test_sltiu() {
        let w = encode32("sltiu", vec![r(1), r(2), imm(10)]);
        assert_eq!(w, 0x00A13093);
    }

    // ── Shift immediate tests ───────────────────────────────

    #[test]
    fn test_slli() {
        // slli x1, x2, 3 → I-type special: imm[11:5]=0, shamt=3, funct3=1, rd=1
        let w = encode32("slli", vec![r(1), r(2), imm(3)]);
        assert_eq!(w, 0x00311093);
    }

    #[test]
    fn test_srli() {
        let w = encode32("srli", vec![r(1), r(2), imm(3)]);
        assert_eq!(w, 0x00315093);
    }

    #[test]
    fn test_srai() {
        // srai x1, x2, 3 → imm[11:5]=0x20, shamt=3
        let w = encode32("srai", vec![r(1), r(2), imm(3)]);
        assert_eq!(w, 0x40315093);
    }

    // ── Load tests ──────────────────────────────────────────

    #[test]
    fn test_lw() {
        // lw x1, 0(x2) → I-type: imm=0, rs1=2, funct3=010, rd=1, op=0x03
        let w = encode32("lw", vec![r(1), memop(2, 0)]);
        assert_eq!(w, 0x00012083);
    }

    #[test]
    fn test_lb() {
        // lb x1, 4(x2)
        let w = encode32("lb", vec![r(1), memop(2, 4)]);
        assert_eq!(w, 0x00410083);
    }

    #[test]
    fn test_lbu() {
        let w = encode32("lbu", vec![r(1), memop(2, 0)]);
        assert_eq!(w, 0x00014083);
    }

    #[test]
    fn test_lh() {
        let w = encode32("lh", vec![r(1), memop(2, 8)]);
        assert_eq!(w, 0x00811083);
    }

    #[test]
    fn test_lhu() {
        let w = encode32("lhu", vec![r(1), memop(2, 0)]);
        assert_eq!(w, 0x00015083);
    }

    #[test]
    fn test_ld_rv64() {
        let w = encode64("ld", vec![r(1), memop(2, 0)]);
        assert_eq!(w, 0x00013083);
    }

    // ── Store tests ─────────────────────────────────────────

    #[test]
    fn test_sw() {
        // sw x1, 0(x2) → S-type: imm=0, rs2=1, rs1=2, funct3=010, op=0x23
        let w = encode32("sw", vec![r(1), memop(2, 0)]);
        assert_eq!(w, 0x00112023);
    }

    #[test]
    fn test_sb() {
        // sb x1, 4(x2)
        let w = encode32("sb", vec![r(1), memop(2, 4)]);
        assert_eq!(w, 0x00110223);
    }

    #[test]
    fn test_sh() {
        let w = encode32("sh", vec![r(1), memop(2, 2)]);
        assert_eq!(w, 0x00111123);
    }

    #[test]
    fn test_sd_rv64() {
        let w = encode64("sd", vec![r(1), memop(2, 0)]);
        assert_eq!(w, 0x00113023);
    }

    // ── U-type tests ────────────────────────────────────────

    #[test]
    fn test_lui() {
        // lui x1, 0x12345 → U-type: imm=0x12345 << 12, rd=1, op=0x37
        let w = encode32("lui", vec![r(1), imm(0x12345)]);
        assert_eq!(w, 0x123450B7);
    }

    // ── B-type tests ────────────────────────────────────────

    #[test]
    fn test_beq_encoding() {
        // beq x1, x2, 8 → B-type: imm=8, rs2=2, rs1=1, funct3=000
        let w = encode32("beq", vec![r(1), r(2), imm(8)]);
        assert_eq!(w, 0x00208463);
    }

    #[test]
    fn test_bne_encoding() {
        let w = encode32("bne", vec![r(1), r(2), imm(8)]);
        assert_eq!(w, 0x00209463);
    }

    // ── J-type tests ────────────────────────────────────────

    #[test]
    fn test_jal_rd_imm() {
        // jal x1, 0 → J-type: imm=0, rd=1, op=0x6F
        let w = encode32("jal", vec![r(1), imm(0)]);
        assert_eq!(w, 0x000000EF);
    }

    // ── System tests ────────────────────────────────────────

    #[test]
    fn test_ecall() {
        let w = encode32("ecall", vec![]);
        assert_eq!(w, 0x00000073);
    }

    #[test]
    fn test_ebreak() {
        let w = encode32("ebreak", vec![]);
        assert_eq!(w, 0x00100073);
    }

    // ── Privileged instruction tests ────────────────────────

    #[test]
    fn test_mret() {
        // mret: 0011000_00010_00000_000_00000_1110011
        let w = encode32("mret", vec![]);
        assert_eq!(w, 0x30200073);
    }

    #[test]
    fn test_sret() {
        // sret: 0001000_00010_00000_000_00000_1110011
        let w = encode32("sret", vec![]);
        assert_eq!(w, 0x10200073);
    }

    #[test]
    fn test_wfi() {
        // wfi: 0001000_00101_00000_000_00000_1110011
        let w = encode32("wfi", vec![]);
        assert_eq!(w, 0x10500073);
    }

    #[test]
    fn test_sfence_vma_no_args() {
        // sfence.vma (no args) → sfence.vma x0, x0
        // 0001001_00000_00000_000_00000_1110011
        let w = encode32("sfence.vma", vec![]);
        assert_eq!(w, 0x12000073);
    }

    #[test]
    fn test_sfence_vma_two_regs() {
        // sfence.vma x1, x2
        // 0001001_00010_00001_000_00000_1110011
        let w = encode32("sfence.vma", vec![r(1), r(2)]);
        assert_eq!(w, 0x12208073);
    }

    // ── NOP ─────────────────────────────────────────────────

    #[test]
    fn test_nop() {
        // nop → addi x0, x0, 0
        let w = encode32("nop", vec![]);
        assert_eq!(w, 0x00000013);
    }

    // ── Pseudo-instruction tests ────────────────────────────

    #[test]
    fn test_mv() {
        // mv x1, x2 → addi x1, x2, 0
        let w = encode32("mv", vec![r(1), r(2)]);
        assert_eq!(w, 0x00010093);
    }

    #[test]
    fn test_not() {
        // not x1, x2 → xori x1, x2, -1
        let w = encode32("not", vec![r(1), r(2)]);
        assert_eq!(w, 0xFFF14093);
    }

    #[test]
    fn test_neg() {
        // neg x1, x2 → sub x1, x0, x2
        let w = encode32("neg", vec![r(1), r(2)]);
        assert_eq!(w, 0x402000B3);
    }

    #[test]
    fn test_ret() {
        // ret → jalr x0, ra, 0
        let w = encode32("ret", vec![]);
        assert_eq!(w, 0x00008067);
    }

    #[test]
    fn test_seqz() {
        // seqz x1, x2 → sltiu x1, x2, 1
        let w = encode32("seqz", vec![r(1), r(2)]);
        assert_eq!(w, 0x00113093);
    }

    #[test]
    fn test_snez() {
        // snez x1, x2 → sltu x1, x0, x2
        let w = encode32("snez", vec![r(1), r(2)]);
        assert_eq!(w, 0x002030B3);
    }

    #[test]
    fn test_li_small() {
        // li x1, 42 → addi x1, x0, 42
        let w = encode32("li", vec![r(1), imm(42)]);
        assert_eq!(w, 0x02A00093);
    }

    #[test]
    fn test_jr() {
        // jr x1 → jalr x0, x1, 0
        let w = encode32("jr", vec![r(1)]);
        assert_eq!(w, 0x00008067);
    }

    // ── M extension tests ───────────────────────────────────

    #[test]
    fn test_mul() {
        // mul x1, x2, x3 → R-type: funct7=1, rs2=3, rs1=2, funct3=0, rd=1, op=0x33
        let w = encode32("mul", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x023100B3);
    }

    #[test]
    fn test_div() {
        let w = encode32("div", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x023140B3);
    }

    #[test]
    fn test_rem() {
        let w = encode32("rem", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x023160B3);
    }

    // ── JALR tests ──────────────────────────────────────────

    #[test]
    fn test_jalr_3op() {
        // jalr x1, x2, 4 → I-type: imm=4, rs1=2, funct3=0, rd=1, op=0x67
        let w = encode32("jalr", vec![r(1), r(2), imm(4)]);
        assert_eq!(w, 0x004100E7);
    }

    // ── RV64 W-suffix ───────────────────────────────────────

    #[test]
    fn test_addw_rv64() {
        let w = encode64("addw", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x003100BB);
    }

    #[test]
    fn test_subw_rv64() {
        let w = encode64("subw", vec![r(1), r(2), r(3)]);
        assert_eq!(w, 0x403100BB);
    }

    #[test]
    fn test_addiw_rv64() {
        let w = encode64("addiw", vec![r(1), r(2), imm(5)]);
        assert_eq!(w, 0x0051009B);
    }

    // ── FENCE ───────────────────────────────────────────────

    #[test]
    fn test_fence() {
        let w = encode32("fence", vec![]);
        assert_eq!(w, 0x0FF0000F);
    }

    #[test]
    fn test_fence_i() {
        let w = encode32("fence.i", vec![]);
        assert_eq!(w, 0x0000100F);
    }

    // ── LI large ────────────────────────────────────────────

    #[test]
    fn test_li_large() {
        // li x1, 0x12345 → lui x1, 0x12 + addi x1, x1, 0x345
        let instr = make_instr("li", vec![r(1), imm(0x12345)]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        assert_eq!(result.bytes.len(), 8); // two 32-bit instructions
    }

    // ── A extension (atomics) ───────────────────────────────

    #[test]
    fn test_lr_w() {
        // lr.w x1, (x2) → amo_type(0b00010, false, false, 0, 2, 0b010, 1)
        let w = encode32("lr.w", vec![r(1), memop(2, 0)]);
        // funct5=00010 aq=0 rl=0 rs2=00000 rs1=00010 funct3=010 rd=00001 opcode=0101111
        let expected = (0b00010 << 27) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_lr_w_aq() {
        let w = encode32("lr.w.aq", vec![r(1), memop(2, 0)]);
        let expected = (0b00010 << 27) | (1 << 26) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_lr_w_rl() {
        let w = encode32("lr.w.rl", vec![r(1), memop(2, 0)]);
        let expected = (0b00010 << 27) | (1 << 25) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_lr_w_aqrl() {
        let w = encode32("lr.w.aqrl", vec![r(1), memop(2, 0)]);
        let expected =
            (0b00010 << 27) | (1 << 26) | (1 << 25) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_lr_d_rv64() {
        let instr = make_instr("lr.d", vec![r(1), memop(2, 0)]);
        let result = encode_riscv(&instr, Arch::Rv64).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0b00010 << 27) | (2 << 15) | (0b011 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_sc_w() {
        // sc.w x1, x3, (x2)
        let w = encode32("sc.w", vec![r(1), r(3), memop(2, 0)]);
        let expected = (0b00011 << 27) | (3 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_sc_w_aqrl() {
        let w = encode32("sc.w.aqrl", vec![r(1), r(3), memop(2, 0)]);
        let expected = (0b00011 << 27)
            | (1 << 26)
            | (1 << 25)
            | (3 << 20)
            | (2 << 15)
            | (0b010 << 12)
            | (1 << 7)
            | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_sc_d_rv64() {
        let instr = make_instr("sc.d", vec![r(1), r(3), memop(2, 0)]);
        let result = encode_riscv(&instr, Arch::Rv64).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0b00011 << 27) | (3 << 20) | (2 << 15) | (0b011 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amoswap_w() {
        // amoswap.w x1, x3, (x2) → funct5=00001
        let w = encode32("amoswap.w", vec![r(1), r(3), memop(2, 0)]);
        let expected = (0b00001 << 27) | (3 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amoswap_w_aqrl() {
        let w = encode32("amoswap.w.aqrl", vec![r(1), r(3), memop(2, 0)]);
        let expected = (0b00001 << 27)
            | (1 << 26)
            | (1 << 25)
            | (3 << 20)
            | (2 << 15)
            | (0b010 << 12)
            | (1 << 7)
            | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amoadd_w() {
        let w = encode32("amoadd.w", vec![r(1), r(3), memop(2, 0)]);
        let expected = (3 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amoand_w() {
        let w = encode32("amoand.w", vec![r(1), r(3), memop(2, 0)]);
        let expected = (0b01100 << 27) | (3 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amoor_w() {
        let w = encode32("amoor.w", vec![r(1), r(3), memop(2, 0)]);
        let expected = (0b01000 << 27) | (3 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amoxor_w() {
        let w = encode32("amoxor.w", vec![r(1), r(3), memop(2, 0)]);
        let expected = (0b00100 << 27) | (3 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amomax_w() {
        let w = encode32("amomax.w", vec![r(1), r(3), memop(2, 0)]);
        let expected = (0b10100 << 27) | (3 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amomaxu_w() {
        let w = encode32("amomaxu.w", vec![r(1), r(3), memop(2, 0)]);
        let expected = (0b11100 << 27) | (3 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amomin_w() {
        let w = encode32("amomin.w", vec![r(1), r(3), memop(2, 0)]);
        let expected = (0b10000 << 27) | (3 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amominu_w() {
        let w = encode32("amominu.w", vec![r(1), r(3), memop(2, 0)]);
        let expected = (0b11000 << 27) | (3 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amoswap_d_rv64() {
        let instr = make_instr("amoswap.d", vec![r(1), r(3), memop(2, 0)]);
        let result = encode_riscv(&instr, Arch::Rv64).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0b00001 << 27) | (3 << 20) | (2 << 15) | (0b011 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_amoswap_d_aq_rv64() {
        let instr = make_instr("amoswap.d.aq", vec![r(1), r(3), memop(2, 0)]);
        let result = encode_riscv(&instr, Arch::Rv64).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected =
            (0b00001 << 27) | (1 << 26) | (3 << 20) | (2 << 15) | (0b011 << 12) | (1 << 7) | 0x2F;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_lr_d_rejects_rv32() {
        let instr = make_instr("lr.d", vec![r(1), memop(2, 0)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    #[test]
    fn test_lr_w_rejects_nonzero_offset() {
        let instr = make_instr("lr.w", vec![r(1), memop(2, 4)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    // ── CSR pseudo-instructions ─────────────────────────────

    #[test]
    fn test_csrr() {
        // csrr a0, mstatus → csrrs a0, 0x300, x0
        let instr = make_instr("csrr", vec![r(10), Operand::Label("mstatus".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0x300 << 20) | (0b010 << 12) | (10 << 7) | OP_SYSTEM;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_csrw() {
        // csrw mtvec, t0 → csrrw x0, 0x305, t0(x5)
        let instr = make_instr("csrw", vec![Operand::Label("mtvec".into()), r(5)]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0x305 << 20) | (5 << 15) | (0b001 << 12) | OP_SYSTEM;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_csrs() {
        // csrs mie, a0 → csrrs x0, 0x304, a0(x10)
        let instr = make_instr("csrs", vec![Operand::Label("mie".into()), r(10)]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0x304 << 20) | (10 << 15) | (0b010 << 12) | OP_SYSTEM;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_csrc() {
        // csrc mstatus, a0 → csrrc x0, 0x300, a0(x10)
        let instr = make_instr("csrc", vec![Operand::Label("mstatus".into()), r(10)]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0x300 << 20) | (10 << 15) | (0b011 << 12) | OP_SYSTEM;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_csrwi() {
        // csrwi mstatus, 3 → csrrwi x0, 0x300, 3
        let instr = make_instr("csrwi", vec![Operand::Label("mstatus".into()), imm(3)]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0x300 << 20) | (3 << 15) | (0b101 << 12) | OP_SYSTEM;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_csrsi() {
        let instr = make_instr("csrsi", vec![Operand::Label("mie".into()), imm(2)]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0x304 << 20) | (2 << 15) | (0b110 << 12) | OP_SYSTEM;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_csrci() {
        let instr = make_instr("csrci", vec![Operand::Label("mstatus".into()), imm(1)]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0x300 << 20) | (1 << 15) | (0b111 << 12) | OP_SYSTEM;
        assert_eq!(w, expected);
    }

    // ── Named CSR with numeric address ──────────────────────

    #[test]
    fn test_csr_numeric_addr() {
        // csrrw a0, 0x300, a1 — numeric CSR address
        let instr = make_instr("csrrw", vec![r(10), imm(0x300), r(11)]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0x300 << 20) | (11 << 15) | (0b001 << 12) | (10 << 7) | OP_SYSTEM;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_csr_named_in_csrrs() {
        // csrrs a0, mstatus, a1 — named CSR in standard instruction
        let instr = make_instr(
            "csrrs",
            vec![r(10), Operand::Label("mstatus".into()), r(11)],
        );
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let w = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        let expected = (0x300 << 20) | (11 << 15) | (0b010 << 12) | (10 << 7) | OP_SYSTEM;
        assert_eq!(w, expected);
    }

    #[test]
    fn test_unknown_csr_name_rejected() {
        let instr = make_instr("csrr", vec![r(10), Operand::Label("bogus_csr".into())]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    // ── la pseudo ───────────────────────────────────────────

    #[test]
    fn test_la_emits_two_instructions() {
        let instr = make_instr("la", vec![r(1), Operand::Label("target".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        assert_eq!(result.bytes.len(), 8); // AUIPC + ADDI
        assert!(result.relocation.is_some());
        assert_eq!(result.relocation.unwrap().kind, RelocKind::RvAuipc20);
    }

    #[test]
    fn test_la_without_label_rejected() {
        let instr = make_instr("la", vec![r(1), imm(42)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    // ── Branch relaxation ───────────────────────────────────

    #[test]
    fn test_beq_label_emits_relaxable() {
        let instr = make_instr("beq", vec![r(1), r(2), Operand::Label("target".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let ri = result.relax.as_ref().expect("should be relaxable");
        // Short form: B-type beq (funct3=000), 4 bytes
        assert_eq!(ri.short_bytes.len(), 4);
        let short_reloc = ri.short_relocation.as_ref().unwrap();
        assert_eq!(short_reloc.kind, RelocKind::RvBranch12);
        // Long form: inverted bne (+8) + JAL x0, 8 bytes
        assert_eq!(result.bytes.len(), 8);
        let long_reloc = result.relocation.as_ref().unwrap();
        assert_eq!(long_reloc.kind, RelocKind::RvJal20);
        assert_eq!(long_reloc.offset, 4);
    }

    #[test]
    fn test_beq_label_long_form_inverts_condition() {
        let instr = make_instr("beq", vec![r(5), r(6), Operand::Label("target".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let ri = result.relax.as_ref().unwrap();
        // Short: beq (funct3=000)
        let short_w = u32::from_le_bytes(ri.short_bytes[0..4].try_into().unwrap());
        assert_eq!((short_w >> 12) & 0b111, 0b000); // beq
                                                    // Long first word: bne (funct3=001) with offset +8
        let long_w0 = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        assert_eq!((long_w0 >> 12) & 0b111, 0b001); // bne (inverted)
                                                    // Long second word: JAL x0
        let long_w1 = u32::from_le_bytes(result.bytes[4..8].try_into().unwrap());
        assert_eq!(long_w1 & 0x7F, OP_JAL); // JAL opcode
        assert_eq!((long_w1 >> 7) & 0x1F, 0); // rd = x0
    }

    #[test]
    fn test_bne_label_relaxable() {
        let instr = make_instr("bne", vec![r(1), r(2), Operand::Label("target".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let ri = result.relax.as_ref().unwrap();
        let short_w = u32::from_le_bytes(ri.short_bytes[0..4].try_into().unwrap());
        assert_eq!((short_w >> 12) & 0b111, 0b001); // bne
        let long_w0 = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        assert_eq!((long_w0 >> 12) & 0b111, 0b000); // beq (inverted)
    }

    #[test]
    fn test_blt_label_relaxable() {
        let instr = make_instr("blt", vec![r(1), r(2), Operand::Label("target".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let ri = result.relax.as_ref().unwrap();
        let short_w = u32::from_le_bytes(ri.short_bytes[0..4].try_into().unwrap());
        assert_eq!((short_w >> 12) & 0b111, 0b100); // blt
        let long_w0 = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        assert_eq!((long_w0 >> 12) & 0b111, 0b101); // bge (inverted)
    }

    #[test]
    fn test_beqz_label_relaxable() {
        let instr = make_instr("beqz", vec![r(5), Operand::Label("target".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let ri = result.relax.as_ref().expect("beqz should be relaxable");
        assert_eq!(ri.short_bytes.len(), 4);
        assert_eq!(result.bytes.len(), 8);
    }

    #[test]
    fn test_bnez_label_relaxable() {
        let instr = make_instr("bnez", vec![r(5), Operand::Label("target".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        assert!(result.relax.is_some());
    }

    #[test]
    fn test_bgt_label_relaxable() {
        let instr = make_instr("bgt", vec![r(1), r(2), Operand::Label("target".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let ri = result.relax.as_ref().unwrap();
        // bgt rs,rt → blt rt,rs, so short funct3 = 100 (blt)
        let short_w = u32::from_le_bytes(ri.short_bytes[0..4].try_into().unwrap());
        assert_eq!((short_w >> 12) & 0b111, 0b100);
        // long inverted: bge (101)
        let long_w0 = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        assert_eq!((long_w0 >> 12) & 0b111, 0b101);
    }

    #[test]
    fn test_branch_imm_not_relaxable() {
        // Immediate branch offsets should NOT produce relaxable output
        let instr = make_instr("beq", vec![r(1), r(2), imm(8)]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        assert!(result.relax.is_none());
        assert_eq!(result.bytes.len(), 4);
    }

    #[test]
    fn test_long_form_inverted_branch_offset_is_8() {
        // The inverted branch in the long form should skip exactly 8 bytes (over the JAL)
        let instr = make_instr("beq", vec![r(1), r(2), Operand::Label("target".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        let long_w0 = u32::from_le_bytes(result.bytes[0..4].try_into().unwrap());
        // Decode B-type immediate from long form first word
        let imm12 = (long_w0 >> 31) & 1;
        let imm10_5 = (long_w0 >> 25) & 0x3F;
        let imm4_1 = (long_w0 >> 8) & 0xF;
        let imm11 = (long_w0 >> 7) & 1;
        let offset = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
        assert_eq!(offset, 8);
    }

    // ── C-extension encoding tests ──────────────────────────

    fn encode_rvc(mnemonic: &str, ops: Vec<Operand>) -> u16 {
        let instr = make_instr(mnemonic, ops);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        assert_eq!(
            result.bytes.len(),
            2,
            "expected 2-byte compressed instruction"
        );
        u16::from_le_bytes(result.bytes[..2].try_into().unwrap())
    }

    fn encode_rvc64(mnemonic: &str, ops: Vec<Operand>) -> u16 {
        let instr = make_instr(mnemonic, ops);
        let result = encode_riscv(&instr, Arch::Rv64).unwrap();
        assert_eq!(
            result.bytes.len(),
            2,
            "expected 2-byte compressed instruction"
        );
        u16::from_le_bytes(result.bytes[..2].try_into().unwrap())
    }

    #[test]
    fn test_c_nop() {
        let hw = encode_rvc("c.nop", vec![]);
        // c.nop: funct3=000, imm=0, rd=0, op=01 → 0x0001
        assert_eq!(hw, 0x0001);
    }

    #[test]
    fn test_c_li() {
        // c.li x10, 5: funct3=010, imm[5]=0, rd=10, imm[4:0]=5, op=01
        let hw = encode_rvc("c.li", vec![r(10), imm(5)]);
        assert_eq!(hw & 0x3, 0x01); // quadrant 1
        assert_eq!((hw >> 13) & 0x7, 0b010); // funct3
        assert_eq!((hw >> 7) & 0x1F, 10); // rd
        assert_eq!((hw >> 2) & 0x1F, 5); // imm[4:0]
        assert_eq!((hw >> 12) & 1, 0); // imm[5]
    }

    #[test]
    fn test_c_li_negative() {
        // c.li x15, -1: sign-extended, imm = 0x3F (6-bit)
        let hw = encode_rvc("c.li", vec![r(15), imm(-1)]);
        assert_eq!(hw & 0x3, 0x01);
        assert_eq!((hw >> 13) & 0x7, 0b010);
        assert_eq!((hw >> 7) & 0x1F, 15);
        assert_eq!((hw >> 2) & 0x1F, 0x1F); // imm[4:0] = 31
        assert_eq!((hw >> 12) & 1, 1); // imm[5] = 1 (sign)
    }

    #[test]
    fn test_c_lui() {
        // c.lui x3, 1: funct3=011, nzimm[17]=0, rd=3, nzimm[16:12]=1, op=01
        let hw = encode_rvc("c.lui", vec![r(3), imm(1)]);
        assert_eq!(hw & 0x3, 0x01);
        assert_eq!((hw >> 13) & 0x7, 0b011);
        assert_eq!((hw >> 7) & 0x1F, 3);
        assert_eq!((hw >> 2) & 0x1F, 1);
    }

    #[test]
    fn test_c_addi() {
        // c.addi x5, 10
        let hw = encode_rvc("c.addi", vec![r(5), imm(10)]);
        assert_eq!(hw & 0x3, 0x01);
        assert_eq!((hw >> 13) & 0x7, 0b000);
        assert_eq!((hw >> 7) & 0x1F, 5);
        assert_eq!((hw >> 2) & 0x1F, 10);
    }

    #[test]
    fn test_c_addi_negative() {
        // c.addi x5, -3
        let hw = encode_rvc("c.addi", vec![r(5), imm(-3)]);
        let imm6 = ((-3i16) as u16) & 0x3F;
        assert_eq!((hw >> 2) & 0x1F, imm6 & 0x1F);
        assert_eq!((hw >> 12) & 1, (imm6 >> 5) & 1);
    }

    #[test]
    fn test_c_mv() {
        // c.mv x1, x2: funct4=1000, rd=1, rs2=2, op=10
        let hw = encode_rvc("c.mv", vec![r(1), r(2)]);
        assert_eq!(hw & 0x3, 0x02); // quadrant 2
        assert_eq!((hw >> 12) & 0xF, 0b1000); // funct4
        assert_eq!((hw >> 7) & 0x1F, 1); // rd
        assert_eq!((hw >> 2) & 0x1F, 2); // rs2
    }

    #[test]
    fn test_c_add() {
        // c.add x3, x4: funct4=1001, rd=3, rs2=4, op=10
        let hw = encode_rvc("c.add", vec![r(3), r(4)]);
        assert_eq!(hw & 0x3, 0x02);
        assert_eq!((hw >> 12) & 0xF, 0b1001);
        assert_eq!((hw >> 7) & 0x1F, 3);
        assert_eq!((hw >> 2) & 0x1F, 4);
    }

    #[test]
    fn test_c_jr() {
        // c.jr x5: funct4=1000, rs1=5, rs2=0, op=10
        let hw = encode_rvc("c.jr", vec![r(5)]);
        assert_eq!(hw & 0x3, 0x02);
        assert_eq!((hw >> 12) & 0xF, 0b1000);
        assert_eq!((hw >> 7) & 0x1F, 5);
        assert_eq!((hw >> 2) & 0x1F, 0);
    }

    #[test]
    fn test_c_jalr() {
        // c.jalr x5: funct4=1001, rs1=5, rs2=0, op=10
        let hw = encode_rvc("c.jalr", vec![r(5)]);
        assert_eq!(hw & 0x3, 0x02);
        assert_eq!((hw >> 12) & 0xF, 0b1001);
        assert_eq!((hw >> 7) & 0x1F, 5);
        assert_eq!((hw >> 2) & 0x1F, 0);
    }

    #[test]
    fn test_c_ebreak() {
        // c.ebreak: funct4=1001, rd/rs1=0, rs2=0, op=10
        let hw = encode_rvc("c.ebreak", vec![]);
        assert_eq!(hw, 0x9002);
    }

    #[test]
    fn test_c_sub() {
        // c.sub x8, x9: CA-type, funct6=100011, rd'/rs1'=0(x8), funct2=00, rs2'=1(x9)
        let hw = encode_rvc("c.sub", vec![r(8), r(9)]);
        assert_eq!(hw & 0x3, 0x01); // Q1
        assert_eq!((hw >> 10) & 0x3F, 0b100011); // funct6
        assert_eq!((hw >> 7) & 0x7, 0); // rd' (x8→0)
        assert_eq!((hw >> 5) & 0x3, 0b00); // funct2
        assert_eq!((hw >> 2) & 0x7, 1); // rs2' (x9→1)
    }

    #[test]
    fn test_c_xor() {
        let hw = encode_rvc("c.xor", vec![r(10), r(11)]);
        assert_eq!((hw >> 5) & 0x3, 0b01); // funct2 for xor
    }

    #[test]
    fn test_c_or() {
        let hw = encode_rvc("c.or", vec![r(12), r(13)]);
        assert_eq!((hw >> 5) & 0x3, 0b10); // funct2 for or
    }

    #[test]
    fn test_c_and() {
        let hw = encode_rvc("c.and", vec![r(14), r(15)]);
        assert_eq!((hw >> 5) & 0x3, 0b11); // funct2 for and
    }

    #[test]
    fn test_c_beqz_immediate() {
        // c.beqz x8, 4: CB-type, funct3=110
        let hw = encode_rvc("c.beqz", vec![r(8), imm(4)]);
        assert_eq!(hw & 0x3, 0x01); // Q1
        assert_eq!((hw >> 13) & 0x7, 0b110); // funct3
    }

    #[test]
    fn test_c_bnez_immediate() {
        // c.bnez x9, -2
        let hw = encode_rvc("c.bnez", vec![r(9), imm(-2)]);
        assert_eq!(hw & 0x3, 0x01);
        assert_eq!((hw >> 13) & 0x7, 0b111); // funct3 for bnez
    }

    #[test]
    fn test_c_j_immediate() {
        // c.j 0: CJ-type, funct3=101
        let hw = encode_rvc("c.j", vec![imm(0)]);
        assert_eq!(hw & 0x3, 0x01);
        assert_eq!((hw >> 13) & 0x7, 0b101);
    }

    #[test]
    fn test_c_slli() {
        // c.slli x1, 4: CI-type, funct3=000, rd=1, shamt=4, op=10
        let hw = encode_rvc("c.slli", vec![r(1), imm(4)]);
        assert_eq!(hw & 0x3, 0x02); // Q2
        assert_eq!((hw >> 13) & 0x7, 0b000);
        assert_eq!((hw >> 7) & 0x1F, 1);
        assert_eq!((hw >> 2) & 0x1F, 4);
    }

    #[test]
    fn test_c_srli() {
        // c.srli x8, 3
        let hw = encode_rvc("c.srli", vec![r(8), imm(3)]);
        assert_eq!(hw & 0x3, 0x01); // Q1
        assert_eq!((hw >> 13) & 0x7, 0b100);
        assert_eq!((hw >> 10) & 0x3, 0b00); // funct2 for srli
    }

    #[test]
    fn test_c_srai() {
        // c.srai x9, 5
        let hw = encode_rvc("c.srai", vec![r(9), imm(5)]);
        assert_eq!((hw >> 10) & 0x3, 0b01); // funct2 for srai
    }

    #[test]
    fn test_c_andi() {
        // c.andi x10, 7
        let hw = encode_rvc("c.andi", vec![r(10), imm(7)]);
        assert_eq!((hw >> 10) & 0x3, 0b10); // funct2 for andi
    }

    #[test]
    fn test_c_lw() {
        // c.lw x8, 0(x9): CL-type, funct3=010
        let hw = encode_rvc("c.lw", vec![r(8), memop(9, 0)]);
        assert_eq!(hw & 0x3, 0x00); // Q0
        assert_eq!((hw >> 13) & 0x7, 0b010); // funct3
    }

    #[test]
    fn test_c_lw_offset() {
        // c.lw x10, 4(x8): check offset encoding
        let hw = encode_rvc("c.lw", vec![r(10), memop(8, 4)]);
        assert_eq!(hw & 0x3, 0x00);
        assert_eq!((hw >> 13) & 0x7, 0b010);
    }

    #[test]
    fn test_c_sw() {
        // c.sw x8, 0(x9): CS-type, funct3=110
        let hw = encode_rvc("c.sw", vec![r(8), memop(9, 0)]);
        assert_eq!(hw & 0x3, 0x00); // Q0
        assert_eq!((hw >> 13) & 0x7, 0b110); // funct3
    }

    #[test]
    fn test_c_lwsp() {
        // c.lwsp x10, 0: CI-type in Q2, funct3=010
        let hw = encode_rvc("c.lwsp", vec![r(10), imm(0)]);
        assert_eq!(hw & 0x3, 0x02); // Q2
        assert_eq!((hw >> 13) & 0x7, 0b010);
        assert_eq!((hw >> 7) & 0x1F, 10);
    }

    #[test]
    fn test_c_swsp() {
        // c.swsp x5, 0: CSS-type, funct3=110
        let hw = encode_rvc("c.swsp", vec![r(5), imm(0)]);
        assert_eq!(hw & 0x3, 0x02); // Q2
        assert_eq!((hw >> 13) & 0x7, 0b110);
    }

    #[test]
    fn test_c_addi16sp() {
        // c.addi16sp 16
        let hw = encode_rvc("c.addi16sp", vec![imm(16)]);
        assert_eq!(hw & 0x3, 0x01); // Q1
        assert_eq!((hw >> 13) & 0x7, 0b011); // funct3
        assert_eq!((hw >> 7) & 0x1F, 2); // rd=sp(x2)
    }

    #[test]
    fn test_c_addi4spn() {
        // c.addi4spn x8, 4: CIW-type, funct3=000, imm=4
        let hw = encode_rvc("c.addi4spn", vec![r(8), imm(4)]);
        assert_eq!(hw & 0x3, 0x00); // Q0
        assert_eq!((hw >> 13) & 0x7, 0b000); // funct3
    }

    // ── RV64C tests ─────────────────────────────────────────

    #[test]
    fn test_c_addiw() {
        let hw = encode_rvc64("c.addiw", vec![r(10), imm(3)]);
        assert_eq!(hw & 0x3, 0x01); // Q1
        assert_eq!((hw >> 13) & 0x7, 0b001); // funct3 for addiw
        assert_eq!((hw >> 7) & 0x1F, 10);
        assert_eq!((hw >> 2) & 0x1F, 3);
    }

    #[test]
    fn test_c_subw() {
        let hw = encode_rvc64("c.subw", vec![r(8), r(9)]);
        assert_eq!((hw >> 10) & 0x3F, 0b100111); // funct6
        assert_eq!((hw >> 5) & 0x3, 0b00); // funct2 for subw
    }

    #[test]
    fn test_c_addw() {
        let hw = encode_rvc64("c.addw", vec![r(8), r(9)]);
        assert_eq!((hw >> 10) & 0x3F, 0b100111); // funct6
        assert_eq!((hw >> 5) & 0x3, 0b01); // funct2 for addw
    }

    #[test]
    fn test_c_ld_rv64() {
        let hw = encode_rvc64("c.ld", vec![r(8), memop(9, 0)]);
        assert_eq!(hw & 0x3, 0x00); // Q0
        assert_eq!((hw >> 13) & 0x7, 0b011);
    }

    #[test]
    fn test_c_sd_rv64() {
        let hw = encode_rvc64("c.sd", vec![r(8), memop(9, 0)]);
        assert_eq!(hw & 0x3, 0x00); // Q0
        assert_eq!((hw >> 13) & 0x7, 0b111);
    }

    #[test]
    fn test_c_ldsp_rv64() {
        let hw = encode_rvc64("c.ldsp", vec![r(10), imm(0)]);
        assert_eq!(hw & 0x3, 0x02); // Q2
        assert_eq!((hw >> 13) & 0x7, 0b011);
    }

    #[test]
    fn test_c_sdsp_rv64() {
        let hw = encode_rvc64("c.sdsp", vec![r(5), imm(0)]);
        assert_eq!(hw & 0x3, 0x02); // Q2
        assert_eq!((hw >> 13) & 0x7, 0b111);
    }

    // ── C-extension error tests ─────────────────────────────

    #[test]
    fn test_c_mv_x0_rejected() {
        let instr = make_instr("c.mv", vec![r(0), r(1)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    #[test]
    fn test_c_li_x0_rejected() {
        let instr = make_instr("c.li", vec![r(0), imm(5)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    #[test]
    fn test_c_sub_non_compact_rejected() {
        // c.sub requires x8-x15 registers
        let instr = make_instr("c.sub", vec![r(1), r(2)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    #[test]
    fn test_c_lw_non_compact_base_rejected() {
        // c.lw requires compact register for base
        let instr = make_instr("c.lw", vec![r(8), memop(1, 0)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    #[test]
    fn test_c_li_overflow_rejected() {
        let instr = make_instr("c.li", vec![r(1), imm(32)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    #[test]
    fn test_c_lw_misaligned_rejected() {
        let instr = make_instr("c.lw", vec![r(8), memop(9, 3)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    #[test]
    fn test_c_addi16sp_zero_rejected() {
        let instr = make_instr("c.addi16sp", vec![imm(0)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    #[test]
    fn test_c_addi16sp_not_aligned() {
        let instr = make_instr("c.addi16sp", vec![imm(8)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    #[test]
    fn test_c_addi4spn_zero_rejected() {
        let instr = make_instr("c.addi4spn", vec![r(8), imm(0)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    // ── Auto-narrowing unit tests ───────────────────────────

    #[test]
    fn test_try_compress_nop() {
        let ops: Vec<Operand> = vec![];
        assert!(try_compress("nop", &ops, false, Span::dummy()).is_some());
    }

    #[test]
    fn test_try_compress_ebreak() {
        let ops: Vec<Operand> = vec![];
        assert!(try_compress("ebreak", &ops, false, Span::dummy()).is_some());
    }

    #[test]
    fn test_try_compress_addi_compressible() {
        // addi x10, x10, 5 → c.addi x10, 5 (x10 != x0, x10 matches, 5 fits in 6-bit)
        let Operand::Register(r10) = r(10) else {
            unreachable!()
        };
        let ops = vec![Operand::Register(r10), Operand::Register(r10), imm(5)];
        let hw = try_compress("addi", &ops, false, Span::dummy());
        assert!(hw.is_some());
        let hw = hw.unwrap();
        assert_eq!(hw & 0x3, 0x01); // Q1
        assert_eq!((hw >> 13) & 0x7, 0b000); // c.addi funct3
    }

    #[test]
    fn test_try_compress_addi_zero_imm_rejected() {
        // addi x10, x10, 0 → NOT compressible (c.addi forbids imm=0)
        let Operand::Register(r10) = r(10) else {
            unreachable!()
        };
        let ops = vec![Operand::Register(r10), Operand::Register(r10), imm(0)];
        assert!(try_compress("addi", &ops, false, Span::dummy()).is_none());
    }

    #[test]
    fn test_try_compress_addi_different_regs_not_compressed() {
        // addi x10, x11, 5 → NOT c.addi (rd != rs1)
        let Operand::Register(r10) = r(10) else {
            unreachable!()
        };
        let Operand::Register(r11) = r(11) else {
            unreachable!()
        };
        let ops = vec![Operand::Register(r10), Operand::Register(r11), imm(5)];
        // This could be c.li if rs1==x0, but x11 != x0, so not c.li either.
        // And rd != rs1 so not c.addi.
        assert!(try_compress("addi", &ops, false, Span::dummy()).is_none());
    }

    #[test]
    fn test_try_compress_mv() {
        // add x1, x0, x2 → c.mv x1, x2
        let Operand::Register(r0) = r(0) else {
            unreachable!()
        };
        let Operand::Register(r1) = r(1) else {
            unreachable!()
        };
        let Operand::Register(r2) = r(2) else {
            unreachable!()
        };
        let ops = vec![
            Operand::Register(r1),
            Operand::Register(r0),
            Operand::Register(r2),
        ];
        let hw = try_compress("add", &ops, false, Span::dummy());
        assert!(hw.is_some());
    }

    #[test]
    fn test_try_compress_lw_sp() {
        // lw x10, 0(x2) → c.lwsp x10, 0
        let ops = vec![r(10), memop(2, 0)];
        let hw = try_compress("lw", &ops, false, Span::dummy());
        assert!(hw.is_some());
    }

    #[test]
    fn test_try_compress_sw_sp() {
        // sw x5, 0(x2) → c.swsp x5, 0
        let ops = vec![r(5), memop(2, 0)];
        let hw = try_compress("sw", &ops, false, Span::dummy());
        assert!(hw.is_some());
    }

    #[test]
    fn test_try_compress_lw_compact() {
        // lw x8, 0(x9) → c.lw x8', 0(x9')
        let ops = vec![r(8), memop(9, 0)];
        let hw = try_compress("lw", &ops, false, Span::dummy());
        assert!(hw.is_some());
    }

    #[test]
    fn test_try_compress_lw_non_compact_fails() {
        // lw x1, 0(x2) — x1 not in x8-x15, x2 is SP but rd=x1 is fine for lwsp
        // Actually lw x1, 0(x2) → c.lwsp x1, 0 (x1 != x0, base=x2=sp)
        let ops = vec![r(1), memop(2, 0)];
        assert!(try_compress("lw", &ops, false, Span::dummy()).is_some()); // lwsp
                                                                           // lw x1, 0(x3) — x1 not compact, x3 not compact → can't compress
        let ops2 = vec![r(1), memop(3, 0)];
        assert!(try_compress("lw", &ops2, false, Span::dummy()).is_none());
    }

    // ── C-extension label relocation tests ──────────────────

    #[test]
    fn test_c_beqz_label() {
        let instr = make_instr("c.beqz", vec![r(8), Operand::Label("target".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        // Now produces a relaxable fragment: short=2B CB-type, long=8B inverted-B+JAL
        assert!(result.relax.is_some());
        let ri = result.relax.as_ref().unwrap();
        assert_eq!(ri.short_bytes.len(), 2);
        let sr = ri.short_relocation.as_ref().unwrap();
        assert!(matches!(sr.kind, RelocKind::RvCBranch8));
        // Long form is 8 bytes with RvJal20 at offset 4
        assert_eq!(result.bytes.len(), 8);
        let lr = result.relocation.as_ref().unwrap();
        assert!(matches!(lr.kind, RelocKind::RvJal20));
        assert_eq!(lr.offset, 4);
    }

    #[test]
    fn test_c_j_label() {
        let instr = make_instr("c.j", vec![Operand::Label("target".into())]);
        let result = encode_riscv(&instr, Arch::Rv32).unwrap();
        // Now produces a relaxable fragment: short=2B CJ-type, long=4B JAL
        assert!(result.relax.is_some());
        let ri = result.relax.as_ref().unwrap();
        assert_eq!(ri.short_bytes.len(), 2);
        let sr = ri.short_relocation.as_ref().unwrap();
        assert!(matches!(sr.kind, RelocKind::RvCJump11));
        // Long form is 4 bytes with RvJal20 at offset 0
        assert_eq!(result.bytes.len(), 4);
        let lr = result.relocation.as_ref().unwrap();
        assert!(matches!(lr.kind, RelocKind::RvJal20));
        assert_eq!(lr.offset, 0);
    }

    // ── C-extension format encoding unit tests ──────────────

    #[test]
    fn test_cr_type_encoding() {
        // cr_type(funct4=1000, rd_rs1=5, rs2=3, op=Q2)
        let hw = cr_type(0b1000, 5, 3, C_OP_Q2);
        assert_eq!(hw & 0x3, 0x02);
        assert_eq!((hw >> 2) & 0x1F, 3);
        assert_eq!((hw >> 7) & 0x1F, 5);
        assert_eq!((hw >> 12) & 0xF, 0b1000);
    }

    #[test]
    fn test_ci_type_encoding() {
        let hw = ci_type(0b010, 0, 10, 5, C_OP_Q1);
        assert_eq!(hw & 0x3, 0x01);
        assert_eq!((hw >> 2) & 0x1F, 5);
        assert_eq!((hw >> 7) & 0x1F, 10);
        assert_eq!((hw >> 12) & 1, 0);
        assert_eq!((hw >> 13) & 0x7, 0b010);
    }

    #[test]
    fn test_ca_type_encoding() {
        // ca_type(funct6=100011, rd_rs1_p=0, funct2=00, rs2_p=1, op=Q1)
        let hw = ca_type(0b100011, 0, 0b00, 1, C_OP_Q1);
        assert_eq!(hw & 0x3, 0x01);
        assert_eq!((hw >> 2) & 0x7, 1);
        assert_eq!((hw >> 5) & 0x3, 0b00);
        assert_eq!((hw >> 7) & 0x7, 0);
        assert_eq!((hw >> 10) & 0x3F, 0b100011);
    }

    #[test]
    fn test_compact_reg_mapping() {
        // x8..x15 → 0..7
        for i in 8u32..=15 {
            assert_eq!(compact_reg(i), Some(i - 8));
        }
        // Others return None
        for i in 0u32..8 {
            assert_eq!(compact_reg(i), None);
        }
        for i in 16u32..32 {
            assert_eq!(compact_reg(i), None);
        }
    }

    // ── F/D extension unit tests ────────────────────────────

    /// Helper to create an FP register operand.
    fn fp(n: u8) -> Operand {
        use Register::*;
        let reg = match n {
            0 => RvF0,
            1 => RvF1,
            2 => RvF2,
            3 => RvF3,
            4 => RvF4,
            5 => RvF5,
            6 => RvF6,
            7 => RvF7,
            8 => RvF8,
            9 => RvF9,
            10 => RvF10,
            11 => RvF11,
            12 => RvF12,
            13 => RvF13,
            14 => RvF14,
            15 => RvF15,
            16 => RvF16,
            17 => RvF17,
            18 => RvF18,
            19 => RvF19,
            20 => RvF20,
            21 => RvF21,
            22 => RvF22,
            23 => RvF23,
            24 => RvF24,
            25 => RvF25,
            26 => RvF26,
            27 => RvF27,
            28 => RvF28,
            29 => RvF29,
            30 => RvF30,
            31 => RvF31,
            _ => panic!("invalid FP register number"),
        };
        Operand::Register(reg)
    }

    /// Helper for FP memory operand with integer base register.
    fn fp_memop(base: u8, disp: i64) -> Operand {
        memop(base, disp)
    }

    // ── FP load/store ───────────────────────────────────────

    #[test]
    fn test_flw() {
        // flw f1, 0(x2)  →  I-type: opcode=0x07, rd=1, funct3=010, rs1=2, imm=0
        let w = encode32("flw", vec![fp(1), fp_memop(2, 0)]);
        let expected = i_type(OP_LOAD_FP, 1, 0b010, 2, 0);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_flw_offset() {
        // flw f5, 16(x10)
        let w = encode32("flw", vec![fp(5), fp_memop(10, 16)]);
        let expected = i_type(OP_LOAD_FP, 5, 0b010, 10, 16);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fld() {
        // fld f3, 8(x4)
        let w = encode32("fld", vec![fp(3), fp_memop(4, 8)]);
        let expected = i_type(OP_LOAD_FP, 3, 0b011, 4, 8);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fsw() {
        // fsw f1, 0(x2)  →  S-type: opcode=0x27, funct3=010, rs1=2, rs2=1, imm=0
        let w = encode32("fsw", vec![fp(1), fp_memop(2, 0)]);
        let expected = s_type(OP_STORE_FP, 0b010, 2, 1, 0);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fsw_offset() {
        // fsw f7, -4(x8)
        let w = encode32("fsw", vec![fp(7), fp_memop(8, -4)]);
        let expected = s_type(OP_STORE_FP, 0b010, 8, 7, -4);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fsd() {
        // fsd f10, 24(x15)
        let w = encode32("fsd", vec![fp(10), fp_memop(15, 24)]);
        let expected = s_type(OP_STORE_FP, 0b011, 15, 10, 24);
        assert_eq!(w, expected);
    }

    // ── FP arithmetic (single) ──────────────────────────────

    #[test]
    fn test_fadd_s() {
        let w = encode32("fadd.s", vec![fp(1), fp(2), fp(3)]);
        let expected = r_type(OP_FP, 1, 0b111, 2, 3, 0b000_0000);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fsub_s() {
        let w = encode32("fsub.s", vec![fp(4), fp(5), fp(6)]);
        let expected = r_type(OP_FP, 4, 0b111, 5, 6, 0b000_0100);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fmul_s() {
        let w = encode32("fmul.s", vec![fp(7), fp(8), fp(9)]);
        let expected = r_type(OP_FP, 7, 0b111, 8, 9, 0b000_1000);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fdiv_s() {
        let w = encode32("fdiv.s", vec![fp(10), fp(11), fp(12)]);
        let expected = r_type(OP_FP, 10, 0b111, 11, 12, 0b000_1100);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fsqrt_s() {
        let w = encode32("fsqrt.s", vec![fp(1), fp(2)]);
        let expected = r_type(OP_FP, 1, 0b111, 2, 0, 0b010_1100);
        assert_eq!(w, expected);
    }

    // ── FP arithmetic (double) ──────────────────────────────

    #[test]
    fn test_fadd_d() {
        let w = encode32("fadd.d", vec![fp(1), fp(2), fp(3)]);
        let expected = r_type(OP_FP, 1, 0b111, 2, 3, 0b000_0001);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fsub_d() {
        let w = encode32("fsub.d", vec![fp(4), fp(5), fp(6)]);
        let expected = r_type(OP_FP, 4, 0b111, 5, 6, 0b000_0101);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fmul_d() {
        let w = encode32("fmul.d", vec![fp(7), fp(8), fp(9)]);
        let expected = r_type(OP_FP, 7, 0b111, 8, 9, 0b000_1001);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fdiv_d() {
        let w = encode32("fdiv.d", vec![fp(10), fp(11), fp(12)]);
        let expected = r_type(OP_FP, 10, 0b111, 11, 12, 0b000_1101);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fsqrt_d() {
        let w = encode32("fsqrt.d", vec![fp(1), fp(2)]);
        let expected = r_type(OP_FP, 1, 0b111, 2, 0, 0b010_1101);
        assert_eq!(w, expected);
    }

    // ── Sign injection / min / max ──────────────────────────

    #[test]
    fn test_fsgnj_s() {
        let w = encode32("fsgnj.s", vec![fp(1), fp(2), fp(3)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b000, 2, 3, 0b001_0000));
    }

    #[test]
    fn test_fsgnjn_s() {
        let w = encode32("fsgnjn.s", vec![fp(1), fp(2), fp(3)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b001, 2, 3, 0b001_0000));
    }

    #[test]
    fn test_fsgnjx_s() {
        let w = encode32("fsgnjx.s", vec![fp(1), fp(2), fp(3)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b010, 2, 3, 0b001_0000));
    }

    #[test]
    fn test_fmin_s() {
        let w = encode32("fmin.s", vec![fp(1), fp(2), fp(3)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b000, 2, 3, 0b001_0100));
    }

    #[test]
    fn test_fmax_s() {
        let w = encode32("fmax.s", vec![fp(1), fp(2), fp(3)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b001, 2, 3, 0b001_0100));
    }

    #[test]
    fn test_fmin_d() {
        let w = encode32("fmin.d", vec![fp(1), fp(2), fp(3)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b000, 2, 3, 0b001_0101));
    }

    #[test]
    fn test_fmax_d() {
        let w = encode32("fmax.d", vec![fp(1), fp(2), fp(3)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b001, 2, 3, 0b001_0101));
    }

    // ── FP compare → integer ────────────────────────────────

    #[test]
    fn test_feq_s() {
        let w = encode32("feq.s", vec![r(10), fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b010, 1, 2, 0b101_0000));
    }

    #[test]
    fn test_flt_s() {
        let w = encode32("flt.s", vec![r(10), fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b001, 1, 2, 0b101_0000));
    }

    #[test]
    fn test_fle_s() {
        let w = encode32("fle.s", vec![r(10), fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b000, 1, 2, 0b101_0000));
    }

    #[test]
    fn test_feq_d() {
        let w = encode32("feq.d", vec![r(10), fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b010, 1, 2, 0b101_0001));
    }

    #[test]
    fn test_flt_d() {
        let w = encode32("flt.d", vec![r(10), fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b001, 1, 2, 0b101_0001));
    }

    #[test]
    fn test_fle_d() {
        let w = encode32("fle.d", vec![r(10), fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b000, 1, 2, 0b101_0001));
    }

    // ── FCLASS ──────────────────────────────────────────────

    #[test]
    fn test_fclass_s() {
        let w = encode32("fclass.s", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b001, 1, 0, 0b111_0000));
    }

    #[test]
    fn test_fclass_d() {
        let w = encode32("fclass.d", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b001, 1, 0, 0b111_0001));
    }

    // ── FP ↔ integer conversions ────────────────────────────

    #[test]
    fn test_fcvt_w_s() {
        let w = encode32("fcvt.w.s", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b111, 1, 0, 0b110_0000));
    }

    #[test]
    fn test_fcvt_wu_s() {
        let w = encode32("fcvt.wu.s", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b111, 1, 1, 0b110_0000));
    }

    #[test]
    fn test_fcvt_s_w() {
        let w = encode32("fcvt.s.w", vec![fp(1), r(10)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b111, 10, 0, 0b110_1000));
    }

    #[test]
    fn test_fcvt_s_wu() {
        let w = encode32("fcvt.s.wu", vec![fp(1), r(10)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b111, 10, 1, 0b110_1000));
    }

    #[test]
    fn test_fcvt_w_d() {
        let w = encode32("fcvt.w.d", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b111, 1, 0, 0b110_0001));
    }

    #[test]
    fn test_fcvt_wu_d() {
        let w = encode32("fcvt.wu.d", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b111, 1, 1, 0b110_0001));
    }

    #[test]
    fn test_fcvt_d_w() {
        let w = encode32("fcvt.d.w", vec![fp(1), r(10)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b111, 10, 0, 0b110_1001));
    }

    #[test]
    fn test_fcvt_d_wu() {
        let w = encode32("fcvt.d.wu", vec![fp(1), r(10)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b111, 10, 1, 0b110_1001));
    }

    #[test]
    fn test_fcvt_s_d() {
        let w = encode32("fcvt.s.d", vec![fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b111, 2, 1, 0b010_0000));
    }

    #[test]
    fn test_fcvt_d_s() {
        let w = encode32("fcvt.d.s", vec![fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b111, 2, 0, 0b010_0001));
    }

    // ── FP move (bitwise) ───────────────────────────────────

    #[test]
    fn test_fmv_x_w() {
        let w = encode32("fmv.x.w", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b000, 1, 0, 0b111_0000));
    }

    #[test]
    fn test_fmv_w_x() {
        let w = encode32("fmv.w.x", vec![fp(1), r(10)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b000, 10, 0, 0b111_1000));
    }

    // ── RV64 conversions ────────────────────────────────────

    #[test]
    fn test_fcvt_l_s_rv64() {
        let w = encode64("fcvt.l.s", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b111, 1, 2, 0b110_0000));
    }

    #[test]
    fn test_fcvt_lu_s_rv64() {
        let w = encode64("fcvt.lu.s", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b111, 1, 3, 0b110_0000));
    }

    #[test]
    fn test_fcvt_s_l_rv64() {
        let w = encode64("fcvt.s.l", vec![fp(1), r(10)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b111, 10, 2, 0b110_1000));
    }

    #[test]
    fn test_fcvt_s_lu_rv64() {
        let w = encode64("fcvt.s.lu", vec![fp(1), r(10)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b111, 10, 3, 0b110_1000));
    }

    #[test]
    fn test_fcvt_l_d_rv64() {
        let w = encode64("fcvt.l.d", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b111, 1, 2, 0b110_0001));
    }

    #[test]
    fn test_fcvt_lu_d_rv64() {
        let w = encode64("fcvt.lu.d", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b111, 1, 3, 0b110_0001));
    }

    #[test]
    fn test_fcvt_d_l_rv64() {
        let w = encode64("fcvt.d.l", vec![fp(1), r(10)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b111, 10, 2, 0b110_1001));
    }

    #[test]
    fn test_fcvt_d_lu_rv64() {
        let w = encode64("fcvt.d.lu", vec![fp(1), r(10)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b111, 10, 3, 0b110_1001));
    }

    #[test]
    fn test_fmv_x_d_rv64() {
        let w = encode64("fmv.x.d", vec![r(10), fp(1)]);
        assert_eq!(w, r_type(OP_FP, 10, 0b000, 1, 0, 0b111_0001));
    }

    #[test]
    fn test_fmv_d_x_rv64() {
        let w = encode64("fmv.d.x", vec![fp(1), r(10)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b000, 10, 0, 0b111_1001));
    }

    // RV64-only should fail on RV32
    #[test]
    fn test_fcvt_l_s_rejects_rv32() {
        let instr = make_instr("fcvt.l.s", vec![r(10), fp(1)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    #[test]
    fn test_fmv_x_d_rejects_rv32() {
        let instr = make_instr("fmv.x.d", vec![r(10), fp(1)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    // ── R4-type fused multiply-add ──────────────────────────

    #[test]
    fn test_fmadd_s() {
        let w = encode32("fmadd.s", vec![fp(1), fp(2), fp(3), fp(4)]);
        let expected = r4_type(OP_MADD, 1, 0b111, 2, 3, 0b00, 4);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fmadd_d() {
        let w = encode32("fmadd.d", vec![fp(1), fp(2), fp(3), fp(4)]);
        let expected = r4_type(OP_MADD, 1, 0b111, 2, 3, 0b01, 4);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fmsub_s() {
        let w = encode32("fmsub.s", vec![fp(1), fp(2), fp(3), fp(4)]);
        let expected = r4_type(OP_MSUB, 1, 0b111, 2, 3, 0b00, 4);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fmsub_d() {
        let w = encode32("fmsub.d", vec![fp(1), fp(2), fp(3), fp(4)]);
        let expected = r4_type(OP_MSUB, 1, 0b111, 2, 3, 0b01, 4);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fnmsub_s() {
        let w = encode32("fnmsub.s", vec![fp(1), fp(2), fp(3), fp(4)]);
        let expected = r4_type(OP_NMSUB, 1, 0b111, 2, 3, 0b00, 4);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fnmsub_d() {
        let w = encode32("fnmsub.d", vec![fp(1), fp(2), fp(3), fp(4)]);
        let expected = r4_type(OP_NMSUB, 1, 0b111, 2, 3, 0b01, 4);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fnmadd_s() {
        let w = encode32("fnmadd.s", vec![fp(1), fp(2), fp(3), fp(4)]);
        let expected = r4_type(OP_NMADD, 1, 0b111, 2, 3, 0b00, 4);
        assert_eq!(w, expected);
    }

    #[test]
    fn test_fnmadd_d() {
        let w = encode32("fnmadd.d", vec![fp(1), fp(2), fp(3), fp(4)]);
        let expected = r4_type(OP_NMADD, 1, 0b111, 2, 3, 0b01, 4);
        assert_eq!(w, expected);
    }

    // ── FP pseudo-instructions ──────────────────────────────

    #[test]
    fn test_fmv_s_pseudo() {
        // fmv.s f1, f2  →  fsgnj.s f1, f2, f2
        let w = encode32("fmv.s", vec![fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b000, 2, 2, 0b001_0000));
    }

    #[test]
    fn test_fmv_d_pseudo() {
        let w = encode32("fmv.d", vec![fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b000, 2, 2, 0b001_0001));
    }

    #[test]
    fn test_fneg_s_pseudo() {
        // fneg.s f1, f2  →  fsgnjn.s f1, f2, f2
        let w = encode32("fneg.s", vec![fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b001, 2, 2, 0b001_0000));
    }

    #[test]
    fn test_fneg_d_pseudo() {
        let w = encode32("fneg.d", vec![fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b001, 2, 2, 0b001_0001));
    }

    #[test]
    fn test_fabs_s_pseudo() {
        // fabs.s f1, f2  →  fsgnjx.s f1, f2, f2
        let w = encode32("fabs.s", vec![fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b010, 2, 2, 0b001_0000));
    }

    #[test]
    fn test_fabs_d_pseudo() {
        let w = encode32("fabs.d", vec![fp(1), fp(2)]);
        assert_eq!(w, r_type(OP_FP, 1, 0b010, 2, 2, 0b001_0001));
    }

    // ── FP encoding bit-exact verification ──────────────────

    #[test]
    fn test_flw_bit_fields() {
        // flw f0, 0(x0) → all zeros except opcode=0b0000111 and funct3=010
        let w = encode32("flw", vec![fp(0), fp_memop(0, 0)]);
        assert_eq!(w & 0x7F, OP_LOAD_FP);
        assert_eq!((w >> 12) & 0x7, 0b010);
        assert_eq!((w >> 7) & 0x1F, 0); // rd=f0
        assert_eq!((w >> 15) & 0x1F, 0); // rs1=x0
        assert_eq!((w >> 20) & 0xFFF, 0); // imm=0
    }

    #[test]
    fn test_fadd_s_bit_fields() {
        // fadd.s f1, f2, f3: opcode=0x53, funct7=0x00, rm=111
        let w = encode32("fadd.s", vec![fp(1), fp(2), fp(3)]);
        assert_eq!(w & 0x7F, OP_FP); // opcode
        assert_eq!((w >> 7) & 0x1F, 1); // rd=f1
        assert_eq!((w >> 12) & 0x7, 0b111); // rm=dynamic
        assert_eq!((w >> 15) & 0x1F, 2); // rs1=f2
        assert_eq!((w >> 20) & 0x1F, 3); // rs2=f3
        assert_eq!((w >> 25) & 0x7F, 0b000_0000); // funct7=FADD
    }

    #[test]
    fn test_r4_type_bit_fields() {
        // fmadd.s f1, f2, f3, f4: verify R4-type encoding
        let w = encode32("fmadd.s", vec![fp(1), fp(2), fp(3), fp(4)]);
        assert_eq!(w & 0x7F, OP_MADD); // opcode
        assert_eq!((w >> 7) & 0x1F, 1); // rd=f1
        assert_eq!((w >> 12) & 0x7, 0b111); // rm=dynamic
        assert_eq!((w >> 15) & 0x1F, 2); // rs1=f2
        assert_eq!((w >> 20) & 0x1F, 3); // rs2=f3
        assert_eq!((w >> 25) & 0x3, 0b00); // fmt=S
        assert_eq!((w >> 27) & 0x1F, 4); // rs3=f4
    }

    #[test]
    fn test_fsgnj_d_bit_fields() {
        let w = encode32("fsgnj.d", vec![fp(10), fp(11), fp(12)]);
        assert_eq!(w & 0x7F, OP_FP);
        assert_eq!((w >> 25) & 0x7F, 0b001_0001); // funct7 for .D sign-inject
        assert_eq!((w >> 12) & 0x7, 0b000); // funct3 for FSGNJ
    }

    // ── FP error path ───────────────────────────────────────

    #[test]
    fn test_fpreg_rejects_integer_register() {
        // fadd.s should reject integer registers
        let instr = make_instr("fadd.s", vec![r(1), fp(2), fp(3)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    #[test]
    fn test_fmv_x_w_rejects_fp_rd() {
        // fmv.x.w expects integer rd, fp rs1
        let instr = make_instr("fmv.x.w", vec![fp(1), fp(2)]);
        assert!(encode_riscv(&instr, Arch::Rv32).is_err());
    }

    // ── V-extension (RVV) tests ─────────────────────────────

    fn v(n: u8) -> Operand {
        use Register::*;
        let reg = match n {
            0 => RvV0,
            1 => RvV1,
            2 => RvV2,
            3 => RvV3,
            4 => RvV4,
            5 => RvV5,
            6 => RvV6,
            7 => RvV7,
            8 => RvV8,
            9 => RvV9,
            10 => RvV10,
            11 => RvV11,
            12 => RvV12,
            13 => RvV13,
            14 => RvV14,
            15 => RvV15,
            16 => RvV16,
            17 => RvV17,
            18 => RvV18,
            19 => RvV19,
            20 => RvV20,
            21 => RvV21,
            22 => RvV22,
            23 => RvV23,
            24 => RvV24,
            25 => RvV25,
            26 => RvV26,
            27 => RvV27,
            28 => RvV28,
            29 => RvV29,
            30 => RvV30,
            31 => RvV31,
            _ => panic!("invalid V register {}", n),
        };
        Operand::Register(reg)
    }

    fn label(s: &str) -> Operand {
        Operand::Label(String::from(s))
    }

    fn rv_mem_base(n: u8) -> Operand {
        Operand::Memory(Box::new(MemoryOperand {
            base: Some(match n {
                10 => Register::RvX10, // a0
                11 => Register::RvX11, // a1
                _ => Register::RvX0,
            }),
            ..Default::default()
        }))
    }

    #[test]
    fn rvv_vsetvli() {
        // vsetvli a0, a1, e32, m1, ta, ma → [0x57,0xf5,0x05,0x0d] = 0x0D05F557
        let w = encode64(
            "vsetvli",
            vec![
                r(10),
                r(11),
                label("e32"),
                label("m1"),
                label("ta"),
                label("ma"),
            ],
        );
        assert_eq!(w, 0x0D05_F557);
    }

    #[test]
    fn rvv_vsetivli() {
        // vsetivli a0, 16, e32, m1, ta, ma → [0x57,0x75,0x08,0xcd] = 0xCD087557
        let w = encode64(
            "vsetivli",
            vec![
                r(10),
                imm(16),
                label("e32"),
                label("m1"),
                label("ta"),
                label("ma"),
            ],
        );
        assert_eq!(w, 0xCD08_7557);
    }

    #[test]
    fn rvv_vsetvl() {
        // vsetvl a0, a1, a2 → [0x57,0xf5,0xc5,0x80] = 0x80C5F557
        let w = encode64("vsetvl", vec![r(10), r(11), r(12)]);
        assert_eq!(w, 0x80C5_F557);
    }

    #[test]
    fn rvv_vle8_v() {
        // vle8.v v1, (a0) → [0x87,0x00,0x05,0x02] = 0x02050087
        let w = encode64("vle8.v", vec![v(1), rv_mem_base(10)]);
        assert_eq!(w, 0x0205_0087);
    }

    #[test]
    fn rvv_vse8_v() {
        // vse8.v v1, (a0) → [0xa7,0x00,0x05,0x02] = 0x020500A7
        let w = encode64("vse8.v", vec![v(1), rv_mem_base(10)]);
        assert_eq!(w, 0x0205_00A7);
    }

    #[test]
    fn rvv_vle32_v() {
        // vle32.v v1, (a0) → [0x87,0x60,0x05,0x02] = 0x02056087
        let w = encode64("vle32.v", vec![v(1), rv_mem_base(10)]);
        assert_eq!(w, 0x0205_6087);
    }

    #[test]
    fn rvv_vse32_v() {
        // vse32.v v1, (a0) → [0xa7,0x60,0x05,0x02] = 0x020560A7
        let w = encode64("vse32.v", vec![v(1), rv_mem_base(10)]);
        assert_eq!(w, 0x0205_60A7);
    }

    #[test]
    fn rvv_vadd_vv() {
        // vadd.vv v1, v2, v3 → [0xd7,0x80,0x21,0x02] = 0x022180D7
        let w = encode64("vadd.vv", vec![v(1), v(2), v(3)]);
        assert_eq!(w, 0x0221_80D7);
    }

    #[test]
    fn rvv_vsub_vv() {
        // vsub.vv v1, v2, v3 → [0xd7,0x80,0x21,0x0a] = 0x0A2180D7
        let w = encode64("vsub.vv", vec![v(1), v(2), v(3)]);
        assert_eq!(w, 0x0A21_80D7);
    }

    #[test]
    fn rvv_vand_vv() {
        // vand.vv v1, v2, v3 → [0xd7,0x80,0x21,0x26] = 0x262180D7
        let w = encode64("vand.vv", vec![v(1), v(2), v(3)]);
        assert_eq!(w, 0x2621_80D7);
    }

    #[test]
    fn rvv_vor_vv() {
        // vor.vv v1, v2, v3 → [0xd7,0x80,0x21,0x2a] = 0x2A2180D7
        let w = encode64("vor.vv", vec![v(1), v(2), v(3)]);
        assert_eq!(w, 0x2A21_80D7);
    }

    #[test]
    fn rvv_vxor_vv() {
        // vxor.vv v1, v2, v3 → [0xd7,0x80,0x21,0x2e] = 0x2E2180D7
        let w = encode64("vxor.vv", vec![v(1), v(2), v(3)]);
        assert_eq!(w, 0x2E21_80D7);
    }

    #[test]
    fn rvv_vmul_vv() {
        // vmul.vv v1, v2, v3 → [0xd7,0xa0,0x21,0x96] = 0x9621A0D7
        let w = encode64("vmul.vv", vec![v(1), v(2), v(3)]);
        assert_eq!(w, 0x9621_A0D7);
    }

    #[test]
    fn rvv_vadd_vx() {
        // vadd.vx v1, v2, a0 → [0xd7,0x40,0x25,0x02] = 0x022540D7
        let w = encode64("vadd.vx", vec![v(1), v(2), r(10)]);
        assert_eq!(w, 0x0225_40D7);
    }

    #[test]
    fn rvv_vadd_vi() {
        // vadd.vi v1, v2, 5 → [0xd7,0xb0,0x22,0x02] = 0x0222B0D7
        let w = encode64("vadd.vi", vec![v(1), v(2), imm(5)]);
        assert_eq!(w, 0x0222_B0D7);
    }

    #[test]
    fn rvv_vle32_v_masked() {
        // vle32.v v1, (a0), v0.t → [0x87,0x60,0x05,0x00] = 0x00056087
        let w = encode64("vle32.v", vec![v(1), rv_mem_base(10), v(0)]);
        assert_eq!(w, 0x0005_6087);
    }
}
