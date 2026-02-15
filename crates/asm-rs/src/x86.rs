//! x86-64 unified instruction encoding module.
//!
//! This module contains the table-driven instruction encoding dispatch for
//! x86-64, including all special instruction encoders. Previously split across
//! `x86_gen.rs` (dispatch tables) and `x86_insn.rs` (special encoders), now
//! unified into a single module for simpler maintenance and zero indirection.
//!
//! ## Instruction Classes
//!
//! - **Fixed encoding** (zero-operand): binary search in const table
//! - **ALU class**: `ADD`/`OR`/`ADC`/`SBB`/`AND`/`SUB`/`XOR`/`CMP`
//! - **Unary class**: `NOT`/`NEG`/`MUL`/`DIV`/`IDIV`
//! - **Shift class**: `SHL`/`SHR`/`SAR`/`ROL`/`ROR`/`RCL`/`RCR`
//! - **Jcc class**: all 16 condition codes with aliases
//! - **SETcc class**: all 16 condition codes with aliases
//! - **CMOVcc class**: all 16 condition codes with aliases
//! - **BT family**: `BT`/`BTS`/`BTR`/`BTC`
//! - **SSE/SSE2/SSE3/SSSE3/SSE4**: 100+ SIMD instructions
//! - **Complex**: `MOV`/`LEA`/`PUSH`/`POP`/`CALL`/`JMP`/`IMUL`/`XCHG`/etc.
//! - **Special**: `CMPXCHG`/`XADD`/`IN`/`OUT`/`ENTER`/`JECXZ`/`JRCXZ`/
//!   `RDRAND`/`RDSEED`/`CMPXCHG8B`/`CMPXCHG16B`/`MOVNTI`/`MOVBE`/`SHLD`/`SHRD`
//!
//! ## Adding a New Instruction
//!
//! 1. Add a match arm in [`dispatch_x86_64`] in the appropriate section.
//! 2. Implement the encoder function in `encoder.rs` or in this module.
//! 3. Add a unit test cross-validated against iced-x86.

use crate::encoder::{
    check_high_byte_rex_conflict, emit_mem_modrm, emit_rex_for_digit_mem, emit_rex_for_reg_mem,
    extract_label, invalid_operands, modrm, needs_rex, reg_size, rex, set_mem_reloc, InstrBytes,
    RelocKind, Relocation,
};
use crate::error::AsmError;
use crate::ir::*;

// ─── Fixed-encoding table ───────────────────────────────────────────────────

/// Fixed-encoding (zero-operand) instruction table.
/// Sorted by mnemonic for binary search lookup.
const FIXED_TABLE: &[(&str, &[u8])] = &[
    ("cbw", &[0x66, 0x98]),
    ("cc", &[0xCC]),
    ("cdq", &[0x99]),
    ("cdqe", &[0x48, 0x98]),
    ("clc", &[0xF8]),
    ("cld", &[0xFC]),
    ("cli", &[0xFA]),
    ("clts", &[0x0F, 0x06]),
    ("cmc", &[0xF5]),
    ("cmpsb", &[0xA6]),
    ("cmpsd", &[0xA7]),
    ("cmpsq", &[0x48, 0xA7]),
    ("cmpsw", &[0x66, 0xA7]),
    ("cpuid", &[0x0F, 0xA2]),
    ("cqo", &[0x48, 0x99]),
    ("cwd", &[0x66, 0x99]),
    ("cwde", &[0x98]),
    ("emms", &[0x0F, 0x77]),
    ("endbr32", &[0xF3, 0x0F, 0x1E, 0xFB]),
    ("endbr64", &[0xF3, 0x0F, 0x1E, 0xFA]),
    ("hlt", &[0xF4]),
    ("icebp", &[0xF1]),
    ("insb", &[0x6C]),
    ("insd", &[0x6D]),
    ("insw", &[0x66, 0x6D]),
    ("int1", &[0xF1]),
    ("int3", &[0xCC]),
    ("invd", &[0x0F, 0x08]),
    ("iret", &[0xCF]),
    ("iretd", &[0xCF]),
    ("iretq", &[0x48, 0xCF]),
    ("lahf", &[0x9F]),
    ("leave", &[0xC9]),
    ("lfence", &[0x0F, 0xAE, 0xE8]),
    ("lodsb", &[0xAC]),
    ("lodsd", &[0xAD]),
    ("lodsq", &[0x48, 0xAD]),
    ("lodsw", &[0x66, 0xAD]),
    ("mfence", &[0x0F, 0xAE, 0xF0]),
    ("monitor", &[0x0F, 0x01, 0xC8]),
    ("movsb", &[0xA4]),
    ("movsd", &[0xA5]),
    ("movsq", &[0x48, 0xA5]),
    ("movsw", &[0x66, 0xA5]),
    ("mwait", &[0x0F, 0x01, 0xC9]),
    ("outsb", &[0x6E]),
    ("outsd", &[0x6F]),
    ("outsw", &[0x66, 0x6F]),
    ("pause", &[0xF3, 0x90]),
    ("popf", &[0x9D]),
    ("popfq", &[0x9D]),
    ("popfw", &[0x66, 0x9D]),
    ("pushf", &[0x9C]),
    ("pushfq", &[0x9C]),
    ("pushfw", &[0x66, 0x9C]),
    ("rdmsr", &[0x0F, 0x32]),
    ("rdtsc", &[0x0F, 0x31]),
    ("rdtscp", &[0x0F, 0x01, 0xF9]),
    ("sahf", &[0x9E]),
    ("scasb", &[0xAE]),
    ("scasd", &[0xAF]),
    ("scasq", &[0x48, 0xAF]),
    ("scasw", &[0x66, 0xAF]),
    ("sfence", &[0x0F, 0xAE, 0xF8]),
    ("stc", &[0xF9]),
    ("std", &[0xFD]),
    ("sti", &[0xFB]),
    ("stosb", &[0xAA]),
    ("stosd", &[0xAB]),
    ("stosq", &[0x48, 0xAB]),
    ("stosw", &[0x66, 0xAB]),
    ("swapgs", &[0x0F, 0x01, 0xF8]),
    ("syscall", &[0x0F, 0x05]),
    ("sysenter", &[0x0F, 0x34]),
    ("sysexit", &[0x0F, 0x35]),
    ("sysret", &[0x0F, 0x07]),
    ("ud2", &[0x0F, 0x0B]),
    ("vzeroall", &[0xC5, 0xFC, 0x77]),
    ("vzeroupper", &[0xC5, 0xF8, 0x77]),
    ("wbinvd", &[0x0F, 0x09]),
    ("wrmsr", &[0x0F, 0x30]),
    ("xend", &[0x0F, 0x01, 0xD5]),
    ("xgetbv", &[0x0F, 0x01, 0xD0]),
    ("xlat", &[0xD7]),
    ("xlatb", &[0xD7]),
    ("xsetbv", &[0x0F, 0x01, 0xD1]),
    ("xtest", &[0x0F, 0x01, 0xD6]),
];

/// Look up a fixed-encoding instruction by mnemonic.
/// Returns `Some(bytes)` if found, `None` otherwise.
#[inline]
fn lookup_fixed(mnemonic: &str) -> Option<&'static [u8]> {
    FIXED_TABLE
        .binary_search_by_key(&mnemonic, |&(m, _)| m)
        .ok()
        .map(|idx| FIXED_TABLE[idx].1)
}

// ─── Immediate validators ───────────────────────────────────────────────────

/// Validate that an immediate value fits in an unsigned byte (0..=255).
#[inline]
fn validate_imm_u8(imm: i128, span: crate::error::Span) -> Result<(), AsmError> {
    if !(0..=255).contains(&imm) {
        return Err(AsmError::ImmediateOverflow {
            value: imm,
            min: 0,
            max: 255,
            span,
        });
    }
    Ok(())
}

/// Validate that an immediate value fits in an unsigned 16-bit word (0..=65535).
#[inline]
fn validate_imm_u16(imm: i128, span: crate::error::Span) -> Result<(), AsmError> {
    if !(0..=0xFFFF).contains(&imm) {
        return Err(AsmError::ImmediateOverflow {
            value: imm,
            min: 0,
            max: 0xFFFF,
            span,
        });
    }
    Ok(())
}

// ─── Instruction dispatch ───────────────────────────────────────────────────

/// Returns true if any operand is a ZMM register.
/// Used as a guard to dispatch to EVEX-encoded forms instead of VEX.
fn has_zmm(ops: &[Operand]) -> bool {
    ops.iter()
        .any(|op| matches!(op, Operand::Register(r) if r.is_zmm()))
}

/// x86-64 instruction dispatch.
///
/// Attempts to encode the given instruction using the table-driven
/// approach. Returns `None` if the mnemonic is not recognized
/// (caller should report `UnknownMnemonic`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_x86_64(
    mnemonic: &str,
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<crate::encoder::Relocation>,
    relax_info: &mut Option<crate::encoder::RelaxInfo>,
) -> Option<Result<(), AsmError>> {
    // 1. Fixed-encoding (zero-operand) instructions
    if let Some(bytes) = lookup_fixed(mnemonic) {
        if ops.is_empty() {
            buf.extend_from_slice(bytes);
            return Some(Ok(()));
        }
        // Fall through: operands present → try parameterised dispatch
        // (handles cmpsd/movsd SSE forms that share a mnemonic with string ops).
    }

    // 2. Parameterized instruction classes
    match mnemonic {
        // ── NOP family ──────────────────────────────────────────────
        "nop" => Some(crate::encoder::encode_nop(buf, ops, instr)),
        "nop2" => Some(crate::encoder::encode_multibyte_nop(buf, "nop2")),
        "nop3" => Some(crate::encoder::encode_multibyte_nop(buf, "nop3")),
        "nop4" => Some(crate::encoder::encode_multibyte_nop(buf, "nop4")),
        "nop5" => Some(crate::encoder::encode_multibyte_nop(buf, "nop5")),
        "nop6" => Some(crate::encoder::encode_multibyte_nop(buf, "nop6")),
        "nop7" => Some(crate::encoder::encode_multibyte_nop(buf, "nop7")),
        "nop8" => Some(crate::encoder::encode_multibyte_nop(buf, "nop8")),
        "nop9" => Some(crate::encoder::encode_multibyte_nop(buf, "nop9")),

        // ── INT ─────────────────────────────────────────────────────
        "int" => Some(crate::encoder::encode_int(buf, ops, instr)),

        // ── RET family ──────────────────────────────────────────────
        "ret" | "retn" => Some(crate::encoder::encode_ret(buf, ops, instr)),
        "retf" | "lret" => Some(crate::encoder::encode_retf(buf, ops, instr)),

        // ── Data movement ───────────────────────────────────────────
        "mov" | "movabs" => Some(crate::encoder::encode_mov(buf, ops, instr, reloc)),
        "movzx" => Some(crate::encoder::encode_movzx(buf, ops, instr)),
        "movsx" | "movsxd" => Some(crate::encoder::encode_movsx(buf, ops, instr)),
        "lea" => Some(crate::encoder::encode_lea(buf, ops, instr, reloc)),
        "xchg" => Some(crate::encoder::encode_xchg(buf, ops, instr)),

        // ── Stack ───────────────────────────────────────────────────
        "push" => Some(crate::encoder::encode_push(buf, ops, instr, reloc)),
        "pop" => Some(crate::encoder::encode_pop(buf, ops, instr)),

        // ── ALU class ───────────────────────────────────────────────
        "add" => Some(crate::encoder::encode_alu(buf, ops, instr, 0, reloc)),
        "or" => Some(crate::encoder::encode_alu(buf, ops, instr, 1, reloc)),
        "adc" => Some(crate::encoder::encode_alu(buf, ops, instr, 2, reloc)),
        "sbb" => Some(crate::encoder::encode_alu(buf, ops, instr, 3, reloc)),
        "and" => Some(crate::encoder::encode_alu(buf, ops, instr, 4, reloc)),
        "sub" => Some(crate::encoder::encode_alu(buf, ops, instr, 5, reloc)),
        "xor" => Some(crate::encoder::encode_alu(buf, ops, instr, 6, reloc)),
        "cmp" => Some(crate::encoder::encode_alu(buf, ops, instr, 7, reloc)),

        // ── TEST ────────────────────────────────────────────────────
        "test" => Some(crate::encoder::encode_test(buf, ops, instr, reloc)),

        // ── Unary class ─────────────────────────────────────────────
        "not" => Some(crate::encoder::encode_unary(buf, ops, instr, 2)),
        "neg" => Some(crate::encoder::encode_unary(buf, ops, instr, 3)),
        "mul" => Some(crate::encoder::encode_unary(buf, ops, instr, 4)),
        "div" => Some(crate::encoder::encode_unary(buf, ops, instr, 6)),
        "idiv" => Some(crate::encoder::encode_unary(buf, ops, instr, 7)),
        "imul" => Some(crate::encoder::encode_imul(buf, ops, instr)),

        // ── INC / DEC ───────────────────────────────────────────────
        "inc" => Some(crate::encoder::encode_inc_dec(buf, ops, instr, 0)),
        "dec" => Some(crate::encoder::encode_inc_dec(buf, ops, instr, 1)),

        // ── Shift / Rotate class ────────────────────────────────────
        "rol" => Some(crate::encoder::encode_shift(buf, ops, instr, 0)),
        "ror" => Some(crate::encoder::encode_shift(buf, ops, instr, 1)),
        "rcl" => Some(crate::encoder::encode_shift(buf, ops, instr, 2)),
        "rcr" => Some(crate::encoder::encode_shift(buf, ops, instr, 3)),
        "shl" | "sal" => Some(crate::encoder::encode_shift(buf, ops, instr, 4)),
        "shr" => Some(crate::encoder::encode_shift(buf, ops, instr, 5)),
        "sar" => Some(crate::encoder::encode_shift(buf, ops, instr, 7)),

        // ── JMP / CALL ──────────────────────────────────────────────
        "jmp" => Some(crate::encoder::encode_jmp(
            buf, ops, instr, reloc, relax_info,
        )),
        "call" => Some(crate::encoder::encode_call(buf, ops, instr, reloc)),

        // ── Jcc class (condition code branches) ─────────────────────
        "jo" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0x0, reloc, relax_info,
        )),
        "jno" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0x1, reloc, relax_info,
        )),
        "jb" | "jc" | "jnae" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0x2, reloc, relax_info,
        )),
        "jnb" | "jnc" | "jae" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0x3, reloc, relax_info,
        )),
        "je" | "jz" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0x4, reloc, relax_info,
        )),
        "jne" | "jnz" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0x5, reloc, relax_info,
        )),
        "jbe" | "jna" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0x6, reloc, relax_info,
        )),
        "jnbe" | "ja" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0x7, reloc, relax_info,
        )),
        "js" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0x8, reloc, relax_info,
        )),
        "jns" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0x9, reloc, relax_info,
        )),
        "jp" | "jpe" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0xA, reloc, relax_info,
        )),
        "jnp" | "jpo" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0xB, reloc, relax_info,
        )),
        "jl" | "jnge" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0xC, reloc, relax_info,
        )),
        "jnl" | "jge" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0xD, reloc, relax_info,
        )),
        "jle" | "jng" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0xE, reloc, relax_info,
        )),
        "jnle" | "jg" => Some(crate::encoder::encode_jcc(
            buf, ops, instr, 0xF, reloc, relax_info,
        )),

        // ── LOOP class ──────────────────────────────────────────────
        "loopne" | "loopnz" => Some(crate::encoder::encode_loop(
            buf, ops, instr, 0xE0, reloc, relax_info,
        )),
        "loope" | "loopz" => Some(crate::encoder::encode_loop(
            buf, ops, instr, 0xE1, reloc, relax_info,
        )),
        "loop" => Some(crate::encoder::encode_loop(
            buf, ops, instr, 0xE2, reloc, relax_info,
        )),

        // ── SETcc class ─────────────────────────────────────────────
        "seto" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0x0)),
        "setno" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0x1)),
        "setb" | "setc" | "setnae" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0x2)),
        "setnb" | "setnc" | "setae" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0x3)),
        "sete" | "setz" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0x4)),
        "setne" | "setnz" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0x5)),
        "setbe" | "setna" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0x6)),
        "setnbe" | "seta" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0x7)),
        "sets" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0x8)),
        "setns" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0x9)),
        "setp" | "setpe" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0xA)),
        "setnp" | "setpo" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0xB)),
        "setl" | "setnge" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0xC)),
        "setnl" | "setge" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0xD)),
        "setle" | "setng" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0xE)),
        "setnle" | "setg" => Some(crate::encoder::encode_setcc(buf, ops, instr, 0xF)),

        // ── CMOVcc class ────────────────────────────────────────────
        "cmovo" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0x0)),
        "cmovno" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0x1)),
        "cmovb" | "cmovc" | "cmovnae" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0x2)),
        "cmovnb" | "cmovnc" | "cmovae" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0x3)),
        "cmove" | "cmovz" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0x4)),
        "cmovne" | "cmovnz" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0x5)),
        "cmovbe" | "cmovna" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0x6)),
        "cmovnbe" | "cmova" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0x7)),
        "cmovs" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0x8)),
        "cmovns" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0x9)),
        "cmovp" | "cmovpe" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0xA)),
        "cmovnp" | "cmovpo" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0xB)),
        "cmovl" | "cmovnge" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0xC)),
        "cmovnl" | "cmovge" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0xD)),
        "cmovle" | "cmovng" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0xE)),
        "cmovnle" | "cmovg" => Some(crate::encoder::encode_cmovcc(buf, ops, instr, 0xF)),

        // ── BT family ───────────────────────────────────────────────
        "bt" => Some(crate::encoder::encode_bt(buf, ops, instr, 4)),
        "bts" => Some(crate::encoder::encode_bt(buf, ops, instr, 5)),
        "btr" => Some(crate::encoder::encode_bt(buf, ops, instr, 6)),
        "btc" => Some(crate::encoder::encode_bt(buf, ops, instr, 7)),

        // ── BSF / BSR ───────────────────────────────────────────────
        "bsf" => Some(crate::encoder::encode_bsf_bsr(buf, ops, instr, 0xBC)),
        "bsr" => Some(crate::encoder::encode_bsf_bsr(buf, ops, instr, 0xBD)),

        // ── Extension ops ───────────────────────────────────────────
        "popcnt" => Some(crate::encoder::encode_popcnt(buf, ops, instr)),
        "lzcnt" => Some(crate::encoder::encode_lzcnt(buf, ops, instr)),
        "tzcnt" => Some(crate::encoder::encode_tzcnt(buf, ops, instr)),
        "bswap" => Some(crate::encoder::encode_bswap(buf, ops, instr)),

        // ── SSE/SSE2/SSE3 xmm,xmm/m ────────────────────────────────
        "addpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x58],
            None,
            reloc,
        )),
        "addps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x58],
            None,
            reloc,
        )),
        "addsd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x58],
            None,
            reloc,
        )),
        "addss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x58],
            None,
            reloc,
        )),
        "addsubpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xD0],
            None,
            reloc,
        )),
        "addsubps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0xD0],
            None,
            reloc,
        )),
        "andnpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x55],
            None,
            reloc,
        )),
        "andnps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x55],
            None,
            reloc,
        )),
        "andpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x54],
            None,
            reloc,
        )),
        "andps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x54],
            None,
            reloc,
        )),
        "comisd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x2F],
            None,
            reloc,
        )),
        "comiss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x2F],
            None,
            reloc,
        )),
        "cvtdq2pd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0xE6],
            None,
            reloc,
        )),
        "cvtdq2ps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x5B],
            None,
            reloc,
        )),
        "cvtpd2dq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0xE6],
            None,
            reloc,
        )),
        "cvtpd2ps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x5A],
            None,
            reloc,
        )),
        "cvtps2dq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x5B],
            None,
            reloc,
        )),
        "cvtps2pd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x5A],
            None,
            reloc,
        )),
        "cvtsd2ss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x5A],
            None,
            reloc,
        )),
        "cvtss2sd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x5A],
            None,
            reloc,
        )),
        "cvttpd2dq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xE6],
            None,
            reloc,
        )),
        "cvttps2dq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x5B],
            None,
            reloc,
        )),
        "divpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x5E],
            None,
            reloc,
        )),
        "divps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x5E],
            None,
            reloc,
        )),
        "divsd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x5E],
            None,
            reloc,
        )),
        "divss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x5E],
            None,
            reloc,
        )),
        "haddpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x7C],
            None,
            reloc,
        )),
        "haddps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x7C],
            None,
            reloc,
        )),
        "hsubpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x7D],
            None,
            reloc,
        )),
        "hsubps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x7D],
            None,
            reloc,
        )),
        "lddqu" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0xF0],
            None,
            reloc,
        )),
        "maxpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x5F],
            None,
            reloc,
        )),
        "maxps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x5F],
            None,
            reloc,
        )),
        "maxsd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x5F],
            None,
            reloc,
        )),
        "maxss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x5F],
            None,
            reloc,
        )),
        "minpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x5D],
            None,
            reloc,
        )),
        "minps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x5D],
            None,
            reloc,
        )),
        "minsd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x5D],
            None,
            reloc,
        )),
        "minss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x5D],
            None,
            reloc,
        )),
        "movddup" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x12],
            None,
            reloc,
        )),
        "movhlps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x12],
            None,
            reloc,
        )),
        "movlhps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x16],
            None,
            reloc,
        )),
        "movmskpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x50],
            None,
            reloc,
        )),
        "movmskps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x50],
            None,
            reloc,
        )),
        "movshdup" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x16],
            None,
            reloc,
        )),
        "movsldup" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x12],
            None,
            reloc,
        )),
        "mulpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x59],
            None,
            reloc,
        )),
        "mulps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x59],
            None,
            reloc,
        )),
        "mulsd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x59],
            None,
            reloc,
        )),
        "mulss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x59],
            None,
            reloc,
        )),
        "orpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x56],
            None,
            reloc,
        )),
        "orps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x56],
            None,
            reloc,
        )),
        "packssdw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x6B],
            None,
            reloc,
        )),
        "packsswb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x63],
            None,
            reloc,
        )),
        "packuswb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x67],
            None,
            reloc,
        )),
        "paddb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xFC],
            None,
            reloc,
        )),
        "paddd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xFE],
            None,
            reloc,
        )),
        "paddq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xD4],
            None,
            reloc,
        )),
        "paddsb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xEC],
            None,
            reloc,
        )),
        "paddsw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xED],
            None,
            reloc,
        )),
        "paddusb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xDC],
            None,
            reloc,
        )),
        "paddusw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xDD],
            None,
            reloc,
        )),
        "paddw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xFD],
            None,
            reloc,
        )),
        "pand" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xDB],
            None,
            reloc,
        )),
        "pandn" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xDF],
            None,
            reloc,
        )),
        "pavgb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xE0],
            None,
            reloc,
        )),
        "pavgw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xE3],
            None,
            reloc,
        )),
        "pcmpeqb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x74],
            None,
            reloc,
        )),
        "pcmpeqd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x76],
            None,
            reloc,
        )),
        "pcmpeqw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x75],
            None,
            reloc,
        )),
        "pcmpgtb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x64],
            None,
            reloc,
        )),
        "pcmpgtd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x66],
            None,
            reloc,
        )),
        "pcmpgtw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x65],
            None,
            reloc,
        )),
        "pmaddwd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xF5],
            None,
            reloc,
        )),
        "pmaxsw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xEE],
            None,
            reloc,
        )),
        "pmaxub" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xDE],
            None,
            reloc,
        )),
        "pminsw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xEA],
            None,
            reloc,
        )),
        "pminub" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xDA],
            None,
            reloc,
        )),
        "pmulhuw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xE4],
            None,
            reloc,
        )),
        "pmulhw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xE5],
            None,
            reloc,
        )),
        "pmullw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xD5],
            None,
            reloc,
        )),
        "pmuludq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xF4],
            None,
            reloc,
        )),
        "por" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xEB],
            None,
            reloc,
        )),
        "psadbw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xF6],
            None,
            reloc,
        )),
        "pslld" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xF2],
            None,
            reloc,
        )),
        "psllq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xF3],
            None,
            reloc,
        )),
        "psllw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xF1],
            None,
            reloc,
        )),
        "psrad" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xE2],
            None,
            reloc,
        )),
        "psraw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xE1],
            None,
            reloc,
        )),
        "psrld" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xD2],
            None,
            reloc,
        )),
        "psrlq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xD3],
            None,
            reloc,
        )),
        "psrlw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xD1],
            None,
            reloc,
        )),
        "psubb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xF8],
            None,
            reloc,
        )),
        "psubd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xFA],
            None,
            reloc,
        )),
        "psubq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xFB],
            None,
            reloc,
        )),
        "psubsb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xE8],
            None,
            reloc,
        )),
        "psubsw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xE9],
            None,
            reloc,
        )),
        "psubusb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xD8],
            None,
            reloc,
        )),
        "psubusw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xD9],
            None,
            reloc,
        )),
        "psubw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xF9],
            None,
            reloc,
        )),
        "punpckhbw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x68],
            None,
            reloc,
        )),
        "punpckhdq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x6A],
            None,
            reloc,
        )),
        "punpckhqdq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x6D],
            None,
            reloc,
        )),
        "punpckhwd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x69],
            None,
            reloc,
        )),
        "punpcklbw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x60],
            None,
            reloc,
        )),
        "punpckldq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x62],
            None,
            reloc,
        )),
        "punpcklqdq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x6C],
            None,
            reloc,
        )),
        "punpcklwd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x61],
            None,
            reloc,
        )),
        "pxor" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xEF],
            None,
            reloc,
        )),
        "rcpps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x53],
            None,
            reloc,
        )),
        "rcpss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x53],
            None,
            reloc,
        )),
        "rsqrtps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x52],
            None,
            reloc,
        )),
        "rsqrtss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x52],
            None,
            reloc,
        )),
        "sqrtpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x51],
            None,
            reloc,
        )),
        "sqrtps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x51],
            None,
            reloc,
        )),
        "sqrtsd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x51],
            None,
            reloc,
        )),
        "sqrtss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x51],
            None,
            reloc,
        )),
        "subpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x5C],
            None,
            reloc,
        )),
        "subps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x5C],
            None,
            reloc,
        )),
        "subsd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x5C],
            None,
            reloc,
        )),
        "subss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x5C],
            None,
            reloc,
        )),
        "ucomisd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x2E],
            None,
            reloc,
        )),
        "ucomiss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x2E],
            None,
            reloc,
        )),
        "unpckhpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x15],
            None,
            reloc,
        )),
        "unpckhps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x15],
            None,
            reloc,
        )),
        "unpcklpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x14],
            None,
            reloc,
        )),
        "unpcklps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x14],
            None,
            reloc,
        )),
        "xorpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x57],
            None,
            reloc,
        )),
        "xorps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x57],
            None,
            reloc,
        )),

        // ── SSE data movement ───────────────────────────────────────
        "movapd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x28],
            Some(&[0x0F, 0x29]),
            reloc,
        )),
        "movaps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x28],
            Some(&[0x0F, 0x29]),
            reloc,
        )),
        "movdqa" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x6F],
            Some(&[0x0F, 0x7F]),
            reloc,
        )),
        "movdqu" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x6F],
            Some(&[0x0F, 0x7F]),
            reloc,
        )),
        "movhpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x16],
            Some(&[0x0F, 0x17]),
            reloc,
        )),
        "movhps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x16],
            Some(&[0x0F, 0x17]),
            reloc,
        )),
        "movlpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x12],
            Some(&[0x0F, 0x13]),
            reloc,
        )),
        "movlps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x12],
            Some(&[0x0F, 0x13]),
            reloc,
        )),
        "movsd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x10],
            Some(&[0x0F, 0x11]),
            reloc,
        )),
        "movss" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x10],
            Some(&[0x0F, 0x11]),
            reloc,
        )),
        "movupd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x10],
            Some(&[0x0F, 0x11]),
            reloc,
        )),
        "movups" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x10],
            Some(&[0x0F, 0x11]),
            reloc,
        )),

        // ── SSE with immediate ──────────────────────────────────────
        "cmppd" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xC2],
            reloc,
        )),
        "cmpps" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0xC2],
            reloc,
        )),
        "cmpsd" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0xC2],
            reloc,
        )),
        "cmpss" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0xC2],
            reloc,
        )),
        "palignr" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x0F],
            reloc,
        )),
        "pshufd" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x70],
            reloc,
        )),
        "pshufhw" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x70],
            reloc,
        )),
        "pshuflw" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x70],
            reloc,
        )),
        "shufpd" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xC6],
            reloc,
        )),
        "shufps" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0xC6],
            reloc,
        )),

        // ── SSE 0F 38 instructions (SSSE3 / SSE4.1 / SSE4.2 / AES / SHA) ──
        "aesdec" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0xDE],
            None,
            reloc,
        )),
        "aesdeclast" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0xDF],
            None,
            reloc,
        )),
        "aesenc" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0xDC],
            None,
            reloc,
        )),
        "aesenclast" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0xDD],
            None,
            reloc,
        )),
        "aesimc" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0xDB],
            None,
            reloc,
        )),
        "blendvpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x15],
            None,
            reloc,
        )),
        "blendvps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x14],
            None,
            reloc,
        )),
        "crc32b" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x38, 0xF0],
            None,
            reloc,
        )),
        "crc32w" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x38, 0xF1],
            None,
            reloc,
        )),
        "movntdqa" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x2A],
            None,
            reloc,
        )),
        "pabsb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x1C],
            None,
            reloc,
        )),
        "pabsd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x1E],
            None,
            reloc,
        )),
        "pabsw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x1D],
            None,
            reloc,
        )),
        "packusdw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x2B],
            None,
            reloc,
        )),
        "pblendvb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x10],
            None,
            reloc,
        )),
        "pcmpeqq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x29],
            None,
            reloc,
        )),
        "pcmpgtq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x37],
            None,
            reloc,
        )),
        "phaddd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x02],
            None,
            reloc,
        )),
        "phaddsw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x03],
            None,
            reloc,
        )),
        "phaddw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x01],
            None,
            reloc,
        )),
        "phminposuw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x41],
            None,
            reloc,
        )),
        "phsubd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x06],
            None,
            reloc,
        )),
        "phsubsw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x07],
            None,
            reloc,
        )),
        "phsubw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x05],
            None,
            reloc,
        )),
        "pmaddubsw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x04],
            None,
            reloc,
        )),
        "pmaxsb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x3C],
            None,
            reloc,
        )),
        "pmaxsd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x3D],
            None,
            reloc,
        )),
        "pmaxud" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x3F],
            None,
            reloc,
        )),
        "pmaxuw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x3E],
            None,
            reloc,
        )),
        "pminsb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x38],
            None,
            reloc,
        )),
        "pminsd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x39],
            None,
            reloc,
        )),
        "pminud" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x3B],
            None,
            reloc,
        )),
        "pminuw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x3A],
            None,
            reloc,
        )),
        "pmovsxbd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x21],
            None,
            reloc,
        )),
        "pmovsxbq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x22],
            None,
            reloc,
        )),
        "pmovsxbw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x20],
            None,
            reloc,
        )),
        "pmovsxdq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x25],
            None,
            reloc,
        )),
        "pmovsxwd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x23],
            None,
            reloc,
        )),
        "pmovsxwq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x24],
            None,
            reloc,
        )),
        "pmovzxbd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x31],
            None,
            reloc,
        )),
        "pmovzxbq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x32],
            None,
            reloc,
        )),
        "pmovzxbw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x30],
            None,
            reloc,
        )),
        "pmovzxdq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x35],
            None,
            reloc,
        )),
        "pmovzxwd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x33],
            None,
            reloc,
        )),
        "pmovzxwq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x34],
            None,
            reloc,
        )),
        "pmuldq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x28],
            None,
            reloc,
        )),
        "pmulhrsw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x0B],
            None,
            reloc,
        )),
        "pmulld" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x40],
            None,
            reloc,
        )),
        "pshufb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x00],
            None,
            reloc,
        )),
        "psignb" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x08],
            None,
            reloc,
        )),
        "psignd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x0A],
            None,
            reloc,
        )),
        "psignw" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x09],
            None,
            reloc,
        )),
        "ptest" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38, 0x17],
            None,
            reloc,
        )),
        "sha1msg1" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38, 0xC9],
            None,
            reloc,
        )),
        "sha1msg2" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38, 0xCA],
            None,
            reloc,
        )),
        "sha1nexte" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38, 0xC8],
            None,
            reloc,
        )),
        "sha256msg1" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38, 0xCC],
            None,
            reloc,
        )),
        "sha256msg2" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38, 0xCD],
            None,
            reloc,
        )),
        "sha256rnds2" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38, 0xCB],
            None,
            reloc,
        )),

        // ── SSE 0F 3A + imm8 instructions ───────────────────────────
        "aeskeygenassist" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0xDF],
            reloc,
        )),
        "blendpd" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x0D],
            reloc,
        )),
        "blendps" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x0C],
            reloc,
        )),
        "dppd" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x41],
            reloc,
        )),
        "dpps" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x40],
            reloc,
        )),
        "extractps" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x17],
            reloc,
        )),
        "insertps" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x21],
            reloc,
        )),
        "mpsadbw" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x42],
            reloc,
        )),
        "pblendw" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x0E],
            reloc,
        )),
        "pclmulqdq" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x44],
            reloc,
        )),
        "pcmpestri" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x61],
            reloc,
        )),
        "pcmpestrm" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x60],
            reloc,
        )),
        "pcmpistri" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x63],
            reloc,
        )),
        "pcmpistrm" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x62],
            reloc,
        )),
        "pextrb" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x14],
            reloc,
        )),
        "pextrd" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x16],
            reloc,
        )),
        "pextrq" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x16],
            reloc,
        )),
        "pextrw_imm" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x15],
            reloc,
        )),
        "pinsrb" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x20],
            reloc,
        )),
        "pinsrd" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x22],
            reloc,
        )),
        "pinsrq" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x22],
            reloc,
        )),
        "roundpd" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x09],
            reloc,
        )),
        "roundps" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x08],
            reloc,
        )),
        "roundsd" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x0B],
            reloc,
        )),
        "roundss" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A, 0x0A],
            reloc,
        )),
        "sha1rnds4" => Some(crate::encoder::encode_sse_imm(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x3A, 0xCC],
            reloc,
        )),

        // ── SSE store-only ──────────────────────────────────────────
        "movntdq" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0xE7],
            None,
            reloc,
        )),
        "movntpd" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x2B],
            None,
            reloc,
        )),
        "movntps" => Some(crate::encoder::encode_sse_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x2B],
            None,
            reloc,
        )),

        // ── Hand-coded SSE special instructions ─────────────────────
        "movd" => Some(crate::encoder::encode_movd_movq(
            buf, ops, instr, reloc, false,
        )),
        "movq" => Some(crate::encoder::encode_movd_movq(
            buf, ops, instr, reloc, true,
        )),
        "cvtsi2ss" => Some(crate::encoder::encode_cvtsi2(buf, ops, instr, 0xF3, reloc)),
        "cvtsi2sd" => Some(crate::encoder::encode_cvtsi2(buf, ops, instr, 0xF2, reloc)),
        "cvtss2si" => Some(crate::encoder::encode_cvt2si(
            buf, ops, instr, 0xF3, 0x2D, reloc,
        )),
        "cvtsd2si" => Some(crate::encoder::encode_cvt2si(
            buf, ops, instr, 0xF2, 0x2D, reloc,
        )),
        "cvttss2si" => Some(crate::encoder::encode_cvt2si(
            buf, ops, instr, 0xF3, 0x2C, reloc,
        )),
        "cvttsd2si" => Some(crate::encoder::encode_cvt2si(
            buf, ops, instr, 0xF2, 0x2C, reloc,
        )),

        // ── Cache / prefetch ────────────────────────────────────────
        "prefetchnta" => Some(crate::encoder::encode_prefetch(buf, ops, instr, 0)),
        "prefetcht0" => Some(crate::encoder::encode_prefetch(buf, ops, instr, 1)),
        "prefetcht1" => Some(crate::encoder::encode_prefetch(buf, ops, instr, 2)),
        "prefetcht2" => Some(crate::encoder::encode_prefetch(buf, ops, instr, 3)),
        "prefetchw" => Some(crate::encoder::encode_prefetchw(buf, ops, instr)),
        "clflush" => Some(crate::encoder::encode_clflush(buf, ops, instr)),
        "clflushopt" => Some(crate::encoder::encode_clflushopt(buf, ops, instr)),
        "clwb" => Some(crate::encoder::encode_clwb(buf, ops, instr)),

        // ── CRC32 ───────────────────────────────────────────────────
        "crc32" => Some(crate::encoder::encode_crc32(buf, ops, instr)),

        // ── Special instruction encoders (formerly x86_insn.rs) ─────
        "cmpxchg" => Some(encode_rm_reg(
            buf,
            ops,
            instr,
            0,
            &[0x0F, 0xB1],
            &[0x0F, 0xB0],
            reloc,
        )),
        "xadd" => Some(encode_rm_reg(
            buf,
            ops,
            instr,
            0,
            &[0x0F, 0xC1],
            &[0x0F, 0xC0],
            reloc,
        )),
        "in" => Some(encode_in(buf, ops, instr)),
        "out" => Some(encode_out(buf, ops, instr)),
        "enter" => Some(encode_enter(buf, ops, instr)),
        "jecxz" => Some(encode_jecxz(buf, ops, instr, reloc, relax_info)),
        "jrcxz" => Some(encode_jrcxz(buf, ops, instr, reloc, relax_info)),
        "rdrand" => Some(encode_rdrand_rdseed(buf, ops, instr, 6)),
        "rdseed" => Some(encode_rdrand_rdseed(buf, ops, instr, 7)),
        "adcx" => Some(encode_adcx_adox(buf, ops, instr, 0x66, reloc)),
        "adox" => Some(encode_adcx_adox(buf, ops, instr, 0xF3, reloc)),
        "cmpxchg8b" => Some(encode_cmpxchg_wide(buf, ops, instr, false, reloc)),
        "cmpxchg16b" => Some(encode_cmpxchg_wide(buf, ops, instr, true, reloc)),
        "movnti" => Some(encode_movnti(buf, ops, instr, reloc)),
        "movbe" => Some(encode_movbe(buf, ops, instr, reloc)),
        "shld" => Some(encode_shld_shrd(
            buf,
            ops,
            instr,
            &[0x0F, 0xA4],
            &[0x0F, 0xA5],
            reloc,
        )),
        "shrd" => Some(encode_shld_shrd(
            buf,
            ops,
            instr,
            &[0x0F, 0xAC],
            &[0x0F, 0xAD],
            reloc,
        )),

        // ── FS/GS base manipulation ─────────────────────────────────
        "rdfsbase" => Some(encode_fsgsbase(buf, ops, instr, 0)),
        "rdgsbase" => Some(encode_fsgsbase(buf, ops, instr, 1)),
        "wrfsbase" => Some(encode_fsgsbase(buf, ops, instr, 2)),
        "wrgsbase" => Some(encode_fsgsbase(buf, ops, instr, 3)),

        // ── Extended state save/restore ─────────────────────────────
        "fxsave" => Some(encode_0f_ae_mem(buf, ops, instr, 0, false, reloc)),
        "fxsave64" => Some(encode_0f_ae_mem(buf, ops, instr, 0, true, reloc)),
        "fxrstor" => Some(encode_0f_ae_mem(buf, ops, instr, 1, false, reloc)),
        "fxrstor64" => Some(encode_0f_ae_mem(buf, ops, instr, 1, true, reloc)),
        "xsave" => Some(encode_0f_ae_mem(buf, ops, instr, 4, false, reloc)),
        "xsave64" => Some(encode_0f_ae_mem(buf, ops, instr, 4, true, reloc)),
        "xrstor" => Some(encode_0f_ae_mem(buf, ops, instr, 5, false, reloc)),
        "xrstor64" => Some(encode_0f_ae_mem(buf, ops, instr, 5, true, reloc)),
        "xsaveopt" => Some(encode_0f_ae_mem(buf, ops, instr, 6, false, reloc)),
        "xsaveopt64" => Some(encode_0f_ae_mem(buf, ops, instr, 6, true, reloc)),
        "xsavec" => Some(encode_0f_c7_mem(buf, ops, instr, 4, false, reloc)),
        "xsavec64" => Some(encode_0f_c7_mem(buf, ops, instr, 4, true, reloc)),
        "xsaves" => Some(encode_0f_c7_mem(buf, ops, instr, 5, false, reloc)),
        "xsaves64" => Some(encode_0f_c7_mem(buf, ops, instr, 5, true, reloc)),
        "xrstors" => Some(encode_0f_c7_mem(buf, ops, instr, 3, false, reloc)),
        "xrstors64" => Some(encode_0f_c7_mem(buf, ops, instr, 3, true, reloc)),

        // ── TSX (Transactional Synchronization Extensions) ──────────
        "xabort" => Some(encode_xabort(buf, ops, instr)),
        "xbegin" => Some(encode_xbegin(buf, ops, instr, reloc)),

        // ─── AVX-512 (EVEX-encoded) instructions ────────────────────────────
        //
        // EVEX map 1 = 0F, map 2 = 0F38, map 3 = 0F3A
        // Guarded arms (has_zmm) MUST precede unconditional VEX arms below.

        // AVX-512F arithmetic (packed single/double)
        "vaddps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x58, None, false, reloc,
        )),
        "vaddpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x58, None, true, reloc,
        )),
        "vsubps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x5C, None, false, reloc,
        )),
        "vsubpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x5C, None, true, reloc,
        )),
        "vmulps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x59, None, false, reloc,
        )),
        "vmulpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x59, None, true, reloc,
        )),
        "vdivps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x5E, None, false, reloc,
        )),
        "vdivpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x5E, None, true, reloc,
        )),
        "vmaxps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x5F, None, false, reloc,
        )),
        "vmaxpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x5F, None, true, reloc,
        )),
        "vminps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x5D, None, false, reloc,
        )),
        "vminpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x5D, None, true, reloc,
        )),
        "vsqrtps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x51, None, false, reloc,
        )),
        "vsqrtpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x51, None, true, reloc,
        )),

        // AVX-512F logical
        "vandps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x54, None, false, reloc,
        )),
        "vandpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x54, None, true, reloc,
        )),
        "vandnps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x55, None, false, reloc,
        )),
        "vandnpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x55, None, true, reloc,
        )),
        "vorps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x56, None, false, reloc,
        )),
        "vorpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x56, None, true, reloc,
        )),
        "vxorps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x57, None, false, reloc,
        )),
        "vxorpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x57, None, true, reloc,
        )),

        // AVX-512F data movement
        "vmovaps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf,
            ops,
            instr,
            0x00,
            1,
            0x28,
            Some(0x29),
            false,
            reloc,
        )),
        "vmovapd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf,
            ops,
            instr,
            0x66,
            1,
            0x28,
            Some(0x29),
            true,
            reloc,
        )),
        "vmovups" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf,
            ops,
            instr,
            0x00,
            1,
            0x10,
            Some(0x11),
            false,
            reloc,
        )),
        "vmovupd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf,
            ops,
            instr,
            0x66,
            1,
            0x10,
            Some(0x11),
            true,
            reloc,
        )),
        "vmovdqa32" => Some(crate::encoder::encode_evex_op(
            buf,
            ops,
            instr,
            0x66,
            1,
            0x6F,
            Some(0x7F),
            false,
            reloc,
        )),
        "vmovdqa64" => Some(crate::encoder::encode_evex_op(
            buf,
            ops,
            instr,
            0x66,
            1,
            0x6F,
            Some(0x7F),
            true,
            reloc,
        )),
        "vmovdqu32" => Some(crate::encoder::encode_evex_op(
            buf,
            ops,
            instr,
            0xF3,
            1,
            0x6F,
            Some(0x7F),
            false,
            reloc,
        )),
        "vmovdqu64" => Some(crate::encoder::encode_evex_op(
            buf,
            ops,
            instr,
            0xF3,
            1,
            0x6F,
            Some(0x7F),
            true,
            reloc,
        )),
        "vmovdqu8" => Some(crate::encoder::encode_evex_op(
            buf,
            ops,
            instr,
            0xF2,
            1,
            0x6F,
            Some(0x7F),
            false,
            reloc,
        )),
        "vmovdqu16" => Some(crate::encoder::encode_evex_op(
            buf,
            ops,
            instr,
            0xF2,
            1,
            0x6F,
            Some(0x7F),
            true,
            reloc,
        )),

        // AVX-512F unpack
        "vunpckhps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x15, None, false, reloc,
        )),
        "vunpckhpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x15, None, true, reloc,
        )),
        "vunpcklps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x14, None, false, reloc,
        )),
        "vunpcklpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x14, None, true, reloc,
        )),

        // AVX-512F FMA (512-bit)
        "vfmadd132ps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x98, None, false, reloc,
        )),
        "vfmadd213ps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0xA8, None, false, reloc,
        )),
        "vfmadd231ps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0xB8, None, false, reloc,
        )),
        "vfmadd132pd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x98, None, true, reloc,
        )),
        "vfmadd213pd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0xA8, None, true, reloc,
        )),
        "vfmadd231pd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0xB8, None, true, reloc,
        )),

        // AVX-512F integer packed arithmetic
        "vpaddd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xFE, None, false, reloc,
        )),
        "vpaddq" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xD4, None, true, reloc,
        )),
        "vpsubd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xFA, None, false, reloc,
        )),
        "vpsubq" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xFB, None, true, reloc,
        )),
        "vpmulld" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x40, None, false, reloc,
        )),
        "vpmullq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x40, None, true, reloc,
        )),
        "vpandd" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xDB, None, false, reloc,
        )),
        "vpandq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xDB, None, true, reloc,
        )),
        "vpandnd" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xDF, None, false, reloc,
        )),
        "vpandnq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xDF, None, true, reloc,
        )),
        "vpord" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xEB, None, false, reloc,
        )),
        "vporq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xEB, None, true, reloc,
        )),
        "vpxord" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xEF, None, false, reloc,
        )),
        "vpxorq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xEF, None, true, reloc,
        )),

        // AVX-512F compare
        "vpcmpeqd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x76, None, false, reloc,
        )),
        "vpcmpgtd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x66, None, false, reloc,
        )),

        // AVX-512F permute/shuffle
        "vpermps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x16, None, false, reloc,
        )),
        "vpermpd"
            if has_zmm(ops)
                && ops.len() == 3
                && matches!(ops.last(), Some(Operand::Register(_))) =>
        {
            Some(crate::encoder::encode_evex_op(
                buf, ops, instr, 0x66, 2, 0x16, None, true, reloc,
            ))
        }
        "vpermd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x36, None, false, reloc,
        )),
        "vpermq"
            if has_zmm(ops)
                && ops.len() == 3
                && matches!(ops.last(), Some(Operand::Register(_))) =>
        {
            Some(crate::encoder::encode_evex_op(
                buf, ops, instr, 0x66, 2, 0x36, None, true, reloc,
            ))
        }
        "vpermq"
            if has_zmm(ops)
                && ops.len() >= 3
                && matches!(ops.last(), Some(Operand::Immediate(_))) =>
        {
            Some(crate::encoder::encode_evex_imm(
                buf, ops, instr, 0x66, 3, 0x00, true, reloc,
            ))
        }
        "vshufps" if has_zmm(ops) => Some(crate::encoder::encode_evex_imm(
            buf, ops, instr, 0x00, 1, 0xC6, false, reloc,
        )),
        "vshufpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_imm(
            buf, ops, instr, 0x66, 1, 0xC6, true, reloc,
        )),
        "vpshufd" if has_zmm(ops) => Some(crate::encoder::encode_evex_imm(
            buf, ops, instr, 0x66, 1, 0x70, false, reloc,
        )),

        // AVX-512F broadcast loads
        "vbroadcastss" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x18, None, false, reloc,
        )),
        "vbroadcastsd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x19, None, true, reloc,
        )),
        "vpbroadcastd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x58, None, false, reloc,
        )),
        "vpbroadcastq" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x59, None, true, reloc,
        )),
        "vpbroadcastb" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x78, None, false, reloc,
        )),
        "vpbroadcastw" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x79, None, false, reloc,
        )),

        // AVX-512F conversions
        "vcvtps2pd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x5A, None, false, reloc,
        )),
        "vcvtpd2ps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x5A, None, true, reloc,
        )),
        "vcvtdq2ps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x5B, None, false, reloc,
        )),
        "vcvtps2dq" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x5B, None, false, reloc,
        )),
        "vcvttps2dq" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0xF3, 1, 0x5B, None, false, reloc,
        )),
        "vcvtdq2pd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0xF3, 1, 0xE6, None, false, reloc,
        )),
        "vcvtpd2dq" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0xF2, 1, 0xE6, None, true, reloc,
        )),
        "vcvttpd2dq" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xE6, None, true, reloc,
        )),

        // AVX-512F ternary logic & blend
        "vpternlogd" => Some(crate::encoder::encode_evex_imm(
            buf, ops, instr, 0x66, 3, 0x25, false, reloc,
        )),
        "vpternlogq" => Some(crate::encoder::encode_evex_imm(
            buf, ops, instr, 0x66, 3, 0x25, true, reloc,
        )),
        "vblendmps" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x65, None, false, reloc,
        )),
        "vblendmpd" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x65, None, true, reloc,
        )),
        "vpblendmd" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x64, None, false, reloc,
        )),
        "vpblendmq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x64, None, true, reloc,
        )),

        // AVX-512F shifts
        "vpslld"
            if has_zmm(ops)
                && ops.len() >= 3
                && matches!(ops.last(), Some(Operand::Register(_))) =>
        {
            Some(crate::encoder::encode_evex_op(
                buf, ops, instr, 0x66, 1, 0xF2, None, false, reloc,
            ))
        }
        "vpsllq"
            if has_zmm(ops)
                && ops.len() >= 3
                && matches!(ops.last(), Some(Operand::Register(_))) =>
        {
            Some(crate::encoder::encode_evex_op(
                buf, ops, instr, 0x66, 1, 0xF3, None, true, reloc,
            ))
        }
        "vpsrld"
            if has_zmm(ops)
                && ops.len() >= 3
                && matches!(ops.last(), Some(Operand::Register(_))) =>
        {
            Some(crate::encoder::encode_evex_op(
                buf, ops, instr, 0x66, 1, 0xD2, None, false, reloc,
            ))
        }
        "vpsrlq"
            if has_zmm(ops)
                && ops.len() >= 3
                && matches!(ops.last(), Some(Operand::Register(_))) =>
        {
            Some(crate::encoder::encode_evex_op(
                buf, ops, instr, 0x66, 1, 0xD3, None, true, reloc,
            ))
        }

        // AVX-512F variable shifts
        "vpsllvd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x47, None, false, reloc,
        )),
        "vpsllvq" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x47, None, true, reloc,
        )),
        "vpsrlvd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x45, None, false, reloc,
        )),
        "vpsrlvq" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x45, None, true, reloc,
        )),
        "vpsravd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x46, None, false, reloc,
        )),
        "vpsravq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x46, None, true, reloc,
        )),

        // AVX-512F compress/expand
        "vcompresspd" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x8A, None, true, reloc,
        )),
        "vcompressps" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x8A, None, false, reloc,
        )),
        "vexpandpd" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x88, None, true, reloc,
        )),
        "vexpandps" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x88, None, false, reloc,
        )),
        "vpcompressd" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x8B, None, false, reloc,
        )),
        "vpcompressq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x8B, None, true, reloc,
        )),
        "vpexpandd" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x89, None, false, reloc,
        )),
        "vpexpandq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x89, None, true, reloc,
        )),

        // AVX-512F gather (basic register-only forms)
        "vgatherdps" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x92, None, false, reloc,
        )),
        "vgatherdpd" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 2, 0x92, None, true, reloc,
        )),

        // AVX-512BW byte/word compare
        "vpcmpeqb" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x74, None, false, reloc,
        )),
        "vpcmpeqw" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x75, None, false, reloc,
        )),
        "vpcmpgtb" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x64, None, false, reloc,
        )),
        "vpcmpgtw" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x65, None, false, reloc,
        )),

        // AVX-512BW byte/word add/sub
        "vpaddb" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xFC, None, false, reloc,
        )),
        "vpaddw" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xFD, None, false, reloc,
        )),
        "vpsubb" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xF8, None, false, reloc,
        )),
        "vpsubw" if has_zmm(ops) => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0xF9, None, false, reloc,
        )),

        // AVX-512DQ specific conversions
        "vcvtqq2ps" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x5B, None, true, reloc,
        )),
        "vcvtqq2pd" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0xF3, 1, 0xE6, None, true, reloc,
        )),
        "vcvtps2qq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x7B, None, false, reloc,
        )),
        "vcvtpd2qq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x66, 1, 0x7B, None, true, reloc,
        )),
        "vcvtps2udq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x79, None, false, reloc,
        )),
        "vcvtpd2udq" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0x00, 1, 0x79, None, true, reloc,
        )),
        "vcvtudq2ps" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0xF2, 1, 0x7A, None, false, reloc,
        )),
        "vcvtudq2pd" => Some(crate::encoder::encode_evex_op(
            buf, ops, instr, 0xF3, 1, 0x7A, None, false, reloc,
        )),

        // ── AVX (VEX-encoded) arithmetic ────────────────────────────
        "vaddpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x58,
            None,
            false,
            reloc,
        )),
        "vaddps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x58,
            None,
            false,
            reloc,
        )),
        "vaddsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x58,
            None,
            false,
            reloc,
        )),
        "vaddss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x58,
            None,
            false,
            reloc,
        )),
        "vaddsubpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xD0,
            None,
            false,
            reloc,
        )),
        "vaddsubps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0xD0,
            None,
            false,
            reloc,
        )),
        "vdivpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x5E,
            None,
            false,
            reloc,
        )),
        "vdivps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x5E,
            None,
            false,
            reloc,
        )),
        "vdivsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x5E,
            None,
            false,
            reloc,
        )),
        "vdivss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x5E,
            None,
            false,
            reloc,
        )),
        "vhaddpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x7C,
            None,
            false,
            reloc,
        )),
        "vhaddps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x7C,
            None,
            false,
            reloc,
        )),
        "vhsubpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x7D,
            None,
            false,
            reloc,
        )),
        "vhsubps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x7D,
            None,
            false,
            reloc,
        )),
        "vmaxpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x5F,
            None,
            false,
            reloc,
        )),
        "vmaxps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x5F,
            None,
            false,
            reloc,
        )),
        "vmaxsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x5F,
            None,
            false,
            reloc,
        )),
        "vmaxss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x5F,
            None,
            false,
            reloc,
        )),
        "vminpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x5D,
            None,
            false,
            reloc,
        )),
        "vminps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x5D,
            None,
            false,
            reloc,
        )),
        "vminsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x5D,
            None,
            false,
            reloc,
        )),
        "vminss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x5D,
            None,
            false,
            reloc,
        )),
        "vmulpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x59,
            None,
            false,
            reloc,
        )),
        "vmulps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x59,
            None,
            false,
            reloc,
        )),
        "vmulsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x59,
            None,
            false,
            reloc,
        )),
        "vmulss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x59,
            None,
            false,
            reloc,
        )),
        "vsubpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x5C,
            None,
            false,
            reloc,
        )),
        "vsubps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x5C,
            None,
            false,
            reloc,
        )),
        "vsubsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x5C,
            None,
            false,
            reloc,
        )),
        "vsubss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x5C,
            None,
            false,
            reloc,
        )),
        "vsqrtpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x51,
            None,
            false,
            reloc,
        )),
        "vsqrtps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x51,
            None,
            false,
            reloc,
        )),
        "vsqrtsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x51,
            None,
            false,
            reloc,
        )),
        "vsqrtss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x51,
            None,
            false,
            reloc,
        )),
        "vrcpps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x53,
            None,
            false,
            reloc,
        )),
        "vrcpss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x53,
            None,
            false,
            reloc,
        )),
        "vrsqrtps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x52,
            None,
            false,
            reloc,
        )),
        "vrsqrtss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x52,
            None,
            false,
            reloc,
        )),

        // ── AVX logical ─────────────────────────────────────────────
        "vandpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x54,
            None,
            false,
            reloc,
        )),
        "vandps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x54,
            None,
            false,
            reloc,
        )),
        "vandnpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x55,
            None,
            false,
            reloc,
        )),
        "vandnps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x55,
            None,
            false,
            reloc,
        )),
        "vorpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x56,
            None,
            false,
            reloc,
        )),
        "vorps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x56,
            None,
            false,
            reloc,
        )),
        "vxorpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x57,
            None,
            false,
            reloc,
        )),
        "vxorps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x57,
            None,
            false,
            reloc,
        )),

        // ── AVX compare ─────────────────────────────────────────────
        "vcomisd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x2F,
            None,
            false,
            reloc,
        )),
        "vcomiss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x2F,
            None,
            false,
            reloc,
        )),
        "vucomisd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x2E,
            None,
            false,
            reloc,
        )),
        "vucomiss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x2E,
            None,
            false,
            reloc,
        )),

        // ── AVX data movement ───────────────────────────────────────
        "vmovapd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x28,
            Some(0x29),
            false,
            reloc,
        )),
        "vmovaps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x28,
            Some(0x29),
            false,
            reloc,
        )),
        "vmovdqa" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x6F,
            Some(0x7F),
            false,
            reloc,
        )),
        "vmovdqu" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x6F,
            Some(0x7F),
            false,
            reloc,
        )),
        "vmovsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x10,
            Some(0x11),
            false,
            reloc,
        )),
        "vmovss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x10,
            Some(0x11),
            false,
            reloc,
        )),
        "vmovupd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x10,
            Some(0x11),
            false,
            reloc,
        )),
        "vmovups" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x10,
            Some(0x11),
            false,
            reloc,
        )),
        "vmovhpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x16,
            Some(0x17),
            false,
            reloc,
        )),
        "vmovhps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x16,
            Some(0x17),
            false,
            reloc,
        )),
        "vmovlpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x12,
            Some(0x13),
            false,
            reloc,
        )),
        "vmovlps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x12,
            Some(0x13),
            false,
            reloc,
        )),
        "vlddqu" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0xF0,
            None,
            false,
            reloc,
        )),
        "vmovntdq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xE7,
            None,
            false,
            reloc,
        )),
        "vmovntpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x2B,
            None,
            false,
            reloc,
        )),
        "vmovntps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x2B,
            None,
            false,
            reloc,
        )),
        "vmovddup" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x12,
            None,
            false,
            reloc,
        )),
        "vmovshdup" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x16,
            None,
            false,
            reloc,
        )),
        "vmovsldup" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x12,
            None,
            false,
            reloc,
        )),

        // ── AVX unpack ──────────────────────────────────────────────
        "vunpckhpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x15,
            None,
            false,
            reloc,
        )),
        "vunpckhps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x15,
            None,
            false,
            reloc,
        )),
        "vunpcklpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x14,
            None,
            false,
            reloc,
        )),
        "vunpcklps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x14,
            None,
            false,
            reloc,
        )),

        // ── AVX integer (packed) ────────────────────────────────────
        "vpaddb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xFC,
            None,
            false,
            reloc,
        )),
        "vpaddw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xFD,
            None,
            false,
            reloc,
        )),
        "vpaddd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xFE,
            None,
            false,
            reloc,
        )),
        "vpaddq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xD4,
            None,
            false,
            reloc,
        )),
        "vpaddsb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xEC,
            None,
            false,
            reloc,
        )),
        "vpaddsw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xED,
            None,
            false,
            reloc,
        )),
        "vpaddusb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xDC,
            None,
            false,
            reloc,
        )),
        "vpaddusw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xDD,
            None,
            false,
            reloc,
        )),
        "vpsubb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xF8,
            None,
            false,
            reloc,
        )),
        "vpsubw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xF9,
            None,
            false,
            reloc,
        )),
        "vpsubd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xFA,
            None,
            false,
            reloc,
        )),
        "vpsubq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xFB,
            None,
            false,
            reloc,
        )),
        "vpsubsb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xE8,
            None,
            false,
            reloc,
        )),
        "vpsubsw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xE9,
            None,
            false,
            reloc,
        )),
        "vpsubusb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xD8,
            None,
            false,
            reloc,
        )),
        "vpsubusw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xD9,
            None,
            false,
            reloc,
        )),
        "vpand" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xDB,
            None,
            false,
            reloc,
        )),
        "vpandn" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xDF,
            None,
            false,
            reloc,
        )),
        "vpor" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xEB,
            None,
            false,
            reloc,
        )),
        "vpxor" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xEF,
            None,
            false,
            reloc,
        )),
        "vpcmpeqb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x74,
            None,
            false,
            reloc,
        )),
        "vpcmpeqw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x75,
            None,
            false,
            reloc,
        )),
        "vpcmpeqd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x76,
            None,
            false,
            reloc,
        )),
        "vpcmpgtb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x64,
            None,
            false,
            reloc,
        )),
        "vpcmpgtw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x65,
            None,
            false,
            reloc,
        )),
        "vpcmpgtd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x66,
            None,
            false,
            reloc,
        )),
        "vpmullw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xD5,
            None,
            false,
            reloc,
        )),
        "vpmulhw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xE5,
            None,
            false,
            reloc,
        )),
        "vpmulhuw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xE4,
            None,
            false,
            reloc,
        )),
        "vpmuludq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xF4,
            None,
            false,
            reloc,
        )),
        "vpmaddwd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xF5,
            None,
            false,
            reloc,
        )),
        "vpsadbw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xF6,
            None,
            false,
            reloc,
        )),
        "vpavgb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xE0,
            None,
            false,
            reloc,
        )),
        "vpavgw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xE3,
            None,
            false,
            reloc,
        )),
        "vpmaxsw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xEE,
            None,
            false,
            reloc,
        )),
        "vpmaxub" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xDE,
            None,
            false,
            reloc,
        )),
        "vpminsw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xEA,
            None,
            false,
            reloc,
        )),
        "vpminub" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xDA,
            None,
            false,
            reloc,
        )),
        "vpacksswb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x63,
            None,
            false,
            reloc,
        )),
        "vpackssdw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x6B,
            None,
            false,
            reloc,
        )),
        "vpackuswb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x67,
            None,
            false,
            reloc,
        )),
        "vpunpckhbw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x68,
            None,
            false,
            reloc,
        )),
        "vpunpckhwd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x69,
            None,
            false,
            reloc,
        )),
        "vpunpckhdq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x6A,
            None,
            false,
            reloc,
        )),
        "vpunpckhqdq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x6D,
            None,
            false,
            reloc,
        )),
        "vpunpcklbw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x60,
            None,
            false,
            reloc,
        )),
        "vpunpcklwd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x61,
            None,
            false,
            reloc,
        )),
        "vpunpckldq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x62,
            None,
            false,
            reloc,
        )),
        "vpunpcklqdq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x6C,
            None,
            false,
            reloc,
        )),

        // ── AVX 0F 38 (SSSE3/SSE4 VEX) ─────────────────────────────
        "vaesenc" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xDC,
            None,
            false,
            reloc,
        )),
        "vaesenclast" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xDD,
            None,
            false,
            reloc,
        )),
        "vaesdec" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xDE,
            None,
            false,
            reloc,
        )),
        "vaesdeclast" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xDF,
            None,
            false,
            reloc,
        )),
        "vaesimc" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xDB,
            None,
            false,
            reloc,
        )),
        "vpshufb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x00,
            None,
            false,
            reloc,
        )),
        "vphaddw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x01,
            None,
            false,
            reloc,
        )),
        "vphaddd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x02,
            None,
            false,
            reloc,
        )),
        "vphaddsw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x03,
            None,
            false,
            reloc,
        )),
        "vpmaddubsw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x04,
            None,
            false,
            reloc,
        )),
        "vphsubw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x05,
            None,
            false,
            reloc,
        )),
        "vphsubd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x06,
            None,
            false,
            reloc,
        )),
        "vphsubsw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x07,
            None,
            false,
            reloc,
        )),
        "vpsignb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x08,
            None,
            false,
            reloc,
        )),
        "vpsignw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x09,
            None,
            false,
            reloc,
        )),
        "vpsignd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x0A,
            None,
            false,
            reloc,
        )),
        "vpmulhrsw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x0B,
            None,
            false,
            reloc,
        )),
        "vptest" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x17,
            None,
            false,
            reloc,
        )),
        "vpabsb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x1C,
            None,
            false,
            reloc,
        )),
        "vpabsw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x1D,
            None,
            false,
            reloc,
        )),
        "vpabsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x1E,
            None,
            false,
            reloc,
        )),
        "vpmovsxbw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x20,
            None,
            false,
            reloc,
        )),
        "vpmovsxbd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x21,
            None,
            false,
            reloc,
        )),
        "vpmovsxbq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x22,
            None,
            false,
            reloc,
        )),
        "vpmovsxwd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x23,
            None,
            false,
            reloc,
        )),
        "vpmovsxwq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x24,
            None,
            false,
            reloc,
        )),
        "vpmovsxdq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x25,
            None,
            false,
            reloc,
        )),
        "vpmuldq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x28,
            None,
            false,
            reloc,
        )),
        "vpcmpeqq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x29,
            None,
            false,
            reloc,
        )),
        "vmovntdqa" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x2A,
            None,
            false,
            reloc,
        )),
        "vpackusdw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x2B,
            None,
            false,
            reloc,
        )),
        "vpmovzxbw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x30,
            None,
            false,
            reloc,
        )),
        "vpmovzxbd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x31,
            None,
            false,
            reloc,
        )),
        "vpmovzxbq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x32,
            None,
            false,
            reloc,
        )),
        "vpmovzxwd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x33,
            None,
            false,
            reloc,
        )),
        "vpmovzxwq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x34,
            None,
            false,
            reloc,
        )),
        "vpmovzxdq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x35,
            None,
            false,
            reloc,
        )),
        "vpcmpgtq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x37,
            None,
            false,
            reloc,
        )),
        "vpminsb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x38,
            None,
            false,
            reloc,
        )),
        "vpminsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x39,
            None,
            false,
            reloc,
        )),
        "vpminuw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x3A,
            None,
            false,
            reloc,
        )),
        "vpminud" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x3B,
            None,
            false,
            reloc,
        )),
        "vpmaxsb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x3C,
            None,
            false,
            reloc,
        )),
        "vpmaxsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x3D,
            None,
            false,
            reloc,
        )),
        "vpmaxuw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x3E,
            None,
            false,
            reloc,
        )),
        "vpmaxud" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x3F,
            None,
            false,
            reloc,
        )),
        "vpmulld" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x40,
            None,
            false,
            reloc,
        )),
        "vphminposuw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x41,
            None,
            false,
            reloc,
        )),

        // ── AVX with immediate (0F / 0F3A) ─────────────────────────
        "vcmppd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xC2,
            false,
            reloc,
        )),
        "vcmpps" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0xC2,
            false,
            reloc,
        )),
        "vcmpsd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0xC2,
            false,
            reloc,
        )),
        "vcmpss" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0xC2,
            false,
            reloc,
        )),
        "vshufpd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xC6,
            false,
            reloc,
        )),
        "vshufps" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0xC6,
            false,
            reloc,
        )),
        "vpshufd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x70,
            false,
            reloc,
        )),
        "vpshufhw" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x70,
            false,
            reloc,
        )),
        "vpshuflw" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x70,
            false,
            reloc,
        )),
        "vroundpd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x09,
            false,
            reloc,
        )),
        "vroundps" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x08,
            false,
            reloc,
        )),
        "vroundsd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x0B,
            false,
            reloc,
        )),
        "vroundss" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x0A,
            false,
            reloc,
        )),
        "vblendpd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x0D,
            false,
            reloc,
        )),
        "vblendps" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x0C,
            false,
            reloc,
        )),
        "vpblendw" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x0E,
            false,
            reloc,
        )),
        "vdppd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x41,
            false,
            reloc,
        )),
        "vdpps" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x40,
            false,
            reloc,
        )),
        "vmpsadbw" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x42,
            false,
            reloc,
        )),
        "vpalignr" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x0F,
            false,
            reloc,
        )),
        "vpclmulqdq" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x44,
            false,
            reloc,
        )),
        "vaeskeygenassist" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0xDF,
            false,
            reloc,
        )),

        // ── BMI1 (VEX-encoded bit manipulation) ─────────────────────
        "andn" => Some(crate::encoder::encode_vex_bmi_vex_ndd(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38],
            0xF2,
            true,
        )),
        "bextr" => Some(crate::encoder::encode_vex_bmi_rmv(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38],
            0xF7,
            true,
        )),
        "blsi" => Some(crate::encoder::encode_vex_bmi_digit(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38],
            0xF3,
            3,
            true,
        )),
        "blsmsk" => Some(crate::encoder::encode_vex_bmi_digit(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38],
            0xF3,
            2,
            true,
        )),
        "blsr" => Some(crate::encoder::encode_vex_bmi_digit(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38],
            0xF3,
            1,
            true,
        )),

        // ── BMI2 (VEX-encoded) ──────────────────────────────────────
        "bzhi" => Some(crate::encoder::encode_vex_bmi_rmv(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F, 0x38],
            0xF5,
            true,
        )),
        "mulx" => Some(crate::encoder::encode_vex_bmi_vex_ndd(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x38],
            0xF6,
            true,
        )),
        "pdep" => Some(crate::encoder::encode_vex_bmi_vex_ndd(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x38],
            0xF5,
            true,
        )),
        "pext" => Some(crate::encoder::encode_vex_bmi_vex_ndd(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x38],
            0xF5,
            true,
        )),
        "rorx" => Some(crate::encoder::encode_vex_bmi_imm(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x3A],
            0xF0,
            true,
        )),
        "sarx" => Some(crate::encoder::encode_vex_bmi_rmv(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F, 0x38],
            0xF7,
            true,
        )),
        "shlx" => Some(crate::encoder::encode_vex_bmi_rmv(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xF7,
            true,
        )),
        "shrx" => Some(crate::encoder::encode_vex_bmi_rmv(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F, 0x38],
            0xF7,
            true,
        )),

        // ── FMA3 (Fused Multiply-Add) ───────────────────────────────
        //
        // All FMA3 instructions use VEX.66.0F38 map.
        // W=0 → single-precision (PS/SS), W=1 → double-precision (PD/SD).
        //
        // VFMADD: dst = (src1 * src2) + src3  (operand order varies by 132/213/231)
        "vfmadd132ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x98,
            None,
            false,
            reloc,
        )),
        "vfmadd132pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x98,
            None,
            true,
            reloc,
        )),
        "vfmadd213ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xA8,
            None,
            false,
            reloc,
        )),
        "vfmadd213pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xA8,
            None,
            true,
            reloc,
        )),
        "vfmadd231ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xB8,
            None,
            false,
            reloc,
        )),
        "vfmadd231pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xB8,
            None,
            true,
            reloc,
        )),
        "vfmadd132ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x99,
            None,
            false,
            reloc,
        )),
        "vfmadd132sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x99,
            None,
            true,
            reloc,
        )),
        "vfmadd213ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xA9,
            None,
            false,
            reloc,
        )),
        "vfmadd213sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xA9,
            None,
            true,
            reloc,
        )),
        "vfmadd231ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xB9,
            None,
            false,
            reloc,
        )),
        "vfmadd231sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xB9,
            None,
            true,
            reloc,
        )),
        // VFMSUB: dst = (src1 * src2) - src3
        "vfmsub132ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9A,
            None,
            false,
            reloc,
        )),
        "vfmsub132pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9A,
            None,
            true,
            reloc,
        )),
        "vfmsub213ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAA,
            None,
            false,
            reloc,
        )),
        "vfmsub213pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAA,
            None,
            true,
            reloc,
        )),
        "vfmsub231ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBA,
            None,
            false,
            reloc,
        )),
        "vfmsub231pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBA,
            None,
            true,
            reloc,
        )),
        "vfmsub132ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9B,
            None,
            false,
            reloc,
        )),
        "vfmsub132sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9B,
            None,
            true,
            reloc,
        )),
        "vfmsub213ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAB,
            None,
            false,
            reloc,
        )),
        "vfmsub213sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAB,
            None,
            true,
            reloc,
        )),
        "vfmsub231ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBB,
            None,
            false,
            reloc,
        )),
        "vfmsub231sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBB,
            None,
            true,
            reloc,
        )),
        // VFNMADD: dst = -(src1 * src2) + src3
        "vfnmadd132ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9C,
            None,
            false,
            reloc,
        )),
        "vfnmadd132pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9C,
            None,
            true,
            reloc,
        )),
        "vfnmadd213ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAC,
            None,
            false,
            reloc,
        )),
        "vfnmadd213pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAC,
            None,
            true,
            reloc,
        )),
        "vfnmadd231ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBC,
            None,
            false,
            reloc,
        )),
        "vfnmadd231pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBC,
            None,
            true,
            reloc,
        )),
        "vfnmadd132ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9D,
            None,
            false,
            reloc,
        )),
        "vfnmadd132sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9D,
            None,
            true,
            reloc,
        )),
        "vfnmadd213ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAD,
            None,
            false,
            reloc,
        )),
        "vfnmadd213sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAD,
            None,
            true,
            reloc,
        )),
        "vfnmadd231ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBD,
            None,
            false,
            reloc,
        )),
        "vfnmadd231sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBD,
            None,
            true,
            reloc,
        )),
        // VFNMSUB: dst = -(src1 * src2) - src3
        "vfnmsub132ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9E,
            None,
            false,
            reloc,
        )),
        "vfnmsub132pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9E,
            None,
            true,
            reloc,
        )),
        "vfnmsub213ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAE,
            None,
            false,
            reloc,
        )),
        "vfnmsub213pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAE,
            None,
            true,
            reloc,
        )),
        "vfnmsub231ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBE,
            None,
            false,
            reloc,
        )),
        "vfnmsub231pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBE,
            None,
            true,
            reloc,
        )),
        "vfnmsub132ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9F,
            None,
            false,
            reloc,
        )),
        "vfnmsub132sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x9F,
            None,
            true,
            reloc,
        )),
        "vfnmsub213ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAF,
            None,
            false,
            reloc,
        )),
        "vfnmsub213sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xAF,
            None,
            true,
            reloc,
        )),
        "vfnmsub231ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBF,
            None,
            false,
            reloc,
        )),
        "vfnmsub231sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xBF,
            None,
            true,
            reloc,
        )),
        // VFMADDSUB / VFMSUBADD (packed only, no scalar)
        "vfmaddsub132ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x96,
            None,
            false,
            reloc,
        )),
        "vfmaddsub132pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x96,
            None,
            true,
            reloc,
        )),
        "vfmaddsub213ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xA6,
            None,
            false,
            reloc,
        )),
        "vfmaddsub213pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xA6,
            None,
            true,
            reloc,
        )),
        "vfmaddsub231ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xB6,
            None,
            false,
            reloc,
        )),
        "vfmaddsub231pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xB6,
            None,
            true,
            reloc,
        )),
        "vfmsubadd132ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x97,
            None,
            false,
            reloc,
        )),
        "vfmsubadd132pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x97,
            None,
            true,
            reloc,
        )),
        "vfmsubadd213ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xA7,
            None,
            false,
            reloc,
        )),
        "vfmsubadd213pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xA7,
            None,
            true,
            reloc,
        )),
        "vfmsubadd231ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xB7,
            None,
            false,
            reloc,
        )),
        "vfmsubadd231pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0xB7,
            None,
            true,
            reloc,
        )),

        // ── AVX/AVX2 packed shifts ──────────────────────────────────
        //                                                       reg_opcode  imm_opcode  /digit
        "vpsllw" => Some(crate::encoder::encode_vex_shift(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xF1,
            0x71,
            6,
            reloc,
        )),
        "vpslld" => Some(crate::encoder::encode_vex_shift(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xF2,
            0x72,
            6,
            reloc,
        )),
        "vpsllq" => Some(crate::encoder::encode_vex_shift(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xF3,
            0x73,
            6,
            reloc,
        )),
        "vpsrlw" => Some(crate::encoder::encode_vex_shift(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xD1,
            0x71,
            2,
            reloc,
        )),
        "vpsrld" => Some(crate::encoder::encode_vex_shift(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xD2,
            0x72,
            2,
            reloc,
        )),
        "vpsrlq" => Some(crate::encoder::encode_vex_shift(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xD3,
            0x73,
            2,
            reloc,
        )),
        "vpsraw" => Some(crate::encoder::encode_vex_shift(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xE1,
            0x71,
            4,
            reloc,
        )),
        "vpsrad" => Some(crate::encoder::encode_vex_shift(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xE2,
            0x72,
            4,
            reloc,
        )),
        "vpslldq" => Some(crate::encoder::encode_vex_shift(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xF3,
            0x73,
            7,
            reloc,
        )),
        "vpsrldq" => Some(crate::encoder::encode_vex_shift(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xD3,
            0x73,
            3,
            reloc,
        )),

        // ── AVX2 variable shifts ────────────────────────────────────
        "vpsllvd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x47,
            None,
            false,
            reloc,
        )),
        "vpsllvq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x47,
            None,
            true,
            reloc,
        )),
        "vpsrlvd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x45,
            None,
            false,
            reloc,
        )),
        "vpsrlvq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x45,
            None,
            true,
            reloc,
        )),
        "vpsravd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x46,
            None,
            false,
            reloc,
        )),

        // ── AVX permute / shuffle ───────────────────────────────────
        "vpermilps" => {
            if ops
                .last()
                .is_some_and(|o| matches!(o, Operand::Immediate(_)))
            {
                Some(crate::encoder::encode_vex_imm(
                    buf,
                    ops,
                    instr,
                    0x66,
                    &[0x0F, 0x3A],
                    0x04,
                    false,
                    reloc,
                ))
            } else {
                Some(crate::encoder::encode_vex_op(
                    buf,
                    ops,
                    instr,
                    0x66,
                    &[0x0F, 0x38],
                    0x0C,
                    None,
                    false,
                    reloc,
                ))
            }
        }
        "vpermilpd" => {
            if ops
                .last()
                .is_some_and(|o| matches!(o, Operand::Immediate(_)))
            {
                Some(crate::encoder::encode_vex_imm(
                    buf,
                    ops,
                    instr,
                    0x66,
                    &[0x0F, 0x3A],
                    0x05,
                    false,
                    reloc,
                ))
            } else {
                Some(crate::encoder::encode_vex_op(
                    buf,
                    ops,
                    instr,
                    0x66,
                    &[0x0F, 0x38],
                    0x0D,
                    None,
                    false,
                    reloc,
                ))
            }
        }
        "vperm2f128" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x06,
            false,
            reloc,
        )),
        "vperm2i128" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x46,
            false,
            reloc,
        )),
        "vpermd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x36,
            None,
            false,
            reloc,
        )),
        "vpermps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x16,
            None,
            false,
            reloc,
        )),
        "vpermq" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x00,
            true,
            reloc,
        )),
        "vpermpd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x01,
            true,
            reloc,
        )),

        // ── AVX broadcast / insert / extract ────────────────────────
        "vbroadcastss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x18,
            None,
            false,
            reloc,
        )),
        "vbroadcastsd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x19,
            None,
            false,
            reloc,
        )),
        "vbroadcastf128" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x1A,
            None,
            false,
            reloc,
        )),
        "vpbroadcastb" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x78,
            None,
            false,
            reloc,
        )),
        "vpbroadcastw" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x79,
            None,
            false,
            reloc,
        )),
        "vpbroadcastd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x58,
            None,
            false,
            reloc,
        )),
        "vpbroadcastq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x59,
            None,
            false,
            reloc,
        )),
        "vbroadcasti128" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x5A,
            None,
            false,
            reloc,
        )),
        "vinsertf128" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x18,
            false,
            reloc,
        )),
        "vinserti128" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x38,
            false,
            reloc,
        )),
        "vextractf128" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x19,
            false,
            reloc,
        )),
        "vextracti128" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x39,
            false,
            reloc,
        )),
        "vpblendd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x02,
            false,
            reloc,
        )),

        // ── AVX masked moves ────────────────────────────────────────
        "vmaskmovps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x2C,
            Some(0x2E),
            false,
            reloc,
        )),
        "vmaskmovpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x2D,
            Some(0x2F),
            false,
            reloc,
        )),
        "vpmaskmovd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x8C,
            Some(0x8E),
            false,
            reloc,
        )),
        "vpmaskmovq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x8C,
            Some(0x8E),
            true,
            reloc,
        )),

        // ── AVX test ────────────────────────────────────────────────
        "vtestps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x0E,
            None,
            false,
            reloc,
        )),
        "vtestpd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x0F,
            None,
            false,
            reloc,
        )),

        // ── AVX conversions ─────────────────────────────────────────
        "vcvtsi2ss" => Some(crate::encoder::encode_vex_cvt(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x2A,
            reloc,
        )),
        "vcvtsi2sd" => Some(crate::encoder::encode_vex_cvt(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x2A,
            reloc,
        )),
        "vcvtss2si" => Some(crate::encoder::encode_vex_cvt(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x2D,
            reloc,
        )),
        "vcvtsd2si" => Some(crate::encoder::encode_vex_cvt(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x2D,
            reloc,
        )),
        "vcvttss2si" => Some(crate::encoder::encode_vex_cvt(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x2C,
            reloc,
        )),
        "vcvttsd2si" => Some(crate::encoder::encode_vex_cvt(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x2C,
            reloc,
        )),
        "vcvtss2sd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x5A,
            None,
            false,
            reloc,
        )),
        "vcvtsd2ss" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0x5A,
            None,
            false,
            reloc,
        )),
        "vcvtdq2ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x5B,
            None,
            false,
            reloc,
        )),
        "vcvtdq2pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0xE6,
            None,
            false,
            reloc,
        )),
        "vcvtps2dq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x5B,
            None,
            false,
            reloc,
        )),
        "vcvtps2pd" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x00,
            &[0x0F],
            0x5A,
            None,
            false,
            reloc,
        )),
        "vcvtpd2ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0x5A,
            None,
            false,
            reloc,
        )),
        "vcvtpd2dq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF2,
            &[0x0F],
            0xE6,
            None,
            false,
            reloc,
        )),
        "vcvttpd2dq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xE6,
            None,
            false,
            reloc,
        )),
        "vcvttps2dq" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0xF3,
            &[0x0F],
            0x5B,
            None,
            false,
            reloc,
        )),
        "vcvtph2ps" => Some(crate::encoder::encode_vex_op(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x38],
            0x13,
            None,
            false,
            reloc,
        )),
        "vcvtps2ph" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x1D,
            false,
            reloc,
        )),

        // ── AVX2 additional integer ─────────────────────────────────
        "vpblendvb" => {
            // 4-operand form with is4 byte: VEX.NDS.128/256.66.0F3A.W0 4C /r /is4
            // Handled as encode_vex_imm since is4 goes in imm8 position
            Some(crate::encoder::encode_vex_imm(
                buf,
                ops,
                instr,
                0x66,
                &[0x0F, 0x3A],
                0x4C,
                false,
                reloc,
            ))
        }
        "vblendvps" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x4A,
            false,
            reloc,
        )),
        "vblendvpd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x4B,
            false,
            reloc,
        )),
        "vpcmpestri" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x61,
            false,
            reloc,
        )),
        "vpcmpestrm" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x60,
            false,
            reloc,
        )),
        "vpcmpistri" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x63,
            false,
            reloc,
        )),
        "vpcmpistrm" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x62,
            false,
            reloc,
        )),
        "vpinsrb" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x20,
            false,
            reloc,
        )),
        "vpinsrw" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xC4,
            false,
            reloc,
        )),
        "vpinsrd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x22,
            false,
            reloc,
        )),
        "vpinsrq" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x22,
            true,
            reloc,
        )),
        "vpextrb" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x14,
            false,
            reloc,
        )),
        "vpextrw" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F],
            0xC5,
            false,
            reloc,
        )),
        "vpextrd" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x16,
            false,
            reloc,
        )),
        "vpextrq" => Some(crate::encoder::encode_vex_imm(
            buf,
            ops,
            instr,
            0x66,
            &[0x0F, 0x3A],
            0x16,
            true,
            reloc,
        )),

        _ => None,
    }
}

// ─── Special instruction encoders ───────────────────────────────────────────

/// Encode `r/m, reg` pattern: cmpxchg, xadd.
///
/// ModR/M encoding: reg-field = second operand (register), r/m-field = first operand.
fn encode_rm_reg(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pfx: u8,
    opcode: &[u8],
    opcode8: &[u8],
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 2 operands",
            instr.span,
        ));
    }
    match (&ops[0], &ops[1]) {
        // r/m=register, reg=register
        (Operand::Register(rm), Operand::Register(reg)) => {
            let size = reg_size(*rm);
            if size != reg_size(*reg) {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "operand size mismatch",
                    instr.span,
                ));
            }
            if pfx != 0 {
                buf.push(pfx);
            }
            if size == 8 && !opcode8.is_empty() {
                check_high_byte_rex_conflict(&[*rm, *reg], instr.span)?;
                let r = reg.is_extended();
                let b = rm.is_extended();
                let need = r || b || reg.requires_rex_for_byte() || rm.requires_rex_for_byte();
                if need {
                    buf.push(rex(false, r, false, b));
                }
                buf.extend_from_slice(opcode8);
                buf.push(modrm(0b11, reg.base_code(), rm.base_code()));
            } else {
                let w = size == 64;
                let r = reg.is_extended();
                let b = rm.is_extended();
                if size == 16 {
                    buf.push(0x66);
                }
                if needs_rex(w, r, false, b) {
                    buf.push(rex(w, r, false, b));
                }
                buf.extend_from_slice(opcode);
                buf.push(modrm(0b11, reg.base_code(), rm.base_code()));
            }
        }
        // r/m=memory, reg=register
        (Operand::Memory(mem), Operand::Register(reg)) => {
            let size = reg_size(*reg);
            if pfx != 0 {
                buf.push(pfx);
            }
            if size == 8 && !opcode8.is_empty() {
                let r = reg.is_extended();
                let x = mem.index.is_some_and(|i| i.is_extended());
                let b = mem.base.is_some_and(|b| b.is_extended());
                let need = r || x || b || reg.requires_rex_for_byte();
                if need {
                    buf.push(rex(false, r, x, b));
                }
                buf.extend_from_slice(opcode8);
                let disp_off = emit_mem_modrm(buf, reg.base_code(), mem);
                set_mem_reloc(reloc, mem, disp_off, buf.len());
            } else {
                emit_rex_for_reg_mem(buf, *reg, mem)?;
                buf.extend_from_slice(opcode);
                let disp_off = emit_mem_modrm(buf, reg.base_code(), mem);
                set_mem_reloc(reloc, mem, disp_off, buf.len());
            }
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected r/m, reg",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// IN — Input from port.
///
/// Forms: `in al/ax/eax, imm8` or `in al/ax/eax, dx`
fn encode_in(buf: &mut InstrBytes, ops: &[Operand], instr: &Instruction) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(
            "in",
            "expected 2 operands (al/ax/eax, imm8/dx)",
            instr.span,
        ));
    }
    match (&ops[0], &ops[1]) {
        // in al, imm8
        (Operand::Register(Register::Al), Operand::Immediate(imm)) => {
            validate_imm_u8(*imm, instr.span)?;
            buf.push(0xE4);
            buf.push(*imm as u8);
        }
        // in ax, imm8
        (Operand::Register(Register::Ax), Operand::Immediate(imm)) => {
            validate_imm_u8(*imm, instr.span)?;
            buf.push(0x66);
            buf.push(0xE5);
            buf.push(*imm as u8);
        }
        // in eax, imm8
        (Operand::Register(Register::Eax), Operand::Immediate(imm)) => {
            validate_imm_u8(*imm, instr.span)?;
            buf.push(0xE5);
            buf.push(*imm as u8);
        }
        // in al, dx
        (Operand::Register(Register::Al), Operand::Register(Register::Dx)) => {
            buf.push(0xEC);
        }
        // in ax, dx
        (Operand::Register(Register::Ax), Operand::Register(Register::Dx)) => {
            buf.push(0x66);
            buf.push(0xED);
        }
        // in eax, dx
        (Operand::Register(Register::Eax), Operand::Register(Register::Dx)) => {
            buf.push(0xED);
        }
        _ => {
            return Err(invalid_operands(
                "in",
                "expected al/ax/eax, imm8 or al/ax/eax, dx",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// OUT — Output to port.
///
/// Forms: `out imm8, al/ax/eax` or `out dx, al/ax/eax`
fn encode_out(buf: &mut InstrBytes, ops: &[Operand], instr: &Instruction) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(
            "out",
            "expected 2 operands (imm8/dx, al/ax/eax)",
            instr.span,
        ));
    }
    match (&ops[0], &ops[1]) {
        // out imm8, al
        (Operand::Immediate(imm), Operand::Register(Register::Al)) => {
            validate_imm_u8(*imm, instr.span)?;
            buf.push(0xE6);
            buf.push(*imm as u8);
        }
        // out imm8, ax
        (Operand::Immediate(imm), Operand::Register(Register::Ax)) => {
            validate_imm_u8(*imm, instr.span)?;
            buf.push(0x66);
            buf.push(0xE7);
            buf.push(*imm as u8);
        }
        // out imm8, eax
        (Operand::Immediate(imm), Operand::Register(Register::Eax)) => {
            validate_imm_u8(*imm, instr.span)?;
            buf.push(0xE7);
            buf.push(*imm as u8);
        }
        // out dx, al
        (Operand::Register(Register::Dx), Operand::Register(Register::Al)) => {
            buf.push(0xEE);
        }
        // out dx, ax
        (Operand::Register(Register::Dx), Operand::Register(Register::Ax)) => {
            buf.push(0x66);
            buf.push(0xEF);
        }
        // out dx, eax
        (Operand::Register(Register::Dx), Operand::Register(Register::Eax)) => {
            buf.push(0xEF);
        }
        _ => {
            return Err(invalid_operands(
                "out",
                "expected imm8/dx, al/ax/eax",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// ENTER — Create stack frame.
///
/// Form: `enter imm16, imm8`
fn encode_enter(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(
            "enter",
            "expected 2 operands (imm16, imm8)",
            instr.span,
        ));
    }
    match (&ops[0], &ops[1]) {
        (Operand::Immediate(frame_size), Operand::Immediate(nesting_level)) => {
            validate_imm_u16(*frame_size, instr.span)?;
            validate_imm_u8(*nesting_level, instr.span)?;
            buf.push(0xC8);
            buf.extend_from_slice(&(*frame_size as u16).to_le_bytes());
            buf.push(*nesting_level as u8);
        }
        _ => {
            return Err(invalid_operands(
                "enter",
                "expected imm16, imm8",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// JECXZ — Jump short if ECX is zero (uses address-size override in 64-bit mode).
///
/// Short form: `67 E3 rel8` (3 bytes)
/// Long form:  `67 E3 02 / EB 05 / E9 rel32` (10 bytes)
///
/// The JECXZ instruction only supports rel8. When the target is out of ±127
/// byte range, the encoder synthesizes a compound sequence:
///   JECXZ +2    → if ECX==0, jump over the skip-JMP to the long JMP
///   JMP short +5 → ECX≠0, skip the long JMP
///   JMP rel32    → long jump to actual target
fn encode_jecxz(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
    relax: &mut Option<crate::encoder::RelaxInfo>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            "jecxz",
            "expected 1 operand (label)",
            instr.span,
        ));
    }
    if let Some((label, addend)) = extract_label(&ops[0]) {
        // Long form (10 bytes): JECXZ +2 / JMP_short +5 / JMP_near rel32
        //   [0] 0x67  address-size override
        //   [1] 0xE3  JECXZ
        //   [2] 0x02  rel8 = +2
        //   [3] 0xEB  JMP short
        //   [4] 0x05  rel8 = +5
        //   [5] 0xE9  JMP near
        //   [6..10] rel32 placeholder
        buf.push(0x67);
        buf.push(0xE3);
        buf.push(0x02);
        buf.push(0xEB);
        buf.push(0x05);
        buf.push(0xE9);
        let reloc_off = buf.len();
        buf.extend_from_slice(&0i32.to_le_bytes());
        *reloc = Some(Relocation {
            offset: reloc_off,
            size: 4,
            label: alloc::rc::Rc::from(label),
            kind: RelocKind::X86Relative,
            addend,
            trailing_bytes: 0,
        });
        // Short form: 67 E3 rel8 (3 bytes)
        *relax = Some(crate::encoder::RelaxInfo {
            short_bytes: InstrBytes::from_slice(&[0x67, 0xE3, 0x00]),
            short_reloc_offset: 2,
            short_relocation: None,
        });
    } else {
        return Err(invalid_operands(
            "jecxz",
            "expected label operand",
            instr.span,
        ));
    }
    Ok(())
}

/// JRCXZ — Jump short if RCX is zero (default in 64-bit mode).
///
/// Short form: `E3 rel8` (2 bytes)
/// Long form:  `E3 02 / EB 05 / E9 rel32` (9 bytes)
///
/// Same relaxation strategy as JECXZ but without the address-size override prefix.
fn encode_jrcxz(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
    relax: &mut Option<crate::encoder::RelaxInfo>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            "jrcxz",
            "expected 1 operand (label)",
            instr.span,
        ));
    }
    if let Some((label, addend)) = extract_label(&ops[0]) {
        // Long form (9 bytes): JRCXZ +2 / JMP_short +5 / JMP_near rel32
        buf.push(0xE3);
        buf.push(0x02);
        buf.push(0xEB);
        buf.push(0x05);
        buf.push(0xE9);
        let reloc_off = buf.len();
        buf.extend_from_slice(&0i32.to_le_bytes());
        *reloc = Some(Relocation {
            offset: reloc_off,
            size: 4,
            label: alloc::rc::Rc::from(label),
            kind: RelocKind::X86Relative,
            addend,
            trailing_bytes: 0,
        });
        // Short form: E3 rel8 (2 bytes)
        *relax = Some(crate::encoder::RelaxInfo {
            short_bytes: InstrBytes::from_slice(&[0xE3, 0x00]),
            short_reloc_offset: 1,
            short_relocation: None,
        });
    } else {
        return Err(invalid_operands(
            "jrcxz",
            "expected label operand",
            instr.span,
        ));
    }
    Ok(())
}

/// RDRAND / RDSEED — Read hardware random number.
///
/// RDRAND: `0F C7 /6` (ModR/M mod=11, digit=6)
/// RDSEED: `0F C7 /7` (ModR/M mod=11, digit=7)
fn encode_rdrand_rdseed(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    digit: u8,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 register operand",
            instr.span,
        ));
    }
    match &ops[0] {
        Operand::Register(reg) => {
            let size = reg_size(*reg);
            if size == 8 {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "8-bit register not supported",
                    instr.span,
                ));
            }
            let w = size == 64;
            let b = reg.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, false, false, b) {
                buf.push(rex(w, false, false, b));
            }
            buf.push(0x0F);
            buf.push(0xC7);
            buf.push(modrm(0b11, digit, reg.base_code()));
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected register operand",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// ADCX / ADOX — Multi-precision add with carry/overflow (ADX extension).
///
/// ADCX: `66 [REX.W] 0F 38 F6 /r`  (add with CF, destination = reg, source = r/m)
/// ADOX: `F3 [REX.W] 0F 38 F6 /r`  (add with OF, destination = reg, source = r/m)
///
/// Only 32-bit and 64-bit operand sizes are supported.
fn encode_adcx_adox(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    prefix: u8,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 2 operands (reg, r/m)",
            instr.span,
        ));
    }
    let dst = match &ops[0] {
        Operand::Register(r) => *r,
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "destination must be a register",
                instr.span,
            ))
        }
    };
    let size = reg_size(dst);
    if size != 32 && size != 64 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "only 32-bit and 64-bit operands supported",
            instr.span,
        ));
    }
    // Mandatory prefix
    buf.push(prefix);
    match &ops[1] {
        Operand::Register(src) => {
            let src_size = reg_size(*src);
            if src_size != size {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "operand size mismatch",
                    instr.span,
                ));
            }
            let w = size == 64;
            let r = dst.is_extended();
            let b = src.is_extended();
            if needs_rex(w, r, false, b) {
                buf.push(rex(w, r, false, b));
            }
            buf.extend_from_slice(&[0x0F, 0x38, 0xF6]);
            buf.push(modrm(0b11, dst.base_code(), src.base_code()));
        }
        Operand::Memory(mem) => {
            emit_rex_for_reg_mem(buf, dst, mem)?;
            buf.extend_from_slice(&[0x0F, 0x38, 0xF6]);
            let reloc_off = emit_mem_modrm(buf, dst.base_code(), mem);
            set_mem_reloc(reloc, mem, reloc_off, buf.len());
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "source must be register or memory",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// CMPXCHG8B / CMPXCHG16B — Compare and exchange 8/16 bytes.
///
/// CMPXCHG8B:  `0F C7 /1` (memory operand only)
/// CMPXCHG16B: `REX.W 0F C7 /1` (memory operand only)
fn encode_cmpxchg_wide(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    wide: bool,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 memory operand",
            instr.span,
        ));
    }
    match &ops[0] {
        Operand::Memory(mem) => {
            if wide {
                // CMPXCHG16B always needs REX.W
                let x = mem.index.is_some_and(|r| r.is_extended());
                let b = mem.base.is_some_and(|r| r.is_extended());
                buf.push(rex(true, false, x, b));
            } else {
                // CMPXCHG8B needs REX only if extended registers are used
                emit_rex_for_digit_mem(buf, 32, mem);
            }
            buf.push(0x0F);
            buf.push(0xC7);
            let disp_off = emit_mem_modrm(buf, 1, mem); // /1
            set_mem_reloc(reloc, mem, disp_off, buf.len());
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected memory operand",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// RDFSBASE / RDGSBASE / WRFSBASE / WRGSBASE — Read/Write FS/GS segment base.
///
/// Encoding: `F3 0F AE /digit` where digit selects the operation:
///   - /0 = rdfsbase, /1 = rdgsbase, /2 = wrfsbase, /3 = wrgsbase
///
/// REX.W selects 64-bit register operand.
fn encode_fsgsbase(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    digit: u8,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 register operand",
            instr.span,
        ));
    }
    match &ops[0] {
        Operand::Register(reg) => {
            let size = reg_size(*reg);
            if size != 32 && size != 64 {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "expected 32-bit or 64-bit register",
                    instr.span,
                ));
            }
            let w = size == 64;
            let b = reg.is_extended();
            buf.push(0xF3);
            if needs_rex(w, false, false, b) {
                buf.push(rex(w, false, false, b));
            }
            buf.push(0x0F);
            buf.push(0xAE);
            buf.push(modrm(0b11, digit, reg.base_code()));
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected register operand",
            instr.span,
        )),
    }
}

/// 0F AE /digit (memory only) — FXSAVE/FXRSTOR/XSAVE/XRSTOR/XSAVEOPT.
///
/// digit: 0=fxsave, 1=fxrstor, 4=xsave, 5=xrstor, 6=xsaveopt
/// wide: when true, emit REX.W for 64-bit variants (fxsave64, xsave64, etc.)
fn encode_0f_ae_mem(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    digit: u8,
    wide: bool,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 memory operand",
            instr.span,
        ));
    }
    match &ops[0] {
        Operand::Memory(mem) => {
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if wide {
                buf.push(rex(true, false, x, b));
            } else if needs_rex(false, false, x, b) {
                buf.push(rex(false, false, x, b));
            }
            buf.push(0x0F);
            buf.push(0xAE);
            let disp_off = emit_mem_modrm(buf, digit, mem);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected memory operand",
            instr.span,
        )),
    }
}

/// 0F C7 /digit (memory only) — XSAVEC/XSAVES/XRSTORS.
///
/// digit: 3=xrstors, 4=xsavec, 5=xsaves
/// wide: when true, emit REX.W for 64-bit variants
fn encode_0f_c7_mem(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    digit: u8,
    wide: bool,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 memory operand",
            instr.span,
        ));
    }
    match &ops[0] {
        Operand::Memory(mem) => {
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if wide {
                buf.push(rex(true, false, x, b));
            } else if needs_rex(false, false, x, b) {
                buf.push(rex(false, false, x, b));
            }
            buf.push(0x0F);
            buf.push(0xC7);
            let disp_off = emit_mem_modrm(buf, digit, mem);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected memory operand",
            instr.span,
        )),
    }
}

/// XABORT — Abort TSX transaction with immediate reason code.
///
/// Encoding: `C6 F8 imm8`
fn encode_xabort(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 immediate operand",
            instr.span,
        ));
    }
    match &ops[0] {
        Operand::Immediate(imm) => {
            let val = *imm;
            if !(0..=255).contains(&val) && !(-128..=127).contains(&val) {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "immediate must fit in 8 bits",
                    instr.span,
                ));
            }
            buf.push(0xC6);
            buf.push(0xF8);
            buf.push(val as u8);
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected immediate operand",
            instr.span,
        )),
    }
}

/// XBEGIN — Begin TSX transaction with rel32 fallback offset.
///
/// Encoding: `C7 F8 rel32`
/// The rel32 is relative to the end of the instruction (6 bytes total).
fn encode_xbegin(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 label/immediate operand",
            instr.span,
        ));
    }
    buf.push(0xC7);
    buf.push(0xF8);
    match &ops[0] {
        Operand::Immediate(imm) => {
            // Direct rel32 offset
            let val = *imm as i32;
            buf.extend_from_slice(&val.to_le_bytes());
            Ok(())
        }
        Operand::Label(name) => {
            // Label reference — emit placeholder and set up relocation
            let disp_offset = buf.len();
            buf.extend_from_slice(&0i32.to_le_bytes());
            *reloc = Some(Relocation {
                offset: disp_offset,
                size: 4,
                label: alloc::rc::Rc::from(name.as_str()),
                kind: RelocKind::X86Relative,
                addend: 0,
                trailing_bytes: 0,
            });
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected label or immediate operand",
            instr.span,
        )),
    }
}

/// MOVNTI — Store doubleword/quadword using non-temporal hint.
///
/// Form: `movnti [mem], r32/r64` → `0F C3 /r`
fn encode_movnti(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(
            "movnti",
            "expected 2 operands ([mem], reg)",
            instr.span,
        ));
    }
    match (&ops[0], &ops[1]) {
        (Operand::Memory(mem), Operand::Register(src)) => {
            let size = reg_size(*src);
            if size != 32 && size != 64 {
                return Err(invalid_operands(
                    "movnti",
                    "requires 32 or 64-bit register",
                    instr.span,
                ));
            }
            emit_rex_for_reg_mem(buf, *src, mem)?;
            buf.push(0x0F);
            buf.push(0xC3);
            let disp_off = emit_mem_modrm(buf, src.base_code(), mem);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
        }
        _ => {
            return Err(invalid_operands(
                "movnti",
                "expected [mem], reg",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// MOVBE — Move data after byte-swapping.
///
/// Load: `movbe r, [mem]` → `0F 38 F0 /r`
/// Store: `movbe [mem], r` → `0F 38 F1 /r`
fn encode_movbe(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands("movbe", "expected 2 operands", instr.span));
    }
    match (&ops[0], &ops[1]) {
        // Load: movbe reg, [mem]
        (Operand::Register(dst), Operand::Memory(mem)) => {
            let size = reg_size(*dst);
            if size == 8 {
                return Err(invalid_operands(
                    "movbe",
                    "8-bit operand not supported",
                    instr.span,
                ));
            }
            emit_rex_for_reg_mem(buf, *dst, mem)?;
            buf.push(0x0F);
            buf.push(0x38);
            buf.push(0xF0);
            let disp_off = emit_mem_modrm(buf, dst.base_code(), mem);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
        }
        // Store: movbe [mem], reg
        (Operand::Memory(mem), Operand::Register(src)) => {
            let size = reg_size(*src);
            if size == 8 {
                return Err(invalid_operands(
                    "movbe",
                    "8-bit operand not supported",
                    instr.span,
                ));
            }
            emit_rex_for_reg_mem(buf, *src, mem)?;
            buf.push(0x0F);
            buf.push(0x38);
            buf.push(0xF1);
            let disp_off = emit_mem_modrm(buf, src.base_code(), mem);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
        }
        _ => {
            return Err(invalid_operands(
                "movbe",
                "expected reg, [mem] or [mem], reg",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// SHLD / SHRD — Double-precision shift.
///
/// SHLD: `shld r/m, reg, imm8` → `0F A4 /r ib`
///       `shld r/m, reg, cl`   → `0F A5 /r`
/// SHRD: `shrd r/m, reg, imm8` → `0F AC /r ib`
///       `shrd r/m, reg, cl`   → `0F AD /r`
fn encode_shld_shrd(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    imm_opcode: &[u8],
    cl_opcode: &[u8],
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 3 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 3 operands (r/m, reg, imm8/cl)",
            instr.span,
        ));
    }
    // SHLD/SHRD only support 16/32/64-bit operands — check both r/m and reg
    if let Operand::Register(r) = &ops[0] {
        if reg_size(*r) == 8 {
            return Err(invalid_operands(
                &instr.mnemonic,
                "8-bit operands not supported",
                instr.span,
            ));
        }
    }
    if let Operand::Register(r) = &ops[1] {
        if reg_size(*r) == 8 {
            return Err(invalid_operands(
                &instr.mnemonic,
                "8-bit operands not supported",
                instr.span,
            ));
        }
    }
    match (&ops[0], &ops[1], &ops[2]) {
        // r/m=reg, reg, imm8
        (Operand::Register(rm), Operand::Register(reg), Operand::Immediate(imm)) => {
            let size = reg_size(*rm);
            if size != reg_size(*reg) {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "operand size mismatch",
                    instr.span,
                ));
            }
            validate_imm_u8(*imm, instr.span)?;
            let w = size == 64;
            let r = reg.is_extended();
            let b = rm.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, r, false, b) {
                buf.push(rex(w, r, false, b));
            }
            buf.extend_from_slice(imm_opcode);
            buf.push(modrm(0b11, reg.base_code(), rm.base_code()));
            buf.push(*imm as u8);
        }
        // r/m=reg, reg, cl
        (Operand::Register(rm), Operand::Register(reg), Operand::Register(Register::Cl)) => {
            let size = reg_size(*rm);
            if size != reg_size(*reg) {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "operand size mismatch",
                    instr.span,
                ));
            }
            let w = size == 64;
            let r = reg.is_extended();
            let b = rm.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, r, false, b) {
                buf.push(rex(w, r, false, b));
            }
            buf.extend_from_slice(cl_opcode);
            buf.push(modrm(0b11, reg.base_code(), rm.base_code()));
        }
        // [mem], reg, imm8
        (Operand::Memory(mem), Operand::Register(reg), Operand::Immediate(imm)) => {
            validate_imm_u8(*imm, instr.span)?;
            emit_rex_for_reg_mem(buf, *reg, mem)?;
            buf.extend_from_slice(imm_opcode);
            let disp_off = emit_mem_modrm(buf, reg.base_code(), mem);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
            buf.push(*imm as u8);
        }
        // [mem], reg, cl
        (Operand::Memory(mem), Operand::Register(reg), Operand::Register(Register::Cl)) => {
            emit_rex_for_reg_mem(buf, *reg, mem)?;
            buf.extend_from_slice(cl_opcode);
            let disp_off = emit_mem_modrm(buf, reg.base_code(), mem);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected r/m, reg, imm8/cl",
                instr.span,
            ))
        }
    }
    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::encode_instruction;
    use crate::error::Span;
    use alloc::string::String;
    use alloc::vec;

    fn span() -> Span {
        Span::new(1, 1, 0, 0)
    }

    fn make_instr(mnemonic: &str, operands: Vec<Operand>) -> Instruction {
        Instruction {
            mnemonic: Mnemonic::from(mnemonic),
            operands: OperandList::from(operands),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        }
    }

    fn encode(mnemonic: &str, operands: Vec<Operand>) -> Vec<u8> {
        let instr = make_instr(mnemonic, operands);
        encode_instruction(&instr, Arch::X86_64)
            .unwrap()
            .bytes
            .to_vec()
    }

    fn encode_with_prefix(mnemonic: &str, operands: Vec<Operand>, prefix: Prefix) -> Vec<u8> {
        let mut instr = make_instr(mnemonic, operands);
        instr.prefixes = PrefixList::from(vec![prefix]);
        encode_instruction(&instr, Arch::X86_64)
            .unwrap()
            .bytes
            .to_vec()
    }

    fn encode_err(mnemonic: &str, operands: Vec<Operand>) -> bool {
        let instr = make_instr(mnemonic, operands);
        encode_instruction(&instr, Arch::X86_64).is_err()
    }

    // ─── Table sort invariant ─────────────────────────────────────────────

    #[test]
    fn fixed_table_is_sorted() {
        for w in FIXED_TABLE.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "FIXED_TABLE not sorted: {:?} >= {:?}",
                w[0].0,
                w[1].0
            );
        }
    }

    // ─── Fixed-encoding instructions ──────────────────────────────────────

    #[test]
    fn leave() {
        assert_eq!(encode("leave", vec![]), vec![0xC9]);
    }

    #[test]
    fn ud2() {
        assert_eq!(encode("ud2", vec![]), vec![0x0F, 0x0B]);
    }

    #[test]
    fn cpuid() {
        assert_eq!(encode("cpuid", vec![]), vec![0x0F, 0xA2]);
    }

    #[test]
    fn rdtsc() {
        assert_eq!(encode("rdtsc", vec![]), vec![0x0F, 0x31]);
    }

    #[test]
    fn rdtscp() {
        assert_eq!(encode("rdtscp", vec![]), vec![0x0F, 0x01, 0xF9]);
    }

    #[test]
    fn pushf_popf() {
        assert_eq!(encode("pushf", vec![]), vec![0x9C]);
        assert_eq!(encode("pushfq", vec![]), vec![0x9C]);
        assert_eq!(encode("pushfw", vec![]), vec![0x66, 0x9C]);
        assert_eq!(encode("popf", vec![]), vec![0x9D]);
        assert_eq!(encode("popfq", vec![]), vec![0x9D]);
        assert_eq!(encode("popfw", vec![]), vec![0x66, 0x9D]);
    }

    #[test]
    fn pause() {
        assert_eq!(encode("pause", vec![]), vec![0xF3, 0x90]);
    }

    #[test]
    fn memory_fences() {
        assert_eq!(encode("mfence", vec![]), vec![0x0F, 0xAE, 0xF0]);
        assert_eq!(encode("lfence", vec![]), vec![0x0F, 0xAE, 0xE8]);
        assert_eq!(encode("sfence", vec![]), vec![0x0F, 0xAE, 0xF8]);
    }

    #[test]
    fn system_instructions() {
        assert_eq!(encode("sysenter", vec![]), vec![0x0F, 0x34]);
        assert_eq!(encode("sysexit", vec![]), vec![0x0F, 0x35]);
        assert_eq!(encode("swapgs", vec![]), vec![0x0F, 0x01, 0xF8]);
        assert_eq!(encode("wrmsr", vec![]), vec![0x0F, 0x30]);
        assert_eq!(encode("rdmsr", vec![]), vec![0x0F, 0x32]);
        assert_eq!(encode("clts", vec![]), vec![0x0F, 0x06]);
        assert_eq!(encode("wbinvd", vec![]), vec![0x0F, 0x09]);
        assert_eq!(encode("invd", vec![]), vec![0x0F, 0x08]);
    }

    #[test]
    fn cet_instructions() {
        assert_eq!(encode("endbr64", vec![]), vec![0xF3, 0x0F, 0x1E, 0xFA]);
        assert_eq!(encode("endbr32", vec![]), vec![0xF3, 0x0F, 0x1E, 0xFB]);
    }

    #[test]
    fn xlatb() {
        assert_eq!(encode("xlatb", vec![]), vec![0xD7]);
        assert_eq!(encode("xlat", vec![]), vec![0xD7]);
    }

    #[test]
    fn iret_variants() {
        assert_eq!(encode("iret", vec![]), vec![0xCF]);
        assert_eq!(encode("iretd", vec![]), vec![0xCF]);
        assert_eq!(encode("iretq", vec![]), vec![0x48, 0xCF]);
    }

    #[test]
    fn string_io() {
        assert_eq!(encode("insb", vec![]), vec![0x6C]);
        assert_eq!(encode("insw", vec![]), vec![0x66, 0x6D]);
        assert_eq!(encode("insd", vec![]), vec![0x6D]);
        assert_eq!(encode("outsb", vec![]), vec![0x6E]);
        assert_eq!(encode("outsw", vec![]), vec![0x66, 0x6F]);
        assert_eq!(encode("outsd", vec![]), vec![0x6F]);
    }

    #[test]
    fn icebp_int1() {
        assert_eq!(encode("icebp", vec![]), vec![0xF1]);
        assert_eq!(encode("int1", vec![]), vec![0xF1]);
    }

    #[test]
    fn misc_fixed() {
        assert_eq!(encode("emms", vec![]), vec![0x0F, 0x77]);
        assert_eq!(encode("monitor", vec![]), vec![0x0F, 0x01, 0xC8]);
        assert_eq!(encode("mwait", vec![]), vec![0x0F, 0x01, 0xC9]);
        assert_eq!(encode("xgetbv", vec![]), vec![0x0F, 0x01, 0xD0]);
        assert_eq!(encode("xsetbv", vec![]), vec![0x0F, 0x01, 0xD1]);
    }

    #[test]
    fn fixed_rejects_operands() {
        assert!(encode_err("leave", vec![Operand::Register(Register::Rax)]));
        assert!(encode_err("cpuid", vec![Operand::Immediate(0)]));
    }

    // ─── CMPXCHG ──────────────────────────────────────────────────────────

    #[test]
    fn cmpxchg_reg_reg_32() {
        // cmpxchg ecx, eax → 0F B1 ModR/M(11, eax=0, ecx=1)
        assert_eq!(
            encode(
                "cmpxchg",
                vec![
                    Operand::Register(Register::Ecx),
                    Operand::Register(Register::Eax)
                ]
            ),
            vec![0x0F, 0xB1, 0xC1]
        );
    }

    #[test]
    fn cmpxchg_reg_reg_64() {
        // cmpxchg rdi, rax → REX.W 0F B1 ModR/M(11, rax=0, rdi=7)
        assert_eq!(
            encode(
                "cmpxchg",
                vec![
                    Operand::Register(Register::Rdi),
                    Operand::Register(Register::Rax)
                ]
            ),
            vec![0x48, 0x0F, 0xB1, 0xC7]
        );
    }

    #[test]
    fn cmpxchg_reg_reg_8() {
        // cmpxchg cl, al → 0F B0 ModR/M(11, al=0, cl=1)
        assert_eq!(
            encode(
                "cmpxchg",
                vec![
                    Operand::Register(Register::Cl),
                    Operand::Register(Register::Al)
                ]
            ),
            vec![0x0F, 0xB0, 0xC1]
        );
    }

    #[test]
    fn cmpxchg_mem_reg() {
        // cmpxchg [rdi], eax → 0F B1 ModR/M(00, eax=0, rdi=7)
        let mem = MemoryOperand {
            base: Some(Register::Rdi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode(
                "cmpxchg",
                vec![
                    Operand::Memory(Box::new(mem)),
                    Operand::Register(Register::Eax)
                ]
            ),
            vec![0x0F, 0xB1, 0x07]
        );
    }

    #[test]
    fn lock_cmpxchg_mem_reg_64() {
        // lock cmpxchg [rdi], rax → F0 REX.W 0F B1 07
        let mem = MemoryOperand {
            base: Some(Register::Rdi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode_with_prefix(
                "cmpxchg",
                vec![
                    Operand::Memory(Box::new(mem)),
                    Operand::Register(Register::Rax)
                ],
                Prefix::Lock,
            ),
            vec![0xF0, 0x48, 0x0F, 0xB1, 0x07]
        );
    }

    // ─── XADD ─────────────────────────────────────────────────────────────

    #[test]
    fn xadd_reg_reg_32() {
        // xadd ecx, eax → 0F C1 ModR/M(11, eax=0, ecx=1)
        assert_eq!(
            encode(
                "xadd",
                vec![
                    Operand::Register(Register::Ecx),
                    Operand::Register(Register::Eax)
                ]
            ),
            vec![0x0F, 0xC1, 0xC1]
        );
    }

    #[test]
    fn xadd_mem_reg_64() {
        // xadd [rsi], rbx → REX.W 0F C1 ModR/M(00, rbx=3, rsi=6)
        let mem = MemoryOperand {
            base: Some(Register::Rsi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode(
                "xadd",
                vec![
                    Operand::Memory(Box::new(mem)),
                    Operand::Register(Register::Rbx)
                ]
            ),
            vec![0x48, 0x0F, 0xC1, 0x1E]
        );
    }

    // ─── IN / OUT ─────────────────────────────────────────────────────────

    #[test]
    fn in_al_imm() {
        assert_eq!(
            encode(
                "in",
                vec![Operand::Register(Register::Al), Operand::Immediate(0x60)]
            ),
            vec![0xE4, 0x60]
        );
    }

    #[test]
    fn in_eax_imm() {
        assert_eq!(
            encode(
                "in",
                vec![Operand::Register(Register::Eax), Operand::Immediate(0x80)]
            ),
            vec![0xE5, 0x80]
        );
    }

    #[test]
    fn in_al_dx() {
        assert_eq!(
            encode(
                "in",
                vec![
                    Operand::Register(Register::Al),
                    Operand::Register(Register::Dx)
                ]
            ),
            vec![0xEC]
        );
    }

    #[test]
    fn in_eax_dx() {
        assert_eq!(
            encode(
                "in",
                vec![
                    Operand::Register(Register::Eax),
                    Operand::Register(Register::Dx)
                ]
            ),
            vec![0xED]
        );
    }

    #[test]
    fn in_ax_dx() {
        assert_eq!(
            encode(
                "in",
                vec![
                    Operand::Register(Register::Ax),
                    Operand::Register(Register::Dx)
                ]
            ),
            vec![0x66, 0xED]
        );
    }

    #[test]
    fn out_imm_al() {
        assert_eq!(
            encode(
                "out",
                vec![Operand::Immediate(0x80), Operand::Register(Register::Al)]
            ),
            vec![0xE6, 0x80]
        );
    }

    #[test]
    fn out_imm_eax() {
        assert_eq!(
            encode(
                "out",
                vec![Operand::Immediate(0x60), Operand::Register(Register::Eax)]
            ),
            vec![0xE7, 0x60]
        );
    }

    #[test]
    fn out_dx_al() {
        assert_eq!(
            encode(
                "out",
                vec![
                    Operand::Register(Register::Dx),
                    Operand::Register(Register::Al)
                ]
            ),
            vec![0xEE]
        );
    }

    #[test]
    fn out_dx_eax() {
        assert_eq!(
            encode(
                "out",
                vec![
                    Operand::Register(Register::Dx),
                    Operand::Register(Register::Eax)
                ]
            ),
            vec![0xEF]
        );
    }

    // ─── ENTER ────────────────────────────────────────────────────────────

    #[test]
    fn enter_basic() {
        // enter 0x100, 0 → C8 00 01 00
        assert_eq!(
            encode(
                "enter",
                vec![Operand::Immediate(0x100), Operand::Immediate(0)]
            ),
            vec![0xC8, 0x00, 0x01, 0x00]
        );
    }

    #[test]
    fn enter_with_nesting() {
        // enter 32, 1 → C8 20 00 01
        assert_eq!(
            encode("enter", vec![Operand::Immediate(32), Operand::Immediate(1)]),
            vec![0xC8, 0x20, 0x00, 0x01]
        );
    }

    // ─── RDRAND / RDSEED ──────────────────────────────────────────────────

    #[test]
    fn rdrand_eax() {
        // rdrand eax → 0F C7 ModR/M(11, /6, eax=0) = 0F C7 F0
        assert_eq!(
            encode("rdrand", vec![Operand::Register(Register::Eax)]),
            vec![0x0F, 0xC7, 0xF0]
        );
    }

    #[test]
    fn rdrand_rax() {
        // rdrand rax → REX.W 0F C7 F0
        assert_eq!(
            encode("rdrand", vec![Operand::Register(Register::Rax)]),
            vec![0x48, 0x0F, 0xC7, 0xF0]
        );
    }

    #[test]
    fn rdrand_r12d() {
        // rdrand r12d → REX.B 0F C7 ModR/M(11, /6, r12=4) = 41 0F C7 F4
        assert_eq!(
            encode("rdrand", vec![Operand::Register(Register::R12d)]),
            vec![0x41, 0x0F, 0xC7, 0xF4]
        );
    }

    #[test]
    fn rdseed_ecx() {
        // rdseed ecx → 0F C7 ModR/M(11, /7, ecx=1) = 0F C7 F9
        assert_eq!(
            encode("rdseed", vec![Operand::Register(Register::Ecx)]),
            vec![0x0F, 0xC7, 0xF9]
        );
    }

    #[test]
    fn rdrand_rejects_8bit() {
        assert!(encode_err("rdrand", vec![Operand::Register(Register::Al)]));
    }

    // ─── CMPXCHG8B / CMPXCHG16B ──────────────────────────────────────────

    #[test]
    fn cmpxchg8b_mem() {
        // cmpxchg8b [rdi] → 0F C7 ModR/M(00, /1, rdi=7) = 0F C7 0F
        let mem = MemoryOperand {
            base: Some(Register::Rdi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode("cmpxchg8b", vec![Operand::Memory(Box::new(mem))]),
            vec![0x0F, 0xC7, 0x0F]
        );
    }

    #[test]
    fn cmpxchg16b_mem() {
        // cmpxchg16b [rdi] → REX.W 0F C7 ModR/M(00, /1, rdi=7) = 48 0F C7 0F
        let mem = MemoryOperand {
            base: Some(Register::Rdi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode("cmpxchg16b", vec![Operand::Memory(Box::new(mem))]),
            vec![0x48, 0x0F, 0xC7, 0x0F]
        );
    }

    #[test]
    fn cmpxchg8b_rejects_register() {
        assert!(encode_err(
            "cmpxchg8b",
            vec![Operand::Register(Register::Rax)]
        ));
    }

    // ─── MOVNTI ───────────────────────────────────────────────────────────

    #[test]
    fn movnti_mem_eax() {
        // movnti [rdi], eax → 0F C3 ModR/M(00, eax=0, rdi=7) = 0F C3 07
        let mem = MemoryOperand {
            base: Some(Register::Rdi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode(
                "movnti",
                vec![
                    Operand::Memory(Box::new(mem)),
                    Operand::Register(Register::Eax)
                ]
            ),
            vec![0x0F, 0xC3, 0x07]
        );
    }

    #[test]
    fn movnti_mem_rax() {
        // movnti [rdi], rax → REX.W 0F C3 07
        let mem = MemoryOperand {
            base: Some(Register::Rdi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode(
                "movnti",
                vec![
                    Operand::Memory(Box::new(mem)),
                    Operand::Register(Register::Rax)
                ]
            ),
            vec![0x48, 0x0F, 0xC3, 0x07]
        );
    }

    // ─── MOVBE ────────────────────────────────────────────────────────────

    #[test]
    fn movbe_load_eax() {
        // movbe eax, [rdi] → 0F 38 F0 07
        let mem = MemoryOperand {
            base: Some(Register::Rdi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode(
                "movbe",
                vec![
                    Operand::Register(Register::Eax),
                    Operand::Memory(Box::new(mem))
                ]
            ),
            vec![0x0F, 0x38, 0xF0, 0x07]
        );
    }

    #[test]
    fn movbe_store_eax() {
        // movbe [rdi], eax → 0F 38 F1 07
        let mem = MemoryOperand {
            base: Some(Register::Rdi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode(
                "movbe",
                vec![
                    Operand::Memory(Box::new(mem)),
                    Operand::Register(Register::Eax)
                ]
            ),
            vec![0x0F, 0x38, 0xF1, 0x07]
        );
    }

    #[test]
    fn movbe_load_rax() {
        // movbe rax, [rdi] → REX.W 0F 38 F0 07
        let mem = MemoryOperand {
            base: Some(Register::Rdi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode(
                "movbe",
                vec![
                    Operand::Register(Register::Rax),
                    Operand::Memory(Box::new(mem))
                ]
            ),
            vec![0x48, 0x0F, 0x38, 0xF0, 0x07]
        );
    }

    // ─── SHLD / SHRD ─────────────────────────────────────────────────────

    #[test]
    fn shld_reg_reg_imm() {
        // shld eax, ecx, 4 → 0F A4 ModR/M(11, ecx=1, eax=0) 04
        assert_eq!(
            encode(
                "shld",
                vec![
                    Operand::Register(Register::Eax),
                    Operand::Register(Register::Ecx),
                    Operand::Immediate(4)
                ]
            ),
            vec![0x0F, 0xA4, 0xC8, 0x04]
        );
    }

    #[test]
    fn shld_reg_reg_cl() {
        // shld eax, ecx, cl → 0F A5 ModR/M(11, ecx=1, eax=0)
        assert_eq!(
            encode(
                "shld",
                vec![
                    Operand::Register(Register::Eax),
                    Operand::Register(Register::Ecx),
                    Operand::Register(Register::Cl)
                ]
            ),
            vec![0x0F, 0xA5, 0xC8]
        );
    }

    #[test]
    fn shrd_reg_reg_imm_64() {
        // shrd rax, rcx, 8 → REX.W 0F AC ModR/M(11, rcx=1, rax=0) 08
        assert_eq!(
            encode(
                "shrd",
                vec![
                    Operand::Register(Register::Rax),
                    Operand::Register(Register::Rcx),
                    Operand::Immediate(8)
                ]
            ),
            vec![0x48, 0x0F, 0xAC, 0xC8, 0x08]
        );
    }

    #[test]
    fn shld_mem_reg_imm() {
        // shld [rdi], ecx, 4 → 0F A4 ModR/M(00, ecx=1, rdi=7) 04
        let mem = MemoryOperand {
            base: Some(Register::Rdi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode(
                "shld",
                vec![
                    Operand::Memory(Box::new(mem)),
                    Operand::Register(Register::Ecx),
                    Operand::Immediate(4)
                ]
            ),
            vec![0x0F, 0xA4, 0x0F, 0x04]
        );
    }

    #[test]
    fn shrd_mem_reg_cl() {
        // shrd [rsi], edx, cl → 0F AD ModR/M(00, edx=2, rsi=6)
        let mem = MemoryOperand {
            base: Some(Register::Rsi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode(
                "shrd",
                vec![
                    Operand::Memory(Box::new(mem)),
                    Operand::Register(Register::Edx),
                    Operand::Register(Register::Cl)
                ]
            ),
            vec![0x0F, 0xAD, 0x16]
        );
    }

    // ─── JECXZ / JRCXZ ───────────────────────────────────────────────────

    #[test]
    fn jecxz_long_form_encoding() {
        // JECXZ now emits long form: 67 E3 02 EB 05 E9 [rel32]
        let instr = Instruction {
            mnemonic: Mnemonic::from("jecxz"),
            operands: vec![Operand::Label(String::from("target"))].into(),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        };
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // Long form: 67 E3 02 EB 05 E9 00 00 00 00 (10 bytes)
        assert_eq!(result.bytes.len(), 10);
        assert_eq!(result.bytes[0], 0x67); // address-size override
        assert_eq!(result.bytes[1], 0xE3); // JECXZ opcode
        assert_eq!(result.bytes[2], 0x02); // rel8 = +2 (skip JMP short)
        assert_eq!(result.bytes[3], 0xEB); // JMP short
        assert_eq!(result.bytes[4], 0x05); // rel8 = +5 (skip JMP near)
        assert_eq!(result.bytes[5], 0xE9); // JMP near
                                           // rel32 placeholder at offset 6
        assert!(result.relocation.is_some());
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "target");
        assert_eq!(reloc.size, 4);
        assert_eq!(reloc.offset, 6);
    }

    #[test]
    fn jecxz_has_relax_info() {
        let instr = Instruction {
            mnemonic: Mnemonic::from("jecxz"),
            operands: vec![Operand::Label(String::from("target"))].into(),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        };
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // Short form via relaxation: 67 E3 rel8 (3 bytes)
        assert!(result.relax.is_some());
        let ri = result.relax.unwrap();
        assert_eq!(ri.short_bytes.len(), 3);
        assert_eq!(ri.short_bytes[0], 0x67);
        assert_eq!(ri.short_bytes[1], 0xE3);
        assert_eq!(ri.short_reloc_offset, 2);
    }

    #[test]
    fn jrcxz_long_form_encoding() {
        // JRCXZ now emits long form: E3 02 EB 05 E9 [rel32]
        let instr = Instruction {
            mnemonic: Mnemonic::from("jrcxz"),
            operands: vec![Operand::Label(String::from("loop_start"))].into(),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        };
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // Long form: E3 02 EB 05 E9 00 00 00 00 (9 bytes)
        assert_eq!(result.bytes.len(), 9);
        assert_eq!(result.bytes[0], 0xE3); // JRCXZ opcode
        assert_eq!(result.bytes[1], 0x02); // rel8 = +2
        assert_eq!(result.bytes[2], 0xEB); // JMP short
        assert_eq!(result.bytes[3], 0x05); // rel8 = +5
        assert_eq!(result.bytes[4], 0xE9); // JMP near
        assert!(result.relocation.is_some());
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "loop_start");
        assert_eq!(reloc.size, 4);
        assert_eq!(reloc.offset, 5);
    }

    #[test]
    fn jrcxz_has_relax_info() {
        let instr = Instruction {
            mnemonic: Mnemonic::from("jrcxz"),
            operands: vec![Operand::Label(String::from("loop_start"))].into(),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        };
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // Short form via relaxation: E3 rel8 (2 bytes)
        assert!(result.relax.is_some());
        let ri = result.relax.unwrap();
        assert_eq!(ri.short_bytes.len(), 2);
        assert_eq!(ri.short_bytes[0], 0xE3);
        assert_eq!(ri.short_reloc_offset, 1);
    }

    // ─── LOOP relaxation ─────────────────────────────────────────────────

    #[test]
    fn loop_long_form_encoding() {
        let instr = Instruction {
            mnemonic: Mnemonic::from("loop"),
            operands: vec![Operand::Label(String::from("top"))].into(),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        };
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // Long form: E2 02 EB 05 E9 [rel32] (9 bytes)
        assert_eq!(result.bytes.len(), 9);
        assert_eq!(result.bytes[0], 0xE2); // LOOP opcode
        assert_eq!(result.bytes[1], 0x02);
        assert_eq!(result.bytes[2], 0xEB);
        assert_eq!(result.bytes[3], 0x05);
        assert_eq!(result.bytes[4], 0xE9);
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "top");
        assert_eq!(reloc.size, 4);
        assert_eq!(reloc.offset, 5);
    }

    #[test]
    fn loop_has_relax_info() {
        let instr = Instruction {
            mnemonic: Mnemonic::from("loop"),
            operands: vec![Operand::Label(String::from("top"))].into(),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        };
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert!(result.relax.is_some());
        let ri = result.relax.unwrap();
        assert_eq!(ri.short_bytes.len(), 2);
        assert_eq!(ri.short_bytes[0], 0xE2);
        assert_eq!(ri.short_reloc_offset, 1);
    }

    #[test]
    fn loopne_long_form_encoding() {
        let instr = Instruction {
            mnemonic: Mnemonic::from("loopne"),
            operands: vec![Operand::Label(String::from("scan"))].into(),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        };
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // Long form: E0 02 EB 05 E9 [rel32] (9 bytes)
        assert_eq!(result.bytes[0], 0xE0);
        assert_eq!(result.bytes.len(), 9);
        assert!(result.relax.is_some());
        let ri = result.relax.unwrap();
        assert_eq!(ri.short_bytes[0], 0xE0);
    }

    #[test]
    fn loope_long_form_encoding() {
        let instr = Instruction {
            mnemonic: Mnemonic::from("loope"),
            operands: vec![Operand::Label(String::from("repeat"))].into(),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        };
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // Long form: E1 02 EB 05 E9 [rel32] (9 bytes)
        assert_eq!(result.bytes[0], 0xE1);
        assert_eq!(result.bytes.len(), 9);
        assert!(result.relax.is_some());
        let ri = result.relax.unwrap();
        assert_eq!(ri.short_bytes[0], 0xE1);
    }

    // ─── Prefix integration ──────────────────────────────────────────────

    #[test]
    fn rep_insb() {
        assert_eq!(
            encode_with_prefix("insb", vec![], Prefix::Rep),
            vec![0xF3, 0x6C]
        );
    }

    #[test]
    fn rep_outsb() {
        assert_eq!(
            encode_with_prefix("outsb", vec![], Prefix::Rep),
            vec![0xF3, 0x6E]
        );
    }

    #[test]
    fn lock_xadd_mem_reg() {
        // lock xadd [rdi], eax → F0 0F C1 07
        let mem = MemoryOperand {
            base: Some(Register::Rdi),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        assert_eq!(
            encode_with_prefix(
                "xadd",
                vec![
                    Operand::Memory(Box::new(mem)),
                    Operand::Register(Register::Eax)
                ],
                Prefix::Lock,
            ),
            vec![0xF0, 0x0F, 0xC1, 0x07]
        );
    }

    #[test]
    fn test_shld_rejects_8bit() {
        let instr = Instruction {
            mnemonic: Mnemonic::from("shld"),
            operands: vec![
                Operand::Register(Register::Al),
                Operand::Register(Register::Cl),
                Operand::Immediate(1),
            ]
            .into(),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: crate::error::Span::new(1, 1, 0, 0),
        };
        assert!(crate::encoder::encode_instruction(&instr, crate::Arch::X86_64).is_err());
    }

    #[test]
    fn test_shrd_rejects_8bit() {
        let instr = Instruction {
            mnemonic: Mnemonic::from("shrd"),
            operands: vec![
                Operand::Register(Register::Al),
                Operand::Register(Register::Cl),
                Operand::Immediate(1),
            ]
            .into(),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: crate::error::Span::new(1, 1, 0, 0),
        };
        assert!(crate::encoder::encode_instruction(&instr, crate::Arch::X86_64).is_err());
    }

    // ─── P0: Table-dispatched label relocation propagation ──────────────

    fn make_rip_mem_label(label: &str) -> MemoryOperand {
        MemoryOperand {
            size: None,
            base: Some(Register::Rip),
            index: None,
            scale: 1,
            disp: 0,
            disp_label: Some(String::from(label)),
            addr_mode: AddrMode::Offset,
            segment: None,
            index_subtract: false,
        }
    }

    fn encode_with_reloc(
        mnemonic: &str,
        operands: Vec<Operand>,
    ) -> (Vec<u8>, Option<crate::encoder::Relocation>) {
        let instr = make_instr(mnemonic, operands);
        let encoded = crate::encoder::encode_instruction(&instr, crate::Arch::X86_64).unwrap();
        (encoded.bytes.to_vec(), encoded.relocation)
    }

    #[test]
    fn cmpxchg_rip_mem_produces_relocation() {
        let (_, reloc) = encode_with_reloc(
            "cmpxchg",
            vec![
                Operand::Memory(Box::new(make_rip_mem_label("lock_var"))),
                Operand::Register(Register::Rax),
            ],
        );
        let r = reloc.expect("cmpxchg [rip+label] must produce relocation");
        assert_eq!(r.kind, RelocKind::X86Relative);
        assert_eq!(&*r.label, "lock_var");
        assert_eq!(r.size, 4);
    }

    #[test]
    fn xadd_rip_mem_produces_relocation() {
        let (_, reloc) = encode_with_reloc(
            "xadd",
            vec![
                Operand::Memory(Box::new(make_rip_mem_label("counter"))),
                Operand::Register(Register::Ecx),
            ],
        );
        let r = reloc.expect("xadd [rip+label] must produce relocation");
        assert_eq!(r.kind, RelocKind::X86Relative);
        assert_eq!(&*r.label, "counter");
    }

    #[test]
    fn movnti_rip_mem_produces_relocation() {
        let (_, reloc) = encode_with_reloc(
            "movnti",
            vec![
                Operand::Memory(Box::new(make_rip_mem_label("data"))),
                Operand::Register(Register::Eax),
            ],
        );
        let r = reloc.expect("movnti [rip+label] must produce relocation");
        assert_eq!(r.kind, RelocKind::X86Relative);
        assert_eq!(&*r.label, "data");
    }

    #[test]
    fn movbe_load_rip_mem_produces_relocation() {
        let (_, reloc) = encode_with_reloc(
            "movbe",
            vec![
                Operand::Register(Register::Eax),
                Operand::Memory(Box::new(make_rip_mem_label("be_data"))),
            ],
        );
        let r = reloc.expect("movbe reg, [rip+label] must produce relocation");
        assert_eq!(r.kind, RelocKind::X86Relative);
        assert_eq!(&*r.label, "be_data");
    }

    #[test]
    fn movbe_store_rip_mem_produces_relocation() {
        let (_, reloc) = encode_with_reloc(
            "movbe",
            vec![
                Operand::Memory(Box::new(make_rip_mem_label("be_dest"))),
                Operand::Register(Register::Eax),
            ],
        );
        let r = reloc.expect("movbe [rip+label], reg must produce relocation");
        assert_eq!(r.kind, RelocKind::X86Relative);
        assert_eq!(&*r.label, "be_dest");
    }

    #[test]
    fn cmpxchg8b_rip_mem_produces_relocation() {
        let (_, reloc) = encode_with_reloc(
            "cmpxchg8b",
            vec![Operand::Memory(Box::new(make_rip_mem_label("atomic_var")))],
        );
        let r = reloc.expect("cmpxchg8b [rip+label] must produce relocation");
        assert_eq!(r.kind, RelocKind::X86Relative);
        assert_eq!(&*r.label, "atomic_var");
    }

    #[test]
    fn shld_mem_rip_produces_relocation() {
        let (_, reloc) = encode_with_reloc(
            "shld",
            vec![
                Operand::Memory(Box::new(make_rip_mem_label("shift_dest"))),
                Operand::Register(Register::Ecx),
                Operand::Immediate(4),
            ],
        );
        let r = reloc.expect("shld [rip+label], reg, imm must produce relocation");
        assert_eq!(r.kind, RelocKind::X86Relative);
        assert_eq!(&*r.label, "shift_dest");
    }

    #[test]
    fn shrd_mem_cl_rip_produces_relocation() {
        let (_, reloc) = encode_with_reloc(
            "shrd",
            vec![
                Operand::Memory(Box::new(make_rip_mem_label("shift_dest"))),
                Operand::Register(Register::Ecx),
                Operand::Register(Register::Cl),
            ],
        );
        let r = reloc.expect("shrd [rip+label], reg, cl must produce relocation");
        assert_eq!(r.kind, RelocKind::X86Relative);
        assert_eq!(&*r.label, "shift_dest");
    }

    // ─── P1: High-byte REX conflict in CMPXCHG/XADD ────────────────────

    #[test]
    fn cmpxchg_ah_spl_rejects_high_byte_rex_conflict() {
        assert!(encode_err(
            "cmpxchg",
            vec![
                Operand::Register(Register::Ah),
                Operand::Register(Register::Spl),
            ]
        ));
    }

    #[test]
    fn xadd_ah_r8b_rejects_high_byte_rex_conflict() {
        assert!(encode_err(
            "xadd",
            vec![
                Operand::Register(Register::Ah),
                Operand::Register(Register::R8b),
            ]
        ));
    }

    // ─── P1: Immediate overflow checks ──────────────────────────────────

    #[test]
    fn in_al_overflow_rejects() {
        assert!(encode_err(
            "in",
            vec![Operand::Register(Register::Al), Operand::Immediate(256),]
        ));
        assert!(encode_err(
            "in",
            vec![Operand::Register(Register::Al), Operand::Immediate(-1),]
        ));
    }

    #[test]
    fn out_overflow_rejects() {
        assert!(encode_err(
            "out",
            vec![Operand::Immediate(256), Operand::Register(Register::Al),]
        ));
    }

    #[test]
    fn enter_overflow_rejects() {
        // frame_size overflow
        assert!(encode_err(
            "enter",
            vec![Operand::Immediate(0x10000), Operand::Immediate(0)]
        ));
        // nesting level overflow
        assert!(encode_err(
            "enter",
            vec![Operand::Immediate(0), Operand::Immediate(256)]
        ));
    }

    #[test]
    fn shld_imm_overflow_rejects() {
        assert!(encode_err(
            "shld",
            vec![
                Operand::Register(Register::Eax),
                Operand::Register(Register::Ecx),
                Operand::Immediate(256),
            ]
        ));
    }

    #[test]
    fn shrd_imm_overflow_rejects() {
        assert!(encode_err(
            "shrd",
            vec![
                Operand::Register(Register::Eax),
                Operand::Register(Register::Ecx),
                Operand::Immediate(-1),
            ]
        ));
    }

    // ─── P1: SHLD/SHRD reject 8-bit register operand ────────────────────

    #[test]
    fn shld_rejects_8bit_reg_source() {
        assert!(encode_err(
            "shld",
            vec![
                Operand::Register(Register::Eax),
                Operand::Register(Register::Cl), // 8-bit second operand
                Operand::Immediate(4),
            ]
        ));
    }

    #[test]
    fn shrd_rejects_8bit_reg_source() {
        assert!(encode_err(
            "shrd",
            vec![
                Operand::Register(Register::Eax),
                Operand::Register(Register::Al), // 8-bit second operand
                Operand::Immediate(4),
            ]
        ));
    }

    // ── ADCX / ADOX ──────────────────────────────────────────

    #[test]
    fn adcx_eax_ebx() {
        // 66 0F 38 F6 C3
        assert_eq!(
            encode(
                "adcx",
                vec![
                    Operand::Register(Register::Eax),
                    Operand::Register(Register::Ebx),
                ]
            ),
            vec![0x66, 0x0F, 0x38, 0xF6, 0xC3]
        );
    }

    #[test]
    fn adcx_rax_rbx() {
        // 66 48 0F 38 F6 C3
        assert_eq!(
            encode(
                "adcx",
                vec![
                    Operand::Register(Register::Rax),
                    Operand::Register(Register::Rbx),
                ]
            ),
            vec![0x66, 0x48, 0x0F, 0x38, 0xF6, 0xC3]
        );
    }

    #[test]
    fn adcx_r8d_ecx() {
        // 66 44 0F 38 F6 C1  (REX.R for r8d)
        assert_eq!(
            encode(
                "adcx",
                vec![
                    Operand::Register(Register::R8d),
                    Operand::Register(Register::Ecx),
                ]
            ),
            vec![0x66, 0x44, 0x0F, 0x38, 0xF6, 0xC1]
        );
    }

    #[test]
    fn adcx_rax_r15() {
        // 66 49 0F 38 F6 C7  (REX.W+B for r15)
        assert_eq!(
            encode(
                "adcx",
                vec![
                    Operand::Register(Register::Rax),
                    Operand::Register(Register::R15),
                ]
            ),
            vec![0x66, 0x49, 0x0F, 0x38, 0xF6, 0xC7]
        );
    }

    #[test]
    fn adox_eax_ebx() {
        // F3 0F 38 F6 C3
        assert_eq!(
            encode(
                "adox",
                vec![
                    Operand::Register(Register::Eax),
                    Operand::Register(Register::Ebx),
                ]
            ),
            vec![0xF3, 0x0F, 0x38, 0xF6, 0xC3]
        );
    }

    #[test]
    fn adox_rax_rbx() {
        // F3 48 0F 38 F6 C3
        assert_eq!(
            encode(
                "adox",
                vec![
                    Operand::Register(Register::Rax),
                    Operand::Register(Register::Rbx),
                ]
            ),
            vec![0xF3, 0x48, 0x0F, 0x38, 0xF6, 0xC3]
        );
    }

    #[test]
    fn adox_r9_r10() {
        // F3 4D 0F 38 F6 CA  (REX.WRB)
        assert_eq!(
            encode(
                "adox",
                vec![
                    Operand::Register(Register::R9),
                    Operand::Register(Register::R10),
                ]
            ),
            vec![0xF3, 0x4D, 0x0F, 0x38, 0xF6, 0xCA]
        );
    }

    #[test]
    fn adcx_rejects_16bit() {
        assert!(encode_err(
            "adcx",
            vec![
                Operand::Register(Register::Ax),
                Operand::Register(Register::Bx),
            ]
        ));
    }

    #[test]
    fn adox_rejects_8bit() {
        assert!(encode_err(
            "adox",
            vec![
                Operand::Register(Register::Al),
                Operand::Register(Register::Bl),
            ]
        ));
    }

    // ─── FMA3 unit tests ────────────────────────────────────────────

    #[test]
    fn fma_vfmadd132ps() {
        assert_eq!(
            encode(
                "vfmadd132ps",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0x98, 0xC2]
        );
    }

    #[test]
    fn fma_vfmadd132pd() {
        assert_eq!(
            encode(
                "vfmadd132pd",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0xF1, 0x98, 0xC2]
        );
    }

    #[test]
    fn fma_vfmadd213ps() {
        assert_eq!(
            encode(
                "vfmadd213ps",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0xA8, 0xC2]
        );
    }

    #[test]
    fn fma_vfmadd231ps() {
        assert_eq!(
            encode(
                "vfmadd231ps",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0xB8, 0xC2]
        );
    }

    #[test]
    fn fma_vfmadd231pd() {
        assert_eq!(
            encode(
                "vfmadd231pd",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0xF1, 0xB8, 0xC2]
        );
    }

    #[test]
    fn fma_vfmadd231ps_ymm() {
        assert_eq!(
            encode(
                "vfmadd231ps",
                vec![r(Register::Ymm0), r(Register::Ymm1), r(Register::Ymm2)]
            ),
            [0xC4, 0xE2, 0x75, 0xB8, 0xC2]
        );
    }

    #[test]
    fn fma_vfmsub231ps() {
        assert_eq!(
            encode(
                "vfmsub231ps",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0xBA, 0xC2]
        );
    }

    #[test]
    fn fma_vfmsub231pd() {
        assert_eq!(
            encode(
                "vfmsub231pd",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0xF1, 0xBA, 0xC2]
        );
    }

    #[test]
    fn fma_vfnmadd231ps() {
        assert_eq!(
            encode(
                "vfnmadd231ps",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0xBC, 0xC2]
        );
    }

    #[test]
    fn fma_vfnmadd231pd() {
        assert_eq!(
            encode(
                "vfnmadd231pd",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0xF1, 0xBC, 0xC2]
        );
    }

    #[test]
    fn fma_vfnmsub231ps() {
        assert_eq!(
            encode(
                "vfnmsub231ps",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0xBE, 0xC2]
        );
    }

    #[test]
    fn fma_vfnmsub231pd() {
        assert_eq!(
            encode(
                "vfnmsub231pd",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0xF1, 0xBE, 0xC2]
        );
    }

    #[test]
    fn fma_vfmaddsub132ps() {
        assert_eq!(
            encode(
                "vfmaddsub132ps",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0x96, 0xC2]
        );
    }

    #[test]
    fn fma_vfmsubadd213ps() {
        assert_eq!(
            encode(
                "vfmsubadd213ps",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0xA7, 0xC2]
        );
    }

    #[test]
    fn fma_vfmadd231ss() {
        assert_eq!(
            encode(
                "vfmadd231ss",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0xB9, 0xC2]
        );
    }

    #[test]
    fn fma_vfmadd231sd() {
        assert_eq!(
            encode(
                "vfmadd231sd",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0xF1, 0xB9, 0xC2]
        );
    }

    // ─── VEX shift unit tests ───────────────────────────────────────

    #[test]
    fn vex_vpslld_reg() {
        assert_eq!(
            encode(
                "vpslld",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC5, 0xF1, 0xF2, 0xC2]
        );
    }

    #[test]
    fn vex_vpslld_imm() {
        assert_eq!(
            encode(
                "vpslld",
                vec![r(Register::Xmm0), r(Register::Xmm1), Operand::Immediate(4)]
            ),
            [0xC5, 0xF9, 0x72, 0xF1, 0x04]
        );
    }

    #[test]
    fn vex_vpslld_ymm_imm() {
        assert_eq!(
            encode(
                "vpslld",
                vec![r(Register::Ymm0), r(Register::Ymm1), Operand::Immediate(4)]
            ),
            [0xC5, 0xFD, 0x72, 0xF1, 0x04]
        );
    }

    #[test]
    fn vex_vpsllw_reg() {
        assert_eq!(
            encode(
                "vpsllw",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC5, 0xF1, 0xF1, 0xC2]
        );
    }

    #[test]
    fn vex_vpsllq_reg() {
        assert_eq!(
            encode(
                "vpsllq",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC5, 0xF1, 0xF3, 0xC2]
        );
    }

    #[test]
    fn vex_vpsrlw_imm() {
        assert_eq!(
            encode(
                "vpsrlw",
                vec![r(Register::Xmm2), r(Register::Xmm3), Operand::Immediate(8)]
            ),
            [0xC5, 0xE9, 0x71, 0xD3, 0x08]
        );
    }

    #[test]
    fn vex_vpsrld_reg() {
        assert_eq!(
            encode(
                "vpsrld",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC5, 0xF1, 0xD2, 0xC2]
        );
    }

    #[test]
    fn vex_vpsrlq_reg() {
        assert_eq!(
            encode(
                "vpsrlq",
                vec![r(Register::Xmm3), r(Register::Xmm4), r(Register::Xmm5)]
            ),
            [0xC5, 0xD9, 0xD3, 0xDD]
        );
    }

    #[test]
    fn vex_vpsraw_imm() {
        assert_eq!(
            encode(
                "vpsraw",
                vec![r(Register::Xmm2), r(Register::Xmm3), Operand::Immediate(8)]
            ),
            [0xC5, 0xE9, 0x71, 0xE3, 0x08]
        );
    }

    #[test]
    fn vex_vpsrad_reg() {
        assert_eq!(
            encode(
                "vpsrad",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC5, 0xF1, 0xE2, 0xC2]
        );
    }

    // ─── AVX2 variable shift unit tests ─────────────────────────────

    #[test]
    fn vex_vpsllvd() {
        assert_eq!(
            encode(
                "vpsllvd",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0x47, 0xC2]
        );
    }

    #[test]
    fn vex_vpsllvq() {
        assert_eq!(
            encode(
                "vpsllvq",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0xF1, 0x47, 0xC2]
        );
    }

    #[test]
    fn vex_vpsrlvd() {
        assert_eq!(
            encode(
                "vpsrlvd",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0x45, 0xC2]
        );
    }

    #[test]
    fn vex_vpsravd() {
        assert_eq!(
            encode(
                "vpsravd",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0x46, 0xC2]
        );
    }

    // ─── AVX permute / broadcast unit tests ─────────────────────────

    #[test]
    fn vex_vpermilps_reg() {
        assert_eq!(
            encode(
                "vpermilps",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0x0C, 0xC2]
        );
    }

    #[test]
    fn vex_vpermilps_imm() {
        assert_eq!(
            encode(
                "vpermilps",
                vec![
                    r(Register::Xmm0),
                    r(Register::Xmm1),
                    Operand::Immediate(0x44)
                ]
            ),
            [0xC4, 0xE3, 0x79, 0x04, 0xC1, 0x44]
        );
    }

    #[test]
    fn vex_vpermilpd_reg() {
        assert_eq!(
            encode(
                "vpermilpd",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Xmm2)]
            ),
            [0xC4, 0xE2, 0x71, 0x0D, 0xC2]
        );
    }

    #[test]
    fn vex_vbroadcastss() {
        assert_eq!(
            encode("vbroadcastss", vec![r(Register::Xmm0), r(Register::Xmm1)]),
            [0xC4, 0xE2, 0x79, 0x18, 0xC1]
        );
    }

    #[test]
    fn vex_vpermq() {
        assert_eq!(
            encode(
                "vpermq",
                vec![
                    r(Register::Ymm0),
                    r(Register::Ymm1),
                    Operand::Immediate(0x44)
                ]
            ),
            [0xC4, 0xE3, 0xFD, 0x00, 0xC1, 0x44]
        );
    }

    #[test]
    fn vex_vpermpd() {
        assert_eq!(
            encode(
                "vpermpd",
                vec![
                    r(Register::Ymm0),
                    r(Register::Ymm1),
                    Operand::Immediate(0xD8)
                ]
            ),
            [0xC4, 0xE3, 0xFD, 0x01, 0xC1, 0xD8]
        );
    }

    #[test]
    fn vex_vpbroadcastd() {
        assert_eq!(
            encode("vpbroadcastd", vec![r(Register::Xmm0), r(Register::Xmm1)]),
            [0xC4, 0xE2, 0x79, 0x58, 0xC1]
        );
    }

    #[test]
    fn vex_vtestps() {
        assert_eq!(
            encode("vtestps", vec![r(Register::Xmm0), r(Register::Xmm1)]),
            [0xC4, 0xE2, 0x79, 0x0E, 0xC1]
        );
    }

    // ─── AVX conversion unit tests ──────────────────────────────────

    #[test]
    fn vex_vcvtsi2ss() {
        assert_eq!(
            encode(
                "vcvtsi2ss",
                vec![r(Register::Xmm0), r(Register::Xmm1), r(Register::Eax)]
            ),
            [0xC5, 0xF2, 0x2A, 0xC0]
        );
    }

    #[test]
    fn vex_vcvtss2si() {
        assert_eq!(
            encode("vcvtss2si", vec![r(Register::Eax), r(Register::Xmm1)]),
            [0xC5, 0xFA, 0x2D, 0xC1]
        );
    }

    #[test]
    fn vex_vcvtdq2ps() {
        assert_eq!(
            encode("vcvtdq2ps", vec![r(Register::Xmm0), r(Register::Xmm1)]),
            [0xC5, 0xF8, 0x5B, 0xC1]
        );
    }

    #[test]
    fn vex_vcvtps2dq() {
        assert_eq!(
            encode("vcvtps2dq", vec![r(Register::Xmm0), r(Register::Xmm1)]),
            [0xC5, 0xF9, 0x5B, 0xC1]
        );
    }

    /// Helper to create a register operand.
    fn r(reg: Register) -> Operand {
        Operand::Register(reg)
    }
}
