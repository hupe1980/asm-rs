#![cfg(not(target_arch = "wasm32"))]
//! Multi-architecture cross-validation tests.
//!
//! Encodes with asm-rs, then decodes with independent Rust decoders to verify
//! byte-level correctness across architectures:
//!
//! - **AArch64**: yaxpeax-arm (armv8/a64) + bad64 (Binary Ninja decoder)
//! - **ARM32 (A32)**: yaxpeax-arm (armv7)
//! - **Thumb (T32/T16)**: yaxpeax-arm (armv7 thumb mode)
//! - **RISC-V**: riscv-decode

use asm_rs::{assemble, Arch};

// ============================================================================
// AArch64 cross-validation helpers
// ============================================================================

mod aarch64_xval {
    use super::*;
    use yaxpeax_arch::{Decoder as _, U8Reader};
    use yaxpeax_arm::armv8::a64::{InstDecoder, Opcode};

    /// Assemble with asm-rs, decode with yaxpeax-arm, return opcode + formatted string.
    pub fn asm_and_decode(source: &str) -> (Opcode, String) {
        let bytes = assemble(source, Arch::Aarch64)
            .unwrap_or_else(|e| panic!("asm_rs failed: `{source}`: {e}"));
        assert_eq!(
            bytes.len(),
            4,
            "AArch64 instructions must be 4 bytes: `{source}` → {bytes:02X?}"
        );

        let decoder = InstDecoder::default();
        let mut reader = U8Reader::new(&bytes);
        let inst = decoder.decode(&mut reader).unwrap_or_else(|e| {
            panic!("yaxpeax-arm failed to decode `{source}` → {bytes:02X?}: {e}")
        });
        (inst.opcode, format!("{}", inst))
    }

    /// Verify the yaxpeax-arm opcode matches expected.
    pub fn verify(source: &str, expected: Opcode) {
        let (opcode, formatted) = asm_and_decode(source);
        assert_eq!(
            opcode, expected,
            "opcode mismatch for `{source}`: yaxpeax decoded `{formatted}`"
        );
    }
}

mod aarch64_bad64_xval {
    use super::*;

    /// Assemble with asm-rs, decode with bad64, return op + formatted string.
    pub fn asm_and_decode(source: &str) -> (bad64::Op, String) {
        let bytes = assemble(source, Arch::Aarch64)
            .unwrap_or_else(|e| panic!("asm_rs failed: `{source}`: {e}"));
        assert_eq!(
            bytes.len(),
            4,
            "AArch64 instructions must be 4 bytes: `{source}` → {bytes:02X?}"
        );

        let word = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        let inst = bad64::decode(word, 0)
            .unwrap_or_else(|e| panic!("bad64 failed to decode `{source}` → {bytes:02X?}: {e}"));
        (inst.op(), format!("{}", inst))
    }

    /// Verify the bad64 Op matches expected.
    pub fn verify(source: &str, expected: bad64::Op) {
        let (op, formatted) = asm_and_decode(source);
        assert_eq!(
            op, expected,
            "op mismatch for `{source}`: bad64 decoded `{formatted}`"
        );
    }
}

// ============================================================================
// ARM32 (A32) cross-validation helpers
// ============================================================================

mod arm32_xval {
    use super::*;
    use yaxpeax_arch::{Decoder as _, U8Reader};
    use yaxpeax_arm::armv7::{InstDecoder, Opcode};

    /// Assemble with asm-rs, decode with yaxpeax-arm (ARMv7), return opcode + formatted string.
    pub fn asm_and_decode(source: &str) -> (Opcode, String) {
        let bytes = assemble(source, Arch::Arm)
            .unwrap_or_else(|e| panic!("asm_rs failed: `{source}`: {e}"));
        assert_eq!(
            bytes.len(),
            4,
            "ARM32 instructions must be 4 bytes: `{source}` → {bytes:02X?}"
        );

        let decoder = InstDecoder::default(); // ARM mode (not thumb)
        let mut reader = U8Reader::new(&bytes);
        let inst = decoder.decode(&mut reader).unwrap_or_else(|e| {
            panic!("yaxpeax-arm(v7) failed to decode `{source}` → {bytes:02X?}: {e}")
        });
        (inst.opcode, format!("{}", inst))
    }

    /// Verify the yaxpeax-arm (ARMv7) opcode matches expected.
    pub fn verify(source: &str, expected: Opcode) {
        let (opcode, formatted) = asm_and_decode(source);
        assert_eq!(
            opcode, expected,
            "opcode mismatch for `{source}`: yaxpeax decoded `{formatted}`"
        );
    }
}

// ============================================================================
// Thumb cross-validation helpers
// ============================================================================

mod thumb_xval {
    use super::*;
    use yaxpeax_arch::{Decoder as _, U8Reader};
    use yaxpeax_arm::armv7::{InstDecoder, Opcode};

    /// Assemble with asm-rs, decode with yaxpeax-arm (Thumb mode), return opcode + formatted string.
    pub fn asm_and_decode(source: &str) -> (Opcode, String) {
        let bytes = assemble(source, Arch::Thumb)
            .unwrap_or_else(|e| panic!("asm_rs failed: `{source}`: {e}"));
        assert!(!bytes.is_empty(), "empty output for `{source}`");
        assert!(
            bytes.len() == 2 || bytes.len() == 4,
            "Thumb instructions must be 2 or 4 bytes: `{source}` → {bytes:02X?}"
        );

        let decoder = InstDecoder::default_thumb();
        let mut reader = U8Reader::new(&bytes);
        let inst = decoder.decode(&mut reader).unwrap_or_else(|e| {
            panic!("yaxpeax-arm(thumb) failed to decode `{source}` → {bytes:02X?}: {e}")
        });
        (inst.opcode, format!("{}", inst))
    }

    /// Verify the yaxpeax-arm (Thumb) opcode matches expected.
    pub fn verify(source: &str, expected: Opcode) {
        let (opcode, formatted) = asm_and_decode(source);
        assert_eq!(
            opcode, expected,
            "opcode mismatch for `{source}`: yaxpeax decoded `{formatted}`"
        );
    }
}

// ============================================================================
// RISC-V cross-validation helpers
// ============================================================================

mod riscv_xval {
    use super::*;

    /// Assemble with asm-rs, decode with riscv-decode, return Instruction variant.
    pub fn asm_and_decode(source: &str, arch: Arch) -> riscv_decode::Instruction {
        let bytes =
            assemble(source, arch).unwrap_or_else(|e| panic!("asm_rs failed: `{source}`: {e}"));
        assert_eq!(
            bytes.len(),
            4,
            "RISC-V standard instructions must be 4 bytes: `{source}` → {bytes:02X?}"
        );

        let word = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        riscv_decode::decode(word)
            .unwrap_or_else(|e| panic!("riscv-decode failed: `{source}` → {bytes:02X?}: {e:?}"))
    }

    /// Assert instruction matches expected variant using discriminant comparison.
    pub fn verify_variant(source: &str, arch: Arch, expected: &str) {
        let inst = asm_and_decode(source, arch);
        let debug = format!("{:?}", inst);
        // Extract variant name from debug output: e.g. "Add(RType(00000033))" → "Add"
        let variant = debug.split('(').next().unwrap_or(&debug);
        assert_eq!(
            variant, expected,
            "variant mismatch for `{source}`: riscv-decode decoded `{debug}`"
        );
    }
}

// ============================================================================
// AArch64 — yaxpeax-arm cross-validation tests
// ============================================================================

use yaxpeax_arm::armv8::a64::Opcode as A64Opcode;

#[test]
fn aarch64_yax_add_x0_x1_x2() {
    aarch64_xval::verify("add x0, x1, x2", A64Opcode::ADD);
}

#[test]
fn aarch64_yax_sub_x0_x1_x2() {
    aarch64_xval::verify("sub x0, x1, x2", A64Opcode::SUB);
}

#[test]
fn aarch64_yax_adds_x0_x1_x2() {
    aarch64_xval::verify("adds x0, x1, x2", A64Opcode::ADDS);
}

#[test]
fn aarch64_yax_subs_x0_x1_x2() {
    aarch64_xval::verify("subs x0, x1, x2", A64Opcode::SUBS);
}

#[test]
fn aarch64_yax_and_x0_x1_x2() {
    aarch64_xval::verify("and x0, x1, x2", A64Opcode::AND);
}

#[test]
fn aarch64_yax_orr_x0_x1_x2() {
    aarch64_xval::verify("orr x0, x1, x2", A64Opcode::ORR);
}

#[test]
fn aarch64_yax_eor_x0_x1_x2() {
    aarch64_xval::verify("eor x0, x1, x2", A64Opcode::EOR);
}

#[test]
fn aarch64_yax_ands_x0_x1_x2() {
    aarch64_xval::verify("ands x0, x1, x2", A64Opcode::ANDS);
}

#[test]
fn aarch64_yax_mov_x0_x1() {
    // MOV Xd, Xn is an alias for ORR Xd, XZR, Xn — yaxpeax decodes as ORR
    let (opcode, _) = aarch64_xval::asm_and_decode("mov x0, x1");
    assert!(opcode == A64Opcode::ORR, "expected ORR, got {:?}", opcode);
}

#[test]
fn aarch64_yax_add_x0_x1_42() {
    aarch64_xval::verify("add x0, x1, 42", A64Opcode::ADD);
}

#[test]
fn aarch64_yax_sub_x0_x1_42() {
    aarch64_xval::verify("sub x0, x1, 42", A64Opcode::SUB);
}

#[test]
fn aarch64_yax_mul_x0_x1_x2() {
    // MUL is alias for MADD Xd, Xn, Xm, XZR — yaxpeax may decode as MADD
    let (opcode, _) = aarch64_xval::asm_and_decode("mul x0, x1, x2");
    assert!(
        opcode == A64Opcode::MADD || opcode == A64Opcode::MUL,
        "expected MADD or MUL, got {:?}",
        opcode
    );
}

#[test]
fn aarch64_yax_sdiv_x0_x1_x2() {
    aarch64_xval::verify("sdiv x0, x1, x2", A64Opcode::SDIV);
}

#[test]
fn aarch64_yax_udiv_x0_x1_x2() {
    aarch64_xval::verify("udiv x0, x1, x2", A64Opcode::UDIV);
}

#[test]
fn aarch64_yax_ldr_x0_x1() {
    aarch64_xval::verify("ldr x0, [x1]", A64Opcode::LDR);
}

#[test]
fn aarch64_yax_str_x0_x1() {
    aarch64_xval::verify("str x0, [x1]", A64Opcode::STR);
}

#[test]
fn aarch64_yax_ldr_x0_x1_8() {
    aarch64_xval::verify("ldr x0, [x1, 8]", A64Opcode::LDR);
}

#[test]
fn aarch64_yax_str_x0_x1_8() {
    aarch64_xval::verify("str x0, [x1, 8]", A64Opcode::STR);
}

#[test]
fn aarch64_yax_ldp_x0_x1_x2() {
    aarch64_xval::verify("ldp x0, x1, [x2]", A64Opcode::LDP);
}

#[test]
fn aarch64_yax_stp_x0_x1_x2() {
    aarch64_xval::verify("stp x0, x1, [x2]", A64Opcode::STP);
}

#[test]
fn aarch64_yax_ldrb_w0_x1() {
    aarch64_xval::verify("ldrb w0, [x1]", A64Opcode::LDRB);
}

#[test]
fn aarch64_yax_ldrh_w0_x1() {
    aarch64_xval::verify("ldrh w0, [x1]", A64Opcode::LDRH);
}

#[test]
fn aarch64_yax_strb_w0_x1() {
    aarch64_xval::verify("strb w0, [x1]", A64Opcode::STRB);
}

#[test]
fn aarch64_yax_strh_w0_x1() {
    aarch64_xval::verify("strh w0, [x1]", A64Opcode::STRH);
}

#[test]
fn aarch64_yax_ldrsw_x0_x1() {
    aarch64_xval::verify("ldrsw x0, [x1]", A64Opcode::LDRSW);
}

#[test]
fn aarch64_yax_lsl_x0_x1_x2() {
    // LSL Xd, Xn, Xm is alias for LSLV
    let (opcode, _) = aarch64_xval::asm_and_decode("lsl x0, x1, x2");
    assert!(opcode == A64Opcode::LSLV, "expected LSLV, got {:?}", opcode);
}

#[test]
fn aarch64_yax_lsr_x0_x1_x2() {
    let (opcode, _) = aarch64_xval::asm_and_decode("lsr x0, x1, x2");
    assert!(opcode == A64Opcode::LSRV, "expected LSRV, got {:?}", opcode);
}

#[test]
fn aarch64_yax_asr_x0_x1_x2() {
    let (opcode, _) = aarch64_xval::asm_and_decode("asr x0, x1, x2");
    assert!(opcode == A64Opcode::ASRV, "expected ASRV, got {:?}", opcode);
}

#[test]
fn aarch64_yax_madd_x0_x1_x2_x3() {
    aarch64_xval::verify("madd x0, x1, x2, x3", A64Opcode::MADD);
}

#[test]
fn aarch64_yax_msub_x0_x1_x2_x3() {
    aarch64_xval::verify("msub x0, x1, x2, x3", A64Opcode::MSUB);
}

#[test]
fn aarch64_yax_clz_x0_x1() {
    aarch64_xval::verify("clz x0, x1", A64Opcode::CLZ);
}

#[test]
fn aarch64_yax_cls_x0_x1() {
    aarch64_xval::verify("cls x0, x1", A64Opcode::CLS);
}

#[test]
fn aarch64_yax_rbit_x0_x1() {
    aarch64_xval::verify("rbit x0, x1", A64Opcode::RBIT);
}

#[test]
fn aarch64_yax_rev_x0_x1() {
    aarch64_xval::verify("rev x0, x1", A64Opcode::REV);
}

#[test]
fn aarch64_yax_rev16_x0_x1() {
    aarch64_xval::verify("rev16 x0, x1", A64Opcode::REV16);
}

#[test]
fn aarch64_yax_rev32_x0_x1() {
    aarch64_xval::verify("rev32 x0, x1", A64Opcode::REV32);
}

#[test]
fn aarch64_yax_csel_x0_x1_x2_eq() {
    aarch64_xval::verify("csel x0, x1, x2, eq", A64Opcode::CSEL);
}

#[test]
fn aarch64_yax_csinc_x0_x1_x2_ne() {
    aarch64_xval::verify("csinc x0, x1, x2, ne", A64Opcode::CSINC);
}

#[test]
fn aarch64_yax_csinv_x0_x1_x2_lt() {
    aarch64_xval::verify("csinv x0, x1, x2, lt", A64Opcode::CSINV);
}

#[test]
fn aarch64_yax_csneg_x0_x1_x2_ge() {
    aarch64_xval::verify("csneg x0, x1, x2, ge", A64Opcode::CSNEG);
}

#[test]
fn aarch64_yax_svc_0() {
    aarch64_xval::verify("svc 0", A64Opcode::SVC);
}

#[test]
fn aarch64_yax_nop() {
    // NOP/WFE/WFI/SEV are all HINT instructions in yaxpeax
    aarch64_xval::verify("nop", A64Opcode::HINT);
}

#[test]
fn aarch64_yax_wfe() {
    aarch64_xval::verify("wfe", A64Opcode::HINT);
}

#[test]
fn aarch64_yax_wfi() {
    aarch64_xval::verify("wfi", A64Opcode::HINT);
}

#[test]
fn aarch64_yax_sev() {
    aarch64_xval::verify("sev", A64Opcode::HINT);
}

#[test]
fn aarch64_yax_br_x0() {
    aarch64_xval::verify("br x0", A64Opcode::BR);
}

#[test]
fn aarch64_yax_blr_x0() {
    aarch64_xval::verify("blr x0", A64Opcode::BLR);
}

#[test]
fn aarch64_yax_ret() {
    aarch64_xval::verify("ret", A64Opcode::RET);
}

#[test]
fn aarch64_yax_bic_x0_x1_x2() {
    aarch64_xval::verify("bic x0, x1, x2", A64Opcode::BIC);
}

#[test]
fn aarch64_yax_orn_x0_x1_x2() {
    aarch64_xval::verify("orn x0, x1, x2", A64Opcode::ORN);
}

#[test]
fn aarch64_yax_eon_x0_x1_x2() {
    aarch64_xval::verify("eon x0, x1, x2", A64Opcode::EON);
}

// ============================================================================
// AArch64 — bad64 cross-validation tests (second decoder)
// ============================================================================

#[test]
fn aarch64_bad64_add_x0_x1_x2() {
    aarch64_bad64_xval::verify("add x0, x1, x2", bad64::Op::ADD);
}

#[test]
fn aarch64_bad64_sub_x0_x1_x2() {
    aarch64_bad64_xval::verify("sub x0, x1, x2", bad64::Op::SUB);
}

#[test]
fn aarch64_bad64_and_x0_x1_x2() {
    aarch64_bad64_xval::verify("and x0, x1, x2", bad64::Op::AND);
}

#[test]
fn aarch64_bad64_orr_x0_x1_x2() {
    aarch64_bad64_xval::verify("orr x0, x1, x2", bad64::Op::ORR);
}

#[test]
fn aarch64_bad64_eor_x0_x1_x2() {
    aarch64_bad64_xval::verify("eor x0, x1, x2", bad64::Op::EOR);
}

#[test]
fn aarch64_bad64_ldr_x0_x1() {
    aarch64_bad64_xval::verify("ldr x0, [x1]", bad64::Op::LDR);
}

#[test]
fn aarch64_bad64_str_x0_x1() {
    aarch64_bad64_xval::verify("str x0, [x1]", bad64::Op::STR);
}

#[test]
fn aarch64_bad64_ldp_x0_x1_x2() {
    aarch64_bad64_xval::verify("ldp x0, x1, [x2]", bad64::Op::LDP);
}

#[test]
fn aarch64_bad64_stp_x0_x1_x2() {
    aarch64_bad64_xval::verify("stp x0, x1, [x2]", bad64::Op::STP);
}

#[test]
fn aarch64_bad64_nop() {
    aarch64_bad64_xval::verify("nop", bad64::Op::NOP);
}

#[test]
fn aarch64_bad64_svc_0() {
    aarch64_bad64_xval::verify("svc 0", bad64::Op::SVC);
}

#[test]
fn aarch64_bad64_ret() {
    aarch64_bad64_xval::verify("ret", bad64::Op::RET);
}

#[test]
fn aarch64_bad64_br_x0() {
    aarch64_bad64_xval::verify("br x0", bad64::Op::BR);
}

#[test]
fn aarch64_bad64_blr_x0() {
    aarch64_bad64_xval::verify("blr x0", bad64::Op::BLR);
}

#[test]
fn aarch64_bad64_sdiv_x0_x1_x2() {
    aarch64_bad64_xval::verify("sdiv x0, x1, x2", bad64::Op::SDIV);
}

#[test]
fn aarch64_bad64_udiv_x0_x1_x2() {
    aarch64_bad64_xval::verify("udiv x0, x1, x2", bad64::Op::UDIV);
}

#[test]
fn aarch64_bad64_clz_x0_x1() {
    aarch64_bad64_xval::verify("clz x0, x1", bad64::Op::CLZ);
}

#[test]
fn aarch64_bad64_rbit_x0_x1() {
    aarch64_bad64_xval::verify("rbit x0, x1", bad64::Op::RBIT);
}

#[test]
fn aarch64_bad64_rev_x0_x1() {
    aarch64_bad64_xval::verify("rev x0, x1", bad64::Op::REV);
}

#[test]
fn aarch64_bad64_csel_x0_x1_x2_eq() {
    aarch64_bad64_xval::verify("csel x0, x1, x2, eq", bad64::Op::CSEL);
}

#[test]
fn aarch64_bad64_csinc_x0_x1_x2_ne() {
    aarch64_bad64_xval::verify("csinc x0, x1, x2, ne", bad64::Op::CSINC);
}

#[test]
fn aarch64_bad64_adds_x0_x1_x2() {
    aarch64_bad64_xval::verify("adds x0, x1, x2", bad64::Op::ADDS);
}

#[test]
fn aarch64_bad64_subs_x0_x1_x2() {
    aarch64_bad64_xval::verify("subs x0, x1, x2", bad64::Op::SUBS);
}

#[test]
fn aarch64_bad64_ands_x0_x1_x2() {
    aarch64_bad64_xval::verify("ands x0, x1, x2", bad64::Op::ANDS);
}

#[test]
fn aarch64_bad64_ldrb_w0_x1() {
    aarch64_bad64_xval::verify("ldrb w0, [x1]", bad64::Op::LDRB);
}

#[test]
fn aarch64_bad64_ldrh_w0_x1() {
    aarch64_bad64_xval::verify("ldrh w0, [x1]", bad64::Op::LDRH);
}

#[test]
fn aarch64_bad64_strb_w0_x1() {
    aarch64_bad64_xval::verify("strb w0, [x1]", bad64::Op::STRB);
}

#[test]
fn aarch64_bad64_strh_w0_x1() {
    aarch64_bad64_xval::verify("strh w0, [x1]", bad64::Op::STRH);
}

#[test]
fn aarch64_bad64_ldrsw_x0_x1() {
    aarch64_bad64_xval::verify("ldrsw x0, [x1]", bad64::Op::LDRSW);
}

#[test]
fn aarch64_bad64_bic_x0_x1_x2() {
    aarch64_bad64_xval::verify("bic x0, x1, x2", bad64::Op::BIC);
}

#[test]
fn aarch64_bad64_orn_x0_x1_x2() {
    aarch64_bad64_xval::verify("orn x0, x1, x2", bad64::Op::ORN);
}

#[test]
fn aarch64_bad64_rev16_x0_x1() {
    aarch64_bad64_xval::verify("rev16 x0, x1", bad64::Op::REV16);
}

#[test]
fn aarch64_bad64_rev32_x0_x1() {
    aarch64_bad64_xval::verify("rev32 x0, x1", bad64::Op::REV32);
}

#[test]
fn aarch64_bad64_madd_x0_x1_x2_x3() {
    aarch64_bad64_xval::verify("madd x0, x1, x2, x3", bad64::Op::MADD);
}

#[test]
fn aarch64_bad64_msub_x0_x1_x2_x3() {
    aarch64_bad64_xval::verify("msub x0, x1, x2, x3", bad64::Op::MSUB);
}

#[test]
fn aarch64_bad64_wfe() {
    aarch64_bad64_xval::verify("wfe", bad64::Op::WFE);
}

#[test]
fn aarch64_bad64_wfi() {
    aarch64_bad64_xval::verify("wfi", bad64::Op::WFI);
}

#[test]
fn aarch64_bad64_sev() {
    aarch64_bad64_xval::verify("sev", bad64::Op::SEV);
}

// ============================================================================
// ARM32 (A32) — yaxpeax-arm cross-validation tests
// ============================================================================

use yaxpeax_arm::armv7::Opcode as ArmOpcode;

#[test]
fn arm32_yax_add_r0_r1_r2() {
    arm32_xval::verify("add r0, r1, r2", ArmOpcode::ADD);
}

#[test]
fn arm32_yax_sub_r0_r1_r2() {
    arm32_xval::verify("sub r0, r1, r2", ArmOpcode::SUB);
}

#[test]
fn arm32_yax_and_r0_r1_r2() {
    arm32_xval::verify("and r0, r1, r2", ArmOpcode::AND);
}

#[test]
fn arm32_yax_orr_r0_r1_r2() {
    arm32_xval::verify("orr r0, r1, r2", ArmOpcode::ORR);
}

#[test]
fn arm32_yax_eor_r0_r1_r2() {
    arm32_xval::verify("eor r0, r1, r2", ArmOpcode::EOR);
}

#[test]
fn arm32_yax_mov_r0_r1() {
    arm32_xval::verify("mov r0, r1", ArmOpcode::MOV);
}

#[test]
fn arm32_yax_mov_r0_42() {
    arm32_xval::verify("mov r0, 42", ArmOpcode::MOV);
}

#[test]
fn arm32_yax_mvn_r0_r1() {
    arm32_xval::verify("mvn r0, r1", ArmOpcode::MVN);
}

#[test]
fn arm32_yax_cmp_r0_r1() {
    arm32_xval::verify("cmp r0, r1", ArmOpcode::CMP);
}

#[test]
fn arm32_yax_cmn_r0_r1() {
    arm32_xval::verify("cmn r0, r1", ArmOpcode::CMN);
}

#[test]
fn arm32_yax_tst_r0_r1() {
    arm32_xval::verify("tst r0, r1", ArmOpcode::TST);
}

#[test]
fn arm32_yax_teq_r0_r1() {
    arm32_xval::verify("teq r0, r1", ArmOpcode::TEQ);
}

#[test]
fn arm32_yax_ldr_r0_r1() {
    arm32_xval::verify("ldr r0, [r1]", ArmOpcode::LDR);
}

#[test]
fn arm32_yax_str_r0_r1() {
    arm32_xval::verify("str r0, [r1]", ArmOpcode::STR);
}

#[test]
fn arm32_yax_ldrb_r0_r1() {
    arm32_xval::verify("ldrb r0, [r1]", ArmOpcode::LDRB);
}

#[test]
fn arm32_yax_strb_r0_r1() {
    arm32_xval::verify("strb r0, [r1]", ArmOpcode::STRB);
}

#[test]
fn arm32_yax_ldr_r0_r1_8() {
    arm32_xval::verify("ldr r0, [r1, 8]", ArmOpcode::LDR);
}

#[test]
fn arm32_yax_str_r0_r1_8() {
    arm32_xval::verify("str r0, [r1, 8]", ArmOpcode::STR);
}

#[test]
fn arm32_yax_add_r0_r1_42() {
    arm32_xval::verify("add r0, r1, 42", ArmOpcode::ADD);
}

#[test]
fn arm32_yax_sub_r0_r1_42() {
    arm32_xval::verify("sub r0, r1, 42", ArmOpcode::SUB);
}

#[test]
fn arm32_yax_bic_r0_r1_r2() {
    arm32_xval::verify("bic r0, r1, r2", ArmOpcode::BIC);
}

#[test]
fn arm32_yax_rsb_r0_r1_r2() {
    arm32_xval::verify("rsb r0, r1, r2", ArmOpcode::RSB);
}

#[test]
fn arm32_yax_rsc_r0_r1_r2() {
    arm32_xval::verify("rsc r0, r1, r2", ArmOpcode::RSC);
}

#[test]
fn arm32_yax_adc_r0_r1_r2() {
    arm32_xval::verify("adc r0, r1, r2", ArmOpcode::ADC);
}

#[test]
fn arm32_yax_sbc_r0_r1_r2() {
    arm32_xval::verify("sbc r0, r1, r2", ArmOpcode::SBC);
}

#[test]
fn arm32_yax_mul_r0_r1_r2() {
    arm32_xval::verify("mul r0, r1, r2", ArmOpcode::MUL);
}

#[test]
fn arm32_yax_mla_r0_r1_r2_r3() {
    arm32_xval::verify("mla r0, r1, r2, r3", ArmOpcode::MLA);
}

// Note: ARM SVC encoding 0xEF000000 is not decoded by yaxpeax-arm v0.4
// (returns 'incomplete decoder'). Cross-validated via llvm-mc separately.

#[test]
fn arm32_yax_nop() {
    // ARM NOP is MOV r0, r0 (with condition AL)
    let (opcode, _) = arm32_xval::asm_and_decode("nop");
    assert!(
        opcode == ArmOpcode::NOP || opcode == ArmOpcode::MOV,
        "expected NOP or MOV, got {:?}",
        opcode
    );
}

#[test]
fn arm32_yax_ldrh_r0_r1() {
    arm32_xval::verify("ldrh r0, [r1]", ArmOpcode::LDRH);
}

#[test]
fn arm32_yax_strh_r0_r1() {
    arm32_xval::verify("strh r0, [r1]", ArmOpcode::STRH);
}

// Note: LDRSB/LDRSH use encoding form that yaxpeax-arm v0.4 flags as reserved.
// Our bytes match llvm-mc, so the issue is in yaxpeax's strict reserved-bit checking.
// Cross-validated via llvm-mc separately.

#[test]
fn arm32_yax_add_r0_r1_r2_lsl_3() {
    arm32_xval::verify("add r0, r1, r2, lsl, 3", ArmOpcode::ADD);
}

#[test]
fn arm32_yax_mov_r0_r1_lsr_4() {
    arm32_xval::verify("mov r0, r1, lsr, 4", ArmOpcode::MOV);
}

#[test]
fn arm32_yax_clz_r0_r1() {
    arm32_xval::verify("clz r0, r1", ArmOpcode::CLZ);
}

// ============================================================================
// Thumb — yaxpeax-arm cross-validation tests
// ============================================================================

#[test]
fn thumb_yax_mov_r0_42() {
    thumb_xval::verify("mov r0, 42", ArmOpcode::MOV);
}

#[test]
fn thumb_yax_add_r0_r1() {
    // Thumb ADD (register) — T2 encoding
    let (opcode, _) = thumb_xval::asm_and_decode("add r0, r1");
    assert!(opcode == ArmOpcode::ADD, "expected ADD, got {:?}", opcode);
}

#[test]
fn thumb_yax_sub_r0_r1_r2() {
    thumb_xval::verify("sub r0, r1, r2", ArmOpcode::SUB);
}

#[test]
fn thumb_yax_and_r0_r1() {
    thumb_xval::verify("and r0, r1", ArmOpcode::AND);
}

#[test]
fn thumb_yax_orr_r0_r1() {
    thumb_xval::verify("orr r0, r1", ArmOpcode::ORR);
}

#[test]
fn thumb_yax_eor_r0_r1() {
    thumb_xval::verify("eor r0, r1", ArmOpcode::EOR);
}

#[test]
fn thumb_yax_ldr_r0_r1() {
    thumb_xval::verify("ldr r0, [r1]", ArmOpcode::LDR);
}

#[test]
fn thumb_yax_str_r0_r1() {
    thumb_xval::verify("str r0, [r1]", ArmOpcode::STR);
}

#[test]
fn thumb_yax_ldrb_r0_r1() {
    thumb_xval::verify("ldrb r0, [r1]", ArmOpcode::LDRB);
}

#[test]
fn thumb_yax_strb_r0_r1() {
    thumb_xval::verify("strb r0, [r1]", ArmOpcode::STRB);
}

#[test]
fn thumb_yax_ldrh_r0_r1() {
    thumb_xval::verify("ldrh r0, [r1]", ArmOpcode::LDRH);
}

#[test]
fn thumb_yax_strh_r0_r1() {
    thumb_xval::verify("strh r0, [r1]", ArmOpcode::STRH);
}

#[test]
fn thumb_yax_cmp_r0_r1() {
    thumb_xval::verify("cmp r0, r1", ArmOpcode::CMP);
}

#[test]
fn thumb_yax_tst_r0_r1() {
    thumb_xval::verify("tst r0, r1", ArmOpcode::TST);
}

#[test]
fn thumb_yax_mvn_r0_r1() {
    thumb_xval::verify("mvn r0, r1", ArmOpcode::MVN);
}

#[test]
fn thumb_yax_svc_0() {
    thumb_xval::verify("svc 0", ArmOpcode::SVC);
}

#[test]
fn thumb_yax_nop() {
    let (opcode, _) = thumb_xval::asm_and_decode("nop");
    assert!(
        opcode == ArmOpcode::NOP || opcode == ArmOpcode::MOV,
        "expected NOP or MOV, got {:?}",
        opcode
    );
}

#[test]
fn thumb_yax_bkpt_0() {
    thumb_xval::verify("bkpt 0", ArmOpcode::BKPT);
}

#[test]
fn thumb_yax_push_r0() {
    // PUSH in thumb — yaxpeax decodes as STM/PUSH
    let (opcode, _) = thumb_xval::asm_and_decode("push {r0}");
    assert!(
        opcode == ArmOpcode::PUSH || opcode == ArmOpcode::STM(false, true, false, false),
        "expected PUSH or STM variant, got {:?}",
        opcode
    );
}

#[test]
fn thumb_yax_pop_r0() {
    let (opcode, _) = thumb_xval::asm_and_decode("pop {r0}");
    assert!(
        opcode == ArmOpcode::POP || opcode == ArmOpcode::LDM(true, false, false, false),
        "expected POP or LDM variant, got {:?}",
        opcode
    );
}

#[test]
fn thumb_yax_add_r0_r1_3() {
    thumb_xval::verify("add r0, r1, 3", ArmOpcode::ADD);
}

#[test]
fn thumb_yax_sub_r0_r1_3() {
    thumb_xval::verify("sub r0, r1, 3", ArmOpcode::SUB);
}

#[test]
fn thumb_yax_lsl_r0_r1() {
    thumb_xval::verify("lsl r0, r1", ArmOpcode::LSL);
}

#[test]
fn thumb_yax_lsr_r0_r1() {
    thumb_xval::verify("lsr r0, r1", ArmOpcode::LSR);
}

#[test]
fn thumb_yax_asr_r0_r1() {
    thumb_xval::verify("asr r0, r1", ArmOpcode::ASR);
}

#[test]
fn thumb_yax_neg_r0_r1() {
    // NEG is RSB Rd, Rm, 0 in unified syntax
    let (opcode, _) = thumb_xval::asm_and_decode("neg r0, r1");
    assert!(opcode == ArmOpcode::RSB, "expected RSB, got {:?}", opcode);
}

#[test]
fn thumb_yax_adc_r0_r1() {
    thumb_xval::verify("adc r0, r1", ArmOpcode::ADC);
}

#[test]
fn thumb_yax_sbc_r0_r1() {
    thumb_xval::verify("sbc r0, r1", ArmOpcode::SBC);
}

#[test]
fn thumb_yax_ror_r0_r1() {
    thumb_xval::verify("ror r0, r1", ArmOpcode::ROR);
}

#[test]
fn thumb_yax_mul_r0_r1() {
    thumb_xval::verify("mul r0, r1", ArmOpcode::MUL);
}

#[test]
fn thumb_yax_bic_r0_r1() {
    thumb_xval::verify("bic r0, r1", ArmOpcode::BIC);
}

#[test]
fn thumb_yax_cmn_r0_r1() {
    thumb_xval::verify("cmn r0, r1", ArmOpcode::CMN);
}

// ============================================================================
// RISC-V — riscv-decode cross-validation tests (RV64I base ISA)
// ============================================================================

#[test]
fn riscv_xval_add_x1_x2_x3() {
    riscv_xval::verify_variant("add x1, x2, x3", Arch::Rv64, "Add");
}

#[test]
fn riscv_xval_sub_x1_x2_x3() {
    riscv_xval::verify_variant("sub x1, x2, x3", Arch::Rv64, "Sub");
}

#[test]
fn riscv_xval_and_x1_x2_x3() {
    riscv_xval::verify_variant("and x1, x2, x3", Arch::Rv64, "And");
}

#[test]
fn riscv_xval_or_x1_x2_x3() {
    riscv_xval::verify_variant("or x1, x2, x3", Arch::Rv64, "Or");
}

#[test]
fn riscv_xval_xor_x1_x2_x3() {
    riscv_xval::verify_variant("xor x1, x2, x3", Arch::Rv64, "Xor");
}

#[test]
fn riscv_xval_sll_x1_x2_x3() {
    riscv_xval::verify_variant("sll x1, x2, x3", Arch::Rv64, "Sll");
}

#[test]
fn riscv_xval_srl_x1_x2_x3() {
    riscv_xval::verify_variant("srl x1, x2, x3", Arch::Rv64, "Srl");
}

#[test]
fn riscv_xval_sra_x1_x2_x3() {
    riscv_xval::verify_variant("sra x1, x2, x3", Arch::Rv64, "Sra");
}

#[test]
fn riscv_xval_slt_x1_x2_x3() {
    riscv_xval::verify_variant("slt x1, x2, x3", Arch::Rv64, "Slt");
}

#[test]
fn riscv_xval_sltu_x1_x2_x3() {
    riscv_xval::verify_variant("sltu x1, x2, x3", Arch::Rv64, "Sltu");
}

#[test]
fn riscv_xval_addi_x1_x2_42() {
    riscv_xval::verify_variant("addi x1, x2, 42", Arch::Rv64, "Addi");
}

#[test]
fn riscv_xval_andi_x1_x2_42() {
    riscv_xval::verify_variant("andi x1, x2, 42", Arch::Rv64, "Andi");
}

#[test]
fn riscv_xval_ori_x1_x2_42() {
    riscv_xval::verify_variant("ori x1, x2, 42", Arch::Rv64, "Ori");
}

#[test]
fn riscv_xval_xori_x1_x2_42() {
    riscv_xval::verify_variant("xori x1, x2, 42", Arch::Rv64, "Xori");
}

#[test]
fn riscv_xval_slti_x1_x2_42() {
    riscv_xval::verify_variant("slti x1, x2, 42", Arch::Rv64, "Slti");
}

#[test]
fn riscv_xval_sltiu_x1_x2_42() {
    riscv_xval::verify_variant("sltiu x1, x2, 42", Arch::Rv64, "Sltiu");
}

#[test]
fn riscv_xval_slli_x1_x2_5() {
    riscv_xval::verify_variant("slli x1, x2, 5", Arch::Rv64, "Slli");
}

#[test]
fn riscv_xval_srli_x1_x2_5() {
    riscv_xval::verify_variant("srli x1, x2, 5", Arch::Rv64, "Srli");
}

#[test]
fn riscv_xval_srai_x1_x2_5() {
    riscv_xval::verify_variant("srai x1, x2, 5", Arch::Rv64, "Srai");
}

#[test]
fn riscv_xval_lui_x1() {
    riscv_xval::verify_variant("lui x1, 0x12345", Arch::Rv64, "Lui");
}

#[test]
fn riscv_xval_auipc_x1() {
    riscv_xval::verify_variant("auipc x1, 0x12345", Arch::Rv64, "Auipc");
}

#[test]
fn riscv_xval_lb_x1_x2() {
    riscv_xval::verify_variant("lb x1, 0(x2)", Arch::Rv64, "Lb");
}

#[test]
fn riscv_xval_lh_x1_x2() {
    riscv_xval::verify_variant("lh x1, 0(x2)", Arch::Rv64, "Lh");
}

#[test]
fn riscv_xval_lw_x1_x2() {
    riscv_xval::verify_variant("lw x1, 0(x2)", Arch::Rv64, "Lw");
}

#[test]
fn riscv_xval_ld_x1_x2() {
    riscv_xval::verify_variant("ld x1, 0(x2)", Arch::Rv64, "Ld");
}

#[test]
fn riscv_xval_lbu_x1_x2() {
    riscv_xval::verify_variant("lbu x1, 0(x2)", Arch::Rv64, "Lbu");
}

#[test]
fn riscv_xval_lhu_x1_x2() {
    riscv_xval::verify_variant("lhu x1, 0(x2)", Arch::Rv64, "Lhu");
}

#[test]
fn riscv_xval_lwu_x1_x2() {
    riscv_xval::verify_variant("lwu x1, 0(x2)", Arch::Rv64, "Lwu");
}

#[test]
fn riscv_xval_sb_x1_x2() {
    riscv_xval::verify_variant("sb x1, 0(x2)", Arch::Rv64, "Sb");
}

#[test]
fn riscv_xval_sh_x1_x2() {
    riscv_xval::verify_variant("sh x1, 0(x2)", Arch::Rv64, "Sh");
}

#[test]
fn riscv_xval_sw_x1_x2() {
    riscv_xval::verify_variant("sw x1, 0(x2)", Arch::Rv64, "Sw");
}

#[test]
fn riscv_xval_sd_x1_x2() {
    riscv_xval::verify_variant("sd x1, 0(x2)", Arch::Rv64, "Sd");
}

#[test]
fn riscv_xval_ecall() {
    riscv_xval::verify_variant("ecall", Arch::Rv64, "Ecall");
}

#[test]
fn riscv_xval_ebreak() {
    riscv_xval::verify_variant("ebreak", Arch::Rv64, "Ebreak");
}

#[test]
fn riscv_xval_fence() {
    riscv_xval::verify_variant("fence", Arch::Rv64, "Fence");
}

// ============================================================================
// RISC-V — M extension (multiply/divide)
// ============================================================================

#[test]
fn riscv_xval_mul_x1_x2_x3() {
    riscv_xval::verify_variant("mul x1, x2, x3", Arch::Rv64, "Mul");
}

#[test]
fn riscv_xval_mulh_x1_x2_x3() {
    riscv_xval::verify_variant("mulh x1, x2, x3", Arch::Rv64, "Mulh");
}

#[test]
fn riscv_xval_mulhsu_x1_x2_x3() {
    riscv_xval::verify_variant("mulhsu x1, x2, x3", Arch::Rv64, "Mulhsu");
}

#[test]
fn riscv_xval_mulhu_x1_x2_x3() {
    riscv_xval::verify_variant("mulhu x1, x2, x3", Arch::Rv64, "Mulhu");
}

#[test]
fn riscv_xval_div_x1_x2_x3() {
    riscv_xval::verify_variant("div x1, x2, x3", Arch::Rv64, "Div");
}

#[test]
fn riscv_xval_divu_x1_x2_x3() {
    riscv_xval::verify_variant("divu x1, x2, x3", Arch::Rv64, "Divu");
}

#[test]
fn riscv_xval_rem_x1_x2_x3() {
    riscv_xval::verify_variant("rem x1, x2, x3", Arch::Rv64, "Rem");
}

#[test]
fn riscv_xval_remu_x1_x2_x3() {
    riscv_xval::verify_variant("remu x1, x2, x3", Arch::Rv64, "Remu");
}

#[test]
fn riscv_xval_mulw_x1_x2_x3() {
    riscv_xval::verify_variant("mulw x1, x2, x3", Arch::Rv64, "Mulw");
}

#[test]
fn riscv_xval_divw_x1_x2_x3() {
    riscv_xval::verify_variant("divw x1, x2, x3", Arch::Rv64, "Divw");
}

#[test]
fn riscv_xval_divuw_x1_x2_x3() {
    riscv_xval::verify_variant("divuw x1, x2, x3", Arch::Rv64, "Divuw");
}

#[test]
fn riscv_xval_remw_x1_x2_x3() {
    riscv_xval::verify_variant("remw x1, x2, x3", Arch::Rv64, "Remw");
}

#[test]
fn riscv_xval_remuw_x1_x2_x3() {
    riscv_xval::verify_variant("remuw x1, x2, x3", Arch::Rv64, "Remuw");
}

// ============================================================================
// RISC-V — A extension (atomics)
// ============================================================================

#[test]
fn riscv_xval_lr_w() {
    riscv_xval::verify_variant("lr.w x10, (x11)", Arch::Rv64, "LrW");
}

#[test]
fn riscv_xval_sc_w() {
    riscv_xval::verify_variant("sc.w x10, x12, (x11)", Arch::Rv64, "ScW");
}

#[test]
fn riscv_xval_amoswap_w() {
    riscv_xval::verify_variant("amoswap.w x10, x12, (x11)", Arch::Rv64, "AmoswapW");
}

#[test]
fn riscv_xval_amoadd_w() {
    riscv_xval::verify_variant("amoadd.w x10, x12, (x11)", Arch::Rv64, "AmoaddW");
}

#[test]
fn riscv_xval_amoand_w() {
    riscv_xval::verify_variant("amoand.w x10, x12, (x11)", Arch::Rv64, "AmoandW");
}

#[test]
fn riscv_xval_amoor_w() {
    riscv_xval::verify_variant("amoor.w x10, x12, (x11)", Arch::Rv64, "AmoorW");
}

#[test]
fn riscv_xval_amoxor_w() {
    riscv_xval::verify_variant("amoxor.w x10, x12, (x11)", Arch::Rv64, "AmoxorW");
}

#[test]
fn riscv_xval_amomax_w() {
    riscv_xval::verify_variant("amomax.w x10, x12, (x11)", Arch::Rv64, "AmomaxW");
}

#[test]
fn riscv_xval_amomin_w() {
    riscv_xval::verify_variant("amomin.w x10, x12, (x11)", Arch::Rv64, "AmominW");
}

#[test]
fn riscv_xval_amomaxu_w() {
    riscv_xval::verify_variant("amomaxu.w x10, x12, (x11)", Arch::Rv64, "AmomaxuW");
}

#[test]
fn riscv_xval_amominu_w() {
    riscv_xval::verify_variant("amominu.w x10, x12, (x11)", Arch::Rv64, "AmominuW");
}

#[test]
fn riscv_xval_lr_d() {
    riscv_xval::verify_variant("lr.d x10, (x11)", Arch::Rv64, "LrD");
}

#[test]
fn riscv_xval_sc_d() {
    riscv_xval::verify_variant("sc.d x10, x12, (x11)", Arch::Rv64, "ScD");
}

#[test]
fn riscv_xval_amoswap_d() {
    riscv_xval::verify_variant("amoswap.d x10, x12, (x11)", Arch::Rv64, "AmoswapD");
}

#[test]
fn riscv_xval_amoadd_d() {
    riscv_xval::verify_variant("amoadd.d x10, x12, (x11)", Arch::Rv64, "AmoaddD");
}

#[test]
fn riscv_xval_amoand_d() {
    riscv_xval::verify_variant("amoand.d x10, x12, (x11)", Arch::Rv64, "AmoandD");
}

#[test]
fn riscv_xval_amoor_d() {
    riscv_xval::verify_variant("amoor.d x10, x12, (x11)", Arch::Rv64, "AmoorD");
}

#[test]
fn riscv_xval_amoxor_d() {
    riscv_xval::verify_variant("amoxor.d x10, x12, (x11)", Arch::Rv64, "AmoxorD");
}

#[test]
fn riscv_xval_amomax_d() {
    riscv_xval::verify_variant("amomax.d x10, x12, (x11)", Arch::Rv64, "AmomaxD");
}

#[test]
fn riscv_xval_amomin_d() {
    riscv_xval::verify_variant("amomin.d x10, x12, (x11)", Arch::Rv64, "AmominD");
}

#[test]
fn riscv_xval_amomaxu_d() {
    riscv_xval::verify_variant("amomaxu.d x10, x12, (x11)", Arch::Rv64, "AmomaxuD");
}

#[test]
fn riscv_xval_amominu_d() {
    riscv_xval::verify_variant("amominu.d x10, x12, (x11)", Arch::Rv64, "AmominuD");
}

// ============================================================================
// RISC-V — RV64I-specific instructions
// ============================================================================

#[test]
fn riscv_xval_addw_x1_x2_x3() {
    riscv_xval::verify_variant("addw x1, x2, x3", Arch::Rv64, "Addw");
}

#[test]
fn riscv_xval_subw_x1_x2_x3() {
    riscv_xval::verify_variant("subw x1, x2, x3", Arch::Rv64, "Subw");
}

#[test]
fn riscv_xval_sllw_x1_x2_x3() {
    riscv_xval::verify_variant("sllw x1, x2, x3", Arch::Rv64, "Sllw");
}

#[test]
fn riscv_xval_srlw_x1_x2_x3() {
    riscv_xval::verify_variant("srlw x1, x2, x3", Arch::Rv64, "Srlw");
}

#[test]
fn riscv_xval_sraw_x1_x2_x3() {
    riscv_xval::verify_variant("sraw x1, x2, x3", Arch::Rv64, "Sraw");
}

#[test]
fn riscv_xval_addiw_x1_x2_42() {
    riscv_xval::verify_variant("addiw x1, x2, 42", Arch::Rv64, "Addiw");
}

#[test]
fn riscv_xval_slliw_x1_x2_5() {
    riscv_xval::verify_variant("slliw x1, x2, 5", Arch::Rv64, "Slliw");
}

#[test]
fn riscv_xval_srliw_x1_x2_5() {
    riscv_xval::verify_variant("srliw x1, x2, 5", Arch::Rv64, "Srliw");
}

#[test]
fn riscv_xval_sraiw_x1_x2_5() {
    riscv_xval::verify_variant("sraiw x1, x2, 5", Arch::Rv64, "Sraiw");
}
