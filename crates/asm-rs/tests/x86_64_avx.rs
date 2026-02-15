//! AVX / AVX2 / AVX-512 / BMI / FMA integration tests for x86-64.
//!
//! These tests exercise the VEX and EVEX encoding paths in x86.rs
//! that are not covered by the cross-validation or core x86_64 tests.

#[cfg(feature = "x86_64")]
use asm_rs::{assemble, Arch};

// Helper: assemble and return byte count
#[cfg(feature = "x86_64")]
fn asm(src: &str) -> Vec<u8> {
    assemble(src, Arch::X86_64).unwrap()
}

// ============================================================================
// AVX (VEX) arithmetic — lines 3156–3575 in x86.rs
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_arith_xmm() {
    // Each VEX 3-operand instruction should produce known-length output
    assert_eq!(asm("vaddps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vaddpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vsubps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vsubpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vmulps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vmulpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vdivps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vdivpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vmaxps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vmaxpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vminps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vminpd xmm0, xmm1, xmm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_arith_ymm() {
    assert_eq!(asm("vaddps ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vaddpd ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vsubps ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vsubpd ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vmulps ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vmulpd ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vdivps ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vdivpd ymm0, ymm1, ymm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_arith_scalar() {
    assert_eq!(asm("vaddss xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vaddsd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vsubss xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vsubsd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vmulss xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vmulsd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vdivss xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vdivsd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vmaxss xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vmaxsd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vminss xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vminsd xmm0, xmm1, xmm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_sqrt() {
    assert_eq!(asm("vsqrtps xmm0, xmm1").len(), 4);
    assert_eq!(asm("vsqrtpd xmm0, xmm1").len(), 4);
    assert_eq!(asm("vsqrtps ymm0, ymm1").len(), 4);
    assert_eq!(asm("vsqrtpd ymm0, ymm1").len(), 4);
    assert_eq!(asm("vsqrtss xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vsqrtsd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vrcpps xmm0, xmm1").len(), 4);
    assert_eq!(asm("vrcpss xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vrsqrtps xmm0, xmm1").len(), 4);
    assert_eq!(asm("vrsqrtss xmm0, xmm1, xmm2").len(), 4);
}

// ============================================================================
// AVX logical — lines 3576–3665
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_logical() {
    assert_eq!(asm("vandps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vandpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vandnps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vandnpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vorps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vorpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vxorps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vxorpd xmm0, xmm1, xmm2").len(), 4);
    // YMM variants
    assert_eq!(asm("vandps ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vorps ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vxorps ymm0, ymm1, ymm2").len(), 4);
}

// ============================================================================
// AVX compare — lines 3666–3711
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_compare() {
    assert_eq!(asm("vcmpps xmm0, xmm1, xmm2, 0").len(), 5);
    assert_eq!(asm("vcmppd xmm0, xmm1, xmm2, 0").len(), 5);
    assert_eq!(asm("vcmpss xmm0, xmm1, xmm2, 0").len(), 5);
    assert_eq!(asm("vcmpsd xmm0, xmm1, xmm2, 0").len(), 5);
    assert_eq!(asm("vcomiss xmm0, xmm1").len(), 4);
    assert_eq!(asm("vcomisd xmm0, xmm1").len(), 4);
    assert_eq!(asm("vucomiss xmm0, xmm1").len(), 4);
    assert_eq!(asm("vucomisd xmm0, xmm1").len(), 4);
}

// ============================================================================
// AVX data movement — lines 3712–3922
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_data_movement() {
    assert_eq!(asm("vmovaps xmm0, xmm1").len(), 4);
    assert_eq!(asm("vmovaps ymm0, ymm1").len(), 4);
    assert_eq!(asm("vmovapd xmm0, xmm1").len(), 4);
    assert_eq!(asm("vmovapd ymm0, ymm1").len(), 4);
    assert_eq!(asm("vmovups xmm0, xmm1").len(), 4);
    assert_eq!(asm("vmovups ymm0, ymm1").len(), 4);
    assert_eq!(asm("vmovupd xmm0, xmm1").len(), 4);
    assert_eq!(asm("vmovupd ymm0, ymm1").len(), 4);
    assert_eq!(asm("vmovdqa xmm0, xmm1").len(), 4);
    assert_eq!(asm("vmovdqa ymm0, ymm1").len(), 4);
    assert_eq!(asm("vmovdqu xmm0, xmm1").len(), 4);
    assert_eq!(asm("vmovdqu ymm0, ymm1").len(), 4);
    assert_eq!(asm("vmovss xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vmovsd xmm0, xmm1, xmm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_data_movement_extended() {
    assert_eq!(asm("vmovsldup xmm0, xmm1").len(), 4);
    assert_eq!(asm("vmovshdup xmm0, xmm1").len(), 4);
    assert_eq!(asm("vmovddup xmm0, xmm1").len(), 4);
}

// ============================================================================
// AVX unpack — lines 3923–3968
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_unpack() {
    assert_eq!(asm("vunpcklps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vunpckhps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vunpcklpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vunpckhpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vunpcklps ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vunpckhps ymm0, ymm1, ymm2").len(), 4);
}

// ============================================================================
// AVX integer — lines 3969–4509
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_integer_basic() {
    assert_eq!(asm("vpaddb xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpaddw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpaddd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpaddq xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsubb xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsubw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsubd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsubq xmm0, xmm1, xmm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_integer_ymm() {
    assert_eq!(asm("vpaddb ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vpaddw ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vpaddd ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vpaddq ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vpsubb ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vpsubw ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vpsubd ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vpsubq ymm0, ymm1, ymm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_integer_multiply() {
    assert_eq!(asm("vpmullw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpmulhw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpmulhuw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpmuludq xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpmulld xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpmuldq xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpmaddwd xmm0, xmm1, xmm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_integer_compare() {
    assert_eq!(asm("vpcmpeqb xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpcmpeqw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpcmpeqd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpcmpeqq xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpcmpgtb xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpcmpgtw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpcmpgtd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpcmpgtq xmm0, xmm1, xmm2").len(), 5);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_integer_logical() {
    assert_eq!(asm("vpand xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpandn xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpor xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpxor xmm0, xmm1, xmm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_integer_pack() {
    assert_eq!(asm("vpacksswb xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpackssdw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpackuswb xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpackusdw xmm0, xmm1, xmm2").len(), 5);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_integer_unpack() {
    assert_eq!(asm("vpunpcklbw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpunpcklwd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpunpckldq xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpunpcklqdq xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpunpckhbw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpunpckhwd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpunpckhdq xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpunpckhqdq xmm0, xmm1, xmm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_integer_shift() {
    assert_eq!(asm("vpsllw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpslld xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsllq xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsrlw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsrld xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsrlq xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsraw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsrad xmm0, xmm1, xmm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_integer_shift_imm() {
    assert_eq!(asm("vpsllw xmm0, xmm1, 4").len(), 5);
    assert_eq!(asm("vpslld xmm0, xmm1, 4").len(), 5);
    assert_eq!(asm("vpsllq xmm0, xmm1, 4").len(), 5);
    assert_eq!(asm("vpsrlw xmm0, xmm1, 4").len(), 5);
    assert_eq!(asm("vpsrld xmm0, xmm1, 4").len(), 5);
    assert_eq!(asm("vpsrlq xmm0, xmm1, 4").len(), 5);
    assert_eq!(asm("vpsraw xmm0, xmm1, 4").len(), 5);
    assert_eq!(asm("vpsrad xmm0, xmm1, 4").len(), 5);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_integer_psadbw() {
    assert_eq!(asm("vpsadbw xmm0, xmm1, xmm2").len(), 4);
}

// vmovmskps/vpmovmskb not implemented — skipped

// ============================================================================
// AVX 0F 38 (SSSE3/SSE4 VEX) — lines 4510–5039
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_ssse3_vex() {
    assert_eq!(asm("vpshufb xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vphaddw xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vphaddd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vphaddsw xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vphsubw xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vphsubd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vphsubsw xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpmaddubsw xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpmulhrsw xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpabsb xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpabsw xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpabsd xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpsignb xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpsignw xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpsignd xmm0, xmm1, xmm2").len(), 5);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_sse41_vex() {
    assert_eq!(asm("vptest xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpminub xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpmaxub xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpminsw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpmaxsw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpminuw xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpmaxuw xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpminsb xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpmaxsb xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpminsd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpmaxsd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpminud xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpmaxud xmm0, xmm1, xmm2").len(), 5);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_sse41_extend_vex() {
    assert_eq!(asm("vpmovsxbw xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpmovsxbd xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpmovsxbq xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpmovsxwd xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpmovsxwq xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpmovsxdq xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpmovzxbw xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpmovzxbd xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpmovzxbq xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpmovzxwd xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpmovzxwq xmm0, xmm1").len(), 5);
    assert_eq!(asm("vpmovzxdq xmm0, xmm1").len(), 5);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_aes_vex() {
    assert_eq!(asm("vaesenc xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vaesenclast xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vaesdec xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vaesdeclast xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vaesimc xmm0, xmm1").len(), 5);
}

// ============================================================================
// AVX with immediate (0F / 0F3A) — lines 5040–5261
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_imm8_ops() {
    assert_eq!(asm("vshufps xmm0, xmm1, xmm2, 0").len(), 5);
    assert_eq!(asm("vshufpd xmm0, xmm1, xmm2, 0").len(), 5);
    assert_eq!(asm("vblendps xmm0, xmm1, xmm2, 0x0F").len(), 6);
    assert_eq!(asm("vblendpd xmm0, xmm1, xmm2, 0x03").len(), 6);
    assert_eq!(asm("vpblendw xmm0, xmm1, xmm2, 0xFF").len(), 6);
    assert_eq!(asm("vroundps xmm0, xmm1, 0").len(), 6);
    assert_eq!(asm("vroundpd xmm0, xmm1, 0").len(), 6);
    assert_eq!(asm("vroundss xmm0, xmm1, xmm2, 0").len(), 6);
    assert_eq!(asm("vroundsd xmm0, xmm1, xmm2, 0").len(), 6);
    assert_eq!(asm("vpalignr xmm0, xmm1, xmm2, 4").len(), 6);
    assert_eq!(asm("vpshufd xmm0, xmm1, 0xFF").len(), 5);
    assert_eq!(asm("vpshufhw xmm0, xmm1, 0xFF").len(), 5);
    assert_eq!(asm("vpshuflw xmm0, xmm1, 0xFF").len(), 5);
}

// vpinsrb/vpinsrd/vpinsrq/vpextrb/vpextrd/vpextrq/vpextrw with GP reg not supported — skipped

#[test]
#[cfg(feature = "x86_64")]
fn avx_dpps_dppd() {
    assert_eq!(asm("vdpps xmm0, xmm1, xmm2, 0xFF").len(), 6);
    assert_eq!(asm("vdppd xmm0, xmm1, xmm2, 0xFF").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_mpsadbw() {
    assert_eq!(asm("vmpsadbw xmm0, xmm1, xmm2, 0").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_aeskeygenassist() {
    assert_eq!(asm("vaeskeygenassist xmm0, xmm1, 0x01").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_pclmulqdq() {
    assert_eq!(asm("vpclmulqdq xmm0, xmm1, xmm2, 0x00").len(), 6);
}

// ============================================================================
// BMI1 — lines 5262–5311
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn bmi1_instructions() {
    assert_eq!(asm("andn eax, ebx, ecx").len(), 5);
    assert_eq!(asm("andn rax, rbx, rcx").len(), 5);
    assert_eq!(asm("bextr eax, ebx, ecx").len(), 5);
    assert_eq!(asm("bextr rax, rbx, rcx").len(), 5);
    assert_eq!(asm("blsi eax, ebx").len(), 5);
    assert_eq!(asm("blsi rax, rbx").len(), 5);
    assert_eq!(asm("blsmsk eax, ebx").len(), 5);
    assert_eq!(asm("blsmsk rax, rbx").len(), 5);
    assert_eq!(asm("blsr eax, ebx").len(), 5);
    assert_eq!(asm("blsr rax, rbx").len(), 5);
}

// ============================================================================
// BMI2 — lines 5312–5385
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn bmi2_instructions() {
    assert_eq!(asm("bzhi eax, ebx, ecx").len(), 5);
    assert_eq!(asm("bzhi rax, rbx, rcx").len(), 5);
    assert_eq!(asm("mulx edx, eax, ebx").len(), 5);
    assert_eq!(asm("mulx rdx, rax, rbx").len(), 5);
    assert_eq!(asm("pdep eax, ebx, ecx").len(), 5);
    assert_eq!(asm("pdep rax, rbx, rcx").len(), 5);
    assert_eq!(asm("pext eax, ebx, ecx").len(), 5);
    assert_eq!(asm("pext rax, rbx, rcx").len(), 5);
    assert_eq!(asm("rorx eax, ebx, 7").len(), 6);
    assert_eq!(asm("rorx rax, rbx, 7").len(), 6);
    assert_eq!(asm("sarx eax, ebx, ecx").len(), 5);
    assert_eq!(asm("sarx rax, rbx, rcx").len(), 5);
    assert_eq!(asm("shlx eax, ebx, ecx").len(), 5);
    assert_eq!(asm("shlx rax, rbx, rcx").len(), 5);
    assert_eq!(asm("shrx eax, ebx, ecx").len(), 5);
    assert_eq!(asm("shrx rax, rbx, rcx").len(), 5);
}

// ============================================================================
// FMA3 — lines 5386–6056
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn fma3_ps_forms() {
    // 132, 213, 231 forms for ps
    assert_eq!(asm("vfmadd132ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmadd213ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmadd231ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsub132ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsub213ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsub231ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmadd132ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmadd213ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmadd231ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmsub132ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmsub213ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmsub231ps xmm0, xmm1, xmm2").len(), 5);
}

#[test]
#[cfg(feature = "x86_64")]
fn fma3_pd_forms() {
    assert_eq!(asm("vfmadd132pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmadd213pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmadd231pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsub132pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsub213pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsub231pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmadd132pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmadd213pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmadd231pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmsub132pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmsub213pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmsub231pd xmm0, xmm1, xmm2").len(), 5);
}

#[test]
#[cfg(feature = "x86_64")]
fn fma3_scalar_forms() {
    assert_eq!(asm("vfmadd132ss xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmadd213ss xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmadd231ss xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmadd132sd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmadd213sd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmadd231sd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsub132ss xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsub231sd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmadd231ss xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfnmsub231sd xmm0, xmm1, xmm2").len(), 5);
}

#[test]
#[cfg(feature = "x86_64")]
fn fma3_addsub_forms() {
    assert_eq!(asm("vfmaddsub132ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmaddsub213ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmaddsub231ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmaddsub132pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmaddsub213pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmaddsub231pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsubadd132ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsubadd213ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsubadd231ps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsubadd132pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsubadd213pd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vfmsubadd231pd xmm0, xmm1, xmm2").len(), 5);
}

#[test]
#[cfg(feature = "x86_64")]
fn fma3_ymm_forms() {
    assert_eq!(asm("vfmadd231ps ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vfmadd231pd ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vfmsub231ps ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vfnmadd231ps ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vfnmsub231pd ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vfmaddsub231ps ymm0, ymm1, ymm2").len(), 5);
}

// ============================================================================
// AVX/AVX2 packed shifts — lines 6057–6169
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx2_packed_shifts_ymm() {
    assert_eq!(asm("vpsllw ymm0, ymm1, xmm2").len(), 4);
    assert_eq!(asm("vpslld ymm0, ymm1, xmm2").len(), 4);
    assert_eq!(asm("vpsllq ymm0, ymm1, xmm2").len(), 4);
    assert_eq!(asm("vpsrlw ymm0, ymm1, xmm2").len(), 4);
    assert_eq!(asm("vpsrld ymm0, ymm1, xmm2").len(), 4);
    assert_eq!(asm("vpsrlq ymm0, ymm1, xmm2").len(), 4);
    assert_eq!(asm("vpsraw ymm0, ymm1, xmm2").len(), 4);
    assert_eq!(asm("vpsrad ymm0, ymm1, xmm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx2_packed_shifts_imm_ymm() {
    assert_eq!(asm("vpsllw ymm0, ymm1, 4").len(), 5);
    assert_eq!(asm("vpslld ymm0, ymm1, 4").len(), 5);
    assert_eq!(asm("vpsllq ymm0, ymm1, 4").len(), 5);
    assert_eq!(asm("vpsrlw ymm0, ymm1, 4").len(), 5);
    assert_eq!(asm("vpsrld ymm0, ymm1, 4").len(), 5);
    assert_eq!(asm("vpsrlq ymm0, ymm1, 4").len(), 5);
    assert_eq!(asm("vpsraw ymm0, ymm1, 4").len(), 5);
    assert_eq!(asm("vpsrad ymm0, ymm1, 4").len(), 5);
    assert_eq!(asm("vpslldq ymm0, ymm1, 4").len(), 5);
    assert_eq!(asm("vpsrldq ymm0, ymm1, 4").len(), 5);
}

// ============================================================================
// AVX2 variable shifts — lines 6170–6226
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx2_variable_shifts() {
    assert_eq!(asm("vpsllvd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpsllvd ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vpsllvq xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpsllvq ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vpsrlvd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpsrlvd ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vpsrlvq xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpsrlvq ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vpsravd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpsravd ymm0, ymm1, ymm2").len(), 5);
}

// ============================================================================
// AVX permute / shuffle — lines 6227–6348
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_permute_shuffle() {
    assert_eq!(asm("vpermilps xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpermilps ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vpermilpd xmm0, xmm1, xmm2").len(), 5);
    assert_eq!(asm("vpermilpd ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vpermilps xmm0, xmm1, 0").len(), 6);
    assert_eq!(asm("vpermilpd xmm0, xmm1, 0").len(), 6);
    assert_eq!(asm("vperm2f128 ymm0, ymm1, ymm2, 0").len(), 6);
    assert_eq!(asm("vperm2i128 ymm0, ymm1, ymm2, 0").len(), 6);
    assert_eq!(asm("vpermd ymm0, ymm1, ymm2").len(), 5);
    assert_eq!(asm("vpermq ymm0, ymm1, 0").len(), 6);
    assert_eq!(asm("vpermpd ymm0, ymm1, 0").len(), 6);
    assert_eq!(asm("vpermps ymm0, ymm1, ymm2").len(), 5);
}

// ============================================================================
// AVX broadcast / insert / extract — lines 6349–6488
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_broadcast() {
    assert_eq!(asm("vbroadcastss xmm0, xmm1").len(), 5);
    assert_eq!(asm("vbroadcastss ymm0, xmm1").len(), 5);
    assert_eq!(asm("vbroadcastsd ymm0, xmm1").len(), 5);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx2_insert_extract_128() {
    assert_eq!(asm("vinsertf128 ymm0, ymm1, xmm2, 0").len(), 6);
    assert_eq!(asm("vinserti128 ymm0, ymm1, xmm2, 0").len(), 6);
    assert_eq!(asm("vextractf128 xmm0, ymm1, 0").len(), 6);
    assert_eq!(asm("vextracti128 xmm0, ymm1, 0").len(), 6);
}

// ============================================================================
// AVX masked moves — lines 6489–6534
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_masked_moves() {
    assert_eq!(asm("vmaskmovps xmm0, xmm1, [rax]").len(), 5);
    assert_eq!(asm("vmaskmovpd xmm0, xmm1, [rax]").len(), 5);
    assert_eq!(asm("vmaskmovps ymm0, ymm1, [rax]").len(), 5);
    assert_eq!(asm("vmaskmovpd ymm0, ymm1, [rax]").len(), 5);
    assert_eq!(asm("vpmaskmovd xmm0, xmm1, [rax]").len(), 5);
    assert_eq!(asm("vpmaskmovq xmm0, xmm1, [rax]").len(), 5);
}

// ============================================================================
// AVX test — lines 6535–6558
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_test_instructions() {
    assert_eq!(asm("vtestps xmm0, xmm1").len(), 5);
    assert_eq!(asm("vtestpd xmm0, xmm1").len(), 5);
    assert_eq!(asm("vtestps ymm0, ymm1").len(), 5);
    assert_eq!(asm("vtestpd ymm0, ymm1").len(), 5);
}

// ============================================================================
// AVX conversions — lines 6559–6745
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_conversions() {
    assert_eq!(asm("vcvtps2pd xmm0, xmm1").len(), 4);
    assert_eq!(asm("vcvtpd2ps xmm0, xmm1").len(), 4);
    assert_eq!(asm("vcvtdq2ps xmm0, xmm1").len(), 4);
    assert_eq!(asm("vcvtps2dq xmm0, xmm1").len(), 4);
    assert_eq!(asm("vcvttps2dq xmm0, xmm1").len(), 4);
    assert_eq!(asm("vcvtdq2pd xmm0, xmm1").len(), 4);
    assert_eq!(asm("vcvtpd2dq xmm0, xmm1").len(), 4);
    assert_eq!(asm("vcvttpd2dq xmm0, xmm1").len(), 4);
    assert_eq!(asm("vcvtsi2ss xmm0, xmm1, eax").len(), 4);
    assert_eq!(asm("vcvtsi2sd xmm0, xmm1, eax").len(), 4);
    assert_eq!(asm("vcvtsi2ss xmm0, xmm1, rax").len(), 5);
    assert_eq!(asm("vcvtsi2sd xmm0, xmm1, rax").len(), 5);
    assert_eq!(asm("vcvtss2si eax, xmm0").len(), 4);
    assert_eq!(asm("vcvtsd2si eax, xmm0").len(), 4);
    assert_eq!(asm("vcvttss2si eax, xmm0").len(), 4);
    assert_eq!(asm("vcvttsd2si eax, xmm0").len(), 4);
    assert_eq!(asm("vcvtss2sd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vcvtsd2ss xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vcvtph2ps xmm0, xmm1").len(), 5);
    assert_eq!(asm("vcvtps2ph xmm0, xmm1, 0").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx_conversions_ymm() {
    assert_eq!(asm("vcvtps2pd ymm0, xmm1").len(), 4);
    assert_eq!(asm("vcvtpd2ps xmm0, ymm1").len(), 4);
    assert_eq!(asm("vcvtdq2ps ymm0, ymm1").len(), 4);
    assert_eq!(asm("vcvtps2dq ymm0, ymm1").len(), 4);
    assert_eq!(asm("vcvttps2dq ymm0, ymm1").len(), 4);
    assert_eq!(asm("vcvtdq2pd ymm0, xmm1").len(), 4);
    assert_eq!(asm("vcvtph2ps ymm0, xmm1").len(), 5);
    assert_eq!(asm("vcvtps2ph xmm0, ymm1, 0").len(), 6);
}

// ============================================================================
// AVX2 additional integer — lines 6746–6905
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx2_additional_integer() {
    assert_eq!(asm("vpaddsb xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpaddsw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpaddusb xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpaddusw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsubsb xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsubsw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsubusb xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpsubusw xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpavgb xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vpavgw xmm0, xmm1, xmm2").len(), 4);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx2_additional_integer_ymm() {
    assert_eq!(asm("vpaddsb ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vpaddusw ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vpsubsw ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vpsubusb ymm0, ymm1, ymm2").len(), 4);
    assert_eq!(asm("vpavgb ymm0, ymm1, ymm2").len(), 4);
}

// ============================================================================
// AVX misc
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_misc() {
    assert_eq!(asm("vzeroall").len(), 3);
    assert_eq!(asm("vzeroupper").len(), 3);
    assert_eq!(asm("vaddsubps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vaddsubpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vhaddps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vhaddpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vhsubps xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vhsubpd xmm0, xmm1, xmm2").len(), 4);
    assert_eq!(asm("vlddqu xmm0, [rax]").len(), 4);
    assert_eq!(asm("vlddqu ymm0, [rax]").len(), 4);
}

// vblendvps/vpblendvb 4-operand form not supported — skipped

// ============================================================================
// AVX-512 (EVEX) — lines 2627–3155
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx512_arithmetic() {
    // EVEX instructions are 6 bytes for reg-reg
    assert_eq!(asm("vaddps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vaddpd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vsubps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vsubpd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vmulps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vmulpd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vdivps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vdivpd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vmaxps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vminps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vsqrtps zmm0, zmm1").len(), 6);
    assert_eq!(asm("vsqrtpd zmm0, zmm1").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_logical() {
    assert_eq!(asm("vandps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vandpd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vandnps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vorps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vorpd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vxorps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vxorpd zmm0, zmm1, zmm2").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_data_movement() {
    assert_eq!(asm("vmovaps zmm0, zmm1").len(), 6);
    assert_eq!(asm("vmovapd zmm0, zmm1").len(), 6);
    assert_eq!(asm("vmovups zmm0, zmm1").len(), 6);
    assert_eq!(asm("vmovupd zmm0, zmm1").len(), 6);
    assert_eq!(asm("vmovdqa32 zmm0, zmm1").len(), 6);
    assert_eq!(asm("vmovdqa64 zmm0, zmm1").len(), 6);
    assert_eq!(asm("vmovdqu32 zmm0, zmm1").len(), 6);
    assert_eq!(asm("vmovdqu64 zmm0, zmm1").len(), 6);
    assert_eq!(asm("vmovdqu8 zmm0, zmm1").len(), 6);
    assert_eq!(asm("vmovdqu16 zmm0, zmm1").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_unpack() {
    assert_eq!(asm("vunpckhps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vunpckhpd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vunpcklps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vunpcklpd zmm0, zmm1, zmm2").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_fma() {
    assert_eq!(asm("vfmadd132ps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vfmadd213ps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vfmadd231ps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vfmadd132pd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vfmadd213pd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vfmadd231pd zmm0, zmm1, zmm2").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_integer() {
    assert_eq!(asm("vpaddd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpaddq zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpsubd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpsubq zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpmulld zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpmullq zmm0, zmm1, zmm2").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_integer_logical() {
    assert_eq!(asm("vpandd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpandq zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpandnd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpandnq zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpord zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vporq zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpxord zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpxorq zmm0, zmm1, zmm2").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_compare() {
    assert_eq!(asm("vpcmpeqd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpcmpgtd zmm0, zmm1, zmm2").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_permute() {
    assert_eq!(asm("vpermps zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpermpd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpermd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpermq zmm0, zmm1, zmm2").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_blend_ternlog() {
    assert_eq!(asm("vpblendmd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpblendmq zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpternlogd zmm0, zmm1, zmm2, 0xFF").len(), 7);
    assert_eq!(asm("vpternlogq zmm0, zmm1, zmm2, 0xFF").len(), 7);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_compress_expand() {
    assert_eq!(asm("vpcompressd zmm0, zmm1").len(), 6);
    assert_eq!(asm("vpcompressq zmm0, zmm1").len(), 6);
    assert_eq!(asm("vpexpandd zmm0, zmm1").len(), 6);
    assert_eq!(asm("vpexpandq zmm0, zmm1").len(), 6);
    assert_eq!(asm("vcompressps zmm0, zmm1").len(), 6);
    assert_eq!(asm("vcompresspd zmm0, zmm1").len(), 6);
    assert_eq!(asm("vexpandps zmm0, zmm1").len(), 6);
    assert_eq!(asm("vexpandpd zmm0, zmm1").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_shuffle() {
    assert_eq!(asm("vshufps zmm0, zmm1, zmm2, 0").len(), 7);
    assert_eq!(asm("vshufpd zmm0, zmm1, zmm2, 0").len(), 7);
    assert_eq!(asm("vpshufd zmm0, zmm1, 0").len(), 7);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_convert() {
    assert_eq!(asm("vcvtps2pd zmm0, ymm1").len(), 6);
    assert_eq!(asm("vcvtpd2ps ymm0, zmm1").len(), 6);
    assert_eq!(asm("vcvtdq2ps zmm0, zmm1").len(), 6);
    assert_eq!(asm("vcvtps2dq zmm0, zmm1").len(), 6);
    assert_eq!(asm("vcvtdq2pd zmm0, ymm1").len(), 6);
    assert_eq!(asm("vcvttps2dq zmm0, zmm1").len(), 6);
    assert_eq!(asm("vcvttpd2dq ymm0, zmm1").len(), 6);
    assert_eq!(asm("vcvtpd2dq ymm0, zmm1").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_broadcast() {
    assert_eq!(asm("vbroadcastss zmm0, xmm1").len(), 6);
    assert_eq!(asm("vbroadcastsd zmm0, xmm1").len(), 6);
    assert_eq!(asm("vpbroadcastd zmm0, xmm1").len(), 6);
    assert_eq!(asm("vpbroadcastq zmm0, xmm1").len(), 6);
    assert_eq!(asm("vpbroadcastb zmm0, xmm1").len(), 6);
    assert_eq!(asm("vpbroadcastw zmm0, xmm1").len(), 6);
}

#[test]
#[cfg(feature = "x86_64")]
fn avx512_variable_shifts() {
    assert_eq!(asm("vpsllvd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpsllvq zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpsrlvd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpsrlvq zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpsravd zmm0, zmm1, zmm2").len(), 6);
    assert_eq!(asm("vpsravq zmm0, zmm1, zmm2").len(), 6);
}

// vinsertf32x4/vextractf32x4 etc. not implemented — skipped

// ============================================================================
// AVX store-only (SSE section, but VEX encoded)
// ============================================================================

// vmovntps store form not supported in this syntax — skipped

// ============================================================================
// Extended registers (exercise REX/VEX extended register bits)
// ============================================================================

#[test]
#[cfg(feature = "x86_64")]
fn avx_extended_registers() {
    // XMM8+, YMM8+ require VEX.R or VEX.B bits — 3-byte VEX = 5 bytes
    assert_eq!(asm("vaddps xmm8, xmm9, xmm10").len(), 5);
    assert_eq!(asm("vaddps ymm8, ymm9, ymm10").len(), 5);
    assert_eq!(asm("vpaddd xmm10, xmm11, xmm12").len(), 5);
    assert_eq!(asm("vmovaps xmm15, xmm14").len(), 5);
}
