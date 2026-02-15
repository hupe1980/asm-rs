//! Intermediate representation types for the assembly pipeline.
//!
//! These types represent the structured output of the parser and serve
//! as input to the encoder and linker passes.

use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::error::Span;

/// Target architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Arch {
    /// 32-bit x86 protected mode.
    X86,
    /// 64-bit x86 long mode.
    X86_64,
    /// ARM A32 (ARMv7 and below).
    Arm,
    /// ARM T32 (Thumb-2).
    Thumb,
    /// ARMv8-A 64-bit.
    Aarch64,
    /// RISC-V 32-bit.
    Rv32,
    /// RISC-V 64-bit.
    Rv64,
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arch::X86 => write!(f, "x86"),
            Arch::X86_64 => write!(f, "x86_64"),
            Arch::Arm => write!(f, "ARM"),
            Arch::Thumb => write!(f, "Thumb"),
            Arch::Aarch64 => write!(f, "AArch64"),
            Arch::Rv32 => write!(f, "RV32"),
            Arch::Rv64 => write!(f, "RV64"),
        }
    }
}

impl Arch {
    /// Convert to the error-side architecture name.
    #[must_use]
    pub fn to_arch_name(self) -> crate::error::ArchName {
        match self {
            Arch::X86 => crate::error::ArchName::X86,
            Arch::X86_64 => crate::error::ArchName::X86_64,
            Arch::Arm => crate::error::ArchName::Arm,
            Arch::Thumb => crate::error::ArchName::Thumb,
            Arch::Aarch64 => crate::error::ArchName::Aarch64,
            Arch::Rv32 => crate::error::ArchName::Rv32,
            Arch::Rv64 => crate::error::ArchName::Rv64,
        }
    }
}

/// Assembly syntax dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Syntax {
    /// Intel / NASM style: `mov rax, 1`
    Intel,
    /// AT&T / GAS style: `movq $1, %rax`
    Att,
    /// ARM Unified Assembly Language.
    Ual,
    /// RISC-V standard syntax.
    RiscV,
}

/// Optimization level for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OptLevel {
    /// No optimizations — predictable output matching reference assembler.
    None,
    /// Prefer shortest encodings (default).
    #[default]
    Size,
}

/// x86/x64 register.
///
/// Covers all general-purpose, segment, control, and SSE registers
/// for 16-bit through 64-bit modes.  Each variant encodes its own size
/// (see [`Register::size_bits`]) and register number
/// (see [`Register::base_code`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Register {
    // -- 64-bit general-purpose registers (RAX–R15) --
    /// RAX — 64-bit accumulator.
    Rax,
    /// RCX — 64-bit counter.
    Rcx,
    /// RDX — 64-bit data.
    Rdx,
    /// RBX — 64-bit base.
    Rbx,
    /// RSP — 64-bit stack pointer.
    Rsp,
    /// RBP — 64-bit frame pointer.
    Rbp,
    /// RSI — 64-bit source index.
    Rsi,
    /// RDI — 64-bit destination index.
    Rdi,
    /// R8–R15 — extended 64-bit registers (require REX prefix).
    R8,
    /// Extended 64-bit register.
    R9,
    /// Extended 64-bit register.
    R10,
    /// Extended 64-bit register.
    R11,
    /// Extended 64-bit register.
    R12,
    /// Extended 64-bit register.
    R13,
    /// Extended 64-bit register.
    R14,
    /// Extended 64-bit register.
    R15,
    // -- 32-bit general-purpose registers --
    /// EAX — 32-bit accumulator.
    Eax,
    /// ECX — 32-bit counter.
    Ecx,
    /// EDX — 32-bit data.
    Edx,
    /// EBX — 32-bit base.
    Ebx,
    /// ESP — 32-bit stack pointer.
    Esp,
    /// EBP — 32-bit frame pointer.
    Ebp,
    /// ESI — 32-bit source index.
    Esi,
    /// EDI — 32-bit destination index.
    Edi,
    /// R8D–R15D — low 32 bits of extended registers.
    R8d,
    /// Low 32 bits of R9.
    R9d,
    /// Low 32 bits of R10.
    R10d,
    /// Low 32 bits of R11.
    R11d,
    /// Low 32 bits of R12.
    R12d,
    /// Low 32 bits of R13.
    R13d,
    /// Low 32 bits of R14.
    R14d,
    /// Low 32 bits of R15.
    R15d,
    // -- 16-bit general-purpose registers --
    /// AX — 16-bit accumulator.
    Ax,
    /// CX — 16-bit counter.
    Cx,
    /// DX — 16-bit data.
    Dx,
    /// BX — 16-bit base.
    Bx,
    /// SP — 16-bit stack pointer.
    Sp,
    /// BP — 16-bit frame pointer.
    Bp,
    /// SI — 16-bit source index.
    Si,
    /// DI — 16-bit destination index.
    Di,
    /// R8W–R15W — low 16 bits of extended registers.
    R8w,
    /// Low 16 bits of R9.
    R9w,
    /// Low 16 bits of R10.
    R10w,
    /// Low 16 bits of R11.
    R11w,
    /// Low 16 bits of R12.
    R12w,
    /// Low 16 bits of R13.
    R13w,
    /// Low 16 bits of R14.
    R14w,
    /// Low 16 bits of R15.
    R15w,
    // -- 8-bit general-purpose registers --
    /// AL — low byte of RAX.
    Al,
    /// CL — low byte of RCX.
    Cl,
    /// DL — low byte of RDX.
    Dl,
    /// BL — low byte of RBX.
    Bl,
    /// SPL — low byte of RSP (requires REX).
    Spl,
    /// BPL — low byte of RBP (requires REX).
    Bpl,
    /// SIL — low byte of RSI (requires REX).
    Sil,
    /// DIL — low byte of RDI (requires REX).
    Dil,
    /// AH — high byte of AX (incompatible with REX prefix).
    Ah,
    /// CH — high byte of CX (incompatible with REX prefix).
    Ch,
    /// DH — high byte of DX (incompatible with REX prefix).
    Dh,
    /// BH — high byte of BX (incompatible with REX prefix).
    Bh,
    /// R8B–R15B — low byte of extended registers.
    R8b,
    /// Low byte of R9.
    R9b,
    /// Low byte of R10.
    R10b,
    /// Low byte of R11.
    R11b,
    /// Low byte of R12.
    R12b,
    /// Low byte of R13.
    R13b,
    /// Low byte of R14.
    R14b,
    /// Low byte of R15.
    R15b,
    // -- Instruction pointer --
    /// RIP — 64-bit instruction pointer (for RIP-relative addressing).
    Rip,
    /// EIP — 32-bit instruction pointer.
    Eip,
    // -- Segment registers --
    /// CS — code segment.
    Cs,
    /// DS — data segment.
    Ds,
    /// ES — extra segment.
    Es,
    /// FS — additional segment (used for TLS on x86-64 Linux).
    Fs,
    /// GS — additional segment (used for TLS on x86-64 Windows/macOS).
    Gs,
    /// SS — stack segment.
    Ss,
    // -- 128-bit SSE registers --
    /// XMM0 — SSE register 0.
    Xmm0,
    /// SSE register 1.
    Xmm1,
    /// SSE register 2.
    Xmm2,
    /// SSE register 3.
    Xmm3,
    /// SSE register 4.
    Xmm4,
    /// SSE register 5.
    Xmm5,
    /// SSE register 6.
    Xmm6,
    /// SSE register 7.
    Xmm7,
    /// XMM8–XMM15 — extended SSE registers (require REX prefix).
    Xmm8,
    /// Extended SSE register 9.
    Xmm9,
    /// Extended SSE register 10.
    Xmm10,
    /// Extended SSE register 11.
    Xmm11,
    /// Extended SSE register 12.
    Xmm12,
    /// Extended SSE register 13.
    Xmm13,
    /// Extended SSE register 14.
    Xmm14,
    /// Extended SSE register 15.
    Xmm15,
    // -- 256-bit AVX registers --
    /// YMM0 — AVX register 0.
    Ymm0,
    /// AVX register 1.
    Ymm1,
    /// AVX register 2.
    Ymm2,
    /// AVX register 3.
    Ymm3,
    /// AVX register 4.
    Ymm4,
    /// AVX register 5.
    Ymm5,
    /// AVX register 6.
    Ymm6,
    /// AVX register 7.
    Ymm7,
    /// YMM8–YMM15 — extended AVX registers.
    Ymm8,
    /// Extended AVX register 9.
    Ymm9,
    /// Extended AVX register 10.
    Ymm10,
    /// Extended AVX register 11.
    Ymm11,
    /// Extended AVX register 12.
    Ymm12,
    /// Extended AVX register 13.
    Ymm13,
    /// Extended AVX register 14.
    Ymm14,
    /// Extended AVX register 15.
    Ymm15,
    // -- 512-bit AVX-512 registers --
    /// ZMM0 — AVX-512 register 0.
    Zmm0,
    /// AVX-512 register 1.
    Zmm1,
    /// AVX-512 register 2.
    Zmm2,
    /// AVX-512 register 3.
    Zmm3,
    /// AVX-512 register 4.
    Zmm4,
    /// AVX-512 register 5.
    Zmm5,
    /// AVX-512 register 6.
    Zmm6,
    /// AVX-512 register 7.
    Zmm7,
    /// ZMM8–ZMM15 — extended by REX/VEX.
    Zmm8,
    /// AVX-512 register 9.
    Zmm9,
    /// AVX-512 register 10.
    Zmm10,
    /// AVX-512 register 11.
    Zmm11,
    /// AVX-512 register 12.
    Zmm12,
    /// AVX-512 register 13.
    Zmm13,
    /// AVX-512 register 14.
    Zmm14,
    /// AVX-512 register 15.
    Zmm15,
    /// ZMM16–ZMM31 — EVEX-only registers.
    Zmm16,
    /// EVEX-only AVX-512 register 17.
    Zmm17,
    /// EVEX-only AVX-512 register 18.
    Zmm18,
    /// EVEX-only AVX-512 register 19.
    Zmm19,
    /// EVEX-only AVX-512 register 20.
    Zmm20,
    /// EVEX-only AVX-512 register 21.
    Zmm21,
    /// EVEX-only AVX-512 register 22.
    Zmm22,
    /// EVEX-only AVX-512 register 23.
    Zmm23,
    /// EVEX-only AVX-512 register 24.
    Zmm24,
    /// EVEX-only AVX-512 register 25.
    Zmm25,
    /// EVEX-only AVX-512 register 26.
    Zmm26,
    /// EVEX-only AVX-512 register 27.
    Zmm27,
    /// EVEX-only AVX-512 register 28.
    Zmm28,
    /// EVEX-only AVX-512 register 29.
    Zmm29,
    /// EVEX-only AVX-512 register 30.
    Zmm30,
    /// EVEX-only AVX-512 register 31.
    Zmm31,
    // -- AVX-512 opmask registers --
    /// K0 — opmask register 0 (implicit, means "no mask").
    K0,
    /// K1 — opmask register 1.
    K1,
    /// K2 — opmask register 2.
    K2,
    /// K3 — opmask register 3.
    K3,
    /// K4 — opmask register 4.
    K4,
    /// K5 — opmask register 5.
    K5,
    /// K6 — opmask register 6.
    K6,
    /// K7 — opmask register 7.
    K7,

    // ── ARM32 general-purpose registers ─────────────────────
    /// ARM R0.
    ArmR0,
    /// ARM R1.
    ArmR1,
    /// ARM R2.
    ArmR2,
    /// ARM R3.
    ArmR3,
    /// ARM R4.
    ArmR4,
    /// ARM R5.
    ArmR5,
    /// ARM R6.
    ArmR6,
    /// ARM R7.
    ArmR7,
    /// ARM R8.
    ArmR8,
    /// ARM R9.
    ArmR9,
    /// ARM R10.
    ArmR10,
    /// ARM R11 (FP by convention).
    ArmR11,
    /// ARM R12 (IP — intra-procedure scratch).
    ArmR12,
    /// ARM R13 / SP — stack pointer.
    ArmSp,
    /// ARM R14 / LR — link register.
    ArmLr,
    /// ARM R15 / PC — program counter.
    ArmPc,
    /// ARM CPSR — current program status register.
    ArmCpsr,

    // ── AArch64 64-bit general-purpose registers ────────────
    /// AArch64 X0.
    A64X0,
    /// AArch64 X1.
    A64X1,
    /// AArch64 X2.
    A64X2,
    /// AArch64 X3.
    A64X3,
    /// AArch64 X4.
    A64X4,
    /// AArch64 X5.
    A64X5,
    /// AArch64 X6.
    A64X6,
    /// AArch64 X7.
    A64X7,
    /// AArch64 X8.
    A64X8,
    /// AArch64 X9.
    A64X9,
    /// AArch64 X10.
    A64X10,
    /// AArch64 X11.
    A64X11,
    /// AArch64 X12.
    A64X12,
    /// AArch64 X13.
    A64X13,
    /// AArch64 X14.
    A64X14,
    /// AArch64 X15.
    A64X15,
    /// AArch64 X16.
    A64X16,
    /// AArch64 X17.
    A64X17,
    /// AArch64 X18.
    A64X18,
    /// AArch64 X19.
    A64X19,
    /// AArch64 X20.
    A64X20,
    /// AArch64 X21.
    A64X21,
    /// AArch64 X22.
    A64X22,
    /// AArch64 X23.
    A64X23,
    /// AArch64 X24.
    A64X24,
    /// AArch64 X25.
    A64X25,
    /// AArch64 X26.
    A64X26,
    /// AArch64 X27.
    A64X27,
    /// AArch64 X28.
    A64X28,
    /// AArch64 X29 / FP — frame pointer.
    A64X29,
    /// AArch64 X30 / LR — link register.
    A64X30,
    /// AArch64 SP — stack pointer (64-bit).
    A64Sp,
    /// AArch64 XZR — zero register (64-bit).
    A64Xzr,

    // ── AArch64 32-bit general-purpose registers ────────────
    /// AArch64 W0.
    A64W0,
    /// AArch64 W1.
    A64W1,
    /// AArch64 W2.
    A64W2,
    /// AArch64 W3.
    A64W3,
    /// AArch64 W4.
    A64W4,
    /// AArch64 W5.
    A64W5,
    /// AArch64 W6.
    A64W6,
    /// AArch64 W7.
    A64W7,
    /// AArch64 W8.
    A64W8,
    /// AArch64 W9.
    A64W9,
    /// AArch64 W10.
    A64W10,
    /// AArch64 W11.
    A64W11,
    /// AArch64 W12.
    A64W12,
    /// AArch64 W13.
    A64W13,
    /// AArch64 W14.
    A64W14,
    /// AArch64 W15.
    A64W15,
    /// AArch64 W16.
    A64W16,
    /// AArch64 W17.
    A64W17,
    /// AArch64 W18.
    A64W18,
    /// AArch64 W19.
    A64W19,
    /// AArch64 W20.
    A64W20,
    /// AArch64 W21.
    A64W21,
    /// AArch64 W22.
    A64W22,
    /// AArch64 W23.
    A64W23,
    /// AArch64 W24.
    A64W24,
    /// AArch64 W25.
    A64W25,
    /// AArch64 W26.
    A64W26,
    /// AArch64 W27.
    A64W27,
    /// AArch64 W28.
    A64W28,
    /// AArch64 W29.
    A64W29,
    /// AArch64 W30.
    A64W30,
    /// AArch64 WZR — zero register (32-bit).
    A64Wzr,

    // ── AArch64 SIMD/FP registers (V0–V31, 128-bit) ────────
    /// AArch64 128-bit SIMD/FP register V0.
    A64V0,
    /// AArch64 128-bit SIMD/FP register V1.
    A64V1,
    /// AArch64 128-bit SIMD/FP register V2.
    A64V2,
    /// AArch64 128-bit SIMD/FP register V3.
    A64V3,
    /// AArch64 128-bit SIMD/FP register V4.
    A64V4,
    /// AArch64 128-bit SIMD/FP register V5.
    A64V5,
    /// AArch64 128-bit SIMD/FP register V6.
    A64V6,
    /// AArch64 128-bit SIMD/FP register V7.
    A64V7,
    /// AArch64 128-bit SIMD/FP register V8.
    A64V8,
    /// AArch64 128-bit SIMD/FP register V9.
    A64V9,
    /// AArch64 128-bit SIMD/FP register V10.
    A64V10,
    /// AArch64 128-bit SIMD/FP register V11.
    A64V11,
    /// AArch64 128-bit SIMD/FP register V12.
    A64V12,
    /// AArch64 128-bit SIMD/FP register V13.
    A64V13,
    /// AArch64 128-bit SIMD/FP register V14.
    A64V14,
    /// AArch64 128-bit SIMD/FP register V15.
    A64V15,
    /// AArch64 128-bit SIMD/FP register V16.
    A64V16,
    /// AArch64 128-bit SIMD/FP register V17.
    A64V17,
    /// AArch64 128-bit SIMD/FP register V18.
    A64V18,
    /// AArch64 128-bit SIMD/FP register V19.
    A64V19,
    /// AArch64 128-bit SIMD/FP register V20.
    A64V20,
    /// AArch64 128-bit SIMD/FP register V21.
    A64V21,
    /// AArch64 128-bit SIMD/FP register V22.
    A64V22,
    /// AArch64 128-bit SIMD/FP register V23.
    A64V23,
    /// AArch64 128-bit SIMD/FP register V24.
    A64V24,
    /// AArch64 128-bit SIMD/FP register V25.
    A64V25,
    /// AArch64 128-bit SIMD/FP register V26.
    A64V26,
    /// AArch64 128-bit SIMD/FP register V27.
    A64V27,
    /// AArch64 128-bit SIMD/FP register V28.
    A64V28,
    /// AArch64 128-bit SIMD/FP register V29.
    A64V29,
    /// AArch64 128-bit SIMD/FP register V30.
    A64V30,
    /// AArch64 128-bit SIMD/FP register V31.
    A64V31,

    // ── AArch64 SIMD/FP scalar registers (Q, D, S, H, B) ───
    /// AArch64 128-bit scalar quad register Q0.
    A64Q0,
    /// AArch64 128-bit scalar quad register Q1.
    A64Q1,
    /// AArch64 128-bit scalar quad register Q2.
    A64Q2,
    /// AArch64 128-bit scalar quad register Q3.
    A64Q3,
    /// AArch64 128-bit scalar quad register Q4.
    A64Q4,
    /// AArch64 128-bit scalar quad register Q5.
    A64Q5,
    /// AArch64 128-bit scalar quad register Q6.
    A64Q6,
    /// AArch64 128-bit scalar quad register Q7.
    A64Q7,
    /// AArch64 128-bit scalar quad register Q8.
    A64Q8,
    /// AArch64 128-bit scalar quad register Q9.
    A64Q9,
    /// AArch64 128-bit scalar quad register Q10.
    A64Q10,
    /// AArch64 128-bit scalar quad register Q11.
    A64Q11,
    /// AArch64 128-bit scalar quad register Q12.
    A64Q12,
    /// AArch64 128-bit scalar quad register Q13.
    A64Q13,
    /// AArch64 128-bit scalar quad register Q14.
    A64Q14,
    /// AArch64 128-bit scalar quad register Q15.
    A64Q15,
    /// AArch64 128-bit scalar quad register Q16.
    A64Q16,
    /// AArch64 128-bit scalar quad register Q17.
    A64Q17,
    /// AArch64 128-bit scalar quad register Q18.
    A64Q18,
    /// AArch64 128-bit scalar quad register Q19.
    A64Q19,
    /// AArch64 128-bit scalar quad register Q20.
    A64Q20,
    /// AArch64 128-bit scalar quad register Q21.
    A64Q21,
    /// AArch64 128-bit scalar quad register Q22.
    A64Q22,
    /// AArch64 128-bit scalar quad register Q23.
    A64Q23,
    /// AArch64 128-bit scalar quad register Q24.
    A64Q24,
    /// AArch64 128-bit scalar quad register Q25.
    A64Q25,
    /// AArch64 128-bit scalar quad register Q26.
    A64Q26,
    /// AArch64 128-bit scalar quad register Q27.
    A64Q27,
    /// AArch64 128-bit scalar quad register Q28.
    A64Q28,
    /// AArch64 128-bit scalar quad register Q29.
    A64Q29,
    /// AArch64 128-bit scalar quad register Q30.
    A64Q30,
    /// AArch64 128-bit scalar quad register Q31.
    A64Q31,
    /// AArch64 64-bit scalar double register D0.
    A64D0,
    /// AArch64 64-bit scalar double register D1.
    A64D1,
    /// AArch64 64-bit scalar double register D2.
    A64D2,
    /// AArch64 64-bit scalar double register D3.
    A64D3,
    /// AArch64 64-bit scalar double register D4.
    A64D4,
    /// AArch64 64-bit scalar double register D5.
    A64D5,
    /// AArch64 64-bit scalar double register D6.
    A64D6,
    /// AArch64 64-bit scalar double register D7.
    A64D7,
    /// AArch64 64-bit scalar double register D8.
    A64D8,
    /// AArch64 64-bit scalar double register D9.
    A64D9,
    /// AArch64 64-bit scalar double register D10.
    A64D10,
    /// AArch64 64-bit scalar double register D11.
    A64D11,
    /// AArch64 64-bit scalar double register D12.
    A64D12,
    /// AArch64 64-bit scalar double register D13.
    A64D13,
    /// AArch64 64-bit scalar double register D14.
    A64D14,
    /// AArch64 64-bit scalar double register D15.
    A64D15,
    /// AArch64 64-bit scalar double register D16.
    A64D16,
    /// AArch64 64-bit scalar double register D17.
    A64D17,
    /// AArch64 64-bit scalar double register D18.
    A64D18,
    /// AArch64 64-bit scalar double register D19.
    A64D19,
    /// AArch64 64-bit scalar double register D20.
    A64D20,
    /// AArch64 64-bit scalar double register D21.
    A64D21,
    /// AArch64 64-bit scalar double register D22.
    A64D22,
    /// AArch64 64-bit scalar double register D23.
    A64D23,
    /// AArch64 64-bit scalar double register D24.
    A64D24,
    /// AArch64 64-bit scalar double register D25.
    A64D25,
    /// AArch64 64-bit scalar double register D26.
    A64D26,
    /// AArch64 64-bit scalar double register D27.
    A64D27,
    /// AArch64 64-bit scalar double register D28.
    A64D28,
    /// AArch64 64-bit scalar double register D29.
    A64D29,
    /// AArch64 64-bit scalar double register D30.
    A64D30,
    /// AArch64 64-bit scalar double register D31.
    A64D31,
    /// AArch64 32-bit scalar single register S0.
    A64S0,
    /// AArch64 32-bit scalar single register S1.
    A64S1,
    /// AArch64 32-bit scalar single register S2.
    A64S2,
    /// AArch64 32-bit scalar single register S3.
    A64S3,
    /// AArch64 32-bit scalar single register S4.
    A64S4,
    /// AArch64 32-bit scalar single register S5.
    A64S5,
    /// AArch64 32-bit scalar single register S6.
    A64S6,
    /// AArch64 32-bit scalar single register S7.
    A64S7,
    /// AArch64 32-bit scalar single register S8.
    A64S8,
    /// AArch64 32-bit scalar single register S9.
    A64S9,
    /// AArch64 32-bit scalar single register S10.
    A64S10,
    /// AArch64 32-bit scalar single register S11.
    A64S11,
    /// AArch64 32-bit scalar single register S12.
    A64S12,
    /// AArch64 32-bit scalar single register S13.
    A64S13,
    /// AArch64 32-bit scalar single register S14.
    A64S14,
    /// AArch64 32-bit scalar single register S15.
    A64S15,
    /// AArch64 32-bit scalar single register S16.
    A64S16,
    /// AArch64 32-bit scalar single register S17.
    A64S17,
    /// AArch64 32-bit scalar single register S18.
    A64S18,
    /// AArch64 32-bit scalar single register S19.
    A64S19,
    /// AArch64 32-bit scalar single register S20.
    A64S20,
    /// AArch64 32-bit scalar single register S21.
    A64S21,
    /// AArch64 32-bit scalar single register S22.
    A64S22,
    /// AArch64 32-bit scalar single register S23.
    A64S23,
    /// AArch64 32-bit scalar single register S24.
    A64S24,
    /// AArch64 32-bit scalar single register S25.
    A64S25,
    /// AArch64 32-bit scalar single register S26.
    A64S26,
    /// AArch64 32-bit scalar single register S27.
    A64S27,
    /// AArch64 32-bit scalar single register S28.
    A64S28,
    /// AArch64 32-bit scalar single register S29.
    A64S29,
    /// AArch64 32-bit scalar single register S30.
    A64S30,
    /// AArch64 32-bit scalar single register S31.
    A64S31,
    /// AArch64 16-bit scalar half register H0.
    A64H0,
    /// AArch64 16-bit scalar half register H1.
    A64H1,
    /// AArch64 16-bit scalar half register H2.
    A64H2,
    /// AArch64 16-bit scalar half register H3.
    A64H3,
    /// AArch64 16-bit scalar half register H4.
    A64H4,
    /// AArch64 16-bit scalar half register H5.
    A64H5,
    /// AArch64 16-bit scalar half register H6.
    A64H6,
    /// AArch64 16-bit scalar half register H7.
    A64H7,
    /// AArch64 16-bit scalar half register H8.
    A64H8,
    /// AArch64 16-bit scalar half register H9.
    A64H9,
    /// AArch64 16-bit scalar half register H10.
    A64H10,
    /// AArch64 16-bit scalar half register H11.
    A64H11,
    /// AArch64 16-bit scalar half register H12.
    A64H12,
    /// AArch64 16-bit scalar half register H13.
    A64H13,
    /// AArch64 16-bit scalar half register H14.
    A64H14,
    /// AArch64 16-bit scalar half register H15.
    A64H15,
    /// AArch64 16-bit scalar half register H16.
    A64H16,
    /// AArch64 16-bit scalar half register H17.
    A64H17,
    /// AArch64 16-bit scalar half register H18.
    A64H18,
    /// AArch64 16-bit scalar half register H19.
    A64H19,
    /// AArch64 16-bit scalar half register H20.
    A64H20,
    /// AArch64 16-bit scalar half register H21.
    A64H21,
    /// AArch64 16-bit scalar half register H22.
    A64H22,
    /// AArch64 16-bit scalar half register H23.
    A64H23,
    /// AArch64 16-bit scalar half register H24.
    A64H24,
    /// AArch64 16-bit scalar half register H25.
    A64H25,
    /// AArch64 16-bit scalar half register H26.
    A64H26,
    /// AArch64 16-bit scalar half register H27.
    A64H27,
    /// AArch64 16-bit scalar half register H28.
    A64H28,
    /// AArch64 16-bit scalar half register H29.
    A64H29,
    /// AArch64 16-bit scalar half register H30.
    A64H30,
    /// AArch64 16-bit scalar half register H31.
    A64H31,
    /// AArch64 8-bit scalar byte register B0.
    A64B0,
    /// AArch64 8-bit scalar byte register B1.
    A64B1,
    /// AArch64 8-bit scalar byte register B2.
    A64B2,
    /// AArch64 8-bit scalar byte register B3.
    A64B3,
    /// AArch64 8-bit scalar byte register B4.
    A64B4,
    /// AArch64 8-bit scalar byte register B5.
    A64B5,
    /// AArch64 8-bit scalar byte register B6.
    A64B6,
    /// AArch64 8-bit scalar byte register B7.
    A64B7,
    /// AArch64 8-bit scalar byte register B8.
    A64B8,
    /// AArch64 8-bit scalar byte register B9.
    A64B9,
    /// AArch64 8-bit scalar byte register B10.
    A64B10,
    /// AArch64 8-bit scalar byte register B11.
    A64B11,
    /// AArch64 8-bit scalar byte register B12.
    A64B12,
    /// AArch64 8-bit scalar byte register B13.
    A64B13,
    /// AArch64 8-bit scalar byte register B14.
    A64B14,
    /// AArch64 8-bit scalar byte register B15.
    A64B15,
    /// AArch64 8-bit scalar byte register B16.
    A64B16,
    /// AArch64 8-bit scalar byte register B17.
    A64B17,
    /// AArch64 8-bit scalar byte register B18.
    A64B18,
    /// AArch64 8-bit scalar byte register B19.
    A64B19,
    /// AArch64 8-bit scalar byte register B20.
    A64B20,
    /// AArch64 8-bit scalar byte register B21.
    A64B21,
    /// AArch64 8-bit scalar byte register B22.
    A64B22,
    /// AArch64 8-bit scalar byte register B23.
    A64B23,
    /// AArch64 8-bit scalar byte register B24.
    A64B24,
    /// AArch64 8-bit scalar byte register B25.
    A64B25,
    /// AArch64 8-bit scalar byte register B26.
    A64B26,
    /// AArch64 8-bit scalar byte register B27.
    A64B27,
    /// AArch64 8-bit scalar byte register B28.
    A64B28,
    /// AArch64 8-bit scalar byte register B29.
    A64B29,
    /// AArch64 8-bit scalar byte register B30.
    A64B30,
    /// AArch64 8-bit scalar byte register B31.
    A64B31,

    // ── AArch64 SVE scalable vector registers (Z0–Z31) ──────
    /// AArch64 SVE scalable vector register Z0.
    A64Z0,
    /// AArch64 SVE scalable vector register Z1.
    A64Z1,
    /// AArch64 SVE scalable vector register Z2.
    A64Z2,
    /// AArch64 SVE scalable vector register Z3.
    A64Z3,
    /// AArch64 SVE scalable vector register Z4.
    A64Z4,
    /// AArch64 SVE scalable vector register Z5.
    A64Z5,
    /// AArch64 SVE scalable vector register Z6.
    A64Z6,
    /// AArch64 SVE scalable vector register Z7.
    A64Z7,
    /// AArch64 SVE scalable vector register Z8.
    A64Z8,
    /// AArch64 SVE scalable vector register Z9.
    A64Z9,
    /// AArch64 SVE scalable vector register Z10.
    A64Z10,
    /// AArch64 SVE scalable vector register Z11.
    A64Z11,
    /// AArch64 SVE scalable vector register Z12.
    A64Z12,
    /// AArch64 SVE scalable vector register Z13.
    A64Z13,
    /// AArch64 SVE scalable vector register Z14.
    A64Z14,
    /// AArch64 SVE scalable vector register Z15.
    A64Z15,
    /// AArch64 SVE scalable vector register Z16.
    A64Z16,
    /// AArch64 SVE scalable vector register Z17.
    A64Z17,
    /// AArch64 SVE scalable vector register Z18.
    A64Z18,
    /// AArch64 SVE scalable vector register Z19.
    A64Z19,
    /// AArch64 SVE scalable vector register Z20.
    A64Z20,
    /// AArch64 SVE scalable vector register Z21.
    A64Z21,
    /// AArch64 SVE scalable vector register Z22.
    A64Z22,
    /// AArch64 SVE scalable vector register Z23.
    A64Z23,
    /// AArch64 SVE scalable vector register Z24.
    A64Z24,
    /// AArch64 SVE scalable vector register Z25.
    A64Z25,
    /// AArch64 SVE scalable vector register Z26.
    A64Z26,
    /// AArch64 SVE scalable vector register Z27.
    A64Z27,
    /// AArch64 SVE scalable vector register Z28.
    A64Z28,
    /// AArch64 SVE scalable vector register Z29.
    A64Z29,
    /// AArch64 SVE scalable vector register Z30.
    A64Z30,
    /// AArch64 SVE scalable vector register Z31.
    A64Z31,

    // ── AArch64 SVE predicate registers (P0–P15) ────────────
    /// AArch64 SVE predicate register P0.
    A64P0,
    /// AArch64 SVE predicate register P1.
    A64P1,
    /// AArch64 SVE predicate register P2.
    A64P2,
    /// AArch64 SVE predicate register P3.
    A64P3,
    /// AArch64 SVE predicate register P4.
    A64P4,
    /// AArch64 SVE predicate register P5.
    A64P5,
    /// AArch64 SVE predicate register P6.
    A64P6,
    /// AArch64 SVE predicate register P7.
    A64P7,
    /// AArch64 SVE predicate register P8.
    A64P8,
    /// AArch64 SVE predicate register P9.
    A64P9,
    /// AArch64 SVE predicate register P10.
    A64P10,
    /// AArch64 SVE predicate register P11.
    A64P11,
    /// AArch64 SVE predicate register P12.
    A64P12,
    /// AArch64 SVE predicate register P13.
    A64P13,
    /// AArch64 SVE predicate register P14.
    A64P14,
    /// AArch64 SVE predicate register P15.
    A64P15,

    // ── RISC-V integer registers (x0–x31) ───────────────────
    /// RISC-V x0 / zero — hardwired zero.
    RvX0,
    /// RISC-V x1 / ra — return address.
    RvX1,
    /// RISC-V x2 / sp — stack pointer.
    RvX2,
    /// RISC-V x3 / gp — global pointer.
    RvX3,
    /// RISC-V x4 / tp — thread pointer.
    RvX4,
    /// RISC-V x5 / t0 — temporary.
    RvX5,
    /// RISC-V x6 / t1 — temporary.
    RvX6,
    /// RISC-V x7 / t2 — temporary.
    RvX7,
    /// RISC-V x8 / s0 / fp — saved / frame pointer.
    RvX8,
    /// RISC-V x9 / s1 — saved register.
    RvX9,
    /// RISC-V x10 / a0 — argument / return value.
    RvX10,
    /// RISC-V x11 / a1 — argument / return value.
    RvX11,
    /// RISC-V x12 / a2 — argument.
    RvX12,
    /// RISC-V x13 / a3 — argument.
    RvX13,
    /// RISC-V x14 / a4 — argument.
    RvX14,
    /// RISC-V x15 / a5 — argument.
    RvX15,
    /// RISC-V x16 / a6 — argument.
    RvX16,
    /// RISC-V x17 / a7 — argument.
    RvX17,
    /// RISC-V x18 / s2 — saved register.
    RvX18,
    /// RISC-V x19 / s3 — saved register.
    RvX19,
    /// RISC-V x20 / s4 — saved register.
    RvX20,
    /// RISC-V x21 / s5 — saved register.
    RvX21,
    /// RISC-V x22 / s6 — saved register.
    RvX22,
    /// RISC-V x23 / s7 — saved register.
    RvX23,
    /// RISC-V x24 / s8 — saved register.
    RvX24,
    /// RISC-V x25 / s9 — saved register.
    RvX25,
    /// RISC-V x26 / s10 — saved register.
    RvX26,
    /// RISC-V x27 / s11 — saved register.
    RvX27,
    /// RISC-V x28 / t3 — temporary.
    RvX28,
    /// RISC-V x29 / t4 — temporary.
    RvX29,
    /// RISC-V x30 / t5 — temporary.
    RvX30,
    /// RISC-V x31 / t6 — temporary.
    RvX31,

    // ── RISC-V floating-point registers (f0–f31) ────────────
    /// RISC-V f0 / ft0 — FP temporary.
    RvF0,
    /// RISC-V f1 / ft1 — FP temporary.
    RvF1,
    /// RISC-V f2 / ft2 — FP temporary.
    RvF2,
    /// RISC-V f3 / ft3 — FP temporary.
    RvF3,
    /// RISC-V f4 / ft4 — FP temporary.
    RvF4,
    /// RISC-V f5 / ft5 — FP temporary.
    RvF5,
    /// RISC-V f6 / ft6 — FP temporary.
    RvF6,
    /// RISC-V f7 / ft7 — FP temporary.
    RvF7,
    /// RISC-V f8 / fs0 — FP saved register.
    RvF8,
    /// RISC-V f9 / fs1 — FP saved register.
    RvF9,
    /// RISC-V f10 / fa0 — FP argument/return value.
    RvF10,
    /// RISC-V f11 / fa1 — FP argument/return value.
    RvF11,
    /// RISC-V f12 / fa2 — FP argument.
    RvF12,
    /// RISC-V f13 / fa3 — FP argument.
    RvF13,
    /// RISC-V f14 / fa4 — FP argument.
    RvF14,
    /// RISC-V f15 / fa5 — FP argument.
    RvF15,
    /// RISC-V f16 / fa6 — FP argument.
    RvF16,
    /// RISC-V f17 / fa7 — FP argument.
    RvF17,
    /// RISC-V f18 / fs2 — FP saved register.
    RvF18,
    /// RISC-V f19 / fs3 — FP saved register.
    RvF19,
    /// RISC-V f20 / fs4 — FP saved register.
    RvF20,
    /// RISC-V f21 / fs5 — FP saved register.
    RvF21,
    /// RISC-V f22 / fs6 — FP saved register.
    RvF22,
    /// RISC-V f23 / fs7 — FP saved register.
    RvF23,
    /// RISC-V f24 / fs8 — FP saved register.
    RvF24,
    /// RISC-V f25 / fs9 — FP saved register.
    RvF25,
    /// RISC-V f26 / fs10 — FP saved register.
    RvF26,
    /// RISC-V f27 / fs11 — FP saved register.
    RvF27,
    /// RISC-V f28 / ft8 — FP temporary.
    RvF28,
    /// RISC-V f29 / ft9 — FP temporary.
    RvF29,
    /// RISC-V f30 / ft10 — FP temporary.
    RvF30,
    /// RISC-V f31 / ft11 — FP temporary.
    RvF31,

    // ── RISC-V vector registers (v0–v31) ────────────────────
    /// RISC-V vector register v0 (also used as mask).
    RvV0,
    /// RISC-V vector register v1.
    RvV1,
    /// RISC-V vector register v2.
    RvV2,
    /// RISC-V vector register v3.
    RvV3,
    /// RISC-V vector register v4.
    RvV4,
    /// RISC-V vector register v5.
    RvV5,
    /// RISC-V vector register v6.
    RvV6,
    /// RISC-V vector register v7.
    RvV7,
    /// RISC-V vector register v8.
    RvV8,
    /// RISC-V vector register v9.
    RvV9,
    /// RISC-V vector register v10.
    RvV10,
    /// RISC-V vector register v11.
    RvV11,
    /// RISC-V vector register v12.
    RvV12,
    /// RISC-V vector register v13.
    RvV13,
    /// RISC-V vector register v14.
    RvV14,
    /// RISC-V vector register v15.
    RvV15,
    /// RISC-V vector register v16.
    RvV16,
    /// RISC-V vector register v17.
    RvV17,
    /// RISC-V vector register v18.
    RvV18,
    /// RISC-V vector register v19.
    RvV19,
    /// RISC-V vector register v20.
    RvV20,
    /// RISC-V vector register v21.
    RvV21,
    /// RISC-V vector register v22.
    RvV22,
    /// RISC-V vector register v23.
    RvV23,
    /// RISC-V vector register v24.
    RvV24,
    /// RISC-V vector register v25.
    RvV25,
    /// RISC-V vector register v26.
    RvV26,
    /// RISC-V vector register v27.
    RvV27,
    /// RISC-V vector register v28.
    RvV28,
    /// RISC-V vector register v29.
    RvV29,
    /// RISC-V vector register v30.
    RvV30,
    /// RISC-V vector register v31.
    RvV31,
}

impl Register {
    /// The 3-bit register encoding (bits 0-2 of the register number).
    pub fn base_code(self) -> u8 {
        use Register::*;
        match self {
            Rax | Eax | Ax | Al | R8 | R8d | R8w | R8b | Xmm0 | Xmm8 | Ymm0 | Ymm8 | Zmm0
            | Zmm8 | Zmm16 | Zmm24 | K0 => 0,
            Rcx | Ecx | Cx | Cl | R9 | R9d | R9w | R9b | Xmm1 | Xmm9 | Ymm1 | Ymm9 | Zmm1
            | Zmm9 | Zmm17 | Zmm25 | K1 => 1,
            Rdx | Edx | Dx | Dl | R10 | R10d | R10w | R10b | Xmm2 | Xmm10 | Ymm2 | Ymm10 | Zmm2
            | Zmm10 | Zmm18 | Zmm26 | K2 => 2,
            Rbx | Ebx | Bx | Bl | R11 | R11d | R11w | R11b | Xmm3 | Xmm11 | Ymm3 | Ymm11 | Zmm3
            | Zmm11 | Zmm19 | Zmm27 | K3 => 3,
            Rsp | Esp | Sp | Spl | Ah | R12 | R12d | R12w | R12b | Xmm4 | Xmm12 | Ymm4 | Ymm12
            | Zmm4 | Zmm12 | Zmm20 | Zmm28 | K4 => 4,
            Rbp | Ebp | Bp | Bpl | Ch | R13 | R13d | R13w | R13b | Xmm5 | Xmm13 | Ymm5 | Ymm13
            | Zmm5 | Zmm13 | Zmm21 | Zmm29 | K5 => 5,
            Rsi | Esi | Si | Sil | Dh | R14 | R14d | R14w | R14b | Xmm6 | Xmm14 | Ymm6 | Ymm14
            | Zmm6 | Zmm14 | Zmm22 | Zmm30 | K6 => 6,
            Rdi | Edi | Di | Dil | Bh | R15 | R15d | R15w | R15b | Xmm7 | Xmm15 | Ymm7 | Ymm15
            | Zmm7 | Zmm15 | Zmm23 | Zmm31 | K7 => 7,
            Rip | Eip => 5, // RIP-relative uses encoding 5 (mod=00, rm=101)
            Cs => 1,
            Ds => 3,
            Es => 0,
            Fs => 4,
            Gs => 5,
            Ss => 2,
            // ARM/AArch64 registers don't use x86 base_code — see arm_reg_num / a64_reg_num
            _ => 0,
        }
    }

    /// Whether this is an extended register (R8–R15, Xmm8–Xmm15, Ymm8–Ymm15, Zmm8–Zmm15, Zmm24–Zmm31)
    /// requiring REX/VEX.R or REX/VEX.B (bit 3 of the register index).
    pub fn is_extended(self) -> bool {
        use Register::*;
        matches!(
            self,
            R8 | R9
                | R10
                | R11
                | R12
                | R13
                | R14
                | R15
                | R8d
                | R9d
                | R10d
                | R11d
                | R12d
                | R13d
                | R14d
                | R15d
                | R8w
                | R9w
                | R10w
                | R11w
                | R12w
                | R13w
                | R14w
                | R15w
                | R8b
                | R9b
                | R10b
                | R11b
                | R12b
                | R13b
                | R14b
                | R15b
                | Xmm8
                | Xmm9
                | Xmm10
                | Xmm11
                | Xmm12
                | Xmm13
                | Xmm14
                | Xmm15
                | Ymm8
                | Ymm9
                | Ymm10
                | Ymm11
                | Ymm12
                | Ymm13
                | Ymm14
                | Ymm15
                | Zmm8
                | Zmm9
                | Zmm10
                | Zmm11
                | Zmm12
                | Zmm13
                | Zmm14
                | Zmm15
                | Zmm24
                | Zmm25
                | Zmm26
                | Zmm27
                | Zmm28
                | Zmm29
                | Zmm30
                | Zmm31
        )
    }

    /// Whether this register requires EVEX (ZMM16–ZMM31).
    /// These need the EVEX.R' and/or EVEX.V' bits.
    pub fn is_evex_extended(self) -> bool {
        use Register::*;
        matches!(
            self,
            Zmm16
                | Zmm17
                | Zmm18
                | Zmm19
                | Zmm20
                | Zmm21
                | Zmm22
                | Zmm23
                | Zmm24
                | Zmm25
                | Zmm26
                | Zmm27
                | Zmm28
                | Zmm29
                | Zmm30
                | Zmm31
        )
    }

    /// Size of the register in bits.
    pub fn size_bits(self) -> u16 {
        use Register::*;
        match self {
            Rax | Rcx | Rdx | Rbx | Rsp | Rbp | Rsi | Rdi | R8 | R9 | R10 | R11 | R12 | R13
            | R14 | R15 | Rip => 64,
            Eax | Ecx | Edx | Ebx | Esp | Ebp | Esi | Edi | R8d | R9d | R10d | R11d | R12d
            | R13d | R14d | R15d | Eip => 32,
            Ax | Cx | Dx | Bx | Sp | Bp | Si | Di | R8w | R9w | R10w | R11w | R12w | R13w
            | R14w | R15w => 16,
            Al | Cl | Dl | Bl | Spl | Bpl | Sil | Dil | Ah | Ch | Dh | Bh | R8b | R9b | R10b
            | R11b | R12b | R13b | R14b | R15b => 8,
            Cs | Ds | Es | Fs | Gs | Ss => 16,
            Xmm0 | Xmm1 | Xmm2 | Xmm3 | Xmm4 | Xmm5 | Xmm6 | Xmm7 | Xmm8 | Xmm9 | Xmm10 | Xmm11
            | Xmm12 | Xmm13 | Xmm14 | Xmm15 => 128,
            Ymm0 | Ymm1 | Ymm2 | Ymm3 | Ymm4 | Ymm5 | Ymm6 | Ymm7 | Ymm8 | Ymm9 | Ymm10 | Ymm11
            | Ymm12 | Ymm13 | Ymm14 | Ymm15 => 256,
            Zmm0 | Zmm1 | Zmm2 | Zmm3 | Zmm4 | Zmm5 | Zmm6 | Zmm7 | Zmm8 | Zmm9 | Zmm10 | Zmm11
            | Zmm12 | Zmm13 | Zmm14 | Zmm15 | Zmm16 | Zmm17 | Zmm18 | Zmm19 | Zmm20 | Zmm21
            | Zmm22 | Zmm23 | Zmm24 | Zmm25 | Zmm26 | Zmm27 | Zmm28 | Zmm29 | Zmm30 | Zmm31 => 512,
            K0 | K1 | K2 | K3 | K4 | K5 | K6 | K7 => 64,
            // ARM32 registers
            ArmR0 | ArmR1 | ArmR2 | ArmR3 | ArmR4 | ArmR5 | ArmR6 | ArmR7 | ArmR8 | ArmR9
            | ArmR10 | ArmR11 | ArmR12 | ArmSp | ArmLr | ArmPc | ArmCpsr => 32,
            // AArch64 64-bit registers
            A64X0 | A64X1 | A64X2 | A64X3 | A64X4 | A64X5 | A64X6 | A64X7 | A64X8 | A64X9
            | A64X10 | A64X11 | A64X12 | A64X13 | A64X14 | A64X15 | A64X16 | A64X17 | A64X18
            | A64X19 | A64X20 | A64X21 | A64X22 | A64X23 | A64X24 | A64X25 | A64X26 | A64X27
            | A64X28 | A64X29 | A64X30 | A64Sp | A64Xzr => 64,
            // AArch64 32-bit registers
            A64W0 | A64W1 | A64W2 | A64W3 | A64W4 | A64W5 | A64W6 | A64W7 | A64W8 | A64W9
            | A64W10 | A64W11 | A64W12 | A64W13 | A64W14 | A64W15 | A64W16 | A64W17 | A64W18
            | A64W19 | A64W20 | A64W21 | A64W22 | A64W23 | A64W24 | A64W25 | A64W26 | A64W27
            | A64W28 | A64W29 | A64W30 | A64Wzr => 32,
            // AArch64 SIMD/FP registers — V and Q are 128-bit
            A64V0 | A64V1 | A64V2 | A64V3 | A64V4 | A64V5 | A64V6 | A64V7 | A64V8 | A64V9
            | A64V10 | A64V11 | A64V12 | A64V13 | A64V14 | A64V15 | A64V16 | A64V17 | A64V18
            | A64V19 | A64V20 | A64V21 | A64V22 | A64V23 | A64V24 | A64V25 | A64V26 | A64V27
            | A64V28 | A64V29 | A64V30 | A64V31 | A64Q0 | A64Q1 | A64Q2 | A64Q3 | A64Q4 | A64Q5
            | A64Q6 | A64Q7 | A64Q8 | A64Q9 | A64Q10 | A64Q11 | A64Q12 | A64Q13 | A64Q14
            | A64Q15 | A64Q16 | A64Q17 | A64Q18 | A64Q19 | A64Q20 | A64Q21 | A64Q22 | A64Q23
            | A64Q24 | A64Q25 | A64Q26 | A64Q27 | A64Q28 | A64Q29 | A64Q30 | A64Q31 => 128,
            // AArch64 SIMD/FP D registers — 64-bit
            A64D0 | A64D1 | A64D2 | A64D3 | A64D4 | A64D5 | A64D6 | A64D7 | A64D8 | A64D9
            | A64D10 | A64D11 | A64D12 | A64D13 | A64D14 | A64D15 | A64D16 | A64D17 | A64D18
            | A64D19 | A64D20 | A64D21 | A64D22 | A64D23 | A64D24 | A64D25 | A64D26 | A64D27
            | A64D28 | A64D29 | A64D30 | A64D31 => 64,
            // AArch64 SIMD/FP S registers — 32-bit
            A64S0 | A64S1 | A64S2 | A64S3 | A64S4 | A64S5 | A64S6 | A64S7 | A64S8 | A64S9
            | A64S10 | A64S11 | A64S12 | A64S13 | A64S14 | A64S15 | A64S16 | A64S17 | A64S18
            | A64S19 | A64S20 | A64S21 | A64S22 | A64S23 | A64S24 | A64S25 | A64S26 | A64S27
            | A64S28 | A64S29 | A64S30 | A64S31 => 32,
            // AArch64 SIMD/FP H registers — 16-bit
            A64H0 | A64H1 | A64H2 | A64H3 | A64H4 | A64H5 | A64H6 | A64H7 | A64H8 | A64H9
            | A64H10 | A64H11 | A64H12 | A64H13 | A64H14 | A64H15 | A64H16 | A64H17 | A64H18
            | A64H19 | A64H20 | A64H21 | A64H22 | A64H23 | A64H24 | A64H25 | A64H26 | A64H27
            | A64H28 | A64H29 | A64H30 | A64H31 => 16,
            // AArch64 SIMD/FP B registers — 8-bit
            A64B0 | A64B1 | A64B2 | A64B3 | A64B4 | A64B5 | A64B6 | A64B7 | A64B8 | A64B9
            | A64B10 | A64B11 | A64B12 | A64B13 | A64B14 | A64B15 | A64B16 | A64B17 | A64B18
            | A64B19 | A64B20 | A64B21 | A64B22 | A64B23 | A64B24 | A64B25 | A64B26 | A64B27
            | A64B28 | A64B29 | A64B30 | A64B31 => 8,
            // AArch64 SVE Z registers — scalable width, report 0
            A64Z0 | A64Z1 | A64Z2 | A64Z3 | A64Z4 | A64Z5 | A64Z6 | A64Z7 | A64Z8 | A64Z9
            | A64Z10 | A64Z11 | A64Z12 | A64Z13 | A64Z14 | A64Z15 | A64Z16 | A64Z17 | A64Z18
            | A64Z19 | A64Z20 | A64Z21 | A64Z22 | A64Z23 | A64Z24 | A64Z25 | A64Z26 | A64Z27
            | A64Z28 | A64Z29 | A64Z30 | A64Z31 => 0,
            // AArch64 SVE predicate registers — scalable width, report 0
            A64P0 | A64P1 | A64P2 | A64P3 | A64P4 | A64P5 | A64P6 | A64P7 | A64P8 | A64P9
            | A64P10 | A64P11 | A64P12 | A64P13 | A64P14 | A64P15 => 0,
            // RISC-V registers — size depends on XLEN (32 or 64), we report 0 here
            // and let the encoder determine width based on Arch::Rv32 vs Rv64.
            RvX0 | RvX1 | RvX2 | RvX3 | RvX4 | RvX5 | RvX6 | RvX7 | RvX8 | RvX9 | RvX10 | RvX11
            | RvX12 | RvX13 | RvX14 | RvX15 | RvX16 | RvX17 | RvX18 | RvX19 | RvX20 | RvX21
            | RvX22 | RvX23 | RvX24 | RvX25 | RvX26 | RvX27 | RvX28 | RvX29 | RvX30 | RvX31 => 0,
            // RISC-V FP registers — size depends on F/D/Q extension; report 0
            // and let the encoder determine width from the instruction.
            RvF0 | RvF1 | RvF2 | RvF3 | RvF4 | RvF5 | RvF6 | RvF7 | RvF8 | RvF9 | RvF10 | RvF11
            | RvF12 | RvF13 | RvF14 | RvF15 | RvF16 | RvF17 | RvF18 | RvF19 | RvF20 | RvF21
            | RvF22 | RvF23 | RvF24 | RvF25 | RvF26 | RvF27 | RvF28 | RvF29 | RvF30 | RvF31 => 0,
            // RISC-V vector registers — scalable width, report 0
            RvV0 | RvV1 | RvV2 | RvV3 | RvV4 | RvV5 | RvV6 | RvV7 | RvV8 | RvV9 | RvV10 | RvV11
            | RvV12 | RvV13 | RvV14 | RvV15 | RvV16 | RvV17 | RvV18 | RvV19 | RvV20 | RvV21
            | RvV22 | RvV23 | RvV24 | RvV25 | RvV26 | RvV27 | RvV28 | RvV29 | RvV30 | RvV31 => 0,
        }
    }

    /// Whether this register requires a REX prefix to be addressable as an 8-bit register.
    /// SPL, BPL, SIL, DIL need REX (even if REX.{W,R,X,B} are all 0).
    pub fn requires_rex_for_byte(self) -> bool {
        use Register::*;
        matches!(self, Spl | Bpl | Sil | Dil)
    }

    /// Whether this is a high-byte register (AH, CH, DH, BH).
    /// These cannot be used with REX prefix.
    pub fn is_high_byte(self) -> bool {
        use Register::*;
        matches!(self, Ah | Ch | Dh | Bh)
    }

    /// Whether this is an XMM (SSE) register.
    #[must_use]
    pub fn is_xmm(self) -> bool {
        use Register::*;
        matches!(
            self,
            Xmm0 | Xmm1
                | Xmm2
                | Xmm3
                | Xmm4
                | Xmm5
                | Xmm6
                | Xmm7
                | Xmm8
                | Xmm9
                | Xmm10
                | Xmm11
                | Xmm12
                | Xmm13
                | Xmm14
                | Xmm15
        )
    }

    /// Whether this is a YMM (AVX) register.
    #[must_use]
    pub fn is_ymm(self) -> bool {
        use Register::*;
        matches!(
            self,
            Ymm0 | Ymm1
                | Ymm2
                | Ymm3
                | Ymm4
                | Ymm5
                | Ymm6
                | Ymm7
                | Ymm8
                | Ymm9
                | Ymm10
                | Ymm11
                | Ymm12
                | Ymm13
                | Ymm14
                | Ymm15
        )
    }

    /// Whether this is a ZMM (AVX-512) register.
    #[must_use]
    pub fn is_zmm(self) -> bool {
        use Register::*;
        matches!(
            self,
            Zmm0 | Zmm1
                | Zmm2
                | Zmm3
                | Zmm4
                | Zmm5
                | Zmm6
                | Zmm7
                | Zmm8
                | Zmm9
                | Zmm10
                | Zmm11
                | Zmm12
                | Zmm13
                | Zmm14
                | Zmm15
                | Zmm16
                | Zmm17
                | Zmm18
                | Zmm19
                | Zmm20
                | Zmm21
                | Zmm22
                | Zmm23
                | Zmm24
                | Zmm25
                | Zmm26
                | Zmm27
                | Zmm28
                | Zmm29
                | Zmm30
                | Zmm31
        )
    }

    /// Whether this is an opmask register (K0–K7).
    #[must_use]
    pub fn is_opmask(self) -> bool {
        use Register::*;
        matches!(self, K0 | K1 | K2 | K3 | K4 | K5 | K6 | K7)
    }

    /// Whether this is any vector register (XMM, YMM, or ZMM).
    #[must_use]
    pub fn is_vector(self) -> bool {
        self.is_xmm() || self.is_ymm() || self.is_zmm()
    }

    /// Return the 32-bit counterpart of this register, if applicable.
    ///
    /// For 64-bit GP registers (RAX, RCX, ..., R15), returns the corresponding
    /// 32-bit register. For registers that are already 32-bit, returns them
    /// unchanged. Returns `None` for 8-bit, 16-bit, segment, and vector registers.
    #[must_use]
    pub fn to_32bit(self) -> Option<Register> {
        use Register::*;
        match self {
            Rax | Eax => Some(Eax),
            Rcx | Ecx => Some(Ecx),
            Rdx | Edx => Some(Edx),
            Rbx | Ebx => Some(Ebx),
            Rsp | Esp => Some(Esp),
            Rbp | Ebp => Some(Ebp),
            Rsi | Esi => Some(Esi),
            Rdi | Edi => Some(Edi),
            R8 | R8d => Some(R8d),
            R9 | R9d => Some(R9d),
            R10 | R10d => Some(R10d),
            R11 | R11d => Some(R11d),
            R12 | R12d => Some(R12d),
            R13 | R13d => Some(R13d),
            R14 | R14d => Some(R14d),
            R15 | R15d => Some(R15d),
            _ => None,
        }
    }

    /// Whether this is an ARM32 register.
    #[must_use]
    pub fn is_arm(self) -> bool {
        use Register::*;
        matches!(
            self,
            ArmR0
                | ArmR1
                | ArmR2
                | ArmR3
                | ArmR4
                | ArmR5
                | ArmR6
                | ArmR7
                | ArmR8
                | ArmR9
                | ArmR10
                | ArmR11
                | ArmR12
                | ArmSp
                | ArmLr
                | ArmPc
                | ArmCpsr
        )
    }

    /// ARM32 4-bit register number (0–15).
    #[must_use]
    pub fn arm_reg_num(self) -> u8 {
        use Register::*;
        match self {
            ArmR0 => 0,
            ArmR1 => 1,
            ArmR2 => 2,
            ArmR3 => 3,
            ArmR4 => 4,
            ArmR5 => 5,
            ArmR6 => 6,
            ArmR7 => 7,
            ArmR8 => 8,
            ArmR9 => 9,
            ArmR10 => 10,
            ArmR11 => 11,
            ArmR12 => 12,
            ArmSp => 13,
            ArmLr => 14,
            ArmPc => 15,
            _ => 0,
        }
    }

    /// Whether this is an AArch64 register.
    #[must_use]
    pub fn is_aarch64(self) -> bool {
        use Register::*;
        matches!(
            self,
            A64X0
                | A64X1
                | A64X2
                | A64X3
                | A64X4
                | A64X5
                | A64X6
                | A64X7
                | A64X8
                | A64X9
                | A64X10
                | A64X11
                | A64X12
                | A64X13
                | A64X14
                | A64X15
                | A64X16
                | A64X17
                | A64X18
                | A64X19
                | A64X20
                | A64X21
                | A64X22
                | A64X23
                | A64X24
                | A64X25
                | A64X26
                | A64X27
                | A64X28
                | A64X29
                | A64X30
                | A64Sp
                | A64Xzr
                | A64W0
                | A64W1
                | A64W2
                | A64W3
                | A64W4
                | A64W5
                | A64W6
                | A64W7
                | A64W8
                | A64W9
                | A64W10
                | A64W11
                | A64W12
                | A64W13
                | A64W14
                | A64W15
                | A64W16
                | A64W17
                | A64W18
                | A64W19
                | A64W20
                | A64W21
                | A64W22
                | A64W23
                | A64W24
                | A64W25
                | A64W26
                | A64W27
                | A64W28
                | A64W29
                | A64W30
                | A64Wzr
                | A64V0
                | A64V1
                | A64V2
                | A64V3
                | A64V4
                | A64V5
                | A64V6
                | A64V7
                | A64V8
                | A64V9
                | A64V10
                | A64V11
                | A64V12
                | A64V13
                | A64V14
                | A64V15
                | A64V16
                | A64V17
                | A64V18
                | A64V19
                | A64V20
                | A64V21
                | A64V22
                | A64V23
                | A64V24
                | A64V25
                | A64V26
                | A64V27
                | A64V28
                | A64V29
                | A64V30
                | A64V31
                | A64Q0
                | A64Q1
                | A64Q2
                | A64Q3
                | A64Q4
                | A64Q5
                | A64Q6
                | A64Q7
                | A64Q8
                | A64Q9
                | A64Q10
                | A64Q11
                | A64Q12
                | A64Q13
                | A64Q14
                | A64Q15
                | A64Q16
                | A64Q17
                | A64Q18
                | A64Q19
                | A64Q20
                | A64Q21
                | A64Q22
                | A64Q23
                | A64Q24
                | A64Q25
                | A64Q26
                | A64Q27
                | A64Q28
                | A64Q29
                | A64Q30
                | A64Q31
                | A64D0
                | A64D1
                | A64D2
                | A64D3
                | A64D4
                | A64D5
                | A64D6
                | A64D7
                | A64D8
                | A64D9
                | A64D10
                | A64D11
                | A64D12
                | A64D13
                | A64D14
                | A64D15
                | A64D16
                | A64D17
                | A64D18
                | A64D19
                | A64D20
                | A64D21
                | A64D22
                | A64D23
                | A64D24
                | A64D25
                | A64D26
                | A64D27
                | A64D28
                | A64D29
                | A64D30
                | A64D31
                | A64S0
                | A64S1
                | A64S2
                | A64S3
                | A64S4
                | A64S5
                | A64S6
                | A64S7
                | A64S8
                | A64S9
                | A64S10
                | A64S11
                | A64S12
                | A64S13
                | A64S14
                | A64S15
                | A64S16
                | A64S17
                | A64S18
                | A64S19
                | A64S20
                | A64S21
                | A64S22
                | A64S23
                | A64S24
                | A64S25
                | A64S26
                | A64S27
                | A64S28
                | A64S29
                | A64S30
                | A64S31
                | A64H0
                | A64H1
                | A64H2
                | A64H3
                | A64H4
                | A64H5
                | A64H6
                | A64H7
                | A64H8
                | A64H9
                | A64H10
                | A64H11
                | A64H12
                | A64H13
                | A64H14
                | A64H15
                | A64H16
                | A64H17
                | A64H18
                | A64H19
                | A64H20
                | A64H21
                | A64H22
                | A64H23
                | A64H24
                | A64H25
                | A64H26
                | A64H27
                | A64H28
                | A64H29
                | A64H30
                | A64H31
                | A64B0
                | A64B1
                | A64B2
                | A64B3
                | A64B4
                | A64B5
                | A64B6
                | A64B7
                | A64B8
                | A64B9
                | A64B10
                | A64B11
                | A64B12
                | A64B13
                | A64B14
                | A64B15
                | A64B16
                | A64B17
                | A64B18
                | A64B19
                | A64B20
                | A64B21
                | A64B22
                | A64B23
                | A64B24
                | A64B25
                | A64B26
                | A64B27
                | A64B28
                | A64B29
                | A64B30
                | A64B31
                | A64Z0
                | A64Z1
                | A64Z2
                | A64Z3
                | A64Z4
                | A64Z5
                | A64Z6
                | A64Z7
                | A64Z8
                | A64Z9
                | A64Z10
                | A64Z11
                | A64Z12
                | A64Z13
                | A64Z14
                | A64Z15
                | A64Z16
                | A64Z17
                | A64Z18
                | A64Z19
                | A64Z20
                | A64Z21
                | A64Z22
                | A64Z23
                | A64Z24
                | A64Z25
                | A64Z26
                | A64Z27
                | A64Z28
                | A64Z29
                | A64Z30
                | A64Z31
                | A64P0
                | A64P1
                | A64P2
                | A64P3
                | A64P4
                | A64P5
                | A64P6
                | A64P7
                | A64P8
                | A64P9
                | A64P10
                | A64P11
                | A64P12
                | A64P13
                | A64P14
                | A64P15
        )
    }

    /// AArch64 5-bit register number (0–31, where 31 = SP or ZR depending on context).
    #[must_use]
    pub fn a64_reg_num(self) -> u8 {
        use Register::*;
        match self {
            A64X0 | A64W0 => 0,
            A64X1 | A64W1 => 1,
            A64X2 | A64W2 => 2,
            A64X3 | A64W3 => 3,
            A64X4 | A64W4 => 4,
            A64X5 | A64W5 => 5,
            A64X6 | A64W6 => 6,
            A64X7 | A64W7 => 7,
            A64X8 | A64W8 => 8,
            A64X9 | A64W9 => 9,
            A64X10 | A64W10 => 10,
            A64X11 | A64W11 => 11,
            A64X12 | A64W12 => 12,
            A64X13 | A64W13 => 13,
            A64X14 | A64W14 => 14,
            A64X15 | A64W15 => 15,
            A64X16 | A64W16 => 16,
            A64X17 | A64W17 => 17,
            A64X18 | A64W18 => 18,
            A64X19 | A64W19 => 19,
            A64X20 | A64W20 => 20,
            A64X21 | A64W21 => 21,
            A64X22 | A64W22 => 22,
            A64X23 | A64W23 => 23,
            A64X24 | A64W24 => 24,
            A64X25 | A64W25 => 25,
            A64X26 | A64W26 => 26,
            A64X27 | A64W27 => 27,
            A64X28 | A64W28 => 28,
            A64X29 | A64W29 => 29,
            A64X30 | A64W30 => 30,
            A64Sp | A64Xzr | A64Wzr => 31,
            // SIMD/FP registers share the same 0–31 numbering
            A64V0 | A64Q0 | A64D0 | A64S0 | A64H0 | A64B0 => 0,
            A64V1 | A64Q1 | A64D1 | A64S1 | A64H1 | A64B1 => 1,
            A64V2 | A64Q2 | A64D2 | A64S2 | A64H2 | A64B2 => 2,
            A64V3 | A64Q3 | A64D3 | A64S3 | A64H3 | A64B3 => 3,
            A64V4 | A64Q4 | A64D4 | A64S4 | A64H4 | A64B4 => 4,
            A64V5 | A64Q5 | A64D5 | A64S5 | A64H5 | A64B5 => 5,
            A64V6 | A64Q6 | A64D6 | A64S6 | A64H6 | A64B6 => 6,
            A64V7 | A64Q7 | A64D7 | A64S7 | A64H7 | A64B7 => 7,
            A64V8 | A64Q8 | A64D8 | A64S8 | A64H8 | A64B8 => 8,
            A64V9 | A64Q9 | A64D9 | A64S9 | A64H9 | A64B9 => 9,
            A64V10 | A64Q10 | A64D10 | A64S10 | A64H10 | A64B10 => 10,
            A64V11 | A64Q11 | A64D11 | A64S11 | A64H11 | A64B11 => 11,
            A64V12 | A64Q12 | A64D12 | A64S12 | A64H12 | A64B12 => 12,
            A64V13 | A64Q13 | A64D13 | A64S13 | A64H13 | A64B13 => 13,
            A64V14 | A64Q14 | A64D14 | A64S14 | A64H14 | A64B14 => 14,
            A64V15 | A64Q15 | A64D15 | A64S15 | A64H15 | A64B15 => 15,
            A64V16 | A64Q16 | A64D16 | A64S16 | A64H16 | A64B16 => 16,
            A64V17 | A64Q17 | A64D17 | A64S17 | A64H17 | A64B17 => 17,
            A64V18 | A64Q18 | A64D18 | A64S18 | A64H18 | A64B18 => 18,
            A64V19 | A64Q19 | A64D19 | A64S19 | A64H19 | A64B19 => 19,
            A64V20 | A64Q20 | A64D20 | A64S20 | A64H20 | A64B20 => 20,
            A64V21 | A64Q21 | A64D21 | A64S21 | A64H21 | A64B21 => 21,
            A64V22 | A64Q22 | A64D22 | A64S22 | A64H22 | A64B22 => 22,
            A64V23 | A64Q23 | A64D23 | A64S23 | A64H23 | A64B23 => 23,
            A64V24 | A64Q24 | A64D24 | A64S24 | A64H24 | A64B24 => 24,
            A64V25 | A64Q25 | A64D25 | A64S25 | A64H25 | A64B25 => 25,
            A64V26 | A64Q26 | A64D26 | A64S26 | A64H26 | A64B26 => 26,
            A64V27 | A64Q27 | A64D27 | A64S27 | A64H27 | A64B27 => 27,
            A64V28 | A64Q28 | A64D28 | A64S28 | A64H28 | A64B28 => 28,
            A64V29 | A64Q29 | A64D29 | A64S29 | A64H29 | A64B29 => 29,
            A64V30 | A64Q30 | A64D30 | A64S30 | A64H30 | A64B30 => 30,
            A64V31 | A64Q31 | A64D31 | A64S31 | A64H31 | A64B31 => 31,
            // SVE Z registers share the same 0–31 numbering
            A64Z0 => 0,
            A64Z1 => 1,
            A64Z2 => 2,
            A64Z3 => 3,
            A64Z4 => 4,
            A64Z5 => 5,
            A64Z6 => 6,
            A64Z7 => 7,
            A64Z8 => 8,
            A64Z9 => 9,
            A64Z10 => 10,
            A64Z11 => 11,
            A64Z12 => 12,
            A64Z13 => 13,
            A64Z14 => 14,
            A64Z15 => 15,
            A64Z16 => 16,
            A64Z17 => 17,
            A64Z18 => 18,
            A64Z19 => 19,
            A64Z20 => 20,
            A64Z21 => 21,
            A64Z22 => 22,
            A64Z23 => 23,
            A64Z24 => 24,
            A64Z25 => 25,
            A64Z26 => 26,
            A64Z27 => 27,
            A64Z28 => 28,
            A64Z29 => 29,
            A64Z30 => 30,
            A64Z31 => 31,
            _ => 0,
        }
    }

    /// Whether this is an AArch64 vector register (V0–V31).
    ///
    /// Only V registers accept arrangement specifiers (e.g., `V0.4S`).
    #[must_use]
    pub fn is_a64_vector(self) -> bool {
        use Register::*;
        matches!(
            self,
            A64V0
                | A64V1
                | A64V2
                | A64V3
                | A64V4
                | A64V5
                | A64V6
                | A64V7
                | A64V8
                | A64V9
                | A64V10
                | A64V11
                | A64V12
                | A64V13
                | A64V14
                | A64V15
                | A64V16
                | A64V17
                | A64V18
                | A64V19
                | A64V20
                | A64V21
                | A64V22
                | A64V23
                | A64V24
                | A64V25
                | A64V26
                | A64V27
                | A64V28
                | A64V29
                | A64V30
                | A64V31
        )
    }

    /// Whether this is an AArch64 SIMD/FP register (V, Q, D, S, H, or B).
    #[must_use]
    pub fn is_a64_simd_fp(self) -> bool {
        use Register::*;
        matches!(
            self,
            A64V0
                | A64V1
                | A64V2
                | A64V3
                | A64V4
                | A64V5
                | A64V6
                | A64V7
                | A64V8
                | A64V9
                | A64V10
                | A64V11
                | A64V12
                | A64V13
                | A64V14
                | A64V15
                | A64V16
                | A64V17
                | A64V18
                | A64V19
                | A64V20
                | A64V21
                | A64V22
                | A64V23
                | A64V24
                | A64V25
                | A64V26
                | A64V27
                | A64V28
                | A64V29
                | A64V30
                | A64V31
                | A64Q0
                | A64Q1
                | A64Q2
                | A64Q3
                | A64Q4
                | A64Q5
                | A64Q6
                | A64Q7
                | A64Q8
                | A64Q9
                | A64Q10
                | A64Q11
                | A64Q12
                | A64Q13
                | A64Q14
                | A64Q15
                | A64Q16
                | A64Q17
                | A64Q18
                | A64Q19
                | A64Q20
                | A64Q21
                | A64Q22
                | A64Q23
                | A64Q24
                | A64Q25
                | A64Q26
                | A64Q27
                | A64Q28
                | A64Q29
                | A64Q30
                | A64Q31
                | A64D0
                | A64D1
                | A64D2
                | A64D3
                | A64D4
                | A64D5
                | A64D6
                | A64D7
                | A64D8
                | A64D9
                | A64D10
                | A64D11
                | A64D12
                | A64D13
                | A64D14
                | A64D15
                | A64D16
                | A64D17
                | A64D18
                | A64D19
                | A64D20
                | A64D21
                | A64D22
                | A64D23
                | A64D24
                | A64D25
                | A64D26
                | A64D27
                | A64D28
                | A64D29
                | A64D30
                | A64D31
                | A64S0
                | A64S1
                | A64S2
                | A64S3
                | A64S4
                | A64S5
                | A64S6
                | A64S7
                | A64S8
                | A64S9
                | A64S10
                | A64S11
                | A64S12
                | A64S13
                | A64S14
                | A64S15
                | A64S16
                | A64S17
                | A64S18
                | A64S19
                | A64S20
                | A64S21
                | A64S22
                | A64S23
                | A64S24
                | A64S25
                | A64S26
                | A64S27
                | A64S28
                | A64S29
                | A64S30
                | A64S31
                | A64H0
                | A64H1
                | A64H2
                | A64H3
                | A64H4
                | A64H5
                | A64H6
                | A64H7
                | A64H8
                | A64H9
                | A64H10
                | A64H11
                | A64H12
                | A64H13
                | A64H14
                | A64H15
                | A64H16
                | A64H17
                | A64H18
                | A64H19
                | A64H20
                | A64H21
                | A64H22
                | A64H23
                | A64H24
                | A64H25
                | A64H26
                | A64H27
                | A64H28
                | A64H29
                | A64H30
                | A64H31
                | A64B0
                | A64B1
                | A64B2
                | A64B3
                | A64B4
                | A64B5
                | A64B6
                | A64B7
                | A64B8
                | A64B9
                | A64B10
                | A64B11
                | A64B12
                | A64B13
                | A64B14
                | A64B15
                | A64B16
                | A64B17
                | A64B18
                | A64B19
                | A64B20
                | A64B21
                | A64B22
                | A64B23
                | A64B24
                | A64B25
                | A64B26
                | A64B27
                | A64B28
                | A64B29
                | A64B30
                | A64B31
        )
    }

    /// Returns the SIMD/FP register bit width (128 for V/Q, 64 for D, 32 for S, 16 for H, 8 for B).
    #[must_use]
    pub fn a64_simd_fp_bits(self) -> u32 {
        use Register::*;
        match self {
            A64V0 | A64V1 | A64V2 | A64V3 | A64V4 | A64V5 | A64V6 | A64V7 | A64V8 | A64V9
            | A64V10 | A64V11 | A64V12 | A64V13 | A64V14 | A64V15 | A64V16 | A64V17 | A64V18
            | A64V19 | A64V20 | A64V21 | A64V22 | A64V23 | A64V24 | A64V25 | A64V26 | A64V27
            | A64V28 | A64V29 | A64V30 | A64V31 | A64Q0 | A64Q1 | A64Q2 | A64Q3 | A64Q4 | A64Q5
            | A64Q6 | A64Q7 | A64Q8 | A64Q9 | A64Q10 | A64Q11 | A64Q12 | A64Q13 | A64Q14
            | A64Q15 | A64Q16 | A64Q17 | A64Q18 | A64Q19 | A64Q20 | A64Q21 | A64Q22 | A64Q23
            | A64Q24 | A64Q25 | A64Q26 | A64Q27 | A64Q28 | A64Q29 | A64Q30 | A64Q31 => 128,
            A64D0 | A64D1 | A64D2 | A64D3 | A64D4 | A64D5 | A64D6 | A64D7 | A64D8 | A64D9
            | A64D10 | A64D11 | A64D12 | A64D13 | A64D14 | A64D15 | A64D16 | A64D17 | A64D18
            | A64D19 | A64D20 | A64D21 | A64D22 | A64D23 | A64D24 | A64D25 | A64D26 | A64D27
            | A64D28 | A64D29 | A64D30 | A64D31 => 64,
            A64S0 | A64S1 | A64S2 | A64S3 | A64S4 | A64S5 | A64S6 | A64S7 | A64S8 | A64S9
            | A64S10 | A64S11 | A64S12 | A64S13 | A64S14 | A64S15 | A64S16 | A64S17 | A64S18
            | A64S19 | A64S20 | A64S21 | A64S22 | A64S23 | A64S24 | A64S25 | A64S26 | A64S27
            | A64S28 | A64S29 | A64S30 | A64S31 => 32,
            A64H0 | A64H1 | A64H2 | A64H3 | A64H4 | A64H5 | A64H6 | A64H7 | A64H8 | A64H9
            | A64H10 | A64H11 | A64H12 | A64H13 | A64H14 | A64H15 | A64H16 | A64H17 | A64H18
            | A64H19 | A64H20 | A64H21 | A64H22 | A64H23 | A64H24 | A64H25 | A64H26 | A64H27
            | A64H28 | A64H29 | A64H30 | A64H31 => 16,
            A64B0 | A64B1 | A64B2 | A64B3 | A64B4 | A64B5 | A64B6 | A64B7 | A64B8 | A64B9
            | A64B10 | A64B11 | A64B12 | A64B13 | A64B14 | A64B15 | A64B16 | A64B17 | A64B18
            | A64B19 | A64B20 | A64B21 | A64B22 | A64B23 | A64B24 | A64B25 | A64B26 | A64B27
            | A64B28 | A64B29 | A64B30 | A64B31 => 8,
            _ => 0,
        }
    }

    /// Whether this is a 64-bit AArch64 X register (vs 32-bit W register).
    #[must_use]
    pub fn is_a64_64bit(self) -> bool {
        use Register::*;
        matches!(
            self,
            A64X0
                | A64X1
                | A64X2
                | A64X3
                | A64X4
                | A64X5
                | A64X6
                | A64X7
                | A64X8
                | A64X9
                | A64X10
                | A64X11
                | A64X12
                | A64X13
                | A64X14
                | A64X15
                | A64X16
                | A64X17
                | A64X18
                | A64X19
                | A64X20
                | A64X21
                | A64X22
                | A64X23
                | A64X24
                | A64X25
                | A64X26
                | A64X27
                | A64X28
                | A64X29
                | A64X30
                | A64Sp
                | A64Xzr
        )
    }

    /// Whether this is a RISC-V integer register.
    #[must_use]
    pub fn is_riscv(self) -> bool {
        use Register::*;
        matches!(
            self,
            RvX0 | RvX1
                | RvX2
                | RvX3
                | RvX4
                | RvX5
                | RvX6
                | RvX7
                | RvX8
                | RvX9
                | RvX10
                | RvX11
                | RvX12
                | RvX13
                | RvX14
                | RvX15
                | RvX16
                | RvX17
                | RvX18
                | RvX19
                | RvX20
                | RvX21
                | RvX22
                | RvX23
                | RvX24
                | RvX25
                | RvX26
                | RvX27
                | RvX28
                | RvX29
                | RvX30
                | RvX31
        )
    }

    /// RISC-V 5-bit register number (0–31).
    #[must_use]
    pub fn rv_reg_num(self) -> u8 {
        use Register::*;
        match self {
            RvX0 => 0,
            RvX1 => 1,
            RvX2 => 2,
            RvX3 => 3,
            RvX4 => 4,
            RvX5 => 5,
            RvX6 => 6,
            RvX7 => 7,
            RvX8 => 8,
            RvX9 => 9,
            RvX10 => 10,
            RvX11 => 11,
            RvX12 => 12,
            RvX13 => 13,
            RvX14 => 14,
            RvX15 => 15,
            RvX16 => 16,
            RvX17 => 17,
            RvX18 => 18,
            RvX19 => 19,
            RvX20 => 20,
            RvX21 => 21,
            RvX22 => 22,
            RvX23 => 23,
            RvX24 => 24,
            RvX25 => 25,
            RvX26 => 26,
            RvX27 => 27,
            RvX28 => 28,
            RvX29 => 29,
            RvX30 => 30,
            RvX31 => 31,
            _ => 0,
        }
    }

    /// Whether this is a RISC-V floating-point register (f0–f31).
    #[must_use]
    pub fn is_riscv_fp(self) -> bool {
        use Register::*;
        matches!(
            self,
            RvF0 | RvF1
                | RvF2
                | RvF3
                | RvF4
                | RvF5
                | RvF6
                | RvF7
                | RvF8
                | RvF9
                | RvF10
                | RvF11
                | RvF12
                | RvF13
                | RvF14
                | RvF15
                | RvF16
                | RvF17
                | RvF18
                | RvF19
                | RvF20
                | RvF21
                | RvF22
                | RvF23
                | RvF24
                | RvF25
                | RvF26
                | RvF27
                | RvF28
                | RvF29
                | RvF30
                | RvF31
        )
    }

    /// RISC-V FP 5-bit register number (0–31).
    #[must_use]
    pub fn rv_fp_reg_num(self) -> u8 {
        use Register::*;
        match self {
            RvF0 => 0,
            RvF1 => 1,
            RvF2 => 2,
            RvF3 => 3,
            RvF4 => 4,
            RvF5 => 5,
            RvF6 => 6,
            RvF7 => 7,
            RvF8 => 8,
            RvF9 => 9,
            RvF10 => 10,
            RvF11 => 11,
            RvF12 => 12,
            RvF13 => 13,
            RvF14 => 14,
            RvF15 => 15,
            RvF16 => 16,
            RvF17 => 17,
            RvF18 => 18,
            RvF19 => 19,
            RvF20 => 20,
            RvF21 => 21,
            RvF22 => 22,
            RvF23 => 23,
            RvF24 => 24,
            RvF25 => 25,
            RvF26 => 26,
            RvF27 => 27,
            RvF28 => 28,
            RvF29 => 29,
            RvF30 => 30,
            RvF31 => 31,
            _ => 0,
        }
    }

    /// Whether this is an AArch64 SVE scalable vector register (Z0–Z31).
    #[must_use]
    pub fn is_a64_sve_z(self) -> bool {
        use Register::*;
        matches!(
            self,
            A64Z0
                | A64Z1
                | A64Z2
                | A64Z3
                | A64Z4
                | A64Z5
                | A64Z6
                | A64Z7
                | A64Z8
                | A64Z9
                | A64Z10
                | A64Z11
                | A64Z12
                | A64Z13
                | A64Z14
                | A64Z15
                | A64Z16
                | A64Z17
                | A64Z18
                | A64Z19
                | A64Z20
                | A64Z21
                | A64Z22
                | A64Z23
                | A64Z24
                | A64Z25
                | A64Z26
                | A64Z27
                | A64Z28
                | A64Z29
                | A64Z30
                | A64Z31
        )
    }

    /// Whether this is an AArch64 SVE predicate register (P0–P15).
    #[must_use]
    pub fn is_a64_sve_p(self) -> bool {
        use Register::*;
        matches!(
            self,
            A64P0
                | A64P1
                | A64P2
                | A64P3
                | A64P4
                | A64P5
                | A64P6
                | A64P7
                | A64P8
                | A64P9
                | A64P10
                | A64P11
                | A64P12
                | A64P13
                | A64P14
                | A64P15
        )
    }

    /// AArch64 SVE predicate register number (0–15).
    #[must_use]
    pub fn a64_p_num(self) -> u8 {
        use Register::*;
        match self {
            A64P0 => 0,
            A64P1 => 1,
            A64P2 => 2,
            A64P3 => 3,
            A64P4 => 4,
            A64P5 => 5,
            A64P6 => 6,
            A64P7 => 7,
            A64P8 => 8,
            A64P9 => 9,
            A64P10 => 10,
            A64P11 => 11,
            A64P12 => 12,
            A64P13 => 13,
            A64P14 => 14,
            A64P15 => 15,
            _ => 0,
        }
    }

    /// Whether this is a RISC-V vector register (v0–v31).
    #[must_use]
    pub fn is_riscv_vec(self) -> bool {
        use Register::*;
        matches!(
            self,
            RvV0 | RvV1
                | RvV2
                | RvV3
                | RvV4
                | RvV5
                | RvV6
                | RvV7
                | RvV8
                | RvV9
                | RvV10
                | RvV11
                | RvV12
                | RvV13
                | RvV14
                | RvV15
                | RvV16
                | RvV17
                | RvV18
                | RvV19
                | RvV20
                | RvV21
                | RvV22
                | RvV23
                | RvV24
                | RvV25
                | RvV26
                | RvV27
                | RvV28
                | RvV29
                | RvV30
                | RvV31
        )
    }

    /// RISC-V vector register number (0–31).
    #[must_use]
    pub fn rv_vec_num(self) -> u8 {
        use Register::*;
        match self {
            RvV0 => 0,
            RvV1 => 1,
            RvV2 => 2,
            RvV3 => 3,
            RvV4 => 4,
            RvV5 => 5,
            RvV6 => 6,
            RvV7 => 7,
            RvV8 => 8,
            RvV9 => 9,
            RvV10 => 10,
            RvV11 => 11,
            RvV12 => 12,
            RvV13 => 13,
            RvV14 => 14,
            RvV15 => 15,
            RvV16 => 16,
            RvV17 => 17,
            RvV18 => 18,
            RvV19 => 19,
            RvV20 => 20,
            RvV21 => 21,
            RvV22 => 22,
            RvV23 => 23,
            RvV24 => 24,
            RvV25 => 25,
            RvV26 => 26,
            RvV27 => 27,
            RvV28 => 28,
            RvV29 => 29,
            RvV30 => 30,
            RvV31 => 31,
            _ => 0,
        }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Zero-allocation: write Debug chars lowercased directly to the formatter.
        use fmt::Write as _;
        struct LowerWriter<'a, 'b>(&'a mut fmt::Formatter<'b>);
        impl fmt::Write for LowerWriter<'_, '_> {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                for c in s.chars() {
                    self.0.write_char(c.to_ascii_lowercase())?;
                }
                Ok(())
            }
        }
        write!(LowerWriter(f), "{:?}", self)
    }
}

/// AArch64 vector arrangement specifier.
///
/// Describes how a 128-bit (or 64-bit) SIMD register is divided into lanes.
/// Used with NEON/ASIMD instructions like `ADD V0.4S, V1.4S, V2.4S`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorArrangement {
    /// 8 bytes (64-bit: 8×B)
    B8,
    /// 16 bytes (128-bit: 16×B)
    B16,
    /// 4 half-words (64-bit: 4×H)
    H4,
    /// 8 half-words (128-bit: 8×H)
    H8,
    /// 2 single-words (64-bit: 2×S)
    S2,
    /// 4 single-words (128-bit: 4×S)
    S4,
    /// 1 double-word (64-bit: 1×D)
    D1,
    /// 2 double-words (128-bit: 2×D)
    D2,
    // ── SVE element sizes (scalable — no lane count) ────────
    /// SVE byte elements (`.B`).
    SveB,
    /// SVE half-word elements (`.H`).
    SveH,
    /// SVE single-word elements (`.S`).
    SveS,
    /// SVE double-word elements (`.D`).
    SveD,
}

impl VectorArrangement {
    /// Parse a vector arrangement specifier string (e.g., "8b", "16b", "4h").
    /// Case-insensitive, zero heap allocations.
    pub fn parse(s: &str) -> Option<Self> {
        // Stack-based lowercase (arrangement specifiers are at most 3 chars).
        let mut buf = [0u8; 4];
        let len = s.len().min(4);
        buf[..len].copy_from_slice(&s.as_bytes()[..len]);
        buf[..len].make_ascii_lowercase();
        // Input was valid UTF-8 and ASCII lowercase preserves validity.
        let s = core::str::from_utf8(&buf[..len]).unwrap_or("");
        match s {
            "8b" => Some(VectorArrangement::B8),
            "16b" => Some(VectorArrangement::B16),
            "4h" => Some(VectorArrangement::H4),
            "8h" => Some(VectorArrangement::H8),
            "2s" => Some(VectorArrangement::S2),
            "4s" => Some(VectorArrangement::S4),
            "1d" => Some(VectorArrangement::D1),
            "2d" => Some(VectorArrangement::D2),
            // SVE scalable element-size specifiers
            "b" => Some(VectorArrangement::SveB),
            "h" => Some(VectorArrangement::SveH),
            "s" => Some(VectorArrangement::SveS),
            "d" => Some(VectorArrangement::SveD),
            _ => None,
        }
    }

    /// Lane element size in bits.
    pub fn element_bits(self) -> u32 {
        match self {
            VectorArrangement::B8 | VectorArrangement::B16 | VectorArrangement::SveB => 8,
            VectorArrangement::H4 | VectorArrangement::H8 | VectorArrangement::SveH => 16,
            VectorArrangement::S2 | VectorArrangement::S4 | VectorArrangement::SveS => 32,
            VectorArrangement::D1 | VectorArrangement::D2 | VectorArrangement::SveD => 64,
        }
    }

    /// Total vector width in bits (64 or 128 for NEON, 0 for SVE scalable).
    pub fn total_bits(self) -> u32 {
        match self {
            VectorArrangement::B8
            | VectorArrangement::H4
            | VectorArrangement::S2
            | VectorArrangement::D1 => 64,
            VectorArrangement::B16
            | VectorArrangement::H8
            | VectorArrangement::S4
            | VectorArrangement::D2 => 128,
            // SVE: scalable, width unknown at assembly time
            VectorArrangement::SveB
            | VectorArrangement::SveH
            | VectorArrangement::SveS
            | VectorArrangement::SveD => 0,
        }
    }

    /// Number of lanes (0 for SVE scalable arrangements).
    pub fn lane_count(self) -> u32 {
        let total = self.total_bits();
        if total == 0 {
            return 0;
        }
        total / self.element_bits()
    }

    /// SVE element size encoding (2-bit `sz` field): B=0, H=1, S=2, D=3.
    /// Returns `None` for non-SVE arrangements.
    pub fn sve_size(self) -> Option<u32> {
        match self {
            VectorArrangement::SveB => Some(0),
            VectorArrangement::SveH => Some(1),
            VectorArrangement::SveS => Some(2),
            VectorArrangement::SveD => Some(3),
            _ => None,
        }
    }
}

impl fmt::Display for VectorArrangement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorArrangement::B8 => write!(f, "8B"),
            VectorArrangement::B16 => write!(f, "16B"),
            VectorArrangement::H4 => write!(f, "4H"),
            VectorArrangement::H8 => write!(f, "8H"),
            VectorArrangement::S2 => write!(f, "2S"),
            VectorArrangement::S4 => write!(f, "4S"),
            VectorArrangement::D1 => write!(f, "1D"),
            VectorArrangement::D2 => write!(f, "2D"),
            VectorArrangement::SveB => write!(f, "B"),
            VectorArrangement::SveH => write!(f, "H"),
            VectorArrangement::SveS => write!(f, "S"),
            VectorArrangement::SveD => write!(f, "D"),
        }
    }
}

/// SVE predicate qualifier (merging or zeroing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SvePredQual {
    /// Merging predication: `/M` — inactive lanes keep their old value.
    Merging,
    /// Zeroing predication: `/Z` — inactive lanes are set to zero.
    Zeroing,
}

impl fmt::Display for SvePredQual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SvePredQual::Merging => write!(f, "/m"),
            SvePredQual::Zeroing => write!(f, "/z"),
        }
    }
}

/// Operand size hint (from `byte ptr`, `dword ptr`, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OperandSize {
    /// 8-bit (`byte ptr`).
    Byte,
    /// 16-bit (`word ptr`).
    Word,
    /// 32-bit (`dword ptr`).
    Dword,
    /// 64-bit (`qword ptr`).
    Qword,
    /// 128-bit (`xmmword ptr` / `oword ptr`).
    Xmmword,
    /// 256-bit (`ymmword ptr`).
    Ymmword,
    /// 512-bit (`zmmword ptr`).
    Zmmword,
}

impl OperandSize {
    /// Return the operand size in bits.
    pub fn bits(self) -> u16 {
        match self {
            OperandSize::Byte => 8,
            OperandSize::Word => 16,
            OperandSize::Dword => 32,
            OperandSize::Qword => 64,
            OperandSize::Xmmword => 128,
            OperandSize::Ymmword => 256,
            OperandSize::Zmmword => 512,
        }
    }
}

impl fmt::Display for OperandSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperandSize::Byte => write!(f, "byte"),
            OperandSize::Word => write!(f, "word"),
            OperandSize::Dword => write!(f, "dword"),
            OperandSize::Qword => write!(f, "qword"),
            OperandSize::Xmmword => write!(f, "xmmword"),
            OperandSize::Ymmword => write!(f, "ymmword"),
            OperandSize::Zmmword => write!(f, "zmmword"),
        }
    }
}

/// Addressing mode for ARM/AArch64 load/store instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AddrMode {
    /// Regular offset: `[Rn, #imm]`
    #[default]
    Offset,
    /// Pre-index with writeback: `[Rn, #imm]!`
    PreIndex,
    /// Post-index with writeback: `[Rn], #imm`
    PostIndex,
}

/// A memory (indirect) operand.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MemoryOperand {
    /// Size qualifier (`byte ptr`, `qword ptr`, …) or `None` to infer.
    pub size: Option<OperandSize>,
    /// Base register (e.g., `rbp` in `[rbp+8]`).
    pub base: Option<Register>,
    /// Index register for SIB addressing (e.g., `rsi` in `[rbx+rsi*4]`).
    pub index: Option<Register>,
    /// SIB scale factor: 1, 2, 4, or 8.
    pub scale: u8,
    /// Displacement (constant offset) in bytes.
    pub disp: i64,
    /// Segment override prefix, if any (e.g., `fs:`).
    pub segment: Option<Register>,
    /// When the displacement is a label reference, the label name.
    pub disp_label: Option<String>,
    /// ARM/AArch64 addressing mode (offset, pre-index, post-index).
    pub addr_mode: AddrMode,
    /// Whether the index register is subtracted (ARM `[Rn, -Rm]`).
    pub index_subtract: bool,
}

impl Default for MemoryOperand {
    fn default() -> Self {
        Self {
            size: None,
            base: None,
            index: None,
            scale: 1,
            disp: 0,
            segment: None,
            disp_label: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        }
    }
}

/// An expression node for label arithmetic.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Expr {
    /// A numeric literal.
    Num(i128),
    /// A label reference.
    Label(String),
    /// Addition: left + right.
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction: left - right.
    Sub(Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Try to evaluate to a constant integer.
    ///
    /// Returns `None` if the expression still contains unresolved label references.
    pub fn eval(&self) -> Option<i128> {
        match self {
            Expr::Num(n) => Some(*n),
            Expr::Label(_) => None,
            Expr::Add(l, r) => Some(l.eval()?.checked_add(r.eval()?)?),
            Expr::Sub(l, r) => Some(l.eval()?.checked_sub(r.eval()?)?),
        }
    }

    /// Decompose into a single label reference plus a numeric addend.
    ///
    /// Returns `Some((label_name, addend))` when the expression is of the form
    /// `label`, `label + const`, or `label - const` (after constant folding).
    /// Returns `None` if the expression has zero or multiple label references,
    /// or a negated label (e.g. `5 - label`).
    pub fn label_addend(&self) -> Option<(&str, i64)> {
        let mut label: Option<&str> = None;
        let mut addend: i64 = 0;
        if self.collect_single_label(&mut label, &mut addend, 1) {
            label.map(|l| (l, addend))
        } else {
            None
        }
    }

    /// Replace `Expr::Label(name)` nodes with `Expr::Num(value)` when
    /// the given lookup returns a value for the name.
    pub fn resolve_constants(&mut self, lookup: impl Fn(&str) -> Option<i128> + Copy) {
        match self {
            Expr::Label(name) => {
                if let Some(val) = lookup(name) {
                    *self = Expr::Num(val);
                }
            }
            Expr::Add(l, r) | Expr::Sub(l, r) => {
                l.resolve_constants(lookup);
                r.resolve_constants(lookup);
            }
            Expr::Num(_) => {}
        }
    }

    /// Recursively find a single label reference with positive sign (no allocation).
    /// Returns `true` if the expression contains exactly one positively-signed label.
    fn collect_single_label<'a>(
        &'a self,
        label: &mut Option<&'a str>,
        addend: &mut i64,
        sign: i64,
    ) -> bool {
        match self {
            Expr::Num(n) => {
                *addend = addend.wrapping_add((*n as i64).wrapping_mul(sign));
                true
            }
            Expr::Label(name) => {
                if sign != 1 || label.is_some() {
                    // Negated label or second label → invalid
                    false
                } else {
                    *label = Some(name.as_str());
                    true
                }
            }
            Expr::Add(l, r) => {
                l.collect_single_label(label, addend, sign)
                    && r.collect_single_label(label, addend, sign)
            }
            Expr::Sub(l, r) => {
                l.collect_single_label(label, addend, sign)
                    && r.collect_single_label(label, addend, -sign)
            }
        }
    }
}

/// A resolved or unresolved operand.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Operand {
    /// A register operand.
    Register(Register),
    /// An immediate value.
    Immediate(i128),
    /// A memory (indirect) operand.
    Memory(Box<MemoryOperand>),
    /// A label reference (resolved later).
    Label(String),
    /// An expression (e.g., `label + 4`).
    Expression(Expr),
    /// A register list (ARM `{R0, R1, R4, LR}`).
    RegisterList(Vec<Register>),
    /// A literal pool value (`LDR Xn, =0x1234`).
    /// The assembler will place the constant in a nearby literal pool
    /// and emit a PC-relative LDR to load it.
    LiteralPoolValue(i128),
    /// A vector register with arrangement specifier (AArch64 NEON/SVE).
    /// E.g., `V0.4S`, `V1.16B`, `Z0.S`, `Z1.D`.
    VectorRegister(Register, VectorArrangement),
    /// SVE predicate register with qualifier (`P0/M`, `P0/Z`).
    SvePredicate(Register, SvePredQual),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Register(r) => write!(f, "{}", r),
            Operand::Immediate(v) => {
                if *v < 0 {
                    write!(f, "-0x{:X}", v.wrapping_neg())
                } else {
                    write!(f, "0x{:X}", v)
                }
            }
            Operand::Memory(mem) => {
                if let Some(sz) = mem.size {
                    write!(f, "{} ptr ", sz)?;
                }
                write!(f, "[")?;
                let mut parts = false;
                if let Some(base) = mem.base {
                    write!(f, "{}", base)?;
                    parts = true;
                }
                if let Some(idx) = mem.index {
                    if parts {
                        write!(f, "+")?;
                    }
                    write!(f, "{}*{}", idx, mem.scale)?;
                    parts = true;
                }
                if mem.disp != 0 || !parts {
                    if parts && mem.disp >= 0 {
                        write!(f, "+")?;
                    }
                    write!(f, "0x{:X}", mem.disp)?;
                }
                write!(f, "]")
            }
            Operand::Label(name) => write!(f, "{}", name),
            Operand::Expression(expr) => write!(f, "{}", expr),
            Operand::RegisterList(regs) => {
                write!(f, "{{")?;
                for (i, r) in regs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", r)?;
                }
                write!(f, "}}")
            }
            Operand::LiteralPoolValue(v) => write!(f, "={}", v),
            Operand::VectorRegister(r, arr) => write!(f, "{}.{}", r, arr),
            Operand::SvePredicate(r, qual) => write!(f, "{}{}", r, qual),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Num(n) => write!(f, "{}", n),
            Expr::Label(name) => write!(f, "{}", name),
            Expr::Add(l, r) => write!(f, "({} + {})", l, r),
            Expr::Sub(l, r) => write!(f, "({} - {})", l, r),
        }
    }
}

/// x86 instruction prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Prefix {
    /// `LOCK` prefix — atomic read-modify-write.
    Lock,
    /// `REP` / `REPE` / `REPZ` prefix — repeat while equal / count.
    Rep,
    /// `REPNE` / `REPNZ` prefix — repeat while not equal.
    Repne,
    /// `FS:` segment override.
    SegFs,
    /// `GS:` segment override.
    SegGs,
}

impl fmt::Display for Prefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Prefix::Lock => write!(f, "lock"),
            Prefix::Rep => write!(f, "rep"),
            Prefix::Repne => write!(f, "repne"),
            Prefix::SegFs => write!(f, "fs"),
            Prefix::SegGs => write!(f, "gs"),
        }
    }
}

// ─── Mnemonic: stack-allocated instruction mnemonic ──────────────────

/// Stack-allocated instruction mnemonic (max 24 ASCII bytes).
///
/// Replaces `String` for zero-allocation instruction construction.
/// All x86-64, AArch64, ARM, and RISC-V mnemonics fit within 24 bytes
/// (longest: `vaeskeygenassist` at 16 bytes).
#[derive(Clone, Copy)]
pub struct Mnemonic {
    buf: [u8; 24],
    len: u8,
}

impl Mnemonic {
    /// Maximum mnemonic length in bytes.
    pub const MAX_LEN: usize = 24;

    /// Creates a new empty `Mnemonic`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            buf: [0; 24],
            len: 0,
        }
    }

    /// Returns the mnemonic as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        // buf always contains valid UTF-8 (ASCII subset, written via from())
        core::str::from_utf8(&self.buf[..self.len as usize]).unwrap_or("")
    }

    /// Returns the length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns true if the mnemonic is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl From<&str> for Mnemonic {
    #[inline]
    fn from(s: &str) -> Self {
        let len = s.len().min(Self::MAX_LEN);
        let mut buf = [0u8; 24];
        buf[..len].copy_from_slice(&s.as_bytes()[..len]);
        Self {
            buf,
            len: len as u8,
        }
    }
}

impl From<String> for Mnemonic {
    #[inline]
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

impl core::ops::Deref for Mnemonic {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq for Mnemonic {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for Mnemonic {}

impl PartialEq<str> for Mnemonic {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for Mnemonic {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<Mnemonic> for str {
    #[inline]
    fn eq(&self, other: &Mnemonic) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<Mnemonic> for &str {
    #[inline]
    fn eq(&self, other: &Mnemonic) -> bool {
        *self == other.as_str()
    }
}

impl PartialEq<String> for Mnemonic {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}

impl fmt::Debug for Mnemonic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.as_str())
    }
}

impl fmt::Display for Mnemonic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Default for Mnemonic {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Mnemonic {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Mnemonic {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Mnemonic;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a mnemonic string (max 24 bytes)")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Mnemonic, E> {
                if v.len() > Mnemonic::MAX_LEN {
                    return Err(E::custom("mnemonic exceeds 24 bytes"));
                }
                Ok(Mnemonic::from(v))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Mnemonic, E> {
                self.visit_str(&v)
            }
        }
        deserializer.deserialize_str(V)
    }
}

// ─── OperandList: stack-allocated operand array ──────────────────────

/// Stack-allocated operand list (max 4 operands).
///
/// Replaces `Vec<Operand>` for zero-allocation instruction construction.
/// All instructions across x86-64, AArch64, ARM, and RISC-V use ≤ 6 operands
/// (RISC-V vector `vsetvli` uses 6: rd, rs1, sew, lmul, ta/tu, ma/mu).
pub struct OperandList {
    items: [Operand; 6],
    len: u8,
}

impl OperandList {
    /// Maximum number of operands.
    pub const MAX_LEN: usize = 6;

    /// Creates a new empty operand list.
    #[inline]
    pub fn new() -> Self {
        Self {
            items: core::array::from_fn(|_| Operand::default()),
            len: 0,
        }
    }

    /// Appends an operand to the list.
    ///
    /// # Panics
    /// Panics if the list is full (> 4 operands).
    #[inline]
    pub fn push(&mut self, op: Operand) {
        assert!(
            (self.len as usize) < Self::MAX_LEN,
            "OperandList overflow: max {} operands",
            Self::MAX_LEN
        );
        self.items[self.len as usize] = op;
        self.len += 1;
    }

    /// Returns the number of operands.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns `true` if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the active operands as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[Operand] {
        &self.items[..self.len as usize]
    }

    /// Returns the active operands as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Operand] {
        &mut self.items[..self.len as usize]
    }
}

impl core::ops::Deref for OperandList {
    type Target = [Operand];
    #[inline]
    fn deref(&self) -> &[Operand] {
        self.as_slice()
    }
}

impl core::ops::DerefMut for OperandList {
    #[inline]
    fn deref_mut(&mut self) -> &mut [Operand] {
        self.as_mut_slice()
    }
}

impl Clone for OperandList {
    fn clone(&self) -> Self {
        let mut items: [Operand; 6] = core::array::from_fn(|_| Operand::default());
        for (i, op) in self.items[..self.len as usize].iter().enumerate() {
            items[i] = op.clone();
        }
        Self {
            items,
            len: self.len,
        }
    }
}

impl PartialEq for OperandList {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<Vec<Operand>> for OperandList {
    fn eq(&self, other: &Vec<Operand>) -> bool {
        self.as_slice() == &other[..]
    }
}

impl fmt::Debug for OperandList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice().iter()).finish()
    }
}

impl Default for OperandList {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<Operand>> for OperandList {
    fn from(v: Vec<Operand>) -> Self {
        assert!(
            v.len() <= Self::MAX_LEN,
            "OperandList: max {} operands, got {}",
            Self::MAX_LEN,
            v.len()
        );
        let mut list = Self::new();
        for op in v {
            list.push(op);
        }
        list
    }
}

impl<'a> IntoIterator for &'a OperandList {
    type Item = &'a Operand;
    type IntoIter = core::slice::Iter<'a, Operand>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut OperandList {
    type Item = &'a mut Operand;
    type IntoIter = core::slice::IterMut<'a, Operand>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for OperandList {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for op in self.as_slice() {
            seq.serialize_element(op)?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for OperandList {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v: Vec<Operand> = Vec::deserialize(deserializer)?;
        if v.len() > Self::MAX_LEN {
            return Err(serde::de::Error::custom(alloc::format!(
                "too many operands: {} > {}",
                v.len(),
                Self::MAX_LEN
            )));
        }
        Ok(Self::from(v))
    }
}

// ─── PrefixList: stack-allocated prefix array ────────────────────────

/// Stack-allocated prefix list (max 4 prefixes).
///
/// Replaces `Vec<Prefix>` for zero-allocation instruction construction.
/// x86-64 instructions use at most 2–3 prefixes in practice.
#[derive(Clone, Copy)]
pub struct PrefixList {
    items: [Prefix; 4],
    len: u8,
}

impl PrefixList {
    /// Maximum number of prefixes.
    pub const MAX_LEN: usize = 4;

    /// Creates a new empty prefix list.
    #[inline]
    pub const fn new() -> Self {
        Self {
            items: [Prefix::Lock; 4], // sentinel, never read beyond len
            len: 0,
        }
    }

    /// Appends a prefix to the list.
    ///
    /// # Panics
    /// Panics if the list is full (> 4 prefixes).
    #[inline]
    pub fn push(&mut self, p: Prefix) {
        assert!(
            (self.len as usize) < Self::MAX_LEN,
            "PrefixList overflow: max {} prefixes",
            Self::MAX_LEN
        );
        self.items[self.len as usize] = p;
        self.len += 1;
    }

    /// Returns the number of prefixes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns `true` if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the active prefixes as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[Prefix] {
        &self.items[..self.len as usize]
    }
}

impl core::ops::Deref for PrefixList {
    type Target = [Prefix];
    #[inline]
    fn deref(&self) -> &[Prefix] {
        self.as_slice()
    }
}

impl PartialEq for PrefixList {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for PrefixList {}

impl PartialEq<Vec<Prefix>> for PrefixList {
    fn eq(&self, other: &Vec<Prefix>) -> bool {
        self.as_slice() == &other[..]
    }
}

impl fmt::Debug for PrefixList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice().iter()).finish()
    }
}

impl Default for PrefixList {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<Prefix>> for PrefixList {
    fn from(v: Vec<Prefix>) -> Self {
        assert!(
            v.len() <= Self::MAX_LEN,
            "PrefixList: max {} prefixes, got {}",
            Self::MAX_LEN,
            v.len()
        );
        let mut list = Self::new();
        for p in v {
            list.push(p);
        }
        list
    }
}

impl<'a> IntoIterator for &'a PrefixList {
    type Item = &'a Prefix;
    type IntoIter = core::slice::Iter<'a, Prefix>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for PrefixList {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for p in self.as_slice() {
            seq.serialize_element(p)?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for PrefixList {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v: Vec<Prefix> = Vec::deserialize(deserializer)?;
        if v.len() > Self::MAX_LEN {
            return Err(serde::de::Error::custom(alloc::format!(
                "too many prefixes: {} > {}",
                v.len(),
                Self::MAX_LEN
            )));
        }
        Ok(Self::from(v))
    }
}

// ─── Default for Operand (used by OperandList) ──────────────────────

impl Default for Operand {
    /// Returns a sentinel default value (`Immediate(0)`).
    /// Only used for unoccupied slots in [`OperandList`]; never exposed.
    #[inline]
    fn default() -> Self {
        Operand::Immediate(0)
    }
}

/// A parsed instruction before encoding.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Instruction {
    /// Instruction mnemonic (lower-cased), e.g. `"mov"`, `"add"`.
    pub mnemonic: Mnemonic,
    /// Parsed operands (0–4), in Intel order (dest, src, …).
    pub operands: OperandList,
    /// Explicit operand-size hint from a `ptr` qualifier.
    pub size_hint: Option<OperandSize>,
    /// Instruction prefixes (`lock`, `rep`, `repne`, segment overrides).
    pub prefixes: PrefixList,
    /// AVX-512 opmask register decorator (`{k1}`–`{k7}`).
    pub opmask: Option<Register>,
    /// AVX-512 zeroing-masking decorator (`{z}`). Only valid with opmask.
    pub zeroing: bool,
    /// AVX-512 broadcast decorator (`{1to2}`, `{1to4}`, `{1to8}`, `{1to16}`).
    pub broadcast: Option<BroadcastMode>,
    /// Source location of the entire instruction.
    pub span: Span,
}

/// AVX-512 embedded broadcast mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BroadcastMode {
    /// Broadcast 1 element to 2 lanes (`{1to2}`).
    OneToTwo,
    /// Broadcast 1 element to 4 lanes (`{1to4}`).
    OneToFour,
    /// Broadcast 1 element to 8 lanes (`{1to8}`).
    OneToEight,
    /// Broadcast 1 element to 16 lanes (`{1to16}`).
    OneToSixteen,
}

/// Data declaration sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DataSize {
    /// 1 byte (`.byte` / `.db`).
    Byte,
    /// 2 bytes (`.word` / `.dw` / `.short`).
    Word,
    /// 4 bytes (`.long` / `.dd` / `.int`).
    Long,
    /// 8 bytes (`.quad` / `.dq`).
    Quad,
}

impl fmt::Display for DataSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataSize::Byte => write!(f, ".byte"),
            DataSize::Word => write!(f, ".word"),
            DataSize::Long => write!(f, ".long"),
            DataSize::Quad => write!(f, ".quad"),
        }
    }
}

/// A data value in a data directive.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DataValue {
    /// A numeric literal.
    Integer(i128),
    /// A label reference with optional addend — resolved to an address during linking.
    Label(String, i64),
    /// Raw byte sequence (from `.ascii` / `.asciz`).
    Bytes(Vec<u8>),
}

/// A data declaration (.byte, .word, .ascii, etc.).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DataDecl {
    /// Element size for each value.
    pub size: DataSize,
    /// One or more data values.
    pub values: Vec<DataValue>,
    /// Source location.
    pub span: Span,
}

/// Alignment directive.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AlignDirective {
    /// Required alignment in bytes (must be a power of two).
    pub alignment: u32,
    /// Fill byte (when `None`, x86 uses multi-byte NOP sequences).
    pub fill: Option<u8>,
    /// If padding would exceed this many bytes, skip the alignment entirely.
    pub max_skip: Option<u32>,
    /// Source location.
    pub span: Span,
}

/// A constant definition (.equ, .set, =).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstDef {
    /// Constant name.
    pub name: String,
    /// Constant value.
    pub value: i128,
    /// Source location.
    pub span: Span,
}

/// Fill directive (.fill count, size, value).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FillDirective {
    /// Number of repetitions.
    pub count: u32,
    /// Size of each unit in bytes (1–8).
    pub size: u8,
    /// Fill value (stored as i64, truncated to `size` bytes in little-endian).
    pub value: i64,
    /// Source location.
    pub span: Span,
}

/// Space directive (.space n / .skip n).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpaceDirective {
    /// Number of bytes to reserve.
    pub size: u32,
    /// Fill byte value (default 0x00).
    pub fill: u8,
    /// Source location.
    pub span: Span,
}

/// Org directive (.org offset[, fill]).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OrgDirective {
    /// Target byte offset.
    pub offset: u64,
    /// Fill byte for padding (default 0x00).
    pub fill: u8,
    /// Source location.
    pub span: Span,
}

/// x86 code width (operand/address size mode).
///
/// Used by `.code16`, `.code32`, `.code64` directives to switch the default
/// operand and address size within the same assembly unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum X86Mode {
    /// 16-bit real mode: default operand size 16, address size 16.
    Mode16,
    /// 32-bit protected mode: default operand size 32, address size 32.
    Mode32,
    /// 64-bit long mode: default operand size 32, address size 64.
    Mode64,
}

/// A statement in the IR.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(clippy::large_enum_variant)] // Instruction is inline to avoid heap allocations
pub enum Statement {
    /// A label definition.
    Label(String, Span),
    /// An instruction.
    Instruction(Instruction),
    /// A data declaration (`.byte`, `.word`, `.ascii`, etc.).
    Data(DataDecl),
    /// An alignment directive.
    Align(AlignDirective),
    /// A constant definition (`.equ`, `.set`, `=`).
    Const(ConstDef),
    /// A `.fill` directive.
    Fill(FillDirective),
    /// A `.space` / `.skip` directive.
    Space(SpaceDirective),
    /// An `.org` directive.
    Org(OrgDirective),
    /// A code-mode switch (`.code16`, `.code32`, `.code64`).
    CodeMode(X86Mode, Span),
    /// A literal pool flush point (`.ltorg` / `.pool`).
    Ltorg(Span),
    /// A RISC-V `.option rvc` / `.option norvc` directive.
    OptionRvc(bool, Span),
    /// A `.thumb` / `.arm` mode switch directive (true = Thumb, false = ARM).
    ThumbMode(bool, Span),
    /// A `.thumb_func` directive — marks the next label as a Thumb function.
    ThumbFunc(Span),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_size_bits() {
        assert_eq!(Register::Rax.size_bits(), 64);
        assert_eq!(Register::Eax.size_bits(), 32);
        assert_eq!(Register::Ax.size_bits(), 16);
        assert_eq!(Register::Al.size_bits(), 8);
        assert_eq!(Register::R8.size_bits(), 64);
        assert_eq!(Register::R8d.size_bits(), 32);
    }

    #[test]
    fn register_base_code() {
        assert_eq!(Register::Rax.base_code(), 0);
        assert_eq!(Register::Rcx.base_code(), 1);
        assert_eq!(Register::Rdx.base_code(), 2);
        assert_eq!(Register::Rbx.base_code(), 3);
        assert_eq!(Register::Rsp.base_code(), 4);
        assert_eq!(Register::Rbp.base_code(), 5);
        assert_eq!(Register::Rsi.base_code(), 6);
        assert_eq!(Register::Rdi.base_code(), 7);
    }

    #[test]
    fn register_is_extended() {
        assert!(!Register::Rax.is_extended());
        assert!(Register::R8.is_extended());
        assert!(Register::R15d.is_extended());
        assert!(Register::R8b.is_extended());
    }

    #[test]
    fn register_high_byte() {
        assert!(Register::Ah.is_high_byte());
        assert!(Register::Ch.is_high_byte());
        assert!(!Register::Al.is_high_byte());
        assert!(!Register::Spl.is_high_byte());
    }

    #[test]
    fn arch_display() {
        assert_eq!(format!("{}", Arch::X86_64), "x86_64");
        assert_eq!(format!("{}", Arch::Aarch64), "AArch64");
    }

    #[test]
    fn operand_size_bits() {
        assert_eq!(OperandSize::Byte.bits(), 8);
        assert_eq!(OperandSize::Word.bits(), 16);
        assert_eq!(OperandSize::Dword.bits(), 32);
        assert_eq!(OperandSize::Qword.bits(), 64);
        assert_eq!(OperandSize::Xmmword.bits(), 128);
        assert_eq!(OperandSize::Ymmword.bits(), 256);
        assert_eq!(OperandSize::Zmmword.bits(), 512);
    }

    #[test]
    fn register_display() {
        assert_eq!(format!("{}", Register::Rax), "rax");
        assert_eq!(format!("{}", Register::R8d), "r8d");
        assert_eq!(format!("{}", Register::Xmm0), "xmm0");
        assert_eq!(format!("{}", Register::Ymm0), "ymm0");
        assert_eq!(format!("{}", Register::Zmm0), "zmm0");
        assert_eq!(format!("{}", Register::K0), "k0");
    }

    #[test]
    fn operand_size_display() {
        assert_eq!(format!("{}", OperandSize::Byte), "byte");
        assert_eq!(format!("{}", OperandSize::Qword), "qword");
    }

    #[test]
    fn data_size_display() {
        assert_eq!(format!("{}", DataSize::Byte), ".byte");
        assert_eq!(format!("{}", DataSize::Quad), ".quad");
    }

    #[test]
    fn prefix_display() {
        assert_eq!(format!("{}", Prefix::Lock), "lock");
        assert_eq!(format!("{}", Prefix::Rep), "rep");
    }

    #[test]
    fn operand_display() {
        assert_eq!(format!("{}", Operand::Register(Register::Rax)), "rax");
        assert_eq!(format!("{}", Operand::Immediate(42)), "0x2A");
        assert_eq!(format!("{}", Operand::Immediate(-1)), "-0x1");
        assert_eq!(format!("{}", Operand::Label(String::from("loop"))), "loop");

        let mem = MemoryOperand {
            base: Some(Register::Rbp),
            disp: 8,
            ..Default::default()
        };
        assert_eq!(format!("{}", Operand::Memory(Box::new(mem))), "[rbp+0x8]");

        let mem2 = MemoryOperand {
            base: Some(Register::Rbx),
            index: Some(Register::Rcx),
            scale: 4,
            disp: 0,
            size: Some(OperandSize::Dword),
            ..Default::default()
        };
        assert_eq!(
            format!("{}", Operand::Memory(Box::new(mem2))),
            "dword ptr [rbx+rcx*4]"
        );
    }

    #[test]
    fn expr_display() {
        let expr = Expr::Add(
            Box::new(Expr::Label(String::from("foo"))),
            Box::new(Expr::Num(4)),
        );
        assert_eq!(format!("{}", expr), "(foo + 4)");
    }

    #[test]
    fn expr_eval_numeric() {
        let e = Expr::Add(Box::new(Expr::Num(10)), Box::new(Expr::Num(20)));
        assert_eq!(e.eval(), Some(30));
    }

    #[test]
    fn expr_eval_with_label_returns_none() {
        let e = Expr::Add(
            Box::new(Expr::Label(String::from("x"))),
            Box::new(Expr::Num(5)),
        );
        assert_eq!(e.eval(), None);
    }

    #[test]
    fn expr_label_addend_add() {
        let e = Expr::Add(
            Box::new(Expr::Label(String::from("data"))),
            Box::new(Expr::Num(8)),
        );
        assert_eq!(e.label_addend(), Some(("data", 8)));
    }

    #[test]
    fn expr_label_addend_sub() {
        let e = Expr::Sub(
            Box::new(Expr::Label(String::from("data"))),
            Box::new(Expr::Num(3)),
        );
        assert_eq!(e.label_addend(), Some(("data", -3)));
    }

    #[test]
    fn expr_label_addend_plain_label() {
        let e = Expr::Label(String::from("foo"));
        assert_eq!(e.label_addend(), Some(("foo", 0)));
    }

    #[test]
    fn expr_label_addend_two_labels_returns_none() {
        let e = Expr::Add(
            Box::new(Expr::Label(String::from("a"))),
            Box::new(Expr::Label(String::from("b"))),
        );
        assert_eq!(e.label_addend(), None);
    }

    #[test]
    fn expr_resolve_constants() {
        let mut e = Expr::Add(
            Box::new(Expr::Label(String::from("SIZE"))),
            Box::new(Expr::Num(1)),
        );
        e.resolve_constants(|name| if name == "SIZE" { Some(10) } else { None });
        assert_eq!(e.eval(), Some(11));
    }

    #[test]
    fn expr_resolve_constants_partial() {
        let mut e = Expr::Add(
            Box::new(Expr::Label(String::from("SIZE"))),
            Box::new(Expr::Label(String::from("unknown"))),
        );
        e.resolve_constants(|name| if name == "SIZE" { Some(10) } else { None });
        // Still has unresolved label
        assert_eq!(e.eval(), None);
    }
}
