#![cfg(not(target_arch = "wasm32"))]
//! Property-based tests using proptest.
//!
//! These tests verify assembler invariants across large, randomly generated
//! input spaces — complementing the targeted unit/integration tests and the
//! libfuzzer-based fuzz targets.

use asm_rs::{assemble, assemble_at, Arch, Assembler};
use proptest::prelude::*;

// ── Strategies ──────────────────────────────────────────────────────────

/// Generates arbitrary ASCII strings (the assembler only accepts text input).
fn arb_asm_input() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::char::range('\0', '\x7f'), 0..256)
        .prop_map(|v| v.into_iter().collect())
}

/// Generates valid x86_64 instruction strings from a curated pool.
fn valid_x86_64_insn() -> impl Strategy<Value = &'static str> {
    prop::sample::select(vec![
        "nop",
        "ret",
        "int3",
        "hlt",
        "clc",
        "stc",
        "cmc",
        "cld",
        "std",
        "pause",
        "xor eax, eax",
        "xor rax, rax",
        "mov eax, 42",
        "mov rax, 0x1234",
        "add eax, 1",
        "sub rax, 8",
        "inc ecx",
        "dec rdx",
        "push rax",
        "pop rbx",
        "push rbp",
        "pop rbp",
        "and eax, 0xFF",
        "or eax, 0x80",
        "xor ecx, edx",
        "shl eax, 1",
        "shr rax, 4",
        "test eax, eax",
        "cmp eax, 0",
        "neg eax",
        "not rax",
        "mov al, 0",
        "mov ah, 0",
        "movzx eax, cl",
        "movsx rax, eax",
        "bswap eax",
        "bswap rax",
        "cdq",
        "cqo",
        "cbw",
        "cwde",
        "cdqe",
        "syscall",
        "cpuid",
        "rdtsc",
        "xchg eax, ebx",
        "rol eax, 1",
        "ror rax, 8",
        "bt eax, 3",
        "bts eax, 7",
        "btr eax, 15",
        "btc eax, 31",
        "bsf eax, ecx",
        "bsr rax, rdx",
        "popcnt eax, ecx",
        "lzcnt eax, ecx",
        "tzcnt eax, ecx",
    ])
}

/// Generates valid ARM instruction strings from a curated pool.
fn valid_arm_insn() -> impl Strategy<Value = &'static str> {
    prop::sample::select(vec![
        "nop",
        "bx lr",
        "mov r0, 0",
        "mov r0, 42",
        "add r0, r1, r2",
        "sub r0, r1, 1",
        "and r0, r1, r2",
        "orr r0, r1, r2",
        "eor r0, r1, r2",
        "cmp r0, 0",
        "tst r0, r1",
        "mvn r0, r1",
        "mul r0, r1, r2",
        "push {lr}",
        "pop {pc}",
    ])
}

/// Generates valid AArch64 instruction strings from a curated pool.
fn valid_aarch64_insn() -> impl Strategy<Value = &'static str> {
    prop::sample::select(vec![
        "nop",
        "ret",
        "mov x0, 0",
        "mov x0, 42",
        "mov w0, 0",
        "add x0, x1, x2",
        "sub x0, x1, 1",
        "and x0, x1, x2",
        "orr x0, x1, x2",
        "eor x0, x1, x2",
        "lsl x0, x1, 1",
        "lsr x0, x1, 4",
        "cmp x0, 0",
        "tst x0, x1",
        "mvn x0, x1",
        "neg x0, x1",
        "mul x0, x1, x2",
        "svc 0",
    ])
}

/// Generates valid RISC-V instruction strings from a curated pool.
fn valid_riscv_insn() -> impl Strategy<Value = &'static str> {
    prop::sample::select(vec![
        "nop",
        "ret",
        "addi x0, x0, 0",
        "addi x1, x0, 42",
        "add x1, x2, x3",
        "sub x1, x2, x3",
        "and x1, x2, x3",
        "or x1, x2, x3",
        "xor x1, x2, x3",
        "sll x1, x2, x3",
        "srl x1, x2, x3",
        "sra x1, x2, x3",
        "slt x1, x2, x3",
        "sltu x1, x2, x3",
        "lui x1, 0x12345",
        "auipc x1, 0",
        "ecall",
        "ebreak",
    ])
}

/// Generates valid x86 (32-bit) instruction strings from a curated pool.
fn valid_x86_insn() -> impl Strategy<Value = &'static str> {
    prop::sample::select(vec![
        "nop",
        "ret",
        "int3",
        "hlt",
        "clc",
        "stc",
        "cmc",
        "cld",
        "std",
        "xor eax, eax",
        "mov eax, 42",
        "add eax, 1",
        "sub eax, 8",
        "inc ecx",
        "dec edx",
        "push eax",
        "pop ebx",
        "push ebp",
        "pop ebp",
        "and eax, 0xFF",
        "or eax, 0x80",
        "xor ecx, edx",
        "shl eax, 1",
        "shr eax, 4",
        "test eax, eax",
        "cmp eax, 0",
        "neg eax",
        "not eax",
        "mov al, 0",
        "cdq",
        "cbw",
        "cwde",
        "xchg eax, ebx",
        "rol eax, 1",
        "ror eax, 8",
        "push es",
        "push cs",
        "push ds",
        "push ss",
        "pop es",
        "pop ds",
    ])
}

/// Generates valid Thumb instruction strings from a curated pool.
fn valid_thumb_insn() -> impl Strategy<Value = &'static str> {
    prop::sample::select(vec![
        "nop",
        "bkpt 0",
        "mov r0, 0",
        "mov r0, 42",
        "mov r1, 100",
        "add r1, r2, r3",
        "sub r1, r2, r3",
        "and r0, r1",
        "orr r0, r1",
        "eor r0, r1",
        "cmp r0, 0",
        "tst r0, r1",
        "mvn r0, r1",
        "neg r0, r1",
        "mul r0, r1",
        "push {lr}",
        "pop {pc}",
    ])
}

// ── Property: No panics on arbitrary input ──────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// The assembler must NEVER panic on arbitrary input — only Ok/Err.
    #[test]
    fn no_panic_on_arbitrary_x86_64(input in arb_asm_input()) {
        let _ = assemble(&input, Arch::X86_64);
    }

    /// Same for assemble_at with arbitrary base addresses.
    #[test]
    fn no_panic_on_arbitrary_x86_64_at(input in arb_asm_input(), base in any::<u64>()) {
        let _ = assemble_at(&input, Arch::X86_64, base);
    }

    /// Builder API must not panic on arbitrary input.
    #[test]
    fn no_panic_builder_api(input in arb_asm_input()) {
        let mut asm = Assembler::new(Arch::X86_64);
        for line in input.lines() {
            if asm.emit(line).is_err() {
                return Ok(());
            }
        }
        let _ = asm.finish();
    }

    /// ARM assembler must not panic on arbitrary input.
    #[test]
    fn no_panic_on_arbitrary_arm(input in arb_asm_input()) {
        let _ = assemble(&input, Arch::Arm);
    }

    /// AArch64 assembler must not panic on arbitrary input.
    #[test]
    fn no_panic_on_arbitrary_aarch64(input in arb_asm_input()) {
        let _ = assemble(&input, Arch::Aarch64);
    }

    /// RISC-V assembler must not panic on arbitrary input.
    #[test]
    fn no_panic_on_arbitrary_riscv(input in arb_asm_input()) {
        let _ = assemble(&input, Arch::Rv64);
    }

    /// x86 (32-bit) assembler must not panic on arbitrary input.
    #[test]
    fn no_panic_on_arbitrary_x86(input in arb_asm_input()) {
        let _ = assemble(&input, Arch::X86);
    }

    /// Thumb assembler must not panic on arbitrary input.
    #[test]
    fn no_panic_on_arbitrary_thumb(input in arb_asm_input()) {
        let _ = assemble(&input, Arch::Thumb);
    }

    /// RISC-V 32-bit assembler must not panic on arbitrary input.
    #[test]
    fn no_panic_on_arbitrary_rv32(input in arb_asm_input()) {
        let _ = assemble(&input, Arch::Rv32);
    }
}

// ── Property: Valid instructions always succeed ─────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn valid_x86_64_always_assembles(insn in valid_x86_64_insn()) {
        let result = assemble(insn, Arch::X86_64);
        prop_assert!(result.is_ok(), "Failed to assemble: {}", insn);
        prop_assert!(!result.unwrap().is_empty(), "Empty output: {}", insn);
    }

    #[test]
    fn valid_arm_always_assembles(insn in valid_arm_insn()) {
        let result = assemble(insn, Arch::Arm);
        prop_assert!(result.is_ok(), "Failed to assemble ARM: {}", insn);
        prop_assert!(!result.unwrap().is_empty(), "Empty ARM output: {}", insn);
    }

    #[test]
    fn valid_aarch64_always_assembles(insn in valid_aarch64_insn()) {
        let result = assemble(insn, Arch::Aarch64);
        prop_assert!(result.is_ok(), "Failed to assemble AArch64: {}", insn);
        prop_assert!(!result.unwrap().is_empty(), "Empty AArch64 output: {}", insn);
    }

    #[test]
    fn valid_riscv_always_assembles(insn in valid_riscv_insn()) {
        let result = assemble(insn, Arch::Rv64);
        prop_assert!(result.is_ok(), "Failed to assemble RISC-V: {}", insn);
        prop_assert!(!result.unwrap().is_empty(), "Empty RISC-V output: {}", insn);
    }

    #[test]
    fn valid_x86_always_assembles(insn in valid_x86_insn()) {
        let result = assemble(insn, Arch::X86);
        prop_assert!(result.is_ok(), "Failed to assemble x86: {}", insn);
        prop_assert!(!result.unwrap().is_empty(), "Empty x86 output: {}", insn);
    }

    #[test]
    fn valid_thumb_always_assembles(insn in valid_thumb_insn()) {
        let result = assemble(insn, Arch::Thumb);
        prop_assert!(result.is_ok(), "Failed to assemble Thumb: {}", insn);
        prop_assert!(!result.unwrap().is_empty(), "Empty Thumb output: {}", insn);
    }

    #[test]
    fn valid_rv32_always_assembles(insn in valid_riscv_insn()) {
        let result = assemble(insn, Arch::Rv32);
        prop_assert!(result.is_ok(), "Failed to assemble RV32: {}", insn);
        prop_assert!(!result.unwrap().is_empty(), "Empty RV32 output: {}", insn);
    }
}

// ── Property: Determinism ───────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn deterministic_x86_64(insn in valid_x86_64_insn()) {
        let r1 = assemble(insn, Arch::X86_64).unwrap();
        let r2 = assemble(insn, Arch::X86_64).unwrap();
        prop_assert_eq!(r1, r2);
    }

    #[test]
    fn deterministic_arm(insn in valid_arm_insn()) {
        let r1 = assemble(insn, Arch::Arm).unwrap();
        let r2 = assemble(insn, Arch::Arm).unwrap();
        prop_assert_eq!(r1, r2);
    }

    #[test]
    fn deterministic_aarch64(insn in valid_aarch64_insn()) {
        let r1 = assemble(insn, Arch::Aarch64).unwrap();
        let r2 = assemble(insn, Arch::Aarch64).unwrap();
        prop_assert_eq!(r1, r2);
    }

    #[test]
    fn deterministic_riscv(insn in valid_riscv_insn()) {
        let r1 = assemble(insn, Arch::Rv64).unwrap();
        let r2 = assemble(insn, Arch::Rv64).unwrap();
        prop_assert_eq!(r1, r2);
    }

    #[test]
    fn deterministic_x86(insn in valid_x86_insn()) {
        let r1 = assemble(insn, Arch::X86).unwrap();
        let r2 = assemble(insn, Arch::X86).unwrap();
        prop_assert_eq!(r1, r2);
    }

    #[test]
    fn deterministic_thumb(insn in valid_thumb_insn()) {
        let r1 = assemble(insn, Arch::Thumb).unwrap();
        let r2 = assemble(insn, Arch::Thumb).unwrap();
        prop_assert_eq!(r1, r2);
    }

    #[test]
    fn deterministic_rv32(insn in valid_riscv_insn()) {
        let r1 = assemble(insn, Arch::Rv32).unwrap();
        let r2 = assemble(insn, Arch::Rv32).unwrap();
        prop_assert_eq!(r1, r2);
    }
}

// ── Property: base_address offset ───────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Instructions with no labels should produce identical bytes regardless
    /// of base address (since there are no relocations to resolve).
    #[test]
    fn base_address_independent_for_label_free_x86_64(
        insn in valid_x86_64_insn(),
        base1 in any::<u64>(),
        base2 in any::<u64>(),
    ) {
        // Skip RIP-relative instructions.
        if insn.contains("rip") {
            return Ok(());
        }
        let r1 = assemble_at(insn, Arch::X86_64, base1).unwrap();
        let r2 = assemble_at(insn, Arch::X86_64, base2).unwrap();
        prop_assert_eq!(r1, r2);
    }
}

// ── Property: Instruction length bounds ─────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn x86_64_length_bounds(insn in valid_x86_64_insn()) {
        let output = assemble(insn, Arch::X86_64).unwrap();
        prop_assert!(!output.is_empty() && output.len() <= 32,
            "Unexpected length {} for: {}", output.len(), insn);
    }

    #[test]
    fn arm_always_4_bytes(insn in valid_arm_insn()) {
        let output = assemble(insn, Arch::Arm).unwrap();
        prop_assert_eq!(output.len(), 4);
    }

    #[test]
    fn aarch64_always_4_bytes(insn in valid_aarch64_insn()) {
        let output = assemble(insn, Arch::Aarch64).unwrap();
        prop_assert_eq!(output.len(), 4);
    }

    #[test]
    fn riscv_length_2_or_4(insn in valid_riscv_insn()) {
        let output = assemble(insn, Arch::Rv64).unwrap();
        prop_assert!(output.len() == 2 || output.len() == 4,
            "RISC-V length {} for: {}", output.len(), insn);
    }

    #[test]
    fn x86_length_bounds(insn in valid_x86_insn()) {
        let output = assemble(insn, Arch::X86).unwrap();
        prop_assert!(!output.is_empty() && output.len() <= 32,
            "Unexpected x86 length {} for: {}", output.len(), insn);
    }

    #[test]
    fn thumb_length_2_or_4(insn in valid_thumb_insn()) {
        let output = assemble(insn, Arch::Thumb).unwrap();
        prop_assert!(output.len() == 2 || output.len() == 4,
            "Thumb length {} for: {}", output.len(), insn);
    }

    #[test]
    fn rv32_length_2_or_4(insn in valid_riscv_insn()) {
        let output = assemble(insn, Arch::Rv32).unwrap();
        prop_assert!(output.len() == 2 || output.len() == 4,
            "RV32 length {} for: {}", output.len(), insn);
    }
}

// ── Property: Multi-instruction programs ────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Assembling N instructions must produce output whose length equals
    /// the sum of individual instruction lengths (no label relocations).
    #[test]
    fn multi_insn_length_additive_x86_64(
        insns in prop::collection::vec(valid_x86_64_insn(), 1..8)
    ) {
        // Skip rip-relative instructions.
        let insns: Vec<_> = insns.into_iter().filter(|i| !i.contains("rip")).collect();
        if insns.is_empty() {
            return Ok(());
        }
        let individual_total: usize = insns.iter()
            .map(|i| assemble(i, Arch::X86_64).unwrap().len())
            .sum();
        let combined = insns.join("\n");
        let combined_output = assemble(&combined, Arch::X86_64).unwrap();
        prop_assert_eq!(combined_output.len(), individual_total);
    }

    #[test]
    fn multi_insn_length_additive_arm(
        insns in prop::collection::vec(valid_arm_insn(), 1..8)
    ) {
        let individual_total: usize = insns.iter()
            .map(|i| assemble(i, Arch::Arm).unwrap().len())
            .sum();
        let combined = insns.join("\n");
        let combined_output = assemble(&combined, Arch::Arm).unwrap();
        prop_assert_eq!(combined_output.len(), individual_total);
    }

    #[test]
    fn multi_insn_length_additive_aarch64(
        insns in prop::collection::vec(valid_aarch64_insn(), 1..8)
    ) {
        let individual_total: usize = insns.iter()
            .map(|i| assemble(i, Arch::Aarch64).unwrap().len())
            .sum();
        let combined = insns.join("\n");
        let combined_output = assemble(&combined, Arch::Aarch64).unwrap();
        prop_assert_eq!(combined_output.len(), individual_total);
    }

    #[test]
    fn multi_insn_length_additive_riscv(
        insns in prop::collection::vec(valid_riscv_insn(), 1..8)
    ) {
        let individual_total: usize = insns.iter()
            .map(|i| assemble(i, Arch::Rv64).unwrap().len())
            .sum();
        let combined = insns.join("\n");
        let combined_output = assemble(&combined, Arch::Rv64).unwrap();
        prop_assert_eq!(combined_output.len(), individual_total);
    }

    #[test]
    fn multi_insn_length_additive_x86(
        insns in prop::collection::vec(valid_x86_insn(), 1..8)
    ) {
        let individual_total: usize = insns.iter()
            .map(|i| assemble(i, Arch::X86).unwrap().len())
            .sum();
        let combined = insns.join("\n");
        let combined_output = assemble(&combined, Arch::X86).unwrap();
        prop_assert_eq!(combined_output.len(), individual_total);
    }

    #[test]
    fn multi_insn_length_additive_thumb(
        insns in prop::collection::vec(valid_thumb_insn(), 1..8)
    ) {
        let individual_total: usize = insns.iter()
            .map(|i| assemble(i, Arch::Thumb).unwrap().len())
            .sum();
        let combined = insns.join("\n");
        let combined_output = assemble(&combined, Arch::Thumb).unwrap();
        prop_assert_eq!(combined_output.len(), individual_total);
    }

    #[test]
    fn multi_insn_length_additive_rv32(
        insns in prop::collection::vec(valid_riscv_insn(), 1..8)
    ) {
        let individual_total: usize = insns.iter()
            .map(|i| assemble(i, Arch::Rv32).unwrap().len())
            .sum();
        let combined = insns.join("\n");
        let combined_output = assemble(&combined, Arch::Rv32).unwrap();
        prop_assert_eq!(combined_output.len(), individual_total);
    }
}

// ── Property: Constant substitution ─────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// `.set` constant substitution must produce the same bytes as literal.
    #[test]
    fn constant_substitution_x86_64(val in 0i32..=127) {
        let with_const = format!(".set MY_CONST, {val}\nadd eax, MY_CONST");
        let direct = format!("add eax, {val}");
        let r1 = assemble(&with_const, Arch::X86_64).unwrap();
        let r2 = assemble(&direct, Arch::X86_64).unwrap();
        prop_assert_eq!(r1, r2);
    }

    /// Constant substitution for ARM.
    #[test]
    fn constant_substitution_arm(val in 0u32..=255) {
        let with_const = format!(".set MY_CONST, {val}\nmov r0, MY_CONST");
        let direct = format!("mov r0, {val}");
        let r1 = assemble(&with_const, Arch::Arm).unwrap();
        let r2 = assemble(&direct, Arch::Arm).unwrap();
        prop_assert_eq!(r1, r2);
    }
}

// ── Property: Label-based programs ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Programs with labels and forward/backward jumps must assemble
    /// without panicking and produce non-empty output.
    #[test]
    fn label_programs_no_panic_x86_64(
        nops_before in 0u32..5,
        nops_after in 0u32..5,
    ) {
        let mut prog = String::new();
        prog.push_str("start:\n");
        for _ in 0..nops_before {
            prog.push_str("  nop\n");
        }
        prog.push_str("  jmp end\n");
        for _ in 0..nops_after {
            prog.push_str("  nop\n");
        }
        prog.push_str("end:\n");
        prog.push_str("  ret\n");

        let result = assemble(&prog, Arch::X86_64);
        prop_assert!(result.is_ok(), "Label program failed");
        prop_assert!(!result.unwrap().is_empty());
    }
}
