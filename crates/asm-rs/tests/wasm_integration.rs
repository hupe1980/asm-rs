//! WASM integration tests for asm-rs.
//!
//! These tests verify that asm-rs produces correct machine code when running
//! inside a WebAssembly environment. Run with:
//!
//! ```sh
//! cargo test --target wasm32-unknown-unknown --test wasm_integration --all-features
//! ```
//!
//! Requires: `wasm-bindgen-test-runner` (install via `cargo install wasm-bindgen-cli`)

#![cfg(target_arch = "wasm32")]

use asm_rs::{assemble, Arch};
use wasm_bindgen_test::*;

// --- x86-64 ---

#[wasm_bindgen_test]
fn wasm_x86_64_nop() {
    let code = assemble("nop", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x90]);
}

#[wasm_bindgen_test]
fn wasm_x86_64_ret() {
    let code = assemble("ret", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xc3]);
}

#[wasm_bindgen_test]
fn wasm_x86_64_mov_eax_42() {
    let code = assemble("mov eax, 42", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xb8, 0x2a, 0x00, 0x00, 0x00]);
}

#[wasm_bindgen_test]
fn wasm_x86_64_xor_eax_eax() {
    let code = assemble("xor eax, eax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x31, 0xc0]);
}

#[wasm_bindgen_test]
fn wasm_x86_64_push_rbp_mov_rbp_rsp() {
    let code = assemble("push rbp\nmov rbp, rsp", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x55, 0x48, 0x89, 0xe5]);
}

#[wasm_bindgen_test]
fn wasm_x86_64_syscall() {
    let code = assemble("syscall", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x05]);
}

#[wasm_bindgen_test]
fn wasm_x86_64_labels() {
    let code = assemble("start: jmp start", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xeb, 0xfe]); // short jump to self
}

#[wasm_bindgen_test]
fn wasm_x86_64_shellcode_pattern() {
    let code = assemble(
        r#"
        xor eax, eax
        mov al, 60
        xor edi, edi
        syscall
    "#,
        Arch::X86_64,
    )
    .unwrap();
    assert_eq!(code, vec![0x31, 0xc0, 0xb0, 0x3c, 0x31, 0xff, 0x0f, 0x05]);
}

// --- x86-32 ---

#[wasm_bindgen_test]
fn wasm_x86_32_mov_eax_1() {
    let code = assemble("mov eax, 1", Arch::X86).unwrap();
    assert_eq!(code, vec![0xb8, 0x01, 0x00, 0x00, 0x00]);
}

#[wasm_bindgen_test]
fn wasm_x86_32_inc_eax() {
    let code = assemble("inc eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x40]); // 32-bit short form
}

#[wasm_bindgen_test]
fn wasm_x86_32_int_0x80() {
    let code = assemble("int 0x80", Arch::X86).unwrap();
    assert_eq!(code, vec![0xcd, 0x80]);
}

// --- AArch64 ---

#[wasm_bindgen_test]
fn wasm_aarch64_nop() {
    let code = assemble("nop", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x1f, 0x20, 0x03, 0xd5]);
}

#[wasm_bindgen_test]
fn wasm_aarch64_ret() {
    let code = assemble("ret", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0xc0, 0x03, 0x5f, 0xd6]);
}

#[wasm_bindgen_test]
fn wasm_aarch64_mov_x0_0() {
    let code = assemble("mov x0, 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x80, 0xd2]);
}

#[wasm_bindgen_test]
fn wasm_aarch64_add_x0_x1_x2() {
    let code = assemble("add x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x02, 0x8b]);
}

#[wasm_bindgen_test]
fn wasm_aarch64_svc_0() {
    let code = assemble("svc 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x01, 0x00, 0x00, 0xd4]);
}

// --- ARM32 ---

#[wasm_bindgen_test]
fn wasm_arm32_mov_r0_1() {
    let code = assemble("mov r0, 1", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x01, 0x00, 0xa0, 0xe3]);
}

#[wasm_bindgen_test]
fn wasm_arm32_add_r0_r1_r2() {
    let code = assemble("add r0, r1, r2", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x02, 0x00, 0x81, 0xe0]);
}

#[wasm_bindgen_test]
fn wasm_arm32_nop() {
    // ARM32 NOP = MOV R0, R0 → E1A00000
    let code = assemble("nop", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0xA0, 0xE1]);
}

// --- Thumb ---

#[wasm_bindgen_test]
fn wasm_thumb_nop() {
    let code = assemble("nop", Arch::Thumb).unwrap();
    assert_eq!(code, vec![0x00, 0xbf]);
}

#[wasm_bindgen_test]
fn wasm_thumb_mov_r0_42() {
    let code = assemble("mov r0, 42", Arch::Thumb).unwrap();
    assert_eq!(code, vec![0x2a, 0x20]);
}

// --- RISC-V 64 ---

#[wasm_bindgen_test]
fn wasm_riscv64_addi_a0_zero_42() {
    let code = assemble("addi a0, zero, 42", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x13, 0x05, 0xa0, 0x02]);
}

#[wasm_bindgen_test]
fn wasm_riscv64_add_a0_a1_a2() {
    let code = assemble("add a0, a1, a2", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0x85, 0xc5, 0x00]);
}

#[wasm_bindgen_test]
fn wasm_riscv64_ecall() {
    let code = assemble("ecall", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x73, 0x00, 0x00, 0x00]);
}

// --- RISC-V 32 ---

#[wasm_bindgen_test]
fn wasm_riscv32_addi_a0_zero_1() {
    let code = assemble("addi a0, zero, 1", Arch::Rv32).unwrap();
    assert_eq!(code, vec![0x13, 0x05, 0x10, 0x00]);
}

// --- Multi-instruction / Builder patterns ---

#[wasm_bindgen_test]
fn wasm_multi_arch_data_directives() {
    let code = assemble(".byte 0xDE, 0xAD, 0xBE, 0xEF", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xDE, 0xAD, 0xBE, 0xEF]);
}

#[wasm_bindgen_test]
fn wasm_x86_64_forward_reference() {
    let code = assemble("jmp end\nnop\nend: ret", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xeb, 0x01, 0x90, 0xc3]);
}

#[wasm_bindgen_test]
fn wasm_error_handling() {
    let result = assemble("invalid_mnemonic_xyz", Arch::X86_64);
    assert!(result.is_err());
}
