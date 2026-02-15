//! Integration tests for `asm_bytes!` and `asm_array!` proc-macros.

use asm_rs_macros::{asm_array, asm_bytes};

// ── x86-64 ──────────────────────────────────────────────────────────────────

#[test]
fn x86_64_nop() {
    const CODE: &[u8] = asm_bytes!(x86_64, "nop");
    assert_eq!(CODE, &[0x90]);
}

#[test]
fn x86_64_ret() {
    const CODE: &[u8] = asm_bytes!(x86_64, "ret");
    assert_eq!(CODE, &[0xC3]);
}

#[test]
fn x86_64_multi_instruction() {
    const CODE: &[u8] = asm_bytes!(x86_64, "xor eax, eax\ninc eax\nret");
    // xor eax,eax = 31 C0, inc eax = FF C0, ret = C3
    assert_eq!(CODE, &[0x31, 0xC0, 0xFF, 0xC0, 0xC3]);
}

#[test]
fn x86_64_semicolon_separator() {
    const CODE: &[u8] = asm_bytes!(x86_64, "nop; nop; ret");
    assert_eq!(CODE, &[0x90, 0x90, 0xC3]);
}

#[test]
fn x86_64_mov_imm() {
    const CODE: &[u8] = asm_bytes!(x86_64, "mov rax, 1");
    // Optimized to mov eax, 1 (zero-extends to rax in 64-bit mode)
    assert_eq!(CODE.len(), 5);
    assert_eq!(CODE[0], 0xB8); // MOV eax, imm32
}

#[test]
fn x86_64_with_label() {
    const CODE: &[u8] = asm_bytes!(
        x86_64,
        "
        start:
            nop
            jmp start
    "
    );
    // nop (1 byte) + jmp rel8 back by -3 (2 bytes)
    assert_eq!(CODE.len(), 3);
}

#[test]
fn x86_64_array_form() {
    const CODE: [u8; 1] = asm_array!(x86_64, "nop");
    assert_eq!(CODE, [0x90]);
}

#[test]
fn x86_64_array_multi() {
    const CODE: [u8; 5] = asm_array!(x86_64, "xor eax, eax\ninc eax\nret");
    assert_eq!(CODE, [0x31, 0xC0, 0xFF, 0xC0, 0xC3]);
}

#[test]
fn x86_64_with_base_address() {
    const CODE: &[u8] = asm_bytes!(x86_64, 0x400000, "nop\nret");
    assert_eq!(CODE, &[0x90, 0xC3]);
}

// ── x86 (32-bit) ────────────────────────────────────────────────────────────

#[test]
fn x86_nop() {
    const CODE: &[u8] = asm_bytes!(x86, "nop");
    assert_eq!(CODE, &[0x90]);
}

#[test]
fn x86_ret() {
    const CODE: &[u8] = asm_bytes!(x86, "ret");
    assert_eq!(CODE, &[0xC3]);
}

// ── AArch64 ─────────────────────────────────────────────────────────────────

#[test]
fn aarch64_nop() {
    const CODE: &[u8] = asm_bytes!(aarch64, "nop");
    assert_eq!(CODE, &[0x1F, 0x20, 0x03, 0xD5]);
}

#[test]
fn aarch64_ret() {
    const CODE: &[u8] = asm_bytes!(aarch64, "ret");
    assert_eq!(CODE, &[0xC0, 0x03, 0x5F, 0xD6]);
}

#[test]
fn aarch64_multi() {
    const CODE: &[u8] = asm_bytes!(aarch64, "nop\nret");
    assert_eq!(CODE.len(), 8);
}

#[test]
fn aarch64_array() {
    const CODE: [u8; 4] = asm_array!(aarch64, "nop");
    assert_eq!(CODE, [0x1F, 0x20, 0x03, 0xD5]);
}

// ── ARM ─────────────────────────────────────────────────────────────────────

#[test]
fn arm_nop() {
    const CODE: &[u8] = asm_bytes!(arm, "nop");
    assert_eq!(CODE.len(), 4);
}

#[test]
fn arm_bx_lr() {
    const CODE: &[u8] = asm_bytes!(arm, "bx lr");
    assert_eq!(CODE, &[0x1E, 0xFF, 0x2F, 0xE1]);
}

// ── Thumb ───────────────────────────────────────────────────────────────────

#[test]
fn thumb_nop() {
    const CODE: &[u8] = asm_bytes!(thumb, "nop");
    assert_eq!(CODE, &[0x00, 0xBF]); // Thumb NOP (0xBF00 little-endian)
}

// ── RISC-V ──────────────────────────────────────────────────────────────────

#[test]
fn rv64_nop() {
    const CODE: &[u8] = asm_bytes!(rv64, "nop");
    // addi x0, x0, 0
    assert_eq!(CODE.len(), 4);
    assert_eq!(CODE, &[0x13, 0x00, 0x00, 0x00]);
}

#[test]
fn rv32_nop() {
    const CODE: &[u8] = asm_bytes!(rv32, "nop");
    assert_eq!(CODE, &[0x13, 0x00, 0x00, 0x00]);
}

#[test]
fn rv64_ret() {
    const CODE: &[u8] = asm_bytes!(rv64, "ret");
    // jalr x0, x1, 0
    assert_eq!(CODE.len(), 4);
}

#[test]
fn rv64_array() {
    const CODE: [u8; 4] = asm_array!(rv64, "nop");
    assert_eq!(CODE, [0x13, 0x00, 0x00, 0x00]);
}

// ── Multi-line / complex ────────────────────────────────────────────────────

#[test]
fn x86_64_raw_string() {
    const CODE: &[u8] = asm_bytes!(
        x86_64,
        r#"
        nop
        nop
        ret
    "#
    );
    assert_eq!(CODE, &[0x90, 0x90, 0xC3]);
}

#[test]
fn const_in_static_context() {
    // Verify the output can be used in any const context
    static BYTES: &[u8] = asm_bytes!(x86_64, "ret");
    assert_eq!(BYTES, &[0xC3]);
}

#[test]
fn array_in_static_context() {
    static BYTES: [u8; 1] = asm_array!(x86_64, "ret");
    assert_eq!(BYTES, [0xC3]);
}

#[test]
fn x86_64_data_directive() {
    const CODE: &[u8] = asm_bytes!(x86_64, ".byte 0xCC");
    assert_eq!(CODE, &[0xCC]);
}

#[test]
fn base_address_integer_literal() {
    const CODE: &[u8] = asm_bytes!(x86_64, 4096, "nop");
    assert_eq!(CODE, &[0x90]);
}
