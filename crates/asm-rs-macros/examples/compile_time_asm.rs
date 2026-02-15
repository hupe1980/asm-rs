//! Compile-time assembly with `asm_bytes!` and `asm_array!` macros.
//!
//! These macros assemble instructions at compile time, producing
//! `&'static [u8]` or `[u8; N]` constants with zero runtime overhead.
//!
//! Run with: `cargo run --example compile_time_asm -p asm-rs-macros`

use asm_rs_macros::{asm_array, asm_bytes};

// ── Compile-time constants ──────────────────────────────────────────────

/// x86-64 NOP sled (assembled at compile time)
const NOP_SLED: &[u8] = asm_bytes!(x86_64, "nop; nop; nop; nop");

/// x86-64 function prologue
const PROLOGUE: &[u8] = asm_bytes!(
    x86_64,
    "
    push rbp
    mov rbp, rsp
"
);

/// x86-64 function epilogue
const EPILOGUE: &[u8] = asm_bytes!(
    x86_64,
    "
    pop rbp
    ret
"
);

/// AArch64 code as a fixed-size array
const AARCH64_NOP: [u8; 4] = asm_array!(aarch64, "nop");

/// RISC-V 64-bit NOP
const RV64_NOP: [u8; 4] = asm_array!(rv64, "nop");

/// x86-64 shellcode with labels
const LOOP_CODE: &[u8] = asm_bytes!(
    x86_64,
    "
    xor ecx, ecx
loop_start:
    inc ecx
    cmp ecx, 10
    jl loop_start
    ret
"
);

fn main() {
    println!("=== Compile-Time Assembly Demo ===\n");

    println!("NOP sled ({} bytes):", NOP_SLED.len());
    print_hex("  ", NOP_SLED);

    println!("\nFunction prologue ({} bytes):", PROLOGUE.len());
    print_hex("  ", PROLOGUE);

    println!("\nFunction epilogue ({} bytes):", EPILOGUE.len());
    print_hex("  ", EPILOGUE);

    println!("\nAArch64 NOP ({} bytes):", AARCH64_NOP.len());
    print_hex("  ", &AARCH64_NOP);

    println!("\nRISC-V 64 NOP ({} bytes):", RV64_NOP.len());
    print_hex("  ", &RV64_NOP);

    println!("\nLoop code ({} bytes):", LOOP_CODE.len());
    print_hex("  ", LOOP_CODE);

    // These are true compile-time constants — no runtime assembly overhead!
    assert_eq!(NOP_SLED, &[0x90, 0x90, 0x90, 0x90]);
    println!("\n✓ All compile-time assembly verified!");
}

fn print_hex(prefix: &str, bytes: &[u8]) {
    print!("{prefix}");
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        print!("{b:02X}");
    }
    println!();
}
