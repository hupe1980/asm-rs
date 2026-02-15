//! Basic assembly example — demonstrates the one-shot and builder APIs.
//!
//! Run with: `cargo run --example basic`

use asm_rs::{assemble, Arch, Assembler};

fn main() {
    println!("=== asm_rs basic example ===\n");

    // --- One-shot assembly ---
    println!("1. One-shot assembly (mov eax, 42; ret):");
    let bytes = assemble("mov eax, 42\nret", Arch::X86_64).unwrap();
    print_hex("   ", &bytes);

    // --- Builder API ---
    println!("\n2. Builder API (function prologue/epilogue):");
    let mut asm = Assembler::new(Arch::X86_64);
    asm.base_address(0x40_1000);
    asm.emit(
        r#"
entry:
    push rbp
    mov rbp, rsp
    sub rsp, 0x20
    # function body would go here
    xor eax, eax        # return 0
    add rsp, 0x20
    pop rbp
    ret
"#,
    )
    .unwrap();

    let result = asm.finish().unwrap();
    print_hex("   ", result.bytes());

    // Label addresses
    println!("\n   Labels:");
    for (name, addr) in result.labels() {
        println!("   {}: 0x{:X}", name, addr);
    }

    // Listing output
    println!("\n   Listing:");
    for line in result.listing().lines() {
        println!("   {}", line);
    }

    // --- Data directives ---
    println!("\n3. Data directives:");
    let bytes = assemble(
        r#"
.asciz "Hello, asm_rs!"
.byte 0xDE, 0xAD, 0xBE, 0xEF
.quad 0x0123456789ABCDEF
"#,
        Arch::X86_64,
    )
    .unwrap();
    print_hex("   ", &bytes);

    // --- Constants ---
    println!("\n4. Constants (.equ):");
    let bytes = assemble(
        r#"
.equ SYS_EXIT, 60
.equ EXIT_SUCCESS, 0
    mov eax, SYS_EXIT
    mov edi, EXIT_SUCCESS
    syscall
"#,
        Arch::X86_64,
    )
    .unwrap();
    print_hex("   ", &bytes);

    // --- Branch relaxation ---
    println!("\n5. Branch relaxation (short vs long):");
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit("je done\ndone:\nret").unwrap();
    let short = asm.finish().unwrap();
    println!("   Short branch: {} bytes", short.len());
    print_hex("   ", short.bytes());

    println!("\n=== Done! ===");
}

fn print_hex(prefix: &str, bytes: &[u8]) {
    print!("{}", prefix);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && i % 16 == 0 {
            println!();
            print!("{}", prefix);
        }
        print!("{:02X} ", b);
    }
    println!();
}
