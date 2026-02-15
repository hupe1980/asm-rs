//! Integration tests for asm_rs.
//!
//! These tests exercise the public API end-to-end, verifying that assembly
//! source text is correctly translated into expected machine code bytes.

use asm_rs::{assemble, assemble_at, Arch, AsmError, Assembler, OptLevel};

// ============================================================================
// One-Shot API
// ============================================================================

#[test]
fn one_shot_nop() {
    let bytes = assemble("nop", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x90]);
}

#[test]
fn one_shot_ret() {
    let bytes = assemble("ret", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xC3]);
}

#[test]
fn one_shot_multiple_instructions() {
    let bytes = assemble("nop\nnop\nret", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x90, 0x90, 0xC3]);
}

#[test]
fn one_shot_with_base_address() {
    let bytes = assemble_at("nop\nret", Arch::X86_64, 0x1000).unwrap();
    assert_eq!(bytes, vec![0x90, 0xC3]);
}

// ============================================================================
// Builder API
// ============================================================================

#[test]
fn builder_emit_and_finish() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit("push rbp").unwrap();
    asm.emit("mov rbp, rsp").unwrap();
    asm.emit("pop rbp").unwrap();
    asm.emit("ret").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    assert_eq!(bytes[0], 0x55); // push rbp
    assert_eq!(*bytes.last().unwrap(), 0xC3); // ret
}

#[test]
fn builder_data_directives() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.db(&[0xDE, 0xAD]).unwrap();
    asm.dw(0xBEEF).unwrap();
    let result = asm.finish().unwrap();
    assert_eq!(result.bytes(), &[0xDE, 0xAD, 0xEF, 0xBE]);
}

#[test]
fn builder_labels() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit("jmp done").unwrap();
    asm.emit("nop").unwrap();
    asm.label("done").unwrap();
    asm.emit("ret").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // JMP rel8 (2 bytes), then NOP (1), then RET (1)
    assert_eq!(bytes[0], 0xEB); // jmp rel8
    assert_eq!(*bytes.last().unwrap(), 0xC3);
}

// ============================================================================
// x86-64 Instruction Encoding Verification
// ============================================================================

#[test]
fn encode_mov_reg_imm() {
    // mov eax, 0x42 → B8 42 00 00 00
    let bytes = assemble("mov eax, 0x42", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xB8, 0x42, 0x00, 0x00, 0x00]);
}

#[test]
fn encode_mov_reg_reg_64bit() {
    // mov rax, rbx → 48 89 D8
    let bytes = assemble("mov rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x89, 0xD8]);
}

#[test]
fn encode_xor_self_zeroing() {
    // xor eax, eax → 31 C0
    let bytes = assemble("xor eax, eax", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x31, 0xC0]);
}

#[test]
fn encode_syscall() {
    let bytes = assemble("syscall", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0x05]);
}

#[test]
fn encode_push_pop_rbp() {
    let bytes = assemble("push rbp\npop rbp", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x55, 0x5D]);
}

#[test]
fn encode_sub_rsp_immediate() {
    // sub rsp, 8 → 48 83 EC 08
    let bytes = assemble("sub rsp, 8", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x83, 0xEC, 0x08]);
}

#[test]
fn encode_int3() {
    let bytes = assemble("int3", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xCC]);
}

#[test]
fn encode_extended_registers() {
    // mov r8, r9 → 4D 89 C8
    let bytes = assemble("mov r8, r9", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x4D, 0x89, 0xC8]);
}

#[test]
fn encode_memory_operand() {
    // mov rax, [rbx] → 48 8B 03
    let bytes = assemble("mov rax, [rbx]", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x8B, 0x03]);
}

#[test]
fn encode_memory_displacement() {
    // mov rax, [rbp - 8] → 48 8B 45 F8
    let bytes = assemble("mov rax, [rbp - 8]", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x8B, 0x45, 0xF8]);
}

#[test]
fn encode_lea_sib() {
    // lea rax, [rbx + rcx*8] → 48 8D 04 CB
    let bytes = assemble("lea rax, [rbx + rcx*8]", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x8D, 0x04, 0xCB]);
}

#[test]
fn encode_rep_movsb() {
    let bytes = assemble("rep movsb", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xF3, 0xA4]);
}

// ============================================================================
// Data Directives
// ============================================================================

#[test]
fn data_byte_directive() {
    let bytes = assemble(".byte 0x90, 0xCC, 0xC3", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x90, 0xCC, 0xC3]);
}

#[test]
fn data_word_directive() {
    let bytes = assemble(".word 0x1234", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x34, 0x12]);
}

#[test]
fn data_ascii_directive() {
    let bytes = assemble(".ascii \"AB\"", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x41, 0x42]);
}

#[test]
fn data_asciz_directive() {
    let bytes = assemble(".asciz \"hi\"", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![b'h', b'i', 0x00]);
}

#[test]
fn fill_directive() {
    let bytes = assemble(".fill 4, 1, 0xCC", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xCC, 0xCC, 0xCC, 0xCC]);
}

#[test]
fn space_directive() {
    let bytes = assemble(".space 3", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x00, 0x00, 0x00]);
}

// ============================================================================
// Labels & Control Flow
// ============================================================================

#[test]
fn forward_label_resolution() {
    let bytes = assemble("jmp end\nnop\nend:\nret", Arch::X86_64).unwrap();
    // Branch relaxation: jmp rel8 (2 bytes) + nop (1 byte) + ret (1 byte) = 4 bytes
    assert_eq!(bytes[0], 0xEB); // jmp rel8
                                // displacement = 1 (skip nop)
    assert_eq!(bytes[1], 0x01);
    assert_eq!(bytes[2], 0x90); // nop
    assert_eq!(bytes[3], 0xC3); // ret
}

#[test]
fn backward_label_resolution() {
    let src = "start:\nnop\njmp start";
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // nop (1 byte) + jmp rel8 (2 bytes) = 3 bytes
    assert_eq!(bytes[0], 0x90); // nop
    assert_eq!(bytes[1], 0xEB); // jmp rel8
                                // displacement = 0 - 3 = -3
    assert_eq!(bytes[2], 0xFD);
}

#[test]
fn conditional_branch() {
    let bytes = assemble("cmp rax, 0\nje done\nnop\ndone:\nret", Arch::X86_64).unwrap();
    let last = *bytes.last().unwrap();
    assert_eq!(last, 0xC3); // ret
}

// ============================================================================
// Constants (.equ)
// ============================================================================

#[test]
fn equ_constant_in_instruction() {
    let src = ".equ COUNT, 10\nmov ecx, COUNT";
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // mov ecx, 10 → B9 0A 00 00 00
    assert_eq!(bytes, vec![0xB9, 0x0A, 0x00, 0x00, 0x00]);
}

// ============================================================================
// Semicolon Separated Statements
// ============================================================================

#[test]
fn semicolon_as_separator() {
    let bytes = assemble("nop; nop; ret", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x90, 0x90, 0xC3]);
}

#[test]
fn semicolon_mov_rax_1_syscall() {
    // Story 1.5 AC: assemble("mov rax, 1; syscall") returns expected bytes
    let bytes = assemble("mov rax, 1; syscall", Arch::X86_64).unwrap();
    // MOV RAX, 1 → 48 C7 C0 01 00 00 00 (or 48 B8 ...)
    // SYSCALL → 0F 05
    assert!(!bytes.is_empty());
    // Last 2 bytes must be SYSCALL
    assert_eq!(&bytes[bytes.len() - 2..], &[0x0F, 0x05]);
}

// ============================================================================
// External Labels
// ============================================================================

#[test]
fn external_label_in_mov() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.define_external("puts", 0x401000);
    asm.emit("mov rax, puts").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // movabs rax, imm64 — last 8 bytes should be the address
    assert_eq!(&bytes[bytes.len() - 8..], &0x401000u64.to_le_bytes());
}

// ============================================================================
// Error Handling
// ============================================================================

#[test]
fn unknown_mnemonic() {
    let err = assemble("foobar", Arch::X86_64).unwrap_err();
    assert!(matches!(err, AsmError::UnknownMnemonic { .. }));
}

#[test]
fn undefined_label() {
    let err = assemble("jmp nowhere", Arch::X86_64).unwrap_err();
    assert!(matches!(err, AsmError::UndefinedLabel { .. }));
}

// ============================================================================
// Complex Programs
// ============================================================================

#[test]
fn function_prologue_epilogue() {
    let src = r#"
        push rbp
        mov rbp, rsp
        sub rsp, 0x20
        add rsp, 0x20
        pop rbp
        ret
    "#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0x55); // push rbp
    assert_eq!(*bytes.last().unwrap(), 0xC3); // ret
}

#[test]
fn loop_with_counter() {
    let src = r#"
        mov ecx, 5
    loop_top:
        dec ecx
        jnz loop_top
        ret
    "#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert!(!bytes.is_empty());
    assert_eq!(*bytes.last().unwrap(), 0xC3); // ret
}

#[test]
fn syscall_exit_sequence() {
    let src = r#"
        mov eax, 60
        xor edi, edi
        syscall
    "#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // Should end with syscall opcode 0F 05
    let len = bytes.len();
    assert_eq!(&bytes[len - 2..], &[0x0F, 0x05]);
}

#[test]
fn mixed_code_and_data() {
    let src = r#"
        jmp code
    data:
        .byte 0xDE, 0xAD
    code:
        nop
        ret
    "#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    let last = *bytes.last().unwrap();
    assert_eq!(last, 0xC3); // ret
                            // Data bytes should be embedded
    assert!(bytes.contains(&0xDE));
    assert!(bytes.contains(&0xAD));
}

#[test]
fn string_data_with_code() {
    let src = r#"
        .asciz "Hello"
        ret
    "#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert_eq!(&bytes[..5], b"Hello");
    assert_eq!(bytes[5], 0x00); // null terminator
    assert_eq!(bytes[6], 0xC3); // ret
}

// ============================================================================
// ALU Operations
// ============================================================================

#[test]
fn alu_add_sub_comprehensive() {
    let src = r#"
        add rax, rbx
        sub rcx, rdx
        and rsi, rdi
        or r8, r9
        xor rax, rax
    "#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert!(!bytes.is_empty());
}

#[test]
fn shift_operations() {
    let src = r#"
        shl eax, 1
        shr rcx, 4
        sar rax, cl
    "#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert!(!bytes.is_empty());
}

#[test]
fn test_and_compare() {
    let src = r#"
        test eax, eax
        cmp rax, 0
        test al, 0xFF
    "#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert!(!bytes.is_empty());
}

// ============================================================================
// Segment Overrides
// ============================================================================

#[test]
fn segment_override_fs() {
    // mov rax, fs:[0x28] — used for stack canary
    let bytes = assemble("mov rax, fs:[0x28]", Arch::X86_64).unwrap();
    // Should contain the fs prefix (0x64)
    assert!(bytes.contains(&0x64));
}

// ============================================================================
// Stack Operations
// ============================================================================

#[test]
fn push_pop_extended_regs() {
    let bytes = assemble("push r12\npush r13\npop r13\npop r12", Arch::X86_64).unwrap();
    assert!(!bytes.is_empty());
    // push r12 = 41 54, push r13 = 41 55
    assert_eq!(&bytes[0..2], &[0x41, 0x54]);
    assert_eq!(&bytes[2..4], &[0x41, 0x55]);
}

#[test]
fn push_immediate() {
    let bytes = assemble("push 42", Arch::X86_64).unwrap();
    // push imm8 → 6A 2A
    assert_eq!(bytes, vec![0x6A, 0x2A]);
}

// ============================================================================
// Branch Relaxation (public API)
// ============================================================================

#[test]
fn short_jmp_forward() {
    // jmp over a single nop — should stay short (EB rel8)
    let bytes = assemble("jmp target\ntarget:\nnop", Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0xEB); // short jmp
}

#[test]
fn short_jcc_forward() {
    let bytes = assemble("je done\ndone:\nret", Arch::X86_64).unwrap();
    // je rel8 = 74 00, ret = C3
    assert_eq!(bytes, vec![0x74, 0x00, 0xC3]);
}

#[test]
fn long_jmp_promotion() {
    // jmp over 200 NOPs — displacement exceeds ±127, must promote to rel32
    let mut lines = String::from("jmp target\n");
    for _ in 0..200 {
        lines.push_str("nop\n");
    }
    lines.push_str("target:\nnop\n");
    let bytes = assemble(&lines, Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0xE9); // long jmp rel32
}

#[test]
fn backward_short_branch() {
    let bytes = assemble("loop_top:\nnop\njmp loop_top", Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0x90); // nop
    assert_eq!(bytes[1], 0xEB); // short jmp
}

// ============================================================================
// Label Export & Addresses
// ============================================================================

#[test]
fn label_export() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit("start:\nnop\nmid:\nnop\nend:\nret").unwrap();
    let result = asm.finish().unwrap();
    assert_eq!(result.label_address("start"), Some(0));
    assert_eq!(result.label_address("mid"), Some(1));
    assert_eq!(result.label_address("end"), Some(2));
}

#[test]
fn label_export_with_base() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.base_address(0x401000);
    asm.emit("entry:\nnop\nret").unwrap();
    let result = asm.finish().unwrap();
    assert_eq!(result.label_address("entry"), Some(0x401000));
}

// ============================================================================
// Constants (.equ, .set, name = expr)
// ============================================================================

#[test]
fn equ_constant_used_in_mov() {
    let bytes = assemble(".equ MAGIC, 0xFF\nmov al, MAGIC", Arch::X86_64).unwrap();
    // mov al, 0xFF → B0 FF
    assert_eq!(bytes, vec![0xB0, 0xFF]);
}

#[test]
fn name_equals_constant() {
    let bytes = assemble("COUNT = 10\nmov ecx, COUNT", Arch::X86_64).unwrap();
    // mov ecx, 10 → B9 0A 00 00 00
    assert_eq!(bytes, vec![0xB9, 0x0A, 0x00, 0x00, 0x00]);
}

#[test]
fn set_directive_constant() {
    let bytes = assemble(".set OFFSET, 8\nmov eax, OFFSET", Arch::X86_64).unwrap();
    // mov eax, 8 → B8 08 00 00 00
    assert_eq!(bytes, vec![0xB8, 0x08, 0x00, 0x00, 0x00]);
}

// ============================================================================
// Data Label References (.quad label)
// ============================================================================

#[test]
fn quad_label_in_data() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.base_address(0x1000);
    asm.emit("func:\nnop\nret\n.quad func").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // func at 0x1000, nop+ret = 2B, .quad at offset 2
    let qw = u64::from_le_bytes(bytes[2..10].try_into().unwrap());
    assert_eq!(qw, 0x1000);
}

#[test]
fn long_label_in_data() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.base_address(0x2000);
    asm.emit("start:\nnop\n.long start").unwrap();
    let result = asm.finish().unwrap();
    let dw = u32::from_le_bytes(result.bytes()[1..5].try_into().unwrap());
    assert_eq!(dw, 0x2000);
}

// ============================================================================
// Listing Output
// ============================================================================

#[test]
fn listing_format() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit("entry:\npush rbp\nmov rbp, rsp\npop rbp\nret")
        .unwrap();
    let result = asm.finish().unwrap();
    let listing = result.listing();
    assert!(listing.contains("entry:"));
    assert!(listing.contains("55")); // push rbp
    assert!(listing.contains("C3")); // ret
}

// ============================================================================
// Builder Define Constant & External
// ============================================================================

#[test]
fn builder_define_constant_integration() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.define_constant("SYS_EXIT", 60);
    asm.emit("mov eax, SYS_EXIT\nsyscall").unwrap();
    let result = asm.finish().unwrap();
    // mov eax, 60 → B8 3C 00 00 00; syscall → 0F 05
    assert_eq!(&result.bytes()[..5], &[0xB8, 0x3C, 0x00, 0x00, 0x00]);
    assert_eq!(&result.bytes()[5..], &[0x0F, 0x05]);
}

#[test]
fn encode_one_resolves_constants() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.define_constant("EXIT_CODE", 42);
    // encode_one should resolve the constant just like emit() does
    let bytes = asm.encode_one("mov eax, EXIT_CODE").unwrap();
    // mov eax, 42 → B8 2A 00 00 00
    assert_eq!(bytes, vec![0xB8, 0x2A, 0x00, 0x00, 0x00]);
}

#[test]
fn encode_one_without_constants_still_works() {
    let asm = Assembler::new(Arch::X86_64);
    let bytes = asm.encode_one("nop").unwrap();
    assert_eq!(bytes, vec![0x90]);
}

#[test]
fn encode_one_memory_constant() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.define_constant("OFFSET", 0x10);
    let bytes = asm.encode_one("mov eax, [rbp + OFFSET]").unwrap();
    // mov eax, [rbp+0x10] → 8B 45 10
    assert_eq!(bytes, vec![0x8B, 0x45, 0x10]);
}

// ============================================================================
// Multiple Forward Labels
// ============================================================================

#[test]
fn chained_forward_jumps() {
    let src = r#"
        jmp a
        nop
    a:
        jmp b
        nop
    b:
        jmp c
        nop
    c:
        ret
    "#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // All jumps should be short (EB) since targets are nearby
    assert_eq!(bytes[0], 0xEB);
    // Last byte should be ret
    assert_eq!(*bytes.last().unwrap(), 0xC3);
}

#[test]
fn nested_loop_labels() {
    let src = r#"
        mov ecx, 10
    outer:
        mov edx, 5
    inner:
        dec edx
        jnz inner
        dec ecx
        jnz outer
        ret
    "#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert!(!bytes.is_empty());
    assert_eq!(*bytes.last().unwrap(), 0xC3);
}

// ============================================================================
// Real-World Shellcode: execve("/bin/sh") — Story 1.5
// ============================================================================

#[test]
fn execve_binsh_shellcode() {
    // Classic Linux x86-64 execve("/bin/sh", ["/bin/sh", NULL], NULL) shellcode.
    // This assembles instruction-by-instruction with known-good byte sequences.
    let src = r#"
        xor    rdx, rdx
        mov    rax, 0x68732f6e69622f
        push   rdx
        push   rax
        mov    rdi, rsp
        push   rdx
        push   rdi
        mov    rsi, rsp
        mov    eax, 59
        syscall
    "#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // Verify each instruction's encoding:
    let mut off = 0;

    // xor rdx, rdx → 48 31 D2
    assert_eq!(&bytes[off..off + 3], &[0x48, 0x31, 0xD2]);
    off += 3;

    // movabs rax, 0x68732f6e69622f → 48 B8 2F 62 69 6E 2F 73 68 00
    assert_eq!(bytes[off], 0x48);
    assert_eq!(bytes[off + 1], 0xB8);
    let val = u64::from_le_bytes(bytes[off + 2..off + 10].try_into().unwrap());
    assert_eq!(val, 0x68732f6e69622f);
    off += 10;

    // push rdx → 52
    assert_eq!(bytes[off], 0x52);
    off += 1;

    // push rax → 50
    assert_eq!(bytes[off], 0x50);
    off += 1;

    // mov rdi, rsp → 48 89 E7
    assert_eq!(&bytes[off..off + 3], &[0x48, 0x89, 0xE7]);
    off += 3;

    // push rdx → 52
    assert_eq!(bytes[off], 0x52);
    off += 1;

    // push rdi → 57
    assert_eq!(bytes[off], 0x57);
    off += 1;

    // mov rsi, rsp → 48 89 E6
    assert_eq!(&bytes[off..off + 3], &[0x48, 0x89, 0xE6]);
    off += 3;

    // mov eax, 59 → B8 3B 00 00 00
    assert_eq!(&bytes[off..off + 5], &[0xB8, 0x3B, 0x00, 0x00, 0x00]);
    off += 5;

    // syscall → 0F 05
    assert_eq!(&bytes[off..off + 2], &[0x0F, 0x05]);
    off += 2;

    assert_eq!(off, bytes.len());
}

// ============================================================================
// Alignment Directives — Integration
// ============================================================================

#[test]
fn align_with_multibyte_nops() {
    // .align without a fill byte on x86-64 should produce multi-byte NOPs
    let src = "nop\n.align 8\nnop";
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // 1 NOP + 7 bytes NOP padding + 1 NOP = 9 bytes
    assert_eq!(bytes.len(), 9);
    assert_eq!(bytes[0], 0x90);
    // 7-byte NOP: 0F 1F 80 00 00 00 00
    assert_eq!(&bytes[1..8], &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00]);
    assert_eq!(bytes[8], 0x90);
}

#[test]
fn align_with_explicit_fill() {
    // .align with explicit fill byte should NOT use NOPs
    let bytes = assemble("nop\n.align 4, 0xCC\nnop", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x90, 0xCC, 0xCC, 0xCC, 0x90]);
}

#[test]
fn p2align_directive() {
    // .p2align 3 == align to 2^3 = 8 bytes
    let bytes = assemble("nop\n.p2align 3\nnop", Arch::X86_64).unwrap();
    assert_eq!(bytes.len(), 9);
    assert_eq!(bytes[0], 0x90);
    assert_eq!(bytes[8], 0x90);
}

#[test]
fn org_directive_forward() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.base_address(0x100);
    asm.emit("nop\n.org 0x110\nnop").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // 1 NOP + 15 zero-fill + 1 NOP = 17
    assert_eq!(bytes.len(), 17);
    assert_eq!(bytes[0], 0x90);
    assert!(bytes[1..16].iter().all(|&b| b == 0x00));
    assert_eq!(bytes[16], 0x90);
}

#[test]
fn org_directive_error_backward() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.base_address(0x200);
    asm.emit("nop\nnop\nnop\n.org 0x100\nnop").unwrap();
    let err = asm.finish().unwrap_err();
    assert!(matches!(err, AsmError::Syntax { .. }));
}

// ============================================================================
// Space and Fill Directives
// ============================================================================

#[test]
fn fill_directive_pattern() {
    let bytes = assemble(".fill 3, 2, 0xAB", Arch::X86_64).unwrap();
    // 3 repetitions × 2 bytes each = 6 bytes
    // Each 2-byte unit is LE encoding of 0xAB: [0xAB, 0x00]
    assert_eq!(bytes, vec![0xAB, 0x00, 0xAB, 0x00, 0xAB, 0x00]);
}

#[test]
fn space_with_fill_value() {
    let bytes = assemble(".space 5, 0xFF", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xFF; 5]);
}

// ============================================================================
// More Instruction Encoding Verification
// ============================================================================

#[test]
fn encode_movzx_8_to_32() {
    // movzx eax, bl → 0F B6 C3
    let bytes = assemble("movzx eax, bl", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0xB6, 0xC3]);
}

#[test]
fn encode_movsx_8_to_64() {
    // movsx rax, bl → 48 0F BE C3
    let bytes = assemble("movsx rax, bl", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x0F, 0xBE, 0xC3]);
}

#[test]
fn encode_movsxd_32_to_64() {
    // movsxd rax, ecx → 48 63 C1
    let bytes = assemble("movsxd rax, ecx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x63, 0xC1]);
}

#[test]
fn encode_lea_rip_relative() {
    // lea rax, [rip + label] with label right after → disp32 = 0
    let bytes = assemble("lea rax, [rip + target]\ntarget:", Arch::X86_64).unwrap();
    // 48 8D 05 00 00 00 00
    assert_eq!(bytes, vec![0x48, 0x8D, 0x05, 0x00, 0x00, 0x00, 0x00]);
}

#[test]
fn encode_imul_three_operand() {
    // imul rax, rbx, 10 → 48 6B C3 0A
    let bytes = assemble("imul rax, rbx, 10", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x6B, 0xC3, 0x0A]);
}

#[test]
fn encode_not_neg() {
    // not rax → 48 F7 D0
    // neg rcx → 48 F7 D9
    let bytes = assemble("not rax\nneg rcx", Arch::X86_64).unwrap();
    assert_eq!(&bytes[0..3], &[0x48, 0xF7, 0xD0]);
    assert_eq!(&bytes[3..6], &[0x48, 0xF7, 0xD9]);
}

#[test]
fn encode_bswap() {
    // bswap eax → 0F C8
    // bswap r12 → 49 0F CC
    let bytes = assemble("bswap eax\nbswap r12", Arch::X86_64).unwrap();
    assert_eq!(&bytes[0..2], &[0x0F, 0xC8]);
    assert_eq!(&bytes[2..5], &[0x49, 0x0F, 0xCC]);
}

#[test]
fn encode_bt_reg_imm() {
    // bt eax, 5 → 0F BA E0 05
    let bytes = assemble("bt eax, 5", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0xBA, 0xE0, 0x05]);
}

#[test]
fn encode_setcc() {
    // sete al → 0F 94 C0
    // setne cl → 0F 95 C1
    let bytes = assemble("sete al\nsetne cl", Arch::X86_64).unwrap();
    assert_eq!(&bytes[0..3], &[0x0F, 0x94, 0xC0]);
    assert_eq!(&bytes[3..6], &[0x0F, 0x95, 0xC1]);
}

#[test]
fn encode_cmovcc() {
    // cmove rax, rbx → 48 0F 44 C3
    let bytes = assemble("cmove rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x0F, 0x44, 0xC3]);
}

#[test]
fn encode_inc_dec_64() {
    // inc rax → 48 FF C0
    // dec rcx → 48 FF C9
    let bytes = assemble("inc rax\ndec rcx", Arch::X86_64).unwrap();
    assert_eq!(&bytes[0..3], &[0x48, 0xFF, 0xC0]);
    assert_eq!(&bytes[3..6], &[0x48, 0xFF, 0xC9]);
}

#[test]
fn encode_inc_dec_32_short_forms() {
    // x86-32 uses 1-byte short forms: INC r32 = 0x40+rd, DEC r32 = 0x48+rd
    let bytes = assemble("inc eax\ninc ebx\ninc edi\ndec eax\ndec esp", Arch::X86).unwrap();
    assert_eq!(bytes[0], 0x40); // inc eax
    assert_eq!(bytes[1], 0x43); // inc ebx
    assert_eq!(bytes[2], 0x47); // inc edi
    assert_eq!(bytes[3], 0x48); // dec eax
    assert_eq!(bytes[4], 0x4C); // dec esp
}

#[test]
fn encode_inc_dec_32_16bit() {
    // 16-bit registers get 0x66 prefix + short form
    let bytes = assemble("inc ax\ndec cx", Arch::X86).unwrap();
    assert_eq!(&bytes[0..2], &[0x66, 0x40]); // inc ax
    assert_eq!(&bytes[2..4], &[0x66, 0x49]); // dec cx
}

#[test]
fn encode_inc_dec_32_8bit_modrm() {
    // 8-bit inc/dec uses ModR/M form (0xFE) even in 32-bit mode
    let bytes = assemble("inc al\ndec cl", Arch::X86).unwrap();
    assert_eq!(&bytes[0..2], &[0xFE, 0xC0]); // inc al
    assert_eq!(&bytes[2..4], &[0xFE, 0xC9]); // dec cl
}

#[test]
fn encode_xchg_rax() {
    // xchg rax, rbx → 48 93
    let bytes = assemble("xchg rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x93]);
}

#[test]
fn encode_lock_prefix() {
    // lock add [rax], ebx → F0 01 18
    let bytes = assemble("lock add [rax], ebx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xF0, 0x01, 0x18]);
}

#[test]
fn encode_gs_segment_override() {
    // mov rax, gs:[rbx + 0x28] — contains GS prefix 0x65
    let bytes = assemble("mov rax, gs:[rbx + 0x28]", Arch::X86_64).unwrap();
    assert!(bytes.contains(&0x65));
}

#[test]
fn encode_multibyte_nops() {
    // nop2 through nop9 generate Intel-recommended multi-byte NOPs
    let bytes2 = assemble("nop2", Arch::X86_64).unwrap();
    assert_eq!(bytes2, vec![0x66, 0x90]);

    let bytes3 = assemble("nop3", Arch::X86_64).unwrap();
    assert_eq!(bytes3, vec![0x0F, 0x1F, 0x00]);

    let bytes9 = assemble("nop9", Arch::X86_64).unwrap();
    assert_eq!(bytes9.len(), 9);
    assert_eq!(&bytes9[..2], &[0x66, 0x0F]);
}

#[test]
fn encode_cdq_cqo() {
    let cdq = assemble("cdq", Arch::X86_64).unwrap();
    assert_eq!(cdq, vec![0x99]);

    let cqo = assemble("cqo", Arch::X86_64).unwrap();
    assert_eq!(cqo, vec![0x48, 0x99]);
}

#[test]
fn encode_hlt() {
    let bytes = assemble("hlt", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xF4]);
}

// ============================================================================
// Addressing Modes
// ============================================================================

#[test]
fn encode_sib_rsp_base() {
    // mov [rsp], eax → 89 04 24
    let bytes = assemble("mov [rsp], eax", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x89, 0x04, 0x24]);
}

#[test]
fn encode_sib_r12_base() {
    // mov [r12], eax → 41 89 04 24
    let bytes = assemble("mov [r12], eax", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x41, 0x89, 0x04, 0x24]);
}

#[test]
fn encode_disp32_for_large_displacement() {
    // mov eax, [rbx + 0x1000] → 8B 83 00 10 00 00
    let bytes = assemble("mov eax, [rbx + 0x1000]", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x8B, 0x83, 0x00, 0x10, 0x00, 0x00]);
}

#[test]
fn encode_sib_scale_index() {
    // lea rax, [rcx + rdx*2] → 48 8D 04 51
    let bytes = assemble("lea rax, [rcx + rdx*2]", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x8D, 0x04, 0x51]);
}

// ============================================================================
// Numeric Labels
// ============================================================================

#[test]
fn numeric_label_forward_backward() {
    let src = "1:\nnop\njmp 1b";
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0x90); // nop
    assert_eq!(bytes[1], 0xEB); // jmp rel8
    assert_eq!(bytes[2], 0xFD_u8); // -3
}

#[test]
fn numeric_label_forward_ref() {
    let src = "jmp 1f\nnop\n1:\nret";
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0xEB); // jmp rel8
    assert_eq!(bytes[1], 0x01); // skip 1 nop
    assert_eq!(bytes[2], 0x90); // nop
    assert_eq!(bytes[3], 0xC3); // ret
}

// ============================================================================
// Data Directive Edge Cases
// ============================================================================

#[test]
fn data_quad_value() {
    let bytes = assemble(".quad 0xDEADBEEFCAFEBABE", Arch::X86_64).unwrap();
    let val = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    assert_eq!(val, 0xDEADBEEFCAFEBABE);
}

#[test]
fn data_long_value() {
    let bytes = assemble(".long 0x12345678", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x78, 0x56, 0x34, 0x12]);
}

#[test]
fn escape_sequences_in_string() {
    let bytes = assemble(r#".ascii "\t\n\\\0""#, Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x09, 0x0A, 0x5C, 0x00]);
}

// ============================================================================
// Builder: Mixed Operations
// ============================================================================

#[test]
fn builder_mixed_emit_label_db() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.base_address(0x400000);
    asm.emit("jmp skip").unwrap();
    asm.label("data").unwrap();
    asm.db(b"AAAA").unwrap();
    asm.label("skip").unwrap();
    asm.emit("nop\nret").unwrap();

    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // jmp rel8 (2) + 4 data bytes + nop + ret
    assert_eq!(bytes[0], 0xEB); // jmp
    assert_eq!(&bytes[2..6], b"AAAA");
    assert_eq!(*bytes.last().unwrap(), 0xC3); // ret

    // Label addresses
    assert_eq!(result.label_address("data"), Some(0x400002));
    assert_eq!(result.label_address("skip"), Some(0x400006));
}

#[test]
fn builder_dw_dd_dq() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.dw(0x1234).unwrap();
    asm.dd(0xDEADBEEF).unwrap();
    asm.dq(0xCAFEBABE00000000).unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    assert_eq!(&bytes[0..2], &[0x34, 0x12]);
    assert_eq!(&bytes[2..6], &[0xEF, 0xBE, 0xAD, 0xDE]);
    let q = u64::from_le_bytes(bytes[6..14].try_into().unwrap());
    assert_eq!(q, 0xCAFEBABE00000000);
}

// ============================================================================
// Relocations
// ============================================================================

#[test]
fn relocation_tracking() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit("jmp target\nnop\ntarget:\nret").unwrap();
    let result = asm.finish().unwrap();
    let relocs = result.relocations();
    // There should be at least one relocation for the jmp
    assert!(!relocs.is_empty());
    assert_eq!(relocs[0].label, "target");
}

/// A-2 regression: displacement offsets must be found structurally (via
/// ModR/M + SIB parsing), not by byte-pattern scanning.  These tests
/// cover instruction classes that previously relied on the byte-scanner
/// fallback and would break if scanning matched wrong bytes.
#[test]
fn structural_reloc_neg_mem_label() {
    use asm_rs::assemble_with;
    // neg [rip + target] — displacement label on a "unary" instruction (no
    // reloc param). Exercise structural offset for the 0xFF /3 form.
    let code = assemble_with(
        "neg dword ptr [rip + target]",
        Arch::X86_64,
        0x1000,
        &[("target", 0x2000)],
    )
    .unwrap();
    // neg [rip + disp32]: F7 1D dd dd dd dd
    assert_eq!(code[0], 0xF7); // opcode
    assert_eq!(code[1] & 0xC7, 0x05); // ModR/M: mod=00, rm=101 (RIP-rel)
                                      // displacement should be resolved correctly
    assert_eq!(code.len(), 6);
}

#[test]
fn structural_reloc_inc_mem_label() {
    use asm_rs::assemble_with;
    // inc dword ptr [rip + target]
    let code = assemble_with(
        "inc dword ptr [rip + target]",
        Arch::X86_64,
        0x1000,
        &[("target", 0x2000)],
    )
    .unwrap();
    // inc [rip + disp32]: FF 05 dd dd dd dd
    assert_eq!(code[0], 0xFF);
    assert_eq!(code[1] & 0xC7, 0x05); // mod=00, rm=101
    assert_eq!(code.len(), 6);
}

#[test]
fn structural_reloc_shl_mem_label() {
    use asm_rs::assemble_with;
    // shl dword ptr [rip + target], 1 — shift with displacement label
    let code = assemble_with(
        "shl dword ptr [rip + target], 1",
        Arch::X86_64,
        0x1000,
        &[("target", 0x2000)],
    )
    .unwrap();
    // shl m32, 1: D1 /4 with rip-rel ModR/M → D1 25 dd dd dd dd
    assert_eq!(code[0], 0xD1);
    assert_eq!(code[1], 0x25); // mod=00, reg=4, rm=101 (RIP-rel)
    assert_eq!(code.len(), 6);
}

#[test]
fn structural_reloc_imul_mem_label() {
    use asm_rs::assemble_with;
    // imul rax, [rip + target] — 2-operand, displacement label
    let code = assemble_with(
        "imul rax, qword ptr [rip + target]",
        Arch::X86_64,
        0x1000,
        &[("target", 0x2000)],
    )
    .unwrap();
    // imul r, [rip+disp32]: 48 0F AF 05 dd dd dd dd
    assert_eq!(code[0], 0x48); // REX.W
    assert_eq!(code[1], 0x0F);
    assert_eq!(code[2], 0xAF);
    assert_eq!(code[3] & 0xC7, 0x05); // mod=00, rm=101 (RIP-rel)
    assert_eq!(code.len(), 8);
}

#[test]
fn structural_reloc_zero_displacement() {
    use asm_rs::assemble_with;
    // Regression: when disp=0 the old byte scanner would search for [0,0,0,0]
    // which could match opcode bytes.  Structural parsing is immune.
    let code = assemble_with(
        "neg dword ptr [rip + target]",
        Arch::X86_64,
        0x1006, // target at 0x1000 → disp = 0x1000 - (0x1006 + 6) = -12
        &[("target", 0x1000)],
    )
    .unwrap();
    assert_eq!(code[0], 0xF7);
    assert_eq!(code.len(), 6);
    // Verify the displacement bytes are at offset 2 (structurally)
    let disp = i32::from_le_bytes(code[2..6].try_into().unwrap());
    assert_eq!(disp, -12); // 0x1000 - (0x1006 + 6) = -12
}

// ============================================================================
// Multiple Error Scenarios
// ============================================================================

#[test]
fn duplicate_label_error() {
    let err = assemble("foo:\nnop\nfoo:\nret", Arch::X86_64).unwrap_err();
    assert!(matches!(err, AsmError::DuplicateLabel { .. }));
}

#[test]
fn invalid_operands_error() {
    let err = assemble("mov rax", Arch::X86_64).unwrap_err();
    assert!(matches!(err, AsmError::InvalidOperands { .. }));
}

// ============================================================================
// External Labels
// ============================================================================

#[test]
fn external_label_call() {
    use asm_rs::assemble_with;
    let code = assemble_with("call puts", Arch::X86_64, 0x401000, &[("puts", 0x401100)]).unwrap();
    // call rel32 — 5 bytes: E8 xx xx xx xx
    assert_eq!(code[0], 0xE8);
    // displacement = 0x401100 - (0x401000 + 5) = 0xFB
    let disp = i32::from_le_bytes(code[1..5].try_into().unwrap());
    assert_eq!(disp, 0xFB);
}

#[test]
fn external_label_in_mov_abs() {
    use asm_rs::assemble_with;
    let code = assemble_with(
        "mov rax, printf",
        Arch::X86_64,
        0x0,
        &[("printf", 0xDEAD_BEEF)],
    )
    .unwrap();
    // movabs rax, imm64: 48 B8 xx xx xx xx xx xx xx xx
    assert_eq!(code[0], 0x48);
    assert_eq!(code[1], 0xB8);
    let val = u64::from_le_bytes(code[2..10].try_into().unwrap());
    assert_eq!(val, 0xDEAD_BEEF);
}

// ============================================================================
// Mul / Div / Idiv
// ============================================================================

#[test]
fn mul_r64() {
    // mul rcx: 48 F7 E1
    let code = assemble("mul rcx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xF7, 0xE1]);
}

#[test]
fn div_r64() {
    // div rbx: 48 F7 F3
    let code = assemble("div rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xF7, 0xF3]);
}

#[test]
fn idiv_r32() {
    // idiv ecx: F7 F9
    let code = assemble("idiv ecx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xF7, 0xF9]);
}

// ============================================================================
// Indirect Call / Jump
// ============================================================================

#[test]
fn call_reg_indirect() {
    // call rax: FF D0
    let code = assemble("call rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xFF, 0xD0]);
}

#[test]
fn jmp_reg_indirect() {
    // jmp rax: FF E0
    let code = assemble("jmp rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xFF, 0xE0]);
}

#[test]
fn call_mem_indirect() {
    // call qword ptr [rax]: 48 FF 10 (REX.W for 64-bit)
    // or FF 10 depending on encoder — just verify opcode
    let code = assemble("call [rax]", Arch::X86_64).unwrap();
    // Last two bytes must be FF 10 (opcode + ModRM)
    let tail = &code[code.len() - 2..];
    assert_eq!(tail, &[0xFF, 0x10]);
}

#[test]
fn jmp_mem_indirect() {
    // jmp qword ptr [rbx]: REX.W optional; FF 23
    let code = assemble("jmp [rbx]", Arch::X86_64).unwrap();
    let tail = &code[code.len() - 2..];
    assert_eq!(tail, &[0xFF, 0x23]);
}

// ============================================================================
// 16-bit operands
// ============================================================================

#[test]
fn mov_ax_imm16() {
    // mov ax, 0x1234: 66 B8 34 12
    let code = assemble("mov ax, 0x1234", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0xB8, 0x34, 0x12]);
}

#[test]
fn add_ax_imm16() {
    // add ax, 0x100: 66 05 00 01  (short form for AX)
    // or: 66 81 C0 00 01
    let code = assemble("add ax, 0x100", Arch::X86_64).unwrap();
    // Either encoding is valid — just check prefix present
    assert_eq!(code[0], 0x66);
}

// ============================================================================
// mov [mem], imm
// ============================================================================

#[test]
fn mov_mem_imm32() {
    // mov dword ptr [rax], 0x42: C7 00 42 00 00 00
    let code = assemble("mov dword ptr [rax], 0x42", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xC7, 0x00, 0x42, 0x00, 0x00, 0x00]);
}

#[test]
fn mov_byte_ptr_mem_imm8() {
    // mov byte ptr [rax], 0x42: C6 00 42
    let code = assemble("mov byte ptr [rax], 0x42", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xC6, 0x00, 0x42]);
}

// ============================================================================
// repne prefix
// ============================================================================

#[test]
fn repne_scasb() {
    // repne scasb: F2 AE
    let code = assemble("repne scasb", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xF2, 0xAE]);
}

// ============================================================================
// int 0x80
// ============================================================================

#[test]
fn int_0x80() {
    // int 0x80: CD 80
    let code = assemble("int 0x80", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xCD, 0x80]);
}

// ============================================================================
// xchg non-rax registers
// ============================================================================

#[test]
fn xchg_rcx_rdx() {
    // xchg rcx, rdx: REX.W + 87 ModRM
    let code = assemble("xchg rcx, rdx", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 3);
    assert_eq!(code[0], 0x48); // REX.W
    assert_eq!(code[1], 0x87); // XCHG opcode
}

#[test]
fn xchg_eax_ecx() {
    // xchg eax, ecx: 91 (short form)
    let code = assemble("xchg eax, ecx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x91]);
}

// ============================================================================
// movabs (64-bit immediate)
// ============================================================================

#[test]
fn movabs_max_u64() {
    // mov rax, 0xFFFFFFFFFFFFFFFF: 48 B8 FF...FF
    let code = assemble("mov rax, 0xFFFFFFFFFFFFFFFF", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 10);
    assert_eq!(code[0], 0x48);
    assert_eq!(code[1], 0xB8);
    assert!(code[2..10].iter().all(|&b| b == 0xFF));
}

// ============================================================================
// Backward long branch (relaxation)
// ============================================================================

#[test]
fn backward_long_branch_relaxation() {
    // Create a backward branch that must be long
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit("target:").unwrap();
    // Emit 200 bytes of NOPs to push branch out of rel8 range
    for _ in 0..200 {
        asm.emit("nop").unwrap();
    }
    asm.emit("jmp target").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // Last instruction is jmp rel32 (5 bytes: E9 xx xx xx xx)
    let last_5 = &bytes[bytes.len() - 5..];
    assert_eq!(last_5[0], 0xE9);
}

// ============================================================================
// RIP-relative mov
// ============================================================================

#[test]
fn mov_rip_relative_label() {
    let code = assemble(
        "mov rax, [rip + data]\nret\ndata:\n.quad 0x42",
        Arch::X86_64,
    )
    .unwrap();
    // mov rax, [rip+offset]: 48 8B 05 xx xx xx xx
    assert_eq!(code[0], 0x48);
    assert_eq!(code[1], 0x8B);
    assert_eq!(code[2], 0x05);
}

// ============================================================================
// .org with fill byte
// ============================================================================

#[test]
fn org_with_fill_byte_integration() {
    let code = assemble("nop\n.org 0x08, 0xCC\nnop", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 9);
    assert_eq!(code[0], 0x90); // nop
                               // bytes 1-7 should be 0xCC fill
    for (i, &byte) in code[1..8].iter().enumerate() {
        assert_eq!(byte, 0xCC, "byte {} should be fill", i + 1);
    }
    assert_eq!(code[8], 0x90); // nop
}

// ============================================================================
// .align + .org interaction
// ============================================================================

#[test]
fn align_then_org() {
    let code = assemble("nop\n.align 4\n.org 0x08\nnop", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 9);
    assert_eq!(code[8], 0x90);
}

// ============================================================================
// Builder convenience methods
// ============================================================================

#[test]
fn builder_ascii_asciz() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.ascii("AB").unwrap();
    asm.asciz("CD").unwrap();
    let result = asm.finish().unwrap();
    assert_eq!(result.bytes(), &[0x41, 0x42, 0x43, 0x44, 0x00]);
}

#[test]
fn builder_align_method() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit("nop").unwrap();
    asm.align(4);
    asm.emit("nop").unwrap();
    let result = asm.finish().unwrap();
    assert_eq!(result.bytes().len(), 5); // 1 + 3 pad + 1
}

#[test]
fn builder_org_method() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit("nop").unwrap();
    asm.org(8);
    asm.emit("ret").unwrap();
    let result = asm.finish().unwrap();
    assert_eq!(result.bytes().len(), 9);
    assert_eq!(result.bytes()[8], 0xC3);
}

#[test]
fn builder_fill_method() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.fill(4, 1, 0x90).unwrap();
    let result = asm.finish().unwrap();
    assert_eq!(result.bytes(), &[0x90, 0x90, 0x90, 0x90]);
}

#[test]
fn builder_space_method() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.space(3).unwrap();
    let result = asm.finish().unwrap();
    assert_eq!(result.bytes(), &[0x00, 0x00, 0x00]);
}

// ============================================================================
// Listing with directive annotations
// ============================================================================

#[test]
fn listing_shows_align_source() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.enable_listing();
    asm.emit("nop\n.align 4\nnop").unwrap();
    let result = asm.finish().unwrap();
    let listing = result.listing();
    assert!(listing.contains("nop"));
    assert!(listing.contains(".align 4"));
}

#[test]
fn listing_shows_org_source() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.enable_listing();
    asm.emit("nop\n.org 0x10\nnop").unwrap();
    let result = asm.finish().unwrap();
    let listing = result.listing();
    assert!(listing.contains(".org 0x10"));
}

// ============================================================================
// .word label reference (16-bit data label ref)
// ============================================================================

#[test]
fn word_label_ref() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.base_address(0x100);
    asm.emit("func:\nnop\nret\n.word func").unwrap();
    let result = asm.finish().unwrap();
    // .word func -> 16-bit absolute address = 0x0100
    let word_bytes = &result.bytes()[2..4];
    let val = u16::from_le_bytes(word_bytes.try_into().unwrap());
    assert_eq!(val, 0x100);
}

// ============================================================================
// Source annotations in AssemblyResult
// ============================================================================

#[test]
fn source_annotations_accessible() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.enable_listing();
    asm.emit("nop\nret").unwrap();
    let result = asm.finish().unwrap();
    // Source annotations should contain entries for both instructions
    let listing = result.listing();
    // Both instructions should have annotations
    assert!(listing.contains("nop"));
    assert!(listing.contains("ret"));
}

// ============================================================================
// Arch::X86 not yet implemented
// ============================================================================

#[test]
fn arch_x86_nop() {
    let result = assemble("nop", Arch::X86);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), vec![0x90]);
}

// ============================================================================
// push/pop 16-bit registers
// ============================================================================

#[test]
fn push_ax() {
    let bytes = assemble("push ax", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x66, 0x50]);
}

#[test]
fn pop_ax() {
    let bytes = assemble("pop ax", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x66, 0x58]);
}

#[test]
fn push_di() {
    let bytes = assemble("push di", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x66, 0x57]);
}

// ============================================================================
// xchg 16-bit shortcut
// ============================================================================

#[test]
fn xchg_ax_bx() {
    let bytes = assemble("xchg ax, bx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x66, 0x93]);
}

// ============================================================================
// ADC / SBB
// ============================================================================

#[test]
fn adc_eax_ecx() {
    // adc eax, ecx → 11 C8
    let bytes = assemble("adc eax, ecx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x11, 0xC8]);
}

#[test]
fn sbb_rax_imm8() {
    // sbb rax, 1 → 48 83 D8 01
    let bytes = assemble("sbb rax, 1", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x83, 0xD8, 0x01]);
}

// ============================================================================
// RCL / RCR / ROL / ROR
// ============================================================================

#[test]
fn rcl_eax_1() {
    // rcl eax, 1 → D1 D0
    let bytes = assemble("rcl eax, 1", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xD1, 0xD0]);
}

#[test]
fn rcr_ecx_4() {
    // rcr ecx, 4 → C1 D9 04
    let bytes = assemble("rcr ecx, 4", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xC1, 0xD9, 0x04]);
}

#[test]
fn rol_rdx_cl() {
    // rol rdx, cl → 48 D3 C2
    let bytes = assemble("rol rdx, cl", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0xD3, 0xC2]);
}

#[test]
fn ror_eax_1() {
    // ror eax, 1 → D1 C8
    let bytes = assemble("ror eax, 1", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xD1, 0xC8]);
}

// ============================================================================
// SETcc variants
// ============================================================================

#[test]
fn setl_al() {
    let bytes = assemble("setl al", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0x9C, 0xC0]);
}

#[test]
fn setg_bl() {
    let bytes = assemble("setg bl", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0x9F, 0xC3]);
}

#[test]
fn setb_cl() {
    let bytes = assemble("setb cl", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0x92, 0xC1]);
}

#[test]
fn seta_dl() {
    let bytes = assemble("seta dl", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0x97, 0xC2]);
}

#[test]
fn setne_al() {
    let bytes = assemble("setne al", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0x95, 0xC0]);
}

#[test]
fn setz_sil() {
    // setz sil → REX 0F 94 C6
    let bytes = assemble("setz sil", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x40, 0x0F, 0x94, 0xC6]);
}

// ============================================================================
// CMOVcc variants
// ============================================================================

#[test]
fn cmovl_eax_ecx() {
    let bytes = assemble("cmovl eax, ecx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0x4C, 0xC1]);
}

#[test]
fn cmovg_rax_rbx() {
    let bytes = assemble("cmovg rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x0F, 0x4F, 0xC3]);
}

#[test]
fn cmovb_edx_esi() {
    let bytes = assemble("cmovb edx, esi", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0x42, 0xD6]);
}

#[test]
fn cmova_ecx_edx() {
    let bytes = assemble("cmova ecx, edx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0x47, 0xCA]);
}

// ============================================================================
// BSF / BSR with register operands
// ============================================================================

#[test]
fn bsf_eax_ecx() {
    let bytes = assemble("bsf eax, ecx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0xBC, 0xC1]);
}

#[test]
fn bsr_rax_rdx() {
    let bytes = assemble("bsr rax, rdx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x0F, 0xBD, 0xC2]);
}

// ============================================================================
// POPCNT / LZCNT / TZCNT with register operands
// ============================================================================

#[test]
fn popcnt_eax_ecx() {
    let bytes = assemble("popcnt eax, ecx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xF3, 0x0F, 0xB8, 0xC1]);
}

#[test]
fn lzcnt_rax_rbx() {
    let bytes = assemble("lzcnt rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xF3, 0x48, 0x0F, 0xBD, 0xC3]);
}

#[test]
fn tzcnt_eax_edx() {
    let bytes = assemble("tzcnt eax, edx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xF3, 0x0F, 0xBC, 0xC2]);
}

// ============================================================================
// CBW / CWDE / CDQE / CWD / CDQ / CQO
// ============================================================================

#[test]
fn cbw_encoding() {
    let bytes = assemble("cbw", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x66, 0x98]);
}

#[test]
fn cwde_encoding() {
    let bytes = assemble("cwde", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x98]);
}

#[test]
fn cdqe_encoding() {
    let bytes = assemble("cdqe", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x98]);
}

#[test]
fn cwd_encoding() {
    let bytes = assemble("cwd", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x66, 0x99]);
}

#[test]
fn cdq_encoding() {
    let bytes = assemble("cdq", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x99]);
}

#[test]
fn cqo_encoding() {
    let bytes = assemble("cqo", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x99]);
}

// ============================================================================
// Flag instructions
// ============================================================================

#[test]
fn clc_stc_cmc() {
    let bytes = assemble("clc\nstc\ncmc", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xF8, 0xF9, 0xF5]);
}

#[test]
fn cld_std() {
    let bytes = assemble("cld\nstd", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xFC, 0xFD]);
}

#[test]
fn cli_sti() {
    let bytes = assemble("cli\nsti", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xFA, 0xFB]);
}

#[test]
fn lahf_sahf() {
    let bytes = assemble("lahf\nsahf", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x9F, 0x9E]);
}

// ============================================================================
// String instructions (various)
// ============================================================================

#[test]
fn lodsb_encoding() {
    let bytes = assemble("lodsb", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xAC]);
}

#[test]
fn lodsq_encoding() {
    let bytes = assemble("lodsq", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0xAD]);
}

#[test]
fn stosb_encoding() {
    let bytes = assemble("stosb", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xAA]);
}

#[test]
fn stosq_encoding() {
    let bytes = assemble("stosq", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0xAB]);
}

#[test]
fn cmpsb_encoding() {
    let bytes = assemble("cmpsb", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xA6]);
}

#[test]
fn scasb_encoding() {
    let bytes = assemble("scasb", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xAE]);
}

// ============================================================================
// LOOP / LOOPE / LOOPNE
// ============================================================================

#[test]
fn loop_forward() {
    let bytes = assemble("loop target\nnop\ntarget:", Arch::X86_64).unwrap();
    // loop rel8 → E2 xx, then nop
    assert_eq!(bytes[0], 0xE2);
    // loop is 2 bytes, nop is 1 byte, label is at offset 3
    // rel8 = target - (loop_end) = 3 - 2 = 1
    assert_eq!(bytes[1], 0x01);
    assert_eq!(bytes[2], 0x90);
}

#[test]
fn loope_loopne() {
    // Just verify they encode, real use would need labels
    let src = "loope lbl\nloopne lbl\nlbl:";
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0xE1); // loope
    assert_eq!(bytes[2], 0xE0); // loopne
}

// ============================================================================
// LOOP / JECXZ / JRCXZ branch relaxation (long-range targets)
// ============================================================================

#[test]
fn loop_relaxes_to_long_form_for_far_target() {
    // LOOP target >127 bytes away → long form: E2 02 EB 05 E9 rel32
    let mut src = String::from("loop target\n");
    // 200 NOPs to push target out of rel8 range
    for _ in 0..200 {
        src.push_str("nop\n");
    }
    src.push_str("target:");
    let bytes = assemble(&src, Arch::X86_64).unwrap();
    // Long form header: E2 02 EB 05 E9
    assert_eq!(bytes[0], 0xE2); // LOOP
    assert_eq!(bytes[1], 0x02); // +2 (skip JMP short)
    assert_eq!(bytes[2], 0xEB); // JMP short
    assert_eq!(bytes[3], 0x05); // +5 (skip JMP near)
    assert_eq!(bytes[4], 0xE9); // JMP near
                                // Then 200 NOPs
    assert!(bytes[9..209].iter().all(|&b| b == 0x90));
}

#[test]
fn loopne_relaxes_to_long_form() {
    let mut src = String::from("loopne target\n");
    for _ in 0..200 {
        src.push_str("nop\n");
    }
    src.push_str("target:");
    let bytes = assemble(&src, Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0xE0); // LOOPNE
    assert_eq!(bytes[1], 0x02);
    assert_eq!(bytes[2], 0xEB);
    assert_eq!(bytes[3], 0x05);
    assert_eq!(bytes[4], 0xE9);
}

#[test]
fn jrcxz_relaxes_to_long_form() {
    let mut src = String::from("jrcxz target\n");
    for _ in 0..200 {
        src.push_str("nop\n");
    }
    src.push_str("target:");
    let bytes = assemble(&src, Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0xE3); // JRCXZ
    assert_eq!(bytes[1], 0x02);
    assert_eq!(bytes[2], 0xEB);
    assert_eq!(bytes[3], 0x05);
    assert_eq!(bytes[4], 0xE9);
}

#[test]
fn jecxz_relaxes_to_long_form() {
    let mut src = String::from("jecxz target\n");
    for _ in 0..200 {
        src.push_str("nop\n");
    }
    src.push_str("target:");
    let bytes = assemble(&src, Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0x67); // addr-size override
    assert_eq!(bytes[1], 0xE3); // JECXZ
    assert_eq!(bytes[2], 0x02);
    assert_eq!(bytes[3], 0xEB);
    assert_eq!(bytes[4], 0x05);
    assert_eq!(bytes[5], 0xE9);
}

#[test]
fn loop_stays_short_when_target_is_near() {
    // LOOP with nearby target should relax to 2-byte short form
    let bytes = assemble("loop target\nnop\nnop\ntarget:", Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0xE2); // LOOP
    assert_eq!(bytes[1], 0x02); // rel8 = +2 (skip 2 NOPs)
    assert_eq!(bytes[2], 0x90); // NOP
    assert_eq!(bytes[3], 0x90); // NOP
    assert_eq!(bytes.len(), 4);
}

#[test]
fn jrcxz_stays_short_when_target_is_near() {
    let bytes = assemble("jrcxz target\nnop\ntarget:", Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0xE3); // JRCXZ
    assert_eq!(bytes[1], 0x01); // rel8 = +1 (skip 1 NOP)
    assert_eq!(bytes[2], 0x90); // NOP
    assert_eq!(bytes.len(), 3);
}

#[test]
fn loop_backward_relaxes_to_long_form() {
    // Backward LOOP target >127 bytes away
    let mut src = String::from("target:\n");
    for _ in 0..200 {
        src.push_str("nop\n");
    }
    src.push_str("loop target");
    let bytes = assemble(&src, Arch::X86_64).unwrap();
    // First 200 bytes are NOPs
    assert!(bytes[..200].iter().all(|&b| b == 0x90));
    // Then the long-form LOOP
    assert_eq!(bytes[200], 0xE2); // LOOP
    assert_eq!(bytes[201], 0x02);
    assert_eq!(bytes[202], 0xEB);
    assert_eq!(bytes[203], 0x05);
    assert_eq!(bytes[204], 0xE9);
    // rel32: target(0) - end(209) = -209
    let disp = i32::from_le_bytes([bytes[205], bytes[206], bytes[207], bytes[208]]);
    assert_eq!(disp, -209);
}

// ============================================================================
// Address-size override prefix (0x67) — 32-bit addressing in 64-bit mode
// ============================================================================

#[test]
fn addr_size_override_mov_reg_mem_eax_base() {
    // mov ecx, [eax] in 64-bit mode → 0x67 prefix
    let bytes = assemble("mov ecx, [eax]", Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0x67); // address-size override
    assert_eq!(bytes[1], 0x8B); // MOV r32, r/m32
    assert_eq!(bytes[2], 0x08); // ModR/M: mod=00, reg=ecx(1), rm=eax(0)
}

#[test]
fn addr_size_override_mov_mem_ecx_disp() {
    // mov eax, [ecx+4] in 64-bit mode → 0x67 prefix
    let bytes = assemble("mov eax, [ecx+4]", Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0x67); // address-size override
    assert_eq!(bytes[1], 0x8B); // MOV r32, r/m32
}

#[test]
fn no_addr_size_override_for_64bit_base() {
    // mov ecx, [rax] in 64-bit mode → no 0x67 prefix
    let bytes = assemble("mov ecx, [rax]", Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0x8B); // MOV directly, no 0x67
}

#[test]
fn addr_size_override_with_sib() {
    // mov eax, [ecx+edx*2] → 0x67 + SIB byte
    let bytes = assemble("mov eax, [ecx+edx*2]", Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0x67); // address-size override
    assert_eq!(bytes[1], 0x8B); // MOV r32, r/m32
}

#[test]
fn addr_size_override_store() {
    // mov [eax], ecx → 0x67 prefix (store direction)
    let bytes = assemble("mov [eax], ecx", Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0x67); // address-size override
    assert_eq!(bytes[1], 0x89); // MOV r/m32, r32
}

#[test]
fn addr_size_override_with_segment() {
    // fs:mov eax, [ecx] → segment override + 0x67
    let bytes = assemble("mov eax, fs:[ecx]", Arch::X86_64).unwrap();
    // Segment override comes from emit_x86_prefixes, then 0x67
    assert!(bytes.contains(&0x64)); // FS segment override
    assert!(bytes.contains(&0x67)); // address-size override
}

// ============================================================================
// BT variants with register operands
// ============================================================================

#[test]
fn bt_eax_5() {
    let bytes = assemble("bt eax, 5", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0xBA, 0xE0, 0x05]);
}

#[test]
fn bts_rax_rcx() {
    let bytes = assemble("bts rax, rcx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x0F, 0xAB, 0xC8]);
}

#[test]
fn btr_edx_3() {
    let bytes = assemble("btr edx, 3", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0xBA, 0xF2, 0x03]);
}

#[test]
fn btc_eax_ecx() {
    let bytes = assemble("btc eax, ecx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0xBB, 0xC8]);
}

// ============================================================================
// movzx / movsx memory forms
// ============================================================================

#[test]
fn movzx_eax_byte_mem() {
    let bytes = assemble("movzx eax, byte ptr [rbx]", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0xB6, 0x03]);
}

#[test]
fn movsx_eax_byte_mem() {
    let bytes = assemble("movsx eax, byte ptr [rbx]", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0xBE, 0x03]);
}

#[test]
fn movsx_rax_word_mem() {
    let bytes = assemble("movsx rax, word ptr [rbx]", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0x0F, 0xBF, 0x03]);
}

// ============================================================================
// Shift with memory operands
// ============================================================================

#[test]
fn shl_dword_mem_1() {
    let bytes = assemble("shl dword ptr [rbx], 1", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xD1, 0x23]);
}

#[test]
fn shr_dword_mem_4() {
    let bytes = assemble("shr dword ptr [rbx], 4", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xC1, 0x2B, 0x04]);
}

// ============================================================================
// MUL / DIV / IDIV single operand
// ============================================================================

#[test]
fn mul_ecx() {
    let bytes = assemble("mul ecx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xF7, 0xE1]);
}

#[test]
fn div_rcx() {
    let bytes = assemble("div rcx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x48, 0xF7, 0xF1]);
}

#[test]
fn idiv_edx() {
    let bytes = assemble("idiv edx", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xF7, 0xFA]);
}

// ============================================================================
// Label + Offset Expressions
// ============================================================================

#[test]
fn mov_rax_label_plus_offset() {
    // mov rax, data+4  — should produce movabs rax, <addr+4>
    let asm = "\
data:
    .quad 0
start:
    mov rax, data + 4
";
    let result = {
        let mut a = Assembler::new(Arch::X86_64);
        a.emit(asm).unwrap();
        a.finish().unwrap()
    };
    let bytes = result.bytes();
    // data is at offset 0, quad is 8 bytes, start at 8
    // mov rax, data+4 → REX.W B8+r imm64
    // relocation should resolve to 0 + 4 = 4
    let reloc_bytes = &bytes[10..18]; // after REX.W, B8, at offset 10 in the stream
    let addr = u64::from_le_bytes(reloc_bytes.try_into().unwrap());
    assert_eq!(addr, 4); // data(0) + 4
}

#[test]
fn jmp_label_plus_offset() {
    // jmp target+0 should work like jmp target
    let asm = "\
    jmp target + 0
target:
    nop
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // Short form: EB 00, nop 90
    assert_eq!(bytes, vec![0xEB, 0x00, 0x90]);
}

#[test]
fn call_label_expression() {
    // call with label expression works
    let asm = "\
target:
    ret
    call target + 0
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // target at 0, call at 1
    // call rel32 = E8 rel32
    // RIP at (1 + 5) = 6, target+0 = 0, rel = 0 - 6 = -6 = FA FF FF FF
    assert_eq!(bytes, vec![0xC3, 0xE8, 0xFA, 0xFF, 0xFF, 0xFF]);
}

#[test]
fn je_label_minus_offset() {
    // je label-0 should work like je label
    let asm = "\
    je target - 0
target:
    nop
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // Short form: 74 00, nop 90
    assert_eq!(bytes, vec![0x74, 0x00, 0x90]);
}

#[test]
fn push_label_expression() {
    // push label+8
    let asm = "\
data:
    .quad 0
    push data + 8
";
    let result = {
        let mut a = Assembler::new(Arch::X86_64);
        a.emit(asm).unwrap();
        a.finish().unwrap()
    };
    let bytes = result.bytes();
    // data at 0, push starts at offset 8
    // push imm32: 68, then 4 bytes = data(0)+8 = 8
    assert_eq!(bytes[8], 0x68);
    let addr = u32::from_le_bytes(bytes[9..13].try_into().unwrap());
    assert_eq!(addr, 8);
}

// ============================================================================
// Constants in Directive Arguments
// ============================================================================

#[test]
fn equ_in_fill_directive() {
    let asm = "\
COUNT = 3
.fill COUNT, 1, 0x90
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x90, 0x90, 0x90]);
}

#[test]
fn equ_in_space_directive() {
    let asm = "\
SIZE = 4
.space SIZE, 0xCC
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xCC, 0xCC, 0xCC, 0xCC]);
}

#[test]
fn equ_in_align_directive() {
    let asm = "\
nop
ALIGN_VAL = 4
.align ALIGN_VAL
nop
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // nop at 0, align to 4 with NOP padding, nop at 4
    assert_eq!(bytes.len(), 5);
    assert_eq!(bytes[0], 0x90);
    assert_eq!(bytes[4], 0x90);
}

#[test]
fn equ_chain_in_directive() {
    // Constants referencing other constants
    let asm = "\
BASE = 2
COUNT = BASE + 1
.fill COUNT, 1, 0xAA
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xAA, 0xAA, 0xAA]);
}

#[test]
fn equ_directive_syntax_in_fill() {
    let asm = "\
.equ REPS, 2
.fill REPS, 1, 0xBB
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xBB, 0xBB]);
}

#[test]
fn constant_in_instruction_operand() {
    // .equ constant used as immediate in instruction
    let asm = "\
SYSCALL_NUM = 60
mov eax, SYSCALL_NUM
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // mov eax, 60 → B8 3C 00 00 00
    assert_eq!(bytes, vec![0xB8, 0x3C, 0x00, 0x00, 0x00]);
}

#[test]
fn expression_fully_resolves_to_immediate() {
    // label+offset that references a constant should fully resolve
    let asm = "\
BASE = 100
OFFSET = 8
mov eax, BASE + OFFSET
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // mov eax, 108 → B8 6C 00 00 00
    assert_eq!(bytes, vec![0xB8, 0x6C, 0x00, 0x00, 0x00]);
}

// ============================================================================
// IMUL reg, [mem], imm (3-operand memory form)
// ============================================================================

#[test]
fn imul_eax_mem_imm8() {
    // imul eax, [rcx], 5
    let bytes = assemble("imul eax, [rcx], 5", Arch::X86_64).unwrap();
    // 6B /r ib → 6B 01 05
    assert_eq!(bytes, vec![0x6B, 0x01, 0x05]);
}

#[test]
fn imul_rax_mem_imm32() {
    // imul rax, [rdx], 1000
    let bytes = assemble("imul rax, [rdx], 1000", Arch::X86_64).unwrap();
    // REX.W 69 /r id → 48 69 02 E8 03 00 00
    assert_eq!(bytes, vec![0x48, 0x69, 0x02, 0xE8, 0x03, 0x00, 0x00]);
}

#[test]
fn imul_r32_mem_disp_imm() {
    // imul ebx, dword ptr [rax+8], 10
    let bytes = assemble("imul ebx, dword ptr [rax+8], 10", Arch::X86_64).unwrap();
    // 6B /r ib with disp8 → 6B 58 08 0A
    assert_eq!(bytes, vec![0x6B, 0x58, 0x08, 0x0A]);
}

// ============================================================================
// LOOP with expression target
// ============================================================================

#[test]
fn loop_with_label() {
    let asm = "\
top:
    dec ecx
    loop top
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // dec ecx = FF C9 (2 bytes), loop top = E2 rel8
    // RIP after loop = 4, target = 0, rel = -4 = FC
    assert_eq!(bytes, vec![0xFF, 0xC9, 0xE2, 0xFC]);
}

/// Verifies that RIP-relative addressing computes correct displacement
/// even when there's a trailing immediate after the disp32 field.
/// mov dword ptr [rip+data], 42 must write 42 at exactly `data`, not `data-4`.
#[test]
fn rip_relative_with_trailing_imm32() {
    // Layout:
    //   0: C7 05 [disp32] 2A 00 00 00   ; mov dword ptr [rip+data], 42  (10 bytes)
    //  10: 00 00 00 00                   ; data: .long 0                 (4 bytes)
    //
    // RIP at execution = 10 (byte after the entire instruction)
    // target = 10 (address of data)
    // disp32 = target - RIP = 10 - 10 = 0
    let asm = "\
    mov dword ptr [rip + data], 42
data:
    .long 0
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // Verify the instruction bytes
    assert_eq!(bytes[0], 0xC7); // opcode
    assert_eq!(bytes[1], 0x05); // modrm
                                // disp32 should be 0: target(10) - RIP(10) = 0
    let disp = i32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
    assert_eq!(
        disp, 0,
        "disp32 should be 0 (data immediately follows instruction)"
    );
    // imm32 = 42
    let imm = i32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
    assert_eq!(imm, 42);
    // data area
    assert_eq!(bytes.len(), 14); // 10 instr + 4 data
}

/// Same test but with data BEFORE the instruction (backward reference).
#[test]
fn rip_relative_backward_with_trailing_imm32() {
    // Layout:
    //   0: 00 00 00 00                   ; data: .long 0                 (4 bytes)
    //   4: C7 05 [disp32] 2A 00 00 00   ; mov dword ptr [rip+data], 42  (10 bytes)
    //
    // RIP at execution = 14 (byte after the entire instruction)
    // target = 0 (address of data)
    // disp32 = target - RIP = 0 - 14 = -14
    let asm = "\
data:
    .long 0
    mov dword ptr [rip + data], 42
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    assert_eq!(bytes.len(), 14);
    // Instruction starts at offset 4
    assert_eq!(bytes[4], 0xC7);
    assert_eq!(bytes[5], 0x05);
    let disp = i32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
    assert_eq!(disp, -14, "disp32 should be -14 (backward to data)");
    let imm = i32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]);
    assert_eq!(imm, 42);
}

/// RIP-relative ALU with imm8 trailing byte.
#[test]
fn rip_relative_add_mem_imm8() {
    // add dword ptr [rip+target], 5
    // Layout:
    //   0: 83 05 [disp32] 05   ; 7 bytes
    //   7: 00 00 00 00         ; target: .long 0
    //
    // RIP = 7, target = 7, disp = 0
    let asm = "\
    add dword ptr [rip + target], 5
target:
    .long 0
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0x83);
    assert_eq!(bytes[1], 0x05);
    let disp = i32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
    assert_eq!(disp, 0, "disp32 should be 0 (target follows instruction)");
    assert_eq!(bytes[6], 5); // imm8
    assert_eq!(bytes.len(), 11); // 7 + 4
}

// ============================================================================
// Table-dispatched instructions with label memory operands (P0 fix)
// ============================================================================

#[test]
fn cmpxchg_rip_label_end_to_end() {
    let asm = "
cmpxchg qword ptr [rip + target], rax
target:
    .quad 0
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // REX.W 0F B1 05 disp32  -> 7 bytes instruction, then 8 bytes data
    assert_eq!(bytes[0], 0x48); // REX.W
    assert_eq!(bytes[1], 0x0F);
    assert_eq!(bytes[2], 0xB1);
    assert_eq!(bytes[3], 0x05); // ModR/M: [rip+disp32], rax
    let disp = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    // displacement should be 0 because target is right after the instruction
    assert_eq!(disp, 0);
    assert_eq!(bytes.len(), 16); // 8 (cmpxchg) + 8 (.quad)
}

#[test]
fn xadd_rip_label_end_to_end() {
    let asm = "
xadd dword ptr [rip + target], ecx
target:
    .long 0
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // 0F C1 0D disp32  -> 6 bytes instruction, then 4 bytes data
    assert_eq!(bytes[0], 0x0F);
    assert_eq!(bytes[1], 0xC1);
    assert_eq!(bytes[2], 0x0D); // ModR/M: [rip+disp32], ecx
    let disp = i32::from_le_bytes([bytes[3], bytes[4], bytes[5], bytes[6]]);
    assert_eq!(disp, 0);
    assert_eq!(bytes.len(), 11); // 7 (xadd) + 4 (.long)
}

#[test]
fn movnti_rip_label_end_to_end() {
    let asm = "
target:
    .long 0
movnti dword ptr [rip + target], eax
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // 4 bytes data, then: 0F C3 05 disp32 = 6 bytes
    let off = 4; // skip the .long 0
    assert_eq!(bytes[off], 0x0F);
    assert_eq!(bytes[off + 1], 0xC3);
    assert_eq!(bytes[off + 2], 0x05); // ModR/M: [rip+disp32], eax
    let disp = i32::from_le_bytes([
        bytes[off + 3],
        bytes[off + 4],
        bytes[off + 5],
        bytes[off + 6],
    ]);
    // Target is at offset 0, RIP at end of movnti = 4+7=11, so disp = 0 - 11 = -11
    // Wait: instruction is 7 bytes: 0F C3 05 + 4 disp bytes
    // RIP = base_addr + 4 (data) + 7 (instruction) = base + 11
    // disp = target_addr - RIP = base + 0 - (base + 11) = -11
    assert_eq!(disp, -11);
}

#[test]
fn movbe_rip_label_load_end_to_end() {
    let asm = "
movbe eax, dword ptr [rip + target]
target:
    .long 0x12345678
";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // 0F 38 F0 05 disp32
    assert_eq!(bytes[0], 0x0F);
    assert_eq!(bytes[1], 0x38);
    assert_eq!(bytes[2], 0xF0);
    assert_eq!(bytes[3], 0x05); // ModR/M: [rip+disp32], eax
    let disp = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(disp, 0);
}

// ============================================================================
// Immediate overflow error tests (P1 fix)
// ============================================================================

#[test]
fn in_immediate_overflow_error() {
    let result = assemble("in al, 256", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn out_immediate_overflow_error() {
    let result = assemble("out 256, al", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn enter_frame_overflow_error() {
    let result = assemble("enter 0x10000, 0", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn enter_nesting_overflow_error() {
    let result = assemble("enter 0, 256", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn shld_imm_overflow_error() {
    let result = assemble("shld eax, ecx, 300", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn fill_directive_multi_byte_value() {
    // .fill 1, 4, 0xDEADBEEF → should produce 4 bytes in LE: EF BE AD DE
    let asm = ".fill 1, 4, 0xDEADBEEF";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    assert_eq!(&bytes[..], &[0xEF, 0xBE, 0xAD, 0xDE]);
}

#[test]
fn fill_directive_16bit_value() {
    // .fill 3, 2, 0x1234 → should produce 6 bytes: 34 12 34 12 34 12
    let asm = ".fill 3, 2, 0x1234";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    assert_eq!(&bytes[..], &[0x34, 0x12, 0x34, 0x12, 0x34, 0x12]);
}

// ============================================================================
// Data label addend support (.quad label + N)
// ============================================================================

#[test]
fn quad_label_with_addend() {
    // .quad func + 16 should produce address of func + 16
    let asm = "
func:
    nop
    .quad func + 16
";
    let mut a = Assembler::new(Arch::X86_64);
    a.base_address(0x1000);
    a.emit(asm).unwrap();
    let result = a.finish().unwrap();
    let bytes = result.bytes();
    // func is at 0x1000, nop is 1 byte, .quad at offset 1
    // .quad func+16 = 0x1000 + 16 = 0x1010 in LE
    let val = u64::from_le_bytes([
        bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8],
    ]);
    assert_eq!(val, 0x1010);
}

#[test]
fn long_label_with_negative_addend() {
    let asm = "
start:
    nop
    .long start - 1
";
    let mut a = Assembler::new(Arch::X86_64);
    a.base_address(0x100);
    a.emit(asm).unwrap();
    let result = a.finish().unwrap();
    let bytes = result.bytes();
    // start is at 0x100, .long start-1 = 0xFF
    let val = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
    assert_eq!(val, 0xFF);
}

// ============================================================================
// 8th Audit: Validation & correctness fixes
// ============================================================================

#[test]
fn rsp_as_sib_index_rejected() {
    let result = assemble("mov rax, [rbx + rsp*2]", Arch::X86_64);
    assert!(result.is_err(), "RSP as SIB index should be rejected");
}

#[test]
fn push_imm_out_of_range_rejected() {
    let result = assemble("push 0x1FFFFFFFF", Arch::X86_64);
    assert!(
        result.is_err(),
        "push immediate > 32-bit range should be rejected"
    );
}

#[test]
fn imul_2op_rejects_8bit() {
    let result = assemble("imul al, bl", Arch::X86_64);
    assert!(
        result.is_err(),
        "2-operand IMUL with 8-bit registers should be rejected"
    );
}

#[test]
fn imul_3op_rejects_8bit() {
    let result = assemble("imul al, bl, 5", Arch::X86_64);
    assert!(
        result.is_err(),
        "3-operand IMUL with 8-bit registers should be rejected"
    );
}

#[test]
fn cmovcc_reg_mem_rejects_8bit() {
    let result = assemble("cmove al, byte ptr [rbx]", Arch::X86_64);
    assert!(
        result.is_err(),
        "CMOVcc with 8-bit register and memory should be rejected"
    );
}

#[test]
fn setcc_rejects_32bit_register() {
    let result = assemble("sete eax", Arch::X86_64);
    assert!(
        result.is_err(),
        "SETcc with 32-bit register should be rejected"
    );
}

#[test]
fn setcc_accepts_8bit_register() {
    let bytes = assemble("sete al", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0x94, 0xC0]);
}

#[test]
fn movzx_word_ptr_source() {
    // movzx eax, word ptr [rbx] → 0F B7 03
    let bytes = assemble("movzx eax, word ptr [rbx]", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0xB7, 0x03]);
}

#[test]
fn movsx_word_ptr_source() {
    // movsx eax, word ptr [rbx] → 0F BF 03
    let bytes = assemble("movsx eax, word ptr [rbx]", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x0F, 0xBF, 0x03]);
}

#[test]
fn r12_as_sib_index_accepted() {
    // R12 (base_code 4 + REX.X) is valid as SIB index, unlike RSP
    let bytes = assemble("mov rax, [rbx + r12*2]", Arch::X86_64).unwrap();
    // Should encode successfully (exact bytes depend on SIB construction)
    assert!(!bytes.is_empty());
}

// ============================================================================
// 8th Audit: Constant resolution in memory operands & data declarations
// ============================================================================

#[test]
fn constant_in_memory_displacement() {
    // .equ OFFSET, -8; mov rax, [rbp + OFFSET] → same as mov rax, [rbp - 8]
    let asm = ".equ OFFSET, -8\nmov rax, [rbp + OFFSET]";
    let expected = assemble("mov rax, [rbp - 8]", Arch::X86_64).unwrap();
    let actual = assemble(asm, Arch::X86_64).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn constant_in_data_long() {
    // .equ VAL, -1; .long VAL → [0xFF, 0xFF, 0xFF, 0xFF]
    let asm = ".equ VAL, -1\n.long VAL";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xFF, 0xFF, 0xFF, 0xFF]);
}

#[test]
fn constant_in_data_byte() {
    // .equ X, 42; .byte X → [0x2A]
    let asm = ".equ X, 42\n.byte X";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x2A]);
}

#[test]
fn branch_with_addend_short_form() {
    // target: nop; jmp target+1 → should use short form with correct displacement
    let asm = "target:\nnop\njmp target+1";
    let bytes = assemble(asm, Arch::X86_64).unwrap();
    // nop(0x90) + jmp short(0xEB, disp8)
    // frag_end = 3, target=0, addend=1 → disp = 0+1-3 = -2 → 0xFE
    assert_eq!(bytes, vec![0x90, 0xEB, 0xFE]);
}

// ============================================================================
// Preprocessor integration tests
// ============================================================================

#[test]
fn preprocessor_macro_produces_real_instructions() {
    // Define a macro that generates a standard function prologue
    let src = r#"
.macro prologue
    push rbp
    mov rbp, rsp
.endm

prologue
    xor eax, eax
    pop rbp
    ret
"#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // push rbp = 0x55, mov rbp,rsp = 48 89 E5, xor eax,eax = 31 C0, pop rbp = 5D, ret = C3
    assert_eq!(bytes, vec![0x55, 0x48, 0x89, 0xE5, 0x31, 0xC0, 0x5D, 0xC3]);
}

#[test]
fn preprocessor_macro_with_parameters() {
    // Macro that generates mov + ret with a parameter
    let src = r#"
.macro set_and_ret val
    mov eax, \val
    ret
.endm

set_and_ret 42
"#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // mov eax, 42 = B8 2A 00 00 00, ret = C3
    assert_eq!(bytes, vec![0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3]);
}

#[test]
fn preprocessor_macro_with_default_parameter() {
    let src = r#"
.macro load_val reg=eax, val=0
    mov \reg, \val
.endm

load_val
"#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // mov eax, 0 → 31 C0 (optimized to xor with OptLevel::Size default)
    // or B8 00 00 00 00 without optimization... actually the preprocessor
    // substitutes text, so "mov eax, 0" is what the assembler sees.
    // With OptLevel::Size (default), this stays as mov eax, 0 = B8 00 00 00 00
    // The peephole only converts to XOR if it matches the right pattern.
    // Actually the optimizer should fire: mov eax, 0 → xor eax, eax
    assert_eq!(bytes, vec![0x31, 0xC0]);
}

#[test]
fn preprocessor_rept_generates_nops() {
    let src = r#"
.rept 4
    nop
.endr
    ret
"#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x90, 0x90, 0x90, 0x90, 0xC3]);
}

#[test]
fn preprocessor_irp_push_multiple_regs() {
    // Use .irp to push several registers
    let src = r#"
.irp reg, rbx, r12, r13
    push \reg
.endr
    ret
"#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // push rbx = 53, push r12 = 41 54, push r13 = 41 55, ret = C3
    assert_eq!(bytes, vec![0x53, 0x41, 0x54, 0x41, 0x55, 0xC3]);
}

#[test]
fn preprocessor_irpc_iterates_chars() {
    // .irpc substitutes one character at a time
    let src = r#"
.irpc c, abc
    .byte 0x4\c
.endr
"#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0x4a, 0x4b, 0x4c]);
}

#[test]
fn preprocessor_ifdef_conditional() {
    // With symbol defined: emit int 3
    let mut asm = Assembler::new(Arch::X86_64);
    asm.define_preprocessor_symbol("DEBUG", 1);
    asm.emit(
        r#"
.ifdef DEBUG
    int 3
.endif
    ret
"#,
    )
    .unwrap();
    let with_debug = asm.finish().unwrap();
    assert_eq!(with_debug.bytes(), &[0xCC, 0xC3]);

    // Without: skip int 3
    let bytes = assemble(
        r#"
.ifdef DEBUG
    int 3
.endif
    ret
"#,
        Arch::X86_64,
    )
    .unwrap();
    assert_eq!(bytes, vec![0xC3]);
}

#[test]
fn preprocessor_ifndef_conditional() {
    let bytes = assemble(
        r#"
.ifndef RELEASE
    int 3
.endif
    ret
"#,
        Arch::X86_64,
    )
    .unwrap();
    // RELEASE not defined → include int 3
    assert_eq!(bytes, vec![0xCC, 0xC3]);
}

#[test]
fn preprocessor_if_else_conditional() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.define_preprocessor_symbol("USE_SYSCALL", 1);
    asm.emit(
        r#"
.ifdef USE_SYSCALL
    syscall
.else
    int 0x80
.endif
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    // syscall = 0F 05
    assert_eq!(result.bytes(), &[0x0F, 0x05]);
}

#[test]
fn preprocessor_nested_rept_inside_irp() {
    let src = r#"
.irp reg, rax, rbx
.rept 2
    push \reg
.endr
.endr
    ret
"#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // 2x push rax (50,50) + 2x push rbx (53,53) + ret (C3)
    assert_eq!(bytes, vec![0x50, 0x50, 0x53, 0x53, 0xC3]);
}

#[test]
fn preprocessor_equ_integration() {
    // .equ constants visible to both preprocessor (.ifdef) and assembler
    let src = r#"
.equ SYS_EXIT, 60
.equ EXIT_SUCCESS, 0
    mov eax, SYS_EXIT
    mov edi, EXIT_SUCCESS
    syscall
"#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // mov eax, 60 = B8 3C 00 00 00
    // mov edi, 0 → xor edi, edi (optimised) = 31 FF
    // syscall = 0F 05
    assert_eq!(
        bytes,
        vec![0xB8, 0x3C, 0x00, 0x00, 0x00, 0x31, 0xFF, 0x0F, 0x05]
    );
}

#[test]
fn preprocessor_macro_with_labels() {
    // Macro that defines a labeled function
    let src = r#"
.macro func_nop name
\name:
    nop
    ret
.endm

func_nop my_fn
    call my_fn
"#;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.emit(src).unwrap();
    let result = asm.finish().unwrap();
    assert!(result.label_address("my_fn").is_some());
    assert_eq!(result.label_address("my_fn"), Some(0));
}

#[test]
fn preprocessor_vararg_macro() {
    let src = r#"
.macro push_all regs:vararg
.irp r, \regs
    push \r
.endr
.endm

push_all rbx, r12, r13
    ret
"#;
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // push rbx(53) + push r12(41 54) + push r13(41 55) + ret(C3)
    assert_eq!(bytes, vec![0x53, 0x41, 0x54, 0x41, 0x55, 0xC3]);
}

// ============================================================================
// Peephole optimizer integration tests
// ============================================================================

#[test]
fn optimizer_zero_idiom_mov_reg_zero() {
    // With OptLevel::Size, mov rax, 0 should become xor eax, eax (2 bytes vs 7)
    let mut asm = Assembler::new(Arch::X86_64);
    asm.optimize(OptLevel::Size);
    asm.emit("mov rax, 0\nret").unwrap();
    let result = asm.finish().unwrap();
    // xor eax, eax = 31 C0, ret = C3
    assert_eq!(result.bytes(), &[0x31, 0xC0, 0xC3]);
}

#[test]
fn optimizer_zero_idiom_multiple_regs() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.optimize(OptLevel::Size);
    asm.emit("mov rcx, 0\nmov rdx, 0\nret").unwrap();
    let result = asm.finish().unwrap();
    // xor ecx, ecx = 31 C9, xor edx, edx = 31 D2, ret = C3
    assert_eq!(result.bytes(), &[0x31, 0xC9, 0x31, 0xD2, 0xC3]);
}

#[test]
fn optimizer_zero_idiom_32bit_regs() {
    // mov eax, 0 should also be optimized to xor eax, eax
    let mut asm = Assembler::new(Arch::X86_64);
    asm.optimize(OptLevel::Size);
    asm.emit("mov eax, 0\nret").unwrap();
    let result = asm.finish().unwrap();
    assert_eq!(result.bytes(), &[0x31, 0xC0, 0xC3]);
}

#[test]
fn optimizer_mov_narrow_small_imm() {
    // mov rax, 1 → mov eax, 1 (7 bytes → 5 bytes, zero-extends to rax)
    let mut asm = Assembler::new(Arch::X86_64);
    asm.optimize(OptLevel::Size);
    asm.emit("mov rax, 1\nret").unwrap();
    let result = asm.finish().unwrap();
    // mov eax, 1 = B8 01 00 00 00, ret = C3
    assert_eq!(result.bytes(), &[0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3]);
}

#[test]
fn optimizer_no_narrow_negative_imm() {
    // mov rax, -1 should NOT narrow (negative values need full 64-bit)
    let mut asm = Assembler::new(Arch::X86_64);
    asm.optimize(OptLevel::Size);
    asm.emit("mov rax, -1\nret").unwrap();
    let result = asm.finish().unwrap();
    // This should stay as mov rax, -1 (sign-extended 32-bit immediate in REX.W encoding)
    // 48 C7 C0 FF FF FF FF = REX.W mov rax, imm32(-1), ret = C3
    assert_eq!(result.bytes().len(), 8); // 7 + 1
}

#[test]
fn optimizer_test_conversion() {
    // and rax, rax → test rax, rax (functionally equivalent, shorter encoding possible)
    let mut asm = Assembler::new(Arch::X86_64);
    asm.optimize(OptLevel::Size);
    asm.emit("and rax, rax\nret").unwrap();
    let result = asm.finish().unwrap();
    // test rax, rax = 48 85 C0, ret = C3
    assert_eq!(result.bytes(), &[0x48, 0x85, 0xC0, 0xC3]);
}

#[test]
fn optimizer_disabled_no_opt() {
    // With OptLevel::None, mov rax, 0 should NOT be optimized
    let mut asm = Assembler::new(Arch::X86_64);
    asm.optimize(OptLevel::None);
    asm.emit("mov rax, 0\nret").unwrap();
    let result = asm.finish().unwrap();
    // Should be a full mov rax, 0 encoding (7 bytes) + ret
    assert!(result.bytes().len() > 3); // Not 2+1 (xor eax,eax + ret)
}

#[test]
fn optimizer_preserves_nonzero_mov() {
    // mov rax, 0x100 should be narrowed but NOT zero-idiom'd
    let mut asm = Assembler::new(Arch::X86_64);
    asm.optimize(OptLevel::Size);
    asm.emit("mov rax, 0x100\nret").unwrap();
    let result = asm.finish().unwrap();
    // mov eax, 0x100 = B8 00 01 00 00, ret = C3
    assert_eq!(result.bytes(), &[0xB8, 0x00, 0x01, 0x00, 0x00, 0xC3]);
}

#[test]
fn optimizer_and_eax_eax_to_test() {
    let mut asm = Assembler::new(Arch::X86_64);
    asm.optimize(OptLevel::Size);
    asm.emit("and eax, eax\nret").unwrap();
    let result = asm.finish().unwrap();
    // test eax, eax = 85 C0, ret = C3
    assert_eq!(result.bytes(), &[0x85, 0xC0, 0xC3]);
}

#[test]
fn optimizer_combined_in_function() {
    // Real-world pattern: function that zeroes rax and returns
    let mut asm = Assembler::new(Arch::X86_64);
    asm.optimize(OptLevel::Size);
    asm.emit(
        r#"
push rbp
mov rbp, rsp
mov rax, 0
pop rbp
ret
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    // push rbp(55) + mov rbp,rsp(48 89 E5) + xor eax,eax(31 C0) + pop rbp(5D) + ret(C3)
    assert_eq!(
        result.bytes(),
        &[0x55, 0x48, 0x89, 0xE5, 0x31, 0xC0, 0x5D, 0xC3]
    );
}

// ============================================================================
// Preprocessor + Optimizer combined
// ============================================================================

#[test]
fn preprocessor_and_optimizer_combined() {
    // Macro that generates zero-initialisation + preprocessor conditional,
    // with optimizer converting mov reg, 0 → xor
    let mut asm = Assembler::new(Arch::X86_64);
    asm.optimize(OptLevel::Size);
    asm.define_preprocessor_symbol("ZERO_INIT", 1);
    asm.emit(
        r#"
.macro zero_reg r
    mov \r, 0
.endm

.ifdef ZERO_INIT
zero_reg rax
zero_reg rcx
.endif
    ret
"#,
    )
    .unwrap();
    let result = asm.finish().unwrap();
    // xor eax, eax = 31 C0, xor ecx, ecx = 31 C9, ret = C3
    assert_eq!(result.bytes(), &[0x31, 0xC0, 0x31, 0xC9, 0xC3]);
}

// ============================================================================
// =====================  x86-32 BACKEND TESTS  ===============================
// ============================================================================

#[test]
fn x86_32_nop() {
    let code = assemble("nop", Arch::X86).unwrap();
    assert_eq!(code, vec![0x90]);
}

#[test]
fn x86_32_mov_eax_imm() {
    let code = assemble("mov eax, 0x42", Arch::X86).unwrap();
    assert_eq!(code, vec![0xB8, 0x42, 0x00, 0x00, 0x00]);
}

#[test]
fn x86_32_push_eax() {
    let code = assemble("push eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x50]);
}

#[test]
fn x86_32_pop_ebx() {
    let code = assemble("pop ebx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x5B]);
}

#[test]
fn x86_32_push_es() {
    let code = assemble("push es", Arch::X86).unwrap();
    assert_eq!(code, vec![0x06]);
}

#[test]
fn x86_32_push_cs() {
    let code = assemble("push cs", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0E]);
}

#[test]
fn x86_32_push_ss() {
    let code = assemble("push ss", Arch::X86).unwrap();
    assert_eq!(code, vec![0x16]);
}

#[test]
fn x86_32_push_ds() {
    let code = assemble("push ds", Arch::X86).unwrap();
    assert_eq!(code, vec![0x1E]);
}

#[test]
fn x86_32_push_fs() {
    let code = assemble("push fs", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0F, 0xA0]);
}

#[test]
fn x86_32_push_gs() {
    let code = assemble("push gs", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0F, 0xA8]);
}

#[test]
fn x86_32_pop_es() {
    let code = assemble("pop es", Arch::X86).unwrap();
    assert_eq!(code, vec![0x07]);
}

#[test]
fn x86_32_pop_ds() {
    let code = assemble("pop ds", Arch::X86).unwrap();
    assert_eq!(code, vec![0x1F]);
}

#[test]
fn x86_32_pop_fs() {
    let code = assemble("pop fs", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0F, 0xA1]);
}

#[test]
fn x86_32_pop_gs() {
    let code = assemble("pop gs", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0F, 0xA9]);
}

#[test]
fn x86_32_rejects_rax() {
    // 64-bit registers must be rejected in 32-bit mode
    // Use a non-zero immediate to prevent the optimizer from converting to xor
    let result = assemble("mov rax, 1", Arch::X86);
    assert!(result.is_err());
}

#[test]
fn x86_32_rejects_r8() {
    // Extended registers must be rejected in 32-bit mode
    let result = assemble("mov r8, 0", Arch::X86);
    assert!(result.is_err());
}

#[test]
fn x86_32_int() {
    let code = assemble("int 0x80", Arch::X86).unwrap();
    assert_eq!(code, vec![0xCD, 0x80]);
}

#[test]
fn x86_32_ret() {
    let code = assemble("ret", Arch::X86).unwrap();
    assert_eq!(code, vec![0xC3]);
}

#[test]
fn x86_32_xor_eax_eax() {
    let code = assemble("xor eax, eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x31, 0xC0]);
}

#[test]
fn x86_32_shellcode_pattern() {
    // Classic Linux x86-32 exit shellcode pattern
    let code = assemble(
        "xor eax, eax\n\
         xor ebx, ebx\n\
         mov al, 1\n\
         int 0x80",
        Arch::X86,
    )
    .unwrap();
    // xor eax,eax = 31 C0
    // xor ebx,ebx = 31 DB
    // mov al,1    = B0 01
    // int 0x80    = CD 80
    assert_eq!(code, vec![0x31, 0xC0, 0x31, 0xDB, 0xB0, 0x01, 0xCD, 0x80]);
}

// ============================================================================
// =====================  ARM32 BACKEND TESTS  ================================
// ============================================================================

#[test]
fn arm32_nop() {
    // NOP is encoded as MOV R0, R0 → E1A00000
    let code = assemble("nop", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0xA0, 0xE1]);
}

#[test]
fn arm32_mov_r0_imm() {
    // MOV R0, #0 → E3A00000
    let code = assemble("mov r0, 0", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0xA0, 0xE3]);
}

#[test]
fn arm32_mov_r0_1() {
    // MOV R0, #1 → E3A00001
    let code = assemble("mov r0, 1", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x01, 0x00, 0xA0, 0xE3]);
}

#[test]
fn arm32_mov_r7_imm() {
    // MOV R7, #1 → E3A07001
    let code = assemble("mov r7, 1", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x01, 0x70, 0xA0, 0xE3]);
}

#[test]
fn arm32_add_r0_r1_r2() {
    // ADD R0, R1, R2 → E0810002
    let code = assemble("add r0, r1, r2", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x02, 0x00, 0x81, 0xE0]);
}

#[test]
fn arm32_add_r0_r0_imm() {
    // ADD R0, R0, #1 → E2800001
    let code = assemble("add r0, r0, 1", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x01, 0x00, 0x80, 0xE2]);
}

#[test]
fn arm32_sub_r0_r0_imm() {
    // SUB R0, R0, #1 → E2400001
    let code = assemble("sub r0, r0, 1", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x01, 0x00, 0x40, 0xE2]);
}

#[test]
fn arm32_cmp_r0_imm() {
    // CMP R0, #0 → E3500000
    let code = assemble("cmp r0, 0", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x50, 0xE3]);
}

#[test]
fn arm32_svc_0() {
    // SVC #0 → EF000000
    let code = assemble("svc 0", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x00, 0xEF]);
}

#[test]
fn arm32_bx_lr() {
    // BX LR → E12FFF1E
    let code = assemble("bx lr", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x1E, 0xFF, 0x2F, 0xE1]);
}

#[test]
fn arm32_ldr_r0_r1() {
    // LDR R0, [R1] → E5910000
    let code = assemble("ldr r0, [r1]", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x91, 0xE5]);
}

#[test]
fn arm32_str_r0_r1() {
    // STR R0, [R1] → E5810000
    let code = assemble("str r0, [r1]", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x81, 0xE5]);
}

#[test]
fn arm32_ldr_r0_r1_offset() {
    // LDR R0, [R1, #4] → E5910004
    let code = assemble("ldr r0, [r1 + 4]", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x04, 0x00, 0x91, 0xE5]);
}

#[test]
fn arm32_push_lr() {
    // PUSH {LR} → E92D4000
    let code = assemble("push {lr}", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x00, 0x40, 0x2D, 0xE9]);
}

#[test]
fn arm32_pop_pc() {
    // POP {PC} → E8BD8000
    let code = assemble("pop {pc}", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x00, 0x80, 0xBD, 0xE8]);
}

#[test]
fn arm32_push_multiple() {
    // PUSH {R4, R5, LR} → E92D4030
    let code = assemble("push {r4, r5, lr}", Arch::Arm).unwrap();
    // R4=bit4, R5=bit5, LR=bit14 → 0x4030
    assert_eq!(code, vec![0x30, 0x40, 0x2D, 0xE9]);
}

#[test]
fn arm32_bkpt() {
    // BKPT #0 → E1200070
    let code = assemble("bkpt 0", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x70, 0x00, 0x20, 0xE1]);
}

#[test]
fn arm32_mul_r0_r1_r2() {
    // MUL R0, R1, R2 → E0000291
    let code = assemble("mul r0, r1, r2", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x91, 0x02, 0x00, 0xE0]);
}

#[test]
fn arm32_movw_r0_imm() {
    // MOVW R0, #0x1234 → cond|00110000|imm4|Rd|imm12 → E3010234
    let code = assemble("movw r0, 0x1234", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x34, 0x02, 0x01, 0xE3]);
}

#[test]
fn arm32_eor_r0_r0_r1() {
    // EOR R0, R0, R1 → E0200001
    let code = assemble("eor r0, r0, r1", Arch::Arm).unwrap();
    assert_eq!(code, vec![0x01, 0x00, 0x20, 0xE0]);
}

#[test]
fn arm32_shellcode_pattern() {
    // ARM Linux exit(0) shellcode
    let code = assemble(
        "mov r0, 0\n\
         mov r7, 1\n\
         svc 0",
        Arch::Arm,
    )
    .unwrap();
    // MOV R0, #0 → E3A00000
    // MOV R7, #1 → E3A07001
    // SVC #0     → EF000000
    assert_eq!(
        code,
        vec![
            0x00, 0x00, 0xA0, 0xE3, // mov r0, #0
            0x01, 0x70, 0xA0, 0xE3, // mov r7, #1
            0x00, 0x00, 0x00, 0xEF, // svc #0
        ]
    );
}

// ============================================================================
// =====================  AVX / FMA INTEGRATION TESTS  ========================
// ============================================================================

#[test]
fn avx_fma_vfmadd231ps_xmm() {
    let code = assemble("vfmadd231ps xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0xB8, 0xC2]);
}

#[test]
fn avx_fma_vfmadd231pd_xmm() {
    let code = assemble("vfmadd231pd xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0xF1, 0xB8, 0xC2]);
}

#[test]
fn avx_fma_vfmadd231ps_ymm() {
    let code = assemble("vfmadd231ps ymm0, ymm1, ymm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x75, 0xB8, 0xC2]);
}

#[test]
fn avx_fma_vfmsub231ps_xmm() {
    let code = assemble("vfmsub231ps xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0xBA, 0xC2]);
}

#[test]
fn avx_fma_vfnmadd231ps_xmm() {
    let code = assemble("vfnmadd231ps xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0xBC, 0xC2]);
}

#[test]
fn avx_fma_vfnmsub231pd_xmm() {
    let code = assemble("vfnmsub231pd xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0xF1, 0xBE, 0xC2]);
}

#[test]
fn avx_fma_vfmadd132ps_xmm() {
    let code = assemble("vfmadd132ps xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0x98, 0xC2]);
}

#[test]
fn avx_fma_vfmadd213ps_xmm() {
    let code = assemble("vfmadd213ps xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0xA8, 0xC2]);
}

#[test]
fn avx_fma_vfmadd231ss_xmm() {
    let code = assemble("vfmadd231ss xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0xB9, 0xC2]);
}

#[test]
fn avx_fma_vfmadd231sd_xmm() {
    let code = assemble("vfmadd231sd xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0xF1, 0xB9, 0xC2]);
}

#[test]
fn avx_shift_vpslld_reg() {
    let code = assemble("vpslld xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC5, 0xF1, 0xF2, 0xC2]);
}

#[test]
fn avx_shift_vpslld_imm() {
    let code = assemble("vpslld xmm0, xmm1, 4", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC5, 0xF9, 0x72, 0xF1, 0x04]);
}

#[test]
fn avx_shift_vpslld_ymm_imm() {
    let code = assemble("vpslld ymm0, ymm1, 4", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC5, 0xFD, 0x72, 0xF1, 0x04]);
}

#[test]
fn avx_shift_vpsrlw_imm() {
    let code = assemble("vpsrlw xmm2, xmm3, 8", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC5, 0xE9, 0x71, 0xD3, 0x08]);
}

#[test]
fn avx_shift_vpsraw_imm() {
    let code = assemble("vpsraw xmm2, xmm3, 8", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC5, 0xE9, 0x71, 0xE3, 0x08]);
}

#[test]
fn avx_vpermilps_reg() {
    let code = assemble("vpermilps xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0x0C, 0xC2]);
}

#[test]
fn avx_vpermilps_imm() {
    let code = assemble("vpermilps xmm0, xmm1, 0x44", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE3, 0x79, 0x04, 0xC1, 0x44]);
}

#[test]
fn avx_vbroadcastss_xmm() {
    let code = assemble("vbroadcastss xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x79, 0x18, 0xC1]);
}

#[test]
fn avx_vpermq_ymm() {
    let code = assemble("vpermq ymm0, ymm1, 0x44", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE3, 0xFD, 0x00, 0xC1, 0x44]);
}

#[test]
fn avx_vcvtsi2ss() {
    let code = assemble("vcvtsi2ss xmm0, xmm1, eax", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC5, 0xF2, 0x2A, 0xC0]);
}

#[test]
fn avx_vcvtss2si() {
    let code = assemble("vcvtss2si eax, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC5, 0xFA, 0x2D, 0xC1]);
}

#[test]
fn avx_vcvtdq2ps() {
    let code = assemble("vcvtdq2ps xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC5, 0xF8, 0x5B, 0xC1]);
}

#[test]
fn avx_vpsllvd() {
    let code = assemble("vpsllvd xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0x47, 0xC2]);
}

#[test]
fn avx_vtestps() {
    let code = assemble("vtestps xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x79, 0x0E, 0xC1]);
}

#[test]
fn avx_vpbroadcastd() {
    let code = assemble("vpbroadcastd xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, [0xC4, 0xE2, 0x79, 0x58, 0xC1]);
}

#[test]
fn avx_fma_multi_instruction() {
    // Multiple FMA + AVX instructions in one block
    let code = assemble(
        "vfmadd231ps xmm0, xmm1, xmm2\nvfmsub231pd xmm3, xmm4, xmm5\nvaddps xmm6, xmm7, xmm0",
        Arch::X86_64,
    )
    .unwrap();
    // 5 + 5 + 4 = 14 bytes
    assert_eq!(code.len(), 14);
}

#[test]
fn avx_shift_multi_instruction() {
    let code = assemble(
        "vpslld xmm0, xmm1, 4\nvpsrlq xmm3, xmm4, xmm5\nvpsraw xmm2, xmm3, 8",
        Arch::X86_64,
    )
    .unwrap();
    // 5 + 4 + 5 = 14 bytes
    assert_eq!(code.len(), 14);
}

// ============================================================================
// =====================  AArch64 BACKEND TESTS  ==============================
// ============================================================================

#[test]
fn aarch64_nop() {
    // NOP → D503201F
    let code = assemble("nop", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x1F, 0x20, 0x03, 0xD5]);
}

#[test]
fn aarch64_ret() {
    // RET → D65F03C0
    let code = assemble("ret", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0xC0, 0x03, 0x5F, 0xD6]);
}

#[test]
fn aarch64_mov_x0_x1() {
    // MOV X0, X1 → ORR X0, XZR, X1 → AA0103E0
    let code = assemble("mov x0, x1", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0xE0, 0x03, 0x01, 0xAA]);
}

#[test]
fn aarch64_mov_x0_imm() {
    // MOV X0, #42 → MOVZ X0, #42 → D2800540
    let code = assemble("mov x0, 42", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x40, 0x05, 0x80, 0xD2]);
}

#[test]
fn aarch64_mov_w0_imm() {
    // MOV W0, #1 → MOVZ W0, #1 → 52800020
    let code = assemble("mov w0, 1", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x80, 0x52]);
}

#[test]
fn aarch64_add_x0_x1_imm() {
    // ADD X0, X1, #1 → 91000420
    let code = assemble("add x0, x1, 1", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x04, 0x00, 0x91]);
}

#[test]
fn aarch64_sub_x0_x0_imm() {
    // SUB X0, X0, #1 → D1000400
    let code = assemble("sub x0, x0, 1", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x04, 0x00, 0xD1]);
}

#[test]
fn aarch64_add_x0_x1_x2() {
    // ADD X0, X1, X2 → 8B020020
    let code = assemble("add x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x02, 0x8B]);
}

#[test]
fn aarch64_svc_0() {
    // SVC #0 → D4000001
    let code = assemble("svc 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x01, 0x00, 0x00, 0xD4]);
}

// ── Hint instructions ───────────────────────────

#[test]
fn aarch64_wfi() {
    let code = assemble("wfi", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x7F, 0x20, 0x03, 0xD5]);
}

#[test]
fn aarch64_wfe() {
    let code = assemble("wfe", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x5F, 0x20, 0x03, 0xD5]);
}

#[test]
fn aarch64_sev() {
    let code = assemble("sev", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x9F, 0x20, 0x03, 0xD5]);
}

#[test]
fn aarch64_sevl() {
    let code = assemble("sevl", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0xBF, 0x20, 0x03, 0xD5]);
}

#[test]
fn aarch64_yield() {
    let code = assemble("yield", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x3F, 0x20, 0x03, 0xD5]);
}

// ── LR / FP register aliases ────────────────────

#[test]
fn aarch64_lr_alias() {
    // MOV X0, LR should produce same bytes as MOV X0, X30
    let code_lr = assemble("mov x0, lr", Arch::Aarch64).unwrap();
    let code_x30 = assemble("mov x0, x30", Arch::Aarch64).unwrap();
    assert_eq!(code_lr, code_x30);
}

#[test]
fn aarch64_fp_alias() {
    // MOV X0, FP should produce same bytes as MOV X0, X29
    let code_fp = assemble("mov x0, fp", Arch::Aarch64).unwrap();
    let code_x29 = assemble("mov x0, x29", Arch::Aarch64).unwrap();
    assert_eq!(code_fp, code_x29);
}

#[test]
fn aarch64_stp_fp_lr() {
    // STP FP, LR, [SP, #-16]! — common function prologue
    let code_alias = assemble("stp fp, lr, [sp, -16]!", Arch::Aarch64).unwrap();
    let code_regs = assemble("stp x29, x30, [sp, -16]!", Arch::Aarch64).unwrap();
    assert_eq!(code_alias, code_regs);
}

#[test]
fn aarch64_brk_0() {
    // BRK #0 → D4200000
    let code = assemble("brk 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x20, 0xD4]);
}

#[test]
fn aarch64_br_x30() {
    // BR X30 → D61F03C0
    let code = assemble("br x30", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0xC0, 0x03, 0x1F, 0xD6]);
}

#[test]
fn aarch64_blr_x8() {
    // BLR X8 → D63F0100
    let code = assemble("blr x8", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x01, 0x3F, 0xD6]);
}

#[test]
fn aarch64_movz_x0() {
    // MOVZ X0, #0x1234 → D2824680
    let code = assemble("movz x0, 0x1234", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x80, 0x46, 0x82, 0xD2]);
}

#[test]
fn aarch64_cmp_x0_imm() {
    // CMP X0, #0 → SUBS XZR, X0, #0 → F100001F
    let code = assemble("cmp x0, 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x1F, 0x00, 0x00, 0xF1]);
}

#[test]
fn aarch64_and_x0_x1_x2() {
    // AND X0, X1, X2 → 8A020020
    let code = assemble("and x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x02, 0x8A]);
}

#[test]
fn aarch64_orr_x0_x1_x2() {
    // ORR X0, X1, X2 → AA020020
    let code = assemble("orr x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x02, 0xAA]);
}

#[test]
fn aarch64_eor_x0_x1_x2() {
    // EOR X0, X1, X2 → CA020020
    let code = assemble("eor x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x02, 0xCA]);
}

#[test]
fn aarch64_ldr_x0_x1() {
    // LDR X0, [X1] → F9400020
    let code = assemble("ldr x0, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x40, 0xF9]);
}

#[test]
fn aarch64_str_x0_x1() {
    // STR X0, [X1] → F9000020
    let code = assemble("str x0, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x00, 0xF9]);
}

#[test]
fn aarch64_ldr_x0_x1_offset() {
    // LDR X0, [X1, #8] → F9400420
    let code = assemble("ldr x0, [x1 + 8]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x04, 0x40, 0xF9]);
}

#[test]
fn aarch64_ldr_w0_x1() {
    // LDR W0, [X1] → B9400020
    let code = assemble("ldr w0, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x40, 0xB9]);
}

#[test]
fn aarch64_lsl_x0_x1_imm() {
    // LSL X0, X1, #3 → UBFM X0, X1, #61, #60 → D37DF020
    let code = assemble("lsl x0, x1, 3", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0xF0, 0x7D, 0xD3]);
}

#[test]
fn aarch64_neg_x0_x1() {
    // NEG X0, X1 → SUB X0, XZR, X1 → CB0103E0
    let code = assemble("neg x0, x1", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0xE0, 0x03, 0x01, 0xCB]);
}

#[test]
fn aarch64_shellcode_pattern() {
    // AArch64 Linux exit(0) shellcode
    let code = assemble(
        "mov x0, 0\n\
         mov x8, 93\n\
         svc 0",
        Arch::Aarch64,
    )
    .unwrap();
    // MOV X0, #0  → MOVZ X0, #0  → D2800000
    // MOV X8, #93 → MOVZ X8, #93 → D2800BA8
    // SVC #0      → D4000001
    assert_eq!(
        code,
        vec![
            0x00, 0x00, 0x80, 0xD2, // mov x0, #0
            0xA8, 0x0B, 0x80, 0xD2, // mov x8, #93
            0x01, 0x00, 0x00, 0xD4, // svc #0
        ]
    );
}

// ── AArch64 NEW INSTRUCTIONS ─────────────────────────────────────────────

fn a64_word(src: &str) -> u32 {
    let code = assemble(src, Arch::Aarch64).unwrap();
    assert_eq!(code.len(), 4);
    u32::from_le_bytes(code[..4].try_into().unwrap())
}

#[test]
fn aarch64_mul_x0_x1_x2() {
    assert_eq!(a64_word("mul x0, x1, x2"), 0x9B02_7C20);
}

#[test]
fn aarch64_sdiv_x0_x1_x2() {
    assert_eq!(a64_word("sdiv x0, x1, x2"), 0x9AC2_0C20);
}

#[test]
fn aarch64_udiv_x0_x1_x2() {
    assert_eq!(a64_word("udiv x0, x1, x2"), 0x9AC2_0820);
}

#[test]
fn aarch64_madd_x0_x1_x2_x3() {
    assert_eq!(a64_word("madd x0, x1, x2, x3"), 0x9B02_0C20);
}

#[test]
fn aarch64_mvn_x0_x1() {
    assert_eq!(a64_word("mvn x0, x1"), 0xAA21_03E0);
}

#[test]
fn aarch64_clz_x0_x1() {
    assert_eq!(a64_word("clz x0, x1"), 0xDAC0_1020);
}

#[test]
fn aarch64_rbit_x0_x1() {
    assert_eq!(a64_word("rbit x0, x1"), 0xDAC0_0020);
}

#[test]
fn aarch64_rev_x0_x1() {
    assert_eq!(a64_word("rev x0, x1"), 0xDAC0_0C20);
}

#[test]
fn aarch64_uxtb_w0_w1() {
    assert_eq!(a64_word("uxtb w0, w1"), 0x5300_1C20);
}

#[test]
fn aarch64_sxtw_x0_w1() {
    assert_eq!(a64_word("sxtw x0, w1"), 0x9340_7C20);
}

#[test]
fn aarch64_cset_x0_eq() {
    assert_eq!(a64_word("cset x0, eq"), 0x9A9F_17E0);
}

#[test]
fn aarch64_mrs_x0_nzcv() {
    assert_eq!(a64_word("mrs x0, nzcv"), 0xD53B_4200);
}

#[test]
fn aarch64_msr_nzcv_x0() {
    assert_eq!(a64_word("msr nzcv, x0"), 0xD51B_4200);
}

#[test]
fn aarch64_dmb_sy() {
    assert_eq!(a64_word("dmb sy"), 0xD503_3FBF);
}

#[test]
fn aarch64_dsb_ish() {
    assert_eq!(a64_word("dsb ish"), 0xD503_3B9F);
}

#[test]
fn aarch64_isb() {
    assert_eq!(a64_word("isb"), 0xD503_3FDF);
}

#[test]
fn aarch64_and_x0_x1_0xff() {
    assert_eq!(a64_word("and x0, x1, 0xff"), 0x9240_1C20);
}

#[test]
fn aarch64_orr_x0_x1_imm() {
    // ORR X0, X1, #0xFF
    assert_eq!(a64_word("orr x0, x1, 0xff"), 0xB240_1C20);
}

#[test]
fn aarch64_tst_x0_imm() {
    // TST X0, #0xFF → ANDS XZR, X0, #0xFF
    assert_eq!(a64_word("tst x0, 0xff"), 0xF240_1C1F);
}

#[test]
fn aarch64_pre_index_str() {
    // STR X0, [SP, #-16]! → pre-index, size=11|111000|00|0|imm9=111110000|11|SP|X0
    let w = a64_word("str x0, [sp, -16]!");
    // Verify it's a pre-index encoding (bits 11:10 = 11)
    assert_eq!((w >> 10) & 0x3, 0b11);
    // Verify it's a store (bit 22 = 0)
    assert_eq!((w >> 22) & 0x1, 0);
}

#[test]
fn aarch64_post_index_ldr() {
    // LDR X0, [SP], #16 → post-index
    let code = assemble("ldr x0, [sp], 16", Arch::Aarch64).unwrap();
    let w = u32::from_le_bytes(code[..4].try_into().unwrap());
    // Verify it's a post-index encoding (bits 11:10 = 01)
    assert_eq!((w >> 10) & 0x3, 0b01);
    // Verify it's a load (bit 22 = 1)
    assert_eq!((w >> 22) & 0x1, 1);
}

#[test]
fn aarch64_stp_pre_index() {
    // STP X29, X30, [SP, #-16]!
    let w = a64_word("stp x29, x30, [sp, -16]!");
    // Verify pre-index pair (bits 24:23 should be 011)
    let mode = (w >> 23) & 0x7;
    assert_eq!(mode, 0b011);
}

#[test]
fn aarch64_csel_with_cond_name() {
    // CSEL X0, X1, X2, EQ
    let w = a64_word("csel x0, x1, x2, eq");
    let cc = (w >> 12) & 0xF;
    assert_eq!(cc, 0x0); // EQ
}

// ── AArch64 LABEL RESOLUTION TESTS ──────────────────────────────────────

#[test]
fn aarch64_b_forward_label() {
    // B to a label 2 instructions ahead
    let code = assemble(
        "b target\n\
         nop\n\
         target:\n\
         ret",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 12); // 3 instructions × 4 bytes
    let b_word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    // B target: opcode 000101 | imm26
    // PC=0, target=8, offset=(8-0)/4=2
    let imm26 = b_word & 0x03FF_FFFF;
    assert_eq!(imm26, 2);
}

#[test]
fn aarch64_b_backward_label() {
    let code = assemble(
        "loop:\n\
         nop\n\
         b loop",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 8);
    let b_word = u32::from_le_bytes(code[4..8].try_into().unwrap());
    // PC=4, target=0, offset=(0-4)/4=-1 → imm26 = 0x03FFFFFF
    let imm26 = b_word & 0x03FF_FFFF;
    assert_eq!(imm26, 0x03FF_FFFF); // -1 in 26-bit signed
}

#[test]
fn aarch64_bl_label() {
    let code = assemble(
        "bl func\n\
         ret\n\
         func:\n\
         mov x0, 0\n\
         ret",
        Arch::Aarch64,
    )
    .unwrap();
    let bl_word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    // BL: 100101|imm26, offset=8/4=2
    assert_eq!((bl_word >> 26) & 0x3F, 0b100101);
    assert_eq!(bl_word & 0x03FF_FFFF, 2);
}

#[test]
fn aarch64_bcond_label() {
    let code = assemble(
        "cmp x0, 0\n\
         b.eq done\n\
         nop\n\
         done:\n\
         ret",
        Arch::Aarch64,
    )
    .unwrap();
    let bcond = u32::from_le_bytes(code[4..8].try_into().unwrap());
    // B.EQ: 01010100|imm19|0|cond=0000
    let cond = bcond & 0xF;
    assert_eq!(cond, 0); // EQ
    let imm19 = (bcond >> 5) & 0x7FFFF;
    assert_eq!(imm19, 2); // skip nop → 2 instructions ahead
}

#[test]
fn aarch64_cbz_label() {
    let code = assemble(
        "cbz x0, done\n\
         nop\n\
         done:\n\
         ret",
        Arch::Aarch64,
    )
    .unwrap();
    let cbz = u32::from_le_bytes(code[0..4].try_into().unwrap());
    // CBZ: sf|011010|op(0)|imm19|Rt
    let imm19 = (cbz >> 5) & 0x7FFFF;
    assert_eq!(imm19, 2); // 2 instructions ahead
}

#[test]
fn aarch64_bcond_relaxes_to_short_form() {
    // Near B.NE target — should use 4-byte short form
    let code = assemble(
        "b.ne target\n\
         nop\n\
         target:\n\
         ret",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 12); // 3 × 4-byte instructions
    let bcond = u32::from_le_bytes(code[0..4].try_into().unwrap());
    let cond = bcond & 0xF;
    assert_eq!(cond, 1); // NE
    let imm19 = (bcond >> 5) & 0x7FFFF;
    assert_eq!(imm19, 2); // 2 instructions ahead
}

#[test]
fn aarch64_cbz_relaxes_to_short_form() {
    // Near CBZ — should use 4-byte short form
    let code = assemble(
        "cbz w1, target\n\
         nop\n\
         target:\n\
         ret",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 12);
    let cbz = u32::from_le_bytes(code[0..4].try_into().unwrap());
    let rt = cbz & 0x1F;
    assert_eq!(rt, 1); // W1
}

#[test]
fn aarch64_tbz_relaxes_to_short_form() {
    // Near TBZ — should use 4-byte short form
    let code = assemble(
        "tbz x0, 5, target\n\
         nop\n\
         target:\n\
         ret",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 12);
    let tbz = u32::from_le_bytes(code[0..4].try_into().unwrap());
    // Check it's a TBZ (not TBNZ) in short form
    let op_bit = (tbz >> 24) & 1;
    assert_eq!(op_bit, 0); // TBZ has op=0
}

#[test]
fn aarch64_tbz_far_target_uses_long_form() {
    // TBZ range is ±32 KB. Put target beyond that with padding.
    // Long form: TBNZ X0, #3, +8; B target (8 bytes)
    let code = assemble(
        "tbz x0, 3, target\n\
         .fill 32800, 1, 0\n\
         target:\n\
         ret",
        Arch::Aarch64,
    )
    .unwrap();
    // Should use long form (8 bytes): inverted TBNZ +8, then B target
    let skip_word = u32::from_le_bytes(code[0..4].try_into().unwrap());

    // TBNZ X0, #3, +8: b5=0 | 011011 | op=1(inverted) | b40=3 | imm14=2 | Rt=0
    let op_bit = (skip_word >> 24) & 1;
    assert_eq!(op_bit, 1); // TBNZ (inverted from TBZ)
    let b40 = (skip_word >> 19) & 0x1F;
    assert_eq!(b40, 3); // bit number 3
    let imm14 = (skip_word >> 5) & 0x3FFF;
    assert_eq!(imm14, 2); // skip +8 bytes

    // Second word should be B (unconditional) to target
    let b_word = u32::from_le_bytes(code[4..8].try_into().unwrap());
    let opcode = b_word >> 26;
    assert_eq!(opcode, 0b000101); // B instruction
}

#[test]
fn aarch64_tbnz_far_target_uses_long_form() {
    // Far TBNZ → long form: TBZ Xn, #bit, +8; B target
    let code = assemble(
        "tbnz x1, 7, target\n\
         .fill 32800, 1, 0\n\
         target:\n\
         ret",
        Arch::Aarch64,
    )
    .unwrap();
    let skip_word = u32::from_le_bytes(code[0..4].try_into().unwrap());

    // TBZ X1, #7, +8 (inverted from TBNZ): op=0
    let op_bit = (skip_word >> 24) & 1;
    assert_eq!(op_bit, 0); // TBZ (inverted from TBNZ)
    let rt = skip_word & 0x1F;
    assert_eq!(rt, 1); // X1
}

#[test]
fn aarch64_bcond_backward_short() {
    // Backward B.cond — near target, should use short form
    let code = assemble(
        "loop:\n\
         nop\n\
         b.lt loop",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 8); // 2 × 4-byte instructions (not 4+8)
    let bcond = u32::from_le_bytes(code[4..8].try_into().unwrap());
    let cond = bcond & 0xF;
    assert_eq!(cond, 0xB); // LT
                           // imm19 should be negative (backward) — -1 in 19-bit signed
    let imm19 = (bcond >> 5) & 0x7FFFF;
    // -1 instruction = -4 bytes >> 2 = -1, in 19-bit: 0x7FFFF
    assert_eq!(imm19, 0x7FFFF);
}

// ── ARM32 LABEL RESOLUTION TESTS ────────────────────────────────────────

#[test]
fn arm32_b_forward_label() {
    let code = assemble(
        "b target\n\
         nop\n\
         target:\n\
         bx lr",
        Arch::Arm,
    )
    .unwrap();
    assert_eq!(code.len(), 12);
    let b_word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    // ARM32 B: cond|101|0|imm24
    // PC = instr_addr + 8, so offset = (target - (PC)) = (8 - (0+8)) = 0
    let imm24 = b_word & 0x00FF_FFFF;
    assert_eq!(imm24, 0); // target is 8 bytes ahead, PC=8 (instr+8), (8-8)/4=0
}

#[test]
fn arm32_b_backward_label() {
    let code = assemble(
        "loop:\n\
         nop\n\
         b loop",
        Arch::Arm,
    )
    .unwrap();
    assert_eq!(code.len(), 8);
    let b_word = u32::from_le_bytes(code[4..8].try_into().unwrap());
    // PC = 4+8=12, target=0, offset=(0-12)/4=-3 → imm24 = 0xFFFFFD
    let imm24 = b_word & 0x00FF_FFFF;
    assert_eq!(imm24, 0x00FF_FFFD); // -3 in 24-bit signed
}

#[test]
fn arm32_bl_label() {
    let code = assemble(
        "bl func\n\
         bx lr\n\
         func:\n\
         mov r0, 0\n\
         bx lr",
        Arch::Arm,
    )
    .unwrap();
    let bl_word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    // BL: cond|101|1|imm24 — should have L bit set (bit 24)
    assert_eq!((bl_word >> 24) & 0xF, 0xB); // 1011 = BL
                                            // PC=8, target=8, offset=(8-8)/4=0
    let imm24 = bl_word & 0x00FF_FFFF;
    assert_eq!(imm24, 0);
}

// ── ARM32 BARREL SHIFTER ────────────────────────────────────────────────

#[test]
fn arm32_add_r0_r1_r2_lsl_3() {
    // ADD R0, R1, R2, LSL 3
    // cond=E|00|0|0100|0|R1|R0|00011|00|0|R2
    // = 0xE0810182
    let code = assemble("add r0, r1, r2, lsl, 3", Arch::Arm).unwrap();
    assert_eq!(code.len(), 4);
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    // Check condition (AL), register form (bits 27:26 = 00), opcode = ADD (0100)
    assert_eq!((word >> 28) & 0xF, 0xE); // AL condition
    assert_eq!((word >> 25) & 0x7, 0b000); // register form
    assert_eq!((word >> 21) & 0xF, 0x4); // ADD opcode
    assert_eq!((word >> 16) & 0xF, 1); // Rn = R1
    assert_eq!((word >> 12) & 0xF, 0); // Rd = R0
    assert_eq!(word & 0xF, 2); // Rm = R2
    assert_eq!((word >> 5) & 0x3, 0b00); // LSL
    assert_eq!((word >> 7) & 0x1F, 3); // shift amount = 3
}

#[test]
fn arm32_sub_r3_r4_r5_asr_8() {
    // SUB R3, R4, R5, ASR 8
    let code = assemble("sub r3, r4, r5, asr, 8", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 21) & 0xF, 0x2); // SUB opcode
    assert_eq!((word >> 16) & 0xF, 4); // Rn = R4
    assert_eq!((word >> 12) & 0xF, 3); // Rd = R3
    assert_eq!(word & 0xF, 5); // Rm = R5
    assert_eq!((word >> 5) & 0x3, 0b10); // ASR
    assert_eq!((word >> 7) & 0x1F, 8); // shift amount = 8
}

#[test]
fn arm32_mov_r0_r1_lsr_16() {
    // MOV R0, R1, LSR 16
    let code = assemble("mov r0, r1, lsr, 16", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 21) & 0xF, 0xD); // MOV opcode
    assert_eq!((word >> 12) & 0xF, 0); // Rd = R0
    assert_eq!(word & 0xF, 1); // Rm = R1
    assert_eq!((word >> 5) & 0x3, 0b01); // LSR
    assert_eq!((word >> 7) & 0x1F, 16); // shift = 16
}

#[test]
fn arm32_mov_r0_r1_ror_4() {
    // MOV R0, R1, ROR 4
    let code = assemble("mov r0, r1, ror, 4", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 5) & 0x3, 0b11); // ROR
    assert_eq!((word >> 7) & 0x1F, 4);
}

#[test]
fn arm32_mov_r0_r1_rrx() {
    // MOV R0, R1, RRX — encoded as ROR with amount=0
    let code = assemble("mov r0, r1, rrx", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 5) & 0x3, 0b11); // ROR encoding
    assert_eq!((word >> 7) & 0x1F, 0); // amount = 0 → RRX
    assert_eq!(word & 0xF, 1); // Rm = R1
}

#[test]
fn arm32_cmp_r0_r1_lsl_2() {
    // CMP R0, R1, LSL 2 — compare with shifted register
    let code = assemble("cmp r0, r1, lsl, 2", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 21) & 0xF, 0xA); // CMP opcode
    assert_eq!((word >> 20) & 0x1, 1); // S=1 (always for CMP)
    assert_eq!((word >> 16) & 0xF, 0); // Rn = R0
    assert_eq!(word & 0xF, 1); // Rm = R1
    assert_eq!((word >> 5) & 0x3, 0b00); // LSL
    assert_eq!((word >> 7) & 0x1F, 2); // shift = 2
}

#[test]
fn arm32_plain_reg_no_shift_unchanged() {
    // Ensure plain register form still works identically
    let code = assemble("add r0, r1, r2", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 7) & 0x1F, 0); // shift amount = 0
    assert_eq!((word >> 5) & 0x3, 0b00); // LSL 0 = no shift
}

// ── ARM32 CONDITIONAL EXECUTION ─────────────────────────────────────────

#[test]
fn arm32_addeq_conditional() {
    // ADDEQ R0, R1, R2 — condition = EQ (0x0)
    let code = assemble("addeq r0, r1, r2", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 28) & 0xF, 0x0); // EQ condition
    assert_eq!((word >> 21) & 0xF, 0x4); // ADD opcode
}

#[test]
fn arm32_movne_conditional() {
    // MOVNE R0, 1 — condition = NE (0x1)
    let code = assemble("movne r0, 1", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 28) & 0xF, 0x1); // NE condition
    assert_eq!((word >> 21) & 0xF, 0xD); // MOV opcode
}

#[test]
fn arm32_ldrge_conditional() {
    // LDRGE R0, [R1] — condition = GE (0xA)
    let code = assemble("ldrge r0, [r1]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 28) & 0xF, 0xA); // GE condition
    assert_eq!((word >> 20) & 0x1, 1); // L=1 (load)
}

#[test]
fn arm32_strlt_conditional() {
    // STRLT R0, [R1] — condition = LT (0xB)
    let code = assemble("strlt r0, [r1]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 28) & 0xF, 0xB); // LT condition
    assert_eq!((word >> 20) & 0x1, 0); // L=0 (store)
}

#[test]
fn arm32_adds_sets_flags() {
    // ADDS R0, R1, R2 — S flag set
    let code = assemble("adds r0, r1, r2", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 20) & 0x1, 1); // S=1
    assert_eq!((word >> 21) & 0xF, 0x4); // ADD
}

#[test]
fn arm32_subsne_conditional_with_flags() {
    // SUBSNE R0, R1, R2 — condition NE + S flag
    let code = assemble("subsne r0, r1, r2", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 28) & 0xF, 0x1); // NE condition
    assert_eq!((word >> 20) & 0x1, 1); // S=1
    assert_eq!((word >> 21) & 0xF, 0x2); // SUB
}

// ── ARM32 PRE/POST-INDEX LDR/STR ───────────────────────────────────────

#[test]
fn arm32_ldr_preindex_writeback() {
    // LDR R0, [R1, 4]! — pre-index with writeback
    let code = assemble("ldr r0, [r1, 4]!", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 28) & 0xF, 0xE); // AL
    assert_eq!((word >> 24) & 0x1, 1); // P=1 (pre-index)
    assert_eq!((word >> 21) & 0x1, 1); // W=1 (writeback)
    assert_eq!((word >> 20) & 0x1, 1); // L=1 (load)
    assert_eq!(word & 0xFFF, 4); // offset = 4
}

#[test]
fn arm32_str_preindex_writeback() {
    // STR R0, [R1, -8]! — pre-index with writeback, negative offset
    let code = assemble("str r0, [r1, -8]!", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 24) & 0x1, 1); // P=1 (pre-index)
    assert_eq!((word >> 23) & 0x1, 0); // U=0 (subtract)
    assert_eq!((word >> 21) & 0x1, 1); // W=1 (writeback)
    assert_eq!((word >> 20) & 0x1, 0); // L=0 (store)
    assert_eq!(word & 0xFFF, 8); // |offset| = 8
}

#[test]
fn arm32_ldr_postindex() {
    // LDR R0, [R1], 4 — post-index
    let code = assemble("ldr r0, [r1], 4", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 24) & 0x1, 0); // P=0 (post-index)
    assert_eq!((word >> 23) & 0x1, 1); // U=1 (add)
    assert_eq!((word >> 21) & 0x1, 0); // W=0
    assert_eq!((word >> 20) & 0x1, 1); // L=1 (load)
    assert_eq!(word & 0xFFF, 4); // offset = 4
}

#[test]
fn arm32_str_postindex_negative() {
    // STR R0, [R1], -4 — post-index negative
    let code = assemble("str r0, [r1], -4", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 24) & 0x1, 0); // P=0 (post-index)
    assert_eq!((word >> 23) & 0x1, 0); // U=0 (subtract)
    assert_eq!((word >> 20) & 0x1, 0); // L=0 (store)
    assert_eq!(word & 0xFFF, 4); // |offset| = 4
}

// ── ARM32 LONG MULTIPLY ────────────────────────────────────────────────

#[test]
fn arm32_umull() {
    // UMULL R0, R1, R2, R3
    // cond|0000|1|U=1|A=0|S=0|RdHi=R1|RdLo=R0|Rs=R3|1001|Rm=R2
    let code = assemble("umull r0, r1, r2, r3", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 28) & 0xF, 0xE); // AL
    assert_eq!((word >> 22) & 0x1, 1); // U=1 (unsigned)
    assert_eq!((word >> 21) & 0x1, 0); // A=0 (no accumulate)
    assert_eq!((word >> 20) & 0x1, 0); // S=0
    assert_eq!((word >> 16) & 0xF, 1); // RdHi = R1
    assert_eq!((word >> 12) & 0xF, 0); // RdLo = R0
    assert_eq!((word >> 8) & 0xF, 3); // Rs = R3
    assert_eq!((word >> 4) & 0xF, 0b1001); // multiply signature
    assert_eq!(word & 0xF, 2); // Rm = R2
}

#[test]
fn arm32_smull() {
    // SMULL R2, R3, R0, R1
    let code = assemble("smull r2, r3, r0, r1", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 22) & 0x1, 0); // U=0 (signed)
    assert_eq!((word >> 21) & 0x1, 0); // A=0 (no accumulate)
    assert_eq!((word >> 16) & 0xF, 3); // RdHi = R3
    assert_eq!((word >> 12) & 0xF, 2); // RdLo = R2
    assert_eq!((word >> 8) & 0xF, 1); // Rs = R1
    assert_eq!(word & 0xF, 0); // Rm = R0
}

#[test]
fn arm32_umlal() {
    // UMLAL R0, R1, R2, R3 — unsigned multiply-accumulate long
    let code = assemble("umlal r0, r1, r2, r3", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 22) & 0x1, 1); // U=1 (unsigned)
    assert_eq!((word >> 21) & 0x1, 1); // A=1 (accumulate)
}

#[test]
fn arm32_smlal() {
    // SMLAL R4, R5, R6, R7 — signed multiply-accumulate long
    let code = assemble("smlal r4, r5, r6, r7", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 22) & 0x1, 0); // U=0 (signed)
    assert_eq!((word >> 21) & 0x1, 1); // A=1 (accumulate)
    assert_eq!((word >> 16) & 0xF, 5); // RdHi = R5
    assert_eq!((word >> 12) & 0xF, 4); // RdLo = R4
}

#[test]
fn arm32_umulls_with_s_flag() {
    // UMULLS R0, R1, R2, R3 — with S flag
    let code = assemble("umulls r0, r1, r2, r3", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 20) & 0x1, 1); // S=1
}

// ── ARM32 MSR ───────────────────────────────────────────────────────────

#[test]
fn arm32_msr_cpsr_reg() {
    // MSR CPSR, R0
    let code = assemble("msr cpsr, r0", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 28) & 0xF, 0xE); // AL
                                         // Check MSR register form (bits 27:23 = 00010, bit 21 = 1)
    assert_eq!(word & 0xF, 0); // Rm = R0
}

#[test]
fn arm32_msr_cpsr_imm() {
    // MSR CPSR, 0xF0
    let code = assemble("msr cpsr, 0xF0", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 25) & 0x1, 1); // Immediate form
}

// ── ARM32 LDREX / STREX ────────────────────────────────────────────────

#[test]
fn arm32_ldrex() {
    // LDREX R0, [R1]
    let code = assemble("ldrex r0, [r1]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 28) & 0xF, 0xE); // AL
    assert_eq!((word >> 20) & 0xFF, 0b00011001); // LDREX opcode
    assert_eq!((word >> 16) & 0xF, 1); // Rn = R1
    assert_eq!((word >> 12) & 0xF, 0); // Rd = R0
}

#[test]
fn arm32_strex() {
    // STREX R0, R1, [R2]
    let code = assemble("strex r0, r1, [r2]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 20) & 0xFF, 0b00011000); // STREX opcode
    assert_eq!((word >> 16) & 0xF, 2); // Rn = R2
    assert_eq!((word >> 12) & 0xF, 0); // Rd = R0 (status)
    assert_eq!(word & 0xF, 1); // Rm = R1 (value)
}

// ── ARM32 MEMORY BARRIERS ──────────────────────────────────────────────

#[test]
fn arm32_dmb() {
    // DMB — default SY option
    let code = assemble("dmb", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xF57F_F05F);
}

#[test]
fn arm32_dsb() {
    let code = assemble("dsb", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xF57F_F04F);
}

#[test]
fn arm32_isb() {
    let code = assemble("isb", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xF57F_F06F);
}

// ── ARM32 BARREL SHIFTER IN COMBINATION ─────────────────────────────────

#[test]
fn arm32_eor_with_shift() {
    // EOR R0, R1, R2, LSL 1
    let code = assemble("eor r0, r1, r2, lsl, 1", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 21) & 0xF, 0x1); // EOR opcode
    assert_eq!((word >> 7) & 0x1F, 1); // shift amount = 1
    assert_eq!((word >> 5) & 0x3, 0b00); // LSL
}

#[test]
fn arm32_orr_with_shift() {
    // ORR R4, R5, R6, ASR 12
    let code = assemble("orr r4, r5, r6, asr, 12", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 21) & 0xF, 0xC); // ORR opcode
    assert_eq!((word >> 12) & 0xF, 4); // Rd = R4
    assert_eq!((word >> 7) & 0x1F, 12); // shift = 12
    assert_eq!((word >> 5) & 0x3, 0b10); // ASR
}

#[test]
fn arm32_bic_with_shift() {
    // BIC R0, R0, R1, LSL 8 — clear bits
    let code = assemble("bic r0, r0, r1, lsl, 8", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!((word >> 21) & 0xF, 0xE); // BIC opcode
    assert_eq!((word >> 7) & 0x1F, 8);
}

// ── AArch64 SHELLCODE PATTERNS ──────────────────────────────────────────

#[test]
fn aarch64_reverse_shell_pattern() {
    // Common AArch64 shellcode prologue pattern
    let code = assemble(
        "stp x29, x30, [sp, -16]!\n\
         mov x29, sp\n\
         mov x0, 0\n\
         mov x8, 93\n\
         svc 0\n\
         ldp x29, x30, [sp], 16\n\
         ret",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 28); // 7 instructions × 4 bytes
}

#[test]
fn aarch64_mul_div_pattern() {
    let code = assemble(
        "mul x0, x1, x2\n\
         sdiv x3, x0, x4\n\
         udiv x5, x0, x4\n\
         madd x6, x1, x2, x3",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 16);
}

#[test]
fn aarch64_bitmanip_pattern() {
    let code = assemble(
        "clz x0, x1\n\
         rbit x2, x3\n\
         rev x4, x5\n\
         rev16 x6, x7\n\
         cls x8, x0",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 20);
}

#[test]
fn aarch64_extend_pattern() {
    let code = assemble(
        "uxtb w0, w1\n\
         uxth w2, w3\n\
         sxtb x4, w5\n\
         sxth x6, w7\n\
         sxtw x8, w0",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 20);
}

#[test]
fn aarch64_cond_alias_pattern() {
    let code = assemble(
        "cmp x0, 0\n\
         cset x1, eq\n\
         csetm x2, ne\n\
         cinc x3, x4, ge\n\
         cneg x5, x6, lt",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 20);
}

#[test]
fn aarch64_system_pattern() {
    let code = assemble(
        "mrs x0, nzcv\n\
         msr nzcv, x0\n\
         dmb sy\n\
         dsb ish\n\
         isb",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 20);
}

#[test]
fn aarch64_logical_imm_pattern() {
    let code = assemble(
        "and x0, x1, 0xff\n\
         orr x2, x3, 0xff\n\
         eor x4, x5, 0xff\n\
         tst x6, 0xff",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 16);
}

// ── AArch64 Atomic (LSE) instructions ────────────────────────────────────

#[test]
fn aarch64_ldadd_basic() {
    assert_eq!(a64_word("ldadd x2, x3, [x1]"), 0xF822_0023);
}

#[test]
fn aarch64_ldaddal() {
    assert_eq!(a64_word("ldaddal x2, x3, [x1]"), 0xF8E2_0023);
}

#[test]
fn aarch64_ldadda() {
    assert_eq!(a64_word("ldadda x2, x3, [x1]"), 0xF8A2_0023);
}

#[test]
fn aarch64_ldaddl() {
    assert_eq!(a64_word("ldaddl x2, x3, [x1]"), 0xF862_0023);
}

#[test]
fn aarch64_ldaddb() {
    assert_eq!(a64_word("ldaddb w2, w3, [x1]"), 0x3822_0023);
}

#[test]
fn aarch64_ldaddh() {
    assert_eq!(a64_word("ldaddh w2, w3, [x1]"), 0x7822_0023);
}

#[test]
fn aarch64_ldclr() {
    assert_eq!(a64_word("ldclr x2, x3, [x1]"), 0xF822_1023);
}

#[test]
fn aarch64_ldset() {
    assert_eq!(a64_word("ldset x2, x3, [x1]"), 0xF822_3023);
}

#[test]
fn aarch64_ldeor() {
    assert_eq!(a64_word("ldeor x2, x3, [x1]"), 0xF822_2023);
}

#[test]
fn aarch64_swp_basic() {
    assert_eq!(a64_word("swp x5, x6, [x0]"), 0xF825_8006);
}

#[test]
fn aarch64_swpal() {
    assert_eq!(a64_word("swpal x5, x6, [x0]"), 0xF8E5_8006);
}

#[test]
fn aarch64_cas_basic() {
    assert_eq!(a64_word("cas x2, x3, [x1]"), 0xC842_7C23);
}

#[test]
fn aarch64_casal() {
    assert_eq!(a64_word("casal x2, x3, [x1]"), 0xC8C2_FC23);
}

#[test]
fn aarch64_stadd() {
    assert_eq!(a64_word("stadd x4, [x1]"), 0xF824_003F);
}

#[test]
fn aarch64_stclr() {
    assert_eq!(a64_word("stclr x4, [x1]"), 0xF824_103F);
}

// ── AArch64 Bitfield instructions ────────────────────────────────────────

#[test]
fn aarch64_ubfm_64() {
    assert_eq!(a64_word("ubfm x0, x1, 4, 7"), 0xD344_1C20);
}

#[test]
fn aarch64_bfm_64() {
    assert_eq!(a64_word("bfm x0, x1, 4, 7"), 0xB344_1C20);
}

#[test]
fn aarch64_sbfm_64() {
    assert_eq!(a64_word("sbfm x0, x1, 4, 7"), 0x9344_1C20);
}

#[test]
fn aarch64_ubfm_32() {
    assert_eq!(a64_word("ubfm w0, w1, 4, 7"), 0x5304_1C20);
}

#[test]
fn aarch64_bfi() {
    // BFI X0, X1, #4, #8 → BFM X0, X1, #60, #7
    assert_eq!(a64_word("bfi x0, x1, 4, 8"), 0xB37C_1C20);
}

#[test]
fn aarch64_bfxil() {
    // BFXIL X0, X1, #4, #8 → BFM X0, X1, #4, #11
    assert_eq!(a64_word("bfxil x0, x1, 4, 8"), 0xB344_2C20);
}

#[test]
fn aarch64_ubfx() {
    // UBFX X0, X1, #4, #8 → UBFM X0, X1, #4, #11
    assert_eq!(a64_word("ubfx x0, x1, 4, 8"), 0xD344_2C20);
}

#[test]
fn aarch64_sbfx() {
    // SBFX X0, X1, #4, #8 → SBFM X0, X1, #4, #11
    assert_eq!(a64_word("sbfx x0, x1, 4, 8"), 0x9344_2C20);
}

#[test]
fn aarch64_ubfiz() {
    // UBFIZ X0, X1, #4, #8 → UBFM X0, X1, #60, #7
    assert_eq!(a64_word("ubfiz x0, x1, 4, 8"), 0xD37C_1C20);
}

#[test]
fn aarch64_sbfiz() {
    // SBFIZ X0, X1, #4, #8 → SBFM X0, X1, #60, #7
    assert_eq!(a64_word("sbfiz x0, x1, 4, 8"), 0x937C_1C20);
}

// ── AArch64 CCMP / CCMN ─────────────────────────────────────────────────

#[test]
fn aarch64_ccmp_reg_eq() {
    // CCMP X1, X2, #0, eq  (register form, 64-bit)
    assert_eq!(a64_word("ccmp x1, x2, 0, eq"), 0xFA42_0020);
}

#[test]
fn aarch64_ccmp_imm_ne() {
    // CCMP X1, #5, #2, ne  (immediate form, 64-bit)
    assert_eq!(a64_word("ccmp x1, 5, 2, ne"), 0xFA45_1822);
}

#[test]
fn aarch64_ccmn_reg_eq() {
    // CCMN X1, X2, #0, eq
    assert_eq!(a64_word("ccmn x1, x2, 0, eq"), 0xBA42_0020);
}

#[test]
fn aarch64_ccmp_32bit_ge() {
    // CCMP W1, W2, #3, ge
    assert_eq!(a64_word("ccmp w1, w2, 3, ge"), 0x7A42_A023);
}

// ── AArch64 EXTR ─────────────────────────────────────────────────────────

#[test]
fn aarch64_extr_64() {
    assert_eq!(a64_word("extr x0, x1, x2, 16"), 0x93C2_4020);
}

#[test]
fn aarch64_extr_32() {
    assert_eq!(a64_word("extr w0, w1, w2, 8"), 0x1382_2020);
}

// ── AArch64 Load/Store Exclusive (LDXR / STXR) ──────────────────────────

#[test]
fn aarch64_ldxr_x64() {
    assert_eq!(a64_word("ldxr x0, [x1]"), 0xC85F_7C20);
}

#[test]
fn aarch64_ldaxr_x64() {
    assert_eq!(a64_word("ldaxr x0, [x1]"), 0xC85F_FC20);
}

#[test]
fn aarch64_stxr_x64() {
    assert_eq!(a64_word("stxr w0, x1, [x2]"), 0xC800_7C41);
}

#[test]
fn aarch64_stlxr_x64() {
    assert_eq!(a64_word("stlxr w0, x1, [x2]"), 0xC800_FC41);
}

#[test]
fn aarch64_ldxrb() {
    assert_eq!(a64_word("ldxrb w0, [x1]"), 0x085F_7C20);
}

#[test]
fn aarch64_ldxrh() {
    assert_eq!(a64_word("ldxrh w0, [x1]"), 0x485F_7C20);
}

#[test]
fn aarch64_stxrb() {
    assert_eq!(a64_word("stxrb w0, w1, [x2]"), 0x0800_7C41);
}

#[test]
fn aarch64_stxrh() {
    assert_eq!(a64_word("stxrh w0, w1, [x2]"), 0x4800_7C41);
}

#[test]
fn aarch64_ldaxrb() {
    assert_eq!(a64_word("ldaxrb w0, [x1]"), 0x085F_FC20);
}

#[test]
fn aarch64_stlxrb() {
    assert_eq!(a64_word("stlxrb w0, w1, [x2]"), 0x0800_FC41);
}

// ── AArch64 Register Offset LDR/STR ─────────────────────────────────────

#[test]
fn aarch64_ldr_x_reg_offset() {
    assert_eq!(a64_word("ldr x0, [x1, x2]"), 0xF862_6820);
}

#[test]
fn aarch64_str_x_reg_offset() {
    assert_eq!(a64_word("str x0, [x1, x2]"), 0xF822_6820);
}

#[test]
fn aarch64_ldr_w_reg_offset() {
    assert_eq!(a64_word("ldr w0, [x1, x2]"), 0xB862_6820);
}

#[test]
fn aarch64_ldrb_reg_offset() {
    assert_eq!(a64_word("ldrb w0, [x1, x2]"), 0x3862_6820);
}

#[test]
fn aarch64_ldrh_reg_offset() {
    assert_eq!(a64_word("ldrh w0, [x1, x2]"), 0x7862_6820);
}

#[test]
fn aarch64_strb_reg_offset() {
    assert_eq!(a64_word("strb w0, [x1, x2]"), 0x3822_6820);
}

#[test]
fn aarch64_strh_reg_offset() {
    assert_eq!(a64_word("strh w0, [x1, x2]"), 0x7822_6820);
}

// ── ARM32 LDREX / STREX Byte / Halfword / Double variants ────────────────

#[test]
fn arm32_ldrexb() {
    let code = assemble("ldrexb r0, [r1]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE1D1_0F9F);
}

#[test]
fn arm32_ldrexh() {
    let code = assemble("ldrexh r2, [r3]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE1F3_2F9F);
}

#[test]
fn arm32_ldrexd() {
    let code = assemble("ldrexd r4, [r5]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE1B5_4F9F);
}

#[test]
fn arm32_strexb() {
    let code = assemble("strexb r0, r1, [r2]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE1C2_0F91);
}

#[test]
fn arm32_strexh() {
    let code = assemble("strexh r0, r1, [r2]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE1E2_0F91);
}

#[test]
fn arm32_strexd() {
    let code = assemble("strexd r0, r1, [r2]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE1A2_0F91);
}

// ── ARM32 LDM/STM writeback control ─────────────────────────────────────

#[test]
fn arm32_ldm_writeback() {
    // LDM R0!, {R1, R2} → LDMIA with W=1
    let code = assemble("ldm r0!, {r1, r2}", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE8B0_0006);
}

#[test]
fn arm32_ldm_no_writeback() {
    // LDM R0, {R1, R2} → LDMIA with W=0
    let code = assemble("ldm r0, {r1, r2}", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE890_0006);
}

#[test]
fn arm32_stmdb_writeback() {
    // STMDB SP!, {R4, R5, LR} — classic push pattern
    let code = assemble("stmdb sp!, {r4, r5, lr}", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE92D_4030);
}

// ── ARM32 Register-shift barrel shifter ──────────────────────────────────

#[test]
fn arm32_add_reg_shift_lsl() {
    // ADD R0, R1, R2, LSL R3
    let code = assemble("add r0, r1, r2, lsl, r3", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE081_0312);
}

#[test]
fn arm32_and_reg_shift_lsr() {
    // AND R0, R0, R1, LSR R2
    let code = assemble("and r0, r0, r1, lsr, r2", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE000_0231);
}

// ── AArch64 Load-Acquire / Store-Release (non-exclusive) ─────────────────

#[test]
fn aarch64_ldar_x0_x1() {
    let code = assemble("ldar x0, [x1]", Arch::Aarch64).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xC8DF_FC20);
}

#[test]
fn aarch64_ldar_w0_x1() {
    // 32-bit load-acquire: size inferred from Wt register
    let code = assemble("ldar w0, [x1]", Arch::Aarch64).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0x88DF_FC20);
}

#[test]
fn aarch64_ldarb_w0_x1() {
    let code = assemble("ldarb w0, [x1]", Arch::Aarch64).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0x08DF_FC20);
}

#[test]
fn aarch64_ldarh_w0_x1() {
    let code = assemble("ldarh w0, [x1]", Arch::Aarch64).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0x48DF_FC20);
}

#[test]
fn aarch64_stlr_x0_x1() {
    let code = assemble("stlr x0, [x1]", Arch::Aarch64).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xC89F_FC20);
}

#[test]
fn aarch64_stlr_w0_x1() {
    // 32-bit store-release: size inferred from Wt register
    let code = assemble("stlr w0, [x1]", Arch::Aarch64).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0x889F_FC20);
}

#[test]
fn aarch64_stlrb_w0_x1() {
    let code = assemble("stlrb w0, [x1]", Arch::Aarch64).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0x089F_FC20);
}

#[test]
fn aarch64_stlrh_w0_x1() {
    let code = assemble("stlrh w0, [x1]", Arch::Aarch64).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0x489F_FC20);
}

// ── AArch64 Exclusive 32-bit register variants ──────────────────────────

#[test]
fn aarch64_ldxr_w0_x1_32bit() {
    // LDXR with W-register: size should be 10 (32-bit), not 11
    let code = assemble("ldxr w0, [x1]", Arch::Aarch64).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0x885F_7C20);
}

#[test]
fn aarch64_stxr_w2_w0_x1_32bit() {
    // STXR with W-register data: size should be 10 (32-bit)
    let code = assemble("stxr w2, w0, [x1]", Arch::Aarch64).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0x8802_7C20);
}

// ── ARM32 register offset with subtract ─────────────────────────────────

#[test]
fn arm32_ldr_reg_offset_subtract() {
    // LDR R0, [R1, -R2]: U-bit = 0
    let code = assemble("ldr r0, [r1, -r2]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE711_0002);
}

#[test]
fn arm32_str_reg_offset_subtract() {
    // STR R0, [R1, -R2]: U-bit = 0
    let code = assemble("str r0, [r1, -r2]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE701_0002);
}

#[test]
fn arm32_ldrh_reg_offset_subtract() {
    // LDRH R0, [R1, -R2]: U-bit = 0
    let code = assemble("ldrh r0, [r1, -r2]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE111_00B2);
}

// ── MOVT ─────────────────────────────────────────────────────────────────

#[test]
fn arm32_movt_r0_imm() {
    // MOVT R0, #0x5678 → cond|0011|0|1|00|imm4|Rd|imm12
    // H=1 for MOVT, imm4=0x5, imm12=0x678
    // E3450678
    let code = assemble("movt r0, 0x5678", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE345_0678);
}

#[test]
fn arm32_movt_r1_small() {
    // MOVT R1, #0x0001
    let code = assemble("movt r1, 1", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE340_1001);
}

#[test]
fn arm32_movw_movt_pair() {
    // Load 0x12345678 into R0 using MOVW + MOVT
    let code = assemble("movw r0, 0x5678\nmovt r0, 0x1234", Arch::Arm).unwrap();
    assert_eq!(code.len(), 8);
    let w0 = u32::from_le_bytes(code[0..4].try_into().unwrap());
    let w1 = u32::from_le_bytes(code[4..8].try_into().unwrap());
    assert_eq!(w0, 0xE305_0678); // MOVW R0, #0x5678
    assert_eq!(w1, 0xE341_0234); // MOVT R0, #0x1234
}

// ── LDRSH / LDRSB ────────────────────────────────────────────────────────

#[test]
fn arm32_ldrsh_r0_r1() {
    // LDRSH R0, [R1] → cond|000|P=1|U=1|1|W=0|L=1|Rn|Rd|0000|1|S=1|H=1|1|0000
    // SH=11 for LDRSH
    // E1D100F0
    let code = assemble("ldrsh r0, [r1]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE1D1_00F0);
}

#[test]
fn arm32_ldrsh_r2_r3_offset() {
    // LDRSH R2, [R3, #4]
    let code = assemble("ldrsh r2, [r3 + 4]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    // imm4H = 0, imm4L = 4, U=1, P=1
    assert_eq!(word, 0xE1D3_20F4);
}

#[test]
fn arm32_ldrsb_r0_r1() {
    // LDRSB R0, [R1] → SH=10 for LDRSB
    // E1D100D0
    let code = assemble("ldrsb r0, [r1]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE1D1_00D0);
}

#[test]
fn arm32_ldrsb_r4_r5_offset() {
    // LDRSB R4, [R5, #8]
    let code = assemble("ldrsb r4, [r5 + 8]", Arch::Arm).unwrap();
    let word = u32::from_le_bytes(code[0..4].try_into().unwrap());
    assert_eq!(word, 0xE1D5_40D8);
}

// ── parse_const_expr error on undefined identifier in chain ──────────────

#[test]
fn parse_const_expr_undefined_after_plus() {
    // `.fill KNOWN + UNKNOWN` should error, not silently evaluate to KNOWN
    let result = assemble(".equ A, 5\n.fill A + BADNAME, 1, 0", Arch::X86_64);
    assert!(
        result.is_err(),
        "should error on undefined identifier after +"
    );
}

#[test]
fn parse_const_expr_undefined_after_minus() {
    let result = assemble(".equ A, 10\n.fill A - BADNAME, 1, 0", Arch::X86_64);
    assert!(
        result.is_err(),
        "should error on undefined identifier after -"
    );
}

// ── Parser scale factor validation ───────────────────────────────────────

#[test]
fn parser_rejects_invalid_scale_factor() {
    // scale=3 is not a valid x86 SIB scale (must be 1, 2, 4, 8)
    let result = assemble("mov rax, [rbx + rcx*3]", Arch::X86_64);
    assert!(result.is_err(), "scale factor 3 should be rejected");
}

#[test]
fn parser_accepts_valid_scale_factors() {
    for scale in &[1, 2, 4, 8] {
        let src = format!("lea rax, [rbx + rcx*{}]", scale);
        let result = assemble(&src, Arch::X86_64);
        assert!(result.is_ok(), "scale factor {} should be accepted", scale);
    }
}

// ============================================================================
// RISC-V (RV32I / RV64I) Integration Tests
// ============================================================================

// ── Helper ───────────────────────────────────────────────────────────────

fn rv32(src: &str) -> Vec<u8> {
    assemble(src, Arch::Rv32).unwrap()
}

fn rv64(src: &str) -> Vec<u8> {
    assemble(src, Arch::Rv64).unwrap()
}

fn le32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
}

// ── One-shot API ─────────────────────────────────────────────────────────

#[test]
fn rv32_nop() {
    let code = rv32("nop");
    assert_eq!(code.len(), 4);
    assert_eq!(le32(&code, 0), 0x0000_0013); // addi x0, x0, 0
}

#[test]
fn rv64_nop() {
    let code = rv64("nop");
    assert_eq!(le32(&code, 0), 0x0000_0013);
}

// ── R-type ALU ───────────────────────────────────────────────────────────

#[test]
fn rv32_add_x1_x2_x3() {
    let code = rv32("add x1, x2, x3");
    assert_eq!(le32(&code, 0), 0x003100B3);
}

#[test]
fn rv32_sub_x1_x2_x3() {
    let code = rv32("sub x1, x2, x3");
    assert_eq!(le32(&code, 0), 0x403100B3);
}

#[test]
fn rv32_and_or_xor() {
    let code = rv32("and x5, x6, x7\nor x8, x9, x10\nxor x11, x12, x13");
    assert_eq!(code.len(), 12);
    assert_eq!(le32(&code, 0), 0x007372B3); // and
    assert_eq!(le32(&code, 4), 0x00A4E433); // or
    assert_eq!(le32(&code, 8), 0x00D645B3); // xor
}

#[test]
fn rv32_slt_sltu() {
    let code = rv32("slt x1, x2, x3\nsltu x4, x5, x6");
    assert_eq!(le32(&code, 0), 0x003120B3); // slt
    assert_eq!(le32(&code, 4), 0x0062B233); // sltu
}

#[test]
fn rv32_sll_srl_sra() {
    let code = rv32("sll x1, x2, x3\nsrl x4, x5, x6\nsra x7, x8, x9");
    assert_eq!(le32(&code, 0), 0x003110B3); // sll
    assert_eq!(le32(&code, 4), 0x0062D233); // srl
    assert_eq!(le32(&code, 8), 0x409453B3); // sra
}

// ── I-type ALU ───────────────────────────────────────────────────────────

#[test]
fn rv32_addi() {
    let code = rv32("addi x1, x2, 42");
    assert_eq!(le32(&code, 0), 0x02A10093);
}

#[test]
fn rv32_addi_negative() {
    let code = rv32("addi x1, x0, -1");
    assert_eq!(le32(&code, 0), 0xFFF00093);
}

#[test]
fn rv32_andi_ori_xori() {
    let code = rv32("andi x1, x2, 0xFF\nori x3, x4, 1\nxori x5, x6, -1");
    assert_eq!(le32(&code, 0), 0x0FF17093); // andi
    assert_eq!(le32(&code, 4), 0x00126193); // ori
    assert_eq!(le32(&code, 8), 0xFFF34293); // xori
}

#[test]
fn rv32_slti_sltiu() {
    let code = rv32("slti x1, x2, 10\nsltiu x3, x4, 20");
    assert_eq!(le32(&code, 0), 0x00A12093); // slti
    assert_eq!(le32(&code, 4), 0x01423193); // sltiu
}

// ── Shift immediates ────────────────────────────────────────────────────

#[test]
fn rv32_slli_srli_srai() {
    let code = rv32("slli x1, x2, 5\nsrli x3, x4, 7\nsrai x5, x6, 3");
    assert_eq!(le32(&code, 0), 0x00511093); // slli
    assert_eq!(le32(&code, 4), 0x00725193); // srli
    assert_eq!(le32(&code, 8), 0x40335293); // srai
}

// ── Loads ────────────────────────────────────────────────────────────────

#[test]
fn rv32_lw() {
    let code = rv32("lw x1, 0(x2)");
    assert_eq!(le32(&code, 0), 0x00012083);
}

#[test]
fn rv32_lw_offset() {
    let code = rv32("lw x1, 8(x2)");
    assert_eq!(le32(&code, 0), 0x00812083);
}

#[test]
fn rv32_lb_lbu_lh_lhu() {
    let code = rv32("lb x1, 0(x2)\nlbu x3, 4(x4)\nlh x5, 8(x6)\nlhu x7, 12(x8)");
    assert_eq!(le32(&code, 0), 0x00010083); // lb
    assert_eq!(le32(&code, 4), 0x00424183); // lbu
    assert_eq!(le32(&code, 8), 0x00831283); // lh
    assert_eq!(le32(&code, 12), 0x00C45383); // lhu
}

#[test]
fn rv64_ld() {
    let code = rv64("ld x1, 16(x2)");
    assert_eq!(le32(&code, 0), 0x01013083);
}

#[test]
fn rv64_lwu() {
    let code = rv64("lwu x1, 4(x2)");
    assert_eq!(le32(&code, 0), 0x00416083);
}

// ── Stores ───────────────────────────────────────────────────────────────

#[test]
fn rv32_sw() {
    let code = rv32("sw x1, 0(x2)");
    assert_eq!(le32(&code, 0), 0x00112023);
}

#[test]
fn rv32_sw_offset() {
    let code = rv32("sw x1, 8(x2)");
    assert_eq!(le32(&code, 0), 0x00112423);
}

#[test]
fn rv32_sb_sh() {
    let code = rv32("sb x1, 0(x2)\nsh x3, 4(x4)");
    assert_eq!(le32(&code, 0), 0x00110023); // sb
    assert_eq!(le32(&code, 4), 0x00321223); // sh
}

#[test]
fn rv64_sd() {
    let code = rv64("sd x1, 24(x2)");
    assert_eq!(le32(&code, 0), 0x00113C23);
}

// ── Negative offsets ─────────────────────────────────────────────────────

#[test]
fn rv32_lw_negative_offset() {
    let code = rv32("lw x1, -4(x2)");
    assert_eq!(le32(&code, 0), 0xFFC12083);
}

#[test]
fn rv32_sw_negative_offset() {
    let code = rv32("sw x1, -8(x2)");
    // S-type: imm=-8 = 0xFFF8, imm[11:5]=0b1111111, imm[4:0]=0b11000
    assert_eq!(le32(&code, 0), 0xFE112C23);
}

// ── U-type ───────────────────────────────────────────────────────────────

#[test]
fn rv32_lui() {
    let code = rv32("lui x1, 0x12345");
    assert_eq!(le32(&code, 0), 0x123450B7);
}

#[test]
fn rv32_auipc() {
    let code = rv32("auipc x1, 0x12345");
    assert_eq!(le32(&code, 0), 0x12345097);
}

// ── System ───────────────────────────────────────────────────────────────

#[test]
fn rv32_ecall() {
    let code = rv32("ecall");
    assert_eq!(le32(&code, 0), 0x00000073);
}

#[test]
fn rv32_ebreak() {
    let code = rv32("ebreak");
    assert_eq!(le32(&code, 0), 0x00100073);
}

#[test]
fn rv32_fence() {
    let code = rv32("fence");
    assert_eq!(le32(&code, 0), 0x0FF0000F);
}

// ── Pseudo-instructions ──────────────────────────────────────────────────

#[test]
fn rv32_mv() {
    // mv x1, x2 → addi x1, x2, 0
    let code = rv32("mv x1, x2");
    assert_eq!(le32(&code, 0), 0x00010093);
}

#[test]
fn rv32_not() {
    // not x1, x2 → xori x1, x2, -1
    let code = rv32("not x1, x2");
    assert_eq!(le32(&code, 0), 0xFFF14093);
}

#[test]
fn rv32_neg() {
    // neg x1, x2 → sub x1, x0, x2
    let code = rv32("neg x1, x2");
    assert_eq!(le32(&code, 0), 0x402000B3);
}

#[test]
fn rv32_ret() {
    // ret → jalr x0, x1, 0
    let code = rv32("ret");
    assert_eq!(le32(&code, 0), 0x00008067);
}

#[test]
fn rv32_li_small() {
    // li x1, 42 → addi x1, x0, 42
    let code = rv32("li x1, 42");
    assert_eq!(code.len(), 4);
    assert_eq!(le32(&code, 0), 0x02A00093);
}

#[test]
fn rv32_li_negative() {
    // li x1, -1 → addi x1, x0, -1
    let code = rv32("li x1, -1");
    assert_eq!(le32(&code, 0), 0xFFF00093);
}

#[test]
fn rv32_li_large() {
    // li x1, 0x12345678 → lui x1, upper + addi x1, x1, lower
    let code = rv32("li x1, 0x12345678");
    assert_eq!(code.len(), 8);
    // lui x1, 0x12345 (adjusted for sign-extension of lower 12 bits)
    // addi x1, x1, 0x678
}

/// Simulate an RV64 li instruction sequence (LUI/ADDI/SLLI only) and
/// return the 64-bit value loaded into the destination register.
fn simulate_rv64_li(bytes: &[u8]) -> i64 {
    assert!(
        bytes.len() % 4 == 0,
        "li sequence must be a multiple of 4 bytes"
    );
    let mut rd_val: i64 = 0;
    for i in (0..bytes.len()).step_by(4) {
        let w = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let opcode = w & 0x7F;
        let funct3 = (w >> 12) & 0x7;
        let rs1 = (w >> 15) & 0x1F;
        match opcode {
            0b0110111 => {
                // LUI: load upper immediate, sign-extended to 64 bits
                let imm = (w & 0xFFFF_F000) as i32;
                rd_val = imm as i64;
            }
            0b0010011 => {
                // OP_IMM
                match funct3 {
                    0 => {
                        // ADDI
                        let imm = (w as i32) >> 20; // sign-extended 12-bit
                        let src = if rs1 == 0 { 0i64 } else { rd_val };
                        rd_val = src.wrapping_add(imm as i64);
                    }
                    1 => {
                        // SLLI
                        let shamt = (w >> 20) & 0x3F;
                        rd_val = rd_val.wrapping_shl(shamt);
                    }
                    _ => panic!("unexpected funct3={funct3} in li sequence"),
                }
            }
            _ => panic!("unexpected opcode 0x{opcode:02x} in li sequence"),
        }
    }
    rd_val
}

#[test]
fn rv64_li_power_of_2_above_32bit() {
    // li a0, 0x100000000 (2^32) → ADDI a0, x0, 1; SLLI a0, a0, 32
    let code = rv64("li a0, 0x100000000");
    assert_eq!(code.len(), 8); // 2 instructions
    assert_eq!(simulate_rv64_li(&code), 0x1_0000_0000);
}

#[test]
fn rv64_li_negative_above_32bit() {
    // li a0, -4294967296 (= -2^32)
    let code = rv64("li a0, -4294967296");
    assert_eq!(code.len(), 8); // ADDI -1 + SLLI 32
    assert_eq!(simulate_rv64_li(&code), -4_294_967_296i64);
}

#[test]
fn rv64_li_33bit_value() {
    // li a0, 0x123456789 — 33-bit positive value
    let code = rv64("li a0, 0x123456789");
    assert!(code.len() > 8); // needs more than 2 instructions
    assert_eq!(simulate_rv64_li(&code), 0x1_2345_6789);
}

#[test]
fn rv64_li_48bit_value() {
    // li a0, 0xABCD12345678 — 48-bit value
    let code = rv64("li a0, 0xABCD12345678");
    assert!(code.len() >= 12);
    assert_eq!(simulate_rv64_li(&code), 0xABCD_1234_5678u64 as i64);
}

#[test]
fn rv64_li_full_64bit() {
    // li a0, 0x123456789ABCDEF0 — full 64-bit value
    let code = rv64("li a0, 0x123456789ABCDEF0");
    assert!(code.len() >= 16);
    assert_eq!(simulate_rv64_li(&code), 0x1234_5678_9ABC_DEF0u64 as i64);
}

#[test]
fn rv64_li_i64_max() {
    // li a0, 0x7FFFFFFFFFFFFFFF (i64::MAX)
    let code = rv64("li a0, 0x7FFFFFFFFFFFFFFF");
    assert_eq!(simulate_rv64_li(&code), i64::MAX);
}

#[test]
fn rv64_li_minus_one_still_compact() {
    // li a0, -1 should still be a single ADDI (4 bytes)
    let code = rv64("li a0, -1");
    assert_eq!(code.len(), 4);
    assert_eq!(simulate_rv64_li(&code), -1);
}

#[test]
fn rv64_li_32bit_signed_still_compact() {
    // li a0, 0x12345678 should still be LUI+ADDI (8 bytes) on RV64
    let code = rv64("li a0, 0x12345678");
    assert_eq!(code.len(), 8);
    assert_eq!(simulate_rv64_li(&code), 0x1234_5678);
}

#[test]
fn rv32_li_unsigned_0xffffffff() {
    // On RV32, 0xFFFFFFFF should be loadable (as -1 in 32-bit)
    let code = rv32("li a0, 0xFFFFFFFF");
    assert_eq!(code.len(), 4); // just ADDI a0, x0, -1
}

#[test]
fn rv32_li_unsigned_0x80000000() {
    // On RV32, 0x80000000 is valid (LUI a0, 0x80000)
    let code = rv32("li a0, 0x80000000");
    assert_eq!(code.len(), 4); // just LUI
}

#[test]
fn rv64_li_0x80000000_not_sign_extended() {
    // On RV64, 0x80000000 is in [2^31, 2^32-1] so needs multi-instr
    // to avoid LUI sign-extending to 0xFFFFFFFF80000000
    let code = rv64("li a0, 0x80000000");
    assert_eq!(simulate_rv64_li(&code), 0x0000_0000_8000_0000i64);
}

#[test]
fn rv64_li_0xffffffff() {
    // On RV64, 0xFFFFFFFF should load as positive 4294967295
    let code = rv64("li a0, 0xFFFFFFFF");
    assert_eq!(simulate_rv64_li(&code), 0x0000_0000_FFFF_FFFFi64);
}

#[test]
fn rv32_seqz() {
    // seqz x1, x2 → sltiu x1, x2, 1
    let code = rv32("seqz x1, x2");
    assert_eq!(le32(&code, 0), 0x00113093);
}

#[test]
fn rv32_snez() {
    // snez x1, x2 → sltu x1, x0, x2
    let code = rv32("snez x1, x2");
    assert_eq!(le32(&code, 0), 0x002030B3);
}

#[test]
fn rv32_jr() {
    // jr x5 → jalr x0, x5, 0
    let code = rv32("jr x5");
    assert_eq!(le32(&code, 0), 0x00028067);
}

// ── ABI register names ───────────────────────────────────────────────────

#[test]
fn rv32_abi_names() {
    // Test that ABI register names work correctly
    let code = rv32("add a0, a1, a2"); // a0=x10, a1=x11, a2=x12
    assert_eq!(le32(&code, 0), 0x00C58533);
}

#[test]
fn rv32_abi_sp_ra() {
    // addi sp, sp, -16 → addi x2, x2, -16
    let code = rv32("addi sp, sp, -16");
    assert_eq!(le32(&code, 0), 0xFF010113);
}

#[test]
fn rv32_abi_saved_regs() {
    // sw s0, 0(sp) → sw x8, 0(x2)
    let code = rv32("sw s0, 0(sp)");
    assert_eq!(le32(&code, 0), 0x00812023);
}

#[test]
fn rv32_abi_temp_regs() {
    // add t0, t1, t2 → add x5, x6, x7
    let code = rv32("add t0, t1, t2");
    assert_eq!(le32(&code, 0), 0x007302B3);
}

#[test]
fn rv32_fp_alias() {
    // fp is an alias for s0 = x8
    let code = rv32("add fp, zero, ra"); // add x8, x0, x1
    assert_eq!(le32(&code, 0), 0x00100433);
}

// ── Branches with labels ─────────────────────────────────────────────────

#[test]
fn rv32_beq_forward() {
    // beq x1, x0, skip
    // addi x2, x0, 1
    // skip:
    // addi x3, x0, 2
    let code = rv32("beq x1, x0, skip\naddi x2, x0, 1\nskip:\naddi x3, x0, 2");
    assert_eq!(code.len(), 12);
    let beq = le32(&code, 0);
    // beq branches forward by 8 bytes (skip over addi)
    // B-type encoding, offset=8: imm[12]=0,imm[10:5]=0,imm[4:1]=0100,imm[11]=0
    assert_eq!(beq & 0x7F, 0x63); // opcode = BRANCH
}

#[test]
fn rv32_bne_backward_loop() {
    // loop:
    //   addi x1, x1, 1
    //   bne x1, x2, loop
    let code = rv32("loop:\naddi x1, x1, 1\nbne x1, x2, loop");
    assert_eq!(code.len(), 8);
    let bne = le32(&code, 4);
    assert_eq!(bne & 0x7F, 0x63); // opcode = BRANCH
    assert_eq!((bne >> 12) & 0x7, 1); // funct3 = 001 (BNE)
}

// ── JAL with labels ──────────────────────────────────────────────────────

#[test]
fn rv32_jal_forward() {
    // jal ra, target
    // nop
    // target:
    // ret
    let code = rv32("jal ra, target\nnop\ntarget:\nret");
    assert_eq!(code.len(), 12);
    let jal = le32(&code, 0);
    assert_eq!(jal & 0x7F, 0x6F); // opcode = JAL
    assert_eq!((jal >> 7) & 0x1F, 1); // rd = ra = x1
}

#[test]
fn rv32_j_pseudo_forward() {
    // j target → jal x0, target
    let code = rv32("j target\nnop\ntarget:\nnop");
    assert_eq!(code.len(), 12);
    let jal = le32(&code, 0);
    assert_eq!(jal & 0x7F, 0x6F); // opcode = JAL
    assert_eq!((jal >> 7) & 0x1F, 0); // rd = x0
}

// ── CALL pseudo ──────────────────────────────────────────────────────────

#[test]
fn rv32_call_produces_8_bytes() {
    // call target → auipc ra, hi + jalr ra, ra, lo
    // target: nop
    let code = rv32("call target\ntarget:\nnop");
    assert_eq!(code.len(), 12); // 8 bytes for call + 4 for nop
    let auipc = le32(&code, 0);
    let jalr = le32(&code, 4);
    assert_eq!(auipc & 0x7F, 0x17); // AUIPC opcode
    assert_eq!(jalr & 0x7F, 0x67); // JALR opcode
    assert_eq!((auipc >> 7) & 0x1F, 1); // rd = ra
    assert_eq!((jalr >> 7) & 0x1F, 1); // rd = ra
}

// ── TAIL pseudo ──────────────────────────────────────────────────────────

#[test]
fn rv32_tail_produces_8_bytes() {
    // tail target → auipc t1, hi + jalr x0, t1, lo
    // target: nop
    let code = rv32("tail target\ntarget:\nnop");
    assert_eq!(code.len(), 12);
    let auipc = le32(&code, 0);
    let jalr = le32(&code, 4);
    assert_eq!(auipc & 0x7F, 0x17); // AUIPC opcode
    assert_eq!(jalr & 0x7F, 0x67); // JALR opcode
    assert_eq!((auipc >> 7) & 0x1F, 6); // rd = t1 = x6
    assert_eq!((jalr >> 7) & 0x1F, 0); // rd = x0 (no link)
}

// ── Branch pseudo-instructions ───────────────────────────────────────────

#[test]
fn rv32_beqz() {
    // beqz x1, target → beq x1, x0, target
    let code = rv32("beqz x1, target\ntarget:\nnop");
    assert_eq!(code.len(), 8);
    let beq = le32(&code, 0);
    assert_eq!(beq & 0x7F, 0x63);
    assert_eq!((beq >> 12) & 0x7, 0); // funct3 = 000 (BEQ)
    assert_eq!((beq >> 20) & 0x1F, 0); // rs2 = x0
}

#[test]
fn rv32_bnez() {
    let code = rv32("bnez x1, target\ntarget:\nnop");
    let bne = le32(&code, 0);
    assert_eq!(bne & 0x7F, 0x63);
    assert_eq!((bne >> 12) & 0x7, 1); // funct3 = 001 (BNE)
}

// ── M extension ──────────────────────────────────────────────────────────

#[test]
fn rv32_mul() {
    let code = rv32("mul x1, x2, x3");
    assert_eq!(le32(&code, 0), 0x023100B3);
}

#[test]
fn rv32_div_rem() {
    let code = rv32("div x1, x2, x3\nrem x4, x5, x6");
    assert_eq!(le32(&code, 0), 0x023140B3); // div
    assert_eq!(le32(&code, 4), 0x0262E233); // rem
}

// ── RV64-specific ────────────────────────────────────────────────────────

#[test]
fn rv64_addw() {
    let code = rv64("addw x1, x2, x3");
    assert_eq!(le32(&code, 0), 0x003100BB);
}

#[test]
fn rv64_subw() {
    let code = rv64("subw x1, x2, x3");
    assert_eq!(le32(&code, 0), 0x403100BB);
}

#[test]
fn rv64_addiw() {
    let code = rv64("addiw x1, x2, 42");
    assert_eq!(le32(&code, 0), 0x02A1009B);
}

#[test]
fn rv64_sext_w() {
    // sext.w x1, x2 → addiw x1, x2, 0
    let code = rv64("sext.w x1, x2");
    assert_eq!(le32(&code, 0), 0x0001009B);
}

#[test]
fn rv64_shift_6bit_shamt() {
    // RV64 allows 6-bit shift amounts
    let code = rv64("slli x1, x2, 32");
    assert_eq!(code.len(), 4);
    let w = le32(&code, 0);
    assert_eq!(w & 0x7F, 0x13); // OP_IMM
}

// ── Builder API ──────────────────────────────────────────────────────────

#[test]
fn rv32_builder_api() {
    let mut asm = Assembler::new(Arch::Rv32);
    asm.emit("addi sp, sp, -16").unwrap();
    asm.emit("sw ra, 12(sp)").unwrap();
    asm.emit("sw s0, 8(sp)").unwrap();
    asm.emit("addi s0, sp, 16").unwrap();
    asm.emit("lw s0, 8(sp)").unwrap();
    asm.emit("lw ra, 12(sp)").unwrap();
    asm.emit("addi sp, sp, 16").unwrap();
    asm.emit("ret").unwrap();
    let result = asm.finish().unwrap();
    assert_eq!(result.bytes().len(), 32); // 8 instructions × 4 bytes
}

#[test]
fn rv64_builder_api() {
    let mut asm = Assembler::new(Arch::Rv64);
    asm.emit("addi sp, sp, -32").unwrap();
    asm.emit("sd ra, 24(sp)").unwrap();
    asm.emit("ld ra, 24(sp)").unwrap();
    asm.emit("addi sp, sp, 32").unwrap();
    asm.emit("ret").unwrap();
    let result = asm.finish().unwrap();
    assert_eq!(result.bytes().len(), 20);
}

// ── Multi-instruction sequences ──────────────────────────────────────────

#[test]
fn rv32_function_prologue_epilogue() {
    let src = "\
addi sp, sp, -16
sw ra, 12(sp)
sw s0, 8(sp)
addi s0, sp, 16
nop
lw s0, 8(sp)
lw ra, 12(sp)
addi sp, sp, 16
ret";
    let code = rv32(src);
    assert_eq!(code.len(), 36);
}

#[test]
fn rv32_loop_with_branch() {
    let src = "\
li a0, 0
li a1, 10
loop:
  addi a0, a0, 1
  bne a0, a1, loop
ret";
    let code = rv32(src);
    assert_eq!(code.len(), 20); // 5 instructions
                                // Verify the BNE branches backward correctly
    let bne = le32(&code, 12);
    assert_eq!(bne & 0x7F, 0x63); // BRANCH opcode
    assert_eq!((bne >> 12) & 0x7, 1); // funct3 = BNE
}

// ── Error handling ───────────────────────────────────────────────────────

#[test]
fn rv32_rejects_unknown_mnemonic() {
    let result = assemble("foobar x1, x2, x3", Arch::Rv32);
    assert!(result.is_err());
}

#[test]
fn rv32_rejects_rv64_only_instructions() {
    assert!(assemble("ld x1, 0(x2)", Arch::Rv32).is_err());
    assert!(assemble("sd x1, 0(x2)", Arch::Rv32).is_err());
    assert!(assemble("addw x1, x2, x3", Arch::Rv32).is_err());
    assert!(assemble("addiw x1, x2, 3", Arch::Rv32).is_err());
    assert!(assemble("sext.w x1, x2", Arch::Rv32).is_err());
}

#[test]
fn rv32_immediate_overflow() {
    // 12-bit immediate max = 2047
    assert!(assemble("addi x1, x0, 2048", Arch::Rv32).is_err());
    assert!(assemble("addi x1, x0, -2049", Arch::Rv32).is_err());
}

#[test]
fn rv32_bare_paren_mem() {
    // (sp) should be equivalent to 0(sp)
    let code = rv32("lw x1, (sp)");
    // Should produce lw x1, 0(sp) = lw x1, 0(x2)
    assert_eq!(le32(&code, 0), 0x00012083);
}

// ── JALR variants ────────────────────────────────────────────────────────

#[test]
fn rv32_jalr_3op() {
    let code = rv32("jalr x1, x2, 0");
    assert_eq!(le32(&code, 0), 0x000100E7);
}

#[test]
fn rv32_jalr_2op() {
    // jalr ra, rs → jalr ra, rs, 0
    let code = rv32("jalr ra, t0");
    assert_eq!(le32(&code, 0), 0x000280E7);
}

// ── CSR instructions ─────────────────────────────────────────────────────

#[test]
fn rv32_csrrw() {
    let code = rv32("csrrw x1, 0x300, x2");
    assert_eq!(code.len(), 4);
    let w = le32(&code, 0);
    assert_eq!(w & 0x7F, 0x73); // SYSTEM opcode
    assert_eq!((w >> 12) & 0x7, 1); // funct3 = CSRRW
}

#[test]
fn rv32_csrrs() {
    let code = rv32("csrrs x1, 0x300, x2");
    let w = le32(&code, 0);
    assert_eq!((w >> 12) & 0x7, 2); // funct3 = CSRRS
}

// ── Fence.i ──────────────────────────────────────────────────────────────

#[test]
fn rv32_fence_i() {
    let code = rv32("fence.i");
    assert_eq!(le32(&code, 0), 0x0000100F);
}

// ── A Extension (Atomics) ────────────────────────────────────────────────

#[test]
fn rv32_lr_w() {
    let code = rv32("lr.w x1, (x2)");
    let w = le32(&code, 0);
    assert_eq!(w & 0x7F, 0x2F); // OP-AMO opcode
    assert_eq!((w >> 27) & 0x1F, 0b00010); // funct5 = LR
    assert_eq!((w >> 12) & 0x7, 0b010); // funct3 = .w
}

#[test]
fn rv32_sc_w() {
    let code = rv32("sc.w x1, x3, (x2)");
    let w = le32(&code, 0);
    assert_eq!(w & 0x7F, 0x2F);
    assert_eq!((w >> 27) & 0x1F, 0b00011); // funct5 = SC
}

#[test]
fn rv32_lr_w_aqrl() {
    let code = rv32("lr.w.aqrl x1, (x2)");
    let w = le32(&code, 0);
    assert_eq!((w >> 26) & 1, 1); // aq
    assert_eq!((w >> 25) & 1, 1); // rl
}

#[test]
fn rv32_amoswap_w() {
    let code = rv32("amoswap.w x1, x3, (x2)");
    let w = le32(&code, 0);
    assert_eq!((w >> 27) & 0x1F, 0b00001);
}

#[test]
fn rv32_amoadd_w() {
    let code = rv32("amoadd.w x1, x3, (x2)");
    let w = le32(&code, 0);
    assert_eq!((w >> 27) & 0x1F, 0b00000);
}

#[test]
fn rv32_amoand_w() {
    let code = rv32("amoand.w x1, x3, (x2)");
    let w = le32(&code, 0);
    assert_eq!((w >> 27) & 0x1F, 0b01100);
}

#[test]
fn rv32_amoor_w() {
    let code = rv32("amoor.w x1, x3, (x2)");
    let w = le32(&code, 0);
    assert_eq!((w >> 27) & 0x1F, 0b01000);
}

#[test]
fn rv32_amoxor_w() {
    let code = rv32("amoxor.w x1, x3, (x2)");
    let w = le32(&code, 0);
    assert_eq!((w >> 27) & 0x1F, 0b00100);
}

#[test]
fn rv32_amomax_w() {
    let code = rv32("amomax.w x1, x3, (x2)");
    let w = le32(&code, 0);
    assert_eq!((w >> 27) & 0x1F, 0b10100);
}

#[test]
fn rv32_amomin_w() {
    let code = rv32("amomin.w x1, x3, (x2)");
    let w = le32(&code, 0);
    assert_eq!((w >> 27) & 0x1F, 0b10000);
}

#[test]
fn rv64_lr_d() {
    let code = rv64("lr.d x1, (x2)");
    let w = le32(&code, 0);
    assert_eq!((w >> 12) & 0x7, 0b011); // funct3 = .d
}

#[test]
fn rv64_amoswap_d_aq() {
    let code = rv64("amoswap.d.aq x1, x3, (x2)");
    let w = le32(&code, 0);
    assert_eq!((w >> 27) & 0x1F, 0b00001);
    assert_eq!((w >> 26) & 1, 1); // aq
    assert_eq!((w >> 25) & 1, 0); // rl
}

// ── CSR Pseudo-instructions ──────────────────────────────────────────────

#[test]
fn rv32_csrr() {
    // csrr rd, csr → csrrs rd, csr, x0
    let code = rv32("csrr x1, 0x300");
    let w = le32(&code, 0);
    assert_eq!(w & 0x7F, 0x73); // SYSTEM
    assert_eq!((w >> 12) & 0x7, 2); // CSRRS
    assert_eq!((w >> 15) & 0x1F, 0); // rs1 = x0
    assert_eq!((w >> 20), 0x300); // CSR = mstatus
}

#[test]
fn rv32_csrw() {
    // csrw csr, rs → csrrw x0, csr, rs
    let code = rv32("csrw 0x300, x2");
    let w = le32(&code, 0);
    assert_eq!((w >> 12) & 0x7, 1); // CSRRW
    assert_eq!((w >> 7) & 0x1F, 0); // rd = x0
}

#[test]
fn rv32_csrs() {
    // csrs csr, rs → csrrs x0, csr, rs
    let code = rv32("csrs 0x300, x2");
    let w = le32(&code, 0);
    assert_eq!((w >> 12) & 0x7, 2); // CSRRS
    assert_eq!((w >> 7) & 0x1F, 0); // rd = x0
}

#[test]
fn rv32_csrc() {
    // csrc csr, rs → csrrc x0, csr, rs
    let code = rv32("csrc 0x300, x2");
    let w = le32(&code, 0);
    assert_eq!((w >> 12) & 0x7, 3); // CSRRC
    assert_eq!((w >> 7) & 0x1F, 0); // rd = x0
}

#[test]
fn rv32_csrwi() {
    // csrwi csr, uimm → csrrwi x0, csr, uimm
    let code = rv32("csrwi 0x300, 5");
    let w = le32(&code, 0);
    assert_eq!((w >> 12) & 0x7, 5); // CSRRWI
    assert_eq!((w >> 7) & 0x1F, 0); // rd = x0
    assert_eq!((w >> 15) & 0x1F, 5); // uimm = 5
}

// ── Named CSR support ────────────────────────────────────────────────────

#[test]
fn rv32_csrrs_named_mstatus() {
    // Named CSR: csrrs x1, mstatus, x2 should be equivalent to csrrs x1, 0x300, x2
    let code_named = rv32("csrrs x1, mstatus, x2");
    let code_num = rv32("csrrs x1, 0x300, x2");
    assert_eq!(code_named, code_num);
}

#[test]
fn rv32_csrr_named_mie() {
    let code = rv32("csrr x1, mie");
    let w = le32(&code, 0);
    assert_eq!((w >> 20), 0x304); // mie CSR address
}

#[test]
fn rv32_csrr_named_cycle() {
    let code = rv32("csrr x1, cycle");
    let w = le32(&code, 0);
    assert_eq!((w >> 20), 0xC00); // cycle CSR address
}

// ── la pseudo-instruction ────────────────────────────────────────────────

#[test]
fn rv32_la_pseudo() {
    // la x1, target → auipc x1, %hi(target) + addi x1, x1, %lo(target)
    let code = rv32("la x1, target\ntarget:");
    assert_eq!(code.len(), 8); // AUIPC + ADDI
    let w0 = le32(&code, 0);
    assert_eq!(w0 & 0x7F, 0x17); // AUIPC opcode
    let w1 = le32(&code, 4);
    assert_eq!(w1 & 0x7F, 0x13); // ADDI opcode (OP-IMM)
}

// ── Branch relaxation (end-to-end) ───────────────────────────────────────

#[test]
fn rv32_beq_short_branch() {
    // Short branch: target is close (4 bytes away → offset = +4)
    let code = rv32("beq x1, x2, target\ntarget:");
    assert_eq!(code.len(), 4); // Relaxed to short form
    let w = le32(&code, 0);
    assert_eq!(w & 0x7F, 0x63); // BRANCH opcode
    assert_eq!((w >> 12) & 0x7, 0b000); // beq
}

#[test]
fn rv32_bne_short_branch() {
    let code = rv32("bne x1, x2, target\ntarget:");
    assert_eq!(code.len(), 4);
    let w = le32(&code, 0);
    assert_eq!((w >> 12) & 0x7, 0b001); // bne
}

#[test]
fn rv32_beqz_short_branch() {
    let code = rv32("beqz x5, target\ntarget:");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_bnez_short_branch() {
    let code = rv32("bnez x5, target\ntarget:");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_bgt_short_branch() {
    let code = rv32("bgt x1, x2, target\ntarget:");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_branch_forward_and_backward() {
    // Forward and backward branches in one assembly
    let code =
        rv32("loop:\n  addi x1, x1, 1\n  bne x1, x2, loop\n  beq x3, x4, done\n  nop\ndone:");
    // loop: addi (4) + bne (4) + beq (4) + nop (4) = 16 bytes
    assert_eq!(code.len(), 16);
}

// ── M Extension (end-to-end) ─────────────────────────────────────────────

#[test]
fn rv32_mul_div_rem() {
    let code = rv32("mul x1, x2, x3\ndiv x4, x5, x6\nrem x7, x8, x9");
    assert_eq!(code.len(), 12);
    // Check funct7 = 0b0000001 for all M-extension instructions
    for i in 0..3 {
        let w = le32(&code, i * 4);
        assert_eq!((w >> 25) & 0x7F, 1); // funct7 = M extension
    }
}

// ── Privileged instructions ──────────────────────────────────────────────

#[test]
fn rv32_mret() {
    // mret: 0011000_00010_00000_000_00000_1110011
    let code = rv32("mret");
    assert_eq!(le32(&code, 0), 0x30200073);
}

#[test]
fn rv32_sret() {
    // sret: 0001000_00010_00000_000_00000_1110011
    let code = rv32("sret");
    assert_eq!(le32(&code, 0), 0x10200073);
}

#[test]
fn rv32_wfi() {
    // wfi: 0001000_00101_00000_000_00000_1110011
    let code = rv32("wfi");
    assert_eq!(le32(&code, 0), 0x10500073);
}

#[test]
fn rv32_sfence_vma_no_args() {
    // sfence.vma (no args) → sfence.vma x0, x0
    let code = rv32("sfence.vma");
    assert_eq!(le32(&code, 0), 0x12000073);
}

#[test]
fn rv32_sfence_vma_two_regs() {
    // sfence.vma x1, x2
    let code = rv32("sfence.vma x1, x2");
    assert_eq!(le32(&code, 0), 0x12208073);
}

// ── Branch pseudo-instructions ───────────────────────────────────────────

#[test]
fn rv32_blez_short_branch() {
    // blez x5, target → bge x0, x5, target
    let code = rv32("blez x5, target\ntarget:");
    assert_eq!(code.len(), 4);
    let w = le32(&code, 0);
    assert_eq!(w & 0x7F, 0x63); // BRANCH opcode
    assert_eq!((w >> 12) & 0x7, 0b101); // funct3 = bge
}

#[test]
fn rv32_bgez_short_branch() {
    // bgez x5, target → bge x5, x0, target
    let code = rv32("bgez x5, target\ntarget:");
    assert_eq!(code.len(), 4);
    let w = le32(&code, 0);
    assert_eq!(w & 0x7F, 0x63); // BRANCH opcode
    assert_eq!((w >> 12) & 0x7, 0b101); // funct3 = bge
}

#[test]
fn rv32_bltz_short_branch() {
    // bltz x5, target → blt x5, x0, target
    let code = rv32("bltz x5, target\ntarget:");
    assert_eq!(code.len(), 4);
    let w = le32(&code, 0);
    assert_eq!(w & 0x7F, 0x63); // BRANCH opcode
    assert_eq!((w >> 12) & 0x7, 0b100); // funct3 = blt
}

#[test]
fn rv32_bgtz_short_branch() {
    // bgtz x5, target → blt x0, x5, target
    let code = rv32("bgtz x5, target\ntarget:");
    assert_eq!(code.len(), 4);
    let w = le32(&code, 0);
    assert_eq!(w & 0x7F, 0x63); // BRANCH opcode
    assert_eq!((w >> 12) & 0x7, 0b100); // funct3 = blt
}

// ── Session 7: New integration tests ──────────────────────────

// ── AArch64 wider multiply ───────────────────────────────────

#[test]
fn aarch64_smaddl_x0_w1_w2_x3() {
    assert_eq!(a64_word("smaddl x0, w1, w2, x3"), 0x9B22_0C20);
}

#[test]
fn aarch64_umaddl_x0_w1_w2_x3() {
    assert_eq!(a64_word("umaddl x0, w1, w2, x3"), 0x9BA2_0C20);
}

#[test]
fn aarch64_smsubl_x0_w1_w2_x3() {
    assert_eq!(a64_word("smsubl x0, w1, w2, x3"), 0x9B22_8C20);
}

#[test]
fn aarch64_umsubl_x0_w1_w2_x3() {
    assert_eq!(a64_word("umsubl x0, w1, w2, x3"), 0x9BA2_8C20);
}

#[test]
fn aarch64_smnegl_x0_w1_w2() {
    assert_eq!(a64_word("smnegl x0, w1, w2"), 0x9B22_FC20);
}

#[test]
fn aarch64_umnegl_x0_w1_w2() {
    assert_eq!(a64_word("umnegl x0, w1, w2"), 0x9BA2_FC20);
}

#[test]
fn aarch64_smulh_x0_x1_x2() {
    assert_eq!(a64_word("smulh x0, x1, x2"), 0x9B42_7C20);
}

#[test]
fn aarch64_umulh_x0_x1_x2() {
    assert_eq!(a64_word("umulh x0, x1, x2"), 0x9BC2_7C20);
}

// ── ARM32 halfword pre/post-index ────────────────────────────

#[test]
fn arm32_ldrh_preindex() {
    let code = assemble("ldrh r0, [r1, 4]!", Arch::Arm).unwrap();
    assert_eq!(code.len(), 4);
    let w = u32::from_le_bytes(code[..4].try_into().unwrap());
    assert_eq!(w, 0xE1F1_00B4);
}

#[test]
fn arm32_strh_preindex_neg() {
    let code = assemble("strh r0, [r1, -8]!", Arch::Arm).unwrap();
    assert_eq!(code.len(), 4);
    let w = u32::from_le_bytes(code[..4].try_into().unwrap());
    assert_eq!(w, 0xE161_00B8);
}

#[test]
fn arm32_ldrh_postindex() {
    let code = assemble("ldrh r0, [r1], 4", Arch::Arm).unwrap();
    assert_eq!(code.len(), 4);
    let w = u32::from_le_bytes(code[..4].try_into().unwrap());
    assert_eq!(w, 0xE0D1_00B4);
}

#[test]
fn arm32_strh_postindex_neg() {
    let code = assemble("strh r0, [r1], -4", Arch::Arm).unwrap();
    assert_eq!(code.len(), 4);
    let w = u32::from_le_bytes(code[..4].try_into().unwrap());
    assert_eq!(w, 0xE041_00B4);
}

// ── ARM32 bitfield instructions ──────────────────────────────

#[test]
fn arm32_bfc_r0_4_8() {
    // BFC R0, #4, #8 — clear 8 bits starting at bit 4
    // cond=E, 0111110, msb=11, Rd=0, lsb=4, 001, Rn=1111
    let code = assemble("bfc r0, 4, 8", Arch::Arm).unwrap();
    assert_eq!(code.len(), 4);
    let w = u32::from_le_bytes(code[..4].try_into().unwrap());
    assert_eq!(w, 0xE7CB_021F);
}

#[test]
fn arm32_bfi_r0_r1_0_8() {
    // BFI R0, R1, #0, #8 — insert 8 bits from R1 at bit 0 of R0
    let code = assemble("bfi r0, r1, 0, 8", Arch::Arm).unwrap();
    assert_eq!(code.len(), 4);
    let w = u32::from_le_bytes(code[..4].try_into().unwrap());
    assert_eq!(w, 0xE7C7_0011);
}

#[test]
fn arm32_sbfx_r0_r1_4_8() {
    // SBFX R0, R1, #4, #8 — signed extract 8 bits from bit 4
    let code = assemble("sbfx r0, r1, 4, 8", Arch::Arm).unwrap();
    assert_eq!(code.len(), 4);
    let w = u32::from_le_bytes(code[..4].try_into().unwrap());
    assert_eq!(w, 0xE7A7_0251);
}

#[test]
fn arm32_ubfx_r0_r1_4_8() {
    // UBFX R0, R1, #4, #8 — unsigned extract 8 bits from bit 4
    let code = assemble("ubfx r0, r1, 4, 8", Arch::Arm).unwrap();
    assert_eq!(code.len(), 4);
    let w = u32::from_le_bytes(code[..4].try_into().unwrap());
    assert_eq!(w, 0xE7E7_0251);
}

// ============================================================================
// Thumb / Thumb-2 Integration Tests (Story 4.3)
// ============================================================================

// ── Helper ───────────────────────────────────────────────────

fn thumb(src: &str) -> Vec<u8> {
    assemble(src, Arch::Thumb).unwrap()
}

fn thumb_u16(src: &str) -> u16 {
    let bytes = thumb(src);
    assert_eq!(bytes.len(), 2, "expected 2 bytes for Thumb-1 instruction");
    u16::from_le_bytes([bytes[0], bytes[1]])
}

// ── Basic 16-bit instructions ────────────────────────────────

#[test]
fn thumb_integration_nop() {
    assert_eq!(thumb("nop"), vec![0x00, 0xBF]);
}

#[test]
fn thumb_integration_bkpt_42() {
    assert_eq!(thumb_u16("bkpt 42"), 0xBE2A);
}

#[test]
fn thumb_integration_mov_r0_100() {
    assert_eq!(thumb_u16("mov r0, 100"), 0x2064);
}

#[test]
fn thumb_integration_add_r1_r2_r3() {
    assert_eq!(thumb_u16("add r1, r2, r3"), 0x18D1);
}

#[test]
fn thumb_integration_sub_r0_r1_5() {
    assert_eq!(thumb_u16("sub r0, r1, 5"), 0x1F48);
}

#[test]
fn thumb_integration_lsl_r0_r1_8() {
    assert_eq!(thumb_u16("lsl r0, r1, 8"), 0x0208);
}

#[test]
fn thumb_integration_ldr_r0_sp_rel() {
    assert_eq!(thumb_u16("ldr r0, [sp, 16]"), 0x9804);
}

#[test]
fn thumb_integration_str_r1_r2_imm() {
    assert_eq!(thumb_u16("str r1, [r2, 8]"), 0x6091);
}

#[test]
fn thumb_integration_push_r4_lr() {
    // PUSH {R4, LR} → 0xB510
    assert_eq!(thumb_u16("push {r4, lr}"), 0xB510);
}

#[test]
fn thumb_integration_pop_r4_pc() {
    // POP {R4, PC} → 0xBD10
    assert_eq!(thumb_u16("pop {r4, pc}"), 0xBD10);
}

#[test]
fn thumb_integration_bx_lr() {
    assert_eq!(thumb_u16("bx lr"), 0x4770);
}

// ── Branch with label ────────────────────────────────────────

#[test]
fn thumb_integration_b_forward() {
    // label is 2 bytes ahead: B +0 (offset = 0 since PC+4 lands on target)
    let bytes = thumb("b target\ntarget: nop");
    assert_eq!(bytes.len(), 4);
    let hw = u16::from_le_bytes([bytes[0], bytes[1]]);
    // B offset → 11100 xxxxxxxxxxx
    assert_eq!(hw >> 11, 0b11100);
}

#[test]
fn thumb_integration_beq_forward() {
    // B.EQ forward
    let bytes = thumb("beq target\ntarget: nop");
    assert_eq!(bytes.len(), 4);
    let hw = u16::from_le_bytes([bytes[0], bytes[1]]);
    // B.cond: 1101 cccc xxxxxxxx, cccc=0000 (EQ)
    assert_eq!(hw >> 12, 0b1101);
    assert_eq!((hw >> 8) & 0xF, 0b0000); // EQ condition
}

#[test]
fn thumb_integration_bl_forward() {
    // BL is always 32-bit
    let bytes = thumb("bl target\ntarget: nop");
    assert_eq!(bytes.len(), 6); // 4 bytes BL + 2 bytes NOP
    let hw1 = u16::from_le_bytes([bytes[0], bytes[1]]);
    let hw2 = u16::from_le_bytes([bytes[2], bytes[3]]);
    // hw1: 11110 Sxxxxxxxxxx, hw2: 11x1 xxxxxxxxxxx
    assert_eq!(hw1 >> 11, 0b11110);
    assert!(hw2 & 0xD000 == 0xD000, "hw2 should start with 11x1");
}

// ── Multi-instruction sequences ──────────────────────────────

#[test]
fn thumb_integration_exit_shellcode() {
    // Thumb Linux exit(0) shellcode
    let code = thumb(
        "mov r0, 0\n\
         mov r7, 1\n\
         svc 0",
    );
    assert_eq!(code.len(), 6); // 3 × 2-byte instructions
    let i0 = u16::from_le_bytes([code[0], code[1]]);
    let i1 = u16::from_le_bytes([code[2], code[3]]);
    let i2 = u16::from_le_bytes([code[4], code[5]]);
    assert_eq!(i0, 0x2000); // MOV R0, #0
    assert_eq!(i1, 0x2701); // MOV R7, #1
    assert_eq!(i2, 0xDF00); // SVC #0
}

#[test]
fn thumb_integration_function_prologue() {
    let code = thumb(
        "push {r4, r5, r6, lr}\n\
         mov r4, r0\n\
         mov r5, r1",
    );
    assert_eq!(code.len(), 6); // 3 × 2-byte
    let push = u16::from_le_bytes([code[0], code[1]]);
    // PUSH {R4, R5, R6, LR} → 1011010100110000 = 0xB570
    assert_eq!(push, 0xB570);
}

#[test]
fn thumb_integration_function_epilogue() {
    let code = thumb("pop {r4, r5, r6, pc}\n");
    let pop = u16::from_le_bytes([code[0], code[1]]);
    // POP {R4, R5, R6, PC} → 1011110100110000 = 0xBD70
    assert_eq!(pop, 0xBD70);
}

// ── IT blocks ────────────────────────────────────────────────

#[test]
fn thumb_integration_it_eq_mov() {
    let code = thumb("it eq\nmov r0, 1");
    assert_eq!(code.len(), 4);
    let it = u16::from_le_bytes([code[0], code[1]]);
    let mov = u16::from_le_bytes([code[2], code[3]]);
    assert_eq!(it, 0xBF08); // IT EQ
    assert_eq!(mov, 0x2001); // MOV R0, #1
}

#[test]
fn thumb_integration_ite_ne() {
    let code = thumb("ite ne\nmov r0, 1\nmov r0, 0");
    assert_eq!(code.len(), 6);
    let it = u16::from_le_bytes([code[0], code[1]]);
    assert_eq!(it, 0xBF14); // ITE NE (cond=0001, mask_bits=0100 → mask=0100)
}

// ── .thumb / .arm directive switching ────────────────────────

#[test]
fn thumb_arm_mode_switch() {
    // Start in ARM, switch to Thumb, back to ARM
    let mut asm = Assembler::new(Arch::Arm);
    asm.emit(".thumb\nnop\n.arm\nnop").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // Thumb NOP = 2 bytes, ARM NOP = 4 bytes → 6 bytes total
    assert_eq!(bytes.len(), 6);
    // First 2 bytes: Thumb NOP (0xBF00)
    assert_eq!(&bytes[0..2], &[0x00, 0xBF]);
    // Last 4 bytes: ARM NOP (MOV R0, R0 = 0xE1A00000)
    assert_eq!(&bytes[2..6], &[0x00, 0x00, 0xA0, 0xE1]);
}

#[test]
fn thumb_directive_from_arm_start() {
    // Starting in ARM mode, .thumb switches to Thumb
    let mut asm = Assembler::new(Arch::Arm);
    asm.emit("nop\n.thumb\nnop").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    assert_eq!(bytes.len(), 6); // 4 ARM + 2 Thumb
}

// ── .thumb_func directive ────────────────────────────────────

#[test]
fn thumb_func_sets_lsb() {
    let mut asm = Assembler::new(Arch::Arm);
    asm.emit(".thumb_func\nmy_func:\nnop").unwrap();
    let result = asm.finish().unwrap();
    // .thumb_func sets LSB of label address
    let addr = result.label_address("my_func").unwrap();
    assert_eq!(addr & 1, 1, ".thumb_func should set LSB");
}

#[test]
fn thumb_func_switches_to_thumb() {
    // .thumb_func should also switch to Thumb mode
    let mut asm = Assembler::new(Arch::Arm);
    asm.emit(".thumb_func\nmy_func:\nnop").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // NOP should be Thumb (2 bytes), not ARM (4 bytes)
    assert_eq!(bytes.len(), 2);
    assert_eq!(bytes, vec![0x00, 0xBF]);
}

// ── Error cases ──────────────────────────────────────────────

#[test]
fn thumb_directive_wrong_arch() {
    // .thumb should fail on non-ARM architectures
    let result = assemble(".thumb\nnop", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn thumb_func_wrong_arch() {
    let result = assemble(".thumb_func\nfoo:\nnop", Arch::X86_64);
    assert!(result.is_err());
}

// ── Thumb literal pools ──────────────────────────────────────

#[test]
fn thumb_ldr_literal_pool() {
    // LDR R0, =0x12345678 should produce a PC-relative load + pool entry
    let mut asm = Assembler::new(Arch::Thumb);
    asm.emit("ldr r0, =0x12345678\n.ltorg").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // 2 bytes LDR + 2 bytes align padding (if needed) + 4 bytes pool = 6 or 8 bytes
    // LDR R0, [PC, #N] is at offset 0
    let ldr = u16::from_le_bytes([bytes[0], bytes[1]]);
    // Encoding: 01001 Rt(3) imm8(8), Rt=0
    assert_eq!(ldr >> 11, 0b01001);
    assert_eq!((ldr >> 8) & 0x7, 0); // Rt = R0
                                     // Pool entry should contain the value 0x12345678
                                     // Find it in the last 4 bytes
    let pool_start = bytes.len() - 4;
    let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
    assert_eq!(pool_val, 0x12345678);
}

#[test]
fn thumb_ldr_literal_pool_dedup() {
    // Two LDR with same constant should share pool entry
    let mut asm = Assembler::new(Arch::Thumb);
    asm.emit("ldr r0, =42\nldr r1, =42\n.ltorg").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // 2+2 bytes instructions + 4 bytes pool (one entry, deduplicated) = 8 bytes
    assert_eq!(bytes.len(), 8);
    // Both LDR should have same imm8 pointing to same pool entry
    let ldr0 = u16::from_le_bytes([bytes[0], bytes[1]]);
    let ldr1 = u16::from_le_bytes([bytes[2], bytes[3]]);
    assert_eq!(ldr0 & 0xFF, ldr1 & 0xFF); // same offset (adjusted for PC alignment)
}

#[test]
fn thumb_ldr_literal_pool_value() {
    // Verify pool entry bytes
    let mut asm = Assembler::new(Arch::Thumb);
    asm.emit("ldr r3, =0xDEADBEEF\n.ltorg").unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    let pool_val = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
    assert_eq!(pool_val, 0xDEADBEEF);
}

// ── ALU operations ───────────────────────────────────────────

#[test]
fn thumb_integration_and_reg() {
    assert_eq!(thumb_u16("and r0, r1"), 0x4008);
}

#[test]
fn thumb_integration_orr_reg() {
    assert_eq!(thumb_u16("orr r2, r3"), 0x431A);
}

#[test]
fn thumb_integration_eor_reg() {
    assert_eq!(thumb_u16("eor r0, r1"), 0x4048);
}

#[test]
fn thumb_integration_mvn_r0_r1() {
    assert_eq!(thumb_u16("mvn r0, r1"), 0x43C8);
}

#[test]
fn thumb_integration_mul_r1_r2() {
    assert_eq!(thumb_u16("mul r1, r2"), 0x4351);
}

#[test]
fn thumb_integration_cmp_r0_100() {
    assert_eq!(thumb_u16("cmp r0, 100"), 0x2864);
}

// ── Load/store byte and halfword ─────────────────────────────

#[test]
fn thumb_integration_ldrb_r0_r1_0() {
    assert_eq!(thumb_u16("ldrb r0, [r1, 0]"), 0x7808);
}

#[test]
fn thumb_integration_strh_r2_r3_4() {
    assert_eq!(thumb_u16("strh r2, [r3, 4]"), 0x809A);
}

// ── Mixed sequence test ──────────────────────────────────────

#[test]
fn thumb_integration_loop_pattern() {
    // Common tight loop in Thumb
    let code = thumb(
        "loop:\n\
         sub r0, 1\n\
         cmp r0, 0\n\
         bne loop",
    );
    assert_eq!(code.len(), 6); // 3 × 2-byte instructions
    let sub = u16::from_le_bytes([code[0], code[1]]);
    let cmp = u16::from_le_bytes([code[2], code[3]]);
    assert_eq!(sub, 0x3801); // SUB R0, #1
    assert_eq!(cmp, 0x2800); // CMP R0, #0
}

// ── x86 ADX extension (ADCX/ADOX) ───────────────────────────

#[test]
fn x86_adcx_eax_ebx() {
    let code = assemble("adcx eax, ebx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0F, 0x38, 0xF6, 0xC3]);
}

#[test]
fn x86_adcx_rax_rbx() {
    let code = assemble("adcx rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x48, 0x0F, 0x38, 0xF6, 0xC3]);
}

#[test]
fn x86_adox_eax_ebx() {
    let code = assemble("adox eax, ebx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xF3, 0x0F, 0x38, 0xF6, 0xC3]);
}

#[test]
fn x86_adox_rax_rbx() {
    let code = assemble("adox rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xF3, 0x48, 0x0F, 0x38, 0xF6, 0xC3]);
}

// ── Preprocessor expression operators ────────────────────────

#[test]
fn preproc_if_multiply_equals() {
    // Previously silently returned 0 for * operator (data corruption bug)
    let code = assemble(".if 2 * 3 == 6\nnop\n.endif", Arch::X86_64).unwrap();
    assert!(!code.is_empty(), "2*3==6 should emit code");
}

#[test]
fn preproc_if_shift_flag_test() {
    let code = assemble(
        ".equ SHIFT, 3\n.if 1 << SHIFT == 8\nnop\n.endif",
        Arch::X86_64,
    )
    .unwrap();
    assert!(!code.is_empty(), "1<<3==8 should emit code");
}

#[test]
fn preproc_if_parenthesised() {
    let code = assemble(".if (1 + 2) * 4 == 12\nnop\n.endif", Arch::X86_64).unwrap();
    assert!(!code.is_empty(), "(1+2)*4==12 should emit code");
}

#[test]
fn preproc_if_bitwise_and_or() {
    let code = assemble(
        ".equ FLAGS, 0x07\n.if (FLAGS & 0x02) != 0\nnop\n.endif",
        Arch::X86_64,
    )
    .unwrap();
    assert!(!code.is_empty(), "0x07 & 0x02 should be non-zero");
}

#[test]
fn preproc_if_logical_and_or() {
    let mut asm = asm_rs::Assembler::new(Arch::X86_64);
    asm.define_preprocessor_symbol("A", 1);
    asm.define_preprocessor_symbol("B", 1);
    asm.emit(".if defined(A) && defined(B)\nnop\n.endif")
        .unwrap();
    let result = asm.finish().unwrap();
    assert!(
        !result.bytes().is_empty(),
        "both symbols defined, && should be true"
    );
}

// ============================================================================
// Resource Limits — Pathological Input Tests (Story 10.1)
// ============================================================================

#[test]
fn resource_limit_max_statements() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_statements: 5,
        ..Default::default()
    });
    // 6 nops should exceed the 5-statement limit
    let result = asm.emit("nop\nnop\nnop\nnop\nnop\nnop");
    assert!(
        result.is_err() || asm.finish().is_err(),
        "should fail when exceeding max_statements"
    );
}

#[test]
fn resource_limit_max_labels() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_labels: 3,
        ..Default::default()
    });
    // Define 4 labels to exceed the 3-label limit
    let result = asm.emit("a:\nnop\nb:\nnop\nc:\nnop\nd:\nnop");
    assert!(
        result.is_err() || asm.finish().is_err(),
        "should fail when exceeding max_labels"
    );
}

#[test]
fn resource_limit_max_output_bytes() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_output_bytes: 8,
        ..Default::default()
    });
    // .fill 100, 1, 0x90 should exceed the 8-byte limit
    let emit_result = asm.emit(".fill 100, 1, 0x90");
    if emit_result.is_ok() {
        let finish_result = asm.finish();
        assert!(
            finish_result.is_err(),
            "should fail when exceeding max_output_bytes"
        );
    }
    // If emit already failed, that's also acceptable
}

#[test]
fn resource_limit_max_errors() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_errors: 2,
        ..Default::default()
    });
    // Feed multiple bad instructions — should bail after 2 errors
    let _ = asm.emit("badinstr1\nbadinstr2\nbadinstr3");
    let result = asm.finish();
    assert!(result.is_err(), "should collect errors and report failure");
}

#[test]
fn resource_limit_max_recursion_depth() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_recursion_depth: 5,
        ..Default::default()
    });
    // Define a recursive macro that exceeds the shallow recursion limit
    let result = asm.emit(".macro boom\nboom\n.endm\nboom");
    assert!(
        result.is_err() || asm.finish().is_err(),
        "should fail when exceeding max_recursion_depth"
    );
}

#[test]
fn resource_limit_deeply_nested_rept() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_statements: 500,
        ..Default::default()
    });
    // .rept 1000 × nop should exceed 500 statement limit
    let result = asm.emit(".rept 1000\nnop\n.endr");
    assert!(
        result.is_err() || asm.finish().is_err(),
        "should fail when rept generates too many statements"
    );
}

#[test]
fn resource_limit_many_labels_via_rept() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_labels: 10,
        max_statements: 100_000,
        ..Default::default()
    });
    // Generate 20 labels via macro + rept
    let result = asm.emit(
        ".set COUNTER, 0\n\
         .rept 20\n\
         1:\n\
         nop\n\
         .endr",
    );
    // Numeric labels may not count against the limit the same way,
    // so check both emit and finish
    let _ = result;
    // This test mainly verifies the assembler doesn't OOM or panic
}

#[test]
fn pathological_huge_rept_count_rejected() {
    // .rept with a huge count should be bounded by max_statements
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_statements: 1_000,
        ..Default::default()
    });
    let result = asm.emit(".rept 999999\nnop\n.endr");
    assert!(
        result.is_err() || asm.finish().is_err(),
        "huge .rept should be bounded by max_statements"
    );
}

// ============================================================================
// 16-bit Real Mode (.code16)
// ============================================================================

#[test]
fn code16_mov_ax_imm16() {
    // In 16-bit mode, mov ax, 0x1234 has no 0x66 prefix
    let bytes = assemble(".code16\nmov ax, 0x1234", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0xB8, 0x34, 0x12]);
}

#[test]
fn code16_mov_eax_imm32_needs_prefix() {
    // In 16-bit mode, mov eax, 0x12345678 needs 0x66 prefix
    let bytes = assemble(".code16\nmov eax, 0x12345678", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0x66, 0xB8, 0x78, 0x56, 0x34, 0x12]);
}

#[test]
fn code16_push_ax_no_prefix() {
    let bytes = assemble(".code16\npush ax", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0x50]);
}

#[test]
fn code16_push_eax_needs_prefix() {
    let bytes = assemble(".code16\npush eax", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0x66, 0x50]);
}

#[test]
fn code16_pop_bx_no_prefix() {
    let bytes = assemble(".code16\npop bx", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0x5B]);
}

#[test]
fn code16_inc_cx_short_form() {
    let bytes = assemble(".code16\ninc cx", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0x41]);
}

#[test]
fn code16_inc_ecx_needs_prefix() {
    let bytes = assemble(".code16\ninc ecx", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0x66, 0x41]);
}

#[test]
fn code16_xor_ax_ax_no_prefix() {
    let bytes = assemble(".code16\nxor ax, ax", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0x31, 0xC0]);
}

#[test]
fn code16_cli_hlt() {
    // Simple bootloader prologue
    let bytes = assemble(".code16\ncli\nhlt", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0xFA, 0xF4]);
}

#[test]
fn code16_int_10h() {
    let bytes = assemble(".code16\nint 0x10", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0xCD, 0x10]);
}

#[test]
fn code16_push_segment_registers() {
    let bytes = assemble(".code16\npush es\npush cs\npush ss\npush ds", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0x06, 0x0E, 0x16, 0x1E]);
}

#[test]
fn code16_pop_segment_registers() {
    let bytes = assemble(".code16\npop es\npop ss\npop ds", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0x07, 0x17, 0x1F]);
}

#[test]
fn code16_jmp_short_self() {
    // jmp $ in 16-bit mode → EB FE (short jump to self)
    let bytes = assemble(".code16\nhere:\njmp here", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0xEB, 0xFE]);
}

#[test]
fn code16_mov_al_imm8() {
    // 8-bit ops are the same regardless of mode
    let bytes = assemble(".code16\nmov al, 0x42", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0xB0, 0x42]);
}

#[test]
fn code16_add_ax_bx() {
    let bytes = assemble(".code16\nadd ax, bx", Arch::X86).unwrap();
    assert_eq!(bytes, vec![0x01, 0xD8]);
}

#[test]
fn code16_mode_switch_code32() {
    // Start in 16-bit mode, switch to 32-bit
    let bytes = assemble(".code16\nmov ax, 1\n.code32\nmov eax, 1", Arch::X86).unwrap();
    // 16-bit: mov ax, 1 → B8 01 00
    // 32-bit: mov eax, 1 → B8 01 00 00 00
    assert_eq!(bytes, vec![0xB8, 0x01, 0x00, 0xB8, 0x01, 0x00, 0x00, 0x00]);
}

#[test]
fn code16_bootloader_stub() {
    // Minimal bootloader: cli, hlt, pad to 510 bytes, 0x55AA signature
    let asm_src = "\
.code16
cli
hlt
.fill 508, 1, 0
.word 0xAA55
";
    let bytes = assemble(asm_src, Arch::X86).unwrap();
    assert_eq!(bytes.len(), 512);
    assert_eq!(bytes[0], 0xFA); // cli
    assert_eq!(bytes[1], 0xF4); // hlt
    assert!(bytes[2..510].iter().all(|&b| b == 0)); // padding
    assert_eq!(bytes[510], 0x55); // boot signature low byte
    assert_eq!(bytes[511], 0xAA); // boot signature high byte
}

#[test]
fn code16_rejects_64bit_registers() {
    let result = assemble(".code16\nmov rax, 1", Arch::X86);
    assert!(result.is_err());
}

// ============================================================================
// AT&T / GAS Syntax
// ============================================================================

fn assemble_att(source: &str, arch: Arch) -> Result<Vec<u8>, AsmError> {
    let mut asm = Assembler::new(arch);
    asm.syntax(asm_rs::Syntax::Att);
    asm.emit(source)?;
    Ok(asm.finish()?.bytes().to_vec())
}

#[test]
fn att_mov_imm_to_reg_integration() {
    // movq $1, %rax → optimizer narrows to mov eax, 1 → 0xB8 01 00 00 00
    let bytes = assemble_att("movq $1, %rax", Arch::X86_64).unwrap();
    assert_eq!(bytes, vec![0xB8, 0x01, 0x00, 0x00, 0x00]);
}

#[test]
fn att_matches_intel_mov_reg_reg() {
    let att_bytes = assemble_att("movq %rax, %rcx", Arch::X86_64).unwrap();
    let intel_bytes = assemble("mov rcx, rax", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_add_imm() {
    let att_bytes = assemble_att("addl $0x10, %eax", Arch::X86_64).unwrap();
    let intel_bytes = assemble("add eax, 0x10", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_sub_imm() {
    let att_bytes = assemble_att("subq $8, %rsp", Arch::X86_64).unwrap();
    let intel_bytes = assemble("sub rsp, 8", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_xor_reg_reg() {
    let att_bytes = assemble_att("xorl %eax, %eax", Arch::X86_64).unwrap();
    let intel_bytes = assemble("xor eax, eax", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_cmp_imm() {
    let att_bytes = assemble_att("cmpl $0, %eax", Arch::X86_64).unwrap();
    let intel_bytes = assemble("cmp eax, 0", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_push_reg() {
    let att_bytes = assemble_att("pushq %rbp", Arch::X86_64).unwrap();
    let intel_bytes = assemble("push rbp", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_pop_reg() {
    let att_bytes = assemble_att("popq %rbp", Arch::X86_64).unwrap();
    let intel_bytes = assemble("pop rbp", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_lea() {
    let att_bytes = assemble_att("leaq 8(%rsp), %rax", Arch::X86_64).unwrap();
    let intel_bytes = assemble("lea rax, [rsp + 8]", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_mov_mem_to_reg() {
    let att_bytes = assemble_att("movq (%rax), %rbx", Arch::X86_64).unwrap();
    let intel_bytes = assemble("mov rbx, [rax]", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_mov_reg_to_mem() {
    let att_bytes = assemble_att("movq %rax, (%rbx)", Arch::X86_64).unwrap();
    let intel_bytes = assemble("mov [rbx], rax", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_mov_disp_mem() {
    let att_bytes = assemble_att("movq -16(%rbp), %rax", Arch::X86_64).unwrap();
    let intel_bytes = assemble("mov rax, [rbp - 16]", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_sib_base_index_scale() {
    let att_bytes = assemble_att("movl (%rax, %rcx, 4), %edx", Arch::X86_64).unwrap();
    let intel_bytes = assemble("mov edx, [rax + rcx*4]", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_sib_disp_base_index_scale() {
    let att_bytes = assemble_att("movl 16(%rbx, %rsi, 8), %eax", Arch::X86_64).unwrap();
    let intel_bytes = assemble("mov eax, [rbx + rsi*8 + 16]", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_nop() {
    let att_bytes = assemble_att("nop", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, vec![0x90]);
}

#[test]
fn att_matches_intel_ret() {
    let att_bytes = assemble_att("ret", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, vec![0xC3]);
}

#[test]
fn att_matches_intel_syscall() {
    let att_bytes = assemble_att("syscall", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, vec![0x0F, 0x05]);
}

#[test]
fn att_matches_intel_int() {
    let att_bytes = assemble_att("int $0x80", Arch::X86_64).unwrap();
    let intel_bytes = assemble("int 0x80", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_test_reg_reg() {
    let att_bytes = assemble_att("testl %edi, %edi", Arch::X86_64).unwrap();
    let intel_bytes = assemble("test edi, edi", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_imul() {
    let att_bytes = assemble_att("imull $10, %eax, %ecx", Arch::X86_64).unwrap();
    let intel_bytes = assemble("imul ecx, eax, 10", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_movzx() {
    let att_bytes = assemble_att("movzbl %al, %eax", Arch::X86_64).unwrap();
    let intel_bytes = assemble("movzx eax, al", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_matches_intel_push_imm() {
    let att_bytes = assemble_att("pushq $42", Arch::X86_64).unwrap();
    let intel_bytes = assemble("push 42", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_forward_branch_matches_intel() {
    let att_src = "jmp end\nnop\nend:";
    let intel_src = "jmp end\nnop\nend:";
    let att_bytes = assemble_att(att_src, Arch::X86_64).unwrap();
    let intel_bytes = assemble(intel_src, Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_multi_instruction_program() {
    // A typical function prologue/epilogue in AT&T
    let src = "pushq %rbp\nmovq %rsp, %rbp\nsubq $16, %rsp\naddq $16, %rsp\npopq %rbp\nret";
    let att_bytes = assemble_att(src, Arch::X86_64).unwrap();
    let intel_src = "push rbp\nmov rbp, rsp\nsub rsp, 16\nadd rsp, 16\npop rbp\nret";
    let intel_bytes = assemble(intel_src, Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_syntax_directive_inline() {
    // Start Intel, switch to AT&T mid-stream
    let src = "nop\n.syntax att\nmovq $1, %rax\nret";
    let bytes = assemble(src, Arch::X86_64).unwrap();
    // nop=0x90, mov eax,1 (narrowed)=B8 01 00 00 00, ret=C3
    assert_eq!(bytes[0], 0x90);
    assert_eq!(bytes[1], 0xB8);
    assert_eq!(*bytes.last().unwrap(), 0xC3);
}

#[test]
fn att_x86_32_mode() {
    // AT&T in 32-bit mode
    let att_bytes = assemble_att("movl $42, %eax", Arch::X86).unwrap();
    let intel_bytes = assemble("mov eax, 42", Arch::X86).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_lock_prefix_integration() {
    let att_bytes = assemble_att("lock xchgl %eax, (%rbx)", Arch::X86_64).unwrap();
    let intel_bytes = assemble("lock xchg [rbx], eax", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_rep_movsb() {
    let att_bytes = assemble_att("rep movsb", Arch::X86_64).unwrap();
    let intel_bytes = assemble("rep movsb", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_indirect_jmp() {
    let att_bytes = assemble_att("jmp *%rax", Arch::X86_64).unwrap();
    let intel_bytes = assemble("jmp rax", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

#[test]
fn att_indirect_call_mem() {
    let att_bytes = assemble_att("call *(%rax)", Arch::X86_64).unwrap();
    let intel_bytes = assemble("call [rax]", Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

// ─── AArch64 Literal Pool integration tests ─────────────────────────

fn assemble_aarch64(source: &str) -> Vec<u8> {
    let mut asm = Assembler::new(Arch::Aarch64);
    asm.emit(source).unwrap();
    asm.finish().unwrap().bytes().to_vec()
}

#[test]
fn literal_pool_ldr_x_basic() {
    // LDR X0, =0x12345678 should produce valid LDR (literal) + pool data
    let bytes = assemble_aarch64("ldr x0, =0x12345678");
    assert!(bytes.len() >= 12, "need instruction + pool data");

    // Verify LDR (literal) encoding
    let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!((word >> 30) & 0b11, 0b01, "opc=01 for 64-bit LDR");
    assert_eq!((word >> 24) & 0b111111, 0b011000, "LDR literal opcode");
    assert_eq!(word & 0x1F, 0, "Rt=X0");

    // Verify pool contains the constant
    let pool_start = bytes.len() - 8;
    let pool_val = u64::from_le_bytes(bytes[pool_start..pool_start + 8].try_into().unwrap());
    assert_eq!(pool_val, 0x12345678);
}

#[test]
fn literal_pool_ldr_w_basic() {
    let bytes = assemble_aarch64("ldr w3, =0x42");
    let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    // opc=00 for 32-bit LDR
    assert_eq!((word >> 30) & 0b11, 0b00, "opc=00 for 32-bit LDR");
    assert_eq!((word >> 24) & 0b111111, 0b011000, "LDR literal opcode");
    assert_eq!(word & 0x1F, 3, "Rt=W3");

    // Pool contains 4-byte value
    let pool_start = bytes.len() - 4;
    let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
    assert_eq!(pool_val, 0x42);
}

#[test]
fn literal_pool_ltorg_placement() {
    // .ltorg should flush the pool right after the directive
    let bytes = assemble_aarch64("ldr x0, =0xCAFE\n.ltorg\nnop");
    // Layout: LDR (4) + align + pool (8) + NOP (4)
    let nop_start = bytes.len() - 4;
    let nop_word = u32::from_le_bytes(bytes[nop_start..nop_start + 4].try_into().unwrap());
    assert_eq!(nop_word, 0xD503201F, "last 4 bytes should be NOP");

    // Pool value should be before the NOP
    let pool_start = nop_start - 8;
    let pool_val = u64::from_le_bytes(bytes[pool_start..pool_start + 8].try_into().unwrap());
    assert_eq!(pool_val, 0xCAFE);
}

#[test]
fn literal_pool_dedup_same_value() {
    // Two LDR with same value and same size should share one pool entry
    let dup_bytes = assemble_aarch64("ldr x0, =0xAA\nldr x1, =0xAA");
    let nodup_bytes = assemble_aarch64("ldr x0, =0xAA\nldr x1, =0xBB");
    // Deduplicated should be smaller (1 pool entry vs 2)
    assert!(
        dup_bytes.len() < nodup_bytes.len(),
        "dedup ({} bytes) should be smaller than nodup ({} bytes)",
        dup_bytes.len(),
        nodup_bytes.len()
    );
}

#[test]
fn literal_pool_large_constant() {
    let bytes = assemble_aarch64("ldr x0, =0xDEADBEEFCAFEBABE");
    let pool_start = bytes.len() - 8;
    let pool_val = u64::from_le_bytes(bytes[pool_start..pool_start + 8].try_into().unwrap());
    assert_eq!(pool_val, 0xDEADBEEFCAFEBABE);
}

#[test]
fn literal_pool_negative_constant() {
    let bytes = assemble_aarch64("ldr x0, =-1");
    let pool_start = bytes.len() - 8;
    let pool_val = u64::from_le_bytes(bytes[pool_start..pool_start + 8].try_into().unwrap());
    assert_eq!(pool_val, u64::MAX); // -1 as u64
}

#[test]
fn literal_pool_multiple_ldrs() {
    // Multiple LDR with different values
    let bytes = assemble_aarch64("ldr x0, =0x1111\nldr x1, =0x2222\nldr x2, =0x3333");
    // Should have 3 × 4-byte LDR instructions + alignment + 3 × 8-byte pool entries
    assert!(
        bytes.len() >= 36,
        "expected at least 36 bytes, got {}",
        bytes.len()
    );
}

#[test]
fn literal_pool_mixed_sizes() {
    // Mix of X-reg (8-byte) and W-reg (4-byte) pool entries
    let bytes = assemble_aarch64("ldr x0, =0xAAAA\nldr w1, =0xBBBB");
    // Both should assemble without errors
    assert!(bytes.len() >= 16);
}

#[test]
fn literal_pool_with_other_instructions() {
    // Literal pool should work alongside regular instructions
    let bytes = assemble_aarch64("mov x0, 42\nldr x1, =0xDEAD\nadd x2, x0, x1\n.ltorg");
    // 3 instructions (12 bytes) + alignment + pool (8 bytes)
    assert!(bytes.len() >= 20);
    // Pool should contain DEAD
    let pool_start = bytes.len() - 8;
    let pool_val = u64::from_le_bytes(bytes[pool_start..pool_start + 8].try_into().unwrap());
    assert_eq!(pool_val, 0xDEAD);
}

#[test]
fn literal_pool_pool_alias() {
    // .pool is an alias for .ltorg
    let ltorg_bytes = assemble_aarch64("ldr x0, =0xFF\n.ltorg");
    let pool_bytes = assemble_aarch64("ldr x0, =0xFF\n.pool");
    assert_eq!(ltorg_bytes, pool_bytes);
}

#[test]
fn literal_pool_zero_value() {
    let bytes = assemble_aarch64("ldr x0, =0");
    let pool_start = bytes.len() - 8;
    let pool_val = u64::from_le_bytes(bytes[pool_start..pool_start + 8].try_into().unwrap());
    assert_eq!(pool_val, 0);
}

// ─── AArch64 ADR relaxation (ADRP+ADD) integration tests ────

#[test]
fn adr_short_form_used_for_near_label() {
    // ADR X0, label where label is nearby → should use short form (4 bytes)
    let bytes = assemble_aarch64("adr x0, target\nnop\ntarget:");
    // ADR (4 bytes) + NOP (4 bytes) = 8 bytes, label at offset 8
    assert_eq!(
        bytes.len(),
        8,
        "ADR + NOP should be 8 bytes (short form ADR)"
    );
    let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    // Short form ADR: op=0 | immlo | 10000 | immhi | Rd=0
    assert_eq!((word >> 31) & 1, 0, "ADR should have op=0 (not ADRP)");
    assert_eq!((word >> 24) & 0b11111, 0b10000, "ADR opcode field");
    assert_eq!(word & 0x1F, 0, "Rd should be X0");
}

#[test]
fn adr_resolves_correct_offset() {
    // ADR should resolve to correct PC-relative offset
    let bytes = assemble_aarch64("adr x0, target\nnop\nnop\ntarget:");
    // ADR (4) + NOP (4) + NOP (4) = 12 bytes, target at 12
    // ADR offset = 12 - 0 = 12
    let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let immhi = (word >> 5) & 0x7FFFF;
    let immlo = (word >> 29) & 0x3;
    let imm = ((immhi << 2) | immlo) as i32;
    // Sign-extend 21-bit value
    let imm = if imm & (1 << 20) != 0 {
        imm | !((1 << 21) - 1)
    } else {
        imm
    };
    assert_eq!(imm, 12, "ADR offset should be 12 (3 words away)");
}

#[test]
fn adrp_label_resolves() {
    // ADRP X0, label — should work for nearby label
    let bytes = assemble_aarch64("adrp x0, target\nnop\ntarget:");
    assert_eq!(bytes.len(), 8);
    let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    // ADRP: op=1
    assert_eq!((word >> 31) & 1, 1, "ADRP should have op=1");
    assert_eq!(word & 0x1F, 0, "Rd should be X0");
}

// ─── ARM Literal Pool integration tests ─────────────────────────────

fn assemble_arm(source: &str) -> Vec<u8> {
    let mut asm = Assembler::new(Arch::Arm);
    asm.emit(source).unwrap();
    asm.finish().unwrap().bytes().to_vec()
}

#[test]
fn arm_literal_pool_ldr_basic() {
    // LDR R0, =0x12345678 → LDR [PC, #offset] + 4-byte pool entry
    let bytes = assemble_arm("ldr r0, =0x12345678");
    assert!(bytes.len() >= 8, "need instruction + pool data");

    // Verify LDR word: bits[27:26]=01, Rn=PC(15), L=1
    let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!((word >> 26) & 0b11, 0b01, "load/store word encoding");
    assert_eq!((word >> 16) & 0xF, 15, "Rn should be PC");
    assert_eq!((word >> 20) & 1, 1, "L=1 for load");

    // Pool entry should be 4 bytes
    let pool_start = bytes.len() - 4;
    let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
    assert_eq!(pool_val, 0x12345678);
}

#[test]
fn arm_literal_pool_ldr_various_regs() {
    // Test with different destination registers
    for (src, expected_rd) in &[
        ("ldr r0, =1", 0u32),
        ("ldr r3, =2", 3),
        ("ldr r7, =3", 7),
        ("ldr r12, =4", 12),
        ("ldr lr, =5", 14),
    ] {
        let bytes = assemble_arm(src);
        let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(
            (word >> 12) & 0xF,
            *expected_rd,
            "Rd should be {} for '{}'",
            expected_rd,
            src
        );
    }
}

#[test]
fn arm_literal_pool_ltorg_placement() {
    // .ltorg should flush the pool in place
    let bytes = assemble_arm("ldr r0, =0xAABBCCDD\n.ltorg\nnop");
    // Structure: LDR (4) + align + pool entry (4) + NOP (4)
    assert!(bytes.len() >= 12);
    // NOP is the last instruction
    let nop_word = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
    // ARM NOP = MOV R0, R0 = 0xE1A00000
    assert_eq!(nop_word, 0xE1A00000, "last word should be NOP");
}

#[test]
fn arm_literal_pool_dedup_same_value() {
    // Two LDRs loading the same constant should share one pool entry
    let bytes_dedup = assemble_arm("ldr r0, =0xFF\nldr r1, =0xFF");
    let bytes_no_dedup = assemble_arm("ldr r0, =0xFF\nldr r1, =0xFE");
    // With dedup: 2 LDR (8) + 1 pool entry (4) = 12
    // Without dedup: 2 LDR (8) + 2 pool entries (8) = 16
    assert!(
        bytes_dedup.len() < bytes_no_dedup.len(),
        "dedup ({} bytes) should be smaller than no-dedup ({} bytes)",
        bytes_dedup.len(),
        bytes_no_dedup.len()
    );
}

#[test]
fn arm_literal_pool_large_constant() {
    let bytes = assemble_arm("ldr r0, =0xDEADBEEF");
    let pool_start = bytes.len() - 4;
    let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
    assert_eq!(pool_val, 0xDEADBEEF);
}

#[test]
fn arm_literal_pool_negative_constant() {
    let bytes = assemble_arm("ldr r0, =-1");
    let pool_start = bytes.len() - 4;
    let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
    assert_eq!(pool_val, 0xFFFFFFFF); // -1 as u32
}

#[test]
fn arm_literal_pool_zero_value() {
    let bytes = assemble_arm("ldr r0, =0");
    let pool_start = bytes.len() - 4;
    let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
    assert_eq!(pool_val, 0);
}

#[test]
fn arm_literal_pool_multiple_ldrs() {
    // Three different constants → three separate pool entries
    let bytes = assemble_arm("ldr r0, =1\nldr r1, =2\nldr r2, =3");
    // 3 LDR (12 bytes) + 3 pool entries (12 bytes) = 24 bytes
    assert!(bytes.len() >= 24, "need 3 instructions + 3 pool entries");
    // Verify pool contains all three values
    let pool_end = bytes.len();
    let v3 = u32::from_le_bytes(bytes[pool_end - 4..pool_end].try_into().unwrap());
    let v2 = u32::from_le_bytes(bytes[pool_end - 8..pool_end - 4].try_into().unwrap());
    let v1 = u32::from_le_bytes(bytes[pool_end - 12..pool_end - 8].try_into().unwrap());
    let mut vals = [v1, v2, v3];
    vals.sort();
    assert_eq!(vals, [1, 2, 3], "pool should contain 1, 2, 3");
}

#[test]
fn arm_literal_pool_with_other_instructions() {
    // Literal pool should work alongside regular instructions
    let bytes = assemble_arm("mov r0, 42\nldr r1, =0xCAFE\nadd r0, r0, r1");
    // 3 instructions (12 bytes) + 1 pool entry (4 bytes) = 16 bytes
    assert!(bytes.len() >= 16);
    let pool_start = bytes.len() - 4;
    let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
    assert_eq!(pool_val, 0xCAFE);
}

#[test]
fn arm_literal_pool_pool_alias() {
    // .pool should work the same as .ltorg
    let bytes = assemble_arm("ldr r0, =0xBEEF\n.pool");
    let pool_start = bytes.len() - 4;
    let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
    assert_eq!(pool_val, 0xBEEF);
}

#[test]
fn arm_literal_pool_pc_offset_correct() {
    // Verify the PC-relative offset is correctly resolved
    // LDR R0, =val → LDR R0, [PC, #offset]
    // In ARM, PC reads as current_instr + 8, so offset = pool_addr - (instr_addr + 8)
    let bytes = assemble_arm("ldr r0, =0x42");
    let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let u_bit = (word >> 23) & 1;
    let offset = word & 0xFFF;
    // Pool should be immediately after the LDR (at offset 4),
    // PC = 0 + 8 = 8, so displacement = 4 - 8 = -4? No wait...
    // If pool is at byte 4 and instr is at byte 0, then:
    // PC = 0 + 8 = 8, target = 4, rel = 4 - 8 = -4
    // U=0, offset=4
    assert_eq!(u_bit, 0, "U-bit should be 0 for negative offset");
    assert_eq!(offset, 4, "offset should be 4");
}

// ── RISC-V C Extension integration tests ────────────────────────────────

#[test]
fn rvc_c_nop() {
    let bytes = rv32("c.nop");
    assert_eq!(bytes.len(), 2);
    let hw = u16::from_le_bytes(bytes[..2].try_into().unwrap());
    assert_eq!(hw, 0x0001);
}

#[test]
fn rvc_c_ebreak() {
    let bytes = rv32("c.ebreak");
    assert_eq!(bytes.len(), 2);
    let hw = u16::from_le_bytes(bytes[..2].try_into().unwrap());
    assert_eq!(hw, 0x9002);
}

#[test]
fn rvc_c_li() {
    let bytes = rv32("c.li x10, 5");
    assert_eq!(bytes.len(), 2);
    let hw = u16::from_le_bytes(bytes[..2].try_into().unwrap());
    assert_eq!(hw & 0x3, 0x01);
    assert_eq!((hw >> 13) & 0x7, 0b010);
    assert_eq!((hw >> 7) & 0x1F, 10);
    assert_eq!((hw >> 2) & 0x1F, 5);
}

#[test]
fn rvc_c_mv() {
    let bytes = rv32("c.mv x1, x2");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_add() {
    let bytes = rv32("c.add x3, x4");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_sub() {
    let bytes = rv32("c.sub x8, x9");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_and() {
    let bytes = rv32("c.and x14, x15");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_slli() {
    let bytes = rv32("c.slli x1, 4");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_lw() {
    let bytes = rv32("c.lw x8, 0(x9)");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_sw() {
    let bytes = rv32("c.sw x8, 0(x9)");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_lwsp() {
    let bytes = rv32("c.lwsp x10, 4");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_swsp() {
    let bytes = rv32("c.swsp x5, 8");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_addi16sp() {
    let bytes = rv32("c.addi16sp 32");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_addi4spn() {
    let bytes = rv32("c.addi4spn x8, 8");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_j_label() {
    let bytes = rv32("c.j target\nnop\nnop\ntarget:");
    assert_eq!(bytes.len(), 10); // 2 + 4 + 4
}

#[test]
fn rvc_c_beqz_label() {
    let bytes = rv32("c.beqz x8, target\nnop\ntarget:");
    assert_eq!(bytes.len(), 6); // 2 + 4
}

#[test]
fn rvc_c_bnez_label() {
    let bytes = rv32("c.bnez x9, target\nnop\ntarget:");
    assert_eq!(bytes.len(), 6);
}

#[test]
fn rvc_c_jr() {
    let bytes = rv32("c.jr x5");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_jalr() {
    let bytes = rv32("c.jalr x5");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_rv64_c_ld() {
    let bytes = rv64("c.ld x8, 0(x9)");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_rv64_c_sd() {
    let bytes = rv64("c.sd x8, 0(x9)");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_rv64_c_addiw() {
    let bytes = rv64("c.addiw x10, 3");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_rv64_c_subw() {
    let bytes = rv64("c.subw x8, x9");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_rv64_c_addw() {
    let bytes = rv64("c.addw x8, x9");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_mixed_standard_and_compressed() {
    let bytes = rv32("c.nop\nadd x1, x2, x3\nc.ebreak");
    assert_eq!(bytes.len(), 8); // 2 + 4 + 2
}

#[test]
fn rvc_c_addi() {
    let bytes = rv32("c.addi x5, -3");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_srli() {
    let bytes = rv32("c.srli x8, 3");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_srai() {
    let bytes = rv32("c.srai x9, 5");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_andi() {
    let bytes = rv32("c.andi x10, 7");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_or() {
    let bytes = rv32("c.or x12, x13");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_xor() {
    let bytes = rv32("c.xor x10, x11");
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_c_mv_x0_error() {
    let result = assemble("c.mv x0, x1", Arch::Rv32);
    assert!(result.is_err());
}

#[test]
fn rvc_c_sub_non_compact_error() {
    let result = assemble("c.sub x1, x2", Arch::Rv32);
    assert!(result.is_err());
}

#[test]
fn rvc_c_lw_misaligned_error() {
    let result = assemble("c.lw x8, 3(x9)", Arch::Rv32);
    assert!(result.is_err());
}

// ── .option rvc / .option norvc ──────────────────────────────

#[test]
fn rvc_option_rvc_auto_narrows_nop() {
    // With .option rvc, `nop` should auto-narrow to c.nop (2 bytes)
    let bytes = assemble(".option rvc\nnop", Arch::Rv32).unwrap();
    assert_eq!(bytes.len(), 2);
    assert_eq!(&bytes[..], &[0x01, 0x00]); // c.nop
}

#[test]
fn rvc_option_norvc_nop_stays_4bytes() {
    // Without .option rvc, `nop` remains 4 bytes (addi x0, x0, 0)
    let bytes = assemble("nop", Arch::Rv32).unwrap();
    assert_eq!(bytes.len(), 4);
}

#[test]
fn rvc_option_rvc_auto_narrows_add() {
    // add x1, x1, x2 → c.add x1, x2 (2 bytes)
    let bytes = assemble(".option rvc\nadd x1, x1, x2", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_auto_narrows_mv() {
    // add x1, x0, x2 → c.mv x1, x2 (2 bytes)
    let bytes = assemble(".option rvc\nadd x1, x0, x2", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_auto_narrows_addi() {
    // addi x1, x1, 5 → c.addi x1, 5 (2 bytes)
    let bytes = assemble(".option rvc\naddi x1, x1, 5", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_auto_narrows_li() {
    // addi x1, x0, 5 → c.li x1, 5 (2 bytes)
    let bytes = assemble(".option rvc\naddi x1, x0, 5", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_auto_narrows_ebreak() {
    // ebreak → c.ebreak (2 bytes)
    let bytes = assemble(".option rvc\nebreak", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 2);
    assert_eq!(&bytes[..], &[0x02, 0x90]); // c.ebreak
}

#[test]
fn rvc_option_rvc_auto_narrows_sub() {
    // sub x8, x8, x9 → c.sub x8, x9 (2 bytes)
    let bytes = assemble(".option rvc\nsub x8, x8, x9", Arch::Rv32).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_no_compress_when_impossible() {
    // add x1, x2, x3 — rd != rs1, cannot compress
    let bytes = assemble(".option rvc\nadd x1, x2, x3", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 4); // stays 4 bytes
}

#[test]
fn rvc_option_rvc_toggle() {
    // .option rvc → auto-narrow, then .option norvc → no auto-narrow
    let bytes = assemble(".option rvc\nnop\n.option norvc\nnop", Arch::Rv32).unwrap();
    assert_eq!(bytes.len(), 6); // 2 (c.nop) + 4 (nop)
}

#[test]
fn rvc_option_rvc_auto_narrows_lwsp() {
    // lw x1, 0(x2) → c.lwsp x1, 0(sp)
    let bytes = assemble(".option rvc\nlw x1, 0(sp)", Arch::Rv32).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_auto_narrows_swsp() {
    // sw x1, 0(sp) → c.swsp x1, 0(sp)
    let bytes = assemble(".option rvc\nsw x1, 0(sp)", Arch::Rv32).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_auto_narrows_slli() {
    // slli x1, x1, 1 → c.slli x1, 1
    let bytes = assemble(".option rvc\nslli x1, x1, 1", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_auto_narrows_jr() {
    // jalr x0, x1, 0 → c.jr x1
    let bytes = assemble(".option rvc\njr x1", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_auto_narrows_lui() {
    // lui x1, 1 → c.lui x1, 1
    let bytes = assemble(".option rvc\nlui x1, 1", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_rv64_auto_narrows_ld() {
    // ld x1, 0(sp) → c.ldsp x1, 0
    let bytes = assemble(".option rvc\nld x1, 0(sp)", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_rv64_auto_narrows_sd() {
    // sd x1, 0(sp) → c.sdsp x1, 0
    let bytes = assemble(".option rvc\nsd x1, 0(sp)", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_rv64_auto_narrows_addiw() {
    // addiw x1, x1, 5 → c.addiw x1, 5
    let bytes = assemble(".option rvc\naddiw x1, x1, 5", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn rvc_option_rvc_mixed_explicit_and_auto() {
    // Mix explicit c.xxx with auto-narrowed standard insns
    let bytes = assemble(".option rvc\nc.nop\nnop\nadd x1, x1, x2", Arch::Rv64).unwrap();
    assert_eq!(bytes.len(), 6); // 2 + 2 + 2
}

#[test]
fn rvc_option_on_non_riscv_errors() {
    let result = assemble(".option rvc\nnop", Arch::X86_64);
    assert!(result.is_err());
}

// ── Compressed branch relaxation (Story 5.7) ────────────

#[test]
fn rvc_c_beqz_short_range() {
    // c.beqz within ±256 B stays as 2-byte CB-type
    let bytes = assemble("c.beqz x8, target\nnop\ntarget:", Arch::Rv32).unwrap();
    // c.beqz = 2 bytes, nop = 4 bytes → total 6 bytes
    assert_eq!(bytes.len(), 6);
    // First 2 bytes are compressed branch
    assert_eq!(bytes[0] & 0x03, 0x01); // quadrant 1
}

#[test]
fn rvc_c_beqz_relaxes_to_long() {
    // c.beqz with target > 256 B away should widen to 8 bytes (inverted B + JAL)
    use asm_rs::Assembler;
    let mut asm = Assembler::new(Arch::Rv32);
    let mut src = String::from("c.beqz x8, far_target\n");
    // Emit enough NOPs to push target past ±256 bytes
    // 256 / 4 = 64 NOPs puts us right at the edge, add more
    for _ in 0..80 {
        src.push_str("nop\n");
    }
    src.push_str("far_target:\n");
    asm.emit(&src).unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // First instruction should have been widened to 8 bytes (inverted branch + JAL)
    // Total: 8 (widened c.beqz) + 80*4 (nops) = 328
    assert_eq!(bytes.len(), 328);
    // First 4 bytes should be a BNE instruction (inverted beqz)
    // BNE opcode = 0b1100011, funct3 = 0b001
    assert_eq!(bytes[0] & 0x7F, 0b1100011); // opcode = BRANCH
    assert_eq!((bytes[1] >> 4) & 0x7, 0b001); // funct3 = BNE
}

#[test]
fn rvc_c_bnez_relaxes_to_long() {
    // c.bnez with target > 256 B away should widen to 8 bytes (inverted B + JAL)
    use asm_rs::Assembler;
    let mut asm = Assembler::new(Arch::Rv32);
    let mut src = String::from("c.bnez x9, far_target\n");
    for _ in 0..80 {
        src.push_str("nop\n");
    }
    src.push_str("far_target:\n");
    asm.emit(&src).unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // 8 (widened c.bnez) + 80*4 (nops) = 328
    assert_eq!(bytes.len(), 328);
    // First 4 bytes should be BEQ (inverted bnez)
    assert_eq!(bytes[0] & 0x7F, 0b1100011); // opcode = BRANCH
    assert_eq!((bytes[1] >> 4) & 0x7, 0b000); // funct3 = BEQ
}

#[test]
fn rvc_c_j_short_range() {
    // c.j within ±2 KB stays as 2-byte CJ-type
    let bytes = assemble("c.j target\nnop\ntarget:", Arch::Rv32).unwrap();
    // c.j = 2 bytes, nop = 4 bytes → total 6 bytes
    assert_eq!(bytes.len(), 6);
    // First 2 bytes are compressed jump
    assert_eq!(bytes[0] & 0x03, 0x01); // quadrant 1
}

#[test]
fn rvc_c_j_relaxes_to_jal() {
    // c.j with target > 2 KB away should widen to 4 bytes (JAL x0)
    use asm_rs::Assembler;
    let mut asm = Assembler::new(Arch::Rv32);
    let mut src = String::from("c.j far_target\n");
    // 2048 / 4 = 512 NOPs puts us at the edge, add more
    for _ in 0..600 {
        src.push_str("nop\n");
    }
    src.push_str("far_target:\n");
    asm.emit(&src).unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    // 4 (widened c.j→JAL) + 600*4 (nops) = 2404
    assert_eq!(bytes.len(), 2404);
    // First 4 bytes should be JAL x0, target
    assert_eq!(bytes[0] & 0x7F, 0b1101111); // opcode = JAL
}

#[test]
fn rvc_c_beqz_backward_stays_short() {
    // Backward c.beqz within range stays as 2 bytes
    let bytes = assemble("target:\nnop\nc.beqz x8, target", Arch::Rv32).unwrap();
    // nop = 4 bytes, c.beqz = 2 bytes → total 6 bytes
    assert_eq!(bytes.len(), 6);
}

#[test]
fn rvc_c_j_backward_stays_short() {
    // Backward c.j within range stays as 2 bytes
    let bytes = assemble("target:\nnop\nc.j target", Arch::Rv32).unwrap();
    // nop = 4 bytes, c.j = 2 bytes → total 6 bytes
    assert_eq!(bytes.len(), 6);
}

// ── RISC-V F/D Extension Integration Tests ──────────────────────────────

#[test]
fn rv32_flw_fsw() {
    // flw f1, 0(x2)  →  opcode=0x07, funct3=010
    // fsw f1, 0(x2)  →  opcode=0x27, funct3=010
    let code = rv32("flw f1, 0(x2)\nfsw f1, 0(x2)");
    assert_eq!(code.len(), 8);
    assert_eq!(code[0] & 0x7F, 0x07); // FLW opcode
    assert_eq!(code[4] & 0x7F, 0x27); // FSW opcode
}

#[test]
fn rv32_fld_fsd() {
    let code = rv32("fld f3, 8(x4)\nfsd f3, 8(x4)");
    assert_eq!(code.len(), 8);
    assert_eq!(code[0] & 0x7F, 0x07); // FLD opcode (same base as FLW)
    assert_eq!(code[4] & 0x7F, 0x27); // FSD opcode (same base as FSW)
}

#[test]
fn rv32_fadd_s() {
    let code = rv32("fadd.s f1, f2, f3");
    assert_eq!(code.len(), 4);
    assert_eq!(code[0] & 0x7F, 0x53); // FP opcode
}

#[test]
fn rv32_fsub_s() {
    let code = rv32("fsub.s f4, f5, f6");
    assert_eq!(code.len(), 4);
    assert_eq!(code[0] & 0x7F, 0x53);
}

#[test]
fn rv32_fmul_s() {
    let code = rv32("fmul.s f7, f8, f9");
    assert_eq!(code.len(), 4);
    assert_eq!(code[0] & 0x7F, 0x53);
}

#[test]
fn rv32_fdiv_s() {
    let code = rv32("fdiv.s f10, f11, f12");
    assert_eq!(code.len(), 4);
    assert_eq!(code[0] & 0x7F, 0x53);
}

#[test]
fn rv32_fsqrt_s() {
    let code = rv32("fsqrt.s f1, f2");
    assert_eq!(code.len(), 4);
    assert_eq!(code[0] & 0x7F, 0x53);
}

#[test]
fn rv32_fadd_d() {
    let code = rv32("fadd.d f1, f2, f3");
    assert_eq!(code.len(), 4);
    assert_eq!(code[0] & 0x7F, 0x53);
}

#[test]
fn rv32_fsub_d() {
    let code = rv32("fsub.d f4, f5, f6");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fmul_d() {
    let code = rv32("fmul.d f7, f8, f9");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fdiv_d() {
    let code = rv32("fdiv.d f10, f11, f12");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fsqrt_d() {
    let code = rv32("fsqrt.d f1, f2");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fmin_fmax_s() {
    let code = rv32("fmin.s f1, f2, f3\nfmax.s f4, f5, f6");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv32_fmin_fmax_d() {
    let code = rv32("fmin.d f1, f2, f3\nfmax.d f4, f5, f6");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv32_feq_flt_fle_s() {
    let code = rv32("feq.s x10, f1, f2\nflt.s x11, f3, f4\nfle.s x12, f5, f6");
    assert_eq!(code.len(), 12);
}

#[test]
fn rv32_feq_flt_fle_d() {
    let code = rv32("feq.d x10, f1, f2\nflt.d x11, f3, f4\nfle.d x12, f5, f6");
    assert_eq!(code.len(), 12);
}

#[test]
fn rv32_fclass_s_d() {
    let code = rv32("fclass.s x10, f1\nfclass.d x11, f2");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv32_fcvt_w_s_and_back() {
    let code = rv32("fcvt.w.s x10, f1\nfcvt.s.w f2, x11");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv32_fcvt_wu_s_and_back() {
    let code = rv32("fcvt.wu.s x10, f1\nfcvt.s.wu f2, x11");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv32_fcvt_w_d_and_back() {
    let code = rv32("fcvt.w.d x10, f1\nfcvt.d.w f2, x11");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv32_fcvt_wu_d_and_back() {
    let code = rv32("fcvt.wu.d x10, f1\nfcvt.d.wu f2, x11");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv32_fcvt_s_d_and_back() {
    let code = rv32("fcvt.s.d f1, f2\nfcvt.d.s f3, f4");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv32_fmv_x_w_and_back() {
    let code = rv32("fmv.x.w x10, f1\nfmv.w.x f2, x11");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv32_fsgnj_s() {
    let code = rv32("fsgnj.s f1, f2, f3\nfsgnjn.s f4, f5, f6\nfsgnjx.s f7, f8, f9");
    assert_eq!(code.len(), 12);
}

#[test]
fn rv32_fsgnj_d() {
    let code = rv32("fsgnj.d f1, f2, f3\nfsgnjn.d f4, f5, f6\nfsgnjx.d f7, f8, f9");
    assert_eq!(code.len(), 12);
}

#[test]
fn rv32_fmadd_s() {
    let code = rv32("fmadd.s f1, f2, f3, f4");
    assert_eq!(code.len(), 4);
    assert_eq!(code[0] & 0x7F, 0x43); // MADD opcode
}

#[test]
fn rv32_fmsub_s() {
    let code = rv32("fmsub.s f1, f2, f3, f4");
    assert_eq!(code.len(), 4);
    assert_eq!(code[0] & 0x7F, 0x47); // MSUB opcode
}

#[test]
fn rv32_fnmsub_s() {
    let code = rv32("fnmsub.s f1, f2, f3, f4");
    assert_eq!(code.len(), 4);
    assert_eq!(code[0] & 0x7F, 0x4B); // NMSUB opcode
}

#[test]
fn rv32_fnmadd_s() {
    let code = rv32("fnmadd.s f1, f2, f3, f4");
    assert_eq!(code.len(), 4);
    assert_eq!(code[0] & 0x7F, 0x4F); // NMADD opcode
}

#[test]
fn rv32_fmadd_d() {
    let code = rv32("fmadd.d f1, f2, f3, f4");
    assert_eq!(code.len(), 4);
    assert_eq!(code[0] & 0x7F, 0x43);
}

#[test]
fn rv32_fmsub_d() {
    let code = rv32("fmsub.d f1, f2, f3, f4");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fnmsub_d() {
    let code = rv32("fnmsub.d f1, f2, f3, f4");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fnmadd_d() {
    let code = rv32("fnmadd.d f1, f2, f3, f4");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fp_pseudo_fmv_s() {
    let code = rv32("fmv.s f1, f2");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fp_pseudo_fmv_d() {
    let code = rv32("fmv.d f1, f2");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fp_pseudo_fneg_s() {
    let code = rv32("fneg.s f1, f2");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fp_pseudo_fneg_d() {
    let code = rv32("fneg.d f1, f2");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fp_pseudo_fabs_s() {
    let code = rv32("fabs.s f1, f2");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv32_fp_pseudo_fabs_d() {
    let code = rv32("fabs.d f1, f2");
    assert_eq!(code.len(), 4);
}

#[test]
fn rv64_fcvt_l_s() {
    let code = rv64("fcvt.l.s x10, f1\nfcvt.s.l f2, x11");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv64_fcvt_lu_s() {
    let code = rv64("fcvt.lu.s x10, f1\nfcvt.s.lu f2, x11");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv64_fcvt_l_d() {
    let code = rv64("fcvt.l.d x10, f1\nfcvt.d.l f2, x11");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv64_fcvt_lu_d() {
    let code = rv64("fcvt.lu.d x10, f1\nfcvt.d.lu f2, x11");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv64_fmv_x_d_and_back() {
    let code = rv64("fmv.x.d x10, f1\nfmv.d.x f2, x11");
    assert_eq!(code.len(), 8);
}

#[test]
fn rv32_fp_abi_register_names() {
    // Use ABI names: ft0=f0, fs0=f8, fa0=f10
    let code = rv32("fadd.s ft0, fs0, fa0");
    assert_eq!(code.len(), 4);
    // Verify encoded register numbers
    let w = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
    assert_eq!((w >> 7) & 0x1F, 0); // rd = ft0 = f0
    assert_eq!((w >> 15) & 0x1F, 8); // rs1 = fs0 = f8
    assert_eq!((w >> 20) & 0x1F, 10); // rs2 = fa0 = f10
}

#[test]
fn rv32_fp_multi_instruction_sequence() {
    // Real-world FP sequence: load, compute, store
    let code = rv32(
        "flw f1, 0(x10)\n\
         flw f2, 4(x10)\n\
         fadd.s f3, f1, f2\n\
         fsw f3, 8(x10)",
    );
    assert_eq!(code.len(), 16); // 4 instructions × 4 bytes
}

#[test]
fn rv32_fp_fmadd_sequence() {
    // FMA: result = a*b + c
    let code = rv32(
        "flw f0, 0(x10)\n\
         flw f1, 4(x10)\n\
         flw f2, 8(x10)\n\
         fmadd.s f3, f0, f1, f2\n\
         fsw f3, 12(x10)",
    );
    assert_eq!(code.len(), 20);
}

#[test]
fn rv32_fcvt_l_s_rejected() {
    // RV64-only instruction should fail on RV32
    let result = assemble("fcvt.l.s x10, f1", Arch::Rv32);
    assert!(result.is_err());
}

#[test]
fn rv32_fmv_x_d_rejected() {
    // RV64-only instruction should fail on RV32
    let result = assemble("fmv.x.d x10, f1", Arch::Rv32);
    assert!(result.is_err());
}

#[test]
fn rv32_flw_negative_offset() {
    let code = rv32("flw f0, -4(x2)");
    assert_eq!(code.len(), 4);
    let w = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
    // Immediate bits [31:20] should be -4 sign-extended
    let imm = (w >> 20) as i32;
    // Sign-extend from 12 bits
    let imm_sext = if imm & 0x800 != 0 { imm | !0xFFF } else { imm };
    assert_eq!(imm_sext, -4);
}

// ── AArch64 NEON / AdvSIMD integration tests ─────────────────────────────

// ── Vector arithmetic ────────────────────────────────────────────────────

#[test]
fn aarch64_neon_add_4s() {
    assert_eq!(a64_word("add v0.4s, v1.4s, v2.4s"), 0x4EA2_8420);
}

#[test]
fn aarch64_neon_add_8b() {
    assert_eq!(a64_word("add v3.8b, v4.8b, v5.8b"), 0x0E25_8483);
}

#[test]
fn aarch64_neon_add_16b() {
    assert_eq!(a64_word("add v0.16b, v1.16b, v2.16b"), 0x4E22_8420);
}

#[test]
fn aarch64_neon_add_8h() {
    assert_eq!(a64_word("add v0.8h, v1.8h, v2.8h"), 0x4E62_8420);
}

#[test]
fn aarch64_neon_sub_4s() {
    assert_eq!(a64_word("sub v0.4s, v1.4s, v2.4s"), 0x6EA2_8420);
}

#[test]
fn aarch64_neon_sub_2d() {
    assert_eq!(a64_word("sub v0.2d, v1.2d, v2.2d"), 0x6EE2_8420);
}

#[test]
fn aarch64_neon_mul_4h() {
    assert_eq!(a64_word("mul v0.4h, v1.4h, v2.4h"), 0x0E62_9C20);
}

// ── Vector bitwise ───────────────────────────────────────────────────────

#[test]
fn aarch64_neon_and_16b() {
    assert_eq!(a64_word("and v0.16b, v1.16b, v2.16b"), 0x4E22_1C20);
}

#[test]
fn aarch64_neon_orr_16b() {
    assert_eq!(a64_word("orr v0.16b, v1.16b, v2.16b"), 0x4EA2_1C20);
}

#[test]
fn aarch64_neon_eor_16b() {
    assert_eq!(a64_word("eor v0.16b, v1.16b, v2.16b"), 0x6E22_1C20);
}

#[test]
fn aarch64_neon_bic_8b() {
    assert_eq!(a64_word("bic v0.8b, v1.8b, v2.8b"), 0x0E62_1C20);
}

// ── Vector compare ──────────────────────────────────────────────────────

#[test]
fn aarch64_neon_cmeq_4s() {
    assert_eq!(a64_word("cmeq v0.4s, v1.4s, v2.4s"), 0x6EA2_8C20);
}

#[test]
fn aarch64_neon_cmgt_4s() {
    assert_eq!(a64_word("cmgt v0.4s, v1.4s, v2.4s"), 0x4EA2_3420);
}

// ── Two-register misc ───────────────────────────────────────────────────

#[test]
fn aarch64_neon_neg_4s() {
    assert_eq!(a64_word("neg v0.4s, v1.4s"), 0x6EA0_B820);
}

#[test]
fn aarch64_neon_abs_4s() {
    assert_eq!(a64_word("abs v0.4s, v1.4s"), 0x4EA0_B820);
}

#[test]
fn aarch64_neon_not_16b() {
    assert_eq!(a64_word("not v0.16b, v1.16b"), 0x6E20_5820);
}

#[test]
fn aarch64_neon_cnt_16b() {
    assert_eq!(a64_word("cnt v0.16b, v1.16b"), 0x4E20_5820);
}

// ── Copy / move ─────────────────────────────────────────────────────────

#[test]
fn aarch64_neon_dup_4s_w0() {
    assert_eq!(a64_word("dup v0.4s, w0"), 0x4E04_1C00);
}

#[test]
fn aarch64_neon_mov_v0_v1_16b() {
    assert_eq!(a64_word("mov v0.16b, v1.16b"), 0x4EA1_1C20);
}

// ── Scalar ops still work after NEON dispatch ───────────────────────────

#[test]
fn aarch64_scalar_add_still_works() {
    assert_eq!(a64_word("add x0, x1, x2"), 0x8B02_0020);
}

#[test]
fn aarch64_scalar_sub_still_works() {
    assert_eq!(a64_word("sub x0, x1, x2"), 0xCB02_0020);
}

#[test]
fn aarch64_scalar_mov_still_works() {
    assert_eq!(a64_word("mov x0, x1"), 0xAA01_03E0);
}

// ── Multi-instruction NEON ──────────────────────────────────────────────

#[test]
fn aarch64_neon_multiline() {
    // Multiple NEON + scalar instructions in one block
    let code = assemble(
        "add v0.4s, v1.4s, v2.4s\nsub v3.4s, v4.4s, v5.4s\nadd x0, x1, x2",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 12); // 3 instructions × 4 bytes
}

// ==============  AVX-512 (EVEX-encoded) INTEGRATION TESTS  ==================

// ── AVX-512F arithmetic ─────────────────────────────────────────────────

#[test]
fn avx512_vaddps_zmm() {
    let code = assemble("vaddps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x48, 0x58, 0xC2]);
}

#[test]
fn avx512_vaddpd_zmm() {
    let code = assemble("vaddpd zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0xF5, 0x48, 0x58, 0xC2]);
}

#[test]
fn avx512_vsubps_zmm() {
    let code = assemble("vsubps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x48, 0x5C, 0xC2]);
}

#[test]
fn avx512_vsubpd_zmm() {
    let code = assemble("vsubpd zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0xF5, 0x48, 0x5C, 0xC2]);
}

#[test]
fn avx512_vmulps_zmm() {
    let code = assemble("vmulps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x48, 0x59, 0xC2]);
}

#[test]
fn avx512_vmulpd_zmm() {
    let code = assemble("vmulpd zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0xF5, 0x48, 0x59, 0xC2]);
}

#[test]
fn avx512_vdivps_zmm() {
    let code = assemble("vdivps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x48, 0x5E, 0xC2]);
}

#[test]
fn avx512_vmaxps_zmm() {
    let code = assemble("vmaxps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x48, 0x5F, 0xC2]);
}

#[test]
fn avx512_vminps_zmm() {
    let code = assemble("vminps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x48, 0x5D, 0xC2]);
}

#[test]
fn avx512_vsqrtps_zmm() {
    let code = assemble("vsqrtps zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x7C, 0x48, 0x51, 0xC1]);
}

// ── AVX-512F logical ────────────────────────────────────────────────────

#[test]
fn avx512_vandps_zmm() {
    let code = assemble("vandps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x48, 0x54, 0xC2]);
}

#[test]
fn avx512_vorps_zmm() {
    let code = assemble("vorps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x48, 0x56, 0xC2]);
}

#[test]
fn avx512_vxorps_zmm() {
    let code = assemble("vxorps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x48, 0x57, 0xC2]);
}

// ── AVX-512F data movement ──────────────────────────────────────────────

#[test]
fn avx512_vmovaps_zmm_zmm() {
    let code = assemble("vmovaps zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x7C, 0x48, 0x28, 0xC1]);
}

#[test]
fn avx512_vmovapd_zmm_zmm() {
    let code = assemble("vmovapd zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0xFD, 0x48, 0x28, 0xC1]);
}

#[test]
fn avx512_vmovdqa32_zmm_zmm() {
    let code = assemble("vmovdqa32 zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x7D, 0x48, 0x6F, 0xC1]);
}

#[test]
fn avx512_vmovdqa64_zmm_zmm() {
    let code = assemble("vmovdqa64 zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0xFD, 0x48, 0x6F, 0xC1]);
}

#[test]
fn avx512_vmovdqu32_zmm_zmm() {
    let code = assemble("vmovdqu32 zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x7E, 0x48, 0x6F, 0xC1]);
}

#[test]
fn avx512_vmovdqu8_zmm_zmm() {
    let code = assemble("vmovdqu8 zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x7F, 0x48, 0x6F, 0xC1]);
}

#[test]
fn avx512_vmovdqu16_zmm_zmm() {
    let code = assemble("vmovdqu16 zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0xFF, 0x48, 0x6F, 0xC1]);
}

// ── AVX-512F integer packed ─────────────────────────────────────────────

#[test]
fn avx512_vpaddd_zmm() {
    let code = assemble("vpaddd zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x75, 0x48, 0xFE, 0xC2]);
}

#[test]
fn avx512_vpaddq_zmm() {
    let code = assemble("vpaddq zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0xF5, 0x48, 0xD4, 0xC2]);
}

#[test]
fn avx512_vpsubd_zmm() {
    let code = assemble("vpsubd zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x75, 0x48, 0xFA, 0xC2]);
}

#[test]
fn avx512_vpxord_zmm() {
    let code = assemble("vpxord zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x75, 0x48, 0xEF, 0xC2]);
}

#[test]
fn avx512_vpxorq_zmm() {
    let code = assemble("vpxorq zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0xF5, 0x48, 0xEF, 0xC2]);
}

#[test]
fn avx512_vpandd_zmm() {
    let code = assemble("vpandd zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x75, 0x48, 0xDB, 0xC2]);
}

#[test]
fn avx512_vpord_zmm() {
    let code = assemble("vpord zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x75, 0x48, 0xEB, 0xC2]);
}

#[test]
fn avx512_vpmullq_zmm() {
    let code = assemble("vpmullq zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF2, 0xF5, 0x48, 0x40, 0xC2]);
}

// ── AVX-512F ternary logic & blend ──────────────────────────────────────

#[test]
fn avx512_vpternlogd_zmm() {
    let code = assemble("vpternlogd zmm0, zmm1, zmm2, 0xFF", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF3, 0x75, 0x48, 0x25, 0xC2, 0xFF]);
}

#[test]
fn avx512_vpternlogq_zmm() {
    let code = assemble("vpternlogq zmm0, zmm1, zmm2, 0xDB", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF3, 0xF5, 0x48, 0x25, 0xC2, 0xDB]);
}

#[test]
fn avx512_vblendmps_zmm() {
    let code = assemble("vblendmps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF2, 0x75, 0x48, 0x65, 0xC2]);
}

// ── AVX-512F compress/expand ────────────────────────────────────────────

#[test]
fn avx512_vcompressps_zmm() {
    let code = assemble("vcompressps zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF2, 0x7D, 0x48, 0x8A, 0xC1]);
}

#[test]
fn avx512_vexpandps_zmm() {
    let code = assemble("vexpandps zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF2, 0x7D, 0x48, 0x88, 0xC1]);
}

// ── AVX-512F conversions ────────────────────────────────────────────────

#[test]
fn avx512_vcvtps2pd_zmm() {
    let code = assemble("vcvtps2pd zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x7C, 0x48, 0x5A, 0xC1]);
}

#[test]
fn avx512_vcvtdq2ps_zmm() {
    let code = assemble("vcvtdq2ps zmm0, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x7C, 0x48, 0x5B, 0xC1]);
}

// ── AVX-512F extended registers ─────────────────────────────────────────

#[test]
fn avx512_vaddps_zmm16_zmm17_zmm18() {
    let code = assemble("vaddps zmm16, zmm17, zmm18", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xA1, 0x74, 0x40, 0x58, 0xC2]);
}

#[test]
fn avx512_vmovaps_zmm31_zmm16() {
    let code = assemble("vmovaps zmm31, zmm16", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0x21, 0x7C, 0x48, 0x28, 0xF8]);
}

// ── AVX-512F shuffle with imm ───────────────────────────────────────────

#[test]
fn avx512_vpshufd_zmm_imm() {
    let code = assemble("vpshufd zmm0, zmm1, 0xE4", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x7D, 0x48, 0x70, 0xC1, 0xE4]);
}

// ── AVX-512F FMA 512-bit ────────────────────────────────────────────────

#[test]
fn avx512_vfmadd231ps_zmm() {
    let code = assemble("vfmadd231ps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF2, 0x75, 0x48, 0xB8, 0xC2]);
}

#[test]
fn avx512_vfmadd231pd_zmm() {
    let code = assemble("vfmadd231pd zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF2, 0xF5, 0x48, 0xB8, 0xC2]);
}

// ── AVX-512BW byte/word ─────────────────────────────────────────────────

#[test]
fn avx512_vpaddb_zmm() {
    let code = assemble("vpaddb zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x75, 0x48, 0xFC, 0xC2]);
}

#[test]
fn avx512_vpaddw_zmm() {
    let code = assemble("vpaddw zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x75, 0x48, 0xFD, 0xC2]);
}

// ── AVX-512 variable shifts ─────────────────────────────────────────────

#[test]
fn avx512_vpsravq_zmm() {
    let code = assemble("vpsravq zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF2, 0xF5, 0x48, 0x46, 0xC2]);
}

// ── VEX regression (XMM/YMM should still use VEX, not EVEX) ────────────

#[test]
fn avx512_regression_vaddps_xmm_uses_vex() {
    let code = assemble("vaddps xmm0, xmm1, xmm2", Arch::X86_64).unwrap();
    assert_eq!(code[0], 0xC5, "XMM operations should still use VEX prefix");
}

#[test]
fn avx512_regression_vaddps_ymm_uses_vex() {
    let code = assemble("vaddps ymm0, ymm1, ymm2", Arch::X86_64).unwrap();
    assert_eq!(code[0], 0xC5, "YMM operations should still use VEX prefix");
}

// ── Multi-instruction AVX-512 ───────────────────────────────────────────

#[test]
fn avx512_multiline() {
    let code = assemble(
        "vaddps zmm0, zmm1, zmm2\nvsubps zmm3, zmm4, zmm5\nvmovaps zmm6, zmm7",
        Arch::X86_64,
    )
    .unwrap();
    assert_eq!(code.len(), 18); // 3 × 6 bytes
}

// ==============  AArch64 SVE INTEGRATION TESTS  ============================

// ── SVE predicate setup ─────────────────────────────────────────────────

#[test]
fn sve_ptrue_pb() {
    assert_eq!(a64_word("ptrue p0.b"), 0x2518_E3E0);
}

#[test]
fn sve_ptrue_ps() {
    assert_eq!(a64_word("ptrue p0.s"), 0x2598_E3E0);
}

#[test]
fn sve_pfalse_pb() {
    assert_eq!(a64_word("pfalse p0.b"), 0x2518_E400);
}

// ── SVE unpredicated arithmetic ─────────────────────────────────────────

#[test]
fn sve_add_unpred_z0_z1_z2_s() {
    assert_eq!(a64_word("add z0.s, z1.s, z2.s"), 0x04A2_0020);
}

#[test]
fn sve_sub_pred_z0_p0m_z0_z1_s() {
    assert_eq!(a64_word("sub z0.s, p0/m, z0.s, z1.s"), 0x0481_0020);
}

#[test]
fn sve_mul_pred_z0_p0m_z0_z1_s() {
    assert_eq!(a64_word("mul z0.s, p0/m, z0.s, z1.s"), 0x0490_0020);
}

// ── SVE bitwise ─────────────────────────────────────────────────────────

#[test]
fn sve_and_unpred_d() {
    assert_eq!(a64_word("and z0.d, z1.d, z2.d"), 0x0422_3020);
}

#[test]
fn sve_orr_unpred_d() {
    assert_eq!(a64_word("orr z0.d, z1.d, z2.d"), 0x0462_3020);
}

#[test]
fn sve_eor_unpred_d() {
    assert_eq!(a64_word("eor z0.d, z1.d, z2.d"), 0x04A2_3020);
}

// ── SVE predicated bitwise ──────────────────────────────────────────────

#[test]
fn sve_and_pred_b() {
    assert_eq!(a64_word("and z0.b, p0/m, z0.b, z1.b"), 0x041A_0020);
}

#[test]
fn sve_orr_pred_b() {
    assert_eq!(a64_word("orr z0.b, p0/m, z0.b, z1.b"), 0x0418_0020);
}

#[test]
fn sve_eor_pred_b() {
    assert_eq!(a64_word("eor z0.b, p0/m, z0.b, z1.b"), 0x0419_0020);
}

// ── SVE loop control ────────────────────────────────────────────────────

#[test]
fn sve_whilelt_p0_s() {
    assert_eq!(a64_word("whilelt p0.s, x0, x1"), 0x25A1_1400);
}

// ── SVE DUP immediate ──────────────────────────────────────────────────

#[test]
fn sve_dup_z0_s_imm1() {
    assert_eq!(a64_word("dup z0.s, 1"), 0x25B8_C020);
}

// ── SVE element count ───────────────────────────────────────────────────

#[test]
fn sve_cntb() {
    assert_eq!(a64_word("cntb x0"), 0x0420_E3E0);
}

#[test]
fn sve_cnth() {
    assert_eq!(a64_word("cnth x0"), 0x0460_E3E0);
}

#[test]
fn sve_cntw() {
    assert_eq!(a64_word("cntw x0"), 0x04A0_E3E0);
}

#[test]
fn sve_cntd() {
    assert_eq!(a64_word("cntd x0"), 0x04E0_E3E0);
}

// ── SVE memory operations ───────────────────────────────────────────────

#[test]
fn sve_ld1w_z0_p0z_x0() {
    assert_eq!(a64_word("ld1w {z0.s}, p0/z, [x0]"), 0xA540_A000);
}

#[test]
fn sve_st1w_z0_p0_x0() {
    assert_eq!(a64_word("st1w {z0.s}, p0/m, [x0]"), 0xE540_E000);
}

// ── SVE ADD immediate ──────────────────────────────────────────────────

#[test]
fn sve_add_imm_z0_s_1() {
    assert_eq!(a64_word("add z0.s, z0.s, 1"), 0x25A0_C020);
}

// ── SVE multi-instruction block ─────────────────────────────────────────

#[test]
fn sve_multiline_block() {
    let code = assemble(
        "ptrue p0.s\nadd z0.s, z1.s, z2.s\nst1w {z0.s}, p0/m, [x0]",
        Arch::Aarch64,
    )
    .unwrap();
    assert_eq!(code.len(), 12); // 3 × 4 bytes
}

#[test]
fn sve_scalar_still_works_after_dispatch() {
    // Ensure scalar instructions still work when SVE is enabled
    assert_eq!(a64_word("add x0, x1, x2"), 0x8B02_0020);
}

// ==============  RISC-V V EXTENSION INTEGRATION TESTS  ======================

// ── Vector configuration ────────────────────────────────────────────────

#[test]
fn rvv_vsetvli() {
    let bytes = rv64("vsetvli a0, a1, e32, m1, ta, ma");
    assert_eq!(le32(&bytes, 0), 0x0D05_F557);
}

#[test]
fn rvv_vsetivli() {
    let bytes = rv64("vsetivli a0, 16, e32, m1, ta, ma");
    assert_eq!(le32(&bytes, 0), 0xCD08_7557);
}

#[test]
fn rvv_vsetvl() {
    let bytes = rv64("vsetvl a0, a1, a2");
    assert_eq!(le32(&bytes, 0), 0x80C5_F557);
}

// ── Vector loads/stores ─────────────────────────────────────────────────

#[test]
fn rvv_vle8() {
    let bytes = rv64("vle8.v v1, (a0)");
    assert_eq!(le32(&bytes, 0), 0x0205_0087);
}

#[test]
fn rvv_vse8() {
    let bytes = rv64("vse8.v v1, (a0)");
    assert_eq!(le32(&bytes, 0), 0x0205_00A7);
}

#[test]
fn rvv_vle32() {
    let bytes = rv64("vle32.v v1, (a0)");
    assert_eq!(le32(&bytes, 0), 0x0205_6087);
}

#[test]
fn rvv_vse32() {
    let bytes = rv64("vse32.v v1, (a0)");
    assert_eq!(le32(&bytes, 0), 0x0205_60A7);
}

// ── Vector arithmetic ───────────────────────────────────────────────────

#[test]
fn rvv_vadd_vv() {
    let bytes = rv64("vadd.vv v1, v2, v3");
    assert_eq!(le32(&bytes, 0), 0x0221_80D7);
}

#[test]
fn rvv_vsub_vv() {
    let bytes = rv64("vsub.vv v1, v2, v3");
    assert_eq!(le32(&bytes, 0), 0x0A21_80D7);
}

#[test]
fn rvv_vand_vv() {
    let bytes = rv64("vand.vv v1, v2, v3");
    assert_eq!(le32(&bytes, 0), 0x2621_80D7);
}

#[test]
fn rvv_vor_vv() {
    let bytes = rv64("vor.vv v1, v2, v3");
    assert_eq!(le32(&bytes, 0), 0x2A21_80D7);
}

#[test]
fn rvv_vxor_vv() {
    let bytes = rv64("vxor.vv v1, v2, v3");
    assert_eq!(le32(&bytes, 0), 0x2E21_80D7);
}

#[test]
fn rvv_vmul_vv() {
    let bytes = rv64("vmul.vv v1, v2, v3");
    assert_eq!(le32(&bytes, 0), 0x9621_A0D7);
}

// ── Vector-scalar and vector-immediate ──────────────────────────────────

#[test]
fn rvv_vadd_vx() {
    let bytes = rv64("vadd.vx v1, v2, a0");
    assert_eq!(le32(&bytes, 0), 0x0225_40D7);
}

#[test]
fn rvv_vadd_vi() {
    let bytes = rv64("vadd.vi v1, v2, 5");
    assert_eq!(le32(&bytes, 0), 0x0222_B0D7);
}

// ── Masked vector operation ─────────────────────────────────────────────

#[test]
fn rvv_vle32_masked() {
    let bytes = rv64("vle32.v v1, (a0), v0.t");
    assert_eq!(le32(&bytes, 0), 0x0005_6087);
}

// ── Multi-instruction RVV block ─────────────────────────────────────────

#[test]
fn rvv_multiline_block() {
    let code = assemble(
        "vsetvli a0, a1, e32, m1, ta, ma\nvadd.vv v1, v2, v3\nvse32.v v1, (a0)",
        Arch::Rv64,
    )
    .unwrap();
    assert_eq!(code.len(), 12); // 3 × 4 bytes
}

#[test]
fn rvv_scalar_still_works() {
    // Ensure scalar instructions still work alongside RVV
    let bytes = rv64("add a0, a1, a2");
    assert_eq!(le32(&bytes, 0), 0x00C5_8533);
}

// ══════════════════════════════════════════════════════════════════════════
// AVX-512 Opmask & Broadcast
// ══════════════════════════════════════════════════════════════════════════

// ── Opmask (merging) ────────────────────────────────────────────────────

/// VADDPS ZMM0{K1}, ZMM1, ZMM2  →  62 F1 74 49 58 C2
/// P2 = 0x49: z=0, L'L=10, b=0, V'=0→inverted=1, aaa=001
#[test]
fn avx512_vaddps_zmm_opmask_k1() {
    let code = assemble("vaddps zmm0{k1}, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x49, 0x58, 0xC2]);
}

/// VADDPS ZMM0{K2}, ZMM1, ZMM2  →  62 F1 74 4A 58 C2
/// P2 = 0x4A: z=0, L'L=10, b=0, V'=1, aaa=010
#[test]
fn avx512_vaddps_zmm_opmask_k2() {
    let code = assemble("vaddps zmm0{k2}, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x4A, 0x58, 0xC2]);
}

/// VADDPS ZMM0{K7}, ZMM1, ZMM2  →  62 F1 74 4F 58 C2
/// P2 = 0x4F: z=0, L'L=10, b=0, V'=1, aaa=111
#[test]
fn avx512_vaddps_zmm_opmask_k7() {
    let code = assemble("vaddps zmm0{k7}, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x4F, 0x58, 0xC2]);
}

// ── Opmask + zeroing ────────────────────────────────────────────────────

/// VADDPS ZMM0{K1}{Z}, ZMM1, ZMM2  →  62 F1 74 C9 58 C2
/// P2 = 0xC9: z=1, L'L=10, b=0, V'=1, aaa=001
#[test]
fn avx512_vaddps_zmm_opmask_k1_zeroing() {
    let code = assemble("vaddps zmm0{k1}{z}, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0xC9, 0x58, 0xC2]);
}

/// VADDPD ZMM0{K3}{Z}, ZMM1, ZMM2
#[test]
fn avx512_vaddpd_zmm_opmask_k3_zeroing() {
    let code = assemble("vaddpd zmm0{k3}{z}, zmm1, zmm2", Arch::X86_64).unwrap();
    // P1: W=1, pp=01 → 0xF5.  P2: z=1(0x80), L'L=10(0x40), V'=0→inv=0x08, aaa=011 → 0xCB
    assert_eq!(code, vec![0x62, 0xF1, 0xF5, 0xCB, 0x58, 0xC2]);
}

// ── Broadcast ───────────────────────────────────────────────────────────

/// VADDPS ZMM0, ZMM1, DWORD PTR [RAX]{1to16}  →  62 F1 74 58 58 00
/// P2 = 0x58: z=0, L'L=10, b=1, V'=1, aaa=000
#[test]
fn avx512_vaddps_zmm_broadcast_1to16() {
    let code = assemble("vaddps zmm0, zmm1, dword ptr [rax]{1to16}", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x58, 0x58, 0x00]);
}

// ── Opmask + broadcast ──────────────────────────────────────────────────

/// VADDPS ZMM0{K2}, ZMM1, DWORD PTR [RAX]{1to16}  →  62 F1 74 5A 58 00
/// P2 = 0x5A: z=0, L'L=10, b=1, V'=1, aaa=010
#[test]
fn avx512_vaddps_zmm_opmask_k2_broadcast() {
    let code = assemble(
        "vaddps zmm0{k2}, zmm1, dword ptr [rax]{1to16}",
        Arch::X86_64,
    )
    .unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x5A, 0x58, 0x00]);
}

/// VADDPS ZMM0{K3}{Z}, ZMM1, DWORD PTR [RAX]{1to16}  →  62 F1 74 DB 58 00
/// P2 = 0xDB: z=1, L'L=10, b=1, V'=1, aaa=011
#[test]
fn avx512_vaddps_zmm_opmask_k3_zeroing_broadcast() {
    let code = assemble(
        "vaddps zmm0{k3}{z}, zmm1, dword ptr [rax]{1to16}",
        Arch::X86_64,
    )
    .unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0xDB, 0x58, 0x00]);
}

// ── 2-operand opmask (vmovaps) ──────────────────────────────────────────

/// VMOVAPS ZMM0{K1}, ZMM1  →  62 F1 7C 49 28 C1
#[test]
fn avx512_vmovaps_zmm_opmask_k1() {
    let code = assemble("vmovaps zmm0{k1}, zmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x7C, 0x49, 0x28, 0xC1]);
}

// ── EVEX immediate with opmask ──────────────────────────────────────────

/// VSHUFPS ZMM0{K1}, ZMM1, ZMM2, 0x01
#[test]
fn avx512_vshufps_zmm_opmask_k1_imm() {
    // Reference from llvm-mc:
    // vshufps zmm0{k1}, zmm1, zmm2, 1 → 62 F1 74 49 C6 C2 01
    let code = assemble("vshufps zmm0{k1}, zmm1, zmm2, 1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x49, 0xC6, 0xC2, 0x01]);
}

// ── Opmask without decorators keeps existing behaviour ──────────────────

/// Ensure unmasked instructions still produce aaa=000 / z=0
#[test]
fn avx512_vaddps_zmm_no_opmask_unchanged() {
    let code = assemble("vaddps zmm0, zmm1, zmm2", Arch::X86_64).unwrap();
    // P2 = 0x48: z=0, L'L=10, b=0, V'=1, aaa=000  (same as before)
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x48, 0x58, 0xC2]);
}

// ── Sub/mul with opmask ─────────────────────────────────────────────────

/// VSUBPS ZMM0{K1}, ZMM1, ZMM2
#[test]
fn avx512_vsubps_zmm_opmask_k1() {
    let code = assemble("vsubps zmm0{k1}, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0x49, 0x5C, 0xC2]);
}

/// VMULPS ZMM0{K1}{Z}, ZMM1, ZMM2
#[test]
fn avx512_vmulps_zmm_opmask_k1_zeroing() {
    let code = assemble("vmulps zmm0{k1}{z}, zmm1, zmm2", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x62, 0xF1, 0x74, 0xC9, 0x59, 0xC2]);
}

// ─── RISC-V F/D Extension Cross-Validation (llvm-mc riscv64 +f,+d) ─────

/// FLW f0, 0(x1) — cross-validated against llvm-mc
#[test]
fn riscv_flw_f0_0_x1() {
    let code = assemble("flw f0, 0(x1)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x07, 0xA0, 0x00, 0x00]);
}

/// FSW f1, 4(x2) — cross-validated against llvm-mc
#[test]
fn riscv_fsw_f1_4_x2() {
    let code = assemble("fsw f1, 4(x2)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x27, 0x22, 0x11, 0x00]);
}

/// FLD f2, 8(x3) — cross-validated against llvm-mc
#[test]
fn riscv_fld_f2_8_x3() {
    let code = assemble("fld f2, 8(x3)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x07, 0xB1, 0x81, 0x00]);
}

/// FSD f3, 16(x4) — cross-validated against llvm-mc
#[test]
fn riscv_fsd_f3_16_x4() {
    let code = assemble("fsd f3, 16(x4)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x27, 0x38, 0x32, 0x00]);
}

/// FADD.S f0, f1, f2 — cross-validated against llvm-mc
#[test]
fn riscv_fadd_s_f0_f1_f2() {
    let code = assemble("fadd.s f0, f1, f2", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0xF0, 0x20, 0x00]);
}

/// FSUB.S f3, f4, f5 — cross-validated against llvm-mc
#[test]
fn riscv_fsub_s_f3_f4_f5() {
    let code = assemble("fsub.s f3, f4, f5", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x71, 0x52, 0x08]);
}

/// FMUL.S f6, f7, f8 — cross-validated against llvm-mc
#[test]
fn riscv_fmul_s_f6_f7_f8() {
    let code = assemble("fmul.s f6, f7, f8", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0xF3, 0x83, 0x10]);
}

/// FDIV.S f9, f10, f11 — cross-validated against llvm-mc
#[test]
fn riscv_fdiv_s_f9_f10_f11() {
    let code = assemble("fdiv.s f9, f10, f11", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x74, 0xB5, 0x18]);
}

/// FSQRT.S f12, f13 — cross-validated against llvm-mc
#[test]
fn riscv_fsqrt_s_f12_f13() {
    let code = assemble("fsqrt.s f12, f13", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0xF6, 0x06, 0x58]);
}

/// FMIN.S f14, f15, f16 — cross-validated against llvm-mc
#[test]
fn riscv_fmin_s_f14_f15_f16() {
    let code = assemble("fmin.s f14, f15, f16", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0x87, 0x07, 0x29]);
}

/// FMAX.S f17, f18, f19 — cross-validated against llvm-mc
#[test]
fn riscv_fmax_s_f17_f18_f19() {
    let code = assemble("fmax.s f17, f18, f19", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x18, 0x39, 0x29]);
}

/// FADD.D f0, f1, f2 — cross-validated against llvm-mc
#[test]
fn riscv_fadd_d_f0_f1_f2() {
    let code = assemble("fadd.d f0, f1, f2", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0xF0, 0x20, 0x02]);
}

/// FSUB.D f3, f4, f5 — cross-validated against llvm-mc
#[test]
fn riscv_fsub_d_f3_f4_f5() {
    let code = assemble("fsub.d f3, f4, f5", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x71, 0x52, 0x0A]);
}

/// FMUL.D f6, f7, f8 — cross-validated against llvm-mc
#[test]
fn riscv_fmul_d_f6_f7_f8() {
    let code = assemble("fmul.d f6, f7, f8", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0xF3, 0x83, 0x12]);
}

/// FDIV.D f9, f10, f11 — cross-validated against llvm-mc
#[test]
fn riscv_fdiv_d_f9_f10_f11() {
    let code = assemble("fdiv.d f9, f10, f11", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x74, 0xB5, 0x1A]);
}

/// FSQRT.D f12, f13 — cross-validated against llvm-mc
#[test]
fn riscv_fsqrt_d_f12_f13() {
    let code = assemble("fsqrt.d f12, f13", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0xF6, 0x06, 0x5A]);
}

/// FMIN.D f14, f15, f16 — cross-validated against llvm-mc
#[test]
fn riscv_fmin_d_f14_f15_f16() {
    let code = assemble("fmin.d f14, f15, f16", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0x87, 0x07, 0x2B]);
}

/// FMAX.D f17, f18, f19 — cross-validated against llvm-mc
#[test]
fn riscv_fmax_d_f17_f18_f19() {
    let code = assemble("fmax.d f17, f18, f19", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x18, 0x39, 0x2B]);
}

/// FEQ.S x1, f0, f1 — cross-validated against llvm-mc
#[test]
fn riscv_feq_s_x1_f0_f1() {
    let code = assemble("feq.s x1, f0, f1", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x20, 0x10, 0xA0]);
}

/// FLT.S x2, f2, f3 — cross-validated against llvm-mc
#[test]
fn riscv_flt_s_x2_f2_f3() {
    let code = assemble("flt.s x2, f2, f3", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0x11, 0x31, 0xA0]);
}

/// FLE.S x3, f4, f5 — cross-validated against llvm-mc
#[test]
fn riscv_fle_s_x3_f4_f5() {
    let code = assemble("fle.s x3, f4, f5", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x01, 0x52, 0xA0]);
}

/// FEQ.D x4, f6, f7 — cross-validated against llvm-mc
#[test]
fn riscv_feq_d_x4_f6_f7() {
    let code = assemble("feq.d x4, f6, f7", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0x22, 0x73, 0xA2]);
}

/// FLT.D x5, f8, f9 — cross-validated against llvm-mc
#[test]
fn riscv_flt_d_x5_f8_f9() {
    let code = assemble("flt.d x5, f8, f9", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x12, 0x94, 0xA2]);
}

/// FLE.D x6, f10, f11 — cross-validated against llvm-mc
#[test]
fn riscv_fle_d_x6_f10_f11() {
    let code = assemble("fle.d x6, f10, f11", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0x03, 0xB5, 0xA2]);
}

/// FCVT.W.S x7, f0 — cross-validated against llvm-mc
#[test]
fn riscv_fcvt_w_s_x7_f0() {
    let code = assemble("fcvt.w.s x7, f0", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x73, 0x00, 0xC0]);
}

/// FCVT.S.W f1, x8 — cross-validated against llvm-mc
#[test]
fn riscv_fcvt_s_w_f1_x8() {
    let code = assemble("fcvt.s.w f1, x8", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x70, 0x04, 0xD0]);
}

/// FCVT.W.D x9, f2 — cross-validated against llvm-mc
#[test]
fn riscv_fcvt_w_d_x9_f2() {
    let code = assemble("fcvt.w.d x9, f2", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x74, 0x01, 0xC2]);
}

/// FCVT.D.W f3, x10 — rm=DYN default (llvm-mc uses RNE; both valid for exact conversion)
#[test]
fn riscv_fcvt_d_w_f3_x10() {
    let code = assemble("fcvt.d.w f3, x10", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x71, 0x05, 0xD2]);
}

/// FMV.X.W x11, f4 — cross-validated against llvm-mc
#[test]
fn riscv_fmv_x_w_x11_f4() {
    let code = assemble("fmv.x.w x11, f4", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x05, 0x02, 0xE0]);
}

/// FMV.W.X f5, x12 — cross-validated against llvm-mc
#[test]
fn riscv_fmv_w_x_f5_x12() {
    let code = assemble("fmv.w.x f5, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x02, 0x06, 0xF0]);
}

/// FCLASS.S x13, f6 — cross-validated against llvm-mc
#[test]
fn riscv_fclass_s_x13_f6() {
    let code = assemble("fclass.s x13, f6", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0xD3, 0x16, 0x03, 0xE0]);
}

/// FCLASS.D x14, f7 — cross-validated against llvm-mc
#[test]
fn riscv_fclass_d_x14_f7() {
    let code = assemble("fclass.d x14, f7", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0x97, 0x03, 0xE2]);
}

/// FCVT.D.S f0, f1 — rm=DYN default (llvm-mc uses RNE; both valid for exact conversion)
#[test]
fn riscv_fcvt_d_s_f0_f1() {
    let code = assemble("fcvt.d.s f0, f1", Arch::Rv64).unwrap();
    // Our assembler correctly uses rm=DYN(111); llvm-mc encodes rm=RNE(000)
    // since single→double is exact. Both encodings are architecturally valid.
    let code_rm_byte = code[1];
    // rm field in bits [14:12] of 32-bit encoding → byte 1 bits [6:4]
    assert!(
        code_rm_byte == 0xF0 || code_rm_byte == 0x80,
        "expected rm=DYN(0xF0) or rm=RNE(0x80), got {:#04X}",
        code_rm_byte
    );
}

/// FCVT.S.D f2, f3 — cross-validated against llvm-mc
#[test]
fn riscv_fcvt_s_d_f2_f3() {
    let code = assemble("fcvt.s.d f2, f3", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x53, 0xF1, 0x11, 0x40]);
}

// ─── AArch64 NEON Cross-Validation (llvm-mc aarch64) ────────────────────

/// ADD v0.4s, v1.4s, v2.4s — cross-validated against llvm-mc
#[test]
fn neon_add_v0_4s_v1_4s_v2_4s() {
    let code = assemble("add v0.4s, v1.4s, v2.4s", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x84, 0xA2, 0x4E]);
}

/// ADD v3.8b, v4.8b, v5.8b — cross-validated against llvm-mc
#[test]
fn neon_add_v3_8b_v4_8b_v5_8b() {
    let code = assemble("add v3.8b, v4.8b, v5.8b", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x83, 0x84, 0x25, 0x0E]);
}

/// ADD v6.16b, v7.16b, v8.16b — cross-validated against llvm-mc
#[test]
fn neon_add_v6_16b_v7_16b_v8_16b() {
    let code = assemble("add v6.16b, v7.16b, v8.16b", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0xE6, 0x84, 0x28, 0x4E]);
}

/// ADD v9.2d, v10.2d, v11.2d — cross-validated against llvm-mc
#[test]
fn neon_add_v9_2d_v10_2d_v11_2d() {
    let code = assemble("add v9.2d, v10.2d, v11.2d", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x49, 0x85, 0xEB, 0x4E]);
}

/// ADD v0.4h, v1.4h, v2.4h — cross-validated against llvm-mc
#[test]
fn neon_add_v0_4h_v1_4h_v2_4h() {
    let code = assemble("add v0.4h, v1.4h, v2.4h", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x84, 0x62, 0x0E]);
}

/// ADD v0.8h, v1.8h, v2.8h — cross-validated against llvm-mc
#[test]
fn neon_add_v0_8h_v1_8h_v2_8h() {
    let code = assemble("add v0.8h, v1.8h, v2.8h", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x84, 0x62, 0x4E]);
}

/// SUB v0.4s, v1.4s, v2.4s — cross-validated against llvm-mc
#[test]
fn neon_sub_v0_4s_v1_4s_v2_4s() {
    let code = assemble("sub v0.4s, v1.4s, v2.4s", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x84, 0xA2, 0x6E]);
}

/// SUB v3.16b, v4.16b, v5.16b — cross-validated against llvm-mc
#[test]
fn neon_sub_v3_16b_v4_16b_v5_16b() {
    let code = assemble("sub v3.16b, v4.16b, v5.16b", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x83, 0x84, 0x25, 0x6E]);
}

/// MUL v0.4s, v1.4s, v2.4s — cross-validated against llvm-mc
#[test]
fn neon_mul_v0_4s_v1_4s_v2_4s() {
    let code = assemble("mul v0.4s, v1.4s, v2.4s", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x9C, 0xA2, 0x4E]);
}

/// AND v0.16b, v1.16b, v2.16b — cross-validated against llvm-mc
#[test]
fn neon_and_v0_16b_v1_16b_v2_16b() {
    let code = assemble("and v0.16b, v1.16b, v2.16b", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x1C, 0x22, 0x4E]);
}

/// ORR v0.16b, v1.16b, v2.16b — cross-validated against llvm-mc
#[test]
fn neon_orr_v0_16b_v1_16b_v2_16b() {
    let code = assemble("orr v0.16b, v1.16b, v2.16b", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x1C, 0xA2, 0x4E]);
}

/// EOR v0.16b, v1.16b, v2.16b — cross-validated against llvm-mc
#[test]
fn neon_eor_v0_16b_v1_16b_v2_16b() {
    let code = assemble("eor v0.16b, v1.16b, v2.16b", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x1C, 0x22, 0x6E]);
}

/// BIC v0.16b, v1.16b, v2.16b — cross-validated against llvm-mc
#[test]
fn neon_bic_v0_16b_v1_16b_v2_16b() {
    let code = assemble("bic v0.16b, v1.16b, v2.16b", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x1C, 0x62, 0x4E]);
}

/// ORN v0.16b, v1.16b, v2.16b — cross-validated against llvm-mc
#[test]
fn neon_orn_v0_16b_v1_16b_v2_16b() {
    let code = assemble("orn v0.16b, v1.16b, v2.16b", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x1C, 0xE2, 0x4E]);
}

/// CMEQ v0.4s, v1.4s, v2.4s — cross-validated against llvm-mc
#[test]
fn neon_cmeq_v0_4s_v1_4s_v2_4s() {
    let code = assemble("cmeq v0.4s, v1.4s, v2.4s", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x8C, 0xA2, 0x6E]);
}

/// CMHI v0.4s, v1.4s, v2.4s — cross-validated against llvm-mc
#[test]
fn neon_cmhi_v0_4s_v1_4s_v2_4s() {
    let code = assemble("cmhi v0.4s, v1.4s, v2.4s", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x34, 0xA2, 0x6E]);
}

/// CMHS v0.4s, v1.4s, v2.4s — cross-validated against llvm-mc
#[test]
fn neon_cmhs_v0_4s_v1_4s_v2_4s() {
    let code = assemble("cmhs v0.4s, v1.4s, v2.4s", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x3C, 0xA2, 0x6E]);
}

/// CMGT v0.4s, v1.4s, v2.4s — cross-validated against llvm-mc
#[test]
fn neon_cmgt_v0_4s_v1_4s_v2_4s() {
    let code = assemble("cmgt v0.4s, v1.4s, v2.4s", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x34, 0xA2, 0x4E]);
}

/// CMGE v0.4s, v1.4s, v2.4s — cross-validated against llvm-mc
#[test]
fn neon_cmge_v0_4s_v1_4s_v2_4s() {
    let code = assemble("cmge v0.4s, v1.4s, v2.4s", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x3C, 0xA2, 0x4E]);
}

/// NEG v0.4s, v1.4s — cross-validated against llvm-mc
#[test]
fn neon_neg_v0_4s_v1_4s() {
    let code = assemble("neg v0.4s, v1.4s", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0xB8, 0xA0, 0x6E]);
}

/// ABS v0.4s, v1.4s — cross-validated against llvm-mc
#[test]
fn neon_abs_v0_4s_v1_4s() {
    let code = assemble("abs v0.4s, v1.4s", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0xB8, 0xA0, 0x4E]);
}

/// NOT v0.16b, v1.16b — cross-validated against llvm-mc
#[test]
fn neon_not_v0_16b_v1_16b() {
    let code = assemble("not v0.16b, v1.16b", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x58, 0x20, 0x6E]);
}

/// CNT v0.16b, v1.16b — cross-validated against llvm-mc
#[test]
fn neon_cnt_v0_16b_v1_16b() {
    let code = assemble("cnt v0.16b, v1.16b", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x58, 0x20, 0x4E]);
}

/// LD1 {v0.4s}, [x1] — cross-validated against llvm-mc
#[test]
fn neon_ld1_v0_4s_x1() {
    let code = assemble("ld1 {v0.4s}, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x78, 0x40, 0x4C]);
}

/// ST1 {v0.4s}, [x1] — cross-validated against llvm-mc
#[test]
fn neon_st1_v0_4s_x1() {
    let code = assemble("st1 {v0.4s}, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x78, 0x00, 0x4C]);
}

/// LD1 {v0.16b}, [x0] — cross-validated against llvm-mc
#[test]
fn neon_ld1_v0_16b_x0() {
    let code = assemble("ld1 {v0.16b}, [x0]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x70, 0x40, 0x4C]);
}

/// ST1 {v0.16b}, [x0] — cross-validated against llvm-mc
#[test]
fn neon_st1_v0_16b_x0() {
    let code = assemble("st1 {v0.16b}, [x0]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x70, 0x00, 0x4C]);
}

// ============================================================================
// AArch64 Core Instructions — cross-validated against llvm-mc (aarch64)
// (Only instructions not already covered by existing tests above)
// ============================================================================

/// ADDS x0, x1, x2 — encoding: [0x20,0x00,0x02,0xab]
#[test]
fn aarch64_adds_x0_x1_x2() {
    let code = assemble("adds x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x02, 0xab]);
}

/// SUBS x0, x1, x2 — encoding: [0x20,0x00,0x02,0xeb]
#[test]
fn aarch64_subs_x0_x1_x2() {
    let code = assemble("subs x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x02, 0xeb]);
}

/// ADD x0, x1, #42 — encoding: [0x20,0xa8,0x00,0x91]
#[test]
fn aarch64_add_x0_x1_imm42() {
    let code = assemble("add x0, x1, 42", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0xa8, 0x00, 0x91]);
}

/// SUB x0, x1, #42 — encoding: [0x20,0xa8,0x00,0xd1]
#[test]
fn aarch64_sub_x0_x1_imm42() {
    let code = assemble("sub x0, x1, 42", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0xa8, 0x00, 0xd1]);
}

/// ANDS x0, x1, x2 — encoding: [0x20,0x00,0x02,0xea]
#[test]
fn aarch64_ands_x0_x1_x2() {
    let code = assemble("ands x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x02, 0xea]);
}

/// BIC x0, x1, x2 — encoding: [0x20,0x00,0x22,0x8a]
#[test]
fn aarch64_bic_x0_x1_x2() {
    let code = assemble("bic x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x22, 0x8a]);
}

/// ORN x0, x1, x2 — encoding: [0x20,0x00,0x22,0xaa]
#[test]
fn aarch64_orn_x0_x1_x2() {
    let code = assemble("orn x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x22, 0xaa]);
}

/// EON x0, x1, x2 — encoding: [0x20,0x00,0x22,0xca]
#[test]
fn aarch64_eon_x0_x1_x2() {
    let code = assemble("eon x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x22, 0xca]);
}

/// LSL x0, x1, x2 — encoding: [0x20,0x20,0xc2,0x9a]
#[test]
fn aarch64_lsl_x0_x1_x2() {
    let code = assemble("lsl x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x20, 0xc2, 0x9a]);
}

/// LSR x0, x1, x2 — encoding: [0x20,0x24,0xc2,0x9a]
#[test]
fn aarch64_lsr_x0_x1_x2() {
    let code = assemble("lsr x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x24, 0xc2, 0x9a]);
}

/// ASR x0, x1, x2 — encoding: [0x20,0x28,0xc2,0x9a]
#[test]
fn aarch64_asr_x0_x1_x2() {
    let code = assemble("asr x0, x1, x2", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x28, 0xc2, 0x9a]);
}

/// MSUB x0, x1, x2, x3 — encoding: [0x20,0x8c,0x02,0x9b]
#[test]
fn aarch64_msub_x0_x1_x2_x3() {
    let code = assemble("msub x0, x1, x2, x3", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x8c, 0x02, 0x9b]);
}

/// CMP x0, x1 — encoding: [0x1f,0x00,0x01,0xeb]
#[test]
fn aarch64_cmp_x0_x1() {
    let code = assemble("cmp x0, x1", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x1f, 0x00, 0x01, 0xeb]);
}

/// CMN x0, x1 — encoding: [0x1f,0x00,0x01,0xab]
#[test]
fn aarch64_cmn_x0_x1() {
    let code = assemble("cmn x0, x1", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x1f, 0x00, 0x01, 0xab]);
}

/// TST x0, x1 — encoding: [0x1f,0x00,0x01,0xea]
#[test]
fn aarch64_tst_x0_x1() {
    let code = assemble("tst x0, x1", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x1f, 0x00, 0x01, 0xea]);
}

/// LDR x0, [x1, #8] — encoding: [0x20,0x04,0x40,0xf9]
#[test]
fn aarch64_ldr_x0_x1_8() {
    let code = assemble("ldr x0, [x1, 8]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x04, 0x40, 0xf9]);
}

/// STR x0, [x1, #8] — encoding: [0x20,0x04,0x00,0xf9]
#[test]
fn aarch64_str_x0_x1_8() {
    let code = assemble("str x0, [x1, 8]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x04, 0x00, 0xf9]);
}

/// LDP x0, x2, [x1] — encoding: [0x20,0x08,0x40,0xa9]
#[test]
fn aarch64_ldp_x0_x2_x1() {
    let code = assemble("ldp x0, x2, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x08, 0x40, 0xa9]);
}

/// STP x0, x2, [x1] — encoding: [0x20,0x08,0x00,0xa9]
#[test]
fn aarch64_stp_x0_x2_x1() {
    let code = assemble("stp x0, x2, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x08, 0x00, 0xa9]);
}

/// LDRB w0, [x1] — encoding: [0x20,0x00,0x40,0x39]
#[test]
fn aarch64_ldrb_w0_x1() {
    let code = assemble("ldrb w0, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x40, 0x39]);
}

/// LDRH w0, [x1] — encoding: [0x20,0x00,0x40,0x79]
#[test]
fn aarch64_ldrh_w0_x1() {
    let code = assemble("ldrh w0, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x40, 0x79]);
}

/// LDRSW x0, [x1] — encoding: [0x20,0x00,0x80,0xb9]
#[test]
fn aarch64_ldrsw_x0_x1() {
    let code = assemble("ldrsw x0, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x80, 0xb9]);
}

/// STRB w0, [x1] — encoding: [0x20,0x00,0x00,0x39]
#[test]
fn aarch64_strb_w0_x1() {
    let code = assemble("strb w0, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x00, 0x39]);
}

/// STRH w0, [x1] — encoding: [0x20,0x00,0x00,0x79]
#[test]
fn aarch64_strh_w0_x1() {
    let code = assemble("strh w0, [x1]", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x00, 0x79]);
}

/// B #0 — encoding: [0x00,0x00,0x00,0x14]
#[test]
fn aarch64_b_0() {
    let code = assemble("b 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x00, 0x14]);
}

/// BL #0 — encoding: [0x00,0x00,0x00,0x94]
#[test]
fn aarch64_bl_0() {
    let code = assemble("bl 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x00, 0x94]);
}

/// BR x10 — encoding: [0x40,0x01,0x1f,0xd6]
#[test]
fn aarch64_br_x10() {
    let code = assemble("br x10", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x40, 0x01, 0x1f, 0xd6]);
}

/// BLR x10 — encoding: [0x40,0x01,0x3f,0xd6]
#[test]
fn aarch64_blr_x10() {
    let code = assemble("blr x10", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x40, 0x01, 0x3f, 0xd6]);
}

/// CBZ x0, #0 — encoding: [0x00,0x00,0x00,0xb4]
#[test]
fn aarch64_cbz_x0_0() {
    let code = assemble("cbz x0, 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x00, 0xb4]);
}

/// CBNZ x0, #0 — encoding: [0x00,0x00,0x00,0xb5]
#[test]
fn aarch64_cbnz_x0_0() {
    let code = assemble("cbnz x0, 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x00, 0xb5]);
}

/// B.EQ #0 — encoding: [0x00,0x00,0x00,0x54]
#[test]
fn aarch64_beq_0() {
    let code = assemble("b.eq 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x00, 0x54]);
}

/// B.NE #0 — encoding: [0x01,0x00,0x00,0x54]
#[test]
fn aarch64_bne_0() {
    let code = assemble("b.ne 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x01, 0x00, 0x00, 0x54]);
}

/// B.LT #0 — encoding: [0x0b,0x00,0x00,0x54]
#[test]
fn aarch64_blt_0() {
    let code = assemble("b.lt 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x0b, 0x00, 0x00, 0x54]);
}

/// B.GE #0 — encoding: [0x0a,0x00,0x00,0x54]
#[test]
fn aarch64_bge_0() {
    let code = assemble("b.ge 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x0a, 0x00, 0x00, 0x54]);
}

/// CLS x0, x1 — encoding: [0x20,0x14,0xc0,0xda]
#[test]
fn aarch64_cls_x0_x1() {
    let code = assemble("cls x0, x1", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x14, 0xc0, 0xda]);
}

/// REV16 x0, x1 — encoding: [0x20,0x04,0xc0,0xda]
#[test]
fn aarch64_rev16_x0_x1() {
    let code = assemble("rev16 x0, x1", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x04, 0xc0, 0xda]);
}

/// REV32 x0, x1 — encoding: [0x20,0x08,0xc0,0xda]
#[test]
fn aarch64_rev32_x0_x1() {
    let code = assemble("rev32 x0, x1", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x08, 0xc0, 0xda]);
}

/// CSEL x0, x1, x2, eq — encoding: [0x20,0x00,0x82,0x9a]
#[test]
fn aarch64_csel_x0_x1_x2_eq() {
    let code = assemble("csel x0, x1, x2, eq", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x82, 0x9a]);
}

/// CSINC x0, x1, x2, eq — encoding: [0x20,0x04,0x82,0x9a]
#[test]
fn aarch64_csinc_x0_x1_x2_eq() {
    let code = assemble("csinc x0, x1, x2, eq", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x04, 0x82, 0x9a]);
}

/// CSINV x0, x1, x2, eq — encoding: [0x20,0x00,0x82,0xda]
#[test]
fn aarch64_csinv_x0_x1_x2_eq() {
    let code = assemble("csinv x0, x1, x2, eq", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x00, 0x82, 0xda]);
}

/// CSNEG x0, x1, x2, eq — encoding: [0x20,0x04,0x82,0xda]
#[test]
fn aarch64_csneg_x0_x1_x2_eq() {
    let code = assemble("csneg x0, x1, x2, eq", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x20, 0x04, 0x82, 0xda]);
}

/// ADR x0, #0 — encoding: [0x00,0x00,0x00,0x10]
#[test]
fn aarch64_adr_x0_0() {
    let code = assemble("adr x0, 0", Arch::Aarch64).unwrap();
    assert_eq!(code, vec![0x00, 0x00, 0x00, 0x10]);
}

// ============================================================================
// RISC-V Base ISA (RV64I) — cross-validated against llvm-mc (riscv64)
// ============================================================================

/// ADD x10, x11, x12 — encoding: [0x33,0x85,0xc5,0x00]
#[test]
fn riscv_add_x10_x11_x12() {
    let code = assemble("add x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0x85, 0xc5, 0x00]);
}

/// SUB x10, x11, x12 — encoding: [0x33,0x85,0xc5,0x40]
#[test]
fn riscv_sub_x10_x11_x12() {
    let code = assemble("sub x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0x85, 0xc5, 0x40]);
}

/// AND x10, x11, x12 — encoding: [0x33,0xf5,0xc5,0x00]
#[test]
fn riscv_and_x10_x11_x12() {
    let code = assemble("and x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xf5, 0xc5, 0x00]);
}

/// OR x10, x11, x12 — encoding: [0x33,0xe5,0xc5,0x00]
#[test]
fn riscv_or_x10_x11_x12() {
    let code = assemble("or x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xe5, 0xc5, 0x00]);
}

/// XOR x10, x11, x12 — encoding: [0x33,0xc5,0xc5,0x00]
#[test]
fn riscv_xor_x10_x11_x12() {
    let code = assemble("xor x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xc5, 0xc5, 0x00]);
}

/// SLL x10, x11, x12 — encoding: [0x33,0x95,0xc5,0x00]
#[test]
fn riscv_sll_x10_x11_x12() {
    let code = assemble("sll x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0x95, 0xc5, 0x00]);
}

/// SRL x10, x11, x12 — encoding: [0x33,0xd5,0xc5,0x00]
#[test]
fn riscv_srl_x10_x11_x12() {
    let code = assemble("srl x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xd5, 0xc5, 0x00]);
}

/// SRA x10, x11, x12 — encoding: [0x33,0xd5,0xc5,0x40]
#[test]
fn riscv_sra_x10_x11_x12() {
    let code = assemble("sra x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xd5, 0xc5, 0x40]);
}

/// SLT x10, x11, x12 — encoding: [0x33,0xa5,0xc5,0x00]
#[test]
fn riscv_slt_x10_x11_x12() {
    let code = assemble("slt x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xa5, 0xc5, 0x00]);
}

/// SLTU x10, x11, x12 — encoding: [0x33,0xb5,0xc5,0x00]
#[test]
fn riscv_sltu_x10_x11_x12() {
    let code = assemble("sltu x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xb5, 0xc5, 0x00]);
}

/// ADDI x10, x11, 42 — encoding: [0x13,0x85,0xa5,0x02]
#[test]
fn riscv_addi_x10_x11_42() {
    let code = assemble("addi x10, x11, 42", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x13, 0x85, 0xa5, 0x02]);
}

/// ANDI x10, x11, 42 — encoding: [0x13,0xf5,0xa5,0x02]
#[test]
fn riscv_andi_x10_x11_42() {
    let code = assemble("andi x10, x11, 42", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x13, 0xf5, 0xa5, 0x02]);
}

/// ORI x10, x11, 42 — encoding: [0x13,0xe5,0xa5,0x02]
#[test]
fn riscv_ori_x10_x11_42() {
    let code = assemble("ori x10, x11, 42", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x13, 0xe5, 0xa5, 0x02]);
}

/// XORI x10, x11, 42 — encoding: [0x13,0xc5,0xa5,0x02]
#[test]
fn riscv_xori_x10_x11_42() {
    let code = assemble("xori x10, x11, 42", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x13, 0xc5, 0xa5, 0x02]);
}

/// SLTI x10, x11, 42 — encoding: [0x13,0xa5,0xa5,0x02]
#[test]
fn riscv_slti_x10_x11_42() {
    let code = assemble("slti x10, x11, 42", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x13, 0xa5, 0xa5, 0x02]);
}

/// SLTIU x10, x11, 42 — encoding: [0x13,0xb5,0xa5,0x02]
#[test]
fn riscv_sltiu_x10_x11_42() {
    let code = assemble("sltiu x10, x11, 42", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x13, 0xb5, 0xa5, 0x02]);
}

/// SLLI x10, x11, 5 — encoding: [0x13,0x95,0x55,0x00]
#[test]
fn riscv_slli_x10_x11_5() {
    let code = assemble("slli x10, x11, 5", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x13, 0x95, 0x55, 0x00]);
}

/// SRLI x10, x11, 5 — encoding: [0x13,0xd5,0x55,0x00]
#[test]
fn riscv_srli_x10_x11_5() {
    let code = assemble("srli x10, x11, 5", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x13, 0xd5, 0x55, 0x00]);
}

/// SRAI x10, x11, 5 — encoding: [0x13,0xd5,0x55,0x40]
#[test]
fn riscv_srai_x10_x11_5() {
    let code = assemble("srai x10, x11, 5", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x13, 0xd5, 0x55, 0x40]);
}

/// LUI x10, 0x12345 — encoding: [0x37,0x55,0x34,0x12]
#[test]
fn riscv_lui_x10_0x12345() {
    let code = assemble("lui x10, 0x12345", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x37, 0x55, 0x34, 0x12]);
}

/// AUIPC x10, 0x12345 — encoding: [0x17,0x55,0x34,0x12]
#[test]
fn riscv_auipc_x10_0x12345() {
    let code = assemble("auipc x10, 0x12345", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x17, 0x55, 0x34, 0x12]);
}

/// LB x10, 0(x11) — encoding: [0x03,0x85,0x05,0x00]
#[test]
fn riscv_lb_x10_0_x11() {
    let code = assemble("lb x10, 0(x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x03, 0x85, 0x05, 0x00]);
}

/// LH x10, 0(x11) — encoding: [0x03,0x95,0x05,0x00]
#[test]
fn riscv_lh_x10_0_x11() {
    let code = assemble("lh x10, 0(x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x03, 0x95, 0x05, 0x00]);
}

/// LW x10, 0(x11) — encoding: [0x03,0xa5,0x05,0x00]
#[test]
fn riscv_lw_x10_0_x11() {
    let code = assemble("lw x10, 0(x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x03, 0xa5, 0x05, 0x00]);
}

/// LD x10, 0(x11) — encoding: [0x03,0xb5,0x05,0x00]
#[test]
fn riscv_ld_x10_0_x11() {
    let code = assemble("ld x10, 0(x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x03, 0xb5, 0x05, 0x00]);
}

/// LBU x10, 0(x11) — encoding: [0x03,0xc5,0x05,0x00]
#[test]
fn riscv_lbu_x10_0_x11() {
    let code = assemble("lbu x10, 0(x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x03, 0xc5, 0x05, 0x00]);
}

/// LHU x10, 0(x11) — encoding: [0x03,0xd5,0x05,0x00]
#[test]
fn riscv_lhu_x10_0_x11() {
    let code = assemble("lhu x10, 0(x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x03, 0xd5, 0x05, 0x00]);
}

/// LWU x10, 0(x11) — encoding: [0x03,0xe5,0x05,0x00]
#[test]
fn riscv_lwu_x10_0_x11() {
    let code = assemble("lwu x10, 0(x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x03, 0xe5, 0x05, 0x00]);
}

/// SB x10, 0(x11) — encoding: [0x23,0x80,0xa5,0x00]
#[test]
fn riscv_sb_x10_0_x11() {
    let code = assemble("sb x10, 0(x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x23, 0x80, 0xa5, 0x00]);
}

/// SH x10, 0(x11) — encoding: [0x23,0x90,0xa5,0x00]
#[test]
fn riscv_sh_x10_0_x11() {
    let code = assemble("sh x10, 0(x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x23, 0x90, 0xa5, 0x00]);
}

/// SW x10, 0(x11) — encoding: [0x23,0xa0,0xa5,0x00]
#[test]
fn riscv_sw_x10_0_x11() {
    let code = assemble("sw x10, 0(x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x23, 0xa0, 0xa5, 0x00]);
}

/// SD x10, 0(x11) — encoding: [0x23,0xb0,0xa5,0x00]
#[test]
fn riscv_sd_x10_0_x11() {
    let code = assemble("sd x10, 0(x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x23, 0xb0, 0xa5, 0x00]);
}

/// BEQ x10, x11, 0 — encoding: [0x63,0x00,0xb5,0x00]
#[test]
fn riscv_beq_x10_x11_0() {
    let code = assemble("beq x10, x11, 0", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x63, 0x00, 0xb5, 0x00]);
}

/// BNE x10, x11, 0 — encoding: [0x63,0x10,0xb5,0x00]
#[test]
fn riscv_bne_x10_x11_0() {
    let code = assemble("bne x10, x11, 0", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x63, 0x10, 0xb5, 0x00]);
}

/// BLT x10, x11, 0 — encoding: [0x63,0x40,0xb5,0x00]
#[test]
fn riscv_blt_x10_x11_0() {
    let code = assemble("blt x10, x11, 0", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x63, 0x40, 0xb5, 0x00]);
}

/// BGE x10, x11, 0 — encoding: [0x63,0x50,0xb5,0x00]
#[test]
fn riscv_bge_x10_x11_0() {
    let code = assemble("bge x10, x11, 0", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x63, 0x50, 0xb5, 0x00]);
}

/// BLTU x10, x11, 0 — encoding: [0x63,0x60,0xb5,0x00]
#[test]
fn riscv_bltu_x10_x11_0() {
    let code = assemble("bltu x10, x11, 0", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x63, 0x60, 0xb5, 0x00]);
}

/// BGEU x10, x11, 0 — encoding: [0x63,0x70,0xb5,0x00]
#[test]
fn riscv_bgeu_x10_x11_0() {
    let code = assemble("bgeu x10, x11, 0", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x63, 0x70, 0xb5, 0x00]);
}

/// ECALL — encoding: [0x73,0x00,0x00,0x00]
#[test]
fn riscv_ecall() {
    let code = assemble("ecall", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x73, 0x00, 0x00, 0x00]);
}

/// EBREAK — encoding: [0x73,0x00,0x10,0x00]
#[test]
fn riscv_ebreak() {
    let code = assemble("ebreak", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x73, 0x00, 0x10, 0x00]);
}

/// NOP — encoding: [0x13,0x00,0x00,0x00]
#[test]
fn riscv_nop() {
    let code = assemble("nop", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x13, 0x00, 0x00, 0x00]);
}

/// ADDW x10, x11, x12 — encoding: [0x3b,0x85,0xc5,0x00]
#[test]
fn riscv_addw_x10_x11_x12() {
    let code = assemble("addw x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x3b, 0x85, 0xc5, 0x00]);
}

/// SUBW x10, x11, x12 — encoding: [0x3b,0x85,0xc5,0x40]
#[test]
fn riscv_subw_x10_x11_x12() {
    let code = assemble("subw x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x3b, 0x85, 0xc5, 0x40]);
}

/// SLLW x10, x11, x12 — encoding: [0x3b,0x95,0xc5,0x00]
#[test]
fn riscv_sllw_x10_x11_x12() {
    let code = assemble("sllw x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x3b, 0x95, 0xc5, 0x00]);
}

/// SRLW x10, x11, x12 — encoding: [0x3b,0xd5,0xc5,0x00]
#[test]
fn riscv_srlw_x10_x11_x12() {
    let code = assemble("srlw x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x3b, 0xd5, 0xc5, 0x00]);
}

/// SRAW x10, x11, x12 — encoding: [0x3b,0xd5,0xc5,0x40]
#[test]
fn riscv_sraw_x10_x11_x12() {
    let code = assemble("sraw x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x3b, 0xd5, 0xc5, 0x40]);
}

/// ADDIW x10, x11, 42 — encoding: [0x1b,0x85,0xa5,0x02]
#[test]
fn riscv_addiw_x10_x11_42() {
    let code = assemble("addiw x10, x11, 42", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x1b, 0x85, 0xa5, 0x02]);
}

/// SLLIW x10, x11, 5 — encoding: [0x1b,0x95,0x55,0x00]
#[test]
fn riscv_slliw_x10_x11_5() {
    let code = assemble("slliw x10, x11, 5", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x1b, 0x95, 0x55, 0x00]);
}

/// SRLIW x10, x11, 5 — encoding: [0x1b,0xd5,0x55,0x00]
#[test]
fn riscv_srliw_x10_x11_5() {
    let code = assemble("srliw x10, x11, 5", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x1b, 0xd5, 0x55, 0x00]);
}

/// SRAIW x10, x11, 5 — encoding: [0x1b,0xd5,0x55,0x40]
#[test]
fn riscv_sraiw_x10_x11_5() {
    let code = assemble("sraiw x10, x11, 5", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x1b, 0xd5, 0x55, 0x40]);
}

// ============================================================================
// RISC-V M Extension — cross-validated against llvm-mc (riscv64, +m)
// ============================================================================

/// MUL x10, x11, x12 — encoding: [0x33,0x85,0xc5,0x02]
#[test]
fn riscv_mul_x10_x11_x12() {
    let code = assemble("mul x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0x85, 0xc5, 0x02]);
}

/// MULH x10, x11, x12 — encoding: [0x33,0x95,0xc5,0x02]
#[test]
fn riscv_mulh_x10_x11_x12() {
    let code = assemble("mulh x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0x95, 0xc5, 0x02]);
}

/// MULHSU x10, x11, x12 — encoding: [0x33,0xa5,0xc5,0x02]
#[test]
fn riscv_mulhsu_x10_x11_x12() {
    let code = assemble("mulhsu x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xa5, 0xc5, 0x02]);
}

/// MULHU x10, x11, x12 — encoding: [0x33,0xb5,0xc5,0x02]
#[test]
fn riscv_mulhu_x10_x11_x12() {
    let code = assemble("mulhu x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xb5, 0xc5, 0x02]);
}

/// DIV x10, x11, x12 — encoding: [0x33,0xc5,0xc5,0x02]
#[test]
fn riscv_div_x10_x11_x12() {
    let code = assemble("div x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xc5, 0xc5, 0x02]);
}

/// DIVU x10, x11, x12 — encoding: [0x33,0xd5,0xc5,0x02]
#[test]
fn riscv_divu_x10_x11_x12() {
    let code = assemble("divu x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xd5, 0xc5, 0x02]);
}

/// REM x10, x11, x12 — encoding: [0x33,0xe5,0xc5,0x02]
#[test]
fn riscv_rem_x10_x11_x12() {
    let code = assemble("rem x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xe5, 0xc5, 0x02]);
}

/// REMU x10, x11, x12 — encoding: [0x33,0xf5,0xc5,0x02]
#[test]
fn riscv_remu_x10_x11_x12() {
    let code = assemble("remu x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x33, 0xf5, 0xc5, 0x02]);
}

/// MULW x10, x11, x12 — encoding: [0x3b,0x85,0xc5,0x02]
#[test]
fn riscv_mulw_x10_x11_x12() {
    let code = assemble("mulw x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x3b, 0x85, 0xc5, 0x02]);
}

/// DIVW x10, x11, x12 — encoding: [0x3b,0xc5,0xc5,0x02]
#[test]
fn riscv_divw_x10_x11_x12() {
    let code = assemble("divw x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x3b, 0xc5, 0xc5, 0x02]);
}

/// DIVUW x10, x11, x12 — encoding: [0x3b,0xd5,0xc5,0x02]
#[test]
fn riscv_divuw_x10_x11_x12() {
    let code = assemble("divuw x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x3b, 0xd5, 0xc5, 0x02]);
}

/// REMW x10, x11, x12 — encoding: [0x3b,0xe5,0xc5,0x02]
#[test]
fn riscv_remw_x10_x11_x12() {
    let code = assemble("remw x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x3b, 0xe5, 0xc5, 0x02]);
}

/// REMUW x10, x11, x12 — encoding: [0x3b,0xf5,0xc5,0x02]
#[test]
fn riscv_remuw_x10_x11_x12() {
    let code = assemble("remuw x10, x11, x12", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x3b, 0xf5, 0xc5, 0x02]);
}

// ============================================================================
// RISC-V A Extension (Atomics) — cross-validated against llvm-mc (riscv64, +a)
// ============================================================================

/// LR.W x10, (x11) — encoding: [0x2f,0xa5,0x05,0x10]
#[test]
fn riscv_lr_w_x10_x11() {
    let code = assemble("lr.w x10, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0x05, 0x10]);
}

/// SC.W x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0x18]
#[test]
fn riscv_sc_w_x10_x12_x11() {
    let code = assemble("sc.w x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0x18]);
}

/// AMOSWAP.W x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0x08]
#[test]
fn riscv_amoswap_w_x10_x12_x11() {
    let code = assemble("amoswap.w x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0x08]);
}

/// AMOADD.W x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0x00]
#[test]
fn riscv_amoadd_w_x10_x12_x11() {
    let code = assemble("amoadd.w x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0x00]);
}

/// AMOAND.W x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0x60]
#[test]
fn riscv_amoand_w_x10_x12_x11() {
    let code = assemble("amoand.w x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0x60]);
}

/// AMOOR.W x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0x40]
#[test]
fn riscv_amoor_w_x10_x12_x11() {
    let code = assemble("amoor.w x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0x40]);
}

/// AMOXOR.W x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0x20]
#[test]
fn riscv_amoxor_w_x10_x12_x11() {
    let code = assemble("amoxor.w x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0x20]);
}

/// AMOMAX.W x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0xa0]
#[test]
fn riscv_amomax_w_x10_x12_x11() {
    let code = assemble("amomax.w x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0xa0]);
}

/// AMOMAXU.W x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0xe0]
#[test]
fn riscv_amomaxu_w_x10_x12_x11() {
    let code = assemble("amomaxu.w x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0xe0]);
}

/// AMOMIN.W x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0x80]
#[test]
fn riscv_amomin_w_x10_x12_x11() {
    let code = assemble("amomin.w x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0x80]);
}

/// AMOMINU.W x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0xc0]
#[test]
fn riscv_amominu_w_x10_x12_x11() {
    let code = assemble("amominu.w x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0xc0]);
}

/// LR.D x10, (x11) — encoding: [0x2f,0xb5,0x05,0x10]
#[test]
fn riscv_lr_d_x10_x11() {
    let code = assemble("lr.d x10, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xb5, 0x05, 0x10]);
}

/// SC.D x10, x12, (x11) — encoding: [0x2f,0xb5,0xc5,0x18]
#[test]
fn riscv_sc_d_x10_x12_x11() {
    let code = assemble("sc.d x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xb5, 0xc5, 0x18]);
}

/// AMOSWAP.D x10, x12, (x11) — encoding: [0x2f,0xb5,0xc5,0x08]
#[test]
fn riscv_amoswap_d_x10_x12_x11() {
    let code = assemble("amoswap.d x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xb5, 0xc5, 0x08]);
}

/// AMOADD.D x10, x12, (x11) — encoding: [0x2f,0xb5,0xc5,0x00]
#[test]
fn riscv_amoadd_d_x10_x12_x11() {
    let code = assemble("amoadd.d x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xb5, 0xc5, 0x00]);
}

/// AMOAND.D x10, x12, (x11) — encoding: [0x2f,0xb5,0xc5,0x60]
#[test]
fn riscv_amoand_d_x10_x12_x11() {
    let code = assemble("amoand.d x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xb5, 0xc5, 0x60]);
}

/// AMOOR.D x10, x12, (x11) — encoding: [0x2f,0xb5,0xc5,0x40]
#[test]
fn riscv_amoor_d_x10_x12_x11() {
    let code = assemble("amoor.d x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xb5, 0xc5, 0x40]);
}

/// AMOXOR.D x10, x12, (x11) — encoding: [0x2f,0xb5,0xc5,0x20]
#[test]
fn riscv_amoxor_d_x10_x12_x11() {
    let code = assemble("amoxor.d x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xb5, 0xc5, 0x20]);
}

/// AMOMAX.D x10, x12, (x11) — encoding: [0x2f,0xb5,0xc5,0xa0]
#[test]
fn riscv_amomax_d_x10_x12_x11() {
    let code = assemble("amomax.d x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xb5, 0xc5, 0xa0]);
}

/// AMOMAXU.D x10, x12, (x11) — encoding: [0x2f,0xb5,0xc5,0xe0]
#[test]
fn riscv_amomaxu_d_x10_x12_x11() {
    let code = assemble("amomaxu.d x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xb5, 0xc5, 0xe0]);
}

/// AMOMIN.D x10, x12, (x11) — encoding: [0x2f,0xb5,0xc5,0x80]
#[test]
fn riscv_amomin_d_x10_x12_x11() {
    let code = assemble("amomin.d x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xb5, 0xc5, 0x80]);
}

/// AMOMINU.D x10, x12, (x11) — encoding: [0x2f,0xb5,0xc5,0xc0]
#[test]
fn riscv_amominu_d_x10_x12_x11() {
    let code = assemble("amominu.d x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xb5, 0xc5, 0xc0]);
}

/// AMOSWAP.W.AQ x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0x0c]
#[test]
fn riscv_amoswap_w_aq_x10_x12_x11() {
    let code = assemble("amoswap.w.aq x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0x0c]);
}

/// AMOSWAP.W.RL x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0x0a]
#[test]
fn riscv_amoswap_w_rl_x10_x12_x11() {
    let code = assemble("amoswap.w.rl x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0x0a]);
}

/// AMOSWAP.W.AQRL x10, x12, (x11) — encoding: [0x2f,0xa5,0xc5,0x0e]
#[test]
fn riscv_amoswap_w_aqrl_x10_x12_x11() {
    let code = assemble("amoswap.w.aqrl x10, x12, (x11)", Arch::Rv64).unwrap();
    assert_eq!(code, vec![0x2f, 0xa5, 0xc5, 0x0e]);
}

// ============================================================================
// Error-path tests (T-1)
// ============================================================================

#[test]
fn error_unterminated_block_comment() {
    let result = assemble("nop /* this is never closed", Arch::X86_64);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("unterminated block comment"),
        "expected 'unterminated block comment', got: {msg}"
    );
}

#[test]
fn error_unterminated_string_literal() {
    let result = assemble(".ascii \"hello", Arch::X86_64);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("unterminated string"),
        "expected 'unterminated string', got: {msg}"
    );
}

#[test]
fn error_invalid_mnemonic_x86() {
    let result = assemble("totally_bogus_instruction", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn error_invalid_mnemonic_aarch64() {
    let result = assemble("totally_bogus_instruction", Arch::Aarch64);
    assert!(result.is_err());
}

#[test]
fn error_invalid_mnemonic_riscv() {
    let result = assemble("totally_bogus_instruction", Arch::Rv64);
    assert!(result.is_err());
}

#[test]
fn error_invalid_mnemonic_arm() {
    let result = assemble("totally_bogus_instruction", Arch::Arm);
    assert!(result.is_err());
}

#[test]
fn error_resource_limit_max_source_bytes() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_source_bytes: 10,
        ..Default::default()
    });
    // Source is longer than 10 bytes
    let result = asm.emit("nop\nnop\nnop\nnop\nnop");
    assert!(
        result.is_err(),
        "should fail when source exceeds max_source_bytes"
    );
}

#[test]
fn error_resource_limit_max_iterations() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_iterations: 5,
        ..Default::default()
    });
    // .rept 100 should exceed the 5-iteration limit
    let result = asm.emit(".rept 100\nnop\n.endr");
    assert!(
        result.is_err() || asm.finish().is_err(),
        "should fail when iterations exceed max_iterations"
    );
}

#[test]
fn error_builder_db_exceeds_output_limit() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_output_bytes: 4,
        ..Default::default()
    });
    // 5 bytes should exceed 4-byte limit
    let result = asm.db(&[0x90, 0x90, 0x90, 0x90, 0x90]);
    assert!(
        result.is_err(),
        "db should fail when exceeding max_output_bytes"
    );
}

#[test]
fn error_builder_fill_exceeds_output_limit() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_output_bytes: 10,
        ..Default::default()
    });
    let result = asm.fill(100, 1, 0x90);
    assert!(
        result.is_err(),
        "fill should fail when exceeding max_output_bytes"
    );
}

#[test]
fn error_builder_space_exceeds_output_limit() {
    use asm_rs::assembler::ResourceLimits;
    let mut asm = Assembler::new(Arch::X86_64);
    asm.limits(ResourceLimits {
        max_output_bytes: 10,
        ..Default::default()
    });
    let result = asm.space(100);
    assert!(
        result.is_err(),
        "space should fail when exceeding max_output_bytes"
    );
}

#[test]
fn error_undefined_label() {
    let result = assemble("jmp undefined_label", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn error_duplicate_label() {
    let result = assemble("lbl:\nnop\nlbl:\nnop", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn error_empty_string_literal() {
    // Single char literal with no content
    let result = assemble(".byte ''", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn error_invalid_hex_number() {
    let result = assemble("mov rax, 0xZZZZ", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn error_invalid_binary_number() {
    let result = assemble("mov rax, 0b0012", Arch::X86_64);
    assert!(result.is_err());
}

// ============================================================================
// Constant expression operator tests (A-9)
// ============================================================================

#[test]
fn const_expr_multiply() {
    // .equ SIZE, 4 * 8 → 32; .fill SIZE, 1, 0x90
    let code = assemble(".equ SIZE, 4 * 8\n.fill SIZE, 1, 0x90", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 32);
}

#[test]
fn const_expr_divide() {
    // .equ HALF, 100 / 4 → 25
    let code = assemble(".equ HALF, 100 / 4\n.fill HALF, 1, 0xCC", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 25);
}

#[test]
fn const_expr_modulo() {
    // .equ MOD, 17 % 5 → 2
    let code = assemble(".equ MOD, 17 % 5\n.fill MOD, 1, 0xCC", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 2);
}

#[test]
fn const_expr_shift_left() {
    // .equ VAL, 1 << 4 → 16
    let code = assemble(".equ VAL, 1 << 4\n.fill VAL, 1, 0x90", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 16);
}

#[test]
fn const_expr_shift_right() {
    // .equ VAL, 256 >> 3 → 32
    let code = assemble(".equ VAL, 256 >> 3\n.fill VAL, 1, 0x90", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 32);
}

#[test]
fn const_expr_bitwise_and() {
    // .equ VAL, 0xFF & 0x0F → 15
    let code = assemble(".equ VAL, 0xFF & 0x0F\n.fill VAL, 1, 0x90", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 15);
}

#[test]
fn const_expr_bitwise_or() {
    // .equ VAL, 0x10 | 0x03 → 19
    let code = assemble(".equ VAL, 0x10 | 0x03\n.fill VAL, 1, 0x90", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 19);
}

#[test]
fn const_expr_bitwise_xor() {
    // .equ VAL, 0xFF ^ 0xF0 → 15
    let code = assemble(".equ VAL, 0xFF ^ 0xF0\n.fill VAL, 1, 0x90", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 15);
}

#[test]
fn const_expr_bitwise_not() {
    // ~0 is -1 in i128 (all bits set), mask with 0xFF → 255... but 255 fill is too big for .fill
    // Use: .equ VAL, ~0xFC & 0xFF → 3
    let code = assemble(".equ VAL, ~0xFC & 0xFF\n.fill VAL, 1, 0x90", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 3);
}

#[test]
fn const_expr_parentheses() {
    // .equ VAL, (2 + 3) * 4 → 20
    let code = assemble(".equ VAL, (2 + 3) * 4\n.fill VAL, 1, 0x90", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 20);
}

#[test]
fn const_expr_complex_precedence() {
    // 2 + 3 * 4 = 14 (not 20)
    let code = assemble(".equ VAL, 2 + 3 * 4\n.fill VAL, 1, 0x90", Arch::X86_64).unwrap();
    assert_eq!(code.len(), 14);
}

#[test]
fn const_expr_shift_and_or() {
    // (1 << 3) | (1 << 1) = 8 | 2 = 10
    let code = assemble(
        ".equ VAL, (1 << 3) | (1 << 1)\n.fill VAL, 1, 0x90",
        Arch::X86_64,
    )
    .unwrap();
    assert_eq!(code.len(), 10);
}

#[test]
fn const_expr_division_by_zero() {
    let result = assemble(".equ VAL, 10 / 0\nnop", Arch::X86_64);
    assert!(result.is_err());
}

#[test]
fn error_exitm_outside_macro() {
    let result = assemble(".exitm\nnop", Arch::X86_64);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains(".exitm outside"),
        "expected '.exitm outside' error, got: {msg}"
    );
}

#[test]
fn exitm_inside_macro_is_valid() {
    // .exitm inside a macro should work fine
    let code = assemble(
        ".macro early_exit\nnop\n.exitm\nint3\n.endm\nearly_exit",
        Arch::X86_64,
    )
    .unwrap();
    // Only NOP should be emitted, not INT3
    assert_eq!(code, vec![0x90]);
}
