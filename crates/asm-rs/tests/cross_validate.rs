#![cfg(not(target_arch = "wasm32"))]
//! Cross-validation tests: encode with asm_rs, decode with iced-x86.
//!
//! Every instruction encoding is verified by decoding the output with iced-x86
//! and checking that the decoded mnemonic and operands match expectations.
//! This provides gold-standard validation against an independent, battle-tested
//! x86-64 decoder.

use asm_rs::{assemble, Arch};
use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter, Mnemonic as IcedMnemonic};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Assemble one instruction with asm_rs, decode with iced-x86, return (mnemonic, formatted).
fn asm_and_decode(source: &str) -> (IcedMnemonic, String) {
    let bytes = assemble(source, Arch::X86_64)
        .unwrap_or_else(|e| panic!("asm_rs failed to assemble `{source}`: {e}"));
    assert!(!bytes.is_empty(), "empty output for `{source}`");

    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    assert_ne!(
        instr.mnemonic(),
        IcedMnemonic::INVALID,
        "iced-x86 decoded INVALID for `{source}` → {:02X?}",
        bytes
    );
    // Verify the full instruction was consumed (no trailing bytes left unmatched)
    assert_eq!(
        instr.len(),
        bytes.len(),
        "iced-x86 decoded {} bytes but asm_rs emitted {} bytes for `{source}` → {:02X?}",
        instr.len(),
        bytes.len(),
        bytes
    );

    let mut formatter = IntelFormatter::new();
    let mut output = String::new();
    formatter.format(&instr, &mut output);
    (instr.mnemonic(), output)
}

/// Assemble + decode, then assert iced-x86 mnemonic matches expected.
fn verify_mnemonic(source: &str, expected: IcedMnemonic) {
    let (mnemonic, formatted) = asm_and_decode(source);
    assert_eq!(
        mnemonic, expected,
        "mnemonic mismatch for `{source}`: iced decoded `{formatted}`"
    );
}

/// Assemble + decode, verify the formatted disassembly starts with expected_name (case-insensitive).
fn verify_name(source: &str, expected_name: &str) {
    let (_mnemonic, formatted) = asm_and_decode(source);
    let lower = formatted.to_lowercase();
    assert!(
        lower.starts_with(&expected_name.to_lowercase()),
        "`{source}` decoded as `{formatted}`, expected to start with `{expected_name}`"
    );
}

/// Assemble + decode, then assert iced-x86 formatted output contains a substring.
fn verify_contains(source: &str, expected: IcedMnemonic, substring: &str) {
    let (mnemonic, formatted) = asm_and_decode(source);
    assert_eq!(
        mnemonic, expected,
        "mnemonic mismatch for `{source}`: iced decoded `{formatted}`"
    );
    let lower = formatted.to_lowercase();
    let sub_lower = substring.to_lowercase();
    assert!(
        lower.contains(&sub_lower),
        "`{source}` decoded as `{formatted}`, expected to contain `{substring}`"
    );
}

/// Assemble a multi-instruction source with labels, verify total byte count.
fn asm_bytes(source: &str) -> Vec<u8> {
    assemble(source, Arch::X86_64)
        .unwrap_or_else(|e| panic!("asm_rs failed: {e}\nsource: {source}"))
}

/// Assemble with explicit optimization level, decode with iced-x86.
fn asm_and_decode_opt(source: &str, opt: asm_rs::OptLevel) -> (IcedMnemonic, String) {
    let mut asm = asm_rs::Assembler::new(Arch::X86_64);
    asm.optimize(opt);
    asm.emit(source).unwrap();
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    assert!(!bytes.is_empty(), "empty output for `{source}`");

    let mut decoder = Decoder::with_ip(64, bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    assert_ne!(instr.mnemonic(), IcedMnemonic::INVALID);
    assert_eq!(instr.len(), bytes.len());

    let mut formatter = IntelFormatter::new();
    let mut output = String::new();
    formatter.format(&instr, &mut output);
    (instr.mnemonic(), output)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Zero-operand instructions
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_nop() {
    verify_mnemonic("nop", IcedMnemonic::Nop);
}

#[test]
fn xv_ret() {
    verify_name("ret", "ret");
}

#[test]
fn xv_syscall() {
    verify_mnemonic("syscall", IcedMnemonic::Syscall);
}

#[test]
fn xv_int3() {
    verify_mnemonic("int3", IcedMnemonic::Int3);
}

#[test]
fn xv_hlt() {
    verify_mnemonic("hlt", IcedMnemonic::Hlt);
}

#[test]
fn xv_leave() {
    verify_name("leave", "leave");
}

#[test]
fn xv_ud2() {
    verify_mnemonic("ud2", IcedMnemonic::Ud2);
}

#[test]
fn xv_cpuid() {
    verify_mnemonic("cpuid", IcedMnemonic::Cpuid);
}

#[test]
fn xv_rdtsc() {
    verify_mnemonic("rdtsc", IcedMnemonic::Rdtsc);
}

#[test]
fn xv_rdtscp() {
    verify_mnemonic("rdtscp", IcedMnemonic::Rdtscp);
}

#[test]
fn xv_pause() {
    verify_mnemonic("pause", IcedMnemonic::Pause);
}

#[test]
fn xv_mfence() {
    verify_mnemonic("mfence", IcedMnemonic::Mfence);
}

#[test]
fn xv_lfence() {
    verify_mnemonic("lfence", IcedMnemonic::Lfence);
}

#[test]
fn xv_sfence() {
    verify_mnemonic("sfence", IcedMnemonic::Sfence);
}

#[test]
fn xv_clc() {
    verify_mnemonic("clc", IcedMnemonic::Clc);
}

#[test]
fn xv_stc() {
    verify_mnemonic("stc", IcedMnemonic::Stc);
}

#[test]
fn xv_cmc() {
    verify_mnemonic("cmc", IcedMnemonic::Cmc);
}

#[test]
fn xv_cld() {
    verify_mnemonic("cld", IcedMnemonic::Cld);
}

#[test]
fn xv_std() {
    verify_mnemonic("std", IcedMnemonic::Std);
}

#[test]
fn xv_cli() {
    verify_mnemonic("cli", IcedMnemonic::Cli);
}

#[test]
fn xv_sti() {
    verify_mnemonic("sti", IcedMnemonic::Sti);
}

#[test]
fn xv_lahf() {
    verify_mnemonic("lahf", IcedMnemonic::Lahf);
}

#[test]
fn xv_sahf() {
    verify_mnemonic("sahf", IcedMnemonic::Sahf);
}

#[test]
fn xv_cdq() {
    verify_mnemonic("cdq", IcedMnemonic::Cdq);
}

#[test]
fn xv_cqo() {
    verify_mnemonic("cqo", IcedMnemonic::Cqo);
}

#[test]
fn xv_cbw() {
    verify_mnemonic("cbw", IcedMnemonic::Cbw);
}

#[test]
fn xv_cwde() {
    verify_mnemonic("cwde", IcedMnemonic::Cwde);
}

#[test]
fn xv_cdqe() {
    verify_mnemonic("cdqe", IcedMnemonic::Cdqe);
}

#[test]
fn xv_cwd() {
    verify_mnemonic("cwd", IcedMnemonic::Cwd);
}

#[test]
fn xv_pushfq() {
    verify_name("pushfq", "pushfq");
}

#[test]
fn xv_popfq() {
    verify_name("popfq", "popfq");
}

#[test]
fn xv_sysenter() {
    verify_mnemonic("sysenter", IcedMnemonic::Sysenter);
}

#[test]
fn xv_swapgs() {
    verify_mnemonic("swapgs", IcedMnemonic::Swapgs);
}

#[test]
fn xv_wrmsr() {
    verify_mnemonic("wrmsr", IcedMnemonic::Wrmsr);
}

#[test]
fn xv_rdmsr() {
    verify_mnemonic("rdmsr", IcedMnemonic::Rdmsr);
}

#[test]
fn xv_endbr64() {
    verify_mnemonic("endbr64", IcedMnemonic::Endbr64);
}

#[test]
fn xv_endbr32() {
    verify_mnemonic("endbr32", IcedMnemonic::Endbr32);
}

#[test]
fn xv_xlatb() {
    verify_name("xlatb", "xlat");
}

#[test]
fn xv_iretq() {
    verify_name("iretq", "iretq");
}

#[test]
fn xv_insb() {
    verify_mnemonic("insb", IcedMnemonic::Insb);
}

#[test]
fn xv_outsb() {
    verify_mnemonic("outsb", IcedMnemonic::Outsb);
}

#[test]
fn xv_emms() {
    verify_mnemonic("emms", IcedMnemonic::Emms);
}

#[test]
fn xv_xgetbv() {
    verify_mnemonic("xgetbv", IcedMnemonic::Xgetbv);
}

// ═══════════════════════════════════════════════════════════════════════════════
// String instructions
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_movsb() {
    verify_mnemonic("movsb", IcedMnemonic::Movsb);
}

#[test]
fn xv_movsq() {
    verify_mnemonic("movsq", IcedMnemonic::Movsq);
}

#[test]
fn xv_stosb() {
    verify_mnemonic("stosb", IcedMnemonic::Stosb);
}

#[test]
fn xv_stosq() {
    verify_mnemonic("stosq", IcedMnemonic::Stosq);
}

#[test]
fn xv_lodsb() {
    verify_mnemonic("lodsb", IcedMnemonic::Lodsb);
}

#[test]
fn xv_scasb() {
    verify_mnemonic("scasb", IcedMnemonic::Scasb);
}

#[test]
fn xv_cmpsb() {
    verify_mnemonic("cmpsb", IcedMnemonic::Cmpsb);
}

#[test]
fn xv_rep_movsb() {
    verify_mnemonic("rep movsb", IcedMnemonic::Movsb);
}

#[test]
fn xv_rep_stosq() {
    verify_mnemonic("rep stosq", IcedMnemonic::Stosq);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data movement
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_mov_reg_reg() {
    verify_contains("mov rax, rbx", IcedMnemonic::Mov, "rax");
}

#[test]
fn xv_mov_reg_imm() {
    verify_contains("mov eax, 42", IcedMnemonic::Mov, "eax");
}

#[test]
fn xv_mov_reg_mem() {
    verify_contains("mov rax, [rbx]", IcedMnemonic::Mov, "rax");
}

#[test]
fn xv_mov_mem_reg() {
    verify_contains("mov [rbx], rax", IcedMnemonic::Mov, "rax");
}

#[test]
fn xv_movzx_reg_reg() {
    verify_mnemonic("movzx eax, cl", IcedMnemonic::Movzx);
}

#[test]
fn xv_movsx_reg_reg() {
    verify_mnemonic("movsx rax, cl", IcedMnemonic::Movsx);
}

#[test]
fn xv_lea() {
    verify_contains("lea rax, [rbx+rcx*4]", IcedMnemonic::Lea, "rax");
}

#[test]
fn xv_xchg_reg_reg() {
    verify_mnemonic("xchg eax, ecx", IcedMnemonic::Xchg);
}

#[test]
fn xv_xchg_mem_reg() {
    verify_mnemonic("xchg [rbx], eax", IcedMnemonic::Xchg);
}

#[test]
fn xv_push_reg() {
    verify_name("push rbp", "push");
}

#[test]
fn xv_pop_reg() {
    verify_name("pop rbp", "pop");
}

#[test]
fn xv_push_imm() {
    verify_name("push 42", "push");
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALU operations
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_add_reg_reg() {
    verify_contains("add rax, rbx", IcedMnemonic::Add, "rax");
}

#[test]
fn xv_add_reg_imm() {
    verify_contains("add eax, 100", IcedMnemonic::Add, "eax");
}

#[test]
fn xv_sub_reg_reg() {
    verify_mnemonic("sub rcx, rdx", IcedMnemonic::Sub);
}

#[test]
fn xv_and_reg_imm() {
    verify_mnemonic("and eax, 0xFF", IcedMnemonic::And);
}

#[test]
fn xv_or_reg_reg() {
    verify_mnemonic("or rax, rbx", IcedMnemonic::Or);
}

#[test]
fn xv_xor_reg_reg() {
    verify_mnemonic("xor rax, rax", IcedMnemonic::Xor);
}

#[test]
fn xv_cmp_reg_imm() {
    verify_mnemonic("cmp eax, 0", IcedMnemonic::Cmp);
}

#[test]
fn xv_test_reg_reg() {
    verify_mnemonic("test eax, eax", IcedMnemonic::Test);
}

#[test]
fn xv_not_reg() {
    verify_mnemonic("not rax", IcedMnemonic::Not);
}

#[test]
fn xv_neg_reg() {
    verify_mnemonic("neg rax", IcedMnemonic::Neg);
}

#[test]
fn xv_inc_reg() {
    verify_mnemonic("inc rax", IcedMnemonic::Inc);
}

#[test]
fn xv_dec_reg() {
    verify_mnemonic("dec rax", IcedMnemonic::Dec);
}

#[test]
fn xv_mul_reg() {
    verify_mnemonic("mul rbx", IcedMnemonic::Mul);
}

#[test]
fn xv_div_reg() {
    verify_mnemonic("div rbx", IcedMnemonic::Div);
}

#[test]
fn xv_imul_two_operand() {
    verify_contains("imul rax, rbx", IcedMnemonic::Imul, "rax");
}

#[test]
fn xv_imul_three_operand() {
    verify_contains("imul rax, rbx, 10", IcedMnemonic::Imul, "rax");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shifts & rotates
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_shl_reg_imm() {
    verify_mnemonic("shl rax, 4", IcedMnemonic::Shl);
}

#[test]
fn xv_shr_reg_cl() {
    verify_mnemonic("shr rbx, cl", IcedMnemonic::Shr);
}

#[test]
fn xv_sar_reg_imm() {
    verify_mnemonic("sar rax, 1", IcedMnemonic::Sar);
}

#[test]
fn xv_rol_reg_imm() {
    verify_mnemonic("rol eax, 3", IcedMnemonic::Rol);
}

#[test]
fn xv_ror_reg_cl() {
    verify_mnemonic("ror rbx, cl", IcedMnemonic::Ror);
}

#[test]
fn xv_shld_reg_reg_imm() {
    verify_mnemonic("shld eax, ecx, 4", IcedMnemonic::Shld);
}

#[test]
fn xv_shld_reg_reg_cl() {
    verify_mnemonic("shld eax, ecx, cl", IcedMnemonic::Shld);
}

#[test]
fn xv_shrd_reg_reg_imm() {
    verify_mnemonic("shrd rax, rcx, 8", IcedMnemonic::Shrd);
}

#[test]
fn xv_shrd_reg_reg_cl() {
    verify_mnemonic("shrd eax, edx, cl", IcedMnemonic::Shrd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Control flow
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_jmp_label() {
    // jmp with forward label resolves to correct encoding
    let bytes = asm_bytes("jmp done\nnop\ndone: ret");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    let mut fmt = IntelFormatter::new();
    let mut out = String::new();
    fmt.format(&instr, &mut out);
    assert!(
        out.to_lowercase().starts_with("jmp"),
        "Expected jmp, got {out}"
    );
}

#[test]
fn xv_call_label() {
    let bytes = asm_bytes("call func\nret\nfunc: nop\nret");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    let mut fmt = IntelFormatter::new();
    let mut out = String::new();
    fmt.format(&instr, &mut out);
    assert!(
        out.to_lowercase().starts_with("call"),
        "Expected call, got {out}"
    );
}

#[test]
fn xv_jmp_reg() {
    verify_name("jmp rax", "jmp");
}

#[test]
fn xv_call_reg() {
    verify_name("call rax", "call");
}

#[test]
fn xv_je_label() {
    let bytes = asm_bytes("je done\nnop\ndone: ret");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    let mut fmt = IntelFormatter::new();
    let mut out = String::new();
    fmt.format(&instr, &mut out);
    assert!(
        out.to_lowercase().starts_with("je"),
        "Expected je, got {out}"
    );
}

#[test]
fn xv_jne_label() {
    let bytes = asm_bytes("jne done\nnop\ndone: ret");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    let mut fmt = IntelFormatter::new();
    let mut out = String::new();
    fmt.format(&instr, &mut out);
    assert!(
        out.to_lowercase().starts_with("jne"),
        "Expected jne, got {out}"
    );
}

// ─── SETcc / CMOVcc ──────────────────────────────────────────────────────

#[test]
fn xv_sete() {
    verify_mnemonic("sete al", IcedMnemonic::Sete);
}

#[test]
fn xv_setb() {
    verify_mnemonic("setb cl", IcedMnemonic::Setb);
}

#[test]
fn xv_cmove() {
    verify_mnemonic("cmove eax, ecx", IcedMnemonic::Cmove);
}

#[test]
fn xv_cmovne() {
    verify_mnemonic("cmovne rax, rbx", IcedMnemonic::Cmovne);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bit manipulation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_bt_reg_imm() {
    verify_mnemonic("bt eax, 5", IcedMnemonic::Bt);
}

#[test]
fn xv_bts_reg_reg() {
    verify_mnemonic("bts eax, ecx", IcedMnemonic::Bts);
}

#[test]
fn xv_bsf_reg_reg() {
    verify_mnemonic("bsf eax, ecx", IcedMnemonic::Bsf);
}

#[test]
fn xv_bsr_reg_reg() {
    verify_mnemonic("bsr eax, ecx", IcedMnemonic::Bsr);
}

#[test]
fn xv_popcnt_reg_reg() {
    verify_mnemonic("popcnt eax, ecx", IcedMnemonic::Popcnt);
}

#[test]
fn xv_lzcnt_reg_reg() {
    verify_mnemonic("lzcnt eax, ecx", IcedMnemonic::Lzcnt);
}

#[test]
fn xv_tzcnt_reg_reg() {
    verify_mnemonic("tzcnt eax, ecx", IcedMnemonic::Tzcnt);
}

#[test]
fn xv_bswap_eax() {
    verify_mnemonic("bswap eax", IcedMnemonic::Bswap);
}

#[test]
fn xv_bswap_rax() {
    verify_mnemonic("bswap rax", IcedMnemonic::Bswap);
}

// ═══════════════════════════════════════════════════════════════════════════════
// New table-driven instructions
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_cmpxchg_reg_reg() {
    verify_mnemonic("cmpxchg ecx, eax", IcedMnemonic::Cmpxchg);
}

#[test]
fn xv_cmpxchg_mem_reg() {
    verify_contains("cmpxchg [rbx], eax", IcedMnemonic::Cmpxchg, "eax");
}

#[test]
fn xv_lock_cmpxchg_mem_reg() {
    let bytes = asm_bytes("lock cmpxchg [rdi], rax");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    assert_eq!(instr.mnemonic(), IcedMnemonic::Cmpxchg);
    assert!(instr.has_lock_prefix());
}

#[test]
fn xv_xadd_reg_reg() {
    verify_mnemonic("xadd ecx, eax", IcedMnemonic::Xadd);
}

#[test]
fn xv_xadd_mem_reg() {
    verify_contains("xadd [rbx], eax", IcedMnemonic::Xadd, "eax");
}

#[test]
fn xv_in_al_imm() {
    verify_mnemonic("in al, 0x60", IcedMnemonic::In);
}

#[test]
fn xv_in_eax_dx() {
    verify_mnemonic("in eax, dx", IcedMnemonic::In);
}

#[test]
fn xv_out_imm_al() {
    verify_mnemonic("out 0x80, al", IcedMnemonic::Out);
}

#[test]
fn xv_out_dx_eax() {
    verify_mnemonic("out dx, eax", IcedMnemonic::Out);
}

#[test]
fn xv_enter() {
    verify_name("enter 256, 0", "enter");
}

#[test]
fn xv_rdrand_eax() {
    verify_contains("rdrand eax", IcedMnemonic::Rdrand, "eax");
}

#[test]
fn xv_rdrand_rax() {
    verify_contains("rdrand rax", IcedMnemonic::Rdrand, "rax");
}

#[test]
fn xv_rdseed_ecx() {
    verify_contains("rdseed ecx", IcedMnemonic::Rdseed, "ecx");
}

#[test]
fn xv_cmpxchg8b() {
    verify_mnemonic("cmpxchg8b [rdi]", IcedMnemonic::Cmpxchg8b);
}

#[test]
fn xv_cmpxchg16b() {
    let bytes = asm_bytes("cmpxchg16b [rdi]");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    assert_eq!(instr.mnemonic(), IcedMnemonic::Cmpxchg16b);
}

#[test]
fn xv_movnti() {
    verify_mnemonic("movnti [rbx], eax", IcedMnemonic::Movnti);
}

#[test]
fn xv_movnti_64() {
    verify_mnemonic("movnti [rbx], rax", IcedMnemonic::Movnti);
}

#[test]
fn xv_movbe_load() {
    verify_mnemonic("movbe eax, [rbx]", IcedMnemonic::Movbe);
}

#[test]
fn xv_movbe_store() {
    verify_mnemonic("movbe [rbx], eax", IcedMnemonic::Movbe);
}

// ═══════════════════════════════════════════════════════════════════════════════
// I/O port instructions with rep prefix
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_rep_insb() {
    let bytes = asm_bytes("rep insb");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    assert_eq!(instr.mnemonic(), IcedMnemonic::Insb);
    assert!(instr.has_rep_prefix() || instr.has_repe_prefix());
}

#[test]
fn xv_rep_outsb() {
    let bytes = asm_bytes("rep outsb");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    assert_eq!(instr.mnemonic(), IcedMnemonic::Outsb);
    assert!(instr.has_rep_prefix() || instr.has_repe_prefix());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Comprehensive ALU with memory operands
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_add_mem_reg() {
    verify_mnemonic("add [rbx], eax", IcedMnemonic::Add);
}

#[test]
fn xv_sub_reg_mem() {
    verify_mnemonic("sub eax, [rbx]", IcedMnemonic::Sub);
}

#[test]
fn xv_xor_mem_imm() {
    verify_mnemonic("xor dword [rbx], 0xFF", IcedMnemonic::Xor);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shellcode patterns — common exploit/pentest sequences
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_shellcode_linux_exit() {
    // Classic Linux x86-64 exit(0) shellcode
    let bytes = asm_bytes("xor edi, edi\nmov eax, 60\nsyscall");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let i1 = decoder.decode();
    let i2 = decoder.decode();
    let i3 = decoder.decode();
    assert_eq!(i1.mnemonic(), IcedMnemonic::Xor);
    assert_eq!(i2.mnemonic(), IcedMnemonic::Mov);
    assert_eq!(i3.mnemonic(), IcedMnemonic::Syscall);
}

#[test]
fn xv_shellcode_function_prologue() {
    // Standard function prologue
    let bytes = asm_bytes("push rbp\nmov rbp, rsp\nsub rsp, 32");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let i1 = decoder.decode();
    let i2 = decoder.decode();
    let i3 = decoder.decode();
    assert_eq!(i1.mnemonic(), IcedMnemonic::Push);
    assert_eq!(i2.mnemonic(), IcedMnemonic::Mov);
    assert_eq!(i3.mnemonic(), IcedMnemonic::Sub);
}

#[test]
fn xv_shellcode_function_epilogue() {
    let bytes = asm_bytes("leave\nret");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let i1 = decoder.decode();
    let i2 = decoder.decode();
    assert_eq!(i1.mnemonic(), IcedMnemonic::Leave);
    let mut fmt = IntelFormatter::new();
    let mut out = String::new();
    fmt.format(&i2, &mut out);
    assert!(
        out.to_lowercase().starts_with("ret"),
        "Expected ret, got {out}"
    );
}

#[test]
fn xv_shellcode_register_zeroing() {
    // Various register zeroing techniques used in shellcode
    let bytes = asm_bytes("xor eax, eax\nxor r8d, r8d");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let i1 = decoder.decode();
    let i2 = decoder.decode();
    assert_eq!(i1.mnemonic(), IcedMnemonic::Xor);
    assert_eq!(i2.mnemonic(), IcedMnemonic::Xor);
}

#[test]
fn xv_anti_debug_rdtsc_pair() {
    // Classic anti-debug: measure TSC delta
    let bytes = asm_bytes("rdtsc\nmov ecx, eax\nrdtsc\nsub eax, ecx");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Rdtsc);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Mov);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Rdtsc);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Sub);
}

#[test]
fn xv_cet_aware_shellcode() {
    // CET-compliant function entry
    let bytes = asm_bytes("endbr64\npush rbp\nmov rbp, rsp\npop rbp\nret");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Endbr64);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Push);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Mov);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Pop);
    let ret_instr = decoder.decode();
    let mut fmt = IntelFormatter::new();
    let mut out = String::new();
    fmt.format(&ret_instr, &mut out);
    assert!(
        out.to_lowercase().starts_with("ret"),
        "Expected ret, got {out}"
    );
}

#[test]
fn xv_atomic_compare_and_swap() {
    // lock cmpxchg pattern for atomic CAS
    let bytes = asm_bytes("lock cmpxchg [rdi], rsi");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    assert_eq!(instr.mnemonic(), IcedMnemonic::Cmpxchg);
    assert!(instr.has_lock_prefix());
    assert_eq!(instr.len(), bytes.len());
}

#[test]
fn xv_spinlock_pause() {
    // Spinlock wait loop pattern
    let bytes = asm_bytes("pause\nnop");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Pause);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Nop);
}

// ─── New Feature Cross-Validation ─────────────────────────────────────────────

#[test]
fn xv_ret_imm16() {
    // ret 8 → C2 08 00
    verify_mnemonic("ret 8", IcedMnemonic::Ret);
}

#[test]
fn xv_ret_imm16_large() {
    // ret 0x1234 → C2 34 12
    verify_mnemonic("ret 0x1234", IcedMnemonic::Ret);
}

#[test]
fn xv_retf() {
    // retf → CB (far return)
    verify_mnemonic("retf", IcedMnemonic::Retf);
}

#[test]
fn xv_retf_imm16() {
    // retf 4 → CA 04 00 (far return with stack cleanup)
    verify_mnemonic("retf 4", IcedMnemonic::Retf);
}

#[test]
fn xv_lret() {
    // lret is alias for retf
    verify_mnemonic("lret", IcedMnemonic::Retf);
}

#[test]
fn xv_lret_imm16() {
    // lret 8 = retf 8
    verify_mnemonic("lret 8", IcedMnemonic::Retf);
}

#[test]
fn xv_movabs_r64_imm64() {
    // movabs rax, 0x1122334455667788
    verify_mnemonic("movabs rax, 0x1122334455667788", IcedMnemonic::Mov);
}

#[test]
fn xv_movabs_rdi_imm64() {
    verify_mnemonic("movabs rdi, 0xdeadbeefcafebabe", IcedMnemonic::Mov);
}

#[test]
fn xv_push_fs() {
    verify_mnemonic("push fs", IcedMnemonic::Push);
}

#[test]
fn xv_push_gs() {
    verify_mnemonic("push gs", IcedMnemonic::Push);
}

#[test]
fn xv_pop_fs() {
    verify_mnemonic("pop fs", IcedMnemonic::Pop);
}

#[test]
fn xv_pop_gs() {
    verify_mnemonic("pop gs", IcedMnemonic::Pop);
}

#[test]
fn xv_xchg_eax_eax_is_nop() {
    // xchg eax, eax decodes as NOP (0x90)
    let bytes = asm_bytes("xchg eax, eax");
    assert_eq!(bytes, vec![0x90]);
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    assert_eq!(instr.mnemonic(), IcedMnemonic::Nop);
}

// ─── 8-bit Operand Forms ──────────────────────────────────────────────────────

#[test]
fn xv_mov_al_bl() {
    verify_mnemonic("mov al, bl", IcedMnemonic::Mov);
}

#[test]
fn xv_add_cl_1() {
    verify_mnemonic("add cl, 1", IcedMnemonic::Add);
}

#[test]
fn xv_xor_dl_dl() {
    verify_mnemonic("xor dl, dl", IcedMnemonic::Xor);
}

#[test]
fn xv_cmp_al_0x41() {
    verify_mnemonic("cmp al, 0x41", IcedMnemonic::Cmp);
}

#[test]
fn xv_or_bl_0xff() {
    verify_mnemonic("or bl, 0xff", IcedMnemonic::Or);
}

#[test]
fn xv_sub_al_bl() {
    verify_mnemonic("sub al, bl", IcedMnemonic::Sub);
}

#[test]
fn xv_and_cl_dl() {
    verify_mnemonic("and cl, dl", IcedMnemonic::And);
}

// ─── 16-bit Operand Forms ─────────────────────────────────────────────────────

#[test]
fn xv_add_ax_bx() {
    verify_mnemonic("add ax, bx", IcedMnemonic::Add);
}

#[test]
fn xv_sub_cx_dx() {
    verify_mnemonic("sub cx, dx", IcedMnemonic::Sub);
}

#[test]
fn xv_mov_ax_0x1234() {
    verify_mnemonic("mov ax, 0x1234", IcedMnemonic::Mov);
}

#[test]
fn xv_xor_si_si() {
    verify_mnemonic("xor si, si", IcedMnemonic::Xor);
}

#[test]
fn xv_cmp_ax_0() {
    verify_mnemonic("cmp ax, 0", IcedMnemonic::Cmp);
}

#[test]
fn xv_inc_dx() {
    verify_name("inc dx", "inc");
}

#[test]
fn xv_dec_bx() {
    verify_name("dec bx", "dec");
}

// ─── Extended Register Forms ──────────────────────────────────────────────────

#[test]
fn xv_add_r8_r9() {
    verify_mnemonic("add r8, r9", IcedMnemonic::Add);
}

#[test]
fn xv_mov_r15d_r12d() {
    verify_mnemonic("mov r15d, r12d", IcedMnemonic::Mov);
}

#[test]
fn xv_xor_r10_r10() {
    verify_mnemonic("xor r10, r10", IcedMnemonic::Xor);
}

#[test]
fn xv_sub_r11_r13() {
    verify_mnemonic("sub r11, r13", IcedMnemonic::Sub);
}

#[test]
fn xv_cmp_r14d_0() {
    verify_mnemonic("cmp r14d, 0", IcedMnemonic::Cmp);
}

#[test]
fn xv_and_r8d_r9d() {
    verify_mnemonic("and r8d, r9d", IcedMnemonic::And);
}

#[test]
fn xv_or_r12_r13() {
    verify_mnemonic("or r12, r13", IcedMnemonic::Or);
}

#[test]
fn xv_xchg_r8_r9() {
    verify_mnemonic("xchg r8, r9", IcedMnemonic::Xchg);
}

#[test]
fn xv_mov_r8b_r9b() {
    verify_mnemonic("mov r8b, r9b", IcedMnemonic::Mov);
}

#[test]
fn xv_add_r8w_r9w() {
    verify_mnemonic("add r8w, r9w", IcedMnemonic::Add);
}

// ─── Memory Operand Forms ─────────────────────────────────────────────────────

#[test]
fn xv_inc_byte_ptr_rax() {
    verify_name("inc byte ptr [rax]", "inc");
}

#[test]
fn xv_inc_dword_ptr_rax() {
    verify_name("inc dword ptr [rax]", "inc");
}

#[test]
fn xv_inc_qword_ptr_rax() {
    verify_name("inc qword ptr [rax]", "inc");
}

#[test]
fn xv_add_dword_ptr_rsi_1() {
    verify_mnemonic("add dword ptr [rsi], 1", IcedMnemonic::Add);
}

#[test]
fn xv_mov_qword_ptr_rdi_rax() {
    verify_mnemonic("mov qword ptr [rdi], rax", IcedMnemonic::Mov);
}

#[test]
fn xv_mov_rax_qword_ptr_rbx() {
    verify_mnemonic("mov rax, qword ptr [rbx]", IcedMnemonic::Mov);
}

#[test]
fn xv_cmp_byte_ptr_rdi_0() {
    verify_mnemonic("cmp byte ptr [rdi], 0", IcedMnemonic::Cmp);
}

#[test]
fn xv_test_dword_ptr_rax_0xff() {
    verify_mnemonic("test dword ptr [rax], 0xff", IcedMnemonic::Test);
}

#[test]
fn xv_sub_qword_ptr_rsp_8() {
    verify_mnemonic("sub qword ptr [rsp], 8", IcedMnemonic::Sub);
}

#[test]
fn xv_xor_dword_ptr_rbp_esi() {
    verify_mnemonic("xor dword ptr [rbp], esi", IcedMnemonic::Xor);
}

// ─── High-byte Registers (correctness) ───────────────────────────────────────

#[test]
fn xv_mov_ah_0x42() {
    // mov ah, 0x42 → B4 42 (verifies correct base_code for AH=4)
    let (mnemonic, formatted) = asm_and_decode("mov ah, 0x42");
    assert_eq!(mnemonic, IcedMnemonic::Mov);
    assert!(
        formatted.to_lowercase().contains("ah"),
        "Expected AH in `{formatted}`"
    );
}

#[test]
fn xv_mov_bh_0x33() {
    let (mnemonic, formatted) = asm_and_decode("mov bh, 0x33");
    assert_eq!(mnemonic, IcedMnemonic::Mov);
    assert!(
        formatted.to_lowercase().contains("bh"),
        "Expected BH in `{formatted}`"
    );
}

#[test]
fn xv_mov_ch_0x11() {
    let (mnemonic, formatted) = asm_and_decode("mov ch, 0x11");
    assert_eq!(mnemonic, IcedMnemonic::Mov);
    assert!(
        formatted.to_lowercase().contains("ch"),
        "Expected CH in `{formatted}`"
    );
}

#[test]
fn xv_mov_dh_0x22() {
    let (mnemonic, formatted) = asm_and_decode("mov dh, 0x22");
    assert_eq!(mnemonic, IcedMnemonic::Mov);
    assert!(
        formatted.to_lowercase().contains("dh"),
        "Expected DH in `{formatted}`"
    );
}

#[test]
fn xv_xor_ah_ch() {
    // xor ah, ch — both high-byte regs, no REX needed
    let (mnemonic, formatted) = asm_and_decode("xor ah, ch");
    assert_eq!(mnemonic, IcedMnemonic::Xor);
    assert!(
        formatted.to_lowercase().contains("ah"),
        "Expected AH in `{formatted}`"
    );
}

#[test]
fn xv_mov_ah_al() {
    // mov ah, al — mix of high-byte and low-byte
    let (mnemonic, formatted) = asm_and_decode("mov ah, al");
    assert_eq!(mnemonic, IcedMnemonic::Mov);
    assert!(
        formatted.to_lowercase().contains("ah"),
        "Expected AH in `{formatted}`"
    );
}

// ─── SIB addressing forms ────────────────────────────────────────────────────

#[test]
fn xv_mov_rax_rdi_rsi_8() {
    // mov rax, [rdi + rsi*8]
    verify_mnemonic("mov rax, [rdi + rsi*8]", IcedMnemonic::Mov);
}

#[test]
fn xv_lea_rax_rbx_rcx_4_16() {
    // lea rax, [rbx + rcx*4 + 16]
    verify_mnemonic("lea rax, [rbx + rcx*4 + 16]", IcedMnemonic::Lea);
}

#[test]
fn xv_add_dword_ptr_rbx_rcx_2_eax() {
    // add [rbx + rcx*2], eax
    verify_mnemonic("add dword ptr [rbx + rcx*2], eax", IcedMnemonic::Add);
}

// ─── LOCK prefix cross-validation ────────────────────────────────────────────

#[test]
fn xv_lock_inc_qword_ptr() {
    let bytes = asm_bytes("lock inc qword ptr [rdi]");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    assert_eq!(instr.mnemonic(), IcedMnemonic::Inc);
    assert!(instr.has_lock_prefix());
    assert_eq!(instr.len(), bytes.len());
}

#[test]
fn xv_lock_xadd_mem() {
    let bytes = asm_bytes("lock xadd [rdi], eax");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    assert_eq!(instr.mnemonic(), IcedMnemonic::Xadd);
    assert!(instr.has_lock_prefix());
    assert_eq!(instr.len(), bytes.len());
}

// ─── Shellcode / Security Patterns ───────────────────────────────────────────

#[test]
fn xv_push_pop_fs_gs_pattern() {
    // Save/restore FS/GS (used in TEB/TIB access on Windows, glibc TLS on Linux)
    let bytes = asm_bytes("push fs\npush gs\npop gs\npop fs");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Push);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Push);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Pop);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Pop);
}

#[test]
fn xv_ret_gadget_with_offset() {
    // ROP gadget: ret imm16 (skip N bytes on stack after return)
    let bytes = asm_bytes("pop rdi\nret 0x10");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Pop);
    let ret_instr = decoder.decode();
    assert_eq!(ret_instr.mnemonic(), IcedMnemonic::Ret);
    assert_eq!(ret_instr.len() + 1, bytes.len()); // pop rdi is 1 byte
}

#[test]
fn xv_tls_access_pattern() {
    // Typical Linux TLS access: mov rax, qword ptr fs:[0]
    // Encoded as: 64 48 8B 04 25 00 00 00 00
    // We test with push/pop fs + normal memory access instead
    let bytes = asm_bytes("push fs\nmov rax, [rdi]\npop fs");
    let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Push);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Mov);
    assert_eq!(decoder.decode().mnemonic(), IcedMnemonic::Pop);
}

// ─── Shift on Memory (size_hint fix verification) ────────────────────────────

#[test]
fn xv_shl_byte_ptr_mem() {
    verify_name("shl byte ptr [rbx], 1", "shl");
}

#[test]
fn xv_shr_qword_ptr_mem_cl() {
    verify_name("shr qword ptr [rax], cl", "shr");
}

#[test]
fn xv_sar_word_ptr_mem_4() {
    verify_name("sar word ptr [rcx], 4", "sar");
}

#[test]
fn xv_shl_dword_ptr_mem_cl() {
    verify_name("shl dword ptr [rdx], cl", "shl");
}

#[test]
fn xv_ror_byte_ptr_mem_1() {
    verify_name("ror byte ptr [rsi], 1", "ror");
}

#[test]
fn xv_rol_qword_ptr_mem_3() {
    verify_name("rol qword ptr [rdi], 3", "rol");
}

// ─── Inc/Dec on memory (cross-validate size hints) ──────────────────────────

#[test]
fn xv_inc_word_ptr_rax() {
    verify_name("inc word ptr [rax]", "inc");
}

#[test]
fn xv_dec_qword_ptr_rbx() {
    verify_name("dec qword ptr [rbx]", "dec");
}

// ─── NOT/NEG on memory ──────────────────────────────────────────────────────

#[test]
fn xv_not_dword_ptr_rax() {
    verify_mnemonic("not dword ptr [rax]", IcedMnemonic::Not);
}

#[test]
fn xv_neg_qword_ptr_rdi() {
    verify_mnemonic("neg qword ptr [rdi]", IcedMnemonic::Neg);
}

// ─── BT family cross-validation ──────────────────────────────────────────────

#[test]
fn xv_bt_eax_5() {
    verify_mnemonic("bt eax, 5", IcedMnemonic::Bt);
}

#[test]
fn xv_bts_ecx_edx() {
    verify_mnemonic("bts ecx, edx", IcedMnemonic::Bts);
}

#[test]
fn xv_btr_reg_imm() {
    verify_mnemonic("btr r8d, 3", IcedMnemonic::Btr);
}

#[test]
fn xv_btc_reg_reg64() {
    verify_mnemonic("btc rax, rbx", IcedMnemonic::Btc);
}

// ── Segment Override Prefix ──────────────────────────────────────────────

#[test]
fn xv_segment_fs_mov() {
    verify_mnemonic("mov rax, fs:[rbx]", IcedMnemonic::Mov);
}

#[test]
fn xv_segment_gs_mov() {
    verify_mnemonic("mov eax, gs:[rdx]", IcedMnemonic::Mov);
}

#[test]
fn xv_segment_fs_add() {
    verify_mnemonic("add rax, fs:[rdi]", IcedMnemonic::Add);
}

#[test]
fn xv_segment_gs_cmp() {
    verify_mnemonic("cmp dword ptr gs:[rbx+8], 0", IcedMnemonic::Cmp);
}

#[test]
fn xv_segment_fs_extended_reg() {
    verify_mnemonic("mov rax, fs:[r12]", IcedMnemonic::Mov);
}

// ── Push/Pop/Jmp/Call [mem] no redundant REX.W ──────────────────────────

#[test]
fn xv_push_mem() {
    let (mnem, _) = asm_and_decode("push qword ptr [rdi]");
    assert_eq!(mnem, IcedMnemonic::Push);
    // Verify no REX.W: assembled bytes should be FF 37 (2 bytes)
    let bytes = assemble("push qword ptr [rdi]", Arch::X86_64).unwrap();
    assert_eq!(bytes.len(), 2, "push [rdi] should be 2 bytes (no REX.W)");
}

#[test]
fn xv_pop_mem() {
    let (mnem, _) = asm_and_decode("pop qword ptr [rdi]");
    assert_eq!(mnem, IcedMnemonic::Pop);
    let bytes = assemble("pop qword ptr [rdi]", Arch::X86_64).unwrap();
    assert_eq!(bytes.len(), 2, "pop [rdi] should be 2 bytes (no REX.W)");
}

#[test]
fn xv_jmp_mem() {
    verify_mnemonic("jmp qword ptr [rdi]", IcedMnemonic::Jmp);
    let bytes = assemble("jmp qword ptr [rdi]", Arch::X86_64).unwrap();
    assert_eq!(bytes.len(), 2, "jmp [rdi] should be 2 bytes (no REX.W)");
}

#[test]
fn xv_call_mem() {
    verify_mnemonic("call qword ptr [rdi]", IcedMnemonic::Call);
    let bytes = assemble("call qword ptr [rdi]", Arch::X86_64).unwrap();
    assert_eq!(bytes.len(), 2, "call [rdi] should be 2 bytes (no REX.W)");
}

#[test]
fn xv_push_mem_extended() {
    verify_mnemonic("push qword ptr [r15]", IcedMnemonic::Push);
    let bytes = assemble("push qword ptr [r15]", Arch::X86_64).unwrap();
    assert_eq!(bytes[0], 0x41, "push [r15] needs REX.B");
}

// ── Xchg short form both directions ──────────────────────────────────────

#[test]
fn xv_xchg_rax_rbx() {
    verify_mnemonic("xchg rax, rbx", IcedMnemonic::Xchg);
}

#[test]
fn xv_xchg_rbx_rax() {
    // xchg rbx, rax should produce same encoding as xchg rax, rbx
    let bytes1 = assemble("xchg rax, rbx", Arch::X86_64).unwrap();
    let bytes2 = assemble("xchg rbx, rax", Arch::X86_64).unwrap();
    assert_eq!(
        bytes1, bytes2,
        "xchg is commutative — both directions should use short form"
    );
    verify_mnemonic("xchg rbx, rax", IcedMnemonic::Xchg);
}

// ── Bswap ────────────────────────────────────────────────────────────────

#[test]
fn xv_bswap_r12d() {
    verify_mnemonic("bswap r12d", IcedMnemonic::Bswap);
}

// ── SSE/SSE2 arithmetic ─────────────────────────────────────────────────

#[test]
fn xv_addps_xmm_xmm() {
    verify_mnemonic("addps xmm0, xmm1", IcedMnemonic::Addps);
}

#[test]
fn xv_addpd_xmm_xmm() {
    verify_mnemonic("addpd xmm0, xmm1", IcedMnemonic::Addpd);
}

#[test]
fn xv_addss_xmm_xmm() {
    verify_mnemonic("addss xmm0, xmm1", IcedMnemonic::Addss);
}

#[test]
fn xv_addsd_xmm_xmm() {
    verify_mnemonic("addsd xmm0, xmm1", IcedMnemonic::Addsd);
}

#[test]
fn xv_subps_xmm_xmm() {
    verify_mnemonic("subps xmm0, xmm1", IcedMnemonic::Subps);
}

#[test]
fn xv_subsd_xmm_xmm() {
    verify_mnemonic("subsd xmm2, xmm3", IcedMnemonic::Subsd);
}

#[test]
fn xv_mulps_xmm_xmm() {
    verify_mnemonic("mulps xmm0, xmm1", IcedMnemonic::Mulps);
}

#[test]
fn xv_mulsd_xmm_xmm() {
    verify_mnemonic("mulsd xmm0, xmm1", IcedMnemonic::Mulsd);
}

#[test]
fn xv_divps_xmm_xmm() {
    verify_mnemonic("divps xmm0, xmm1", IcedMnemonic::Divps);
}

#[test]
fn xv_divsd_xmm_xmm() {
    verify_mnemonic("divsd xmm4, xmm5", IcedMnemonic::Divsd);
}

#[test]
fn xv_sqrtps_xmm_xmm() {
    verify_mnemonic("sqrtps xmm0, xmm1", IcedMnemonic::Sqrtps);
}

#[test]
fn xv_sqrtsd_xmm_xmm() {
    verify_mnemonic("sqrtsd xmm0, xmm1", IcedMnemonic::Sqrtsd);
}

#[test]
fn xv_rcpps_xmm_xmm() {
    verify_mnemonic("rcpps xmm0, xmm1", IcedMnemonic::Rcpps);
}

#[test]
fn xv_rsqrtps_xmm_xmm() {
    verify_mnemonic("rsqrtps xmm0, xmm1", IcedMnemonic::Rsqrtps);
}

#[test]
fn xv_maxps_xmm_xmm() {
    verify_mnemonic("maxps xmm0, xmm1", IcedMnemonic::Maxps);
}

#[test]
fn xv_minsd_xmm_xmm() {
    verify_mnemonic("minsd xmm0, xmm1", IcedMnemonic::Minsd);
}

// ── SSE/SSE2 logical ────────────────────────────────────────────────────

#[test]
fn xv_xorps_xmm_xmm() {
    verify_mnemonic("xorps xmm0, xmm0", IcedMnemonic::Xorps);
}

#[test]
fn xv_xorpd_xmm_xmm() {
    verify_mnemonic("xorpd xmm0, xmm0", IcedMnemonic::Xorpd);
}

#[test]
fn xv_andps_xmm_xmm() {
    verify_mnemonic("andps xmm0, xmm1", IcedMnemonic::Andps);
}

#[test]
fn xv_orps_xmm_xmm() {
    verify_mnemonic("orps xmm0, xmm1", IcedMnemonic::Orps);
}

#[test]
fn xv_andnps_xmm_xmm() {
    verify_mnemonic("andnps xmm0, xmm1", IcedMnemonic::Andnps);
}

// ── SSE/SSE2 comparison ────────────────────────────────────────────────

#[test]
fn xv_comiss_xmm_xmm() {
    verify_mnemonic("comiss xmm0, xmm1", IcedMnemonic::Comiss);
}

#[test]
fn xv_comisd_xmm_xmm() {
    verify_mnemonic("comisd xmm0, xmm1", IcedMnemonic::Comisd);
}

#[test]
fn xv_ucomiss_xmm_xmm() {
    verify_mnemonic("ucomiss xmm0, xmm1", IcedMnemonic::Ucomiss);
}

#[test]
fn xv_ucomisd_xmm_xmm() {
    verify_mnemonic("ucomisd xmm0, xmm1", IcedMnemonic::Ucomisd);
}

// ── SSE/SSE2 data movement ─────────────────────────────────────────────

#[test]
fn xv_movaps_xmm_xmm() {
    verify_mnemonic("movaps xmm0, xmm1", IcedMnemonic::Movaps);
}

#[test]
fn xv_movups_xmm_xmm() {
    verify_mnemonic("movups xmm0, xmm1", IcedMnemonic::Movups);
}

#[test]
fn xv_movapd_xmm_xmm() {
    verify_mnemonic("movapd xmm0, xmm1", IcedMnemonic::Movapd);
}

#[test]
fn xv_movupd_xmm_xmm() {
    verify_mnemonic("movupd xmm0, xmm1", IcedMnemonic::Movupd);
}

#[test]
fn xv_movdqa_xmm_xmm() {
    verify_mnemonic("movdqa xmm0, xmm1", IcedMnemonic::Movdqa);
}

#[test]
fn xv_movdqu_xmm_xmm() {
    verify_mnemonic("movdqu xmm0, xmm1", IcedMnemonic::Movdqu);
}

#[test]
fn xv_movss_xmm_xmm() {
    verify_mnemonic("movss xmm0, xmm1", IcedMnemonic::Movss);
}

#[test]
fn xv_movsd_xmm_xmm() {
    verify_mnemonic("movsd xmm0, xmm1", IcedMnemonic::Movsd);
}

// ── SSE/SSE2 extended register (REX) ────────────────────────────────────

#[test]
fn xv_addps_xmm8_xmm9() {
    verify_mnemonic("addps xmm8, xmm9", IcedMnemonic::Addps);
}

#[test]
fn xv_movaps_xmm15_xmm0() {
    verify_mnemonic("movaps xmm15, xmm0", IcedMnemonic::Movaps);
}

#[test]
fn xv_xorps_xmm10_xmm10() {
    verify_mnemonic("xorps xmm10, xmm10", IcedMnemonic::Xorps);
}

// ── SSE/SSE2 with memory ───────────────────────────────────────────────

#[test]
fn xv_movaps_xmm_mem() {
    verify_mnemonic("movaps xmm0, [rax]", IcedMnemonic::Movaps);
}

#[test]
fn xv_movups_mem_xmm() {
    verify_mnemonic("movups [rdi], xmm1", IcedMnemonic::Movups);
}

#[test]
fn xv_addss_xmm_mem() {
    verify_mnemonic("addss xmm0, [rbx]", IcedMnemonic::Addss);
}

#[test]
fn xv_mulpd_xmm_mem() {
    verify_mnemonic("mulpd xmm0, [rcx+rdx*8]", IcedMnemonic::Mulpd);
}

// ── SSE integer (packed) ────────────────────────────────────────────────

#[test]
fn xv_paddb_xmm_xmm() {
    verify_mnemonic("paddb xmm0, xmm1", IcedMnemonic::Paddb);
}

#[test]
fn xv_paddw_xmm_xmm() {
    verify_mnemonic("paddw xmm0, xmm1", IcedMnemonic::Paddw);
}

#[test]
fn xv_paddd_xmm_xmm() {
    verify_mnemonic("paddd xmm0, xmm1", IcedMnemonic::Paddd);
}

#[test]
fn xv_paddq_xmm_xmm() {
    verify_mnemonic("paddq xmm0, xmm1", IcedMnemonic::Paddq);
}

#[test]
fn xv_psubb_xmm_xmm() {
    verify_mnemonic("psubb xmm0, xmm1", IcedMnemonic::Psubb);
}

#[test]
fn xv_pmullw_xmm_xmm() {
    verify_mnemonic("pmullw xmm0, xmm1", IcedMnemonic::Pmullw);
}

#[test]
fn xv_pcmpeqb_xmm_xmm() {
    verify_mnemonic("pcmpeqb xmm0, xmm1", IcedMnemonic::Pcmpeqb);
}

#[test]
fn xv_pcmpgtd_xmm_xmm() {
    verify_mnemonic("pcmpgtd xmm0, xmm1", IcedMnemonic::Pcmpgtd);
}

#[test]
fn xv_pxor_xmm_xmm() {
    verify_mnemonic("pxor xmm0, xmm0", IcedMnemonic::Pxor);
}

#[test]
fn xv_por_xmm_xmm() {
    verify_mnemonic("por xmm0, xmm1", IcedMnemonic::Por);
}

#[test]
fn xv_pand_xmm_xmm() {
    verify_mnemonic("pand xmm0, xmm1", IcedMnemonic::Pand);
}

#[test]
fn xv_pandn_xmm_xmm() {
    verify_mnemonic("pandn xmm0, xmm1", IcedMnemonic::Pandn);
}

// ── SSE with immediate ─────────────────────────────────────────────────

#[test]
fn xv_shufps_xmm_xmm_imm() {
    verify_mnemonic("shufps xmm0, xmm1, 0", IcedMnemonic::Shufps);
}

#[test]
fn xv_shufpd_xmm_xmm_imm() {
    verify_mnemonic("shufpd xmm0, xmm1, 1", IcedMnemonic::Shufpd);
}

#[test]
fn xv_cmpps_xmm_xmm_imm() {
    verify_mnemonic("cmpps xmm0, xmm1, 0", IcedMnemonic::Cmpps);
}

#[test]
fn xv_pshufd_xmm_xmm_imm() {
    verify_mnemonic("pshufd xmm0, xmm1, 0xFF", IcedMnemonic::Pshufd);
}

// ── SSE3 ────────────────────────────────────────────────────────────────

#[test]
fn xv_addsubps_xmm_xmm() {
    verify_mnemonic("addsubps xmm0, xmm1", IcedMnemonic::Addsubps);
}

#[test]
fn xv_haddps_xmm_xmm() {
    verify_mnemonic("haddps xmm0, xmm1", IcedMnemonic::Haddps);
}

#[test]
fn xv_hsubpd_xmm_xmm() {
    verify_mnemonic("hsubpd xmm0, xmm1", IcedMnemonic::Hsubpd);
}

#[test]
fn xv_movddup_xmm_xmm() {
    verify_mnemonic("movddup xmm0, xmm1", IcedMnemonic::Movddup);
}

// ── SSSE3 (0F 38) ──────────────────────────────────────────────────────

#[test]
fn xv_pshufb_xmm_xmm() {
    verify_mnemonic("pshufb xmm0, xmm1", IcedMnemonic::Pshufb);
}

#[test]
fn xv_pabsb_xmm_xmm() {
    verify_mnemonic("pabsb xmm0, xmm1", IcedMnemonic::Pabsb);
}

#[test]
fn xv_pmaddubsw_xmm_xmm() {
    verify_mnemonic("pmaddubsw xmm0, xmm1", IcedMnemonic::Pmaddubsw);
}

// ── SSE4.1 (0F 38) ─────────────────────────────────────────────────────

#[test]
fn xv_ptest_xmm_xmm() {
    verify_mnemonic("ptest xmm0, xmm1", IcedMnemonic::Ptest);
}

#[test]
fn xv_pmovzxbw_xmm_xmm() {
    verify_mnemonic("pmovzxbw xmm0, xmm1", IcedMnemonic::Pmovzxbw);
}

#[test]
fn xv_pmovsxwd_xmm_xmm() {
    verify_mnemonic("pmovsxwd xmm0, xmm1", IcedMnemonic::Pmovsxwd);
}

#[test]
fn xv_pmulld_xmm_xmm() {
    verify_mnemonic("pmulld xmm0, xmm1", IcedMnemonic::Pmulld);
}

#[test]
fn xv_phminposuw_xmm_xmm() {
    verify_mnemonic("phminposuw xmm0, xmm1", IcedMnemonic::Phminposuw);
}

#[test]
fn xv_packusdw_xmm_xmm() {
    verify_mnemonic("packusdw xmm0, xmm1", IcedMnemonic::Packusdw);
}

// ── SSE4.1 (0F 3A) with imm ────────────────────────────────────────────

#[test]
fn xv_roundps_xmm_xmm_imm() {
    verify_mnemonic("roundps xmm0, xmm1, 0", IcedMnemonic::Roundps);
}

#[test]
fn xv_roundsd_xmm_xmm_imm() {
    verify_mnemonic("roundsd xmm0, xmm1, 3", IcedMnemonic::Roundsd);
}

#[test]
fn xv_blendps_xmm_xmm_imm() {
    verify_mnemonic("blendps xmm0, xmm1, 5", IcedMnemonic::Blendps);
}

#[test]
fn xv_dpps_xmm_xmm_imm() {
    verify_mnemonic("dpps xmm0, xmm1, 0xFF", IcedMnemonic::Dpps);
}

#[test]
fn xv_insertps_xmm_xmm_imm() {
    verify_mnemonic("insertps xmm0, xmm1, 0x10", IcedMnemonic::Insertps);
}

#[test]
fn xv_pblendw_xmm_xmm_imm() {
    verify_mnemonic("pblendw xmm0, xmm1, 0xAA", IcedMnemonic::Pblendw);
}

// ── SSE4.2 ──────────────────────────────────────────────────────────────

#[test]
fn xv_pcmpgtq_xmm_xmm() {
    verify_mnemonic("pcmpgtq xmm0, xmm1", IcedMnemonic::Pcmpgtq);
}

#[test]
fn xv_pcmpistrm_xmm_xmm_imm() {
    verify_mnemonic("pcmpistrm xmm0, xmm1, 0", IcedMnemonic::Pcmpistrm);
}

#[test]
fn xv_pcmpistri_xmm_xmm_imm() {
    verify_mnemonic("pcmpistri xmm0, xmm1, 0", IcedMnemonic::Pcmpistri);
}

// ── AES-NI ──────────────────────────────────────────────────────────────

#[test]
fn xv_aesenc_xmm_xmm() {
    verify_mnemonic("aesenc xmm0, xmm1", IcedMnemonic::Aesenc);
}

#[test]
fn xv_aesenclast_xmm_xmm() {
    verify_mnemonic("aesenclast xmm0, xmm1", IcedMnemonic::Aesenclast);
}

#[test]
fn xv_aesdec_xmm_xmm() {
    verify_mnemonic("aesdec xmm0, xmm1", IcedMnemonic::Aesdec);
}

#[test]
fn xv_aesdeclast_xmm_xmm() {
    verify_mnemonic("aesdeclast xmm0, xmm1", IcedMnemonic::Aesdeclast);
}

#[test]
fn xv_aesimc_xmm_xmm() {
    verify_mnemonic("aesimc xmm0, xmm1", IcedMnemonic::Aesimc);
}

#[test]
fn xv_aeskeygenassist_xmm_xmm_imm() {
    verify_mnemonic(
        "aeskeygenassist xmm0, xmm1, 0x01",
        IcedMnemonic::Aeskeygenassist,
    );
}

// ── PCLMULQDQ ───────────────────────────────────────────────────────────

#[test]
fn xv_pclmulqdq_xmm_xmm_imm() {
    verify_mnemonic("pclmulqdq xmm0, xmm1, 0", IcedMnemonic::Pclmulqdq);
}

// ── Movd / Movq (GP ↔ XMM) ─────────────────────────────────────────────

#[test]
fn xv_movd_xmm_r32() {
    verify_mnemonic("movd xmm0, eax", IcedMnemonic::Movd);
}

#[test]
fn xv_movd_r32_xmm() {
    verify_mnemonic("movd eax, xmm0", IcedMnemonic::Movd);
}

#[test]
fn xv_movq_xmm_r64() {
    verify_mnemonic("movq xmm0, rax", IcedMnemonic::Movq);
}

#[test]
fn xv_movq_r64_xmm() {
    verify_mnemonic("movq rax, xmm0", IcedMnemonic::Movq);
}

#[test]
fn xv_movq_xmm_xmm() {
    verify_mnemonic("movq xmm0, xmm1", IcedMnemonic::Movq);
}

// ── Conversion instructions ─────────────────────────────────────────────

#[test]
fn xv_cvtsi2ss_xmm_r32() {
    verify_mnemonic("cvtsi2ss xmm0, eax", IcedMnemonic::Cvtsi2ss);
}

#[test]
fn xv_cvtsi2sd_xmm_r64() {
    verify_mnemonic("cvtsi2sd xmm0, rax", IcedMnemonic::Cvtsi2sd);
}

#[test]
fn xv_cvtss2si_r32_xmm() {
    verify_mnemonic("cvtss2si eax, xmm0", IcedMnemonic::Cvtss2si);
}

#[test]
fn xv_cvtsd2si_r64_xmm() {
    verify_mnemonic("cvtsd2si rax, xmm0", IcedMnemonic::Cvtsd2si);
}

#[test]
fn xv_cvttss2si_r32_xmm() {
    verify_mnemonic("cvttss2si eax, xmm0", IcedMnemonic::Cvttss2si);
}

#[test]
fn xv_cvttsd2si_r64_xmm() {
    verify_mnemonic("cvttsd2si rax, xmm0", IcedMnemonic::Cvttsd2si);
}

// ── Prefetch / Cache ────────────────────────────────────────────────────

#[test]
fn xv_prefetchnta_mem() {
    verify_mnemonic("prefetchnta [rax]", IcedMnemonic::Prefetchnta);
}

#[test]
fn xv_prefetcht0_mem() {
    verify_mnemonic("prefetcht0 [rbx]", IcedMnemonic::Prefetcht0);
}

#[test]
fn xv_prefetcht1_mem() {
    verify_mnemonic("prefetcht1 [rcx]", IcedMnemonic::Prefetcht1);
}

#[test]
fn xv_prefetcht2_mem() {
    verify_mnemonic("prefetcht2 [rdx]", IcedMnemonic::Prefetcht2);
}

#[test]
fn xv_clflush_mem() {
    verify_mnemonic("clflush [rax]", IcedMnemonic::Clflush);
}

#[test]
fn xv_clflushopt_mem() {
    verify_mnemonic("clflushopt [rax]", IcedMnemonic::Clflushopt);
}

// ── Non-temporal stores ─────────────────────────────────────────────────

#[test]
fn xv_movntps_mem_xmm() {
    verify_mnemonic("movntps [rax], xmm0", IcedMnemonic::Movntps);
}

#[test]
fn xv_movntpd_mem_xmm() {
    verify_mnemonic("movntpd [rax], xmm0", IcedMnemonic::Movntpd);
}

#[test]
fn xv_movntdq_mem_xmm() {
    verify_mnemonic("movntdq [rax], xmm0", IcedMnemonic::Movntdq);
}

// ── CRC32 ───────────────────────────────────────────────────────────────

#[test]
fn xv_crc32_r32_r32() {
    verify_mnemonic("crc32 eax, ecx", IcedMnemonic::Crc32);
}

#[test]
fn xv_crc32_r64_r64() {
    verify_mnemonic("crc32 rax, rcx", IcedMnemonic::Crc32);
}

// ── movsd (string op, zero-operand) still works ─────────────────────────

#[test]
fn xv_movsd_string_op() {
    verify_mnemonic("movsd", IcedMnemonic::Movsd);
}

#[test]
fn xv_cmpsd_string_op() {
    verify_mnemonic("cmpsd", IcedMnemonic::Cmpsd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// AVX (VEX-encoded) instructions
// ═══════════════════════════════════════════════════════════════════════════════

// ── VEX zero-operand ────────────────────────────────────────────────────

#[test]
fn xv_vzeroall() {
    verify_mnemonic("vzeroall", IcedMnemonic::Vzeroall);
}

#[test]
fn xv_vzeroupper() {
    verify_mnemonic("vzeroupper", IcedMnemonic::Vzeroupper);
}

// ── VEX arithmetic ──────────────────────────────────────────────────────

#[test]
fn xv_vaddps_xmm() {
    verify_contains("vaddps xmm0, xmm1, xmm2", IcedMnemonic::Vaddps, "xmm0");
}

#[test]
fn xv_vaddps_ymm() {
    verify_contains("vaddps ymm0, ymm1, ymm2", IcedMnemonic::Vaddps, "ymm0");
}

#[test]
fn xv_vaddpd_xmm() {
    verify_contains("vaddpd xmm3, xmm4, xmm5", IcedMnemonic::Vaddpd, "xmm3");
}

#[test]
fn xv_vaddsd_xmm() {
    verify_contains("vaddsd xmm0, xmm1, xmm2", IcedMnemonic::Vaddsd, "xmm0");
}

#[test]
fn xv_vaddss_xmm() {
    verify_contains("vaddss xmm0, xmm1, xmm2", IcedMnemonic::Vaddss, "xmm0");
}

#[test]
fn xv_vsubps_xmm() {
    verify_contains("vsubps xmm0, xmm1, xmm2", IcedMnemonic::Vsubps, "xmm0");
}

#[test]
fn xv_vmulps_ymm() {
    verify_contains("vmulps ymm4, ymm5, ymm6", IcedMnemonic::Vmulps, "ymm4");
}

#[test]
fn xv_vdivpd_xmm() {
    verify_contains("vdivpd xmm0, xmm1, xmm2", IcedMnemonic::Vdivpd, "xmm0");
}

#[test]
fn xv_vsqrtps_xmm() {
    verify_contains("vsqrtps xmm0, xmm1", IcedMnemonic::Vsqrtps, "xmm0");
}

#[test]
fn xv_vminps_ymm() {
    verify_contains("vminps ymm0, ymm1, ymm2", IcedMnemonic::Vminps, "ymm0");
}

#[test]
fn xv_vmaxsd_xmm() {
    verify_contains("vmaxsd xmm0, xmm1, xmm2", IcedMnemonic::Vmaxsd, "xmm0");
}

// ── VEX logical ─────────────────────────────────────────────────────────

#[test]
fn xv_vxorps_xmm_zero() {
    verify_contains("vxorps xmm0, xmm0, xmm0", IcedMnemonic::Vxorps, "xmm0");
}

#[test]
fn xv_vxorps_ymm() {
    verify_contains("vxorps ymm0, ymm0, ymm0", IcedMnemonic::Vxorps, "ymm0");
}

#[test]
fn xv_vxorpd_xmm() {
    verify_contains("vxorpd xmm1, xmm2, xmm3", IcedMnemonic::Vxorpd, "xmm1");
}

#[test]
fn xv_vandps_xmm() {
    verify_contains("vandps xmm0, xmm1, xmm2", IcedMnemonic::Vandps, "xmm0");
}

#[test]
fn xv_vorpd_ymm() {
    verify_contains("vorpd ymm0, ymm1, ymm2", IcedMnemonic::Vorpd, "ymm0");
}

// ── VEX data movement ──────────────────────────────────────────────────

#[test]
fn xv_vmovaps_xmm_xmm() {
    verify_contains("vmovaps xmm0, xmm1", IcedMnemonic::Vmovaps, "xmm0");
}

#[test]
fn xv_vmovaps_ymm_ymm() {
    verify_contains("vmovaps ymm0, ymm1", IcedMnemonic::Vmovaps, "ymm0");
}

#[test]
fn xv_vmovdqa_xmm() {
    verify_contains("vmovdqa xmm0, xmm1", IcedMnemonic::Vmovdqa, "xmm0");
}

#[test]
fn xv_vmovdqu_ymm() {
    verify_contains("vmovdqu ymm0, ymm1", IcedMnemonic::Vmovdqu, "ymm0");
}

#[test]
fn xv_vmovupd_xmm() {
    verify_contains("vmovupd xmm0, xmm1", IcedMnemonic::Vmovupd, "xmm0");
}

// ── VEX integer packed ──────────────────────────────────────────────────

#[test]
fn xv_vpxor_xmm() {
    verify_contains("vpxor xmm0, xmm0, xmm0", IcedMnemonic::Vpxor, "xmm0");
}

#[test]
fn xv_vpand_xmm() {
    verify_contains("vpand xmm0, xmm1, xmm2", IcedMnemonic::Vpand, "xmm0");
}

#[test]
fn xv_vpaddd_xmm() {
    verify_contains("vpaddd xmm0, xmm1, xmm2", IcedMnemonic::Vpaddd, "xmm0");
}

#[test]
fn xv_vpsubb_ymm() {
    verify_contains("vpsubb ymm0, ymm1, ymm2", IcedMnemonic::Vpsubb, "ymm0");
}

#[test]
fn xv_vpcmpeqb_xmm() {
    verify_contains("vpcmpeqb xmm0, xmm1, xmm2", IcedMnemonic::Vpcmpeqb, "xmm0");
}

// ── VEX AES ─────────────────────────────────────────────────────────────

#[test]
fn xv_vaesenc_xmm() {
    verify_contains("vaesenc xmm0, xmm1, xmm2", IcedMnemonic::Vaesenc, "xmm0");
}

#[test]
fn xv_vaesdec_xmm() {
    verify_contains("vaesdec xmm0, xmm1, xmm2", IcedMnemonic::Vaesdec, "xmm0");
}

#[test]
fn xv_vaesenclast_xmm() {
    verify_contains(
        "vaesenclast xmm0, xmm1, xmm2",
        IcedMnemonic::Vaesenclast,
        "xmm0",
    );
}

// ── VEX with immediate ─────────────────────────────────────────────────

#[test]
fn xv_vshufps_imm() {
    verify_contains(
        "vshufps xmm0, xmm1, xmm2, 0x55",
        IcedMnemonic::Vshufps,
        "xmm0",
    );
}

#[test]
fn xv_vcmpps_imm() {
    verify_contains("vcmpps xmm0, xmm1, xmm2, 0", IcedMnemonic::Vcmpps, "xmm0");
}

#[test]
fn xv_vpshufd_imm() {
    verify_contains("vpshufd xmm0, xmm1, 0xFF", IcedMnemonic::Vpshufd, "xmm0");
}

#[test]
fn xv_vpalignr_imm() {
    verify_contains(
        "vpalignr xmm0, xmm1, xmm2, 8",
        IcedMnemonic::Vpalignr,
        "xmm0",
    );
}

#[test]
fn xv_vpclmulqdq_imm() {
    verify_contains(
        "vpclmulqdq xmm0, xmm1, xmm2, 0",
        IcedMnemonic::Vpclmulqdq,
        "xmm0",
    );
}

// ── VEX extended registers ──────────────────────────────────────────────

#[test]
fn xv_vaddps_xmm8() {
    verify_contains("vaddps xmm8, xmm9, xmm10", IcedMnemonic::Vaddps, "xmm8");
}

#[test]
fn xv_vmovaps_xmm15() {
    verify_contains("vmovaps xmm15, xmm14", IcedMnemonic::Vmovaps, "xmm15");
}

#[test]
fn xv_vpxor_ymm10() {
    verify_contains("vpxor ymm10, ymm11, ymm12", IcedMnemonic::Vpxor, "ymm10");
}

// ═══════════════════════════════════════════════════════════════════════════════
// BMI1 / BMI2 instructions
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn xv_andn_r32() {
    verify_contains("andn eax, ebx, ecx", IcedMnemonic::Andn, "eax");
}

#[test]
fn xv_andn_r64() {
    verify_contains("andn rax, rbx, rcx", IcedMnemonic::Andn, "rax");
}

#[test]
fn xv_bextr_r32() {
    verify_contains("bextr eax, ecx, edx", IcedMnemonic::Bextr, "eax");
}

#[test]
fn xv_bextr_r64() {
    verify_contains("bextr rax, rcx, rdx", IcedMnemonic::Bextr, "rax");
}

#[test]
fn xv_blsi_r32() {
    verify_contains("blsi eax, ecx", IcedMnemonic::Blsi, "eax");
}

#[test]
fn xv_blsi_r64() {
    verify_contains("blsi rax, rcx", IcedMnemonic::Blsi, "rax");
}

#[test]
fn xv_blsmsk_r32() {
    verify_contains("blsmsk eax, ecx", IcedMnemonic::Blsmsk, "eax");
}

#[test]
fn xv_blsr_r32() {
    verify_contains("blsr eax, ecx", IcedMnemonic::Blsr, "eax");
}

#[test]
fn xv_bzhi_r32() {
    verify_contains("bzhi eax, ecx, edx", IcedMnemonic::Bzhi, "eax");
}

#[test]
fn xv_bzhi_r64() {
    verify_contains("bzhi rax, rcx, rdx", IcedMnemonic::Bzhi, "rax");
}

#[test]
fn xv_pdep_r32() {
    verify_contains("pdep eax, ebx, ecx", IcedMnemonic::Pdep, "eax");
}

#[test]
fn xv_pdep_r64() {
    verify_contains("pdep rax, rbx, rcx", IcedMnemonic::Pdep, "rax");
}

#[test]
fn xv_pext_r32() {
    verify_contains("pext eax, ebx, ecx", IcedMnemonic::Pext, "eax");
}

#[test]
fn xv_rorx_r32() {
    verify_contains("rorx eax, ecx, 5", IcedMnemonic::Rorx, "eax");
}

#[test]
fn xv_rorx_r64() {
    verify_contains("rorx rax, rcx, 5", IcedMnemonic::Rorx, "rax");
}

#[test]
fn xv_sarx_r32() {
    verify_contains("sarx eax, ecx, edx", IcedMnemonic::Sarx, "eax");
}

#[test]
fn xv_shlx_r32() {
    verify_contains("shlx eax, ecx, edx", IcedMnemonic::Shlx, "eax");
}

#[test]
fn xv_shrx_r64() {
    verify_contains("shrx rax, rcx, rdx", IcedMnemonic::Shrx, "rax");
}

// ── FS/GS base manipulation ────────────────────────────────────────

#[test]
fn xv_rdfsbase_r32() {
    verify_contains("rdfsbase eax", IcedMnemonic::Rdfsbase, "eax");
}

#[test]
fn xv_rdfsbase_r64() {
    verify_contains("rdfsbase rax", IcedMnemonic::Rdfsbase, "rax");
}

#[test]
fn xv_rdgsbase_r32() {
    verify_contains("rdgsbase ecx", IcedMnemonic::Rdgsbase, "ecx");
}

#[test]
fn xv_rdgsbase_r64() {
    verify_contains("rdgsbase r8", IcedMnemonic::Rdgsbase, "r8");
}

#[test]
fn xv_wrfsbase_r32() {
    verify_contains("wrfsbase edx", IcedMnemonic::Wrfsbase, "edx");
}

#[test]
fn xv_wrfsbase_r64() {
    verify_contains("wrfsbase rsi", IcedMnemonic::Wrfsbase, "rsi");
}

#[test]
fn xv_wrgsbase_r32() {
    verify_contains("wrgsbase ebx", IcedMnemonic::Wrgsbase, "ebx");
}

#[test]
fn xv_wrgsbase_r64() {
    verify_contains("wrgsbase r15", IcedMnemonic::Wrgsbase, "r15");
}

// ── Extended state save/restore ─────────────────────────────────────

#[test]
fn xv_fxsave_mem() {
    verify_contains("fxsave [rax]", IcedMnemonic::Fxsave, "rax");
}

#[test]
fn xv_fxrstor_mem() {
    verify_contains("fxrstor [rcx]", IcedMnemonic::Fxrstor, "rcx");
}

#[test]
fn xv_fxsave64_mem() {
    verify_contains("fxsave64 [rax]", IcedMnemonic::Fxsave64, "rax");
}

#[test]
fn xv_fxrstor64_mem() {
    verify_contains("fxrstor64 [rcx]", IcedMnemonic::Fxrstor64, "rcx");
}

#[test]
fn xv_xsave_mem() {
    verify_contains("xsave [rdx]", IcedMnemonic::Xsave, "rdx");
}

#[test]
fn xv_xrstor_mem() {
    verify_contains("xrstor [rbx]", IcedMnemonic::Xrstor, "rbx");
}

#[test]
fn xv_xsave64_mem() {
    verify_contains("xsave64 [rdx]", IcedMnemonic::Xsave64, "rdx");
}

#[test]
fn xv_xrstor64_mem() {
    verify_contains("xrstor64 [rbx]", IcedMnemonic::Xrstor64, "rbx");
}

#[test]
fn xv_xsaveopt_mem() {
    verify_contains("xsaveopt [rsp]", IcedMnemonic::Xsaveopt, "rsp");
}

#[test]
fn xv_xsaveopt64_mem() {
    verify_contains("xsaveopt64 [rsp]", IcedMnemonic::Xsaveopt64, "rsp");
}

#[test]
fn xv_xsavec_mem() {
    verify_contains("xsavec [rax]", IcedMnemonic::Xsavec, "rax");
}

#[test]
fn xv_xsavec64_mem() {
    verify_contains("xsavec64 [rax]", IcedMnemonic::Xsavec64, "rax");
}

#[test]
fn xv_xsaves_mem() {
    verify_contains("xsaves [rdi]", IcedMnemonic::Xsaves, "rdi");
}

#[test]
fn xv_xsaves64_mem() {
    verify_contains("xsaves64 [rdi]", IcedMnemonic::Xsaves64, "rdi");
}

#[test]
fn xv_xrstors_mem() {
    verify_contains("xrstors [rsi]", IcedMnemonic::Xrstors, "rsi");
}

#[test]
fn xv_xrstors64_mem() {
    verify_contains("xrstors64 [rsi]", IcedMnemonic::Xrstors64, "rsi");
}

// ── TSX instructions ───────────────────────────────────────────────

#[test]
fn xv_xend() {
    verify_contains("xend", IcedMnemonic::Xend, "xend");
}

#[test]
fn xv_xtest() {
    verify_contains("xtest", IcedMnemonic::Xtest, "xtest");
}

#[test]
fn xv_xabort() {
    verify_contains("xabort 0x42", IcedMnemonic::Xabort, "42");
}

// ── ADX extension (ADCX / ADOX) ────────────────────────────────────

#[test]
fn xv_adcx_r32() {
    verify_mnemonic("adcx eax, ecx", IcedMnemonic::Adcx);
}

#[test]
fn xv_adcx_r64() {
    verify_contains("adcx rax, rbx", IcedMnemonic::Adcx, "rax");
}

#[test]
fn xv_adcx_mem() {
    verify_contains("adcx eax, [rsi]", IcedMnemonic::Adcx, "rsi");
}

#[test]
fn xv_adcx_r64_ext() {
    verify_contains("adcx r12, r13", IcedMnemonic::Adcx, "r12");
}

#[test]
fn xv_adox_r32() {
    verify_mnemonic("adox eax, ecx", IcedMnemonic::Adox);
}

#[test]
fn xv_adox_r64() {
    verify_contains("adox rax, rbx", IcedMnemonic::Adox, "rax");
}

#[test]
fn xv_adox_mem() {
    verify_contains("adox eax, [rdi]", IcedMnemonic::Adox, "rdi");
}

#[test]
fn xv_adox_r64_ext() {
    verify_contains("adox r14, r15", IcedMnemonic::Adox, "r14");
}

// ── Extended INC / DEC coverage ─────────────────────────────────────

#[test]
fn xv_inc_al() {
    verify_contains("inc al", IcedMnemonic::Inc, "al");
}

#[test]
fn xv_inc_ax() {
    verify_contains("inc ax", IcedMnemonic::Inc, "ax");
}

#[test]
fn xv_inc_rax() {
    verify_contains("inc rax", IcedMnemonic::Inc, "rax");
}

#[test]
fn xv_inc_r8() {
    verify_contains("inc r8", IcedMnemonic::Inc, "r8");
}

#[test]
fn xv_inc_r15() {
    verify_contains("inc r15", IcedMnemonic::Inc, "r15");
}

#[test]
fn xv_dec_al() {
    verify_contains("dec al", IcedMnemonic::Dec, "al");
}

#[test]
fn xv_dec_rax() {
    verify_contains("dec rax", IcedMnemonic::Dec, "rax");
}

#[test]
fn xv_dec_r8() {
    verify_contains("dec r8", IcedMnemonic::Dec, "r8");
}

#[test]
fn xv_dec_r15() {
    verify_contains("dec r15", IcedMnemonic::Dec, "r15");
}

// ── REPNE prefix + string ops variants ──────────────────────────────

#[test]
fn xv_repne_scasb() {
    verify_contains("repne scasb", IcedMnemonic::Scasb, "rdi");
}

#[test]
fn xv_repne_cmpsb() {
    verify_mnemonic("repne cmpsb", IcedMnemonic::Cmpsb);
}

#[test]
fn xv_rep_stosb() {
    verify_mnemonic("rep stosb", IcedMnemonic::Stosb);
}

#[test]
fn xv_stosw() {
    verify_mnemonic("stosw", IcedMnemonic::Stosw);
}

#[test]
fn xv_stosd() {
    verify_mnemonic("stosd", IcedMnemonic::Stosd);
}

#[test]
fn xv_lodsw() {
    verify_mnemonic("lodsw", IcedMnemonic::Lodsw);
}

#[test]
fn xv_lodsd() {
    verify_mnemonic("lodsd", IcedMnemonic::Lodsd);
}

#[test]
fn xv_lodsq() {
    verify_mnemonic("lodsq", IcedMnemonic::Lodsq);
}

#[test]
fn xv_scasw() {
    verify_mnemonic("scasw", IcedMnemonic::Scasw);
}

#[test]
fn xv_scasd() {
    verify_mnemonic("scasd", IcedMnemonic::Scasd);
}

#[test]
fn xv_scasq() {
    verify_mnemonic("scasq", IcedMnemonic::Scasq);
}

#[test]
fn xv_cmpsw() {
    verify_mnemonic("cmpsw", IcedMnemonic::Cmpsw);
}

#[test]
fn xv_cmpsd_string() {
    verify_mnemonic("cmpsd", IcedMnemonic::Cmpsd);
}

#[test]
fn xv_cmpsq() {
    verify_mnemonic("cmpsq", IcedMnemonic::Cmpsq);
}

#[test]
fn xv_movsw() {
    verify_mnemonic("movsw", IcedMnemonic::Movsw);
}

#[test]
fn xv_movsd_string() {
    verify_mnemonic("movsd", IcedMnemonic::Movsd);
}

// ── RDSEED r64 ──────────────────────────────────────────────────────

#[test]
fn xv_rdseed_r64() {
    verify_contains("rdseed rax", IcedMnemonic::Rdseed, "rax");
}

// ── BMI2 MULX ───────────────────────────────────────────────────────

#[test]
fn xv_mulx_r32() {
    verify_contains("mulx eax, ebx, ecx", IcedMnemonic::Mulx, "eax");
}

#[test]
fn xv_mulx_r64() {
    verify_contains("mulx rax, rbx, rcx", IcedMnemonic::Mulx, "rax");
}

// ── Optimizer Cross-Validation (Story 2.6) ──────────────────────────
// Verify that peephole-optimized output is semantically correct by
// decoding with iced-x86 and checking the resulting instruction.

#[test]
fn xv_opt_zero_idiom_mov_eax_0() {
    // mov eax, 0 → xor eax, eax (OptLevel::Size)
    let (mnemonic, formatted) = asm_and_decode_opt("mov eax, 0", asm_rs::OptLevel::Size);
    assert_eq!(mnemonic, IcedMnemonic::Xor, "zero-idiom: {formatted}");
    assert!(
        formatted.to_lowercase().contains("eax"),
        "should use eax: {formatted}"
    );
}

#[test]
fn xv_opt_zero_idiom_mov_rax_0() {
    // mov rax, 0 → xor eax, eax (narrows to 32-bit, zero-extends)
    let (mnemonic, formatted) = asm_and_decode_opt("mov rax, 0", asm_rs::OptLevel::Size);
    assert_eq!(mnemonic, IcedMnemonic::Xor, "zero-idiom 64→32: {formatted}");
    assert!(
        formatted.to_lowercase().contains("eax"),
        "should use eax (narrowed): {formatted}"
    );
}

#[test]
fn xv_opt_zero_idiom_disabled() {
    // With OptLevel::None, mov eax, 0 should stay as MOV
    let (mnemonic, _) = asm_and_decode_opt("mov eax, 0", asm_rs::OptLevel::None);
    assert_eq!(
        mnemonic,
        IcedMnemonic::Mov,
        "should NOT optimize with OptLevel::None"
    );
}

#[test]
fn xv_opt_mov_narrow_r64_to_r32() {
    // mov rax, 0x7f → mov eax, 0x7f (narrowed, zero-extends)
    let (mnemonic, formatted) = asm_and_decode_opt("mov rax, 0x7f", asm_rs::OptLevel::Size);
    assert_eq!(mnemonic, IcedMnemonic::Mov, "still a mov: {formatted}");
    assert!(
        formatted.to_lowercase().contains("eax"),
        "should narrow to eax: {formatted}"
    );
}

#[test]
fn xv_opt_rex_elim_and() {
    // and rax, 0xff → and eax, 0xff (REX elimination)
    let (mnemonic, formatted) = asm_and_decode_opt("and rax, 0xff", asm_rs::OptLevel::Size);
    assert_eq!(mnemonic, IcedMnemonic::And, "still AND: {formatted}");
    assert!(
        formatted.to_lowercase().contains("eax"),
        "should narrow to eax: {formatted}"
    );
}

#[test]
fn xv_opt_rex_elim_and_disabled() {
    // With OptLevel::None, and rax, 0xff should keep RAX
    let (mnemonic, formatted) = asm_and_decode_opt("and rax, 0xff", asm_rs::OptLevel::None);
    assert_eq!(mnemonic, IcedMnemonic::And, "still AND: {formatted}");
    assert!(
        formatted.to_lowercase().contains("rax"),
        "should keep rax: {formatted}"
    );
}

#[test]
fn xv_opt_test_conversion() {
    // and eax, eax → test eax, eax
    let (mnemonic, formatted) = asm_and_decode_opt("and eax, eax", asm_rs::OptLevel::Size);
    assert_eq!(
        mnemonic,
        IcedMnemonic::Test,
        "and→test conversion: {formatted}"
    );
}

#[test]
fn xv_opt_test_conversion_disabled() {
    // With OptLevel::None, and eax, eax stays as AND
    let (mnemonic, _) = asm_and_decode_opt("and eax, eax", asm_rs::OptLevel::None);
    assert_eq!(
        mnemonic,
        IcedMnemonic::And,
        "should NOT convert with OptLevel::None"
    );
}

// ─── Address-size override (0x67) cross-validation ───────────────────────────

#[test]
fn xv_addr32_mov_reg_mem_eax() {
    // mov ecx, [eax] → 0x67 prefix, iced-x86 should decode as using 32-bit addressing
    let (mnemonic, formatted) = asm_and_decode("mov ecx, [eax]");
    assert_eq!(mnemonic, IcedMnemonic::Mov, "should be MOV: {formatted}");
    assert!(
        formatted.to_lowercase().contains("eax"),
        "should reference eax as base: {formatted}"
    );
}

#[test]
fn xv_addr32_mov_mem_ecx_disp() {
    // mov eax, [ecx+0x10] → 0x67 prefix
    let (mnemonic, formatted) = asm_and_decode("mov eax, [ecx+16]");
    assert_eq!(mnemonic, IcedMnemonic::Mov, "should be MOV: {formatted}");
    assert!(
        formatted.to_lowercase().contains("ecx"),
        "should reference ecx as base: {formatted}"
    );
}

#[test]
fn xv_addr32_mov_store() {
    // mov [eax], ecx → 0x67 prefix
    let (mnemonic, formatted) = asm_and_decode("mov [eax], ecx");
    assert_eq!(mnemonic, IcedMnemonic::Mov, "should be MOV: {formatted}");
    assert!(
        formatted.to_lowercase().contains("eax"),
        "should reference eax as base: {formatted}"
    );
}

#[test]
fn xv_addr32_with_sib() {
    // mov eax, [ecx+edx*4] → 0x67 prefix + SIB
    let (mnemonic, formatted) = asm_and_decode("mov eax, [ecx+edx*4]");
    assert_eq!(mnemonic, IcedMnemonic::Mov, "should be MOV: {formatted}");
    assert!(
        formatted.to_lowercase().contains("ecx"),
        "should reference ecx: {formatted}"
    );
    assert!(
        formatted.to_lowercase().contains("edx"),
        "should reference edx: {formatted}"
    );
}

// ─── AT&T / GAS Syntax Cross-Validation ──────────────────────────────────────

/// Assemble one instruction in AT&T syntax, decode with iced-x86.
fn att_asm_and_decode(source: &str) -> (IcedMnemonic, String) {
    let mut asm = asm_rs::Assembler::new(Arch::X86_64);
    asm.syntax(asm_rs::Syntax::Att);
    asm.emit(source).unwrap_or_else(|e| {
        panic!("asm_rs AT&T failed to assemble `{source}`: {e}");
    });
    let result = asm.finish().unwrap();
    let bytes = result.bytes();
    assert!(!bytes.is_empty(), "empty output for AT&T `{source}`");

    let mut decoder = Decoder::with_ip(64, bytes, 0, DecoderOptions::NONE);
    let instr = decoder.decode();
    assert_ne!(
        instr.mnemonic(),
        IcedMnemonic::INVALID,
        "iced-x86 decoded INVALID for AT&T `{source}` → {:02X?}",
        bytes
    );
    assert_eq!(
        instr.len(),
        bytes.len(),
        "iced-x86 decoded {} bytes but asm_rs emitted {} for AT&T `{source}` → {:02X?}",
        instr.len(),
        bytes.len(),
        bytes
    );

    let mut formatter = IntelFormatter::new();
    let mut output = String::new();
    formatter.format(&instr, &mut output);
    (instr.mnemonic(), output)
}

/// Verify AT&T source produces the same bytes as Intel source.
fn att_matches_intel(att_src: &str, intel_src: &str) {
    let mut att_asm = asm_rs::Assembler::new(Arch::X86_64);
    att_asm.syntax(asm_rs::Syntax::Att);
    att_asm.emit(att_src).unwrap_or_else(|e| {
        panic!("asm_rs AT&T failed: `{att_src}`: {e}");
    });
    let att_bytes = att_asm.finish().unwrap().bytes().to_vec();

    let intel_bytes = assemble(intel_src, Arch::X86_64).unwrap_or_else(|e| {
        panic!("asm_rs Intel failed: `{intel_src}`: {e}");
    });

    assert_eq!(
        att_bytes, intel_bytes,
        "AT&T `{att_src}` → {:02X?} != Intel `{intel_src}` → {:02X?}",
        att_bytes, intel_bytes
    );
}

#[test]
fn xv_att_mov_imm_reg() {
    let (mnemonic, _) = att_asm_and_decode("movq $0x12345678, %rax");
    assert_eq!(mnemonic, IcedMnemonic::Mov);
    att_matches_intel("movq $0x12345678, %rax", "mov rax, 0x12345678");
}

#[test]
fn xv_att_mov_reg_reg() {
    let (mnemonic, formatted) = att_asm_and_decode("movq %rax, %rcx");
    assert_eq!(mnemonic, IcedMnemonic::Mov, "{formatted}");
    att_matches_intel("movq %rax, %rcx", "mov rcx, rax");
}

#[test]
fn xv_att_add_imm_reg() {
    let (mnemonic, _) = att_asm_and_decode("addl $0x10, %eax");
    assert_eq!(mnemonic, IcedMnemonic::Add);
    att_matches_intel("addl $0x10, %eax", "add eax, 0x10");
}

#[test]
fn xv_att_sub_imm_reg() {
    let (mnemonic, _) = att_asm_and_decode("subq $8, %rsp");
    assert_eq!(mnemonic, IcedMnemonic::Sub);
    att_matches_intel("subq $8, %rsp", "sub rsp, 8");
}

#[test]
fn xv_att_xor_reg_reg() {
    let (mnemonic, _) = att_asm_and_decode("xorl %eax, %eax");
    assert_eq!(mnemonic, IcedMnemonic::Xor);
    att_matches_intel("xorl %eax, %eax", "xor eax, eax");
}

#[test]
fn xv_att_cmp_imm_reg() {
    let (mnemonic, _) = att_asm_and_decode("cmpl $0, %eax");
    assert_eq!(mnemonic, IcedMnemonic::Cmp);
    att_matches_intel("cmpl $0, %eax", "cmp eax, 0");
}

#[test]
fn xv_att_push_reg() {
    let (mnemonic, _) = att_asm_and_decode("pushq %rbp");
    assert_eq!(mnemonic, IcedMnemonic::Push);
    att_matches_intel("pushq %rbp", "push rbp");
}

#[test]
fn xv_att_pop_reg() {
    let (mnemonic, _) = att_asm_and_decode("popq %rbp");
    assert_eq!(mnemonic, IcedMnemonic::Pop);
    att_matches_intel("popq %rbp", "pop rbp");
}

#[test]
fn xv_att_lea_mem() {
    let (mnemonic, _) = att_asm_and_decode("leaq 8(%rsp), %rax");
    assert_eq!(mnemonic, IcedMnemonic::Lea);
    att_matches_intel("leaq 8(%rsp), %rax", "lea rax, [rsp + 8]");
}

#[test]
fn xv_att_mov_mem_load() {
    let (mnemonic, _) = att_asm_and_decode("movq (%rax), %rbx");
    assert_eq!(mnemonic, IcedMnemonic::Mov);
    att_matches_intel("movq (%rax), %rbx", "mov rbx, [rax]");
}

#[test]
fn xv_att_mov_mem_store() {
    let (mnemonic, _) = att_asm_and_decode("movq %rax, (%rbx)");
    assert_eq!(mnemonic, IcedMnemonic::Mov);
    att_matches_intel("movq %rax, (%rbx)", "mov [rbx], rax");
}

#[test]
fn xv_att_mov_disp_mem() {
    let (mnemonic, _) = att_asm_and_decode("movq -16(%rbp), %rax");
    assert_eq!(mnemonic, IcedMnemonic::Mov);
    att_matches_intel("movq -16(%rbp), %rax", "mov rax, [rbp - 16]");
}

#[test]
fn xv_att_sib_full() {
    let (mnemonic, formatted) = att_asm_and_decode("movl 16(%rbx, %rsi, 8), %eax");
    assert_eq!(mnemonic, IcedMnemonic::Mov, "{formatted}");
    att_matches_intel(
        "movl 16(%rbx, %rsi, 8), %eax",
        "mov eax, [rbx + rsi*8 + 16]",
    );
}

#[test]
fn xv_att_test_reg_reg() {
    let (mnemonic, _) = att_asm_and_decode("testl %edi, %edi");
    assert_eq!(mnemonic, IcedMnemonic::Test);
    att_matches_intel("testl %edi, %edi", "test edi, edi");
}

#[test]
fn xv_att_imul_three_operand() {
    let (mnemonic, _) = att_asm_and_decode("imull $10, %eax, %ecx");
    assert_eq!(mnemonic, IcedMnemonic::Imul);
    att_matches_intel("imull $10, %eax, %ecx", "imul ecx, eax, 10");
}

#[test]
fn xv_att_movzbl() {
    let (mnemonic, _) = att_asm_and_decode("movzbl %al, %eax");
    assert_eq!(mnemonic, IcedMnemonic::Movzx);
    att_matches_intel("movzbl %al, %eax", "movzx eax, al");
}

#[test]
fn xv_att_movsbl() {
    let (mnemonic, _) = att_asm_and_decode("movsbl %al, %eax");
    assert_eq!(mnemonic, IcedMnemonic::Movsx);
    att_matches_intel("movsbl %al, %eax", "movsx eax, al");
}

#[test]
fn xv_att_movslq() {
    let (mnemonic, _) = att_asm_and_decode("movslq %eax, %rax");
    assert_eq!(mnemonic, IcedMnemonic::Movsxd);
    att_matches_intel("movslq %eax, %rax", "movsxd rax, eax");
}

#[test]
fn xv_att_syscall() {
    let (mnemonic, _) = att_asm_and_decode("syscall");
    assert_eq!(mnemonic, IcedMnemonic::Syscall);
}

#[test]
fn xv_att_nop() {
    let (mnemonic, _) = att_asm_and_decode("nop");
    assert_eq!(mnemonic, IcedMnemonic::Nop);
}

#[test]
fn xv_att_ret() {
    let (mnemonic, _) = att_asm_and_decode("ret");
    assert_eq!(mnemonic, IcedMnemonic::Ret);
}

#[test]
fn xv_att_int() {
    let (mnemonic, _) = att_asm_and_decode("int $0x80");
    assert_eq!(mnemonic, IcedMnemonic::Int);
    att_matches_intel("int $0x80", "int 0x80");
}

#[test]
fn xv_att_lock_xchg() {
    att_matches_intel("lock xchgl %eax, (%rbx)", "lock xchg [rbx], eax");
}

#[test]
fn xv_att_rep_movsb() {
    att_matches_intel("rep movsb", "rep movsb");
}

#[test]
fn xv_att_byte_suffix() {
    let (mnemonic, _) = att_asm_and_decode("movb $0x41, %al");
    assert_eq!(mnemonic, IcedMnemonic::Mov);
    att_matches_intel("movb $0x41, %al", "mov al, 0x41");
}

#[test]
fn xv_att_word_suffix() {
    let (mnemonic, _) = att_asm_and_decode("movw $0x1234, %ax");
    assert_eq!(mnemonic, IcedMnemonic::Mov);
    att_matches_intel("movw $0x1234, %ax", "mov ax, 0x1234");
}

#[test]
fn xv_att_push_imm() {
    let (mnemonic, _) = att_asm_and_decode("pushq $42");
    assert_eq!(mnemonic, IcedMnemonic::Push);
    att_matches_intel("pushq $42", "push 42");
}

#[test]
fn xv_att_neg_imm() {
    let (mnemonic, _) = att_asm_and_decode("addq $-1, %rax");
    assert_eq!(mnemonic, IcedMnemonic::Add);
    att_matches_intel("addq $-1, %rax", "add rax, -1");
}

#[test]
fn xv_att_function_prologue() {
    // Verify multi-instruction sequence matches
    let att_src = "pushq %rbp\nmovq %rsp, %rbp\nsubq $16, %rsp";
    let intel_src = "push rbp\nmov rbp, rsp\nsub rsp, 16";

    let mut att_asm = asm_rs::Assembler::new(Arch::X86_64);
    att_asm.syntax(asm_rs::Syntax::Att);
    att_asm.emit(att_src).unwrap();
    let att_bytes = att_asm.finish().unwrap().bytes().to_vec();

    let intel_bytes = assemble(intel_src, Arch::X86_64).unwrap();
    assert_eq!(att_bytes, intel_bytes);
}

// ── FMA3 cross-validation ───────────────────────────────────────────────

#[test]
fn xv_vfmadd132ps_xmm() {
    verify_contains(
        "vfmadd132ps xmm0, xmm1, xmm2",
        IcedMnemonic::Vfmadd132ps,
        "xmm0",
    );
}

#[test]
fn xv_vfmadd213ps_xmm() {
    verify_contains(
        "vfmadd213ps xmm0, xmm1, xmm2",
        IcedMnemonic::Vfmadd213ps,
        "xmm0",
    );
}

#[test]
fn xv_vfmadd231ps_xmm() {
    verify_contains(
        "vfmadd231ps xmm0, xmm1, xmm2",
        IcedMnemonic::Vfmadd231ps,
        "xmm0",
    );
}

#[test]
fn xv_vfmadd231ps_ymm() {
    verify_contains(
        "vfmadd231ps ymm0, ymm1, ymm2",
        IcedMnemonic::Vfmadd231ps,
        "ymm0",
    );
}

#[test]
fn xv_vfmadd231pd_xmm() {
    verify_contains(
        "vfmadd231pd xmm0, xmm1, xmm2",
        IcedMnemonic::Vfmadd231pd,
        "xmm0",
    );
}

#[test]
fn xv_vfmadd231ss_xmm() {
    verify_contains(
        "vfmadd231ss xmm0, xmm1, xmm2",
        IcedMnemonic::Vfmadd231ss,
        "xmm0",
    );
}

#[test]
fn xv_vfmadd231sd_xmm() {
    verify_contains(
        "vfmadd231sd xmm0, xmm1, xmm2",
        IcedMnemonic::Vfmadd231sd,
        "xmm0",
    );
}

#[test]
fn xv_vfmsub231ps_xmm() {
    verify_contains(
        "vfmsub231ps xmm0, xmm1, xmm2",
        IcedMnemonic::Vfmsub231ps,
        "xmm0",
    );
}

#[test]
fn xv_vfmsub231pd_xmm() {
    verify_contains(
        "vfmsub231pd xmm0, xmm1, xmm2",
        IcedMnemonic::Vfmsub231pd,
        "xmm0",
    );
}

#[test]
fn xv_vfnmadd231ps_xmm() {
    verify_contains(
        "vfnmadd231ps xmm0, xmm1, xmm2",
        IcedMnemonic::Vfnmadd231ps,
        "xmm0",
    );
}

#[test]
fn xv_vfnmadd231pd_xmm() {
    verify_contains(
        "vfnmadd231pd xmm0, xmm1, xmm2",
        IcedMnemonic::Vfnmadd231pd,
        "xmm0",
    );
}

#[test]
fn xv_vfnmsub231ps_xmm() {
    verify_contains(
        "vfnmsub231ps xmm0, xmm1, xmm2",
        IcedMnemonic::Vfnmsub231ps,
        "xmm0",
    );
}

#[test]
fn xv_vfnmsub231pd_xmm() {
    verify_contains(
        "vfnmsub231pd xmm0, xmm1, xmm2",
        IcedMnemonic::Vfnmsub231pd,
        "xmm0",
    );
}

#[test]
fn xv_vfmaddsub132ps_xmm() {
    verify_contains(
        "vfmaddsub132ps xmm0, xmm1, xmm2",
        IcedMnemonic::Vfmaddsub132ps,
        "xmm0",
    );
}

#[test]
fn xv_vfmsubadd213ps_xmm() {
    verify_contains(
        "vfmsubadd213ps xmm0, xmm1, xmm2",
        IcedMnemonic::Vfmsubadd213ps,
        "xmm0",
    );
}

// ── VEX shift cross-validation ──────────────────────────────────────────

#[test]
fn xv_vpslld_reg() {
    verify_contains("vpslld xmm0, xmm1, xmm2", IcedMnemonic::Vpslld, "xmm0");
}

#[test]
fn xv_vpslld_imm() {
    verify_contains("vpslld xmm0, xmm1, 4", IcedMnemonic::Vpslld, "xmm0");
}

#[test]
fn xv_vpslld_ymm_imm() {
    verify_contains("vpslld ymm0, ymm1, 4", IcedMnemonic::Vpslld, "ymm0");
}

#[test]
fn xv_vpsllw_reg() {
    verify_contains("vpsllw xmm0, xmm1, xmm2", IcedMnemonic::Vpsllw, "xmm0");
}

#[test]
fn xv_vpsllq_reg() {
    verify_contains("vpsllq xmm0, xmm1, xmm2", IcedMnemonic::Vpsllq, "xmm0");
}

#[test]
fn xv_vpsrlw_imm() {
    verify_contains("vpsrlw xmm2, xmm3, 8", IcedMnemonic::Vpsrlw, "xmm2");
}

#[test]
fn xv_vpsrld_reg() {
    verify_contains("vpsrld xmm0, xmm1, xmm2", IcedMnemonic::Vpsrld, "xmm0");
}

#[test]
fn xv_vpsrlq_reg() {
    verify_contains("vpsrlq xmm3, xmm4, xmm5", IcedMnemonic::Vpsrlq, "xmm3");
}

#[test]
fn xv_vpsraw_imm() {
    verify_contains("vpsraw xmm2, xmm3, 8", IcedMnemonic::Vpsraw, "xmm2");
}

#[test]
fn xv_vpsrad_reg() {
    verify_contains("vpsrad xmm0, xmm1, xmm2", IcedMnemonic::Vpsrad, "xmm0");
}

// ── AVX permute / broadcast cross-validation ────────────────────────────

#[test]
fn xv_vpermilps_reg() {
    verify_contains(
        "vpermilps xmm0, xmm1, xmm2",
        IcedMnemonic::Vpermilps,
        "xmm0",
    );
}

#[test]
fn xv_vpermilps_imm() {
    verify_contains(
        "vpermilps xmm0, xmm1, 0x44",
        IcedMnemonic::Vpermilps,
        "xmm0",
    );
}

#[test]
fn xv_vbroadcastss_xmm() {
    verify_contains(
        "vbroadcastss xmm0, xmm1",
        IcedMnemonic::Vbroadcastss,
        "xmm0",
    );
}

#[test]
fn xv_vpermq_ymm() {
    verify_contains("vpermq ymm0, ymm1, 0x44", IcedMnemonic::Vpermq, "ymm0");
}

#[test]
fn xv_vpsllvd_xmm() {
    verify_contains("vpsllvd xmm0, xmm1, xmm2", IcedMnemonic::Vpsllvd, "xmm0");
}

#[test]
fn xv_vpsrlvd_xmm() {
    verify_contains("vpsrlvd xmm0, xmm1, xmm2", IcedMnemonic::Vpsrlvd, "xmm0");
}

#[test]
fn xv_vpsravd_xmm() {
    verify_contains("vpsravd xmm0, xmm1, xmm2", IcedMnemonic::Vpsravd, "xmm0");
}

#[test]
fn xv_vpbroadcastd_xmm() {
    verify_contains(
        "vpbroadcastd xmm0, xmm1",
        IcedMnemonic::Vpbroadcastd,
        "xmm0",
    );
}

#[test]
fn xv_vtestps_xmm() {
    verify_contains("vtestps xmm0, xmm1", IcedMnemonic::Vtestps, "xmm0");
}

// ── AVX conversion cross-validation ─────────────────────────────────────

#[test]
fn xv_vcvtsi2ss_xmm_eax() {
    verify_contains("vcvtsi2ss xmm0, xmm1, eax", IcedMnemonic::Vcvtsi2ss, "xmm0");
}

#[test]
fn xv_vcvtss2si_eax_xmm() {
    verify_contains("vcvtss2si eax, xmm1", IcedMnemonic::Vcvtss2si, "eax");
}

#[test]
fn xv_vcvtdq2ps_xmm() {
    verify_contains("vcvtdq2ps xmm0, xmm1", IcedMnemonic::Vcvtdq2ps, "xmm0");
}

#[test]
fn xv_vcvtps2dq_xmm() {
    verify_contains("vcvtps2dq xmm0, xmm1", IcedMnemonic::Vcvtps2dq, "xmm0");
}

// ==================  AVX-512 (EVEX) CROSS-VALIDATION  ======================

// ── AVX-512F arithmetic ─────────────────────────────────────────────────

#[test]
fn xv_evex_vaddps_zmm() {
    verify_contains("vaddps zmm0, zmm1, zmm2", IcedMnemonic::Vaddps, "zmm0");
}

#[test]
fn xv_evex_vaddpd_zmm() {
    verify_contains("vaddpd zmm0, zmm1, zmm2", IcedMnemonic::Vaddpd, "zmm0");
}

#[test]
fn xv_evex_vsubps_zmm() {
    verify_contains("vsubps zmm0, zmm1, zmm2", IcedMnemonic::Vsubps, "zmm0");
}

#[test]
fn xv_evex_vsubpd_zmm() {
    verify_contains("vsubpd zmm0, zmm1, zmm2", IcedMnemonic::Vsubpd, "zmm0");
}

#[test]
fn xv_evex_vmulps_zmm() {
    verify_contains("vmulps zmm0, zmm1, zmm2", IcedMnemonic::Vmulps, "zmm0");
}

#[test]
fn xv_evex_vmulpd_zmm() {
    verify_contains("vmulpd zmm0, zmm1, zmm2", IcedMnemonic::Vmulpd, "zmm0");
}

#[test]
fn xv_evex_vdivps_zmm() {
    verify_contains("vdivps zmm0, zmm1, zmm2", IcedMnemonic::Vdivps, "zmm0");
}

#[test]
fn xv_evex_vdivpd_zmm() {
    verify_contains("vdivpd zmm0, zmm1, zmm2", IcedMnemonic::Vdivpd, "zmm0");
}

#[test]
fn xv_evex_vmaxps_zmm() {
    verify_contains("vmaxps zmm0, zmm1, zmm2", IcedMnemonic::Vmaxps, "zmm0");
}

#[test]
fn xv_evex_vminps_zmm() {
    verify_contains("vminps zmm0, zmm1, zmm2", IcedMnemonic::Vminps, "zmm0");
}

#[test]
fn xv_evex_vsqrtps_zmm() {
    verify_contains("vsqrtps zmm0, zmm1", IcedMnemonic::Vsqrtps, "zmm0");
}

#[test]
fn xv_evex_vsqrtpd_zmm() {
    verify_contains("vsqrtpd zmm0, zmm1", IcedMnemonic::Vsqrtpd, "zmm0");
}

// ── AVX-512F logical ────────────────────────────────────────────────────

#[test]
fn xv_evex_vandps_zmm() {
    verify_contains("vandps zmm0, zmm1, zmm2", IcedMnemonic::Vandps, "zmm0");
}

#[test]
fn xv_evex_vandnps_zmm() {
    verify_contains("vandnps zmm0, zmm1, zmm2", IcedMnemonic::Vandnps, "zmm0");
}

#[test]
fn xv_evex_vorps_zmm() {
    verify_contains("vorps zmm0, zmm1, zmm2", IcedMnemonic::Vorps, "zmm0");
}

#[test]
fn xv_evex_vxorps_zmm() {
    verify_contains("vxorps zmm0, zmm1, zmm2", IcedMnemonic::Vxorps, "zmm0");
}

#[test]
fn xv_evex_vandpd_zmm() {
    verify_contains("vandpd zmm0, zmm1, zmm2", IcedMnemonic::Vandpd, "zmm0");
}

#[test]
fn xv_evex_vorpd_zmm() {
    verify_contains("vorpd zmm0, zmm1, zmm2", IcedMnemonic::Vorpd, "zmm0");
}

#[test]
fn xv_evex_vxorpd_zmm() {
    verify_contains("vxorpd zmm0, zmm1, zmm2", IcedMnemonic::Vxorpd, "zmm0");
}

// ── AVX-512F data movement ──────────────────────────────────────────────

#[test]
fn xv_evex_vmovaps_zmm() {
    verify_contains("vmovaps zmm0, zmm1", IcedMnemonic::Vmovaps, "zmm0");
}

#[test]
fn xv_evex_vmovapd_zmm() {
    verify_contains("vmovapd zmm0, zmm1", IcedMnemonic::Vmovapd, "zmm0");
}

#[test]
fn xv_evex_vmovups_zmm() {
    verify_contains("vmovups zmm0, zmm1", IcedMnemonic::Vmovups, "zmm0");
}

#[test]
fn xv_evex_vmovupd_zmm() {
    verify_contains("vmovupd zmm0, zmm1", IcedMnemonic::Vmovupd, "zmm0");
}

#[test]
fn xv_evex_vmovdqa32_zmm() {
    verify_contains("vmovdqa32 zmm0, zmm1", IcedMnemonic::Vmovdqa32, "zmm0");
}

#[test]
fn xv_evex_vmovdqa64_zmm() {
    verify_contains("vmovdqa64 zmm0, zmm1", IcedMnemonic::Vmovdqa64, "zmm0");
}

#[test]
fn xv_evex_vmovdqu32_zmm() {
    verify_contains("vmovdqu32 zmm0, zmm1", IcedMnemonic::Vmovdqu32, "zmm0");
}

#[test]
fn xv_evex_vmovdqu64_zmm() {
    verify_contains("vmovdqu64 zmm0, zmm1", IcedMnemonic::Vmovdqu64, "zmm0");
}

#[test]
fn xv_evex_vmovdqu8_zmm() {
    verify_contains("vmovdqu8 zmm0, zmm1", IcedMnemonic::Vmovdqu8, "zmm0");
}

#[test]
fn xv_evex_vmovdqu16_zmm() {
    verify_contains("vmovdqu16 zmm0, zmm1", IcedMnemonic::Vmovdqu16, "zmm0");
}

// ── AVX-512F integer packed ─────────────────────────────────────────────

#[test]
fn xv_evex_vpaddd_zmm() {
    verify_contains("vpaddd zmm0, zmm1, zmm2", IcedMnemonic::Vpaddd, "zmm0");
}

#[test]
fn xv_evex_vpaddq_zmm() {
    verify_contains("vpaddq zmm0, zmm1, zmm2", IcedMnemonic::Vpaddq, "zmm0");
}

#[test]
fn xv_evex_vpsubd_zmm() {
    verify_contains("vpsubd zmm0, zmm1, zmm2", IcedMnemonic::Vpsubd, "zmm0");
}

#[test]
fn xv_evex_vpsubq_zmm() {
    verify_contains("vpsubq zmm0, zmm1, zmm2", IcedMnemonic::Vpsubq, "zmm0");
}

#[test]
fn xv_evex_vpmulld_zmm() {
    verify_contains("vpmulld zmm0, zmm1, zmm2", IcedMnemonic::Vpmulld, "zmm0");
}

#[test]
fn xv_evex_vpmullq_zmm() {
    verify_contains("vpmullq zmm0, zmm1, zmm2", IcedMnemonic::Vpmullq, "zmm0");
}

#[test]
fn xv_evex_vpxord_zmm() {
    verify_contains("vpxord zmm0, zmm1, zmm2", IcedMnemonic::Vpxord, "zmm0");
}

#[test]
fn xv_evex_vpxorq_zmm() {
    verify_contains("vpxorq zmm0, zmm1, zmm2", IcedMnemonic::Vpxorq, "zmm0");
}

#[test]
fn xv_evex_vpandd_zmm() {
    verify_contains("vpandd zmm0, zmm1, zmm2", IcedMnemonic::Vpandd, "zmm0");
}

#[test]
fn xv_evex_vpandq_zmm() {
    verify_contains("vpandq zmm0, zmm1, zmm2", IcedMnemonic::Vpandq, "zmm0");
}

#[test]
fn xv_evex_vpord_zmm() {
    verify_contains("vpord zmm0, zmm1, zmm2", IcedMnemonic::Vpord, "zmm0");
}

#[test]
fn xv_evex_vporq_zmm() {
    verify_contains("vporq zmm0, zmm1, zmm2", IcedMnemonic::Vporq, "zmm0");
}

// ── AVX-512F ternary/blend ──────────────────────────────────────────────

#[test]
fn xv_evex_vpternlogd_zmm() {
    verify_contains(
        "vpternlogd zmm0, zmm1, zmm2, 0xFF",
        IcedMnemonic::Vpternlogd,
        "zmm0",
    );
}

#[test]
fn xv_evex_vpternlogq_zmm() {
    verify_contains(
        "vpternlogq zmm0, zmm1, zmm2, 0xDB",
        IcedMnemonic::Vpternlogq,
        "zmm0",
    );
}

#[test]
fn xv_evex_vblendmps_zmm() {
    verify_contains(
        "vblendmps zmm0, zmm1, zmm2",
        IcedMnemonic::Vblendmps,
        "zmm0",
    );
}

#[test]
fn xv_evex_vblendmpd_zmm() {
    verify_contains(
        "vblendmpd zmm0, zmm1, zmm2",
        IcedMnemonic::Vblendmpd,
        "zmm0",
    );
}

#[test]
fn xv_evex_vpblendmd_zmm() {
    verify_contains(
        "vpblendmd zmm0, zmm1, zmm2",
        IcedMnemonic::Vpblendmd,
        "zmm0",
    );
}

#[test]
fn xv_evex_vpblendmq_zmm() {
    verify_contains(
        "vpblendmq zmm0, zmm1, zmm2",
        IcedMnemonic::Vpblendmq,
        "zmm0",
    );
}

// ── AVX-512F compress/expand ────────────────────────────────────────────

#[test]
fn xv_evex_vcompressps_zmm() {
    verify_contains("vcompressps zmm0, zmm1", IcedMnemonic::Vcompressps, "zmm0");
}

#[test]
fn xv_evex_vcompresspd_zmm() {
    verify_contains("vcompresspd zmm0, zmm1", IcedMnemonic::Vcompresspd, "zmm0");
}

#[test]
fn xv_evex_vexpandps_zmm() {
    verify_contains("vexpandps zmm0, zmm1", IcedMnemonic::Vexpandps, "zmm0");
}

#[test]
fn xv_evex_vexpandpd_zmm() {
    verify_contains("vexpandpd zmm0, zmm1", IcedMnemonic::Vexpandpd, "zmm0");
}

#[test]
fn xv_evex_vpcompressd_zmm() {
    verify_contains("vpcompressd zmm0, zmm1", IcedMnemonic::Vpcompressd, "zmm0");
}

#[test]
fn xv_evex_vpcompressq_zmm() {
    verify_contains("vpcompressq zmm0, zmm1", IcedMnemonic::Vpcompressq, "zmm0");
}

#[test]
fn xv_evex_vpexpandd_zmm() {
    verify_contains("vpexpandd zmm0, zmm1", IcedMnemonic::Vpexpandd, "zmm0");
}

#[test]
fn xv_evex_vpexpandq_zmm() {
    verify_contains("vpexpandq zmm0, zmm1", IcedMnemonic::Vpexpandq, "zmm0");
}

// ── AVX-512F FMA 512-bit ────────────────────────────────────────────────

#[test]
fn xv_evex_vfmadd132ps_zmm() {
    verify_contains(
        "vfmadd132ps zmm0, zmm1, zmm2",
        IcedMnemonic::Vfmadd132ps,
        "zmm0",
    );
}

#[test]
fn xv_evex_vfmadd213ps_zmm() {
    verify_contains(
        "vfmadd213ps zmm0, zmm1, zmm2",
        IcedMnemonic::Vfmadd213ps,
        "zmm0",
    );
}

#[test]
fn xv_evex_vfmadd231ps_zmm() {
    verify_contains(
        "vfmadd231ps zmm0, zmm1, zmm2",
        IcedMnemonic::Vfmadd231ps,
        "zmm0",
    );
}

#[test]
fn xv_evex_vfmadd132pd_zmm() {
    verify_contains(
        "vfmadd132pd zmm0, zmm1, zmm2",
        IcedMnemonic::Vfmadd132pd,
        "zmm0",
    );
}

#[test]
fn xv_evex_vfmadd213pd_zmm() {
    verify_contains(
        "vfmadd213pd zmm0, zmm1, zmm2",
        IcedMnemonic::Vfmadd213pd,
        "zmm0",
    );
}

#[test]
fn xv_evex_vfmadd231pd_zmm() {
    verify_contains(
        "vfmadd231pd zmm0, zmm1, zmm2",
        IcedMnemonic::Vfmadd231pd,
        "zmm0",
    );
}

// ── AVX-512F unpack ─────────────────────────────────────────────────────

#[test]
fn xv_evex_vunpckhps_zmm() {
    verify_contains(
        "vunpckhps zmm0, zmm1, zmm2",
        IcedMnemonic::Vunpckhps,
        "zmm0",
    );
}

#[test]
fn xv_evex_vunpcklps_zmm() {
    verify_contains(
        "vunpcklps zmm0, zmm1, zmm2",
        IcedMnemonic::Vunpcklps,
        "zmm0",
    );
}

#[test]
fn xv_evex_vunpckhpd_zmm() {
    verify_contains(
        "vunpckhpd zmm0, zmm1, zmm2",
        IcedMnemonic::Vunpckhpd,
        "zmm0",
    );
}

#[test]
fn xv_evex_vunpcklpd_zmm() {
    verify_contains(
        "vunpcklpd zmm0, zmm1, zmm2",
        IcedMnemonic::Vunpcklpd,
        "zmm0",
    );
}

// ── AVX-512F shuffle with imm ───────────────────────────────────────────

#[test]
fn xv_evex_vpshufd_zmm() {
    verify_contains("vpshufd zmm0, zmm1, 0xE4", IcedMnemonic::Vpshufd, "zmm0");
}

#[test]
fn xv_evex_vshufps_zmm() {
    verify_contains(
        "vshufps zmm0, zmm1, zmm2, 0x44",
        IcedMnemonic::Vshufps,
        "zmm0",
    );
}

#[test]
fn xv_evex_vshufpd_zmm() {
    verify_contains(
        "vshufpd zmm0, zmm1, zmm2, 0x55",
        IcedMnemonic::Vshufpd,
        "zmm0",
    );
}

// ── AVX-512F conversions ────────────────────────────────────────────────

#[test]
fn xv_evex_vcvtps2pd_zmm() {
    verify_contains("vcvtps2pd zmm0, zmm1", IcedMnemonic::Vcvtps2pd, "zmm0");
}

#[test]
fn xv_evex_vcvtdq2ps_zmm() {
    verify_contains("vcvtdq2ps zmm0, zmm1", IcedMnemonic::Vcvtdq2ps, "zmm0");
}

#[test]
fn xv_evex_vcvtps2dq_zmm() {
    verify_contains("vcvtps2dq zmm0, zmm1", IcedMnemonic::Vcvtps2dq, "zmm0");
}

// ── AVX-512F extended registers 16-31 ───────────────────────────────────

#[test]
fn xv_evex_vaddps_zmm16() {
    verify_contains("vaddps zmm16, zmm17, zmm18", IcedMnemonic::Vaddps, "zmm16");
}

#[test]
fn xv_evex_vmovaps_zmm31() {
    verify_contains("vmovaps zmm31, zmm16", IcedMnemonic::Vmovaps, "zmm31");
}

#[test]
fn xv_evex_vpaddd_zmm24() {
    verify_contains("vpaddd zmm24, zmm25, zmm26", IcedMnemonic::Vpaddd, "zmm24");
}

// ── AVX-512BW ───────────────────────────────────────────────────────────

#[test]
fn xv_evex_vpaddb_zmm() {
    verify_contains("vpaddb zmm0, zmm1, zmm2", IcedMnemonic::Vpaddb, "zmm0");
}

#[test]
fn xv_evex_vpaddw_zmm() {
    verify_contains("vpaddw zmm0, zmm1, zmm2", IcedMnemonic::Vpaddw, "zmm0");
}

#[test]
fn xv_evex_vpsubb_zmm() {
    verify_contains("vpsubb zmm0, zmm1, zmm2", IcedMnemonic::Vpsubb, "zmm0");
}

#[test]
fn xv_evex_vpsubw_zmm() {
    verify_contains("vpsubw zmm0, zmm1, zmm2", IcedMnemonic::Vpsubw, "zmm0");
}

// ── AVX-512F variable shifts ────────────────────────────────────────────

#[test]
fn xv_evex_vpsravq_zmm() {
    verify_contains("vpsravq zmm0, zmm1, zmm2", IcedMnemonic::Vpsravq, "zmm0");
}

#[test]
fn xv_evex_vpsllvd_zmm() {
    verify_contains("vpsllvd zmm0, zmm1, zmm2", IcedMnemonic::Vpsllvd, "zmm0");
}

#[test]
fn xv_evex_vpsrlvd_zmm() {
    verify_contains("vpsrlvd zmm0, zmm1, zmm2", IcedMnemonic::Vpsrlvd, "zmm0");
}

// ── VEX regression with ZMM cross-validation ────────────────────────────

#[test]
fn xv_evex_regression_vaddps_xmm_still_vex() {
    // XMM should decode as VEX-encoded, not EVEX
    let (mnemonic, formatted) = asm_and_decode("vaddps xmm0, xmm1, xmm2");
    assert_eq!(mnemonic, IcedMnemonic::Vaddps);
    assert!(
        formatted.to_lowercase().contains("xmm0"),
        "should contain xmm0"
    );
    // First byte should be VEX (0xC5 or 0xC4), not EVEX (0x62)
    let bytes = asm_bytes("vaddps xmm0, xmm1, xmm2");
    assert_ne!(bytes[0], 0x62, "XMM ops should NOT use EVEX prefix");
}
