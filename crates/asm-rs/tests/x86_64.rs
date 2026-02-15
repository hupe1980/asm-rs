//! x86-64 cross-validated test file.
//!
//! These tests contain x86-64 core and SSE/SSE2 instruction encodings
//! cross-validated against llvm-mc (x86_64).

#[cfg(feature = "x86_64")]
use asm_rs::{assemble, Arch};

// ============================================================================
// x86-64 Core Instructions — cross-validated against llvm-mc (x86_64)
// ============================================================================

/// NOP — encoding: [0x90]
#[test]
fn x64_xval_nop() {
    let code = assemble("nop", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x90]);
}

/// RET — encoding: [0xc3]
#[test]
fn x64_xval_ret() {
    let code = assemble("ret", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xc3]);
}

/// PUSH RAX — encoding: [0x50]
#[test]
fn x64_xval_push_rax() {
    let code = assemble("push rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x50]);
}

/// PUSH RBX — encoding: [0x53]
#[test]
fn x64_xval_push_rbx() {
    let code = assemble("push rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x53]);
}

/// POP RAX — encoding: [0x58]
#[test]
fn x64_xval_pop_rax() {
    let code = assemble("pop rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x58]);
}

/// POP RBX — encoding: [0x5b]
#[test]
fn x64_xval_pop_rbx() {
    let code = assemble("pop rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x5b]);
}

/// MOV RAX, RBX — encoding: [0x48,0x89,0xd8]
#[test]
fn x64_xval_mov_rax_rbx() {
    let code = assemble("mov rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x89, 0xd8]);
}

/// MOV RAX, 0x12345678 — our assembler optimizes to MOV EAX, imm32 (5 bytes)
/// which zero-extends to RAX, vs llvm-mc's REX.W MOV RAX, imm32 (7 bytes).
/// Both are correct; ours is shorter.
#[test]
fn x64_xval_mov_rax_imm32() {
    let code = assemble("mov rax, 0x12345678", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xb8, 0x78, 0x56, 0x34, 0x12]);
}

/// MOV EAX, EBX — encoding: [0x89,0xd8]
#[test]
fn x64_xval_mov_eax_ebx() {
    let code = assemble("mov eax, ebx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x89, 0xd8]);
}

/// MOV AL, BL — encoding: [0x88,0xd8]
#[test]
fn x64_xval_mov_al_bl() {
    let code = assemble("mov al, bl", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x88, 0xd8]);
}

/// MOV RAX, [RBX] — encoding: [0x48,0x8b,0x03]
#[test]
fn x64_xval_mov_rax_mem_rbx() {
    let code = assemble("mov rax, [rbx]", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x8b, 0x03]);
}

/// MOV [RBX], RAX — encoding: [0x48,0x89,0x03]
#[test]
fn x64_xval_mov_mem_rbx_rax() {
    let code = assemble("mov [rbx], rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x89, 0x03]);
}

/// MOV RAX, [RBX+8] — encoding: [0x48,0x8b,0x43,0x08]
#[test]
fn x64_xval_mov_rax_mem_rbx_disp8() {
    let code = assemble("mov rax, [rbx+8]", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x8b, 0x43, 0x08]);
}

/// MOV RAX, [RBX+RCX*4] — encoding: [0x48,0x8b,0x04,0x8b]
#[test]
fn x64_xval_mov_rax_sib() {
    let code = assemble("mov rax, [rbx+rcx*4]", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x8b, 0x04, 0x8b]);
}

/// MOV RAX, [RBX+RCX*4+16] — encoding: [0x48,0x8b,0x44,0x8b,0x10]
#[test]
fn x64_xval_mov_rax_sib_disp() {
    let code = assemble("mov rax, [rbx+rcx*4+16]", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x8b, 0x44, 0x8b, 0x10]);
}

/// MOV BYTE PTR [RAX], 0x42 — encoding: [0xc6,0x00,0x42]
#[test]
fn x64_xval_mov_byte_ptr_imm() {
    let code = assemble("mov byte ptr [rax], 0x42", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xc6, 0x00, 0x42]);
}

/// MOV WORD PTR [RAX], 0x1234 — encoding: [0x66,0xc7,0x00,0x34,0x12]
#[test]
fn x64_xval_mov_word_ptr_imm() {
    let code = assemble("mov word ptr [rax], 0x1234", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0xc7, 0x00, 0x34, 0x12]);
}

/// MOV DWORD PTR [RAX], 0x12345678 — encoding: [0xc7,0x00,0x78,0x56,0x34,0x12]
#[test]
fn x64_xval_mov_dword_ptr_imm() {
    let code = assemble("mov dword ptr [rax], 0x12345678", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xc7, 0x00, 0x78, 0x56, 0x34, 0x12]);
}

/// MOV QWORD PTR [RAX], 0x12345678 — encoding: [0x48,0xc7,0x00,0x78,0x56,0x34,0x12]
#[test]
fn x64_xval_mov_qword_ptr_imm() {
    let code = assemble("mov qword ptr [rax], 0x12345678", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xc7, 0x00, 0x78, 0x56, 0x34, 0x12]);
}

/// ADD RAX, RBX — encoding: [0x48,0x01,0xd8]
#[test]
fn x64_xval_add_rax_rbx() {
    let code = assemble("add rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x01, 0xd8]);
}

/// ADD RAX, 42 — encoding: [0x48,0x83,0xc0,0x2a]
#[test]
fn x64_xval_add_rax_imm8() {
    let code = assemble("add rax, 42", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x83, 0xc0, 0x2a]);
}

/// SUB RAX, RBX — encoding: [0x48,0x29,0xd8]
#[test]
fn x64_xval_sub_rax_rbx() {
    let code = assemble("sub rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x29, 0xd8]);
}

/// SUB RAX, 42 — encoding: [0x48,0x83,0xe8,0x2a]
#[test]
fn x64_xval_sub_rax_imm8() {
    let code = assemble("sub rax, 42", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x83, 0xe8, 0x2a]);
}

/// AND RAX, RBX — encoding: [0x48,0x21,0xd8]
#[test]
fn x64_xval_and_rax_rbx() {
    let code = assemble("and rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x21, 0xd8]);
}

/// OR RAX, RBX — encoding: [0x48,0x09,0xd8]
#[test]
fn x64_xval_or_rax_rbx() {
    let code = assemble("or rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x09, 0xd8]);
}

/// XOR RAX, RBX — encoding: [0x48,0x31,0xd8]
#[test]
fn x64_xval_xor_rax_rbx() {
    let code = assemble("xor rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x31, 0xd8]);
}

/// CMP RAX, RBX — encoding: [0x48,0x39,0xd8]
#[test]
fn x64_xval_cmp_rax_rbx() {
    let code = assemble("cmp rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x39, 0xd8]);
}

/// TEST RAX, RBX — encoding: [0x48,0x85,0xd8]
#[test]
fn x64_xval_test_rax_rbx() {
    let code = assemble("test rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x85, 0xd8]);
}

/// INC RAX — encoding: [0x48,0xff,0xc0]
#[test]
fn x64_xval_inc_rax() {
    let code = assemble("inc rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xff, 0xc0]);
}

/// DEC RAX — encoding: [0x48,0xff,0xc8]
#[test]
fn x64_xval_dec_rax() {
    let code = assemble("dec rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xff, 0xc8]);
}

/// NEG RAX — encoding: [0x48,0xf7,0xd8]
#[test]
fn x64_xval_neg_rax() {
    let code = assemble("neg rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xf7, 0xd8]);
}

/// NOT RAX — encoding: [0x48,0xf7,0xd0]
#[test]
fn x64_xval_not_rax() {
    let code = assemble("not rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xf7, 0xd0]);
}

/// SHL RAX, 1 — encoding: [0x48,0xd1,0xe0]
#[test]
fn x64_xval_shl_rax_1() {
    let code = assemble("shl rax, 1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xd1, 0xe0]);
}

/// SHR RAX, 1 — encoding: [0x48,0xd1,0xe8]
#[test]
fn x64_xval_shr_rax_1() {
    let code = assemble("shr rax, 1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xd1, 0xe8]);
}

/// SAR RAX, 1 — encoding: [0x48,0xd1,0xf8]
#[test]
fn x64_xval_sar_rax_1() {
    let code = assemble("sar rax, 1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xd1, 0xf8]);
}

/// SHL RAX, CL — encoding: [0x48,0xd3,0xe0]
#[test]
fn x64_xval_shl_rax_cl() {
    let code = assemble("shl rax, cl", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xd3, 0xe0]);
}

/// IMUL RAX, RBX — encoding: [0x48,0x0f,0xaf,0xc3]
#[test]
fn x64_xval_imul_rax_rbx() {
    let code = assemble("imul rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xaf, 0xc3]);
}

/// IMUL RAX, RBX, 42 — encoding: [0x48,0x6b,0xc3,0x2a]
#[test]
fn x64_xval_imul_rax_rbx_42() {
    let code = assemble("imul rax, rbx, 42", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x6b, 0xc3, 0x2a]);
}

/// LEA RAX, [RBX+RCX*4+16] — encoding: [0x48,0x8d,0x44,0x8b,0x10]
#[test]
fn x64_xval_lea_rax_sib_disp() {
    let code = assemble("lea rax, [rbx+rcx*4+16]", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x8d, 0x44, 0x8b, 0x10]);
}

/// CALL RAX — encoding: [0xff,0xd0]
#[test]
fn x64_xval_call_rax() {
    let code = assemble("call rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xff, 0xd0]);
}

/// JMP RAX — encoding: [0xff,0xe0]
#[test]
fn x64_xval_jmp_rax() {
    let code = assemble("jmp rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xff, 0xe0]);
}

/// MOVZX RAX, BL — encoding: [0x48,0x0f,0xb6,0xc3]
#[test]
fn x64_xval_movzx_rax_bl() {
    let code = assemble("movzx rax, bl", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xb6, 0xc3]);
}

/// MOVZX RAX, BX — encoding: [0x48,0x0f,0xb7,0xc3]
#[test]
fn x64_xval_movzx_rax_bx() {
    let code = assemble("movzx rax, bx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xb7, 0xc3]);
}

/// MOVSX RAX, BL — encoding: [0x48,0x0f,0xbe,0xc3]
#[test]
fn x64_xval_movsx_rax_bl() {
    let code = assemble("movsx rax, bl", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xbe, 0xc3]);
}

/// MOVSX RAX, BX — encoding: [0x48,0x0f,0xbf,0xc3]
#[test]
fn x64_xval_movsx_rax_bx() {
    let code = assemble("movsx rax, bx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xbf, 0xc3]);
}

/// MOVSXD RAX, EBX — encoding: [0x48,0x63,0xc3]
#[test]
fn x64_xval_movsxd_rax_ebx() {
    let code = assemble("movsxd rax, ebx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x63, 0xc3]);
}

/// CDQ — encoding: [0x99]
#[test]
fn x64_xval_cdq() {
    let code = assemble("cdq", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x99]);
}

/// CQO — encoding: [0x48,0x99]
#[test]
fn x64_xval_cqo() {
    let code = assemble("cqo", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x99]);
}

/// XCHG RAX, RBX — encoding: [0x48,0x93]
#[test]
fn x64_xval_xchg_rax_rbx() {
    let code = assemble("xchg rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x93]);
}

/// BSWAP RAX — encoding: [0x48,0x0f,0xc8]
#[test]
fn x64_xval_bswap_rax() {
    let code = assemble("bswap rax", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xc8]);
}

/// ROL RAX, 1 — encoding: [0x48,0xd1,0xc0]
#[test]
fn x64_xval_rol_rax_1() {
    let code = assemble("rol rax, 1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xd1, 0xc0]);
}

/// ROR RAX, 1 — encoding: [0x48,0xd1,0xc8]
#[test]
fn x64_xval_ror_rax_1() {
    let code = assemble("ror rax, 1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0xd1, 0xc8]);
}

/// BT RAX, RBX — encoding: [0x48,0x0f,0xa3,0xd8]
#[test]
fn x64_xval_bt_rax_rbx() {
    let code = assemble("bt rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xa3, 0xd8]);
}

/// BTS RAX, RBX — encoding: [0x48,0x0f,0xab,0xd8]
#[test]
fn x64_xval_bts_rax_rbx() {
    let code = assemble("bts rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xab, 0xd8]);
}

/// BTR RAX, RBX — encoding: [0x48,0x0f,0xb3,0xd8]
#[test]
fn x64_xval_btr_rax_rbx() {
    let code = assemble("btr rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xb3, 0xd8]);
}

/// BTC RAX, RBX — encoding: [0x48,0x0f,0xbb,0xd8]
#[test]
fn x64_xval_btc_rax_rbx() {
    let code = assemble("btc rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xbb, 0xd8]);
}

/// BSF RAX, RBX — encoding: [0x48,0x0f,0xbc,0xc3]
#[test]
fn x64_xval_bsf_rax_rbx() {
    let code = assemble("bsf rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xbc, 0xc3]);
}

/// BSR RAX, RBX — encoding: [0x48,0x0f,0xbd,0xc3]
#[test]
fn x64_xval_bsr_rax_rbx() {
    let code = assemble("bsr rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0xbd, 0xc3]);
}

/// POPCNT RAX, RBX — encoding: [0xf3,0x48,0x0f,0xb8,0xc3]
#[test]
fn x64_xval_popcnt_rax_rbx() {
    let code = assemble("popcnt rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xf3, 0x48, 0x0f, 0xb8, 0xc3]);
}

/// LZCNT RAX, RBX — encoding: [0xf3,0x48,0x0f,0xbd,0xc3]
#[test]
fn x64_xval_lzcnt_rax_rbx() {
    let code = assemble("lzcnt rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xf3, 0x48, 0x0f, 0xbd, 0xc3]);
}

/// TZCNT RAX, RBX — encoding: [0xf3,0x48,0x0f,0xbc,0xc3]
#[test]
fn x64_xval_tzcnt_rax_rbx() {
    let code = assemble("tzcnt rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xf3, 0x48, 0x0f, 0xbc, 0xc3]);
}

/// CMOVZ RAX, RBX — encoding: [0x48,0x0f,0x44,0xc3]
#[test]
fn x64_xval_cmovz_rax_rbx() {
    let code = assemble("cmovz rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0x44, 0xc3]);
}

/// CMOVNZ RAX, RBX — encoding: [0x48,0x0f,0x45,0xc3]
#[test]
fn x64_xval_cmovnz_rax_rbx() {
    let code = assemble("cmovnz rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0x45, 0xc3]);
}

/// CMOVS RAX, RBX — encoding: [0x48,0x0f,0x48,0xc3]
#[test]
fn x64_xval_cmovs_rax_rbx() {
    let code = assemble("cmovs rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0x48, 0xc3]);
}

/// CMOVL RAX, RBX — encoding: [0x48,0x0f,0x4c,0xc3]
#[test]
fn x64_xval_cmovl_rax_rbx() {
    let code = assemble("cmovl rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0x4c, 0xc3]);
}

/// CMOVG RAX, RBX — encoding: [0x48,0x0f,0x4f,0xc3]
#[test]
fn x64_xval_cmovg_rax_rbx() {
    let code = assemble("cmovg rax, rbx", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x48, 0x0f, 0x4f, 0xc3]);
}

/// SETZ AL — encoding: [0x0f,0x94,0xc0]
#[test]
fn x64_xval_setz_al() {
    let code = assemble("setz al", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x94, 0xc0]);
}

/// SETNZ AL — encoding: [0x0f,0x95,0xc0]
#[test]
fn x64_xval_setnz_al() {
    let code = assemble("setnz al", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x95, 0xc0]);
}

/// SETS AL — encoding: [0x0f,0x98,0xc0]
#[test]
fn x64_xval_sets_al() {
    let code = assemble("sets al", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x98, 0xc0]);
}

/// SETL AL — encoding: [0x0f,0x9c,0xc0]
#[test]
fn x64_xval_setl_al() {
    let code = assemble("setl al", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x9c, 0xc0]);
}

/// SYSCALL — encoding: [0x0f,0x05]
#[test]
fn x64_xval_syscall() {
    let code = assemble("syscall", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x05]);
}

/// INT 0x80 — encoding: [0xcd,0x80]
#[test]
fn x64_xval_int_80() {
    let code = assemble("int 0x80", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xcd, 0x80]);
}

/// CPUID — encoding: [0x0f,0xa2]
#[test]
fn x64_xval_cpuid() {
    let code = assemble("cpuid", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0xa2]);
}

/// RDTSC — encoding: [0x0f,0x31]
#[test]
fn x64_xval_rdtsc() {
    let code = assemble("rdtsc", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x31]);
}

// ============================================================================
// x86-64 SSE/SSE2 Instructions — cross-validated against llvm-mc (x86_64)
// ============================================================================

/// MOVAPS XMM0, XMM1 — encoding: [0x0f,0x28,0xc1]
#[test]
fn x64_xval_movaps_xmm0_xmm1() {
    let code = assemble("movaps xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x28, 0xc1]);
}

/// MOVUPS XMM0, XMM1 — encoding: [0x0f,0x10,0xc1]
#[test]
fn x64_xval_movups_xmm0_xmm1() {
    let code = assemble("movups xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x10, 0xc1]);
}

/// MOVAPD XMM0, XMM1 — encoding: [0x66,0x0f,0x28,0xc1]
#[test]
fn x64_xval_movapd_xmm0_xmm1() {
    let code = assemble("movapd xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0f, 0x28, 0xc1]);
}

/// MOVUPD XMM0, XMM1 — encoding: [0x66,0x0f,0x10,0xc1]
#[test]
fn x64_xval_movupd_xmm0_xmm1() {
    let code = assemble("movupd xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0f, 0x10, 0xc1]);
}

/// MOVDQA XMM0, XMM1 — encoding: [0x66,0x0f,0x6f,0xc1]
#[test]
fn x64_xval_movdqa_xmm0_xmm1() {
    let code = assemble("movdqa xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0f, 0x6f, 0xc1]);
}

/// MOVDQU XMM0, XMM1 — encoding: [0xf3,0x0f,0x6f,0xc1]
#[test]
fn x64_xval_movdqu_xmm0_xmm1() {
    let code = assemble("movdqu xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xf3, 0x0f, 0x6f, 0xc1]);
}

/// ADDPS XMM0, XMM1 — encoding: [0x0f,0x58,0xc1]
#[test]
fn x64_xval_addps_xmm0_xmm1() {
    let code = assemble("addps xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x58, 0xc1]);
}

/// ADDPD XMM0, XMM1 — encoding: [0x66,0x0f,0x58,0xc1]
#[test]
fn x64_xval_addpd_xmm0_xmm1() {
    let code = assemble("addpd xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0f, 0x58, 0xc1]);
}

/// ADDSS XMM0, XMM1 — encoding: [0xf3,0x0f,0x58,0xc1]
#[test]
fn x64_xval_addss_xmm0_xmm1() {
    let code = assemble("addss xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xf3, 0x0f, 0x58, 0xc1]);
}

/// ADDSD XMM0, XMM1 — encoding: [0xf2,0x0f,0x58,0xc1]
#[test]
fn x64_xval_addsd_xmm0_xmm1() {
    let code = assemble("addsd xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0xf2, 0x0f, 0x58, 0xc1]);
}

/// SUBPS XMM0, XMM1 — encoding: [0x0f,0x5c,0xc1]
#[test]
fn x64_xval_subps_xmm0_xmm1() {
    let code = assemble("subps xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x5c, 0xc1]);
}

/// MULPS XMM0, XMM1 — encoding: [0x0f,0x59,0xc1]
#[test]
fn x64_xval_mulps_xmm0_xmm1() {
    let code = assemble("mulps xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x59, 0xc1]);
}

/// DIVPS XMM0, XMM1 — encoding: [0x0f,0x5e,0xc1]
#[test]
fn x64_xval_divps_xmm0_xmm1() {
    let code = assemble("divps xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x5e, 0xc1]);
}

/// SQRTPS XMM0, XMM1 — encoding: [0x0f,0x51,0xc1]
#[test]
fn x64_xval_sqrtps_xmm0_xmm1() {
    let code = assemble("sqrtps xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x0f, 0x51, 0xc1]);
}

/// PXOR XMM0, XMM1 — encoding: [0x66,0x0f,0xef,0xc1]
#[test]
fn x64_xval_pxor_xmm0_xmm1() {
    let code = assemble("pxor xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0f, 0xef, 0xc1]);
}

/// POR XMM0, XMM1 — encoding: [0x66,0x0f,0xeb,0xc1]
#[test]
fn x64_xval_por_xmm0_xmm1() {
    let code = assemble("por xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0f, 0xeb, 0xc1]);
}

/// PAND XMM0, XMM1 — encoding: [0x66,0x0f,0xdb,0xc1]
#[test]
fn x64_xval_pand_xmm0_xmm1() {
    let code = assemble("pand xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0f, 0xdb, 0xc1]);
}

/// PADDB XMM0, XMM1 — encoding: [0x66,0x0f,0xfc,0xc1]
#[test]
fn x64_xval_paddb_xmm0_xmm1() {
    let code = assemble("paddb xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0f, 0xfc, 0xc1]);
}

/// PADDD XMM0, XMM1 — encoding: [0x66,0x0f,0xfe,0xc1]
#[test]
fn x64_xval_paddd_xmm0_xmm1() {
    let code = assemble("paddd xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0f, 0xfe, 0xc1]);
}

/// PADDQ XMM0, XMM1 — encoding: [0x66,0x0f,0xd4,0xc1]
#[test]
fn x64_xval_paddq_xmm0_xmm1() {
    let code = assemble("paddq xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0f, 0xd4, 0xc1]);
}

/// PSUBD XMM0, XMM1 — encoding: [0x66,0x0f,0xfa,0xc1]
#[test]
fn x64_xval_psubd_xmm0_xmm1() {
    let code = assemble("psubd xmm0, xmm1", Arch::X86_64).unwrap();
    assert_eq!(code, vec![0x66, 0x0f, 0xfa, 0xc1]);
}
