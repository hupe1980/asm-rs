//! x86-32 (i686) cross-validated test file.
//!
//! These tests contain x86-32 instruction encodings cross-validated against
//! llvm-mc -triple=i686 (LLVM 20.1.8).

#[cfg(feature = "x86")]
use asm_rs::{assemble, Arch};

// =============================================================================
// x86-32 (i686) cross-validation against llvm-mc -triple=i686
// =============================================================================
// Reference encodings collected from:
//   llvm-mc -triple=i686 -show-encoding (LLVM 20.1.8)
// All tests use Intel syntax (our native) with byte patterns verified against
// the AT&T-syntax llvm-mc output.

// --- Core: NOP, RET, PUSH, POP ---

/// NOP — encoding: [0x90]
#[test]
fn x86_xval_nop() {
    let code = assemble("nop", Arch::X86).unwrap();
    assert_eq!(code, vec![0x90]);
}

/// RET — encoding: [0xc3]
#[test]
fn x86_xval_ret() {
    let code = assemble("ret", Arch::X86).unwrap();
    assert_eq!(code, vec![0xc3]);
}

/// PUSH EAX — encoding: [0x50]
#[test]
fn x86_xval_push_eax() {
    let code = assemble("push eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x50]);
}

/// PUSH EBX — encoding: [0x53]
#[test]
fn x86_xval_push_ebx() {
    let code = assemble("push ebx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x53]);
}

/// PUSH ECX — encoding: [0x51]
#[test]
fn x86_xval_push_ecx() {
    let code = assemble("push ecx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x51]);
}

/// PUSH EDX — encoding: [0x52]
#[test]
fn x86_xval_push_edx() {
    let code = assemble("push edx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x52]);
}

/// PUSH ESI — encoding: [0x56]
#[test]
fn x86_xval_push_esi() {
    let code = assemble("push esi", Arch::X86).unwrap();
    assert_eq!(code, vec![0x56]);
}

/// PUSH EDI — encoding: [0x57]
#[test]
fn x86_xval_push_edi() {
    let code = assemble("push edi", Arch::X86).unwrap();
    assert_eq!(code, vec![0x57]);
}

/// PUSH EBP — encoding: [0x55]
#[test]
fn x86_xval_push_ebp() {
    let code = assemble("push ebp", Arch::X86).unwrap();
    assert_eq!(code, vec![0x55]);
}

/// PUSH ESP — encoding: [0x54]
#[test]
fn x86_xval_push_esp() {
    let code = assemble("push esp", Arch::X86).unwrap();
    assert_eq!(code, vec![0x54]);
}

/// POP EAX — encoding: [0x58]
#[test]
fn x86_xval_pop_eax() {
    let code = assemble("pop eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x58]);
}

/// POP EBX — encoding: [0x5b]
#[test]
fn x86_xval_pop_ebx() {
    let code = assemble("pop ebx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x5b]);
}

/// POP ECX — encoding: [0x59]
#[test]
fn x86_xval_pop_ecx() {
    let code = assemble("pop ecx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x59]);
}

/// POP EDX — encoding: [0x5a]
#[test]
fn x86_xval_pop_edx() {
    let code = assemble("pop edx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x5a]);
}

/// POP ESI — encoding: [0x5e]
#[test]
fn x86_xval_pop_esi() {
    let code = assemble("pop esi", Arch::X86).unwrap();
    assert_eq!(code, vec![0x5e]);
}

/// POP EDI — encoding: [0x5f]
#[test]
fn x86_xval_pop_edi() {
    let code = assemble("pop edi", Arch::X86).unwrap();
    assert_eq!(code, vec![0x5f]);
}

/// POP EBP — encoding: [0x5d]
#[test]
fn x86_xval_pop_ebp() {
    let code = assemble("pop ebp", Arch::X86).unwrap();
    assert_eq!(code, vec![0x5d]);
}

// --- MOV ---

/// MOV EAX, 42 — encoding: [0xb8,0x2a,0x00,0x00,0x00]
#[test]
fn x86_xval_mov_eax_imm32() {
    let code = assemble("mov eax, 42", Arch::X86).unwrap();
    assert_eq!(code, vec![0xb8, 0x2a, 0x00, 0x00, 0x00]);
}

/// MOV ECX, 0 — our optimizer converts to XOR ECX, ECX (zero idiom): [0x31,0xc9]
/// llvm-mc reference: [0xb9,0x00,0x00,0x00,0x00] (unoptimized MOV r32, imm32)
#[test]
fn x86_xval_mov_ecx_0() {
    let code = assemble("mov ecx, 0", Arch::X86).unwrap();
    assert_eq!(code, vec![0x31, 0xc9]); // optimized: XOR ECX, ECX
}

/// MOV ECX, EAX — encoding: [0x89,0xc1]
#[test]
fn x86_xval_mov_ecx_eax() {
    let code = assemble("mov ecx, eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x89, 0xc1]);
}

/// MOV EDX, EBX — encoding: [0x89,0xda]
#[test]
fn x86_xval_mov_edx_ebx() {
    let code = assemble("mov edx, ebx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x89, 0xda]);
}

/// MOV AL, 0x41 — encoding: [0xb0,0x41]
#[test]
fn x86_xval_mov_al_imm8() {
    let code = assemble("mov al, 0x41", Arch::X86).unwrap();
    assert_eq!(code, vec![0xb0, 0x41]);
}

/// MOV AX, 0x1234 — encoding: [0x66,0xb8,0x34,0x12]
#[test]
fn x86_xval_mov_ax_imm16() {
    let code = assemble("mov ax, 0x1234", Arch::X86).unwrap();
    assert_eq!(code, vec![0x66, 0xb8, 0x34, 0x12]);
}

// --- ALU ---

/// ADD ECX, EAX — encoding: [0x01,0xc1]
#[test]
fn x86_xval_add_ecx_eax() {
    let code = assemble("add ecx, eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x01, 0xc1]);
}

/// ADD EAX, 1 — encoding: [0x83,0xc0,0x01]
#[test]
fn x86_xval_add_eax_imm8() {
    let code = assemble("add eax, 1", Arch::X86).unwrap();
    assert_eq!(code, vec![0x83, 0xc0, 0x01]);
}

/// SUB EAX, EBX — encoding: [0x29,0xd8]
#[test]
fn x86_xval_sub_eax_ebx() {
    let code = assemble("sub eax, ebx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x29, 0xd8]);
}

/// SUB ESP, 16 — encoding: [0x83,0xec,0x10]
#[test]
fn x86_xval_sub_esp_16() {
    let code = assemble("sub esp, 16", Arch::X86).unwrap();
    assert_eq!(code, vec![0x83, 0xec, 0x10]);
}

/// AND EDX, ECX — encoding: [0x21,0xca]
#[test]
fn x86_xval_and_edx_ecx() {
    let code = assemble("and edx, ecx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x21, 0xca]);
}

/// AND EAX, 0xFF — encoding: [0x25,0xff,0x00,0x00,0x00]
#[test]
fn x86_xval_and_eax_0xff() {
    let code = assemble("and eax, 0xff", Arch::X86).unwrap();
    assert_eq!(code, vec![0x25, 0xff, 0x00, 0x00, 0x00]);
}

/// OR EDI, ESI — encoding: [0x09,0xf7]
#[test]
fn x86_xval_or_edi_esi() {
    let code = assemble("or edi, esi", Arch::X86).unwrap();
    assert_eq!(code, vec![0x09, 0xf7]);
}

/// OR EAX, 0x80 — encoding: [0x0d,0x80,0x00,0x00,0x00]
#[test]
fn x86_xval_or_eax_0x80() {
    let code = assemble("or eax, 0x80", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0d, 0x80, 0x00, 0x00, 0x00]);
}

/// XOR EAX, EAX — encoding: [0x31,0xc0]
#[test]
fn x86_xval_xor_eax_eax() {
    let code = assemble("xor eax, eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x31, 0xc0]);
}

/// XOR ECX, 0xFF — encoding: [0x81,0xf1,0xff,0x00,0x00,0x00]
#[test]
fn x86_xval_xor_ecx_0xff() {
    let code = assemble("xor ecx, 0xff", Arch::X86).unwrap();
    assert_eq!(code, vec![0x81, 0xf1, 0xff, 0x00, 0x00, 0x00]);
}

/// CMP ECX, EAX — encoding: [0x39,0xc1]
#[test]
fn x86_xval_cmp_ecx_eax() {
    let code = assemble("cmp ecx, eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x39, 0xc1]);
}

/// CMP EAX, 0 — encoding: [0x83,0xf8,0x00]
#[test]
fn x86_xval_cmp_eax_0() {
    let code = assemble("cmp eax, 0", Arch::X86).unwrap();
    assert_eq!(code, vec![0x83, 0xf8, 0x00]);
}

/// TEST EAX, EAX — encoding: [0x85,0xc0]
#[test]
fn x86_xval_test_eax_eax() {
    let code = assemble("test eax, eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x85, 0xc0]);
}

/// TEST EAX, 0xFF — encoding: [0xa9,0xff,0x00,0x00,0x00]
#[test]
fn x86_xval_test_eax_0xff() {
    let code = assemble("test eax, 0xff", Arch::X86).unwrap();
    assert_eq!(code, vec![0xa9, 0xff, 0x00, 0x00, 0x00]);
}

/// NEG EAX — encoding: [0xf7,0xd8]
#[test]
fn x86_xval_neg_eax() {
    let code = assemble("neg eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0xf7, 0xd8]);
}

/// NOT EAX — encoding: [0xf7,0xd0]
#[test]
fn x86_xval_not_eax() {
    let code = assemble("not eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0xf7, 0xd0]);
}

// --- INC/DEC (x86-32 short forms 0x40-0x4F) ---

/// INC EAX — encoding: [0x40] (32-bit short form)
#[test]
fn x86_xval_inc_eax() {
    let code = assemble("inc eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x40]);
}

/// DEC EAX — encoding: [0x48] (32-bit short form)
#[test]
fn x86_xval_dec_eax() {
    let code = assemble("dec eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x48]);
}

/// INC EBX — encoding: [0x43] (32-bit short form)
#[test]
fn x86_xval_inc_ebx() {
    let code = assemble("inc ebx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x43]);
}

/// DEC ECX — encoding: [0x49] (32-bit short form)
#[test]
fn x86_xval_dec_ecx() {
    let code = assemble("dec ecx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x49]);
}

// --- Shifts ---

/// SHL EAX, 1 — encoding: [0xd1,0xe0]
#[test]
fn x86_xval_shl_eax_1() {
    let code = assemble("shl eax, 1", Arch::X86).unwrap();
    assert_eq!(code, vec![0xd1, 0xe0]);
}

/// SHR EAX, 1 — encoding: [0xd1,0xe8]
#[test]
fn x86_xval_shr_eax_1() {
    let code = assemble("shr eax, 1", Arch::X86).unwrap();
    assert_eq!(code, vec![0xd1, 0xe8]);
}

/// SAR EAX, 1 — encoding: [0xd1,0xf8]
#[test]
fn x86_xval_sar_eax_1() {
    let code = assemble("sar eax, 1", Arch::X86).unwrap();
    assert_eq!(code, vec![0xd1, 0xf8]);
}

/// SHL EAX, CL — encoding: [0xd3,0xe0]
#[test]
fn x86_xval_shl_eax_cl() {
    let code = assemble("shl eax, cl", Arch::X86).unwrap();
    assert_eq!(code, vec![0xd3, 0xe0]);
}

/// SHR EAX, CL — encoding: [0xd3,0xe8]
#[test]
fn x86_xval_shr_eax_cl() {
    let code = assemble("shr eax, cl", Arch::X86).unwrap();
    assert_eq!(code, vec![0xd3, 0xe8]);
}

/// SAR EAX, CL — encoding: [0xd3,0xf8]
#[test]
fn x86_xval_sar_eax_cl() {
    let code = assemble("sar eax, cl", Arch::X86).unwrap();
    assert_eq!(code, vec![0xd3, 0xf8]);
}

// --- IMUL ---

/// IMUL EAX, ECX — encoding: [0x0f,0xaf,0xc1]
#[test]
fn x86_xval_imul_eax_ecx() {
    let code = assemble("imul eax, ecx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xaf, 0xc1]);
}

/// IMUL EAX, ECX, 10 — encoding: [0x6b,0xc1,0x0a]
#[test]
fn x86_xval_imul_eax_ecx_10() {
    let code = assemble("imul eax, ecx, 10", Arch::X86).unwrap();
    assert_eq!(code, vec![0x6b, 0xc1, 0x0a]);
}

// --- MOVZX / MOVSX ---

/// MOVZX EAX, AL — encoding: [0x0f,0xb6,0xc0]
#[test]
fn x86_xval_movzx_eax_al() {
    let code = assemble("movzx eax, al", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xb6, 0xc0]);
}

/// MOVZX EDX, CL — encoding: [0x0f,0xb6,0xd1]
#[test]
fn x86_xval_movzx_edx_cl() {
    let code = assemble("movzx edx, cl", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xb6, 0xd1]);
}

/// MOVZX EAX, AX — encoding: [0x0f,0xb7,0xc0]
#[test]
fn x86_xval_movzx_eax_ax() {
    let code = assemble("movzx eax, ax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xb7, 0xc0]);
}

/// MOVZX EDX, CX — encoding: [0x0f,0xb7,0xd1]
#[test]
fn x86_xval_movzx_edx_cx() {
    let code = assemble("movzx edx, cx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xb7, 0xd1]);
}

/// MOVSX EAX, AL — encoding: [0x0f,0xbe,0xc0]
#[test]
fn x86_xval_movsx_eax_al() {
    let code = assemble("movsx eax, al", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xbe, 0xc0]);
}

/// MOVSX EDX, CL — encoding: [0x0f,0xbe,0xd1]
#[test]
fn x86_xval_movsx_edx_cl() {
    let code = assemble("movsx edx, cl", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xbe, 0xd1]);
}

/// MOVSX EAX, AX — encoding: [0x0f,0xbf,0xc0]
#[test]
fn x86_xval_movsx_eax_ax() {
    let code = assemble("movsx eax, ax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xbf, 0xc0]);
}

/// MOVSX EDX, CX — encoding: [0x0f,0xbf,0xd1]
#[test]
fn x86_xval_movsx_edx_cx() {
    let code = assemble("movsx edx, cx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xbf, 0xd1]);
}

// --- LEA ---

/// LEA EDX, [EAX+ECX*4] — encoding: [0x8d,0x14,0x88]
#[test]
fn x86_xval_lea_sib() {
    let code = assemble("lea edx, [eax+ecx*4]", Arch::X86).unwrap();
    assert_eq!(code, vec![0x8d, 0x14, 0x88]);
}

/// LEA EAX, [EBP+8] — encoding: [0x8d,0x45,0x08]
#[test]
fn x86_xval_lea_ebp_disp8() {
    let code = assemble("lea eax, [ebp+8]", Arch::X86).unwrap();
    assert_eq!(code, vec![0x8d, 0x45, 0x08]);
}

// --- Misc ---

/// CDQ — encoding: [0x99]
#[test]
fn x86_xval_cdq() {
    let code = assemble("cdq", Arch::X86).unwrap();
    assert_eq!(code, vec![0x99]);
}

/// XCHG EAX, EBX — encoding: [0x93]
#[test]
fn x86_xval_xchg_eax_ebx() {
    let code = assemble("xchg eax, ebx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x93]);
}

/// BSWAP EAX — encoding: [0x0f,0xc8]
#[test]
fn x86_xval_bswap_eax() {
    let code = assemble("bswap eax", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xc8]);
}

/// BSWAP ECX — encoding: [0x0f,0xc9]
#[test]
fn x86_xval_bswap_ecx() {
    let code = assemble("bswap ecx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xc9]);
}

/// INT 0x80 — encoding: [0xcd,0x80]
#[test]
fn x86_xval_int_0x80() {
    let code = assemble("int 0x80", Arch::X86).unwrap();
    assert_eq!(code, vec![0xcd, 0x80]);
}

/// INT3 — encoding: [0xcc]
#[test]
fn x86_xval_int3() {
    let code = assemble("int 3", Arch::X86).unwrap();
    assert_eq!(code, vec![0xcc]);
}

// --- CMOVcc ---

/// CMOVNE EAX, ECX — encoding: [0x0f,0x45,0xc1]
#[test]
fn x86_xval_cmovne_eax_ecx() {
    let code = assemble("cmovne eax, ecx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0x45, 0xc1]);
}

/// CMOVE EBX, EDX — encoding: [0x0f,0x44,0xda]
#[test]
fn x86_xval_cmove_ebx_edx() {
    let code = assemble("cmove ebx, edx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0x44, 0xda]);
}

/// CMOVB EDI, ESI — encoding: [0x0f,0x42,0xfe]
#[test]
fn x86_xval_cmovb_edi_esi() {
    let code = assemble("cmovb edi, esi", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0x42, 0xfe]);
}

/// CMOVAE ESI, EDI — encoding: [0x0f,0x43,0xf7]
#[test]
fn x86_xval_cmovae_esi_edi() {
    let code = assemble("cmovae esi, edi", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0x43, 0xf7]);
}

// --- SETcc ---

/// SETE AL — encoding: [0x0f,0x94,0xc0]
#[test]
fn x86_xval_sete_al() {
    let code = assemble("sete al", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0x94, 0xc0]);
}

/// SETNE CL — encoding: [0x0f,0x95,0xc1]
#[test]
fn x86_xval_setne_cl() {
    let code = assemble("setne cl", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0x95, 0xc1]);
}

/// SETL DL — encoding: [0x0f,0x9c,0xc2]
#[test]
fn x86_xval_setl_dl() {
    let code = assemble("setl dl", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0x9c, 0xc2]);
}

/// SETG BL — encoding: [0x0f,0x9f,0xc3]
#[test]
fn x86_xval_setg_bl() {
    let code = assemble("setg bl", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0x9f, 0xc3]);
}

// --- Rotates ---

/// ROL AL, 1 — encoding: [0xd0,0xc0]
#[test]
fn x86_xval_rol_al_1() {
    let code = assemble("rol al, 1", Arch::X86).unwrap();
    assert_eq!(code, vec![0xd0, 0xc0]);
}

/// ROR AL, 1 — encoding: [0xd0,0xc8]
#[test]
fn x86_xval_ror_al_1() {
    let code = assemble("ror al, 1", Arch::X86).unwrap();
    assert_eq!(code, vec![0xd0, 0xc8]);
}

/// ROL EAX, 4 — encoding: [0xc1,0xc0,0x04]
#[test]
fn x86_xval_rol_eax_4() {
    let code = assemble("rol eax, 4", Arch::X86).unwrap();
    assert_eq!(code, vec![0xc1, 0xc0, 0x04]);
}

/// ROR EAX, 4 — encoding: [0xc1,0xc8,0x04]
#[test]
fn x86_xval_ror_eax_4() {
    let code = assemble("ror eax, 4", Arch::X86).unwrap();
    assert_eq!(code, vec![0xc1, 0xc8, 0x04]);
}

// --- Bit test ---

/// BT EAX, 7 — encoding: [0x0f,0xba,0xe0,0x07]
#[test]
fn x86_xval_bt_eax_7() {
    let code = assemble("bt eax, 7", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xba, 0xe0, 0x07]);
}

/// BTS EAX, 0 — encoding: [0x0f,0xba,0xe8,0x00]
#[test]
fn x86_xval_bts_eax_0() {
    let code = assemble("bts eax, 0", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xba, 0xe8, 0x00]);
}

/// BTR EAX, 7 — encoding: [0x0f,0xba,0xf0,0x07]
#[test]
fn x86_xval_btr_eax_7() {
    let code = assemble("btr eax, 7", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xba, 0xf0, 0x07]);
}

// --- BSF / BSR ---

/// BSF EAX, ECX — encoding: [0x0f,0xbc,0xc1]
#[test]
fn x86_xval_bsf_eax_ecx() {
    let code = assemble("bsf eax, ecx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xbc, 0xc1]);
}

/// BSR EBX, EDX — encoding: [0x0f,0xbd,0xda]
#[test]
fn x86_xval_bsr_ebx_edx() {
    let code = assemble("bsr ebx, edx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xbd, 0xda]);
}

// --- Flags ---

/// CLC — encoding: [0xf8]
#[test]
fn x86_xval_clc() {
    let code = assemble("clc", Arch::X86).unwrap();
    assert_eq!(code, vec![0xf8]);
}

/// STC — encoding: [0xf9]
#[test]
fn x86_xval_stc() {
    let code = assemble("stc", Arch::X86).unwrap();
    assert_eq!(code, vec![0xf9]);
}

/// CLD — encoding: [0xfc]
#[test]
fn x86_xval_cld() {
    let code = assemble("cld", Arch::X86).unwrap();
    assert_eq!(code, vec![0xfc]);
}

/// STD — encoding: [0xfd]
#[test]
fn x86_xval_std() {
    let code = assemble("std", Arch::X86).unwrap();
    assert_eq!(code, vec![0xfd]);
}

/// CPUID — encoding: [0x0f,0xa2]
#[test]
fn x86_xval_cpuid() {
    let code = assemble("cpuid", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0xa2]);
}

/// RDTSC — encoding: [0x0f,0x31]
#[test]
fn x86_xval_rdtsc() {
    let code = assemble("rdtsc", Arch::X86).unwrap();
    assert_eq!(code, vec![0x0f, 0x31]);
}

// --- Memory operands ---

/// MOV ECX, [EAX] — encoding: [0x8b,0x08]
#[test]
fn x86_xval_mov_ecx_mem_eax() {
    let code = assemble("mov ecx, [eax]", Arch::X86).unwrap();
    assert_eq!(code, vec![0x8b, 0x08]);
}

/// MOV [EBX], EDX — encoding: [0x89,0x13]
#[test]
fn x86_xval_mov_mem_ebx_edx() {
    let code = assemble("mov [ebx], edx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x89, 0x13]);
}

/// MOV EAX, [ESP+4] — encoding: [0x8b,0x44,0x24,0x04]
#[test]
fn x86_xval_mov_eax_esp_disp8() {
    let code = assemble("mov eax, [esp+4]", Arch::X86).unwrap();
    assert_eq!(code, vec![0x8b, 0x44, 0x24, 0x04]);
}

/// MOV [EBP+8], EBX — encoding: [0x89,0x5d,0x08]
#[test]
fn x86_xval_mov_ebp_disp8_ebx() {
    let code = assemble("mov [ebp+8], ebx", Arch::X86).unwrap();
    assert_eq!(code, vec![0x89, 0x5d, 0x08]);
}

// --- Sign extension ---

/// CBW — encoding: [0x66,0x98]
#[test]
fn x86_xval_cbw() {
    let code = assemble("cbw", Arch::X86).unwrap();
    assert_eq!(code, vec![0x66, 0x98]);
}

/// CWDE — encoding: [0x98]
#[test]
fn x86_xval_cwde() {
    let code = assemble("cwde", Arch::X86).unwrap();
    assert_eq!(code, vec![0x98]);
}

// --- Stack frame ---

/// PUSHF — encoding: [0x9c] (our mnemonic: pushf, not pushfd)
#[test]
fn x86_xval_pushf() {
    let code = assemble("pushf", Arch::X86).unwrap();
    assert_eq!(code, vec![0x9c]);
}

/// POPF — encoding: [0x9d] (our mnemonic: popf, not popfd)
#[test]
fn x86_xval_popf() {
    let code = assemble("popf", Arch::X86).unwrap();
    assert_eq!(code, vec![0x9d]);
}

/// LEAVE — encoding: [0xc9]
#[test]
fn x86_xval_leave() {
    let code = assemble("leave", Arch::X86).unwrap();
    assert_eq!(code, vec![0xc9]);
}

// --- String operations ---

/// STOSB — encoding: [0xaa]
#[test]
fn x86_xval_stosb() {
    let code = assemble("stosb", Arch::X86).unwrap();
    assert_eq!(code, vec![0xaa]);
}

/// STOSW — encoding: [0x66,0xab]
#[test]
fn x86_xval_stosw() {
    let code = assemble("stosw", Arch::X86).unwrap();
    assert_eq!(code, vec![0x66, 0xab]);
}

/// STOSD — encoding: [0xab]
#[test]
fn x86_xval_stosd() {
    let code = assemble("stosd", Arch::X86).unwrap();
    assert_eq!(code, vec![0xab]);
}

/// LODSB — encoding: [0xac]
#[test]
fn x86_xval_lodsb() {
    let code = assemble("lodsb", Arch::X86).unwrap();
    assert_eq!(code, vec![0xac]);
}

/// LODSW — encoding: [0x66,0xad]
#[test]
fn x86_xval_lodsw() {
    let code = assemble("lodsw", Arch::X86).unwrap();
    assert_eq!(code, vec![0x66, 0xad]);
}

/// LODSD — encoding: [0xad]
#[test]
fn x86_xval_lodsd() {
    let code = assemble("lodsd", Arch::X86).unwrap();
    assert_eq!(code, vec![0xad]);
}

/// MOVSB — encoding: [0xa4]
#[test]
fn x86_xval_movsb() {
    let code = assemble("movsb", Arch::X86).unwrap();
    assert_eq!(code, vec![0xa4]);
}

/// MOVSW — encoding: [0x66,0xa5]
#[test]
fn x86_xval_movsw() {
    let code = assemble("movsw", Arch::X86).unwrap();
    assert_eq!(code, vec![0x66, 0xa5]);
}

/// MOVSD — encoding: [0xa5]
#[test]
fn x86_xval_movsd_string() {
    let code = assemble("movsd", Arch::X86).unwrap();
    assert_eq!(code, vec![0xa5]);
}

/// SCASB — encoding: [0xae]
#[test]
fn x86_xval_scasb() {
    let code = assemble("scasb", Arch::X86).unwrap();
    assert_eq!(code, vec![0xae]);
}

/// REP STOSB — encoding: [0xf3,0xaa]
#[test]
fn x86_xval_rep_stosb() {
    let code = assemble("rep stosb", Arch::X86).unwrap();
    assert_eq!(code, vec![0xf3, 0xaa]);
}

/// REP MOVSB — encoding: [0xf3,0xa4]
#[test]
fn x86_xval_rep_movsb() {
    let code = assemble("rep movsb", Arch::X86).unwrap();
    assert_eq!(code, vec![0xf3, 0xa4]);
}
