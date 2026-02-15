//! Serde round-trip tests for `asm_rs` IR types.
//!
//! Validates that all public types serialize to JSON and deserialize back
//! to identical values — Story 9.1 acceptance criteria.

#![cfg(feature = "serde")]

use asm_rs::{
    AddrMode, Arch, AsmError, DataDecl, DataSize, DataValue, Expr, Instruction, MemoryOperand,
    Operand, OperandSize, OptLevel, Prefix, Register, Span, Statement, Syntax,
};

/// Helper: serialize to JSON, deserialize back, assert equality.
fn round_trip<T>(val: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + core::fmt::Debug,
{
    let json = serde_json::to_string(val).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(val, &back, "round-trip mismatch for JSON: {json}");
}

// ─── Span ───────────────────────────────────────────────────────────────────

#[test]
fn serde_span() {
    round_trip(&Span::new(1, 5, 10, 3));
    round_trip(&Span::default());
}

// ─── Arch ───────────────────────────────────────────────────────────────────

#[test]
fn serde_arch() {
    for arch in [
        Arch::X86,
        Arch::X86_64,
        Arch::Arm,
        Arch::Thumb,
        Arch::Aarch64,
        Arch::Rv32,
        Arch::Rv64,
    ] {
        round_trip(&arch);
    }
}

// ─── Register ───────────────────────────────────────────────────────────────

#[test]
fn serde_register_samples() {
    let regs = [
        Register::Rax,
        Register::Eax,
        Register::Al,
        Register::Sp,
        Register::Xmm0,
        Register::Ymm15,
        Register::Zmm31,
    ];
    for r in &regs {
        round_trip(r);
    }
}

// ─── Operand ────────────────────────────────────────────────────────────────

#[test]
fn serde_operand_immediate() {
    round_trip(&Operand::Immediate(42));
    round_trip(&Operand::Immediate(-1));
    round_trip(&Operand::Immediate(0x7FFF_FFFF_FFFF_FFFF));
}

#[test]
fn serde_operand_register() {
    round_trip(&Operand::Register(Register::Rax));
    round_trip(&Operand::Register(Register::Rcx));
}

#[test]
fn serde_operand_memory() {
    let mem = Operand::Memory(Box::new(MemoryOperand {
        size: None,
        base: Some(Register::Rbp),
        index: Some(Register::Rcx),
        scale: 8,
        disp: -16,
        segment: None,
        disp_label: None,
        addr_mode: AddrMode::Offset,
        index_subtract: false,
    }));
    round_trip(&mem);
}

#[test]
fn serde_operand_label() {
    round_trip(&Operand::Label("my_label".into()));
}

// ─── Expr ───────────────────────────────────────────────────────────────────

#[test]
fn serde_expr() {
    let expr = Expr::Add(
        Box::new(Expr::Label("count".into())),
        Box::new(Expr::Num(4)),
    );
    round_trip(&expr);
}

// ─── Instruction ────────────────────────────────────────────────────────────

#[test]
fn serde_instruction() {
    let instr = Instruction {
        mnemonic: "mov".into(),
        operands: vec![Operand::Register(Register::Rax), Operand::Immediate(42)].into(),
        size_hint: Some(OperandSize::Qword),
        prefixes: vec![].into(),
        opmask: None,
        zeroing: false,
        broadcast: None,
        span: Span::new(1, 1, 0, 12),
    };
    round_trip(&instr);
}

#[test]
fn serde_instruction_with_prefix() {
    let instr = Instruction {
        mnemonic: "rep movsb".into(),
        operands: vec![].into(),
        size_hint: None,
        prefixes: vec![Prefix::Rep].into(),
        opmask: None,
        zeroing: false,
        broadcast: None,
        span: Span::new(3, 1, 20, 9),
    };
    round_trip(&instr);
}

// ─── Statement ──────────────────────────────────────────────────────────────

#[test]
fn serde_statement_label() {
    let stmt = Statement::Label("start".into(), Span::new(1, 1, 0, 6));
    round_trip(&stmt);
}

#[test]
fn serde_statement_instruction() {
    let stmt = Statement::Instruction(Instruction {
        mnemonic: "nop".into(),
        operands: vec![].into(),
        size_hint: None,
        prefixes: vec![].into(),
        opmask: None,
        zeroing: false,
        broadcast: None,
        span: Span::new(2, 1, 7, 3),
    });
    round_trip(&stmt);
}

#[test]
fn serde_statement_data() {
    let stmt = Statement::Data(DataDecl {
        size: DataSize::Byte,
        values: vec![DataValue::Integer(0x90), DataValue::Integer(0xCC)],
        span: Span::new(5, 1, 30, 10),
    });
    round_trip(&stmt);
}

// ─── AsmError ───────────────────────────────────────────────────────────────

#[test]
fn serde_asm_error() {
    let err = AsmError::Syntax {
        msg: "unexpected token".into(),
        span: Span::new(1, 5, 4, 3),
    };
    round_trip(&err);

    let err2 = AsmError::UndefinedLabel {
        label: "missing".into(),
        span: Span::new(10, 1, 50, 7),
    };
    round_trip(&err2);
}

// ─── AssemblyResult ─────────────────────────────────────────────────────────

#[test]
fn serde_assembly_result() {
    let mut asm = asm_rs::Assembler::new(Arch::X86_64);
    asm.base_address(0x400000);
    asm.emit("start:\nmov rax, 1\nret").unwrap();
    let result = asm.finish().unwrap();

    let json = serde_json::to_string(&result).expect("serialize result");
    let back: asm_rs::AssemblyResult = serde_json::from_str(&json).expect("deserialize result");

    assert_eq!(result.bytes(), back.bytes());
    assert_eq!(result.labels(), back.labels());
    assert_eq!(result.base_address(), back.base_address());
}

// ─── Enum Coverage ──────────────────────────────────────────────────────────

#[test]
fn serde_syntax() {
    round_trip(&Syntax::Intel);
    round_trip(&Syntax::Att);
}

#[test]
fn serde_opt_level() {
    round_trip(&OptLevel::None);
    round_trip(&OptLevel::Size);
}

#[test]
fn serde_data_size() {
    for ds in [
        DataSize::Byte,
        DataSize::Word,
        DataSize::Long,
        DataSize::Quad,
    ] {
        round_trip(&ds);
    }
}

#[test]
fn serde_operand_size() {
    for os in [
        OperandSize::Byte,
        OperandSize::Word,
        OperandSize::Dword,
        OperandSize::Qword,
    ] {
        round_trip(&os);
    }
}

#[test]
fn serde_addr_mode() {
    round_trip(&AddrMode::Offset);
    round_trip(&AddrMode::PreIndex);
    round_trip(&AddrMode::PostIndex);
}

// ─── Complex Nested Structures ──────────────────────────────────────────────

#[test]
fn serde_complex_memory_operand() {
    let mem = MemoryOperand {
        size: Some(OperandSize::Qword),
        base: Some(Register::Rsp),
        index: Some(Register::Rax),
        scale: 4,
        disp: 0x1000,
        segment: Some(Register::Gs),
        disp_label: Some("data_table".into()),
        addr_mode: AddrMode::Offset,
        index_subtract: false,
    };
    round_trip(&mem);
}

#[test]
fn serde_deeply_nested_expr() {
    let expr = Expr::Add(
        Box::new(Expr::Add(
            Box::new(Expr::Label("base".into())),
            Box::new(Expr::Num(8)),
        )),
        Box::new(Expr::Sub(
            Box::new(Expr::Num(100)),
            Box::new(Expr::Label("offset".into())),
        )),
    );
    round_trip(&expr);
}

#[test]
fn serde_full_statement_sequence() {
    let stmts: Vec<Statement> = vec![
        Statement::Label("func".into(), Span::new(1, 1, 0, 5)),
        Statement::Instruction(Instruction {
            mnemonic: "push".into(),
            operands: vec![Operand::Register(Register::Rbp)].into(),
            size_hint: None,
            prefixes: vec![].into(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: Span::new(2, 5, 6, 8),
        }),
        Statement::Instruction(Instruction {
            mnemonic: "mov".into(),
            operands: vec![
                Operand::Register(Register::Rbp),
                Operand::Register(Register::Rsp),
            ]
            .into(),
            size_hint: None,
            prefixes: vec![].into(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: Span::new(3, 5, 15, 12),
        }),
    ];
    round_trip(&stmts);
}
