//! # asm-rs — Pure Rust Multi-Architecture Assembly Engine
//!
//! `asm-rs` is a pure Rust, zero-C-dependency, multi-architecture runtime assembler
//! that turns human-readable assembly text into machine-code bytes.
//!
//! ## Quick Start
//!
//! ```rust
//! use asm_rs::{assemble, Arch};
//!
//! let code = assemble("nop", Arch::X86_64).unwrap();
//! assert_eq!(code, vec![0x90]);
//! ```
//!
//! ## Features
//!
//! - **Pure Rust** — no C/C++ FFI, no LLVM, no system assembler at runtime.
//! - **Multi-arch** — x86, x86-64, ARM, AArch64, RISC-V (feature-gated).
//! - **Runtime text parsing** — assemble from strings at runtime.
//! - **`no_std` + `alloc`** — embeddable in firmware, kernels, WASM.
//! - **Labels & branch relaxation** — automatic forward/backward label resolution.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
// ── Pedantic lint policy ─────────────────────────────────────────────────
// An assembler intentionally performs many narrowing / sign-changing casts
// between integer widths (i128→u8, u8→u32, etc.) and uses dense hex literals
// without separators (0xFFD0, 0x0F38F6).  The lints below are expected and
// acceptable in this context.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::unreadable_literal,
    clippy::match_same_arms,
    clippy::redundant_closure_for_method_calls,
    clippy::bool_to_int_with_if,
    clippy::wildcard_imports,
    clippy::enum_glob_use,
    clippy::needless_raw_string_hashes,
    clippy::semicolon_if_nothing_returned,
    clippy::must_use_candidate,
    clippy::module_name_repetitions,
    clippy::uninlined_format_args,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::fn_params_excessive_bools,
    clippy::too_many_lines,
    clippy::single_match_else,
    clippy::manual_let_else,
    clippy::unnecessary_wraps,
    clippy::unused_self,
    clippy::map_unwrap_or,
    clippy::many_single_char_names,
    clippy::redundant_else,
    clippy::return_self_not_must_use,
    clippy::missing_errors_doc,
    clippy::needless_continue
)]

extern crate alloc;

#[cfg(feature = "aarch64")]
pub(crate) mod aarch64;
#[cfg(feature = "arm")]
pub(crate) mod arm;
/// Public assembler API — builder pattern, one-shot assembly, and `AssemblyResult`.
pub mod assembler;
/// x86-64 instruction encoder (REX, ModR/M, SIB, immediate, relocation).
pub mod encoder;
/// Error types and source-span diagnostics.
pub mod error;
/// Intermediate representation: registers, operands, instructions, directives.
pub mod ir;
/// Zero-copy lexer (tokenizer) with span tracking.
pub mod lexer;
/// Fragment-based linker: label resolution, branch relaxation, patching.
pub mod linker;
/// Peephole optimizations for instruction encoding.
pub mod optimize;
/// Intel-syntax parser producing IR statements.
pub mod parser;
/// Preprocessor: macros, repeat loops, and conditional assembly.
pub mod preprocessor;
#[cfg(feature = "riscv")]
pub(crate) mod riscv;
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) mod x86;

// Re-exports
pub use assembler::{Assembler, AssemblyResult, ResourceLimits};
pub use encoder::RelocKind;
pub use error::{ArchName, AsmError, Span};
pub use ir::{
    AddrMode, AlignDirective, Arch, BroadcastMode, ConstDef, DataDecl, DataSize, DataValue, Expr,
    FillDirective, Instruction, MemoryOperand, Mnemonic, Operand, OperandList, OperandSize,
    OptLevel, OrgDirective, Prefix, PrefixList, Register, SpaceDirective, Statement, SvePredQual,
    Syntax, VectorArrangement, X86Mode,
};
pub use linker::AppliedRelocation;
pub use preprocessor::Preprocessor;

use alloc::vec::Vec;

/// Assemble a string of assembly into machine code bytes.
///
/// Semicolons or newlines separate instructions.
/// Labels are defined with a trailing colon: `loop:`
///
/// # Errors
///
/// Returns [`AsmError`] if the input contains syntax errors, unknown
/// mnemonics, invalid operand combinations, undefined labels, or any
/// other encoding issue.
///
/// # Examples
///
/// ```rust
/// use asm_rs::{assemble, Arch};
///
/// let code = assemble("nop", Arch::X86_64).unwrap();
/// assert_eq!(code, vec![0x90]);
/// ```
pub fn assemble(source: &str, arch: Arch) -> Result<Vec<u8>, AsmError> {
    assemble_at(source, arch, 0)
}

/// Assemble with an explicit base virtual address.
///
/// # Errors
///
/// Returns [`AsmError`] on assembly failure (see [`assemble`] for details).
///
/// # Examples
///
/// ```rust
/// use asm_rs::{assemble_at, Arch};
///
/// let code = assemble_at("nop", Arch::X86_64, 0x1000).unwrap();
/// assert_eq!(code, vec![0x90]);
/// ```
pub fn assemble_at(source: &str, arch: Arch, base_addr: u64) -> Result<Vec<u8>, AsmError> {
    let mut asm = Assembler::new(arch);
    asm.base_address(base_addr);
    asm.emit(source)?;
    let result = asm.finish()?;
    Ok(result.into_bytes())
}

/// Assemble with external labels pre-defined at known addresses.
///
/// # Errors
///
/// Returns [`AsmError`] on assembly failure (see [`assemble`] for details).
///
/// # Examples
///
/// ```rust
/// use asm_rs::{assemble_with, Arch};
///
/// let code = assemble_with("nop", Arch::X86_64, 0x0, &[]).unwrap();
/// assert_eq!(code, vec![0x90]);
/// ```
pub fn assemble_with(
    source: &str,
    arch: Arch,
    base_addr: u64,
    external_labels: &[(&str, u64)],
) -> Result<Vec<u8>, AsmError> {
    let mut asm = Assembler::new(arch);
    asm.base_address(base_addr);
    for &(name, addr) in external_labels {
        asm.define_external(name, addr);
    }
    asm.emit(source)?;
    let result = asm.finish()?;
    Ok(result.into_bytes())
}
