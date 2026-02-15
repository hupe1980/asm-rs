//! Error types and source span tracking for diagnostics.

#[allow(unused_imports)]
use alloc::format;
use alloc::string::String;
#[allow(unused_imports)]
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

/// Source location for diagnostics.
///
/// Tracks the line, column, byte offset, and length of a token or construct
/// in the original assembly source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Span {
    /// 1-based line number.
    pub line: u32,
    /// 1-based column number (byte offset within line).
    pub col: u32,
    /// 0-based byte offset from start of source.
    pub offset: usize,
    /// Byte length of the spanned region.
    pub len: usize,
}

impl Span {
    /// Create a new span.
    #[must_use]
    pub fn new(line: u32, col: u32, offset: usize, len: usize) -> Self {
        Self {
            line,
            col,
            offset,
            len,
        }
    }

    /// A dummy span for generated/internal constructs.
    #[must_use]
    pub fn dummy() -> Self {
        Self {
            line: 0,
            col: 0,
            offset: 0,
            len: 0,
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// The architecture for which assembly failed — carried in some error variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ArchName {
    /// 32-bit x86.
    X86,
    /// 64-bit x86.
    X86_64,
    /// ARM A32.
    Arm,
    /// ARM Thumb-2.
    Thumb,
    /// ARMv8-A 64-bit.
    Aarch64,
    /// RISC-V 32-bit.
    Rv32,
    /// RISC-V 64-bit.
    Rv64,
}

impl fmt::Display for ArchName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchName::X86 => write!(f, "x86"),
            ArchName::X86_64 => write!(f, "x86_64"),
            ArchName::Arm => write!(f, "ARM"),
            ArchName::Thumb => write!(f, "Thumb"),
            ArchName::Aarch64 => write!(f, "AArch64"),
            ArchName::Rv32 => write!(f, "RV32"),
            ArchName::Rv64 => write!(f, "RV64"),
        }
    }
}

/// Assembly error with source location and descriptive message.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AsmError {
    /// Unknown mnemonic for the target architecture.
    UnknownMnemonic {
        /// The mnemonic that was not recognized.
        mnemonic: String,
        /// The target architecture name.
        arch: ArchName,
        /// Source location of the unknown mnemonic.
        span: Span,
    },

    /// Invalid operand combination for the instruction.
    InvalidOperands {
        /// Description of why the operands are invalid.
        detail: String,
        /// Source location of the instruction.
        span: Span,
    },

    /// Immediate value exceeds the allowed range.
    ImmediateOverflow {
        /// The immediate value that overflowed.
        value: i128,
        /// Minimum allowed value.
        min: i128,
        /// Maximum allowed value.
        max: i128,
        /// Source location of the immediate.
        span: Span,
    },

    /// Referenced label was never defined.
    UndefinedLabel {
        /// The undefined label name.
        label: String,
        /// Source location of the reference.
        span: Span,
    },

    /// Label was defined more than once.
    DuplicateLabel {
        /// The duplicated label name.
        label: String,
        /// Source location of the duplicate definition.
        span: Span,
        /// Source location of the first definition.
        first_span: Span,
    },

    /// Branch target is out of range even after relaxation.
    BranchOutOfRange {
        /// The target label name.
        label: String,
        /// The actual displacement to the target.
        disp: i64,
        /// Maximum allowed displacement.
        max: i64,
        /// Source location of the branch instruction.
        span: Span,
    },

    /// Syntax error during lexing or parsing.
    Syntax {
        /// The syntax error message.
        msg: String,
        /// Source location of the syntax error.
        span: Span,
    },

    /// Branch relaxation did not converge within the allowed number of passes.
    RelaxationLimit {
        /// Maximum number of relaxation passes allowed.
        max: usize,
    },

    /// A configurable resource limit was exceeded (defense against DoS).
    ResourceLimitExceeded {
        /// Human-readable name of the resource (e.g. "statements", "labels").
        resource: String,
        /// The configured limit that was exceeded.
        limit: usize,
    },

    /// Multiple errors collected during assembly.
    Multiple {
        /// The collected assembly errors.
        errors: Vec<AsmError>,
    },
}

impl fmt::Display for AsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsmError::UnknownMnemonic {
                mnemonic,
                arch,
                span,
            } => {
                write!(f, "{}: unknown mnemonic '{}' for {}", span, mnemonic, arch)
            }
            AsmError::InvalidOperands { detail, span } => {
                write!(f, "{}: invalid operand combination: {}", span, detail)
            }
            AsmError::ImmediateOverflow {
                value,
                min,
                max,
                span,
            } => {
                write!(
                    f,
                    "{}: immediate value {} out of range [{}..{}]",
                    span, value, min, max
                )
            }
            AsmError::UndefinedLabel { label, span } => {
                write!(f, "{}: undefined label '{}'", span, label)
            }
            AsmError::DuplicateLabel {
                label,
                span,
                first_span,
            } => {
                write!(
                    f,
                    "{}: duplicate label '{}' (first defined at {})",
                    span, label, first_span
                )
            }
            AsmError::BranchOutOfRange {
                label,
                disp,
                max,
                span,
            } => {
                write!(
                    f,
                    "{}: branch target '{}' out of range (displacement={}, max=±{})",
                    span, label, disp, max
                )
            }
            AsmError::Syntax { msg, span } => {
                write!(f, "{}: {}", span, msg)
            }
            AsmError::RelaxationLimit { max } => {
                write!(
                    f,
                    "assembly exceeded maximum of {} relaxation passes (possible oscillation)",
                    max
                )
            }
            AsmError::ResourceLimitExceeded { resource, limit } => {
                write!(
                    f,
                    "resource limit exceeded: {} (limit: {})",
                    resource, limit
                )
            }
            AsmError::Multiple { errors } => {
                for (i, e) in errors.iter().enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write!(f, "{}", e)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AsmError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_display() {
        let span = Span::new(3, 12, 45, 5);
        assert_eq!(format!("{}", span), "3:12");
    }

    #[test]
    fn span_dummy() {
        let span = Span::dummy();
        assert_eq!(span.line, 0);
        assert_eq!(span.col, 0);
    }

    #[test]
    fn error_unknown_mnemonic_display() {
        let err = AsmError::UnknownMnemonic {
            mnemonic: "foobar".into(),
            arch: ArchName::X86_64,
            span: Span::new(3, 12, 0, 6),
        };
        assert_eq!(
            format!("{}", err),
            "3:12: unknown mnemonic 'foobar' for x86_64"
        );
    }

    #[test]
    fn error_syntax_display() {
        let err = AsmError::Syntax {
            msg: "unexpected token '!'".into(),
            span: Span::new(1, 5, 4, 1),
        };
        assert_eq!(format!("{}", err), "1:5: unexpected token '!'");
    }

    #[test]
    fn error_undefined_label_display() {
        let err = AsmError::UndefinedLabel {
            label: "my_label".into(),
            span: Span::new(10, 1, 100, 8),
        };
        assert_eq!(format!("{}", err), "10:1: undefined label 'my_label'");
    }

    #[test]
    fn error_immediate_overflow_display() {
        let err = AsmError::ImmediateOverflow {
            value: 256,
            min: -128,
            max: 127,
            span: Span::new(5, 10, 50, 3),
        };
        assert_eq!(
            format!("{}", err),
            "5:10: immediate value 256 out of range [-128..127]"
        );
    }

    #[test]
    fn error_duplicate_label_display() {
        let err = AsmError::DuplicateLabel {
            label: "loop".into(),
            span: Span::new(20, 1, 200, 4),
            first_span: Span::new(5, 1, 50, 4),
        };
        assert_eq!(
            format!("{}", err),
            "20:1: duplicate label 'loop' (first defined at 5:1)"
        );
    }

    #[test]
    fn error_relaxation_limit_display() {
        let err = AsmError::RelaxationLimit { max: 20 };
        assert_eq!(
            format!("{}", err),
            "assembly exceeded maximum of 20 relaxation passes (possible oscillation)"
        );
    }

    #[test]
    fn error_branch_out_of_range_display() {
        let err = AsmError::BranchOutOfRange {
            label: "far_away".into(),
            disp: 500000,
            max: 127,
            span: Span::new(1, 1, 0, 10),
        };
        assert_eq!(
            format!("{}", err),
            "1:1: branch target 'far_away' out of range (displacement=500000, max=±127)"
        );
    }

    #[test]
    fn error_multiple_display() {
        let err = AsmError::Multiple {
            errors: vec![
                AsmError::Syntax {
                    msg: "err1".into(),
                    span: Span::new(1, 1, 0, 1),
                },
                AsmError::Syntax {
                    msg: "err2".into(),
                    span: Span::new(2, 1, 5, 1),
                },
            ],
        };
        let s = format!("{}", err);
        assert!(s.contains("err1"));
        assert!(s.contains("err2"));
    }

    #[test]
    fn error_resource_limit_exceeded_display() {
        let err = AsmError::ResourceLimitExceeded {
            resource: "statements".into(),
            limit: 1_000_000,
        };
        assert_eq!(
            format!("{}", err),
            "resource limit exceeded: statements (limit: 1000000)"
        );
    }
}
