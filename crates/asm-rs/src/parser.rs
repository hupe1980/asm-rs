//! Multi-architecture assembly parser.
//!
//! Converts a stream of `Token`s from the lexer into a `Statement` list.
//! Handles instructions, labels, directives, memory operands, size hints, and prefixes.
//! Architecture-aware register parsing resolves naming conflicts (e.g. `r8` is
//! x86-64 R8 vs ARM R8) based on the target `Arch`.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::error::{AsmError, Span};
use crate::ir::*;
use crate::lexer::{Token, TokenKind};

/// Zero-allocation ASCII-lowercase into a caller-provided stack buffer.
/// Returns `&str` of the lowered text. Inputs longer than `buf` are truncated.
#[inline]
fn to_lower_buf<'b>(s: &str, buf: &'b mut [u8]) -> &'b str {
    let len = s.len().min(buf.len());
    buf[..len].copy_from_slice(&s.as_bytes()[..len]);
    buf[..len].make_ascii_lowercase();
    // Input was valid UTF-8 and ASCII lowercase preserves validity,
    // so from_utf8 is infallible here.
    core::str::from_utf8(&buf[..len]).unwrap_or("")
}

/// Parse a token stream into a list of IR statements.
///
/// # Errors
///
/// Returns `Err(AsmError)` if the token stream contains an unexpected token,
/// a malformed directive, or an invalid instruction syntax.
pub fn parse(tokens: &[Token<'_>]) -> Result<Vec<Statement>, AsmError> {
    parse_with_arch(tokens, Arch::X86_64)
}

/// Parse with explicit architecture for register-name disambiguation.
pub fn parse_with_arch(tokens: &[Token<'_>], arch: Arch) -> Result<Vec<Statement>, AsmError> {
    parse_with_syntax(tokens, arch, Syntax::Intel)
}

/// Parse with explicit architecture and syntax dialect.
pub fn parse_with_syntax(
    tokens: &[Token<'_>],
    arch: Arch,
    syntax: Syntax,
) -> Result<Vec<Statement>, AsmError> {
    let mut parser = Parser::new(tokens, arch, syntax);
    parser.parse_program()
}

struct Parser<'a> {
    tokens: &'a [Token<'a>],
    pos: usize,
    /// Target architecture — controls register name resolution.
    arch: Arch,
    /// Syntax dialect — Intel (default) or AT&T/GAS.
    syntax: Syntax,
    /// Constants defined so far (via `.equ` or `NAME = expr`) — available to
    /// `parse_const_expr` so that directive arguments can reference previously
    /// defined constants.
    constants: BTreeMap<String, i128>,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token<'a>], arch: Arch, syntax: Syntax) -> Self {
        Self {
            tokens,
            pos: 0,
            arch,
            syntax,
            constants: BTreeMap::new(),
        }
    }

    #[inline]
    fn peek(&self) -> &Token<'a> {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    #[inline]
    fn advance(&mut self) -> &Token<'a> {
        let tok = &self.tokens[self.pos.min(self.tokens.len() - 1)];
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    #[inline]
    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len() || self.peek().kind == TokenKind::Eof
    }

    fn expect_ident(&mut self) -> Result<(String, Span), AsmError> {
        let tok = self.advance();
        if tok.kind == TokenKind::Ident {
            Ok((tok.text.to_string(), tok.span))
        } else {
            Err(AsmError::Syntax {
                msg: alloc::format!("expected identifier, found '{}'", tok.text),
                span: tok.span,
            })
        }
    }

    #[inline]
    fn skip_newlines(&mut self) {
        while !self.at_end() && self.peek().kind == TokenKind::Newline {
            self.advance();
        }
    }

    fn parse_program(&mut self) -> Result<Vec<Statement>, AsmError> {
        // Heuristic: ~3 tokens per statement on average.
        let mut stmts = Vec::with_capacity(self.tokens.len() / 3 + 1);
        self.skip_newlines();
        while !self.at_end() {
            if let Some(stmt) = self.parse_statement()? {
                stmts.push(stmt);
            }
            self.skip_newlines();
        }
        Ok(stmts)
    }

    fn parse_statement(&mut self) -> Result<Option<Statement>, AsmError> {
        let tok = self.peek().clone();

        match &tok.kind {
            TokenKind::Eof => Ok(None),
            TokenKind::Newline => {
                self.advance();
                Ok(None)
            }

            // Label definition
            TokenKind::LabelDef => {
                self.advance();
                Ok(Some(Statement::Label(tok.text.to_string(), tok.span)))
            }

            // Numeric label definition
            TokenKind::NumericLabelDef(n) => {
                self.advance();
                let name = alloc::format!("{}", n);
                Ok(Some(Statement::Label(name, tok.span)))
            }

            // Directive
            TokenKind::Directive => self.parse_directive(),

            // Instruction or prefix
            TokenKind::Ident => self.parse_instruction_or_prefix(),

            _ => Err(AsmError::Syntax {
                msg: alloc::format!("unexpected token '{}'", tok.text),
                span: tok.span,
            }),
        }
    }

    fn parse_directive(&mut self) -> Result<Option<Statement>, AsmError> {
        let tok = self.advance().clone();
        let mut dir_buf = [0u8; 32];
        let dir = to_lower_buf(&tok.text, &mut dir_buf);
        let span = tok.span;

        match dir {
            // Data directives
            ".byte" | ".db" => self.parse_data_directive(DataSize::Byte, span),
            ".word" | ".dw" | ".short" => self.parse_data_directive(DataSize::Word, span),
            ".long" | ".dd" | ".int" => self.parse_data_directive(DataSize::Long, span),
            ".quad" | ".dq" => self.parse_data_directive(DataSize::Quad, span),

            // String directives
            ".ascii" => self.parse_string_directive(false, span),
            ".asciz" | ".string" => self.parse_string_directive(true, span),

            // Constant
            ".equ" | ".set" => self.parse_equ_directive(span),

            // Alignment
            ".align" | ".balign" | ".p2align" => {
                let is_p2 = dir == ".p2align";
                self.parse_align_directive(is_p2, span)
            }

            // Fill
            ".fill" => self.parse_fill_directive(span),

            // Space/skip
            ".space" | ".skip" => self.parse_space_directive(span),

            // Org
            ".org" => self.parse_org_directive(span),

            // Global/extern (accepted but ignored for pure code generation)
            ".global" | ".globl" | ".extern" => {
                // Consume the symbol name
                if !self.at_end() && self.peek().kind == TokenKind::Ident {
                    self.advance();
                }
                Ok(None)
            }

            // Section directives (accepted but ignored)
            ".text" | ".data" | ".bss" | ".rodata" | ".section" => {
                // Skip to end of line
                while !self.at_end()
                    && self.peek().kind != TokenKind::Newline
                    && self.peek().kind != TokenKind::Eof
                {
                    self.advance();
                }
                Ok(None)
            }

            // Code mode switching
            ".code16" => Ok(Some(Statement::CodeMode(crate::ir::X86Mode::Mode16, span))),
            ".code32" => Ok(Some(Statement::CodeMode(crate::ir::X86Mode::Mode32, span))),
            ".code64" => Ok(Some(Statement::CodeMode(crate::ir::X86Mode::Mode64, span))),

            // Literal pool flush
            ".ltorg" | ".pool" => Ok(Some(Statement::Ltorg(span))),

            // ARM/Thumb mode switching
            ".thumb" => Ok(Some(Statement::ThumbMode(true, span))),
            ".arm" => Ok(Some(Statement::ThumbMode(false, span))),
            ".thumb_func" => Ok(Some(Statement::ThumbFunc(span))),

            // Syntax switching: .syntax att / .syntax intel
            ".syntax" => {
                let next = self.peek().clone();
                if next.kind == TokenKind::Ident {
                    self.advance();
                    if next.text.eq_ignore_ascii_case("att") {
                        self.syntax = Syntax::Att;
                        Ok(None)
                    } else if next.text.eq_ignore_ascii_case("intel") {
                        self.syntax = Syntax::Intel;
                        Ok(None)
                    } else {
                        Err(AsmError::Syntax {
                            msg: alloc::format!(
                                "unknown syntax '{}' (expected 'att' or 'intel')",
                                next.text
                            ),
                            span: next.span,
                        })
                    }
                } else {
                    Err(AsmError::Syntax {
                        msg: String::from("expected 'att' or 'intel' after .syntax"),
                        span: next.span,
                    })
                }
            }

            // RISC-V options: .option rvc / .option norvc
            ".option" => {
                let next = self.peek().clone();
                if next.kind == TokenKind::Ident {
                    self.advance();
                    if next.text.eq_ignore_ascii_case("rvc") {
                        Ok(Some(Statement::OptionRvc(true, span)))
                    } else if next.text.eq_ignore_ascii_case("norvc") {
                        Ok(Some(Statement::OptionRvc(false, span)))
                    } else {
                        Err(AsmError::Syntax {
                            msg: alloc::format!(
                                "unknown option '{}' (expected 'rvc' or 'norvc')",
                                next.text
                            ),
                            span: next.span,
                        })
                    }
                } else {
                    Err(AsmError::Syntax {
                        msg: String::from("expected 'rvc' or 'norvc' after .option"),
                        span: next.span,
                    })
                }
            }

            _ => Err(AsmError::Syntax {
                msg: alloc::format!("unknown directive '{}'", dir),
                span,
            }),
        }
    }

    fn parse_data_directive(
        &mut self,
        size: DataSize,
        span: Span,
    ) -> Result<Option<Statement>, AsmError> {
        let mut values = Vec::new();
        loop {
            let val = self.parse_data_value()?;
            values.push(val);
            if self.peek().kind == TokenKind::Comma {
                self.advance();
            } else {
                break;
            }
        }
        Ok(Some(Statement::Data(DataDecl { size, values, span })))
    }

    fn parse_data_value(&mut self) -> Result<DataValue, AsmError> {
        let tok = self.peek().clone();
        match &tok.kind {
            TokenKind::Number(n) => {
                self.advance();
                Ok(DataValue::Integer(*n))
            }
            TokenKind::CharLit(ch) => {
                self.advance();
                Ok(DataValue::Integer(*ch as i128))
            }
            TokenKind::Ident => {
                self.advance();
                let label = tok.text.to_string();
                // Parse optional addend: label + N or label - N
                let addend = if self.peek().kind == TokenKind::Plus {
                    self.advance();
                    let n = self.parse_const_expr()?;
                    n as i64
                } else if self.peek().kind == TokenKind::Minus {
                    self.advance();
                    let n = self.parse_const_expr()?;
                    -(n as i64)
                } else {
                    0
                };
                Ok(DataValue::Label(label, addend))
            }
            TokenKind::Minus => {
                self.advance();
                let next = self.peek().clone();
                if let TokenKind::Number(n) = next.kind {
                    self.advance();
                    Ok(DataValue::Integer(-n))
                } else {
                    Err(AsmError::Syntax {
                        msg: String::from("expected number after '-'"),
                        span: tok.span,
                    })
                }
            }
            _ => Err(AsmError::Syntax {
                msg: alloc::format!("expected data value, found '{}'", tok.text),
                span: tok.span,
            }),
        }
    }

    fn parse_string_directive(
        &mut self,
        null_terminate: bool,
        span: Span,
    ) -> Result<Option<Statement>, AsmError> {
        let tok = self.advance().clone();
        if tok.kind != TokenKind::StringLit {
            return Err(AsmError::Syntax {
                msg: String::from("expected string literal"),
                span: tok.span,
            });
        }
        let mut bytes: Vec<u8> = tok.text.as_bytes().to_vec();
        if null_terminate {
            bytes.push(0);
        }
        Ok(Some(Statement::Data(DataDecl {
            size: DataSize::Byte,
            values: vec![DataValue::Bytes(bytes)],
            span,
        })))
    }

    fn parse_equ_directive(&mut self, span: Span) -> Result<Option<Statement>, AsmError> {
        let (name, _) = self.expect_ident()?;
        // Expect comma
        if self.peek().kind == TokenKind::Comma {
            self.advance();
        }
        let value = self.parse_const_expr()?;
        self.constants.insert(name.clone(), value);
        Ok(Some(Statement::Const(ConstDef { name, value, span })))
    }

    fn parse_align_directive(
        &mut self,
        is_p2: bool,
        span: Span,
    ) -> Result<Option<Statement>, AsmError> {
        let raw = self.parse_const_expr()? as u32;
        let alignment = if is_p2 { 1u32 << raw } else { raw };

        // Validate alignment is a power of two (0 and 1 are no-ops)
        if alignment > 1 && !alignment.is_power_of_two() {
            return Err(AsmError::Syntax {
                msg: alloc::format!("alignment must be a power of 2, got {alignment}"),
                span,
            });
        }

        let fill = if self.peek().kind == TokenKind::Comma {
            self.advance();
            // Try to parse a constant expression for the fill byte
            if matches!(
                self.peek().kind,
                TokenKind::Number(_) | TokenKind::Minus | TokenKind::Ident
            ) {
                Some(self.parse_const_expr()? as u8)
            } else {
                None
            }
        } else {
            None
        };

        let max_skip = if self.peek().kind == TokenKind::Comma {
            self.advance();
            if matches!(
                self.peek().kind,
                TokenKind::Number(_) | TokenKind::Minus | TokenKind::Ident
            ) {
                Some(self.parse_const_expr()? as u32)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Some(Statement::Align(AlignDirective {
            alignment,
            fill,
            max_skip,
            span,
        })))
    }

    fn parse_fill_directive(&mut self, span: Span) -> Result<Option<Statement>, AsmError> {
        let count = self.parse_const_expr()? as u32;
        let mut size = 1u8;
        let mut value = 0i64;
        if self.peek().kind == TokenKind::Comma {
            self.advance();
            size = self.parse_const_expr()? as u8;
            if self.peek().kind == TokenKind::Comma {
                self.advance();
                value = self.parse_const_expr()? as i64;
            }
        }
        Ok(Some(Statement::Fill(FillDirective {
            count,
            size,
            value,
            span,
        })))
    }

    fn parse_space_directive(&mut self, span: Span) -> Result<Option<Statement>, AsmError> {
        let size = self.parse_const_expr()? as u32;
        let fill = if self.peek().kind == TokenKind::Comma {
            self.advance();
            self.parse_const_expr()? as u8
        } else {
            0
        };
        Ok(Some(Statement::Space(SpaceDirective { size, fill, span })))
    }

    fn parse_org_directive(&mut self, span: Span) -> Result<Option<Statement>, AsmError> {
        let offset = self.parse_const_expr()? as u64;
        // Optional fill byte: .org offset, fill
        let fill = if self.peek().kind == TokenKind::Comma {
            self.advance(); // skip comma
            self.parse_const_expr()? as u8
        } else {
            0x00
        };
        Ok(Some(Statement::Org(OrgDirective { offset, fill, span })))
    }

    /// Parse a constant expression with full operator support.
    ///
    /// Supports (from lowest to highest precedence):
    /// - `|` (bitwise OR)
    /// - `^` (bitwise XOR)
    /// - `&` (bitwise AND)
    /// - `<<`, `>>` (shifts)
    /// - `+`, `-` (additive)
    /// - `*`, `/`, `%` (multiplicative)
    /// - Unary: `-`, `~` (negate, complement)
    /// - Atoms: numbers, identifiers (constants), parenthesized sub-expressions
    fn parse_const_expr(&mut self) -> Result<i128, AsmError> {
        self.const_expr_or()
    }

    // ── Precedence levels ──────────────────────────────────

    /// Bitwise OR: `a | b`
    fn const_expr_or(&mut self) -> Result<i128, AsmError> {
        let mut val = self.const_expr_xor()?;
        while self.peek().kind == TokenKind::Pipe {
            self.advance();
            val |= self.const_expr_xor()?;
        }
        Ok(val)
    }

    /// Bitwise XOR: `a ^ b`
    fn const_expr_xor(&mut self) -> Result<i128, AsmError> {
        let mut val = self.const_expr_and()?;
        while self.peek().kind == TokenKind::Caret {
            self.advance();
            val ^= self.const_expr_and()?;
        }
        Ok(val)
    }

    /// Bitwise AND: `a & b`
    fn const_expr_and(&mut self) -> Result<i128, AsmError> {
        let mut val = self.const_expr_shift()?;
        while self.peek().kind == TokenKind::Ampersand {
            self.advance();
            val &= self.const_expr_shift()?;
        }
        Ok(val)
    }

    /// Shifts: `a << b`, `a >> b`
    fn const_expr_shift(&mut self) -> Result<i128, AsmError> {
        let mut val = self.const_expr_add()?;
        loop {
            match self.peek().kind {
                TokenKind::LShift => {
                    self.advance();
                    let rhs = self.const_expr_add()?;
                    val = val.wrapping_shl(rhs as u32);
                }
                TokenKind::RShift => {
                    self.advance();
                    let rhs = self.const_expr_add()?;
                    val = val.wrapping_shr(rhs as u32);
                }
                _ => break,
            }
        }
        Ok(val)
    }

    /// Addition/subtraction: `a + b`, `a - b`
    fn const_expr_add(&mut self) -> Result<i128, AsmError> {
        let mut val = self.const_expr_mul()?;
        loop {
            match self.peek().kind {
                TokenKind::Plus => {
                    self.advance();
                    val = val.wrapping_add(self.const_expr_mul()?);
                }
                TokenKind::Minus => {
                    self.advance();
                    val = val.wrapping_sub(self.const_expr_mul()?);
                }
                _ => break,
            }
        }
        Ok(val)
    }

    /// Multiplication/division/modulo: `a * b`, `a / b`, `a % b`
    fn const_expr_mul(&mut self) -> Result<i128, AsmError> {
        let mut val = self.const_expr_unary()?;
        loop {
            match self.peek().kind {
                TokenKind::Star => {
                    self.advance();
                    val = val.wrapping_mul(self.const_expr_unary()?);
                }
                TokenKind::Slash => {
                    let span = self.peek().span;
                    self.advance();
                    let rhs = self.const_expr_unary()?;
                    if rhs == 0 {
                        return Err(AsmError::Syntax {
                            msg: String::from("division by zero in constant expression"),
                            span,
                        });
                    }
                    val /= rhs;
                }
                TokenKind::Percent => {
                    let span = self.peek().span;
                    self.advance();
                    let rhs = self.const_expr_unary()?;
                    if rhs == 0 {
                        return Err(AsmError::Syntax {
                            msg: String::from("modulo by zero in constant expression"),
                            span,
                        });
                    }
                    val %= rhs;
                }
                _ => break,
            }
        }
        Ok(val)
    }

    /// Unary operators: `-x`, `~x`
    fn const_expr_unary(&mut self) -> Result<i128, AsmError> {
        match self.peek().kind {
            TokenKind::Minus => {
                self.advance();
                Ok(-self.const_expr_unary()?)
            }
            TokenKind::Tilde => {
                self.advance();
                Ok(!self.const_expr_unary()?)
            }
            _ => self.const_expr_atom(),
        }
    }

    /// Atoms: numbers, identifiers (constant lookup), and `(expr)`.
    fn const_expr_atom(&mut self) -> Result<i128, AsmError> {
        let tok = self.peek().clone();
        match &tok.kind {
            TokenKind::Number(n) => {
                self.advance();
                Ok(*n)
            }
            TokenKind::Ident => {
                if let Some(&val) = self.constants.get(&*tok.text) {
                    self.advance();
                    Ok(val)
                } else {
                    Err(AsmError::Syntax {
                        msg: alloc::format!(
                            "expected constant expression, found undefined identifier '{}'",
                            tok.text
                        ),
                        span: tok.span,
                    })
                }
            }
            TokenKind::OpenParen => {
                self.advance(); // skip '('
                let val = self.parse_const_expr()?;
                if self.peek().kind != TokenKind::CloseParen {
                    return Err(AsmError::Syntax {
                        msg: String::from("expected ')' in constant expression"),
                        span: self.peek().span,
                    });
                }
                self.advance(); // skip ')'
                Ok(val)
            }
            _ => Err(AsmError::Syntax {
                msg: alloc::format!("expected constant expression, found '{}'", tok.text),
                span: tok.span,
            }),
        }
    }

    fn parse_instruction_or_prefix(&mut self) -> Result<Option<Statement>, AsmError> {
        let tok = self.peek().clone();

        // Check for `name = expression` constant assignment syntax
        if self.pos + 1 < self.tokens.len() && self.tokens[self.pos + 1].kind == TokenKind::Equals {
            let name = tok.text.to_string();
            let span = tok.span;
            self.advance(); // consume name
            self.advance(); // consume '='
            let value = self.parse_const_expr()?;
            self.constants.insert(name.clone(), value);
            return Ok(Some(Statement::Const(ConstDef { name, value, span })));
        }

        // Check for prefixes (case-insensitive, zero extra allocations).
        // We track the token position so we can lowercase the final mnemonic
        // only once, after the prefix loop.
        let mut prefixes = PrefixList::new();
        let mut mnemonic_pos = self.pos;
        let mut current_span = tok.span;

        loop {
            let prefix = {
                let text = &*self.tokens[mnemonic_pos].text;
                if text.eq_ignore_ascii_case("lock") {
                    Some(Prefix::Lock)
                } else if text.eq_ignore_ascii_case("rep")
                    || text.eq_ignore_ascii_case("repe")
                    || text.eq_ignore_ascii_case("repz")
                {
                    Some(Prefix::Rep)
                } else if text.eq_ignore_ascii_case("repne") || text.eq_ignore_ascii_case("repnz") {
                    Some(Prefix::Repne)
                } else {
                    None
                }
            };
            match prefix {
                Some(p) => {
                    prefixes.push(p);
                    self.advance();
                }
                None => break,
            }
            if self.at_end() || self.peek().kind != TokenKind::Ident {
                // Standalone prefix — treat as instruction with no operands
                return Ok(Some(Statement::Instruction(Instruction {
                    mnemonic: {
                        let mut lbuf = [0u8; 32];
                        Mnemonic::from(to_lower_buf(&self.tokens[mnemonic_pos].text, &mut lbuf))
                    },
                    operands: OperandList::new(),
                    size_hint: None,
                    prefixes,
                    opmask: None,
                    zeroing: false,
                    broadcast: None,
                    span: current_span,
                })));
            }
            mnemonic_pos = self.pos;
            current_span = self.tokens[mnemonic_pos].span;
        }

        // Now lowercase the mnemonic into stack-allocated Mnemonic (zero alloc).
        let mut mnemonic = {
            let mut lbuf = [0u8; 32];
            Mnemonic::from(to_lower_buf(&self.tokens[mnemonic_pos].text, &mut lbuf))
        };
        let mnemonic_span = current_span;
        self.advance(); // consume mnemonic

        // Parse operands
        let mut operands = OperandList::new();
        let mut size_hint = None;
        let mut opmask: Option<Register> = None;
        let mut zeroing = false;
        let mut broadcast: Option<BroadcastMode> = None;

        // AT&T syntax: strip size suffix from mnemonic (movq→mov, addl→add, etc.)
        if self.syntax == Syntax::Att {
            if let Some((base, sz)) = strip_att_suffix(&mnemonic) {
                mnemonic = base;
                size_hint = Some(sz);
            }
        }

        if !self.at_end() && !self.is_statement_end() {
            let (op, hint) = self.parse_operand()?;
            // ARM writeback: `R0!` after a register → wrap as Memory(PreIndex, base=R0)
            let op = if self.peek().kind == TokenKind::Bang {
                if let Operand::Register(r) = &op {
                    if r.is_arm() || r.is_aarch64() {
                        self.advance(); // consume '!'
                        Operand::Memory(Box::new(MemoryOperand {
                            base: Some(*r),
                            index: None,
                            scale: 1,
                            disp: 0,
                            disp_label: None,
                            segment: None,
                            size: None,
                            addr_mode: AddrMode::PreIndex,
                            index_subtract: false,
                        }))
                    } else {
                        op
                    }
                } else {
                    op
                }
            } else {
                op
            };
            operands.push(op);
            if hint.is_some() && size_hint.is_none() {
                size_hint = hint;
            }

            // ── AVX-512 decorators after first operand: {k1}, {k1}{z} ──
            if self.arch == Arch::X86_64 || self.arch == Arch::X86 {
                self.parse_evex_decorators(&mut opmask, &mut zeroing, &mut broadcast)?;
            }

            while self.peek().kind == TokenKind::Comma {
                self.advance(); // skip comma
                let (op, hint) = self.parse_operand()?;
                operands.push(op);
                if hint.is_some() && size_hint.is_none() {
                    size_hint = hint;
                }
                // ── AVX-512 decorators after subsequent operands: {1to8} etc. ──
                if self.arch == Arch::X86_64 || self.arch == Arch::X86 {
                    self.parse_evex_decorators(&mut opmask, &mut zeroing, &mut broadcast)?;
                }
            }
        }

        // AT&T syntax: reverse operand order (src, dst → dst, src)
        // Only for instructions with 2 or 3 operands.
        // Skip for instructions that don't reverse (e.g., 0 or 1 operand).
        if self.syntax == Syntax::Att && operands.len() >= 2 {
            operands.reverse();
        }

        Ok(Some(Statement::Instruction(Instruction {
            mnemonic,
            operands,
            size_hint,
            prefixes,
            opmask,
            zeroing,
            broadcast,
            span: mnemonic_span,
        })))
    }

    fn is_statement_end(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Newline | TokenKind::Eof)
    }

    /// Parse one operand. Returns the operand and optional size hint.
    /// Parse a single atom in an expression after `+` or `-`:
    /// either a numeric literal or an identifier (label / constant name).
    fn parse_expr_atom(&mut self, ctx_tok: &Token<'a>) -> Result<Expr, AsmError> {
        let next = self.peek().clone();
        match &next.kind {
            TokenKind::Number(n) => {
                self.advance();
                Ok(Expr::Num(*n))
            }
            TokenKind::Ident => {
                self.advance();
                // If identifier is a known constant, substitute immediately
                if let Some(&val) = self.constants.get(&*next.text) {
                    Ok(Expr::Num(val))
                } else {
                    Ok(Expr::Label(next.text.to_string()))
                }
            }
            _ => Err(AsmError::Syntax {
                msg: alloc::format!(
                    "expected number or identifier after '+'/'-' near '{}'",
                    ctx_tok.text
                ),
                span: next.span,
            }),
        }
    }

    /// Parse AVX-512 EVEX decorators: `{k1}`, `{k1}{z}`, `{1to2}`, `{1to4}`, `{1to8}`, `{1to16}`.
    ///
    /// Called after each operand to pick up decorators attached to that operand.
    /// Opmask and zeroing typically follow the destination, broadcast follows a memory operand.
    fn parse_evex_decorators(
        &mut self,
        opmask: &mut Option<Register>,
        zeroing: &mut bool,
        broadcast: &mut Option<BroadcastMode>,
    ) -> Result<(), AsmError> {
        while self.peek().kind == TokenKind::OpenBrace {
            let brace_span = self.peek().span;
            self.advance(); // skip '{'
            let tok = self.peek().clone();
            match tok.kind {
                TokenKind::Ident => {
                    let mut lbuf = [0u8; 32];
                    let lower = to_lower_buf(&tok.text, &mut lbuf);
                    if lower == "z" {
                        *zeroing = true;
                        self.advance();
                    } else if let Some(kreg) = parse_register_lower(lower, Arch::X86_64) {
                        if kreg.is_opmask() {
                            *opmask = Some(kreg);
                            self.advance();
                        } else {
                            return Err(AsmError::Syntax {
                                msg: String::from("expected opmask register k0-k7"),
                                span: tok.span,
                            });
                        }
                    } else {
                        return Err(AsmError::Syntax {
                            msg: String::from("unexpected identifier in AVX-512 decorator"),
                            span: tok.span,
                        });
                    }
                }
                TokenKind::Number(_) => {
                    // {1to2}, {1to4}, {1to8}, {1to16}
                    if tok.text == "1" {
                        self.advance(); // skip '1'
                                        // Expect identifier "to2", "to4", "to8", or "to16"
                        let next = self.peek().clone();
                        if next.kind == TokenKind::Ident {
                            let mut lbuf = [0u8; 32];
                            let nlower = to_lower_buf(&next.text, &mut lbuf);
                            let mode = match nlower {
                                "to2" => Some(BroadcastMode::OneToTwo),
                                "to4" => Some(BroadcastMode::OneToFour),
                                "to8" => Some(BroadcastMode::OneToEight),
                                "to16" => Some(BroadcastMode::OneToSixteen),
                                _ => None,
                            };
                            if let Some(m) = mode {
                                *broadcast = Some(m);
                                self.advance();
                            } else {
                                return Err(AsmError::Syntax {
                                    msg: String::from("expected 1to2, 1to4, 1to8, or 1to16"),
                                    span: next.span,
                                });
                            }
                        } else {
                            return Err(AsmError::Syntax {
                                msg: String::from("expected broadcast specifier (1to2/4/8/16)"),
                                span: next.span,
                            });
                        }
                    } else {
                        return Err(AsmError::Syntax {
                            msg: String::from("unexpected number in AVX-512 decorator"),
                            span: tok.span,
                        });
                    }
                }
                _ => {
                    // Not an AVX-512 decorator — this is a regular brace (ARM register list etc.)
                    // Put back by not advancing and returning. But we already consumed '{'.
                    // This shouldn't happen in normal x86 flow. Return error.
                    return Err(AsmError::Syntax {
                        msg: String::from("unexpected token in AVX-512 decorator"),
                        span: brace_span,
                    });
                }
            }
            // Expect closing '}'
            if self.peek().kind == TokenKind::CloseBrace {
                self.advance();
            } else {
                return Err(AsmError::Syntax {
                    msg: String::from("expected '}' after AVX-512 decorator"),
                    span: self.peek().span,
                });
            }
        }
        Ok(())
    }

    fn parse_operand(&mut self) -> Result<(Operand, Option<OperandSize>), AsmError> {
        let tok = self.peek().clone();

        // Check for size hint: byte/word/dword/qword [ptr]
        if tok.kind == TokenKind::Ident {
            let mut lbuf = [0u8; 32];
            let lower = to_lower_buf(&tok.text, &mut lbuf);
            if let Some(sz) = self.try_parse_size_hint(lower) {
                // Must be followed by a memory operand or ptr keyword
                if self.peek().kind == TokenKind::Ident
                    && self.peek().text.eq_ignore_ascii_case("ptr")
                {
                    self.advance(); // skip "ptr"
                }
                let (op, _) = self.parse_operand_inner()?;
                return Ok((op, Some(sz)));
            }
        }

        self.parse_operand_inner()
    }

    /// Case-insensitive size hint parsing (zero allocations).
    fn try_parse_size_hint(&mut self, ident: &str) -> Option<OperandSize> {
        if ident.eq_ignore_ascii_case("byte") {
            self.advance();
            Some(OperandSize::Byte)
        } else if ident.eq_ignore_ascii_case("word") {
            self.advance();
            Some(OperandSize::Word)
        } else if ident.eq_ignore_ascii_case("dword") {
            self.advance();
            Some(OperandSize::Dword)
        } else if ident.eq_ignore_ascii_case("qword") {
            self.advance();
            Some(OperandSize::Qword)
        } else if ident.eq_ignore_ascii_case("xmmword") || ident.eq_ignore_ascii_case("oword") {
            self.advance();
            Some(OperandSize::Xmmword)
        } else if ident.eq_ignore_ascii_case("ymmword") {
            self.advance();
            Some(OperandSize::Ymmword)
        } else if ident.eq_ignore_ascii_case("zmmword") {
            self.advance();
            Some(OperandSize::Zmmword)
        } else {
            None
        }
    }

    fn parse_operand_inner(&mut self) -> Result<(Operand, Option<OperandSize>), AsmError> {
        // AT&T syntax: dispatch to AT&T operand parser
        if self.syntax == Syntax::Att {
            return self.parse_att_operand();
        }

        let tok = self.peek().clone();

        match &tok.kind {
            // ARM register list: {R0, R1, R4, LR}
            // SVE braced vector register: {z0.s}
            // NEON braced vector register: {v0.4s}
            TokenKind::OpenBrace => {
                self.advance(); // consume '{'
                                // Check for SVE/NEON braced vector register: {z0.s}, {v0.4s}
                let first = self.peek().clone();
                if let TokenKind::Ident = &first.kind {
                    let mut lbuf = [0u8; 32];
                    let lower = to_lower_buf(&first.text, &mut lbuf);
                    if let Some(dot_pos) = lower.find('.') {
                        let reg_part = &lower[..dot_pos];
                        let arr_part = &lower[dot_pos + 1..];
                        if let Some(reg) = parse_register_lower(reg_part, self.arch) {
                            if reg.is_a64_sve_z() || reg.is_a64_vector() {
                                if let Some(arr) = VectorArrangement::parse(arr_part) {
                                    self.advance(); // consume 'z0.s' / 'v0.4s'
                                    if self.peek().kind != TokenKind::CloseBrace {
                                        return Err(AsmError::Syntax {
                                            msg: String::from("expected '}' after vector register"),
                                            span: self.peek().span,
                                        });
                                    }
                                    self.advance(); // consume '}'
                                    return Ok((Operand::VectorRegister(reg, arr), None));
                                }
                            }
                        }
                    }
                }
                // Fall through to regular register list parsing
                let mut regs = Vec::new();
                loop {
                    let rtok = self.peek().clone();
                    if rtok.kind == TokenKind::CloseBrace {
                        self.advance();
                        break;
                    }
                    if rtok.kind == TokenKind::Comma {
                        self.advance();
                        continue;
                    }
                    if let TokenKind::Ident = &rtok.kind {
                        let mut lbuf = [0u8; 32];
                        let lower = to_lower_buf(&rtok.text, &mut lbuf);
                        if let Some(reg) = parse_register_lower(lower, self.arch) {
                            self.advance();
                            regs.push(reg);
                            continue;
                        }
                    }
                    return Err(AsmError::Syntax {
                        msg: alloc::format!(
                            "expected register in register list, found '{}'",
                            rtok.text
                        ),
                        span: rtok.span,
                    });
                }
                Ok((Operand::RegisterList(regs), None))
            }

            // Memory operand
            TokenKind::OpenBracket => {
                let mem = self.parse_memory_operand()?;
                Ok((Operand::Memory(Box::new(mem)), None))
            }

            // Literal pool value: =imm (ARM/AArch64 LDR Rn, =value)
            TokenKind::Equals => {
                self.advance(); // consume '='
                let next = self.peek().clone();
                match next.kind {
                    TokenKind::Number(n) => {
                        self.advance();
                        Ok((Operand::LiteralPoolValue(n), None))
                    }
                    TokenKind::Minus => {
                        self.advance();
                        if let TokenKind::Number(n) = self.peek().kind {
                            self.advance();
                            Ok((Operand::LiteralPoolValue(-n), None))
                        } else {
                            Err(AsmError::Syntax {
                                msg: String::from("expected number after '=-'"),
                                span: next.span,
                            })
                        }
                    }
                    _ => Err(AsmError::Syntax {
                        msg: alloc::format!("expected number after '=', found '{}'", next.text),
                        span: next.span,
                    }),
                }
            }

            // RISC-V bare (reg) memory operand — equivalent to 0(reg)
            TokenKind::OpenParen if matches!(self.arch, Arch::Rv32 | Arch::Rv64) => {
                self.parse_riscv_mem_operand(0)
            }

            // Immediate — or RISC-V memory operand: offset(reg)
            TokenKind::Number(n) => {
                let val = *n;
                self.advance();
                // RISC-V: 0(sp), 8(a0), etc.
                if matches!(self.arch, Arch::Rv32 | Arch::Rv64)
                    && self.peek().kind == TokenKind::OpenParen
                {
                    return self.parse_riscv_mem_operand(val);
                }
                Ok((Operand::Immediate(val), None))
            }

            // Negative immediate — or RISC-V memory operand: -offset(reg)
            TokenKind::Minus => {
                self.advance();
                let next = self.peek().clone();
                if let TokenKind::Number(n) = next.kind {
                    self.advance();
                    let val = -n;
                    // RISC-V: -4(sp), -8(s0), etc.
                    if matches!(self.arch, Arch::Rv32 | Arch::Rv64)
                        && self.peek().kind == TokenKind::OpenParen
                    {
                        return self.parse_riscv_mem_operand(val);
                    }
                    Ok((Operand::Immediate(val), None))
                } else {
                    Err(AsmError::Syntax {
                        msg: String::from("expected number after '-'"),
                        span: tok.span,
                    })
                }
            }

            // Character literal as immediate
            TokenKind::CharLit(ch) => {
                self.advance();
                Ok((Operand::Immediate(*ch as i128), None))
            }

            // Identifier: register or label reference
            TokenKind::Ident => {
                let mut lbuf = [0u8; 32];
                let lower = to_lower_buf(&tok.text, &mut lbuf);

                // Check for segment:memory pair (e.g., fs:[rax])
                if is_segment_name(lower) {
                    let seg = match parse_segment(lower) {
                        Some(s) => s,
                        None => {
                            return Err(AsmError::Syntax {
                                msg: alloc::format!("unknown segment register: {}", lower),
                                span: tok.span,
                            });
                        }
                    };
                    // Check if followed by colon and bracket
                    if self.pos + 1 < self.tokens.len()
                        && self.tokens[self.pos + 1].kind == TokenKind::Colon
                    {
                        self.advance(); // consume seg name
                        self.advance(); // consume ':'

                        if self.peek().kind == TokenKind::OpenBracket {
                            let mut mem = self.parse_memory_operand()?;
                            mem.segment = Some(seg);
                            return Ok((Operand::Memory(Box::new(mem)), None));
                        }
                    }
                }

                // Try register (possibly with vector arrangement: v0.4s, z0.s, p0.b)
                if let Some(dot_pos) = lower.find('.') {
                    // Check for vector register with arrangement specifier
                    let reg_part = &lower[..dot_pos];
                    let arr_part = &lower[dot_pos + 1..];
                    if let Some(reg) = parse_register_lower(reg_part, self.arch) {
                        if reg.is_a64_vector() || reg.is_a64_sve_z() || reg.is_a64_sve_p() {
                            if let Some(arr) = VectorArrangement::parse(arr_part) {
                                self.advance();
                                return Ok((Operand::VectorRegister(reg, arr), None));
                            }
                        }
                        // RISC-V V extension: v0.t → mask operand (bare register)
                        if reg.is_riscv_vec() && arr_part == "t" {
                            self.advance();
                            return Ok((Operand::Register(reg), None));
                        }
                    }
                }

                if let Some(reg) = parse_register_lower(lower, self.arch) {
                    self.advance();
                    // SVE predicate qualifier: p0/m or p0/z
                    if reg.is_a64_sve_p() && self.peek().kind == TokenKind::Slash {
                        let next_pos = self.pos + 1;
                        if next_pos < self.tokens.len() {
                            let qual_text = &self.tokens[next_pos].text;
                            let qual = if qual_text.eq_ignore_ascii_case("m") {
                                Some(SvePredQual::Merging)
                            } else if qual_text.eq_ignore_ascii_case("z") {
                                Some(SvePredQual::Zeroing)
                            } else {
                                None
                            };
                            if let Some(q) = qual {
                                self.advance(); // consume '/'
                                self.advance(); // consume 'm' or 'z'
                                return Ok((Operand::SvePredicate(reg, q), None));
                            }
                        }
                    }
                    return Ok((Operand::Register(reg), None));
                }

                // Check if it's a previously-defined constant (bare identifier → Immediate)
                if let Some(&val) = self.constants.get(&*tok.text) {
                    self.advance();
                    // Check for trailing +/- chains (e.g., CONST + 5)
                    let mut result = val;
                    loop {
                        if self.peek().kind == TokenKind::Plus {
                            self.advance();
                            let next = self.peek().clone();
                            match &next.kind {
                                TokenKind::Number(n) => {
                                    self.advance();
                                    result += n;
                                }
                                TokenKind::Ident => {
                                    if let Some(&v) = self.constants.get(&*next.text) {
                                        self.advance();
                                        result += v;
                                    } else {
                                        break;
                                    }
                                }
                                _ => break,
                            }
                        } else if self.peek().kind == TokenKind::Minus {
                            self.advance();
                            let next = self.peek().clone();
                            match &next.kind {
                                TokenKind::Number(n) => {
                                    self.advance();
                                    result -= n;
                                }
                                TokenKind::Ident => {
                                    if let Some(&v) = self.constants.get(&*next.text) {
                                        self.advance();
                                        result -= v;
                                    } else {
                                        break;
                                    }
                                }
                                _ => break,
                            }
                        } else {
                            break;
                        }
                    }
                    return Ok((Operand::Immediate(result), None));
                }

                // Label reference
                self.advance();
                // Build an expression tree for label+offset, label-offset,
                // label+ident, etc.
                let mut expr: Expr = Expr::Label(tok.text.to_string());
                let mut is_expression = false;
                loop {
                    if self.peek().kind == TokenKind::Plus {
                        self.advance();
                        let rhs = self.parse_expr_atom(&tok)?;
                        expr = Expr::Add(Box::new(expr), Box::new(rhs));
                        is_expression = true;
                    } else if self.peek().kind == TokenKind::Minus {
                        self.advance();
                        let rhs = self.parse_expr_atom(&tok)?;
                        expr = Expr::Sub(Box::new(expr), Box::new(rhs));
                        is_expression = true;
                    } else {
                        break;
                    }
                }

                if is_expression {
                    // Try to resolve all-constant expressions eagerly
                    expr.resolve_constants(|name| self.constants.get(name).copied());
                    if let Some(val) = expr.eval() {
                        return Ok((Operand::Immediate(val), None));
                    }
                    return Ok((Operand::Expression(expr), None));
                }

                Ok((Operand::Label(tok.text.to_string()), None))
            }

            // Numeric label references
            TokenKind::NumericLabelFwd(n) => {
                self.advance();
                Ok((Operand::Label(alloc::format!("{}f", n)), None))
            }
            TokenKind::NumericLabelBwd(n) => {
                self.advance();
                Ok((Operand::Label(alloc::format!("{}b", n)), None))
            }

            _ => Err(AsmError::Syntax {
                msg: alloc::format!("expected operand, found '{}'", tok.text),
                span: tok.span,
            }),
        }
    }

    // ── AT&T / GAS syntax operand parsing ────────────────────────────────

    /// Parse a single AT&T-syntax operand.
    ///
    /// Forms:
    /// - `$imm` or `$-imm` or `$label` — immediate / label reference
    /// - `%reg` — register
    /// - `%seg:disp(%base, %index, scale)` — memory with segment override
    /// - `disp(%base, %index, scale)` — memory
    /// - `(%base)` — memory (zero displacement)
    /// - `label` or `label+offset` — label reference (for branches)
    fn parse_att_operand(&mut self) -> Result<(Operand, Option<OperandSize>), AsmError> {
        let tok = self.peek().clone();

        match &tok.kind {
            // $imm — immediate
            TokenKind::Dollar => {
                self.advance(); // consume '$'
                let next = self.peek().clone();
                match &next.kind {
                    TokenKind::Number(n) => {
                        let val = *n;
                        self.advance();
                        Ok((Operand::Immediate(val), None))
                    }
                    TokenKind::Minus => {
                        self.advance(); // consume '-'
                        let num_tok = self.peek().clone();
                        if let TokenKind::Number(n) = num_tok.kind {
                            self.advance();
                            Ok((Operand::Immediate(-n), None))
                        } else {
                            Err(AsmError::Syntax {
                                msg: String::from("expected number after '$-'"),
                                span: num_tok.span,
                            })
                        }
                    }
                    TokenKind::Ident => {
                        let name = next.text.to_string();
                        self.advance();
                        // Check if it's a known constant
                        if let Some(&val) = self.constants.get(&name) {
                            Ok((Operand::Immediate(val), None))
                        } else {
                            // Label reference as immediate
                            Ok((Operand::Label(name), None))
                        }
                    }
                    _ => Err(AsmError::Syntax {
                        msg: alloc::format!(
                            "expected number or identifier after '$', found '{}'",
                            next.text
                        ),
                        span: next.span,
                    }),
                }
            }

            // %reg or %seg:... — register or segment-prefixed memory
            TokenKind::Percent => {
                self.advance(); // consume '%'
                let reg_tok = self.peek().clone();
                if reg_tok.kind != TokenKind::Ident {
                    return Err(AsmError::Syntax {
                        msg: alloc::format!(
                            "expected register name after '%', found '{}'",
                            reg_tok.text
                        ),
                        span: reg_tok.span,
                    });
                }
                let mut lbuf = [0u8; 32];
                let lower = to_lower_buf(&reg_tok.text, &mut lbuf);

                // Check for segment register followed by ':'
                if is_segment_name(lower) {
                    // SAFETY: is_segment_name() matches the exact same set as
                    // parse_segment(), so unwrap is unreachable.
                    let seg = parse_segment(lower).unwrap();
                    // Peek ahead: if next is ':', this is a segment-prefixed memory
                    if self.pos + 1 < self.tokens.len()
                        && self.tokens[self.pos + 1].kind == TokenKind::Colon
                    {
                        self.advance(); // consume segment name
                        self.advance(); // consume ':'
                                        // Parse the rest as memory: could be disp(%base,...), (%base,...), or bare disp
                        let (seg_disp, seg_disp_label) = match self.peek().kind {
                            TokenKind::Number(n) => {
                                let val = n;
                                self.advance();
                                (val as i64, None)
                            }
                            TokenKind::Minus => {
                                self.advance();
                                if let TokenKind::Number(n) = self.peek().kind {
                                    let val = n;
                                    self.advance();
                                    (-(val as i64), None)
                                } else {
                                    (0, None)
                                }
                            }
                            _ => (0, None),
                        };
                        let mut mem = self.parse_att_memory_operand(seg_disp, seg_disp_label)?;
                        mem.segment = Some(seg);
                        return Ok((Operand::Memory(Box::new(mem)), None));
                    }
                }

                // Regular register
                if let Some(reg) = parse_register_lower(lower, self.arch) {
                    self.advance();
                    Ok((Operand::Register(reg), None))
                } else {
                    Err(AsmError::Syntax {
                        msg: alloc::format!("unknown register: %{}", lower),
                        span: reg_tok.span,
                    })
                }
            }

            // (%base, ...) — memory with zero displacement
            TokenKind::OpenParen => {
                let mem = self.parse_att_memory_operand(0, None)?;
                Ok((Operand::Memory(Box::new(mem)), None))
            }

            // number — could be displacement for memory or standalone immediate (for branches)
            TokenKind::Number(n) => {
                let val = *n;
                self.advance();
                // If followed by '(', this is a memory operand: disp(%base, %index, scale)
                if self.peek().kind == TokenKind::OpenParen {
                    let mem = self.parse_att_memory_operand(val as i64, None)?;
                    Ok((Operand::Memory(Box::new(mem)), None))
                } else {
                    // Standalone number (e.g., for branch targets, port numbers)
                    Ok((Operand::Immediate(val), None))
                }
            }

            // -number — negative displacement for memory
            TokenKind::Minus => {
                self.advance(); // consume '-'
                let next = self.peek().clone();
                if let TokenKind::Number(n) = next.kind {
                    self.advance();
                    let val = -n;
                    if self.peek().kind == TokenKind::OpenParen {
                        let mem = self.parse_att_memory_operand(val as i64, None)?;
                        Ok((Operand::Memory(Box::new(mem)), None))
                    } else {
                        Ok((Operand::Immediate(val), None))
                    }
                } else {
                    Err(AsmError::Syntax {
                        msg: String::from("expected number after '-' in AT&T operand"),
                        span: tok.span,
                    })
                }
            }

            // Identifier — label reference (for branches: jmp label, call label)
            TokenKind::Ident => {
                let name = tok.text.to_string();
                self.advance();
                // Check if it's a known constant
                if let Some(&val) = self.constants.get(&name) {
                    // If followed by '(', treat as memory displacement
                    if self.peek().kind == TokenKind::OpenParen {
                        let mem = self.parse_att_memory_operand(val as i64, None)?;
                        return Ok((Operand::Memory(Box::new(mem)), None));
                    }
                    return Ok((Operand::Immediate(val), None));
                }
                // Build expression with optional +/- offset
                let mut expr = Expr::Label(name.clone());
                let mut has_offset = false;
                loop {
                    if self.peek().kind == TokenKind::Plus {
                        self.advance();
                        let atom = self.parse_expr_atom(&tok)?;
                        expr = Expr::Add(Box::new(expr), Box::new(atom));
                        has_offset = true;
                    } else if self.peek().kind == TokenKind::Minus {
                        self.advance();
                        let atom = self.parse_expr_atom(&tok)?;
                        expr = Expr::Sub(Box::new(expr), Box::new(atom));
                        has_offset = true;
                    } else {
                        break;
                    }
                }
                if has_offset {
                    Ok((Operand::Expression(expr), None))
                } else {
                    Ok((Operand::Label(name), None))
                }
            }

            // Numeric label forward/backward references
            TokenKind::NumericLabelFwd(n) => {
                let n = *n;
                self.advance();
                Ok((Operand::Label(alloc::format!("{}f", n)), None))
            }
            TokenKind::NumericLabelBwd(n) => {
                let n = *n;
                self.advance();
                Ok((Operand::Label(alloc::format!("{}b", n)), None))
            }

            // Star — indirect jump/call: *%rax, *(%rax), *label
            TokenKind::Star => {
                self.advance(); // consume '*'
                                // The suboperand is the actual target — parse recursively
                self.parse_att_operand()
            }

            _ => Err(AsmError::Syntax {
                msg: alloc::format!("unexpected token in AT&T operand: '{}'", tok.text),
                span: tok.span,
            }),
        }
    }

    /// Parse AT&T memory operand: `(%base)`, `(%base, %index)`,
    /// `(%base, %index, scale)`, `disp(%base, ...)`.
    /// `disp` has already been parsed; it's passed as the `disp` param.
    fn parse_att_memory_operand(
        &mut self,
        disp: i64,
        disp_label: Option<String>,
    ) -> Result<MemoryOperand, AsmError> {
        let open = self.peek().clone();
        if open.kind != TokenKind::OpenParen {
            return Err(AsmError::Syntax {
                msg: alloc::format!("expected '(' in AT&T memory operand, found '{}'", open.text),
                span: open.span,
            });
        }
        self.advance(); // consume '('

        let mut base = None;
        let mut index = None;
        let mut scale: u8 = 1;

        // Parse base register: %reg
        if self.peek().kind == TokenKind::Percent {
            self.advance(); // consume '%'
            let reg_tok = self.peek().clone();
            let mut lbuf = [0u8; 32];
            let lower = to_lower_buf(&reg_tok.text, &mut lbuf);
            base =
                Some(
                    parse_register_lower(lower, self.arch).ok_or_else(|| AsmError::Syntax {
                        msg: alloc::format!("unknown register: %{}", lower),
                        span: reg_tok.span,
                    })?,
                );
            self.advance();
        }

        // Comma → index register
        if self.peek().kind == TokenKind::Comma {
            self.advance(); // consume ','
            if self.peek().kind == TokenKind::Percent {
                self.advance(); // consume '%'
                let reg_tok = self.peek().clone();
                let mut lbuf = [0u8; 32];
                let lower = to_lower_buf(&reg_tok.text, &mut lbuf);
                index = Some(parse_register_lower(lower, self.arch).ok_or_else(|| {
                    AsmError::Syntax {
                        msg: alloc::format!("unknown register: %{}", lower),
                        span: reg_tok.span,
                    }
                })?);
                self.advance();
            }

            // Comma → scale factor
            if self.peek().kind == TokenKind::Comma {
                self.advance(); // consume ','
                let scale_tok = self.peek().clone();
                if let TokenKind::Number(n) = scale_tok.kind {
                    scale = n as u8;
                    self.advance();
                } else {
                    return Err(AsmError::Syntax {
                        msg: alloc::format!(
                            "expected scale factor (1,2,4,8), found '{}'",
                            scale_tok.text
                        ),
                        span: scale_tok.span,
                    });
                }
            }
        }

        // Expect closing ')'
        let close = self.peek().clone();
        if close.kind != TokenKind::CloseParen {
            return Err(AsmError::Syntax {
                msg: alloc::format!(
                    "expected ')' in AT&T memory operand, found '{}'",
                    close.text
                ),
                span: close.span,
            });
        }
        self.advance();

        Ok(MemoryOperand {
            base,
            index,
            scale,
            disp,
            disp_label,
            segment: None,
            size: None,
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        })
    }

    /// Parse a RISC-V memory operand: `offset(reg)`.
    /// Called after the offset has been consumed. Expects `(` reg `)`.
    fn parse_riscv_mem_operand(
        &mut self,
        offset: i128,
    ) -> Result<(Operand, Option<OperandSize>), AsmError> {
        let open_tok = self.advance().clone(); // consume '('
        debug_assert_eq!(open_tok.kind, TokenKind::OpenParen);

        let reg_tok = self.peek().clone();
        let mut lbuf = [0u8; 32];
        let lower = to_lower_buf(&reg_tok.text, &mut lbuf);
        let reg = if let Some(r) = parse_register_lower(lower, self.arch) {
            self.advance();
            r
        } else {
            return Err(AsmError::Syntax {
                msg: alloc::format!(
                    "expected register in memory operand, found '{}'",
                    reg_tok.text
                ),
                span: reg_tok.span,
            });
        };

        // Expect closing ')'
        let close = self.peek().clone();
        if close.kind != TokenKind::CloseParen {
            return Err(AsmError::Syntax {
                msg: alloc::format!("expected ')' after register, found '{}'", close.text),
                span: close.span,
            });
        }
        self.advance();

        let mem = MemoryOperand {
            base: Some(reg),
            disp: offset as i64,
            ..Default::default()
        };
        Ok((Operand::Memory(Box::new(mem)), None))
    }

    /// Parse a memory operand: `[base + index*scale + disp]`
    fn parse_memory_operand(&mut self) -> Result<MemoryOperand, AsmError> {
        let open = self.advance().clone(); // consume '['
        debug_assert_eq!(open.kind, TokenKind::OpenBracket);

        let mut mem = MemoryOperand::default();
        let mut _expect_term = true;
        let mut sign: i64 = 1;

        while self.peek().kind != TokenKind::CloseBracket {
            if self.at_end() {
                return Err(AsmError::Syntax {
                    msg: String::from("unterminated memory operand, expected ']'"),
                    span: open.span,
                });
            }

            let tok = self.peek().clone();

            match &tok.kind {
                TokenKind::Plus | TokenKind::Comma => {
                    // Comma inside brackets is ARM/AArch64 syntax: [Rn, #offset]
                    self.advance();
                    sign = 1;
                    _expect_term = true;
                    continue;
                }
                TokenKind::Minus => {
                    self.advance();
                    sign = -1;
                    _expect_term = true;
                    continue;
                }
                TokenKind::Ident => {
                    let mut lbuf = [0u8; 32];
                    let lower = to_lower_buf(&tok.text, &mut lbuf);
                    if let Some(reg) = parse_register_lower(lower, self.arch) {
                        self.advance();
                        // Check if this register is multiplied by a scale
                        if self.peek().kind == TokenKind::Star {
                            self.advance(); // consume '*'
                            let scale_tok = self.peek().clone();
                            if let TokenKind::Number(s) = scale_tok.kind {
                                if !matches!(s, 1 | 2 | 4 | 8) {
                                    return Err(AsmError::Syntax {
                                        msg: String::from("scale factor must be 1, 2, 4, or 8"),
                                        span: scale_tok.span,
                                    });
                                }
                                self.advance();
                                mem.index = Some(reg);
                                mem.scale = s as u8;
                                mem.index_subtract = sign < 0;
                            } else {
                                return Err(AsmError::Syntax {
                                    msg: String::from("expected scale factor (1, 2, 4, or 8)"),
                                    span: scale_tok.span,
                                });
                            }
                        } else if mem.base.is_none() {
                            mem.base = Some(reg);
                        } else if mem.index.is_none() {
                            mem.index = Some(reg);
                            mem.scale = 1;
                            mem.index_subtract = sign < 0;
                        } else {
                            return Err(AsmError::Syntax {
                                msg: String::from("too many registers in memory operand"),
                                span: tok.span,
                            });
                        }
                    } else {
                        // Label in memory operand
                        self.advance();
                        mem.disp_label = Some(tok.text.to_string());
                    }
                    _expect_term = false;
                }
                TokenKind::Number(n) => {
                    self.advance();
                    // Check if number*register (e.g., 4*rbx)
                    if self.peek().kind == TokenKind::Star {
                        self.advance(); // consume '*'
                        let reg_tok = self.peek().clone();
                        if reg_tok.kind == TokenKind::Ident {
                            let mut lbuf = [0u8; 32];
                            let lower = to_lower_buf(&reg_tok.text, &mut lbuf);
                            if let Some(reg) = parse_register_lower(lower, self.arch) {
                                if !matches!(*n, 1 | 2 | 4 | 8) {
                                    return Err(AsmError::Syntax {
                                        msg: String::from("scale factor must be 1, 2, 4, or 8"),
                                        span: tok.span,
                                    });
                                }
                                self.advance();
                                mem.index = Some(reg);
                                mem.scale = *n as u8;
                                mem.index_subtract = sign < 0;
                                _expect_term = false;
                                continue;
                            }
                        }
                        return Err(AsmError::Syntax {
                            msg: String::from("expected register after scale factor"),
                            span: reg_tok.span,
                        });
                    }
                    mem.disp = mem.disp.wrapping_add(sign * (*n as i64));
                    _expect_term = false;
                }
                _ => {
                    return Err(AsmError::Syntax {
                        msg: alloc::format!("unexpected token '{}' in memory operand", tok.text),
                        span: tok.span,
                    });
                }
            }
        }

        self.advance(); // consume ']'

        // ARM/AArch64 writeback: [Rn, #imm]!  → pre-index
        if self.peek().kind == TokenKind::Bang {
            self.advance(); // consume '!'
            mem.addr_mode = AddrMode::PreIndex;
        }

        // Validate: RSP/ESP/SP cannot be used as a SIB index register.
        // In the SIB byte, index code 0b100 (RSP's base_code) means "no index."
        // R12 (base_code 4 + REX.X) IS valid because REX.X distinguishes it.
        if let Some(idx) = mem.index {
            if idx.base_code() == 4 && !idx.is_extended() {
                return Err(AsmError::Syntax {
                    msg: String::from("RSP/ESP/SP cannot be used as a SIB index register"),
                    span: open.span,
                });
            }
        }

        Ok(mem)
    }
}

/// Parse a register name — **case-insensitive**, zero heap allocations.
///
/// Architecture-aware: conflicting names like `r8`-`r15`, `sp` are resolved
/// based on the target architecture.
pub fn parse_register(name: &str, arch: Arch) -> Option<Register> {
    // Stack-based lowercase (register names are at most ~5 chars; 16 is plenty).
    let mut buf = [0u8; 16];
    let name = to_lower_buf(name, &mut buf);
    parse_register_lower(name, arch)
}

/// Inner register parser — expects **already-lowered** input.
fn parse_register_lower(name: &str, arch: Arch) -> Option<Register> {
    use Register::*;

    // Architecture-specific fast path for ARM / AArch64 / RISC-V
    match arch {
        Arch::Arm | Arch::Thumb => return parse_register_arm(name),
        Arch::Aarch64 => return parse_register_aarch64(name),
        Arch::Rv32 | Arch::Rv64 => return parse_register_riscv(name),
        _ => {}
    }

    // x86 / x86-64 register names
    match name {
        // 64-bit GP
        "rax" => Some(Rax),
        "rcx" => Some(Rcx),
        "rdx" => Some(Rdx),
        "rbx" => Some(Rbx),
        "rsp" => Some(Rsp),
        "rbp" => Some(Rbp),
        "rsi" => Some(Rsi),
        "rdi" => Some(Rdi),
        "r8" => Some(R8),
        "r9" => Some(R9),
        "r10" => Some(R10),
        "r11" => Some(R11),
        "r12" => Some(R12),
        "r13" => Some(R13),
        "r14" => Some(R14),
        "r15" => Some(R15),
        // 32-bit GP
        "eax" => Some(Eax),
        "ecx" => Some(Ecx),
        "edx" => Some(Edx),
        "ebx" => Some(Ebx),
        "esp" => Some(Esp),
        "ebp" => Some(Ebp),
        "esi" => Some(Esi),
        "edi" => Some(Edi),
        "r8d" => Some(R8d),
        "r9d" => Some(R9d),
        "r10d" => Some(R10d),
        "r11d" => Some(R11d),
        "r12d" => Some(R12d),
        "r13d" => Some(R13d),
        "r14d" => Some(R14d),
        "r15d" => Some(R15d),
        // 16-bit GP
        "ax" => Some(Ax),
        "cx" => Some(Cx),
        "dx" => Some(Dx),
        "bx" => Some(Bx),
        "sp" => Some(Sp),
        "bp" => Some(Bp),
        "si" => Some(Si),
        "di" => Some(Di),
        "r8w" => Some(R8w),
        "r9w" => Some(R9w),
        "r10w" => Some(R10w),
        "r11w" => Some(R11w),
        "r12w" => Some(R12w),
        "r13w" => Some(R13w),
        "r14w" => Some(R14w),
        "r15w" => Some(R15w),
        // 8-bit GP
        "al" => Some(Al),
        "cl" => Some(Cl),
        "dl" => Some(Dl),
        "bl" => Some(Bl),
        "spl" => Some(Spl),
        "bpl" => Some(Bpl),
        "sil" => Some(Sil),
        "dil" => Some(Dil),
        "ah" => Some(Ah),
        "ch" => Some(Ch),
        "dh" => Some(Dh),
        "bh" => Some(Bh),
        "r8b" => Some(R8b),
        "r9b" => Some(R9b),
        "r10b" => Some(R10b),
        "r11b" => Some(R11b),
        "r12b" => Some(R12b),
        "r13b" => Some(R13b),
        "r14b" => Some(R14b),
        "r15b" => Some(R15b),
        // Special
        "rip" => Some(Rip),
        "eip" => Some(Eip),
        // Segment
        "cs" => Some(Cs),
        "ds" => Some(Ds),
        "es" => Some(Es),
        "fs" => Some(Fs),
        "gs" => Some(Gs),
        "ss" => Some(Ss),
        // XMM
        "xmm0" => Some(Xmm0),
        "xmm1" => Some(Xmm1),
        "xmm2" => Some(Xmm2),
        "xmm3" => Some(Xmm3),
        "xmm4" => Some(Xmm4),
        "xmm5" => Some(Xmm5),
        "xmm6" => Some(Xmm6),
        "xmm7" => Some(Xmm7),
        "xmm8" => Some(Xmm8),
        "xmm9" => Some(Xmm9),
        "xmm10" => Some(Xmm10),
        "xmm11" => Some(Xmm11),
        "xmm12" => Some(Xmm12),
        "xmm13" => Some(Xmm13),
        "xmm14" => Some(Xmm14),
        "xmm15" => Some(Xmm15),
        // YMM
        "ymm0" => Some(Ymm0),
        "ymm1" => Some(Ymm1),
        "ymm2" => Some(Ymm2),
        "ymm3" => Some(Ymm3),
        "ymm4" => Some(Ymm4),
        "ymm5" => Some(Ymm5),
        "ymm6" => Some(Ymm6),
        "ymm7" => Some(Ymm7),
        "ymm8" => Some(Ymm8),
        "ymm9" => Some(Ymm9),
        "ymm10" => Some(Ymm10),
        "ymm11" => Some(Ymm11),
        "ymm12" => Some(Ymm12),
        "ymm13" => Some(Ymm13),
        "ymm14" => Some(Ymm14),
        "ymm15" => Some(Ymm15),
        // ZMM
        "zmm0" => Some(Zmm0),
        "zmm1" => Some(Zmm1),
        "zmm2" => Some(Zmm2),
        "zmm3" => Some(Zmm3),
        "zmm4" => Some(Zmm4),
        "zmm5" => Some(Zmm5),
        "zmm6" => Some(Zmm6),
        "zmm7" => Some(Zmm7),
        "zmm8" => Some(Zmm8),
        "zmm9" => Some(Zmm9),
        "zmm10" => Some(Zmm10),
        "zmm11" => Some(Zmm11),
        "zmm12" => Some(Zmm12),
        "zmm13" => Some(Zmm13),
        "zmm14" => Some(Zmm14),
        "zmm15" => Some(Zmm15),
        "zmm16" => Some(Zmm16),
        "zmm17" => Some(Zmm17),
        "zmm18" => Some(Zmm18),
        "zmm19" => Some(Zmm19),
        "zmm20" => Some(Zmm20),
        "zmm21" => Some(Zmm21),
        "zmm22" => Some(Zmm22),
        "zmm23" => Some(Zmm23),
        "zmm24" => Some(Zmm24),
        "zmm25" => Some(Zmm25),
        "zmm26" => Some(Zmm26),
        "zmm27" => Some(Zmm27),
        "zmm28" => Some(Zmm28),
        "zmm29" => Some(Zmm29),
        "zmm30" => Some(Zmm30),
        "zmm31" => Some(Zmm31),
        // Opmask
        "k0" => Some(K0),
        "k1" => Some(K1),
        "k2" => Some(K2),
        "k3" => Some(K3),
        "k4" => Some(K4),
        "k5" => Some(K5),
        "k6" => Some(K6),
        "k7" => Some(K7),
        _ => None,
    }
}

fn is_segment_name(name: &str) -> bool {
    name.eq_ignore_ascii_case("cs")
        || name.eq_ignore_ascii_case("ds")
        || name.eq_ignore_ascii_case("es")
        || name.eq_ignore_ascii_case("fs")
        || name.eq_ignore_ascii_case("gs")
        || name.eq_ignore_ascii_case("ss")
}

fn parse_segment(name: &str) -> Option<Register> {
    if name.eq_ignore_ascii_case("cs") {
        Some(Register::Cs)
    } else if name.eq_ignore_ascii_case("ds") {
        Some(Register::Ds)
    } else if name.eq_ignore_ascii_case("es") {
        Some(Register::Es)
    } else if name.eq_ignore_ascii_case("fs") {
        Some(Register::Fs)
    } else if name.eq_ignore_ascii_case("gs") {
        Some(Register::Gs)
    } else if name.eq_ignore_ascii_case("ss") {
        Some(Register::Ss)
    } else {
        None
    }
}

/// ARM32 register name parser.
fn parse_register_arm(name: &str) -> Option<Register> {
    use Register::*;
    match name {
        "r0" => Some(ArmR0),
        "r1" => Some(ArmR1),
        "r2" => Some(ArmR2),
        "r3" => Some(ArmR3),
        "r4" => Some(ArmR4),
        "r5" => Some(ArmR5),
        "r6" => Some(ArmR6),
        "r7" => Some(ArmR7),
        "r8" => Some(ArmR8),
        "r9" => Some(ArmR9),
        "r10" => Some(ArmR10),
        "r11" | "fp" => Some(ArmR11),
        "r12" | "ip" => Some(ArmR12),
        "r13" | "sp" => Some(ArmSp),
        "r14" | "lr" => Some(ArmLr),
        "r15" | "pc" => Some(ArmPc),
        "cpsr" => Some(ArmCpsr),
        _ => None,
    }
}

/// AArch64 register name parser.
fn parse_register_aarch64(name: &str) -> Option<Register> {
    use Register::*;
    match name {
        "x0" => Some(A64X0),
        "x1" => Some(A64X1),
        "x2" => Some(A64X2),
        "x3" => Some(A64X3),
        "x4" => Some(A64X4),
        "x5" => Some(A64X5),
        "x6" => Some(A64X6),
        "x7" => Some(A64X7),
        "x8" => Some(A64X8),
        "x9" => Some(A64X9),
        "x10" => Some(A64X10),
        "x11" => Some(A64X11),
        "x12" => Some(A64X12),
        "x13" => Some(A64X13),
        "x14" => Some(A64X14),
        "x15" => Some(A64X15),
        "x16" => Some(A64X16),
        "x17" => Some(A64X17),
        "x18" => Some(A64X18),
        "x19" => Some(A64X19),
        "x20" => Some(A64X20),
        "x21" => Some(A64X21),
        "x22" => Some(A64X22),
        "x23" => Some(A64X23),
        "x24" => Some(A64X24),
        "x25" => Some(A64X25),
        "x26" => Some(A64X26),
        "x27" => Some(A64X27),
        "x28" => Some(A64X28),
        "x29" => Some(A64X29),
        "x30" => Some(A64X30),
        "fp" => Some(A64X29),
        "lr" => Some(A64X30),
        "sp" => Some(A64Sp),
        "xzr" => Some(A64Xzr),
        "w0" => Some(A64W0),
        "w1" => Some(A64W1),
        "w2" => Some(A64W2),
        "w3" => Some(A64W3),
        "w4" => Some(A64W4),
        "w5" => Some(A64W5),
        "w6" => Some(A64W6),
        "w7" => Some(A64W7),
        "w8" => Some(A64W8),
        "w9" => Some(A64W9),
        "w10" => Some(A64W10),
        "w11" => Some(A64W11),
        "w12" => Some(A64W12),
        "w13" => Some(A64W13),
        "w14" => Some(A64W14),
        "w15" => Some(A64W15),
        "w16" => Some(A64W16),
        "w17" => Some(A64W17),
        "w18" => Some(A64W18),
        "w19" => Some(A64W19),
        "w20" => Some(A64W20),
        "w21" => Some(A64W21),
        "w22" => Some(A64W22),
        "w23" => Some(A64W23),
        "w24" => Some(A64W24),
        "w25" => Some(A64W25),
        "w26" => Some(A64W26),
        "w27" => Some(A64W27),
        "w28" => Some(A64W28),
        "w29" => Some(A64W29),
        "w30" => Some(A64W30),
        "wzr" => Some(A64Wzr),
        // SIMD/FP vector registers (V0–V31)
        "v0" => Some(A64V0),
        "v1" => Some(A64V1),
        "v2" => Some(A64V2),
        "v3" => Some(A64V3),
        "v4" => Some(A64V4),
        "v5" => Some(A64V5),
        "v6" => Some(A64V6),
        "v7" => Some(A64V7),
        "v8" => Some(A64V8),
        "v9" => Some(A64V9),
        "v10" => Some(A64V10),
        "v11" => Some(A64V11),
        "v12" => Some(A64V12),
        "v13" => Some(A64V13),
        "v14" => Some(A64V14),
        "v15" => Some(A64V15),
        "v16" => Some(A64V16),
        "v17" => Some(A64V17),
        "v18" => Some(A64V18),
        "v19" => Some(A64V19),
        "v20" => Some(A64V20),
        "v21" => Some(A64V21),
        "v22" => Some(A64V22),
        "v23" => Some(A64V23),
        "v24" => Some(A64V24),
        "v25" => Some(A64V25),
        "v26" => Some(A64V26),
        "v27" => Some(A64V27),
        "v28" => Some(A64V28),
        "v29" => Some(A64V29),
        "v30" => Some(A64V30),
        "v31" => Some(A64V31),
        // SIMD/FP scalar quad registers (Q0–Q31)
        "q0" => Some(A64Q0),
        "q1" => Some(A64Q1),
        "q2" => Some(A64Q2),
        "q3" => Some(A64Q3),
        "q4" => Some(A64Q4),
        "q5" => Some(A64Q5),
        "q6" => Some(A64Q6),
        "q7" => Some(A64Q7),
        "q8" => Some(A64Q8),
        "q9" => Some(A64Q9),
        "q10" => Some(A64Q10),
        "q11" => Some(A64Q11),
        "q12" => Some(A64Q12),
        "q13" => Some(A64Q13),
        "q14" => Some(A64Q14),
        "q15" => Some(A64Q15),
        "q16" => Some(A64Q16),
        "q17" => Some(A64Q17),
        "q18" => Some(A64Q18),
        "q19" => Some(A64Q19),
        "q20" => Some(A64Q20),
        "q21" => Some(A64Q21),
        "q22" => Some(A64Q22),
        "q23" => Some(A64Q23),
        "q24" => Some(A64Q24),
        "q25" => Some(A64Q25),
        "q26" => Some(A64Q26),
        "q27" => Some(A64Q27),
        "q28" => Some(A64Q28),
        "q29" => Some(A64Q29),
        "q30" => Some(A64Q30),
        "q31" => Some(A64Q31),
        // SIMD/FP scalar double registers (D0–D31)
        "d0" => Some(A64D0),
        "d1" => Some(A64D1),
        "d2" => Some(A64D2),
        "d3" => Some(A64D3),
        "d4" => Some(A64D4),
        "d5" => Some(A64D5),
        "d6" => Some(A64D6),
        "d7" => Some(A64D7),
        "d8" => Some(A64D8),
        "d9" => Some(A64D9),
        "d10" => Some(A64D10),
        "d11" => Some(A64D11),
        "d12" => Some(A64D12),
        "d13" => Some(A64D13),
        "d14" => Some(A64D14),
        "d15" => Some(A64D15),
        "d16" => Some(A64D16),
        "d17" => Some(A64D17),
        "d18" => Some(A64D18),
        "d19" => Some(A64D19),
        "d20" => Some(A64D20),
        "d21" => Some(A64D21),
        "d22" => Some(A64D22),
        "d23" => Some(A64D23),
        "d24" => Some(A64D24),
        "d25" => Some(A64D25),
        "d26" => Some(A64D26),
        "d27" => Some(A64D27),
        "d28" => Some(A64D28),
        "d29" => Some(A64D29),
        "d30" => Some(A64D30),
        "d31" => Some(A64D31),
        // SIMD/FP scalar single registers (S0–S31)
        "s0" => Some(A64S0),
        "s1" => Some(A64S1),
        "s2" => Some(A64S2),
        "s3" => Some(A64S3),
        "s4" => Some(A64S4),
        "s5" => Some(A64S5),
        "s6" => Some(A64S6),
        "s7" => Some(A64S7),
        "s8" => Some(A64S8),
        "s9" => Some(A64S9),
        "s10" => Some(A64S10),
        "s11" => Some(A64S11),
        "s12" => Some(A64S12),
        "s13" => Some(A64S13),
        "s14" => Some(A64S14),
        "s15" => Some(A64S15),
        "s16" => Some(A64S16),
        "s17" => Some(A64S17),
        "s18" => Some(A64S18),
        "s19" => Some(A64S19),
        "s20" => Some(A64S20),
        "s21" => Some(A64S21),
        "s22" => Some(A64S22),
        "s23" => Some(A64S23),
        "s24" => Some(A64S24),
        "s25" => Some(A64S25),
        "s26" => Some(A64S26),
        "s27" => Some(A64S27),
        "s28" => Some(A64S28),
        "s29" => Some(A64S29),
        "s30" => Some(A64S30),
        "s31" => Some(A64S31),
        // SIMD/FP scalar half registers (H0–H31)
        "h0" => Some(A64H0),
        "h1" => Some(A64H1),
        "h2" => Some(A64H2),
        "h3" => Some(A64H3),
        "h4" => Some(A64H4),
        "h5" => Some(A64H5),
        "h6" => Some(A64H6),
        "h7" => Some(A64H7),
        "h8" => Some(A64H8),
        "h9" => Some(A64H9),
        "h10" => Some(A64H10),
        "h11" => Some(A64H11),
        "h12" => Some(A64H12),
        "h13" => Some(A64H13),
        "h14" => Some(A64H14),
        "h15" => Some(A64H15),
        "h16" => Some(A64H16),
        "h17" => Some(A64H17),
        "h18" => Some(A64H18),
        "h19" => Some(A64H19),
        "h20" => Some(A64H20),
        "h21" => Some(A64H21),
        "h22" => Some(A64H22),
        "h23" => Some(A64H23),
        "h24" => Some(A64H24),
        "h25" => Some(A64H25),
        "h26" => Some(A64H26),
        "h27" => Some(A64H27),
        "h28" => Some(A64H28),
        "h29" => Some(A64H29),
        "h30" => Some(A64H30),
        "h31" => Some(A64H31),
        // SIMD/FP scalar byte registers (B0–B31)
        "b0" => Some(A64B0),
        "b1" => Some(A64B1),
        "b2" => Some(A64B2),
        "b3" => Some(A64B3),
        "b4" => Some(A64B4),
        "b5" => Some(A64B5),
        "b6" => Some(A64B6),
        "b7" => Some(A64B7),
        "b8" => Some(A64B8),
        "b9" => Some(A64B9),
        "b10" => Some(A64B10),
        "b11" => Some(A64B11),
        "b12" => Some(A64B12),
        "b13" => Some(A64B13),
        "b14" => Some(A64B14),
        "b15" => Some(A64B15),
        "b16" => Some(A64B16),
        "b17" => Some(A64B17),
        "b18" => Some(A64B18),
        "b19" => Some(A64B19),
        "b20" => Some(A64B20),
        "b21" => Some(A64B21),
        "b22" => Some(A64B22),
        "b23" => Some(A64B23),
        "b24" => Some(A64B24),
        "b25" => Some(A64B25),
        "b26" => Some(A64B26),
        "b27" => Some(A64B27),
        "b28" => Some(A64B28),
        "b29" => Some(A64B29),
        "b30" => Some(A64B30),
        "b31" => Some(A64B31),
        // SVE scalable vector registers (Z0–Z31)
        "z0" => Some(A64Z0),
        "z1" => Some(A64Z1),
        "z2" => Some(A64Z2),
        "z3" => Some(A64Z3),
        "z4" => Some(A64Z4),
        "z5" => Some(A64Z5),
        "z6" => Some(A64Z6),
        "z7" => Some(A64Z7),
        "z8" => Some(A64Z8),
        "z9" => Some(A64Z9),
        "z10" => Some(A64Z10),
        "z11" => Some(A64Z11),
        "z12" => Some(A64Z12),
        "z13" => Some(A64Z13),
        "z14" => Some(A64Z14),
        "z15" => Some(A64Z15),
        "z16" => Some(A64Z16),
        "z17" => Some(A64Z17),
        "z18" => Some(A64Z18),
        "z19" => Some(A64Z19),
        "z20" => Some(A64Z20),
        "z21" => Some(A64Z21),
        "z22" => Some(A64Z22),
        "z23" => Some(A64Z23),
        "z24" => Some(A64Z24),
        "z25" => Some(A64Z25),
        "z26" => Some(A64Z26),
        "z27" => Some(A64Z27),
        "z28" => Some(A64Z28),
        "z29" => Some(A64Z29),
        "z30" => Some(A64Z30),
        "z31" => Some(A64Z31),
        // SVE predicate registers (P0–P15)
        "p0" => Some(A64P0),
        "p1" => Some(A64P1),
        "p2" => Some(A64P2),
        "p3" => Some(A64P3),
        "p4" => Some(A64P4),
        "p5" => Some(A64P5),
        "p6" => Some(A64P6),
        "p7" => Some(A64P7),
        "p8" => Some(A64P8),
        "p9" => Some(A64P9),
        "p10" => Some(A64P10),
        "p11" => Some(A64P11),
        "p12" => Some(A64P12),
        "p13" => Some(A64P13),
        "p14" => Some(A64P14),
        "p15" => Some(A64P15),
        _ => None,
    }
}

/// Parse a RISC-V register name, supporting both hardware names (x0–x31) and
/// ABI names (zero, ra, sp, gp, tp, t0–t6, s0–s11, a0–a7, fp).
fn parse_register_riscv(name: &str) -> Option<Register> {
    use Register::*;
    match name {
        // Hardware names
        "x0" => Some(RvX0),
        "x1" => Some(RvX1),
        "x2" => Some(RvX2),
        "x3" => Some(RvX3),
        "x4" => Some(RvX4),
        "x5" => Some(RvX5),
        "x6" => Some(RvX6),
        "x7" => Some(RvX7),
        "x8" => Some(RvX8),
        "x9" => Some(RvX9),
        "x10" => Some(RvX10),
        "x11" => Some(RvX11),
        "x12" => Some(RvX12),
        "x13" => Some(RvX13),
        "x14" => Some(RvX14),
        "x15" => Some(RvX15),
        "x16" => Some(RvX16),
        "x17" => Some(RvX17),
        "x18" => Some(RvX18),
        "x19" => Some(RvX19),
        "x20" => Some(RvX20),
        "x21" => Some(RvX21),
        "x22" => Some(RvX22),
        "x23" => Some(RvX23),
        "x24" => Some(RvX24),
        "x25" => Some(RvX25),
        "x26" => Some(RvX26),
        "x27" => Some(RvX27),
        "x28" => Some(RvX28),
        "x29" => Some(RvX29),
        "x30" => Some(RvX30),
        "x31" => Some(RvX31),
        // ABI names
        "zero" => Some(RvX0),
        "ra" => Some(RvX1),
        "sp" => Some(RvX2),
        "gp" => Some(RvX3),
        "tp" => Some(RvX4),
        "t0" => Some(RvX5),
        "t1" => Some(RvX6),
        "t2" => Some(RvX7),
        "s0" => Some(RvX8),
        "fp" => Some(RvX8), // fp is alias for s0
        "s1" => Some(RvX9),
        "a0" => Some(RvX10),
        "a1" => Some(RvX11),
        "a2" => Some(RvX12),
        "a3" => Some(RvX13),
        "a4" => Some(RvX14),
        "a5" => Some(RvX15),
        "a6" => Some(RvX16),
        "a7" => Some(RvX17),
        "s2" => Some(RvX18),
        "s3" => Some(RvX19),
        "s4" => Some(RvX20),
        "s5" => Some(RvX21),
        "s6" => Some(RvX22),
        "s7" => Some(RvX23),
        "s8" => Some(RvX24),
        "s9" => Some(RvX25),
        "s10" => Some(RvX26),
        "s11" => Some(RvX27),
        "t3" => Some(RvX28),
        "t4" => Some(RvX29),
        "t5" => Some(RvX30),
        "t6" => Some(RvX31),
        // FP hardware names
        "f0" => Some(RvF0),
        "f1" => Some(RvF1),
        "f2" => Some(RvF2),
        "f3" => Some(RvF3),
        "f4" => Some(RvF4),
        "f5" => Some(RvF5),
        "f6" => Some(RvF6),
        "f7" => Some(RvF7),
        "f8" => Some(RvF8),
        "f9" => Some(RvF9),
        "f10" => Some(RvF10),
        "f11" => Some(RvF11),
        "f12" => Some(RvF12),
        "f13" => Some(RvF13),
        "f14" => Some(RvF14),
        "f15" => Some(RvF15),
        "f16" => Some(RvF16),
        "f17" => Some(RvF17),
        "f18" => Some(RvF18),
        "f19" => Some(RvF19),
        "f20" => Some(RvF20),
        "f21" => Some(RvF21),
        "f22" => Some(RvF22),
        "f23" => Some(RvF23),
        "f24" => Some(RvF24),
        "f25" => Some(RvF25),
        "f26" => Some(RvF26),
        "f27" => Some(RvF27),
        "f28" => Some(RvF28),
        "f29" => Some(RvF29),
        "f30" => Some(RvF30),
        "f31" => Some(RvF31),
        // FP ABI names
        "ft0" => Some(RvF0),
        "ft1" => Some(RvF1),
        "ft2" => Some(RvF2),
        "ft3" => Some(RvF3),
        "ft4" => Some(RvF4),
        "ft5" => Some(RvF5),
        "ft6" => Some(RvF6),
        "ft7" => Some(RvF7),
        "fs0" => Some(RvF8),
        "fs1" => Some(RvF9),
        "fa0" => Some(RvF10),
        "fa1" => Some(RvF11),
        "fa2" => Some(RvF12),
        "fa3" => Some(RvF13),
        "fa4" => Some(RvF14),
        "fa5" => Some(RvF15),
        "fa6" => Some(RvF16),
        "fa7" => Some(RvF17),
        "fs2" => Some(RvF18),
        "fs3" => Some(RvF19),
        "fs4" => Some(RvF20),
        "fs5" => Some(RvF21),
        "fs6" => Some(RvF22),
        "fs7" => Some(RvF23),
        "fs8" => Some(RvF24),
        "fs9" => Some(RvF25),
        "fs10" => Some(RvF26),
        "fs11" => Some(RvF27),
        "ft8" => Some(RvF28),
        "ft9" => Some(RvF29),
        "ft10" => Some(RvF30),
        "ft11" => Some(RvF31),
        // Vector registers (V extension) — hardware names
        "v0" => Some(RvV0),
        "v1" => Some(RvV1),
        "v2" => Some(RvV2),
        "v3" => Some(RvV3),
        "v4" => Some(RvV4),
        "v5" => Some(RvV5),
        "v6" => Some(RvV6),
        "v7" => Some(RvV7),
        "v8" => Some(RvV8),
        "v9" => Some(RvV9),
        "v10" => Some(RvV10),
        "v11" => Some(RvV11),
        "v12" => Some(RvV12),
        "v13" => Some(RvV13),
        "v14" => Some(RvV14),
        "v15" => Some(RvV15),
        "v16" => Some(RvV16),
        "v17" => Some(RvV17),
        "v18" => Some(RvV18),
        "v19" => Some(RvV19),
        "v20" => Some(RvV20),
        "v21" => Some(RvV21),
        "v22" => Some(RvV22),
        "v23" => Some(RvV23),
        "v24" => Some(RvV24),
        "v25" => Some(RvV25),
        "v26" => Some(RvV26),
        "v27" => Some(RvV27),
        "v28" => Some(RvV28),
        "v29" => Some(RvV29),
        "v30" => Some(RvV30),
        "v31" => Some(RvV31),
        _ => None,
    }
}

/// Convenience: parse assembly text directly into statements.
pub fn parse_str(source: &str) -> Result<Vec<Statement>, AsmError> {
    let tokens = crate::lexer::tokenize(source)?;
    parse(&tokens)
}

/// Strip AT&T mnemonic size suffix and return (base_mnemonic, size_hint).
///
/// GAS convention: `movq` → `mov` + Qword, `addl` → `add` + Dword, etc.
/// Also handles AT&T-specific mnemonics that have no Intel equivalent via
/// simple suffix stripping (e.g., `movzbl` → `movzx`, `cltq` → `cdqe`).
fn strip_att_suffix(mnemonic: &str) -> Option<(Mnemonic, OperandSize)> {
    // AT&T-specific mnemonic translations — these have completely different
    // naming conventions from Intel and cannot be derived by suffix stripping.
    let att_translations: &[(&str, &str, OperandSize)] = &[
        // movzx variants: movz{src_size}{dst_size}
        ("movzbl", "movzx", OperandSize::Dword),
        ("movzbw", "movzx", OperandSize::Word),
        ("movzbq", "movzx", OperandSize::Qword),
        ("movzwl", "movzx", OperandSize::Dword),
        ("movzwq", "movzx", OperandSize::Qword),
        // movsx variants: movs{src_size}{dst_size}
        ("movsbl", "movsx", OperandSize::Dword),
        ("movsbw", "movsx", OperandSize::Word),
        ("movsbq", "movsx", OperandSize::Qword),
        ("movswl", "movsx", OperandSize::Dword),
        ("movswq", "movsx", OperandSize::Qword),
        ("movslq", "movsxd", OperandSize::Qword),
        // Sign/zero-extend accumulator
        ("cbtw", "cbw", OperandSize::Word),
        ("cwtl", "cwde", OperandSize::Dword),
        ("cwtd", "cwd", OperandSize::Word),
        ("cltd", "cdq", OperandSize::Dword),
        ("cltq", "cdqe", OperandSize::Qword),
        ("cqto", "cqo", OperandSize::Qword),
    ];

    for &(att, intel, size) in att_translations {
        if mnemonic == att {
            return Some((Mnemonic::from(intel), size));
        }
    }

    if mnemonic.len() < 2 {
        return None;
    }

    // Mnemonics that should NOT be stripped — they end in b/w/l/q naturally
    // and are not size-suffixed GAS mnemonics.
    let no_strip = [
        "call",
        "jmp",
        "ret",
        "nop",
        "hlt",
        "int",
        "syscall",
        "sysenter",
        "sysexit",
        "cpuid",
        "rdtsc",
        "rdtscp",
        "ud2",
        "leave",
        "enter",
        "pushf",
        "popf",
        "pushfq",
        "popfq",
        "lahf",
        "sahf",
        "clc",
        "stc",
        "cmc",
        "cld",
        "std",
        "cli",
        "sti",
        "rep",
        "repe",
        "repne",
        "repz",
        "repnz",
        "lock",
        "pause",
        "mfence",
        "lfence",
        "sfence",
        "endbr64",
        "endbr32",
        "iretq",
        "cdq",
        "cqo",
        "cbw",
        "cwde",
        "cdqe",
        "cwd",
        "xlat",
        "xlatb",
        "swapgs",
        "wrmsr",
        "rdmsr",
        "invd",
        "wbinvd",
        "clts",
        "monitor",
        "mwait",
        "rdrand",
        "rdseed",
        "xtest",
        "xend",
        "vzeroall",
        "vzeroupper",
        "int3",
        // setcc and cmovcc mnemonics — they end in condition codes, not size suffixes
        "setal",
        "setbl",
        "setcl",
        "setgl",
        "setol",
        "setnl",
        "setpl",
        // condition code endings
        "jal",
        "jbl",
        "jcl",
        "jgl",
        "jol",
        "jnl",
        "jpl",
        // string ops — movsb/w/d/q, stosb/w/d/q, lodsb/w/d/q, scasb/w/d/q, cmpsb/w/d/q
        "movsb",
        "movsw",
        "movsd",
        "movsq",
        "stosb",
        "stosw",
        "stosd",
        "stosq",
        "lodsb",
        "lodsw",
        "lodsd",
        "lodsq",
        "scasb",
        "scasw",
        "scasd",
        "scasq",
        "cmpsb",
        "cmpsw",
        "cmpsd",
        "cmpsq",
        "insb",
        "insw",
        "insd",
        "outsb",
        "outsw",
        "outsd",
        // IN/OUT with size are their own mnemonics (inb, outb, etc.)
        "inb",
        "inw",
        "inl",
        "outb",
        "outw",
        "outl",
        // Loop family
        "loop",
        "loope",
        "loopne",
        "loopz",
        "loopnz",
        "jecxz",
        "jrcxz",
        // cmpxchg family
        "cmpxchg8b",
        "cmpxchg16b",
        // bswap, xchg, cmpxchg, xadd
        "bswap",
    ];

    let suffix = mnemonic.as_bytes()[mnemonic.len() - 1];
    let size = match suffix {
        b'b' => OperandSize::Byte,
        b'w' => OperandSize::Word,
        b'l' => OperandSize::Dword,
        b'q' => OperandSize::Qword,
        _ => return None,
    };

    if no_strip.contains(&mnemonic) {
        return None;
    }

    let base = &mnemonic[..mnemonic.len() - 1];

    // Don't strip if the base would be empty
    if base.is_empty() {
        return None;
    }

    // Known base mnemonics that commonly accept size suffixes
    let known_bases = [
        "mov", "add", "sub", "adc", "sbb", "and", "or", "xor", "cmp", "test", "push", "pop", "inc",
        "dec", "neg", "not", "mul", "imul", "div", "idiv", "lea", "xchg", "cmpxchg", "xadd",
        "movzx", "movsx", "movsxd", "shl", "shr", "sar", "rol", "ror", "rcl", "rcr", "bt", "bts",
        "btr", "btc", "bsf", "bsr", "set", "cmov", // + condition code suffix
        "in", "out", "movabs",
    ];

    // Direct match or prefix match for setcc/cmovcc
    if known_bases.contains(&base) {
        return Some((Mnemonic::from(base), size));
    }

    // cmovcc: cmovnel → cmovne (base = cmovne, suffix was 'l')
    // setcc: setnel → setne (won't happen typically, but handle it)
    // Jcc: jnel → jne (won't happen due to no_strip, but be safe)
    if base.starts_with("cmov") || base.starts_with("set") || base.starts_with('j') {
        return Some((Mnemonic::from(base), size));
    }

    // Fallback: strip and return — GAS accepts suffixes on most mnemonics
    Some((Mnemonic::from(base), size))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one(src: &str) -> Statement {
        let stmts = parse_str(src).unwrap();
        assert_eq!(
            stmts.len(),
            1,
            "expected 1 statement, got {}: {:?}",
            stmts.len(),
            stmts
        );
        stmts.into_iter().next().unwrap()
    }

    fn parse_instr(src: &str) -> Instruction {
        match parse_one(src) {
            Statement::Instruction(i) => i,
            s => panic!("expected instruction, got {:?}", s),
        }
    }

    // === Basic Instructions ===

    #[test]
    fn parse_nop() {
        let i = parse_instr("nop");
        assert_eq!(i.mnemonic, "nop");
        assert!(i.operands.is_empty());
    }

    #[test]
    fn parse_ret() {
        let i = parse_instr("ret");
        assert_eq!(i.mnemonic, "ret");
    }

    #[test]
    fn parse_syscall() {
        let i = parse_instr("syscall");
        assert_eq!(i.mnemonic, "syscall");
    }

    // === Register-Register ===

    #[test]
    fn parse_mov_reg_reg() {
        let i = parse_instr("mov rax, rbx");
        assert_eq!(i.mnemonic, "mov");
        assert_eq!(i.operands.len(), 2);
        assert_eq!(i.operands[0], Operand::Register(Register::Rax));
        assert_eq!(i.operands[1], Operand::Register(Register::Rbx));
    }

    #[test]
    fn parse_add_r32() {
        let i = parse_instr("add eax, ecx");
        assert_eq!(i.mnemonic, "add");
        assert_eq!(i.operands[0], Operand::Register(Register::Eax));
        assert_eq!(i.operands[1], Operand::Register(Register::Ecx));
    }

    #[test]
    fn parse_xor_r8() {
        let i = parse_instr("xor al, bl");
        assert_eq!(i.operands[0], Operand::Register(Register::Al));
        assert_eq!(i.operands[1], Operand::Register(Register::Bl));
    }

    // === Register-Immediate ===

    #[test]
    fn parse_mov_reg_imm() {
        let i = parse_instr("mov rax, 42");
        assert_eq!(i.operands[0], Operand::Register(Register::Rax));
        assert_eq!(i.operands[1], Operand::Immediate(42));
    }

    #[test]
    fn parse_mov_reg_hex() {
        let i = parse_instr("mov rdi, 0xDEAD");
        assert_eq!(i.operands[1], Operand::Immediate(0xDEAD));
    }

    #[test]
    fn parse_add_imm_negative() {
        let i = parse_instr("add rsp, -8");
        assert_eq!(i.operands[1], Operand::Immediate(-8));
    }

    #[test]
    fn parse_char_immediate() {
        let i = parse_instr("mov al, 'A'");
        assert_eq!(i.operands[1], Operand::Immediate(65));
    }

    // === Memory Operands ===

    #[test]
    fn parse_mem_base() {
        let i = parse_instr("mov rax, [rbx]");
        assert_eq!(i.operands[0], Operand::Register(Register::Rax));
        match &i.operands[1] {
            Operand::Memory(m) => {
                assert_eq!(m.base, Some(Register::Rbx));
                assert_eq!(m.index, None);
                assert_eq!(m.disp, 0);
            }
            _ => panic!("expected memory operand"),
        }
    }

    #[test]
    fn parse_mem_base_disp() {
        let i = parse_instr("mov rax, [rbp + 8]");
        match &i.operands[1] {
            Operand::Memory(m) => {
                assert_eq!(m.base, Some(Register::Rbp));
                assert_eq!(m.disp, 8);
            }
            _ => panic!("expected memory operand"),
        }
    }

    #[test]
    fn parse_mem_base_neg_disp() {
        let i = parse_instr("mov rax, [rbp - 0x10]");
        match &i.operands[1] {
            Operand::Memory(m) => {
                assert_eq!(m.base, Some(Register::Rbp));
                assert_eq!(m.disp, -16);
            }
            _ => panic!("expected memory operand"),
        }
    }

    #[test]
    fn parse_mem_base_index() {
        let i = parse_instr("mov rax, [rbx + rcx]");
        match &i.operands[1] {
            Operand::Memory(m) => {
                assert_eq!(m.base, Some(Register::Rbx));
                assert_eq!(m.index, Some(Register::Rcx));
                assert_eq!(m.scale, 1);
            }
            _ => panic!("expected memory operand"),
        }
    }

    #[test]
    fn parse_mem_base_index_scale() {
        let i = parse_instr("lea rax, [rbx + rcx*8]");
        match &i.operands[1] {
            Operand::Memory(m) => {
                assert_eq!(m.base, Some(Register::Rbx));
                assert_eq!(m.index, Some(Register::Rcx));
                assert_eq!(m.scale, 8);
            }
            _ => panic!("expected memory operand"),
        }
    }

    #[test]
    fn parse_mem_full() {
        let i = parse_instr("mov rax, [rbx + rcx*4 + 16]");
        match &i.operands[1] {
            Operand::Memory(m) => {
                assert_eq!(m.base, Some(Register::Rbx));
                assert_eq!(m.index, Some(Register::Rcx));
                assert_eq!(m.scale, 4);
                assert_eq!(m.disp, 16);
            }
            _ => panic!("expected memory operand"),
        }
    }

    #[test]
    fn parse_mem_disp_only() {
        let i = parse_instr("mov rax, [0x1000]");
        match &i.operands[1] {
            Operand::Memory(m) => {
                assert_eq!(m.base, None);
                assert_eq!(m.disp, 0x1000);
            }
            _ => panic!("expected memory operand"),
        }
    }

    // === Size Hints ===

    #[test]
    fn parse_byte_ptr() {
        let i = parse_instr("mov byte ptr [rax], 0");
        assert_eq!(i.size_hint, Some(OperandSize::Byte));
        match &i.operands[0] {
            Operand::Memory(m) => assert_eq!(m.base, Some(Register::Rax)),
            _ => panic!("expected memory operand"),
        }
    }

    #[test]
    fn parse_qword_no_ptr() {
        let i = parse_instr("mov qword [rax], 0");
        assert_eq!(i.size_hint, Some(OperandSize::Qword));
    }

    #[test]
    fn parse_dword_ptr() {
        let i = parse_instr("add dword ptr [rbp - 4], 1");
        assert_eq!(i.size_hint, Some(OperandSize::Dword));
    }

    // === Labels ===

    #[test]
    fn parse_label_def() {
        let stmt = parse_one("start:");
        match stmt {
            Statement::Label(name, _) => assert_eq!(name, "start"),
            _ => panic!("expected label"),
        }
    }

    #[test]
    fn parse_label_ref() {
        let i = parse_instr("jmp loop");
        assert_eq!(i.operands[0], Operand::Label(String::from("loop")));
    }

    #[test]
    fn parse_call_label() {
        let i = parse_instr("call printf");
        assert_eq!(i.operands[0], Operand::Label(String::from("printf")));
    }

    #[test]
    fn parse_label_with_offset() {
        let i = parse_instr("lea rax, data + 4");
        match &i.operands[1] {
            Operand::Expression(Expr::Add(l, r)) => {
                assert_eq!(**l, Expr::Label(String::from("data")));
                assert_eq!(**r, Expr::Num(4));
            }
            _ => panic!("expected expression operand"),
        }
    }

    // === Prefixes ===

    #[test]
    fn parse_lock_prefix() {
        let i = parse_instr("lock add [rax], 1");
        assert_eq!(i.prefixes, vec![Prefix::Lock]);
        assert_eq!(i.mnemonic, "add");
    }

    #[test]
    fn parse_rep_prefix() {
        let i = parse_instr("rep movsb");
        assert_eq!(i.prefixes, vec![Prefix::Rep]);
        assert_eq!(i.mnemonic, "movsb");
    }

    // === Directives ===

    #[test]
    fn parse_byte_directive() {
        let stmt = parse_one(".byte 0x90, 0xCC");
        match stmt {
            Statement::Data(d) => {
                assert_eq!(d.size, DataSize::Byte);
                assert_eq!(
                    d.values,
                    vec![DataValue::Integer(0x90), DataValue::Integer(0xCC)]
                );
            }
            _ => panic!("expected data"),
        }
    }

    #[test]
    fn parse_word_directive() {
        let stmt = parse_one(".word 0x1234");
        match stmt {
            Statement::Data(d) => {
                assert_eq!(d.size, DataSize::Word);
                assert_eq!(d.values, vec![DataValue::Integer(0x1234)]);
            }
            _ => panic!("expected data"),
        }
    }

    #[test]
    fn parse_ascii_directive() {
        let stmt = parse_one(".ascii \"hello\"");
        match stmt {
            Statement::Data(d) => {
                assert_eq!(d.size, DataSize::Byte);
                assert_eq!(d.values, vec![DataValue::Bytes(b"hello".to_vec())]);
            }
            _ => panic!("expected data"),
        }
    }

    #[test]
    fn parse_asciz_null_terminates() {
        let stmt = parse_one(".asciz \"ok\"");
        match stmt {
            Statement::Data(d) => {
                assert_eq!(d.values, vec![DataValue::Bytes(b"ok\0".to_vec())]);
            }
            _ => panic!("expected data"),
        }
    }

    #[test]
    fn parse_equ_directive() {
        let stmt = parse_one(".equ SYS_WRITE, 1");
        match stmt {
            Statement::Const(c) => {
                assert_eq!(c.name, "SYS_WRITE");
                assert_eq!(c.value, 1);
            }
            _ => panic!("expected const"),
        }
    }

    #[test]
    fn parse_align_directive() {
        let stmt = parse_one(".align 16");
        match stmt {
            Statement::Align(a) => {
                assert_eq!(a.alignment, 16);
                assert_eq!(a.fill, None);
            }
            _ => panic!("expected align"),
        }
    }

    #[test]
    fn parse_p2align_directive() {
        let stmt = parse_one(".p2align 4");
        match stmt {
            Statement::Align(a) => {
                assert_eq!(a.alignment, 16); // 2^4 = 16
            }
            _ => panic!("expected align"),
        }
    }

    #[test]
    fn parse_fill_directive() {
        let stmt = parse_one(".fill 10, 1, 0x90");
        match stmt {
            Statement::Fill(f) => {
                assert_eq!(f.count, 10);
                assert_eq!(f.size, 1);
                assert_eq!(f.value, 0x90);
            }
            _ => panic!("expected fill"),
        }
    }

    #[test]
    fn parse_space_directive() {
        let stmt = parse_one(".space 64");
        match stmt {
            Statement::Space(s) => {
                assert_eq!(s.size, 64);
                assert_eq!(s.fill, 0);
            }
            _ => panic!("expected space"),
        }
    }

    #[test]
    fn parse_org_directive() {
        let stmt = parse_one(".org 0x1000");
        match stmt {
            Statement::Org(o) => {
                assert_eq!(o.offset, 0x1000);
                assert_eq!(o.fill, 0x00);
            }
            _ => panic!("expected org"),
        }
    }

    #[test]
    fn parse_org_with_fill() {
        let stmt = parse_one(".org 0x100, 0xFF");
        match stmt {
            Statement::Org(o) => {
                assert_eq!(o.offset, 0x100);
                assert_eq!(o.fill, 0xFF);
            }
            _ => panic!("expected org"),
        }
    }

    // === Multi-statement ===

    #[test]
    fn parse_multi_line() {
        let stmts = parse_str("nop\nret").unwrap();
        assert_eq!(stmts.len(), 2);
        match (&stmts[0], &stmts[1]) {
            (Statement::Instruction(i1), Statement::Instruction(i2)) => {
                assert_eq!(i1.mnemonic, "nop");
                assert_eq!(i2.mnemonic, "ret");
            }
            _ => panic!("expected two instructions"),
        }
    }

    #[test]
    fn parse_label_and_instruction() {
        let stmts = parse_str("start:\n  mov rax, 1").unwrap();
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Statement::Label(name, _) if name == "start"));
        assert!(matches!(&stmts[1], Statement::Instruction(_)));
    }

    #[test]
    fn parse_semicolon_separated() {
        let stmts = parse_str("nop; ret").unwrap();
        assert_eq!(stmts.len(), 2);
    }

    // === Case Insensitivity ===

    #[test]
    fn case_insensitive_mnemonic() {
        let i = parse_instr("MOV RAX, RBX");
        assert_eq!(i.mnemonic, "mov");
        assert_eq!(i.operands[0], Operand::Register(Register::Rax));
    }

    #[test]
    fn case_insensitive_register() {
        let i = parse_instr("xor EAX, eax");
        assert_eq!(i.operands[0], Operand::Register(Register::Eax));
        assert_eq!(i.operands[1], Operand::Register(Register::Eax));
    }

    // === Extended Registers ===

    #[test]
    fn parse_extended_reg() {
        let i = parse_instr("mov r8, r15");
        assert_eq!(i.operands[0], Operand::Register(Register::R8));
        assert_eq!(i.operands[1], Operand::Register(Register::R15));
    }

    #[test]
    fn parse_extended_reg_dword() {
        let i = parse_instr("mov r8d, r15d");
        assert_eq!(i.operands[0], Operand::Register(Register::R8d));
        assert_eq!(i.operands[1], Operand::Register(Register::R15d));
    }

    // === Edge Cases ===

    #[test]
    fn parse_push_pop() {
        let i = parse_instr("push rbp");
        assert_eq!(i.mnemonic, "push");
        assert_eq!(i.operands[0], Operand::Register(Register::Rbp));
    }

    #[test]
    fn parse_lea() {
        let i = parse_instr("lea rdi, [rip + 0x10]");
        assert_eq!(i.mnemonic, "lea");
        match &i.operands[1] {
            Operand::Memory(m) => {
                assert_eq!(m.base, Some(Register::Rip));
                assert_eq!(m.disp, 0x10);
            }
            _ => panic!("expected memory"),
        }
    }

    #[test]
    fn parse_three_operand_imul() {
        let i = parse_instr("imul rax, rbx, 10");
        assert_eq!(i.operands.len(), 3);
        assert_eq!(i.operands[0], Operand::Register(Register::Rax));
        assert_eq!(i.operands[1], Operand::Register(Register::Rbx));
        assert_eq!(i.operands[2], Operand::Immediate(10));
    }

    #[test]
    fn global_directive_ignored() {
        let stmts = parse_str(".global main\nmov rax, 1").unwrap();
        // .global is ignored, should get 1 instruction
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn section_directive_ignored() {
        let stmts = parse_str(".section .text\nnop").unwrap();
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn empty_input() {
        let stmts = parse_str("").unwrap();
        assert!(stmts.is_empty());
    }

    #[test]
    fn only_labels() {
        let stmts = parse_str("start:\nend:").unwrap();
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Statement::Label(n, _) if n == "start"));
        assert!(matches!(&stmts[1], Statement::Label(n, _) if n == "end"));
    }

    #[test]
    fn mem_with_label() {
        let i = parse_instr("mov rax, [msg]");
        match &i.operands[1] {
            Operand::Memory(m) => {
                assert_eq!(m.base, None);
                assert_eq!(m.disp_label, Some(String::from("msg")));
            }
            _ => panic!("expected memory operand with label"),
        }
    }

    #[test]
    fn xmm_registers() {
        let i = parse_instr("movaps xmm0, xmm1");
        assert_eq!(i.operands[0], Operand::Register(Register::Xmm0));
        assert_eq!(i.operands[1], Operand::Register(Register::Xmm1));
    }

    #[test]
    fn segment_override_mem() {
        let i = parse_instr("mov rax, fs:[0x28]");
        match &i.operands[1] {
            Operand::Memory(m) => {
                assert_eq!(m.segment, Some(Register::Fs));
                assert_eq!(m.disp, 0x28);
            }
            _ => panic!("expected segment memory operand"),
        }
    }

    // === name = expression syntax ===

    #[test]
    fn parse_name_equals_constant() {
        let stmt = parse_one("EXIT = 60");
        match stmt {
            Statement::Const(c) => {
                assert_eq!(c.name, "EXIT");
                assert_eq!(c.value, 60);
            }
            _ => panic!("expected const, got {:?}", stmt),
        }
    }

    #[test]
    fn parse_name_equals_hex() {
        let stmt = parse_one("MAGIC = 0xDEAD");
        match stmt {
            Statement::Const(c) => {
                assert_eq!(c.name, "MAGIC");
                assert_eq!(c.value, 0xDEAD);
            }
            _ => panic!("expected const"),
        }
    }

    #[test]
    fn parse_name_equals_negative() {
        let stmt = parse_one("NEG = -1");
        match stmt {
            Statement::Const(c) => {
                assert_eq!(c.name, "NEG");
                assert_eq!(c.value, -1);
            }
            _ => panic!("expected const"),
        }
    }

    #[test]
    fn parse_set_directive() {
        let stmt = parse_one(".set COUNT, 42");
        match stmt {
            Statement::Const(c) => {
                assert_eq!(c.name, "COUNT");
                assert_eq!(c.value, 42);
            }
            _ => panic!("expected const"),
        }
    }

    #[test]
    fn name_equals_used_in_program() {
        let stmts = parse_str("EXIT = 60\nmov eax, EXIT").unwrap();
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Statement::Const(_)));
        assert!(matches!(&stmts[1], Statement::Instruction(_)));
    }

    #[test]
    fn parse_const_expr_with_identifier() {
        // .equ SIZE, 10 then .fill SIZE, 1, 0
        let stmts = parse_str("SIZE = 10\n.fill SIZE, 1, 0").unwrap();
        assert_eq!(stmts.len(), 2);
        match &stmts[1] {
            Statement::Fill(f) => assert_eq!(f.count, 10),
            _ => panic!("expected Fill"),
        }
    }

    #[test]
    fn parse_const_chain() {
        // Constants referencing earlier constants
        let stmts = parse_str("A = 5\nB = A + 3\n.space B, 0").unwrap();
        assert_eq!(stmts.len(), 3);
        match &stmts[2] {
            Statement::Space(s) => assert_eq!(s.size, 8),
            _ => panic!("expected Space"),
        }
    }

    #[test]
    fn parse_equ_identifier_in_const_expr() {
        let stmts = parse_str(".equ BASE, 100\n.equ TOTAL, BASE + 50").unwrap();
        match &stmts[1] {
            Statement::Const(c) => assert_eq!(c.value, 150),
            _ => panic!("expected Const"),
        }
    }

    #[test]
    fn parse_label_plus_identifier_expression() {
        // When OFFSET is a constant, label+OFFSET should produce Expression
        // that gets partially resolved at parse time
        let stmts = parse_str("OFF = 8\nmov rax, data + OFF").unwrap();
        match &stmts[1] {
            Statement::Instruction(i) => {
                match &i.operands[1] {
                    // The parser resolves OFF→8, so this becomes Expression(label+8)
                    Operand::Expression(Expr::Add(l, r)) => {
                        assert_eq!(**l, Expr::Label(String::from("data")));
                        assert_eq!(**r, Expr::Num(8));
                    }
                    other => panic!("expected Expression, got {:?}", other),
                }
            }
            _ => panic!("expected Instruction"),
        }
    }

    #[test]
    fn parse_all_constants_resolve_to_immediate() {
        // When all parts are constants, expression should collapse to Immediate
        let stmts = parse_str("BASE = 100\nOFF = 8\nmov eax, BASE + OFF").unwrap();
        match &stmts[2] {
            Statement::Instruction(i) => {
                assert_eq!(i.operands[1], Operand::Immediate(108));
            }
            _ => panic!("expected Instruction"),
        }
    }

    #[test]
    fn parse_align_with_constant() {
        let stmts = parse_str("ALIGN_VAL = 8\n.align ALIGN_VAL").unwrap();
        match &stmts[1] {
            Statement::Align(a) => assert_eq!(a.alignment, 8),
            _ => panic!("expected Align"),
        }
    }

    #[test]
    fn parse_label_minus_offset() {
        let i = parse_instr("jmp target - 8");
        match &i.operands[0] {
            Operand::Expression(Expr::Sub(l, r)) => {
                assert_eq!(**l, Expr::Label(String::from("target")));
                assert_eq!(**r, Expr::Num(8));
            }
            _ => panic!("expected Sub expression"),
        }
    }

    #[test]
    fn parse_const_negation_precedence() {
        // -A + B should evaluate as (-A) + B, not -(A + B)
        let stmts = parse_str("A = 10\nB = 3\nX = -A + B\nmov eax, X").unwrap();
        match &stmts[3] {
            Statement::Instruction(i) => {
                assert_eq!(
                    i.operands[1],
                    Operand::Immediate(-7),
                    "-10 + 3 should be -7, not -13"
                );
            }
            _ => panic!("expected Instruction"),
        }
    }

    #[test]
    fn parse_const_negation_only() {
        // -A should evaluate as -10
        let stmts = parse_str("A = 10\nX = -A\nmov eax, X").unwrap();
        match &stmts[2] {
            Statement::Instruction(i) => {
                assert_eq!(i.operands[1], Operand::Immediate(-10));
            }
            _ => panic!("expected Instruction"),
        }
    }

    #[test]
    fn parse_const_negation_sub_chain() {
        // -A - B should be (-A) - B = -10 - 3 = -13
        let stmts = parse_str("A = 10\nB = 3\nX = -A - B\nmov eax, X").unwrap();
        match &stmts[3] {
            Statement::Instruction(i) => {
                assert_eq!(
                    i.operands[1],
                    Operand::Immediate(-13),
                    "-10 - 3 should be -13"
                );
            }
            _ => panic!("expected Instruction"),
        }
    }

    #[test]
    fn parse_align_rejects_non_power_of_2() {
        let result = crate::parser::parse_str(".align 3");
        assert!(result.is_err(), ".align 3 should be rejected");
        let err = result.unwrap_err();
        let msg = alloc::format!("{err:?}");
        assert!(
            msg.contains("power of 2"),
            "error should mention power of 2, got: {msg}"
        );
    }

    #[test]
    fn parse_align_accepts_power_of_2() {
        // These should all succeed
        for val in &["1", "2", "4", "8", "16", "32", "64", "4096"] {
            let src = alloc::format!(".align {val}");
            let result = crate::parser::parse_str(&src);
            assert!(
                result.is_ok(),
                ".align {val} should be accepted, got: {result:?}"
            );
        }
    }

    // === 8th Audit: RSP/ESP as SIB index rejection ===

    #[test]
    fn parse_rsp_as_index_rejects() {
        // [rbx + rsp*2] should fail — RSP cannot be SIB index
        let result = crate::parser::parse_str("mov rax, [rbx + rsp*2]");
        assert!(result.is_err(), "RSP as SIB index should be rejected");
    }

    #[test]
    fn parse_esp_as_index_rejects() {
        // [ebx + esp*1] should fail — ESP cannot be SIB index
        let result = crate::parser::parse_str("mov eax, [ebx + esp*1]");
        assert!(result.is_err(), "ESP as SIB index should be rejected");
    }

    #[test]
    fn parse_r12_as_index_accepts() {
        // [rbx + r12*2] should succeed — R12 IS valid as SIB index (uses REX.X)
        let result = crate::parser::parse_str("mov rax, [rbx + r12*2]");
        assert!(
            result.is_ok(),
            "R12 as SIB index should be accepted, got: {result:?}"
        );
    }

    // === AT&T / GAS Syntax ===

    fn parse_att(src: &str) -> Vec<Statement> {
        let tokens = crate::lexer::tokenize(src).unwrap();
        parse_with_syntax(&tokens, Arch::X86_64, Syntax::Att).unwrap()
    }

    fn parse_att_instr(src: &str) -> Instruction {
        let stmts = parse_att(src);
        assert_eq!(stmts.len(), 1, "expected 1 statement, got {:?}", stmts);
        match stmts.into_iter().next().unwrap() {
            Statement::Instruction(i) => i,
            s => panic!("expected instruction, got {s:?}"),
        }
    }

    #[test]
    fn att_register_operand() {
        let i = parse_att_instr("nop");
        assert_eq!(i.mnemonic, "nop");
        assert!(i.operands.is_empty());
    }

    #[test]
    fn att_mov_imm_to_reg() {
        let i = parse_att_instr("movq $42, %rax");
        assert_eq!(i.mnemonic, "mov");
        assert_eq!(i.size_hint, Some(OperandSize::Qword));
        // Operands reversed: AT&T src,dst → Intel dst,src
        assert_eq!(i.operands.len(), 2);
        assert_eq!(i.operands[0], Operand::Register(Register::Rax));
        assert_eq!(i.operands[1], Operand::Immediate(42));
    }

    #[test]
    fn att_mov_reg_to_reg() {
        let i = parse_att_instr("movl %eax, %ecx");
        assert_eq!(i.mnemonic, "mov");
        assert_eq!(i.size_hint, Some(OperandSize::Dword));
        assert_eq!(i.operands[0], Operand::Register(Register::Ecx));
        assert_eq!(i.operands[1], Operand::Register(Register::Eax));
    }

    #[test]
    fn att_add_imm_to_reg() {
        let i = parse_att_instr("addl $0x10, %eax");
        assert_eq!(i.mnemonic, "add");
        assert_eq!(i.size_hint, Some(OperandSize::Dword));
        assert_eq!(i.operands[0], Operand::Register(Register::Eax));
        assert_eq!(i.operands[1], Operand::Immediate(0x10));
    }

    #[test]
    fn att_negative_immediate() {
        let i = parse_att_instr("addq $-1, %rax");
        assert_eq!(i.mnemonic, "add");
        assert_eq!(i.operands[1], Operand::Immediate(-1));
    }

    #[test]
    fn att_byte_suffix() {
        let i = parse_att_instr("movb $0x41, %al");
        assert_eq!(i.mnemonic, "mov");
        assert_eq!(i.size_hint, Some(OperandSize::Byte));
        assert_eq!(i.operands[0], Operand::Register(Register::Al));
    }

    #[test]
    fn att_word_suffix() {
        let i = parse_att_instr("movw $0x1234, %ax");
        assert_eq!(i.mnemonic, "mov");
        assert_eq!(i.size_hint, Some(OperandSize::Word));
        assert_eq!(i.operands[0], Operand::Register(Register::Ax));
    }

    #[test]
    fn att_memory_base_only() {
        let i = parse_att_instr("movq (%rax), %rbx");
        assert_eq!(i.mnemonic, "mov");
        // After reversal: [0]=rbx (dst), [1]=(%rax) (src)
        assert_eq!(i.operands[0], Operand::Register(Register::Rbx));
        if let Operand::Memory(m) = &i.operands[1] {
            assert_eq!(m.base, Some(Register::Rax));
            assert_eq!(m.disp, 0);
            assert!(m.index.is_none());
        } else {
            panic!("expected memory operand");
        }
    }

    #[test]
    fn att_memory_disp_base() {
        let i = parse_att_instr("movl 8(%rsp), %eax");
        assert_eq!(i.mnemonic, "mov");
        if let Operand::Memory(m) = &i.operands[1] {
            assert_eq!(m.base, Some(Register::Rsp));
            assert_eq!(m.disp, 8);
        } else {
            panic!("expected memory operand");
        }
    }

    #[test]
    fn att_memory_negative_disp() {
        let i = parse_att_instr("movq -16(%rbp), %rax");
        if let Operand::Memory(m) = &i.operands[1] {
            assert_eq!(m.base, Some(Register::Rbp));
            assert_eq!(m.disp, -16);
        } else {
            panic!("expected memory operand");
        }
    }

    #[test]
    fn att_memory_base_index() {
        let i = parse_att_instr("movl (%rax, %rcx), %edx");
        if let Operand::Memory(m) = &i.operands[1] {
            assert_eq!(m.base, Some(Register::Rax));
            assert_eq!(m.index, Some(Register::Rcx));
            assert_eq!(m.scale, 1);
        } else {
            panic!("expected memory operand");
        }
    }

    #[test]
    fn att_memory_base_index_scale() {
        let i = parse_att_instr("movq (%rax, %rcx, 4), %rdx");
        if let Operand::Memory(m) = &i.operands[1] {
            assert_eq!(m.base, Some(Register::Rax));
            assert_eq!(m.index, Some(Register::Rcx));
            assert_eq!(m.scale, 4);
            assert_eq!(m.disp, 0);
        } else {
            panic!("expected memory operand");
        }
    }

    #[test]
    fn att_memory_disp_base_index_scale() {
        let i = parse_att_instr("movl 16(%rbx, %rsi, 8), %eax");
        if let Operand::Memory(m) = &i.operands[1] {
            assert_eq!(m.base, Some(Register::Rbx));
            assert_eq!(m.index, Some(Register::Rsi));
            assert_eq!(m.scale, 8);
            assert_eq!(m.disp, 16);
        } else {
            panic!("expected memory operand");
        }
    }

    #[test]
    fn att_segment_override() {
        let i = parse_att_instr("movq %fs:0x28(%rax), %rbx");
        if let Operand::Memory(m) = &i.operands[1] {
            assert_eq!(m.segment, Some(Register::Fs));
            assert_eq!(m.base, Some(Register::Rax));
            assert_eq!(m.disp, 0x28);
        } else {
            panic!("expected memory operand");
        }
    }

    #[test]
    fn att_push_pop() {
        let i = parse_att_instr("pushq %rbp");
        assert_eq!(i.mnemonic, "push");
        assert_eq!(i.operands[0], Operand::Register(Register::Rbp));

        let i2 = parse_att_instr("popq %rbp");
        assert_eq!(i2.mnemonic, "pop");
        assert_eq!(i2.operands[0], Operand::Register(Register::Rbp));
    }

    #[test]
    fn att_xor_reg_reg() {
        let i = parse_att_instr("xorl %eax, %eax");
        assert_eq!(i.mnemonic, "xor");
        // After reversal: both operands are eax
        assert_eq!(i.operands[0], Operand::Register(Register::Eax));
        assert_eq!(i.operands[1], Operand::Register(Register::Eax));
    }

    #[test]
    fn att_call_label() {
        let i = parse_att_instr("call func");
        assert_eq!(i.mnemonic, "call");
        assert_eq!(i.operands[0], Operand::Label(String::from("func")));
    }

    #[test]
    fn att_jmp_label() {
        let i = parse_att_instr("jmp done");
        assert_eq!(i.mnemonic, "jmp");
        assert_eq!(i.operands[0], Operand::Label(String::from("done")));
    }

    #[test]
    fn att_jcc_label() {
        let i = parse_att_instr("jne loop");
        assert_eq!(i.mnemonic, "jne");
        assert_eq!(i.operands[0], Operand::Label(String::from("loop")));
    }

    #[test]
    fn att_ret() {
        let i = parse_att_instr("ret");
        assert_eq!(i.mnemonic, "ret");
        assert!(i.operands.is_empty());
    }

    #[test]
    fn att_syscall() {
        let i = parse_att_instr("syscall");
        assert_eq!(i.mnemonic, "syscall");
    }

    #[test]
    fn att_lock_prefix() {
        let i = parse_att_instr("lock xchgl %eax, (%rbx)");
        assert_eq!(i.mnemonic, "xchg");
        assert!(i.prefixes.contains(&Prefix::Lock));
    }

    #[test]
    fn att_lea() {
        let i = parse_att_instr("leaq 8(%rsp), %rax");
        assert_eq!(i.mnemonic, "lea");
        // After reversal: [0]=rax, [1]=8(%rsp)
        assert_eq!(i.operands[0], Operand::Register(Register::Rax));
        if let Operand::Memory(m) = &i.operands[1] {
            assert_eq!(m.base, Some(Register::Rsp));
            assert_eq!(m.disp, 8);
        } else {
            panic!("expected memory operand");
        }
    }

    #[test]
    fn att_imm_label_ref() {
        let i = parse_att_instr("movq $myvar, %rax");
        assert_eq!(i.mnemonic, "mov");
        assert_eq!(i.operands[1], Operand::Label(String::from("myvar")));
    }

    #[test]
    fn att_no_suffix_no_size_hint() {
        // No suffix → no size_hint
        let i = parse_att_instr("nop");
        assert!(i.size_hint.is_none());
    }

    #[test]
    fn att_int_not_stripped() {
        // "int" should not have suffix stripped to "in" + Dword
        let i = parse_att_instr("int $0x80");
        assert_eq!(i.mnemonic, "int");
        assert_eq!(i.operands[0], Operand::Immediate(0x80));
    }

    #[test]
    fn att_string_ops_not_stripped() {
        let i = parse_att_instr("movsb");
        assert_eq!(i.mnemonic, "movsb");
        let i = parse_att_instr("stosq");
        assert_eq!(i.mnemonic, "stosq");
    }

    #[test]
    fn att_rep_prefix() {
        let i = parse_att_instr("rep movsb");
        assert_eq!(i.mnemonic, "movsb");
        assert!(i.prefixes.contains(&Prefix::Rep));
    }

    #[test]
    fn att_cmp_operand_order() {
        // AT&T: cmpl $0, %eax → Intel: cmp eax, 0
        let i = parse_att_instr("cmpl $0, %eax");
        assert_eq!(i.mnemonic, "cmp");
        assert_eq!(i.operands[0], Operand::Register(Register::Eax));
        assert_eq!(i.operands[1], Operand::Immediate(0));
    }

    #[test]
    fn att_test_operand_order() {
        // AT&T: testl %eax, %eax → Intel: test eax, eax
        let i = parse_att_instr("testl %eax, %eax");
        assert_eq!(i.mnemonic, "test");
        assert_eq!(i.operands[0], Operand::Register(Register::Eax));
        assert_eq!(i.operands[1], Operand::Register(Register::Eax));
    }

    #[test]
    fn att_sub_mem_to_reg() {
        let i = parse_att_instr("subq 8(%rbp), %rax");
        assert_eq!(i.mnemonic, "sub");
        assert_eq!(i.operands[0], Operand::Register(Register::Rax));
        if let Operand::Memory(m) = &i.operands[1] {
            assert_eq!(m.base, Some(Register::Rbp));
            assert_eq!(m.disp, 8);
        } else {
            panic!("expected memory operand");
        }
    }

    #[test]
    fn att_push_immediate() {
        let i = parse_att_instr("pushq $42");
        assert_eq!(i.mnemonic, "push");
        assert_eq!(i.operands[0], Operand::Immediate(42));
    }

    #[test]
    fn att_numeric_label_fwd() {
        let i = parse_att_instr("jmp 1f");
        assert_eq!(i.mnemonic, "jmp");
        assert_eq!(i.operands[0], Operand::Label(String::from("1f")));
    }

    #[test]
    fn att_numeric_label_bwd() {
        let i = parse_att_instr("jne 1b");
        assert_eq!(i.mnemonic, "jne");
        assert_eq!(i.operands[0], Operand::Label(String::from("1b")));
    }

    #[test]
    fn att_syntax_directive_switches_mode() {
        let src = ".syntax att\nmovq $1, %rax";
        // Start with Intel, switch to AT&T via directive
        let tokens = crate::lexer::tokenize(src).unwrap();
        let stmts = parse_with_syntax(&tokens, Arch::X86_64, Syntax::Intel).unwrap();
        // Should have parsed the mov in AT&T mode after .syntax att
        let instr = stmts
            .iter()
            .find_map(|s| {
                if let Statement::Instruction(i) = s {
                    Some(i)
                } else {
                    None
                }
            })
            .expect("no instruction found");
        assert_eq!(instr.mnemonic, "mov");
        // Operand order should be reversed (AT&T)
        assert_eq!(instr.operands[0], Operand::Register(Register::Rax));
        assert_eq!(instr.operands[1], Operand::Immediate(1));
    }

    #[test]
    fn att_star_indirect_reg() {
        let i = parse_att_instr("jmp *%rax");
        assert_eq!(i.mnemonic, "jmp");
        assert_eq!(i.operands[0], Operand::Register(Register::Rax));
    }

    #[test]
    fn att_star_indirect_mem() {
        let i = parse_att_instr("call *(%rax)");
        assert_eq!(i.mnemonic, "call");
        if let Operand::Memory(m) = &i.operands[0] {
            assert_eq!(m.base, Some(Register::Rax));
        } else {
            panic!("expected memory operand");
        }
    }

    // ── Literal pool parsing ────────────────────────────────

    fn parse_aarch64(src: &str) -> Vec<Statement> {
        let tokens = crate::lexer::tokenize(src).unwrap();
        parse_with_syntax(&tokens, Arch::Aarch64, Syntax::Ual).unwrap()
    }

    #[test]
    fn parse_ldr_literal_pool_x_reg() {
        let stmts = parse_aarch64("ldr x0, =0x12345678");
        assert_eq!(stmts.len(), 1);
        if let Statement::Instruction(instr) = &stmts[0] {
            assert_eq!(instr.mnemonic, "ldr");
            assert_eq!(instr.operands.len(), 2);
            assert!(matches!(
                &instr.operands[0],
                Operand::Register(Register::A64X0)
            ));
            assert!(matches!(
                &instr.operands[1],
                Operand::LiteralPoolValue(0x12345678)
            ));
        } else {
            panic!("expected instruction");
        }
    }

    #[test]
    fn parse_ldr_literal_pool_w_reg() {
        let stmts = parse_aarch64("ldr w5, =42");
        assert_eq!(stmts.len(), 1);
        if let Statement::Instruction(instr) = &stmts[0] {
            assert_eq!(instr.mnemonic, "ldr");
            assert!(matches!(&instr.operands[1], Operand::LiteralPoolValue(42)));
        } else {
            panic!("expected instruction");
        }
    }

    #[test]
    fn parse_ldr_literal_pool_negative() {
        let stmts = parse_aarch64("ldr x1, =-1");
        if let Statement::Instruction(instr) = &stmts[0] {
            assert!(matches!(&instr.operands[1], Operand::LiteralPoolValue(-1)));
        } else {
            panic!("expected instruction");
        }
    }

    #[test]
    fn parse_ldr_literal_pool_hex_large() {
        let stmts = parse_aarch64("ldr x0, =0xDEADBEEFCAFEBABE");
        if let Statement::Instruction(instr) = &stmts[0] {
            if let Operand::LiteralPoolValue(v) = &instr.operands[1] {
                assert_eq!(*v, 0xDEADBEEFCAFEBABEu64 as i128);
            } else {
                panic!("expected LiteralPoolValue");
            }
        } else {
            panic!("expected instruction");
        }
    }

    #[test]
    fn parse_ltorg_directive() {
        let stmts = parse_aarch64("ldr x0, =1\n.ltorg");
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[1], Statement::Ltorg(_)));
    }

    #[test]
    fn parse_pool_directive() {
        let stmts = parse_aarch64("ldr x0, =1\n.pool");
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[1], Statement::Ltorg(_)));
    }

    // ── SIMD/FP register parsing ────────────────────────────

    #[test]
    fn parse_simd_v_registers() {
        // V0–V31 should parse as AArch64 vector registers
        for i in 0..32 {
            let src = alloc::format!("fmov v{}, v0", i);
            let stmts = parse_aarch64(&src);
            assert_eq!(stmts.len(), 1, "parsing 'fmov v{}, v0' failed", i);
            if let Statement::Instruction(instr) = &stmts[0] {
                if let Operand::Register(r) = &instr.operands[0] {
                    assert!(r.is_a64_simd_fp(), "v{} should be SIMD/FP", i);
                    assert_eq!(r.a64_reg_num(), i as u8, "v{} reg num", i);
                    assert_eq!(r.a64_simd_fp_bits(), 128, "v{} should be 128 bits", i);
                } else {
                    panic!("expected register for v{}", i);
                }
            }
        }
    }

    #[test]
    fn parse_simd_q_registers() {
        for i in 0..32 {
            let src = alloc::format!("mov q{}, q0", i);
            let stmts = parse_aarch64(&src);
            if let Statement::Instruction(instr) = &stmts[0] {
                if let Operand::Register(r) = &instr.operands[0] {
                    assert!(r.is_a64_simd_fp(), "q{} should be SIMD/FP", i);
                    assert_eq!(r.a64_reg_num(), i as u8);
                    assert_eq!(r.a64_simd_fp_bits(), 128);
                }
            }
        }
    }

    #[test]
    fn parse_simd_d_registers() {
        for i in 0..32 {
            let src = alloc::format!("fmov d{}, d0", i);
            let stmts = parse_aarch64(&src);
            if let Statement::Instruction(instr) = &stmts[0] {
                if let Operand::Register(r) = &instr.operands[0] {
                    assert!(r.is_a64_simd_fp(), "d{} should be SIMD/FP", i);
                    assert_eq!(r.a64_reg_num(), i as u8);
                    assert_eq!(r.a64_simd_fp_bits(), 64);
                }
            }
        }
    }

    #[test]
    fn parse_simd_s_registers() {
        for i in 0..32 {
            let src = alloc::format!("fmov s{}, s0", i);
            let stmts = parse_aarch64(&src);
            if let Statement::Instruction(instr) = &stmts[0] {
                if let Operand::Register(r) = &instr.operands[0] {
                    assert!(r.is_a64_simd_fp(), "s{} should be SIMD/FP", i);
                    assert_eq!(r.a64_reg_num(), i as u8);
                    assert_eq!(r.a64_simd_fp_bits(), 32);
                }
            }
        }
    }

    #[test]
    fn parse_simd_h_registers() {
        for i in 0..32 {
            let src = alloc::format!("fmov h{}, h0", i);
            let stmts = parse_aarch64(&src);
            if let Statement::Instruction(instr) = &stmts[0] {
                if let Operand::Register(r) = &instr.operands[0] {
                    assert!(r.is_a64_simd_fp(), "h{} should be SIMD/FP", i);
                    assert_eq!(r.a64_reg_num(), i as u8);
                    assert_eq!(r.a64_simd_fp_bits(), 16);
                }
            }
        }
    }

    #[test]
    fn parse_simd_b_registers() {
        for i in 0..32 {
            let src = alloc::format!("fmov b{}, b0", i);
            let stmts = parse_aarch64(&src);
            if let Statement::Instruction(instr) = &stmts[0] {
                if let Operand::Register(r) = &instr.operands[0] {
                    assert!(r.is_a64_simd_fp(), "b{} should be SIMD/FP", i);
                    assert_eq!(r.a64_reg_num(), i as u8);
                    assert_eq!(r.a64_simd_fp_bits(), 8);
                }
            }
        }
    }

    // ── Vector arrangement specifier parsing ──────────────────────────
    #[test]
    fn parse_vector_arrangement_all_specifiers() {
        let cases = [
            ("add v0.8b, v1.8b, v2.8b", VectorArrangement::B8),
            ("add v0.16b, v1.16b, v2.16b", VectorArrangement::B16),
            ("add v0.4h, v1.4h, v2.4h", VectorArrangement::H4),
            ("add v0.8h, v1.8h, v2.8h", VectorArrangement::H8),
            ("add v0.2s, v1.2s, v2.2s", VectorArrangement::S2),
            ("add v0.4s, v1.4s, v2.4s", VectorArrangement::S4),
            ("add v0.1d, v1.1d, v2.1d", VectorArrangement::D1),
            ("add v0.2d, v1.2d, v2.2d", VectorArrangement::D2),
        ];
        for (src, expected_arr) in &cases {
            let stmts = parse_aarch64(src);
            if let Statement::Instruction(instr) = &stmts[0] {
                assert_eq!(instr.operands.len(), 3, "source: {}", src);
                for (j, op) in instr.operands.iter().enumerate() {
                    match op {
                        Operand::VectorRegister(_, arr) => {
                            assert_eq!(arr, expected_arr, "source: {}, operand {}", src, j);
                        }
                        other => panic!(
                            "expected VectorRegister, got {:?} for source: {}, operand {}",
                            other, src, j
                        ),
                    }
                }
            } else {
                panic!("expected instruction for source: {}", src);
            }
        }
    }

    #[test]
    fn parse_vector_arrangement_register_numbers() {
        // Verify that various register numbers parse correctly with arrangement
        for i in [0u32, 1, 7, 15, 16, 31] {
            let src = alloc::format!("add v{}.4s, v0.4s, v0.4s", i);
            let stmts = parse_aarch64(&src);
            if let Statement::Instruction(instr) = &stmts[0] {
                match &instr.operands[0] {
                    Operand::VectorRegister(reg, arr) => {
                        assert_eq!(reg.a64_reg_num(), i as u8, "v{}.4s reg num", i);
                        assert_eq!(*arr, VectorArrangement::S4);
                        assert!(reg.is_a64_vector());
                    }
                    other => panic!("expected VectorRegister, got {:?}", other),
                }
            }
        }
    }

    #[test]
    fn parse_vector_arrangement_case_insensitive() {
        // Arrangement specifiers should be case insensitive
        let cases = ["add v0.4S, v1.4S, v2.4S", "add V0.4s, V1.4s, V2.4s"];
        for src in &cases {
            let stmts = parse_aarch64(src);
            if let Statement::Instruction(instr) = &stmts[0] {
                for op in &instr.operands {
                    match op {
                        Operand::VectorRegister(_, arr) => {
                            assert_eq!(*arr, VectorArrangement::S4, "source: {}", src);
                        }
                        other => panic!(
                            "expected VectorRegister, got {:?} for source: {}",
                            other, src
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn parse_vector_reg_without_arrangement() {
        // V register without arrangement specifier should parse as plain Register
        let stmts = parse_aarch64("mov v0, v1");
        if let Statement::Instruction(instr) = &stmts[0] {
            match &instr.operands[0] {
                Operand::Register(reg) => {
                    assert!(reg.is_a64_vector());
                    assert_eq!(reg.a64_reg_num(), 0);
                }
                other => panic!("expected Register, got {:?}", other),
            }
        }
    }

    #[test]
    fn parse_vector_arrangement_display() {
        let stmts = parse_aarch64("add v3.2d, v4.2d, v5.2d");
        if let Statement::Instruction(instr) = &stmts[0] {
            if let Operand::VectorRegister(reg, arr) = &instr.operands[0] {
                assert_eq!(reg.a64_reg_num(), 3);
                assert_eq!(*arr, VectorArrangement::D2);
                let display = alloc::format!("{}", instr.operands[0]);
                // Display format is "{register_debug_lower}.{arrangement_display}"
                assert!(
                    display.contains("2D") || display.contains("2d"),
                    "Display should contain arrangement: {}",
                    display
                );
            } else {
                panic!("expected VectorRegister");
            }
        }
    }

    #[test]
    fn parse_vector_arrangement_element_properties() {
        // Verify element_bits, total_bits, lane_count through parsed arrangements
        let cases = [
            ("add v0.8b, v0.8b, v0.8b", 8u32, 64u32, 8u32),
            ("add v0.16b, v0.16b, v0.16b", 8, 128, 16),
            ("add v0.4h, v0.4h, v0.4h", 16, 64, 4),
            ("add v0.8h, v0.8h, v0.8h", 16, 128, 8),
            ("add v0.2s, v0.2s, v0.2s", 32, 64, 2),
            ("add v0.4s, v0.4s, v0.4s", 32, 128, 4),
            ("add v0.1d, v0.1d, v0.1d", 64, 64, 1),
            ("add v0.2d, v0.2d, v0.2d", 64, 128, 2),
        ];
        for (src, elem_bits, total_bits, lanes) in &cases {
            let stmts = parse_aarch64(src);
            if let Statement::Instruction(instr) = &stmts[0] {
                if let Operand::VectorRegister(_, arr) = &instr.operands[0] {
                    assert_eq!(arr.element_bits(), *elem_bits, "{}", src);
                    assert_eq!(arr.total_bits(), *total_bits, "{}", src);
                    assert_eq!(arr.lane_count(), *lanes, "{}", src);
                }
            }
        }
    }

    // === RISC-V Register Parsing ===

    fn parse_rv64(src: &str) -> Vec<Statement> {
        let tokens = crate::lexer::tokenize(src).unwrap();
        parse_with_syntax(&tokens, Arch::Rv64, Syntax::RiscV).unwrap()
    }

    fn parse_rv64_instr(src: &str) -> Instruction {
        let stmts = parse_rv64(src);
        assert_eq!(stmts.len(), 1, "expected 1 statement, got {}", stmts.len());
        match stmts.into_iter().next().unwrap() {
            Statement::Instruction(i) => i,
            s => panic!("expected instruction, got {:?}", s),
        }
    }

    #[test]
    fn riscv_fp_register_hardware_names() {
        // Verify all hardware names f0–f31 parse correctly
        for i in 0u8..32 {
            let src = alloc::format!("fadd.d f{}, f{}, f{}", i, i, i);
            let instr = parse_rv64_instr(&src);
            assert_eq!(instr.mnemonic, "fadd.d");
            assert_eq!(instr.operands.len(), 3);
            for op in &instr.operands {
                if let Operand::Register(reg) = op {
                    assert!(reg.is_riscv_fp(), "f{} should be FP register", i);
                    assert_eq!(reg.rv_fp_reg_num(), i, "f{} should map to reg {}", i, i);
                } else {
                    panic!("expected register operand for f{}", i);
                }
            }
        }
    }

    #[test]
    fn riscv_fp_register_abi_ft() {
        // ft0–ft7 → f0–f7, ft8–ft11 → f28–f31
        let mapping: &[(&str, u8)] = &[
            ("ft0", 0),
            ("ft1", 1),
            ("ft2", 2),
            ("ft3", 3),
            ("ft4", 4),
            ("ft5", 5),
            ("ft6", 6),
            ("ft7", 7),
            ("ft8", 28),
            ("ft9", 29),
            ("ft10", 30),
            ("ft11", 31),
        ];
        for &(name, expected_num) in mapping {
            let src = alloc::format!("fmv.d {}, {}", name, name);
            let instr = parse_rv64_instr(&src);
            if let Operand::Register(reg) = &instr.operands[0] {
                assert!(reg.is_riscv_fp(), "{} should be FP", name);
                assert_eq!(
                    reg.rv_fp_reg_num(),
                    expected_num,
                    "{} → f{}",
                    name,
                    expected_num
                );
            } else {
                panic!("expected register for {}", name);
            }
        }
    }

    #[test]
    fn riscv_fp_register_abi_fs() {
        // fs0–fs1 → f8–f9, fs2–fs11 → f18–f27
        let mapping: &[(&str, u8)] = &[
            ("fs0", 8),
            ("fs1", 9),
            ("fs2", 18),
            ("fs3", 19),
            ("fs4", 20),
            ("fs5", 21),
            ("fs6", 22),
            ("fs7", 23),
            ("fs8", 24),
            ("fs9", 25),
            ("fs10", 26),
            ("fs11", 27),
        ];
        for &(name, expected_num) in mapping {
            let src = alloc::format!("fmv.d {}, {}", name, name);
            let instr = parse_rv64_instr(&src);
            if let Operand::Register(reg) = &instr.operands[0] {
                assert!(reg.is_riscv_fp(), "{} should be FP", name);
                assert_eq!(
                    reg.rv_fp_reg_num(),
                    expected_num,
                    "{} → f{}",
                    name,
                    expected_num
                );
            } else {
                panic!("expected register for {}", name);
            }
        }
    }

    #[test]
    fn riscv_fp_register_abi_fa() {
        // fa0–fa7 → f10–f17
        let mapping: &[(&str, u8)] = &[
            ("fa0", 10),
            ("fa1", 11),
            ("fa2", 12),
            ("fa3", 13),
            ("fa4", 14),
            ("fa5", 15),
            ("fa6", 16),
            ("fa7", 17),
        ];
        for &(name, expected_num) in mapping {
            let src = alloc::format!("fmv.d {}, {}", name, name);
            let instr = parse_rv64_instr(&src);
            if let Operand::Register(reg) = &instr.operands[0] {
                assert!(reg.is_riscv_fp(), "{} should be FP", name);
                assert_eq!(
                    reg.rv_fp_reg_num(),
                    expected_num,
                    "{} → f{}",
                    name,
                    expected_num
                );
            } else {
                panic!("expected register for {}", name);
            }
        }
    }

    #[test]
    fn riscv_fp_mixed_with_integer_regs() {
        // FP load: flw ft0, 0(sp)  — FP dest, integer base
        let instr = parse_rv64_instr("flw ft0, 0(sp)");
        assert_eq!(instr.mnemonic, "flw");
        if let Operand::Register(reg) = &instr.operands[0] {
            assert!(reg.is_riscv_fp());
            assert_eq!(reg.rv_fp_reg_num(), 0);
        } else {
            panic!("expected FP register");
        }
    }

    #[test]
    fn riscv_fp_not_integer() {
        // FP registers should not report as integer RISC-V registers
        let instr = parse_rv64_instr("fadd.d fa0, fa1, fa2");
        for op in &instr.operands {
            if let Operand::Register(reg) = op {
                assert!(reg.is_riscv_fp());
                assert!(!reg.is_riscv());
            }
        }
    }

    #[test]
    fn riscv_integer_not_fp() {
        // Integer registers should not report as FP
        let instr = parse_rv64_instr("add a0, a1, a2");
        for op in &instr.operands {
            if let Operand::Register(reg) = op {
                assert!(reg.is_riscv());
                assert!(!reg.is_riscv_fp());
            }
        }
    }
}
