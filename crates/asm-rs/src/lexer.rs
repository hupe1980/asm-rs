//! Lexer for assembly source text.
//!
//! The lexer tokenizes assembly source into a stream of [`Token`](crate::lexer::Token)s, each
//! carrying its [`Span`](crate::error::Span) (source position) so that error messages can
//! point back to the exact location in the original input.

use alloc::borrow::Cow;
use alloc::string::String;
#[allow(unused_imports)]
use alloc::vec;
use alloc::vec::Vec;
use core::str;

use crate::error::{AsmError, Span};

/// A token produced by the lexer.
///
/// Token text is borrowed from the source string (`Cow::Borrowed`) in the
/// common case, avoiding per-token heap allocation.  String literals with
/// escape sequences are the only tokens that own their text on the heap.
#[derive(Debug, Clone, PartialEq)]
pub struct Token<'src> {
    /// Token classification.
    pub kind: TokenKind,
    /// Source text of the token — borrowed from input in the common case.
    pub text: Cow<'src, str>,
    /// Source location.
    pub span: Span,
}

impl<'src> Token<'src> {
    /// Returns the token text as a `&str`.
    #[inline]
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// The type of a token.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// An identifier: mnemonic, register name, or label reference.
    Ident,
    /// A numeric literal (integer).
    Number(i128),
    /// A string literal (content without quotes).
    StringLit,
    /// A character literal (e.g., 'A').
    CharLit(u8),
    /// A directive (starts with `.`).
    Directive,
    /// Label definition (`name:`).
    LabelDef,
    /// Numeric label definition (`1:`).
    NumericLabelDef(u32),
    /// Numeric label forward reference (`1f`).
    NumericLabelFwd(u32),
    /// Numeric label backward reference (`1b`).
    NumericLabelBwd(u32),
    /// Comma separator.
    Comma,
    /// Open bracket `[`.
    OpenBracket,
    /// Close bracket `]`.
    CloseBracket,
    /// Plus `+`.
    Plus,
    /// Minus `-`.
    Minus,
    /// Asterisk `*` (for scale in memory operands).
    Star,
    /// Colon `:` (segment override: `fs:`).
    Colon,
    /// Equals `=` (constant assignment: `name = value`).
    Equals,
    /// Open brace `{` (ARM register list).
    OpenBrace,
    /// Close brace `}` (ARM register list).
    CloseBrace,
    /// Open parenthesis `(` (RISC-V memory operand).
    OpenParen,
    /// Close parenthesis `)` (RISC-V memory operand).
    CloseParen,
    /// Exclamation mark `!` (ARM writeback).
    Bang,
    /// Percent sign `%` (AT&T register prefix).
    Percent,
    /// Dollar sign `$` (AT&T immediate prefix).
    Dollar,
    /// Forward slash `/` (SVE predicate qualifier: p0/m, p0/z).
    Slash,
    /// Ampersand `&` (bitwise AND in constant expressions).
    Ampersand,
    /// Pipe `|` (bitwise OR in constant expressions).
    Pipe,
    /// Caret `^` (bitwise XOR in constant expressions).
    Caret,
    /// Tilde `~` (bitwise NOT in constant expressions).
    Tilde,
    /// Left shift `<<`.
    LShift,
    /// Right shift `>>`.
    RShift,
    /// A newline (statement separator).
    Newline,
    /// End of input.
    Eof,
}

/// Tokenize assembly source text into a vector of tokens.
///
/// The lexer recognizes:
/// - Identifiers (mnemonics, registers, label references)
/// - Numeric literals (decimal, hex `0x`, binary `0b`, octal `0o`)
/// - String literals (`"..."`)
/// - Character literals (`'A'`)
/// - Directives (`.byte`, `.equ`, etc.)
/// - Label definitions (`name:`)
/// - Numeric labels (`1:`, `1b`, `1f`)
/// - Punctuation: `,`, `[`, `]`, `+`, `-`, `*`, `:`
/// - Comments: `#` to end of line
/// - Newlines and semicolons as statement separators
///
/// # Errors
///
/// Returns `Err(AsmError::Syntax)` if the input contains an unrecognised
/// character or a malformed token (e.g. an unterminated string literal).
pub fn tokenize<'s>(source: &'s str) -> Result<Vec<Token<'s>>, AsmError> {
    // Heuristic: ~4 chars per token on average (mnemonics, registers, punctuation).
    let mut tokens = Vec::with_capacity(source.len() / 3 + 1);
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut pos = 0;
    let mut line: u32 = 1;
    let mut col: u32 = 1;
    let mut line_start = 0usize;

    while pos < len {
        let ch = bytes[pos];

        // Skip whitespace (but not newlines)
        if ch == b' ' || ch == b'\t' || ch == b'\r' {
            pos += 1;
            col += 1;
            continue;
        }

        // Newline
        if ch == b'\n' {
            tokens.push(Token {
                kind: TokenKind::Newline,
                text: Cow::Borrowed("\n"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            line += 1;
            col = 1;
            line_start = pos;
            continue;
        }

        // Semicolon as statement separator
        if ch == b';' {
            let start = pos;
            tokens.push(Token {
                kind: TokenKind::Newline,
                text: Cow::Borrowed(";"),
                span: Span::new(line, col, start, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Comment: # to EOL
        if ch == b'#' {
            pos += 1;
            while pos < len && bytes[pos] != b'\n' {
                pos += 1;
            }
            col = (pos - line_start) as u32 + 1;
            continue;
        }

        // Comma
        if ch == b',' {
            tokens.push(Token {
                kind: TokenKind::Comma,
                text: Cow::Borrowed(","),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Brackets
        if ch == b'[' {
            tokens.push(Token {
                kind: TokenKind::OpenBracket,
                text: Cow::Borrowed("["),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }
        if ch == b']' {
            tokens.push(Token {
                kind: TokenKind::CloseBracket,
                text: Cow::Borrowed("]"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Plus
        if ch == b'+' {
            tokens.push(Token {
                kind: TokenKind::Plus,
                text: Cow::Borrowed("+"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Minus (standalone, not part of negative number if preceded by identifier or number)
        if ch == b'-' {
            // Check if this is a negative sign for a number
            let is_unary = tokens.is_empty()
                || matches!(
                    tokens.last().map(|t| &t.kind),
                    Some(
                        TokenKind::Comma
                            | TokenKind::OpenBracket
                            | TokenKind::OpenBrace
                            | TokenKind::Plus
                            | TokenKind::Minus
                            | TokenKind::Star
                            | TokenKind::Newline
                            | TokenKind::Equals
                    )
                );

            if is_unary && pos + 1 < len && bytes[pos + 1].is_ascii_digit() {
                // Parse as negative number
                let start = pos;
                let start_col = col;
                pos += 1; // skip '-'
                let value = parse_number_at(bytes, &mut pos, line, start_col)?;
                let token_len = pos - start;
                let text = Cow::Borrowed(str::from_utf8(&bytes[start..pos]).unwrap_or(""));
                tokens.push(Token {
                    kind: TokenKind::Number(-value),
                    text,
                    span: Span::new(line, start_col, start, token_len),
                });
                col = (pos - line_start) as u32 + 1;
                continue;
            }

            tokens.push(Token {
                kind: TokenKind::Minus,
                text: Cow::Borrowed("-"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Star
        if ch == b'*' {
            tokens.push(Token {
                kind: TokenKind::Star,
                text: Cow::Borrowed("*"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Colon (standalone, used for segment overrides)
        if ch == b':' {
            tokens.push(Token {
                kind: TokenKind::Colon,
                text: Cow::Borrowed(":"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Equals sign (constant assignment)
        if ch == b'=' {
            tokens.push(Token {
                kind: TokenKind::Equals,
                text: Cow::Borrowed("="),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // String literal
        if ch == b'"' {
            let start = pos;
            let start_col = col;
            pos += 1;
            col += 1;
            let mut content = Vec::new();
            while pos < len && bytes[pos] != b'"' {
                if bytes[pos] == b'\\' && pos + 1 < len {
                    pos += 1;
                    col += 1;
                    match bytes[pos] {
                        b'n' => content.push(b'\n'),
                        b't' => content.push(b'\t'),
                        b'\\' => content.push(b'\\'),
                        b'"' => content.push(b'"'),
                        b'0' => content.push(0),
                        b'x' => {
                            // \xHH
                            if pos + 2 < len {
                                let hi = hex_digit(bytes[pos + 1]);
                                let lo = hex_digit(bytes[pos + 2]);
                                if let (Some(h), Some(l)) = (hi, lo) {
                                    content.push(h * 16 + l);
                                    pos += 2;
                                    col += 2;
                                } else {
                                    return Err(AsmError::Syntax {
                                        msg: String::from("invalid \\xHH escape sequence"),
                                        span: Span::new(line, col, pos, 3),
                                    });
                                }
                            }
                        }
                        _ => {
                            return Err(AsmError::Syntax {
                                msg: alloc::format!(
                                    "unknown escape sequence '\\{}'",
                                    bytes[pos] as char
                                ),
                                span: Span::new(line, col, pos - 1, 2),
                            });
                        }
                    }
                } else if bytes[pos] == b'\n' {
                    return Err(AsmError::Syntax {
                        msg: String::from("unterminated string literal"),
                        span: Span::new(line, start_col, start, pos - start),
                    });
                } else {
                    content.push(bytes[pos]);
                }
                pos += 1;
                col += 1;
            }
            if pos >= len {
                return Err(AsmError::Syntax {
                    msg: String::from("unterminated string literal"),
                    span: Span::new(line, start_col, start, pos - start),
                });
            }
            pos += 1; // skip closing quote
            col += 1;
            let text_str = Cow::Owned(String::from_utf8(content).unwrap_or_default());
            tokens.push(Token {
                kind: TokenKind::StringLit,
                text: text_str,
                span: Span::new(line, start_col, start, pos - start),
            });
            continue;
        }

        // Character literal
        if ch == b'\'' {
            let start = pos;
            let start_col = col;
            pos += 1;
            col += 1;
            if pos >= len {
                return Err(AsmError::Syntax {
                    msg: String::from("unterminated character literal"),
                    span: Span::new(line, start_col, start, 1),
                });
            }
            let ch_val = if bytes[pos] == b'\\' && pos + 1 < len {
                pos += 1;
                col += 1;
                match bytes[pos] {
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'\\' => b'\\',
                    b'\'' => b'\'',
                    b'0' => 0,
                    _ => {
                        return Err(AsmError::Syntax {
                            msg: "unknown escape in character literal".into(),
                            span: Span::new(line, col, pos - 1, 2),
                        });
                    }
                }
            } else {
                bytes[pos]
            };
            pos += 1;
            col += 1;
            if pos >= len || bytes[pos] != b'\'' {
                return Err(AsmError::Syntax {
                    msg: String::from("unterminated character literal"),
                    span: Span::new(line, start_col, start, pos - start),
                });
            }
            pos += 1;
            col += 1;
            tokens.push(Token {
                kind: TokenKind::CharLit(ch_val),
                text: Cow::Owned(alloc::format!("'{}'", ch_val as char)),
                span: Span::new(line, start_col, start, pos - start),
            });
            continue;
        }

        // Directive (starts with '.')
        if ch == b'.' {
            let start = pos;
            let start_col = col;
            pos += 1;
            col += 1;
            while pos < len && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
                pos += 1;
                col += 1;
            }
            let text = Cow::Borrowed(str::from_utf8(&bytes[start..pos]).unwrap_or(""));
            tokens.push(Token {
                kind: TokenKind::Directive,
                text,
                span: Span::new(line, start_col, start, pos - start),
            });
            continue;
        }

        // Number
        if ch.is_ascii_digit() {
            let start = pos;
            let start_col = col;

            // Check for numeric label: digit(s) followed by `:`, `b`, or `f`
            // but NOT hex prefix 0x, 0b (binary), 0o
            let mut temp = pos;
            while temp < len && bytes[temp].is_ascii_digit() {
                temp += 1;
            }
            // Check for numeric label def: `1:`
            // Only single-digit labels (0-9) are valid, matching GAS convention.
            // Multi-digit numbers followed by `:` are rejected to avoid
            // silent mismatch with references (which must be single-digit).
            if temp < len && bytes[temp] == b':' && (temp + 1 >= len || bytes[temp + 1] != b':') {
                // Must be all digits
                let num_str = str::from_utf8(&bytes[start..temp]).unwrap_or("0");
                if let Ok(n) = num_str.parse::<u32>() {
                    if temp != start + 1 {
                        return Err(AsmError::Syntax {
                            msg: alloc::format!(
                                "numeric labels must be a single digit (0-9), got `{}`",
                                n
                            ),
                            span: Span::new(line, start_col, start, temp - start + 1),
                        });
                    }
                    pos = temp + 1; // past the ':'
                    col = (pos - line_start) as u32 + 1;
                    tokens.push(Token {
                        kind: TokenKind::NumericLabelDef(n),
                        text: Cow::Owned(alloc::format!("{}:", n)),
                        span: Span::new(line, start_col, start, pos - start),
                    });
                    continue;
                }
            }
            // Check for numeric label ref: `1b` or `1f` (only single digit before b/f)
            if temp < len && temp == start + 1 && (bytes[temp] == b'b' || bytes[temp] == b'f') {
                // Make sure it's not '0b' (binary prefix) — '0b' followed by 0/1 is binary
                let digit = bytes[start] - b'0';
                let suffix = bytes[temp];
                if !(digit == 0
                    && suffix == b'b'
                    && temp + 1 < len
                    && (bytes[temp + 1] == b'0' || bytes[temp + 1] == b'1'))
                {
                    pos = temp + 1;
                    col = (pos - line_start) as u32 + 1;
                    let kind = if suffix == b'b' {
                        TokenKind::NumericLabelBwd(digit as u32)
                    } else {
                        TokenKind::NumericLabelFwd(digit as u32)
                    };
                    tokens.push(Token {
                        kind,
                        text: Cow::Owned(alloc::format!("{}{}", digit, suffix as char)),
                        span: Span::new(line, start_col, start, pos - start),
                    });
                    continue;
                }
            }

            let value = parse_number_at(bytes, &mut pos, line, start_col)?;
            let token_len = pos - start;
            let text = Cow::Borrowed(str::from_utf8(&bytes[start..pos]).unwrap_or(""));
            tokens.push(Token {
                kind: TokenKind::Number(value),
                text,
                span: Span::new(line, start_col, start, token_len),
            });
            col = (pos - line_start) as u32 + 1;
            continue;
        }

        // Identifier or keyword (including register names)
        if ch.is_ascii_alphabetic() || ch == b'_' {
            let start = pos;
            let start_col = col;
            while pos < len
                && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_' || bytes[pos] == b'.')
            {
                pos += 1;
            }
            let text = Cow::Borrowed(str::from_utf8(&bytes[start..pos]).unwrap_or(""));
            let token_len = pos - start;

            // Check if followed by ':' → label definition
            // But NOT if it's a segment register (cs, ds, es, fs, gs, ss)
            if pos < len && bytes[pos] == b':' {
                let is_segment_reg = text.eq_ignore_ascii_case("cs")
                    || text.eq_ignore_ascii_case("ds")
                    || text.eq_ignore_ascii_case("es")
                    || text.eq_ignore_ascii_case("fs")
                    || text.eq_ignore_ascii_case("gs")
                    || text.eq_ignore_ascii_case("ss");
                if is_segment_reg {
                    // Emit as Ident; the ':' will be consumed next iteration
                    tokens.push(Token {
                        kind: TokenKind::Ident,
                        text,
                        span: Span::new(line, start_col, start, token_len),
                    });
                    col = (pos - line_start) as u32 + 1;
                    continue;
                }
                pos += 1; // consume ':'
                tokens.push(Token {
                    kind: TokenKind::LabelDef,
                    text,
                    span: Span::new(line, start_col, start, pos - start),
                });
                col = (pos - line_start) as u32 + 1;
                continue;
            }

            tokens.push(Token {
                kind: TokenKind::Ident,
                text,
                span: Span::new(line, start_col, start, token_len),
            });
            col = (pos - line_start) as u32 + 1;
            continue;
        }

        // Open brace (ARM register lists)
        if ch == b'{' {
            tokens.push(Token {
                kind: TokenKind::OpenBrace,
                text: Cow::Borrowed("{"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Close brace (ARM register lists)
        if ch == b'}' {
            tokens.push(Token {
                kind: TokenKind::CloseBrace,
                text: Cow::Borrowed("}"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Open parenthesis (RISC-V memory operands)
        if ch == b'(' {
            tokens.push(Token {
                kind: TokenKind::OpenParen,
                text: Cow::Borrowed("("),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Close parenthesis (RISC-V memory operands)
        if ch == b')' {
            tokens.push(Token {
                kind: TokenKind::CloseParen,
                text: Cow::Borrowed(")"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Bang (ARM writeback)
        if ch == b'!' {
            tokens.push(Token {
                kind: TokenKind::Bang,
                text: Cow::Borrowed("!"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Percent (AT&T register prefix)
        if ch == b'%' {
            tokens.push(Token {
                kind: TokenKind::Percent,
                text: Cow::Borrowed("%"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Dollar (AT&T immediate prefix)
        if ch == b'$' {
            tokens.push(Token {
                kind: TokenKind::Dollar,
                text: Cow::Borrowed("$"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Slash (SVE predicate qualifier)
        if ch == b'/' {
            // Check for C-style comments first
            if pos + 1 < len && bytes[pos + 1] == b'/' {
                // Line comment: skip to end of line
                pos += 2;
                while pos < len && bytes[pos] != b'\n' {
                    pos += 1;
                }
                col = (pos - line_start) as u32 + 1;
                continue;
            }
            if pos + 1 < len && bytes[pos + 1] == b'*' {
                // Block comment: skip to matching */
                let comment_start_line = line;
                let comment_start_col = col;
                let comment_start_pos = pos;
                pos += 2;
                col += 2;
                while pos + 1 < len && !(bytes[pos] == b'*' && bytes[pos + 1] == b'/') {
                    if bytes[pos] == b'\n' {
                        line += 1;
                        col = 1;
                        line_start = pos + 1;
                    } else {
                        col += 1;
                    }
                    pos += 1;
                }
                if pos + 1 < len {
                    pos += 2; // skip */
                    col += 2;
                } else {
                    // Reached EOF without finding */
                    return Err(AsmError::Syntax {
                        msg: String::from("unterminated block comment"),
                        span: Span::new(
                            comment_start_line,
                            comment_start_col,
                            comment_start_pos,
                            2,
                        ),
                    });
                }
                continue;
            }
            tokens.push(Token {
                kind: TokenKind::Slash,
                text: Cow::Borrowed("/"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Ampersand (bitwise AND)
        if ch == b'&' {
            tokens.push(Token {
                kind: TokenKind::Ampersand,
                text: Cow::Borrowed("&"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Pipe (bitwise OR)
        if ch == b'|' {
            tokens.push(Token {
                kind: TokenKind::Pipe,
                text: Cow::Borrowed("|"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Caret (bitwise XOR)
        if ch == b'^' {
            tokens.push(Token {
                kind: TokenKind::Caret,
                text: Cow::Borrowed("^"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Tilde (bitwise NOT)
        if ch == b'~' {
            tokens.push(Token {
                kind: TokenKind::Tilde,
                text: Cow::Borrowed("~"),
                span: Span::new(line, col, pos, 1),
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Shift operators << >>
        if ch == b'<' && pos + 1 < len && bytes[pos + 1] == b'<' {
            tokens.push(Token {
                kind: TokenKind::LShift,
                text: Cow::Borrowed("<<"),
                span: Span::new(line, col, pos, 2),
            });
            pos += 2;
            col += 2;
            continue;
        }
        if ch == b'>' && pos + 1 < len && bytes[pos + 1] == b'>' {
            tokens.push(Token {
                kind: TokenKind::RShift,
                text: Cow::Borrowed(">>"),
                span: Span::new(line, col, pos, 2),
            });
            pos += 2;
            col += 2;
            continue;
        }

        // Unknown character
        return Err(AsmError::Syntax {
            msg: alloc::format!("unexpected character '{}'", ch as char),
            span: Span::new(line, col, pos, 1),
        });
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        text: Cow::Borrowed(""),
        span: Span::new(line, col, pos, 0),
    });

    Ok(tokens)
}

/// Parse a number starting at `pos` in `bytes`. Advances `pos` past the number.
#[inline]
fn parse_number_at(
    bytes: &[u8],
    pos: &mut usize,
    span_line: u32,
    span_col: u32,
) -> Result<i128, AsmError> {
    let start = *pos;
    let len = bytes.len();

    if *pos >= len {
        return Err(AsmError::Syntax {
            msg: String::from("expected number"),
            span: Span::new(span_line, span_col, start, 0),
        });
    }

    // Check for hex, binary, octal prefix
    if bytes[*pos] == b'0' && *pos + 1 < len {
        match bytes[*pos + 1] {
            b'x' | b'X' => {
                *pos += 2;
                let num_start = *pos;
                while *pos < len && bytes[*pos].is_ascii_hexdigit() {
                    *pos += 1;
                }
                if *pos == num_start {
                    return Err(AsmError::Syntax {
                        msg: String::from("expected hex digits after '0x'"),
                        span: Span::new(span_line, span_col, start, *pos - start),
                    });
                }
                let s = str::from_utf8(&bytes[num_start..*pos]).unwrap_or("0");
                return i128::from_str_radix(s, 16).map_err(|_| AsmError::Syntax {
                    msg: alloc::format!("invalid hex number '0x{}'", s),
                    span: Span::new(span_line, span_col, start, *pos - start),
                });
            }
            b'b' | b'B' => {
                // Could be binary 0b prefix — check if next chars are 0 or 1
                if *pos + 2 < len && (bytes[*pos + 2] == b'0' || bytes[*pos + 2] == b'1') {
                    *pos += 2;
                    let num_start = *pos;
                    while *pos < len && (bytes[*pos] == b'0' || bytes[*pos] == b'1') {
                        *pos += 1;
                    }
                    let s = str::from_utf8(&bytes[num_start..*pos]).unwrap_or("0");
                    return i128::from_str_radix(s, 2).map_err(|_| AsmError::Syntax {
                        msg: alloc::format!("invalid binary number '0b{}'", s),
                        span: Span::new(span_line, span_col, start, *pos - start),
                    });
                }
                // Otherwise, just '0' followed by 'b' which is not a binary prefix
            }
            b'o' | b'O' => {
                *pos += 2;
                let num_start = *pos;
                while *pos < len && bytes[*pos] >= b'0' && bytes[*pos] <= b'7' {
                    *pos += 1;
                }
                if *pos == num_start {
                    return Err(AsmError::Syntax {
                        msg: String::from("expected octal digits after '0o'"),
                        span: Span::new(span_line, span_col, start, *pos - start),
                    });
                }
                let s = str::from_utf8(&bytes[num_start..*pos]).unwrap_or("0");
                return i128::from_str_radix(s, 8).map_err(|_| AsmError::Syntax {
                    msg: alloc::format!("invalid octal number '0o{}'", s),
                    span: Span::new(span_line, span_col, start, *pos - start),
                });
            }
            _ => {}
        }
    }

    // Decimal
    while *pos < len && bytes[*pos].is_ascii_digit() {
        *pos += 1;
    }
    // Check for hex suffix (e.g., 0FFh) — common in NASM/MASM
    if *pos < len && (bytes[*pos] == b'h' || bytes[*pos] == b'H') {
        let s = str::from_utf8(&bytes[start..*pos]).unwrap_or("0");
        *pos += 1; // consume 'h'
        return i128::from_str_radix(s, 16).map_err(|_| AsmError::Syntax {
            msg: alloc::format!("invalid hex number '{}h'", s),
            span: Span::new(span_line, span_col, start, *pos - start),
        });
    }
    let s = str::from_utf8(&bytes[start..*pos]).unwrap_or("0");
    s.parse::<i128>().map_err(|_| AsmError::Syntax {
        msg: alloc::format!("invalid number '{}'", s),
        span: Span::new(span_line, span_col, start, *pos - start),
    })
}

#[inline]
fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok_kinds(src: &str) -> Vec<TokenKind> {
        tokenize(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[allow(dead_code)]
    fn tok_texts(src: &str) -> Vec<String> {
        tokenize(src)
            .unwrap()
            .into_iter()
            .map(|t| t.text.into_owned())
            .collect()
    }

    #[test]
    fn empty_input() {
        let tokens = tokenize("").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    #[test]
    fn only_whitespace() {
        let tokens = tokenize("   \t  ").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    #[test]
    fn only_comment() {
        // Hash is the comment marker; semicolons are statement separators
        let tokens = tokenize("# this is a comment").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    #[test]
    fn hash_comment() {
        let tokens = tokenize("# comment").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    #[test]
    fn simple_instruction() {
        let kinds = tok_kinds("mov rax, rbx");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident, // mov
                TokenKind::Ident, // rax
                TokenKind::Comma,
                TokenKind::Ident, // rbx
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn instruction_with_immediate() {
        let tokens = tokenize("mov rax, 42").unwrap();
        assert_eq!(tokens[3].kind, TokenKind::Number(42));
    }

    #[test]
    fn hex_immediate() {
        let tokens = tokenize("mov rax, 0xFF").unwrap();
        assert_eq!(tokens[3].kind, TokenKind::Number(255));
    }

    #[test]
    fn hex_uppercase() {
        let tokens = tokenize("mov rax, 0XAB").unwrap();
        assert_eq!(tokens[3].kind, TokenKind::Number(0xAB));
    }

    #[test]
    fn binary_immediate() {
        let tokens = tokenize("mov rax, 0b1010").unwrap();
        assert_eq!(tokens[3].kind, TokenKind::Number(10));
    }

    #[test]
    fn octal_immediate() {
        let tokens = tokenize("mov rax, 0o77").unwrap();
        assert_eq!(tokens[3].kind, TokenKind::Number(63));
    }

    #[test]
    fn negative_immediate() {
        let tokens = tokenize("mov rax, -1").unwrap();
        assert_eq!(tokens[3].kind, TokenKind::Number(-1));
    }

    #[test]
    fn negative_hex() {
        let tokens = tokenize("add rsp, -0x10").unwrap();
        assert_eq!(tokens[3].kind, TokenKind::Number(-16));
    }

    #[test]
    fn label_definition() {
        let tokens = tokenize("entry_point:").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::LabelDef);
        assert_eq!(tokens[0].text, "entry_point");
    }

    #[test]
    fn label_definition_with_instruction() {
        let kinds = tok_kinds("loop: dec rcx");
        assert_eq!(kinds[0], TokenKind::LabelDef);
        assert_eq!(kinds[1], TokenKind::Ident); // dec
        assert_eq!(kinds[2], TokenKind::Ident); // rcx
    }

    #[test]
    fn numeric_label_def() {
        let tokens = tokenize("1:").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::NumericLabelDef(1));
    }

    #[test]
    fn numeric_label_backward_ref() {
        let tokens = tokenize("jnz 1b").unwrap();
        assert_eq!(tokens[1].kind, TokenKind::NumericLabelBwd(1));
    }

    #[test]
    fn numeric_label_forward_ref() {
        let tokens = tokenize("jmp 2f").unwrap();
        assert_eq!(tokens[1].kind, TokenKind::NumericLabelFwd(2));
    }

    #[test]
    fn directive() {
        let tokens = tokenize(".byte 0x90").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Directive);
        assert_eq!(tokens[0].text, ".byte");
        assert_eq!(tokens[1].kind, TokenKind::Number(0x90));
    }

    #[test]
    fn equ_directive() {
        let tokens = tokenize(".equ SYS_WRITE, 1").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Directive);
        assert_eq!(tokens[0].text, ".equ");
        assert_eq!(tokens[1].kind, TokenKind::Ident);
        assert_eq!(tokens[1].text, "SYS_WRITE");
    }

    #[test]
    fn memory_operand_tokens() {
        let kinds = tok_kinds("[rax + rbx*4 + 8]");
        assert_eq!(
            kinds,
            vec![
                TokenKind::OpenBracket,
                TokenKind::Ident, // rax
                TokenKind::Plus,
                TokenKind::Ident, // rbx
                TokenKind::Star,
                TokenKind::Number(4),
                TokenKind::Plus,
                TokenKind::Number(8),
                TokenKind::CloseBracket,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn string_literal() {
        let tokens = tokenize(".asciz \"hello\"").unwrap();
        assert_eq!(tokens[1].kind, TokenKind::StringLit);
        assert_eq!(tokens[1].text, "hello");
    }

    #[test]
    fn string_escape_sequences() {
        let tokens = tokenize(".ascii \"a\\nb\\t\\\\c\\0\\x41\"").unwrap();
        assert_eq!(tokens[1].kind, TokenKind::StringLit);
        assert_eq!(tokens[1].text, "a\nb\t\\c\0A");
    }

    #[test]
    fn character_literal() {
        let tokens = tokenize("mov al, 'A'").unwrap();
        assert_eq!(tokens[3].kind, TokenKind::CharLit(b'A'));
    }

    #[test]
    fn semicolon_separator() {
        let kinds = tok_kinds("nop; ret");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident,   // nop
                TokenKind::Newline, // ;
                TokenKind::Ident,   // ret
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn newline_separator() {
        let kinds = tok_kinds("nop\nret");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident, // nop
                TokenKind::Newline,
                TokenKind::Ident, // ret
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn segment_override_tokens() {
        let kinds = tok_kinds("fs:[rax]");
        assert_eq!(kinds[0], TokenKind::Ident); // fs
        assert_eq!(kinds[1], TokenKind::Colon);
        assert_eq!(kinds[2], TokenKind::OpenBracket);
        assert_eq!(kinds[3], TokenKind::Ident); // rax
        assert_eq!(kinds[4], TokenKind::CloseBracket);
    }

    #[test]
    fn size_hint_tokens() {
        let kinds = tok_kinds("byte ptr [rax]");
        assert_eq!(kinds[0], TokenKind::Ident); // byte
        assert_eq!(kinds[1], TokenKind::Ident); // ptr
        assert_eq!(kinds[2], TokenKind::OpenBracket);
    }

    #[test]
    fn prefix_and_instruction() {
        let kinds = tok_kinds("lock add [rax], 1");
        assert_eq!(kinds[0], TokenKind::Ident); // lock
        assert_eq!(kinds[1], TokenKind::Ident); // add
    }

    #[test]
    fn span_tracking() {
        let tokens = tokenize("mov rax, 1").unwrap();
        assert_eq!(tokens[0].span, Span::new(1, 1, 0, 3)); // "mov"
        assert_eq!(tokens[1].span, Span::new(1, 5, 4, 3)); // "rax"
        assert_eq!(tokens[2].span, Span::new(1, 8, 7, 1)); // ","
    }

    #[test]
    fn multiline_span_tracking() {
        let tokens = tokenize("nop\nmov rax, 1").unwrap();
        assert_eq!(tokens[0].span.line, 1); // nop
        assert_eq!(tokens[2].span.line, 2); // mov (after newline)
    }

    #[test]
    fn unknown_character_error() {
        let err = tokenize("mov rax, @").unwrap_err();
        match err {
            AsmError::Syntax { msg, .. } => {
                assert!(msg.contains("unexpected character '@'"));
            }
            _ => panic!("expected Syntax error"),
        }
    }

    #[test]
    fn unterminated_string() {
        let err = tokenize(".ascii \"hello").unwrap_err();
        match err {
            AsmError::Syntax { msg, .. } => {
                assert!(msg.contains("unterminated string"));
            }
            _ => panic!("expected Syntax error"),
        }
    }

    #[test]
    fn unterminated_block_comment() {
        let err = tokenize("nop /* this is never closed").unwrap_err();
        match err {
            AsmError::Syntax { msg, span } => {
                assert!(
                    msg.contains("unterminated block comment"),
                    "expected 'unterminated block comment', got: {msg}"
                );
                // Span should point to the start of the comment, not (0,0)
                assert!(span.line > 0 || span.col > 0, "span should not be (0,0)");
            }
            _ => panic!("expected Syntax error"),
        }
    }

    #[test]
    fn complex_instruction() {
        let tokens = tokenize("mov qword ptr [rbp - 0x10], rax").unwrap();
        let texts: Vec<_> = tokens.iter().map(|t| &*t.text).collect();
        assert_eq!(
            texts,
            vec!["mov", "qword", "ptr", "[", "rbp", "-", "0x10", "]", ",", "rax", ""]
        );
    }

    #[test]
    fn all_punctuation() {
        let kinds = tok_kinds(", [ ] + - * :");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Comma,
                TokenKind::OpenBracket,
                TokenKind::CloseBracket,
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Colon,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn trailing_whitespace() {
        let tokens = tokenize("nop   ").unwrap();
        assert_eq!(tokens.len(), 2); // nop + Eof
    }

    #[test]
    fn zero_immediate() {
        let tokens = tokenize("xor eax, 0").unwrap();
        assert_eq!(tokens[3].kind, TokenKind::Number(0));
    }

    #[test]
    fn large_hex_immediate() {
        let tokens = tokenize("mov rdi, 0x68732f2f6e69622f").unwrap();
        assert_eq!(tokens[3].kind, TokenKind::Number(0x68732f2f6e69622f));
    }

    #[test]
    fn minus_in_memory_operand_is_not_unary() {
        // After identifier, '-' should be an operator, not unary
        let kinds = tok_kinds("[rbp - 0x10]");
        assert_eq!(
            kinds,
            vec![
                TokenKind::OpenBracket,
                TokenKind::Ident, // rbp
                TokenKind::Minus,
                TokenKind::Number(0x10),
                TokenKind::CloseBracket,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn equals_token() {
        let kinds = tok_kinds("EXIT = 60");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident, // EXIT
                TokenKind::Equals,
                TokenKind::Number(60),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn equals_with_negative() {
        let kinds = tok_kinds("NEG = -1");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident,
                TokenKind::Equals,
                TokenKind::Number(-1),
                TokenKind::Eof,
            ]
        );
    }
}
