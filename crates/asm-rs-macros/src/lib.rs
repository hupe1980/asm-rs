//! Compile-time assembly proc-macros for [`asm-rs`](https://crates.io/crates/asm-rs).
//!
//! Provides the [`asm_bytes!`] macro that assembles source text at compile time,
//! producing a `&'static [u8]` constant with zero runtime overhead.
//!
//! # Usage
//!
//! ```rust,ignore
//! use asm_rs_macros::asm_bytes;
//!
//! // x86-64 shellcode assembled at compile time
//! const SHELLCODE: &[u8] = asm_bytes!(x86_64, "mov rax, 1\nret");
//!
//! // AArch64
//! const A64_CODE: &[u8] = asm_bytes!(aarch64, "add x0, x1, x2\nret");
//!
//! // ARM32
//! const ARM_CODE: &[u8] = asm_bytes!(arm, "add r0, r1, r2\nbx lr");
//!
//! // RISC-V 64-bit
//! const RV_CODE: &[u8] = asm_bytes!(rv64, "add a0, a1, a2\nret");
//! ```

use proc_macro::TokenStream;

/// Assemble source text at compile time, producing a `&'static [u8]` byte slice.
///
/// # Syntax
///
/// ```rust,ignore
/// asm_bytes!(ARCH, "assembly source")
/// ```
///
/// where `ARCH` is one of: `x86`, `x86_64`, `arm`, `thumb`, `aarch64`, `rv32`, `rv64`.
///
/// # Examples
///
/// ```rust,ignore
/// use asm_rs_macros::asm_bytes;
///
/// // Single instruction
/// const NOP: &[u8] = asm_bytes!(x86_64, "nop");
/// assert_eq!(NOP, &[0x90]);
///
/// // Multi-instruction with labels
/// const CODE: &[u8] = asm_bytes!(x86_64, "
///     start:
///         xor eax, eax
///         inc eax
///         ret
/// ");
///
/// // With base address
/// const BASED: &[u8] = asm_bytes!(x86_64, 0x400000, "mov rax, 1\nret");
/// ```
///
/// # Compile-time errors
///
/// If the assembly source contains errors, the macro emits a compile-time error
/// with the full `AsmError` diagnostic message.
#[proc_macro]
pub fn asm_bytes(input: TokenStream) -> TokenStream {
    match asm_bytes_impl(input) {
        Ok(ts) => ts,
        Err(err) => err.into_compile_error(),
    }
}

/// Assemble source text at compile time, producing a fixed-size array `[u8; N]`.
///
/// Unlike [`asm_bytes!`] which returns `&'static [u8]`, this macro returns a
/// `[u8; N]` value that can be used where a fixed-size array is needed.
///
/// # Syntax
///
/// ```rust,ignore
/// asm_array!(ARCH, "assembly source")
/// ```
///
/// # Examples
///
/// ```rust,ignore
/// use asm_rs_macros::asm_array;
///
/// const NOP: [u8; 1] = asm_array!(x86_64, "nop");
/// const RET: [u8; 1] = asm_array!(x86_64, "ret");
/// ```
#[proc_macro]
pub fn asm_array(input: TokenStream) -> TokenStream {
    match asm_array_impl(input) {
        Ok(ts) => ts,
        Err(err) => err.into_compile_error(),
    }
}

// ─── Implementation ─────────────────────────────────────────────────────────

struct MacroInput {
    arch: asm_rs::Arch,
    base_addr: u64,
    source: String,
    /// Span of the source literal for error reporting.
    source_span: proc_macro::Span,
}

fn parse_input(input: TokenStream) -> Result<MacroInput, syn_free::Error> {
    let mut tokens = input.into_iter().peekable();

    // 1. Parse architecture identifier
    let arch_tt = tokens.next().ok_or_else(|| {
        syn_free::Error::new(
            "expected architecture identifier (x86_64, aarch64, arm, thumb, rv32, rv64)",
        )
    })?;
    let arch = parse_arch(&arch_tt)?;

    // 2. Expect comma
    expect_comma(&mut tokens)?;

    // 3. Optional base address (integer literal followed by comma)
    let base_addr;
    let source;
    let source_span;

    if let Some(tt) = tokens.peek() {
        if is_integer_literal(tt) {
            let tt = tokens.next().unwrap();
            base_addr = parse_integer_literal(&tt)?;
            expect_comma(&mut tokens)?;
            let (src, span) = parse_string_literal(&mut tokens)?;
            source = src;
            source_span = span;
        } else {
            base_addr = 0;
            let (src, span) = parse_string_literal(&mut tokens)?;
            source = src;
            source_span = span;
        }
    } else {
        return Err(syn_free::Error::new("expected assembly source string"));
    }

    // Ensure no trailing tokens
    if tokens.next().is_some() {
        return Err(syn_free::Error::new(
            "unexpected extra tokens after source string",
        ));
    }

    Ok(MacroInput {
        arch,
        base_addr,
        source,
        source_span,
    })
}

fn asm_bytes_impl(input: TokenStream) -> Result<TokenStream, syn_free::Error> {
    let mi = parse_input(input)?;
    let bytes = do_assemble(&mi)?;
    Ok(bytes_to_slice_expr(&bytes))
}

fn asm_array_impl(input: TokenStream) -> Result<TokenStream, syn_free::Error> {
    let mi = parse_input(input)?;
    let bytes = do_assemble(&mi)?;
    Ok(bytes_to_array_expr(&bytes))
}

fn do_assemble(mi: &MacroInput) -> Result<Vec<u8>, syn_free::Error> {
    let result = if mi.base_addr != 0 {
        asm_rs::assemble_at(&mi.source, mi.arch, mi.base_addr)
    } else {
        asm_rs::assemble(&mi.source, mi.arch)
    };

    result.map_err(|e| syn_free::Error::with_span(mi.source_span, &format!("assembly error: {e}")))
}

fn parse_arch(tt: &proc_macro::TokenTree) -> Result<asm_rs::Arch, syn_free::Error> {
    let ident = match tt {
        proc_macro::TokenTree::Ident(id) => id.to_string(),
        _ => {
            return Err(syn_free::Error::new(
                "expected architecture identifier (x86, x86_64, aarch64, arm, thumb, rv32, rv64)",
            ));
        }
    };
    match ident.as_str() {
        "x86" => Ok(asm_rs::Arch::X86),
        "x86_64" => Ok(asm_rs::Arch::X86_64),
        "arm" => Ok(asm_rs::Arch::Arm),
        "thumb" => Ok(asm_rs::Arch::Thumb),
        "aarch64" => Ok(asm_rs::Arch::Aarch64),
        "rv32" => Ok(asm_rs::Arch::Rv32),
        "rv64" => Ok(asm_rs::Arch::Rv64),
        _ => Err(syn_free::Error::with_span(
            tt.span(),
            &format!(
                "unknown architecture `{ident}`, expected: x86, x86_64, arm, thumb, aarch64, rv32, rv64"
            ),
        )),
    }
}

fn expect_comma(
    tokens: &mut std::iter::Peekable<proc_macro::token_stream::IntoIter>,
) -> Result<(), syn_free::Error> {
    match tokens.next() {
        Some(proc_macro::TokenTree::Punct(p)) if p.as_char() == ',' => Ok(()),
        Some(other) => Err(syn_free::Error::with_span(other.span(), "expected `,`")),
        None => Err(syn_free::Error::new("expected `,`")),
    }
}

fn is_integer_literal(tt: &proc_macro::TokenTree) -> bool {
    matches!(tt, proc_macro::TokenTree::Literal(lit) if {
        let s = lit.to_string();
        s.starts_with(|c: char| c.is_ascii_digit())
            && !s.starts_with('"')
            && !s.starts_with('\'')
    })
}

fn parse_integer_literal(tt: &proc_macro::TokenTree) -> Result<u64, syn_free::Error> {
    let proc_macro::TokenTree::Literal(lit) = tt else {
        return Err(syn_free::Error::with_span(
            tt.span(),
            "expected integer literal",
        ));
    };
    let s = lit.to_string();
    let val = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16)
    } else {
        s.parse::<u64>()
    };
    val.map_err(|_| syn_free::Error::with_span(tt.span(), "invalid integer literal"))
}

fn parse_string_literal(
    tokens: &mut std::iter::Peekable<proc_macro::token_stream::IntoIter>,
) -> Result<(String, proc_macro::Span), syn_free::Error> {
    let tt = tokens
        .next()
        .ok_or_else(|| syn_free::Error::new("expected string literal"))?;
    let proc_macro::TokenTree::Literal(lit) = &tt else {
        return Err(syn_free::Error::with_span(
            tt.span(),
            "expected string literal",
        ));
    };
    let raw = lit.to_string();
    // Strip quotes — handle both `"..."` and `r"..."` / `r#"..."#`
    let content = if raw.starts_with("r#\"") {
        raw.strip_prefix("r#\"")
            .and_then(|s| s.strip_suffix("\"#"))
            .ok_or_else(|| syn_free::Error::with_span(tt.span(), "malformed raw string"))?
    } else if raw.starts_with("r\"") {
        raw.strip_prefix("r\"")
            .and_then(|s| s.strip_suffix('"'))
            .ok_or_else(|| syn_free::Error::with_span(tt.span(), "malformed raw string"))?
    } else if raw.starts_with('"') {
        // Regular string — need to unescape
        let inner = raw
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .ok_or_else(|| syn_free::Error::with_span(tt.span(), "malformed string literal"))?;
        return Ok((unescape_string(inner), tt.span()));
    } else {
        return Err(syn_free::Error::with_span(
            tt.span(),
            "expected string literal",
        ));
    };
    Ok((content.to_string(), tt.span()))
}

fn unescape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('0') => out.push('\0'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn bytes_to_slice_expr(bytes: &[u8]) -> TokenStream {
    let byte_strs: Vec<String> = bytes.iter().map(|b| format!("{b:#04X}u8")).collect();
    let inner = byte_strs.join(", ");
    let code = format!("{{ const BYTES: &[u8] = &[{inner}]; BYTES }}");
    code.parse().expect("generated code should parse")
}

fn bytes_to_array_expr(bytes: &[u8]) -> TokenStream {
    let len = bytes.len();
    let byte_strs: Vec<String> = bytes.iter().map(|b| format!("{b:#04X}u8")).collect();
    let inner = byte_strs.join(", ");
    let code = format!("{{ const BYTES: [u8; {len}] = [{inner}]; BYTES }}");
    code.parse().expect("generated code should parse")
}

// ─── Minimal syn-free error type ─────────────────────────────────────────────
// We avoid the `syn` dependency entirely for fast compile times — the macro
// input is simple enough to parse manually from `proc_macro::TokenStream`.

mod syn_free {
    use proc_macro::{Span, TokenStream};

    pub struct Error {
        message: String,
        span: Option<Span>,
    }

    impl Error {
        pub fn new(msg: &str) -> Self {
            Self {
                message: msg.to_string(),
                span: None,
            }
        }

        pub fn with_span(span: Span, msg: &str) -> Self {
            Self {
                message: msg.to_string(),
                span: Some(span),
            }
        }

        pub fn into_compile_error(self) -> TokenStream {
            let msg = self.message.replace('"', "\\\"");
            let code = format!("compile_error!(\"{msg}\")");
            // Try to set the span for better diagnostics
            if let Some(span) = self.span {
                let ts: TokenStream = code.parse().unwrap();
                ts.into_iter()
                    .map(|mut tt| {
                        tt.set_span(span);
                        tt
                    })
                    .collect()
            } else {
                code.parse().unwrap()
            }
        }
    }
}
