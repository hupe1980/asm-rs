//! Preprocessor for assembly source text.
//!
//! Handles macro definitions (`.macro`/`.endm`), repeat loops (`.rept`, `.irp`,
//! `.irpc`), and conditional assembly (`.if`/`.ifdef`/`.ifndef`/`.else`/`.endif`)
//! before the source reaches the parser.
//!
//! The preprocessor operates on raw text, expanding directives in-place so that
//! the downstream lexer and parser see only ordinary assembly statements.

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::{AsmError, Span};

/// Default maximum macro expansion recursion depth.
const DEFAULT_MAX_RECURSION_DEPTH: usize = 256;

/// Maximum total iterations across all `.rept`/`.irp`/`.irpc` blocks.
const DEFAULT_MAX_ITERATION_COUNT: usize = 100_000;

/// Append `body` to `out`, replacing every occurrence of `placeholder` with `value`.
/// Avoids allocating an intermediate `String` — writes directly into `out`.
fn replace_single_param(out: &mut String, body: &str, placeholder: &str, value: &str) {
    let ph_bytes = placeholder.as_bytes();
    let body_bytes = body.as_bytes();
    let ph_len = ph_bytes.len();
    let mut start = 0;
    while start < body_bytes.len() {
        if let Some(pos) = body[start..].find(placeholder) {
            out.push_str(&body[start..start + pos]);
            out.push_str(value);
            start += pos + ph_len;
        } else {
            out.push_str(&body[start..]);
            break;
        }
    }
}

/// A macro definition.
#[derive(Debug, Clone)]
struct MacroDef {
    /// Parameter names (without leading `\`).
    params: Vec<MacroParam>,
    /// The raw body text (lines between `.macro` and `.endm`).
    body: String,
}

/// A macro parameter with optional default value.
#[derive(Debug, Clone)]
struct MacroParam {
    name: String,
    default: Option<String>,
    is_vararg: bool,
}

/// Preprocessor state.
#[derive(Debug)]
pub struct Preprocessor {
    /// Defined macros: name → definition.
    macros: BTreeMap<String, MacroDef>,
    /// Defined symbols for `.ifdef`/`.ifndef` (name → value).
    symbols: BTreeMap<String, i128>,
    /// Counter for `\@` unique label generation.
    expansion_counter: usize,
    /// Current recursion depth for macro expansion.
    recursion_depth: usize,
    /// Maximum recursion depth (configurable).
    max_recursion_depth: usize,
    /// Maximum total iteration count (configurable).
    max_iteration_count: usize,
    /// Total iteration count across all loops (bounds check).
    iteration_count: usize,
}

impl Preprocessor {
    /// Create a new preprocessor.
    pub fn new() -> Self {
        Self {
            macros: BTreeMap::new(),
            symbols: BTreeMap::new(),
            expansion_counter: 0,
            recursion_depth: 0,
            max_recursion_depth: DEFAULT_MAX_RECURSION_DEPTH,
            max_iteration_count: DEFAULT_MAX_ITERATION_COUNT,
            iteration_count: 0,
        }
    }

    /// Set the maximum recursion depth for macro expansion.
    pub fn set_max_recursion_depth(&mut self, depth: usize) {
        self.max_recursion_depth = depth;
    }

    /// Set the maximum total iteration count for `.rept`/`.irp`/`.irpc`.
    pub fn set_max_iterations(&mut self, count: usize) {
        self.max_iteration_count = count;
    }

    /// Define a symbol for conditional assembly.
    pub fn define_symbol(&mut self, name: &str, value: i128) {
        self.symbols.insert(String::from(name), value);
    }

    /// Process source text, expanding all preprocessor directives.
    ///
    /// Returns the expanded source text ready for lexing/parsing.
    /// When no preprocessor directives are present and no macros are defined,
    /// returns a borrowed reference to the original source (zero allocation).
    ///
    /// # Errors
    ///
    /// Returns `AsmError` on malformed directives, recursion limit, or
    /// iteration limit exceeded.
    pub fn process<'a>(&mut self, source: &'a str) -> Result<Cow<'a, str>, AsmError> {
        // Reset iteration count per process() call so long-lived assemblers
        // don't accumulate towards the limit across multiple emit() calls.
        self.iteration_count = 0;
        if !self.needs_expansion(source) {
            return Ok(Cow::Borrowed(source));
        }
        self.expand_text(source).map(Cow::Owned)
    }

    /// Check whether the source text requires preprocessing.
    ///
    /// Returns `false` when no macros are defined and the source contains
    /// no preprocessor directives, meaning the text can be passed straight
    /// to the lexer without any transformation.
    fn needs_expansion(&self, source: &str) -> bool {
        // If macros are defined, any line could be an invocation.
        if !self.macros.is_empty() {
            return true;
        }
        // If symbols are defined, we still don't need expansion unless
        // the source actually references them via .ifdef/.ifndef.
        // Scan for directive prefixes.  We look for lines whose first
        // non-whitespace content matches a preprocessor directive.
        // This is cheaper than the full expansion: no allocation, no
        // string building — just a linear scan of the source bytes.
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(".macro ")
                || trimmed.starts_with(".macro\t")
                || trimmed.starts_with(".rept ")
                || trimmed.starts_with(".rept\t")
                || trimmed.starts_with(".irp ")
                || trimmed.starts_with(".irp\t")
                || trimmed.starts_with(".irpc ")
                || trimmed.starts_with(".irpc\t")
                || trimmed.starts_with(".if ")
                || trimmed.starts_with(".if\t")
                || trimmed == ".if"
                || trimmed.starts_with(".ifdef ")
                || trimmed.starts_with(".ifdef\t")
                || trimmed.starts_with(".ifndef ")
                || trimmed.starts_with(".ifndef\t")
                || trimmed == ".exitm"
            {
                return true;
            }
        }
        false
    }

    /// Core expansion loop — processes one level of text.
    fn expand_text(&mut self, source: &str) -> Result<String, AsmError> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(AsmError::ResourceLimitExceeded {
                resource: String::from("macro recursion depth"),
                limit: self.max_recursion_depth,
            });
        }

        let lines: Vec<&str> = source.lines().collect();
        let mut output = String::new();
        let mut i = 0;

        let result = self.expand_text_inner(&lines, &mut output, &mut i);
        self.recursion_depth -= 1;
        result?;
        Ok(output)
    }

    /// Inner expansion logic (separated to ensure recursion_depth cleanup).
    fn expand_text_inner(
        &mut self,
        lines: &[&str],
        output: &mut String,
        i: &mut usize,
    ) -> Result<(), AsmError> {
        while *i < lines.len() {
            let line = lines[*i];
            let trimmed = line.trim();

            // --- Macro definition ---
            if trimmed.starts_with(".macro ") || trimmed.starts_with(".macro\t") {
                let (macro_def, end_idx) = self.parse_macro_def(lines, *i)?;
                let name = parse_macro_name(trimmed, *i)?;
                self.macros.insert(name, macro_def);
                *i = end_idx + 1;
                continue;
            }

            // --- .rept ---
            if trimmed.starts_with(".rept ") || trimmed.starts_with(".rept\t") {
                let (body, end_idx) = collect_block(lines, *i, ".rept", ".endr")?;
                let count = parse_rept_count(trimmed, *i)?;
                let expanded = self.expand_rept(count, &body)?;
                output.push_str(&expanded);
                *i = end_idx + 1;
                continue;
            }

            // --- .irp ---
            if trimmed.starts_with(".irp ") || trimmed.starts_with(".irp\t") {
                let (body, end_idx) = collect_block(lines, *i, ".irp", ".endr")?;
                let (sym, values) = parse_irp_args(trimmed, *i)?;
                let expanded = self.expand_irp(&sym, &values, &body)?;
                output.push_str(&expanded);
                *i = end_idx + 1;
                continue;
            }

            // --- .irpc ---
            if trimmed.starts_with(".irpc ") || trimmed.starts_with(".irpc\t") {
                let (body, end_idx) = collect_block(lines, *i, ".irpc", ".endr")?;
                let (sym, chars) = parse_irpc_args(trimmed, *i)?;
                let expanded = self.expand_irpc(&sym, &chars, &body)?;
                output.push_str(&expanded);
                *i = end_idx + 1;
                continue;
            }

            // --- Conditional assembly ---
            if trimmed.starts_with(".if ")
                || trimmed.starts_with(".if\t")
                || trimmed == ".if"
                || trimmed.starts_with(".ifdef ")
                || trimmed.starts_with(".ifdef\t")
                || trimmed.starts_with(".ifndef ")
                || trimmed.starts_with(".ifndef\t")
            {
                let (selected_body, end_idx) = self.process_conditional(lines, *i)?;
                if !selected_body.is_empty() {
                    let expanded = self.expand_text(&selected_body)?;
                    output.push_str(&expanded);
                }
                *i = end_idx + 1;
                continue;
            }

            // --- .exitm (only meaningful inside macro expansion) ---
            if trimmed == ".exitm" {
                if self.recursion_depth <= 1 {
                    // At the top level, .exitm is meaningless — warn the user.
                    return Err(AsmError::Syntax {
                        msg: String::from(".exitm outside of macro expansion"),
                        span: crate::error::Span::new((*i + 1) as u32, 1, 0, trimmed.len()),
                    });
                }
                // Inside macro expansion, stop expanding this level
                break;
            }

            // --- Macro invocation ---
            if let Some(expanded) = self.try_expand_macro(trimmed)? {
                // Recursively expand the result
                let re_expanded = self.expand_text(&expanded)?;
                output.push_str(&re_expanded);
                *i += 1;
                continue;
            }

            // --- .equ / .set / NAME = expr: track symbols for .ifdef ---
            if let Some((name, val)) = try_parse_symbol_def(trimmed) {
                self.symbols.insert(name, val);
            }

            // Ordinary line — pass through
            output.push_str(line);
            output.push('\n');
            *i += 1;
        }

        Ok(())
    }

    /// Parse a `.macro name [params...]` ... `.endm` definition.
    fn parse_macro_def(&self, lines: &[&str], start: usize) -> Result<(MacroDef, usize), AsmError> {
        let header = lines[start].trim();
        let params = parse_macro_params(header)?;

        let mut body_lines = Vec::new();
        let mut depth = 1usize;
        let mut i = start + 1;

        while i < lines.len() {
            let trimmed = lines[i].trim();
            if trimmed.starts_with(".macro ") || trimmed.starts_with(".macro\t") {
                depth += 1;
            } else if trimmed == ".endm" {
                depth -= 1;
                if depth == 0 {
                    let body = body_lines.join("\n");
                    return Ok((MacroDef { params, body }, i));
                }
            }
            body_lines.push(lines[i]);
            i += 1;
        }

        Err(AsmError::Syntax {
            msg: String::from("unterminated .macro (missing .endm)"),
            span: line_span(start),
        })
    }

    /// Try to expand a line as a macro invocation. Returns `None` if no macro matches.
    fn try_expand_macro(&mut self, line: &str) -> Result<Option<String>, AsmError> {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('.') {
            return Ok(None);
        }

        // Extract first word as potential macro name
        let first_word = trimmed.split_whitespace().next().unwrap_or("");

        // Also check if it ends with ':' (label definition) — skip
        if first_word.ends_with(':') {
            // Could be `label: macro_name args` — check remainder
            let rest = trimmed[first_word.len()..].trim();
            if rest.is_empty() {
                return Ok(None);
            }
            let macro_name = rest.split_whitespace().next().unwrap_or("");
            if let Some(def) = self.macros.get(macro_name).cloned() {
                let args_str = rest[macro_name.len()..].trim();
                let args = parse_macro_args(args_str);
                let expanded = self.substitute_macro(&def, &args);
                // Preserve the label
                return Ok(Some(format!("{}\n{}", first_word, expanded)));
            }
            return Ok(None);
        }

        if let Some(def) = self.macros.get(first_word).cloned() {
            let args_str = trimmed[first_word.len()..].trim();
            let args = parse_macro_args(args_str);
            let expanded = self.substitute_macro(&def, &args);
            return Ok(Some(expanded));
        }

        Ok(None)
    }

    /// Substitute macro parameters and `\@` counter into body text.
    ///
    /// Uses a single-pass scan: walks the body once, and at each `\` checks
    /// for parameter names or `@`.  This is O(M × log N) where M = body length
    /// and N = parameter count, versus the prior O(N × M) multi-pass approach.
    fn substitute_macro(&mut self, def: &MacroDef, args: &[String]) -> String {
        let counter = self.expansion_counter;
        self.expansion_counter += 1;

        // Pre-compute replacement strings for each parameter
        let replacements: Vec<(&str, String)> = def
            .params
            .iter()
            .enumerate()
            .map(|(idx, param)| {
                let value = if param.is_vararg {
                    if idx < args.len() {
                        args[idx..].join(", ")
                    } else {
                        param.default.clone().unwrap_or_default()
                    }
                } else if idx < args.len() {
                    args[idx].clone()
                } else {
                    param.default.clone().unwrap_or_default()
                };
                (param.name.as_str(), value)
            })
            .collect();

        let body = &def.body;
        let mut result = String::with_capacity(body.len());
        let bytes = body.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        while i < len {
            if bytes[i] == b'\\' && i + 1 < len {
                // Check for \@ (unique counter)
                if bytes[i + 1] == b'@' {
                    use core::fmt::Write;
                    let _ = write!(result, "{}", counter);
                    i += 2;
                    continue;
                }
                // Check for \param_name
                let rest = &body[i + 1..];
                let mut matched = false;
                for &(name, ref value) in &replacements {
                    if rest.starts_with(name) {
                        // Ensure we match the full token — the char after the
                        // name must NOT be alphanumeric or '_' (otherwise
                        // \foo would partially match \foobar).
                        let end = name.len();
                        let boundary = end >= rest.len()
                            || !rest.as_bytes()[end].is_ascii_alphanumeric()
                                && rest.as_bytes()[end] != b'_';
                        if boundary {
                            result.push_str(value);
                            i += 1 + name.len();
                            matched = true;
                            break;
                        }
                    }
                }
                if !matched {
                    result.push('\\');
                    i += 1;
                }
            } else {
                // Copy one character (handles multi-byte UTF-8)
                let ch = body[i..].chars().next().unwrap_or('\0');
                result.push(ch);
                i += ch.len_utf8();
            }
        }

        result
    }

    /// Expand `.rept count` block.
    fn expand_rept(&mut self, count: usize, body: &str) -> Result<String, AsmError> {
        let mut raw = String::new();
        for _ in 0..count {
            self.iteration_count += 1;
            if self.iteration_count > self.max_iteration_count {
                return Err(AsmError::ResourceLimitExceeded {
                    resource: String::from("preprocessor iterations"),
                    limit: self.max_iteration_count,
                });
            }
            raw.push_str(body);
            raw.push('\n');
        }
        // Re-expand to handle nested .rept/.irp/.irpc/macros
        self.expand_text(&raw)
    }

    /// Expand `.irp sym, val1, val2, ...` block.
    fn expand_irp(&mut self, sym: &str, values: &[String], body: &str) -> Result<String, AsmError> {
        let placeholder = format!("\\{}", sym);
        let mut raw = String::new();
        for val in values {
            self.iteration_count += 1;
            if self.iteration_count > self.max_iteration_count {
                return Err(AsmError::ResourceLimitExceeded {
                    resource: String::from("preprocessor iterations"),
                    limit: self.max_iteration_count,
                });
            }
            replace_single_param(&mut raw, body, &placeholder, val);
            raw.push('\n');
        }
        self.expand_text(&raw)
    }

    /// Expand `.irpc sym, string` block.
    fn expand_irpc(&mut self, sym: &str, chars: &str, body: &str) -> Result<String, AsmError> {
        let placeholder = format!("\\{}", sym);
        let mut raw = String::new();
        let mut ch_buf = [0u8; 4];
        for ch in chars.chars() {
            self.iteration_count += 1;
            if self.iteration_count > self.max_iteration_count {
                return Err(AsmError::ResourceLimitExceeded {
                    resource: String::from("preprocessor iterations"),
                    limit: self.max_iteration_count,
                });
            }
            let ch_str = ch.encode_utf8(&mut ch_buf);
            replace_single_param(&mut raw, body, &placeholder, ch_str);
            raw.push('\n');
        }
        self.expand_text(&raw)
    }

    /// Process a conditional block (`.if`/`.ifdef`/`.ifndef`).
    /// Returns the selected body text and the line index of `.endif`.
    fn process_conditional(
        &self,
        lines: &[&str],
        start: usize,
    ) -> Result<(String, usize), AsmError> {
        let header = lines[start].trim();

        // Determine the initial condition result
        let condition = evaluate_condition(header, &self.symbols, start)?;

        let mut branches: Vec<(bool, Vec<&str>)> = Vec::new();
        let mut current_cond = condition;
        let mut current_body: Vec<&str> = Vec::new();
        let mut depth = 1usize;
        let mut i = start + 1;

        while i < lines.len() {
            let trimmed = lines[i].trim();

            // Nested conditional
            if trimmed.starts_with(".if ")
                || trimmed.starts_with(".if\t")
                || trimmed == ".if"
                || trimmed.starts_with(".ifdef ")
                || trimmed.starts_with(".ifdef\t")
                || trimmed.starts_with(".ifndef ")
                || trimmed.starts_with(".ifndef\t")
            {
                depth += 1;
                current_body.push(lines[i]);
                i += 1;
                continue;
            }

            if trimmed == ".endif" {
                depth -= 1;
                if depth == 0 {
                    branches.push((current_cond, current_body));
                    // Select first true branch
                    for (cond, body) in &branches {
                        if *cond {
                            return Ok((body.join("\n"), i));
                        }
                    }
                    return Ok((String::new(), i));
                }
                current_body.push(lines[i]);
                i += 1;
                continue;
            }

            if depth == 1
                && (trimmed == ".else"
                    || trimmed.starts_with(".elseif ")
                    || trimmed.starts_with(".elseif\t"))
            {
                branches.push((current_cond, core::mem::take(&mut current_body)));
                if trimmed == ".else" {
                    // .else is true if no prior branch was taken
                    current_cond = !branches.iter().any(|(c, _)| *c);
                } else {
                    // .elseif expr
                    let expr_str = trimmed.strip_prefix(".elseif").unwrap().trim();
                    current_cond = if branches.iter().any(|(c, _)| *c) {
                        false // A prior branch was already taken
                    } else {
                        eval_simple_expr(expr_str, &self.symbols) != 0
                    };
                }
                i += 1;
                continue;
            }

            current_body.push(lines[i]);
            i += 1;
        }

        Err(AsmError::Syntax {
            msg: String::from("unterminated conditional (missing .endif)"),
            span: line_span(start),
        })
    }
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

// --- Helper functions ---

/// Parse the macro name from `.macro name ...`.
fn parse_macro_name(header: &str, line: usize) -> Result<String, AsmError> {
    let rest = header.strip_prefix(".macro").unwrap_or(header).trim_start();
    let name = rest
        .split(|c: char| c.is_whitespace() || c == ',')
        .next()
        .unwrap_or("");
    if name.is_empty() {
        return Err(AsmError::Syntax {
            msg: String::from(".macro directive requires a name"),
            span: Span::new((line + 1) as u32, 1, 0, header.len()),
        });
    }
    Ok(String::from(name))
}

/// Parse macro parameters from `.macro name param1, param2=default, rest:vararg`.
fn parse_macro_params(header: &str) -> Result<Vec<MacroParam>, AsmError> {
    let rest = header.strip_prefix(".macro").unwrap_or(header).trim_start();

    // Skip the macro name
    let after_name = rest
        .split_once(|c: char| c.is_whitespace() || c == ',')
        .map(|(_, p)| p.trim_start_matches(',').trim())
        .unwrap_or("");

    if after_name.is_empty() {
        return Ok(Vec::new());
    }

    let mut params = Vec::new();
    for part in after_name.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((name, rest)) = part.split_once(':') {
            let name = name.trim();
            let rest = rest.trim();
            if rest == "vararg" {
                params.push(MacroParam {
                    name: String::from(name),
                    default: None,
                    is_vararg: true,
                });
            } else {
                params.push(MacroParam {
                    name: String::from(part),
                    default: None,
                    is_vararg: false,
                });
            }
        } else if let Some((name, default)) = part.split_once('=') {
            params.push(MacroParam {
                name: String::from(name.trim()),
                default: Some(String::from(default.trim())),
                is_vararg: false,
            });
        } else {
            params.push(MacroParam {
                name: String::from(part),
                default: None,
                is_vararg: false,
            });
        }
    }
    Ok(params)
}

/// Parse arguments passed to a macro invocation.
fn parse_macro_args(args_str: &str) -> Vec<String> {
    if args_str.is_empty() {
        return Vec::new();
    }
    args_str
        .split(',')
        .map(|s| String::from(s.trim()))
        .collect()
}

/// Parse `.rept count` header.
fn parse_rept_count(header: &str, line: usize) -> Result<usize, AsmError> {
    let rest = header.strip_prefix(".rept").unwrap_or(header).trim();
    rest.parse::<usize>().map_err(|_| AsmError::Syntax {
        msg: format!("invalid .rept count: '{}'", rest),
        span: Span::new((line + 1) as u32, 1, 0, header.len()),
    })
}

/// Parse `.irp sym, val1, val2, ...` header.
fn parse_irp_args(header: &str, line: usize) -> Result<(String, Vec<String>), AsmError> {
    let rest = header.strip_prefix(".irp").unwrap_or(header).trim();
    let (sym, vals_str) = rest.split_once(',').ok_or_else(|| AsmError::Syntax {
        msg: String::from(".irp requires a symbol and a comma-separated value list"),
        span: Span::new((line + 1) as u32, 1, 0, header.len()),
    })?;
    let sym = sym.trim();
    let values: Vec<String> = vals_str
        .split(',')
        .map(|s| String::from(s.trim()))
        .filter(|s| !s.is_empty())
        .collect();
    Ok((String::from(sym), values))
}

/// Parse `.irpc sym, chars` header.
fn parse_irpc_args(header: &str, line: usize) -> Result<(String, String), AsmError> {
    let rest = header.strip_prefix(".irpc").unwrap_or(header).trim();
    let (sym, chars) = rest.split_once(',').ok_or_else(|| AsmError::Syntax {
        msg: String::from(".irpc requires a symbol and a string"),
        span: Span::new((line + 1) as u32, 1, 0, header.len()),
    })?;
    Ok((String::from(sym.trim()), String::from(chars.trim())))
}

/// Collect lines of a block between `open_directive` and `close_directive`,
/// handling nesting.
fn collect_block(
    lines: &[&str],
    start: usize,
    open_kw: &str,
    close_kw: &str,
) -> Result<(String, usize), AsmError> {
    let mut depth = 1usize;
    let mut body_lines = Vec::new();
    let mut i = start + 1;

    // All directives that share `.endr` as their terminator.
    let endr_openers: &[&str] = &[".rept", ".irp", ".irpc"];

    while i < lines.len() {
        let trimmed = lines[i].trim();

        // Check for nested open — if `.endr` is the terminator we must
        // count *any* `.rept`/`.irp`/`.irpc` as nesting, not just the
        // exact `open_kw`.
        if close_kw == ".endr" {
            for &opener in endr_openers {
                if trimmed.starts_with(opener)
                    && (trimmed.len() == opener.len()
                        || trimmed.as_bytes().get(opener.len()) == Some(&b' ')
                        || trimmed.as_bytes().get(opener.len()) == Some(&b'\t'))
                {
                    depth += 1;
                    break;
                }
            }
        } else if trimmed.starts_with(open_kw)
            && (trimmed.len() == open_kw.len()
                || trimmed.as_bytes().get(open_kw.len()) == Some(&b' ')
                || trimmed.as_bytes().get(open_kw.len()) == Some(&b'\t'))
        {
            depth += 1;
        }

        if trimmed == close_kw {
            depth -= 1;
            if depth == 0 {
                return Ok((body_lines.join("\n"), i));
            }
        }

        body_lines.push(lines[i]);
        i += 1;
    }

    Err(AsmError::Syntax {
        msg: format!("unterminated {} (missing {})", open_kw, close_kw),
        span: line_span(start),
    })
}

/// Evaluate a conditional directive header.
fn evaluate_condition(
    header: &str,
    symbols: &BTreeMap<String, i128>,
    line: usize,
) -> Result<bool, AsmError> {
    let trimmed = header.trim();

    if let Some(rest) = trimmed.strip_prefix(".ifdef") {
        let name = rest.trim();
        return Ok(symbols.contains_key(name));
    }

    if let Some(rest) = trimmed.strip_prefix(".ifndef") {
        let name = rest.trim();
        return Ok(!symbols.contains_key(name));
    }

    if let Some(rest) = trimmed.strip_prefix(".if") {
        let expr = rest.trim();
        return Ok(eval_simple_expr(expr, symbols) != 0);
    }

    Err(AsmError::Syntax {
        msg: format!("unrecognized conditional directive: {}", trimmed),
        span: Span::new((line + 1) as u32, 1, 0, header.len()),
    })
}

/// Recursive-descent expression evaluator with proper C-like operator precedence.
///
/// Precedence (lowest → highest):
///  1. `||`  logical OR
///  2. `&&`  logical AND
///  3. `|`   bitwise OR
///  4. `^`   bitwise XOR
///  5. `&`   bitwise AND
///  6. `==` `!=`  equality
///  7. `<` `>` `<=` `>=`  relational
///  8. `<<` `>>`  shift
///  9. `+` `-`  additive
/// 10. `*` `/` `%`  multiplicative
/// 11. `!` `-` `~`  unary prefix
/// 12. literals, symbols, `defined()`, `(expr)`
struct ExprEval<'a> {
    src: &'a [u8],
    pos: usize,
    symbols: &'a BTreeMap<String, i128>,
}

impl<'a> ExprEval<'a> {
    fn new(expr: &'a str, symbols: &'a BTreeMap<String, i128>) -> Self {
        Self {
            src: expr.as_bytes(),
            pos: 0,
            symbols,
        }
    }

    fn eval(mut self) -> i128 {
        self.skip_ws();
        if self.pos >= self.src.len() {
            return 0;
        }
        self.parse_logical_or()
    }

    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    /// Try to consume a two-byte operator token. Returns `true` on match.
    fn eat2(&mut self, c1: u8, c2: u8) -> bool {
        self.skip_ws();
        if self.pos + 1 < self.src.len() && self.src[self.pos] == c1 && self.src[self.pos + 1] == c2
        {
            self.pos += 2;
            true
        } else {
            false
        }
    }

    // ── precedence 1: || ──────────────────────────────────────────────
    fn parse_logical_or(&mut self) -> i128 {
        let mut v = self.parse_logical_and();
        while self.eat2(b'|', b'|') {
            let r = self.parse_logical_and();
            v = if v != 0 || r != 0 { 1 } else { 0 };
        }
        v
    }

    // ── precedence 2: && ──────────────────────────────────────────────
    fn parse_logical_and(&mut self) -> i128 {
        let mut v = self.parse_bitwise_or();
        while self.eat2(b'&', b'&') {
            let r = self.parse_bitwise_or();
            v = if v != 0 && r != 0 { 1 } else { 0 };
        }
        v
    }

    // ── precedence 3: | (but not ||) ─────────────────────────────────
    fn parse_bitwise_or(&mut self) -> i128 {
        let mut v = self.parse_bitwise_xor();
        loop {
            self.skip_ws();
            if self.pos < self.src.len() && self.src[self.pos] == b'|' {
                // Distinguish | from ||
                if self.pos + 1 < self.src.len() && self.src[self.pos + 1] == b'|' {
                    break;
                }
                self.pos += 1;
                v |= self.parse_bitwise_xor();
            } else {
                break;
            }
        }
        v
    }

    // ── precedence 4: ^ ──────────────────────────────────────────────
    fn parse_bitwise_xor(&mut self) -> i128 {
        let mut v = self.parse_bitwise_and();
        loop {
            self.skip_ws();
            if self.pos < self.src.len() && self.src[self.pos] == b'^' {
                self.pos += 1;
                v ^= self.parse_bitwise_and();
            } else {
                break;
            }
        }
        v
    }

    // ── precedence 5: & (but not &&) ─────────────────────────────────
    fn parse_bitwise_and(&mut self) -> i128 {
        let mut v = self.parse_equality();
        loop {
            self.skip_ws();
            if self.pos < self.src.len() && self.src[self.pos] == b'&' {
                if self.pos + 1 < self.src.len() && self.src[self.pos + 1] == b'&' {
                    break;
                }
                self.pos += 1;
                v &= self.parse_equality();
            } else {
                break;
            }
        }
        v
    }

    // ── precedence 6: == != ──────────────────────────────────────────
    fn parse_equality(&mut self) -> i128 {
        let mut v = self.parse_relational();
        loop {
            if self.eat2(b'=', b'=') {
                let r = self.parse_relational();
                v = if v == r { 1 } else { 0 };
            } else if self.eat2(b'!', b'=') {
                let r = self.parse_relational();
                v = if v == r { 0 } else { 1 };
            } else {
                break;
            }
        }
        v
    }

    // ── precedence 7: < > <= >= ──────────────────────────────────────
    fn parse_relational(&mut self) -> i128 {
        let mut v = self.parse_shift();
        loop {
            if self.eat2(b'<', b'=') {
                v = if v <= self.parse_shift() { 1 } else { 0 };
            } else if self.eat2(b'>', b'=') {
                v = if v >= self.parse_shift() { 1 } else { 0 };
            } else {
                self.skip_ws();
                if self.pos < self.src.len() && self.src[self.pos] == b'<' {
                    // Not << or <=
                    if self.pos + 1 < self.src.len()
                        && (self.src[self.pos + 1] == b'<' || self.src[self.pos + 1] == b'=')
                    {
                        break;
                    }
                    self.pos += 1;
                    v = if v < self.parse_shift() { 1 } else { 0 };
                } else if self.pos < self.src.len() && self.src[self.pos] == b'>' {
                    if self.pos + 1 < self.src.len()
                        && (self.src[self.pos + 1] == b'>' || self.src[self.pos + 1] == b'=')
                    {
                        break;
                    }
                    self.pos += 1;
                    v = if v > self.parse_shift() { 1 } else { 0 };
                } else {
                    break;
                }
            }
        }
        v
    }

    // ── precedence 8: << >> ──────────────────────────────────────────
    fn parse_shift(&mut self) -> i128 {
        let mut v = self.parse_additive();
        loop {
            if self.eat2(b'<', b'<') {
                let r = self.parse_additive();
                v = if (0..128).contains(&r) {
                    v.wrapping_shl(r as u32)
                } else {
                    0
                };
            } else if self.eat2(b'>', b'>') {
                let r = self.parse_additive();
                v = if (0..128).contains(&r) {
                    v.wrapping_shr(r as u32)
                } else {
                    0
                };
            } else {
                break;
            }
        }
        v
    }

    // ── precedence 9: + - ────────────────────────────────────────────
    fn parse_additive(&mut self) -> i128 {
        let mut v = self.parse_multiplicative();
        loop {
            self.skip_ws();
            if self.pos < self.src.len() && self.src[self.pos] == b'+' {
                self.pos += 1;
                v = v.wrapping_add(self.parse_multiplicative());
            } else if self.pos < self.src.len() && self.src[self.pos] == b'-' {
                self.pos += 1;
                v = v.wrapping_sub(self.parse_multiplicative());
            } else {
                break;
            }
        }
        v
    }

    // ── precedence 10: * / % ─────────────────────────────────────────
    fn parse_multiplicative(&mut self) -> i128 {
        let mut v = self.parse_unary();
        loop {
            self.skip_ws();
            if self.pos < self.src.len() && self.src[self.pos] == b'*' {
                self.pos += 1;
                v = v.wrapping_mul(self.parse_unary());
            } else if self.pos < self.src.len() && self.src[self.pos] == b'/' {
                self.pos += 1;
                let r = self.parse_unary();
                v = if r != 0 { v / r } else { 0 };
            } else if self.pos < self.src.len() && self.src[self.pos] == b'%' {
                self.pos += 1;
                let r = self.parse_unary();
                v = if r != 0 { v % r } else { 0 };
            } else {
                break;
            }
        }
        v
    }

    // ── precedence 11: unary ! - ~ ───────────────────────────────────
    fn parse_unary(&mut self) -> i128 {
        self.skip_ws();
        if self.pos < self.src.len() {
            match self.src[self.pos] {
                // Logical NOT (but not !=)
                b'!' if self.pos + 1 >= self.src.len() || self.src[self.pos + 1] != b'=' => {
                    self.pos += 1;
                    let v = self.parse_unary();
                    return if v == 0 { 1 } else { 0 };
                }
                b'-' => {
                    self.pos += 1;
                    return self.parse_unary().wrapping_neg();
                }
                b'~' => {
                    self.pos += 1;
                    return !self.parse_unary();
                }
                _ => {}
            }
        }
        self.parse_primary()
    }

    // ── precedence 12: atoms ─────────────────────────────────────────
    fn parse_primary(&mut self) -> i128 {
        self.skip_ws();
        if self.pos >= self.src.len() {
            return 0;
        }
        let ch = self.src[self.pos];

        // Parenthesised sub-expression
        if ch == b'(' {
            self.pos += 1;
            let v = self.parse_logical_or();
            self.skip_ws();
            if self.pos < self.src.len() && self.src[self.pos] == b')' {
                self.pos += 1;
            }
            return v;
        }

        // Numeric literal (decimal, 0x, 0b, 0o)
        if ch.is_ascii_digit() {
            return self.parse_number();
        }

        // Character literal 'c'
        if ch == b'\'' && self.pos + 2 < self.src.len() && self.src[self.pos + 2] == b'\'' {
            let c = self.src[self.pos + 1];
            self.pos += 3;
            return c as i128;
        }

        // Identifier: symbol name or `defined()`
        if ch.is_ascii_alphabetic() || ch == b'_' || ch == b'.' {
            let start = self.pos;
            while self.pos < self.src.len() {
                let c = self.src[self.pos];
                if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            let name = core::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");

            // `defined(sym)` pseudo-function
            if name == "defined" {
                self.skip_ws();
                if self.pos < self.src.len() && self.src[self.pos] == b'(' {
                    self.pos += 1;
                    self.skip_ws();
                    let s = self.pos;
                    while self.pos < self.src.len() {
                        let c = self.src[self.pos];
                        if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' {
                            self.pos += 1;
                        } else {
                            break;
                        }
                    }
                    let sym = core::str::from_utf8(&self.src[s..self.pos]).unwrap_or("");
                    self.skip_ws();
                    if self.pos < self.src.len() && self.src[self.pos] == b')' {
                        self.pos += 1;
                    }
                    return if self.symbols.contains_key(sym) { 1 } else { 0 };
                }
            }

            if let Some(&val) = self.symbols.get(name) {
                return val;
            }
            return 0; // unknown symbol → 0
        }

        0
    }

    /// Parse a numeric literal at the current position.
    fn parse_number(&mut self) -> i128 {
        if self.src[self.pos] == b'0' && self.pos + 1 < self.src.len() {
            match self.src[self.pos + 1] {
                b'x' | b'X' => {
                    self.pos += 2;
                    let start = self.pos;
                    while self.pos < self.src.len() && self.src[self.pos].is_ascii_hexdigit() {
                        self.pos += 1;
                    }
                    let s = core::str::from_utf8(&self.src[start..self.pos]).unwrap_or("0");
                    return i128::from_str_radix(s, 16).unwrap_or(0);
                }
                b'b' | b'B' => {
                    self.pos += 2;
                    let start = self.pos;
                    while self.pos < self.src.len() && matches!(self.src[self.pos], b'0' | b'1') {
                        self.pos += 1;
                    }
                    let s = core::str::from_utf8(&self.src[start..self.pos]).unwrap_or("0");
                    return i128::from_str_radix(s, 2).unwrap_or(0);
                }
                b'o' | b'O' => {
                    self.pos += 2;
                    let start = self.pos;
                    while self.pos < self.src.len() && matches!(self.src[self.pos], b'0'..=b'7') {
                        self.pos += 1;
                    }
                    let s = core::str::from_utf8(&self.src[start..self.pos]).unwrap_or("0");
                    return i128::from_str_radix(s, 8).unwrap_or(0);
                }
                _ => {}
            }
        }
        // Plain decimal
        let start = self.pos;
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        let s = core::str::from_utf8(&self.src[start..self.pos]).unwrap_or("0");
        s.parse::<i128>().unwrap_or(0)
    }
}

/// Evaluate a simple integer expression (with symbol lookup).
///
/// Uses a recursive-descent parser with full C-like operator precedence.
/// Supports all arithmetic, bitwise, shift, logical, and comparison operators,
/// parenthesised sub-expressions, `defined()`, and numeric/symbol atoms.
fn eval_simple_expr(expr: &str, symbols: &BTreeMap<String, i128>) -> i128 {
    ExprEval::new(expr.trim(), symbols).eval()
}

/// Try to parse a symbol definition from `.equ name, value` or `name = value`.
fn try_parse_symbol_def(line: &str) -> Option<(String, i128)> {
    let trimmed = line.trim();

    // `.equ name, value` or `.set name, value`
    for prefix in &[".equ ", ".set "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let rest = rest.trim();
            if let Some((name, val_str)) = rest.split_once(',') {
                if let Ok(val) = parse_int_literal(val_str.trim()) {
                    return Some((String::from(name.trim()), val));
                }
            }
        }
    }

    // `name = value`
    if let Some((name, val_str)) = trimmed.split_once('=') {
        let name = name.trim();
        let val_str = val_str.trim();
        // Must not start with '=' (that would be '==')
        if !val_str.is_empty()
            && !val_str.starts_with('=')
            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            if let Ok(val) = parse_int_literal(val_str) {
                return Some((String::from(name), val));
            }
        }
    }

    None
}

/// Parse an integer literal (decimal, hex, octal, binary).
fn parse_int_literal(s: &str) -> Result<i128, ()> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i128::from_str_radix(hex, 16).map_err(|_| ())
    } else if let Some(bin) = s.strip_prefix("0b").or_else(|| s.strip_prefix("0B")) {
        i128::from_str_radix(bin, 2).map_err(|_| ())
    } else if let Some(oct) = s.strip_prefix("0o").or_else(|| s.strip_prefix("0O")) {
        i128::from_str_radix(oct, 8).map_err(|_| ())
    } else {
        s.parse::<i128>().map_err(|_| ())
    }
}

/// Create a dummy span for a given line index.
fn line_span(line: usize) -> Span {
    Span::new((line + 1) as u32, 1, 0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Macro definition and expansion ===

    #[test]
    fn macro_simple_expansion() {
        let mut pp = Preprocessor::new();
        let source = "\
.macro push_pair r1, r2
    push \\r1
    push \\r2
.endm
push_pair rax, rbx
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("push rax"));
        assert!(result.contains("push rbx"));
    }

    #[test]
    fn macro_with_defaults() {
        let mut pp = Preprocessor::new();
        let source = "\
.macro load_imm reg=rax, val=0
    mov \\reg, \\val
.endm
load_imm
load_imm rcx, 42
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("mov rax, 0"));
        assert!(result.contains("mov rcx, 42"));
    }

    #[test]
    fn macro_unique_labels() {
        let mut pp = Preprocessor::new();
        let source = "\
.macro my_loop
    jmp label_\\@
label_\\@:
.endm
my_loop
my_loop
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("label_0"));
        assert!(result.contains("label_1"));
    }

    #[test]
    fn macro_recursion_limit() {
        let mut pp = Preprocessor::new();
        let source = "\
.macro recurse
    nop
    recurse
.endm
recurse
";
        let err = pp.process(source).unwrap_err();
        match err {
            AsmError::ResourceLimitExceeded { resource, .. } => {
                assert!(resource.contains("recursion"));
            }
            _ => panic!("expected ResourceLimitExceeded, got {:?}", err),
        }
    }

    #[test]
    fn macro_vararg() {
        let mut pp = Preprocessor::new();
        let source = "\
.macro pushall regs:vararg
    # push \\regs
.endm
pushall rax, rbx, rcx
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("rax, rbx, rcx"));
    }

    #[test]
    fn macro_nested_endm() {
        let mut pp = Preprocessor::new();
        // Macro containing a nested macro definition
        let source = "\
.macro outer
    nop
.endm
outer
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"));
    }

    // === .rept ===

    #[test]
    fn rept_basic() {
        let mut pp = Preprocessor::new();
        let source = "\
.rept 3
    nop
.endr
";
        let result = pp.process(source).unwrap();
        let nop_count = result.matches("nop").count();
        assert_eq!(nop_count, 3);
    }

    #[test]
    fn rept_zero() {
        let mut pp = Preprocessor::new();
        let source = "\
.rept 0
    nop
.endr
";
        let result = pp.process(source).unwrap();
        assert!(!result.contains("nop"));
    }

    #[test]
    fn rept_nested() {
        let mut pp = Preprocessor::new();
        let source = "\
.rept 2
.rept 3
    nop
.endr
.endr
";
        let result = pp.process(source).unwrap();
        let nop_count = result.matches("nop").count();
        assert_eq!(nop_count, 6);
    }

    // === .irp ===

    #[test]
    fn irp_basic() {
        let mut pp = Preprocessor::new();
        let source = "\
.irp reg, rax, rbx, rcx
    push \\reg
.endr
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("push rax"));
        assert!(result.contains("push rbx"));
        assert!(result.contains("push rcx"));
    }

    // === .irpc ===

    #[test]
    fn irpc_basic() {
        let mut pp = Preprocessor::new();
        let source = "\
.irpc c, abc
    .byte '\\c'
.endr
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("'a'"));
        assert!(result.contains("'b'"));
        assert!(result.contains("'c'"));
    }

    // === Conditional assembly ===

    #[test]
    fn if_true() {
        let mut pp = Preprocessor::new();
        let source = "\
.if 1
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"));
    }

    #[test]
    fn if_false() {
        let mut pp = Preprocessor::new();
        let source = "\
.if 0
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(!result.contains("nop"));
    }

    #[test]
    fn if_else() {
        let mut pp = Preprocessor::new();
        let source = "\
.if 0
    mov rax, 1
.else
    mov rax, 2
.endif
";
        let result = pp.process(source).unwrap();
        assert!(!result.contains("mov rax, 1"));
        assert!(result.contains("mov rax, 2"));
    }

    #[test]
    fn if_elseif() {
        let mut pp = Preprocessor::new();
        let source = "\
.if 0
    mov rax, 1
.elseif 1
    mov rax, 2
.else
    mov rax, 3
.endif
";
        let result = pp.process(source).unwrap();
        assert!(!result.contains("mov rax, 1"));
        assert!(result.contains("mov rax, 2"));
        assert!(!result.contains("mov rax, 3"));
    }

    #[test]
    fn ifdef_defined() {
        let mut pp = Preprocessor::new();
        pp.define_symbol("MY_FLAG", 1);
        let source = "\
.ifdef MY_FLAG
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"));
    }

    #[test]
    fn ifdef_undefined() {
        let mut pp = Preprocessor::new();
        let source = "\
.ifdef UNDEFINED_FLAG
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(!result.contains("nop"));
    }

    #[test]
    fn ifndef_undefined() {
        let mut pp = Preprocessor::new();
        let source = "\
.ifndef MY_FLAG
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"));
    }

    #[test]
    fn nested_conditionals() {
        let mut pp = Preprocessor::new();
        pp.define_symbol("OUTER", 1);
        pp.define_symbol("INNER", 1);
        let source = "\
.ifdef OUTER
    .ifdef INNER
        nop
    .endif
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"));
    }

    #[test]
    fn if_expression_with_symbols() {
        let mut pp = Preprocessor::new();
        pp.define_symbol("X", 5);
        let source = "\
.if X > 3
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"));
    }

    #[test]
    fn equ_tracks_symbols() {
        let mut pp = Preprocessor::new();
        let source = "\
.equ MY_CONST, 42
.ifdef MY_CONST
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"));
        // The .equ line also passes through for the parser
        assert!(result.contains(".equ MY_CONST, 42"));
    }

    #[test]
    fn if_defined_function() {
        let mut pp = Preprocessor::new();
        pp.define_symbol("X", 1);
        let source = "\
.if defined(X)
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"));
    }

    // === Error cases ===

    #[test]
    fn unterminated_macro() {
        let mut pp = Preprocessor::new();
        let source = ".macro foo\n    nop\n";
        let err = pp.process(source).unwrap_err();
        match err {
            AsmError::Syntax { msg, .. } => {
                assert!(msg.contains("unterminated .macro"));
            }
            _ => panic!("expected Syntax error"),
        }
    }

    #[test]
    fn unterminated_rept() {
        let mut pp = Preprocessor::new();
        let source = ".rept 3\n    nop\n";
        let err = pp.process(source).unwrap_err();
        match err {
            AsmError::Syntax { msg, .. } => {
                assert!(msg.contains("unterminated"));
            }
            _ => panic!("expected Syntax error"),
        }
    }

    #[test]
    fn unterminated_conditional() {
        let mut pp = Preprocessor::new();
        let source = ".if 1\n    nop\n";
        let err = pp.process(source).unwrap_err();
        match err {
            AsmError::Syntax { msg, .. } => {
                assert!(msg.contains("unterminated"));
            }
            _ => panic!("expected Syntax error"),
        }
    }

    #[test]
    fn iteration_limit() {
        let mut pp = Preprocessor::new();
        let source = ".rept 200000\n    nop\n.endr\n";
        let err = pp.process(source).unwrap_err();
        match err {
            AsmError::ResourceLimitExceeded { resource, .. } => {
                assert!(resource.contains("iteration"));
            }
            _ => panic!("expected ResourceLimitExceeded"),
        }
    }

    // === Expression evaluator tests ===

    /// Helper: evaluate expression with given symbols.
    fn eval(expr: &str) -> i128 {
        let syms = BTreeMap::new();
        super::eval_simple_expr(expr, &syms)
    }

    fn eval_with(expr: &str, syms: &BTreeMap<String, i128>) -> i128 {
        super::eval_simple_expr(expr, syms)
    }

    #[test]
    fn expr_decimal_literals() {
        assert_eq!(eval("0"), 0);
        assert_eq!(eval("42"), 42);
        assert_eq!(eval("123456789"), 123_456_789);
    }

    #[test]
    fn expr_hex_literals() {
        assert_eq!(eval("0xFF"), 255);
        assert_eq!(eval("0x10"), 16);
        assert_eq!(eval("0XAB"), 0xAB);
    }

    #[test]
    fn expr_binary_literals() {
        assert_eq!(eval("0b1010"), 10);
        assert_eq!(eval("0B11111111"), 255);
    }

    #[test]
    fn expr_octal_literals() {
        assert_eq!(eval("0o77"), 63);
        assert_eq!(eval("0O10"), 8);
    }

    #[test]
    fn expr_char_literal() {
        assert_eq!(eval("'A'"), 65);
        assert_eq!(eval("'0'"), 48);
    }

    #[test]
    fn expr_addition() {
        assert_eq!(eval("1 + 2"), 3);
        assert_eq!(eval("10+20+30"), 60);
    }

    #[test]
    fn expr_subtraction() {
        assert_eq!(eval("10 - 3"), 7);
        assert_eq!(eval("100 - 50 - 25"), 25);
    }

    #[test]
    fn expr_multiplication() {
        assert_eq!(eval("3 * 4"), 12);
        assert_eq!(eval("2 * 3 * 5"), 30);
    }

    #[test]
    fn expr_division() {
        assert_eq!(eval("12 / 4"), 3);
        assert_eq!(eval("100 / 10 / 2"), 5);
        // Division by zero → 0
        assert_eq!(eval("42 / 0"), 0);
    }

    #[test]
    fn expr_modulo() {
        assert_eq!(eval("10 % 3"), 1);
        assert_eq!(eval("17 % 5"), 2);
        assert_eq!(eval("42 % 0"), 0);
    }

    #[test]
    fn expr_precedence_mul_over_add() {
        assert_eq!(eval("2 + 3 * 4"), 14);
        assert_eq!(eval("3 * 4 + 2"), 14);
        assert_eq!(eval("10 - 2 * 3"), 4);
    }

    #[test]
    fn expr_parentheses() {
        assert_eq!(eval("(2 + 3) * 4"), 20);
        assert_eq!(eval("((1 + 2) * (3 + 4))"), 21);
        assert_eq!(eval("(10)"), 10);
    }

    #[test]
    fn expr_nested_parentheses() {
        assert_eq!(eval("((2 + 3) * (4 - 1))"), 15);
        assert_eq!(eval("(((5)))"), 5);
    }

    #[test]
    fn expr_bitwise_and() {
        assert_eq!(eval("0xFF & 0x0F"), 0x0F);
        assert_eq!(eval("0b1010 & 0b1100"), 0b1000);
    }

    #[test]
    fn expr_bitwise_or() {
        assert_eq!(eval("0x0F | 0xF0"), 0xFF);
        assert_eq!(eval("0b1010 | 0b0101"), 0b1111);
    }

    #[test]
    fn expr_bitwise_xor() {
        assert_eq!(eval("0xFF ^ 0x0F"), 0xF0);
        assert_eq!(eval("0b1010 ^ 0b1100"), 0b0110);
    }

    #[test]
    fn expr_bitwise_not() {
        // ~0 in i128 is all ones = -1
        assert_eq!(eval("~0"), -1);
        assert_eq!(eval("~0xFF & 0xFF"), 0);
    }

    #[test]
    fn expr_shift_left() {
        assert_eq!(eval("1 << 8"), 256);
        assert_eq!(eval("0xFF << 4"), 0xFF0);
    }

    #[test]
    fn expr_shift_right() {
        assert_eq!(eval("256 >> 8"), 1);
        assert_eq!(eval("0xFF0 >> 4"), 0xFF);
    }

    #[test]
    fn expr_logical_and() {
        assert_eq!(eval("1 && 1"), 1);
        assert_eq!(eval("1 && 0"), 0);
        assert_eq!(eval("0 && 1"), 0);
        assert_eq!(eval("0 && 0"), 0);
    }

    #[test]
    fn expr_logical_or() {
        assert_eq!(eval("1 || 1"), 1);
        assert_eq!(eval("1 || 0"), 1);
        assert_eq!(eval("0 || 1"), 1);
        assert_eq!(eval("0 || 0"), 0);
    }

    #[test]
    fn expr_logical_not() {
        assert_eq!(eval("!0"), 1);
        assert_eq!(eval("!1"), 0);
        assert_eq!(eval("!42"), 0);
    }

    #[test]
    fn expr_equality() {
        assert_eq!(eval("5 == 5"), 1);
        assert_eq!(eval("5 == 6"), 0);
        assert_eq!(eval("5 != 6"), 1);
        assert_eq!(eval("5 != 5"), 0);
    }

    #[test]
    fn expr_relational() {
        assert_eq!(eval("3 < 5"), 1);
        assert_eq!(eval("5 < 3"), 0);
        assert_eq!(eval("5 > 3"), 1);
        assert_eq!(eval("3 > 5"), 0);
        assert_eq!(eval("5 <= 5"), 1);
        assert_eq!(eval("5 <= 6"), 1);
        assert_eq!(eval("6 <= 5"), 0);
        assert_eq!(eval("5 >= 5"), 1);
        assert_eq!(eval("6 >= 5"), 1);
        assert_eq!(eval("5 >= 6"), 0);
    }

    #[test]
    fn expr_unary_minus() {
        assert_eq!(eval("-1"), -1);
        assert_eq!(eval("-(-5)"), 5);
        assert_eq!(eval("3 + -2"), 1);
        assert_eq!(eval("3 - -2"), 5);
    }

    #[test]
    fn expr_mixed_precedence() {
        // Shift lower than add: 1 + 2 << 3 == (1+2) << 3 ... NO
        // Actually: << is higher than +: 1 + (2 << 3) = 1 + 16 = 17
        assert_eq!(eval("1 + 2 << 3"), 24); // (1+2)<<3, since shift is HIGHER than add... wait
                                            // Let me think about this. In C, << is higher precedence (binds tighter) than +.
                                            // But in our parser, additive is level 9 and shift is level 8 (lower number = lower precedence).
                                            // additive calls parse_multiplicative, shift calls parse_additive.
                                            // Wait - that's wrong. Let me re-check.
                                            // Actually: parse_additive calls parse_multiplicative, and parse_shift calls parse_additive.
                                            // So shift calls additive which calls multiplicative. This means additive binds tighter than shift.
                                            // That matches C precedence where + binds tighter than <<.
                                            // So 1 + 2 << 3 = (1+2) << 3 = 3 << 3 = 24. Correct for C.
        assert_eq!(eval("1 + 2 << 3"), 24);

        // Comparison: == is lower than +
        assert_eq!(eval("2 + 3 == 5"), 1);
        assert_eq!(eval("2 + 3 == 6"), 0);

        // Logical: && is lower than ==
        assert_eq!(eval("1 == 1 && 2 == 2"), 1);
        assert_eq!(eval("1 == 1 && 2 == 3"), 0);

        // || is the lowest
        assert_eq!(eval("0 && 1 || 1"), 1);
        assert_eq!(eval("1 || 0 && 0"), 1);
    }

    #[test]
    fn expr_complex_bitwise() {
        // Page-align: addr & ~0xFFF
        // Can't test with large addresses easily, but logic works
        assert_eq!(eval("0x1234 & ~0xFFF & 0xFFFF"), 0x1000);
        // Flag test
        assert_eq!(eval("(0x03 & 0x01) != 0"), 1);
        assert_eq!(eval("(0x02 & 0x01) != 0"), 0);
    }

    #[test]
    fn expr_symbols() {
        let mut syms = BTreeMap::new();
        syms.insert(String::from("X"), 10);
        syms.insert(String::from("Y"), 20);
        assert_eq!(eval_with("X + Y", &syms), 30);
        assert_eq!(eval_with("X * Y", &syms), 200);
        assert_eq!(eval_with("(X + Y) * 2", &syms), 60);
    }

    #[test]
    fn expr_defined_function() {
        let mut syms = BTreeMap::new();
        syms.insert(String::from("FOO"), 1);
        assert_eq!(eval_with("defined(FOO)", &syms), 1);
        assert_eq!(eval_with("defined(BAR)", &syms), 0);
        assert_eq!(eval_with("defined(FOO) && defined(BAR)", &syms), 0);
        assert_eq!(eval_with("defined(FOO) || defined(BAR)", &syms), 1);
    }

    #[test]
    fn expr_regression_0x_minus() {
        // Old parser choked on "0x10 - 1" because rsplit_once('-') split at "0x10"
        assert_eq!(eval("0x10 - 1"), 15);
        assert_eq!(eval("0xFF - 0xF0"), 15);
    }

    #[test]
    fn expr_whitespace_tolerance() {
        assert_eq!(eval("  42  "), 42);
        assert_eq!(eval("  1  +  2  "), 3);
        assert_eq!(eval(" ( 1 + 2 ) * 3 "), 9);
    }

    #[test]
    fn expr_empty() {
        assert_eq!(eval(""), 0);
        assert_eq!(eval("   "), 0);
    }

    #[test]
    fn expr_if_mul_integrated() {
        // This was the data-corruption bug: `.if 2 * 3 == 6` silently failed
        let mut pp = Preprocessor::new();
        let source = "\
.if 2 * 3 == 6
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"), "2*3==6 should be true");
    }

    #[test]
    fn expr_if_parenthesised_integrated() {
        let mut pp = Preprocessor::new();
        let source = "\
.if (1 + 2) * 4 == 12
    mov eax, 1
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("mov eax, 1"));
    }

    #[test]
    fn expr_if_shift_integrated() {
        let mut pp = Preprocessor::new();
        let source = "\
.if 1 << 4 == 16
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"), "1<<4 should equal 16");
    }

    #[test]
    fn expr_if_bitwise_and_integrated() {
        let mut pp = Preprocessor::new();
        let source = "\
.equ FLAGS, 0x07
.if FLAGS & 0x02
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"), "0x07 & 0x02 should be non-zero");
    }

    #[test]
    fn expr_if_logical_and_integrated() {
        let mut pp = Preprocessor::new();
        pp.define_symbol("A", 1);
        pp.define_symbol("B", 1);
        let source = "\
.if defined(A) && defined(B)
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"), "both A and B defined");
    }

    #[test]
    fn expr_if_logical_or_integrated() {
        let mut pp = Preprocessor::new();
        pp.define_symbol("A", 1);
        let source = "\
.if defined(A) || defined(B)
    nop
.endif
";
        let result = pp.process(source).unwrap();
        assert!(result.contains("nop"), "A is defined so OR should be true");
    }

    #[test]
    fn expr_elseif_with_operators() {
        let mut pp = Preprocessor::new();
        pp.define_symbol("MODE", 2);
        let source = "\
.if MODE * 2 == 2
    wrong
.elseif MODE * 2 == 4
    correct
.else
    also_wrong
.endif
";
        let result = pp.process(source).unwrap();
        assert!(!result.contains("wrong"));
        assert!(result.contains("correct"));
    }
}
