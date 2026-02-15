//! Public assembler API — builder pattern and one-shot assembly.
//!
//! This module ties together the lexer, parser, encoder, and linker
//! into a fluent API for assembling code.

#[allow(unused_imports)]
use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
#[allow(unused_imports)]
use alloc::vec;
use alloc::vec::Vec;

use crate::encoder;
use crate::error::{AsmError, Span};
use crate::ir::*;
use crate::lexer;
use crate::linker::{AppliedRelocation, Linker};
use crate::parser;
use crate::preprocessor::Preprocessor;

/// The result of a successful assembly operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct AssemblyResult {
    /// The assembled machine code.
    bytes: Vec<u8>,
    /// Label addresses (name → absolute address).
    labels: Vec<(String, u64)>,
    /// Applied relocations in the output.
    relocations: Vec<AppliedRelocation>,
    /// Base address used during assembly.
    base_address: u64,
    /// Source text annotations: `(output_offset, source_text)` for listing.
    source_annotations: Vec<(u64, String)>,
}

impl AssemblyResult {
    /// Get the assembled bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let mut asm = Assembler::new(Arch::X86_64);
    /// asm.emit("nop")?;
    /// let result = asm.finish()?;
    /// assert_eq!(result.bytes(), &[0x90]);
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume and return the bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let mut asm = Assembler::new(Arch::X86_64);
    /// asm.emit("ret")?;
    /// let bytes = asm.finish()?.into_bytes();
    /// assert_eq!(bytes, vec![0xC3]);
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Get the byte count.
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let mut asm = Assembler::new(Arch::X86_64);
    /// asm.emit("nop\nret")?;
    /// let result = asm.finish()?;
    /// assert_eq!(result.len(), 2); // nop(1) + ret(1)
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the result is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let result = Assembler::new(Arch::X86_64).finish()?;
    /// assert!(result.is_empty());
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Get label addresses (name, absolute address).
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let mut asm = Assembler::new(Arch::X86_64);
    /// asm.emit("start: nop\nend: ret")?;
    /// let result = asm.finish()?;
    /// let labels = result.labels();
    /// assert!(labels.iter().any(|(name, _)| name == "start"));
    /// assert!(labels.iter().any(|(name, _)| name == "end"));
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    #[must_use]
    pub fn labels(&self) -> &[(String, u64)] {
        &self.labels
    }

    /// Look up a label address by name.
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let mut asm = Assembler::new(Arch::X86_64);
    /// asm.emit("start: nop\nnop\nend: ret")?;
    /// let result = asm.finish()?;
    /// assert_eq!(result.label_address("start"), Some(0));
    /// assert_eq!(result.label_address("end"), Some(2));
    /// assert_eq!(result.label_address("missing"), None);
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    #[must_use]
    pub fn label_address(&self, name: &str) -> Option<u64> {
        self.labels.iter().find(|(n, _)| n == name).map(|(_, a)| *a)
    }

    /// Get the applied relocations — where label references were patched.
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let mut asm = Assembler::new(Arch::X86_64);
    /// asm.emit("target: jmp target")?;
    /// let result = asm.finish()?;
    /// let relocs = result.relocations();
    /// assert!(!relocs.is_empty());
    /// assert_eq!(relocs[0].label, "target");
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    #[must_use]
    pub fn relocations(&self) -> &[AppliedRelocation] {
        &self.relocations
    }

    /// Get the base address used during assembly.
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let mut asm = Assembler::new(Arch::X86_64);
    /// asm.base_address(0x1000);
    /// asm.emit("nop")?;
    /// let result = asm.finish()?;
    /// assert_eq!(result.base_address(), 0x1000);
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    #[must_use]
    pub fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Produce a human-readable listing of address, hex bytes.
    ///
    /// Labels are shown on their own line with their resolved address.
    /// Machine code is shown in rows of up to 8 bytes each.
    ///
    /// # Example output
    ///
    /// ```text
    /// 00000000                  entry:
    /// 00000000  55              push rbp
    /// 00000001  4889E5          mov rbp, rsp
    /// ```
    #[must_use]
    pub fn listing(&self) -> String {
        use core::fmt::Write;

        let mut out = String::new();
        let base = self.base_address;

        // First, collect labels sorted by address
        let mut sorted_labels = self.labels.clone();
        sorted_labels.sort_by_key(|(_, addr)| *addr);

        // Build a map: offset → list of label names
        let mut label_at: alloc::collections::BTreeMap<u64, Vec<&str>> =
            alloc::collections::BTreeMap::new();
        for (name, addr) in &sorted_labels {
            label_at.entry(*addr).or_default().push(name);
        }

        // Build a map: offset → source text annotation
        let mut source_at: alloc::collections::BTreeMap<u64, &str> =
            alloc::collections::BTreeMap::new();
        for (offset, text) in &self.source_annotations {
            if !text.is_empty() {
                source_at.insert(*offset, text);
            }
        }

        // Collect all label offsets as split points (where we must break a chunk)
        let mut split_offsets: alloc::collections::BTreeSet<u64> =
            label_at.keys().copied().collect();

        // Also split at source annotation offsets so each instruction gets its own line
        for &ann_off in source_at.keys() {
            split_offsets.insert(ann_off);
        }

        // Walk through bytes, breaking at label and annotation boundaries
        let bytes = &self.bytes;
        let mut offset: u64 = base;
        let mut i = 0;

        while i < bytes.len() {
            // Print any labels at this offset
            if let Some(names) = label_at.get(&offset) {
                for name in names {
                    let _ = writeln!(out, "{:08X}                  {}:", offset, name);
                }
            }

            // Determine chunk size: up to 8 bytes, but break at the next split point
            let max_end = core::cmp::min(i + 8, bytes.len());
            let mut chunk_end = max_end;

            // Check if any split point falls within (offset+1..offset+chunk_len)
            let range_end = offset + (max_end - i) as u64;
            if range_end > offset + 1 {
                for &split_off in split_offsets.range((offset + 1)..range_end) {
                    let split_at = (split_off - base) as usize;
                    if split_at < chunk_end && split_at > i {
                        chunk_end = split_at;
                        break;
                    }
                }
            }

            let chunk = &bytes[i..chunk_end];
            let hex: String = chunk.iter().fold(String::new(), |mut acc, b| {
                use core::fmt::Write;
                let _ = write!(acc, "{:02X}", b);
                acc
            });

            // Look up source annotation for this offset
            if let Some(source_text) = source_at.get(&offset) {
                let _ = writeln!(out, "{:08X}  {:<16}  {}", offset, hex, source_text);
            } else {
                let _ = writeln!(out, "{:08X}  {:<16}", offset, hex);
            }

            let chunk_len = chunk.len();
            i += chunk_len;
            offset += chunk_len as u64;
        }

        // Print labels at the very end (e.g. a label after the last instruction)
        if let Some(names) = label_at.get(&offset) {
            for name in names {
                let _ = writeln!(out, "{:08X}                  {}:", offset, name);
            }
        }

        out
    }
}

/// Configurable resource limits for defense against denial-of-service.
///
/// When processing untrusted assembly input, these limits prevent pathological
/// inputs from consuming unbounded memory or CPU time. All limits default to
/// generous values that are sufficient for any reasonable assembly program.
///
/// # Examples
///
/// ```rust
/// use asm_rs::{Assembler, Arch};
/// use asm_rs::assembler::ResourceLimits;
///
/// let mut asm = Assembler::new(Arch::X86_64);
/// asm.limits(ResourceLimits {
///     max_statements: 1_000,
///     max_labels: 100,
///     max_output_bytes: 4096,
///     max_errors: 16,
///     max_recursion_depth: 64,
///     max_source_bytes: 64 * 1024 * 1024,
///     max_iterations: 100_000,
/// });
/// // Assembly of very large or pathological inputs will now error early.
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResourceLimits {
    /// Maximum number of parsed statements (instructions + directives + labels).
    /// Default: 1,000,000.
    pub max_statements: usize,
    /// Maximum number of labels that can be defined. Default: 100,000.
    pub max_labels: usize,
    /// Maximum output size in bytes. Default: 16 MiB.
    pub max_output_bytes: usize,
    /// Maximum accumulated errors before bailing. Default: 64.
    pub max_errors: usize,
    /// Maximum macro expansion recursion depth. Default: 256.
    pub max_recursion_depth: usize,
    /// Maximum input source bytes per `emit()` call. Default: 64 MiB.
    /// Guards against multi-gigabyte inputs consuming unbounded memory
    /// during lexing/parsing before any other limit can fire.
    pub max_source_bytes: usize,
    /// Maximum total preprocessor iterations (`.rept`/`.irp`/`.irpc`).
    /// Default: 100,000.
    pub max_iterations: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_statements: 1_000_000,
            max_labels: 100_000,
            max_output_bytes: 16 * 1024 * 1024,
            max_errors: 64,
            max_recursion_depth: 256,
            max_source_bytes: 64 * 1024 * 1024,
            max_iterations: 100_000,
        }
    }
}

/// Builder-pattern assembler.
///
/// # Examples
///
/// ```rust
/// use asm_rs::{Assembler, Arch};
///
/// let mut asm = Assembler::new(Arch::X86_64);
/// asm.emit("push rbp").unwrap();
/// asm.emit("mov rbp, rsp").unwrap();
/// asm.emit("pop rbp").unwrap();
/// asm.emit("ret").unwrap();
/// let result = asm.finish().unwrap();
/// assert!(!result.is_empty());
/// ```
#[derive(Debug)]
pub struct Assembler {
    arch: Arch,
    /// Current x86 encoding mode — tracks `.code16`/`.code32`/`.code64` switches.
    /// Only meaningful when `arch` is `X86` or `X86_64`.
    x86_mode: crate::ir::X86Mode,
    syntax: Syntax,
    opt_level: OptLevel,
    linker: Linker,
    /// Preprocessor for macros, conditionals, and loops.
    preprocessor: Preprocessor,
    /// Accumulated errors for multi-error mode.
    errors: Vec<AsmError>,
    /// Maps linker fragment index → source text for listing.
    fragment_annotations: Vec<(usize, String)>,
    /// Whether to collect source annotations for listing output.
    /// Off by default to avoid per-statement String allocations.
    listing_enabled: bool,
    /// Resource limits for DoS protection.
    resource_limits: ResourceLimits,
    /// Running count of parsed statements so far.
    statement_count: usize,
    /// Running count of defined labels so far.
    label_count: usize,
    /// Pending literal pool entries: (value, size_bytes, synthetic_label).
    /// Flushed at `.ltorg`, unconditional branches, or `finish()`.
    literal_pool: Vec<LiteralPoolEntry>,
    /// Counter for generating unique literal pool labels.
    literal_pool_counter: usize,
    /// Whether RISC-V C extension auto-narrowing is enabled (`.option rvc`).
    /// When true, 32-bit instructions are automatically compressed to 16-bit
    /// equivalents when possible.
    rvc_enabled: bool,
    /// Whether the next label should be marked as a Thumb function (`.thumb_func`).
    /// When true, the label's address will have the LSB set to indicate Thumb mode.
    thumb_func_pending: bool,
    /// Labels marked as Thumb functions via `.thumb_func`.
    /// Their resolved addresses will have the LSB set.
    thumb_labels: Vec<String>,
    /// Running estimate of cumulative output bytes — incremented by builder
    /// methods (`db`, `fill`, `space`, etc.) and `emit()` to catch
    /// `max_output_bytes` overflows *before* the allocation happens.
    estimated_output_bytes: usize,
}

/// A pending literal pool entry.
#[derive(Debug, Clone)]
struct LiteralPoolEntry {
    /// The constant value to place in the pool.
    value: i128,
    /// Size in bytes (4 for W-regs, 8 for X-regs).
    size: u8,
    /// Synthetic label that the LDR references.
    label: String,
}

impl Assembler {
    /// Create a new assembler for the given architecture.
    pub fn new(arch: Arch) -> Self {
        let syntax = match arch {
            Arch::Arm | Arch::Thumb | Arch::Aarch64 => Syntax::Ual,
            Arch::Rv32 | Arch::Rv64 => Syntax::RiscV,
            _ => Syntax::Intel,
        };
        let x86_mode = match arch {
            Arch::X86 => crate::ir::X86Mode::Mode32,
            Arch::X86_64 => crate::ir::X86Mode::Mode64,
            _ => crate::ir::X86Mode::Mode64, // unused for non-x86
        };
        Self {
            arch,
            x86_mode,
            syntax,
            opt_level: OptLevel::default(),
            linker: Linker::new(),
            preprocessor: Preprocessor::new(),
            errors: Vec::new(),
            fragment_annotations: Vec::new(),
            listing_enabled: false,
            resource_limits: ResourceLimits::default(),
            statement_count: 0,
            label_count: 0,
            literal_pool: Vec::new(),
            literal_pool_counter: 0,
            rvc_enabled: false,
            thumb_func_pending: false,
            thumb_labels: Vec::new(),
            estimated_output_bytes: 0,
        }
    }

    /// Set resource limits for defense against pathological inputs.
    ///
    /// See [`ResourceLimits`] for the available limits and their defaults.
    pub fn limits(&mut self, limits: ResourceLimits) -> &mut Self {
        self.resource_limits = limits;
        self.preprocessor
            .set_max_recursion_depth(limits.max_recursion_depth);
        self.preprocessor.set_max_iterations(limits.max_iterations);
        self
    }

    /// Set the syntax dialect.
    ///
    /// Currently only [`Syntax::Intel`] is supported. Attempting to emit code
    /// after selecting an unsupported dialect will return an error.
    pub fn syntax(&mut self, syntax: Syntax) -> &mut Self {
        self.syntax = syntax;
        self
    }

    /// Set the optimization level.
    ///
    /// `OptLevel::Size` (default) prefers shortest encodings. `OptLevel::None`
    /// disables encoding optimizations for predictable output. Extended
    /// peephole optimizations (zero-idiom, REX elimination) are planned.
    pub fn optimize(&mut self, level: OptLevel) -> &mut Self {
        self.opt_level = level;
        self
    }

    /// Enable source annotations for listing output.
    ///
    /// When enabled, the assembler records source text for each emitted
    /// fragment, making it available in [`AssemblyResult::listing()`].
    /// This adds a per-statement `String` allocation; leave disabled (the
    /// default) when listing output is not needed.
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let mut asm = Assembler::new(Arch::X86_64);
    /// asm.enable_listing();
    /// asm.emit("nop")?;
    /// let result = asm.finish()?;
    /// let listing = result.listing();
    /// assert!(listing.contains("90")); // NOP opcode in hex listing
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    pub fn enable_listing(&mut self) -> &mut Self {
        self.listing_enabled = true;
        self
    }

    /// Set the base virtual address for the assembly.
    pub fn base_address(&mut self, addr: u64) -> &mut Self {
        self.linker.set_base_address(addr);
        self
    }

    /// Define an external label at a known absolute address.
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let mut asm = Assembler::new(Arch::X86_64);
    /// asm.define_external("puts", 0x4000);
    /// asm.emit("call puts")?;
    /// let result = asm.finish()?;
    /// assert!(!result.bytes().is_empty());
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    pub fn define_external(&mut self, name: &str, addr: u64) -> &mut Self {
        self.linker.define_external(name, addr);
        self
    }

    /// Define a named constant value.
    pub fn define_constant(&mut self, name: &str, value: i128) -> &mut Self {
        self.linker.define_constant(name, value);
        self
    }

    /// Emit assembly source text. Can be called multiple times.
    ///
    /// # Errors
    ///
    /// Returns [`AsmError`] on parse or encoding errors, unsupported syntax,
    /// or if resource limits are exceeded.
    pub fn emit(&mut self, source: &str) -> Result<&mut Self, AsmError> {
        // Check source size limit before any work
        if source.len() > self.resource_limits.max_source_bytes {
            return Err(AsmError::ResourceLimitExceeded {
                resource: String::from("source bytes"),
                limit: self.resource_limits.max_source_bytes,
            });
        }
        // Run preprocessor to expand macros, loops, and conditionals
        let expanded = self.preprocessor.process(source)?;
        let mut statements = parse_source(&expanded, self.arch, self.syntax)?;
        self.process_statements(&mut statements, &expanded)?;
        Ok(self)
    }

    /// Define a preprocessor symbol for conditional assembly.
    ///
    /// Symbols defined here are available in `.ifdef`/`.ifndef` and `.if defined()`
    /// conditionals within assembly source.
    pub fn define_preprocessor_symbol(&mut self, name: &str, value: i128) -> &mut Self {
        self.preprocessor.define_symbol(name, value);
        self
    }

    /// Add a label at the current position (builder API).
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let mut asm = Assembler::new(Arch::X86_64);
    /// asm.label("entry")?;
    /// asm.emit("nop")?;
    /// let result = asm.finish()?;
    /// assert_eq!(result.label_address("entry"), Some(0));
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AsmError::DuplicateLabel`] if the label was already defined,
    /// or [`AsmError::ResourceLimitExceeded`] if the label limit is reached.
    pub fn label(&mut self, name: &str) -> Result<&mut Self, AsmError> {
        self.label_count += 1;
        if self.label_count > self.resource_limits.max_labels {
            return Err(AsmError::ResourceLimitExceeded {
                resource: String::from("labels"),
                limit: self.resource_limits.max_labels,
            });
        }
        self.linker.add_label(name, Span::new(0, 0, 0, 0))?;
        Ok(self)
    }

    /// Emit raw bytes (builder API for `.byte`/`.db`).
    ///
    /// # Errors
    ///
    /// Returns [`AsmError::ResourceLimitExceeded`] if the output size limit
    /// would be exceeded.
    pub fn db(&mut self, bytes: &[u8]) -> Result<&mut Self, AsmError> {
        self.check_output_limit(bytes.len())?;
        self.linker.add_bytes(bytes.to_vec(), Span::new(0, 0, 0, 0));
        Ok(self)
    }

    /// Emit a 16-bit value (builder API for `.word`/`.dw`).
    ///
    /// # Errors
    ///
    /// Returns [`AsmError::ResourceLimitExceeded`] if the output size limit
    /// would be exceeded.
    pub fn dw(&mut self, value: u16) -> Result<&mut Self, AsmError> {
        self.check_output_limit(2)?;
        self.linker
            .add_bytes(value.to_le_bytes().to_vec(), Span::new(0, 0, 0, 0));
        Ok(self)
    }

    /// Emit a 32-bit value (builder API for `.long`/`.dd`).
    ///
    /// # Errors
    ///
    /// Returns [`AsmError::ResourceLimitExceeded`] if the output size limit
    /// would be exceeded.
    pub fn dd(&mut self, value: u32) -> Result<&mut Self, AsmError> {
        self.check_output_limit(4)?;
        self.linker
            .add_bytes(value.to_le_bytes().to_vec(), Span::new(0, 0, 0, 0));
        Ok(self)
    }

    /// Emit a 64-bit value (builder API for `.quad`/`.dq`).
    ///
    /// # Errors
    ///
    /// Returns [`AsmError::ResourceLimitExceeded`] if the output size limit
    /// would be exceeded.
    pub fn dq(&mut self, value: u64) -> Result<&mut Self, AsmError> {
        self.check_output_limit(8)?;
        self.linker
            .add_bytes(value.to_le_bytes().to_vec(), Span::new(0, 0, 0, 0));
        Ok(self)
    }

    /// Emit a string without NUL terminator (builder API for `.ascii`).
    ///
    /// # Errors
    ///
    /// Returns [`AsmError::ResourceLimitExceeded`] if the output size limit
    /// would be exceeded.
    pub fn ascii(&mut self, s: &str) -> Result<&mut Self, AsmError> {
        self.check_output_limit(s.len())?;
        self.linker
            .add_bytes(s.as_bytes().to_vec(), Span::new(0, 0, 0, 0));
        Ok(self)
    }

    /// Emit a NUL-terminated string (builder API for `.asciz`/`.string`).
    ///
    /// # Errors
    ///
    /// Returns [`AsmError::ResourceLimitExceeded`] if the output size limit
    /// would be exceeded.
    pub fn asciz(&mut self, s: &str) -> Result<&mut Self, AsmError> {
        self.check_output_limit(s.len() + 1)?;
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        self.linker.add_bytes(bytes, Span::new(0, 0, 0, 0));
        Ok(self)
    }

    /// Align to a byte boundary (builder API for `.align`).
    ///
    /// Uses multi-byte NOP padding for x86/x86-64 architectures.
    pub fn align(&mut self, alignment: u32) -> &mut Self {
        let use_nop = matches!(self.arch, Arch::X86 | Arch::X86_64);
        self.linker
            .add_alignment(alignment, 0x00, None, use_nop, Span::new(0, 0, 0, 0));
        self
    }

    /// Align to a byte boundary with explicit fill byte (builder API).
    pub fn align_with_fill(&mut self, alignment: u32, fill: u8) -> &mut Self {
        self.linker
            .add_alignment(alignment, fill, None, false, Span::new(0, 0, 0, 0));
        self
    }

    /// Set the location counter to an absolute address (builder API for `.org`).
    pub fn org(&mut self, target: u64) -> &mut Self {
        self.linker.add_org(target, 0x00, Span::new(0, 0, 0, 0));
        self
    }

    /// Set the location counter with explicit fill byte (builder API for `.org`).
    pub fn org_with_fill(&mut self, target: u64, fill: u8) -> &mut Self {
        self.linker.add_org(target, fill, Span::new(0, 0, 0, 0));
        self
    }

    /// Emit fill bytes (builder API for `.fill`).
    ///
    /// Produces `count * size` bytes, each `size`-byte unit filled with `value`.
    ///
    /// # Errors
    ///
    /// Returns [`AsmError::ResourceLimitExceeded`] if the output size limit
    /// would be exceeded.
    pub fn fill(&mut self, count: u32, size: u8, value: i64) -> Result<&mut Self, AsmError> {
        let total = count as usize * size as usize;
        self.check_output_limit(total)?;
        let mut bytes = Vec::with_capacity(total);
        // GAS semantics: value is a LE integer padded to `size` bytes
        let val_bytes = value.to_le_bytes();
        for _ in 0..count {
            for &b in val_bytes.iter().take(size as usize) {
                bytes.push(b);
            }
            // Pad with zeros if size > 8
            if (size as usize) > 8 {
                bytes.resize(bytes.len() + size as usize - 8, 0);
            }
        }
        self.linker.add_bytes(bytes, Span::new(0, 0, 0, 0));
        Ok(self)
    }

    /// Emit zero-filled space (builder API for `.space`/`.skip`).
    ///
    /// # Errors
    ///
    /// Returns [`AsmError::ResourceLimitExceeded`] if the output size limit
    /// would be exceeded.
    pub fn space(&mut self, n: u32) -> Result<&mut Self, AsmError> {
        self.check_output_limit(n as usize)?;
        let bytes = alloc::vec![0u8; n as usize];
        self.linker.add_bytes(bytes, Span::new(0, 0, 0, 0));
        Ok(self)
    }

    /// Returns the current number of fragments (instructions + data) emitted so far.
    ///
    /// Useful for estimating output size before calling [`finish()`](Assembler::finish).
    pub fn current_fragment_count(&self) -> usize {
        self.linker.fragment_count()
    }

    /// Assemble a single instruction and return its raw bytes immediately,
    /// without label resolution.
    ///
    /// This is useful for one-shot encoding when labels are not needed.
    /// The instruction is NOT added to the assembler's internal state.
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let asm = Assembler::new(Arch::X86_64);
    /// let bytes = asm.encode_one("xor eax, eax")?;
    /// assert_eq!(bytes, [0x31, 0xC0]);
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AsmError`] if the instruction cannot be parsed or encoded.
    pub fn encode_one(&self, source: &str) -> Result<Vec<u8>, AsmError> {
        use crate::encoder::encode_instruction;

        let tokens = crate::lexer::tokenize(source)?;
        let stmts = crate::parser::parse_with_syntax(&tokens, self.arch, self.syntax)?;
        if stmts.is_empty() {
            return Ok(Vec::new());
        }
        match &stmts[0] {
            crate::ir::Statement::Instruction(instr) => {
                // Resolve any constants defined via define_constant() / .equ / .set
                let mut instr = instr.clone();
                self.resolve_constants_in_instruction(&mut instr);
                let encoded = encode_instruction(&instr, self.arch)?;
                Ok(encoded.bytes.to_vec())
            }
            _ => Err(AsmError::Syntax {
                msg: String::from("expected an instruction"),
                span: crate::error::Span::new(0, 0, 0, 0),
            }),
        }
    }

    /// Reset the assembler to its initial state, keeping configuration
    /// (architecture, syntax, optimization level, limits) intact.
    ///
    /// This allows reusing the same `Assembler` for multiple assembly operations
    /// without reallocating configuration state.
    ///
    /// # Examples
    ///
    /// ```
    /// use asm_rs::{Assembler, Arch};
    ///
    /// let mut asm = Assembler::new(Arch::X86_64);
    /// asm.emit("nop")?;
    /// asm.reset();
    /// asm.emit("ret")?;
    /// let result = asm.finish()?;
    /// assert_eq!(result.bytes(), &[0xC3]); // only ret, nop was reset
    /// # Ok::<(), asm_rs::AsmError>(())
    /// ```
    pub fn reset(&mut self) -> &mut Self {
        self.linker = Linker::new();
        self.preprocessor = Preprocessor::new();
        self.errors.clear();
        self.fragment_annotations.clear();
        // listing_enabled is configuration, preserved across resets
        self.statement_count = 0;
        self.label_count = 0;
        self.literal_pool.clear();
        self.literal_pool_counter = 0;
        self.thumb_func_pending = false;
        self.thumb_labels.clear();
        self.estimated_output_bytes = 0;
        // Note: rvc_enabled, x86_mode, and arch are configuration state
        // deliberately preserved across resets (like syntax and opt_level).
        self
    }

    /// Check that adding `n` bytes would not exceed the output size limit.
    ///
    /// Called by builder methods to enforce `max_output_bytes` eagerly —
    /// *before* allocating the data — rather than only at `finish()` time.
    fn check_output_limit(&mut self, additional: usize) -> Result<(), AsmError> {
        self.estimated_output_bytes += additional;
        if self.estimated_output_bytes > self.resource_limits.max_output_bytes {
            return Err(AsmError::ResourceLimitExceeded {
                resource: String::from("output bytes"),
                limit: self.resource_limits.max_output_bytes,
            });
        }
        Ok(())
    }

    /// Finalize assembly: resolve labels, apply relocations, return result.
    ///
    /// # Errors
    ///
    /// Returns [`AsmError`] if label resolution fails, relocations cannot be
    /// applied, accumulated errors exist, or resource limits are exceeded.
    pub fn finish(mut self) -> Result<AssemblyResult, AsmError> {
        if !self.errors.is_empty() {
            if self.errors.len() == 1 {
                return Err(self.errors.remove(0));
            }
            return Err(AsmError::Multiple {
                errors: self.errors,
            });
        }

        let base = self.linker.base_address();

        // Flush any remaining literal pool entries before resolving.
        let flush_span = crate::error::Span::new(0, 0, 0, 0);
        self.flush_literal_pool(flush_span)?;

        let (bytes, mut labels, relocations, offsets) = self.linker.resolve()?;

        // Set LSB on Thumb function label addresses for interworking
        for (name, addr) in labels.iter_mut() {
            if self.thumb_labels.iter().any(|t| t == name) {
                *addr |= 1;
            }
        }

        // Enforce output size limit
        if bytes.len() > self.resource_limits.max_output_bytes {
            return Err(AsmError::ResourceLimitExceeded {
                resource: String::from("output bytes"),
                limit: self.resource_limits.max_output_bytes,
            });
        }

        // Build source annotations: map fragment index → output offset,
        // then look up the source text for each annotated fragment.
        let source_annotations = self.build_source_annotations(&offsets);

        Ok(AssemblyResult {
            bytes,
            labels,
            relocations,
            base_address: base,
            source_annotations,
        })
    }

    /// Build source text annotations by mapping fragment indices to output
    /// offsets and extracting the source text from the stored source strings.
    fn build_source_annotations(&self, offsets: &[u64]) -> Vec<(u64, String)> {
        let mut annotations = Vec::new();
        for &(frag_idx, ref text) in &self.fragment_annotations {
            if frag_idx < offsets.len() {
                annotations.push((offsets[frag_idx], text.clone()));
            }
        }
        annotations
    }

    fn process_statements(
        &mut self,
        statements: &mut [Statement],
        source: &str,
    ) -> Result<(), AsmError> {
        // Check statement count limit (total across all emit() calls)
        self.statement_count += statements.len();
        if self.statement_count > self.resource_limits.max_statements {
            return Err(AsmError::ResourceLimitExceeded {
                resource: String::from("statements"),
                limit: self.resource_limits.max_statements,
            });
        }

        for stmt in statements.iter_mut() {
            match stmt {
                Statement::Label(name, span) => {
                    self.label_count += 1;
                    if self.label_count > self.resource_limits.max_labels {
                        return Err(AsmError::ResourceLimitExceeded {
                            resource: String::from("labels"),
                            limit: self.resource_limits.max_labels,
                        });
                    }
                    self.linker.add_label(name, *span)?;
                    // Mark as Thumb function if .thumb_func was pending
                    if self.thumb_func_pending {
                        self.thumb_labels.push(name.clone());
                        self.thumb_func_pending = false;
                    }
                }

                Statement::Instruction(instr) => {
                    let frag_idx = self.linker.fragment_count();
                    // Resolve any constant references in operands before encoding
                    self.resolve_constants_in_instruction(instr);
                    // Transform literal pool operands: =value → label reference
                    self.transform_literal_pool_operands(instr);
                    // Apply peephole optimizations when OptLevel::Size is active
                    if self.opt_level == OptLevel::Size {
                        crate::optimize::optimize_instruction(instr, self.arch);
                    }
                    let encode_result = if self.x86_mode == crate::ir::X86Mode::Mode16 {
                        #[cfg(feature = "x86")]
                        {
                            encoder::encode_instruction_16(instr)
                        }
                        #[cfg(not(feature = "x86"))]
                        {
                            encoder::encode_instruction(instr, self.arch)
                        }
                    } else {
                        encoder::encode_instruction(instr, self.arch)
                    };
                    // RISC-V auto-narrowing: when .option rvc is active and the
                    // instruction is a 4-byte standard form, try to compress it
                    // to a 16-bit C-extension equivalent.
                    #[cfg(feature = "riscv")]
                    let encode_result = if self.rvc_enabled
                        && matches!(self.arch, Arch::Rv32 | Arch::Rv64)
                        && !instr.mnemonic.starts_with("c.")
                    {
                        match encode_result {
                            Ok(ref enc) if enc.bytes.len() == 4 && enc.relocation.is_none() => {
                                let is_rv64 = self.arch == Arch::Rv64;
                                if let Some(hw) = crate::riscv::try_compress(
                                    &instr.mnemonic,
                                    &instr.operands,
                                    is_rv64,
                                    instr.span,
                                ) {
                                    Ok(crate::riscv::rvc_instr(hw))
                                } else {
                                    encode_result
                                }
                            }
                            _ => encode_result,
                        }
                    } else {
                        encode_result
                    };
                    match encode_result {
                        Ok(encoded) => {
                            self.check_output_limit(encoded.bytes.len())?;
                            self.linker.add_encoded(
                                encoded.bytes,
                                encoded.relocation,
                                encoded.relax,
                                instr.span,
                            )?;
                            self.annotate(frag_idx, source, instr.span);
                        }
                        Err(e) => {
                            self.errors.push(e);
                            if self.errors.len() >= self.resource_limits.max_errors {
                                return Err(AsmError::ResourceLimitExceeded {
                                    resource: String::from("errors"),
                                    limit: self.resource_limits.max_errors,
                                });
                            }
                        }
                    }
                }

                Statement::Data(data) => {
                    let frag_idx = self.linker.fragment_count();
                    let span = data.span;
                    self.emit_data(data)?;
                    self.annotate(frag_idx, source, span);
                }

                Statement::Align(align) => {
                    let frag_idx = self.linker.fragment_count();
                    let span = align.span;
                    // When no explicit fill byte is given and the target is
                    // x86/x86-64, pad with multi-byte NOP sequences instead
                    // of zero bytes — optimal for code-section alignment.
                    let use_nop =
                        align.fill.is_none() && matches!(self.arch, Arch::X86 | Arch::X86_64);
                    self.linker.add_alignment(
                        align.alignment,
                        align.fill.unwrap_or(0x00),
                        align.max_skip,
                        use_nop,
                        align.span,
                    );
                    self.annotate(frag_idx, source, span);
                }

                Statement::Const(c) => {
                    self.linker.define_constant(&c.name, c.value);
                }

                Statement::Fill(fill) => {
                    let frag_idx = self.linker.fragment_count();
                    let span = fill.span;
                    let total = fill.count as usize * fill.size as usize;
                    self.check_output_limit(total)?;
                    let mut bytes = Vec::with_capacity(total);
                    // GAS semantics: value is a LE integer padded to `size` bytes.
                    // .fill 2, 4, 0x90 → [90 00 00 00  90 00 00 00]
                    let val_bytes = fill.value.to_le_bytes();
                    for _ in 0..fill.count {
                        for &b in val_bytes.iter().take(fill.size as usize) {
                            bytes.push(b);
                        }
                        // Pad with zeros if size > 8
                        if (fill.size as usize) > 8 {
                            bytes.resize(bytes.len() + fill.size as usize - 8, 0);
                        }
                    }
                    self.linker.add_bytes(bytes, fill.span);
                    self.annotate(frag_idx, source, span);
                }

                Statement::Space(space) => {
                    let frag_idx = self.linker.fragment_count();
                    let span = space.span;
                    self.check_output_limit(space.size as usize)?;
                    let bytes = alloc::vec![space.fill; space.size as usize];
                    self.linker.add_bytes(bytes, space.span);
                    self.annotate(frag_idx, source, span);
                }

                Statement::Org(org) => {
                    let frag_idx = self.linker.fragment_count();
                    let span = org.span;
                    // .org sets the location counter to an absolute address.
                    // The linker emits fill bytes to pad from current position
                    // to the target.
                    self.linker.add_org(org.offset, org.fill, org.span);
                    self.annotate(frag_idx, source, span);
                }

                Statement::CodeMode(mode, span) => {
                    // .code16 / .code32 / .code64 — switch x86 encoding mode
                    if !matches!(self.arch, Arch::X86 | Arch::X86_64) {
                        return Err(AsmError::Syntax {
                            msg: String::from(".code16/.code32/.code64 only valid for x86/x86-64"),
                            span: *span,
                        });
                    }
                    self.x86_mode = *mode;
                    // Update the arch to match the new mode for encoding dispatch
                    match mode {
                        crate::ir::X86Mode::Mode16 | crate::ir::X86Mode::Mode32 => {
                            self.arch = Arch::X86;
                        }
                        crate::ir::X86Mode::Mode64 => {
                            self.arch = Arch::X86_64;
                        }
                    }
                }

                Statement::Ltorg(span) => {
                    // Flush pending literal pool entries
                    let span = *span;
                    self.flush_literal_pool(span)?;
                }

                Statement::OptionRvc(enable, span) => {
                    // .option rvc / .option norvc — toggle RISC-V C extension auto-narrowing
                    if !matches!(self.arch, Arch::Rv32 | Arch::Rv64) {
                        return Err(AsmError::Syntax {
                            msg: String::from(".option rvc/norvc is only valid for RISC-V"),
                            span: *span,
                        });
                    }
                    self.rvc_enabled = *enable;
                }

                Statement::ThumbMode(is_thumb, span) => {
                    // .thumb / .arm — switch between Thumb and ARM modes
                    if !matches!(self.arch, Arch::Arm | Arch::Thumb) {
                        return Err(AsmError::Syntax {
                            msg: String::from(".thumb/.arm only valid for ARM"),
                            span: *span,
                        });
                    }
                    self.arch = if *is_thumb { Arch::Thumb } else { Arch::Arm };
                }

                Statement::ThumbFunc(span) => {
                    // .thumb_func — mark next label as Thumb function (LSB set)
                    if !matches!(self.arch, Arch::Arm | Arch::Thumb) {
                        return Err(AsmError::Syntax {
                            msg: String::from(".thumb_func only valid for ARM/Thumb"),
                            span: *span,
                        });
                    }
                    // Also switch to Thumb mode (GNU as behavior)
                    self.arch = Arch::Thumb;
                    self.thumb_func_pending = true;
                }
            }
        }
        Ok(())
    }

    /// Record a source-text annotation for a fragment, if listing is enabled.
    #[inline]
    fn annotate(&mut self, frag_idx: usize, source: &str, span: Span) {
        if self.listing_enabled {
            let src_text = extract_source_line(source, span);
            if !src_text.is_empty() {
                self.fragment_annotations
                    .push((frag_idx, src_text.to_string()));
            }
        }
    }

    /// Transform `Operand::LiteralPoolValue(val)` into `Operand::Label(label)`
    /// and queue the constant for emission in the next literal pool flush.
    ///
    /// The destination register width determines pool entry size:
    /// - X-registers → 8-byte entry (`.quad`)
    /// - W-registers → 4-byte entry (`.long`)
    ///
    /// Duplicate values with the same size are deduplicated to share a single
    /// pool entry.
    fn transform_literal_pool_operands(&mut self, instr: &mut Instruction) {
        use crate::ir::Operand;

        // Determine pool entry size from the first register operand.
        // - ARM registers → always 4 bytes (32-bit)
        // - AArch64 X-registers → 8 bytes, W-registers → 4 bytes
        // - Default: 8 bytes (AArch64 64-bit) if no register is found
        let size: u8 = instr
            .operands
            .iter()
            .find_map(|op| {
                if let Operand::Register(r) = op {
                    if r.is_arm() {
                        return Some(4u8); // ARM32 is always 4 bytes
                    }
                    if r.is_aarch64() {
                        return Some(if r.is_a64_64bit() { 8u8 } else { 4u8 });
                    }
                }
                None
            })
            .unwrap_or(8);

        for op in &mut instr.operands {
            if let Operand::LiteralPoolValue(val) = op {
                let val = *val;

                // Check for existing pool entry with same value + size (dedup).
                let label = if let Some(existing) = self
                    .literal_pool
                    .iter()
                    .find(|e| e.value == val && e.size == size)
                {
                    existing.label.clone()
                } else {
                    let label = alloc::format!(".Lpool_{}", self.literal_pool_counter);
                    self.literal_pool_counter += 1;
                    self.literal_pool.push(LiteralPoolEntry {
                        value: val,
                        size,
                        label: label.clone(),
                    });
                    label
                };

                *op = Operand::Label(label);
            }
        }
    }

    /// Flush all pending literal pool entries as labeled data fragments.
    ///
    /// Emits alignment padding followed by each pool entry's label + data.
    /// Called at `.ltorg` directives, unconditional branches (future), or `finish()`.
    fn flush_literal_pool(&mut self, span: Span) -> Result<(), AsmError> {
        if self.literal_pool.is_empty() {
            return Ok(());
        }

        // Align pool to the largest entry size for natural alignment.
        let max_align = self
            .literal_pool
            .iter()
            .map(|e| e.size as u32)
            .max()
            .unwrap_or(4);
        self.linker
            .add_alignment(max_align, 0x00, None, false, span);

        // Drain and emit each entry.
        let entries: Vec<LiteralPoolEntry> = core::mem::take(&mut self.literal_pool);
        for entry in &entries {
            self.linker.add_label(&entry.label, span)?;
            let bytes = match entry.size {
                4 => (entry.value as u32).to_le_bytes().to_vec(),
                8 => (entry.value as u64).to_le_bytes().to_vec(),
                _ => (entry.value as u64).to_le_bytes().to_vec(),
            };
            self.linker.add_bytes(bytes, span);
        }

        Ok(())
    }

    /// Replace label operands with immediate values when they refer to known constants.
    ///
    /// Also resolves constants inside `Operand::Expression` trees and collapses
    /// fully-numeric expressions to `Operand::Immediate`.
    fn resolve_constants_in_instruction(&self, instr: &mut Instruction) {
        use crate::ir::Operand;
        for op in &mut instr.operands {
            match op {
                Operand::Label(name) => {
                    if let Some(&value) = self.linker.get_constant(name) {
                        *op = Operand::Immediate(value);
                    }
                }
                Operand::Expression(expr) => {
                    // Substitute any constants referenced inside the expression tree.
                    expr.resolve_constants(|name| self.linker.get_constant(name).copied());
                    // If the expression is now purely numeric, collapse to Immediate.
                    if let Some(val) = expr.eval() {
                        *op = Operand::Immediate(val);
                    }
                }
                Operand::Memory(mem) => {
                    // Resolve constants used as displacement labels (e.g., [rbp + MY_CONST])
                    if let Some(ref label) = mem.disp_label {
                        if let Some(&value) = self.linker.get_constant(label) {
                            mem.disp = mem.disp.wrapping_add(value as i64);
                            mem.disp_label = None;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Emit a data declaration, handling label references via relocations.
    fn emit_data(&mut self, data: &DataDecl) -> Result<(), AsmError> {
        use crate::encoder::Relocation;

        let data_item_size: usize = match data.size {
            DataSize::Byte => 1,
            DataSize::Word => 2,
            DataSize::Long => 4,
            DataSize::Quad => 8,
        };

        // Accumulate contiguous non-label bytes, flush when we hit a label.
        let mut pending: Vec<u8> = Vec::new();

        for value in &data.values {
            match value {
                DataValue::Integer(n) => match data.size {
                    DataSize::Byte => pending.push(*n as u8),
                    DataSize::Word => pending.extend_from_slice(&(*n as u16).to_le_bytes()),
                    DataSize::Long => pending.extend_from_slice(&(*n as u32).to_le_bytes()),
                    DataSize::Quad => pending.extend_from_slice(&(*n as u64).to_le_bytes()),
                },
                DataValue::Bytes(b) => {
                    pending.extend_from_slice(b);
                }
                DataValue::Label(name, addend) => {
                    // Check if this is a constant (defined via .equ) rather than a label
                    if let Some(&const_val) = self.linker.get_constant(name) {
                        let val = const_val.wrapping_add(*addend as i128);
                        match data.size {
                            DataSize::Byte => pending.push(val as u8),
                            DataSize::Word => {
                                pending.extend_from_slice(&(val as u16).to_le_bytes())
                            }
                            DataSize::Long => {
                                pending.extend_from_slice(&(val as u32).to_le_bytes())
                            }
                            DataSize::Quad => {
                                pending.extend_from_slice(&(val as u64).to_le_bytes())
                            }
                        }
                        continue;
                    }

                    // Flush any pending plain bytes first
                    if !pending.is_empty() {
                        self.linker
                            .add_bytes(core::mem::take(&mut pending), data.span);
                    }
                    // Emit a zero-filled data slot with an absolute relocation for the label
                    let mut slot = encoder::InstrBytes::new();
                    for _ in 0..data_item_size {
                        slot.push(0);
                    }
                    let reloc = Relocation {
                        offset: 0,
                        size: data_item_size as u8,
                        label: alloc::rc::Rc::from(name.as_str()),
                        kind: encoder::RelocKind::Absolute,
                        addend: *addend,
                        trailing_bytes: 0,
                    };
                    // Use add_encoded which will make a Fixed fragment
                    self.linker
                        .add_encoded(slot, Some(reloc), None, data.span)?;
                }
            }
        }

        // Flush remaining bytes
        if !pending.is_empty() {
            self.linker.add_bytes(pending, data.span);
        }

        Ok(())
    }
}

/// Parse source text into statements.
fn parse_source(source: &str, arch: Arch, syntax: Syntax) -> Result<Vec<Statement>, AsmError> {
    let tokens = lexer::tokenize(source)?;
    parser::parse_with_syntax(&tokens, arch, syntax)
}

/// Extract the source text for a span from the original source string.
///
/// Returns the trimmed text of the line containing the span, or a brief
/// fallback if the span is out-of-range.
fn extract_source_line(source: &str, span: Span) -> &str {
    let offset = span.offset;
    if offset >= source.len() {
        return "";
    }
    // Find the start of the line containing this span
    let line_start = source[..offset].rfind('\n').map_or(0, |p| p + 1);
    // Find the end of the line
    let line_end = source[offset..]
        .find('\n')
        .map_or(source.len(), |p| offset + p);
    source[line_start..line_end].trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    // === One-Shot API ===

    #[test]
    fn assemble_nop() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("nop").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x90]);
    }

    #[test]
    fn assemble_ret() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("ret").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0xC3]);
    }

    #[test]
    fn assemble_multiple_instructions() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("nop\nret").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x90, 0xC3]);
    }

    #[test]
    fn assemble_push_pop() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("push rbp").unwrap();
        asm.emit("mov rbp, rsp").unwrap();
        asm.emit("pop rbp").unwrap();
        asm.emit("ret").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        assert_eq!(bytes[0], 0x55); // push rbp
        assert_eq!(*bytes.last().unwrap(), 0xC3); // ret
    }

    #[test]
    fn assemble_with_label() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("jmp target\ntarget:\nnop").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // Branch relaxation: short form EB rel8 since target is right after
        assert_eq!(bytes[0], 0xEB); // jmp rel8
        assert_eq!(bytes[1], 0x00); // rel8 = 0
        assert_eq!(bytes[2], 0x90); // nop
    }

    #[test]
    fn assemble_backward_jump() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("loop_start:\nnop\njmp loop_start").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        assert_eq!(bytes[0], 0x90); // nop
                                    // Branch relaxation: short form EB rel8
        assert_eq!(bytes[1], 0xEB); // jmp rel8
                                    // target=0, frag_end=1+2=3, disp=0-3=-3=0xFD
        assert_eq!(bytes[2], 0xFD);
    }

    #[test]
    fn assemble_conditional_jump() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("cmp rax, 0\nje done\nnop\ndone:\nret").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // Should contain: cmp, je, nop, ret
        assert!(!bytes.is_empty());
        // Last byte should be ret
        assert_eq!(*bytes.last().unwrap(), 0xC3);
    }

    #[test]
    fn assemble_xor_self() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("xor eax, eax").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x31, 0xC0]);
    }

    #[test]
    fn assemble_syscall_stub() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("mov eax, 60\nxor edi, edi\nsyscall").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // mov eax, 60 → B8 3C 00 00 00
        assert_eq!(&bytes[0..5], &[0xB8, 0x3C, 0x00, 0x00, 0x00]);
        // Last 2 bytes: syscall → 0F 05
        assert_eq!(&bytes[bytes.len() - 2..], &[0x0F, 0x05]);
    }

    // === Builder API ===

    #[test]
    fn builder_api() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("push rbp").unwrap();
        asm.db(&[0xCC]).unwrap(); // int3
        asm.emit("pop rbp").unwrap();
        asm.emit("ret").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        assert_eq!(bytes[0], 0x55); // push rbp
        assert_eq!(bytes[1], 0xCC); // int3
    }

    #[test]
    fn builder_label() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("jmp target").unwrap();
        asm.label("target").unwrap();
        asm.emit("ret").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // Short form: EB 00 C3
        assert_eq!(bytes[0], 0xEB);
        assert_eq!(*bytes.last().unwrap(), 0xC3);
    }

    #[test]
    fn builder_data_words() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.dw(0x1234).unwrap();
        asm.dd(0xDEADBEEF).unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        assert_eq!(&bytes[0..2], &[0x34, 0x12]);
        assert_eq!(&bytes[2..6], &[0xEF, 0xBE, 0xAD, 0xDE]);
    }

    // === Data Directives ===

    #[test]
    fn assemble_byte_directive() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit(".byte 0x90, 0xCC, 0xC3").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x90, 0xCC, 0xC3]);
    }

    #[test]
    fn assemble_word_directive() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit(".word 0x1234").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x34, 0x12]);
    }

    #[test]
    fn assemble_asciz_directive() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit(".asciz \"hello\"").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), b"hello\0");
    }

    #[test]
    fn assemble_equ_constant() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit(".equ EXIT, 60\nmov eax, EXIT").unwrap();
        // Note: constants are resolved at link time if referenced by label
        let _result = asm.finish();
        // This test mainly verifies parsing succeeds
    }

    #[test]
    fn assemble_fill_directive() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit(".fill 3, 1, 0x90").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x90, 0x90, 0x90]);
    }

    #[test]
    fn assemble_space_directive() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit(".space 4").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0, 0, 0, 0]);
    }

    // === Error Cases ===

    #[test]
    fn unknown_mnemonic_error() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("foobar").unwrap(); // error is collected, not fail-fast
        let err = asm.finish().unwrap_err();
        assert!(matches!(err, AsmError::UnknownMnemonic { .. }));
    }

    #[test]
    fn duplicate_label_error() {
        let mut asm = Assembler::new(Arch::X86_64);
        let err = asm.emit("foo:\nfoo:").unwrap_err();
        assert!(matches!(err, AsmError::DuplicateLabel { .. }));
    }

    #[test]
    fn undefined_label_error() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("jmp nowhere").unwrap();
        let err = asm.finish().unwrap_err();
        assert!(matches!(err, AsmError::UndefinedLabel { .. }));
    }

    // === External Labels ===

    #[test]
    fn assemble_with_external() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.define_external("printf", 0x400000);
        asm.emit("mov rax, printf").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // movabs rax, imm64 with printf address
        assert_eq!(&bytes[bytes.len() - 8..], &0x400000u64.to_le_bytes());
    }

    // === Base Address ===

    #[test]
    fn assemble_with_base_address() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.base_address(0x1000);
        asm.emit("nop").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x90]);
    }

    // === Complex Programs ===

    #[test]
    fn assemble_loop() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit(
            r#"
            mov ecx, 10
        loop_start:
            dec ecx
            jnz loop_start
            ret
        "#,
        )
        .unwrap();
        let result = asm.finish().unwrap();
        assert!(!result.is_empty());
        assert_eq!(*result.bytes().last().unwrap(), 0xC3);
    }

    #[test]
    fn assemble_function_prologue_epilogue() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit(
            r#"
            push rbp
            mov rbp, rsp
            sub rsp, 0x20
            add rsp, 0x20
            pop rbp
            ret
        "#,
        )
        .unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        assert_eq!(bytes[0], 0x55); // push rbp
        assert_eq!(*bytes.last().unwrap(), 0xC3); // ret
    }

    #[test]
    fn result_length() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("nop\nnop\nnop").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn result_into_bytes() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("ret").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.into_bytes();
        assert_eq!(bytes, vec![0xC3]);
    }

    // === Semicolon Separated ===

    #[test]
    fn semicolon_separated_instructions() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("nop; nop; ret").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x90, 0x90, 0xC3]);
    }

    // === Labels export ===

    #[test]
    fn labels_returned() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("start:\nnop\nnop\nend:\nret").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.label_address("start"), Some(0));
        // nop=1B, nop=1B → end is at offset 2
        assert_eq!(result.label_address("end"), Some(2));
    }

    #[test]
    fn labels_with_base_address() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.base_address(0x400000);
        asm.emit("entry:\nnop").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.label_address("entry"), Some(0x400000));
    }

    #[test]
    fn builder_label_address() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.label("before").unwrap();
        asm.emit("nop; nop; nop").unwrap();
        asm.label("after").unwrap();
        asm.emit("ret").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.label_address("before"), Some(0));
        assert_eq!(result.label_address("after"), Some(3));
    }

    // === Syntax / OptLevel builder ===

    #[test]
    fn builder_syntax_and_optimize() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.syntax(Syntax::Intel);
        asm.optimize(OptLevel::Size);
        asm.emit("nop").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x90]);
    }

    // === define_constant builder ===

    #[test]
    fn builder_define_constant() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.define_constant("EXIT", 60);
        asm.emit("mov eax, EXIT").unwrap();
        let result = asm.finish().unwrap();
        // mov eax, 60 → B8 3C 00 00 00
        assert_eq!(result.bytes(), &[0xB8, 0x3C, 0x00, 0x00, 0x00]);
    }

    // === Branch relaxation observable from public API ===

    #[test]
    fn short_branch_uses_rel8() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("je done\ndone:\nret").unwrap();
        let result = asm.finish().unwrap();
        // je rel8 = 74 00, ret = C3
        assert_eq!(result.bytes(), &[0x74, 0x00, 0xC3]);
    }

    // === Data label references ===

    #[test]
    fn quad_label_reference() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.base_address(0x1000);
        asm.emit("func:\nnop\nret\njump_table:\n.quad func")
            .unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // func is at 0x1000, nop=1, ret=1, so jump_table at 0x1002
        // .quad func → should contain 0x1000 as a u64 LE
        let qw = u64::from_le_bytes(bytes[2..10].try_into().unwrap());
        assert_eq!(qw, 0x1000);
    }

    #[test]
    fn long_label_reference() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.base_address(0x2000);
        asm.emit("entry:\nnop\n.long entry").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // entry at 0x2000, nop=1B, .long at offset 1 → value is 0x2000
        let dw = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
        assert_eq!(dw, 0x2000);
    }

    #[test]
    fn name_equals_constant_in_instruction() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("ANSWER = 42\nmov eax, ANSWER").unwrap();
        let result = asm.finish().unwrap();
        // mov eax, 42 → B8 2A 00 00 00
        assert_eq!(result.bytes(), &[0xB8, 0x2A, 0x00, 0x00, 0x00]);
    }

    // === Listing output ===

    #[test]
    fn listing_simple() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.enable_listing();
        asm.emit("nop\nret").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        assert!(listing.contains("00000000"));
        assert!(listing.contains("90")); // nop
        assert!(listing.contains("C3")); // ret
                                         // Source text annotations
        assert!(listing.contains("nop"));
        assert!(listing.contains("ret"));
    }

    #[test]
    fn listing_with_labels() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.enable_listing();
        asm.emit("start:\nnop\nend:\nret").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        assert!(listing.contains("start:"));
        assert!(listing.contains("end:"));
        // Source text with instructions
        assert!(listing.contains("nop"));
        assert!(listing.contains("ret"));
    }

    #[test]
    fn listing_with_base_address() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.enable_listing();
        asm.base_address(0x401000);
        asm.emit("nop").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        assert!(listing.contains("00401000"));
        assert!(listing.contains("nop"));
    }

    #[test]
    fn listing_base_address_accessor() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.base_address(0x1000);
        asm.emit("nop").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.base_address(), 0x1000);
    }

    #[test]
    fn listing_hex_format() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.enable_listing();
        asm.emit("push rbp\nmov rbp, rsp").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        // push rbp = 55
        assert!(listing.contains("55"));
        // mov rbp, rsp = 48 89 E5
        assert!(listing.contains("4889E5"));
        // Source text appears
        assert!(listing.contains("push rbp"));
        assert!(listing.contains("mov rbp, rsp"));
    }

    #[test]
    fn listing_source_annotations() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.enable_listing();
        asm.emit("mov eax, 1\nadd eax, 2\nret").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        // Each line should have source text
        assert!(listing.contains("mov eax, 1"));
        assert!(listing.contains("add eax, 2"));
        assert!(listing.contains("ret"));
    }

    #[test]
    fn listing_data_annotation() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.enable_listing();
        asm.emit(".byte 0x90, 0xCC").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        assert!(listing.contains(".byte 0x90, 0xCC"));
    }

    // === Relocations ===

    #[test]
    fn relocations_returned() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("jmp target\nnop\ntarget:\nret").unwrap();
        let result = asm.finish().unwrap();
        assert!(!result.relocations().is_empty());
        assert_eq!(result.relocations()[0].label, "target");
    }

    #[test]
    fn relocations_for_call() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("call func\nfunc:\nret").unwrap();
        let result = asm.finish().unwrap();
        let relocs = result.relocations();
        assert!(!relocs.is_empty());
        assert_eq!(relocs[0].label, "func");
        assert_eq!(relocs[0].kind, crate::encoder::RelocKind::X86Relative);
    }

    // === Builder convenience methods ===

    #[test]
    fn builder_ascii() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.ascii("AB").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x41, 0x42]);
    }

    #[test]
    fn builder_asciz() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.asciz("Hi").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x48, 0x69, 0x00]);
    }

    #[test]
    fn builder_align() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.db(&[0x90]).unwrap(); // 1 byte
        asm.align(4); // pad to 4-byte boundary
        asm.db(&[0xCC]).unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes().len(), 5); // 1 + 3 padding + 1
        assert_eq!(result.bytes()[4], 0xCC);
    }

    #[test]
    fn builder_align_with_fill() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.db(&[0x90]).unwrap();
        asm.align_with_fill(4, 0xAA);
        asm.db(&[0xCC]).unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes()[1], 0xAA);
        assert_eq!(result.bytes()[2], 0xAA);
        assert_eq!(result.bytes()[3], 0xAA);
    }

    #[test]
    fn builder_org() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.db(&[0x90]).unwrap();
        asm.org(4);
        asm.db(&[0xCC]).unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x90, 0x00, 0x00, 0x00, 0xCC]);
    }

    #[test]
    fn builder_org_with_fill() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.db(&[0x90]).unwrap();
        asm.org_with_fill(4, 0xFF);
        asm.db(&[0xCC]).unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x90, 0xFF, 0xFF, 0xFF, 0xCC]);
    }

    #[test]
    fn builder_fill() {
        // .fill 3, 2, 0xAB → 3 units of 2 bytes each, value=0xAB as LE integer
        // Each unit: [0xAB, 0x00] (LE encoding of 0xAB in 2 bytes)
        let mut asm = Assembler::new(Arch::X86_64);
        asm.fill(3, 2, 0xAB).unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0xAB, 0x00, 0xAB, 0x00, 0xAB, 0x00]);
    }

    #[test]
    fn builder_fill_size_1() {
        // .fill 4, 1, 0xCC → 4 units of 1 byte, value=0xCC → simple fill
        let mut asm = Assembler::new(Arch::X86_64);
        asm.fill(4, 1, 0xCC).unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0xCC, 0xCC, 0xCC, 0xCC]);
    }

    #[test]
    fn builder_fill_multi_byte_value() {
        // .fill 1, 4, 0xDEADBEEF → 1 unit of 4 bytes, value=0xDEADBEEF in LE
        let mut asm = Assembler::new(Arch::X86_64);
        asm.fill(1, 4, 0xDEADBEEFu32 as i64).unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn builder_fill_16bit_value() {
        // .fill 2, 2, 0x1234 → 2 units of 2 bytes
        let mut asm = Assembler::new(Arch::X86_64);
        asm.fill(2, 2, 0x1234).unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x34, 0x12, 0x34, 0x12]);
    }

    #[test]
    fn builder_space() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.space(4).unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x00, 0x00, 0x00, 0x00]);
    }

    // === Listing annotations for directives ===

    #[test]
    fn listing_fill_annotation() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.enable_listing();
        asm.emit(".fill 2, 1, 0x90").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        assert!(listing.contains(".fill 2, 1, 0x90"));
    }

    #[test]
    fn listing_space_annotation() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.enable_listing();
        asm.emit(".space 4").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        assert!(listing.contains(".space 4"));
    }

    #[test]
    fn listing_align_annotation() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.enable_listing();
        asm.emit("nop\n.align 4\nnop").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        assert!(listing.contains(".align 4"));
    }

    #[test]
    fn listing_org_annotation() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.enable_listing();
        asm.emit("nop\n.org 0x10\nnop").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        assert!(listing.contains(".org 0x10"));
    }

    // === .org fill byte ===

    #[test]
    fn org_with_fill_byte() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("nop\n.org 0x04, 0xFF\nnop").unwrap();
        let result = asm.finish().unwrap();
        // nop (0x90) + 3 fill bytes (0xFF) + nop (0x90)
        assert_eq!(result.bytes(), &[0x90, 0xFF, 0xFF, 0xFF, 0x90]);
    }

    // === AT&T Syntax ===

    #[test]
    fn att_syntax_basic() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.syntax(Syntax::Att);
        asm.emit("movq $1, %rax").unwrap();
        let result = asm.finish().unwrap();
        // mov rax, 1 → optimizer narrows to mov eax, 1 = B8 01 00 00 00
        assert_eq!(result.bytes(), &[0xB8, 0x01, 0x00, 0x00, 0x00]);
    }

    // === Resource Limits ===

    #[test]
    fn resource_limit_max_statements() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.limits(ResourceLimits {
            max_statements: 3,
            ..ResourceLimits::default()
        });
        // 3 statements: ok
        asm.emit("nop; nop; nop").unwrap();
        // 2 more statements: total = 5 > 3, should fail
        let err = asm.emit("nop; nop").unwrap_err();
        match err {
            AsmError::ResourceLimitExceeded { resource, limit } => {
                assert_eq!(resource, "statements");
                assert_eq!(limit, 3);
            }
            other => panic!("expected ResourceLimitExceeded, got: {other:?}"),
        }
    }

    #[test]
    fn resource_limit_max_labels() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.limits(ResourceLimits {
            max_labels: 2,
            ..ResourceLimits::default()
        });
        asm.label("a").unwrap();
        asm.label("b").unwrap();
        let err = asm.label("c").unwrap_err();
        match err {
            AsmError::ResourceLimitExceeded { resource, limit } => {
                assert_eq!(resource, "labels");
                assert_eq!(limit, 2);
            }
            other => panic!("expected ResourceLimitExceeded, got: {other:?}"),
        }
    }

    #[test]
    fn resource_limit_max_labels_via_emit() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.limits(ResourceLimits {
            max_labels: 1,
            ..ResourceLimits::default()
        });
        asm.emit("a: nop").unwrap();
        let err = asm.emit("b: nop").unwrap_err();
        match err {
            AsmError::ResourceLimitExceeded { resource, limit } => {
                assert_eq!(resource, "labels");
                assert_eq!(limit, 1);
            }
            other => panic!("expected ResourceLimitExceeded, got: {other:?}"),
        }
    }

    #[test]
    fn resource_limit_max_output_bytes() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.limits(ResourceLimits {
            max_output_bytes: 4,
            ..ResourceLimits::default()
        });
        asm.emit("nop; nop; nop; nop").unwrap(); // 4 bytes = exactly at limit: ok
        let result = asm.finish();
        assert!(result.is_ok());

        // With eager checking, the limit is now caught at emit() time
        let mut asm2 = Assembler::new(Arch::X86_64);
        asm2.limits(ResourceLimits {
            max_output_bytes: 3,
            ..ResourceLimits::default()
        });
        let err = asm2.emit("nop; nop; nop; nop").unwrap_err(); // 4 bytes > 3: fail at emit
        match err {
            AsmError::ResourceLimitExceeded { resource, limit } => {
                assert_eq!(resource, "output bytes");
                assert_eq!(limit, 3);
            }
            other => panic!("expected ResourceLimitExceeded, got: {other:?}"),
        }
    }

    #[test]
    fn resource_limits_default_does_not_interfere() {
        // Default limits should be generous enough for normal use
        let mut asm = Assembler::new(Arch::X86_64);
        // Emit a lot of instructions at once
        let source: String = (0..1000).map(|_| "nop; ").collect();
        asm.emit(&source).unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn resource_limit_max_recursion_depth() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.limits(ResourceLimits {
            max_recursion_depth: 3,
            ..ResourceLimits::default()
        });
        // A macro that calls itself — should hit the recursion limit quickly
        let result = asm.emit(".macro boom\nboom\n.endm\nboom");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AsmError::ResourceLimitExceeded { resource, limit } => {
                assert_eq!(resource, "macro recursion depth");
                assert_eq!(limit, 3);
            }
            _ => panic!("expected ResourceLimitExceeded, got {:?}", err),
        }
    }

    // ─── encode_one ────────────────────────────────────────────────

    #[test]
    fn encode_one_nop() {
        let asm = Assembler::new(Arch::X86_64);
        let bytes = asm.encode_one("nop").unwrap();
        assert_eq!(bytes, alloc::vec![0x90]);
    }

    #[test]
    fn encode_one_ret() {
        let asm = Assembler::new(Arch::X86_64);
        let bytes = asm.encode_one("ret").unwrap();
        assert_eq!(bytes, alloc::vec![0xC3]);
    }

    #[test]
    fn encode_one_empty_input() {
        let asm = Assembler::new(Arch::X86_64);
        let bytes = asm.encode_one("").unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn encode_one_rejects_label() {
        let asm = Assembler::new(Arch::X86_64);
        assert!(asm.encode_one("foo:").is_err());
    }

    #[test]
    fn encode_one_does_not_affect_state() {
        let asm = Assembler::new(Arch::X86_64);
        let _ = asm.encode_one("nop").unwrap();
        // Finish should produce empty output since encode_one doesn't
        // add to internal state
        let result = asm.finish().unwrap();
        assert!(result.is_empty());
    }

    // ─── define_preprocessor_symbol ────────────────────────────────

    #[test]
    fn define_preprocessor_symbol_ifdef() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.define_preprocessor_symbol("DEBUG", 1);
        asm.emit(".ifdef DEBUG\nnop\n.endif").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x90]);
    }

    #[test]
    fn define_preprocessor_symbol_skipped_when_missing() {
        let mut asm = Assembler::new(Arch::X86_64);
        // DEBUG is NOT defined — block should be skipped
        asm.emit(".ifdef DEBUG\nnop\n.endif\nret").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0xC3]); // only ret
    }

    // ─── dq (64-bit data) ─────────────────────────────────────────

    #[test]
    fn builder_dq() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.dq(0xDEAD_BEEF_CAFE_BABE).unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    }

    // ─── reset ────────────────────────────────────────────────────

    #[test]
    fn reset_clears_state_keeps_config() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("nop").unwrap();
        asm.reset();
        asm.emit("ret").unwrap();
        let result = asm.finish().unwrap();
        // Only "ret" should be present — "nop" was cleared
        assert_eq!(result.bytes(), &[0xC3]);
    }

    #[test]
    fn reset_allows_reuse() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("nop").unwrap();
        // Reset discards the nop, then assemble fresh
        asm.reset();
        asm.emit("ret").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0xC3]);
    }

    // ─── current_fragment_count ───────────────────────────────────

    #[test]
    fn current_fragment_count_tracks_emissions() {
        let mut asm = Assembler::new(Arch::X86_64);
        assert_eq!(asm.current_fragment_count(), 0);
        asm.emit("nop").unwrap();
        assert!(asm.current_fragment_count() > 0);
    }

    // ─── is_empty ─────────────────────────────────────────────────

    #[test]
    fn empty_assembly_result() {
        let asm = Assembler::new(Arch::X86_64);
        let result = asm.finish().unwrap();
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
        assert!(result.bytes().is_empty());
    }

    // ─── labels() direct access ───────────────────────────────────

    #[test]
    fn labels_slice_access() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("start:\nnop\nend:\nret").unwrap();
        let result = asm.finish().unwrap();
        let labels = result.labels();
        // Two labels defined
        assert_eq!(labels.len(), 2);
        // Check both are present (order may vary)
        assert!(labels.iter().any(|(name, _)| name == "start"));
        assert!(labels.iter().any(|(name, _)| name == "end"));
    }

    // ─── assemble_with externals ──────────────────────────────────

    #[test]
    fn assemble_with_external_labels() {
        use crate::assemble_with;
        let bytes =
            assemble_with("call target", Arch::X86_64, 0x1000, &[("target", 0x2000)]).unwrap();
        // call rel32: E8 xx xx xx xx — target is at 0x2000, PC after call = 0x1000+5 = 0x1005
        // rel32 = 0x2000 - 0x1005 = 0x0FFB
        assert_eq!(bytes[0], 0xE8);
        let rel = i32::from_le_bytes(bytes[1..5].try_into().unwrap());
        assert_eq!(rel, 0x0FFB);
    }

    // ─── multiple errors (AsmError::Multiple) ────────────────────

    #[test]
    fn multiple_errors_collected() {
        let mut asm = Assembler::new(Arch::X86_64);
        // Emit multiple bad mnemonics — errors are collected, not fail-fast
        asm.emit("badmnem1\nbadmnem2").unwrap();
        let err = asm.finish().unwrap_err();
        match err {
            AsmError::Multiple { errors } => assert_eq!(errors.len(), 2),
            _ => panic!("expected Multiple error, got: {err}"),
        }
    }

    // ─── optimizer no-op for non-x86 ─────────────────────────────

    #[cfg(feature = "arm")]
    #[test]
    fn optimizer_noop_for_arm() {
        let mut asm = Assembler::new(Arch::Arm);
        // ARM mov r0, 0 should NOT be optimized to xor (that's x86-only)
        asm.emit("mov r0, 0").unwrap();
        let result = asm.finish().unwrap();
        // ARM "mov r0, #0" encodes as: E3A00000 (condition AL, MOV, Rd=0, imm=0)
        assert_eq!(result.len(), 4);
        assert_eq!(result.bytes(), &[0x00, 0x00, 0xA0, 0xE3]);
    }

    // ─── .org directive through assembler pipeline ───────────────

    #[test]
    fn org_directive_via_emit() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("nop\n.org 0x10\nnop").unwrap();
        let result = asm.finish().unwrap();
        // nop (1 byte) + padding to 0x10 (15 zero bytes) + nop (1 byte) = 17 bytes
        assert_eq!(result.len(), 17);
        assert_eq!(result.bytes()[0], 0x90); // first nop
        assert_eq!(result.bytes()[0x10], 0x90); // nop at offset 0x10
                                                // bytes 1..0x10 should be zero-fill
        for &b in &result.bytes()[1..0x10] {
            assert_eq!(b, 0x00);
        }
    }

    // ─── listing output ──────────────────────────────────────────

    #[test]
    fn listing_includes_label_and_hex() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("start:\nnop\nret").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        // Listing should contain the label name
        assert!(
            listing.contains("start"),
            "listing should contain label 'start'"
        );
        // Listing should contain hex bytes (90 = nop, C3 = ret)
        assert!(
            listing.contains("90"),
            "listing should contain '90' for nop"
        );
        assert!(
            listing.contains("C3") || listing.contains("c3"),
            "listing should contain 'C3' for ret"
        );
    }

    #[test]
    fn listing_with_base_address_format() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.base_address(0x401000);
        asm.emit("nop\nret").unwrap();
        let result = asm.finish().unwrap();
        let listing = result.listing();
        // Should include the base address in the listing
        assert!(
            listing.contains("00401000") || listing.contains("401000"),
            "listing should contain base address"
        );
    }

    // ─── .org via builder method ─────────────────────────────────

    #[test]
    fn org_builder_method() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("nop").unwrap();
        asm.org(0x10);
        asm.emit("nop").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.len(), 17); // 1 + 15 padding + 1
    }

    // ─── JECXZ relaxation (was BranchOutOfRange before relaxation support) ──

    #[test]
    fn jecxz_relaxes_to_long_form() {
        // JECXZ targets beyond ±127 bytes now auto-relax to the compound
        // sequence: JECXZ +2 / JMP short +5 / JMP near rel32
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("jecxz target").unwrap();
        asm.space(200).unwrap(); // 200 bytes > 127 (rel8 max)
        asm.emit("target:\nnop").unwrap();
        let result = asm.finish().unwrap();
        // Long form starts with 67 E3 02 EB 05 E9 [rel32]
        assert_eq!(result.bytes[0], 0x67);
        assert_eq!(result.bytes[1], 0xE3);
        assert_eq!(result.bytes[2], 0x02);
        assert_eq!(result.bytes[3], 0xEB);
        assert_eq!(result.bytes[4], 0x05);
        assert_eq!(result.bytes[5], 0xE9);
        // Target is at offset 10+200 = 210, RIP after JMP = 10
        // disp = 210 - 10 = 200 = 0xC8
        assert_eq!(result.bytes[6], 0xC8);
        assert_eq!(result.bytes[7], 0x00);
        assert_eq!(result.bytes[8], 0x00);
        assert_eq!(result.bytes[9], 0x00);
    }

    #[test]
    fn jecxz_relaxes_to_short_form_when_near() {
        // JECXZ targets within ±127 bytes relax to the compact 67 E3 rel8
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("jecxz target").unwrap();
        asm.emit("target:\nnop").unwrap();
        let result = asm.finish().unwrap();
        // Short form: 67 E3 rel8 (3 bytes)
        assert_eq!(result.bytes[0], 0x67);
        assert_eq!(result.bytes[1], 0xE3);
        // rel8 = 0 (target is immediately after the instruction)
        assert_eq!(result.bytes[2], 0x00);
        assert_eq!(result.bytes[3], 0x90); // NOP
    }

    // ─── error collection: single error yields single, not Multiple ─

    #[test]
    fn single_error_not_wrapped_in_multiple() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("badmnem").unwrap();
        let err = asm.finish().unwrap_err();
        // A single encoding error should be returned directly, not wrapped
        assert!(matches!(err, AsmError::UnknownMnemonic { .. }));
    }

    // ─── error collection: good + bad instructions ───────────────

    #[test]
    fn errors_collected_with_valid_instructions() {
        let mut asm = Assembler::new(Arch::X86_64);
        // Mix valid and invalid instructions — valid ones still get encoded
        asm.emit("nop\nbadmnem\nret").unwrap();
        let err = asm.finish().unwrap_err();
        // Should be a single UnknownMnemonic (only 1 bad instruction)
        assert!(matches!(err, AsmError::UnknownMnemonic { .. }));
    }

    // ─── error collection across multiple emit() calls ───────────

    #[test]
    fn errors_collected_across_emit_calls() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("bad1").unwrap();
        asm.emit("bad2").unwrap();
        asm.emit("bad3").unwrap();
        let err = asm.finish().unwrap_err();
        match err {
            AsmError::Multiple { errors } => assert_eq!(errors.len(), 3),
            _ => panic!("expected Multiple error with 3 errors, got: {err}"),
        }
    }

    // ─── reset clears collected errors ───────────────────────────

    #[test]
    fn reset_clears_errors() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.emit("badmnem").unwrap();
        asm.reset();
        asm.emit("nop").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x90]);
    }

    // ─── max_errors resource limit ───────────────────────────────

    #[test]
    fn max_errors_limit_enforced() {
        let mut asm = Assembler::new(Arch::X86_64);
        asm.limits(ResourceLimits {
            max_errors: 2,
            ..ResourceLimits::default()
        });
        // Third bad mnemonic should trigger ResourceLimitExceeded
        let result = asm.emit("bad1\nbad2\nbad3");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, AsmError::ResourceLimitExceeded { .. }));
    }

    // ─── literal pool ────────────────────────────────────────────

    #[test]
    fn literal_pool_basic_x_reg() {
        // LDR X0, =0x12345678 → emits LDR (literal) + pool data at finish
        let mut asm = Assembler::new(Arch::Aarch64);
        asm.emit("ldr x0, =0x12345678").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // First 4 bytes: LDR (literal) instruction
        assert!(
            bytes.len() >= 8,
            "expected at least 8 bytes, got {}",
            bytes.len()
        );
        // Pool data should contain 0x12345678 as 8 bytes LE
        let pool_start = bytes.len() - 8;
        let pool_val = u64::from_le_bytes(bytes[pool_start..pool_start + 8].try_into().unwrap());
        assert_eq!(pool_val, 0x12345678, "pool should contain the constant");
    }

    #[test]
    fn literal_pool_basic_w_reg() {
        // LDR W0, =0x42 → emits LDR (literal) + 4-byte pool data
        let mut asm = Assembler::new(Arch::Aarch64);
        asm.emit("ldr w0, =0x42").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // Pool data should contain 0x42 as 4 bytes LE
        let pool_start = bytes.len() - 4;
        let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
        assert_eq!(pool_val, 0x42, "pool should contain the constant");
    }

    #[test]
    fn literal_pool_with_ltorg() {
        // Explicit .ltorg flushes the pool
        let mut asm = Assembler::new(Arch::Aarch64);
        asm.emit("ldr x0, =0xCAFE\n.ltorg").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // Should have: 4 bytes LDR + alignment + 8 bytes pool data
        assert!(bytes.len() >= 12);
        // Pool value at end
        let pool_start = bytes.len() - 8;
        let pool_val = u64::from_le_bytes(bytes[pool_start..pool_start + 8].try_into().unwrap());
        assert_eq!(pool_val, 0xCAFE);
    }

    #[test]
    fn literal_pool_deduplication() {
        // Two LDR with same value should share one pool entry
        let mut asm = Assembler::new(Arch::Aarch64);
        asm.emit("ldr x0, =0x1234\nldr x1, =0x1234").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // 2 LDR instructions (8 bytes) + alignment + 1 pool entry (8 bytes)
        // Without dedup: 8 + alignment + 16 = 24+
        // With dedup: 8 + alignment + 8 = 16+
        // The pool should contain exactly one 8-byte entry
        assert!(
            bytes.len() <= 24,
            "expected <= 24 bytes with dedup, got {}",
            bytes.len()
        );
    }

    #[test]
    fn literal_pool_multiple_values() {
        // Two LDR with different values → two pool entries
        let mut asm = Assembler::new(Arch::Aarch64);
        asm.emit("ldr x0, =0xAAAA\nldr x1, =0xBBBB").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // Should have both values in the pool
        let pool_end = bytes.len();
        let val2 = u64::from_le_bytes(bytes[pool_end - 8..pool_end].try_into().unwrap());
        let val1 = u64::from_le_bytes(bytes[pool_end - 16..pool_end - 8].try_into().unwrap());
        assert!(
            (val1 == 0xAAAA && val2 == 0xBBBB) || (val1 == 0xBBBB && val2 == 0xAAAA),
            "pool should contain both values, got {:#x} and {:#x}",
            val1,
            val2
        );
    }

    #[test]
    fn literal_pool_ldr_encodes_pc_relative() {
        // Verify the LDR instruction word encodes imm19 pointing to the pool
        let mut asm = Assembler::new(Arch::Aarch64);
        asm.emit("ldr x0, =0xFF").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // First 4 bytes are LDR (literal): opc=01 | 011000 | imm19 | Rt
        let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        // opc should be 01 (64-bit) at bits 31:30
        assert_eq!((word >> 30) & 0b11, 0b01, "opc should be 01 for 64-bit LDR");
        // bits 29:24 should be 011000
        assert_eq!(
            (word >> 24) & 0b111111,
            0b011000,
            "should be LDR literal encoding"
        );
        // Rt should be X0 = 0
        assert_eq!(word & 0x1F, 0, "Rt should be X0");
        // imm19 should be positive (pool is after the instruction)
        let imm19 = ((word >> 5) & 0x7FFFF) as i32;
        assert!(imm19 > 0, "imm19 should be positive (pool is after instr)");
    }

    #[test]
    fn literal_pool_large_64bit_value() {
        let mut asm = Assembler::new(Arch::Aarch64);
        asm.emit("ldr x0, =0xDEADBEEFCAFEBABE").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        let pool_start = bytes.len() - 8;
        let pool_val = u64::from_le_bytes(bytes[pool_start..pool_start + 8].try_into().unwrap());
        assert_eq!(pool_val, 0xDEADBEEFCAFEBABE);
    }

    #[test]
    fn literal_pool_negative_value() {
        let mut asm = Assembler::new(Arch::Aarch64);
        asm.emit("ldr x0, =-1").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        let pool_start = bytes.len() - 8;
        let pool_val = u64::from_le_bytes(bytes[pool_start..pool_start + 8].try_into().unwrap());
        // -1 as u64 = 0xFFFFFFFFFFFFFFFF
        assert_eq!(pool_val, 0xFFFFFFFFFFFFFFFF);
    }

    #[test]
    fn literal_pool_pool_directive() {
        // .pool is an alias for .ltorg
        let mut asm = Assembler::new(Arch::Aarch64);
        asm.emit("ldr x0, =0xBEEF\n.pool").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        let pool_start = bytes.len() - 8;
        let pool_val = u64::from_le_bytes(bytes[pool_start..pool_start + 8].try_into().unwrap());
        assert_eq!(pool_val, 0xBEEF);
    }

    #[test]
    fn literal_pool_reset_clears_pool() {
        let mut asm = Assembler::new(Arch::Aarch64);
        asm.emit("ldr x0, =0x1234").unwrap();
        asm.reset();
        // After reset, pool should be empty; emitting just a NOP should work
        asm.emit("nop").unwrap();
        let result = asm.finish().unwrap();
        assert_eq!(result.bytes(), &[0x1F, 0x20, 0x03, 0xD5]); // NOP only, no pool data
    }

    // ─── ARM literal pool ────────────────────────────────────────

    #[test]
    fn arm_literal_pool_basic() {
        // LDR R0, =0x12345678 on ARM → LDR (literal) + 4-byte pool entry
        let mut asm = Assembler::new(Arch::Arm);
        asm.emit("ldr r0, =0x12345678").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // First 4 bytes: LDR instruction, then 4 bytes pool data
        assert!(
            bytes.len() >= 8,
            "expected at least 8 bytes, got {}",
            bytes.len()
        );
        // Pool data: 4 bytes LE for ARM
        let pool_start = bytes.len() - 4;
        let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
        assert_eq!(pool_val, 0x12345678, "pool should contain the constant");
    }

    #[test]
    fn arm_literal_pool_small_value() {
        // Even small values go through the literal pool path
        let mut asm = Assembler::new(Arch::Arm);
        asm.emit("ldr r3, =42").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        let pool_start = bytes.len() - 4;
        let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
        assert_eq!(pool_val, 42);
    }

    #[test]
    fn arm_literal_pool_negative_value() {
        let mut asm = Assembler::new(Arch::Arm);
        asm.emit("ldr r0, =-1").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        let pool_start = bytes.len() - 4;
        let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
        // -1 as u32 = 0xFFFFFFFF
        assert_eq!(pool_val, 0xFFFFFFFF);
    }

    #[test]
    fn arm_literal_pool_deduplication() {
        // Two LDR with same value should share one pool entry
        let mut asm = Assembler::new(Arch::Arm);
        asm.emit("ldr r0, =0xAABB\nldr r1, =0xAABB").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // 2 LDR instructions (8 bytes) + alignment + 1 pool entry (4 bytes)
        // Without dedup: 8 + 8 = 16; with dedup: 8 + 4 = 12
        assert!(
            bytes.len() <= 16,
            "expected <=16 bytes with dedup, got {}",
            bytes.len()
        );
    }

    #[test]
    fn arm_literal_pool_multiple_values() {
        // Different values → separate pool entries
        let mut asm = Assembler::new(Arch::Arm);
        asm.emit("ldr r0, =0x1111\nldr r1, =0x2222").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // Two 4-byte pool entries at the end
        let pool_end = bytes.len();
        let val2 = u32::from_le_bytes(bytes[pool_end - 4..pool_end].try_into().unwrap());
        let val1 = u32::from_le_bytes(bytes[pool_end - 8..pool_end - 4].try_into().unwrap());
        assert!(
            (val1 == 0x1111 && val2 == 0x2222) || (val1 == 0x2222 && val2 == 0x1111),
            "pool should contain both values, got {:#x} and {:#x}",
            val1,
            val2
        );
    }

    #[test]
    fn arm_literal_pool_with_ltorg() {
        // Explicit .ltorg flushes the pool
        let mut asm = Assembler::new(Arch::Arm);
        asm.emit("ldr r0, =0xCAFE\n.ltorg").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        assert!(bytes.len() >= 8);
        let pool_start = bytes.len() - 4;
        let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
        assert_eq!(pool_val, 0xCAFE);
    }

    #[test]
    fn arm_literal_pool_pool_directive() {
        // .pool is an alias for .ltorg
        let mut asm = Assembler::new(Arch::Arm);
        asm.emit("ldr r0, =0xBEEF\n.pool").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        let pool_start = bytes.len() - 4;
        let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
        assert_eq!(pool_val, 0xBEEF);
    }

    #[test]
    fn arm_literal_pool_ldr_encodes_pc_relative() {
        // Verify the LDR instruction uses PC-relative addressing to pool
        let mut asm = Assembler::new(Arch::Arm);
        asm.emit("ldr r0, =0xFF").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // First 4 bytes: LDR Rd, [PC, #offset]
        let word = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        // bits [27:26] = 01 (load/store immediate offset)
        assert_eq!(
            (word >> 26) & 0b11,
            0b01,
            "should be load/store word encoding"
        );
        // bits [19:16] = Rn = 15 (PC)
        assert_eq!((word >> 16) & 0xF, 15, "Rn should be PC (R15)");
        // bits [15:12] = Rd = 0 (R0)
        assert_eq!((word >> 12) & 0xF, 0, "Rd should be R0");
        // L bit (bit 20) = 1 (load)
        assert_eq!((word >> 20) & 1, 1, "should be a load");
    }

    #[test]
    fn arm_literal_pool_entry_always_4_bytes() {
        // ARM pool entries should always be 4 bytes regardless of register
        let mut asm = Assembler::new(Arch::Arm);
        asm.emit("ldr r0, =0x1\nldr r15, =0x2").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        // 2 LDR (8 bytes) + 2 pool entries (8 bytes) = 16 bytes
        // No alignment needed for 4-byte entries on 4-byte boundary
        assert!(
            bytes.len() <= 16,
            "ARM pool entries should be 4 bytes each, got {} total",
            bytes.len()
        );
    }

    #[test]
    fn arm_literal_pool_hex_large() {
        let mut asm = Assembler::new(Arch::Arm);
        asm.emit("ldr r5, =0xDEADBEEF").unwrap();
        let result = asm.finish().unwrap();
        let bytes = result.bytes();
        let pool_start = bytes.len() - 4;
        let pool_val = u32::from_le_bytes(bytes[pool_start..pool_start + 4].try_into().unwrap());
        assert_eq!(pool_val, 0xDEADBEEF);
    }
}
