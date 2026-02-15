//! x86-64 instruction encoder.
//!
//! Encodes parsed `Instruction`s into machine-code bytes.
//! Uses a table-driven approach for opcode lookup with manual
//! ModR/M, SIB, REX prefix construction.

// VEX/SSE encoding helpers inherently require many parameters (opcode bytes,
// prefix, W-bit, register operands, etc.); suppressing this lint is the
// pragmatic choice over wrapping parameters in a struct that adds no clarity.
#![allow(clippy::too_many_arguments)]

use alloc::string::String;
#[allow(unused_imports)]
use alloc::string::ToString;
#[allow(unused_imports)]
use alloc::vec;
use alloc::vec::Vec;

use crate::error::AsmError;
#[allow(unused_imports)]
use crate::error::Span;
use crate::ir::*;

// ─── InstrBytes: stack-allocated instruction buffer ────────────────────

/// Stack-allocated instruction byte buffer — eliminates per-instruction heap
/// allocation on the encoding hot path.
///
/// x86/x86-64 instructions are at most 15 bytes; AArch64 and ARM32 are
/// fixed at 4; RISC-V pseudo-instructions expand to at most 8 words (32
/// bytes for RV64 `li` with a full 64-bit immediate).  This inline buffer
/// covers **all** architectures without touching the heap.
///
/// Capacity: 32 bytes on the stack.
#[derive(Clone)]
pub struct InstrBytes {
    data: [u8; 32],
    len: u8,
}

impl InstrBytes {
    /// Create an empty buffer.
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [0; 32],
            len: 0,
        }
    }

    /// Create a buffer pre-filled from a byte slice (max 32 bytes).
    #[inline]
    pub fn from_slice(src: &[u8]) -> Self {
        let mut buf = Self::new();
        buf.extend_from_slice(src);
        buf
    }

    /// Append a single byte.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is already full (15 bytes).
    #[inline]
    pub fn push(&mut self, byte: u8) {
        assert!(
            (self.len as usize) < 32,
            "InstrBytes overflow: cannot push beyond 32 bytes"
        );
        self.data[self.len as usize] = byte;
        self.len += 1;
    }

    /// Append a slice of bytes.
    ///
    /// # Panics
    ///
    /// Panics if appending would exceed the 15-byte capacity.
    #[inline]
    pub fn extend_from_slice(&mut self, bytes: &[u8]) {
        let start = self.len as usize;
        let end = start + bytes.len();
        assert!(
            end <= 32,
            "InstrBytes overflow: {} + {} exceeds 32-byte capacity",
            start,
            bytes.len()
        );
        self.data[start..end].copy_from_slice(bytes);
        self.len = end as u8;
    }

    /// Insert a byte at the given position, shifting subsequent bytes right.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is full or `pos` is out of bounds.
    #[inline]
    pub fn insert(&mut self, pos: usize, byte: u8) {
        let len = self.len as usize;
        assert!(
            pos <= len && len < 32,
            "InstrBytes insert: pos={} len={} out of bounds",
            pos,
            len
        );
        // Shift bytes from pos..len right by 1
        let mut i = len;
        while i > pos {
            self.data[i] = self.data[i - 1];
            i -= 1;
        }
        self.data[pos] = byte;
        self.len += 1;
    }

    /// Remove a byte at the given position, shifting subsequent bytes left.
    ///
    /// # Panics
    ///
    /// Panics if `pos` is out of bounds.
    #[inline]
    pub fn remove(&mut self, pos: usize) {
        let len = self.len as usize;
        assert!(
            pos < len,
            "InstrBytes remove: pos={} out of bounds (len={})",
            pos,
            len
        );
        let mut i = pos;
        while i + 1 < len {
            self.data[i] = self.data[i + 1];
            i += 1;
        }
        self.data[len - 1] = 0;
        self.len -= 1;
    }

    /// Number of bytes in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Convert to a heap-allocated `Vec<u8>`.
    #[inline]
    pub fn to_vec(&self) -> Vec<u8> {
        self.as_ref().to_vec()
    }
}

impl Default for InstrBytes {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl core::ops::Deref for InstrBytes {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        &self.data[..self.len as usize]
    }
}

impl core::ops::DerefMut for InstrBytes {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.data[..self.len as usize]
    }
}

impl AsRef<[u8]> for InstrBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl AsMut<[u8]> for InstrBytes {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self
    }
}

impl core::fmt::Debug for InstrBytes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl PartialEq for InstrBytes {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for InstrBytes {}

impl PartialEq<[u8]> for InstrBytes {
    fn eq(&self, other: &[u8]) -> bool {
        **self == *other
    }
}

impl PartialEq<Vec<u8>> for InstrBytes {
    fn eq(&self, other: &Vec<u8>) -> bool {
        **self == **other
    }
}

// ─── EncodedInstr ──────────────────────────────────────────

/// Result of encoding a single instruction.
#[derive(Debug, Clone)]
pub struct EncodedInstr {
    /// The machine code bytes (long form for relaxable instructions).
    pub bytes: InstrBytes,
    /// If the instruction references a label, this records it for the linker.
    pub relocation: Option<Relocation>,
    /// If present, the instruction can be shortened via branch relaxation.
    pub relax: Option<RelaxInfo>,
}

/// How the linker should patch the relocation target into the instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RelocKind {
    /// x86 RIP-relative: raw `i8`/`i32` displacement written at `offset`.
    /// `trailing_bytes` accounts for any immediate after the disp field.
    X86Relative,
    /// Absolute address written as raw LE bytes (1/2/4/8).
    Absolute,
    /// ARM32 B/BL: PC-relative offset >> 2 in bits 23:0 of the 32-bit word.
    #[cfg(feature = "arm")]
    ArmBranch24,
    /// ARM32 LDR literal (PC-relative): 12-bit offset in bits 11:0, U-bit in bit 23.
    #[cfg(feature = "arm")]
    ArmLdrLit,
    /// ARM32 ADR (data-processing immediate): 8-bit value with 4-bit rotation in bits 11:0,
    /// ADD/SUB opcode in bits 24:21, U-sense via opcode (ADD=0x4, SUB=0x2).
    #[cfg(feature = "arm")]
    ArmAdr,
    /// Thumb-2 conditional branch (16-bit): 8-bit signed offset >> 1 in bits 7:0.
    /// Range: ±256 bytes from PC+4.
    #[cfg(feature = "arm")]
    ThumbBranch8,
    /// Thumb unconditional branch (16-bit): 11-bit signed offset >> 1 in bits 10:0.
    /// Range: ±2 KB from PC+4.
    #[cfg(feature = "arm")]
    ThumbBranch11,
    /// Thumb-2 BL (32-bit): 25-bit offset across two halfwords. Range: ±16 MB.
    #[cfg(feature = "arm")]
    ThumbBl,
    /// Thumb-2 B.W (32-bit wide unconditional branch): 24-bit offset. Range: ±16 MB.
    #[cfg(feature = "arm")]
    ThumbBranchW,
    /// Thumb-2 B.cond.W (32-bit wide conditional branch): 20-bit offset. Range: ±1 MB.
    #[cfg(feature = "arm")]
    ThumbCondBranchW,
    /// Thumb LDR Rt, \[PC, #imm8×4\]: 8-bit word-aligned PC-relative literal load.
    /// Range: 0–1020 bytes forward only. PC = (instr_addr + 4) & ~3.
    #[cfg(feature = "arm")]
    ThumbLdrLit8,
    /// AArch64 B/BL: PC-relative offset >> 2 in bits 25:0 of the 32-bit word.
    #[cfg(feature = "aarch64")]
    Aarch64Jump26,
    /// AArch64 B.cond / CBZ / CBNZ: PC-relative offset >> 2 in bits 23:5.
    #[cfg(feature = "aarch64")]
    Aarch64Branch19,
    /// AArch64 TBZ / TBNZ: PC-relative offset >> 2 in bits 18:5 (14-bit imm).
    #[cfg(feature = "aarch64")]
    Aarch64Branch14,
    /// AArch64 LDR (literal): PC-relative offset >> 2 in bits 23:5.
    #[cfg(feature = "aarch64")]
    Aarch64LdrLit19,
    /// AArch64 ADR: PC-relative offset with immhi (bits 23:5) and immlo (bits 30:29).
    #[cfg(feature = "aarch64")]
    Aarch64Adr21,
    /// AArch64 ADRP: page-relative offset with immhi/immlo, target &= ~0xFFF.
    #[cfg(feature = "aarch64")]
    Aarch64Adrp,
    /// AArch64 ADRP+ADD pair (8 bytes): ADR relaxation long form.
    /// The first word is ADRP (patched with page offset), the second word
    /// is ADD (patched with lo12 = target & 0xFFF).
    #[cfg(feature = "aarch64")]
    Aarch64AdrpAddPair,
    /// RISC-V JAL: PC-relative offset in J-type immediate (bits 31:12), ±1MB range.
    #[cfg(feature = "riscv")]
    RvJal20,
    /// RISC-V B-type branch: PC-relative offset in B-type immediate (bits 31:7), ±4KB range.
    #[cfg(feature = "riscv")]
    RvBranch12,
    /// RISC-V AUIPC: upper 20 bits of PC-relative offset (bits 31:12).
    #[cfg(feature = "riscv")]
    RvAuipc20,
    /// RISC-V C-extension CB-type branch: 9-bit signed PC-relative offset (±256 B).
    #[cfg(feature = "riscv")]
    RvCBranch8,
    /// RISC-V C-extension CJ-type jump: 12-bit signed PC-relative offset (±2 KB).
    #[cfg(feature = "riscv")]
    RvCJump11,
}

/// A relocation record for unresolved labels.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Relocation {
    /// Offset within the instruction bytes where the relocation target is.
    pub offset: usize,
    /// Size of the relocation field in bytes (1, 2, or 4).
    pub size: u8,
    /// The label name to resolve.  Stored as `Rc<str>` so that cloning during
    /// relocation propagation and linker resolution is a cheap refcount bump
    /// instead of a heap allocation.
    pub label: alloc::rc::Rc<str>,
    /// How the linker patches the target address into the instruction bytes.
    pub kind: RelocKind,
    /// The addend for the relocation (constant offset).
    pub addend: i64,
    /// Number of instruction bytes that follow the relocation field.
    /// For x86 RIP-relative relocations, the CPU computes EA = RIP + disp where
    /// RIP = address of the byte AFTER the entire instruction.  When a trailing
    /// immediate follows the displacement, we need this to calculate the correct
    /// RIP at link time: `rip = reloc_addr + size + trailing_bytes`.
    pub trailing_bytes: u8,
}

/// Information for branch relaxation — allows the linker to try a shorter encoding.
///
/// When present on an [`EncodedInstr`], the linker starts with this short form
/// and only promotes to the long form (in `bytes`) when the target is out of
/// ±127 byte range.  This implements Szymanski-style monotonic growth.
#[derive(Debug, Clone)]
pub struct RelaxInfo {
    /// Complete short-form instruction bytes (opcode + placeholder rel8).
    pub short_bytes: InstrBytes,
    /// Offset of the rel8 displacement byte within `short_bytes`.
    pub short_reloc_offset: usize,
    /// Optional relocation for the short form.  When `Some`, the linker
    /// applies this relocation to `short_bytes` instead of raw byte-patching.
    /// Used for architectures like RISC-V where even the short form needs
    /// complex bitfield manipulation.
    pub short_relocation: Option<Relocation>,
}

/// Encode one instruction into machine code bytes.
/// Extract label name and addend from a label or expression operand.
///
/// Returns `Some((label, addend))` for:
/// - `Operand::Label("foo")` → `("foo", 0)`
/// - `Operand::Expression(label + N)` → `("label", N)`
///
/// Returns `None` for non-label operands or expressions with multiple labels.
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64", feature = "riscv"))]
pub(crate) fn extract_label(op: &Operand) -> Option<(&str, i64)> {
    match op {
        Operand::Label(name) => Some((name.as_str(), 0)),
        Operand::Expression(expr) => expr.label_addend(),
        _ => None,
    }
}

/// # Errors
///
/// Returns `Err(AsmError)` if the instruction mnemonic is unknown, the
/// operand combination is invalid, or the target architecture is not
/// supported.
#[inline]
pub fn encode_instruction(instr: &Instruction, arch: Arch) -> Result<EncodedInstr, AsmError> {
    #[allow(unreachable_patterns)]
    match arch {
        #[cfg(feature = "x86_64")]
        Arch::X86_64 => encode_x86_64(instr),
        #[cfg(feature = "x86")]
        Arch::X86 => encode_x86_32(instr),
        #[cfg(feature = "arm")]
        Arch::Arm | Arch::Thumb => crate::arm::encode_arm(instr, arch),
        #[cfg(feature = "aarch64")]
        Arch::Aarch64 => crate::aarch64::encode_aarch64(instr),
        #[cfg(feature = "riscv")]
        Arch::Rv32 | Arch::Rv64 => crate::riscv::encode_riscv(instr, arch),
        _ => Err(AsmError::Syntax {
            msg: alloc::format!(
                "encoder not implemented for {} (enable the feature flag)",
                arch
            ),
            span: instr.span,
        }),
    }
}

// ─── Shared x86/x86-64 helpers ───────────────────────────────

/// Emit legacy prefixes (LOCK/REP/REPNE/segment) and memory-operand
/// segment overrides into `buf`.  Returns the byte length of all emitted
/// prefix bytes (needed later for displacement scanning).
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn emit_x86_prefixes(buf: &mut InstrBytes, instr: &Instruction, ops: &[Operand]) -> usize {
    for pfx in &instr.prefixes {
        match pfx {
            Prefix::Lock => buf.push(0xF0),
            Prefix::Rep => buf.push(0xF3),
            Prefix::Repne => buf.push(0xF2),
            Prefix::SegFs => buf.push(0x64),
            Prefix::SegGs => buf.push(0x65),
        }
    }

    // Emit segment override from memory operand (must come before REX/opcode).
    // Skip if already emitted via instr.prefixes (avoid double emission).
    let has_seg_prefix = instr
        .prefixes
        .iter()
        .any(|p| matches!(p, Prefix::SegFs | Prefix::SegGs));
    if !has_seg_prefix {
        for op in ops {
            if let Operand::Memory(mem) = op {
                if let Some(seg) = mem.segment {
                    let seg_byte = match seg {
                        Register::Cs => Some(0x2E_u8),
                        Register::Ds => Some(0x3E),
                        Register::Es => Some(0x26),
                        Register::Fs => Some(0x64),
                        Register::Gs => Some(0x65),
                        Register::Ss => Some(0x36),
                        _ => None,
                    };
                    if let Some(b) = seg_byte {
                        buf.push(b);
                    }
                }
            }
        }
    }

    buf.len()
}

/// Check whether any memory operand uses 32-bit base/index registers,
/// requiring the address-size override prefix (0x67) in 64-bit mode.
///
/// In 64-bit mode the default address size is 64 bits. Using `[eax]` or
/// `[ecx+edx*4]` requires the `0x67` prefix to select 32-bit addressing.
/// This is emitted only in `encode_x86_64`, not in `encode_x86_32` where
/// 32-bit addressing is already the default.
#[cfg(feature = "x86_64")]
fn needs_addr_size_override(ops: &[Operand]) -> bool {
    for op in ops {
        if let Operand::Memory(mem) = op {
            if let Some(base) = mem.base {
                // RIP-relative is always 64-bit addressing — no override
                if base == Register::Rip {
                    continue;
                }
                if base.size_bits() == 32 {
                    return true;
                }
            }
            if let Some(idx) = mem.index {
                if idx.size_bits() == 32 {
                    return true;
                }
            }
        }
    }
    false
}

/// Validate that a LOCK prefix is only used with a memory destination operand.
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn validate_lock_prefix(instr: &Instruction, ops: &[Operand]) -> Result<(), AsmError> {
    if instr.prefixes.contains(&Prefix::Lock) {
        let has_memory_dst = matches!(ops.first(), Some(Operand::Memory(_)));
        if !has_memory_dst {
            return Err(AsmError::InvalidOperands {
                detail: String::from("LOCK prefix requires a memory destination operand"),
                span: instr.span,
            });
        }
    }
    Ok(())
}

/// If a memory operand carries a `disp_label` and the encoder didn't set a
/// relocation explicitly, scan the encoded bytes for the displacement field
/// and create the relocation.  Also computes `trailing_bytes` for
/// RIP-relative relocations.
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn propagate_disp_label_reloc(
    buf: &[u8],
    ops: &[Operand],
    prefix_len: usize,
    reloc: &mut Option<Relocation>,
) {
    if reloc.is_some() {
        return;
    }

    for op in ops {
        if let Operand::Memory(mem) = op {
            if let Some(ref label) = mem.disp_label {
                // Structurally locate the displacement offset by parsing
                // the instruction's ModR/M + SIB layout, rather than
                // byte-pattern scanning which is fragile when displacement
                // bytes accidentally match opcode/prefix bytes.
                if let Some(off) = find_disp_offset_structural(buf, prefix_len) {
                    let kind = if mem.base == Some(Register::Rip) {
                        RelocKind::X86Relative
                    } else {
                        RelocKind::Absolute
                    };
                    *reloc = Some(Relocation {
                        offset: off,
                        size: 4,
                        label: alloc::rc::Rc::from(&**label),
                        kind,
                        addend: mem.disp,
                        trailing_bytes: 0,
                    });
                }
                break; // At most one relocation per instruction
            }
        }
    }

    // Compute trailing_bytes for RIP-relative relocations
    if let Some(ref mut r) = reloc {
        if r.kind == RelocKind::X86Relative {
            let end_of_reloc = r.offset + r.size as usize;
            r.trailing_bytes = (buf.len() - end_of_reloc) as u8;
        }
    }
}

/// Returns `true` for x86 legacy prefix bytes that an individual encoder may
/// emit after the top-level `emit_x86_prefixes()` call: operand-size override
/// (0x66), address-size override (0x67), and SSE mandatory prefixes (0xF2/0xF3).
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn is_post_prefix_legacy_byte(b: u8) -> bool {
    matches!(b, 0x66 | 0x67 | 0xF2 | 0xF3)
}

/// Structurally locate the displacement field offset within an encoded x86
/// instruction by parsing the prefix → opcode → ModR/M → SIB chain.
///
/// This replaces the previous byte-pattern scanning approach (which searched
/// for `(disp as i32).to_le_bytes()` in the buffer) and is immune to false
/// matches when displacement bytes happen to equal opcode or prefix bytes.
///
/// Returns `Some(offset)` — the byte index in `buf` where the displacement
/// field starts — or `None` if the instruction has no memory displacement.
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn find_disp_offset_structural(buf: &[u8], prefix_len: usize) -> Option<usize> {
    if buf.len() <= prefix_len {
        return None;
    }

    let mut pos = prefix_len;

    // Skip legacy prefixes emitted by individual encoders (0x66 operand-size,
    // 0x67 address-size, 0xF2/0xF3 mandatory SSE prefixes).  These always
    // precede the REX/VEX/EVEX prefix and opcode.
    while pos < buf.len() && is_post_prefix_legacy_byte(buf[pos]) {
        pos += 1;
    }

    if pos >= buf.len() {
        return None;
    }

    // ── VEX / EVEX ──────────────────────────────────────────────────
    // VEX 2-byte: [C5] [RvvvvLpp]            → 1 opcode byte follows
    // VEX 3-byte: [C4] [RXBmmmmm] [WvvvvLpp] → 1 opcode byte follows
    // EVEX:       [62] [P0] [P1] [P2]         → 1 opcode byte follows
    //
    // The opcode map (0F / 0F38 / 0F3A) is encoded *inside* the VEX/EVEX
    // prefix, so exactly one opcode byte appears before ModR/M.
    let modrm_pos = if buf[pos] == 0xC5 {
        pos + 3 // C5 + 1 prefix byte + 1 opcode
    } else if buf[pos] == 0xC4 {
        pos + 4 // C4 + 2 prefix bytes + 1 opcode
    } else if buf[pos] == 0x62 {
        pos + 5 // 62 + 3 prefix bytes + 1 opcode
    } else {
        // ── Legacy encoding ─────────────────────────────────────────
        // Skip REX prefix (0x40..0x4F).  In 32-bit mode these bytes are
        // the INC/DEC short forms which only encode register operands, so
        // they never appear when a memory displacement needs locating.
        if (buf[pos] & 0xF0) == 0x40 {
            pos += 1;
        }
        if pos >= buf.len() {
            return None;
        }

        // Parse opcode escape:
        //   0x0F 0x38 xx → 3-byte opcode (ModR/M follows xx)
        //   0x0F 0x3A xx → 3-byte opcode (ModR/M follows xx)
        //   0x0F xx      → 2-byte opcode (ModR/M follows xx)
        //   xx           → 1-byte opcode (ModR/M follows xx)
        if buf[pos] == 0x0F {
            pos += 1;
            if pos >= buf.len() {
                return None;
            }
            if buf[pos] == 0x38 || buf[pos] == 0x3A {
                pos += 2; // escape extension byte + opcode byte
            } else {
                pos += 1; // opcode byte
            }
        } else {
            pos += 1; // single-byte opcode
        }

        pos
    };

    // ── Parse ModR/M ────────────────────────────────────────────────
    if modrm_pos >= buf.len() {
        return None;
    }

    let modrm = buf[modrm_pos];
    let mod_bits = (modrm >> 6) & 0x03;
    let rm = modrm & 0x07;

    // mod=11 → register-direct, no memory displacement
    if mod_bits == 0x03 {
        return None;
    }

    let mut disp_pos = modrm_pos + 1;

    // r/m=100 → SIB byte follows ModR/M
    if rm == 0x04 {
        if disp_pos >= buf.len() {
            return None;
        }
        let sib_base = buf[disp_pos] & 0x07;
        disp_pos += 1; // skip SIB

        match mod_bits {
            0b00 if sib_base == 0x05 => Some(disp_pos), // [index*scale + disp32]
            0b00 => None,                               // [base + index*scale]
            0b01 | 0b10 => Some(disp_pos),              // [base + index*scale + disp]
            _ => None,
        }
    } else {
        match mod_bits {
            0b00 if rm == 0x05 => Some(disp_pos), // [RIP + disp32] or [disp32]
            0b00 => None,                         // [base], no displacement
            0b01 | 0b10 => Some(disp_pos),        // [base + disp8/32]
            _ => None,
        }
    }
}

/// Fix up `trailing_bytes` for any RIP-relative relocation.  Must be called
/// AFTER the instruction buffer is final — works for both dispatch-created
/// and disp_label-propagated relocations.
#[cfg(feature = "x86_64")]
fn fixup_rip_trailing_bytes(buf: &[u8], reloc: &mut Option<Relocation>) {
    if let Some(ref mut r) = reloc {
        if r.kind == RelocKind::X86Relative {
            let end_of_reloc = r.offset + r.size as usize;
            r.trailing_bytes = (buf.len() - end_of_reloc) as u8;
        }
    }
}

#[cfg(feature = "x86_64")]
fn encode_x86_64(instr: &Instruction) -> Result<EncodedInstr, AsmError> {
    let mut buf = InstrBytes::new();
    let mut reloc: Option<Relocation> = None;
    let mut relax_info: Option<RelaxInfo> = None;

    let ops = &instr.operands;
    let prefix_len = emit_x86_prefixes(&mut buf, instr, ops);

    // Address-size override: emit 0x67 when memory operands use 32-bit
    // base/index registers in 64-bit mode (e.g. `mov eax, [ecx]`).
    if needs_addr_size_override(ops) {
        buf.push(0x67);
    }

    let mnemonic = instr.mnemonic.as_str();

    match crate::x86::dispatch_x86_64(mnemonic, &mut buf, ops, instr, &mut reloc, &mut relax_info) {
        Some(Ok(())) => {}
        Some(Err(e)) => return Err(e),
        None => {
            return Err(AsmError::UnknownMnemonic {
                mnemonic: String::from(mnemonic),
                arch: crate::error::ArchName::X86_64,
                span: instr.span,
            });
        }
    }

    validate_lock_prefix(instr, ops)?;
    propagate_disp_label_reloc(&buf, ops, prefix_len, &mut reloc);

    // Compute trailing_bytes for RIP-relative relocations: the number of
    // instruction bytes that follow the relocation field.  The CPU computes
    // EA = RIP + disp32, where RIP = address past the ENTIRE instruction.
    fixup_rip_trailing_bytes(&buf, &mut reloc);

    Ok(EncodedInstr {
        bytes: buf,
        relocation: reloc,
        relax: relax_info,
    })
}

// ─── x86-32 encoder ──────────────────────────────────────────

/// Validate that an instruction's operands are legal for 32-bit protected mode.
#[cfg(feature = "x86")]
fn validate_x86_32(instr: &Instruction) -> Result<(), AsmError> {
    for op in &instr.operands {
        match op {
            Operand::Register(reg) => {
                if reg.size_bits() == 64 {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from("64-bit registers are not available in 32-bit mode"),
                        span: instr.span,
                    });
                }
                if reg.is_extended() {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from(
                            "extended registers (R8-R15) are not available in 32-bit mode",
                        ),
                        span: instr.span,
                    });
                }
                if reg.requires_rex_for_byte() {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from(
                            "SPL/BPL/SIL/DIL are not available in 32-bit mode (require REX)",
                        ),
                        span: instr.span,
                    });
                }
                if matches!(reg, Register::Rip) {
                    return Err(AsmError::InvalidOperands {
                        detail: String::from(
                            "RIP-relative addressing is not available in 32-bit mode",
                        ),
                        span: instr.span,
                    });
                }
            }
            Operand::Memory(mem) => {
                if let Some(base) = mem.base {
                    if base == Register::Rip {
                        return Err(AsmError::InvalidOperands {
                            detail: String::from(
                                "RIP-relative addressing is not available in 32-bit mode",
                            ),
                            span: instr.span,
                        });
                    }
                    if base.size_bits() == 64 || base.is_extended() {
                        return Err(AsmError::InvalidOperands {
                            detail: String::from(
                                "64-bit/extended registers cannot be used as memory base in 32-bit mode",
                            ),
                            span: instr.span,
                        });
                    }
                }
                if let Some(idx) = mem.index {
                    if idx.size_bits() == 64 || idx.is_extended() {
                        return Err(AsmError::InvalidOperands {
                            detail: String::from(
                                "64-bit/extended registers cannot be used as memory index in 32-bit mode",
                            ),
                            span: instr.span,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    if instr.mnemonic == "movsxd" {
        return Err(AsmError::UnknownMnemonic {
            mnemonic: String::from("movsxd"),
            arch: crate::error::ArchName::X86,
            span: instr.span,
        });
    }

    Ok(())
}

/// x86-32 (protected mode) encoder.
///
/// Shares the instruction encoding logic with x86-64 but:
/// - Rejects 64-bit registers, extended registers (R8-R15), RIP-relative
/// - push/pop accept 32-bit registers (default operand size)
/// - All other instructions produce identical byte sequences
#[cfg(feature = "x86")]
fn encode_x86_32(instr: &Instruction) -> Result<EncodedInstr, AsmError> {
    validate_x86_32(instr)?;

    let mut buf = InstrBytes::new();
    let mut reloc: Option<Relocation> = None;
    let mut relax_info: Option<RelaxInfo> = None;

    let ops = &instr.operands;
    let prefix_len = emit_x86_prefixes(&mut buf, instr, ops);
    let mnemonic = instr.mnemonic.as_str();

    // Handle push/pop/inc/dec specially — 32-bit mode has dedicated short forms.
    match mnemonic {
        "push" => {
            encode_push_32(&mut buf, ops, instr, &mut reloc)?;
            return Ok(EncodedInstr {
                bytes: buf,
                relocation: reloc,
                relax: relax_info,
            });
        }
        "pop" => {
            encode_pop_32(&mut buf, ops, instr)?;
            return Ok(EncodedInstr {
                bytes: buf,
                relocation: reloc,
                relax: relax_info,
            });
        }
        "inc" | "dec" => {
            // In 32-bit mode, INC r16/r32 (0x40+rd) and DEC r16/r32 (0x48+rd)
            // are single-byte short forms. These opcodes are repurposed as
            // REX prefixes in 64-bit mode, so this path is x86-32 only.
            // For 8-bit regs and memory operands, fall through to the
            // generic encoder which uses the 0xFE/0xFF ModR/M form.
            if ops.len() == 1 {
                if let Operand::Register(reg) = &ops[0] {
                    let size = reg_size(*reg);
                    if size == 16 || size == 32 {
                        if size == 16 {
                            buf.push(0x66);
                        }
                        let base = if mnemonic == "inc" { 0x40 } else { 0x48 };
                        buf.push(base + reg.base_code());
                        return Ok(EncodedInstr {
                            bytes: buf,
                            relocation: reloc,
                            relax: relax_info,
                        });
                    }
                }
            }
            // 8-bit / memory forms — fall through to generic dispatch
        }
        _ => {}
    }

    // Everything else: reuse the x86-64 dispatch.
    // Since we've validated no 64-bit/extended registers are present,
    // the encoder functions will not emit REX prefixes — producing
    // valid 32-bit code.
    match crate::x86::dispatch_x86_64(mnemonic, &mut buf, ops, instr, &mut reloc, &mut relax_info) {
        Some(Ok(())) => {}
        Some(Err(e)) => return Err(e),
        None => {
            return Err(AsmError::UnknownMnemonic {
                mnemonic: String::from(mnemonic),
                arch: crate::error::ArchName::X86,
                span: instr.span,
            });
        }
    }

    validate_lock_prefix(instr, ops)?;
    propagate_disp_label_reloc(&buf, ops, prefix_len, &mut reloc);

    Ok(EncodedInstr {
        bytes: buf,
        relocation: reloc,
        relax: relax_info,
    })
}

// ─── x86-16 encoder (real mode) ─────────────────────────────

/// x86-16 (real mode) encoder.
///
/// In 16-bit mode the default operand size is 16 bits and the default
/// address size is 16 bits.  The `0x66` prefix switches operand size to
/// 32 bits, and `0x67` switches address size to 32 bits — the reverse
/// of 32-bit protected mode.
///
/// Implementation: reuse the 32-bit encoder (which handles push/pop/inc/dec
/// short forms, segment registers, etc.) and then toggle the `0x66` prefix:
///
///   - If `0x66` is present → remove it (16-bit is now the default)
///   - If `0x66` is absent AND the instruction uses 32-bit GPRs → add it
#[cfg(feature = "x86")]
pub fn encode_instruction_16(instr: &Instruction) -> Result<EncodedInstr, AsmError> {
    // Step 1: Encode using the 32-bit encoder (handles push/pop/inc/dec
    // short forms, segment registers, LOCK prefix validation, etc.)
    let mut result = encode_x86_32(instr)?;

    // Step 2: Toggle the 0x66 operand-size prefix.
    // The 32-bit encoder adds 0x66 for 16-bit operands (non-default in
    // 32-bit mode).  In 16-bit mode the semantics reverse: 16-bit is
    // default (remove prefix) and 32-bit needs the prefix (add it).
    toggle_operand_size_prefix_16(&mut result.bytes, &instr.operands, &mut result.relocation);

    Ok(result)
}

/// Toggle the 0x66 operand-size prefix for 16-bit mode encoding.
///
/// In 16-bit mode: remove 0x66 if present (16-bit is now default),
/// or add 0x66 if absent and instruction uses 32-bit registers.
#[cfg(feature = "x86")]
fn toggle_operand_size_prefix_16(
    buf: &mut InstrBytes,
    ops: &[Operand],
    reloc: &mut Option<Relocation>,
) {
    // Scan for a 0x66 byte in the prefix region (before the first non-prefix byte).
    let mut found_66_at = None;
    for i in 0..buf.len() {
        let b = buf[i];
        if b == 0x66 {
            found_66_at = Some(i);
            break;
        }
        // Stop at the first non-prefix byte (0x67 is addr-size, still a prefix)
        if !is_legacy_prefix(b) && b != 0x67 {
            break;
        }
    }

    if let Some(pos) = found_66_at {
        // Remove 0x66 — in 16-bit mode, 16-bit operands don't need it
        buf.remove(pos);
        if let Some(ref mut r) = reloc {
            if r.offset > pos {
                r.offset -= 1;
            }
        }
    } else {
        // Check if the instruction uses 32-bit GP registers (not EIP).
        // If so, add 0x66 — in 16-bit mode, 32-bit operands need the override.
        let has_32bit_gpr = ops.iter().any(|op| {
            if let Operand::Register(r) = op {
                r.size_bits() == 32 && !matches!(r, Register::Eip)
            } else {
                false
            }
        });

        // Also check for DWORD memory size annotation (e.g., mov dword [bx], 1).
        let has_dword_mem = ops.iter().any(
            |op| matches!(op, Operand::Memory(m) if m.size == Some(crate::ir::OperandSize::Dword)),
        );

        if has_32bit_gpr || has_dword_mem {
            // Insert 0x66 after any existing prefixes
            let mut insert_pos = 0;
            for i in 0..buf.len() {
                let b = buf[i];
                if is_legacy_prefix(b) || b == 0x67 {
                    insert_pos = i + 1;
                } else {
                    break;
                }
            }
            buf.insert(insert_pos, 0x66);
            if let Some(ref mut r) = reloc {
                if r.offset >= insert_pos {
                    r.offset += 1;
                }
            }
        }
    }
}

/// Check if a byte is a legacy x86 prefix.
#[cfg(feature = "x86")]
#[inline]
fn is_legacy_prefix(b: u8) -> bool {
    matches!(
        b,
        0xF0 | 0xF2 | 0xF3 | 0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67
    )
}
#[cfg(feature = "x86")]
fn encode_push_32(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands("push", "expected 1 operand", instr.span));
    }
    match &ops[0] {
        Operand::Register(reg) => {
            let size = reg.size_bits();
            // Segment register push — all 6 valid in 32-bit mode
            match reg {
                Register::Es => {
                    buf.push(0x06);
                    return Ok(());
                }
                Register::Cs => {
                    buf.push(0x0E);
                    return Ok(());
                }
                Register::Ss => {
                    buf.push(0x16);
                    return Ok(());
                }
                Register::Ds => {
                    buf.push(0x1E);
                    return Ok(());
                }
                Register::Fs => {
                    buf.push(0x0F);
                    buf.push(0xA0);
                    return Ok(());
                }
                Register::Gs => {
                    buf.push(0x0F);
                    buf.push(0xA8);
                    return Ok(());
                }
                _ => {}
            }
            if size == 8 {
                return Err(invalid_operands(
                    "push",
                    "push does not accept 8-bit registers",
                    instr.span,
                ));
            }
            if size == 16 {
                buf.push(0x66); // operand size override for 16-bit
            }
            buf.push(0x50 + reg.base_code());
        }
        Operand::Immediate(imm) => {
            if *imm >= i8::MIN as i128 && *imm <= i8::MAX as i128 {
                buf.push(0x6A);
                buf.push(*imm as i8 as u8);
            } else if *imm >= i32::MIN as i128 && *imm <= u32::MAX as i128 {
                buf.push(0x68);
                buf.extend_from_slice(&(*imm as i32).to_le_bytes());
            } else {
                return Err(invalid_operands(
                    "push",
                    "immediate value out of range for push (must fit in 32 bits)",
                    instr.span,
                ));
            }
        }
        Operand::Memory(mem) => {
            let msize = mem.size.map(|s| s.bits()).unwrap_or(32);
            if msize == 16 {
                buf.push(0x66);
            }
            buf.push(0xFF);
            emit_mem_modrm(buf, 6, mem);
        }
        op @ (Operand::Label(_) | Operand::Expression(_)) => {
            let Some((label, addend)) = extract_label(op) else {
                return Err(invalid_operands("push", "unsupported operand", instr.span));
            };
            buf.push(0x68);
            let reloc_off = buf.len();
            buf.extend_from_slice(&0i32.to_le_bytes());
            *reloc = Some(Relocation {
                offset: reloc_off,
                size: 4,
                label: alloc::rc::Rc::from(label),
                kind: RelocKind::Absolute,
                addend,
                trailing_bytes: 0,
            });
        }
        _ => return Err(invalid_operands("push", "unsupported operand", instr.span)),
    }
    Ok(())
}

/// x86-32 pop: accepts 32-bit and 16-bit GP registers + FS/GS/DS/ES/SS.
#[cfg(feature = "x86")]
fn encode_pop_32(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands("pop", "expected 1 operand", instr.span));
    }
    match &ops[0] {
        Operand::Register(reg) => {
            let size = reg.size_bits();
            match reg {
                Register::Es => {
                    buf.push(0x07);
                    return Ok(());
                }
                Register::Ss => {
                    buf.push(0x17);
                    return Ok(());
                }
                Register::Ds => {
                    buf.push(0x1F);
                    return Ok(());
                }
                Register::Fs => {
                    buf.push(0x0F);
                    buf.push(0xA1);
                    return Ok(());
                }
                Register::Gs => {
                    buf.push(0x0F);
                    buf.push(0xA9);
                    return Ok(());
                }
                Register::Cs => {
                    return Err(invalid_operands("pop", "cannot pop into CS", instr.span));
                }
                _ => {}
            }
            if size == 8 {
                return Err(invalid_operands(
                    "pop",
                    "pop does not accept 8-bit registers",
                    instr.span,
                ));
            }
            if size == 16 {
                buf.push(0x66);
            }
            buf.push(0x58 + reg.base_code());
        }
        Operand::Memory(mem) => {
            let msize = mem.size.map(|s| s.bits()).unwrap_or(32);
            if msize == 16 {
                buf.push(0x66);
            }
            buf.push(0x8F);
            emit_mem_modrm(buf, 0, mem);
        }
        _ => return Err(invalid_operands("pop", "unsupported operand", instr.span)),
    }
    Ok(())
}

// ─── REX / ModR/M / SIB helpers ──────────────────────────────

/// Build a REX prefix byte.
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn rex(w: bool, r: bool, x: bool, b: bool) -> u8 {
    let mut val: u8 = 0x40;
    if w {
        val |= 0x08;
    }
    if r {
        val |= 0x04;
    }
    if x {
        val |= 0x02;
    }
    if b {
        val |= 0x01;
    }
    val
}

/// Whether a REX prefix with at least one flag is needed.
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn needs_rex(w: bool, r: bool, x: bool, b: bool) -> bool {
    w || r || x || b
}

/// Build ModR/M byte.
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn modrm(mod_: u8, reg: u8, rm: u8) -> u8 {
    (mod_ << 6) | ((reg & 7) << 3) | (rm & 7)
}

/// Build SIB byte.
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn sib(scale: u8, index: u8, base: u8) -> u8 {
    let ss = match scale {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => 0,
    };
    (ss << 6) | ((index & 7) << 3) | (base & 7)
}

/// Get the operand size from a register as u8 (for GP registers only, panics for vector).
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn reg_size(reg: Register) -> u8 {
    let s = reg.size_bits();
    debug_assert!(s <= 128, "reg_size() used on vector register wider than u8");
    s as u8
}

/// Check if using a high-byte register (AH, BH, CH, DH) together with any operand
/// that requires a REX prefix.  On x86-64, a REX byte changes the meaning of
/// register codes 4-7 from AH/CH/DH/BH to SPL/BPL/SIL/DIL, so the two are
/// incompatible.  Returns an error if the conflict is detected.
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn check_high_byte_rex_conflict(
    regs: &[Register],
    span: crate::error::Span,
) -> Result<(), AsmError> {
    let has_high = regs.iter().any(|r| r.is_high_byte());
    let needs_rex = regs
        .iter()
        .any(|r| r.is_extended() || r.requires_rex_for_byte() || r.size_bits() == 64);
    if has_high && needs_rex {
        return Err(AsmError::InvalidOperands {
            detail: String::from(
                "high-byte registers (AH, BH, CH, DH) cannot be used with REX-requiring operands (64-bit regs, extended regs R8-R15, SPL/BPL/SIL/DIL)"
            ),
            span,
        });
    }
    Ok(())
}

/// Emit REX prefix if needed, then opcode + ModR/M for reg,reg.
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn emit_rr(
    buf: &mut InstrBytes,
    opcode: &[u8],
    dst: Register,
    src: Register,
    span: crate::error::Span,
) -> Result<(), AsmError> {
    let size = reg_size(dst);
    let w = size == 64;
    let r = src.is_extended();
    let b = dst.is_extended();

    // Validate high-byte / REX conflict
    if size == 8 {
        check_high_byte_rex_conflict(&[dst, src], span)?;
    }

    // 16-bit operand size prefix
    if size == 16 {
        buf.push(0x66);
    }

    // REX
    let need_rex =
        needs_rex(w, r, false, b) || dst.requires_rex_for_byte() || src.requires_rex_for_byte();
    if need_rex {
        buf.push(rex(w, r, false, b));
    }

    buf.extend_from_slice(opcode);
    buf.push(modrm(0b11, src.base_code(), dst.base_code()));
    Ok(())
}

/// If the memory operand has a `disp_label`, create a relocation entry.
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn set_mem_reloc(
    reloc: &mut Option<Relocation>,
    mem: &MemoryOperand,
    disp_offset: Option<usize>,
    buf_len: usize,
) {
    if let Some(ref label) = mem.disp_label {
        *reloc = Some(Relocation {
            offset: disp_offset.unwrap_or(buf_len),
            size: 4,
            label: alloc::rc::Rc::from(&**label),
            kind: if mem.base == Some(Register::Rip) {
                RelocKind::X86Relative
            } else {
                RelocKind::Absolute
            },
            addend: mem.disp,
            trailing_bytes: 0, // updated by encode_x86_64 after instruction is complete
        });
    }
}

/// Emit ModR/M + SIB + displacement for a memory operand.
/// Returns the offset where a relocation displacement starts (if any).
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn emit_mem_modrm(
    buf: &mut InstrBytes,
    reg_field: u8,
    mem: &MemoryOperand,
) -> Option<usize> {
    // NOTE: Segment override prefix is emitted in encode_x86_64 BEFORE the
    // REX/opcode bytes. Do NOT emit it here (after the opcode).

    let base = mem.base;
    let index = mem.index;
    let disp = mem.disp;

    // RIP-relative addressing: [rip + disp32]
    if base == Some(Register::Rip) && index.is_none() {
        buf.push(modrm(0b00, reg_field, 0b101));
        let reloc_offset = buf.len();
        buf.extend_from_slice(&(disp as i32).to_le_bytes());
        return Some(reloc_offset);
    }

    // Absolute address / displacement only: [disp32]
    if base.is_none() && index.is_none() {
        // In 64-bit mode, we need SIB to encode absolute address
        buf.push(modrm(0b00, reg_field, 0b100));
        buf.push(sib(1, 0b100, 0b101));
        let reloc_offset = buf.len();
        buf.extend_from_slice(&(disp as i32).to_le_bytes());
        return Some(reloc_offset);
    }

    // SIB index-only: [index*scale + disp32] — no base register.
    // Must use mod=00, base=101 (means "no base, disp32 follows").
    if let (None, Some(idx_reg)) = (base, index) {
        buf.push(modrm(0b00, reg_field, 0b100));
        buf.push(sib(mem.scale, idx_reg.base_code(), 0b101));
        let reloc_offset = buf.len();
        buf.extend_from_slice(&(disp as i32).to_le_bytes());
        return Some(reloc_offset);
    }

    // SAFETY: At this point base is guaranteed Some — displacement-only
    // (mod=00, r/m=5) and index-only (SIB with base=5) both returned early
    // above.  Every remaining path uses a base register.
    let base = base?;

    // Determine if we need SIB
    let need_sib = index.is_some() || base.base_code() == 4; // RSP/R12 need SIB

    let (mod_bits, disp_size) = if disp == 0 && base.base_code() != 5 {
        // mod=00, no displacement (unless base is RBP/R13)
        (0b00, 0)
    } else if (-128..=127).contains(&disp) {
        (0b01, 1)
    } else {
        (0b10, 4)
    };

    if need_sib {
        let idx_reg = index.unwrap_or(Register::Rsp); // 0b100 = no index

        buf.push(modrm(mod_bits, reg_field, 0b100));
        buf.push(sib(mem.scale, idx_reg.base_code(), base.base_code()));
    } else {
        buf.push(modrm(mod_bits, reg_field, base.base_code()));
    }

    let reloc_offset = if disp_size > 0 { Some(buf.len()) } else { None };

    match disp_size {
        1 => buf.push(disp as i8 as u8),
        4 => buf.extend_from_slice(&(disp as i32).to_le_bytes()),
        _ => {}
    }

    reloc_offset
}

/// Emit REX prefix for a reg+mem operation.
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn emit_rex_for_reg_mem(
    buf: &mut InstrBytes,
    reg: Register,
    mem: &MemoryOperand,
) -> Result<(), AsmError> {
    let w = reg.size_bits() == 64;
    let r = reg.is_extended();
    let x = mem.index.is_some_and(|r| r.is_extended());
    let b = mem.base.is_some_and(|r| r.is_extended());

    if reg.size_bits() == 16 {
        buf.push(0x66);
    }

    // Validate high-byte / REX conflict: AH/BH/CH/DH + extended base/index
    if reg.size_bits() == 8 && reg.is_high_byte() {
        let mem_needs_rex = x || b;
        if mem_needs_rex {
            return Err(AsmError::InvalidOperands {
                detail: String::from(
                    "high-byte registers (AH, BH, CH, DH) cannot be used with memory operands requiring REX prefix (R8-R15 base/index)"
                ),
                span: crate::error::Span { line: 0, col: 0, offset: 0, len: 0 },
            });
        }
    }

    let need = needs_rex(w, r, x, b) || reg.requires_rex_for_byte();
    if need {
        buf.push(rex(w, r, x, b));
    }
    Ok(())
}

/// Emit REX prefix for a /digit+mem operation (no separate reg operand).
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn emit_rex_for_digit_mem(buf: &mut InstrBytes, size: u8, mem: &MemoryOperand) {
    let w = size == 64;
    let x = mem.index.is_some_and(|r| r.is_extended());
    let b = mem.base.is_some_and(|r| r.is_extended());

    if size == 16 {
        buf.push(0x66);
    }
    if needs_rex(w, false, x, b) {
        buf.push(rex(w, false, x, b));
    }
}

// ─── Instruction encoders ─────────────────────────────────────

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_nop(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.is_empty() {
        buf.push(0x90);
        Ok(())
    } else {
        Err(AsmError::InvalidOperands {
            detail: String::from("nop takes no operands"),
            span: instr.span,
        })
    }
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_multibyte_nop(buf: &mut InstrBytes, mnemonic: &str) -> Result<(), AsmError> {
    let n: usize = mnemonic[3..].parse().unwrap_or(1);
    // Intel recommended multi-byte NOP sequences
    match n {
        2 => buf.extend_from_slice(&[0x66, 0x90]),
        3 => buf.extend_from_slice(&[0x0F, 0x1F, 0x00]),
        4 => buf.extend_from_slice(&[0x0F, 0x1F, 0x40, 0x00]),
        5 => buf.extend_from_slice(&[0x0F, 0x1F, 0x44, 0x00, 0x00]),
        6 => buf.extend_from_slice(&[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00]),
        7 => buf.extend_from_slice(&[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00]),
        8 => buf.extend_from_slice(&[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]),
        9 => buf.extend_from_slice(&[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]),
        _ => buf.push(0x90),
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_int(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            "int",
            "expected one immediate operand",
            instr.span,
        ));
    }
    match &ops[0] {
        Operand::Immediate(3) => buf.push(0xCC), // INT 3 → single byte
        Operand::Immediate(n) if *n >= 0 && *n <= 255 => {
            buf.push(0xCD);
            buf.push(*n as u8);
        }
        _ => {
            return Err(invalid_operands(
                "int",
                "expected immediate 0-255",
                instr.span,
            ))
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_ret(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.is_empty() {
        buf.push(0xC3); // Near return
    } else if ops.len() == 1 {
        match &ops[0] {
            Operand::Immediate(n) if *n >= 0 && *n <= 65535 => {
                buf.push(0xC2); // Near return with stack pop
                buf.extend_from_slice(&(*n as u16).to_le_bytes());
            }
            _ => {
                return Err(invalid_operands(
                    "ret",
                    "expected immediate 0-65535",
                    instr.span,
                ))
            }
        }
    } else {
        return Err(invalid_operands(
            "ret",
            "expected 0 or 1 operands",
            instr.span,
        ));
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_retf(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.is_empty() {
        buf.push(0xCB); // Far return
    } else if ops.len() == 1 {
        match &ops[0] {
            Operand::Immediate(n) if *n >= 0 && *n <= 65535 => {
                buf.push(0xCA); // Far return with stack pop
                buf.extend_from_slice(&(*n as u16).to_le_bytes());
            }
            _ => {
                return Err(invalid_operands(
                    "retf",
                    "expected immediate 0-65535",
                    instr.span,
                ))
            }
        }
    } else {
        return Err(invalid_operands(
            "retf",
            "expected 0 or 1 operands",
            instr.span,
        ));
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_mov(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands("mov", "expected 2 operands", instr.span));
    }

    match (&ops[0], &ops[1]) {
        // mov r, r
        (Operand::Register(dst), Operand::Register(src)) => {
            let size = reg_size(*dst);
            if size != reg_size(*src) {
                return Err(invalid_operands("mov", "operand size mismatch", instr.span));
            }
            let opcode = if size == 8 {
                &[0x88u8] as &[u8]
            } else {
                &[0x89u8]
            };
            emit_rr(buf, opcode, *dst, *src, instr.span)?;
        }

        // mov r, imm
        (Operand::Register(dst), Operand::Immediate(imm)) => {
            encode_mov_reg_imm(buf, *dst, *imm, instr.span)?;
        }

        // mov r, label  /  mov r, label+offset
        (Operand::Register(dst), op @ (Operand::Label(_) | Operand::Expression(_))) => {
            let Some((label, addend)) = extract_label(op) else {
                return Err(invalid_operands(
                    "mov",
                    "expected label expression",
                    instr.span,
                ));
            };
            // movabs r64, imm64 with relocation
            let size = reg_size(*dst);
            if size == 64 {
                let w = true;
                let b = dst.is_extended();
                buf.push(rex(w, false, false, b));
                buf.push(0xB8 + dst.base_code());
                let reloc_off = buf.len();
                buf.extend_from_slice(&0u64.to_le_bytes());
                *reloc = Some(Relocation {
                    offset: reloc_off,
                    size: 8,
                    label: alloc::rc::Rc::from(label),
                    kind: RelocKind::Absolute,
                    addend,
                    trailing_bytes: 0,
                });
            } else {
                return Err(invalid_operands(
                    "mov",
                    "label operand requires 64-bit register",
                    instr.span,
                ));
            }
        }

        // mov r, [mem]
        (Operand::Register(dst), Operand::Memory(mem)) => {
            let size = reg_size(*dst);
            let opcode: u8 = if size == 8 { 0x8A } else { 0x8B };
            emit_rex_for_reg_mem(buf, *dst, mem)?;
            buf.push(opcode);
            let reloc_off = emit_mem_modrm(buf, dst.base_code(), mem);
            if let Some(ref label) = mem.disp_label {
                *reloc = Some(Relocation {
                    offset: reloc_off.unwrap_or(buf.len()),
                    size: 4,
                    label: alloc::rc::Rc::from(&**label),
                    kind: if mem.base == Some(Register::Rip) {
                        RelocKind::X86Relative
                    } else {
                        RelocKind::Absolute
                    },
                    addend: mem.disp,
                    trailing_bytes: 0,
                });
            }
        }

        // mov [mem], r
        (Operand::Memory(mem), Operand::Register(src)) => {
            let size = reg_size(*src);
            let opcode: u8 = if size == 8 { 0x88 } else { 0x89 };
            emit_rex_for_reg_mem(buf, *src, mem)?;
            buf.push(opcode);
            let disp_off = emit_mem_modrm(buf, src.base_code(), mem);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
        }

        // mov [mem], imm
        (Operand::Memory(mem), Operand::Immediate(imm)) => {
            let size = instr
                .size_hint
                .map_or(mem.size.map_or(32u8, |s| s.bits() as u8), |s| {
                    s.bits() as u8
                });
            // mov r/m64, imm32 sign-extends — reject values that don't fit
            if size == 64 {
                let v = *imm;
                if v > i64::from(i32::MAX) as i128 || v < i64::from(i32::MIN) as i128 {
                    return Err(invalid_operands(
                        "mov",
                        "immediate too large for mov [mem], imm (max sign-extended imm32); use mov reg, imm64 + mov [mem], reg",
                        instr.span,
                    ));
                }
            }
            let opcode: u8 = if size == 8 { 0xC6 } else { 0xC7 };
            emit_rex_for_digit_mem(buf, size, mem);
            buf.push(opcode);
            let disp_off = emit_mem_modrm(buf, 0, mem); // /0
            set_mem_reloc(reloc, mem, disp_off, buf.len());
            emit_imm(buf, *imm, if size > 32 { 32 } else { size }); // max imm32 for mov r/m, imm
        }

        _ => {
            return Err(invalid_operands(
                "mov",
                "unsupported operand combination",
                instr.span,
            ))
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_mov_reg_imm(
    buf: &mut InstrBytes,
    dst: Register,
    imm: i128,
    span: Span,
) -> Result<(), AsmError> {
    let size = reg_size(dst);

    match size {
        8 => {
            let b = dst.is_extended();
            let need = b || dst.requires_rex_for_byte();
            if need {
                buf.push(rex(false, false, false, b));
            }
            buf.push(0xB0 + dst.base_code());
            buf.push(imm as u8);
        }
        16 => {
            buf.push(0x66);
            let b = dst.is_extended();
            if b {
                buf.push(rex(false, false, false, b));
            }
            buf.push(0xB8 + dst.base_code());
            buf.extend_from_slice(&(imm as u16).to_le_bytes());
        }
        32 => {
            let b = dst.is_extended();
            if b {
                buf.push(rex(false, false, false, b));
            }
            buf.push(0xB8 + dst.base_code());
            buf.extend_from_slice(&(imm as u32).to_le_bytes());
        }
        64 => {
            // Check if we can use shorter encoding
            let b = dst.is_extended();
            if imm >= 0 && imm <= u32::MAX as i128 {
                // mov r32, imm32 (zero-extends to r64)
                if b {
                    buf.push(rex(false, false, false, true));
                }
                buf.push(0xB8 + dst.base_code());
                buf.extend_from_slice(&(imm as u32).to_le_bytes());
            } else if imm >= i32::MIN as i128 && imm <= i32::MAX as i128 {
                // mov r64, sign-extended imm32
                buf.push(rex(true, false, false, b));
                buf.push(0xC7);
                buf.push(modrm(0b11, 0, dst.base_code()));
                buf.extend_from_slice(&(imm as i32).to_le_bytes());
            } else {
                // movabs r64, imm64
                buf.push(rex(true, false, false, b));
                buf.push(0xB8 + dst.base_code());
                buf.extend_from_slice(&(imm as u64).to_le_bytes());
            }
        }
        _ => {
            return Err(AsmError::InvalidOperands {
                detail: String::from("unsupported register size for mov immediate"),
                span,
            });
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_lea(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands("lea", "expected 2 operands", instr.span));
    }
    match (&ops[0], &ops[1]) {
        (Operand::Register(dst), Operand::Memory(mem)) => {
            emit_rex_for_reg_mem(buf, *dst, mem)?;
            buf.push(0x8D);
            let reloc_off = emit_mem_modrm(buf, dst.base_code(), mem);
            if let Some(ref label) = mem.disp_label {
                *reloc = Some(Relocation {
                    offset: reloc_off.unwrap_or(buf.len()),
                    size: 4,
                    label: alloc::rc::Rc::from(&**label),
                    kind: if mem.base == Some(Register::Rip) {
                        RelocKind::X86Relative
                    } else {
                        RelocKind::Absolute
                    },
                    addend: mem.disp,
                    trailing_bytes: 0,
                });
            }
        }
        _ => return Err(invalid_operands("lea", "expected reg, [mem]", instr.span)),
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_push(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands("push", "expected 1 operand", instr.span));
    }
    match &ops[0] {
        Operand::Register(reg) => {
            let size = reg.size_bits();
            // Segment register push (only FS and GS in 64-bit mode)
            match reg {
                Register::Fs => {
                    buf.push(0x0F);
                    buf.push(0xA0);
                    return Ok(());
                }
                Register::Gs => {
                    buf.push(0x0F);
                    buf.push(0xA8);
                    return Ok(());
                }
                Register::Cs | Register::Ds | Register::Es | Register::Ss => {
                    return Err(invalid_operands(
                        "push",
                        "CS/DS/ES/SS push not valid in 64-bit mode",
                        instr.span,
                    ));
                }
                _ => {}
            }
            if size == 8 || size == 32 {
                return Err(invalid_operands(
                    "push",
                    "push requires 64-bit or 16-bit register in 64-bit mode",
                    instr.span,
                ));
            }
            let b = reg.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            if b {
                buf.push(rex(false, false, false, true));
            }
            buf.push(0x50 + reg.base_code());
        }
        Operand::Immediate(imm) => {
            if *imm >= i8::MIN as i128 && *imm <= i8::MAX as i128 {
                buf.push(0x6A);
                buf.push(*imm as i8 as u8);
            } else if *imm >= i32::MIN as i128 && *imm <= u32::MAX as i128 {
                buf.push(0x68);
                buf.extend_from_slice(&(*imm as i32).to_le_bytes());
            } else {
                return Err(invalid_operands(
                    "push",
                    "immediate value out of range for push (must fit in 32 bits)",
                    instr.span,
                ));
            }
        }
        Operand::Memory(mem) => {
            // push defaults to 64-bit operand size — REX.W is redundant.
            // Still need REX.B/X for extended base/index registers.
            emit_rex_for_digit_mem(buf, 0, mem);
            buf.push(0xFF);
            emit_mem_modrm(buf, 6, mem); // /6
        }
        op @ (Operand::Label(_) | Operand::Expression(_)) => {
            let Some((label, addend)) = extract_label(op) else {
                return Err(invalid_operands("push", "unsupported operand", instr.span));
            };
            // push imm32 with relocation
            buf.push(0x68);
            let reloc_off = buf.len();
            buf.extend_from_slice(&0i32.to_le_bytes());
            *reloc = Some(Relocation {
                offset: reloc_off,
                size: 4,
                label: alloc::rc::Rc::from(label),
                kind: RelocKind::Absolute,
                addend,
                trailing_bytes: 0,
            });
        }
        _ => return Err(invalid_operands("push", "unsupported operand", instr.span)),
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_pop(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands("pop", "expected 1 operand", instr.span));
    }
    match &ops[0] {
        Operand::Register(reg) => {
            let size = reg.size_bits();
            // Segment register pop (only FS and GS in 64-bit mode)
            match reg {
                Register::Fs => {
                    buf.push(0x0F);
                    buf.push(0xA1);
                    return Ok(());
                }
                Register::Gs => {
                    buf.push(0x0F);
                    buf.push(0xA9);
                    return Ok(());
                }
                Register::Cs | Register::Ds | Register::Es | Register::Ss => {
                    return Err(invalid_operands(
                        "pop",
                        "CS/DS/ES/SS pop not valid in 64-bit mode",
                        instr.span,
                    ));
                }
                _ => {}
            }
            if size == 8 || size == 32 {
                return Err(invalid_operands(
                    "pop",
                    "pop requires 64-bit or 16-bit register in 64-bit mode",
                    instr.span,
                ));
            }
            let b = reg.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            if b {
                buf.push(rex(false, false, false, true));
            }
            buf.push(0x58 + reg.base_code());
        }
        Operand::Memory(mem) => {
            // pop defaults to 64-bit operand size — REX.W is redundant.
            emit_rex_for_digit_mem(buf, 0, mem);
            buf.push(0x8F);
            emit_mem_modrm(buf, 0, mem);
        }
        _ => return Err(invalid_operands("pop", "unsupported operand", instr.span)),
    }
    Ok(())
}

/// Encode ALU instructions: add/or/adc/sbb/and/sub/xor/cmp.
/// `alu_num` is 0=add, 1=or, 2=adc, 3=sbb, 4=and, 5=sub, 6=xor, 7=cmp.
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_alu(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    alu_num: u8,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 2 operands",
            instr.span,
        ));
    }

    match (&ops[0], &ops[1]) {
        // r/m, r
        (Operand::Register(dst), Operand::Register(src)) => {
            let size = reg_size(*dst);
            let base_opcode = if size == 8 {
                alu_num * 8
            } else {
                alu_num * 8 + 1
            };
            emit_rr(buf, &[base_opcode], *dst, *src, instr.span)?;
        }

        // r, imm
        (Operand::Register(dst), Operand::Immediate(imm)) => {
            encode_alu_reg_imm(buf, *dst, *imm, alu_num)?;
        }

        // r, [mem]
        (Operand::Register(dst), Operand::Memory(mem)) => {
            let size = reg_size(*dst);
            let opcode: u8 = if size == 8 {
                alu_num * 8 + 2
            } else {
                alu_num * 8 + 3
            };
            emit_rex_for_reg_mem(buf, *dst, mem)?;
            buf.push(opcode);
            let disp_off = emit_mem_modrm(buf, dst.base_code(), mem);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
        }

        // [mem], r
        (Operand::Memory(mem), Operand::Register(src)) => {
            let size = reg_size(*src);
            let opcode: u8 = if size == 8 {
                alu_num * 8
            } else {
                alu_num * 8 + 1
            };
            emit_rex_for_reg_mem(buf, *src, mem)?;
            buf.push(opcode);
            let disp_off = emit_mem_modrm(buf, src.base_code(), mem);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
        }

        // [mem], imm
        (Operand::Memory(mem), Operand::Immediate(imm)) => {
            let size = instr
                .size_hint
                .map_or(mem.size.map_or(32u8, |s| s.bits() as u8), |s| {
                    s.bits() as u8
                });
            let disp_off = encode_alu_mem_imm(buf, mem, *imm, alu_num, size)?;
            set_mem_reloc(reloc, mem, disp_off, buf.len());
        }

        // al/ax/eax/rax, imm (special short forms)
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "unsupported operand combination",
                instr.span,
            ));
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_alu_reg_imm(
    buf: &mut InstrBytes,
    dst: Register,
    imm: i128,
    alu_num: u8,
) -> Result<(), AsmError> {
    let size = reg_size(dst);

    // Special case: al/ax/eax/rax, imm (short form)
    if dst.base_code() == 0 && !dst.is_extended() && size == 8 {
        buf.push(alu_num * 8 + 4);
        buf.push(imm as u8);
        return Ok(());
    }

    if size == 8 {
        let b = dst.is_extended();
        let need = b || dst.requires_rex_for_byte();
        if need {
            buf.push(rex(false, false, false, b));
        }
        buf.push(0x80);
        buf.push(modrm(0b11, alu_num, dst.base_code()));
        buf.push(imm as u8);
    } else if imm >= i8::MIN as i128 && imm <= i8::MAX as i128 {
        // Sign-extended imm8
        let w = size == 64;
        let b = dst.is_extended();
        if size == 16 {
            buf.push(0x66);
        }
        if needs_rex(w, false, false, b) {
            buf.push(rex(w, false, false, b));
        }
        buf.push(0x83);
        buf.push(modrm(0b11, alu_num, dst.base_code()));
        buf.push(imm as i8 as u8);
    } else {
        // Full immediate
        let w = size == 64;
        let b = dst.is_extended();
        if size == 16 {
            buf.push(0x66);
        }
        if needs_rex(w, false, false, b) {
            buf.push(rex(w, false, false, b));
        }

        // Special case: eax/rax can use shorter opcode
        if dst.base_code() == 0 && !dst.is_extended() {
            buf.push(alu_num * 8 + 5);
        } else {
            buf.push(0x81);
            buf.push(modrm(0b11, alu_num, dst.base_code()));
        }
        let imm_size = if size > 32 { 32 } else { size };
        emit_imm(buf, imm, imm_size);
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_alu_mem_imm(
    buf: &mut InstrBytes,
    mem: &MemoryOperand,
    imm: i128,
    alu_num: u8,
    size: u8,
) -> Result<Option<usize>, AsmError> {
    if size == 8 {
        emit_rex_for_digit_mem(buf, size, mem);
        buf.push(0x80);
        let disp_off = emit_mem_modrm(buf, alu_num, mem);
        buf.push(imm as u8);
        Ok(disp_off)
    } else if imm >= i8::MIN as i128 && imm <= i8::MAX as i128 {
        emit_rex_for_digit_mem(buf, size, mem);
        buf.push(0x83);
        let disp_off = emit_mem_modrm(buf, alu_num, mem);
        buf.push(imm as i8 as u8);
        Ok(disp_off)
    } else {
        emit_rex_for_digit_mem(buf, size, mem);
        buf.push(0x81);
        let disp_off = emit_mem_modrm(buf, alu_num, mem);
        let imm_size = if size > 32 { 32 } else { size };
        emit_imm(buf, imm, imm_size);
        Ok(disp_off)
    }
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_test(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands("test", "expected 2 operands", instr.span));
    }
    match (&ops[0], &ops[1]) {
        (Operand::Register(dst), Operand::Register(src)) => {
            let size = reg_size(*dst);
            let opcode = if size == 8 { 0x84u8 } else { 0x85u8 };
            emit_rr(buf, &[opcode], *dst, *src, instr.span)?;
        }
        (Operand::Register(dst), Operand::Immediate(imm)) => {
            let size = reg_size(*dst);
            // Short form for AL/AX/EAX/RAX
            if dst.base_code() == 0 && !dst.is_extended() && size == 8 {
                buf.push(0xA8);
                buf.push(*imm as u8);
            } else if dst.base_code() == 0 && !dst.is_extended() && size > 8 {
                let w = size == 64;
                if size == 16 {
                    buf.push(0x66);
                }
                if w {
                    buf.push(rex(true, false, false, false));
                }
                buf.push(0xA9);
                let imm_size = if size > 32 { 32 } else { size };
                emit_imm(buf, *imm, imm_size);
            } else {
                let w = size == 64;
                let b = dst.is_extended();
                if size == 16 {
                    buf.push(0x66);
                }
                let need = needs_rex(w, false, false, b) || dst.requires_rex_for_byte();
                if need {
                    buf.push(rex(w, false, false, b));
                }
                buf.push(if size == 8 { 0xF6 } else { 0xF7 });
                buf.push(modrm(0b11, 0, dst.base_code()));
                let imm_size = if size == 8 {
                    8
                } else if size > 32 {
                    32
                } else {
                    size
                };
                emit_imm(buf, *imm, imm_size);
            }
        }
        (Operand::Memory(mem), Operand::Register(src)) => {
            let size = reg_size(*src);
            let opcode = if size == 8 { 0x84u8 } else { 0x85u8 };
            emit_rex_for_reg_mem(buf, *src, mem)?;
            buf.push(opcode);
            let disp_off = emit_mem_modrm(buf, src.base_code(), mem);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
        }
        (Operand::Memory(mem), Operand::Immediate(imm)) => {
            let size = instr
                .size_hint
                .map_or(mem.size.map_or(32u8, |s| s.bits() as u8), |s| {
                    s.bits() as u8
                });
            let opcode = if size == 8 { 0xF6u8 } else { 0xF7u8 };
            emit_rex_for_digit_mem(buf, size, mem);
            buf.push(opcode);
            let disp_off = emit_mem_modrm(buf, 0, mem); // /0
            let imm_size = if size == 8 {
                8
            } else if size > 32 {
                32
            } else {
                size
            };
            emit_imm(buf, *imm, imm_size);
            set_mem_reloc(reloc, mem, disp_off, buf.len());
        }
        _ => {
            return Err(invalid_operands(
                "test",
                "unsupported operand combination",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// Encode unary instructions: NOT, NEG, MUL, DIV, IDIV.
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_unary(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    digit: u8,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 operand",
            instr.span,
        ));
    }
    match &ops[0] {
        Operand::Register(reg) => {
            let size = reg_size(*reg);
            let w = size == 64;
            let b = reg.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            let need = needs_rex(w, false, false, b) || reg.requires_rex_for_byte();
            if need {
                buf.push(rex(w, false, false, b));
            }
            buf.push(if size == 8 { 0xF6 } else { 0xF7 });
            buf.push(modrm(0b11, digit, reg.base_code()));
        }
        Operand::Memory(mem) => {
            let size = instr
                .size_hint
                .map_or(mem.size.map_or(32u8, |s| s.bits() as u8), |s| {
                    s.bits() as u8
                });
            emit_rex_for_digit_mem(buf, size, mem);
            buf.push(if size == 8 { 0xF6 } else { 0xF7 });
            emit_mem_modrm(buf, digit, mem);
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected register or memory operand",
                instr.span,
            ))
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_imul(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    match ops.len() {
        1 => {
            // One-operand IMUL: EDX:EAX = EAX * r/m
            encode_unary(buf, ops, instr, 5)
        }
        2 => {
            // Two-operand IMUL: r = r * r/m
            match (&ops[0], &ops[1]) {
                (Operand::Register(dst), Operand::Register(src)) => {
                    let size = reg_size(*dst);
                    if size == 8 {
                        return Err(invalid_operands(
                            "imul",
                            "8-bit operands not supported for 2/3-operand IMUL",
                            instr.span,
                        ));
                    }
                    let w = size == 64;
                    let r = dst.is_extended();
                    let b = src.is_extended();
                    if size == 16 {
                        buf.push(0x66);
                    }
                    if needs_rex(w, r, false, b) {
                        buf.push(rex(w, r, false, b));
                    }
                    buf.push(0x0F);
                    buf.push(0xAF);
                    buf.push(modrm(0b11, dst.base_code(), src.base_code()));
                }
                (Operand::Register(dst), Operand::Memory(mem)) => {
                    if reg_size(*dst) == 8 {
                        return Err(invalid_operands(
                            "imul",
                            "8-bit operands not supported for 2/3-operand IMUL",
                            instr.span,
                        ));
                    }
                    emit_rex_for_reg_mem(buf, *dst, mem)?;
                    buf.push(0x0F);
                    buf.push(0xAF);
                    emit_mem_modrm(buf, dst.base_code(), mem);
                }
                _ => {
                    return Err(invalid_operands(
                        "imul",
                        "unsupported operand combination",
                        instr.span,
                    ))
                }
            }
            Ok(())
        }
        3 => {
            // Three-operand IMUL: r = r/m * imm
            match (&ops[0], &ops[1], &ops[2]) {
                (Operand::Register(dst), Operand::Register(src), Operand::Immediate(imm)) => {
                    let size = reg_size(*dst);
                    if size == 8 {
                        return Err(invalid_operands(
                            "imul",
                            "8-bit operands not supported for 2/3-operand IMUL",
                            instr.span,
                        ));
                    }
                    let w = size == 64;
                    let r = dst.is_extended();
                    let b = src.is_extended();
                    if size == 16 {
                        buf.push(0x66);
                    }
                    if needs_rex(w, r, false, b) {
                        buf.push(rex(w, r, false, b));
                    }

                    if *imm >= i8::MIN as i128 && *imm <= i8::MAX as i128 {
                        buf.push(0x6B);
                        buf.push(modrm(0b11, dst.base_code(), src.base_code()));
                        buf.push(*imm as i8 as u8);
                    } else {
                        buf.push(0x69);
                        buf.push(modrm(0b11, dst.base_code(), src.base_code()));
                        let imm_size = if size > 32 { 32 } else { size };
                        emit_imm(buf, *imm, imm_size);
                    }
                }
                (Operand::Register(dst), Operand::Memory(mem), Operand::Immediate(imm)) => {
                    let size = reg_size(*dst);
                    if size == 8 {
                        return Err(invalid_operands(
                            "imul",
                            "8-bit operands not supported for 2/3-operand IMUL",
                            instr.span,
                        ));
                    }
                    emit_rex_for_reg_mem(buf, *dst, mem)?;

                    if *imm >= i8::MIN as i128 && *imm <= i8::MAX as i128 {
                        buf.push(0x6B);
                        emit_mem_modrm(buf, dst.base_code(), mem);
                        buf.push(*imm as i8 as u8);
                    } else {
                        buf.push(0x69);
                        emit_mem_modrm(buf, dst.base_code(), mem);
                        let imm_size = if size > 32 { 32 } else { size };
                        emit_imm(buf, *imm, imm_size);
                    }
                }
                _ => {
                    return Err(invalid_operands(
                        "imul",
                        "expected reg, r/m, imm",
                        instr.span,
                    ))
                }
            }
            Ok(())
        }
        _ => Err(invalid_operands(
            "imul",
            "expected 1-3 operands",
            instr.span,
        )),
    }
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_inc_dec(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    digit: u8,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 operand",
            instr.span,
        ));
    }
    match &ops[0] {
        Operand::Register(reg) => {
            let size = reg_size(*reg);
            let w = size == 64;
            let b = reg.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            let need = needs_rex(w, false, false, b) || reg.requires_rex_for_byte();
            if need {
                buf.push(rex(w, false, false, b));
            }
            buf.push(if size == 8 { 0xFE } else { 0xFF });
            buf.push(modrm(0b11, digit, reg.base_code()));
        }
        Operand::Memory(mem) => {
            let size = instr
                .size_hint
                .map_or(mem.size.map_or(32u8, |s| s.bits() as u8), |s| {
                    s.bits() as u8
                });
            emit_rex_for_digit_mem(buf, size, mem);
            buf.push(if size == 8 { 0xFE } else { 0xFF });
            emit_mem_modrm(buf, digit, mem);
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected register or memory",
                instr.span,
            ))
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_shift(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    digit: u8,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 2 operands",
            instr.span,
        ));
    }
    match (&ops[0], &ops[1]) {
        (Operand::Register(dst), Operand::Immediate(1)) => {
            let size = reg_size(*dst);
            let w = size == 64;
            let b = dst.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            let need = needs_rex(w, false, false, b) || dst.requires_rex_for_byte();
            if need {
                buf.push(rex(w, false, false, b));
            }
            buf.push(if size == 8 { 0xD0 } else { 0xD1 });
            buf.push(modrm(0b11, digit, dst.base_code()));
        }
        (Operand::Register(dst), Operand::Immediate(imm)) => {
            let size = reg_size(*dst);
            let w = size == 64;
            let b = dst.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            let need = needs_rex(w, false, false, b) || dst.requires_rex_for_byte();
            if need {
                buf.push(rex(w, false, false, b));
            }
            buf.push(if size == 8 { 0xC0 } else { 0xC1 });
            buf.push(modrm(0b11, digit, dst.base_code()));
            buf.push(*imm as u8);
        }
        (Operand::Register(dst), Operand::Register(Register::Cl)) => {
            let size = reg_size(*dst);
            let w = size == 64;
            let b = dst.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            let need = needs_rex(w, false, false, b) || dst.requires_rex_for_byte();
            if need {
                buf.push(rex(w, false, false, b));
            }
            buf.push(if size == 8 { 0xD2 } else { 0xD3 });
            buf.push(modrm(0b11, digit, dst.base_code()));
        }
        (Operand::Memory(mem), Operand::Immediate(1)) => {
            let size = instr
                .size_hint
                .map_or(mem.size.map_or(32u8, |s| s.bits() as u8), |s| {
                    s.bits() as u8
                });
            let w = size == 64;
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, false, x, b) {
                buf.push(rex(w, false, x, b));
            }
            buf.push(if size == 8 { 0xD0 } else { 0xD1 });
            emit_mem_modrm(buf, digit, mem);
        }
        (Operand::Memory(mem), Operand::Immediate(imm)) => {
            let size = instr
                .size_hint
                .map_or(mem.size.map_or(32u8, |s| s.bits() as u8), |s| {
                    s.bits() as u8
                });
            let w = size == 64;
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, false, x, b) {
                buf.push(rex(w, false, x, b));
            }
            buf.push(if size == 8 { 0xC0 } else { 0xC1 });
            emit_mem_modrm(buf, digit, mem);
            buf.push(*imm as u8);
        }
        (Operand::Memory(mem), Operand::Register(Register::Cl)) => {
            let size = instr
                .size_hint
                .map_or(mem.size.map_or(32u8, |s| s.bits() as u8), |s| {
                    s.bits() as u8
                });
            let w = size == 64;
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, false, x, b) {
                buf.push(rex(w, false, x, b));
            }
            buf.push(if size == 8 { 0xD2 } else { 0xD3 });
            emit_mem_modrm(buf, digit, mem);
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected r/m, imm or r/m, cl",
                instr.span,
            ))
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_jmp(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
    relax: &mut Option<RelaxInfo>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands("jmp", "expected 1 operand", instr.span));
    }
    match &ops[0] {
        op @ (Operand::Label(_) | Operand::Expression(_)) => {
            let Some((label, addend)) = extract_label(op) else {
                return Err(invalid_operands("jmp", "expected label", instr.span));
            };
            // Long form: E9 rel32 (5 bytes) — linker may relax to EB rel8 (2 bytes)
            buf.push(0xE9);
            let reloc_off = buf.len();
            buf.extend_from_slice(&0i32.to_le_bytes());
            *reloc = Some(Relocation {
                offset: reloc_off,
                size: 4,
                label: alloc::rc::Rc::from(label),
                kind: RelocKind::X86Relative,
                addend,
                trailing_bytes: 0,
            });
            // Short form for relaxation (only when no addend — with addend
            // the displacement arithmetic is the same but we still offer it)
            *relax = Some(RelaxInfo {
                short_bytes: InstrBytes::from_slice(&[0xEB, 0x00]),
                short_reloc_offset: 1,
                short_relocation: None,
            });
        }
        Operand::Immediate(target) => {
            // Short jump with known offset
            buf.push(0xE9);
            buf.extend_from_slice(&(*target as i32).to_le_bytes());
        }
        Operand::Register(reg) => {
            let b = reg.is_extended();
            if b {
                buf.push(rex(false, false, false, true));
            }
            buf.push(0xFF);
            buf.push(modrm(0b11, 4, reg.base_code()));
        }
        Operand::Memory(mem) => {
            // jmp defaults to 64-bit operand size — REX.W is redundant.
            emit_rex_for_digit_mem(buf, 0, mem);
            buf.push(0xFF);
            emit_mem_modrm(buf, 4, mem);
        }
        _ => return Err(invalid_operands("jmp", "unsupported operand", instr.span)),
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_call(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands("call", "expected 1 operand", instr.span));
    }
    match &ops[0] {
        op @ (Operand::Label(_) | Operand::Expression(_)) => {
            let Some((label, addend)) = extract_label(op) else {
                return Err(invalid_operands("call", "expected label", instr.span));
            };
            buf.push(0xE8);
            let reloc_off = buf.len();
            buf.extend_from_slice(&0i32.to_le_bytes());
            *reloc = Some(Relocation {
                offset: reloc_off,
                size: 4,
                label: alloc::rc::Rc::from(label),
                kind: RelocKind::X86Relative,
                addend,
                trailing_bytes: 0,
            });
        }
        Operand::Register(reg) => {
            let b = reg.is_extended();
            if b {
                buf.push(rex(false, false, false, true));
            }
            buf.push(0xFF);
            buf.push(modrm(0b11, 2, reg.base_code()));
        }
        Operand::Memory(mem) => {
            // call defaults to 64-bit operand size — REX.W is redundant.
            emit_rex_for_digit_mem(buf, 0, mem);
            buf.push(0xFF);
            emit_mem_modrm(buf, 2, mem);
        }
        _ => return Err(invalid_operands("call", "unsupported operand", instr.span)),
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_jcc(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    cc: u8,
    reloc: &mut Option<Relocation>,
    relax: &mut Option<RelaxInfo>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 operand",
            instr.span,
        ));
    }
    match &ops[0] {
        op @ (Operand::Label(_) | Operand::Expression(_)) => {
            let Some((label, addend)) = extract_label(op) else {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "expected label or offset",
                    instr.span,
                ));
            };
            // Long form: 0F 8x rel32 (6 bytes) — linker may relax to 7x rel8 (2 bytes)
            buf.push(0x0F);
            buf.push(0x80 + cc);
            let reloc_off = buf.len();
            buf.extend_from_slice(&0i32.to_le_bytes());
            *reloc = Some(Relocation {
                offset: reloc_off,
                size: 4,
                label: alloc::rc::Rc::from(label),
                kind: RelocKind::X86Relative,
                addend,
                trailing_bytes: 0,
            });
            // Short form for relaxation
            *relax = Some(RelaxInfo {
                short_bytes: InstrBytes::from_slice(&[0x70 + cc, 0x00]),
                short_reloc_offset: 1,
                short_relocation: None,
            });
        }
        Operand::Immediate(off) => {
            // With known numeric offset — emit near form
            buf.push(0x0F);
            buf.push(0x80 + cc);
            buf.extend_from_slice(&(*off as i32).to_le_bytes());
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected label or offset",
                instr.span,
            ))
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_loop(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    opcode: u8,
    reloc: &mut Option<Relocation>,
    relax: &mut Option<RelaxInfo>,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 operand",
            instr.span,
        ));
    }
    match &ops[0] {
        op @ (Operand::Label(_) | Operand::Expression(_)) => {
            let Some((label, addend)) = extract_label(op) else {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "expected label",
                    instr.span,
                ));
            };
            // Long form (9 bytes): LOOPx +2 / JMP_short +5 / JMP_near rel32
            //
            // If the target is within ±127 bytes the linker relaxes to the
            // short form (2 bytes): LOOPx rel8.
            //
            // Long-form layout:
            //   [0] opcode  LOOPx
            //   [1] 0x02    rel8 = +2  → skips the JMP_short, lands on JMP_near
            //   [2] 0xEB    JMP short
            //   [3] 0x05    rel8 = +5  → skips the JMP_near (CX was zero / cond false)
            //   [4] 0xE9    JMP near
            //   [5..9] rel32 placeholder → target label
            buf.push(opcode);
            buf.push(0x02);
            buf.push(0xEB);
            buf.push(0x05);
            buf.push(0xE9);
            let reloc_off = buf.len();
            buf.extend_from_slice(&0i32.to_le_bytes());
            *reloc = Some(Relocation {
                offset: reloc_off,
                size: 4,
                label: alloc::rc::Rc::from(label),
                kind: RelocKind::X86Relative,
                addend,
                trailing_bytes: 0,
            });
            // Short form for relaxation: LOOPx rel8 (2 bytes)
            *relax = Some(RelaxInfo {
                short_bytes: InstrBytes::from_slice(&[opcode, 0x00]),
                short_reloc_offset: 1,
                short_relocation: None,
            });
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected label",
                instr.span,
            ))
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_setcc(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    cc: u8,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 1 operand",
            instr.span,
        ));
    }
    match &ops[0] {
        Operand::Register(reg) => {
            if reg.size_bits() != 8 {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "SETcc requires an 8-bit register operand",
                    instr.span,
                ));
            }
            let b = reg.is_extended();
            let need = b || reg.requires_rex_for_byte();
            if need {
                buf.push(rex(false, false, false, b));
            }
            buf.push(0x0F);
            buf.push(0x90 + cc);
            buf.push(modrm(0b11, 0, reg.base_code()));
        }
        Operand::Memory(mem) => {
            emit_rex_for_digit_mem(buf, 8, mem);
            buf.push(0x0F);
            buf.push(0x90 + cc);
            emit_mem_modrm(buf, 0, mem);
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected register or memory",
                instr.span,
            ))
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_cmovcc(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    cc: u8,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 2 operands",
            instr.span,
        ));
    }
    match (&ops[0], &ops[1]) {
        (Operand::Register(dst), Operand::Register(src)) => {
            let size = reg_size(*dst);
            if size == 8 {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "CMOVcc requires 16/32/64-bit operands",
                    instr.span,
                ));
            }
            let w = size == 64;
            let r = dst.is_extended();
            let b = src.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, r, false, b) {
                buf.push(rex(w, r, false, b));
            }
            buf.push(0x0F);
            buf.push(0x40 + cc);
            buf.push(modrm(0b11, dst.base_code(), src.base_code()));
        }
        (Operand::Register(dst), Operand::Memory(mem)) => {
            if reg_size(*dst) == 8 {
                return Err(invalid_operands(
                    &instr.mnemonic,
                    "CMOVcc requires 16/32/64-bit operands",
                    instr.span,
                ));
            }
            emit_rex_for_reg_mem(buf, *dst, mem)?;
            buf.push(0x0F);
            buf.push(0x40 + cc);
            emit_mem_modrm(buf, dst.base_code(), mem);
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected reg, r/m",
                instr.span,
            ))
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_movzx(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands("movzx", "expected 2 operands", instr.span));
    }
    match (&ops[0], &ops[1]) {
        (Operand::Register(dst), Operand::Register(src)) => {
            let dst_size = reg_size(*dst);
            let src_size = reg_size(*src);
            let w = dst_size == 64;
            let r = dst.is_extended();
            let b = src.is_extended();
            if dst_size == 16 {
                buf.push(0x66);
            }
            let need = needs_rex(w, r, false, b) || src.requires_rex_for_byte();
            if need {
                buf.push(rex(w, r, false, b));
            }
            buf.push(0x0F);
            buf.push(if src_size == 8 { 0xB6 } else { 0xB7 }); // B6=byte, B7=word
            buf.push(modrm(0b11, dst.base_code(), src.base_code()));
        }
        (Operand::Register(dst), Operand::Memory(mem)) => {
            let src_size = instr
                .size_hint
                .map_or(mem.size.map_or(8u8, |s| s.bits() as u8), |s| s.bits() as u8);
            emit_rex_for_reg_mem(buf, *dst, mem)?;
            buf.push(0x0F);
            buf.push(if src_size == 8 { 0xB6 } else { 0xB7 });
            emit_mem_modrm(buf, dst.base_code(), mem);
        }
        _ => return Err(invalid_operands("movzx", "expected reg, r/m", instr.span)),
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_movsx(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands("movsx", "expected 2 operands", instr.span));
    }
    match (&ops[0], &ops[1]) {
        (Operand::Register(dst), Operand::Register(src)) => {
            let dst_size = reg_size(*dst);
            let src_size = reg_size(*src);
            let w = dst_size == 64;
            let r = dst.is_extended();
            let b = src.is_extended();
            if dst_size == 16 {
                buf.push(0x66);
            }
            let need = needs_rex(w, r, false, b) || src.requires_rex_for_byte();
            if need {
                buf.push(rex(w, r, false, b));
            }
            if src_size == 32 {
                // movsxd: 63 /r
                buf.push(0x63);
            } else {
                buf.push(0x0F);
                buf.push(if src_size == 8 { 0xBE } else { 0xBF });
            }
            buf.push(modrm(0b11, dst.base_code(), src.base_code()));
        }
        (Operand::Register(dst), Operand::Memory(mem)) => {
            let src_size = instr
                .size_hint
                .map_or(mem.size.map_or(8u8, |s| s.bits() as u8), |s| s.bits() as u8);
            let dst_size = reg_size(*dst);
            if src_size == 32 {
                // movsxd: REX.W 63 /r
                let w = dst_size == 64;
                let r = dst.is_extended();
                let x = mem.index.is_some_and(|r| r.is_extended());
                let b = mem.base.is_some_and(|r| r.is_extended());
                if dst_size == 16 {
                    buf.push(0x66);
                }
                if needs_rex(w, r, x, b) {
                    buf.push(rex(w, r, x, b));
                }
                buf.push(0x63);
            } else {
                emit_rex_for_reg_mem(buf, *dst, mem)?;
                buf.push(0x0F);
                buf.push(if src_size == 8 { 0xBE } else { 0xBF });
            }
            emit_mem_modrm(buf, dst.base_code(), mem);
        }
        _ => return Err(invalid_operands("movsx", "expected reg, r/m", instr.span)),
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_xchg(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands("xchg", "expected 2 operands", instr.span));
    }
    match (&ops[0], &ops[1]) {
        (Operand::Register(dst), Operand::Register(src)) => {
            let size = reg_size(*dst);
            // Special case: xchg ax, r16 or xchg r16, ax → 66 90+rd
            if size == 16
                && ((dst.base_code() == 0 && !dst.is_extended())
                    || (src.base_code() == 0 && !src.is_extended()))
            {
                let other = if dst.base_code() == 0 && !dst.is_extended() {
                    *src
                } else {
                    *dst
                };
                let b = other.is_extended();
                buf.push(0x66);
                if b {
                    buf.push(rex(false, false, false, b));
                }
                buf.push(0x90 + other.base_code());
                return Ok(());
            }
            // Special case: xchg eax/rax, reg or xchg reg, eax/rax → 90+rd
            if size >= 32
                && ((dst.base_code() == 0 && !dst.is_extended())
                    || (src.base_code() == 0 && !src.is_extended()))
            {
                let other = if dst.base_code() == 0 && !dst.is_extended() {
                    *src
                } else {
                    *dst
                };
                let w = size == 64;
                let b = other.is_extended();
                if needs_rex(w, false, false, b) {
                    buf.push(rex(w, false, false, b));
                }
                buf.push(0x90 + other.base_code());
                return Ok(());
            }
            let opcode = if size == 8 { 0x86u8 } else { 0x87u8 };
            emit_rr(buf, &[opcode], *dst, *src, instr.span)?;
        }
        // xchg reg, [mem]
        (Operand::Register(reg), Operand::Memory(mem)) => {
            let size = reg_size(*reg);
            let opcode = if size == 8 { 0x86u8 } else { 0x87u8 };
            emit_rex_for_reg_mem(buf, *reg, mem)?;
            buf.push(opcode);
            emit_mem_modrm(buf, reg.base_code(), mem);
        }
        // xchg [mem], reg
        (Operand::Memory(mem), Operand::Register(reg)) => {
            let size = reg_size(*reg);
            let opcode = if size == 8 { 0x86u8 } else { 0x87u8 };
            emit_rex_for_reg_mem(buf, *reg, mem)?;
            buf.push(opcode);
            emit_mem_modrm(buf, reg.base_code(), mem);
        }
        _ => {
            return Err(invalid_operands(
                "xchg",
                "unsupported operand combination",
                instr.span,
            ))
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_bt(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    digit: u8,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 2 operands",
            instr.span,
        ));
    }
    // BT family only supports 16/32/64-bit operands
    if let Operand::Register(r) = &ops[0] {
        if reg_size(*r) == 8 {
            return Err(invalid_operands(
                &instr.mnemonic,
                "8-bit operands not supported",
                instr.span,
            ));
        }
    }
    match (&ops[0], &ops[1]) {
        (Operand::Register(dst), Operand::Immediate(imm)) => {
            let size = reg_size(*dst);
            let w = size == 64;
            let b = dst.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, false, false, b) {
                buf.push(rex(w, false, false, b));
            }
            buf.push(0x0F);
            buf.push(0xBA);
            buf.push(modrm(0b11, digit, dst.base_code()));
            buf.push(*imm as u8);
        }
        (Operand::Register(dst), Operand::Register(src)) => {
            let size = reg_size(*dst);
            let w = size == 64;
            let r = src.is_extended();
            let b = dst.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, r, false, b) {
                buf.push(rex(w, r, false, b));
            }
            buf.push(0x0F);
            let base = match digit {
                4 => 0xA3, // BT
                5 => 0xAB, // BTS
                6 => 0xB3, // BTR
                7 => 0xBB, // BTC
                _ => 0xA3,
            };
            buf.push(base);
            buf.push(modrm(0b11, src.base_code(), dst.base_code()));
        }
        (Operand::Memory(mem), Operand::Register(src)) => {
            let size = mem.size.map_or(reg_size(*src), |s| s.bits() as u8);
            let w = size == 64;
            let r = src.is_extended();
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, r, x, b) {
                buf.push(rex(w, r, x, b));
            }
            buf.push(0x0F);
            let base = match digit {
                4 => 0xA3,
                5 => 0xAB,
                6 => 0xB3,
                7 => 0xBB,
                _ => 0xA3,
            };
            buf.push(base);
            emit_mem_modrm(buf, src.base_code(), mem);
        }
        (Operand::Memory(mem), Operand::Immediate(imm)) => {
            let size = instr
                .size_hint
                .map_or(mem.size.map_or(32u8, |s| s.bits() as u8), |s| {
                    s.bits() as u8
                });
            let w = size == 64;
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, false, x, b) {
                buf.push(rex(w, false, x, b));
            }
            buf.push(0x0F);
            buf.push(0xBA);
            emit_mem_modrm(buf, digit, mem);
            buf.push(*imm as u8);
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "unsupported operand combination",
                instr.span,
            ))
        }
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_bsf_bsr(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    opcode2: u8,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(
            &instr.mnemonic,
            "expected 2 operands",
            instr.span,
        ));
    }
    // BSF/BSR only support 16/32/64-bit operands
    if let Operand::Register(r) = &ops[0] {
        if reg_size(*r) == 8 {
            return Err(invalid_operands(
                &instr.mnemonic,
                "8-bit operands not supported",
                instr.span,
            ));
        }
    }
    match (&ops[0], &ops[1]) {
        (Operand::Register(dst), Operand::Register(src)) => {
            let size = reg_size(*dst);
            let w = size == 64;
            let r = dst.is_extended();
            let b = src.is_extended();
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, r, false, b) {
                buf.push(rex(w, r, false, b));
            }
            buf.push(0x0F);
            buf.push(opcode2);
            buf.push(modrm(0b11, dst.base_code(), src.base_code()));
        }
        (Operand::Register(dst), Operand::Memory(mem)) => {
            emit_rex_for_reg_mem(buf, *dst, mem)?;
            buf.push(0x0F);
            buf.push(opcode2);
            emit_mem_modrm(buf, dst.base_code(), mem);
        }
        _ => {
            return Err(invalid_operands(
                &instr.mnemonic,
                "expected reg, r/m",
                instr.span,
            ))
        }
    }
    Ok(())
}

/// Unified encoder for F3 0F xx reg,r/m class instructions (popcnt, lzcnt, tzcnt).
#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn encode_f3_0f_rm(
    buf: &mut InstrBytes,
    ops: &[Operand],
    mnemonic: &str,
    opcode: u8,
    span: Span,
) -> Result<(), AsmError> {
    if ops.len() != 2 {
        return Err(invalid_operands(mnemonic, "expected 2 operands", span));
    }
    if let Operand::Register(r) = &ops[0] {
        if reg_size(*r) == 8 {
            return Err(invalid_operands(
                mnemonic,
                "8-bit operands not supported",
                span,
            ));
        }
    }
    match (&ops[0], &ops[1]) {
        (Operand::Register(dst), Operand::Register(src)) => {
            let size = reg_size(*dst);
            let w = size == 64;
            let r = dst.is_extended();
            let b = src.is_extended();
            buf.push(0xF3);
            if size == 16 {
                buf.push(0x66);
            }
            if needs_rex(w, r, false, b) {
                buf.push(rex(w, r, false, b));
            }
            buf.push(0x0F);
            buf.push(opcode);
            buf.push(modrm(0b11, dst.base_code(), src.base_code()));
        }
        (Operand::Register(dst), Operand::Memory(mem)) => {
            buf.push(0xF3);
            emit_rex_for_reg_mem(buf, *dst, mem)?;
            buf.push(0x0F);
            buf.push(opcode);
            emit_mem_modrm(buf, dst.base_code(), mem);
        }
        _ => return Err(invalid_operands(mnemonic, "expected reg, r/m", span)),
    }
    Ok(())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_popcnt(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    encode_f3_0f_rm(buf, ops, "popcnt", 0xB8, instr.span)
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_lzcnt(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    encode_f3_0f_rm(buf, ops, "lzcnt", 0xBD, instr.span)
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_tzcnt(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    encode_f3_0f_rm(buf, ops, "tzcnt", 0xBC, instr.span)
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_bswap(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    if ops.len() != 1 {
        return Err(invalid_operands("bswap", "expected 1 operand", instr.span));
    }
    match &ops[0] {
        Operand::Register(reg) => {
            let size = reg_size(*reg);
            if size == 8 {
                return Err(invalid_operands(
                    "bswap",
                    "8-bit operands not supported",
                    instr.span,
                ));
            }
            if size == 16 {
                return Err(invalid_operands(
                    "bswap",
                    "16-bit bswap has undefined behavior; use xchg or rol instead",
                    instr.span,
                ));
            }
            let w = size == 64;
            let b = reg.is_extended();
            if needs_rex(w, false, false, b) {
                buf.push(rex(w, false, false, b));
            }
            buf.push(0x0F);
            buf.push(0xC8 + reg.base_code());
        }
        _ => return Err(invalid_operands("bswap", "expected register", instr.span)),
    }
    Ok(())
}

/// Emit an immediate value of the given size.
///
/// # Panics
///
/// Panics if `size` is not one of 8, 16, 32, or 64.
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn emit_imm(buf: &mut InstrBytes, imm: i128, size: u8) {
    match size {
        8 => buf.push(imm as u8),
        16 => buf.extend_from_slice(&(imm as u16).to_le_bytes()),
        32 => buf.extend_from_slice(&(imm as u32).to_le_bytes()),
        64 => buf.extend_from_slice(&(imm as u64).to_le_bytes()),
        other => panic!("emit_imm: unsupported immediate size {other} (expected 8, 16, 32, or 64)"),
    }
}

#[inline]
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn invalid_operands(_mnemonic: &str, detail: &str, span: Span) -> AsmError {
    AsmError::InvalidOperands {
        detail: String::from(detail),
        span,
    }
}

// ─── SSE / XMM encoder helpers ────────────────────────────────

/// Emit REX prefix for an XMM-register,XMM-register or XMM-register,GPR form.
/// `w` controls REX.W (needed for movd/movq 64-bit forms).
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn emit_rex_sse_rr(buf: &mut InstrBytes, w: bool, reg: Register, rm: Register) {
    let r = reg.is_extended();
    let b = rm.is_extended();
    if needs_rex(w, r, false, b) {
        buf.push(rex(w, r, false, b));
    }
}

/// Emit REX prefix for XMM register + memory operand.
/// `w` controls REX.W.
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn emit_rex_sse_rm(buf: &mut InstrBytes, w: bool, reg: Register, mem: &MemoryOperand) {
    let r = reg.is_extended();
    let x = mem.index.is_some_and(|r| r.is_extended());
    let b = mem.base.is_some_and(|r| r.is_extended());
    if needs_rex(w, r, x, b) {
        buf.push(rex(w, r, x, b));
    }
}

/// Encode an SSE instruction: xmm, xmm/m  (or xmm/m, xmm for stores).
///
/// Pattern: `[mandatory_prefix] [REX] opcode_bytes ModR/M`
///
/// `opcode` is the full opcode slice (e.g. `&[0x0F, 0x58]` for ADDPS).
/// `mandatory_prefix` is 0 (none), 0x66, 0xF3, or 0xF2.
/// `rex_w` forces REX.W (needed for 64-bit movd/movq).
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_sse_rr(
    buf: &mut InstrBytes,
    mandatory_prefix: u8,
    opcode: &[u8],
    dst: Register,
    src: Register,
    rex_w: bool,
) {
    if mandatory_prefix != 0 {
        buf.push(mandatory_prefix);
    }
    emit_rex_sse_rr(buf, rex_w, dst, src);
    buf.extend_from_slice(opcode);
    buf.push(modrm(0b11, dst.base_code(), src.base_code()));
}

/// Encode SSE xmm, mem (load direction): [prefix] REX opcode ModR/M [SIB] [disp]
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_sse_rm(
    buf: &mut InstrBytes,
    mandatory_prefix: u8,
    opcode: &[u8],
    reg: Register,
    mem: &MemoryOperand,
    reloc: &mut Option<Relocation>,
    rex_w: bool,
) {
    if mandatory_prefix != 0 {
        buf.push(mandatory_prefix);
    }
    emit_rex_sse_rm(buf, rex_w, reg, mem);
    buf.extend_from_slice(opcode);
    let disp_off = emit_mem_modrm(buf, reg.base_code(), mem);
    set_mem_reloc(reloc, mem, disp_off, buf.len());
}

/// Encode SSE mem, xmm (store direction): [prefix] REX opcode ModR/M [SIB] [disp]
/// Same as encode_sse_rm but with swapped semantics (reg is source, mem is dest).
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_sse_mr(
    buf: &mut InstrBytes,
    mandatory_prefix: u8,
    opcode: &[u8],
    mem: &MemoryOperand,
    reg: Register,
    reloc: &mut Option<Relocation>,
    rex_w: bool,
) {
    // Encoding is identical — reg field goes in ModR/M.reg, mem in ModR/M.rm
    encode_sse_rm(buf, mandatory_prefix, opcode, reg, mem, reloc, rex_w);
}

/// Generic SSE two-operand encoder: xmm, xmm/m or xmm/m, xmm.
///
/// `reverse` = true means the first operand is memory (store direction, uses `store_opcode`).
/// `load_opcode` / `store_opcode` are the full opcode byte sequences.
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_sse_op(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    mandatory_prefix: u8,
    load_opcode: &[u8],
    store_opcode: Option<&[u8]>,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    match (ops.first(), ops.get(1)) {
        // xmm, xmm
        (Some(Operand::Register(dst)), Some(Operand::Register(src)))
            if dst.is_xmm() && src.is_xmm() =>
        {
            encode_sse_rr(buf, mandatory_prefix, load_opcode, *dst, *src, false);
            Ok(())
        }
        // xmm, mem
        (Some(Operand::Register(dst)), Some(Operand::Memory(mem))) if dst.is_xmm() => {
            encode_sse_rm(buf, mandatory_prefix, load_opcode, *dst, mem, reloc, false);
            Ok(())
        }
        // mem, xmm (store)
        (Some(Operand::Memory(mem)), Some(Operand::Register(src))) if src.is_xmm() => {
            let opcode = store_opcode.unwrap_or(load_opcode);
            encode_sse_mr(buf, mandatory_prefix, opcode, mem, *src, reloc, false);
            Ok(())
        }
        _ => Err(invalid_operands(
            instr.mnemonic.as_str(),
            "expected xmm,xmm/m or m,xmm operands",
            instr.span,
        )),
    }
}

/// SSE instruction with an immediate byte: xmm, xmm/m, imm8.
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_sse_imm(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    mandatory_prefix: u8,
    opcode: &[u8],
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    let imm = match ops.get(2) {
        Some(Operand::Immediate(v)) => *v,
        _ => {
            return Err(invalid_operands(
                instr.mnemonic.as_str(),
                "expected xmm, xmm/m, imm8",
                instr.span,
            ));
        }
    };
    match (ops.first(), ops.get(1)) {
        (Some(Operand::Register(dst)), Some(Operand::Register(src)))
            if dst.is_xmm() && src.is_xmm() =>
        {
            encode_sse_rr(buf, mandatory_prefix, opcode, *dst, *src, false);
        }
        (Some(Operand::Register(dst)), Some(Operand::Memory(mem))) if dst.is_xmm() => {
            encode_sse_rm(buf, mandatory_prefix, opcode, *dst, mem, reloc, false);
        }
        _ => {
            return Err(invalid_operands(
                instr.mnemonic.as_str(),
                "expected xmm, xmm/m, imm8",
                instr.span,
            ));
        }
    }
    buf.push(imm as u8);
    Ok(())
}

/// Encode MOVD/MOVQ — GP ↔ XMM transfers.
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_movd_movq(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    reloc: &mut Option<Relocation>,
    is_movq: bool,
) -> Result<(), AsmError> {
    match (ops.first(), ops.get(1)) {
        // movd/movq xmm, r/m32/64 — load: 66 [REX.W] 0F 6E /r
        (Some(Operand::Register(dst)), Some(Operand::Register(src)))
            if dst.is_xmm() && !src.is_xmm() =>
        {
            let w = is_movq || src.size_bits() == 64;
            encode_sse_rr(buf, 0x66, &[0x0F, 0x6E], *dst, *src, w);
            Ok(())
        }
        (Some(Operand::Register(dst)), Some(Operand::Memory(mem))) if dst.is_xmm() => {
            let w = is_movq;
            encode_sse_rm(buf, 0x66, &[0x0F, 0x6E], *dst, mem, reloc, w);
            Ok(())
        }
        // movd/movq r/m32/64, xmm — store: 66 [REX.W] 0F 7E /r
        (Some(Operand::Register(dst)), Some(Operand::Register(src)))
            if !dst.is_xmm() && src.is_xmm() =>
        {
            let w = is_movq || dst.size_bits() == 64;
            // Note: in the store form, the XMM reg is the source but goes in ModR/M.reg
            encode_sse_rr(buf, 0x66, &[0x0F, 0x7E], *src, *dst, w);
            Ok(())
        }
        (Some(Operand::Memory(mem)), Some(Operand::Register(src))) if src.is_xmm() => {
            let w = is_movq;
            encode_sse_mr(buf, 0x66, &[0x0F, 0x7E], mem, *src, reloc, w);
            Ok(())
        }
        // movq xmm, xmm — use F3 0F 7E /r
        (Some(Operand::Register(dst)), Some(Operand::Register(src)))
            if dst.is_xmm() && src.is_xmm() =>
        {
            encode_sse_rr(buf, 0xF3, &[0x0F, 0x7E], *dst, *src, false);
            Ok(())
        }
        _ => Err(invalid_operands(
            instr.mnemonic.as_str(),
            "expected xmm,r/m or r/m,xmm operands",
            instr.span,
        )),
    }
}

/// Encode CVTSI2SS / CVTSI2SD: xmm, r/m32/64.
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_cvtsi2(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    mandatory_prefix: u8,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    match (ops.first(), ops.get(1)) {
        (Some(Operand::Register(dst)), Some(Operand::Register(src)))
            if dst.is_xmm() && !src.is_xmm() =>
        {
            let w = src.size_bits() == 64;
            encode_sse_rr(buf, mandatory_prefix, &[0x0F, 0x2A], *dst, *src, w);
            Ok(())
        }
        (Some(Operand::Register(dst)), Some(Operand::Memory(mem))) if dst.is_xmm() => {
            let w = mem.size == Some(OperandSize::Qword);
            encode_sse_rm(buf, mandatory_prefix, &[0x0F, 0x2A], *dst, mem, reloc, w);
            Ok(())
        }
        _ => Err(invalid_operands(
            instr.mnemonic.as_str(),
            "expected xmm, r/m32 or xmm, r/m64",
            instr.span,
        )),
    }
}

/// Encode CVTSS2SI / CVTSD2SI / CVTTSS2SI / CVTTSD2SI: r32/64, xmm/m.
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_cvt2si(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    mandatory_prefix: u8,
    opcode2: u8,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    match (ops.first(), ops.get(1)) {
        (Some(Operand::Register(dst)), Some(Operand::Register(src)))
            if !dst.is_xmm() && src.is_xmm() =>
        {
            let w = dst.size_bits() == 64;
            encode_sse_rr(buf, mandatory_prefix, &[0x0F, opcode2], *dst, *src, w);
            Ok(())
        }
        (Some(Operand::Register(dst)), Some(Operand::Memory(mem))) if !dst.is_xmm() => {
            let w = dst.size_bits() == 64;
            encode_sse_rm(buf, mandatory_prefix, &[0x0F, opcode2], *dst, mem, reloc, w);
            Ok(())
        }
        _ => Err(invalid_operands(
            instr.mnemonic.as_str(),
            "expected r32/r64, xmm/m operands",
            instr.span,
        )),
    }
}

/// Encode prefetch family: 0F 18 /digit (memory-only).
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_prefetch(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    digit: u8,
) -> Result<(), AsmError> {
    match ops.first() {
        Some(Operand::Memory(mem)) => {
            // No REX.W, no prefix for prefetch
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if needs_rex(false, false, x, b) {
                buf.push(rex(false, false, x, b));
            }
            buf.extend_from_slice(&[0x0F, 0x18]);
            emit_mem_modrm(buf, digit, mem);
            Ok(())
        }
        _ => Err(invalid_operands(
            instr.mnemonic.as_str(),
            "expected memory operand",
            instr.span,
        )),
    }
}

/// Encode CLFLUSH: 0F AE /7 (memory-only).
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_clflush(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    match ops.first() {
        Some(Operand::Memory(mem)) => {
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if needs_rex(false, false, x, b) {
                buf.push(rex(false, false, x, b));
            }
            buf.extend_from_slice(&[0x0F, 0xAE]);
            emit_mem_modrm(buf, 7, mem);
            Ok(())
        }
        _ => Err(invalid_operands(
            "clflush",
            "expected memory operand",
            instr.span,
        )),
    }
}

/// Encode CLFLUSHOPT: 66 0F AE /7 (memory-only).
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_clflushopt(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    match ops.first() {
        Some(Operand::Memory(mem)) => {
            buf.push(0x66);
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if needs_rex(false, false, x, b) {
                buf.push(rex(false, false, x, b));
            }
            buf.extend_from_slice(&[0x0F, 0xAE]);
            emit_mem_modrm(buf, 7, mem);
            Ok(())
        }
        _ => Err(invalid_operands(
            "clflushopt",
            "expected memory operand",
            instr.span,
        )),
    }
}

/// Encode CLWB: 66 0F AE /6 (memory-only).
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_clwb(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    match ops.first() {
        Some(Operand::Memory(mem)) => {
            buf.push(0x66);
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if needs_rex(false, false, x, b) {
                buf.push(rex(false, false, x, b));
            }
            buf.extend_from_slice(&[0x0F, 0xAE]);
            emit_mem_modrm(buf, 6, mem);
            Ok(())
        }
        _ => Err(invalid_operands(
            "clwb",
            "expected memory operand",
            instr.span,
        )),
    }
}

/// Encode PREFETCHW: 0F 0D /1 (memory-only).
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_prefetchw(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    match ops.first() {
        Some(Operand::Memory(mem)) => {
            let x = mem.index.is_some_and(|r| r.is_extended());
            let b = mem.base.is_some_and(|r| r.is_extended());
            if needs_rex(false, false, x, b) {
                buf.push(rex(false, false, x, b));
            }
            buf.extend_from_slice(&[0x0F, 0x0D]);
            emit_mem_modrm(buf, 1, mem);
            Ok(())
        }
        _ => Err(invalid_operands(
            "prefetchw",
            "expected memory operand",
            instr.span,
        )),
    }
}

/// Encode CRC32: F2 [REX.W] 0F 38 F0/F1 /r.
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_crc32(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
) -> Result<(), AsmError> {
    match (ops.first(), ops.get(1)) {
        (Some(Operand::Register(dst)), Some(Operand::Register(src))) => {
            let dst_s = dst.size_bits();
            let src_s = src.size_bits();
            if dst_s != 32 && dst_s != 64 {
                return Err(invalid_operands(
                    "crc32",
                    "destination must be r32 or r64",
                    instr.span,
                ));
            }
            buf.push(0xF2);
            let w = dst_s == 64;
            let opcode2 = if src_s == 8 { 0xF0u8 } else { 0xF1 };
            if src_s == 16 {
                buf.push(0x66);
            }
            emit_rex_sse_rr(buf, w, *dst, *src);
            buf.extend_from_slice(&[0x0F, 0x38, opcode2]);
            buf.push(modrm(0b11, dst.base_code(), src.base_code()));
            Ok(())
        }
        (Some(Operand::Register(dst)), Some(Operand::Memory(mem))) => {
            let dst_s = dst.size_bits();
            if dst_s != 32 && dst_s != 64 {
                return Err(invalid_operands(
                    "crc32",
                    "destination must be r32 or r64",
                    instr.span,
                ));
            }
            let src_s = instr.size_hint.map_or_else(
                || mem.size.map_or(32u8, |s| s.bits() as u8),
                |s| s.bits() as u8,
            );
            buf.push(0xF2);
            let w = dst_s == 64;
            let opcode2 = if src_s == 8 { 0xF0u8 } else { 0xF1 };
            if src_s == 16 {
                buf.push(0x66);
            }
            emit_rex_sse_rm(buf, w, *dst, mem);
            buf.extend_from_slice(&[0x0F, 0x38, opcode2]);
            emit_mem_modrm(buf, dst.base_code(), mem);
            Ok(())
        }
        _ => Err(invalid_operands(
            "crc32",
            "expected r32/r64, r/m operands",
            instr.span,
        )),
    }
}

// ─── VEX / EVEX prefix encoding infrastructure ──────────────────────────────
//
// These functions implement VEX and EVEX prefix encoding for SSE/AVX instructions.
// They are not yet wired into the text-assembly path but are fully tested and
// ready for use once the mnemonic dispatch tables include VEX-encoded mnemonics.

/// VEX prefix "pp" field (implied mandatory prefix).
///   0 = none, 1 = 0x66, 2 = 0xF3, 3 = 0xF2
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn vex_pp(mandatory_prefix: u8) -> u8 {
    match mandatory_prefix {
        0x00 => 0b00,
        0x66 => 0b01,
        0xF3 => 0b10,
        0xF2 => 0b11,
        _ => 0b00,
    }
}

/// VEX "m-mmmm" field (implied escape bytes).
///   1 = 0F, 2 = 0F 38, 3 = 0F 3A
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn vex_mmmmm(escape: &[u8]) -> u8 {
    match escape {
        [0x0F] => 0b00001,
        [0x0F, 0x38] => 0b00010,
        [0x0F, 0x3A] => 0b00011,
        _ => 0b00001,
    }
}

/// Emit a 2-byte VEX prefix: C5 [R vvvv L pp]
/// - R: inverted REX.R (1 = no extension)
/// - vvvv: inverted source register (NDS), 0b1111 = unused
/// - L: vector length (0 = 128-bit, 1 = 256-bit)
/// - pp: implied prefix
///
/// 2-byte VEX can only be used when:
///   - m-mmmm == 0b00001 (0F escape)
///   - W == 0
///   - X == 1 (no REX.X) and B == 1 (no REX.B)
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn emit_vex2(buf: &mut InstrBytes, r: bool, vvvv: u8, l: bool, pp: u8) {
    let byte1 = (if r { 0 } else { 0x80 })
        | (((!vvvv) & 0x0F) << 3)
        | (if l { 0x04 } else { 0 })
        | (pp & 0x03);
    buf.push(0xC5);
    buf.push(byte1);
}

/// Emit a 3-byte VEX prefix: C4 [R X B mmmmm] [W vvvv L pp]
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn emit_vex3(
    buf: &mut InstrBytes,
    r: bool,
    x: bool,
    b: bool,
    mmmmm: u8,
    w: bool,
    vvvv: u8,
    l: bool,
    pp: u8,
) {
    let byte1 = (if r { 0 } else { 0x80 })
        | (if x { 0 } else { 0x40 })
        | (if b { 0 } else { 0x20 })
        | (mmmmm & 0x1F);
    let byte2 = (if w { 0x80 } else { 0 })
        | (((!vvvv) & 0x0F) << 3)
        | (if l { 0x04 } else { 0 })
        | (pp & 0x03);
    buf.push(0xC4);
    buf.push(byte1);
    buf.push(byte2);
}

/// Choose and emit the most compact VEX prefix (2-byte if possible, else 3-byte).
/// For VEX-encoded instructions.
///
/// Parameters:
/// - `reg`: the ModR/M reg field register (or the first source for some forms)
/// - `vvvv_reg`: the VEX.vvvv register (NDS/NDD source), or 0 if unused
/// - `rm_extended`: whether the R/M or base register is extended (R8-R15, etc.)
/// - `x_extended`: whether the SIB index register is extended
/// - `w`: REX.W equivalent
/// - `l`: vector length (false = 128, true = 256)
/// - `pp`: implied mandatory prefix
/// - `escape`: the escape byte sequence (e.g., &[0x0F] or &[0x0F, 0x38])
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn emit_vex_prefix(
    buf: &mut InstrBytes,
    reg_extended: bool,
    x_extended: bool,
    rm_extended: bool,
    w: bool,
    vvvv: u8,
    l: bool,
    pp: u8,
    escape: &[u8],
) {
    let mmmmm = vex_mmmmm(escape);
    // 2-byte VEX can be used when: mmmmm == 0b00001, W == 0, X == 1 (not extended), B == 1 (not extended)
    if mmmmm == 0b00001 && !w && !x_extended && !rm_extended {
        emit_vex2(buf, reg_extended, vvvv, l, pp);
    } else {
        emit_vex3(
            buf,
            reg_extended,
            x_extended,
            rm_extended,
            mmmmm,
            w,
            vvvv,
            l,
            pp,
        );
    }
}

/// Encode a VEX-prefix instruction: reg, reg (3-operand non-destructive, e.g., vaddps xmm0, xmm1, xmm2)
/// `dst` is the ModR/M reg field, `src1` is VEX.vvvv (NDS), `src2` is ModR/M r/m field
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_vex_rrr(
    buf: &mut InstrBytes,
    pp: u8,
    escape: &[u8],
    opcode: u8,
    dst: Register,
    src1: Register,
    src2: Register,
    w: bool,
    l: bool,
) {
    let mandatory_pp = vex_pp(pp);
    emit_vex_prefix(
        buf,
        dst.is_extended(),
        false,
        src2.is_extended(),
        w,
        src1.base_code() | if src1.is_extended() { 8 } else { 0 },
        l,
        mandatory_pp,
        escape,
    );
    buf.push(opcode);
    buf.push(0xC0 | (dst.base_code() << 3) | src2.base_code());
}

/// Encode VEX reg, [mem] (3-operand: dst = reg, src1 = vvvv, src2 = mem)
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_vex_rrm(
    buf: &mut InstrBytes,
    pp: u8,
    escape: &[u8],
    opcode: u8,
    dst: Register,
    src1: Register,
    mem: &MemoryOperand,
    reloc: &mut Option<Relocation>,
    w: bool,
    l: bool,
) {
    let mandatory_pp = vex_pp(pp);
    let x_ext = mem.index.is_some_and(|r| r.is_extended());
    let b_ext = mem.base.is_some_and(|r| r.is_extended());
    emit_vex_prefix(
        buf,
        dst.is_extended(),
        x_ext,
        b_ext,
        w,
        src1.base_code() | if src1.is_extended() { 8 } else { 0 },
        l,
        mandatory_pp,
        escape,
    );
    buf.push(opcode);
    emit_mem_modrm(buf, dst.base_code(), mem);
    if let Some(ref mut rel) = reloc {
        rel.offset = buf.len() - 4;
    }
}

/// Generic VEX-encoded SSE/AVX instruction dispatcher.
/// Handles the common patterns:
///   - vop xmm/ymm, xmm/ymm, xmm/ymm  (3 register operands, NDS form)
///   - vop xmm/ymm, xmm/ymm, [mem]     (reg, reg, mem)
///   - vop xmm/ymm, xmm/ymm            (2 operands => dst=src1=first, src2=second for moves)
///   - vop [mem], xmm/ymm               (store, if store_opcode given)
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_vex_op(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pp: u8,
    escape: &[u8],
    load_opcode: u8,
    store_opcode: Option<u8>,
    w: bool,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    use Operand::*;
    let o = (ops.first(), ops.get(1), ops.get(2));
    match o {
        // 3-operand: reg, reg, reg
        (Some(Register(dst)), Some(Register(src1)), Some(Register(src2)))
            if dst.is_vector() && src1.is_vector() && src2.is_vector() =>
        {
            let l = dst.is_ymm() || src1.is_ymm();
            encode_vex_rrr(buf, pp, escape, load_opcode, *dst, *src1, *src2, w, l);
            Ok(())
        }
        // 3-operand: reg, reg, mem
        (Some(Register(dst)), Some(Register(src1)), Some(Memory(mem)))
            if dst.is_vector() && src1.is_vector() =>
        {
            let l = dst.is_ymm() || src1.is_ymm();
            encode_vex_rrm(buf, pp, escape, load_opcode, *dst, *src1, mem, reloc, w, l);
            Ok(())
        }
        // 2-operand reg, reg (move-like: vvvv unused)
        (Some(Register(dst)), Some(Register(src)), None) if dst.is_vector() && src.is_vector() => {
            let l = dst.is_ymm() || src.is_ymm();
            let mandatory_pp = vex_pp(pp);
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                false,
                src.is_extended(),
                w,
                0,
                l,
                mandatory_pp,
                escape,
            );
            buf.push(load_opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src.base_code());
            Ok(())
        }
        // 2-operand reg, mem (load)
        (Some(Register(dst)), Some(Memory(mem)), None) if dst.is_vector() => {
            let l = dst.is_ymm();
            let mandatory_pp = vex_pp(pp);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                x_ext,
                b_ext,
                w,
                0,
                l,
                mandatory_pp,
                escape,
            );
            buf.push(load_opcode);
            emit_mem_modrm(buf, dst.base_code(), mem);
            Ok(())
        }
        // 2-operand mem, reg (store)
        (Some(Memory(mem)), Some(Register(src)), None)
            if src.is_vector() && store_opcode.is_some() =>
        {
            let l = src.is_ymm();
            let mandatory_pp = vex_pp(pp);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_vex_prefix(
                buf,
                src.is_extended(),
                x_ext,
                b_ext,
                w,
                0,
                l,
                mandatory_pp,
                escape,
            );
            // SAFETY: match guard `store_opcode.is_some()` guarantees Some
            buf.push(store_opcode.unwrap_or(0));
            emit_mem_modrm(buf, src.base_code(), mem);
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected VEX xmm/ymm operands",
            instr.span,
        )),
    }
}

/// VEX instruction with an immediate byte (4 operands: dst, src1, src2/mem, imm8)
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_vex_imm(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pp: u8,
    escape: &[u8],
    opcode: u8,
    w: bool,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    use Operand::*;
    let o = (ops.first(), ops.get(1), ops.get(2), ops.get(3));
    match o {
        // 4-operand: reg, reg, reg, imm8
        (Some(Register(dst)), Some(Register(src1)), Some(Register(src2)), Some(Immediate(imm)))
            if dst.is_vector() && src1.is_vector() && src2.is_vector() =>
        {
            let l = dst.is_ymm() || src1.is_ymm();
            encode_vex_rrr(buf, pp, escape, opcode, *dst, *src1, *src2, w, l);
            buf.push(*imm as u8);
            Ok(())
        }
        // 4-operand: reg, reg, mem, imm8
        (Some(Register(dst)), Some(Register(src1)), Some(Memory(mem)), Some(Immediate(imm)))
            if dst.is_vector() && src1.is_vector() =>
        {
            let l = dst.is_ymm() || src1.is_ymm();
            encode_vex_rrm(buf, pp, escape, opcode, *dst, *src1, mem, reloc, w, l);
            buf.push(*imm as u8);
            Ok(())
        }
        // 3-operand: reg, reg/mem, imm8 (vvvv unused, like vpshufd)
        (Some(Register(dst)), Some(Register(src)), Some(Immediate(imm)), None)
            if dst.is_vector() && src.is_vector() =>
        {
            let l = dst.is_ymm() || src.is_ymm();
            let mandatory_pp = vex_pp(pp);
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                false,
                src.is_extended(),
                w,
                0,
                l,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src.base_code());
            buf.push(*imm as u8);
            Ok(())
        }
        (Some(Register(dst)), Some(Memory(mem)), Some(Immediate(imm)), None) if dst.is_vector() => {
            let l = dst.is_ymm();
            let mandatory_pp = vex_pp(pp);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                x_ext,
                b_ext,
                w,
                0,
                l,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            emit_mem_modrm(buf, dst.base_code(), mem);
            buf.push(*imm as u8);
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected VEX xmm/ymm operands with imm8",
            instr.span,
        )),
    }
}

/// VEX-encoded BMI instruction: reg, reg/mem (2-operand, VEX.vvvv = dst)
/// e.g., ANDN r32, r32, r/m32; BLSI r32, r/m32; TZCNT-like
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_vex_bmi_vex_ndd(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pp: u8,
    escape: &[u8],
    opcode: u8,
    w_from_size: bool,
) -> Result<(), AsmError> {
    use Operand::*;
    match (ops.first(), ops.get(1), ops.get(2)) {
        // 3-operand: r, r, r/m (e.g., andn eax, ebx, ecx)
        (Some(Register(dst)), Some(Register(src1)), Some(Register(src2))) => {
            let w = if w_from_size {
                dst.size_bits() == 64
            } else {
                false
            };
            let mandatory_pp = vex_pp(pp);
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                false,
                src2.is_extended(),
                w,
                src1.base_code() | if src1.is_extended() { 8 } else { 0 },
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src2.base_code());
            Ok(())
        }
        (Some(Register(dst)), Some(Register(src1)), Some(Memory(mem))) => {
            let w = if w_from_size {
                dst.size_bits() == 64
            } else {
                false
            };
            let mandatory_pp = vex_pp(pp);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                x_ext,
                b_ext,
                w,
                src1.base_code() | if src1.is_extended() { 8 } else { 0 },
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            emit_mem_modrm(buf, dst.base_code(), mem);
            Ok(())
        }
        // 2-operand forms (e.g., blsi r32, r/m32 — dst in VEX.vvvv, src in ModR/M r/m)
        (Some(Register(dst)), Some(Register(src)), None) => {
            let w = if w_from_size {
                dst.size_bits() == 64
            } else {
                false
            };
            let mandatory_pp = vex_pp(pp);
            emit_vex_prefix(
                buf,
                false, // reg field is the /digit, not a real register
                false,
                src.is_extended(),
                w,
                dst.base_code() | if dst.is_extended() { 8 } else { 0 },
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            // The reg field contains the /digit (set by caller via opcode encoding)
            // Actually for BLSI/BLSR/BLSMSK the reg field is fixed (/3, /1, /2)
            // This function sets reg=0; caller must set via a wrapper
            buf.push(0xC0 | src.base_code());
            Ok(())
        }
        (Some(Register(dst)), Some(Memory(mem)), None) => {
            let w = if w_from_size {
                dst.size_bits() == 64
            } else {
                false
            };
            let mandatory_pp = vex_pp(pp);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_vex_prefix(
                buf,
                false,
                x_ext,
                b_ext,
                w,
                dst.base_code() | if dst.is_extended() { 8 } else { 0 },
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            emit_mem_modrm(buf, 0, mem);
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected GP register operands",
            instr.span,
        )),
    }
}

/// VEX BMI with /digit in the reg field + VEX.vvvv = dst  (BLSI, BLSR, BLSMSK)
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_vex_bmi_digit(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pp: u8,
    escape: &[u8],
    opcode: u8,
    digit: u8,
    w_from_size: bool,
) -> Result<(), AsmError> {
    use Operand::*;
    match (ops.first(), ops.get(1)) {
        (Some(Register(dst)), Some(Register(src))) => {
            let w = if w_from_size {
                dst.size_bits() == 64
            } else {
                false
            };
            let mandatory_pp = vex_pp(pp);
            emit_vex_prefix(
                buf,
                false, // reg field is /digit, never extended
                false,
                src.is_extended(),
                w,
                dst.base_code() | if dst.is_extended() { 8 } else { 0 },
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            buf.push(0xC0 | (digit << 3) | src.base_code());
            Ok(())
        }
        (Some(Register(dst)), Some(Memory(mem))) => {
            let w = if w_from_size {
                dst.size_bits() == 64
            } else {
                false
            };
            let mandatory_pp = vex_pp(pp);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_vex_prefix(
                buf,
                false,
                x_ext,
                b_ext,
                w,
                dst.base_code() | if dst.is_extended() { 8 } else { 0 },
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            emit_mem_modrm(buf, digit, mem);
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected r32/r64, r/m32/r/m64",
            instr.span,
        )),
    }
}

/// VEX BMI2 with immediate (RORX: VEX.LZ.F2.0F3A.W0/W1 F0 /r ib)
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_vex_bmi_imm(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pp: u8,
    escape: &[u8],
    opcode: u8,
    w_from_size: bool,
) -> Result<(), AsmError> {
    use Operand::*;
    match (ops.first(), ops.get(1), ops.get(2)) {
        (Some(Register(dst)), Some(Register(src)), Some(Immediate(imm))) => {
            let w = if w_from_size {
                dst.size_bits() == 64
            } else {
                false
            };
            let mandatory_pp = vex_pp(pp);
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                false,
                src.is_extended(),
                w,
                0,
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src.base_code());
            buf.push(*imm as u8);
            Ok(())
        }
        (Some(Register(dst)), Some(Memory(mem)), Some(Immediate(imm))) => {
            let w = if w_from_size {
                dst.size_bits() == 64
            } else {
                false
            };
            let mandatory_pp = vex_pp(pp);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                x_ext,
                b_ext,
                w,
                0,
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            emit_mem_modrm(buf, dst.base_code(), mem);
            buf.push(*imm as u8);
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected r, r/m, imm8",
            instr.span,
        )),
    }
}

/// VEX BMI instruction with reversed operand mapping: dst(reg), src(r/m), control(vvvv)
/// Used for BEXTR, BZHI, SARX, SHLX, SHRX where the second operand is
/// the ModR/M r/m field and the third operand is VEX.vvvv.
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_vex_bmi_rmv(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pp: u8,
    escape: &[u8],
    opcode: u8,
    w_from_size: bool,
) -> Result<(), AsmError> {
    use Operand::*;
    match (ops.first(), ops.get(1), ops.get(2)) {
        // 3-operand: r, r, r (dst=reg, src=r/m, control=vvvv)
        (Some(Register(dst)), Some(Register(src)), Some(Register(ctrl))) => {
            let w = if w_from_size {
                dst.size_bits() == 64
            } else {
                false
            };
            let mandatory_pp = vex_pp(pp);
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                false,
                src.is_extended(),
                w,
                ctrl.base_code() | if ctrl.is_extended() { 8 } else { 0 },
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src.base_code());
            Ok(())
        }
        // 3-operand: r, mem, r (dst=reg, src=r/m, control=vvvv)
        (Some(Register(dst)), Some(Memory(mem)), Some(Register(ctrl))) => {
            let w = if w_from_size {
                dst.size_bits() == 64
            } else {
                false
            };
            let mandatory_pp = vex_pp(pp);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                x_ext,
                b_ext,
                w,
                ctrl.base_code() | if ctrl.is_extended() { 8 } else { 0 },
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            emit_mem_modrm(buf, dst.base_code(), mem);
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected r, r/m, r",
            instr.span,
        )),
    }
}

/// VEX-encoded packed shift instruction dispatcher.
///
/// Handles two forms:
///   1. `vpsllw xmm1, xmm2, xmm3/m128` — 3-operand NDS (reg, reg, reg/mem)
///   2. `vpsllw xmm1, xmm2, imm8`       — 3-operand NDD with /digit (reg, reg, imm)
///
/// Parameters:
///   - `reg_opcode`: opcode for the reg,reg,reg/mem form (ModR/M reg = dst)
///   - `imm_opcode`: opcode for the reg,reg,imm8 form (/digit in reg field)
///   - `digit`: the /digit value for the immediate form
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_vex_shift(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pp: u8,
    escape: &[u8],
    reg_opcode: u8,
    imm_opcode: u8,
    digit: u8,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    use Operand::*;
    match (ops.first(), ops.get(1), ops.get(2)) {
        // 3-operand: reg, reg, reg (shift by register/xmm)
        (Some(Register(dst)), Some(Register(src1)), Some(Register(src2)))
            if dst.is_vector() && src1.is_vector() && src2.is_vector() =>
        {
            let l = dst.is_ymm() || src1.is_ymm();
            encode_vex_rrr(buf, pp, escape, reg_opcode, *dst, *src1, *src2, false, l);
            Ok(())
        }
        // 3-operand: reg, reg, mem (shift by memory)
        (Some(Register(dst)), Some(Register(src1)), Some(Memory(mem)))
            if dst.is_vector() && src1.is_vector() =>
        {
            let l = dst.is_ymm() || src1.is_ymm();
            encode_vex_rrm(
                buf, pp, escape, reg_opcode, *dst, *src1, mem, reloc, false, l,
            );
            Ok(())
        }
        // 3-operand: reg, reg, imm8 (shift by immediate — NDD form)
        (Some(Register(dst)), Some(Register(src)), Some(Immediate(imm)))
            if dst.is_vector() && src.is_vector() =>
        {
            let l = dst.is_ymm() || src.is_ymm();
            let mandatory_pp = vex_pp(pp);
            // NDD: VEX.vvvv = dst, ModR/M r/m = src, ModR/M reg = /digit
            emit_vex_prefix(
                buf,
                false, // reg field is /digit, never extended
                false,
                src.is_extended(),
                false, // W = 0 (WIG)
                dst.base_code() | if dst.is_extended() { 8 } else { 0 },
                l,
                mandatory_pp,
                escape,
            );
            buf.push(imm_opcode);
            buf.push(0xC0 | (digit << 3) | src.base_code());
            buf.push(*imm as u8);
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected VEX xmm/ymm shift operands",
            instr.span,
        )),
    }
}

/// VEX-encoded instruction with mixed GP/XMM operands.
///
/// Handles conversions like VCVTSI2SS (xmm, xmm, r/m32/64):
///   - dst(xmm) = ModR/M reg, src1(xmm) = VEX.vvvv (NDS), src2(GP/mem) = ModR/M r/m
///
/// Also handles the reverse (r32, xmm/m): VCVTSS2SI, VCVTTSS2SI
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_vex_cvt(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pp: u8,
    escape: &[u8],
    opcode: u8,
    _reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    use Operand::*;
    match (ops.first(), ops.get(1), ops.get(2)) {
        // 3-operand: xmm, xmm, r/m (VCVTSI2SS xmm, xmm, r32/r64/m32/m64)
        (Some(Register(dst)), Some(Register(src1)), Some(Register(src2)))
            if dst.is_xmm() && src1.is_xmm() =>
        {
            let w = src2.size_bits() == 64;
            let mandatory_pp = vex_pp(pp);
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                false,
                src2.is_extended(),
                w,
                src1.base_code() | if src1.is_extended() { 8 } else { 0 },
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src2.base_code());
            Ok(())
        }
        (Some(Register(dst)), Some(Register(src1)), Some(Memory(mem)))
            if dst.is_xmm() && src1.is_xmm() =>
        {
            let w = mem.size.is_some_and(|s| s.bits() == 64);
            let mandatory_pp = vex_pp(pp);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                x_ext,
                b_ext,
                w,
                src1.base_code() | if src1.is_extended() { 8 } else { 0 },
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            emit_mem_modrm(buf, dst.base_code(), mem);
            Ok(())
        }
        // 2-operand: r32/r64, xmm (VCVTSS2SI r32, xmm)
        (Some(Register(dst)), Some(Register(src)), None) if !dst.is_vector() && src.is_xmm() => {
            let w = dst.size_bits() == 64;
            let mandatory_pp = vex_pp(pp);
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                false,
                src.is_extended(),
                w,
                0,
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src.base_code());
            Ok(())
        }
        // 2-operand: r32/r64, mem (VCVTSS2SI r32, m32)
        (Some(Register(dst)), Some(Memory(mem)), None) if !dst.is_vector() => {
            let w = dst.size_bits() == 64;
            let mandatory_pp = vex_pp(pp);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_vex_prefix(
                buf,
                dst.is_extended(),
                x_ext,
                b_ext,
                w,
                0,
                false,
                mandatory_pp,
                escape,
            );
            buf.push(opcode);
            emit_mem_modrm(buf, dst.base_code(), mem);
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected conversion operands",
            instr.span,
        )),
    }
}

// ─── EVEX prefix encoding infrastructure (AVX-512) ──────────────────────────
//
// EVEX is a 4-byte prefix (62h + P0 P1 P2) used by AVX-512 instructions.
//
// P0: [R  X  B  R' 0 0 m m]      — R/X/B from legacy REX, R' extends ModR/M.reg to 5 bits, mm = map
// P1: [W  v3 v2 v1 v0 1  p p]    — W, vvvv (inverted NDS src), fixed 1, pp
// P2: [z  L' L  b  V' a a a]     — z=zeroing, L'L=vector length, b=broadcast, V' extends vvvv, aaa=opmask
//
// Vector length: L'L = 00 → 128, 01 → 256, 10 → 512

/// Emit a 4-byte EVEX prefix: 62 [P0] [P1] [P2]
///
/// Parameters:
/// - `r_ext`: ModR/M.reg is extended (bit 3 via ~R, like REX.R)
/// - `x_ext`: SIB.index is extended (bit 3 via ~X, like REX.X)
/// - `b_ext`: ModR/M.rm / SIB.base is extended (bit 3 via ~B, like REX.B)
/// - `r_prime`: ModR/M.reg bit 4 (EVEX.R', inverted — for regs 16-31)
/// - `mm`: map select (01 = 0F, 02 = 0F38, 03 = 0F3A)
/// - `w`: operand size promotion (like REX.W)
/// - `vvvv`: NDS source register (4-bit, inverted in prefix)
/// - `v_prime`: vvvv bit 4 (EVEX.V', inverted — for NDS regs 16-31)
/// - `pp`: implied mandatory prefix (00=none, 01=66, 10=F3, 11=F2)
/// - `z`: zeroing-masking (1 = zero, 0 = merge)
/// - `ll`: vector length (0=128, 1=256, 2=512)
/// - `b_bit`: broadcast / rounding control / SAE
/// - `aaa`: opmask register number (0 = no mask)
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn emit_evex(
    buf: &mut InstrBytes,
    r_ext: bool,
    x_ext: bool,
    b_ext: bool,
    r_prime: bool,
    mm: u8,
    w: bool,
    vvvv: u8,
    v_prime: bool,
    pp: u8,
    z: bool,
    ll: u8,
    b_bit: bool,
    aaa: u8,
) {
    // P0: ~R ~X ~B ~R' 0 0 mm
    let p0 = (if r_ext { 0 } else { 0x80 })
        | (if x_ext { 0 } else { 0x40 })
        | (if b_ext { 0 } else { 0x20 })
        | (if r_prime { 0 } else { 0x10 })
        | (mm & 0x03);

    // P1: W ~v3 ~v2 ~v1 ~v0 1 pp
    let p1 = (if w { 0x80 } else { 0 })
        | (((!vvvv) & 0x0F) << 3)
        | 0x04 // fixed bit
        | (pp & 0x03);

    // P2: z L'L b ~V' aaa
    let p2 = (if z { 0x80 } else { 0 })
        | ((ll & 0x03) << 5)
        | (if b_bit { 0x10 } else { 0 })
        | (if v_prime { 0 } else { 0x08 })
        | (aaa & 0x07);

    buf.push(0x62);
    buf.push(p0);
    buf.push(p1);
    buf.push(p2);
}

/// Helper: compute EVEX register encoding bits from a Register.
/// Returns (base_code_3bit, is_extended_bit3, is_evex_bit4)
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn evex_reg_bits(reg: Register) -> (u8, bool, bool) {
    (reg.base_code(), reg.is_extended(), reg.is_evex_extended())
}

/// Helper: Compute the EVEX vector length field from operand registers.
/// Returns the L'L value: 0 = 128 (XMM), 1 = 256 (YMM), 2 = 512 (ZMM).
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn evex_ll(reg: Register) -> u8 {
    if reg.is_zmm() {
        2
    } else if reg.is_ymm() {
        1
    } else {
        0
    }
}

/// Emit EVEX prefix for a reg,reg,reg instruction.
/// `dst` = ModR/M.reg (destination), `src1` = vvvv (NDS), `src2` = ModR/M.rm
///
/// `aaa` and `z` support opmask + zeroing. Pass aaa=0, z=false for unmasked.
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn emit_evex_prefix_rrr(
    buf: &mut InstrBytes,
    dst: Register,
    src1: Register,
    src2: Register,
    mm: u8,
    w: bool,
    pp: u8,
    ll: u8,
    aaa: u8,
    z: bool,
) {
    let (_, dst_ext, dst_evex) = evex_reg_bits(dst);
    let (src1_code, src1_ext, src1_evex) = evex_reg_bits(src1);
    let (_, src2_ext, src2_evex) = evex_reg_bits(src2);

    let vvvv = src1_code | if src1_ext { 8 } else { 0 };

    emit_evex(
        buf, dst_ext,   // R: reg bit 3
        src2_evex, // X: r/m bit 4 (repurposed in reg-reg form)
        src2_ext,  // B: r/m bit 3
        dst_evex,  // R': reg bit 4
        mm, w, vvvv, src1_evex, // V': vvvv bit 4
        pp, z, ll, false, // b: no broadcast for reg-reg
        aaa,
    );
}

/// Emit EVEX prefix for a reg,reg,mem instruction.
/// `dst` = ModR/M.reg (destination), `src1` = vvvv (NDS), `mem` = memory operand
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn emit_evex_prefix_rrm(
    buf: &mut InstrBytes,
    dst: Register,
    src1: Register,
    mem: &MemoryOperand,
    mm: u8,
    w: bool,
    pp: u8,
    ll: u8,
    b_bit: bool,
    aaa: u8,
    z: bool,
) {
    let (_, dst_ext, dst_evex) = evex_reg_bits(dst);
    let (src1_code, src1_ext, src1_evex) = evex_reg_bits(src1);

    let vvvv = src1_code | if src1_ext { 8 } else { 0 };
    let x_ext = mem.index.is_some_and(|r| r.is_extended());
    let b_ext = mem.base.is_some_and(|r| r.is_extended());

    emit_evex(
        buf, dst_ext, x_ext, b_ext, dst_evex, mm, w, vvvv, src1_evex, pp, z, ll, b_bit, aaa,
    );
}

/// Generic EVEX-encoded AVX-512 instruction dispatcher.
/// Handles the common patterns:
///   - evex_op zmm, zmm, zmm    (3 register operands, NDS form)
///   - evex_op zmm, zmm, [mem]  (reg, reg, mem — optional broadcast)
///   - evex_op zmm, zmm         (2-operand reg-reg, vvvv=0)
///   - evex_op zmm, [mem]       (2-operand load)
///   - evex_op [mem], zmm       (2-operand store)
///
/// Opmask/zeroing/broadcast are read from `instr.opmask`, `instr.zeroing`,
/// and `instr.broadcast` fields set by the parser.
#[cfg(any(feature = "x86", feature = "x86_64"))]
fn evex_aaa(instr: &Instruction) -> u8 {
    instr.opmask.map_or(0, |reg| reg.base_code())
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
fn evex_broadcast_bit(instr: &Instruction) -> bool {
    instr.broadcast.is_some()
}

#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_evex_op(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pp: u8,
    mm: u8,
    load_opcode: u8,
    store_opcode: Option<u8>,
    w: bool,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    use Operand::*;
    let mandatory_pp = vex_pp(pp);
    let o = (ops.first(), ops.get(1), ops.get(2));
    match o {
        // 3-operand: reg, reg, reg
        (Some(Register(dst)), Some(Register(src1)), Some(Register(src2)))
            if dst.is_vector() && src1.is_vector() && src2.is_vector() =>
        {
            let ll = evex_ll(*dst);
            let aaa = evex_aaa(instr);
            emit_evex_prefix_rrr(
                buf,
                *dst,
                *src1,
                *src2,
                mm,
                w,
                mandatory_pp,
                ll,
                aaa,
                instr.zeroing,
            );
            buf.push(load_opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src2.base_code());
            Ok(())
        }
        // 3-operand: reg, reg, mem
        (Some(Register(dst)), Some(Register(src1)), Some(Memory(mem)))
            if dst.is_vector() && src1.is_vector() =>
        {
            let ll = evex_ll(*dst);
            let aaa = evex_aaa(instr);
            let b_bit = evex_broadcast_bit(instr);
            emit_evex_prefix_rrm(
                buf,
                *dst,
                *src1,
                mem,
                mm,
                w,
                mandatory_pp,
                ll,
                b_bit,
                aaa,
                instr.zeroing,
            );
            buf.push(load_opcode);
            emit_mem_modrm(buf, dst.base_code(), mem);
            if let Some(ref mut rel) = reloc {
                rel.offset = buf.len() - 4;
            }
            Ok(())
        }
        // 2-operand reg, reg (move-like: vvvv unused, use K0 as dummy NDS)
        (Some(Register(dst)), Some(Register(src)), None) if dst.is_vector() && src.is_vector() => {
            let ll = evex_ll(*dst);
            let aaa = evex_aaa(instr);
            // vvvv = 0 (K0 stand-in for unused NDS)
            let (_, src_ext, src_evex) = evex_reg_bits(*src);
            let (_, dst_ext, dst_evex) = evex_reg_bits(*dst);
            emit_evex(
                buf,
                dst_ext,
                src_evex,
                src_ext,
                dst_evex,
                mm,
                w,
                0,
                false,
                mandatory_pp,
                instr.zeroing,
                ll,
                false,
                aaa,
            );
            buf.push(load_opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src.base_code());
            Ok(())
        }
        // 2-operand reg, mem (load)
        (Some(Register(dst)), Some(Memory(mem)), None) if dst.is_vector() => {
            let ll = evex_ll(*dst);
            let aaa = evex_aaa(instr);
            let b_bit = evex_broadcast_bit(instr);
            let (_, dst_ext, dst_evex) = evex_reg_bits(*dst);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_evex(
                buf,
                dst_ext,
                x_ext,
                b_ext,
                dst_evex,
                mm,
                w,
                0,
                false,
                mandatory_pp,
                instr.zeroing,
                ll,
                b_bit,
                aaa,
            );
            buf.push(load_opcode);
            emit_mem_modrm(buf, dst.base_code(), mem);
            if let Some(ref mut rel) = reloc {
                rel.offset = buf.len() - 4;
            }
            Ok(())
        }
        // 2-operand mem, reg (store)
        (Some(Memory(mem)), Some(Register(src)), None)
            if src.is_vector() && store_opcode.is_some() =>
        {
            let ll = evex_ll(*src);
            let aaa = evex_aaa(instr);
            let (_, src_ext, src_evex) = evex_reg_bits(*src);
            let x_ext = mem.index.is_some_and(|r| r.is_extended());
            let b_ext = mem.base.is_some_and(|r| r.is_extended());
            emit_evex(
                buf,
                src_ext,
                x_ext,
                b_ext,
                src_evex,
                mm,
                w,
                0,
                false,
                mandatory_pp,
                instr.zeroing,
                ll,
                false,
                aaa,
            );
            // SAFETY: match guard `store_opcode.is_some()` guarantees Some
            buf.push(store_opcode.unwrap_or(0));
            emit_mem_modrm(buf, src.base_code(), mem);
            if let Some(ref mut rel) = reloc {
                rel.offset = buf.len() - 4;
            }
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected EVEX zmm/ymm/xmm operands",
            instr.span,
        )),
    }
}

/// EVEX instruction with immediate byte (4 operands: dst, src1, src2/mem, imm8)
#[cfg(any(feature = "x86", feature = "x86_64"))]
pub(crate) fn encode_evex_imm(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pp: u8,
    mm: u8,
    opcode: u8,
    w: bool,
    reloc: &mut Option<Relocation>,
) -> Result<(), AsmError> {
    use Operand::*;
    let mandatory_pp = vex_pp(pp);
    match (ops.first(), ops.get(1), ops.get(2), ops.get(3)) {
        // reg, reg, reg, imm8
        (Some(Register(dst)), Some(Register(src1)), Some(Register(src2)), Some(Immediate(imm)))
            if dst.is_vector() && src1.is_vector() && src2.is_vector() =>
        {
            let ll = evex_ll(*dst);
            let aaa = evex_aaa(instr);
            emit_evex_prefix_rrr(
                buf,
                *dst,
                *src1,
                *src2,
                mm,
                w,
                mandatory_pp,
                ll,
                aaa,
                instr.zeroing,
            );
            buf.push(opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src2.base_code());
            buf.push(*imm as u8);
            Ok(())
        }
        // reg, reg, mem, imm8
        (Some(Register(dst)), Some(Register(src1)), Some(Memory(mem)), Some(Immediate(imm)))
            if dst.is_vector() && src1.is_vector() =>
        {
            let ll = evex_ll(*dst);
            let aaa = evex_aaa(instr);
            let b_bit = evex_broadcast_bit(instr);
            emit_evex_prefix_rrm(
                buf,
                *dst,
                *src1,
                mem,
                mm,
                w,
                mandatory_pp,
                ll,
                b_bit,
                aaa,
                instr.zeroing,
            );
            buf.push(opcode);
            emit_mem_modrm(buf, dst.base_code(), mem);
            if let Some(ref mut rel) = reloc {
                rel.offset = buf.len() - 4;
            }
            buf.push(*imm as u8);
            Ok(())
        }
        // 3-operand with imm: reg, reg/mem, imm8 (some instructions are NDS-free)
        (Some(Register(dst)), Some(Register(src)), Some(Immediate(imm)), None)
            if dst.is_vector() && src.is_vector() =>
        {
            let ll = evex_ll(*dst);
            let aaa = evex_aaa(instr);
            let (_, src_ext, src_evex) = evex_reg_bits(*src);
            let (_, dst_ext, dst_evex) = evex_reg_bits(*dst);
            emit_evex(
                buf,
                dst_ext,
                src_evex,
                src_ext,
                dst_evex,
                mm,
                w,
                0,
                false,
                mandatory_pp,
                instr.zeroing,
                ll,
                false,
                aaa,
            );
            buf.push(opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src.base_code());
            buf.push(*imm as u8);
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected EVEX operands with immediate",
            instr.span,
        )),
    }
}

/// EVEX opmask instruction: `KADDB k1, k2, k3` etc.
/// Opmask-to-opmask operations use EVEX with vvvv=k, reg=k, r/m=k.
/// Will be wired up when opmask instruction dispatch is added.
#[cfg(any(feature = "x86", feature = "x86_64"))]
#[allow(dead_code)]
pub(crate) fn encode_evex_opmask(
    buf: &mut InstrBytes,
    ops: &[Operand],
    instr: &Instruction,
    pp: u8,
    _mm: u8,
    opcode: u8,
    w: bool,
) -> Result<(), AsmError> {
    use Operand::*;
    let mandatory_pp = vex_pp(pp);
    match (ops.first(), ops.get(1), ops.get(2)) {
        // 3-operand: k, k, k (e.g., KADDB k1, k2, k3)
        (Some(Register(dst)), Some(Register(src1)), Some(Register(src2)))
            if dst.is_opmask() && src1.is_opmask() && src2.is_opmask() =>
        {
            // Opmask instructions use VEX encoding (not EVEX) per Intel SDM
            // VEX.L=1 for most opmask ops, vvvv=src1
            let src1_vvvv = src1.base_code();
            emit_vex_prefix(
                buf,
                false,
                false,
                false,
                w,
                src1_vvvv,
                true,
                mandatory_pp,
                &[0x0F],
            );
            buf.push(opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src2.base_code());
            Ok(())
        }
        // 2-operand: k, k (e.g., KNOTB k1, k2)
        (Some(Register(dst)), Some(Register(src)), None) if dst.is_opmask() && src.is_opmask() => {
            emit_vex_prefix(buf, false, false, false, w, 0, true, mandatory_pp, &[0x0F]);
            buf.push(opcode);
            buf.push(0xC0 | (dst.base_code() << 3) | src.base_code());
            Ok(())
        }
        _ => Err(invalid_operands(
            &instr.mnemonic,
            "expected opmask register operands (k0-k7)",
            instr.span,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;

    fn span() -> Span {
        Span::new(1, 1, 0, 0)
    }

    fn make_instr(mnemonic: &str, operands: Vec<Operand>) -> Instruction {
        Instruction {
            mnemonic: Mnemonic::from(mnemonic),
            operands: OperandList::from(operands),
            size_hint: None,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        }
    }

    fn make_instr_with_hint(
        mnemonic: &str,
        operands: Vec<Operand>,
        hint: Option<OperandSize>,
    ) -> Instruction {
        Instruction {
            mnemonic: Mnemonic::from(mnemonic),
            operands: OperandList::from(operands),
            size_hint: hint,
            prefixes: PrefixList::new(),
            opmask: None,
            zeroing: false,
            broadcast: None,
            span: span(),
        }
    }

    fn encode(mnemonic: &str, operands: Vec<Operand>) -> Vec<u8> {
        let instr = make_instr(mnemonic, operands);
        encode_instruction(&instr, Arch::X86_64)
            .unwrap()
            .bytes
            .to_vec()
    }

    fn encode_with_hint(mnemonic: &str, operands: Vec<Operand>, hint: OperandSize) -> Vec<u8> {
        let mut instr = make_instr(mnemonic, operands);
        instr.size_hint = Some(hint);
        encode_instruction(&instr, Arch::X86_64)
            .unwrap()
            .bytes
            .to_vec()
    }

    fn encode_with_prefix(mnemonic: &str, operands: Vec<Operand>, prefix: Prefix) -> Vec<u8> {
        let mut instr = make_instr(mnemonic, operands);
        instr.prefixes = PrefixList::from(alloc::vec![prefix]);
        encode_instruction(&instr, Arch::X86_64)
            .unwrap()
            .bytes
            .to_vec()
    }

    use crate::ir::Register::*;
    use Operand::*;

    // === Zero-operand instructions ===

    #[test]
    fn test_nop() {
        assert_eq!(encode("nop", vec![]), vec![0x90]);
    }

    #[test]
    fn test_ret() {
        assert_eq!(encode("ret", vec![]), vec![0xC3]);
    }

    #[test]
    fn test_syscall() {
        assert_eq!(encode("syscall", vec![]), vec![0x0F, 0x05]);
    }

    #[test]
    fn test_int3() {
        assert_eq!(encode("int3", vec![]), vec![0xCC]);
    }

    #[test]
    fn test_int_0x80() {
        assert_eq!(encode("int", vec![Immediate(0x80)]), vec![0xCD, 0x80]);
    }

    #[test]
    fn test_hlt() {
        assert_eq!(encode("hlt", vec![]), vec![0xF4]);
    }

    // === MOV reg, reg ===

    #[test]
    fn test_mov_rax_rbx() {
        // REX.W + 89 /r (mov r/m64, r64)
        let bytes = encode("mov", vec![Register(Rax), Register(Rbx)]);
        assert_eq!(bytes, vec![0x48, 0x89, 0xD8]);
    }

    #[test]
    fn test_mov_eax_ecx() {
        // 89 /r (mov r/m32, r32)
        let bytes = encode("mov", vec![Register(Eax), Register(Ecx)]);
        assert_eq!(bytes, vec![0x89, 0xC8]);
    }

    #[test]
    fn test_mov_r8_r9() {
        // REX.W+REX.R+REX.B + 89 /r
        let bytes = encode("mov", vec![Register(R8), Register(R9)]);
        assert_eq!(bytes, vec![0x4D, 0x89, 0xC8]);
    }

    #[test]
    fn test_mov_al_bl() {
        let bytes = encode("mov", vec![Register(Al), Register(Bl)]);
        assert_eq!(bytes, vec![0x88, 0xD8]);
    }

    // === MOV reg, imm ===

    #[test]
    fn test_mov_eax_1() {
        let bytes = encode("mov", vec![Register(Eax), Immediate(1)]);
        assert_eq!(bytes, vec![0xB8, 0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_mov_al_0xff() {
        let bytes = encode("mov", vec![Register(Al), Immediate(0xFF)]);
        assert_eq!(bytes, vec![0xB0, 0xFF]);
    }

    #[test]
    fn test_mov_rax_small_imm() {
        // Should use mov eax, imm32 (zero extends to rax)
        let bytes = encode("mov", vec![Register(Rax), Immediate(1)]);
        assert_eq!(bytes, vec![0xB8, 0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_mov_rax_neg1() {
        // mov rax, -1 → REX.W C7 /0 imm32 (sign-extended)
        let bytes = encode("mov", vec![Register(Rax), Immediate(-1)]);
        assert_eq!(bytes, vec![0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn test_mov_rax_large_imm() {
        // movabs rax, imm64
        let bytes = encode("mov", vec![Register(Rax), Immediate(0x0102030405060708)]);
        assert_eq!(
            bytes,
            vec![0x48, 0xB8, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
        );
    }

    #[test]
    fn test_mov_r8d_imm() {
        let bytes = encode("mov", vec![Register(R8d), Immediate(42)]);
        assert_eq!(bytes, vec![0x41, 0xB8, 0x2A, 0x00, 0x00, 0x00]);
    }

    // === MOV reg, [mem] and [mem], reg ===

    #[test]
    fn test_mov_rax_mem_rbx() {
        // mov rax, [rbx] → REX.W 8B 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode("mov", vec![Register(Rax), Memory(Box::new(mem))]);
        assert_eq!(bytes, vec![0x48, 0x8B, 0x03]);
    }

    #[test]
    fn test_mov_mem_rbx_rax() {
        // mov [rbx], rax → REX.W 89 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode("mov", vec![Memory(Box::new(mem)), Register(Rax)]);
        assert_eq!(bytes, vec![0x48, 0x89, 0x03]);
    }

    #[test]
    fn test_mov_rax_mem_rbp_disp8() {
        // mov rax, [rbp + 8] → REX.W 8B 45 08
        let mem = MemoryOperand {
            base: Some(Rbp),
            disp: 8,
            ..Default::default()
        };
        let bytes = encode("mov", vec![Register(Rax), Memory(Box::new(mem))]);
        assert_eq!(bytes, vec![0x48, 0x8B, 0x45, 0x08]);
    }

    #[test]
    fn test_mov_rax_mem_rbp_disp32() {
        // mov rax, [rbp + 0x200] → REX.W 8B 85 00 02 00 00
        let mem = MemoryOperand {
            base: Some(Rbp),
            disp: 0x200,
            ..Default::default()
        };
        let bytes = encode("mov", vec![Register(Rax), Memory(Box::new(mem))]);
        assert_eq!(bytes, vec![0x48, 0x8B, 0x85, 0x00, 0x02, 0x00, 0x00]);
    }

    // === PUSH / POP ===

    #[test]
    fn test_push_rbp() {
        assert_eq!(encode("push", vec![Register(Rbp)]), vec![0x55]);
    }

    #[test]
    fn test_push_r12() {
        assert_eq!(encode("push", vec![Register(R12)]), vec![0x41, 0x54]);
    }

    #[test]
    fn test_pop_rbp() {
        assert_eq!(encode("pop", vec![Register(Rbp)]), vec![0x5D]);
    }

    #[test]
    fn test_push_imm8() {
        assert_eq!(encode("push", vec![Immediate(1)]), vec![0x6A, 0x01]);
    }

    #[test]
    fn test_push_imm32() {
        assert_eq!(
            encode("push", vec![Immediate(0x1000)]),
            vec![0x68, 0x00, 0x10, 0x00, 0x00]
        );
    }

    // === ALU ===

    #[test]
    fn test_add_rax_rbx() {
        let bytes = encode("add", vec![Register(Rax), Register(Rbx)]);
        assert_eq!(bytes, vec![0x48, 0x01, 0xD8]); // REX.W 01 /r
    }

    #[test]
    fn test_add_eax_1() {
        // add eax, 1 → 83 C0 01 (sign-extended imm8)
        let bytes = encode("add", vec![Register(Eax), Immediate(1)]);
        assert_eq!(bytes, vec![0x83, 0xC0, 0x01]);
    }

    #[test]
    fn test_sub_rsp_8() {
        // sub rsp, 8 → REX.W 83 EC 08
        let bytes = encode("sub", vec![Register(Rsp), Immediate(8)]);
        assert_eq!(bytes, vec![0x48, 0x83, 0xEC, 0x08]);
    }

    #[test]
    fn test_xor_eax_eax() {
        let bytes = encode("xor", vec![Register(Eax), Register(Eax)]);
        assert_eq!(bytes, vec![0x31, 0xC0]);
    }

    #[test]
    fn test_cmp_rax_0() {
        let bytes = encode("cmp", vec![Register(Rax), Immediate(0)]);
        assert_eq!(bytes, vec![0x48, 0x83, 0xF8, 0x00]);
    }

    #[test]
    fn test_and_al_imm() {
        // and al, 0x0F → 24 0F (short form)
        let bytes = encode("and", vec![Register(Al), Immediate(0x0F)]);
        assert_eq!(bytes, vec![0x24, 0x0F]);
    }

    #[test]
    fn test_or_eax_large_imm() {
        // or eax, 0x1000 → 0D 00 10 00 00 (short form for eax)
        let bytes = encode("or", vec![Register(Eax), Immediate(0x1000)]);
        assert_eq!(bytes, vec![0x0D, 0x00, 0x10, 0x00, 0x00]);
    }

    // === TEST ===

    #[test]
    fn test_test_al_imm() {
        let bytes = encode("test", vec![Register(Al), Immediate(1)]);
        assert_eq!(bytes, vec![0xA8, 0x01]);
    }

    #[test]
    fn test_test_eax_eax() {
        let bytes = encode("test", vec![Register(Eax), Register(Eax)]);
        assert_eq!(bytes, vec![0x85, 0xC0]);
    }

    // === Shifts ===

    #[test]
    fn test_shl_eax_1() {
        let bytes = encode("shl", vec![Register(Eax), Immediate(1)]);
        assert_eq!(bytes, vec![0xD1, 0xE0]);
    }

    #[test]
    fn test_shr_rcx_4() {
        let bytes = encode("shr", vec![Register(Rcx), Immediate(4)]);
        assert_eq!(bytes, vec![0x48, 0xC1, 0xE9, 0x04]);
    }

    #[test]
    fn test_sar_rax_cl() {
        let bytes = encode("sar", vec![Register(Rax), Register(Cl)]);
        assert_eq!(bytes, vec![0x48, 0xD3, 0xF8]);
    }

    // === INC / DEC ===

    #[test]
    fn test_inc_rax() {
        let bytes = encode("inc", vec![Register(Rax)]);
        assert_eq!(bytes, vec![0x48, 0xFF, 0xC0]);
    }

    #[test]
    fn test_dec_ecx() {
        let bytes = encode("dec", vec![Register(Ecx)]);
        assert_eq!(bytes, vec![0xFF, 0xC9]);
    }

    // === NEG / NOT ===

    #[test]
    fn test_neg_rax() {
        let bytes = encode("neg", vec![Register(Rax)]);
        assert_eq!(bytes, vec![0x48, 0xF7, 0xD8]);
    }

    #[test]
    fn test_not_eax() {
        let bytes = encode("not", vec![Register(Eax)]);
        assert_eq!(bytes, vec![0xF7, 0xD0]);
    }

    // === JMP / CALL ===

    #[test]
    fn test_jmp_label_relocation() {
        let instr = make_instr("jmp", vec![Label(String::from("target"))]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert_eq!(result.bytes, vec![0xE9, 0x00, 0x00, 0x00, 0x00]);
        assert!(result.relocation.is_some());
        let r = result.relocation.unwrap();
        assert_eq!(&*r.label, "target");
        assert_eq!(r.kind, RelocKind::X86Relative);
        assert_eq!(r.size, 4);
    }

    #[test]
    fn test_jmp_reg() {
        let bytes = encode("jmp", vec![Register(Rax)]);
        assert_eq!(bytes, vec![0xFF, 0xE0]);
    }

    #[test]
    fn test_call_reg() {
        let bytes = encode("call", vec![Register(Rax)]);
        assert_eq!(bytes, vec![0xFF, 0xD0]);
    }

    #[test]
    fn test_call_r12() {
        let bytes = encode("call", vec![Register(R12)]);
        assert_eq!(bytes, vec![0x41, 0xFF, 0xD4]);
    }

    // === Jcc ===

    #[test]
    fn test_je_label() {
        let instr = make_instr("je", vec![Label(String::from("target"))]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert_eq!(result.bytes[0..2], [0x0F, 0x84]);
        assert!(result.relocation.is_some());
    }

    #[test]
    fn test_jne_label() {
        let instr = make_instr("jne", vec![Label(String::from("target"))]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert_eq!(result.bytes[0..2], [0x0F, 0x85]);
    }

    // === SETcc ===

    #[test]
    fn test_sete_al() {
        let bytes = encode("sete", vec![Register(Al)]);
        assert_eq!(bytes, vec![0x0F, 0x94, 0xC0]);
    }

    // === CMOVcc ===

    #[test]
    fn test_cmove_rax_rbx() {
        let bytes = encode("cmove", vec![Register(Rax), Register(Rbx)]);
        assert_eq!(bytes, vec![0x48, 0x0F, 0x44, 0xC3]);
    }

    // === MOVZX / MOVSX ===

    #[test]
    fn test_movzx_eax_al() {
        let bytes = encode("movzx", vec![Register(Eax), Register(Al)]);
        assert_eq!(bytes, vec![0x0F, 0xB6, 0xC0]);
    }

    #[test]
    fn test_movsx_rax_eax() {
        let bytes = encode("movsx", vec![Register(Rax), Register(Eax)]);
        assert_eq!(bytes, vec![0x48, 0x63, 0xC0]);
    }

    // === LEA ===

    #[test]
    fn test_lea_rax_rbx_rcx_8() {
        let mem = MemoryOperand {
            base: Some(Rbx),
            index: Some(Rcx),
            scale: 8,
            disp: 0,
            ..Default::default()
        };
        let bytes = encode("lea", vec![Register(Rax), Memory(Box::new(mem))]);
        assert_eq!(bytes, vec![0x48, 0x8D, 0x04, 0xCB]);
    }

    // === Prefix ===

    #[test]
    fn test_lock_prefix() {
        let mem = MemoryOperand {
            base: Some(Rax),
            ..Default::default()
        };
        let bytes = encode_with_prefix(
            "add",
            vec![Memory(Box::new(mem)), Immediate(1)],
            Prefix::Lock,
        );
        assert_eq!(bytes[0], 0xF0); // LOCK prefix
    }

    // === IMUL ===

    #[test]
    fn test_imul_r_r() {
        let bytes = encode("imul", vec![Register(Rax), Register(Rbx)]);
        assert_eq!(bytes, vec![0x48, 0x0F, 0xAF, 0xC3]);
    }

    #[test]
    fn test_imul_r_r_imm8() {
        let bytes = encode("imul", vec![Register(Rax), Register(Rbx), Immediate(10)]);
        assert_eq!(bytes, vec![0x48, 0x6B, 0xC3, 0x0A]);
    }

    // === BSWAP ===

    #[test]
    fn test_bswap_eax() {
        let bytes = encode("bswap", vec![Register(Eax)]);
        assert_eq!(bytes, vec![0x0F, 0xC8]);
    }

    #[test]
    fn test_bswap_rax() {
        let bytes = encode("bswap", vec![Register(Rax)]);
        assert_eq!(bytes, vec![0x48, 0x0F, 0xC8]);
    }

    // === String Instructions ===

    #[test]
    fn test_rep_movsb() {
        let mut instr = make_instr("movsb", vec![]);
        instr.prefixes = PrefixList::from(alloc::vec![Prefix::Rep]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert_eq!(result.bytes, vec![0xF3, 0xA4]);
    }

    // === Multi-byte NOP ===

    #[test]
    fn test_nop3_encoding() {
        let instr = make_instr("nop3", vec![]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert_eq!(result.bytes, vec![0x0F, 0x1F, 0x00]);
    }

    // === RSP/RBP edge cases (SIB byte required) ===

    #[test]
    fn test_mov_rax_mem_rsp() {
        // [rsp] requires SIB byte
        let mem = MemoryOperand {
            base: Some(Rsp),
            ..Default::default()
        };
        let bytes = encode("mov", vec![Register(Rax), Memory(Box::new(mem))]);
        // REX.W 8B 04 24 (ModRM=04h → SIB follows, SIB=24h → base=RSP, no index)
        assert_eq!(bytes, vec![0x48, 0x8B, 0x04, 0x24]);
    }

    // === CDQ / CQO ===

    #[test]
    fn test_cdq() {
        assert_eq!(encode("cdq", vec![]), vec![0x99]);
    }

    #[test]
    fn test_cqo() {
        assert_eq!(encode("cqo", vec![]), vec![0x48, 0x99]);
    }

    // === 16-bit operations ===

    #[test]
    fn test_mov_ax_bx() {
        let bytes = encode("mov", vec![Register(Ax), Register(Bx)]);
        assert_eq!(bytes, vec![0x66, 0x89, 0xD8]);
    }

    #[test]
    fn test_add_ax_imm() {
        let bytes = encode("add", vec![Register(Ax), Immediate(1)]);
        assert_eq!(bytes, vec![0x66, 0x83, 0xC0, 0x01]);
    }

    // === Arch::X86 basic smoke test ===

    #[test]
    fn test_arch_x86_nop() {
        let instr = make_instr("nop", vec![]);
        let result = encode_instruction(&instr, Arch::X86);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().bytes, vec![0x90]);
    }

    // === x86-32 INC/DEC short forms (0x40+rd / 0x48+rd) ===

    #[test]
    fn test_x86_32_inc_eax() {
        // inc eax → 0x40 (short form)
        let instr = make_instr("inc", vec![Register(Eax)]);
        let result = encode_instruction(&instr, Arch::X86).unwrap();
        assert_eq!(result.bytes, vec![0x40]);
    }

    #[test]
    fn test_x86_32_inc_ebx() {
        // inc ebx → 0x43 (0x40 + 3)
        let instr = make_instr("inc", vec![Register(Ebx)]);
        let result = encode_instruction(&instr, Arch::X86).unwrap();
        assert_eq!(result.bytes, vec![0x43]);
    }

    #[test]
    fn test_x86_32_inc_edi() {
        // inc edi → 0x47 (0x40 + 7)
        let instr = make_instr("inc", vec![Register(Edi)]);
        let result = encode_instruction(&instr, Arch::X86).unwrap();
        assert_eq!(result.bytes, vec![0x47]);
    }

    #[test]
    fn test_x86_32_dec_eax() {
        // dec eax → 0x48 (short form)
        let instr = make_instr("dec", vec![Register(Eax)]);
        let result = encode_instruction(&instr, Arch::X86).unwrap();
        assert_eq!(result.bytes, vec![0x48]);
    }

    #[test]
    fn test_x86_32_dec_esp() {
        // dec esp → 0x4C (0x48 + 4)
        let instr = make_instr("dec", vec![Register(Esp)]);
        let result = encode_instruction(&instr, Arch::X86).unwrap();
        assert_eq!(result.bytes, vec![0x4C]);
    }

    #[test]
    fn test_x86_32_inc_ax() {
        // inc ax → 66 40 (16-bit override + short form)
        let instr = make_instr("inc", vec![Register(Ax)]);
        let result = encode_instruction(&instr, Arch::X86).unwrap();
        assert_eq!(result.bytes, vec![0x66, 0x40]);
    }

    #[test]
    fn test_x86_32_dec_cx() {
        // dec cx → 66 49 (16-bit override + short form)
        let instr = make_instr("dec", vec![Register(Cx)]);
        let result = encode_instruction(&instr, Arch::X86).unwrap();
        assert_eq!(result.bytes, vec![0x66, 0x49]);
    }

    #[test]
    fn test_x86_32_inc_al_uses_modrm() {
        // inc al → FE C0 (8-bit uses ModR/M form, not short form)
        let instr = make_instr("inc", vec![Register(Al)]);
        let result = encode_instruction(&instr, Arch::X86).unwrap();
        assert_eq!(result.bytes, vec![0xFE, 0xC0]);
    }

    // === push/pop 16-bit registers ===

    #[test]
    fn test_push_ax() {
        // push ax → 66 50
        let bytes = encode("push", vec![Register(Ax)]);
        assert_eq!(bytes, vec![0x66, 0x50]);
    }

    #[test]
    fn test_pop_ax() {
        // pop ax → 66 58
        let bytes = encode("pop", vec![Register(Ax)]);
        assert_eq!(bytes, vec![0x66, 0x58]);
    }

    #[test]
    fn test_push_bx() {
        let bytes = encode("push", vec![Register(Bx)]);
        assert_eq!(bytes, vec![0x66, 0x53]);
    }

    #[test]
    fn test_pop_bx() {
        let bytes = encode("pop", vec![Register(Bx)]);
        assert_eq!(bytes, vec![0x66, 0x5B]);
    }

    // === xchg 16-bit ===

    #[test]
    fn test_xchg_ax_bx_shortcut() {
        // xchg ax, bx → 66 93
        let bytes = encode("xchg", vec![Register(Ax), Register(Bx)]);
        assert_eq!(bytes, vec![0x66, 0x93]);
    }

    // === movsx with memory operand ===

    #[test]
    fn test_movsx_eax_byte_mem() {
        // movsx eax, byte [rbx] → 0F BE 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode_with_hint(
            "movsx",
            vec![Register(Eax), Memory(Box::new(mem))],
            OperandSize::Byte,
        );
        assert_eq!(bytes, vec![0x0F, 0xBE, 0x03]);
    }

    #[test]
    fn test_movsx_rax_word_mem() {
        // movsx rax, word [rbx] → 48 0F BF 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode_with_hint(
            "movsx",
            vec![Register(Rax), Memory(Box::new(mem))],
            OperandSize::Word,
        );
        assert_eq!(bytes, vec![0x48, 0x0F, 0xBF, 0x03]);
    }

    #[test]
    fn test_movsxd_rax_dword_mem() {
        // movsxd rax, dword [rbx] → 48 63 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode_with_hint(
            "movsx",
            vec![Register(Rax), Memory(Box::new(mem))],
            OperandSize::Dword,
        );
        assert_eq!(bytes, vec![0x48, 0x63, 0x03]);
    }

    // === bsf/bsr with memory operand ===

    #[test]
    fn test_bsf_eax_mem() {
        // bsf eax, [rbx] → 0F BC 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode("bsf", vec![Register(Eax), Memory(Box::new(mem))]);
        assert_eq!(bytes, vec![0x0F, 0xBC, 0x03]);
    }

    #[test]
    fn test_bsr_rax_mem() {
        // bsr rax, [rbx] → 48 0F BD 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode("bsr", vec![Register(Rax), Memory(Box::new(mem))]);
        assert_eq!(bytes, vec![0x48, 0x0F, 0xBD, 0x03]);
    }

    // === popcnt/lzcnt/tzcnt with memory operand ===

    #[test]
    fn test_popcnt_eax_mem() {
        // popcnt eax, [rbx] → F3 0F B8 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode("popcnt", vec![Register(Eax), Memory(Box::new(mem))]);
        assert_eq!(bytes, vec![0xF3, 0x0F, 0xB8, 0x03]);
    }

    #[test]
    fn test_lzcnt_rax_mem() {
        // lzcnt rax, [rbx] → F3 48 0F BD 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode("lzcnt", vec![Register(Rax), Memory(Box::new(mem))]);
        assert_eq!(bytes, vec![0xF3, 0x48, 0x0F, 0xBD, 0x03]);
    }

    #[test]
    fn test_tzcnt_eax_mem() {
        // tzcnt eax, [rbx] → F3 0F BC 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode("tzcnt", vec![Register(Eax), Memory(Box::new(mem))]);
        assert_eq!(bytes, vec![0xF3, 0x0F, 0xBC, 0x03]);
    }

    // === bt/bts/btr/btc with memory operands ===

    #[test]
    fn test_bt_mem_reg() {
        // bt [rbx], eax → 0F A3 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode("bt", vec![Memory(Box::new(mem)), Register(Eax)]);
        assert_eq!(bytes, vec![0x0F, 0xA3, 0x03]);
    }

    #[test]
    fn test_bts_mem_imm() {
        // bts dword [rbx], 5 → 0F BA 2B 05
        let mem = MemoryOperand {
            base: Some(Rbx),
            size: Some(OperandSize::Dword),
            ..Default::default()
        };
        let bytes = encode("bts", vec![Memory(Box::new(mem)), Immediate(5)]);
        assert_eq!(bytes, vec![0x0F, 0xBA, 0x2B, 0x05]);
    }

    // === shift instructions with memory operands ===

    #[test]
    fn test_shl_mem_1() {
        // shl dword [rbx], 1 → D1 23
        let mem = MemoryOperand {
            base: Some(Rbx),
            size: Some(OperandSize::Dword),
            ..Default::default()
        };
        let bytes = encode("shl", vec![Memory(Box::new(mem)), Immediate(1)]);
        assert_eq!(bytes, vec![0xD1, 0x23]);
    }

    #[test]
    fn test_shr_mem_imm() {
        // shr dword [rbx], 4 → C1 2B 04
        let mem = MemoryOperand {
            base: Some(Rbx),
            size: Some(OperandSize::Dword),
            ..Default::default()
        };
        let bytes = encode("shr", vec![Memory(Box::new(mem)), Immediate(4)]);
        assert_eq!(bytes, vec![0xC1, 0x2B, 0x04]);
    }

    #[test]
    fn test_sar_mem_cl() {
        // sar dword [rbx], cl → D3 3B
        let mem = MemoryOperand {
            base: Some(Rbx),
            size: Some(OperandSize::Dword),
            ..Default::default()
        };
        let bytes = encode("sar", vec![Memory(Box::new(mem)), Register(Cl)]);
        assert_eq!(bytes, vec![0xD3, 0x3B]);
    }

    // === mov_reg_imm error on unsupported size ===

    #[test]
    fn test_mov_xmm_imm_error() {
        let instr = make_instr("mov", vec![Register(Xmm0), Immediate(1)]);
        let result = encode_instruction(&instr, Arch::X86_64);
        assert!(result.is_err());
    }

    // === Expression operands ===

    #[test]
    fn test_mov_rax_label_expression() {
        // mov rax, label+8 → REX.W B8 <imm64> with relocation addend=8
        let expr = Expr::Add(
            Box::new(Expr::Label(String::from("data"))),
            Box::new(Expr::Num(8)),
        );
        let instr = make_instr("mov", vec![Register(Rax), Expression(expr)]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert_eq!(result.bytes[0], 0x48); // REX.W
        assert_eq!(result.bytes[1], 0xB8); // mov rax, imm64
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "data");
        assert_eq!(reloc.addend, 8);
        assert_eq!(reloc.size, 8);
        assert_eq!(reloc.kind, RelocKind::Absolute);
    }

    #[test]
    fn test_jmp_label_expression() {
        let expr = Expr::Sub(
            Box::new(Expr::Label(String::from("target"))),
            Box::new(Expr::Num(2)),
        );
        let instr = make_instr("jmp", vec![Expression(expr)]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert_eq!(result.bytes[0], 0xE9); // jmp rel32
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "target");
        assert_eq!(reloc.addend, -2);
        assert_eq!(reloc.kind, RelocKind::X86Relative);
        assert!(result.relax.is_some()); // Should have relaxation info
    }

    #[test]
    fn test_call_label_expression() {
        let expr = Expr::Add(
            Box::new(Expr::Label(String::from("func"))),
            Box::new(Expr::Num(4)),
        );
        let instr = make_instr("call", vec![Expression(expr)]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert_eq!(result.bytes[0], 0xE8); // call rel32
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "func");
        assert_eq!(reloc.addend, 4);
    }

    #[test]
    fn test_jcc_label_expression() {
        let expr = Expr::Add(
            Box::new(Expr::Label(String::from("dest"))),
            Box::new(Expr::Num(0)),
        );
        let instr = make_instr("je", vec![Expression(expr)]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert_eq!(result.bytes[0], 0x0F); // je near
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "dest");
        assert_eq!(reloc.addend, 0);
    }

    #[test]
    fn test_push_label_expression() {
        let expr = Expr::Add(
            Box::new(Expr::Label(String::from("data"))),
            Box::new(Expr::Num(16)),
        );
        let instr = make_instr("push", vec![Expression(expr)]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert_eq!(result.bytes[0], 0x68); // push imm32
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "data");
        assert_eq!(reloc.addend, 16);
    }

    #[test]
    fn test_loop_label_expression() {
        let expr = Expr::Sub(
            Box::new(Expr::Label(String::from("top"))),
            Box::new(Expr::Num(1)),
        );
        let instr = make_instr("loop", vec![Expression(expr)]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // Long form: E2 02 EB 05 E9 [rel32]
        assert_eq!(result.bytes[0], 0xE2); // loop
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "top");
        assert_eq!(reloc.addend, -1);
        assert_eq!(reloc.size, 4); // rel32 in long form
                                   // Relaxation short form provided
        assert!(result.relax.is_some());
        let ri = result.relax.unwrap();
        assert_eq!(ri.short_bytes[0], 0xE2);
        assert_eq!(ri.short_bytes.len(), 2);
    }

    // === extract_label helper ===

    #[test]
    fn test_extract_label_plain() {
        let op = Label(String::from("foo"));
        assert_eq!(extract_label(&op), Some(("foo", 0)));
    }

    #[test]
    fn test_extract_label_expression() {
        let expr = Expr::Add(
            Box::new(Expr::Label(String::from("bar"))),
            Box::new(Expr::Num(10)),
        );
        assert_eq!(extract_label(&Expression(expr)), Some(("bar", 10)));
    }

    #[test]
    fn test_extract_label_non_label() {
        assert_eq!(extract_label(&Immediate(42)), None);
        assert_eq!(extract_label(&Register(Rax)), None);
    }

    // === IMUL reg, mem, imm ===

    #[test]
    fn test_imul_reg_mem_imm8() {
        // imul eax, [rcx], 5 → 6B 01 05
        let mem = MemoryOperand {
            base: Some(Rcx),
            ..Default::default()
        };
        let bytes = encode(
            "imul",
            vec![Register(Eax), Memory(Box::new(mem)), Immediate(5)],
        );
        assert_eq!(bytes, vec![0x6B, 0x01, 0x05]);
    }

    #[test]
    fn test_imul_reg_mem_imm32() {
        // imul rax, [rdx], 1000 → 48 69 02 E8 03 00 00
        let mem = MemoryOperand {
            base: Some(Rdx),
            ..Default::default()
        };
        let bytes = encode(
            "imul",
            vec![Register(Rax), Memory(Box::new(mem)), Immediate(1000)],
        );
        assert_eq!(bytes, vec![0x48, 0x69, 0x02, 0xE8, 0x03, 0x00, 0x00]);
    }

    // === ret imm16 ===

    #[test]
    fn test_ret_imm16() {
        // ret 8 → C2 08 00
        let bytes = encode("ret", vec![Immediate(8)]);
        assert_eq!(bytes, vec![0xC2, 0x08, 0x00]);
    }

    #[test]
    fn test_ret_imm16_large() {
        // ret 0x1234 → C2 34 12
        let bytes = encode("ret", vec![Immediate(0x1234)]);
        assert_eq!(bytes, vec![0xC2, 0x34, 0x12]);
    }

    #[test]
    fn test_retn_alias() {
        // retn = ret (near return)
        assert_eq!(encode("retn", vec![]), vec![0xC3]);
        assert_eq!(encode("retn", vec![Immediate(4)]), vec![0xC2, 0x04, 0x00]);
    }

    // === retf / lret (far return) ===

    #[test]
    fn test_retf() {
        assert_eq!(encode("retf", vec![]), vec![0xCB]);
    }

    #[test]
    fn test_retf_imm16() {
        // retf 4 → CA 04 00
        let bytes = encode("retf", vec![Immediate(4)]);
        assert_eq!(bytes, vec![0xCA, 0x04, 0x00]);
    }

    #[test]
    fn test_lret_alias() {
        assert_eq!(encode("lret", vec![]), vec![0xCB]);
        assert_eq!(encode("lret", vec![Immediate(8)]), vec![0xCA, 0x08, 0x00]);
    }

    // === movabs alias ===

    #[test]
    fn test_movabs_alias() {
        // movabs rax, 0x12345678 should work like mov rax, 0x12345678
        let bytes_mov = encode("mov", vec![Register(Rax), Immediate(0x12345678)]);
        let bytes_movabs = encode("movabs", vec![Register(Rax), Immediate(0x12345678)]);
        assert_eq!(bytes_mov, bytes_movabs);
    }

    #[test]
    fn test_movabs_imm64() {
        // movabs rax, 0x0102030405060708
        let bytes = encode("movabs", vec![Register(Rax), Immediate(0x0102030405060708)]);
        assert_eq!(
            bytes,
            vec![0x48, 0xB8, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
        );
    }

    // === REX + high-byte conflict detection ===

    #[test]
    fn test_high_byte_rex_conflict_rejected() {
        // mov ah, sil should fail: AH is incompatible with REX (needed for SIL)
        let instr = make_instr("mov", vec![Register(Ah), Register(Sil)]);
        let result = encode_instruction(&instr, Arch::X86_64);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AsmError::InvalidOperands { detail, .. } => {
                assert!(detail.contains("high-byte"));
            }
            other => panic!("expected InvalidOperands, got {:?}", other),
        }
    }

    #[test]
    fn test_high_byte_extended_reg_conflict_rejected() {
        // add ah, r8b should fail: AH + extended register
        let instr = make_instr("add", vec![Register(Ah), Register(R8b)]);
        let result = encode_instruction(&instr, Arch::X86_64);
        assert!(result.is_err());
    }

    #[test]
    fn test_high_byte_without_rex_ok() {
        // mov ah, al — both legacy 8-bit, no REX needed → should work
        // AH=code 4, AL=code 0, opcode 0x88 r/m,r: modrm(11, 0, 4) = 0xC4
        let bytes = encode("mov", vec![Register(Ah), Register(Al)]);
        assert_eq!(bytes, vec![0x88, 0xC4]);
    }

    #[test]
    fn test_high_byte_pair_ok() {
        // xor ah, ch — two high-byte regs, no REX needed → should work
        // AH=code 4, CH=code 5, opcode 0x30 r/m,r: modrm(11, 5, 4) = 0xEC
        let bytes = encode("xor", vec![Register(Ah), Register(Ch)]);
        assert_eq!(bytes, vec![0x30, 0xEC]);
    }

    // === LOCK prefix validation ===

    #[test]
    fn test_lock_valid_memory_dest() {
        // lock add dword ptr [rax], 1 — valid (memory destination)
        let mem = MemoryOperand {
            base: Some(Rax),
            ..Default::default()
        };
        let bytes = encode_with_prefix(
            "add",
            vec![Memory(Box::new(mem)), Immediate(1)],
            Prefix::Lock,
        );
        assert_eq!(bytes[0], 0xF0);
    }

    #[test]
    fn test_lock_invalid_reg_dest() {
        // lock add eax, ebx — invalid (register destination)
        let mut instr = make_instr("add", vec![Register(Eax), Register(Ebx)]);
        instr.prefixes = PrefixList::from(alloc::vec![Prefix::Lock]);
        let result = encode_instruction(&instr, Arch::X86_64);
        assert!(result.is_err());
        match result.unwrap_err() {
            AsmError::InvalidOperands { detail, .. } => {
                assert!(detail.contains("LOCK"));
            }
            other => panic!("expected InvalidOperands for LOCK, got {:?}", other),
        }
    }

    #[test]
    fn test_lock_invalid_imm_dest() {
        // lock xchg eax, ecx — invalid (register destination, even though xchg is lockable)
        let mut instr = make_instr("xchg", vec![Register(Eax), Register(Ecx)]);
        instr.prefixes = PrefixList::from(alloc::vec![Prefix::Lock]);
        let result = encode_instruction(&instr, Arch::X86_64);
        assert!(result.is_err());
    }

    // === Push/Pop segment registers ===

    #[test]
    fn test_push_fs() {
        // push fs → 0F A0
        assert_eq!(encode("push", vec![Register(Fs)]), vec![0x0F, 0xA0]);
    }

    #[test]
    fn test_push_gs() {
        // push gs → 0F A8
        assert_eq!(encode("push", vec![Register(Gs)]), vec![0x0F, 0xA8]);
    }

    #[test]
    fn test_pop_fs() {
        // pop fs → 0F A1
        assert_eq!(encode("pop", vec![Register(Fs)]), vec![0x0F, 0xA1]);
    }

    #[test]
    fn test_pop_gs() {
        // pop gs → 0F A9
        assert_eq!(encode("pop", vec![Register(Gs)]), vec![0x0F, 0xA9]);
    }

    // === xchg eax,eax → NOP (0x90 single-byte) ===

    #[test]
    fn test_xchg_eax_eax_is_nop() {
        // xchg eax, eax → 90 (the canonical NOP encoding)
        let bytes = encode("xchg", vec![Register(Eax), Register(Eax)]);
        assert_eq!(bytes, vec![0x90]);
    }

    #[test]
    fn test_xchg_rax_rax() {
        // xchg rax, rax → 48 90 (REX.W + NOP form)
        let bytes = encode("xchg", vec![Register(Rax), Register(Rax)]);
        assert_eq!(bytes, vec![0x48, 0x90]);
    }

    #[test]
    fn test_xchg_ax_ax() {
        // xchg ax, ax → 66 90 (16-bit xchg, operand-size prefix + NOP form)
        let bytes = encode("xchg", vec![Register(Ax), Register(Ax)]);
        assert_eq!(bytes, vec![0x66, 0x90]);
    }

    // === High-byte register encoding correctness ===

    #[test]
    fn test_mov_ah_imm8() {
        // mov ah, 0x42 → B4 42 (B0+4=B4 for AH)
        let bytes = encode("mov", vec![Register(Ah), Immediate(0x42)]);
        assert_eq!(bytes, vec![0xB4, 0x42]);
    }

    #[test]
    fn test_mov_ch_imm8() {
        // mov ch, 0x11 → B5 11 (B0+5=B5 for CH)
        let bytes = encode("mov", vec![Register(Ch), Immediate(0x11)]);
        assert_eq!(bytes, vec![0xB5, 0x11]);
    }

    #[test]
    fn test_mov_dh_imm8() {
        // mov dh, 0x22 → B6 22 (B0+6=B6 for DH)
        let bytes = encode("mov", vec![Register(Dh), Immediate(0x22)]);
        assert_eq!(bytes, vec![0xB6, 0x22]);
    }

    #[test]
    fn test_mov_bh_imm8() {
        // mov bh, 0x33 → B7 33 (B0+7=B7 for BH)
        let bytes = encode("mov", vec![Register(Bh), Immediate(0x33)]);
        assert_eq!(bytes, vec![0xB7, 0x33]);
    }

    // === Shift memory operand with size_hint ===

    #[test]
    fn test_shl_byte_ptr_mem_1() {
        // shl byte ptr [rbx], 1 → D0 /4 with byte operand
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let bytes = encode_with_hint(
            "shl",
            vec![Memory(Box::new(mem)), Immediate(1)],
            OperandSize::Byte,
        );
        // Should encode as D0 23 (D0=shift byte by 1, ModRM /4 with [rbx])
        assert_eq!(bytes[0], 0xD0); // byte opcode, not D1 (dword)
    }

    #[test]
    fn test_shr_qword_ptr_mem_cl() {
        // shr qword ptr [rax], cl → REX.W D3 /5
        let mem = MemoryOperand {
            base: Some(Rax),
            ..Default::default()
        };
        let bytes = encode_with_hint(
            "shr",
            vec![Memory(Box::new(mem)), Register(Cl)],
            OperandSize::Qword,
        );
        assert_eq!(bytes[0], 0x48); // REX.W
        assert_eq!(bytes[1], 0xD3); // qword shift by cl
    }

    #[test]
    fn test_shl_word_ptr_mem_imm() {
        // shl word ptr [rcx], 4 → 66 C1 /4 imm8
        let mem = MemoryOperand {
            base: Some(Rcx),
            ..Default::default()
        };
        let bytes = encode_with_hint(
            "shl",
            vec![Memory(Box::new(mem)), Immediate(4)],
            OperandSize::Word,
        );
        assert_eq!(bytes[0], 0x66); // 16-bit prefix
        assert_eq!(bytes[1], 0xC1); // word shift by imm8
    }

    // === Push/Pop validation ===

    #[test]
    fn test_push_eax_rejected() {
        // push eax is invalid in 64-bit mode
        let instr = make_instr("push", vec![Register(Eax)]);
        let result = encode_instruction(&instr, Arch::X86_64);
        assert!(result.is_err());
    }

    #[test]
    fn test_pop_eax_rejected() {
        // pop eax is invalid in 64-bit mode
        let instr = make_instr("pop", vec![Register(Eax)]);
        let result = encode_instruction(&instr, Arch::X86_64);
        assert!(result.is_err());
    }

    #[test]
    fn test_push_al_rejected() {
        // push al is invalid
        let instr = make_instr("push", vec![Register(Al)]);
        let result = encode_instruction(&instr, Arch::X86_64);
        assert!(result.is_err());
    }

    #[test]
    fn test_push_cs_rejected() {
        // push cs invalid in 64-bit mode
        let instr = make_instr("push", vec![Register(Cs)]);
        let result = encode_instruction(&instr, Arch::X86_64);
        assert!(result.is_err());
    }

    #[test]
    fn test_pop_ds_rejected() {
        // pop ds invalid in 64-bit mode
        let instr = make_instr("pop", vec![Register(Ds)]);
        let result = encode_instruction(&instr, Arch::X86_64);
        assert!(result.is_err());
    }

    // === CMOVcc 8-bit rejection ===

    #[test]
    fn test_cmove_8bit_rejected() {
        let instr = make_instr("cmove", vec![Register(Al), Register(Bl)]);
        let result = encode_instruction(&instr, Arch::X86_64);
        assert!(result.is_err());
    }

    // === disp_label relocation propagation ===

    #[test]
    fn test_add_rax_mem_label_reloc() {
        // add rax, [my_data] — should produce relocation
        let mem = MemoryOperand {
            base: Some(Rip),
            disp_label: Some(alloc::string::String::from("my_data")),
            addr_mode: AddrMode::Offset,
            ..Default::default()
        };
        let instr = make_instr("add", vec![Register(Rax), Memory(Box::new(mem))]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert!(
            result.relocation.is_some(),
            "expected relocation for add rax, [my_data]"
        );
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "my_data");
        assert_eq!(reloc.kind, RelocKind::X86Relative);
    }

    #[test]
    fn test_cmp_mem_label_imm_reloc() {
        // cmp dword ptr [my_data], 0 — should produce relocation via centralized scan
        let mem = MemoryOperand {
            base: Some(Rip),
            disp_label: Some(alloc::string::String::from("counter")),
            addr_mode: AddrMode::Offset,
            ..Default::default()
        };
        let mut instr = make_instr("cmp", vec![Memory(Box::new(mem)), Immediate(0)]);
        instr.size_hint = Some(OperandSize::Dword);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert!(
            result.relocation.is_some(),
            "expected relocation for cmp [counter], 0"
        );
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "counter");
    }

    #[test]
    fn test_mov_mem_label_reg_reloc() {
        // mov [my_var], rax — should produce relocation
        let mem = MemoryOperand {
            base: Some(Rip),
            disp_label: Some(alloc::string::String::from("my_var")),
            addr_mode: AddrMode::Offset,
            ..Default::default()
        };
        let instr = make_instr("mov", vec![Memory(Box::new(mem)), Register(Rax)]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert!(
            result.relocation.is_some(),
            "expected relocation for mov [my_var], rax"
        );
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "my_var");
        assert_eq!(reloc.kind, RelocKind::X86Relative);
    }

    #[test]
    fn test_mov_mem_label_imm_reloc() {
        // mov dword ptr [flag], 1 — should produce relocation
        let mem = MemoryOperand {
            base: Some(Rip),
            disp_label: Some(alloc::string::String::from("flag")),
            addr_mode: AddrMode::Offset,
            ..Default::default()
        };
        let mut instr = make_instr("mov", vec![Memory(Box::new(mem)), Immediate(1)]);
        instr.size_hint = Some(OperandSize::Dword);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert!(
            result.relocation.is_some(),
            "expected relocation for mov [flag], 1"
        );
        let reloc = result.relocation.unwrap();
        assert_eq!(&*reloc.label, "flag");
        assert_eq!(reloc.kind, RelocKind::X86Relative);
    }

    #[test]
    fn test_test_mem_label_reloc() {
        // test dword ptr [status], 1 — should produce relocation via centralized scan
        let mem = MemoryOperand {
            base: Some(Rip),
            disp_label: Some(alloc::string::String::from("status")),
            addr_mode: AddrMode::Offset,
            ..Default::default()
        };
        let mut instr = make_instr("test", vec![Memory(Box::new(mem)), Immediate(1)]);
        instr.size_hint = Some(OperandSize::Dword);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert!(
            result.relocation.is_some(),
            "expected relocation for test [status], 1"
        );
    }

    // ── P0-1: Segment override prefix must come BEFORE REX/opcode ──

    #[test]
    fn test_segment_override_fs_prefix_position() {
        // mov rax, fs:[rbx] → 64 48 8B 03
        let mem = MemoryOperand {
            base: Some(Rbx),
            segment: Some(Fs),
            ..Default::default()
        };
        let bytes = encode("mov", vec![Register(Rax), Memory(Box::new(mem))]);
        // First byte must be 0x64 (FS override), not the REX prefix
        assert_eq!(bytes[0], 0x64, "FS segment override must be first byte");
        assert_eq!(bytes[1], 0x48, "REX.W must follow segment override");
        assert_eq!(bytes[2], 0x8B, "opcode must follow REX");
    }

    #[test]
    fn test_segment_override_gs_prefix_position() {
        // mov eax, gs:[rdx] → 65 8B 02
        let mem = MemoryOperand {
            base: Some(Rdx),
            segment: Some(Gs),
            ..Default::default()
        };
        let bytes = encode("mov", vec![Register(Eax), Memory(Box::new(mem))]);
        assert_eq!(bytes[0], 0x65, "GS segment override must be first byte");
        assert_eq!(bytes[1], 0x8B, "opcode must follow segment override");
    }

    #[test]
    fn test_segment_override_with_extended_reg() {
        // add rax, fs:[r12] → 64 49 03 04 24 (FS + REX.WB + opcode + SIB)
        let mem = MemoryOperand {
            base: Some(R12),
            segment: Some(Fs),
            ..Default::default()
        };
        let bytes = encode("add", vec![Register(Rax), Memory(Box::new(mem))]);
        assert_eq!(bytes[0], 0x64, "FS must precede REX");
        assert_eq!(bytes[1] & 0xF0, 0x40, "REX must follow segment override");
    }

    // ── P0-2: SIB index-only addressing ──

    #[test]
    fn test_sib_index_only_disp0() {
        // mov rax, [rsi*4] → 48 8B 04 B5 00 00 00 00
        // Must have mod=00, base=101, and 4 bytes of disp32=0
        let mem = MemoryOperand {
            index: Some(Rsi),
            scale: 4,
            ..Default::default()
        };
        let bytes = encode("mov", vec![Register(Rax), Memory(Box::new(mem))]);
        assert_eq!(bytes.len(), 8, "SIB index-only must include 4-byte disp32");
        assert_eq!(bytes[0], 0x48); // REX.W
        assert_eq!(bytes[1], 0x8B); // MOV
        assert_eq!(bytes[2] & 0xC7, 0x04); // mod=00, rm=100 (SIB)
                                           // SIB: scale=4 (log2=10), index=RSI (110), base=101 (no base)
        assert_eq!(bytes[3], 0xB5); // 10_110_101
        assert_eq!(&bytes[4..8], &[0, 0, 0, 0]); // disp32=0
    }

    #[test]
    fn test_sib_index_only_with_disp() {
        // lea rax, [rdi*8+16] → 48 8D 04 FD 10 00 00 00
        // NOT [rbp + rdi*8 + 16]
        let mem = MemoryOperand {
            index: Some(Rdi),
            scale: 8,
            disp: 16,
            ..Default::default()
        };
        let bytes = encode("lea", vec![Register(Rax), Memory(Box::new(mem))]);
        assert_eq!(bytes.len(), 8, "SIB index-only with disp must use disp32");
        assert_eq!(bytes[2] & 0xC0, 0x00, "mod must be 00 (disp32, no base)");
        // disp32 should be 16
        assert_eq!(&bytes[4..8], &(16i32).to_le_bytes());
    }

    #[test]
    fn test_sib_index_only_extended_index() {
        // mov eax, [r9*2] → 42 8B 04 4D 00 00 00 00
        let mem = MemoryOperand {
            index: Some(R9),
            scale: 2,
            ..Default::default()
        };
        let bytes = encode("mov", vec![Register(Eax), Memory(Box::new(mem))]);
        assert_eq!(bytes.len(), 8, "SIB index-only with extended index");
        assert_eq!(bytes[0] & 0x42, 0x42, "REX.X must be set for R9 index");
    }

    // ── P2: 8-bit operand validation ──

    #[test]
    fn test_bt_rejects_8bit() {
        let instr = make_instr("bt", vec![Register(Al), Immediate(1)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    #[test]
    fn test_bsf_rejects_8bit() {
        let instr = make_instr("bsf", vec![Register(Al), Register(Cl)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    #[test]
    fn test_bsr_rejects_8bit() {
        let instr = make_instr("bsr", vec![Register(Al), Register(Cl)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    #[test]
    fn test_popcnt_rejects_8bit() {
        let instr = make_instr("popcnt", vec![Register(Al), Register(Cl)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    #[test]
    fn test_lzcnt_rejects_8bit() {
        let instr = make_instr("lzcnt", vec![Register(Al), Register(Cl)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    #[test]
    fn test_tzcnt_rejects_8bit() {
        let instr = make_instr("tzcnt", vec![Register(Al), Register(Cl)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    #[test]
    fn test_bswap_rejects_8bit() {
        let instr = make_instr("bswap", vec![Register(Al)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    #[test]
    fn test_bswap_rejects_16bit() {
        let instr = make_instr("bswap", vec![Register(Ax)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    // ── P3-1: No redundant REX.W on push/pop/jmp/call [mem] ──

    #[test]
    fn test_push_mem_no_redundant_rex() {
        // push [rdi] → FF 37 (not 48 FF 37)
        let mem = MemoryOperand {
            base: Some(Rdi),
            ..Default::default()
        };
        let bytes = encode("push", vec![Memory(Box::new(mem))]);
        assert_eq!(bytes, &[0xFF, 0x37], "push [rdi] should not have REX.W");
    }

    #[test]
    fn test_pop_mem_no_redundant_rex() {
        // pop [rdi] → 8F 07 (not 48 8F 07)
        let mem = MemoryOperand {
            base: Some(Rdi),
            ..Default::default()
        };
        let bytes = encode("pop", vec![Memory(Box::new(mem))]);
        assert_eq!(bytes, &[0x8F, 0x07], "pop [rdi] should not have REX.W");
    }

    #[test]
    fn test_jmp_mem_no_redundant_rex() {
        // jmp [rdi] → FF 27 (not 48 FF 27)
        let mem = MemoryOperand {
            base: Some(Rdi),
            ..Default::default()
        };
        let bytes = encode("jmp", vec![Memory(Box::new(mem))]);
        assert_eq!(bytes, &[0xFF, 0x27], "jmp [rdi] should not have REX.W");
    }

    #[test]
    fn test_call_mem_no_redundant_rex() {
        // call [rdi] → FF 17 (not 48 FF 17)
        let mem = MemoryOperand {
            base: Some(Rdi),
            ..Default::default()
        };
        let bytes = encode("call", vec![Memory(Box::new(mem))]);
        assert_eq!(bytes, &[0xFF, 0x17], "call [rdi] should not have REX.W");
    }

    #[test]
    fn test_push_mem_extended_needs_rex() {
        // push [r15] → 41 FF 37 (REX.B for r15, but not REX.W)
        let mem = MemoryOperand {
            base: Some(R15),
            ..Default::default()
        };
        let bytes = encode("push", vec![Memory(Box::new(mem))]);
        assert_eq!(bytes[0], 0x41, "push [r15] needs REX.B");
        assert_eq!(bytes[1], 0xFF);
    }

    // ── P3-2: xchg short form both directions ──

    #[test]
    fn test_xchg_rbx_rax_short_form() {
        // xchg rbx, rax → 48 93 (same as xchg rax, rbx)
        let bytes = encode("xchg", vec![Register(Rbx), Register(Rax)]);
        assert_eq!(bytes, &[0x48, 0x93], "xchg rbx, rax should use short form");
    }

    #[test]
    fn test_xchg_ecx_eax_short_form() {
        // xchg ecx, eax → 91 (same as xchg eax, ecx)
        let bytes = encode("xchg", vec![Register(Ecx), Register(Eax)]);
        assert_eq!(bytes, &[0x91], "xchg ecx, eax should use short form");
    }

    // ── P1-2: ALU mem,imm reloc uses explicit disp offset (not heuristic) ──

    #[test]
    fn test_alu_mem_imm_reloc_explicit() {
        // add dword ptr [label], 5 — relocation must point to displacement, not imm
        let mem = MemoryOperand {
            base: Some(Rip),
            disp_label: Some(alloc::string::String::from("data")),
            addr_mode: AddrMode::Offset,
            ..Default::default()
        };
        let mut instr = make_instr("add", vec![Memory(Box::new(mem)), Immediate(5)]);
        instr.size_hint = Some(OperandSize::Dword);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        assert!(
            result.relocation.is_some(),
            "expected relocation for add [label], imm"
        );
        // The relocation offset should point inside the instruction, not at the trailing imm8
        let roff = result.relocation.as_ref().unwrap().offset;
        // For add [rip+disp32], imm8: opcode(1) + modrm(1) → disp32 at offset 2
        assert_eq!(roff, 2, "reloc should point at displacement, not immediate");
    }

    // ── P1-3: No double segment override from builder API ──

    #[test]
    fn test_no_double_segment_override() {
        // Builder constructs: prefixes=[SegFs] AND mem.segment=Some(Fs)
        let mem = MemoryOperand {
            base: Some(Rbx),
            segment: Some(Fs),
            ..Default::default()
        };
        let mut instr = make_instr("mov", vec![Register(Rax), Memory(Box::new(mem))]);
        instr.prefixes.push(Prefix::SegFs);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // Should only have ONE 0x64 byte, not two
        let count_64 = result.bytes.iter().filter(|&&b| b == 0x64).count();
        assert_eq!(count_64, 1, "should not emit double FS segment override");
    }

    #[test]
    fn test_rip_relative_trailing_bytes_mov_mem_imm() {
        // mov dword ptr [rip+label], 42  →  C7 05 [disp32] [imm32]
        // The reloc should have trailing_bytes = 4 (the imm32 after disp32)
        let mem = MemoryOperand {
            base: Some(crate::ir::Register::Rip),
            index: None,
            scale: 1,
            disp: 0,
            size: Some(OperandSize::Dword),
            segment: None,
            disp_label: Some(String::from("data")),
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        let instr = make_instr_with_hint(
            "mov",
            vec![Memory(Box::new(mem)), Immediate(42)],
            Some(OperandSize::Dword),
        );
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // Expected: C7 05 00 00 00 00 2A 00 00 00
        assert_eq!(result.bytes[0], 0xC7); // opcode
        assert_eq!(result.bytes[1], 0x05); // modrm: mod=00, /0, rm=101 (RIP)
        let reloc = result.relocation.as_ref().unwrap();
        assert_eq!(reloc.offset, 2); // disp32 starts at byte 2
        assert_eq!(reloc.size, 4); // 4-byte disp32
        assert_eq!(reloc.kind, RelocKind::X86Relative);
        assert_eq!(reloc.trailing_bytes, 4); // 4 bytes of imm32 follow
        assert_eq!(result.bytes.len(), 10); // total: opcode(1) + modrm(1) + disp32(4) + imm32(4) = 10
    }

    #[test]
    fn test_rip_relative_trailing_bytes_mov_mem_reg() {
        // mov [rip+label], rax  →  48 89 05 [disp32]
        // The reloc should have trailing_bytes = 0 (nothing after disp32)
        let mem = MemoryOperand {
            base: Some(crate::ir::Register::Rip),
            index: None,
            scale: 1,
            disp: 0,
            size: None,
            segment: None,
            disp_label: Some(String::from("data")),
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        let instr = make_instr(
            "mov",
            vec![
                Memory(Box::new(mem)),
                Operand::Register(crate::ir::Register::Rax),
            ],
        );
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        let reloc = result.relocation.as_ref().unwrap();
        assert_eq!(reloc.kind, RelocKind::X86Relative);
        assert_eq!(reloc.trailing_bytes, 0); // disp32 is at the end of instruction
    }

    #[test]
    fn test_rip_relative_trailing_bytes_alu_mem_imm8() {
        // add dword ptr [rip+label], 5  →  83 05 [disp32] 05
        // trailing_bytes = 1 (the imm8 after disp32)
        let mem = MemoryOperand {
            base: Some(crate::ir::Register::Rip),
            index: None,
            scale: 1,
            disp: 0,
            size: Some(OperandSize::Dword),
            segment: None,
            disp_label: Some(String::from("target")),
            addr_mode: AddrMode::Offset,
            index_subtract: false,
        };
        let instr = make_instr_with_hint(
            "add",
            vec![Memory(Box::new(mem)), Immediate(5)],
            Some(OperandSize::Dword),
        );
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        let reloc = result.relocation.as_ref().unwrap();
        assert_eq!(reloc.kind, RelocKind::X86Relative);
        assert_eq!(reloc.trailing_bytes, 1); // 1 byte of imm8 follows
    }

    #[test]
    fn test_rip_relative_trailing_bytes_jmp() {
        // jmp label  →  E9 [disp32]
        // trailing_bytes = 0
        let instr = make_instr("jmp", vec![Label(String::from("target"))]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        let reloc = result.relocation.as_ref().unwrap();
        assert_eq!(reloc.kind, RelocKind::X86Relative);
        assert_eq!(reloc.trailing_bytes, 0);
    }

    // === 8th Audit: Push imm range validation ===

    #[test]
    fn test_push_imm8_short_form() {
        // push 0x42 → 6A 42
        assert_eq!(encode("push", vec![Immediate(0x42)]), vec![0x6A, 0x42]);
    }

    #[test]
    fn test_push_imm32_full_form() {
        assert_eq!(
            encode("push", vec![Immediate(0x12345678)]),
            vec![0x68, 0x78, 0x56, 0x34, 0x12]
        );
    }

    #[test]
    fn test_push_imm_out_of_range_rejects() {
        // push 0x1_0000_0000 → should error (doesn't fit imm32)
        let instr = make_instr("push", vec![Immediate(0x1_0000_0000)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    // === 8th Audit: IMUL 8-bit rejection ===

    #[test]
    fn test_imul_2op_rejects_8bit() {
        let instr = make_instr("imul", vec![Register(Al), Register(Bl)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    #[test]
    fn test_imul_2op_mem_rejects_8bit() {
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let instr = make_instr("imul", vec![Register(Al), Memory(Box::new(mem))]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    #[test]
    fn test_imul_3op_rejects_8bit() {
        let instr = make_instr("imul", vec![Register(Al), Register(Bl), Immediate(5)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    #[test]
    fn test_imul_3op_mem_rejects_8bit() {
        let mem = MemoryOperand {
            base: Some(Rcx),
            ..Default::default()
        };
        let instr = make_instr(
            "imul",
            vec![Register(Al), Memory(Box::new(mem)), Immediate(5)],
        );
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    // === 8th Audit: CMOVcc 8-bit rejection (reg, mem path) ===

    #[test]
    fn test_cmovcc_reg_mem_rejects_8bit() {
        let mem = MemoryOperand {
            base: Some(Rbx),
            ..Default::default()
        };
        let instr = make_instr("cmove", vec![Register(Al), Memory(Box::new(mem))]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    // === 8th Audit: SETcc rejects non-8-bit registers ===

    #[test]
    fn test_setcc_rejects_32bit_register() {
        let instr = make_instr("sete", vec![Register(Eax)]);
        assert!(encode_instruction(&instr, Arch::X86_64).is_err());
    }

    #[test]
    fn test_setcc_accepts_8bit_register() {
        // sete al → 0F 94 C0
        assert_eq!(encode("sete", vec![Register(Al)]), vec![0x0F, 0x94, 0xC0]);
    }

    // === 8th Audit: movzx/movsx mem.size fallback ===

    #[test]
    fn test_movzx_mem_word_source_via_mem_size() {
        // movzx eax, word ptr [rbx] via mem.size (builder API path)
        let mem = MemoryOperand {
            base: Some(Rbx),
            size: Some(OperandSize::Word),
            ..Default::default()
        };
        // No size_hint on instruction — should use mem.size
        let instr = make_instr("movzx", vec![Register(Eax), Memory(Box::new(mem))]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // movzx eax, word [rbx] → 0F B7 03  (B7 = word source, not B6 = byte)
        assert_eq!(result.bytes, vec![0x0F, 0xB7, 0x03]);
    }

    #[test]
    fn test_movsx_mem_word_source_via_mem_size() {
        // movsx eax, word ptr [rbx] via mem.size (builder API path)
        let mem = MemoryOperand {
            base: Some(Rbx),
            size: Some(OperandSize::Word),
            ..Default::default()
        };
        let instr = make_instr("movsx", vec![Register(Eax), Memory(Box::new(mem))]);
        let result = encode_instruction(&instr, Arch::X86_64).unwrap();
        // movsx eax, word [rbx] → 0F BF 03  (BF = word source, not BE = byte)
        assert_eq!(result.bytes, vec![0x0F, 0xBF, 0x03]);
    }

    // ─── 16-bit mode (encode_instruction_16) tests ───────────────

    #[test]
    fn test_16bit_mov_ax_imm16() {
        // In 16-bit mode, mov ax, 0x1234 → B8 34 12 (no 0x66 prefix)
        // The 32-bit encoder produces 66 B8 34 12, toggle removes 0x66.
        let instr = make_instr("mov", vec![Register(Ax), Immediate(0x1234)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0xB8, 0x34, 0x12]);
    }

    #[test]
    fn test_16bit_mov_eax_imm32() {
        // In 16-bit mode, mov eax, 0x12345678 → 66 B8 78 56 34 12
        // The 32-bit encoder produces B8 78 56 34 12, toggle adds 0x66.
        let instr = make_instr("mov", vec![Register(Eax), Immediate(0x1234_5678)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x66, 0xB8, 0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_16bit_xor_ax_ax() {
        // xor ax, ax → 31 C0 (no prefix in 16-bit mode)
        let instr = make_instr("xor", vec![Register(Ax), Register(Ax)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x31, 0xC0]);
    }

    #[test]
    fn test_16bit_xor_eax_eax() {
        // xor eax, eax → 66 31 C0 (prefix needed for 32-bit in 16-bit mode)
        let instr = make_instr("xor", vec![Register(Eax), Register(Eax)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x66, 0x31, 0xC0]);
    }

    #[test]
    fn test_16bit_push_ax() {
        // push ax → 50 (no prefix in 16-bit mode)
        let instr = make_instr("push", vec![Register(Ax)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x50]);
    }

    #[test]
    fn test_16bit_push_eax() {
        // push eax → 66 50 (prefix needed for 32-bit in 16-bit mode)
        let instr = make_instr("push", vec![Register(Eax)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x66, 0x50]);
    }

    #[test]
    fn test_16bit_pop_bx() {
        // pop bx → 5B (no prefix in 16-bit mode)
        let instr = make_instr("pop", vec![Register(Bx)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x5B]);
    }

    #[test]
    fn test_16bit_pop_ebx() {
        // pop ebx → 66 5B (prefix needed for 32-bit)
        let instr = make_instr("pop", vec![Register(Ebx)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x66, 0x5B]);
    }

    #[test]
    fn test_16bit_inc_cx() {
        // inc cx → 41 (short form, no prefix)
        let instr = make_instr("inc", vec![Register(Cx)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x41]);
    }

    #[test]
    fn test_16bit_inc_ecx() {
        // inc ecx → 66 41 (prefix needed for 32-bit)
        let instr = make_instr("inc", vec![Register(Ecx)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x66, 0x41]);
    }

    #[test]
    fn test_16bit_dec_dx() {
        // dec dx → 4A (short form, no prefix)
        let instr = make_instr("dec", vec![Register(Dx)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x4A]);
    }

    #[test]
    fn test_16bit_nop() {
        // nop → 90 (identical in all modes)
        let instr = make_instr("nop", vec![]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x90]);
    }

    #[test]
    fn test_16bit_cli() {
        // cli → FA (identical in all modes)
        let instr = make_instr("cli", vec![]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0xFA]);
    }

    #[test]
    fn test_16bit_int_10h() {
        // int 0x10 → CD 10 (identical in all modes)
        let instr = make_instr("int", vec![Immediate(0x10)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0xCD, 0x10]);
    }

    #[test]
    fn test_16bit_push_es() {
        // push es → 06 (segment push valid in 16-bit mode, no prefix toggle)
        let instr = make_instr("push", vec![Register(Es)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x06]);
    }

    #[test]
    fn test_16bit_push_cs() {
        // push cs → 0E
        let instr = make_instr("push", vec![Register(Cs)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x0E]);
    }

    #[test]
    fn test_16bit_pop_ds() {
        // pop ds → 1F
        let instr = make_instr("pop", vec![Register(Ds)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x1F]);
    }

    #[test]
    fn test_16bit_push_imm8() {
        // push 0x42 → 6A 42 (imm8 encoding, same in all modes)
        let instr = make_instr("push", vec![Immediate(0x42)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x6A, 0x42]);
    }

    #[test]
    fn test_16bit_add_ax_bx() {
        // add ax, bx → 01 D8 (no prefix in 16-bit mode)
        let instr = make_instr("add", vec![Register(Ax), Register(Bx)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0x01, 0xD8]);
    }

    #[test]
    fn test_16bit_mov_al_imm8() {
        // mov al, 0x42 → B0 42 (8-bit, no operand-size prefix in any mode)
        let instr = make_instr("mov", vec![Register(Al), Immediate(0x42)]);
        let result = encode_instruction_16(&instr).unwrap();
        assert_eq!(result.bytes, vec![0xB0, 0x42]);
    }

    #[test]
    fn test_16bit_rejects_64bit_register() {
        // 64-bit registers are invalid in 16-bit mode
        let instr = make_instr("mov", vec![Register(Rax), Immediate(1)]);
        assert!(encode_instruction_16(&instr).is_err());
    }

    // ===================================================================
    // AVX-512 (EVEX-encoded) instruction tests
    // ===================================================================

    // ── EVEX prefix helper unit tests ────────────────────────────────

    #[test]
    fn test_emit_evex_basic_prefix() {
        // Test the raw EVEX prefix bytes for known inputs
        let mut buf = InstrBytes::new();
        // Simulate VADDPS zmm0, zmm1, zmm2:
        // R=false(zmm0 not ext), X=false, B=false(zmm2 not ext),
        // R'=false(zmm0 bit4=0), mm=1, W=false, vvvv=1(zmm1), V'=false, pp=0,
        // z=false, ll=2(512), b=false, aaa=0
        emit_evex(
            &mut buf, false, false, false, false, 1, false, 1, false, 0, false, 2, false, 0,
        );
        assert_eq!(buf[0], 0x62, "EVEX escape byte");
        // P0: ~R=1 ~X=1 ~B=1 ~R'=1 0 0 01 = 0xF1
        assert_eq!(buf[1], 0xF1, "P0");
        // P1: W=0 ~vvvv=~0001=1110 1 pp=00 = 0x74
        assert_eq!(buf[2], 0x74, "P1");
        // P2: z=0 L'L=10 b=0 ~V'=1 aaa=000 = 0x48
        assert_eq!(buf[3], 0x48, "P2");
    }

    #[test]
    fn test_emit_evex_w_bit() {
        let mut buf = InstrBytes::new();
        // W=1, everything else zero-ish, mm=1
        emit_evex(
            &mut buf, false, false, false, false, 1, true, 0, false, 0, false, 2, false, 0,
        );
        // P1 should have W=1: 0x80 | ~0000<<3=0x78 | 0x04 | pp=0 = 0xFC
        assert_eq!(buf[2], 0xFC, "P1 with W=1");
    }

    #[test]
    fn test_emit_evex_extended_reg() {
        let mut buf = InstrBytes::new();
        // R=true (extended), R'=true (evex extended) for zmm28 etc.
        emit_evex(
            &mut buf, true, false, false, true, 1, false, 0, false, 0, false, 2, false, 0,
        );
        // P0: ~R=0 ~X=1 ~B=1 ~R'=0 00 01 = 0x61
        assert_eq!(buf[1], 0x61, "P0 with R,R' extended");
    }

    // ── EVEX evex_ll tests ──────────────────────────────────────────

    #[test]
    fn test_evex_ll_zmm() {
        assert_eq!(evex_ll(Zmm0), 2);
        assert_eq!(evex_ll(Zmm31), 2);
    }

    #[test]
    fn test_evex_ll_ymm() {
        assert_eq!(evex_ll(Ymm0), 1);
    }

    #[test]
    fn test_evex_ll_xmm() {
        assert_eq!(evex_ll(Xmm0), 0);
    }

    // ── EVEX full instruction encoding ──────────────────────────────

    #[test]
    fn test_evex_vaddps_zmm0_zmm1_zmm2() {
        // VADDPS zmm0, zmm1, zmm2 → 62 F1 74 48 58 C2
        let bytes = encode(
            "vaddps",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0x74, 0x48, 0x58, 0xC2]);
    }

    #[test]
    fn test_evex_vaddpd_zmm0_zmm1_zmm2() {
        // VADDPD zmm0, zmm1, zmm2 → 62 F1 F5 48 58 C2
        // W=1 → P1 = 0x80 | 0x74 = 0xF4, but vvvv=1 so ~vvvv=1110,
        // P1 = 0x80 | (0x0E << 3) | 0x04 | 0x01 = 0x80|0x70|0x04|0x01 = 0xF5
        let bytes = encode(
            "vaddpd",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0xF5, 0x48, 0x58, 0xC2]);
    }

    #[test]
    fn test_evex_vsubps_zmm3_zmm4_zmm5() {
        // VSUBPS zmm3, zmm4, zmm5
        // dst=zmm3(code=3), src1=zmm4(code=4), src2=zmm5(code=5)
        // P0: all not ext → 0xF1
        // P1: W=0, vvvv=4 → ~4=~0100=1011 → 01011, 1, 00 → 0x5C...
        // ~vvvv = 0x0B, P1 = (0x0B << 3) | 0x04 | 0x00 = 0x58|0x04 = 0x5C
        // P2: z=0, L'L=10, b=0, ~V'=1, aaa=0 → 0x48
        // opcode=0x5C, ModRM: 0xC0|(3<<3)|5 = 0xDD
        let bytes = encode(
            "vsubps",
            vec![Register(Zmm3), Register(Zmm4), Register(Zmm5)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0x5C, 0x48, 0x5C, 0xDD]);
    }

    #[test]
    fn test_evex_vmulps_zmm0_zmm1_zmm2() {
        let bytes = encode(
            "vmulps",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0x74, 0x48, 0x59, 0xC2]);
    }

    #[test]
    fn test_evex_vdivps_zmm0_zmm1_zmm2() {
        let bytes = encode(
            "vdivps",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0x74, 0x48, 0x5E, 0xC2]);
    }

    #[test]
    fn test_evex_vmovaps_zmm0_zmm1() {
        // VMOVAPS zmm0, zmm1 (2-op, vvvv=0)
        // P1: W=0, vvvv=0 → ~0=1111, P1 = (0x0F<<3)|0x04|0x00 = 0x78|0x04 = 0x7C
        // P2: z=0, L'L=10, b=0, ~V'=1, aaa=0 → 0x48
        // opcode=0x28, ModRM: 0xC0|(0<<3)|1 = 0xC1
        let bytes = encode("vmovaps", vec![Register(Zmm0), Register(Zmm1)]);
        assert_eq!(bytes, vec![0x62, 0xF1, 0x7C, 0x48, 0x28, 0xC1]);
    }

    #[test]
    fn test_evex_vmovdqa32_zmm0_zmm1() {
        // VMOVDQA32 zmm0, zmm1 → pp=0x66(→1), W=0
        // P1: W=0, vvvv=0, pp=0x01 → 0x7C|0x01 = 0x7D
        let bytes = encode("vmovdqa32", vec![Register(Zmm0), Register(Zmm1)]);
        assert_eq!(bytes, vec![0x62, 0xF1, 0x7D, 0x48, 0x6F, 0xC1]);
    }

    #[test]
    fn test_evex_vmovdqa64_zmm0_zmm1() {
        // VMOVDQA64 zmm0, zmm1 → pp=0x66(→1), W=1
        // P1: W=1, vvvv=0, pp=0x01 → 0xFC|0x01 = 0xFD
        let bytes = encode("vmovdqa64", vec![Register(Zmm0), Register(Zmm1)]);
        assert_eq!(bytes, vec![0x62, 0xF1, 0xFD, 0x48, 0x6F, 0xC1]);
    }

    #[test]
    fn test_evex_vpternlogd_zmm0_zmm1_zmm2_imm() {
        // VPTERNLOGD zmm0, zmm1, zmm2, 0xFF → EVEX map3, opcode=0x25, W=0
        // P0: mm=3 → 0xF3
        // P1: W=0, vvvv=1 → 0x74|0x01 = 0x71... wait pp=0x66→1
        // P1: (0x0E<<3)|0x04|0x01 = 0x70|0x04|0x01 = 0x75
        let bytes = encode(
            "vpternlogd",
            vec![
                Register(Zmm0),
                Register(Zmm1),
                Register(Zmm2),
                Immediate(0xFF),
            ],
        );
        assert_eq!(bytes, vec![0x62, 0xF3, 0x75, 0x48, 0x25, 0xC2, 0xFF]);
    }

    #[test]
    fn test_evex_vpternlogq_zmm0_zmm1_zmm2_imm() {
        // VPTERNLOGQ zmm0, zmm1, zmm2, 0xDB → W=1
        let bytes = encode(
            "vpternlogq",
            vec![
                Register(Zmm0),
                Register(Zmm1),
                Register(Zmm2),
                Immediate(0xDB),
            ],
        );
        assert_eq!(bytes, vec![0x62, 0xF3, 0xF5, 0x48, 0x25, 0xC2, 0xDB]);
    }

    #[test]
    fn test_evex_vpaddd_zmm0_zmm1_zmm2() {
        // VPADDD zmm0, zmm1, zmm2 → map1, pp=66, W=0, opcode=0xFE
        let bytes = encode(
            "vpaddd",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0x75, 0x48, 0xFE, 0xC2]);
    }

    #[test]
    fn test_evex_vpaddq_zmm0_zmm1_zmm2() {
        // VPADDQ zmm0, zmm1, zmm2 → map1, pp=66, W=1, opcode=0xD4
        let bytes = encode(
            "vpaddq",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0xF5, 0x48, 0xD4, 0xC2]);
    }

    #[test]
    fn test_evex_vpxord_zmm0_zmm1_zmm2() {
        let bytes = encode(
            "vpxord",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0x75, 0x48, 0xEF, 0xC2]);
    }

    #[test]
    fn test_evex_vpxorq_zmm0_zmm1_zmm2() {
        let bytes = encode(
            "vpxorq",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0xF5, 0x48, 0xEF, 0xC2]);
    }

    #[test]
    fn test_evex_vblendmps_zmm0_zmm1_zmm2() {
        // VBLENDMPS zmm0, zmm1, zmm2 → map2(0F38), pp=66, W=0, opcode=0x65
        let bytes = encode(
            "vblendmps",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF2, 0x75, 0x48, 0x65, 0xC2]);
    }

    #[test]
    fn test_evex_vpmullq_zmm0_zmm1_zmm2() {
        // VPMULLQ zmm0, zmm1, zmm2 → map2(0F38), pp=66, W=1, opcode=0x40
        let bytes = encode(
            "vpmullq",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF2, 0xF5, 0x48, 0x40, 0xC2]);
    }

    // ── Extended registers (ZMM16+) ─────────────────────────────────

    #[test]
    fn test_evex_vaddps_zmm16_zmm17_zmm18() {
        // ZMM16: index=16, base_code=0, is_ext=false(bit3=0), is_evex_ext=true(bit4=1)
        // ZMM17: index=17, base_code=1, is_ext=false(bit3=0), is_evex_ext=true(bit4=1)
        // ZMM18: index=18, base_code=2, is_ext=false(bit3=0), is_evex_ext=true(bit4=1)
        // dst=zmm16: R=ext=false, R'=evex=true
        // src1=zmm17: vvvv=1, V'=evex=true
        // src2=zmm18: B=ext=false, X=evex=true
        // P0: ~R=1(0x80), ~X=0, ~B=1(0x20), ~R'=0, mm=01 → 0xA1
        // P1: W=0, vvvv=1 → ~1=1110 → (0xE<<3)|0x04|0x00 = 0x74
        // P2: z=0, L'L=10, b=0, ~V'=0, aaa=0 → 0x40
        let bytes = encode(
            "vaddps",
            vec![Register(Zmm16), Register(Zmm17), Register(Zmm18)],
        );
        assert_eq!(bytes, vec![0x62, 0xA1, 0x74, 0x40, 0x58, 0xC2]);
    }

    #[test]
    fn test_evex_vmovaps_zmm31_zmm16() {
        // ZMM31: base_code=7, is_ext=true, is_evex_ext=true
        // ZMM16: base_code=0, is_ext=false, is_evex_ext=true
        // 2-op form: vvvv=0, V'=false
        // dst=zmm31: R=ext=true, R'=evex=true
        // src=zmm16: B=ext=false, X=evex=true
        // P0: ~R=0, ~X=0, ~B=1(0x20), ~R'=0, mm=01 → 0x21
        // P1: W=0, vvvv=0, pp=0 → (0x0F<<3)|0x04 = 0x7C
        // P2: z=0, L'L=10, b=0, ~V'=1, aaa=0 → 0x48
        // ModRM: 0xC0|(7<<3)|0 = 0xF8
        let bytes = encode("vmovaps", vec![Register(Zmm31), Register(Zmm16)]);
        assert_eq!(bytes, vec![0x62, 0x21, 0x7C, 0x48, 0x28, 0xF8]);
    }

    // ── EVEX-only instructions (no VEX equivalent) ──────────────────

    #[test]
    fn test_evex_vmovdqu8_zmm0_zmm1() {
        // VMOVDQU8 zmm0, zmm1 → pp=F2(→3), W=0, map1, opcode=0x6F
        // P1: (0x0F<<3)|0x04|0x03 = 0x7C|0x03 = 0x7F
        let bytes = encode("vmovdqu8", vec![Register(Zmm0), Register(Zmm1)]);
        assert_eq!(bytes, vec![0x62, 0xF1, 0x7F, 0x48, 0x6F, 0xC1]);
    }

    #[test]
    fn test_evex_vmovdqu16_zmm0_zmm1() {
        // VMOVDQU16 zmm0, zmm1 → pp=F2(→3), W=1
        // P1: 0x80|(0x0F<<3)|0x04|0x03 = 0xFC|0x03 = 0xFF
        let bytes = encode("vmovdqu16", vec![Register(Zmm0), Register(Zmm1)]);
        assert_eq!(bytes, vec![0x62, 0xF1, 0xFF, 0x48, 0x6F, 0xC1]);
    }

    #[test]
    fn test_evex_vpsravq_zmm0_zmm1_zmm2() {
        // VPSRAVQ zmm0, zmm1, zmm2 → map2, pp=66, W=1, opcode=0x46
        let bytes = encode(
            "vpsravq",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF2, 0xF5, 0x48, 0x46, 0xC2]);
    }

    #[test]
    fn test_evex_vpandd_zmm0_zmm1_zmm2() {
        let bytes = encode(
            "vpandd",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0x75, 0x48, 0xDB, 0xC2]);
    }

    #[test]
    fn test_evex_vpord_zmm0_zmm1_zmm2() {
        let bytes = encode(
            "vpord",
            vec![Register(Zmm0), Register(Zmm1), Register(Zmm2)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0x75, 0x48, 0xEB, 0xC2]);
    }

    // ── VEX instructions still work (regression) ────────────────────

    #[test]
    fn test_vex_vaddps_xmm_still_works() {
        // VADDPS xmm0, xmm1, xmm2 should use VEX, not EVEX
        let bytes = encode(
            "vaddps",
            vec![Register(Xmm0), Register(Xmm1), Register(Xmm2)],
        );
        // VEX.128.0F 58 → C5 F0 58 C2 (2-byte VEX)
        assert_eq!(bytes[0], 0xC5, "should be 2-byte VEX prefix");
    }

    #[test]
    fn test_vex_vaddps_ymm_still_works() {
        // VADDPS ymm0, ymm1, ymm2 should use VEX, not EVEX
        let bytes = encode(
            "vaddps",
            vec![Register(Ymm0), Register(Ymm1), Register(Ymm2)],
        );
        // VEX.256.0F 58 → C5 F4 58 C2
        assert_eq!(bytes[0], 0xC5, "should be 2-byte VEX prefix for ymm");
    }

    // ── Compress/expand (EVEX-only) ─────────────────────────────────

    #[test]
    fn test_evex_vcompressps_zmm0_zmm1() {
        // VCOMPRESSPS zmm0, zmm1 → map2, pp=66, W=0, opcode=0x8A
        let bytes = encode("vcompressps", vec![Register(Zmm0), Register(Zmm1)]);
        assert_eq!(bytes, vec![0x62, 0xF2, 0x7D, 0x48, 0x8A, 0xC1]);
    }

    #[test]
    fn test_evex_vexpandps_zmm0_zmm1() {
        // VEXPANDPS zmm0, zmm1 → map2, pp=66, W=0, opcode=0x88
        let bytes = encode("vexpandps", vec![Register(Zmm0), Register(Zmm1)]);
        assert_eq!(bytes, vec![0x62, 0xF2, 0x7D, 0x48, 0x88, 0xC1]);
    }

    // ── EVEX shuffle / pshufd with imm ──────────────────────────────

    #[test]
    fn test_evex_vpshufd_zmm0_zmm1_imm() {
        // VPSHUFD zmm0, zmm1, 0xE4 → map1, pp=66, W=0, opcode=0x70
        // 3-op imm form: dst=zmm0, src=zmm1, imm=0xE4
        let bytes = encode(
            "vpshufd",
            vec![Register(Zmm0), Register(Zmm1), Immediate(0xE4)],
        );
        assert_eq!(bytes, vec![0x62, 0xF1, 0x7D, 0x48, 0x70, 0xC1, 0xE4]);
    }
}
