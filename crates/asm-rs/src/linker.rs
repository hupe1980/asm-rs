//! Label resolution, branch relaxation, and final layout.
//!
//! The linker collects fragments (fixed code/data, alignment padding, and
//! relaxable branches), resolves labels, performs Szymanski-style branch
//! relaxation (monotonic growth, guaranteed convergence), applies
//! relocations, and emits the final machine code.

use alloc::collections::BTreeMap;
#[allow(unused_imports)]
use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
#[allow(unused_imports)]
use alloc::vec;
use alloc::vec::Vec;

use crate::encoder::{InstrBytes, RelaxInfo, RelocKind, Relocation};
use crate::error::{AsmError, Span};

// ─── FragmentBytes ─────────────────────────────────────────

/// Compact byte storage for fragment payloads.
///
/// Instructions (≤15 bytes) are stored inline as [`InstrBytes`] — zero heap
/// allocations on the hot encoding path.  Data directives that may exceed
/// 15 bytes fall back to a heap-allocated `Vec<u8>`.
#[derive(Debug, Clone)]
pub enum FragmentBytes {
    /// Inline storage (≤15 bytes) — no heap allocation.
    Inline(InstrBytes),
    /// Heap-allocated storage for data larger than 15 bytes.
    Heap(Vec<u8>),
}

impl core::ops::Deref for FragmentBytes {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        match self {
            FragmentBytes::Inline(ib) => ib,
            FragmentBytes::Heap(v) => v,
        }
    }
}

impl core::ops::DerefMut for FragmentBytes {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        match self {
            FragmentBytes::Inline(ib) => ib,
            FragmentBytes::Heap(v) => v,
        }
    }
}

/// Maximum number of relaxation iterations before giving up.
const MAX_RELAXATION_ITERS: usize = 100;

/// Read a little-endian u32 from `bytes` at `offset`, with bounds checking.
/// Returns a descriptive error instead of panicking on out-of-bounds access.
#[cfg(any(feature = "arm", feature = "aarch64", feature = "riscv"))]
fn read_le32(bytes: &[u8], offset: usize, label: &str, span: Span) -> Result<u32, AsmError> {
    if offset + 4 > bytes.len() {
        return Err(AsmError::Syntax {
            msg: alloc::format!(
                "relocation offset {offset} out of bounds (buffer len {}) for label '{label}'",
                bytes.len()
            ),
            span,
        });
    }
    // The bounds check above guarantees the slice is exactly 4 bytes,
    // so the try_into conversion is infallible.
    let arr: [u8; 4] = match bytes[offset..offset + 4].try_into() {
        Ok(a) => a,
        Err(_) => {
            return Err(AsmError::Syntax {
                msg: alloc::format!(
                    "relocation offset {offset} out of bounds (buffer len {}) for label '{label}'",
                    bytes.len()
                ),
                span,
            });
        }
    };
    Ok(u32::from_le_bytes(arr))
}

/// Read a little-endian u16 from `bytes` at `offset`, with bounds checking.
#[cfg(any(feature = "arm", feature = "riscv"))]
fn read_le16(bytes: &[u8], offset: usize, label: &str, span: Span) -> Result<u16, AsmError> {
    if offset + 2 > bytes.len() {
        return Err(AsmError::Syntax {
            msg: alloc::format!(
                "relocation offset {offset} out of bounds (buffer len {}) for label '{label}'",
                bytes.len()
            ),
            span,
        });
    }
    let arr: [u8; 2] = match bytes[offset..offset + 2].try_into() {
        Ok(a) => a,
        Err(_) => {
            return Err(AsmError::Syntax {
                msg: alloc::format!(
                    "relocation offset {offset} out of bounds (buffer len {}) for label '{label}'",
                    bytes.len()
                ),
                span,
            });
        }
    };
    Ok(u16::from_le_bytes(arr))
}

/// ARM modified-immediate encoder for the linker: find (imm8, rot) such that
/// `value == imm8.rotate_right(rot * 2)`.
#[cfg(feature = "arm")]
fn encode_arm_imm_for_linker(value: u32) -> Option<(u8, u8)> {
    for rot in 0..16u8 {
        let shift = rot * 2;
        let rotated = value.rotate_left(shift as u32);
        if rotated <= 0xFF {
            return Some((rotated as u8, rot));
        }
    }
    None
}

/// The resolved output: (machine code bytes, label→address table, applied relocations, fragment offsets).
type ResolveOutput = (
    Vec<u8>,
    Vec<(String, u64)>,
    Vec<AppliedRelocation>,
    Vec<u64>,
);

/// An applied relocation in the final output — describes where a label
/// reference was patched. Useful for tooling, debugging, and re-linking.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AppliedRelocation {
    /// Offset in the output byte stream where the value was written.
    pub offset: usize,
    /// Size of the relocated value in bytes (1, 2, 4, or 8).
    pub size: u8,
    /// Target label name.
    pub label: String,
    /// How the linker patches the target address into the instruction bytes.
    pub kind: RelocKind,
    /// Addend.
    pub addend: i64,
}

// ─── Fragment ──────────────────────────────────────────────

/// A fragment of assembled output.
///
/// The linker operates on an ordered list of fragments.  During branch
/// relaxation the sizes of [`Fragment::Relaxable`] and [`Fragment::Align`]
/// fragments may change, but they only ever *grow* (monotonic), which
/// guarantees convergence.
#[derive(Debug, Clone)]
pub enum Fragment {
    /// Fixed-size bytes with optional relocation.
    Fixed {
        /// The raw assembled bytes (inline for instructions, heap for large data).
        bytes: FragmentBytes,
        /// Optional relocation to apply to these bytes.
        relocation: Option<Relocation>,
        /// Source span of the originating instruction or directive.
        span: Span,
    },
    /// Alignment padding — size depends on preceding layout.
    ///
    /// When `use_nop` is true (x86/x86-64 code alignment with no explicit
    /// fill byte), pad with Intel-recommended multi-byte NOP sequences
    /// instead of repeating a single fill byte.
    Align {
        /// Required byte alignment (must be a power of two).
        alignment: u32,
        /// Byte value used for padding when `use_nop` is false.
        fill: u8,
        /// If set, skip this alignment entirely when the required padding
        /// exceeds this many bytes.
        max_skip: Option<u32>,
        /// Use multi-byte NOP sequences instead of repeating `fill`.
        use_nop: bool,
        /// Source span of the alignment directive.
        span: Span,
    },
    /// A relaxable branch instruction.
    ///
    /// Starts in short form (rel8).  If the linker determines the target is
    /// beyond ±127 bytes it promotes to the long form (rel32) and re-lays
    /// out.  Promotion is irreversible (Szymanski monotonic growth).
    Relaxable {
        /// Short-form bytes (e.g. `[0xEB, 0x00]` for JMP rel8).
        short_bytes: InstrBytes,
        /// Offset of the rel8 displacement within `short_bytes`.
        short_reloc_offset: usize,
        /// Optional relocation for the short form.  When `Some`, the linker
        /// applies this relocation to `short_bytes` instead of raw byte-patching.
        /// Used for RISC-V B-type branches.
        short_relocation: Option<Relocation>,
        /// Long-form bytes (e.g. `[0xE9, 0,0,0,0]` for JMP rel32).
        long_bytes: InstrBytes,
        /// Relocation for the long form (contains label, offset, etc.).
        long_relocation: Relocation,
        /// Whether this fragment has been promoted to long form.
        is_long: bool,
        /// Source span.
        span: Span,
    },
    /// Advance the location counter to an absolute address, padding with
    /// `fill` bytes.  If the target is behind the current position, an
    /// error is raised during emission.
    Org {
        /// Absolute target address for the location counter.
        target: u64,
        /// Byte value used to fill the gap.
        fill: u8,
        /// Source span of the `.org` directive.
        span: Span,
    },
}

// ─── Linker internals ──────────────────────────────────────

/// A label definition tracking which fragment it precedes.
#[derive(Debug, Clone)]
struct LabelDef {
    fragment_index: usize,
    span: Span,
}

/// Numeric label tracking (supports forward/backward references like `1:` / `1b` / `1f`).
#[derive(Debug, Clone, Default)]
struct NumericLabels {
    defs: BTreeMap<u32, Vec<usize>>,
}

// ─── Public API ────────────────────────────────────────────

/// The linker: collects fragments and labels, resolves everything.
#[derive(Debug)]
pub struct Linker {
    fragments: Vec<Fragment>,
    labels: BTreeMap<String, LabelDef>,
    externals: BTreeMap<String, u64>,
    numeric: NumericLabels,
    constants: BTreeMap<String, i128>,
    base_address: u64,
}

impl Default for Linker {
    fn default() -> Self {
        Self::new()
    }
}

impl Linker {
    /// Create a new, empty linker with base address 0.
    pub fn new() -> Self {
        Self {
            fragments: Vec::new(),
            labels: BTreeMap::new(),
            externals: BTreeMap::new(),
            numeric: NumericLabels::default(),
            constants: BTreeMap::new(),
            base_address: 0,
        }
    }

    /// Set the base (origin) address for the assembled output.
    pub fn set_base_address(&mut self, addr: u64) {
        self.base_address = addr;
    }

    /// Get the base address.
    pub fn base_address(&self) -> u64 {
        self.base_address
    }

    /// The number of fragments currently added.
    pub fn fragment_count(&self) -> usize {
        self.fragments.len()
    }

    /// Define an external label at a known absolute address.
    pub fn define_external(&mut self, name: &str, addr: u64) {
        self.externals.insert(String::from(name), addr);
    }

    /// Define a constant value (`.equ` / `.set`).
    pub fn define_constant(&mut self, name: &str, value: i128) {
        self.constants.insert(String::from(name), value);
    }

    /// Look up a constant value by name.
    pub fn get_constant(&self, name: &str) -> Option<&i128> {
        self.constants.get(name)
    }

    /// Add a label definition at the current position (before the next fragment).
    pub fn add_label(&mut self, name: &str, span: Span) -> Result<(), AsmError> {
        // Numeric labels (e.g. `1:`) can be redefined.
        if let Ok(n) = name.parse::<u32>() {
            self.numeric
                .defs
                .entry(n)
                .or_default()
                .push(self.fragments.len());
            return Ok(());
        }

        if let Some(existing) = self.labels.get(name) {
            return Err(AsmError::DuplicateLabel {
                label: String::from(name),
                span,
                first_span: existing.span,
            });
        }
        self.labels.insert(
            String::from(name),
            LabelDef {
                fragment_index: self.fragments.len(),
                span,
            },
        );
        Ok(())
    }

    /// Add a pre-built fragment.
    pub fn add_fragment(&mut self, fragment: Fragment) {
        self.fragments.push(fragment);
    }

    /// Convenience: add fixed bytes (data, non-branch instructions, etc.).
    pub fn add_bytes(&mut self, bytes: Vec<u8>, span: Span) {
        self.fragments.push(Fragment::Fixed {
            bytes: FragmentBytes::Heap(bytes),
            relocation: None,
            span,
        });
    }

    /// Add an encoded instruction, automatically choosing `Fixed` or `Relaxable`.
    pub fn add_encoded(
        &mut self,
        bytes: InstrBytes,
        relocation: Option<Relocation>,
        relax: Option<RelaxInfo>,
        span: Span,
    ) -> Result<(), AsmError> {
        if let Some(ri) = relax {
            let long_relocation = relocation.ok_or_else(|| AsmError::Syntax {
                msg: String::from("internal: relaxable instruction missing relocation"),
                span,
            })?;
            self.fragments.push(Fragment::Relaxable {
                short_bytes: ri.short_bytes,
                short_reloc_offset: ri.short_reloc_offset,
                short_relocation: ri.short_relocation,
                long_bytes: bytes,
                long_relocation,
                is_long: false,
                span,
            });
        } else {
            self.fragments.push(Fragment::Fixed {
                bytes: FragmentBytes::Inline(bytes),
                relocation,
                span,
            });
        }
        Ok(())
    }

    /// Add alignment padding.
    pub fn add_alignment(
        &mut self,
        alignment: u32,
        fill: u8,
        max_skip: Option<u32>,
        use_nop: bool,
        span: Span,
    ) {
        self.fragments.push(Fragment::Align {
            alignment,
            fill,
            max_skip,
            use_nop,
            span,
        });
    }

    /// Add an `.org` directive: advance the location counter to `target`,
    /// padding with `fill` bytes.
    pub fn add_org(&mut self, target: u64, fill: u8, span: Span) {
        self.fragments.push(Fragment::Org { target, fill, span });
    }

    // ── resolve ────────────────────────────────────────────

    /// Resolve all labels, perform branch relaxation, and return
    /// the final bytes together with a label→address table and applied relocations.
    ///
    /// # Note
    ///
    /// This method **consumes** the linker's internal fragment list. After calling
    /// `resolve()`, the `Linker` is left in an empty state — calling `resolve()`
    /// again will produce an empty output. If you need to re-link, create a new
    /// `Linker` instance and re-add all fragments.
    pub fn resolve(&mut self) -> Result<ResolveOutput, AsmError> {
        // Phase 1: branch relaxation (Szymanski monotonic growth)
        let offsets = self.relax()?;

        // Phase 2: emit final bytes with patched relocations (reuse offsets)
        self.emit_final(offsets)
    }

    // ── branch relaxation ──────────────────────────────────

    /// Iteratively grow short branches that cannot reach their targets.
    /// Returns the final computed offsets on success so callers can reuse them.
    fn relax(&mut self) -> Result<Vec<u64>, AsmError> {
        let mut offsets = Vec::with_capacity(self.fragments.len() + 1);
        let mut to_expand: Vec<usize> = Vec::new();

        for _iter in 0..MAX_RELAXATION_ITERS {
            self.compute_offsets_into(&mut offsets);
            to_expand.clear();

            for (i, frag) in self.fragments.iter().enumerate() {
                if let Fragment::Relaxable {
                    short_bytes,
                    short_relocation,
                    long_relocation,
                    is_long,
                    ..
                } = frag
                {
                    if !is_long {
                        let frag_end = offsets[i] + short_bytes.len() as u64;
                        match self.resolve_label_with_offsets(&long_relocation.label, i, &offsets) {
                            Ok(target) => {
                                let disp = target as i64 - frag_end as i64 + long_relocation.addend;
                                let in_range = if let Some(ref sr) = short_relocation {
                                    // Architecture-specific short form range check
                                    match sr.kind {
                                        #[cfg(feature = "riscv")]
                                        RelocKind::RvBranch12 => {
                                            // B-type: PC-relative from instruction start
                                            let pc_offset = disp + (short_bytes.len() as i64);
                                            (-(1i64 << 12)..(1i64 << 12)).contains(&pc_offset)
                                        }
                                        #[cfg(feature = "riscv")]
                                        RelocKind::RvCBranch8 => {
                                            // CB-type c.beqz/c.bnez: ±256 B (9-bit signed)
                                            let pc_offset = disp + (short_bytes.len() as i64);
                                            (-(1i64 << 8)..(1i64 << 8)).contains(&pc_offset)
                                        }
                                        #[cfg(feature = "riscv")]
                                        RelocKind::RvCJump11 => {
                                            // CJ-type c.j: ±2 KB (12-bit signed)
                                            let pc_offset = disp + (short_bytes.len() as i64);
                                            (-(1i64 << 11)..(1i64 << 11)).contains(&pc_offset)
                                        }
                                        #[cfg(feature = "aarch64")]
                                        RelocKind::Aarch64Branch19 => {
                                            // B.cond / CBZ / CBNZ: ±1 MB (19-bit signed × 4)
                                            let pc_offset = disp + (short_bytes.len() as i64);
                                            (-(1i64 << 20)..(1i64 << 20)).contains(&pc_offset)
                                        }
                                        #[cfg(feature = "aarch64")]
                                        RelocKind::Aarch64Branch14 => {
                                            // TBZ / TBNZ: ±32 KB (14-bit signed × 4)
                                            let pc_offset = disp + (short_bytes.len() as i64);
                                            (-(1i64 << 15)..(1i64 << 15)).contains(&pc_offset)
                                        }
                                        #[cfg(feature = "aarch64")]
                                        RelocKind::Aarch64Adr21 => {
                                            // ADR: ±1 MB (21-bit signed)
                                            let pc_offset = disp + (short_bytes.len() as i64);
                                            (-(1i64 << 20)..(1i64 << 20)).contains(&pc_offset)
                                        }
                                        #[cfg(feature = "arm")]
                                        RelocKind::ThumbBranch8 => {
                                            // B<cond> narrow: signed 8-bit offset >> 1 → ±256 B
                                            // PC = instr + 4 in Thumb
                                            let pc_offset = disp + (short_bytes.len() as i64);
                                            (-(1i64 << 8)..(1i64 << 8)).contains(&pc_offset)
                                        }
                                        #[cfg(feature = "arm")]
                                        RelocKind::ThumbBranch11 => {
                                            // B narrow: signed 11-bit offset >> 1 → ±2 KB
                                            let pc_offset = disp + (short_bytes.len() as i64);
                                            (-(1i64 << 11)..(1i64 << 11)).contains(&pc_offset)
                                        }
                                        _ => (-128..=127).contains(&disp),
                                    }
                                } else {
                                    // x86 default: rel8 ±127 from frag_end
                                    (-128..=127).contains(&disp)
                                };
                                if !in_range {
                                    to_expand.push(i);
                                }
                            }
                            Err(_) => {
                                // Undefined label — conservatively assume long form.
                                // The real error will surface during emit_final.
                                to_expand.push(i);
                            }
                        }
                    }
                }
            }

            if to_expand.is_empty() {
                return Ok(offsets);
            }

            for &idx in &to_expand {
                if let Fragment::Relaxable {
                    ref mut is_long, ..
                } = self.fragments[idx]
                {
                    *is_long = true;
                }
            }
        }

        Err(AsmError::RelaxationLimit {
            max: MAX_RELAXATION_ITERS,
        })
    }

    // ── offset computation ─────────────────────────────────

    /// Build an offset table: `offsets[i]` is the absolute address of fragment `i`.
    ///
    /// `offsets[fragments.len()]` is a sentinel for the total end address.
    /// Reuses the provided vector to avoid repeated allocation.
    fn compute_offsets_into(&self, offsets: &mut Vec<u64>) {
        offsets.clear();
        let mut current = self.base_address;
        for frag in &self.fragments {
            offsets.push(current);
            match frag {
                Fragment::Fixed { bytes, .. } => {
                    current += bytes.len() as u64;
                }
                Fragment::Align {
                    alignment,
                    max_skip,
                    ..
                } => {
                    let a = *alignment as u64;
                    if a > 1 {
                        let aligned = current.div_ceil(a) * a;
                        let padding = aligned - current;
                        if max_skip.map_or(true, |ms| padding <= ms as u64) {
                            current = aligned;
                        }
                    }
                }
                Fragment::Relaxable {
                    short_bytes,
                    long_bytes,
                    is_long,
                    ..
                } => {
                    if *is_long {
                        current += long_bytes.len() as u64;
                    } else {
                        current += short_bytes.len() as u64;
                    }
                }
                Fragment::Org { target, .. } => {
                    if *target > current {
                        current = *target;
                    }
                    // If target <= current, no advancement (error at emit time)
                }
            }
        }
        offsets.push(current);
    }

    // ── final emit ─────────────────────────────────────────

    fn emit_final(&mut self, offsets: Vec<u64>) -> Result<ResolveOutput, AsmError> {
        let total_size = offsets.last().copied().unwrap_or(self.base_address) - self.base_address;
        let mut output = Vec::with_capacity(total_size as usize);
        let mut applied_relocs = Vec::new();

        // Take fragments out so `self` is free for apply_relocation / resolve_label calls.
        // This avoids cloning every fragment's byte buffer on the final emit path.
        let mut fragments = core::mem::take(&mut self.fragments);

        for (i, frag) in fragments.iter_mut().enumerate() {
            match frag {
                Fragment::Fixed {
                    bytes,
                    relocation,
                    span,
                } => {
                    if let Some(ref mut reloc) = relocation {
                        let frag_output_offset = output.len();
                        // Patch in-place — no heap clone needed.
                        self.apply_relocation(bytes, reloc, offsets[i], &offsets, i, *span)?;
                        applied_relocs.push(AppliedRelocation {
                            offset: frag_output_offset + reloc.offset,
                            size: reloc.size,
                            // Take ownership — emit_final is terminal, labels won't be read again.
                            label: reloc.label.to_string(),
                            kind: reloc.kind,
                            addend: reloc.addend,
                        });
                        output.extend_from_slice(bytes);
                    } else {
                        output.extend_from_slice(bytes);
                    }
                }

                Fragment::Align {
                    alignment,
                    fill,
                    max_skip,
                    use_nop,
                    ..
                } => {
                    let a = *alignment as u64;
                    if a > 1 {
                        let current = offsets[i];
                        let aligned = current.div_ceil(a) * a;
                        let padding = (aligned - current) as usize;
                        // skip if max_skip is exceeded
                        if max_skip.is_some_and(|ms| padding > ms as usize) {
                            // no padding emitted
                        } else if *use_nop {
                            emit_nop_padding(&mut output, padding);
                        } else {
                            output.extend(core::iter::repeat(*fill).take(padding));
                        }
                    }
                }

                Fragment::Relaxable {
                    short_bytes,
                    short_reloc_offset,
                    short_relocation,
                    long_bytes,
                    long_relocation,
                    is_long,
                    span,
                } => {
                    if *is_long {
                        let frag_output_offset = output.len();
                        // Patch long_bytes in-place — no heap clone needed.
                        self.apply_relocation(
                            long_bytes,
                            long_relocation,
                            offsets[i],
                            &offsets,
                            i,
                            *span,
                        )?;
                        applied_relocs.push(AppliedRelocation {
                            offset: frag_output_offset + long_relocation.offset,
                            size: long_relocation.size,
                            label: (*long_relocation.label).into(),
                            kind: long_relocation.kind,
                            addend: long_relocation.addend,
                        });
                        output.extend_from_slice(long_bytes);
                    } else if let Some(ref mut sr) = short_relocation {
                        // Short form with architecture-specific relocation
                        // (e.g. RISC-V B-type branch).
                        let frag_output_offset = output.len();
                        // Patch short_bytes in-place — no heap clone needed.
                        self.apply_relocation(short_bytes, sr, offsets[i], &offsets, i, *span)?;
                        applied_relocs.push(AppliedRelocation {
                            offset: frag_output_offset + sr.offset,
                            size: sr.size,
                            label: (*sr.label).into(),
                            kind: sr.kind,
                            addend: sr.addend,
                        });
                        output.extend_from_slice(short_bytes);
                    } else {
                        // Short form — patch rel8 displacement (x86)
                        let frag_output_offset = output.len();
                        let target =
                            self.resolve_label_with_offsets(&long_relocation.label, i, &offsets)?;
                        let frag_end = offsets[i] + short_bytes.len() as u64;
                        let disp = target as i64 - frag_end as i64 + long_relocation.addend;
                        if !(-128..=127).contains(&disp) {
                            return Err(AsmError::BranchOutOfRange {
                                label: long_relocation.label.to_string(),
                                disp,
                                max: 127,
                                span: *span,
                            });
                        }
                        // Patch short_bytes in-place — no heap clone needed.
                        short_bytes[*short_reloc_offset] = disp as i8 as u8;
                        applied_relocs.push(AppliedRelocation {
                            offset: frag_output_offset + *short_reloc_offset,
                            size: 1,
                            label: (*long_relocation.label).into(),
                            kind: RelocKind::X86Relative,
                            addend: long_relocation.addend,
                        });
                        output.extend_from_slice(short_bytes);
                    }
                }

                Fragment::Org {
                    target, fill, span, ..
                } => {
                    let current = offsets[i];
                    if *target < current {
                        return Err(AsmError::Syntax {
                            msg: alloc::format!(
                                ".org target 0x{:X} is behind current position 0x{:X}",
                                target,
                                current
                            ),
                            span: *span,
                        });
                    }
                    let padding = (*target - current) as usize;
                    output.extend(core::iter::repeat(*fill).take(padding));
                }
            }
        }

        // Restore fragments (now patched in-place, but structure intact).
        self.fragments = fragments;

        // Collect label addresses
        let label_table: Vec<(String, u64)> = self
            .labels
            .iter()
            .map(|(name, def)| (name.clone(), offsets[def.fragment_index]))
            .collect();

        Ok((output, label_table, applied_relocs, offsets))
    }

    // ── relocation patching ────────────────────────────────

    fn apply_relocation(
        &self,
        bytes: &mut [u8],
        reloc: &Relocation,
        frag_abs: u64,
        offsets: &[u64],
        from_fragment: usize,
        span: Span,
    ) -> Result<(), AsmError> {
        let target_addr = self.resolve_label_with_offsets(&reloc.label, from_fragment, offsets)?;
        let reloc_abs = frag_abs + reloc.offset as u64;

        match reloc.kind {
            RelocKind::X86Relative => {
                // RIP = address past the entire instruction, not just past the reloc field.
                // trailing_bytes accounts for any immediate bytes following the displacement.
                let rip = reloc_abs + reloc.size as u64 + reloc.trailing_bytes as u64;
                let rel = (target_addr as i64)
                    .wrapping_sub(rip as i64)
                    .wrapping_add(reloc.addend);
                match reloc.size {
                    1 => {
                        if rel < i8::MIN as i64 || rel > i8::MAX as i64 {
                            return Err(AsmError::BranchOutOfRange {
                                label: reloc.label.to_string(),
                                disp: rel,
                                max: 127,
                                span,
                            });
                        }
                        bytes[reloc.offset] = rel as i8 as u8;
                    }
                    4 => {
                        if rel < i32::MIN as i64 || rel > i32::MAX as i64 {
                            return Err(AsmError::BranchOutOfRange {
                                label: reloc.label.to_string(),
                                disp: rel,
                                max: i32::MAX as i64,
                                span,
                            });
                        }
                        let b = (rel as i32).to_le_bytes();
                        bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&b);
                    }
                    other => {
                        return Err(AsmError::Syntax {
                            msg: alloc::format!(
                                "unsupported RIP-relative relocation size: {other}"
                            ),
                            span,
                        });
                    }
                }
            }
            RelocKind::Absolute => {
                let addr = target_addr.wrapping_add(reloc.addend as u64);
                match reloc.size {
                    1 => {
                        if addr > u8::MAX as u64 {
                            return Err(AsmError::Syntax {
                                msg: alloc::format!(
                                    "absolute address 0x{addr:X} exceeds 8-bit relocation range for '{}'",
                                    reloc.label
                                ),
                                span,
                            });
                        }
                        bytes[reloc.offset] = addr as u8;
                    }
                    2 => {
                        if addr > u16::MAX as u64 {
                            return Err(AsmError::Syntax {
                                msg: alloc::format!(
                                    "absolute address 0x{addr:X} exceeds 16-bit relocation range for '{}'",
                                    reloc.label
                                ),
                                span,
                            });
                        }
                        bytes[reloc.offset..reloc.offset + 2]
                            .copy_from_slice(&(addr as u16).to_le_bytes());
                    }
                    4 => {
                        if addr > u32::MAX as u64 {
                            return Err(AsmError::Syntax {
                                msg: alloc::format!(
                                    "absolute address 0x{addr:X} exceeds 32-bit relocation range for '{}'",
                                    reloc.label
                                ),
                                span,
                            });
                        }
                        bytes[reloc.offset..reloc.offset + 4]
                            .copy_from_slice(&(addr as u32).to_le_bytes());
                    }
                    8 => {
                        bytes[reloc.offset..reloc.offset + 8].copy_from_slice(&addr.to_le_bytes());
                    }
                    other => {
                        return Err(AsmError::Syntax {
                            msg: alloc::format!("unsupported absolute relocation size: {other}"),
                            span,
                        });
                    }
                }
            }
            #[cfg(feature = "arm")]
            RelocKind::ArmBranch24 => {
                // ARM32 B/BL: PC = instr_addr + 8, offset = (target - PC) >> 2, packed bits 23:0
                let pc = reloc_abs + 8;
                let rel = (target_addr as i64)
                    .wrapping_sub(pc as i64)
                    .wrapping_add(reloc.addend);
                let offset = rel >> 2;
                if !(-(1 << 23)..(1 << 23)).contains(&offset) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 25) - 4,
                        span,
                    });
                }
                let imm24 = (offset as u32) & 0x00FF_FFFF;
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                word = (word & 0xFF00_0000) | imm24;
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            #[cfg(feature = "arm")]
            RelocKind::ArmLdrLit => {
                // ARM32 LDR Rd, label: PC = instr_addr + 8, 12-bit offset, U-bit (bit 23)
                let pc = reloc_abs + 8;
                let rel = (target_addr as i64)
                    .wrapping_sub(pc as i64)
                    .wrapping_add(reloc.addend);
                let abs_rel = rel.unsigned_abs();
                if abs_rel > 4095 {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: 4095,
                        span,
                    });
                }
                let u_bit = if rel >= 0 { 1u32 } else { 0u32 };
                let imm12 = (abs_rel as u32) & 0xFFF;
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                word = (word & 0xFF7F_F000) | (u_bit << 23) | imm12;
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            #[cfg(feature = "arm")]
            RelocKind::ArmAdr => {
                // ARM32 ADR Rd, label → ADD/SUB Rd, PC, #rotated_imm
                // PC = instr_addr + 8 (ARM pipeline)
                // The data-processing immediate format uses 8-bit imm + 4-bit rotation
                let pc = reloc_abs + 8;
                let rel = (target_addr as i64)
                    .wrapping_sub(pc as i64)
                    .wrapping_add(reloc.addend);
                let abs_rel = rel.unsigned_abs() as u32;
                let (op, imm8, rot) = if rel >= 0 {
                    // ADD Rd, PC, #imm
                    let (i, r) = encode_arm_imm_for_linker(abs_rel).ok_or_else(|| {
                        AsmError::BranchOutOfRange {
                            label: reloc.label.to_string(),
                            disp: rel,
                            max: 255, // max unrotated; actual range depends on pattern
                            span,
                        }
                    })?;
                    (0x4u32, i, r)
                } else {
                    // SUB Rd, PC, #imm
                    let (i, r) = encode_arm_imm_for_linker(abs_rel).ok_or_else(|| {
                        AsmError::BranchOutOfRange {
                            label: reloc.label.to_string(),
                            disp: rel,
                            max: 255,
                            span,
                        }
                    })?;
                    (0x2u32, i, r)
                };
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                // Clear opcode (bits 24:21) and immediate field (bits 11:0)
                word = (word & 0xF1F0_F000) | (op << 21) | ((rot as u32) << 8) | (imm8 as u32);
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            #[cfg(feature = "arm")]
            RelocKind::ThumbBranch8 => {
                // Thumb conditional branch (16-bit): PC = instr + 4
                let pc = reloc_abs + 4;
                let rel = (target_addr as i64)
                    .wrapping_sub(pc as i64)
                    .wrapping_add(reloc.addend);
                let offset = rel >> 1;
                if !(-(1i64 << 7)..(1i64 << 7)).contains(&offset) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: 254,
                        span,
                    });
                }
                let imm8 = (offset as u8) as u16;
                let mut hw = read_le16(bytes, reloc.offset, &reloc.label, span)?;
                hw = (hw & 0xFF00) | (imm8 & 0xFF);
                bytes[reloc.offset..reloc.offset + 2].copy_from_slice(&hw.to_le_bytes());
            }
            #[cfg(feature = "arm")]
            RelocKind::ThumbBranch11 => {
                // Thumb unconditional branch (16-bit): PC = instr + 4
                let pc = reloc_abs + 4;
                let rel = (target_addr as i64)
                    .wrapping_sub(pc as i64)
                    .wrapping_add(reloc.addend);
                let offset = rel >> 1;
                if !(-(1i64 << 10)..(1i64 << 10)).contains(&offset) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: 2046,
                        span,
                    });
                }
                let imm11 = (offset as u16) & 0x7FF;
                let mut hw = read_le16(bytes, reloc.offset, &reloc.label, span)?;
                hw = (hw & 0xF800) | imm11;
                bytes[reloc.offset..reloc.offset + 2].copy_from_slice(&hw.to_le_bytes());
            }
            #[cfg(feature = "arm")]
            RelocKind::ThumbBl => {
                // Thumb-2 BL (32-bit): PC = instr + 4
                let pc = reloc_abs + 4;
                let rel = (target_addr as i64)
                    .wrapping_sub(pc as i64)
                    .wrapping_add(reloc.addend);
                let offset = rel >> 1;
                if !(-(1i64 << 23)..(1i64 << 23)).contains(&offset) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 24) - 2,
                        span,
                    });
                }
                let s = if offset < 0 { 1_u16 } else { 0 };
                let imm = offset as u32;
                let imm10 = ((imm >> 11) & 0x3FF) as u16;
                let imm11 = (imm & 0x7FF) as u16;
                let j1 = (!((imm >> 23) ^ (s as u32)) & 1) as u16;
                let j2 = (!((imm >> 22) ^ (s as u32)) & 1) as u16;
                let hw1 = 0xF000 | (s << 10) | imm10;
                let hw2 = 0xD000 | (j1 << 13) | (j2 << 11) | imm11;
                bytes[reloc.offset..reloc.offset + 2].copy_from_slice(&hw1.to_le_bytes());
                bytes[reloc.offset + 2..reloc.offset + 4].copy_from_slice(&hw2.to_le_bytes());
            }
            #[cfg(feature = "arm")]
            RelocKind::ThumbBranchW => {
                // Thumb-2 B.W (32-bit wide unconditional): PC = instr + 4
                let pc = reloc_abs + 4;
                let rel = (target_addr as i64)
                    .wrapping_sub(pc as i64)
                    .wrapping_add(reloc.addend);
                let offset = rel >> 1;
                if !(-(1i64 << 23)..(1i64 << 23)).contains(&offset) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 24) - 2,
                        span,
                    });
                }
                let s = if offset < 0 { 1_u16 } else { 0 };
                let imm = offset as u32;
                let imm10 = ((imm >> 11) & 0x3FF) as u16;
                let imm11 = (imm & 0x7FF) as u16;
                let j1 = (!((imm >> 23) ^ (s as u32)) & 1) as u16;
                let j2 = (!((imm >> 22) ^ (s as u32)) & 1) as u16;
                let hw1 = 0xF000 | (s << 10) | imm10;
                let hw2 = 0x9000 | (j1 << 13) | (j2 << 11) | imm11;
                bytes[reloc.offset..reloc.offset + 2].copy_from_slice(&hw1.to_le_bytes());
                bytes[reloc.offset + 2..reloc.offset + 4].copy_from_slice(&hw2.to_le_bytes());
            }
            #[cfg(feature = "arm")]
            RelocKind::ThumbCondBranchW => {
                // Thumb-2 B<cond>.W (32-bit wide conditional): PC = instr + 4
                let pc = reloc_abs + 4;
                let rel = (target_addr as i64)
                    .wrapping_sub(pc as i64)
                    .wrapping_add(reloc.addend);
                let offset = rel >> 1;
                if !(-(1i64 << 19)..(1i64 << 19)).contains(&offset) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 20) - 2,
                        span,
                    });
                }
                let s = if offset < 0 { 1_u16 } else { 0 };
                let imm = offset as u32;
                let imm6 = ((imm >> 11) & 0x3F) as u16;
                let imm11 = (imm & 0x7FF) as u16;
                let j1 = ((imm >> 17) & 1) as u16;
                let j2 = ((imm >> 18) & 1) as u16;
                // Read existing hw1 to preserve condition code bits
                let existing_hw1 = read_le16(bytes, reloc.offset, &reloc.label, span)?;
                let cond = (existing_hw1 >> 6) & 0xF;
                let hw1 = 0xF000 | (s << 10) | (cond << 6) | imm6;
                let hw2 = 0x8000 | (j1 << 13) | (j2 << 11) | imm11;
                bytes[reloc.offset..reloc.offset + 2].copy_from_slice(&hw1.to_le_bytes());
                bytes[reloc.offset + 2..reloc.offset + 4].copy_from_slice(&hw2.to_le_bytes());
            }
            #[cfg(feature = "arm")]
            RelocKind::ThumbLdrLit8 => {
                // Thumb LDR Rt, [PC, #imm8×4]: PC = (instr_addr + 4) & ~3
                // Forward only, word-aligned, 0..1020 byte range
                let pc = (reloc_abs + 4) & !3;
                let rel = (target_addr as i64)
                    .wrapping_sub(pc as i64)
                    .wrapping_add(reloc.addend);
                if !(0..=1020).contains(&rel) || (rel & 3) != 0 {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: 1020,
                        span,
                    });
                }
                let imm8 = (rel >> 2) as u16;
                let existing = read_le16(bytes, reloc.offset, &reloc.label, span)?;
                let hw = (existing & 0xFF00) | imm8;
                bytes[reloc.offset..reloc.offset + 2].copy_from_slice(&hw.to_le_bytes());
            }
            #[cfg(feature = "aarch64")]
            RelocKind::Aarch64Jump26 => {
                // AArch64 B/BL: PC-relative offset >> 2 in bits 25:0
                let rel = (target_addr as i64)
                    .wrapping_sub(reloc_abs as i64)
                    .wrapping_add(reloc.addend);
                let offset = rel >> 2;
                if !(-(1 << 25)..(1 << 25)).contains(&offset) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 27) - 4,
                        span,
                    });
                }
                let imm26 = (offset as u32) & 0x03FF_FFFF;
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                word = (word & 0xFC00_0000) | imm26;
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            #[cfg(feature = "aarch64")]
            RelocKind::Aarch64Branch19 => {
                // AArch64 B.cond / CBZ / CBNZ: PC-relative offset >> 2 in bits 23:5
                let rel = (target_addr as i64)
                    .wrapping_sub(reloc_abs as i64)
                    .wrapping_add(reloc.addend);
                let offset = rel >> 2;
                if !(-(1 << 18)..(1 << 18)).contains(&offset) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 20) - 4,
                        span,
                    });
                }
                let imm19 = (offset as u32) & 0x7FFFF;
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                word = (word & 0xFF00_001F) | (imm19 << 5);
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            #[cfg(feature = "aarch64")]
            RelocKind::Aarch64Branch14 => {
                // AArch64 TBZ / TBNZ: PC-relative offset >> 2 in bits 18:5
                let rel = (target_addr as i64)
                    .wrapping_sub(reloc_abs as i64)
                    .wrapping_add(reloc.addend);
                let offset = rel >> 2;
                if !(-(1 << 13)..(1 << 13)).contains(&offset) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 15) - 4,
                        span,
                    });
                }
                let imm14 = (offset as u32) & 0x3FFF;
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                word = (word & 0xFFF8_001F) | (imm14 << 5);
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            #[cfg(feature = "aarch64")]
            RelocKind::Aarch64LdrLit19 => {
                // AArch64 LDR (literal): PC-relative offset >> 2 in bits 23:5
                let rel = (target_addr as i64)
                    .wrapping_sub(reloc_abs as i64)
                    .wrapping_add(reloc.addend);
                let offset = rel >> 2;
                if !(-(1 << 18)..(1 << 18)).contains(&offset) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 20) - 4,
                        span,
                    });
                }
                let imm19 = (offset as u32) & 0x7FFFF;
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                word = (word & 0xFF00_001F) | (imm19 << 5);
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            #[cfg(feature = "aarch64")]
            RelocKind::Aarch64Adr21 => {
                // AArch64 ADR: PC-relative, immhi (bits 23:5), immlo (bits 30:29)
                let rel = (target_addr as i64)
                    .wrapping_sub(reloc_abs as i64)
                    .wrapping_add(reloc.addend);
                if !(-(1 << 20)..(1 << 20)).contains(&rel) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 20) - 1,
                        span,
                    });
                }
                let immhi = ((rel >> 2) as u32) & 0x7FFFF;
                let immlo = (rel as u32) & 0x3;
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                word = (word & 0x9F00_001F) | (immlo << 29) | (immhi << 5);
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            #[cfg(feature = "aarch64")]
            RelocKind::Aarch64Adrp => {
                // AArch64 ADRP: page-relative, page = addr & ~0xFFF
                let pc_page = reloc_abs & !0xFFF;
                let target_page = target_addr.wrapping_add(reloc.addend as u64) & !0xFFF;
                let rel = (target_page as i64).wrapping_sub(pc_page as i64);
                let page_off = rel >> 12;
                if !(-(1 << 20)..(1 << 20)).contains(&page_off) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1i64 << 32) - 1,
                        span,
                    });
                }
                let immhi = ((page_off >> 2) as u32) & 0x7FFFF;
                let immlo = (page_off as u32) & 0x3;
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                word = (word & 0x9F00_001F) | (immlo << 29) | (immhi << 5);
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            #[cfg(feature = "aarch64")]
            RelocKind::Aarch64AdrpAddPair => {
                // AArch64 ADRP+ADD pair: first word is ADRP, second is ADD.
                // ADRP: page-relative offset in immhi/immlo
                let pc_page = reloc_abs & !0xFFF;
                let target_with_addend = target_addr.wrapping_add(reloc.addend as u64);
                let target_page = target_with_addend & !0xFFF;
                let rel = (target_page as i64).wrapping_sub(pc_page as i64);
                let page_off = rel >> 12;
                if !(-(1 << 20)..(1 << 20)).contains(&page_off) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1i64 << 32) - 1,
                        span,
                    });
                }
                let immhi_p = ((page_off >> 2) as u32) & 0x7FFFF;
                let immlo_p = (page_off as u32) & 0x3;
                let mut adrp_word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                adrp_word = (adrp_word & 0x9F00_001F) | (immlo_p << 29) | (immhi_p << 5);
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&adrp_word.to_le_bytes());

                // ADD: lo12 bits of target address in imm12 (bits 21:10)
                let lo12 = (target_with_addend & 0xFFF) as u32;
                let add_offset = reloc.offset + 4;
                let mut add_word = read_le32(bytes, add_offset, &reloc.label, span)?;
                add_word = (add_word & 0xFFC003FF) | (lo12 << 10);
                bytes[add_offset..add_offset + 4].copy_from_slice(&add_word.to_le_bytes());
            }
            #[cfg(feature = "riscv")]
            RelocKind::RvJal20 => {
                // RISC-V JAL: 21-bit signed PC-relative offset (bit 0 always 0)
                // J-type immediate: imm[20|10:1|11|19:12] packed into bits 31:12
                let rel = (target_addr as i64)
                    .wrapping_sub(reloc_abs as i64)
                    .wrapping_add(reloc.addend);
                if !(-(1i64 << 20)..(1i64 << 20)).contains(&rel) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 20) - 2,
                        span,
                    });
                }
                let imm = rel as u32;
                let packed = (imm & 0x0010_0000)          // imm[20]   → bit 31
                    | ((imm & 0x7FE) << 20)               // imm[10:1] → bits 30:21
                    | ((imm & 0x800) << 9)                // imm[11]   → bit 20
                    | (imm & 0x000F_F000); // imm[19:12]→ bits 19:12
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                word = (word & 0xFFF) | packed;
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            #[cfg(feature = "riscv")]
            RelocKind::RvBranch12 => {
                // RISC-V B-type: 13-bit signed PC-relative offset (bit 0 always 0)
                // B-type immediate: imm[12|10:5] in bits 31:25, imm[4:1|11] in bits 11:7
                let rel = (target_addr as i64)
                    .wrapping_sub(reloc_abs as i64)
                    .wrapping_add(reloc.addend);
                if !(-(1i64 << 12)..(1i64 << 12)).contains(&rel) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 12) - 2,
                        span,
                    });
                }
                let imm = rel as u32;
                let packed_hi = ((imm & 0x1000) << 19)    // imm[12]   → bit 31
                    | ((imm & 0x7E0) << 20); // imm[10:5] → bits 30:25
                let packed_lo = ((imm & 0x1E) << 7)       // imm[4:1]  → bits 11:8
                    | ((imm & 0x800) >> 4); // imm[11]   → bit 7
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                word = (word & 0x01FF_F07F) | packed_hi | packed_lo;
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            #[cfg(feature = "riscv")]
            RelocKind::RvAuipc20 => {
                // RISC-V AUIPC+JALR pair: patches both instructions.
                // AUIPC at reloc.offset, JALR at reloc.offset+4.
                // hi20 = (offset + 0x800) >> 12  (rounds for sign-extension of lo12)
                // lo12 = offset - (hi20 << 12)
                let rel = (target_addr as i64)
                    .wrapping_sub(reloc_abs as i64)
                    .wrapping_add(reloc.addend);
                let hi20 = ((rel + 0x800) >> 12) as u32;
                let lo12 = (rel as u32).wrapping_sub(hi20 << 12);
                // Patch AUIPC: upper 20 bits in bits 31:12
                let mut word = read_le32(bytes, reloc.offset, &reloc.label, span)?;
                word = (word & 0xFFF) | (hi20 << 12);
                bytes[reloc.offset..reloc.offset + 4].copy_from_slice(&word.to_le_bytes());
                // Patch JALR: lower 12 bits in bits 31:20 (I-type immediate)
                let jalr_off = reloc.offset + 4;
                let mut jalr = read_le32(bytes, jalr_off, &reloc.label, span)?;
                jalr = (jalr & 0x000F_FFFF) | ((lo12 & 0xFFF) << 20);
                bytes[jalr_off..jalr_off + 4].copy_from_slice(&jalr.to_le_bytes());
            }
            #[cfg(feature = "riscv")]
            RelocKind::RvCBranch8 => {
                // RISC-V C-extension CB-type branch: 9-bit signed PC-relative offset
                // CB-type immediate: imm[8|4:3] in bits 12:10, imm[7:6|2:1|5] in bits 6:2
                let rel = (target_addr as i64)
                    .wrapping_sub(reloc_abs as i64)
                    .wrapping_add(reloc.addend);
                if !(-(1i64 << 8)..(1i64 << 8)).contains(&rel) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 8) - 2,
                        span,
                    });
                }
                let imm = rel as u16;
                // Reconstruct the CB-type halfword with the new offset
                let mut hw = read_le16(bytes, reloc.offset, &reloc.label, span)?;
                // Clear the immediate fields: bits 12:10 and 6:2
                hw &= 0xE383; // keep funct3(15:13), rs1'(9:7), op(1:0)
                              // Pack: bit8→12, bit4:3→11:10, bit7:6→6:5, bit2:1→4:3, bit5→2
                hw |= ((imm >> 8) & 1) << 12;
                hw |= ((imm >> 3) & 3) << 10;
                hw |= ((imm >> 6) & 3) << 5;
                hw |= ((imm >> 1) & 3) << 3;
                hw |= ((imm >> 5) & 1) << 2;
                bytes[reloc.offset..reloc.offset + 2].copy_from_slice(&hw.to_le_bytes());
            }
            #[cfg(feature = "riscv")]
            RelocKind::RvCJump11 => {
                // RISC-V C-extension CJ-type jump: 12-bit signed PC-relative offset
                // CJ-type immediate: imm[11|4|9:8|10|6|7|3:1|5] in bits 12:2
                let rel = (target_addr as i64)
                    .wrapping_sub(reloc_abs as i64)
                    .wrapping_add(reloc.addend);
                if !(-(1i64 << 11)..(1i64 << 11)).contains(&rel) {
                    return Err(AsmError::BranchOutOfRange {
                        label: reloc.label.to_string(),
                        disp: rel,
                        max: (1 << 11) - 2,
                        span,
                    });
                }
                let imm = rel as u16;
                let mut hw = read_le16(bytes, reloc.offset, &reloc.label, span)?;
                // Clear immediate fields: bits 12:2
                hw &= 0xE003; // keep funct3(15:13) and op(1:0)
                              // Pack: bit11→12, bit4→11, bit9:8→10:9, bit10→8, bit6→7, bit7→6, bit3:1→5:3, bit5→2
                hw |= ((imm >> 11) & 1) << 12;
                hw |= ((imm >> 4) & 1) << 11;
                hw |= ((imm >> 8) & 3) << 9;
                hw |= ((imm >> 10) & 1) << 8;
                hw |= ((imm >> 6) & 1) << 7;
                hw |= ((imm >> 7) & 1) << 6;
                hw |= ((imm >> 1) & 7) << 3;
                hw |= ((imm >> 5) & 1) << 2;
                bytes[reloc.offset..reloc.offset + 2].copy_from_slice(&hw.to_le_bytes());
            }
        }
        Ok(())
    }

    // ── label resolution ───────────────────────────────────

    fn resolve_label_with_offsets(
        &self,
        name: &str,
        from_fragment: usize,
        offsets: &[u64],
    ) -> Result<u64, AsmError> {
        // Constants
        if let Some(&value) = self.constants.get(name) {
            // Treat constant as a signed value that fits the target address space.
            // For negative constants, we rely on wrapping semantics (e.g., -1 → 0xFFFF_FFFF_FFFF_FFFF).
            return Ok(value as i64 as u64);
        }
        // Externals
        if let Some(&addr) = self.externals.get(name) {
            return Ok(addr);
        }
        // Numeric labels (e.g. "1f", "1b")
        if name.len() >= 2 {
            let last = name.as_bytes()[name.len() - 1];
            let num_part = &name[..name.len() - 1];
            if last == b'f' || last == b'b' {
                if let Ok(n) = num_part.parse::<u32>() {
                    return self.resolve_numeric_with_offsets(
                        n,
                        from_fragment,
                        last == b'f',
                        offsets,
                    );
                }
            }
        }
        // Named labels
        if let Some(def) = self.labels.get(name) {
            return Ok(offsets[def.fragment_index]);
        }

        Err(AsmError::UndefinedLabel {
            label: String::from(name),
            span: Span::new(0, 0, 0, 0),
        })
    }

    fn resolve_numeric_with_offsets(
        &self,
        num: u32,
        from_fragment: usize,
        forward: bool,
        offsets: &[u64],
    ) -> Result<u64, AsmError> {
        if let Some(defs) = self.numeric.defs.get(&num) {
            if forward {
                for &def_idx in defs {
                    if def_idx > from_fragment {
                        return Ok(offsets[def_idx]);
                    }
                }
            } else {
                for &def_idx in defs.iter().rev() {
                    if def_idx <= from_fragment {
                        return Ok(offsets[def_idx]);
                    }
                }
            }
        }
        Err(AsmError::UndefinedLabel {
            label: alloc::format!("{}{}", num, if forward { 'f' } else { 'b' }),
            span: Span::new(0, 0, 0, 0),
        })
    }
}

// ─── Multi-byte NOP padding (x86/x86-64) ──────────────────

/// Intel-recommended multi-byte NOP instruction sequences.
///
/// These are architecturally guaranteed to behave as NOPs on all modern
/// x86/x86-64 processors and execute in a single cycle on most
/// microarchitectures.
const NOP_SEQUENCES: [&[u8]; 10] = [
    &[],                                                     // 0 bytes (unused)
    &[0x90],                                                 // 1 byte : NOP
    &[0x66, 0x90],                                           // 2 bytes: 66 NOP
    &[0x0F, 0x1F, 0x00],                                     // 3 bytes: NOP DWORD ptr [EAX]
    &[0x0F, 0x1F, 0x40, 0x00],                               // 4 bytes: NOP DWORD ptr [EAX + 00H]
    &[0x0F, 0x1F, 0x44, 0x00, 0x00], // 5 bytes: NOP DWORD ptr [EAX + EAX*1 + 00H]
    &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00], // 6 bytes: 66 NOP DWORD ptr [EAX + EAX*1 + 00H]
    &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00], // 7 bytes: NOP DWORD ptr [EAX + 00000000H]
    &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00], // 8 bytes: NOP DWORD ptr [EAX + EAX*1 + 00000000H]
    &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00], // 9 bytes: 66 NOP DWORD ptr [EAX + EAX*1 + 00000000H]
];

/// Emit optimal multi-byte NOP padding of exactly `n` bytes.
///
/// Uses the largest available NOP sequences first, then fills the
/// remainder with smaller ones.
fn emit_nop_padding(output: &mut Vec<u8>, mut n: usize) {
    while n > 0 {
        let chunk = core::cmp::min(n, 9);
        output.extend_from_slice(NOP_SEQUENCES[chunk]);
        n -= chunk;
    }
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn span() -> Span {
        Span::new(1, 1, 0, 0)
    }

    fn fixed(bytes: Vec<u8>, reloc: Option<Relocation>) -> Fragment {
        Fragment::Fixed {
            bytes: FragmentBytes::Heap(bytes),
            relocation: reloc,
            span: span(),
        }
    }

    fn nop() -> Fragment {
        fixed(vec![0x90], None)
    }

    fn relaxable_jmp(label: &str) -> Fragment {
        Fragment::Relaxable {
            short_bytes: InstrBytes::from_slice(&[0xEB, 0x00]),
            short_reloc_offset: 1,
            short_relocation: None,
            long_bytes: InstrBytes::from_slice(&[0xE9, 0, 0, 0, 0]),
            long_relocation: Relocation {
                offset: 1,
                size: 4,
                label: alloc::rc::Rc::from(label),
                kind: RelocKind::X86Relative,
                addend: 0,
                trailing_bytes: 0,
            },
            is_long: false,
            span: span(),
        }
    }

    fn relaxable_jcc(cc: u8, label: &str) -> Fragment {
        Fragment::Relaxable {
            short_bytes: InstrBytes::from_slice(&[0x70 + cc, 0x00]),
            short_reloc_offset: 1,
            short_relocation: None,
            long_bytes: InstrBytes::from_slice(&[0x0F, 0x80 + cc, 0, 0, 0, 0]),
            long_relocation: Relocation {
                offset: 2,
                size: 4,
                label: alloc::rc::Rc::from(label),
                kind: RelocKind::X86Relative,
                addend: 0,
                trailing_bytes: 0,
            },
            is_long: false,
            span: span(),
        }
    }

    // ── Basic label resolution (fixed fragments) ────────

    #[test]
    fn resolve_forward_label() {
        let mut linker = Linker::new();
        linker.add_fragment(fixed(
            vec![0xE9, 0, 0, 0, 0],
            Some(Relocation {
                offset: 1,
                size: 4,
                label: alloc::rc::Rc::from("target"),
                kind: RelocKind::X86Relative,
                addend: 0,
                trailing_bytes: 0,
            }),
        ));
        linker.add_label("target", span()).unwrap();
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output, vec![0xE9, 0x00, 0x00, 0x00, 0x00, 0x90]);
    }

    #[test]
    fn resolve_backward_label() {
        let mut linker = Linker::new();
        linker.add_label("top", span()).unwrap();
        linker.add_fragment(nop());
        linker.add_fragment(fixed(
            vec![0xE9, 0, 0, 0, 0],
            Some(Relocation {
                offset: 1,
                size: 4,
                label: alloc::rc::Rc::from("top"),
                kind: RelocKind::X86Relative,
                addend: 0,
                trailing_bytes: 0,
            }),
        ));

        let (output, _, _, _) = linker.resolve().unwrap();
        let rel = i32::from_le_bytes([output[2], output[3], output[4], output[5]]);
        assert_eq!(rel, -6);
    }

    #[test]
    fn resolve_with_base_address() {
        let mut linker = Linker::new();
        linker.set_base_address(0x1000);
        linker.add_fragment(fixed(
            vec![0xE9, 0, 0, 0, 0],
            Some(Relocation {
                offset: 1,
                size: 4,
                label: alloc::rc::Rc::from("target"),
                kind: RelocKind::X86Relative,
                addend: 0,
                trailing_bytes: 0,
            }),
        ));
        linker.add_label("target", span()).unwrap();
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output, vec![0xE9, 0x00, 0x00, 0x00, 0x00, 0x90]);
    }

    #[test]
    fn resolve_external_label() {
        let mut linker = Linker::new();
        linker.define_external("printf", 0xDEAD_BEEF);
        linker.add_fragment(fixed(
            vec![0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0],
            Some(Relocation {
                offset: 2,
                size: 8,
                label: alloc::rc::Rc::from("printf"),
                kind: RelocKind::Absolute,
                addend: 0,
                trailing_bytes: 0,
            }),
        ));

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output[2..10], 0xDEAD_BEEFu64.to_le_bytes());
    }

    #[test]
    fn resolve_constant() {
        let mut linker = Linker::new();
        linker.define_constant("SYS_WRITE", 1);
        linker.add_fragment(fixed(
            vec![0xB8, 0, 0, 0, 0],
            Some(Relocation {
                offset: 1,
                size: 4,
                label: alloc::rc::Rc::from("SYS_WRITE"),
                kind: RelocKind::Absolute,
                addend: 0,
                trailing_bytes: 0,
            }),
        ));

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output, vec![0xB8, 0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn duplicate_label_error() {
        let mut linker = Linker::new();
        linker.add_label("foo", span()).unwrap();
        linker.add_fragment(nop());
        let err = linker.add_label("foo", span()).unwrap_err();
        assert!(matches!(err, AsmError::DuplicateLabel { .. }));
    }

    #[test]
    fn undefined_label_error() {
        let mut linker = Linker::new();
        linker.add_fragment(fixed(
            vec![0xE9, 0, 0, 0, 0],
            Some(Relocation {
                offset: 1,
                size: 4,
                label: alloc::rc::Rc::from("nowhere"),
                kind: RelocKind::X86Relative,
                addend: 0,
                trailing_bytes: 0,
            }),
        ));
        let err = linker.resolve().unwrap_err();
        assert!(matches!(err, AsmError::UndefinedLabel { .. }));
    }

    // ── Numeric labels ────────

    #[test]
    fn numeric_label_forward() {
        let mut linker = Linker::new();
        linker.add_fragment(fixed(
            vec![0xE9, 0, 0, 0, 0],
            Some(Relocation {
                offset: 1,
                size: 4,
                label: alloc::rc::Rc::from("1f"),
                kind: RelocKind::X86Relative,
                addend: 0,
                trailing_bytes: 0,
            }),
        ));
        linker.add_label("1", span()).unwrap();
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(&output[1..5], &[0, 0, 0, 0]);
    }

    #[test]
    fn numeric_label_backward() {
        let mut linker = Linker::new();
        linker.add_label("1", span()).unwrap();
        linker.add_fragment(nop());
        linker.add_fragment(fixed(
            vec![0xE9, 0, 0, 0, 0],
            Some(Relocation {
                offset: 1,
                size: 4,
                label: alloc::rc::Rc::from("1b"),
                kind: RelocKind::X86Relative,
                addend: 0,
                trailing_bytes: 0,
            }),
        ));

        let (output, _, _, _) = linker.resolve().unwrap();
        let rel = i32::from_le_bytes([output[2], output[3], output[4], output[5]]);
        assert_eq!(rel, -6);
    }

    // ── Branch relaxation ────────

    #[test]
    fn relaxation_short_jmp_forward() {
        let mut linker = Linker::new();
        linker.add_fragment(relaxable_jmp("target"));
        linker.add_label("target", span()).unwrap();
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        // Short form: EB 00 90
        assert_eq!(output, vec![0xEB, 0x00, 0x90]);
    }

    #[test]
    fn relaxation_short_jmp_backward() {
        let mut linker = Linker::new();
        linker.add_label("top", span()).unwrap();
        linker.add_fragment(nop());
        linker.add_fragment(relaxable_jmp("top"));

        let (output, _, _, _) = linker.resolve().unwrap();
        // top=0, nop@0 (1B), jmp_short@1 (2B), frag_end=3, disp=0-3=-3
        assert_eq!(output, vec![0x90, 0xEB, 0xFD]);
    }

    #[test]
    fn relaxation_promotes_jmp_to_long() {
        let mut linker = Linker::new();
        linker.add_fragment(relaxable_jmp("target"));
        linker.add_fragment(fixed(vec![0x90; 200], None));
        linker.add_label("target", span()).unwrap();
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output[0], 0xE9); // long form
        assert_eq!(output.len(), 5 + 200 + 1);
        let rel = i32::from_le_bytes([output[1], output[2], output[3], output[4]]);
        assert_eq!(rel, 200);
    }

    #[test]
    fn relaxation_short_jcc() {
        let mut linker = Linker::new();
        linker.add_fragment(relaxable_jcc(0x4, "done")); // je
        linker.add_label("done", span()).unwrap();
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output, vec![0x74, 0x00, 0x90]);
    }

    #[test]
    fn relaxation_promotes_jcc_to_long() {
        let mut linker = Linker::new();
        linker.add_fragment(relaxable_jcc(0x4, "done"));
        linker.add_fragment(fixed(vec![0x90; 200], None));
        linker.add_label("done", span()).unwrap();
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output[0], 0x0F);
        assert_eq!(output[1], 0x84);
        let rel = i32::from_le_bytes([output[2], output[3], output[4], output[5]]);
        assert_eq!(rel, 200);
    }

    #[test]
    fn relaxation_boundary_127() {
        // Exactly 127 bytes displacement: should stay short
        let mut linker = Linker::new();
        linker.add_fragment(relaxable_jmp("target"));
        linker.add_fragment(fixed(vec![0x90; 125], None)); // 2 + 125 = 127
        linker.add_label("target", span()).unwrap();
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output[0], 0xEB); // still short
        assert_eq!(output[1], 125u8); // disp = 127 - 2 = 125
    }

    #[test]
    fn relaxation_boundary_128() {
        // 128 bytes displacement: must go long
        // short jmp = 2B, 128 NOPs: target at 130, frag_end at 2, disp = 128 > 127 → promote
        let mut linker = Linker::new();
        linker.add_fragment(relaxable_jmp("target"));
        linker.add_fragment(fixed(vec![0x90; 128], None));
        linker.add_label("target", span()).unwrap();
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output[0], 0xE9); // promoted to long
    }

    #[test]
    fn cascading_relaxation() {
        // Two branches where expanding the second forces the first to expand too.
        let mut linker = Linker::new();

        // jmp L1 (2 or 5 bytes)
        linker.add_fragment(relaxable_jmp("L1"));
        // 125 NOPs
        linker.add_fragment(fixed(vec![0x90; 125], None));
        // jne L2 (2 or 6 bytes)
        linker.add_fragment(relaxable_jcc(0x5, "L2"));

        linker.add_label("L1", span()).unwrap();
        linker.add_fragment(fixed(vec![0x90; 130], None));
        linker.add_label("L2", span()).unwrap();
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        // Both should be long form due to cascading
        assert_eq!(output[0], 0xE9); // jmp rel32
        assert_eq!(output[5 + 125], 0x0F); // jne rel32
        assert_eq!(output[5 + 125 + 1], 0x85);
    }

    // ── Alignment ────────

    #[test]
    fn alignment_fragment() {
        let mut linker = Linker::new();
        linker.add_fragment(nop()); // 1 byte
        linker.add_alignment(4, 0x00, None, false, span());
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output, vec![0x90, 0x00, 0x00, 0x00, 0x90]);
    }

    #[test]
    fn alignment_already_aligned() {
        let mut linker = Linker::new();
        linker.add_fragment(fixed(vec![0x90; 4], None));
        linker.add_alignment(4, 0xCC, None, false, span());
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output, vec![0x90, 0x90, 0x90, 0x90, 0x90]);
    }

    #[test]
    fn alignment_with_base_address() {
        let mut linker = Linker::new();
        linker.set_base_address(0x1001); // base is 1 past alignment
        linker.add_alignment(4, 0xCC, None, false, span());
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        // Needs 3 bytes padding to reach 0x1004
        assert_eq!(output, vec![0xCC, 0xCC, 0xCC, 0x90]);
    }

    // ── Label table ────────

    #[test]
    fn label_table_exported() {
        let mut linker = Linker::new();
        linker.add_label("start", span()).unwrap();
        linker.add_fragment(nop());
        linker.add_fragment(nop());
        linker.add_label("end", span()).unwrap();
        linker.add_fragment(nop());

        let (_, labels, _, _) = linker.resolve().unwrap();
        let m: BTreeMap<String, u64> = labels.into_iter().collect();
        assert_eq!(m["start"], 0);
        assert_eq!(m["end"], 2);
    }

    #[test]
    fn label_table_with_base_address() {
        let mut linker = Linker::new();
        linker.set_base_address(0x1000);
        linker.add_label("func", span()).unwrap();
        linker.add_fragment(fixed(vec![0x90; 10], None));

        let (_, labels, _, _) = linker.resolve().unwrap();
        assert_eq!(labels[0].1, 0x1000);
    }

    // ── Misc ────────

    #[test]
    fn multiple_fragments_no_reloc() {
        let mut linker = Linker::new();
        linker.add_fragment(nop());
        linker.add_fragment(fixed(vec![0xCC], None));
        linker.add_fragment(fixed(vec![0xC3], None));
        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output, vec![0x90, 0xCC, 0xC3]);
    }

    #[test]
    fn empty_linker() {
        let mut linker = Linker::new();
        let (output, labels, _, _) = linker.resolve().unwrap();
        assert!(output.is_empty());
        assert!(labels.is_empty());
    }

    #[test]
    fn relocation_with_addend() {
        let mut linker = Linker::new();
        linker.add_label("data", span()).unwrap();
        linker.add_fragment(fixed(vec![0; 16], None));
        linker.add_fragment(fixed(
            vec![0x48, 0x8D, 0x05, 0, 0, 0, 0],
            Some(Relocation {
                offset: 3,
                size: 4,
                label: alloc::rc::Rc::from("data"),
                kind: RelocKind::X86Relative,
                addend: 4,
                trailing_bytes: 0,
            }),
        ));

        let (output, _, _, _) = linker.resolve().unwrap();
        let rel = i32::from_le_bytes([output[19], output[20], output[21], output[22]]);
        assert_eq!(rel, -19);
    }

    // ── Relaxation + alignment interaction ────────

    #[test]
    fn relaxation_with_alignment() {
        let mut linker = Linker::new();
        linker.add_label("top", span()).unwrap();
        linker.add_fragment(nop()); // 1 byte
        linker.add_alignment(16, 0xCC, None, false, span()); // pad to 16
                                                             // jne top — after alignment, offset = 16, so disp = 0-18 = -18 for short (or -22 for long)
        linker.add_fragment(relaxable_jcc(0x5, "top"));

        let (output, _, _, _) = linker.resolve().unwrap();
        // 1 NOP + 15 padding = 16 bytes. With short jne (2B): disp = 0 - 18 = -18, fits in rel8.
        assert_eq!(output[0], 0x90);
        assert_eq!(output[16], 0x75); // short jne
        let disp = output[17] as i8;
        assert_eq!(disp, -18);
    }

    // ── add_encoded helper ────────

    #[test]
    fn add_encoded_creates_relaxable() {
        let mut linker = Linker::new();
        linker
            .add_encoded(
                InstrBytes::from_slice(&[0xE9, 0, 0, 0, 0]),
                Some(Relocation {
                    offset: 1,
                    size: 4,
                    label: alloc::rc::Rc::from("target"),
                    kind: RelocKind::X86Relative,
                    addend: 0,
                    trailing_bytes: 0,
                }),
                Some(RelaxInfo {
                    short_bytes: InstrBytes::from_slice(&[0xEB, 0x00]),
                    short_reloc_offset: 1,
                    short_relocation: None,
                }),
                span(),
            )
            .unwrap();
        linker.add_label("target", span()).unwrap();
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output, vec![0xEB, 0x00, 0x90]);
    }

    #[test]
    fn add_encoded_creates_fixed() {
        let mut linker = Linker::new();
        linker
            .add_encoded(InstrBytes::from_slice(&[0x90]), None, None, span())
            .unwrap();

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output, vec![0x90]);
    }

    // ── Multi-byte NOP alignment ────────

    #[test]
    fn alignment_with_nop_padding() {
        let mut linker = Linker::new();
        linker.add_fragment(nop()); // 1 byte at offset 0
                                    // Align to 4 bytes with NOP padding (use_nop=true)
        linker.add_alignment(4, 0x00, None, true, span());
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        // 1 NOP + 3-byte NOP + 1 NOP = 5 bytes
        assert_eq!(output.len(), 5);
        assert_eq!(output[0], 0x90); // 1-byte NOP
                                     // Next 3 bytes should be the Intel 3-byte NOP: 0F 1F 00
        assert_eq!(&output[1..4], &[0x0F, 0x1F, 0x00]);
        assert_eq!(output[4], 0x90); // final NOP
    }

    #[test]
    fn alignment_nop_padding_large() {
        let mut linker = Linker::new();
        linker.add_fragment(nop()); // 1 byte at offset 0
                                    // Align to 16 bytes with NOP padding
        linker.add_alignment(16, 0x00, None, true, span());
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        // 1 + 15 padding + 1 = 17 bytes
        assert_eq!(output.len(), 17);
        assert_eq!(output[0], 0x90);
        // 15 bytes of NOP padding: 9-byte NOP + 6-byte NOP
        assert_eq!(
            &output[1..10],
            &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        assert_eq!(&output[10..16], &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00]);
        assert_eq!(output[16], 0x90);
    }

    #[test]
    fn alignment_max_skip_respected() {
        let mut linker = Linker::new();
        linker.add_fragment(nop()); // 1 byte at offset 0
                                    // Align to 16, but max_skip = 2 — padding needed is 15, which exceeds 2
        linker.add_alignment(16, 0x00, Some(2), false, span());
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        // Alignment should be skipped — just 2 NOPs
        assert_eq!(output, vec![0x90, 0x90]);
    }

    #[test]
    fn alignment_max_skip_allows_small_padding() {
        let mut linker = Linker::new();
        linker.add_fragment(fixed(vec![0x90; 3], None)); // 3 bytes
                                                         // Align to 4 with max_skip = 2 — padding needed is 1, which is ≤ 2
        linker.add_alignment(4, 0xCC, Some(2), false, span());
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output, vec![0x90, 0x90, 0x90, 0xCC, 0x90]);
    }

    // ── .org directive ────────

    #[test]
    fn org_forward_padding() {
        let mut linker = Linker::new();
        linker.set_base_address(0x100);
        linker.add_fragment(nop()); // 1 byte at 0x100
        linker.add_org(0x110, 0x00, span()); // advance to 0x110
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        // 1 NOP + 15 zero-fill + 1 NOP = 17 bytes
        assert_eq!(output.len(), 17);
        assert_eq!(output[0], 0x90);
        // 15 zero bytes
        assert!(output[1..16].iter().all(|&b| b == 0x00));
        assert_eq!(output[16], 0x90);
    }

    #[test]
    fn org_already_at_target() {
        let mut linker = Linker::new();
        linker.set_base_address(0x100);
        linker.add_fragment(fixed(vec![0x90; 16], None)); // exactly at 0x110
        linker.add_org(0x110, 0x00, span()); // already there
        linker.add_fragment(nop());

        let (output, _, _, _) = linker.resolve().unwrap();
        assert_eq!(output.len(), 17); // 16 + 0 padding + 1
    }

    #[test]
    fn org_backward_error() {
        let mut linker = Linker::new();
        linker.set_base_address(0x200);
        linker.add_fragment(fixed(vec![0x90; 16], None)); // at 0x210
        linker.add_org(0x100, 0x00, span()); // behind!

        let err = linker.resolve().unwrap_err();
        assert!(matches!(err, AsmError::Syntax { .. }));
    }

    #[test]
    fn org_with_labels() {
        let mut linker = Linker::new();
        linker.set_base_address(0x1000);
        linker.add_fragment(nop());
        linker.add_org(0x1010, 0x00, span());
        linker.add_label("after_org", span()).unwrap();
        linker.add_fragment(nop());

        let (_, labels, _, _) = linker.resolve().unwrap();
        let m: BTreeMap<String, u64> = labels.into_iter().collect();
        assert_eq!(m["after_org"], 0x1010);
    }

    // === 8th Audit: Branch relaxation with addend ===

    #[test]
    fn relaxable_jmp_with_positive_addend() {
        // target: (offset 0)
        //   nop   (1 byte)
        //   jmp target+1  (short: EB xx, 2 bytes)
        // frag_end = 1 + 2 = 3.  target=0.  addend=1.
        // disp = 0 + 1 - 3 = -2  → short fits, should encode [EB FE]
        let mut linker = Linker::new();
        linker.add_label("target", span()).unwrap();
        linker.add_fragment(nop());
        linker.add_fragment(Fragment::Relaxable {
            short_bytes: InstrBytes::from_slice(&[0xEB, 0x00]),
            short_reloc_offset: 1,
            short_relocation: None,
            long_bytes: InstrBytes::from_slice(&[0xE9, 0, 0, 0, 0]),
            long_relocation: Relocation {
                offset: 1,
                size: 4,
                label: alloc::rc::Rc::from("target"),
                kind: RelocKind::X86Relative,
                addend: 1,
                trailing_bytes: 0,
            },
            is_long: false,
            span: span(),
        });

        let (output, _, _, _) = linker.resolve().unwrap();
        // nop + jmp target+1 → [0x90, 0xEB, 0xFE]
        assert_eq!(output, vec![0x90, 0xEB, 0xFE_u8]); // -2 as i8 = 0xFE
    }

    #[test]
    fn relaxable_jmp_addend_forces_long_form() {
        // target: (offset 0)
        //   .space 126  (126 bytes of NOP)
        //   jmp target - 200  (addend = -200)
        // frag_end = 126 + 2 = 128.  target=0.  addend=-200.
        // disp = 0 + (-200) - 128 = -328  → doesn't fit rel8, must use long form
        let mut linker = Linker::new();
        linker.add_label("target", span()).unwrap();
        // Add 126 bytes of padding
        linker.add_fragment(fixed(vec![0x90; 126], None));
        linker.add_fragment(Fragment::Relaxable {
            short_bytes: InstrBytes::from_slice(&[0xEB, 0x00]),
            short_reloc_offset: 1,
            short_relocation: None,
            long_bytes: InstrBytes::from_slice(&[0xE9, 0, 0, 0, 0]),
            long_relocation: Relocation {
                offset: 1,
                size: 4,
                label: alloc::rc::Rc::from("target"),
                kind: RelocKind::X86Relative,
                addend: -200,
                trailing_bytes: 0,
            },
            is_long: false,
            span: span(),
        });

        let (output, _, _, _) = linker.resolve().unwrap();
        // Should have been promoted to long form: 0xE9 + 4 bytes
        assert_eq!(output.len(), 126 + 5); // 126 nops + 5-byte jmp
                                           // Long form: disp32 = target + addend - frag_end = 0 + (-200) - (126+5) = -331
        let disp = i32::from_le_bytes([output[127], output[128], output[129], output[130]]);
        assert_eq!(disp, -331);
    }
}
