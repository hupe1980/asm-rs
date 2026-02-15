---
layout: default
title: Architecture
---

# Architecture Guide

Deep dive into the asm-rs assembler pipeline, module responsibilities,
encoding architecture, and testing strategy.

---

## Pipeline Overview

```
Source Text
     │
     ▼
┌──────────────┐
│ Preprocessor │  Expands macros, loops, conditionals
└──────┬───────┘  (.macro/.rept/.irp/.if)
       │
       ▼
┌─────────┐
│  Lexer  │  Zero-copy tokenization into Token<'src>
└────┬────┘  with source spans for error reporting
     │
     ▼
┌─────────┐
│  Parser  │  Produces intermediate representation (IR)
└────┬─────┘  from token stream using Intel syntax rules
     │
     ▼
┌───────────┐
│ Optimizer │  Peephole optimizations (zero-idiom, MOV narrowing,
└─────┬─────┘  AND→TEST conversion) when OptLevel::Size is active
      │
      ▼
┌──────────┐
│ Encoder  │  Translates IR instructions into machine code
└────┬─────┘  bytes with relocations and relax info
     │
     ▼
┌──────────┐
│  Linker  │  Resolves labels, relaxes branches (Szymanski),
└────┬─────┘  applies relocations, produces final output
     │
     ▼
  Output Bytes + Labels + Applied Relocations
```

---

## Module Responsibilities

### Preprocessor

The preprocessor operates on raw source text **before** the lexer. It performs
text-level expansion of macros, loops, and conditional assembly directives.

**Macro definitions** (`.macro` / `.endm`):
- Named parameters with positional substitution (`\param`)
- Default parameter values (`.macro name reg=rax`)
- Variadic parameters (`:vararg`) for collecting remaining arguments
- Unique label generation via `\@` counter
- Recursive expansion with bounded depth (256 levels)

**Repeat loops**:
- `.rept count` / `.endr` — repeat body `count` times
- `.irp symbol, value1, value2, ...` / `.endr` — iterate substituting symbol
- `.irpc symbol, string` / `.endr` — iterate over characters
- Nestable across directive types
- Bounded total iterations (100,000) to prevent runaway expansion

**Conditional assembly**:
- `.if expression` / `.else` / `.elseif expression` / `.endif`
- `.ifdef symbol` / `.ifndef symbol`
- `defined(symbol)` function in expressions

**Expression evaluator** (recursive-descent):
- Full C-like operator precedence across 12 levels
- Operators: `||`, `&&`, `|`, `^`, `&`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `<<`, `>>`, `+`, `-`, `*`, `/`, `%`, unary `!`, `-`, `~`
- Numeric literals: decimal, `0x` hex, `0b` binary, `0o` octal, character `'A'`
- Wrapping arithmetic, division-by-zero returns 0

**Design decisions**:
- Text-level processing avoids coupling to the parser grammar
- Expansion is re-entrant: expanded text is re-processed to handle nesting
- Integrated into `Assembler.emit()` — transparent to callers

---

### Lexer

The lexer performs zero-copy tokenization of assembly source text. It produces
a `Vec<Token<'src>>` where each token carries:

- **Kind** — `Ident`, `Number`, `Directive`, `LabelDef`, `Comma`, etc.
- **Text** — `Cow<'src, str>` borrowing directly from the source string (zero
  allocation for identifiers, numbers, directives, and punctuation; only
  string/char literals and numeric labels allocate)
- **Span** — line, column, byte offset, and length for error reporting

**Design decisions**:
- Numbers parsed eagerly and stored as `i128` in the token
- Case-insensitive detection uses `eq_ignore_ascii_case()` (zero allocation)
- Semicolons are statement separators; hash (`#`) starts comments
- Numeric labels (`1:`, `1b`, `1f`) recognized at lexer level
- `#[inline]` on hot helpers: `parse_number_at`, `hex_digit`

---

### Parser

The parser consumes `&[Token<'_>]` and produces `Vec<Statement>`. It handles:

- **Instructions** with prefixes, size hints, and up to 3 operands
- **Memory operands** with full SIB addressing: `[base + index*scale + disp]`  
  (boxed via `Box<MemoryOperand>` to shrink `Operand` enum)
- **Labels** (named and numeric)
- **Directives** (`.byte`, `.word`, `.equ`, `.align`, `.fill`, `.syntax`, etc.)
- **Segment overrides** (`fs:[rax]`, `%fs:0x28(%rax)`)
- **AT&T / GAS syntax** via `parse_with_syntax()`

The parser is a simple recursive-descent parser producing flat IR statements.
`Instruction` fields are fully stack-allocated: `Mnemonic` (inline `[u8; 24]`),
`OperandList` (inline `[Operand; 6]`), `PrefixList` (inline `[Prefix; 4]`) —
yielding **zero heap allocations** per instruction.

#### Constant Expression Evaluator

| Precedence | Operators | Associativity |
|:----------:|:----------|:-------------:|
| 1 (lowest) | `\|` (bitwise OR) | Left |
| 2 | `^` (bitwise XOR) | Left |
| 3 | `&` (bitwise AND) | Left |
| 4 | `<<`, `>>` (shift) | Left |
| 5 | `+`, `-` (add/sub) | Left |
| 6 | `*`, `/`, `%` (mul/div/mod) | Left |
| 7 | unary `-`, `~` (negate, NOT) | Right |
| 8 (highest) | atoms: numbers, constants, `(expr)` | — |

#### AT&T / GAS Syntax Support

When `Syntax::Att` is active, the parser switches to AT&T operand parsing:

| Feature | AT&T Syntax | Intel Equivalent |
|:--------|:------------|:-----------------|
| Register prefix | `%rax`, `%eax` | `rax`, `eax` |
| Immediate prefix | `$42`, `$0xFF` | `42`, `0xFF` |
| Operand order | `movq $1, %rax` (src, dst) | `mov rax, 1` (dst, src) |
| Memory | `disp(%base, %index, scale)` | `[base + index*scale + disp]` |
| Segment override | `%fs:0x28(%rax)` | `fs:[rax + 0x28]` |
| Mnemonic suffix | `movq`, `addl`, `movb` | Size from operands |
| Indirect | `*%rax`, `*(%rax)` | `rax`, `[rax]` |

**Mnemonic translations**: `movzbl`→`movzx`, `movsbl`→`movsx`, `movslq`→`movsxd`,
`cltq`→`cdqe`, `cqto`→`cqo`, etc.

---

### Optimizer

The peephole optimizer runs after parsing, before encoding.
It transforms individual instructions for shorter machine code when
`OptLevel::Size` is active.

| Pattern | Replacement | Savings |
|:--------|:------------|:--------|
| `mov reg64, 0` / `mov reg32, 0` | `xor reg32, reg32` | 5–7 → 2 bytes |
| `mov reg64, small_imm` (0 < imm ≤ u32::MAX) | `mov reg32, imm32` | 7 → 5 bytes |
| `and reg64, u32_imm` | `and reg32, u32_imm` | Saves 1 byte (REX removed) |
| `and reg, reg` (same register) | `test reg, reg` | Equivalent, better for flags |

**Design**: operates on IR `Statement` level, each optimization is a pure function
returning `Option<Statement>`, easy to extend.

---

### Encoder

The encoder translates one `Instruction` into machine code bytes.

**x86-64 encoding**:
- REX prefix construction (W, R, X, B bits)
- ModR/M byte encoding for register and memory operands
- SIB byte construction for scaled-index addressing
- Displacement and immediate encoding (8, 16, 32, or 64 bits)
- Relocation records for label references
- Relax info for branch instructions (short/long forms)
- Legacy register encoding (AL/AH, SPL requiring REX)
- Extended registers (R8–R15, XMM8–XMM15)
- 16-bit operand prefix (0x66) and 0x67 address-size override
- LOCK prefix validation, push/pop size validation
- CMOVcc operand size validation
- LOOP/JECXZ/JRCXZ automatic relaxation

**Zero-allocation design**:
- `InstrBytes` stack-allocated `[u8; 32]` replaces per-instruction `Vec<u8>`
- `FragmentBytes::Inline(InstrBytes)` for instructions; `Heap(Vec<u8>)` only for data
- `Operand::Memory(Box<MemoryOperand>)` — boxed to shrink `Operand` from 56 → ~32 bytes
- Lazy listing annotations: zero cost when `enable_listing()` is not called

---

### Unified x86 Dispatch

The x86 module covers **~725 mnemonics** across these encoding classes:

| Class | Examples |
|:------|:---------|
| Fixed encoding (85) | Zero-operand instructions via const sorted table + binary search |
| ALU class (8) | `ADD`/`OR`/`ADC`/`SBB`/`AND`/`SUB`/`XOR`/`CMP` |
| Unary class | `NOT`/`NEG`/`MUL`/`DIV`/`IDIV` |
| Shift class | `SHL`/`SHR`/`SAR`/`ROL`/`ROR`/`RCL`/`RCR` |
| Condition-code class | All 16 conditions → `Jcc`, `SETcc`, `CMOVcc` |
| SSE/SSE2/SSE3/SSSE3/SSE4 | 100+ SIMD instructions |
| AVX/AVX2 (VEX) | 300+ instructions with FMA3, permutes, broadcasts |
| AVX-512 (EVEX) | 120+ instructions with ZMM0-ZMM31 |
| BMI1/BMI2, ADX, TSX | Specialized extensions |

---

### ARM32 Encoder

Key features:
- All 16 data processing opcodes with barrel shifter
- Load/store with immediate offset, register offset, pre/post-index
- Block transfer (PUSH/POP/LDM/STM with addressing modes)
- MOVW/MOVT fallback for large immediates
- Bitfield operations (BFC/BFI/SBFX/UBFX)
- Literal pools with ±4,092 byte range

---

### Thumb / Thumb-2 Encoder

Shares `arm.rs` with ARM32 encoder:
- 16-bit and 32-bit instruction encoding
- IT block generation with T/E mask
- Branch relaxation: narrow (2B) ↔ wide (4B)
- `.thumb`/`.arm` mode switching, `.thumb_func` LSB

---

### AArch64 Encoder

Key features:
- Bitmask immediate encoding (N:immr:imms scheme)
- Conditional select and compare instructions
- LSE atomics with ordering variants
- NEON/AdvSIMD vector instructions with arrangement parsing
- Literal pool support with deduplication
- Branch relaxation for B.cond, CBZ/CBNZ, TBZ/TBNZ, ADR

---

### RISC-V Encoder

Key features:
- Full RV32I/RV64I base ISA + M, A, F, D extensions
- C extension (16-bit compressed) with all 9 formats
- 64-bit `li` decomposition for large constants
- Auto-narrowing via `.option rvc`/`.option norvc`
- Branch relaxation: B-type → inverted-B + JAL
- ~50 named CSRs, privileged instructions

---

### Linker

The linker collects encoded fragments and resolves references:

**Fragment model**:
- `Fragment::Fixed` — fixed-size data
- `Fragment::Align` — dynamic alignment padding (multi-byte NOP for x86)
- `Fragment::Relaxable` — branches with short/long forms
- `Fragment::Org` — advance location counter

**Szymanski's branch relaxation**:
1. All relaxable branches start as short form
2. Each iteration checks displacements
3. Out-of-range branches promoted to long form
4. Growth is monotonic — once promoted, never shrinks
5. Guaranteed convergence (max 100 iterations)

**Relocations** — 17 relocation kinds across all architectures:

| Kind | Architecture | Description |
|:-----|:-------------|:------------|
| `X86Relative` | x86 | RIP-relative displacement |
| `Absolute` | All | Raw LE byte write (1–8 bytes) |
| `ArmBranch24` | ARM32 | B/BL 24-bit offset |
| `ArmLdrLit` | ARM32 | LDR literal 12-bit |
| `Aarch64Jump26` | AArch64 | B/BL 26-bit offset |
| `Aarch64Branch19` | AArch64 | B.cond/CBZ/CBNZ |
| `Aarch64Branch14` | AArch64 | TBZ/TBNZ |
| `RvJal20` | RISC-V | JAL 21-bit J-type |
| `RvBranch12` | RISC-V | B-type 13-bit |
| `RvAuipc20` | RISC-V | AUIPC+JALR pair |
| `ThumbBranch8/11` | Thumb | 16-bit branches |
| `ThumbBl/BranchW` | Thumb | 32-bit branches |

---

## Encoding Format Reference

### x86-64 Instruction Format

```
┌────────┬────┬───────┬─────┬──────────────┬───────────┐
│ Prefix │REX │Opcode │ModRM│     SIB      │Disp/Imm   │
│ (opt)  │opt │1-3 B  │(opt)│    (opt)     │  (opt)    │
└────────┴────┴───────┴─────┴──────────────┴───────────┘
```

### REX Prefix (0x40–0x4F)

```
  0  1  0  0  W  R  X  B
                │  │  │  └─ Extension of ModRM.rm or SIB.base
                │  │  └──── Extension of SIB.index
                │  └─────── Extension of ModRM.reg
                └────────── 64-bit operand size
```

### ModR/M Byte

```
  mod   reg/opcode   r/m
  [7:6]   [5:3]     [2:0]
```

- `mod=00`: `[r/m]` (no displacement)
- `mod=01`: `[r/m + disp8]`
- `mod=10`: `[r/m + disp32]`
- `mod=11`: register direct

### SIB Byte

```
  scale   index   base
  [7:6]   [5:3]   [2:0]
```

Required when using RSP/R12 as base or scaled index addressing.

---

## Testing Strategy

The project maintains an extensive test suite across multiple categories:

- **Unit tests** — per-module tests for all components
- **Integration tests** — end-to-end through the public API
- **Cross-validation** — multi-decoder verification (iced-x86, yaxpeax-arm, bad64, riscv-decode)
- **Property-based** — proptest: no-panic, determinism, length bounds
- **Fuzz testing** — `cargo-fuzz` targets for all architectures
- **Code coverage** — 80% minimum threshold enforced in CI

Zero warnings, zero clippy warnings, Miri clean.
