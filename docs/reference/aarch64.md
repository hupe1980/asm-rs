---
layout: default
title: AArch64
parent: ISA Reference
nav_order: 2
---

# AArch64 (ARM64) Instruction Reference
{: .fs-8 }

Complete reference for all AArch64 instructions supported by asm-rs,
including NEON (AdvSIMD) and SVE extensions.
{: .fs-5 .fw-300 }

---

## Notation

| Symbol | Meaning |
|:-------|:--------|
| `Wd` / `Wn` / `Wm` | 32-bit general-purpose register (W0–W30, WZR) |
| `Xd` / `Xn` / `Xm` | 64-bit general-purpose register (X0–X30, XZR, SP) |
| `Vd.T` | NEON vector register with arrangement (e.g. `V0.4S`) |
| `Zd.T` | SVE scalable vector register (Z0–Z31) |
| `Pd.T` | SVE predicate register (P0–P15) |
| `#imm` | Immediate value |
| `label` | Label reference |
| `cond` | Condition code (eq, ne, cs, cc, mi, pl, vs, vc, hi, ls, ge, lt, gt, le, al) |
| `SP` | Stack pointer (X31 context-dependent) |
| `LR` | Link register (X30) |
| `XZR` / `WZR` | Zero register (reads as 0, writes discarded) |

---

## Hint / System

```asm
nop                            // No operation
wfi                            // Wait for interrupt
wfe                            // Wait for event
sev                            // Send event
sevl                           // Send event local
yield                          // Yield hint
svc   #0                      // Supervisor call
brk   #0                      // Breakpoint
hlt   #0                      // Halt

mrs   x0, NZCV                // Read system register → GPR
msr   NZCV, x0                // Write GPR → system register

dmb   sy                      // Data memory barrier
dsb   sy                      // Data synchronization barrier
isb                            // Instruction synchronization barrier
```

Barrier options: `sy`, `ish`, `ishld`, `ishst`, `nsh`, `nshld`, `nshst`, `osh`, `oshld`, `oshst`.

---

## Branch

```asm
b     label                   // Branch (unconditional, ±128 MB)
bl    label                   // Branch and link (±128 MB)
br    x0                      // Branch to register
blr   x0                      // Branch and link to register
ret                            // Return (branch to LR)
ret   x1                      // Return via specified register
```

### Conditional Branch

All 16 condition codes:

```asm
b.eq  label                   // Branch if equal (Z=1)
b.ne  label                   // Branch if not equal
b.cs  label                   // Branch if carry set (unsigned >=)
b.cc  label                   // Branch if carry clear (unsigned <)
b.mi  label                   // Branch if minus (negative)
b.pl  label                   // Branch if plus (positive/zero)
b.vs  label                   // Branch if overflow
b.vc  label                   // Branch if no overflow
b.hi  label                   // Branch if unsigned >
b.ls  label                   // Branch if unsigned <=
b.ge  label                   // Branch if signed >=
b.lt  label                   // Branch if signed <
b.gt  label                   // Branch if signed >
b.le  label                   // Branch if signed <=
b.al  label                   // Branch always
```

### Compare & Branch

```asm
cbz   x0, label               // Branch if register is zero
cbnz  x0, label               // Branch if register is not zero
cbz   w0, label               // 32-bit register form
tbz   x0, #31, label          // Test bit and branch if zero
tbnz  x0, #0, label           // Test bit and branch if not zero
```

{: .note }
> `B.cond`, `CBZ`/`CBNZ`, and `TBZ`/`TBNZ` support automatic branch relaxation:
> instructions widen to inverted-condition + unconditional `B` pairs when the target
> exceeds the direct offset range.

### PC-Relative

```asm
adr   x0, label               // Load PC-relative address (±1 MB)
adrp  x0, label               // Load page address (±4 GB)
```

{: .note }
> `ADR` relaxes to `ADRP` + `ADD` pair (8 bytes, ±4 GB) when target exceeds ±1 MB.

---

## Data Processing — Immediate

```asm
add   x0, x1, #42             // Add immediate (12-bit, optional shift)
sub   x0, x1, #42             // Subtract immediate
adds  x0, x1, #42             // Add and set flags
subs  x0, x1, #42             // Subtract and set flags
cmp   x0, #42                 // Compare (alias: subs xzr, x0, #42)
cmn   x0, #42                 // Compare negative (alias: adds xzr, x0, #42)
```

## Data Processing — Register

```asm
add   x0, x1, x2              // Add registers
sub   x0, x1, x2              // Subtract registers
add   x0, x1, x2, lsl #3     // Add with shifted register
neg   x0, x1                  // Negate (alias: sub x0, xzr, x1)
```

## Multiply / Divide

```asm
mul   x0, x1, x2              // Multiply
mneg  x0, x1, x2              // Multiply-negate
madd  x0, x1, x2, x3          // Multiply-add (x0 = x1*x2 + x3)
msub  x0, x1, x2, x3          // Multiply-subtract
smull x0, w1, w2               // Signed multiply long (32→64)
umull x0, w1, w2               // Unsigned multiply long
sdiv  x0, x1, x2              // Signed divide
udiv  x0, x1, x2              // Unsigned divide
```

---

## Logical — Register

```asm
and   x0, x1, x2              // Bitwise AND
orr   x0, x1, x2              // Bitwise OR
eor   x0, x1, x2              // Bitwise XOR
bic   x0, x1, x2              // Bit clear (AND NOT)
ands  x0, x1, x2              // AND and set flags
bics  x0, x1, x2              // Bit clear and set flags
orn   x0, x1, x2              // OR NOT
eon   x0, x1, x2              // XOR NOT
tst   x0, x1                  // Test (alias: ands xzr, x0, x1)
mvn   x0, x1                  // Bitwise NOT (alias: orn x0, xzr, x1)
```

## Logical — Immediate

```asm
and   x0, x1, #0xFF           // AND with bitmask immediate
orr   x0, x1, #0xFF00         // OR with bitmask immediate
eor   x0, x1, #0xF0F0         // XOR with bitmask immediate
ands  x0, x1, #0xFF           // AND and set flags
tst   x0, #0xFF               // Test with bitmask immediate
```

{: .note }
> AArch64 bitmask immediates use a special N:immr:imms encoding scheme
> that allows a wide variety of repeating bit patterns. Not all arbitrary
> 64-bit constants can be encoded as bitmask immediates.

---

## Move

```asm
mov   x0, x1                  // Move register
mov   x0, #42                 // Move immediate (auto-selects MOVZ/MOVN/ORR)
movz  x0, #0x1234             // Move wide with zero
movz  x0, #0x5678, lsl #16   // Move wide with shift
movn  x0, #0                  // Move wide negative
movk  x0, #0xABCD, lsl #32   // Move wide keep (insert 16-bit slice)
```

---

## Shift

```asm
lsl   x0, x1, #4              // Logical shift left (immediate)
lsr   x0, x1, #4              // Logical shift right
asr   x0, x1, #4              // Arithmetic shift right
ror   x0, x1, #4              // Rotate right
lsl   x0, x1, x2              // Shift by register
```

---

## Bit Manipulation

```asm
clz   x0, x1                  // Count leading zeros
cls   x0, x1                  // Count leading sign bits
rbit  x0, x1                  // Reverse bits
rev   x0, x1                  // Reverse bytes (64-bit)
rev16 x0, x1                  // Reverse bytes in each 16-bit halfword
rev32 x0, x1                  // Reverse bytes in each 32-bit word
```

---

## Extend

```asm
uxtb  w0, w1                  // Unsigned extend byte (8 → 32)
uxth  w0, w1                  // Unsigned extend halfword (16 → 32)
sxtb  x0, w1                  // Signed extend byte (8 → 64)
sxth  x0, w1                  // Signed extend halfword (16 → 64)
sxtw  x0, w1                  // Signed extend word (32 → 64)
```

---

## Conditional Select

```asm
csel  x0, x1, x2, eq          // Select x1 if EQ, else x2
csinc x0, x1, x2, ne          // Select x1 if NE, else x2+1
csinv x0, x1, x2, lt          // Select x1 if LT, else ~x2
csneg x0, x1, x2, ge          // Select x1 if GE, else -x2

// Aliases
cset  x0, eq                  // Set 1 if EQ, else 0
csetm x0, eq                  // Set -1 if EQ, else 0
cinc  x0, x1, ne              // Increment x1 if NE
cneg  x0, x1, ne              // Negate x1 if NE
```

---

## Conditional Compare

```asm
ccmp  x0, x1, #0, eq          // Compare x0,x1 if EQ; else set nzcv=#0
ccmp  x0, #5, #0, ne          // Compare x0,#5 if NE; else set nzcv=#0
ccmn  x0, x1, #0, eq          // Compare negative if EQ
ccmn  x0, #5, #0, ne          // Compare negative with immediate if NE
```

---

## Bitfield

```asm
bfm   x0, x1, #immr, #imms   // Bitfield move
ubfm  x0, x1, #immr, #imms   // Unsigned bitfield move
sbfm  x0, x1, #immr, #imms   // Signed bitfield move
bfi   x0, x1, #lsb, #width   // Bitfield insert
bfxil x0, x1, #lsb, #width   // Bitfield extract and insert low
ubfx  x0, x1, #lsb, #width   // Unsigned bitfield extract
sbfx  x0, x1, #lsb, #width   // Signed bitfield extract
ubfiz x0, x1, #lsb, #width   // Unsigned bitfield insert in zero
sbfiz x0, x1, #lsb, #width   // Signed bitfield insert in zero
extr  x0, x1, x2, #lsb       // Extract from pair of registers
```

---

## Load / Store

### Basic Forms

```asm
ldr   x0, [x1]                // Load 64-bit
ldr   w0, [x1]                // Load 32-bit
ldrb  w0, [x1]                // Load byte (zero-extend)
ldrh  w0, [x1]                // Load halfword (zero-extend)
ldrsb x0, [x1]                // Load byte (sign-extend to 64)
ldrsh x0, [x1]                // Load halfword (sign-extend to 64)
ldrsw x0, [x1]                // Load word (sign-extend to 64)
str   x0, [x1]                // Store 64-bit
str   w0, [x1]                // Store 32-bit
strb  w0, [x1]                // Store byte
strh  w0, [x1]                // Store halfword
```

### Offset Addressing

```asm
ldr   x0, [x1, #8]            // Unsigned scaled offset
ldr   x0, [x1, x2]            // Register offset
ldur  x0, [x1, #-16]          // Unscaled offset (±256)
stur  x0, [x1, #-16]          // Unscaled store
```

### Pre-Index / Post-Index

```asm
ldr   x0, [x1, #16]!          // Pre-index: x1 += 16, then load
str   x0, [x1, #-8]!          // Pre-index store
ldr   x0, [x1], #16           // Post-index: load, then x1 += 16
str   x0, [x1], #-8           // Post-index store
```

### Load/Store Pair

```asm
stp   x0, x1, [sp, #-16]!     // Store pair (pre-index)
ldp   x0, x1, [sp], #16       // Load pair (post-index)
stp   x0, x1, [sp, #16]       // Store pair (offset)
ldp   x0, x1, [sp, #16]       // Load pair (offset)
```

### Label-Relative Load

```asm
ldr   x0, label               // Load from PC-relative label
ldr   x0, =0xDEADBEEF         // Load from literal pool
ldr   w0, =42                 // 32-bit literal pool load
```

---

## Atomics (LSE)

### Load-Op

```asm
ldadd   x0, x1, [x2]          // Atomic load-add
ldclr   x0, x1, [x2]          // Atomic load-clear (AND NOT)
ldset   x0, x1, [x2]          // Atomic load-set (OR)
ldeor   x0, x1, [x2]          // Atomic load-XOR
swp     x0, x1, [x2]          // Atomic swap
cas     x0, x1, [x2]          // Compare and swap
```

### Ordering Variants

Each operation supports ordering suffixes:

```asm
ldadda  x0, x1, [x2]          // Acquire
ldaddl  x0, x1, [x2]          // Release
ldaddal x0, x1, [x2]          // Acquire-release
ldaddb  w0, w1, [x2]          // Byte
ldaddh  w0, w1, [x2]          // Halfword
ldaddab w0, w1, [x2]          // Byte + acquire
```

### Store-Only Variants

```asm
stadd   x0, [x1]              // Atomic store-add (no return)
stclr   x0, [x1]              // Atomic store-clear
stset   x0, [x1]              // Atomic store-set
steor   x0, [x1]              // Atomic store-XOR
```

---

## Exclusive Access

```asm
ldxr    x0, [x1]              // Load exclusive
ldaxr   x0, [x1]              // Load-acquire exclusive
stxr    w2, x0, [x1]          // Store exclusive (w2 = success)
stlxr   w2, x0, [x1]          // Store-release exclusive
# Byte/halfword variants: ldxrb, ldaxrb, stxrb, stlxrb, ldxrh, ...
```

---

## Load-Acquire / Store-Release

```asm
ldar    x0, [x1]              // Load-acquire (C11 atomic_load)
ldarb   w0, [x1]              // Load-acquire byte
ldarh   w0, [x1]              // Load-acquire halfword
stlr    x0, [x1]              // Store-release (C11 atomic_store)
stlrb   w0, [x1]              // Store-release byte
stlrh   w0, [x1]              // Store-release halfword
```

---

## NEON (AdvSIMD)

### Vector Arithmetic

```asm
add   v0.4s, v1.4s, v2.4s     // Vector add (4x 32-bit)
sub   v0.4s, v1.4s, v2.4s     // Vector subtract
mul   v0.4s, v1.4s, v2.4s     // Vector multiply
```

### Vector Logical

```asm
and   v0.16b, v1.16b, v2.16b  // Bitwise AND
orr   v0.16b, v1.16b, v2.16b  // Bitwise OR
eor   v0.16b, v1.16b, v2.16b  // Bitwise XOR
bic   v0.16b, v1.16b, v2.16b  // Bit clear
```

### Vector Compare

```asm
cmeq  v0.4s, v1.4s, v2.4s     // Compare equal
cmgt  v0.4s, v1.4s, v2.4s     // Compare greater than
cmge  v0.4s, v1.4s, v2.4s     // Compare greater or equal
```

### Vector Copy

```asm
dup   v0.4s, w0               // Duplicate scalar to all lanes
mov   v0.s[0], w0             // Insert into lane
umov  w0, v0.s[0]             // Extract from lane
ins   v0.s[0], v1.s[1]        // Copy lane between vectors
```

### Vector Load / Store

```asm
ld1   {v0.4s}, [x0]           // Load single structure
st1   {v0.4s}, [x0]           // Store single structure
```

### Vector Arrangements

Supported element arrangements:

| Arrangement | Elements | Total Size |
|:------------|:---------|:-----------|
| `.8B` | 8 × byte | 64-bit (D register) |
| `.16B` | 16 × byte | 128-bit (Q register) |
| `.4H` | 4 × halfword | 64-bit |
| `.8H` | 8 × halfword | 128-bit |
| `.2S` | 2 × word | 64-bit |
| `.4S` | 4 × word | 128-bit |
| `.1D` | 1 × doubleword | 64-bit |
| `.2D` | 2 × doubleword | 128-bit |

---

## Literal Pools

The `LDR Xn, =value` pseudo-instruction loads constants from a literal pool:

```asm
ldr x0, =0xDEADBEEF           // 64-bit constant
ldr w1, =42                   // 32-bit constant
.ltorg                         // Flush literal pool here
```

- Pool entries size determined by register width (X-reg → 8 bytes, W-reg → 4 bytes)
- Identical values with same size share a single pool entry (deduplication)
- Pool aligned to largest entry size for natural alignment
- Pool flushed at `.ltorg`/`.pool` or automatically at end of assembly
