---
layout: default
title: ARM32
---

# ARM32 (A32) Instruction Reference

Complete reference for all ARM32 instructions supported by asm-rs.

See also: [Thumb/Thumb-2 Reference]({{ site.baseurl }}/reference/thumb).

---

## Notation

| Symbol | Meaning |
|:-------|:--------|
| `Rd`, `Rn`, `Rm`, `Rs` | General-purpose registers R0–R15 |
| `SP` | Stack pointer (R13) |
| `LR` | Link register (R14) |
| `PC` | Program counter (R15) |
| `#imm` | Immediate value (8-bit rotated for A32) |
| `#imm16` | 16-bit immediate (MOVW/MOVT) |
| `label` | Label reference (resolved to offset) |
| `{reglist}` | Register list, e.g. `{R0-R7, LR}` |
| `cond` | Optional condition suffix (see below) |
| `S` | Optional flag-setting suffix |

---

## Condition Codes

Every A32 instruction can be conditionally executed by appending a two-letter
condition suffix. The default is `al` (always).

| Suffix | Meaning | Flags |
|:-------|:--------|:------|
| `eq` | Equal | Z=1 |
| `ne` | Not equal | Z=0 |
| `cs` / `hs` | Carry set / Unsigned ≥ | C=1 |
| `cc` / `lo` | Carry clear / Unsigned < | C=0 |
| `mi` | Minus (negative) | N=1 |
| `pl` | Plus (positive or zero) | N=0 |
| `vs` | Overflow | V=1 |
| `vc` | No overflow | V=0 |
| `hi` | Unsigned > | C=1 and Z=0 |
| `ls` | Unsigned ≤ | C=0 or Z=1 |
| `ge` | Signed ≥ | N=V |
| `lt` | Signed < | N≠V |
| `gt` | Signed > | Z=0 and N=V |
| `le` | Signed ≤ | Z=1 or N≠V |
| `al` | Always (default) | — |

Example: `addeq r0, r1, r2` — adds only when zero flag is set.

---

## Data Processing

All data-processing instructions support the `S` suffix for flag-setting
and any condition code suffix.

### Arithmetic

```asm
add   r0, r1, r2              @ Register add
add   r0, r1, #42             @ Immediate add
adds  r0, r1, r2              @ Add and set flags
adc   r0, r1, r2              @ Add with carry
sub   r0, r1, r2              @ Subtract
subs  r0, r1, r2              @ Subtract and set flags
sbc   r0, r1, r2              @ Subtract with carry
rsb   r0, r1, r2              @ Reverse subtract (r2 - r1)
rsc   r0, r1, r2              @ Reverse subtract with carry
```

### Logical

```asm
and   r0, r1, r2              @ Bitwise AND
orr   r0, r1, r2              @ Bitwise OR
eor   r0, r1, r2              @ Bitwise XOR
bic   r0, r1, r2              @ Bit clear (AND NOT)
```

### Move

```asm
mov   r0, r1                  @ Move register
mov   r0, #42                 @ Move immediate
mvn   r0, r1                  @ Move NOT
movw  r0, #0x1234             @ Move 16-bit immediate (low halfword)
movt  r0, #0x5678             @ Move top halfword
```

> `mov r0, #large` with a non-encodable immediate automatically emits a
> MOVW+MOVT pair to load the full 32-bit value.

### Compare / Test

```asm
cmp   r0, r1                  @ Compare (sets flags, no result)
cmp   r0, #42                 @ Compare with immediate
cmn   r0, r1                  @ Compare negative
tst   r0, r1                  @ Test bits (AND, flags only)
teq   r0, r1                  @ Test equivalence (XOR, flags only)
```

### Barrel Shifter

All data processing operands support the barrel shifter:

```asm
add   r0, r1, r2, lsl #3     @ r0 = r1 + (r2 << 3)
sub   r0, r1, r2, lsr #4     @ Logical shift right
and   r0, r1, r2, asr #2     @ Arithmetic shift right
orr   r0, r1, r2, ror #8     @ Rotate right
mov   r0, r1, rrx             @ Rotate right through carry (1 bit)

@ Register-specified shift amount
add   r0, r1, r2, lsl r3     @ r0 = r1 + (r2 << r3)
```

### Standalone Shifts

```asm
lsl   r0, r1, #4              @ Logical shift left
lsr   r0, r1, #4              @ Logical shift right
asr   r0, r1, #4              @ Arithmetic shift right
ror   r0, r1, #4              @ Rotate right
lsl   r0, r1, r2              @ Register shift amount
```

---

## Multiply

```asm
mul   r0, r1, r2              @ r0 = r1 * r2
mla   r0, r1, r2, r3          @ r0 = r1 * r2 + r3
muls  r0, r1, r2              @ Multiply and set flags
mlas  r0, r1, r2, r3          @ Multiply-accumulate and set flags
```

### Long Multiply

```asm
umull r0, r1, r2, r3          @ Unsigned: {r1,r0} = r2 * r3 (64-bit result)
smull r0, r1, r2, r3          @ Signed: {r1,r0} = r2 * r3
umlal r0, r1, r2, r3          @ Unsigned multiply-accumulate long
smlal r0, r1, r2, r3          @ Signed multiply-accumulate long
```

---

## Bit Manipulation

```asm
clz   r0, r1                  @ Count leading zeros
rev   r0, r1                  @ Reverse bytes (32-bit)
rev16 r0, r1                  @ Reverse bytes in each halfword
revsh r0, r1                  @ Reverse bytes, sign-extend halfword
```

### Bitfield

```asm
bfc   r0, #4, #8              @ Bit field clear: clear 8 bits from bit 4
bfi   r0, r1, #4, #8          @ Bit field insert: insert 8 bits from r1 at bit 4
sbfx  r0, r1, #4, #8          @ Signed bit field extract
ubfx  r0, r1, #4, #8          @ Unsigned bit field extract
```

---

## Extend

```asm
uxtb  r0, r1                  @ Unsigned extend byte (8 → 32)
uxth  r0, r1                  @ Unsigned extend halfword (16 → 32)
sxtb  r0, r1                  @ Signed extend byte
sxth  r0, r1                  @ Signed extend halfword
```

---

## Load / Store

### Basic Forms

```asm
ldr   r0, [r1]                @ Load word from [r1]
ldr   r0, [r1, #4]            @ Load with immediate offset
str   r0, [r1]                @ Store word to [r1]
str   r0, [r1, #4]            @ Store with immediate offset
ldrb  r0, [r1]                @ Load byte (zero-extend)
strb  r0, [r1]                @ Store byte
ldrh  r0, [r1]                @ Load halfword (zero-extend)
strh  r0, [r1]                @ Store halfword
ldrsh r0, [r1]                @ Load halfword (sign-extend)
ldrsb r0, [r1]                @ Load byte (sign-extend)
```

### Register Offset

```asm
ldr   r0, [r1, r2]            @ Load with register offset
str   r0, [r1, r2]            @ Store with register offset
```

### Pre-Index / Post-Index

```asm
ldr   r0, [r1, #4]!           @ Pre-index: r1 += 4, then load
str   r0, [r1, #-4]!          @ Pre-index store
ldr   r0, [r1], #4            @ Post-index: load, then r1 += 4
str   r0, [r1], #-4           @ Post-index store
```

---

## Exclusive Access

```asm
ldrex   r0, [r1]              @ Load exclusive
ldrexb  r0, [r1]              @ Load exclusive byte
ldrexh  r0, [r1]              @ Load exclusive halfword
ldrexd  r0, r1, [r2]          @ Load exclusive doubleword
strex   r0, r1, [r2]          @ Store exclusive (r0 = success)
strexb  r0, r1, [r2]          @ Store exclusive byte
strexh  r0, r1, [r2]          @ Store exclusive halfword
strexd  r0, r2, r3, [r4]      @ Store exclusive doubleword
```

---

## Block Transfer (PUSH / POP / LDM / STM)

```asm
push  {r4, r5, lr}            @ Push registers (STMDB SP!)
pop   {r4, r5, pc}            @ Pop registers (LDMIA SP!) and return

@ Full LDM/STM instructions
ldm   r0, {r1-r4}             @ Load multiple (Increment After)
ldmia r0!, {r1-r4}            @ Load with writeback
ldmdb r0!, {r1-r4}            @ Decrement Before
ldmib r0!, {r1-r4}            @ Increment Before
ldmda r0!, {r1-r4}            @ Decrement After
stm   r0, {r1-r4}             @ Store multiple
stmia r0!, {r1-r4}            @ Store with writeback
stmdb r0!, {r1-r4}            @ Decrement Before (used for PUSH)
```

---

## Branch

```asm
b     label                   @ Branch (±32 MB)
bl    label                   @ Branch and link
bx    r0                      @ Branch and exchange (ARM ↔ Thumb)
blx   r0                      @ Branch, link, and exchange
```

### PC-Relative

```asm
adr   r0, label               @ Load PC-relative address (ADD/SUB Rd, PC, #offset)
```

---

## System

```asm
svc   #0                      @ Supervisor call (formerly SWI)
swi   #0                      @ Alias for SVC
bkpt  #0                      @ Breakpoint
nop                            @ No operation
mrs   r0, cpsr                @ Read CPSR → register
msr   cpsr_f, r0              @ Write register → CPSR flags
dmb                            @ Data memory barrier
dsb                            @ Data synchronization barrier
isb                            @ Instruction synchronization barrier
```

---

## Literal Pools

The `LDR Rd, =value` pseudo-instruction loads constants from a literal pool:

```asm
ldr   r0, =0x12345678         @ Load 32-bit constant from pool
ldr   r1, =42                 @ Small constant
.ltorg                         @ Flush literal pool here
```

- Pool entries are always 4 bytes (32-bit)
- Identical constants share a single pool entry (deduplication)
- Range limit: ±4,092 bytes from the instruction
- Pool flushed at `.ltorg`/`.pool` or automatically at end of assembly

---

## Memory Addressing Summary

| Form | Syntax | Description |
|:-----|:-------|:------------|
| Immediate offset | `[Rn, #imm]` | Base + signed 12-bit offset |
| Register offset | `[Rn, Rm]` | Base + register |
| Pre-index | `[Rn, #imm]!` | Base + offset, writeback |
| Post-index | `[Rn], #imm` | Base, then add offset |
| Register list | `{R0-R7, LR}` | Multiple registers for LDM/STM/PUSH/POP |
