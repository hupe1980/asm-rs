---
layout: default
title: Thumb
parent: ISA Reference
nav_order: 4
---

# Thumb / Thumb-2 (T32) Instruction Reference
{: .fs-8 }

Complete reference for all Thumb and Thumb-2 instructions supported by asm-rs.
{: .fs-5 .fw-300 }

See also: [ARM32 Reference]({{ site.baseurl }}/reference/arm32).

---

## Overview

Thumb is a mixed 16/32-bit instruction set for ARM processors:
- **Thumb-1** (T16): 16-bit instructions with compact encodings for common operations
- **Thumb-2** (T32): 32-bit extensions for the full ARM instruction set

The assembler selects the shortest encoding automatically. Use the `.w` suffix
to force a 32-bit wide encoding.

## Notation

| Symbol | Meaning |
|:-------|:--------|
| `Rd`, `Rn`, `Rm` | Low registers R0–R7 (unless noted) |
| `Rd_hi` | Any register R0–R15 (high-register form) |
| `#imm3/5/8` | Immediate of specified bit width |
| `.w` | Suffix to force 32-bit (Thumb-2) encoding |

---

## Thumb-1 (16-bit) Instructions

### Data Processing

```asm
mov   r0, #42                 @ Move 8-bit immediate (R0–R7)
mov   r0, r8                  @ Move any register to any register
movs  r0, #0                  @ Move with flag-setting

add   r0, r1, r2              @ Three-register add (R0–R7)
add   r0, r1, #3              @ Add 3-bit immediate
add   r0, #100                @ Add 8-bit immediate to self
add   r0, r8                  @ High-register add (any regs)
adds  r0, r1, r2              @ Add with flag-setting

sub   r0, r1, r2              @ Three-register subtract
sub   r0, r1, #3              @ Subtract 3-bit immediate
sub   r0, #100                @ Subtract 8-bit immediate from self
subs  r0, r1, r2              @ Subtract with flag-setting

cmp   r0, #42                 @ Compare with 8-bit immediate
cmp   r0, r1                  @ Compare registers (low or high)
cmn   r0, r1                  @ Compare negative (low regs)
tst   r0, r1                  @ Test bits (AND, flags only)
```

### ALU Register Operations (16-bit)

```asm
and   r0, r1                  @ Bitwise AND
orr   r0, r1                  @ Bitwise OR
eor   r0, r1                  @ Bitwise XOR
bic   r0, r1                  @ Bit clear
mvn   r0, r1                  @ Bitwise NOT
mul   r0, r1                  @ Multiply
adc   r0, r1                  @ Add with carry
sbc   r0, r1                  @ Subtract with carry
ror   r0, r1                  @ Rotate right by register
neg   r0, r1                  @ Negate (rsb r0, r1, #0)
```

### Shifts (16-bit)

```asm
lsl   r0, r1, #4              @ Logical shift left (5-bit imm)
lsr   r0, r1, #4              @ Logical shift right
asr   r0, r1, #4              @ Arithmetic shift right
lsl   r0, r1                  @ Shift left by register
lsr   r0, r1                  @ Shift right by register
asr   r0, r1                  @ Arithmetic shift right by register
```

### Load / Store (16-bit)

```asm
ldr   r0, [r1]                @ Load word
ldr   r0, [r1, #20]           @ Load with 5-bit word offset (×4)
ldr   r0, [r1, r2]            @ Load with register offset
ldr   r0, [sp, #40]           @ SP-relative load (8-bit ×4)
ldrb  r0, [r1]                @ Load byte
ldrb  r0, [r1, #5]            @ Load byte with 5-bit offset
ldrh  r0, [r1]                @ Load halfword
ldrh  r0, [r1, #10]           @ Load halfword with 5-bit offset (×2)
ldrsb r0, [r1, r2]            @ Load signed byte (register offset)
ldrsh r0, [r1, r2]            @ Load signed halfword

str   r0, [r1]                @ Store word
str   r0, [r1, #20]           @ Store with offset
str   r0, [sp, #40]           @ SP-relative store
strb  r0, [r1]                @ Store byte
strh  r0, [r1]                @ Store halfword
```

### Stack (16-bit)

```asm
push  {r0-r7}                 @ Push low registers
push  {r4, r5, lr}            @ Push with LR
pop   {r0-r7}                 @ Pop low registers
pop   {r4, r5, pc}            @ Pop with PC (return)
```

### Branch (16-bit)

```asm
b     label                   @ Unconditional branch (±2 KB)
b.eq  label                   @ Conditional branch (±256 B)
b.ne  label
b.cs  label
@ ... all 14 non-AL condition codes
bx    r0                      @ Branch and exchange
blx   r0                      @ Branch, link, and exchange
```

### System (16-bit)

```asm
nop                            @ No operation (0xBF00)
bkpt  #0                      @ Breakpoint
svc   #0                      @ Supervisor call
```

---

## Thumb-2 (32-bit) Instructions

### Wide Branch

```asm
bl    label                   @ Branch and link (±16 MB, 32-bit)
b.w   label                   @ Unconditional wide branch (±16 MB)
b.eq.w label                  @ Conditional wide branch (±1 MB)
```

### Wide Data Processing

Use the `.w` suffix to force 32-bit encoding:

```asm
add.w r0, r1, r2              @ Wide add
sub.w r0, r1, #1000           @ Wide subtract with large immediate
and.w r0, r1, r2              @ Wide AND
orr.w r0, r1, r2              @ Wide OR
```

{: .note }
> Thumb-2 uses a modified immediate encoding that supports a wider range of
> constants than Thumb-1's limited immediate fields.

---

## IT Blocks

The IT (If-Then) instruction makes up to 4 following instructions conditional:

```asm
@ IT{x{y{z}}} cond
@ where x,y,z are T (then) or E (else)

it    eq                       @ Next instruction is conditional (EQ)
moveq r0, #1

ite   ne                      @ If-Then-Else
movne r0, #1
moveq r0, #0

itete gt                      @ Complex: If-Then-Else-Then-Else
movgt r0, #4
movle r0, #3
addgt r0, r0, #1
addle r0, r0, #2

itt   cs                      @ If-Then-Then
movcs r0, #1
addcs r0, r0, #2
```

---

## Branch Relaxation

Thumb branches automatically relax between narrow and wide forms:

| Form | Encoding | Range |
|:-----|:---------|:------|
| B (narrow) | 16-bit | ±2 KB |
| B.cond (narrow) | 16-bit | ±256 B |
| B.W (wide) | 32-bit | ±16 MB |
| B.cond.W (wide) | 32-bit | ±1 MB |
| BL | 32-bit | ±16 MB |

The linker automatically promotes narrow branches to wide when the target
is out of range (Szymanski algorithm).

---

## Mode Switching Directives

```asm
.thumb                         @ Switch to Thumb mode
.arm                           @ Switch to ARM mode
.thumb_func                    @ Mark next label as Thumb function (sets LSB)
```

{: .tip }
> `.thumb_func` sets the least significant bit on the label address,
> which is required for correct `BX`/`BLX` interworking.
