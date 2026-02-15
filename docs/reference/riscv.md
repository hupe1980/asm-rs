---
layout: default
title: RISC-V
parent: ISA Reference
nav_order: 5
---

# RISC-V Instruction Reference
{: .fs-8 }

Complete reference for all RISC-V instructions supported by asm-rs,
covering RV32I/RV64I base ISAs and the M, A, F, D, V, and C extensions.
{: .fs-5 .fw-300 }

---

## Notation

| Symbol | Meaning |
|:-------|:--------|
| `rd` | Destination register (x0–x31 or ABI name) |
| `rs1` / `rs2` | Source register(s) |
| `imm` | Immediate value |
| `offset` | Branch/jump offset or memory displacement |
| `label` | Label reference |
| `csr` | Control/status register name or address |
| `fd` / `fs1` / `fs2` | Floating-point register (f0–f31 or ABI name) |
| `vd` / `vs1` / `vs2` | Vector register (v0–v31) |

### ABI Register Names

| ABI | Reg | ABI | Reg | ABI | Reg | ABI | Reg |
|:----|:----|:----|:----|:----|:----|:----|:----|
| `zero` | x0 | `ra` | x1 | `sp` | x2 | `gp` | x3 |
| `tp` | x4 | `t0` | x5 | `t1` | x6 | `t2` | x7 |
| `s0`/`fp` | x8 | `s1` | x9 | `a0` | x10 | `a1` | x11 |
| `a2` | x12 | `a3` | x13 | `a4` | x14 | `a5` | x15 |
| `a6` | x16 | `a7` | x17 | `s2` | x18 | `s3` | x19 |
| `s4` | x20 | `s5` | x21 | `s6` | x22 | `s7` | x23 |
| `s8` | x24 | `s9` | x25 | `s10` | x26 | `s11` | x27 |
| `t3` | x28 | `t4` | x29 | `t5` | x30 | `t6` | x31 |

---

## Base Integer ISA (RV32I)

### Upper-Immediate (U-type)

```asm
lui   rd, imm                  # Load upper immediate (bits 31:12)
auipc rd, imm                  # Add upper immediate to PC
```

### Jump (J-type / I-type)

```asm
jal   rd, label                # Jump and link (±1 MB)
jal   label                    # Jump and link (rd = ra implied)
jalr  rd, rs1, offset          # Jump and link register
jalr  rs1                      # Jump register (rd = ra, offset = 0)
```

### Branch (B-type)

```asm
beq   rs1, rs2, label          # Branch if equal
bne   rs1, rs2, label          # Branch if not equal
blt   rs1, rs2, label          # Branch if less than (signed)
bge   rs1, rs2, label          # Branch if greater or equal (signed)
bltu  rs1, rs2, label          # Branch if less than (unsigned)
bgeu  rs1, rs2, label          # Branch if greater or equal (unsigned)
```

{: .note }
> B-type branches automatically relax from 4-byte short form (±4 KiB) to
> 8-byte long form (±1 MiB) using inverted-branch + JAL pair.

### Load (I-type)

```asm
lb    rd, offset(rs1)          # Load byte (sign-extend)
lh    rd, offset(rs1)          # Load halfword (sign-extend)
lw    rd, offset(rs1)          # Load word (sign-extend on RV64)
ld    rd, offset(rs1)          # Load doubleword (RV64 only)
lbu   rd, offset(rs1)          # Load byte (unsigned)
lhu   rd, offset(rs1)          # Load halfword (unsigned)
lwu   rd, offset(rs1)          # Load word (unsigned, RV64 only)
lw    rd, (rs1)                # Bare register = 0(rs1)
```

### Store (S-type)

```asm
sb    rs2, offset(rs1)         # Store byte
sh    rs2, offset(rs1)         # Store halfword
sw    rs2, offset(rs1)         # Store word
sd    rs2, offset(rs1)         # Store doubleword (RV64 only)
```

### ALU — Immediate (I-type)

```asm
addi  rd, rs1, imm             # Add immediate
slti  rd, rs1, imm             # Set if less than (signed)
sltiu rd, rs1, imm             # Set if less than (unsigned)
xori  rd, rs1, imm             # XOR immediate
ori   rd, rs1, imm             # OR immediate
andi  rd, rs1, imm             # AND immediate
```

### Shift — Immediate

```asm
slli  rd, rs1, shamt           # Shift left logical (5-bit shamt RV32, 6-bit RV64)
srli  rd, rs1, shamt           # Shift right logical
srai  rd, rs1, shamt           # Shift right arithmetic
```

### ALU — Register (R-type)

```asm
add   rd, rs1, rs2             # Add
sub   rd, rs1, rs2             # Subtract
sll   rd, rs1, rs2             # Shift left logical
slt   rd, rs1, rs2             # Set if less than (signed)
sltu  rd, rs1, rs2             # Set if less than (unsigned)
xor   rd, rs1, rs2             # XOR
srl   rd, rs1, rs2             # Shift right logical
sra   rd, rs1, rs2             # Shift right arithmetic
or    rd, rs1, rs2             # OR
and   rd, rs1, rs2             # AND
```

---

## RV64I Extensions

W-suffix instructions operate on the lower 32 bits and sign-extend the result:

```asm
addiw rd, rs1, imm             # Add word immediate
slliw rd, rs1, shamt           # Shift left word
srliw rd, rs1, shamt           # Shift right logical word
sraiw rd, rs1, shamt           # Shift right arithmetic word
addw  rd, rs1, rs2             # Add word
subw  rd, rs1, rs2             # Subtract word
sllw  rd, rs1, rs2             # Shift left word
srlw  rd, rs1, rs2             # Shift right logical word
sraw  rd, rs1, rs2             # Shift right arithmetic word
```

---

## M Extension (Multiply / Divide)

```asm
mul    rd, rs1, rs2            # Multiply (low bits)
mulh   rd, rs1, rs2            # Multiply high (signed × signed)
mulhsu rd, rs1, rs2            # Multiply high (signed × unsigned)
mulhu  rd, rs1, rs2            # Multiply high (unsigned × unsigned)
div    rd, rs1, rs2            # Signed divide
divu   rd, rs1, rs2            # Unsigned divide
rem    rd, rs1, rs2            # Signed remainder
remu   rd, rs1, rs2            # Unsigned remainder
```

### RV64M W-suffix

```asm
mulw  rd, rs1, rs2             # Multiply word
divw  rd, rs1, rs2             # Divide word (signed)
divuw rd, rs1, rs2             # Divide word (unsigned)
remw  rd, rs1, rs2             # Remainder word (signed)
remuw rd, rs1, rs2             # Remainder word (unsigned)
```

---

## A Extension (Atomics)

### Load-Reserved / Store-Conditional

```asm
lr.w   rd, (rs1)               # Load reserved word
lr.d   rd, (rs1)               # Load reserved doubleword (RV64)
sc.w   rd, rs2, (rs1)          # Store conditional word
sc.d   rd, rs2, (rs1)          # Store conditional doubleword
```

### Atomic Memory Operations

```asm
amoswap.w  rd, rs2, (rs1)     # Atomic swap
amoadd.w   rd, rs2, (rs1)     # Atomic add
amoand.w   rd, rs2, (rs1)     # Atomic AND
amoor.w    rd, rs2, (rs1)     # Atomic OR
amoxor.w   rd, rs2, (rs1)     # Atomic XOR
amomax.w   rd, rs2, (rs1)     # Atomic max (signed)
amomaxu.w  rd, rs2, (rs1)     # Atomic max (unsigned)
amomin.w   rd, rs2, (rs1)     # Atomic min (signed)
amominu.w  rd, rs2, (rs1)     # Atomic min (unsigned)
```

All atomics support `.d` suffix for 64-bit (RV64) and ordering suffixes:

```asm
amoswap.w.aq   rd, rs2, (rs1) # Acquire
amoswap.w.rl   rd, rs2, (rs1) # Release
amoswap.w.aqrl rd, rs2, (rs1) # Acquire-release
```

---

## CSR Instructions

### Standard CSR Operations

```asm
csrrw  rd, csr, rs1            # CSR read/write
csrrs  rd, csr, rs1            # CSR read and set bits
csrrc  rd, csr, rs1            # CSR read and clear bits
csrrwi rd, csr, uimm5          # CSR read/write immediate
csrrsi rd, csr, uimm5          # CSR read and set bits (immediate)
csrrci rd, csr, uimm5          # CSR read and clear bits (immediate)
```

### CSR Pseudo-Instructions

```asm
csrr  rd, csr                  # Read CSR (csrrs rd, csr, x0)
csrw  csr, rs1                 # Write CSR (csrrw x0, csr, rs1)
csrs  csr, rs1                 # Set bits in CSR
csrc  csr, rs1                 # Clear bits in CSR
csrwi csr, uimm5               # Write CSR (immediate)
csrsi csr, uimm5               # Set bits (immediate)
csrci csr, uimm5               # Clear bits (immediate)
```

~50 named CSRs are supported (e.g., `mstatus`, `mtvec`, `mepc`, `mcause`,
`mhartid`, `cycle`, `time`, `instret`, etc.).

---

## System

```asm
ecall                          # Environment call (syscall)
ebreak                         # Environment breakpoint
fence                          # Memory fence
fence.i                        # Instruction fence
```

### Privileged

```asm
mret                           # Machine return
sret                           # Supervisor return
wfi                            # Wait for interrupt
sfence.vma rs1, rs2            # Supervisor fence (address + ASID)
sfence.vma                     # Full TLB flush
```

---

## Pseudo-Instructions

```asm
nop                            # No operation (addi x0, x0, 0)
li    rd, imm                  # Load immediate (multi-instruction for large values)
la    rd, label                # Load address (AUIPC + ADDI pair)
mv    rd, rs1                  # Move register (addi rd, rs1, 0)
not   rd, rs1                  # Bitwise NOT (xori rd, rs1, -1)
neg   rd, rs1                  # Negate (sub rd, x0, rs1)
negw  rd, rs1                  # Negate word (subw rd, x0, rs1)
seqz  rd, rs1                  # Set if equal to zero (sltiu rd, rs1, 1)
snez  rd, rs1                  # Set if not zero (sltu rd, x0, rs1)
sltz  rd, rs1                  # Set if less than zero (slt rd, rs1, x0)
sgtz  rd, rs1                  # Set if greater than zero (slt rd, x0, rs1)
j     label                    # Unconditional jump (jal x0, label)
jr    rs1                      # Jump register (jalr x0, rs1, 0)
ret                            # Return (jalr x0, ra, 0)
call  label                    # Far call (AUIPC + JALR pair)
tail  label                    # Far tail call (AUIPC + JALR pair)
sext.w rd, rs1                 # Sign-extend word (addiw rd, rs1, 0)
```

### Branch Pseudo-Instructions

```asm
beqz  rs1, label               # Branch if zero (beq rs1, x0, label)
bnez  rs1, label               # Branch if not zero
blez  rs1, label               # Branch if ≤ 0 (bge x0, rs1, label)
bgez  rs1, label               # Branch if ≥ 0
bltz  rs1, label               # Branch if < 0
bgtz  rs1, label               # Branch if > 0
bgt   rs1, rs2, label          # Branch if > (blt rs2, rs1, label)
ble   rs1, rs2, label          # Branch if ≤
bgtu  rs1, rs2, label          # Branch if > (unsigned)
bleu  rs1, rs2, label          # Branch if ≤ (unsigned)
```

### 64-bit Load Immediate (RV64)

For large 64-bit constants, `li` produces an optimal multi-instruction
sequence using LUI/ADDI/SLLI:

```asm
li    a0, 0xDEADBEEFCAFEBABE   # Up to 8 instructions (32 bytes)
```

{: .note }
> The `li` decomposition correctly handles sign-extension edge cases
> (e.g., `li a0, 0x80000000` produces positive 2^31 on RV64).

---

## C Extension (Compressed — 16-bit)

### CR-type (Register)

```asm
c.mv    rd, rs2                # Move (rd ≠ x0, rs2 ≠ x0)
c.add   rd, rs2                # Add (rd ≠ x0, rs2 ≠ x0)
c.jr    rs1                    # Jump register (rs1 ≠ x0)
c.jalr  rs1                    # Jump and link register (rs1 ≠ x0)
```

### CI-type (Immediate)

```asm
c.li     rd, imm               # Load 6-bit signed immediate
c.lui    rd, imm               # Load upper immediate (rd ≠ x0, x2)
c.addi   rd, imm               # Add 6-bit signed immediate (imm ≠ 0)
c.addiw  rd, imm               # Add word immediate (RV64, rd ≠ x0)
c.addi16sp imm                 # Add to SP (scaled ×16, imm ≠ 0)
c.slli   rd, shamt             # Shift left (rd ≠ x0)
c.lwsp   rd, offset            # Load word, SP-relative
c.ldsp   rd, offset            # Load double, SP-relative (RV64)
```

### CSS-type (Stack Store)

```asm
c.swsp  rs2, offset            # Store word, SP-relative
c.sdsp  rs2, offset            # Store double, SP-relative (RV64)
```

### CIW-type (Wide Immediate)

```asm
c.addi4spn rd', imm            # Add scaled immediate to SP (rd' = x8–x15)
```

### CL/CS-type (Load/Store)

```asm
c.lw    rd', offset(rs1')      # Load word (compact registers x8–x15)
c.ld    rd', offset(rs1')      # Load double (RV64)
c.sw    rs2', offset(rs1')     # Store word
c.sd    rs2', offset(rs1')     # Store double (RV64)
```

### CA-type (Arithmetic)

```asm
c.sub   rd', rs2'              # Subtract (compact registers)
c.xor   rd', rs2'              # XOR
c.or    rd', rs2'              # OR
c.and   rd', rs2'              # AND
c.subw  rd', rs2'              # Subtract word (RV64)
c.addw  rd', rs2'              # Add word (RV64)
```

### CB-type (Branch)

```asm
c.beqz  rs1', label            # Branch if zero (±256 B)
c.bnez  rs1', label            # Branch if not zero (±256 B)
c.srli  rd', shamt             # Shift right logical
c.srai  rd', shamt             # Shift right arithmetic
c.andi  rd', imm               # AND immediate
```

### CJ-type (Jump)

```asm
c.j     label                  # Unconditional jump (±2 KB)
```

### Misc

```asm
c.nop                          # No operation
c.ebreak                       # Environment breakpoint
```

### Auto-Narrowing

Enable automatic compression of full-width instructions to C-extension
equivalents when constraints are met:

```asm
.option rvc                    # Enable auto-narrowing
addi  a0, a0, 1               # → c.addi a0, 1 (2 bytes instead of 4)
add   a0, a0, a1              # → c.add a0, a1
lw    a0, 0(sp)               # → c.lwsp a0, 0
.option norvc                  # Disable auto-narrowing
```

### Compressed Branch Relaxation

```asm
c.beqz  a0, far_label         # Widens to inverted-B + JAL (8B, ±1 MiB)
c.bnez  a0, far_label         # when target exceeds ±256 B
c.j     far_label              # Widens to JAL (4B, ±1 MiB)
                                # when target exceeds ±2 KB
```

---

## Memory Addressing

RISC-V uses `offset(register)` memory addressing:

```asm
lw  a0, 0(sp)                 # Load word from [sp + 0]
sw  a0, 8(sp)                 # Store word to [sp + 8]
lw  a0, -4(s0)                # Negative offset
lw  a0, (sp)                  # Bare register = 0(sp)
```
