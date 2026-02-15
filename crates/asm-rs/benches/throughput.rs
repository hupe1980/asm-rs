//! Performance benchmarks for `asm_rs`.
//!
//! Measures:
//! - Single instruction latency (per architecture)
//! - Multi-instruction throughput (KB/s of source text)
//! - Label-heavy workloads (100+ labels)
//! - Branch relaxation passes
//! - Preprocessor macro expansion
//!
//! Run with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use asm_rs::{assemble, Arch, Assembler};

// ─── Single-Instruction Latency ──────────────────────────────────────────────

fn bench_single_instruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_instruction");

    group.bench_function("x86_64_nop", |b| {
        b.iter(|| assemble(black_box("nop"), Arch::X86_64).unwrap())
    });

    group.bench_function("x86_64_mov_reg_imm", |b| {
        b.iter(|| assemble(black_box("mov rax, 0x1234"), Arch::X86_64).unwrap())
    });

    group.bench_function("x86_64_add_reg_reg", |b| {
        b.iter(|| assemble(black_box("add rax, rbx"), Arch::X86_64).unwrap())
    });

    group.bench_function("x86_64_mov_mem", |b| {
        b.iter(|| assemble(black_box("mov [rax+rcx*8+0x10], rdx"), Arch::X86_64).unwrap())
    });

    group.bench_function("x86_64_vaddps_avx", |b| {
        b.iter(|| assemble(black_box("vaddps ymm0, ymm1, ymm2"), Arch::X86_64).unwrap())
    });

    group.bench_function("x86_64_vaddps_avx512", |b| {
        b.iter(|| assemble(black_box("vaddps zmm0, zmm1, zmm2"), Arch::X86_64).unwrap())
    });

    group.bench_function("aarch64_add", |b| {
        b.iter(|| assemble(black_box("add x0, x1, x2"), Arch::Aarch64).unwrap())
    });

    group.bench_function("aarch64_ldr", |b| {
        b.iter(|| assemble(black_box("ldr x0, [x1 + 8]"), Arch::Aarch64).unwrap())
    });

    group.bench_function("arm_add", |b| {
        b.iter(|| assemble(black_box("add r0, r1, r2"), Arch::Arm).unwrap())
    });

    group.bench_function("riscv64_add", |b| {
        b.iter(|| assemble(black_box("add a0, a1, a2"), Arch::Rv64).unwrap())
    });

    group.bench_function("riscv64_lw", |b| {
        b.iter(|| assemble(black_box("lw a0, 0(a1)"), Arch::Rv64).unwrap())
    });

    group.finish();
}

// ─── Multi-Instruction Throughput ─────────────────────────────────────────────

/// Generate a block of N x86-64 instructions (no labels).
fn gen_x86_64_block(n: usize) -> String {
    let mut s = String::with_capacity(n * 20);
    for i in 0..n {
        match i % 6 {
            0 => s.push_str("mov rax, rbx\n"),
            1 => s.push_str("add rcx, rdx\n"),
            2 => s.push_str("sub rsi, rdi\n"),
            3 => s.push_str("xor r8, r9\n"),
            4 => s.push_str("and r10, r11\n"),
            5 => s.push_str("or r12, r13\n"),
            _ => unreachable!(),
        }
    }
    s
}

/// Generate a block of N AArch64 instructions (no labels).
fn gen_aarch64_block(n: usize) -> String {
    let mut s = String::with_capacity(n * 20);
    for i in 0..n {
        match i % 4 {
            0 => s.push_str("add x0, x1, x2\n"),
            1 => s.push_str("sub x3, x4, x5\n"),
            2 => s.push_str("and x6, x7, x8\n"),
            3 => s.push_str("orr x9, x10, x11\n"),
            _ => unreachable!(),
        }
    }
    s
}

/// Generate a block of N RISC-V instructions (no labels).
fn gen_riscv_block(n: usize) -> String {
    let mut s = String::with_capacity(n * 20);
    for i in 0..n {
        match i % 4 {
            0 => s.push_str("add a0, a1, a2\n"),
            1 => s.push_str("sub a3, a4, a5\n"),
            2 => s.push_str("and a6, a7, t0\n"),
            3 => s.push_str("or t1, t2, t3\n"),
            _ => unreachable!(),
        }
    }
    s
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // x86-64: 100 instructions
    let src_100 = gen_x86_64_block(100);
    group.throughput(Throughput::Bytes(src_100.len() as u64));
    group.bench_function("x86_64_100_insn", |b| {
        b.iter(|| assemble(black_box(&src_100), Arch::X86_64).unwrap())
    });

    // x86-64: 1000 instructions
    let src_1k = gen_x86_64_block(1000);
    group.throughput(Throughput::Bytes(src_1k.len() as u64));
    group.bench_function("x86_64_1000_insn", |b| {
        b.iter(|| assemble(black_box(&src_1k), Arch::X86_64).unwrap())
    });

    // x86-64: 5000 instructions
    let src_5k = gen_x86_64_block(5000);
    group.throughput(Throughput::Bytes(src_5k.len() as u64));
    group.bench_function("x86_64_5000_insn", |b| {
        b.iter(|| assemble(black_box(&src_5k), Arch::X86_64).unwrap())
    });

    // AArch64: 1000 instructions
    let src_a64 = gen_aarch64_block(1000);
    group.throughput(Throughput::Bytes(src_a64.len() as u64));
    group.bench_function("aarch64_1000_insn", |b| {
        b.iter(|| assemble(black_box(&src_a64), Arch::Aarch64).unwrap())
    });

    // RISC-V: 1000 instructions
    let src_rv = gen_riscv_block(1000);
    group.throughput(Throughput::Bytes(src_rv.len() as u64));
    group.bench_function("riscv_1000_insn", |b| {
        b.iter(|| assemble(black_box(&src_rv), Arch::Rv64).unwrap())
    });

    group.finish();
}

// ─── Label-Heavy Workloads ────────────────────────────────────────────────────

/// Generate code with many labels and references.
fn gen_label_heavy(n_labels: usize) -> String {
    let mut s = String::with_capacity(n_labels * 40);
    for i in 0..n_labels {
        s.push_str(&format!("label_{i}:\n"));
        s.push_str("nop\n");
    }
    // Forward references
    for i in 0..n_labels.min(50) {
        let target = (i + n_labels / 2) % n_labels;
        s.push_str(&format!("jmp label_{target}\n"));
    }
    s
}

fn bench_labels(c: &mut Criterion) {
    let mut group = c.benchmark_group("labels");

    let src_50 = gen_label_heavy(50);
    group.bench_function("50_labels", |b| {
        b.iter(|| assemble(black_box(&src_50), Arch::X86_64).unwrap())
    });

    let src_200 = gen_label_heavy(200);
    group.bench_function("200_labels", |b| {
        b.iter(|| assemble(black_box(&src_200), Arch::X86_64).unwrap())
    });

    let src_500 = gen_label_heavy(500);
    group.bench_function("500_labels", |b| {
        b.iter(|| assemble(black_box(&src_500), Arch::X86_64).unwrap())
    });

    group.finish();
}

// ─── Branch Relaxation ────────────────────────────────────────────────────────

/// Generate code that forces branch relaxation (long jumps past NOP sleds).
fn gen_relaxation_workload(n_nops: usize) -> String {
    let mut s = String::with_capacity(n_nops + 200);
    s.push_str("start:\n");
    // Conditional branches that may need relaxation
    s.push_str("je far_target\n");
    s.push_str("jne far_target\n");
    s.push_str("jl far_target\n");
    // NOP sled to push target beyond short-branch range
    for _ in 0..n_nops {
        s.push_str("nop\n");
    }
    s.push_str("far_target:\n");
    s.push_str("ret\n");
    s
}

fn bench_relaxation(c: &mut Criterion) {
    let mut group = c.benchmark_group("relaxation");

    // Within short-branch range: no relaxation needed
    let src_short = gen_relaxation_workload(10);
    group.bench_function("short_branch_10_nop", |b| {
        b.iter(|| assemble(black_box(&src_short), Arch::X86_64).unwrap())
    });

    // Near the boundary: ~127 bytes pushes to edge
    let src_edge = gen_relaxation_workload(120);
    group.bench_function("edge_branch_120_nop", |b| {
        b.iter(|| assemble(black_box(&src_edge), Arch::X86_64).unwrap())
    });

    // Past the boundary: relaxation to long form required
    let src_long = gen_relaxation_workload(200);
    group.bench_function("long_branch_200_nop", |b| {
        b.iter(|| assemble(black_box(&src_long), Arch::X86_64).unwrap())
    });

    group.finish();
}

// ─── Preprocessor / Macro Expansion ──────────────────────────────────────────

fn bench_preprocessor(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessor");

    // .rept expansion
    let src_rept = ".rept 100\nnop\n.endr";
    group.bench_function("rept_100_nop", |b| {
        b.iter(|| assemble(black_box(src_rept), Arch::X86_64).unwrap())
    });

    // Macro with parameters
    let src_macro = "\
.macro push_pair r1, r2
push \\r1
push \\r2
.endm
push_pair rax, rbx
push_pair rcx, rdx
push_pair rsi, rdi
push_pair r8, r9
push_pair r10, r11
push_pair r12, r13
push_pair r14, r15
";
    group.bench_function("macro_push_pair_7x", |b| {
        b.iter(|| assemble(black_box(src_macro), Arch::X86_64).unwrap())
    });

    // Conditional assembly
    let src_cond = "\
.equ USE_FAST_PATH, 1
.if USE_FAST_PATH
mov rax, rbx
add rax, rcx
.else
xor rax, rax
.endif
ret
";
    group.bench_function("conditional_assembly", |b| {
        b.iter(|| assemble(black_box(src_cond), Arch::X86_64).unwrap())
    });

    group.finish();
}

// ─── Builder API vs One-Shot API ──────────────────────────────────────────────

fn bench_builder_vs_oneshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("api_comparison");

    let source = "mov rax, 1\nadd rax, rbx\nsub rcx, rdx\nret";

    group.bench_function("oneshot_4_insn", |b| {
        b.iter(|| assemble(black_box(source), Arch::X86_64).unwrap())
    });

    group.bench_function("builder_4_insn", |b| {
        b.iter(|| {
            let mut asm = Assembler::new(Arch::X86_64);
            asm.emit(black_box("mov rax, 1")).unwrap();
            asm.emit(black_box("add rax, rbx")).unwrap();
            asm.emit(black_box("sub rcx, rdx")).unwrap();
            asm.emit(black_box("ret")).unwrap();
            let result = asm.finish().unwrap();
            black_box(result.bytes());
        })
    });

    // Builder with base address
    group.bench_function("builder_4_insn_at_base", |b| {
        b.iter(|| {
            let mut asm = Assembler::new(Arch::X86_64);
            asm.base_address(0x400000);
            asm.emit(black_box("mov rax, 1")).unwrap();
            asm.emit(black_box("add rax, rbx")).unwrap();
            asm.emit(black_box("sub rcx, rdx")).unwrap();
            asm.emit(black_box("ret")).unwrap();
            let result = asm.finish().unwrap();
            black_box(result.bytes());
        })
    });

    group.finish();
}

// ─── Realistic Workloads ──────────────────────────────────────────────────────

fn bench_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic");

    // Syscall stub (Linux x86-64)
    let syscall_stub = "\
mov rax, 59
mov rdi, rsi
xor rsi, rsi
xor rdx, rdx
syscall
";
    group.bench_function("syscall_stub", |b| {
        b.iter(|| assemble(black_box(syscall_stub), Arch::X86_64).unwrap())
    });

    // Function prologue/epilogue
    let fn_prolog = "\
push rbp
mov rbp, rsp
sub rsp, 32
mov [rbp-8], rdi
mov [rbp-16], rsi
mov rax, [rbp-8]
add rax, [rbp-16]
leave
ret
";
    group.bench_function("function_prolog_epilog", |b| {
        b.iter(|| assemble(black_box(fn_prolog), Arch::X86_64).unwrap())
    });

    // AVX-512 vector loop body
    let avx512_body = "\
vmovaps zmm0, zmm1
vaddps zmm2, zmm3, zmm4
vmulps zmm5, zmm6, zmm7
vsubps zmm8, zmm9, zmm10
vmovaps zmm11, zmm12
vfmadd231ps zmm0, zmm1, zmm2
";
    group.bench_function("avx512_vector_body", |b| {
        b.iter(|| assemble(black_box(avx512_body), Arch::X86_64).unwrap())
    });

    // AArch64 function
    let aarch64_fn = "\
stp x29, x30, [sp, -16]!
mov x29, sp
add x0, x0, x1
sub x2, x0, 1
mul x3, x0, x2
ldp x29, x30, [sp], 16
ret
";
    group.bench_function("aarch64_function", |b| {
        b.iter(|| assemble(black_box(aarch64_fn), Arch::Aarch64).unwrap())
    });

    // RISC-V function
    let riscv_fn = "\
addi sp, sp, -16
sd ra, 8(sp)
sd s0, 0(sp)
addi s0, sp, 16
add a0, a0, a1
sub a2, a0, a3
ld ra, 8(sp)
ld s0, 0(sp)
addi sp, sp, 16
ret
";
    group.bench_function("riscv_function", |b| {
        b.iter(|| assemble(black_box(riscv_fn), Arch::Rv64).unwrap())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_instruction,
    bench_throughput,
    bench_labels,
    bench_relaxation,
    bench_preprocessor,
    bench_builder_vs_oneshot,
    bench_realistic,
);
criterion_main!(benches);
