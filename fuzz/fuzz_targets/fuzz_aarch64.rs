#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    let _ = asm_rs::assemble(data, asm_rs::Arch::Aarch64);
    let _ = asm_rs::assemble_at(data, asm_rs::Arch::Aarch64, 0x400000);

    let mut asm = asm_rs::Assembler::new(asm_rs::Arch::Aarch64);
    for line in data.lines() {
        if asm.emit(line).is_err() {
            return;
        }
    }
    let _ = asm.finish();
});
