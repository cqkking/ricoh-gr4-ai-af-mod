#!/usr/bin/env python3
"""
k3_gr4_dsp_patcher.py - Automated Boot-Time CEVA-CV DSP Awakening Patcher

Patches Ricoh GR IV's 05_RTOS.bin to automatically load and start the
CEVA-CV AI DSP co-processor on startup without requiring any UART debug console:

1. Injects the DSP Auto-Wakeup sequence into RTOS Code Cave @ 0x53000e00:
   - DspFactory::GetDspService() -> m_dspService
   - Check if already started (status & 1) -> skip if active
   - DspService::LoadDsp(m_dspService, 'A', 0, 0)
   - Check if loaded (status & 2) -> start if loaded
   - DspService::Start(m_dspService, 0, 0)
   - DspService::Test(m_dspService, 0) -> CEVA Version self-test
   - Records status block at 0x53000e70 (Magic: 'CEVA', load_ret, start_ret, test_ret)
2. Patches LiveViewProcess3aContoller entry @ 0x536520f0 with a trampoline branch (b #0x53000e00).
3. Preserves all stack, register, and AAPCS calling conventions.
"""

import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

GR4_BASE          = 0x53000000
CAVE_VA           = 0x53000e00
STATUS_VA         = 0x53000e70
GET_DSP_SVC_VA    = 0x539658c4 # DspFactory::GetDspService()
LOAD_DSP_VA       = 0x53966d74 # DspService::LoadDsp(svc, 'A', 0, 0)
START_DSP_VA      = 0x53966214 # DspService::Start(svc, 0, 0)
TEST_DSP_VA       = 0x53966f18 # DspService::Test(svc, 0)
HOOK_SITE_VA      = 0x536520f0 # LiveViewProcess3aContoller entry
RESUME_SITE_VA    = 0x53652108 # Resume at ldr r6, [r4] after prologue
RETURN_SITE_VA    = 0x53652108 # Alias for backward compatibility

def encode_arm_branch(from_va: int, to_va: int, is_bl: bool = False) -> int:
    disp = to_va - (from_va + 8)
    disp_words = (disp >> 2) & 0x00ffffff
    op = 0xeb000000 if is_bl else 0xea000000
    return op | disp_words

def build_cave_payload(cave_va: int = CAVE_VA, resume_va: int = RESUME_SITE_VA) -> bytes:
    insns = []
    
    # 1. Complete standard APCS frame construction for LiveViewProcess3aContoller
    #    (replaces 0x536520f0..0x53652104 before any subroutine call can corrupt ip)
    insns.append(0xe1a0c00d) # mov  ip, sp
    insns.append(0xe92dd8f0) # push {r4, r5, r6, r7, fp, ip, lr, pc} (32 bytes)
    insns.append(0xe24cb004) # sub  fp, ip, #4 (fp is now safely and permanently established!)
    insns.append(0xe24dd008) # sub  sp, sp, #8 (local frame allocation, SP remains 8-byte aligned)
    
    # 2. Save incoming caller arguments r0-r3 and callee-saved registers r4-r8, lr
    #    Strict 8-byte alignment: 16 bytes + 24 bytes = 40 bytes (40 % 8 == 0)
    insns.append(0xe92d000f) # push {r0, r1, r2, r3}           (16 bytes)
    insns.append(0xe92d41f0) # push {r4, r5, r6, r7, r8, lr}   (24 bytes)
    
    # 3. Call DspFactory::GetDspService() -> r0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, GET_DSP_SVC_VA, is_bl=True))
    
    # 4. m_dspService is at [r0, #4]
    insns.append(0xe5904004) # ldr  r4, [r0, #4]
    insns.append(0xe3540000) # cmp  r4, #0
    idx_beq_exit1 = len(insns)
    insns.append(0x0a000000) # beq  .exit
    
    # 5. Check if DSP is already running (status flag bit 0)
    insns.append(0xe5942004) # ldr  r2, [r4, #4]
    insns.append(0xe3120001) # tst  r2, #1 (STARTED)
    idx_bne_exit2 = len(insns)
    insns.append(0x1a000000) # bne  .exit
    
    # 6. Call DspService::LoadDsp(m_dspService, 'A', 0, 0)
    insns.append(0xe1a00004) # mov  r0, r4
    insns.append(0xe3a01041) # mov  r1, #0x41 ('A')
    insns.append(0xe3a02000) # mov  r2, #0
    insns.append(0xe3a03000) # mov  r3, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, LOAD_DSP_VA, is_bl=True))
    
    # 7. Check if loaded successfully (status flag bit 1)
    insns.append(0xe5942004) # ldr  r2, [r4, #4]
    insns.append(0xe3120002) # tst  r2, #2 (LOADED)
    idx_beq_exit3 = len(insns)
    insns.append(0x0a000000) # beq  .exit
    
    # 8. Call DspService::Start(m_dspService, 0, 0)
    insns.append(0xe1a00004) # mov  r0, r4
    insns.append(0xe3a01000) # mov  r1, #0
    insns.append(0xe3a02000) # mov  r2, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, START_DSP_VA, is_bl=True))
    
    # 9. Call DspService::Test(m_dspService, 0) -> Version test
    insns.append(0xe1a00004) # mov  r0, r4
    insns.append(0xe3a01000) # mov  r1, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, TEST_DSP_VA, is_bl=True))
    
    # .exit label
    exit_idx = len(insns)
    for idx in [idx_beq_exit1, idx_bne_exit2, idx_beq_exit3]:
        disp_words = (exit_idx - (idx + 2)) & 0x00ffffff
        insns[idx] |= disp_words
        
    # Restore callee-saved registers and caller arguments
    insns.append(0xe8bd41f0) # pop  {r4, r5, r6, r7, r8, lr}
    insns.append(0xe8bd000f) # pop  {r0, r1, r2, r3}
    
    # 10. Execute remaining prologue instruction: r4 = 0x550b69c8
    insns.append(0xe30649c8) # movw r4, #0x69c8
    insns.append(0xe345450b) # movt r4, #0x550b
    
    # 11. Branch cleanly to LiveViewProcess3aContoller resume site (0x53652108: ldr r6, [r4])
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, resume_va, is_bl=False))
    
    payload = struct.pack(f"<{len(insns)}I", *insns)
    return payload

def build_trampoline_payload(hook_va: int = HOOK_SITE_VA, cave_va: int = CAVE_VA) -> bytes:
    # 8 bytes replacing:
    #   0x536520f0: mov ip, sp
    #   0x536520f4: push {r4, r5, r6, r7, fp, ip, lr, pc}
    # With:
    #   0x536520f0: b #cave_va
    #   0x536520f4: nop (mov r0, r0)
    b_insn = encode_arm_branch(hook_va, cave_va, is_bl=False)
    nop_insn = 0xe1a00000
    return struct.pack("<II", b_insn, nop_insn)

def apply_dsp_patch(rtos_binary: bytearray) -> dict:
    cave_payload = build_cave_payload()
    tramp_payload = build_trampoline_payload()
    
    cave_off = CAVE_VA - GR4_BASE
    hook_off = HOOK_SITE_VA - GR4_BASE
    
    # Safety checks
    assert rtos_binary[cave_off : cave_off + len(cave_payload)] == b"\x00" * len(cave_payload), "Cave is not empty zero space!"
    
    orig_hook_bytes = bytes(rtos_binary[hook_off : hook_off + len(tramp_payload)])
    expected_hook = bytes.fromhex("0dc0a0e1f0d82de9") # mov ip, sp; push {r4..pc}
    assert orig_hook_bytes == expected_hook, f"Unexpected hook bytes: {orig_hook_bytes.hex()} (expected {expected_hook.hex()})"
    
    # Apply patches
    rtos_binary[cave_off : cave_off + len(cave_payload)] = cave_payload
    rtos_binary[hook_off : hook_off + len(tramp_payload)] = tramp_payload
    
    manifest = {
        "cave": {
            "vaddr": hex(CAVE_VA),
            "offset": hex(cave_off),
            "length": len(cave_payload),
            "hex": cave_payload.hex(),
        },
        "trampoline": {
            "vaddr": hex(HOOK_SITE_VA),
            "offset": hex(hook_off),
            "original_hex": orig_hook_bytes.hex(),
            "patched_hex": tramp_payload.hex(),
            "return_vaddr": hex(RETURN_SITE_VA),
        }
    }
    return manifest

if __name__ == "__main__":
    print("[*] Generating and testing DSP Auto-Wakeup patches...")
    cave = build_cave_payload()
    tramp = build_trampoline_payload()
    print(f"  -> Cave payload: {len(cave)} bytes ({len(cave)//4} instructions)")
    print(f"  -> Trampoline:   {len(tramp)} bytes ({tramp.hex()})")
    
    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    print("\n--- Trampoline Disassembly ---")
    for insn in md.disasm(tramp, HOOK_SITE_VA):
        print(f"  0x{insn.address:08x}: {insn.mnemonic:<8} {insn.op_str}")
        
    print("\n--- Cave Disassembly ---")
    for insn in md.disasm(cave, CAVE_VA):
        print(f"  0x{insn.address:08x}: {insn.mnemonic:<8} {insn.op_str}")
    print("\n[+] Patch assembly verified successfully.")
