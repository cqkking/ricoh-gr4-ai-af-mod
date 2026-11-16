#!/usr/bin/env python3
"""
step1_osd_mock_patcher.py - Step 1: Pure OSD / PDAF Canonical Injection Verification

Single-variable verification patch for Ricoh GR IV v1.20:
- Completely omits DspService instantiation, SD card DF_*.BIN loading, and DSP IPC.
- Hooks FaceDetectionService::Task @ 0x5389639c -> Cave @ 0x53000e00.
- Reads active frame dimensions [r4, #0xfc] (fd_w) x [r4, #0xfe] (fd_h) dynamically
  (supporting 3:2, 4:3, 16:9, and 1:1 aspect ratios) and injects a centered 48x48 px
  (0x3000 in Q8 fixed-point) face box into m_DetectedFaceResult[0] ([r4, #0x78]).
- Seeds FaceDetectionService tracking state and FdLibraryService face_count=1, then
  resumes at 0x538963ac to unlock mutex r6, run UpdateFaceDetectionResult (0x538959f4),
  and dispatch EVENT 0x04070001 to LCD OSD & PDAF.
"""

import struct
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

GR4_BASE           = 0x53000000
RTOS_PAYLOAD_START = 0x00017D28

CAVE_VA            = 0x53000e00
HOOK2_SITE_VA      = 0x5389639c  # FaceDetectionService::Task detect call site
HOOK2_AI_RESUME_VA = 0x538963ac  # OsMutex::Unlock(r6) + UpdateFaceDetectionResult(r4)

BOX_SIZE_Q8        = 0x3000      # 48 px in Q8 fixed-point (> 0x1e00 SelectMainFace floor)
CONFIDENCE_SCORE   = 95


def encode_arm_branch(from_va: int, to_va: int, is_bl: bool = False, cond: int = 0xe) -> int:
    disp = to_va - (from_va + 8)
    disp_words = (disp >> 2) & 0x00ffffff
    op = (cond << 28) | (0x0b000000 if is_bl else 0x0a000000)
    return op | disp_words


def encode_movw(rd: int, imm16: int, cond: int = 0xe) -> int:
    imm4 = (imm16 >> 12) & 0xf
    imm12 = imm16 & 0xfff
    return (cond << 28) | 0x03000000 | (imm4 << 16) | (rd << 12) | imm12


def build_step1_cave(cave_va: int = CAVE_VA, box_size_q8: int = BOX_SIZE_Q8) -> bytes:
    """Assemble the pure OSD/PDAF mock injection cave at cave_va."""
    insns = []
    labels = {}
    relocs = []

    def mark_label(name: str):
        labels[name] = cave_va + len(insns) * 4

    def branch_to(target_label: str, is_bl: bool = False, cond: int = 0xe):
        idx = len(insns)
        relocs.append((idx, target_label, is_bl, cond))
        insns.append(0)

    # Preserve callee-saved registers & lr (32 bytes, 8-byte stack aligned)
    insns.append(0xe92d47f0)  # push {r4, r5, r6, r7, r8, sb, sl, lr}

    # Fetch m_DetectedFaceResult ([r4, #0x78]); abort cleanly if NULL
    insns.append(0xe594a078)  # ldr  sl, [r4, #0x78]
    insns.append(0xe35a0000)  # cmp  sl, #0
    branch_to("exit_done", cond=0x0)  # beq .exit_done

    # Read current frame fd_w ([r4, #0xfc]) and fd_h ([r4, #0xfe]) with safe defaults
    insns.append(0xe1d47fbc)  # ldrh r7, [r4, #0xfc]
    insns.append(0xe3570000)  # cmp  r7, #0
    insns.append(encode_movw(7, 320, cond=0x0))  # moveq r7, #320
    insns.append(0xe1d48fbe)  # ldrh r8, [r4, #0xfe]
    insns.append(0xe3580000)  # cmp  r8, #0
    insns.append(encode_movw(8, 212, cond=0x0))  # moveq r8, #212

    # Compute centered Q8 rect: left = ((fd_w << 8) - box) >> 1, top = ((fd_h << 8) - box) >> 1
    insns.append(encode_movw(5, box_size_q8))  # movw r5, #box_size_q8
    insns.append(0xe1a07407)  # lsl  r7, r7, #8
    insns.append(0xe1a08408)  # lsl  r8, r8, #8
    insns.append(0xe0570005)  # subs r0, r7, r5
    insns.append(0x43a00000)  # movmi r0, #0
    insns.append(0xe1a070a0)  # lsr  r7, r0, #1 (left)
    insns.append(0xe0581005)  # subs r1, r8, r5
    insns.append(0x43a01000)  # movmi r1, #0
    insns.append(0xe1a080a1)  # lsr  r8, r1, #1 (top)

    # Zero all 100 bytes (0x64) of m_DetectedFaceResult[0]
    insns.append(0xe3a00000)  # mov  r0, #0
    insns.append(0xe3a01000)  # mov  r1, #0
    mark_label("zero_face_loop")
    insns.append(0xe78a0001)  # str  r0, [sl, r1]
    insns.append(0xe2811004)  # add  r1, r1, #4
    insns.append(0xe3510064)  # cmp  r1, #0x64
    branch_to("zero_face_loop", cond=0xb)  # blt .zero_face_loop

    # Populate m_DetectedFaceResult[0]: face rect + confidence + face_id
    insns.append(0xe58a7000)  # str  r7, [sl, #0x00] (left)
    insns.append(0xe58a8004)  # str  r8, [sl, #0x04] (top)
    insns.append(0xe58a5008)  # str  r5, [sl, #0x08] (width)
    insns.append(0xe58a500c)  # str  r5, [sl, #0x0c] (height)
    insns.append(encode_movw(9, CONFIDENCE_SCORE))  # movw sb, #95
    insns.append(0xe58a904c)  # str  sb, [sl, #0x4c] (confidence)
    insns.append(0xe5941118)  # ldr  r1, [r4, #0x118]
    insns.append(0xe3510000)  # cmp  r1, #0
    insns.append(0x03a01001)  # moveq r1, #1
    insns.append(0xe58a1050)  # str  r1, [sl, #0x50] (face_id)

    # Seed FaceDetectionService tracking state so SelectMainFace never rejects or fades out
    insns.append(0xe08720a5)  # add  r2, r7, r5, lsr #1 (cx)
    insns.append(0xe5842100)  # str  r2, [r4, #0x100]
    insns.append(0xe08820a5)  # add  r2, r8, r5, lsr #1 (cy)
    insns.append(0xe5842104)  # str  r2, [r4, #0x104]
    insns.append(0xe5845108)  # str  r5, [r4, #0x108]
    insns.append(0xe584510c)  # str  r5, [r4, #0x10c]
    insns.append(0xe5841118)  # str  r1, [r4, #0x118]
    insns.append(0xe584111c)  # str  r1, [r4, #0x11c]
    insns.append(0xe3a02001)  # mov  r2, #1
    insns.append(0xe5c4213c)  # strb r2, [r4, #0x13c] (tracking_active = 1)
    insns.append(0xe3a00000)  # mov  r0, #0
    insns.append(0xe5c40121)  # strb r0, [r4, #0x121] (is_stabilizing = 0)
    insns.append(0xe2843f49)  # add  r3, r4, #0x124
    insns.append(0xe1c300b0)  # strh r0, [r3]         (lost_frames_counter = 0)

    # Update FdLibraryService ([r4, #0x38]) face_count = 1, main_face_idx = 0
    insns.append(0xe5943038)  # ldr  r3, [r4, #0x38]
    insns.append(0xe3530000)  # cmp  r3, #0
    branch_to("exit_done", cond=0x0)  # beq .exit_done
    insns.append(0xe3a01001)  # mov  r1, #1
    insns.append(0xe5831030)  # str  r1, [r3, #0x30]
    insns.append(0xe3a01000)  # mov  r1, #0
    insns.append(0xe5831034)  # str  r1, [r3, #0x34]

    mark_label("exit_done")
    insns.append(0xe8bd47f0)  # pop  {r4, r5, r6, r7, r8, sb, sl, lr}
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, HOOK2_AI_RESUME_VA, is_bl=False))

    for idx, target_label, is_bl, cond in relocs:
        target_va = labels[target_label]
        from_va = cave_va + idx * 4
        insns[idx] = encode_arm_branch(from_va, target_va, is_bl=is_bl, cond=cond)

    return struct.pack(f"<{len(insns)}I", *insns)


def build_tramp2_payload(hook_va: int = HOOK2_SITE_VA, cave_va: int = CAVE_VA) -> bytes:
    b_insn = encode_arm_branch(hook_va, cave_va, is_bl=False)
    nop_insn = 0xe1a00000
    return struct.pack("<II", b_insn, nop_insn)


def apply_step1_patch(payload_buf: bytearray, rtos_payload_offset: int = RTOS_PAYLOAD_START) -> dict:
    """Inject Step 1 OSD/PDAF mock cave and trampoline into decoded v1.20 payload."""
    cave_payload = build_step1_cave()
    tramp_payload = build_tramp2_payload()

    cave_off = CAVE_VA - GR4_BASE + rtos_payload_offset
    tramp_off = HOOK2_SITE_VA - GR4_BASE + rtos_payload_offset

    assert CAVE_VA + len(cave_payload) <= 0x53002000, "Cave exceeds ROM zero block!"
    assert payload_buf[cave_off : cave_off + len(cave_payload)] == b"\x00" * len(cave_payload), (
        "Destination cave is not empty zero space!"
    )

    orig_hook2 = bytes(payload_buf[tramp_off : tramp_off + len(tramp_payload)])
    expected_hook2 = bytes.fromhex("380094e57b40ffeb")  # ldr r0, [r4, #0x38]; bl #0x53866594
    assert orig_hook2 == expected_hook2, (
        f"Unexpected hook2 bytes: {orig_hook2.hex()} (expected {expected_hook2.hex()})"
    )

    payload_buf[cave_off : cave_off + len(cave_payload)] = cave_payload
    payload_buf[tramp_off : tramp_off + len(tramp_payload)] = tramp_payload

    return {
        "stage": "Step 1: Pure OSD / PDAF Canonical Injection Verification",
        "baseline": "Official Ricoh GR IV Firmware v1.20",
        "hook": {
            "site_vaddr": hex(HOOK2_SITE_VA),
            "offset": hex(tramp_off),
            "original_hex": orig_hook2.hex(),
            "patched_hex": tramp_payload.hex(),
            "resume_vaddr": hex(HOOK2_AI_RESUME_VA),
        },
        "code_cave": {
            "vaddr": hex(CAVE_VA),
            "offset": hex(cave_off),
            "length": len(cave_payload),
            "box_size_q8": hex(BOX_SIZE_Q8),
        },
    }


if __name__ == "__main__":
    c = build_step1_cave()
    t = build_tramp2_payload()
    print(f"Step 1 Cave: {len(c)} bytes @ {hex(CAVE_VA)}")
    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    for ins in md.disasm(c, CAVE_VA):
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")
    print("Trampoline:")
    for ins in md.disasm(t, HOOK2_SITE_VA):
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")
