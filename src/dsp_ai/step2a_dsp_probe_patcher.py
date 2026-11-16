#!/usr/bin/env python3
"""
step2a_dsp_probe_patcher.py - Step 2A: Non-Blocking CEVA-XM4 DSP Probe & Visual Telemetry

Isolates DSP boot + IPC handshake from LiveView/AF blocking:
- Reuses Cave 1 + v1.20 DspService ROM/Pool patches from stage3_ai_patcher.py.
- Unlocks FaceDetectionService mutex (r4 + 0x48) during all DSP/SD calls so half-shutter
  never freezes (0 ms blocking).
- Uses non-blocking WaitMessage(..., timeout=0) polling state machine at STATUS_VA:
    state 0: WARMUP (frames 0..19)           -> box (16, 16)
    state 1: LOAD_FAIL / RETRYING          -> box (16, 56)
             START_FAIL                    -> box (16, 96)
             SEND_FAIL                     -> box (16, 136)
    state 2: SEND_ONCE (synthetic RGB @ TENSOR_IN_VA, canary @ TENSOR_OUT_VA)
    state 3: IN_FLIGHT (WaitMessage(0))    -> box (64 + min(poll, 200), 56)
    state 4: DONE_OK (WaitMessage == 0):
             - output canary changed       -> box (220 + min(poll, 15)*4, 96)
             - output canary untouched     -> box (220, 136)
"""

import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

import stage3_ai_patcher as s3

# STATUS_VA (0x52c00100) layout:
# +0x00: state (0=init/warmup, 1=load/start, 2=send, 3=in_flight, 4=done_ok, 5=start_err, 6=send_err, 7=done_no_write)
# +0x04: boot_ticks
# +0x08: seq (u16)
# +0x0c: poll_frames
# +0x10: last_err
# +0x14: canary_diff_cnt
RESP_BUF_VA = s3.TENSOR_MSG_VA + 0x40
CANARY_VAL  = 0x55555555


def build_step2a_cave2_payload(cave_va: int = s3.CAVE2_VA) -> bytes:
    insns = []
    labels = {}
    relocs = []

    def mark(name: str):
        labels[name] = cave_va + len(insns) * 4

    def b(target: str, bl: bool = False, cond: int = 0xe):
        relocs.append((len(insns), target, bl, cond))
        insns.append(0)

    def call(va: int):
        insns.append(s3.encode_arm_branch(cave_va + len(insns) * 4, va, is_bl=True))

    def ldr32(rd: int, val: int):
        s3.emit_ldr_imm32(insns, rd, val)

    def unlock_fd_mutex():
        insns.append(0xe2840048)  # add r0, r4, #0x48
        call(s3.UNLOCK_MUTEX_VA)

    def lock_fd_mutex():
        insns.append(0xe2840048)  # add r0, r4, #0x48
        ldr32(3, s3.MUTEX_COOKIE_VA)
        insns.append(0xe5931000)  # ldr r1, [r3]
        call(s3.LOCK_MUTEX_VA)

    # Prologue (aligned 8B: 8 regs = 32B + 8B scratch = 40B)
    insns.append(0xe92d47f0)  # push {r4, r5, r6, r7, r8, sb, sl, lr}
    insns.append(0xe24dd008)  # sub  sp, sp, #8

    # Guarantee one-shot zeroing of STATUS_VA on this boot via constructor-zeroed [r4, #0x13e]
    ldr32(8, s3.STATUS_VA)
    insns.append(0xe5d4213e)  # ldrb r2, [r4, #0x13e]
    insns.append(0xe3520000)  # cmp  r2, #0
    b("status_inited", cond=0x1)  # bne
    insns.append(0xe3a00000)  # mov  r0, #0
    insns.append(0xe3a01000)  # mov  r1, #0
    mark("zero_st_loop")
    insns.append(0xe7880001)  # str  r0, [r8, r1]
    insns.append(0xe2811004)  # add  r1, r1, #4
    insns.append(0xe3510020)  # cmp  r1, #0x20
    b("zero_st_loop", cond=0xb)  # blt
    insns.append(0xe3a02001)  # mov  r2, #1
    insns.append(0xe5c4213e)  # strb r2, [r4, #0x13e]

    mark("status_inited")
    # Unlock FaceDetectionService mutex BEFORE any DSP/SD operations
    unlock_fd_mutex()

    # Check state in [STATUS_VA + 0]
    insns.append(0xe5987000)  # ldr  r7, [r8, #0] (state)
    insns.append(0xe3570004)  # cmp  r7, #4 (DONE_OK)
    b("show_done_ok", cond=0x0)
    insns.append(0xe3570007)  # cmp  r7, #7 (DONE_NO_WRITE)
    b("show_done_no_write", cond=0x0)
    insns.append(0xe3570005)  # cmp  r7, #5 (START_ERR)
    b("show_start_err", cond=0x0)
    insns.append(0xe3570006)  # cmp  r7, #6 (SEND_ERR)
    b("show_send_err", cond=0x0)
    insns.append(0xe3570003)  # cmp  r7, #3 (IN_FLIGHT)
    b("do_poll_in_flight", cond=0x0)
    insns.append(0xe3570002)  # cmp  r7, #2 (SEND_ONCE)
    b("do_send_probe", cond=0x0)

    # --- States 0 / 1: Warmup & Non-blocking retry LoadDsp + Start ---
    insns.append(0xe5982004)  # ldr  r2, [r8, #4] (boot_ticks)
    insns.append(0xe2822001)  # add  r2, r2, #1
    insns.append(0xe5882004)  # str  r2, [r8, #4]
    insns.append(0xe2523014)  # subs r3, r2, #20
    b("show_warmup", cond=0x3)  # blo (first 20 frames)
    insns.append(0xe3530d1e)  # cmp  r3, #0x780 (30 retries)
    b("show_load_err", cond=0x2)  # bhs
    insns.append(0xe313003f)  # tst  r3, #0x3f
    b("show_load_err", cond=0x1)  # bne (between retries)

    # Get or create DspService
    call(s3.GET_DSP_SVC_VA)
    insns.append(0xe5905004)  # ldr  r5, [r0, #4]
    insns.append(0xe3550000)  # cmp  r5, #0
    b("have_svc_init", cond=0x1)
    insns.append(0xe3a01000)  # mov  r1, #0
    call(s3.CREATE_DSP_SVC_VA)
    ldr32(0, 0x553a3160)
    insns.append(0xe5905004)  # ldr  r5, [r0, #4]
    insns.append(0xe3550000)  # cmp  r5, #0
    b("show_load_err", cond=0x0)

    mark("have_svc_init")
    # Try Drive 'C' then 'I'
    insns.append(0xe1a00005)  # mov  r0, r5
    insns.append(0xe3a01043)  # mov  r1, #'C'
    insns.append(0xe3a02000)
    insns.append(0xe3a03000)
    call(s3.LOAD_DSP_VA)
    insns.append(0xe3500000)
    b("load_ok", cond=0x1)

    insns.append(0xe1a00005)
    insns.append(0xe3a01049)  # mov  r1, #'I'
    insns.append(0xe3a02000)
    insns.append(0xe3a03000)
    call(s3.LOAD_DSP_VA)
    insns.append(0xe3500000)
    b("load_ok", cond=0x1)

    # Load failed
    insns.append(0xe3a02000)
    insns.append(0xe5852004)  # clear false loaded bit
    insns.append(0xe3a00001)
    insns.append(0xe5880000)  # state = 1
    b("show_load_err")

    mark("load_ok")
    # Start DSP @ 0x52100000
    insns.append(0xe1a00005)
    ldr32(1, 0x52100000)
    insns.append(0xe3a02000)
    call(s3.START_DSP_VA)
    insns.append(0xe5880010)  # last_err = Start() ret
    insns.append(0xe5952004)  # ldr  r2, [r5, #4]
    insns.append(0xe3120004)  # tst  r2, #4 (STARTED)
    b("start_ok", cond=0x1)
    insns.append(0xe3a00005)  # state = 5 (START_ERR)
    insns.append(0xe5880000)
    b("show_start_err")

    mark("start_ok")
    insns.append(0xe3a00002)  # state = 2 (SEND_ONCE)
    insns.append(0xe5880000)
    b("show_warmup")  # yield 1 frame before first SendMessage

    # --- State 2: Prepare deterministic synthetic input + canary output + SendMessage ---
    mark("do_send_probe")
    call(s3.GET_DSP_SVC_VA)
    insns.append(0xe5905004)  # r5 = m_dspService

    # Fill TENSOR_IN_VA (0x52b00000, size 0x2a00) with 0x80 neutral pattern,
    # and TENSOR_OUT_VA (0x52b02a00, size 0x5400) with 0x55555555 canary
    ldr32(0, s3.TENSOR_IN_VA)
    ldr32(1, 0x80808080)
    ldr32(2, 0x2a00)
    mark("fill_in_loop")
    insns.append(0xe2522004)  # subs r2, r2, #4
    insns.append(0xe7801002)  # str  r1, [r0, r2]
    b("fill_in_loop", cond=0x1)  # bne

    ldr32(0, s3.TENSOR_OUT_VA)
    ldr32(1, CANARY_VAL)
    ldr32(2, 0x5400)
    mark("fill_out_loop")
    insns.append(0xe2522004)
    insns.append(0xe7801002)
    b("fill_out_loop", cond=0x1)

    # Flush ARM D-Cache for 0x52b00000, size 0xee00
    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, s3.TENSOR_IN_VA)
    insns.append(0xe3a02cee)  # mov r2, #0xee00
    call(s3.FLUSH_DCACHE_VA)

    # Populate 10-word DspCommandParameter @ TENSOR_MSG_VA (0x52c00000)
    ldr32(7, s3.TENSOR_MSG_VA)
    ldr32(0, s3.DESC_TABLE_VA)
    insns.append(0xe5870000)
    insns.append(0xe3a00000)
    insns.append(0xe5870004)
    insns.append(0xe5870020)
    insns.append(0xe5870024)
    ldr32(0, s3.TENSOR_IN_VA)
    insns.append(0xe5870008)
    ldr32(0, s3.HEAD_WEIGHTS_VA)
    insns.append(0xe587000c)
    ldr32(0, s3.MEAN_WEIGHTS_VA)
    insns.append(0xe5870010)
    ldr32(0, 0x00080808)
    insns.append(0xe5870014)
    ldr32(0, 0x0c0c0c0a)
    insns.append(0xe5870018)
    ldr32(0, 0x0c0c0c0c)
    insns.append(0xe587001c)

    # Call SendMessage(svc, &seq, msg_id=4, timeout=100, &msg)
    insns.append(0xe2886008)  # r6 = STATUS_VA + 8 (&seq)
    insns.append(0xe3a00000)
    insns.append(0xe1c600b0)  # strh r0, [r6]
    insns.append(0xe1a00005)  # mov  r0, r5
    insns.append(0xe1a01006)  # mov  r1, r6
    insns.append(0xe3a02004)  # mov  r2, #4
    insns.append(0xe3a03064)  # mov  r3, #100
    insns.append(0xe58d7000)  # str  r7, [sp]
    call(s3.SEND_MSG_VA)
    insns.append(0xe5880010)  # last_err = SendMessage ret
    insns.append(0xe3500000)
    b("send_ok", cond=0x0)
    insns.append(0xe3a00006)  # state = 6 (SEND_ERR)
    insns.append(0xe5880000)
    b("show_send_err")

    mark("send_ok")
    insns.append(0xe3a00000)
    insns.append(0xe588000c)  # poll_frames = 0
    insns.append(0xe3a00003)  # state = 3 (IN_FLIGHT)
    insns.append(0xe5880000)
    b("show_in_flight")

    # --- State 3: Non-blocking WaitMessage(..., timeout=0) ---
    mark("do_poll_in_flight")
    call(s3.GET_DSP_SVC_VA)
    insns.append(0xe5905004)  # r5 = svc
    insns.append(0xe2886008)  # r6 = STATUS_VA + 8
    insns.append(0xe1a00005)  # r0 = svc
    insns.append(0xe1d610b0)  # r1 = seq
    insns.append(0xe3a02000)  # r2 = 0 (NON-BLOCKING TMO_POL!)
    ldr32(3, RESP_BUF_VA)
    call(s3.WAIT_MSG_VA)
    insns.append(0xe5880010)  # last_err = WaitMessage ret
    insns.append(0xe3500000)
    b("wait_got_reply", cond=0x0)

    # Still pending (-3)
    insns.append(0xe598200c)  # poll_frames++
    insns.append(0xe2822001)
    insns.append(0xe588200c)
    b("show_in_flight")

    mark("wait_got_reply")
    # Invalidate ARM D-Cache on TENSOR_OUT_VA (0x52b02a00, size 0x5400)
    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, s3.TENSOR_OUT_VA)
    insns.append(0xe3a02b15)  # mov r2, #0x5400
    call(s3.INVAL_DCACHE_VA)

    # Check if any word in first 256 bytes of TENSOR_OUT_VA changed from CANARY_VAL
    ldr32(0, s3.TENSOR_OUT_VA)
    ldr32(1, CANARY_VAL)
    insns.append(0xe3a02000)  # off = 0
    insns.append(0xe3a03000)  # diff_cnt = 0
    mark("chk_canary_loop")
    insns.append(0xe7907002)  # ldr r7, [r0, r2]
    insns.append(0xe1570001)  # cmp r7, r1
    insns.append(0x12833001)  # addne r3, r3, #1
    insns.append(0xe2822004)  # add r2, r2, #4
    insns.append(0xe3520c01)  # cmp r2, #0x100
    b("chk_canary_loop", cond=0xb)

    insns.append(0xe5883014)  # canary_diff_cnt
    insns.append(0xe3530000)
    b("done_no_write", cond=0x0)
    insns.append(0xe3a00004)  # state = 4 (DONE_OK)
    insns.append(0xe5880000)
    b("show_done_ok")

    mark("done_no_write")
    insns.append(0xe3a00007)  # state = 7 (DONE_NO_WRITE)
    insns.append(0xe5880000)
    b("show_done_no_write")

    # --- Visual Telemetry Coordinates (px -> Q8 below) ---
    mark("show_warmup")
    insns.append(0xe3a07010)  # x = 16
    insns.append(0xe3a08010)  # y = 16
    b("render_box")

    mark("show_load_err")
    insns.append(0xe3a07010)  # x = 16
    insns.append(0xe3a08038)  # y = 56
    b("render_box")

    mark("show_start_err")
    insns.append(0xe3a07010)  # x = 16
    insns.append(0xe3a08060)  # y = 96
    b("render_box")

    mark("show_send_err")
    insns.append(0xe3a07010)  # x = 16
    insns.append(0xe3a08088)  # y = 136
    b("render_box")

    mark("show_in_flight")
    # x = 64 + min(poll_frames, 200), y = 56
    insns.append(0xe598200c)  # r2 = poll_frames
    insns.append(0xe35200c8)  # cmp r2, #200
    insns.append(0xc3a020c8)  # movgt r2, #200
    insns.append(0xe2827040)  # add r7, r2, #64
    insns.append(0xe3a08038)  # y = 56
    b("render_box")

    mark("show_done_ok")
    # x = 220 + min(poll_frames, 15)*4, y = 96
    insns.append(0xe598200c)
    insns.append(0xe352000f)  # cmp r2, #15
    insns.append(0xc3a0200f)  # movgt r2, #15
    insns.append(0xe3a070dc)  # mov r7, #220
    insns.append(0xe0877102)  # add r7, r7, r2, lsl #2
    insns.append(0xe3a08060)  # y = 96
    b("render_box")

    mark("show_done_no_write")
    insns.append(0xe3a070dc)  # x = 220
    insns.append(0xe3a08088)  # y = 136

    # --- Render Step-1-verified 32x32 Q8 Box & Re-lock Mutex ---
    mark("render_box")
    lock_fd_mutex()

    insns.append(0xe1a07407)  # lsl r7, r7, #8 (left Q8)
    insns.append(0xe1a08408)  # lsl r8, r8, #8 (top Q8)
    insns.append(s3.encode_movw(5, 0x2000))  # box_size = 32 px in Q8 (> 0x1e00 floor)

    insns.append(0xe594a078)  # ldr sl, [r4, #0x78]
    insns.append(0xe35a0000)
    b("exit_done", cond=0x0)

    # Zero 100 bytes of m_DetectedFaceResult[0]
    insns.append(0xe3a00000)
    insns.append(0xe3a01000)
    mark("zero_face_loop")
    insns.append(0xe78a0001)
    insns.append(0xe2811004)
    insns.append(0xe3510064)
    b("zero_face_loop", cond=0xb)

    insns.append(0xe58a7000)  # left
    insns.append(0xe58a8004)  # top
    insns.append(0xe58a5008)  # width
    insns.append(0xe58a500c)  # height
    insns.append(s3.encode_movw(9, 95))
    insns.append(0xe58a904c)  # score = 95
    insns.append(0xe5941118)
    insns.append(0xe3510000)
    insns.append(0x03a01001)
    insns.append(0xe58a1050)  # face_id

    # Seed tracking state
    insns.append(0xe08720a5)
    insns.append(0xe5842100)  # cx
    insns.append(0xe08820a5)
    insns.append(0xe5842104)  # cy
    insns.append(0xe5845108)  # prev_w
    insns.append(0xe584510c)  # prev_h
    insns.append(0xe5841118)
    insns.append(0xe584111c)
    insns.append(0xe3a02001)
    insns.append(0xe5c4213c)  # tracking_active = 1
    insns.append(0xe3a00000)
    insns.append(0xe5c40121)  # is_stabilizing = 0
    insns.append(0xe2843f49)
    insns.append(0xe1c300b0)  # lost_frames_counter = 0

    insns.append(0xe5943038)  # FdLibraryService*
    insns.append(0xe3530000)
    b("exit_done", cond=0x0)
    insns.append(0xe3a01001)
    insns.append(0xe5831030)  # face_count = 1
    insns.append(0xe3a01000)
    insns.append(0xe5831034)  # main_face_idx = 0

    mark("exit_done")
    insns.append(0xe28dd008)  # add sp, sp, #8
    insns.append(0xe8bd47f0)  # pop {r4, r5, r6, r7, r8, sb, sl, lr}
    insns.append(s3.encode_arm_branch(cave_va + len(insns) * 4, s3.HOOK2_AI_RESUME_VA))

    for idx, target, is_bl, cond in relocs:
        insns[idx] = s3.encode_arm_branch(cave_va + idx * 4, labels[target], is_bl=is_bl, cond=cond)

    return struct.pack(f"<{len(insns)}I", *insns)


def apply_step2a_patch(rtos_binary: bytearray, rtos_payload_offset: int = s3.RTOS_PAYLOAD_START) -> dict:
    # Temporarily swap stage3's build_cave2_payload with Step 2A probe cave so all ROM/Pool/Cave1 patches stay identical
    orig_builder = s3.build_cave2_payload
    s3.build_cave2_payload = build_step2a_cave2_payload
    try:
        manifest = s3.apply_stage3_patch(rtos_binary, rtos_payload_offset=rtos_payload_offset)
    finally:
        s3.build_cave2_payload = orig_builder
    manifest["stage"] = "Step 2A: Non-Blocking CEVA-XM4 DSP Probe & Visual Telemetry"
    return manifest


if __name__ == "__main__":
    c2 = build_step2a_cave2_payload()
    print(f"Step 2A Cave 2: {len(c2)} bytes @ {hex(s3.CAVE2_VA)} (margin to Desc: {s3.DESC_TABLE_VA - (s3.CAVE2_VA + len(c2))} B)")
    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    for ins in md.disasm(c2[:64], s3.CAVE2_VA):
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")
