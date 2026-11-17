#!/usr/bin/env python3
"""
step2c_diag_probe_patcher.py - Step 2C: Crash-Proof Continuous AI Probe + Oscillating Heartbeat

Fixes & diagnostics over Step 2B:
1. No fall-through from inference_done -> state_idle_send (separates harvest frame and send frame).
2. Strict DDR pointer validation (0x40000000 <= Y < 0x60000000) before any AHIP dereference,
   plus luma-only safe mode (U=0, V=0) to eliminate any invalid U/V pointer Data Abort.
3. Oscillating heartbeat box (x = 220 + (boot_ticks & 0x1f), y = 140) so we can visually see
   whether Cave 2 + continuous DSP loop stays alive past 1 second!
4. Diagnostic corner boxes:
   - (16, 56): LoadDsp fail OR AHIP Y-plane NULL/invalid
   - (16, 96): Start fail
   - (16, 136): SendMessage fail OR WaitMessage(0) stalled > 90 frames
5. Lowers detection threshold slightly to (max_diff >= 2, area_cnt >= 1) @ 0x52b09a00.
"""

import struct
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM
import stage3_ai_patcher as s3

TENSOR_OUT_REAL_VA = s3.TENSOR_IN_VA + 0x9a00  # 0x52b09a00
RESP_BUF_VA        = s3.TENSOR_MSG_VA + 0x40


def build_step2c_cave2_payload(cave_va: int = s3.CAVE2_VA) -> bytes:
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
        insns.append(0xe2840048)
        call(s3.UNLOCK_MUTEX_VA)

    def lock_fd_mutex():
        insns.append(0xe2840048)
        ldr32(3, s3.MUTEX_COOKIE_VA)
        insns.append(0xe5931000)
        call(s3.LOCK_MUTEX_VA)

    insns.append(0xe92d47f0)  # push {r4, r5, r6, r7, r8, sb, sl, lr}
    insns.append(0xe24dd008)  # sub  sp, sp, #8

    ldr32(8, s3.STATUS_VA)
    insns.append(0xe5d4213e)
    insns.append(0xe3520000)
    b("status_inited", cond=0x1)
    insns.append(0xe3a00000)
    insns.append(0xe3a01000)
    mark("zero_st_loop")
    insns.append(0xe7880001)
    insns.append(0xe2811004)
    insns.append(0xe3510028)
    b("zero_st_loop", cond=0xb)
    insns.append(0xe3a00b04)  # 0x1000 (16 px Q8)
    insns.append(0xe5880010)
    insns.append(0xe5880014)
    insns.append(s3.encode_movw(0, 0x2000))
    insns.append(0xe5880018)
    insns.append(0xe3a00050)
    insns.append(0xe588001c)
    insns.append(0xe3a02001)
    insns.append(0xe5c4213e)

    mark("status_inited")
    # Increment global frame counter boot_ticks every entry
    insns.append(0xe5982004)
    insns.append(0xe2822001)
    insns.append(0xe5882004)

    unlock_fd_mutex()

    insns.append(0xe5987000)  # state
    insns.append(0xe3570003)
    b("state_in_flight", cond=0x0)
    insns.append(0xe3570002)
    b("state_idle_send", cond=0x0)

    # --- State 0: Warmup + LoadDsp + Start ---
    insns.append(0xe5982004)
    insns.append(0xe2523014)  # subs r3, r2, #20
    b("use_cached_box", cond=0x3)
    insns.append(0xe3530d1e)
    b("diag_load_err", cond=0x2)
    insns.append(0xe313003f)
    b("diag_load_err", cond=0x1)

    call(s3.GET_DSP_SVC_VA)
    insns.append(0xe5905004)
    insns.append(0xe3550000)
    b("have_svc_init", cond=0x1)
    insns.append(0xe3a01000)
    call(s3.CREATE_DSP_SVC_VA)
    ldr32(0, 0x553a3160)
    insns.append(0xe5905004)
    insns.append(0xe3550000)
    b("diag_load_err", cond=0x0)

    mark("have_svc_init")
    insns.append(0xe1a00005)
    insns.append(0xe3a01043)
    insns.append(0xe3a02000)
    insns.append(0xe3a03000)
    call(s3.LOAD_DSP_VA)
    insns.append(0xe3500000)
    b("load_ok", cond=0x1)

    insns.append(0xe1a00005)
    insns.append(0xe3a01049)
    insns.append(0xe3a02000)
    insns.append(0xe3a03000)
    call(s3.LOAD_DSP_VA)
    insns.append(0xe3500000)
    b("load_ok", cond=0x1)

    insns.append(0xe3a02000)
    insns.append(0xe5852004)
    b("diag_load_err")

    mark("load_ok")
    insns.append(0xe1a00005)
    ldr32(1, 0x52100000)
    insns.append(0xe3a02000)
    call(s3.START_DSP_VA)
    insns.append(0xe5952004)
    insns.append(0xe3120004)
    b("diag_start_err", cond=0x0)

    insns.append(0xe3a00002)
    insns.append(0xe5880000)
    b("use_cached_box")

    # --- State 3: Non-blocking WaitMessage(timeout=0) ---
    mark("state_in_flight")
    call(s3.GET_DSP_SVC_VA)
    insns.append(0xe5905004)
    insns.append(0xe2886008)
    insns.append(0xe1a00005)
    insns.append(0xe1d610b0)
    insns.append(0xe3a02000)
    ldr32(3, RESP_BUF_VA)
    call(s3.WAIT_MSG_VA)
    insns.append(0xe3500000)
    b("inference_done", cond=0x0)

    insns.append(0xe598200c)
    insns.append(0xe2822001)
    insns.append(0xe588200c)
    insns.append(0xe352005a)
    b("diag_stall_err", cond=0xc)
    b("use_cached_box")

    mark("inference_done")
    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, TENSOR_OUT_REAL_VA)
    insns.append(0xe3a02b15)  # 0x5400
    call(s3.INVAL_DCACHE_VA)

    ldr32(0, TENSOR_OUT_REAL_VA)
    b("sub_find_peak", bl=True)

    insns.append(0xe3500002)  # max_diff >= 2
    b("set_heartbeat", cond=0xb)
    insns.append(0xe3530001)  # area_cnt >= 1
    b("set_heartbeat", cond=0xb)

    # Subject detected!
    insns.append(s3.encode_movw(5, 0x2000))
    insns.append(0xe0855283)
    insns.append(s3.encode_movw(6, 0x4800))
    insns.append(0xe1550006)
    insns.append(0xc1a05006)

    insns.append(0xe280901e)
    insns.append(0xe3590064)
    insns.append(0xc3a09064)

    insns.append(0xe1d43fbc)
    insns.append(0xe3530000)
    insns.append(s3.encode_movw(3, 320, cond=0x0))
    insns.append(0xe0070391)
    insns.append(0xe3a03049)
    insns.append(0xe0070793)
    insns.append(0xe1a07227)

    insns.append(0xe1d43fbe)
    insns.append(0xe3530000)
    insns.append(s3.encode_movw(3, 212, cond=0x0))
    insns.append(0xe00c0392)
    insns.append(0xe3a03049)
    insns.append(0xe00c0c93)
    insns.append(0xe1a0c22c)

    insns.append(0xe05770a5)
    insns.append(0x43a07000)
    insns.append(0xe05c60a5)
    insns.append(0x43a06000)
    b("store_cached")

    mark("set_heartbeat")
    # Oscillating heartbeat: x = 220 + (boot_ticks & 0x1f), y = 140 (in Q8)
    insns.append(0xe5987004)  # r7 = boot_ticks
    insns.append(0xe207701f)  # r7 &= 0x1f
    insns.append(0xe28770dc)  # r7 += 220
    insns.append(0xe1a07407)  # r7 <<= 8
    ldr32(6, 0x08c00)         # y = 140 px Q8
    insns.append(s3.encode_movw(5, 0x2000))
    insns.append(0xe3a09063)

    mark("store_cached")
    insns.append(0xe5887010)
    insns.append(0xe5886014)
    insns.append(0xe5885018)
    insns.append(0xe588901c)
    insns.append(0xe3a00002)  # state = 2 (IDLE_READY for next frame; DO NOT fall through!)
    insns.append(0xe5880000)
    b("use_cached_box")

    # --- State 2: Safe AHIP Luma Extraction -> 56x56 RGB -> SendMessage ---
    mark("state_idle_send")
    ldr32(3, s3.AHIP_DRV_BASE_VA)
    insns.append(0xe593900c)  # slot byte size
    insns.append(0xe5932424)  # *(*(0x55081350 + 0x424))
    insns.append(0xe3520000)
    insns.append(0x15922000)
    insns.append(0xe3520000)
    b("try_ahip_434", cond=0x0)
    insns.append(0xe5926004)  # Y_plane
    b("check_y_ptr")

    mark("try_ahip_434")
    insns.append(0xe5932434)
    insns.append(0xe3520000)
    insns.append(0x15922000)
    insns.append(0xe3520000)
    b("try_ahip_legacy", cond=0x0)
    insns.append(0xe5926004)
    b("check_y_ptr")

    mark("try_ahip_legacy")
    insns.append(0xe593201c)
    insns.append(0xe3520000)
    insns.append(0x05932014)
    insns.append(0xe3520000)
    b("diag_load_err", cond=0x0)
    insns.append(0xe592600c)

    mark("check_y_ptr")
    # Strict DDR address guard: 0x40000000 <= Y < 0x60000000
    insns.append(0xe3560101)  # cmp r6, #0x40000000
    b("diag_load_err", cond=0x3)  # blo
    insns.append(0xe3560206)  # cmp r6, #0x60000000
    b("diag_load_err", cond=0x2)  # bhs

    # Safe luma-only downsampling (U = 0, V = 0) to guarantee zero invalid U/V dereferences
    insns.append(0xe3590000)
    insns.append(0x03a09802)
    call(s3.GET_CACHE_MGR_VA)
    insns.append(0xe1a01006)
    insns.append(0xe1a02009)
    call(s3.INVAL_DCACHE_VA)

    insns.append(0xe1d4cfbc)
    insns.append(0xe35c0000)
    insns.append(s3.encode_movw(12, 320, cond=0x0))
    insns.append(0xe1d49fbe)
    insns.append(0xe3590000)
    insns.append(s3.encode_movw(9, 212, cond=0x0))
    insns.append(0xe1a00006)  # Y
    ldr32(1, s3.TENSOR_IN_VA)
    insns.append(0xe3a02000)  # U = 0 (safe luma-only)
    insns.append(0xe3a03000)  # V = 0 (safe luma-only)
    b("sub_downsample", bl=True)

    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, s3.TENSOR_IN_VA)
    insns.append(0xe3a02cee)
    call(s3.FLUSH_DCACHE_VA)

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

    call(s3.GET_DSP_SVC_VA)
    insns.append(0xe5905004)
    insns.append(0xe2886008)
    insns.append(0xe3a00000)
    insns.append(0xe1c600b0)
    insns.append(0xe1a00005)
    insns.append(0xe1a01006)
    insns.append(0xe3a02004)
    insns.append(0xe3a03064)
    insns.append(0xe58d7000)
    call(s3.SEND_MSG_VA)
    insns.append(0xe3500000)
    b("diag_send_err", cond=0x1)

    insns.append(0xe3a00000)
    insns.append(0xe588000c)
    insns.append(0xe3a00003)  # IN_FLIGHT
    insns.append(0xe5880000)
    b("use_cached_box")

    # --- Diagnostic boxes ---
    mark("diag_load_err")
    insns.append(0xe3a07b04)
    insns.append(s3.encode_movw(6, 0x3800))
    b("render_diag")

    mark("diag_start_err")
    insns.append(0xe3a07b04)
    insns.append(s3.encode_movw(6, 0x6000))
    b("render_diag")

    mark("diag_send_err")
    mark("diag_stall_err")
    insns.append(0xe3a07b04)
    insns.append(s3.encode_movw(6, 0x8800))

    mark("render_diag")
    insns.append(s3.encode_movw(5, 0x2000))
    insns.append(0xe3a09058)
    b("render_box")

    mark("use_cached_box")
    insns.append(0xe5987010)
    insns.append(0xe5986014)
    insns.append(0xe5985018)
    insns.append(0xe598901c)

    mark("render_box")
    insns.append(0xe58d6004)
    lock_fd_mutex()
    insns.append(0xe59d6004)

    insns.append(0xe594a078)
    insns.append(0xe35a0000)
    b("exit_done", cond=0x0)

    insns.append(0xe3a00000)
    insns.append(0xe3a01000)
    mark("zero_face_loop")
    insns.append(0xe78a0001)
    insns.append(0xe2811004)
    insns.append(0xe3510064)
    b("zero_face_loop", cond=0xb)

    insns.append(0xe58a7000)
    insns.append(0xe58a6004)
    insns.append(0xe58a5008)
    insns.append(0xe58a500c)
    insns.append(0xe58a904c)
    insns.append(0xe5941118)
    insns.append(0xe3510000)
    insns.append(0x03a01001)
    insns.append(0xe58a1050)

    insns.append(0xe08720a5)
    insns.append(0xe5842100)
    insns.append(0xe08620a5)
    insns.append(0xe5842104)
    insns.append(0xe5845108)
    insns.append(0xe584510c)
    insns.append(0xe5841118)
    insns.append(0xe584111c)
    insns.append(0xe3a02001)
    insns.append(0xe5c4213c)
    insns.append(0xe3a00000)
    insns.append(0xe5c40121)
    insns.append(0xe2843f49)
    insns.append(0xe1c300b0)

    insns.append(0xe5943038)
    insns.append(0xe3530000)
    b("exit_done", cond=0x0)
    insns.append(0xe3a01001)
    insns.append(0xe5831030)
    insns.append(0xe3a01000)
    insns.append(0xe5831034)

    mark("exit_done")
    insns.append(0xe28dd008)
    insns.append(0xe8bd47f0)
    insns.append(s3.encode_arm_branch(cave_va + len(insns) * 4, s3.HOOK2_AI_RESUME_VA))

    # Compact Luma-only 56x56 Downsampler (r0=Y, r1=dst, ip=fd_w, sb=fd_h)
    mark("sub_downsample")
    insns.append(0xe92d47f0)
    insns.append(0xe3a04000)
    mark("loop_ds_row")
    insns.append(0xe59d6014)
    insns.append(0xe0060694)
    insns.append(0xe0866186)
    insns.append(0xe1a064a6)
    insns.append(0xe0270c96)
    insns.append(0xe0818304)
    insns.append(0xe3a05000)
    mark("loop_ds_col")
    insns.append(0xe0060c95)
    insns.append(0xe0866186)
    insns.append(0xe1a064a6)
    insns.append(0xe7d79006)
    insns.append(0xe3590030)
    insns.append(0x31a09089)
    b("luma_lifted", cond=0x3)
    insns.append(0xe3590090)
    insns.append(0x33a0a048)
    insns.append(0x308a90a9)
    mark("luma_lifted")
    insns.append(0xe0886005)
    insns.append(0xe5c69000)
    insns.append(0xe5c69e00)
    insns.append(0xe2866c0e)
    insns.append(0xe5c69e00)
    insns.append(0xe2855001)
    insns.append(0xe3550038)
    b("loop_ds_col", cond=0xb)
    insns.append(0xe3a06000)
    insns.append(0xe5886038)
    insns.append(0xe588603c)
    insns.append(0xe2888c0e)
    insns.append(0xe5886038)
    insns.append(0xe588603c)
    insns.append(0xe2888c0e)
    insns.append(0xe5886038)
    insns.append(0xe588603c)
    insns.append(0xe2844001)
    insns.append(0xe3540038)
    b("loop_ds_row", cond=0xb)
    insns.append(0xe8bd87f0)

    # Heatmap scanner (r0 = 0x52b09a00)
    mark("sub_find_peak")
    insns.append(0xe92d4ff0)
    insns.append(0xe3a01000)
    insns.append(0xe3a0201c)
    insns.append(0xe3a0301c)
    insns.append(0xe3a0b000)
    insns.append(0xe2804b07)
    insns.append(0xe2805b0e)
    insns.append(0xe3a06000)
    mark("loop_peak_row")
    insns.append(0xe3a07000)
    mark("loop_peak_col")
    insns.append(0xe0878306)
    insns.append(0xe1a08088)
    insns.append(0xe19090f8)
    insns.append(0xe194a0f8)
    insns.append(0xe195c0f8)
    insns.append(0xe24cc002)
    insns.append(0xe15c000a)
    insns.append(0xc1a0a00c)
    insns.append(0xe04aa009)
    insns.append(0xe35a0002)
    insns.append(0xa28bb001)
    insns.append(0xe15a0001)
    b("skip_peak_update", cond=0xd)
    insns.append(0xe1a0100a)
    insns.append(0xe1a02007)
    insns.append(0xe1a03006)
    mark("skip_peak_update")
    insns.append(0xe2877001)
    insns.append(0xe3570038)
    b("loop_peak_col", cond=0xb)
    insns.append(0xe2866001)
    insns.append(0xe3560038)
    b("loop_peak_row", cond=0xb)
    insns.append(0xe1a00001)
    insns.append(0xe1a01002)
    insns.append(0xe1a02003)
    insns.append(0xe1a0300b)
    insns.append(0xe8bd8ff0)

    for idx, target, is_bl, cond in relocs:
        insns[idx] = s3.encode_arm_branch(cave_va + idx * 4, labels[target], is_bl=is_bl, cond=cond)

    return struct.pack(f"<{len(insns)}I", *insns)


def apply_step2c_patch(rtos_binary: bytearray, rtos_payload_offset: int = s3.RTOS_PAYLOAD_START) -> dict:
    orig_builder = s3.build_cave2_payload
    s3.build_cave2_payload = build_step2c_cave2_payload
    try:
        manifest = s3.apply_stage3_patch(rtos_binary, rtos_payload_offset=rtos_payload_offset)
    finally:
        s3.build_cave2_payload = orig_builder
    manifest["stage"] = "Step 2C: Crash-Proof Continuous AI Probe + Oscillating Heartbeat"
    manifest["dsp_io_buffers"]["heatmap_output_3ch_56x64_s16"] = hex(TENSOR_OUT_REAL_VA)
    return manifest


if __name__ == "__main__":
    c2 = build_step2c_cave2_payload()
    print(f"Step 2C Cave 2: {len(c2)} bytes @ {hex(s3.CAVE2_VA)} (margin to Desc: {s3.DESC_TABLE_VA - (s3.CAVE2_VA + len(c2))} B)")
