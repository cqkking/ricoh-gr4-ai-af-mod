#!/usr/bin/env python3
"""
step2b_live_probe_patcher.py - Step 2B: Non-Blocking Continuous LiveView AI Probe

Combines two verified breakthroughs:
1. Non-blocking DSP pipeline (WaitMessage(..., timeout=0) + OsMutex r4+0x48 unlocked during DSP/SD/AHIP).
2. True K-3 III 3-channel 56x56 segmentation heatmap offset: work_buf + 0x9a00 (0x52b09a00)
   (work_buf + 0x2a00 is the 14x14x64 intermediate backbone feature map for scene classification).

Telemetry & Tracking Behavior:
- Warmup (frames 0..19): box (16, 16)
- Load/Start fail:       box (16, 56) / (16, 96)
- AHIP frame NULL:       box (16, 56)
- Send fail:             box (16, 136)
- IN_FLIGHT (>90 frames stall): box (16, 136)
- Blank scene (below threshold): heartbeat box (240, 140, 32x32)
- Subject detected (max_diff >= 3, area_cnt >= 2 @ 0x52b09a00):
  Tracks subject with Q8 box clamped to 32..72 px (0x2000..0x4800), 0 ms shutter freeze.
"""

import struct
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM
import stage3_ai_patcher as s3

TENSOR_OUT_REAL_VA = s3.TENSOR_IN_VA + 0x9a00  # 0x52b09a00 (true 3ch 56x56 s16 heatmap)
RESP_BUF_VA        = s3.TENSOR_MSG_VA + 0x40

# STATUS_VA (0x52c00100) layout:
# +0x00: state (0=init/load, 2=idle_ready, 3=in_flight)
# +0x04: boot_ticks
# +0x08: seq (u16)
# +0x0c: poll_frames
# +0x10: cached_left_q8
# +0x14: cached_top_q8
# +0x18: cached_box_q8
# +0x1c: cached_score


def build_step2b_cave2_payload(cave_va: int = s3.CAVE2_VA) -> bytes:
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

    # Save incoming r4..r8, sb, sl, lr + 8B scratch
    insns.append(0xe92d47f0)  # push {r4, r5, r6, r7, r8, sb, sl, lr}
    insns.append(0xe24dd008)  # sub  sp, sp, #8

    # One-shot boot zeroing of STATUS_VA + initial cached box (16, 16, 32x32)
    ldr32(8, s3.STATUS_VA)
    insns.append(0xe5d4213e)  # ldrb r2, [r4, #0x13e]
    insns.append(0xe3520000)
    b("status_inited", cond=0x1)
    insns.append(0xe3a00000)
    insns.append(0xe3a01000)
    mark("zero_st_loop")
    insns.append(0xe7880001)
    insns.append(0xe2811004)
    insns.append(0xe3510020)
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
    unlock_fd_mutex()

    # Branch on state
    insns.append(0xe5987000)  # ldr r7, [r8, #0]
    insns.append(0xe3570003)  # cmp r7, #3 (IN_FLIGHT)
    b("state_in_flight", cond=0x0)
    insns.append(0xe3570002)  # cmp r7, #2 (IDLE_READY)
    b("state_idle_send", cond=0x0)

    # --- State 0: Warmup + LoadDsp + Start ---
    insns.append(0xe5982004)
    insns.append(0xe2822001)
    insns.append(0xe5882004)
    insns.append(0xe2523014)  # subs r3, r2, #20
    b("use_cached_box", cond=0x3)  # blo (first 20 frames -> (16, 16))
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
    insns.append(0xe3a01043)  # 'C'
    insns.append(0xe3a02000)
    insns.append(0xe3a03000)
    call(s3.LOAD_DSP_VA)
    insns.append(0xe3500000)
    b("load_ok", cond=0x1)

    insns.append(0xe1a00005)
    insns.append(0xe3a01049)  # 'I'
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

    insns.append(0xe3a00002)  # state = 2 (IDLE_READY)
    insns.append(0xe5880000)
    b("use_cached_box")

    # --- State 3: Non-blocking WaitMessage(timeout=0) ---
    mark("state_in_flight")
    call(s3.GET_DSP_SVC_VA)
    insns.append(0xe5905004)
    insns.append(0xe2886008)  # &seq
    insns.append(0xe1a00005)
    insns.append(0xe1d610b0)  # seq
    insns.append(0xe3a02000)  # timeout = 0 (TMO_POL)
    ldr32(3, RESP_BUF_VA)
    call(s3.WAIT_MSG_VA)
    insns.append(0xe3500000)
    b("inference_done", cond=0x0)

    # Still pending: increment poll_frames, keep cached box (flag stall if > 90 frames)
    insns.append(0xe598200c)
    insns.append(0xe2822001)
    insns.append(0xe588200c)
    insns.append(0xe352005a)  # cmp r2, #90
    b("diag_stall_err", cond=0xc)  # bgt
    b("use_cached_box")

    mark("inference_done")
    # Invalidate ARM D-Cache for true output heatmap @ 0x52b09a00 (size 0x5400)
    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, TENSOR_OUT_REAL_VA)
    insns.append(0xe3a02b15)  # mov r2, #0x5400
    call(s3.INVAL_DCACHE_VA)

    ldr32(0, TENSOR_OUT_REAL_VA)
    b("sub_find_peak", bl=True)  # returns r0=max_diff, r1=col, r2=row, r3=area_cnt

    insns.append(0xe3500003)  # cmp r0, #3
    b("set_heartbeat", cond=0xb)
    insns.append(0xe3530002)  # cmp r3, #2
    b("set_heartbeat", cond=0xb)

    # Subject detected! box_size = clamp(0x2000 + (area_cnt << 5), 0x2000, 0x4800) (32..72 px Q8)
    insns.append(s3.encode_movw(5, 0x2000))
    insns.append(0xe0855283)  # add r5, r5, r3, lsl #5
    insns.append(s3.encode_movw(6, 0x4800))
    insns.append(0xe1550006)  # cmp r5, #0x4800
    insns.append(0xc1a05006)  # movgt r5, r6

    insns.append(0xe280901e)  # score = clamp(max_diff + 30, 30, 100)
    insns.append(0xe3590064)
    insns.append(0xc3a09064)

    # cx = (col * fd_w * 73) >> 4, cy = (row * fd_h * 73) >> 4
    insns.append(0xe1d43fbc)
    insns.append(0xe3530000)
    insns.append(s3.encode_movw(3, 320, cond=0x0))
    insns.append(0xe0070391)
    insns.append(0xe3a03049)
    insns.append(0xe0070793)
    insns.append(0xe1a07227)  # cx

    insns.append(0xe1d43fbe)
    insns.append(0xe3530000)
    insns.append(s3.encode_movw(3, 212, cond=0x0))
    insns.append(0xe00c0392)
    insns.append(0xe3a03049)
    insns.append(0xe00c0c93)
    insns.append(0xe1a0c22c)  # cy

    insns.append(0xe05770a5)  # subs r7, r7, r5, lsr #1 (left)
    insns.append(0x43a07000)  # movmi r7, #0
    insns.append(0xe05c60a5)  # subs r6, ip, r5, lsr #1 (top)
    insns.append(0x43a06000)  # movmi r6, #0
    b("store_cached")

    mark("set_heartbeat")
    ldr32(7, 0x0f000)  # x = 240 px Q8
    ldr32(6, 0x08c00)  # y = 140 px Q8
    insns.append(s3.encode_movw(5, 0x2000))  # 32x32 Q8
    insns.append(0xe3a09063)  # score = 99

    mark("store_cached")
    insns.append(0xe5887010)  # cached_left_q8
    insns.append(0xe5886014)  # cached_top_q8
    insns.append(0xe5885018)  # cached_box_q8
    insns.append(0xe588901c)  # cached_score
    insns.append(0xe3a00002)  # state = 2 (IDLE_READY -> immediately send next frame!)
    insns.append(0xe5880000)

    # --- State 2: Extract live AHIP YUV420 frame -> 56x56 RGB -> SendMessage ---
    mark("state_idle_send")
    ldr32(3, s3.AHIP_DRV_BASE_VA)
    insns.append(0xe593900c)  # sb = slot byte size
    insns.append(0xe5932424)
    insns.append(0xe3520000)
    insns.append(0x15922000)
    insns.append(0xe3520000)
    b("try_ahip_434", cond=0x0)
    insns.append(0xe5926004)
    insns.append(0xe3560000)
    b("try_ahip_434", cond=0x0)
    insns.append(0xe5927008)
    insns.append(0xe592a00c)  # sl = V_plane
    b("have_yuv")

    mark("try_ahip_434")
    insns.append(0xe5932434)
    insns.append(0xe3520000)
    insns.append(0x15922000)
    insns.append(0xe3520000)
    b("try_ahip_legacy", cond=0x0)
    insns.append(0xe5926004)
    insns.append(0xe3560000)
    b("try_ahip_legacy", cond=0x0)
    insns.append(0xe5927008)
    insns.append(0xe592a00c)
    b("have_yuv")

    mark("try_ahip_legacy")
    insns.append(0xe593201c)
    insns.append(0xe3520000)
    insns.append(0x05932014)
    insns.append(0xe3520000)
    b("diag_load_err", cond=0x0)  # AHIP NULL -> show (16, 56)
    insns.append(0xe592600c)
    insns.append(0xe3560000)
    b("diag_load_err", cond=0x0)
    insns.append(0xe5927010)
    insns.append(0xe592a014)

    mark("have_yuv")
    # Invalidate YUV slot D-Cache
    insns.append(0xe3590000)
    insns.append(0x03a09802)
    call(s3.GET_CACHE_MGR_VA)
    insns.append(0xe1a01006)
    insns.append(0xe1a02009)
    call(s3.INVAL_DCACHE_VA)

    # Downsample fd_w (ip) x fd_h (sb) -> 56x56 RGB @ 0x52b00000
    insns.append(0xe1d4cfbc)
    insns.append(0xe35c0000)
    insns.append(s3.encode_movw(12, 320, cond=0x0))
    insns.append(0xe1d49fbe)
    insns.append(0xe3590000)
    insns.append(s3.encode_movw(9, 212, cond=0x0))
    insns.append(0xe1a00006)  # Y
    ldr32(1, s3.TENSOR_IN_VA)
    insns.append(0xe1a02007)  # U
    insns.append(0xe1a0300a)  # V
    b("sub_downsample", bl=True)

    # Flush work_buf (0x52b00000, size 0xee00)
    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, s3.TENSOR_IN_VA)
    insns.append(0xe3a02cee)
    call(s3.FLUSH_DCACHE_VA)

    # Populate TENSOR_MSG_VA (0x52c00000) & SendMessage(..., timeout=100)
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
    insns.append(0xe2886008)  # &seq
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
    insns.append(0xe588000c)  # poll_frames = 0
    insns.append(0xe3a00003)  # state = 3 (IN_FLIGHT)
    insns.append(0xe5880000)
    b("use_cached_box")

    # --- Diagnostic override positions ---
    mark("diag_load_err")
    insns.append(0xe3a07b04)  # x = 16 px Q8 (0x1000)
    insns.append(s3.encode_movw(6, 0x3800))  # y = 56 px Q8
    b("render_diag")

    mark("diag_start_err")
    insns.append(0xe3a07b04)
    insns.append(s3.encode_movw(6, 0x6000))  # y = 96 px Q8
    b("render_diag")

    mark("diag_send_err")
    mark("diag_stall_err")
    insns.append(0xe3a07b04)
    insns.append(s3.encode_movw(6, 0x8800))  # y = 136 px Q8

    mark("render_diag")
    insns.append(s3.encode_movw(5, 0x2000))
    insns.append(0xe3a09058)  # score = 88
    b("render_box")

    mark("use_cached_box")
    insns.append(0xe5987010)  # left Q8
    insns.append(0xe5986014)  # top Q8
    insns.append(0xe5985018)  # box Q8
    insns.append(0xe598901c)  # score

    # --- Re-lock mutex r4+0x48 and populate Step-1-verified m_DetectedFaceResult ---
    mark("render_box")
    # Save box params across lock_fd_mutex (r5, r7, r9 are callee-saved; save r6 in [sp, #4]!)
    insns.append(0xe58d6004)  # str r6, [sp, #4] (save top_q8 across lock_fd_mutex)
    lock_fd_mutex()
    insns.append(0xe59d6004)  # ldr r6, [sp, #4] (restore top_q8)

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

    insns.append(0xe58a7000)  # left
    insns.append(0xe58a6004)  # top
    insns.append(0xe58a5008)  # w
    insns.append(0xe58a500c)  # h
    insns.append(0xe58a904c)  # score
    insns.append(0xe5941118)
    insns.append(0xe3510000)
    insns.append(0x03a01001)
    insns.append(0xe58a1050)  # face_id

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

    # --- Reuse exact Downsampler & Heatmap Peak Scanner subroutines ---
    # Extract subroutines cleanly from stage3_ai_patcher by assembling only the helper tail,
    # OR inline compactly:
    # Sub 1: sub_downsample (r0=Y, r1=dst, r2=U, r3=V, ip=fd_w, sb=fd_h)
    mark("sub_downsample")
    insns.append(0xe92d47f0)  # push {r4, r5, r6, r7, r8, sb, sl, lr}
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
    # 2x backlit recovery
    insns.append(0xe3590030)
    insns.append(0x31a09089)
    b("luma_lifted", cond=0x3)
    insns.append(0xe3590090)
    insns.append(0x33a0a048)
    insns.append(0x308a90a9)
    mark("luma_lifted")
    insns.append(0xe1a0a009)
    insns.append(0xe1a0e009)
    insns.append(0xe3520000)
    b("store_rgb", cond=0x0)
    insns.append(0xe3530000)
    b("store_rgb", cond=0x0)
    insns.append(0xe59da014)
    insns.append(0xe00a0a94)
    insns.append(0xe08aa18a)
    insns.append(0xe1a0a52a)
    insns.append(0xe1a0e0ac)
    insns.append(0xe00a0a9e)
    insns.append(0xe08aa0a6)
    insns.append(0xe7d2e00a)
    insns.append(0xe7d3a00a)
    insns.append(0xe24ee080)
    insns.append(0xe24aa080)
    insns.append(0xe049614e)
    insns.append(0xe04660ca)
    insns.append(0xe6e86016)
    insns.append(0xe08ee0ce)
    insns.append(0xe089e00e)
    insns.append(0xe6e8e01e)
    insns.append(0xe08aa0ca)
    insns.append(0xe089900a)
    insns.append(0xe6e89019)
    insns.append(0xe1a0a006)
    mark("store_rgb")
    insns.append(0xe0886005)
    insns.append(0xe5c69000)
    insns.append(0xe5c6ae00)
    insns.append(0xe2866c0e)
    insns.append(0xe5c6ee00)
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

    # Sub 2: sub_find_peak (r0 = 0x52b09a00) -> r0=max_diff, r1=col, r2=row, r3=area_cnt
    mark("sub_find_peak")
    insns.append(0xe92d4ff0)
    insns.append(0xe3a01000)
    insns.append(0xe3a0201c)
    insns.append(0xe3a0301c)
    insns.append(0xe3a0b000)
    insns.append(0xe2804b07)  # +0x1c00 (Ch1)
    insns.append(0xe2805b0e)  # +0x3800 (Ch2)
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


def apply_step2b_patch(rtos_binary: bytearray, rtos_payload_offset: int = s3.RTOS_PAYLOAD_START) -> dict:
    orig_builder = s3.build_cave2_payload
    s3.build_cave2_payload = build_step2b_cave2_payload
    try:
        manifest = s3.apply_stage3_patch(rtos_binary, rtos_payload_offset=rtos_payload_offset)
    finally:
        s3.build_cave2_payload = orig_builder
    manifest["stage"] = "Step 2B: Non-Blocking LiveView AI Probe (0x52b09a00 Heatmap + 0ms Shutter)"
    manifest["dsp_io_buffers"]["heatmap_output_3ch_56x64_s16"] = hex(TENSOR_OUT_REAL_VA)
    return manifest


if __name__ == "__main__":
    c2 = build_step2b_cave2_payload()
    print(f"Step 2B Cave 2: {len(c2)} bytes @ {hex(s3.CAVE2_VA)} (margin to Desc: {s3.DESC_TABLE_VA - (s3.CAVE2_VA + len(c2))} B)")
