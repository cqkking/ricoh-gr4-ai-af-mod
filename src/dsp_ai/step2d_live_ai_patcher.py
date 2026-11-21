#!/usr/bin/env python3
"""Step 2D v1.2.19: K-3 III SPResize 1:1 aspect letterboxing, live posture +0x7da & probability-first AI AF (Option B).

Builds on the real-hardware-verified v1.2.15 OSD-commit-first lifecycle and v1.2.17/v1.2.18
AHIP IMAGE2 live-frame pipeline, fixing the 3 root causes uncovered by disassembling K-3 III
v2.20 and GR IV v1.20 RTOS:
1. Lock-free LiveView gravity posture read via GR IV CameraPostureService (0x5537B298 /
   0x5537B2A0 -> [+0x518] -> [0] + 0x7da, matching 0x5377d960 / 0x538b40c8 instead of the
   still-capture Exif MakerNote offset +0xd8), rotating the 56x56 planar RGB input upright
   using K-3 III's exact (start, c_step, r_step) tables (0x539323d4/2278/2290);
2. K-3 III Retouch::SPResize (0x530acfa0..0x530ad528) exact 1:1 square-pixel letterboxing
   for 3:2 frames: active sensor rows s_r = 9..46 map to 0..fd_h (centered 56x38 in landscape,
   38x56 in portrait), while out-of-bounds letterbox rows s_r = 0..8 and 47..55 are padded
   with neutral black (R=0, G=0, B=0) and inverse-mapped identically in sub_update_tracking_box;
3. Exact 3-class softmax via 16-byte exp(-0.25*d)*256 LUT @ EXP_LUT_VA (0x53001FF0) with
   K-3 III probability-first ranking (other = ea + eb < primary, log_odds = da + db > tie-break,
   distance < tie-break):
   - Strict Tier 1 (Ch1 Head/Face/Cat Face): P(Ch1) >= 60% (other <= 170), peak >= 65%
     (other <= 135), >= 3 cells in +-6 local cluster; strictly blocks Tier 2 when hit;
   - Strict Tier 2 (Ch2 Body/Back/Profile fallback): P(Ch2) >= 77% (other <= 75),
     >= 8 cells in +-8 local cluster;
4. 2-iteration local mean-shift probability-weighted sub-pixel centroid in Q4, tight local-span
   box sizing (head 28..52 px, body 34..56 px), 2-cycle cold confirmation, position/size
   deadbands (2.5 px / 3.0 px), smooth EMA, 3-cycle hold + Option B hide.
"""

import struct
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

import stage3_ai_patcher as s3
import step2c_osd_recovery as osd

LIVE_AI_COOKIE = 0x41493244  # "AI2D"

WARMUP_FRAMES = 20
LOAD_RETRY_MASK = 0x3F       # Retry LoadDsp every 64 frames on cold storage delay
MAX_POLL_FRAMES = 30
LOST_HOLD_CYCLES = 3         # Hold box for 3 empty cycles, hide on 4th (Option B)

# GR IV CameraPostureService globals (verified from 0x53750008 / 0x5377d960 / 0x53782114 / 0x538b40c8)
ORIENT_GUARD_VA = 0x5537B298
ORIENT_SVC_PTR_VA = 0x5537B2A0  # ORIENT_GUARD_VA + 8
ORIENT_LIVE_OFF = 0x7DA         # Live gravity posture byte ([buf + 0x7da]: 0=0deg, 1=90CW, 2=180, 3=90CCW)

# K-3 III Retouch::SPResize (0x530acfa0..0x530ad528) 1:1 aspect-ratio letterbox bounds for 3:2 frames
LETTERBOX_ROW_MIN = 9        # Active 3:2 sensor rows inside 56x56 grid: s_r = 9..46 (38 rows)
LETTERBOX_ROW_MAX = 46       # Rows 0..8 and 47..55 are padded with black (R=0, G=0, B=0)

# K-3 III exact softmax sensitivity, cluster, sizing, and anti-jitter constants
CH1_OTHER_MAX = 170          # e0 + e2 <= 170 <=> P(Ch1) >= 60%
CH1_PEAK_OTHER_MAX = 135     # e0 + e2 <= 135 <=> peak P(Ch1) >= 65%
CH1_MIN_LOCAL = 3            # Minimum local +-6 cluster cell count for Ch1 head/face
CH2_OTHER_MAX = 75           # e0 + e1 <= 75  <=> P(Ch2) >= 77%
CH2_MIN_LOCAL = 8            # Minimum local +-8 cluster cell count for Ch2 body fallback
LOCAL_RADIUS_CH1 = 6
LOCAL_RADIUS_CH2 = 8
ACQ_CONFIRM_CYCLES = 2       # Cold start has_box 0 -> 1 confirmation cycles

MIN_HEAD_BOX_Q8 = 0x1C00     # 28 px
MAX_HEAD_BOX_Q8 = 0x3400     # 52 px
MIN_BODY_BOX_Q8 = 0x2200     # 34 px
MAX_BODY_BOX_Q8 = 0x3800     # 56 px
POS_DEADBAND_Q8 = 0x280      # 2.5 px position deadband
SIZE_DEADBAND_Q8 = 0x300     # 3.0 px size deadband

STATUS_SIZE = 0x50           # 80 bytes: STATUS_VA + 0x00 .. + 0x4c
RESP_BUF_VA = s3.TENSOR_MSG_VA + 0x40
HEATMAP_OUT_VA = s3.TENSOR_IN_VA + 0x9A00   # 0x52b09a00 (true K-3 III 3-ch 56x56 heatmap)
HEATMAP_LEN = 0x5400                        # 3 * 0x1c00 bytes
WORK_BUF_FLUSH_LEN = 0xEE00                 # 0x52b00000..0x52b0edff

STOP_DSP_VA = 0x5398ED08     # DspService::Stop(svc, timeout=100)
OS_EF_WAIT_VA = 0x539030A0   # OsEventFlag::Wait(ef, waiptn, &flgptn, timeout=0)
OS_EF_CLEAR_VA = 0x53902FE8  # OsEventFlag::Clear(ef, clrptn)

HOOK_UPDATE_VA = 0x538963B8
EXPECTED_UPDATE = bytes.fromhex("8dfdffeb")

CAVE3_VA = 0x530018E0        # Code cave immediately after DESC_TABLE_VA (ends 0x530018d8)
EXP_LUT_VA = 0x53001FF0      # 16-byte softmax LUT placed cleanly after Cave 3 code
CAVE3_CODE_LIMIT_VA = EXP_LUT_VA
CAVE3_LIMIT_VA = 0x53002000  # End of verified contiguous ROM zero cave
EXP_LUT_BYTES = bytes([255, 199, 155, 121, 94, 73, 57, 44, 35, 27, 21, 16, 13, 10, 8, 6])


def _make_asm(base_va: int):
    """Tiny deterministic A32 instruction emitter with label/branch resolution."""
    insns: list[int] = []
    labels: dict[str, int] = {}
    relocs: list[tuple[int, str, bool, int]] = []

    def mark(name: str) -> None:
        if name in labels:
            raise ValueError(f"Duplicate label: {name}")
        labels[name] = base_va + len(insns) * 4

    def b(target: str, bl: bool = False, cond: int = 0xE) -> None:
        relocs.append((len(insns), target, bl, cond))
        insns.append(0)

    def call(va: int) -> None:
        insns.append(s3.encode_arm_branch(base_va + len(insns) * 4, va, is_bl=True))

    def jump(va: int, cond: int = 0xE) -> None:
        insns.append(s3.encode_arm_branch(base_va + len(insns) * 4, va, is_bl=False, cond=cond))

    def ldr32(rd: int, val: int) -> None:
        s3.emit_ldr_imm32(insns, rd, val)

    def finalize() -> tuple[bytes, dict[str, int]]:
        for idx, target, is_bl, cond in relocs:
            insns[idx] = s3.encode_arm_branch(base_va + idx * 4, labels[target], is_bl=is_bl, cond=cond)
        return struct.pack(f"<{len(insns)}I", *insns), labels

    return insns, mark, b, call, jump, ldr32, finalize


def build_live_ai_cave1() -> bytes:
    """Cave 1 @ 0x53000d60: zero full 80-byte STATUS_VA region on first boot & publish cookie."""
    insns, mark, b, _, jump, ldr32, finalize = _make_asm(s3.CAVE1_VA)

    insns.extend((
        0xE1A0C00D,  # mov  ip, sp
        0xE92DD8F0,  # push {r4, r5, r6, r7, fp, ip, lr, pc}
        0xE24CB004,  # sub  fp, ip, #4
        0xE24DD008,  # sub  sp, sp, #8
    ))
    ldr32(4, s3.HOOK1_SINGLETON_VA)
    insns.extend((
        0xE5946000,  # ldr  r6, [r4]
        0xE3560000,  # cmp  r6, #0
    ))
    b("exit", cond=0x1)

    ldr32(3, s3.STATUS_VA)
    insns.extend((
        0xE3A02000,                # mov  r2, #0
        0xE3A01000 | STATUS_SIZE,  # mov  r1, #0x50 (80 bytes = STATUS_VA+0x00..0x4c)
    ))
    mark("zero_status_loop")
    insns.extend((
        0xE2511004,  # subs r1, r1, #4
        0xE7832001,  # str  r2, [r3, r1]
    ))
    b("zero_status_loop", cond=0x1)

    ldr32(2, LIVE_AI_COOKIE)
    insns.append(0xE5832018)  # str r2, [r3, #0x18]

    mark("exit")
    jump(s3.HOOK1_RESUME_VA)

    code, _ = finalize()
    if s3.CAVE1_VA + len(code) > s3.CAVE2_VA:
        raise ValueError(f"Cave 1 ({len(code)} B) overlaps Cave 2")
    return code


def _build_cave3_body(cave2_syms: dict[str, int]) -> tuple[bytes, dict[str, int]]:
    """Cave 3 @ 0x530018e0..0x53001ff0: AHIP rotated RGB downsampler + K-3 III softmax scanner."""
    insns, mark, b, call, _, ldr32, finalize = _make_asm(CAVE3_VA)

    # ---------------------------------------------------------------------
    # 1. sub_sample_ahip_rgb(r0 = fd_obj) -> r0 (1=sampled, 0=skip)
    # ---------------------------------------------------------------------
    mark("sub_sample_ahip_rgb")
    # 9 regs (36 B) + 28 B locals = 64 B (8-byte aligned)
    insns.extend((
        0xE92D4FF0,  # push {r4, r5, r6, r7, r8, sb, sl, fp, lr}
        0xE24DD01C,  # sub  sp, sp, #28
        0xE3500000,  # cmp  r0, #0
    ))
    b("sample_fail", cond=0x0)

    # Validate fd_w ([r0,#0xfc]) and fd_h ([r0,#0xfe]): 64..1280, w%8==0, h%4==0
    insns.extend((
        0xE1D04FBC,  # ldrh r4, [r0, #0xfc]
        0xE1D05FBE,  # ldrh r5, [r0, #0xfe]
        0xE3540040,  # cmp  r4, #64
    ))
    b("sample_fail", cond=0x3)
    insns.append(0xE3540C05)    # cmp  r4, #0x500 (1280)
    b("sample_fail", cond=0x8)
    insns.append(0xE3140007)    # tst  r4, #7
    b("sample_fail", cond=0x1)
    insns.append(0xE3550040)    # cmp  r5, #64
    b("sample_fail", cond=0x3)
    insns.append(0xE3550C05)    # cmp  r5, #0x500 (1280)
    b("sample_fail", cond=0x8)
    insns.append(0xE3150003)    # tst  r5, #3
    b("sample_fail", cond=0x1)

    ldr32(6, s3.AHIP_DRV_BASE_VA)
    insns.extend((
        0xE5961030,  # ldr  r1, [r6, #0x30]
        0xE3510002,  # cmp  r1, #2
    ))
    b("sample_fail", cond=0x1)

    # 5x AHIP DDR-range guards (0x40000000 <= ptr < 0xe0000000)
    insns.extend((
        0xE5961424,  # ldr  r1, [r6, #0x424] (slot_pp)
        0xE2412101,  # sub  r2, r1, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
    ))
    b("sample_fail", cond=0x2)

    insns.extend((
        0xE5911000,  # ldr  r1, [r1] (slot)
        0xE2412101,  # sub  r2, r1, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
    ))
    b("sample_fail", cond=0x2)

    insns.extend((
        0xE5917004,  # ldr  r7, [r1, #0x04] (Y_ptr)
        0xE2472101,  # sub  r2, r7, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
    ))
    b("sample_fail", cond=0x2)

    insns.extend((
        0xE5918008,  # ldr  r8, [r1, #0x08] (U_ptr)
        0xE2482101,  # sub  r2, r8, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
        0x23A08000,  # movhs r8, #0
        0xE591900C,  # ldr  sb, [r1, #0x0c] (V_ptr)
        0xE2492101,  # sub  r2, sb, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
        0x23A09000,  # movhs sb, #0
        0xE58D8008,  # str  r8, [sp, #8]
        0xE58D900C,  # str  sb, [sp, #12]
    ))

    # Invalidate ARM D-Cache for live YUV420 slot before CPU reads
    insns.extend((
        0xE0030594,  # mul  r3, r4, r5
        0xE596100C,  # ldr  r1, [r6, #0x0c]
        0xE1510003,  # cmp  r1, r3
    ))
    b("use_default_sz", cond=0x3)
    insns.append(0xE3510502)       # cmp  r1, #0x00800000
    b("have_total_sz", cond=0x9)
    mark("use_default_sz")
    insns.append(0xE08310A3)       # add  r1, r3, r3, lsr #1
    mark("have_total_sz")
    insns.append(0xE1A0A001)       # mov  sl, r1
    call(s3.GET_CACHE_MGR_VA)
    insns.extend((
        0xE1A01007,  # mov  r1, r7
        0xE1A0200A,  # mov  r2, sl
    ))
    call(s3.INVAL_DCACHE_VA)

    # Lock-free live gravity posture read with 3 DDR range guards -> K-3 III upright rotation
    ldr32(8, s3.STATUS_VA)
    insns.extend((
        0xE5980040,  # ldr   r0, [r8, #0x40] (last_orient)
        0xE3500003,  # cmp   r0, #3
        0x83A00000,  # movhi r0, #0
    ))
    ldr32(1, ORIENT_GUARD_VA)      # ORIENT_SVC_PTR_VA == ORIENT_GUARD_VA + 8
    insns.extend((
        0xE5913000,  # ldr  r3, [r1]
        0xE3130001,  # tst  r3, #1
    ))
    b("orient_keep", cond=0x0)
    insns.extend((
        0xE5911008,  # ldr  r1, [r1, #8] (svc @ 0x5537B2A0)
        0xE2412101,  # sub  r2, r1, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
    ))
    b("orient_keep", cond=0x2)
    insns.extend((
        0xE5911518,  # ldr  r1, [r1, #0x518] (obj)
        0xE2412101,  # sub  r2, r1, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
    ))
    b("orient_keep", cond=0x2)
    insns.extend((
        0xE5911000,  # ldr  r1, [r1] (buf)
        0xE2412101,  # sub  r2, r1, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
    ))
    b("orient_keep", cond=0x2)
    insns.extend((
        0xE5D137DA,  # ldrb r3, [r1, #0x7da] (live gravity posture @ +0x7da)
        0xE3530003,  # cmp  r3, #3 (preserve last_orient on TU/TD/--)
        0x91A00003,  # movls r0, r3
        0x95880040,  # strls r0, [r8, #0x40]
    ))
    mark("orient_keep")
    insns.extend((
        0xE5880044,  # str  r0, [r8, #0x44] (in_flight_orient)
        0xE3A06000,  # mov  r6, #0   (orient 0: start=0, c=+1, r=+64)
        0xE3A01001,  # mov  r1, #1
        0xE3A02040,  # mov  r2, #64
        0xE3500001,  # cmp  r0, #1
        s3.encode_movw(6, 3520, cond=0x0),
        0x03E0103F,  # mvneq r1, #63         (-64)
        0x03A02001,  # moveq r2, #1
        0xE3500002,  # cmp  r0, #2
        s3.encode_movw(6, 3575, cond=0x0),
        0x03E01000,  # mvneq r1, #0          (-1)
        0x03E0203F,  # mvneq r2, #63         (-64)
        0xE3500003,  # cmp  r0, #3
        0x03A06037,  # moveq r6, #55
        0x03A01040,  # moveq r1, #64
        0x03E02000,  # mvneq r2, #0          (-1)
        0xE58D1014,  # str  r1, [sp, #20] (c_step)
        0xE58D2018,  # str  r2, [sp, #24] (r_step)
    ))

    # K-3 III SPResize 1:1 aspect-ratio letterboxed YUV420 -> rotated upright 56x56 (stride 64) planar RGB @ TENSOR_IN_VA
    # Zero entire 0x7200 work buffer (TENSOR_IN_VA + HEATMAP_OUT_VA) first:
    # clears letterbox sensor rows 0..8 & 47..55 (R=G=B=0), stride-64 pad cols 56..63, and output heatmap.
    ldr32(9, s3.TENSOR_IN_VA)
    insns.extend((
        0xE3A00000,  # mov  r0, #0
        s3.encode_movw(3, WORK_BUF_FLUSH_LEN),
    ))
    mark("ds_zero_work")
    insns.extend((
        0xE2533004,  # subs r3, r3, #4
        0xE7890003,  # str  r0, [sb, r3]
    ))
    b("ds_zero_work", cond=0x1)

    insns.extend((
        0xE0899006,  # add  sb, sb, r6          (TENSOR_IN_VA + orient start offset)
        0xE0823182,  # add  r3, r2, r2, lsl #3  (9 * r_step: advance past top 9 letterbox rows)
        0xE0899003,  # add  sb, sb, r3
        0xE3A06000,  # mov  r6, #0              (active_r = 0..37 for sensor rows 9..46)
    ))

    mark("ds_row_loop")
    insns.extend((
        0xE0860006,  # add  r0, r6, r6          (2 * active_r)
        0xE2800001,  # add  r0, r0, #1          (2 * active_r + 1)
        0xE0000095,  # mul  r0, r5, r0          (fd_h * (2 * active_r + 1))
        0xE3A0106C,  # mov  r1, #108            (108 / 8192 = 1 / 75.852 -> 38 active rows)
        0xE0000091,  # mul  r0, r1, r0
        0xE1A006A0,  # lsr  r0, r0, #13
        0xE1500005,  # cmp  r0, r5
        0x22450001,  # subhs r0, r5, #1
        0xE0287490,  # mla  r8, r0, r4, r7
        0xE1A010A0,  # lsr  r1, r0, #1
        0xE1A020A4,  # lsr  r2, r4, #1
        0xE0010192,  # mul  r1, r2, r1
        0xE58D1010,  # str  r1, [sp, #16]
        0xE1A0B009,  # mov  fp, sb              (dst_px_ptr)
        0xE3A0A000,  # mov  sl, #0              (col = 0)
    ))

    mark("ds_col_loop")
    insns.extend((
        0xE08A000A,  # add  r0, sl, sl
        0xE2800001,  # add  r0, r0, #1
        0xE0000094,  # mul  r0, r4, r0
        0xE3A01049,  # mov  r1, #73
        0xE0000091,  # mul  r0, r1, r0
        0xE1A006A0,  # lsr  r0, r0, #13
        0xE7D81000,  # ldrb r1, [r8, r0]
        0xE59D2008,  # ldr  r2, [sp, #8]
        0xE59D300C,  # ldr  r3, [sp, #12]
        0xE3520000,  # cmp  r2, #0
    ))
    b("ds_gray", cond=0x0)
    insns.append(0xE3530000)
    b("ds_gray", cond=0x0)

    insns.extend((
        0xE59DC010,  # ldr  ip, [sp, #16]
        0xE08CC0A0,  # add  ip, ip, r0, lsr #1
        0xE7D2200C,  # ldrb r2, [r2, ip]
        0xE7D3300C,  # ldrb r3, [r3, ip]
        0xE2422080,  # sub  r2, r2, #128
        0xE2433080,  # sub  r3, r3, #128
        0xE08300C3,  # add  r0, r3, r3, asr #1
        0xE04001C3,  # sub  r0, r0, r3, asr #3
        0xE0810000,  # add  r0, r1, r0
        0xE6E80010,  # usat r0, #8, r0
        0xE082C082,  # add  ip, r2, r2, lsl #1
        0xE08CC182,  # add  ip, ip, r2, lsl #3
        0xE063E183,  # rsb  lr, r3, r3, lsl #3
        0xE08EE203,  # add  lr, lr, r3, lsl #4
        0xE08CC00E,  # add  ip, ip, lr
        0xE041C2CC,  # sub  ip, r1, ip, asr #5
        0xE6E8C01C,  # usat ip, #8, ip
        0xE082E0C2,  # add  lr, r2, r2, asr #1
        0xE08E2142,  # add  r2, lr, r2, asr #2
        0xE0812002,  # add  r2, r1, r2
        0xE6E82012,  # usat r2, #8, r2
    ))
    b("ds_store_rgb")

    mark("ds_gray")
    insns.extend((
        0xE1A00001,
        0xE1A0C001,
        0xE1A02001,
    ))

    mark("ds_store_rgb")
    insns.extend((
        0xE5CB0000,  # strb r0, [fp, #0]
        0xE5CBCE00,  # strb ip, [fp, #0xe00]
        0xE28B3C0E,  # add  r3, fp, #0xe00
        0xE5C32E00,  # strb r2, [r3, #0xe00]
        0xE59D0014,  # ldr  r0, [sp, #20]
        0xE08BB000,  # add  fp, fp, r0
        0xE28AA001,  # add  sl, sl, #1
        0xE35A0038,  # cmp  sl, #56
    ))
    b("ds_col_loop", cond=0xB)

    insns.extend((
        0xE59D0018,  # ldr  r0, [sp, #24]
        0xE0899000,  # add  sb, sb, r0
        0xE2866001,  # add  r6, r6, #1
        0xE3560000 | (LETTERBOX_ROW_MAX - LETTERBOX_ROW_MIN + 1),  # cmp r6, #38
    ))
    b("ds_row_loop", cond=0xB)

    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, s3.TENSOR_IN_VA)
    insns.append(s3.encode_movw(2, WORK_BUF_FLUSH_LEN))
    call(s3.FLUSH_DCACHE_VA)

    insns.extend((
        0xE3A00001,
        0xE28DD01C,
        0xE8BD8FF0,
    ))

    mark("sample_fail")
    insns.extend((
        0xE3A00000,
        0xE28DD01C,
        0xE8BD8FF0,
    ))

    # ---------------------------------------------------------------------
    # 2. sub_scan_heatmap(r0 = HEATMAP_OUT_VA)
    #    -> r0=(tier<<16)|prob_pct (or 0), r1=u_c_q4, r2=u_r_q4,
    #       r3=(local_cnt<<16)|(span_r<<8)|span_c
    # ---------------------------------------------------------------------
    mark("sub_scan_heatmap")
    # 9 regs (36 B) + 68 B locals = 104 B (8-byte aligned)
    # Pass 1 locals:
    #   [sp+0..20]  = t1: score(-1), other(999), dist, best_c, best_r, cnt(0)
    #   [sp+24..44] = t2: score(-1), other(999), dist, best_c, best_r, cnt(0)
    #   [sp+48]=ref_c, [sp+52]=ref_r, [sp+56]=best_other, [sp+60]=row_prev_cnt
    insns.extend((
        0xE92D4FF0,  # push {r4, r5, r6, r7, r8, sb, sl, fp, lr}
        0xE24DD044,  # sub  sp, sp, #68
        0xE1A04000,  # mov  r4, r0             (Ch0 base)
        0xE2845B07,  # add  r5, r4, #0x1c00    (Ch1 base)
        0xE2849B0E,  # add  sb, r4, #0x3800    (Ch2 base)
    ))

    # Anchor hysteresis: prev_anchor if has_box != 0, else (28, 28)
    ldr32(0, s3.STATUS_VA)
    insns.extend((
        0xE3A0101C,  # mov   r1, #28
        0xE3A0201C,  # mov   r2, #28
        0xE5903024,  # ldr   r3, [r0, #0x24]
        0xE3530000,  # cmp   r3, #0
        0x1590304C,  # ldrne r3, [r0, #0x4c]
        0x120310FF,  # andne r1, r3, #0xff
        0x11A02423,  # lsrne r2, r3, #8
        0xE3510037,  # cmp   r1, #55
        0x83A0101C,  # movhi r1, #28
        0xE3520037,  # cmp   r2, #55
        0x83A0201C,  # movhi r2, #28
        0xE58D1030,  # str   r1, [sp, #48]     (ref_c)
        0xE58D2034,  # str   r2, [sp, #52]     (ref_r)
        0xE3E00000,  # mvn   r0, #0            (-1: best_score init)
        s3.encode_movw(1, 999),  # 999: best_other init (first valid hit unconditionally wins blo)
        0xE3A03000,  # mov   r3, #0
        0xE58D0000,  # str   r0, [sp, #0]      (t1_score = -1)
        0xE58D1004,  # str   r1, [sp, #4]      (t1_other = 999)
        0xE58D0018,  # str   r0, [sp, #24]     (t2_score = -1)
        0xE58D101C,  # str   r1, [sp, #28]     (t2_other = 999)
        0xE58D3014,  # str   r3, [sp, #20]     (t1_cnt = 0)
        0xE58D302C,  # str   r3, [sp, #44]     (t2_cnt = 0)
        0xE3A06000,  # mov   r6, #0            (row = 0)
    ))

    # Unified Pass 1: only strict argmax (c1 > c2 or c2 > c1) can satisfy P >= 60% or P >= 77%
    mark("p1_row_loop")
    insns.append(0xE3A07000)  # mov r7, #0 (col = 0)
    mark("p1_col_loop")
    insns.extend((
        0xE087A306,  # add   sl, r7, r6, lsl #6
        0xE1A0A08A,  # lsl   sl, sl, #1
        0xE19400FA,  # ldrsh r0, [r4, sl]      (c0)
        0xE19510FA,  # ldrsh r1, [r5, sl]      (c1)
        0xE19920FA,  # ldrsh r2, [sb, sl]      (c2)
        0xE1510002,  # cmp   r1, r2
    ))
    b("p1_next", cond=0x0)  # beq (neither can exceed 50%)
    insns.extend((
        0xE1A0C000,  # mov   ip, r0            (c0)
        0xC1A00001,  # movgt r0, r1            (p = c1)
        0xC3A03001,  # movgt r3, #1            (tier = 1)
        0xB1A00002,  # movlt r0, r2            (p = c2)
        0xB1A02001,  # movlt r2, r1            (b = c1)
        0xB3A03002,  # movlt r3, #2            (tier = 2)
        0xE1A0100C,  # mov   r1, ip            (a = c0)
    ))
    call(cave2_syms["softmax_other"])  # r0=other (999 if not argmax), r1=da+db (log-odds score), r3 preserved
    insns.extend((
        0xE3530001,  # cmp   r3, #1
        0x03A0C000 | CH1_OTHER_MAX,  # moveq ip, #170
        0x13A0C000 | CH2_OTHER_MAX,  # movne ip, #75
        0xE150000C,  # cmp   r0, ip
    ))
    b("p1_next", cond=0x8)  # bhi
    insns.extend((
        0xE28DB000,  # add   fp, sp, #0        (&t1)
        0xE3530002,  # cmp   r3, #2
        0x028BB018,  # addeq fp, fp, #24       (&t2)
        0xE59BC014,  # ldr   ip, [fp, #20]
        0xE28CC001,  # add   ip, ip, #1
        0xE58BC014,  # str   ip, [fp, #20]     (cnt++)
        # Compare K-3 III probability-first lexicographic (other <, score = da+db >, dist <)
        0xE59D2030,  # ldr   r2, [sp, #48]     (ref_c)
        0xE0572002,  # subs  r2, r7, r2
        0x42622000,  # rsbmi r2, r2, #0
        0xE59DC034,  # ldr   ip, [sp, #52]     (ref_r)
        0xE056C00C,  # subs  ip, r6, ip
        0x426CC000,  # rsbmi ip, ip, #0
        0xE082200C,  # add   r2, r2, ip        (dist)
        0xE59B3004,  # ldr   r3, [fp, #4]      (best_other)
        0xE1500003,  # cmp   r0, r3
    ))
    b("p1_next", cond=0x8)  # bhi (larger other = lower softmax probability)
    b("p1_take", cond=0x3)  # blo (smaller other = higher softmax probability!)
    insns.extend((
        0xE59B3000,  # ldr   r3, [fp, #0]      (best_score = da + db)
        0xE1510003,  # cmp   r1, r3
    ))
    b("p1_next", cond=0xB)  # blt
    b("p1_take", cond=0xC)  # bgt
    insns.extend((
        0xE59B3008,  # ldr   r3, [fp, #8]      (best_dist)
        0xE1520003,  # cmp   r2, r3
    ))
    b("p1_next", cond=0x2)  # bhs
    mark("p1_take")
    insns.extend((
        0xE58B1000,  # str   r1, [fp, #0]      (score = da + db)
        0xE58B0004,  # str   r0, [fp, #4]      (other)
        0xE58B2008,  # str   r2, [fp, #8]      (dist)
        0xE58B700C,  # str   r7, [fp, #12]     (best_c)
        0xE58B6010,  # str   r6, [fp, #16]     (best_r)
    ))

    mark("p1_next")
    insns.extend((
        0xE2877001,  # add  r7, r7, #1
        0xE3570038,  # cmp  r7, #56
    ))
    b("p1_col_loop", cond=0xB)
    insns.extend((
        0xE2866001,  # add  r6, r6, #1
        0xE3560038,  # cmp  r6, #56
    ))
    b("p1_row_loop", cond=0xB)

    # Tier 1 evaluated FIRST (r4=Ch0, r5=Ch1, sb=Ch2 intact)
    insns.extend((
        0xE59D0014,  # ldr  r0, [sp, #20]      (t1_cnt)
        0xE3500000 | CH1_MIN_LOCAL,
    ))
    b("check_tier2", cond=0xB)
    insns.extend((
        0xE59D1004,  # ldr  r1, [sp, #4]       (t1_other)
        0xE3510000 | CH1_PEAK_OTHER_MAX,
    ))
    b("check_tier2", cond=0x8)
    insns.extend((
        0xE28DB000,  # add  fp, sp, #0        (&t1)
        0xE3A00001,  # tier = 1
        0xE3A01000 | LOCAL_RADIUS_CH1,
        0xE3A02000 | CH1_OTHER_MAX,
        0xE3A03000 | CH1_MIN_LOCAL,
    ))
    b("run_pass2")

    # Tier 2 evaluated ONLY if Tier 1 failed (swap r5 <-> sb -> r5=Ch2, sb=Ch1)
    mark("check_tier2")
    insns.extend((
        0xE59D002C,  # ldr  r0, [sp, #44]      (t2_cnt)
        0xE3500000 | CH2_MIN_LOCAL,
    ))
    b("scan_none", cond=0xB)
    insns.extend((
        0xE1A00005,  # swap r5 <-> sb
        0xE1A05009,
        0xE1A09000,
        0xE28DB018,  # add  fp, sp, #24       (&t2)
        0xE3A00002,  # tier = 2
        0xE3A01000 | LOCAL_RADIUS_CH2,
        0xE3A02000 | CH2_OTHER_MAX,
        0xE3A03000 | CH2_MIN_LOCAL,
    ))

    # Shared 2-iteration local mean-shift centroid & local span extractor
    mark("run_pass2")
    insns.extend((
        0xE59BC004,  # ldr  ip, [fp, #4]       (best_other)
        0xE58DC038,  # str  ip, [sp, #56]
        0xE59BC00C,  # ldr  ip, [fp, #12]      (ac = best_c)
        0xE59B6010,  # ldr  r6, [fp, #16]      (ar = best_r)
        0xE58D0000,  # tier
        0xE58D1004,  # radius
        0xE58D2008,  # other_max
        0xE58D300C,  # min_local
        0xE58DC010,  # ac
        0xE58D6014,  # ar
        0xE3A00000,
        0xE58D0018,  # iter = 0
    ))

    mark("p2_iter_loop")
    insns.extend((
        0xE3A08000,  # sum_w = 0
        0xE3A0A000,  # sum_wc = 0
        0xE3A0B000,  # sum_wr = 0
        0xE58D801C,  # local_cnt = 0
        0xE3A00037,  # 55
        0xE58D0020,  # min_c = 55
        0xE58D8024,  # max_c = 0
        0xE58D0028,  # min_r = 55
        0xE58D802C,  # max_r = 0
        0xE59D0004,  # radius
        0xE59D1014,  # ar
        0xE0516000,  # subs r6, r1, r0
        0xB3A06000,  # r_start
        0xE0811000,  # add  r1, r1, r0
        0xE3510037,
        0xC3A01037,
        0xE58D1030,  # r_end
        0xE59D1010,  # ac
        0xE0512000,
        0xB3A02000,
        0xE58D2010,  # c_start
        0xE0811000,
        0xE3510037,
        0xC3A01037,
        0xE58D1034,  # c_end
    ))

    mark("p2_row_loop")
    insns.extend((
        0xE59D001C,  # ldr  r0, [sp, #28]
        0xE58D003C,  # str  r0, [sp, #60]      (row_prev_cnt)
        0xE59D7010,  # ldr  r7, [sp, #16]      (col = c_start)
    ))
    mark("p2_col_loop")
    insns.extend((
        0xE0870306,  # add   r0, r7, r6, lsl #6
        0xE1A00080,  # lsl   r0, r0, #1        (byte_off)
        0xE19410F0,  # ldrsh r1, [r4, r0]      (a)
        0xE19920F0,  # ldrsh r2, [sb, r0]      (b)
        0xE19500F0,  # ldrsh r0, [r5, r0]      (p, loads r0 LAST)
    ))
    call(cave2_syms["softmax_other"])  # r0 = other (or 999)
    insns.extend((
        0xE59D2008,  # ldr   r2, [sp, #8]      (other_max)
        0xE1500002,  # cmp   r0, r2
    ))
    b("p2_col_next", cond=0x8)
    insns.extend((
        0xE2600C01,  # rsb   r0, r0, #256      (w = 256 - other)
        0xE0888000,  # add   r8, r8, r0        (sum_w += w)
        0xE1A01207,  # lsl   r1, r7, #4
        0xE2811008,  # add   r1, r1, #8
        0xE02AA190,  # mla   sl, r0, r1, sl    (sum_wc)
        0xE1A01206,  # lsl   r1, r6, #4
        0xE2811008,  # add   r1, r1, #8
        0xE02BB190,  # mla   fp, r0, r1, fp    (sum_wr)
        0xE59D101C,  # ldr   r1, [sp, #28]
        0xE2811001,  # add   r1, r1, #1
        0xE58D101C,  # str   r1, [sp, #28]     (local_cnt++)
        0xE59D1020,  # ldr   r1, [sp, #32]
        0xE1570001,  # cmp   r7, r1
        0xB58D7020,  # strlt r7, [sp, #32]     (min_c)
        0xE59D1024,  # ldr   r1, [sp, #36]
        0xE1570001,  # cmp   r7, r1
        0xC58D7024,  # strgt r7, [sp, #36]     (max_c)
    ))
    mark("p2_col_next")
    insns.extend((
        0xE2877001,  # add  r7, r7, #1
        0xE59D0034,  # ldr  r0, [sp, #52]
        0xE1570000,  # cmp  r7, r0
    ))
    b("p2_col_loop", cond=0xD)

    # Track min_r and max_r once per row (row increases monotonically)
    insns.extend((
        0xE59D001C,  # ldr  r0, [sp, #28]
        0xE59D103C,  # ldr  r1, [sp, #60]
        0xE1500001,  # cmp  r0, r1
    ))
    b("p2_row_next", cond=0x0)
    insns.extend((
        0xE59D1028,  # ldr   r1, [sp, #40]     (min_r)
        0xE1560001,  # cmp   r6, r1
        0xB58D6028,  # strlt r6, [sp, #40]
        0xE58D602C,  # str   r6, [sp, #44]     (max_r = row)
    ))
    mark("p2_row_next")
    insns.extend((
        0xE2866001,  # add  r6, r6, #1
        0xE59D0030,  # ldr  r0, [sp, #48]
        0xE1560000,  # cmp  r6, r0
    ))
    b("p2_row_loop", cond=0xD)

    insns.extend((
        0xE59D001C,  # local_cnt
        0xE59D100C,  # min_local
        0xE1500001,
    ))
    b("p2_cnt_ok", cond=0xA)
    insns.extend((
        0xE59D0000,  # tier
        0xE3500001,
    ))
    b("check_tier2", cond=0x0)
    b("scan_none")

    mark("p2_cnt_ok")
    insns.extend((
        0xE1A0000A,  # r0 = sum_wc
        0xE1A01008,  # r1 = sum_w
    ))
    call(cave2_syms["udiv10_round"])
    insns.extend((
        0xE1A06000,  # r6 = u_c_q4
        0xE1A0000B,  # r0 = sum_wr
        0xE1A01008,  # r1 = sum_w
    ))
    call(cave2_syms["udiv10_round"])
    insns.extend((
        0xE1A07000,  # r7 = u_r_q4
        0xE59D0018,  # iter
        0xE3500000,
    ))
    b("p2_done", cond=0x1)
    insns.extend((
        0xE3A00001,
        0xE58D0018,  # iter = 1
        0xE1A01226,
        0xE58D1010,  # ac = u_c_q4 >> 4
        0xE1A01227,
        0xE58D1014,  # ar = u_r_q4 >> 4
    ))
    b("p2_iter_loop")

    mark("p2_done")
    insns.extend((
        0xE1A00226,
        0xE1A01227,
        0xE1800401,
    ))
    ldr32(2, s3.STATUS_VA)
    insns.extend((
        0xE582004C,  # str  r0, [r2, #0x4c]    (prev_anchor_packed)
        0xE3A00C64,  # mov  r0, #25600
        0xE59D1038,  # ldr  r1, [sp, #56]      (best_other)
        0xE2811C01,  # add  r1, r1, #256
    ))
    call(cave2_syms["udiv10_round"])
    insns.extend((
        0xE59D1000,
        0xE1800801,  # (tier << 16) | prob_pct (best_other <= 255 -> 50..100 already)
        0xE59D3024,
        0xE59DC020,
        0xE043300C,
        0xE2833001,  # span_c
        0xE59D102C,
        0xE59DC028,
        0xE041100C,
        0xE2811001,  # span_r
        0xE1833401,
        0xE59D101C,
        0xE1833801,
        0xE1A01006,  # u_c_q4
        0xE1A02007,  # u_r_q4
        0xE28DD044,
        0xE8BD8FF0,
    ))

    mark("scan_none")
    insns.extend((
        0xE3A00000,
        0xE28DD044,
        0xE8BD8FF0,
    ))

    code, labels = finalize()
    if CAVE3_VA + len(code) > CAVE3_CODE_LIMIT_VA:
        raise ValueError(
            f"Cave 3 ({len(code)} B, ends {hex(CAVE3_VA + len(code))}) overlaps EXP_LUT @ {hex(EXP_LUT_VA)}"
        )
    return code, {
        "sub_sample_ahip_rgb": labels["sub_sample_ahip_rgb"],
        "sub_scan_heatmap": labels["sub_scan_heatmap"],
    }


def _build_render_osd(render_va: int, bypass_va: int) -> bytes:
    """OSD renderer @ CAVE2_VA (runs while native FD mutex r6 is held; Option B hiding)."""
    insns, mark, b, _, jump, ldr32, finalize = _make_asm(render_va)

    insns.extend((0xE5940078, 0xE3500000))
    jump(bypass_va, cond=0x0)
    insns.extend((0xE5940038, 0xE3500000))
    jump(bypass_va, cond=0x0)
    insns.extend((0xE1D40FBC, 0xE3500030))
    jump(bypass_va, cond=0x3)
    insns.extend((0xE1D40FBE, 0xE3500030))
    jump(bypass_va, cond=0x3)

    insns.append(0xE92D47F0)  # push {r4, r5, r6, r7, r8, sb, sl, lr}

    ldr32(3, s3.STATUS_VA)
    ldr32(1, LIVE_AI_COOKIE)
    insns.extend((
        0xE5932018,
        0xE1520001,
    ))
    b("hide_box", cond=0x1)
    insns.extend((
        0xE5932024,
        0xE3520000,
    ))
    b("hide_box", cond=0x0)

    # Load smoothed Q8 box & clamp safely against current fd_w/fd_h (floor 0x1c00 = 28 px)
    insns.extend((
        0xE5937028,
        0xE593802C,
        0xE5935030,
        0xE5939034,
        0xE1D40FBC,
        0xE1A00400,
        0xE1D41FBE,
        0xE1A01401,
        0xE3A02B07,  # mov r2, #0x1c00 (defensive 28 px floor < 30 px MIN_HEAD_BOX_Q8)
        0xE1550002,
        0x31A05002,
        0xE1500001,
        0x31A02000,
        0x21A02001,
        0xE1550002,
        0x81A05002,
        0xE0402005,
        0xE1570002,
        0x81A07002,
        0xE0412005,
        0xE1580002,
        0x81A08002,
    ))

    # Zero 100 bytes (0x64) of m_DetectedFaceResult[0]
    insns.extend((
        0xE594A078,
        0xE3A00000,
        0xE3A01000,
    ))
    mark("zero_face_loop")
    insns.extend((
        0xE78A0001,
        0xE2811004,
        0xE3510064,
    ))
    b("zero_face_loop", cond=0xB)

    insns.extend((
        0xE58A7000,
        0xE58A8004,
        0xE58A5008,
        0xE58A500C,
        0xE58A904C,
        0xE5941118,
        0xE3510000,
        0x03A01001,
        0xE58A1050,
        0xE08720A5,
        0xE5842100,
        0xE08820A5,
        0xE5842104,
        0xE5845108,
        0xE584510C,
        0xE5841118,
        0xE584111C,
        0xE3A02001,
        0xE5C4213C,
        0xE3A00000,
        0xE5C40121,
        0xE2843F49,
        0xE1C300B0,
        0xE5943038,
        0xE3A01001,
        0xE5831030,
        0xE3A01000,
    ))
    b("set_main_idx")

    mark("hide_box")
    insns.extend((
        0xE5943038,
        0xE3A01000,
        0xE5831030,
    ))

    mark("set_main_idx")
    insns.append(0xE5831034)  # main_face_idx = 0

    mark("render_done")
    insns.append(0xE8BD47F0)
    jump(s3.HOOK2_AI_RESUME_VA)

    code, _ = finalize()
    return code


def _build_live_ai_core(core_va: int, cave3_syms: dict[str, int]) -> tuple[bytes, dict[str, int]]:
    """Continuous fault-tolerant DSP core + tracker + shared math helpers inside Cave 2."""
    insns, mark, b, call, jump, ldr32, finalize = _make_asm(core_va)

    insns.extend((
        0xE92D47F0,  # push {r4, r5, r6, r7, r8, sb, sl, lr}
        0xE24DD008,  # sub  sp, sp, #8
        0xE1A0A000,  # mov  sl, r0
    ))

    ldr32(8, s3.STATUS_VA)
    insns.append(0xE5982018)
    ldr32(3, LIVE_AI_COOKIE)
    insns.append(0xE1520003)
    b("exit_done", cond=0x1)

    insns.extend((
        0xE5987000,
        0xE3570003,
    ))
    b("do_poll_in_flight", cond=0x0)
    insns.append(0xE3570002)
    b("do_send_next", cond=0x0)
    insns.append(0xE3570000)
    b("exit_done", cond=0x1)

    # --- State 0: Warmup (20 frames) + One-shot LoadDsp ---
    insns.extend((
        0xE5982004,
        0xE2822001,
        0xE5882004,
        0xE2523000 | WARMUP_FRAMES,
    ))
    b("exit_done", cond=0x3)
    insns.append(0xE3130000 | LOAD_RETRY_MASK)
    b("exit_done", cond=0x1)

    call(s3.GET_DSP_SVC_VA)
    insns.append(0xE3500000)
    b("exit_done", cond=0x0)
    insns.extend((
        0xE5905004,
        0xE3550000,
    ))
    b("have_svc_init", cond=0x1)

    insns.append(0xE3A01000)
    call(s3.CREATE_DSP_SVC_VA)
    call(s3.GET_DSP_SVC_VA)
    insns.append(0xE3500000)
    b("exit_done", cond=0x0)
    insns.extend((
        0xE5905004,
        0xE3550000,
    ))
    b("exit_done", cond=0x0)

    mark("have_svc_init")
    insns.extend((
        0xE5952004,
        0xE3120004,
    ))
    b("exit_done", cond=0x1)

    insns.extend((
        0xE1A00005,
        0xE3A01043,
        0xE3A02000,
        0xE3A03000,
    ))
    call(s3.LOAD_DSP_VA)
    insns.append(0xE3500000)
    b("load_ok", cond=0x1)

    insns.extend((
        0xE1A00005,
        0xE3A01049,
        0xE3A02000,
        0xE3A03000,
    ))
    call(s3.LOAD_DSP_VA)
    insns.append(0xE3500000)
    b("load_ok", cond=0x1)

    insns.extend((
        0xE3A02000,
        0xE5852004,
        0xE3A02001,
        0xE5882010,
    ))
    b("exit_done")

    mark("load_ok")
    insns.extend((
        0xE5885020,
        0xE3A00002,
        0xE5880000,
    ))
    b("exit_done")

    # --- State 2: Sample rotated AHIP RGB -> Start(0x52100000) -> SendMessage(4) ---
    mark("do_send_next")
    call(s3.GET_DSP_SVC_VA)
    b("check_service_identity", bl=True)
    insns.extend((
        0xE5952004,
        0xE3120004,
    ))
    b("stop_and_retry", cond=0x1)

    insns.append(0xE1A0000A)
    call(cave3_syms["sub_sample_ahip_rgb"])
    insns.append(0xE3500000)
    b("exit_done", cond=0x0)

    insns.append(0xE1A00005)
    ldr32(1, 0x52100000)
    insns.append(0xE3A02000)
    call(s3.START_DSP_VA)
    insns.extend((
        0xE5880010,
        0xE5952004,
        0xE3120004,
    ))
    b("exit_done", cond=0x0)

    ldr32(7, s3.TENSOR_MSG_VA)
    ldr32(0, s3.DESC_TABLE_VA)
    insns.extend((
        0xE5870000,
        0xE3A00000,
        0xE5870004,
        0xE5870020,
        0xE5870024,
    ))
    ldr32(0, s3.TENSOR_IN_VA)
    insns.append(0xE5870008)
    ldr32(0, s3.HEAD_WEIGHTS_VA)
    insns.append(0xE587000C)
    ldr32(0, s3.MEAN_WEIGHTS_VA)
    insns.append(0xE5870010)
    ldr32(0, 0x00080808)
    insns.append(0xE5870014)
    ldr32(0, 0x0C0C0C0A)
    insns.append(0xE5870018)
    insns.append(0xE2800002)  # 0x0C0C0C0C
    insns.append(0xE587001C)

    insns.extend((
        0xE2886008,
        0xE3A00000,
        0xE1C600B0,
        0xE1A00005,
        0xE1A01006,
        0xE3A02004,
        0xE3A03064,
        0xE58D7000,
    ))
    call(s3.SEND_MSG_VA)
    insns.extend((
        0xE5880010,
        0xE3500000,
    ))
    b("stop_and_retry", cond=0x1)

    insns.extend((
        0xE3A00000,
        0xE588000C,
        0xE3A00003,
        0xE5880000,
    ))
    b("exit_done")

    # --- State 3: Service 0x1f + read-only [svc+0x54] slot poll -> Wait -> Stop(100) -> Scan ---
    mark("do_poll_in_flight")
    call(s3.GET_DSP_SVC_VA)
    b("check_service_identity", bl=True)
    insns.extend((
        0xE5952004,
        0xE3120004,
    ))
    b("reset_to_send_next", cond=0x0)

    insns.extend((
        0xE2886008,
        0xE1D610B0,
        0xE3510005,
    ))
    b("stop_and_retry", cond=0x8)

    insns.extend((
        0xE3A00000,
        0xE58D0000,
        0xE285000C,
        0xE3A01801,
        0xE1A0200D,
        0xE3A03000,
    ))
    call(OS_EF_WAIT_VA)
    insns.append(0xE3500000)
    b("skip_flush1f", cond=0x0)

    insns.extend((
        0xE5953058,
        0xE3530000,
    ))
    b("clear_1f", cond=0x0)
    insns.extend((
        0xE5932008,
        0xE3520000,
    ))
    b("clear_1f", cond=0x0)
    insns.append(0xE1A04003)
    call(s3.GET_CACHE_MGR_VA)
    insns.extend((
        0xE5941004,
        0xE5942008,
    ))
    call(s3.FLUSH_DCACHE_VA)
    insns.extend((
        0xE3A00000,
        0xE5840004,
        0xE5840008,
    ))

    mark("clear_1f")
    insns.extend((
        0xE285000C,
        0xE3A01801,
    ))
    call(OS_EF_CLEAR_VA)

    mark("skip_flush1f")
    insns.extend((
        0xE1D610B0,
        0xE5950054,
        0xE3500000,
    ))
    b("stop_and_retry", cond=0x0)

    insns.append(0xE3A02000)
    mark("scan_slot")
    insns.extend((
        0xE7D03002,
        0xE3530000,
    ))
    b("scan_next", cond=0x0)
    insns.extend((
        0xE0807002,
        0xE1D730B2,
        0xE1530001,
    ))
    b("slot_ready", cond=0x0)
    mark("scan_next")
    insns.extend((
        0xE282202C,
        0xE3520F42,
    ))
    b("scan_slot", cond=0x1)

    insns.extend((
        0xE598200C,
        0xE2822001,
        0xE588200C,
        0xE3520000 | MAX_POLL_FRAMES,
    ))
    b("exit_done", cond=0x3)

    insns.extend((
        0xE3A00008,
        0xE5880010,
    ))
    b("stop_and_retry")

    mark("slot_ready")
    insns.extend((
        0xE1A00005,
        0xE1D610B0,
        0xE3A02000,
    ))
    ldr32(3, RESP_BUF_VA)
    call(s3.WAIT_MSG_VA)
    insns.append(0xE1A04000)

    b("stop_dsp", bl=True)
    insns.extend((
        0xE3A00002,
        0xE5880000,
        0xE3540000,
    ))
    b("wait_ok", cond=0x0)
    insns.append(0xE5884010)
    b("exit_done")

    mark("wait_ok")
    insns.extend((
        0xE5982014,
        0xE2822001,
        0xE5882014,
    ))
    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, HEATMAP_OUT_VA)
    insns.append(s3.encode_movw(2, HEATMAP_LEN))
    call(s3.INVAL_DCACHE_VA)

    ldr32(0, HEATMAP_OUT_VA)
    call(cave3_syms["sub_scan_heatmap"])
    insns.append(0xE58DA000)
    b("sub_update_tracking_box", bl=True)
    b("exit_done")

    mark("stop_and_retry")
    b("stop_dsp", bl=True)
    mark("reset_to_send_next")
    insns.extend((
        0xE3A00002,
        0xE5880000,
    ))
    b("exit_done")

    mark("stop_dsp")
    insns.extend((
        0xE1A00005,
        0xE3A01064,
    ))
    jump(STOP_DSP_VA)

    mark("check_service_identity")
    insns.append(0xE3500000)
    b("exit_done", cond=0x0)
    insns.extend((0xE5905004, 0xE3550000))
    b("exit_done", cond=0x0)
    insns.extend((0xE5982020, 0xE1550002))
    b("exit_done", cond=0x1)
    insns.append(0xE12FFF1E)

    mark("exit_done")
    insns.extend((
        0xE28DD008,
        0xE8BD87F0,
    ))

    # ---------------------------------------------------------------------
    # sub_update_tracking_box: 2-cycle cold confirmation, inverse orientation,
    # tight head/body span sizing, position/size deadbands & smooth EMA
    # ---------------------------------------------------------------------
    mark("sub_update_tracking_box")
    insns.append(0xE92D4FF0)  # push {r4..fp, lr} (36 B -> fd_obj @ [sp,#36])
    ldr32(4, s3.STATUS_VA)
    insns.extend((
        0xE5840038,  # last_peak
        0xE1A0C823,  # lsr ip, r3, #16
        0xE584C03C,  # last_area
        0xE3500000,
    ))
    b("trk_no_subject", cond=0x0)

    insns.extend((
        0xE3A0C000,
        0xE584C01C,  # lost_cycles = 0
        0xE594B024,  # has_box
        0xE35B0000,
    ))
    b("trk_confirmed", cond=0x1)
    insns.extend((
        0xE594C048,
        0xE28CC001,
        0xE584C048,
        0xE35C0000 | ACQ_CONFIRM_CYCLES,
        0xB8BD8FF0,  # poplt {r4..fp, pc}
    ))

    mark("trk_confirmed")
    insns.extend((
        0xE59D5024,  # fd_obj
        0xE1D56FBC,  # fd_w
        0xE3560000,
        s3.encode_movw(6, 320, cond=0x0),
        0xE1D57FBE,  # fd_h
        0xE3570000,
        s3.encode_movw(7, 212, cond=0x0),
        0xE1A05820,  # r5 = tier (1 or 2)
        0xE6FF0070,  # r0 = prob_pct
        # Compact conditional inverse rotation: r8=s_c_q4, sb=s_r_q4, r1=span_x, r2=span_y
        0xE594C044,  # ip = in_flight_orient
        s3.encode_movw(14, 896),
        0xE1A08001,  # default orient 0: s_c = u_c_q4
        0xE1A09002,  # default orient 0: s_r = u_r_q4
        0xE35C0001,  # orient 1 ('L')?
        0x004E8002,  # subeq r8, lr, r2
        0x01A09001,  # moveq sb, r1
        0xE35C0002,  # orient 2 ('D')?
        0x004E8001,  # subeq r8, lr, r1
        0x004E9002,  # subeq sb, lr, r2
        0xE35C0003,  # orient 3 ('R')?
        0x01A08002,  # moveq r8, r2
        0x004E9001,  # subeq sb, lr, r1
        # Sub-pixel Q8 center (K-3 III 56x38 letterbox: s_c_q4 in 0..896, s_r_q4 in 144..752)
        s3.encode_movw(14, 585),
        0xE0080896,  # mul   r8, r6, r8
        0xE008089E,  # mul   r8, lr, r8
        0xE1A085A8,  # lsr   r8, r8, #11      (cx_q8)
        0xE2599090,  # subs  sb, sb, #144     (s_r_q4 - 9*16 letterbox top bar)
        0x43A09000,  # movmi sb, #0
        s3.encode_movw(14, 862),
        0xE0090997,  # mul   sb, r7, sb
        0xE009099E,  # mul   sb, lr, sb
        0xE1A095A9,  # lsr   sb, sb, #11      (cy_q8 across 38 active rows = 608 Q4 units)
        # Isotropic 1:1 square-pixel Q8 box size from max(span_c, span_r) * fd_w / 56
        0xE20310FF,  # and   r1, r3, #0xff    (span_c)
        0xE1A02423,  # lsr   r2, r3, #8
        0xE20220FF,  # and   r2, r2, #0xff    (span_r)
        0xE1510002,  # cmp   r1, r2
        0xB1A01002,  # movlt r1, r2           (max_span)
        0xE0010196,  # mul   r1, r6, r1       (max_span * fd_w)
        0xE3A0C049,  # mov   ip, #73
        0xE00A0C91,  # mul   sl, r1, ip
        0xE1A0A22A,  # lsr   sl, sl, #4       (raw_size_q8)
        0xE3550001,  # cmp   r5, #1
        0x108AA08A,  # addne sl, sl, sl, lsl #1
        0x11A0A12A,  # lsrne sl, sl, #2       (Tier 2: raw = (raw * 3) >> 2)
        0x03A0CC1C,  # moveq ip, #0x1c00      (MIN_HEAD_BOX_Q8)
        0x13A0CC22,  # movne ip, #0x2200      (MIN_BODY_BOX_Q8)
        0xE15A000C,  # cmp   sl, ip
        0xB1A0A00C,  # movlt sl, ip
        0x03A0CC34,  # moveq ip, #0x3400      (MAX_HEAD_BOX_Q8)
        0x13A0CC38,  # movne ip, #0x3800      (MAX_BODY_BOX_Q8)
        0xE15A000C,  # cmp   sl, ip
        0xC1A0A00C,  # movgt sl, ip
    ))

    mark("size_aspect_cap")
    insns.extend((
        0xE1560007,
        0xD1A0C006,
        0xC1A0C007,
        0xE1A0C38C,
        0xE15A000C,
        0xC1A0A00C,
        0xE35B0000,
    ))
    b("trk_clamp_store", cond=0x0)

    insns.extend((
        0xE5941028,
        0xE594202C,
        0xE5943030,
        0xE08110A3,
        0xE08220A3,
        0xE0585001,
        0x42655000,
        s3.encode_movw(12, POS_DEADBAND_Q8),
        0xE155000C,
    ))
    b("trk_ema_pos", cond=0xC)
    insns.extend((
        0xE059E002,
        0x426EE000,
        0xE15E000C,
    ))
    b("trk_ema_pos", cond=0xC)
    insns.extend((
        0xE1A08001,
        0xE1A09002,
    ))
    b("trk_smooth_size")

    mark("trk_ema_pos")
    insns.extend((
        0xE0811081,
        0xE0888001,
        0xE2888002,
        0xE1A08128,
        0xE0822082,
        0xE0899002,
        0xE2899002,
        0xE1A09129,
    ))

    mark("trk_smooth_size")
    insns.extend((
        0xE05A5003,
        0x42655000,
        s3.encode_movw(12, SIZE_DEADBAND_Q8),
        0xE155000C,
        0xD1A0A003,
    ))
    b("trk_clamp_store", cond=0xD)
    insns.extend((
        0xE0633183,
        0xE08AA003,
        0xE28AA004,
        0xE1A0A1AA,
    ))

    mark("trk_clamp_store")
    insns.extend((
        0xE05880AA,
        0x43A08000,
        0xE1A0C406,
        0xE04CC00A,
        0xE158000C,
        0xC1A0800C,
        0xE05990AA,
        0x43A09000,
        0xE1A0C407,
        0xE04CC00A,
        0xE159000C,
        0xC1A0900C,
        0xE5848028,
        0xE584902C,
        0xE584A030,
        0xE5840034,
        0xE3A01001,
        0xE5841024,
        0xE3A01000,
        0xE5841048,
        0xE8BD8FF0,
    ))

    mark("trk_no_subject")
    insns.extend((
        0xE3A01000,
        0xE5841048,
        0xE594101C,
        0xE2811001,
        0xE584101C,
        0xE3510000 | LOST_HOLD_CYCLES,
        0xC3A02000,
        0xC5842024,
        0xE8BD8FF0,
    ))

    # ---------------------------------------------------------------------
    # Shared leaf helper: udiv10_round(r0 = num, r1 = den) -> (num + den/2) // den
    # ---------------------------------------------------------------------
    mark("udiv10_round")
    insns.extend((
        0xE3510000,
        0x03A00000,
        0x012FFF1E,
        0xE08000A1,
        0xE3A02000,
        0xE1A01481,
        0xE3A03C02,
    ))
    mark("udiv_loop")
    insns.extend((
        0xE1500001,
        0x20400001,
        0x21822003,
        0xE1A010A1,
        0xE1B030A3,
    ))
    b("udiv_loop", cond=0x1)
    insns.extend((
        0xE1A00002,
        0xE12FFF1E,
    ))

    # ---------------------------------------------------------------------
    # Shared leaf helper: softmax_other(r0=p, r1=a, r2=b) -> r0=other (or 999), r1=da+db
    # Preserves r3 and all callee-saved registers.
    # ---------------------------------------------------------------------
    mark("softmax_other")
    insns.extend((
        0xE1500001,  # cmp   r0, r1 (p > a?)
    ))
    b("sm_fail", cond=0xD)
    insns.append(0xE1500002)  # cmp r0, r2 (p > b?)
    b("sm_fail", cond=0xD)
    insns.extend((
        0xE0401001,  # sub    r1, r0, r1 (da > 0)
        0xE0402002,  # sub    r2, r0, r2 (db > 0)
    ))
    ldr32(12, EXP_LUT_VA)
    insns.extend((
        0xE351000F,  # cmp    r1, #15
        0x97DC0001,  # ldrlsb r0, [ip, r1]
        0x83A00000,  # movhi  r0, #0
        0xE352000F,  # cmp    r2, #15
        0x97DCC002,  # ldrlsb ip, [ip, r2]
        0x83A0C000,  # movhi  ip, #0
        0xE080000C,  # add    r0, r0, ip (other = ea + eb)
        0xE0811002,  # add    r1, r1, r2 (score = da + db)
        0xE12FFF1E,  # bx     lr
    ))
    mark("sm_fail")
    insns.extend((
        s3.encode_movw(0, 999),
        0xE12FFF1E,
    ))

    code, labels = finalize()
    return code, {
        "sub_update_tracking_box": labels["sub_update_tracking_box"],
        "udiv10_round": labels["udiv10_round"],
        "softmax_other": labels["softmax_other"],
    }


def _resolve_all() -> dict:
    """Two-pass deterministic build resolving mutual calls between Cave 2 and Cave 3."""
    render_len = len(_build_render_osd(s3.CAVE2_VA, s3.CAVE2_VA))
    shim_va = s3.CAVE2_VA + render_len
    shim_len = 28
    bypass_va = shim_va + shim_len
    bypass_len = 20
    core_va = bypass_va + bypass_len

    # Pass 1: discover exact symbol VAs (all bl instructions are constant 4 B regardless of target)
    dummy_c3 = {"sub_sample_ahip_rgb": CAVE3_VA, "sub_scan_heatmap": CAVE3_VA}
    _, cave2_syms = _build_live_ai_core(core_va, dummy_c3)
    _, cave3_syms = _build_cave3_body(cave2_syms)

    # Pass 2: assemble final linked payloads with true VAs
    core, cave2_syms2 = _build_live_ai_core(core_va, cave3_syms)
    cave3, cave3_syms2 = _build_cave3_body(cave2_syms2)
    assert cave2_syms == cave2_syms2 and cave3_syms == cave3_syms2
    return {
        "render_len": render_len,
        "shim_va": shim_va,
        "shim_len": shim_len,
        "bypass_va": bypass_va,
        "bypass_len": bypass_len,
        "core_va": core_va,
        "core": core,
        "cave3": cave3,
        "cave2_syms": cave2_syms2,
        "cave3_syms": cave3_syms2,
    }


def build_live_ai_cave3() -> tuple[bytes, dict[str, int]]:
    resolved = _resolve_all()
    return resolved["cave3"], resolved["cave3_syms"]


def cave_layout() -> dict:
    """Compute exact virtual addresses and sizes for Caves 1, 2, 3, and EXP_LUT."""
    cave1 = build_live_ai_cave1()
    res = _resolve_all()
    core_len = len(res["core"])
    cave2_len = res["render_len"] + res["shim_len"] + res["bypass_len"] + core_len
    cave3_len = len(res["cave3"])

    return {
        "cave1_va": s3.CAVE1_VA,
        "cave1_len": len(cave1),
        "cave1_margin": s3.CAVE2_VA - (s3.CAVE1_VA + len(cave1)),
        "render_va": s3.CAVE2_VA,
        "render_len": res["render_len"],
        "shim_va": res["shim_va"],
        "shim_len": res["shim_len"],
        "bypass_va": res["bypass_va"],
        "bypass_len": res["bypass_len"],
        "core_va": res["core_va"],
        "core_len": core_len,
        "cave2_len": cave2_len,
        "cave2_end_va": s3.CAVE2_VA + cave2_len,
        "cave2_margin": s3.DESC_TABLE_VA - (s3.CAVE2_VA + cave2_len),
        "cave3_va": CAVE3_VA,
        "cave3_len": cave3_len,
        "cave3_end_va": CAVE3_VA + cave3_len,
        "cave3_margin": CAVE3_CODE_LIMIT_VA - (CAVE3_VA + cave3_len),
        "exp_lut_va": EXP_LUT_VA,
        "exp_lut_len": len(EXP_LUT_BYTES),
        "cave2_syms": res["cave2_syms"],
        "cave3_syms": res["cave3_syms"],
    }


def update_shim_address() -> int:
    return cave_layout()["shim_va"]


def build_update_hook() -> bytes:
    return struct.pack("<I", s3.encode_arm_branch(HOOK_UPDATE_VA, update_shim_address(), is_bl=True))


def build_live_ai_cave2() -> bytes:
    """Assemble Cave 2 @ 0x53000e00 (OSD renderer + post-update shim + bypass + DSP core + tracker)."""
    res = _resolve_all()
    render = _build_render_osd(s3.CAVE2_VA, res["bypass_va"])
    shim_va = res["shim_va"]
    bypass_va = res["bypass_va"]
    core_va = res["core_va"]
    core = res["core"]

    wrapper = struct.pack(
        "<7I",
        0xE92D4010,  # push {r4, lr}
        s3.encode_arm_branch(shim_va + 4, s3.UPDATE_FACE_RES_VA, is_bl=True),
        0xE92D500F,  # push {r0-r3, ip, lr}
        0xE1A00004,  # mov  r0, r4 (pass FaceDetectionService*)
        s3.encode_arm_branch(shim_va + 16, core_va, is_bl=True),
        0xE8BD500F,  # pop  {r0-r3, ip, lr}
        0xE8BD8010,  # pop  {r4, pc}
    )
    bypass = struct.pack(
        "<5I",
        0xE1A00006,  # mov r0, r6
        s3.encode_arm_branch(bypass_va + 4, s3.UNLOCK_MUTEX_VA, is_bl=True),
        0xE1A00004,  # mov r0, r4
        s3.encode_arm_branch(bypass_va + 12, s3.UPDATE_FACE_RES_VA, is_bl=True),
        s3.encode_arm_branch(bypass_va + 16, HOOK_UPDATE_VA + 4),
    )

    cave2 = render + wrapper + bypass + core
    if s3.CAVE2_VA + len(cave2) > s3.DESC_TABLE_VA:
        raise ValueError(f"Cave 2 ({len(cave2)} B) exceeds zero space before DESC_TABLE_VA")
    return cave2


def apply_live_ai_patch(payload: bytearray, rtos_start: int, rtos_end: int) -> dict:
    """Apply Step 2D v1.2.19 K-3 III SPResize & Live Posture AI AF patch non-destructively inside RTOS."""
    if not __debug__:
        raise RuntimeError("Stage 3 preflight uses assertions; do not run with python -O")
    update_off = rtos_start + HOOK_UPDATE_VA - s3.GR4_BASE
    if (
        rtos_start != s3.RTOS_PAYLOAD_START
        or rtos_end != 0x1423328
        or not rtos_start <= update_off < update_off + 4 <= rtos_end <= len(payload)
    ):
        raise ValueError("Invalid v1.20 RTOS bounds")
    if payload[update_off : update_off + 4] != EXPECTED_UPDATE:
        raise ValueError("Unexpected stock UpdateFaceDetectionResult call")

    cave3, _ = build_live_ai_cave3()
    cave3_off = rtos_start + CAVE3_VA - s3.GR4_BASE
    lut_off = rtos_start + EXP_LUT_VA - s3.GR4_BASE
    if not rtos_start <= cave3_off < cave3_off + len(cave3) <= lut_off < lut_off + len(EXP_LUT_BYTES) <= rtos_end:
        raise ValueError("Cave 3 or EXP_LUT out of RTOS bounds")
    if payload[cave3_off : cave3_off + len(cave3)] != bytes(len(cave3)):
        raise ValueError("Cave 3 destination is not empty zero space")
    if payload[lut_off : lut_off + len(EXP_LUT_BYTES)] != bytes(len(EXP_LUT_BYTES)):
        raise ValueError("EXP_LUT destination is not empty zero space")

    check = bytearray(payload)
    manifest = s3.apply_stage3_patch(
        check,
        rtos_payload_offset=rtos_start,
        cave1_builder=build_live_ai_cave1,
        cave2_builder=build_live_ai_cave2,
    )
    check[cave3_off : cave3_off + len(cave3)] = cave3
    check[lut_off : lut_off + len(EXP_LUT_BYTES)] = EXP_LUT_BYTES
    check[update_off : update_off + 4] = build_update_hook()

    if payload[:rtos_start] != check[:rtos_start] or payload[rtos_end:] != check[rtos_end:]:
        raise ValueError("Live AI patch writes outside RTOS")

    payload[:] = check
    info = cave_layout()
    manifest.update({
        "stage": "Step 2D v1.2.19: K-3 III SPResize 56x38 Letterbox, Live Posture (+0x7da) & Softmax Probability AI AF (Option B)",
        "post_update_hook": {
            "va": hex(HOOK_UPDATE_VA),
            "offset": hex(update_off),
            "original": EXPECTED_UPDATE.hex(),
            "replacement": build_update_hook().hex(),
        },
        "cave3": {
            "va": hex(CAVE3_VA),
            "offset": hex(cave3_off),
            "length": len(cave3),
            "margin": info["cave3_margin"],
            "symbols": {k: hex(v) for k, v in info["cave3_syms"].items()},
        },
        "exp_lut": {
            "va": hex(EXP_LUT_VA),
            "offset": hex(lut_off),
            "length": len(EXP_LUT_BYTES),
            "hex": EXP_LUT_BYTES.hex(),
        },
        "layout": {
            k: hex(v) if isinstance(v, int) and "va" in k else v
            for k, v in info.items()
            if k not in ("cave2_syms", "cave3_syms")
        },
        "heatmap_out_va": hex(HEATMAP_OUT_VA),
        "heatmap_len": hex(HEATMAP_LEN),
        "work_buf_flush_len": hex(WORK_BUF_FLUSH_LEN),
        "k3_tuning": {
            "orient_live_off": hex(ORIENT_LIVE_OFF),
            "letterbox_rows": [LETTERBOX_ROW_MIN, LETTERBOX_ROW_MAX],
            "ch1_other_max": CH1_OTHER_MAX,
            "ch1_peak_other_max": CH1_PEAK_OTHER_MAX,
            "ch1_min_local": CH1_MIN_LOCAL,
            "local_radius_ch1": LOCAL_RADIUS_CH1,
            "ch2_other_max": CH2_OTHER_MAX,
            "ch2_min_local": CH2_MIN_LOCAL,
            "local_radius_ch2": LOCAL_RADIUS_CH2,
            "acq_confirm_cycles": ACQ_CONFIRM_CYCLES,
            "lost_hold_cycles": LOST_HOLD_CYCLES,
            "head_box_q8": [MIN_HEAD_BOX_Q8, MAX_HEAD_BOX_Q8],
            "body_box_q8": [MIN_BODY_BOX_Q8, MAX_BODY_BOX_Q8],
            "pos_deadband_q8": POS_DEADBAND_Q8,
            "size_deadband_q8": SIZE_DEADBAND_Q8,
        },
    })
    return manifest


if __name__ == "__main__":
    info = cave_layout()
    c1 = build_live_ai_cave1()
    c2 = build_live_ai_cave2()
    c3, syms = build_live_ai_cave3()
    print(f"Cave 1: {len(c1)} B @ {hex(s3.CAVE1_VA)}..{hex(s3.CAVE1_VA + len(c1))} (margin {info['cave1_margin']} B)")
    print(f"Cave 2: {len(c2)} B @ {hex(s3.CAVE2_VA)}..{hex(info['cave2_end_va'])} (margin {info['cave2_margin']} B)")
    print(f"Cave 3: {len(c3)} B @ {hex(CAVE3_VA)}..{hex(info['cave3_end_va'])} (margin {info['cave3_margin']} B before EXP_LUT @ {hex(EXP_LUT_VA)})")
    print("Cave 2 symbols:", {k: hex(v) for k, v in info["cave2_syms"].items()})
    print("Cave 3 symbols:", {k: hex(v) for k, v in syms.items()})

    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    dec1 = list(md.disasm(c1, s3.CAVE1_VA))
    dec2 = list(md.disasm(c2, s3.CAVE2_VA))
    dec3 = list(md.disasm(c3, CAVE3_VA))
    assert len(dec1) * 4 == len(c1), f"Cave 1 decode mismatch: {len(dec1)*4} != {len(c1)}"
    assert len(dec2) * 4 == len(c2), f"Cave 2 decode mismatch: {len(dec2)*4} != {len(c2)}"
    assert len(dec3) * 4 == len(c3), f"Cave 3 decode mismatch: {len(dec3)*4} != {len(c3)}"
    print(f"Capstone clean decode: Cave 1 = {len(dec1)} insns, Cave 2 = {len(dec2)} insns, Cave 3 = {len(dec3)} insns")
