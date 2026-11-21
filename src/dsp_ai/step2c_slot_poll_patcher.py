#!/usr/bin/env python3
"""Step 2C v1.2.13: OSD-commit-first + zero-side-effect [m_dspService + 0x54] slot polling.

Keeps v1.2.11/v1.2.12's verified stock-OSD-first wrapper while replacing the rigid
3-frame blind WaitMessage(0) attempt with:
1. A read-only scan of the 6-slot reply queue at [m_dspService + 0x54]
   (stride 0x2c, bytes +0x00 occupied != 0 and u16 +0x02 == seq) up to MAX_POLL_FRAMES (30),
   calling WaitMessage(svc, seq, 0, &resp) only after the DSP ISR populates the slot.
2. Full 0xee00 D-Cache flush and 0xc400 D-Cache invalidation verifying writes to BOTH
   0x52b02a00 (14x14x64 intermediate backbone) and 0x52b09a00 (3-ch 56x56 segmentation heatmap).
3. 9-grid visual diagnostics separating 30-frame slot timeout (Top-Right) from protocol/service
   errors (Top-Center) and partial DDR writes (Bottom-Center).
"""

import struct
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

import stage3_ai_patcher as s3
import step2c_osd_recovery as osd

SLOT_POLL_COOKIE = 0x53503233  # "SP23"
BUDGET = 4
MAX_POLL_FRAMES = 30
CANARY_VAL = 0x55555555
REPLY_MARK = 0xa5c23d7e

RESP_BUF_VA = s3.TENSOR_MSG_VA + 0x40
BACKBONE_OUT_VA = s3.TENSOR_OUT_VA          # 0x52b02a00
HEATMAP_OUT_VA = s3.TENSOR_IN_VA + 0x9a00   # 0x52b09a00
WORK_BUF_FLUSH_LEN = 0xee00                 # 0x52b00000..0x52b0edff
OUT_INVAL_LEN = 0xc400                      # 0x52b02a00..0x52b0edff

HOOK_UPDATE_VA = 0x538963B8
EXPECTED_UPDATE = bytes.fromhex('8dfdffeb')
RENDER_SITE = s3.CAVE2_VA + 0x7c
EXPECTED_RENDER = bytes.fromhex('0000a0e3')

# Terminal / intermediate states at STATUS_VA (0x52c00100):
# 0=warmup, 1=load_err, 2=send_next, 3=in_flight, 4=done_ok,
# 5=start_err, 6=send_err, 7=no_ddr, 8=slot_timeout,
# 9=wait_err, 10=reply_or_seq_err, 11=svc_err, 12=partial_ddr


def build_slot_poll_cave1() -> bytes:
    return s3.build_cave1_payload(status_cookie=SLOT_POLL_COOKIE)


def update_shim_address() -> int:
    return s3.CAVE2_VA + len(osd.build_osd_recovery_cave())


def build_update_hook() -> bytes:
    return struct.pack('<I', s3.encode_arm_branch(HOOK_UPDATE_VA, update_shim_address(), is_bl=True))


def build_slot_poll_core(core_va: int) -> bytes:
    """Assemble the post-update DSP state machine subroutine starting at core_va."""
    insns = []
    labels = {}
    relocs = []

    def mark(name: str):
        labels[name] = core_va + len(insns) * 4

    def b(target: str, bl: bool = False, cond: int = 0xe):
        relocs.append((len(insns), target, bl, cond))
        insns.append(0)

    def call(va: int):
        insns.append(s3.encode_arm_branch(core_va + len(insns) * 4, va, is_bl=True))

    def ldr32(rd: int, val: int):
        s3.emit_ldr_imm32(insns, rd, val)

    # Prologue (aligned 8B: 8 regs = 32B + 8B scratch = 40B)
    insns.append(0xe92d47f0)  # push {r4, r5, r6, r7, r8, sb, sl, lr}
    insns.append(0xe24dd008)  # sub  sp, sp, #8

    # Require valid boot initialization cookie before reading or mutating STATUS_VA
    ldr32(8, s3.STATUS_VA)
    insns.append(0xe5982018)  # ldr  r2, [r8, #0x18]
    ldr32(3, SLOT_POLL_COOKIE)
    insns.append(0xe1520003)  # cmp  r2, r3
    b("exit_done", cond=0x1)  # bne

    # Dispatch active states only; all terminal states (1, 4, 5..12) fail closed
    insns.append(0xe5987000)  # ldr  r7, [r8, #0]
    insns.append(0xe3570003)  # cmp  r7, #3 (IN_FLIGHT)
    b("do_poll_in_flight", cond=0x0)
    insns.append(0xe3570002)  # cmp  r7, #2 (SEND_NEXT)
    b("do_send_probe", cond=0x0)
    insns.append(0xe3570000)  # cmp  r7, #0 (WARMUP / INIT)
    b("exit_done", cond=0x1)

    # --- State 0: Warmup (20 frames) + One-Shot CreateDspService / LoadDsp / Start ---
    insns.append(0xe5982004)  # ldr  r2, [r8, #4] (boot_ticks)
    insns.append(0xe2822001)  # add  r2, r2, #1
    insns.append(0xe5882004)  # str  r2, [r8, #4]
    insns.append(0xe3520014)  # cmp  r2, #20
    b("exit_done", cond=0x3)  # blo (frames 1..19)

    call(s3.GET_DSP_SVC_VA)
    insns.append(0xe3500000)  # cmp  r0, #0
    b("halt_svc_err", cond=0x0)
    insns.append(0xe5905004)  # ldr  r5, [r0, #4]
    insns.append(0xe3550000)  # cmp  r5, #0
    b("have_svc_init", cond=0x1)

    insns.append(0xe3a01000)  # mov  r1, #0
    call(s3.CREATE_DSP_SVC_VA)
    call(s3.GET_DSP_SVC_VA)
    insns.append(0xe3500000)  # cmp  r0, #0
    b("halt_svc_err", cond=0x0)
    insns.append(0xe5905004)  # ldr  r5, [r0, #4]
    insns.append(0xe3550000)  # cmp  r5, #0
    b("halt_load_err", cond=0x0)

    mark("have_svc_init")
    # Never load a second firmware image onto an already-started service
    insns.append(0xe5952004)  # ldr  r2, [r5, #4]
    insns.append(0xe3120004)  # tst  r2, #4
    b("halt_svc_err", cond=0x1)

    # Try Drive 'C' (SD card) then Drive 'I' (Internal 53 GB flash)
    insns.append(0xe1a00005)  # mov  r0, r5
    insns.append(0xe3a01043)  # mov  r1, #'C'
    insns.append(0xe3a02000)
    insns.append(0xe3a03000)
    call(s3.LOAD_DSP_VA)
    insns.append(0xe3500000)
    b("load_ok", cond=0x1)

    insns.append(0xe1a00005)  # mov  r0, r5
    insns.append(0xe3a01049)  # mov  r1, #'I'
    insns.append(0xe3a02000)
    insns.append(0xe3a03000)
    call(s3.LOAD_DSP_VA)
    insns.append(0xe3500000)
    b("load_ok", cond=0x1)
    b("halt_load_err")

    mark("load_ok")
    insns.append(0xe1a00005)  # mov  r0, r5
    ldr32(1, 0x52100000)
    insns.append(0xe3a02000)  # mov  r2, #0
    call(s3.START_DSP_VA)
    insns.append(0xe5880010)  # last_err = Start() ret
    insns.append(0xe5952004)  # ldr  r2, [r5, #4]
    insns.append(0xe3120004)  # tst  r2, #4 (STARTED)
    b("halt_start_err", cond=0x0)

    insns.append(0xe5885020)  # service_ptr = svc
    insns.append(0xe3a00002)  # state = 2 (SEND_NEXT)
    insns.append(0xe5880000)
    b("exit_done")

    # --- State 2: Fill synthetic input & dual output canaries, flush 0xee00, SendMessage ---
    mark("do_send_probe")
    call(s3.GET_DSP_SVC_VA)
    b("check_started_service", bl=True)

    # Fill TENSOR_IN_VA (0x2a00 B) with 0x80808080
    ldr32(0, s3.TENSOR_IN_VA)
    ldr32(1, 0x80808080)
    insns.append(s3.encode_movw(2, 0x2a00))
    mark("fill_in_loop")
    insns.append(0xe2522004)  # subs r2, r2, #4
    insns.append(0xe7801002)  # str  r1, [r0, r2]
    b("fill_in_loop", cond=0x1)

    # Fill BACKBONE_OUT_VA .. end of HEATMAP_OUT_VA (0xc400 B = 0x7000 + 0x5400) with CANARY_VAL
    ldr32(0, BACKBONE_OUT_VA)
    ldr32(1, CANARY_VAL)
    insns.append(s3.encode_movw(2, OUT_INVAL_LEN))
    mark("fill_out_loop")
    insns.append(0xe2522004)  # subs r2, r2, #4
    insns.append(0xe7801002)  # str  r1, [r0, r2]
    b("fill_out_loop", cond=0x1)

    # Flush full 0xee00 B work_buf (input + backbone + segmentation heatmap) to DDR
    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, s3.TENSOR_IN_VA)
    insns.append(s3.encode_movw(2, WORK_BUF_FLUSH_LEN))
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
    b("halt_send_err", cond=0x1)

    insns.append(0xe3a00000)
    insns.append(0xe588000c)  # poll_frames = 0
    insns.append(0xe3a00003)  # state = 3 (IN_FLIGHT)
    insns.append(0xe5880000)
    b("exit_done")

    # --- State 3: Zero-side-effect read-only poll of [svc + 0x54] reply slots ---
    mark("do_poll_in_flight")
    call(s3.GET_DSP_SVC_VA)
    b("check_started_service", bl=True)

    insns.append(0xe2886008)  # r6 = STATUS_VA + 8 (&seq)
    insns.append(0xe1d610b0)  # ldrh r1, [r6]
    insns.append(0xe3510005)  # cmp  r1, #5 (stock seq domain 0..5)
    b("halt_reply_or_seq_err", cond=0x8)  # bhi

    insns.append(0xe5950054)  # ldr  r0, [r5, #0x54] (6-slot reply table)
    insns.append(0xe3500000)  # cmp  r0, #0
    b("halt_svc_err", cond=0x0)

    # Read-only scan of 6 x 0x2c = 0x108 bytes at [svc + 0x54]
    insns.append(0xe3a02000)  # mov  r2, #0
    mark("scan_slot")
    insns.append(0xe7d03002)  # ldrb r3, [r0, r2] (occupied)
    insns.append(0xe3530000)  # cmp  r3, #0
    b("scan_next", cond=0x0)  # beq
    insns.append(0xe0807002)  # add  r7, r0, r2
    insns.append(0xe1d730b2)  # ldrh r3, [r7, #2] (slot seq)
    insns.append(0xe1530001)  # cmp  r3, r1
    b("slot_ready", cond=0x0)  # beq
    mark("scan_next")
    insns.append(0xe282202c)  # add  r2, r2, #0x2c
    insns.append(0xe3520f42)  # cmp  r2, #0x108
    b("scan_slot", cond=0x1)  # bne

    # Slot not ready yet: advance poll_frames without touching WaitMessage
    insns.append(0xe598200c)  # ldr  r2, [r8, #0x0c]
    insns.append(0xe2822001)  # add  r2, r2, #1
    insns.append(0xe588200c)  # str  r2, [r8, #0x0c]
    insns.append(0xe3520000 | MAX_POLL_FRAMES)  # cmp r2, #30
    b("exit_done", cond=0x3)  # blo (wait another frame)
    b("halt_slot_timeout")

    # Slot is occupied by ISR for matching seq: dequeue safely via WaitMessage(..., 0)
    mark("slot_ready")
    ldr32(3, RESP_BUF_VA + 4)
    ldr32(2, REPLY_MARK)
    insns.append(0xe3a00024)  # mov  r0, #0x24
    mark("prime_reply_marker")
    insns.append(0xe2500004)  # subs r0, r0, #4
    insns.append(0xe7832000)  # str  r2, [r3, r0]
    b("prime_reply_marker", cond=0x1)

    insns.append(0xe1a00005)  # mov  r0, r5 (svc)
    insns.append(0xe1d610b0)  # ldrh r1, [r6] (seq)
    insns.append(0xe3a02000)  # mov  r2, #0 (TMO_POL)
    ldr32(3, RESP_BUF_VA)
    call(s3.WAIT_MSG_VA)
    insns.append(0xe5880010)  # last_err = WaitMessage ret
    insns.append(0xe3500000)  # cmp  r0, #0
    b("halt_wait_err", cond=0x1)

    # Confirm 0x5398e718 actually copied reply words into RESP_BUF_VA + 4..+0x27
    ldr32(0, RESP_BUF_VA + 4)
    ldr32(1, REPLY_MARK)
    insns.append(0xe3a02024)
    mark("check_reply_marker")
    insns.append(0xe2522004)  # subs r2, r2, #4
    insns.append(0xe7903002)  # ldr  r3, [r0, r2]
    insns.append(0xe1530001)  # cmp  r3, r1
    b("reply_copied", cond=0x1)
    insns.append(0xe3520000)
    b("check_reply_marker", cond=0x1)
    b("halt_reply_or_seq_err")

    mark("reply_copied")
    # Invalidate ARM D-Cache over both 0x52b02a00 (backbone) and 0x52b09a00 (heatmap)
    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, BACKBONE_OUT_VA)
    insns.append(s3.encode_movw(2, OUT_INVAL_LEN))
    call(s3.INVAL_DCACHE_VA)

    ldr32(0, BACKBONE_OUT_VA)
    ldr32(1, CANARY_VAL)
    b("count_canary_diff", bl=True)
    insns.append(0xe1a04000)  # mov  r4, r0 (diff_bb)

    ldr32(0, HEATMAP_OUT_VA)
    b("count_canary_diff", bl=True)
    insns.append(0xe1a07000)  # mov  r7, r0 (diff_hm)

    # Pack canary_diff_cnt = (diff_hm << 16) | diff_bb
    insns.append(0xe1843807)  # orr  r3, r4, r7, lsl #16
    insns.append(0xe5883014)  # str  r3, [r8, #0x14]

    insns.append(0xe1843007)  # orr  r3, r4, r7
    insns.append(0xe3530000)  # cmp  r3, #0
    b("halt_no_ddr", cond=0x0)
    insns.append(0xe3540000)  # cmp  r4, #0
    b("halt_partial_ddr", cond=0x0)
    insns.append(0xe3570000)  # cmp  r7, #0
    b("halt_partial_ddr", cond=0x0)

    # Both backbone and heatmap regions updated by DSP!
    insns.append(0xe598201c)  # ldr  r2, [r8, #0x1c] (completed)
    insns.append(0xe2822001)  # add  r2, r2, #1
    insns.append(0xe588201c)  # str  r2, [r8, #0x1c]
    insns.append(0xe3520000 | BUDGET)  # cmp r2, #4
    insns.append(0x33a00002)  # movlo r0, #2 (SEND_NEXT)
    insns.append(0x23a00004)  # movhs r0, #4 (DONE_OK)
    insns.append(0xe5880000)  # str   r0, [r8, #0]
    b("exit_done")

    # --- Local helper: count words != r1 in [r0 .. r0 + 0x100) ---
    mark("count_canary_diff")
    insns.append(0xe3a0c000)  # mov  ip, #0
    insns.append(0xe3a02000)  # mov  r2, #0
    mark("chk_diff_loop")
    insns.append(0xe7903002)  # ldr  r3, [r0, r2]
    insns.append(0xe1530001)  # cmp  r3, r1
    insns.append(0x128cc001)  # addne ip, ip, #1
    insns.append(0xe2822004)  # add  r2, r2, #4
    insns.append(0xe3520c01)  # cmp  r2, #0x100
    b("chk_diff_loop", cond=0xb)  # blt
    insns.append(0xe1a0000c)  # mov  r0, ip
    insns.append(0xe12fff1e)  # bx   lr

    # --- Verify factory != NULL, svc != NULL, svc unchanged, STARTED bit set ---
    mark("check_started_service")
    insns.append(0xe3500000)
    b("halt_svc_err", cond=0x0)
    insns.extend((0xe5905004, 0xe3550000))  # ldr r5, [r0, #4]; cmp r5, #0
    b("halt_svc_err", cond=0x0)
    insns.extend((0xe5982020, 0xe1550002))  # ldr r2, [r8, #0x20]; cmp r5, r2
    b("halt_svc_err", cond=0x1)
    insns.extend((0xe5952004, 0xe3120004))  # ldr r2, [r5, #4]; tst r2, #4
    b("halt_svc_err", cond=0x0)
    insns.append(0xe12fff1e)  # bx lr

    # --- Terminal state latches (all share single exit) ---
    for name, code in (
        ("halt_load_err", 1),
        ("halt_start_err", 5),
        ("halt_send_err", 6),
        ("halt_no_ddr", 7),
        ("halt_slot_timeout", 8),
        ("halt_wait_err", 9),
        ("halt_reply_or_seq_err", 10),
        ("halt_svc_err", 11),
        ("halt_partial_ddr", 12),
    ):
        mark(name)
        insns.append(0xe3a00000 | code)  # mov r0, #code
        insns.append(0xe5880000)         # str r0, [r8, #0]
        if name != "halt_partial_ddr":
            b("exit_done")

    mark("exit_done")
    insns.append(0xe28dd008)  # add sp, sp, #8
    insns.append(0xe8bd87f0)  # pop {r4-r10, pc}

    for idx, target, is_bl, cond in relocs:
        insns[idx] = s3.encode_arm_branch(core_va + idx * 4, labels[target], is_bl=is_bl, cond=cond)

    return struct.pack(f"<{len(insns)}I", *insns)


def build_position_helper(helper_va: int) -> bytes:
    """Display-only 9-grid position mapper reading latched STATUS_VA state.

    Called from RENDER_SITE (mov r0, #0 replaced with BL helper_va) with FD mutex
    held and r7/r8 centered Q8 coordinates. Replays mov r0, #0 and returns.
    """
    ops, labels, branches = [], {}, []

    def mark(name: str):
        labels[name] = len(ops)

    def b(name: str, cond: int = 0xe):
        branches.append((len(ops), name, cond))
        ops.append(0)

    s3.emit_ldr_imm32(ops, 3, s3.STATUS_VA)
    s3.emit_ldr_imm32(ops, 1, SLOT_POLL_COOKIE)
    ops.extend((0xe5932018, 0xe1520001))  # ldr r2, [r3, #0x18]; cmp r2, r1
    b("finish", 0x1)
    ops.append(0xe5930000)  # ldr r0, [r3, #0] (state)

    for state, label in (
        (1, "load_error"),
        (4, "done"),
        (5, "start_error"),
        (6, "send_error"),
        (7, "no_ddr"),
        (8, "slot_timeout"),
        (9, "proto_error"),
        (10, "proto_error"),
        (11, "proto_error"),
        (12, "partial_ddr"),
    ):
        ops.append(0xe3500000 | state)
        b(label, 0x0)
    b("finish")

    left, right = 0xe04770a7, 0xe08770a7  # sub/add r7, r7, r7, lsr #1
    up, down = 0xe04880a8, 0xe08880a8    # sub/add r8, r8, r8, lsr #1
    for label, x, y in (
        ("load_error", left, up),       # Top-Left
        ("start_error", left, None),    # Mid-Left
        ("send_error", left, down),     # Bottom-Left
        ("no_ddr", right, down),        # Bottom-Right
        ("slot_timeout", right, up),    # Top-Right
        ("proto_error", None, up),      # Top-Center
        ("partial_ddr", None, down),    # Bottom-Center
    ):
        mark(label)
        if x is not None:
            ops.append(x)
        if y is not None:
            ops.append(y)
        b("finish")

    mark("done")
    ops.extend((0xe593201c, 0xe3520000 | BUDGET))
    b("finish", 0x1)  # bne finish unless completed == BUDGET
    ops.append(right)  # Mid-Right

    mark("finish")
    ops.extend((0xe3a00000, 0xe12fff1e))  # mov r0, #0; bx lr
    for i, label, cond in branches:
        ops[i] = s3.encode_arm_branch(helper_va + i * 4, helper_va + labels[label] * 4, cond=cond)
    return struct.pack(f"<{len(ops)}I", *ops)


def build_slot_poll_cave2() -> bytes:
    """Assemble complete Cave 2 (OSD recovery + post-update shim + slot-poll core + 9-grid helper)."""
    render_len = len(osd.build_osd_recovery_cave())
    if render_len != 280:
        raise ValueError(f"Unexpected OSD recovery cave length: {render_len}")

    shim_va = s3.CAVE2_VA + render_len
    core_va = shim_va + 24
    core = build_slot_poll_core(core_va)
    bypass_va = core_va + len(core)
    helper_va = bypass_va + 20
    helper = build_position_helper(helper_va)

    render_raw = osd.build_osd_recovery_cave(invalid_resume_va=bypass_va)
    render_off = RENDER_SITE - s3.CAVE2_VA
    if len(render_raw) != render_len or render_raw[render_off:render_off + 4] != EXPECTED_RENDER:
        raise ValueError("OSD recovery cave layout mismatch")
    render_branch = struct.pack('<I', s3.encode_arm_branch(RENDER_SITE, helper_va, is_bl=True))
    render_patched = render_raw[:render_off] + render_branch + render_raw[render_off + 4:]

    wrapper = (
        0xe92d4010,  # push {r4, lr}
        s3.encode_arm_branch(shim_va + 4, s3.UPDATE_FACE_RES_VA, is_bl=True),
        0xe92d500f,  # push {r0-r3, ip, lr}
        s3.encode_arm_branch(shim_va + 12, core_va, is_bl=True),
        0xe8bd500f,  # pop {r0-r3, ip, lr}
        0xe8bd8010,  # pop {r4, pc}
    )
    invalid_path = (
        0xe1a00006,  # mov r0, r6
        s3.encode_arm_branch(bypass_va + 4, s3.UNLOCK_MUTEX_VA, is_bl=True),
        0xe1a00004,  # mov r0, r4
        s3.encode_arm_branch(bypass_va + 12, s3.UPDATE_FACE_RES_VA, is_bl=True),
        s3.encode_arm_branch(bypass_va + 16, HOOK_UPDATE_VA + 4),
    )

    cave = render_patched + struct.pack('<6I', *wrapper) + core + struct.pack('<5I', *invalid_path) + helper
    if s3.CAVE2_VA + len(cave) > s3.DESC_TABLE_VA:
        raise ValueError(f"Step 2C slot-poll cave ({len(cave)} B) exceeds zero space before DESC_TABLE_VA")
    return cave


def cave2_layout() -> dict:
    render_len = len(osd.build_osd_recovery_cave())
    shim_va = s3.CAVE2_VA + render_len
    core_va = shim_va + 24
    core_len = len(build_slot_poll_core(core_va))
    bypass_va = core_va + core_len
    helper_va = bypass_va + 20
    helper_len = len(build_position_helper(helper_va))
    total_len = render_len + 24 + core_len + 20 + helper_len
    return {
        "render_va": s3.CAVE2_VA,
        "render_len": render_len,
        "shim_va": shim_va,
        "core_va": core_va,
        "core_len": core_len,
        "bypass_va": bypass_va,
        "helper_va": helper_va,
        "helper_len": helper_len,
        "total_len": total_len,
        "end_va": s3.CAVE2_VA + total_len,
        "margin_to_desc": s3.DESC_TABLE_VA - (s3.CAVE2_VA + total_len),
    }


def apply_slot_poll_experiment(payload: bytearray, rtos_start: int, rtos_end: int) -> dict:
    """Apply Step 2C v1.2.13 slot-polling experiment non-destructively inside RTOS bounds."""
    if not __debug__:
        raise RuntimeError('Stage 3 preflight uses assertions; do not run with python -O')
    off = rtos_start + HOOK_UPDATE_VA - s3.GR4_BASE
    if (rtos_start != s3.RTOS_PAYLOAD_START or rtos_end != 0x1423328
            or not rtos_start <= off < off + 4 <= rtos_end <= len(payload)):
        raise ValueError('Invalid v1.20 RTOS bounds')
    if payload[off:off + 4] != EXPECTED_UPDATE:
        raise ValueError('Unexpected stock UpdateFaceDetectionResult call')

    check = bytearray(payload)
    manifest = s3.apply_stage3_patch(
        check,
        rtos_payload_offset=rtos_start,
        cave1_builder=build_slot_poll_cave1,
        cave2_builder=build_slot_poll_cave2,
    )
    check[off:off + 4] = build_update_hook()
    if payload[:rtos_start] != check[:rtos_start] or payload[rtos_end:] != check[rtos_end:]:
        raise ValueError('Slot-poll experiment writes outside RTOS')

    payload[:] = check
    manifest.update({
        'stage': 'Step 2C v1.2.13: OSD-commit-first + read-only [svc+0x54] slot-poll + dual heatmap verification',
        'post_update_hook': {
            'va': hex(HOOK_UPDATE_VA),
            'offset': hex(off),
            'original': EXPECTED_UPDATE.hex(),
            'replacement': build_update_hook().hex(),
        },
        'layout': {k: hex(v) if 'va' in k else v for k, v in cave2_layout().items()},
        'request_budget': BUDGET,
        'max_poll_frames': MAX_POLL_FRAMES,
        'work_buf_flush_len': hex(WORK_BUF_FLUSH_LEN),
        'out_inval_len': hex(OUT_INVAL_LEN),
        'backbone_out_va': hex(BACKBONE_OUT_VA),
        'heatmap_out_va': hex(HEATMAP_OUT_VA),
    })
    return manifest


if __name__ == '__main__':
    info = cave2_layout()
    c1 = build_slot_poll_cave1()
    c2 = build_slot_poll_cave2()
    print(f"Cave 1: {len(c1)} B @ {hex(s3.CAVE1_VA)}..{hex(s3.CAVE1_VA + len(c1))}")
    print(f"Cave 2: {len(c2)} B @ {hex(s3.CAVE2_VA)}..{hex(info['end_va'])} (margin {info['margin_to_desc']} B to DESC_TABLE_VA)")
    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    dec1 = list(md.disasm(c1, s3.CAVE1_VA))
    dec2 = list(md.disasm(c2, s3.CAVE2_VA))
    assert len(dec1) * 4 == len(c1), f"Cave 1 decode mismatch: {len(dec1)*4} != {len(c1)}"
    assert len(dec2) * 4 == len(c2), f"Cave 2 decode mismatch: {len(dec2)*4} != {len(c2)}"
    print(f"Capstone clean decode: Cave 1 = {len(dec1)} insns, Cave 2 = {len(dec2)} insns")
