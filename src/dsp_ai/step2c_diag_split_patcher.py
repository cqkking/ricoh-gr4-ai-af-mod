#!/usr/bin/env python3
"""Step 2C v1.2.14: 0x1f cache-flush servicing + timeout post-mortem diagnostic split.

Builds on v1.2.13's OSD-commit-first wrapper and read-only [m_dspService + 0x54] slot poll:
1. Services DSP msg_id == 0x1f (DspSF bit 0x10000 + [svc + 0x58] addr/size) during
   in-flight polling without calling WaitMessage(..., 0) prematurely.
2. When 30-frame polling still expires, runs a non-destructive post-mortem classifier:
   - completed >= 1                -> state 13 (Bottom-Center)
   - DDR sentinel modified         -> state 15 (Bottom-Right, alongside no/partial DDR)
   - any [svc+0x54] slot or DspSF  -> state 14 (Top-Center, alongside proto/svc errors)
   - completely silent DSP         -> state  8 (Top-Right)
"""

import struct
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

import stage3_ai_patcher as s3
import step2c_osd_recovery as osd

DIAG_SPLIT_COOKIE = 0x53503234  # "SP24"
BUDGET = 4
MAX_POLL_FRAMES = 30
CANARY_VAL = 0x55555555
REPLY_MARK = 0xa5c23d7e

RESP_BUF_VA = s3.TENSOR_MSG_VA + 0x40
BACKBONE_OUT_VA = s3.TENSOR_OUT_VA          # 0x52b02a00
HEATMAP_OUT_VA = s3.TENSOR_IN_VA + 0x9a00   # 0x52b09a00
WORK_BUF_FLUSH_LEN = 0xee00                 # 0x52b00000..0x52b0edff
OUT_INVAL_LEN = 0xc400                      # 0x52b02a00..0x52b0edff

OS_EF_WAIT_VA = 0x539030a0   # OsEventFlag::Wait(ef, waiptn, &flgptn, timeout=0)
OS_EF_CLEAR_VA = 0x53902fe8  # OsEventFlag::Clear(ef, clrptn)

HOOK_UPDATE_VA = 0x538963B8
EXPECTED_UPDATE = bytes.fromhex('8dfdffeb')
RENDER_SITE = s3.CAVE2_VA + 0x7c
EXPECTED_RENDER = bytes.fromhex('0000a0e3')

# State encoding at STATUS_VA + 0x00:
#  0 = warmup / init
#  1 = load_err (Top-Left)
#  2 = send_next
#  3 = in_flight
#  4 = done_ok (Mid-Right when completed == 4)
#  5 = start_err (Mid-Left)
#  6 = send_err (Bottom-Left)
#  7 = post-reply no_ddr (Bottom-Right)
#  8 = silent_timeout (Top-Right: completed==0, DDR untouched, slots empty, DspSF==0)
#  9 = wait_err (Top-Center)
# 10 = reply_or_seq_err (Top-Center)
# 11 = svc_err (Top-Center)
# 12 = post-reply partial_ddr (Bottom-Right)
# 13 = timeout_after_pass (Bottom-Center: completed >= 1 before later request stalled)
# 14 = timeout_activity (Top-Center: completed==0, DDR untouched, wrong-seq slot or DspSF!=0)
# 15 = timeout_ddr_written (Bottom-Right: completed==0, DDR written without matching reply)


def build_diag_split_cave1() -> bytes:
    return s3.build_cave1_payload(status_cookie=DIAG_SPLIT_COOKIE)


def update_shim_address() -> int:
    return s3.CAVE2_VA + len(osd.build_osd_recovery_cave())


def build_update_hook() -> bytes:
    return struct.pack('<I', s3.encode_arm_branch(HOOK_UPDATE_VA, update_shim_address(), is_bl=True))


def build_diag_split_core(core_va: int) -> bytes:
    """Assemble the v1.2.14 post-update DSP state machine starting at core_va."""
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

    # Require valid boot initialization cookie before touching STATUS_VA
    ldr32(8, s3.STATUS_VA)
    insns.append(0xe5982018)  # ldr  r2, [r8, #0x18]
    ldr32(3, DIAG_SPLIT_COOKIE)
    insns.append(0xe1520003)  # cmp  r2, r3
    b("exit_done", cond=0x1)  # bne

    # Dispatch active states only; all terminal states fail closed
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
    b("exit_done", cond=0x3)  # blo

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
    insns.append(0xe5952004)  # ldr  r2, [r5, #4]
    insns.append(0xe3120004)  # tst  r2, #4
    b("halt_svc_err", cond=0x1)

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
    insns.append(0xe3120004)  # tst  r2, #4
    b("halt_start_err", cond=0x0)

    insns.append(0xe5885020)  # service_ptr = svc
    insns.append(0xe3a00002)  # state = 2 (SEND_NEXT)
    insns.append(0xe5880000)
    b("exit_done")

    # --- State 2: Fill synthetic input & dual output canaries, flush 0xee00, SendMessage ---
    mark("do_send_probe")
    call(s3.GET_DSP_SVC_VA)
    b("check_started_service", bl=True)

    ldr32(0, s3.TENSOR_IN_VA)
    ldr32(1, 0x80808080)
    insns.append(s3.encode_movw(2, 0x2a00))
    mark("fill_in_loop")
    insns.append(0xe2522004)
    insns.append(0xe7801002)
    b("fill_in_loop", cond=0x1)

    ldr32(0, BACKBONE_OUT_VA)
    ldr32(1, CANARY_VAL)
    insns.append(s3.encode_movw(2, OUT_INVAL_LEN))
    mark("fill_out_loop")
    insns.append(0xe2522004)
    insns.append(0xe7801002)
    b("fill_out_loop", cond=0x1)

    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, s3.TENSOR_IN_VA)
    insns.append(s3.encode_movw(2, WORK_BUF_FLUSH_LEN))
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

    # --- State 3: Service 0x1f cache-flush + read-only [svc+0x54] slot poll ---
    mark("do_poll_in_flight")
    call(s3.GET_DSP_SVC_VA)
    b("check_started_service", bl=True)

    insns.append(0xe2886008)  # r6 = STATUS_VA + 8 (&seq)
    insns.append(0xe1d610b0)  # ldrh r1, [r6]
    insns.append(0xe3510005)  # cmp  r1, #5
    b("halt_reply_or_seq_err", cond=0x8)

    # Check read-only if DSP ISR raised msg_id == 0x1f (DspSF bit 0x10000 + [svc+0x58])
    insns.append(0xe3a00000)  # mov  r0, #0
    insns.append(0xe58d0000)  # str  r0, [sp]
    insns.append(0xe285000c)  # add  r0, r5, #0xc (&DspSF)
    insns.append(0xe3a01801)  # mov  r1, #0x10000
    insns.append(0xe1a0200d)  # mov  r2, sp
    insns.append(0xe3a03000)  # mov  r3, #0 (TMO_POL)
    call(OS_EF_WAIT_VA)
    insns.append(0xe3500000)  # cmp  r0, #0
    b("skip_flush1f", cond=0x0)

    insns.append(0xe5953058)  # ldr  r3, [r5, #0x58]
    insns.append(0xe3530000)  # cmp  r3, #0
    b("clear_1f", cond=0x0)
    insns.append(0xe5932008)  # ldr  r2, [r3, #8] (size)
    insns.append(0xe3520000)  # cmp  r2, #0
    b("clear_1f", cond=0x0)
    insns.append(0xe1a04003)  # mov  r4, r3
    call(s3.GET_CACHE_MGR_VA)
    insns.append(0xe5941004)  # ldr  r1, [r4, #4] (addr)
    insns.append(0xe5942008)  # ldr  r2, [r4, #8] (size)
    call(s3.FLUSH_DCACHE_VA)
    insns.append(0xe3a00000)  # mov  r0, #0
    insns.append(0xe5840004)  # str  r0, [r4, #4]
    insns.append(0xe5840008)  # str  r0, [r4, #8]

    mark("clear_1f")
    insns.append(0xe285000c)  # add  r0, r5, #0xc
    insns.append(0xe3a01801)  # mov  r1, #0x10000
    call(OS_EF_CLEAR_VA)

    mark("skip_flush1f")
    insns.append(0xe1d610b0)  # ldrh r1, [r6] (reload expected seq)
    insns.append(0xe5950054)  # ldr  r0, [r5, #0x54]
    insns.append(0xe3500000)  # cmp  r0, #0
    b("halt_svc_err", cond=0x0)

    # Scan 6 x 0x2c = 0x108 bytes at [svc + 0x54] for matching reply
    insns.append(0xe3a02000)  # mov  r2, #0
    mark("scan_slot")
    insns.append(0xe7d03002)  # ldrb r3, [r0, r2]
    insns.append(0xe3530000)  # cmp  r3, #0
    b("scan_next", cond=0x0)
    insns.append(0xe0807002)  # add  r7, r0, r2
    insns.append(0xe1d730b2)  # ldrh r3, [r7, #2]
    insns.append(0xe1530001)  # cmp  r3, r1
    b("slot_ready", cond=0x0)
    mark("scan_next")
    insns.append(0xe282202c)  # add  r2, r2, #0x2c
    insns.append(0xe3520f42)  # cmp  r2, #0x108
    b("scan_slot", cond=0x1)

    # Not ready yet: advance poll_frames or run post-mortem timeout classifier
    insns.append(0xe598200c)  # ldr  r2, [r8, #0x0c]
    insns.append(0xe2822001)  # add  r2, r2, #1
    insns.append(0xe588200c)  # str  r2, [r8, #0x0c]
    insns.append(0xe3520000 | MAX_POLL_FRAMES)  # cmp r2, #30
    b("exit_done", cond=0x3)  # blo

    # --- Post-mortem classifier after 30 frames without matching reply ---
    insns.append(0xe598201c)  # ldr  r2, [r8, #0x1c] (completed)
    insns.append(0xe3520000)  # cmp  r2, #0
    b("halt_timeout_after_pass", cond=0x1)  # bne (at least 1 full tx already passed!)

    b("scan_ddr_canaries", bl=True)
    insns.append(0xe1843007)  # orr  r3, r4, r7
    insns.append(0xe3530000)  # cmp  r3, #0
    b("halt_timeout_ddr_written", cond=0x1)  # bne (DSP wrote DDR without matching reply)

    # Check if ANY of the 6 slots in [svc+0x54] has occupied != 0 (wrong seq)
    insns.append(0xe5950054)  # ldr  r0, [r5, #0x54]
    insns.append(0xe3a02000)  # mov  r2, #0
    mark("chk_any_slot")
    insns.append(0xe7d03002)  # ldrb r3, [r0, r2]
    insns.append(0xe3530000)  # cmp  r3, #0
    b("halt_timeout_activity", cond=0x1)  # bne
    insns.append(0xe282202c)  # add  r2, r2, #0x2c
    insns.append(0xe3520f42)  # cmp  r2, #0x108
    b("chk_any_slot", cond=0x1)

    # Check read-only if ANY DspSF event flag bit is non-zero
    insns.append(0xe3a00000)  # mov  r0, #0
    insns.append(0xe58d0000)  # str  r0, [sp]
    insns.append(0xe285000c)  # add  r0, r5, #0xc
    insns.append(0xe3e01000)  # mvn  r1, #0 (0xffffffff)
    insns.append(0xe1a0200d)  # mov  r2, sp
    insns.append(0xe3a03000)  # mov  r3, #0
    call(OS_EF_WAIT_VA)
    insns.append(0xe59d2000)  # ldr  r2, [sp]
    insns.append(0xe5882010)  # str  r2, [r8, #0x10] (record observed DspSF pattern)
    insns.append(0xe3500000)  # cmp  r0, #0
    b("halt_timeout_activity", cond=0x1)  # bne
    b("halt_slot_timeout")    # completely silent DSP (state 8 -> Top-Right)

    # --- Matching reply slot ready: dequeue via WaitMessage(0) ---
    mark("slot_ready")
    ldr32(3, RESP_BUF_VA + 4)
    ldr32(2, REPLY_MARK)
    insns.append(0xe3a00024)
    mark("prime_reply_marker")
    insns.append(0xe2500004)
    insns.append(0xe7832000)
    b("prime_reply_marker", cond=0x1)

    insns.append(0xe1a00005)  # mov  r0, r5
    insns.append(0xe1d610b0)  # ldrh r1, [r6]
    insns.append(0xe3a02000)  # mov  r2, #0
    ldr32(3, RESP_BUF_VA)
    call(s3.WAIT_MSG_VA)
    insns.append(0xe5880010)  # last_err = WaitMessage ret
    insns.append(0xe3500000)
    b("halt_wait_err", cond=0x1)

    ldr32(0, RESP_BUF_VA + 4)
    ldr32(1, REPLY_MARK)
    insns.append(0xe3a02024)
    mark("check_reply_marker")
    insns.append(0xe2522004)
    insns.append(0xe7903002)
    insns.append(0xe1530001)
    b("reply_copied", cond=0x1)
    insns.append(0xe3520000)
    b("check_reply_marker", cond=0x1)
    b("halt_reply_or_seq_err")

    mark("reply_copied")
    b("scan_ddr_canaries", bl=True)

    insns.append(0xe1843007)  # orr  r3, r4, r7
    insns.append(0xe3530000)  # cmp  r3, #0
    b("halt_no_ddr", cond=0x0)
    insns.append(0xe3540000)  # cmp  r4, #0
    b("halt_partial_ddr", cond=0x0)
    insns.append(0xe3570000)  # cmp  r7, #0
    b("halt_partial_ddr", cond=0x0)

    insns.append(0xe598201c)  # ldr  r2, [r8, #0x1c]
    insns.append(0xe2822001)  # add  r2, r2, #1
    insns.append(0xe588201c)  # str  r2, [r8, #0x1c]
    insns.append(0xe3520000 | BUDGET)
    insns.append(0x33a00002)  # movlo r0, #2 (SEND_NEXT)
    insns.append(0x23a00004)  # movhs r0, #4 (DONE_OK)
    insns.append(0xe5880000)
    b("exit_done")

    # --- Shared helper: invalidate 0xc400 B & count canary diffs -> r4=bb, r7=hm ---
    mark("scan_ddr_canaries")
    insns.append(0xe92d5000)  # push {ip, lr} (8-byte SP aligned)
    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, BACKBONE_OUT_VA)
    insns.append(s3.encode_movw(2, OUT_INVAL_LEN))
    call(s3.INVAL_DCACHE_VA)

    ldr32(0, BACKBONE_OUT_VA)
    ldr32(1, CANARY_VAL)
    b("count_canary_diff", bl=True)
    insns.append(0xe1a04000)  # mov r4, r0

    ldr32(0, HEATMAP_OUT_VA)
    b("count_canary_diff", bl=True)
    insns.append(0xe1a07000)  # mov r7, r0

    insns.append(0xe1843807)  # orr r3, r4, r7, lsl #16
    insns.append(0xe5883014)  # str r3, [r8, #0x14]
    insns.append(0xe8bd9000)  # pop {ip, pc}

    # --- Local helper: count words != r1 in [r0 .. r0 + 0x100) ---
    mark("count_canary_diff")
    insns.append(0xe3a0c000)
    insns.append(0xe3a02000)
    mark("chk_diff_loop")
    insns.append(0xe7903002)
    insns.append(0xe1530001)
    insns.append(0x128cc001)
    insns.append(0xe2822004)
    insns.append(0xe3520c01)
    b("chk_diff_loop", cond=0xb)
    insns.append(0xe1a0000c)
    insns.append(0xe12fff1e)

    # --- Verify factory != NULL, svc != NULL, svc unchanged, STARTED bit set ---
    mark("check_started_service")
    insns.append(0xe3500000)
    b("halt_svc_err", cond=0x0)
    insns.extend((0xe5905004, 0xe3550000))
    b("halt_svc_err", cond=0x0)
    insns.extend((0xe5982020, 0xe1550002))
    b("halt_svc_err", cond=0x1)
    insns.extend((0xe5952004, 0xe3120004))
    b("halt_svc_err", cond=0x0)
    insns.append(0xe12fff1e)

    halt_codes = (
        ("halt_load_err", 1),
        ("halt_start_err", 5),
        ("halt_send_err", 6),
        ("halt_no_ddr", 7),
        ("halt_slot_timeout", 8),
        ("halt_wait_err", 9),
        ("halt_reply_or_seq_err", 10),
        ("halt_svc_err", 11),
        ("halt_partial_ddr", 12),
        ("halt_timeout_after_pass", 13),
        ("halt_timeout_activity", 14),
        ("halt_timeout_ddr_written", 15),
    )
    for idx, (name, code) in enumerate(halt_codes):
        mark(name)
        insns.append(0xe3a00000 | code)
        insns.append(0xe5880000)
        if idx < len(halt_codes) - 1:
            b("exit_done")

    mark("exit_done")
    insns.append(0xe28dd008)
    insns.append(0xe8bd87f0)

    for idx, target, is_bl, cond in relocs:
        insns[idx] = s3.encode_arm_branch(core_va + idx * 4, labels[target], is_bl=is_bl, cond=cond)

    return struct.pack(f"<{len(insns)}I", *insns)


def build_position_helper(helper_va: int) -> bytes:
    """Display-only 9-grid position mapper reading latched STATUS_VA state."""
    ops, labels, branches = [], {}, []

    def mark(name: str):
        labels[name] = len(ops)

    def b(name: str, cond: int = 0xe):
        branches.append((len(ops), name, cond))
        ops.append(0)

    s3.emit_ldr_imm32(ops, 3, s3.STATUS_VA)
    s3.emit_ldr_imm32(ops, 1, DIAG_SPLIT_COOKIE)
    ops.extend((0xe5932018, 0xe1520001))  # cookie == DIAG_SPLIT_COOKIE
    b("finish", 0x1)
    ops.append(0xe5930000)  # ldr r0, [r3, #0] (state)

    for state, label in (
        (1, "load_error"),
        (4, "done"),
        (5, "start_error"),
        (6, "send_error"),
        (7, "ddr_diag"),
        (8, "silent_timeout"),
        (9, "activity_or_proto"),
        (10, "activity_or_proto"),
        (11, "activity_or_proto"),
        (12, "ddr_diag"),
        (13, "timeout_after_pass"),
        (14, "activity_or_proto"),
        (15, "ddr_diag"),
    ):
        ops.append(0xe3500000 | state)
        b(label, 0x0)
    b("finish")

    left, right = 0xe04770a7, 0xe08770a7
    up, down = 0xe04880a8, 0xe08880a8
    for label, x, y in (
        ("load_error", left, up),           # Top-Left
        ("start_error", left, None),        # Mid-Left
        ("send_error", left, down),         # Bottom-Left
        ("ddr_diag", right, down),          # Bottom-Right
        ("silent_timeout", right, up),      # Top-Right
        ("activity_or_proto", None, up),    # Top-Center
        ("timeout_after_pass", None, down), # Bottom-Center
    ):
        mark(label)
        if x is not None:
            ops.append(x)
        if y is not None:
            ops.append(y)
        b("finish")

    mark("done")
    ops.extend((0xe593201c, 0xe3520000 | BUDGET))
    b("finish", 0x1)
    ops.append(right)  # Mid-Right

    mark("finish")
    ops.extend((0xe3a00000, 0xe12fff1e))  # mov r0, #0; bx lr
    for i, label, cond in branches:
        ops[i] = s3.encode_arm_branch(helper_va + i * 4, helper_va + labels[label] * 4, cond=cond)
    return struct.pack(f"<{len(ops)}I", *ops)


def build_diag_split_cave2() -> bytes:
    """Assemble complete Cave 2 (OSD recovery + post-update shim + diag-split core + 9-grid helper)."""
    render_len = len(osd.build_osd_recovery_cave())
    if render_len != 280:
        raise ValueError(f"Unexpected OSD recovery cave length: {render_len}")

    shim_va = s3.CAVE2_VA + render_len
    core_va = shim_va + 24
    core = build_diag_split_core(core_va)
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
        raise ValueError(f"Step 2C diag-split cave ({len(cave)} B) exceeds zero space before DESC_TABLE_VA")
    return cave


def cave2_layout() -> dict:
    render_len = len(osd.build_osd_recovery_cave())
    shim_va = s3.CAVE2_VA + render_len
    core_va = shim_va + 24
    core_len = len(build_diag_split_core(core_va))
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


def apply_diag_split_experiment(payload: bytearray, rtos_start: int, rtos_end: int) -> dict:
    """Apply Step 2C v1.2.14 diag-split experiment non-destructively inside RTOS bounds."""
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
        cave1_builder=build_diag_split_cave1,
        cave2_builder=build_diag_split_cave2,
    )
    check[off:off + 4] = build_update_hook()
    if payload[:rtos_start] != check[:rtos_start] or payload[rtos_end:] != check[rtos_end:]:
        raise ValueError('Diag-split experiment writes outside RTOS')

    payload[:] = check
    manifest.update({
        'stage': 'Step 2C v1.2.14: 0x1f cache-flush + timeout post-mortem split',
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
    c1 = build_diag_split_cave1()
    c2 = build_diag_split_cave2()
    print(f"Cave 1: {len(c1)} B @ {hex(s3.CAVE1_VA)}..{hex(s3.CAVE1_VA + len(c1))}")
    print(f"Cave 2: {len(c2)} B @ {hex(s3.CAVE2_VA)}..{hex(info['end_va'])} (margin {info['margin_to_desc']} B to DESC_TABLE_VA)")
    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    dec1 = list(md.disasm(c1, s3.CAVE1_VA))
    dec2 = list(md.disasm(c2, s3.CAVE2_VA))
    assert len(dec1) * 4 == len(c1)
    assert len(dec2) * 4 == len(c2)
    print(f"Capstone clean decode: Cave 1 = {len(dec1)} insns, Cave 2 = {len(dec2)} insns")
