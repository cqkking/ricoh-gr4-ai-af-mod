#!/usr/bin/env python3
"""Step 2D v1.2.17: Live-frame continuous CEVA-XM4 AI subject-tracking AF (Option B).

Builds on the real-hardware-verified v1.2.15 OSD-commit-first architecture and
Pentax K-3 III Start -> SendMessage(4) -> 0x1f/Slot-Poll -> WaitMessage(0) -> Stop(100)
lifecycle, adding:
- Live GR IV AHIP YUV420 frame extraction (*([0x55081350 + 0x424]), Y@+0x04, U@+0x08, V@+0x0c)
  with strict null, GR IV DDR-range (0x40000000..0xe0000000 covering RTOS, IMAGE1, and IMAGE2),
  dimension, and D-Cache guards;
- Pixel-center BT.601 YUV420 -> 56x56 (stride 64) planar RGB downsampler into 0x52b00000;
- True K-3 III 3-channel 56x56 int16_t (Q2) segmentation heatmap scanner at 0x52b09a00
  (Ch0 background, Ch1 human/head priority, Ch2 animal/body);
- Dynamic Q8 box sizing, EMA smoothing, 3-cycle lost-frame hold, and Option B clean box
  hiding (face_count = 0) when no subject is present.
"""

import struct
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

import stage3_ai_patcher as s3
import step2c_osd_recovery as osd

LIVE_AI_COOKIE = 0x41493244  # "AI2D"

WARMUP_FRAMES = 20
LOAD_RETRY_MASK = 0x3F       # Retry LoadDsp every 64 frames on cold storage delay
MAX_POLL_FRAMES = 30
PEAK_THRESH = 3              # +0.75 logit in Q2 (0.25/LSB)
AREA_THRESH = 2              # Minimum foreground cells (out of 56x56)
LOST_HOLD_CYCLES = 3         # Hold box for 3 empty cycles, hide on 4th (Option B)
MIN_BOX_Q8 = 0x2800          # 40 px in Q8
MAX_BOX_Q8 = 0x5800          # 88 px in Q8

RESP_BUF_VA = s3.TENSOR_MSG_VA + 0x40
HEATMAP_OUT_VA = s3.TENSOR_IN_VA + 0x9A00   # 0x52b09a00 (true K-3 III 3-ch 56x56 heatmap)
HEATMAP_LEN = 0x5400                        # 3 * 0x1c00 bytes
WORK_BUF_FLUSH_LEN = 0xEE00                 # 0x52b00000..0x52b0edff

STOP_DSP_VA = 0x5398ED08     # DspService::Stop(svc, timeout=100)
OS_EF_WAIT_VA = 0x539030A0   # OsEventFlag::Wait(ef, waiptn, &flgptn, timeout=0)
OS_EF_CLEAR_VA = 0x53902FE8  # OsEventFlag::Clear(ef, clrptn)

HOOK_UPDATE_VA = 0x538963B8
EXPECTED_UPDATE = bytes.fromhex("8dfdffeb")

CAVE3_VA = 0x530018E0        # Zero cave immediately after DESC_TABLE_VA (ends 0x530018d8)
CAVE3_LIMIT_VA = 0x53002000  # End of verified contiguous ROM zero cave


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
    """Cave 1 @ 0x53000d60: zero full 64-byte STATUS_VA region on first boot & publish cookie."""
    insns, mark, b, _, jump, ldr32, finalize = _make_asm(s3.CAVE1_VA)

    # Reconstruct LiveViewProcess3aContoller APCS prologue
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
    b("exit", cond=0x1)  # bne .exit

    ldr32(3, s3.STATUS_VA)
    insns.extend((
        0xE3A02000,  # mov  r2, #0
        0xE3A01040,  # mov  r1, #0x40 (64 bytes = STATUS_VA+0x00..0x3c)
    ))
    mark("zero_status_loop")
    insns.extend((
        0xE2511004,  # subs r1, r1, #4
        0xE7832001,  # str  r2, [r3, r1]
    ))
    b("zero_status_loop", cond=0x1)  # bne .zero_status_loop

    ldr32(2, LIVE_AI_COOKIE)
    insns.append(0xE5832018)  # str r2, [r3, #0x18] (publish cookie last)

    mark("exit")
    jump(s3.HOOK1_RESUME_VA)

    code, _ = finalize()
    if s3.CAVE1_VA + len(code) > s3.CAVE2_VA:
        raise ValueError(f"Cave 1 ({len(code)} B) overlaps Cave 2")
    return code


def build_live_ai_cave3() -> tuple[bytes, dict[str, int]]:
    """Cave 3 @ 0x530018e0: AHIP RGB downsampler, 0x52b09a00 heatmap scanner, EMA tracker."""
    insns, mark, b, call, _, ldr32, finalize = _make_asm(CAVE3_VA)

    # ---------------------------------------------------------------------
    # 1. sub_sample_ahip_rgb(r0 = fd_obj) -> r0 (1=sampled, 0=skip)
    # ---------------------------------------------------------------------
    mark("sub_sample_ahip_rgb")
    # 9 regs (36 B) + 20 B locals = 56 B (8-byte aligned)
    # Locals: [sp+0]=fd_w, [sp+4]=fd_h, [sp+8]=U_ptr, [sp+12]=V_ptr, [sp+16]=uv_row_off
    insns.extend((
        0xE92D4FF0,  # push {r4, r5, r6, r7, r8, sb, sl, fp, lr}
        0xE24DD014,  # sub  sp, sp, #20
        0xE3500000,  # cmp  r0, #0
    ))
    b("sample_fail", cond=0x0)

    # Read & validate fd_w ([r0,#0xfc]) and fd_h ([r0,#0xfe]): 64..1280, w%8==0, h%4==0
    insns.extend((
        0xE1D04FBC,  # ldrh r4, [r0, #0xfc] (fd_w)
        0xE1D05FBE,  # ldrh r5, [r0, #0xfe] (fd_h)
        0xE3540040,  # cmp  r4, #64
    ))
    b("sample_fail", cond=0x3)  # blo
    insns.append(0xE3540C05)    # cmp  r4, #0x500 (1280)
    b("sample_fail", cond=0x8)  # bhi
    insns.append(0xE3140007)    # tst  r4, #7
    b("sample_fail", cond=0x1)  # bne
    insns.append(0xE3550040)    # cmp  r5, #64
    b("sample_fail", cond=0x3)  # blo
    insns.append(0xE3550C05)    # cmp  r5, #0x500 (1280)
    b("sample_fail", cond=0x8)  # bhi
    insns.append(0xE3150003)    # tst  r5, #3
    b("sample_fail", cond=0x1)  # bne

    insns.extend((
        0xE58D4000,  # str  r4, [sp, #0]
        0xE58D5004,  # str  r5, [sp, #4]
    ))

    # Inspect AHIP driver @ 0x55081350: must be active ([+0x30] == 2)
    ldr32(6, s3.AHIP_DRV_BASE_VA)
    insns.extend((
        0xE5961030,  # ldr  r1, [r6, #0x30]
        0xE3510002,  # cmp  r1, #2
    ))
    b("sample_fail", cond=0x1)

    # slot_pp = [ahip + 0x424]; check 0x40000000 <= ptr < 0xe0000000 (RTOS + IMAGE1 + IMAGE2)
    insns.extend((
        0xE5961424,  # ldr  r1, [r6, #0x424]
        0xE2412101,  # sub  r2, r1, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
    ))
    b("sample_fail", cond=0x2)  # bhs

    # slot = [slot_pp]; check DDR range
    insns.extend((
        0xE5911000,  # ldr  r1, [r1]
        0xE2412101,  # sub  r2, r1, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
    ))
    b("sample_fail", cond=0x2)  # bhs

    # Y_ptr = [slot + 0x04]; check DDR range
    insns.extend((
        0xE5917004,  # ldr  r7, [r1, #0x04]
        0xE2472101,  # sub  r2, r7, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
    ))
    b("sample_fail", cond=0x2)  # bhs

    # U_ptr = [slot + 0x08], V_ptr = [slot + 0x0c] (fallback to 0 -> grayscale if invalid)
    insns.extend((
        0xE5918008,  # ldr  r8, [r1, #0x08]
        0xE2482101,  # sub  r2, r8, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
        0x23A08000,  # movhs r8, #0
        0xE591900C,  # ldr  sb, [r1, #0x0c]
        0xE2492101,  # sub  r2, sb, #0x40000000
        0xE352020A,  # cmp  r2, #0xa0000000
        0x23A09000,  # movhs sb, #0
        0xE58D8008,  # str  r8, [sp, #8]
        0xE58D900C,  # str  sb, [sp, #12]
    ))

    # Invalidate ARM D-Cache for live YUV420 slot before CPU reads
    insns.extend((
        0xE0030594,  # mul  r3, r4, r5 (min_size = fd_w * fd_h)
        0xE596100C,  # ldr  r1, [r6, #0x0c] (AHIP slot total size)
        0xE1510003,  # cmp  r1, r3
    ))
    b("use_default_sz", cond=0x3)  # blo
    insns.append(0xE3510502)       # cmp  r1, #0x00800000 (8 MB cap)
    b("have_total_sz", cond=0x9)   # bls
    mark("use_default_sz")
    insns.append(0xE08310A3)       # add  r1, r3, r3, lsr #1 (1.5 * fd_w * fd_h)
    mark("have_total_sz")
    insns.append(0xE1A0A001)       # mov  sl, r1
    call(s3.GET_CACHE_MGR_VA)
    insns.extend((
        0xE1A01007,  # mov  r1, r7 (Y_ptr)
        0xE1A0200A,  # mov  r2, sl (total_size)
    ))
    call(s3.INVAL_DCACHE_VA)

    # Pixel-center BT.601 YUV420 -> 56x56 (stride 64) planar RGB @ TENSOR_IN_VA
    # r4=fd_w, r5=fd_h, r7=Y_ptr, fp=TENSOR_IN_VA, r6=row (0..55)
    ldr32(11, s3.TENSOR_IN_VA)
    insns.append(0xE3A06000)  # mov r6, #0 (row = 0)

    mark("ds_row_loop")
    insns.extend((
        0xE0860006,  # add  r0, r6, r6       (2*row)
        0xE2800001,  # add  r0, r0, #1       (2*row + 1)
        0xE0000095,  # mul  r0, r5, r0       ((2*row + 1) * fd_h)
        0xE3A01049,  # mov  r1, #73
        0xE0000091,  # mul  r0, r1, r0       (* 73)
        0xE1A006A0,  # lsr  r0, r0, #13      (src_y)
        0xE0287490,  # mla  r8, r0, r4, r7   (y_row_ptr = Y_ptr + src_y * fd_w)
        0xE1A010A0,  # lsr  r1, r0, #1       (uv_y = src_y >> 1)
        0xE1A020A4,  # lsr  r2, r4, #1       (uv_w = fd_w >> 1)
        0xE0010192,  # mul  r1, r2, r1       (uv_row_off = uv_y * uv_w)
        0xE58D1010,  # str  r1, [sp, #16]
        0xE08B9306,  # add  sb, fp, r6, lsl #6 (dst_row_ptr = TENSOR_IN_VA + row * 64)
        0xE3A0A000,  # mov  sl, #0           (col = 0)
    ))

    mark("ds_col_loop")
    insns.extend((
        0xE08A000A,  # add  r0, sl, sl       (2*col)
        0xE2800001,  # add  r0, r0, #1       (2*col + 1)
        0xE0000094,  # mul  r0, r4, r0       ((2*col + 1) * fd_w)
        0xE3A01049,  # mov  r1, #73
        0xE0000091,  # mul  r0, r1, r0       (* 73)
        0xE1A006A0,  # lsr  r0, r0, #13      (src_x)
        0xE7D81000,  # ldrb r1, [r8, r0]     (Y)
        0xE59D2008,  # ldr  r2, [sp, #8]     (U_ptr)
        0xE59D300C,  # ldr  r3, [sp, #12]    (V_ptr)
        0xE3520000,  # cmp  r2, #0
    ))
    b("ds_gray", cond=0x0)
    insns.append(0xE3530000)  # cmp r3, #0
    b("ds_gray", cond=0x0)

    # Pure BT.601 YUV420 -> RGB
    insns.extend((
        0xE59DC010,  # ldr  ip, [sp, #16]         (uv_row_off)
        0xE08CC0A0,  # add  ip, ip, r0, lsr #1    (uv_idx = uv_row_off + (src_x >> 1))
        0xE7D2200C,  # ldrb r2, [r2, ip]          (U_raw)
        0xE7D3300C,  # ldrb r3, [r3, ip]          (V_raw)
        0xE2422080,  # sub  r2, r2, #128          (u = U_raw - 128)
        0xE2433080,  # sub  r3, r3, #128          (v = V_raw - 128)
        # R = usat8(Y + v + (v >> 1) - (v >> 3))  [1.375*v vs exact 1.402*v]
        0xE08300C3,  # add  r0, r3, r3, asr #1
        0xE04001C3,  # sub  r0, r0, r3, asr #3
        0xE0810000,  # add  r0, r1, r0
        0xE6E80010,  # usat r0, #8, r0
        # G = usat8(Y - ((11*u + 23*v) >> 5))     [0.34375*u + 0.71875*v]
        0xE082C082,  # add  ip, r2, r2, lsl #1    (3*u)
        0xE08CC182,  # add  ip, ip, r2, lsl #3    (11*u)
        0xE063E183,  # rsb  lr, r3, r3, lsl #3    (7*v)
        0xE08EE203,  # add  lr, lr, r3, lsl #4    (23*v)
        0xE08CC00E,  # add  ip, ip, lr            (11*u + 23*v)
        0xE041C2CC,  # sub  ip, r1, ip, asr #5
        0xE6E8C01C,  # usat ip, #8, ip
        # B = usat8(Y + u + (u >> 1) + (u >> 2))  [1.75*u vs exact 1.772*u]
        0xE082E0C2,  # add  lr, r2, r2, asr #1
        0xE08E2142,  # add  r2, lr, r2, asr #2
        0xE0812002,  # add  r2, r1, r2
        0xE6E82012,  # usat r2, #8, r2
    ))
    b("ds_store_rgb")

    mark("ds_gray")
    insns.extend((
        0xE1A00001,  # mov  r0, r1 (R = Y)
        0xE1A0C001,  # mov  ip, r1 (G = Y)
        0xE1A02001,  # mov  r2, r1 (B = Y)
    ))

    mark("ds_store_rgb")
    insns.extend((
        0xE089300A,  # add  r3, sb, sl            (dst_row_ptr + col)
        0xE5C30000,  # strb r0, [r3, #0]          (Plane 0: R @ +0x0000)
        0xE5C3CE00,  # strb ip, [r3, #0xe00]      (Plane 1: G @ +0x0e00)
        0xE2833C0E,  # add  r3, r3, #0xe00
        0xE5C32E00,  # strb r2, [r3, #0xe00]      (Plane 2: B @ +0x1c00)
        0xE28AA001,  # add  sl, sl, #1            (col++)
        0xE35A0038,  # cmp  sl, #56
    ))
    b("ds_col_loop", cond=0xB)  # blt

    # Zero 8-byte stride padding (cols 56..63) in all 3 planes
    insns.extend((
        0xE3A00000,  # mov  r0, #0
        0xE5890038,  # str  r0, [sb, #0x38]
        0xE589003C,  # str  r0, [sb, #0x3c]
        0xE2899C0E,  # add  sb, sb, #0xe00
        0xE5890038,  # str  r0, [sb, #0x38]
        0xE589003C,  # str  r0, [sb, #0x3c]
        0xE2899C0E,  # add  sb, sb, #0xe00
        0xE5890038,  # str  r0, [sb, #0x38]
        0xE589003C,  # str  r0, [sb, #0x3c]
        0xE2866001,  # add  r6, r6, #1            (row++)
        0xE3560038,  # cmp  r6, #56
    ))
    b("ds_row_loop", cond=0xB)  # blt

    # Clear 0x52b09a00..0x52b0edff (0x5400 B) before DSP DMA so stale heatmaps cannot linger
    ldr32(0, HEATMAP_OUT_VA)
    insns.extend((
        0xE3A01000,  # mov  r1, #0
        s3.encode_movw(2, HEATMAP_LEN),
    ))
    mark("ds_zero_hm")
    insns.extend((
        0xE2522004,  # subs r2, r2, #4
        0xE7801002,  # str  r1, [r0, r2]
    ))
    b("ds_zero_hm", cond=0x1)  # bne

    # Flush full work_buf (0x52b00000, 0xee00 B) to DDR for DSP DMA
    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, s3.TENSOR_IN_VA)
    insns.append(s3.encode_movw(2, WORK_BUF_FLUSH_LEN))
    call(s3.FLUSH_DCACHE_VA)

    insns.extend((
        0xE3A00001,  # mov  r0, #1
        0xE28DD014,  # add  sp, sp, #20
        0xE8BD8FF0,  # pop  {r4, r5, r6, r7, r8, sb, sl, fp, pc}
    ))

    mark("sample_fail")
    insns.extend((
        0xE3A00000,  # mov  r0, #0
        0xE28DD014,  # add  sp, sp, #20
        0xE8BD8FF0,  # pop  {r4, r5, r6, r7, r8, sb, sl, fp, pc}
    ))

    # ---------------------------------------------------------------------
    # 2. sub_scan_heatmap(r0 = HEATMAP_OUT_VA)
    #    -> r0=max_diff, r1=best_col, r2=best_row, r3=area_cnt
    # ---------------------------------------------------------------------
    mark("sub_scan_heatmap")
    insns.append(0xE92D4FF0)  # push {r4, r5, r6, r7, r8, sb, sl, fp, lr}
    ldr32(1, 0xFFFF8000)      # r1 = max_diff = -32768 (signed)
    insns.extend((
        0xE3A0201C,  # mov  r2, #28           (best_col)
        0xE3A0301C,  # mov  r3, #28           (best_row)
        0xE3A0B000,  # mov  fp, #0            (area_cnt)
        0xE2804B07,  # add  r4, r0, #0x1c00   (Ch1: Human/Head)
        0xE2805B0E,  # add  r5, r0, #0x3800   (Ch2: Animal/Body)
        0xE3A06000,  # mov  r6, #0            (row = 0)
    ))

    mark("hm_row_loop")
    insns.append(0xE3A07000)  # mov r7, #0 (col = 0)
    mark("hm_col_loop")
    insns.extend((
        0xE0878306,  # add  r8, r7, r6, lsl #6 (idx = row * 64 + col)
        0xE1A08088,  # lsl  r8, r8, #1         (byte_off = idx * 2)
        0xE19090F8,  # ldrsh sb, [r0, r8]      (bg = Ch0)
        0xE194A0F8,  # ldrsh sl, [r4, r8]      (c1 = Ch1)
        0xE195C0F8,  # ldrsh ip, [r5, r8]      (c2 = Ch2)
        0xE24CC002,  # sub  ip, ip, #2         (head priority: c2 - 0.5 logit)
        0xE15C000A,  # cmp  ip, sl
        0xC1A0A00C,  # movgt sl, ip            (fg = max(c1, c2 - 2))
        0xE04AA009,  # sub  sl, sl, sb         (diff = fg - bg)
        0xE35A0000 | PEAK_THRESH,  # cmp sl, #PEAK_THRESH
        0xA28BB001,  # addge fp, fp, #1        (area_cnt += (diff >= PEAK_THRESH))
        0xE15A0001,  # cmp  sl, r1
    ))
    b("hm_skip_peak", cond=0xD)  # ble (signed)
    insns.extend((
        0xE1A0100A,  # mov  r1, sl             (max_diff = diff)
        0xE1A02007,  # mov  r2, r7             (best_col = col)
        0xE1A03006,  # mov  r3, r6             (best_row = row)
    ))
    mark("hm_skip_peak")
    insns.extend((
        0xE2877001,  # add  r7, r7, #1
        0xE3570038,  # cmp  r7, #56
    ))
    b("hm_col_loop", cond=0xB)  # blt
    insns.extend((
        0xE2866001,  # add  r6, r6, #1
        0xE3560038,  # cmp  r6, #56
    ))
    b("hm_row_loop", cond=0xB)  # blt

    insns.extend((
        0xE1A00001,  # mov  r0, r1 (max_diff)
        0xE1A01002,  # mov  r1, r2 (best_col)
        0xE1A02003,  # mov  r2, r3 (best_row)
        0xE1A0300B,  # mov  r3, fp (area_cnt)
        0xE8BD8FF0,  # pop  {r4, r5, r6, r7, r8, sb, sl, fp, pc}
    ))

    # ---------------------------------------------------------------------
    # 3. sub_update_tracking_box(r0=max_diff, r1=best_col, r2=best_row,
    #                            r3=area_cnt, [sp+0]=fd_obj)
    # ---------------------------------------------------------------------
    mark("sub_update_tracking_box")
    insns.append(0xE92D47F0)  # push {r4, r5, r6, r7, r8, sb, sl, lr} (32 B -> fd_obj @ [sp,#32])
    ldr32(4, s3.STATUS_VA)
    insns.extend((
        0xE5840038,  # str  r0, [r4, #0x38] (last_peak)
        0xE584303C,  # str  r3, [r4, #0x3c] (last_area)
        0xE3500000 | PEAK_THRESH,  # cmp r0, #PEAK_THRESH
    ))
    b("trk_no_subject", cond=0xB)  # blt (signed)
    insns.append(0xE3530000 | AREA_THRESH)  # cmp r3, #AREA_THRESH
    b("trk_no_subject", cond=0xB)  # blt (signed)

    # Read fd_w, fd_h from fd_obj ([sp, #32])
    insns.extend((
        0xE59D5020,  # ldr  r5, [sp, #32]   (fd_obj)
        0xE1D56FBC,  # ldrh r6, [r5, #0xfc] (fd_w)
        0xE3560000,  # cmp  r6, #0
        s3.encode_movw(6, 320, cond=0x0),
        0xE1D57FBE,  # ldrh r7, [r5, #0xfe] (fd_h)
        0xE3570000,  # cmp  r7, #0
        s3.encode_movw(7, 212, cond=0x0),
        # Pixel-center Q8 coordinates: cx_q8 = ((2*col + 1) * fd_w * 73) >> 5
        0xE0818001,  # add  r8, r1, r1
        0xE2888001,  # add  r8, r8, #1
        0xE0080896,  # mul  r8, r6, r8
        0xE3A0C049,  # mov  ip, #73
        0xE008089C,  # mul  r8, ip, r8
        0xE1A082A8,  # lsr  r8, r8, #5      (cx_q8)
        # cy_q8 = ((2*row + 1) * fd_h * 73) >> 5
        0xE0829002,  # add  sb, r2, r2
        0xE2899001,  # add  sb, sb, #1
        0xE0090997,  # mul  sb, r7, sb
        0xE009099C,  # mul  sb, ip, sb
        0xE1A092A9,  # lsr  sb, sb, #5      (cy_q8)
        # Dynamic Q8 box size: clamp(0x2800 + (area_cnt << 5), 0x2800, min(0x5800, min(w,h)<<7))
        s3.encode_movw(10, MIN_BOX_Q8),
        0xE08AA283,  # add  sl, sl, r3, lsl #5
        s3.encode_movw(12, MAX_BOX_Q8),
        0xE15A000C,  # cmp  sl, ip
        0xC1A0A00C,  # movgt sl, ip
        0xE1560007,  # cmp  r6, r7
        0xD1A0C006,  # movle ip, r6
        0xC1A0C007,  # movgt ip, r7
        0xE1A0C38C,  # lsl  ip, ip, #7      (min(fd_w, fd_h) << 7)
        0xE15A000C,  # cmp  sl, ip
        0xC1A0A00C,  # movgt sl, ip
        # Clamp new_left = clamp(cx_q8 - size/2, 0, (fd_w << 8) - size)
        0xE05880AA,  # subs r8, r8, sl, lsr #1
        0x43A08000,  # movmi r8, #0
        0xE1A0C406,  # lsl  ip, r6, #8
        0xE04CC00A,  # sub  ip, ip, sl
        0xE158000C,  # cmp  r8, ip
        0xC1A0800C,  # movgt r8, ip
        # Clamp new_top = clamp(cy_q8 - size/2, 0, (fd_h << 8) - size)
        0xE05990AA,  # subs sb, sb, sl, lsr #1
        0x43A09000,  # movmi sb, #0
        0xE1A0C407,  # lsl  ip, r7, #8
        0xE04CC00A,  # sub  ip, ip, sl
        0xE159000C,  # cmp  sb, ip
        0xC1A0900C,  # movgt sb, ip
        # Confidence = clamp(max_diff + 40, 40, 100)
        0xE2800028,  # add  r0, r0, #40
        0xE3500064,  # cmp  r0, #100
        0xC3A00064,  # movgt r0, #100
        # Cold-start instant snap if has_box == 0; else EMA smooth
        0xE594C024,  # ldr  ip, [r4, #0x24]
        0xE35C0000,  # cmp  ip, #0
    ))
    b("trk_store", cond=0x0)  # beq (cold snap)

    insns.extend((
        0xE5941028,  # ldr  r1, [r4, #0x28] (prev_left)
        0xE0888001,  # add  r8, r8, r1
        0xE2888001,  # add  r8, r8, #1
        0xE1A080A8,  # lsr  r8, r8, #1      ((prev + new + 1) >> 1)
        0xE594102C,  # ldr  r1, [r4, #0x2c] (prev_top)
        0xE0899001,  # add  sb, sb, r1
        0xE2899001,  # add  sb, sb, #1
        0xE1A090A9,  # lsr  sb, sb, #1
        0xE5941030,  # ldr  r1, [r4, #0x30] (prev_size)
        0xE0811081,  # add  r1, r1, r1, lsl #1 (prev * 3)
        0xE08AA001,  # add  sl, sl, r1
        0xE28AA002,  # add  sl, sl, #2
        0xE1A0A12A,  # lsr  sl, sl, #2      ((prev * 3 + new + 2) >> 2)
    ))

    mark("trk_store")
    insns.extend((
        0xE5848028,  # str  r8, [r4, #0x28] (box_left)
        0xE584902C,  # str  sb, [r4, #0x2c] (box_top)
        0xE584A030,  # str  sl, [r4, #0x30] (box_size)
        0xE5840034,  # str  r0, [r4, #0x34] (confidence)
        0xE3A01001,  # mov  r1, #1
        0xE5841024,  # str  r1, [r4, #0x24] (has_box = 1)
        0xE3A01000,  # mov  r1, #0
        0xE584101C,  # str  r1, [r4, #0x1c] (lost_cycles = 0)
        0xE8BD87F0,  # pop  {r4, r5, r6, r7, r8, sb, sl, pc}
    ))

    mark("trk_no_subject")
    insns.extend((
        0xE594101C,  # ldr  r1, [r4, #0x1c] (lost_cycles)
        0xE2811001,  # add  r1, r1, #1
        0xE584101C,  # str  r1, [r4, #0x1c]
        0xE3510000 | LOST_HOLD_CYCLES,  # cmp r1, #LOST_HOLD_CYCLES
        0xC3A02000,  # movgt r2, #0
        0xC5842024,  # strgt r2, [r4, #0x24] (Option B: hide box when lost_cycles > 3)
        0xE8BD87F0,  # pop  {r4, r5, r6, r7, r8, sb, sl, pc}
    ))

    code, labels = finalize()
    if CAVE3_VA + len(code) > CAVE3_LIMIT_VA:
        raise ValueError(f"Cave 3 ({len(code)} B) exceeds ROM zero ceiling {hex(CAVE3_LIMIT_VA)}")
    return code, {
        "sub_sample_ahip_rgb": labels["sub_sample_ahip_rgb"],
        "sub_scan_heatmap": labels["sub_scan_heatmap"],
        "sub_update_tracking_box": labels["sub_update_tracking_box"],
    }


def _build_render_osd(render_va: int, bypass_va: int) -> bytes:
    """OSD renderer @ CAVE2_VA (runs while native FD mutex r6 is held; Option B hiding)."""
    insns, mark, b, _, jump, ldr32, finalize = _make_asm(render_va)

    # Validate 4 prerequisites identically to step2c_osd_recovery.py
    insns.extend((0xE5940078, 0xE3500000))  # result != NULL
    jump(bypass_va, cond=0x0)
    insns.extend((0xE5940038, 0xE3500000))  # FdLibraryService != NULL
    jump(bypass_va, cond=0x0)
    insns.extend((0xE1D40FBC, 0xE3500030))  # fd_w >= 48
    jump(bypass_va, cond=0x3)
    insns.extend((0xE1D40FBE, 0xE3500030))  # fd_h >= 48
    jump(bypass_va, cond=0x3)

    insns.append(0xE92D47F0)  # push {r4, r5, r6, r7, r8, sb, sl, lr}

    ldr32(3, s3.STATUS_VA)
    ldr32(1, LIVE_AI_COOKIE)
    insns.extend((
        0xE5932018,  # ldr  r2, [r3, #0x18] (cookie)
        0xE1520001,  # cmp  r2, r1
    ))
    b("hide_box", cond=0x1)  # bne
    insns.extend((
        0xE5932024,  # ldr  r2, [r3, #0x24] (has_box)
        0xE3520000,  # cmp  r2, #0
    ))
    b("hide_box", cond=0x0)  # beq

    # Load smoothed Q8 box & clamp safely against current fd_w/fd_h
    insns.extend((
        0xE5937028,  # ldr  r7, [r3, #0x28] (left)
        0xE593802C,  # ldr  r8, [r3, #0x2c] (top)
        0xE5935030,  # ldr  r5, [r3, #0x30] (size)
        0xE5939034,  # ldr  sb, [r3, #0x34] (conf)
        0xE1D40FBC,  # ldrh r0, [r4, #0xfc] (fd_w)
        0xE1A00400,  # lsl  r0, r0, #8      (fd_w_q8)
        0xE1D41FBE,  # ldrh r1, [r4, #0xfe] (fd_h)
        0xE1A01401,  # lsl  r1, r1, #8      (fd_h_q8)
        0xE3A02B08,  # mov  r2, #0x2000
        0xE1550002,  # cmp  r5, r2
        0x31A05002,  # movlo r5, r2
        0xE1500001,  # cmp  r0, r1
        0x31A02000,  # movlo r2, r0
        0x21A02001,  # movhs r2, r1
        0xE1550002,  # cmp  r5, r2
        0x81A05002,  # movhi r5, r2
        0xE0402005,  # sub  r2, r0, r5
        0xE1570002,  # cmp  r7, r2
        0x81A07002,  # movhi r7, r2
        0xE0412005,  # sub  r2, r1, r5
        0xE1580002,  # cmp  r8, r2
        0x81A08002,  # movhi r8, r2
    ))

    # Zero 100 bytes (0x64) of m_DetectedFaceResult[0]
    insns.extend((
        0xE594A078,  # ldr  sl, [r4, #0x78]
        0xE3A00000,  # mov  r0, #0
        0xE3A01000,  # mov  r1, #0
    ))
    mark("zero_face_loop")
    insns.extend((
        0xE78A0001,  # str  r0, [sl, r1]
        0xE2811004,  # add  r1, r1, #4
        0xE3510064,  # cmp  r1, #0x64
    ))
    b("zero_face_loop", cond=0xB)

    # Populate m_DetectedFaceResult[0] & FaceDetectionService tracking state
    insns.extend((
        0xE58A7000,  # str  r7, [sl, #0x00] (left)
        0xE58A8004,  # str  r8, [sl, #0x04] (top)
        0xE58A5008,  # str  r5, [sl, #0x08] (width)
        0xE58A500C,  # str  r5, [sl, #0x0c] (height)
        0xE58A904C,  # str  sb, [sl, #0x4c] (confidence)
        0xE5941118,  # ldr  r1, [r4, #0x118]
        0xE3510000,  # cmp  r1, #0
        0x03A01001,  # moveq r1, #1
        0xE58A1050,  # str  r1, [sl, #0x50] (face_id)
        0xE08720A5,  # add  r2, r7, r5, lsr #1 (cx)
        0xE5842100,  # str  r2, [r4, #0x100]
        0xE08820A5,  # add  r2, r8, r5, lsr #1 (cy)
        0xE5842104,  # str  r2, [r4, #0x104]
        0xE5845108,  # str  r5, [r4, #0x108]
        0xE584510C,  # str  r5, [r4, #0x10c]
        0xE5841118,  # str  r1, [r4, #0x118]
        0xE584111C,  # str  r1, [r4, #0x11c]
        0xE3A02001,  # mov  r2, #1
        0xE5C4213C,  # strb r2, [r4, #0x13c] (tracking_active = 1)
        0xE3A00000,  # mov  r0, #0
        0xE5C40121,  # strb r0, [r4, #0x121] (is_stabilizing = 0)
        0xE2843F49,  # add  r3, r4, #0x124
        0xE1C300B0,  # strh r0, [r3]         (lost_frames_counter = 0)
        0xE5943038,  # ldr  r3, [r4, #0x38]  (FdLibraryService)
        0xE3A01001,  # mov  r1, #1
        0xE5831030,  # str  r1, [r3, #0x30]  (face_count = 1)
        0xE3A01000,  # mov  r1, #0
        0xE5831034,  # str  r1, [r3, #0x34]  (main_face_idx = 0)
    ))
    b("render_done")

    # Option B: hide box cleanly (face_count = 0), MUST still resume at HOOK2_AI_RESUME_VA!
    mark("hide_box")
    insns.extend((
        0xE5943038,  # ldr  r3, [r4, #0x38]
        0xE3A01000,  # mov  r1, #0
        0xE5831030,  # str  r1, [r3, #0x30]  (face_count = 0)
        0xE5831034,  # str  r1, [r3, #0x34]  (main_face_idx = 0)
    ))

    mark("render_done")
    insns.append(0xE8BD47F0)  # pop {r4, r5, r6, r7, r8, sb, sl, lr}
    jump(s3.HOOK2_AI_RESUME_VA)

    code, _ = finalize()
    return code


def _build_live_ai_core(core_va: int, cave3_syms: dict[str, int]) -> bytes:
    """Continuous fault-tolerant Start -> Send -> 0x1f/Slot-Poll -> Wait -> Stop(100) core."""
    insns, mark, b, call, jump, ldr32, finalize = _make_asm(core_va)

    # Prologue: 8 regs (32 B) + 8 B scratch = 40 B (8-byte aligned)
    insns.extend((
        0xE92D47F0,  # push {r4, r5, r6, r7, r8, sb, sl, lr}
        0xE24DD008,  # sub  sp, sp, #8
        0xE1A0A000,  # mov  sl, r0          (sl = fd_obj)
    ))

    ldr32(8, s3.STATUS_VA)
    insns.append(0xE5982018)  # ldr r2, [r8, #0x18]
    ldr32(3, LIVE_AI_COOKIE)
    insns.append(0xE1520003)  # cmp r2, r3
    b("exit_done", cond=0x1)

    insns.extend((
        0xE5987000,  # ldr r7, [r8, #0] (state)
        0xE3570003,  # cmp r7, #3 (IN_FLIGHT)
    ))
    b("do_poll_in_flight", cond=0x0)
    insns.append(0xE3570002)  # cmp r7, #2 (SEND_NEXT)
    b("do_send_next", cond=0x0)
    insns.append(0xE3570000)  # cmp r7, #0 (WARMUP / LOAD)
    b("exit_done", cond=0x1)

    # --- State 0: Warmup (20 frames) + One-shot LoadDsp (retries every 64 frames if storage busy) ---
    insns.extend((
        0xE5982004,  # ldr  r2, [r8, #4] (ticks)
        0xE2822001,  # add  r2, r2, #1
        0xE5882004,  # str  r2, [r8, #4]
        0xE2523000 | WARMUP_FRAMES,  # subs r3, r2, #WARMUP_FRAMES
    ))
    b("exit_done", cond=0x3)  # blo (wait warmup)
    insns.append(0xE3130000 | LOAD_RETRY_MASK)  # tst r3, #0x3f
    b("exit_done", cond=0x1)  # bne (retry cadence)

    call(s3.GET_DSP_SVC_VA)
    insns.append(0xE3500000)
    b("exit_done", cond=0x0)
    insns.extend((
        0xE5905004,  # ldr r5, [r0, #4]
        0xE3550000,  # cmp r5, #0
    ))
    b("have_svc_init", cond=0x1)

    insns.append(0xE3A01000)  # mov r1, #0
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
    # Never hijack an already-started DspService
    insns.extend((
        0xE5952004,  # ldr r2, [r5, #4]
        0xE3120004,  # tst r2, #4 (STARTED)
    ))
    b("exit_done", cond=0x1)

    # Try Drive 'C' (SD card), fallback to Drive 'I' (internal flash)
    insns.extend((
        0xE1A00005,  # mov r0, r5
        0xE3A01043,  # mov r1, #'C'
        0xE3A02000,
        0xE3A03000,
    ))
    call(s3.LOAD_DSP_VA)
    insns.append(0xE3500000)
    b("load_ok", cond=0x1)

    insns.extend((
        0xE1A00005,  # mov r0, r5
        0xE3A01049,  # mov r1, #'I'
        0xE3A02000,
        0xE3A03000,
    ))
    call(s3.LOAD_DSP_VA)
    insns.append(0xE3500000)
    b("load_ok", cond=0x1)

    # Both failed: clear false LOADED bit and retry in 64 frames
    insns.extend((
        0xE3A02000,
        0xE5852004,  # str r2, [r5, #4]
        0xE3A02001,
        0xE5882010,  # last_err = 1
    ))
    b("exit_done")

    mark("load_ok")
    insns.extend((
        0xE5885020,  # service_ptr = r5
        0xE3A00002,  # state = 2 (SEND_NEXT)
        0xE5880000,
    ))
    b("exit_done")

    # --- State 2: Sample live AHIP RGB -> Start(0x52100000) -> SendMessage(4) ---
    mark("do_send_next")
    call(s3.GET_DSP_SVC_VA)
    b("check_service_identity", bl=True)
    insns.extend((
        0xE5952004,
        0xE3120004,  # if unexpectedly still started, stop cleanly and retry next frame
    ))
    b("stop_and_retry", cond=0x1)

    # Sample live AHIP frame into 0x52b00000 & flush cache
    insns.append(0xE1A0000A)  # mov r0, sl (fd_obj)
    call(cave3_syms["sub_sample_ahip_rgb"])
    insns.append(0xE3500000)
    b("exit_done", cond=0x0)  # skip cycle safely if AHIP not ready

    # Start DSP at 0x52100000
    insns.append(0xE1A00005)
    ldr32(1, 0x52100000)
    insns.append(0xE3A02000)
    call(s3.START_DSP_VA)
    insns.extend((
        0xE5880010,  # last_err = Start() ret
        0xE5952004,
        0xE3120004,  # verify STARTED bit set
    ))
    b("exit_done", cond=0x0)

    # Prepare 10-word DspCommandParameter @ TENSOR_MSG_VA
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
    ldr32(0, 0x0C0C0C0C)
    insns.append(0xE587001C)

    insns.extend((
        0xE2886008,  # r6 = STATUS_VA + 8 (&seq)
        0xE3A00000,
        0xE1C600B0,  # strh r0, [r6]
        0xE1A00005,  # mov  r0, r5
        0xE1A01006,  # mov  r1, r6
        0xE3A02004,  # mov  r2, #4
        0xE3A03064,  # mov  r3, #100
        0xE58D7000,  # str  r7, [sp]
    ))
    call(s3.SEND_MSG_VA)
    insns.extend((
        0xE5880010,  # last_err
        0xE3500000,
    ))
    b("stop_and_retry", cond=0x1)

    insns.extend((
        0xE3A00000,
        0xE588000C,  # poll_frames = 0
        0xE3A00003,  # state = 3 (IN_FLIGHT)
        0xE5880000,
    ))
    b("exit_done")

    # --- State 3: Service 0x1f + read-only [svc+0x54] slot poll -> Wait -> Stop(100) -> Heatmap ---
    mark("do_poll_in_flight")
    call(s3.GET_DSP_SVC_VA)
    b("check_service_identity", bl=True)
    insns.extend((
        0xE5952004,
        0xE3120004,  # if unexpectedly stopped, return to SEND_NEXT
    ))
    b("reset_to_send_next", cond=0x0)

    insns.extend((
        0xE2886008,  # r6 = STATUS_VA + 8 (&seq)
        0xE1D610B0,  # ldrh r1, [r6]
        0xE3510005,  # cmp  r1, #5
    ))
    b("stop_and_retry", cond=0x8)  # bhi

    # Service DSP msg_id==0x1f cache flush flag (DspSF bit 0x10000) read-only
    insns.extend((
        0xE3A00000,
        0xE58D0000,
        0xE285000C,  # r0 = svc + 0x0c (DspSF)
        0xE3A01801,  # r1 = 0x10000
        0xE1A0200D,  # r2 = sp (&flgptn)
        0xE3A03000,  # r3 = 0 (timeout=0)
    ))
    call(OS_EF_WAIT_VA)
    insns.append(0xE3500000)
    b("skip_flush1f", cond=0x0)

    insns.extend((
        0xE5953058,  # cache_req_table = [svc + 0x58]
        0xE3530000,
    ))
    b("clear_1f", cond=0x0)
    insns.extend((
        0xE5932008,  # size = [table + 8]
        0xE3520000,
    ))
    b("clear_1f", cond=0x0)
    insns.append(0xE1A04003)  # mov r4, r3
    call(s3.GET_CACHE_MGR_VA)
    insns.extend((
        0xE5941004,  # addr = [r4 + 4]
        0xE5942008,  # size = [r4 + 8]
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
        0xE1D610B0,  # ldrh r1, [r6] (seq)
        0xE5950054,  # r0 = [svc + 0x54] (6-slot reply table)
        0xE3500000,
    ))
    b("stop_and_retry", cond=0x0)

    insns.append(0xE3A02000)  # off = 0
    mark("scan_slot")
    insns.extend((
        0xE7D03002,  # ldrb r3, [r0, r2] (occupied)
        0xE3530000,
    ))
    b("scan_next", cond=0x0)
    insns.extend((
        0xE0807002,  # add  r7, r0, r2
        0xE1D730B2,  # ldrh r3, [r7, #2] (slot seq)
        0xE1530001,  # cmp  r3, r1
    ))
    b("slot_ready", cond=0x0)
    mark("scan_next")
    insns.extend((
        0xE282202C,  # off += 0x2c
        0xE3520F42,  # cmp off, #0x108
    ))
    b("scan_slot", cond=0x1)

    # Not ready yet: increment poll_frames; auto-recover on timeout >= MAX_POLL_FRAMES
    insns.extend((
        0xE598200C,
        0xE2822001,
        0xE588200C,
        0xE3520000 | MAX_POLL_FRAMES,
    ))
    b("exit_done", cond=0x3)  # blo (still waiting, 0 ms block)

    insns.extend((
        0xE3A00008,
        0xE5880010,  # last_err = 8 (timeout)
    ))
    b("stop_and_retry")

    # Matching reply ready: WaitMessage(0) -> Stop(100) -> Invalidate 0x52b09a00 -> Scan -> Update Box
    mark("slot_ready")
    insns.extend((
        0xE1A00005,  # r0 = svc
        0xE1D610B0,  # r1 = seq
        0xE3A02000,  # r2 = 0 (timeout)
    ))
    ldr32(3, RESP_BUF_VA)
    call(s3.WAIT_MSG_VA)
    insns.append(0xE1A04000)  # r4 = WaitMessage return code

    # Always Stop(100) cleanly around every inference (K-3 III lifecycle)
    b("stop_dsp", bl=True)
    insns.extend((
        0xE3A00002,  # state = 2 (SEND_NEXT)
        0xE5880000,
        0xE3540000,  # cmp r4, #0
    ))
    b("wait_ok", cond=0x0)
    insns.append(0xE5884010)  # last_err = WaitMessage error
    b("exit_done")

    mark("wait_ok")
    insns.extend((
        0xE5982014,  # cycles_ok++
        0xE2822001,
        0xE5882014,
    ))
    call(s3.GET_CACHE_MGR_VA)
    ldr32(1, HEATMAP_OUT_VA)
    insns.append(s3.encode_movw(2, HEATMAP_LEN))
    call(s3.INVAL_DCACHE_VA)

    ldr32(0, HEATMAP_OUT_VA)
    call(cave3_syms["sub_scan_heatmap"])
    insns.append(0xE58DA000)  # str sl, [sp, #0] (pass fd_obj)
    call(cave3_syms["sub_update_tracking_box"])
    b("exit_done")

    # --- Helpers ---
    mark("stop_and_retry")
    b("stop_dsp", bl=True)
    mark("reset_to_send_next")
    insns.extend((
        0xE3A00002,  # state = 2 (SEND_NEXT)
        0xE5880000,
    ))
    b("exit_done")

    mark("stop_dsp")
    insns.extend((
        0xE1A00005,  # mov r0, r5
        0xE3A01064,  # mov r1, #100
    ))
    jump(STOP_DSP_VA)

    mark("check_service_identity")
    insns.append(0xE3500000)
    b("exit_done", cond=0x0)
    insns.extend((0xE5905004, 0xE3550000))
    b("exit_done", cond=0x0)
    insns.extend((0xE5982020, 0xE1550002))
    b("exit_done", cond=0x1)
    insns.append(0xE12FFF1E)  # bx lr

    mark("exit_done")
    insns.extend((
        0xE28DD008,  # add sp, sp, #8
        0xE8BD87F0,  # pop {r4, r5, r6, r7, r8, sb, sl, pc}
    ))

    code, _ = finalize()
    return code


def cave_layout() -> dict:
    """Compute exact virtual addresses and sizes for Caves 1, 2, and 3."""
    cave1 = build_live_ai_cave1()
    cave3, cave3_syms = build_live_ai_cave3()

    # Render block size is independent of bypass_va target address
    render_len = len(_build_render_osd(s3.CAVE2_VA, s3.CAVE2_VA))
    shim_va = s3.CAVE2_VA + render_len
    shim_len = 28  # 7 words (includes mov r0, r4)
    bypass_va = shim_va + shim_len
    bypass_len = 20  # 5 words
    core_va = bypass_va + bypass_len
    core_len = len(_build_live_ai_core(core_va, cave3_syms))
    cave2_len = render_len + shim_len + bypass_len + core_len

    return {
        "cave1_va": s3.CAVE1_VA,
        "cave1_len": len(cave1),
        "cave1_margin": s3.CAVE2_VA - (s3.CAVE1_VA + len(cave1)),
        "render_va": s3.CAVE2_VA,
        "render_len": render_len,
        "shim_va": shim_va,
        "shim_len": shim_len,
        "bypass_va": bypass_va,
        "bypass_len": bypass_len,
        "core_va": core_va,
        "core_len": core_len,
        "cave2_len": cave2_len,
        "cave2_end_va": s3.CAVE2_VA + cave2_len,
        "cave2_margin": s3.DESC_TABLE_VA - (s3.CAVE2_VA + cave2_len),
        "cave3_va": CAVE3_VA,
        "cave3_len": len(cave3),
        "cave3_end_va": CAVE3_VA + len(cave3),
        "cave3_margin": CAVE3_LIMIT_VA - (CAVE3_VA + len(cave3)),
        "cave3_syms": cave3_syms,
    }


def update_shim_address() -> int:
    return cave_layout()["shim_va"]


def build_update_hook() -> bytes:
    return struct.pack("<I", s3.encode_arm_branch(HOOK_UPDATE_VA, update_shim_address(), is_bl=True))


def build_live_ai_cave2() -> bytes:
    """Assemble Cave 2 @ 0x53000e00 (OSD renderer + post-update shim + bypass + DSP core)."""
    info = cave_layout()
    render = _build_render_osd(info["render_va"], info["bypass_va"])
    shim_va = info["shim_va"]
    bypass_va = info["bypass_va"]
    core_va = info["core_va"]
    core = _build_live_ai_core(core_va, info["cave3_syms"])

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
    """Apply Step 2D v1.2.17 live-frame AI AF patch non-destructively inside RTOS bounds."""
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
    if not rtos_start <= cave3_off < cave3_off + len(cave3) <= rtos_end:
        raise ValueError("Cave 3 out of RTOS bounds")
    if payload[cave3_off : cave3_off + len(cave3)] != bytes(len(cave3)):
        raise ValueError("Cave 3 destination is not empty zero space")

    check = bytearray(payload)
    manifest = s3.apply_stage3_patch(
        check,
        rtos_payload_offset=rtos_start,
        cave1_builder=build_live_ai_cave1,
        cave2_builder=build_live_ai_cave2,
    )
    check[cave3_off : cave3_off + len(cave3)] = cave3
    check[update_off : update_off + 4] = build_update_hook()

    if payload[:rtos_start] != check[:rtos_start] or payload[rtos_end:] != check[rtos_end:]:
        raise ValueError("Live AI patch writes outside RTOS")

    payload[:] = check
    info = cave_layout()
    manifest.update({
        "stage": "Step 2D v1.2.17: Live-frame continuous CEVA-XM4 AI subject-tracking AF (Option B)",
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
        "layout": {
            k: hex(v) if isinstance(v, int) and "va" in k else v
            for k, v in info.items()
            if k != "cave3_syms"
        },
        "heatmap_out_va": hex(HEATMAP_OUT_VA),
        "heatmap_len": hex(HEATMAP_LEN),
        "work_buf_flush_len": hex(WORK_BUF_FLUSH_LEN),
        "peak_thresh": PEAK_THRESH,
        "area_thresh": AREA_THRESH,
        "lost_hold_cycles": LOST_HOLD_CYCLES,
    })
    return manifest


if __name__ == "__main__":
    info = cave_layout()
    c1 = build_live_ai_cave1()
    c2 = build_live_ai_cave2()
    c3, syms = build_live_ai_cave3()
    print(f"Cave 1: {len(c1)} B @ {hex(s3.CAVE1_VA)}..{hex(s3.CAVE1_VA + len(c1))} (margin {info['cave1_margin']} B)")
    print(f"Cave 2: {len(c2)} B @ {hex(s3.CAVE2_VA)}..{hex(info['cave2_end_va'])} (margin {info['cave2_margin']} B)")
    print(f"Cave 3: {len(c3)} B @ {hex(CAVE3_VA)}..{hex(info['cave3_end_va'])} (margin {info['cave3_margin']} B)")
    print("Cave 3 symbols:", {k: hex(v) for k, v in syms.items()})

    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    dec1 = list(md.disasm(c1, s3.CAVE1_VA))
    dec2 = list(md.disasm(c2, s3.CAVE2_VA))
    dec3 = list(md.disasm(c3, CAVE3_VA))
    assert len(dec1) * 4 == len(c1), f"Cave 1 decode mismatch: {len(dec1)*4} != {len(c1)}"
    assert len(dec2) * 4 == len(c2), f"Cave 2 decode mismatch: {len(dec2)*4} != {len(c2)}"
    assert len(dec3) * 4 == len(c3), f"Cave 3 decode mismatch: {len(dec3)*4} != {len(c3)}"
    print(f"Capstone clean decode: Cave 1 = {len(dec1)} insns, Cave 2 = {len(dec2)} insns, Cave 3 = {len(dec3)} insns")
