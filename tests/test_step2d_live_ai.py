"""Executable A32 Unicorn + container tests for Step 2D v1.2.18 K-3 III tuned AI AF (Option B)."""

import os
from pathlib import Path
import struct
import unittest

from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM
from unicorn import Uc, UC_ARCH_ARM, UC_MODE_ARM, UC_HOOK_CODE
from unicorn.arm_const import (
    UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2,
    UC_ARM_REG_R3, UC_ARM_REG_R4, UC_ARM_REG_R6, UC_ARM_REG_SP,
)

from test_step2c_osd_recovery import CONTINUATION, FD, LIB, RESULT, SP, UPDATE, fw, osd, s3, words
import step2d_live_ai_patcher as lp
import build_step2d_live_ai as builder

FACTORY, SERVICE, SLOT_TABLE, CACHE_REQ_TABLE = 0x15000, 0x15100, 0x16000, 0x16200
# AHIP pointers across RTOS / IMAGE1 / IMAGE2 (Heap[15] @ 0xa0000000..0xe0000000)
AHIP_SLOT_PP = 0x55081578
AHIP_SLOT_VA = 0x68000100
YUV_BASE_VA = 0xA0010000

ORIENT_SVC_VA = 0x55380100
ORIENT_OBJ_VA = 0x55380200
ORIENT_BUF_VA = 0x55380300


def _clamp8(v: int) -> int:
    return 0 if v < 0 else (255 if v > 255 else v)


def bt601_ref(y: int, u_raw: int, v_raw: int) -> tuple[int, int, int]:
    """Exact reference matching Cave 3 integer BT.601 conversion."""
    u = u_raw - 128
    v = v_raw - 128
    r = _clamp8(y + (v + (v >> 1) - (v >> 3)))
    g = _clamp8(y - ((11 * u + 23 * v) >> 5))
    b = _clamp8(y + (u + (u >> 1) + (u >> 2)))
    return r, g, b


def _udiv10_round(num: int, den: int) -> int:
    return 0 if den == 0 else (num + (den >> 1)) // den


def _softmax_other_ref(p: int, a: int, b: int) -> tuple[int, int]:
    if p <= a or p <= b:
        return 999, 0
    da, db = p - a, p - b
    ea = lp.EXP_LUT_BYTES[da] if da <= 15 else 0
    eb = lp.EXP_LUT_BYTES[db] if db <= 15 else 0
    return ea + eb, min(da, db)


def expected_scan_ref(hm: bytes, ref_c: int = 28, ref_r: int = 28):
    """Exact Python reference matching sub_scan_heatmap (Unified Pass 1 + 2-iter Pass 2)."""
    ch = [[], [], []]
    for k in range(3):
        base = k * 0x1C00
        ch[k] = [
            [struct.unpack_from("<h", hm, base + ((r << 6) + c) * 2)[0] for c in range(56)]
            for r in range(56)
        ]

    t1 = [-1, 999, 999, 28, 28, 0]
    t2 = [-1, 999, 999, 28, 28, 0]
    for r in range(56):
        for c in range(56):
            c0, c1, c2 = ch[0][r][c], ch[1][r][c], ch[2][r][c]
            if c1 == c2:
                continue
            if c1 > c2:
                tier, p, a, b, limit, t = 1, c1, c0, c2, lp.CH1_OTHER_MAX, t1
            else:
                tier, p, a, b, limit, t = 2, c2, c0, c1, lp.CH2_OTHER_MAX, t2
            other, margin = _softmax_other_ref(p, a, b)
            if other > limit:
                continue
            t[5] += 1
            dist = abs(c - ref_c) + abs(r - ref_r)
            if (margin, -other, -dist) > (t[0], -t[1], -t[2]):
                t[0], t[1], t[2], t[3], t[4] = margin, other, dist, c, r

    def try_tier(tier: int, t: list[int]):
        primary, otherA, otherB = (1, 0, 2) if tier == 1 else (2, 0, 1)
        radius = lp.LOCAL_RADIUS_CH1 if tier == 1 else lp.LOCAL_RADIUS_CH2
        omax = lp.CH1_OTHER_MAX if tier == 1 else lp.CH2_OTHER_MAX
        mloc = lp.CH1_MIN_LOCAL if tier == 1 else lp.CH2_MIN_LOCAL
        ac, ar = t[3], t[4]
        for _ in range(2):
            sw = swc = swr = lcnt = 0
            min_c, max_c, min_r, max_r = 55, 0, 55, 0
            for r in range(max(0, ar - radius), min(55, ar + radius) + 1):
                r_prev = lcnt
                for c in range(max(0, ac - radius), min(55, ac + radius) + 1):
                    other, _ = _softmax_other_ref(ch[primary][r][c], ch[otherA][r][c], ch[otherB][r][c])
                    if other <= omax:
                        w = 256 - other
                        sw += w
                        swc += w * ((c << 4) + 8)
                        swr += w * ((r << 4) + 8)
                        lcnt += 1
                        if c < min_c:
                            min_c = c
                        if c > max_c:
                            max_c = c
                if lcnt != r_prev:
                    if r < min_r:
                        min_r = r
                    max_r = r
            if lcnt < mloc:
                return None
            uc_q4 = _udiv10_round(swc, sw)
            ur_q4 = _udiv10_round(swr, sw)
            ac, ar = uc_q4 >> 4, ur_q4 >> 4
        prob_pct = _udiv10_round(25600, 256 + t[1])
        return tier, prob_pct, uc_q4, ur_q4, max_c - min_c + 1, max_r - min_r + 1, lcnt

    if t1[5] >= lp.CH1_MIN_LOCAL and t1[1] <= lp.CH1_PEAK_OTHER_MAX:
        res = try_tier(1, t1)
        if res is not None:
            return res
    if t2[5] >= lp.CH2_MIN_LOCAL:
        res = try_tier(2, t2)
        if res is not None:
            return res
    return None


def expected_v1218_box(
    hm: bytes, width: int = 320, height: int = 212, orient: int = 0, ref_c: int = 28, ref_r: int = 28
) -> tuple[int, int, int, int, int]:
    """Return (left, top, size, prob_pct, tier) for a cold-snap detection."""
    res = expected_scan_ref(hm, ref_c=ref_c, ref_r=ref_r)
    assert res is not None
    tier, prob_pct, uc_q4, ur_q4, span_c, span_r, _ = res
    if orient == 0:
        sc, sr, sx, sy = uc_q4, ur_q4, span_c, span_r
    elif orient == 1:
        sc, sr, sx, sy = 896 - ur_q4, uc_q4, span_r, span_c
    elif orient == 2:
        sc, sr, sx, sy = 896 - uc_q4, 896 - ur_q4, span_c, span_r
    elif orient == 3:
        sc, sr, sx, sy = ur_q4, 896 - uc_q4, span_r, span_c
    else:
        raise ValueError(orient)
    cx = (sc * width * 585) >> 11
    cy = (sr * height * 585) >> 11
    raw = (max(sx * width, sy * height) * 73) >> 4
    if tier == 1:
        size = max(lp.MIN_HEAD_BOX_Q8, min(lp.MAX_HEAD_BOX_Q8, raw))
    else:
        raw = (raw * 3) >> 2
        size = max(lp.MIN_BODY_BOX_Q8, min(lp.MAX_BODY_BOX_Q8, raw))
    size = min(size, min(width, height) << 7)
    left = max(0, min((width << 8) - size, cx - (size >> 1)))
    top = max(0, min((height << 8) - size, cy - (size >> 1)))
    return left, top, size, prob_pct, tier


def make_heatmap_blob(ch1_cells=(), ch2_cells=(), bg_val=0) -> bytes:
    """Build a 3-channel 56x56 (stride 64) int16_t heatmap (0x5400 B)."""
    buf = bytearray(lp.HEATMAP_LEN)
    if bg_val != 0:
        for r in range(56):
            for c in range(56):
                off = ((r << 6) + c) << 1
                struct.pack_into("<h", buf, off, bg_val)
    for c, r, val in ch1_cells:
        off = 0x1C00 + (((r << 6) + c) << 1)
        struct.pack_into("<h", buf, off, val)
    for c, r, val in ch2_cells:
        off = 0x3800 + (((r << 6) + c) << 1)
        struct.pack_into("<h", buf, off, val)
    return bytes(buf)


def fill_rect(cells: list, c0: int, r0: int, c1: int, r1: int, val: int) -> None:
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cells.append((c, r, val))


class LiveAiMachine:
    def __init__(
        self,
        *,
        width=320,
        height=212,
        latencies=(1,),
        flush1f_on_reqs=(),
        fail_load_attempts=0,
        fail_start_on_req=None,
        fail_send_on_req=None,
        drop_reqs=(),
        boot_seen=True,
        ahip_active=True,
        include_uv=True,
        orient=0,
    ):
        self.u = Uc(UC_ARCH_ARM, UC_MODE_ARM)
        self.width = width
        self.height = height
        self.latencies = list(latencies)
        self.flush1f_on_reqs = set(flush1f_on_reqs)
        self.fail_load_attempts = fail_load_attempts
        self.load_attempts = 0
        self.fail_start_on_req = fail_start_on_req
        self.fail_send_on_req = fail_send_on_req
        self.drop_reqs = set(drop_reqs)

        self.selected_service = SERVICE
        self.loaded = False
        self.started = False
        self.shot_used_in_boot = False
        self.start_count = 0
        self.send_count = 0
        self.dspsf_flags = 0
        self.frame = 0
        self.locked = True
        self.req_idx = 0
        self.seq = 0
        self.waiting_1f = False
        self.pending_isr = None
        self.events = []
        self.osd_snapshots = []
        self.cache_invals = []
        self.cache_flushes = []
        self.heatmap_provider = lambda req_num: make_heatmap_blob()

        self.service_sites = {
            s3.UNLOCK_MUTEX_VA, UPDATE, s3.GET_DSP_SVC_VA, s3.CREATE_DSP_SVC_VA,
            s3.LOAD_DSP_VA, s3.START_DSP_VA, lp.STOP_DSP_VA, s3.SEND_MSG_VA, s3.WAIT_MSG_VA,
            s3.GET_CACHE_MGR_VA, s3.FLUSH_DCACHE_VA, s3.INVAL_DCACHE_VA,
            lp.OS_EF_WAIT_VA, lp.OS_EF_CLEAR_VA,
        }
        pages = {
            x & ~0xFFF
            for x in self.service_sites | {
                FD, SP - 4, FACTORY, SERVICE, SLOT_TABLE, CACHE_REQ_TABLE, s3.TENSOR_MSG_VA,
                s3.CAVE1_VA, lp.HOOK_UPDATE_VA, s3.HOOK1_RESUME_VA, s3.HOOK1_SINGLETON_VA,
                s3.AHIP_DRV_BASE_VA, s3.AHIP_DRV_BASE_VA + 0x424, AHIP_SLOT_PP, AHIP_SLOT_VA,
                lp.ORIENT_GUARD_VA, lp.ORIENT_SVC_PTR_VA,
                ORIENT_SVC_VA, ORIENT_SVC_VA + 0x518, ORIENT_OBJ_VA, ORIENT_BUF_VA,
            }
        }
        pages.update(range(s3.CAVE2_VA & ~0xFFF, lp.CAVE3_LIMIT_VA + 0x1000, 0x1000))
        for page in pages:
            self.u.mem_map(page, 0x1000)
        self.u.mem_map(s3.TENSOR_IN_VA, 0x10000)
        self.u.mem_map(YUV_BASE_VA, 0x200000)

        # Pre-seed STATUS_VA with non-zero garbage to verify Cave 1 zeroes all 80 bytes
        self.u.mem_write(s3.STATUS_VA, b"\xaa" * lp.STATUS_SIZE)

        c3, _ = lp.build_live_ai_cave3()
        self.u.mem_write(s3.CAVE1_VA, lp.build_live_ai_cave1())
        self.u.mem_write(s3.CAVE2_VA, lp.build_live_ai_cave2())
        self.u.mem_write(lp.CAVE3_VA, c3)
        self.u.mem_write(lp.EXP_LUT_VA, lp.EXP_LUT_BYTES)
        self.u.mem_write(osd.HOOK2_SITE_VA, osd.build_tramp2_payload())
        self.u.mem_write(CONTINUATION, words(
            0xE1A00006,
            s3.encode_arm_branch(CONTINUATION + 4, s3.UNLOCK_MUTEX_VA, is_bl=True),
            0xE1A00004,
        ))
        self.u.mem_write(lp.HOOK_UPDATE_VA, lp.build_update_hook())
        self.u.mem_write(FD + 0x78, words(RESULT))
        self.u.mem_write(FD + 0x38, words(LIB))
        self.u.mem_write(FD + 0xFC, struct.pack("<HH", width, height))

        self.set_orientation(orient, guard_bit=1)
        self.setup_ahip_frame(active=ahip_active, include_uv=include_uv)
        self.u.hook_add(UC_HOOK_CODE, self.hook)
        if boot_seen:
            self.boot_init()

    def set_orientation(self, raw_val: int, guard_bit: int = 1) -> None:
        self.u.mem_write(lp.ORIENT_GUARD_VA, words(guard_bit))
        self.u.mem_write(lp.ORIENT_SVC_PTR_VA, words(ORIENT_SVC_VA))
        self.u.mem_write(ORIENT_SVC_VA + 0x518, words(ORIENT_OBJ_VA))
        self.u.mem_write(ORIENT_OBJ_VA, words(ORIENT_BUF_VA))
        self.u.mem_write(ORIENT_BUF_VA + 0xD8, bytes([raw_val & 0xFF]))

    def disable_orientation_guard(self) -> None:
        self.u.mem_write(lp.ORIENT_GUARD_VA, words(0))

    def setup_ahip_frame(
        self, *, y_bytes=None, u_bytes=None, v_bytes=None, active=True, include_uv=True
    ):
        y_len = self.width * self.height
        uv_len = (self.width // 2) * (self.height // 2)
        y_ptr = YUV_BASE_VA
        u_ptr = (y_ptr + y_len + 7) & ~7
        v_ptr = (u_ptr + uv_len + 7) & ~7
        total_len = ((v_ptr + uv_len - y_ptr) + 0x1F) & ~0x1F

        self.y_ptr = y_ptr
        self.u_ptr = u_ptr if include_uv else 0
        self.v_ptr = v_ptr if include_uv else 0
        self.ahip_total_len = total_len

        self.u.mem_write(y_ptr, y_bytes if y_bytes is not None else bytes([128]) * y_len)
        if include_uv:
            self.u.mem_write(u_ptr, u_bytes if u_bytes is not None else bytes([128]) * uv_len)
            self.u.mem_write(v_ptr, v_bytes if v_bytes is not None else bytes([128]) * uv_len)

        self.u.mem_write(s3.AHIP_DRV_BASE_VA + 0x0C, words(total_len))
        self.u.mem_write(s3.AHIP_DRV_BASE_VA + 0x30, words(2 if active else 0))
        self.u.mem_write(s3.AHIP_DRV_BASE_VA + 0x424, words(AHIP_SLOT_PP))
        self.u.mem_write(AHIP_SLOT_PP, words(AHIP_SLOT_VA))
        self.u.mem_write(AHIP_SLOT_VA + 0x00, words(0x34, y_ptr, self.u_ptr, self.v_ptr))

    def boot_init(self):
        self.u.reg_write(UC_ARM_REG_SP, SP)
        self.u.emu_start(s3.CAVE1_VA, s3.HOOK1_RESUME_VA, count=300)
        assert self.u.reg_read(UC_ARM_REG_PC) == s3.HOOK1_RESUME_VA

    def deliver_pending_isr(self):
        if not self.pending_isr or self.waiting_1f or not self.started:
            return
        deliver_frame, seq, slot_idx, req_num = self.pending_isr
        if self.frame < deliver_frame:
            return
        self.pending_isr = None
        if req_num in self.drop_reqs:
            return
        hm = self.heatmap_provider(req_num)
        assert len(hm) == lp.HEATMAP_LEN
        self.u.mem_write(lp.HEATMAP_OUT_VA, hm)
        slot_va = SLOT_TABLE + slot_idx * 0x2C
        reply_payload = words(0x7000 + req_num) + bytes(0x20)
        self.u.mem_write(slot_va, struct.pack("<BBH", 1, 0, seq) + words(0) + reply_payload)

    def hook(self, u, address, size, data):
        if address not in self.service_sites:
            return
        r0, r1, r2, r3 = (
            u.reg_read(r) for r in (UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3)
        )
        assert (u.reg_read(UC_ARM_REG_SP) & 7) == 0, f"Misaligned SP at {hex(address)}"

        if address == s3.UNLOCK_MUTEX_VA:
            assert self.locked and r0 == FD + 0x48
            self.locked = False
        elif address == UPDATE:
            assert not self.locked and r0 == FD
            fc = struct.unpack("<I", u.mem_read(LIB + 0x30, 4))[0]
            rect = tuple(struct.unpack("<4I", u.mem_read(RESULT, 16)))
            self.osd_snapshots.append((fc, rect))
            self.events.append((self.frame, "osd"))
            u.reg_write(UC_ARM_REG_R0, 0x12345678)
        else:
            assert not self.locked and self.events[-1][0] == self.frame
            names = {
                s3.GET_DSP_SVC_VA: "get",
                s3.CREATE_DSP_SVC_VA: "create",
                s3.LOAD_DSP_VA: "load",
                s3.START_DSP_VA: "start",
                lp.STOP_DSP_VA: "stop",
                s3.SEND_MSG_VA: "send",
                s3.WAIT_MSG_VA: "wait",
                s3.GET_CACHE_MGR_VA: "cache",
                s3.FLUSH_DCACHE_VA: "flush",
                s3.INVAL_DCACHE_VA: "invalidate",
                lp.OS_EF_WAIT_VA: "ef_wait",
                lp.OS_EF_CLEAR_VA: "ef_clear",
            }
            self.events.append((self.frame, names[address]))
            if address == s3.GET_DSP_SVC_VA:
                u.mem_write(FACTORY + 4, words(self.selected_service))
                u.reg_write(UC_ARM_REG_R0, FACTORY)
            elif address == s3.LOAD_DSP_VA:
                if self.load_attempts < self.fail_load_attempts:
                    self.load_attempts += 1
                    u.mem_write(SERVICE + 4, words(2))
                    u.reg_write(UC_ARM_REG_R0, 0)
                else:
                    self.load_attempts += 1
                    self.loaded = True
                    u.mem_write(SERVICE + 4, words(2))
                    u.reg_write(UC_ARM_REG_R0, 1)
            elif address == s3.START_DSP_VA:
                assert self.loaded and not self.started
                idx = self.start_count
                self.start_count += 1
                if self.fail_start_on_req is not None and idx == self.fail_start_on_req:
                    u.reg_write(UC_ARM_REG_R0, 0xFFFFFFFF)
                else:
                    self.started = True
                    self.shot_used_in_boot = False
                    self.dspsf_flags = 0
                    u.mem_write(SERVICE + 4, words(6))
                    u.mem_write(SERVICE + 0x54, words(SLOT_TABLE))
                    u.mem_write(SERVICE + 0x58, words(CACHE_REQ_TABLE))
                    u.mem_write(SLOT_TABLE, bytes(0x108))
                    u.mem_write(CACHE_REQ_TABLE, bytes(0x48))
                    u.reg_write(UC_ARM_REG_R0, 0)
            elif address == lp.STOP_DSP_VA:
                assert r0 == SERVICE and r1 == 100 and self.started
                self.started = False
                self.dspsf_flags = 0
                u.mem_write(SERVICE + 4, words(2))
                u.reg_write(UC_ARM_REG_R0, 0)
            elif address == lp.OS_EF_WAIT_VA:
                matched = self.dspsf_flags & r1
                if r2 != 0:
                    u.mem_write(r2, words(matched))
                u.reg_write(UC_ARM_REG_R0, 1 if matched != 0 else 0)
            elif address == lp.OS_EF_CLEAR_VA:
                self.dspsf_flags &= ~r1
                if (r1 & 0x10000) and self.waiting_1f:
                    self.waiting_1f = False
                u.reg_write(UC_ARM_REG_R0, 0)
            elif address == s3.SEND_MSG_VA:
                assert self.started
                idx = self.send_count
                self.send_count += 1
                if self.fail_send_on_req is not None and idx == self.fail_send_on_req:
                    u.reg_write(UC_ARM_REG_R0, 0xFFFFFFFE)
                else:
                    second_shot = self.shot_used_in_boot
                    self.shot_used_in_boot = True
                    req_num = self.req_idx
                    self.req_idx += 1
                    self.seq = req_num % 6
                    u.mem_write(r1, struct.pack("<H", self.seq))
                    u.reg_write(UC_ARM_REG_R0, 0)
                    if not second_shot:
                        if req_num in self.flush1f_on_reqs:
                            self.u.mem_write(CACHE_REQ_TABLE, words(0, s3.TENSOR_IN_VA, 0x1000))
                            self.dspsf_flags |= 0x10000
                            self.waiting_1f = True
                        lat = self.latencies[req_num % len(self.latencies)]
                        slot_idx = (req_num * 2 + 1) % 6
                        self.pending_isr = (self.frame + 1 + lat, self.seq, slot_idx, req_num)
                        if lat == 0 and not self.waiting_1f:
                            self.deliver_pending_isr()
            elif address == s3.WAIT_MSG_VA:
                slots_raw = u.mem_read(SLOT_TABLE, 0x108)
                matched_off = None
                for off in range(0, 0x108, 0x2C):
                    occ, _, s_seq = struct.unpack_from("<BBH", slots_raw, off)
                    if occ != 0 and s_seq == r1:
                        matched_off = off
                        break
                assert matched_off is not None, "WaitMessage called before matching reply slot ready!"
                u.mem_write(r3 + 4, bytes(slots_raw[matched_off + 8 : matched_off + 0x2C]))
                u.mem_write(SLOT_TABLE + matched_off, b"\x00")
                u.reg_write(UC_ARM_REG_R0, 0)
            elif address == s3.INVAL_DCACHE_VA:
                self.cache_invals.append((r1, r2))
            elif address == s3.FLUSH_DCACHE_VA:
                self.cache_flushes.append((r1, r2))
        u.reg_write(UC_ARM_REG_PC, u.reg_read(UC_ARM_REG_LR))

    def run_frames(self, count: int):
        for _ in range(count):
            self.frame += 1
            self.deliver_pending_isr()
            self.locked = True
            self.u.reg_write(UC_ARM_REG_SP, SP)
            self.u.reg_write(UC_ARM_REG_R4, FD)
            self.u.reg_write(UC_ARM_REG_R6, FD + 0x48)
            self.u.emu_start(osd.HOOK2_SITE_VA, lp.HOOK_UPDATE_VA + 4, count=300000)
            assert self.u.reg_read(UC_ARM_REG_PC) == lp.HOOK_UPDATE_VA + 4
            assert self.u.reg_read(UC_ARM_REG_SP) == SP
            assert self.u.reg_read(UC_ARM_REG_R0) == 0x12345678

    def run_until_cycles(self, target: int, max_frames: int = 40):
        for _ in range(max_frames):
            if self.word(0x14) >= target:
                break
            self.run_frames(1)
        assert self.word(0x14) == target, f"cycles_ok={self.word(0x14)} != {target}"
        self.run_frames(1)  # next frame commits latched box to OSD

    def calls(self, name: str):
        return [f for f, n in self.events if n == name]

    def word(self, offset: int) -> int:
        return struct.unpack("<I", self.u.mem_read(s3.STATUS_VA + offset, 4))[0]


class TestStep2DLiveAi(unittest.TestCase):
    def test_capstone_clean_decode_and_bounds(self):
        info = lp.cave_layout()
        self.assertGreaterEqual(info["cave1_margin"], 0)
        self.assertGreaterEqual(info["cave2_margin"], 0)
        self.assertGreaterEqual(info["cave3_margin"], 0)
        md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
        c1 = lp.build_live_ai_cave1()
        c2 = lp.build_live_ai_cave2()
        c3, _ = lp.build_live_ai_cave3()
        dec1 = list(md.disasm(c1, s3.CAVE1_VA))
        dec2 = list(md.disasm(c2, s3.CAVE2_VA))
        dec3 = list(md.disasm(c3, lp.CAVE3_VA))
        self.assertEqual(len(dec1) * 4, len(c1))
        self.assertEqual(len(dec2) * 4, len(c2))
        self.assertEqual(len(dec3) * 4, len(c3))
        # Verify ZERO UDIV/SDIV hardware dependency in any cave
        self.assertFalse(any("div" in ins.mnemonic for ins in dec1 + dec2 + dec3))
        # Verify all 8 DDR range guards in Cave 3 (5 AHIP + 3 gravity orientation)
        ddr_cmps = [ins for ins in dec3 if ins.mnemonic == "cmp" and ins.op_str == "r2, #0xa0000000"]
        self.assertEqual(len(ddr_cmps), 8)

    def test_cave1_zeroes_full_80_bytes_and_publishes_cookie(self):
        m = LiveAiMachine(boot_seen=True)
        for off in range(0, lp.STATUS_SIZE, 4):
            expected = lp.LIVE_AI_COOKIE if off == 0x18 else 0
            self.assertEqual(m.word(off), expected, f"offset {hex(off)} mismatch")

    def test_bt601_yuv420_to_56x56_rgb_downsampler_and_clamping(self):
        w, h = 320, 212
        y = bytearray(w * h)
        u = bytearray((w // 2) * (h // 2))
        v = bytearray((w // 2) * (h // 2))
        quads = {
            (0, 0): (100, 80, 240),
            (1, 0): (10, 250, 250),
            (0, 1): (245, 220, 220),
            (1, 1): (90, 230, 90),
        }
        for row in range(h):
            for col in range(w):
                qy, qx = (1 if row >= h // 2 else 0), (1 if col >= w // 2 else 0)
                y[row * w + col] = quads[(qx, qy)][0]
        for row in range(h // 2):
            for col in range(w // 2):
                qy, qx = (1 if row >= h // 4 else 0), (1 if col >= w // 4 else 0)
                u[row * (w // 2) + col] = quads[(qx, qy)][1]
                v[row * (w // 2) + col] = quads[(qx, qy)][2]

        m = LiveAiMachine(width=w, height=h, latencies=(1,), orient=0)
        m.setup_ahip_frame(y_bytes=bytes(y), u_bytes=bytes(u), v_bytes=bytes(v))
        m.u.mem_write(lp.HEATMAP_OUT_VA, b"\xff" * lp.HEATMAP_LEN)
        m.run_frames(21)

        self.assertEqual(len(m.calls("send")), 1)
        self.assertIn((m.y_ptr, m.ahip_total_len), m.cache_invals)
        self.assertIn((s3.TENSOR_IN_VA, lp.WORK_BUF_FLUSH_LEN), m.cache_flushes)
        self.assertEqual(bytes(m.u.mem_read(lp.HEATMAP_OUT_VA, lp.HEATMAP_LEN)), bytes(lp.HEATMAP_LEN))

        tensor = m.u.mem_read(s3.TENSOR_IN_VA, 0x2A00)
        for (row, col), q in (((10, 10), (0, 0)), ((10, 45), (1, 0)), ((45, 10), (0, 1)), ((45, 45), (1, 1))):
            exp_r, exp_g, exp_b = bt601_ref(*quads[q])
            got = (tensor[row * 64 + col], tensor[0x0E00 + row * 64 + col], tensor[0x1C00 + row * 64 + col])
            self.assertEqual(got, (exp_r, exp_g, exp_b), f"pixel ({row},{col}) mismatch")
        for p_off in (0, 0x0E00, 0x1C00):
            for row in range(56):
                self.assertEqual(bytes(tensor[p_off + row * 64 + 56 : p_off + row * 64 + 64]), bytes(8))

    def test_ahip_guards_orientation_guards_and_grayscale_fallback(self):
        # 1. AHIP inactive ([+0x30] != 2): skips cleanly
        m = LiveAiMachine(ahip_active=False)
        m.run_frames(25)
        self.assertEqual(len(m.calls("load")), 1)
        self.assertEqual(len(m.calls("start")), 0)

        # 2. Out-of-range AHIP slot_pp / slot / Y_ptr
        for bad_slot_pp, bad_slot, bad_y in (
            (0x0, AHIP_SLOT_VA, YUV_BASE_VA),
            (0x2985B000, AHIP_SLOT_VA, YUV_BASE_VA),
            (0x3FFFFFFF, AHIP_SLOT_VA, YUV_BASE_VA),
            (0xE0000000, AHIP_SLOT_VA, YUV_BASE_VA),
            (AHIP_SLOT_PP, 0xE0000000, YUV_BASE_VA),
            (AHIP_SLOT_PP, 0x20000000, YUV_BASE_VA),
            (AHIP_SLOT_PP, AHIP_SLOT_VA, 0x0),
            (AHIP_SLOT_PP, AHIP_SLOT_VA, 0xE0000000),
            (AHIP_SLOT_PP, AHIP_SLOT_VA, 0x3FFFFFFF),
        ):
            m = LiveAiMachine()
            m.u.mem_write(s3.AHIP_DRV_BASE_VA + 0x424, words(bad_slot_pp))
            if bad_slot_pp == AHIP_SLOT_PP:
                m.u.mem_write(AHIP_SLOT_PP, words(bad_slot))
            if bad_slot == AHIP_SLOT_VA:
                m.u.mem_write(AHIP_SLOT_VA + 0x04, words(bad_y))
            m.run_frames(24)
            self.assertEqual(len(m.calls("start")), 0)

        # 3. Corrupted orientation pointers fallback cleanly without crashing
        for bad_ptr_addr, bad_val in (
            (lp.ORIENT_SVC_PTR_VA, 0x0),
            (lp.ORIENT_SVC_PTR_VA, 0xE0000000),
            (ORIENT_SVC_VA + 0x518, 0x20000000),
            (ORIENT_OBJ_VA, 0xFFFFFFFF),
        ):
            m = LiveAiMachine(orient=1)
            m.u.mem_write(bad_ptr_addr, words(bad_val))
            m.run_frames(21)
            self.assertEqual(len(m.calls("send")), 1)
            self.assertEqual(m.word(0x44), 0)

        # 4. NULL / out-of-range U/V pointers fall back to grayscale R==G==B==Y
        m = LiveAiMachine(include_uv=False)
        m.setup_ahip_frame(y_bytes=bytes([77]) * (320 * 212), include_uv=False)
        m.u.mem_write(AHIP_SLOT_VA + 0x08, words(0xE0000000))
        m.u.mem_write(AHIP_SLOT_VA + 0x0C, words(0x20000000))
        m.run_frames(21)
        tensor = m.u.mem_read(s3.TENSOR_IN_VA, 0x2A00)
        self.assertEqual((tensor[0], tensor[0x0E00], tensor[0x1C00]), (77, 77, 77))

    def test_k3_exact_softmax_thresholds_and_clutter_rejection(self):
        # A. Ch1 P = 48.1% (c1=2, c0=0, c2=-1 => e0+e2=276 > 255) REJECTED even with 16 cells
        cells_48 = []
        fill_rect(cells_48, 26, 26, 29, 29, 2)
        ch2_sub = [(c, r, -1) for c, r, _ in cells_48]
        blob_ch1_48 = make_heatmap_blob(ch1_cells=cells_48, ch2_cells=ch2_sub)

        # B. Ch1 P = 50.7% (c1=2, c0=0, c2=-2 => 249 <= 255): fails peak >= 60% gate (best_other=249 > 170)
        ch2_m2 = [(c, r, -2) for c, r, _ in cells_48]
        blob_ch1_50_no_peak = make_heatmap_blob(ch1_cells=cells_48, ch2_cells=ch2_m2)

        # C. Ch1 scattered 2 cells (< CH1_MIN_LOCAL=3) with high peak REJECTED
        blob_ch1_2cells = make_heatmap_blob(ch1_cells=[(28, 28, 10), (29, 28, 10)])

        # D. Ch1 valid: 50.7% perimeter (2 cells) + 1 peak >= 60% (c1=4, c0=0, c2=-2 => 94+27=121 <= 170) => PASS Tier 1
        ch1_valid = [(27, 28, 2), (29, 28, 2), (28, 28, 4)]
        ch2_for_valid = [(27, 28, -2), (29, 28, -2), (28, 28, -2)]
        blob_ch1_valid = make_heatmap_blob(ch1_cells=ch1_valid, ch2_cells=ch2_for_valid)

        # E. Ch2 P = 73.0% (c2=4, c0=0, c1=-20 => 94 > 85) with 12 cells REJECTED (crushes complex clutter!)
        cells_ch2_73 = []
        fill_rect(cells_ch2_73, 20, 20, 23, 22, 4)
        ch1_m20 = [(c, r, -20) for c, r, _ in cells_ch2_73]
        blob_ch2_73 = make_heatmap_blob(ch1_cells=ch1_m20, ch2_cells=cells_ch2_73)

        # F. Ch2 P = 77.8% (c2=5, c0=0, c1=-20 => 73 <= 85) with ONLY 5 cells (< 6) REJECTED
        cells_ch2_5 = [(20 + i, 20, 5) for i in range(5)]
        blob_ch2_5 = make_heatmap_blob(ch1_cells=[(c, r, -20) for c, r, _ in cells_ch2_5], ch2_cells=cells_ch2_5)

        # G. Ch2 P = 77.8% (c2=5, c0=0, c1=-20) with 6 cells (>= 6) PASSES Tier 2!
        cells_ch2_6 = []
        fill_rect(cells_ch2_6, 20, 20, 22, 21, 5)
        blob_ch2_6 = make_heatmap_blob(ch1_cells=[(c, r, -20) for c, r, _ in cells_ch2_6], ch2_cells=cells_ch2_6)

        # Verify rejected blobs keep box completely hidden (Option B) while DSP keeps running
        for name, bad_blob in (
            ("ch1_48%", blob_ch1_48),
            ("ch1_50%_no_60%_peak", blob_ch1_50_no_peak),
            ("ch1_2cells", blob_ch1_2cells),
            ("ch2_73%", blob_ch2_73),
            ("ch2_5cells", blob_ch2_5),
        ):
            with self.subTest(rejected=name):
                self.assertIsNone(expected_scan_ref(bad_blob))
                m = LiveAiMachine(latencies=(1,))
                m.heatmap_provider = lambda _, b=bad_blob: b
                m.run_until_cycles(3)
                self.assertEqual(m.word(0x38), 0)
                self.assertEqual(m.word(0x24), 0)
                self.assertTrue(all(fc == 0 for fc, _ in m.osd_snapshots))

        # Verify valid Ch1 borderline (3 cells, peak>=60%, neighbors 50.7%) acquires on 2nd cycle as Tier 1
        m1 = LiveAiMachine(latencies=(1,))
        m1.heatmap_provider = lambda _: blob_ch1_valid
        m1.run_until_cycles(2)
        exp_l, exp_t, exp_s, exp_p, exp_tier = expected_v1218_box(blob_ch1_valid)
        self.assertEqual(exp_tier, 1)
        self.assertEqual(m1.word(0x38), (1 << 16) | exp_p)
        self.assertEqual(m1.word(0x24), 1)
        self.assertEqual(m1.osd_snapshots[-1], (1, (exp_l, exp_t, exp_s, exp_s)))

        # Verify valid Ch2 77.8% x 6 cells acquires on 2nd cycle as Tier 2
        m2 = LiveAiMachine(latencies=(1,))
        m2.heatmap_provider = lambda _: blob_ch2_6
        m2.run_until_cycles(2)
        exp_l, exp_t, exp_s, exp_p, exp_tier = expected_v1218_box(blob_ch2_6)
        self.assertEqual(exp_tier, 2)
        self.assertEqual(m2.word(0x38), (2 << 16) | exp_p)
        self.assertEqual(m2.osd_snapshots[-1], (1, (exp_l, exp_t, exp_s, exp_s)))

    def test_strict_ch1_head_priority_blocks_stronger_ch2_body(self):
        ch1 = []
        ch2 = []
        fill_rect(ch1, 13, 13, 15, 15, 8)   # 3x3 head (~88% P(Ch1)) at (14, 14)
        fill_rect(ch2, 34, 34, 43, 43, 20)  # Huge stronger 10x10 body (~99%) at (38, 38)
        blob = make_heatmap_blob(ch1_cells=ch1, ch2_cells=ch2)

        for width, height in ((320, 212), (320, 240), (320, 180), (240, 240)):
            with self.subTest(dims=(width, height)):
                m = LiveAiMachine(width=width, height=height, latencies=(1,))
                m.heatmap_provider = lambda _: blob
                m.run_until_cycles(2)
                exp_l, exp_t, exp_s, exp_p, exp_tier = expected_v1218_box(blob, width=width, height=height)
                self.assertEqual(exp_tier, 1)
                self.assertLessEqual(exp_s, lp.MAX_HEAD_BOX_Q8)
                self.assertEqual(m.word(0x38), (1 << 16) | exp_p)
                fc, rect = m.osd_snapshots[-1]
                self.assertEqual(fc, 1)
                self.assertEqual(rect, (exp_l, exp_t, exp_s, exp_s))
                # Box center MUST be in top-left quadrant (Ch1 head), nowhere near Ch2 (38, 38)!
                self.assertLess(rect[0] + (rect[2] >> 1), (width << 8) // 2)
                self.assertLess(rect[1] + (rect[3] >> 1), (height << 8) // 2)

    def test_plateau_centroid_eliminates_top_left_drift_and_rejects_second_subject(self):
        # Uniform saturated 8x8 Ch1 plateau at [10..17] x [12..19] PLUS distant 2nd face at [42..47] x [40..45]
        ch1 = []
        fill_rect(ch1, 10, 12, 17, 19, 20)
        fill_rect(ch1, 42, 40, 47, 45, 18)
        blob = make_heatmap_blob(ch1_cells=ch1)

        m = LiveAiMachine(width=320, height=212, latencies=(1,))
        m.heatmap_provider = lambda _: blob
        m.run_until_cycles(2)
        exp_l, exp_t, exp_s, _, _ = expected_v1218_box(blob)
        fc, rect = m.osd_snapshots[-1]
        self.assertEqual(fc, 1)
        self.assertEqual(rect, (exp_l, exp_t, exp_s, exp_s))

        # Geometric center of [10..17] x [12..19] is (14.0, 16.0) / 56 -> (80 px, 60.5 px) in Q8
        # Verify NOT stuck at top-left corner (10, 12) and NOT pulled toward 2nd face (44, 42)
        cx_px = (rect[0] + (rect[2] >> 1)) / 256.0
        cy_px = (rect[1] + (rect[3] >> 1)) / 256.0
        self.assertAlmostEqual(cx_px, 14.0 * 320 / 56, delta=2.0)
        self.assertAlmostEqual(cy_px, 16.0 * 212 / 56, delta=2.0)

    def test_portrait_and_180_orientation_rotation_and_inverse_mapping(self):
        w, h = 320, 212
        # Place 4 unique brightness markers in TL(20), TR(80), BL(150), BR(220) of sensor Y plane
        y = bytearray(w * h)
        for r in range(h):
            for c in range(w):
                if r < h // 2 and c < w // 2:
                    y[r * w + c] = 20
                elif r < h // 2:
                    y[r * w + c] = 80
                elif c < w // 2:
                    y[r * w + c] = 150
                else:
                    y[r * w + c] = 220

        # Head placed in CNN top-center (cnn_c = 27..29, cnn_r = 7..9)
        ch1 = []
        fill_rect(ch1, 27, 7, 29, 9, 10)
        head_blob = make_heatmap_blob(ch1_cells=ch1)

        # Also test tilt orientation 4 ('TU') preserves previous orientation 1 ('L'),
        # and guard_bit=0 falls back cleanly to 0 ('U')
        seq = [(0, 0), (1, 1), (4, 1), (2, 2), (3, 3), (None, 0)]
        for raw_orient, eff_orient in seq:
            with self.subTest(raw=raw_orient, eff=eff_orient):
                init_orient = 1 if raw_orient in (4, None) else raw_orient
                m = LiveAiMachine(width=w, height=h, latencies=(1,), orient=init_orient)
                m.setup_ahip_frame(y_bytes=bytes(y), include_uv=False)
                m.heatmap_provider = lambda _: head_blob
                if raw_orient == 4:
                    m.run_until_cycles(1)
                    m.set_orientation(4)
                    m.run_until_cycles(2)
                elif raw_orient is None:
                    m.disable_orientation_guard()
                    m.run_until_cycles(2)
                else:
                    m.run_until_cycles(2)

                tensor = m.u.mem_read(s3.TENSOR_IN_VA, 0x2A00)
                expected_tl_y = {0: 20, 1: 80, 2: 220, 3: 150}[eff_orient]
                self.assertEqual(tensor[2 * 64 + 2], expected_tl_y)
                for p_off in (0, 0x0E00, 0x1C00):
                    for r in range(56):
                        self.assertEqual(bytes(tensor[p_off + r * 64 + 56 : p_off + r * 64 + 64]), bytes(8))

                self.assertEqual(m.word(0x44), eff_orient)
                exp_l, exp_t, exp_s, _, _ = expected_v1218_box(head_blob, width=w, height=h, orient=eff_orient)
                fc, rect = m.osd_snapshots[-1]
                self.assertEqual(fc, 1)
                self.assertEqual(rect, (exp_l, exp_t, exp_s, exp_s))
                cx = rect[0] + (rect[2] >> 1)
                cy = rect[1] + (rect[3] >> 1)
                if eff_orient == 0:
                    self.assertLess(cy, (h << 8) // 4)
                elif eff_orient == 1:
                    self.assertGreater(cx, (w << 8) * 3 // 4)
                elif eff_orient == 2:
                    self.assertGreater(cy, (h << 8) * 3 // 4)
                elif eff_orient == 3:
                    self.assertLess(cx, (w << 8) // 4)

    def test_2_cycle_cold_debounce_deadband_and_smooth_ema(self):
        m = LiveAiMachine(latencies=(1,))

        # blob_a: 3x3 head at (12, 12)
        c_a = []
        fill_rect(c_a, 11, 11, 13, 13, 8)
        blob_a = make_heatmap_blob(ch1_cells=c_a)

        # blob_a_jitter: shift col 11..13 weight slightly -> sub-pixel <= 1 px shift (< 2.5 px deadband)
        c_aj = [(11, 11, 7), (12, 11, 8), (13, 11, 8),
                (11, 12, 7), (12, 12, 8), (13, 12, 8),
                (11, 13, 7), (12, 13, 8), (13, 13, 8)]
        blob_aj = make_heatmap_blob(ch1_cells=c_aj)

        # blob_b: big real shift to (40, 40)
        c_b = []
        fill_rect(c_b, 39, 39, 41, 41, 8)
        blob_b = make_heatmap_blob(ch1_cells=c_b)

        # blob_c: reacquire at (20, 30)
        c_c = []
        fill_rect(c_c, 19, 29, 21, 31, 8)
        blob_c = make_heatmap_blob(ch1_cells=c_c)

        empty = make_heatmap_blob()
        seq_maps = [blob_a, blob_a, blob_aj, blob_b, empty, empty, empty, empty, blob_c, blob_c]
        m.heatmap_provider = lambda req: seq_maps[min(req, len(seq_maps) - 1)]

        # Cycle 0: cold 1st hit -> MUST stay hidden (no 1-frame flash!)
        m.run_until_cycles(1)
        self.assertEqual(m.word(0x48), 1)
        self.assertEqual(m.word(0x24), 0)
        self.assertEqual(m.osd_snapshots[-1][0], 0)

        # Cycle 1: 2nd consecutive hit -> acquires + snaps immediately to blob_a!
        m.run_until_cycles(2)
        l0, t0, s0, _, _ = expected_v1218_box(blob_a)
        self.assertEqual(m.word(0x24), 1)
        self.assertEqual(m.osd_snapshots[-1], (1, (l0, t0, s0, s0)))

        # Cycle 2: micro-jitter within 2.5 px deadband -> box MUST remain bit-for-bit identical!
        m.run_until_cycles(3)
        self.assertEqual(m.osd_snapshots[-1], (1, (l0, t0, s0, s0)))

        # Cycle 3: big motion -> smooth 3:1 center EMA (size unchanged)
        m.run_until_cycles(4)
        lb, tb, sb_new, _, _ = expected_v1218_box(blob_b)
        prev_cx = l0 + (s0 >> 1)
        prev_cy = t0 + (s0 >> 1)
        new_cx = lb + (sb_new >> 1)
        new_cy = tb + (sb_new >> 1)
        ema_cx = (3 * prev_cx + new_cx + 2) >> 2
        ema_cy = (3 * prev_cy + new_cy + 2) >> 2
        ema_l = max(0, min((320 << 8) - s0, ema_cx - (s0 >> 1)))
        ema_t = max(0, min((212 << 8) - s0, ema_cy - (s0 >> 1)))
        self.assertEqual(m.osd_snapshots[-1], (1, (ema_l, ema_t, s0, s0)))

        # Cycles 4, 5, 6: 3 empty cycles -> held
        m.run_until_cycles(7)
        self.assertEqual(m.word(0x1C), 3)
        self.assertEqual(m.word(0x24), 1)
        self.assertEqual(m.osd_snapshots[-1][0], 1)

        # Cycle 7: 4th empty cycle -> Option B hides box
        m.run_until_cycles(8)
        self.assertEqual(m.word(0x1C), 4)
        self.assertEqual(m.word(0x24), 0)
        self.assertEqual(m.osd_snapshots[-1][0], 0)

        # Cycle 8: 1st hit at blob_c -> cold debounce keeps hidden
        m.run_until_cycles(9)
        self.assertEqual(m.word(0x24), 0)
        self.assertEqual(m.osd_snapshots[-1][0], 0)

        # Cycle 9: 2nd hit at blob_c -> snaps to blob_c!
        m.run_until_cycles(10)
        lc, tc, sc, _, _ = expected_v1218_box(blob_c)
        self.assertEqual(m.word(0x24), 1)
        self.assertEqual(m.osd_snapshots[-1], (1, (lc, tc, sc, sc)))

    def test_fault_tolerance_load_retry_1f_timeout_and_transient_errors(self):
        m = LiveAiMachine(
            latencies=(1,),
            fail_load_attempts=2,
            flush1f_on_reqs=(0,),
            drop_reqs=(1,),
            fail_start_on_req=2,
            fail_send_on_req=3,
        )
        ch1 = []
        fill_rect(ch1, 27, 27, 29, 29, 10)
        blob = make_heatmap_blob(ch1_cells=ch1)
        m.heatmap_provider = lambda _: blob

        m.run_frames(83)
        self.assertEqual(m.word(0x00), 0)
        self.assertEqual(len(m.calls("start")), 0)

        m.run_frames(65)
        expected_stops = len(m.calls("start")) - 1 - (1 if m.started else 0)
        self.assertEqual(len(m.calls("stop")), expected_stops)
        self.assertGreaterEqual(m.word(0x14), 4)
        self.assertEqual(m.word(0x24), 1)
        self.assertGreaterEqual(len(m.calls("ef_clear")), 1)
        self.assertEqual(m.osd_snapshots[-1][0], 1)

    def test_packaged_v1218_binary_roundtrip_and_unicorn_end_to_end(self):
        stock_path = Path(os.environ.get(
            "GR4_V120_STOCK",
            r"d:\Ricoh GR4 firmware Mod\firmware\official\v1.20\fwdc248b.bin",
        ))
        if not stock_path.is_file():
            self.skipTest("Official GR IV v1.20 stock firmware not present")

        image, manifest = builder.build_with_manifest(stock_path.read_bytes())
        self.assertEqual(manifest["outer_word_sum"], 0)
        self.assertEqual(manifest["inner_word_sum"], 0)

        payload, _, consumed = fw.decode_container(image)
        fw.verify_container(image, payload, consumed)

        info = lp.cave_layout()
        for orient in (0, 1):
            with self.subTest(orient=orient):
                m = LiveAiMachine(latencies=(1, 2), flush1f_on_reqs=(0,), boot_seen=False, orient=orient)
                for va, length in (
                    (s3.CAVE1_VA, info["cave1_len"]),
                    (s3.HOOK1_SITE_VA, 8),
                    (s3.CAVE2_VA, info["cave2_len"]),
                    (osd.HOOK2_SITE_VA, 8),
                    (lp.CAVE3_VA, info["cave3_len"]),
                    (lp.EXP_LUT_VA, info["exp_lut_len"]),
                    (lp.HOOK_UPDATE_VA, 4),
                ):
                    off = s3.RTOS_PAYLOAD_START + va - s3.GR4_BASE
                    m.u.mem_write(va, payload[off : off + length])

                ch1 = []
                fill_rect(ch1, 17, 21, 19, 23, 12)
                blob = make_heatmap_blob(ch1_cells=ch1)
                m.heatmap_provider = lambda _: blob

                m.boot_init()
                m.run_frames(40)
                self.assertGreaterEqual(m.word(0x14), 3)
                exp_l, exp_t, exp_s, _, _ = expected_v1218_box(blob, orient=orient)
                self.assertEqual(m.osd_snapshots[-1], (1, (exp_l, exp_t, exp_s, exp_s)))


if __name__ == "__main__":
    unittest.main()
