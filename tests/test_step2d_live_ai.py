"""Executable A32 Unicorn + container tests for Step 2D v1.2.17 live-frame AI AF (Option B)."""

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
# Place AHIP slot pointer/struct in RTOS/IMAGE1 DDR and YUV420 planes in authentic Heap[15] IMAGE2 [0xa0000000..0xe0000000)
AHIP_SLOT_PP = 0x55081578
AHIP_SLOT_VA = 0x68000100
YUV_BASE_VA = 0xA0010000


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


def expected_q8_box(
    col: int, row: int, area_cnt: int, width: int = 320, height: int = 212
) -> tuple[int, int, int]:
    cx = ((2 * col + 1) * width * 73) >> 5
    cy = ((2 * row + 1) * height * 73) >> 5
    size = min(lp.MAX_BOX_Q8, max(lp.MIN_BOX_Q8, lp.MIN_BOX_Q8 + (area_cnt << 5)))
    size = min(size, min(width, height) << 7)
    left = max(0, min((width << 8) - size, cx - (size >> 1)))
    top = max(0, min((height << 8) - size, cy - (size >> 1)))
    return left, top, size


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
        self.osd_snapshots = []  # (face_count, rect_tuple)
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
            }
        }
        pages.update(range(s3.CAVE2_VA & ~0xFFF, lp.CAVE3_LIMIT_VA + 0x1000, 0x1000))
        for page in pages:
            self.u.mem_map(page, 0x1000)
        self.u.mem_map(s3.TENSOR_IN_VA, 0x10000)
        self.u.mem_map(YUV_BASE_VA, 0x200000)  # 2 MB for live YUV420 up to 1280x1280

        # Pre-seed STATUS_VA with non-zero garbage to verify Cave 1 zeroes all 64 bytes
        self.u.mem_write(s3.STATUS_VA, b"\xaa" * 0x40)

        c3, _ = lp.build_live_ai_cave3()
        self.u.mem_write(s3.CAVE1_VA, lp.build_live_ai_cave1())
        self.u.mem_write(s3.CAVE2_VA, lp.build_live_ai_cave2())
        self.u.mem_write(lp.CAVE3_VA, c3)
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

        self.setup_ahip_frame(active=ahip_active, include_uv=include_uv)
        self.u.hook_add(UC_HOOK_CODE, self.hook)
        if boot_seen:
            self.boot_init()

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
        # Verify 8-byte AAPCS stack alignment at every firmware call boundary!
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
                # Count each C/I pair: fail_load_attempts fails individual LoadDsp calls
                if self.load_attempts < self.fail_load_attempts:
                    self.load_attempts += 1
                    u.mem_write(SERVICE + 4, words(2))  # mimic stock bug leaving false LOADED bit
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
            self.u.emu_start(osd.HOOK2_SITE_VA, lp.HOOK_UPDATE_VA + 4, count=250000)
            assert self.u.reg_read(UC_ARM_REG_PC) == lp.HOOK_UPDATE_VA + 4
            assert self.u.reg_read(UC_ARM_REG_SP) == SP
            assert self.u.reg_read(UC_ARM_REG_R0) == 0x12345678

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
        self.assertEqual(len(list(md.disasm(c1, s3.CAVE1_VA))) * 4, len(c1))
        self.assertEqual(len(list(md.disasm(c2, s3.CAVE2_VA))) * 4, len(c2))
        dec3 = list(md.disasm(c3, lp.CAVE3_VA))
        self.assertEqual(len(dec3) * 4, len(c3))
        ddr_cmps = [ins for ins in dec3 if ins.mnemonic == "cmp" and ins.op_str == "r2, #0xa0000000"]
        self.assertEqual(len(ddr_cmps), 5)

    def test_cave1_zeroes_full_64_bytes_and_publishes_cookie(self):
        m = LiveAiMachine(boot_seen=True)
        for off in range(0, 0x40, 4):
            expected = lp.LIVE_AI_COOKIE if off == 0x18 else 0
            self.assertEqual(m.word(off), expected, f"offset {hex(off)} mismatch")

    def test_bt601_yuv420_to_56x56_rgb_downsampler_and_clamping(self):
        w, h = 320, 212
        y = bytearray(w * h)
        u = bytearray((w // 2) * (h // 2))
        v = bytearray((w // 2) * (h // 2))
        # 4 quadrants:
        # TL: vivid red (Y=100, U=80, V=240)
        # TR: dark clamped (Y=10, U=250, V=250) -> tests negative G clamp to 0 & bright R/B
        # BL: bright clamped (Y=245, U=220, V=220) -> tests >255 R/B clamp to 255
        # BR: vivid blue (Y=90, U=230, V=90)
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

        m = LiveAiMachine(width=w, height=h, latencies=(1,))
        m.setup_ahip_frame(y_bytes=bytes(y), u_bytes=bytes(u), v_bytes=bytes(v))
        # Pre-fill heatmap with stale garbage to ensure downsampler zeroes it
        m.u.mem_write(lp.HEATMAP_OUT_VA, b"\xff" * lp.HEATMAP_LEN)
        m.run_frames(21)  # frame 20 loads, frame 21 samples & sends!

        self.assertEqual(len(m.calls("send")), 1)
        self.assertIn((m.y_ptr, m.ahip_total_len), m.cache_invals)
        self.assertIn((s3.TENSOR_IN_VA, lp.WORK_BUF_FLUSH_LEN), m.cache_flushes)
        self.assertEqual(bytes(m.u.mem_read(lp.HEATMAP_OUT_VA, lp.HEATMAP_LEN)), bytes(lp.HEATMAP_LEN))

        tensor = m.u.mem_read(s3.TENSOR_IN_VA, 0x2A00)
        for (row, col), q in (((10, 10), (0, 0)), ((10, 45), (1, 0)), ((45, 10), (0, 1)), ((45, 45), (1, 1))):
            exp_r, exp_g, exp_b = bt601_ref(*quads[q])
            got = (tensor[row * 64 + col], tensor[0x0E00 + row * 64 + col], tensor[0x1C00 + row * 64 + col])
            self.assertEqual(got, (exp_r, exp_g, exp_b), f"pixel ({row},{col}) mismatch")
        # Verify cols 56..63 zeroed across all 56 rows & 3 planes
        for p_off in (0, 0x0E00, 0x1C00):
            for row in range(56):
                pad = bytes(tensor[p_off + row * 64 + 56 : p_off + row * 64 + 64])
                self.assertEqual(pad, bytes(8))

    def test_ahip_guards_and_grayscale_fallback(self):
        # 1. AHIP inactive ([+0x30] != 2): skips cleanly without starting DSP
        m = LiveAiMachine(ahip_active=False)
        m.run_frames(25)
        self.assertEqual(len(m.calls("load")), 1)
        self.assertEqual(len(m.calls("start")), 0)

        # 2. Out-of-range slot_pp / slot / Y_ptr (<0x40000000 or >=0xe0000000): skips cleanly without crashing
        for bad_slot_pp, bad_slot, bad_y in (
            (0x0, AHIP_SLOT_VA, YUV_BASE_VA),
            (0x2985B000, AHIP_SLOT_VA, YUV_BASE_VA),  # MMIO
            (0x3FFFFFFF, AHIP_SLOT_VA, YUV_BASE_VA),
            (0xE0000000, AHIP_SLOT_VA, YUV_BASE_VA),
            (0xFFFFFFFF, AHIP_SLOT_VA, YUV_BASE_VA),
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

        # 2b. Verify accepted RTOS/IMAGE1/IMAGE2 combinations all run cleanly
        #     (55081578, 68000100, a0010000 is already tested by default LiveAiMachine)
        #     Also check boundary Y_ptr just above IMAGE1 (0x5d8a0000) and U/V=0 vs out-of-range U/V
        m = LiveAiMachine()
        m.u.mem_write(AHIP_SLOT_VA + 0x08, words(0xE0000000))  # invalid U -> clamped to gray fallback
        m.setup_ahip_frame(y_bytes=bytes([99]) * (320 * 212), include_uv=False)
        m.u.mem_write(AHIP_SLOT_VA + 0x08, words(0xE0000000))
        m.u.mem_write(AHIP_SLOT_VA + 0x0C, words(0x20000000))
        m.run_frames(21)
        self.assertEqual(len(m.calls("send")), 1)
        tensor = m.u.mem_read(s3.TENSOR_IN_VA, 0x2A00)
        self.assertEqual((tensor[0], tensor[0x0E00], tensor[0x1C00]), (99, 99, 99))

        # 3. NULL U/V pointers fall back cleanly to grayscale R==G==B==Y
        m = LiveAiMachine(include_uv=False)
        m.setup_ahip_frame(y_bytes=bytes([77]) * (320 * 212), include_uv=False)
        m.run_frames(21)
        self.assertEqual(len(m.calls("send")), 1)
        tensor = m.u.mem_read(s3.TENSOR_IN_VA, 0x2A00)
        self.assertEqual((tensor[0], tensor[0x0E00], tensor[0x1C00]), (77, 77, 77))

    def test_option_b_empty_scene_hides_box_while_dsp_keeps_running(self):
        m = LiveAiMachine(latencies=(1,))
        # Below PEAK_THRESH=3 / AREA_THRESH=2 should keep box hidden
        m.heatmap_provider = lambda _: make_heatmap_blob(ch1_cells=[(28, 28, 2), (29, 28, 2)])
        m.run_frames(40)
        self.assertGreaterEqual(m.word(0x14), 4)  # multiple live cycles completed
        self.assertEqual(m.word(0x24), 0)         # has_box == 0
        self.assertTrue(all(fc == 0 for fc, _ in m.osd_snapshots))

    def test_dynamic_heatmap_tracking_across_aspect_ratios_and_head_priority(self):
        for width, height in ((320, 212), (320, 240), (320, 180), (240, 240)):
            with self.subTest(dims=(width, height)):
                m = LiveAiMachine(width=width, height=height, latencies=(1,))
                # Competing peaks: Ch1=9 at (14, 14) beats Ch2=10 at (42, 42) because 10 - 2 = 8 < 9!
                ch1 = [(14 + dx, 14 + dy, 9) for dy in range(2) for dx in range(2)]  # 4 cells
                ch2 = [(42 + dx, 42 + dy, 10) for dy in range(2) for dx in range(2)] # 4 cells -> total area=8
                m.heatmap_provider = lambda _: make_heatmap_blob(ch1_cells=ch1, ch2_cells=ch2)
                m.run_frames(25)
                self.assertEqual(m.word(0x24), 1)
                exp_l, exp_t, exp_s = expected_q8_box(14, 14, 8, width=width, height=height)
                fc, rect = m.osd_snapshots[-1]
                self.assertEqual(fc, 1)
                self.assertEqual(rect, (exp_l, exp_t, exp_s, exp_s))

    def test_ema_smoothing_3_cycle_hold_option_b_hide_and_reacquire(self):
        m = LiveAiMachine(latencies=(1,))
        # Cycle 0: subject at (12, 12), area=4 -> instant cold snap
        # Cycle 1: subject moves to (40, 40), area=4 -> 50/50 EMA midpoint
        # Cycles 2, 3, 4: empty -> held (lost_cycles = 1, 2, 3 -> still has_box=1)
        # Cycle 5: empty -> lost_cycles = 4 > 3 -> Option B hides box (has_box=0, face_count=0)!
        # Cycle 6: subject reappears at (20, 30) -> instant cold snap back to has_box=1!
        blob_a = make_heatmap_blob(ch1_cells=[(12 + dx, 12 + dy, 8) for dy in range(2) for dx in range(2)])
        blob_b = make_heatmap_blob(ch1_cells=[(40 + dx, 40 + dy, 8) for dy in range(2) for dx in range(2)])
        blob_c = make_heatmap_blob(ch1_cells=[(20 + dx, 30 + dy, 8) for dy in range(2) for dx in range(2)])
        empty = make_heatmap_blob()
        seq_maps = [blob_a, blob_b, empty, empty, empty, empty, blob_c]
        m.heatmap_provider = lambda req: seq_maps[min(req, len(seq_maps) - 1)]

        def run_until_cycles(target: int):
            for _ in range(40):
                if m.word(0x14) >= target:
                    break
                m.run_frames(1)
            self.assertEqual(m.word(0x14), target)
            m.run_frames(1)  # next frame commits latched box to OSD

        # Cycle 0 -> instant cold snap
        run_until_cycles(1)
        l0, t0, s0 = expected_q8_box(12, 12, 4)
        self.assertEqual(m.osd_snapshots[-1], (1, (l0, t0, s0, s0)))

        # Cycle 1 -> 50/50 EMA midpoint
        run_until_cycles(2)
        l1, t1, s1 = expected_q8_box(40, 40, 4)
        ema_l = (l0 + l1 + 1) >> 1
        ema_t = (t0 + t1 + 1) >> 1
        self.assertEqual(m.osd_snapshots[-1], (1, (ema_l, ema_t, s0, s0)))

        # Cycles 2, 3, 4 (3 empty cycles): box MUST remain held!
        run_until_cycles(5)
        self.assertEqual(m.word(0x1C), 3)
        self.assertEqual(m.word(0x24), 1)
        self.assertEqual(m.osd_snapshots[-1][0], 1)

        # Cycle 5 (4th empty cycle): Option B MUST hide box cleanly!
        run_until_cycles(6)
        self.assertEqual(m.word(0x1C), 4)
        self.assertEqual(m.word(0x24), 0)
        self.assertEqual(m.osd_snapshots[-1][0], 0)

        # Cycle 6 (re-acquire): MUST snap immediately to (20, 30)!
        run_until_cycles(7)
        lc, tc, sc = expected_q8_box(20, 30, 4)
        self.assertEqual(m.word(0x24), 1)
        self.assertEqual(m.word(0x1C), 0)
        self.assertEqual(m.osd_snapshots[-1], (1, (lc, tc, sc, sc)))

    def test_fault_tolerance_load_retry_1f_timeout_and_transient_errors(self):
        # Fail first 2 LoadDsp calls (both 'C' and 'I' on tick 20) -> retries on tick 84 and succeeds!
        # Also test 0x1f flush on req 0, dropped reply (timeout recovery) on req 1,
        # transient Start fail & Send fail -> must auto-recover and keep tracking!
        m = LiveAiMachine(
            latencies=(1,),
            fail_load_attempts=2,
            flush1f_on_reqs=(0,),
            drop_reqs=(1,),
            fail_start_on_req=2,
            fail_send_on_req=3,
        )
        blob = make_heatmap_blob(ch1_cells=[(28 + dx, 28 + dy, 10) for dy in range(2) for dx in range(2)])
        m.heatmap_provider = lambda _: blob

        m.run_frames(83)
        self.assertEqual(m.word(0x00), 0)  # waiting for retry at frame 84
        self.assertEqual(len(m.calls("start")), 0)

        m.run_frames(60)
        # Every completed/timed-out/failed-send Start must have been paired with Stop(100)
        expected_stops = len(m.calls("start")) - 1 - (1 if m.started else 0)  # -1 for fail_start_on_req=2
        self.assertEqual(len(m.calls("stop")), expected_stops)
        self.assertGreaterEqual(m.word(0x14), 3)  # recovered from 1f, timeout, start err, send err!
        self.assertEqual(m.word(0x24), 1)
        self.assertGreaterEqual(len(m.calls("ef_clear")), 1)
        self.assertEqual(m.osd_snapshots[-1][0], 1)

    def test_packaged_v1217_binary_roundtrip_and_unicorn_end_to_end(self):
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
        m = LiveAiMachine(latencies=(1, 2), flush1f_on_reqs=(0,), boot_seen=False)
        for va, length in (
            (s3.CAVE1_VA, info["cave1_len"]),
            (s3.HOOK1_SITE_VA, 8),
            (s3.CAVE2_VA, info["cave2_len"]),
            (osd.HOOK2_SITE_VA, 8),
            (lp.CAVE3_VA, info["cave3_len"]),
            (lp.HOOK_UPDATE_VA, 4),
        ):
            off = s3.RTOS_PAYLOAD_START + va - s3.GR4_BASE
            m.u.mem_write(va, payload[off : off + length])

        blob = make_heatmap_blob(ch1_cells=[(18 + dx, 22 + dy, 12) for dy in range(3) for dx in range(3)])
        m.heatmap_provider = lambda _: blob

        m.boot_init()
        m.run_frames(35)
        self.assertGreaterEqual(m.word(0x14), 3)
        exp_l, exp_t, exp_s = expected_q8_box(18, 22, 9)
        self.assertEqual(m.osd_snapshots[-1], (1, (exp_l, exp_t, exp_s, exp_s)))


if __name__ == "__main__":
    unittest.main()
