"""Executable A32 Unicorn + container tests for Step 2C v1.2.14 0x1f+diag-split."""

import hashlib
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

from test_step2c_osd_recovery import CONTINUATION, FD, LIB, RESULT, SP, UPDATE, fw, osd, recovery, s3, words
import step2c_diag_split_patcher as ds

FACTORY, SERVICE, SLOT_TABLE, CACHE_REQ_TABLE = 0x15000, 0x15100, 0x16000, 0x16200
CENTER = (136 << 8, 82 << 8, 48 << 8, 48 << 8)


def grid_rect(gx: int, gy: int, width: int = 320, height: int = 212) -> tuple[int, int, int, int]:
    w = h = 48 << 8
    margin_x = ((width - 48) << 7)
    margin_y = ((height - 48) << 7)
    return (margin_x + gx * (margin_x // 2), margin_y + gy * (margin_y // 2), w, h)


class DiagSplitMachine:
    def __init__(
        self,
        *,
        latencies=(1, 3, 2, 1),
        slot_indices=(0, 3, 5, 1),
        boot_seen=True,
        flush1f_on_reqs=(),
        drop_from_req=None,
        silent_ddr_region=None,
        wrong_seq_slot=None,
        stray_dspsf_bits=0,
    ):
        self.u = Uc(UC_ARCH_ARM, UC_MODE_ARM)
        self.latencies = list(latencies)
        self.slot_indices = list(slot_indices)
        self.flush1f_on_reqs = set(flush1f_on_reqs)
        self.drop_from_req = drop_from_req
        self.silent_ddr_region = silent_ddr_region
        self.wrong_seq_slot = wrong_seq_slot
        self.stray_dspsf_bits = stray_dspsf_bits

        self.selected_service = SERVICE
        self.dspsf_flags = 0
        self.frame = 0
        self.locked = True
        self.req_idx = 0
        self.seq = 0
        self.waiting_1f = False
        self.pending_isr = None
        self.events = []
        self.rects = []
        self.flushes = []

        self.service_sites = {
            s3.UNLOCK_MUTEX_VA, UPDATE, s3.GET_DSP_SVC_VA, s3.CREATE_DSP_SVC_VA,
            s3.LOAD_DSP_VA, s3.START_DSP_VA, s3.SEND_MSG_VA, s3.WAIT_MSG_VA,
            s3.GET_CACHE_MGR_VA, s3.FLUSH_DCACHE_VA, s3.INVAL_DCACHE_VA,
            ds.OS_EF_WAIT_VA, ds.OS_EF_CLEAR_VA,
        }
        pages = {
            x & ~0xfff
            for x in self.service_sites | {
                FD, SP - 4, FACTORY, SERVICE, SLOT_TABLE, CACHE_REQ_TABLE, s3.TENSOR_MSG_VA,
                s3.CAVE1_VA, ds.HOOK_UPDATE_VA, s3.HOOK1_RESUME_VA, s3.HOOK1_SINGLETON_VA,
            }
        }
        pages.update(range(s3.CAVE2_VA & ~0xfff, s3.DESC_TABLE_VA + 0x1000, 0x1000))
        for page in pages:
            self.u.mem_map(page, 0x1000)
        self.u.mem_map(s3.TENSOR_IN_VA, 0x10000)

        self.u.mem_write(s3.CAVE1_VA, ds.build_diag_split_cave1())
        self.u.mem_write(s3.CAVE2_VA, ds.build_diag_split_cave2())
        self.u.mem_write(osd.HOOK2_SITE_VA, osd.build_tramp2_payload())
        self.u.mem_write(CONTINUATION, words(
            0xe1a00006,
            s3.encode_arm_branch(CONTINUATION + 4, s3.UNLOCK_MUTEX_VA, is_bl=True),
            0xe1a00004,
        ))
        self.u.mem_write(ds.HOOK_UPDATE_VA, ds.build_update_hook())
        self.u.mem_write(FD + 0x78, words(RESULT))
        self.u.mem_write(FD + 0x38, words(LIB))
        self.u.mem_write(FD + 0xfc, struct.pack('<HH', 320, 212))
        self.u.hook_add(UC_HOOK_CODE, self.hook)
        if boot_seen:
            self.boot_init()

    def boot_init(self):
        self.u.reg_write(UC_ARM_REG_SP, SP)
        self.u.emu_start(s3.CAVE1_VA, s3.HOOK1_RESUME_VA, count=200)
        assert self.u.reg_read(UC_ARM_REG_PC) == s3.HOOK1_RESUME_VA

    def deliver_pending_isr(self):
        if not self.pending_isr:
            return
        if self.waiting_1f:
            return  # Blocked until CPU services msg_id==0x1f cache flush + clears 0x10000
        deliver_frame, seq, slot_idx, req_num = self.pending_isr
        if self.frame < deliver_frame:
            return
        self.pending_isr = None
        if self.drop_from_req is not None and req_num >= self.drop_from_req:
            if self.silent_ddr_region in ('backbone', 'both'):
                self.u.mem_write(ds.BACKBONE_OUT_VA, words(0x44440000 + req_num))
            if self.silent_ddr_region in ('heatmap', 'both'):
                self.u.mem_write(ds.HEATMAP_OUT_VA, words(0x55550000 + req_num))
            if self.wrong_seq_slot is not None:
                s_idx, bad_seq = self.wrong_seq_slot
                self.u.mem_write(SLOT_TABLE + s_idx * 0x2c, struct.pack('<BBH', 1, 0, bad_seq) + bytes(0x28))
            if self.stray_dspsf_bits:
                self.dspsf_flags |= self.stray_dspsf_bits
            return

        self.u.mem_write(ds.BACKBONE_OUT_VA, words(0x1100 + req_num))
        self.u.mem_write(ds.HEATMAP_OUT_VA, words(0x3300 + req_num))
        slot_va = SLOT_TABLE + slot_idx * 0x2c
        reply_payload = words(0x7000 + req_num) + bytes(0x20)
        self.u.mem_write(slot_va, struct.pack('<BBH', 1, 0, seq) + words(0) + reply_payload)

    def hook(self, u, address, size, data):
        if address not in self.service_sites:
            return
        r0, r1, r2, r3 = (
            u.reg_read(r) for r in (UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3)
        )
        if address == s3.UNLOCK_MUTEX_VA:
            assert self.locked and r0 == FD + 0x48
            self.locked = False
        elif address == UPDATE:
            assert not self.locked and r0 == FD
            self.rects.append(tuple(struct.unpack('<4I', u.mem_read(RESULT, 16))))
            self.events.append((self.frame, 'osd'))
            u.reg_write(UC_ARM_REG_R0, 0x12345678)
        else:
            assert not self.locked and self.events[-1][0] == self.frame
            names = {
                s3.GET_DSP_SVC_VA: 'get',
                s3.CREATE_DSP_SVC_VA: 'create',
                s3.LOAD_DSP_VA: 'load',
                s3.START_DSP_VA: 'start',
                s3.SEND_MSG_VA: 'send',
                s3.WAIT_MSG_VA: 'wait',
                s3.GET_CACHE_MGR_VA: 'cache',
                s3.FLUSH_DCACHE_VA: 'flush',
                s3.INVAL_DCACHE_VA: 'invalidate',
                ds.OS_EF_WAIT_VA: 'ef_wait',
                ds.OS_EF_CLEAR_VA: 'ef_clear',
            }
            self.events.append((self.frame, names[address]))
            if address == s3.GET_DSP_SVC_VA:
                u.mem_write(FACTORY + 4, words(self.selected_service))
                u.reg_write(UC_ARM_REG_R0, FACTORY)
            elif address == s3.LOAD_DSP_VA:
                u.reg_write(UC_ARM_REG_R0, 1)
            elif address == s3.START_DSP_VA:
                self.dspsf_flags = 0
                u.mem_write(SERVICE + 4, words(4))
                u.mem_write(SERVICE + 0x54, words(SLOT_TABLE))
                u.mem_write(SERVICE + 0x58, words(CACHE_REQ_TABLE))
                u.mem_write(SLOT_TABLE, bytes(0x108))
                u.mem_write(CACHE_REQ_TABLE, bytes(0x48))
                u.reg_write(UC_ARM_REG_R0, 0)
            elif address == s3.FLUSH_DCACHE_VA:
                self.flushes.append((self.frame, r1, r2))
            elif address == ds.OS_EF_WAIT_VA:
                assert r0 == SERVICE + 0xc and r3 == 0
                matched = self.dspsf_flags & r1
                if r2 != 0:
                    u.mem_write(r2, words(matched))
                u.reg_write(UC_ARM_REG_R0, 1 if matched != 0 else 0)
            elif address == ds.OS_EF_CLEAR_VA:
                assert r0 == SERVICE + 0xc
                self.dspsf_flags &= ~r1
                if (r1 & 0x10000) and self.waiting_1f:
                    # Assert firmware flushed the 0x1f buffer AND cleared entry before clearing 0x10000
                    assert bytes(u.mem_read(CACHE_REQ_TABLE, 12)) == words(0, 0, 0)
                    self.waiting_1f = False
                u.reg_write(UC_ARM_REG_R0, 0)
            elif address == s3.SEND_MSG_VA:
                req_num = self.req_idx
                self.req_idx += 1
                self.seq = req_num % 6
                u.mem_write(r1, struct.pack('<H', self.seq))
                u.reg_write(UC_ARM_REG_R0, 0)
                if req_num in self.flush1f_on_reqs:
                    # Simulate ISR 0x5398e6c4..0x5398e6e0 posting msg_id=0x1f for [0x52b00000, 0x1000]
                    self.u.mem_write(CACHE_REQ_TABLE, words(0, 0x52b00000, 0x1000))
                    self.dspsf_flags |= 0x10000
                    self.waiting_1f = True
                lat = self.latencies[req_num % len(self.latencies)]
                slot_idx = self.slot_indices[req_num % len(self.slot_indices)]
                self.pending_isr = (self.frame + 1 + lat, self.seq, slot_idx, req_num)
                if lat == 0 and not self.waiting_1f:
                    self.deliver_pending_isr()
            elif address == s3.WAIT_MSG_VA:
                slots_raw = u.mem_read(SLOT_TABLE, 0x108)
                matched_off = None
                for off in range(0, 0x108, 0x2c):
                    occ, _, s_seq = struct.unpack_from('<BBH', slots_raw, off)
                    if occ != 0 and s_seq == r1:
                        matched_off = off
                        break
                assert matched_off is not None
                u.mem_write(r3 + 4, bytes(slots_raw[matched_off + 8:matched_off + 0x2c]))
                u.mem_write(SLOT_TABLE + matched_off, b'\x00')
                u.reg_write(UC_ARM_REG_R0, 0)
        u.reg_write(UC_ARM_REG_PC, u.reg_read(UC_ARM_REG_LR))

    def run_frames(self, count: int):
        for _ in range(count):
            self.frame += 1
            self.deliver_pending_isr()
            self.locked = True
            self.u.reg_write(UC_ARM_REG_SP, SP)
            self.u.reg_write(UC_ARM_REG_R4, FD)
            self.u.reg_write(UC_ARM_REG_R6, FD + 0x48)
            self.u.emu_start(osd.HOOK2_SITE_VA, ds.HOOK_UPDATE_VA + 4, count=50000)
            assert self.u.reg_read(UC_ARM_REG_PC) == ds.HOOK_UPDATE_VA + 4
            assert self.u.reg_read(UC_ARM_REG_SP) == SP
            assert self.u.reg_read(UC_ARM_REG_R0) == 0x12345678

    def calls(self, name: str):
        return [f for f, n in self.events if n == name]

    def word(self, offset: int) -> int:
        return struct.unpack('<I', self.u.mem_read(s3.STATUS_VA + offset, 4))[0]


class TestStep2CDiagSplit(unittest.TestCase):
    def test_capstone_clean_decode_and_margin(self):
        info = ds.cave2_layout()
        self.assertLessEqual(info['end_va'], s3.DESC_TABLE_VA)
        self.assertGreaterEqual(info['margin_to_desc'], 0)
        md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
        c1 = ds.build_diag_split_cave1()
        c2 = ds.build_diag_split_cave2()
        self.assertEqual(len(list(md.disasm(c1, s3.CAVE1_VA))) * 4, len(c1))
        self.assertEqual(len(list(md.disasm(c2, s3.CAVE2_VA))) * 4, len(c2))

    def test_0x1f_cache_flush_unblocks_full_completion_to_mid_right(self):
        m = DiagSplitMachine(flush1f_on_reqs=(0, 1))
        m.run_frames(80)
        self.assertEqual(m.word(0x00), 4)
        self.assertEqual(m.word(0x1c), 4)
        self.assertEqual(len(m.calls('wait')), 4)
        # Verify both 0x1f flushes (addr=0x52b00000, size=0x1000) were serviced
        flush_1f = [(a, sz) for _, a, sz in m.flushes if (a, sz) == (0x52b00000, 0x1000)]
        self.assertEqual(len(flush_1f), 2)
        self.assertEqual(m.rects[-1], grid_rect(1, 0))

    def test_timeout_after_first_pass_latches_bottom_center(self):
        m = DiagSplitMachine(drop_from_req=1)
        m.run_frames(80)
        self.assertEqual(m.calls('send'), [21, 24])
        self.assertEqual(m.calls('wait'), [23])
        self.assertEqual(m.word(0x00), 13)
        self.assertEqual(m.word(0x1c), 1)
        self.assertEqual(m.rects[-1], grid_rect(0, 1))

    def test_timeout_with_ddr_written_latches_bottom_right(self):
        for region in ('backbone', 'heatmap', 'both'):
            with self.subTest(region=region):
                m = DiagSplitMachine(drop_from_req=0, silent_ddr_region=region)
                m.run_frames(80)
                self.assertEqual(m.word(0x00), 15)
                self.assertEqual(m.word(0x1c), 0)
                self.assertNotEqual(m.word(0x14), 0)
                self.assertEqual(m.calls('wait'), [])
                self.assertEqual(m.rects[-1], grid_rect(1, 1))

    def test_timeout_with_wrong_seq_slot_latches_top_center(self):
        m = DiagSplitMachine(drop_from_req=0, wrong_seq_slot=(2, 3))
        m.run_frames(80)
        self.assertEqual(m.word(0x00), 14)
        self.assertEqual(m.word(0x1c), 0)
        self.assertEqual(m.calls('wait'), [])
        self.assertEqual(m.rects[-1], grid_rect(0, -1))

    def test_timeout_with_stray_dspsf_latches_top_center(self):
        m = DiagSplitMachine(drop_from_req=0, stray_dspsf_bits=0x100)
        m.run_frames(80)
        self.assertEqual(m.word(0x00), 14)
        self.assertEqual(m.word(0x10), 0x100)
        self.assertEqual(m.calls('wait'), [])
        self.assertEqual(m.rects[-1], grid_rect(0, -1))

    def test_completely_silent_dsp_latches_top_right(self):
        m = DiagSplitMachine(drop_from_req=0)
        m.run_frames(80)
        self.assertEqual(m.word(0x00), 8)
        self.assertEqual(m.word(0x0c), 30)
        self.assertEqual(m.word(0x10), 0)
        self.assertEqual(m.word(0x14), 0)
        self.assertEqual(m.word(0x1c), 0)
        self.assertEqual(m.calls('wait'), [])
        self.assertEqual(m.rects[-1], grid_rect(1, -1))

    @unittest.skipUnless(os.environ.get('GR4_DIAG_SPLIT_PACKAGED'), 'optional packaged v1.2.14 image')
    def test_packaged_v1214_binary_executes_end_to_end(self):
        raw = Path(os.environ['GR4_DIAG_SPLIT_PACKAGED']).read_bytes()
        payload, _, consumed = fw.decode_container(raw)
        fw.verify_container(raw, payload, consumed)

        info = ds.cave2_layout()
        m = DiagSplitMachine(latencies=(1, 3, 2, 1), flush1f_on_reqs=(0, 1), boot_seen=False)
        for va, length in (
            (s3.CAVE1_VA, len(ds.build_diag_split_cave1())),
            (s3.HOOK1_SITE_VA, 8),
            (s3.CAVE2_VA, info['total_len']),
            (osd.HOOK2_SITE_VA, 8),
            (ds.HOOK_UPDATE_VA, 4),
        ):
            off = s3.RTOS_PAYLOAD_START + va - s3.GR4_BASE
            m.u.mem_write(va, payload[off:off + length])

        m.boot_init()
        m.run_frames(80)
        self.assertEqual(m.word(0x00), 4)
        self.assertEqual(m.word(0x1c), 4)
        self.assertEqual(len(m.calls('wait')), 4)
        self.assertEqual(m.rects[-1], grid_rect(1, 0))


if __name__ == '__main__':
    unittest.main()
