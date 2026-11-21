"""Executable A32 Unicorn + container tests for Step 2C v1.2.13 read-only slot polling."""

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
import build_step2c_slot_poll as build_sp
import step2c_slot_poll_patcher as sp

FACTORY, SERVICE, SLOT_TABLE = 0x15000, 0x15100, 0x16000
CENTER = (136 << 8, 82 << 8, 48 << 8, 48 << 8)


def grid_rect(gx: int, gy: int, width: int = 320, height: int = 212) -> tuple[int, int, int, int]:
    w = h = 48 << 8
    margin_x = ((width - 48) << 7)
    margin_y = ((height - 48) << 7)
    return (margin_x + gx * (margin_x // 2), margin_y + gy * (margin_y // 2), w, h)


class SlotPollMachine:
    def __init__(
        self,
        *,
        latencies=(1, 5, 12, 0),
        slot_indices=(0, 3, 5, 1),
        boot_seen=True,
        create_from_null=False,
        create_fails=False,
        load_ret_c=1,
        load_ret_i=1,
        start_ok=True,
        pre_started=False,
        null_slot_table=False,
        send_ret=0,
        malformed_seq=False,
        wait_ret=0,
        reply_copied=True,
        write_backbone=True,
        write_heatmap=True,
        drop_isr=False,
    ):
        self.u = Uc(UC_ARCH_ARM, UC_MODE_ARM)
        self.latencies = list(latencies)
        self.slot_indices = list(slot_indices)
        self.create_from_null = create_from_null
        self.create_fails = create_fails
        self.load_ret_c = load_ret_c
        self.load_ret_i = load_ret_i
        self.start_ok = start_ok
        self.pre_started = pre_started
        self.null_slot_table = null_slot_table
        self.send_ret = send_ret
        self.malformed_seq = malformed_seq
        self.wait_ret = wait_ret
        self.reply_copied = reply_copied
        self.write_backbone = write_backbone
        self.write_heatmap = write_heatmap
        self.drop_isr = drop_isr

        self.selected_service = 0 if (create_from_null or create_fails) else SERVICE
        self.factory_present = True
        self.frame = 0
        self.locked = True
        self.valid_render = True
        self.req_idx = 0
        self.seq = 0
        self.pending_isr = None
        self.events = []
        self.rects = []

        self.service_sites = {
            s3.UNLOCK_MUTEX_VA, UPDATE, s3.GET_DSP_SVC_VA, s3.CREATE_DSP_SVC_VA,
            s3.LOAD_DSP_VA, s3.START_DSP_VA, s3.SEND_MSG_VA, s3.WAIT_MSG_VA,
            s3.GET_CACHE_MGR_VA, s3.FLUSH_DCACHE_VA, s3.INVAL_DCACHE_VA,
        }
        pages = {
            x & ~0xfff
            for x in self.service_sites | {
                FD, SP - 4, FACTORY, SERVICE, SLOT_TABLE, s3.TENSOR_MSG_VA,
                s3.CAVE1_VA, sp.HOOK_UPDATE_VA, s3.HOOK1_RESUME_VA, s3.HOOK1_SINGLETON_VA,
            }
        }
        pages.update(range(s3.CAVE2_VA & ~0xfff, s3.DESC_TABLE_VA + 0x1000, 0x1000))
        for page in pages:
            self.u.mem_map(page, 0x1000)
        self.u.mem_map(s3.TENSOR_IN_VA, 0x10000)

        self.u.mem_write(s3.CAVE1_VA, sp.build_slot_poll_cave1())
        self.u.mem_write(s3.CAVE2_VA, sp.build_slot_poll_cave2())
        self.u.mem_write(osd.HOOK2_SITE_VA, osd.build_tramp2_payload())
        self.u.mem_write(CONTINUATION, words(
            0xe1a00006,
            s3.encode_arm_branch(CONTINUATION + 4, s3.UNLOCK_MUTEX_VA, is_bl=True),
            0xe1a00004,
        ))
        self.u.mem_write(sp.HOOK_UPDATE_VA, sp.build_update_hook())
        self.u.mem_write(FD + 0x78, words(RESULT))
        self.u.mem_write(FD + 0x38, words(LIB))
        self.u.mem_write(FD + 0xfc, struct.pack('<HH', 320, 212))
        if pre_started:
            self.u.mem_write(SERVICE + 4, words(4))
            self.u.mem_write(SERVICE + 0x54, words(0 if null_slot_table else SLOT_TABLE))
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
        deliver_frame, seq, slot_idx, req_num = self.pending_isr
        if self.frame < deliver_frame:
            return
        self.pending_isr = None
        if self.drop_isr:
            return
        # CEVA-XM4 writes output DDR before raising mailbox reply IRQ
        if self.write_backbone:
            self.u.mem_write(sp.BACKBONE_OUT_VA, words(0x1100 + req_num, 0x2200 + req_num))
        if self.write_heatmap:
            self.u.mem_write(sp.HEATMAP_OUT_VA, words(0x3300 + req_num))
        # ISR calls 0x5398e5dc: stores occupied=1 (+0x00), seq (+0x02), and 0x24B reply (+0x08)
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
            rect = tuple(struct.unpack('<4I', u.mem_read(RESULT, 16)))
            self.rects.append(rect)
            self.events.append((self.frame, 'osd' if self.valid_render else 'native_update_only'))
            u.reg_write(UC_ARM_REG_R0, 0x12345678)
        else:
            assert not self.locked and self.events[-1][0] == self.frame
            assert (self.frame, 'osd') in self.events, 'DSP must not precede stock OSD commit'
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
            }
            self.events.append((self.frame, names[address]))
            if address == s3.GET_DSP_SVC_VA:
                if self.factory_present:
                    u.mem_write(FACTORY + 4, words(self.selected_service))
                    u.reg_write(UC_ARM_REG_R0, FACTORY)
                else:
                    u.reg_write(UC_ARM_REG_R0, 0)
            elif address == s3.CREATE_DSP_SVC_VA:
                assert r0 == FACTORY and r1 == 0
                if not self.create_fails:
                    self.selected_service = SERVICE
                    u.mem_write(FACTORY + 4, words(SERVICE))
                u.reg_write(UC_ARM_REG_R0, self.selected_service)
            elif address == s3.LOAD_DSP_VA:
                assert r0 == SERVICE and r1 in (0x43, 0x49)
                u.reg_write(UC_ARM_REG_R0, self.load_ret_c if r1 == 0x43 else self.load_ret_i)
            elif address == s3.START_DSP_VA:
                assert r0 == SERVICE and r1 == 0x52100000
                if self.start_ok:
                    u.mem_write(SERVICE + 4, words(4))
                    u.mem_write(SERVICE + 0x54, words(0 if self.null_slot_table else SLOT_TABLE))
                    u.mem_write(SLOT_TABLE, bytes(0x108))
                    u.reg_write(UC_ARM_REG_R0, 0)
                else:
                    u.mem_write(SERVICE + 4, words(2))
                    u.reg_write(UC_ARM_REG_R0, 0xffffffff)
            elif address == s3.FLUSH_DCACHE_VA:
                assert (r1, r2) == (s3.TENSOR_IN_VA, sp.WORK_BUF_FLUSH_LEN)
            elif address == s3.INVAL_DCACHE_VA:
                assert (r1, r2) == (sp.BACKBONE_OUT_VA, sp.OUT_INVAL_LEN)
            elif address == s3.SEND_MSG_VA:
                assert r0 == SERVICE and r1 == s3.STATUS_VA + 8 and r2 == 4 and r3 == 100
                assert bytes(u.mem_read(s3.TENSOR_IN_VA, 0x2a00)) == b'\x80' * 0x2a00
                assert bytes(u.mem_read(sp.BACKBONE_OUT_VA, sp.OUT_INVAL_LEN)) == words(sp.CANARY_VAL) * (sp.OUT_INVAL_LEN // 4)
                assert bytes(u.mem_read(u.reg_read(UC_ARM_REG_SP), 4)) == words(s3.TENSOR_MSG_VA)
                msg_words = struct.unpack('<10I', u.mem_read(s3.TENSOR_MSG_VA, 40))
                assert msg_words == (
                    s3.DESC_TABLE_VA, 0, s3.TENSOR_IN_VA, s3.HEAD_WEIGHTS_VA, s3.MEAN_WEIGHTS_VA,
                    0x00080808, 0x0c0c0c0a, 0x0c0c0c0c, 0, 0,
                )
                req_num = self.req_idx
                self.req_idx += 1
                self.seq = req_num % 6
                reported_seq = 6 if self.malformed_seq else self.seq
                u.mem_write(r1, struct.pack('<H', reported_seq))
                u.reg_write(UC_ARM_REG_R0, self.send_ret & 0xffffffff)
                if self.send_ret == 0 and not self.malformed_seq:
                    lat = self.latencies[req_num % len(self.latencies)]
                    slot_idx = self.slot_indices[req_num % len(self.slot_indices)]
                    self.pending_isr = (self.frame + 1 + lat, self.seq, slot_idx, req_num)
                    if lat == 0:
                        self.deliver_pending_isr()
            elif address == s3.WAIT_MSG_VA:
                assert r0 == SERVICE and r1 == self.seq and r2 == 0 and r3 == sp.RESP_BUF_VA
                assert bytes(u.mem_read(sp.RESP_BUF_VA + 4, 0x24)) == words(sp.REPLY_MARK) * 9
                # Prove WaitMessage is ONLY called after ISR populated matching reply slot
                slots_raw = u.mem_read(SLOT_TABLE, 0x108)
                matched_off = None
                for off in range(0, 0x108, 0x2c):
                    occ, _, s_seq = struct.unpack_from('<BBH', slots_raw, off)
                    if occ != 0 and s_seq == r1:
                        matched_off = off
                        break
                assert matched_off is not None, 'WaitMessage called before ISR populated [svc+0x54] slot!'
                if self.wait_ret == 0:
                    if self.reply_copied:
                        u.mem_write(r3 + 4, bytes(slots_raw[matched_off + 8:matched_off + 0x2c]))
                    u.mem_write(SLOT_TABLE + matched_off, b'\x00')  # clear slot occupied byte like 0x5398e78c
                u.reg_write(UC_ARM_REG_R0, self.wait_ret & 0xffffffff)
        u.reg_write(UC_ARM_REG_PC, u.reg_read(UC_ARM_REG_LR))

    def run_frames(self, count: int):
        for _ in range(count):
            self.frame += 1
            self.deliver_pending_isr()
            self.locked = True
            self.u.reg_write(UC_ARM_REG_SP, SP)
            self.u.reg_write(UC_ARM_REG_R4, FD)
            self.u.reg_write(UC_ARM_REG_R6, FD + 0x48)
            self.u.emu_start(osd.HOOK2_SITE_VA, sp.HOOK_UPDATE_VA + 4, count=50000)
            assert self.u.reg_read(UC_ARM_REG_PC) == sp.HOOK_UPDATE_VA + 4
            assert self.u.reg_read(UC_ARM_REG_SP) == SP
            assert self.u.reg_read(UC_ARM_REG_R0) == 0x12345678

    def calls(self, name: str):
        return [f for f, n in self.events if n == name]

    def word(self, offset: int) -> int:
        return struct.unpack('<I', self.u.mem_read(s3.STATUS_VA + offset, 4))[0]


class TestStep2CSlotPoll(unittest.TestCase):
    def test_capstone_clean_decode_and_margin(self):
        info = sp.cave2_layout()
        self.assertLessEqual(info['end_va'], s3.DESC_TABLE_VA)
        self.assertGreaterEqual(info['margin_to_desc'], 0)
        md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
        c1 = sp.build_slot_poll_cave1()
        c2 = sp.build_slot_poll_cave2()
        self.assertEqual(len(list(md.disasm(c1, s3.CAVE1_VA))) * 4, len(c1))
        self.assertEqual(len(list(md.disasm(c2, s3.CAVE2_VA))) * 4, len(c2))

    def test_variable_latency_multislot_completion(self):
        # Latencies [1, 5, 12, 0] across slots [0, 3, 5, 1]:
        # WaitMessage is NEVER called prematurely; box stays centered until 4/4 DONE_OK -> Mid-Right
        m = SlotPollMachine(latencies=(1, 5, 12, 0), slot_indices=(0, 3, 5, 1))
        m.run_frames(80)
        self.assertEqual(m.calls('osd'), list(range(1, 81)))
        self.assertEqual(m.calls('send'), [21, 24, 31, 45])
        self.assertEqual(m.calls('wait'), [23, 30, 44, 46])
        self.assertEqual(m.word(0x00), 4)
        self.assertEqual(m.word(0x1c), 4)
        diffs = m.word(0x14)
        self.assertEqual((diffs & 0xffff, diffs >> 16), (2, 1))
        self.assertEqual(m.rects, [CENTER] * 46 + [grid_rect(1, 0)] * 34)

    def test_create_service_from_null_and_internal_flash_fallback(self):
        m = SlotPollMachine(create_from_null=True, load_ret_c=0, load_ret_i=1, latencies=(2, 2, 2, 2))
        m.run_frames(80)
        self.assertEqual(len(m.calls('create')), 1)
        self.assertEqual(len(m.calls('load')), 2)
        self.assertEqual(m.word(0x00), 4)
        self.assertEqual(m.word(0x1c), 4)
        self.assertEqual(m.rects[-1], grid_rect(1, 0))

    def test_slot_timeout_30_frames_calls_zero_wait_and_latches_top_right(self):
        m = SlotPollMachine(drop_isr=True)
        m.run_frames(80)
        self.assertEqual(m.calls('send'), [21])
        self.assertEqual(m.calls('wait'), [])
        self.assertEqual(m.word(0x00), 8)
        self.assertEqual(m.word(0x0c), 30)
        self.assertEqual(m.rects, [CENTER] * 51 + [grid_rect(1, -1)] * 29)

    def test_non_timeout_protocol_and_service_errors_show_top_center(self):
        cases = (
            ({'wait_ret': -1}, 9, [23]),
            ({'reply_copied': False}, 10, [23]),
            ({'malformed_seq': True}, 10, []),
            ({'pre_started': True}, 11, []),
            ({'null_slot_table': True}, 11, []),
        )
        for kwargs, expected_state, expected_wait in cases:
            with self.subTest(kwargs=kwargs):
                m = SlotPollMachine(**kwargs)
                m.run_frames(80)
                self.assertEqual(m.word(0x00), expected_state)
                self.assertEqual(m.calls('wait'), expected_wait)
                self.assertEqual(m.rects[-1], grid_rect(0, -1))
                self.assertNotIn(grid_rect(1, 0), m.rects)

    def test_service_swapped_mid_flight_latches_top_center(self):
        m = SlotPollMachine(latencies=(2, 2, 2, 2))
        m.run_frames(21)
        m.selected_service = SERVICE + 0x100
        m.run_frames(59)
        self.assertEqual(m.word(0x00), 11)
        self.assertEqual(m.calls('wait'), [])
        self.assertEqual(m.rects[-1], grid_rect(0, -1))

    def test_ddr_write_split_vs_none(self):
        for write_bb, write_hm, expected_state, expected_xy in (
            (True, False, 12, (0, 1)),
            (False, True, 12, (0, 1)),
            (False, False, 7, (1, 1)),
        ):
            with self.subTest(write_bb=write_bb, write_hm=write_hm):
                m = SlotPollMachine(write_backbone=write_bb, write_heatmap=write_hm)
                m.run_frames(80)
                self.assertEqual(m.word(0x00), expected_state)
                self.assertEqual(m.word(0x1c), 0)
                self.assertEqual(m.calls('send'), [21])
                self.assertEqual(m.calls('wait'), [23])
                self.assertEqual(m.rects[-1], grid_rect(*expected_xy))

    def test_left_column_boot_and_send_errors(self):
        cases = (
            ({'load_ret_c': 0, 'load_ret_i': 0}, 1, (-1, -1)),
            ({'create_fails': True}, 1, (-1, -1)),
            ({'start_ok': False}, 5, (-1, 0)),
            ({'send_ret': -3}, 6, (-1, 1)),
        )
        for kwargs, expected_state, expected_xy in cases:
            with self.subTest(kwargs=kwargs):
                m = SlotPollMachine(**kwargs)
                m.run_frames(80)
                self.assertEqual(m.word(0x00), expected_state)
                self.assertEqual(m.calls('wait'), [])
                self.assertEqual(m.rects[-1], grid_rect(*expected_xy))

    def test_all_9_grid_positions_stay_in_bounds_across_aspect_ratios(self):
        states = {
            1: (-1, -1), 4: (1, 0), 5: (-1, 0), 6: (-1, 1), 7: (1, 1),
            8: (1, -1), 9: (0, -1), 10: (0, -1), 11: (0, -1), 12: (0, 1),
        }
        for width, height in ((48, 48), (320, 212), (320, 240), (320, 180), (320, 320)):
            for state, xy in states.items():
                m = SlotPollMachine()
                m.u.mem_write(FD + 0xfc, struct.pack('<HH', width, height))
                m.u.mem_write(s3.STATUS_VA, words(state))
                m.u.mem_write(s3.STATUS_VA + 0x1c, words(4))
                m.run_frames(1)
                left, top, bw, bh = m.rects[0]
                self.assertEqual((left, top, bw, bh), grid_rect(*xy, width, height))
                self.assertLessEqual(left + bw, width << 8)
                self.assertLessEqual(top + bh, height << 8)

    @unittest.skipUnless(os.environ.get('GR4_V120_STOCK'), 'optional pinned stock container')
    def test_stock_payload_apply_and_ipl_untouched(self):
        stock = Path(os.environ['GR4_V120_STOCK']).read_bytes()
        self.assertEqual(hashlib.sha256(stock).hexdigest(), recovery.OFFICIAL_SHA256)
        payload, _, consumed = fw.decode_container(stock)
        fw.verify_container(stock, payload, consumed)
        rtos, = (s for s in fw.parse_sections(payload) if s['name'] == 'RTOS')
        mod = bytearray(payload)
        manifest = sp.apply_slot_poll_experiment(mod, rtos['data_start'], rtos['data_end'])
        self.assertEqual(manifest['request_budget'], 4)
        self.assertEqual(mod[:rtos['data_start']], payload[:rtos['data_start']])
        self.assertEqual(mod[rtos['data_end']:], payload[rtos['data_end']:])

    @unittest.skipUnless(os.environ.get('GR4_SLOT_POLL_PACKAGED'), 'optional packaged v1.2.13 image')
    def test_packaged_binary_bytes_execute_end_to_end(self):
        raw = Path(os.environ['GR4_SLOT_POLL_PACKAGED']).read_bytes()
        payload, _, consumed = fw.decode_container(raw)
        fw.verify_container(raw, payload, consumed)

        info = sp.cave2_layout()
        m = SlotPollMachine(latencies=(1, 5, 12, 0), slot_indices=(0, 3, 5, 1), boot_seen=False)
        for va, length in (
            (s3.CAVE1_VA, len(sp.build_slot_poll_cave1())),
            (s3.HOOK1_SITE_VA, 8),
            (s3.CAVE2_VA, info['total_len']),
            (osd.HOOK2_SITE_VA, 8),
            (sp.HOOK_UPDATE_VA, 4),
        ):
            off = s3.RTOS_PAYLOAD_START + va - s3.GR4_BASE
            m.u.mem_write(va, payload[off:off + length])

        m.boot_init()
        m.run_frames(80)
        self.assertEqual(m.calls('send'), [21, 24, 31, 45])
        self.assertEqual(m.calls('wait'), [23, 30, 44, 46])
        self.assertEqual(m.word(0x00), 4)
        self.assertEqual(m.word(0x1c), 4)
        self.assertEqual(m.rects, [CENTER] * 46 + [grid_rect(1, 0)] * 34)


if __name__ == '__main__':
    unittest.main()
