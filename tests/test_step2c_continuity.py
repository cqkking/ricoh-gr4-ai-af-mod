"""Run the actual Step 2C A32 hooks with synthetic RTOS/CEVA service responses."""

import hashlib
import os
from pathlib import Path
import struct
import unittest

from unicorn import Uc, UC_ARCH_ARM, UC_MODE_ARM, UC_HOOK_CODE
from unicorn.arm_const import (
    UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2,
    UC_ARM_REG_R3, UC_ARM_REG_R4, UC_ARM_REG_R6, UC_ARM_REG_SP,
)

from test_step2c_osd_recovery import CONTINUATION, FD, LIB, RESULT, SP, UPDATE, fw, osd, s3, words
import step2a_dsp_probe_patcher as a
import step2c_continuity_patcher as c

FACTORY, SERVICE = 0x15000, 0x15100
EXPECT_STEP2A_CAVE_SHA = '951edd4af5d63d0281fa4552a6e11acb44d7526621e33a1ff5f04611c6b31d89'
CENTER = (136 << 8, 82 << 8, 48 << 8, 48 << 8)


class ProbeMachine:
    def __init__(self, *, wait_ret=0, send_ret=0, output_change=True, boot_seen=True, load_ret=1, start_ok=True, reply_copied=True, malformed_seq=False):
        self.u = Uc(UC_ARCH_ARM, UC_MODE_ARM)
        self.events = []
        self.wait_ret, self.send_ret, self.output_change = wait_ret, send_ret, output_change
        self.load_ret, self.start_ok = load_ret, start_ok
        self.reply_copied, self.malformed_seq = reply_copied, malformed_seq
        self.selected_service = SERVICE
        self.frame = 0
        self.locked = True
        self.valid_render = True
        self.seq = 0
        self.service_sites = {s3.UNLOCK_MUTEX_VA, UPDATE, s3.GET_DSP_SVC_VA, s3.LOAD_DSP_VA, s3.START_DSP_VA,
                              s3.SEND_MSG_VA, s3.WAIT_MSG_VA, s3.GET_CACHE_MGR_VA, s3.FLUSH_DCACHE_VA,
                              s3.INVAL_DCACHE_VA, s3.CREATE_DSP_SVC_VA}
        for page in {x & ~0xfff for x in self.service_sites | {FD, SP-4, FACTORY, s3.TENSOR_MSG_VA,
                         s3.CAVE2_VA, s3.CAVE2_VA+0x600, c.HOOK_UPDATE_VA, s3.HOOK1_RESUME_VA,
                         s3.HOOK1_SINGLETON_VA}}:
            self.u.mem_map(page, 0x1000)
        self.u.mem_map(s3.TENSOR_IN_VA, 0x10000)
        self.u.mem_write(s3.CAVE2_VA, c.build_continuity_cave2())
        self.u.mem_write(s3.CAVE1_VA, c.build_continuity_cave1())
        self.u.mem_write(osd.HOOK2_SITE_VA, osd.build_tramp2_payload())
        self.u.mem_write(CONTINUATION, words(0xe1a00006,
                           s3.encode_arm_branch(CONTINUATION+4, s3.UNLOCK_MUTEX_VA, is_bl=True), 0xe1a00004))
        self.u.mem_write(c.HOOK_UPDATE_VA, c.build_update_hook())
        self.u.mem_write(FD+0x78, words(RESULT))
        self.u.mem_write(FD+0x38, words(LIB))
        self.u.mem_write(FD+0xfc, struct.pack('<HH', 320, 212))
        self.u.hook_add(UC_HOOK_CODE, self.hook)
        if boot_seen:
            self.boot_init()

    def boot_init(self):
        self.u.reg_write(UC_ARM_REG_SP, SP)
        self.u.emu_start(s3.CAVE1_VA, s3.HOOK1_RESUME_VA, count=200)
        assert self.u.reg_read(UC_ARM_REG_PC) == s3.HOOK1_RESUME_VA

    def hook(self, u, address, size, data):
        if address not in self.service_sites:
            return
        r0, r1, r2, r3 = (u.reg_read(r) for r in (UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3))
        if address == s3.UNLOCK_MUTEX_VA:
            assert self.locked and r0 == FD + 0x48
            self.locked = False
        elif address == UPDATE:
            assert not self.locked and r0 == FD
            self.check_render(tuple(struct.unpack('<4I', u.mem_read(RESULT, 16))))
            self.events.append((self.frame, 'osd' if self.valid_render else 'native_update_only'))
            u.reg_write(UC_ARM_REG_R0, 0x12345678)
        else:
            assert not self.locked and self.events[-1][0] == self.frame
            assert (self.frame, 'osd') in self.events, 'DSP must not precede stock OSD commit'
            names = {s3.GET_DSP_SVC_VA:'get', s3.LOAD_DSP_VA:'load', s3.START_DSP_VA:'start', s3.SEND_MSG_VA:'send',
                     s3.WAIT_MSG_VA:'wait', s3.GET_CACHE_MGR_VA:'cache', s3.FLUSH_DCACHE_VA:'flush', s3.INVAL_DCACHE_VA:'invalidate'}
            self.events.append((self.frame, names[address]))
            if address == s3.GET_DSP_SVC_VA:
                u.mem_write(FACTORY+4, words(self.selected_service))
                u.reg_write(UC_ARM_REG_R0, FACTORY)
            elif address == s3.LOAD_DSP_VA:
                assert r0 == SERVICE and r1 in (0x43, 0x49)
                u.reg_write(UC_ARM_REG_R0, self.load_ret)
            elif address == s3.START_DSP_VA:
                assert r0 == SERVICE and r1 == 0x52100000
                u.mem_write(SERVICE+4, words(4 if self.start_ok else 0))
            elif address == s3.FLUSH_DCACHE_VA:
                assert (r1,r2) == (s3.TENSOR_IN_VA,0x7e00)
            elif address == s3.INVAL_DCACHE_VA:
                assert (r1,r2) == (s3.TENSOR_OUT_VA,0x5400)
            elif address == s3.SEND_MSG_VA:
                assert r0 == SERVICE and r1 == s3.STATUS_VA+8 and r2 == 4 and r3 == 100
                assert bytes(u.mem_read(s3.TENSOR_IN_VA, 0x2a00)) == b'\x80'*0x2a00
                assert bytes(u.mem_read(s3.TENSOR_OUT_VA, 0x5400)) == words(a.CANARY_VAL)*(0x5400//4)
                assert bytes(u.mem_read(u.reg_read(UC_ARM_REG_SP), 4)) == words(s3.TENSOR_MSG_VA)
                self.seq = (self.seq+1) % 6
                u.mem_write(r1, struct.pack('<H', 6 if self.malformed_seq else self.seq))
                u.reg_write(UC_ARM_REG_R0, self.send_ret & 0xffffffff)
            elif address == s3.WAIT_MSG_VA:
                assert r0 == SERVICE and self.seq == r1 and r2 == 0 and r3 == a.RESP_BUF_VA
                self.assert_primed_reply(u)
                if self.reply_copied and self.wait_ret == 0:
                    u.mem_write(r3+4, words(self.seq) + bytes(0x20))
                if self.output_change and self.wait_ret == 0:
                    u.mem_write(s3.TENSOR_OUT_VA, words(self.seq))
                u.reg_write(UC_ARM_REG_R0, self.wait_ret & 0xffffffff)
        u.reg_write(UC_ARM_REG_PC, u.reg_read(UC_ARM_REG_LR))

    def check_render(self, rect):
        assert rect == (CENTER if self.valid_render else (0,)*4)

    @staticmethod
    def assert_primed_reply(u):
        assert bytes(u.mem_read(a.RESP_BUF_VA+4, 0x24)) == words(a.CONTINUITY_REPLY_MARK)*9

    def run_frames(self, count):
        for _ in range(count):
            self.frame += 1
            self.locked = True
            self.u.reg_write(UC_ARM_REG_SP, SP)
            self.u.reg_write(UC_ARM_REG_R4, FD)
            self.u.reg_write(UC_ARM_REG_R6, FD+0x48)
            self.u.emu_start(osd.HOOK2_SITE_VA, c.HOOK_UPDATE_VA+4, count=40000)
            assert self.u.reg_read(UC_ARM_REG_PC) == c.HOOK_UPDATE_VA+4
            assert self.u.reg_read(UC_ARM_REG_SP) == SP
            assert self.u.reg_read(UC_ARM_REG_R0) == 0x12345678

    def calls(self, name):
        return [f for f, n in self.events if n == name]

    def word(self, offset):
        return struct.unpack('<I', self.u.mem_read(s3.STATUS_VA+offset, 4))[0]


class TestStep2CContinuity(unittest.TestCase):
    def assert_osd(self, machine, n=80):
        self.assertEqual(machine.calls('osd'), list(range(1,n+1)))

    def test_four_transactions_then_stop_no_same_frame_resend(self):
        m=ProbeMachine(); m.run_frames(80); self.assert_osd(m)
        self.assertEqual(m.calls('send'), [21,25,29,33])
        self.assertEqual(m.calls('wait'), [24,28,32,36])
        self.assertEqual((m.word(0),m.word(0x1c)), (4,4))

    def test_timeout_keeps_osd_and_never_repolls_or_reuses_buffers(self):
        m=ProbeMachine(wait_ret=-3); m.run_frames(80); self.assert_osd(m)
        self.assertEqual(m.calls('send'), [21]); self.assertEqual(m.calls('wait'), [24])
        self.assertEqual((m.word(0),m.word(0x1c)), (8,0))

    def test_success_without_dequeued_reply_stops_even_if_ddr_changes(self):
        m=ProbeMachine(reply_copied=False); m.run_frames(80); self.assert_osd(m)
        self.assertEqual(m.calls('send'), [21]); self.assertEqual(m.calls('wait'), [24])
        self.assertFalse(m.calls('invalidate')); self.assertEqual((m.word(0),m.word(0x1c)), (8,0))

    def test_out_of_range_sequence_never_enters_wait(self):
        m=ProbeMachine(malformed_seq=True); m.run_frames(80); self.assert_osd(m)
        self.assertEqual(m.calls('send'), [21]); self.assertFalse(m.calls('wait'))
        self.assertEqual(m.word(0),8)

    def test_send_error_no_retry(self):
        m=ProbeMachine(send_ret=-3); m.run_frames(80); self.assert_osd(m)
        self.assertEqual(m.calls('send'), [21]); self.assertFalse(m.calls('wait')); self.assertEqual(m.word(0),6)

    def test_unchanged_canary_no_retry(self):
        m=ProbeMachine(output_change=False); m.run_frames(80); self.assert_osd(m)
        self.assertEqual(m.calls('send'), [21]); self.assertEqual(m.calls('wait'), [24]); self.assertEqual(m.word(0),7)

    def test_existing_started_service_is_not_reloaded(self):
        m=ProbeMachine()
        m.u.mem_write(SERVICE+4, words(4))
        m.run_frames(80); self.assert_osd(m)
        self.assertEqual(m.calls('get'), [20]); self.assertFalse(m.calls('load'))
        self.assertFalse(m.calls('send')); self.assertEqual(m.word(0),8)

    def test_no_cookie_does_not_touch_dsp(self):
        m=ProbeMachine(boot_seen=False); m.run_frames(80); self.assert_osd(m)
        self.assertFalse(m.calls('get')); self.assertEqual(m.word(0),0)

    def test_missing_or_replaced_service_aborts_before_wait(self):
        for service in (0, SERVICE+0x100):
            with self.subTest(service=service):
                m=ProbeMachine(); m.run_frames(21); m.selected_service=service; m.run_frames(59); self.assert_osd(m)
                self.assertEqual(m.calls('send'),[21]); self.assertFalse(m.calls('wait')); self.assertEqual(m.word(0),8)

    def test_load_or_start_failure_keeps_osd_without_retry(self):
        for kwargs,state in (({'load_ret':0},1), ({'start_ok':False},5)):
            with self.subTest(kwargs=kwargs):
                m=ProbeMachine(**kwargs); m.run_frames(80); self.assert_osd(m)
                self.assertFalse(m.calls('send')); self.assertFalse(m.calls('wait')); self.assertEqual(m.word(0),state)

    def test_invalid_geometry_never_launches_dsp(self):
        m=ProbeMachine()
        m.valid_render = False
        m.u.mem_write(FD+0xfc, struct.pack('<HH', 47, 212))
        m.run_frames(80)
        self.assertEqual(m.calls('native_update_only'), list(range(1,81)))
        self.assertFalse(m.calls('get')); self.assertEqual(m.word(4), 0)

    def test_original_step2a_emitter_unchanged(self):
        self.assertEqual(hashlib.sha256(a.build_step2a_cave2_payload()).hexdigest(), EXPECT_STEP2A_CAVE_SHA)

    @unittest.skipUnless(os.environ.get('GR4_CONTINUITY_PACKAGED'), 'optional exact candidate file')
    def test_directly_execute_packaged_hooks(self):
        candidate = Path(os.environ['GR4_CONTINUITY_PACKAGED']).read_bytes()
        self.assertEqual(hashlib.sha256(candidate).hexdigest(),
                         'd385fffe1dbf8ebdd6a0aacb62ffbf1c9bf6e466948b43bdd02643a3f3df9284')
        payload = fw.verify_container(candidate)
        m = ProbeMachine(boot_seen=False)
        for va,length in ((s3.CAVE1_VA,100), (s3.CAVE2_VA,1288), (osd.HOOK2_SITE_VA,8), (c.HOOK_UPDATE_VA,4)):
            loc = s3.RTOS_PAYLOAD_START + va-s3.GR4_BASE
            m.u.mem_write(va, payload[loc:loc+length])
        m.boot_init()
        m.run_frames(80)
        self.assert_osd(m)
        self.assertEqual(m.calls('wait'), [24,28,32,36]); self.assertEqual((m.word(0),m.word(0x1c)), (4,4))

    @unittest.skipUnless(os.environ.get('GR4_V120_STOCK'), 'optional pinned stock container')
    def test_exact_stock_hooks_and_preflight_no_partial_write(self):
        stock = Path(os.environ['GR4_V120_STOCK']).read_bytes()
        self.assertEqual(hashlib.sha256(stock).hexdigest(), '7563d7e462f0c53b6ab2fbab7cdfbbb57151a41192ee82f4f3a2108f9d77650d')
        payload = fw.verify_container(stock)
        rtos, = (s for s in fw.parse_sections(payload) if s['name']=='RTOS')
        start,end = rtos['data_start'],rtos['data_end']
        modified = bytearray(payload)
        manifest = c.apply_continuity_experiment(modified,start,end)
        off = int(manifest['post_update_hook']['offset'],16)
        self.assertEqual(modified[off:off+4],c.build_update_hook())
        self.assertEqual((modified[:start],modified[end:]),(payload[:start],payload[end:]))
        unrelated = bytearray(payload)
        unrelated[s3.POOL8_PAYLOAD_OFF+8] ^= 1
        bad_snapshot = bytes(unrelated)
        with self.assertRaises(AssertionError):
            c.apply_continuity_experiment(unrelated,start,end)
        self.assertEqual(unrelated,bad_snapshot)
