"""Executable A32 tests, with synthetic FD objects and native service stubs."""

import hashlib
import os
from pathlib import Path
import struct
import sys
import unittest

from unicorn import Uc, UcError, UC_ARCH_ARM, UC_MODE_ARM, UC_HOOK_CODE
from unicorn.arm_const import (
    UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R4, UC_ARM_REG_R6, UC_ARM_REG_SP,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src' / 'dsp_ai'), str(ROOT / 'tools')]
import fw_container as fw
import stage3_ai_patcher as s3
import step1_osd_mock_patcher as osd
import step2a_dsp_probe_patcher as step2a
import step2b_live_probe_patcher as step2b
import step2c_diag_probe_patcher as step2c
import step2c_osd_recovery as recovery

FD = 0x10000
RESULT = FD + 0x200
LIB = FD + 0x400
SP = 0x14000
CONTINUATION = osd.HOOK2_AI_RESUME_VA
UPDATE = s3.UPDATE_FACE_RES_VA


def words(*values):
    return struct.pack(f'<{len(values)}I', *values)


class ArmFrame:
    def __init__(self, cave, *, legacy=False, result=RESULT, library=LIB, width=320, height=212):
        self.uc = Uc(UC_ARCH_ARM, UC_MODE_ARM)
        self.calls = []
        self.mutex_held = True
        pages = {FD, 0x13000, osd.CAVE_VA & ~0xfff, osd.HOOK2_SITE_VA & ~0xfff,
                 UPDATE & ~0xfff, s3.UNLOCK_MUTEX_VA & ~0xfff}
        pages.update(range(osd.CAVE_VA - osd.CAVE_VA % 0x1000, osd.CAVE_VA + len(cave), 0x1000))
        if legacy:
            pages |= {s3.STATUS_VA & ~0xfff, s3.LOCK_MUTEX_VA & ~0xfff,
                      s3.MUTEX_COOKIE_VA & ~0xfff, s3.AHIP_DRV_BASE_VA & ~0xfff}
        for page in pages:
            self.uc.mem_map(page, 0x1000)
        self.uc.mem_write(osd.CAVE_VA, cave)
        self.uc.mem_write(osd.HOOK2_SITE_VA, osd.build_tramp2_payload())
        self.uc.mem_write(CONTINUATION, words(
            0xe1a00006,  # mov r0, r6
            osd.encode_arm_branch(CONTINUATION + 4, s3.UNLOCK_MUTEX_VA, is_bl=True),
            0xe1a00004,  # mov r0, r4
            osd.encode_arm_branch(CONTINUATION + 12, UPDATE, is_bl=True),
        ))
        self.uc.mem_write(FD + 0x78, words(result))
        self.uc.mem_write(FD + 0x38, words(library))
        self.uc.mem_write(FD + 0xfc, struct.pack('<HH', width, height))
        self.uc.hook_add(UC_HOOK_CODE, self.native_calls)

    def native_calls(self, uc, address, size, userdata):
        if address in (s3.UNLOCK_MUTEX_VA, s3.LOCK_MUTEX_VA):
            assert uc.reg_read(UC_ARM_REG_R0) == FD + 0x48
            assert self.mutex_held == (address == s3.UNLOCK_MUTEX_VA)
            self.mutex_held = not self.mutex_held
            self.calls.append('lock' if self.mutex_held else 'unlock')
            uc.reg_write(UC_ARM_REG_PC, uc.reg_read(UC_ARM_REG_LR))
        elif address == UPDATE:
            assert not self.mutex_held
            assert uc.reg_read(UC_ARM_REG_R0) == FD
            self.calls.append('update')
            uc.emu_stop()

    def run(self):
        self.calls.clear()
        self.mutex_held = True  # native lock immediately before Hook 2
        self.uc.reg_write(UC_ARM_REG_R4, FD)
        self.uc.reg_write(UC_ARM_REG_R6, FD + 0x48)
        self.uc.reg_write(UC_ARM_REG_SP, SP)
        self.uc.emu_start(osd.HOOK2_SITE_VA, 0, count=3000)
        if self.calls[-1:] != ['update'] or self.uc.reg_read(UC_ARM_REG_SP) != SP:
            raise AssertionError('Missing native update or stack not restored')

    def rect(self):
        return struct.unpack('<4I', self.uc.mem_read(RESULT, 16))


class TestOsdRecovery(unittest.TestCase):
    def test_prior_caves_cold_frame(self):
        for builder in (step2a.build_step2a_cave2_payload, step2b.build_step2b_cave2_payload,
                        step2c.build_step2c_cave2_payload):
            with self.subTest(builder=builder.__name__):
                frame = ArmFrame(builder(), legacy=True)
                # Old implementation unlocks and locks again even before DSP init.
                frame.run()
                self.assertEqual(frame.calls, ['unlock', 'lock', 'unlock', 'update'])
                self.assertGreater(frame.rect()[2], 0)

    def test_old_cave_reads_intermediate_pointer_before_y_guard(self):
        frame = ArmFrame(step2c.build_step2c_cave2_payload(), legacy=True)
        frame.uc.mem_write(FD + 0x13e, b'\1')
        frame.uc.mem_write(s3.STATUS_VA, words(2))  # ready to queue a new frame
        frame.uc.mem_write(s3.AHIP_DRV_BASE_VA + 0x424, words(0x44440000))
        with self.assertRaises(UcError):
            frame.run()
        self.assertEqual(frame.calls, ['unlock'])
        # Fault is the descriptor load itself, not the subsequent Y-range test.
        self.assertEqual(bytes(frame.uc.mem_read(frame.uc.reg_read(UC_ARM_REG_PC), 4)),
                         words(0x15922000))  # ldrne r2, [r2]

    def test_recovery_keeps_rendering_without_any_dsp_ddr_or_ahip_mapping(self):
        frame = ArmFrame(recovery.build_osd_recovery_cave())
        for _ in range(100):
            frame.run()
            self.assertEqual(frame.calls, ['unlock', 'update'])
            self.assertEqual(frame.rect(), (136 << 8, 82 << 8, 48 << 8, 48 << 8))
            self.assertEqual(bytes(frame.uc.mem_read(LIB + 0x30, 8)), words(1, 0))
        self.assertEqual(bytes(frame.uc.mem_read(FD + 0x13e, 1)), b'\0')

    def test_recovery_invalid_prerequisites_skip_writes(self):
        for kwargs in ({'result': 0}, {'library': 0}, {'width': 47}, {'height': 0}):
            with self.subTest(kwargs=kwargs):
                frame = ArmFrame(recovery.build_osd_recovery_cave(), **kwargs)
                before = bytes(frame.uc.mem_read(FD, 0x500))
                frame.run()
                self.assertEqual(frame.calls, ['unlock', 'update'])
                self.assertEqual(bytes(frame.uc.mem_read(FD, 0x500)), before)

    def test_recovery_smallest_supported_dimensions(self):
        frame = ArmFrame(recovery.build_osd_recovery_cave(), width=48, height=48)
        frame.run()
        self.assertEqual(frame.rect(), (0, 0, 48 << 8, 48 << 8))

    def test_payload_preflight_is_all_or_nothing(self):
        start = osd.RTOS_PAYLOAD_START
        hook = start + osd.HOOK2_SITE_VA - osd.GR4_BASE
        resume = start + CONTINUATION - osd.GR4_BASE
        cave = start + osd.CAVE_VA - osd.GR4_BASE
        payload = bytearray(resume + 4)
        payload[hook:hook+8] = recovery.EXPECTED_HOOK
        payload[resume:resume+4] = recovery.EXPECTED_STOCK_CONTINUATION
        for index in (hook, resume, cave):
            with self.subTest(index=index):
                damaged = bytearray(payload)
                damaged[index] ^= 1
                before = bytes(damaged)
                with self.assertRaises(ValueError):
                    recovery.apply_to_decoded_payload(damaged, start, len(damaged))
                self.assertEqual(damaged, before)
        with self.assertRaises(ValueError):
            recovery.apply_to_decoded_payload(payload, start, hook + 4)
        recovery.apply_to_decoded_payload(payload, start, len(payload))
        with self.assertRaises(ValueError):
            recovery.apply_to_decoded_payload(payload, start, len(payload))

    @unittest.skipUnless(os.environ.get('GR4_V120_STOCK'), 'optional fixed official input')
    def test_stock_payload_diff_allowlist(self):
        raw = Path(os.environ['GR4_V120_STOCK']).read_bytes()
        self.assertEqual(len(raw), recovery.OFFICIAL_SIZE)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), recovery.OFFICIAL_SHA256)
        official, _, consumed = fw.decode_container(raw)
        fw.verify_container(raw, official, consumed)
        rtos, = [s for s in fw.parse_sections(official) if s['name'] == 'RTOS']
        patched = bytearray(official)
        changes = recovery.apply_to_decoded_payload(patched, rtos['data_start'], rtos['data_end'])
        self.assertEqual(len(official), len(patched))
        cursor = 0
        for change in changes:
            self.assertEqual(patched[cursor:change.offset], official[cursor:change.offset])
            cursor = change.offset + len(change.replacement)
            self.assertEqual(patched[change.offset:cursor], change.replacement)
        self.assertEqual(patched[cursor:], official[cursor:])

    @unittest.skipUnless(os.environ.get('GR4_V129_DIAG'), 'optional fixed release input')
    def test_release_cave_matches_source_and_runs_cold_frame(self):
        raw = Path(os.environ['GR4_V129_DIAG']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         '71318aa75d018f00022c9f752b4641015de5269b9e0e75efffb9c0b456056530')
        payload, _, consumed = fw.decode_container(raw)
        fw.verify_container(raw, payload, consumed)
        def payload_offset(va):
            return osd.RTOS_PAYLOAD_START + va - osd.GR4_BASE
        cave = payload_offset(osd.CAVE_VA)
        hook = payload_offset(osd.HOOK2_SITE_VA)
        expected = step2c.build_step2c_cave2_payload()
        self.assertEqual(payload[hook:hook+8], osd.build_tramp2_payload())
        self.assertEqual(payload[cave:cave+len(expected)], expected)
        frame = ArmFrame(payload[cave:cave+len(expected)], legacy=True)
        frame.run()
        self.assertEqual(frame.rect(), (16 << 8, 16 << 8, 32 << 8, 32 << 8))


if __name__ == '__main__':
    unittest.main()
