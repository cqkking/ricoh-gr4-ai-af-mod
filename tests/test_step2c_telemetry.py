"""Execute the exact telemetry A32 hook and the preserved synthetic DSP core."""
import hashlib
import os
from pathlib import Path
import struct
import sys
import unittest

from test_step2c_continuity import ProbeMachine, SERVICE, CENTER, a, c, s3, words
from test_step2c_osd_recovery import FD, RESULT, fw
import step2c_telemetry_patcher as t
import build_step2c_telemetry as build

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_step2c_telemetry import DELTAS, check_differences


def position(x, y, width=320, height=212):
    w = h = 48 << 8
    margin_x, margin_y = ((width-48) << 7), ((height-48) << 7)
    return (margin_x+x*(margin_x//2), margin_y+y*(margin_y//2), w, h)


class TelemetryMachine(ProbeMachine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rects = []
        blank = bytes(len(t.build_position_helper()))
        patched, helper = t.patch_cave(c.build_continuity_cave2(), blank)
        self.u.mem_write(s3.CAVE2_VA, patched+helper)

    def check_render(self, rect):
        self.rects.append(rect)


class TestTelemetry(unittest.TestCase):
    def test_center_until_four_then_right_no_change_to_dsp_cadence(self):
        m=TelemetryMachine(); m.run_frames(80)
        self.assertEqual(m.rects, [CENTER]*36+[position(1,0)]*44)
        self.assertEqual(m.calls('send'), [21,25,29,33]); self.assertEqual(m.calls('wait'), [24,28,32,36])
        self.assertEqual((m.word(0), m.word(0x1c)), (4,4))

    def test_error_landmarks_and_never_false_success(self):
        cases = (({'load_ret':0},(-1,-1)), ({'start_ok':False},(-1,0)),
                 ({'send_ret':-3},(-1,1)), ({'wait_ret':-3},(1,-1)),
                 ({'reply_copied':False},(1,-1)), ({'malformed_seq':True},(1,-1)),
                 ({'output_change':False},(1,1)))
        for kwargs, xy in cases:
            with self.subTest(kwargs=kwargs):
                m=TelemetryMachine(**kwargs); m.run_frames(80)
                self.assertEqual(m.rects[0], CENTER)
                self.assertEqual(m.rects[-12:], [position(*xy)]*12)
                self.assertNotIn(position(1,0), m.rects)
                self.assertLessEqual(len(m.calls('send')), 1)

    def test_done_state_requires_exact_completed_count(self):
        for count in (0, 3, 5, 0xffffffff):
            with self.subTest(count=count):
                m=TelemetryMachine(); m.u.mem_write(s3.STATUS_VA, words(4)); m.u.mem_write(s3.STATUS_VA+0x1c, words(count))
                m.run_frames(4); self.assertEqual(m.rects, [CENTER]*4)

    def test_uninitialized_and_invalid_status_center(self):
        for ready, state in ((False,4),(True,0xffffffff),(True,9)):
            with self.subTest(ready=ready,state=state):
                m=TelemetryMachine(boot_seen=ready); m.u.mem_write(s3.STATUS_VA, words(state)); m.u.mem_write(s3.STATUS_VA+0x1c, words(4))
                m.run_frames(4); self.assertEqual(m.rects, [CENTER]*4); self.assertFalse(m.calls('get'))

    def test_positions_stay_within_smallest_and_odd_geometry(self):
        for width,height in ((48,48),(73,57),(257,139),(65535,65535)):
            m=TelemetryMachine(); m.u.mem_write(FD+0xfc, struct.pack('<HH',width,height))
            m.u.mem_write(s3.STATUS_VA, words(8)); m.run_frames(2)
            left,top,w,h=m.rects[0]
            self.assertEqual(m.rects, [position(1,-1,width,height)]*2)
            self.assertLessEqual(left+w,width<<8); self.assertLessEqual(top+h,height<<8)

    def test_invalid_frame_skips_dsp_and_overlay(self):
        m=TelemetryMachine(); m.valid_render=False
        m.u.mem_write(FD+0xfc, struct.pack('<HH',47,212))
        m.run_frames(40)
        self.assertEqual(m.rects, [(0,)*4]*40); self.assertFalse(m.calls('get'))

    def test_rejects_nonzero_neighbor_wrong_source_and_extra_difference(self):
        base=c.build_continuity_cave2(); size=len(t.build_position_helper())
        with self.assertRaisesRegex(ValueError, 'zero space'):
            t.patch_cave(base, b'\1'+bytes(size-1))
        with self.assertRaisesRegex(ValueError, 'exact audited'):
            build.build(bytes(39_109_096))
        before=bytes(0x20000); after=bytearray(before); after[0x18ba0]=1
        with self.assertRaisesRegex(ValueError, 'before render_branch'):
            check_differences(before, after, DELTAS, 'synthetic')

    @unittest.skipUnless(os.environ.get('GR4_TELEMETRY_PACKAGED'), 'optional physical candidate')
    def test_packaged_telemetry_bytes_execute(self):
        raw=Path(os.environ['GR4_TELEMETRY_PACKAGED']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '327a93285fd312887994c1f79bec313be920f012d890e56f386edb7b512c259e')
        payload=fw.verify_container(raw)
        m=TelemetryMachine(boot_seen=False)
        for va,size in ((s3.CAVE1_VA,100),(s3.CAVE2_VA,1452),(s3.HOOK2_SITE_VA,8),(c.HOOK_UPDATE_VA,4)):
            off=s3.RTOS_PAYLOAD_START+va-s3.GR4_BASE
            m.u.mem_write(va, payload[off:off+size])
        m.boot_init(); m.run_frames(80)
        self.assertEqual(m.rects, [CENTER]*36+[position(1,0)]*44)
        self.assertEqual(m.calls('send'), [21,25,29,33]); self.assertEqual(m.calls('wait'), [24,28,32,36])
