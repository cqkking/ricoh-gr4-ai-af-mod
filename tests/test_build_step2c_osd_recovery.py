"""Pre-release checks for the output path, input pin, and actual packaged A32 code."""
from pathlib import Path
import os
import tempfile
import unittest

from test_step2c_osd_recovery import ArmFrame, fw, osd, recovery
import build_step2c_osd_recovery as builder


class TestOsdRecoveryBuild(unittest.TestCase):
    def test_rejects_wrong_input_before_decode(self):
        with self.assertRaisesRegex(ValueError, 'pinned size and SHA-256'):
            builder.build(bytes(recovery.OFFICIAL_SIZE))

    def test_output_publish_never_overwrites(self):
        with tempfile.TemporaryDirectory() as dirname:
            parent = Path(dirname)
            out = parent / 'candidate.bin'
            builder.write_new(out, b'first')
            with self.assertRaises(FileExistsError):
                builder.write_new(out, b'second')
            self.assertEqual(out.read_bytes(), b'first')
            self.assertEqual(list(parent.iterdir()), [out])

    @unittest.skipUnless(os.environ.get('GR4_OSD_PACKAGED'), 'optional actual staged candidate')
    def test_packaged_cave_executes_100_frames(self):
        raw = Path(os.environ['GR4_OSD_PACKAGED']).read_bytes()
        payload, _, consumed = fw.decode_container(raw)
        fw.verify_container(raw, payload, consumed)
        start = osd.RTOS_PAYLOAD_START + osd.CAVE_VA - osd.GR4_BASE
        hook = osd.RTOS_PAYLOAD_START + osd.HOOK2_SITE_VA - osd.GR4_BASE
        self.assertEqual(payload[hook:hook+8], osd.build_tramp2_payload())
        body = recovery.build_osd_recovery_cave()
        self.assertEqual(payload[start:start+len(body)], body)
        frame = ArmFrame(payload[start:start+len(body)])
        for _ in range(100):
            frame.run()
            self.assertEqual(frame.calls, ['unlock', 'update'])
            self.assertEqual(frame.rect(), (136 << 8, 82 << 8, 48 << 8, 48 << 8))
