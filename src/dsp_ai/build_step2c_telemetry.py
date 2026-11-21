"""Build a local v1.2.12 telemetry candidate from the exact v1.2.11 image."""
import argparse
from pathlib import Path

from build_step2c_osd_recovery import fw, write_new
import step2c_telemetry_patcher as t

VERSION = 'v1.2.12-step2c-dsp-telemetry'
BASE_SHA = 'd385fffe1dbf8ebdd6a0aacb62ffbf1c9bf6e466948b43bdd02643a3f3df9284'
FW_NAME = 'fwdc248b_step2c_telemetry_v1.2.12.bin'


def build(base: bytes) -> bytes:
    if len(base) != 39_109_096 or fw.sha256_hex(base) != BASE_SHA:
        raise ValueError('Wrong source: require exact audited v1.2.11 firmware')
    before, frames, consumed = fw.decode_container(base)
    fw.verify_container(base, before, consumed)
    rtos, = (p for p in fw.parse_sections(before) if p['name'] == 'RTOS')
    if (rtos['data_start'], rtos['data_end']) != (0x17d28, 0x1423328):
        raise ValueError('Unexpected RTOS bounds')
    start = rtos['data_start'] + t.s3.CAVE2_VA - t.s3.GR4_BASE
    end = start + t.BASE_LEN
    size = len(t.build_position_helper())
    cave, helper = t.patch_cave(before[start:end], before[end:end+size])
    data = bytearray(before)
    data[start:end+size] = cave + helper
    patched = fw.rezero_payload(data)
    trailer = (consumed+3) & ~3
    if base[consumed:trailer] != bytes(trailer-consumed):
        raise ValueError('Invalid alignment padding')
    source = base[:consumed] + base[trailer:]
    candidate, _ = fw.rebuild_container(source, frames, consumed, patched, fw.diff_offsets(before, patched))
    return candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.parent.resolve() == args.base.parent.resolve():
        p.error('use a new output in an existing isolated directory')
    candidate = build(args.base.read_bytes())
    write_new(args.output, candidate)
    print(VERSION, len(candidate), fw.sha256_hex(candidate))


if __name__ == '__main__':
    main()
