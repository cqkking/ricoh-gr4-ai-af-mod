#!/usr/bin/env python3
"""Build the Step 2C continuity test image from the pinned official GR IV v1.20."""

import argparse
from pathlib import Path

from build_step2c_osd_recovery import fw, write_new
import step2c_osd_recovery as osd
from step2c_continuity_patcher import apply_continuity_experiment

VERSION = 'v1.2.11-step2c-continuity-test'
FW_NAME = 'fwdc248b_step2c_continuity_v1.2.11.bin'


def build(stock: bytes) -> bytes:
    if len(stock) != osd.OFFICIAL_SIZE or fw.sha256_hex(stock) != osd.OFFICIAL_SHA256:
        raise ValueError('Wrong stock GR IV v1.20 size or SHA-256')
    before, frames, consumed = fw.decode_container(stock)
    fw.verify_container(stock, before, consumed)
    rtos, = (x for x in fw.parse_sections(before) if x['name'] == 'RTOS')
    patched = bytearray(before)
    apply_continuity_experiment(patched, rtos['data_start'], rtos['data_end'])
    patched = fw.rezero_payload(patched)
    trailer_at = (consumed + 3) & ~3
    if stock[consumed:trailer_at] != bytes(trailer_at-consumed):
        raise ValueError('Nonzero stock frame-stream alignment padding')
    source = stock[:consumed] + stock[trailer_at:]
    output, _ = fw.rebuild_container(source, frames, consumed, patched, fw.diff_offsets(before, patched))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.parent.resolve() == args.stock.parent.resolve():
        p.error('use a new output file in an existing isolated directory')
    image = build(args.stock.read_bytes())
    write_new(args.output, image)
    print(VERSION, len(image), fw.sha256_hex(image))


if __name__ == '__main__':
    main()
