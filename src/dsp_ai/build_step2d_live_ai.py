#!/usr/bin/env python3
"""Build the Step 2D v1.2.16 live-frame continuous CEVA-XM4 AI AF candidate (Option B)."""

import argparse
from pathlib import Path

from build_step2c_osd_recovery import fw, write_new
import build_step2c_slot_poll as slot_build
import step2c_osd_recovery as osd
from step2d_live_ai_patcher import apply_live_ai_patch

VERSION = "v1.2.16-step2d-live-ai"
FW_NAME = "fwdc248b_step2d_live_ai_v1.2.16.bin"


def build_with_manifest(stock: bytes) -> tuple[bytes, dict]:
    if len(stock) != osd.OFFICIAL_SIZE or fw.sha256_hex(stock) != osd.OFFICIAL_SHA256:
        raise ValueError("Wrong stock GR IV v1.20 size or SHA-256")
    before, frames, consumed = fw.decode_container(stock)
    fw.verify_container(stock, before, consumed)
    rtos, = (x for x in fw.parse_sections(before) if x["name"] == "RTOS")

    patched_buf = bytearray(before)
    manifest = apply_live_ai_patch(patched_buf, rtos["data_start"], rtos["data_end"])
    patched = fw.rezero_payload(patched_buf)

    trailer_at = (consumed + 3) & ~3
    if stock[consumed:trailer_at] != bytes(trailer_at - consumed):
        raise ValueError("Nonzero stock frame-stream alignment padding")
    source = stock[:consumed] + stock[trailer_at:]
    output, reemitted = fw.rebuild_container(source, frames, consumed, patched, fw.diff_offsets(before, patched))

    again, _, used = fw.decode_container(output)
    if again != patched:
        raise ValueError("Round-trip decoded payload mismatch")
    fw.verify_container(output, again, used)
    if again[: rtos["data_start"]] != before[: rtos["data_start"]]:
        raise ValueError("Bytes before RTOS were modified")
    if again[rtos["data_end"] : -4] != before[rtos["data_end"] : -4]:
        raise ValueError("Non-checksum bytes after RTOS were modified")

    manifest.update({
        "version": VERSION,
        "stock_sha256": fw.sha256_hex(stock),
        "output_sha256": fw.sha256_hex(output),
        "output_size": len(output),
        "reemitted_frames": reemitted,
        "outer_word_sum": fw.word_sum(output),
        "inner_word_sum": fw.word_sum(again),
    })
    return output, manifest


def build(stock: bytes) -> bytes:
    output, _ = build_with_manifest(stock)
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stock", type=Path, required=True)
    p.add_argument("--output", type=Path, help="Single output firmware path (isolated directory)")
    p.add_argument("--bundle-dir", type=Path, help="Emit TEST_FILES bundle with fwdc248b.bin + DF_*.BIN + SHA256SUMS.txt")
    p.add_argument("--dsp-dir", type=Path, help="Directory containing DF_EX.BIN, DF_ID_S.BIN, DF_IP.BIN")
    args = p.parse_args()

    if bool(args.output) == bool(args.bundle_dir):
        p.error("specify exactly one of --output or --bundle-dir")
    if args.output and (args.output.exists() or args.output.parent.resolve() == args.stock.parent.resolve()):
        p.error("use a new output file in an existing isolated directory")
    if args.bundle_dir:
        if not args.dsp_dir:
            p.error("--dsp-dir is required with --bundle-dir")
        if args.bundle_dir.resolve() == args.stock.parent.resolve():
            p.error("do not write bundle into stock firmware directory")

    image, manifest = build_with_manifest(args.stock.read_bytes())
    if args.output:
        write_new(args.output, image)
    else:
        slot_build.FW_NAME = FW_NAME
        slot_build.write_test_bundle(image, args.bundle_dir, args.dsp_dir)
    print(
        f"{VERSION} {manifest['output_size']} bytes {manifest['output_sha256']} "
        f"(reemitted={manifest['reemitted_frames']}, word_sums=({manifest['outer_word_sum']},{manifest['inner_word_sum']}))"
    )


if __name__ == "__main__":
    main()
