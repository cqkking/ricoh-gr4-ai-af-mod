#!/usr/bin/env python3
"""Build a local Step 2C OSD-only image from the pinned stock GR IV v1.20."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import fw_container as fw
import step2c_osd_recovery as osd

VERSION = "v1.2.10-step2c-osd-recovery"


def build(source: bytes) -> bytes:
    if len(source) != osd.OFFICIAL_SIZE or fw.sha256_hex(source) != osd.OFFICIAL_SHA256:
        raise ValueError("Expected unmodified RICOH GR IV v1.20 (pinned size and SHA-256)")
    stock, frames, consumed = fw.decode_container(source)
    fw.verify_container(source, stock, consumed)
    rtos, = (section for section in fw.parse_sections(stock) if section["name"] == "RTOS")
    payload = bytearray(stock)
    osd.apply_to_decoded_payload(payload, rtos["data_start"], rtos["data_end"])
    patched = fw.rezero_payload(payload)
    changed = fw.diff_offsets(stock, patched)

    # The source stream ends unaligned. Remove only its alignment zeros here,
    # so the rebuilder adds canonical padding for the new stream length.
    trailer_start = (consumed + 3) & ~3
    if source[consumed:trailer_start] != bytes(trailer_start - consumed):
        raise ValueError("Unexpected bytes in alignment padding before the trailer")
    normalized_source = source[:consumed] + source[trailer_start:]
    candidate, _ = fw.rebuild_container(normalized_source, frames, consumed, patched, changed)
    return candidate


def write_new(path: Path, data: bytes) -> None:
    """Publish a complete file without replacing an existing target or sample."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".osd-", delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        # Hard link creates the destination atomically and rejects an existing
        # path even if concurrently created. The tempfile is on the same volume.
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.stock.resolve():
        parser.error("output must be a new file in a separate local directory")
    if args.output.parent.resolve() == args.stock.parent.resolve():
        parser.error("do not write to the stock sample directory")
    candidate = build(args.stock.read_bytes())
    write_new(args.output, candidate)
    print(f"{VERSION} {len(candidate)} bytes {fw.sha256_hex(candidate)}")


if __name__ == "__main__":
    main()
