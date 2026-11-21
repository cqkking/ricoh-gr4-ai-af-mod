#!/usr/bin/env python3
"""Read-only Step 2C OSD package audit using a separate framed-LZ decoder."""

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import struct

STOCK_BYTES = 38_919_904
STOCK_SHA = "7563d7e462f0c53b6ab2fbab7cdfbbb57151a41192ee82f4f3a2108f9d77650d"
DECODED_BYTES = 69_632_740
RTOS_HEADER = 0x17D18
RTOS_START = 0x17D28
RTOS_END = 0x1423328
EXPECTED_PATCH = (
    (0x53000E00, 0x18B28, bytes(280), "2f1bbf0031af7381546ef4cf3a38ec11a27a569da03ea9b97fe81eaef1682ae6"),
    (0x5389639C, 0x8AE0C4, bytes.fromhex("380094e57b40ffeb"), sha256(bytes.fromhex("97aaddea0000a0e1")).hexdigest()),
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def inspect(raw, decoder):
    require(raw[:20] == b"RICOH\0\0\0GR IV" + bytes(7), "Not the pinned GR IV model")
    require(raw[0x38:0x3c] == bytes((1, 20, 10, 11)), "Unexpected internal version")
    payload, frames, used = decoder.decode_frames(raw, max_output_size=DECODED_BYTES)
    require(len(payload) == DECODED_BYTES, "Unexpected decoded length")
    require(decoder.additive_checksum(raw) == 0 and decoder.additive_checksum(payload) == 0,
            "Container or payload checksum failed")
    trailer_start = (used + 3) & ~3
    require(raw[used:trailer_start] == bytes(trailer_start - used), "Invalid alignment padding")
    trailer = raw[trailer_start:]
    require(len(trailer) == 408 and trailer[-20:-16] == bytes.fromhex("a55a5aa5"), "Invalid trailer")
    require(struct.unpack_from("<II", trailer, len(trailer)-12) == (used-0x80, len(payload)),
            "Framing lengths disagree with trailer")
    require(payload[RTOS_HEADER:RTOS_HEADER+8] == b"RTOS\0\0\0\0", "Missing fixed RTOS header")
    require(struct.unpack_from("<I", payload, RTOS_HEADER+12)[0] == RTOS_END-RTOS_START,
            "Unexpected RTOS section length")
    return payload, frames, trailer


def audit(stock, candidate, decoder):
    require(len(stock) == STOCK_BYTES and sha256(stock).hexdigest() == STOCK_SHA, "Wrong stock sample")
    before, old_frames, tail = inspect(stock, decoder)
    after, frames, new_tail = inspect(candidate, decoder)
    require(stock[:0x80] == candidate[:0x80] and tail[:-12] == new_tail[:-12],
            "Header or non-derived trailer fields changed")
    scope = []
    cursor = 0
    for va, off, original, expected_hash in EXPECTED_PATCH:
        end = off + len(original)
        require(RTOS_START <= off < end <= RTOS_END, "Allowlist span outside RTOS")
        require(before[off:end] == original, f"Unexpected original bytes at {va:#x}")
        require(sha256(after[off:end]).hexdigest() == expected_hash, f"Unexpected patched bytes at {va:#x}")
        require(before[cursor:off] == after[cursor:off], "Unauthorized payload difference between spans")
        scope.append({"partition": "RTOS", "va": hex(va), "offset": hex(off), "length": len(original),
                      "original_sha256": sha256(original).hexdigest(), "patched_sha256": expected_hash})
        cursor = end
    require(before[cursor:-4] == after[cursor:-4], "Unauthorized payload difference after Hook 2")
    require(len(old_frames) == len(frames), "Changed frame count")
    literal_reemits = []
    for i, (a, b) in enumerate(zip(old_frames, frames)):
        start, end = a["output_start"], a["output_end"]
        require((start, end) == (b["output_start"], b["output_end"]), f"Changed decoded frame bounds at {i}")
        reused = stock[a["file_start"]:a["file_end"]] == candidate[b["file_start"]:b["file_end"]]
        if reused:
            continue
        # A changed span in the preceding copy window can affect this frame.
        permit = any(off < end and off+len(orig) > start-0x2000 for _, off, orig, _ in EXPECTED_PATCH)
        require((permit or end == len(after)) and b["type"] == "literal" and 0 < end-start <= 0x6000,
                f"Unexpected re-emission at frame {i}")
        literal_reemits.append(i)
    return {
        "status": "gate_1_offline_audit_pass",
        "gate_2": "not_authorized",
        "model": "GR IV",
        "internal_version": "1.20.10.11",
        "stock_sha256": STOCK_SHA,
        "candidate_sha256": sha256(candidate).hexdigest(),
        "candidate_size": len(candidate),
        "decoded_size": len(after),
        "changed_payload_bytes_including_checksum": sum(x != y for _, off, orig, _ in EXPECTED_PATCH
                                                         for x, y in zip(orig, after[off:off+len(orig)]))
                                                      + sum(x != y for x, y in zip(before[-4:], after[-4:])),
        "rtos_replacements": scope,
        "derived": {"payload_checksum_offset": hex(len(before)-4), "encoded_size_and_outer_checksum": True},
        "frames": len(frames),
        "literal_reemits_zero_based": literal_reemits,
        "boot_and_non_target_payload": "identical_except_derived_payload_checksum",
        "camera_verification": "not_run",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stock", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True, help="Audited external read-only unpack_gr4_firmware.py")
    p.add_argument("--expect-sha256", help="Reject any result not matching this frozen candidate hash")
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("independent_gr4_decoder", args.decoder)
    if spec is None or spec.loader is None:
        p.error("unable to load independent decoder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = audit(args.stock.read_bytes(), args.candidate.read_bytes(), module)
    require(not args.expect_sha256 or result["candidate_sha256"] == args.expect_sha256.lower(),
            "Unexpected candidate SHA-256")
    result["independent_decoder_sha256"] = sha256(args.decoder.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
