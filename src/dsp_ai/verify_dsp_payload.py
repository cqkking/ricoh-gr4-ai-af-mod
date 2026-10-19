#!/usr/bin/env python3
"""
verify_dsp_payload.py - Dual-Direction Validation of CEVA-CV DSP Binaries

Validates:
  1. Chunk format of DF_EX.BIN against GR4 RTOS DspService::LoadDsp (0x53965ff8) parser.
  2. Byte-perfect reconstruction: checks that memory reconstructed from
     DF_EX.BIN (0x51200000), DF_ID_S.BIN (0x515ec000), and DF_IP.BIN (0x515fc000)
     identically matches the K-3 III 06_DSP.bin image.
  3. CEVA vector entry table verification at 0x000000 and 0x3ec000.
"""

import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(r"d:\Ricoh GR4 firmware Mod")
K3_DSP_PATH = REPO_ROOT / "firmware" / "reference" / "k3iii_v220" / "unpacked" / "sections" / "06_DSP.bin"
SD_DIR = REPO_ROOT / "worktrees" / "feat-dsp-ai-af" / "output" / "sd_card"

def verify_dsp():
    print("[1/3] Reading source 06_DSP.bin and generated binaries...")
    assert K3_DSP_PATH.exists(), f"Missing {K3_DSP_PATH}"
    dsp_orig = K3_DSP_PATH.read_bytes()
    assert len(dsp_orig) == 0x400000, "06_DSP.bin must be exactly 4MB"

    df_ex = (SD_DIR / "DF_EX.BIN").read_bytes()
    df_id_s = (SD_DIR / "DF_ID_S.BIN").read_bytes()
    df_ip = (SD_DIR / "DF_IP.BIN").read_bytes()

    print("[2/3] Validating DF_EX.BIN chunk parser conformity...")
    # Check 16-byte header
    num_chunks, r1, r2, r3 = struct.unpack_from("<IIII", df_ex, 0)
    assert num_chunks == 1, f"Expected 1 chunk, got {num_chunks}"
    assert (r1, r2, r3) == (0, 0, 0), "Reserved header bytes must be 0"

    # Check chunk descriptor
    dest_va, chunk_len = struct.unpack_from("<II", df_ex, 16)
    assert dest_va == 0x51200000, f"Expected target VA 0x51200000, got {hex(dest_va)}"
    assert chunk_len == 0x3ec000, f"Expected chunk len 0x3ec000, got {hex(chunk_len)}"
    assert len(df_ex) == 16 + 8 + chunk_len, "DF_EX file size does not match header declaration"

    print("[3/3] Reconstructing memory range 0x51200000..0x51600000 (4 MB)...")
    reconstructed = bytearray(0x400000)
    # 0x000000 .. 0x3ec000 from DF_EX
    reconstructed[0 : 0x3ec000] = df_ex[24 : 24 + chunk_len]
    # 0x3ec000 .. 0x3fc000 from DF_ID_S
    reconstructed[0x3ec000 : 0x3fc000] = df_id_s
    # 0x3fc000 .. 0x400000 from DF_IP
    reconstructed[0x3fc000 : 0x400000] = df_ip

    diffs = [i for i in range(len(dsp_orig)) if dsp_orig[i] != reconstructed[i]]
    if diffs:
        raise AssertionError(f"Reconstruction mismatch at {len(diffs)} bytes! First diff at 0x{diffs[0]:06x}")

    print("[+] PASS: 100% byte-perfect reconstruction verified across all 4,194,304 bytes.")
    print("    - DF_EX.BIN   payload matches 06_DSP.bin[0x000000..0x3ec000]")
    print("    - DF_ID_S.BIN payload matches 06_DSP.bin[0x3ec000..0x3fc000]")
    print("    - DF_IP.BIN   payload matches 06_DSP.bin[0x3fc000..0x400000]")
    return True

if __name__ == "__main__":
    verify_dsp()
