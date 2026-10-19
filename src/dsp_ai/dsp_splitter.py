#!/usr/bin/env python3
"""
dsp_splitter.py - Extract and Format CEVA-CV DSP Binaries for Ricoh GR IV

Splits Pentax K-3 Mark III's 06_DSP.bin into the exact 3-file payload format
required by Ricoh GR IV RTOS DspService::LoadDsp (0x53966d74):
  1. DF_EX.BIN   - Executable CEVA-CV vector kernels & runtime (loaded @ 0x51200000)
  2. DF_ID_S.BIN - Shared vector parameter / CNN weights table (loaded @ 0x515ec000, 64 KB)
  3. DF_IP.BIN   - Instruction program entry / bootstrap vector (loaded @ 0x515fc000, 16 KB)

Placing these three files on the SD card root (A:\\) enables instant execution via:
  DSP_load [-S][-V] [driveletter]
  DSP_start
  DSP_test 0..3
"""

import hashlib
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(r"d:\Ricoh GR4 firmware Mod")
K3_DSP_PATH = REPO_ROOT / "firmware" / "reference" / "k3iii_v220" / "unpacked" / "sections" / "06_DSP.bin"
DEFAULT_OUT_DIR = REPO_ROOT / "worktrees" / "feat-dsp-ai-af" / "output" / "sd_card"

CHUNK_DEST_VA = 0x51200000
SPLIT_OFF_ID_S = 0x003ec000
SIZE_ID_S = 0x00010000  # 64 KB
SPLIT_OFF_IP = 0x003fc000
SIZE_IP = 0x00004000    # 16 KB
TOTAL_DSP_SIZE = 0x00400000 # 4 MB

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def extract_dsp_files(dsp_binary_path: Path = K3_DSP_PATH, out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    if not dsp_binary_path.exists():
        raise FileNotFoundError(f"Source DSP binary not found: {dsp_binary_path}")

    dsp_bytes = dsp_binary_path.read_bytes()
    if len(dsp_bytes) != TOTAL_DSP_SIZE:
        raise ValueError(f"Unexpected DSP binary size: {len(dsp_bytes)} bytes (expected {TOTAL_DSP_SIZE})")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build DF_EX.BIN
    # Format parsed by GR4 RTOS 0x53965ff8:
    #   Header (16 bytes): uint32 num_chunks, uint32 res[3]
    #   Chunk (8 bytes): uint32 dest_va (0x51200000), uint32 chunk_len (0x3ec000)
    #   Payload: 0x3ec000 bytes
    ex_payload = dsp_bytes[0:SPLIT_OFF_ID_S]
    ex_hdr = struct.pack("<IIII", 1, 0, 0, 0)
    ex_chunk_hdr = struct.pack("<II", CHUNK_DEST_VA, len(ex_payload))
    df_ex_bytes = ex_hdr + ex_chunk_hdr + ex_payload
    df_ex_path = out_dir / "DF_EX.BIN"
    df_ex_path.write_bytes(df_ex_bytes)

    # 2. Build DF_ID_S.BIN (64 KB @ 0x515ec000)
    df_id_s_bytes = dsp_bytes[SPLIT_OFF_ID_S : SPLIT_OFF_ID_S + SIZE_ID_S]
    assert len(df_id_s_bytes) == SIZE_ID_S
    df_id_s_path = out_dir / "DF_ID_S.BIN"
    df_id_s_path.write_bytes(df_id_s_bytes)

    # 3. Build DF_IP.BIN (16 KB @ 0x515fc000)
    df_ip_bytes = dsp_bytes[SPLIT_OFF_IP : SPLIT_OFF_IP + SIZE_IP]
    assert len(df_ip_bytes) == SIZE_IP
    df_ip_path = out_dir / "DF_IP.BIN"
    df_ip_path.write_bytes(df_ip_bytes)

    manifest = {
        "source_dsp": str(dsp_binary_path),
        "source_sha256": sha256_hex(dsp_bytes),
        "source_size": len(dsp_bytes),
        "files": {
            "DF_EX.BIN": {
                "path": str(df_ex_path),
                "size": len(df_ex_bytes),
                "sha256": sha256_hex(df_ex_bytes),
                "target_va": hex(CHUNK_DEST_VA),
                "payload_size": len(ex_payload),
            },
            "DF_ID_S.BIN": {
                "path": str(df_id_s_path),
                "size": len(df_id_s_bytes),
                "sha256": sha256_hex(df_id_s_bytes),
                "target_va": hex(CHUNK_DEST_VA + SPLIT_OFF_ID_S),
            },
            "DF_IP.BIN": {
                "path": str(df_ip_path),
                "size": len(df_ip_bytes),
                "sha256": sha256_hex(df_ip_bytes),
                "target_va": hex(CHUNK_DEST_VA + SPLIT_OFF_IP),
            }
        }
    }

    return manifest

if __name__ == "__main__":
    print(f"[*] Extracting CEVA-CV DSP binaries from {K3_DSP_PATH}...")
    manifest = extract_dsp_files()
    print("[+] Successfully generated all 3 DSP firmware binaries:")
    for fname, meta in manifest["files"].items():
        print(f"    - {fname:12s}: {meta['size']:8d} B -> Load VA {meta['target_va']} (SHA-256: {meta['sha256'][:16]}...)")
