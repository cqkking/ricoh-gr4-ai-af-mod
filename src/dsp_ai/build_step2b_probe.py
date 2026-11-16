#!/usr/bin/env python3
"""
build_step2b_probe.py - Build & Verify Step 2B Non-Blocking LiveView AI Probe Firmware
"""

import json
import shutil
import sys
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

REPO_ROOT = Path(r"d:\Ricoh GR4 firmware Mod")
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
OFFICIAL_FW = REPO_ROOT / "firmware" / "official" / "v1.20" / "fwdc248b.bin"
OUT_NAME = "fwdc248b_step2b_live_probe.bin"
WORKTREE_OUT = WORKTREE_ROOT / "output" / OUT_NAME
ROOT_OUT = REPO_ROOT / "output" / OUT_NAME
ACTIVE_OUT = WORKTREE_ROOT / "output" / "fwdc248b.bin"
SD_DIR = WORKTREE_ROOT / "output" / "sd_card"
MANIFEST_PATH = WORKTREE_ROOT / "analysis" / "step2b_probe_manifest.json"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fw_container import (
    decode_container,
    verify_container,
    rebuild_container,
    rezero_payload,
    diff_offsets,
    sha256_hex,
    word_sum,
)
from dsp_splitter import extract_dsp_files
from verify_dsp_payload import verify_dsp
import stage3_ai_patcher as s3
from step2b_live_probe_patcher import apply_step2b_patch


def main():
    sd_manifest = extract_dsp_files(out_dir=SD_DIR)
    verify_dsp()
    root_sd = REPO_ROOT / "output" / "sd_card"
    root_sd.mkdir(parents=True, exist_ok=True)
    for f in SD_DIR.glob("*.BIN"):
        shutil.copy2(f, root_sd / f.name)

    raw_fw = OFFICIAL_FW.read_bytes()
    payload, frames, consumed = decode_container(raw_fw)
    verify_container(raw_fw, payload, consumed)

    buf = bytearray(payload)
    patch_info = apply_step2b_patch(buf, rtos_payload_offset=s3.RTOS_PAYLOAD_START)

    patched = rezero_payload(bytes(buf))
    assert word_sum(patched) == 0, "Inner payload word_sum != 0"

    changed = diff_offsets(payload, patched)
    out_container, _ = rebuild_container(raw_fw, frames, consumed, patched, changed)

    again, _, used = decode_container(out_container)
    assert again == patched, "Round-trip mismatch!"
    verify_container(out_container, again, used)
    assert word_sum(out_container) == 0, "Outer container word_sum != 0"

    WORKTREE_OUT.parent.mkdir(parents=True, exist_ok=True)
    WORKTREE_OUT.write_bytes(out_container)
    ACTIVE_OUT.write_bytes(out_container)
    ROOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    ROOT_OUT.write_bytes(out_container)

    manifest = {
        "firmware": {
            "path": str(ROOT_OUT),
            "size": len(out_container),
            "sha256": sha256_hex(out_container),
            "modified_payload_bytes": len(changed),
            "word_sum": "0x0",
        },
        "sd_card_files": sd_manifest["files"],
        "patch": patch_info,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    h2 = patch_info["dual_hooks"]["hook2_continuous_ai_af"]
    t2_off = s3.HOOK2_SITE_VA - s3.GR4_BASE + s3.RTOS_PAYLOAD_START
    print(f"[+] Built {OUT_NAME}: {len(out_container):,} B, SHA-256 = {manifest['firmware']['sha256']}")
    print(f"[+] Cave 2: {h2['cave_length']} B @ {h2['cave_vaddr']}, dual word_sum == 0 PASS")
    for ins in md.disasm(again[t2_off : t2_off + 8], s3.HOOK2_SITE_VA):
        print(f"    Hook 2 0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")


if __name__ == "__main__":
    main()
