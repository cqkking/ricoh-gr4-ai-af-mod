#!/usr/bin/env python3
"""
build_step1_osd_test.py - Build & Verify Step 1 Pure OSD / PDAF Mock Firmware

Outputs:
- worktrees/feat-dsp-ai-af/output/fwdc248b_step1_osd_mock.bin
- output/fwdc248b_step1_osd_mock.bin
- worktrees/feat-dsp-ai-af/analysis/step1_osd_mock_manifest.json
"""

import json
import sys
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

REPO_ROOT = Path(r"d:\Ricoh GR4 firmware Mod")
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
OFFICIAL_FW = REPO_ROOT / "firmware" / "official" / "v1.20" / "fwdc248b.bin"
OUTPUT_NAME = "fwdc248b_step1_osd_mock.bin"
WORKTREE_OUT = WORKTREE_ROOT / "output" / OUTPUT_NAME
ROOT_OUT = REPO_ROOT / "output" / OUTPUT_NAME
MANIFEST_PATH = WORKTREE_ROOT / "analysis" / "step1_osd_mock_manifest.json"

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
from step1_osd_mock_patcher import (
    apply_step1_patch,
    CAVE_VA,
    HOOK2_SITE_VA,
    GR4_BASE,
    RTOS_PAYLOAD_START,
)


def v2p(vaddr: int) -> int:
    return vaddr - GR4_BASE + RTOS_PAYLOAD_START


def main():
    assert OFFICIAL_FW.exists(), f"Missing official v1.20 firmware: {OFFICIAL_FW}"
    raw_fw = OFFICIAL_FW.read_bytes()
    payload, frames, consumed = decode_container(raw_fw)
    verify_container(raw_fw, payload, consumed)

    buf = bytearray(payload)
    patch_info = apply_step1_patch(buf, rtos_payload_offset=RTOS_PAYLOAD_START)

    patched_payload = rezero_payload(bytes(buf))
    assert word_sum(patched_payload) == 0, "Inner payload word_sum != 0!"

    changed = diff_offsets(payload, patched_payload)
    out_container, _ = rebuild_container(raw_fw, frames, consumed, patched_payload, changed)

    again, _, used = decode_container(out_container)
    assert again == patched_payload, "Round-trip payload mismatch!"
    verify_container(out_container, again, used)
    assert word_sum(out_container) == 0, "Outer container word_sum != 0!"

    WORKTREE_OUT.parent.mkdir(parents=True, exist_ok=True)
    WORKTREE_OUT.write_bytes(out_container)
    ROOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    ROOT_OUT.write_bytes(out_container)

    manifest = {
        "firmware": {
            "worktree_path": str(WORKTREE_OUT),
            "root_path": str(ROOT_OUT),
            "size": len(out_container),
            "sha256": sha256_hex(out_container),
            "modified_payload_bytes": len(changed),
            "inner_word_sum": hex(word_sum(patched_payload)),
            "outer_word_sum": hex(word_sum(out_container)),
        },
        "patch": patch_info,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    tramp_off = v2p(HOOK2_SITE_VA)
    cave_off = v2p(CAVE_VA)
    cave_len = patch_info["code_cave"]["length"]

    print(f"[+] Built {OUTPUT_NAME}: {len(out_container):,} bytes, SHA-256 = {manifest['firmware']['sha256']}")
    print(f"[+] Modified payload bytes: {len(changed)}, dual word_sum == 0 PASS")
    print(f"\n=== Rebuilt Hook 2 @ {hex(HOOK2_SITE_VA)} ===")
    for ins in md.disasm(again[tramp_off : tramp_off + 8], HOOK2_SITE_VA):
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")
    print(f"\n=== Rebuilt Step 1 Cave @ {hex(CAVE_VA)} ({cave_len} B) ===")
    for ins in md.disasm(again[cave_off : cave_off + cave_len], CAVE_VA):
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")


if __name__ == "__main__":
    main()
