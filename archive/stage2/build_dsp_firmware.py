#!/usr/bin/env python3
"""
build_dsp_firmware.py - Pack Verified DSP Auto-Wakeup Firmware (fwdc248b.bin)

Generates:
  1. Staged SD card payload (output/sd_card/):
     - DF_EX.BIN (4,112,408 bytes)
     - DF_ID_S.BIN (65,536 bytes)
     - DF_IP.BIN (16,384 bytes)
  2. Flashable firmware image (output/fwdc248b.bin):
     - Patches 05_RTOS.bin with the Auto-Wakeup Trampoline & Code Cave (0x53000e00)
     - Dual 32-bit word-sum checksum validation (word_sum == 0)
     - Flashable via official SD card update process (FWUPD)
  3. Release manifest (analysis/dsp_wake_manifest.json)
"""

import json
import struct
import sys
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

REPO_ROOT = Path(r"d:\Ricoh GR4 firmware Mod")
WORKTREE_ROOT = REPO_ROOT / "worktrees" / "feat-dsp-ai-af"
TOOLS_DIR = REPO_ROOT / "tools"
OFFICIAL_FW = REPO_ROOT / "firmware" / "official" / "v1.11" / "fwdc248b.bin"
OUTPUT_DIR = WORKTREE_ROOT / "output"
OUTPUT_FW = OUTPUT_DIR / "fwdc248b.bin"
SD_DIR = OUTPUT_DIR / "sd_card"
MANIFEST_PATH = WORKTREE_ROOT / "analysis" / "dsp_wake_manifest.json"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(WORKTREE_ROOT / "src" / "dsp_ai"))
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
from k3_gr4_dsp_patcher import (
    build_cave_payload,
    build_trampoline_payload,
    CAVE_VA,
    HOOK_SITE_VA,
    RETURN_SITE_VA,
    GR4_BASE,
)

RTOS_PAYLOAD_START = 0x00017D10

def v2p(vaddr: int) -> int:
    return vaddr - GR4_BASE + RTOS_PAYLOAD_START

def build_dsp_release():
    print("=" * 65)
    print("Ricoh GR IV CEVA-CV AI DSP Auto-Wakeup Firmware Builder (Stage 2)")
    print("=" * 65)

    # 1. Extract and stage SD card binaries
    print("\n[Step 1/4] Extracting and staging DSP split binaries...")
    sd_manifest = extract_dsp_files(out_dir=SD_DIR)
    print(f"  -> Generated DF_EX.BIN, DF_ID_S.BIN, DF_IP.BIN in {SD_DIR}")
    verify_dsp()

    # 2. Decode official firmware container
    print("\n[Step 2/4] Decoding official firmware container...")
    raw_fw = OFFICIAL_FW.read_bytes()
    payload, frames, consumed = decode_container(raw_fw)
    verify_container(raw_fw, payload, consumed)
    print(f"  -> Official payload decoded: {len(payload):,} bytes")

    # 3. Patch payload with Auto-Wakeup Cave and Trampoline
    print("\n[Step 3/4] Splicing DSP Auto-Wakeup Trampoline & Code Cave...")
    buf = bytearray(payload)
    cave_payload = build_cave_payload()
    tramp_payload = build_trampoline_payload()

    # Check and patch Cave @ 0x53000e00
    cave_poff = v2p(CAVE_VA)
    assert buf[cave_poff : cave_poff + len(cave_payload)] == b"\x00" * len(cave_payload), "Cave not empty!"
    buf[cave_poff : cave_poff + len(cave_payload)] = cave_payload
    print(f"  [+] Injected {len(cave_payload)} B Cave Payload @ VA {hex(CAVE_VA)} (payload offset {hex(cave_poff)})")

    # Check and patch Trampoline @ 0x536520f0 (LiveViewProcess3aContoller)
    tramp_poff = v2p(HOOK_SITE_VA)
    expected_hook = bytes.fromhex("0dc0a0e1f0d82de9") # mov ip, sp; push {r4..pc}
    actual_hook = bytes(buf[tramp_poff : tramp_poff + len(expected_hook)])
    assert actual_hook == expected_hook, f"Hook mismatch: {actual_hook.hex()} != {expected_hook.hex()}"
    buf[tramp_poff : tramp_poff + len(tramp_payload)] = tramp_payload
    print(f"  [+] Injected {len(tramp_payload)} B Trampoline @ VA {hex(HOOK_SITE_VA)} (payload offset {hex(tramp_poff)})")

    # Rezero checksum word
    print("[*] Rezeroing 32-bit payload checksum word...")
    patched_payload = rezero_payload(bytes(buf))
    assert word_sum(patched_payload) == 0, f"Payload word_sum != 0 (got {word_sum(patched_payload):#x})"
    print("  [+] Payload word_sum == 0 validated")

    # Rebuild outer container
    print("\n[Step 4/4] Rebuilding flashable container with LZSS framing...")
    changed = diff_offsets(payload, patched_payload)
    print(f"  -> Modified payload bytes: {len(changed)}")
    out_container, reemitted = rebuild_container(raw_fw, frames, consumed, patched_payload, changed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FW.write_bytes(out_container)
    print(f"  [+] Written flashable image: {OUTPUT_FW} ({len(out_container):,} bytes, {reemitted} re-emitted frames)")

    # Full Round-Trip Verification
    print("[*] Executing round-trip container verification...")
    again, _, used = decode_container(out_container)
    assert again == patched_payload, "Round-trip payload mismatch!"
    verify_container(out_container, again, used)
    assert word_sum(out_container) == 0, "Outer container word_sum != 0!"
    print("  [+] Container verification PASSED! Dual word-sums == 0.")

    # Save release manifest
    manifest = {
        "release_type": "Stage 2 Automated Boot-Time CEVA-CV DSP Awakening",
        "official_fw_sha256": sha256_hex(raw_fw),
        "flashable_firmware": {
            "path": str(OUTPUT_FW),
            "size": len(out_container),
            "sha256": sha256_hex(out_container),
        },
        "sd_card_dependencies": sd_manifest["files"],
        "patch_details": {
            "hook_va": hex(HOOK_SITE_VA),
            "cave_va": hex(CAVE_VA),
            "return_va": hex(RETURN_SITE_VA),
            "cave_size_bytes": len(cave_payload),
            "trampoline_bytes": tramp_payload.hex(),
        },
        "execution_model": (
            "Camera boots -> LiveView starts -> Trampoline branches to 0x53000e00 -> "
            "Loads DF_EX/ID_S/IP from SD card (A:\\) -> Unresets and starts CEVA-CV DSP -> "
            "Runs CEVA Version test -> Returns to LiveView sequence cleanly."
        )
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] Saved release manifest to: {MANIFEST_PATH}")

    # Disassembly verification on final patched binary
    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    print("\n=== Disassembly of Rebuilt LiveView Hook in Flashable Image ===")
    tramp_in_fw = again[tramp_poff : tramp_poff + 8]
    for ins in md.disasm(tramp_in_fw, HOOK_SITE_VA):
        print(f"    0x{ins.address:08x}: {ins.mnemonic:8s} {ins.op_str}")

    print("\n=== Disassembly of Rebuilt Code Cave in Flashable Image ===")
    cave_in_fw = again[cave_poff : cave_poff + len(cave_payload)]
    for ins in md.disasm(cave_in_fw, CAVE_VA):
        print(f"    0x{ins.address:08x}: {ins.mnemonic:8s} {ins.op_str}")

    print("\n" + "=" * 65)
    print("SUCCESS: Flashable fwdc248b.bin and SD card payload ready!")
    print("=" * 65)

if __name__ == "__main__":
    build_dsp_release()
