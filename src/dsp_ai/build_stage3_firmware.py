#!/usr/bin/env python3
"""
build_stage3_firmware.py - Pack Verified Stage 3 AI Live View Tracking Firmware

Builds:
  1. Preserved Stage 3 Firmware: output/fwdc248b_stage3_ai_track.bin
  2. Active Flashable Firmware:   output/fwdc248b.bin
     (Ensures Stage 2 binary output/fwdc248b_stage2_dsp_wake.bin is strictly preserved!)
  3. Release Manifest:           analysis/dsp_stage3_manifest.json
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
STAGE2_FW = OUTPUT_DIR / "fwdc248b_stage2_dsp_wake.bin"
STAGE3_FW = OUTPUT_DIR / "fwdc248b_stage3_ai_track.bin"
ACTIVE_FW = OUTPUT_DIR / "fwdc248b.bin"
SD_DIR = OUTPUT_DIR / "sd_card"
MANIFEST_PATH = WORKTREE_ROOT / "analysis" / "dsp_stage3_manifest.json"

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
from stage3_ai_patcher import (
    apply_stage3_patch,
    CAVE_VA,
    HOOK_SITE_VA,
    RETURN_SITE_VA,
    DESC_TABLE_VA,
    MEAN_TABLE_VA,
    STD_TABLE_VA,
    STATUS_VA,
    TENSOR_IN_VA,
    TENSOR_OUT_VA,
    UPDATE_TRACK_VA,
    GR4_BASE,
)

RTOS_PAYLOAD_START = 0x00017D10

def v2p(vaddr: int) -> int:
    return vaddr - GR4_BASE + RTOS_PAYLOAD_START

def build_stage3_release():
    print("=" * 70)
    print("Ricoh GR IV Stage 3 AI Live View Tracking Firmware Rebuilder")
    print("=" * 70)

    # 0. Safety Guard: Check that Stage 2 binary exists and will not be overwritten
    assert STAGE2_FW.exists(), f"Error: Stage 2 binary {STAGE2_FW} missing!"
    stage2_bytes_len = len(STAGE2_FW.read_bytes())
    print(f"[*] Confirmed Stage 2 binary intact: {STAGE2_FW.name} ({stage2_bytes_len:,} bytes)")

    # 1. Verify SD card payload files
    print("\n[Step 1/4] Verifying staged SD card DSP binaries...")
    sd_manifest = extract_dsp_files(out_dir=SD_DIR)
    verify_dsp()
    print(f"  [+] All 3 SD card binaries verified in {SD_DIR}")

    # 2. Decode official firmware container
    print("\n[Step 2/4] Decoding official firmware container baseline...")
    raw_fw = OFFICIAL_FW.read_bytes()
    payload, frames, consumed = decode_container(raw_fw)
    verify_container(raw_fw, payload, consumed)
    print(f"  -> Official payload decoded: {len(payload):,} bytes")

    # 3. Apply Stage 3 Deep Learning Tracking Patches
    print("\n[Step 3/4] Splicing Stage 3 Code Cave, Live View Trampoline & AI Tables...")
    buf = bytearray(payload)
    patch_manifest = apply_stage3_patch(buf, rtos_payload_offset=RTOS_PAYLOAD_START)

    print(f"  [+] Injected Code Cave ({patch_manifest['code_cave']['length']} B) @ VA {patch_manifest['code_cave']['vaddr']}")
    print(f"  [+] Injected Trampoline ({len(bytes.fromhex(patch_manifest['trampoline']['patched_hex']))} B) @ VA {patch_manifest['trampoline']['vaddr']}")
    print(f"  [+] Embedded Desc Table ({patch_manifest['embedded_tables']['desc_table']['length']} B) @ VA {patch_manifest['embedded_tables']['desc_table']['vaddr']}")
    print(f"  [+] Embedded Mean Table ({patch_manifest['embedded_tables']['mean_table']['length']} B) @ VA {patch_manifest['embedded_tables']['mean_table']['vaddr']}")
    print(f"  [+] Embedded Std Table  ({patch_manifest['embedded_tables']['std_table']['length']} B) @ VA {patch_manifest['embedded_tables']['std_table']['vaddr']}")

    # Rezero checksum word
    print("[*] Rezeroing 32-bit payload checksum word...")
    patched_payload = rezero_payload(bytes(buf))
    assert word_sum(patched_payload) == 0, f"Payload word_sum != 0 (got {word_sum(patched_payload):#x})"
    print("  [+] Payload word_sum == 0 validated successfully.")

    # Rebuild outer container
    print("\n[Step 4/4] Rebuilding flashable container with LZSS framing...")
    changed = diff_offsets(payload, patched_payload)
    print(f"  -> Modified payload bytes: {len(changed)}")
    out_container, reemitted = rebuild_container(raw_fw, frames, consumed, patched_payload, changed)

    # Write both Stage 3 preserved binary and active firmware
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGE3_FW.write_bytes(out_container)
    ACTIVE_FW.write_bytes(out_container)
    print(f"  [+] Written Stage 3 preserved binary: {STAGE3_FW} ({len(out_container):,} bytes)")
    print(f"  [+] Written active flashable binary:  {ACTIVE_FW} ({len(out_container):,} bytes)")

    # Full Round-Trip Verification
    print("\n[*] Executing round-trip container verification...")
    again, _, used = decode_container(out_container)
    assert again == patched_payload, "Round-trip payload mismatch!"
    verify_container(out_container, again, used)
    assert word_sum(out_container) == 0, "Outer container word_sum != 0!"
    print("  [+] Round-trip container verification PASSED! Dual word-sums == 0.")

    # Save release manifest
    manifest = {
        "release_type": "Stage 3 Deep Learning Inference Pipeline & Live View Tracking Bridge",
        "official_fw_sha256": sha256_hex(raw_fw),
        "stage2_preserved_fw": {
            "path": str(STAGE2_FW),
            "size": stage2_bytes_len,
            "sha256": sha256_hex(STAGE2_FW.read_bytes()),
        },
        "stage3_firmware": {
            "path": str(STAGE3_FW),
            "size": len(out_container),
            "sha256": sha256_hex(out_container),
        },
        "sd_card_dependencies": sd_manifest["files"],
        "patch_manifest": patch_manifest,
        "runtime_pipeline": {
            "hook_location": "FaceDetectionService::Task (0x5386ea8c) - continuous LiveView AF thread",
            "decimation": "Every 2 Live View frames (~30 fps inference)",
            "frame_source": "CaptureControlService [0x55395a18, #8] -> [m_CaptureViewControlService, #4]",
            "downsampler": "960x640 YUV LiveView -> 56x56 RGB tensor @ 0x51600000",
            "dsp_inference": "msg_id 4 via SendMessage (0x539663c0) & WaitMessage (0x539666dc)",
            "output_heatmap": "64x64 peak scan @ 0x51610000",
            "confidence_threshold": 60,
            "face_result_structure": "m_DetectedFaceResult [r4, #0x78] + face_count [r4, #0x38, #0x30]",
            "af_tracking_target": "UpdateTrackTarget (0x538e32c4) on [r4 + 8] & [0x550b6aa4]",
            "lcd_osd_rendering": "UpdateFaceDetectionResult (0x5386e0e4) renders green AI Face/Eye box on LCD",
            "fallback_mechanism": "FotoNation LibAHFD (0x5383ec84) when confidence < 60",
            "af_system": "GR4 165-point on-sensor PDAF (Zaf) locks onto detected subject centroid"
        }
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] Saved release manifest to: {MANIFEST_PATH}")

    # Disassembly verification on final patched binary
    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    print("\n=== Disassembly of Rebuilt LiveView Hook in Flashable Image ===")
    tramp_poff = v2p(HOOK_SITE_VA)
    tramp_in_fw = again[tramp_poff : tramp_poff + 8]
    for ins in md.disasm(tramp_in_fw, HOOK_SITE_VA):
        print(f"    0x{ins.address:08x}: {ins.mnemonic:8s} {ins.op_str}")

    print("\n=== First 20 Instructions of Code Cave in Flashable Image ===")
    cave_poff = v2p(CAVE_VA)
    cave_in_fw = again[cave_poff : cave_poff + 80]
    for ins in md.disasm(cave_in_fw, CAVE_VA):
        print(f"    0x{ins.address:08x}: {ins.mnemonic:8s} {ins.op_str}")

    print("\n" + "=" * 70)
    print("SUCCESS: Stage 3 AI Live View Tracking Firmware Built & Verified!")
    print("=" * 70)

if __name__ == "__main__":
    build_stage3_release()
