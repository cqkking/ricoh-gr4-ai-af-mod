#!/usr/bin/env python3
"""
build_stage3_firmware.py - Pack Verified Pure K-3 III Deep Learning Exclusive AF Firmware (v1.2.3)

Rebased onto official Ricoh GR IV v1.20 baseline firmware.
Stock FotoNation Level 1 AF bypassed 100% in Cave 2 with lower-right Heartbeat Box diagnostic.

Builds:
  1. Preserved Stage 3 Firmware: output/fwdc248b_stage3_ai_track.bin
  2. Active Flashable Firmware:   output/fwdc248b.bin
  3. Release Manifest:           analysis/dsp_stage3_manifest.json
"""

import json
import shutil
import struct
import sys
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

REPO_ROOT = Path(r"d:\Ricoh GR4 firmware Mod")
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
OFFICIAL_FW = REPO_ROOT / "firmware" / "official" / "v1.20" / "fwdc248b.bin"
OUTPUT_DIR = WORKTREE_ROOT / "output"
ROOT_OUTPUT_DIR = REPO_ROOT / "output"
STAGE3_FW = OUTPUT_DIR / "fwdc248b_stage3_ai_track.bin"
ACTIVE_FW = OUTPUT_DIR / "fwdc248b.bin"
SD_DIR = OUTPUT_DIR / "sd_card"
MANIFEST_PATH = WORKTREE_ROOT / "analysis" / "dsp_stage3_manifest.json"
ROOT_MANIFEST_PATH = REPO_ROOT / "analysis" / "dsp_stage3_manifest.json"

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
    CAVE1_VA,
    CAVE2_VA,
    CAVE_VA,
    HOOK1_SITE_VA,
    HOOK2_SITE_VA,
    HOOK_SITE_VA,
    HOOK1_RESUME_VA,
    HOOK2_FALLBACK_VA,
    HOOK2_AI_RESUME_VA,
    DESC_TABLE_VA,
    MEAN_TABLE_VA,
    STD_TABLE_VA,
    STATUS_VA,
    TENSOR_IN_VA,
    TENSOR_OUT_VA,
    GR4_BASE,
    RTOS_PAYLOAD_START,
)

def v2p(vaddr: int) -> int:
    return vaddr - GR4_BASE + RTOS_PAYLOAD_START

def build_stage3_release():
    print("=" * 70)
    print("Ricoh GR IV Hybrid AI Live View Tracking Firmware Rebuilder (v1.2.1)")
    print("Baseline: Official Ricoh GR IV Firmware v1.20")
    print("=" * 70)

    # 1. Verify SD card payload files
    print("\n[Step 1/4] Verifying staged SD card DSP binaries...")
    sd_manifest = extract_dsp_files(out_dir=SD_DIR)
    verify_dsp()
    print(f"  [+] All 3 SD card binaries verified in {SD_DIR}")

    # 2. Decode official v1.20 firmware container
    print(f"\n[Step 2/4] Decoding official firmware container baseline: {OFFICIAL_FW}...")
    assert OFFICIAL_FW.exists(), f"Official firmware missing at {OFFICIAL_FW}"
    raw_fw = OFFICIAL_FW.read_bytes()
    payload, frames, consumed = decode_container(raw_fw)
    verify_container(raw_fw, payload, consumed)
    print(f"  -> Official v1.20 payload decoded: {len(payload):,} bytes")

    # 3. Apply Stage 3 Dual-Hook Deep Learning Tracking Patches
    print("\n[Step 3/4] Splicing Stage 3 Dual Caves, Trampolines & AI Tables...")
    buf = bytearray(payload)
    patch_manifest = apply_stage3_patch(buf, rtos_payload_offset=RTOS_PAYLOAD_START)

    h1 = patch_manifest["dual_hooks"]["hook1_boot_awakening"]
    h2 = patch_manifest["dual_hooks"]["hook2_continuous_ai_af"]
    print(f"  [+] Hook 1 (Boot Wakeup): {h1['cave_length']} B Cave @ VA {h1['cave_vaddr']}, Hook @ VA {h1['site_vaddr']}")
    print(f"  [+] Hook 2 (LiveView AI): {h2['cave_length']} B Cave @ VA {h2['cave_vaddr']}, Hook @ VA {h2['site_vaddr']}")
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

    # Sync to root output directory as well
    ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT_OUTPUT_DIR / "fwdc248b_stage3_ai_track.bin").write_bytes(out_container)
    (ROOT_OUTPUT_DIR / "fwdc248b.bin").write_bytes(out_container)
    root_sd_dir = ROOT_OUTPUT_DIR / "sd_card"
    root_sd_dir.mkdir(parents=True, exist_ok=True)
    for f in SD_DIR.glob("*.BIN"):
        shutil.copy2(f, root_sd_dir / f.name)

    # Full Round-Trip Verification
    print("\n[*] Executing round-trip container verification...")
    again, _, used = decode_container(out_container)
    assert again == patched_payload, "Round-trip payload mismatch!"
    verify_container(out_container, again, used)
    assert word_sum(out_container) == 0, "Outer container word_sum != 0!"
    print("  [+] Round-trip container verification PASSED! Dual word-sums == 0.")

    # Save release manifest
    manifest = {
        "release_type": "Pure K-3 III Deep Learning Exclusive AF with Heartbeat Box (v1.2.3)",
        "version": "1.2.3",
        "baseline_firmware": "Official Ricoh GR IV Firmware v1.20",
        "official_fw_sha256": sha256_hex(raw_fw),
        "firmware": {
            "path": str(STAGE3_FW),
            "size": len(out_container),
            "sha256": sha256_hex(out_container),
        },
        "sd_card_dependencies": sd_manifest["files"],
        "patch_manifest": patch_manifest,
        "runtime_pipeline": {
            "baseline": "Official Ricoh GR IV v1.20 baseline (China Mainland)",
            "stock_rom_bugfix": "Patched v1.20 DspService::LoadDsp (0x5398ef68 & 0x5398efac) to pass `sub r1, fp, #0x40` (DF_ID_S.BIN & DF_IP.BIN) with dual-drive polling ('C': SD card, 'I': Internal 53GB flash) and relaxed drive letter check (0x5398eedc) up to drive 15",
            "mmu_and_dsp_base": "CEVA-XM4 VLIW instructions require native 0x52100000 base. Swapped Pool[3] (DspFw) to 0x52100000 (4 MB) and Pool[8] (ImageAdjustWork) to 0x51200000 (15 MB), patched ChunkLoader (0x5398e178) & LoadDsp (0x5398ee9c) base checks to 0x5210, and patched boot vector parameter r1=0x52100000 into DspService::Start",
            "af_architecture": "Pure K-3 III Deep Learning Exclusive (Stock FotoNation 100% Disabled): Cave 2 directly invokes CEVA-XM4 DSP pipeline. Eliminates all ambiguity between stock FotoNation tracking and CEVA-CV neural inference.",
            "heartbeat_box_diagnostic": "When WaitMessage succeeds but no foreground subject is detected (blank wall), a fixed 32x32 diagnostic heartbeat box (left=240, top=140, score=99) is injected at the lower-right quadrant, providing unmistakable proof of DSP boot, AHIP dequeue, IPC round-trip, and OSD rendering. When a subject is detected, the dynamic AI bounding box overrides the heartbeat box.",
            "hook1_boot_awakening": "LiveViewProcess3aContoller (0x53676ab8) -> Cave 1 (0x53000d60) zeroes STATUS_VA (0x52c00100..0x52c0011f) on singleton creation without blocking early-boot 3A init on SD I/O",
            "hook2_continuous_ai_af": "FaceDetectionService::Task (0x5389639c) -> Cave 2 (0x53000e00) uses constructor-zeroed [r4, #0x13e] to initialize boot retry counter, polls SD card ('C') and internal memory ('I'), uses correct return check (bne cave2_load_ok for 1==SUCCESS), calls StartDsp(r1=0x52100000), then executes pure K-3 III CEVA-XM4 CNN inference",
            "safe_ddr_zones": "All shared buffers relocated to verified unmanaged, fully-mapped DDR (0x52930000..0x53000000): Weights @ 0x52960000..0x52a76220, Tensor In @ 0x52b00000, Tensor Out @ 0x52b09a00, Msg @ 0x52c00000, Status @ 0x52c00100 - zero collision with weights or RTOS Live View Pool[8]",
            "cold_start_lock": "Seeds [r4, #0x100..0x10c] = (rect_cx, rect_cy, box_size, box_size), [r4, #0x118]/[r4, #0x11c] = face_id, [r4, #0x13c] = 1, [r4, #0x121] = 0, [r4, #0x124] = 0 so SelectMainFace and EVENT 0x04070001 lock instantaneously on Frame 1 from any head pose",
            "frame_source": "Completed Ahip ring descriptor *(*(0x550813d0 + 0x434)) (fallback 0x424 / 0x1c / 0x14) dequeued by AhipService::WaitFrame + ARM D-Cache invalidate (0x5391801c) prior to CPU read",
            "downsampler": "Aspect-ratio-aware ([r4, #0xfc] x [r4, #0xfe]) YUV420 -> 3-plane 56x56 (row stride 64) RGB tensor @ 0x52b00000 with C0-continuous 2.0x backlit shadow recovery (Y < 48 -> 2x, 48..144 -> 72 + Y/2) + ARM D-Cache clean/barrier (0x5391808c)",
            "dsp_inference": "msg_id 4 via SendMessage (0x5398e478) & WaitMessage (0x5398e794, timeout 100 ms) with full K-3 III weights @ 0x52960000 (677,824 B) & 0x52a057c0 (461,408 B)",
            "output_heatmap": "Head-priority (fg = max(Ch1, Ch2 - 2)) 3-channel 56x56 (row stride 64) int16_t segmentation map @ 0x52b09a00 + D-Cache invalidate (0x5391801c)",
            "confidence_threshold": "max_diff >= 3 and area_cnt >= 2",
            "face_result_structure": "100-byte zeroed m_DetectedFaceResult [r4, #0x78] in aspect-ratio-scaled Q8 fixed-point with dynamic blob-area box sizing [0x2800..0x5c00]",
            "af_tracking_target": "Native dispatch via UpdateFaceDetectionResult (0x538959f4); directly populates m_DetectedFaceResult and updates FdLibraryService state",
            "lcd_osd_rendering": "UpdateFaceDetectionResult (0x538959f4) renders green AI Face/Eye box on LCD and fires EVENT 0x04070001 to AF",
            "af_system": "GR4 165-point on-sensor PDAF (Zaf) locks onto detected subject centroid"
        }
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] Saved release manifest to: {MANIFEST_PATH}")
    ROOT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROOT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Disassembly verification on final patched binary
    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    print(f"\n=== Disassembly of Rebuilt Hook 1 ({hex(HOOK1_SITE_VA)}) in Flashable Image ===")
    tramp1_poff = v2p(HOOK1_SITE_VA)
    tramp1_in_fw = again[tramp1_poff : tramp1_poff + 8]
    for ins in md.disasm(tramp1_in_fw, HOOK1_SITE_VA):
        print(f"    0x{ins.address:08x}: {ins.mnemonic:8s} {ins.op_str}")

    print(f"\n=== Disassembly of Rebuilt Hook 2 ({hex(HOOK2_SITE_VA)}) in Flashable Image ===")
    tramp2_poff = v2p(HOOK2_SITE_VA)
    tramp2_in_fw = again[tramp2_poff : tramp2_poff + 8]
    for ins in md.disasm(tramp2_in_fw, HOOK2_SITE_VA):
        print(f"    0x{ins.address:08x}: {ins.mnemonic:8s} {ins.op_str}")

    print(f"\n=== Cave 1 ({hex(CAVE1_VA)}) Snippet in Flashable Image ===")
    c1_poff = v2p(CAVE1_VA)
    c1_in_fw = again[c1_poff : c1_poff + 32]
    for ins in md.disasm(c1_in_fw, CAVE1_VA):
        print(f"    0x{ins.address:08x}: {ins.mnemonic:8s} {ins.op_str}")

    print(f"\n=== Cave 2 ({hex(CAVE2_VA)}) Snippet in Flashable Image ===")
    c2_poff = v2p(CAVE2_VA)
    c2_in_fw = again[c2_poff : c2_poff + 32]
    for ins in md.disasm(c2_in_fw, CAVE2_VA):
        print(f"    0x{ins.address:08x}: {ins.mnemonic:8s} {ins.op_str}")

    print("\n" + "=" * 70)
    print("SUCCESS: Stage 3 Pure K-3 III Deep Learning Exclusive AF (v1.2.3) Built & Verified!")
    print("=" * 70)

if __name__ == "__main__":
    build_stage3_release()
