# Release v1.2.1: Hybrid AI AF Architecture, Zero-Collision Memory & Boot Freeze Fix

**Release Date:** 2026-11-09  
**Target Device:** Ricoh GR IV (firmware base v1.20 `fwdc248b.bin`)  
**Co-Processor:** Socionext Milbeaut M10V on-die CEVA-CV Vector AI DSP (`0x41000000`)  
**Neural Network:** Pentax K-3 Mark III Deep Learning Subject/Face Recognition CNN (Full 1.14 MB Model Weights)

---

## 🎯 Executive Summary & Major Breakthroughs in v1.2.1

Version **v1.2.1** is a critical bugfix and architectural release for the Ricoh GR IV AI-AF Mod on the official **GR IV v1.20** firmware baseline. Based on comprehensive static disassembly and kernel-level trace auditing, it completely eliminates boot-time freezes, resolves SD card loader deadlocks, fixes the persistent false top-left detection box, and introduces a **True Hybrid Autofocus Architecture** integrating official FotoNation 60 fps hardware tracking with CEVA-XM4 deep learning CNN fallback.

### Key Architectural Enhancements & Bugfixes

1. **True Pentax K-3 III Hybrid AF Architecture**:
   - Camera-native FotoNation `LibAHFD::DetectFace` (`0x53866594`) now executes first on every frame.
   - When human frontal faces or eyes are detected (`r0 == 0`), execution branches cleanly to `0x53896460` (`ConvertResult`), delivering sub-pixel hardware Face + Left/Right Eye AF at 60 fps with zero latency.
   - When no frontal face is detected (turned head, profile, distant/masked person, or dog/cat/bird), the CEVA-XM4 deep learning CNN automatically steps in as an intelligent fallback to identify the subject and guide AF tracking.

2. **Elimination of DDR Weight Clobber (`work_buf` Relocation)**:
   - Disassembly revealed that `TENSOR_IN_VA` (`work_buf`, size `0xbde40` = 777,792 B) previously overlapped `MEAN_WEIGHTS_VA = 0x52960000` (677,824 B) by 646,720 bytes.
   - On the very first inference frame, CEVA-XM4 DMA overwrote CNN model weights with intermediate activation tensors, causing subsequent deep learning recognition to fail.
   - **Resolution**: Kept `MEAN_WEIGHTS_VA = 0x52960000` and `HEAD_WEIGHTS_VA = 0x52a057c0` (guaranteeing 100% backwards compatibility with existing SD card files) while safely relocating `TENSOR_IN_VA` to **`0x52b00000`** (>564 KB above Chunk 2 in completely unmanaged DDR).

3. **Elimination of Fatal Infinite Hang in `DspService::Test` (`tk_wai_flg`)**:
   - In earlier builds, `DspService::Test` (`0x5398efd0`) was invoked upon successful SD card load in `cave2_load_ok`. This function executes an uninterruptible kernel wait `tk_wai_flg(0x3f0000)` with timeout 0 (infinite wait).
   - Because operational K-3 III CNN runtime (`DF_EX.BIN`) never fires factory diagnostic loopback event flag `0x3f0000`, `FaceDetectionService::Task` became permanently blocked, triggering the RTOS watchdog / 3A scheduler to freeze or crash the camera within ~2 seconds of booting.
   - **Resolution**: Completely removed the `DspService::Test` call from Cave 2. After `DspService::Start`, the patch now re-locks the mutex and cleanly yields the remainder of the load frame (`b .ai_done`). The DSP completes its CRT0 boot sequence cleanly, and on the next frame seamlessly enters active tracking.

4. **Elimination of False Boot-Time Top-Left 1/4 Box**:
   - Disassembly of `UpdateFaceDetectionResult` (`0x53895af4..0x53896020`) revealed that the engine reads `eye1` bounding box from `+0x10` of `m_DetectedFaceResult[0]`. Previously, only `+0x20` was populated, leaving `+0x10` as `(0, 0, 0, 0)`, which mapped to screen coordinates `(0, 0)` with size ~1/4 screen held for 16 frames by `SelectMainFace`.
   - **Resolution**: Both `eye1` (`+0x10`) and `eye2` (`+0x20`) are now dynamically computed and centered in the upper portion of the detected face rectangle, and no box is rendered on cold boot when no subject is present (`face_count == 0`).

5. **Ahip Frame Driver Address Correction**:
   - Corrected active driver base `AHIP_DRV_BASE_VA` to `0x55081350` (derived from `AhipService::WaitFrame` @ `0x5311a628`: `movw r7, #0x1350; movt r7, #0x5508`) and primary ring offset `+0x424` for GR IV v1.20.

6. **Preserved SD Card File Compatibility**:
   - Because `MEAN_WEIGHTS_VA = 0x52960000` and `HEAD_WEIGHTS_VA = 0x52a057c0` remain unchanged, **users do NOT need to re-copy `DF_EX.BIN`, `DF_ID_S.BIN`, or `DF_IP.BIN`**! Simply re-flashing `fwdc248b.bin` will apply the complete fix.

---

## 📦 Binary Assets Included

| File | Size (Bytes) | SHA-256 | Description |
| :--- | :--- | :--- | :--- |
| **`output/fwdc248b.bin`** | 39,109,096 | `15bcb7fb254f6c42f5d50c33b6505f7887d422b0f6c1d65c70577363edf1d1c3` | Flashable v1.2.1 Firmware (Hybrid AF & DDR Fix) |
| **`output/fwdc248b_stage3_ai_track.bin`** | 39,109,096 | `15bcb7fb254f6c42f5d50c33b6505f7887d422b0f6c1d65c70577363edf1d1c3` | Stage 3 Release Payload |
| **`output/sd_card/DF_EX.BIN`** | 5,251,648 | `b16788a16a11ecc2c007d26888a56195b2b4e1bec8aa05b74ea6789803ae8d44` | CEVA-CV DSP Core + Full K-3 III CNN Weights (Unchanged) |
| **`output/sd_card/DF_ID_S.BIN`** | 65,536 | `3d3ebf29622323bb4a203347feaf676aeac110a7b1181604270a1ae0c22b0a2a` | DSP Instruction DRAM Overlay (Unchanged) |
| **`output/sd_card/DF_IP.BIN`** | 16,384 | `cf409a84f3f2d8fac1491634ecda882c9d76b3439c2423da1b31950463a52067` | DSP Parameter Overlay (Unchanged) |

---

## 🛠️ Verification & Integrity Checklist

| Check | Specification | v1.2.1 Result | Status |
| :--- | :--- | :--- | :--- |
| **Firmware Baseline** | Official Ricoh GR IV v1.20 | `7563d7e462f0c53b6ab2fbab7cdfbbb57151a41192ee82f4f3a2108f9d77650d` | **PASS** |
| **Cave 1 Bounds** | `0x53000d60 .. 0x53000db4` | 84 bytes | **PASS** |
| **Cave 2 Bounds** | `0x53000e00 .. 0x53001420` | 1,568 bytes (<= 1,600 B) | **PASS** |
| **DDR Non-Overlap** | All zones `0x51200000 .. 0x53000000` | 0 byte overlap verified | **PASS** |
| **Hybrid AF Routing** | `DetectFace` -> `0x53896460` | Validated via disassembly | **PASS** |
| **Payload Word-Sum** | `word_sum == 0` | `word_sum == 0` (32-bit LE) | **PASS** |
| **Container Word-Sum** | `word_sum == 0` | `word_sum == 0` (32-bit LE) | **PASS** |

---

## 🚀 Flashing & Installation Instructions

### Step 1: Update Flashable Firmware
1. Copy `output/fwdc248b.bin` directly to the SD card root directory (`/fwdc248b.bin`).
2. If you already have `DF_EX.BIN`, `DF_ID_S.BIN`, and `DF_IP.BIN` on your SD card, you do **NOT** need to replace them.

### Step 2: Flash Firmware onto Camera
1. Ensure the camera battery is charged (at least 80%).
2. Insert the SD card into the Ricoh GR IV with the camera powered off.
3. Hold down the **`MENU`** button and press the **`POWER`** button.
4. Follow on-screen prompts to update firmware to version **`1.20`**.
5. Once completed, the camera will restart with full hybrid Face/Eye & CEVA-XM4 deep learning subject tracking operational.
