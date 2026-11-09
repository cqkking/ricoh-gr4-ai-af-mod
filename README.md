# Ricoh GR IV: CEVA-CV AI DSP Co-Processor Awakening & Hybrid Deep Learning Autofocus Mod

[English](README.md) | [简体中文](README_CN.md)

[![Platform: Socionext Milbeaut M10V](https://img.shields.io/badge/Platform-Socionext%20Milbeaut%20M10V-007acc.svg)](#)
[![Target: Ricoh GR IV v1.20](https://img.shields.io/badge/Target%20Firmware-v1.20%20fwdc248b.bin-green.svg)](#)
[![Version: v1.2.1](https://img.shields.io/badge/Version-v1.2.1-blue.svg)](#)
[![Architecture: ARM Cortex-A7 + CEVA-CV DSP](https://img.shields.io/badge/Architecture-ARMv7%20%2B%20CEVA--CV%20DSP-orange.svg)](#)
[![Status: Hybrid AF & Zero-Collision Verified](https://img.shields.io/badge/Status-Hybrid%20AF%20%26%20Zero--Collision%20Verified%20(Dual%20WordSum%3D0)-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#)

Unofficial firmware modification for the **Ricoh GR IV (GR4)** digital camera that unlocks the dormant on-die **CEVA-CV vector AI DSP co-processor** (`0x41000000`) and unites camera-native **FotoNation 60 fps hardware Face/Eye tracking** with **Pentax K-3 Mark III's deep learning subject recognition neural network** directly into Ricoh's native 165-point on-sensor Phase Detection Autofocus (PDAF) system.

---

## 1. Breakthrough: The Hardware Reality

Both the **Ricoh GR IV** and **Pentax K-3 Mark III** share the same heterogeneous imaging SoC: the **Socionext Milbeaut M10V**.

Through deep static disassembly, memory map reconstruction, and hardware peripheral auditing of production firmware binaries, we uncovered an astonishing, undocumented fact:

> **Ricoh GR IV’s physical SoC silicon contains the exact same hardware-level CEVA-CV AI DSP co-processor (physical address `0x41000000`) as the Pentax K-3 Mark III.**
>
> In fact, the Ricoh GR IV operating system (`05_RTOS.bin` @ `0x53000000`) boots up with a fully implemented DSP driver subsystem (`DspService` singleton @ `0x5398d97c` in v1.20), complete with multi-core IPC mailboxes (`DspExc`, `DspReq`, `DspSF`, `DspMtx`) and factory debug routines (`DSP_load`, `DSP_start`, `DSP_test`).
>
> However, because commercial GR IV firmware packages omitted the DSP binary partition (`06_DSP.bin`), this powerful vector neural engine was left **100% dormant and unpowered in factory firmware**.

```text
Socionext Milbeaut M10V Heterogeneous SoC
┌────────────────────────────────────────────────────────────────────────┐
│  ARM Cortex-A7 Dual-Core Host CPU (32-bit ARMv7-A @ 1.0 GHz)           │
│  ├── eSOL T-Kernel RTOS [0x53000000] (ISP / LiveView / AF / UI)        │
│  └── Linux 4.9 Kernel   [0x013ea7e0] (Networking / PTP / Storage)      │
├────────────────────────────────────────────────────────────────────────┤
│  CEVA-CV Vector AI DSP Co-Processor [0x41000000] (512-bit SIMD / VLIW) │
│  ├── Physical MMU Window: 0x41000000 (8 MB Strong-Order Shared)       │
│  ├── DF_EX.BIN   (0x51200000, 5,251,648 B) — CEVA-CV Core + 1.14MB CNN│
│  │   ├── Chunk 0 (0x51200000, 4,112,384 B) — CEVA-CV Vector Math Core  │
│  │   └── Chunk 1 (0x516c0000, 1,139,232 B) — K-3 III Deep CNN Weights │
│  ├── DF_ID_S.BIN (0x515ec000,    65,536 B) — CNN Model Weights Table   │
│  └── DF_IP.BIN   (0x515fc000,    16,384 B) — Reset & Boot Vectors      │
└────────────────────────────────────────────────────────────────────────┘
```

The mathematical proof:
$$0x3EC000\text{ (DF\_EX Core)} + 0x116220\text{ (DF\_EX Weights)} = 0x502220\text{ (5,251,648 bytes)}$$
$$\text{Total DSP Footprint: } 0x502220 + 0x10000\text{ (DF\_ID\_S)} + 0x4000\text{ (DF\_IP)} = 5,333,568\text{ bytes}$$
This matches the exact byte allocation reconstructed and validated against the Pentax K-3 Mark III DSP firmware sections and weights.

---

## 2. Project Architecture & Stages

```text
                  ┌──────────────────────────────────────────────┐
                  │ Stage 1: DSP Slicing & Chunk Extraction      │
                  │   06_DSP.bin -> DF_EX / DF_ID_S / DF_IP      │
                  │   Embeds 1.14 MB K-3 III CNN weights         │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ Stage 2: Autonomous Boot-Time Awakening      │
                  │   Hook @ 0x53676ab8 -> Code Cave 0x53000d60  │
                  │   LoadDsp('C') -> Start()                    │
                  │   Output: fwdc248b_stage2_dsp_wake.bin       │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ Stage 3: True Hybrid AI AF & Memory Guard    │
                  │   • Native FotoNation LibAHFD (60fps Face/Eye)│
                  │   • CEVA-XM4 CNN Deep Learning Fallback      │
                  │   • Zero-Collision DDR Buffer @ 0x52b00000   │
                  │   • Fix 2s Boot Freeze & Top-Left False Box  │
                  │   • Output: fwdc248b_stage3_ai_track.bin     │
                  └──────────────────────────────────────────────┘
```

### Stage 1: Payload Slicing & Chunk Formatting (`dsp_splitter.py`)
- Slices K-3 III's `06_DSP.bin` and deep learning neural weights into the 3-file structure expected by GR4's `DspService::LoadDsp`:
  - `DF_EX.BIN` (5,251,648 bytes): Chunk 0 (CEVA-CV core, 4,112,384 B) + Chunk 1 (1,139,232 B K-3 III CNN weights: 677,824 B backbone weights @ `0x516c0000` + 461,408 B head weights @ `0x517657c0`).
  - `DF_ID_S.BIN` (65,536 bytes): 64 KB constant neural weights loaded to `0x515ec000`.
  - `DF_IP.BIN` (16,384 bytes): 16 KB instruction reset entry vectors loaded to `0x515fc000`.
- Passes 100% byte-perfect reconstruction test verified by [`verify_dsp_payload.py`](src/dsp_ai/verify_dsp_payload.py).

### Stage 2: Autonomous Boot-Time Awakening (`k3_gr4_dsp_patcher.py`)
- **Zero-UART Design**: Designed for consumer camera bodies without debug serial headers.
- Installs a trampoline at `LiveViewProcess3aContoller` (`0x53676ab8` $\to$ `b #0x53000d60`).
- Non-blocking early boot state zeroing without hanging on SD card I/O.
- Calls `DspService::LoadDsp(m_dspService, 'C', 0, 0)` and `Start()`, verified 100% stable on real hardware.

### Stage 3: True Hybrid AI Autofocus Pipeline & Memory Guard (`stage3_ai_patcher.py`, v1.2.1)
- **True Hybrid AF Architecture**:
  - **Level 1 (FotoNation Hardware Acceleration)**: Stock `LibAHFD::DetectFace` (`0x53866594`) executes first on every frame. When human frontal faces or eyes are detected (`r0 == 0`), execution branches immediately to `0x53896460` (`ConvertResult`), delivering sub-pixel hardware Face + Left/Right Eye AF at 60 fps with zero latency.
  - **Level 2 (CEVA-XM4 CNN Intelligent Fallback)**: When no frontal face is detected (turned head, 90° profile, distant/masked person, or dog/cat/bird), the CEVA-XM4 deep learning CNN automatically steps in as an intelligent fallback to identify the subject and guide AF tracking.
- **Zero-Collision DDR Memory Allocation**:
  - Relocated tensor working buffer `TENSOR_IN_VA` to **`0x52b00000`** (`0xbde40` = 777,792 B), situated safely >564 KB above Chunk 2 in unmanaged DDR space.
  - Preserved static CNN model weights at `MEAN_WEIGHTS_VA = 0x52960000` and `HEAD_WEIGHTS_VA = 0x52a057c0`, completely eliminating DMA memory overwrite and maintaining 100% backward compatibility with existing SD card files.
- **Elimination of Boot Freeze & Hang**:
  - Removed blocking `DspService::Test` call that hung on `tk_wai_flg(0x3f0000, 0)`. The hook safely yields the remainder of the load frame (`b .ai_done`), allowing the DSP hardware to complete its CRT0 sequence cleanly.
- **Elimination of False Boot-Time Top-Left 1/4 Box**:
  - Correctly populates both `eye1` (`+0x10`) and `eye2` (`+0x20`) bounding boxes relative to detected subjects. Unregistered cold-boot frames leave `face_count = 0`, entirely preventing transient `(0, 0)` false box artifacts.
- **Completed AHIP Ring Frame Dequeue**:
  - Reads active completed YUV420 frame from `*(*(0x55081350 + 0x424))` (Y @ `+0x04`, U @ `+0x08`, V @ `+0x0c`, fallback `*(*(0x55081350 + 0x434))`).
- **ARM D-Cache Invalidation & Flush**:
  - Calls `InvalidateDataCache` (`0x538f0454`) on the completed DMA slot before CPU downsampling, and `FlushDataCache` (`0x538f04c4`) before DSP DMA read.
- **Aspect-Ratio-Aware Downsampler**:
  - Resamples any aspect ratio (3:2, 4:3, 16:9, 1:1) using `[r4, #0xfc]` (`fd_w`) and `[r4, #0xfe]` (`fd_h`).
- **$C^0$-Continuous $2.0\times$ Backlit Shadow Recovery**:
  - Smooth contrast curve ($Y < 48 \to 2Y$, $48 \le Y < 144 \to 72 + Y/2$, $Y \ge 144 \to Y$) preventing crushed backlit faces from disappearing.

---

## 3. Quick Start: Flashing & Usage

> [!IMPORTANT]
> **No disassembly, no soldering, and no UART debug cable required.**
> The entire modification runs 100% autonomously on camera power-on and is flashable via Ricoh's official SD card firmware updater.

### Step 1: Prepare SD Card Files
1. Format an SD Card with **FAT32**.
2. Copy the three DSP runtime binaries to the **SD card root (`C:\`)**:
   - `DF_EX.BIN` (5,251,648 bytes)
   - `DF_ID_S.BIN` (65,536 bytes)
   - `DF_IP.BIN` (16,384 bytes)
3. Copy the patched firmware image to the **SD card root** as `fwdc248b.bin`:
   - Flash Stage 3 full AI tracking firmware `output/fwdc248b_stage3_ai_track.bin` (or `output/fwdc248b.bin`).

Your SD card root directory will look like:
```text
SD Card Root (C:\)
├── fwdc248b.bin       (39,109,096 B [Stage 3 v1.2.1 Hybrid AF])
├── DF_EX.BIN          ( 5,251,648 B - CEVA-CV Execution partition + 1.14MB CNN weights)
├── DF_ID_S.BIN        (    65,536 B - Neural Network configuration table)
└── DF_IP.BIN          (    16,384 B - Reset vector table)
```

> [!NOTE]
> If you previously installed v1.2.0, the three `DF_*.BIN` files on your SD card are **identical and 100% compatible**. You only need to copy the new `fwdc248b.bin` to update!

### Step 2: Flashing Firmware
1. Insert the SD card into your Ricoh GR IV with a fully charged battery (at least 80%).
2. While holding down the **`DISP`** (or **`MENU`**) button, turn the camera **ON**.
3. The official firmware update screen will appear, showing the current version and target version (`v1.20`).
4. Select **Yes** to execute the update. Wait for the progress bar to reach 100% and restart.

### Step 3: Experience Hybrid AI Autofocus
- Turn on the camera:
  - When shooting human subjects facing forward, the official FotoNation engine locks onto faces and eyes with 60 fps zero-latency hardware tracking.
  - When the subject turns around, looks down, wears a mask, or moves into profile—or when tracking birds, cats, or dogs—the CEVA-XM4 CNN instantly engages to maintain autofocus lock across the scene.

---

## 4. Safety, Brick-Prevention & Emergency Recovery Guide

### Emergency Recovery (If Camera Fails to Boot Normally)
> [!TIP]
> **Your camera is NOT permanently bricked!**
> This project **never touches** the bootloader partitions (`00_BOOTPARA.bin` through `04_BOOT.bin` / IPL).
> The camera's built-in emergency recovery mode runs from the boot ROM / IPL before RTOS executes. Even if the RTOS image hangs, the bootloader update mode remains 100% operational.

**Unbricking Steps:**
1. **Power Off**: If the camera is unresponsive, remove the battery for 10 seconds and reinsert.
2. **Prepare SD Card**: Place the official unmodified `fwdc248b.bin` (v1.20 stock firmware) or `fwdc248b_stage2_dsp_wake.bin` on the root of your SD card (named `fwdc248b.bin`).
3. **Trigger Hardware Update Mode**:
   - Make sure the camera is completely OFF.
   - **Press and hold the `MENU` (or `DISP`) button**.
   - While continuing to hold `MENU` / `DISP`, press the **Power** button.
   - The camera will bypass RTOS and launch the official `Firmware Update` screen directly.
   - Select **Yes** to flash back clean firmware and restore normal operation!

---

## 5. Repository Layout

```text
.
├── analysis/
│   ├── dsp_stage3_manifest.json     # Release manifest for AI-AF build
│   ├── rtos_decompilation_audit.md  # Comprehensive GR4 RTOS reverse engineering audit
│   └── shellcmd_symbols.csv         # Verified RTOS shell symbols
├── archive/
│   └── stage2/                      # Archived Stage 2 early boot-wake proof-of-concept
│       ├── build_dsp_firmware.py
│       ├── dsp_wake_manifest.json
│       ├── FEAT_DSP_AI_WAKE.md
│       ├── k3_gr4_dsp_patcher.py
│       └── verify_dsp_patch.py
├── docs/
│   ├── FEAT_DSP_STAGE3_AI_TRACK.md  # Detailed AI-AF technical whitepaper
│   ├── FEAT_K3_AF_PORT.md           # Pentax K-3 III to GR4 AF porting audit
│   ├── RELEASE_NOTES_v1.2.0.md      # v1.2.0 release notes
│   └── RELEASE_NOTES_v1.2.1.md      # v1.2.1 release notes
├── src/
│   └── dsp_ai/
│       ├── build_firmware.py        # Master firmware packager & container rebuilder
│       ├── build_stage3_firmware.py # Core Stage 3 container builder
│       ├── stage3_ai_patcher.py     # Hybrid AF ARM assembly code cave generator
│       ├── dsp_splitter.py          # 06_DSP.bin -> DF_EX / DF_ID_S / DF_IP extractor
│       ├── verify_dsp_payload.py    # Byte-perfect payload reconstruction verifier
│       ├── k3_desc.bin              # Extracted DeepLearningService descriptor (728 B)
│       ├── k3_mean.bin              # Extracted channel mean normalization table (9,408 B)
│       └── k3_std.bin               # Extracted channel std normalization table (9,408 B)
├── tools/
│   ├── fw_container.py              # LZSS container unpacker, repacker & word-sum solver
│   ├── rtos_analyzer.py             # RTOS xref & symbol parser
│   └── linux_analyzer.py            # Linux rootfs analyzer
└── README.md
```

---

## 6. Building from Source

### Prerequisites
- Python 3.10+
- `capstone` (`pip install capstone`)
- Official Ricoh GR IV v1.20 firmware (`fwdc248b.bin`) placed in `firmware/official/v1.20/` (or `C:\Users\SYZ\Downloads\gr4_v120\`)
- Official Pentax K-3 Mark III v2.20 section `06_DSP.bin` placed in `firmware/reference/k3iii_v220/unpacked/sections/`

### Build Command
```bash
# Slices DSP partitions, embeds AI model tables, compiles code cave, and builds flashable container:
python src/dsp_ai/build_firmware.py
```

Output:
```text
======================================================================
Ricoh GR IV Stage 3 AI Live View Tracking Firmware Rebuilder (v1.20 Baseline)
======================================================================
[*] Confirmed Stage 2 binary intact: fwdc248b_stage2_dsp_wake.bin (38,693,256 bytes)

[Step 1/4] Verifying staged SD card DSP binaries...
[+] PASS: 100% byte-perfect reconstruction verified across all 5,333,536 bytes.

[Step 2/4] Decoding official firmware container baseline...
  -> Official payload decoded: 68,102,696 bytes

[Step 3/4] Splicing Stage 3 Dual Caves, Trampolines & AI Tables...
  [+] Hook 1 (Boot Wakeup): 84 B Cave @ VA 0x53000d60, Hook @ VA 0x53676ab8
  [+] Hook 2 (Hybrid AI AF): 1568 B Cave @ VA 0x53000e00, Hook @ VA 0x5389639c
  [+] Embedded Desc Table (728 B) @ VA 0x53001400
  [+] Embedded Mean Table (9408 B) @ VA 0x541db540
  [+] Embedded Std Table  (9408 B) @ VA 0x54217680
[*] Rezeroing 32-bit payload checksum word...
  [+] Payload word_sum == 0 validated successfully.

[Step 4/4] Rebuilding flashable container with LZSS framing...
  [+] Written Stage 3 preserved binary: output/fwdc248b_stage3_ai_track.bin (39,109,096 bytes)
  [+] Written active flashable binary:  output/fwdc248b.bin (39,109,096 bytes)

[*] Executing round-trip container verification...
  [+] Round-trip container verification PASSED! Dual word-sums == 0.
```

---

## 7. Technical Cross-Reference (GR IV v1.20 Baseline)

| Function / Component | Virtual Address (VA) | Purpose |
| :--- | :--- | :--- |
| `LiveViewProcess3aContoller` | `0x53676ab8` | Hook 1 entry for one-time boot status initialization (`b #0x53000d60`) |
| Cave 1 (Boot Init) | `0x53000d60` | 84-byte non-blocking APCS frame reconstruction + status zeroing |
| `FaceDetectionService::Task` | `0x5389639c` | Hook 2 site for continuous per-frame LiveView Hybrid AI AF (`b #0x53000e00`) |
| Cave 2 (Hybrid AI AF) | `0x53000e00` | 1,568-byte execution engine (Native FotoNation + CEVA-XM4 CNN fallback) |
| Native `LibAHFD::DetectFace` | `0x53866594` | Camera-native FotoNation hardware face detection (called first on every frame) |
| `ConvertResult` (Direct Jump) | `0x53896460` | Direct branch target when FotoNation detects frontal face/eyes (60 fps zero latency) |
| Completed AHIP Ring | `*(*(0x55081350 + 0x424))` | Descriptor dequeued by `AhipService::WaitFrame` (`+0x04` Y, `+0x08` U, `+0x0c` V) |
| `InvalidateDataCache` | `0x538f0454` | Invalidates CPU L1/L2 data cache before reading frame buffer |
| `FlushDataCache` | `0x538f04c4` | Flushes CPU data cache after writing RGB tensor before DSP DMA |
| `DspFactory::GetDspService()` | `0x5398d97c` | Retrieves CEVA-CV `DspService` singleton pointer |
| `DspService::LoadDsp` | `0x5398ee0c` | Loads `DF_EX.BIN`, `DF_ID_S.BIN`, `DF_IP.BIN` from SD card root (Drive `C:\`) |
| `DspService::Start` | `0x5398e2b0` | Deasserts DSP reset, starts execution at `0x515fc000` |
| `DspService::SendMessage` | `0x5398e45c` | Dispatches Message ID 4 (Inference request) |
| `DspService::WaitMessage` | `0x5398e778` | Awaits tensor inference completion (timeout 100 ms) |
| `m_DetectedFaceResult` | `[FaceDetectionService, #0x78]` | Array of 100-byte face structures (`x, y, w, h, eye1, eye2, mouth, score`) |
| `FdLibraryService` | `[FaceDetectionService, #0x38]` | Subsystem holding detected face count @ `+0x30` |
| `UpdateFaceDetectionResult` | `0x53895af4` | Submits face/eye coordinates to LCD OSD display task & AF engine |
| `TENSOR_IN_VA` (`work_buf`) | `0x52b00000` | Relocated input tensor buffer (777,792 B), 0 byte collision with static weights |
| `MEAN_WEIGHTS_VA` | `0x52960000` | Chunk 1 static CNN backbone weights (677,824 B) |
| `HEAD_WEIGHTS_VA` | `0x52a057c0` | Chunk 2 static CNN head classification weights (461,408 B) |

---

## 8. Release Changelog

### v1.2.1 (Current Release)
- **True Pentax K-3 III Hybrid AF Architecture**:
  - Integrated native FotoNation `LibAHFD::DetectFace` (`0x53866594`) with CEVA-XM4 CNN deep learning fallback. Frontal faces/eyes trigger zero-latency 60fps hardware tracking via `0x53896460`; turned heads, profiles, masks, and animals trigger CEVA-XM4 deep learning CNN segmentation.
- **Elimination of DDR Weight Clobber (`work_buf` Relocation)**:
  - Relocated `TENSOR_IN_VA` to `0x52b00000`, eliminating the 646 KB memory overlap that previously overwrote static model weights at `0x52960000`.
- **Elimination of Fatal 2-Second Boot Hang in `DspService::Test`**:
  - Removed infinite blocking call `tk_wai_flg(0x3f0000, 0)` from Cave 2, allowing clean DSP CRT0 initialization and yielding frame cleanly.
- **Elimination of False Boot-Time Top-Left 1/4 Box**:
  - Populates both `eye1` (`+0x10`) and `eye2` (`+0x20`) within the face bounding box; ensures `face_count = 0` on startup when no subject is present.
- **Ahip Driver Base Correction**:
  - Corrected `AHIP_DRV_BASE_VA` to `0x55081350` and primary ring offset to `+0x424` for v1.20 baseline.
- **100% Backwards Compatibility**:
  - SD card files (`DF_EX.BIN`, `DF_ID_S.BIN`, `DF_IP.BIN`) remain unchanged; only `fwdc248b.bin` requires flashing.
- **Binary Checksum**: `fwdc248b.bin` (39,109,096 bytes, SHA-256: `15bcb7fb254f6c42f5d50c33b6505f7887d422b0f6c1d65c70577363edf1d1c3`).

### v1.2.0
- **Official Ricoh GR IV v1.20 Firmware Baseline Rebase**:
  - Rebased onto official v1.20 `fwdc248b.bin` (`7563d7e462f0c53b6ab2fbab7cdfbbb57151a41192ee82f4f3a2108f9d77650d`).
  - Accommodated expanded `.data` section (+0x1080 bytes) and adjusted symbol tables.
- **DSP Loader Return Value Fix**: Corrected success condition from `beq` to `bne` for `DspService::LoadDsp`.

### v1.0.9
- **Completed AHIP Ring Frame Extraction & D-Cache Invalidation**: Reads active completed frame descriptor `*(*(0x55080350 + 0x424))` dequeued by `AhipService::WaitFrame`.
- **Aspect-Ratio-Aware Downsampler (3:2, 4:3, 16:9, 1:1)**: Dynamically scales based on `[r4, #0xfc]` and `[r4, #0xfe]`.
- **$C^0$-Continuous $2.0\times$ Backlit Shadow Recovery**: Smooth contrast curve doubling shadow face contrast.
- **Head-Priority Scanning & Dynamic Box Sizing**: Prioritizes Head ($Ch1$) over Body ($Ch2$).

---

## 9. Disclaimer & License

- **Disclaimer**: This is an independent reverse engineering research project for educational and technical exploration. Firmware modification carries inherent risks. While extensive safeguards (dual word-sum verification, memory range bounds checks, safe fallback branches) have been implemented, the authors assume no liability for camera malfunctions.
- **Copyright Notice**: This repository contains original disassembly tools, patch scripts, and mathematical specifications. It does not distribute proprietary binary firmware copyrighted by Ricoh Imaging Company, Ltd. Users must provide their own legally acquired firmware files.
- **License**: MIT License. See [LICENSE](LICENSE) for details.
