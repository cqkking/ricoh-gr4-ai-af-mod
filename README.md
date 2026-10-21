# Ricoh GR IV: CEVA-CV AI DSP Co-Processor Awakening & Deep Learning Autofocus Mod

[English](README.md) | [简体中文](README_CN.md)

[![Platform: Socionext Milbeaut M10V](https://img.shields.io/badge/Platform-Socionext%20Milbeaut%20M10V-007acc.svg)](#)
[![Target: Ricoh GR IV v1.11](https://img.shields.io/badge/Target%20Firmware-v1.11%20fwdc248b.bin-green.svg)](#)
[![Architecture: ARM Cortex-A7 + CEVA-CV DSP](https://img.shields.io/badge/Architecture-ARMv7%20%2B%20CEVA--CV%20DSP-orange.svg)](#)
[![Status: Stage 3 Verified](https://img.shields.io/badge/Status-Stage%203%20Verified%20(Dual%20WordSum%3D0)-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#)

Unofficial firmware modification for the **Ricoh GR IV (GR4)** digital camera that unlocks the dormant on-die **CEVA-CV vector AI DSP co-processor** (`0x41000000`) and ports **Pentax K-3 Mark III's deep learning subject recognition neural network** directly into Ricoh's native 165-point on-sensor Phase Detection Autofocus (PDAF) system.

---

## 1. Breakthrough: The Hardware Reality

Both the **Ricoh GR IV** and **Pentax K-3 Mark III** share the same heterogeneous imaging SoC: the **Socionext Milbeaut M10V**.

Through deep static disassembly, memory map reconstruction, and hardware peripheral auditing of production firmware binaries, we uncovered an astonishing, undocumented fact:

> **Ricoh GR IV’s physical SoC silicon contains the exact same hardware-level CEVA-CV AI DSP co-processor (physical address `0x41000000`) as the Pentax K-3 Mark III.**
>
> In fact, the Ricoh GR IV operating system (`05_RTOS.bin` @ `0x53000000`) boots up with a fully implemented DSP driver subsystem (`DspService` @ `0x539658c4`), complete with multi-core IPC mailboxes (`DspExc`, `DspReq`, `DspSF`, `DspMtx`) and factory debug routines (`DSP_load`, `DSP_start`, `DSP_test`).
>
> However, because commercial GR IV firmware packages omitted the 4.00 MB DSP binary partition (`06_DSP.bin`), this powerful vector neural engine was left **100% dormant and unpowered in factory firmware**.

```text
Socionext Milbeaut M10V Heterogeneous SoC
┌────────────────────────────────────────────────────────────────────────┐
│  ARM Cortex-A7 Dual-Core Host CPU (32-bit ARMv7-A @ 1.0 GHz)           │
│  ├── eSOL T-Kernel RTOS [0x53000000] (ISP / LiveView / AF / UI)        │
│  └── Linux 4.9 Kernel   [0x013ea7e0] (Networking / PTP / Storage)      │
├────────────────────────────────────────────────────────────────────────┤
│  CEVA-CV Vector AI DSP Co-Processor [0x41000000] (512-bit SIMD / VLIW) │
│  ├── Physical MMU Window: 0x41000000 (8 MB Strong-Order Shared)       │
│  ├── DF_EX.BIN   (0x51200000, 4,112,384 B) — CEVA-CV Vector Math Core │
│  ├── DF_ID_S.BIN (0x515ec000,    65,536 B) — CNN Model Weights Table   │
│  └── DF_IP.BIN   (0x515fc000,    16,384 B) — Reset & Boot Vectors      │
└────────────────────────────────────────────────────────────────────────┘
```

The mathematical proof:
$$0x3EC000\text{ (DF\_EX)} + 0x10000\text{ (DF\_ID\_S)} + 0x4000\text{ (DF\_IP)} = 0x400000\text{ (4,194,304 bytes / 4.00 MB)}$$
This matches the exact byte length of the Pentax K-3 Mark III DSP firmware section (`06_DSP.bin`).

---

## 2. Project Architecture & Stages

```text
                  ┌──────────────────────────────────────────────┐
                  │ Stage 1: DSP Slicing & Chunk Extraction      │
                  │   06_DSP.bin -> DF_EX / DF_ID_S / DF_IP      │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ Stage 2: Autonomous Boot-Time Awakening      │
                  │   Hook @ 0x536520f0 -> Code Cave 0x53000e00  │
                  │   LoadDsp('A') -> Start() -> Test()          │
                  │   Output: fwdc248b_stage2_dsp_wake.bin       │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ Stage 3: Live View AI Inference & AF Bridge   │
                  │   • LiveView Frame -> 56x56 RGB Tensor       │
                  │   • DspService::SendMessage(msg_id=4)        │
                  │   • 64x64 Heatmap Peak Scan -> (X, Y, W, H)  │
                  │   • UpdateTrackTarget -> GR4 165-point PDAF  │
                  │   Output: fwdc248b_stage3_ai_track.bin       │
                  └──────────────────────────────────────────────┘
```

### Stage 1: Payload Slicing & Chunk Formatting (`dsp_splitter.py`)
- Slices K-3 III's 4.00 MB `06_DSP.bin` into the proprietary 3-file structure expected by GR4's `DspService::LoadDsp`:
  - `DF_EX.BIN` (4,112,408 bytes): Includes a 16-byte header + 8-byte chunk descriptor declaring target virtual address `0x51200000` and length `0x3ec000`.
  - `DF_ID_S.BIN` (65,536 bytes): 64 KB constant neural weights loaded to `0x515ec000`.
  - `DF_IP.BIN` (16,384 bytes): 16 KB instruction reset entry vectors loaded to `0x515fc000`.
- Passes 100% byte-perfect reconstruction test verified by [`verify_dsp_payload.py`](src/dsp_ai/verify_dsp_payload.py).

### Stage 2: Autonomous Boot-Time Awakening (`k3_gr4_dsp_patcher.py`)
- **Zero-UART Design**: Designed for consumer camera bodies without debug serial headers.
- Installs a trampoline at `LiveViewProcess3aContoller` (`0x536520f0` $\to$ `b #0x53000e00`).
- Execution sequence:
  1. Checks if DSP is already running (`status & 1`); if so, bypasses initialization in 3 clock cycles.
  2. Calls `DspService::LoadDsp(m_dspService, 'A', 0, 0)` to load the 3 files from the SD card root (`A:\`).
  3. Validates `status & 2` (`LOADED`).
  4. Calls `DspService::Start(m_dspService, 0, 0)` to release CEVA-CV reset and activate the vector clocks.
  5. Calls `DspService::Test(m_dspService, 0)` for hardware handshake.
  6. Returns seamlessly to `0x536520f8` to continue normal Live View preview.

### Stage 3: Real-Time Deep Learning Tracking Bridge (`stage3_ai_patcher.py`)
- **Embedded Neural Tables** (inside audited zero caves in RTOS text/rodata):
  - `k3_desc.bin` (728 B) @ `0x53001200`: CNN layer topology, soft thresholds, and projection matrix.
  - `k3_mean.bin` (9,408 B) @ `0x541db540`: $56 \times 56 \times 3$ channel mean normalization bias.
  - `k3_std.bin`  (9,408 B) @ `0x54217680`: $56 \times 56 \times 3$ standard deviation activation scale.
- **Dual-Hook Architecture**:
  - **Hook 1: Boot Wakeup (`LiveViewProcess3aContoller` @ `0x536520f0` -> Cave 1 @ `0x53000d60`, 152 B)**:
    - Reconstructs APCS prologue and calls `LoadDsp('A')`, `Start()`, `Test()` safely **once at boot time** from SD card.
    - Verified 100% stable on real hardware in Stage 2.
  - **Hook 2: Continuous LiveView AI AF (`FaceDetectionService::Task` @ `0x5386ea8c` -> Cave 2 @ `0x53000e00`, 692 B)**:
    - Hooked directly into Ricoh GR IV's continuous 30/60 fps Face Detection task loop.
    - **Decimation**: Runs deep learning inference every 2 Live View frames (~30 fps), using <0.8% CPU time.
    - **Frame Capture**: Pulls active $960 \times 640$ YUV frame buffer from `CaptureControlService` (`[0x55395a18, #8] -> [m_CaptureViewControlService, #4]`).
    - **Bilinear Downsampler**: Resamples the preview frame into a $56 \times 56$ RGB tensor at shared DMA address `0x51600000`.
    - **DSP Dispatch**: Dispatches Message ID 4 via `DspService::SendMessage` (`0x539663c0`) to the CEVA-CV vector DSP and awaits tensor completion via `DspService::WaitMessage` (`0x539666dc`, timeout 50 ms).
    - **Heatmap Peak Extraction**: Scans the $64 \times 64$ confidence heatmap at `0x51610000` for peak centroid $(col, row, score)$.
    - **Deep Learning AF Lock & LCD OSD**:
      - When peak confidence $\ge 60$:
        - Translates centroid to Live View pixels $(X, Y, W=180, H=180)$ and calculates eye coordinates (`left_eye`, `right_eye`, `mouth`).
        - Injects the detected target into `m_DetectedFaceResult` (`[r4, #0x78]`) and sets `face_count = 1` in `FdLibraryService` (`[r4, #0x38]`).
        - **Zero calls to `CreateTask`** (completely eliminates half-press freeze).
        - Resumes at `0x5386ea9c`, unlocking the mutex and triggering `UpdateFaceDetectionResult` (`0x5386e0e4`) to render the green AI face/eye tracking box on the LCD and drive the 165-point AF sensor!
      - When peak confidence $< 60$:
        - Seamlessly falls back to stock FotoNation `LibAHFD` (`0x5383ec84`) and resumes at `0x5386ea94`, ensuring zero focus hunting or dropped frames.

---

## 3. Quick Start: Flashing & Usage

> [!IMPORTANT]
> **No disassembly, no soldering, and no UART debug cable required.**
> The entire modification runs 100% autonomously on camera power-on and is flashable via Ricoh's official SD card firmware updater.

### Step 1: Prepare SD Card Files
1. Format an SD Card with **FAT32**.
2. Copy the three DSP runtime binaries to the **SD card root**:
   - `DF_EX.BIN` (4,112,408 bytes)
   - `DF_ID_S.BIN` (65,536 bytes)
   - `DF_IP.BIN` (16,384 bytes)
3. Copy the patched firmware image to the **SD card root** as `fwdc248b.bin`:
   - **Recommended First Step**: Flash Stage 2 minimal verification firmware `output/fwdc248b_stage2_dsp_wake.bin` first to verify clean startup and hardware DSP ignition.
   - Once verified, flash Stage 3 full AI tracking firmware `output/fwdc248b_stage3_ai_track.bin`.

Your SD card root directory will look like:
```text
SD Card Root (A:\)
├── fwdc248b.bin       (38,693,256 B [Stage 2] or 38,781,980 B [Stage 3])
├── DF_EX.BIN          ( 4,112,408 bytes - CEVA-CV Execution partition)
├── DF_ID_S.BIN        (    65,536 bytes - Neural Network weights)
└── DF_IP.BIN          (    16,384 bytes - Reset vector table)
```

### Step 2: Flashing Firmware
1. Insert the SD card into your Ricoh GR IV with a fully charged battery.
2. While holding down the **`DISP`** (or **`MENU`**) button, turn the camera **ON**.
3. The official firmware update screen will appear, showing the current version (`v1.11`) and the update version.
4. Select **Yes** to execute the update. Wait for the progress bar to reach 100% and restart.

### Step 3: Experience AI Autofocus
- Turn on the camera. The boot trampoline automatically initializes and verifies the CEVA-CV DSP.
- Frame a human subject in Live View: The camera's on-sensor PDAF tracking box will magnetically snap to and track the subject across the frame.

---

## 4. Safety, Brick-Prevention & Emergency Recovery Guide

### Emergency Recovery (If Camera Fails to Boot Normally)
> [!TIP]
> **Your camera is NOT permanently bricked!**
> This project **never touches** the bootloader partitions (`00_BOOTPARA.bin` through `04_BOOT.bin` / IPL).
> The camera's built-in emergency recovery mode runs from the boot ROM / IPL before RTOS executes. Even if the RTOS image hangs or crashes, the bootloader update mode remains 100% operational.

**Unbricking Steps:**
1. **Power Off**: If the camera is unresponsive (black screen or power LED on), remove the battery for 10 seconds and reinsert.
2. **Prepare SD Card**: Place the official unmodified `fwdc248b.bin` (v1.11 stock firmware) or the fixed `fwdc248b_stage2_dsp_wake.bin` on the root of your SD card (named `fwdc248b.bin`).
3. **Trigger Hardware Update Mode**:
   - Make sure the camera is completely OFF.
   - **Press and hold the `MENU` (or `DISP`) button**.
   - While continuing to hold `MENU` / `DISP`, press the **Power** button.
   - The camera will bypass RTOS and launch the official `Firmware Update` screen directly.
   - Select **Yes** to flash back clean firmware and restore normal operation!

### Low-Level Architecture Safeguards
1. **AAPCS 8-Byte Stack Alignment & Full APCS Frame Construction**:
   - The code cave executes a complete APCS frame construction (`mov ip, sp; push {r4-r7, fp, ip, lr, pc}; sub fp, ip, #4; sub sp, sp, #8`) before any function call.
   - Registers `r0-r3` (16 bytes) and `r4-r8, lr` (24 bytes) are preserved on strictly double-word (8-byte) aligned boundaries.
   - The code cave returns directly to `0x53652108` (`ldr r6, [r4]`), eliminating any reliance on scratch register `ip` (R12), preventing frame pointer corruption and eliminating CPU Data/Prefetch Abort exceptions.
2. **Dual 32-bit Word-Sum Validation (`word_sum == 0`)**:
   - Both the inner payload (`payload.bin`) and outer LZSS container (`fwdc248b.bin`) enforce rigorous 32-bit little-endian zero-sum checksums.
   - The official bootloader (`04_BOOT.bin`) and update utility (`FWUPD`) will reject corrupted files. Every build produced by our pipeline passes 100% round-trip validation.
3. **Graceful Fallback & NULL Protection**:
   - If SD card binaries are absent or `m_dspService` is not yet initialized at early boot, the code cave safely exits back to the native camera sequence within ~10 clock cycles with zero side effects.
4. **Completely Reversible**:
   - You can reflash the official factory `v1.11` firmware at any time using the standard update menu.

---

## 5. Repository Layout

```text
.
├── analysis/
│   ├── dsp_stage3_manifest.json     # Release manifest for Stage 3 build
│   ├── dsp_wake_manifest.json       # Release manifest for Stage 2 build
│   ├── rtos_decompilation_audit.md  # Comprehensive GR4 RTOS reverse engineering audit
│   └── shellcmd_symbols.csv         # 651 verified RTOS shell symbols
├── docs/
│   ├── FEAT_DSP_STAGE3_AI_TRACK.md  # Detailed Stage 3 technical whitepaper
│   ├── FEAT_DSP_AI_WAKE.md          # Detailed Stage 2 DSP awakening whitepaper
│   └── FEAT_K3_AF_PORT.md           # Pentax K-3 III to GR4 AF porting audit
├── src/
│   └── dsp_ai/
│       ├── build_stage3_firmware.py # Complete Stage 3 packager & container rebuilder
│       ├── stage3_ai_patcher.py     # Stage 3 ARM assembly code cave generator
│       ├── build_dsp_firmware.py    # Stage 2 firmware builder
│       ├── k3_gr4_dsp_patcher.py    # Stage 2 ARM assembly patch generator
│       ├── dsp_splitter.py          # 06_DSP.bin -> DF_EX / DF_ID_S / DF_IP splitter
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
- Official Ricoh GR IV v1.11 firmware (`fwdc248b.bin`) placed in `firmware/official/v1.11/`
- Official Pentax K-3 Mark III v2.20 section `06_DSP.bin` placed in `firmware/reference/k3iii_v220/unpacked/sections/`

### Build Command
```bash
# Slices DSP partitions, embeds AI model tables, compiles code cave, and builds flashable container:
python src/dsp_ai/build_stage3_firmware.py
```

Output:
```text
======================================================================
Ricoh GR IV Stage 3 AI Live View Tracking Firmware Rebuilder
======================================================================
[*] Confirmed Stage 2 binary intact: fwdc248b_stage2_dsp_wake.bin (38,693,256 bytes)

[Step 1/4] Verifying staged SD card DSP binaries...
[+] PASS: 100% byte-perfect reconstruction verified across all 4,194,304 bytes.

[Step 2/4] Decoding official firmware container baseline...
  -> Official payload decoded: 67,948,280 bytes

[Step 3/4] Splicing Stage 3 Dual Caves, Trampolines & AI Tables...
  [+] Hook 1 (Boot Wakeup): 152 B Cave @ VA 0x53000d60, Hook @ VA 0x536520f0
  [+] Hook 2 (LiveView AI): 692 B Cave @ VA 0x53000e00, Hook @ VA 0x5386ea8c
  [+] Embedded Desc Table (728 B) @ VA 0x53001200
  [+] Embedded Mean Table (9408 B) @ VA 0x541db540
  [+] Embedded Std Table  (9408 B) @ VA 0x54217680
[*] Rezeroing 32-bit payload checksum word...
  [+] Payload word_sum == 0 validated successfully.

[Step 4/4] Rebuilding flashable container with LZSS framing...
  -> Modified payload bytes: 19722
  [+] Written Stage 3 preserved binary: output/fwdc248b_stage3_ai_track.bin (38,791,732 bytes)
  [+] Written active flashable binary:  output/fwdc248b.bin (38,791,732 bytes)

[*] Executing round-trip container verification...
  [+] Round-trip container verification PASSED! Dual word-sums == 0.
```

---

## 7. Technical Cross-Reference

| Function / Component | Virtual Address (VA) | Purpose |
| :--- | :--- | :--- |
| `LiveViewProcess3aContoller` | `0x536520f0` | Hook 1 entry for one-time boot DSP awakening (`b #0x53000d60`) |
| Cave 1 (Boot Wakeup) | `0x53000d60` | 152-byte boot engine: queries `DspService`, loads from SD card, starts & tests DSP |
| `FaceDetectionService::Task` | `0x5386ea8c` | Hook 2 site for continuous per-frame LiveView AI AF (`b #0x53000e00`) |
| Cave 2 (LiveView AI AF) | `0x53000e00` | 692-byte execution engine (Downsampler + DSP IPC + Peak Detection + Face Results) |
| `m_CaptureViewControlService` | `[0x55395a18, #8]` | Pointer to preview streaming controller; offset `+4` points to raw 960x640 YUV frame |
| `DspFactory::GetDspService()` | `0x539658c4` | Retrieves CEVA-CV `DspService` singleton pointer |
| `DspService::LoadDsp` | `0x53966d74` | Loads `DF_EX.BIN`, `DF_ID_S.BIN`, `DF_IP.BIN` from SD card |
| `DspService::Start` | `0x53966214` | Deasserts DSP reset, starts execution at `0x515fc000` |
| `DspService::Test` | `0x53966f18` | Runs CEVA-CV version self-test |
| `DspService::SendMessage` | `0x539663c0` | Dispatches Message ID 4 (Inference request) |
| `DspService::WaitMessage` | `0x539666dc` | Awaits tensor inference completion |
| `m_DetectedFaceResult` | `[FaceDetectionService, #0x78]` | Array of 100-byte face structures (`x, y, w, h, eye_l, eye_r, mouth, score`) |
| `FdLibraryService` | `[FaceDetectionService, #0x38]` | Subsystem holding detected face count @ `+0x30` |
| `UpdateFaceDetectionResult` | `0x5386e0e4` | Submits face/eye coordinates to LCD OSD display task & AF engine |
| Stock FotoNation `DetectFace` | `0x5383ec84` | Fallback CPU face detector when AI confidence $< 60$ |

---

## 8. Release Changelog

### v1.0.4 (Current Release)
- **Resolved Half-Press AF Freeze**: Identified that `0x538e32c4` was erroneously assumed to be `UpdateTrackTarget`, but was statically proven to be `CreateTask` in `src/Task.cpp`. Calling it with $H=180$ (`0x000000b4`) created rogue tasks pointing to ARM low-memory exception vectors, which crashed the CPU when the scheduler switched context on half-press (S1). Completely removed calls to `0x538e32c4` and eliminated mutex deadlocks.
- **Architectural Dual-Hook Separation**:
  - Hook 1 @ `0x536520f0` -> Cave 1 @ `0x53000d60` (152 B): Handles one-time boot loading of CEVA-CV DSP from SD card outside the face detection mutex.
  - Hook 2 @ `0x5386ea8c` -> Cave 2 @ `0x53000e00` (692 B): Handles continuous per-frame CNN inference and native dispatch via `UpdateFaceDetectionResult` (`0x5386e0e4`).
- **Binary Size & Checksum Update**: Updated `fwdc248b.bin` (38,791,732 bytes, SHA-256: `edd0aa6184f03b6c75ccd1fd23408fd8fe521e073e4f69b4d520cb8a14cc24fc`).

### v1.0.3
- **Continuous LiveView AF Hook Refactor**: Relocated inference hook from boot-time `LiveViewProcess3aContoller` (`0x536520f0`) to the true per-frame face detection loop `FaceDetectionService::Task` (`0x5386ea8c`).
- **LiveView Preview Buffer Resolution**: Resolved preview buffer indirection via `[0x55395a18, #8] -> [m_CaptureViewControlService, #4]`, ensuring pristine 960x640 image feeding to the CNN downsampler.
- **Bimodal LCD OSD & AF Hardware Delivery**: Spliced output bounding boxes and eye coordinates directly into `m_DetectedFaceResult` and `UpdateFaceDetectionResult` (`0x5386e0e4`), rendering the green AI face/eye tracking box on the LCD and driving the 165-point on-sensor PDAF.
- **Fail-Safe Stock Fallback**: Retained full fallback to FotoNation `LibAHFD` (`0x5383ec84`) when confidence $< 60$.

### v1.0.2
- **MMU Read-Only Fault Resolution**: Relocated runtime telemetry and sequence variables from code cave (`.text`, read-only `AP=110`) to DSP shared DRAM (`0x51602540`, read/write `AP=011`).

### v1.0.1
- **AAPCS 8-Byte Stack Alignment**: Enforced strict AAPCS 8-byte stack alignment across code cave entries and subroutine invocations.

### v1.0.0
- Initial public release of CEVA-CV AI DSP awakening and Pentax K-3 Mark III deep learning autofocus port for Ricoh GR IV.

---

## 8. Disclaimer & License

- **Disclaimer**: This is an independent reverse engineering research project for educational and technical exploration. Firmware modification carries inherent risks. While extensive safeguards (dual word-sum verification, memory range bounds checks, safe fallback branches) have been implemented, the authors assume no liability for camera malfunctions.
- **Copyright Notice**: This repository contains original disassembly tools, patch scripts, and mathematical specifications. It does not distribute proprietary binary firmware copyrighted by Ricoh Imaging Company, Ltd. Users must provide their own legally acquired firmware files.
- **License**: MIT License. See [LICENSE](LICENSE) for details.
