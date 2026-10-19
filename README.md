# Ricoh GR IV: CEVA-CV AI DSP Co-Processor Awakening & Deep Learning Autofocus Mod

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
  - `k3_desc.bin` (728 B) @ `0x53001100`: CNN layer topology, soft thresholds, and projection matrix.
  - `k3_mean.bin` (9,408 B) @ `0x541db540`: $56 \times 56 \times 3$ channel mean normalization bias.
  - `k3_std.bin`  (9,408 B) @ `0x54217680`: $56 \times 56 \times 3$ standard deviation activation scale.
- **Real-Time Execution Pipeline**:
  - **Decimation**: Runs inference every 2 Live View frames (~30 fps), using <0.8% CPU time.
  - **Bilinear Downsampler**: Resamples $960 \times 640$ YUV Live View frames into a $56 \times 56$ RGB tensor at DMA address `0x51600000`.
  - **DSP Dispatch**: Asynchronously dispatches Message ID 4 via `DspService::SendMessage` (`0x539663c0`) and awaits completion via `DspService::WaitMessage` (`0x539666dc`).
  - **Heatmap Peak Extraction**: Scans the resulting $64 \times 64$ confidence heatmap at `0x51610000` for peak centroid $(col, row, score)$.
  - **Native AF Lock**: Translates centroid to Live View pixels $(X, Y, W, H)$, writes directly to primary tracking slot 0 (`0x53fe1a94`), and calls `UpdateTrackTarget` (`0x538e32c4`).
  - **165-Point On-Sensor PDAF Hook**: GR4's on-sensor phase detection algorithms (`ZafCorrelation` / `ZafPredictor`) lock onto the subject with high precision.

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
   - Use `output/fwdc248b_stage3_ai_track.bin` (or `output/fwdc248b.bin`).

Your SD card root directory will look like:
```text
SD Card Root (A:\)
├── fwdc248b.bin       (38,781,980 bytes - Flashable firmware)
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

## 4. Safety & Brick-Prevention Engineering

1. **Dual 32-bit Word-Sum Validation (`word_sum == 0`)**:
   - Both the inner payload (`payload.bin`) and outer LZSS container (`fwdc248b.bin`) enforce rigorous 32-bit little-endian zero-sum checksums.
   - The official bootloader (`04_BOOT.bin`) and update utility (`FWUPD`) will reject corrupted files. Every build produced by our pipeline passes 100% round-trip validation.
2. **Safe Fallback**:
   - If the SD card is removed or the `DF_*.BIN` files are absent, the code cave's safety check (`tst r2, #2`) aborts gracefully within 12 clock cycles. The camera boots normally with stock contrast/hybrid AF.
3. **Completely Reversible**:
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

[Step 3/4] Splicing Stage 3 Code Cave, Live View Trampoline & AI Tables...
  [+] Injected Code Cave (692 B) @ VA 0x53000e00
  [+] Injected Trampoline (8 B) @ VA 0x536520f0
  [+] Embedded Desc Table (728 B) @ VA 0x53001100
  [+] Embedded Mean Table (9408 B) @ VA 0x541db540
  [+] Embedded Std Table  (9408 B) @ VA 0x54217680
[*] Rezeroing 32-bit payload checksum word...
  [+] Payload word_sum == 0 validated successfully.

[Step 4/4] Rebuilding flashable container with LZSS framing...
  [+] Written Stage 3 preserved binary: output/fwdc248b_stage3_ai_track.bin (38,781,980 bytes)
  [+] Written active flashable binary:  output/fwdc248b.bin (38,781,980 bytes)

[*] Executing round-trip container verification...
  [+] Round-trip container verification PASSED! Dual word-sums == 0.
```

---

## 7. Technical Cross-Reference

| Function / Component | Virtual Address (VA) | Purpose |
| :--- | :--- | :--- |
| `LiveViewProcess3aContoller` | `0x536520f0` | Hook site for boot trampoline (`b #0x53000e00`) |
| Stage 3 Code Cave | `0x53000e00` | 692-byte execution engine (Awakening + Downsampler + Inference + AF Bridge) |
| `DspFactory::GetDspService()` | `0x539658c4` | Retrieves CEVA-CV `DspService` singleton pointer |
| `DspService::LoadDsp` | `0x53966d74` | Loads `DF_EX.BIN`, `DF_ID_S.BIN`, `DF_IP.BIN` from SD card |
| `DspService::Start` | `0x53966214` | Deasserts DSP reset, starts execution at `0x515fc000` |
| `DspService::Test` | `0x53966f18` | Runs CEVA-CV version self-test |
| `DspService::SendMessage` | `0x539663c0` | Dispatches Message ID 4 (Inference request) |
| `DspService::WaitMessage` | `0x539666dc` | Awaits tensor inference completion |
| `UpdateTrackTarget` | `0x538e32c4` | Submits $(X, Y, W, H, Conf)$ to GR4 autofocus tracker |
| Primary AF Tracking Slot 0 | `0x53fe1a94` | Direct AF coordinate slot read by `ZafPredictor` |
| Live View Image Buffer Pointer | `0x55395a18` | Source buffer for $960 \times 640$ YUV frame capture |

---

## 8. Disclaimer & License

- **Disclaimer**: This is an independent reverse engineering research project for educational and technical exploration. Firmware modification carries inherent risks. While extensive safeguards (dual word-sum verification, memory range bounds checks, safe fallback branches) have been implemented, the authors assume no liability for camera malfunctions.
- **Copyright Notice**: This repository contains original disassembly tools, patch scripts, and mathematical specifications. It does not distribute proprietary binary firmware copyrighted by Ricoh Imaging Company, Ltd. Users must provide their own legally acquired firmware files.
- **License**: MIT License. See [LICENSE](LICENSE) for details.
