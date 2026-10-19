# Pentax K-3 Mark III -> Ricoh GR IV Autofocus (AF) Porting & Architecture Audit

## 1. Platform & Firmware Baseline

| Property | Pentax K-3 Mark III (v2.20) | Ricoh GR IV (v1.11) |
| :--- | :--- | :--- |
| **Firmware Binary** | `fwdc233b.bin` (`Kb582` / `Kb620`) | `fwdc248b.bin` (`Kb636`) |
| **SoC Platform** | Socionext Milbeaut M10V (ARMv7 Cortex-A7 + CEVA-CV DSP) | Socionext Milbeaut M10V (ARMv7 Cortex-A7 + CEVA-CV DSP) |
| **RTOS Load Base (`05_RTOS.bin`)** | `0x52500000` (18,274,800 B) | `0x53000000` (21,291,680 B) |
| **Source Tree Root** | `/mnt/d/dev/M10V/M10VKCommon/m10v-k-common/A7RTOS/` | `/mnt/c/dev/Git/m10v-gr-common/A7RTOS/` |
| **C++ RTTI Classes** | 1,941 classes | 1,548 classes |
| **SHELLCMD Symbols** | 958 commands (`11_SHELLCMD.bin`) | 651 commands (`10_SHELLCMD.bin`) |
| **CEVA-CV DSP Memory (`02_PARTPARA.bin`)** | `0x41000000` (`0x00800000` = 8 MB) + `06_DSP.bin` (4 MB) | `0x41000000` (`0x00800000` = 8 MB, **dormant**) |

---

## 2. Subsystem Feasibility Matrix

### Tier 1 — High-Value & Feasible Ports

#### A. On-Die CEVA-CV AI DSP (`0x41000000`) & Deep Learning Subject Recognition (`DeepLearningService`)
- **Hardware & Memory Map**:
  - Both K-3 III and GR4 `02_PARTPARA.bin` (offset `0x0058..0x0068`) reserve physical address `0x41000000` (size `0x00800000` = 8 MB) for the on-die CEVA-CV DSP.
- **GR4 RTOS Driver Status (Initialized at Boot, Idle at Runtime)**:
  - `DspService` (`src/DspService.cpp`, code @ `0x539658c4..0x53967d00`, `vt=0x5411c520`) is fully compiled and active in GR4.
  - `PFFactory.cpp` (`DeviceBuildPackage` @ `0x5376e18c` & `DeviceInitialize` @ `0x5376e4e4`) calls `DspFactory::GetDspService()->BuildPackage()` (`0x53965950`) and `DspService::Initialize()` (`0x53965bf0`), initializing RTOS tasks/mailboxes (`DspExc`, `DspReq`, `DspSF`, `DspMtx`) and CEVA-CV runtime functions (`ceva_cv_init_sequence`, `ceva_cv_add_kernel`, `ceva_cv_verify_sequence`, `ceva_cv_execute_sequence`).
  - Shell commands present in GR4: `DSP_load` (`0x5305aea4`), `DSP_start` (`0x5305b4d0`), `DSP_test` (`0x5305b68c`).
- **K-3 III Live View Deep Learning Pipeline**:
  - `DeepLearningService` (`src/DeepLearningService.cpp` @ `0x5302c390..0x5302cb38`, `vt=0x53a48a90`, task `DLPipeSvc`, **10,848 bytes**).
  - Used in both `AeSensorService.cpp` (`0x52d8a430`) and **Live View** (`LiveViewProcessAeAfAwb.cpp` @ `0x52d337b4`, `SetupDeepLearningConfig` @ `0x52d3a06c`) + `AeAlgorithm.cpp` (`0x52db331c`).
  - Resizes Live View frames into **56×56×24-bit RGB (`0x38`×`0x38`)** tensors (`0x5302c860`), allocates a `0x1e0000` (1.875 MB) `DLPipe` buffer, executes inference on `06_DSP.bin` via `DspService::SendMessage` (`0x5319b7b4`) / `WaitMessage` (`0x5319bad0`), and populates a **64×64 `m_humanDetectionMap`** (`0x1000` bytes).

#### B. AF-C Moving-Subject Prediction & Filtering (`AfCommonFilter` + `PafPredictor` -> `ZafPredictor`)
- **Code Size & Capability Gap**:
  - `AfCommonFilter`: **3,156 B** in K-3 III (`0x52c73a78`, `vt=0x53887380`) vs **988 B** in GR4 (`0x535e6a80`, `vt=0x53ea6838`).
  - `AfCommonCorrelation`: **2,348 B** in K-3 III (`0x52c7314c`) vs **1,576 B** in GR4 (`0x535e6458`).
  - `AfCommonPredictor`: **2,868 B** in K-3 III (`0x52c746cc`) vs **2,468 B** in GR4 (`0x535e6e5c`).
  - `ZafPredictor` (GR4 `0x536192a4..0x53619608`, `vt=0x53eae5c8`): Only **868 B** (simple linear least-squares via `AfCalculation::LeastSquare` @ `0x535e81f4`), compared to K-3 III's `PafPredictor` (**7,276 B** @ `0x52cdd5d8..0x52cdf244`, `vt=0x5389bf08`) + `PafMovableObject` (**9,088 B** @ `0x52cbe480`) + `PafTrackingAlgorithm` (**21,436 B** @ `0x52cdf244`).
- **Porting Mechanism**:
  - Both `Paf*` and `Zaf*` normalize phase shift into `[Defocus (mm/256)]` and convert to lens drive pulses via $K$-value (`[pulse/mm]`), inheriting from `AfCommonPredictor`.
  - K-3 III's full `AfCommonFilter` and higher-order velocity/acceleration predictor (`CalcPredictivePulseForApproximation`, moving-object state machine) can be adapted into GR4's `ZafPredictor` (`0x536192a4`).

#### C. Star AF / Astrophotography Pinpoint AF (`StarafProcess` + `AstroCalcFwhm`)
- **K-3 III Modules (~39 KB total, 100% CPU/VFPv4 Software)**:
  - `src/AstroCalcFwhm.cpp` + `FwhmCalcPipeService` (`0x53031898..0x530343d0`, `vt=0x53a49108`, **11,896 B**): Computes stellar point-spread Full Width at Half Maximum (`FWHM[um]`), `StarCnt`, `BGLevel`, and `ThrLevel` directly from the Live View Bayer buffer (`[LVS3A] Bayer Adrs FWHM:0x%08X`) using ARM Cortex-A7 VFPv4 `f64` instructions.
  - `src/StarafProcess.cpp` (`0x52cfc55c..0x52d02f50`, `vt=0x5389efd0`, **27,124 B**): Inherits from `AfProcessBase` (identical 42-slot vtable layout to GR4's `AfProcessBase` @ `0x535ec718`), driving the lens through `DRIVE_INF` -> `DRIVE_OPPOSITE_SIDE` -> `DRIVE_PEAK_AROUND` -> `DRIVE_PEAK`.

---

### Tier 2 — Already Superior on GR4 (Keep Native)
- **On-Sensor Phase-Detection (`Zaf*`) & Hybrid AF (`HybridAfInfo`)**:
  - K-3 III lacks on-sensor PDAF pixels in Live View and relies solely on `CafProcess`.
  - GR4 features 165-block (`0xa5`) on-sensor PDAF (`ZafCorrelation` @ `0x5360f250`, `ZafDefocus` @ `0x53611334`, `ZafOperationAlgorithm` @ `0x536165c0`, `PdafCorrectPipeService` @ `0x53fc6540`) orchestrated by `HybridAfInfo` (`0x535ecc58`, `vt=0x53ea7b70`).
- **Modular 3A Live View Architecture**:
  - GR4 uses decoupled `3A/src/LiveViewProcess3aContoller.cpp` (`0x536520f0`), `LiveViewProcessAf.cpp`, `LiveViewProcessFaceDetection.cpp`, and `LiveViewProcessTrackObject.cpp`.

---

### Tier 3 — Hardware-Specific (Tune Locally on GR4, Do Not Port from K-3 III)
- **OVF Dedicated Sensors**: K-3 III's SAFOX 13 (`PafSensor*`) and 307k-pixel RGB-IR `AeSensor` communicate over the `PPU2` hardware bus (`PPU2ControlPafRegDriver` @ `0x5319f8d8`), which does not exist on GR4.
- **Physical Lens Actuators**:
  - K-3 III uses K-mount lens communication (`AfInMotorLensControl`, `13_FDRVPARA.bin`) and screw-drive (`AfJointLensControl`).
  - GR4 drives its internal 18.3mm f/2.8 lens via Rohm stepping motor ICs **`MotorDriverBU24026GU`** (`vt=0x54320978`) and **`MotorDriverBU24038GW`** (`vt=0x54320b48`) + **`SricM343Interface`** (`08_SRIC.bin`), controlled by `LensFocusControl` (`src/LensFocusControl.cpp`, `vt=0x541da860`), `AFDFTuning.cpp` (`0x54043d64`), and `FocusCommand` (`focus_spd` @ `0x53056230`, `focus_req` @ `0x53056144`).
