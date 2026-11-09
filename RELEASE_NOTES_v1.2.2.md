# Ricoh GR IV Firmware Mod v1.2.2: Hybrid AF Branch Hardening & Boundary Assurance

## Summary of Changes in v1.2.2

### 1. Embedded Neural Descriptor Table Relocation (`0x53001600`)
- **Safety Margin Expansion**: Relocated `Embedded Desc Table` (`k3_desc.bin`, 728 B) from `0x53001440` to **`0x53001600`**.
- **Zero Collision Margin**: Cave 2 ends at `0x53000e00 + 1604 B = 0x53001444`. The new location provides **444 pure zero bytes (`0x00`) of safety gap** (`0x53001444 .. 0x53001600`), completely precluding any possibility of literal pool or descriptor overlap.
- **ROM Padding Clearance**: The descriptor table ends at `0x530018d8`, well below the first active RTOS code instruction at `0x53002000` (1,832 bytes of zeros remaining).

### 2. Level 1 FotoNation `DetectFace` $\to$ Level 2 CEVA-XM4 CNN Branch Hardening
- **Root Cause of Short-Circuit**: In embedded C ABI, `LibAHFD::DetectFace` (`0x53866594`) returns `r0 == 0` on every normal frame (`ERR_NONE` / `AHFD_OK`), regardless of whether human faces were detected. Evaluating `cmp r0, #0; beq stock` previously forced 100% of frames into the stock fallback, starving Level 2 CEVA-XM4 CNN inference.
- **Hierarchical Deterministic Fallback**:
  1. Cave 2 calls `LibAHFD::DetectFace` (`0x53866594`). If hardware error (`r0 != 0`), falls through to CEVA-XM4 CNN.
  2. If `r0 == 0`, Cave 2 immediately executes `FdLibraryService::GetResults` (`0x5386666c`) with `r0 = [r4, #0x38]`, `r1 = [r4, #0x78]`.
  3. `GetResults` updates detected faces and stores `face_count` into `[FdLibraryService, #0x30]`.
  4. Cave 2 tests converted `face_count > 0`:
     - **If `face_count > 0`**: Genuine frontal faces/eyes detected! Directly branches to `0x538963ac` (unlocks mutex, updates OSD/PDAF at 60 fps zero latency).
     - **If `face_count == 0`**: No frontal face found! Deterministically falls through into Level 2 CEVA-XM4 CNN to detect profiles, turned heads, pets (cats/dogs), and birds.

### 3. Binary Verification & Checksums
- Dual word-sum checksums verified: `0x00000000` (100% compliant with IPL container requirements).
- `fwdc248b.bin`:
  - Size: 39,109,096 bytes
  - SHA-256: `49ea52a5620c89a59f03cbd79e38bc70dda0a281bdc1dad59c0621d5e1ce8cbf`
- SD Card Binaries (Unchanged from v1.2.1):
  - `DF_EX.BIN`: 5,251,648 bytes (SHA-256: `e8f49514757c2bc38545a909e3a6a1253457a82cbf3dfec737c35f29910d6526`)
  - `DF_ID_S.BIN`: 65,536 bytes (SHA-256: `6608922883e200c286b24d788118ce5757d9b936a28e750ca9e1a1b16262b9a7`)
  - `DF_IP.BIN`: 16,384 bytes (SHA-256: `01217fc35a299146dfd150bcf96aee0563405c1d1a61ae9a04a58d19760086db`)
