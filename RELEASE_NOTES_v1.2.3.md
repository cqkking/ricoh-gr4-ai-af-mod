# Ricoh GR IV Firmware Mod v1.2.3: Pure K-3 III Deep Learning Exclusive AF & Diagnostic Heartbeat Box

## 概述 (Overview)

v1.2.3 是专为**单一变量验证（Single-Variable Verification）**设计的特别诊断固件。为了彻底消除理光 GR4 原厂 FotoNation 局部模板追踪器对真机测试的“假阳性”混淆与状态污染，v1.2.3 **100% 旁路禁用了原厂人脸识别函数**，并在 Cave 2 中引入了**右下角诊断心跳框（Diagnostic Heartbeat Box）**。

刷入本版本后，取景器屏幕上出现的所有对焦框**100% 来自 CEVA-XM4 DSP 深度学习推理与状态机**，不再存在任何原厂代码介入的可能。

---

## 核心改动 (Key Changes in v1.2.3)

### 1. 原厂 FotoNation 识别管线 100% 彻底物理旁路 (Stock Level 1 Bypass)
- **切除调用点**：彻底移除 Cave 2 入口对 `LibAHFD::DetectFace` (`0x53866594`) 与 `FdLibraryService::GetResults` (`0x5386666c`) 的所有调用。
- **独占控制权**：`FaceDetectionService` 的 `m_DetectedFaceResult`（`[r4, #0x78]`）和 `FdLibraryService`（`[r4, #0x38]`）的 `face_count`（`+0x30`）完全由 K-3 III 深度学习管线独占读写，彻底杜绝原厂代码在后台清零或覆盖坐标。
- **防御性画幅兜底**：动态下采样读取 `[r4, #0xfc]` (`fd_w`) 和 `[r4, #0xfe]` (`fd_h`) 时，若读出为 0 则自动兜底为 `320 x 212`，杜绝因原厂函数未调用导致的除零或全零乘积。
- **安全跳回收尾**：管线执行完毕后直接跳转至 `0x538963ac`（释放互斥锁并调用 `UpdateFaceDetectionResult`），绝不落入原厂 `0x53896460` (`ConvertResult`)。

### 2. 诊断“心跳框”机制 (Diagnostic Heartbeat Box)
在 `DspService::WaitMessage` (`0x5398e778`) 成功返回 0（代表 CEVA-XM4 完成一帧 IPC 推理并回传热力图）、但画面未检出目标（`max_diff < 3` 或 `area_cnt < 2`，如对准大白墙）时，注入一个固定的**右下角 32×32 像素诊断心跳框**：
- **坐标定义 (Q8 定点数)**：
  - `left = 240 px (0x0f000)`
  - `top = 140 px (0x08c00)`
  - `box_size = 32 px (0x02000)`（高于原厂 `SelectMainFace` 阈值 `0x1e00`，确保 OSD 稳定显示）
  - `cx = 256 px (0x10000)`, `cy = 156 px (0x09c00)`
  - `score = 99`, `face_id = 1`
  - 自动生成合规的眼部微框（`eye1`、`eye2`）

---

## 4 种真机测试诊断结论判定表 (Diagnostic Matrix)

| 屏幕表现 | 判定结论 | 底层状态解释 |
| :--- | :--- | :--- |
| **现象 A：屏幕右下角出现固定绿色小方框**（对着大白墙） | **全链路 100% 跑通！** | 证明 SD 卡固件加载成功、CEVA-XM4 成功进入 STARTED 状态、AHIP 正常出帧、CPU 下采样完成、跨核 IPC 握手成功、`WaitMessage` 正常返回，OSD 画框与 PDAF 引导彻底畅通！ |
| **现象 B：对准人物/猫狗时，框从右下角跳到人脸上并追焦** | **深度学习模型与权重 100% 完美！** | 证明 CEVA-CV 卷积神经网络、CNN 权重、输出热力图扫描、动态坐标映射全部工作正常。 |
| **现象 C：对准人脸时框依然停在右下角不移动** | **数据输入/权重断点** | 链路已通，但下采样张量颜色空间或网络权重产生的值未越过检测阈值，仅需微调阈值或颜色格式。 |
| **现象 D：屏幕上自始至终没有任何对焦框** | **WaitMessage 前断点** | 说明故障发生在 `WaitMessage` 之前（如 SD 卡文件未找到、AHIP 描述符空指针、或启动锁死）。 |

---

## 固件镜像与校验和 (Firmware Artifacts & Checksums)

- **固件容器校验和**：双字累加和（Dual word-sums）严格等于 `0x00000000`（100% 符合 IPL 引导校验要求）。
- **可刷写固件文件**：`output/fwdc248b.bin`
  - 文件大小：`39,109,096` 字节
  - SHA-256：`9e150fde61ba0ed57afaf6b1a62ba3c04abb4823362faa29519f008be8679d98`
- **SD 卡根目录依赖文件**：
  - `DF_EX.BIN` (5,251,648 字节, SHA-256: `b16788a16a11ecc2c007d26888a56195b2b4e1bec8aa05b74ea6789803ae8d44`)
  - `DF_ID_S.BIN` (65,536 字节, SHA-256: `3d3ebf29622323bb4a203347feaf676aeac110a7b1181604270a1ae0c22b0a2a`)
  - `DF_IP.BIN` (16,384 字节, SHA-256: `cf409a84f3f2d8fac1491634ecda882c9d76b3439c2423da1b31950463a52067`)
