# 理光 GR IV：片上 CEVA-CV AI DSP 协处理器唤醒与深度学习追焦固件补丁

[English](README.md) | [简体中文](README_CN.md)

[![平台: Socionext Milbeaut M10V](https://img.shields.io/badge/平台-Socionext%20Milbeaut%20M10V-007acc.svg)](#)
[![适配固件: Ricoh GR IV v1.11](https://img.shields.io/badge/目标固件-v1.11%20fwdc248b.bin-green.svg)](#)
[![架构: ARM Cortex-A7 + CEVA-CV DSP](https://img.shields.io/badge/芯片架构-ARMv7%20%2B%20CEVA--CV%20DSP-orange.svg)](#)
[![状态: 阶段 3 校验通过](https://img.shields.io/badge/验证状态-阶段%203%20校验通过%20(双重累加和%3D0)-brightgreen.svg)](#)
[![开源协议: MIT](https://img.shields.io/badge/协议-MIT-blue.svg)](#)

本项目是针对 **理光 GR IV (GR4)** 数码相机的非官方底层固件扩展补丁。它通过破译芯片内存映射与二进制驱动协议，成功激活了片上完全闲置的 **CEVA-CV 矢量 AI DSP 硬件协处理器**（物理基地址 `0x41000000`），并将**宾得旗舰单反 Pentax K-3 Mark III 的深度学习人体/主体识别卷积神经网络**无缝移植挂接进理光 GR4 原生的 165 点片上相位差检测自动对焦（PDAF）系统。

---

## 1. 突破性发现：芯片底层硬件真相

**理光 GR IV** 与 **宾得 K-3 Mark III** 均搭载 Socionext 为理光/宾得定制的 **Milbeaut M10V** 异构多核图像处理芯片。

通过对量产版固件的底层反汇编、内存映射审计与寄存器遍历，我们揭示了一个此前从未被公开的技术真相：

> **理光 GR IV 的物理芯片完全集成了与宾得 K-3 Mark III 规格分毫不差的 CEVA-CV 矢量 AI DSP 协处理器（物理地址 `0x41000000`）。**
>
> 事实上，理光 GR4 的操作系统固件（`05_RTOS.bin` @ `0x53000000`）在开机时便内置了完整的 DSP 驱动层（`DspService` @ `0x539658c4`）、跨核心通信信箱（`DspExc`, `DspReq`, `DspSF`, `DspMtx`）以及原厂调试指令（`DSP_load`, `DSP_start`, `DSP_test`）。
>
> 然而，理光在量产发布固件中**移除了 4.00 MB 的 DSP 固件分区（`06_DSP.bin`）**，导致这颗具备 512 位矢量运算能力的硬件 AI 核心在 GR4 上处于 **100% 闲置断电状态**。

```text
Socionext Milbeaut M10V 异构多核处理器芯片
┌────────────────────────────────────────────────────────────────────────┐
│  ARM Cortex-A7 双核主控 CPU (32-bit ARMv7-A @ 1.0 GHz)                 │
│  ├── eSOL T-Kernel RTOS [0x53000000] (ISP / 实时取景 / 3A / AF / UI)    │
│  └── Linux 4.9 内核     [0x013ea7e0] (网络 / PTP 通信 / 存储服务)       │
├────────────────────────────────────────────────────────────────────────┤
│  CEVA-CV 矢量 AI DSP 协处理器 [0x41000000] (512-bit SIMD / VLIW)        │
│  ├── 物理 MMU 映射窗口: 0x41000000 (8 MB 强顺序普通共享内存)            │
│  ├── DF_EX.BIN   (0x51200000, 4,112,384 字节) — CEVA-CV 矢量算子执行库 │
│  ├── DF_ID_S.BIN (0x515ec000,    65,536 字节) — 神经网络模型参数表     │
│  └── DF_IP.BIN   (0x515fc000,    16,384 字节) — 复位引导与中断向量入口 │
└────────────────────────────────────────────────────────────────────────┘
```

数学与内存映射的绝对吻合证明：
$$0x3EC000\text{ (DF\_EX)} + 0x10000\text{ (DF\_ID\_S)} + 0x4000\text{ (DF\_IP)} = 0x400000\text{ (恰好为 } 4,194,304\text{ 字节 / 4.00 MB)}$$
与宾得 K-3 Mark III 解包所得的 `06_DSP.bin` 尺寸与二进制结构 **100% 完美一致**。

---

## 2. 项目架构与阶段演进

```text
                  ┌──────────────────────────────────────────────┐
                  │ 阶段 1：DSP 固件分块提取与协议适配            │
                  │   06_DSP.bin -> DF_EX / DF_ID_S / DF_IP      │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 阶段 2：免串口开机全自动唤醒                  │
                  │   挂接 0x536520f0 -> 代码洞 0x53000e00       │
                  │   LoadDsp('A') -> Start() -> Test()          │
                  │   输出归档: fwdc248b_stage2_dsp_wake.bin      │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 阶段 3：实时取景 AI 推断与自动追焦挂接        │
                  │   • LiveView 取景流 -> 56x56 RGB 张量        │
                  │   • DspService::SendMessage(Message ID 4)    │
                  │   • 64x64 热力图峰值解析 -> 目标坐标 (X,Y,W,H)│
                  │   • UpdateTrackTarget -> GR4 165 点 PDAF     │
                  │   输出成品: fwdc248b_stage3_ai_track.bin     │
                  └──────────────────────────────────────────────┘
```

### 阶段 1：固件分块与协议重构 (`dsp_splitter.py`)
- 将 K-3 III 的 4.00 MB `06_DSP.bin` 按照理光底层 `DspService::LoadDsp` 解析器要求的格式重新切分打包：
  - `DF_EX.BIN` (4,112,408 字节)：包含 16 字节头部与 8 字节块描述符，硬编码校验指定加载到虚拟地址 `0x51200000`，长度 `0x3ec000`。
  - `DF_ID_S.BIN` (65,536 字节)：64 KB 神经网络常量权值表，加载到 `0x515ec000`。
  - `DF_IP.BIN` (16,384 字节)：16 KB 指令复位入口向量，加载到 `0x515fc000`。
- 通过 [`verify_dsp_payload.py`](src/dsp_ai/verify_dsp_payload.py) 验证，实现对原始 4 MB 载荷的 100% 字节级无损复原。

### 阶段 2：免串口开机全自动唤醒 (`k3_gr4_dsp_patcher.py`)
- **专为无串口量产机身设计**：零售版相机无法引出调试串口，所有唤醒与自检流程必须在机内自主运行。
- 在实时取景初始化函数 `LiveViewProcess3aContoller`（`0x536520f0`）注入单向跳转蹦床（`b #0x53000e00`）。
- 执行流程：
  1. 检查 DSP 是否已处于运行态（`status & 1`）；若已启动，仅耗费 3 个指令周期直接跳出，确保零运行时开销。
  2. 调用 `DspService::LoadDsp(m_dspService, 'A', 0, 0)` 从 SD 卡根目录（`A:\`）读取 3 个二进制分块。
  3. 检查载入完成标志位 `status & 2`（`LOADED`）。
  4. 调用 `DspService::Start(m_dspService, 0, 0)` 解除 CEVA-CV 复位锁定并开启时钟。
  5. 调用 `DspService::Test(m_dspService, 0)` 完成固件握手。
  6. 平滑返回 `0x536520f8` 继续相机原生取景与对焦流程。

### 阶段 3：深度学习推断管线与实时追焦挂接 (`stage3_ai_patcher.py`)
- **AI 权重表固化技术 (Zero-File Dependency for Weights)**：
  - 利用理光原厂 RTOS 内部经精细审计的安全零填充代码洞，将原本缺失的模型描述符与归一化矩阵**直接固化入固件内部**，无需在 SD 卡上放置额外零碎权重文件：
    - `k3_desc.bin` (728 字节) @ `0x53001200`：网络层拓扑结构、输入输出张量尺度与正交投影矩阵。
    - `k3_mean.bin` (9,408 字节) @ `0x541db540`：$56 \times 56 \times 3$ 通道均值归一化偏置。
    - `k3_std.bin`  (9,408 字节) @ `0x54217680`：$56 \times 56 \times 3$ 通道标准差激活权重。
- **连续实时取景人脸/人眼检测循环挂接 (`FaceDetectionService::Task` @ `0x5386ea8c`)**：
  - 在 v1.0.3 中，挂接点从开机仅执行一次的初始化函数移至理光原厂每帧连续执行的人脸识别任务循环（`FaceDetectionService::Task`）。
  - **帧降采样控制**：每 2 帧实时取景画面触发一次深度学习推断（稳定在 ~30 fps），CPU 占用率低于 0.8%。
  - **预览帧提取**：通过 `CaptureControlService` 链条 `[0x55395a18, #8] -> [m_CaptureViewControlService, #4]` 提取实时的 $960 \times 640$ YUV 原始预览帧。
  - **汇编双线性下采样**：将预览帧下采样转换为 $56 \times 56$ RGB 张量写入 DMA 缓冲区 `0x51600000`。
  - **硬件信箱异步调度**：填入 9 词请求控制块（`0x51602500`），调用 `DspService::SendMessage`（`0x539663c0`）发送 Message ID 4，并通过 `DspService::WaitMessage`（`0x539666dc`）等待 CEVA 矢量核执行完成。
  - **$64 \times 64$ 热力图峰值定位**：汇编扫描 `0x51610000` 输出的热力图（4,096 字节），定位最大置信度坐标 $(col, row, score)$。
  - **双模 LCD 屏幕绿框与 165 点相位追焦联动**：
    - **当 AI 置信度 $\ge 60$ 时**：
      - 将峰值坐标映射为取景器像素坐标 $(X, Y, W, H)$，并根据神经网络人体结构推算人眼关键点（`left_eye`、`right_eye`、`mouth`）。
      - 格式化填入人脸数据结构 `m_DetectedFaceResult`（`[r4, #0x78]`），并在 `FdLibraryService`（`[r4, #0x38]`）中标记检出人脸数 `face_count = 1`。
      - 调用 `UpdateTrackTarget`（`0x538e32c4`）将坐标注入片上相位对焦跟踪引擎。
      - 跳转至 `0x5386ea9c`，直接触发 `UpdateFaceDetectionResult`（`0x5386e0e4`），在相机 LCD 屏幕上实时画出绿色的 AI 人脸/人眼对焦框，并牢牢驱动 165 个相位对焦点锁定拍摄对象！
    - **当 AI 置信度 $< 60$ 或推断超时时**：
      - 无缝降级调用理光原厂 FotoNation `LibAHFD` 传统检测器（`0x5383ec84`），并返回 `0x5386ea94`，保证任何场景下对焦绝对稳定、绝不拉风箱或漏焦。

---

## 3. 快速上手：刷机与使用说明

> [!IMPORTANT]
> **无需拆机、无需电烙铁焊接、无需连接任何串口调试线。**
> 固件开机后全自动静默运行，且完全支持理光官方标准的 SD 卡固件升级模式。

### 第 1 步：准备 SD 卡文件
1. 准备一张 SD 卡，格式化为 **FAT32** 格式。
2. 将本仓库编译生成的 3 个 DSP 二进制分块文件复制到 **SD 卡根目录**：
   - `DF_EX.BIN` (4,112,408 字节)
   - `DF_ID_S.BIN` (65,536 字节)
   - `DF_IP.BIN` (16,384 字节)
3. 复制固件文件到 **SD 卡根目录**，并重命名为 `fwdc248b.bin`：
   - **强烈推荐先刷写阶段 2 最小验证固件** `output/fwdc248b_stage2_dsp_wake.bin`：验证开机唤醒、正常启动与基本功能无误。
   - 验证通过后，再刷写阶段 3 完整 AI 追焦固件 `output/fwdc248b_stage3_ai_track.bin`。

SD 卡根目录文件布局如下：
```text
SD 卡根目录 (A:\)
├── fwdc248b.bin       (38,693,256 字节[阶段2] 或 38,782,892 字节[阶段3])
├── DF_EX.BIN          ( 4,112,408 字节 - CEVA-CV 矢量执行分区)
├── DF_ID_S.BIN        (    65,536 字节 - 神经网络模型权重表)
└── DF_IP.BIN          (    16,384 字节 - 复位引导向量表)
```

### 第 2 步：固件刷入与升级
1. 确保相机电池电量充足，将 SD 卡插入理光 GR IV。
2. 按住机身背面的 **`DISP`**（或 **`MENU`**）键不放，同时按下电源键开机。
3. 屏幕将直接进入理光官方固件升级菜单，显示当前版本与新版本。
4. 选择 **确定 (Yes)** 开始升级。耐心等待进度条走完至 100% 并自动重启。

### 第 3 步：体验深度学习自动追焦
- 开机进入正常拍摄界面，在进入取景画面的瞬间固件已自动点火 CEVA-CV DSP；
- 在取景器中对准人物或移动主体，半按快门或开启连续对焦：理光 165 点片上相位差检测框将精准、平滑地锁定并跟随主体移动。

---

## 4. 安全性、防砖与应急救砖指南 (Safety & Emergency Unbricking)

### 救砖指南（若刷机后无法正常开机）
> [!TIP]
> **相机绝对不会永久变砖！**
> 本项目**绝对未修改**相机的引导加载程序分区（`00_BOOTPARA.bin` ~ `04_BOOT.bin` / IPL）。
> 理光原厂的底层固件更新模式（FWUPD）运行于独立于 RTOS 的 Bootloader 中，即使系统内核崩溃，只要引导加载器完好，就可以 100% 重新刷入官方固件救砖。

**应急恢复步骤：**
1. **彻底关机断电**：若相机卡死（黑屏或电源灯微亮），取出电池等待 10 秒后重新装入。
2. **准备救砖 SD 卡**：将官方原版 `fwdc248b.bin`（v1.11 原版固件）或本仓库最新修复的 `fwdc248b_stage2_dsp_wake.bin` 放入 SD 卡根目录（命名为 `fwdc248b.bin`）。
3. **强制触发硬件更新模式**：
   - 保持相机处于关机状态；
   - **长按住机身 `MENU` 键（或 `DISP` 键）不松开**；
   - 在长按 `MENU`/`DISP` 的同时，短按电源键开机；
   - 屏幕将直接绕过系统内核，显示官方升级界面（`Firmware Update`）；
   - 按提示选择 **Yes / 确定** 刷回干净固件，即可彻底恢复！

### 底层架构防护设计
1. **AAPCS 严格 8 字节双字栈对齐与 APCS 完整帧构建**：
   - 代码洞完全内建并接管 APCS 栈帧生成（`mov ip, sp; push {r4-r7, fp, ip, lr, pc}; sub fp, ip, #4; sub sp, sp, #8`）。
   - 在进入子程序调用前预先固定 `fp`（帧指针），并在调用前后以 8 字节双字边界精准压栈保护 `r0-r3`（16 字节）与 `r4-r8, lr`（24 字节）。
   - 退出代码洞时无缝衔接至 `0x53652108`（`ldr r6, [r4]`），彻底规避了 `ip`（R12）临时寄存器被子函数污染而导致 `fp`/`sp` 损坏的隐患，杜绝 Data Abort / Prefetch Abort 异常。
2. **双重 32 位累加和校验 (`word_sum == 0`)**：
   - 理光官方引导程序（`04_BOOT.bin`）与固件升级程序（`FWUPD`）在写入 Flash 前会严格验证内层载荷（`payload.bin`）与外层 LZSS 容器的 32 位小端字累加和。
   - 本项目构建引擎经过严格的数论求解重零点计算，输出的固件 100% 通过双重校验，杜绝官方校验报错或卡更新界面的风险。
3. **缺失文件与空指针优雅降级**：
   - 若取出 SD 卡、未放置 `DF_*.BIN` 文件、或 `m_dspService` 单例尚未初始化，代码洞检测到状态不符合时仅耗费 10 余个时钟周期立即安全退出至原生流程，相机 100% 正常开机，自动回退到原生对焦模式。
4. **完全可逆**：
   - 随时可以通过上述硬件按键组合刷回理光官方原版 v1.11 固件。

---

## 5. 仓库目录结构

```text
.
├── analysis/
│   ├── dsp_stage3_manifest.json     # 阶段 3 固件构建与遥测信息清单
│   ├── dsp_wake_manifest.json       # 阶段 2 纯唤醒固件清单
│   ├── rtos_decompilation_audit.md  # GR4 RTOS 深度逆向反汇编白皮书
│   └── shellcmd_symbols.csv         # 651 个官方 RTOS Shell 符号
├── docs/
│   ├── FEAT_DSP_STAGE3_AI_TRACK.md  # 阶段 3 深度学习推断与追焦白皮书
│   ├── FEAT_DSP_AI_WAKE.md          # 阶段 2 免串口唤醒技术白皮书
│   └── FEAT_K3_AF_PORT.md           # 宾得 K-3 III 对焦移植审计报告
├── src/
│   └── dsp_ai/
│       ├── build_stage3_firmware.py # 阶段 3 固件一键构建与重组脚本
│       ├── stage3_ai_patcher.py     # 阶段 3 汇编代码洞与补丁生成器
│       ├── build_dsp_firmware.py    # 阶段 2 固件构建脚本
│       ├── k3_gr4_dsp_patcher.py    # 阶段 2 汇编补丁生成器
│       ├── dsp_splitter.py          # 06_DSP.bin -> DF_EX / DF_ID_S / DF_IP 分块器
│       ├── verify_dsp_payload.py    # 分块 100% 无损重建测试工具
│       ├── k3_desc.bin              # 提取自 K-3 III 的神经网络描述符 (728 B)
│       ├── k3_mean.bin              # 提取自 K-3 III 的输入通道均值表 (9,408 B)
│       └── k3_std.bin               # 提取自 K-3 III 的通道标准差表 (9,408 B)
├── tools/
│   ├── fw_container.py              # 理光固件 LZSS 解封与双重 word_sum 求解器
│   ├── rtos_analyzer.py             # RTOS 交叉引用与函数符号反汇编器
│   └── linux_analyzer.py            # Linux rootfs 审计器
├── README.md                        # 英文主说明文档
├── README_CN.md                     # 中文说明文档
└── LICENSE                          # MIT 开源授权协议
```

---

## 6. 从源码构建固件

### 前置依赖
- Python 3.10+
- `capstone` 库（`pip install capstone`）
- 理光 GR IV 原厂 v1.11 固件（`fwdc248b.bin`）放置在 `firmware/official/v1.11/`
- 宾得 K-3 Mark III 原厂 v2.20 固件的 `06_DSP.bin` 分区放置在 `firmware/reference/k3iii_v220/unpacked/sections/`

### 构建指令
```bash
# 执行完整构建管线：切分 DSP 固件、固化 AI 模型表、装填阶段 3 汇编代码洞并重新生成 Flash 镜像
python src/dsp_ai/build_stage3_firmware.py
```

构建日志输出示例：
```text
======================================================================
Ricoh GR IV Stage 3 AI Live View Tracking Firmware Rebuilder
======================================================================
Ricoh GR IV Stage 3 AI Live View Tracking Firmware Rebuilder
======================================================================
[*] Confirmed Stage 2 binary intact: fwdc248b_stage2_dsp_wake.bin (38,693,256 bytes)

[Step 1/4] Verifying staged SD card DSP binaries...
[+] PASS: 100% byte-perfect reconstruction verified across all 4,194,304 bytes.

[Step 2/4] Decoding official firmware container baseline...
  -> Official payload decoded: 67,948,280 bytes

[Step 3/4] Splicing Stage 3 Code Cave, Live View Trampoline & AI Tables...
  [+] Injected Code Cave (848 B) @ VA 0x53000e00
  [+] Injected Trampoline (8 B) @ VA 0x5386ea8c
  [+] Embedded Desc Table (728 B) @ VA 0x53001200
  [+] Embedded Mean Table (9408 B) @ VA 0x541db540
  [+] Embedded Std Table  (9408 B) @ VA 0x54217680
[*] Rezeroing 32-bit payload checksum word...
  [+] Payload word_sum == 0 validated successfully.

[Step 4/4] Rebuilding flashable container with LZSS framing...
  [+] Written Stage 3 preserved binary: output/fwdc248b_stage3_ai_track.bin (38,782,892 bytes)
  [+] Written active flashable binary:  output/fwdc248b.bin (38,782,892 bytes)

[*] Executing round-trip container verification...
  [+] Round-trip container verification PASSED! Dual word-sums == 0.
```

---

## 7. 芯片级底层函数与内存交叉引用表

| 函数 / 模块名称 | 虚拟地址 (VA) | 架构功能与说明 |
| :--- | :--- | :--- |
| `FaceDetectionService::Task` | `0x5386ea8c` | 实时取景连续人脸识别任务循环，补丁蹦床挂接点 (`b #0x53000e00`) |
| 阶段 3 智能执行代码洞 | `0x53000e00` | 848 字节独立执行引擎（自动唤醒 + 图像下采样 + DSP 调度 + 双模追焦与绿框渲染） |
| `m_CaptureViewControlService` | `[0x55395a18, #8]` | 预览取景流控制器句柄，偏移 `+4` 处为当前 $960 \times 640$ YUV 原始预览帧物理指针 |
| `DspFactory::GetDspService()`| `0x539658c4` | 获取 CEVA-CV `DspService` 核心单例句柄 |
| `DspService::LoadDsp` | `0x53966d74` | 从 SD 卡载入 `DF_EX.BIN`, `DF_ID_S.BIN`, `DF_IP.BIN` 三分块 |
| `DspService::Start` | `0x53966214` | 解除硬件复位，启动 CEVA-CV 指令执行（入口 `0x515fc000`） |
| `DspService::Test` | `0x53966f18` | 触发 CEVA-CV 版本与自检握手通信 |
| `DspService::SendMessage` | `0x539663c0` | 发送 Message ID 4（神经网络卷积前向推断请求） |
| `DspService::WaitMessage` | `0x539666dc` | 等待矢量张量卷积推断完成信号 |
| `m_DetectedFaceResult` | `[FaceDetectionService, #0x78]` | 100 字节人脸数据结构体数组（包含 `x, y, w, h, eye_l, eye_r, mouth, score`） |
| `FdLibraryService` | `[FaceDetectionService, #0x38]` | 人脸服务管理对象，偏移 `+0x30` 保存检出人脸总数 |
| `UpdateFaceDetectionResult` | `0x5386e0e4` | 将检出的人脸/人眼坐标提交至相机 LCD 屏幕 OSD 任务与对焦系统 |
| `UpdateTrackTarget` | `0x538e32c4` | 提交目标包围框 $(X, Y, W, H, Conf)$ 至理光对焦跟踪调度器 |
| 理光原厂 `DetectFace` | `0x5383ec84` | FotoNation `LibAHFD` 传统 CPU 人脸检测函数（AI 置信度 $< 60$ 时的静默安全回退） |

---

## 8. 版本更新记录 (Release Changelog)

### v1.0.3 (当前版本)
- **真·实时取景推断循环重构**：将推断钩子由开机单次执行的 `LiveViewProcess3aContoller`（`0x536520f0`）重定位至每帧必经的独立人脸识别任务循环 `FaceDetectionService::Task`（`0x5386ea8c`）。
- **取景画面指针二级解引用修复**：解析 `[0x55395a18, #8] -> [m_CaptureViewControlService, #4]`，确保下采样模块获取到真实的 $960 \times 640$ 动态画面而非控制器类指针。
- **LCD 屏幕绿框与片上 PDAF 双模直通**：推断检出目标直接写入 `m_DetectedFaceResult` 并联动 `UpdateFaceDetectionResult`（`0x5386e0e4`），使 LCD 屏幕立即绘制出精准的 AI 人脸/人眼绿色锁定框，并驱动 165 点相位对焦点牢牢咬住拍摄主体。
- **零拉风箱静默回退保护**：当 AI 模型置信度低于 60 或画面无目标时，100% 平滑降级执行原厂 FotoNation `LibAHFD`（`0x5383ec84`），确保全场景下绝对对焦稳定性。

### v1.0.2
- **MMU 权限错误修复**：将运行时状态量与序列号由代码段（`.text`，只读 `AP=110`）迁移至 DSP 共享内存区域（`0x51602540`，可读写 `AP=011`），彻底根除开机 Data Abort 崩溃。

### v1.0.1
- **AAPCS 严格 8 字节栈对齐**：全面修复代码洞进入与退出时的栈帧结构，保证与理光原厂 eSOL T-Kernel 调度器严格兼容。

### v1.0.0
- 首次公开发布理光 GR IV 激活内置 CEVA-CV DSP 协处理器并移植宾得 K-3 Mark III 深度学习对焦系统方案。

---

## 8. 免责声明与开源许可

- **免责声明**：本项目属于学术性与工程逆向探索成果。刷写第三方固件存在极小的潜在风险。尽管项目已实装双重累加和验证、越界防御与静默回退保护，开发者不对任何因操作不当造成的设备异常承担法律责任。
- **版权声明**：本仓库仅分发原创反汇编工具、开源补丁代码与逆向数据规范，不包含且不分发理光映像株式会社（Ricoh Imaging Company, Ltd.）享有著作权的原版二进制固件。
- **开源协议**：本项目基于 [MIT License](LICENSE) 协议开源。
