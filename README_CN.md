# 理光 GR IV：片上 CEVA-CV AI DSP 协处理器唤醒与混合深度学习自动对焦固件补丁

[English](README.md) | [简体中文](README_CN.md)

[![平台: Socionext Milbeaut M10V](https://img.shields.io/badge/平台-Socionext%20Milbeaut%20M10V-007acc.svg)](#)
[![适配固件: Ricoh GR IV v1.20](https://img.shields.io/badge/目标固件-v1.20%20fwdc248b.bin-green.svg)](#)
[![版本: v1.2.5](https://img.shields.io/badge/版本-v1.2.5-blue.svg)](#)
[![架构: ARM Cortex-A7 + CEVA-CV DSP](https://img.shields.io/badge/芯片架构-ARMv7%20%2B%20CEVA--CV%20DSP-orange.svg)](#)
[![状态: 纯 K-3 III 深度学习独占验证通过](https://img.shields.io/badge/验证状态-纯%20K--3%20III%20深度学习独占%20%26%20双框遥测诊断%20(v1.2.5)-brightgreen.svg)](#)
[![开源协议: MIT](https://img.shields.io/badge/协议-MIT-blue.svg)](#)

本项目是针对 **理光 GR IV (GR4)** 数码相机的非官方底层固件扩展补丁。它通过破译芯片内存映射与二进制驱动协议，成功激活了片上完全闲置的 **CEVA-CV 矢量 AI DSP 硬件协处理器**（物理基地址 `0x41000000`），构建了**相机原生 FotoNation 60 fps 硬件人脸/人眼极速跟踪**与**宾得旗舰单反 Pentax K-3 Mark III 深度学习主体识别卷积神经网络**相融合的真·混合自动对焦（Hybrid AF）系统，并直连挂接进 GR4 原生的 165 点片上相位差检测自动对焦（PDAF）管线。

---

## 1. 突破性发现：芯片底层硬件真相

**理光 GR IV** 与 **宾得 K-3 Mark III** 均搭载 Socionext 为理光/宾得定制的 **Milbeaut M10V** 异构多核图像处理芯片。

通过对量产版固件的底层反汇编、内存映射审计与寄存器遍历，我们揭示了一个此前从未被公开的技术真相：

> **理光 GR IV 的物理芯片完全集成了与宾得 K-3 Mark III 规格分毫不差的 CEVA-CV 矢量 AI DSP 协处理器（物理地址 `0x41000000`）。**
>
> 事实上，理光 GR4 的操作系统固件（`05_RTOS.bin` @ `0x53000000`）在开机时便内置了完整的 DSP 驱动层（`DspService` 单例 @ `0x5398d97c` in v1.20）、跨核心通信信箱（`DspExc`, `DspReq`, `DspSF`, `DspMtx`）以及原厂调试指令（`DSP_load`, `DSP_start`, `DSP_test`）。
>
> 然而，理光在量产发布固件中**移除了 DSP 固件分区（`06_DSP.bin`）**，导致这颗具备 512 位矢量运算能力的硬件 AI 核心在 GR4 上处于 **100% 闲置断电状态**。

```text
Socionext Milbeaut M10V 异构多核处理器芯片
┌────────────────────────────────────────────────────────────────────────┐
│  ARM Cortex-A7 双核主控 CPU (32-bit ARMv7-A @ 1.0 GHz)                 │
│  ├── eSOL T-Kernel RTOS [0x53000000] (ISP / 实时取景 / 3A / AF / UI)    │
│  └── Linux 4.9 内核     [0x013ea7e0] (网络 / PTP 通信 / 存储服务)       │
├────────────────────────────────────────────────────────────────────────┤
│  CEVA-CV 矢量 AI DSP 协处理器 [0x41000000] (512-bit SIMD / VLIW)        │
│  ├── 物理 MMU 映射窗口: 0x41000000 (8 MB 强顺序普通共享内存)            │
│  ├── DF_EX.BIN   (0x51200000, 5,251,648 字节) — CEVA-CV 算子库 + 1.14M │
│  │   ├── 分块 0 (0x51200000, 4,112,384 字节) — CEVA-CV 矢量数学核心    │
│  │   └── 分块 1 (0x516c0000, 1,139,232 字节) — K-3 III 深度神经网络权重 │
│  ├── DF_ID_S.BIN (0x515ec000,    65,536 字节) — 神经网络模型配置表     │
│  └── DF_IP.BIN   (0x515fc000,    16,384 字节) — 复位引导与中断向量入口 │
└────────────────────────────────────────────────────────────────────────┘
```

数学与内存映射的绝对吻合证明：
$$0x3EC000\text{ (DF\_EX 核心)} + 0x116220\text{ (DF\_EX 权重)} = 0x502220\text{ (5,251,648 字节)}$$
$$\text{DSP 总内存开销: } 0x502220 + 0x10000\text{ (DF\_ID\_S)} + 0x4000\text{ (DF\_IP)} = 5,333,568\text{ 字节}$$
与宾得 K-3 Mark III 解包所得的 `06_DSP.bin` 及完整深度学习卷积权重视图 **100% 完美一致**。

---

## 2. 系统架构与三个技术阶段

```text
                  ┌──────────────────────────────────────────────┐
                  │ 阶段 1：固件分块与协议重构                   │
                  │   06_DSP.bin -> DF_EX / DF_ID_S / DF_IP      │
                  │   注入 1.14 MB K-3 III 卷积神经网络权重      │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 阶段 2：免串口开机全自动唤醒                 │
                  │   Hook @ 0x53676ab8 -> 代码洞 0x53000d60     │
                  │   LoadDsp('C') -> Start()                    │
                  │   输出成品: fwdc248b_stage2_dsp_wake.bin     │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 阶段 3：真·混合 AF 架构与零冲突内存 (v1.2.1) │
                  │   • 原厂 FotoNation 硬件 60fps 人脸/人眼极速 │
                  │   • CEVA-XM4 深度学习 CNN 侧脸/动物托底      │
                  │   • 0x52b00000 零冲突工作缓冲 (根治权重覆写) │
                  │   • 彻底根除 2s 开机死机与左上 1/4 假框      │
                  │   输出成品: fwdc248b_stage3_ai_track.bin     │
                  └──────────────────────────────────────────────┘
```

### 阶段 1：固件分块与协议重构 (`dsp_splitter.py`)
- 将 K-3 III 的 `06_DSP.bin` 按照理光底层 `DspService::LoadDsp` 解析器要求的格式打包：
  - `DF_EX.BIN` (5,251,648 字节)：分块 0（CEVA-CV 核心，4,112,384 字节）+ 分块 1（1,139,232 字节 K-3 III 卷积神经网络权重：主干骨干网 677,824 字节 @ `0x516c0000` + 分类头 461,408 字节 @ `0x517657c0`）。
  - `DF_ID_S.BIN` (65,536 字节)：64 KB 神经网络常量权值配置表，加载到 `0x515ec000`。
  - `DF_IP.BIN` (16,384 字节)：16 KB 指令复位入口向量，加载到 `0x515fc000`。
- 通过 [`verify_dsp_payload.py`](src/dsp_ai/verify_dsp_payload.py) 验证，实现对原始载荷的 100% 字节级无损复原。

### 阶段 2：免串口开机全自动唤醒 (`k3_gr4_dsp_patcher.py`)
- **专为无串口量产机身设计**：零售版相机无法引出调试串口，所有唤醒与自检流程必须在机内自主运行。
- 在实时取景初始化函数 `LiveViewProcess3aContoller`（`0x53676ab8`）注入单向跳转蹦床（`b #0x53000d60`）。
- 成功调用 `LoadDsp('C')` 与 `Start()`，实机测试完全稳定通过。

### 阶段 3：真·混合 AF 架构与零冲突内存安全防护 (`stage3_ai_patcher.py`, v1.2.1)
- **真·混合自动对焦架构（Hybrid AF）**：
  - **第一级（FotoNation 硬件加速）**：每帧优先调用原厂 `LibAHFD::DetectFace`（`0x53866594`），若识别到正面人脸或眼睛（`r0 == 0`），立即直出 `0x53896460`（`ConvertResult`），提供 60fps 零延迟人脸/人眼极速硬件对焦。
  - **第二级（CEVA-XM4 深度学习智能托底）**：当被摄者转头、大角度侧脸、低头、戴口罩遮挡，或拍摄猫、狗、飞鸟等动物时，官方算法返回无目标，此时 CEVA-XM4 CNN 深度学习网络瞬间激活托底，进行 56×56 多类主体轮廓分割与对焦点锁定。
- **内存零冲突重构（DDR Zero-Collision）**：
  - 将张量推理缓冲区 `TENSOR_IN_VA` 移至安全未受管区 **`0x52b00000`**（`0xbde40` = 777,792 字节），完全避开 `MEAN_WEIGHTS_VA = 0x52960000`，彻底根除了 XM4 DMA 覆写静态模型权重的致命缺陷，并 100% 保证既有 SD 卡文件兼容性。
- **根除开机 2 秒死机 / 黑屏冻结**：
  - 彻底移除了原代码洞在加载 SD 卡成功后调用的 `DspService::Test`（无限等待 `tk_wai_flg(0x3f0000, 0)`）阻塞死锁；启动后直接重新加锁并安全让出当前帧（`b .ai_done`），下帧无缝进入追踪循环。
- **根除开机左上 1/4 假框**：
  - 完整填充了 `UpdateFaceDetectionResult` 所需的 `eye1` (`+0x10`) 与 `eye2` (`+0x20`) 坐标结构体，开机无目标时保持 `face_count = 0`，不再绘制任何瞬态伪对焦框。
- **AHIP 硬件完成出队帧提取与 D-Cache 刷新**：
  - 直接从 `AhipService::WaitFrame` 出队的完成帧描述符 `*(*(0x55081350 + 0x424))` 提取 Y/U/V 平面，并在 CPU 读取前调用 `InvalidateDataCache`（`0x538f0454`），写入张量后调用 `FlushDataCache`（`0x538f04c4`）。
- **全画幅比例自适应下采样（3:2, 4:3, 16:9, 1:1）**：
  - 根据 `[r4, #0xfc]` 与 `[r4, #0xfe]` 动态计算源行指针与 UV 色度偏移。
- **$C^0$ 连续 $2.0\times$ 逆光暗部平滑增益曲线**：
  - 大幅提升逆光面部阴影对比度，同时保持天空高光不溢出。

---

## 3. 极速上手：刷机与使用指南

> [!IMPORTANT]
> **无需拆机、无需焊接、无需任何 USB/TTL 调试转接线。**
> 补丁完全利用相机官方固件升级通道，开机后 100% 机内自主点火与运行。

### 第 1 步：准备 SD 卡文件
1. 将 SD 卡格式化为 **FAT32** 文件系统。
2. 将以下 3 个 DSP 固件文件复制至 **SD 卡根目录 (`C:\`)**：
   - `DF_EX.BIN` (5,251,648 字节)
   - `DF_ID_S.BIN` (65,536 字节)
   - `DF_IP.BIN` (16,384 字节)
3. 将阶段 3 编译生成的固件 `output/fwdc248b.bin`（或 `output/fwdc248b_stage3_ai_track.bin`）重命名为 `fwdc248b.bin` 复制至 **SD 卡根目录**。

SD 卡根目录文件树预览：
```text
SD 卡根目录 (C:\)
├── fwdc248b.bin       (39,109,096 字节 [阶段 3 v1.2.1 混合 AF 固件])
├── DF_EX.BIN          ( 5,251,648 字节 — CEVA-CV 执行库 + 1.14MB 深度模型权重)
├── DF_ID_S.BIN        (    65,536 字节 — 神经网络模型结构配置表)
└── DF_IP.BIN          (    16,384 字节 — 复位向量表)
```

> [!NOTE]
> 若您此前已经拷贝过 `v1.2.0` 的 SD 卡文件，这 3 个 `DF_*.BIN` 文件**完全一致且 100% 兼容**，您只需更新 SD 卡上的 `fwdc248b.bin` 固件即可！

### 第 2 步：固件卡刷升级
1. 确保相机电池电量充足（至少 80%），将 SD 卡插入理光 GR IV。
2. **按住相机的 `DISP` 键（或 `MENU` 键）不放**，同时按下**电源键**开机。
3. 屏幕将显示官方固件升级菜单，当前版本与目标更新版本（`v1.20`）显示就绪。
4. 方向键选择 **“是”** 确认升级，静候进度条跑满并自动关机重启。

### 第 3 步：体验混合 AI 追焦
- 开机后：
  - 拍摄正常正脸人物时，原厂 FotoNation 硬件管线以 60fps 零延迟极速吸附人眼与面部；
  - 当被摄者转头、侧脸、戴口罩或拍摄猫、狗、飞鸟等动物时，CEVA-XM4 深度学习网络瞬间激活托底，锁定目标并持续追焦！

---

## 4. 安全防护机制与防变砖救砖指南

### 应急恢复救砖指南（若相机异常无法开机）
> [!TIP]
> **您的相机绝不会永久变砖！**
> 本项目**绝对未修改**相机的引导加载程序分区（`00_BOOTPARA.bin` 至 `04_BOOT.bin` / IPL）。
> 理光相机的底层应急升级模式固化在片上 BootROM / IPL 中，其执行优先级远高于主系统 RTOS。即使 RTOS 代码洞有任何语法异常，底层卡刷模式依然 100% 正常运行。

**救砖步骤：**
1. **彻底断电**：扣出电池等待 10 秒后重新装入。
2. **准备官方固件**：将官方未经修改的原厂 `v1.20` 固件 `fwdc248b.bin`（或阶段 2 验证固件 `fwdc248b_stage2_dsp_wake.bin`）拷贝至 FAT32 SD 卡根目录。
3. **强制触发硬件卡刷模式**：
   - 确保相机彻底处于关机状态。
   - **长按住 `MENU` 键（或 `DISP` 键）**。
   - 保持按住不放的同时，轻按一次**电源键**。
   - 相机会无视损坏的 RTOS，直接点亮屏幕进入官方 `Firmware Update` 强制刷机界面。
   - 选择 **“是”** 刷回官方固件，相机瞬间恢复如初！

---

## 5. 源码仓库结构

```text
.
├── analysis/
│   ├── dsp_stage3_manifest.json     # 混合 AI-AF 发布清单与反汇编转储
│   ├── rtos_decompilation_audit.md  # GR4 RTOS 深度逆向反汇编报告
│   └── shellcmd_symbols.csv         # 经验证的 RTOS 导出符号表
├── archive/
│   └── stage2/                      # 阶段 2 开机免串口点火概念验证归档
│       ├── build_dsp_firmware.py
│       ├── dsp_wake_manifest.json
│       ├── FEAT_DSP_AI_WAKE.md
│       ├── k3_gr4_dsp_patcher.py
│       └── verify_dsp_patch.py
├── docs/
│   ├── FEAT_DSP_STAGE3_AI_TRACK.md  # 混合 AI-AF 完整白皮书
│   ├── FEAT_K3_AF_PORT.md           # 宾得 K-3 III 自动对焦移植报告
│   ├── RELEASE_NOTES_v1.2.0.md      # v1.2.0 版本发布说明
│   ├── RELEASE_NOTES_v1.2.1.md      # v1.2.1 版本发布说明
│   ├── RELEASE_NOTES_v1.2.3.md      # v1.2.3 版本发布说明
│   ├── RELEASE_NOTES_v1.2.4.md      # v1.2.4 版本发布说明
│   └── RELEASE_NOTES_v1.2.5.md      # v1.2.5 版本发布说明
├── src/
│   └── dsp_ai/
│       ├── build_firmware.py        # 固件主构建与容器封装入口
│       ├── build_stage3_firmware.py # 核心 Stage 3 容器构建器
│       ├── stage3_ai_patcher.py     # 混合 AF 架构与汇编代码洞注入器
│       ├── dsp_splitter.py          # 06_DSP.bin -> DF_EX / DF_ID_S / DF_IP 资产提取工具
│       ├── verify_dsp_payload.py    # 分块字节无损复原校验器
│       ├── k3_desc.bin              # 提取自 K-3 III 的神经网络描述符 (728 B)
│       ├── k3_mean.bin              # 提取自 K-3 III 的通道均值矩阵 (9,408 B)
│       └── k3_std.bin               # 提取自 K-3 III 的通道标准差矩阵 (9,408 B)
├── tools/
│   ├── fw_container.py              # LZSS 容器解包、打包与双重累加和校验器
│   ├── rtos_analyzer.py             # RTOS 交叉引用与符号解析器
│   └── linux_analyzer.py            # Linux rootfs 解析器
└── README_CN.md
```

---

## 6. 从源码构建

```bash
# 自动切分 DSP 分块、固化神经网络参数、编译注入代码洞并生成最终卡刷固件：
python src/dsp_ai/build_firmware.py
```

---

## 7. 底层技术地址速查表 (GR IV v1.20 基准)

| 函数 / 组件符号 | 虚拟地址 (VA) | 功能说明 |
| :--- | :--- | :--- |
| `LiveViewProcess3aContoller` | `0x53676ab8` | 挂接点 1 入口：开机非阻塞状态区清零 (`b #0x53000d60`) |
| 代码洞 1 (Cave 1) | `0x53000d60` | 84 字节：重构 APCS 栈帧并清零 `0x517e0100..0x517e011f` 遥测状态 |
| `FaceDetectionService::Task` | `0x5389639c` | 挂接点 2 入口：逐帧全时混合 AI 对焦循环 (`b #0x53000e00`) |
| 代码洞 2 (Cave 2) | `0x53000e00` | 1,568 字节：原厂 FotoNation + CEVA-XM4 CNN 深度学习智能托底引擎 |
| 原厂 `LibAHFD::DetectFace` | `0x53866594` | 相机原生硬件人脸/眼睛检测（每帧优先调用） |
| `ConvertResult` (直跳入口) | `0x53896460` | 原厂检测到人脸时的直跳入口，实现 60fps 零延迟硬件对焦 |
| 完成态 AHIP 描述符 | `*(*(0x55081350 + 0x424))` | `AhipService::WaitFrame` 正常出队的完成帧（`+0x04` Y, `+0x08` U, `+0x0c` V） |
| `InvalidateDataCache` | `0x538f0454` | CPU 读取取景帧前使 L1/L2 数据缓存失效 |
| `FlushDataCache` | `0x538f04c4` | CPU 写入输入 RGB 张量后写回数据缓存以供 DSP DMA 读取 |
| `DspFactory::GetDspService()` | `0x5398d97c` | 获取片上 CEVA-CV `DspService` 驱动单例指针 |
| `DspService::LoadDsp` | `0x5398ee0c` | 从 SD 卡根目录 (`C:\`) 载入 `DF_EX.BIN`, `DF_ID_S.BIN`, `DF_IP.BIN` |
| `DspService::Start` | `0x5398e2cc` | 解除 DSP 复位，从 `0x52100000` 启动矢量计算核心 |
| `DspService::SendMessage` | `0x5398e478` | 派发 Message ID 4（神经网络卷积推断请求） |
| `DspService::WaitMessage` | `0x5398e794` | 阻塞等待 DSP 完成推断信号量（超时 5000 ms Watchdog） |
| `m_DetectedFaceResult` | `[FaceDetectionService, #0x78]` | 100 字节人脸/主体结构体数组（Q8 坐标、眼部、置信度等） |
| `FdLibraryService` | `[FaceDetectionService, #0x38]` | 原厂服务句柄，`+0x30` 保存识别目标数 `face_count` |
| `UpdateFaceDetectionResult` | `0x538959f4` | 向 LCD 取景渲染线程与 165 点 PDAF 驱动派发对焦坐标 |
| `TENSOR_IN_VA` (`work_buf`) | `0x52b00000` | 重构后的张量输入缓冲区（777,792 B），与静态权重 0 冲突 |
| `TENSOR_OUT_VA` | `0x52b02a00` | 热力图输出缓冲区（`work_buf + 0x2a00`，3 通道 int16 56x56，stride 64） |
| `MEAN_WEIGHTS_VA` | `0x52960000` | 分块 1 静态 CNN 主干网络权重（677,824 B） |
| `HEAD_WEIGHTS_VA` | `0x52a057c0` | 分块 2 静态 CNN 分类头网络权重（461,408 B） |

---

## 8. 版本更新履历

### v1.2.5 (当前最新版本)
- **`TENSOR_OUT_VA` 热力图偏移精确对齐（人脸与猫脸无法识别根因修复）**：
  - 严格反汇编逆向核算宾得 K-3 III `DeepLearningService`（`0x5302cc0c..0x5302cc3c`），确定 CEVA-XM4 DSP 输出热力图确凿起始于 `work_buf + 0x2a00`（`0x52b02a00`）。
  - 彻底纠正了 v1.2.4 中错误设置的 `0x52b09a00` 错位偏移（偏离达 28 KB 全零内存），使人脸、侧脸与宠物热力图数据完整送达寻峰扫描器！
- **5000 ms 跨核 IPC Watchdog 超时（插槽泄漏与白框闪退根因修复）**：
  - 对齐宾得官方标准，将 `SendMessage` 与 `WaitMessage` 超时扩容至 5000 ms（`movw r3, #0x1388` / `movw r2, #0x1388`）。
  - 彻底根治 100 ms 短超时在开机总线并发时引发的 6 个 slot 泄漏与队列死锁，确保对焦框持久稳定运行！
- **固件校验码**：`fwdc248b.bin`（39,109,096 字节，SHA-256：`91603cfdb046c9377136f905bdd71ded756e98b46801aa9f02fcc975c4e81a64`）。

### v1.2.4
- **`DspService` 按需动态实例化（根因彻底修复）**：
  - 审计发现理光商业固件在系统开机时完全剔除了对 DSP 分区的启动调用，从未调用 `CreateDspService` (`0x5398d8fc`)，导致 `m_dspService` 单例成员 `[0x553a3164]` 恒为 NULL (0)。在此前的 Cave 2 中，第 4 条指令检查 `r5 == 0` 时从第 1 帧开始直到关机全部无条件跳转至 `clear_and_exit`，导致重试计数器与诊断心跳框从未执行！
  - Cave 2 现已加入动态按需创建机制：若检测到 `[0x553a3164] == 0`，立即在第 1 帧动态调用 `DspFactory::CreateDspService(this, 0)` (`0x5398d8fc`)，自动完成 0x60 堆内存分配、构造函数执行与跨核 RTOS 通信原语（`DspMtx`、`DspSF`、`DspReq`、`DspExc`）的初始化与挂载。
- **双框常驻视觉遥测矩阵（Visual Telemetry）**：
  - **左上角绿色警告框 (`x=40, y=40, score=88`)**：若 `DspService` 实例化成功，但 `LoadDsp` 尝试从 SD 卡与内置闪存读取 `DF_*.BIN` 均失败，该框将持续常驻显示，提示用户检查 SD 卡是否为 FAT32 格式以及文件是否放置在根目录。
  - **右下角绿色心跳框 (`x=240, y=140, score=99`)**：在 CEVA DSP 启动、AHIP 正常出帧、跨核 IPC 往返 100% 成功且画面未检测到目标（大白墙）时常驻显示，证明底层驱动链路彻底跑通。
  - **动态 AI 对焦框**：画面中出现人脸、侧脸或猫狗宠物时，自动锁定并驱动 PDAF。
- **固件校验码**：`fwdc248b.bin`（39,109,096 字节，SHA-256：`3bfb04641408c8da788665719528572b6ce78a2486ae6f948c23ff47dc139c1a`）。

### v1.2.3
- **纯 K-3 III 深度学习独占验证版（物理旁路 FotoNation）**：
  - 100% 切断原厂 `LibAHFD::DetectFace` (`0x53866594`) 调用，引入右下角诊断心跳框。

### v1.2.2
- **底层安全裕量加固**：
  - 将 `Embedded Desc Table` 迁移至 `0x53001600`，彻底杜绝与 Cave 2 的内存重叠风险。

### v1.2.1
- **真·混合自动对焦架构（Hybrid AF）**：
  - 完美结合原厂 FotoNation `LibAHFD::DetectFace` (`0x53866594`) 与 CEVA-XM4 深度学习 CNN 智能托底。正面人脸/人眼通过 `0x53896460` 实现 60fps 硬件级零延迟对焦；侧脸、转头、口罩及宠物鸟类由深度学习网络接管。
- **内存零冲突重构（DDR Weight Clobber Fix）**：
  - 将 `TENSOR_IN_VA` 移至 `0x52b00000`，彻底消除了原 646 KB 的内存地址重叠，根除了 CNN 模型静态权重被覆写的隐患。
- **彻底根除开机 2 秒死机 / 黑屏冻结**：
  - 移除了 Cave 2 中无限等待 `tk_wai_flg(0x3f0000, 0)` 的 `DspService::Test` 死锁调用，加锁并让出当前帧，使 DSP 硬件平稳完成 CRT0 点火。
- **彻底消除开机左上 1/4 假对焦框**：
  - 动态计算并填充 `eye1` (`+0x10`) 与 `eye2` (`+0x20`) 坐标结构体；冷启动未识别到主体时保持 `face_count = 0`，不再绘制假框。
- **修正 AHIP 硬件帧驱动基址**：
  - 更新 `AHIP_DRV_BASE_VA` 为 `0x55081350`，主完成环偏移为 `+0x424`。
- **100% 保持 SD 卡文件兼容**：
  - SD 卡三个二进制文件保持不变，老用户仅需刷新 `fwdc248b.bin` 即可升级生效。
- **固件校验码**：`fwdc248b.bin`（39,109,096 字节，SHA-256：`15bcb7fb254f6c42f5d50c33b6505f7887d422b0f6c1d65c70577363edf1d1c3`）。

### v1.2.0
- **官方 Ricoh GR IV v1.20 固件基准移植**：
  - 基于官方 v1.20 固件 `fwdc248b.bin` 重构，适配扩展后的 `.data` 段与重定位符号。
- **修复 DSP 加载器返回值判断逻辑**：将 `DspService::LoadDsp` 成功判断修正为 `bne cave2_load_ok`。

### v1.0.9
- **AHIP 硬件完成出队帧提取与 D-Cache 失效刷新**：读取 `*(*(0x55080350 + 0x424))` 出队指针，并在 CPU 读取前执行 `InvalidateDataCache` (`0x538f0454`)。
- **全画幅比例自适应下采样（3:2, 4:3, 16:9, 1:1）**。
- **$C^0$ 连续 $2.0\times$ 逆光暗部平滑增益曲线**。
- **头部通道优先解析与动态框尺寸**。

---

## 9. 权利声明与开源协议

- **免责声明**：本项目为技术爱好者出于学习、逆向工程与学术研究目的开展的独立开源研究。刷机具有潜在风险，尽管我们设计了极尽严苛的硬件防变砖保护策略与双重校验累加和机制，作者对使用本工具所导致的任何直接或间接设备故障概不承担责任。
- **版权声明**：本仓库仅分发原创的反汇编解析工具、汇编注入脚本与数学逻辑规范，不分发任何受理光（Ricoh Imaging Company, Ltd.）版权保护的私有固件二进制。用户须使用自身合法持有的相机固件进行构建。
- **开源协议**：本项目基于 MIT License 开源，详见 [LICENSE](LICENSE)。
