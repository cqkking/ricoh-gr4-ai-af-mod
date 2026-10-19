# CEVA-CV AI DSP 协处理器唤醒与深度学习追焦 (Option A)

## 1. 架构总览与芯片硬件真相

Ricoh GR IV 与 Pentax K-3 Mark III 均搭载 **Socionext Milbeaut M10V** 异构多核图像处理器。
通过对量产版固件的底层反汇编与内存映射分析，证实了一个此前未被公开的重大事实：

> **Ricoh GR IV 片上硬件完全集成了 CEVA-CV 矢量 AI DSP 协处理器（基地址 `0x41000000`）**。
> GR4 的 RTOS 系统在开机初始化时（`PFFactory::DeviceInitialize` @ `0x5376e4e4`）已经完整创建了 DSP 核心驱动 `DspService`（`0x539658c4`）及底层通信信箱（`DspExc`, `DspReq`, `DspSF`, `DspMtx`），并内置了完整的 DSP 控制调试指令（`DSP_load`, `DSP_start`, `DSP_test`）。
> 但由于量产固件砍掉了 DSP 固件分区，这颗具备实时卷积神经网络推断能力的硬件核在 GR4 上处于 **100% 闲置休眠状态**。

---

## 2. 内存映射与三文件加载协议

通过逆向 GR4 RTOS 的 `DspService::LoadDsp`（`0x53966d74`）及底层辅助例程（`0x53965ff8` 与 `0x53965e6c`），我们完全破译了理光官方用于加载 CEVA-CV DSP 固件的二进制协议：

### 2.1 内存分配与物理映射

| 存储域 / 模块 | 物理基地址 | 虚拟地址 (VA) | 长度 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **CEVA MMU 物理窗口** | `0x41000000` | `0x41000000` | 8 MB (`0x00800000`) | MMU L1 页表映射属性：`0x41004c22`（强顺序共享普通内存） |
| **`DF_EX.BIN`** (执行区) | `0x51200000` | `0x51200000` | 4,112,384 B (`0x3ec000`) | CEVA-CV 矢量算子库、运行时调度器与基础通信栈 |
| **`DF_ID_S.BIN`** (常量数据) | `0x515ec000` | `0x515ec000` | 65,536 B (`0x10000`) | 深度学习 CNN 权重、网络层拓扑与共享向量参数 |
| **`DF_IP.BIN`** (指令入口) | `0x515fc000` | `0x515fc000` | 16,384 B (`0x4000`) | DSP 复位中断向量表与引导入口例程 |

这三段内存相加：
$$0x3EC000 + 0x10000 + 0x4000 = 0x400000\text{ (恰好为 } 4,194,304\text{ 字节 / 4.00 MB)}$$
与 Pentax K-3 Mark III 解包得到的 [`06_DSP.bin`](file:///d:/Ricoh%20GR4%20firmware%20Mod/firmware/reference/k3iii_v220/unpacked/sections/06_DSP.bin) **分毫不差，100% 精确吻合**。

### 2.2 `DF_EX.BIN` 专有块格式解析

GR4 RTOS `0x53965ff8` 解析器要求的 `DF_EX.BIN` 二进制布局：
1. **文件头 (16 字节)**：
   * `uint32_t num_chunks;`（必须为分块数量，此处为 1）
   * `uint32_t reserved[3];`（全 0）
2. **块描述符 (8 字节)**：
   * `uint32_t dest_va;`（目标加载虚拟地址，硬编码校验必须等于 `0x51200000`）
   * `uint32_t chunk_length;`（分块数据长度，此处为 `0x003ec000`）
3. **有效载荷 (4,112,384 字节)**：紧随其后的原始 CEVA 二进制代码。

---

## 3. 实机验证与操作指南

### 3.1 SD 卡免刷机热加载验证（最安全）

利用理光 RTOS 内置的驱动机制，无需刷写或修改机身闪存即可快速点亮验证：

1. 运行提取脚本生成配套文件：
   ```bash
   python src/dsp_ai/dsp_splitter.py
   ```
2. 将输出目录 [`output/sd_card/`](file:///d:/Ricoh%20GR4%20firmware%20Mod/worktrees/feat-dsp-ai-af/output/sd_card) 下生成的三个文件：
   * `DF_EX.BIN`
   * `DF_ID_S.BIN`
   * `DF_IP.BIN`
   直接复制到 SD 卡根目录（RTOS 挂载为 `A:\`）。
3. 通过串口终端或 Shell 注入执行：
   ```text
   DSP_load A:
   DSP_start
   DSP_test 0
   ```
   * `DSP_test 0`：验证 CEVA 版本响应与握手回执。
   * `DSP_test 1`：验证 CEVA-CV RGB->YUV 矢量转换流水线。
   * `DSP_test 2`：验证 2D 卷积滤波（Filter2D）。

### 3.2 固件集成固化（开机自启动）

在固件容器层面将 DSP 代码常驻并挂接自动启动：
1. 通过 [`build_dsp_firmware.py`](file:///d:/Ricoh%20GR4%20firmware%20Mod/worktrees/feat-dsp-ai-af/src/dsp_ai/build_dsp_firmware.py) 将 Section 06 注册为 DSP 分区并重新打包固件容器。
2. 在 `DeviceInitialize`（`0x5376e4e4`）末端蹦床跳板自动调用 `DspService::LoadDsp` 与 `Start`。

---

## 4. 后续阶段：挂接 Live View 深度学习主体识别

在 DSP 正常启动后，推进下一子阶段：
1. 提取 K-3 III 的 [`DeepLearningService`](file:///d:/Ricoh%20GR4%20firmware%20Mod/firmware/reference/k3iii_v220/unpacked/sections/05_RTOS.bin)（~10.8 KB）与 `DeepLearningPipeService` 逻辑。
2. 在 Live View 帧缓冲区中挂接 56×56 RGB 下采样。
3. 提取 DSP 生成的 64×64 人体/主体识别热力图（`m_humanDetectionMap`），注入 GR4 原生 `LiveViewProcessTrackObject`（`0x536551b0`），实现类似索尼/佳能的深度学习实时追踪对焦。
