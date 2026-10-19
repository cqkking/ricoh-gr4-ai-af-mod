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

## 3. 阶段 2 实装：免串口开机全自动唤醒 (Automated Boot-time Wakeup)

由于零售版机身通常**无法引出调试串口**，我们已全面实装**阶段 2 自动唤醒补丁**，将 DSP 加载、解除复位启动与内核自检固化进固件运行管线。

### 3.1 运行执行模型与蹦床架构

```text
相机开机 / 启动实时取景 (Live View)
       │
       ▼
[0x536520f0] LiveViewProcess3aContoller 钩子蹦床 (b #0x53000e00)
       │
       ▼
[0x53000e00] RTOS 代码洞 (Code Cave @ 0x53000e00)
       │  1. 执行覆盖的原指令 (mov ip, sp; push {r4..pc})
       │  2. 保存通用暂存寄存器 (push {r0-r3, lr})
       │  3. 调用 DspFactory::GetDspService() 获取驱动句柄
       │  4. 检查 DSP 运行状态:
       │     └─ 若 bit 0 置位 (已启动): 直接跳至出口 (保证单次执行，防重复载入)
       │  5. 调用 DspService::LoadDsp(svc, 'A', 0, 0) 从 SD 卡 (A:\) 载入三文件
       │  6. 检查载入状态:
       │     └─ 若 bit 1 未置位 (无 SD 卡或文件不存在): 干净跳出，不影响相机正常拍摄
       │  7. 调用 DspService::Start(svc, 0, 0) 解除 CEVA-CV 复位锁定并点火运行
       │  8. 调用 DspService::Test(svc, 0) 触发 CEVA 版本握手自检
       │  9. 恢复暂存寄存器 (pop {r0-r3, lr})
       │
       ▼
返回原流程 [0x536520f8] 继续执行 3A/Live View 预览循环 (平滑无感)
```

### 3.2 刷机与使用步骤（无需任何串口接线）

1. **准备 SD 卡依赖文件**：
   * 确保将 [`output/sd_card/`](file:///d:/Ricoh%20GR4%20firmware%20Mod/worktrees/feat-dsp-ai-af/output/sd_card) 下的三个文件复制到 SD 卡根目录：
     * `DF_EX.BIN` (4,112,408 B)
     * `DF_ID_S.BIN` (65,536 B)
     * `DF_IP.BIN` (16,384 B)
2. **刷入修补后固件**：
   * 将自动生成的 Flashable 固件 [`output/fwdc248b.bin`](file:///d:/Ricoh%20GR4%20firmware%20Mod/worktrees/feat-dsp-ai-af/output/fwdc248b.bin)（38,693,256 字节，双重字累加和 `word_sum == 0` 校验已通过）复制到 SD 卡根目录。
   * 相机按住 `DISP` 或 `MENU` 键开机，进入官方固件升级菜单确认升级。
3. **点火生效**：
   * 升级完成后开机，在进入实时取景界面的瞬间，固件蹦床即自动完成 CEVA-CV 载入与握手启动。即使拔出存储卡，相机原版反差/片上相位对焦依然完全正常工作。

---

## 4. 后续阶段：挂接 Live View 深度学习主体识别

在 DSP 正常启动后，推进下一子阶段：
1. 提取 K-3 III 的 [`DeepLearningService`](file:///d:/Ricoh%20GR4%20firmware%20Mod/firmware/reference/k3iii_v220/unpacked/sections/05_RTOS.bin)（~10.8 KB）与 `DeepLearningPipeService` 逻辑。
2. 在 Live View 帧缓冲区中挂接 56×56 RGB 下采样。
3. 提取 DSP 生成的 64×64 人体/主体识别热力图（`m_humanDetectionMap`），注入 GR4 原生 `LiveViewProcessTrackObject`（`0x536551b0`），实现类似索尼/佳能的深度学习实时追踪对焦。
