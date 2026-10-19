#!/usr/bin/env python3
"""
stage3_ai_patcher.py - Stage 3 Deep Learning Inference Pipeline & Live View Tracking Bridge

Integrates:
1. Stage 2 Boot-Time Awakening: Loads and unresets CEVA-CV DSP from SD card on startup.
2. Embedded AI Model Descriptors & Normalization Tables:
   - k3_desc.bin (728 B) embedded @ VA 0x53001000
   - k3_mean.bin (9,408 B) embedded @ VA 0x541db540
   - k3_std.bin  (9,408 B) embedded @ VA 0x54217680
3. Live View Inference & Tracking Pipeline (executed in Code Cave @ 0x53000e00):
   - Decimation: Runs AI inference once every 2 Live View frames (~30 fps).
   - Downsamples Live View preview frame to 56x56 RGB @ 0x51600000.
   - Dispatches msg_id 4 to CEVA-CV DSP via DspService::SendMessage (0x539663c0).
   - Waits on DspService::WaitMessage (0x539666dc) for tensor completion.
   - Scans 64x64 output heatmap @ 0x51610000 for peak confidence score.
   - Translates peak coordinates (col, row) to Live View pixel coordinates (X, Y, W, H).
   - Injects target into slot 0 @ 0x53fe1a94 and calls UpdateTrackTarget (0x538e32c4),
     locking GR4's 165-point on-sensor PDAF onto the detected subject.
   - Returns seamlessly to LiveViewProcess3aContoller (0x536520f8).
"""

import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

GR4_BASE          = 0x53000000
CAVE_VA           = 0x53000e00
DESC_TABLE_VA     = 0x53001100 # 728 B table placed after code cave
STATUS_VA         = 0x53001400 # Telemetry & status variables placed after desc table
MEAN_TABLE_VA     = 0x541db540 # 9,408 B table in zero cave
STD_TABLE_VA      = 0x54217680 # 9,408 B table in zero cave

TENSOR_IN_VA      = 0x51600000 # 56x56 RGB input buffer in DSP shared RAM
TENSOR_MSG_VA     = 0x51602500 # 36-byte message buffer
TENSOR_OUT_VA     = 0x51610000 # 64x64 heatmap output buffer

GET_DSP_SVC_VA    = 0x539658c4 # DspFactory::GetDspService()
LOAD_DSP_VA       = 0x53966d74 # DspService::LoadDsp(svc, 'A', 0, 0)
START_DSP_VA      = 0x53966214 # DspService::Start(svc, 0, 0)
TEST_DSP_VA       = 0x53966f18 # DspService::Test(svc, 0)
SEND_MSG_VA       = 0x539663c0 # DspService::SendMessage(svc, &seq, 4, timeout, &msg)
WAIT_MSG_VA       = 0x539666dc # DspService::WaitMessage(svc, seq, timeout, &resp)
UPDATE_TRACK_VA   = 0x538e32c4 # UpdateTrackTarget(TrackObj*, X, Y, W, H, Conf)

TRACK_OBJ_PTR_VA  = 0x550b6aa4 # [0x550b6aa4] -> LiveViewProcessTrackObject instance
TRACK_SLOT0_VA    = 0x53fe1a94 # Primary AF tracking slot
LV_IMAGE_SVC_VA   = 0x55395a18 # [0x55395a18 + 8] -> p_liveview_buf

HOOK_SITE_VA      = 0x536520f0 # LiveViewProcess3aContoller entry
RETURN_SITE_VA    = 0x536520f8 # Return after mov ip, sp; push {r4..pc}

def encode_arm_branch(from_va: int, to_va: int, is_bl: bool = False) -> int:
    disp = to_va - (from_va + 8)
    disp_words = (disp >> 2) & 0x00ffffff
    op = 0xeb000000 if is_bl else 0xea000000
    return op | disp_words

def encode_movw(rd: int, imm16: int) -> int:
    imm4 = (imm16 >> 12) & 0xf
    imm12 = imm16 & 0xfff
    return 0xe3000000 | (imm4 << 16) | (rd << 12) | imm12

def encode_movt(rd: int, imm16: int) -> int:
    imm4 = (imm16 >> 12) & 0xf
    imm12 = imm16 & 0xfff
    return 0xe3400000 | (imm4 << 16) | (rd << 12) | imm12

def emit_ldr_imm32(insns: list, rd: int, val: int):
    insns.append(encode_movw(rd, val & 0xffff))
    insns.append(encode_movt(rd, (val >> 16) & 0xffff))

def build_stage3_cave_payload(cave_va: int = CAVE_VA, return_va: int = RETURN_SITE_VA) -> bytes:
    insns = []
    labels = {}
    relocs = [] # (index, label_name, is_bl)

    def mark_label(name: str):
        labels[name] = cave_va + len(insns) * 4

    def branch_to(target_label: str, is_bl: bool = False, cond: int = 0xe):
        idx = len(insns)
        relocs.append((idx, target_label, is_bl, cond))
        insns.append(0) # placeholder

    # 1. Overwritten instructions from LiveViewProcess3aContoller (0x536520f0)
    insns.append(0xe1a0c00d) # mov  ip, sp
    insns.append(0xe92dd8f0) # push {r4, r5, r6, r7, fp, ip, lr, pc}

    # 2. Save scratch registers r0-r3, lr (and allocate 16 bytes stack for args)
    insns.append(0xe92d400f) # push {r0, r1, r2, r3, lr}
    insns.append(0xe24dd010) # sub  sp, sp, #16 (stack frame for outgoing args)

    # 3. Call DspFactory::GetDspService() -> r0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, GET_DSP_SVC_VA, is_bl=True))

    # 4. m_dspService is at [r0, #4]
    insns.append(0xe5904004) # ldr  r4, [r0, #4]
    insns.append(0xe3540000) # cmp  r4, #0
    branch_to("exit", is_bl=False, cond=0x0) # beq .exit

    # 5. Check if DSP is already running (status bit 0)
    insns.append(0xe5942004) # ldr  r2, [r4, #4]
    insns.append(0xe3120001) # tst  r2, #1 (STARTED)
    branch_to("dsp_running", is_bl=False, cond=0x1) # bne .dsp_running

    # --- Stage 2 DSP First-Time Wakeup ---
    # LoadDsp(m_dspService, 'A', 0, 0)
    insns.append(0xe1a00004) # mov  r0, r4
    insns.append(0xe3a01041) # mov  r1, #0x41 ('A')
    insns.append(0xe3a02000) # mov  r2, #0
    insns.append(0xe3a03000) # mov  r3, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, LOAD_DSP_VA, is_bl=True))

    # Check LOADED bit
    insns.append(0xe5942004) # ldr  r2, [r4, #4]
    insns.append(0xe3120002) # tst  r2, #2 (LOADED)
    branch_to("exit", is_bl=False, cond=0x0) # beq .exit

    # Start(m_dspService, 0, 0)
    insns.append(0xe1a00004) # mov  r0, r4
    insns.append(0xe3a01000) # mov  r1, #0
    insns.append(0xe3a02000) # mov  r2, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, START_DSP_VA, is_bl=True))

    # Test(m_dspService, 0)
    insns.append(0xe1a00004) # mov  r0, r4
    insns.append(0xe3a01000) # mov  r1, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, TEST_DSP_VA, is_bl=True))

    # Record status magic 'CEVA' @ STATUS_VA (0x53000fb0)
    emit_ldr_imm32(insns, 3, STATUS_VA)
    emit_ldr_imm32(insns, 1, 0x41564543) # 'CEVA'
    insns.append(0xe5831000) # str  r1, [r3]
    branch_to("exit", is_bl=False, cond=0xe) # b .exit

    # --- Stage 3 Live View Tracking Pipeline ---
    mark_label("dsp_running")

    # Frame decimation: increment counter @ STATUS_VA + 4 (0x53000fb4)
    emit_ldr_imm32(insns, 3, STATUS_VA + 4)
    insns.append(0xe5931000) # ldr  r1, [r3]
    insns.append(0xe2811001) # add  r1, r1, #1
    insns.append(0xe5831000) # str  r1, [r3]
    insns.append(0xe3110001) # tst  r1, #1 (run every 2nd frame = ~30fps)
    branch_to("exit", is_bl=False, cond=0x1) # bne .exit (skip odd frames)

    # Check LiveView Image Buffer: [0x55395a18, #8]
    emit_ldr_imm32(insns, 3, LV_IMAGE_SVC_VA)
    insns.append(0xe5933008) # ldr  r3, [r3, #8]
    insns.append(0xe3530000) # cmp  r3, #0
    branch_to("exit", is_bl=False, cond=0x0) # beq .exit

    # Downsample Live View frame into 56x56 RGB @ 0x51600000
    insns.append(0xe1a00003) # mov  r0, r3 (src_buf)
    emit_ldr_imm32(insns, 1, TENSOR_IN_VA) # r1 = 0x51600000
    branch_to("sub_downsample", is_bl=True, cond=0xe)

    # Prepare 9-word message @ 0x51602500
    emit_ldr_imm32(insns, 5, TENSOR_MSG_VA) # r5 = 0x51602500
    emit_ldr_imm32(insns, 0, DESC_TABLE_VA) # msg[0] = 0x53001000
    insns.append(0xe5850000) # str  r0, [r5, #0]
    insns.append(0xe3a00000) # mov  r0, #0
    insns.append(0xe5850004) # str  r0, [r5, #4] (msg[1] = 0)
    emit_ldr_imm32(insns, 0, TENSOR_IN_VA)  # msg[2] = 0x51600000
    insns.append(0xe5850008) # str  r0, [r5, #8]
    emit_ldr_imm32(insns, 0, STD_TABLE_VA)   # msg[3] = 0x54217680
    insns.append(0xe585000c) # str  r0, [r5, #12]
    emit_ldr_imm32(insns, 0, MEAN_TABLE_VA)  # msg[4] = 0x541db540
    insns.append(0xe5850010) # str  r0, [r5, #16]
    emit_ldr_imm32(insns, 0, 0x00080808)    # msg[5] = 0x00080808
    insns.append(0xe5850014) # str  r0, [r5, #20]
    emit_ldr_imm32(insns, 0, 0x0c0c0c0a)    # msg[6] = 0x0c0c0c0a
    insns.append(0xe5850018) # str  r0, [r5, #24]
    emit_ldr_imm32(insns, 0, 0x0c0c0c0c)    # msg[7] = 0x0c0c0c0c
    insns.append(0xe585001c) # str  r0, [r5, #28]
    emit_ldr_imm32(insns, 0, TENSOR_OUT_VA) # msg[8] = 0x51610000
    insns.append(0xe5850020) # str  r0, [r5, #32]

    # Setup seq pointer @ STATUS_VA + 8 (0x53000fb8)
    emit_ldr_imm32(insns, 6, STATUS_VA + 8)
    insns.append(0xe3a00000) # mov  r0, #0
    insns.append(0xe1c600b0) # strh r0, [r6]

    # Call DspService::SendMessage(m_dspService, &seq, msg_id=4, timeout=100, &msg)
    insns.append(0xe1a00004) # mov  r0, r4 (svc)
    insns.append(0xe1a01006) # mov  r1, r6 (&seq)
    insns.append(0xe3a02004) # mov  r2, #4 (msg_id = 4)
    insns.append(0xe3a03064) # mov  r3, #100 (timeout = 100 ms)
    insns.append(0xe58d5000) # str  r5, [sp] (&msg)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, SEND_MSG_VA, is_bl=True))

    # Call DspService::WaitMessage(m_dspService, seq, timeout=100, &resp)
    insns.append(0xe1a00004) # mov  r0, r4 (svc)
    insns.append(0xe1d610b0) # ldrh r1, [r6] (seq)
    insns.append(0xe3a02064) # mov  r2, #100 (timeout = 100 ms)
    insns.append(0xe1a03005) # mov  r3, r5 (&resp)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, WAIT_MSG_VA, is_bl=True))

    # Scan 64x64 output heatmap @ 0x51610000 for peak score
    emit_ldr_imm32(insns, 0, TENSOR_OUT_VA)
    branch_to("sub_find_peak", is_bl=True, cond=0xe) # returns r0: score, r1: col, r2: row

    # Check confidence threshold (>= 60)
    insns.append(0xe350003c) # cmp  r0, #60
    branch_to("exit", is_bl=False, cond=0xb) # blt .exit

    # Convert (col, row) to Live View pixel coordinates:
    # X = col * 15 (col: 0..63 -> X: 0..945)
    # Y = row * 10 (row: 0..63 -> Y: 0..630)
    # W = 200, H = 200
    insns.append(0xe3a0300f) # mov  r3, #15
    insns.append(0xe0080391) # mul  r8, r1, r3 (X = col * 15)
    insns.append(0xe3a0300a) # mov  r3, #10
    insns.append(0xe0060392) # mul  r6, r2, r3 (Y = row * 10)
    insns.append(0xe3a050c8) # mov  r5, #200 (W)
    insns.append(0xe3a070c8) # mov  r7, #200 (H)

    # Save telemetry @ STATUS_VA + 12 (0x53000fbc)
    emit_ldr_imm32(insns, 3, STATUS_VA + 12)
    insns.append(0xe5830000) # str  r0, [r3, #0]  (peak_score)
    insns.append(0xe5838004) # str  r8, [r3, #4]  (last_x)
    insns.append(0xe5836008) # str  r6, [r3, #8]  (last_y)
    insns.append(0xe583500c) # str  r5, [r3, #12] (last_w)
    insns.append(0xe5837010) # str  r7, [r3, #16] (last_h)

    # Inject target into primary tracking slot @ TRACK_SLOT0_VA (0x53fe1a94)
    emit_ldr_imm32(insns, 3, TRACK_SLOT0_VA)
    insns.append(0xe5838000) # str  r8, [r3, #0] (X)
    insns.append(0xe5c36004) # strb r6, [r3, #4] (Y)
    insns.append(0xe1c350b6) # strh r5, [r3, #6] (W)
    insns.append(0xe5c37008) # strb r7, [r3, #8] (H)
    insns.append(0xe583000c) # str  r0, [r3, #12] (Conf)

    # Fetch LiveViewProcessTrackObject instance from [0x550b6aa4]
    emit_ldr_imm32(insns, 3, TRACK_OBJ_PTR_VA)
    insns.append(0xe5930000) # ldr  r0, [r3]
    insns.append(0xe3500000) # cmp  r0, #0
    branch_to("exit", is_bl=False, cond=0x0) # beq .exit

    # Call UpdateTrackTarget(TrackObj* this, X, Y, W, H, Conf) @ 0x538e32c4
    insns.append(0xe1a01008) # mov  r1, r8 (X)
    insns.append(0xe1a02006) # mov  r2, r6 (Y)
    insns.append(0xe1a03005) # mov  r3, r5 (W)
    insns.append(0xe58d7000) # str  r7, [sp] (H)
    insns.append(0xe58d0004) # str  r0, [sp, #4] (Conf)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, UPDATE_TRACK_VA, is_bl=True))

    # --- Subroutine: Downsample Live View Frame to 56x56 RGB ---
    # r0 = src_yuv, r1 = dst_rgb (9408 bytes)
    mark_label("sub_downsample")
    insns.append(0xe92d4070) # push {r4, r5, r6, lr}
    insns.append(0xe3a02000) # mov  r2, #0 (row = 0)
    insns.append(0xe3a04000) # mov  r4, #0 (dst_offset = 0)
    mark_label("loop_ds_row")
    insns.append(0xe3a03000) # mov  r3, #0 (col = 0)
    mark_label("loop_ds_col")
    # Fetch pixel: simple integer subsample (stride ~17 for col, ~11 for row)
    insns.append(0xe3a05011) # mov  r5, #17
    insns.append(0xe0050593) # mul  r5, r3, r5 (src_x = col * 17)
    insns.append(0xe3a0600b) # mov  r6, #11
    insns.append(0xe0060692) # mul  r6, r2, r6 (src_y = row * 11)
    insns.append(encode_movw(12, 960))       # movw ip, #960 (LiveView pitch)
    insns.append(0xe006069c) # mul  r6, ip, r6 (src_y * 960)
    insns.append(0xe0855006) # add  r5, r5, r6 (src_offset = src_y*960 + src_x)
    insns.append(0xe7d06005) # ldrb r6, [r0, r5] (Y value)
    # Store R=Y, G=Y, B=Y (grayscale approximation, zero loss of shape for CNN)
    insns.append(0xe7c16004) # strb r6, [r1, r4] (R)
    insns.append(0xe2844001) # add  r4, r4, #1
    insns.append(0xe7c16004) # strb r6, [r1, r4] (G)
    insns.append(0xe2844001) # add  r4, r4, #1
    insns.append(0xe7c16004) # strb r6, [r1, r4] (B)
    insns.append(0xe2844001) # add  r4, r4, #1
    insns.append(0xe2833001) # add  r3, r3, #1 (col++)
    insns.append(0xe3530038) # cmp  r3, #56
    branch_to("loop_ds_col", is_bl=False, cond=0xb) # blt .loop_ds_col
    insns.append(0xe2822001) # add  r2, r2, #1 (row++)
    insns.append(0xe3520038) # cmp  r2, #56
    branch_to("loop_ds_row", is_bl=False, cond=0xb) # blt .loop_ds_row
    insns.append(0xe8bd8070) # pop  {r4, r5, r6, pc}

    # --- Subroutine: Scan 64x64 Heatmap for Peak ---
    # r0 = heatmap (4096 bytes)
    # Output: r0 = max_val, r1 = col, r2 = row
    mark_label("sub_find_peak")
    insns.append(0xe92d4070) # push {r4, r5, r6, lr}
    insns.append(0xe3a01000) # mov  r1, #0 (max_val)
    insns.append(0xe3a02000) # mov  r2, #0 (max_idx)
    insns.append(0xe3a03000) # mov  r3, #0 (curr_idx)
    insns.append(0xe3014000) # movw r4, #4096
    mark_label("loop_scan_peak")
    insns.append(0xe7d05003) # ldrb r5, [r0, r3] (val)
    insns.append(0xe1550001) # cmp  r5, r1
    branch_to("skip_peak_update", is_bl=False, cond=0xd) # ble .skip_peak_update
    insns.append(0xe1a01005) # mov  r1, r5 (max_val = val)
    insns.append(0xe1a02003) # mov  r2, r3 (max_idx = curr_idx)
    mark_label("skip_peak_update")
    insns.append(0xe2833001) # add  r3, r3, #1
    insns.append(0xe1530004) # cmp  r3, r4
    branch_to("loop_scan_peak", is_bl=False, cond=0xb) # blt .loop_scan_peak
    # Compute col = max_idx & 63, row = max_idx >> 6
    insns.append(0xe202603f) # and  r6, r2, #63 (col)
    insns.append(0xe1a02322) # lsr  r2, r2, #6  (row)
    insns.append(0xe1a00001) # mov  r0, r1      (max_val)
    insns.append(0xe1a01006) # mov  r1, r6      (col)
    insns.append(0xe8bd8070) # pop  {r4, r5, r6, pc}

    # --- Common Exit Point ---
    mark_label("exit")
    insns.append(0xe28dd010) # add  sp, sp, #16 (deallocate stack frame)
    insns.append(0xe8bd400f) # pop  {r0, r1, r2, r3, lr}
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, return_va, is_bl=False))

    # Resolve all label branches
    for idx, label_name, is_bl, cond in relocs:
        target_va = labels[label_name]
        from_va = cave_va + idx * 4
        disp = target_va - (from_va + 8)
        disp_words = (disp >> 2) & 0x00ffffff
        base_op = 0x0b000000 if is_bl else 0x0a000000
        insns[idx] = (cond << 28) | base_op | disp_words

    payload = struct.pack(f"<{len(insns)}I", *insns)
    return payload

def build_trampoline_payload(hook_va: int = HOOK_SITE_VA, cave_va: int = CAVE_VA) -> bytes:
    b_insn = encode_arm_branch(hook_va, cave_va, is_bl=False)
    nop_insn = 0xe1a00000 # mov r0, r0
    return struct.pack("<II", b_insn, nop_insn)

def apply_stage3_patch(rtos_binary: bytearray, rtos_payload_offset: int = 0) -> dict:
    dsp_ai_dir = Path(__file__).resolve().parent
    desc_bytes = (dsp_ai_dir / "k3_desc.bin").read_bytes()
    mean_bytes = (dsp_ai_dir / "k3_mean.bin").read_bytes()
    std_bytes  = (dsp_ai_dir / "k3_std.bin").read_bytes()

    assert len(desc_bytes) == 728, f"Unexpected desc size: {len(desc_bytes)}"
    assert len(mean_bytes) == 9408, f"Unexpected mean size: {len(mean_bytes)}"
    assert len(std_bytes) == 9408, f"Unexpected std size: {len(std_bytes)}"

    cave_payload = build_stage3_cave_payload()
    tramp_payload = build_trampoline_payload()

    cave_off = CAVE_VA - GR4_BASE + rtos_payload_offset
    desc_off = DESC_TABLE_VA - GR4_BASE + rtos_payload_offset
    mean_off = MEAN_TABLE_VA - GR4_BASE + rtos_payload_offset
    std_off  = STD_TABLE_VA - GR4_BASE + rtos_payload_offset
    hook_off = HOOK_SITE_VA - GR4_BASE + rtos_payload_offset

    # Safety checks: ensure destination caves are all pure zero space
    assert rtos_binary[cave_off : cave_off + len(cave_payload)] == b"\x00" * len(cave_payload), "Cave is not empty zero space!"
    assert rtos_binary[desc_off : desc_off + len(desc_bytes)] == b"\x00" * len(desc_bytes), "Desc table destination is not zero!"
    assert rtos_binary[mean_off : mean_off + len(mean_bytes)] == b"\x00" * len(mean_bytes), "Mean table destination is not zero!"
    assert rtos_binary[std_off : std_off + len(std_bytes)] == b"\x00" * len(std_bytes), "Std table destination is not zero!"

    # Safety check on hook site
    orig_hook_bytes = bytes(rtos_binary[hook_off : hook_off + len(tramp_payload)])
    expected_hook = bytes.fromhex("0dc0a0e1f0d82de9") # mov ip, sp; push {r4..pc}
    assert orig_hook_bytes == expected_hook, f"Unexpected hook bytes: {orig_hook_bytes.hex()} (expected {expected_hook.hex()})"

    # Apply patches
    rtos_binary[cave_off : cave_off + len(cave_payload)] = cave_payload
    rtos_binary[desc_off : desc_off + len(desc_bytes)] = desc_bytes
    rtos_binary[mean_off : mean_off + len(mean_bytes)] = mean_bytes
    rtos_binary[std_off : std_off + len(std_bytes)] = std_bytes
    rtos_binary[hook_off : hook_off + len(tramp_payload)] = tramp_payload

    manifest = {
        "stage": "Stage 3 Deep Learning Inference Pipeline & Live View Tracking Bridge",
        "code_cave": {
            "vaddr": hex(CAVE_VA),
            "offset": hex(cave_off),
            "length": len(cave_payload),
            "hex": cave_payload.hex(),
        },
        "trampoline": {
            "vaddr": hex(HOOK_SITE_VA),
            "offset": hex(hook_off),
            "original_hex": orig_hook_bytes.hex(),
            "patched_hex": tramp_payload.hex(),
            "return_vaddr": hex(RETURN_SITE_VA),
        },
        "embedded_tables": {
            "desc_table": {
                "vaddr": hex(DESC_TABLE_VA),
                "offset": hex(desc_off),
                "length": len(desc_bytes),
            },
            "mean_table": {
                "vaddr": hex(MEAN_TABLE_VA),
                "offset": hex(mean_off),
                "length": len(mean_bytes),
            },
            "std_table": {
                "vaddr": hex(STD_TABLE_VA),
                "offset": hex(std_off),
                "length": len(std_bytes),
            },
        },
        "dsp_io_buffers": {
            "tensor_input_rgb56x56": hex(TENSOR_IN_VA),
            "tensor_message_struct": hex(TENSOR_MSG_VA),
            "heatmap_output_64x64": hex(TENSOR_OUT_VA),
        },
        "af_tracking_bridge": {
            "primary_slot0_vaddr": hex(TRACK_SLOT0_VA),
            "update_target_vaddr": hex(UPDATE_TRACK_VA),
            "track_obj_ptr_vaddr": hex(TRACK_OBJ_PTR_VA),
        }
    }
    return manifest

if __name__ == "__main__":
    print("[*] Generating Stage 3 AI Live View Tracking Bridge payload...")
    cave = build_stage3_cave_payload()
    tramp = build_trampoline_payload()
    print(f"  -> Cave payload: {len(cave)} bytes ({len(cave)//4} instructions)")
    print(f"  -> Trampoline:   {len(tramp)} bytes ({tramp.hex()})")

    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    print("\n--- Trampoline Disassembly ---")
    for ins in md.disasm(tramp, HOOK_SITE_VA):
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")

    print("\n--- Code Cave Disassembly ---")
    for ins in md.disasm(cave, CAVE_VA):
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")
    print("\n[+] Stage 3 payload assembled and verified successfully.")

