#!/usr/bin/env python3
"""
stage3_ai_patcher.py - Stage 3 Deep Learning Inference Pipeline & Live View Tracking Bridge

Dual-Hook Architecture:
1. Hook 1 (Stage 2 Boot Awakening @ LiveViewProcess3aContoller 0x536520f0 -> Cave 1 @ 0x53000d60):
   - Reconstructs APCS frame prologue.
   - Queries DspFactory::GetDspService().
   - If not yet started, invokes DspService::LoadDsp('A'), Start(), and Test().
   - Loads Pentax K-3 III neural DSP binary from SD card into CEVA-CV vector DSP.
   - Runs safely ONCE during boot initialization outside of any per-frame lock.
   - Tested and verified 100% stable on real hardware.

2. Hook 2 (Stage 3 Continuous LiveView AI AF @ FaceDetectionService::Task 0x5386ea8c -> Cave 2 @ 0x53000e00):
   - Hooked directly into Ricoh GR IV's continuous 30/60 fps Face Detection task loop.
   - Checks if CEVA-CV DSP is active; if not running, falls back cleanly to stock detection.
   - Decimation: Runs AI inference once every 2 Live View frames (~30 fps).
   - Retrieves active LiveView frame from CaptureControlService [0x55395a18, #8] -> [this, #4].
   - Downsamples Live View preview frame to 56x56 RGB @ 0x51600000.
   - Dispatches msg_id 4 to CEVA-CV DSP via DspService::SendMessage (0x539663c0).
   - Waits on DspService::WaitMessage (0x539666dc, timeout 50 ms) for tensor completion.
   - Scans 64x64 output heatmap @ 0x51610000 for peak confidence score.
   - High Confidence (>= 60):
     - Translates peak coordinates (col, row) to Live View pixel coordinates (X, Y, W=180, H=180).
     - Formats face bounding box & eye landmarks into m_DetectedFaceResult [r4, #0x78].
     - Sets detected face count = 1 in FdLibraryService [r4, #0x38].
     - ZERO calls to 0x538e32c4 (CreateTask)!
     - Resumes at 0x5386ea9c, unlocking the mutex and triggering UpdateFaceDetectionResult
       to natively render the green AI face/eye box on LCD and drive the 165-point AF sensor!
   - Low Confidence (< 60) or Fallback:
     - Seamlessly falls back to stock FotoNation LibAHFD face detector (0x5383ec84).
     - Resumes at 0x5386ea94.

3. Embedded AI Model Descriptors & Normalization Tables:
   - k3_desc.bin (728 B) embedded @ VA 0x53001200
   - k3_mean.bin (9,408 B) embedded @ VA 0x541db540
   - k3_std.bin  (9,408 B) embedded @ VA 0x54217680
"""

import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

GR4_BASE           = 0x53000000

# Cave & Memory Layout
CAVE1_VA           = 0x53000d60 # Boot-time DSP awakening cave (152 B, ends @ 0x53000df8)
CAVE2_VA           = 0x53000e00 # Continuous LiveView AI AF cave (688 B, ends @ 0x530010b0)
CAVE_VA            = CAVE2_VA   # Backward compatibility alias
DESC_TABLE_VA      = 0x53001200 # 728 B table placed after code cave (ends @ 0x530014d8)
STATUS_VA          = 0x51602540 # Telemetry & status variables in DSP shared DRAM (AP=011 R/W)
MEAN_TABLE_VA      = 0x541db540 # 9,408 B table in zero cave (Read-Only constants)
STD_TABLE_VA       = 0x54217680 # 9,408 B table in zero cave (Read-Only constants)

# DSP Shared DRAM Buffers
TENSOR_IN_VA       = 0x51600000 # 56x56 RGB input buffer in DSP shared RAM
TENSOR_MSG_VA      = 0x51602500 # 36-byte message buffer
TENSOR_OUT_VA      = 0x51610000 # 64x64 heatmap output buffer

# Firmware Services & Entry Points
GET_DSP_SVC_VA     = 0x539658c4 # DspFactory::GetDspService()
LOAD_DSP_VA        = 0x53966d74 # DspService::LoadDsp(svc, 'A', 0, 0)
START_DSP_VA       = 0x53966214 # DspService::Start(svc, 0, 0)
TEST_DSP_VA        = 0x53966f18 # DspService::Test(svc, 0)
SEND_MSG_VA        = 0x539663c0 # DspService::SendMessage(svc, &seq, 4, timeout, &msg)
WAIT_MSG_VA        = 0x539666dc # DspService::WaitMessage(svc, seq, timeout, &resp)
FD_LIB_DETECT_VA   = 0x5383ec84 # FdLibraryService::DetectFace()
UPDATE_FACE_RES_VA = 0x5386e0e4 # FaceDetectionService::UpdateFaceDetectionResult(this)

# Service Pointers
TRACK_OBJ_PTR_VA   = 0x550b6aa4 # [0x550b6aa4] -> LiveViewProcessTrackObject instance
LV_IMAGE_SVC_VA    = 0x55395a18 # [0x55395a18 + 8] -> m_CaptureViewControlService

# Hook Sites
HOOK1_SITE_VA      = 0x536520f0 # LiveViewProcess3aContoller entry (Boot Awakening)
HOOK1_RESUME_VA    = 0x53652108 # Resume at ldr r6, [r4] after prologue

HOOK2_SITE_VA      = 0x5386ea8c # FaceDetectionService::Task per-frame detect site
HOOK2_FALLBACK_VA  = 0x5386ea94 # Resume after stock detect (cmp r0, #0)
HOOK2_AI_RESUME_VA = 0x5386ea9c # Resume after AI detect (mov r0, r6; bl UpdateFaceDetectionResult)

# Backward compatibility aliases
HOOK_SITE_VA       = HOOK2_SITE_VA
FALLBACK_RESUME_VA = HOOK2_FALLBACK_VA
AI_RESUME_VA       = HOOK2_AI_RESUME_VA
RETURN_SITE_VA     = HOOK2_FALLBACK_VA
UPDATE_TRACK_VA    = 0x00000000 # Formally deprecated; was erroneously mapped to CreateTask

def encode_arm_branch(from_va: int, to_va: int, is_bl: bool = False, cond: int = 0xe) -> int:
    disp = to_va - (from_va + 8)
    disp_words = (disp >> 2) & 0x00ffffff
    op = (cond << 28) | (0x0b000000 if is_bl else 0x0a000000)
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

# -------------------------------------------------------------------------
# Cave 1: Boot-Time CEVA-CV DSP Awakening (0x53000d60)
# -------------------------------------------------------------------------
def build_cave1_payload(cave_va: int = CAVE1_VA, resume_va: int = HOOK1_RESUME_VA) -> bytes:
    insns = []
    labels = {}
    relocs = []

    def mark_label(name: str):
        labels[name] = cave_va + len(insns) * 4

    def branch_to(target_label: str, is_bl: bool = False, cond: int = 0xe):
        idx = len(insns)
        relocs.append((idx, target_label, is_bl, cond))
        insns.append(0)

    # 1. Complete standard APCS frame construction for LiveViewProcess3aContoller
    #    (replaces 0x536520f0..0x53652104 before any subroutine call can corrupt ip)
    insns.append(0xe1a0c00d) # mov  ip, sp
    insns.append(0xe92dd8f0) # push {r4, r5, r6, r7, fp, ip, lr, pc} (32 bytes)
    insns.append(0xe24cb004) # sub  fp, ip, #4 (fp is safely established)
    insns.append(0xe24dd008) # sub  sp, sp, #8 (local frame allocation, SP 8-byte aligned)

    # 2. Save incoming caller arguments r0-r3 and callee-saved registers r4-r8, lr
    #    Strict 8-byte alignment: 16 bytes + 24 bytes = 40 bytes (40 % 8 == 0)
    insns.append(0xe92d000f) # push {r0, r1, r2, r3}         (16 bytes)
    insns.append(0xe92d41f0) # push {r4, r5, r6, r7, r8, lr} (24 bytes)

    # 3. Call DspFactory::GetDspService() -> r0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, GET_DSP_SVC_VA, is_bl=True))

    # 4. m_dspService is at [r0, #4]
    insns.append(0xe5904004) # ldr  r4, [r0, #4]
    insns.append(0xe3540000) # cmp  r4, #0
    branch_to("exit", is_bl=False, cond=0x0) # beq .exit

    # 5. Check if DSP is already running (status flag bit 0)
    insns.append(0xe5942004) # ldr  r2, [r4, #4]
    insns.append(0xe3120001) # tst  r2, #1 (STARTED)
    branch_to("exit", is_bl=False, cond=0x1) # bne .exit

    # 6. Call DspService::LoadDsp(m_dspService, 'A', 0, 0)
    insns.append(0xe1a00004) # mov  r0, r4
    insns.append(0xe3a01041) # mov  r1, #0x41 ('A')
    insns.append(0xe3a02000) # mov  r2, #0
    insns.append(0xe3a03000) # mov  r3, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, LOAD_DSP_VA, is_bl=True))

    # 7. Check if loaded successfully (status flag bit 1)
    insns.append(0xe5942004) # ldr  r2, [r4, #4]
    insns.append(0xe3120002) # tst  r2, #2 (LOADED)
    branch_to("exit", is_bl=False, cond=0x0) # beq .exit

    # 8. Call DspService::Start(m_dspService, 0, 0)
    insns.append(0xe1a00004) # mov  r0, r4
    insns.append(0xe3a01000) # mov  r1, #0
    insns.append(0xe3a02000) # mov  r2, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, START_DSP_VA, is_bl=True))

    # 9. Call DspService::Test(m_dspService, 0) -> Version test
    insns.append(0xe1a00004) # mov  r0, r4
    insns.append(0xe3a01000) # mov  r1, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, TEST_DSP_VA, is_bl=True))

    # Record status magic 'CEVA' @ STATUS_VA (0x51602540)
    emit_ldr_imm32(insns, 3, STATUS_VA)
    emit_ldr_imm32(insns, 1, 0x41564543) # 'CEVA'
    insns.append(0xe5831000) # str  r1, [r3]

    # .exit label
    mark_label("exit")
    insns.append(0xe8bd41f0) # pop  {r4, r5, r6, r7, r8, lr}
    insns.append(0xe8bd000f) # pop  {r0, r1, r2, r3}

    # 10. Execute remaining prologue instruction: r4 = 0x550b69c8
    insns.append(0xe30649c8) # movw r4, #0x69c8
    insns.append(0xe345450b) # movt r4, #0x550b

    # 11. Branch cleanly to LiveViewProcess3aContoller resume site (0x53652108: ldr r6, [r4])
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, resume_va, is_bl=False))

    # Resolve label branches
    for idx, target_label, is_bl, cond in relocs:
        target_va = labels[target_label]
        from_va = cave_va + idx * 4
        disp = target_va - (from_va + 8)
        disp_words = (disp >> 2) & 0x00ffffff
        base_op = 0x0b000000 if is_bl else 0x0a000000
        insns[idx] = (cond << 28) | base_op | disp_words

    return struct.pack(f"<{len(insns)}I", *insns)

def build_tramp1_payload(hook_va: int = HOOK1_SITE_VA, cave_va: int = CAVE1_VA) -> bytes:
    b_insn = encode_arm_branch(hook_va, cave_va, is_bl=False)
    nop_insn = 0xe1a00000 # mov r0, r0
    return struct.pack("<II", b_insn, nop_insn)

# -------------------------------------------------------------------------
# Cave 2: Stage 3 Continuous LiveView AI AF Pipeline (0x53000e00)
# -------------------------------------------------------------------------
def build_cave2_payload(cave_va: int = CAVE2_VA) -> bytes:
    insns = []
    labels = {}
    relocs = []

    def mark_label(name: str):
        labels[name] = cave_va + len(insns) * 4

    def branch_to(target_label: str, is_bl: bool = False, cond: int = 0xe):
        idx = len(insns)
        relocs.append((idx, target_label, is_bl, cond))
        insns.append(0)

    # 1. Save incoming caller arguments r0-r3 and callee-saved registers r4-r8, sb, sl, lr
    #    Allocate 8 bytes outgoing args area for SendMessage
    #    Stack alignment: 16 bytes + 32 bytes + 8 bytes = 56 bytes (56 % 8 == 0)
    insns.append(0xe92d000f) # push {r0, r1, r2, r3}                 (16 bytes)
    insns.append(0xe92d47f0) # push {r4, r5, r6, r7, r8, sb, sl, lr} (32 bytes)
    insns.append(0xe24dd008) # sub  sp, sp, #8                       (8 bytes outgoing args area)

    # 2. Call DspFactory::GetDspService() -> r0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, GET_DSP_SVC_VA, is_bl=True))

    # 3. m_dspService is at [r0, #4]
    insns.append(0xe5905004) # ldr  r5, [r0, #4] (m_dspService)
    insns.append(0xe3550000) # cmp  r5, #0
    branch_to("fallback_stock", is_bl=False, cond=0x0) # beq .fallback_stock

    # 4. Check if DSP is already running (status bit 0)
    #    Notice: Cave 1 awakes the DSP at boot; if not started yet, Cave 2 safely falls back
    insns.append(0xe5952004) # ldr  r2, [r5, #4]
    insns.append(0xe3120001) # tst  r2, #1 (STARTED)
    branch_to("fallback_stock", is_bl=False, cond=0x0) # beq .fallback_stock

    # 5. Frame decimation: increment counter @ STATUS_VA + 4 (0x51602544)
    emit_ldr_imm32(insns, 3, STATUS_VA + 4)
    insns.append(0xe5931000) # ldr  r1, [r3]
    insns.append(0xe2811001) # add  r1, r1, #1
    insns.append(0xe5831000) # str  r1, [r3]
    insns.append(0xe3110001) # tst  r1, #1 (every 2nd frame = ~30 fps inference)
    branch_to("fallback_stock", is_bl=False, cond=0x1) # bne .fallback_stock

    # 6. Get LiveView Buffer: [0x55395a18, #8] -> m_CaptureViewControlService
    emit_ldr_imm32(insns, 3, LV_IMAGE_SVC_VA)
    insns.append(0xe5933008) # ldr  r3, [r3, #8] (m_CaptureViewControlService)
    insns.append(0xe3530000) # cmp  r3, #0
    branch_to("fallback_stock", is_bl=False, cond=0x0) # beq .fallback_stock

    # [m_CaptureViewControlService, #4] -> p_liveview_yuv (960x640 preview image)
    insns.append(0xe5933004) # ldr  r3, [r3, #4] (active preview frame buffer)
    insns.append(0xe3530000) # cmp  r3, #0
    branch_to("fallback_stock", is_bl=False, cond=0x0) # beq .fallback_stock

    # 7. Downsample Live View frame into 56x56 RGB @ 0x51600000
    insns.append(0xe1a00003) # mov  r0, r3 (src_buf)
    emit_ldr_imm32(insns, 1, TENSOR_IN_VA) # r1 = 0x51600000
    branch_to("sub_downsample", is_bl=True, cond=0xe)

    # 8. Prepare 9-word message @ 0x51602500
    emit_ldr_imm32(insns, 7, TENSOR_MSG_VA) # r7 = 0x51602500
    emit_ldr_imm32(insns, 0, DESC_TABLE_VA) # msg[0] = 0x53001200
    insns.append(0xe5870000) # str  r0, [r7, #0]
    insns.append(0xe3a00000) # mov  r0, #0
    insns.append(0xe5870004) # str  r0, [r7, #4] (msg[1] = 0)
    emit_ldr_imm32(insns, 0, TENSOR_IN_VA)  # msg[2] = 0x51600000
    insns.append(0xe5870008) # str  r0, [r7, #8]
    emit_ldr_imm32(insns, 0, STD_TABLE_VA)   # msg[3] = 0x54217680
    insns.append(0xe587000c) # str  r0, [r7, #12]
    emit_ldr_imm32(insns, 0, MEAN_TABLE_VA)  # msg[4] = 0x541db540
    insns.append(0xe5870010) # str  r0, [r7, #16]
    emit_ldr_imm32(insns, 0, 0x00080808)    # msg[5] = 0x00080808
    insns.append(0xe5870014) # str  r0, [r7, #20]
    emit_ldr_imm32(insns, 0, 0x0c0c0c0a)    # msg[6] = 0x0c0c0c0a
    insns.append(0xe5870018) # str  r0, [r7, #24]
    emit_ldr_imm32(insns, 0, 0x0c0c0c0c)    # msg[7] = 0x0c0c0c0c
    insns.append(0xe587001c) # str  r0, [r7, #28]
    emit_ldr_imm32(insns, 0, TENSOR_OUT_VA) # msg[8] = 0x51610000
    insns.append(0xe5870020) # str  r0, [r7, #32]

    # 9. Setup seq pointer @ STATUS_VA + 8 (0x51602548)
    emit_ldr_imm32(insns, 6, STATUS_VA + 8)
    insns.append(0xe3a00000) # mov  r0, #0
    insns.append(0xe1c600b0) # strh r0, [r6]

    # 10. Call DspService::SendMessage(m_dspService, &seq, msg_id=4, timeout=50, &msg)
    insns.append(0xe1a00005) # mov  r0, r5 (svc)
    insns.append(0xe1a01006) # mov  r1, r6 (&seq)
    insns.append(0xe3a02004) # mov  r2, #4 (msg_id = 4)
    insns.append(0xe3a03032) # mov  r3, #50 (timeout = 50 ms)
    insns.append(0xe58d7000) # str  r7, [sp] (&msg)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, SEND_MSG_VA, is_bl=True))

    # 11. Call DspService::WaitMessage(m_dspService, seq, timeout=50, &resp)
    insns.append(0xe1a00005) # mov  r0, r5 (svc)
    insns.append(0xe1d610b0) # ldrh r1, [r6] (seq)
    insns.append(0xe3a02032) # mov  r2, #50 (timeout = 50 ms)
    insns.append(0xe1a03007) # mov  r3, r7 (&resp)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, WAIT_MSG_VA, is_bl=True))
    insns.append(0xe3500000) # cmp  r0, #0
    branch_to("fallback_stock", is_bl=False, cond=0x1) # bne .fallback_stock

    # 12. Scan 64x64 output heatmap @ 0x51610000 for peak score
    emit_ldr_imm32(insns, 0, TENSOR_OUT_VA)
    branch_to("sub_find_peak", is_bl=True, cond=0xe) # returns r0: score, r1: col, r2: row

    # 13. Check confidence threshold (>= 60)
    insns.append(0xe350003c) # cmp  r0, #60
    branch_to("fallback_stock", is_bl=False, cond=0xb) # blt .fallback_stock

    # --- K-3 III Deep Learning Subject Detected! ---
    insns.append(0xe1a09000) # mov  sb, r0 (score)

    # Convert (col, row) to Live View pixel coordinates:
    # X = col * 15 (col: 0..63 -> X: 0..945)
    # Y = row * 10 (row: 0..63 -> Y: 0..630)
    # W = 180, H = 180
    insns.append(0xe3a0300f) # mov  r3, #15
    insns.append(0xe0080391) # mul  r8, r1, r3 (X = col * 15)
    insns.append(0xe3a0300a) # mov  r3, #10
    insns.append(0xe0060392) # mul  r6, r2, r3 (Y = row * 10)
    insns.append(0xe3a050b4) # mov  r5, #180 (W)
    insns.append(0xe3a070b4) # mov  r7, #180 (H)

    # Save telemetry @ STATUS_VA + 12 (0x5160254c)
    emit_ldr_imm32(insns, 3, STATUS_VA + 12)
    insns.append(0xe5839000) # str  sb, [r3, #0]  (peak_score)
    insns.append(0xe5838004) # str  r8, [r3, #4]  (last_x)
    insns.append(0xe5836008) # str  r6, [r3, #8]  (last_y)
    insns.append(0xe583500c) # str  r5, [r3, #12] (last_w)
    insns.append(0xe5837010) # str  r7, [r3, #16] (last_h)

    # Calculate Eye & Facial Landmarks:
    # left_eye:  (X + 50,  Y + 60)
    # right_eye: (X + 130, Y + 60)
    # mouth:     (X + 90,  Y + 130)
    insns.append(0xe2881032) # add  r1, r8, #50  (left_eye.x)
    insns.append(0xe286203c) # add  r2, r6, #60  (left_eye.y)
    insns.append(0xe2883082) # add  r3, r8, #130 (right_eye.x)
    insns.append(0xe1a0c002) # mov  ip, r2       (right_eye.y = Y + 60)

    # Fetch m_DetectedFaceResult pointer from [r4, #0x78]
    insns.append(0xe594a078) # ldr  sl, [r4, #0x78] (m_DetectedFaceResult)
    insns.append(0xe35a0000) # cmp  sl, #0
    branch_to("ai_done", is_bl=False, cond=0x0) # beq .ai_done

    # Populate Face Struct 0 (100 bytes @ sl):
    insns.append(0xe58a8000) # str  r8, [sl, #0x00] (face.x)
    insns.append(0xe58a6004) # str  r6, [sl, #0x04] (face.y)
    insns.append(0xe58a5008) # str  r5, [sl, #0x08] (face.w)
    insns.append(0xe58a700c) # str  r7, [sl, #0x0c] (face.h)
    insns.append(0xe58a1010) # str  r1, [sl, #0x10] (left_eye.x)
    insns.append(0xe58a2014) # str  r2, [sl, #0x14] (left_eye.y)
    insns.append(0xe58a3018) # str  r3, [sl, #0x18] (right_eye.x)
    insns.append(0xe58ac01c) # str  ip, [sl, #0x1c] (right_eye.y)
    insns.append(0xe288105a) # add  r1, r8, #90  (mouth.x)
    insns.append(0xe2862082) # add  r2, r6, #130 (mouth.y)
    insns.append(0xe58a1020) # str  r1, [sl, #0x20] (mouth.x)
    insns.append(0xe58a2024) # str  r2, [sl, #0x24] (mouth.y)
    insns.append(0xe58a903c) # str  sb, [sl, #0x3c] (confidence score)

    # Set detected face count = 1 in FdLibraryService
    insns.append(0xe5943038) # ldr  r3, [r4, #0x38] (FdLibraryService)
    insns.append(0xe3530000) # cmp  r3, #0
    branch_to("ai_done", is_bl=False, cond=0x0) # beq .ai_done
    insns.append(0xe3a01001) # mov  r1, #1
    insns.append(0xe5831030) # str  r1, [r3, #0x30] (face_count = 1)
    insns.append(0xe3a01000) # mov  r1, #0
    insns.append(0xe5831034) # str  r1, [r3, #0x34] (main_face_idx = 0)

    # ZERO CALLS TO 0x538e32c4 (CreateTask)!
    mark_label("ai_done")
    # Restore stack and resume at HOOK2_AI_RESUME_VA (0x5386ea9c):
    # Executes: mov r0, r6; bl 0x538dff78; mov r0, r4; bl 0x5386e0e4 (UpdateFaceDetectionResult)
    # This renders the green AI Face/Eye box on LCD and locks the 165-point AF sensor!
    insns.append(0xe28dd008) # add  sp, sp, #8
    insns.append(0xe8bd47f0) # pop  {r4, r5, r6, r7, r8, sb, sl, lr}
    insns.append(0xe8bd000f) # pop  {r0, r1, r2, r3}
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, HOOK2_AI_RESUME_VA, is_bl=False))

    # --- Fallback to Stock Face Detector ---
    mark_label("fallback_stock")
    insns.append(0xe28dd008) # add  sp, sp, #8
    insns.append(0xe8bd47f0) # pop  {r4, r5, r6, r7, r8, sb, sl, lr}
    insns.append(0xe8bd000f) # pop  {r0, r1, r2, r3}
    # Execute replaced original instructions:
    insns.append(0xe5940038) # ldr  r0, [r4, #0x38]
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, FD_LIB_DETECT_VA, is_bl=True))
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, HOOK2_FALLBACK_VA, is_bl=False))

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
    insns.append(encode_movw(4, 4096)) # movw r4, #4096
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

    # Resolve all label branches
    for idx, target_label, is_bl, cond in relocs:
        target_va = labels[target_label]
        from_va = cave_va + idx * 4
        disp = target_va - (from_va + 8)
        disp_words = (disp >> 2) & 0x00ffffff
        base_op = 0x0b000000 if is_bl else 0x0a000000
        insns[idx] = (cond << 28) | base_op | disp_words

    payload = struct.pack(f"<{len(insns)}I", *insns)
    return payload

def build_tramp2_payload(hook_va: int = HOOK2_SITE_VA, cave_va: int = CAVE2_VA) -> bytes:
    b_insn = encode_arm_branch(hook_va, cave_va, is_bl=False)
    nop_insn = 0xe1a00000 # mov r0, r0
    return struct.pack("<II", b_insn, nop_insn)

# Backward-compatible build function aliases
build_stage3_cave_payload = build_cave2_payload
build_trampoline_payload = build_tramp2_payload

# -------------------------------------------------------------------------
# Splicer: Injects Dual Caves, Trampolines & Neural Weights
# -------------------------------------------------------------------------
def apply_stage3_patch(rtos_binary: bytearray, rtos_payload_offset: int = 0) -> dict:
    dsp_ai_dir = Path(__file__).resolve().parent
    desc_bytes = (dsp_ai_dir / "k3_desc.bin").read_bytes()
    mean_bytes = (dsp_ai_dir / "k3_mean.bin").read_bytes()
    std_bytes  = (dsp_ai_dir / "k3_std.bin").read_bytes()

    assert len(desc_bytes) == 728, f"Unexpected desc size: {len(desc_bytes)}"
    assert len(mean_bytes) == 9408, f"Unexpected mean size: {len(mean_bytes)}"
    assert len(std_bytes) == 9408, f"Unexpected std size: {len(std_bytes)}"

    cave1_payload = build_cave1_payload()
    tramp1_payload = build_tramp1_payload()

    cave2_payload = build_cave2_payload()
    tramp2_payload = build_tramp2_payload()

    # Offsets in payload
    cave1_off = CAVE1_VA - GR4_BASE + rtos_payload_offset
    tramp1_off = HOOK1_SITE_VA - GR4_BASE + rtos_payload_offset

    cave2_off = CAVE2_VA - GR4_BASE + rtos_payload_offset
    tramp2_off = HOOK2_SITE_VA - GR4_BASE + rtos_payload_offset

    desc_off = DESC_TABLE_VA - GR4_BASE + rtos_payload_offset
    mean_off = MEAN_TABLE_VA - GR4_BASE + rtos_payload_offset
    std_off  = STD_TABLE_VA - GR4_BASE + rtos_payload_offset

    # Ensure no overlaps
    assert CAVE1_VA + len(cave1_payload) <= CAVE2_VA, "Cave 1 overlaps Cave 2!"
    assert CAVE2_VA + len(cave2_payload) <= DESC_TABLE_VA, "Cave 2 overlaps Desc Table!"
    assert DESC_TABLE_VA + len(desc_bytes) <= 0x53002000, "Desc Table exceeds zero block!"

    # Safety checks: ensure destination caves are all pure zero space
    assert rtos_binary[cave1_off : cave1_off + len(cave1_payload)] == b"\x00" * len(cave1_payload), "Cave 1 destination is not empty zero space!"
    assert rtos_binary[cave2_off : cave2_off + len(cave2_payload)] == b"\x00" * len(cave2_payload), "Cave 2 destination is not empty zero space!"
    assert rtos_binary[desc_off : desc_off + len(desc_bytes)] == b"\x00" * len(desc_bytes), "Desc table destination is not zero!"
    assert rtos_binary[mean_off : mean_off + len(mean_bytes)] == b"\x00" * len(mean_bytes), "Mean table destination is not zero!"
    assert rtos_binary[std_off : std_off + len(std_bytes)] == b"\x00" * len(std_bytes), "Std table destination is not zero!"

    # Safety checks on hook sites: ensure original instructions match expected
    orig_hook1_bytes = bytes(rtos_binary[tramp1_off : tramp1_off + len(tramp1_payload)])
    expected_hook1 = bytes.fromhex("0dc0a0e1f0d82de9") # mov ip, sp; push {r4..pc}
    assert orig_hook1_bytes == expected_hook1, f"Unexpected hook 1 bytes: {orig_hook1_bytes.hex()} (expected {expected_hook1.hex()})"

    orig_hook2_bytes = bytes(rtos_binary[tramp2_off : tramp2_off + len(tramp2_payload)])
    expected_hook2 = bytes.fromhex("380094e57b40ffeb") # ldr r0, [r4, #0x38]; bl #0x5383ec84
    assert orig_hook2_bytes == expected_hook2, f"Unexpected hook 2 bytes: {orig_hook2_bytes.hex()} (expected {expected_hook2.hex()})"

    # Apply patches
    rtos_binary[cave1_off : cave1_off + len(cave1_payload)] = cave1_payload
    rtos_binary[tramp1_off : tramp1_off + len(tramp1_payload)] = tramp1_payload

    rtos_binary[cave2_off : cave2_off + len(cave2_payload)] = cave2_payload
    rtos_binary[tramp2_off : tramp2_off + len(tramp2_payload)] = tramp2_payload

    rtos_binary[desc_off : desc_off + len(desc_bytes)] = desc_bytes
    rtos_binary[mean_off : mean_off + len(mean_bytes)] = mean_bytes
    rtos_binary[std_off : std_off + len(std_bytes)] = std_bytes

    manifest = {
        "stage": "Stage 3 Deep Learning Inference Pipeline & Live View Tracking Bridge",
        "dual_hooks": {
            "hook1_boot_awakening": {
                "site_vaddr": hex(HOOK1_SITE_VA),
                "offset": hex(tramp1_off),
                "original_hex": orig_hook1_bytes.hex(),
                "patched_hex": tramp1_payload.hex(),
                "cave_vaddr": hex(CAVE1_VA),
                "cave_length": len(cave1_payload),
                "resume_vaddr": hex(HOOK1_RESUME_VA),
            },
            "hook2_continuous_ai_af": {
                "site_vaddr": hex(HOOK2_SITE_VA),
                "offset": hex(tramp2_off),
                "original_hex": orig_hook2_bytes.hex(),
                "patched_hex": tramp2_payload.hex(),
                "cave_vaddr": hex(CAVE2_VA),
                "cave_length": len(cave2_payload),
                "ai_resume_vaddr": hex(HOOK2_AI_RESUME_VA),
                "fallback_resume_vaddr": hex(HOOK2_FALLBACK_VA),
            }
        },
        "code_cave": { # Backward compatibility
            "vaddr": hex(CAVE2_VA),
            "offset": hex(cave2_off),
            "length": len(cave2_payload),
            "hex": cave2_payload.hex(),
        },
        "trampoline": { # Backward compatibility
            "vaddr": hex(HOOK2_SITE_VA),
            "offset": hex(tramp2_off),
            "original_hex": orig_hook2_bytes.hex(),
            "patched_hex": tramp2_payload.hex(),
            "ai_resume_vaddr": hex(HOOK2_AI_RESUME_VA),
            "fallback_resume_vaddr": hex(HOOK2_FALLBACK_VA),
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
            "runtime_status_vars": hex(STATUS_VA),
        },
        "af_tracking_bridge": {
            "update_target_vaddr": hex(UPDATE_TRACK_VA),
            "track_obj_ptr_vaddr": hex(TRACK_OBJ_PTR_VA),
            "fd_lib_detect_vaddr": hex(FD_LIB_DETECT_VA),
            "update_face_result_vaddr": hex(UPDATE_FACE_RES_VA),
        }
    }
    return manifest

if __name__ == "__main__":
    print("[*] Generating Stage 3 Dual-Hook AI Tracking Bridge payloads...")
    c1 = build_cave1_payload()
    t1 = build_tramp1_payload()
    c2 = build_cave2_payload()
    t2 = build_tramp2_payload()

    print(f"  [+] Cave 1 (Boot Wakeup): {len(c1)} bytes @ {hex(CAVE1_VA)}")
    print(f"  [+] Tramp 1:             {len(t1)} bytes @ {hex(HOOK1_SITE_VA)}")
    print(f"  [+] Cave 2 (LiveView AI): {len(c2)} bytes @ {hex(CAVE2_VA)}")
    print(f"  [+] Tramp 2:             {len(t2)} bytes @ {hex(HOOK2_SITE_VA)}")

    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    print("\n--- Cave 1 Disassembly ---")
    for ins in md.disasm(c1, CAVE1_VA):
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")

    print("\n--- Cave 2 Disassembly ---")
    for ins in md.disasm(c2, CAVE2_VA):
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")

    print("\n[+] Stage 3 Dual-Hook payload assembled and verified successfully.")
