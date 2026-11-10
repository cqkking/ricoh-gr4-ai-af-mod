#!/usr/bin/env python3
"""
stage3_ai_patcher.py - Stage 3 Deep Learning Inference Pipeline & Live View Tracking Bridge (v1.2.0)

Rebased onto official Ricoh GR IV v1.20 firmware baseline.

Dual-Hook Architecture:
1. Hook 1 (Boot Status Init @ LiveViewProcess3aContoller 0x53676ab8 -> Cave 1 @ 0x53000d60):
   - Reconstructs APCS frame prologue without stalling early-boot 3A initialization on SD I/O.
   - Loads singleton pointer from 0x550b8650 and zeroes the 32-byte DSP status/telemetry region
     @ STATUS_VA (0x52a80100..0x52a8011f) on first boot singleton initialization.

2. Hook 2 (Pure Full-Time CEVA-CV AI AF @ FaceDetectionService::Task 0x5389639c -> Cave 2 @ 0x53000e00):
   - Zero Fallback to Stock LibAHFD: FdLibraryService::DetectFace (0x53866594) is completely bypassed
     so 100% of face/subject boxes on LCD originate from CEVA-CV hardware CNN.
   - Boot-Guaranteed Initialization Counter via constructor-zeroed [r4, #0x13e] + STATUS_VA + 4:
     Eliminates uninitialized DDR bugs. Waits 20 frames (~0.65s) for SD card mount, then retries
     LoadDsp('C') every 64 frames (~2.1s) for up to 30 attempts (~65s) while unlocking mutex (r4 + 0x48)
     during SD card / storage I/O.
   - Fixed LoadDsp return value handling: DspService::LoadDsp returns 1 on SUCCESS (bne cave2_load_ok).
   - Dual-Drive Polling: Checks Drive 'C' (SD card) first, then falls back to Drive 'I' (Internal 53GB flash).
   - Completed AHIP Ring Frame Extraction & ARM D-Cache Invalidation:
     Reads the completed frame ring descriptor *(*(0x550813d0 + 0x434)) (fallback 0x424 / 0x1c / 0x14)
     dequeued by AhipService::WaitFrame (Y @ +0x04, U @ +0x08, V @ +0x0c) and calls InvalidateDataCache
     before CPU reads.
   - Aspect-Ratio-Aware Downsampler + C0-Continuous 2.0x Backlit Shadow Recovery:
     Maps [r4, #0xfc] x [r4, #0xfe] YUV420 to 3-plane 56x56 (stride 64) RGB @ 0x52940000 while lifting
     crushed backlit (逆光) shadows (Y < 48 -> 2x, 48..144 -> 72 + Y/2) without clipping highlights.
   - Head-Priority Heatmap Scanner, Dynamic Box Sizing & Full SelectMainFace State Seeding:
     Prioritizes Ch1 (Human/Head) over Ch2 (Body/Animal), scales Q8 coordinates across all aspect ratios,
     sizes the AF box dynamically from foreground blob area, and seeds [r4, #0x100..0x10c] so
     SelectMainFace & EVENT 0x04070001 lock instantaneously on Frame 1.
"""

import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

GR4_BASE               = 0x53000000
RTOS_PAYLOAD_START     = 0x00017D28

# Cave & Memory Layout in v1.20 ROM
CAVE1_VA               = 0x53000d60 # Boot-time DSP awakening cave (84 B)
CAVE2_VA               = 0x53000e00 # Hybrid Continuous LiveView AI AF cave (<= 2048 B)
CAVE_VA                = CAVE2_VA   # Backward compatibility alias
DESC_TABLE_VA          = 0x53001600 # 728 B table placed after code cave (ends @ 0x530018d8)
MEAN_TABLE_VA          = 0x54212890 # 9,408 B table inside 13,167 byte ROM zero cave
STD_TABLE_VA           = 0x5424e9d0 # 9,408 B table inside 16,061 byte ROM zero cave

# Non-Overlapping DSP Shared DRAM Layout (0x52930000 .. 0x52d00000)
# Safely relocated to 0x52b00000+ unallocated DDR (outside any RTOS heap or weights!)
MEAN_WEIGHTS_VA        = 0x52960000 # Full K-3 III 0x53932548..0x539d7d08 block (677,824 B) loaded by DF_EX.BIN Chunk 1
HEAD_WEIGHTS_VA        = 0x52a057c0 # Full K-3 III 0x539d7d08..0x53a48768 block (461,408 B) loaded by DF_EX.BIN Chunk 1
# Chunk 2 ends at 0x52a76220. All runtime buffers placed safely at 0x52b00000+ (>565 KB above Chunk 2!):
TENSOR_IN_VA           = 0x52b00000 # work_buf (0xbde40 B): 3-plane 56x56 (stride 64) RGB input @ +0x0000, +0x0e00, +0x1c00
TENSOR_OUT_VA          = 0x52b09a00 # work_buf + 0x9a00: 3-channel 56x56 (stride 64) int16_t segmentation output
TENSOR_MSG_VA          = 0x52c00000 # 40-byte (0x28 B) DspCommandParameter buffer (outside work_buf!)
STATUS_VA              = 0x52c00100 # Telemetry & one-shot wakeup guard in DSP shared DRAM (outside work_buf!)

# Firmware Services & Entry Points in v1.20
GET_DSP_SVC_VA         = 0x5398d97c # DspFactory::GetDspService() -> singleton struct @ 0x553a3160
LOAD_DSP_VA            = 0x5398ee2c # DspService::LoadDsp(svc, 'C', 0, 0)
START_DSP_VA           = 0x5398e2cc # DspService::Start(svc, boot_addr=0x52100000, 0)
TEST_DSP_VA            = 0x5398efd0 # DspService::Test(svc, 0) - DO NOT CALL: factory diagnostic loopback that hangs in tk_wai_flg(0x3f0000)
SEND_MSG_VA            = 0x5398e478 # DspService::SendMessage(svc, &seq, 4, timeout, &msg)
WAIT_MSG_VA            = 0x5398e794 # DspService::WaitMessage(svc, seq, timeout, &resp)
FD_LIB_DETECT_VA       = 0x53866594 # FdLibraryService::DetectFace() (stock FotoNation LibAHFD detect entry)
FD_LIB_GET_RES_VA      = 0x5386666c # FdLibraryService::GetResults(FdLibraryService*, m_DetectedFaceResult)
UPDATE_FACE_RES_VA     = 0x538959f4 # FaceDetectionService::UpdateFaceDetectionResult(this)
LOCK_MUTEX_VA          = 0x53907930 # OsMutex::Lock(mutex, timeout)
UNLOCK_MUTEX_VA        = 0x53907a84 # OsMutex::Unlock(mutex)
MUTEX_COOKIE_VA        = 0x540184a0 # Mutex cookie pointer for FaceDetectionService
GET_CACHE_MGR_VA       = 0x53918010 # GetCacheManager() -> returns cache manager 0x553a26c4
FLUSH_DCACHE_VA        = 0x5391808c # FlushDataCache(mgr, addr, size)
INVAL_DCACHE_VA        = 0x5391801c # InvalidateDataCache(mgr, addr, size)

# Stock ROM Bug Sites & Patches in v1.20 DspService::LoadDsp (0x5398ee2c):
LOAD_DSP_ID_BUG_VA     = 0x5398ef68 # movw r1, #0xffac; movt r1, #0x5433 -> sub r1, fp, #0x40; nop
LOAD_DSP_IP_BUG1_VA    = 0x5398efac # movw r1, #0xff9c -> sub r1, fp, #0x40
LOAD_DSP_IP_BUG2_VA    = 0x5398efb8 # movt r1, #0x5433 -> nop
LOAD_DSP_DRV_CMP_VA    = 0x5398eedc # cmp r3, #5 -> cmp r3, #15 (allows drives 'A'..'P')
CHUNK_LOADER_BASE_VA   = 0x5398e178 # movt r3, #0x5120 -> movt r3, #0x5210
LOAD_DSP_P3_BASE_VA    = 0x5398ee9c # movt r3, #0x5120 -> movt r3, #0x5210
LOAD_DSP_IDS_DEST_VA   = 0x5398ef78 # movt r0, #0x515e -> movt r0, #0x524e
LOAD_DSP_IP_DEST_VA    = 0x5398efb4 # movt r0, #0x515f -> movt r0, #0x524f

# String pointers in v1.20 ROM
STR_DF_EX_VA           = 0x5433ff8c # "A:\DF_EX.BIN" -> "C:\DF_EX.BIN"
STR_DF_IP_VA           = 0x5433ff9c # "A:\DF_IP.BIN" -> "C:\DF_IP.BIN"
STR_DF_ID_S_VA         = 0x5433ffac # "A:\DF_ID_S.BIN" -> "C:\DF_ID_S.BIN"

# Hardware & Service Pointers in v1.20
AHIP_DRV_BASE_VA       = 0x55081350 # Active Ahip frame driver base in v1.20 (0x5311a632: movw r7, #0x1350; movt r7, #0x5508)
TRACK_OBJ_PTR_VA       = 0x550b8650 # Singleton pointer

# Hook Sites in v1.20
HOOK1_SITE_VA          = 0x53676ab8 # LiveViewProcess3aContoller entry (Boot Awakening)
HOOK1_RESUME_VA        = 0x53676ad4 # Resume at original code after prologue
HOOK1_SINGLETON_VA     = 0x550b8650 # Singleton struct pointer checked by Hook 1

HOOK2_SITE_VA          = 0x5389639c # FaceDetectionService::Task per-frame detect site
HOOK2_FALLBACK_VA      = 0x53896460 # Stock ConvertResult path when LibAHFD detects face (0x53896460)
HOOK2_AI_RESUME_VA     = 0x538963ac # Resume after AI detect (mov r0, r6; bl UpdateFaceDetectionResult)

# Memory Pool Table Offsets in Payload (.data)
POOL3_PAYLOAD_OFF      = 0x013e5b78 # Pool[3] (DspFw) @ 0x543cde50
POOL8_PAYLOAD_OFF      = 0x013e5bc8 # Pool[8] (ImageAdjustWork) @ 0x543cdea0

# Backward compatibility aliases
HOOK_SITE_VA           = HOOK2_SITE_VA
FALLBACK_RESUME_VA     = HOOK2_FALLBACK_VA
AI_RESUME_VA           = HOOK2_AI_RESUME_VA
RETURN_SITE_VA         = HOOK2_FALLBACK_VA
UPDATE_TRACK_VA        = 0x00000000

def encode_arm_branch(from_va: int, to_va: int, is_bl: bool = False, cond: int = 0xe) -> int:
    disp = to_va - (from_va + 8)
    disp_words = (disp >> 2) & 0x00ffffff
    op = (cond << 28) | (0x0b000000 if is_bl else 0x0a000000)
    return op | disp_words

def encode_movw(rd: int, imm16: int, cond: int = 0xe) -> int:
    imm4 = (imm16 >> 12) & 0xf
    imm12 = imm16 & 0xfff
    return (cond << 28) | 0x03000000 | (imm4 << 16) | (rd << 12) | imm12

def encode_movt(rd: int, imm16: int, cond: int = 0xe) -> int:
    imm4 = (imm16 >> 12) & 0xf
    imm12 = imm16 & 0xfff
    return (cond << 28) | 0x03400000 | (imm4 << 16) | (rd << 12) | imm12

def emit_ldr_imm32(insns: list, rd: int, val: int):
    insns.append(encode_movw(rd, val & 0xffff))
    insns.append(encode_movt(rd, (val >> 16) & 0xffff))

# -------------------------------------------------------------------------
# Cave 1: Non-Blocking Boot Status Init (0x53000d60)
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
    insns.append(0xe1a0c00d) # mov  ip, sp
    insns.append(0xe92dd8f0) # push {r4, r5, r6, r7, fp, ip, lr, pc} (32 bytes)
    insns.append(0xe24cb004) # sub  fp, ip, #4
    insns.append(0xe24dd008) # sub  sp, sp, #8

    # 2. Execute original prologue setup: r4 = HOOK1_SINGLETON_VA (0x550b8650)
    emit_ldr_imm32(insns, 4, HOOK1_SINGLETON_VA)

    # 3. Only zero STATUS_VA (0x52a80100..0x52a8011f) on first boot singleton creation ([r4] == 0)
    #    Avoids early-boot SD card I/O contention before C:\ finishes mounting!
    insns.append(0xe5946000) # ldr  r6, [r4]
    insns.append(0xe3560000) # cmp  r6, #0
    branch_to("exit", is_bl=False, cond=0x1) # bne .exit

    emit_ldr_imm32(insns, 3, STATUS_VA)
    insns.append(0xe3a02000) # mov  r2, #0
    insns.append(0xe5832000) # str  r2, [r3, #0x00]
    insns.append(0xe5832004) # str  r2, [r3, #0x04]
    insns.append(0xe5832008) # str  r2, [r3, #0x08]
    insns.append(0xe583200c) # str  r2, [r3, #0x0c]
    insns.append(0xe5832010) # str  r2, [r3, #0x10]
    insns.append(0xe5832014) # str  r2, [r3, #0x14]
    insns.append(0xe5832018) # str  r2, [r3, #0x18]
    insns.append(0xe583201c) # str  r2, [r3, #0x1c]

    mark_label("exit")
    # 4. Branch cleanly to LiveViewProcess3aContoller resume site (0x53676ad4)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, resume_va, is_bl=False))

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
# Cave 2: Stage 3 Pure CEVA-CV LiveView AI AF Pipeline (0x53000e00)
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
    #    Stack alignment: 16 + 32 + 8 = 56 bytes (56 % 8 == 0)
    insns.append(0xe92d000f) # push {r0, r1, r2, r3}
    insns.append(0xe92d47f0) # push {r4, r5, r6, r7, r8, sb, sl, lr}
    insns.append(0xe24dd008) # sub  sp, sp, #8

    # 2. Pure K-3 III Deep Learning Exclusive: Stock LibAHFD (0x53866594) is 100% disabled!
    #    Call DspFactory::GetDspService() -> r0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, GET_DSP_SVC_VA, is_bl=True))

    # 3. m_dspService is at [r0, #4]
    insns.append(0xe5905004) # ldr  r5, [r0, #4] (m_dspService)
    insns.append(0xe3550000) # cmp  r5, #0
    branch_to("clear_and_exit", is_bl=False, cond=0x0) # beq .clear_and_exit

    # 5. Check if DSP is already running (status Bit 2 = 0x4 STARTED!)
    insns.append(0xe5952004) # ldr  r2, [r5, #4]
    insns.append(0xe3120004) # tst  r2, #4 (STARTED)
    branch_to("dsp_is_running", is_bl=False, cond=0x1) # bne .dsp_is_running

    # --- Boot-Guaranteed Retry Counter via constructor-zeroed [r4, #0x13e] ---
    emit_ldr_imm32(insns, 8, STATUS_VA)
    insns.append(0xe5d4213e) # ldrb r2, [r4, #0x13e]
    insns.append(0xe3520000) # cmp  r2, #0
    branch_to("counter_inited", is_bl=False, cond=0x1) # bne .counter_inited
    insns.append(0xe5882000) # str  r2, [r8, #0] (zero STATUS_VA + 0)
    insns.append(0xe5882004) # str  r2, [r8, #4] (zero STATUS_VA + 4)
    insns.append(0xe3a02001) # mov  r2, #1
    insns.append(0xe5c4213e) # strb r2, [r4, #0x13e] (mark initialized for this boot!)

    mark_label("counter_inited")
    # Wait 20 frames (~0.65s) for SD card mount, then retry every 64 frames (~2.1s)
    # up to 1920 frames (~65s = 30 attempts) while unlocking mutex (r4 + 0x48)!
    insns.append(0xe5982004) # ldr  r2, [r8, #4]
    insns.append(0xe2822001) # add  r2, r2, #1
    insns.append(0xe5882004) # str  r2, [r8, #4]
    insns.append(0xe2523014) # subs r3, r2, #20
    branch_to("clear_and_exit", is_bl=False, cond=0x3) # blo .clear_and_exit (wait first 20 frames)
    insns.append(0xe3530d1e) # cmp  r3, #0x780 (1920 frames = 30 retries)
    branch_to("clear_and_exit", is_bl=False, cond=0x2) # bhs .clear_and_exit
    insns.append(0xe313003f) # tst  r3, #0x3f  (retry on ticks 20, 84, 148, ...)
    branch_to("clear_and_exit", is_bl=False, cond=0x1) # bne .clear_and_exit

    # Unlock FaceDetectionService mutex (r4 + 0x48) BEFORE Storage/SD I/O
    insns.append(0xe2840048) # add  r0, r4, #0x48
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, UNLOCK_MUTEX_VA, is_bl=True))

    # Dual-Drive Polling: Try Drive 'C' (SD Card) first, then Drive 'I' (Internal 53GB Flash)
    insns.append(0xe1a00005) # mov  r0, r5 (m_dspService)
    insns.append(0xe3a01043) # mov  r1, #0x43 ('C' -> Drive C:\ on SD Card)
    insns.append(0xe3a02000) # mov  r2, #0
    insns.append(0xe3a03000) # mov  r3, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, LOAD_DSP_VA, is_bl=True))

    insns.append(0xe3500000) # cmp  r0, #0 (1 is SUCCESS in DspService::LoadDsp!)
    branch_to("cave2_load_ok", is_bl=False, cond=0x1) # bne .cave2_load_ok

    # If Drive 'C' failed, try Drive 'I' (Built-in 53GB Memory)
    insns.append(0xe1a00005) # mov  r0, r5 (m_dspService)
    insns.append(0xe3a01049) # mov  r1, #0x49 ('I' -> Drive I:\ on Built-in 53GB Flash)
    insns.append(0xe3a02000) # mov  r2, #0
    insns.append(0xe3a03000) # mov  r3, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, LOAD_DSP_VA, is_bl=True))

    insns.append(0xe3500000) # cmp  r0, #0
    branch_to("cave2_load_ok", is_bl=False, cond=0x1) # bne .cave2_load_ok

    # Both C:\ and I:\ failed: clear false LOADED bit and return
    insns.append(0xe3a02000) # mov  r2, #0
    insns.append(0xe5852004) # str  r2, [r5, #4] (clear false LOADED bit)
    # Re-lock FaceDetectionService mutex (r4 + 0x48) before returning to clear_and_exit
    insns.append(0xe2840048) # add  r0, r4, #0x48
    emit_ldr_imm32(insns, 3, MUTEX_COOKIE_VA)
    insns.append(0xe5931000) # ldr  r1, [r3]
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, LOCK_MUTEX_VA, is_bl=True))
    branch_to("clear_and_exit", is_bl=False, cond=0xe) # b .clear_and_exit

    mark_label("cave2_load_ok")
    # Start CEVA-CV DSP at native K-3 III base address 0x52100000!
    insns.append(0xe1a00005) # mov  r0, r5
    emit_ldr_imm32(insns, 1, 0x52100000) # mov  r1, #0x52100000 (DSP boot address)
    insns.append(0xe3a02000) # mov  r2, #0
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, START_DSP_VA, is_bl=True))

    emit_ldr_imm32(insns, 8, STATUS_VA)
    emit_ldr_imm32(insns, 1, 0x41564543) # 'CEVA'
    insns.append(0xe5881000) # str  r1, [r8, #0]
    # Re-lock FaceDetectionService mutex (r4 + 0x48)
    insns.append(0xe2840048) # add  r0, r4, #0x48
    emit_ldr_imm32(insns, 3, MUTEX_COOKIE_VA)
    insns.append(0xe5931000) # ldr  r1, [r3]
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, LOCK_MUTEX_VA, is_bl=True))

    # Verify STARTED bit (0x4) is now set
    insns.append(0xe5952004) # ldr  r2, [r5, #4]
    insns.append(0xe3120004) # tst  r2, #4
    branch_to("clear_and_exit", is_bl=False, cond=0x0) # beq .clear_and_exit

    # Yield current frame to let CEVA DSP core complete boot sequence cleanly
    branch_to("clear_and_exit", is_bl=False, cond=0xe) # b .clear_and_exit

    mark_label("dsp_is_running")

    # 6. Extract Completed Live View Planar YUV420 Frame from Ahip Ring Buffer:
    #    Official v1.20 AhipService::WaitFrame (0x5311a628):
    #    ring_entry = *(*(0x55081350 + 0x424)) (Y @ +0x04, U @ +0x08, V @ +0x0c)
    #    Fallback 1:       ring_entry = *(*(0x55081350 + 0x434))
    #    Fallback 2:       DMA node *(0x55081350 + 0x1c) / *(0x55081350 + 0x14)
    emit_ldr_imm32(insns, 3, AHIP_DRV_BASE_VA)
    insns.append(0xe593900c) # ldr  sb, [r3, #0x0c] (Ahip YUV slot byte size)
    insns.append(0xe5932424) # ldr  r2, [r3, #0x424] (primary completed ring entry in v1.20)
    insns.append(0xe3520000) # cmp  r2, #0
    insns.append(0x15922000) # ldrne r2, [r2, #0]
    insns.append(0xe3520000) # cmp  r2, #0
    branch_to("try_ahip_434", is_bl=False, cond=0x0) # beq .try_ahip_434
    insns.append(0xe5926004) # ldr  r6, [r2, #0x04] (Y_plane)
    insns.append(0xe3560000) # cmp  r6, #0
    branch_to("try_ahip_434", is_bl=False, cond=0x0) # beq .try_ahip_434
    insns.append(0xe5927008) # ldr  r7, [r2, #0x08] (U_plane)
    insns.append(0xe592800c) # ldr  r8, [r2, #0x0c] (V_plane)
    branch_to("have_yuv_ptrs", is_bl=False, cond=0xe) # b .have_yuv_ptrs

    mark_label("try_ahip_434")
    insns.append(0xe5932434) # ldr  r2, [r3, #0x434] (fallback ring entry)
    insns.append(0xe3520000) # cmp  r2, #0
    insns.append(0x15922000) # ldrne r2, [r2, #0]
    insns.append(0xe3520000) # cmp  r2, #0
    branch_to("try_ahip_legacy", is_bl=False, cond=0x0) # beq .try_ahip_legacy
    insns.append(0xe5926004) # ldr  r6, [r2, #0x04] (Y_plane)
    insns.append(0xe3560000) # cmp  r6, #0
    branch_to("try_ahip_legacy", is_bl=False, cond=0x0) # beq .try_ahip_legacy
    insns.append(0xe5927008) # ldr  r7, [r2, #0x08] (U_plane)
    insns.append(0xe592800c) # ldr  r8, [r2, #0x0c] (V_plane)
    branch_to("have_yuv_ptrs", is_bl=False, cond=0xe) # b .have_yuv_ptrs

    mark_label("try_ahip_legacy")
    insns.append(0xe593201c) # ldr  r2, [r3, #0x1c]
    insns.append(0xe3520000) # cmp  r2, #0
    insns.append(0x05932014) # ldreq r2, [r3, #0x14]
    insns.append(0xe3520000) # cmp  r2, #0
    branch_to("clear_and_exit", is_bl=False, cond=0x0) # beq .clear_and_exit
    insns.append(0xe592600c) # ldr  r6, [r2, #0x0c] (Y_plane)
    insns.append(0xe3560000) # cmp  r6, #0
    branch_to("clear_and_exit", is_bl=False, cond=0x0) # beq .clear_and_exit
    insns.append(0xe5927010) # ldr  r7, [r2, #0x10] (U_plane)
    insns.append(0xe5928014) # ldr  r8, [r2, #0x14] (V_plane)

    mark_label("have_yuv_ptrs")
    # 7. Invalidate ARM D-Cache on the completed YUV420 slot BEFORE CPU downsampling!
    insns.append(0xe3590000) # cmp  sb, #0
    insns.append(0x03a09802) # moveq sb, #0x20000 (default 128 KB if size unset)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, GET_CACHE_MGR_VA, is_bl=True))
    insns.append(0xe1a01006) # mov  r1, r6 (Y_plane base)
    insns.append(0xe1a02009) # mov  r2, sb (slot byte size covering Y+U+V)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, INVAL_DCACHE_VA, is_bl=True))

    # 8. Downsample Planar YUV420 (fd_w x fd_h) into 3-plane 56x56 (stride 64) RGB @ 0x52940000
    insns.append(0xe1d4cfbc) # ldrh ip, [r4, #0xfc] (fd_w, default 320)
    insns.append(0xe35c0000) # cmp  ip, #0
    insns.append(encode_movw(12, 320, cond=0x0)) # moveq ip, #320
    insns.append(0xe1d49fbe) # ldrh sb, [r4, #0xfe] (fd_h, default 212)
    insns.append(0xe3590000) # cmp  sb, #0
    insns.append(encode_movw(9, 212, cond=0x0))  # moveq sb, #212

    insns.append(0xe1a00006) # mov  r0, r6 (Y_plane)
    emit_ldr_imm32(insns, 1, TENSOR_IN_VA) # r1 = 0x52940000
    insns.append(0xe1a02007) # mov  r2, r7 (U_plane)
    insns.append(0xe1a03008) # mov  r3, r8 (V_plane)
    branch_to("sub_downsample", is_bl=True, cond=0xe)

    # 9. Flush ARM D-Cache for work_buf (0x52940000, size 0xee00) before DSP DMA
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, GET_CACHE_MGR_VA, is_bl=True))
    emit_ldr_imm32(insns, 1, TENSOR_IN_VA)
    insns.append(0xe3a02cee) # mov  r2, #0xee00
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, FLUSH_DCACHE_VA, is_bl=True))

    # 10. Prepare 10-word (0x28 B) DspCommandParameter @ TENSOR_MSG_VA (0x52a80000)
    emit_ldr_imm32(insns, 7, TENSOR_MSG_VA)   # r7 = 0x52a80000
    emit_ldr_imm32(insns, 0, DESC_TABLE_VA)   # msg[+0x00] = vtable (0x53001440)
    insns.append(0xe5870000)                  # str  r0, [r7, #0x00]
    insns.append(0xe3a00000)                  # mov  r0, #0
    insns.append(0xe5870004)                  # str  r0, [r7, #0x04] (seq_id | msg_id)
    insns.append(0xe5870020)                  # str  r0, [r7, #0x20] (0)
    insns.append(0xe5870024)                  # str  r0, [r7, #0x24] (0)
    emit_ldr_imm32(insns, 0, TENSOR_IN_VA)    # msg[+0x08] = work_buf (0x52940000)
    insns.append(0xe5870008)                  # str  r0, [r7, #0x08]
    emit_ldr_imm32(insns, 0, HEAD_WEIGHTS_VA) # msg[+0x0c] = 0x52a057c0 (461,408 B)
    insns.append(0xe587000c)                  # str  r0, [r7, #0x0c]
    emit_ldr_imm32(insns, 0, MEAN_WEIGHTS_VA) # msg[+0x10] = 0x52960000 (677,824 B)
    insns.append(0xe5870010)                  # str  r0, [r7, #0x10]
    emit_ldr_imm32(insns, 0, 0x00080808)      # msg[+0x14] = 0x00080808
    insns.append(0xe5870014)                  # str  r0, [r7, #0x14]
    emit_ldr_imm32(insns, 0, 0x0c0c0c0a)      # msg[+0x18] = 0x0c0c0c0a
    insns.append(0xe5870018)                  # str  r0, [r7, #0x18]
    emit_ldr_imm32(insns, 0, 0x0c0c0c0c)      # msg[+0x1c] = 0x0c0c0c0c
    insns.append(0xe587001c)                  # str  r0, [r7, #0x1c]

    # 11. Setup seq pointer @ STATUS_VA + 8 (0x52a80108)
    emit_ldr_imm32(insns, 6, STATUS_VA + 8)
    insns.append(0xe3a00000) # mov  r0, #0
    insns.append(0xe1c600b0) # strh r0, [r6]

    # 12. Call DspService::SendMessage(m_dspService, &seq, msg_id=4, timeout=100, &msg)
    insns.append(0xe1a00005) # mov  r0, r5 (svc)
    insns.append(0xe1a01006) # mov  r1, r6 (&seq)
    insns.append(0xe3a02004) # mov  r2, #4 (msg_id = 4)
    insns.append(0xe3a03064) # mov  r3, #100 (timeout = 100 ms)
    insns.append(0xe58d7000) # str  r7, [sp] (&msg)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, SEND_MSG_VA, is_bl=True))
    insns.append(0xe3500000) # cmp  r0, #0
    branch_to("clear_and_exit", is_bl=False, cond=0x1) # bne .clear_and_exit

    # 13. Call DspService::WaitMessage(m_dspService, seq, timeout=100, &resp)
    insns.append(0xe1a00005) # mov  r0, r5 (svc)
    insns.append(0xe1d610b0) # ldrh r1, [r6] (seq)
    insns.append(0xe3a02064) # mov  r2, #100 (timeout = 100 ms)
    insns.append(0xe1a03007) # mov  r3, r7 (&resp)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, WAIT_MSG_VA, is_bl=True))
    insns.append(0xe3500000) # cmp  r0, #0
    branch_to("clear_and_exit", is_bl=False, cond=0x1) # bne .clear_and_exit

    # 14. Invalidate ARM D-Cache for output heatmap @ 0x52b09a00 (size 0x5400 = 3 * 0x1c00)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, GET_CACHE_MGR_VA, is_bl=True))
    emit_ldr_imm32(insns, 1, TENSOR_OUT_VA)
    insns.append(0xe3a02b15) # mov  r2, #0x5400 (0x15 << 10)
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, INVAL_DCACHE_VA, is_bl=True))

    # 15. Scan 3-channel 56x56 (stride 64) int16_t output heatmap @ 0x52b09a00
    emit_ldr_imm32(insns, 0, TENSOR_OUT_VA)
    branch_to("sub_find_peak", is_bl=True, cond=0xe) # returns r0: max_diff, r1: col, r2: row, r3: area_cnt

    # 16. Noise-immune detection gate:
    #     Require max_diff >= 3 (+0.75 logit) AND area_cnt >= 2
    #     If conditions not met, route to heartbeat_box (diagnostic box in lower-right)
    insns.append(0xe3500003) # cmp  r0, #3
    branch_to("heartbeat_box", is_bl=False, cond=0xb) # blt .heartbeat_box
    insns.append(0xe3530002) # cmp  r3, #2
    branch_to("heartbeat_box", is_bl=False, cond=0xb) # blt .heartbeat_box

    mark_label("subject_detected")
    # --- K-3 III Deep Learning Subject Detected! ---
    # Dynamic Q8 bounding box size from foreground segmentation area:
    # box_size = clamp(0x2800 + (area_cnt << 6), 0x2800, 0x5c00) (40..92 px in 320w)
    insns.append(0xe3a05b0a) # mov  r5, #0x2800
    insns.append(0xe0855303) # add  r5, r5, r3, lsl #6
    insns.append(0xe3550b17) # cmp  r5, #0x5c00
    insns.append(0xc3a05b17) # movgt r5, #0x5c00

    # Normalize confidence score into [30..100] range
    insns.append(0xe280901e) # add  sb, r0, #30
    insns.append(0xe3590064) # cmp  sb, #100
    insns.append(0xc3a09064) # movgt sb, #100

    # Aspect-ratio-exact Q8 coordinates matching UpdateFaceDetectionResult:
    # cx = (best_col * fd_w * 73) >> 4, cy = (best_row * fd_h * 73) >> 4
    insns.append(0xe1d43fbc)                     # ldrh r3, [r4, #0xfc] (fd_w)
    insns.append(0xe3530000)                     # cmp  r3, #0
    insns.append(encode_movw(3, 320, cond=0x0))  # moveq r3, #320
    insns.append(0xe0070391)                     # mul  r7, r1, r3 (col * fd_w)
    insns.append(0xe3a03049)                     # mov  r3, #73
    insns.append(0xe0070793)                     # mul  r7, r3, r7
    insns.append(0xe1a07227)                     # lsr  r7, r7, #4 (cx)

    insns.append(0xe1d43fbe)                     # ldrh r3, [r4, #0xfe] (fd_h)
    insns.append(0xe3530000)                     # cmp  r3, #0
    insns.append(encode_movw(3, 212, cond=0x0))  # moveq r3, #212
    insns.append(0xe00c0392)                     # mul  ip, r2, r3 (row * fd_h)
    insns.append(0xe3a03049)                     # mov  r3, #73
    insns.append(0xe00c0c93)                     # mul  ip, r3, ip
    insns.append(0xe1a0c22c)                     # lsr  ip, ip, #4 (cy)

    # left = max(0, cx - (box_size >> 1)), top = max(0, cy - (box_size >> 1))
    insns.append(0xe05780a5) # subs r8, r7, r5, lsr #1
    insns.append(0x43a08000) # movmi r8, #0 (left)
    insns.append(0xe05c60a5) # subs r6, ip, r5, lsr #1
    insns.append(0x43a06000) # movmi r6, #0 (top)

    branch_to("populate_face_struct", is_bl=False, cond=0xe) # b .populate_face_struct

    mark_label("heartbeat_box")
    # Heartbeat diagnostic box in lower-right quadrant:
    # left = 240 px (0x0f000 in Q8), top = 140 px (0x08c00 in Q8)
    # box_size = 32 px (0x02000 in Q8, > 0x1e00 SelectMainFace threshold)
    # score = 99
    # cx = 256 px (0x10000 in Q8), cy = 156 px (0x09c00 in Q8)
    emit_ldr_imm32(insns, 8, 0x0f000) # r8 = left (240 px in Q8)
    emit_ldr_imm32(insns, 6, 0x08c00) # r6 = top  (140 px in Q8)
    insns.append(0xe3a05b08)          # mov  r5, #0x2000 (box_size = 32 px = 0x2000)
    insns.append(0xe3a09063)          # mov  sb, #99     (confidence = 99)
    emit_ldr_imm32(insns, 7, 0x10000) # r7 = cx (256 px in Q8)
    emit_ldr_imm32(insns, 12, 0x09c00) # ip = cy (156 px in Q8)
    # falls through into populate_face_struct

    mark_label("populate_face_struct")
    # Save telemetry @ STATUS_VA + 12 (0x52c0010c)
    emit_ldr_imm32(insns, 3, STATUS_VA + 12)
    insns.append(0xe5839000) # str  sb, [r3, #0]  (peak_score)
    insns.append(0xe5838004) # str  r8, [r3, #4]  (last_x Q8)
    insns.append(0xe5836008) # str  r6, [r3, #8]  (last_y Q8)
    insns.append(0xe583500c) # str  r5, [r3, #12] (last_w Q8)
    insns.append(0xe5835010) # str  r5, [r3, #16] (last_h Q8)

    # Fetch m_DetectedFaceResult pointer from [r4, #0x78]
    insns.append(0xe594a078) # ldr  sl, [r4, #0x78] (m_DetectedFaceResult)
    insns.append(0xe35a0000) # cmp  sl, #0
    branch_to("clear_and_exit", is_bl=False, cond=0x0) # beq .clear_and_exit

    # First zero all 100 bytes (25 words) of m_DetectedFaceResult[0] @ sl
    insns.append(0xe3a00000) # mov  r0, #0
    insns.append(0xe3a01000) # mov  r1, #0
    mark_label("zero_face_loop")
    insns.append(0xe78a0001) # str  r0, [sl, r1]
    insns.append(0xe2811004) # add  r1, r1, #4
    insns.append(0xe3510064) # cmp  r1, #100
    branch_to("zero_face_loop", is_bl=False, cond=0xb) # blt .zero_face_loop

    # Populate Face Struct 0 (100 bytes @ sl):
    # +0x00..+0x0c: face_rect (left, top, width, height)
    insns.append(0xe58a8000) # str  r8, [sl, #0x00] (face.left)
    insns.append(0xe58a6004) # str  r6, [sl, #0x04] (face.top)
    insns.append(0xe58a5008) # str  r5, [sl, #0x08] (face.width)
    insns.append(0xe58a500c) # str  r5, [sl, #0x0c] (face.height)

    # +0x10..+0x2c: Dynamic eye target boxes within face bounds
    # eye_w = box_size >> 2 (e.g. 0x0a00 = 10 px)
    # eye_off = box_size >> 3 (e.g. 0x0500 = 5 px)
    insns.append(0xe1a03125) # lsr  r3, r5, #2 (eye_w = box_size >> 2)
    insns.append(0xe1a021a5) # lsr  r2, r5, #3 (eye_off = box_size >> 3)

    # eye1 (right eye in image / camera left): cx - eye_off, cy - eye_off
    insns.append(0xe0470002) # sub  r0, r7, r2 (eye1_cx = cx - eye_off)
    insns.append(0xe05000a3) # subs r0, r0, r3, lsr #1 (eye1.left)
    insns.append(0x43a00000) # movmi r0, #0
    insns.append(0xe04c1002) # sub  r1, ip, r2 (eye1_cy = cy - eye_off)
    insns.append(0xe05110a3) # subs r1, r1, r3, lsr #1 (eye1.top)
    insns.append(0x43a01000) # movmi r1, #0
    insns.append(0xe58a0010) # str  r0, [sl, #0x10] (eye1.left)
    insns.append(0xe58a1014) # str  r1, [sl, #0x14] (eye1.top)
    insns.append(0xe58a3018) # str  r3, [sl, #0x18] (eye1.width)
    insns.append(0xe58a301c) # str  r3, [sl, #0x1c] (eye1.height)

    # eye2 (left eye in image / camera right): cx + eye_off, cy - eye_off
    insns.append(0xe0870002) # add  r0, r7, r2 (eye2_cx = cx + eye_off)
    insns.append(0xe05000a3) # subs r0, r0, r3, lsr #1 (eye2.left)
    insns.append(0x43a00000) # movmi r0, #0
    insns.append(0xe58a0020) # str  r0, [sl, #0x20] (eye2.left)
    insns.append(0xe58a1024) # str  r1, [sl, #0x24] (eye2.top)
    insns.append(0xe58a3028) # str  r3, [sl, #0x28] (eye2.width)
    insns.append(0xe58a302c) # str  r3, [sl, #0x2c] (eye2.height)

    # +0x4c: confidence score, +0x50: face_id
    # Complete SelectMainFace & EVENT 0x04070001 State Seeding:
    # Must seed [r4, #0x100..0x10c] = (left + w/2, top + h/2, w, h)
    # in addition to [r4, #0x118..0x11c], [r4, #0x121] = 0, [r4, #0x124] = 0, and
    # [r4, #0x13c] = 1 so SelectMainFace never rejects cold-start coordinates!
    insns.append(0xe58a904c) # str  sb, [sl, #0x4c] (confidence)
    insns.append(0xe5941118) # ldr  r1, [r4, #0x118] (previous tracked face_id)
    insns.append(0xe3510000) # cmp  r1, #0
    insns.append(0x03a01001) # moveq r1, #1
    insns.append(0xe58a1050) # str  r1, [sl, #0x50]  (face_id)
    insns.append(0xe08820a5) # add  r2, r8, r5, lsr #1 (rect_cx = left + box_size/2)
    insns.append(0xe5842100) # str  r2, [r4, #0x100] (prev_main_face_cx)
    insns.append(0xe08620a5) # add  r2, r6, r5, lsr #1 (rect_cy = top + box_size/2)
    insns.append(0xe5842104) # str  r2, [r4, #0x104] (prev_main_face_cy)
    insns.append(0xe5845108) # str  r5, [r4, #0x108] (prev_main_face_w  = box_size)
    insns.append(0xe584510c) # str  r5, [r4, #0x10c] (prev_main_face_h  = box_size)
    insns.append(0xe5841118) # str  r1, [r4, #0x118] (seed active tracked face_id)
    insns.append(0xe584111c) # str  r1, [r4, #0x11c] (seed primary tracked face_id)
    insns.append(0xe3a02001) # mov  r2, #1
    insns.append(0xe5c4213c) # strb r2, [r4, #0x13c] (mark tracking active = 1)
    insns.append(0xe3a00000) # mov  r0, #0
    insns.append(0xe5c40121) # strb r0, [r4, #0x121] (clear is_stabilizing = 0)
    insns.append(0xe2843f49) # add  r3, r4, #0x124
    insns.append(0xe1c300b0) # strh r0, [r3]         (lost_frames_counter = 0)

    # Set detected face_count = 1 and main_face_idx = 0 in FdLibraryService ([r4, #0x38])
    insns.append(0xe5943038) # ldr  r3, [r4, #0x38] (FdLibraryService)
    insns.append(0xe3530000) # cmp  r3, #0
    branch_to("clear_and_exit", is_bl=False, cond=0x0) # beq .clear_and_exit
    insns.append(0xe3a01001) # mov  r1, #1
    insns.append(0xe5831030) # str  r1, [r3, #0x30] (face_count = 1)
    insns.append(0xe3a01000) # mov  r1, #0
    insns.append(0xe5831034) # str  r1, [r3, #0x34] (main_face_idx = 0)
    branch_to("ai_done", is_bl=False, cond=0xe) # b .ai_done

    mark_label("clear_and_exit")
    # Clear detected face_count = 0 in FdLibraryService ([r4, #0x38])
    insns.append(0xe5943038) # ldr  r3, [r4, #0x38] (FdLibraryService)
    insns.append(0xe3530000) # cmp  r3, #0
    insns.append(0x13a01000) # movne r1, #0
    insns.append(0x15831030) # strne r1, [r3, #0x30] (face_count = 0)
    # falls through to ai_done

    mark_label("ai_done")
    # Restore stack and resume at HOOK2_AI_RESUME_VA (0x538963ac):
    # Unlocks mutex r6 and calls UpdateFaceDetectionResult(r4) (0x538959f4)
    insns.append(0xe28dd008) # add  sp, sp, #8
    insns.append(0xe8bd47f0) # pop  {r4, r5, r6, r7, r8, sb, sl, lr}
    insns.append(0xe8bd000f) # pop  {r0, r1, r2, r3}
    curr_pc = cave_va + len(insns) * 4
    insns.append(encode_arm_branch(curr_pc, HOOK2_AI_RESUME_VA, is_bl=False))

    # ---------------------------------------------------------------------
    # Subroutine: Aspect-Ratio-Aware Downsampler + 2.0x Backlit Shadow Recovery
    # ---------------------------------------------------------------------
    mark_label("sub_downsample")
    insns.append(0xe92d47f0) # push {r4, r5, r6, r7, r8, sb, sl, lr}
    insns.append(0xe3a04000) # mov  r4, #0 (row = 0)
    mark_label("loop_ds_row")
    insns.append(0xe59d6014) # ldr  r6, [sp, #0x14] (fd_h)
    insns.append(0xe0060694) # mul  r6, r4, r6      (row * fd_h)
    insns.append(0xe0866186) # add  r6, r6, r6, lsl #3 (row * fd_h * 9)
    insns.append(0xe1a064a6) # lsr  r6, r6, #9      (src_y)
    insns.append(0xe0270c96) # mla  r7, r6, ip, r0  (y_row_ptr = Y_plane + src_y * fd_w)
    insns.append(0xe0818304) # add  r8, r1, r4, lsl #6 (dst_row_ptr = r1 + row * 64)

    insns.append(0xe3a05000) # mov  r5, #0 (col = 0)
    mark_label("loop_ds_col")
    insns.append(0xe0060c95) # mul  r6, r5, ip      (col * fd_w)
    insns.append(0xe0866186) # add  r6, r6, r6, lsl #3 (col * fd_w * 9)
    insns.append(0xe1a064a6) # lsr  r6, r6, #9      (src_x)
    insns.append(0xe7d79006) # ldrb sb, [r7, r6]    (Y_raw = y_row_ptr[src_x])

    # C0-Continuous 2.0x Backlit Shadow Recovery Curve:
    #   Y < 48       -> Y' = 2 * Y          (doubles crushed backlit face contrast!)
    #   48 <= Y < 144 -> Y' = 72 + (Y >> 1) (smooth shoulder into k3_mean ~115)
    #   Y >= 144     -> Y' = Y              (preserves bright sky/window highlights)
    insns.append(0xe3590030) # cmp   sb, #48
    insns.append(0x31a09089) # lsllo sb, sb, #1
    branch_to("luma_lifted", is_bl=False, cond=0x3) # blo .luma_lifted
    insns.append(0xe3590090) # cmp   sb, #144
    insns.append(0x33a0a048) # movlo sl, #72
    insns.append(0x308a90a9) # addlo sb, sl, sb, lsr #1
    mark_label("luma_lifted")

    # Default R = G = B = Y' if U/V planes are null
    insns.append(0xe1a0a009) # mov  sl, sb (G = Y')
    insns.append(0xe1a0e009) # mov  lr, sb (B = Y')
    insns.append(0xe3520000) # cmp  r2, #0
    branch_to("store_rgb", is_bl=False, cond=0x0) # beq .store_rgb
    insns.append(0xe3530000) # cmp  r3, #0
    branch_to("store_rgb", is_bl=False, cond=0x0) # beq .store_rgb

    # Compute UV offset: uv_y = (row * fd_h * 9) >> 10, uv_w = fd_w >> 1, uv_x = src_x >> 1
    insns.append(0xe59da014) # ldr  sl, [sp, #0x14] (fd_h)
    insns.append(0xe00a0a94) # mul  sl, r4, sl
    insns.append(0xe08aa18a) # add  sl, sl, sl, lsl #3
    insns.append(0xe1a0a52a) # lsr  sl, sl, #10 (uv_y = src_y >> 1)
    insns.append(0xe1a0e0ac) # lsr  lr, ip, #1  (uv_w = fd_w >> 1)
    insns.append(0xe00a0a9e) # mul  sl, lr, sl  (uv_y * uv_w)
    insns.append(0xe08aa0a6) # add  sl, sl, r6, lsr #1 (uv_off)
    insns.append(0xe7d2e00a) # ldrb lr, [r2, sl] (U_raw)
    insns.append(0xe7d3a00a) # ldrb sl, [r3, sl] (V_raw)
    insns.append(0xe24ee080) # sub  lr, lr, #128 (u = U_raw - 128)
    insns.append(0xe24aa080) # sub  sl, sl, #128 (v = V_raw - 128)

    # G = usat8(Y' - (u >> 2) - (v >> 1))
    insns.append(0xe049614e) # sub  r6, sb, lr, asr #2
    insns.append(0xe04660ca) # sub  r6, r6, sl, asr #1
    insns.append(0xe6e86016) # usat r6, #8, r6 (G clamped to 0..255)

    # B = usat8(Y' + 1.5*u)
    insns.append(0xe08ee0ce) # add  lr, lr, lr, asr #1 (1.5*u)
    insns.append(0xe089e00e) # add  lr, sb, lr (B = Y' + 1.5*u)
    insns.append(0xe6e8e01e) # usat lr, #8, lr (B clamped to 0..255)

    # R = usat8(Y' + 1.5*v)
    insns.append(0xe08aa0ca) # add  sl, sl, sl, asr #1 (1.5*v)
    insns.append(0xe089900a) # add  sb, sb, sl (R = Y' + 1.5*v)
    insns.append(0xe6e89019) # usat sb, #8, sb (R clamped to 0..255)
    insns.append(0xe1a0a006) # mov  sl, r6 (G)

    mark_label("store_rgb")
    # Write R to Plane 0 (+0x0000), G to Plane 1 (+0x0e00), B to Plane 2 (+0x1c00)
    insns.append(0xe0886005) # add  r6, r8, r5 (dst_ptr = dst_row_ptr + col)
    insns.append(0xe5c69000) # strb sb, [r6, #0]     (Plane 0: R)
    insns.append(0xe5c6ae00) # strb sl, [r6, #0xe00] (Plane 1: G)
    insns.append(0xe2866c0e) # add  r6, r6, #0xe00
    insns.append(0xe5c6ee00) # strb lr, [r6, #0xe00] (Plane 2: B @ +0x1c00)

    insns.append(0xe2855001) # add  r5, r5, #1 (col++)
    insns.append(0xe3550038) # cmp  r5, #56
    branch_to("loop_ds_col", is_bl=False, cond=0xb) # blt .loop_ds_col

    # Zero the 8 padding bytes (cols 56..63) of this row in all 3 planes
    insns.append(0xe3a06000) # mov  r6, #0
    insns.append(0xe5886038) # str  r6, [r8, #0x38]
    insns.append(0xe588603c) # str  r6, [r8, #0x3c]
    insns.append(0xe2888c0e) # add  r8, r8, #0xe00
    insns.append(0xe5886038) # str  r6, [r8, #0x38]
    insns.append(0xe588603c) # str  r6, [r8, #0x3c]
    insns.append(0xe2888c0e) # add  r8, r8, #0xe00
    insns.append(0xe5886038) # str  r6, [r8, #0x38]
    insns.append(0xe588603c) # str  r6, [r8, #0x3c]

    insns.append(0xe2844001) # add  r4, r4, #1 (row++)
    insns.append(0xe3540038) # cmp  r4, #56
    branch_to("loop_ds_row", is_bl=False, cond=0xb) # blt .loop_ds_row
    insns.append(0xe8bd87f0) # pop  {r4, r5, r6, r7, r8, sb, sl, pc}

    # ---------------------------------------------------------------------
    # Subroutine: Head-Priority 3-Channel 56x56 (Stride 64) int16_t Heatmap Scanner
    # Input:  r0 = TENSOR_OUT_VA (0x52b09a00)
    # Output: r0 = max_diff, r1 = best_col (0..55), r2 = best_row (0..55), r3 = area_cnt
    # ---------------------------------------------------------------------
    mark_label("sub_find_peak")
    insns.append(0xe92d4ff0) # push {r4, r5, r6, r7, r8, sb, sl, fp, lr}
    insns.append(0xe3a01000) # mov  r1, #0  (max_diff = 0)
    insns.append(0xe3a0201c) # mov  r2, #28 (best_col = 28)
    insns.append(0xe3a0301c) # mov  r3, #28 (best_row = 28)
    insns.append(0xe3a0b000) # mov  fp, #0  (area_cnt = 0)
    insns.append(0xe2804b07) # add  r4, r0, #0x1c00 (Ch1 base: Human/Head/Face)
    insns.append(0xe2805b0e) # add  r5, r0, #0x3800 (Ch2 base: Animal/Body/Subject)
    insns.append(0xe3a06000) # mov  r6, #0 (row = 0)

    mark_label("loop_peak_row")
    insns.append(0xe3a07000) # mov  r7, #0 (col = 0)
    mark_label("loop_peak_col")
    insns.append(0xe0878306) # add  r8, r7, r6, lsl #6 (idx = row * 64 + col)
    insns.append(0xe1a08088) # lsl  r8, r8, #1         (byte_off = idx * 2)
    insns.append(0xe19090f8) # ldrsh sb, [r0, r8]      (bg = Ch0[idx])
    insns.append(0xe194a0f8) # ldrsh sl, [r4, r8]      (c1 = Ch1[idx])
    insns.append(0xe195c0f8) # ldrsh ip, [r5, r8]      (c2 = Ch2[idx])
    insns.append(0xe24cc002) # sub  ip, ip, #2         (head priority: c2 - 0.5 logit)
    insns.append(0xe15c000a) # cmp  ip, sl
    insns.append(0xc1a0a00c) # movgt sl, ip            (fg = max(c1, c2 - 2))
    insns.append(0xe04aa009) # sub  sl, sl, sb         (diff = fg - bg)
    insns.append(0xe35a0002) # cmp  sl, #2
    insns.append(0xa28bb001) # addge fp, fp, #1        (area_cnt += (diff >= 2))
    insns.append(0xe15a0001) # cmp  sl, r1
    branch_to("skip_peak_update", is_bl=False, cond=0xd) # ble .skip_peak_update
    insns.append(0xe1a0100a) # mov  r1, sl (max_diff = diff)
    insns.append(0xe1a02007) # mov  r2, r7 (best_col = col)
    insns.append(0xe1a03006) # mov  r3, r6 (best_row = row)
    mark_label("skip_peak_update")
    insns.append(0xe2877001) # add  r7, r7, #1 (col++)
    insns.append(0xe3570038) # cmp  r7, #56
    branch_to("loop_peak_col", is_bl=False, cond=0xb) # blt .loop_peak_col
    insns.append(0xe2866001) # add  r6, r6, #1 (row++)
    insns.append(0xe3560038) # cmp  r6, #56
    branch_to("loop_peak_row", is_bl=False, cond=0xb) # blt .loop_peak_row

    insns.append(0xe1a00001) # mov  r0, r1 (max_diff)
    insns.append(0xe1a01002) # mov  r1, r2 (best_col)
    insns.append(0xe1a02003) # mov  r2, r3 (best_row)
    insns.append(0xe1a0300b) # mov  r3, fp (area_cnt)
    insns.append(0xe8bd8ff0) # pop  {r4, r5, r6, r7, r8, sb, sl, fp, pc}

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
# Splicer: Injects Dual Caves, Trampolines & Neural Descriptors
# -------------------------------------------------------------------------
def apply_stage3_patch(rtos_binary: bytearray, rtos_payload_offset: int = RTOS_PAYLOAD_START) -> dict:
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
    assert CAVE1_VA + len(cave1_payload) <= CAVE2_VA, f"Cave 1 ({len(cave1_payload)} B) overlaps Cave 2!"
    assert CAVE2_VA + len(cave2_payload) <= DESC_TABLE_VA, f"Cave 2 ({len(cave2_payload)} B) overlaps Desc Table!"
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
    expected_hook2 = bytes.fromhex("380094e57b40ffeb") # ldr r0, [r4, #0x38]; bl #0x53866594
    assert orig_hook2_bytes == expected_hook2, f"Unexpected hook 2 bytes: {orig_hook2_bytes.hex()} (expected {expected_hook2.hex()})"

    # Apply caves and trampolines
    rtos_binary[cave1_off : cave1_off + len(cave1_payload)] = cave1_payload
    rtos_binary[tramp1_off : tramp1_off + len(tramp1_payload)] = tramp1_payload

    rtos_binary[cave2_off : cave2_off + len(cave2_payload)] = cave2_payload
    rtos_binary[tramp2_off : tramp2_off + len(tramp2_payload)] = tramp2_payload

    rtos_binary[desc_off : desc_off + len(desc_bytes)] = desc_bytes
    rtos_binary[mean_off : mean_off + len(mean_bytes)] = mean_bytes
    rtos_binary[std_off : std_off + len(std_bytes)] = std_bytes

    # Stock ROM Bug Sites in v1.20 DspService::LoadDsp (0x5398ee2c):
    id_bug_off  = LOAD_DSP_ID_BUG_VA  - GR4_BASE + rtos_payload_offset
    ip_bug1_off = LOAD_DSP_IP_BUG1_VA - GR4_BASE + rtos_payload_offset
    ip_bug2_off = LOAD_DSP_IP_BUG2_VA - GR4_BASE + rtos_payload_offset
    assert bytes(rtos_binary[id_bug_off : id_bug_off + 8]) == bytes.fromhex("ac1f0fe3331445e3"), "Unexpected bytes at LOAD_DSP_ID_BUG_VA"
    assert bytes(rtos_binary[ip_bug1_off : ip_bug1_off + 4]) == bytes.fromhex("9c1f0fe3"), "Unexpected bytes at LOAD_DSP_IP_BUG1_VA"
    assert bytes(rtos_binary[ip_bug2_off : ip_bug2_off + 4]) == bytes.fromhex("331445e3"), "Unexpected bytes at LOAD_DSP_IP_BUG2_VA"

    rtos_binary[id_bug_off : id_bug_off + 8] = struct.pack("<II", 0xe24b1040, 0xe1a00000) # sub r1, fp, #0x40; nop
    rtos_binary[ip_bug1_off : ip_bug1_off + 4] = struct.pack("<I", 0xe24b1040)            # sub r1, fp, #0x40
    rtos_binary[ip_bug2_off : ip_bug2_off + 4] = struct.pack("<I", 0xe1a00000)            # nop

    # Drive letter validation patch (cmp r3, #5 -> cmp r3, #15)
    drv_cmp_off = LOAD_DSP_DRV_CMP_VA - GR4_BASE + rtos_payload_offset
    assert bytes(rtos_binary[drv_cmp_off : drv_cmp_off + 4]) == struct.pack("<I", 0xe3530005)
    rtos_binary[drv_cmp_off : drv_cmp_off + 4] = struct.pack("<I", 0xe353000f)

    # ChunkLoader (0x5398e178) & LoadDsp (0x5398ee9c) base address checks (0x5120 -> 0x5210)
    chunk_off = CHUNK_LOADER_BASE_VA - GR4_BASE + rtos_payload_offset
    p3chk_off = LOAD_DSP_P3_BASE_VA  - GR4_BASE + rtos_payload_offset
    assert bytes(rtos_binary[chunk_off : chunk_off + 4]) == struct.pack("<I", 0xe3453120)
    assert bytes(rtos_binary[p3chk_off : p3chk_off + 4]) == struct.pack("<I", 0xe3453120)
    rtos_binary[chunk_off : chunk_off + 4] = struct.pack("<I", 0xe3453210)
    rtos_binary[p3chk_off : p3chk_off + 4] = struct.pack("<I", 0xe3453210)

    # DF_ID_S & DF_IP destination VAs in LoadDsp (0x515e -> 0x524e, 0x515f -> 0x524f)
    ids_dest_off = LOAD_DSP_IDS_DEST_VA - GR4_BASE + rtos_payload_offset
    ip_dest_off  = LOAD_DSP_IP_DEST_VA  - GR4_BASE + rtos_payload_offset
    assert bytes(rtos_binary[ids_dest_off : ids_dest_off + 4]) == struct.pack("<I", 0xe345015e)
    assert bytes(rtos_binary[ip_dest_off : ip_dest_off + 4])   == struct.pack("<I", 0xe345015f)
    rtos_binary[ids_dest_off : ids_dest_off + 4] = struct.pack("<I", 0xe345024e)
    rtos_binary[ip_dest_off : ip_dest_off + 4]   = struct.pack("<I", 0xe345024f)

    # Patch drive letter strings 'A' -> 'C'
    for str_va in (STR_DF_EX_VA, STR_DF_IP_VA, STR_DF_ID_S_VA):
        s_off = str_va - GR4_BASE + rtos_payload_offset
        assert rtos_binary[s_off] == 0x41, f"Expected 'A' at {hex(str_va)}"
        rtos_binary[s_off] = 0x43 # 'C' ("C:\DF_*.BIN")

    # Reconfigure ROM .data Memory Pool Table:
    # Pool[3] (DspFw) @ 0x543cde50: 0x51200000 -> 0x52100000 (4 MB)
    # Pool[8] (ImageAdjustWork) @ 0x543cdea0: 0x01210000, 0x51600000 -> 0x00f00000, 0x51200000 (15 MB)
    p3_off = POOL3_PAYLOAD_OFF
    p8_off = POOL8_PAYLOAD_OFF
    assert bytes(rtos_binary[p3_off + 8 : p3_off + 16]) == struct.pack("<II", 0x00400000, 0x51200000)
    assert bytes(rtos_binary[p8_off + 8 : p8_off + 16]) == struct.pack("<II", 0x01210000, 0x51600000)

    rtos_binary[p3_off + 8 : p3_off + 16] = struct.pack("<II", 0x00400000, 0x52100000)
    rtos_binary[p8_off + 8 : p8_off + 16] = struct.pack("<II", 0x00f00000, 0x51200000)

    manifest = {
        "stage": "Stage 3 Deep Learning Inference Pipeline & Live View Tracking Bridge (v1.2.0)",
        "baseline": "Official Ricoh GR IV Firmware v1.20",
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
        "code_cave": {
            "vaddr": hex(CAVE2_VA),
            "offset": hex(cave2_off),
            "length": len(cave2_payload),
            "hex": cave2_payload.hex(),
        },
        "trampoline": {
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
            "tensor_input_rgb56x56_stride64": hex(TENSOR_IN_VA),
            "tensor_message_struct": hex(TENSOR_MSG_VA),
            "heatmap_output_3ch_56x64_s16": hex(TENSOR_OUT_VA),
            "k3_mean_and_backbone_weights": hex(MEAN_WEIGHTS_VA),
            "k3_head_weights": hex(HEAD_WEIGHTS_VA),
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
    print("[*] Generating Stage 3 Dual-Hook AI Tracking Bridge payloads (v1.2.0)...")
    c1 = build_cave1_payload()
    t1 = build_tramp1_payload()
    c2 = build_cave2_payload()
    t2 = build_tramp2_payload()

    print(f"  [+] Cave 1 (Boot Wakeup): {len(c1)} bytes @ {hex(CAVE1_VA)} .. {hex(CAVE1_VA + len(c1))}")
    print(f"  [+] Tramp 1:             {len(t1)} bytes @ {hex(HOOK1_SITE_VA)}")
    print(f"  [+] Cave 2 (LiveView AI): {len(c2)} bytes @ {hex(CAVE2_VA)} .. {hex(CAVE2_VA + len(c2))}")
    print(f"  [+] Tramp 2:             {len(t2)} bytes @ {hex(HOOK2_SITE_VA)}")

    md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    print("\n--- Cave 1 Disassembly ---")
    for ins in md.disasm(c1, CAVE1_VA):
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")

    print("\n--- Cave 2 Disassembly ---")
    for ins in md.disasm(c2, CAVE2_VA):
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str}")

    print("\n[+] Stage 3 Dual-Hook v1.2.0 payload assembled and verified successfully.")
