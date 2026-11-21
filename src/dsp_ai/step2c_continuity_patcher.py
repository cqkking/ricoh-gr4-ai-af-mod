"""Step 2C continuity experiment: OSD commit first; a bounded DSP probe second.

Development source only. This module has no package or SD writer.
"""

import struct

import stage3_ai_patcher as s3
import step2a_dsp_probe_patcher as a
import step2c_osd_recovery as osd

HOOK_UPDATE_VA = 0x538963B8
EXPECTED_UPDATE = bytes.fromhex('8dfdffeb')  # verified against the v1.20 image


def build_continuity_cave1() -> bytes:
    return s3.build_cave1_payload(status_cookie=a.CONTINUITY_COOKIE)


def update_shim_address() -> int:
    return s3.CAVE2_VA + len(osd.build_osd_recovery_cave())


def build_update_hook() -> bytes:
    return struct.pack('<I', s3.encode_arm_branch(HOOK_UPDATE_VA, update_shim_address(), is_bl=True))


def build_continuity_cave2() -> bytes:
    shim = update_shim_address()
    # The stock continuation has already unlocked FD. Preserve the original
    # BL return and native return registers, then call the separate DSP core.
    wrapper = (
        0xe92d4010,  # push {r4, lr}; maintain 8-byte SP alignment
        s3.encode_arm_branch(shim + 4, s3.UPDATE_FACE_RES_VA, is_bl=True),
        0xe92d500f,  # push {r0-r3, ip, lr}
        s3.encode_arm_branch(shim + 12, shim + 24, is_bl=True),
        0xe8bd500f,  # pop {r0-r3, ip, lr}
        0xe8bd8010,  # pop {r4, pc}
    )
    core = a.build_step2a_cave2_payload(shim + 24, post_update=True)
    bypass = shim + 24 + len(core)
    # If the Hook 2 guards reject this frame, follow the same native
    # unlock/commit sequence without entering the post-update DSP hook.
    invalid_path = (
        0xe1a00006,  # mov r0, r6
        s3.encode_arm_branch(bypass + 4, s3.UNLOCK_MUTEX_VA, is_bl=True),
        0xe1a00004,  # mov r0, r4
        s3.encode_arm_branch(bypass + 12, s3.UPDATE_FACE_RES_VA, is_bl=True),
        s3.encode_arm_branch(bypass + 16, HOOK_UPDATE_VA + 4),
    )
    render = osd.build_osd_recovery_cave(invalid_resume_va=bypass)
    cave = render + struct.pack('<6I', *wrapper) + core + struct.pack('<5I', *invalid_path)
    if s3.CAVE2_VA + len(cave) > s3.DESC_TABLE_VA:
        raise ValueError('Step 2C continuity cave exceeds the validated zero space')
    return cave


def apply_continuity_experiment(payload: bytearray, rtos_start: int, rtos_end: int) -> dict:
    """Apply to a copy first so both old and new original-byte checks are non-destructive."""
    if not __debug__:
        raise RuntimeError('Stage 3 preflight uses assertions; do not run with python -O')
    off = rtos_start + HOOK_UPDATE_VA - s3.GR4_BASE
    if (rtos_start != s3.RTOS_PAYLOAD_START or rtos_end != 0x1423328
            or not rtos_start <= off < off + 4 <= rtos_end <= len(payload)):
        raise ValueError('Invalid v1.20 RTOS bounds')
    if payload[off:off+4] != EXPECTED_UPDATE:
        raise ValueError('Unexpected stock UpdateFaceDetectionResult call')
    check = bytearray(payload)
    manifest = s3.apply_stage3_patch(check, rtos_payload_offset=rtos_start,
                                    cave1_builder=build_continuity_cave1,
                                    cave2_builder=build_continuity_cave2)
    check[off:off+4] = build_update_hook()
    if payload[:rtos_start] != check[:rtos_start] or payload[rtos_end:] != check[rtos_end:]:
        raise ValueError('Continuity experiment writes outside RTOS')
    payload[:] = check
    manifest.update({
        'stage': 'Step 2C continuity experiment: committed OSD + bounded synthetic DSP',
        'status': 'development_only',
        'post_update_hook': {'va': hex(HOOK_UPDATE_VA), 'offset': hex(off), 'original': EXPECTED_UPDATE.hex(),
                             'replacement': build_update_hook().hex()},
        'dsp_input': 'Step 2A synthetic only (no AHIP)',
        'request_budget': a.CONTINUITY_BUDGET,
        'wait_after_fd_frames': a.CONTINUITY_WAIT_FRAMES,
    })
    return manifest
