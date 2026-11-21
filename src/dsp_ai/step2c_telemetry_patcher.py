"""Step 2C display-only instrumentation for the frozen v1.2.11 RTOS cave."""
import hashlib
import struct

import stage3_ai_patcher as s3
import step2a_dsp_probe_patcher as probe

BASE_LEN = 1288
BASE_CAVE_SHA = '622720ee04e599c2fbc374d5e75c1dc4f5c15d5e63c6cd14fe88d90e7d876b29'
RENDER_SITE = s3.CAVE2_VA + 0x7c  # original mov r0, #0 before result zeroing
HELPER_VA = s3.CAVE2_VA + BASE_LEN
EXPECTED_RENDER = bytes.fromhex('0000a0e3')
RENDER_BRANCH = struct.pack('<I', s3.encode_arm_branch(RENDER_SITE, HELPER_VA, is_bl=True))


def build_position_helper() -> bytes:
    """Select position from a prior frame's latched state; no writes or calls.

    The caller has formed a valid centered rectangle and holds the existing FD
    mutex. r7/r8 are the only nonvolatile registers intentionally changed.
    """
    ops, labels, branches = [], {}, []

    def mark(name):
        labels[name] = len(ops)

    def b(name, cond=0xe):
        branches.append((len(ops), name, cond))
        ops.append(0)

    s3.emit_ldr_imm32(ops, 3, s3.STATUS_VA)
    s3.emit_ldr_imm32(ops, 1, probe.CONTINUITY_COOKIE)
    ops.extend((0xe5932018, 0xe1520001))  # cookie == expected
    b('finish', 0x1)
    ops.append(0xe5930000)  # state
    for state, label in ((1, 'load_error'), (4, 'done'), (5, 'start_error'),
                         (6, 'send_error'), (7, 'no_ddr'), (8, 'uncertain')):
        ops.append(0xe3500000 | state)
        b(label, 0x0)
    b('finish')

    # Shift by half the available center margin, not by fixed pixels.
    # Effect stays within frame even at 48 pixels or with a changed aspect ratio.
    left, right = 0xe04770a7, 0xe08770a7  # sub/add r7, r7, r7, lsr #1
    up, down = 0xe04880a8, 0xe08880a8    # sub/add r8, r8, r8, lsr #1
    for label, x, y in (('load_error', left, up), ('start_error', left, None),
                        ('send_error', left, down), ('no_ddr', right, down),
                        ('uncertain', right, up)):
        mark(label)
        ops.append(x)
        if y is not None:
            ops.append(y)
        b('finish')

    mark('done')
    ops.extend((0xe593201c, 0xe3520000 | probe.CONTINUITY_BUDGET))
    b('finish', 0x1)  # do not claim success from terminal state alone
    ops.append(right)
    mark('finish')
    ops.extend((0xe3a00000, 0xe12fff1e))  # replay displaced mov r0,#0 then return
    for i, label, cond in branches:
        ops[i] = s3.encode_arm_branch(HELPER_VA + i*4, HELPER_VA + labels[label]*4, cond=cond)
    return struct.pack(f'<{len(ops)}I', *ops)


def patch_cave(base_cave: bytes, zero_tail: bytes) -> tuple[bytes, bytes]:
    helper = build_position_helper()
    off = RENDER_SITE - s3.CAVE2_VA
    if len(base_cave) != BASE_LEN or hashlib.sha256(base_cave).hexdigest() != BASE_CAVE_SHA:
        raise ValueError('Not the frozen v1.2.11 continuity cave')
    if base_cave[off:off+4] != EXPECTED_RENDER:
        raise ValueError('Wrong original render instruction')
    if HELPER_VA + len(helper) > s3.DESC_TABLE_VA or zero_tail != bytes(len(helper)):
        raise ValueError('Telemetry helper would leave or overwrite the zero space')
    return base_cave[:off] + RENDER_BRANCH + base_cave[off+4:], helper
