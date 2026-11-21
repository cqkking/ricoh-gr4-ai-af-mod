"""Step 2C OSD recovery experiment. Applies only to decoded stock v1.20 in memory.

Deliberately excludes the DSP, shared DDR, and AHIP paths. Not a replacement
for continuous AI AF and not a flashable package builder.
"""

import struct
from dataclasses import dataclass

import step1_osd_mock_patcher as osd

OFFICIAL_SIZE = 38_919_904
OFFICIAL_SHA256 = "7563d7e462f0c53b6ab2fbab7cdfbbb57151a41192ee82f4f3a2108f9d77650d"
EXPECTED_HOOK = bytes.fromhex("380094e57b40ffeb")
EXPECTED_STOCK_CONTINUATION = bytes.fromhex("0600a0e1")  # mov r0, r6


@dataclass(frozen=True)
class Replacement:
    va: int
    offset: int
    original: bytes
    replacement: bytes


def build_osd_recovery_cave(*, invalid_resume_va: int = osd.HOOK2_AI_RESUME_VA) -> bytes:
    """Validate render prerequisites while the native FD mutex is still held."""
    # r0 is caller-saved and reloaded at the stock continuation. Neither the
    # guards nor the Step 1 body acquire or release the native FD mutex.
    guards = (
        (0xe5940078, 0xe3500000, 0x0),  # result array != NULL (BEQ skips)
        (0xe5940038, 0xe3500000, 0x0),  # FdLibraryService != NULL
        (0xe1d40fbc, 0xe3500030, 0x3),  # width >= 48 (BLO skips)
        (0xe1d40fbe, 0xe3500030, 0x3),  # height >= 48
    )
    words = []
    for load, compare, cond in guards:
        words.extend((load, compare))
        words.append(osd.encode_arm_branch(
            osd.CAVE_VA + len(words) * 4, invalid_resume_va, cond=cond
        ))
    prefix = struct.pack(f"<{len(words)}I", *words)
    return prefix + osd.build_step1_cave(osd.CAVE_VA + len(prefix))


def apply_to_decoded_payload(payload: bytearray, rtos_start: int, rtos_end: int) -> tuple[Replacement, ...]:
    """Validate both original ranges before mutating either; no container or I/O."""
    if not 0 <= rtos_start < rtos_end <= len(payload) or rtos_start != osd.RTOS_PAYLOAD_START:
        raise ValueError("Unexpected v1.20 RTOS bounds")

    def offset(va: int) -> int:
        return rtos_start + va - osd.GR4_BASE

    cave = build_osd_recovery_cave()
    if osd.CAVE_VA + len(cave) > 0x53001600:
        raise ValueError("OSD recovery cave exceeds its allowed range")
    changes = (
        Replacement(osd.CAVE_VA, offset(osd.CAVE_VA), bytes(len(cave)), cave),
        Replacement(osd.HOOK2_SITE_VA, offset(osd.HOOK2_SITE_VA), EXPECTED_HOOK, osd.build_tramp2_payload()),
    )
    previous_end = rtos_start
    for change in changes:
        end = change.offset + len(change.original)
        if not previous_end <= change.offset < end <= rtos_end or len(change.replacement) != len(change.original):
            raise ValueError("Overlapping, out-of-range or growing replacement")
        if payload[change.offset:end] != change.original:
            raise ValueError(f"Unexpected original bytes at {change.va:#010x}")
        previous_end = end
    resume = offset(osd.HOOK2_AI_RESUME_VA)
    if resume + 4 > rtos_end or payload[resume:resume + 4] != EXPECTED_STOCK_CONTINUATION:
        raise ValueError("Missing native unlock/update continuation")
    for change in changes:
        payload[change.offset:change.offset + len(change.original)] = change.replacement
    return changes
