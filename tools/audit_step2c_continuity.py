#!/usr/bin/env python3
"""Audit the physical Step 2C candidate and its three DSP files without importing the patcher."""

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

from audit_step2c_osd_recovery import STOCK_BYTES, STOCK_SHA, RTOS_START, RTOS_END, inspect, require

STEP2A_SHA = '8a258f0207ea9973c68161bc55613260b167525efb80c76dc493efa21eaa28cf'
STEP2A_BYTES = 39109096
DSP_FILES = {
    'DF_EX.BIN': (5251648, 'b16788a16a11ecc2c007d26888a56195b2b4e1bec8aa05b74ea6789803ae8d44'),
    'DF_ID_S.BIN': (65536, '3d3ebf29622323bb4a203347feaf676aeac110a7b1181604270a1ae0c22b0a2a'),
    'DF_IP.BIN': (16384, 'cf409a84f3f2d8fac1491634ecda882c9d76b3439c2423da1b31950463a52067'),
}
# Frozen output code digests, checked independently of the builder and disassembler.
NEW = (
    ('cave1', 0x18A88, 100, '431b2d07c60117337fdc56ee0a506bd696b9e98a33f5e1c206ae5c236c3156f9'),
    ('cave2', 0x18B28, 1288, '622720ee04e599c2fbc374d5e75c1dc4f5c15d5e63c6cd14fe88d90e7d876b29'),
    ('after_update', 0x8AE0E0, 4, '834c4da0bbc81bfa6e97f1aba4a51f8a1c449ef8a050f3c0489c2c630a505490'),
)
COMMON = (
    ('descriptor', 0x19328, 728), ('hook1', 0x68E7E0, 8), ('hook2', 0x8AE0C4, 8),
    ('chunk_base', 0x9A5EA0, 4), ('load_base', 0x9A6BC4, 4), ('drive_enum', 0x9A6C04, 4),
    ('id_path', 0x9A6C90, 8), ('ids_dest', 0x9A6CA0, 4), ('ip_path_1', 0x9A6CD4, 4),
    ('ip_dest', 0x9A6CDC, 4), ('ip_path_2', 0x9A6CE0, 4),
    ('mean', 0x122A5B8, 9408), ('std', 0x12666F8, 9408),
    ('ex_drive', 0x1357CB4, 1), ('ip_drive', 0x1357CC4, 1), ('id_drive', 0x1357CD4, 1),
    ('dsp_pool', 0x13E5B80, 8), ('image_adjust_pool', 0x13E5BD0, 8),
)


def pin(raw, metrics, label):
    size, digest = metrics
    require(len(raw) == size and sha256(raw).hexdigest() == digest, f'{label}: wrong size or SHA-256')


def pin_dsp_files(directory):
    meta = {}
    for name, metrics in DSP_FILES.items():
        raw = (directory / name).read_bytes()
        pin(raw, metrics, name)
        meta[name] = {'size': metrics[0], 'sha256': metrics[1]}
    return meta


def check_differences(before, after, ranges, label):
    require(len(before) == len(after), f'{label}: decoded payload grew or shrank')
    cursor = 0
    changes = []
    for name, off, size in sorted(ranges, key=lambda item: item[1]):
        end = off + size
        require(cursor <= off and RTOS_START <= off < end <= RTOS_END, f'{label}: invalid range {name}')
        require(before[cursor:off] == after[cursor:off], f'{label}: unexpected differences before {name}')
        original, patched = before[off:end], after[off:end]
        changes.append({'name': name, 'offset': hex(off), 'length': size,
                        'before_sha256': sha256(original).hexdigest(), 'after_sha256': sha256(patched).hexdigest(),
                        'changed_bytes': sum(l != r for l, r in zip(original, patched))})
        cursor = end
    require(before[cursor:-4] == after[cursor:-4], f'{label}: unexpected differences after RTOS allowlist')
    changes.append({'name': 'derived_payload_checksum', 'offset': hex(len(before)-4), 'length': 4,
                    'before': before[-4:].hex(), 'after': after[-4:].hex(),
                    'changed_bytes': sum(l != r for l, r in zip(before[-4:], after[-4:]))})
    return changes


def check_frames(stock, candidate, orig_frames, new_frames, scope, payload_end):
    require(len(orig_frames) == len(new_frames), 'Changed frame count')
    reemits = []
    for i, (a, b) in enumerate(zip(orig_frames, new_frames)):
        start, end = a['output_start'], a['output_end']
        require((start, end) == (b['output_start'], b['output_end']), f'Changed decoded frame bounds: {i}')
        if stock[a['file_start']:a['file_end']] == candidate[b['file_start']:b['file_end']]:
            continue
        permit = any(off < end and off + length > start-0x2000 for _, off, length in scope)
        require((permit or end == payload_end) and b['type'] == 'literal' and 0 < end-start <= 0x6000,
                f'Unexpected re-emission: {i}')
        reemits.append(i)
    return reemits


def audit(stock, old, candidate, decoder, dsp_dir):
    pin(stock, (STOCK_BYTES, STOCK_SHA), 'official sample')
    pin(old, (STEP2A_BYTES, STEP2A_SHA), 'reported Step 2A hardware baseline')
    dsp = pin_dsp_files(dsp_dir)
    before, stock_frames, stock_tail = inspect(stock, decoder)
    old_dec, _, old_tail = inspect(old, decoder)
    new_dec, new_frames, new_tail = inspect(candidate, decoder)
    require(stock[:0x80] == old[:0x80] == candidate[:0x80], 'Product/version header changed')
    # The older Step 2A re-emitter can leave an extra leading zero pad word.
    require(stock_tail[:-12] == old_tail[:-12] == new_tail[:-12], 'Non-derived trailer fields changed')
    require(before[0x8AE0E0:0x8AE0E4] == bytes.fromhex('8dfdffeb'), 'Wrong native update call')
    require(before[0x18A88:0x18A88+100] == bytes(100), 'Cave 1 preimage is not zero')
    require(before[0x18B28:0x18B28+1288] == bytes(1288), 'Cave 2 preimage is not zero')
    scope = tuple((name, off, size) for name, off, size, _ in NEW)
    for name, off, size, digest in NEW:
        require(sha256(new_dec[off:off+size]).hexdigest() == digest, f'Unexpected frozen candidate code: {name}')
    inherited = (('cave1', 0x18A88, 84), ('cave2', 0x18B28, 1104)) + COMMON
    check_differences(before, old_dec, inherited, 'Step 2A vs official')
    stock_deltas = check_differences(before, new_dec, scope + COMMON, 'candidate vs official')
    old_deltas = check_differences(old_dec, new_dec, scope, 'candidate vs Step 2A')
    reemits = check_frames(stock, candidate, stock_frames, new_frames, scope+COMMON, len(before))
    return {
        'version': 'v1.2.11-step2c-continuity-test', 'status': 'package_checks_pass', 'gate_2': 'not_authorized',
        'scope': 'OSD submit first, bounded Step 2A synthetic DSP; no AHIP, recognition, or real-camera verification',
        'model': 'GR IV', 'internal_version': '1.20.10.11', 'stock_sha256': STOCK_SHA, 'step2a_sha256': STEP2A_SHA,
        'candidate_size': len(candidate), 'candidate_sha256': sha256(candidate).hexdigest(),
        'decoded_size': len(new_dec), 'frames': len(new_frames), 'literal_reemits_zero_based': reemits,
        'header_and_footer_identity': 'unchanged; encoded size and outer checksum derived',
        'boot_and_non_target_payload': 'identical except derived global payload checksum',
        'delta_from_stock': stock_deltas, 'delta_from_step2a': old_deltas, 'dsp_assets': dsp,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('stock','step2a','candidate','decoder','dsp-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expect-sha256')
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('independent_gr4_decoder', args.decoder)
    require(spec is not None and spec.loader is not None, 'Cannot load the independent decoder')
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    result = audit(args.stock.read_bytes(), args.step2a.read_bytes(), args.candidate.read_bytes(), decoder, args.dsp_dir)
    require(not args.expect_sha256 or result['candidate_sha256'] == args.expect_sha256.lower(), 'Wrong candidate SHA-256')
    result['independent_decoder_sha256'] = sha256(args.decoder.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
