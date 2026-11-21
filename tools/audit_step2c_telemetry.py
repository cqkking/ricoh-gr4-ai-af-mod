"""Independently audit the v1.2.12 disk image against stock and v1.2.11."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

from audit_step2c_continuity import (
    COMMON, NEW, check_differences, check_frames, pin, pin_dsp_files,
)
from audit_step2c_osd_recovery import STOCK_BYTES, STOCK_SHA, inspect, require

BASE_METRICS = (39109096, 'd385fffe1dbf8ebdd6a0aacb62ffbf1c9bf6e466948b43bdd02643a3f3df9284')
DELTAS = (('render_branch', 0x18ba4, 4), ('read_only_position_helper', 0x19030, 164))
HELPER_SHA = 'a0653bd69aebfbc51e77655a4d10c6712ea4db169a6a714f46221d5bb39c9233'


def audit(stock, base, candidate, decoder, dsp_dir):
    pin(stock, (STOCK_BYTES, STOCK_SHA), 'official')
    pin(base, BASE_METRICS, 'v1.2.11')
    files = pin_dsp_files(dsp_dir)
    official, _, t0 = inspect(stock, decoder)
    old, old_frames, t1 = inspect(base, decoder)
    new, new_frames, t2 = inspect(candidate, decoder)
    require(stock[:128] == base[:128] == candidate[:128], 'Model/version header changed')
    require(t0[:-12] == t1[:-12] == t2[:-12], 'Non-derived trailer changed')
    require(old[0x18ba4:0x18ba8] == bytes.fromhex('0000a0e3') and
            new[0x18ba4:0x18ba8] == bytes.fromhex('210100eb'), 'Wrong original or final render instruction')
    require(old[0x19030:0x190d4] == bytes(164) and
            sha256(new[0x19030:0x190d4]).hexdigest() == HELPER_SHA, 'Wrong helper or nonzero preimage')
    baseline_scope = tuple((label, off, length) for label, off, length, _ in NEW) + COMMON
    check_differences(official, old, baseline_scope, 'v1.2.11 vs official')
    delta = check_differences(old, new, DELTAS, 'v1.2.12 vs v1.2.11')
    reemits = check_frames(base, candidate, old_frames, new_frames, DELTAS, len(new))
    return {
        'status': 'package_checks_pass', 'version': 'v1.2.12-step2c-dsp-telemetry', 'gate_2': 'not_authorized',
        'stock_sha256': STOCK_SHA, 'base_sha256': BASE_METRICS[1], 'model': 'GR IV',
        'internal_version': '1.20.10.11', 'candidate_sha256': sha256(candidate).hexdigest(),
        'candidate_size': len(candidate), 'decoded_size': len(new), 'frames': len(new_frames),
        'literal_reemits_vs_base': reemits, 'delta_from_base': delta,
        'probe_core_and_native_dsp_hooks_unchanged': True, 'boot_and_non_target_regions_unchanged': True,
        'dsp_assets': files, 'camera_verification': 'not_run', 'AHIP': 'not_enabled',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('stock', 'base', 'candidate', 'decoder', 'dsp-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expect-sha256')
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('independent_gr4_decoder', args.decoder)
    require(spec is not None and spec.loader is not None, 'Cannot load independent decoder')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = audit(args.stock.read_bytes(), args.base.read_bytes(), args.candidate.read_bytes(), module, args.dsp_dir)
    require(not args.expect_sha256 or result['candidate_sha256'] == args.expect_sha256.lower(), 'Wrong candidate SHA-256')
    result['independent_decoder_sha256'] = sha256(args.decoder.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
