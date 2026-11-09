#!/usr/bin/env python3
"""
build_firmware.py - Master build entry point for Ricoh GR IV AI-AF Firmware Mod

Unites:
  - DSP payload splitting & verification (dsp_splitter.py)
  - Hybrid AF Dual-Hook code cave injection (stage3_ai_patcher.py)
  - Zero-collision memory allocation & word-sum checksum packaging (build_stage3_firmware.py)
"""

import sys
from pathlib import Path

# Delegate directly to build_stage3_release
from build_stage3_firmware import build_stage3_release

if __name__ == "__main__":
    build_stage3_release()
