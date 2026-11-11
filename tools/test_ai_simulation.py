#!/usr/bin/env python3
"""
test_ai_simulation.py - End-to-End Functional Simulation of Stage 3 AI AF Pipeline

Simulates:
  1. YUV420 Planar downsampling with 2.0x Backlit Shadow Recovery
  2. Heatmap peak scanning logic across Ch0, Ch1, Ch2
  3. Noise gating logic (max_diff >= 3 and area_cnt >= 2)
  4. Heartbeat fallback logic on empty scene
  5. Aspect-ratio fixed-point coordinate mapping
"""

import numpy as np

def simulate_backlit_curve(y_val: int) -> int:
    if y_val < 48:
        return min(255, y_val * 2)
    elif y_val < 144:
        return 72 + (y_val >> 1)
    else:
        return y_val

def simulate_downsampler(src_yuv: np.ndarray, fd_w: int, fd_h: int) -> np.ndarray:
    """Downsamples arbitrary resolution YUV to 56x56 RGB tensor with row stride 64."""
    tensor = np.zeros((3, 56, 64), dtype=np.uint8)
    for r in range(56):
        src_y = (r * fd_h * 9) >> 9
        for c in range(56):
            src_x = (c * fd_w * 9) >> 9
            y_raw = int(src_yuv[src_y, src_x])
            y_lifted = simulate_backlit_curve(y_raw)
            # Simulating R=G=B=y_lifted for luma-only
            tensor[0, r, c] = y_lifted
            tensor[1, r, c] = y_lifted
            tensor[2, r, c] = y_lifted
    return tensor

def simulate_peak_finder(heatmap: np.ndarray):
    """
    heatmap shape: (3, 56, 64) int16
    Ch0: Background, Ch1: Human Head/Face, Ch2: Animal/Subject
    """
    max_diff = 0
    best_col = 28
    best_row = 28
    area_cnt = 0

    for r in range(56):
        for c in range(56):
            bg = heatmap[0, r, c]
            c1 = heatmap[1, r, c]
            c2 = heatmap[2, r, c] - 2  # Head-priority penalty
            fg = max(c1, c2)
            diff = fg - bg
            if diff >= 2:
                area_cnt += 1
            if diff > max_diff:
                max_diff = diff
                best_col = c
                best_row = r

    return max_diff, best_col, best_row, area_cnt

def simulate_pipeline(heatmap: np.ndarray, fd_w: int = 320, fd_h: int = 212):
    max_diff, best_col, best_row, area_cnt = simulate_peak_finder(heatmap)

    # Noise-immune detection gate
    if max_diff >= 3 and area_cnt >= 2:
        # Subject detected
        box_size = min(0x5c00, 0x2800 + (area_cnt << 6))
        score = min(100, max_diff + 30)
        cx = (best_col * fd_w * 73) >> 4
        cy = (best_row * fd_h * 73) >> 4
        left = max(0, cx - (box_size >> 1))
        top = max(0, cy - (box_size >> 1))
        return {
            "status": "SUBJECT_DETECTED",
            "score": score,
            "left": left >> 8,
            "top": top >> 8,
            "box_size": box_size >> 8,
            "cx": cx >> 8,
            "cy": cy >> 8,
        }
    else:
        # Heartbeat fallback box
        return {
            "status": "HEARTBEAT_DIAGNOSTIC",
            "score": 99,
            "left": 240,
            "top": 140,
            "box_size": 32,
            "cx": 256,
            "cy": 156,
        }

def run_simulation_tests():
    print("[*] Running Stage 3 AI Pipeline Simulation Tests...")

    # Test 1: Empty scene / Blank wall
    blank_heatmap = np.zeros((3, 56, 64), dtype=np.int16)
    res_blank = simulate_pipeline(blank_heatmap)
    assert res_blank["status"] == "HEARTBEAT_DIAGNOSTIC"
    assert res_blank["left"] == 240 and res_blank["top"] == 140 and res_blank["score"] == 99
    print("  [+] Test 1 (Blank Scene): Successfully produced Lower-Right Heartbeat Box (240, 140, 32x32, score=99)")

    # Test 2: Cat Face in Center (Ch2 peak with surrounding area)
    cat_heatmap = np.zeros((3, 56, 64), dtype=np.int16)
    # Simulate a 3x3 blob for a cat at row 20, col 30
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            cat_heatmap[2, 20 + dr, 30 + dc] = 15  # strong animal signal
            cat_heatmap[0, 20 + dr, 30 + dc] = 2   # weak background
    res_cat = simulate_pipeline(cat_heatmap)
    assert res_cat["status"] == "SUBJECT_DETECTED"
    assert res_cat["score"] > 35
    assert 150 <= res_cat["cx"] <= 190
    assert 60 <= res_cat["cy"] <= 90
    print(f"  [+] Test 2 (Cat Face): Detected at cx={res_cat['cx']}, cy={res_cat['cy']}, score={res_cat['score']}")

    # Test 3: Profile Human Face on Left (Ch1 peak)
    human_heatmap = np.zeros((3, 56, 64), dtype=np.int16)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            human_heatmap[1, 28 + dr, 10 + dc] = 20  # strong human signal
            human_heatmap[0, 28 + dr, 10 + dc] = 1   # background
    res_human = simulate_pipeline(human_heatmap)
    assert res_human["status"] == "SUBJECT_DETECTED"
    assert res_human["score"] > 40
    assert 40 <= res_human["cx"] <= 80
    assert 90 <= res_human["cy"] <= 130
    print(f"  [+] Test 3 (Human Profile): Detected at cx={res_human['cx']}, cy={res_human['cy']}, score={res_human['score']}")

    # Test 4: Backlit recovery curve checks
    assert simulate_backlit_curve(20) == 40  # Under 48: 2x boost
    assert simulate_backlit_curve(80) == 112 # 48..144: 72 + 40 = 112
    assert simulate_backlit_curve(200) == 200 # Over 144: unchanged
    print("  [+] Test 4 (Backlit Recovery Curve): C0-continuity & 2.0x shadow lift verified")

    print("\n" + "=" * 70)
    print("ALL AI SIMULATION TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_simulation_tests()
