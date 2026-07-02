#!/usr/bin/env python3
"""
Diagnostic: compare the Block 1 -> Block 2 wire tap against the golden
Block 1 output. Pinpoints the first divergence in (channel, row, col).

Usage:
    python diagnose_block1_tap.py
"""

import numpy as np
import sys
import os

INTERCEPT_FILE = "sim_block1_intercept.hex"
GOLDEN_NPZ     = "block1_targets_fixed.npz"

OUT_CH, OUT_H, OUT_W = 16, 4, 4
TOLERANCE = 4  # match the testbench's +/- LSB tolerance


def parse_signed_hex_16(path):
    """Read a hex-per-line file, interpret each as signed 16-bit (Q8.8 word)."""
    vals = []
    with open(path, "r") as f:
        for lineno, raw in enumerate(f, 1):
            tok = raw.strip()
            if not tok:
                continue
            try:
                u = int(tok, 16) & 0xFFFF
            except ValueError:
                print(f"  ! line {lineno}: cannot parse '{tok}' as hex")
                continue
            # two's complement sign extension
            vals.append(u - 0x10000 if u & 0x8000 else u)
    return np.array(vals, dtype=np.int32)


def main():
    for p in (INTERCEPT_FILE, GOLDEN_NPZ):
        if not os.path.exists(p):
            print(f"ERROR: missing '{p}' (run the sim / goldenReference.py first)")
            sys.exit(1)

    # --- Golden: stored as [channel, row, col]; reorder to channel-fastest ---
    golden_chw = np.load(GOLDEN_NPZ)["final_output"][0]  # shape (16, 4, 4)
    # transpose to (row, col, channel) then flatten -> matches HW stream order
    golden_stream = golden_chw.transpose(1, 2, 0).reshape(-1).astype(np.int32)

    actual = parse_signed_hex_16(INTERCEPT_FILE)

    print(f"Golden pixels   : {golden_stream.size}")
    print(f"Captured pixels : {actual.size}")

    if actual.size != golden_stream.size:
        print(
            f"\n*** COUNT MISMATCH: expected {golden_stream.size}, "
            f"captured {actual.size}. Block 1 is not emitting a clean "
            f"{OUT_CH*OUT_H*OUT_W}-pixel frame — investigate Block 1 "
            f"valid/FSM before trusting per-pixel diffs.\n"
        )

    n = min(actual.size, golden_stream.size)

    def decode(stream_idx):
        # stream order is (row, col, channel) with channel fastest
        ch = stream_idx % OUT_CH
        rc = stream_idx // OUT_CH
        col = rc % OUT_W
        row = rc // OUT_W
        return row, col, ch

    first_bad = None
    n_exceed = 0
    for i in range(n):
        diff = int(actual[i]) - int(golden_stream[i])
        if abs(diff) > TOLERANCE:
            n_exceed += 1
            if first_bad is None:
                first_bad = i

    if first_bad is None:
        print(f"\n=== Block 1 output CLEAN: all {n} pixels within +/-{TOLERANCE} LSB ===")
        print("    -> The fault is in Block 2's load/compute path, not Block 1.")
        return

    r, c, ch = decode(first_bad)
    print(
        f"\n*** FIRST DIVERGENCE at stream index {first_bad} "
        f"-> (channel={ch}, row={r}, col={c})"
    )
    print(f"    golden={golden_stream[first_bad]}  actual={actual[first_bad]}  "
          f"diff={int(actual[first_bad]) - int(golden_stream[first_bad])}")
    print(f"    total pixels exceeding tolerance: {n_exceed}/{n}\n")

    # Context window around the first failure.
    lo = max(0, first_bad - 2)
    hi = min(n, first_bad + 8)
    print("  idx  (ch,row,col)   golden   actual   diff")
    print("  ---  -----------    ------   ------   ----")
    for i in range(lo, hi):
        rr, cc, cch = decode(i)
        d = int(actual[i]) - int(golden_stream[i])
        flag = "  <-- first bad" if i == first_bad else ""
        print(f"  {i:3d}  ({cch:2d},{rr},{cc})    "
              f"{golden_stream[i]:6d}   {actual[i]:6d}   {d:4d}{flag}")


if __name__ == "__main__":
    main()
