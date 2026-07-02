#!/usr/bin/env python3
"""
gen_hex_files.py
================
Generates all .hex files needed by the Verilog testbench from your
Python Golden Reference Script.

Run this AFTER you have:
  - Trained / loaded the LDC model (or your test weight tensors)
  - Applied the BatchNorm folding to get folded_w and folded_b

Usage:
    python gen_hex_files.py

Outputs (all in the current directory):
    conv1_weights.hex   — 432 entries, 16-bit signed Q8.8
    conv1_biases.hex    —  16 entries, 32-bit signed Q16.16
    conv2_weights.hex   — 2304 entries, 16-bit signed Q8.8
    conv2_biases.hex    —  16 entries, 32-bit signed Q16.16
    input_image.hex     —  192 entries, 16-bit signed Q8.8
    expected_output.hex —  256 entries, 16-bit signed Q8.8

Q8.8 encoding  : float → round(f * 256) → int16 → 4 hex chars
Q16.16 encoding: float → round(f * 65536) → int32 → 8 hex chars
"""

import numpy as np

# =============================================================================
# QUANTISATION HELPERS
# =============================================================================

Q8_8_SCALE   = 256          # 2^8
Q16_16_SCALE = 65536        # 2^16
INT16_MIN    = -32768
INT16_MAX    =  32767
INT32_MIN    = -(1 << 31)
INT32_MAX    =  (1 << 31) - 1


def to_q8_8(arr: np.ndarray) -> np.ndarray:
    """Convert float array to Q8.8 int16, with saturation."""
    scaled = np.round(arr * Q8_8_SCALE).astype(np.int64)
    clipped = np.clip(scaled, INT16_MIN, INT16_MAX)
    return clipped.astype(np.int16)


def to_q16_16(arr: np.ndarray) -> np.ndarray:
    """Convert float array to Q16.16 int32, with saturation."""
    scaled = np.round(arr * Q16_16_SCALE).astype(np.int64)
    clipped = np.clip(scaled, INT32_MIN, INT32_MAX)
    return clipped.astype(np.int32)


def write_hex_16(filename: str, data: np.ndarray) -> None:
    """Write 16-bit signed int array as hex (4 chars per line, 2's complement)."""
    data_u16 = data.astype(np.int16).view(np.uint16)
    with open(filename, 'w') as f:
        for val in data_u16.flat:
            f.write(f"{val:04X}\n")
    print(f"  Written: {filename}  ({data.size} entries × 16-bit)")


def write_hex_32(filename: str, data: np.ndarray) -> None:
    """Write 32-bit signed int array as hex (8 chars per line, 2's complement)."""
    data_u32 = data.astype(np.int32).view(np.uint32)
    with open(filename, 'w') as f:
        for val in data_u32.flat:
            f.write(f"{val:08X}\n")
    print(f"  Written: {filename}  ({data.size} entries × 32-bit)")


# =============================================================================
# WEIGHT EXPORT
# =============================================================================

def export_conv_weights(
    weight_float: np.ndarray,   # shape: [out_ch, in_ch, kH, kW]
    bias_float:   np.ndarray,   # shape: [out_ch]  — FOLDED bias (float)
    w_hex_file:   str,
    b_hex_file:   str,
    conv_name:    str
) -> None:
    """
    Flatten and quantise conv weights and biases, then write hex files.

    Weight flattening order:
      for oc in range(out_ch):
        for ic in range(in_ch):
          for tap in range(9):   # row-major: (0,0)(0,1)(0,2)(1,0)...(2,2)
            w_flat[oc * in_ch * 9 + ic * 9 + tap] = weight[oc, ic, kr, kc]

    This matches the weight address computation in ldc_block1.v:
      addr = out_ch * (in_ch * 9) + in_ch * 9 + tap
    """
    out_ch, in_ch, kH, kW = weight_float.shape
    assert kH == 3 and kW == 3, "Only 3x3 kernels supported"

    print(f"\n[{conv_name}] weight shape: {weight_float.shape}, bias shape: {bias_float.shape}")

    # Flatten weights: row-major kernel tap order
    w_flat = weight_float.reshape(out_ch, in_ch, kH * kW)  # [oc, ic, 9]
    w_flat = w_flat.reshape(-1)                             # flatten all

    w_q  = to_q8_8(w_flat)
    b_q  = to_q16_16(bias_float)

    write_hex_16(w_hex_file, w_q)
    write_hex_32(b_hex_file, b_q)

    # Return the quantized arrays (reshaped back to [out_ch, in_ch, 3, 3] /
    # [out_ch]) so the golden forward pass can use the EXACT same quantized
    # values the RTL reads from these hex files, instead of the original
    # float tensors.
    w_q_reshaped = w_q.reshape(out_ch, in_ch, kH, kW)
    return w_q_reshaped, b_q


# =============================================================================
# IMAGE EXPORT
# =============================================================================

def export_image(
    image_float: np.ndarray,   # shape: [3, 8, 8] — single image, 3-channel
    filename:    str
) -> None:
    """
    Export input image in Q8.8 format.
    Streaming order: channel 0 all pixels (row-major), then ch1, ch2.
    """
    assert image_float.shape == (3, 8, 8), f"Expected [3,8,8], got {image_float.shape}"
    # [3, 8, 8] → [3, 64] → flatten to 192
    flat = image_float.reshape(3, 64).reshape(-1)
    q = to_q8_8(flat)
    write_hex_16(filename, q)
    print(f"  Image pixel range: [{image_float.min():.4f}, {image_float.max():.4f}]")
    return q.reshape(3, 8, 8)


# =============================================================================
# OUTPUT EXPORT
# =============================================================================

def export_expected_output(
    output_q: np.ndarray,  # shape: [16, 4, 4] -- ALREADY Q8.8 INTEGERS
    filename: str
) -> None:
    """
    Export expected output (already quantized Q8.8 integers from the
    bit-exact fixed-point golden forward pass -- see run_golden_forward).
    Order matches Conv2 output streaming order in ldc_block1.v:
      outer->inner: out_r, out_c, out_ch  (channel innermost)
    transpose(1,2,0) on [16,4,4] gives [4,4,16] (ch fastest), which is
    exactly this (r,c,ch) order once flattened row-major.
    """
    assert output_q.shape == (16, 4, 4), f"Expected [16,4,4], got {output_q.shape}"
    reordered = output_q.transpose(1, 2, 0).reshape(-1)  # [4,4,16] -> 256
    q = np.clip(reordered, INT16_MIN, INT16_MAX).astype(np.int16)
    write_hex_16(filename, q)
    print(f"  Output range (Q8.8 int): [{q.min()}, {q.max()}]  "
          f"(float: [{q.min()/Q8_8_SCALE:.4f}, {q.max()/Q8_8_SCALE:.4f}])")


# =============================================================================
# GOLDEN-REFERENCE FORWARD PASS (bit-exact fixed-point, matches RTL)
# =============================================================================
#
# IMPORTANT: this operates entirely in the INTEGER Q8.8 / Q16.16 domain on
# the ALREADY-QUANTIZED weight/bias/image arrays (the same ints written to
# the .hex files) -- not on the original float tensors. A float-math golden
# reference that only quantizes once at the very end is *more precise* than
# the hardware: the RTL quantizes (truncates via >>>8, which floors towards
# -infinity) the conv1+ReLU intermediate result BEFORE it ever reaches
# conv2. That intermediate truncation loses a small amount of precision per
# channel, and summing 16 of those slightly-low intermediate values in
# conv2 compounds into a multi-LSB systematic shortfall versus a pure-float
# reference. To match the hardware bit-for-bit, the golden reference must
# replicate that intermediate quantization, not skip it.

def conv2d_int_forward(x_q: np.ndarray, w_q: np.ndarray, b_q: np.ndarray,
                        stride: int, padding: int) -> np.ndarray:
    """
    Integer Q8.8 conv2d + bias + ReLU, with a TRUNCATING (floor) right
    shift by 8 -- matching ldc_block1.v's `>>> 8` on the Q16.16 accumulator
    (mac products are Q8.8 x Q8.8 = Q16.16; bias is already Q16.16; the
    final >>>8 brings the Q16.16 result back down to Q8.8 by FLOORING,
    not rounding).

    x_q : int16/int64 array [in_ch, H, W]      -- Q8.8 integers
    w_q : int16/int64 array [out_ch, in_ch,3,3] -- Q8.8 integers
    b_q : int32/int64 array [out_ch]            -- Q16.16 integers

    Returns int64 array [out_ch, out_H, out_W] of Q8.8 integers, already
    ReLU'd and ready to feed into the next stage (or write to hex).
    """
    in_ch, H, W = x_q.shape
    out_ch = w_q.shape[0]
    out_H = (H + 2 * padding - 3) // stride + 1
    out_W = (W + 2 * padding - 3) // stride + 1

    x_q64 = x_q.astype(np.int64)
    w_q64 = w_q.astype(np.int64)
    b_q64 = b_q.astype(np.int64)

    x_pad = np.pad(x_q64, ((0, 0), (padding, padding), (padding, padding)))
    out = np.zeros((out_ch, out_H, out_W), dtype=np.int64)

    for oc in range(out_ch):
        for r in range(out_H):
            for c in range(out_W):
                patch = x_pad[:, r * stride:r * stride + 3, c * stride:c * stride + 3]
                # Q8.8 * Q8.8 = Q16.16, summed over in_ch*9 taps
                acc_q16_16 = int(np.sum(patch * w_q64[oc])) + int(b_q64[oc])
                # Floor (arithmetic) right shift by 8: Python's native >>
                # on a Python int floors towards -infinity for negative
                # values too, which is exactly what Verilog's >>> does on
                # a signed value -- NOT the same as truncation-towards-zero.
                shifted_q8_8 = acc_q16_16 >> 8
                relu_val = shifted_q8_8 if shifted_q8_8 > 0 else 0
                # Saturate to int16 range, matching the 16-bit registers
                relu_val = max(INT16_MIN, min(INT16_MAX, relu_val))
                out[oc, r, c] = relu_val

    return out


def run_golden_forward(image_q, conv1_w_q, conv1_b_q,
                        conv2_w_q, conv2_b_q) -> np.ndarray:
    """
    Block1 = Conv1(stride=2, pad=1) -> ReLU -> Conv2(stride=1, pad=1) -> ReLU
    Bit-exact fixed-point version. All inputs are already-quantized integer
    arrays (Q8.8 for weights/image, Q16.16 for biases) -- the same values
    written into the .hex files the RTL reads.

    Matches ldc_block1.v address generation:
      c1_src_r/c = out_r*2 + kr - 1, out_c*2 + kc - 1   (stride 2, pad 1)
      c2_src_r/c = out_r   + kr - 1, out_c   + kc - 1   (stride 1, pad 1)
    """
    mid_q = conv2d_int_forward(image_q, conv1_w_q, conv1_b_q,
                                stride=2, padding=1)        # [16,4,4] Q8.8 int
    out_q = conv2d_int_forward(mid_q, conv2_w_q, conv2_b_q,
                                stride=1, padding=1)        # [16,4,4] Q8.8 int
    return out_q


# =============================================================================
# MAIN — EDIT THIS SECTION TO PLUG IN YOUR MODEL WEIGHTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  LDC Block1 Hex File Generator")
    print("  Q8.8 weights/pixels | Q16.16 biases")
    print("=" * 60)

    # ------------------------------------------------------------------
    # EDIT HERE: replace with your actual tensors from the Python golden
    # reference.  All inputs should be float32 numpy arrays.
    #
    # If using PyTorch:
    #   import torch
    #   conv1_w = model.double_conv[0].weight.detach().numpy()  # [16,3,3,3]
    #   conv1_b = folded_bias_conv1.detach().numpy()            # [16] float
    #   conv2_w = model.double_conv[3].weight.detach().numpy()  # [16,16,3,3]
    #   conv2_b = folded_bias_conv2.detach().numpy()            # [16] float
    #
    # FOLDED BIAS:
    #   The BN-folded bias absorbs both the conv bias and the BN shift.
    #   Compute it in Python as:
    #     w_fold = conv_w * (bn_gamma / bn_std)[:, None, None, None]
    #     b_fold = bn_beta + (conv_b - bn_mean) * bn_gamma / bn_std
    #   where bn_std = sqrt(bn_var + epsilon).
    # ------------------------------------------------------------------

    # ---- PLACEHOLDER: random weights for smoke-test compilation ----
    # Replace these with your actual model tensors:
    np.random.seed(42)
    conv1_w_float = np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1
    conv1_b_float = np.random.randn(16).astype(np.float32) * 0.05
    conv2_w_float = np.random.randn(16, 16, 3, 3).astype(np.float32) * 0.1
    conv2_b_float = np.random.randn(16).astype(np.float32) * 0.05

    # Fake RGB image in [0,1] range (Q8.8: values scale to ~0..256 in raw)
    # Normalise to [-0.5, 0.5] first if your model expects that.
    image_float = np.random.rand(3, 8, 8).astype(np.float32)

    # ---- Export weights/biases/image first -- capture their quantized
    #      integer values so the golden forward pass uses EXACTLY what
    #      the RTL reads from these same hex files. ----
    conv1_w_q, conv1_b_q = export_conv_weights(
        conv1_w_float, conv1_b_float,
        "conv1_weights.hex", "conv1_biases.hex", "Conv1")

    conv2_w_q, conv2_b_q = export_conv_weights(
        conv2_w_float, conv2_b_float,
        "conv2_weights.hex", "conv2_biases.hex", "Conv2")

    image_q = export_image(image_float, "input_image.hex")

    # ---- Bit-exact fixed-point golden forward pass (operates on the
    #      quantized integers above, not the original floats) ----
    expected_output_q = run_golden_forward(
        image_q, conv1_w_q, conv1_b_q,
        conv2_w_q, conv2_b_q)

    export_expected_output(expected_output_q, "expected_output.hex")

    print("\n[DONE] All 6 hex files generated.")
    print("       Copy them to the same directory as your Verilog simulation.")