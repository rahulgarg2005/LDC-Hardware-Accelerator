"""
LDC Block 1 — Golden Reference Script
=======================================
PURPOSE:
    Run ONLY Block 1 (DoubleConvBlock) of LDC on a tiny 8x8 test image
    and dump every intermediate and final value so you can verify your
    Verilog implementation produces the exact same numbers.

WHAT THIS GIVES YOU:
    1. The fixed test input (8x8 RGB image, deterministic)
    2. Weights of every conv layer (for loading into Verilog ROM)
    3. Output after Conv1, BN1, ReLU1, Conv2, BN2, ReLU2
    4. A fixed-point (Q8.8) version of everything — ready for Verilog
    5. A .txt file with ALL values — your Verilog verification checklist

HOW TO USE:
    python golden_reference.py
    Then open golden_reference_output.txt — those are your target values.
    Build your Verilog conv unit and check it matches line by line.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# ─────────────────────────────────────────────
# 1.  Replicate Block 1 from modelb4.py exactly
# ─────────────────────────────────────────────

class DoubleConvBlock(nn.Module):
    """Exact copy of DoubleConvBlock from modelb4.py"""
    def __init__(self, in_features, mid_features,
                 out_features=None, stride=1, use_act=True):
        super().__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_features)
        self.relu  = nn.ReLU(inplace=False)   # inplace=False so we can inspect

    def forward_verbose(self, x):
        """Forward pass that returns every intermediate tensor"""
        intermediates = {}
        intermediates['input'] = x.clone()

        out = self.conv1(x)
        intermediates['after_conv1'] = out.clone()

        out = self.bn1(out)
        intermediates['after_bn1'] = out.clone()

        out = self.relu(out)
        intermediates['after_relu1'] = out.clone()

        out = self.conv2(out)
        intermediates['after_conv2'] = out.clone()

        out = self.bn2(out)
        intermediates['after_bn2'] = out.clone()

        if self.use_act:
            out = self.relu(out)
        intermediates['final_output'] = out.clone()

        return out, intermediates

# ─────────────────────────────────────────────
# 2.  Fixed-point helper  (Q8.8 format)
#     value_int = round(float_value * 2^8)
#     To recover float: value_int / 256.0
# ─────────────────────────────────────────────

FRAC_BITS = 8
SCALE     = 2 ** FRAC_BITS   # 256

def to_fixed(x: np.ndarray) -> np.ndarray:
    """Convert float32 numpy array to Q8.8 integers (int32)"""
    return np.round(x * SCALE).astype(np.int32)

def from_fixed(x: np.ndarray) -> np.ndarray:
    """Convert Q8.8 integers back to float32 for error checking"""
    return x.astype(np.float32) / SCALE

# ─────────────────────────────────────────────
# 3.  Create a deterministic 8×8 test image
#     (same every run — critical for Verilog verification)
# ─────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)

# Batch=1, Channels=3 (RGB), Height=8, Width=8
# Values in [0,1] like a normalised image
INPUT_H, INPUT_W = 8, 8
test_input = torch.rand(1, 3, INPUT_H, INPUT_W)

# Also create a simple ramp input for extra debugging ease
ramp_input = torch.zeros(1, 3, INPUT_H, INPUT_W)
for c in range(3):
    for i in range(INPUT_H):
        for j in range(INPUT_W):
            ramp_input[0, c, i, j] = ((c * 64 + i * 8 + j) % 256) / 255.0

# ─────────────────────────────────────────────
# 4.  Initialise Block 1 with fixed seed weights
#     (Xavier init from the original code)
# ─────────────────────────────────────────────

torch.manual_seed(0)   # fixed seed so weights are reproducible

block1 = DoubleConvBlock(3, 16, 16, stride=2)  # exactly as in LDC.__init__

# Xavier init (matching weight_init in modelb4.py)
for m in block1.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

block1.eval()   # eval mode: BN uses running stats (no batch noise)

# ─────────────────────────────────────────────
# 5.  Run the forward pass — capture everything
# ─────────────────────────────────────────────

with torch.no_grad():
    output_rand, steps_rand = block1.forward_verbose(test_input)
    output_ramp, steps_ramp = block1.forward_verbose(ramp_input)

# ─────────────────────────────────────────────
# 6.  Extract conv weights (what goes into Verilog ROM)
# ─────────────────────────────────────────────

w1 = block1.conv1.weight.detach().numpy()   # shape: [16, 3, 3, 3]
b1 = block1.conv1.bias.detach().numpy()     # shape: [16]
w2 = block1.conv2.weight.detach().numpy()   # shape: [16, 16, 3, 3]
b2 = block1.conv2.bias.detach().numpy()     # shape: [16]

# BN parameters (you'll pre-fold these into conv bias in Verilog)
bn1_gamma  = block1.bn1.weight.detach().numpy()
bn1_beta   = block1.bn1.bias.detach().numpy()
bn1_mean   = block1.bn1.running_mean.detach().numpy()
bn1_var    = block1.bn1.running_var.detach().numpy()
bn1_eps    = block1.bn1.eps

bn2_gamma  = block1.bn2.weight.detach().numpy()
bn2_beta   = block1.bn2.bias.detach().numpy()
bn2_mean   = block1.bn2.running_mean.detach().numpy()
bn2_var    = block1.bn2.running_var.detach().numpy()
bn2_eps    = block1.bn2.eps

# ─────────────────────────────────────────────
# 7.  Pre-fold BN into Conv weights
#     In hardware, running BN separately is expensive.
#     Instead: fold gamma/beta/mean/var into the conv
#     weights and bias as constants. This is standard.
#
#     w_folded = w * (gamma / sqrt(var + eps))
#     b_folded = (b - mean) * (gamma / sqrt(var + eps)) + beta
# ─────────────────────────────────────────────

def fold_bn(w, b, gamma, beta, mean, var, eps):
    """Fold BatchNorm into preceding Conv weights"""
    std      = np.sqrt(var + eps)           # [C_out]
    scale    = gamma / std                  # [C_out]
    w_folded = w * scale[:, None, None, None]
    b_folded = (b - mean) * scale + beta
    return w_folded, b_folded

w1f, b1f = fold_bn(w1, b1, bn1_gamma, bn1_beta, bn1_mean, bn1_var, bn1_eps)
w2f, b2f = fold_bn(w2, b2, bn2_gamma, bn2_beta, bn2_mean, bn2_var, bn2_eps)

# Verify folded weights reproduce same output
with torch.no_grad():
    # Manual folded conv
    x = test_input
    # Conv1 folded
    x_c1 = F.conv2d(x, torch.tensor(w1f).float(),
                     torch.tensor(b1f).float(), stride=2, padding=1)
    x_r1 = F.relu(x_c1)
    # Conv2 folded
    x_c2 = F.conv2d(x_r1, torch.tensor(w2f).float(),
                     torch.tensor(b2f).float(), stride=1, padding=1)
    x_r2 = F.relu(x_c2)

folded_match = np.allclose(
    x_r2.numpy(),
    steps_rand['final_output'].numpy(),
    atol=1e-4
)

# ─────────────────────────────────────────────
# 8.  Convert everything to Q8.8 fixed-point
# ─────────────────────────────────────────────

inp_fp    = to_fixed(test_input.numpy())
w1f_fp    = to_fixed(w1f)
b1f_fp    = to_fixed(b1f)
w2f_fp    = to_fixed(w2f)
b2f_fp    = to_fixed(b2f)

out_c1_fp = to_fixed(steps_rand['after_conv1'].numpy())
out_r1_fp = to_fixed(steps_rand['after_relu1'].numpy())
out_c2_fp = to_fixed(steps_rand['after_conv2'].numpy())
out_r2_fp = to_fixed(steps_rand['final_output'].numpy())

# ─────────────────────────────────────────────
# 9.  Write everything to the output text file
# ─────────────────────────────────────────────

lines = []
def s(txt=""): lines.append(txt)
def h(txt):    s("=" * 60); s(f"  {txt}"); s("=" * 60)
def sub(txt):  s(f"\n--- {txt} ---")

h("LDC Block 1 — Golden Reference Values")
s("Generated for Verilog verification")
s(f"Fixed-point format : Q8.8  (multiply floats by {SCALE})")
s(f"Input size         : 1 x 3 x {INPUT_H} x {INPUT_W}")
s(f"Output size        : 1 x 16 x {INPUT_H//2} x {INPUT_W//2}  (stride=2)")
s(f"BN folded into conv: {'YES — verified match' if folded_match else 'ERROR'}")

# ── Input ──
h("TEST INPUT  (float32)")
s("Shape: [batch=1, channels=3, H=8, W=8]")
s("Use this exact tensor as your Verilog test input.")
for c in range(3):
    s(f"\nChannel {c} (float):")
    for row in range(INPUT_H):
        vals = [f"{test_input[0,c,row,j].item():7.4f}" for j in range(INPUT_W)]
        s("  " + "  ".join(vals))

s()
h("TEST INPUT  (Q8.8 fixed-point integers)")
s("Divide by 256 to get back to float. Load these into your Verilog.")
for c in range(3):
    s(f"\nChannel {c} (Q8.8):")
    for row in range(INPUT_H):
        vals = [f"{inp_fp[0,c,row,j]:5d}" for j in range(INPUT_W)]
        s("  " + "  ".join(vals))

# ── Folded weights ──
h("FOLDED CONV1 WEIGHTS  (w1_folded)")
s("Shape: [out_ch=16, in_ch=3, kH=3, kW=3]")
s("BN1 already folded in. Load these as ROM in your Verilog.")
s("(float values):")
for oc in range(16):
    s(f"\n  Output channel {oc:2d}:")
    for ic in range(3):
        s(f"    Input ch {ic}: " +
          str(np.round(w1f[oc, ic], 5).tolist()))

s()
s("Folded Conv1 weights (Q8.8):")
for oc in range(16):
    s(f"\n  Output channel {oc:2d}:")
    for ic in range(3):
        s(f"    Input ch {ic}: " + str(w1f_fp[oc, ic].tolist()))

s()
s("Folded Conv1 bias (float): " + str(np.round(b1f, 5).tolist()))
s("Folded Conv1 bias (Q8.8) : " + str(b1f_fp.tolist()))

h("FOLDED CONV2 WEIGHTS  (w2_folded)")
s("Shape: [out_ch=16, in_ch=16, kH=3, kW=3]")
s("(Showing first 2 output channels for brevity)")
for oc in range(2):
    s(f"\n  Output channel {oc:2d} (float):")
    for ic in range(16):
        s(f"    Input ch {ic:2d}: " +
          str(np.round(w2f[oc, ic], 5).tolist()))
s(f"\n  ... (see full array in numpy files)")
s("\nFolded Conv2 bias (float): " + str(np.round(b2f, 5).tolist()))
s("Folded Conv2 bias (Q8.8) : " + str(b2f_fp.tolist()))

# ── Intermediate outputs (the golden values) ──
h("GOLDEN INTERMEDIATE VALUES")
s("These are the TARGET values your Verilog must match.")
s("Input: random 8x8 image (seed=42), weights (seed=0)")

sub("After Conv1 + BN1  (before ReLU1) — float")
s("Shape: [1, 16, 4, 4]  (stride=2 halves spatial dims)")
for ch in range(4):   # show first 4 channels
    s(f"\n  Channel {ch} (float):")
    for row in range(4):
        vals = [f"{steps_rand['after_bn1'][0,ch,row,j].item():8.4f}" for j in range(4)]
        s("  " + "  ".join(vals))

sub("After ReLU1 — float  (negatives become 0)")
for ch in range(4):
    s(f"\n  Channel {ch} (float):")
    for row in range(4):
        vals = [f"{steps_rand['after_relu1'][0,ch,row,j].item():8.4f}" for j in range(4)]
        s("  " + "  ".join(vals))

sub("After ReLU1 — Q8.8 fixed-point  <- VERILOG TARGET 1")
s("These are the values after your first MAC+ReLU unit should produce.")
for ch in range(4):
    s(f"\n  Channel {ch} (Q8.8):")
    for row in range(4):
        vals = [f"{out_r1_fp[0,ch,row,j]:6d}" for j in range(4)]
        s("  " + "  ".join(vals))

sub("FINAL OUTPUT — After Conv2 + BN2 + ReLU2 — float")
s("Shape: [1, 16, 4, 4]")
for ch in range(4):
    s(f"\n  Channel {ch} (float):")
    for row in range(4):
        vals = [f"{steps_rand['final_output'][0,ch,row,j].item():8.4f}" for j in range(4)]
        s("  " + "  ".join(vals))

sub("FINAL OUTPUT — Q8.8 fixed-point  <- VERILOG TARGET 2")
s("These are the exact values your full Block 1 Verilog must output.")
for ch in range(16):
    s(f"\n  Channel {ch} (Q8.8):")
    for row in range(4):
        vals = [f"{out_r2_fp[0,ch,row,j]:6d}" for j in range(4)]
        s("  " + "  ".join(vals))

# ── Ramp input targets ──
h("RAMP INPUT GOLDEN VALUES  (easier to debug)")
s("A second test: input is a simple ramp pattern, easier to trace.")
s("Use this first when debugging — ramp patterns make errors obvious.")
sub("Ramp Input (Q8.8):")
for c in range(3):
    s(f"\nChannel {c}:")
    for row in range(INPUT_H):
        vals = [f"{to_fixed(ramp_input.numpy())[0,c,row,j]:5d}" for j in range(INPUT_W)]
        s("  " + "  ".join(vals))

ramp_out_fp = to_fixed(steps_ramp['final_output'].numpy())
sub("Ramp -> Block1 Final Output (Q8.8)  <- VERILOG TARGET 3")
for ch in range(16):
    s(f"\n  Channel {ch}:")
    for row in range(4):
        vals = [f"{ramp_out_fp[0,ch,row,j]:6d}" for j in range(4)]
        s("  " + "  ".join(vals))

# ── How to use in Verilog ──
h("HOW TO USE THESE VALUES IN VERILOG")
s("""
STEP 1 — Store weights in ROM
  In Vivado, declare a parameter array or .coe memory file
  with the Q8.8 weight values printed above.

STEP 2 — MAC unit
  For each output pixel at (row, col, out_ch):
    acc = 0
    for each in_ch (0..2):
      for ki in 0..2, kj in 0..2:
        pixel = input[in_ch][row*2+ki-1][col*2+kj-1]  // stride=2
        weight = w1f_fp[out_ch][in_ch][ki][kj]
        acc += pixel * weight                            // Q8.8 * Q8.8 = Q16.16
    acc = acc >> 8                                       // back to Q8.8
    acc += b1f_fp[out_ch]

STEP 3 — ReLU
  if (acc < 0) acc = 0;   // that's it — you know this from MUX!

STEP 4 — Verify
  Compare your Verilog output values against the Q8.8 values in
  "VERILOG TARGET 1" and "VERILOG TARGET 2" above.
  If they match within ±1 LSB (rounding), your implementation is correct.

STEP 5 — When all values match, you're ready for the PYNQ board.
""")

h("FIXED-POINT ARITHMETIC QUICK REFERENCE")
s("""
Format    : Q8.8  (8 integer bits, 8 fractional bits, total 16 bits signed)
Range     : -128.0  to  +127.996
Precision : 1/256 ≈ 0.0039  (smallest representable step)

Examples:
  Float   ->  Q8.8    ->  Hex
  0.5     ->  128     ->  0x0080
  1.0     ->  256     ->  0x0100
  -0.25   ->  -64     ->  0xFFC0
  0.0312  ->  8       ->  0x0008

Multiply rule:
  (a_Q8.8) * (b_Q8.8) = product in Q16.16
  Shift right 8 bits to get result back in Q8.8
  In Verilog: result = (a * b) >>> 8;

Overflow check:
  16-channel accumulation of 9 products: use 32-bit accumulator
  to be safe, then clip to 16-bit at the end.
""")

h("FILES SAVED")
s("golden_reference_output.txt  — this file (your checklist)")
s("block1_weights_float.npz     — all float weights/biases")
s("block1_weights_fixed.npz     — all Q8.8 weights/biases")
s("block1_targets_fixed.npz     — all intermediate Q8.8 targets")

# Write the text file
out_path = "golden_reference_output.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# Save numpy arrays
np.savez("block1_weights_float.npz",
         w1=w1, b1=b1, w2=w2, b2=b2,
         w1_folded=w1f, b1_folded=b1f,
         w2_folded=w2f, b2_folded=b2f)

np.savez("block1_weights_fixed.npz",
         w1_folded_fp=w1f_fp, b1_folded_fp=b1f_fp,
         w2_folded_fp=w2f_fp, b2_folded_fp=b2f_fp)

np.savez("block1_targets_fixed.npz",
         input_fp     = inp_fp,
         after_conv1  = to_fixed(steps_rand['after_conv1'].numpy()),
         after_relu1  = out_r1_fp,
         after_conv2  = to_fixed(steps_rand['after_conv2'].numpy()),
         final_output = out_r2_fp,
         ramp_output  = ramp_out_fp)

# ═══════════════════════════════════════════════════════
# BLOCK 2 — DoubleConvBlock(16→32, stride=1, on 4×4)
# ═══════════════════════════════════════════════════════

torch.manual_seed(1)   # different seed from Block 1
block2 = DoubleConvBlock(16, 32, 32, stride=1)

for m in block2.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

block2.eval()

# Block 2 input = Block 1 final output (float, shape [1,16,4,4])
block2_input = output_rand.detach()   # reuse Block 1 random-input result

with torch.no_grad():
    block2_out, block2_steps = block2.forward_verbose(block2_input)

# ── Fold BN into Conv weights ──────────────────────────
w1b2  = block2.conv1.weight.detach().numpy()  # [32,16,3,3]
b1b2  = block2.conv1.bias.detach().numpy()    # [32]
w2b2  = block2.conv2.weight.detach().numpy()  # [32,32,3,3]
b2b2  = block2.conv2.bias.detach().numpy()    # [32]

bn1b2_gamma = block2.bn1.weight.detach().numpy()
bn1b2_beta  = block2.bn1.bias.detach().numpy()
bn1b2_mean  = block2.bn1.running_mean.detach().numpy()
bn1b2_var   = block2.bn1.running_var.detach().numpy()
bn1b2_eps   = block2.bn1.eps

bn2b2_gamma = block2.bn2.weight.detach().numpy()
bn2b2_beta  = block2.bn2.bias.detach().numpy()
bn2b2_mean  = block2.bn2.running_mean.detach().numpy()
bn2b2_var   = block2.bn2.running_var.detach().numpy()
bn2b2_eps   = block2.bn2.eps

w1b2f, b1b2f = fold_bn(w1b2, b1b2, bn1b2_gamma, bn1b2_beta,
                        bn1b2_mean, bn1b2_var, bn1b2_eps)
w2b2f, b2b2f = fold_bn(w2b2, b2b2, bn2b2_gamma, bn2b2_beta,
                        bn2b2_mean, bn2b2_var, bn2b2_eps)

# ── Verify folding ─────────────────────────────────────
with torch.no_grad():
    xb = block2_input
    xb = F.relu(F.conv2d(xb, torch.tensor(w1b2f).float(),
                         torch.tensor(b1b2f).float(), stride=1, padding=1))
    xb = F.relu(F.conv2d(xb, torch.tensor(w2b2f).float(),
                         torch.tensor(b2b2f).float(), stride=1, padding=1))
b2_folded_match = np.allclose(xb.numpy(),
                               block2_steps['final_output'].numpy(), atol=1e-4)
print(f"Block2 BN folding: {'✓ verified' if b2_folded_match else '✗ ERROR'}")

# ── Quantise ───────────────────────────────────────────
w1b2f_fp = to_fixed(w1b2f)   # Q8.8 int16, shape [32,16,3,3]
b1b2f_fp = to_fixed(b1b2f)   # Q16.16 int32, shape [32]  ← NOTE: bias stays Q8.8
b2b2f_fp = to_fixed(b2b2f)   # Q8.8 int32
w2b2f_fp = to_fixed(w2b2f)   # Q8.8 int16, shape [32,32,3,3]

# Bias must be in Q16.16 for the Verilog accumulator (matches conv1_biases.hex format)
b1b2f_fp_q1616 = np.round(b1b2f * 65536).astype(np.int32)
b2b2f_fp_q1616 = np.round(b2b2f * 65536).astype(np.int32)

out_b2_fp = to_fixed(block2_steps['final_output'].numpy())  # [1,32,4,4]

# ── Write hex files ────────────────────────────────────
# conv1_block2_weights.hex  : 4608 entries × 16-bit (Q8.8)
# Flatten order: oc*144 + ic*9 + tap  (matches c1_w_addr in Verilog)
w1_flat = w1b2f_fp.reshape(32, 16, 9)   # [32,16,9]
with open("conv1_block2_weights.hex", "w") as f:
    for oc in range(32):
        for ic in range(16):
            for tap in range(9):
                v = int(w1_flat[oc, ic, tap]) & 0xFFFF
                f.write(f"{v:04X}\n")

# conv1_block2_biases.hex   : 32 entries × 32-bit (Q16.16)
with open("conv1_block2_biases.hex", "w") as f:
    for v in b1b2f_fp_q1616:
        f.write(f"{int(v) & 0xFFFFFFFF:08X}\n")

# conv2_block2_weights.hex  : 9216 entries × 16-bit (Q8.8)
# Flatten order: oc*288 + ic*9 + tap  (matches c2_w_addr in Verilog)
w2_flat = w2b2f_fp.reshape(32, 32, 9)   # [32,32,9]
with open("conv2_block2_weights.hex", "w") as f:
    for oc in range(32):
        for ic in range(32):
            for tap in range(9):
                v = int(w2_flat[oc, ic, tap]) & 0xFFFF
                f.write(f"{v:04X}\n")

# conv2_block2_biases.hex   : 32 entries × 32-bit (Q16.16)
with open("conv2_block2_biases.hex", "w") as f:
    for v in b2b2f_fp_q1616:
        f.write(f"{int(v) & 0xFFFFFFFF:08X}\n")

# expected_output_block2.hex : 512 entries × 16-bit (Q8.8)
# Streaming order matches Verilog FSM: out_r → out_c → out_ch
# i.e., transpose [32,4,4] → [4,4,32] then flatten
out_reordered = out_b2_fp[0].transpose(1, 2, 0).reshape(-1)  # [4,4,32]→512
with open("expected_output_block2.hex", "w") as f:
    for v in out_reordered:
        f.write(f"{int(v) & 0xFFFF:04X}\n")

print("Block2 hex files written:")
print("  conv1_block2_weights.hex  (4608 entries)")
print("  conv1_block2_biases.hex   (32 entries)")
print("  conv2_block2_weights.hex  (9216 entries)")
print("  conv2_block2_biases.hex   (32 entries)")
print("  expected_output_block2.hex (512 entries)")

# ─────────────────────────────────────────────
# 10.  Print summary to terminal
# ─────────────────────────────────────────────
print("=" * 56)
print("  LDC Block 1 — Golden Reference  DONE")
print("=" * 56)
print(f"  Input shape  : {list(test_input.shape)}")
print(f"  Output shape : {list(output_rand.shape)}")
print(f"  BN folding   : {'✓ verified' if folded_match else '✗ ERROR'}")
print()
print("  Files written:")
print(f"    {out_path}")
print(f"    block1_weights_float.npz")
print(f"    block1_weights_fixed.npz")
print(f"    block1_targets_fixed.npz")
print()
print("  Open golden_reference_output.txt")
print("  -> use it as your Verilog verification checklist")
print("=" * 56)

# ─────────────────────────────────────────────
# 11.  Quick sanity check printout
# ─────────────────────────────────────────────
print("\nSANITY CHECK — Final output Q8.8, channel 0, 4x4 grid:")
print("(These are your primary Verilog targets)\n")
for row in range(4):
    vals = [f"{out_r2_fp[0,0,row,j]:6d}" for j in range(4)]
    print("  " + "  ".join(vals))
print()
print("Float equivalents (divide above by 256):")
for row in range(4):
    vals = [f"{out_r2_fp[0,0,row,j]/256:7.4f}" for j in range(4)]
    print("  " + "  ".join(vals))
# ── Write Block 1 output as input features for Block 2 testbench ──────────
# Shape: [1, 16, 4, 4] → streaming order ch→row→col (matches S_LOAD_INPUT)
# i.e., for ch in 0..15: for r in 0..3: for c in 0..3: emit pixel
out_b1_for_b2 = out_r2_fp[0]  # shape [16, 4, 4], Q8.8 int32

with open("block1_output_features.hex", "w") as f:
    for ch in range(16):
        for r in range(4):
            for c in range(4):
                v = int(out_b1_for_b2[ch, r, c]) & 0xFFFF
                f.write(f"{v:04X}\n")

print("  block1_output_features.hex  (256 entries)")

# =====================================================================
# ADD THIS TO THE BOTTOM OF YOUR SCRIPT
# =====================================================================

# ==============================================================================
# MISSING PYTORCH BLOCK 2 DEFINITIONS (Added so the Opus code runs)
# ==============================================================================
print("\nInitializing Block 2 PyTorch model...")
block2 = DoubleConvBlock(16, 32, 32, stride=1)
for m in block2.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
block2.eval()

# Extract and fold Block 2 weights
w1b2, b1b2 = block2.conv1.weight.detach().numpy(), block2.conv1.bias.detach().numpy()
w2b2, b2b2 = block2.conv2.weight.detach().numpy(), block2.conv2.bias.detach().numpy()

bn1b2_gamma, bn1b2_beta = block2.bn1.weight.detach().numpy(), block2.bn1.bias.detach().numpy()
bn1b2_mean, bn1b2_var, bn1b2_eps = block2.bn1.running_mean.detach().numpy(), block2.bn1.running_var.detach().numpy(), block2.bn1.eps

bn2b2_gamma, bn2b2_beta = block2.bn2.weight.detach().numpy(), block2.bn2.bias.detach().numpy()
bn2b2_mean, bn2b2_var, bn2b2_eps = block2.bn2.running_mean.detach().numpy(), block2.bn2.running_var.detach().numpy(), block2.bn2.eps

w1b2f, b1b2f = fold_bn(w1b2, b1b2, bn1b2_gamma, bn1b2_beta, bn1b2_mean, bn1b2_var, bn1b2_eps)
w2b2f, b2b2f = fold_bn(w2b2, b2b2, bn2b2_gamma, bn2b2_beta, bn2b2_mean, bn2b2_var, bn2b2_eps)


# ==============================================================================
# OPUS BIT-ACCURATE VERILOG SIMULATION
# ==============================================================================

SCALE_Q88 = 2 ** 8       # 256
SCALE_Q1616 = 2 ** 16    # 65536

def to_q88(x: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(x * SCALE_Q88 + 0.5), -32768, 32767).astype(np.int32)

def to_q1616(x: np.ndarray) -> np.ndarray:
    return np.floor(x * SCALE_Q1616 + 0.5).astype(np.int32)

def sim_conv2d_q88(input_q88, weight_q88, bias_q1616, stride=1, apply_relu=True):
    C_in, H_in, W_in = input_q88.shape
    C_out = weight_q88.shape[0]
    H_out = (H_in + 2 * 1 - 3) // stride + 1  
    W_out = (W_in + 2 * 1 - 3) // stride + 1

    padded = np.zeros((C_in, H_in + 2, W_in + 2), dtype=np.int32)
    padded[:, 1:-1, 1:-1] = input_q88
    output = np.zeros((C_out, H_out, W_out), dtype=np.int32)

    for oc in range(C_out):
        for oh in range(H_out):
            for ow in range(W_out):
                acc = np.int64(0)  
                ih_base, iw_base = oh * stride, ow * stride
                
                for ic in range(C_in):
                    for kh in range(3):
                        for kw in range(3):
                            pixel = np.int32(padded[ic, ih_base + kh, iw_base + kw])
                            weight = np.int32(weight_q88[oc, ic, kh, kw])
                            acc += np.int64(pixel) * np.int64(weight)
                
                acc += np.int64(bias_q1616[oc])
                
                if acc >= 0:
                    result = int(acc >> 8)
                else:
                    result = -int((-acc) >> 8) if (-acc) % 256 == 0 else -int((-acc) >> 8) - 1
                
                result = max(-32768, min(32767, result))
                if apply_relu and result < 0:
                    result = 0
                
                output[oc, oh, ow] = result
    return output.astype(np.int32)


# ==============================================================================
# EXECUTE SIMULATION & HEX EXPORT
# ==============================================================================

input_q88 = to_q88(test_input.numpy()[0]) 

# Block 1
block1_conv1_out = sim_conv2d_q88(input_q88, to_q88(w1f), to_q1616(b1f), stride=2)
block1_final_out = sim_conv2d_q88(block1_conv1_out, to_q88(w2f), to_q1616(b2f), stride=1)

# Block 2
block2_conv1_out = sim_conv2d_q88(block1_final_out, to_q88(w1b2f), to_q1616(b1b2f), stride=1)
block2_final_out = sim_conv2d_q88(block2_conv1_out, to_q88(w2b2f), to_q1616(b2b2f), stride=1)


def export_hex_16bit(array_data, filename):
    with open(filename, 'w') as f:
        for val in array_data.flatten():
            f.write(f"{int(val) & 0xFFFF:04X}\n")

def export_hex_32bit_q1616(array_data_float, filename):
    with open(filename, 'w') as f:
        for val in array_data_float.flatten():
            f.write(f"{int(round(val * 65536.0)) & 0xFFFFFFFF:08X}\n")


print("\nExporting updated hex files for Vivado...")
# Export Block 1 dependencies
export_hex_16bit(input_q88, "input_image.hex")
export_hex_16bit(to_q88(w1f), "conv1_block1_weights.hex")
export_hex_32bit_q1616(b1f, "conv1_block1_biases.hex")
export_hex_16bit(to_q88(w2f), "conv2_block1_weights.hex")
export_hex_32bit_q1616(b2f, "conv2_block1_biases.hex")

# Export Block 2 dependencies
export_hex_16bit(to_q88(w1b2f), "conv1_block2_weights.hex")
export_hex_32bit_q1616(b1b2f, "conv1_block2_biases.hex")
export_hex_16bit(to_q88(w2b2f), "conv2_block2_weights.hex")
export_hex_32bit_q1616(b2b2f, "conv2_block2_biases.hex")

# THE FIX: Transpose the final output to Pixel-Major [H, W, C] and export!
hw_aligned_target = np.transpose(block2_final_out, (1, 2, 0))
export_hex_16bit(hw_aligned_target, "expected_output_block2.hex")

print("\nSUCCESS! Expected target is now Bit-Accurate and Pixel-Major aligned.")