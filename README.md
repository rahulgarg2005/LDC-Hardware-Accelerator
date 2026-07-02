# Hardware Acceleration of Lightweight Dense CNN (LDC) for Edge Detection

This repository contains the synthesizable, high-throughput structural Verilog RTL design and hardware acceleration architecture for the **Lightweight Dense CNN (LDC)** edge detection network, optimized for edge-computing FPGA fabrics.

---

## 📊 Algorithmic Comparison & Design Motivation

Before designing the custom RTL accelerator fabric, a comprehensive structural and mathematical evaluation was conducted comparing traditional spatial filters with deep learning-based edge detection.

| Criteria | Sobel Filter | Canny Edge Detector | LDC (Lightweight Dense CNN) |
| :--- | :--- | :--- | :--- |
| **Mathematical Base** | First-order derivative (gradient intensity vectors). | Multi-stage algorithm (Gaussian blur, Non-max suppression, Hysteresis). | Parameter-efficient Dense Deep Learning Layer blocks. |
| **Hardware Mapping** | Simple $3\times3$ spatial convolution matrix (no multipliers needed, shift-add operation). | Complex due to non-maximum suppression conditional checks and dynamic buffering. | High MAC intensive, parallel spatial computation blocks. |
| **Edge Semantics** | Captures high-frequency noise; misses deep contextual boundary definitions. | Sharp thin edges but sensitive to lighting variations and manual thresholds. | Robust **Context-aware semantic edges** across complex real-world illumination. |
| **Hardware Footprint** | Low hardware footprint (few LUTs, no BRAM required). | Moderate; requires local register arrays for structural pipeline checks. | **Ultra-optimized:** Shrinks parameter footprint to just **1.9% of DexiNed** while staying within 0.6% of SOTA ODS accuracy. |

---

## 🛠 Hardware Architecture Contributions

- **Signed Q8.8 Fixed-Point Math Core:** Translated standard floating-point PyTorch weights/gradients into silicon-optimized 16-bit Signed Q8.8 arithmetic to optimize physical DSP block constraints.
- **2D Convolution Line Buffers:** Configured multi-row streaming line structures to enable sequential pixel-stream feeding without stalling the FPGA pipeline matrix.
- **Combinational Tapping Pipeline:** Integrated combinational tapping logic (`assign` interfaces) across multi-channel MAC (Multiply-Accumulate) structural processing nodes to wipe out multi-cycle loop latencies.
- **Differential Verification:** Built a custom Python golden reference pipeline simulation testbench architecture verifying cycle-accurate bit-level compatibility between RTL logic states and model weight tensors.

---

## 📂 Repository Structure

- `/rtl` - Synthesizable structural Verilog source files (MAC, line buffers, processing blocks).
- `/tb` - Self-checking testbenches and automated test vectors.
- `/utils` - PyTorch models, evaluation baselines, and Golden Reference verification scripts.
- `/docs` - Complete detailed technical PDF manual, LaTeX scripts, and theoretical background breakdown.
