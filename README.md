# GPU Data Stream Demo

**Goal:** Minimal example of real-time GPU data processing using [CuPy](https://cupy.dev/).

## Features
- Simulates streaming sensor data.
- Sends each data chunk to GPU for filtering and FFT.
- Displays real-time results using Matplotlib.
- Compact, fully commented (under 150 lines).

## Requirements
```bash
pip install cupy-cuda12x numpy matplotlib

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/tianwainanfeng/gpu-data-stream-demo/blob/main/gpu_streaming_demo.ipynb)

