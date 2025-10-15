"""
Real-Time GPU Data Stream Demo
Author: Your Name
Description:
Simulates a continuous data stream (e.g., sensor waveform),
processes each chunk on GPU using CuPy for filtering & FFT,
and visualizes the processed results in real time.
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

# ----- Parameters -----
STREAM_RATE_HZ = 20          # chunks per second
CHUNK_SIZE = 2048            # samples per chunk
GPU_FILTER_ALPHA = 0.9       # exponential moving average
SHOW_FFT = True              # toggle to show frequency domain
DURATION_SEC = 10            # total runtime

# ----- Initialize -----
plt.ion()
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, CHUNK_SIZE)
ax.set_ylim(-2, 2)
ax.set_title("GPU Filtered Data Stream")
ax.set_xlabel("Sample index")
ax.set_ylabel("Amplitude")

# GPU buffer (persistent state for EMA filter)
prev_gpu = cp.zeros(CHUNK_SIZE, dtype=cp.float32)

# ----- Main streaming loop -----
start_time = time.time()
while time.time() - start_time < DURATION_SEC:
    # (1) Simulate sensor data on CPU
    t = np.linspace(0, 1, CHUNK_SIZE)
    noise = np.random.normal(0, 0.3, CHUNK_SIZE)
    signal = np.sin(2 * np.pi * 5 * t) + noise  # 5 Hz sine + noise

    # (2) Transfer to GPU
    gpu_data = cp.asarray(signal, dtype=cp.float32)

    # (3) GPU filtering (exponential moving average)
    gpu_filtered = GPU_FILTER_ALPHA * prev_gpu + (1 - GPU_FILTER_ALPHA) * gpu_data

    # (4) Optional FFT on GPU
    if SHOW_FFT:
        gpu_fft = cp.fft.fft(gpu_filtered)
        gpu_filtered = cp.abs(gpu_fft[:CHUNK_SIZE // 2])
        ax.set_xlim(0, CHUNK_SIZE // 2)
        ax.set_ylim(0, 200)

    # (5) Copy back to CPU for plotting
    filtered = cp.asnumpy(gpu_filtered)

    # (6) Update plot
    line.set_data(range(len(filtered)), filtered)
    plt.pause(1.0 / STREAM_RATE_HZ)

    # (7) Save state
    prev_gpu = gpu_filtered

plt.ioff()
plt.show()

