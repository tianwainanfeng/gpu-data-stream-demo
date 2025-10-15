"""
Real-Time GPU Data Stream Demo
Description:
Simulates a continuous data stream (e.g., sensor waveform),
processes each chunk on GPU using CuPy for filtering & FFT,
and visualizes the processed results in real time.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

try:
    import cupy as cp
    USE_GPU = True
    print("Using GPU (CuPy)")
except ImportError:
    cp = np #alias cp to numpy
    USE_GPU = False
    print("CuPy not found. Using CPU (NumPy)")

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

# Persistent state for EMA filter
prev = cp.zeros(CHUNK_SIZE, dtype=cp.float32)

# ----- Main streaming loop -----
start_time = time.time()
while time.time() - start_time < DURATION_SEC:
    # (1) Simulate sensor data on CPU
    t = np.linspace(0, 1, CHUNK_SIZE)
    noise = np.random.normal(0, 0.3, CHUNK_SIZE)
    signal = np.sin(2 * np.pi * 5 * t) + noise  # 5 Hz sine + noise

    # (2) Transfer to GPU
    data = cp.asarray(signal, dtype=cp.float32)

    # (3) GPU filtering (exponential moving average)
    filtered = GPU_FILTER_ALPHA * prev + (1 - GPU_FILTER_ALPHA) * data
    prev = filtered

    # (4) Optional FFT
    if SHOW_FFT:
        fft_data = cp.fft.fft(filtered)
        fft_magnitude = cp.abs(fft_data[:CHUNK_SIZE // 2])
        x_vals = range(CHUNK_SIZE // 2)
        y_vals = cp.asnumpy(fft_magnitude) if USE_GPU else fft_magnitude
        ax.set_xlim(0, CHUNK_SIZE // 2)
        ax.set_ylim(0, max(y_vals) * 1.2)
    else:
        x_vals = range(CHUNK_SIZE)
        y_vals = cp.asnumpy(filtered) if USE_GPU else filtered
        ax.set_xlim(0, CHUNK_SIZE)
        ax.set_ylim(-2, 2)

    # (5) Update plot
    #line.set_data(range(len(filtered)), filtered) # raw filtered waveform
    line.set_data(x_vals, cp.asnumpy(y_vals) if USE_GPU else y_vals)
    plt.pause(1.0 / STREAM_RATE_HZ)

print("Show plot")
plt.ioff()
#plt.show()

# Save the final plot
fig.savefig("filtered_plot.png", dpi=150)  # or .pdf/.svg
print("Saved final plot as filtered_plot.png")

print("Done")
