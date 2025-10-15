GPU Data Streaming Demo

Description:
Simulates a real-time data stream (e.g., sensor waveform), filters it using an exponential moving average (EMA), optionally computes FFT, and visualizes results in real time. Designed to demonstrate GPU-accelerated data processing and streaming, with CPU fallback for systems without NVIDIA GPUs (like macOS).

Features:
- Real-time simulated sensor waveform streaming
- Exponential Moving Average (EMA) filtering
- Optional FFT visualization
- GPU acceleration using CuPy (if available)
- CPU fallback using NumPy (works on macOS, Linux, Windows)
- Saves final plot automatically

Installation:

1. Clone the repository:
   git clone https://github.com/tianwainanfeng/gpu-data-stream-demo.git
   cd gpu-data-stream-demo

2. Install dependencies:

CPU-only (works on macOS, Linux, Windows):
   pip install numpy matplotlib

Optional GPU support (Linux / Colab with NVIDIA GPU):
   pip install cupy-cuda12x

Note: CuPy requires NVIDIA GPU and CUDA. On macOS, the demo will automatically run on CPU.

Running the Demo:

To run the Python script:
   python stream_gpu_demo.py

- The plot will update in real time.
- The final plot will be saved as "filtered_plot.png" in the repository folder.

Running in Google Colab:

- Open the notebook:

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tianwainanfeng/gpu-data-stream-demo/blob/main/gpu_data_stream_demo.ipynb)

- In Colab, you can select Runtime → Change runtime type → GPU for GPU acceleration.
- Run all cells to execute the demo.

Example Output:

- Real-time filtered waveform / FFT plot
- Saved image: filtered_plot.png

Notes:

- Works on macOS, Linux, and Windows.
- On systems without CUDA, it will automatically use CPU.
- Designed as a minimal demo to show GPU/CPU data streaming and real-time visualization.

License: MIT License

