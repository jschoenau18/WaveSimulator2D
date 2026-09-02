"""
GPU/CPU compatibility layer. Uses cupy (GPU) when available, otherwise falls
back to numpy/scipy (CPU) so the simulator also runs on machines without an
NVIDIA GPU, e.g. macOS.
"""
try:
    import cupy as cp  # type: ignore[import-not-found]
    import cupyx.scipy.signal as cp_signal  # type: ignore[import-not-found]

    def to_numpy(arr):
        """ copy an array off the GPU """
        return arr.get()
except ImportError:
    import numpy as cp
    import scipy.signal as cp_signal

    def to_numpy(arr):
        """ arr is already a numpy array on CPU-only fallback """
        return arr
