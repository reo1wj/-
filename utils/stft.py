# utils/stft.py

import numpy as np
import cv2
from scipy.signal import stft

def iq_to_spectrogram(iq_sample):
    I, Q = iq_sample
    signal = I + 1j * Q

    _, _, Zxx = stft(signal, nperseg=64, noverlap=32)

    spec = np.abs(Zxx)
    spec = spec / (np.max(spec) + 1e-8)
    spec = (spec * 255).astype(np.uint8)

    spec = cv2.resize(spec, (128, 128))
    return spec