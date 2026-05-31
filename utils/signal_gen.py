# utils/signal_gen.py

import numpy as np

def gen_cw(N=128):
    t = np.arange(N)
    return np.exp(1j * 2 * np.pi * 0.1 * t)

def gen_bpsk(N=128):
    bits = np.random.choice([1, -1], size=N)
    return bits.astype(np.complex64)

def gen_qpsk(N=128):
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=N)
    return symbols / np.sqrt(2)

def gen_bfsk(N=128):
    t = np.arange(N)
    f1, f2 = 0.05, 0.15
    bits = np.random.choice([0, 1], size=N)

    signal = np.zeros(N, dtype=complex)
    for i in range(N):
        f = f1 if bits[i] == 0 else f2
        signal[i] = np.exp(1j * 2 * np.pi * f * i)
    return signal

def gen_lfm(N=128):
    t = np.arange(N)
    f0 = 0.05
    k = 0.002
    return np.exp(1j * 2 * np.pi * (f0*t + 0.5*k*t**2))


def add_noise(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    snr = 10**(snr_db / 10)
    noise_power = signal_power / snr

    noise = np.sqrt(noise_power/2) * (
        np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape)
    )
    return signal + noise


def to_iq(signal):
    return np.stack([signal.real, signal.imag])