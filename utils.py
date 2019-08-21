import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from scipy.io.wavfile import read

griffin_lim_iters = 60
fft_size = 1024
hop_size = 256


def griffin_lim(stftm_matrix, shape, min_iter=20, max_iter=50, delta=20):
    y = np.random.random(shape)
    y_iter = []

    for i in range(max_iter):
        if i >= min_iter and (i - min_iter) % delta == 0:
            y_iter.append((y, i))
        stft_matrix = librosa.core.stft(y)
        stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
        y = librosa.core.istft(stft_matrix)
    y_iter.append((y, max_iter))

    return y_iter


# Based on https://github.com/librosa/librosa/issues/434
def _griffin_lim(S):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    for i in range(griffin_lim_iters):
        if i > 0:
            angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _stft(y):
    return librosa.stft(y=y, n_fft=fft_size, hop_length=hop_size)


def _istft(y):
    return librosa.istft(y, hop_length=hop_size)
