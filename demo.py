import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
import soundfile


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


if __name__ == '__main__':
    filename = 'audios/007064.wav'
    # assume 1 channel wav file
    sr, data = read(filename)
    data = data.astype(np.float16)

    # print(data.dtype)

    # 由 STFT -> STFT magnitude
    stftm_matrix = np.abs(librosa.core.stft(data))
    # + random 模拟 modification
    stftm_matrix_modified = stftm_matrix + np.random.random(stftm_matrix.shape)
    # stftm_matrix_modified = stftm_matrix

    # Griffin-Lim 估计音频信号
    y_iters = griffin_lim(stftm_matrix_modified, data.shape)
    y = y_iters[0][0]

    print(y)

    plt.plot(y)
    plt.show()

    soundfile.write('out.wav', y, sr)
