import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from scipy.io.wavfile import read
from utils import griffin_lim

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

    # y = _griffin_lim(stftm_matrix_modified)

    print(y)

    plt.plot(y)
    plt.show()

    soundfile.write('out.wav', y, sr)
