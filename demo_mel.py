import matplotlib.pyplot as plt
import numpy as np

from utils_mel import inv_melspectrogram, save_wav

if __name__ == '__main__':
    mel = np.load('mel_outputs.npy')
    plt.imshow(mel, aspect='auto', origin='bottom', interpolation='none')
    plt.show()

    print("Mel spectrograms dim: ")
    print(mel.shape)

    audio = inv_melspectrogram(mel)
    save_wav(audio, 'output.wav')
