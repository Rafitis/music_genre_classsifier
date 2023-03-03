
import os
import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = r"data\fma_small"
N_MELS = 128
N_FFT = 2048

for dirpath, _, filenames in os.walk(DATASET_PATH):
    for filename in filenames:
        if f'{filename[:-4]}.png' in os.listdir(os.path.join('data', 'mel_spec')):
            print('Spectrogram already saved!')
            continue
        
        y, sr = librosa.load(os.path.join(dirpath, filename))
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)

        fig, ax = plt.subplots()
        mel_dB = librosa.power_to_db(mel, ref=np.max)
        
        img = librosa.display.specshow(mel_dB, sr=sr, ax=ax)
        print(f'{filename[:-4]}.png Spectrogram saved!')
        fig.savefig(os.path.join('data', 'mel_spec', f'{filename[:-4]}.png'))
        plt.close(fig)