import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = r"data\fma_small"
N_MELS = 128
N_FFT = 2048


def extract_mel_spec():
    for dirpath, _, filenames in os.walk(DATASET_PATH):
        for filename in filenames:
            if f"{filename[:-4]}.png" in os.listdir(os.path.join("data", "mel_spec")):
                print("Spectrogram already saved!")
                continue

            y, sr = librosa.load(os.path.join(dirpath, filename))

            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)

            mel_dB = librosa.power_to_db(mel, ref=np.max)  # type: ignore

            fig = plt.figure(figsize=[0.72, 0.72])
            ax = fig.add_subplot(111)  # type: ignore
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            librosa.display.specshow(mel_dB, sr=sr, ax=ax)

            fig.savefig(
                os.path.join("data", "mel_spec", f"{filename[:-4]}.png"),
                dpi=400,
                bbox_inches="tight",
                pad_inches=0,
            )
            print(f"{filename[:-4]}.png Spectrogram saved!")
            plt.close(fig)


if __name__ == "__main__":
    extract_mel_spec()
