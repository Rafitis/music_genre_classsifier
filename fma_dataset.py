import os
from ast import literal_eval
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchvision.io import read_image
from torch.utils.data import Dataset
from skimage import io
from mapping_genres_ids import MAPPING_IDs
import matplotlib.pyplot as plt
import torchvision.transforms as T
class FMADataset(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        transformation,
        target_sample_rate,
        num_samples,
        device,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)

        # signal, sr = torchaudio.load(audio_sample_path)
        signal = read_image(audio_sample_path)
        signal = self._resize_image(signal)
        # signal = signal.to(self.device)
        # signal = self._resample_if_necessary(signal, sr)
        # signal = self._mix_down_if_necessary(signal)
        # signal = self._cut_if_necessary(signal)
        # signal = self._right_pad_if_necessary(signal)
        # signal = self.transformation(signal)

        # label = torch.tensor(label).to(self.device)
        return signal, label

    def _resize_image(self, signal):
        transform = T.Resize(255)
        signal = transform(signal)
        return signal

    def _spec_to_image(spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

    def _cut_if_necessary(self, signal):
        # signal -> Tensor (1, num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
        resampler = resampler.to(self.device)
        if sr != self.target_sample_rate:
            # print(f"Audio resampled to {self.target_sample_rate} KHz")
            signal = resampler(signal)  # pylint: disable=not-callable
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:  # La señal tiene más de un canal
            signal = torch.mean(signal, dim=0, keepdims=True)
        return signal

    def _get_audio_sample_path(self, index):
        format_index = f"{int(self.annotations.iloc[index, 0]):06}"
        path = os.path.join(self.audio_dir, format_index + ".png")
        return path

    def _get_audio_sample_label(self, index):

        col_index = self.annotations.columns.get_loc("track_genres")
        genre_data = self.annotations.iloc[index, col_index]
        # Función que cambia un string a lista de dict
        genre_data = list(literal_eval(genre_data))[0]
        # data["mapping"].append(genre_data["genre_title"])
        label = genre_data["genre_id"]
        return MAPPING_IDs[label]  # Se mapea la label


def _create_custom_csv(anno_file, audio_dir):
    df = pd.read_csv(anno_file)
    tracks_id = []
    for _, _, filenames in os.walk(audio_dir):
        for file in filenames:
            tracks_id.append(int(file.split(".")[0]))
    new_df = df.loc[df["track_id"].isin(tracks_id)]

    new_df.to_csv("full_dataset.csv", index=False)


def main():
    ANNOTATIONS_FILE = r"data\fma_metadata\raw_tracks.csv"
    AUDIO_DIR = r"data\mel_spec"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}...")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )

    fma = FMADataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    print(f"Num Audios {len(fma)}")
    train_data, label = fma[0]
    print(train_data.size())
    img = train_data[0].squeeze().cpu()
    print(img.shape, img)
    plt.imshow(img, cmap='gray')
    plt.show()
    print(label)

    _create_custom_csv(ANNOTATIONS_FILE, AUDIO_DIR)


if __name__ == "__main__":

    main()
