import os
from ast import literal_eval
import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class FMADataset(Dataset):
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)

        signal, _ = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, index):
        format_index = f"{int(self.annotations.iloc[index, 0]):06}"
        path = os.path.join(self.audio_dir, format_index + ".mp3")
        return path

    def _get_audio_sample_label(self, index):
        col_index = self.annotations.columns.get_loc("track_genres")
        genre_data = self.annotations.iloc[index, col_index]
        # Función que cambia un string a lista de dict
        genre_data = list(literal_eval(genre_data))[0]
        # data["mapping"].append(genre_data["genre_title"])
        label = genre_data["genre_id"]
        return label


def main():
    ANNOTATIONS_FILE = r"data\fma_metadata\raw_tracks.csv"
    AUDIO_DIR = r"data\test"
    fma = FMADataset(ANNOTATIONS_FILE, AUDIO_DIR)

    print(f"Num Audios {len(fma)}")
    _, label = fma[0]
    print(label)


if __name__ == "__main__":

    main()
