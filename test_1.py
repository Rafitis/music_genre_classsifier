from ast import literal_eval
import os
import shutil

import pandas as pd
import soundfile as sf

AUDIOS_FOLDER = r"data\fma_small"
CSV_FILE = r"data\full_dataset.csv"


def copy_audios_to_root_folder():
    for dir_path, _, filenames in os.walk(AUDIOS_FOLDER):
        for file in filenames:
            shutil.copy(os.path.join(dir_path, file), r"data\tmp")
            print(f"File {os.path.join(dir_path, file)} copied to -> data/tmp")


def process_labels_dataset(file_csv):
    df = pd.read_csv(file_csv)

    # col_index = df.columns.get_loc("track_genres")
    # genre_data = df.iloc[:, col_index]
    # # Función que cambia un string a lista de dict
    # # data["mapping"].append(genre_data["genre_title"])
    label = df["track_genres"]
    labels = []
    for l in label:
        genre_data = list(literal_eval(l))[0]

        labels.append(int(genre_data["genre_id"]))
    print(len(set(labels)))
    new_labels = sorted(list(set(labels)))
    data_mapping = {}
    for i, l in enumerate(new_labels):
        data_mapping[str(l)] = i

    print(data_mapping)


def open_audio(audio_sample):
    audio_sample_path = os.path.join(AUDIOS_FOLDER, audio_sample)
    print(audio_sample)
    _, sr = sf.read(audio_sample_path)
    if not isinstance(sr, int):
        print(sr, audio_sample)


process_labels_dataset(CSV_FILE)

for audio in reversed(os.listdir(AUDIOS_FOLDER)):
    open_audio(audio)
