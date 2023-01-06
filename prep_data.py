import json
import os
import pickle
from ast import literal_eval

import librosa
import pandas as pd

DATASET_PATH = r"data\test"
METADATA_PATH = r"data\fma_metadata\raw_tracks.csv"
JSON_PATH = r"data.json"
PICKLE_PATH = r"data.pickle"


def read_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    return df


def save_mfcc(dataset_path, json_path, n_mfcc=13):

    # Dict data struct
    data = {"mapping": [], "mfcc": [], "labels": []}

    metadata = read_metadata(METADATA_PATH)

    for dirpath, _, filenames in os.walk(dataset_path):
        if dirpath is not dataset_path:

            for track in filenames:

                track_id = int(track.split(".")[0])
                genre_data = metadata.loc[metadata["track_id"] == track_id]["track_genres"].values[0]
                # Función que cambia un string a lista de dict
                genre_data = list(literal_eval(genre_data))[0]
                data["mapping"].append(genre_data["genre_title"])
                data["labels"].append(genre_data["genre_id"])

                # Extraer los MFCCs
                file_path = os.path.join(dirpath, track)
                signal, sr = librosa.load(file_path)
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, hop_length=2048)
                mfcc = mfcc.T

                data["mfcc"].append(mfcc.tolist())

                print(f"{track} processed")

    with open(json_path, "w", encoding="utf8") as fp:
        json.dump(data, fp, indent=4)
    with open(PICKLE_PATH, "wb") as pickle_file:
        pickle.dump(data, pickle_file)


save_mfcc(DATASET_PATH, JSON_PATH)
