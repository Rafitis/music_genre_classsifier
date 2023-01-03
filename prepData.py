import os
import pandas as pd
import numpy as np
from ast import literal_eval

DATASET_PATH = r'data\test'
METADATA_PATH = r'data\fma_metadata\raw_tracks.csv'
JSON_PATH = r'data.json'

def read_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    return df

def save_mfcc(dataset_path, json_path, n_mfcc=13, num_segments=5):

    # Dict data struct
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    metadata = read_metadata(METADATA_PATH)
    # print(metadata.dtypes)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            for track in filenames:
                track_id = int(track.split('.')[0])
                genre_data = metadata.loc[metadata['track_id'] == track_id]['track_genres'].values[0]
                change_type = lambda genre_data: list(eval(genre_data))[0] # Función que cambia un string a lista de dict
                genre_data = change_type(genre_data)
                data['mapping'] = genre_data['genre_title']
                data['labels'] = genre_data['genre_id']

save_mfcc(DATASET_PATH, JSON_PATH)