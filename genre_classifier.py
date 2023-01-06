import json
import pickle

import numpy as np

JSON_FILE = r"data.json"
PICKLE_FILE = r"data.pickle"


def load_data(json_path):

    with open(json_path, "r", encoding="utf8") as fp:
        data = json.load(fp=fp)

    mfccs = np.array(data["mfccs"])
    labels = np.array(data["labels"])

    return mfccs, labels


if __name__ == "__main__":

    # Cargar datos
    inputs, targets = load_data(JSON_FILE)
    # Separar en train y test

    # Crear la NN

    # Compilar la red

    # Entrenar la red
