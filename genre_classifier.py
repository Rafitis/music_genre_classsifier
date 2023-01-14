import json

# import pickle

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
from torchsummary import summary

from fma_dataset import FMADataset
from cnn_model import CNN_Model

JSON_FILE = r"data.json"
PICKLE_FILE = r"data.pickle"

EPOCHS = 10


def load_data(json_path):

    with open(json_path, "r", encoding="utf8") as fp:
        data = json.load(fp=fp)

    mfccs = np.array(data["mfcc"], dtype=object)
    labels = np.array(data["labels"], dtype=object)

    print(len(set(labels)))
    return mfccs, labels


def train_one_epoch(model, data_loader, loss_func, optimizer):
    for inputs, targets in data_loader:

        predictions = model(inputs)
        loss = loss_func(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_func, optimizer, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_func, optimizer)
        print("----------------------")
    print("Training Complete")


def main():
    ANNOTATIONS_FILE = r"data\test_dataset.csv"
    AUDIO_DIR = r"data\test"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    # Usar GPU si la hubiera.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} V{torch.version.cuda} device.")

    # Preprar Dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )
    fma_data = FMADataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    data_loader = DataLoader(fma_data, batch_size=64, shuffle=True)

    # Create CNN
    cnn_net = CNN_Model(num_classes=16).to(device)
    summary(cnn_net, input_size=(1, 64, 44))
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_net.parameters(), lr=0.0001)

    # Entrenar la red
    train(cnn_net, data_loader, loss_func, optimizer, EPOCHS)

    torch.save(cnn_net.state_dict(), "cnn_net.pth")
    print("Model Saved!")


if __name__ == "__main__":

    main()
