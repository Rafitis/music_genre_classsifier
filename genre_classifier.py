import json

# import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
import torchvision
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
    running_loss = 0.0
    for _, data in enumerate(data_loader):
        inputs, labels = data

        predictions = model(inputs)
        loss = loss_func(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_func, optimizer, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_func, optimizer)
        print("----------------------")
    print("Training Complete")


def main():
    ANNOTATIONS_FILE = r"data\full_dataset.csv"
    AUDIO_DIR = r"data\fma_small"
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
    # cnn_model = CNN_Model(num_classes=103).to(device)
    cnn_model = torchvision.models.resnet50(num_classes=103)
    num_ftrs = cnn_model.fc.in_features
    cnn_model.fc = torch.nn.Linear(num_ftrs, 103)
    print(cnn_model)
    summary(cnn_model, input_size=(1, 64, 44))
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)

    # Entrenar la red
    train(cnn_model, data_loader, loss_func, optimizer, EPOCHS)

    torch.save(cnn_model.state_dict(), "cnn_model.pth")
    print("Model Saved!")


if __name__ == "__main__":

    main()
