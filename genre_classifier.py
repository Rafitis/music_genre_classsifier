import json

# import pickle

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn

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


class FeedForwardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flatten_data = self.flatten(input_data)
        logits = self.dense_layers(flatten_data)
        predictions = self.softmax(logits)
        return predictions


def train_one_epoch(model, data_loader, loss_func, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        predictions = model(inputs)
        loss = loss_func(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_func, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_func, optimizer, device)
        print("----------------------")
    print("Training Complete")


def main():
    # Cargar datos
    inputs, targets = load_data(JSON_FILE)

    # Separar en train y test
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, targets, test_size=0.2)

    # Crear la NN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} V{torch.version.cuda} device.")

    dnn_net = FeedForwardModel().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dnn_net.parameters, lr=0.0001)

    # Entrenar la red
    train(dnn_net, X_train, loss_func, optimizer, device, EPOCHS)

    torch.save(dnn_net.state_dict(), "dnn_model.pth")
    print("Model Saved!")


if __name__ == "__main__":

    main()
