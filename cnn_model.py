from torch import nn
import torchaudio


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, pooling=2, dropout=0.1) -> None:
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=2)
        # self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class CNN_Model(nn.Module):
    def __init__(
        self,
        num_channels=16,
        sample_rate=22050,
        n_fft=1024,
        f_min=0.0,
        f_max=11025.0,
        n_mels=128,
        num_classes=10,
    ) -> None:
        super().__init__()

        # Mel_Spectrogram
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.input_bn = nn.BatchNorm2d(1)

        # Conv Layers
        self.layer1 = Conv_2d(1, 16)
        self.layer2 = Conv_2d(16, 64)
        self.layer3 = Conv_2d(64, 128)
        self.layer4 = Conv_2d(128, 256)
        # self.layer5 = Conv_2d(num_channels * 2, num_channels * 4)

        # Dense Layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(256 * 5 * 4, num_classes)
        self.dense_bn = nn.BatchNorm1d(num_channels * 4)
        self.dense2 = nn.Linear(num_channels * 4, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # input Preprocessing
        # x = self.melspec(x)
        # x = self.amplitude_to_db(x)

        # # input batch normalization
        # x = x.unsqueeze(1)
        # x = self.input_bn(x)

        # convolutional layers

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # reshape. (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        # x = x.reshape(len(x), -1)

        # dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.softmax(x)

        return x
