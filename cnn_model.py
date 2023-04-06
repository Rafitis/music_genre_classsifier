from torch import nn

class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, pooling=2, dropout=0.1) -> None:
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=1)
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
    def __init__(self, num_classes):
        super(CNN_Model, self).__init__()
        
        self.conv1 = Conv_2d(input_channels=4, output_channels=64, kernel_size=5)
        self.conv2 = Conv_2d(input_channels=64, output_channels=128, kernel_size=5)
        self.conv4 = Conv_2d(input_channels=128, output_channels=256, kernel_size=5)
        self.conv5 = Conv_2d(input_channels=256, output_channels=512, kernel_size=5)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512*14*14, num_classes)

    def forward(self, input):
        output = self.conv1(input)  
        output = self.conv2(output)                         
        output = self.conv4(output)   
        output = self.conv5(output)
        output = self.bn5(output)
        # output = torch.mean(output, dim=3)
        # output, _ = torch.max(output, dim=2)
        print(output.shape)
        
        output = output.reshape(len(output), -1)
        print(output.shape)
        output = self.fc1(output)

        return output