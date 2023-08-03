import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.0

class ResNetBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResNetBlock, self).__init__()

        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(output_size),
            nn.Dropout(dropout_value)
        )

        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(output_size),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(output_size)
        )

    def forward(self, x):
        x = self.convlayer1(x)
        r1 = self.resblock(x)
        return x+r1


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()

        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )
        
        self.layer1 = ResNetBlock(64, 128)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)
        )

        self.layer3 = ResNetBlock(256, 512)

        self.pool = nn.MaxPool2d(4, 4)

        self.fc = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        x = self.preplayer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)