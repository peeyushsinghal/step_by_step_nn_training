import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.filter = nn.Conv2d(16, 8, 1, bias=False)
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 16, 3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
        )
        self.conv6 = nn.Conv2d(16, 10, 3, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.filter(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
