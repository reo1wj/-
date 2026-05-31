# models/cnn.py

import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1,16,5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*16*16,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,num_classes)
        )

    def forward(self, x):
        return self.net(x)