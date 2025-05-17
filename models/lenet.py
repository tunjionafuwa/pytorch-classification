"""
LeNet-5 model implementation in PyTorch, that was proposed by 'Yann LeCun et al. in 1998'.
The original paper uses Tanh activation functions and average pooling layers.
The model can be adaped using ReLU activation functions and max pooling layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, input_shape:int=1, num_classes:int=10) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.tanh = nn.Tanh()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.pool(self.tanh(self.conv1(x)))
        x = self.pool(self.tanh(self.conv2(x)))
        x = self.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=1)
        

def lenet(**kwargs) -> LeNet:
    """Create a LeNet model."""
    return LeNet(**kwargs)


if __name__ == "__main__":
    # Test the model
    model = lenet(input_shape=1, num_classes=10)

    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    print(output) 
