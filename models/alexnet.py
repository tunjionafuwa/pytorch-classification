import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, input_shape:int=3, num_classes:int=1000) -> None:
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(self.relu(self.conv5(x)))

        print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def alexnet(**kwargs) -> AlexNet:
    """Create an AlexNet model."""
    return AlexNet(**kwargs)

if __name__ == "__main__":
    # Test the model
    model = alexnet(input_shape=3, num_classes=1000)

    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output)



