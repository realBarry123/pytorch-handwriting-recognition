
import torch.nn as nn
import torch.nn.functional as F  # F is just an interface with nice functions in it


# max_pool2d(): downscales an image, but in a nice way so that the edges are still clear
class Net(nn.Module):  # inherits nn.Module (base class for all nns)

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # dropout: chops random parts of the neural network to make it more adaptable & prevent overfitting

        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)

        x = x.view(-1, 320)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
