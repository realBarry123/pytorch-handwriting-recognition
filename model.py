import torch.nn as nn
import torch.nn.functional as F  # F is just a static class with nice functions in it

# max_pool2d(): downscales an image, but in a nice way so that the edges are still clear

# convolving 2 lists:
# [1, 2, 3] * [4, 5, 6]
#    4  5  6
# 1  4  5  6
# 2  8  10 12
# 3  12 15 18
# [4, 5+8, 6+10+12, 12+15, 18]
# [4, 13, 28, 27, 18]


class Net(nn.Module):  # inherits nn.Module (base class for all nns)

    def __init__(self):
        super(Net, self).__init__()
        # kernel is the small grid that is moved in convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # dropout: chops random parts of the neural network to prevent overfitting
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x