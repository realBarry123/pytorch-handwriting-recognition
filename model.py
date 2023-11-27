
import torch.nn as nn
import torch.nn.functional as F  # F is just an interface with nice functions in it


# max_pool2d(): downscales an image, but in a nice way so that the edges are still clear
class Net(nn.Module):  # inherits nn.Module (base class for all nns)

    def __init__(self):
        super(Net, self).__init__()

        # define convolution layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)

        # define dropout function
        self.conv2_drop = nn.Dropout2d()

        # define fully connected layers
        self.fc1 = nn.Linear(in_features=320, out_features=50) # fc stands for fully connected
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        """
        Feeds a tensor into the network
        :param x: tensor containing an image
        :return: tensor containing the last layer
        """
        x = self.conv1(x)  # run through convolution layer 1
        x = F.max_pool2d(x, kernel_size=2)  # max pooling
        x = F.relu(x)  # apply squishification

        x = self.conv2(x)  # run through convolution layer 2
        x = self.conv2_drop(x)  # dropout
        x = F.max_pool2d(x, kernel_size=2)  # max pooling
        x = F.relu(x)  # relu squishification

        x = x.view(-1, 320)  # flatten tensor

        x = self.fc1(x)  # run through fully connected layer 1
        x = F.relu(x)  # relu squishification
        x = F.dropout(x, training=self.training)  # dropout

        x = self.fc2(x)  # run through fully connected layer 2
        x = F.log_softmax(x, dim=1)  # softmax squishification
        return x
