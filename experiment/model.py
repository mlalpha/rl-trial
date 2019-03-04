import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        filter_size_1 = 2
        self.conv1 = nn.Conv2d(3, 20, filter_size_1)
        self.conv1_bn = nn.BatchNorm2d(20)
        filter_size_2 = 4
        self.conv2 = nn.Conv2d(20, 20, filter_size_2)
        self.conv2_bn = nn.BatchNorm2d(20)
        flattened_size = (state_size[0]-filter_size_1-filter_size_2-2) \
                         *(state_size[1]-filter_size_1-filter_size_2-2) \
                         *20 // 16
        # print(flattened_size)
        self.fc1 = nn.Linear(
            flattened_size
            , fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        print(self)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(state)), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # return torch.sigmoid(self.fc3(x))
        return self.fc3(x)
