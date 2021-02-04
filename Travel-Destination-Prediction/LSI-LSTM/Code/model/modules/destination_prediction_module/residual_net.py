import torch.nn as nn
import torch.nn.functional as F


class ResidualNet(nn.Module):
    # The residual Networks to train easily and more robust.
    def __init__(self, input_size, num_final_fcs=4, hidden_size=128):
        super(ResidualNet, self).__init__()

        self.input2hid = nn.Linear(input_size, hidden_size)

        self.residuals = nn.ModuleList()

        for i in range(num_final_fcs):
            self.residuals.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, inputs):
        hidden = F.leaky_relu(self.input2hid(inputs))

        for i in range(len(self.residuals)):
            residual = F.leaky_relu(self.residuals[i](hidden))
            hidden = hidden + residual
        return hidden