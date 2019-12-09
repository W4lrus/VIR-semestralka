import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np


class Policy(nn.Module):
    def __init__(self, tanh=False, std_fixed=True):
        super(Policy, self).__init__()  # just a stupid basic CNN

        self.tanh = tanh
        self.act_dim = 3

        self.cnv1 = nn.Conv2d(3, 16, 3, dilation=2)
        self.m1 = nn.InstanceNorm2d(16)
        self.cnv2 = nn.Conv2d(16, 32, 3)
        self.m2 = nn.InstanceNorm2d(32)
        self.cnv3 = nn.Conv2d(32, 32, 3)
        self.m3 = nn.InstanceNorm2d(32)

        vector = T.ones((1, 3, 144, 256))
        size = T.prod(T.tensor(self.cnv3(self.cnv2(self.cnv1(vector))).shape))  # compute tensor elements

        self.fc4 = nn.Linear(size, self.act_dim)

        T.nn.init.kaiming_normal_(self.cnv1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.cnv2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.cnv3.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='linear')

        if std_fixed:
            self.log_std = T.zeros(1, self.act_dim)
        else:
            self.log_std = nn.Parameter(T.zeros(1, self.act_dim))

    def forward(self, x):
        x = F.leaky_relu(self.m1(self.cnv1(x)))
        x = F.leaky_relu(self.m2(self.cnv2(x)))
        x = F.leaky_relu(self.m3(self.cnv3(x)))
        x = x.flatten()
        if self.tanh:
            x = T.tanh(self.fc4(x))
        else:
            x = self.fc4(x)
        return x

    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))

    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)