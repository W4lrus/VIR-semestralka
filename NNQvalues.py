import torch.nn as nn
import torch as T
import numpy as np


class Policy(nn.Module):
    def __init__(self, image_dims=(3, 144, 256), action_space=2, tanh=False, std_fixed=True):
        super(Policy, self).__init__()  # just a stupid basic CNN

        self.tanh = tanh
        self.act_dim = action_space

        self.CNV = nn.Sequential(
            AlexNet(),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 4, 3, dilation=2),
            nn.InstanceNorm2d(4),
            nn.LeakyReLU(),

            nn.Conv2d(4, 7, 3),
            nn.MaxPool2d(2, 2),
            nn.InstanceNorm2d(10),

            nn.Conv2d(7, 10, 3),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
        )

        vector = T.ones((1, image_dims[0], image_dims[1], image_dims[2]))
        size = T.prod(T.tensor(self.CNV(vector).shape))  # compute tensor elements for FC input size

        self.FC = nn.Linear(size, self.act_dim)

        for module in self.CNV.modules():
            if isinstance(module, nn.Conv2d):
                T.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.FC.weight, mode='fan_in', nonlinearity='linear')

        if std_fixed:
            self.log_std = T.zeros(1, self.act_dim)
        else:
            self.log_std = nn.Parameter(T.zeros(1, self.act_dim))

    def forward(self, x):
        x = self.CNV(x)
        x = x.flatten(1)
        if self.tanh:
            x = T.tanh(self.FC(x)/20)
        else:
            x = self.FC(x)
        return x

    def soft_clip_grads(self, bnd=1):
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            # print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd

    def sample_action(self, s, random=True):
        if random:
            return T.normal(self.forward(s), T.exp(self.log_std))
        else:
            return self.forward(s)

    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        weights = T.load('./AlexNetWeights')

        self.CNN = nn.Conv2d(3, 64, 11, stride=3)
        state = self.CNN.state_dict()
        state['weight'] = weights
        self.CNN.load_state_dict(state)

        for param in self.CNN.parameters():  # freeze layer
            param.requires_grad = False

    def forward(self, x):
        return self.CNN(x)


if __name__ == '__main__':
    pass
    #model = T.hub.load('pytorch/vision:v0.4.2', 'alexnet', pretrained=True)

