import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, skip=4, channels=3, d_hidden=32, d_out=64, mean=0.5, std=0.2):
        super().__init__()
        self.mean = mean
        self.std = std
        self.conv1_1 = nn.Conv2d(skip*channels, 18, 1, groups=channels)
        self.conv1_2 = nn.Conv2d(skip*channels, 16, 1)

        self.conv2_1 = nn.Sequential(*[
            nn.Conv2d((18+16), d_hidden, 5, stride=3),
            nn.GroupNorm(4, d_hidden),
        ])
        self.conv2_2 = nn.Sequential(*[
            nn.Conv2d(d_hidden, d_hidden, 5, stride=3, groups=4),
            nn.GroupNorm(4, d_hidden),
        ])

        self.conv3_1 = nn.Sequential(*[
            nn.Conv2d(d_hidden, d_hidden, 3, stride=2, groups=4),
            nn.GroupNorm(4, d_hidden)
        ])
        self.conv3_2 = nn.Sequential(*[
            nn.Conv2d(d_hidden, d_out, 3, stride=2, groups=4),
            nn.GroupNorm(4, d_out)
        ])

    def forward(self, x):
        y = (x/255-self.mean) / self.std

        y1 = F.relu(self.conv1_1(y))
        y2 = F.relu(self.conv1_2(y))

        y = torch.cat([y1, y2], dim=1)
        y = F.relu(self.conv2_1(y))
        y = F.relu(self.conv2_2(y))

        y = F.relu(self.conv3_1(y))
        y = torch.tanh(self.conv3_2(y))
        return y


class Decoder(nn.Module):
    def __init__(self, skip=4, channels=3, d_in=64, d_hidden=32):
        super().__init__()

        self.conv1_1 = nn.Sequential(*[
            nn.ConvTranspose2d(d_in, d_hidden, 3, 2),
            nn.GroupNorm(4, d_hidden)
        ])
        self.conv1_2 = nn.Sequential(*[
            nn.ConvTranspose2d(d_hidden, d_hidden, 3, 2, groups=4),
            nn.GroupNorm(4, d_hidden)
        ])

        self.conv2_1 = nn.Sequential(*[
            nn.ConvTranspose2d(d_hidden, d_hidden, 5, 3, groups=4),
            nn.GroupNorm(4, d_hidden)
        ])
        self.conv2_2 = nn.Sequential(*[
            nn.ConvTranspose2d(d_hidden, d_hidden, 5, 3, groups=4),
            nn.GroupNorm(4, d_hidden)
        ])

        self.conv3_1 = nn.Sequential(*[
            nn.ConvTranspose2d(d_hidden, d_hidden, 3, 1),
            nn.GroupNorm(4, d_hidden)
        ])
        self.conv3_2 = nn.ConvTranspose2d(d_hidden, channels*skip, 3, 1)

    def forward(self, x):
        y = F.relu(self.conv1_1(x))
        y = F.relu(self.conv1_2(y))

        y = F.relu(self.conv2_1(y))
        y = F.relu(self.conv2_2(y))

        y = F.relu(self.conv3_1(y))
        y = self.conv3_2(y)
        y = (torch.tanh(y) + 1) / 2 * 255

        return y[:, :, 1:-2, 1:-2]


class Dynamics(nn.Module):
    def __init__(self, d_in=64*3*3, n_action=21, d_hidden=512):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_action = n_action

        self.lin1 = nn.Linear(d_in+n_action, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_in)
        self.lin3 = nn.Linear(d_hidden, 1)
        self.lin4 = nn.Linear(d_hidden,1)

    def forward(self, last_state, last_action):
        action_t = F.one_hot(last_action, num_classes=self.n_action)
        x = torch.cat([last_state, action_t], dim=1)

        y = F.relu(self.lin1(x))
        reward_hat = self.lin3(y)
        done_hat = self.lin4(y)
        h_next_hat = torch.tanh(self.lin2(y))

        return h_next_hat, reward_hat, done_hat


class WorldModel(nn.Module):
    def __init__(self, skip=4, channels=3,
                 mean=0.5, std=0.2,
                 d_hidden=32, d_conv=64,
                 n_world_grid=3, d_world_hidden=512,
                 n_actions=12):
        super().__init__()
        self.encoder = Encoder(skip, channels, d_hidden, d_conv, mean, std)
        self.decoder = Decoder(skip, channels, d_conv, d_hidden)

        d_world = n_world_grid * n_world_grid * d_conv
        self.d_conv = d_conv
        self.n_world_grid = n_world_grid
        self.dynamics = Dynamics(d_world, n_actions, d_world_hidden)

    def forward(self, x, action):
        bs = x.shape[0]
        h = self.encoder(x).reshape(bs, -1)
        h_next_hat, reward_hat, done_hat = self.dynamics(h, action)

        n = self.n_world_grid
        d = self.d_conv
        x_hat = self.decoder(h.reshape(bs, d, n, n))
        x_next_hat = self.decoder(h_next_hat.reshape(bs, d, n, n))
        return (h, h_next_hat), (reward_hat, done_hat), (x_hat, x_next_hat)


class ActorCritic(nn.Module):
    def __init__(self, d_in=64*3*3, d_hidden=512, n_actions=12):
        super().__init__()

        self.lin = nn.Linear(d_in, d_hidden)

        self.actor = nn.Linear(d_hidden, n_actions)
        self.value = nn.Linear(d_hidden, 1)

    def forward(self, x):
        h = self.lin(x)
        return self.actor(h), self.value(h)
