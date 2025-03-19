import torch
from torch import nn

import pufferlib.models
import numpy as np

NUM_POSITION_FREQUENCES = 16

class MadronaFPSLSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256):
        super().__init__(env, policy, input_size, hidden_size)

class Policy(nn.Module):
    def __init__(self, env, num_embed_channels=64, hidden_size=256, **kwargs):
        super().__init__()

        self.num_self_features = (
            env.obs['self_obs'].shape[-1] +
            np.product(env.obs['fwd_lidar'].shape[1:]) +
            np.product(env.obs['rear_lidar'].shape[1:]) +
            env.obs['reward_coefs'].shape[-1]
        )

        self.num_teammate_features = env.obs['teammates'].shape[-1]
        self.num_opponent_features = env.obs['opponents'].shape[-1]
        self.num_opponent_last_known_features = env.obs['opponents_last_known'].shape[-1]

        self.num_features = { k: v.shape[1:] for k, v in env.obs.items() }

        self.num_agents_per_env = env.num_agents_per_env
        self.team_size = env.team_size

        self.self_encode = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(
                    self.num_self_features, num_embed_channels)),
            #nn.LayerNorm(),
            nn.ReLU(),
        )

        self.teammates_encode = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(self.num_teammate_features, num_embed_channels)),
            #nn.LayerNorm(),
            nn.ReLU(),
        )

        self.opponents_encode = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(self.num_opponent_features, num_embed_channels)),
            #nn.LayerNorm(),
            nn.ReLU(),
        )

        self.opponents_last_known_encode = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(self.num_opponent_last_known_features,
                          num_embed_channels)),
            #nn.LayerNorm(),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(num_embed_channels * 4,
                          hidden_size)),
            #nn.LayerNorm(),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, hidden_size)),
            #nn.LayerNorm(),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, hidden_size)),
            #nn.LayerNorm(),
            nn.ReLU()
        )

        self.action_nvec = tuple(env.single_action_space.nvec)
        num_logits = np.sum(self.action_nvec)

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, num_logits), std=0.01)

        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

        self.is_continuous = False
        self.hidden_size = hidden_size


    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        self_ob = observations[:, 0:self.num_self_features]

        num_teammates = self.team_size - 1

        teammate_ob_start = self.num_self_features
        teammate_ob_end = teammate_ob_start + self.num_teammate_features * num_teammates
        teammate_ob = observations[:, teammate_ob_start:teammate_ob_end]
        teammate_ob = teammate_ob.view(
            teammate_ob.shape[0], num_teammates, self.num_teammate_features)

        opponent_ob_start = teammate_ob_end
        opponent_ob_end = opponent_ob_start + self.num_opponent_features * self.team_size
        opponent_ob = observations[:, opponent_ob_start:opponent_ob_end]
        opponent_ob = opponent_ob.view(
            opponent_ob.shape[0], self.team_size, self.num_opponent_features)

        opponent_last_known_ob_start = opponent_ob_end
        opponent_last_known_ob_end = opponent_last_known_ob_start + self.num_opponent_last_known_features * self.team_size
        opponent_last_known_ob = observations[:, opponent_last_known_ob_start:opponent_last_known_ob_end]
        opponent_last_known_ob = opponent_last_known_ob.view(
            opponent_last_known_ob.shape[0], self.team_size, self.num_opponent_last_known_features)

        self_features = self.self_encode(self_ob)

        teammates_features = self.teammates_encode(teammate_ob)
        opponents_features = self.opponents_encode(opponent_ob)
        opponents_last_known_features = self.opponents_last_known_encode(
            opponent_last_known_ob)

        teammates_features, _ = torch.max(teammates_features, dim=1)
        opponents_features, _ = torch.max(opponents_features, dim=1)
        opponents_last_known_features, _ = torch.max(opponents_last_known_features, dim=1)

        features = torch.cat(
            (self_features, teammates_features, opponents_features,
             opponents_last_known_features), dim=-1)

        return self.mlp(features), None


    def decode_actions(self, flat_hidden, lookup, concat=None):
        logits = self.actor(flat_hidden)
        value = self.critic(flat_hidden)

        return logits.split(self.action_nvec, dim=-1), value
