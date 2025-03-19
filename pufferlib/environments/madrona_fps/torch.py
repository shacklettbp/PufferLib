import torch
from torch import nn

import pufferlib.models
import numpy as np

NUM_POSITION_FREQUENCIES = 16

class MadronaFPSLSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512):
        super().__init__(env, policy, input_size, hidden_size)

class Policy(nn.Module):
    def __init__(self, env, num_embed_channels=64, hidden_size=512, **kwargs):
        super().__init__()

        self.obs_unpack_info = env.obs_unpack_info

        fake_obs = self._unpack_obs(
            torch.zeros((1, env.total_flattened_obs_size), dtype=torch.float32))

        self.num_agents_per_env = env.num_agents_per_env
        self.team_size = env.team_size

        num_self_in_channels = fake_obs['self_obs'].shape[-1] + NUM_POSITION_FREQUENCIES * 3

        self.self_encode = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(num_self_in_channels, num_embed_channels)),
            nn.LayerNorm(num_embed_channels),
            nn.ReLU(),
        )

        def build_lidar_encoder(lidar_ob, num_conv_channels=8):
            num_conv_flattened = num_conv_channels * (lidar_ob.shape[-2] // 4)

            return nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Conv1d(
                        in_channels=lidar_ob.shape[-1] * lidar_ob.shape[-3],
                        out_channels=num_conv_channels,
                        kernel_size=3, stride=2, padding=1)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(
                    nn.Conv1d(
                        in_channels=num_conv_channels,
                        out_channels=num_conv_channels,
                        kernel_size=3, stride=2, padding=1)),
                nn.Flatten(),
                nn.LayerNorm(num_conv_flattened),
                nn.ReLU(),
                nn.Linear(num_conv_flattened, num_embed_channels),
                nn.LayerNorm(num_embed_channels),
                nn.ReLU(),
            )


        self.fwd_lidar_encode = build_lidar_encoder(fake_obs['fwd_lidar'])
        self.rear_lidar_encode = build_lidar_encoder(fake_obs['rear_lidar'])

        self.teammates_encode = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(fake_obs['teammates'].shape[-1], num_embed_channels)),
            nn.LayerNorm(num_embed_channels),
            nn.ReLU(),
        )

        self.opponents_encode = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(fake_obs['opponents'].shape[-1], num_embed_channels)),
            nn.LayerNorm(num_embed_channels),
            nn.ReLU(),
        )

        self.opponents_last_known_encode = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(fake_obs['opponents_last_known'].shape[-1], num_embed_channels)),
            nn.LayerNorm(num_embed_channels),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(num_embed_channels * 6, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
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

    def _unpack_obs(self, flattened):
        unpacked = {}
        for k, unpack_info in self.obs_unpack_info.items():
            sliced = flattened[:, unpack_info.flattened_offset:unpack_info.flattened_offset + unpack_info.num_flattened_channels]

            unpacked[k] = sliced.view(sliced.shape[0], *unpack_info.orig_shape)

        return unpacked


    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, flattened_obs):
        obs = self._unpack_obs(flattened_obs)

        def vaswani_positional_embedding(embed_size, pos):
            embedding = torch.empty(*pos.shape[:-1], embed_size, pos.shape[-1], 
                                    device=pos.device)
            
            for i in range(embed_size // 2):
                # Compute the scaled position: pos * (2^i) * pi
                v = pos * (2.0 ** i) * torch.pi
                sin_embedding = torch.sin(v)
                cos_embedding = torch.cos(v)
                
                embedding[..., 2 * i, :] = sin_embedding
                embedding[..., 2 * i + 1, :] = cos_embedding
            
            # Reshape to flatten the last two dimensions
            embedding = embedding.view(*pos.shape[:-1], -1)
            return embedding

        self_pos_enc = vaswani_positional_embedding(
            NUM_POSITION_FREQUENCIES, obs['self_pos'])

        self_features = self.self_encode(torch.cat([
                self_pos_enc,
                obs['self_obs'],
            ], dim=-1))
        
        fwd_lidar_reshaped = obs['fwd_lidar'].transpose(-1, -2)
        fwd_lidar_reshaped = fwd_lidar_reshaped.reshape(
            fwd_lidar_reshaped.shape[0], -1, fwd_lidar_reshaped.shape[-1])
        fwd_lidar_features = self.fwd_lidar_encode(fwd_lidar_reshaped)

        rear_lidar_reshaped = obs['rear_lidar'].transpose(-1, -2)
        rear_lidar_reshaped = rear_lidar_reshaped.reshape(
            rear_lidar_reshaped.shape[0], -1, rear_lidar_reshaped.shape[-1])
        rear_lidar_features = self.rear_lidar_encode(rear_lidar_reshaped)


        teammates_features = self.teammates_encode(obs['teammates'])
        opponents_features = self.opponents_encode(obs['opponents'])
        opponents_features = opponents_features * obs['opponent_masks']

        opponents_last_known_features = self.opponents_last_known_encode(
            obs['opponents_last_known'])

        teammates_features, _ = torch.max(teammates_features, dim=1)
        opponents_features, _ = torch.max(opponents_features, dim=1)
        opponents_last_known_features, _ = torch.max(opponents_last_known_features, dim=1)

        features = torch.cat(
            (self_features, fwd_lidar_features, rear_lidar_features,
             teammates_features, opponents_features,
             opponents_last_known_features), dim=-1)

        return self.mlp(features), None


    def decode_actions(self, flat_hidden, lookup, concat=None):
        logits = self.actor(flat_hidden)
        value = self.critic(flat_hidden)

        return logits.split(self.action_nvec, dim=-1), value
