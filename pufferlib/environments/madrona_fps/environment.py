import functools

import pufferlib
import pufferlib.emulation
import gymnasium
import numpy as np
import torch

class MadronaFPSPufferEnv(pufferlib.environment.PufferEnv):
    def __init__(self, sim, team_size, num_envs=1, buf=None):
        train_iface = sim.train_interface()
        print(train_iface)

        obs = {}
        #obs['self_pos'] = sim.self_pos().to_torch()
        obs['self_obs'] = sim.self_obs().to_torch()
        obs['fwd_lidar'] = sim.fwd_lidar().to_torch()
        obs['rear_lidar'] = sim.rear_lidar().to_torch()
        obs['reward_coefs'] = sim.reward_coefs().to_torch()
        obs['teammates'] = sim.teammates().to_torch()
        obs['opponents'] = sim.opponents().to_torch()
        obs['opponents_last_known'] = sim.opponents_last_known().to_torch()
        obs['opponent_masks'] = sim.opponent_masks().to_torch()

        self.flattened_obs_size = 0
        for k, v in obs.items():
            self.flattened_obs_size += np.product(v.shape[1:])

        self.single_observation_space = gymnasium.spaces.Box(low=0, high=0,
            shape=(self.flattened_obs_size,), dtype=np.float32)

        self.single_action_space = gymnasium.spaces.MultiDiscrete(
            [3, 8, 3, 3, 13, 7])

        num_agents_per_env = 2 * team_size

        self.num_agents = num_envs * num_agents_per_env
        self.num_agents_per_env = num_agents_per_env
        self.team_size = team_size

        super().__init__(buf)
        self.env = sim
        self.obs = obs

        self.sim_discrete_actions = sim.pvp_action_tensor().to_torch()
        self.sim_aim_actions = sim.aim_action_tensor().to_torch()
        self.sim_rewards = sim.reward_tensor().to_torch()
        self.sim_terminals = sim.done_tensor().to_torch()
        self.sim_resets = sim.reset_tensor().to_torch()

        self.truncations = torch.zeros((self.num_agents,), dtype=torch.float32,
                                       device=self.sim_resets.device)
        self.render_mode = 'raylib'

        self.foo = 0


    def _get_obs(self):
        out = torch.zeros((self.num_agents, self.flattened_obs_size),
                          dtype=torch.float32)

        cur_offset = 0
        for k, v in self.obs.items():
            flattened_v = v.view(self.num_agents, -1)

            end_offset = cur_offset+flattened_v.shape[-1]
            out[:, cur_offset:end_offset] = flattened_v.float()
            cur_offset = end_offset

        return out

    def reset(self, params=None):
        self.sim_resets[:] = 1
        self.env.step()
        self.sim_resets[:] = 0

        obs = self._get_obs()
        self.observations = obs
        infos = []

        self.terminals = self.sim_terminals[:, 0]
        
        return obs, infos

    def render(self):
        pass

    def close(self):
        pass

    def step(self, action):
        gpu_action = torch.tensor(action)
        pvp_actions = gpu_action[:, :4]
        aim_actions = gpu_action[:, 4:]

        self.sim_discrete_actions[:] = pvp_actions
        self.sim_aim_actions[:] = aim_actions

        self.env.step()

        obs = self._get_obs()
        self.observations = obs

        self.foo += 1

        infos = [{'foo': self.foo}]

        self.rewards = self.sim_rewards[:, 0]
        self.terminals = self.sim_terminals[:, 0]

        return obs, self.rewards, self.terminals, self.truncations, infos


def env_creator(name='Madrona-FPS'):
    return functools.partial(make, name)


def make(name, scene_path, num_envs=2048, buf=None, gpu_id=0):
    import madrona_mp_env
    from madrona_mp_env import Task, SimFlags

    assert scene_path != None

    sim_flags = SimFlags.Default
    game_mode = Task.Zone

    team_size = 6

    sim = madrona_mp_env.SimManager(
        exec_mode = madrona_mp_env.madrona.ExecMode.CUDA,
        gpu_id = gpu_id,
        num_worlds = num_envs,
        auto_reset = True,
        sim_flags = sim_flags,
        task_type = game_mode,
        team_size = team_size,
        rand_seed = 5,
        num_pbt_policies = 1,
        policy_history_size = 0,
        scene_path = scene_path,
        curriculum_data_path = None,
    )
    sim.init()

    return MadronaFPSPufferEnv(sim, 
                               team_size=team_size,
                               num_envs=num_envs,
                               buf=buf)
