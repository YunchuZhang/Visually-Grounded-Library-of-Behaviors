from policy.policy import Policy

import os

import pcp_utils
from pcp_utils.utils import Config
from pcp_utils.load_ddpg import load_policy
#from pcp_utils.rollouts import simple_rollouts

class PytorchPolicy(Policy):
    class Config(Config):
        policy_name = "pytorch_policy"
        policy_model_path = ""
        model_name = None
        obs_arg= None

    def __init__(self, config:Config):
        self.config = config
        self.policy_name = config.policy_name

        expert_model_path = os.path.join(pcp_utils.utils.get_root_dir(), "trained_models")
        trained_model_path = os.path.join(expert_model_path, config.policy_model_path)
        # if model name is not specified, take the latest one
        if config.model_name == None:
             all_files = os.listdir(os.path.join(trained_model_path, "models"))
             latest_step = sorted([int(file[4:]) for file in all_files])[-1]
             config.model_name = 'save' + str(latest_step)
        self.policy_model = load_policy(os.path.join(expert_model_path, config.policy_model_path), config.model_name, obs_arg=self.config.obs_arg)

    def run_forwards(self, env, num_rollouts, path_length, obj=None, render=False):
        self.env = env
        stats = self.simple_rollouts(env,
                self.policy_model,
                num_rollouts=num_rollouts,
                path_length=path_length,
                obs_arg=self.config.obs_arg,
                render=render)
        return stats
    def rescale_observations(self, obs, scale):
        grip_pos = obs[0:3]
        obj_pos = obs[3:6]
    
        grip_pos = (grip_pos - obj_pos)*scale + grip_pos #reposition gripper as per the scale
        obj_pos[3] = obj_pos[2] * scale #rescale z coordinate
    
        obs[0:3] = grip_pos
        obs[3:6] = obj_pos
        return obs
    def convert_action_based_on_robot_type(self, action):
        if action.shape[0] == 4 and self.env.action_space.shape[0] == 8:
            return self.convert_robot_4dof_to_8dof(action)
        else:
            return action
    def simple_rollouts(self, env, policy, scale=1, num_rollouts=100, path_length=50, rescale_obs=False, obs_arg=None, render=True):
    
        avg_reward = 0
        success_rate = 0
        for _ in range(num_rollouts):
            obs = env.reset(obs_arg=obs_arg)
            cur_reward = 0
            for _ in range(path_length):
                actions, _, _, _ = policy.step(obs)
                actions = self.convert_action_based_on_robot_type(actions)

                obs, rew, done, _ = env.step(actions, obs_arg=obs_arg)
                if render:
                    env.render()
                if rescale_obs:
                    obs = self.rescale_observations(obs)
                cur_reward += rew
            if cur_reward > -1 * path_length:
                success_rate += 1
            avg_reward += cur_reward
            print("Reward {}".format(cur_reward))
        success_rate/=num_rollouts
        return {'avg_reward':avg_reward/num_rollouts, 'success_rate':success_rate}