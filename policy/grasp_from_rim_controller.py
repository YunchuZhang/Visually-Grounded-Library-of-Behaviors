from policy.policy import Policy

import os
import numpy as np
import pcp_utils
from pcp_utils.utils import Config
from pcp_utils.load_ddpg import load_policy

class GraspFromRimController(Policy):
    class Config(Config):
        policy_name = "grasp_from_rim_controller"
        policy_model_path = ""
        model_name = None

    def __init__(self, config:Config):
        self.config=config
        self.policy_name = config.policy_name

    def run_forwards(self, env, num_rollouts, obj=None,  path_length=None, render=True):

        self.render = render
        self.env = env
        acc_reward = 0
        acc_success = 0
        
        for iter_id in range(num_rollouts):
            print("ITERATION NUMBER ", iter_id)
            obs = env.reset()
            #ep_actions, ep_observations, ep_infos 
            success, cur_reward = self.goToGoal(env, obs)
            acc_reward += cur_reward
            acc_success += success
        success_rate = acc_success/num_rollouts
        avg_reward = acc_reward/num_rollouts
        return {'avg_reward':avg_reward, 'success_rate':success_rate}

    def convert_action_based_on_robot_type(self, action):
        if self.env.action_space.shape[0] == 8:
            return self.convert_robot_4dof_to_8dof(action)
        else:
            return action


    def goToGoal(self, env, lastObs):
        goal = lastObs['desired_goal']
        objectPos = lastObs['observation'][3:6]
        object_rel_pos = lastObs['observation'][6:9]
        episodeAcs = []
        episodeObs = []
        episodeInfo = []
    
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[1] -= 0.08
        object_oriented_goal[2] += 0.12 # first make the gripper go slightly above the object
    
        timeStep = 0 #count the total number of timesteps
        episodeObs.append(lastObs)
        cur_reward = []
    
        while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
            self.env_render()

            action = np.zeros(4,)
            object_oriented_goal = object_rel_pos.copy()
            object_oriented_goal[1] -= 0.08
            object_oriented_goal[2] += 0.12
    
            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]*6
    
            action[len(action)-1] = 0.05 #open


            action = self.convert_action_based_on_robot_type(action)
            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            timeStep += 1
    
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
    
            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
    
        while np.linalg.norm(object_rel_pos) >= 0.032 and timeStep <= env._max_episode_steps:
            self.env_render()
            action = np.zeros(4,)
            for i in range(len(object_rel_pos)):
                action[i] = object_rel_pos[i]*6
    
            action[len(action)-1] = 0.05
            action = self.convert_action_based_on_robot_type(action)
            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            timeStep += 1
    
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
    
            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
            object_rel_pos[1] -= 0.03
            # while True:
                # env.render()
    
        ## ... for properly grasping the cup before lifting ... ##
        for i in range(30):
            self.env_render()
            action = np.zeros(4,)
            action[len(action)-1] = -0.025
            action = self.convert_action_based_on_robot_type(action)
            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1
            cur_reward.append(reward)
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            #env.render()
    
        # while True: env.render()
    
        while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
            # while True:
                # env.render()
            # import ipdb; ipdb.set_trace()
            self.env_render()
            action = np.zeros(4,)
            for i in range(len(goal - objectPos)):
                action[i] = (goal - objectPos)[i]*6
    
            action[len(action)-1] = -0.005
            action = self.convert_action_based_on_robot_type(action)
            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            timeStep += 1
    
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
    
            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
    
        while True: #limit the number of timesteps in the episode to a fixed duration
            self.env_render()
            action = np.zeros(4,)
            action[len(action)-1] = -0.05 # keep the gripper closed
            action = self.convert_action_based_on_robot_type(action)
    
            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            timeStep += 1
    
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
    
            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
    
            if timeStep >= env._max_episode_steps: break #env._max_episode_steps: 70

        success = 0
        cur_reward = np.sum(cur_reward)
        if np.sum(cur_reward) > -1 * env._max_episode_steps:
            success = 1

        return success, cur_reward
        #return episodeAcs, episodeObs, episodeInfo