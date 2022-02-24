from policy.policy import Policy

import os
import numpy as np
import pcp_utils
from pcp_utils.utils import Config
from pcp_utils.load_ddpg import load_policy

class GraspFromCenterController(Policy):
    class Config(Config):
        policy_name = "grasp_from_rim_controller"
        policy_model_path = ""
        model_name = None
        max_path_length = 110

    def __init__(self, config:Config):
        self.config = config
        self.policy_name = config.policy_name
        self.max_path_length = config.max_path_length

    def convert_action_based_on_robot_type(self, action):
        if self.env.action_space.shape[0] == 8:
            return self.convert_robot_4dof_to_8dof(action)
        else:
            return action

    def run_forwards(self, env, num_rollouts, obj, path_length=None, render=True):
        self.render = render
        self.env = env
        acc_reward = 0
        acc_success = 0
        #obj_xpos = env.env.sim.data.get_body_xpos('object0').copy()
        #obj_xmat = env.env.sim.data.get_body_xmat('object0').copy()
        self.obj = obj
        for iter_id in range(num_rollouts):
            obs = env.reset()
            #ep_actions, ep_observations, ep_infos 
            success, cur_reward = self.goToGoal(env, obs)
            print("ITERATION NUMBER ", iter_id, success)
            acc_reward += cur_reward
            acc_success += success
        success_rate = acc_success/num_rollouts
        avg_reward = acc_reward/num_rollouts
        return {'avg_reward':avg_reward, 'success_rate':success_rate}

    def get_object_rel_pos(self, env, lastObs):
        agentPos = lastObs['observation'][:3]
        objectPos = lastObs['observation'][3:6]
        object_rel_pos = lastObs['observation'][6:9]
        obj_xpos, obj_xmat = env.get_object_pos("object0")
        bbox_points = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(self.obj.obj_xml_file, obj_xpos, obj_xmat, self.obj.scale)
        bounds, center, extents = pcp_utils.np_vis.get_bbox_attribs(bbox_points)
        object_center_rel_pos = center - agentPos
        return object_center_rel_pos, center, extents
    
    def goToGoal(self, env, lastObs):
        goal = lastObs['desired_goal']


        episodeAcs = []
        episodeObs = []
        episodeInfo = []
        cur_reward = []
        object_center_rel_pos, center, extents = self.get_object_rel_pos(env, lastObs)

        # first make the gripper go slightly above the object
        object_oriented_goal = object_center_rel_pos.copy()
        object_oriented_goal[2] += extents[2]/2.0 # add height of half bbox
        object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object
    
        timeStep = 0 #count the total number of timesteps
        episodeObs.append(lastObs)
        while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep < self.max_path_length:
            self.env_render()
            action = np.zeros(4,)
            object_oriented_goal = object_center_rel_pos.copy()
            object_oriented_goal[2] += 0.03
    
            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]*3
            action[len(action)-1] = 0.05 #open

            action = self.convert_action_based_on_robot_type(action)
            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            episodeAcs.append(action)
            episodeInfo.append(info)
            obsDataNew['observation'][8] += extents[2]/2.0
            episodeObs.append(obsDataNew)
            timeStep += 1

            #objectPos = obsDataNew['observation'][3:6]
            #object_rel_pos = obsDataNew['observation'][6:9]
            object_center_rel_pos, center, extents = self.get_object_rel_pos(env, obsDataNew)
            object_center_rel_pos[2] +=  extents[2]/2.0



        object_center_rel_pos, center, extents = self.get_object_rel_pos(env, obsDataNew)
        # move down a bit
        while np.linalg.norm(object_center_rel_pos) >= 0.005 and timeStep < self.max_path_length :
            self.env_render()
            action = np.zeros(4,)
            for i in range(len(object_center_rel_pos)):
                action[i] = object_center_rel_pos[i]*6
    
            action[len(action)-1] = -0.005
    
            action = self.convert_action_based_on_robot_type(action)
            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            episodeAcs.append(action)
            episodeInfo.append(info)
            obsDataNew['observation'][8] += extents[2]/2.0 - 0.02

            episodeObs.append(obsDataNew)
            timeStep += 1

            #objectPos = obsDataNew['observation'][3:6]
            #object_rel_pos = obsDataNew['observation'][6:9]
            object_center_rel_pos, center, extents = self.get_object_rel_pos(env, obsDataNew)
            object_center_rel_pos[2] += extents[2]/2.0 - 0.02



        # close gripper
        if timeStep < self.max_path_length:
            for i in range(30):
                self.env_render()
                action = np.zeros(4,)
                action[len(action)-1] = -0.025
                action = self.convert_action_based_on_robot_type(action)
                obsDataNew, reward, done, info = env.step(action)
                cur_reward.append(reward)               
                episodeAcs.append(action)
                episodeInfo.append(info)
                obsDataNew['observation'][8] += extents[2]/2.0- 0.02
                episodeObs.append(obsDataNew)
                timeStep += 1

        objectPos = obsDataNew['observation'][3:6]
        # moving toward goal
        while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep < self.max_path_length :
            self.env_render()
            action = np.zeros(4,)
            for i in range(len(goal - objectPos)):
                action[i] = (goal - objectPos)[i]*6
    
            action[len(action)-1] = -0.025
    
            action = self.convert_action_based_on_robot_type(action)
            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            episodeAcs.append(action)
            episodeInfo.append(info)
            obsDataNew['observation'][8] += extents[2]/2.0
            episodeObs.append(obsDataNew)
            timeStep += 1
            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9] 

        while timeStep < self.max_path_length: #env._max_episode_steps: #limit the number of timesteps in the episode to a fixed duration
            self.env_render()
            action = np.zeros(4,)
            action[len(action)-1] = -0.025 # keep the gripper closed
            action = self.convert_action_based_on_robot_type(action)
            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            episodeAcs.append(action)
            episodeInfo.append(info)
            obsDataNew['observation'][8] += extents[2]/2.0
            episodeObs.append(obsDataNew)
            timeStep += 1
            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
    
        success = 0
        curr_reward = np.sum(cur_reward)
        if np.sum(curr_reward) > -1 * self.max_path_length and cur_reward[-1] == 0: # env._max_episode_steps:
            success = 1

        return success, curr_reward
        
        #return episodeAcs, episodeObs, episodeInfo
