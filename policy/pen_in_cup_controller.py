from policy.policy import Policy

import os
import numpy as np
import pcp_utils
from pcp_utils.utils import Config
from pcp_utils.load_ddpg import load_policy

class PenincupController(Policy):
    class Config(Config):
        policy_name = "pen_in_cup_controller"
        policy_model_path = ""
        model_name = None
        max_path_length = 110

    def __init__(self, config:Config):
        self.config = config
        self.policy_name = config.policy_name
        self.max_path_length = config.max_path_length

    def run_forwards(self, env, num_rollouts, obj, path_length=None):
        acc_reward = 0
        acc_success = 0
        obj_xpos = env.env.sim.data.get_body_xpos('object0').copy()
        obj_xmat = env.env.sim.data.get_body_xmat('object0').copy()

        bbox_points = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(obj.obj_xml_file,obj_xpos,obj_xmat,obj.scale)
        bounds, center, extents = pcp_utils.np_vis.get_bbox_attribs(bbox_points)
        self.center, self.extents = center, extents

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


    def goToGoal(self, env, lastObs):
        goal = lastObs['desired_goal']
        objectPos = lastObs['observation'][3:6]
        object_rel_pos = lastObs['observation'][6:9]
        episodeAcs = []
        episodeObs = []
        episodeInfo = []
        cur_reward = []
        object_oriented_goal = object_rel_pos.copy()
        # object_oriented_goal[2] += self.extents[2]/2.0 # add height of half bbox
        object_oriented_goal[2] += 0.08 # first make the gripper go slightly above the object

        timeStep = 0 #count the total number of timesteps
        episodeObs.append(lastObs)

        while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
            env.render()
            action = np.zeros(4,)
            object_oriented_goal = object_rel_pos.copy()
            object_oriented_goal[2] += 0.08

            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]*10

            action[len(action)-1] = 0.05 #open

            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            cur_reward.append(reward)

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]

        while np.linalg.norm(object_rel_pos) >= 0.02 and timeStep <= env._max_episode_steps:
            env.render()
            action = np.zeros(4,)
            for i in range(len(object_rel_pos)):
                action[i] = object_rel_pos[i]*10

            # action[len(action)-1] = -0.005
            action[len(action)-1] -= 0.005

            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            cur_reward.append(reward)

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]

        ## ... for properly grasping the cup before lifting ... ##
        for i in range(12):
            env.render()
            action = np.zeros(4,)
            action[len(action)-1] = -0.005
            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            cur_reward.append(reward)

        # now that I have grasped the object I am just going to lift it
        lift_pos = objectPos.copy()
        lift_pos[2] += 0.22
        # print(f'lift_pos: {lift_pos}')
        # print(f'objectPos: {objectPos}')
        # print(lift_pos)
        while np.linalg.norm(lift_pos - objectPos) >= 0.05 and timeStep <= env._max_episode_steps:
            env.render()
            action = np.zeros(4,)
            for j in range(len(lift_pos - objectPos)):
                action[j] = (lift_pos - objectPos)[j]*10
            action[len(action)-1] = -0.005
            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            cur_reward.append(reward)
            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]

        goal[2] += 0.21
        while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps:
            env.render()
            action = np.zeros(4,)
            for i in range(len(goal - objectPos)):
                action[i] = (goal - objectPos)[i]*10

            action[len(action)-1] = -0.005

            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            cur_reward.append(reward)

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]

        while True: #limit the number of timesteps in the episode to a fixed duration
            env.render()
            action = np.zeros(4,)
            action[len(action)-1] = 0.05 # keep the gripper closed

            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            cur_reward.append(reward)

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
            if timeStep >= env._max_episode_steps: break

        success = 0
        cur_reward = np.sum(cur_reward)
        if np.sum(cur_reward) > -1 * env._max_episode_steps:
            success = 1
        return success, cur_reward

        #return episodeAcs, episodeObs, episodeInfo