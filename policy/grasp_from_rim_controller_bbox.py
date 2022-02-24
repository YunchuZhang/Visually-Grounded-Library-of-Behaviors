from policy.policy import Policy

import os
import trimesh
import numpy as np
import pcp_utils
import transformations
from pcp_utils.utils import Config
from pcp_utils.load_ddpg import load_policy

class GraspFromRimControllerBBox(Policy):
    class Config(Config):
        policy_name = "grasp_from_rim_controller_bbox"
        policy_model_path = ""
        model_name = None

    def __init__(self, config:Config):
        self.config=config
        self.policy_name = config.policy_name
    
    @staticmethod
    def compute_geom_center_from_mujoco(env, object_name):
        # first get the idx of the object_geoms
        body_id = env.env.sim.model.body_name2id('object0')
        geom_idxs = list()
        for i, assoc_body in enumerate(env.env.sim.model.geom_bodyid):
            if assoc_body == body_id:
                geom_idxs.append(i)

        # now get the xpos and xmat of these idxs
        geom_xpos = env.env.sim.data.geom_xpos[geom_idxs]
        geom_xmat = env.env.sim.data.geom_xmat[geom_idxs]

        # now get the vertices of belonging to each geom
        # first I get the idxs associated with the name of the mesh
        object_mesh_idxs = list()
        for m in env.env.sim.model.mesh_names:
            if object_name in m:
                object_mesh_idxs.append(env.env.sim.model.mesh_name2id(m))
        
        # now get the vertex location address
        addr_in_vert_array = list()
        for idx in object_mesh_idxs:
            addr_in_vert_array.append(env.env.sim.model.mesh_vertadr[idx])

        # finally get the vertices for each geom
        geom_mesh_vertices = list()
        for i in range(len(addr_in_vert_array)-1):
            low_idx = addr_in_vert_array[i]
            high_idx = addr_in_vert_array[i+1]
            verts = env.env.sim.model.mesh_vert[low_idx:high_idx]
            geom_mesh_vertices.append(verts)
        
        geom_mesh_vertices.append(env.env.sim.model.mesh_vert[addr_in_vert_array[-1]:])

        # now transform each of these vertices in world_coordinate frame
        verts_in_world = list()
        for i, vert in enumerate(geom_mesh_vertices):
            trans = geom_xpos[i]
            rot_mat = geom_xmat[i]
            transform_mat = np.eye(4)
            transform_mat[:3,:3] = rot_mat.reshape(3,3)
            transform_mat[:3,3] = trans
            h_vert = np.c_[vert, np.ones(len(vert))]
            rot_vert = np.dot(transform_mat, h_vert.T).T[:,:3]
            verts_in_world.append(rot_vert)
        print(f'length in world {len(verts_in_world)}')
        verts_in_world = np.concatenate(verts_in_world)
        return verts_in_world

    @staticmethod
    def get_bbox_properties(env, obj_info):

        obj_xml = obj_info.obj_xml_file
        obj_xpos = env.env.sim.data.get_body_xpos('object0')
        obj_xmat = env.env.sim.data.get_body_xmat('object0')
        obj_xquat = env.env.sim.data.get_body_xquat('object0')
        scale = obj_info.scale
        coords, combined_mesh = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(
            obj_info.obj_xml_file, obj_xpos, obj_xmat, scale, return_combined_mesh=True)

        # now get the properties
        bounds, center, extents = pcp_utils.np_vis.get_bbox_attribs(coords)

        # check here if bounding box is fine for the object
        # I will draw the box using my computed values
        transform = np.eye(4)
        transform[:3, 3] = center
        bounding_box_outline = trimesh.primitives.Box(
            transform=transform, extents=extents
        )
        bounding_box_outline.visual.face_colors = [0, 0, 255, 100]

        # just to make sure that the bounding box is tight here
        assert np.allclose(bounding_box_outline.bounds, combined_mesh.bounds)

        # # plot the box and mesh
        # scene = trimesh.Scene()
        # scene.add_geometry([combined_mesh, bounding_box_outline])
        # scene.show()
        ### ... checking of bounding box for every step ends ... ###
        bbox_xpos = center.copy()

        return bounds, center, extents, obj_xquat, bbox_xpos

    @staticmethod
    def compute_top_of_rim(center, extents):
        # center : position of the object
        # extents : size of the bounding box
        # compute the position of the left side rim while looking at screen
        rim_pos = center.copy()
        rim_pos[2] += extents[2] / 2.0 + 0.05
        rim_pos[1] -= extents[1] / 2.0
        return rim_pos
    
    @staticmethod
    def compute_rim_point(center, extents):
        # center : position of the object
        # extents : size of the bounding box
        rim_pos = center.copy()
        rim_pos[1] -= extents[1] / 2.0
        # rim_pos[2] += 0.02
        rim_pos[2] += extents[2] / 2.0
        rim_pos[2] -= 0.023
        return rim_pos
    
    @staticmethod
    def compute_rotation(env):
        # computes a quaternion to orient gripper and object.
        obj_xmat = env.env.sim.data.get_body_xmat('object0')
        gripper_xmat = env.env.sim.data.get_site_xmat('robot0:grip')
        # now I want the x-axis of gripper to be equal to z-axis of object
        obj_zaxis = obj_xmat[:3, 2]
        gripper_xmat[:3, 0] = -obj_zaxis
        h_gripper = np.eye(4)
        h_gripper[:3, :3] = gripper_xmat
        # convert to quaternion
        gripper_xquat = transformations.quaternion_from_matrix(h_gripper)
        return gripper_xquat
    
    @staticmethod
    def vis_bbox(env, center, xquat):
        env.env.move_indicator(center, xquat)
        # env.env.sim.forward()

    def run_forwards(self, env, num_rollouts, path_length=None, obj=None, render=False):
        self.render = render
        self.env = env
        obj_info = obj
        acc_reward = 0
        acc_success = 0
        
        for iter_id in range(num_rollouts):
            obs = env.reset()
            #ep_actions, ep_observations, ep_infos 
            success, cur_reward = self.goToGoal(env, obs, mesh_obj_info=obj_info)
            print("ITERATION NUMBER ", iter_id, 'success', success)

            acc_reward += cur_reward
            acc_success += success
        success_rate = acc_success/num_rollouts
        avg_reward = acc_reward/num_rollouts
        return {'avg_reward':avg_reward, 'success_rate':success_rate}


    def goToGoal(self, env, lastObs, mesh_obj_info=None):
        goal = lastObs['desired_goal']

        # computing the object position and the position to go using bounding_box
        bounds, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
        self.vis_bbox(env, bbox_xpos, obj_xquat)
        # while True:
        #     # env.render()

        rim_pos = self.compute_top_of_rim(center, extents)
        gripper_position = env.env.sim.data.get_site_xpos('robot0:grip')
        # move the gripper to the top of the rim
        bboxobject_rel_pos = rim_pos - gripper_position

        episodeAcs = []
        episodeObs = []
        episodeInfo = []
    
        object_oriented_goal = bboxobject_rel_pos.copy()
    
        timeStep = 0 #count the total number of timesteps
        episodeObs.append(lastObs)
        cur_reward = []
    
        # go on top of the rim point upto some tolerance. top is computed using compute_top_of_rim
        while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
            self.env_render()
            action = np.zeros(8,)
            object_oriented_goal = bboxobject_rel_pos.copy()
    
            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]*6
    
            action[len(action)-1] = 0.05 #open
            action[3:7] = [1., 0., 1., 0.]

            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            timeStep += 1
    
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)

            # compute the objectPos and object_rel_pos using bbox
            bounds, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
            rim_pos = self.compute_top_of_rim(center, extents)
            gripper_position = env.env.sim.data.get_site_xpos('robot0:grip')
            # move the gripper to the top of the rim
            bboxobject_rel_pos = rim_pos - gripper_position
            # now before executing the action move the box, step calls forward
            # which would actually move the box
            self.vis_bbox(env, bbox_xpos, obj_xquat)
        
        # go towards the rim point
        while np.linalg.norm(bboxobject_rel_pos) >= 0.003 and timeStep <= env._max_episode_steps:
            self.env_render()
            action = np.zeros(8,)
            for i in range(len(bboxobject_rel_pos)):
                action[i] = bboxobject_rel_pos[i]*6
    
            # start closing the gripper while going down!
            action[len(action)-1] = -0.025
            action[3:7] = [1., 0., 1., 0.]
    
            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            timeStep += 1
    
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
    
            # computing the same thing using bounding box
            bounds, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
            rim_pos = self.compute_rim_point(center, extents)
            gripper_position = env.env.sim.data.get_site_xpos('robot0:grip')
            bboxobject_rel_pos = rim_pos - gripper_position
            self.vis_bbox(env, bbox_xpos, obj_xquat)
            # while True:
                # # env.render()

        # for properly grasping the cup before liftting
        for i in range(20):
            self.env_render()
            action = np.zeros(8,)
            action[len(action)-1] = -0.05
            action[3:7] = [1., 0., 1., 0.]
            obsDataNew, reward, done, info = env.step(action) # actually simulating for some timesteps
            timeStep += 1
            cur_reward.append(reward)
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            # keep on updating the object xpos
            bounds, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
            bboxobjectPos = center
            self.vis_bbox(env, bbox_xpos, obj_xquat)
    
        # now that the object is grasped move towards the goal
        while np.linalg.norm(goal - bboxobjectPos) >= 0.01 and timeStep <= env._max_episode_steps:
            # while True:
                # # env.render()
            self.env_render()
            action = np.zeros(8,)
            goal_objectPos_vec = goal - bboxobjectPos
            for i in range(len(goal_objectPos_vec)):
                action[i] = (goal_objectPos_vec)[i]*6
    
            action[len(action)-1] = -0.005
            # now the object is grasped, make the gripper compliant by rotation
            # rot_quat = self.compute_rotation(env)
            # action[3:7] = rot_quat
            action[3:7] = [1., 0., 1., 0.]

            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            timeStep += 1
    
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
    

            # again compute the same thing using the bounding box
            bounds, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
            bboxobjectPos = center
            self.vis_bbox(env, bbox_xpos, obj_xquat)
            # # now here to avoid slippage need to keep the gripper in contact with the object
            # rim_pos = self.compute_rim_point(center, extents)
            # grip_pos = env.env.sim.data.get_site_xpos('robot0:grip')  
            # res = rim_pos - grip_pos
            # bboxobjectPos += res
    
        # limit the number of timesteps in the episode to a fixed duration
        while True:
            self.env_render()
            action = np.zeros(8,)
            action[-1] = -0.005
            # rot_quat = self.compute_rotation(env)
            # action[3:7] = rot_quat
            action[3:7] = [1., 0., 1., 0.]
    
            obsDataNew, reward, done, info = env.step(action)
            cur_reward.append(reward)
            timeStep += 1
    
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)

            bounds, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
            self.vis_bbox(env, bbox_xpos, obj_xquat)
    
            if timeStep >= env._max_episode_steps: break #env._max_episode_steps: 70

        success = 0
        curr_reward = np.sum(cur_reward)
        if np.sum(curr_reward) > -1 * env._max_episode_steps and cur_reward[-1] == 0:
            success = 1

        return success, curr_reward
        #return episodeAcs, episodeObs, episodeInfo
