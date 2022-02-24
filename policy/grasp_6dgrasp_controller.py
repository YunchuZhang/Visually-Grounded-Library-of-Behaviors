from policy.policy import Policy

import os
import trimesh
import numpy as np
import pcp_utils
import transformations
import pcp_utils
from pcp_utils.utils import Config
from pcp_utils.load_ddpg import load_policy

from pcp_utils import grasp_sampler

from scipy.spatial.transform import Rotation as R

class Grasp6DGraspController(Policy):
    class Config(Config):
        policy_name = "grasp_6dgrasp_controller"
        policy_model_path = ""
        model_name = None
        max_path_length = 150

        # keypoint = center + alpha * haf_bounding_box + beta
        # angle = elevation * gamma * elevation*90
        # [alpha1, alpha2, alpha3, beta1, beta2, beta3, gamma]
        # alpha1, alpha2, alph3 = [-1, 1],   beta1, beta2, beta3 = [-0.4, 0.4]

        params = []
        elevation_max = 90
        elevation_min = 0
        roll_max = 90
        roll_min = 0


        # camera params
        camera_img_height = 128
        camera_img_width = 128
        camera_radius = 0.3
        camera_fov_y = 45
        camera_pitch = [40, 41, 2]
        camera_yaw = [0, 350, 36]
        camera_yaw_list = None #[0, 60, 300]
        camera_save_image = False
        camera_recon_scene = True
        camera_lookat_pos = [1.3, 0.75, 0.4]


        data_collection_mode = False
        data_collection_from_trained_model = False
        save_data_name = ""
        fix_view = False

    def __init__(self, config:Config, detector=None):

        self.config=config
        self.policy_name = config.policy_name
        self.max_path_length = config.max_path_length

        self.elevation_max = config.elevation_max
        self.elevation_min = config.elevation_min

        self.roll_max = config.roll_max
        self.roll_min = config.roll_min


        self.save_video = True
        if self.save_video:
            self.imgs = []

        grasp_sampler_config = grasp_sampler.GraspSampler.Config()
        grasp_sampler_config.data_collection_mode = self.config.data_collection_mode
        grasp_sampler_config.data_collection_from_trained_model  = self.config.data_collection_from_trained_model
        grasp_sampler_config.camera_yaw_list = self.config.camera_yaw_list
        grasp_sampler_config.fix_view = self.config.fix_view
        grasp_sampler_config.save_data_name = self.config.save_data_name

        self.grasp_sampler = grasp_sampler.GraspSampler(grasp_sampler_config)

        if self.config.data_collection_mode:
            npoints =1024
            self.npoints = npoints
            import collections
            data_config = collections.OrderedDict()
            data_config['datapoints_per_file'] = 10
            data_config['fields'] = collections.OrderedDict()
            data_config['fields']["pc"] = dict()
            data_config['fields']["pc"]['dtype'] = "float32"
            data_config['fields']["pc"]['height'] = npoints
            data_config['fields']["pc"]['width'] = 3

            data_config['fields']["grasp_rt"] = dict()
            data_config['fields']["grasp_rt"]['dtype'] = "float32"
            data_config['fields']["grasp_rt"]['height'] = 4
            data_config['fields']["grasp_rt"]['width'] = 4


            data_config['fields']["label"] = dict()
            data_config['fields']["label"]['dtype'] = "int32"

            data_config['fields']["pc_pose"] = dict()
            data_config['fields']["pc_pose"]['dtype'] = "float32"
            data_config['fields']["pc_pose"]['height'] = 4
            data_config['fields']["pc_pose"]['width'] = 4

            from autolab_core import TensorDataset

            self.tensordata = TensorDataset(self.config.save_data_name, data_config)


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
            obj_info.obj_xml_file, obj_xpos, obj_xmat, scale=scale, euler=obj_info.euler, return_combined_mesh=True)

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
    def get_bbox_properties_norot(env, obj_info):

        obj_xml = obj_info.obj_xml_file
        obj_xpos = env.env.sim.data.get_body_xpos('object0')
        obj_xmat = np.eye(3)
        obj_xquat = np.zeros(4)
        obj_xquat[0] = 1

        #obj_xmat = env.env.sim.data.get_body_xmat('object0')
        #obj_xquat = env.env.sim.data.get_body_xquat('object0')
        scale = obj_info.scale
        coords, combined_mesh = pcp_utils.np_vis.compute_bonuding_box_from_obj_xml(
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
    def compute_top_of_rim(center, extents, obj_quat=np.array([1, 0, 0, 0])):
        # center : position of the object
        # extents : size of the bounding box
        # compute the position of the left side rim while looking at screen
        d_rim_pos = np.zeros(4) #center.copy()
        d_rim_pos[3] = 1
        d_rim_pos[2] = extents[2] / 2.0 + 0.08
        d_rim_pos[1] = -extents[1] / 2.0


        ori_T_obj = transformations.quaternion_matrix(obj_quat)

        rotated_d_rim_pos = np.dot(ori_T_obj, d_rim_pos)[:3]
        rim_pos = center.copy() + rotated_d_rim_pos

        return rim_pos


    @staticmethod
    def compute_pts_away_from_grasp_point(grasp_point, gripper_quat, dist=0):
        # center : position of the object
        # extents : size of the bounding box
        # compute the position of the left side rim while looking at screen
        gripper_xmat = transformations.quaternion_matrix(gripper_quat)
        new_grasp_point = grasp_point + (-1) * gripper_xmat[:3, 0] * dist

        return new_grasp_point
    
    
    #    @staticmethod
    #    def compute_rim_point(center, extents, obj_quat=np.array([1, 0, 0, 0])):
    #        # center : position of the object
    #        # extents : size of the bounding box
    #        d_rim_pos =  np.zeros(4)
    #        d_rim_pos[3] = 1
    #        d_rim_pos[2] = extents[2] / 2.0 * self.alpha[2] + self.beta[2]
    #        d_rim_pos[1] = extents[1] / 2.0 * self.alpha[1] + self.beta[1]
    #        d_rim_pos[0] = extents[0] / 2.0 * self.alpha[0] + self.beta[0]
    #
    #        import ipdb; ipdb.set_trace()
    #
    #
    #        #ori_T_obj = transformations.quaternion_matrix(obj_quat)
    #        #rotated_d_rim_pos = np.dot(ori_T_obj, d_rim_pos)[:3]
    #        object_rot = transformations.quaternion_matrix(obj_quat)
    #        rotated_d_rim_pos = np.dot(object_rot, d_rim_pos)[:3]
    #        rim_pos = center.copy() + rotated_d_rim_pos
    #
    #        return rim_pos
    
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

    def run_forwards(self, env, num_rollouts, path_length=None, obj=None, render=False, accept_threshold=0, cluster_name=""):
        self.render = render
        self.env = env
        obj_info = obj
        acc_reward = 0
        acc_success = 0

        max_num_failure = (1 - accept_threshold) * num_rollouts
        num_failure = 0
        
        for iter_id in range(num_rollouts):
            obs = env.reset()
            #ep_actions, ep_observations, ep_infos 
            self.reset_everything_on_table(env, mesh_obj_info=obj_info)
            success, cur_reward = self.goToGoal(env, obs, mesh_obj_info=obj_info)
            print("ITERATION NUMBER ", iter_id, 'success', success)

            if self.config.data_collection_mode:
                datapoint = dict()
                datapoint["label"] = success
                #grasp_pose = np.zeros((6), np.float32)
                #grasp_pose[2] = self.current_grasp.depth

                datapoint["pc"] = grasp_sampler._regularize_pc_point_count(self.cache_pc, self.npoints)
                datapoint["pc_pose"] = self.cache_pc_pose
                datapoint["grasp_rt"] = self.cache_grasp_rt
                self.tensordata.add(datapoint)

            if self.save_video:
                self.dump_video(f"tmp/6dgrasp/{obj_info.name}_{iter_id}_{success}.mp4")


            acc_reward += cur_reward
            acc_success += success
            if success < 0.1:
                num_failure += 1
            if num_failure > max_num_failure:
                break

        success_rate = acc_success/num_rollouts
        avg_reward = acc_reward/num_rollouts
        return {'avg_reward':avg_reward, 'success_rate':success_rate}

    def reset_everything_on_table(self, env, mesh_obj_info, max_run=100):
        _, center, _, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
        center_old = center
        for i in range(max_run):
            obsDataNew, reward, done, info = env.step(np.zeros(8))
            _, center, _, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
            if np.linalg.norm(center_old - center) < 0.000001:
                return
            center_old = center
        return

        #raise ValueError


    def unstable_grasp(self, finger_tip_states):
        return np.var(finger_tip_states) > 0.00001

    def gripper_goto_pos_orn(self, env, target_pos, target_quat=np.array([1, 0, 1, 0]), goal_current_thres= 0.005, speed=6, open=True, timeStep=0, mesh_obj_info=None, debug=False, max_time_limit=10000):

        gripper_position = env.env.sim.data.get_site_xpos('robot0:grip')
        gripper_xmat = env.env.sim.data.get_site_xmat('robot0:grip')

        gripper_quat = R.from_matrix(gripper_xmat).as_quat()[[3, 0, 1, 2]]


        rel_pos = target_pos - gripper_position

        cur_reward = []
        episodeAcs = []
        episodeObs = []
        episodeInfo = []
        finger_tip_states = []


        grasp_harder = False
        while np.linalg.norm(rel_pos) >= goal_current_thres and timeStep <= max_time_limit and timeStep <= self.max_path_length:
            self.env_render()
            action = np.zeros(8,)
            finger_tip_state = env.get_finger_tip_state()
            finger_tip_states.append(finger_tip_state)

            gripper_xmat = env.env.sim.data.get_site_xmat('robot0:grip')
            gripper_quat = R.from_matrix(gripper_xmat).as_quat()[[3, 0, 1, 2]]
            delta_quat = self._get_intermediate_delta_quats(gripper_quat, target_quat, num_intermediates=4)
            action[3:7] = delta_quat[1]

            for i in range(len(rel_pos)):
                action[i] = rel_pos[i]*speed
    
            if open:
                action[len(action)-1] = 0.05 #open
            else:
                action[len(action)-1] = -0.05
                if not grasp_harder  and self.unstable_grasp(finger_tip_states):
                    action[len(action)-1] =  -0.05
                    grasp_harder=True
                elif grasp_harder:
                    action[len(action)-1] = -0.05


            obsDataNew, reward, done, info = env.step(action)

            cur_reward.append(reward)
            timeStep += 1
    
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)

            # compute the objectPos and object_rel_pos using bbox
            if debug:
                print("action", action, np.linalg.norm(rel_pos))
            gripper_position = env.env.sim.data.get_site_xpos('robot0:grip')

            # move the gripper to the top of the rim
            rel_pos = target_pos - gripper_position
            # now before executing the action move the box, step calls forward
            # which would actually move the box
            if mesh_obj_info is None:
               bounds, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
               self.vis_bbox(env, bbox_xpos, obj_xquat)
        


#        for step in range(10):
#            self.env_render()
#            timeStep += 1
#            action = np.zeros(8,)
#            action[3:7] = delta_quat[step]
#            obsDataNew, reward, done, info = env.step(action)
#            cur_reward.append(reward)


 
        return cur_reward, timeStep

    def close_gripper(self, env, iter=50, timeStep=0, gripper_pos= -0.03, mesh_obj_info=None):
        cur_reward = []
        episodeAcs = []
        episodeObs = []
        episodeInfo = []
        for i in range(iter):
            self.env_render()
            action = np.zeros(8,)
            action[len(action)-1] = gripper_pos
            gripper_xmat = env.env.sim.data.get_site_xmat('robot0:grip')
            gripper_quat = R.from_matrix(gripper_xmat).as_quat()[[3, 0, 1, 2]]

            action[3:7] = gripper_quat #todo, replace with what the gipper is at
            obsDataNew, reward, done, info = env.step(action) # actually simulating for some timesteps
            timeStep += 1
            cur_reward.append(reward)
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            # keep on updating the object xpos
            if mesh_obj_info is None:
                bounds, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
                self.vis_bbox(env, bbox_xpos, obj_xquat)
        return cur_reward, timeStep
    
    def _get_intermediate_delta_quats(self, init_quat, des_quat, num_intermediates=10):
        """
        init_quat: initial quaternion
        des_quat: desired quaternion
        TODO: ideally since the quaternions are like complex numbers in high
        dimensions, the multiplication should give addition, but for some reason
        the simple addtion is fine here, investigate why this is the case
        """
        # class should be pyquaternion.Quaternion
        from pyquaternion import Quaternion
        quat_to_array = lambda q: np.asarray([q.w, q.x, q.y, q.z])
        if isinstance(init_quat, np.ndarray):
            init_quat = Quaternion(init_quat)
        if isinstance(des_quat, np.ndarray):
            des_quat = Quaternion(des_quat)
 
        assert isinstance(init_quat, Quaternion)
        assert isinstance(des_quat, Quaternion)

        # generator for the intermediates
        intermediate_quats = list()
        for q in Quaternion.intermediates(init_quat, des_quat,
            num_intermediates, include_endpoints=True):
            qu = quat_to_array(q)
            intermediate_quats.append(qu)
        
        # go through the intermediates to generate the delta quats
        delta_quats = list()
        prev_quat = quat_to_array(init_quat).copy()
        for q in intermediate_quats:
            delta_quat = q - prev_quat
            delta_quats.append(delta_quat)
            prev_quat = q.copy()
        
        # now delta quats when combined with initial quat should sum to 1
        add_val = quat_to_array(init_quat) + np.sum(delta_quats, axis=0)
        assert np.allclose(add_val, quat_to_array(des_quat))
        return delta_quats


    def compute_grasping_direction(self):

        #gripper_xquat = transformations.quaternion_from_matrix(h_gripper)
        # at 1, 0, gripper is pointing toward -y, gripper right is pointing -x, 
        ori_gripper_xmat = np.array([[-1,0,0],
                                     [0,1,0],
                                     [0,0,-1]])


        d_rim_pos = self.get_grasp_point_to_center()

        d_xy = d_rim_pos[:2]
        if d_xy[0] == 0 and d_xy[1] == 0:
            d_xy[1] = 0.00000001
        d_xy = d_xy / np.linalg.norm(d_xy)
        cos_theta = d_xy[0]
        sin_theta = d_xy[1]



        elevation = self.get_grasp_point_elevation()
        roll = self.get_grasp_point_roll()


        #ori_gripper_quat =  transformations.quaternion_from_matrix(ori_gripper_xmat)
        # add rotation on the xy plane
        xy_rot_xmat = np.array([[cos_theta,-sin_theta,0],
                                     [sin_theta,cos_theta,0],
                                     [0,0, 1]])



        # add elevation: elevation higher means camera looking more downwards
        roll_xmat = np.array([[1, 0, 0],
                             [0, np.cos(np.deg2rad(-roll)), -np.sin(np.deg2rad(-roll))],
                             [0, np.sin(np.deg2rad(-roll)), np.cos(np.deg2rad(-roll))]
                            ])

        ele_xmat = np.array([[np.cos(np.deg2rad(-elevation)), 0, np.sin(np.deg2rad(-elevation))],
                             [0, 1, 0],
                             [-np.sin(np.deg2rad(-elevation)),0, np.cos(np.deg2rad(-elevation))]
                            ])


        final_rot  =  np.matmul(np.matmul(xy_rot_xmat, np.matmul(ele_xmat, ori_gripper_xmat)), roll_xmat)

        # making the "thumb" of the gripper to point to y+
        # if not: rotate gripper with 180 degree
        if final_rot[1,1] < 0:
            rot = 180
            xmat = np.array([[1, 0, 0],
                             [0, np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot))],
                             [0, np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot))]
                            ])
            final_rot = np.matmul(final_rot, xmat)


        final_rot_4x4 = np.eye(4)
        final_rot_4x4[:3, :3] = final_rot


        gripper_xquat = transformations.quaternion_from_matrix(final_rot_4x4)

        return gripper_xquat
        
    def get_grasp_point_roll(self):
        return self.roll_min + self.eta * (self.roll_max - self.roll_min)

    def get_grasp_point_elevation(self):
        return self.elevation_min + self.gamma * (self.elevation_max - self.elevation_min)

    def get_grasp_point_to_center(self):

        d_rim_pos = np.zeros(3)
        d_rim_pos[2] = self.extents[2] / 2.0 * self.alpha[2] + self.beta[2]
        d_rim_pos[1] = self.extents[1] / 2.0 * self.alpha[1] + self.beta[1]
        d_rim_pos[0] = self.extents[0] / 2.0 * self.alpha[0] + self.beta[0]
        return d_rim_pos

    def get_grasp_point(self, center):
        grasp_point = center + self.get_grasp_point_to_center()
        return grasp_point



    def split_grasps(self, grasp):
        pos = grasp[:3, 3]
        orn = grasp[:3, :3]
        return pos, orn


    def convert_grasp_orn_to_fetch(self, orn):
        gripper_right = orn[:3,0]
        gripper_down = orn[:3,1]
        gripper_point = orn[:3,2]


        final_rot_4x4 = np.eye(4)
        final_rot_4x4[:3, 0] = gripper_point
        final_rot_4x4[:3, 1] = gripper_right
        final_rot_4x4[:3, 2] = gripper_down


        if final_rot_4x4[1,1] < 0:
            rot = 180
            xmat = np.array([[1, 0, 0, 0],
                             [0, np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                             [0, np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                             [0, 0, 0, 1]])
            final_rot_4x4 = np.matmul(final_rot_4x4, xmat)

        r = R.from_matrix(final_rot_4x4[:3, :3])
        gripper_xquat = r.as_quat()
        # change from xyzw to wxyz
        wxyz = np.zeros(4, np.float32)
        wxyz[1:] = gripper_xquat[:3]
        wxyz[0] = gripper_xquat[3]



        #sgripper_xquat = transformations.quaternion_from_matrix(final_rot_4x4)
        return wxyz

    def get_grasp_from_grasp_sampler(self, env, mesh_obj_info):
        data = self.grasp_sampler.sample_grasp_from_camera(env, mesh_obj_info, canonicalize_pcs=False)


        #data = np.load("vae_generated_grasps/grasps_fn_159e56c18906830278d8f8c02c47cde0_15.npy", allow_pickle=True).item()

        _, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
        
        # pick one
        top_k = min(10, len(data['generated_scores']))

        if top_k == 0: # if no grasp is detect : just grasp from center
            grasp_pos = center
            gripper_target_quat = np.array([1,0,1,0])
            a_grasp = np.eye(4)
            a_grasp[2,3] = 0.02
        else:
            grasp_ids = np.argsort(data['generated_scores'])[-top_k:][::-1]
            if self.config.data_collection_mode and self.config.data_collection_from_trained_model:
                picked_id = np.random.randint(top_k)
                grasp_id = grasp_ids[picked_id]
            elif self.config.data_collection_mode:
                grasp_id = np.random.randint(len(data['generated_scores']))
            else:
                picked_id = 0
                grasp_id = grasp_ids[picked_id]

            #   if self.config.data_collection_mode and not self.config.data_collection_from_trained_model:
            #       grasp_id = np.random.randint(len(data['generated_scores']))
            #   else:
            #       grasp_ids = np.argsort(data['generated_scores'])[-top_k:][::-1]
            #       picked_id = 0#np.random.randint(top_k)
            #       grasp_id = grasp_ids[picked_id]
            a_grasp = data['generated_grasps'][grasp_id]
            #a_grasp_X = data['generated_grasps_X'][grasp_id]

            grasp_pos, grasp_orn = self.split_grasps(a_grasp)
            grasp_pos, grasp_orn = self.grasp_sampler.convert_grasp_to_mujoco(grasp_pos, grasp_orn)

            gripper_target_quat = self.convert_grasp_orn_to_fetch(grasp_orn) #self.compute_grasping_direction()

        if self.config.data_collection_mode:
            self.cache_pc = data['obj_pcd_X'] # reduce number of points
            self.cache_grasp_rt = a_grasp
            self.cache_pc_pose = data["camR_T_camX"]

        return grasp_pos, gripper_target_quat 

    def goToGoal(self, env, lastObs, mesh_obj_info=None):

        goal = lastObs['desired_goal']



        grasp_pos, gripper_target_quat  = self.get_grasp_from_grasp_sampler(env, mesh_obj_info)
       
        grasp_point = grasp_pos #self.get_grasp_point(center)
        _, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)


        self.extents = extents
        # transform it from adam to 

        self.env_render()
        self.vis_bbox(env, bbox_xpos, obj_xquat)
        # while True:
        #     # env.render()



        gripper_position = env.env.sim.data.get_site_xpos('robot0:grip')
        # move the gripper to the top of the rim
    
    
        timeStep = 0 #count the total number of timesteps
        cur_reward = []
        episodeAcs = []
        episodeObs = []
        episodeInfo = []


        target_pos = self.compute_pts_away_from_grasp_point(grasp_point, gripper_target_quat, dist=0) # the gripper is -0.105 away from the grasp point

        ## go to the top of the rim
        #rim_top_pos = self.compute_top_of_rim(center, extents, gripper_target_quat)     
        #gripper_quat = np.array([1, 0, 1, 0])
        new_gripper_quat  = gripper_target_quat#transformations.quaternion_multiply(obj_xquat, gripper_quat)

        rewards, timeStep = self.gripper_goto_pos_orn(env, target_pos, new_gripper_quat, goal_current_thres= 0.002, speed=6, open=True, timeStep=timeStep, mesh_obj_info=mesh_obj_info, max_time_limit=self.max_path_length/3)
        cur_reward += rewards

        # go toward rim pos
        _, center, _, _, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
        #grasp_point = self.get_grasp_point(center)
        rim_pos = self.compute_pts_away_from_grasp_point(grasp_point, gripper_target_quat, dist=-0.105)



        rewards, timeStep = self.gripper_goto_pos_orn(env, rim_pos, new_gripper_quat, goal_current_thres= 0.002, speed=6, open=True, timeStep=timeStep, mesh_obj_info=mesh_obj_info, max_time_limit=self.max_path_length*2/3)
        cur_reward += rewards

        #        bounds, center_new, _, _, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
        #        if np.linalg.norm(center - center_new) >= 0.02: # some objects are still dropping
        #           rim_pos =  self.get_grasp_point(center_new)
        #           #rim_pos = self.compute_rim_point(center_new, extents)
        #           rewards, timeStep = self.gripper_goto_pos_orn(env, rim_pos, new_gripper_quat, goal_current_thres= 0.002, speed=6, open=True, timeStep=timeStep, mesh_obj_info=mesh_obj_info)
        #           cur_reward += rewards     

        # close gripper


        rewards, timeStep = self.close_gripper(env, iter=20, gripper_pos= -0.01, timeStep=timeStep, mesh_obj_info=mesh_obj_info)
        cur_reward += rewards
        # move to target location
        goal_pos_for_gripper = goal - bbox_xpos + gripper_position
        run_one = True
        while  np.linalg.norm(goal - bbox_xpos) >= 0.01 and timeStep <= self.max_path_length:
            #print(np.linalg.norm(goal - bbox_xpos), goal_pos_for_gripper, gripper_position)
            if run_one: # first time, go a rroughly toward the goal
                thres = 0.01
            else:
                thres = 0.002
            rewards, timeStep = self.gripper_goto_pos_orn(env, goal_pos_for_gripper, new_gripper_quat, goal_current_thres= thres, speed=6, open=False, timeStep=timeStep, mesh_obj_info=mesh_obj_info)
            run_one=False
            cur_reward += rewards
            if cur_reward[-1] > 0:
                break
            # retriev again the poses
            bounds, center, _, _, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
            gripper_position = env.env.sim.data.get_site_xpos('robot0:grip')
            # recalculate
            goal_pos_for_gripper = goal - bbox_xpos + gripper_position
            if timeStep >= self.max_path_length: break #env._max_episode_steps: 70



        while True:
            rewards, timeStep = self.close_gripper(env, iter=1, timeStep=timeStep, gripper_pos=-0.005, mesh_obj_info=mesh_obj_info)
            cur_reward += rewards
            if timeStep >= self.max_path_length: break #env._max_episode_steps: 70


        success = 0
        curr_reward = np.sum(cur_reward)
        if np.sum(curr_reward) > -1 * self.max_path_length and cur_reward[-1] == 0:
            success = 1

        return success, curr_reward
        #return episodeAcs, episodeObs, episodeInfo