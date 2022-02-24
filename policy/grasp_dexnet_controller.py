from policy.policy import Policy
import sys
import os
import json
import time
import trimesh
import numpy as np
import copy
import pcp_utils
import transformations

from autolab_core import YamlConfig, Logger

import pcp_utils
from pcp_utils.utils import Config
from pcp_utils.load_ddpg import load_policy

from pcp_utils import grasp_sampler

from scipy.spatial.transform import Rotation as R


gqcnn_dir = "/home/htung/2020/gqcnn/"
sys.path.append(gqcnn_dir)
#from gqcnn.utils import GripperMode
from perception import (BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage)
from gqcnn.grasping import (CrossEntropyRobustGraspingPolicy, RgbdImageState)
from autolab_core import YamlConfig, Logger

#logger = Logger.get_logger("examples/policy.py")

#, CrossEntropyRobustGraspingPolicy,
#                            RgbdImageState, FullyConvolutionalGraspingPolicyParallelJaw)

class GraspDexnetController(Policy):
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

        camera_img_height = 128 * 3
        camera_img_width = 128 * 3
        camera_radius = 0.3
        camera_fov_y = 45
        table_top = [1.3, 0.75, 0.4]

        data_collection_mode = False
        save_data_name = ""

    def __init__(self, config:Config, detector=None):

        self.config=config
        self.policy_name = config.policy_name
        self.max_path_length = config.max_path_length

        self.elevation_max = config.elevation_max
        self.elevation_min = config.elevation_min

        self.roll_max = config.roll_max
        self.roll_min = config.roll_min



        self.save_video = False #True #False
        if self.save_video:
            self.imgs = []



        #grasp_sampler_config = grasp_sampler.GraspSampler.Config()
        #self.grasp_sampler = grasp_sampler.GraspSampler(grasp_sampler_config)


        model_name = "gqcnn_example_pj_fish" #"GQCNN-4.0-PJ"
        fully_conv = False
        gqcnn_dir = "/home/htung/2020/gqcnn/"
    
        # Set model if provided.
        model_dir = os.path.join(gqcnn_dir, "models")
        model_path = os.path.join(model_dir, model_name)
        config_filename= None
    
        # Get configs.
        model_config = json.load(open(os.path.join(model_path, "config.json"),
                                      "r"))

        try:
            gqcnn_config = model_config["gqcnn"]
            gripper_mode = gqcnn_config["gripper_mode"]
        except KeyError:
            gqcnn_config = model_config["gqcnn_config"]
            input_data_mode = gqcnn_config["input_data_mode"]
            if input_data_mode == "tf_image":
                gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
            elif input_data_mode == "tf_image_suction":
                gripper_mode = GripperMode.LEGACY_SUCTION
            elif input_data_mode == "suction":
                gripper_mode = GripperMode.SUCTION
            elif input_data_mode == "multi_suction":
                gripper_mode = GripperMode.MULTI_SUCTION
            elif input_data_mode == "parallel_jaw": # this is picked
                gripper_mode = GripperMode.PARALLEL_JAW
            else:
                raise ValueError(
                    "Input data mode {} not supported!".format(input_data_mode))
       
        config_filename = os.path.join(
                        gqcnn_dir,
                        "cfg/examples/gqcnn_pj.yaml")

       
        
        # Read config.
        config = YamlConfig(config_filename)
        inpaint_rescale_factor = config["inpaint_rescale_factor"]
        policy_config = config["policy"]

        # original_gripper_width = 0.05
        original_gripper_width = policy_config["gripper_width"]
        
        self.real_gripper_width = 0.112 #0.15 #0.112
        self.rescale_factor = original_gripper_width / self.real_gripper_width
        #self.config.camera_fov_y = self.config.camera_fov_y * self.rescale_factor
        

        #policy_config["gripper_width"] = 0.112
        # gripper distance to the grasp point
        # min_depth_offset = config['policy']["sampling"]["min_depth_offset"] = 0.015
        # max_depth_offset = config['policy']["sampling"]["max_depth_offset"] = 0.05



        
        # Make relative paths absolute.
        if "gqcnn_model" in policy_config["metric"]:
            policy_config["metric"]["gqcnn_model"] = model_path
            if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
                policy_config["metric"]["gqcnn_model"] = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    policy_config["metric"]["gqcnn_model"])

        # Set input sizes for fully-convolutional policy.
        if fully_conv:
            policy_config["metric"]["fully_conv_gqcnn_config"][
                "im_height"] = depth_im.shape[0]
            policy_config["metric"]["fully_conv_gqcnn_config"][
                "im_width"] = depth_im.shape[1]
    
        # Init policy.
        self.policy_config = policy_config
        self.policy_config["vis"]["vmin"] = 0
        self.policy_config["vis"]["vmax"] = 0.5

        if self.config.data_collection_mode:
            self.policy_config["num_iters"] = 0
            self.policy_config["random_sample"] = True

            self.gqcnn_image_size = 96
            import collections

            data_config = collections.OrderedDict()
            data_config['datapoints_per_file'] = 50
            data_config['fields'] = collections.OrderedDict()
            data_config['fields']["grasp_metrics"] = dict()
            data_config['fields']["grasp_metrics"]['dtype'] = "float32"

            data_config['fields']["grasps"] = dict()
            data_config['fields']["grasps"]['dtype'] = "float32"
            data_config['fields']["grasps"]['height'] = 6

            data_config['fields']["split"] = dict()
            data_config['fields']["split"]['dtype'] = "uint8"

            data_config['fields']["tf_depth_ims"] = dict()
            data_config['fields']["tf_depth_ims"]['dtype'] = "float32"
            data_config['fields']["tf_depth_ims"]['height'] = self.gqcnn_image_size
            data_config['fields']["tf_depth_ims"]['width'] = self.gqcnn_image_size
            data_config['fields']["tf_depth_ims"]['channels'] = 1


            from autolab_core import TensorDataset
            self.tensordata = TensorDataset(self.config.save_data_name, data_config)


        #import ipdb; ipdb.set_trace()
        policy_type = "cem"
        if "type" in policy_config:
            policy_type = policy_config["type"]
        if policy_type == "ranking":
            self.policy = RobustGraspingPolicy(policy_config)
        elif policy_type == "cem":
            self.policy = CrossEntropyRobustGraspingPolicy(policy_config)
        else:
            raise ValueError("Invalid policy type: {}".format(policy_type))




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
        """
        cluster_name: add as predix when saving the mp4
        """

        self.render = render
        self.env = env
        obj_info = obj
        acc_reward = 0
        acc_success = 0

        max_num_failure = (1 - accept_threshold) * num_rollouts
        num_failure = 0
        
        for iter_id in range(num_rollouts):
            obs = env.reset()
            self.iter_id = iter_id
            #ep_actions, ep_observations, ep_infos 
            self.reset_everything_on_table(env, mesh_obj_info=obj_info)
            success, cur_reward = self.goToGoal(env, obs, mesh_obj_info=obj_info)
            print("ITERATION NUMBER ", iter_id, 'success', success)

            if self.config.data_collection_mode:
                datapoint = dict()
                datapoint["grasp_metrics"] = success
                grasp_pose = np.zeros((6), np.float32)
                grasp_pose[2] = self.current_grasp.depth


                datapoint["grasps"] = grasp_pose
                datapoint["split"] = 0
                datapoint["tf_depth_ims"] = self.current_patch
                self.tensordata.add(datapoint)

            if self.save_video:
                self.dump_video(f"tmp/dexnet2/{obj_info.name}_{iter_id}_{success}.mp4")

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
        data = self.grasp_sampler.sample_grasp_from_camera(env, mesh_obj_info)
        #data = np.load("vae_generated_grasps/grasps_fn_159e56c18906830278d8f8c02c47cde0_15.npy", allow_pickle=True).item()

        _, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)
        
        # pick one
        top_k = min(5, len(data['generated_scores']))

        if top_k == 0: # if no grasp is detect : just grasp from center
            grasp_pos = center
            gripper_target_quat = np.array([1,0,1,0])
        else:
            grasp_ids = np.argsort(data['generated_scores'])[-top_k:][::-1]
            picked_id = np.random.randint(top_k)
            grasp_id = grasp_ids[picked_id]
            a_grasp = data['generated_grasps'][grasp_id]

            grasp_pos, grasp_orn = self.split_grasps(a_grasp)
            grasp_pos, grasp_orn = self.grasp_sampler.convert_grasp_to_mujoco(grasp_pos, grasp_orn)

            gripper_target_quat = self.convert_grasp_orn_to_fetch(grasp_orn) #self.compute_grasping_direction()



        return grasp_pos, gripper_target_quat 


    def sample_grasp(self, env, mesh_obj, vis=True):
        # Setup sensor.
        #camera_intr = CameraIntrinsics.load(camera_intr_filename)
    
        # Read images.
        # get depth image from camera

        inpaint_rescale_factor = 1.0
        segmask_filename = None

        _, center, extents, obj_xquat, bbox_xpos = self.get_bbox_properties(env, mesh_obj)
        camera_pos = copy.deepcopy(center)
        camera_pos[2] += 0.5 * extents[2] + 0.2
        

        env.set_camera(camera_pos, np.array([ 0.7071068, 0, 0, -0.7071068]), camera_name= f"ext_camera_0")
        rgb, depth = env.render_from_camera(int(self.config.camera_img_height) , int(self.config.camera_img_width), camera_name=f"ext_camera_0")
        # enforce zoom out

        from scipy.interpolate import interp2d

        center_x = self.config.camera_img_height/2 + 1
        center_y = self.config.camera_img_width/2 + 1
        img_height = self.config.camera_img_height
        img_width = self.config.camera_img_width
        xdense = np.linspace(0, img_height - 1, img_height)
        ydense = np.linspace(0, img_width - 1, img_width)
        xintr = (xdense - center_x)* (1.0/self.rescale_factor) + center_x
        yintr = (ydense - center_y)* (1.0/self.rescale_factor) + center_y
        xintr[xintr < 0] = 0
        xintr[xintr > (img_height - 1)] = img_height - 1
        yintr[yintr < 0] = 0
        yintr[yintr > (img_width - 1)] = img_width - 1

        fr = interp2d(xdense, ydense, rgb[:,:,0], kind="linear")
        rgb_r_new = fr(xintr, yintr)
        fg = interp2d(xdense, ydense, rgb[:,:,1], kind="linear")
        rgb_g_new = fg(xintr, yintr)
        fb = interp2d(xdense, ydense, rgb[:,:,2], kind="linear")
        rgb_b_new = fb(xintr, yintr)
        rgb_new = np.stack([rgb_r_new, rgb_g_new, rgb_b_new], axis=2)

        fd = interp2d(xdense, ydense, depth, kind="linear")
        depth_new = fd(xintr, yintr)

        #from skimage.transform import resize
        #rgb22, depth2 = env.render_from_camera(int(self.config.camera_img_height) , int(self.config.camera_img_width), camera_name=f"ext_camera_0")

        #import ipdb; ipdb.set_trace()

        # visualize the interpolation
        #import imageio
        #imageio.imwrite(f"tmp/rgb_{self.iter_id}.png", rgb)
        #imageio.imwrite(f"tmp/rgb2_{self.iter_id}.png", rgb_new)
        #imageio.imwrite(f"tmp/depth_{self.iter_id}.png", depth)
        #imageio.imwrite(f"tmp/depth2_{self.iter_id}.png", depth_new)
        #import ipdb; ipdb.set_trace()

        rgb = rgb_new
        depth = depth_new

        depth = depth * self.rescale_factor

        # rgb: 128 x 128 x 1
        # depth: 128 x 128 x 1
        scaled_camera_fov_y = self.config.camera_fov_y
        aspect = 1
        scaled_fovx = 2 * np.arctan(np.tan(np.deg2rad(scaled_camera_fov_y) * 0.5) * aspect)
        scaled_fovx = np.rad2deg(scaled_fovx)
        scaled_fovy = scaled_camera_fov_y

        cx = self.config.camera_img_width*0.5
        cy = self.config.camera_img_height*0.5
        scaled_fx = cx / np.tan(np.deg2rad(scaled_fovx / 2.)) * (self.rescale_factor)
        scaled_fy = cy / np.tan(np.deg2rad(scaled_fovy / 2.)) * (self.rescale_factor)



        camera_intr = CameraIntrinsics(frame='phoxi', fx=scaled_fx, fy=scaled_fy, cx=self.config.camera_img_width*0.5, cy=self.config.camera_img_height*0.5,
                                       height=self.config.camera_img_height, width=self.config.camera_img_width)

        depth_im = DepthImage(depth, frame=camera_intr.frame)
        color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                        3]).astype(np.uint8),
                              frame=camera_intr.frame)
    
        # Optionally read a segmask.

        valid_px_mask = depth_im.invalid_pixel_mask().inverse()
        segmask = valid_px_mask
    
        # Inpaint.
        depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
    
    
        # Create state.
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)
    

    
        # Query policy.
        policy_start = time.time()
        action = self.policy(state)
        print("Planning took %.3f sec" % (time.time() - policy_start))
        # numpy array with 2 values
        grasp_center = action.grasp.center._data[:,0] #(width, depth)
        grasp_depth = action.grasp.depth * (1/self.rescale_factor)
        grasp_angle = action.grasp.angle #np.pi*0.3

        if self.config.data_collection_mode:

            self.current_grasp = action.grasp
    
            depth_im = state.rgbd_im.depth
            scale = 1.0
            depth_im_scaled = depth_im.resize(scale)
            translation = scale * np.array([
                    depth_im.center[0] - grasp_center[1],
                    depth_im.center[1] - grasp_center[0]
                ])
            im_tf = depth_im_scaled
            im_tf = depth_im_scaled.transform(translation, grasp_angle)
            im_tf = im_tf.crop(self.gqcnn_image_size, self.gqcnn_image_size)
    
            # get the patch
            self.current_patch = im_tf.raw_data

        XYZ_origin, gripper_quat =  self.compute_grasp_pts_from_grasp_sample(grasp_center, grasp_depth, grasp_angle, env)


        return XYZ_origin[:,0], gripper_quat
    
        # Vis final grasp.
        if vis:
            from visualization import Visualizer2D as vis
            vis.figure(size=(10, 10))
            vis.imshow(rgbd_im.depth,
                       vmin=self.policy_config["vis"]["vmin"],
                       vmax=self.policy_config["vis"]["vmax"])
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
                action.grasp.depth, action.q_value))
            vis.show(f"tmp/grasp2_{mesh_obj.name}_{self.iter_id}.png")

            vis.figure(size=(10, 10))

            vis.imshow(im_tf,
                       vmin=self.policy_config["vis"]["vmin"],
                       vmax=self.policy_config["vis"]["vmax"])
            vis.show(f"tmp/cropd_{mesh_obj.name}_{self.iter_id}.png")


        import ipdb; ipdb.set_trace()

        return XYZ_origin[:,0], gripper_quat





        import ipdb; ipdb.set_trace()

    def compute_grasp_pts_from_grasp_sample(self, grasp_center, grasp_depth, grasp_angle, env):


        xy_pixel = np.ones((3, 1))
        xy_pixel[:2, 0] =  grasp_center


        # scale back the depth
        depth = grasp_depth

        #xy_pixel[2, 0] = depth
        
        fovy = self.config.camera_fov_y
        pix_T_cam = pcp_utils.cameras.get_intrinsics(fovy, self.config.camera_img_width, self.config.camera_img_height)

        pix_T_cam[0,0] = pix_T_cam[0,0] * self.rescale_factor
        pix_T_cam[1,1] = pix_T_cam[1,1] * self.rescale_factor
        cam_T_pix = np.linalg.inv(pix_T_cam)
        XYZ_camX = np.matmul(cam_T_pix, xy_pixel) * depth

        XYZ_camX[2,0] = XYZ_camX[2,0] #-0.09
        
        #XYZ_camX[2,0] = depth
        origin_T_camX = pcp_utils.cameras.gymenv_get_extrinsics(env, f"ext_camera_0")
        XYZ_camX_one = np.ones((4, 1))
        XYZ_camX_one[:3,:] = XYZ_camX



        XYZ_origin_one = np.matmul(origin_T_camX, XYZ_camX_one)
        XYZ_origin = XYZ_origin_one[:3]

        gripper_target_quat = np.array([1, 0, 1, 0])

        gripper_rot = transformations.quaternion_matrix(gripper_target_quat)[:3,:3]

        cos_theta = np.cos(-grasp_angle)
        sin_theta = np.sin(-grasp_angle)


        xy_rot_xmat = np.array([[cos_theta,-sin_theta,0],
                                [sin_theta,cos_theta,0],
                                 [0,0, 1]])
        final_rot = np.matmul(xy_rot_xmat, gripper_rot)

        if final_rot[1,1] < 0:
            rot = 180
            xmat = np.array([[1, 0, 0],
                             [0, np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot))],
                             [0, np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot))]
                            ])
            final_rot = np.matmul(final_rot, xmat)
        final_rot_tmp = np.eye((4))
        final_rot_tmp[:3, :3] = final_rot

        gripper_quat = transformations.quaternion_from_matrix(final_rot_tmp)

        # don't poke inside the table
        XYZ_origin[2,0] = np.maximum(XYZ_origin[2,0], self.config.table_top[2])
        return XYZ_origin, gripper_quat




    def goToGoal(self, env, lastObs, mesh_obj_info=None):

        goal = lastObs['desired_goal']



        #grasp_pos, gripper_target_quat  = self.get_grasp_from_grasp_sampler(env, mesh_obj_info)
        grasp_pos, gripper_target_quat = self.sample_grasp(env, mesh_obj_info)
        #gripper_target_quat = np.array([1, 0, 1, 0])


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


        target_pos = self.compute_pts_away_from_grasp_point(grasp_point, gripper_target_quat, dist=0.2) # the gripper is -0.105 away from the grasp point

        ## go to the top of the rim
        #rim_top_pos = self.compute_top_of_rim(center, extents, gripper_target_quat)     
         # face down
        new_gripper_quat  = gripper_target_quat#transformations.quaternion_multiply(obj_xquat, gripper_quat)

        rewards, timeStep = self.gripper_goto_pos_orn(env, target_pos, new_gripper_quat, goal_current_thres= 0.002, speed=6, open=True, timeStep=timeStep, mesh_obj_info=mesh_obj_info, max_time_limit=self.max_path_length/3)
        cur_reward += rewards

        # go toward rim pos
        _, center, _, _, bbox_xpos = self.get_bbox_properties(env, mesh_obj_info)

        #grasp_point = self.get_grasp_point(center)
        rim_pos = self.compute_pts_away_from_grasp_point(grasp_point, gripper_target_quat, dist=0.01)



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