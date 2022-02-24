import sys
import os
import click
import yaml
import open3d as o3d
import argparse
import random
RANDOM_SEED = 0

import pcp_utils
# add gym and baseline to the dir
gym_path = pcp_utils.utils.get_gym_dir()
baseline_path = pcp_utils.utils.get_baseline_dir()
sys.path.append(gym_path)
sys.path.append(baseline_path)


# make symbolic link of the mesh under quantize-gym/gym/envs/robotics/assets/stls
source_mesh_dir = pcp_utils.utils.get_mesh_dir()
gym_mesh_dir = os.path.join(pcp_utils.utils.get_gym_dir(), 'gym/envs/robotics/assets/stls/meshes')


if not os.path.exists(gym_mesh_dir):
    os.symlink(source_mesh_dir, gym_mesh_dir,  target_is_directory=True)


import pcp_utils.load_ddpg as load_ddpg
#from pcp_utils.rollouts import simple_rollouts
from pcp_utils.mesh_object import MeshObject
from pcp_utils.utils import Config
from pcp_utils.parse_task_files import generate_integrated_xml

import tqdm
import numpy as np


##### Imports related to environment #############
import gym


class PolicyCompressor:
    class Config(Config):
        num_rollouts = 100
        max_path_length = 50
        accept_threshold = 0.9
        num_threads = 5
        bbox_indicator = False
        
        # having a selector here
        selector = None

        # params for the environment
        env_name = None
        env_base_xml_file = "" #table, the kind of robot, object placeholder
        n_robot_dof = 8 # 4 means xyz, 6 means having rotations
        render = True
        randomize_color = True
        init_info = None
        #render = False

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

        table_top = [1.3, 0.75, 0.4]
        table_T_camR = [0, 0, 0]

        # train_val_test split
        train_val_test_ratios = [0.5, 0.5, 0.5]





    def __init__(self,  config:Config, initial_policies=[], output_file="", run_name=""):

        self.config = config

        self.num_rollouts = config.num_rollouts
        self.max_path_length = config.max_path_length
        self.accept_threshold = config.accept_threshold
        self.num_threads = config.num_threads
        self.env_name = config.env_name
        self.env_base_xml_file = config.env_base_xml_file
        self.bbox_indicator = config.bbox_indicator
        self.n_robot_dof = config.n_robot_dof
        self.randomize_color = config.randomize_color
        self.init_info = config.init_info
         
        self.run_name = run_name

        self.policy_bank = initial_policies

        self.output_file = output_file
        output_folder = "/".join(output_file.split("/")[:-1])
        pcp_utils.utils.makedir(output_folder)
        self.objects_output_xml = dict()
        self.object_output_xml_id = 0

        self.clusters = {}
        # key: name of the cluster(c1)
        # value:
        #        objects: objects inside this clusters  (objects, success rate)
        #        expert_id: expert associate to it
        #        expert_name:
        #        3d tensor model:

        self.object_to_cluster = [] #just a lookup
        self.num_clusters = 0
        self.object_not_clustered = []

        self.init_clusters()

        # having a selector
        if self.config.selector is not None:
            selector_class, selector_config = pcp_utils.utils.import_class_from_config(self.config.selector)
            self.selector = selector_class(selector_config)

        # cameras

        self.camera_positions, self.camera_quats = pcp_utils.cameras.generate_new_cameras_hemisphere(radius=self.config.camera_radius,
            lookat_point=self.config.camera_lookat_pos, pitch=self.config.camera_pitch, yaw=self.config.camera_yaw, yaw_list=self.config.camera_yaw_list)
        self.n_cams = len(self.camera_positions)

        mujoco_T_adam = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        origin_T_camR_xpos = np.array(config.table_top, np.float32) + np.array(config.table_T_camR, np.float)

        origin_T_adam = np.zeros((4,4), dtype=np.float32)
        origin_T_adam[:3, :3] = mujoco_T_adam
        origin_T_adam[:3, 3] = origin_T_camR_xpos
        origin_T_adam[3,3] = 1
        self.origin_T_adam = origin_T_adam
        self.adam_T_origin = np.linalg.inv(self.origin_T_adam)

    def init_clusters(self):
        for policy in self.policy_bank:
            cluster_id = self.num_clusters
            self.clusters[f'c{cluster_id}'] =dict()
            self.clusters[f'c{cluster_id}']['objects'] = []
            self.clusters[f'c{cluster_id}']['expert_name'] = policy.policy_name
            self.clusters[f'c{cluster_id}']['expert'] = policy #.policy_model
            self.num_clusters += 1


    # Access to a minimal policy bank, also has information about which meshes to run on which policy
    # Takes in a new object and determines if it can be merged with an existing policy or can spawn a new policy
    # input: new object that needs to be classified
    # mesh should be mesh_id like 159e56c18906830278d8f8c02c47cde0, or b9004dcda66abf95b99d2a3bbaea842a which are ShapeNet ids

    def render_images(self, env):
        """
        go through all the cameras and get rgbd
        """
        rgbs = []
        depths = []
        pix_T_cams = []
        origin_T_camXs = []
        for cam_id in range(self.n_cams):
            # need to reset everytime you want to take the picture: the camera has mass and it will fall during execution
            env.set_camera(self.camera_positions[cam_id, :], self.camera_quats[cam_id, :], camera_name= f"ext_camera_0")
            rgb, depth = env.render_from_camera(self.config.camera_img_height, self.config.camera_img_width, camera_name=f"ext_camera_0")

            # need to convert depth to real numbers
            pix_T_camX = pcp_utils.cameras.get_intrinsics(self.config.camera_fov_y, self.config.camera_img_width, self.config.camera_img_height)
            origin_T_camX = pcp_utils.cameras.gymenv_get_extrinsics(env, f'ext_camera_0')

            rgbs.append(rgb)
            depths.append(depth)
            pix_T_cams.append(pix_T_camX)
            origin_T_camXs.append(origin_T_camX)

        images = dict()
        images['rgb_camXs'] = np.stack(rgbs, axis=0)
        images['depth_camXs'] = np.stack(depths, axis=0)
        images['pix_T_cams'] = np.stack(pix_T_cams, axis=0)
        images['origin_T_camXs'] = np.stack(origin_T_camXs, axis=0)

        return images

    def convert_to_adam(self, images, bbox_points_from_mesh_adam=None):
        origin_T_camXs = images['origin_T_camXs']
        camR_T_camXs = []
        for origin_T_camX in origin_T_camXs:
            camR_T_camX = np.dot(self.adam_T_origin, origin_T_camX)
            camR_T_camXs.append(camR_T_camX)

        camR_T_camXs = np.stack(camR_T_camXs, axis=0)
        images['origin_T_camXs'] = camR_T_camXs

    def visualize(self, images, bbox_points_from_mesh=None):
        depths = images['depth_camXs']
        pix_T_cams = images['pix_T_cams']
        origin_T_camXs = images['origin_T_camXs']

        _, xyz_camRs, _ = pcp_utils.np_vis.unproject_depth(depths,
            pix_T_cams,
            origin_T_camXs,
            camR_T_origin = None, #np.linalg.inv(self.origin_T_adam),
            clip_radius=5.0,
            do_vis=False)

        all_xyz_camR = np.concatenate(xyz_camRs[:2], axis=0)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=np.zeros(3), size=0.8)
        object_pcd = pcp_utils.np_vis.get_pcd_object(all_xyz_camR, clip_radius=3.0)

        things_to_print = [object_pcd, frame]

        if bbox_points_from_mesh is not None:
            bbox_lineset_from_mesh = pcp_utils.np_vis.make_lineset(bbox_points_from_mesh)
            things_to_print.append(bbox_lineset_from_mesh)


        # transform object xpos and xmat to the adam coordinate (x right, y downs)
        #bbox_lineset_from_mesh_adam = pcp_utils.np_vis.make_lineset(bbox_points_from_mesh_adam)
        o3d.visualization.draw_geometries(things_to_print) #, bbox_lineset_from_mesh_adam])

    def get_object_pose(self, env, object_name, coord="origin"):
        object_xpos, object_xmat = env.get_object_pos(object_name)
        if coord == "origin":
            return object_xpos, object_mat

        composed_xmat = np.eye(4, dtype=np.float32)
        composed_xmat[:3, 3] = object_xpos
        composed_xmat[:3, :3] = np.reshape(object_xmat, [3, 3])
        composed_xmat = np.dot(self.adam_T_origin, composed_xmat)
        object_xpos_adam =  composed_xmat[:3, 3] #np.dot(self.adam_T_origin, pcp_utils.geom.pts_addone(np.reshape(object_xpos, [1, 3])).T)[:3, 0]
        object_xmat_adam =  composed_xmat[:3, :3]#np.dot(self.adam_T_origin[:3, :3], np.reshape(object_xmat, [3, 3]))

        return object_xpos_adam, object_xmat_adam

    def get_inputs_for_selector(self, env, obj):
        feed = dict()
        # get images
        images = self.render_images(env)
        # convert to adam coordinate
        self.convert_to_adam(images)
        object_xpos_adam, object_xmat_adam = self.get_object_pose(env, "object0", coord="adam")
        bbox_points_from_mesh_adam, combined_mesh = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(obj.obj_xml_file, object_xpos_adam, object_xmat_adam, scale=obj.scale, euler=obj.euler, return_combined_mesh=True)
        #print(bbox_points_from_mesh_adam)
        #print(object_xpos_adam, object_xmat_adam)
        #print("center of mass", combined_mesh.center_mass)
        #print("mean of vertices", combined_mesh.vertices.mean(axis=0))

        # #import imageio
        #for img_id in range(10):
        #    imageio.imwrite(f"tmp/{img_id}.png", images["rgb_camXs"][img_id])
        #self.visualize(images, bbox_points_from_mesh=bbox_points_from_mesh_adam)

        #import ipdb; ipdb.set_trace()

        NVIEWS, _, _, _ = images["rgb_camXs"].shape
        feed.update(images)
        feed["bbox_in_ref_cam"] = np.expand_dims(bbox_points_from_mesh_adam, axis=0)
        feed["origin_T_camRs"] = np.tile(np.expand_dims(np.eye(4), axis=0), [NVIEWS, 1, 1])
        feed["set_num"] = 1
        feed["set_name"] = "val"
        return feed

    def add_object(self, obj):
        #share the env

        # make xml for the object:
        integrated_xml = generate_integrated_xml(self.env_base_xml_file, obj.obj_xml_file, scale=obj.scale, mass=obj.mass, euler=obj.euler,
                add_bbox_indicator=self.bbox_indicator, randomize_color=self.randomize_color, prefix=self.run_name)

        #obj.xml_file)# xml_path="fetch/pick_and_place_kp30000_debug.xml") #xml_path=obj.xml_file)
        env = gym.make(self.env_name, xml_path=integrated_xml, use_bbox_indicator=self.bbox_indicator,
            n_actions=self.n_robot_dof, init_info=self.init_info)
        print(f'max env steps are: {env._max_episode_steps}')
        env.seed(RANDOM_SEED)
        env.action_space.seed(RANDOM_SEED)

        obs = env.reset()
        # this part will decide which cluster to try first
        current_vis_state = self.get_inputs_for_selector(env, obj)
        results = self.selector.predict_forward(current_vis_state)
        cluster_rank, cluster_score = self.selector.compute_nearest_cluster(results["object_tensor"][0])

        # env.render()
        # this ordering should be something learnable
        is_clustered = False
        cname = cluster_rank[0]
        cluster = self.clusters[cname]

        # load policy of the first mesh (parent mesh) in an existing cluster
        #print("Checking performance of {} on policy for {}".format(obj, cid))
        stats = cluster['expert'].run_forwards(env, obj=obj, num_rollouts=self.num_rollouts, path_length=self.max_path_length)

        success_rate = stats['success_rate']
        print("Success Rate ", success_rate, "using object: ", obj.name, "by running on cluster", cname)

        cluster['objects'].append((obj, success_rate))
        self.object_to_cluster.append((obj, cname))

        env.close()

    def output_xml(self):
        # output cluster name

        clusters_output = dict()
        for cluster_name in self.clusters:
            clusters_output[cluster_name] = dict()
            clusters_output[cluster_name]['expert_name'] = self.clusters[cluster_name]['expert_name']
            clusters_output[cluster_name]['fn'] = self.clusters[cluster_name]['expert'].__class__.__module__ + ":" + \
                                                         self.clusters[cluster_name]['expert'].__class__.__name__
            clusters_output[cluster_name]['params'] = self.clusters[cluster_name]['expert'].config.__dict__


        output_clusters = dict()
        output_clusters['clusters'] = clusters_output

        output_objs = dict()
        # output objects
        for obj, cluster_id in self.object_to_cluster[self.object_output_xml_id:]:
            self.objects_output_xml[obj.name] = dict()
            for key_id, item in obj.config.items():
                self.objects_output_xml[obj.name][key_id] = item
            self.objects_output_xml[obj.name]['cluster_id'] = cluster_id

        self.object_output_xml_id = len(self.objects_output_xml)
        output_objs["objs"] = self.objects_output_xml

        with open(self.output_file, 'w') as file:
            yaml.dump(self.config.__dict__, file)
            yaml.dump(output_clusters, file)
            yaml.dump(output_objs, file)

    def print_policy_summary(self):
        # want to write out something with the policy and objects, and we can load from it.

        print("Compressed Meshes")
        for cid, cluster in self.clusters.items():
            print(f'cluster {cid}')
            expert_name = cluster['expert_name']
            print(f'expert used in the cluster: {expert_name}')
            for obj, success_rate in cluster['objects']:
                print(f'    {obj.name} ({success_rate})')
        print("==========================")
        for obj in self.object_not_clustered:
            print(f'    {obj.name}')

        #print("Percentage compression for threshold {} : {} ".format(self.accept_threshold, self.num_clusters/len(MESHES)))
        #print("Cluster Accuracies")
        #print(self.cluster_acc)

@click.command()
@click.argument("config_file")#config
@click.option("--task_config_file") #for the objects
@click.option("--output_file") #for the objects
@click.option("--run_name") #for the objects
def main(config_file, task_config_file, output_file, run_name):


    config = pcp_utils.utils.config_from_yaml_file(config_file)
    # init all the policy
    initial_policies = []
    for policy_name in config["initial_policies"]:
        policy_param = config["initial_policies"][policy_name]

        if policy_param['params'] is None:
            policy_param['params'] = dict()
        # add name to the parameter
        policy_param['params']["policy_name"] = policy_name
        policy_class, policy_config = pcp_utils.utils.import_class_from_config(policy_param)
        policy = policy_class(policy_config)
        initial_policies.append(policy)
    print(f"find {len(initial_policies)} initial policies")

    # init all objects
    objs_config = pcp_utils.utils.config_from_yaml_file(task_config_file)
    objects_to_cluster = []
    for obj_name, obj_config in objs_config['objs'].items():
        obj = MeshObject(obj_config, obj_name)
        objects_to_cluster.append(obj)
    print(f"find {len(objects_to_cluster)} objects to cluster")

    updated_config = PolicyCompressor.Config().update(config)
    compressor = PolicyCompressor(updated_config,
                                  initial_policies=initial_policies,
                                  output_file=output_file,
                                  run_name = run_name
                 )

    nmeshes = len(objects_to_cluster)
    for mesh_id, meshobj in enumerate(objects_to_cluster):
        print(f"============ {run_name} processing object {mesh_id}/{nmeshes} ============ ")
        compressor.add_object(meshobj)
    compressor.output_xml()
    compressor.print_policy_summary()

if __name__=="__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    main()
