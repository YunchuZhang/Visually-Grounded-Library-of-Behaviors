
import pcp_utils
import sys
import os
import click
import yaml


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



import argparse
from pcp_utils import utils
from pcp_utils.mesh_object import MeshObject
from pcp_utils.utils import Config
from pcp_utils.parse_task_files import generate_integrated_xml

import tqdm
import numpy as np


##### Imports related to environment #############
import gym


class Parser:
    class Config(Config):
        num_rollouts = 100
        max_path_length = 50
        accept_threshold = 0.9
        num_threads = 5
        bbox_indicator = False


        # params for the environment
        env_name = None
        env_base_xml_file = "" #table, the kind of robot, object placeholder
        n_robot_dof = 8 # 4 means xyz, 6 means having rotations
        render = True
        randomize_color = False
        init_info = None

        #camera_parameters
        camera_img_height = 128
        camera_img_width = 128
        camera_radius = 0.3
        camera_fov_y = 45
        camera_pitch = [20, 60, 20]
        camera_yaw = [0, 350, 36]
        camera_yaw_list = None
        camera_save_image = False
        camera_recon_scene = True
        camera_lookat_pos = [0, 0, 0.4]
    
        table_top = [1.3, 0.75, 0.4]
        table_T_camR = [0, 0, 0]
        cut_off_points = [0.5, 0.5, 0.5]  # for cropping pointcloud



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
        self.cut_off_points = config.cut_off_points

        self.run_name = run_name

        self.policy_bank = initial_policies
        self.output_file = output_file

        output_folder = "/".join(output_file.split("/")[:-1])
        utils.makedir(output_folder)
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


        # set camera for this:
        self.camera_positions, self.camera_quats = pcp_utils.cameras.generate_new_cameras_hemisphere(radius=self.config.camera_radius,
            lookat_point=self.config.camera_lookat_pos, pitch=self.config.camera_pitch, yaw=self.config.camera_yaw, yaw_list=self.config.camera_yaw_list)
        self.n_cams = len(self.camera_positions)

        # mujoco: x right, y front; adam: x right, y down
        mujoco_T_adam = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        origin_T_camR_xpos = np.array(config.table_top, np.float32) + np.array(config.table_T_camR, np.float)

        origin_T_adam = np.eye(4, dtype=np.float32)
        origin_T_adam[:3, :3] = mujoco_T_adam
        origin_T_adam[:3, 3] = origin_T_camR_xpos
        origin_T_adam[3,3] = 1
        self.origin_T_adam = origin_T_adam
        self.adam_T_origin = np.linalg.inv(self.origin_T_adam)


        table_T_camR = np.ones((4, 1), np.float32)
        table_T_camR[:3, 0] = config.table_T_camR
        self.table_top_R = np.dot(self.adam_T_origin, table_T_camR)



    def render_images(self, env):
        """
        This is copied from policy_compression_with_init_policies_3dtensor.py
        TODO: @Fish: would it be better to move this func to utils
        """
        rgbs = []
        depths = []
        pix_T_camXs = []
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
            pix_T_camXs.append(pix_T_camX)
            origin_T_camXs.append(origin_T_camX)
    
        images = dict()
        images['rgb_camXs'] = np.stack(rgbs, axis=0)
        images['depth_camXs'] = np.stack(depths, axis=0)
        images['pix_T_camXs'] = np.stack(pix_T_camXs, axis=0)
        images['origin_T_camXs'] = np.stack(origin_T_camXs, axis=0)
        return images
    def preprocess_points_for_6dgrasp(self, env, obj_info, images, save=False, save_dir=None, obj_name=None):
        rgbs = images['rgb_camXs']
        depths = images['depth_camXs']
        pix_T_camXs = images['pix_T_camXs']
        camR_T_camXs = images['camR_T_camXs']
        
        # unproject to get the pointcloud
        _, xyz_camRs, _ = pcp_utils.np_vis.unproject_depth(
            depths, pix_T_camXs, camR_T_camXs,
            camR_T_origin = None, clip_radius=1.0,
            do_vis=False
        )
        
        # NOTE: I am using all the views here, TODO: make it use one view
        xyz_camRs = np.stack(xyz_camRs, axis=0)
        xyz_camRs = xyz_camRs.reshape(-1, 3)

        # compute the inliers, I am using priviledged info about table size

        # not good. should use object bounding box

        inliers_x = np.abs(xyz_camRs[:, 0]) <= self.cut_off_points[0]
        inliers_y = np.abs(xyz_camRs[:, 1]) <= self.cut_off_points[1]
        inliers_z = np.abs(xyz_camRs[:, 2]) <= self.cut_off_points[2]

        selection = inliers_x & inliers_y & inliers_z
        
        pc = xyz_camRs.copy()
        pc = pc[selection, :]


        pc_colors = np.stack(rgbs, axis=0)
        pc_colors = np.reshape(pc_colors, [-1, 3])
        pc_colors = pc_colors[selection, :]

        # extract the object pointcloud from table and object pointcloud

        # no, just get it from object box
        obj_xml = obj_info.obj_xml_file
        obj_xpos = env.env.sim.data.get_body_xpos('object0')
        obj_xmat = env.env.sim.data.get_body_xmat('object0')

        composed_xmat = np.eye(4, dtype=np.float32)
        composed_xmat[:3, 3] = obj_xpos
        composed_xmat[:3, :3] = np.reshape(obj_xmat, [3, 3])
        composed_xmat = np.dot(self.adam_T_origin, composed_xmat)
        object_xpos_adam =  composed_xmat[:3, 3] #np.dot(self.adam_T_origin, pcp_utils.geom.pts_addone(np.reshape(object_xpos, [1, 3])).T)[:3, 0]
        object_xmat_adam =  composed_xmat[:3, :3]#np.dot(self.adam_T_origin[:3, :3], np.reshape(object_xmat, [3, 3]))

        scale = obj_info.scale
        coords, combined_mesh = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(
            obj_info.obj_xml_file, object_xpos_adam, object_xmat_adam, scale=scale, euler=obj_info.euler, return_combined_mesh=True)
        

        print("x real xmat", obj_xmat)
        dalpha, dbeta, dgamma = obj_info.euler
        import transformations
        print("x computed xmat", transformations.euler_matrix(dalpha, dbeta, dgamma))
        #coords2 = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(
        #    obj_info.obj_xml_file, object_xpos_adam, object_xmat_adam)

        inliers_x = pc[:, 0] >= coords[0, 0]
        inliers_x2 = pc[:, 0] <= coords[-1, 0]
        inliers_y = pc[:, 1] >= coords[0, 1]
        inliers_y2 =  pc[:, 1] <= coords[-1, 1]
        inliers_z = pc[:, 2] >= coords[0, 2]
        inliers_z2 =  pc[:, 2] <= coords[-1, 2]
        #inliers_table =  pc[:, 1] < -0.002

        selection = inliers_x & inliers_y & inliers_z & inliers_x2 & inliers_y2 & inliers_z2 #& inliers_table

        object_pc = pc.copy()
        object_pc = object_pc[selection, :]

        if save:
            self.save_pcd(pc, save_dir, obj_name, pc_colors)
            self.save_pcd(object_pc, save_dir, obj_name + "_obj", pc_colors)
        import ipdb; ipdb.set_trace()

        return object_pc, pc, pc_colors

    @staticmethod
    def save_pcd(pts, save_dir=None, obj_name=None, color=None):
        if save_dir is None:
            save_dir = "tmp"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # if the name is none then assign one
        if obj_name is None:
            obj_name = "recon_pts.npy"
        else:
            obj_name = f"{obj_name}.npy"

        if color is not None:
            save_dict = dict(pts = pts, color = color)
            np.save(os.path.join(save_dir, obj_name), save_dict)
        else:
            np.save(os.path.join(save_dir, obj_name), pts)

    def convert_to_adam(self, images):
        """
        This is copied from policy_compression_with_init_policies_3dtensor.py
        TODO: @Fish: would it be better to move this func to utils
        NOTE: Modifies the images dict to add a new key
        """
        origin_T_camXs = images['origin_T_camXs']
        camR_T_camXs = []
        for origin_T_camX in origin_T_camXs:
            camR_T_camX = np.dot(self.adam_T_origin, origin_T_camX)
            camR_T_camXs.append(camR_T_camX)

        camR_T_camXs = np.stack(camR_T_camXs, axis=0)
        images['camR_T_camXs'] = camR_T_camXs
        return images

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
    def add_object(self, obj):
        #share the env

        # make xml for the object:
        integrated_xml = generate_integrated_xml(self.env_base_xml_file, obj.obj_xml_file, scale=obj.scale, mass=obj.mass,
                 euler=obj.euler, add_bbox_indicator=self.bbox_indicator, randomize_color=self.randomize_color, prefix=self.run_name)

        #obj.xml_file)# xml_path="fetch/pick_and_place_kp30000_debug.xml") #xml_path=obj.xml_file)
        env = gym.make(self.env_name, xml_path=integrated_xml, use_bbox_indicator=self.bbox_indicator,
            n_actions=self.n_robot_dof, init_info=self.init_info)
        print(f'max env steps are: {env._max_episode_steps}')

        # env.render()
        # this ordering should be something learnable
        is_clustered = False
        success_rates = [float(item) for item in obj.success_rates_over_class.split(" ")]

        import ipdb; ipdb.set_trace()

        for cid, cluster in self.clusters.items():
 
            cluster_id = int(cluster["expert_name"][1:])
            if success_rates[cluster_id] >= 0.8:
                for _ in range(self.num_rollouts):

                    obs = env.reset()
                    for _ in range(20):
                        obsDataNew, reward, done, info = env.step(np.zeros(self.n_robot_dof))
                        if self.config.render:
                            env.render()
       
                    # if object success rate is high -> collect postive grasping
                    images = self.render_images(env)
                    images = self.convert_to_adam(images)
                    import ipdb; ipdb.set_trace()
                    obj_pc, pc, pc_colors = self.preprocess_points_for_6dgrasp(env, obj, images, obj_name=obj.name, save_dir="tmp", save=True)

                    # collect pcs
                            
                    import ipdb; ipdb.set_trace()
                
                # collect grasp points: grasp_rt


        if not is_clustered:
            self.object_not_clustered.append(obj)
            self.object_to_cluster.append((obj, "not_assigned"))

        gym_xml_path = os.path.join(pcp_utils.utils.get_gym_dir(), 'gym/envs/robotics/assets')
        integrated_xml_full_path = os.path.join(gym_xml_path,  integrated_xml)
        os.remove(integrated_xml_full_path)


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
            for key_id, item in obj.config.__dict__.items():
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
            print(f'    {obj.name} ')

        #print("Percentage compression for threshold {} : {} ".format(self.accept_threshold, self.num_clusters/len(MESHES)))
        #print("Cluster Accuracies")
        #print(self.cluster_acc)

@click.command()
@click.argument("config_file")#config
@click.option("--task_config_file") #for the objects
@click.option("--output_file") #for the objects
@click.option("--run_name") #define run name to avoid generate_xml to overwrite
def main(config_file, task_config_file, output_file, run_name):


    config = utils.config_from_yaml_file(config_file)
    objs_config = utils.config_from_yaml_file(task_config_file)
    initial_policies = []
    for policy_name in objs_config["clusters"]:
        policy_param = objs_config["clusters"][policy_name]

        if policy_param['params'] is None:
            policy_param['params'] = dict()
        # add name to the parameter
        policy_param['params']["policy_name"] = policy_name
        policy_class, policy_config = utils.import_class_from_config(policy_param)
        policy = policy_class(policy_config)
        initial_policies.append(policy)
    print(f"find {len(initial_policies)} initial policies")    


    obj_list = [item for item in objs_config['objs'].items()]

    # init all objects
    # objs_config = utils.config_from_yaml_file(task_config_file)
    objects_to_cluster = []
    for obj_name, obj_config in objs_config['objs'].items():
        obj_config_class = MeshObject.Config().update(obj_config)
        obj = MeshObject(obj_config_class, obj_name)
        objects_to_cluster.append(obj)
    print(f"find {len(objects_to_cluster)} objects to cluster")


    updated_config = Parser.Config().update(config)
    compressor = Parser(updated_config,
                        initial_policies,
                                  output_file=output_file,
                                  run_name = run_name
                 )

    nmeshes = len(objects_to_cluster)
    for mesh_id, meshobj in enumerate(objects_to_cluster):
        print(f"============ processing object {mesh_id}/{nmeshes} ============ ")
        compressor.add_object(meshobj)
        if (mesh_id + 1) %10 == 0:
            compressor.output_xml()
    compressor.print_policy_summary()

if __name__=="__main__":
    main()