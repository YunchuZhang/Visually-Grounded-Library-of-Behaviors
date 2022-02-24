import pcp_utils
import sys
import os
import click
import yaml
import open3d as o3d
import tensorflow as tf

# add gym and baseline to the dir
gym_path = pcp_utils.utils.get_gym_dir()
baseline_path = pcp_utils.utils.get_baseline_dir()
graspnet_path = pcp_utils.utils.get_6dof_graspnet_dir()
sys.path.append(gym_path)
sys.path.append(baseline_path)
sys.path.append(graspnet_path)

# make symbolic link of the mesh under quantize-gym/gym/envs/robotics/assets/stls
source_mesh_dir = pcp_utils.utils.get_mesh_dir()
gym_mesh_dir = os.path.join(pcp_utils.utils.get_gym_dir(), 'gym/envs/robotics/assets/stls/meshes')

if not os.path.exists(gym_mesh_dir):
    os.symlink(source_mesh_dir, gym_mesh_dir,  target_is_directory=True)

import argparse
import pcp_utils.load_ddpg as load_ddpg
from pcp_utils import utils
import pcp_utils
from pcp_utils.mesh_object import MeshObject
from pcp_utils.utils import Config
from pcp_utils.parse_task_files import generate_integrated_xml

import grasp_estimator
import glob
from grasp_data_reader import regularize_pc_point_count

import tqdm
import numpy as np


##### Imports related to environment #############
import gym

class Grasp6d_graspnet:
    class Config(Config):
        num_rollouts = 100
        max_path_length = 50
        accept_threshold = 0.9
        num_threads = 5
        bbox_indicator = False

        env_name = None
        env_base_xml_file = "" #table, the kind of robot, object placeholder
        n_robot_dof = 4 # 4 means xyz, 8 means having rotations

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
        cut_off_points = [0.3, 0.5, 0.5]  # for cropping pointcloud

        # train_val_test split
        train_val_test_ratios = [0.5, 0.5, 0.5]

        # path for checkpoints
        vae_checkpoint_folder = f'{graspnet_path}/checkpoints/latent_size_2_ngpus_1_gan_1_confidence_weight_0.1_npoints_1024_num_grasps_per_object_256_train_evaluator_0_'
        evaluator_checkpoint_folder = f'{graspnet_path}/checkpoints/npoints_1024_train_evaluator_1_allowed_categories__ngpus_8_/'
        gradient_based_refinement = False
        grasp_conf_threshold = 0.8
    
    def __init__(self, config:Config):
        self.config = config
        
        self.num_rollouts = config.num_rollouts
        self.max_path_length = config.max_path_length
        self.accept_threshold = config.accept_threshold
        self.num_threads = config.num_threads
        self.env_name = config.env_name
        self.env_base_xml_file = config.env_base_xml_file
        self.bbox_indicator = config.bbox_indicator
        self.n_robot_dof = config.n_robot_dof
        self.vae_checkpoint_folder = config.vae_checkpoint_folder
        self.evaluator_checkpoint_folder = config.evaluator_checkpoint_folder
        self.gradient_based_refinement = False
        self.grasp_conf_threshold = 0.8
        self.cut_off_points = config.cut_off_points
        self.output_grasps_dir = "vae_generated_grasps"

        if not os.path.exists(self.output_grasps_dir):
            os.makedirs(self.output_grasps_dir)

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

        # to the center of the table
        origin_T_tc = np.zeros((4,4), dtype=np.float32)
        origin_T_tc[:3, :3] = np.eye(3, dtype=np.float32)
        origin_T_tc[:3, 3] = origin_T_camR_xpos
        origin_T_tc[3,3] = 1

        self.origin_T_tc = origin_T_tc
        self.tc_T_origin = np.linalg.inv(self.origin_T_tc)
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
    
    def extract_object_pcd(self, obj_and_table_pts, save=False, obj_name=None):
        pcd = o3d.geometry.PointCloud()

        obj_and_table_pts[:,2] > self.config.table_top[2]

        pcd.points = o3d.utility.Vector3dVector(obj_and_table_pts)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        save_dir = "cropped_pc"
        obj_pts = np.asarray(outlier_cloud.points)
        
        if save:
            self.save_pcd(obj_pts, save_dir, obj_name)
        return obj_pts
        

    def preprocess_points_for_6dgrasp(self, images, save=False, save_dir=None, obj_name=None):
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
        object_pc = self.extract_object_pcd(pc, save=True, obj_name=obj_name)

        if save:
            self.save_pcd(pc, save_dir, obj_name, pc_colors)

        return object_pc, pc, pc_colors

    def convert_to_tc(self, images):
        """
        This is copied from policy_compression_with_init_policies_3dtensor.py
        TODO: @Fish: would it be better to move this func to utils
        NOTE: Modifies the images dict to add a new key
        """
        origin_T_camXs = images['origin_T_camXs']
        camR_T_camXs = []
        for origin_T_camX in origin_T_camXs:
            camR_T_camX = np.dot(self.tc_T_origin, origin_T_camX)
            camR_T_camXs.append(camR_T_camX)

        camR_T_camXs = np.stack(camR_T_camXs, axis=0)
        images['camR_T_camXs'] = camR_T_camXs
        return images

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
    
    def visualize(self, images, save=False, save_dir=None, obj_name=None):
        """
        This is copied from policy_compression_with_init_policies_3dtensor.py
        TODO: @Fish: would it be better to move this func to utils
        """
        depths = images['depth_camXs']
        pix_T_camXs = images['pix_T_camXs']
        origin_T_camXs = images['origin_T_camXs']
        camR_T_camXs = images['camR_T_camXs']

        _, xyz_camRs, _ = pcp_utils.np_vis.unproject_depth(depths,
            pix_T_camXs,
            camR_T_camXs,
            camR_T_origin = None, #np.linalg.inv(self.origin_T_adam),
            clip_radius=5.0,
            do_vis=False)
        if not save:
            all_xyz_camR = np.concatenate(xyz_camRs, axis=0)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=np.zeros(3), size=0.8)
            object_pcd = pcp_utils.np_vis.get_pcd_object(all_xyz_camR, clip_radius=2.0)

            # transform object xpos and xmat to the adam coordinate (x right, y downs)
            # bbox_lineset_from_mesh_adam = pcp_utils.np_vis.make_lineset(bbox_points_from_mesh_adam)
            o3d.visualization.draw_geometries([object_pcd, frame]) #, bbox_lineset_from_mesh_adam])
        else:
            self.save_pcd(xyz_camRs, save_dir, obj_name)
            

    def add_object(self, obj):

        integrated_xml = generate_integrated_xml(self.env_base_xml_file,
                                                 obj.obj_xml_file,
                                                 scale=obj.scale,
                                                 add_bbox_indicator=self.bbox_indicator)
        
        env = gym.make(self.env_name,
                       xml_path=integrated_xml,
                       use_bbox_indicator=self.bbox_indicator,
                       n_actions=self.n_robot_dof)

        obs = env.reset()
        pcp_utils.env_utils.reset_everything_on_table(env, obj, max_run=100)

        images = self.render_images(env)
        images = self.convert_to_adam(images)

        #### Do you need to save it everytime ? ####
        # self.visualize(images, save=True, obj_name=obj.name)
        #### I asked you a question above lolll ####

        #### Convert things to pointcloud and preprocess it, I am following instructions in 6dof_graspnet/demo/main.py ####
        obj_pc, pc, pc_colors = self.preprocess_points_for_6dgrasp(images, save=True, obj_name=obj.name)
        #### Pointcloud is ready for the forward pass ####

        #### Save the environment state too so it can be reloaded offline ####
        env_state = env.env.sim.get_state()

        return obj_pc, pc, pc_colors, env_state

@click.command()
@click.argument("config_file")#config
@click.option("--task_config_file") #for the objects
@click.option("--output_file") # for generating output report
def main(config_file, task_config_file, output_file):
    config = utils.config_from_yaml_file(config_file)
    
    # object configurations
    objs_config = utils.config_from_yaml_file(task_config_file)
    objects_to_get_grasps_for = list()
    for obj_name, obj_config in objs_config['objs'].items():
        obj_config_class = MeshObject.Config().update(obj_config)
        obj = MeshObject(obj_config_class, obj_name)
        objects_to_get_grasps_for.append(obj)
    
    print(f'found {len(objects_to_get_grasps_for)} objects to find grasps for')
    updated_config = Grasp6d_graspnet.Config().update(config)
    grasper = Grasp6d_graspnet(updated_config)

    ##### Prepare the 6dof graspnet network for forward pass ######
    cfg = grasp_estimator.joint_config(
        grasper.vae_checkpoint_folder,
        grasper.evaluator_checkpoint_folder,
    )
    cfg['threshold'] = grasper.grasp_conf_threshold
    cfg['sample_based_improvement'] = 1 - int(grasper.gradient_based_refinement)
    cfg['num_refine_steps'] = 10 if grasper.gradient_based_refinement else 20
    estimator = grasp_estimator.GraspEstimator(cfg)
    sess = tf.Session()
    estimator.build_network()
    estimator.load_weights(sess)
    ##### End of 6dof graspnet network preparation ################

    nmeshes = len(objects_to_get_grasps_for)
    for mesh_id, meshobj in enumerate(objects_to_get_grasps_for):
        print(f'---- processing {mesh_id}/{nmeshes} ------')
        obj_pc, pc, pc_colors, env_state = grasper.add_object(meshobj)

        # the object is added, data is collected, processed and returned
        # now sample the grasps, and save them
        latents = estimator.sample_latents()
        generated_grasps, generated_scores, _ = estimator.predict_grasps(
            sess,
            obj_pc,
            latents,
            num_refine_steps = cfg.num_refine_steps,
        )

        print(f'------ number of generated grasps are: {len(generated_grasps)} ---------')
        save_file_path = os.path.join(grasper.output_grasps_dir, f'grasps_{meshobj.name}.npy')
        print(f'---- saving to {save_file_path} -----')

        save_dict = {
            'generated_grasps': generated_grasps,
            'generated_scores': generated_scores,
            'pcd': pc,
            'pcd_color': pc_colors,
            'obj_pcd': obj_pc,
            'env_state': env_state,
        }

        np.save(save_file_path, save_dict)

if __name__ == '__main__':
    main()