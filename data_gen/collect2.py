import mujoco_py
import os
import sys
import click
import PIL.Image as Image
import imageio
import copy
import random
import gym
import matplotlib
import matplotlib.pyplot as plt
#import utils
import numpy as np
from attrdict import AttrDict
#from dm_control import mjcf
# from dm_control import viewer

from pcp_utils.parse_task_files import generate_integrated_xml

#os.sys.path.append('/Users/gspat/quant_codes/quantized_policies')
# from preprocessing import process_mesh
# from bounding_box import bounding_box as bb

cur_path = os.getcwd()
sys.path.append(cur_path)

import cv2
from itertools import combinations
import open3d as o3d
#from data_gen import generate_xml_camera


import pcp_utils
from pcp_utils.mesh_object import MeshObject
from pcp_utils.utils import Config


import trimesh

# make symbolic link of the mesh under quantize-gym/gym/envs/robotics/assets/
# because here it takes absolute path, so don't put it under assets/fetch
source_mesh_dir = pcp_utils.utils.get_mesh_dir()
gym_mesh_dir = os.path.join(pcp_utils.utils.get_gym_dir(), 'gym/envs/robotics/assets/meshes')

if not os.path.exists(gym_mesh_dir):
    os.symlink(source_mesh_dir, gym_mesh_dir,  target_is_directory=True)



class DataGenConfig(Config):
    # params for the environment
    # env_base_xml_file is expected to be under quantize-gym/gym/envs/robotics/assets
    # example: fetch/pick_ad_place_camera.xml
    
    env_name = None
    env_base_xml_file = "" #
    n_robot_dof = 8 # 4 means xyz, 6 means having rotations
    output_dir = "./data"

    #camera_parameters
    render = False
    camera_img_height = 128
    camera_img_width = 128
    camera_radius = 0.3
    camera_fov = 45
    camera_pitch = [20, 60, 20]
    camera_yaw = [0, 350, 36]
    camera_save_image = False
    camera_recon_scene = True
    camera_lookat_pos = [0, 0, 0.4]

    table_top = [1.3, 0.75, 0.4]
    table_T_camR = [0, 0, 0]

    # train_val_test split
    train_val_test_ratios = [0.5, 0.5, 0.5]

    # data
    randomize_color = False
    init_info = None
    # debug
    remove_site = False
    save_image = False
    visualize_bbox = False
    bbox_indicator = False
    sr_from_single_trial = False


class DataGen:
    def __init__(self, config_input, data_name, detector=None, run_name=""):
        config = DataGenConfig()
        config.update(config_input)

        self.env_base_xml_file = config.env_base_xml_file
        self.randomize_color = config.randomize_color
        self.run_name = run_name
        self.detector = detector
        self.sr_from_single_trial = config.sr_from_single_trial

        gym_path = pcp_utils.utils.get_gym_dir()
        self.env_xml_file_root = os.path.join(gym_path, "gym/envs/robotics/assets")
        self.env_name = config.env_name
        self.n_robot_dof = config.n_robot_dof
        self.randomize_color = config.randomize_color
        self.init_info = config.init_info
        self.bbox_indicator = config.bbox_indicator

        mesh_root_path = pcp_utils.utils.get_mesh_root_dir()
        self.obj_mesh_abs_root_path = mesh_root_path
        self.output_dir = os.path.join(config.output_dir, data_name)
        self.output_dir_exclud_root = data_name
        pcp_utils.utils.makedir(self.output_dir)
        self.config = config


        # mujoco: x right, y front; adam: x right, y down
        mujoco_T_adam = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        origin_T_camR_xpos = np.array(config.table_top, np.float32) + np.array(config.table_T_camR, np.float)

        origin_T_adam = np.zeros((4,4), dtype=np.float32)
        origin_T_adam[:3, :3] = mujoco_T_adam
        origin_T_adam[:3, 3] = origin_T_camR_xpos
        origin_T_adam[3,3] = 1
        self.origin_T_adam = origin_T_adam
        self.adam_T_origin = np.linalg.inv(self.origin_T_adam)


        table_T_camR = np.ones((4, 1), np.float32)
        table_T_camR[:3, 0] = config.table_T_camR
        self.table_top_R = np.dot(self.adam_T_origin, table_T_camR)

        # train val split
        self.train_ratio = config.train_val_test_ratios[0]
        self.val_ratio = config.train_val_test_ratios[1]
        self.test_ratio = config.train_val_test_ratios[2]
        assert self.train_ratio <= 1, "train ratio needs to be smaller than 1"
        assert self.train_ratio + self.test_ratio == 1, "train test ratios need to sum to 1"
        assert self.train_ratio >= self.val_ratio, "val ratio needs to be lower than train ratio"



        self.all_files = []

    def write_train_val_test_files(self):

        random.shuffle(self.all_files)
        num_files = len(self.all_files)
        num_train_files = int(num_files * self.train_ratio)

        if len(self.all_files[:num_train_files]) > 0:
            with open(self.output_dir + "_train.txt", 'w') as f:
                f.writelines("\n".join(self.all_files[:num_train_files]))

        if len(self.all_files[num_train_files:]) > 0:
            with open(self.output_dir + "_val.txt", 'w') as f:
                f.writelines("\n".join(self.all_files[num_train_files:]))

        if int(num_files * self.val_ratio) > 0:
            with open(self.output_dir + "_trainval.txt", 'w') as f:
                tmp_train_files = copy.deepcopy(self.all_files[:num_train_files])
                random.shuffle(tmp_train_files)
                num_trainval_files = int(num_files * self.val_ratio)
                f.writelines("\n".join(tmp_train_files[:num_trainval_files]))

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

    def convert_to_adam(self, images):
        origin_T_camXs = images['origin_T_camXs']
        camR_T_camXs = []
        for origin_T_camX in origin_T_camXs:
            camR_T_camX = np.dot(self.adam_T_origin, origin_T_camX)
            camR_T_camXs.append(camR_T_camX)

        camR_T_camXs = np.stack(camR_T_camXs, axis=0)
        images['origin_T_camXs'] = camR_T_camXs

    def scene_to_npz(self, mesh_obj, repeat_id):

        integrated_xml = generate_integrated_xml(self.env_base_xml_file, mesh_obj.obj_xml_file, randomize_color=self.randomize_color, scale=mesh_obj.scale, mass=mesh_obj.mass, euler=mesh_obj.euler,
                                                 prefix=self.run_name, remove_site=self.config.remove_site)
        #integrated_xml = "fetch/None_generated_env.xml"
        env = gym.make(self.env_name, xml_path=integrated_xml, use_bbox_indicator=self.bbox_indicator,
            n_actions=self.n_robot_dof, init_info=self.init_info)
        obs = env.reset()

        if self.config.render:
            env.render()
        for _ in range(20):
            obsDataNew, reward, done, info = env.step(np.zeros(self.n_robot_dof))
            if self.config.render:
                env.render()

        
        # add cameras
        camera_positions, camera_quats = pcp_utils.cameras.generate_new_cameras_hemisphere(radius=self.config.camera_radius,
            lookat_point=self.config.camera_lookat_pos, pitch=self.config.camera_pitch, yaw=self.config.camera_yaw)

        images = pcp_utils.cameras.render_images(env, self.config.camera_img_height, self.config.camera_img_width, self.config.camera_fov, camera_positions, camera_quats, camera_name="ext_camera_0")
        self.convert_to_adam(images)
        image_ref = pcp_utils.cameras.render_image_from_camX(env, self.config.camera_img_height, self.config.camera_img_width, self.config.camera_fov, camera_name="ref_cam")
        self.convert_to_adam(image_ref)

        scene_name = mesh_obj.name
        if self.config.save_image:
            imageio.imwrite(f"tmp/img_ref.png", image_ref["rgb_camXs"][0])
            n_images = images["rgb_camXs"].shape[0]
            imageio.imwrite(f"tmp/{scene_name}_nviews.png", np.squeeze(np.concatenate(np.split(images["rgb_camXs"], n_images, axis=0), axis=2), axis=0))
            print(f"save image tmp/{scene_name}_nviews.png....")
            import ipdb; ipdb.set_trace()


        if not self.detector:
            object_xpos, object_xmat = env.get_object_pos("object0")
            composed_xmat = np.eye(4, dtype=np.float32)
            composed_xmat[:3, 3] = object_xpos
            composed_xmat[:3, :3] = np.reshape(object_xmat, [3, 3])
            composed_xmat = np.dot(self.adam_T_origin, composed_xmat)
            object_xpos_adam =  composed_xmat[:3, 3] #np.dot(self.adam_T_origin, pcp_utils.geom.pts_addone(np.reshape(object_xpos, [1, 3])).T)[:3, 0]
            object_xmat_adam =  composed_xmat[:3, :3]#np.dot(self.adam_T_origin[:3, :3], np.reshape(object_xmat, [3, 3]))
    
            bbox_points_from_mesh_adam = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(mesh_obj.obj_xml_file, object_xpos_adam, object_xmat_adam, scale=mesh_obj.scale, euler=mesh_obj.euler)

        else:
            object_xpos, object_xmat = env.get_object_pos("object0")
            bbox_points_from_mesh_gt = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(mesh_obj.obj_xml_file, object_xpos, object_xmat, scale=mesh_obj.scale, euler=mesh_obj.euler)
            results, _ = self.detector.detect_objects(env)
            bounds, center, extents, xyz_origin, xyz_origin_cp = results


            x_min = bounds[0,0]
            x_max = bounds[1,0]
            y_min = bounds[0,1]
            y_max = bounds[1,1]
            z_min = bounds[0,2]
            z_max = bounds[1,2]

            bbox_points_from_mesh = np.array([[x_min, y_min, z_min],
                      [x_max, y_min, z_min],
                      [x_min, y_min, z_max],
                      [x_max, y_min, z_max],
                      [x_min, y_max, z_min],
                      [x_max, y_max, z_min],
                      [x_min, y_max, z_max],
                      [x_max, y_max, z_max],
                      ])
            bbox_points_ones = np.concatenate([bbox_points_from_mesh, np.ones([8, 1])], axis=1).T
            bbox_point_adam = np.dot(self.adam_T_origin,  bbox_points_ones)[:3, :].T

            x_min = bbox_point_adam[0,0]
            x_max = bbox_point_adam[-1,0]
            y_min = bbox_point_adam[-1,1]
            y_max = bbox_point_adam[0,1]
            z_min = bbox_point_adam[0,2]
            z_max = bbox_point_adam[-1,2]
            bbox_points_from_mesh_adam = np.array([[x_min, y_min, z_min],
                      [x_max, y_min, z_min],
                      [x_min, y_min, z_max],
                      [x_max, y_min, z_max],
                      [x_min, y_max, z_min],
                      [x_max, y_max, z_min],
                      [x_min, y_max, z_max],
                      [x_max, y_max, z_max],
                      ])

            diff = np.linalg.norm(bbox_points_from_mesh_gt - bbox_points_from_mesh)
            if diff > 0.1:
                print("distance to gt is too high", diff)
            
        if self.config.visualize_bbox:
            ep_depths = images["depth_camXs"]
            ep_pix_T_cams = images["pix_T_cams"]
            ep_camR_T_camXs = images["origin_T_camXs"]


            _, xyz_camRs, _ = pcp_utils.np_vis.unproject_depth(ep_depths,
                ep_pix_T_cams,
                ep_camR_T_camXs,
                camR_T_origin = None, #np.linalg.inv(self.origin_T_adam),
                clip_radius=5.0,
                do_vis=False)

            all_xyz_camR = np.concatenate(xyz_camRs, axis=0)
            object_pcd = pcp_utils.np_vis.get_pcd_object(all_xyz_camR, clip_radius=2.0)

            # transform object xpos and xmat to the adam coordinate (x right, y downs)
            bbox_lineset_from_mesh_adam = pcp_utils.np_vis.make_lineset(bbox_points_from_mesh_adam)
            o3d.visualization.draw_geometries([object_pcd, bbox_lineset_from_mesh_adam])
            #o3d.visualization.draw_geometries([object_pcd, bbox_lineset_from_mesh])
            import ipdb; ipdb.set_trace()



        # create a dictionary to save the data and ship it !!
        save_dict = AttrDict()
        """
        last view is reference frame
        rgb_camXs: nviews x height x width x 3
        depth_camXs: nviews x height x width
        pix_T_cams: nviews x 3 x 3
        camR_T_camXs: nviews x 4 x 4
        bbox_camR: 8 x 3
        cluster_id: 'string'
        """
        save_dict.rgb_camXs = np.concatenate([images['rgb_camXs'], image_ref['rgb_camXs']], axis=0)
        save_dict.depth_camXs = np.concatenate([images['depth_camXs'], image_ref['depth_camXs']], axis=0)
        save_dict.pix_T_cams = np.concatenate([images['pix_T_cams'], image_ref['pix_T_cams']], axis=0)
        save_dict.camR_T_camXs = np.concatenate([images['origin_T_camXs'], image_ref['origin_T_camXs']], axis=0)
        save_dict.bbox_camR = bbox_points_from_mesh_adam
        save_dict.cluster_id = mesh_obj.cluster_id
        save_dict.success_rates_over_class = np.array([float(x) for x in mesh_obj.success_rates_over_class.split(" ")])


        if self.sr_from_single_trial:
            srs = np.array([float(x) for x in mesh_obj.success_rates_over_class.split(" ")])

            sr = [np.random.binomial(1, sr) for sr in srs]
            save_dict.success_rates_over_class

        vis_save_path = os.path.join(self.output_dir, f"visual_data_{scene_name}_r{repeat_id}.npy")
        np.save(vis_save_path, save_dict)
        self.all_files.append(os.path.join(self.output_dir_exclud_root, f"visual_data_{scene_name}_r{repeat_id}.npy"))

        print(f'---- done saving {vis_save_path} ----')


@click.command()
@click.argument("config_file")#config
@click.option("--object_config_file") #for the objects
@click.option("--data_name") #for the objects
@click.option("--run_name") #define run name to avoid generate_xml to overwrite

def main(config_file, object_config_file, data_name, run_name):
    random.seed(1)

    config = pcp_utils.utils.config_from_yaml_file(config_file)

    if "use_detector" in config and config["use_detector"]:
        detector_param = config["detector"]

        detector_class, detector_config = pcp_utils.utils.import_class_from_config(detector_param)
        detector = detector_class(detector_config)
    else:
        detector = None

    objs_config = pcp_utils.utils.config_from_yaml_file(object_config_file)
    objects_to_cluster = []
    for obj_name, obj_config in objs_config['objs'].items():
        updated_obj_config = MeshObject.Config().update(obj_config)
        obj = MeshObject(updated_obj_config, obj_name)
        objects_to_cluster.append(obj)
    print(f"find {len(objects_to_cluster)} objects to cluster")

    data_gen = DataGen(config, data_name=data_name, detector=detector, run_name=run_name)

    nmeshes = len(objects_to_cluster)
    for repeat_id in range(config.repeat):
        for mesh_id, meshobj in enumerate(objects_to_cluster):
            print(f'working for {meshobj.name}', f"============ run {repeat_id} /{config.repeat}, processing object {mesh_id}/{nmeshes} ============ ")
            #meshobj.xml_file
            data_gen.scene_to_npz(meshobj, repeat_id)

    # write the list train_file and test_file
    data_gen.write_train_val_test_files()


if __name__ == '__main__':
    main()