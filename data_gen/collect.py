import mujoco_py
import os
import sys
import click
import PIL.Image as Image
import imageio
import copy
import random
from dm_control import mjcf
# from dm_control import viewer
import matplotlib
import matplotlib.pyplot as plt
import utils
import numpy as np
from attrdict import AttrDict
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
    env_base_xml_file = "" #
    output_dir = "./data"

    #camera_parameters
    render = False
    camera_img_height = 128
    camera_img_width = 128
    camera_radius = 0.3
    camera_fov_y = 45
    camera_pitch = [20, 60, 20]
    camera_yaw = [0, 350, 36]
    camera_save_image = False
    camera_recon_scene = True
    camera_lookat_pos = [0, 0, 0.4]

    table_top = [0, 0, 0.4]
    table_T_camR = [0, 0, 0]

    # train_val_test split
    train_val_test_ratios = [0.5, 0.5, 0.5]


    # debug
    save_image = False
    visualize_bbox = False

class DataGen:
    def __init__(self, config_input, data_name):
        config = DataGenConfig()
        config.update(config_input)

        self.env_base_xml_file = config.env_base_xml_file

        gym_path = pcp_utils.utils.get_gym_dir()
        self.env_xml_file_root = os.path.join(gym_path, "gym/envs/robotics/assets")

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
        assert self.train_ratio < 1, "train ratio needs to be smaller than 1"
        assert self.train_ratio + self.test_ratio == 1, "train test ratios need to sum to 1"
        assert self.train_ratio >= self.val_ratio, "val ratio needs to be lower than train ratio"



        self.all_files = []

    def write_train_val_test_files(self):

        random.shuffle(self.all_files)
        num_files = len(self.all_files)
        num_train_files = int(num_files * self.train_ratio)

        with open(self.output_dir + "_train.txt", 'w') as f:
            f.writelines("\n".join(self.all_files[:num_train_files]))

        with open(self.output_dir + "_val.txt", 'w') as f:
            f.writelines("\n".join(self.all_files[num_train_files:]))

        with open(self.output_dir + "_trainval.txt", 'w') as f:
            tmp_train_files = copy.deepcopy(self.all_files[:num_train_files])
            random.shuffle(tmp_train_files)
            num_trainval_files = int(num_files * self.val_ratio)
            f.writelines("\n".join(tmp_train_files[:num_trainval_files]))

    def scene_to_npz(self, mesh_obj):

        integrated_xml = generate_integrated_xml(self.env_base_xml_file, mesh_obj.obj_xml_file, scale=mesh_obj.scale)
        abs_integrated_xml_path = os.path.join(self.env_xml_file_root, integrated_xml)
        visual_mjcf = mjcf.from_path(abs_integrated_xml_path) #cannot have "include" in xml when loading with mjcf

        # camera pose is related the object pose?
        # for visualization
        if self.config.render:
            model = mujoco_py.load_model_from_path(abs_integrated_xml_path)
            sim = mujoco_py.MjSim(model)
            viewer = mujoco_py.MjViewer(sim)
            for _ in range(5000):
                sim.step()
            for _ in range(20):
                viewer.render()

        print(mesh_obj.name, self.config.camera_lookat_pos)
        
        # add cameras
        camera_positions, camera_quats = pcp_utils.cameras.generate_new_cameras_hemisphere(radius=self.config.camera_radius,
            lookat_point=self.config.camera_lookat_pos, pitch=self.config.camera_pitch, yaw=self.config.camera_yaw)


        # add all the cameras to the environtmnet
        for i, (pos, quat) in enumerate(zip(camera_positions, camera_quats)):
            visual_mjcf.worldbody.add('camera', name=f'vis_cam:{i}', pos=pos, quat=quat, fovy=self.config.camera_fov_y)
        visual_mjcf.worldbody.add('camera', name='ref_cam', pos = [0,-0.3,0.5], zaxis = [0,-1,0], fovy=self.config.camera_fov_y)


        # start simulate
        physics = mjcf.Physics.from_mjcf_model(visual_mjcf)
        physics.forward()

        for step_id in range(2000):
            physics.step()
        object_xpos = physics.data.xpos[physics.model.name2id('object0', 'body')]
        #print(f'object final location', object_xpos)
        object_xmat = physics.data.xmat[physics.model.name2id('object0', 'body')]
        #print(f'object final qpos', object_xmat)


        ep_imgs = list()
        ep_depths = list()
        ep_pix_T_cams = list()
        ep_camR_T_camXs = list()

        for i in range(len(camera_positions)):
            img = physics.render(self.config.camera_img_height, self.config.camera_img_width, camera_id=f'vis_cam:{i}')
            depth = physics.render(self.config.camera_img_height, self.config.camera_img_width, camera_id=f'vis_cam:{i}', depth=True)

            assert img.shape[0] == self.config.camera_img_height, "color img height is wrong"
            assert img.shape[1] == self.config.camera_img_width, "color img width is wrong"
            assert depth.shape[0] == self.config.camera_img_height, "depth img height is wrong"
            assert depth.shape[1] == self.config.camera_img_width, "depth img width is wrong"

            #if self.config.save_image:
            #    imageio.imwrite(f"tmp/img_{i}.png", img)
            #    imageio.imwrite(f"tmp/depth_{i}.png", np.minimum(depth, 5))


            pix_T_cams = pcp_utils.cameras.get_intrinsics(self.config.camera_fov_y, self.config.camera_img_width, self.config.camera_img_height)
            origin_T_camX = pcp_utils.cameras.dm_get_extrinsics(physics, physics.model.name2id(f'vis_cam:{i}', 'camera'))
            camR_T_camX = np.dot(self.adam_T_origin, origin_T_camX)
            
            # camR should be the top of the table
            #camR_T_camX =


            ep_imgs.append(img)
            ep_depths.append(depth)
            ep_pix_T_cams.append(pix_T_cams)
            # this extrinsics is in mujoco frame
            ep_camR_T_camXs.append(camR_T_camX)


        img = physics.render(self.config.camera_img_height, self.config.camera_img_width, camera_id='ref_cam')
        depth = physics.render(self.config.camera_img_height, self.config.camera_img_width, camera_id='ref_cam', depth=True)
        pix_T_cams = pcp_utils.cameras.get_intrinsics(self.config.camera_fov_y, self.config.camera_img_width, self.config.camera_img_height)
        origin_T_camX = pcp_utils.cameras.dm_get_extrinsics(physics, physics.model.name2id('ref_cam', 'camera'))
        camR_T_camX = np.dot(self.adam_T_origin, origin_T_camX)

        if self.config.save_image:
            imageio.imwrite(f"tmp/img_ref.png", img)

        ep_imgs.append(img)
        ep_depths.append(depth)
        ep_pix_T_cams.append(pix_T_cams)
        ep_camR_T_camXs.append(camR_T_camX)

        composed_xmat = np.eye(4, dtype=np.float32)
        composed_xmat[:3, 3] = object_xpos
        composed_xmat[:3, :3] = np.reshape(object_xmat, [3, 3])
        composed_xmat = np.dot(self.adam_T_origin, composed_xmat)
        object_xpos_adam =  composed_xmat[:3, 3] #np.dot(self.adam_T_origin, pcp_utils.geom.pts_addone(np.reshape(object_xpos, [1, 3])).T)[:3, 0]
        object_xmat_adam =  composed_xmat[:3, :3]#np.dot(self.adam_T_origin[:3, :3], np.reshape(object_xmat, [3, 3]))


        bbox_points_from_mesh_adam = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(mesh_obj.obj_xml_file, object_xpos_adam, object_xmat_adam, scale=mesh_obj.scale)

        if self.config.visualize_bbox:
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
        save_dict.rgb_camXs = np.stack(ep_imgs)
        save_dict.depth_camXs = np.stack(ep_depths)
        save_dict.pix_T_cams = np.stack(ep_pix_T_cams)
        save_dict.camR_T_camXs = np.stack(ep_camR_T_camXs)
        save_dict.bbox_camR = bbox_points_from_mesh_adam
        save_dict.cluster_id = mesh_obj.cluster_id


        scene_name = mesh_obj.name
        if self.config.save_image:
            imageio.imwrite(f"tmp/{scene_name}_nviews.png", np.concatenate(ep_imgs, axis=1))
        vis_save_path = os.path.join(self.output_dir, f"visual_data_{scene_name}.npy")
        np.save(vis_save_path, save_dict)

        self.all_files.append(os.path.join(self.output_dir_exclud_root, f"visual_data_{scene_name}.npy"))

        print('---- done ----')


@click.command()
@click.argument("config_file")#config
@click.option("--object_config_file") #for the objects
@click.option("--data_name") #for the objects

def main(config_file, object_config_file, data_name):
    random.seed(1)

    config = pcp_utils.utils.config_from_yaml_file(config_file)

    objs_config = pcp_utils.utils.config_from_yaml_file(object_config_file)
    objects_to_cluster = []
    for obj_name, obj_config in objs_config['objs'].items():
        updated_obj_config = MeshObject.Config().update(obj_config)
        obj = MeshObject(updated_obj_config, obj_name)
        objects_to_cluster.append(obj)
    print(f"find {len(objects_to_cluster)} objects to cluster")

    data_gen = DataGen(config, data_name=data_name)

    for meshobj in objects_to_cluster:
        print(f'working for {meshobj.name}')
        #meshobj.xml_file
        data_gen.scene_to_npz(meshobj)

    # write the list train_file and test_file
    data_gen.write_train_val_test_files()



if __name__ == '__main__':
    main()