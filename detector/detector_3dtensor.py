import sys
import pickle
import imageio
from munch import Munch
import torch
import numpy as np


from detector.detector import Detector
import pcp_utils
from pcp_utils.utils import Config

grnn_dir = pcp_utils.utils.get_grnn_dir()
sys.path.append(grnn_dir)

import model_nets
from backend import saverloader
import utils


class Detector3DTensor:
    class Config(Config):
        selector_name = "tensor3d feat selector"
        selector_model_path = ""
        model_name = None
        max_path_length = 110
        model_type = None
        model_config_file = None
        ckpt_dict = None
        cluster_pkl_file = None

        camera_img_height = 128
        camera_img_width = 128
        camera_fov_y = 45
        camera_radius =  0.65  # since we discussed we want cameras to be far
        camera_yaw = [0, 350, 36]  # 0-350 with gap of 36 in between
        camera_pitch = [20, 60, 20]  # [20, 40, 60]
        camera_yaw = [0, 350, 36]
        camera_yaw_list = None #[0, 60, 300]
        camera_lookat_pos = [1.3, 0.75, 0.45]  ## wanted to make it a bit higher
        table_top = [1.3, 0.75, 0.4]
        table_T_camR = [0, 0, 0.05]  # copied from fish config

    def __init__(self, config:Config):
        self.config = config


        with open(config.model_config_file, "rb") as f:
            model_config = pickle.load(f)
        model_config = Munch(model_config)
        model_config.B = 1

        self.model = model_nets.mujoco_offline.MujocoOffline(model_config)
        self.saveload_config = dict()
        self.saveload_config["total_init"] = True
        self.saveload_config["reset_iter"] = False

        self.saveload_config["loadname"] = dict()
        ckpt_dict = eval(config.ckpt_dict)
        for key in ckpt_dict:
            self.saveload_config["loadname"][key] = ckpt_dict[key]
        self.saveload_config = Munch(self.saveload_config)

        self.saverloader = saverloader.SaverLoader(self.saveload_config, self.model, load_only=True)
        start_iter = self.saverloader.load_weights(optimizer=None)


        # setup camera
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
        images['pix_T_cams'] = np.stack(pix_T_camXs, axis=0)
        images['origin_T_camXs'] = np.stack(origin_T_camXs, axis=0)

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


    def predict_forward(self, feed):
        """
        feed:
            rgb_camXs: B x nviews x 3 x 128 x 128
            depth_camXs: B x nviews x 1 x 128 x 128
            pix_T_cams: B x nviews x 3 x 3
            origin_T_camXs: B x nviews x 4 x 4
            origin_T_Rs: B x nviews x 4 x 4
            bbox_in_ref_cam: B x 8 x 3
            cluster_id: [string] * B
            set_num = 1
            set_name = "val"
        """
        rgbs = feed["rgb_camXs"]
        depths = feed["depth_camXs"]
        pix_T_cams = feed["pix_T_cams"]
        origin_T_camXs = feed["origin_T_camXs"]

        #import ipdb; ipdb.set_trace()
        NVIEWS, _, _ = origin_T_camXs.shape
        origin_T_camRs = np.tile(np.expand_dims(np.eye(4), axis=0), [NVIEWS, 1, 1])

        #bbox_in_ref_cam = feed["bbox_in_ref_cam"]

        rgbs = np.transpose(rgbs, [0,3,1,2])
        rgbs = torch.from_numpy(rgbs).float()
        depths = torch.from_numpy(depths).float().unsqueeze(1)
        pix_T_cams = torch.from_numpy(pix_T_cams).float()
        origin_T_camXs = torch.from_numpy(origin_T_camXs).float()
        origin_T_camRs = torch.from_numpy(origin_T_camRs).float()
        #bbox_in_ref_cam = torch.from_numpy(bbox_in_ref_cam).float()

        xyz_camXs = utils.geom.depth2pointcloud(depths, pix_T_cams, device=torch.device('cpu'))
        rgbs = rgbs / 255.
        rgbs = rgbs - 0.5

        feed["set_num"] = 1
        feed["set_name"] = "val"
        feed['rgb_camXs'] = rgbs.unsqueeze(0)
        feed['xyz_camXs'] = xyz_camXs.unsqueeze(0)
        feed['pix_T_cams'] = pix_T_cams.unsqueeze(0)
        feed['origin_T_camXs'] = origin_T_camXs.unsqueeze(0)
        feed["origin_T_camRs"] = origin_T_camRs.unsqueeze(0)
        #feed["bbox_in_ref_cam"] = bbox_in_ref_cam.unsqueeze(0)

        results = self.model.predict_forward_bbox_detector(feed)
        # results: B X NOBJS X 32 X 32 X 32 X 32
        #imageio.imwrite("tmp/rgb_e.png", results['rgb_e'][0])
        #imageio.imwrite("tmp/rgb_camXs.png", feed['rgb_camXs'][0, 0].permute(1, 2, 0).detach().numpy())
        return results

    def detect_objects(self, env):
        images = self.render_images(env)
        images = self.convert_to_adam(images)
        results = self.predict_forward(images)


    def compare_objects(self, env, obj):
        # output should be a dictionary
        # 'avg_reward': float32
        # 'success_rate': float32
        # using env to 
        raise NotImplementedError("Must be implemented in subclass.")



