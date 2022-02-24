import sys
import pickle
import imageio
from munch import Munch
import torch
import numpy as np

import pcp_utils
from pcp_utils.utils import Config

#grnn_dir = pcp_utils.utils.get_grnn_dir()
#sys.path.append(grnn_dir)
#
#import model_nets
#from backend import saverloader
#import utils



class Detector:

    def __init__(self, config:Config):
        self.config = config
 
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
        pass

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



