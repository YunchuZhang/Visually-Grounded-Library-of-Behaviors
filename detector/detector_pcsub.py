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

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")


def Pixels2Camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are B x H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    # there is no randomness here

    B, H, W = list(z.shape)

    fx = np.reshape(fx, [B,1,1])
    fy = np.reshape(fy, [B,1,1])
    x0 = np.reshape(x0, [B,1,1])
    y0 = np.reshape(y0, [B,1,1])

    # unproject
    EPS = 1e-6
    x = ((z+EPS)/fx)*(x-x0)
    y = ((z+EPS)/fy)*(y-y0)

    x = np.reshape(x, [B,-1])
    y = np.reshape(y, [B,-1])
    z = np.reshape(z, [B,-1])
    xyz = np.stack([x,y,z], dim=2)
    return xyz


def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def normalize_grid2D(grid_y, grid_x, Y, X):
    # make things in [-1,1]
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    return grid_y, grid_x

def meshgrid2D(B, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Y x X

    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, 1, X])
    grid_x = np.tile(grid_x, [B, Y, 1])

    if norm:
        grid_y, grid_x = normalize_grid2D(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = np.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x

def depth2pointcloud(z, pix_T_cam):
    B, C, H, W = list(z.shape)  # this is 1, 1, H, W
    y, x = meshgrid2D(B, H, W)
    z = np.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz

def apply_4x4(RT, XYZ):
    """
    RT: B x 4 x 4
    XYZ: B x N x 3
    """
    #RT = RT.to(XYZ.device)
    B, N, _ = list(XYZ.shape)
    ones = np.ones([B, N, 1])
    XYZ1 = np.concatenate([XYZ, ones], 2)
    XYZ1_t = np.transpose(XYZ1, 1, 2)
    # this is B x 4 x N
    XYZ2_t = np.matmul(RT, XYZ1_t)
    XYZ2 = np.transpose(XYZ2_t, 1, 2)
    XYZ2 = XYZ2[:,:,:3]
    return XYZ2


class DetectorPCSub:
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



        # setup camera
        self.camera_positions, self.camera_quats = pcp_utils.cameras.generate_new_cameras_hemisphere(radius=self.config.camera_radius,
            lookat_point=self.config.camera_lookat_pos, pitch=self.config.camera_pitch, yaw=self.config.camera_yaw, yaw_list=self.config.camera_yaw_list)
        self.n_cams = len(self.camera_positions)

        mujoco_T_adam = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        self.table_top = np.array(config.table_top)
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

        xyz_camXs = utils.geom.depth2pointcloud(depths, pix_T_cams)

        xyz_origin = utils.geom.apply_4x4(origin_T_camXs, xyz_camXs)
        xyz_origin = np.reshape(xyz_origin.numpy(), [-1, 3])

        
        # remove out of range thing
        table_top = np.expand_dims(self.table_top, axis=0)
        dist = np.max(np.abs(xyz_origin - table_top), axis=1)
        xyz_origin = xyz_origin[dist < 0.5]

        # remove table
        #print(xyz_origin.shape)
        #import matplotlib.pyplot as plt
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(xyz_origin[:, 0], xyz_origin[:, 1], xyz_origin[:, 2], c='b')
        #plt.savefig(f"tmp/pcs.png")

        # #import imageio
        #imageio.imwrite(f"tmp/rgb.png", np.concatenate([feed["rgb_camXs"][0],feed["rgb_camXs"][1]], axis=1))

        # #import ipdb; ipdb.set_trace()

        import copy
        xyz_origin_cp = copy.deepcopy(xyz_origin)

        xyz_origin = xyz_origin[xyz_origin[:,2] > (self.table_top[2] + 0.005)]
        xyz_origin = xyz_origin[xyz_origin[:,2] < 0.55]

        if xyz_origin.shape[0] == 0:
            xyz_origin = np.expand_dims(self.table_top, axis=0)
            xyz_origin[0, 2] = 0.5



        min_x = np.min(xyz_origin[:,0])
        max_x = np.max(xyz_origin[:,0])
        min_y = np.min(xyz_origin[:,1])
        max_y = np.max(xyz_origin[:,1])
        min_z = np.min(xyz_origin[:,2])
        max_z = np.max(xyz_origin[:,2])


        bounds = np.array([min_x, min_y, min_z, max_x, max_y, max_z])
        bounds = bounds.reshape([2,3])
        center = np.mean(bounds, axis=0)
        extents = bounds[1,:] - bounds[0,:]

        return bounds, center, extents, xyz_origin, xyz_origin_cp

    def detect_objects(self, env):
        images = self.render_images(env)

        #images = self.convert_to_adam(images)
        results = self.predict_forward(images)

        return results, images


    def compare_objects(self, env, obj):
        # output should be a dictionary
        # 'avg_reward': float32
        # 'success_rate': float32
        # using env to 
        raise NotImplementedError("Must be implemented in subclass.")



