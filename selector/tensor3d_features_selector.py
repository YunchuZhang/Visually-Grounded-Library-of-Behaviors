import sys
import pickle
import imageio
from munch import Munch
import torch
import numpy as np

from selector.Selector import Selector
from pcp_utils.utils import Config
import pcp_utils

grnn_dir = pcp_utils.utils.get_grnn_dir()
sys.path.append(grnn_dir)

import model_nets
from backend import saverloader
import utils


class Tensor3DFeatSelector(Selector):
    class Config(Config):
        selector_name = "tensor3d feat selector"
        selector_model_path = ""
        model_name = None
        max_path_length = 110
        model_type = None
        model_config_file = None
        ckpt_dict = None
        cluster_pkl_file = None



    def __init__(self, config:Config):

        #if model_type == "mujoco_offline":

        self.selector_name = config.selector_name

        with open(config.model_config_file, "rb") as f:
            model_config = pickle.load(f)
        model_config = Munch(model_config)
        model_config.B = 1

        if "fix_crop" not in model_config:
            model_config.fix_crop = False
        if config.model_type == "mujoco_offline":
            self.model = model_nets.mujoco_offline.MujocoOffline(model_config)
        elif config.model_type == "mujoco_offline_metric":
            self.model = model_nets.mujoco_offline_metric.MujocoOfflineMetric(model_config)

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

        if config.model_type == "mujoco_offline":
            with open(config.cluster_pkl_file, 'rb') as f:
                self.cluster_centers = pickle.load(f)
            for center_name in self.cluster_centers:
                # key is string
                # center is B x H x W x D x C
                self.cluster_centers[center_name] = np.expand_dims(self.cluster_centers[center_name], axis=0)
        elif config.model_type == "mujoco_offline_metric":
            self.cluster_centers  = self.model.get_cluster_centers_dict()



    def compute_nearest_cluster(self, object_tensor):

        cluster_names = []
        cluster_scores = []
        for cluster_name in self.cluster_centers:
            cluster_center = self.cluster_centers[cluster_name]

            score = np.mean(object_tensor[0] * cluster_center)
            cluster_names.append(cluster_name)
            cluster_scores.append(score)

        argsort_id = np.argsort(cluster_scores)[::-1]
        sorted_cluster_name = [cluster_names[id_] for id_ in argsort_id]
        sorted_cluster_scores = [cluster_scores[id_] for id_ in argsort_id]

        return sorted_cluster_name, sorted_cluster_scores







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
        origin_T_camRs = feed["origin_T_camRs"]
        bbox_in_ref_cam = feed["bbox_in_ref_cam"]

        rgbs = np.transpose(rgbs, [0,3,1,2])
        rgbs = torch.from_numpy(rgbs).float()
        depths = torch.from_numpy(depths).float().unsqueeze(1)
        pix_T_cams = torch.from_numpy(pix_T_cams).float()
        origin_T_camXs = torch.from_numpy(origin_T_camXs).float()
        origin_T_camRs = torch.from_numpy(origin_T_camRs).float()
        bbox_in_ref_cam = torch.from_numpy(bbox_in_ref_cam).float()

        xyz_camXs = utils.geom.depth2pointcloud(depths, pix_T_cams, device=torch.device('cpu'))
        rgbs = rgbs / 255.
        rgbs = rgbs - 0.5


        feed['rgb_camXs'] = rgbs.unsqueeze(0)
        feed['xyz_camXs'] = xyz_camXs.unsqueeze(0)
        feed['pix_T_cams'] = pix_T_cams.unsqueeze(0)
        feed['origin_T_camXs'] = origin_T_camXs.unsqueeze(0)
        feed["origin_T_camRs"] = origin_T_camRs.unsqueeze(0)
        feed["bbox_in_ref_cam"] = bbox_in_ref_cam.unsqueeze(0)

        results = self.model.predict_forward(feed)
        # results: B X NOBJS X 32 X 32 X 32 X 32
        #imageio.imwrite("tmp/rgb_e.png", results['rgb_e'][0])
        #imageio.imwrite("tmp/rgb_camXs.png", feed['rgb_camXs'][0, 0].permute(1, 2, 0).detach().numpy())
        return results

    def compare_objects(self, env, obj):
        # output should be a dictionary
        # 'avg_reward': float32
        # 'success_rate': float32
        # using env to 
        raise NotImplementedError("Must be implemented in subclass.")




if __name__ == '__main__':
    import model_nets
    from backend import saverloader
    import pickle
    from munch import Munch

    config_file = "/Users/sfish0101/Documents/2020/Spring/quantized_policies/trained_models/checkpoints/MUJOCO_OFFLINE/train_viewpred_occ/config.pkl"
    model_file = "/Users/sfish0101/Documents/2020/Spring/quantized_policies/trained_models/checkpoints/MUJOCO_OFFLINE/train_viewpred_occ/model-40000.pth"
    with open(config_file, "rb") as f:
        config = pickle.load(f)
    config = Munch(config)

    with open("trained_models/feed.pkl", 'rb') as f:
        feed = pickle.load(f)


    model = model_nets.mujoco_offline.MujocoOffline(config)

    saveload_config = dict()
    saveload_config["total_init"] = True
    saveload_config["reset_iter"] = False
    saveload_config["loadname"] = dict()
    saveload_config["loadname"]["model"] = model_file
    saveload_config = Munch(saveload_config)



    saverloader = saverloader.SaverLoader(saveload_config, model, load_only=True)
    start_iter = saverloader.load_weights(optimizer=None)

    result = model.predict_forward(feed)

    imageio.imwrite("tmp/rgb_e.png", result['rgb_e'][0])
    imageio.imwrite("tmp/rgb_camXs.png", feed['rgb_camXs'][0, 0].permute(1, 2, 0).detach().numpy())
