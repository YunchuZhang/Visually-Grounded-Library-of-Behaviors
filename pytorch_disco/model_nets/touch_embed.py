import torch
import transformations
from backend import saverloader
import torch.nn as nn
import hyperparams as hyp
import numpy as np

from model_base import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.embnet3D import EmbNet3D
from nets.moc_net import MOCTrainingTouch
from archs import VGGNet
from archs import bottle3D

import torch.nn.functional as F

from utils_basic import *
import utils_vox
import utils_samp
import utils_geom
import utils_improc
import utils_misc
from utils_eval import compute_precision, plot_nearest_neighbours

# import open3d as o3d

np.set_printoptions(precision=5)
np.random.seed(0)
torch.manual_seed(0)

class TOUCH_EMBED(Model):
    """class model has the go function which basically runs the forward here
       Need to override the infer function which creates a self.model which is
       nn.Module function
    """
    def infer(self):
        print('--- Building inference graph ---')
        self.model = TouchEmbed2D()  # go of model_bases calls this
        self.model.to(self.device)

class TouchEmbed2D(nn.Module):
    def __init__(self):
        """The idea is here the featnet would be frozen, and using some X number
           of images it will form the features for the object, which will be used
           to get the closeby embeddings to learn metric learning
        """
        super(TouchEmbed2D, self).__init__()
        if hyp.do_feat:
            print('using the visual feat net to generate visual feature tensor')
            self.featnet = FeatNet(input_dim=4)  # passing occXs and occXs * unpXs

        if hyp.do_touch_feat:
            print('using 2d backbone network to generate features from sensor depth image')
            self.backbone_2D = VGGNet.Feat2d(touch_emb_dim=hyp.feat_dim, do_bn=hyp.do_bn)
            # if I need to add the 3d encoder decoder I need to add the next line
            # self.touch_featnet = FeatNet(input_dim=1)  # just passing occRs here

        if hyp.do_touch_forward:
            # now here I need to pass this through the bottle3d architecture to predict
            # 1-d vectors
            print('using context net to turn 3d context grid into 1d feature tensor')
            self.context_net = bottle3D.Bottle3D(in_channel=hyp.feat_dim,\
                pred_dim=hyp.feat_dim)

        if hyp.do_touch_occ:
            print('this should not be turned on')
            from IPython import embed; embed()
            self.touch_occnet = OccNet()

        # if hyp.do_touch_embML:
        #     self.touch_embnet3D = EmbNet3D()  # metric learning for making touch feature tensor same as visual feature tensor

        if hyp.do_freeze_feat:
            print('freezing visual features')
            self.featnet = self.featnet.eval()
            assert self.featnet.training == False, "since I am not training FeatNet it should be false"

        if hyp.do_freeze_touch_feat:
            print('freezing backbone_2D')
            self.backbone_2D = self.backbone_2D.eval()

        if hyp.do_freeze_touch_forward:
            print('freezing context net')
            self.context_net = self.context_net.eval()

        # hyperparams for embedding training not really needed as of now
        if hyp.do_touch_embML:
            print('Instantiating Contrastive ML loss')
            self.num_pos_samples = hyp.emb_3D_num_samples
            self.batch_k = 2
            self.n_negatives = 1
            assert (self.num_pos_samples > 0)
            self.sampler = utils_misc.DistanceWeightedSampling(batch_k=self.batch_k, normalize=False,
                num_neg_samples=self.n_negatives)
            self.criterion = utils_misc.MarginLoss()
            self.beta = 1.2

        if hyp.do_moc or hyp.do_eval_recall:
            print('Instantiating MOC net')
            self.key_touch_featnet = VGGNet.Feat2d(touch_emb_dim=hyp.feat_dim,\
                do_bn=hyp.do_bn)
            key_weights = self.backbone_2D.state_dict()
            self.key_touch_featnet.load_state_dict(key_weights)

            self.key_context_net = bottle3D.Bottle3D(in_channel=hyp.feat_dim,\
                pred_dim=hyp.feat_dim)
            key_context_weights = self.context_net.state_dict()
            self.key_context_net.load_state_dict(key_context_weights)

            # check that the two networks indeed have the same weights
            p1 = get_params(self.backbone_2D)
            p2 = get_params(self.key_touch_featnet)
            assert check_equal(p1, p2),\
                "initially both the touch networks should have same weights"

            cp1 = get_params(self.context_net)
            cp2 = get_params(self.key_context_net)
            assert check_equal(cp1, cp2),\
                "initially both the context networks should have same weights"

            self.moc_ml_net = MOCTrainingTouch(dict_len=hyp.dict_len,\
                num_neg_samples=hyp.num_neg_samples)

    def forward(self, feed, moc_init_done=False, debug=False):
        summ_writer = utils_improc.Summ_writer(
            writer = feed['writer'],
            global_step = feed['global_step'],
            set_name= feed['set_name'],
            fps=8)

        writer = feed['writer']
        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        ### ... All things sensor ... ###
        sensor_rgbs = feed['sensor_imgs']
        sensor_depths = feed['sensor_depths']
        center_sensor_H, center_sensor_W = sensor_depths[0][0].shape[-1] // 2, sensor_depths[0][0].shape[-2] // 2
        ### ... All things sensor end ... ###

        # 1. Form the memory tensor using the feat net and visual images.
        # check what all do you need for this and create only those things

        ##  .... Input images ....  ##
        rgb_camRs = feed['rgb_camRs']
        rgb_camXs = feed['rgb_camXs']
        ##  .... Input images end ....  ##

        ## ... Hyperparams ... ##
        B, H, W, V, S = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S
        __p = lambda x: pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)
        PH, PW = hyp.PH, hyp.PW
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        ## ... Hyperparams end ... ##

        ## .... VISUAL TRANSFORMS BEGIN .... ##
        pix_T_cams = feed['pix_T_cams']
        pix_T_cams_ = __p(pix_T_cams)
        origin_T_camRs = feed['origin_T_camRs']
        origin_T_camRs_ = __p(origin_T_camRs)
        origin_T_camXs = feed['origin_T_camXs']
        origin_T_camXs_ = __p(origin_T_camXs)
        camRs_T_camXs_ = torch.matmul(utils_geom.safe_inverse(
            origin_T_camRs_), origin_T_camXs_)
        camXs_T_camRs_ = utils_geom.safe_inverse(camRs_T_camXs_)
        camRs_T_camXs = __u(camRs_T_camXs_)
        camXs_T_camRs = __u(camXs_T_camRs_)
        pix_T_cams_ = utils_geom.pack_intrinsics(pix_T_cams_[:, 0, 0], pix_T_cams_[:, 1, 1], pix_T_cams_[:, 0, 2],
            pix_T_cams_[:, 1, 2])
        pix_T_camRs_ = torch.matmul(pix_T_cams_, camXs_T_camRs_)
        pix_T_camRs = __u(pix_T_camRs_)
        ## ... VISUAL TRANSFORMS END ... ##

        ## ... SENSOR TRANSFORMS BEGIN ... ##
        sensor_origin_T_camXs = feed['sensor_extrinsics']
        sensor_origin_T_camXs_ = __p(sensor_origin_T_camXs)
        sensor_origin_T_camRs = feed['sensor_origin_T_camRs']
        sensor_origin_T_camRs_ = __p(sensor_origin_T_camRs)
        sensor_camRs_T_origin_ = utils_geom.safe_inverse(sensor_origin_T_camRs_)

        sensor_camRs_T_camXs_ = torch.matmul(utils_geom.safe_inverse(
            sensor_origin_T_camRs_), sensor_origin_T_camXs_)
        sensor_camXs_T_camRs_ = utils_geom.safe_inverse(sensor_camRs_T_camXs_)

        sensor_camRs_T_camXs = __u(sensor_camRs_T_camXs_)
        sensor_camXs_T_camRs = __u(sensor_camXs_T_camRs_)

        sensor_pix_T_cams = feed['sensor_intrinsics']
        sensor_pix_T_cams_ = __p(sensor_pix_T_cams)
        sensor_pix_T_cams_ = utils_geom.pack_intrinsics(sensor_pix_T_cams_[:, 0, 0], sensor_pix_T_cams_[:, 1, 1],
            sensor_pix_T_cams_[:, 0, 2], sensor_pix_T_cams_[:, 1, 2])
        sensor_pix_T_camRs_ = torch.matmul(sensor_pix_T_cams_, sensor_camXs_T_camRs_)
        sensor_pix_T_camRs = __u(sensor_pix_T_camRs_)
        ## .... SENSOR TRANSFORMS END .... ##

        ## .... Visual Input point clouds .... ##
        xyz_camXs = feed['xyz_camXs']
        xyz_camXs_ = __p(xyz_camXs)
        xyz_camRs_ = utils_geom.apply_4x4(camRs_T_camXs_, xyz_camXs_)  # (40, 4, 4) (B*S, N, 3)
        xyz_camRs = __u(xyz_camRs_)
        assert all([torch.allclose(xyz_camR, inp_xyz_camR) for xyz_camR, inp_xyz_camR in zip(
            xyz_camRs, feed['xyz_camRs']
        )]), "computation of xyz_camR here and those computed in input do not match"
        ## .... Visual Input point clouds end .... ##

        ## ... Sensor input point clouds ... ##
        sensor_xyz_camXs = feed['sensor_xyz_camXs']
        sensor_xyz_camXs_ = __p(sensor_xyz_camXs)
        sensor_xyz_camRs_ = utils_geom.apply_4x4(sensor_camRs_T_camXs_, sensor_xyz_camXs_)
        sensor_xyz_camRs = __u(sensor_xyz_camRs_)
        assert all([torch.allclose(sensor_xyz, inp_sensor_xyz) for sensor_xyz, inp_sensor_xyz in zip(
            sensor_xyz_camRs, feed['sensor_xyz_camRs']
        )]), "the sensor_xyz_camRs computed in forward do not match those computed in input"

        ## ... visual occupancy computation voxelize the pointcloud from above ... ##
        occRs_ = utils_vox.voxelize_xyz(xyz_camRs_, Z, Y, X)
        occXs_ = utils_vox.voxelize_xyz(xyz_camXs_, Z, Y, X)
        occRs_half_ = utils_vox.voxelize_xyz(xyz_camRs_, Z2, Y2, X2)
        occXs_half_ = utils_vox.voxelize_xyz(xyz_camXs_, Z2, Y2, X2)
        ## ... visual occupancy computation end ... NOTE: no unpacking ##

        ## .. visual occupancy computation for sensor inputs .. ##
        sensor_occRs_ = utils_vox.voxelize_xyz(sensor_xyz_camRs_, Z, Y, X)
        sensor_occXs_ = utils_vox.voxelize_xyz(sensor_xyz_camXs_, Z, Y, X)
        sensor_occRs_half_ = utils_vox.voxelize_xyz(sensor_xyz_camRs_, Z2, Y2, X2)
        sensor_occXs_half_ = utils_vox.voxelize_xyz(sensor_xyz_camXs_, Z2, Y2, X2)

        ## ... unproject rgb images ... ##
        unpRs_ = utils_vox.unproject_rgb_to_mem(__p(rgb_camXs), Z, Y, X, pix_T_camRs_)
        unpXs_ = utils_vox.unproject_rgb_to_mem(__p(rgb_camXs), Z, Y, X, pix_T_cams_)
        ## ... unproject rgb finish ... NOTE: no unpacking ##

        ## ... Make depth images ... ##
        depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(pix_T_cams_, xyz_camXs_, H, W)
        dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, pix_T_cams_)
        dense_xyz_camRs_ = utils_geom.apply_4x4(camRs_T_camXs_, dense_xyz_camXs_)
        inbound_camXs_ = utils_vox.get_inbounds(dense_xyz_camRs_, Z, Y, X).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])
        valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)
        ## ... Make depth images ... ##

        ## ... Make sensor depth images ... ##
        sensor_depth_camXs_, sensor_valid_camXs_ = utils_geom.create_depth_image(sensor_pix_T_cams_,
            sensor_xyz_camXs_, H, W)
        sensor_dense_xyz_camXs_ = utils_geom.depth2pointcloud(sensor_depth_camXs_, sensor_pix_T_cams_)
        sensor_dense_xyz_camRs_ = utils_geom.apply_4x4(sensor_camRs_T_camXs_, sensor_dense_xyz_camXs_)
        sensor_inbound_camXs_ = utils_vox.get_inbounds(sensor_dense_xyz_camRs_, Z, Y, X).float()
        sensor_inbound_camXs_ = torch.reshape(sensor_inbound_camXs_, [B*hyp.sensor_S, 1, H, W])
        sensor_valid_camXs = __u(sensor_valid_camXs_) * __u(sensor_inbound_camXs_)
        ### .. Done making sensor depth images .. ##

        ### ... Sanity check ... Write to tensorboard ... ###
        summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(__u(depth_camXs_), dim=1))
        summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
        summ_writer.summ_occs('3d_inputs/occXs', torch.unbind(__u(occXs_), dim=1), reduce_axes=[2])
        summ_writer.summ_unps('3d_inputs/unpXs', torch.unbind(__u(unpXs_), dim=1),\
            torch.unbind(__u(occXs_), dim=1))

        # A different approach for viewing occRs of sensors
        sensor_occRs = __u(sensor_occRs_)
        vis_sensor_occRs = torch.max(sensor_occRs, dim=1, keepdim=True)[0]
        # summ_writer.summ_occs('3d_inputs/sensor_occXs', torch.unbind(__u(sensor_occXs_), dim=1),
        #     reduce_axes=[2])
        summ_writer.summ_occs('3d_inputs/sensor_occRs', torch.unbind(vis_sensor_occRs, dim=1), reduce_axes=[2])

        ### ... code for visualizing sensor depths and sensor rgbs ... ###
        # summ_writer.summ_oneds('2D_inputs/depths_sensor', torch.unbind(sensor_depths, dim=1))
        # summ_writer.summ_rgbs('2D_inputs/rgbs_sensor', torch.unbind(sensor_rgbs, dim=1))
        # summ_writer.summ_oneds('2D_inputs/validXs_sensor', torch.unbind(sensor_valid_camXs, dim=1))

        if summ_writer.save_this:
            unpRs_ = utils_vox.unproject_rgb_to_mem(__p(rgb_camXs), Z, Y, X, matmul2(pix_T_cams_, camXs_T_camRs_))
            unpRs = __u(unpRs_)
            occRs_ = utils_vox.voxelize_xyz(xyz_camRs_, Z, Y, X)
            summ_writer.summ_occs('3d_inputs/occRs', torch.unbind(__u(occRs_), dim=1), reduce_axes=[2])
            summ_writer.summ_unps('3d_inputs/unpRs', torch.unbind(unpRs, dim=1),\
                torch.unbind(__u(occRs_), dim=1))
        ### ... Sanity check ... Writing to tensoboard complete ... ###
        results = list()

        mask_ = None
        ### ... Visual featnet part .... ###
        if hyp.do_feat:
            featXs_input = torch.cat([__u(occXs_), __u(occXs_)*__u(unpXs_)], dim=2)  # B, S, 4, H, W, D
            featXs_input_ = __p(featXs_input)

            freeXs_ = utils_vox.get_freespace(__p(xyz_camXs), occXs_half_)
            freeXs = __u(freeXs_)
            visXs = torch.clamp(__u(occXs_half_) + freeXs, 0.0, 1.0)

            if type(mask_) != type(None):
                assert(list(mask_.shape)[2:5] == list(featXs_input.shape)[2:5])
            featXs_, validXs_, _ = self.featnet(featXs_input_, summ_writer, mask=occXs_)
            # total_loss += feat_loss  # Note no need of loss

            validXs, featXs = __u(validXs_), __u(featXs_) # unpacked into B, S, C, D, H, W
            # bring everything to ref_frame
            validRs = utils_vox.apply_4x4_to_voxs(camRs_T_camXs, validXs)
            visRs = utils_vox.apply_4x4_to_voxs(camRs_T_camXs, visXs)
            featRs = utils_vox.apply_4x4_to_voxs(camRs_T_camXs, featXs)  # This is now in memory coordinates

            emb3D_e = torch.mean(featRs[:, 1:], dim=1)  # context, or the features of the scene
            emb3D_g = featRs[:, 0]  # this is to predict, basically I will pass emb3D_e as input and hope to predict emb3D_g
            vis3D_e = torch.max(validRs[:, 1:], dim=1)[0] * torch.max(visRs[:, 1:], dim=1)[0]
            vis3D_g = validRs[:, 0] * visRs[:, 0]

            #### ... I do not think I need this ... ####
            results = {}
        #     # if hyp.do_eval_recall:
        #     #     results['emb3D_e'] = emb3D_e
        #     #     results['emb3D_g'] = emb3D_g
        #     #### ... Check if you need the above

            summ_writer.summ_feats('3D_feats/featXs_input', torch.unbind(featXs_input, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featRs_output', torch.unbind(featRs, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/validRs', torch.unbind(validRs, dim=1), pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_e', vis3D_e, pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_g', vis3D_g, pca=False)

            # I need to aggregate the features and detach to prevent the backward pass on featnet
            featRs = torch.mean(featRs, dim=1)
            featRs = featRs.detach()
            #  ... HERE I HAVE THE VISUAL FEATURE TENSOR ... WHICH IS MADE USING 5 EVENLY SPACED VIEWS #

        # FOR THE TOUCH PART, I HAVE THE OCC and THE AIM IS TO PREDICT FEATURES FROM THEM #
        if hyp.do_touch_feat:
            # 1. Pass all the sensor depth images through the backbone network
            input_sensor_depths = __p(sensor_depths)
            sensor_features_ = self.backbone_2D(input_sensor_depths)

            # should normalize these feature tensors
            sensor_features_ = l2_normalize(sensor_features_, dim=1)

            sensor_features = __u(sensor_features_)
            assert torch.allclose(torch.norm(sensor_features_, dim=1), torch.Tensor([1.0]).cuda()),\
                "normalization has no effect on you huh."

            if hyp.do_eval_recall:
              results['sensor_features'] = sensor_features_
              results['sensor_depths'] = input_sensor_depths
              results['object_img'] = rgb_camRs
              results['sensor_imgs'] = __p(sensor_rgbs)

            # if moco is used do the same procedure as above but with a different network #
            if hyp.do_moc or hyp.do_eval_recall:
                # 1. Pass all the sensor depth images through the key network
                key_input_sensor_depths = copy.deepcopy(__p(sensor_depths)) # bx1024x1x16x16->(2048x1x16x16)
                self.key_touch_featnet.eval()
                with torch.no_grad():
                    key_sensor_features_ = self.key_touch_featnet(key_input_sensor_depths)

                key_sensor_features_ = l2_normalize(key_sensor_features_, dim=1)
                key_sensor_features = __u(key_sensor_features_)
                assert torch.allclose(torch.norm(key_sensor_features_, dim=1), torch.Tensor([1.0]).cuda()),\
                    "normalization has no effect on you huh."

        # doing the same procedure for moco but with a different network end #

        # do you want to do metric learning voxel point based using visual features and sensor features
        if hyp.do_touch_embML and not hyp.do_touch_forward:
            # trial 1: I do not pass the above obtained features through some encoder decoder in 3d
            # So compute the location is ref_frame which the center of these depth images will occupy
            # at all of these locations I will sample the from the visual tensor. It forms the positive pairs
            # negatives are simply everything except the positive
            sensor_depths_centers_x = center_sensor_W * torch.ones((hyp.B, hyp.sensor_S))
            sensor_depths_centers_x = sensor_depths_centers_x.cuda()
            sensor_depths_centers_y = center_sensor_H * torch.ones((hyp.B, hyp.sensor_S))
            sensor_depths_centers_y = sensor_depths_centers_y.cuda()
            sensor_depths_centers_z = sensor_depths[:, :, 0, center_sensor_H, center_sensor_W]

            # Next use Pixels2Camera to unproject all of these together.
            # merge the batch and the sequence dimension
            sensor_depths_centers_x = sensor_depths_centers_x.reshape(-1, 1, 1)  # BxHxW as required by Pixels2Camera
            sensor_depths_centers_y = sensor_depths_centers_y.reshape(-1, 1, 1)
            sensor_depths_centers_z = sensor_depths_centers_z.reshape(-1, 1, 1)

            fx, fy, x0, y0 = utils_geom.split_intrinsics(sensor_pix_T_cams_)
            sensor_depths_centers_in_camXs_ = utils_geom.Pixels2Camera(sensor_depths_centers_x, sensor_depths_centers_y,
                sensor_depths_centers_z, fx, fy, x0, y0)

            # finally use apply4x4 to get the locations in ref_cam
            sensor_depths_centers_in_ref_cam_ = utils_geom.apply_4x4(sensor_camRs_T_camXs_, sensor_depths_centers_in_camXs_)

            # NOTE: convert them to memory coordinates, the name is xyz so I presume it returns xyz but talk to ADAM
            sensor_depths_centers_in_mem_ = utils_vox.Ref2Mem(sensor_depths_centers_in_ref_cam_, Z2, Y2, X2)
            sensor_depths_centers_in_mem = sensor_depths_centers_in_mem_.reshape(hyp.B, hyp.sensor_S, -1)

            if debug:
                print('assert that you are not entering here')
                from IPython import embed; embed()
                # form a (0, 1) volume here at these locations and see if it resembles a cup
                dim1 = X2 * Y2 * Z2
                dim2 = X2 * Y2
                dim3 = X2
                binary_voxel_grid = torch.zeros((hyp.B, X2, Y2, Z2))
                # NOTE: Z is the leading dimension
                rounded_idxs = torch.round(sensor_depths_centers_in_mem)
                flat_idxs = dim2 * rounded_idxs[0, :, 0] + dim3 * rounded_idxs[0, :, 1] + rounded_idxs[0, :, 2]
                flat_idxs1 = dim2 * rounded_idxs[1, :, 0] + dim3 * rounded_idxs[1, :, 1] + rounded_idxs[1, :, 2]
                flat_idxs1 = flat_idxs1 + dim1
                flat_idxs1 = flat_idxs1.long()
                flat_idxs = flat_idxs.long()

                flattened_grid = binary_voxel_grid.flatten()
                flattened_grid[flat_idxs] = 1.
                flattened_grid[flat_idxs1] = 1.

                binary_voxel_grid = flattened_grid.view(B, X2, Y2, Z2)

                assert binary_voxel_grid[0].sum() == len(torch.unique(flat_idxs)), "some indexes are missed here"
                assert binary_voxel_grid[1].sum() == len(torch.unique(flat_idxs1)), "some indexes are missed here"

                # o3d.io.write_voxel_grid("forward_pass_save/grid0.ply", binary_voxel_grid[0])
                # o3d.io.write_voxel_grid("forward_pass_save/grid1.ply", binary_voxel_grid[0])
                # need to save these voxels
                save_voxel(binary_voxel_grid[0].cpu().numpy(), "forward_pass_save/grid0.binvox")
                save_voxel(binary_voxel_grid[1].cpu().numpy(), "forward_pass_save/grid1.binvox")
                from IPython import embed; embed()

            # use grid sample to get the visual touch tensor at these locations, NOTE: visual tensor features shape is (B, C, N)
            visual_tensor_features = utils_samp.bilinear_sample3D(featRs, sensor_depths_centers_in_mem[:, :, 0],
                sensor_depths_centers_in_mem[:, :, 1], sensor_depths_centers_in_mem[:, :, 2])
            visual_feature_tensor = visual_tensor_features.permute(0, 2, 1)
            # pack it
            visual_feature_tensor_ = __p(visual_feature_tensor)
            C = list(visual_feature_tensor.shape)[-1]
            print('C=', C)

            # do the metric learning this is the same as before.
            # the code is basically copied from embnet3d.py but some changes are being made very minor
            emb_vec = torch.stack((sensor_features_, visual_feature_tensor_), dim=1).view(B*self.num_samples*self.batch_k, C)
            y = torch.stack([torch.range(0,self.num_samples*B-1), torch.range(0,self.num_samples*B-1)], dim=1).view(self.num_samples*B*self.batch_k)
            a_indices, anchors, positives, negatives, _ = self.sampler(emb_vec)

            # I need to write my own version of margin loss since the negatives and anchors may not be same dim
            d_ap = torch.sqrt(torch.sum((positives - anchors)**2, dim=1) + 1e-8)
            pos_loss = torch.clamp(d_ap - beta + self._margin, min=0.0)

            # TODO: expand the dims of anchors and tile them and compute the negative loss

            # do the pair count where you average by contributors only

            # this is your total loss


            # Further idea is to check what volumetric locations do each of the depth images corresponds to
            # unproject the entire depth image and convert to ref. and then sample.

        if hyp.do_touch_forward:
            ## ... Begin code for getting crops from visual memory ... ##
            sensor_depths_centers_x = center_sensor_W * torch.ones((hyp.B, hyp.sensor_S))
            sensor_depths_centers_x = sensor_depths_centers_x.cuda()
            sensor_depths_centers_y = center_sensor_H * torch.ones((hyp.B, hyp.sensor_S))
            sensor_depths_centers_y = sensor_depths_centers_y.cuda()
            sensor_depths_centers_z = sensor_depths[:, :, 0, center_sensor_H, center_sensor_W]

            # Next use Pixels2Camera to unproject all of these together.
            # merge the batch and the sequence dimension
            sensor_depths_centers_x = sensor_depths_centers_x.reshape(-1, 1, 1)
            sensor_depths_centers_y = sensor_depths_centers_y.reshape(-1, 1, 1)
            sensor_depths_centers_z = sensor_depths_centers_z.reshape(-1, 1, 1)

            fx, fy, x0, y0 = utils_geom.split_intrinsics(sensor_pix_T_cams_)
            sensor_depths_centers_in_camXs_ = utils_geom.Pixels2Camera(sensor_depths_centers_x, sensor_depths_centers_y,
                sensor_depths_centers_z, fx, fy, x0, y0)
            sensor_depths_centers_in_world_ = utils_geom.apply_4x4(sensor_origin_T_camXs_, sensor_depths_centers_in_camXs_)  # not used by the algorithm
            ## this will be later used for visualization hence saving it here for now
            sensor_depths_centers_in_ref_cam_ = utils_geom.apply_4x4(sensor_camRs_T_camXs_, sensor_depths_centers_in_camXs_)  # not used by the algorithm

            sensor_depths_centers_in_camXs = __u(sensor_depths_centers_in_camXs_).squeeze(2)

            # There has to be a better way to do this, for each of the cameras in the batch I want a box of size (ch, cw, cd)
            # TODO: rotation is the deviation of the box from the axis aligned do I want this
            tB, tN, _ = list(sensor_depths_centers_in_camXs.shape)  # 2, 512, _
            boxlist = torch.zeros(tB, tN, 9)  # 2, 512, 9
            boxlist[:, :, :3] = sensor_depths_centers_in_camXs  # this lies on the object
            boxlist[:, :, 3:6] = torch.FloatTensor([hyp.contextW, hyp.contextH, hyp.contextD])

            # convert the boxlist to lrtlist and to cuda
            # the rt here transforms the from box coordinates to camera coordinates
            box_lrtlist = utils_geom.convert_boxlist_to_lrtlist(boxlist)

            # Now I will use crop_zoom_from_mem functionality to get the features in each of the boxes
            # I will do it for each of the box separately as required by the api
            context_grid_list = list()
            for m in range(box_lrtlist.shape[1]):
                curr_box = box_lrtlist[:, m, :]
                context_grid = utils_vox.crop_zoom_from_mem(featRs, curr_box, 8, 8, 8,
                    sensor_camRs_T_camXs[:, m, :, :])
                context_grid_list.append(context_grid)
            context_grid_list = torch.stack(context_grid_list, dim=1)
            context_grid_list_ = __p(context_grid_list)
            ## ... till here I believe I have not introduced any randomness, so the points are still in
            ## ... End code for getting crops around this center of certain height, width and depth ... ##

            ## ... Begin code for passing the context grid through 3D CNN to obtain a vector ... ##
            sensor_cam_locs = feed['sensor_locs']  # these are in origin coordinates
            sensor_cam_quats = feed['sensor_quats'] # this too in in world_coordinates
            sensor_cam_locs_ = __p(sensor_cam_locs)
            sensor_cam_quats_ = __p(sensor_cam_quats)
            sensor_cam_locs_in_R_ = utils_geom.apply_4x4(sensor_camRs_T_origin_, sensor_cam_locs_.unsqueeze(1)).squeeze(1)
            # TODO TODO TODO confirm that this is right? TODO TODO TODO
            get_r_mat = lambda cam_quat: transformations.quaternion_matrix_py(cam_quat)
            rot_mat_Xs_ = torch.from_numpy(np.stack(list(map(get_r_mat, sensor_cam_quats_.cpu().numpy())))).to(sensor_cam_locs_.device).float()
            rot_mat_Rs_ = torch.bmm(sensor_camRs_T_origin_, rot_mat_Xs_)
            get_quat = lambda r_mat: transformations.quaternion_from_matrix_py(r_mat)
            sensor_quats_in_R_ = torch.from_numpy(np.stack(list(map(get_quat, rot_mat_Rs_.cpu().numpy())))).to(sensor_cam_locs_.device).float()

            pred_features_ = self.context_net(context_grid_list_,\
                sensor_cam_locs_in_R_, sensor_quats_in_R_)

            # normalize
            pred_features_ = l2_normalize(pred_features_, dim=1)
            pred_features = __u(pred_features_)

            # if doing moco I have to pass the inputs through the key(slow) network as well #
            if hyp.do_moc or hyp.do_eval_recall:
                key_context_grid_list_ = copy.deepcopy(context_grid_list_)
                key_sensor_cam_locs_in_R_ = copy.deepcopy(sensor_cam_locs_in_R_)
                key_sensor_quats_in_R_ = copy.deepcopy(sensor_quats_in_R_)
                self.key_context_net.eval()
                with torch.no_grad():
                    key_pred_features_ = self.key_context_net(key_context_grid_list_,\
                        key_sensor_cam_locs_in_R_, key_sensor_quats_in_R_)

                # normalize, normalization is very important why though
                key_pred_features_ = l2_normalize(key_pred_features_, dim=1)
                key_pred_features = __u(key_pred_features_)
            # end passing of the input through the slow network this is necessary for moco #
            ## ... End code for passing the context grid through 3D CNN to obtain a vector ... ##

        ## ... Begin code for doing metric learning between pred_features and sensor features ... ##
        # 1. Subsample both based on the number of positive samples
        if hyp.do_touch_embML:
            assert(hyp.do_touch_forward)
            assert(hyp.do_touch_feat)
            perm = torch.randperm(len(pred_features_))  ## 1024
            chosen_sensor_feats_ = sensor_features_[perm[:self.num_pos_samples*hyp.B]]
            chosen_pred_feats_ = pred_features_[perm[:self.num_pos_samples*B]]

            # 2. form the emb_vec and get pos and negative samples for the batch
            emb_vec = torch.stack((chosen_sensor_feats_, chosen_pred_feats_), dim=1).view(hyp.B*self.num_pos_samples*self.batch_k, -1)
            y = torch.stack([torch.range(0, self.num_pos_samples*B-1), torch.range(0, self.num_pos_samples*B-1)],\
                dim=1).view(B*self.num_pos_samples*self.batch_k) # (0, 0, 1, 1, ..., 255, 255)

            a_indices, anchors, positives, negatives, _ = self.sampler(emb_vec)

            # 3. Compute the loss, ML loss and the l2 distance betwee the embeddings
            margin_loss, _ = self.criterion(anchors, positives, negatives, self.beta, y[a_indices])
            total_loss = utils_misc.add_loss('embtouch/emb_touch_ml_loss', total_loss, margin_loss,
                hyp.emb_3D_ml_coeff, summ_writer)

            # the l2 loss between the embeddings
            l2_loss = torch.nn.functional.mse_loss(chosen_sensor_feats_, chosen_pred_feats_)
            total_loss = utils_misc.add_loss('embtouch/emb_l2_loss', total_loss, l2_loss,
                hyp.emb_3D_l2_coeff, summ_writer)
        ## ... End code for doing metric learning between pred_features and sensor_features ... ##

        ## ... Begin code for doing moc inspired ML between pred_features and sensor_features ... ##
        if hyp.do_moc and moc_init_done:
            moc_loss = self.moc_ml_net(sensor_features_, key_sensor_features_,\
                pred_features_, key_pred_features_, summ_writer)
            total_loss += moc_loss
        ## ... End code for doing moc inspired ML between pred_features and sensor_feature ... ##

        ## ... add code for filling up results needed for eval recall ... ##
        if hyp.do_eval_recall and moc_init_done:
            results['context_features'] = pred_features_
            results['sensor_depth_centers_in_world'] = sensor_depths_centers_in_world_
            results['sensor_depths_centers_in_ref_cam'] = sensor_depths_centers_in_ref_cam_
            results['object_name'] = feed['object_name']

            # I will do precision recall here at different recall values and summarize it using tensorboard
            recalls = [1, 5, 10, 50, 100, 200]
            # also should not include any gradients because of this
            # fast_sensor_emb_e = sensor_features_
            # fast_context_emb_e = pred_features_
            # slow_sensor_emb_g = key_sensor_features_
            # slow_context_emb_g = key_context_features_
            fast_sensor_emb_e = sensor_features_.clone().detach()
            fast_context_emb_e = pred_features_.clone().detach()

            # I will do multiple eval recalls here
            slow_sensor_emb_g = key_sensor_features_.clone().detach()
            slow_context_emb_g = key_pred_features_.clone().detach()

            # assuming the above thing goes well
            fast_sensor_emb_e = fast_sensor_emb_e.cpu().numpy()
            fast_context_emb_e = fast_context_emb_e.cpu().numpy()
            slow_sensor_emb_g = slow_sensor_emb_g.cpu().numpy()
            slow_context_emb_g = slow_context_emb_g.cpu().numpy()

            # now also move the vis to numpy and plot it using matplotlib
            vis_e = __p(sensor_rgbs)
            vis_g = __p(sensor_rgbs)
            np_vis_e = vis_e.cpu().detach().numpy()
            np_vis_e = np.transpose(np_vis_e, [0, 2, 3, 1])
            np_vis_g = vis_g.cpu().detach().numpy()
            np_vis_g = np.transpose(np_vis_g, [0, 2, 3, 1])

            # bring it back to original color
            np_vis_g = ((np_vis_g+0.5) * 255).astype(np.uint8)
            np_vis_e = ((np_vis_e+0.5) * 255).astype(np.uint8)

            # now compare fast_sensor_emb_e with slow_context_emb_g
            # since I am doing positive against this
            fast_sensor_emb_e_list = [fast_sensor_emb_e, np_vis_e]
            slow_context_emb_g_list = [slow_context_emb_g, np_vis_g]

            prec, vis, chosen_inds_and_neighbors_inds = compute_precision(
                fast_sensor_emb_e_list, slow_context_emb_g_list, recalls=recalls
            )

            # finally plot the nearest neighbour retrieval and move ahead
            if feed['global_step'] % 1 == 0:
                plot_nearest_neighbours(vis, step=feed['global_step'],
                                        save_dir='/home/gauravp/eval_results',
                                        name='fast_sensor_slow_context')

            # plot the precisions at different recalls
            for pr, re in enumerate(recalls):
                summ_writer.summ_scalar(f'evrefast_sensor_slow_context/recall@{re}',\
                    prec[pr])

            # now compare fast_context_emb_e with slow_sensor_emb_g
            fast_context_emb_e_list = [fast_context_emb_e, np_vis_e]
            slow_sensor_emb_g_list = [slow_sensor_emb_g, np_vis_g]

            prec, vis, chosen_inds_and_neighbors_inds = compute_precision(
                fast_context_emb_e_list, slow_sensor_emb_g_list, recalls=recalls
            )
            if feed['global_step'] % 1 == 0:
                plot_nearest_neighbours(vis, step=feed['global_step'],
                                        save_dir='/home/gauravp/eval_results',
                                        name='fast_context_slow_sensor')

            # plot the precisions at different recalls
            for pr, re in enumerate(recalls):
                summ_writer.summ_scalar(f'evrefast_context_slow_sensor/recall@{re}',\
                    prec[pr])


            # now finally compare both the fast, I presume we want them to go closer too
            fast_sensor_list = [fast_sensor_emb_e, np_vis_e]
            fast_context_list = [fast_context_emb_e, np_vis_g]

            prec, vis, chosen_inds_and_neighbors_inds = compute_precision(
                fast_sensor_list, fast_context_list, recalls=recalls
            )
            if feed['global_step'] % 1 == 0:
                plot_nearest_neighbours(vis, step=feed['global_step'],
                                        save_dir='/home/gauravp/eval_results',
                                        name='fast_sensor_fast_context')

            for pr, re in enumerate(recalls):
                summ_writer.summ_scalar(f'evrefast_sensor_fast_context/recall@{re}',\
                    prec[pr])

        ## ... done code for filling up results needed for eval recall ... ##
        summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, [key_sensor_features_, key_pred_features_]
