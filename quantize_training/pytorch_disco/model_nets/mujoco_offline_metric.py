import copy
import torch
import torch.nn as nn
import numpy as np
import ipdb;
st=ipdb.set_trace

from model import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.viewnet import ViewNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D
from nets.moc_net import MOCTraining
from nets.metric_learner import MetricLearner
from nets.quant_policy_metric_models import EmbeddingGenerator1D, EmbeddingGenerator3D

import torch.nn.functional as F

import utils

from utils.protos import VoxProto
from utils.memcoord import Coord, VoxCoord

np.set_printoptions(precision=2)
np.random.seed(0)

class MUJOCO_OFFLINE_METRIC(Model):
    """Model has the go and other functions
       Need to override the infer function"""
    def infer(self):
        print('---- BUILDING INFERENCE GRAPH -----')
        self.model = MujocoOfflineMetric(self.config)

class MujocoOfflineMetric(nn.Module):
    def __init__(self, config):
        super(MujocoOfflineMetric, self).__init__()

        self.config=config
        if self.config.do_feat:
            print('------- adding featnet --------')
            self.featnet = FeatNet(self.config)
        if self.config.do_occ:
            print('------- adding occnet ---------')
            self.occnet = OccNet(self.config)
        if self.config.do_view:
            print('------- adding viewnet --------')
            self.viewnet = ViewNet(self.config)


        # coordinate range
        self.coord_cam_front = Coord(-0.5, 0.5, -0.5, 0.5, 0.2, 1.2, 0.0, -0.4)
        self.coord_mem = Coord(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.0, -0.4)


        MH, MW, MD = self.config.Y, self.config.X, self.config.Z
        MH2, MW2, MD2 = int(MH/2), int(MW/2), int(MD/2)
        # voxel size
        mem_protos = VoxProto([MH, MW, MD])
        halfmem_protos = VoxProto([MH2, MW2, MD2])

        #combine
        self.mem_coord_cams = VoxCoord(self.coord_cam_front, mem_protos)
        self.mem_coord_Rs = VoxCoord(self.coord_mem, mem_protos)
        self.halfmem_coord_cams = VoxCoord(self.coord_cam_front, halfmem_protos)
        self.halfmem_coord_Rs = VoxCoord(self.coord_mem, halfmem_protos)
        self.feat_mem_coord_cams = None #self.halfmem_coord_Rs
        self.feat_mem_coord_Rs = None

        
        self.is_learned_cluster_centers = True
        self.cluster_name_to_id = dict()
        self.cluster_id_to_name = dict()
        self.num_clusters = 0
        self.max_clusters = self.config.max_clusters


        self.embeddings_shape = [self.config.feat_dim, MH2, MW2, MD2]
        if self.config.is_refine_net:
            self.embeddings_shape = [self.config.feat_dim, 16, 16, 16]

        self.embedding_dim = self.embeddings_shape[0] * self.embeddings_shape[1] * self.embeddings_shape[2] * self.embeddings_shape[3]
        self.embeddings = torch.nn.Embedding(self.max_clusters,
                                             self.embedding_dim)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.embeddings.to(device=device)
        
        if self.config.is_refine_net:
            self.object_refine_model = EmbeddingGenerator3D(self.config.feat_dim, en_channel=self.config.feat_dim)

    def get_centers(self):
        feat_dim = self.embeddings_shape[0]
        MH, MW, MD = self.embeddings_shape[1], self.embeddings_shape[2], self.embeddings_shape[3]

        embeddings_reshape = torch.reshape(self.embeddings.weight, [self.max_clusters, feat_dim, MH, MW, MD]).permute([0, 2, 3, 4, 1])
        return embeddings_reshape.detach().cpu().numpy()
    
    def get_cluster_centers_dict(self):
        cluster_center = dict()
        centers_tensor = self.get_centers()
        for cluster_id in range(self.num_clusters):
            cluster_name = self.cluster_id_to_name[cluster_id]
            cluster_center[cluster_name] = centers_tensor[cluster_id]

        return cluster_center

    def save_local_variables(self):
        output = dict()
        output["cluster_name_to_id"] = self.cluster_name_to_id
        output["cluster_id_to_name"] = self.cluster_id_to_name
        output["num_clusters"] = self.num_clusters

        return output



    def unproject(self, cam_rgbd_inputs, cam_info_inputs):

        rgb_camXs, xyz_camXs = cam_rgbd_inputs
        pix_T_cams, origin_T_camXs, origin_T_camRs = cam_info_inputs

        B, H, W, V, S = self.config.B, self.config.H, self.config.W, self.config.V, self.config.S
        PH, PW = self.config.PH, self.config.PW  # this is the size of the predicted image
        # the next are the memory dimensions, do not know why this naming though

        # merge sequence and batch dimensions
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        # unmerge sequence and batch dimensions
        __u = lambda x: utils.basic.unpack_seqdim(x, B)


        pix_T_cams_ = __p(pix_T_cams)  # packing here the (B,S) => (B*S)
        # intrinsic matrix packed and unpacked end
        origin_T_camRs_ = __p(origin_T_camRs)
        origin_T_camXs_ = __p(origin_T_camXs)
        # origin_T_camXs unpacked and packed end

        # completed getting inputs now combining them
        # 1. Converts from camX to camR which is Adam's coordinate system
        # get from camX_T_camR and camR_T_camX and pack unpack it
        camRs_T_camXs_ = torch.matmul(utils.geom.safe_inverse(
            origin_T_camRs_), origin_T_camXs_)
        camXs_T_camRs_ = utils.geom.safe_inverse(camRs_T_camXs_)
        camRs_T_camXs = __u(camRs_T_camXs_)
        camXs_T_camRs = __u(camXs_T_camRs_)
        # end of camX_T_camR and camR_T_camX and pack unpack it

        # goes directly from camR to image in each camera image frame
        pix_T_cams_ = utils.geom.pack_intrinsics(pix_T_cams_[:, 0, 0], pix_T_cams_[:, 1, 1], pix_T_cams_[:, 0, 2],
            pix_T_cams_[:, 1, 2])
        pix_T_camRs_ = torch.matmul(pix_T_cams_, camXs_T_camRs_)
        pix_T_camRs = __u(pix_T_camRs_)
        # end of computation for matrix which goes from camR to each camera image frame

        # pointclouds in each camera frame
        xyz_camXs_ = __p(xyz_camXs)
        # pointclouds converted to camR coordinate system
        xyz_camRs_ = utils.geom.apply_4x4(camRs_T_camXs_, xyz_camXs_)
        xyz_camRs = __u(xyz_camRs_)
        # TODO: visualize the point cloud here and check that it makes sense

        # get occupancy maps from pointclouds
        # QUESTION: what is the space you are discretizing, I mean the extent of the space
        occRs_ = utils.vox.voxelize_xyz(xyz_camRs_, self.mem_coord_Rs)
        occXs_ = utils.vox.voxelize_xyz(xyz_camXs_, self.mem_coord_cams)

        occRs_half_ = utils.vox.voxelize_xyz(xyz_camRs_, self.halfmem_coord_Rs)
        occXs_half_ = utils.vox.voxelize_xyz(xyz_camXs_, self.halfmem_coord_cams)
        occRs = __u(occRs_)
        occXs = __u(occXs_)
        occRs_half = __u(occRs_half_)
        occXs_half = __u(occXs_half_)

        # unproject depth images, This is done for the color images not the depths
        ## rgb unprojection, bilinearly samples and fills the grid
        my_device = rgb_camXs.device

        unpRs_ = utils.vox.unproject_rgb_to_mem(__p(rgb_camXs), pix_T_camRs_, self.mem_coord_Rs, device=my_device)
        unpXs_ = utils.vox.unproject_rgb_to_mem(__p(rgb_camXs), pix_T_cams_, self.mem_coord_cams, device=my_device)
        unpRs = __u(unpRs_)
        unpXs = __u(unpXs_)

        # NOTE: still do not know why is this required or where is this used for that matter
        depth_camXs_, valid_camXs_ = utils.geom.create_depth_image(pix_T_cams_, xyz_camXs_, H, W)
        dense_xyz_camXs_ = utils.geom.depth2pointcloud(depth_camXs_, pix_T_cams_)
        dense_xyz_camRs_ = utils.geom.apply_4x4(camRs_T_camXs_, dense_xyz_camXs_)

        # this is B*S x H*W x 3
        inbound_camXs_ = utils.vox.get_inbounds(dense_xyz_camRs_, self.mem_coord_cams).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])  # NOTE: Here there is a difference in tensorflow code
        inbound_camXs = __u(inbound_camXs_)

        depth_camXs = __u(depth_camXs_)
        valid_camXs = __u(valid_camXs_) * inbound_camXs

        return depth_camXs, valid_camXs, camRs_T_camXs, camXs_T_camRs, unpXs, unpRs, occXs, occRs, occXs_half, occRs_half

    def predict_forward(self, feed):
        inputs_device = feed["rgb_camXs"].device

        cam_rgbd_inputs = (feed["rgb_camXs"], feed["xyz_camXs"])
        cam_info_inputs = (feed["pix_T_cams"], feed["origin_T_camXs"], feed["origin_T_camRs"])
        depth_camXs, valid_camXs, camRs_T_camXs, camXs_T_camRs, unpXs, unpRs, occXs, occRs, occXs_half, occRs_half = self.unproject(cam_rgbd_inputs, cam_info_inputs)

        B = self.config.B
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        # unmerge sequence and batch dimensions
        __u = lambda x: utils.basic.unpack_seqdim(x, B)

        if self.config.do_feat:
            rgb_camXs, xyz_camXs = cam_rgbd_inputs

            featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)  # B, S, 4, H, W, D
            featXs_input_ = __p(featXs_input)

            freeXs_ = utils.vox.get_freespace(__p(xyz_camXs), __p(occXs_half), self.halfmem_coord_cams)
            freeXs = __u(freeXs_)
            visXs = torch.clamp(occXs_half + freeXs, 0.0, 1.0)

            #if type(mask_) != type(None): # featXs_input: B x NVIEWS x 4 x 64 x 64 x 64
            assert(list(occXs.shape)[3:6] == list(featXs_input.shape)[3:6])

            featXs_, validXs_, feat_loss = self.featnet(featXs_input_, mask=__p(occXs),
                set_num=feed['set_num'])
            assert feat_loss.item() == 0.0, "there is nothing to guide featnet by itself"
            # for each view features are being predicted, NOTE that nothing is brought into common view yet
            validXs, featXs = __u(validXs_), __u(featXs_)

            #### .... BEGIN Converting everything to ref frame .... ####
            validRs = utils.vox.apply_4x4_to_voxs(camRs_T_camXs, validXs, mem_coord_As=self.halfmem_coord_cams, mem_coord_Bs = self.halfmem_coord_Rs)
            visRs = utils.vox.apply_4x4_to_voxs(camRs_T_camXs, visXs, mem_coord_As=self.halfmem_coord_cams, mem_coord_Bs =self.halfmem_coord_Rs)
            featRs = utils.vox.apply_4x4_to_voxs(camRs_T_camXs, featXs, mem_coord_As=self.halfmem_coord_cams, mem_coord_Bs = self.halfmem_coord_Rs)
            if self.feat_mem_coord_Rs == None:
                self.feat_mem_coord_Rs = self.halfmem_coord_Rs
            #### .... featRs_without_target_view contains features from all the views
            #### .... warped and brought into common frame and aggregated .... Using
            #### .... features occupancy and target view should be predicted .... ####
            # B x 32 x H x W x D
            featsRs_without_target_view = torch.mean(featRs[:, 1:], dim=1)

        #crop object features
        bbox_in_ref_cam = feed['bbox_in_ref_cam']
        # based on the batch size this would be B, N, 8, 3

        min_bounds = bbox_in_ref_cam[:, :,  0, :]
        max_bounds = bbox_in_ref_cam[:, :, -1, :]

        lengths = torch.abs(max_bounds - min_bounds)
        center = (max_bounds + min_bounds) * 0.5

        if self.config.fix_crop:
            lengths *= 0.0
            lengths += 0.32

        # now form the box and then covert to lrt list
        B = self.config.B# since i have only one box
        N = 1 # number of objects
        # 9 is cx, cy, cz, lx, ly, lz, rx, ry, rz
        boxlist = torch.zeros(B, N, 9)

        # NOTE: Note: I am assuming here that N = 1 !!!!!!
        boxlist[:, :, :3] = center#.unsqueeze(1)
        boxlist[:, :, 3:6] = lengths#.unsqueeze(1)

        # convert it to lrt list, it contains box length and rt to go
        # from box coordinates to ref coordinate system.
        box_lrtlist = utils.geom.convert_boxlist_to_lrtlist(boxlist)

        # now this is already in the ref coordinate system which was not
        # the case with my previous use of the crop_zoom_from_mem func.
        # Hence I had previously included camR_T_camXs which is not req here
        _, _, box_dim = box_lrtlist.shape


        presumably_object_tensor = utils.vox.crop_zoom_from_mem(
            featsRs_without_target_view,
            self.feat_mem_coord_Rs,
            torch.reshape(box_lrtlist[:, :, :], [B * N, box_dim]),
            32, 32, 32
        )
        
        if self.config.is_refine_net:
            presumably_object_tensor = self.object_refine_model(presumably_object_tensor)

        _, C, H, W, D = presumably_object_tensor.shape
        presumably_object_tensor = torch.reshape(presumably_object_tensor.permute([0, 2, 3, 4, 1]), [B, N, H, W, D, C])
        

        if self.config.do_view:
            assert self.config.do_feat

            PH, PW = self.config.PH, self.config.PW
            sy = float(PH) / float(self.config.H)
            sx = float(PW) / float(self.config.W)

            assert(sx == 0.5)
            assert(sy == 0.5)

            # projpix_T_cams, are the intrinsics for the projection, just scale the true intrinsics
            pix_T_cams = feed["pix_T_cams"]
            projpix_T_cams = __u(utils.geom.scale_intrinsics(__p(pix_T_cams), sx, sy))

            # now I will project the predicted feats to target view (warp)
            feat_projtarget_view = utils.vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:, 0], camXs_T_camRs[:, 0], self.halfmem_coord_Rs, featsRs_without_target_view,
                self.config.view_depth, PH, PW)

            rgb_X0 = utils.basic.downsample(rgb_camXs[:, 0], 2)  ## NOTE: this is the ground truth

            # rgb_e: b x 3 x 64 x 64
            view_loss, rgb_e, emb2D_e = self.viewnet(
                feat_projtarget_view,
                rgb_X0,
                set_num=feed['set_num'])


        # NOTE: As of now I am not doing backprop through this Tensor so
        # no need to keep it in gpu anymore
        results = dict()
        results['object_tensor'] = presumably_object_tensor.cpu().detach().numpy()
        results['featsRs_without_target_view'] = featsRs_without_target_view.permute([0, 2, 3, 4, 1]).cpu().detach().numpy()
        if self.config.do_view:
            results['rgb_e'] = rgb_e.permute(0, 2, 3, 1).cpu().detach().numpy()

        # Add the plot of this to tensorboard, and also think how can you
        # visualize if the correct thing is being returned to you.

        return results

    def convert_objects_to_features(self, feed):
        results = self.predict_forward(feed)

        return results['object_tensor']


    def dump_one_batch(self, feed):
        import pickle
        import copy

        feed_copy = dict()
        i = 0
        for key in feed:
            if key in ['record', 'writer', 'global_step']:
                continue
            if torch.is_tensor(feed[key]):
                tensor_np = feed[key].cpu()
                feed_copy[key] = tensor_np
            else:
                feed_copy[key] = feed[key]
            i += 1
            #if i > 1:
            #    break

        with open("tmp/feed.pkl", "wb") as f:
            pickle.dump(feed_copy, f)

        import ipdb; ipdb.set_trace()

    def forward(self, feed):
        # feed is the input here, let's see what it has
        results = dict()

        #self.dump_one_batch(feed)

        # Whenever forward is called, this is instantiated which creates summ_writer object
        # save this is True if global_step % log_freq == 0
        summ_writer = utils.improc.Summ_writer(
            config = self.config,
            writer = feed['writer'],
            global_step = feed['global_step'],
            set_name= feed['set_name'],
            fps=8)

        writer = feed['writer']
        inputs_device = feed["rgb_camXs"].device
        #global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        cam_rgbd_inputs = (feed["rgb_camXs"], feed["xyz_camXs"])
        cam_info_inputs = (feed["pix_T_cams"], feed["origin_T_camXs"], feed["origin_T_camRs"])

        depth_camXs, valid_camXs, camRs_T_camXs, camXs_T_camRs, unpXs, unpRs, occXs, occRs, occXs_half, occRs_half = self.unproject(cam_rgbd_inputs, cam_info_inputs)

        B = self.config.B
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        # unmerge sequence and batch dimensions
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        #### ... VISUALIZE what we got ... ####
        if not feed['set_num'] == 1:

            rgb_camXs, xyz_camXs = cam_rgbd_inputs
            rgb_camRs = feed["rgb_camRs"]

            summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(depth_camXs, dim=1))
            summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
            summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
            summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
            summ_writer.summ_occs('3d_inputs/occXs', torch.unbind(occXs, dim=1), reduce_axes=[2])
            summ_writer.summ_unps('3d_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))
            if summ_writer.save_this:
                # why compute again?
                #unpRs_ = utils.vox.unproject_rgb_to_mem(__p(rgb_camXs), utils.basic.matmul2(pix_T_cams_, camXs_T_camRs_), self.mem_coord_Rs)
                #unpRs = __u(unpRs_)
                #occRs_ = utils.vox.voxelize_xyz(xyz_camRs_, self.mem_coord_Rs)
                summ_writer.summ_occs('3d_inputs/occRs', torch.unbind(occRs, dim=1), reduce_axes=[2])
                summ_writer.summ_unps('3d_inputs/unpRs', torch.unbind(unpRs, dim=1), torch.unbind(occRs, dim=1))
        else:

            rgb_camXs, xyz_camXs = cam_rgbd_inputs
            rgb_camRs = feed["rgb_camRs"]
            summ_writer.summ_oneds('2D_inputs_val/depth_camXs', torch.unbind(depth_camXs, dim=1))
            summ_writer.summ_oneds('2D_inputs_val/valid_camXs', torch.unbind(valid_camXs, dim=1))
            summ_writer.summ_rgbs('2D_inputs_val/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
            summ_writer.summ_rgbs('2D_inputs_val/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
            summ_writer.summ_occs('3d_inputs_val/occXs', torch.unbind(occXs, dim=1), reduce_axes=[2])
            summ_writer.summ_unps('3d_inputs_val/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))
            if summ_writer.save_this:
                #unpRs_ = utils.vox.unproject_rgb_to_mem(__p(rgb_camXs), Z, Y, X, utils.basic.matmul2(pix_T_cams_, camXs_T_camRs_))
                #unpRs = __u(unpRs_)
                #occRs_ = utils.vox.voxelize_xyz(xyz_camRs_, Z, Y, X)
                summ_writer.summ_occs('3d_inputs_val/occRs', torch.unbind(occRs, dim=1), reduce_axes=[2])
                summ_writer.summ_unps('3d_inputs_val/unpRs', torch.unbind(unpRs, dim=1), torch.unbind(occRs, dim=1))


        # the idea behind view-pred is form memory with the remaining views project it to target view and
        # then use this memory to predict the target image
        # idea behind occ_prediction is use the memory to predict occupancy in ref view and compare it
        # with the ground truth occupancy in the ref view

        if self.config.do_feat:
            rgb_camXs, xyz_camXs = cam_rgbd_inputs

            featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)  # B, S, 4, H, W, D
            featXs_input_ = __p(featXs_input)

            freeXs_ = utils.vox.get_freespace(__p(xyz_camXs), __p(occXs_half), self.halfmem_coord_cams)
            freeXs = __u(freeXs_)
            visXs = torch.clamp(occXs_half + freeXs, 0.0, 1.0)

            #if type(mask_) != type(None): # featXs_input: B x NVIEWS x 4 x 64 x 64 x 64
            assert(list(occXs.shape)[3:6] == list(featXs_input.shape)[3:6])

            featXs_, validXs_, feat_loss = self.featnet(featXs_input_, summ_writer, mask=__p(occXs),
                set_num=feed['set_num'])
            total_loss += feat_loss
            assert feat_loss.item() == 0.0, "there is nothing to guide featnet by itself"
            # for each view features are being predicted, NOTE that nothing is brought into common view yet
            validXs, featXs = __u(validXs_), __u(featXs_)

            #### .... BEGIN Converting everything to ref frame .... ####
            validRs = utils.vox.apply_4x4_to_voxs(camRs_T_camXs, validXs, mem_coord_As=self.halfmem_coord_cams, mem_coord_Bs = self.halfmem_coord_Rs)
            visRs = utils.vox.apply_4x4_to_voxs(camRs_T_camXs, visXs, mem_coord_As=self.halfmem_coord_cams, mem_coord_Bs =self.halfmem_coord_Rs)
            featRs = utils.vox.apply_4x4_to_voxs(camRs_T_camXs, featXs, mem_coord_As=self.halfmem_coord_cams, mem_coord_Bs = self.halfmem_coord_Rs)
            if self.feat_mem_coord_Rs == None:
                self.feat_mem_coord_Rs = self.halfmem_coord_Rs
            #### .... END converting everything to ref frame .... ####

            ### ... Remember _e added at the end means it is estimated ... ###
            vis3D_e = torch.max(validRs[:, 1:], dim=1)[0] * torch.max(visRs[:, 1:], dim=1)[0]
            ### ... only thing which is using _e is below visualization ... ###

            if not feed['set_num'] == 1:
                summ_writer.summ_feats('3D_feats/featXs_input', torch.unbind(featXs_input, dim=1), pca=True)
                summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), pca=True)
                summ_writer.summ_feats('3D_feats/featRs_output', torch.unbind(featRs, dim=1), pca=True)
                summ_writer.summ_feats('3D_feats/validRs', torch.unbind(validRs, dim=1), pca=False)
                summ_writer.summ_feat('3D_feats/vis3D_e', vis3D_e, pca=False)
            else:
                summ_writer.summ_feats('3D_feats_val/featXs_input', torch.unbind(featXs_input, dim=1), pca=True)
                summ_writer.summ_feats('3D_feats_val/featXs_output', torch.unbind(featXs, dim=1), pca=True)
                summ_writer.summ_feats('3D_feats_val/featRs_output', torch.unbind(featRs, dim=1), pca=True)
                summ_writer.summ_feats('3D_feats_val/validRs', torch.unbind(validRs, dim=1), pca=False)
                summ_writer.summ_feat('3D_feats_val/vis3D_e', vis3D_e, pca=False)

            #### .... featRs_without_target_view contains features from all the views
            #### .... warped and brought into common frame and aggregated .... Using
            #### .... features occupancy and target view should be predicted .... ####
            # import ipdb;ipdb.set_trace()
            featsRs_without_target_view = torch.mean(featRs[:, 1:], dim=1)

            #if self.config.do_generate_data or (self.config.do_validation and feed['set_num'] == 1):
            featRs_with_target_view = torch.mean(featRs, dim=1)

        if self.config.do_occ and self.config.occ_do_cheap:

            occRs_sup, freeRs_sup, freeXs = utils.vox.prep_occs_supervision(
                xyz_camXs,
                occRs_half,
                occXs_half,
                camRs_T_camXs,
                self.halfmem_coord_Rs, 
                self.halfmem_coord_cams,
                agg=True)

            if feed['set_num'] != 1:
                summ_writer.summ_occ('occ_sup/occ_sup', occRs_sup, reduce_axes=[2])
                summ_writer.summ_occ('occ_sup/free_sup', freeRs_sup, reduce_axes=[2])
                summ_writer.summ_occs('occ_sup/freeXs_sup', torch.unbind(freeXs, dim=1), reduce_axes=[2])
                summ_writer.summ_occs('occ_sup/occXs_sup', torch.unbind(occXs_half, dim=1), reduce_axes=[2])
            else:
                summ_writer.summ_occ('occ_sup_val/occ_sup', occRs_sup, reduce_axes=[2])
                summ_writer.summ_occ('occ_sup_val/free_sup', freeRs_sup, reduce_axes=[2])
                summ_writer.summ_occs('occ_sup_val/freeXs_sup', torch.unbind(freeXs, dim=1), reduce_axes=[2])
                summ_writer.summ_occs('occ_sup_val/occXs_sup', torch.unbind(occXs_half, dim=1), reduce_axes=[2])

            occ_loss, occRs_pred_ = self.occnet(featsRs_without_target_view,
                                                occRs_sup,
                                                freeRs_sup,
                                                torch.max(validRs[:, 1:], dim=1)[0],
                                                summ_writer,
                                                set_num=feed['set_num'])

            occRs_pred = __u(occRs_pred_)
            total_loss += occ_loss

        # update the center embedding

        if self.config.do_view:
            assert self.config.do_feat
            # we warped the features into canonical view which is featR
            # now we resample to target view which is view (0) and decode
            # be sure not to pass in the features of the view to decode
            # use featRs_without_target_view as the features in the canonical view

            PH, PW = self.config.PH, self.config.PW
            sy = float(PH) / float(self.config.H)
            sx = float(PW) / float(self.config.W)

            assert(sx == 0.5)
            assert(sy == 0.5)

            # projpix_T_cams, are the intrinsics for the projection, just scale the true intrinsics
            pix_T_cams = feed["pix_T_cams"]
            projpix_T_cams = __u(utils.geom.scale_intrinsics(__p(pix_T_cams), sx, sy))

            # now I will project the predicted feats to target view (warp)
            feat_projtarget_view = utils.vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:, 0], camXs_T_camRs[:, 0], self.halfmem_coord_Rs, featsRs_without_target_view,
                self.config.view_depth, PH, PW)

            rgb_X0 = utils.basic.downsample(rgb_camXs[:, 0], 2)  ## NOTE: this is the ground truth

            view_loss, rgb_e, emb2D_e = self.viewnet(
                feat_projtarget_view,
                rgb_X0,
                summ_writer,
                set_num=feed['set_num'])

            total_loss += view_loss

        ########   get the object tensor ###############
        bbox_in_ref_cam = feed['bbox_in_ref_cam']
        # based on the batch size this would be B, N, 3
        min_bounds = bbox_in_ref_cam[:, :, 0]
        max_bounds = bbox_in_ref_cam[:, :, -1]

        lengths = torch.abs(max_bounds - min_bounds)
        center = (max_bounds + min_bounds) * 0.5

        # now form the box and then covert to lrt list
        B, N = self.config.B, 1 # since i have only one box
        # 9 is cx, cy, cz, lx, ly, lz, rx, ry, rz
        boxlist = torch.zeros(B, N, 9)
        # NOTE: Note: I am assuming here that N = 1 !!!!!!
        boxlist[:, :, :3] = center #.unsqueeze(1)
        boxlist[:, :, 3:6] = lengths #.unsqueeze(1)

        # convert it to lrt list, it contains box length and rt to go
        # from box coordinates to ref coordinate system.
        box_lrtlist = utils.geom.convert_boxlist_to_lrtlist(boxlist)

        # now this is already in the ref coordinate system which was not
        # the case with my previous use of the crop_zoom_from_mem func.
        # Hence I had previously included camR_T_camXs which is not req here
        presumably_object_tensor = utils.vox.crop_zoom_from_mem(
            featRs_with_target_view,
            self.feat_mem_coord_Rs,
            box_lrtlist[:, 0, :],
            32, 32, 32
        )
        # NOTE: As of now I am not doing backprop through this Tensor so
        # no need to keep it in gpu anymore


        
        _, C, H, W, D = presumably_object_tensor.shape

        #self.object_refine_model(presumably_object_tensor)

        summ_writer.summ_feats('crop_feats_val/object_tensor', tuple([presumably_object_tensor]), pca=True)

        if self.config.is_refine_net:
            presumably_object_tensor = self.object_refine_model(presumably_object_tensor)
        #self.object_refine_model(p)
        input_embs = presumably_object_tensor
        _, C, H, W, D = input_embs.shape
        
        input_shape = input_embs.shape
        nobjs = len(feed["cluster_id"])
        bsize = len(feed["cluster_id"][0])

        if self.config.metric_learning_loss_type == "cluster_id":
            encoding_indices = torch.zeros(bsize, nobjs, device=inputs_device)

            for obj_id in range(nobjs):
                for batch_id in range(bsize):
                    cluster_name =  feed["cluster_id"][obj_id][batch_id]
                    if cluster_name not in self.cluster_name_to_id:

                        if self.config.is_init_cluter_with_instance:
                            self.embeddings.weight.data[self.num_clusters, :] = presumably_object_tensor[batch_id * nobjs +  obj_id, :].detach().view(-1)

                        self.cluster_name_to_id[cluster_name] = self.num_clusters
                        self.cluster_id_to_name[self.num_clusters] = cluster_name
                        self.num_clusters += 1
                    encoding_indices[batch_id, obj_id] = self.cluster_name_to_id[cluster_name]

            encoding_indices = torch.reshape(encoding_indices, [nobjs * bsize, 1]).long()
            encodings = torch.zeros(encoding_indices.shape[0], self.max_clusters, device=inputs_device)

            encodings.scatter_(1, encoding_indices, 1)
            quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)

            e_latent_loss = F.mse_loss(quantized.detach(), input_embs)
            q_latent_loss = F.mse_loss(quantized, input_embs.detach())
            quat_loss = q_latent_loss + self.config.commitment_cost * e_latent_loss
            total_loss += quat_loss
        elif self.config.metric_learning_loss_type == "success_rate":

            if len(self.cluster_name_to_id.keys()) == 0:
                self.num_clusters = feed['success_rates'].shape[1]
                for cluster_id in range(self.num_clusters):
                    self.cluster_name_to_id[f'c{cluster_id}'] = cluster_id
                    self.cluster_id_to_name[cluster_id] = f'c{cluster_id}'

            bin_label = (feed['success_rates'] >= 0.8).float()
            B, C, H, W, D = input_embs.shape
            input_embs_flat = input_embs.view(B, -1)
            scores = torch.matmul(input_embs_flat, self.embeddings.weight.T)
            logits = scores[:, :self.num_clusters]

            bce_loss = torch.nn.BCEWithLogitsLoss()


            quat_loss = bce_loss(logits/1000.0, bin_label)
            total_loss += quat_loss

        results['object_tensor'] = presumably_object_tensor.detach().cpu()
        results['record_name'] = feed['record']

        # Add the plot of this to tensorboard, and also think how can you
        # visualize if the correct thing is being returned to you.


        # if hyp.do_metric_learning:
        #     B, _, _, _, _  = presumably_object_tensor.shape
        #     assert B >= 2, "Metric learner requires one positive and atleast one negative example to train" 
        #     metric_loss, _ = self.metric_learner(presumably_object_tensor,feed["object_id"])
        #     total_loss += metric_loss
        #     summ_writer.summ_scalar('metric_learn/metric_loss', metric_loss.cpu().item())

        summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results
