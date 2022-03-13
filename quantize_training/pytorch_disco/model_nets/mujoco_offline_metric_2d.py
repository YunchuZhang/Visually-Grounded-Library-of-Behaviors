import copy
import torch
import torch.nn as nn
import numpy as np

from model import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.viewnet import ViewNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D
from nets.moc_net import MOCTraining
from nets.metric_learner import MetricLearner
from nets.quant_policy_metric_models import EmbeddingGenerator1DFrom2D, EmbeddingGenerator3D

import torch.nn.functional as F

import utils

from utils.protos import VoxProto
from utils.memcoord import Coord, VoxCoord
from archs.encoder2D import Net2D

np.set_printoptions(precision=2)
np.random.seed(0)

class MUJOCO_OFFLINE_METRIC_2D(Model):
    """Model has the go and other functions
       Need to override the infer function"""
    def infer(self):
        print('---- BUILDING INFERENCE GRAPH -----')
        self.model = MujocoOfflineMetric2D(self.config)

class MujocoOfflineMetric2D(nn.Module):
    def __init__(self, config):
        super(MujocoOfflineMetric2D, self).__init__()

        self.config=config
        #if self.config.do_feat:
        #    print('------- adding featnet --------')
        #    self.featnet = FeatNet(self.config)
        #if self.config.do_occ:
        #    print('------- adding occnet ---------')
        #    self.occnet = OccNet(self.config)
        #if self.config.do_view:
        #    print('------- adding viewnet --------')
        #    self.viewnet = ViewNet(self.config)


        # coordinate range
        #self.coord_cam_front = Coord(-0.5, 0.5, -0.5, 0.5, 0.2, 1.2, 0.0, -0.4)
        #self.coord_mem = Coord(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.0, -0.4)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.featnet = Net2D(4, out_chans=self.config.feat_dim).to(device=device)


        MH, MW, MD = self.config.Y, self.config.X, self.config.Z
        MH2, MW2, MD2 = int(MH/2), int(MW/2), int(MD/2)
        self.output_dim = 128
        ## voxel size
        #mem_protos = VoxProto([MH, MW, MD])
        #halfmem_protos = VoxProto([MH2, MW2, MD2])

        # ##combine
        #self.mem_coord_cams = VoxCoord(self.coord_cam_front, mem_protos)
        #self.mem_coord_Rs = VoxCoord(self.coord_mem, mem_protos)
        #self.halfmem_coord_cams = VoxCoord(self.coord_cam_front, halfmem_protos)
        #self.halfmem_coord_Rs = VoxCoord(self.coord_mem, halfmem_protos)
        #self.feat_mem_coord_cams = None #self.halfmem_coord_Rs
        #self.feat_mem_coord_Rs = None

        
        self.is_learned_cluster_centers = True
        self.cluster_name_to_id = dict()
        self.cluster_id_to_name = dict()
        self.num_clusters = 0
        self.max_clusters = self.config.max_clusters

        self.object_refine_model = EmbeddingGenerator1DFrom2D(self.output_dim, in_channel=self.config.feat_dim * self.config.S)

        self.embeddings_shape = [self.output_dim]


        self.embedding_dim = self.embeddings_shape[0]
        self.embeddings = torch.nn.Embedding(self.max_clusters,
                                             self.embedding_dim)
        self.embeddings.cuda()
        
        #if self.config.is_refine_net:

    def get_centers(self):
        feat_dim = self.embeddings_shape[0]
        #MH, MW, MD = self.embeddings_shape[1], self.embeddings_shape[2], self.embeddings_shape[3]

        embeddings_reshape = torch.reshape(self.embeddings.weight, [self.max_clusters, feat_dim])
        return embeddings_reshape.detach().cpu().numpy()

    def get_cluster_centers_dict(self):
        cluster_center = dict()
        centers_tensor = self.get_centers()
        for cluster_id in range(self.num_clusters):
            cluster_name = self.cluster_id_to_name[cluster_id]
            cluster_center[cluster_name] = centers_tensor[cluster_id]

        import ipdb; ipdb.set_trace()
        return cluster_center

    def save_local_variables(self):
        output = dict()
        output["cluster_name_to_id"] = self.cluster_name_to_id
        output["cluster_id_to_name"] = self.cluster_id_to_name
        output["num_clusters"] = self.num_clusters

        return output

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
 
        _, C, H, W, D = presumably_object_tensor.shape
        presumably_object_tensor = torch.reshape(presumably_object_tensor.permute([0, 2, 3, 4, 1]), [B, N, H, W, D, C])
        

        # NOTE: As of now I am not doing backprop through this Tensor so
        # no need to keep it in gpu anymore
        results = dict()
        results['object_tensor'] = presumably_object_tensor.cpu().detach().numpy()
        results['featsRs_without_target_view'] = featsRs_without_target_view.permute([0, 2, 3, 4, 1]).cpu().detach().numpy()
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

        B = self.config.B
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        # unmerge sequence and batch dimensions
        __u = lambda x: utils.basic.unpack_seqdim(x, B)


        b, Nview, _, _ =  feed["origin_T_camXs"].shape
        _, nobjs, _, _ = feed['bbox_in_ref_cam'].shape
        bbox_in_ref_cam = feed['bbox_in_ref_cam'].repeat(1, Nview, 1, 1)

        bbox_in_ref_cam_ = __p(bbox_in_ref_cam)
        rgb_camXs_ = __p(feed["rgb_camXs"])
        _, _, H, W = rgb_camXs_.shape
        xyz_camXs_ = __p(feed["xyz_camXs"])
        camXs_T_origin_ = utils.geom.safe_inverse(__p(feed["origin_T_camXs"]))
        pix_T_cams_ = __p(feed["pix_T_cams"])
        depth_camXs_, _  = utils.geom.create_depth_image(pix_T_cams_, xyz_camXs_, H, W)

        rgbd_camXs_ = torch.cat((rgb_camXs_, depth_camXs_), dim=1)



        

        bbox_in_camXs = utils.geom.apply_4x4(camXs_T_origin_, bbox_in_ref_cam_)
        # (batch_size x nviews x nobjs) x 8 x 3
        bbox_in_pix = utils.geom.apply_pix_T_cam(pix_T_cams_, bbox_in_camXs)

        bb_min, bb = torch.min(bbox_in_pix[:, :, :2], axis=1)
        bb_max, bb = torch.max(bbox_in_pix[:, :, :2], axis=1)

        bbox_2d = torch.cat((bb_min, bb_max), dim=1)
        bbox_2d = torch.reshape(bbox_2d, [b * Nview, nobjs, 4])
        #bbox_2d = __u(bbox_2d)
        #import ipdb; ipdb.set_trace()
        bbox_2d_list = torch.unbind(bbox_2d, dim=0)#[ts.unsqueeze(0) for ts in torch.unbind(bbox_2d, dim=0)]

        import torchvision

        crop_images = torchvision.ops.roi_align(rgbd_camXs_, bbox_2d_list, (64, 64))


        feat_camXs_ = self.featnet(crop_images)
        _, C, FH, FW = feat_camXs_.shape
        feat_camXs = torch.reshape(feat_camXs_, [B, Nview, nobjs, C, FH, FW]).permute([0, 2, 1, 3, 4, 5])
        feat_camXs_stack = torch.reshape(feat_camXs, [B * nobjs, Nview * C, FH, FW])

        #feat_camXs = __u(feat_camXs_)
        #B, Nview, C, H, W = feat_camXs.shape
        #feat_camXs_stack = torch.reshape(feat_camXs, [B, Nview * C, H, W])
        presumably_object_tensor = self.object_refine_model(feat_camXs_stack)

        _, C = presumably_object_tensor.shape

        #self.object_refine_model(presumably_object_tensor)

        #summ_writer.summ_feats('crop_feats_val/object_tensor', tuple([presumably_object_tensor]), pca=True)

        input_embs = presumably_object_tensor
        _, C = input_embs.shape
        input_shape = input_embs.shape
        nobjs = len(feed["cluster_id"])
        bsize = len(feed["cluster_id"][0])

        encoding_indices = torch.zeros(bsize, nobjs, device=inputs_device)

        if self.config.metric_learning_loss_type == "cluster_id":

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

            B, C = input_embs.shape
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
