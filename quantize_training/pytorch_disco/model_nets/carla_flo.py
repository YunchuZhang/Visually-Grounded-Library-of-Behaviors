import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np

from model_base import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.flownet import FlowNet
from nets.viewnet import ViewNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

import torch.nn.functional as F

from utils_basic import *
import utils_vox
import utils_samp
import utils_geom
import utils_misc
import utils_improc

np.set_printoptions(precision=2)
np.random.seed(0)

class CARLA_FLO(Model):
    def infer(self):
        print("------ BUILDING INFERENCE GRAPH ------")
        self.model = CarlaFloNet().to(self.device)

class CarlaFloNet(nn.Module):
    def __init__(self):
        super(CarlaFloNet, self).__init__()
        self.featnet = FeatNet()
        # self.occnet = OccNet()
        self.flownet = FlowNet()
        # self.viewnet = ViewNet()
        # self.embnet2D = EmbNet2D()
        # self.embnet3D = EmbNet3D()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.autograd.set_detect_anomaly(True)

    def forward(self, feed):
        results = dict()
        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8)
        
        writer = feed['writer']
        global_step = feed['global_step']

        total_loss = torch.tensor(0.0)

        __p = lambda x: pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)

        B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        PH, PW = hyp.PH, hyp.PW
        K = hyp.K
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        D = 9

        rgb_camRs = feed["rgb_camRs"]
        rgb_camXs = feed["rgb_camXs"]
        pix_T_cams = feed["pix_T_cams"]
        cam_T_velos = feed["cam_T_velos"]
        boxlist_camRs = feed["boxes3D"]
        tidlist_s = feed["tids"] # coordinate-less and plural
        scorelist_s = feed["scores"] # coordinate-less and plural
        # # postproc the boxes:
        # scorelist_s = __u(utils_misc.rescore_boxlist_with_inbound(__p(boxlist_camRs), __p(tidlist_s), Z, Y, X))
        boxlist_camRs_, tidlist_s_, scorelist_s_ = __p(boxlist_camRs), __p(tidlist_s), __p(scorelist_s)
        boxlist_camRs_, tidlist_s_, scorelist_s_ = utils_misc.shuffle_valid_and_sink_invalid_boxes(
            boxlist_camRs_, tidlist_s_, scorelist_s_)
        boxlist_camRs = __u(boxlist_camRs_)
        tidlist_s = __u(tidlist_s_)
        scorelist_s = __u(scorelist_s_)
        

        origin_T_camRs = feed["origin_T_camRs"]
        origin_T_camRs_ = __p(origin_T_camRs)
        origin_T_camXs = feed["origin_T_camXs"]
        origin_T_camXs_ = __p(origin_T_camXs)

        camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
        camX0_T_camXs_ = __p(camX0_T_camXs)
        camRs_T_camXs_ = torch.matmul(origin_T_camRs_.inverse(), origin_T_camXs_)
        camXs_T_camRs_ = camRs_T_camXs_.inverse()
        camRs_T_camXs = __u(camRs_T_camXs_)
        camXs_T_camRs = __u(camXs_T_camRs_)

        xyz_veloXs = feed["xyz_veloXs"]
        xyz_camXs = __u(utils_geom.apply_4x4(__p(cam_T_velos), __p(xyz_veloXs)))
        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))

        occRs = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z, Y, X))
        occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
        occX0s = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z, Y, X))

        occRs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z2, Y2, X2))
        occXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))
        occX0s_half = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z2, Y2, X2))

        unpRs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(torch.matmul(pix_T_cams,camXs_T_camRs))))
        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))
        unpX0s = utils_vox.apply_4x4_to_voxs(camX0_T_camXs, unpXs)
 
        unpRs_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, __p(torch.matmul(pix_T_cams,camXs_T_camRs))))

        #####################
        ## visualize what we got
        #####################
        summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
        summ_writer.summ_occs('3D_inputs/occRs', torch.unbind(occRs, dim=1))
        summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occXs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpRs', torch.unbind(unpRs, dim=1), torch.unbind(occRs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpX0s', torch.unbind(unpX0s, dim=1), torch.unbind(occX0s, dim=1))
        
        lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(boxlist_camRs_)).reshape(B, S, N, 19)
        lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(camXs_T_camRs), __p(lrtlist_camRs)))
        # stabilize boxes for ego/cam motion
        lrtlist_camX0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(camX0_T_camXs), __p(lrtlist_camXs)))
        # these are is B x S x N x 19
        
        summ_writer.summ_lrtlist('lrtlist_camR0', rgb_camRs[:,0], lrtlist_camRs[:,0],
                                 scorelist_s[:,0], tidlist_s[:,0], pix_T_cams[:,0])
        summ_writer.summ_lrtlist('lrtlist_camR1', rgb_camRs[:,1], lrtlist_camRs[:,1],
                                 scorelist_s[:,1], tidlist_s[:,1], pix_T_cams[:,1])
        summ_writer.summ_lrtlist('lrtlist_camX0', rgb_camXs[:,0], lrtlist_camXs[:,0],
                                 scorelist_s[:,0], tidlist_s[:,0], pix_T_cams[:,0])
        summ_writer.summ_lrtlist('lrtlist_camX1', rgb_camXs[:,1], lrtlist_camXs[:,1],
                                 scorelist_s[:,1], tidlist_s[:,1], pix_T_cams[:,1])
        (obj_lrtlist_camXs,
         obj_scorelist_s,
        ) = utils_misc.collect_object_info(lrtlist_camXs,
                                           tidlist_s,
                                           scorelist_s,
                                           pix_T_cams, 
                                           K, mod='X',
                                           do_vis=True,
                                           summ_writer=summ_writer)
        (obj_lrtlist_camRs,
         obj_scorelist_s,
        ) = utils_misc.collect_object_info(lrtlist_camRs,
                                           tidlist_s,
                                           scorelist_s,
                                           pix_T_cams, 
                                           K, mod='R',
                                           do_vis=True,
                                           summ_writer=summ_writer)
        (obj_lrtlist_camX0s,
         obj_scorelist_s,
        ) = utils_misc.collect_object_info(lrtlist_camX0s,
                                           tidlist_s,
                                           scorelist_s,
                                           pix_T_cams, 
                                           K, mod='X0',
                                           do_vis=False)

        masklist_memR = utils_vox.assemble_padded_obj_masklist(
            lrtlist_camRs[:,0], scorelist_s[:,0], Z, Y, X, coeff=1.0)
        masklist_memX = utils_vox.assemble_padded_obj_masklist(
            lrtlist_camXs[:,0], scorelist_s[:,0], Z, Y, X, coeff=1.0)
        # obj_mask_memR is B x N x 1 x Z x Y x X
        summ_writer.summ_occ('obj/masklist_memR', torch.sum(masklist_memR, dim=1))
        summ_writer.summ_occ('obj/masklist_memX', torch.sum(masklist_memX, dim=1))

        # to do tracking or whatever, i need to be able to extract a 3d object crop
        cropX0_obj0 = utils_vox.crop_zoom_from_mem(occXs[:,0], lrtlist_camXs[:,0,0], Z2, Y2, X2)
        cropX0_obj1 = utils_vox.crop_zoom_from_mem(occXs[:,0], lrtlist_camXs[:,0,1], Z2, Y2, X2)
        cropR0_obj0 = utils_vox.crop_zoom_from_mem(occRs[:,0], lrtlist_camRs[:,0,0], Z2, Y2, X2)
        cropR0_obj1 = utils_vox.crop_zoom_from_mem(occRs[:,0], lrtlist_camRs[:,0,1], Z2, Y2, X2)
        # print('got it:')
        # print(cropX00.shape)
        # summ_writer.summ_occ('crops/cropX0_obj0', cropX0_obj0)
        # summ_writer.summ_occ('crops/cropX0_obj1', cropX0_obj1)
        summ_writer.summ_feat('crops/cropX0_obj0', cropX0_obj0, pca=False)
        summ_writer.summ_feat('crops/cropX0_obj1', cropX0_obj1, pca=False)
        summ_writer.summ_feat('crops/cropR0_obj0', cropR0_obj0, pca=False)
        summ_writer.summ_feat('crops/cropR0_obj1', cropR0_obj1, pca=False)

        if hyp.do_feat:
            if hyp.flow_do_synth_rt:
                result = utils_misc.get_synth_flow(unpRs_half,
                                                   occRs_half,
                                                   obj_lrtlist_camX0s,
                                                   obj_scorelist_s,
                                                   occXs_half,
                                                   feed['set_name'],
                                                   K=K,
                                                   summ_writer=summ_writer,
                                                   sometimes_zero=True,
                                                   sometimes_real=False)
                occXs,unpXs,flowX0,camX1_T_camX0,is_synth = result
            else:
                # ego-stabilized flow from X00 to X01
                flowX0 = utils_misc.get_gt_flow(obj_lrtlist_camX0s,
                                                obj_scorelist_s,
                                                utils_geom.eye_4x4s(B, S),
                                                occXs_half[:,0],
                                                K=K, 
                                                occ_only=False, # get the dense flow
                                                mod='X0',
                                                summ_writer=summ_writer)

            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D
            # featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)
            featX0s_input = torch.cat([occX0s, occX0s*unpX0s], dim=2)
            featX0s_input_ = __p(featX0s_input)
            featX0s_, validX0s_, feat_loss = self.featnet(featX0s_input_, summ_writer)
            total_loss += feat_loss
            featX0s = __u(featX0s_)
            # _featX00 = featXs[:,0:1]
            # _featX01 = utils_vox.apply_4x4_to_voxs(camX0_T_camXs[:,1:], featXs[:,1:])
            # featX0s = torch.cat([_featX00, _featX01], dim=1)

            validX0s = 1.0 - (featX0s==0).all(dim=2, keepdim=True).float() #this shall be B x S x 1 x H x W x D

            summ_writer.summ_feats('3D_feats/featX0s_input', torch.unbind(featX0s_input, dim=1), pca=True)
            # summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featX0s_output', torch.unbind(featX0s, dim=1), pca=True)
        
        if hyp.do_flow:
            # total flow from X0 to X1
            flowX = utils_misc.get_gt_flow(obj_lrtlist_camXs,
                                           obj_scorelist_s,
                                           camX0_T_camXs,
                                           occXs_half[:,0],
                                           K=K, 
                                           occ_only=False, # get the dense flow
                                           mod='X',
                                           vis=False,
                                           summ_writer=None)
            
            # # vis this to confirm it's ok (it is)
            # unpX0_e = utils_samp.backwarp_using_3D_flow(unpXs[:,1], flowX)
            # occX0_e = utils_samp.backwarp_using_3D_flow(occXs[:,1], flowX)
            # summ_writer.summ_unps('flow/backwarpX', [unpX0s[:,0], unpX0_e], [occXs[:,0], occX0_e])

            # unpX0_e = utils_samp.backwarp_using_3D_flow(unpX0s[:,1], flowX0)
            # occX0_e = utils_samp.backwarp_using_3D_flow(occX0s[:,1], flowX0, binary_feat=True)
            # summ_writer.summ_unps('flow/backwarpX0', [unpX0s[:,0], unpX0_e], [occXs[:,0], occX0_e])

            flow_loss, flowX0_pred = self.flownet(
                featX0s[:,0],
                featX0s[:,1],
                flowX0, # gt flow
                torch.max(validX0s[:,1:], dim=1)[0],
                is_synth,
                summ_writer)
            total_loss += flow_loss

            # g = flowX.reshape(-1)
            # summ_writer.summ_histogram('flowX_g_nonzero_hist', g[torch.abs(g)>0.01])
            
            # g = flowX0.reshape(-1)
            # e = flowX0_pred.reshape(-1)
            # summ_writer.summ_histogram('flowX0_g_nonzero_hist', g[torch.abs(g)>0.01])
            # summ_writer.summ_histogram('flowX0_e_nonzero_hist', e[torch.abs(g)>0.01])
            
        summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results

    
