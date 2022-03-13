import torch
import torch.nn as nn
import numpy as np

from model_base import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.viewnet import ViewNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

import torch.nn.functional as F

from utils_basic import *
import utils_vox
import utils_samp
import utils_geom
import utils_improc

np.set_printoptions(precision=2)
np.random.seed(0)

class CARLA_STA(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def infer(self):
        print("------ BUILDING INFERENCE GRAPH ------")
        # self.model = CarlaStaNet().to(self.device)
        self.model = CarlaStaNet()

class CarlaStaNet(nn.Module):
    def __init__(self):
        super(CarlaStaNet, self).__init__()
        if hyp.do_feat:
            self.featnet = FeatNet()
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_view:
            self.viewnet = ViewNet()
        if hyp.do_emb2D:
            self.embnet2D = EmbNet2D()
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    def forward(self, feed):
        results = dict()
        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8)
        writer = feed['writer']
        global_step = feed['global_step']

        total_loss = torch.tensor(0.0).cuda()

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

        origin_T_camRs = feed["origin_T_camRs"]
        origin_T_camRs_ = __p(origin_T_camRs)
        origin_T_camXs = feed["origin_T_camXs"]
        origin_T_camXs_ = __p(origin_T_camXs)

        camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
        camX0_T_camXs_ = __p(camX0_T_camXs)
        camRs_T_camXs_ = torch.matmul(utils_geom.safe_inverse(origin_T_camRs_), origin_T_camXs_)
        camXs_T_camRs_ = utils_geom.safe_inverse(camRs_T_camXs_)
        camRs_T_camXs = __u(camRs_T_camXs_)
        camXs_T_camRs = __u(camXs_T_camRs_)

        xyz_veloXs = feed["xyz_veloXs"]
        xyz_camXs = __u(utils_geom.apply_4x4(__p(cam_T_velos), __p(xyz_veloXs)))
        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))
        
        occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
        occXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))
        occX0s_half = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z2, Y2, X2))

        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))

        ## projected depth, and inbound mask
        depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
        dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))
        dense_xyz_camX0s_ = utils_geom.apply_4x4(__p(camX0_T_camXs), dense_xyz_camXs_)
        inbound_camXs_ = utils_vox.get_inbounds(dense_xyz_camX0s_, Z, Y, X).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])
        
        depth_camXs = __u(depth_camXs_)
        valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)
        
        #####################
        ## visualize what we got
        #####################
        summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(depth_camXs, dim=1))
        summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
        summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
        summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occXs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))
        if summ_writer.save_this:
            unpRs = __u(utils_vox.unproject_rgb_to_mem(
                __p(rgb_camXs), Z, Y, X, matmul2(__p(pix_T_cams), utils_geom.safe_inverse(__p(camRs_T_camXs)))))
            occRs = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z, Y, X))
            summ_writer.summ_occs('3D_inputs/occRs', torch.unbind(occRs, dim=1))
            summ_writer.summ_unps('3D_inputs/unpRs', torch.unbind(unpRs, dim=1), torch.unbind(occRs, dim=1))

        #####################
        ## run the nets
        #####################

        mask_ = None
        if hyp.do_occ and (not hyp.occ_do_cheap):
            '''
            occRs_sup, freeRs_sup, freeXs = utils_vox.prep_occs_supervision(xyz_camXs,
                                                                            occRs_half,
                                                                            occXs_half,
                                                                            camRs_T_camXs,
                                                                            agg=True)
            
            featRs_input = torch.cat([occRs, occRs*unpRs], dim=2)
            featRs_input_ = __p(featRs_input)
            occRs_sup_ = __p(occRs_sup)
            freeRs_sup_ = __p(freeRs_sup)
            occ_loss, occRs_pred_ = self.occnet(featRs_input_,
                                                occRs_sup_,
                                                freeRs_sup_,
                                                summ_writer
            )
            occRs_pred = __u(occRs_pred_)
            total_loss += occ_loss
            
            mask_ = F.upsample(occRs_pred_, scale_factor=2)
            '''
            occXs_ = __p(occXs)
            mask_ = occXs_


        if hyp.do_feat:
            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D
            featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)
            featXs_input_ = __p(featXs_input)

            # it is useful to keep track of what was visible from each viewpoint
            freeXs_ = utils_vox.get_freespace(__p(xyz_camXs), __p(occXs_half))
            freeXs = __u(freeXs_)
            visXs = torch.clamp(occXs_half+freeXs, 0.0, 1.0)
            
            if(type(mask_)!=type(None)):
                assert(list(mask_.shape)[2:5]==list(featXs_input_.shape)[2:5])
            featXs_, validXs_, feat_loss = self.featnet(featXs_input_, summ_writer, mask=__p(occXs))#mask_)
            total_loss += feat_loss
            
            validXs = __u(validXs_)
            _validX00 = validXs[:,0:1]
            _validX01 = utils_vox.apply_4x4_to_voxs(camX0_T_camXs[:,1:], validXs[:,1:])
            validX0s = torch.cat([_validX00, _validX01], dim=1)
            
            _visX00 = visXs[:,0:1]
            _visX01 = utils_vox.apply_4x4_to_voxs(camX0_T_camXs[:,1:], visXs[:,1:])
            visX0s = torch.cat([_visX00, _visX01], dim=1)
            
            featXs = __u(featXs_)
            _featX00 = featXs[:,0:1]
            _featX01 = utils_vox.apply_4x4_to_voxs(camX0_T_camXs[:,1:], featXs[:,1:])
            featX0s = torch.cat([_featX00, _featX01], dim=1)

            emb3D_e = torch.mean(featX0s[:,1:], dim=1) # context
            emb3D_g = featX0s[:,0] # obs
            vis3D_e = torch.max(validX0s[:,1:], dim=1)[0]*torch.max(visX0s[:,1:], dim=1)[0]
            vis3D_g = validX0s[:,0]*visX0s[:,0] # obs

            if hyp.do_eval_recall:
                results['emb3D_e'] = emb3D_e
                results['emb3D_g'] = emb3D_g

            summ_writer.summ_feats('3D_feats/featXs_input', torch.unbind(featXs_input, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featX0s_output', torch.unbind(featX0s, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/validX0s', torch.unbind(validX0s, dim=1), pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_e', vis3D_e, pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_g', vis3D_g, pca=False)
            
        if hyp.do_occ and hyp.occ_do_cheap:
            occX0_sup, freeX0_sup, freeXs = utils_vox.prep_occs_supervision(
                xyz_camXs,
                occX0s_half,
                occXs_half,
                camX0_T_camXs,
                agg=True)
        
            summ_writer.summ_occ('occ_sup/occ_sup', occX0_sup)
            summ_writer.summ_occ('occ_sup/free_sup', freeX0_sup)
            summ_writer.summ_occs('occ_sup/freeXs_sup', torch.unbind(freeXs, dim=1))
            summ_writer.summ_occs('occ_sup/occXs_sup', torch.unbind(occXs_half, dim=1))
                
            occ_loss, occRs_pred_ = self.occnet(torch.mean(featX0s[:,1:], dim=1),
                                                occX0_sup,
                                                freeX0_sup,
                                                torch.max(validX0s[:,1:], dim=1)[0],
                                                summ_writer)
            occRs_pred = __u(occRs_pred_)
            total_loss += occ_loss

        if hyp.do_view:
            assert(hyp.do_feat)
            # we warped the features into the canonical view
            # now we resample to the target view and decode

            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5) # else we need a fancier downsampler
            assert(sy==0.5)
            projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(pix_T_cams), sx, sy))

            assert(S==2) # else we should warp each feat in 1:
            feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], camX0_T_camXs[:,1], featXs[:,1],
                hyp.view_depth, PH, PW)
            # feat_projX0 is B x hyp.feat_dim x hyp.view_depth x PH x PW
            rgb_X00 = downsample(rgb_camXs[:,0], 2)

            if summ_writer.save_this:
                # for vis, let's also project some rgb
                rgb_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camXs_T_camRs[:,0], unpRs[:,0],
                    hyp.view_depth, PH, PW)
                rgb_projX01 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,1], camXs_T_camRs[:,1], unpRs[:,1],
                    hyp.view_depth, PH, PW)
                occ_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camXs_T_camRs[:,0], occRs[:,0],
                    hyp.view_depth, PH, PW)
                occ_projX01 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,1], camXs_T_camRs[:,1], occRs[:,1],
                    hyp.view_depth, PH, PW)
                rgb_projX00_vis = reduce_masked_mean(rgb_projX00, occ_projX00.repeat([1, 3, 1, 1, 1]), dim=2)
                rgb_projX01_vis = reduce_masked_mean(rgb_projX01, occ_projX01.repeat([1, 3, 1, 1, 1]), dim=2)
                summ_writer.summ_rgbs('projection/rgb_projX', [rgb_projX00_vis, rgb_projX01_vis])
                rgb_X01 = downsample(rgb_camXs[:,1], 2)
                summ_writer.summ_rgbs('projection/rgb_origX', [rgb_X00, rgb_X01])

            # decode the perspective volume into an image
            view_loss, rgb_e, emb2D_e = self.viewnet(
                feat_projX00,
                rgb_X00,
                summ_writer)
            total_loss += view_loss
            
        if hyp.do_emb2D:
            assert(hyp.do_view)
            # create an embedding image, representing the bottom-up 2D feature tensor

            emb_loss_2D, emb2D_g = self.embnet2D(
                rgb_camXs[:,0],
                emb2D_e,
                valid_camXs[:,0],
                summ_writer)
            total_loss += emb_loss_2D

        if hyp.do_emb3D:
            occX0_sup, freeX0_sup, freeXs = utils_vox.prep_occs_supervision(
                xyz_camXs,
                occX0s_half,
                occXs_half,
                camX0_T_camXs,
                agg=True)
            
            emb_loss_3D = self.embnet3D(
                emb3D_e,
                emb3D_g,
                vis3D_e,
                vis3D_g,
                summ_writer)
            total_loss += emb_loss_3D

        if hyp.do_eval_recall:
            results['emb2D_e'] = None
            results['emb2D_g'] = None
            
        summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results

