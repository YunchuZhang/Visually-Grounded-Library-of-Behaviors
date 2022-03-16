import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler

# import sys
# sys.path.append("..")

import archs.encoder3D

import improc
import utils

class FlowNet(nn.Module):
    def __init__(self):
        super(FlowNet, self).__init__()

        print('FlowNet...')

        self.debug = False
        # self.debug = True
        
        self.heatmap_size = hyp.flow_patch_size
        # self.compress_dim = 8
        # self.scales = [0.25, 0.5, 1.0]
        self.scales = [0.125, 0.25, 0.5, 0.75, 1.0]
        self.num_scales = len(self.scales)
        
        # # slightly diff from flownet, here i am using one set of params across all scales
        # self.compressor = nn.Sequential(
        #     nn.Conv3d(in_channels=32, out_channels=self.compress_dim, kernel_size=1, stride=1, padding=0),
        # )
        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.heatmap_size,
            stride=1,
            padding=0,
            dilation_patch=1,
        )
        self.flow_predictor = nn.Sequential(
            nn.Conv3d(in_channels=(self.heatmap_size**3), out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0),
        )
        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='none')
        self.smoothl1_mean = torch.nn.SmoothL1Loss(reduction='mean')

    def generate_flow(self, feat0, feat1, sc):
        B, C, D, H, W = list(feat0.shape)
        utils.basic.assert_same_shape(feat0, feat1)

        if self.debug:
            print('scale = %.2f' % sc)
            print('inputs:')
            print(feat0.shape)
            print(feat1.shape)

        if not sc==1.0:
            # assert(sc==0.5 or sc==0.25) # please only use 0.25, 0.5, or 1.0 right now
            feat0 = F.interpolate(feat0, scale_factor=sc, mode='trilinear')
            feat1 = F.interpolate(feat1, scale_factor=sc, mode='trilinear')
            D, H, W = int(D*sc), int(H*sc), int(W*sc)
            if self.debug:
                print('downsamps:')
                print(feat0.shape)
                print(feat1.shape)

        feat0 = feat0.contiguous()
        feat1 = feat1.contiguous()
        cc = self.correlation_sampler(feat0, feat1)
        if self.debug:
            print('cc:')
            print(cc.shape)
        cc = cc.view(B, self.heatmap_size**3, D, H, W)

        cc = F.relu(cc) # relu works better than leaky relu here
        if self.debug:
            print(cc.shape)
        cc = utils.basic.l2_normalize(cc, dim=1)

        flow = self.flow_predictor(cc)
        if self.debug:
            print('flow:')
            print(flow.shape)
        
        # flow = flow[:,max_disp:-max_disp,max_disp:-max_disp,max_disp:-max_disp]
        # flow = tf.pad(flow, [[0,0], [max_disp,max_disp], [max_disp,max_disp], [max_disp,max_disp], [0,0]], 'CONSTANT')
        # # pad_mask = 1.0-tf.cast(tf.equal(flow[:,:,:,:,0:1], 0.0), tf.float32)

        if not sc==1.0:
            # note 1px here means 1px/sc at the real scale
            # first let's put the pixels in the right places
            flow = F.interpolate(flow, scale_factor=(1./sc), mode='trilinear')
            # now let's correct the scale
            flow = flow/sc

        if self.debug:
            print('flow up:')
            print(flow.shape)
            
        return flow

    def forward(self, feat0, feat1, flow_g, mask_g, is_synth, summ_writer):
        total_loss = torch.tensor(0.0)

        B, C, D, H, W = list(feat0.shape)
        utils.basic.assert_same_shape(feat0, feat1)

        # feats = torch.cat([feat0, feat1], dim=0)
        # feats = self.compressor(feats)
        # feats = utils_basic.l2_normalize(feats, dim=1)
        # feat0, feat1 = feats[:B], feats[B:]

        flow_total_forw = torch.zeros(B, 3, D, H, W)
        flow_total_back = torch.zeros(B, 3, D, H, W)

        feat0_aligned = feat0.clone()
        feat1_aligned = feat1.clone()

        # cycle_losses = []
        # l1_losses = []

        # torch does not like it when we overwrite, so let's pre-allocate
        l1_loss = torch.tensor(0.0)
        
        for sc in self.scales:

            flow_forw = self.generate_flow(feat0, feat1_aligned, sc)
            flow_back = self.generate_flow(feat1, feat0_aligned, sc)

            flow_total_forw = flow_total_forw + flow_forw
            flow_total_back = flow_total_back + flow_back

            # compositional LK: warp the original thing using the cumulative flow
            feat1_aligned = utils.samp.backwarp_using_3D_flow(feat1, flow_total_forw)
            feat0_aligned = utils.samp.backwarp_using_3D_flow(feat0, flow_total_back)

            l1_diff_3chan = self.smoothl1(flow_total_forw, flow_g)
            l1_diff = torch.mean(l1_diff_3chan, dim=1, keepdim=True)

            nonzero_mask = (torch.sum(torch.abs(flow_g), axis=1, keepdim=True) > 0.01).float()
            yeszero_mask = 1.0-nonzero_mask
            l1_loss_nonzero = utils.basic.reduce_masked_mean(l1_diff, nonzero_mask)
            l1_loss_yeszero = utils.basic.reduce_masked_mean(l1_diff, yeszero_mask)
            l1_loss_balanced = (l1_loss_nonzero + l1_loss_yeszero)*0.5
            l1_loss = l1_loss + l1_loss_balanced*sc

            synth_l1_loss = l1_loss*is_synth
            total_loss = utils.misc.add_loss('flow/l1_synth_loss',total_loss,synth_l1_loss, 
                                             hyp.flow_synth_l1_coeff,summ_writer)
            
            if sc==1.0:
                # warp flow
                flow_back_aligned_to_forw = utils.samp.backwarp_using_3D_flow(flow_total_back, flow_total_forw.detach())
                flow_forw_aligned_to_back = utils.samp.backwarp_using_3D_flow(flow_total_forw, flow_total_back.detach())

                cancelled_flow_forw = flow_total_forw + flow_back_aligned_to_forw
                cancelled_flow_back = flow_total_back + flow_forw_aligned_to_back

                cycle_forw = self.smoothl1_mean(cancelled_flow_forw, torch.zeros_like(cancelled_flow_forw))
                cycle_back = self.smoothl1_mean(cancelled_flow_back, torch.zeros_like(cancelled_flow_back))
                cycle_loss = cycle_forw + cycle_back
                total_loss = utils.misc.add_loss('flow/cycle_loss', total_loss, cycle_loss, hyp.flow_cycle_coeff, summ_writer)

                summ_writer.summ_3D_flow('flow/flow_e_%.2f' % sc, flow_total_forw, clip=0.0)
                summ_writer.summ_3D_flow('flow/flow_g_%.2f' % sc, flow_g, clip=0.0)

                # l1_losses.append(l1_loss_balanced*sc)
                if sc==1.0:
                    utils.misc.add_loss('flow/l1_loss_nonzero', 0, l1_loss_nonzero, 0, summ_writer)
                    utils.misc.add_loss('flow/l1_loss_yeszero', 0, l1_loss_yeszero, 0, summ_writer)
                    utils.misc.add_loss('flow/l1_loss_balanced', 0, l1_loss_balanced, 0, summ_writer)
                
                # total_loss = utils_misc.add_loss('flow/l1_loss_balanced', total_loss, l1_loss_balanced, hyp.flow_l1_coeff, summ_writer)
                # total_loss = utils_misc.add_loss('flow/l1_loss_balanced', total_loss, l1_loss_balanced, hyp.flow_l1_coeff, summ_writer)
                # total_loss = utils_misc.add_loss('flow/l1_loss', total_loss, l1_loss, hyp.flow_l1_coeff*(sc==1.0), summ_writer)

        total_loss = utils.misc.add_loss('flow/l1_loss', total_loss, l1_loss, hyp.flow_l1_coeff, summ_writer)
        # total_loss = utils_misc.add_loss('flow/cycle_loss', total_loss, torch.sum(torch.stack(cycle_losses)), hyp.flow_cycle_coeff, summ_writer)
        # total_loss = utils_misc.add_loss('flow/l1_loss', total_loss, torch.sum(torch.stack(l1_losses)), hyp.flow_l1_coeff, summ_writer)
        
        # # smooth loss
        # dx, dy, dz = gradient3D(flow_e_, absolute=True)
        # smooth_vox = torch.mean(dx+dy+dx, dim=1, keepdims=True)
        
        # summ_writer.summ_oned('flow/smooth_loss', torch.mean(smooth_vox, dim=3))
        # smooth_loss = torch.mean(smooth_vox)

        # total_loss = utils_misc.add_loss('flow/smooth_loss', total_loss, smooth_loss, hyp.flow_smooth_coeff, summ_writer)
    
        # flow_e = F.sigmoid(flow_e_)
        # flow_e_binary = torch.round(flow_e)

        # # collect some accuracy stats 
        # flow_match = flow_g*torch.eq(flow_e_binary, flow_g).float()
        # free_match = free_g*torch.eq(1.0-flow_e_binary, free_g).float()
        # either_match = torch.clamp(flow_match+free_match, 0.0, 1.0)
        # either_have = torch.clamp(flow_g+free_g, 0.0, 1.0)
        # acc_flow = reduce_masked_mean(flow_match, flow_g*valid)
        # acc_free = reduce_masked_mean(free_match, free_g*valid)
        # acc_total = reduce_masked_mean(either_match, either_have*valid)

        # summ_writer.summ_scalar('flow/acc_flow', acc_flow.cpu().item())
        # summ_writer.summ_scalar('flow/acc_free', acc_free.cpu().item())
        # summ_writer.summ_scalar('flow/acc_total', acc_total.cpu().item())

        # # vis
        # summ_writer.summ_flow('flow/flow_g', flow_g)
        # summ_writer.summ_flow('flow/free_g', free_g) 
        # summ_writer.summ_flow('flow/flow_e', flow_e)
        # summ_writer.summ_flow('flow/valid', valid)
        
        # prob_loss = self.compute_loss(flow_e_, flow_g, free_g, valid, summ_writer)
        # total_loss = utils_misc.add_loss('flow/prob_loss', total_loss, prob_loss, hyp.flow_coeff, summ_writer)

        # return total_loss, flow_e
        return total_loss, flow_total_forw

