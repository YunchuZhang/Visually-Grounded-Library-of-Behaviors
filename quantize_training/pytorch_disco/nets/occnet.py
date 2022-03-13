import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D

import utils


class OccNet(nn.Module):
    def __init__(self, config):
        super(OccNet, self).__init__()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        print('OccNet...')
        self.occ_smooth_coeff = config.occ_smooth_coeff
        self.occ_coeff = config.occ_coeff

        if not config.occ_do_cheap:
            self.conv3d = nn.Sequential(
            archs.encoder3D.Net3D(in_channel=4, pred_dim=8),
            nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0),
            ).to(device=device)

        else:
            self.conv3d = nn.Conv3d(in_channels=config.feat_dim, out_channels=1, kernel_size=1, stride=1, padding=0).to(device=device)
            print('conv3D, [in_channels={}, out_channels={}, ksize={}]'.format(config.feat_dim, 1, 1))

    def forward(self, feat, occ_g, free_g, valid, summ_writer, set_num=None):
        total_loss = torch.tensor(0.0).to(device=self.device)

        occ_e_ = self.conv3d(feat)
        # occ_e_ is B x 1 x Z x Y x X

        # smooth loss
        dz, dy, dx = utils.basic.gradient3D(occ_e_, absolute=True)
        smooth_vox = torch.mean(dx+dy+dx, dim=1, keepdims=True)

        summ_writer.summ_oned('occ/smooth_loss', torch.mean(smooth_vox, dim=3))
        smooth_loss = torch.mean(smooth_vox)

        total_loss = utils.misc.add_loss('occ/smooth_loss', total_loss, smooth_loss, self.occ_smooth_coeff, summ_writer)

        occ_e = F.sigmoid(occ_e_)
        occ_e_binary = torch.round(occ_e)

        # collect some accuracy stats
        occ_match = occ_g*torch.eq(occ_e_binary, occ_g).float()
        free_match = free_g*torch.eq(1.0-occ_e_binary, free_g).float()
        either_match = torch.clamp(occ_match+free_match, 0.0, 1.0)
        either_have = torch.clamp(occ_g+free_g, 0.0, 1.0)
        acc_occ = utils.basic.reduce_masked_mean(occ_match, occ_g*valid)
        acc_free = utils.basic.reduce_masked_mean(free_match, free_g*valid)
        acc_total = utils.basic.reduce_masked_mean(either_match, either_have*valid)

        summ_writer.summ_scalar('occ/acc_occ', acc_occ.cpu().item())
        summ_writer.summ_scalar('occ/acc_free', acc_free.cpu().item())
        summ_writer.summ_scalar('occ/acc_total', acc_total.cpu().item())

        # vis
        if set_num is not None and set_num == 1:
            summ_writer.summ_occ('occ_val/occ_g', occ_g, reduce_axes=[2])
            summ_writer.summ_occ('occ_val/free_g', free_g, reduce_axes=[2])
            summ_writer.summ_occ('occ_val/occ_e', occ_e, reduce_axes=[2])
            summ_writer.summ_occ('occ_val/valid', valid, reduce_axes=[2])
        else:
            summ_writer.summ_occ('occ/occ_g', occ_g, reduce_axes=[2])
            summ_writer.summ_occ('occ/free_g', free_g, reduce_axes=[2])
            summ_writer.summ_occ('occ/occ_e', occ_e, reduce_axes=[2])
            summ_writer.summ_occ('occ/valid', valid, reduce_axes=[2])

        prob_loss = self.compute_loss(occ_e_, occ_g, free_g, valid, summ_writer)
        total_loss = utils.misc.add_loss('occ/prob_loss', total_loss, prob_loss, self.occ_coeff, summ_writer)

        return total_loss, occ_e

    def compute_loss(self, pred, occ, free, valid, summ_writer):
        pos = occ.clone()
        neg = free.clone()

        # occ is B x 1 x Z x Y x X

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        loss_vis = torch.mean(loss*mask_*valid, dim=3)
        summ_writer.summ_oned('occ/prob_loss', loss_vis, summ_writer)

        pos_loss = utils.basic.reduce_masked_mean(loss, pos*valid)
        neg_loss = utils.basic.reduce_masked_mean(loss, neg*valid)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss

