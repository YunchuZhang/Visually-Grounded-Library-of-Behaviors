import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import archs.encoder3D2D as encoder3D2D

import utils

class ViewNet(nn.Module):
    def __init__(self, config):
        super(ViewNet, self).__init__()

        print('ViewNet...')
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device

        self.med_dim = 32
        self.config = config

        self.net = encoder3D2D.Net3D2D(in_chans=config.feat_dim, mid_chans=32, out_chans=self.med_dim, depth=config.view_depth).to(device=device)
        self.emb_layer = nn.Conv2d(in_channels=self.med_dim, out_channels=config.feat_dim, kernel_size=1, stride=1, padding=0).to(device=device)
        self.rgb_layer = nn.Conv2d(in_channels=self.med_dim, out_channels=3, kernel_size=1, stride=1, padding=0).to(device=device)
        print('rgb_layer, [in_channels={}, out_channels={}, ksize={}]'.format(config.feat_dim, 1, 1))

    def forward(self, feat, rgb_g, summ_writer=None, set_num=None):
        total_loss = torch.tensor(0.0).to(device=self.device)
        
        feat = self.net(feat)
        emb_e = self.emb_layer(feat)
        rgb_e = self.rgb_layer(feat)
        # postproc
        emb_e = utils.basic.l2_normalize(emb_e, dim=1)
        rgb_e = torch.nn.functional.tanh(rgb_e) * 0.5

        loss_im = utils.basic.l2_on_axis(rgb_e-rgb_g, 1, keepdim=True)
        if summ_writer:
            summ_writer.summ_oned('view/rgb_loss', loss_im)
        rgb_loss = torch.mean(loss_im)

        total_loss = utils.misc.add_loss('view/rgb_l1_loss', total_loss, rgb_loss, self.config.view_l1_coeff, summ_writer)

        # vis
        if summ_writer:
            if set_num is not None and set_num == 1:
                summ_writer.summ_rgbs('view_val/rgb', [rgb_e, rgb_g])
            else:
                summ_writer.summ_rgbs('view/rgb', [rgb_e, rgb_g])
        return total_loss, rgb_e, emb_e
