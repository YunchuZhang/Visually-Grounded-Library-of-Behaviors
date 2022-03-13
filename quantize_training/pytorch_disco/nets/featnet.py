import torch
import torch.nn as nn

import sys
sys.path.append("..")

#import hyperparams as hyp
import archs.encoder3D as encoder3D



#if hyp.feat_do_sb:
#    import archs.sparse_encoder3D as sparse_encoder3D
#if hyp.feat_do_sparse_invar:
#    import archs.sparse_invar_encoder3D as sparse_invar_encoder3D
import utils

EPS = 1e-4
class FeatNet(nn.Module):
    def __init__(self, config, input_dim=4):
        super(FeatNet, self).__init__()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        self.config = config
        if config.feat_do_sb:
            import archs.sparse_encoder3D as sparse_encoder3D
            if config.feat_do_resnet:
                self.net = sparse_encoder3D.SparseResNet3D(in_channel=input_dim, pred_dim=config.feat_dim).to(device=device)
            else:
                self.net = sparse_encoder3D.SparseNet3D(in_channel=input_dim, pred_dim=config.feat_dim).to(device=device)
        else:
            import archs.sparse_invar_encoder3D as sparse_invar_encoder3D
            if config.feat_do_resnet:
                self.net = encoder3D.ResNet3D(in_channel=input_dim, pred_dim=config.feat_dim).to(device=device)
            elif config.feat_do_sparse_invar:
                # self.net = sparse_invar_encoder3D.ResNet3D(in_channel=4, pred_dim=hyp.feat_dim).cuda()
                self.net = sparse_invar_encoder3D.Custom3D(in_channel=input_dim, pred_dim=config.feat_dim).to(device=device)
            else:
                self.net = encoder3D.Net3D(in_channel=input_dim, pred_dim=config.feat_dim).to(device=device)
        # print(self.net.named_parameters)

    def forward(self, feat, summ_writer=None, mask=None, touch=False, set_num=None):
        total_loss = torch.tensor(0.0).to(device=self.device)
        B, C, D, H, W = list(feat.shape)

        name_to_log = 'feat/feat0_input' if not touch else 'feat/sensor_feat0_input'
        if set_num is not None and set_num == 1:
            name_to_log = 'feat_val/feat0_input'
        do_pca = True if not touch else False
        if summ_writer:
             summ_writer.summ_feat(name_to_log, feat, pca=do_pca)

        if self.config.feat_do_rt:
            # apply a random rt to the feat
            # Y_T_X = utils_geom.get_random_rt(B, r_amount=5.0, t_amount=8.0).cuda()
            # Y_T_X = utils_geom.get_random_rt(B, r_amount=1.0, t_amount=8.0).cuda()
            Y_T_X = utils.geom.get_random_rt(B, r_amount=1.0, t_amount=4.0).cuda()
            feat = utils.vox.apply_4x4_to_vox(Y_T_X, feat)
            name_to_log = 'feat/feat1_rt' if not touch else 'feat/sensor_feat1_rt'
            if summ_writer:
                 summ_writer.summ_feat(name_to_log, feat)

        if self.config.feat_do_flip:
            # randomly flip the input
            flip0 = torch.rand(1)
            flip1 = torch.rand(1)
            flip2 = torch.rand(1)
            if flip0 > 0.5:
                # transpose width/depth (rotate 90deg)
                feat = feat.permute(0,1,4,3,2)
            if flip1 > 0.5:
                # flip depth
                feat = feat.flip(2)
            if flip2 > 0.5:
                # flip width
                feat = feat.flip(4)
            name_to_log = 'feat/feat2_flip' if not touch else 'feat/sensor_feat2_flip'
            if summ_writer:
                 summ_writer.summ_feat(name_to_log, feat)

        if self.config.feat_do_sb:
            feat = self.net(feat, mask)
        elif self.config.feat_do_sparse_invar:
            feat, mask = self.net(feat, mask)
        else:
            feat = self.net(feat)
        feat = utils.basic.l2_normalize(feat, dim=1)
        name_to_log = 'feat/feat3_out' if not touch else 'feat/sensor_feat3_out'
        if set_num is not None and set_num == 1:
            name_to_log = 'feat_val/feat3_out'
        if summ_writer:
            summ_writer.summ_feat(name_to_log, feat)

        if self.config.feat_do_flip:
            if flip2 > 0.5:
                # unflip width
                feat = feat.flip(4)
            if flip1 > 0.5:
                # unflip depth
                feat = feat.flip(2)
            if flip0 > 0.5:
                # untranspose width/depth
                feat = feat.permute(0,1,4,3,2)
            name_to_log = 'feat/feat4_unflip' if not touch else 'feat/sensor_feat4_unflip'
            if summ_writer:
                summ_writer.summ_feat(name_to_log, feat)

        if self.config.feat_do_rt:
            # undo the random rt
            X_T_Y = utils.geom.safe_inverse(Y_T_X)
            feat = utils.vox.apply_4x4_to_vox(X_T_Y, feat)
            name_to_log = 'feat/feat5_unrt' if not touch else 'feat/sensor_feat5_unrt'
            if summ_writer:
                summ_writer.summ_feat(name_to_log, feat)

        valid_mask = 1.0 - (feat==0).all(dim=1, keepdim=True).float() #feat is B x C x D x H x W
        if self.config.feat_do_sparse_invar:
            valid_mask = valid_mask * mask

        return feat, valid_mask, total_loss
