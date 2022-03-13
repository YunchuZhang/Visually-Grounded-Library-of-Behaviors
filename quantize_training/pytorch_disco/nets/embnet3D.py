import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import utils

class EmbNet3D(nn.Module):
    def __init__(self):
        super(EmbNet3D, self).__init__()

        print('EmbNet3D...')
        self.batch_k = 2
        self.num_samples = hyp.emb_3D_num_samples  # TODO: I will keep a big number here
        assert(self.num_samples > 0)
        self.sampler = utils.misc.DistanceWeightedSampling(batch_k=self.batch_k, normalize=False)
        self.criterion = utils.misc.MarginLoss() #margin=args.margin,nu=args.nu)
        self.beta = 1.2

    def sample_embs(self, emb0, emb1, valid, B, Z, Y, X, mod='', do_vis=False, summ_writer=None):
        """
        parameters:
        -----------------
            emb0: sensor embedding tensor
            emb1: visual embedding tensor
            valid: visual vis tensor
            B=2, Z=32, Y=32, X=32
            mod = 'g' #TODO: understand what this means
            do_vis = True
            summ_writer : summary writer object
        """
        if hyp.emb_3D_mindist == 0.0:
            # assume for now that I keep it as zero
            # pure random
            perm = torch.randperm(B*Z*Y*X)
            emb0 = emb0.contiguous().view(B*Z*Y*X, -1)
            emb1 = emb1.contiguous().view(B*Z*Y*X, -1)
            valid = valid.view(B*Z*Y*X, -1)
            emb0 = emb0[perm[:self.num_samples*B]]
            emb1 = emb1[perm[:self.num_samples*B]]
            valid = valid[perm[:self.num_samples*B]]
            return emb0, emb1, valid
        else:
            emb0_all = []
            emb1_all = []
            valid_all = []
            for b in list(range(B)):
                sample_indices, sample_locs, sample_valids = utils.misc.get_safe_samples(
                    valid[b], (Z, Y, X), self.num_samples, mode='3D', tol=hyp.emb_3D_mindist)
                emb0_s_ = emb0[b, sample_indices]
                emb1_s_ = emb1[b, sample_indices]
                # these are N x D
                emb0_all.append(emb0_s_)
                emb1_all.append(emb1_s_)
                valid_all.append(sample_valids)

            if do_vis and (summ_writer is not None):
                sample_occ = utils.vox.voxelize_xyz(torch.unsqueeze(sample_locs, dim=0), Z, Y, X, already_mem=True)
                summ_writer.summ_occ('emb3D/samples_%s/sample_occ' % mod, sample_occ, reduce_axes=[2,3])
                summ_writer.summ_occ('emb3D/samples_%s/valid' % mod, torch.reshape(valid, [B, 1, Z, Y, X]), reduce_axes=[2,3])

            emb0_all = torch.cat(emb0_all, axis=0)
            emb1_all = torch.cat(emb1_all, axis=0)
            valid_all = torch.cat(valid_all, axis=0)
            return emb0_all, emb1_all, valid_all
        
    def compute_margin_loss(self, B, C, Z, Y, X, emb_e_vec, emb_g_vec, valid_vec, mod='', do_vis=False, summ_writer=None):
        """
        parameters: I have provided values as an example
        -------------
            B: batch size 2 (each is potentially a different object)
            C: 32 (channels, feature dimensions)
            Z: depth of voxel grid
            Y: height of voxel grid
            X: width of voxel grid
            emb_e_vec : sensor embedding tensor
            emb_g_vec : visual embedding tensor
            valid_vec : visual vis tensor
            mod : 'g' TODO: Understand/Ask what this means
            do_vis : True
            summ_writer : Summary writer object
        
        Function:
        ------------------
            This is the metric learning loss, positive samples distance should be minimized and
            negative samples distance should be maximized above a margin. if the distance is above
            the margin the loss is zero for the location.
        """
        emb_e_vec, emb_g_vec, valid_vec = self.sample_embs(emb_e_vec,
                                                           emb_g_vec,
                                                           valid_vec,
                                                           B, Z, Y, X,
                                                           mod=mod,
                                                           do_vis=do_vis,
                                                           summ_writer=summ_writer)
        emb_vec = torch.stack((emb_e_vec, emb_g_vec), dim=1).view(B*self.num_samples*self.batch_k, C)
        # this tensor goes e,g,e,g,... on dim 0
        # note this means 2 samples per class; batch_k=2
        y = torch.stack([torch.range(0,self.num_samples*B-1), torch.range(0,self.num_samples*B-1)], dim=1).view(self.num_samples*B*self.batch_k)
        # this tensor goes 0,0,1,1,2,2,...

        # for each e I have a positive of which I want to minimize the distance
        # and I have a negative from which I want to maximize the distance
        a_indices, anchors, positives, negatives, _ = self.sampler(emb_vec)
        margin_loss, _ = self.criterion(anchors, positives, negatives, self.beta, y[a_indices])
        return margin_loss
            
    def forward(self, emb_e, emb_g, vis_e, vis_g, summ_writer):
        total_loss = torch.tensor(0.0).cuda()

        # emb_e is sensor_embeddings, emb_g is visual_embeddings
        # if any of them goes to nan raise raise AssertionError
        if torch.isnan(emb_e).any() or torch.isnan(emb_g).any():
            assert(False)

        B, C, D, H, W = list(emb_e.shape)  # 2, 32, 32, 32, 32
        # put channels on the end
        emb_e_vec = emb_e.permute(0,2,3,4,1).reshape(B, D*H*W, C)  # sensor embedding converted to vector
        emb_g_vec = emb_g.permute(0,2,3,4,1).reshape(B, D*H*W, C)  # visual embedding converted to vector
        vis_e_vec = vis_e.permute(0,2,3,4,1).reshape(B, D*H*W, 1)  # sensor vis converting to vector
        vis_g_vec = vis_g.permute(0,2,3,4,1).reshape(B, D*H*W, 1)  # visual vis converted to vector

        # ensure they are both nonzero, else we probably masked or warped something
        valid_vec_e = 1.0 - (emb_e_vec==0).all(dim=2, keepdim=True).float()
        valid_vec_g = 1.0 - (emb_g_vec==0).all(dim=2, keepdim=True).float()
        valid_vec = valid_vec_e * valid_vec_g
        vis_e_vec *= valid_vec
        vis_g_vec *= valid_vec

        assert(self.num_samples < (B*D*H*W))
        # we will take num_samples from each one

        # ~18% of vis_e is on
        # ~25% of vis_g is on
        # print('it looks like %.2f of vis_e is 1' % (torch.sum(vis_e_vec).cpu()/len(vis_g_vec)))
        # print('it looks like %.2f of vis_g is 1' % (torch.sum(vis_g_vec).cpu()/len(vis_g_vec)))

        # where g is valid, we use it as reference and pull up e
        margin_loss = self.compute_margin_loss(B, C, D, H, W, emb_e_vec, emb_g_vec.detach(), vis_g_vec, 'g', True, summ_writer)
        l2_loss = reduce_masked_mean(sql2_on_axis(emb_e-emb_g.detach(), 1, keepdim=True), vis_g)
        total_loss = utils.misc.add_loss('emb3D/emb_3D_ml_loss', total_loss, margin_loss, hyp.emb_3D_ml_coeff, summ_writer)
        total_loss = utils.misc.add_loss('emb3D/emb_3D_l2_loss', total_loss, l2_loss, hyp.emb_3D_l2_coeff, summ_writer)
        
        # # where e is valid, we use it as reference and pull up g
        # margin_loss_e = self.compute_margin_loss(B, C, D, H, W, emb_e_vec.detach(), emb_g_vec, vis_e_vec, 'e', True, summ_writer)
        # l2_loss_e = reduce_masked_mean(sql2_on_axis(emb_e.detach()-emb_g, 1, keepdim=True), vis_e)
        # # where g is valid, we use it as reference and pull up e
        # margin_loss_g = self.compute_margin_loss(B, C, D, H, W, emb_e_vec, emb_g_vec.detach(), vis_g_vec, 'g', True, summ_writer)
        # l2_loss_g = reduce_masked_mean(sql2_on_axis(emb_e-emb_g.detach(), 1, keepdim=True), vis_g)
        # # where both are valid OR neither is valid, we pull them together
        # vis_both_or_neither_vec = torch.clamp(vis_e_vec*vis_g_vec + (1.0-vis_e_vec)*(1.0-vis_g_vec), 0, 1)
        # vis_both_or_neither = torch.clamp(vis_e*vis_g + (1.0-vis_e)*(1.0-vis_g), 0, 1)
        # margin_loss_n = self.compute_margin_loss(B, C, D, H, W, emb_e_vec, emb_g_vec, vis_both_or_neither_vec, 'n', True, summ_writer)
        # l2_loss_n = reduce_masked_mean(sql2_on_axis(emb_e-emb_g, 1, keepdim=True), vis_both_or_neither)
        # margin_loss = (margin_loss_e + margin_loss_g + margin_loss_n)/3.0
        # l2_loss = (l2_loss_e + l2_loss_g + l2_loss_n)/3.0
        # total_loss = utils_misc.add_loss('emb3D/emb_3D_ml_loss', total_loss, margin_loss, hyp.emb_3D_ml_coeff, summ_writer)
        # total_loss = utils_misc.add_loss('emb3D/emb_3D_l2_loss', total_loss, l2_loss, hyp.emb_3D_l2_coeff, summ_writer)

        l2_loss_im = torch.mean(sql2_on_axis(emb_e-emb_g, 1, keepdim=True), dim=3)
        summ_writer.summ_oned('emb3D/emb_3D_l2_loss', l2_loss_im)

        dz, dy, dx = utils.basic.gradient3D(emb_g, absolute=True)
        smooth_loss_im = torch.mean(torch.sum(dz + dy + dx, dim=1, keepdim=True), dim=3)
        summ_writer.summ_oned('emb3D/emb_3D_smooth_loss', smooth_loss_im)
        emb_smooth_loss = torch.mean(smooth_loss_im)
        total_loss = utils.misc.add_loss('emb3D/emb_3D_smooth_loss', total_loss, emb_smooth_loss, hyp.emb_3D_smooth_coeff, summ_writer)
        
        summ_writer.summ_feats('emb3D/embs_3D', [emb_e, emb_g], pca=True)
        return total_loss

