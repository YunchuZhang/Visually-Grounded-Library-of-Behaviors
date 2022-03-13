import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')

from archs.VGGNet import Feat2d
import hyperparams as hyp
from utils_basic import *
import utils_basic
import utils_misc
import utils_improc

from nets.embnet2D import EmbNet2D

class TouchEmbNet2DML(nn.Module):
	def __init__(self):
		super(TouchEmbNet2DML, self).__init__()
		self.touch_net = Feat2d(hyp.touch_emb_dim)
		self.num_samples = hyp.emb_2D_touch_num_samples
		assert (self.num_samples > 0), "num samples should be greater than zero"
		self.batch_k = 2
		self.sampler  = utils_misc.DistanceWeightedSampling(batch_k=self.batch_k,
			normalize=False)
		self.criterion = utils_misc.MarginLoss()
		self.beta = 1.2

	def sample_embs(self, emb0, emb1, B, Y, X, mod='', do_vis=False, summ_writer=None):
		"""
		parameters:
		-------------------------
		emb0: sensor_embeddings
		emb1: sampled_embeddings
		B = Batch SIZE
		Y = 1 for me
		X = 1 for me
		"""
		if hyp.emb_2D_mindist == 0.0:
			# pure random # mindist is not zero but I believe this would be simpler to understand
			perm = torch.randperm(B*Y*X) # say batch was 8x1x1=8 random permutation
			emb0 = emb0.reshape(B*Y*X, -1)  # 8x32
			emb1 = emb1.reshape(B*Y*X, -1)  # 8x32
			emb0 = emb0[perm[:self.num_samples*B]]  # 8
			emb1 = emb1[perm[:self.num_samples*B]]  # 8
			return emb0, emb1

	def compute_margin_loss(self, B, C, Y, X, emb_e_vec, emb_g_vec, mod='', do_vis=False, summ_writer=None):
		emb_e_vec, emb_g_vec = self.sample_embs(emb_e_vec,
												emb_g_vec,
												B, Y, X,
												mod=mod,
												do_vis=do_vis,
												summ_writer=summ_writer)
		assert not torch.allclose(emb_e_vec, torch.zeros_like(emb_e_vec)), "embeddings have become zero investigate"
		assert not torch.allclose(emb_g_vec, torch.zeros_like(emb_g_vec)), "sampled embeddings are all close to zero, why?"

		emb_vec = torch.stack((emb_e_vec, emb_g_vec), dim=1).view(B*self.num_samples*self.batch_k,C)
		# this tensor goes e,g,e,g,... on dim 0, shape is (8*1*2, 32)
		# note this means 2 samples per class; batch_k=2
		y = torch.stack([torch.arange(0,self.num_samples*B), torch.arange(0,self.num_samples*B)], dim=1).view(self.num_samples*B*self.batch_k)
		# this tensor goes 0,0,1,1,2,2,... , shape is (16, 32)

		# carefully examine each step of this function and then good to go for training
		a_indices, anchors, positives, negatives, _ = self.sampler(emb_vec)
		# a_indices are (0, 1, 2, ...), anchors is itself
		# positives are (1 for 0) (0 for 1) (3 for 2) (2 for 3) and so on, this is for k=2
		margin_loss, _ = self.criterion(anchors, positives, negatives, self.beta, y[a_indices])
		return margin_loss

	def forward(self, sensor_imgs, sampled_embeddings, do_ml, summ_writer):
		total_loss = torch.tensor(0.0).cuda()
		sensor_feats = self.touch_net(sensor_imgs)
		sensor_embeddings = l2_normalize(sensor_feats, dim=1)

		# Now I have the sensor embeddings and the sampled_embedding compute simple l2 loss on them
		simple_l2_loss = F.mse_loss(sensor_embeddings, sampled_embeddings)
		total_loss = utils_misc.add_loss('embtouch/emb_touch_l2_loss', total_loss,
			simple_l2_loss, hyp.emb_2D_touch_l2_coeff, summ_writer)

		if len(list(sensor_embeddings.shape)) == 2:
			prev_B, prev_C = list(sensor_embeddings.shape)
			assert len(list(sampled_embeddings)) == len(list(sensor_embeddings))
			sampled_embeddings = sampled_embeddings.view(prev_B, prev_C, 1, 1)
			sensor_embeddings = sensor_embeddings.view(prev_B, prev_C, 1, 1)
		
		if do_ml:
			print('doing ml loss, so this iteration you should not be printed')
			import ipdb; ipdb.set_trace()
			B, C, H, W = list(sensor_embeddings.shape)
			# to make it compatible with the api of compute margin loss need to reshape
			emb_e_vec = sensor_embeddings.permute(0,2,3,1).view(B, H*W, C)
			emb_g_vec = sampled_embeddings.permute(0,2,3,1).view(B, H*W, C)

			assert B == hyp.B, "batch should be same"
			assert C == hyp.touch_emb_dim, "this is the network output I specified"
			assert self.num_samples < (B*H*W), "num samples in hyp is problem"

			margin_loss = self.compute_margin_loss(B, C, H, W, emb_e_vec, emb_g_vec,
				'all', True, summ_writer)

			# this adds a two plots to tensorboard, raw and scaled loss
			# add the curr loss to total loss and returns
			total_loss = utils_misc.add_loss('embtouch/emb_touch_ml_loss', total_loss,
			margin_loss, hyp.emb_2D_touch_ml_coeff, summ_writer)
			print('L2Loss: {}\t MarginLoss: {}\t total_loss: {}'.format(simple_l2_loss.item(), margin_loss.item(), total_loss.item()))
		
		# summarize the pred_embeddings
		summ_writer.summ_feats('embtouch/embs_touch', [sensor_embeddings, sampled_embeddings], pca=True)
		return total_loss, sensor_embeddings

		# if 0:
		# 	# to make it compatible with others and to not change the api
		# 	if len(list(sensor_embeddings.shape)) == 2:
		# 		prev_B, prev_C = list(sensor_embeddings.shape)
		# 		assert len(list(sampled_embeddings)) == len(list(sensor_embeddings))
		# 		sampled_embeddings = sampled_embeddings.view(prev_B, prev_C, 1, 1)
		# 		sensor_embeddings = sensor_embeddings.view(prev_B, prev_C, 1, 1)

		# 	if do_ml:
		# 		print('doing ml loss, so this iteration you should not be printed')
		# 		import ipdb; ipdb.set_trace()
		# 		B, C, H, W = list(sensor_embeddings.shape)
		# 		# to make it compatible with the api of compute margin loss need to reshape
		# 		emb_e_vec = sensor_embeddings.permute(0,2,3,1).view(B, H*W, C)
		# 		emb_g_vec = sampled_embeddings.permute(0,2,3,1).view(B, H*W, C)

		# 		assert B == hyp.B, "batch should be same"
		# 		assert C == hyp.touch_emb_dim, "this is the network output I specified"
		# 		assert self.num_samples < (B*H*W), "num samples in hyp is problem"

		# 		margin_loss = self.compute_margin_loss(B, C, H, W, emb_e_vec, emb_g_vec,
		# 			'all', True, summ_writer)

		# 		# this adds a two plots to tensorboard, raw and scaled loss
		# 		# add the curr loss to total loss and returns
		# 		total_loss = utils_misc.add_loss('embtouch/emb_touch_ml_loss', total_loss,
		# 		margin_loss, hyp.emb_2D_touch_ml_coeff, summ_writer)

		# 	# compute the l2 loss between the two embeddings, TODO: Valid only penalizes
		# 	# the object part I should do this or correct view part.
		# 	l2_loss_im = sql2_on_axis(sensor_embeddings-sampled_embeddings, 1, keepdim=True)
		# 	# NOTE: I understand why you are black, you are single and then normalized to zero :(
		# 	summ_writer.summ_oned('embtouch/emb_touch_l2_loss', l2_loss_im)
		# 	emb_l2_loss = reduce_masked_mean(l2_loss_im, torch.ones_like(l2_loss_im))
		# 	total_loss = utils_misc.add_loss('embtouch/emb_touch_l2_loss', total_loss,
		# 		emb_l2_loss, hyp.emb_2D_touch_l2_coeff, summ_writer)
