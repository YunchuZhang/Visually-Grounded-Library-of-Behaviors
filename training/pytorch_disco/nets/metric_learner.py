import torch
import torch.nn as nn
import torch.nn.functional as F

from archs.encoder3D import SimpleEncoder3D
import numpy as np 

class MetricLearner(nn.Module):
    def __init__(self, num_objects=150, repr_dim=128):
        super(MetricLearner, self).__init__()
        self.pred_dim = repr_dim
        # We need a Siamese style network
        self.encoder1 = SimpleEncoder3D(in_channel=32, pred_dim=128, chans=32).cuda()
        self.encoder2 = SimpleEncoder3D(in_channel=32, pred_dim=128, chans=32).cuda()
        
        # This size (first dimension) could be num_objects (if we want to learn feature per object) [seems correct to me]
        # num_tasks or even completely independent num_clusters (how to choose idx key in this case?)
        self.repr =  nn.Embedding(num_objects, repr_dim).cuda()

        # Initialize weights to extremely small values with mean 0 and std 0.001
        self.repr.weight.data.normal_(mean=0,std=0.001)

        
    # We assume that inputs have both positive labels and negative labels
    # We assume that inputs consists of one positive sample and B-1 negative samples
    def forward(self, feat_tensors, labels):
        # We assume that positive sample is at index 0 
        # And other samples are negative samples

        # Inputs is (1, H, W, B, C)
        # Output is (1, F)
        pos_feat = self.encoder1(feat_tensors[0].unsqueeze(0))

        # Inputs is (B-1, H, W, B, C)
        # Output is (B-1, F)
        neg_feat = self.encoder2(feat_tensors[1:])

        # feats is (B, F)
        feats = torch.cat([pos_feat, neg_feat], dim=0)
        if len(feats.shape) > 2:
            feats = torch.squeeze(feats)
        import ipdb; ipdb.set_trace()
        # We need to normalize the output of encoders
        norm = feats.norm(p=2, dim=1, keepdim=True)
        feats_normalized = feats.div(norm)

        loss = self.compute_loss(feats_normalized, labels, loss_type="n-class")
        return loss, feats_normalized

    # Note: feats contains both negative and positive feats
    # feats[0] belongs to the positive sample
    # labels[0] is the key of positive sample
    # we are learning embedding per object. We should use object labels and not object category labels here!
    def compute_loss(self, feats, object_labels, loss_type, temp=1):
        loss = 0
        if loss_type == "n-class":
            import ipdb; ipdb.set_trace()
            indices = self.repr(object_labels)
            assert len(feats) == 2, "feats tensor should have shape (batch_size, pred_dim)"
            prod = torch.bmm(feats.view(-1, 1, self.pred_dim), indices.view(-1, self.pred_dim, 1))/temp
            max_val = torch.max(prod)
            prod = prod - max_val
            softmax = torch.div(torch.exp(prod[0]), torch.sum(torch.exp(prod)))
            loss = -1 * torch.log(softmax)
            loss = torch.squeeze(loss) 
        return loss