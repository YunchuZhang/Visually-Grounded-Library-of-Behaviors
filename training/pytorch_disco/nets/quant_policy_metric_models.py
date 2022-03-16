import torch
import torch.nn as nn
import torch.nn.functional as F

from archs.encoder3D import SimpleEncoder3D, Net3D
from archs.encoder2D import SimpleEncoder2D, Net2D

def compute_loss1D(embeddings, labels, device, temperature=0.7):
    B, C = list(embeddings.shape)

    cup_idxs = (labels == 0).nonzero()
    cup_embeddings = embeddings[cup_idxs.squeeze(1)]
    
    # check the dimensions:
    assert cup_embeddings.ndim == 2
    assert cup_embeddings.shape[1] == C

    bottle_idxs = (labels == 1).nonzero()
    bottle_embeddings = embeddings[bottle_idxs.squeeze(1)]
    
    # check the dimensions
    assert bottle_embeddings.ndim == 2
    assert bottle_embeddings.shape[1] == C

    # generate the positives for the cup, strategy is for 0, 1 serves as positive and so on
    # finally for the last entry the zeroth entry serves as the positive
    positive_cup_embeddings = cup_embeddings[1:,:].clone()
    positive_cup_embeddings = torch.cat((positive_cup_embeddings, cup_embeddings[0][None].clone()))
    
    # check the dimensions
    assert cup_embeddings.shape == positive_cup_embeddings.shape

    # generate positives for the bottle, strategy is same as that for the cup
    positive_bottle_embeddings = bottle_embeddings[1:,:].clone()
    positive_bottle_embeddings = torch.cat((positive_bottle_embeddings, bottle_embeddings[0][None].clone()))
    
    # check the dimensions
    assert bottle_embeddings.shape == positive_bottle_embeddings.shape

    # check that the order of the positives is one after the another.
    assert torch.equal(positive_cup_embeddings[:-1, :], cup_embeddings[1:, :])
    assert torch.equal(positive_bottle_embeddings[:-1, :], bottle_embeddings[1:, :])
    assert torch.equal(positive_cup_embeddings[-1], cup_embeddings[0])
    assert torch.equal(positive_bottle_embeddings[-1], bottle_embeddings[0])

    # collate the positives together to compute the positive loss
    pos_query_embeddings = torch.cat((cup_embeddings, bottle_embeddings))
    assert len(pos_query_embeddings) == B
    pos_key_embeddings = torch.cat((positive_cup_embeddings, positive_bottle_embeddings))
    assert len(pos_key_embeddings) == B

    # again checking that concat has not screwed up anything
    assert torch.equal(pos_query_embeddings[1:len(cup_embeddings)], pos_key_embeddings[0:len(cup_embeddings)-1])
    assert pos_query_embeddings.shape == pos_key_embeddings.shape

    # now bring the positives closer, well here just compute the dot product
    l_pos = torch.bmm(pos_query_embeddings.view(B, 1, C), pos_key_embeddings.view(B, C, 1))
    l_pos = l_pos.view(-1, 1)

    # now make the negatives far apart, means the dot product of this should go close to zero
    # for each cup every bottle is negative and for each bottle every cup is negative.
    # ASHWINI: I am keeping the min as the negatives, so it will change in each batch would that be a problem?
    num_neg_samples = min(len(cup_embeddings), len(bottle_embeddings))

    # negs for cups, cups are N,C and bottles we would want C, K=min_num assuming bottles are more
    l_negs_cup = torch.mm(cup_embeddings, bottle_embeddings[:num_neg_samples, :].view(C, num_neg_samples))
    l_negs_bottle = torch.mm(bottle_embeddings, cup_embeddings[:num_neg_samples, :].view(C, num_neg_samples))
    l_negs = torch.cat((l_negs_cup, l_negs_bottle), dim=0)
    
    # finally compute the loss and be done with it
    logits = torch.cat((l_pos, l_negs), dim=1)

    # labels, positives are the zero-th 
    labels = torch.zeros(len(logits)).long().to(device)

    # compute the cross entropy loss
    loss = F.cross_entropy(logits/temperature, labels)
    return loss


def compute_loss3D(embeddings, labels, device, temperature=0.7):
    """
        embeddings: a B, C, D, H, W array
        lables: B, 1 array
        device: torch.device instance, either cuda or cpu
    """
    # all of them are 3d embeddings, I again need to separate
    # them and then do the dot product things.
    B, C, D, H, W = list(embeddings.shape)

    class1_idx = (labels == 0).nonzero()
    class1_embs = embeddings[class1_idx.squeeze(1)]
    assert class1_embs.shape[1] == C

    class2_idx = (labels == 1).nonzero()
    class2_embs =  embeddings[class2_idx.squeeze(1)]
    assert class2_embs.shape[1] == C

    # reshape the embeddings to be B, C, (D*H*W)
    class1_embs = class1_embs.view(B//2, C, -1)
    class2_embs = class2_embs.view(B//2, C, -1)

    # generate positives for class1 and class2
    class1_pos = class1_embs[1:].clone()
    class1_pos = torch.cat((class1_pos, class1_embs[0].unsqueeze(0).clone()), dim=0)
    assert torch.equal(class1_embs[1:], class1_pos[:-1])
    assert torch.equal(class1_embs[0], class1_pos[-1])

    class2_pos = class2_embs[1:].clone()
    class2_pos = torch.cat((class2_pos, class2_embs[0].unsqueeze(0).clone()), dim=0)
    assert torch.equal(class2_embs[1:], class2_pos[:-1])
    assert torch.equal(class2_embs[0], class2_pos[-1])

    # join the keys, and queries together
    pos_query_embs = torch.cat((class1_embs, class2_embs), dim=0).view(B, -1)
    pos_key_embs = torch.cat((class1_pos, class2_pos), dim=0).view(B, -1)

    # now finally do the dot product one-to-one
    logits_pos = torch.bmm(pos_query_embs.view(B, 1, -1), pos_key_embs.view(B, -1, 1))
    logits_pos = logits_pos.div(D*H*W).view(-1, 1)

    num_neg_samples = min(len(class1_embs), len(class2_embs))
    logits_neg_class1 = torch.mm(class1_embs.view(num_neg_samples,-1),
        class2_embs.view(num_neg_samples, -1).t())
    logits_neg_class1 = logits_neg_class1.div(D*H*W)
    
    logits_neg_class2 = torch.mm(class2_embs.view(num_neg_samples,-1),
        class1_embs.view(num_neg_samples, -1).t())
    logits_neg_class2 = logits_neg_class2.div(D*H*W)
    logits_negs = torch.cat((logits_neg_class1, logits_neg_class2), dim=0)

    # logits
    logits = torch.cat((logits_pos, logits_negs), dim=1)
    labels = torch.zeros(len(logits)).long().to(device)

    # compute the loss
    loss = F.cross_entropy(logits/temperature, labels)
    return loss


# declare and make the model
class EmbeddingGenerator1DFrom2D(nn.Module):
    def __init__(self, pred_dim=128, in_channel=32):
        super(EmbeddingGenerator1DFrom2D, self).__init__()
        self.pred_dim = pred_dim
        # define the network here
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.net = SimpleEncoder2D(in_channel=in_channel, pred_dim=pred_dim, chans=32).to(device=device)
        self.linear = nn.Linear(self.pred_dim * 16, pred_dim).to(device=device)

    def forward(self, inputs):
        # inputs are 3d tensors corresponding to the object shape
        B, _, _, _ = list(inputs.shape)
        unnormalized_out = self.net(inputs)
        unnormalized_out = unnormalized_out.view(B, -1)
        unnormalized_out  = self.linear(unnormalized_out )
        # normalize the embeddings

        out = F.normalize(unnormalized_out, p=2, dim=1)
        return out

# NOTE: It returns normalized embeddings
class EmbeddingGenerator1D(nn.Module):
    def __init__(self, emb_dim=128):
        super(EmbeddingGenerator1D, self).__init__()
        self.emb_dim = 128
        # define the network here
        self.net = SimpleEncoder3D(in_channel=32, pred_dim=emb_dim, chans=32)
    
    def forward(self, inputs):
        # inputs are 3d tensors corresponding to the object shape
        B, _, _, _, _ = list(inputs.shape)
        unnormalized_out = self.net(inputs)
        # normalize the embeddings
        unnormalized_out = unnormalized_out.view(B, -1)
        out = F.normalize(unnormalized_out, p=2, dim=1)
        return out
    
# 3d encoder decoder with skip connections for metric learning as recommended by fish
class EmbeddingGenerator3D(nn.Module):
    def __init__(self, emb_dim=16, en_channel=32):
        super(EmbeddingGenerator3D, self).__init__()
        # I am reducing the embedding dimensions, TODO: ask fish if this is okay
        self.emb_dim = emb_dim
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        
        # network definition
        self.net = Net3D(in_channel=en_channel, pred_dim=emb_dim, chans=32).to(device=device)
        
    def forward(self, inputs):
        # inputs are 3d tensor -> object centered tensor, of size B, 32, 32, 32, 32
        B, _, _, _, _ = list(inputs.shape)
        unnormalized_out = self.net(inputs)
        # normalize the embeddings
        normalized_out = F.normalize(unnormalized_out, p=2, dim=1)
        return normalized_out
        
if __name__ == '__main__':
    model = EmbeddingGenerator3D(emb_dim=16)
    x = torch.randn(16, 32, 32, 32, 32)
    out = model(x)
    print(out.shape)
    
    # for the 1d case sanity check
    model = EmbeddingGenerator(emb_dim=128)
    x = torch.randn(16, 32, 32, 32, 32)
    out = model(x)
    print(out.shape) ## should be 16, 128