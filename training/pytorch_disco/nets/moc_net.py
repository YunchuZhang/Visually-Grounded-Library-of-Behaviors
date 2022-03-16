import torch
import torch.nn as nn
import os
os.sys.path.append('/home/gauravp/pytorch_disco')
from backend.double_pool import DoublePoolMoc
from tensorboardX import SummaryWriter

import utils

import numpy as np

class MOCTraining(nn.Module):
    def __init__(self, dict_len, num_neg_samples):
        """
        params:
        ---------
            dict_len: length of the dictionary
            num_neg_samples: how many negative samples we have
        """
        super(MOCTraining, self).__init__()
        self.neg_pool = DoublePoolMoc(dict_len)
        self.num_neg_samples = num_neg_samples
        self.criterion = torch.nn.CrossEntropyLoss()
        self.temp = 0.07
        self.momentum_update = 0.999

    def init_pool(self, inp_dataloader, log_dir, MAX_QUEUE, model):
        """
            Here I fill the data with one epoch worth of samples
            inp_dataloader : train dataloader for doing initial filling of the pool
            log_dir        : a necessity for doing forward pass
            MAX_QUEUE      : a necessity for doing forward pass
            model          : model instance which is used for usual forward pass
        """
        # so I have this data_loader instance, as of now not writing anything to this summ_writer
        # but it will show up in tensorboard and can be used later
        moc_summ_writer = SummaryWriter(log_dir + '/' + 'moc_init', max_queue=MAX_QUEUE)
        for i, data_dict in enumerate(inp_dataloader):
            ## extra stuff required to run a forward pass through my model
            feed_cuda = data_dict
            for k in data_dict:
                if k != 'object_name':
                    feed_cuda[k] = data_dict[k].cuda(non_blocking=True)

            feed_cuda['writer'] = moc_summ_writer
            feed_cuda['global_step'] = i
            feed_cuda['set_name'] = 'moc_init'

            # model.eval()
            with torch.no_grad():
                _, results, neg_encodings = model(feed_cuda, moc_init_done=False, debug=False)

            neg_encodings = neg_encodings.detach()
            assert len(list(neg_encodings.shape)) == 2, "I want neg encodings of the form b*num_pos, 32"

            assert not neg_encodings.requires_grad

            # fill it in the pool, #NOTE: moving to cpu this will be slow but atleast not memory
            # intensive on the gpu
            self.neg_pool.update(neg_encodings.cpu())
            if i == 2:
                break

    def update_slow_network(self, slow_net, fast_net):
        """
            slow_net : a nn.Module instance which you want to update slowly
            fast_net : a nn.Module instance whose parameters will be used to
                       update the slow network
            Do the momentum update of the slow network
        """
        beta = self.momentum_update
        param_k = slow_net.state_dict()
        param_q = fast_net.named_parameters()
        for n, q in param_q:
            if n in param_k:
                param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
        slow_net.load_state_dict(param_k)

    def enqueue(self, features_to_enqueue):
        """
            Added the latest features to the pool
            It also handles dequeuing of the features
        """
        neg_touch, neg_context = features_to_enqueue
        assert not neg_touch.requires_grad
        assert not neg_context.requires_grad

        # self.neg_pool.update(features_to_enqueue.cpu())
        self.key_context_neg_pool.update(neg_context.cpu())
        self.key_sensor_neg_pool.update(neg_touch.cpu())

    def forward(self, q, k, summ_writer=None):
        # q : N x emb_dim
        # k : N x emb_dim
        total_loss = torch.tensor(0.0).cuda()
        N, C = list(q.shape)
        N1, C1 = list(k.shape)
        assert (N==N1)
        assert (C==C1)

        # now I am not not going to update the visual encoder
        k = k.detach()
        assert not k.requires_grad

        # TODO: somehow the features are not coming out to be normalized, check why and correct it

        # handle the positives
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        # handle the negatives
        negs = self.neg_pool.fetch(num=None)
        negs = negs.to(q.device)
        N2, C2 = list(negs.shape)
        assert (C2 == C)
        # now these negatives should be far away from all the positives
        l_negs = torch.mm(q.view(N, C), negs.view(C, N2)) ## NxN2

        # do the loss between them
        logits = torch.cat([l_pos.view(-1, 1), l_negs], dim=1)
        labels = torch.zeros(len(q), dtype=torch.long).to(q.device)
        # Now I am doing (neg_samples + 1) - way classification
        # softmaxing is handled by the cross entropy loss itself
        emb_loss = self.criterion(logits/self.temp, labels)
        total_loss = utils.misc.add_loss('moco/ml_loss', total_loss, emb_loss, 1.0, summ_writer)
        return emb_loss


class MOCTrainingTouch(nn.Module):
    def __init__(self, dict_len, num_neg_samples):
        """
        params:
        ---------
            dict_len: length of the dictionary
            num_neg_samples: how many negative samples we have
        """
        super(MOCTrainingTouch, self).__init__()
        self.key_context_neg_pool = DoublePoolMoc(dict_len)
        self.key_sensor_neg_pool = DoublePoolMoc(dict_len)
        self.num_neg_samples = num_neg_samples
        self.criterion = torch.nn.CrossEntropyLoss()
        self.temp = 0.07
        self.momentum_update = 0.999
        self.step_num = 0 ## every time forward is called I increase you by one
        self.values_over_time = list()

    def init_pool(self, inp_dataloader, log_dir, MAX_QUEUE, model):
        """
            Here I fill the data with one epoch worth of samples
            inp_dataloader : train dataloader for doing initial filling of the pool
            log_dir        : a necessity for doing forward pass
            MAX_QUEUE      : a necessity for doing forward pass
            model          : model instance which is used for usual forward pass
        """
        # so I have this data_loader instance, as of now not writing anything to this summ_writer
        # but it will show up in tensorboard and can be used later
        moc_summ_writer = SummaryWriter(log_dir + '/' + 'moc_init', max_queue=MAX_QUEUE)
        for i, data_dict in enumerate(inp_dataloader):
            ## extra stuff required to run a forward pass through my model
            feed_cuda = data_dict
            for k in data_dict:
                if k != 'object_name':
                    feed_cuda[k] = data_dict[k].cuda(non_blocking=True)

            feed_cuda['writer'] = moc_summ_writer
            feed_cuda['global_step'] = i
            feed_cuda['set_name'] = 'moc_init'

            model.eval()
            assert not model.featnet.training
            with torch.no_grad():
                _, results, moco_embeddings = model(feed_cuda, moc_init_done=False, debug=False)

            assert isinstance(moco_embeddings, list),\
                "this should be list containing embeddings for touch network and context network"

            touch_neg_embeddings, context_neg_embeddings = moco_embeddings

            touch_neg_embeddings = touch_neg_embeddings.detach()
            context_neg_embeddings = context_neg_embeddings.detach()
            assert len(list(touch_neg_embeddings.shape)) == 2, "I want neg embeddings of the form b*num_pos, 32"
            assert len(list(context_neg_embeddings.shape)) == 2, "check the dim of context neg_embeddings"
            assert not touch_neg_embeddings.requires_grad
            assert not context_neg_embeddings.requires_grad

            # fill it in the pool, #NOTE: moving to cpu this will be slow but atleast not memory
            # intensive on the gpu
            # self.neg_pool.update(torch.cat((touch_neg_embeddings, context_neg_embeddings), dim=0).cpu())
            self.key_context_neg_pool.update(context_neg_embeddings.cpu())
            self.key_sensor_neg_pool.update(touch_neg_embeddings.cpu())
            if self.key_context_neg_pool.is_full() and self.key_sensor_neg_pool.is_full():
                print('breaking out because the pool is full')
                break

    def update_slow_network(self, slow_net, fast_net):
        """
            slow_net : a nn.Module instance which you want to update slowly
            fast_net : a nn.Module instance whose parameters will be used to
                       update the slow network
            Do the momentum update of the slow network
        """
        beta = self.momentum_update
        param_k = slow_net.state_dict()
        param_q = fast_net.named_parameters()
        for n, q in param_q:
            # checking if here is not good, I want them to be same
            assert n in param_k, "parameters of q and k are not the same"
            param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
        slow_net.load_state_dict(param_k)

    def enqueue(self, features_to_enqueue):
        """
            Added the latest features to the pool
            It also handles dequeuing of the features
        """
        neg_touch, neg_context = features_to_enqueue
        assert not neg_touch.requires_grad
        assert not neg_context.requires_grad

        # self.neg_pool.update(features_to_enqueue.cpu())
        self.key_context_neg_pool.update(neg_context.cpu())
        self.key_sensor_neg_pool.update(neg_touch.cpu())

    def forward(self,
                sensor_feats,
                key_sensor_feats,
                context_feats,
                key_context_feats,
                summ_writer=None):
        # sensor_feats : N, 32
        # key_sensor_feats : N, 32
        # context_feats : N, 32
        # key_context_feats : N, 32
        # summ_writer : summ_writer object
        total_loss = torch.tensor(0.0).cuda()

        N, C = list(sensor_feats.shape)
        N1, C1 = list(key_sensor_feats.shape)
        N2, C2 = list(context_feats.shape)
        N3, C3 = list(key_context_feats.shape)

        assert (N==N1==N2==N3)
        assert (C==C1==C2==C3)

        # now I am not not going to update the visual encoder
        key_sensor_feats = key_sensor_feats.detach()
        key_context_feats = key_context_feats.detach()
        assert not key_context_feats.requires_grad
        assert not key_sensor_feats.requires_grad

        # handling the moco of touch features begin #
        # I checked the positions after concat and it seems fine #
        logits_sensor_pos = torch.bmm(sensor_feats.view(N, 1, C),\
            key_context_feats.view(N, C, 1))

        ### ... Debug or book-keeping start ... ###
        # dot product to see if its far from others over time
        cloned_sensor_feats = sensor_feats.clone().detach()
        cloned_key_context_feats = key_context_feats.clone().detach()

        # compute the affinity score
        aff_score = torch.mm(cloned_sensor_feats, cloned_key_context_feats.T)
        preds_inds = torch.max(aff_score, dim=1)[1]
        target_inds = torch.arange(len(aff_score), dtype=torch.long).to(aff_score.device)
        acc = preds_inds.eq(target_inds).sum().cpu().item()
        acc /= float(len(preds_inds))
        print(f'classification acc for sensor_feats and key_context_feats {acc}')
        summ_writer.summ_scalar('train_set/class_acc', acc)

        ## .. below code computes the affinity score of positives and max .. ##
        ## .. stores it in a file, if you want to compare for later .. ##
        # dots = aff_score.cpu().numpy()
        # true_positive_affinity = torch.diag(dots)
        # # now get the indices of the max
        # max_indices = np.argmax(dots, axis=1)
        # affinity_value_max_indices = dots[np.arange(len(dots)), max_indices]
        # save_dict = {'step': self.step_num,
        #              'affinity_true_positives': true_positive_affinity,
        #              'max_indices': max_indices,
        #              'affinity_false_positives': affinity_value_max_indices
        #             }
        # self.values_over_time.append(save_dict)
        # if self.step_num % 50 == 0:
        #     np.save(f'vals_{self.step_num}.npy', self.values_over_time)
        #     self.values_over_time = []
        # self.step_num += 1
        # ### ... Debug or book-keeping ends ... ###

        # handle the negatives
        # negs = self.neg_pool.fetch(num=None)

        # negs for each positive from the same batch should be added here
        #sample_mask = torch.ones((len(sensor_feats), len(sensor_feats)))
        context_negs = self.key_context_neg_pool.fetch(num=None)
        context_negs = context_negs.to(sensor_feats.device)
        N2, C2 = list(context_negs.shape)
        assert (C2 == C)
        # now these negatives should be far away from all the positives
        logits_sensor_negs = torch.mm(sensor_feats.view(N, C),
                                      context_negs.view(C, N2)) ## NxN2

        # do the loss between them, first is the true class
        sensor_logits = torch.cat([logits_sensor_pos.view(-1, 1),
                                   logits_sensor_negs], dim=1)
        labels = torch.zeros(len(sensor_feats),
                             dtype=torch.long).to(sensor_feats.device)
        # Now I am doing (neg_samples + 1) - way classification
        # softmaxing is handled by the cross entropy loss itself
        # what is the use of this temperature term in softmax
        sensor_emb_loss = self.criterion(sensor_logits/self.temp, labels)
        total_loss = utils.misc.add_loss('moco/ml_sensor_loss', total_loss, sensor_emb_loss,\
            0.5, summ_writer)
        # handling the moco of touch features end #

        # handling of moco context features begin #
        logits_context_pos = torch.bmm(context_feats.view(N, 1, C),
                                       key_sensor_feats.view(N, C, 1))

        # now sample the negatives for the context
        sensor_negs = self.key_sensor_neg_pool.fetch(num=None)
        sensor_negs = sensor_negs.to(context_feats.device)
        N3, C3 = list(sensor_negs.shape)
        assert (C3 == C)
        logits_context_negs = torch.mm(context_feats.view(N, C),
                                       sensor_negs.view(C, N3))

        context_logits = torch.cat([logits_context_pos.view(-1, 1),
                                    logits_context_negs], dim=1)

        ## ... plot the classification acc, this is same as prec@recall/1 .. ##
        cloned_context_feats = context_feats.clone().detach()
        cloned_key_sensor_feats = key_sensor_feats.clone().detach()

        caff_score = torch.mm(cloned_context_feats, cloned_key_sensor_feats.T)
        pred_idxs = torch.max(caff_score, dim=1)[1]
        target_idxs = torch.arange(len(pred_idxs), dtype=torch.long).to(pred_idxs.device)
        # compare the two and compute acc
        cacc = pred_idxs.eq(target_idxs).sum().cpu().item()
        cacc /= float(len(pred_idxs))
        print(f'classification accuracy for context_feats and key_sensor_feats {cacc}')
        summ_writer.summ_scalar('train_set/class_cacc', cacc)

        c_labels = torch.zeros(len(context_feats), dtype=torch.long).to(context_feats.device)
        context_emb_loss = self.criterion(context_logits/self.temp, c_labels)
        total_loss = utils.misc.add_loss('moco/ml_context_loss', total_loss, context_emb_loss,\
            0.5, summ_writer)
        # handling of moco context features end   #

        return total_loss

if __name__ == '__main__':
    moc_trainer = MOCTraining(dict_len=10000, num_neg_samples=2000)

    # now I need to fill up the neg_pool for the initialization
    for i in range(100):
        x = torch.randn(100, 32)
        moc_trainer.neg_pool.update(x)

    # okay now it is full lets do the forward pass
    # I will do some forward passes to check if the thing works
    # each forward pass samples from the neg_pool
    for i in range(10):
        q = torch.randn(1024, 32)
        k = torch.randn(1024, 32)
        loss = moc_trainer(q, k)
        print('Loss : {}'.format(loss))
