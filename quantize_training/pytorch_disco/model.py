import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from backend import saverloader, inputs
from backend.inputs import MuJoCoOfflineData
# from backend.double_pool import DoublePool
#from torchvision import datasets, transforms
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import ipdb
st=ipdb.set_trace
np.set_printoptions(precision=5)
EPS = 1e-6
np.random.seed(0)
MAX_QUEUE = 10 #how many items before the summaryWriter flush


class Model(object):
    def __init__(self, config):

        print('------ CREATING NEW MODEL ------')
        print(config.run_full_name)
        self.checkpoint_dir = config.checkpoint_dir
        self.log_dir = config.log_dir
        self.config=config
        self.lr = config.lr
        self.all_inputs = inputs.get_inputs(config)
        self.big_list_of_results = list()  # NOTE: this is temporary

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        #self.device = torch.device("cuda")
        self.moc_init_flag = False
        self.tensor_in_clusters = None
        self.nn_results_dir = config.vis_dir#os.path.join("results_dir", config.name)
        # only generate the directory if necessary, don't clutter stuff
        
        data_root_dir = utils.utils.get_data_dir()
        for set_name in config.data_paths:
            data_path = config.data_paths[set_name]
            if not data_path.startswith("/"):
                data_path = os.path.join(data_root_dir, data_path)
                config.data_paths[set_name] = data_path

        if config.do_validation:
            if not os.path.exists(self.nn_results_dir):
                os.makedirs(self.nn_results_dir)


        if config.feat_init is not None:
            model_dir = config.feat_init
            self.extracted_tensor_save_dir = os.path.join("extracted_tensor", model_dir, config.run_full_name)
            # again only create directory if necessary
            if config.do_generate_data:
                if not os.path.exists(self.extracted_tensor_save_dir):
                    os.makedirs(self.extracted_tensor_save_dir)

    def save_local_variables(self):
        return dict()

    def infer(self):
        print('nothing to infer!')
    
    @staticmethod
    def plot_top_k_retrieval(records_list, sorted_idxs, query_records_list=None):
        # NOTE: Hard code alert but I think it should be fine because img size will not change
        vis = list()
        for i, s in enumerate(sorted_idxs):
            minivis = list()
            # get image of the query and its top 10 neighbours
            query_record = query_records_list[i]
            data = np.load(query_record, allow_pickle=True).item()

            qref_img = data['rgb_camXs'][-1]
            minivis.append(qref_img)
            keys_record = [records_list[k] for k in s]
            for j, k in enumerate(keys_record):
                # get the filename first, load it and then get the ref images for it,
                # TODO: this all can be precomputed once during init
                data = np.load(k, allow_pickle=True).item()
                ref_img = data['rgb_camXs'][-1]
                minivis.append(ref_img)
            vis.append(minivis)
        return vis

    def compute_nearest_neighbours_ed(self):
        # compute nearest neighbours using euclidean distances
        raise NotImplementedError

    def compute_nearest_neighbours_dp(self, val_results_list, val_cluster_id_list, re_results_list=None, re_cluster_id_list=None, vis_top_k=10):
        """
        val_results_list: [feature_dim] * num_samples
        success_rate_list: np.array(#class): if select the top, then it counts
        """
        records_list = list()
        object_tensors = list()
        for rr in val_results_list:
            if "record_name" not in rr:
                import ipdb; ipdb.set_trace()
            rec = rr['record_name']
            obj_tensor = rr['object_tensor']
            # each of them is a 2 element list
            for re, objt in zip(rec, obj_tensor):
                records_list.append(re)
                object_tensors.append(objt)

        # do the dot product to compute the nearest neighbours
        resized_object_tensors = torch.stack(object_tensors, dim=0)

        if len(resized_object_tensors.shape) == 5:
            N, C, D, H, W = list(resized_object_tensors.shape)

        else:
            N, C = list(resized_object_tensors.shape)
            D, H, W = 1, 1, 1
        emb_vectors = resized_object_tensors.view(N, C, -1)

        emb_vectors = emb_vectors.permute(0, 2, 1)

        if re_results_list is not None:
            re_records_list = list()
            re_object_tensors = list()
            for rr in re_results_list:
                rec = rr['record_name']
                re_obj_tensor = rr['object_tensor']
                # each of them is a 2 element list
                for re, objt in zip(rec, re_obj_tensor):
                    re_records_list.append(re)
                    re_object_tensors.append(objt)

            # do the dot product to compute the nearest neighbours
            resized_re_object_tensors = torch.stack(re_object_tensors, dim=0)
            if len(resized_re_object_tensors.shape) == 5:
                N, C, D, H, W = list(resized_re_object_tensors.shape)
            else:
                N, C = list(resized_re_object_tensors.shape)
            re_emb_vectors = resized_re_object_tensors.view(N, C, -1)
            re_emb_vectors = re_emb_vectors.permute(0, 2, 1)
            cluster_id_list = re_cluster_id_list
            query_records_list = copy.deepcopy(records_list)
            records_list = re_records_list

        else:
            re_emb_vectors = emb_vectors
            cluster_id_list = val_cluster_id_list
            query_records_list = records_list


        # TODO: Naive version now for each of the emb_vectors compute its distance from every emb_vector
        dists = list()
        for e in range(len(emb_vectors)):
            curr_emb = emb_vectors[e]
            curr_dist = list()
            for f in range(len(re_emb_vectors)):
                if re_results_list is None and e == f:
                    curr_dist.append(1)
                    continue
                other_emb = re_emb_vectors[f]

                dot = curr_emb * other_emb
                dot = dot.sum(axis=1)
                assert len(dot) == (D*H*W)
                avg_dot = dot.sum() / len(dot)
                curr_dist.append(avg_dot)
            dists.append(curr_dist)

        dists = np.stack(dists, axis=0)
        # now that you have distance of current from every one else, get the top 10 nearest neighbours
        sort_idxs = np.argsort(-dists, axis=1)[:, :vis_top_k]
        
        # plot the stuff, for now I am doing it in matplotlib, it is just easier for me


        cluster_ids = []
        batch_size  = sort_idxs.shape[0]
        for bid in range(batch_size):
            sort_idxs_irow = sort_idxs[bid, :]
            cluster_id_irows = [cluster_id_list[id_] for id_ in sort_idxs_irow]
            cluster_ids.append(cluster_id_irows)
        cluster_ids = np.array(cluster_ids)


        if re_results_list is None: # test data on test data
            recall_1 = np.mean(cluster_ids[:,0] == cluster_ids[:,1])
            #            ndata =  cluster_ids.shape[0]
            #            for data_id in range(ndata):
            #                success_rate = success_rate_list[data_id]
            #                cluster_ids[data_id, 0]
            #
            #            import ipdb; ipdb.set_trace()
            #
            #            print("hello")


        else:
            recall_1 = np.mean(np.array(val_cluster_id_list) == cluster_ids[:,0])

        vis = self.plot_top_k_retrieval(records_list, sort_idxs, query_records_list=query_records_list)


        return vis, recall_1

    def get_features(self, dataloader, summ_writer, step, ndata=100000):
        results_list = list()
        cluster_id_list = list()
        for i, feed in enumerate(dataloader):
            # move everything to cuda
            feed_cuda = feed
            for k in feed:
                try:
                    feed_cuda[k] = feed[k].cuda()
                except:
                    # some of them are not tensor
                    feed_cuda[k] = feed[k]
            
            feed_cuda['writer'] = summ_writer 
            feed_cuda['global_step'] = step
            feed_cuda['set_num'] = 'val' #self.config.set_nums[data_name]
            feed_cuda['set_name'] = 1 #data_name
            feed_cuda['record'] = feed['record']

            # now I am ready for the forward pass, I want it to return to me
            # the 3d tensor which belongs to the object
            with torch.no_grad():
                loss, results =  self.model(feed_cuda)
            # for now since I am only debugging stuff
            # a list of dictionary
            # "object_tensor": B x C x H x W x D
            # "record_name": list of strings
            bsize = len(feed_cuda["cluster_id"][0])
            cluster_id_list += [feed_cuda["cluster_id"][0][bid] for bid in range(bsize)]
            results_list.append(results)
            if len(cluster_id_list) >= ndata:
                break

        return results_list, cluster_id_list, feed

    def get_data(self, dataloader, summ_writer, step, ndata=100000):
        data_list = list()
        total_num_data = 0
        for i, feed in enumerate(dataloader):
            # move everything to cuda
            feed_cuda = feed
            for k in feed:
                try:
                    feed_cuda[k] = feed[k].cuda()
                except:
                    # some of them are not tensor
                    feed_cuda[k] = feed[k]

            feed_cuda['writer'] = summ_writer
            feed_cuda['global_step'] = step
            feed_cuda['set_num'] = 1 #"val" #self.config.set_nums[data_name]
            feed_cuda['set_name'] = "val" #data_name
            feed_cuda['record'] = feed['record']

            # now I am ready for the forward pass, I want it to return to me
            # the 3d tensor which belongs to the object
            data_list.append(feed_cuda)
            total_num_data += feed_cuda["rgb_camXs"].shape[0]
            if total_num_data >= ndata:
                break

        return data_list
    def get_features_from_data_list(self, data_list, step):
        results_list = list()
        cluster_id_list = list()
        success_rate_list = list()

        for feed_cuda in data_list:
            feed_cuda['global_step'] = step
            with torch.no_grad():
                loss, results =  self.model(feed_cuda)
            # for now since I am only debugging stuff
            # a list of dictionary
            # "object_tensor": B x C x H x W x D
            # "record_name": list of strings
            bsize = len(feed_cuda["cluster_id"][0])
            cluster_id_list += [feed_cuda["cluster_id"][0][bid] for bid in range(bsize)]
            results_list.append(results)

            success_rate_list.append(feed_cuda["success_rates"])


        return results_list, cluster_id_list, success_rate_list, feed_cuda

    def validate_nn_on_test_from_train(self, step, val_summ_writer=None, train_summ_writer=None):
        print("start validate test on train pool")
        # first put the model in eval mode
        self.model.eval()
        # everytime make the val list empty

        if self.train_data_nn is None:
            print("make nn pool from train")
            assert not self.model.training, "Model should be in eval mode"
            train_data_path = self.config.data_paths['train']
            # now form the data-loader with the valset path
            train_dataloader = torch.utils.data.DataLoader(MuJoCoOfflineData(
                config = self.config,
                dataset_path=train_data_path,
                plot=False, train=False,
                fix_view=True,
                ndata = 40
                ), batch_size=2, shuffle=True, drop_last=True)


            self.train_data_nn = self.get_data(train_dataloader, train_summ_writer, step, ndata=40)

        train_results_list, train_cluster_id_list, _ = self.get_features_from_data_list(self.train_data_nn, step)

        if self.val_data_nn is None:
            print("make nn pool from test")
            test_data_path = self.config.data_paths['test']
            # now form the data-loader with the valset path

            val_dataloader = torch.utils.data.DataLoader(MuJoCoOfflineData(
                config = self.config,
                dataset_path=test_data_path,
                plot=False, train=False,
                fix_view=True
                ), batch_size=2, shuffle=True, drop_last=True)
            self.val_data_nn = self.get_data(val_dataloader,  val_summ_writer, step)

        val_results_list, val_cluster_id_list, feed = self.get_features_from_data_list(self.val_data_nn, step)
        vis_nearest_neighbours, recall_1 = self.compute_nearest_neighbours_dp(val_results_list, val_cluster_id_list,
                                               re_results_list=train_results_list, re_cluster_id_list=train_cluster_id_list)

        # now the only thing that remains is plotting this on tensorboard
        # just to satisfy my paranoia I will also save the matplotlib images
        # but after 500 iterations
        if feed['global_step'] % 5000 == 0:
            n_rows = len(vis_nearest_neighbours)
            n_cols = len(vis_nearest_neighbours[0])
            fig_size = 2 * np.asarray([n_rows, n_cols])
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size,
                sharex=True, sharey=True)
            for i in range(n_rows):
                for j in range(n_cols):
                    axes[i][j].imshow(vis_nearest_neighbours[i][j])

            # save the figure and you are done
            fig.savefig(f'{self.nn_results_dir}/test_train_nn_result_{step}.jpg')
            plt.close()

        H, W, C = list(vis_nearest_neighbours[0][1].shape)

        # finally add it to tensorboard and you are done !!!
        # #test_data x (#top_retrieve + 1)
        vis_nearest_neighbours = np.stack(vis_nearest_neighbours, axis=0)
        # convert to torch
        vis_nearest = torch.from_numpy(vis_nearest_neighbours).permute(0, 1, 4, 2, 3)
        # resize
        vis_nearest = vis_nearest.view(-1, C, H, W)
        # make the grid
        grid = make_grid(vis_nearest, nrow=11)
        
        # add it to the tensorboard
        feed['writer'].add_scalar('valtrain_nn_recall@1', recall_1, step)
        feed['writer'].add_image('valtrain_nn/imgs', grid, step)



    def validate_on_test(self, step, summ_writer=None):
        # first put the model in eval mode
        self.model.eval()
        # everytime make the val list empty
        #val_results_list = list()
        #cluster_id_list = list()
        print("start validate on test")
        assert not self.model.training, "Model should be in eval mode"
        if self.val_data_nn is None:
            test_data_path = self.config.data_paths['test']
            # now form the data-loader with the valset path

            val_dataloader = torch.utils.data.DataLoader(MuJoCoOfflineData(
                config = self.config,
                dataset_path=test_data_path,
                plot=False, train=False,
                fix_view=True, num_workers=1
                ), batch_size=2, shuffle=True, drop_last=True)

            print(f'Length of val_data is {len(val_dataloader)}')
            self.val_data_nn = self.get_data(val_dataloader,  summ_writer, step)

        print("finish loading data")

        val_results_list, cluster_id_list, success_rate_list, feed = self.get_features_from_data_list(self.val_data_nn, step)
        #val_results_list, cluster_id_list, feed = self.get_features(val_dataloader, summ_writer, step)

        # now that you have results think about how can you collate and do nearest neighbor
        vis_nearest_neighbours, recall_1 = self.compute_nearest_neighbours_dp(val_results_list, cluster_id_list)

        
        # now the only thing that remains is plotting this on tensorboard
        # just to satisfy my paranoia I will also save the matplotlib images
        # but after 500 iterations
        if feed['global_step'] % 5000 == 0:
            n_rows = len(vis_nearest_neighbours)
            n_cols = len(vis_nearest_neighbours[0])
            fig_size = 2 * np.asarray([n_rows, n_cols])
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size,
                sharex=True, sharey=True)
            for i in range(n_rows):
                for j in range(n_cols):
                    axes[i][j].imshow(vis_nearest_neighbours[i][j])
            
            # save the figure and you are done
            fig.savefig(f'{self.nn_results_dir}/nn_result_{step}.jpg')
            plt.close()

        H, W, C = list(vis_nearest_neighbours[0][1].shape)

        # finally add it to tensorboard and you are done !!!
        # #test_data x (#top_retrieve + 1)
        vis_nearest_neighbours = np.stack(vis_nearest_neighbours, axis=0)
        # convert to torch
        vis_nearest = torch.from_numpy(vis_nearest_neighbours).permute(0, 1, 4, 2, 3)
        # resize
        vis_nearest = vis_nearest.view(-1, C, H, W)
        # make the grid
        grid = make_grid(vis_nearest, nrow=11)
        
        # add it to the tensorboard
        feed['writer'].add_scalar('val_nn_recall@1', recall_1, step)
        feed['writer'].add_image('val_nn/imgs', grid, step)

    def validate_on_train_using_computed_center(self, step, summ_writer, normalize=True):
        return self.validate_on_test_using_computed_center(step, summ_writer, normalize=True, data_name="train", ndata=40)

    def validate_on_test_using_computed_center(self, step, summ_writer, normalize=True, data_name="test", ndata=None):
        print(f"start validate on {data_name} using computeed center")

        if self.model.is_learned_cluster_centers:

            tensor_in_clusters = self.model.get_centers()
            cluster_names = []
            for cluster_id in range(self.model.num_clusters):
                cluster_names.append(self.model.cluster_id_to_name[cluster_id])
            for i in range(self.model.num_clusters, self.model.max_clusters):
                cluster_names.append(f"not_assigned_{i}")

        else:

            tensor_in_cluster = self.compute_cluster_center_from_train(epochs=3, ndata=200, using_success_rate=True)
            tensor_in_clusters = []
            cluster_names = []
            for cluster_name in tensor_in_cluster:
                cluster_names.append(cluster_name)
                tensor_in_clusters.append(tensor_in_cluster[cluster_name])
            tensor_in_clusters = np.stack(tensor_in_clusters, axis=0)

        # remove "not_assigned" from the list
        for cluster_name in cluster_names:
            if "not_assigned" in cluster_name:
                cid = cluster_names.index(cluster_name)
                cluster_names.remove(cluster_name)
                tensor_in_clusters = np.concatenate([tensor_in_clusters[:cid], tensor_in_clusters[cid+1:]], axis=0)




        assert not self.model.training, "Model should be in eval mode"
        if data_name=="test" and self.val_data_nn is None:
            test_data_path = self.config.data_paths[data_name]
            # now form the data-loader with the valset path

            val_dataloader = torch.utils.data.DataLoader(MuJoCoOfflineData(
                config = self.config,
                dataset_path=test_data_path,
                plot=False, train=False,
                fix_view=True, num_workers=1,
                ndata=ndata,
                ), batch_size=2, shuffle=False, drop_last=True)

            print(f'Length of val_data is {len(val_dataloader)}')
            self.val_data_nn = self.get_data(val_dataloader,  summ_writer, step)
            data_nn = self.val_data_nn

        elif data_name=="train" and self.trainval_data_nn is None:
            test_data_path = self.config.data_paths[data_name]
            # now form the data-loader with the valset path

            val_dataloader = torch.utils.data.DataLoader(MuJoCoOfflineData(
                config = self.config,
                dataset_path=test_data_path,
                plot=False, train=False,
                fix_view=True, num_workers=1,
                ndata=ndata,
                ), batch_size=2, shuffle=True, drop_last=True)

            print(f'Length of val_data is {len(val_dataloader)}')
            self.trainval_data_nn = self.get_data(val_dataloader,  summ_writer, step)
            data_nn = self.trainval_data_nn
        elif data_name == "test":
            data_nn = self.val_data_nn
        elif data_name == "train":
            data_nn = self.trainval_data_nn


        #tensor_in_clusters = []
        #cluster_names = []
        #for cluster_name in tensor_in_cluster:
        #    cluster_names.append(cluster_name)
        #    tensor_in_clusters.append(tensor_in_cluster[cluster_name])
        #tensor_in_clusters = np.stack(tensor_in_clusters, axis=0)
        #nc, H, W, D, C = tensor_in_clusters.shape

        if  len(tensor_in_clusters.shape) == 5:
            nc, H, W, D, C = tensor_in_clusters.shape
            emb_g_flat = torch.from_numpy(np.reshape(tensor_in_clusters, [nc, -1]))
        else:
            emb_g_flat = torch.from_numpy(tensor_in_clusters)

        #if normalize:
        emb_g_flat_nonorm = emb_g_flat
        #emb_g_flat = torch.nn.functional.normalize(emb_g_flat, dim=-1)

        correct = 0
        correct_nonorm = 0
        nsamples = 0
        success_rate_for_selected_cluster = []
        for feed_cuda in data_nn:
            with torch.no_grad():
                loss, results =  self.model(feed_cuda)
            # for now since I am only debugging stuff
            # a list of dictionary
            # "object_tensor": B x C x H x W x D
            # "record_name": list of strings

            if len(results["object_tensor"].shape) == 5: #3d tensor
                object_tensors = results["object_tensor"].permute(0, 2, 3, 4, 1)
                batch_size = object_tensors.shape[0]
                emb_e_flat = torch.reshape(object_tensors, [batch_size, -1])
            else:
                batch_size = results["object_tensor"].shape[0]
                emb_e_flat = results["object_tensor"]
            emb_e_flat_nonorm = emb_e_flat
            #emb_e_flat = torch.nn.functional.normalize(emb_e_flat, dim=-1)

            scores = torch.matmul(emb_e_flat, emb_g_flat.T)

            # top grasp only

            if self.config.top_grasp_only:
                mask = torch.zeros_like(scores)
                mask[:,:12] = 1
                mask[:,23:] = 1
                min_scores = torch.min(scores)
                scores = scores * mask + (1-mask) * min_scores

            scores_nonorm = torch.matmul(emb_e_flat_nonorm, emb_g_flat_nonorm.T)
            for batch_id in range(batch_size):
                nsamples += 1
                best_match_id = np.argmax(scores[batch_id].numpy())
                best_match_class = cluster_names[best_match_id]

                #if best_match_class is not "not_assigned":
                selected_cluster_id = int(best_match_class[1:])
                print(nsamples, "selected_cluster_id", selected_cluster_id, feed_cuda["success_rates"][batch_id][selected_cluster_id])
                print(nsamples, feed_cuda["success_rates"][batch_id])
                


                sr = feed_cuda["success_rates"][batch_id][selected_cluster_id]
                success_rate_for_selected_cluster.append(sr.cpu().numpy())

                if best_match_class == feed_cuda["cluster_id"][0][batch_id]:
                    correct += 1

                best_match_id_nonorm = np.argmax(scores_nonorm[batch_id].numpy())
                best_match_class_nonorm = cluster_names[best_match_id_nonorm]

                if best_match_class_nonorm == feed_cuda["cluster_id"][0][batch_id]:
                    correct_nonorm += 1

        print(" avg success_rate:", np.mean(success_rate_for_selected_cluster))

        #import ipdb; ipdb.set_trace()
        cluster_mean_acc = correct/nsamples
        cluster_mean_acc_nonorm = correct_nonorm/nsamples

        if data_name == "train":
            summ_writer.add_scalar('train_nn_cluster_mean', cluster_mean_acc, step)
            summ_writer.add_scalar('train_nn_cluster_mean_nonorm', cluster_mean_acc_nonorm, step)
            summ_writer.add_scalar('train_mean_success_rate', np.mean(success_rate_for_selected_cluster), step)
        else:
            summ_writer.add_scalar('val_nn_cluster_mean', cluster_mean_acc, step)
            summ_writer.add_scalar('val_nn_cluster_mean_nonorm', cluster_mean_acc_nonorm, step)
            summ_writer.add_scalar('val_mean_success_rate', np.mean(success_rate_for_selected_cluster), step)
        return  cluster_mean_acc

    def compute_cluster_center_from_train(self, epochs=3, ndata=None, using_success_rate=False):
        # first put the model in eval mode
        self.model.eval()
        # everytime make the val list empty
        train_results_list = list()
        assert not self.model.training, "Model should be in eval mode"

        #test_data_path = self.config.data_paths['test']
        train_data_path = self.config.data_paths['train']
        # now form the data-loader with the valset path

        train_dataloader = torch.utils.data.DataLoader(MuJoCoOfflineData(
            config = self.config,
            dataset_path=train_data_path,
            plot=False, train=False,
            fix_view=False, ndata=ndata
            ), batch_size=self.config.B, shuffle=False, drop_last=True)


        print(f'Length of train_data is {len(train_dataloader)}')

        tensor_in_cluster = dict()

        # do the forward pass now
        for epoch_id in range(epochs):
            set_loader = iter(train_dataloader) 
            for i, feed in enumerate(set_loader):
                # move everything to cuda
                feed_cuda = feed
                for k in feed:
                    try:
                        feed_cuda[k] = feed[k].cuda()
                    except:
                        # some of them are not tensor
                        feed_cuda[k] = feed[k]
    
                #feed_cuda['writer'] = summ_writer 
                #feed_cuda['global_step'] = step
                feed_cuda['set_num'] = 1
                #feed_cuda['set_name'] = 'val'
                #feed_cuda['record'] = feed['record']
    
                # now I am ready for the forward pass, I want it to return to me
                # the 3d tensor which belongs to the object
                with torch.no_grad():
                    results =  self.model.convert_objects_to_features(feed_cuda)
    
    
                if not using_success_rate:
                    for object_id in range(len(feed['cluster_id'])):
                        object_in_batch = feed["cluster_id"][object_id]
    
                        for batch_id in range(len(object_in_batch)):
                            object_cluster_id = object_in_batch[batch_id]
                            if object_cluster_id not in tensor_in_cluster:
                                tensor_in_cluster[object_cluster_id] = []
                            tensor_in_cluster[object_cluster_id].append(results[batch_id, object_id])
                else:

                    object_id = 0
                    for batch_id in range(len(feed['success_rates'])):
                        success_rate = feed["success_rates"][batch_id]

                        nclusters = len(success_rate)
                        for cluster_id in range(nclusters):
                            if success_rate[cluster_id] >= 0.8:
                                cluster_name = f"c{cluster_id}"
                                if cluster_id not in tensor_in_cluster:
                                    tensor_in_cluster[cluster_name] = []
                                tensor_in_cluster[cluster_name].append(results[batch_id, object_id])
                        #max_sr = np.maximum(success_rate)




        for cluster_id in tensor_in_cluster:
            tensor_in_cluster[cluster_id] = np.mean(np.stack(tensor_in_cluster[cluster_id], axis=0), axis=0)
        import pickle

        with open(os.path.join(self.config.vis_dir, "clusters.pkl"), 'wb') as f:
            pickle.dump(tensor_in_cluster, f)
        return tensor_in_cluster


    @staticmethod
    def get_obj_name_and_class(full_path):
        splits = full_path[0].split('/')
        cls_name = splits[-2]
        extra_info = splits[-1]
        split_1 = extra_info.split('_')
        ob_name = split_1[-1][:-4]
        return cls_name, ob_name

    def go(self):
        self.start_time = time.time()
        self.infer()  # defines the model
        # build the saveloader
        self.saverloader = saverloader.SaverLoader(self.config, self.model)
        self.saverloader.save_config()

        print("------ Start loading weights -----")
        self.start_iter = self.saverloader.load_weights(optimizer=None)  ## load the weights for each part
        print(f'---- self.start_iter = {self.start_iter}')
        print("------ Done loading weights ------")


        if self.config.do_compute_cluster_center: # only during testing
            self.compute_cluster_center_from_train()
            return


        # ... Declare the optimzer ... #
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print('----- Done making optimizer ---------')

        # for nearest_neightbor retrieval
        self.train_data_nn = None
        self.val_data_nn = None
        self.trainval_data_nn = None


        set_nums = []
        set_names = []
        set_inputs = []
        set_writers = []
        set_log_freqs = []
        set_do_backprops = []
        set_dicts = []
        set_loaders = []

        for set_name in self.config.set_names:
            # sets to run are determined by if the field corresponding to
            # trainset, valset, testset is None or not
            if self.config.sets_to_run[set_name]:
                set_nums.append(self.config.set_nums[set_name])
                set_names.append(set_name)
                set_inputs.append(self.all_inputs[set_name])  # dict formed in the input function
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue = MAX_QUEUE))
                set_log_freqs.append(self.config.log_freqs[set_name])
                set_do_backprops.append(self.config.sets_to_backprop[set_name])
                set_dicts.append({})
                set_loaders.append(iter(set_inputs[-1])) #use the latest set_inputs

        for step in range(self.start_iter+1, self.config.max_iters+1):
            for i, (set_input) in enumerate(set_inputs):
                if step % len(set_input) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
                    # this means the epoch is done, for val I want to break here
                    # if the epoch is done and set is validation or test then break it
                    # break
                    # train: 0, val: 1, test:2
                    set_num = set_nums[i]
                    if set_num == 1 and self.config.do_generate_data:
                        # we only want to go through once for the valset and if data-generation mode is true
                        break
                    else:
                        # while collecting data as well, when the test set is exhausted it will just refresh
                        # the iterator, the design could have so much better if just epochs would have been used.
                        set_loaders[i] = iter(set_input)  # refresh the iterators
            
            for (set_num,
                    set_name,
                    set_input,
                    set_writer,
                    set_log_freq,
                    set_do_backprop,
                    set_dict,
                    set_loader
                    ) in zip(
                    set_nums,
                    set_names,
                    set_inputs,
                    set_writers,
                    set_log_freqs,
                    set_do_backprops,
                    set_dicts,
                    set_loaders
                    ):

                # this loop will run atmost 3 times, once with train, then val and then test
                # log_this for val and test is 50 and set_do_backprop is 0 hence it does not
                # evaluates to true only if the condition below is true
                log_this = np.mod(step, set_log_freq) == 0  ## so this will be true everytime since using fastest logging
                total_time, read_time, iter_time = 0.0, 0.0, 0.0

                if log_this or set_do_backprop: # training or logging
                    #print('%s: set_num %d; log_this %d; set_do_backprop %d; ' % (set_name, set_num, log_this, set_do_backprop))

                    read_start_time = time.time()
                    feed = next(set_loader)
                    feed_cuda = feed

                    for k in feed:
                        try:
                            feed_cuda[k] = feed[k].cuda()
                        except:
                            feed_cuda[k] = feed[k]

                    read_time = time.time() - read_start_time
                    feed_cuda['writer'] = set_writer
                    feed_cuda['global_step'] = step
                    feed_cuda['set_num'] = set_num
                    feed_cuda['set_name'] = set_name
                    feed_cuda['record'] = feed['record']
                    #print(f'working on {feed["record"]}')

                    iter_start_time = time.time()
                    if set_do_backprop:
                        self.model.train()
                        loss, results = self.model(feed_cuda)
                    else:
                        self.model.eval()
                        assert self.model.training == False, "in eval code"
                        print('----- using the eval branch of the code ------')
                        with torch.no_grad():
                            loss, results = self.model(feed_cuda)

                    loss_vis = loss.cpu().item()

                    if set_do_backprop:
                        # if not hyp.do_metric_learning:
                        featnet_before = utils.basic.get_params(self.model.featnet)
                        if self.config.do_occ:
                            occnet_before = utils.basic.get_params(self.model.occnet)
                        if self.config.do_view:
                            viewnet_before = utils.basic.get_params(self.model.viewnet)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()




                        featnet_after = utils.basic.get_params(self.model.featnet)
                        if self.config.do_occ:
                            occnet_after = utils.basic.get_params(self.model.occnet)
                        if self.config.do_view:
                            viewnet_after = utils.basic.get_params(self.model.viewnet)

                        # check if they are changed
                        #                        assert utils.basic.check_notequal(featnet_before, featnet_after)
                        #                        if self.config.do_occ:
                        #                            import ipdb; ipdb.set_trace()
                        #                            assert utils.basic.check_notequal(occnet_before, occnet_after)
                        #                        if self.config.do_view:
                        #                            assert utils.basic.check_notequal(viewnet_before, viewnet_after)

                    if log_this:  # this is a new hyper-parameter which does nearest neighbour evaluation
                        # I will write the validation function here
                        if self.config.do_validation and set_name=="test" and step % self.config.validate_after == 0:
                            #self.validate_on_test(feed['global_step'], feed_cuda['writer'])
                            #self.validate_nn_on_test_from_train(feed['global_step'], val_summ_writer=feed_cuda['writer'], train_summ_writer=set_writers[0])
                            self.validate_on_test_using_computed_center(feed['global_step'], summ_writer=feed_cuda['writer'])
                            self.validate_on_train_using_computed_center(feed['global_step'], summ_writer=feed_cuda['writer'])

                        iter_time = time.time()-iter_start_time
                        total_time = time.time()-self.start_time

                        print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (self.config.run_full_name,
                                                                                            step,
                                                                                            self.config.max_iters,
                                                                                            total_time,
                                                                                            read_time,
                                                                                            iter_time,
                                                                                            loss_vis,
                                                                                            set_name))

            if np.mod(step, self.config.snap_freq) == 0:
                self.saverloader.save(step, self.optimizer)


        for writer in set_writers: #close writers to flush cache into file
            writer.close()

